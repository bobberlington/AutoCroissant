from difflib import SequenceMatcher, get_close_matches
from discord import Interaction
from discord.errors import InteractionResponded
import pandas
from pickle import load, dump
from requests import get
from warnings import filterwarnings
filterwarnings('ignore')

from global_config import ALIAS_PKL, STATS_PKL

REPOSITORY  = "MichaelJSr/TTSCardMaker"
MATCH_RATIO = 0.6

git_file_alias: dict[str, str]              = {}
git_files: dict[str, str]                   = {}
ambiguous_names: dict[str, tuple[str, ...]] = {}
git_filenames: list[str]                    = []
# Cards df, but filtered to not have the cards that have no ability, also it is all lowercase.
cards_df_filtered   = pandas.DataFrame()

headers = None
git_token = None
try:
    import config
    git_token = config.git_token
    headers = {'Authorization': 'token ' + git_token}
    print("Git token found, api will be limited to 5000 requests/hour.")
except AttributeError:
    print("No git token in config, api will be limited to 60 requests/hour.")

def populate_files():
    global git_filenames
    git_files.clear()
    ambiguous_names.clear()
    # Grab repo
    repo = get(f"https://api.github.com/repos/{REPOSITORY}/git/trees/main?recursive=1", headers=headers)
    if repo.status_code != 200:
        print(f"Error when trying to connect to {REPOSITORY}")
        return repo.status_code

    # Make a dictionary of all list of pngs in the github
    # Keys consist of the filename, for example Bomb.png
    # Values consist of the whole path, for example Items/Attack/2 Stars/Bomb.png
    for i in repo.json()["tree"]:
        path: str = i["path"]
        if path.endswith(".png"):
            png_filename = path[path.rindex("/") + 1:].lower()
            # If we're putting the same name twice, it will change both names to include the top level folder
            if png_filename in git_files:
                new_filename = f"{path[0:path.index('/')]}/{png_filename}".lower()
                old_filename = f"{git_files[png_filename][0:git_files[png_filename].index('/')]}/{png_filename}".lower()
                git_files[new_filename] = path.replace(" ", "%20")
                git_files[old_filename] = git_files[png_filename]

                # Mark which files are ambiguous
                if png_filename in ambiguous_names:
                    if old_filename not in ambiguous_names[png_filename]:
                        ambiguous_names[png_filename] = (*ambiguous_names[png_filename], old_filename)
                    ambiguous_names[png_filename] = (*ambiguous_names[png_filename], new_filename)
                else:
                    ambiguous_names[png_filename] = (old_filename, new_filename)
            else:
                git_files[png_filename] = path.replace(" ", "%20")

    # Fill up the aliases
    # keys of aliases are the aliases, values are the filenames they point to
    for i in git_file_alias:
        if git_file_alias[i] in git_files:
            git_files[f"{i}.png"] = git_files[git_file_alias[i]]
        else:
            val = [val for val in git_files.values() if val.lower().endswith(git_file_alias[i])]
            if val:
                git_files[f"{i}.png"] = val[0]
                # Adds any files that end with val to ambiguous_names
                #ambiguous_names[f"{i}.png"] = [v[0:v.index('/')] + v[v.rindex('/'):] for v in val]

    git_filenames = git_files.keys()
    return 200

def populate_descriptions(desc: str):
    import re
    desc = desc.strip()
    if "|" in desc:
        desc1, desc2 = desc.split("|", 1)
        return pandas.concat((populate_descriptions(desc1.strip()), populate_descriptions(desc2.strip()))).drop_duplicates(subset=['name', 'ability'], keep='first')
    if "&" in desc:
        desc1, desc2 = desc.split("&", 1)
        return pandas.merge(populate_descriptions(desc1.strip()), populate_descriptions(desc2.strip()), left_index=True, right_index=True, how="inner", copy=False)

    opposite = False
    if desc.startswith("!"):
        desc = desc[1:].strip()
        opposite = True

    # Exact Matches
    if desc.startswith("\"") and desc.endswith("\""):
        desc = desc[1:-1]
        if opposite:
            return cards_df_filtered[~cards_df_filtered["ability"].str.contains((r"(?:^|\s|$|\b)" + re.escape(desc) + r"(?:^|\s|$|\b)"))]
        return cards_df_filtered[cards_df_filtered["ability"].str.contains((r"(?:^|\s|$|\b)" + re.escape(desc) + r"(?:^|\s|$|\b)"))]
    else:
        # Find abilities that don't contain the string if opposite.
        if opposite:
            return cards_df_filtered[~cards_df_filtered["ability"].str.contains(re.escape(desc))]
        # Otherwise find abilities that have the substring or match well enough.
        cards_df_contains = cards_df_filtered[cards_df_filtered["ability"].str.contains(re.escape(desc))]
        cards_df_scores = cards_df_filtered["ability"].apply(lambda x: SequenceMatcher(None, desc, x.lower()).ratio())
        cards_df_scores = cards_df_scores[cards_df_scores > MATCH_RATIO]
        return pandas.concat([cards_df_filtered.loc[cards_df_scores.index], cards_df_contains]).drop_duplicates(subset=['name', 'ability'], keep='first')

def try_open_alias():
    global git_file_alias

    print(f"Trying to open {ALIAS_PKL}")
    try:
        with open(ALIAS_PKL, 'rb') as f:
            git_file_alias = load(f)
    except (EOFError, FileNotFoundError):
        print(f"{ALIAS_PKL} doesn't exist, populating with empty dict...")
        with open(ALIAS_PKL, 'wb') as f:
            dump(git_file_alias, f)

def try_open_descriptions():
    global cards_df_filtered

    print(f"Trying to open {STATS_PKL}")
    try:
        with open(STATS_PKL, 'rb') as f:
            cards_df = pandas.DataFrame.from_dict(load(f)).transpose()
            try:
                cards_df_filtered = cards_df[cards_df["ability"].notna()].copy()
                cards_df_filtered["ability"] = cards_df_filtered["ability"].str.lower()
            except KeyError:
                print(f"{STATS_PKL} exists but has no relevant data in it.")
    except (EOFError, FileNotFoundError):
        print(f"{STATS_PKL} is completely empty.")

async def query_remote(interaction: Interaction, query: str):
    card = query.replace(" ", "_").lower()
    if not card.endswith(".png"):
        card += ".png"
    try:
        closest = get_close_matches(card, git_filenames, n=1, cutoff=MATCH_RATIO)[0]
    except IndexError:
        await interaction.response.send_message("No card found!")
        return
    await interaction.response.send_message(f"https://raw.githubusercontent.com/{REPOSITORY}/main/{git_files[closest]}")

    # If the filename was ambiguous, make a note of that.
    if closest in ambiguous_names:
        ambiguous_message = "Ambiguous name found. If this wasn't the card you wanted, try typing: \n"
        for i in ambiguous_names[closest]:
            ambiguous_message += f"{i}\n"
        await interaction.followup.send(ambiguous_message)

async def query_pickle(interaction: Interaction, desc: str):
    if cards_df_filtered.empty:
        return await interaction.response.send_message(f"{STATS_PKL} is empty. run ```/update_stats``` first.")

    desc = desc.strip().lower()
    closest = populate_descriptions(desc)

    for index, close in closest.iterrows():
        try:
            await interaction.response.send_message(f"https://raw.githubusercontent.com/{REPOSITORY}/main/{index[0:-4].replace(' ', '%20')}.png")
        except InteractionResponded:
            await interaction.followup.send(f"https://raw.githubusercontent.com/{REPOSITORY}/main/{index[0:-4].replace(' ', '%20')}.png")
    try:
        await interaction.response.send_message(f"{len(closest)} Results found for {desc}!")
    except InteractionResponded:
        await interaction.followup.send(f"{len(closest)} Results found for {desc}!")

async def howmany_description(interaction: Interaction, desc: str):
    if cards_df_filtered.empty:
        return await interaction.response.send_message(f"{STATS_PKL} is empty. run ```/update_stats``` first.")

    desc = desc.strip().lower()
    closest = populate_descriptions(desc)

    await interaction.response.send_message(f"{len(closest)} Results found for {desc}!")

def print_all_aliases():
    all_aliases = "```"
    for key, val in git_file_alias.items():
        all_aliases += f"{key:20s} -> {val}\n"
    all_aliases += "```"
    return all_aliases

async def alias_card(interaction: Interaction, key: str, val: str):
    global git_filenames
    if not key or not val:
        await interaction.response.send_message(print_all_aliases())
        return

    key = key.lower()
    val = val.lower()

    if not val.endswith(".png"):
        val += ".png"

    invalid_val = True
    if val in git_files:
        git_files[f"{key}.png"] = git_files[val]
        invalid_val = False
    else:
        for val_in_dict in git_files.values():
            if val_in_dict.lower().endswith(val):
                git_files[f"{key}.png"] = val_in_dict
                invalid_val = False
                break
    if invalid_val:
        await interaction.response.send_message(f"No such value exists: {val}\nCouldnt add alias into dictionary.")
        return

    git_filenames = git_files.keys()
    git_file_alias[key] = val
    await interaction.response.send_message(f"Created alias: {key} -> {val}")

    with open(ALIAS_PKL, 'wb') as f:
        dump(git_file_alias, f)

async def delete_alias(interaction: Interaction, key: str):
    global git_filenames
    if key in git_file_alias:
        await interaction.response.send_message(f"Deleted alias: {key} -> {git_file_alias.pop(key)}")

        if not key + ".png" in git_files:
            await interaction.followup.send(f"No such key exists: {key}\nCouldnt pop alias from dictionary.")
            return

        git_files.pop(f"{key}.png")
        git_filenames = git_files.keys()
    else:
        await interaction.response.send_message(f"No value exists for the alias: {key}")
        return

async def set_match_ratio(interaction: Interaction, value: float):
    global MATCH_RATIO
    if not value:
        await interaction.response.send_message(f"The match ratio is {MATCH_RATIO}.")
        return

    MATCH_RATIO = value
    await interaction.response.send_message(f"New match ratio of {MATCH_RATIO} set!")

async def set_repository(interaction: Interaction, new_repo: str):
    global REPOSITORY
    if not new_repo:
        await interaction.response.send_message(f"The repository is {REPOSITORY}.")
        return

    REPOSITORY = new_repo
    status = populate_files()
    if status != 200:
       print(f"Error {status} when requesting github.")
    await interaction.response.send_message(f"New repository of {REPOSITORY} set!")
