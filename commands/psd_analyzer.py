from datetime import datetime
from discord import Interaction, File
from collections import defaultdict
from github import Github, Repository
from logging import getLogger, CRITICAL
from os import path as path_os, walk
from os.path import getmtime, basename
import pandas
from pickle import load, dump
from psd_tools import PSDImage
from re import finditer, sub
from requests import get, Response
from urllib.request import urlretrieve

from global_config import LOCAL_DIR_LOC, STATS_PKL, OLD_STATS_PKL, METAD_PKL
from commands.query_card import try_open_stats, query_psd_path
from commands.utils import edit_messages, message_queue, command_queue

getLogger("psd_tools").setLevel(CRITICAL)
UPDATE_RATE         = 25
LOCAL_REPO: str     = path_os.expanduser(LOCAL_DIR_LOC)
EXCLUDE_FOLDERS     = ["markers", "MDW"]
EXPORTED_STATS_NAME = "stats"
# ACCURSED_COMMIT     = "77b97e4760d385a82cc404b62212644f81167ac4"
EXPORTED_RULES_NAME = "rules"

stats       = {}
old_stats   = defaultdict(list)
metadata    = {}
all_types   = []
dirty_files = []

headers: dict   = None
git_token: str  = None
try:
    import config
    git_token = config.git_token
    headers = {'Authorization': 'token ' + git_token}
    print("Git token found, api will be limited to 5000 requests/hour.")
except AttributeError:
    print("No git token in config, api will be limited to 60 requests/hour.")

def pickle_stats():
    with open(STATS_PKL, 'wb') as f:
        dump(stats, f)
    with open(OLD_STATS_PKL, 'wb') as f:
        dump(old_stats, f)
    with open(METAD_PKL, 'wb') as f:
        dump(metadata, f)

def load_stats():
    global stats, old_stats, metadata

    print(f"Trying to open {METAD_PKL}")
    try:
        with open(METAD_PKL, 'rb') as f:
            metadata = load(f)
        print(f"Existing dict found in {METAD_PKL}, updating entries...")
    except (EOFError, FileNotFoundError):
        print(f"{METAD_PKL} is completely empty or doesn't exist, rebuilding entire dict...")

    print(f"Trying to open {STATS_PKL}")
    try:
        with open(STATS_PKL, 'rb') as f:
            stats = load(f)
            for key in metadata:
                if key in stats:
                    stats[key].update(metadata[key])
        print(f"Existing dict found in {STATS_PKL}, updating entries...")
    except (EOFError, FileNotFoundError):
        print(f"{STATS_PKL} is completely empty or doesn't exist, rebuilding entire dict...")

    print(f"Trying to open {OLD_STATS_PKL}")
    try:
        with open(OLD_STATS_PKL, 'rb') as f:
            old_stats = load(f)
        print(f"Existing dict found in {OLD_STATS_PKL}, updating entries...")
    except (EOFError, FileNotFoundError):
        print(f"{OLD_STATS_PKL} is completely empty or doesn't exist, rebuilding entire dict...")

def parse_clean_cards():
    prune_stats = []
    for name in stats:
        if name not in dirty_files:
            old_stats[name].append(stats[name])
            prune_stats.append(name)
    for name in prune_stats:
        stats.pop(name)
    # Should we do this for metadata too?

def update_stats(interaction: Interaction, output_problematic_cards: bool = True, use_local_repo: bool = True, use_local_timestamp: bool = True) -> tuple[str, ...]:
    load_stats()
    problem_cards = None
    if use_local_repo:
        problem_cards = traverse_local_repo(interaction, output_problematic_cards, use_local_timestamp)
    else:
        problem_cards = traverse_repo(interaction, output_problematic_cards)
    parse_clean_cards() # Now check which cards we actually edited in dirty_files, and move clean ones to old_stats since they don't exist anymore
    pickle_stats() # Now that we updated the descriptions, store them back
    return problem_cards

def populate_types_stars(resp: Response | None = None, use_local_timestamp: bool = True):
    if use_local_timestamp and resp is None:
        for folder, _, files in walk(LOCAL_REPO + "/Types"):
            for file in files:
                if folder.endswith("Types"):
                    all_types.append(file[:-4].lower())
    elif resp:
        for i in resp.json()["tree"]:
            path: str = i["path"]
            if path.startswith("Types") and "Stars" not in path and '.' in path:
                all_types.append(basename(path)[:-4].lower())

def classify_card(relative_loc: str):
    if not relative_loc:
        return

    folders = relative_loc.split('/')[:-1]
    if len(folders) > 0 and folders[0] == "MDW":
        return {
            "type" : "MDW",
            "ability" : None
        }
    elif len(folders) > 0 and folders[0] == "Field":
        return {
            "type" : "field",
            "stars" : int(folders[-1].split()[0])
        }
    elif len(folders) > 0 and folders[0] == "Items":
        return {
            "type" : "item",
            "subtype" : folders[-2].lower(),
            "stars" : int(folders[-1].split()[0])
        }
    elif len(folders) > 0 and folders[0] == "Creatures":
        return {
            "type": "creature",
            "stars": int(folders[-1].split()[0]),
            "series": folders[-2].lower(),
            "hp" : -1,
            "def" : -1,
            "atk" : -1,
            "spd" : -1
        }
    elif len(folders) > 0 and folders[0] == "Auxiliary":
        if len(folders) > 1 and folders[1] == "Minions":
            return {
                "type": "minion",
                "hp" : -1,
                "def" : -1,
                "atk" : -1,
                "spd" : -1
            }
        elif len(folders) > 1 and folders[1] == "Items":
            return {
                "type": "aux item"
            }
        elif len(folders) > 2 and folders[2] == "Debuffs":
            return {
                "type": "debuff",
                "stars": int(folders[-1].split()[0])
            }
        elif len(folders) > 2 and folders[2] == "Buffs":
            return {
                "type": "buff"
            }
        else:
            return {
                "type": folders[-1].lower()
            }
    elif len(folders) > 0 and folders[0] == "N.M.E":
        return {
            "type": "nme"
        }
    else:
        return {
            "type" : "unknown"
        }

def sort_bbox(bboxes: tuple[tuple[str, tuple[int, int]]], epsilon: int = 10) -> tuple[tuple[str, tuple[int, int]]]:
    """
    Sorts a list of (label, (x, y)) pairs first by y-coordinate, then by x-coordinate.
    Rows are grouped if their y-coordinates are within epsilon.

    :param bboxes: List of (label, (x, y)) tuples.
    :param epsilon: Tolerance for grouping items in the same row.
    :return: Sorted list of bboxes.
    """
    # Sort primarily by y with epsilon tolerance, then by x
    return sorted(bboxes, key=lambda item: (item[1][1] // epsilon, item[1][0]))

def prune_bbox(bboxes: tuple[tuple[str, tuple[int, int]]]) -> tuple[tuple[str, tuple[int, int]]]:
    if len(bboxes) == 0:
        return bboxes
    max_bbox_height = max(bboxes[-1][1][1] // 3, 400)
    return [bbox for bbox in bboxes if bbox[1][1] >= max_bbox_height]

# Well, we can extract all images from a psd file, if we want to for some reason
def extract_all_images_from_psd(file_loc: str):
    count = 0
    for layer in PSDImage.open(file_loc).descendants():
        if layer.has_pixels():
            layer_image = layer.topil()
            layer_image.save(f"{layer.name}{str(count)}.png")
            count += 1

def extract_info_from_psd(file_loc: str, relative_loc: str = ""):
    card = classify_card(relative_loc)
    longest_text = ""
    num_stars = 0
    hp = df = atk = spd = None
    get_stars_from_psd = is_rulepage = hp_found = df_found = atk_found = spd_found = False
    type_bboxes: list[tuple[str, tuple[int, int]]] = []
    types: list[str] = []
    abilities: list[str] = []

    if "Rulebook" in relative_loc:
        is_rulepage = True
    elif "Auxiliary/Items" in relative_loc or "Auxiliary/Minions" in relative_loc or "N.M.E" in relative_loc:
        get_stars_from_psd = True
    for layer in PSDImage.open(file_loc).descendants():
        if layer.kind == "type":
            layer_text = str(layer.engine_dict["Editor"]["Text"]).replace('\\r', '\n').replace('\\n', '\n').replace('\\t', ' ').replace('\\x03', '\n').replace('\\ufeff', '').rstrip()
            if layer.name.lower() == "ability" or is_rulepage:
                abilities.append((layer_text.strip('\'" '), layer.bbox[:2]))
            elif layer.bbox[1] > 400 and len(layer_text) > len(longest_text):
                longest_text = layer_text
        elif "dark" in layer.parent.name.lower() or (layer.parent.parent and "dark" in layer.parent.parent.name.lower()) or (layer.parent.parent and layer.parent.parent.parent and "dark" in layer.parent.parent.parent.name.lower()) \
            or "bars" in layer.parent.name.lower() or (layer.parent.parent and "bars" in layer.parent.parent.name.lower()) or (layer.parent.parent and layer.parent.parent.parent and "bars" in layer.parent.parent.parent.name.lower()):
            if ("hp" in layer.parent.name.lower() or "hp" in layer.parent.parent.name.lower()) and layer.name.isdigit():
                if layer.is_visible():
                    if hp is None:
                        hp = 0
                    hp += int(layer.name)
                hp_found = True
            elif ("def" in layer.parent.name.lower() or "def" in layer.parent.parent.name.lower()) and layer.name.isdigit():
                if layer.is_visible():
                    if df is None:
                        df = 0
                    df += int(layer.name)
                df_found = True
            elif ("atk" in layer.parent.name.lower() or "atk" in layer.parent.parent.name.lower()) and layer.name.isdigit():
                if layer.is_visible():
                    if atk is None:
                        atk = 0
                    atk += int(layer.name)
                atk_found = True
            elif ("spd" in layer.parent.name.lower() or "spd" in layer.parent.parent.name.lower()) and layer.name.isdigit():
                if layer.is_visible():
                    if spd is None:
                        spd = 0
                    spd += int(layer.name)
                spd_found = True
        elif layer.name.lower() in all_types and layer.is_visible() and layer.has_pixels():
            if layer.bbox[1] < 400:
                types.append(layer.name.lower())
            else:
                type_bboxes.append((layer.name.lower(), layer.bbox[:2]))
        elif get_stars_from_psd and ("stars" in layer.parent.name.lower() or (layer.parent.parent and "stars" in layer.parent.parent.name.lower()) or (layer.parent.parent and layer.parent.parent.parent and "stars" in layer.parent.parent.parent.name.lower())):
            if layer.is_visible() and layer.has_pixels():
                num_stars += 1

    abilities = sort_bbox(abilities)
    ability = '\n'.join([i[0] for i in abilities])
    # Failsafe for when creature does not have a text layer called "ability"
    if not ability:
        ability = longest_text
        if "problem" not in card:
            card["problem"] = []
        card["problem"].append("NO ABILITY LAYER")
    if ability:
        type_bboxes = prune_bbox(sort_bbox(type_bboxes))
        # Find all instances of multiple whitespace, or whitespace followed by colon
        matches = [match for match in finditer(r'\s{3,}|\s{3,}:', ability)]

        if len(matches) > 0 and len(type_bboxes) > 0:
            if len(type_bboxes) < len(matches):
                if "problem" not in card:
                    card["problem"] = []
                card["problem"].append("INCORRECT TYPE NAMES")
            else:
                count = len(type_bboxes) - 1
                for match in matches[::-1]:
                    ability = ability[:match.start()] + ' ' + type_bboxes[count][0] + ' ' + ability[match.end():]
                    count -= 1
        card["ability"] = sub(r'\s+([:;,\.\?!])', r'\1', ability).strip('\'" ').strip()

    # If we actually did find a stat value inside the card, but there was no visible "dark" layer,
    # assume the stat is equal to 10.
    if hp_found and hp is None:
        hp = 10
    if df_found and df is None:
        df = 10
    if atk_found and atk is None:
        atk = 10
    if spd_found and spd is None:
        spd = 10

    if hp is not None:
        card["hp"] = hp
    if df is not None:
        card["def"] = df
    if atk is not None:
        card["atk"] = atk
    if spd is not None:
        card["spd"] = spd

    if types:
        card["types"] = types

    if get_stars_from_psd:
        card["stars"] = num_stars

    return card

def problem_card_checker(card: dict[str, str]) -> tuple[str, ...]:
    problems = []
    if card["type"] == "unknown":
        problems.append("UNKOWN TYPE")
    if card["type"] != "unknown" and (not "ability" in card or card["ability"] is None):
        problems.append("ABILITY TEXT NOT FOUND")

    if (card["type"] == "creature" or card["type"] == "minion") and card["hp"] == -1:
        problems.append("HP NOT FOUND")
    if (card["type"] == "creature" or card["type"] == "minion") and card["def"] == -1:
        problems.append("DEF NOT FOUND")
    if (card["type"] == "creature" or card["type"] == "minion") and not ("types" in card and "active" in card["types"]) and card["atk"] == -1:
        problems.append("ATK NOT FOUND")
    if (card["type"] == "creature" or card["type"] == "minion") and not ("types" in card and "active" in card["types"]) and card["spd"] == -1:
        problems.append("SPD NOT FOUND")
    if (card["type"] == "creature" or card["type"] == "minion") and (card["hp"] > 10 or card["def"] > 10 or card["atk"] > 10 or card["spd"] > 10):
        problems.append("STATS TOO HIGH")

    if "problem" in card and card["problem"]:
        problems.extend(card["problem"])

    return problems

def log_problematic_cards(problematic_cards):
    problem_cards = []
    for loc, card, problems in problematic_cards:
        problem_cards.append(f"{loc}\n" + '```' + '\n'.join(problems) + '```')
    return problem_cards

def get_card_stats(interaction: Interaction, query: str):
    if not stats:
        load_stats()
    
    path = query_psd_path(query)
    name = basename(path)[:-4].replace('_', ' ')
    if name in stats:
        pretty_print_dict = '\n'.join("{!r}: {!r},".format(k, v) for k, v in stats[name].items())
        message_queue.append((interaction, f"Stats for {name}:```{pretty_print_dict}```"))
    else:
        message_queue.append((interaction, f"No stats found for {name}."))

def list_orphans(interaction: Interaction):
    if not stats:
        load_stats()
    
    orphans = []
    for name in stats:
        if "author" not in stats[name] or not stats[name]["author"]:
            orphans.append(name)

    bundle = ""
    for card in orphans:
        bundle += card + '\n'
        if len(bundle) > 1000:
            message_queue.append((interaction, bundle))
            bundle = ""
    if not bundle:
        bundle = "No orphans."
    message_queue.append((interaction, bundle))

def mass_replace_author(interaction: Interaction, author1: str = "", author2: str = ""):
    if not stats:
        load_stats()

    num_replaced = 0
    for name in metadata:
        if metadata[name]["author"] == author1:
            metadata[name]["author"] = author2
            if name in stats:
                stats[name]["author"] = author2
            num_replaced += 1

    pickle_stats()
    message_queue.append((interaction, f"{num_replaced} instances of {author1} replaced with {author2}."))

def manual_metadata_entry(interaction: Interaction, query: str, del_entry: bool = False, author: str = ""):
    if not stats:
        load_stats()

    path = query_psd_path(query)
    name = basename(path)[:-4].replace('_', ' ')
    if del_entry:
        if name in metadata:
            del metadata[name]
            pickle_stats()
        message_queue.append((interaction, f"{name} key was deleted from metadata."))
        return

    if name not in metadata:
        metadata[name] = {}
    if author:
        metadata[name]["author"] = author

    if name in stats:
        stats[name].update(metadata[name])
    pickle_stats()
    message_queue.append((interaction, f"{name} metadata updated to {metadata[name]}."))

def set_metadata(commits: list, name: str):
    # if commits[commits.totalCount - 1].commit.sha != ACCURSED_COMMIT:
    author = commits[commits.totalCount - 1].commit.committer.name
    if name not in metadata:
        metadata[name] = {}
    if "author" not in metadata[name] or not metadata[name]["author"]:
        metadata[name]["author"] = author

def set_remote_timestamp(repo: Repository.Repository, path: str):
    commits = repo.get_commits(path=path)
    # If this is somehow an empty list, return an invalid old timestamp
    if commits.totalCount == 0:
        print(f"{path} has no valid timestamp.")
        return datetime(1000, 1, 1).timestamp()

    set_metadata(commits, basename(path)[:-4].replace('_', ' '))
    return commits[0].commit.committer.date.timestamp()

def traverse_repo(interaction: Interaction = None, output_problematic_cards: bool = True) -> tuple[str, ...]:
    from commands.query_card import REPOSITORY
    resp = get(f"https://api.github.com/repos/{REPOSITORY}/git/trees/main?recursive=1", headers=headers)
    if resp.status_code != 200:
        print(f"Error when trying to connect to {REPOSITORY}")
        return resp.status_code
    # This uses an api request
    repo = Github(login_or_token=git_token).get_repo(REPOSITORY)

    populate_types_stars(resp, False)
    problematic_cards = []
    num_updated = 0
    num_new = 0
    num_old = 0
    for i in resp.json()["tree"]:
        path: str = i["path"]
        if "MDW" in path:
            continue
        if path.endswith('.psd'):
            date: datetime = set_remote_timestamp(repo, path)
            name = basename(path)[:-4].replace('_', ' ')
            dirty_files.append(name)

            if name in stats and stats[name]["timestamp"] >= date:
                num_old += 1
            else:
                num_new += 1
                if name in stats and stats[name]["timestamp"] < date:
                    old_stats[name].append(stats[name])

                path_no_spaces = path.replace(" ", "%20")
                # This uses an api request
                card = extract_info_from_psd(urlretrieve(f"https://raw.githubusercontent.com/{REPOSITORY}/main/{path_no_spaces}")[0], path)
                card["path"] = path
                card["timestamp"] = date
                stats[name] = card

                if output_problematic_cards and card["type"] not in EXCLUDE_FOLDERS:
                    problems = problem_card_checker(card)
                    if problems:
                        problematic_cards.append((path, card, problems))

            num_updated += 1
            if not num_updated % UPDATE_RATE:
                if interaction:
                    edit_messages.append((interaction, f"{num_updated} Cards updated.", []))
                else:
                    print(f"{num_updated} Cards updated.")

    if interaction:
        message_queue.append((interaction, f"{num_new} Had newer timestamps."))
        message_queue.append((interaction, f"{num_old} Did not have newer timestamps."))
    else:
        print(f"{num_new} Had newer timestamps.")
        print(f"{num_old} Did not have newer timestamps.")
    return log_problematic_cards(problematic_cards)

def traverse_local_repo(interaction: Interaction = None, output_problematic_cards: bool = True, use_local_timestamp: bool = True):
    from commands.query_card import REPOSITORY
    repo = None
    if not use_local_timestamp:
        print(f"Warning: Getting the timestamp of a remote file uses up an api request, you can make up to {5000 if git_token else 60} requests")
        repo = Github(login_or_token=git_token).get_repo(REPOSITORY)

    populate_types_stars(None, True)
    problematic_cards = []
    num_updated = 0
    num_new = 0
    num_old = 0
    for folder, _, files in walk(LOCAL_REPO):
        if "MDW" in folder:
            continue
        folder += '/'
        for file in files:
            if file.endswith('.psd'):
                full_file = folder.replace('\\', '/') + file
                truncated_file = full_file.split("TTSCardMaker")[-1].strip('/')
                name = basename(truncated_file)[:-4].replace('_', ' ')
                dirty_files.append(name)

                date = -1.0
                if use_local_timestamp:
                    date = getmtime(full_file)
                else:
                    date: datetime = set_remote_timestamp(repo, truncated_file)

                if name in stats and stats[name]["timestamp"] >= date:
                    num_old += 1
                else:
                    num_new += 1
                    if name in stats and stats[name]["timestamp"] < date:
                        old_stats[name].append(stats[name])

                    card = extract_info_from_psd(full_file, truncated_file)
                    card["path"] = truncated_file
                    card["timestamp"] = date
                    stats[name] = card

                    if output_problematic_cards and card["type"] not in EXCLUDE_FOLDERS:
                        problems = problem_card_checker(card)
                        if problems:
                            problematic_cards.append((truncated_file, card, problems))

                num_updated += 1
                if not num_updated % UPDATE_RATE:
                    if interaction:
                        edit_messages.append((interaction, f"{num_updated} Cards updated.", []))
                    else:
                        print(f"{num_updated} Cards updated.")

    if interaction:
        message_queue.append((interaction, f"{num_new} Had newer timestamps."))
        message_queue.append((interaction, f"{num_old} Did not have newer timestamps."))
    else:
        print(f"{num_new} Had newer timestamps.")
        print(f"{num_old} Did not have newer timestamps.")
    return log_problematic_cards(problematic_cards)

def manual_update_stats(interaction: Interaction, output_problematic_cards: bool = True, use_local_repo: bool = True, use_local_timestamp: bool = True):
    if interaction:
        message_queue.append((interaction, "Going to update the database for card statistics in the background, this will take a while."))
    else:
        print("Going to update the database for card statistics in the background, this will take a while.")

    problem_cards = update_stats(interaction, output_problematic_cards, use_local_repo, use_local_timestamp)
    for key in metadata:
        if key in stats:
            stats[key].update(metadata[key])

    if interaction:
        message_queue.append((interaction, "Done updating card statistics."))
    else:
        print("Done updating card statistics.")

    if problem_cards:
        if interaction:
            # Loop 1: Output in-depth problems
            bundle = ""
            for card in problem_cards:
                bundle += card
                if len(bundle) > 1000:
                    message_queue.append((interaction, bundle))
                    bundle = ""

            # Loop 2: Output cardnames only
            for card in problem_cards:
                message_queue.append((interaction, card.split('```')[0]))
        else:
            for card in problem_cards:
                print(card)
            print()
            for card in problem_cards:
                print(card.split('```')[0])

    command_queue.append(((), try_open_stats))

async def export_stats_to_file(interaction: Interaction, only_ability: bool = True, as_csv: bool = True):
    if not stats:
        load_stats()

    if as_csv:
        cards_df = pandas.DataFrame.from_dict(stats).transpose()
        cards_dff = cards_df[cards_df["ability"].notna()].copy()
        cards_dff.to_csv(path_or_buf=EXPORTED_STATS_NAME + '.csv')
        with open(EXPORTED_STATS_NAME + '.csv', 'rb') as f:
            await interaction.followup.send(file=File(f, EXPORTED_STATS_NAME + '.csv'))
        return

    with open(EXPORTED_STATS_NAME + '.txt', 'w') as f:
        for name, stat in stats.items():
            f.write(name + '\n')
            for field, metric in stat.items():
                if not only_ability or field == "ability":
                    if type(metric) == list:
                        f.write(' '.join(metric) + '\n')
                    else:
                        f.write(sub(r'[^\x00-\x7f]',r'', str(metric)) + '\n')
            f.write('\n')

    with open(EXPORTED_STATS_NAME + '.txt', 'rb') as f:
        await interaction.followup.send(file=File(f, EXPORTED_STATS_NAME + '.txt'))

async def export_rulebook_to_file(interaction: Interaction):
    if not stats:
        load_stats()

    with open(EXPORTED_RULES_NAME + '.txt', 'w') as f:
        for name, stat in stats.items():
            if "Rulebook" in stat["path"]:
                f.write(name + '\n')
                if "ability" in stat:
                    f.write(sub(r'[^\x00-\x7f]',r'', str(stat["ability"])) + '\n')
                f.write('\n')

    with open(EXPORTED_RULES_NAME + '.txt', 'rb') as f:
        await interaction.followup.send(file=File(f, EXPORTED_RULES_NAME + '.txt'))
