from difflib import SequenceMatcher, get_close_matches
from discord import Message
from pickle import load, dump
from requests import get

from commands.tools import messages

repository = "MichaelJSr/TTSCardMaker"

alias_pickle_name = "aliases.pkl"
git_file_alias: dict[str, str] = {}
git_files: dict[str, str] = {}
ambiguous_names: dict[str, tuple[str, ...]] = {}
git_filenames: list[str] = []

descriptions = {}
file_descriptions = []
descriptions_pickle_name = "descriptions.pkl"
descriptions_dir = "descriptions"
match_ratio = 0.6

def populate_files():
    global git_filenames
    # Grab repo
    repo = get(f"https://api.github.com/repos/{repository}/git/trees/main?recursive=1")
    if repo.status_code != 200:
        return repo.status_code

    # Make a dictionary of all list of pngs in the github
    # Keys consist of the filename, for example Bomb.png
    # Values consist of the whole path, for example Items/Attack/2 Stars/Bomb.png
    for i in repo.json()["tree"]:
        if i["path"].endswith(".png"):
            filepath: str = i["path"]
            png_filename = filepath[filepath.rindex("/") + 1:].lower()
            # If we're putting the same name twice, it will change both names to include the top level folder
            if png_filename in git_files:
                new_filename = f"{filepath[0:filepath.index('/')]}/{png_filename}".lower()
                old_filename = f"{git_files[png_filename][0:git_files[png_filename].index('/')]}/{png_filename}".lower()
                git_files[new_filename] = filepath.replace(" ", "%20")
                git_files[old_filename] = git_files[png_filename]

                # Mark which files are ambiguous
                if png_filename in ambiguous_names:
                    if old_filename not in ambiguous_names[png_filename]:
                        ambiguous_names[png_filename] = (*ambiguous_names[png_filename], old_filename)
                    ambiguous_names[png_filename] = (*ambiguous_names[png_filename], new_filename)
                else:
                    ambiguous_names[png_filename] = (old_filename, new_filename)
            else:
                git_files[png_filename] = filepath.replace(" ", "%20")

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
        return list(set(populate_descriptions(desc1.strip()) + populate_descriptions(desc2.strip())))
    if "&" in desc:
        desc1, desc2 = desc.split("&", 1)
        return list(set(populate_descriptions(desc1.strip())) & set(populate_descriptions(desc2.strip())))

    opposite = False
    exact_match = False
    if desc.startswith("!"):
        desc = desc[1:].strip()
        opposite = True

    if desc.startswith("\"") and desc.endswith("\""):
        desc = desc[1:-1].strip()
        exact_match = True

    closest = []
    farthest = []
    for card in file_descriptions:
        elem_added = False
        for line in re.split("(?<!\d)\.(?!\d)|,|:|;", card):
            line = line.strip()
            if desc in line:
                closest.append(card)
                elem_added = True
                break
            elif not exact_match and SequenceMatcher(None, desc, line).ratio() > match_ratio:
                closest.append(card)
                elem_added = True
                break

        if opposite and not elem_added:
            farthest.append(card)

    if opposite:
        return farthest
    return closest

def should_it_be_pickled(line: str):
    return (line != "attribute" and line != "ability" and line != "name" and line != "line"
            and line != "health" and line != "hp" and line != "defense" and line != "def"
            and line != "attack" and line != "atk" and line != "speed" and line != "spd" and not line.isdigit())

def pickle_descriptions():
    import os
    dir = os.fsencode(descriptions_dir)
    for file in os.listdir(dir):
        filename = os.fsdecode(file).lower()
        if filename.endswith(".txt"): 
            with open(os.path.join(dir, file)) as card:
                card_description = ""
                for line in card.lower():
                    line = line.strip()
                    if line.startswith("["):
                        line = line[1:-1].strip()
                    if should_it_be_pickled(line):
                        card_description += line + "."
                if card_description != "":
                    descriptions[card_description] = filename.split(".psd")[0] + ".png"

    with open(descriptions_pickle_name, 'wb') as f:
        dump(descriptions, f)

def try_open_alias():
    global git_file_alias

    print("Trying to open %s" % alias_pickle_name)
    try:
        with open(alias_pickle_name, 'rb') as f:
            git_file_alias = load(f)
    except EOFError:
        print("%s is completely empty, populating with empty dict..." % alias_pickle_name)
        with open(alias_pickle_name, 'wb') as f:
            dump(git_file_alias, f)
    except FileNotFoundError:
        print("%s doesnt exist, populating with empty dict..." % alias_pickle_name)
        with open(alias_pickle_name, 'wb') as f:
            dump(git_file_alias, f)

def try_open_descriptions():
    global descriptions, file_descriptions

    print("Trying to open %s" % descriptions_pickle_name)
    try:
        with open(descriptions_pickle_name, 'rb') as f:
            descriptions = load(f)
    except EOFError:
        print("%s is completely empty, populating with empty dict..." % descriptions_pickle_name)
        with open(descriptions_pickle_name, 'wb') as f:
            dump(descriptions, f)
    except FileNotFoundError:
        print("%s doesnt exist, populating with empty dict..." % descriptions_pickle_name)
        with open(descriptions_pickle_name, 'wb') as f:
            dump(descriptions, f)

    if not descriptions:
        try:
            pickle_descriptions()
        except FileNotFoundError:
            print("No descriptions are pickled, and no files designated for pickling either.")
            return
    file_descriptions = descriptions.keys()

async def query_remote(message: Message):
    # If it's just a ?, ignore everything
    if len(message.content) == 1:
        return

    card = message.content[1:].replace(" ", "_").lower()
    if not card.endswith(".png"):
        card += ".png"
    try:
        closest = get_close_matches(card, git_filenames, n=1, cutoff=match_ratio)[0]
    except IndexError:
        await message.channel.send("No card found!")
        return
    await message.channel.send(f"https://raw.githubusercontent.com/{repository}/main/{git_files[closest]}")

    # If the filename was ambiguous, make a note of that.
    if closest in ambiguous_names:
        ambiguous_message = "Ambiguous name found. If this wasn't the card you wanted, try typing: \n"
        for i in ambiguous_names[closest]:
            ambiguous_message += f"{i}\n"
        await message.channel.send(ambiguous_message)

async def query_pickle(message: Message):
    if len(message.content.split()) < 2:
        await message.channel.send("Must specify atleast one argument, the search query.")
        return

    desc = " ".join(message.content.split()[1:]).strip().lower()
    closest = populate_descriptions(desc)

    for close in closest:
        try:
            await message.channel.send(f"https://raw.githubusercontent.com/{repository}/main/{git_files[descriptions[close]]}")
        except KeyError:
            await message.channel.send("No such key: %s" % descriptions[close])
            descriptions.pop(close)
            with open(descriptions_pickle_name, 'wb') as f:
                dump(descriptions, f)
    await message.channel.send("%d Results found for %s!" % (len(closest), desc))

async def howmany_description(message: Message):
    if len(message.content.split()) < 2:
        await message.channel.send("Must specify atleast one argument, the search query.")
        return

    desc = " ".join(message.content.split()[1:]).strip().lower()
    closest = populate_descriptions(desc)

    await message.channel.send("%d Results found for %s!" % (len(closest), desc))

def print_all_aliases(message: Message):
    all_aliases = "```"
    for key, val in git_file_alias.items():
        all_aliases += f"{key:20s} -> {val}\n"

    messages.append((message.channel.id, all_aliases + "```Done."))

async def alias_card(message: Message):
    global git_filenames
    if len(message.content.split()) != 3:
         await message.channel.send("Must specify exactly two arguments, the key and value, or 'del' and the key to delete.")
         return print_all_aliases(message)

    key, val = message.content.split(".alias")[1].lower().split()

    if key == "del":
        if val in git_file_alias:        
            await message.channel.send(f"Deleted alias: {val} -> {git_file_alias.pop(val)}")

            if not val + ".png" in git_files:
                await message.channel.send("No such key exists: %s\nCouldnt pop alias from dictionary." % val)
                return

            git_files.pop(f"{val}.png")
            git_filenames = git_files.keys()
        else:
            await message.channel.send("No value exists for the alias: %s" % val)
            return
    else:
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
            await message.channel.send("No such value exists: %s\nCouldnt add alias into dictionary." % val)
            return

        git_filenames = git_files.keys()
        git_file_alias[key] = val
        await message.channel.send(f"Created alias: {key} -> {val}")

    with open(alias_pickle_name, 'wb') as f:
        dump(git_file_alias, f)

async def set_match_ratio(message: Message):
    global match_ratio
    if len(message.content.split()) < 2:
        await message.channel.send("Must specify exactly one argument, the new match ratio. The old match ratio was %f." % match_ratio)
        return

    match_ratio = float(message.content.split()[1])
    await message.channel.send("New match ratio of %f set!" % match_ratio)

async def set_repository(message: Message):
    global repository
    if len(message.content.split()) < 2:
        await message.channel.send("Must specify exactly one argument, the new repository (USER/REPO). The old repository was %s." % repository)
        return

    repository = message.content.split()[1]
    status = populate_files()
    if status != 200:
       print(f"Error {status} when requesting github.")
    await message.channel.send("New repository of %s set!" % repository)
