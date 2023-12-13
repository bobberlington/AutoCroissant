import requests
import difflib
import pickle

# Only need this import when populating card descriptions
import os
import re
#from global_config import list_of_all_attributes, list_of_all_stars

alias_pickle_name = "aliases.pkl"
file_alias = {}
files = {}
ambiguous_names = {}
filenames = []

descriptions = {}
file_descriptions = []
descriptions_pickle_name = "descriptions.pkl"
descriptions_dir = "descriptions"
match_ratio = 0.6

def populate_files():
    # Grab repo
    repo = requests.get("https://api.github.com/repos/MichaelJSr/TTSCardMaker/git/trees/main?recursive=1")
    if repo.status_code != 200:
        return repo.status_code

    text = repo.json()
    # Make a dictionary of all list of pngs in the github
    # Keys consist of the filename, for example Bomb.png
    # Values consist of the whole path, for example Items/Attack/2 Stars/Bomb.png
    for i in text["tree"]:
        if i["path"].endswith(".png"):
            filepath = i["path"]
            png_filename = filepath[filepath.rindex("/") + 1:].lower()
            # If we're putting the same name twice, it will change both names to include the top level folder
            if png_filename in files:
                new_filename = f"{filepath[0:filepath.index('/')]}/{png_filename}".lower()
                old_filename = f"{files[png_filename][0:files[png_filename].index('/')]}/{png_filename}".lower()
                files[new_filename] = filepath.replace(" ", "%20")
                files[old_filename] = files[png_filename]

                # Mark which files are ambiguous
                if png_filename in ambiguous_names.keys():
                    if old_filename not in ambiguous_names[png_filename]:
                        ambiguous_names[png_filename].append(old_filename)
                    ambiguous_names[png_filename].append(new_filename)
                else:
                    ambiguous_names[png_filename] = []
                    ambiguous_names[png_filename].append(old_filename)
                    ambiguous_names[png_filename].append(new_filename)
            else:
                files[png_filename] = filepath.replace(" ", "%20")

    # Fill up the aliases
    # keys of aliases are the aliases, values are the filenames they point to
    for i in file_alias.keys():
        if file_alias[i] in files:
            files[f"{i}.png"] = files[file_alias[i]]
        else:
            files[f"{i}.png"] = [val for _, val in files.items() if val.lower().endswith(file_alias[i])][0]
    
    return 200

def populate_descriptions(desc):
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
            elif not exact_match and difflib.SequenceMatcher(None, desc, line).ratio() > match_ratio:
                closest.append(card)
                elem_added = True
                break

        if opposite and not elem_added:
            farthest.append(card)
    
    if opposite:
        return farthest
    return closest

def should_it_be_pickled(line):
    return (line != "attribute" and line != "ability" and line != "name" and line != "line"
            and line != "health" and line != "hp" and line != "defense" and line != "def"
            and line != "attack" and line != "atk" and line != "speed" and line != "spd" and not line.isdigit())

def pickle_descriptions():
    dir = os.fsencode(descriptions_dir)
        
    for file in os.listdir(dir):
        filename = os.fsdecode(file)
        if filename.endswith(".txt"): 
            card_name = filename.split(".psd")[0].lower() + ".png"
            with open(os.path.join(dir, file)) as card:
                card_description = ""
                for line in card:
                    line = line.strip().lower()
                    if line.startswith("["):
                        line = line[1:-1].strip()
                    if should_it_be_pickled(line):
                        card_description += line + "."
                if card_description != "":
                    descriptions[card_description] = card_name
        else:
            continue
    
    with open(descriptions_pickle_name, 'wb') as f:
        pickle.dump(descriptions, f)

async def try_open_alias(message):
    global file_alias

    await message.channel.send("Trying to open %s" % alias_pickle_name)
    try:
        with open(alias_pickle_name, 'rb') as f:
            file_alias = pickle.load(f)
    except EOFError:
        await message.channel.send("%s is completely empty, populating with empty dict..." % alias_pickle_name)
        with open(alias_pickle_name, 'wb') as f:
            pickle.dump(file_alias, f)
    except FileNotFoundError:
        await message.channel.send("%s doesnt exist, populating with empty dict..." % alias_pickle_name)
        with open(alias_pickle_name, 'wb') as f:
            pickle.dump(file_alias, f)

async def try_open_descriptions(message):
    global descriptions

    await message.channel.send("Trying to open %s" % descriptions_pickle_name)
    try:
        with open(descriptions_pickle_name, 'rb') as f:
            descriptions = pickle.load(f)
    except EOFError:
        await message.channel.send("%s is completely empty, populating with empty dict..." % descriptions_pickle_name)
        with open(descriptions_pickle_name, 'wb') as f:
            pickle.dump(descriptions, f)
    except FileNotFoundError:
        await message.channel.send("%s doesnt exist, populating with empty dict..." % descriptions_pickle_name)
        with open(descriptions_pickle_name, 'wb') as f:
            pickle.dump(descriptions, f)

async def query_remote(message):
    global filenames

    # If it's just a ?, ignore everything
    if len(message.content) == 1:
        return

    if not file_alias:
        await try_open_alias(message)
    if not files:
        status = populate_files()
        if status != 200:
            await message.channel.send(f"Error {status} when requesting github.")
            return
    if not filenames:
        filenames = files.keys()

    card = message.content[1:].replace(" ", "_").lower()
    if not card.endswith(".png"):
        card += ".png"
    try:
        closest = difflib.get_close_matches(card, filenames, n=1, cutoff=match_ratio)[0]
    except IndexError:
        await message.channel.send("No card found!")
        return
    await message.channel.send(f"https://raw.githubusercontent.com/MichaelJSr/TTSCardMaker/main/{files[closest]}")

    # If the filename was ambiguous, make a note of that.
    if closest in ambiguous_names:
        ambiguous_message = "Ambiguous name found. If this wasn't the card you wanted, try typing: \n"
        for i in ambiguous_names[closest]:
            ambiguous_message += f"{i}\n"
        await message.channel.send(ambiguous_message)

async def query_pickle(message):
    global file_descriptions

    if len(message.content.split()) < 2:
        await message.channel.send("Must specify atleast one argument, the search query.")
        return

    if not file_alias:
        await try_open_alias(message)
    if not files:
        populate_files()
    if not descriptions:
        await try_open_descriptions(message)
        if not descriptions:
            try:
                pickle_descriptions()
            except FileNotFoundError:
                await message.channel.send("No descriptions are pickled, and no files designated for pickling either.")
                return
    if not file_descriptions:
        file_descriptions = descriptions.keys()
    
    desc = " ".join(message.content.split()[1:]).strip().lower()
    closest = populate_descriptions(desc)

    for close in closest:
        try:
            await message.channel.send(f"https://raw.githubusercontent.com/MichaelJSr/TTSCardMaker/main/{files[descriptions[close]]}")
        except KeyError:
            await message.channel.send("No such key: %s" % descriptions[close])
            descriptions.pop(close)
            with open(descriptions_pickle_name, 'wb') as f:
                pickle.dump(descriptions, f)
    await message.channel.send("%d Results found for %s!" % (len(closest), desc))

async def howmany_description(message):
    global file_descriptions

    if len(message.content.split()) < 2:
        await message.channel.send("Must specify atleast one argument, the search query.")
        return

    if not file_alias:
        await try_open_alias(message)
    if not files:
        populate_files()
    if not descriptions:
        await try_open_descriptions(message)
        if not descriptions:
            try:
                pickle_descriptions()
            except FileNotFoundError:
                await message.channel.send("No descriptions are pickled, and no files designated for pickling either.")
                return
    if not file_descriptions:
        file_descriptions = descriptions.keys()
    
    desc = " ".join(message.content.split()[1:]).strip().lower()
    closest = populate_descriptions(desc)

    await message.channel.send("%d Results found for %s!" % (len(closest), desc))

async def alias_card(message):
    global filenames

    if len(message.content.split()) != 3:
         await message.channel.send("Must specify exactly two arguments, the key and value.")
         return

    if not file_alias:
        await try_open_alias(message)
    if not files:
        populate_files()
    
    alias = message.content.split(".alias")
    key, val = alias[1].split()
    key = key.strip().lower()
    val = val.strip().lower()
    if not val.endswith(".png"):
        val += ".png"

    check_val_validity = False
    for _, val_in_dict in files.items():
        if val_in_dict.lower().endswith(val):
            files[f"{key}.png"] = val_in_dict
            filenames = list(files.keys())
            check_val_validity = True
            break
    if not check_val_validity:
        await message.channel.send("No such value exists: %s\nCouldnt add alias into dictionary." % val)
        return

    file_alias[key] = val
    await message.channel.send(f"Created alias: {key} -> {val}")

    with open(alias_pickle_name, 'wb') as f:
        pickle.dump(file_alias, f)

async def delete_alias(message):
    global filenames

    if len(message.content.split()) != 2:
        await message.channel.send("Must specify exactly one argument, the key.")
        return

    if not file_alias:
        await try_open_alias(message)
    if not files:
        populate_files()
    
    alias = message.content.split(".del_alias")[1].strip().lower()
    if alias in file_alias:        
        await message.channel.send(f"Deleted alias: {alias} -> {file_alias.pop(alias)}")

        if alias + ".png" in files:
            files.pop(f"{alias}.png")
            filenames = list(files.keys())
        else:
            await message.channel.send("No such key exists: %s\nCouldnt pop alias from dictionary." % alias)
            return
    else:
        await message.channel.send("No value exists for the alias: %s" % alias)
        return

    with open(alias_pickle_name, 'wb') as f:
        pickle.dump(file_alias, f)

async def print_all_aliases(message):
    if not file_alias:
        await try_open_alias(message)
    
    all_aliases = "```"
    for key, val in file_alias.items():
        all_aliases += f"{key:20s} -> {val}\n"
    
    await message.channel.send(f"{all_aliases}```Done.")

async def set_match_ratio(message):
    global match_ratio

    if len(message.content.split()) < 2:
        await message.channel.send("Must specify exactly one argument, the new match ratio. The old match ratio was %f." % match_ratio)
        return

    match_ratio = float(message.content.split()[1])

    await message.channel.send("New match ratio of %f set!" % match_ratio)
