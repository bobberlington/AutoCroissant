from datetime import datetime
from discord import Message
from github import Github
from os import path, walk
from os.path import getmtime, basename
from pickle import load, dump
from psd_tools import PSDImage
from requests import get, Response
from urllib.request import urlretrieve

repository = "MichaelJSr/TTSCardMaker" # Remove this after I am done testing
local_dir_loc = "~/Desktop/TTSCardMaker"
descriptions_pickle_name = "descriptions.pkl"

local_repo: str = path.expanduser(local_dir_loc)
use_local: bool = True
# Should we use Michael time?
use_local_mtime: bool = False

headers: dict = None
git_token: str = None
try:
    #import config
    #git_token = config.git_token
    headers = {'Authorization': 'token ' + git_token}
    print("Git token found, api will be limited to 5000 requests/hour.")
except AttributeError:
    print("No git token in config, api will be limited to 60 requests/hour.")

descs = {}
all_types = []
all_stars = []
all_attrs = []

def pickle_descriptions():
    with open(descriptions_pickle_name, 'wb') as f:
        dump(descs, f)

def update_descriptions():
    global descs

    print("Trying to open %s" % descriptions_pickle_name)
    try:
        with open(descriptions_pickle_name, 'rb') as f:
            descs = load(f)
        print("Existing dict found in %s, updating entries..." % descriptions_pickle_name)
    except (EOFError, FileNotFoundError):
        print("%s is completely empty or doesn't exist, rebuilding entire dict..." % descriptions_pickle_name)

    if use_local:
        traverse_local_repo()
    else:
        traverse_repo()
    pickle_descriptions() # Now that we updated the descriptions, store them back

def populate_types_stars_attrs(resp: Response | None = None):
    if use_local and not resp:
        for folder, _, files in walk(local_repo + "/Types"):
            for file in files:
                if folder.endswith("Types"):
                    all_types.append(file)
                elif folder.endswith("Stars"):
                    all_stars.append(file)
                elif folder.endswith("Attributes"):
                    all_attrs.append(file)
    elif resp:
        for i in resp.json()["tree"]:
            path: str = i["path"]
            if path.startswith("Types"):
                if path.find('.') == -1:
                    continue

                file = path.split('/')[-1][:-len('.png')]
                if path.startswith("Types/Stars"):
                    all_stars.append(file)
                elif path.startswith("Types/Attributes"):
                    all_attrs.append(file)
                else:
                    all_types.append(file)

def classify_card(relative_loc: str):
    if not relative_loc:
        return

    folders = relative_loc.split('/')[:-1]
    print(folders)
    if len(folders) > 0 and folders[0] == "MDW":
        return
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
    elif len(folders) > 0 and folders[0] == "Auxillary":
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

#TODO: Make this a sorting function?
def compare_bbox(bbox1: tuple[int, int], bbox2: tuple[int, int]):
    # Compare y values within an epsilon of 10 pixels
    # Then compare x values
    return

def extract_info_from_psd(file_loc: str, relative_loc: str = ""):
    card = classify_card(relative_loc)
    ability = ""
    hp = 0
    hp_found = False
    df = 0
    df_found = False
    atk = 0
    atk_found = False
    spd = 0
    spd_found = False
    for layer in PSDImage.open(file_loc).descendants():
        print(layer.name)
        print(layer.bbox[:2])
        # layer_image = layer.as_PIL()
        if layer.name.lower() == "ability":
            ability = layer.text
        elif ("hp dark" in layer.parent.name.lower()) and layer.name.isdigit():
            if layer.is_visible():
                hp += int(layer.name)
            hp_found = True
        elif ("def dark" in layer.parent.name.lower()) and layer.name.isdigit():
            if layer.is_visible():
                df += int(layer.name)
            df_found = True
        elif ("atk dark" in layer.parent.name.lower()) and layer.name.isdigit():
            if layer.is_visible():
                atk += int(layer.name)
            atk_found = True
        elif ("spd dark" in layer.parent.name.lower()) and layer.name.isdigit():
            if layer.is_visible():
                spd += int(layer.name)
            spd_found = True

        # If we actually did find a stat value inside the card, but there was no visible "dark" layer,
        # assume the stat is equal to 10.
        if hp_found and not hp:
            hp = 10
        if df_found and not df:
            df = 10
        if atk_found and not atk:
            atk = 10
        if spd_found and not spd:
            spd = 10

    if ability:
        card["ability"] = ability.strip('\'').strip('\"').strip()
    if hp_found:
        card["hp"] = hp
    if df_found:
        card["def"] = df
    if atk_found:
        card["atk"] = atk
    if spd_found:
        card["spd"] = spd

    if card["type"] == "unknown":
        print("UNKOWN TYPE %s" % relative_loc)
    if card["type"] != "unknown" and not ability:
        print("ABILITY TEXT NOT FOUND FOR %s" % relative_loc)

    if card == "creature" and not hp:
        print("HP NOT FOUND FOR %s" % relative_loc)
    if card == "creature" and not df:
        print("DEF NOT FOUND FOR %s" % relative_loc)
    if card == "creature" and not atk:
        print("ATK NOT FOUND FOR %s" % relative_loc)
    if card == "creature" and not spd:
        print("SPD NOT FOUND FOR %s" % relative_loc)

    return card

def traverse_repo():
    #from commands.query_card import repository
    resp = get(f"https://api.github.com/repos/{repository}/git/trees/main?recursive=1", headers=headers)
    if resp.status_code != 200:
        print("Error when trying to connect to %s" % repository)
        return resp.status_code
    # This uses an api request
    repo = Github(login_or_token=git_token).get_repo(repository)

    populate_types_stars_attrs(resp)
    print(all_types)
    print(all_stars)
    print(all_attrs)
    for i in resp.json()["tree"]:
        path: str = i["path"]
        if path.endswith('.psd'):
            commits = repo.get_commits(path=path)

            date: datetime = commits[0].commit.committer.date.timestamp()

            # This uses an api request
            card = extract_info_from_psd(urlretrieve(f"https://raw.githubusercontent.com/{repository}/main/{path}")[0], path)
            card["name"] = basename(path)[:-len('.psd')]
            card["timestamp"] = date
            print(card)
            descs[path] = card
            return

def traverse_local_repo():
    repo = None
    if not use_local_mtime:
        print("Warning: Getting the timestamp of a remote file uses up an api request, you can make up to %d requests" % (5000 if git_token else 60))
        repo = Github(login_or_token=git_token).get_repo(repository)

    populate_types_stars_attrs()
    print(all_types)
    print(all_stars)
    print(all_attrs)
    for folder, _, files in walk(local_repo):
        folder += '/'
        for file in files:
            if file.endswith('.psd'):
                full_file = folder.replace('\\', '/') + file
                truncated_file = full_file.split("TTSCardMaker")[-1].strip('/')

                date = -1.0
                if not use_local_mtime:
                    commits = repo.get_commits(path=truncated_file)
                    date = commits[0].commit.committer.date.timestamp()
                else:
                    date = getmtime(full_file)

                card = extract_info_from_psd(full_file, truncated_file)
                card["name"] = basename(full_file)[:-len('.psd')]
                card["timestamp"] = date
                print(card)
                descs[truncated_file] = card
                return

traverse_repo()