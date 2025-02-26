from datetime import datetime
from discord import Interaction
from collections import defaultdict
from github import Github
from os import path, walk
from os.path import getmtime, basename
from pickle import load, dump
from psd_tools import PSDImage
from re import finditer, sub
from requests import get, Response
from urllib.request import urlretrieve

from commands.utils import messages

LOCAL_DIR_LOC   = "~/Desktop/TTSCardMaker"
STATS_PKL       = "stats.pkl"
OLD_STATS_PKL   = "old_stats.pkl"

local_repo: str = path.expanduser(LOCAL_DIR_LOC)
use_local: bool = True
# Should we use Michael time?
use_local_mtime: bool = False

headers: dict = None
git_token: str = None
try:
    import config
    git_token = config.git_token
    headers = {'Authorization': 'token ' + git_token}
    print("Git token found, api will be limited to 5000 requests/hour.")
except AttributeError:
    print("No git token in config, api will be limited to 60 requests/hour.")

stats = {}
old_stats = defaultdict(list)
all_types = []
all_stars = []
all_attrs = []

def pickle_stats():
    with open(STATS_PKL, 'wb') as f:
        dump(stats, f)
    with open(OLD_STATS_PKL, 'wb') as f:
        dump(old_stats, f)

def update_stats():
    global stats, old_stats

    print(f"Trying to open {STATS_PKL}")
    try:
        with open(STATS_PKL, 'rb') as f:
            stats = load(f)
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

    problem_strs = []
    if use_local:
        problem_strs = traverse_local_repo()
    else:
        problem_strs = traverse_repo()
    pickle_stats() # Now that we updated the descriptions, store them back
    return problem_strs

def populate_types_stars_attrs(resp: Response | None = None):
    if use_local and not resp:
        for folder, _, files in walk(local_repo + "/Types"):
            for file in files:
                file = file[:-len('.png')].lower()
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
                if not '.' in path:
                    continue

                file = path.split('/')[-1][:-len('.png')].lower()
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
    if len(folders) > 0 and folders[0] == "MDW":
        return {
            "type" : "MDW",
            "ability" : "Placeholder"
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
    type_bboxes: list[tuple[str, tuple[int, int]]] = []
    for layer in PSDImage.open(file_loc).descendants():
        # layer_image = layer.as_PIL()
        if layer.name.lower() == "ability" and not layer.is_group():
            ability = str(layer.engine_dict["Editor"]["Text"]).replace('\\r', '\n').replace('\\t', '').replace('\\x00', '').replace('\\x01', '').replace('\\x03', '').replace('\\x10', '').replace('\\x0bge', '').rstrip()
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
        elif layer.name.lower() in all_types and not "stat" in layer.parent.name.lower():
            if not layer.is_group() and layer.is_visible():
                type_bboxes.append((layer.name.lower(), layer.bbox[:2]))

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
        type_bboxes = prune_bbox(sort_bbox(type_bboxes))
        # Find all instances of multiple whitespace, or whitespace followed by colon
        matches = [match for match in finditer(r'\s{3,}|\s:', ability)]

        if len(matches) > 0 and len(type_bboxes) > 0:
            if len(type_bboxes) < len(matches):
                if not "problem" in card:
                    card["problem"] = []
                card["problem"].append("INCORRECT TYPE NAMES")
            else:
                count = len(type_bboxes) - 1
                for match in matches[::-1]:
                    ability = ability[:match.start()] + ' ' + type_bboxes[count][0] + ' ' + ability[match.end():]
                    count -= 1
        card["ability"] = sub(r'\s+([:;,\.\?!])', r'\1', ability).strip('\'').strip('\"').strip()

    if hp_found:
        card["hp"] = hp
    if df_found:
        card["def"] = df
    if atk_found:
        card["atk"] = atk
    if spd_found:
        card["spd"] = spd

    return card

def problem_card_checker(card: dict[str, str]):
    problems = []
    if card["type"] == "unknown":
        problems.append("UNKOWN TYPE")
    if card["type"] != "unknown" and (not "ability" in card or not card["ability"]):
        problems.append("ABILITY TEXT NOT FOUND")

    if card == "creature" and (not "hp" in card or not card["hp"]):
        problems.append("HP NOT FOUND")
    if card == "creature" and (not "def" in card or not card["def"]):
        problems.append("DEF NOT FOUND")
    if card == "creature" and (not "atk" in card or not card["atk"]):
        problems.append("ATK NOT FOUND")
    if card == "creature" and (not "spd" in card or not card["spd"]):
        problems.append("SPD NOT FOUND")

    if "problem" in card and card["problem"]:
        problems.extend(card["problem"])

    return problems

def log_problematic_cards(problematic_cards):
    problem_strs = []
    for loc, card, problems in problematic_cards:
        problem_strs.append(f"{loc}\n" + '```' + '\n'.join(problems) + '```')
    return problem_strs

def traverse_repo():
    from commands.query_card import REPOSITORY
    resp = get(f"https://api.github.com/repos/{REPOSITORY}/git/trees/main?recursive=1", headers=headers)
    if resp.status_code != 200:
        print(f"Error when trying to connect to {REPOSITORY}")
        return resp.status_code
    # This uses an api request
    repo = Github(login_or_token=git_token).get_repo(REPOSITORY)

    populate_types_stars_attrs(resp)
    problematic_cards = []
    for i in resp.json()["tree"]:
        path: str = i["path"]
        if path.endswith('.psd'):
            commits = repo.get_commits(path=path)

            date: datetime = commits[0].commit.committer.date.timestamp()

            if path in stats and stats[path]["timestamp"] >= date:
                continue
            else:
                if path in stats and stats[path]["timestamp"] < date:
                    old_stats[path].append(stats[path])

                # This uses an api request
                card = extract_info_from_psd(urlretrieve(f"https://raw.githubusercontent.com/{REPOSITORY}/main/{path}")[0], path)
                card["name"] = basename(path)[:-len('.psd')]
                card["timestamp"] = date

                problems = problem_card_checker(card)
                if problems:
                    problematic_cards.append((path, card, problems))

                stats[path] = card

    return log_problematic_cards(problematic_cards)

def traverse_local_repo():
    from commands.query_card import REPOSITORY
    repo = None
    if not use_local_mtime:
        print(f"Warning: Getting the timestamp of a remote file uses up an api request, you can make up to {5000 if git_token else 60} requests")
        repo = Github(login_or_token=git_token).get_repo(REPOSITORY)

    populate_types_stars_attrs()
    problematic_cards = []
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

                if truncated_file in stats and stats[truncated_file]["timestamp"] >= date:
                    continue
                else:
                    if truncated_file in stats and stats[truncated_file]["timestamp"] < date:
                        old_stats[truncated_file].append(stats[truncated_file])

                    card = extract_info_from_psd(full_file, truncated_file)
                    card["name"] = basename(full_file)[:-len('.psd')]
                    card["timestamp"] = date

                    problems = problem_card_checker(card)
                    if problems:
                        problematic_cards.append((truncated_file, card, problems))

                    stats[truncated_file] = card

    return log_problematic_cards(problematic_cards)

def manual_update_stats(interaction: Interaction, output_problematic_cards: bool = True):
    messages.append((interaction, "Going to update the database for card statistics in the background, this will take a while."))
    problem_strs = update_stats()
    messages.append((interaction, "Done updating card statistics."))
    if output_problematic_cards:
        for card in problem_strs:
            messages.append((interaction, card))
