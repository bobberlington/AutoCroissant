from datetime import datetime

import pandas
from discord import Interaction
from collections import defaultdict
from github import Github
from logging import getLogger, CRITICAL
from os import path, walk
from os.path import getmtime, basename
from pickle import load, dump
from psd_tools import PSDImage
from re import finditer, sub
from requests import get, Response
from urllib.request import urlretrieve

from commands.utils import edit_messages, messages

getLogger("psd_tools").setLevel(CRITICAL)
LOCAL_DIR_LOC   = "~/Desktop/TTSCardMaker"
STATS_PKL       = "stats.pkl"
OLD_STATS_PKL   = "old_stats.pkl"
DF_PKL          = "stats_df.pkl"
UPDATE_RATE     = 25

local_repo: str = path.expanduser(LOCAL_DIR_LOC)
use_local: bool = True
# Should we use Michael time?
use_local_mtime: bool = True

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
stats_df = pandas.DataFrame()
all_types = []
all_stars = []
all_attrs = []

def pickle_stats():
    with open(STATS_PKL, 'wb') as f:
        dump(stats, f)
    with open(OLD_STATS_PKL, 'wb') as f:
        dump(old_stats, f)
    with open(DF_PKL, 'wb') as f:
        dump(stats_df, f)

def update_stats(interaction: Interaction) -> tuple[str, ...]:
    global stats, old_stats, stats_df

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

    problem_cards = []
    if use_local:
        problem_cards = traverse_local_repo(interaction)
    else:
        problem_cards = traverse_repo(interaction)
    stats_df = pandas.DataFrame.from_dict(stats).transpose()
    pickle_stats() # Now that we updated the descriptions, store them back
    return problem_cards

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
    ability = ""
    hp = df = atk = spd = None
    hp_found = df_found = atk_found = spd_found = False
    type_bboxes: list[tuple[str, tuple[int, int]]] = []
    for layer in PSDImage.open(file_loc).descendants():
        if layer.name.lower() == "ability" and not layer.is_group():
            ability = str(layer.engine_dict["Editor"]["Text"]).replace('\\r', ' ').replace('\\t', '').replace('\\x03', '').replace('\\ufeff', '').replace('\\n', ' ').rstrip()
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
        elif layer.name.lower() in all_types and not "stat" in layer.parent.name.lower():
            if not layer.is_group() and layer.is_visible():
                type_bboxes.append((layer.name.lower(), layer.bbox[:2]))

    if ability:
        type_bboxes = prune_bbox(sort_bbox(type_bboxes))
        # Find all instances of multiple whitespace, or whitespace followed by colon
        matches = [match for match in finditer(r'\s{3,}|\s{3,}:', ability)]

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
    if (card["type"] == "creature" or card["type"] == "minion") and card["atk"] == -1:
        problems.append("ATK NOT FOUND")
    if (card["type"] == "creature" or card["type"] == "minion") and card["spd"] == -1:
        problems.append("SPD NOT FOUND")

    if "problem" in card and card["problem"]:
        problems.extend(card["problem"])

    return problems

def log_problematic_cards(problematic_cards):
    problem_cards = []
    for loc, card, problems in problematic_cards:
        problem_cards.append(f"{loc}\n" + '```' + '\n'.join(problems) + '```')
    return problem_cards

def traverse_repo(interaction: Interaction) -> tuple[str, ...]:
    from commands.query_card import REPOSITORY
    resp = get(f"https://api.github.com/repos/{REPOSITORY}/git/trees/main?recursive=1", headers=headers)
    if resp.status_code != 200:
        print(f"Error when trying to connect to {REPOSITORY}")
        return resp.status_code
    # This uses an api request
    repo = Github(login_or_token=git_token).get_repo(REPOSITORY)

    populate_types_stars_attrs(resp)
    problematic_cards = []
    num_updated = 0
    num_new = 0
    num_old = 0
    for i in resp.json()["tree"]:
        path: str = i["path"]
        if path.endswith('.psd'):
            commits = repo.get_commits(path=path)

            date: datetime = commits[0].commit.committer.date.timestamp()

            if path in stats and stats[path]["timestamp"] >= date:
                num_old += 1
                continue
            else:
                num_new += 1
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

            num_updated += 1
            if not num_updated % UPDATE_RATE:
                edit_messages.append((interaction, f"{num_updated} Cards updated.", []))

    messages.append((interaction, f"{num_new} Had newer timestamps."))
    messages.append((interaction, f"{num_old} Did not have newer timestamps."))
    return log_problematic_cards(problematic_cards)

def traverse_local_repo(interaction: Interaction):
    from commands.query_card import REPOSITORY
    repo = None
    if not use_local_mtime:
        print(f"Warning: Getting the timestamp of a remote file uses up an api request, you can make up to {5000 if git_token else 60} requests")
        repo = Github(login_or_token=git_token).get_repo(REPOSITORY)

    populate_types_stars_attrs()
    problematic_cards = []
    num_updated = 0
    num_new = 0
    num_old = 0
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
                    num_old += 1
                    continue
                else:
                    num_new += 1
                    if truncated_file in stats and stats[truncated_file]["timestamp"] < date:
                        old_stats[truncated_file].append(stats[truncated_file])

                    card = extract_info_from_psd(full_file, truncated_file)
                    card["name"] = basename(full_file)[:-len('.psd')]
                    card["timestamp"] = date

                    problems = problem_card_checker(card)
                    if problems:
                        problematic_cards.append((truncated_file, card, problems))

                    stats[truncated_file] = card

                num_updated += 1
                if not num_updated % UPDATE_RATE:
                    edit_messages.append((interaction, f"{num_updated} Cards updated.", []))

    messages.append((interaction, f"{num_new} Had newer timestamps."))
    messages.append((interaction, f"{num_old} Did not have newer timestamps."))
    return log_problematic_cards(problematic_cards)

def manual_update_stats(interaction: Interaction, output_problematic_cards: bool = True, use_local_repo: bool = True, use_local_timestamp: bool = True):
    global use_local, use_local_mtime
    use_local = use_local_repo
    use_local_mtime = use_local_timestamp

    messages.append((interaction, "Going to update the database for card statistics in the background, this will take a while."))
    problem_cards = update_stats(interaction)
    messages.append((interaction, "Done updating card statistics."))
    if output_problematic_cards:
        # Loop 1: Output in-depth problems
        bundle = ""
        for card in problem_cards:
            bundle += card
            if len(bundle) > 1000:
                messages.append((interaction, bundle))
                bundle = ""

        # Loop 2: Output cardnames only
        list_of_all_cards = []
        for card in problem_cards:
            list_of_all_cards.append(card.split('```')[0])
            if len(list_of_all_cards) > 30:
                messages.append((interaction, ''.join(list_of_all_cards)))
                list_of_all_cards = []
        if list_of_all_cards:
            messages.append((interaction, ''.join(list_of_all_cards)))
