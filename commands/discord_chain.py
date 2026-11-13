"""
Card timelines: each live card's chain of versions before git and its Created date (docs/Future/CARD_TIMELINE.md, steps
4 and 5; STAGE1 S5 + S6). Reads the stage-2 parts' outputs under the archive's derived/ (never a repo) and the card
repo (read only):

    python -m commands.discord_chain build  [--root <dir>] [--cards-repo <dir>]
    python -m commands.discord_chain checks [--root <dir>] [--n 20] [--seed 1113]

build   Under derived/chain/:
        timelines/<card id>.json  every live card: its entries oldest first, then the stop (its first git record:
                        the old repo's, else the card repo's) and its Created date. Entries:
                          paper     a paper card's photo the matcher is sure of (owner Q6=A: first, no day, never
                                    Created)
                          discord   one look posted before the card's first git record: a git look (version_sure) or
                                    a look not in git (draft); its reposts merged (the earliest post dates it); a
                                    split card's (owner Q3=A) "as <old name>" on both halves
                          text      a sure changelog fact before the stop (owner Q4=B: a field change as data, or
                                    "ability changed" with no text); item kinds never (hand-crafted metadata)
                          old repo  the old pre-wipe repo's PNG versions (derived/oldrows), with their fold
                        Each version's fields keep their method and confidence; only sure values are kept, the rest
                        name their review item. Each is diffed with the version before (sure fields on both sides):
                        stats with the website's buff/nerf rule (analytics.change: the total up or down, a total
                        over OFF_SCALE not judged), stars, name, types, the ability (text_key; formatting alone is
                        no change, owner 11-09). Discord looks fold as minor only when every field is sure on both
                        sides and unchanged (as the old rows; their types are never sure, so they never fold).
        created.json    Created = the earliest confident appearance (a card-sure Discord picture, the old repo, the
                        card repo) with its source, against site/Card Creation Dates.json: same, earlier (proposed),
                        later (review only: never moved later), new (no date yet), hand (never touched).
        review.jsonl    what the chain keeps out (stable ids): looks not told, named-only pictures before the card's
                        first picture evidence (one section to confirm in bulk), joke drafts, unsure fields, facts a
                        picture contradicts, Created changes, splits, paper photos, the decisions taken without the
                        owner (Q1-Q6).
        stats.txt       counts.
checks  Contact sheets of n random cards' whole timelines (each version's picture beside its fields) for the by-eye
        check, under derived/chain/checks/.
No message text, posters, authors of Discord posts or links anywhere: dates, channels, picture ids and read fields only.
"""
from argparse import ArgumentParser
from collections import Counter, defaultdict
from datetime import datetime, timezone
from hashlib import sha1
from json import dumps, loads
from os import makedirs, remove
from os.path import exists, join
from random import Random
from typing import Callable, Optional

from commands import discord_match as dm
from commands.changelog import text_key
from commands.discord_verdicts import read_marks
from global_config import CARD_BRANCH, DATA_BRANCH, LOCAL_DIR_LOC

# ========================
# Configuration
# ========================
OUT = ("derived", "chain")
VERDICTS, LOST = ("derived", "match", "verdicts.jsonl"), ("derived", "match", "lost_groups.json")
TEXT_READS, TEXT_TRUTH = ("derived", "readers", "text", "reads.jsonl"), ("derived", "readers", "text", "truth.jsonl")
BAR_READS = ("derived", "readers", "bars", "reads.jsonl")
OLD_ROWS, OLD_REVIEW = ("derived", "oldrows", "rows.jsonl"), ("derived", "oldrows", "review.jsonl")
FACTS, FACT_SWITCHES = ("derived", "facts", "facts.jsonl"), ("derived", "facts", "switches.json")
CREATED_PATH = "site/Card Creation Dates.json"
STATS_FILE, OLD_STATS_FILE = "db/stats.json", "db/old_stats.json"
SURE_CLASSES = ("version_sure", "version_unsure", "draft")    # the card is sure (discord_verdicts)
ENTRY_CLASSES = ("version_sure", "draft")                    # ... and its look: a timeline version
FIELDS = ("name", "stars", "types", "stats", "ability")
STATS = ("hp", "def", "atk", "spd")
NO_BARS = {"field", "buff", "debuff"}                         # card kinds that print no stat bars
OFF_SCALE = 40                                                # analytics.OFF_SCALE: a total over this is not judged
MERGE_BITS = dm.LOOK_BITS                                     # drafts one look: whole-card hash this near ...
BLIND_BITS = 1.5                                              # ... this near when either has too little text read
NEAR_DAYS = 180                                               # a fact checked against a picture this near (facts' NEAR)
ITEM_KINDS = {"equip", "fragile equip", "active", "heal", "food", "support", "attack", "infliction", "counter",
              "negation", "deploy"}                           # an item's kind (its subtype): never a fact here
PAPER_LABEL = "paper card, before Discord"
ERA_RANK = {"A": 1, "B": 1, "C": 1, "A-item": 1, "B-item": 1, "C-item": 1, "D1": 2, "D2": 3, "D3": 3,
            "D-item": 3}                                      # template eras in time (A, B, C overlapped 2020-22)
REPOST_DAYS = 60                                              # D1 after D2 this long after D2's first: a repost
DECISIONS = (                                                 # STAGE1's questions, answered without the owner
    ("Q1", "C", "Meme/joke channels: meme-cards, joke-ideas, cursed and idea-graveyard; patty-vault and ai-images left "
                "out entirely."),
    ("Q2", "A", "Morgue candidates: only cards really played (port or update-port picture, update changelog line, "
                "card-classification, old repo file) plus the joke cards; drafts seen only in idea channels are "
                "'ideas never ported'."),
    ("Q3", "A", "Split cards: both halves' timelines start with the joint card's versions 'as <old name>', each with "
                "its own half's stats (to confirm here)."),
    ("Q4", "B", "Changelog lines: only field changes stated as data (e.g. 'ATK 5 -> 4') and a dated 'ability changed' "
                "mark without text; never the line's text."),
    ("Q5", "B", "First repo push notices: evidence only (no rows); text-only rows (A) prepared behind the switch "
                "first_repo_rows (derived/facts/firstrepo.jsonl)."),
    ("Q6", "A", "Paper cards: shown first, labelled 'paper card, before Discord', no day; never set Created."))


# ========================
# Helpers
# ========================
class Paths:
    def __init__(self, root: str):
        self.root, self.out = root, join(root, *OUT)
        self.timelines, self.checks = join(self.out, "timelines"), join(self.out, "checks")
        self.created, self.review = join(self.out, "created.json"), join(self.out, "review.jsonl")
        self.stats = join(self.out, "stats.txt")

    def at(self, parts: tuple) -> str:
        return join(self.root, *parts)


def review_id(kind: str, key: str) -> str:
    return "chain-" + sha1(f"{kind}|{key}".encode()).hexdigest()[:12]


def day(seconds: Optional[float]) -> Optional[str]:
    return None if seconds is None else datetime.fromtimestamp(seconds, timezone.utc).strftime("%Y-%m-%d")


def sure(value, method: str, conf=None, **extra) -> dict:
    out = {"value": value, "method": method, "confidence": "sure"}
    if conf is not None:
        out["conf"] = round(float(conf), 4)
    return {**out, **{k: v for k, v in extra.items() if v is not None}}


def unsure(method: str, why: str, proposed=None, conf=None) -> dict:
    """A field kept out of the timeline: its proposed value goes to the review item only."""
    out = {"method": method, "confidence": "review", "why": why}
    if conf is not None:
        out["conf"] = round(float(conf), 4)
    if proposed is not None:
        out["proposed"] = proposed
    return out


def is_sure(field: Optional[dict]) -> bool:
    return bool(field) and field.get("confidence") == "sure" and field.get("value") is not None


def stat_line(value) -> Optional[tuple]:
    """A stats value as the website's line (hp, def, atk, spd), None when it isn't one creature's four stats."""
    if not isinstance(value, dict) or any(not isinstance(value.get(k), int) for k in STATS):
        return None
    return tuple(value[k] for k in STATS)


def ability_change(a: dict, b: dict) -> str:
    """same | formatting | changed | unknown: two versions' abilities (formatting alone, owner 11-09; an OCR'd text
    against a PSD's: icons aside, they print as pictures)."""
    if not (is_sure(a) and is_sure(b)):
        return "unknown"
    x, y = a["value"] or "", b["value"] or ""
    if x == y:
        return "same"
    if text_key(x) == text_key(y):
        return "formatting"
    if "psd" not in (a["method"] + b["method"]) or "ocr" in (a["method"] + b["method"]) or "eye" in (a["method"] +
                                                                                                    b["method"]):
        if text_key(dm.strip_tokens(x)) == text_key(dm.strip_tokens(y)):
            return "formatting"
    return "changed"


def compare(before: dict, after: dict) -> dict:
    """What changed from one version's fields to the next: only fields sure on both sides; the rest are unknown.
    The buff/nerf verdict is the website's (build analytics.change): the stat total up or down, the same total
    'shift', a total over OFF_SCALE not judged."""
    out, unknown = {}, []
    for key in ("name", "stars", "types", "stats"):
        a, b = before.get(key), after.get(key)
        if not a and not b:
            continue
        if not (is_sure(a) and is_sure(b)):
            unknown.append(key)
            continue
        x, y = a["value"], b["value"]
        if key == "name" and dm.squash(x) != dm.squash(y):
            out["name"] = [x, y]
        elif key == "stars" and x != y:
            out["stars"] = [x, y]
        elif key == "types" and Counter(x or []) != Counter(y or []):
            out["types"] = {"add": sorted((Counter(y) - Counter(x)).elements()),
                            "drop": sorted((Counter(x) - Counter(y)).elements())}
        elif key == "stats" and x != y:
            out["stats"] = [x, y]
            la, lb = stat_line(x), stat_line(y)
            if la and lb and sum(la) <= OFF_SCALE and sum(lb) <= OFF_SCALE:
                out["bal"] = "buff" if sum(lb) > sum(la) else "nerf" if sum(lb) < sum(la) else "shift"
            moves = [(k.upper(), x.get(k), y.get(k)) for k in STATS if isinstance(x.get(k), int) and
                     isinstance(y.get(k), int) and x[k] != y[k]]
            out["buffs"] = [f"{k} {p} -> {q}" for k, p, q in moves if q > p]
            out["nerfs"] = [f"{k} {p} -> {q}" for k, p, q in moves if q < p]
    if before.get("ability") or after.get("ability"):
        out["ability_change"] = ability_change(before.get("ability"), after.get("ability"))
    if unknown:
        out["unknown"] = unknown
    return out


def settled(before: dict, after: dict) -> bool:
    """Every field either version has is sure on both and unchanged (the old rows' fold rule)."""
    keys = {k for k in FIELDS if before.get(k) or after.get(k)}
    if not keys or any(not (is_sure(before.get(k)) and is_sure(after.get(k))) for k in keys):
        return False
    change = compare(before, after)
    return not (set(change) - {"ability_change"}) and change.get("ability_change") in (None, "same", "formatting")


# ========================
# Inputs
# ========================
class Inputs:
    """The parts' outputs and the card repo's data, read once."""
    def __init__(self, root: str, cards_repo: str):
        p = Paths(root)
        pairs = dm.read_json(dm.Paths(root).pairs, {}) or {}
        if not pairs.get("cards"):
            raise SystemExit("no matcher pairs: run python -m commands.discord_match score first")
        self.cards = pairs["cards"]
        self.by_key = {c["key"]: cid for cid, c in self.cards.items()}
        self.versions = defaultdict(list)
        for v in pairs.get("versions") or []:
            self.versions[v["card"]].append(v)
        self.images = {}                         # sha256 -> the matcher's image (its OCR lines, cells)
        for image in (pairs.get("images") or {}).values():
            self.images.setdefault(image["sha256"], image)
        self.verdicts = dm.read_jsonl(p.at(VERDICTS))
        if not self.verdicts:
            raise SystemExit("no verdicts: run python -m commands.discord_verdicts build first")
        self.lost = (dm.read_json(p.at(LOST), {}) or {}).get("groups") or []
        self.text = {r["file"]: r for r in dm.read_jsonl(p.at(TEXT_READS)) if r.get("src") == "discord"}
        self.text_truth = {r["id"]: r["fields"] for r in dm.read_jsonl(p.at(TEXT_TRUTH)) if r.get("src") == "discord"}
        self.bars = {r["file"]: r for r in dm.read_jsonl(p.at(BAR_READS)) if r.get("src") == "discord"}
        self.old_rows = defaultdict(list)
        for row in dm.read_jsonl(p.at(OLD_ROWS)):
            self.old_rows[row["card"]].append(row)
        self.old_review = dm.read_jsonl(p.at(OLD_REVIEW))
        self.facts = dm.read_jsonl(p.at(FACTS))
        self.switches = dm.read_json(p.at(FACT_SWITCHES), {}) or {}
        git_dir = dm.git_dir_of(cards_repo)
        if not git_dir:
            raise SystemExit(f"no card repo at {cards_repo}")
        made = loads(dm.git_out(git_dir, "show", f"origin/{CARD_BRANCH}:{CREATED_PATH}") or b"{}")
        self.created_file = made.get("cards") or {}
        self.first_data = first_versions(git_dir)


def first_versions(git_dir: str) -> dict:
    """{card key: its first version in the card data (the oldest old_stats entry, else stats.json)}."""
    out = {}
    stats = loads(dm.git_out(git_dir, "show", f"origin/{DATA_BRANCH}:{STATS_FILE}") or b"{}")
    for key, entry in (stats.get("cards") or stats).items():
        if isinstance(entry, dict):
            out[key] = entry
    old = loads(dm.git_out(git_dir, "show", f"origin/{DATA_BRANCH}:{OLD_STATS_FILE}") or b"{}")
    for key, entries in (old.get("cards") or {}).items():
        if entries:
            out[key] = min(entries, key=lambda e: e.get("timestamp") or 0)
    return out


def data_fields(entry: dict) -> dict:
    """A card-data version's fields (its PSD's parse): stats only when all read (no -1, no STATS UNKNOWN problem),
    types only from parser 2 on (the website's legacy rule)."""
    out, problems = {}, " ".join(str(p) for p in entry.get("problems") or [])
    if entry.get("name"):
        out["name"] = sure(entry.get("display_name") or entry["name"], "psd (card data)")
    if isinstance(entry.get("stars"), int):
        out["stars"] = sure(entry["stars"], "psd (card data)")
    if entry.get("parser_version") is not None:
        out["types"] = sure(sorted(entry.get("types") or []), "psd (card data)")
    s = entry.get("stats") or {}
    line = {"hp": s.get("hp"), "def": s.get("defense"), "atk": s.get("attack"), "spd": s.get("speed")}
    if all(isinstance(v, int) and v >= 0 for v in line.values()) and "STATS UNKNOWN" not in problems \
            and entry.get("card_type") not in NO_BARS and not entry.get("creatures"):
        out["stats"] = sure(line, "psd (card data)")
    if entry.get("ability") is not None:
        out["ability"] = sure(entry["ability"], "psd (card data)")
    return out


# ========================
# Discord pictures
# ========================
def picture_lines(inputs: Inputs, row: dict) -> list:
    image = inputs.images.get(row["sha256"]) or {}
    if row.get("cell"):
        cells = image.get("cells") or []
        return (cells[row["cell"] - 1].get("lines") or []) if len(cells) >= row["cell"] else []
    return image.get("lines") or []


def picture_fields(inputs: Inputs, row: dict, kind: Optional[str]) -> dict:
    """A Discord picture's own reads: an eye truth first (the readers' by-eye files), else the readers' sure flags;
    a sheet's cell has no read of its own."""
    if row.get("cell"):
        return {k: unsure("none", "a sheet's cell: not read") for k in FIELDS if k != "stats" or kind not in NO_BARS}
    fields = {}
    text = inputs.text.get(row["file"]) or {}
    eye = inputs.text_truth.get("d:" + row["sha256"]) or {}
    for key, src in (("name", "title"), ("stars", "stars"), ("types", "types"), ("ability", "ability")):
        read = text.get(src) or {}
        if src in eye and eye[src] not in (None, ""):
            value = sorted(eye[src]) if key == "types" else eye[src]
            fields[key] = sure(value, "eye")
        elif read.get("sure") and read.get("value") not in (None, ""):
            fields[key] = sure(sorted(read["value"]) if key == "types" else read["value"], read.get("method") or "ocr",
                               read.get("conf"))
        elif text:
            why = "Discord types are never sure (icons outside Types/)" if key == "types" else "read under its bar"
            fields[key] = unsure(read.get("method") or "ocr", why, read.get("value"), read.get("conf"))
        else:
            fields[key] = unsure("none", "not read (no card era)")
    if kind not in NO_BARS:
        fields["stats"] = picture_stats(inputs.bars.get(row["file"]), kind)
    return fields


def picture_stats(read: Optional[dict], kind: Optional[str]) -> dict:
    """The era bar readers' stats: a truth (eye, psd) first, else every printed row sure; two creatures to review."""
    if not read:
        return unsure("none", "bars not read")
    truth = read.get("truth") or {}
    if truth.get("stats2") or read.get("two"):
        top = truth.get("stats") or {k: (v or {}).get("v") for k, v in (read.get("stats") or {}).items()}
        bottom = truth.get("stats2") or {k: (v or {}).get("v") for k, v in (read.get("stats2") or {}).items()}
        return unsure(truth.get("from") or "picture", "two creatures: each half to review",
                      {"creatures": [top, bottom]})
    if truth.get("stats"):
        return sure({k: v for k, v in truth["stats"].items() if v is not None}, truth.get("from") or "eye")
    stats = read.get("stats") or {}
    printed = {k: v for k, v in stats.items() if (v or {}).get("row") != "none"}
    values = {k: v.get("v") for k, v in printed.items() if v and v.get("v") is not None}
    if printed and all(v.get("sure") and v.get("v") is not None for v in printed.values()) and \
            (kind not in ("creature", "creatures", "minion") or set(printed) == set(STATS)):
        return sure(values, "picture", min(v.get("conf", 0) for v in printed.values()), era=read.get("era"))
    holds = sorted({v.get("hold") for v in printed.values() if v and v.get("hold")})
    return unsure("picture", "; ".join(holds) or "bars under their bar", values or None)


def old_fields(row: dict, review_of: dict) -> dict:
    """An old repo row's fields: sure values kept, the rest withheld with their review item ids."""
    out = {}
    for key, field in (row.get("fields") or {}).items():
        if field.get("confidence") == "sure":
            out[key] = sure(field.get("value"), field.get("method") or "psd", None, check=field.get("check"))
        else:
            out[key] = {"method": field.get("method") or "psd", "confidence": "review", "why": field.get("why"),
                        "review": review_of.get((row["id"], key))}
    return out


def differs(key: str, a, b) -> bool:
    """Two sure values of a field say different things: names by their letters, abilities by their words (icons
    aside: an OCR'd text has none), types as a set, stats on the rows both have."""
    if key == "name":
        return dm.squash(a) != dm.squash(b)
    if key == "ability":
        return text_key(dm.strip_tokens(a)) != text_key(dm.strip_tokens(b))
    if key == "types":
        return sorted(a or []) != sorted(b or [])
    if key == "stats" and isinstance(a, dict) and isinstance(b, dict):
        return any(a[k] != b[k] for k in set(a) & set(b)) or not set(a) & set(b)
    return a != b


def merge_fields(fields: dict, other: dict, how: str) -> dict:
    """fields filled from another picture of the same look: a sure value where this one has none; two sure values
    that differ make the field unsure."""
    out = dict(fields)
    for key, field in other.items():
        mine = out.get(key)
        if is_sure(field) and not is_sure(mine):
            out[key] = {**field, "method": f"{field['method']} ({how})"}
        elif is_sure(field) and is_sure(mine) and differs(key, field["value"], mine["value"]):
            out[key] = unsure(mine["method"], f"pictures of this look disagree ({how})", mine["value"])
    return out


def conflict(a: dict, b: dict) -> bool:
    """Two pictures' sure reads say different things (stats, stars, the ability's words, the title); stats also by
    the values read under their bar (two creatures' halves too): a bar change the hash can't see."""
    if any(is_sure(a.get(k)) and is_sure(b.get(k)) and differs(k, a[k]["value"], b[k]["value"])
           for k in ("name", "stars", "stats", "ability")):
        return True
    x, y = (f.get("stats") or {} for f in (a, b))
    x, y = (f.get("value") if is_sure(f) else f.get("proposed") for f in (x, y))
    if isinstance(x, dict) and isinstance(y, dict):
        if "creatures" in x or "creatures" in y:
            return x != y
        return any(isinstance(x.get(k), int) and isinstance(y.get(k), int) and x[k] != y[k] for k in STATS)
    return False


def same_look(a: dict, b: dict) -> bool:
    """Two drafts of a card show one look: whole cards within MERGE_BITS with the same timing tags and numbers read
    (within BLIND_BITS when either has too little text to tell), and no sure read differs."""
    if not a["row"].get("whole") or not b["row"].get("whole"):
        return False
    bits = dm.hash_distance(a["row"]["whole"], b["row"]["whole"])
    if a["marks"] is None or b["marks"] is None:
        near = bits <= BLIND_BITS
    else:
        near = bits <= MERGE_BITS and a["marks"] == b["marks"]
    return near and not conflict(a["fields"], b["fields"])


# ========================
# The chain
# ========================
class Chain:
    def __init__(self, inputs: Inputs):
        self.inputs = inputs
        self.review: dict = {}
        self.tally = Counter()
        self.review_of = {(item.get("row"), item.get("field")): item["id"] for item in inputs.old_review
                          if item.get("row") and item.get("field")}
        self.splits = {g["id"]: g for g in inputs.lost if g.get("split_of")}
        self.maybe = {}                          # card -> its earliest dated picture the matcher wasn't sure of
        for row in sorted((r for r in inputs.verdicts if r.get("first") and r["verdict"] == "unsure"),
                          key=lambda r: r["first"]["date"]):
            for cid in row.get("maybe") or []:
                self.maybe.setdefault(cid, row)

    def add_review(self, kind: str, key: str, **item) -> str:
        rid = review_id(kind, key)
        self.review.setdefault(rid, {"id": rid, "kind": kind, **{k: v for k, v in item.items() if v is not None}})
        return rid

    # Discord rows -> per card candidates
    def discord_rows(self) -> tuple:
        """({card: [picture, ...]} that may enter a timeline, {card: [split picture, ...]}), the rest to review."""
        cards, inputs = self.inputs.cards, self.inputs
        found, split = defaultdict(list), defaultdict(list)
        for row in inputs.verdicts:
            cid, first = row.get("card"), row.get("first")
            if row["verdict"] == "lost" and row.get("lost_group") in self.splits and first:
                group = self.splits[row["lost_group"]]
                for half in group["split_of"]["into"]:
                    split[half].append({**row, "as": group["split_of"]["name"]})
                continue
            if row["verdict"] not in SURE_CLASSES or cid not in cards:
                if row.get("paper"):
                    self.add_review("paper photo", row["id"], picture=row["id"], maybe=row.get("maybe"),
                                    why="a paper card's photo (Q6=A: shown undated once you name its card)")
                continue
            if row.get("paper"):                   # Q6=A: first, undated, never Created
                if row["verdict"] in ENTRY_CLASSES:
                    found[cid].append(row)
                else:
                    self.add_review("paper photo", row["id"], card=cid, picture=row["id"],
                                    why="a paper card's photo whose look isn't told")
                continue
            if not first:
                self.tally["discord: card sure, no dating post"] += 1
                continue
            when = dm.epoch(first["date"])
            if cards[cid].get("fs") and when >= cards[cid]["fs"]:
                self.tally["discord: after the card's first git record"] += 1
                continue
            if row.get("joke_draft") or (row.get("joke") and row["verdict"] != "version_sure"):
                self.add_review("joke picture", row["id"], card=cid, picture=row["id"], date=first["date"],
                                channel=first["channel"], why=row.get("review") or "a joke picture's look")
                self.tally["discord: joke picture (review)"] += 1
            elif row.get("created_review"):
                self.add_review("named only", row["id"], card=cid, picture=row["id"], date=first["date"],
                                channel=first["channel"], why=row["created_review"], verdict=row["verdict"],
                                section="confirm in bulk")
                self.tally["discord: named only before its first picture evidence (review)"] += 1
            elif row["verdict"] == "version_sure" and self.later_look(cid, row):
                self.tally["discord: a later git look posted before git (the git part has it)"] += 1
            elif row["verdict"] == "version_unsure":
                self.add_review("look not told", row["id"], card=cid, picture=row["id"], date=first["date"],
                                channel=first["channel"], why=(row.get("version") or {}).get("unsure") or
                                "which version it shows isn't sure", near=(row.get("version") or {}).get("near_look"))
                self.tally["discord: look not told (review)"] += 1
            else:
                found[cid].append(row)
        return found, split

    def later_look(self, cid: str, row: dict) -> bool:
        """A git look whose first git version is after the card's first git record: posted before git, but the
        git part of the chain has it (else it would show twice, around the card's older first version)."""
        look = (row.get("version") or {}).get("look")
        vs = [v["ts"] for v in self.inputs.versions.get(cid, []) if v.get("look") == look] if look is not None else []
        fs = self.inputs.cards[cid].get("fs")
        return bool(vs and fs) and min(vs) > fs + 1

    def git_look_fields(self, cid: str, row: dict) -> tuple:
        """(fields, git) of the git look a version_sure picture shows: its first git version's old repo row when the
        look is the old repo's (the exact version), else none; git = where that version is."""
        version = row.get("version") or {}
        look = version.get("look")
        vs = [v for v in self.inputs.versions.get(cid, []) if v.get("look") == look] if look is not None else []
        if not vs:
            return {}, None
        first = min(vs, key=lambda v: v["ts"])
        prefix = first["key"][1:21]
        git = {"source": "old repo" if first.get("old") else "git", "commit": first.get("commit"),
               "path": first.get("path"), "date": dm.iso(first["ts"])}
        key = self.inputs.cards[cid]["key"]
        old = sorted((r for r in self.inputs.old_rows.get(key, []) if r["blob"].startswith(prefix)),
                     key=lambda r: r["ts"])
        if old:
            git["row"] = old[0]["id"]
            return old_fields(old[0], self.review_of), git
        return {}, git

    def picture(self, cid: str, row: dict) -> dict:
        kind = self.inputs.cards[cid].get("type")
        fields = picture_fields(self.inputs, row, kind)
        git = None
        if row["verdict"] == "version_sure":
            exact, git = self.git_look_fields(cid, row)
            if exact:                          # the exact version's fields; one it left to review stays so
                own = fields
                fields = {k: {**v, "method": f"git look: {v['method']}"} if is_sure(v) else v for k, v in exact.items()}
                for key, mine in own.items():
                    if key in fields and not is_sure(fields[key]) and is_sure(mine):
                        fields[key] = unsure(fields[key].get("method") or "psd", "its git version's field is to "
                                             f"review ({fields[key].get('why')}); the picture reads it", mine["value"])
                fields = merge_fields(fields, {k: v for k, v in own.items() if is_sure(fields.get(k))}, "the picture")
        return {"row": row, "fields": fields, "git": git, "when": dm.epoch((row.get("first") or {}).get("date")),
                "marks": read_marks(picture_lines(self.inputs, row))}

    def looks(self, cid: str, rows: list) -> list:
        """Pictures grouped into looks: one git look each; drafts by same_look; oldest first."""
        groups = []
        pictures = sorted((self.picture(cid, r) for r in rows),
                          key=lambda p: (p["when"] is not None, p["when"] or 0, p["row"]["id"]))
        for pic in pictures:
            row = pic["row"]
            key = ("git", row["version"].get("look")) if row["verdict"] == "version_sure" else None
            home = next((g for g in groups if key and g["key"] == key and not row.get("paper") and not g["paper"]), None)
            if home is None and not key and not row.get("paper"):
                home = next((g for g in groups if g["key"] is None and not g["paper"] and g.get("as") == row.get("as")
                             and same_look(g["pictures"][0], pic)
                             and not conflict(g["fields"], pic["fields"])), None)
            if home is None:
                groups.append({"key": key, "pictures": [pic], "fields": dict(pic["fields"]), "git": pic["git"],
                               "paper": bool(row.get("paper")), "as": row.get("as")})
            else:
                home["pictures"].append(pic)
                home["fields"] = merge_fields(home["fields"], pic["fields"], "repost " + row["id"])
        return groups

    def split_stats(self, cid: str, group: dict) -> None:
        """A split card's picture (owner Q3=A): this half's own stats, to confirm (the halves as the old rows
        matched them)."""
        halves = {}
        for item in self.inputs.old_review:
            if (item.get("split") or {}).get("halves") and item.get("old_id") == group["as"]:
                halves = item["split"]["halves"]
        name = self.inputs.cards[cid]["key"]
        stats = group["fields"].get("stats") or {}
        creatures = (stats.get("proposed") or {}).get("creatures") if isinstance(stats.get("proposed"), dict) else None
        half = halves.get(name)
        value = creatures[0 if half == "top" else 1] if creatures and half in ("top", "bottom") else None
        group["fields"]["stats"] = unsure(stats.get("method") or "picture",
                                          f"{half or 'which'} half of {group['as']}'s rows (owner Q3=A, provisional)",
                                          value)

    def discord_entries(self, cid: str, rows: list, split: list) -> list:
        out, card = [], self.inputs.cards[cid]
        for group in self.looks(cid, rows) + self.looks(cid, split):
            first = group["pictures"][0]["row"]
            if group["as"]:
                self.split_stats(cid, group)
            entry = {"id": "D-" + first["id"], "kind": "paper" if group["paper"] else "discord"}
            if group["paper"]:
                entry.update(date=None, day=None, label=PAPER_LABEL)
            else:
                entry.update(date=dm.iso(group["pictures"][0]["when"]), day=day(group["pictures"][0]["when"]),
                             channel=first["first"]["channel"])
            if group["as"]:
                entry["as"] = group["as"]
            entry["look"] = {"git": {k: v for k, v in group["git"].items() if v is not None}} if group["git"] else \
                {"draft": True} if group["key"] is None else {"git": None}
            entry["pictures"] = [{"id": p["row"]["id"], "date": (p["row"].get("first") or {}).get("date"),
                                  "channel": (p["row"].get("first") or {}).get("channel"), "file": p["row"]["file"],
                                  "thumb": p["row"].get("thumb"), "cell": p["row"].get("cell"),
                                  "box": p["row"].get("box"), "verdict": p["row"]["verdict"]}
                                 for p in group["pictures"]]
            if any(p["row"].get("name_only") for p in group["pictures"]):
                entry["name_only"] = True
            entry["fields"] = {k: v for k, v in group["fields"].items() if v}
            if card.get("type") in NO_BARS:
                entry["fields"].pop("stats", None)
            era = next((self.era(p["row"]) for p in group["pictures"] if self.era(p["row"])), None)
            if era:
                entry["era"] = era
            out.append(entry)
        return self.reposted_eras(cid, out)

    def era(self, row: dict) -> Optional[str]:
        """A picture's template era (the era readers'): none for a sheet's cell."""
        if row.get("cell"):
            return None
        return (self.inputs.bars.get(row["file"]) or {}).get("era") or (self.inputs.text.get(row["file"]) or
                                                                       {}).get("era")

    def reposted_eras(self, cid: str, entries: list) -> list:
        """A look in an older template than a version posted before it (A, B, C before D1 before D2/D3) is an old
        look posted again: review, not a later version."""
        kept, top, since = [], 0, {}
        for e in sorted(entries, key=lambda e: (e["date"] is not None, e["date"] or "")):
            rank = ERA_RANK.get(e.get("era"))
            when = dm.epoch(e["date"]) if e.get("date") else None
            if rank and when and (rank < top) and (rank < 2 or when - since.get(top, when) > REPOST_DAYS * 86400):
                self.add_review("older template later", f"{cid}|{e['id']}", card=cid, entry=e["id"],
                                pictures=[p["id"] for p in e["pictures"]], date=e["date"], era=e.get("era"),
                                why="an older template than a version posted before it: an earlier look posted "
                                    "again? (kept out of the timeline)")
                self.tally["discord: an older template posted later (review)"] += 1
                continue
            if rank and when:
                since.setdefault(rank, when)
                top = max(top, rank)
            kept.append(e)
        return kept

    def old_entries(self, cid: str) -> list:
        key, out = self.inputs.cards[cid]["key"], []
        rows = list(self.inputs.old_rows.get(key, []))
        for item in self.inputs.old_review:
            split = (item.get("split") or {}).get("rows") or {}
            rows += split.get(key, [])
        for row in sorted(rows, key=lambda r: (r["ts"], r["id"])):
            entry = {"id": "O-" + row["id"], "kind": "old repo", "date": dm.iso(row["ts"]), "day": day(row["ts"]),
                     "badge": "old repo", "author": row.get("author"), "subject": row.get("subject"),
                     "commit": row.get("commit"), "path": row.get("path"), "blob": row.get("blob"),
                     "thumb": row.get("thumb"), "minor": bool(row.get("minor")), "share": row.get("share"),
                     "old_ability_change": row.get("ability_change"), "fields": old_fields(row, self.review_of)}
            if row.get("split_from"):
                entry["as"] = row["split_from"]
            for k in ("old_name", "old_file", "in_tc_init", "two_creatures"):
                if row.get(k):
                    entry[k] = row[k]
            out.append(entry)
        return out

    def text_entries(self, cid: str) -> list:
        """Sure changelog facts before the stop (owner Q4=B: data only, no text); chat and push lines are evidence."""
        card, out = self.inputs.cards[cid], []
        stop = card.get("fs") or float("inf")
        for fact in self.inputs.facts:
            if fact.get("source") != "changelog" or fact.get("card") != card["key"]:
                continue
            when = dm.epoch(fact.get("date"))
            if fact.get("confidence") != "sure":
                continue
            if when is None or when >= stop:
                self.tally["facts: sure, after the card's first git record"] += 1
                continue
            if fact.get("field") == "type" and card.get("type") in ("item", "aux item") and \
                    str(fact.get("to")).lower() in ITEM_KINDS:
                self.tally["facts: an item's kind (never collected)"] += 1
                continue
            change = {"ability": "changed"} if fact.get("kind") == "ability changed" else \
                {k: fact[k] for k in ("field", "from", "to", "by", "op", "unit", "stated") if fact.get(k) is not None}
            entry = {"id": "F-" + fact["id"], "kind": "text", "date": dm.iso(when), "day": day(when),
                     "channel": fact.get("channel"), "source": "changelog", "change": change, "confidence": "sure"}
            if fact.get("repeat_of"):
                entry["repeat_of"] = "F-" + fact["repeat_of"]
            out.append(entry)
        return out

    def check_facts(self, cid: str, entries: list) -> list:
        """A stat or stars fact whose next picture (within NEAR_DAYS, before the next fact on that field) reads
        another value goes to review; one it agrees with is confirmed by it."""
        kept = []
        versions = [e for e in entries if e["kind"] in ("discord", "old repo") and e.get("date")]
        for e in entries:
            if e["kind"] != "text" or e["change"].get("field") not in (*STATS, "stars") or "to" not in e["change"]:
                kept.append(e)
                continue
            field, to, when = e["change"]["field"], e["change"]["to"], dm.epoch(e["date"])
            later = [f for f in entries if f["kind"] == "text" and f["change"].get("field") == field
                     and dm.epoch(f["date"]) > when]
            until = min([dm.epoch(f["date"]) for f in later] + [when + NEAR_DAYS * 86400])
            seen = None
            for v in versions:
                t = dm.epoch(v["date"])
                if when <= t < until:
                    got = v["fields"].get("stats" if field in STATS else "stars")
                    if is_sure(got):
                        seen = (v, got["value"].get(field) if field in STATS else got["value"])
                        if seen[1] is not None:
                            break
            if seen and seen[1] is not None and seen[1] != to:
                self.add_review("fact vs picture", e["id"], card=cid, fact=e["id"][2:], entry=seen[0]["id"],
                                field=field, fact_to=to, picture_reads=seen[1], date=e["date"],
                                why="the next picture reads another value")
                self.tally["facts: a picture reads another value (review)"] += 1
                continue
            if seen and seen[1] == to:
                e["confirmed_by"] = seen[0]["id"]
            kept.append(e)
        return kept

    def diff(self, entries: list) -> None:
        """Each version against the one before (text entries aside); Discord looks fold when settled."""
        before = None
        for e in entries:
            if e["kind"] == "text":
                continue
            if before is None:
                e["changes"] = {"ability_change": "first"} if e["fields"].get("ability") else {}
            else:
                e["changes"] = compare(before["fields"], e["fields"])
                if e["kind"] == "old repo" and before["kind"] == "old repo" and e.get("old_ability_change"):
                    e["changes"]["ability_change"] = e["old_ability_change"]
            if e["kind"] in ("discord", "paper"):
                e["minor"] = bool(before) and before["kind"] in ("discord", "paper") and settled(before["fields"],
                                                                                                e["fields"])
            before = e
        for e in entries:
            e.pop("old_ability_change", None)

    def field_reviews(self, cid: str, entries: list) -> None:
        for e in entries:
            if e["kind"] not in ("discord", "paper"):
                continue
            for key, field in e["fields"].items():
                if field.get("confidence") == "review":
                    if field.get("proposed") is None and key == "types":
                        continue
                    field["review"] = self.add_review(
                        "field", f"{cid}|{e['id']}|{key}", card=cid, entry=e["id"], field=key, date=e.get("date"),
                        proposed=field.pop("proposed", None), method=field.get("method"), conf=field.get("conf"),
                        why=field.get("why"))
                    self.tally[f"discord field to review: {key}"] += 1

    def card(self, cid: str, rows: list, split: list) -> dict:
        card = self.inputs.cards[cid]
        entries = self.discord_entries(cid, rows, split) + self.text_entries(cid) + self.old_entries(cid)
        entries.sort(key=lambda e: (e["date"] is not None, dm.epoch(e["date"]) or 0,
                                    {"paper": 0, "discord": 1, "text": 2, "old repo": 3}[e["kind"]], e["id"]))
        entries = self.check_facts(cid, entries)
        self.diff(entries)
        self.field_reviews(cid, entries)
        stop = None
        if card.get("fs"):
            stop = {"date": dm.iso(card["fs"]), "day": day(card["fs"]),
                    "source": "old repo" if card.get("fs_from") == "old git" else "git"}
            firsts = [v for v in self.inputs.versions.get(cid, []) if abs(v["ts"] - card["fs"]) < 1]
            if firsts:
                stop.update(commit=firsts[0].get("commit"), path=firsts[0].get("path"))
        out = {"card": cid, "name": card.get("name"), "key": card["key"], "kind": card.get("type"), "stop": stop,
               "entries": entries}
        data = self.inputs.first_data.get(card["key"])
        if stop and stop["source"] == "git" and data:      # the card repo's first version, for the last diff
            fields = data_fields(data)
            last = next((e for e in reversed(entries) if e["kind"] != "text"), None)
            out["git_first"] = {"from": "card data (its first recorded version)",
                                "date": dm.iso(data.get("timestamp")) if data.get("timestamp") else None,
                                "fields": fields, "changes": compare(last["fields"], fields) if last else {}}
        return out

    # Created
    def created(self, cid: str, timeline: dict, rows_all: list) -> dict:
        """Created = the earliest confident appearance with its source; against the file."""
        card, inputs = self.inputs.cards[cid], self.inputs
        found = []
        for e in timeline["entries"]:
            if e["kind"] == "discord":
                found.append((dm.epoch(e["date"]), "discord", {"picture": e["pictures"][0]["id"],
                                                               "channel": e["channel"], "as": e.get("as")}))
            elif e["kind"] == "old repo":
                found.append((dm.epoch(e["date"]), "old repo", {"commit": e["commit"], "as": e.get("as")}))
        for row in rows_all:                       # every card-sure picture counts, its look told or not
            if row.get("first") and not row.get("created_review") and not row.get("paper") and \
                    not row.get("joke_draft") and not (row.get("joke") and row["verdict"] != "version_sure"):
                found.append((dm.epoch(row["first"]["date"]), "discord", {"picture": row["id"],
                                                                          "channel": row["first"]["channel"]}))
        if card.get("fs"):
            found.append((card["fs"], "old repo" if card.get("fs_from") == "old git" else "git", {}))
        when, source, detail = min(found, key=lambda f: (f[0], f[1]))
        ours = {"date": dm.iso(when), "day": day(when), "source": source,
                **{k: v for k, v in detail.items() if v is not None}}
        file = inputs.created_file.get(card["key"])
        out = {"ours": ours, "file": file}
        if file and file.get("from") == "hand":
            out.update(status="hand", date=file["date"], proposed=False)
        elif not file or not file.get("date"):
            out.update(status="new", date=ours["day"], proposed=True)
        elif ours["day"] == file["date"]:
            out.update(status="same", date=file["date"], proposed=False)
        elif ours["day"] < file["date"]:
            out.update(status="earlier", date=ours["day"], proposed=True)
        else:
            out.update(status="later", date=file["date"], proposed=False)
        if out["status"] in ("earlier", "later", "new"):
            hint = self.maybe.get(cid)
            hint = {"picture": hint["id"], "date": hint["first"]["date"], "channel": hint["first"]["channel"]} \
                if hint and dm.epoch(hint["first"]["date"]) < when else None
            out["review"] = self.add_review(
                f"created {out['status']}", cid, card=cid, ours=ours, file=file, earlier_unsure=hint,
                why={"earlier": "an earlier confident appearance: proposed (the preview uses it)",
                     "later": "our earliest is later than the file's date: the file's kept (never moved later)",
                     "new": "no date in the file yet: proposed"}[out["status"]])
        named = [r for r in rows_all if r.get("created_review") and r.get("first")]
        if named:
            row = min(named, key=lambda r: dm.epoch(r["first"]["date"]))
            if day(dm.epoch(row["first"]["date"])) < ours["day"]:
                out["named_only_earlier"] = {"day": day(dm.epoch(row["first"]["date"])), "picture": row["id"],
                                             "review": review_id("named only", row["id"])}
        facts = [f for f in inputs.facts if f.get("card") == card["key"] and f.get("source") == "changelog"
                 and f.get("confidence") == "sure" and dm.epoch(f.get("date"))]
        if facts:
            fact = min(facts, key=lambda f: dm.epoch(f["date"]))
            if day(dm.epoch(fact["date"])) < ours["day"]:
                out["changelog_earlier"] = {"day": day(dm.epoch(fact["date"])), "fact": fact["id"],
                                            "review": self.add_review("changelog earlier", cid, card=cid,
                                                                      fact=fact["id"], date=fact["date"],
                                                                      why="a sure changelog line names it before "
                                                                          "its Created date")}
        return out


def build(root: str, cards_repo: str = LOCAL_DIR_LOC, log: Callable = print) -> dict:
    root = dm.check_root(root)
    paths = Paths(root)
    inputs = Inputs(root, cards_repo)
    chain = Chain(inputs)
    found, split = chain.discord_rows()
    by_card = defaultdict(list)
    for row in inputs.verdicts:
        if row.get("card") in inputs.cards and row["verdict"] in SURE_CLASSES:
            by_card[row["card"]].append(row)
    makedirs(paths.timelines, exist_ok=True)
    for name in [n for n in _listdir(paths.timelines) if n.endswith(".json")]:
        remove(join(paths.timelines, name))
    created, kinds, timelines = {}, Counter(), {}
    for cid in sorted(inputs.cards):
        timeline = chain.card(cid, found.get(cid, []), split.get(cid, []))
        timeline["created"] = chain.created(cid, timeline, by_card.get(cid, []))
        created[inputs.cards[cid]["key"]] = {"card": cid, **timeline["created"]}
        timelines[cid] = timeline
        dm.write_json(join(paths.timelines, f"{cid}.json"), timeline)
        for e in timeline["entries"]:
            kinds[e["kind"]] += 1
    for group in chain.splits.values():
        halves = group["split_of"]["into"]
        chain.add_review("split", group["id"], name=group["split_of"]["name"], halves=halves, group=group["id"],
                         entries={h: sum(1 for e in timelines[h]["entries"] if e.get("as")) for h in halves
                                  if h in timelines},
                         why="an old card that became two (owner Q3=A, provisional): both halves start with its "
                             "versions, each with its own half's stats")
    for q, answer, text in DECISIONS:
        chain.add_review("decided without you", q, question=q, answer=answer, text=text,
                         why="decided without you: change it here")
    first_repo = dm.read_jsonl(join(root, "derived", "facts", "firstrepo.jsonl"))
    chain.add_review("first repo rows", "Q5", rows=len(first_repo), cards=len({r.get("card") for r in first_repo}),
                     shown=bool(inputs.switches.get("first_repo_rows")), file="derived/facts/firstrepo.jsonl",
                     why="Q5=A prepared: text-only rows of the first repo's push notices, off until you choose A")
    dm.write_json(paths.created, {
        "about": "Created dates (STAGE1 S6): ours = the earliest confident appearance (a card-sure Discord picture, the "
                 "old repo, the card repo) with its source; status against site/Card Creation Dates.json: same, "
                 "earlier (proposed), later (review only, the file's date kept), new (proposed), hand (untouched). "
                 "date = what the preview uses. Named-only pictures and changelog lines before it are review only.",
        "generated": dm.now_iso(), "cards": created})
    items = sorted(chain.review.values(), key=lambda i: (i["kind"], i["id"]))
    dm.write_text(paths.review, "".join(dumps(i, ensure_ascii=False) + "\n" for i in items))
    text = stats_text(timelines, created, items, chain.tally, kinds)
    dm.write_text(paths.stats, text)
    log(text)
    return {"timelines": len(timelines), "review": len(items), "entries": dict(kinds)}


def _listdir(path: str) -> list:
    from os import listdir
    return listdir(path) if exists(path) else []


def stats_text(timelines: dict, created: dict, items: list, tally: Counter, kinds: Counter) -> str:
    lines = [f"CARD TIMELINES (S5 + S6), built {dm.now_iso()}", ""]
    with_discord = [t for t in timelines.values() if any(e["kind"] == "discord" for e in t["entries"])]
    lines += [f"cards {len(timelines)}; with a Discord version {len(with_discord)}; with any entry "
              f"{sum(1 for t in timelines.values() if t['entries'])}",
              "entries: " + ", ".join(f"{k} {n}" for k, n in sorted(kinds.items()))]
    discord = [e for t in timelines.values() for e in t["entries"] if e["kind"] == "discord"]
    lines.append(f"discord versions {len(discord)}: git looks {sum(1 for e in discord if e['look'].get('git'))}, "
                 f"drafts {sum(1 for e in discord if e['look'].get('draft'))}, split 'as' "
                 f"{sum(1 for e in discord if e.get('as'))}, pictures {sum(len(e['pictures']) for e in discord)}, "
                 f"minor {sum(1 for e in discord if e.get('minor'))}")
    for key in FIELDS:
        have = [e["fields"][key] for e in discord if key in e["fields"]]
        lines.append(f"  {key}: sure {sum(1 for f in have if is_sure(f))} of {len(have)}"
                     + "".join(f", {m} {n}" for m, n in Counter(f["method"] for f in have if is_sure(f)).most_common(6)))
    olds = [e for t in timelines.values() for e in t["entries"] if e["kind"] == "old repo"]
    lines.append(f"old repo versions {len(olds)} (minor {sum(1 for e in olds if e['minor'])}, split 'as' "
                 f"{sum(1 for e in olds if e.get('as'))})")
    texts = [e for t in timelines.values() for e in t["entries"] if e["kind"] == "text"]
    lines.append(f"text entries {len(texts)}: " + ", ".join(
        f"{k} {n}" for k, n in Counter(e["change"].get("field") or "ability" for e in texts).most_common()) +
        f"; confirmed by a picture {sum(1 for e in texts if e.get('confirmed_by'))}")
    bal = Counter(e["changes"].get("bal") for t in timelines.values() for e in t["entries"] if e.get("changes"))
    lines.append("buff/nerf (any version vs the one before): " + ", ".join(f"{k} {n}" for k, n in bal.items() if k))
    ab = Counter(e["changes"].get("ability_change") for t in timelines.values() for e in t["entries"]
                 if e.get("changes") and e["kind"] in ("discord", "paper"))
    lines.append("discord ability changes: " + ", ".join(f"{k} {n}" for k, n in ab.most_common()))
    status = Counter(c["status"] for c in created.values())
    sources = Counter(c["ours"]["source"] for c in created.values())
    lines += ["", "CREATED: " + ", ".join(f"{k} {n}" for k, n in status.most_common()),
              "  ours from: " + ", ".join(f"{k} {n}" for k, n in sources.most_common()),
              f"  named-only picture earlier (review): {sum(1 for c in created.values() if c.get('named_only_earlier'))}",
              f"  changelog line earlier (review): {sum(1 for c in created.values() if c.get('changelog_earlier'))}"]
    lines += ["", "REVIEW ITEMS: " + ", ".join(f"{k} {n}" for k, n in Counter(i["kind"] for i in items).most_common())]
    lines += ["", "TALLY"] + [f"  {k}: {n}" for k, n in sorted(tally.items())]
    return "\n".join(lines) + "\n"


# ========================
# Checks (by eye)
# ========================
def checks(root: str, n: int = 20, seed: int = 1113, log: Callable = print) -> list:
    """Sheets of n random cards (with a Discord version) whose whole timelines are drawn: each version's picture
    beside its fields (sure values; '?' to review)."""
    root = dm.check_root(root)
    paths = Paths(root)
    names = sorted(name for name in _listdir(paths.timelines) if name.endswith(".json"))
    pool = []
    for name in names:
        t = dm.read_json(join(paths.timelines, name))
        if any(e["kind"] == "discord" for e in t["entries"]):
            pool.append(t)
    picks = Random(seed).sample(pool, min(n, len(pool)))
    makedirs(paths.checks, exist_ok=True)
    out = []
    for t in picks:
        path = join(paths.checks, f"{t['card']}.jpg")
        card_sheet(t, root, path)
        out.append(path)
        log(path)
    dm.write_json(join(paths.checks, "picks.json"), [t["card"] for t in picks])
    return out


def show(field: Optional[dict]) -> str:
    if not field:
        return "-"
    if not is_sure(field):
        return "?"
    v = field["value"]
    if isinstance(v, dict):
        return "/".join(str(v.get(k, "-")) for k in STATS)
    if isinstance(v, list):
        return ",".join(v) or "(none)"
    return str(v)


def entry_image(e: dict, root: str):
    if e["kind"] == "old repo":
        try:
            return dm.load_image(join(root, "derived", e["thumb"]))
        except OSError:
            return None
    pic = e["pictures"][0]
    try:
        img = dm.load_image(join(root, pic["file"]))
    except OSError:
        return None
    if img is not None and pic.get("box"):
        x0, y0, x1, y1 = pic["box"]
        w, h = img.size
        img = img.crop((int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)))
    return img


def card_sheet(t: dict, root: str, path: str, side: int = 420):
    from PIL import Image, ImageDraw
    from commands.discord_verdicts import _font
    versions = [e for e in t["entries"] if e["kind"] != "text"]
    texts = [e for e in t["entries"] if e["kind"] == "text"]
    cols, tile_w, tile_h = 2, side * 2, side * 2 + 20
    rows = (len(versions) + cols - 1) // cols
    head = 60 + 18 * len(texts)
    sheet = Image.new("RGB", (cols * tile_w, head + rows * tile_h), "white")
    draw, font, small = ImageDraw.Draw(sheet), _font(16), _font(13)
    draw.text((8, 6), f"{t['name']} ({t['card']}) stop {t['stop'] and t['stop']['day']} "
                      f"{t['stop'] and t['stop']['source']}  created {t['created']['date']} "
                      f"[{t['created']['status']}; ours {t['created']['ours']['day']} {t['created']['ours']['source']}]",
              fill="black", font=font)
    for i, e in enumerate(texts):
        draw.text((8, 32 + 18 * i), f"TEXT {e['day']} {e['channel']}: {dumps(e['change'])}"
                                    + (f" confirmed by {e['confirmed_by']}" if e.get('confirmed_by') else ""),
                  fill="navy", font=small)
    for i, e in enumerate(versions):
        x, y = (i % cols) * tile_w, head + (i // cols) * tile_h
        img = entry_image(e, root)
        if img is not None:
            img = img.convert("RGB")
            img.thumbnail((side, side * 2))
            sheet.paste(img, (x + 4, y + 4))
        lines = [f"#{i + 1} {e['kind']} {e.get('day')} {e.get('channel') or e.get('subject') or ''}"[:60],
                 e["id"][:58], f"as: {e['as']}" if e.get("as") else "",
                 "look: " + (("git " + (e["look"]["git"] or {}).get("source", "?")) if e.get("look", {}).get("git")
                             else "draft" if e.get("look", {}).get("draft") else "-") if e["kind"] != "old repo" else
                 f"minor {e['minor']}", f"pictures {len(e.get('pictures') or [])}" if e.get("pictures") else ""]
        for key in FIELDS:
            f = e["fields"].get(key)
            lines.append(f"{key}: {show(f)}  [{(f or {}).get('method', '')}]")
        ch = e.get("changes") or {}
        lines.append("changes: " + dumps({k: v for k, v in ch.items() if k != "unknown"})[:120])
        if ch.get("unknown"):
            lines.append("unknown: " + ",".join(ch["unknown"]))
        ty = y + 8
        for line in lines:
            for part in wrap(line, 52):
                draw.text((x + side + 12, ty), part, fill="black", font=small)
                ty += 16
    sheet.save(path, quality=80)


def wrap(text: str, width: int) -> list:
    out, text = [], " / ".join(part.strip() for part in str(text).splitlines() if part.strip())
    while len(text) > width:
        out.append(text[:width])
        text = text[width:]
    return out + ([text] if text else [])


def main(argv: Optional[list] = None) -> int:
    parser = ArgumentParser(prog="python -m commands.discord_chain", description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=("build", "checks"))
    parser.add_argument("--root", default=dm.DEFAULT_ROOT)
    parser.add_argument("--cards-repo", default=LOCAL_DIR_LOC)
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1113)
    args = parser.parse_args(argv)
    if args.command == "build":
        build(args.root, args.cards_repo)
    else:
        checks(args.root, args.n, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
