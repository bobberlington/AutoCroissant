"""
Card timelines: each live card's chain of versions before git and its Created date (docs/Future/CARD_TIMELINE.md, steps
4 and 5; STAGE1 S5 + S6). Reads the stage-2 parts' outputs under the archive's derived/ (never a repo) and the card
repo (read only):

    python -m commands.discord_chain build  [--root <dir>] [--cards-repo <dir>]
    python -m commands.discord_chain checks [--root <dir>] [--n 20] [--seed 1113]

build   Under derived/chain/:
        timelines/<card id>.json  every live card: its entries oldest first, then the stop (its first git record:
                        the old repo's, else the card repo's) and its Created date. Entries (the physical card
                        photos never: owner Q6, 11-14, discord_match.LEFT_OUT):
                          discord     one look posted before the card's first git record: a git look (version_sure) or
                                      a look not in git (draft); its reposts merged (looks: hash, counted numbers,
                                      words, class line, no local edit; the earliest post dates
                                      it); a split card's (owner Q3) "as <old name>" on both halves
                          text        a sure changelog fact before the stop (owner Q4: a field change as data, or
                                      "ability changed" with no text); item kinds never (hand-crafted metadata)
                          old repo    the old pre-wipe repo's PNG versions (derived/oldrows), with their fold; the
                                      first repo's too (owner Q5, 11-14: picture rows badged 'first repo')
                        Each version's fields keep their method and confidence; only sure values are kept, the rest
                        name their review item (types with nothing read: 'unread'; an item card's kinds read alone,
                        sure or not: 'not collected', no item; its other types kept or to review without them; an
                        icon-read star change nothing backs: review, star_guard). Each is diffed with the version
                        before (sure fields on both sides): stats with the website's buff/nerf rule (analytics.change:
                        the total up or down, a total over OFF_SCALE not judged), stars, name, types, the ability
                        (text_key; formatting alone is no change, owner 11-09); types as the card data's tokens
                        (type_tokens: an answer's 'defensive minion', a minion's 2023 'protective' shield or 'equip'
                        sword, 'infection': owner 11-17 'infected'); an item's kinds are no types change (never
                        collected). A title read that is another live card's name goes to review (other_name). A
                        Discord look folds as minor only when every field is sure on both sides and unchanged and its
                        picture is near the one before (the same git look, or under png_changes.VISIBLE of the card
                        changed: png_changes' own rule for two PNGs of one size, as the old rows fold, else aligned:
                        look_share; owner 11-09).
        created.json    Created = the earliest confident appearance (a card-sure Discord picture, the first repo, the
                        old repo, the card repo; "Before 2020" from a physical card photo for PHOTO_CREATED only, the
                        owner's ruling of 11-14 22:20) with its source, against
                        site/Card Creation Dates.json: same, earlier (proposed), later (review only: never moved later),
                        new (no date yet), hand (never touched). sort orders "Before 2020" first. A changelog line or a
                        first-repo push notice is a hint to review only when earlier than the Created date in use (the
                        file's for 'later', the owner's pick when answered).
        review.jsonl    what the chain keeps out (stable ids): looks not told, named-only pictures before the card's
                        first picture evidence (one section to confirm in bulk), joke drafts, unsure fields, facts a
                        picture contradicts, Created changes, splits, and the owner's answers
                        (Q1-Q6, read only); pictures that are no card on its own (read only, below).
        Only a card on its own is shown (owner 11-16 23:10/23:15): a picture the eye found is not one (a cut-off piece,
        a screenshot of several cards or of a window: derived/match/eye.json dates_only) enters no timeline and asks
        nothing, but still counts for dates (Created) as any card-sure picture does, and for the other live cards
        printed plainly on it (eye.json dates_also: {picture: {cards, why}}; ours.beside names its own card); a
        read-only 'dates only' item shows it. The whole card cut out of a bigger picture (eye.json crops: {picture:
        {box, why}}, the box a share of the trimmed picture, as a sheet's cell) is shown as that crop; a read-only
        'cropped' item shows it.
        stats.txt       counts.
        The owner's answers on the review page (derived/review/rulings.json: commands.timeline_review, applied by
        scripts/apply_rulings.py) are honoured: pictures named, told (a git look by its first version's key) or
        refused, fields given (method 'owner'; types in the chain's tokens), unsure changelog facts and old maps, older
        templates shown, facts against pictures, Created picked (status 'owner', owner.source shaped like ours: where
        the date came from).
        ruled.json      the answers this build acted on (only while rulings.json has answers).
checks  Contact sheets of n random cards' whole timelines (each version's picture beside its fields) for the by-eye
        check, under derived/chain/checks/.
No message text, posters, authors of Discord posts or links anywhere: dates, channels, picture ids and read fields only.
"""
from argparse import ArgumentParser
from collections import Counter, defaultdict
from datetime import datetime, timezone
from difflib import SequenceMatcher
from functools import lru_cache
from hashlib import sha1
from json import dumps, loads
from os import makedirs, remove
from os.path import exists, join
from random import Random
from re import findall, fullmatch, search, sub
from typing import Callable, Optional

from commands import discord_match as dm
from commands.changelog import text_key
from commands.discord_verdicts import NUMBER, TAG, read_marks
from global_config import CARD_BRANCH, DATA_BRANCH, LOCAL_DIR_LOC

# ========================
# Configuration
# ========================
OUT = ("derived", "chain")
VERDICTS, LOST = ("derived", "match", "verdicts.jsonl"), ("derived", "match", "lost_groups.json")
TEXT_READS, TEXT_TRUTH = ("derived", "readers", "text", "reads.jsonl"), ("derived", "readers", "text", "truth.jsonl")
TEXT_OCR = ("derived", "readers", "text", "ocr.jsonl")      # the text readers' OCR lines with their boxes
BAR_READS = ("derived", "readers", "bars", "reads.jsonl")
OLD_ROWS, OLD_REVIEW = ("derived", "oldrows", "rows.jsonl"), ("derived", "oldrows", "review.jsonl")
FACTS = ("derived", "facts", "facts.jsonl")
FIRST_REPO = ("derived", "facts", "firstrepo.jsonl")     # push notices naming one card: Created hints only
EYE_LINES = ("derived", "facts", "eye_lines.json")        # heads: 2019 list heads whose first picture keeps the name
PSDS = ("derived", "psds", "psds.jsonl")                    # card PSDs posted on Discord (commands/discord_psds.py)
HEADER = ("derived", "readers", "types", "header.jsonl")    # types by the git versions' header (header_types.py)
HEADER_MISS = 0.6                                           # a header this unlike every git version's: other icons
HEADER_EYE = ("derived", "readers", "types", "header_eye.jsonl")   # eras A-C: types by eye references' headers
MORGUE = ("derived", "morgue", "candidates.json")          # the Morgue's eye: pictures it took from a live card
PSD_BITS = 3.0                                              # a look this near a posted PSD's composite shows it
RULINGS = ("derived", "review", "rulings.json")             # the owner's answers (scripts/apply_rulings.py)
ISSUED = ("derived", "review", "ids.json")                  # every review id the page showed, with its card
PICTURE_KINDS = ("named only", "look not told", "joke picture")   # review ids keyed by their picture
OWNER = ("derived", "review", "owner.json")                 # the owner's rulings outside the review page (answers)
PSD_FIRST = ("derived", "oldrows", "psd_first.json")        # cards first recorded as a PSD in the old repo (S6)
MATCH_EYE = ("derived", "match", "eye.json")                # dates_only {picture: why}, dates_also {picture: {cards,
                                                            # why}}, crops {picture: {box, why}}
FS_SOURCE = {"old git": "old repo", "first git": "first repo"}   # a card's first git record: where (else the card repo)
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
SAME_WORDS, ABILITY_WORDS = 0.92, 4                           # ... ability words this alike (Jaccard), this many read
NEAR_WORD, LABEL_WORD = 0.75, 0.6                             # an OCR'd word's near twin; a stat label misread
STAT_LABELS = ("health", "defense", "defence", "attack", "speed", "armor")
WORDS = "/usr/share/dict/words"                               # real words: OCR noise aside
CLASS_Y, CLASS_MID, CLASS_WORDS = 0.96, 0.13, 5               # a class line: middle under this, centred, words
                                                              # (timeline_text.class_like)
LOCAL_W, LOCAL_CELL = 240, 12                                 # drafts compared in grey at this width, by these cells
LOCAL_EDIT, LOCAL_RATIO, LOCAL_FLOOR = 15, 10, 1.5           # a local edit: a cell this far apart (grey levels) and
                                                              # this many times the pair's 90th percentile cell (floored)
FOLD_W, FOLD_BLUR, FOLD_TOL, FOLD_PATCH = 160, 1.0, 32, 3     # a Discord look's changed share: both this wide, aligned,
                                                              # blurred; a pixel off by more than this (of 255) inside a
                                                              # square this big (JPEG noise, resizing, a few px of crop
                                                              # aside; calibration 11-16)
FOLD_ASPECT, FOLD_MARGIN, FOLD_PICTURES = 0.08, 0.02, 3       # shapes this far apart: not compared; border left out;
                                                              # each look's first pictures compared
BACKED = ("eye", "psd", "git look", "owner", "triage")        # star reads no icon reader made
APART_DAYS = 30                                               # two icon reads this far apart: read separately
FACT_GRACE_DAYS = 3                                           # a picture this near before a changelog line: its port
NEAR_DAYS = 180                                               # a fact checked against a picture this near (facts' NEAR)
ITEM_KINDS = {"equip", "fragile equip", "active", "heal", "food", "support", "attack", "infliction", "counter",
              "negation", "deploy", "offense", "protective"}  # an item's kind (its subtype; the last two the old
                                                              # repo's): never a fact or a version's change here
ITEM_CARDS = ("item", "aux item", "nme")                      # cards whose types are item kinds
TYPE_FORMS = {"defensive minion": "defensive_minion", "offensive minion": "offensive_minion",
              "fragile_equip": "fragile equip", "dark matter": "darkmatter",
              "undread": "undead", "tornado": "wind",
              "infection": "infected"}                        # a type as the card data writes it; renamed types
                                                              # (owner 11-17: the deprecated type is 'infected')
MINION_ROLES = {"equip": "offensive_minion", "protective": "defensive_minion"}  # a minion's 2023 role icons
BEFORE_DAY, BEFORE_SORT = "Before 2020", "0000-00-00"         # owner Q6 (11-14 22:20): a Created with no day, first
PHOTO_SOURCE = "physical card photo"                          # its source (the photos themselves are in no timeline)
PHOTO_CREATED = {"pep-drink": "Pep Drink", "wham-bam-bone": "Wham Bam Bone"}   # the live cards dated so: only these
PHOTO_MORGUE = {"LC-1c11be0504": "M'uru"}                     # the Morgue candidate dated so (discord_morgue)
ERA_RANK = {"A": 1, "B": 1, "C": 1, "A-item": 1, "B-item": 1, "C-item": 1, "D1": 2, "D2": 3, "D3": 3,
            "D-item": 3}                                      # template eras in time (A, B, C overlapped 2020-22)
REPOST_DAYS = 60                                              # D1 after D2 this long after D2's first: a repost
OWNER_ANSWERS = (                                             # STAGE1's questions: the owner's answers, when
    ("Q1", "C", "2026-11-13 08:15", "Meme/joke channels: meme-cards, joke-ideas, cursed and idea-graveyard; "
                                    "patty-vault and ai-images left out entirely."),
    ("Q2", "A", "2026-11-13 08:15", "Morgue candidates: only cards really played (port or update-port picture, update "
                                    "changelog line, card-classification, old repo file) plus the joke cards; drafts "
                                    "seen only in idea channels are 'ideas never ported'."),
    ("Q3", "A", "2026-11-13 08:15", "Split cards: both halves' timelines start with the joint card's versions 'as <old "
                                    "name>', each with its own half's stats."),
    ("Q4", "B", "2026-11-13 08:15", "Changelog lines: only field changes stated as data (e.g. 'ATK 5 -> 4') and a "
                                    "dated 'ability changed' mark without text; never the line's text."),
    ("Q5", "picture rows", "2026-11-14 21:55", "The first repo's commits: picture rows badged 'first repo', like the "
                                               "old repo's; no text-only rows. Its push notices are evidence only."),
    ("Q6", "photos left out", "2026-11-14 22:20", "Physical card photos (the 16 of the 'Old stuff' thread, nothing "
                                                 "else): in no timeline, review item or Morgue evidence, never cropped. "
                                                 "Created 'Before 2020' (sorting first, source: physical card photo) "
                                                 "for exactly Pep Drink, Wham Bam Bone and M'uru (a Morgue candidate); "
                                                 "no other Created comes from them."))


# ========================
# Helpers
# ========================
class Paths:
    def __init__(self, root: str):
        self.root, self.out = root, join(root, *OUT)
        self.timelines, self.checks = join(self.out, "timelines"), join(self.out, "checks")
        self.created, self.review = join(self.out, "created.json"), join(self.out, "review.jsonl")
        self.stats, self.ruled = join(self.out, "stats.txt"), join(self.out, "ruled.json")

    def at(self, parts: tuple) -> str:
        return join(self.root, *parts)


def review_id(kind: str, key: str) -> str:
    return "chain-" + sha1(f"{kind}|{key}".encode()).hexdigest()[:12]


def row_item_id(row: str, key: str) -> str:
    """An old row's field item id (oldgit_rows.review_items: 'oldrows-' + stable_id('field', row, field))."""
    return "oldrows-" + sha1("\x1f".join(("field", row, key)).encode()).hexdigest()[:12]


def picture_id(kind: str, picture: str, card: Optional[str], issued: dict) -> str:
    """A picture-keyed item's id (PICTURE_KINDS): the picture's alone while the page showed it for that card (or
    never showed it); asked about another card, a new id keyed by the card too: an answered id keeps its meaning."""
    rid = review_id(kind, picture)
    had = (issued.get(rid) or {}).get("card")
    return rid if had in (None, card) else review_id(kind, f"{picture}|{card}")


def ruled_by(r: dict) -> str:
    """A field's method from an answer: 'owner', or 'triage' for the triage's owner-delegated answers."""
    return "triage" if str(r.get("source") or "").startswith("triage") else "owner"


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


def blank(e: dict) -> bool:
    """A version whose ability box is blank by eye (the readers' eye files: ability '')."""
    field = (e.get("fields") or {}).get("ability") or {}
    return is_sure(field) and field["value"] == "" and field.get("method") == "eye"


def stat_line(value) -> Optional[tuple]:
    """A stats value as the website's line (hp, def, atk, spd), None when it isn't one creature's four stats."""
    if not isinstance(value, dict) or any(not isinstance(value.get(k), int) for k in STATS):
        return None
    return tuple(value[k] for k in STATS)


def ability_change(a: dict, b: dict) -> str:
    """same | formatting | changed | removed | unknown: two versions' abilities (formatting alone, owner 11-09; an
    OCR'd text against a PSD's: icons aside, they print as pictures; removed: a blank box after a text, owner 11-17,
    read by eye or a PSD's own layers)."""
    if not (is_sure(a) and is_sure(b)):
        return "unknown"
    x, y = a["value"] or "", b["value"] or ""
    if x == y:
        return "same"
    if not y.strip():                                   # card data's or OCR's '' can be a read failure (Shadow Queen)
        return "unknown" if b["method"] == "psd (card data)" or "ocr" in b["method"] else "removed"
    if text_key(x) == text_key(y):
        return "formatting"
    if "psd" not in (a["method"] + b["method"]) or "ocr" in (a["method"] + b["method"]) or "eye" in (a["method"] +
                                                                                                    b["method"]):
        if text_key(dm.strip_tokens(x)) == text_key(dm.strip_tokens(y)):
            return "formatting"
    return "changed"


def compare(before: dict, after: dict, kind: Optional[str] = None) -> dict:
    """What changed from one version's fields to the next: only fields sure on both sides; the rest are unknown.
    The buff/nerf verdict is the website's (build analytics.change): the stat total up or down, the same total
    'shift', a total over OFF_SCALE not judged. Types compare as the chain's tokens (type_tokens); an item's kinds
    are no types change (kind: the card's)."""
    out, unknown = {}, []
    for key in ("name", "stars", "types", "stats"):
        a, b = before.get(key), after.get(key)
        if key == "types":
            a, b = kinds_read(a), kinds_read(b)
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
        elif key == "types":
            x, y = ([t for t in type_tokens(v or [], kind) if kind not in ITEM_CARDS or t not in ITEM_KINDS]
                    for v in (x, y))
            if Counter(x) != Counter(y):
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


def first_changes(last: Optional[dict], fields: dict, kind: Optional[str], versions: list, stop: dict,
                  until: Optional[float] = None) -> dict:
    """git_first's changes: the last version before the card repo against its first card-data version (until: its
    epoch, git_first's date; None: every later version). No name change when that last version is a Discord look of
    the first git version (stop) whose title read is in the printed title (the matcher's OCR) of every version from
    the stop up to until: the data's name is its PSD layers' (The Bagel's 'Or... is it'), not a rename; then
    same_title is set. A version retitled in between (Grape Juice) keeps it."""
    changes = compare(last["fields"], fields, kind) if last else {}
    git = ((last or {}).get("look") or {}).get("git") or {}
    name = dm.squash(((last or {}).get("fields", {}).get("name") or {}).get("value") or "")
    if "name" in changes and last["kind"] == "discord" and name and git.get("commit") and \
            (git["commit"], git.get("path")) == ((stop or {}).get("commit"), (stop or {}).get("path")):
        start = min((v["ts"] for v in versions if (v.get("commit"), v.get("path")) == (git["commit"], git.get("path"))
                     and v.get("ts") is not None), default=None)
        span = [v for v in versions if start is not None and v.get("ts") is not None and start - 1 <= v["ts"] and
                (until is None or v["ts"] <= max(until, start) + 1)]
        if span and all(name in dm.squash(v.get("title") or "") for v in span):
            del changes["name"]
            changes["same_title"] = True
    return changes


def type_tokens(value, kind: Optional[str] = None):
    """A types list in the chain's tokens (the card data's): a spaced or renamed form ('defensive minion', 'tornado'),
    and on a minion the 2023 role icons (the Equip sword, the Protective shield), as oldgit_rows maps raw icons."""
    if not isinstance(value, list):
        return value
    out = [TYPE_FORMS.get(t, t) for t in (" ".join(str(x).lower().split()) for x in value)]
    return [MINION_ROLES.get(t, t) for t in out] if kind == "minion" else out


def without_kinds(value, kind: Optional[str]):
    """Types read on an item card without its kinds (owner: item subtypes are not collected); else as read."""
    return [t for t in value if t not in ITEM_KINDS] if kind in ITEM_CARDS and isinstance(value, list) else value


def kinds_read(field: Optional[dict]) -> Optional[dict]:
    """A types field read sure as an item card's kinds alone (no_kinds: not collected) compares as no types."""
    return {**field, "confidence": "sure", "value": []} if field and field.get("kinds_alone") else field


def kinds_only(value, kind: Optional[str]) -> bool:
    """Types read on an item card that are its kinds alone: nothing to collect, so nothing to review."""
    return bool(value) and isinstance(value, list) and without_kinds(value, kind) == []


def old_kinds_only(item: dict, kind: Optional[str]) -> bool:
    """An old row's types item whose every read (proposed, the reads) is an item card's kinds alone."""
    values = [v for v in [item.get("proposed"), *(item.get("reads") or {}).values()] if v is not None]
    return item.get("field") == "types" and bool(values) and all(kinds_only(v, kind) for v in values)


def settled(before: dict, after: dict) -> bool:
    """Every field either version has is sure on both and unchanged (the old rows' fold rule)."""
    before, after = ({**f, "types": kinds_read(f["types"])} if f.get("types") else f for f in (before, after))
    keys = {k for k in FIELDS if before.get(k) or after.get(k)}
    if not keys or any(not (is_sure(before.get(k)) and is_sure(after.get(k))) for k in keys):
        return False
    change = compare(before, after)
    return not (set(change) - {"ability_change"}) and change.get("ability_change") in (None, "same", "formatting")


@lru_cache(maxsize=256)
def fold_picture(path: str, box: Optional[tuple]):
    """A picture as the preview cuts it (a sheet's cell: its box of the trimmed sheet), trimmed, RGB at FOLD_W wide
    (float32); None when it can't be read."""
    import numpy as np
    from PIL import Image
    try:
        img = dm.trim(dm.load_image(path))
        if box:
            w, h = img.size
            img = dm.trim(img.crop((round(box[0] * w), round(box[1] * h), round(box[2] * w), round(box[3] * h))))
    except (OSError, ValueError, Image.DecompressionBombError):
        return None
    if img.width < 8 or img.height < 8:
        return None
    return np.asarray(img.resize((FOLD_W, max(8, round(FOLD_W * img.height / img.width))), Image.LANCZOS),
                      dtype=np.float32)


def changed_share(a, b) -> Optional[float]:
    """The share of the card that changed from picture a to b (fold_picture's): a aligned onto b (ECC, affine, at half
    size first), both blurred, a pixel changed when a channel is off by more than FOLD_TOL inside a FOLD_PATCH square,
    the border's margin left out; None when their shapes differ (FOLD_ASPECT)."""
    import cv2
    import numpy as np
    if abs(a.shape[1] / a.shape[0] - b.shape[1] / b.shape[0]) > FOLD_ASPECT * b.shape[1] / b.shape[0]:
        return None
    h = b.shape[0]
    x = cv2.resize(a, (FOLD_W, h), interpolation=cv2.INTER_AREA)
    gx, gb = x.mean(axis=2), b.mean(axis=2)
    warp = np.eye(2, 3, dtype=np.float32)
    steps = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 80, 1e-5)
    try:
        half = [cv2.resize(g, (FOLD_W // 2, h // 2), interpolation=cv2.INTER_AREA) for g in (gb, gx)]
        _, warp = cv2.findTransformECC(*half, warp, cv2.MOTION_AFFINE, steps, None, 3)
        warp[:, 2] *= 2
        _, warp = cv2.findTransformECC(gb, gx, warp, cv2.MOTION_AFFINE, steps, None, 3)
    except cv2.error:
        pass                                     # no alignment found: compared as it stands
    x = cv2.warpAffine(x, warp, (FOLD_W, h), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                       borderMode=cv2.BORDER_REPLICATE)
    d = np.abs(cv2.GaussianBlur(x, (0, 0), FOLD_BLUR) - cv2.GaussianBlur(b, (0, 0), FOLD_BLUR)).max(axis=2) > FOLD_TOL
    m = round(FOLD_MARGIN * FOLD_W)
    k = np.ones((FOLD_PATCH, FOLD_PATCH), np.uint8)
    return float(cv2.dilate(cv2.erode(d[m:h - m, m:FOLD_W - m].astype(np.uint8), k), k).mean())


def png_share(a: str, b: str) -> Optional[float]:
    """png_changes' own share (the git versions' rule) of two PNG files of one size; None otherwise."""
    from PIL import Image
    from commands.png_changes import changed_share as exact
    try:
        with Image.open(a) as x, Image.open(b) as y:
            if x.format != "PNG" or y.format != "PNG" or x.size != y.size:
                return None
        with open(a, "rb") as x, open(b, "rb") as y:
            return exact(x.read(), y.read())
    except (OSError, ValueError, Image.DecompressionBombError):
        return None


def look_share(root: Optional[str], before: dict, after: dict) -> Optional[float]:
    """How much of the card changed from one Discord look to the next: 0 for one git look; png_share of their shown
    pictures when both are PNGs of one size (no sheet cell); else the least changed_share between their first
    FOLD_PICTURES pictures (reposts of one look); None when none compare."""
    git = [((e.get("look") or {}).get("git") or {}) for e in (before, after)]
    if git[0] and (git[0].get("row") or (git[0].get("commit"), git[0].get("path"))) == \
            (git[1].get("row") or (git[1].get("commit"), git[1].get("path"))):
        return 0.0
    if not root:
        return None
    shown = [(e.get("pictures") or [{}])[0] for e in (before, after)]
    if all(p.get("file") and not p.get("box") for p in shown):
        exact = png_share(join(root, shown[0]["file"]), join(root, shown[1]["file"]))
        if exact is not None:
            return exact
    cut = lambda e: [fold_picture(join(root, p["file"]), tuple(p["box"]) if p.get("box") else None)
                     for p in (e.get("pictures") or [])[:FOLD_PICTURES] if p.get("file")]
    shares = [changed_share(a, b) for a in cut(before) if a is not None for b in cut(after) if b is not None]
    return min((s for s in shares if s is not None), default=None)


def fold(root: Optional[str], before: Optional[dict], after: dict) -> tuple:
    """(minor, share) of a Discord look after the one before: minor only when settled (every field sure and unchanged)
    and its picture near (look_share under png_changes.VISIBLE: the owner's 11-09 rule, as the old rows fold)."""
    from commands.png_changes import VISIBLE
    if not before or before["kind"] != "discord" or not settled(before["fields"], after["fields"]):
        return False, None
    share = look_share(root, before, after)
    return share is not None and share < VISIBLE, share


# ========================
# Inputs
# ========================
class Inputs:
    """The parts' outputs and the card repo's data, read once."""
    def __init__(self, root: str, cards_repo: str):
        p = Paths(root)
        self.root = root
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
        self.text_truth = defaultdict(dict)      # an id's fields from every eye file
        self.old_eye = defaultdict(dict)         # an old/first repo PNG's fields by eye (o:<blob>; not the PSD's)
        for r in dm.read_jsonl(p.at(TEXT_TRUTH)):
            if r.get("src") == "discord":
                self.text_truth[r["id"]].update(r["fields"])
            elif str(r.get("id")).startswith("o:") and str(r.get("how")).startswith("eye:"):
                self.old_eye[r["id"][2:]].update(r["fields"])
        self.bars = {r["file"]: r for r in dm.read_jsonl(p.at(BAR_READS)) if r.get("src") == "discord"}
        self.old_rows = defaultdict(list)
        for row in dm.read_jsonl(p.at(OLD_ROWS)):
            self.old_rows[row["card"]].append(row)
            cid = self.by_key.get(row["card"])
            if row.get("psd_only") and cid and row["ts"] < (self.cards[cid].get("fs") or float("inf")):
                self.cards[cid].update(fs=row["ts"], fs_psd=row,                      # its first record: a PSD
                                       fs_from="first git" if row.get("repo") == "first" else "old git")
        psd_first = dm.read_json(p.at(PSD_FIRST), {}) or {}
        self.psd_first = psd_first.get("cards") or {}
        for key, f in (psd_first.get("folders") or {}).items():    # MDW drafts before the card folders (oldgit_rows)
            cid = self.by_key.get(key)
            if cid and f.get("ts"):
                self.cards[cid]["fs_folder"] = f["ts"]
        self.owner = dm.read_json(p.at(OWNER), {}) or {}
        self.old_review = dm.read_jsonl(p.at(OLD_REVIEW))
        self.facts = dm.read_jsonl(p.at(FACTS))
        self.first_repo = dm.read_jsonl(p.at(FIRST_REPO))
        self.heads = (dm.read_json(p.at(EYE_LINES), {}) or {}).get("heads") or {}
        self.header_rows = {r["id"]: r for r in dm.read_jsonl(p.at(HEADER))}
        self.header = {k: r for k, r in self.header_rows.items() if r.get("sure")}
        self.header_eye = {r["id"]: r for r in dm.read_jsonl(p.at(HEADER_EYE)) if r.get("sure")}
        taken = {t["picture"]: t.get("entry") for t in (dm.read_json(p.at(MORGUE), {}) or {}).get("taken") or []}
        for row in self.verdicts:                # a joke or lost card's picture by the Morgue's eye: never a live card's
            if row["id"] in taken:
                row["morgue_taken"] = taken[row["id"]]
                if row["verdict"] in SURE_CLASSES:
                    row["verdict"] = "unsure"
        eye = dm.read_json(p.at(MATCH_EYE), {}) or {}
        only, crops = eye.get("dates_only") or {}, eye.get("crops") or {}
        for row in self.verdicts:                # no card on its own (by eye): dates only, never shown
            if row["id"] in only:
                row["dates_only"] = only[row["id"]]
            elif row["id"] in crops and row.get("cell") is None:    # the whole card cut out: the crop is shown
                row["box"], row["crop"] = list(crops[row["id"]]["box"]), crops[row["id"]]["why"]
        self.dates_also = {pid: list((v or {}).get("cards") or ()) for pid, v in (eye.get("dates_also") or {}).items()
                           if pid in only}           # other cards printed plainly on it: it dates them too
        self.psds = defaultdict(list)
        for r in dm.read_jsonl(p.at(PSDS)):
            if r.get("card") and r.get("fields"):
                self.psds[r["card"]].append(r)
        self.rulings = (dm.read_json(p.at(RULINGS), {}) or {}).get("rulings") or {}
        self.issued = (dm.read_json(p.at(ISSUED), {}) or {}).get("ids") or {}
        git_dir = dm.git_dir_of(cards_repo)
        if not git_dir:
            raise SystemExit(f"no card repo at {cards_repo}")
        made = loads(dm.git_out(git_dir, "show", f"origin/{CARD_BRANCH}:{CREATED_PATH}") or b"{}")
        self.created_file = made.get("cards") or {}
        self.first_data = first_versions(git_dir)
        self.statless = statless(git_dir)


def statless(git_dir: str) -> set:
    """The card keys whose data stats are all -1 (most items, N.M.E. cards): no bars printed today."""
    stats = loads(dm.git_out(git_dir, "show", f"origin/{DATA_BRANCH}:{STATS_FILE}") or b"{}")
    out = set()
    for key, e in (stats.get("cards") or stats).items():
        line = [(e.get("stats") or {}).get(s) for s in ("hp", "defense", "attack", "speed")] if isinstance(e, dict) \
            else [0]
        if not any(isinstance(v, int) and v >= 0 for v in line):
            out.add(key)
    return out


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


def other_name(inputs: Inputs, cid: Optional[str], value) -> bool:
    """A title read snapped to another live card's name or key (P.A.N.D.A. Mk A read 'Panda'), none of this card's
    own: its names, aliases and old repo titles (an eye truth comes first)."""
    if not cid or not value:
        return False
    names = getattr(inputs, "name_cards", None)
    if names is None:
        names = inputs.name_cards = defaultdict(set)
        for c, card in inputs.cards.items():
            for n in (card.get("name"), card.get("key")):
                if n:
                    names[dm.squash(n)].add(c)
    s = dm.squash(value)
    if not names.get(s, set()) - {cid}:
        return False
    card = inputs.cards.get(cid) or {}
    own = {card.get("name"), card.get("key"), *(card.get("aliases") or ())}
    for r in (getattr(inputs, "old_rows", None) or {}).get(card.get("key"), ()):
        name = (r.get("fields") or {}).get("name") or {}
        own |= {r.get("old_id"), name.get("value") if name.get("confidence") == "sure" else None}
    return s not in {dm.squash(n) for n in own if n}


def read_key(row: dict) -> str:
    """A picture's key in the readers' outputs: its file; a sheet's cell's ends '#c<n>' (read cut by its box)."""
    return row["file"] + (f"#c{row['cell']}" if row.get("cell") else "")


def picture_fields(inputs: Inputs, row: dict, kind: Optional[str]) -> dict:
    """A Discord picture's own reads (a sheet's cell's too): an eye truth first (the readers' by-eye files), else the
    readers' sure flags."""
    fields = {}
    text = inputs.text.get(read_key(row)) or {}
    eye = inputs.text_truth.get("d:" + row["sha256"] + (f"#c{row['cell']}" if row.get("cell") else "")) or {}
    for key, src in (("name", "title"), ("stars", "stars"), ("types", "types"), ("ability", "ability")):
        read = text.get(src) or {}
        if src in eye and (eye[src] not in (None, "") or key == "ability" and eye[src] == ""):   # "": a blank box
            value = sorted(type_tokens(eye[src], kind)) if key == "types" else eye[src]
            fields[key] = sure(value, "eye")
        elif key == "name" and read.get("sure") and other_name(inputs, row.get("card"), read.get("value")):
            ocr = f" (OCR '{read['ocr']}')" if read.get("ocr") else ""
            fields[key] = unsure(read.get("method") or "ocr", f"its title read is another live card's name{ocr}",
                                 read["value"], read.get("conf"))
        elif read.get("sure") and read.get("value") not in (None, ""):
            fields[key] = sure(sorted(type_tokens(read["value"], kind)) if key == "types" else read["value"],
                               read.get("method") or "ocr", read.get("conf"))
        elif key == "types" and row["id"] in getattr(inputs, "header", {}) and \
                inputs.header[row["id"]].get("card") == row.get("card"):
            got = inputs.header[row["id"]]                 # the header's icons as a git version's (same template)
            fields[key] = sure(got["types"], "git header", got["ncc"], version=got["version"])
        elif key == "types" and row["id"] in getattr(inputs, "header_eye", {}) and \
                inputs.header_eye[row["id"]].get("card") == row.get("card"):
            got = inputs.header_eye[row["id"]]             # eras A-C: an eye reference's icons (header_types eye)
            fields[key] = sure(got["types"], "eye" if got["ref"] == row["id"] else "eye header", got["ncc"],
                               ref=None if got["ref"] == row["id"] else got["ref"])
        elif text:
            why = "Discord types are never sure (icons outside Types/)" if key == "types" else \
                f"stars: {read['doubt']}" if read.get("doubt") else "read under its bar"
            fields[key] = unsure(read.get("method") or "ocr", why, read.get("value"), read.get("conf"))
            miss = (getattr(inputs, "header_rows", {}) or {}).get(row["id"]) or {}
            if key == "types" and miss.get("card") == row.get("card") and miss.get("ncc", 1) < HEADER_MISS:
                fields[key].update(header_miss=True, why=why + "; its header icons match no git version's")
        else:
            fields[key] = unsure("none", "a sheet's cell: not read" if row.get("cell") else "not read (no card era)")
    read = inputs.bars.get(read_key(row))
    card = inputs.cards.get(row.get("card")) or {}
    if kind not in NO_BARS and not (read and not read.get("labels") and card.get("key") in
                                    getattr(inputs, "statless", ())):
        fields["stats"] = picture_stats(read, kind)    # a card with no stats today, no bar label read: not printed
    return fields


def picture_stats(read: Optional[dict], kind: Optional[str]) -> dict:
    """The era bar readers' stats: a truth (eye, psd) first, else every printed row sure; two creatures to review."""
    if not read:
        return unsure("none", "bars not read")
    truth = read.get("truth") or {}
    if truth.get("stats2") or read.get("two"):
        top = truth.get("stats") or {k: (v or {}).get("v") for k, v in (read.get("stats") or {}).items()}
        bottom = truth.get("stats2") or {k: (v or {}).get("v") for k, v in (read.get("stats2") or {}).items()}
        if all(isinstance((h or {}).get(k), int) for h in (truth.get("stats"), truth.get("stats2")) for k in STATS):
            return sure({"creatures": [{k: top[k] for k in STATS}, {k: bottom[k] for k in STATS}]},
                        truth.get("from") or "eye")             # both halves told (eye, psd): each its own, never summed
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
    """An old repo row's fields: sure values kept, the rest withheld with their review item ids (a row oldrows has
    none for, a split's: its value proposed for the chain's own item)."""
    out = {}
    for key, field in (row.get("fields") or {}).items():
        if field.get("confidence") == "sure":
            out[key] = sure(field.get("value"), field.get("method") or "psd", None, check=field.get("check"),
                            half=field.get("half"))     # a split card's half (owner Q3)
            continue
        out[key] = {"method": field.get("method") or "psd", "confidence": "review", "why": field.get("why")}
        rid = review_of.get((row["id"], key))
        if rid:
            out[key]["review"] = rid
        elif field.get("value") is not None:
            out[key]["proposed"] = field["value"]
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


def same_read(a: dict, b: dict) -> bool:
    """Two posted PSDs read alike (one look): name, types, ability words and composite stats."""
    ra, rb = a.get("read") or {}, b.get("read") or {}
    sa, sb = (((x.get("fields") or {}).get("stats") or {}).get("value") for x in (a, b))
    return dm.squash(ra.get("name") or "") == dm.squash(rb.get("name") or "") and \
        sorted(ra.get("types") or []) == sorted(rb.get("types") or []) and \
        dm.words(ra.get("ability") or "") == dm.words(rb.get("ability") or "") and (sa is None or sb is None or sa == sb)


def psd_home(r: dict, mine: set, whole: Optional[dict], entries: list, seen: dict, held: dict) -> tuple:
    """(the entry a posted PSD's fields go on, how): its own message's look; else the nearest whole card within
    PSD_BITS that holds no PSD read otherwise (another look); else (None, None)."""
    looks = [e for e in entries if e["kind"] == "discord" and e["id"] in seen]
    near = []
    for e in looks:
        bits = min((dm.hash_distance(whole, w) for w in seen[e["id"]][1]), default=None) if whole else None
        own = bool(seen[e["id"]][0] & mine)
        if own or (bits is not None and bits <= PSD_BITS and all(same_read(r, o) for o in held.get(e["id"], ()))):
            near.append((not own, 99.0 if bits is None else bits, e))
    if not near:
        return None, None
    best = min(near, key=lambda t: t[:2])
    return best[2], "bits" if best[0] else "message"


def merge_fields(fields: dict, other: dict, how: str, fill: bool = True) -> dict:
    """fields filled from another picture of the same look: a sure value its own unsure read agrees with; one it
    didn't read (a sheet's cell, unread bars, no era, nothing read) is only proposed, one its read contradicts stays
    unsure; two sure values that differ make the field unsure. fill False: disagreements only (a git look's exact
    fields)."""
    out = dict(fields)
    for key, field in other.items():
        mine = out.get(key)
        if header_apart(key, mine, field) or header_apart(key, field, mine):
            sure_one = mine if is_sure(mine) else field
            out[key] = {**unsure("git header", f"a picture of this look has other header icons ({how})",
                                 sure_one["value"]), "header_miss": True}
            continue
        if not is_sure(field):
            if fill and contrary(key, mine, field):
                out[key] = unsure(mine["method"], f"a picture of this look reads otherwise ({how})", mine["value"])
            continue
        if is_sure(mine):
            if differs(key, field["value"], mine["value"]):
                out[key] = unsure(mine["method"], f"pictures of this look disagree ({how})", mine["value"])
            continue
        if not fill or mine is None:
            continue
        own = mine.get("proposed")
        if own is None or mine.get("method") == "none":
            if own is None:
                out[key] = unsure(mine.get("method") or "none", f"{mine.get('why') or 'not read'}; a repost reads "
                                  f"it ({how})", field["value"])
        elif differs(key, own, field["value"]):
            out[key] = unsure(mine["method"], f"pictures of this look disagree ({how})", own)
        else:
            out[key] = {**field, "method": f"{field['method']} ({how})"}
    return out


def header_apart(key: str, a: Optional[dict], b: Optional[dict]) -> bool:
    """Types one picture takes from a git version's header while another's header matches no git version's (an
    icon added or gone: Warcraftian Priest's light icon)."""
    return key == "types" and is_sure(a) and str(a["method"]).startswith("git header") and bool((b or {}).get(
        "header_miss"))


def contrary(key: str, mine: Optional[dict], other: Optional[dict]) -> bool:
    """A repost's own unsure read of stars or stats says otherwise than the look's sure value, read by a reader (an
    eye or PSD value stands): the rows both read differ."""
    if key not in ("stars", "stats") or not is_sure(mine) or str(mine["method"]).startswith(BACKED) or \
            not other or other.get("proposed") is None:
        return False
    a, b = mine["value"], other["proposed"]
    if key == "stats":
        return isinstance(a, dict) and isinstance(b, dict) and any(
            isinstance(a.get(k), int) and isinstance(b.get(k), int) and a[k] != b[k] for k in STATS)
    return a != b


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


def near_look(a: dict, b: dict) -> bool:
    """Two drafts' whole cards within MERGE_BITS with the same timing tags and numbers read (within BLIND_BITS when
    either has too little text to tell)."""
    if not a["row"].get("whole") or not b["row"].get("whole"):
        return False
    bits = dm.hash_distance(a["row"]["whole"], b["row"]["whole"])
    if a["marks"] is None or b["marks"] is None:
        return bits <= BLIND_BITS
    return bits <= MERGE_BITS and a["marks"] == b["marks"]


def mark_counts(lines) -> Optional[Counter]:
    """discord_verdicts.read_marks counted (a multiset): a number swapped for one printed elsewhere on the card
    (5/7/6/6 -> 5/7/7/6) tells two looks apart; None when too little text is read."""
    if read_marks(lines) is None:
        return None
    raw = " ".join(line if isinstance(line, str) else line[0] for line in lines or []
                   if isinstance(line, str) or line[1] >= 0.3)
    found = Counter("t:" + next(g for g in m.groups() if g).lower().translate(dm.CONFUSABLE) for m in TAG.finditer(raw))
    found.update(f"n:{sub(r'^0+(?=[0-9])', '', n.replace(',', '.'))}:{w}" for n, w in NUMBER.findall(dm.fold(raw)))
    return found


@lru_cache(maxsize=2048)
def grey_card(root: str, file: str, box: Optional[tuple]):
    """A picture (its sheet cell's box) in grey at LOCAL_W wide, None when it can't be read."""
    import numpy as np
    from PIL import Image
    try:
        with Image.open(join(root, file)) as img:
            img = img.convert("L")
            if box:
                w, h = img.size
                img = img.crop((int(box[0] * w), int(box[1] * h), int(box[2] * w), int(box[3] * h)))
            if img.width < 8 or img.height < 8:
                return None
            return np.asarray(img.resize((LOCAL_W, max(1, round(LOCAL_W * img.height / img.width))),
                                         Image.BILINEAR), dtype=np.float32)
    except OSError:
        return None


def edited_cells(root: str, a: dict, b: dict) -> list:
    """Two pictures alike but for one area (an icon added, stars, a bar, a digit, a word): the grid cells far apart
    (LOCAL_EDIT, and LOCAL_RATIO times the pair's 90th percentile cell), what the whole-card hash averages away, as
    [x0, y0, x1, y1] shares of the card. Pictures of other shapes or crops can't tell ([])."""
    import numpy as np
    x = grey_card(root, a["file"], tuple(a["box"]) if a.get("box") else None)
    y = grey_card(root, b["file"], tuple(b["box"]) if b.get("box") else None)
    if x is None or y is None or x.shape != y.shape:
        return []
    d = np.abs(x - y)
    h, w = (d.shape[0] // LOCAL_CELL) * LOCAL_CELL, (d.shape[1] // LOCAL_CELL) * LOCAL_CELL
    if not h or not w:
        return []
    cells = d[:h, :w].reshape(h // LOCAL_CELL, LOCAL_CELL, w // LOCAL_CELL, LOCAL_CELL).mean(axis=(1, 3))
    bar = max(LOCAL_EDIT, LOCAL_RATIO * max(float(np.percentile(cells, 90)), LOCAL_FLOOR))
    H, W = d.shape
    return [[j * LOCAL_CELL / W, i * LOCAL_CELL / H, (j + 1) * LOCAL_CELL / W, (i + 1) * LOCAL_CELL / H]
            for i, j in zip(*np.nonzero(cells >= bar))]


def formatting_edit(cells: list, mine: list, theirs: list) -> bool:
    """Every edited cell lies on OCR'd lines (boxes: x, y, w, h shares) that say the same words on both pictures
    (text_key: case, punctuation, spacing aside): a formatting change, one look (owner 11-09)."""
    def on(lines, c):
        return {text_key(l[0]) for l in lines if l[2] < c[2] and c[0] < l[2] + l[4] and l[3] < c[3] and
                c[1] < l[3] + l[5]}
    return bool(cells) and all(on(mine, c) and on(mine, c) == on(theirs, c) for c in cells)


@lru_cache(maxsize=1)
def real_words() -> frozenset:
    try:
        with open(WORDS, encoding="utf-8", errors="ignore") as f:
            return frozenset(w.strip().lower() for w in f if w.strip())
    except OSError:
        return frozenset()


def ocr_words(lines, labels: bool = False) -> set:
    """A picture's OCR'd words (3+ letters; lines read with confidence 0.3 or more, as dm.read_text); labels: only
    those on one-word lines (where a stat label prints)."""
    out = set()
    for line in lines or []:
        if isinstance(line, str) or line[1] >= 0.3:
            words = {w for w in findall(r"[a-z]+", (line if isinstance(line, str) else line[0]).lower()) if len(w) >= 3}
            if not labels or len(words) == 1:
                out |= words
    return out


def class_line(lines) -> Optional[str]:
    """A creature's class line ('Royalty/Mage/Higher Being') as text_key: the bottom printed row of OCR'd lines
    (boxes: x, y, w, h shares), its middle under CLASS_Y, centred, short, no digit, bracket or final period, under
    2 lower-case words (timeline_text.class_like); None when none is read."""
    def mid(line):
        return line[3] + line[5] / 2
    lines = [l for l in lines or [] if not isinstance(l, str) and len(l) >= 6 and str(l[0]).strip()]
    if not lines:
        return None
    low = max(lines, key=mid)
    row = sorted((l for l in lines if abs(mid(l) - mid(low)) < 0.5 * min(l[5], low[5])), key=lambda l: l[2])
    text = " ".join(l[0] for l in row)
    x0, x1 = min(l[2] for l in row), max(l[2] + l[4] for l in row)
    lower = sum(1 for w in text.split() if fullmatch(r"[^A-Za-z]*[a-z][a-z'’\-]*[^A-Za-z]*", w))
    if mid(low) < CLASS_Y or abs((x0 + x1) / 2 - 0.5) >= CLASS_MID or len(text.split()) > CLASS_WORDS or lower >= 2 \
            or search(r"[0-9()]|\.\s*$", text) or (search(r"[,;]\s*$", text) and search(r"[a-z]", text)):
        return None
    return text_key(text) or None


def solid(words: set, labels: set = frozenset()) -> set:
    """Words OCR noise can't make: 4+ letters, a dictionary word, not a stat label misread (on a label's line)."""
    vocabulary = real_words()
    return {w for w in words if len(w) >= 4 and w in vocabulary and not (
            w in labels and any(SequenceMatcher(None, w, label).ratio() >= LABEL_WORD for label in STAT_LABELS))}


def misread(w: str, x: str) -> bool:
    """x may be w misread: alike (NEAR_WORD), not w with letters added at one end (demon/demonic, mage/damage)."""
    return SequenceMatcher(None, w, x).ratio() >= NEAR_WORD and not (
        x.startswith(w) or w.startswith(x) or x.endswith(w) or w.endswith(x))


def lone_words(mine: set, theirs: set, labels: set = frozenset()) -> list:
    """My solid words with no near twin in their words (a misread aside): a word changed, added or dropped."""
    return sorted(w for w in solid(mine, labels) - theirs if not any(misread(w, x) for x in theirs))


def same_words(a: dict, b: dict) -> bool:
    """Two pictures print the same text, as far as read: the same class line when both read one, their abilities'
    words (sure or not) alike (Jaccard SAME_WORDS, ABILITY_WORDS each), and no real word on one with no near twin on
    the other (a reworded ability, a class line renamed: what the hash and the marks can't see)."""
    if a.get("class") and b.get("class") and a["class"] != b["class"]:
        return False
    x, y = a.get("ability_words") or set(), b.get("ability_words") or set()
    if len(x) >= ABILITY_WORDS and len(y) >= ABILITY_WORDS and len(x & y) / len(x | y) < SAME_WORDS:
        return False
    return not (lone_words(a.get("words") or set(), b.get("words") or set(), a.get("labels") or set()) or
                lone_words(b.get("words") or set(), a.get("words") or set(), b.get("labels") or set()))


def told_look(inputs, cid: Optional[str], key) -> Optional[int]:
    """The card's look today holding the version a 'look' answer names (its key: look numbers shift when a version
    is added); None when that version is gone."""
    return next((v.get("look", 0) for v in inputs.versions.get(cid) or [] if v["key"] == key), None) \
        if isinstance(key, str) else None


def confirms(r: Optional[dict], value) -> bool:
    """An answer carries what it confirms (the review page's radio value): it holds only while the build proposes
    that same value (a field's, a Created day, a picture's card); a rerun proposing another leaves it not applied."""
    return bool(r) and value is not None and dumps(r.get("value"), sort_keys=True) == dumps(value, sort_keys=True)


def owner_rows(inputs) -> Counter:
    """The owner's answers that change what enters, applied to the inputs in place: a picture's card named, its look
    told (a git look by its first version's key, or a draft), a named-only picture confirmed, 'not this card', a joke
    picture that is the live card's, an unsure changelog fact right or wrong, an unsure old map's
    rows given to a card. A picture's answers (but a git look's key) and a map's 'yes' carry the card they were given
    on: another card now, not applied. Fields, older templates, facts against pictures and Created are answered where
    the chain makes them. The ids acted on go into inputs.used (ruled.json: the review page tells an answer taken in
    from one whose item changed)."""
    rulings, done = getattr(inputs, "rulings", None) or {}, Counter()
    used = inputs.used = getattr(inputs, "used", None) or set()
    if not rulings:
        return done
    cards = inputs.cards
    issued = getattr(inputs, "issued", None) or {}
    for row in inputs.verdicts:
        rid = row["id"]
        match = rulings.get("match:" + rid)
        mine = lambda r: r.get("choice") == "look" or confirms(r, row.get("card"))        # given on its card
        if match and match.get("choice") == "card" and match.get("value") in cards and \
                row["verdict"] not in SURE_CLASSES:
            row.update(card=match["value"], cards=[match["value"]], verdict="version_unsure", owner="card named",
                       version={"unsure": "the owner named its card: which look?"})
            used.add("match:" + rid)
            done["owner: a picture's card named"] += 1
        elif match and match.get("choice") == "none":
            used.add("match:" + rid)
        ids = {k: picture_id(k, rid, row.get("card"), issued) for k in PICTURE_KINDS}
        named, told, joke = (rulings.get(ids[k]) for k in ids)
        for k, r in (("named only", named), ("look not told", told), ("joke picture", joke)):
            if r and r.get("choice") in ("yes", "no", "draft", "live") and not mine(r):
                done[f"owner: a {k} answer given on another card (not applied)"] += 1
        named, told, joke = (r if r and mine(r) else None for r in (named, told, joke))
        no = [k for k, r in (("named only", named), ("look not told", told)) if r and r.get("choice") == "no"]
        if no:
            row.update(verdict="none", owner="not this card")
            used.update(ids[k] for k in no)
            done["owner: not this card"] += 1
            continue
        if named and named.get("choice") == "yes":
            row.pop("created_review", None)
            row["owner"] = "the card's picture"
            used.add(ids["named only"])
            done["owner: a named-only picture confirmed"] += 1
        if told and told.get("choice") == "draft":
            row.update(verdict="draft", version={**(row.get("version") or {}), "shows": "draft", "look": None},
                       owner="a look not in git")
            used.add(ids["look not told"])
            done["owner: a look not in git"] += 1
        elif told and told.get("choice") == "look" and told.get("value") is not None:
            look = told_look(inputs, row.get("card"), told["value"])
            if look is None:                   # its version left the card's history: not applied
                done["owner: a git look told, its version gone (not applied)"] += 1
            else:
                row.update(verdict="version_sure", version={**(row.get("version") or {}), "look": look},
                           owner="a git look")
                used.add(ids["look not told"])
                done["owner: a git look told"] += 1
        if joke and joke.get("choice") == "live":
            row.pop("joke_draft", None)
            row.update(joke=False, owner="the live card's picture")
            used.add(ids["joke picture"])
            done["owner: a joke picture is the live card's"] += 1
    facts = {f["id"]: f for f in inputs.facts}
    maps = {i["id"]: i for i in inputs.old_review if i.get("kind") == "unsure map" and not i.get("split")}
    live = {c.get("key") for c in cards.values()}
    for rid, r in rulings.items():
        fact = facts.get((r.get("item") or {}).get("fact")) if r.get("kind") == "unsure fact" else None
        if fact and fact.get("confidence") == "unsure" and r.get("choice") in ("yes", "no"):
            if r.get("choice") == "yes":
                fact.update(confidence="sure", owner=True, **({"unit": r["value"]} if r.get("value") else {}))
            else:
                fact["confidence"] = "refused"
            used.add(rid)
            done[f"owner: an unsure fact {r.get('choice')}"] += 1
        if rid in maps:
            if r.get("choice") == "yes" and not confirms(r, maps[rid].get("card")):
                done["owner: an unsure old map's yes, its card changed (not applied)"] += 1
                continue
            key = maps[rid].get("card") if r.get("choice") == "yes" else \
                (cards.get(r.get("value")) or {}).get("key") if r.get("choice") == "card" else None
            if key not in live and r.get("choice") != "no":
                done["owner: an unsure old map's card is no live card (nothing added)"] += 1
                continue
            rows = [row for row in maps[rid].get("rows") or [] if key and (not row.get("replaces") or
                                                                            key == maps[rid].get("card"))]
            if key and not rows:               # a PSD alone that gives the card nothing: never counted as applied
                done[f"owner: an unsure old map {r.get('choice')}, no rows (nothing added)"] += 1
                continue
            for row in rows:
                mine, given = inputs.old_rows[key], {**row, "card": key, "owner": True}
                at = next((n for n, x in enumerate(mine) if x["id"] == row.get("replaces")), None)
                if row.get("replaces") and at is None:
                    done["owner: an unsure old map's PSD, its row gone (not applied)"] += 1
                    continue
                if at is not None:             # the owner's PSD of that commit is the picture's (Freeze Spell)
                    mine[at] = given
                    continue
                mine.append(given)
                cid = inputs.by_key.get(key)
                if row.get("psd_only") and cid and row["ts"] < (cards[cid].get("fs") or float("inf")):
                    cards[cid].update(fs=row["ts"], fs_psd=given,
                                      fs_from="first git" if row.get("repo") == "first" else "old git")
            used.add(rid)
            done[f"owner: an unsure old map {r.get('choice')}"] += 1
    return done


def named_on_or_after(inputs) -> int:
    """A named-only picture (created_review) posted on or after the card's Created day in the file (or a hand
    date) can't move Created: it enters as a card-sure picture. Only one that would move Created earlier stays for
    review."""
    n = 0
    for row in inputs.verdicts:
        file = inputs.created_file.get((inputs.cards.get(row.get("card")) or {}).get("key")) or {}
        if row.get("created_review") and row.get("first") and file.get("date") and (
                file.get("from") == "hand" or day(dm.epoch(row["first"]["date"])) >= file["date"]):
            row.pop("created_review")
            row["named_after_created"] = True
            n += 1
    return n


def created_sort(date: Optional[str]) -> str:
    return BEFORE_SORT if date == BEFORE_DAY else date or "9999"


def photo_missing(cards: dict, keyed: dict = PHOTO_CREATED) -> list:
    """The owner's 'Before 2020' ids (keyed: {id: name}) whose card is gone or renamed: their Created lapsed."""
    return [f"{k} ({name})" for k, name in sorted(keyed.items()) if (cards.get(k) or {}).get("name") != name]


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
        self.rulings = getattr(inputs, "rulings", None) or {}
        self.used = getattr(inputs, "used", None) or set()   # rulings acted on (owner_rows' too): ruled.json
        self.old_proposed = {i["id"]: i.get("proposed") for i in inputs.old_review}
        self.old_items = {i["id"]: i for i in inputs.old_review}
        self.not_applied: set = set()            # field answers whose value is no longer the one proposed
        self._ocr = None
        self.maybe = {}                          # card -> its earliest dated picture the matcher wasn't sure of
        for row in sorted((r for r in inputs.verdicts if r.get("first") and r["verdict"] == "unsure"
                           and not r.get("morgue_taken")), key=lambda r: r["first"]["date"]):
            for cid in row.get("maybe") or []:
                self.maybe.setdefault(cid, row)
        self.also = defaultdict(list)            # card -> dates-only pictures it is printed on beside their own card
        also = getattr(inputs, "dates_also", None) or {}
        for row in inputs.verdicts:
            for cid in also.get(row["id"]) or ():
                if cid in inputs.cards and cid != row.get("card"):
                    self.also[cid].append(row)
                else:
                    self.tally["discord: dates_also names no other live card (left out)"] += 1

    def ocr_lines(self, sha: str) -> list:
        """A picture's OCR'd lines with their boxes (the text readers' whole-card read), read once when first asked."""
        if self._ocr is None:
            self._ocr = {r["id"][2:]: r.get("fix") or [] for r in dm.read_jsonl(join(self.inputs.root, *TEXT_OCR))
                         if str(r.get("id", "")).startswith("d:")}
        return self._ocr.get(sha) or []

    def local_edit(self, a: dict, b: dict) -> bool:
        """Two pictures alike but for one edited area that isn't formatting alone (edited_cells, formatting_edit;
        a sheet's cell has no lines of its own)."""
        cells = edited_cells(self.inputs.root, a, b)
        if not cells:
            return False
        if a.get("cell") or b.get("cell"):
            return True
        return not formatting_edit(cells, self.ocr_lines(a["sha256"]), self.ocr_lines(b["sha256"]))

    def add_review(self, kind: str, key: str, **item) -> str:
        issued = getattr(self.inputs, "issued", {})
        rid = picture_id(kind, key, item.get("card"), issued) if kind in PICTURE_KINDS else review_id(kind, key)
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
                self.to_halves(self.splits[row["lost_group"]], row, split)
                continue
            if row["verdict"] not in SURE_CLASSES or cid not in cards:
                continue
            joint = self.joint_of(cid, row)
            if joint and first:                    # a split half's picture titled as the old card: both halves'
                self.to_halves(joint, row, split)
                self.tally["discord: a split half's picture titled as the old card (both halves)"] += 1
                continue
            if not first:
                self.tally["discord: card sure, no dating post"] += 1
                continue
            when = dm.epoch(first["date"])
            if cards[cid].get("fs") and when >= cards[cid]["fs"]:
                if not self.before_folders(cid, row, when):
                    self.tally["discord: after the card's first git record"] += 1
                    continue
                self.tally["discord: a draft after an MDW draft, before the card folders"] += 1
            if row.get("dates_only") and not row.get("created_review"):
                self.dates_only(cid, row)
                continue
            if row.get("crop"):                    # shown as its crop: the owner sees each (read only)
                self.add_review("cropped", row["id"], card=cid, picture=row["id"], date=first["date"],
                                channel=first["channel"], why=row["crop"], box=row["box"], read_only=True)
                self.tally["discord: the card cut out of a bigger picture (by eye): the crop is shown"] += 1
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

    def dates_only(self, cid: str, row: dict) -> None:
        """A picture that is no card on its own (owner 11-16 23:10/23:15; derived/match/eye.json dates_only): never on
        a timeline, nothing asked about it; created() still dates the card by it, and the other cards printed plainly
        on it (eye.json dates_also). A read-only item shows it."""
        also = [c for c in (getattr(self.inputs, "dates_also", None) or {}).get(row["id"]) or ()
                if c in self.inputs.cards and c != cid]
        self.add_review("dates only", row["id"], card=cid, picture=row["id"], date=row["first"]["date"],
                        channel=row["first"]["channel"], why=row["dates_only"], also=also or None, read_only=True)
        self.tally["discord: no card on its own (by eye): dates only, not shown"] += 1

    def before_folders(self, cid: str, row: dict, when: float) -> bool:
        """A draft (a look in no git record) posted after the card's first record, an MDW draft, but before its
        first card-folder record (fs_folder): still the card's (Tainted Bethany 2023-09-09 12:55)."""
        return row["verdict"] == "draft" and not (row.get("joke") or row.get("joke_draft")) and \
            when < (self.inputs.cards[cid].get("fs_folder") or 0)

    def to_halves(self, group: dict, row: dict, split: dict) -> None:
        """A split card's picture into each half's chain, before that half's first git record (one that is no card on
        its own: neither's)."""
        when = dm.epoch(row["first"]["date"])
        for half in group["split_of"]["into"]:
            fs = (self.inputs.cards.get(half) or {}).get("fs")
            if half in self.inputs.cards and (not fs or when < fs):
                if row.get("dates_only"):
                    self.dates_only(half, row)
                    continue
                split[half].append({**row, "as": group["split_of"]["name"]})

    def joint_of(self, cid: str, row: dict) -> Optional[dict]:
        """The split a card is a half of, when the picture's title reads the old card's name (sure)."""
        title = (self.inputs.text.get(read_key(row)) or {}).get("title") or {}
        if not (title.get("sure") and title.get("value")):
            return None
        return next((g for g in self.splits.values() if cid in g["split_of"]["into"] and
                     dm.squash(title["value"]) == dm.squash(g["split_of"]["name"])), None)

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
        git = {"source": ("first repo" if first.get("repo") == "first" else "old repo") if first.get("old") else "git",
               "commit": first.get("commit"),
               "path": first.get("path"), "date": dm.iso(first["ts"])}
        key = self.inputs.cards[cid]["key"]
        old = sorted((r for r in self.inputs.old_rows.get(key, []) if r["blob"].startswith(prefix)),
                     key=lambda r: r["ts"])
        if old:
            git["row"] = old[0]["id"]
            return self.row_fields(cid, old[0]), git
        return {}, git

    def row_fields(self, cid: str, row: dict) -> dict:
        """An old/first repo row's fields (old_fields), its PNG's by-eye fields over them (the readers' eye files,
        o:<blob>: a picture printing what its PSD or folder doesn't say)."""
        fields = old_fields(row, self.review_of)
        for key, src in (("name", "title"), ("stars", "stars"), ("types", "types"), ("ability", "ability")):
            eye = getattr(self.inputs, "old_eye", {}).get(row.get("blob") or "", {}).get(src)
            if eye not in (None, "") or key == "ability" and eye == "":     # "": a blank box by eye
                fields[key] = sure(sorted(type_tokens(eye, self.inputs.cards[cid].get("type"))) if key == "types"
                                   else eye, "eye")
        return fields

    def picture(self, cid: str, row: dict) -> dict:
        card = self.inputs.cards[cid]
        fields = picture_fields(self.inputs, row, card.get("type"))
        git = None
        if row["verdict"] == "version_sure" and not row.get("as"):     # a split's look is the old card's
            exact, git = self.git_look_fields(cid, row)
            if exact:                          # the exact version's fields; one it left to review stays so
                own = fields
                fields = {k: {**v, "method": f"git look: {v['method']}"} if is_sure(v) else v for k, v in exact.items()}
                for key, mine in own.items():
                    if key not in fields and is_sure(mine):     # its git version has none (a PNG with no PSD)
                        fields[key] = mine
                        self.tally[f"discord: git look, its version has no {key}: the picture's own"] += 1
                for key, mine in own.items():
                    if key in fields and not is_sure(fields[key]) and is_sure(mine):
                        fields[key] = unsure(fields[key].get("method") or "psd", "its git version's field is to "
                                             f"review ({fields[key].get('why')}); the picture reads it", mine["value"])
                fields = merge_fields(fields, {k: v for k, v in own.items() if is_sure(fields.get(k))}, "the picture")
        lines = picture_lines(self.inputs, row)
        ability = (self.inputs.text.get(read_key(row)) or {}).get("ability") or {}
        return {"row": row, "fields": fields, "git": git, "when": dm.epoch((row.get("first") or {}).get("date")),
                "marks": mark_counts(lines), "words": ocr_words(lines), "labels": ocr_words(lines, labels=True),
                "class": None if row.get("cell") else class_line(self.ocr_lines(row["sha256"])),
                "ability_words": set(findall(r"[a-z]+", str(ability.get("value") or "").lower()))}

    def looks(self, cid: str, rows: list) -> list:
        """Pictures grouped into looks, oldest first: one git look each; a draft into its
        nearest group (the hash to its first picture) near_look and without a conflict, whose every picture prints the
        same words and class line (same_words) with no local edit (Chain.local_edit)."""
        groups = []
        pictures = sorted((self.picture(cid, r) for r in rows),
                          key=lambda p: (p["when"] is not None, p["when"] or 0, p["row"]["id"]))
        for pic in pictures:
            row = pic["row"]
            key = ("git", row["version"].get("look")) if row["verdict"] == "version_sure" and not row.get("as") else None
            if key:
                home = next((g for g in groups if g["key"] == key), None)
            else:
                near = [g for g in groups if g["key"] is None and g.get("as") == row.get("as")
                        and near_look(g["pictures"][0], pic) and not conflict(g["pictures"][0]["fields"], pic["fields"])
                        and not conflict(g["fields"], pic["fields"])]
                worded = [g for g in near if all(same_words(q, pic) for q in g["pictures"])]
                fits = [g for g in worded if not any(self.local_edit(q["row"], row) for q in g["pictures"])]
                if near and not worded:
                    self.tally["discord: a near look kept apart (its words differ)"] += 1
                elif worded and not fits:
                    self.tally["discord: a near look kept apart (a local edit)"] += 1
                home = min(fits, key=lambda g: dm.hash_distance(g["pictures"][0]["row"]["whole"], row["whole"]),
                           default=None)
            if home is None:
                groups.append({"key": key, "pictures": [pic], "fields": dict(pic["fields"]), "git": pic["git"],
                               "as": row.get("as")})
            else:
                home["pictures"].append(pic)
                home["fields"] = merge_fields(home["fields"], pic["fields"], "repost " + row["id"],
                                              fill=not (home["git"] or {}).get("row"))
        return groups

    def split_stats(self, cid: str, group: dict) -> None:
        """A split card's picture (owner Q3): this half's own stats, to confirm (the halves as the old rows matched
        them)."""
        halves = {}
        for item in self.inputs.old_review:
            if (item.get("split") or {}).get("halves") and item.get("old_id") == group["as"]:
                halves = item["split"]["halves"]
        name = self.inputs.cards[cid]["key"]
        stats = group["fields"].get("stats") or {}
        whole = stats.get("value") if is_sure(stats) else stats.get("proposed")
        creatures = whole.get("creatures") if isinstance(whole, dict) else None
        half = halves.get(name)
        value = creatures[0 if half == "top" else 1] if creatures and half in ("top", "bottom") else None
        where = f"{half or 'which'} half of {group['as']}'s rows (owner Q3)"
        group["fields"]["stats"] = sure(value, stats.get("method") or "eye", stats.get("conf"), half=where) if \
            is_sure(stats) and value else unsure(stats.get("method") or "picture", where, value)

    def discord_entries(self, cid: str, rows: list, split: list) -> list:
        out, card, seen = [], self.inputs.cards[cid], {}
        for group in self.looks(cid, rows) + self.looks(cid, split):
            first = group["pictures"][0]["row"]
            if group["as"]:
                self.split_stats(cid, group)
            entry = {"id": "D-" + first["id"], "kind": "discord", "date": dm.iso(group["pictures"][0]["when"]),
                     "day": day(group["pictures"][0]["when"]), "channel": first["first"]["channel"]}
            if group["as"]:
                entry["as"] = group["as"]
            entry["look"] = {"git": {k: v for k, v in group["git"].items() if v is not None}} if group["git"] else \
                {"draft": True} if group["key"] is None else {"git": None}
            entry["pictures"] = [{"id": p["row"]["id"], "date": (p["row"].get("first") or {}).get("date"),
                                  "channel": (p["row"].get("first") or {}).get("channel"), "file": p["row"]["file"],
                                  "thumb": p["row"].get("thumb"), "cell": p["row"].get("cell"),
                                  "box": p["row"].get("box"), "verdict": p["row"]["verdict"],
                                  **({"crop": True} if p["row"].get("crop") else {})}
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
            seen[entry["id"]] = ({q["message"] for p in group["pictures"] for q in p["row"].get("posts") or ()},
                                 [p["row"]["whole"] for p in group["pictures"] if p["row"].get("whole")])
        return self.reposted_eras(cid, self.psd_entries(cid, out, seen))

    def psd_entries(self, cid: str, entries: list, seen: dict) -> list:
        """The card's PSDs posted on Discord before its first git record: their exact fields onto the look posted
        with them (its own message's first; else the nearest whole card within PSD_BITS that no PSD read otherwise
        holds), else a version of their own dated by the post (its composite as the picture). A field another PSD
        set is never overwritten (two that differ: review)."""
        card = self.inputs.cards[cid]
        held = defaultdict(list)                       # entry id -> the PSDs on it
        for r in sorted(getattr(self.inputs, "psds", {}).get(cid) or (), key=lambda r: int(r["posts"][0]["message"])):
            post = next((q for q in r["posts"] if not q.get("bot")), r["posts"][0])
            when = dm.epoch(post["date"])
            if card.get("fs") and when >= card["fs"]:
                self.tally["psd: after the card's first git record"] += 1
                continue
            mine = {q["message"] for q in r["posts"]}
            whole = (r.get("look") or {}).get("whole")
            home, how = psd_home(r, mine, whole, entries, seen, held)
            fields = {k: sure(v["value"], "psd (posted PSD)") if v.get("confidence") == "sure" else
                      unsure("psd (posted PSD)", "a posted PSD's " + k + ": its text layers' words aren't all printed "
                             "in that order on its composite" if k == "ability" else
                             "a posted PSD's unknown type icon", v["value"])
                      for k, v in r["fields"].items()}
            if home is not None:
                for k, v in fields.items():
                    old = home["fields"].get(k)
                    if (old or {}).get("psd"):         # another PSD's: kept
                        if is_sure(v) and is_sure(old) and differs(k, v["value"], old["value"]):
                            home["fields"][k] = {**unsure("psd (posted PSD)", f"two PSDs posted with this look differ "
                                                          f"in {k}", old["value"]), "psd": old["psd"]}
                        continue
                    if is_sure(old) and old.get("method") in ("eye", "owner", "triage"):
                        continue
                    if is_sure(v) or not is_sure(old):
                        home["fields"][k] = {**v, "psd": r["id"]}
                home.setdefault("psds", []).append(r["id"])
                held[home["id"]].append(r)
                self.tally[f"psd: fields onto the look posted with it ({how})"] += 1
                continue
            for k in FIELDS:
                if k not in fields and (k != "stats" or (card.get("type") not in NO_BARS and card.get("key") not in
                                                         getattr(self.inputs, "statless", ()))):
                    fields[k] = unsure("psd (posted PSD)", f"a posted PSD's {k}: not read")
            entry = {"id": "D-" + r["id"], "kind": "discord", "date": dm.iso(when), "day": day(when),
                     "channel": post["channel"], "look": {"draft": True}, "psds": [r["id"]],
                     "pictures": [{"id": r["id"], "date": post["date"], "channel": post["channel"],
                                   "file": r["picture"], "thumb": None, "cell": None, "box": None, "verdict": "psd"}],
                     "fields": {k: {**v, "psd": r["id"]} if k in r["fields"] else v for k, v in fields.items()}}
            entries.append(entry)
            seen[entry["id"]] = (mine, [whole] if whole else [])
            held[entry["id"]].append(r)
            self.tally["psd: a version of its own"] += 1
        return entries

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
                older = review_id("older template later", f"{cid}|{e['id']}")
                if (self.rulings.get(older) or {}).get("choice") == "show":
                    self.used.add(older)
                    self.tally["owner: an older template shown"] += 1
                    kept.append(e)
                    continue
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
                     "badge": row.get("badge") or "old repo", "repo": row.get("repo") or "old",
                     "author": row.get("author"), "subject": row.get("subject"),
                     "commit": row.get("commit"), "path": row.get("path"), "blob": row.get("blob"),
                     "thumb": row.get("thumb"), "minor": bool(row.get("minor")), "share": row.get("share"),
                     "old_ability_change": row.get("ability_change"), "fields": self.row_fields(cid, row),
                     **({"meme_copy": True} if row.get("meme_copy") else {})}
            if row.get("split_from"):
                entry["as"] = row["split_from"]
            for k in ("old_name", "old_file", "in_tc_init", "two_creatures", "psd_only"):
                if row.get(k):
                    entry[k] = row[k]
            out.append(entry)
        return out

    def text_entries(self, cid: str) -> list:
        """Sure changelog facts before the stop (owner Q4: data only, no text); chat and push lines are evidence."""
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
            if fact.get("field") == "type" and card.get("type") in ITEM_CARDS and \
                    str(fact.get("to")).lower() in ITEM_KINDS:
                self.tally["facts: an item's kind (never collected)"] += 1
                continue
            change = {"ability": "changed"} if fact.get("kind") == "ability changed" else \
                {k: fact[k] for k in ("field", "from", "to", "by", "op", "unit", "stated") if fact.get(k) is not None}
            if change.get("field") == "type" and isinstance(change.get("to"), str):
                change["to"] = type_tokens([change["to"]])[0]          # the chain's token ('defensive minion')
            entry = {"id": "F-" + fact["id"], "kind": "text", "date": dm.iso(when), "day": day(when),
                     "channel": fact.get("channel"), "source": "changelog", "change": change, "confidence": "sure"}
            if fact.get("repeat_of"):
                entry["repeat_of"] = "F-" + fact["repeat_of"]
            out.append(entry)
        return out

    def star_guard(self, entries: list) -> None:
        """Icon-read star counts that change with nothing to back them go to review. The versions' sure stars in runs
        of one value; a run is backed by a read no icon reader made (BACKED, an old repo row), by reads apart (two
        template eras, or APART_DAYS), or by a sure changelog fact giving its value since the run before. An
        unbacked run between two others (a blip), at an end beside a backed run, or last after an unbacked run (the
        change's new value, nothing after agrees) goes to review; again until none is left."""
        versions = [e for e in entries if e["kind"] in ("discord", "old repo") and is_sure(e["fields"].get("stars"))]
        facts = [e for e in entries if e["kind"] == "text" and e["change"].get("field") == "stars" and
                 "to" in e["change"]]
        out = set()

        def backed(runs: list, i: int) -> bool:
            run, value = runs[i], runs[i][0]["fields"]["stars"]["value"]
            if any(e["kind"] != "discord" or str(e["fields"]["stars"]["method"]).startswith(BACKED) for e in run):
                return True
            when = [dm.epoch(e["date"]) for e in run]
            if len({e.get("era") for e in run if e.get("era")}) > 1 or when[-1] - when[0] >= APART_DAYS * 86400:
                return True
            since = dm.epoch(runs[i - 1][-1]["date"]) if i else float("-inf")
            return any(since <= dm.epoch(f["date"]) <= when[-1] and f["change"]["to"] == value for f in facts)
        while True:
            runs = []
            for e in versions:
                if e["id"] in out:
                    continue
                if runs and runs[-1][-1]["fields"]["stars"]["value"] == e["fields"]["stars"]["value"]:
                    runs[-1].append(e)
                else:
                    runs.append([e])
            if len(runs) < 2:
                break
            ok, last = [backed(runs, i) for i in range(len(runs))], len(runs) - 1
            pick = [r for i, r in enumerate(runs) if 0 < i < last and not ok[i]] or \
                [r for i, r in enumerate(runs) if i in (0, last) and not ok[i] and ok[1 if i == 0 else i - 1]] or \
                ([runs[last]] if not ok[last] and not ok[last - 1] else [])
            if not pick:
                break
            out |= {e["id"] for r in pick for e in r}
        for e in versions:
            if e["id"] in out:
                f = e["fields"]["stars"]
                e["fields"]["stars"] = unsure(f["method"], "an icon read change no other read backs (the versions "
                                              "around it say otherwise)", f["value"], f.get("conf"))
                self.tally["discord: stars, an icon read change nothing backs (review)"] += 1

    def star_fact_guard(self, entries: list) -> None:
        """Icon-read star counts in the run of versions just before a sure changelog line that changes the stars to
        that count (Charge's overlapped 2 read 1 before 'stars to 1') go to review: the line says it was another.
        Not when a read no icon reader made (BACKED) is in the run, or the run's last picture is within
        FACT_GRACE_DAYS of the line (a port picture with its changelog)."""
        versions = [e for e in entries if e["kind"] in ("discord", "old repo") and e.get("date")]
        facts = [e for e in entries if e["kind"] == "text" and e["change"].get("field") == "stars" and
                 "to" in e["change"]]
        for f in facts:
            to, when = f["change"]["to"], dm.epoch(f["date"])
            run = []
            for e in reversed([v for v in versions if dm.epoch(v["date"]) < when]):
                got = e["fields"].get("stars")
                if not is_sure(got) or got["value"] != to:
                    break
                run.append(e)
            if not run or any(e["kind"] != "discord" or str(e["fields"]["stars"]["method"]).startswith(BACKED)
                              for e in run) or when - dm.epoch(run[0]["date"]) <= FACT_GRACE_DAYS * 86400:
                continue
            for e in run:
                got = e["fields"]["stars"]
                e["fields"]["stars"] = unsure(got["method"], f"a later changelog line changes the stars to {to} "
                                              "(it read so before)", got["value"], got.get("conf"))
                self.tally["discord: stars, the count a later changelog line changes them to (review)"] += 1

    def check_facts(self, cid: str, entries: list) -> list:
        """A stat or stars fact against the pictures. Its new value (to; else by on the value before it: the last sure
        picture's, moved by the kept facts since) read otherwise by the next picture (within NEAR_DAYS, before the next
        kept fact on that field) goes to review, again until none is left; one the next picture agrees with is
        confirmed by it. With no picture that near, one whose move the pictures around it show (the last sure value
        before it and the next after it, at any distance, no other fact on that field between) is that change too:
        confirmed by the later picture (counted once)."""
        versions = [e for e in entries if e["kind"] in ("discord", "old repo") and e.get("date")]
        facts = [e for e in entries if e["kind"] == "text" and e["change"].get("field") in (*STATS, "stars")
                 and ("to" in e["change"] or "by" in e["change"])]
        gone, ruled = set(), set()
        at = lambda e: dm.epoch(e["date"])

        def read(v: dict, field: str):
            got = v["fields"].get("stats" if field in STATS else "stars")
            if not is_sure(got):
                return None
            return got["value"].get(field) if field in STATS else got["value"]

        def kept_on(e: dict) -> list:
            return [f for f in facts if f is not e and f["id"] not in gone
                    and f["change"]["field"] == e["change"]["field"]]

        def before(e: dict) -> tuple:
            """(the value before e, the picture it was read on; None when a kept fact since moved it)."""
            field = e["change"]["field"]
            pic = next((v for v in reversed(versions) if at(v) < at(e) and read(v, field) is not None), None)
            value, since = (read(pic, field) if pic else None), (at(pic) if pic else float("-inf"))
            moved = False
            for f in kept_on(e):
                if since < at(f) and facts.index(f) < facts.index(e):
                    value = f["change"]["to"] if "to" in f["change"] else value + f["change"]["by"] \
                        if value is not None else None
                    moved = True
            return value, (None if moved else pic)

        def wanted(e: dict):
            ch = e["change"]
            if "to" in ch:
                return ch["to"]
            value = before(e)[0]
            return value + ch["by"] if value is not None else None

        def next_read(e: dict, near: bool = True):
            """(picture, value): the first sure read at or after e, before the next kept fact on its field."""
            field, when = e["change"]["field"], at(e)
            until = min([at(f) for f in kept_on(e) if at(f) > when] +
                        [when + NEAR_DAYS * 86400 if near else float("inf")])
            return next(((v, read(v, field)) for v in versions if when <= at(v) < until and read(v, field) is not None),
                        None)

        again = True
        while again:                             # a fact gone widens the window of the one before it on that field
            again = False
            for e in facts:
                if e["id"] in gone or e["id"] in ruled:
                    continue
                to, seen = wanted(e), next_read(e)
                if to is None or not seen or seen[1] == to:
                    continue
                again, field = True, e["change"]["field"]
                against = review_id("fact vs picture", e["id"])
                choice = (self.rulings.get(against) or {}).get("choice")
                if choice in ("fact", "picture"):
                    self.used.add(against)
                    self.tally[f"owner: fact vs picture, the {choice} kept"] += 1
                    (ruled if choice == "fact" else gone).add(e["id"])
                    continue
                self.add_review("fact vs picture", e["id"], card=cid, fact=e["id"][2:], entry=seen[0]["id"],
                                field=field, fact_to=to, fact_by=e["change"].get("by"), picture_reads=seen[1],
                                date=e["date"], why="the next picture reads another value")
                self.tally["facts: a picture reads another value (review)"] += 1
                gone.add(e["id"])
        for e in facts:
            if e["id"] in gone or e["id"] in ruled:
                continue
            seen = next_read(e)
            if seen:
                if seen[1] == wanted(e):
                    e["confirmed_by"] = seen[0]["id"]
                continue
            value, pic = before(e)
            after = next_read(e, near=False)
            ch = e["change"]
            if pic and after and after[1] != value and ch.get("from", value) == value and \
                    (ch["to"] if "to" in ch else value + ch["by"]) == after[1]:
                e["confirmed_by"] = after[0]["id"]         # the pictures around it show that move
                self.tally["facts: confirmed by the move the pictures around it show"] += 1
        return [e for e in entries if e["id"] not in gone]

    def diff(self, entries: list, kind: Optional[str] = None) -> None:
        """Each version against the one before (text rows aside); a Discord look folds when settled and its picture
        is near the one before (fold). kind: the card's (an item's kinds are no change)."""
        before = None
        for e in entries:
            if e["kind"] == "text":
                continue
            if before is None:
                e["changes"] = {"ability_change": "first"} if e["fields"].get("ability") else {}
            else:
                e["changes"] = compare(before["fields"], e["fields"], kind)
                if e["kind"] == "old repo" and before["kind"] == "old repo" and e.get("old_ability_change") and \
                        e["changes"].get("ability_change") != "removed" and not (blank(before) or blank(e)):
                    e["changes"]["ability_change"] = e["old_ability_change"]
            if e["kind"] == "discord":
                e["minor"], share = fold(self.inputs.root, before, e)
                e.pop("share", None)
                if share is not None:
                    e["share"] = round(share, 4)
                if not e["minor"] and before and before["kind"] == "discord" and settled(before["fields"], e["fields"]):
                    self.tally["discord: settled, kept apart (a new picture)" if share is not None else
                               "discord: settled, kept apart (pictures not compared)"] += 1
            if e.get("meme_copy"):                 # the owner's Meme folder copy beside the card's own file: shown,
                e["changes"] = {}                  # no change of its own, never the version the next is diffed with
                continue
            before = e
        for e in entries:
            e.pop("old_ability_change", None)

    def no_kinds(self, cid: str, entries: list) -> None:
        """An item card's types lose its kinds (owner: item subtypes are not collected): sure ones keep the rest, or
        are 'not collected' when only kinds were read (kinds_alone: no types in diffs; so is another card's version
        read sure as kinds alone, an item then: Rock Bottom's 2023 'equip'); to review, read as its kinds alone (an
        old row's: every read), they get no review item ('not collected')."""
        card_kind = self.inputs.cards[cid].get("type")
        for e in entries:
            field = e["fields"].get("types") if e["kind"] in ("discord", "old repo") else None
            kind = card_kind if card_kind in ITEM_CARDS else \
                "item" if is_sure(field) and kinds_only(field["value"], "item") else None
            if kind is None:
                continue
            if is_sure(field) and isinstance(field["value"], list) and field["value"] != without_kinds(field["value"],
                                                                                                     kind):
                if kinds_only(field["value"], kind):
                    e["fields"]["types"] = {"method": field.get("method"), "confidence": "not collected",
                                            "why": "an item's kinds alone (not collected)", "kinds_alone": True}
                    self.tally[f"{e['kind']} types not collected (sure, an item's kinds alone)"] += 1
                else:
                    field["value"] = without_kinds(field["value"], kind)
                    self.tally[f"{e['kind']} types sure without an item's kinds"] += 1
                continue
            if not field or field.get("confidence") != "review":
                continue
            rid = field.get("review")
            read = self.old_proposed.get(rid) if rid else field.get("proposed")
            if old_kinds_only(self.old_items[rid], kind) if rid in self.old_items else kinds_only(read, kind):
                e["fields"]["types"] = {"method": field.get("method"), "confidence": "not collected",
                                        "why": "an item's kinds alone (not collected)"}
                self.tally[f"{e['kind']} types not collected (an item's kinds alone)"] += 1
            elif read != without_kinds(read, kind):
                left = None if kinds_only(read, kind) else without_kinds(read, kind)
                if rid:
                    self.old_proposed[rid] = left
                else:
                    field["proposed"] = left
                self.tally[f"{e['kind']} types to review without an item's kinds"] += 1

    def owner_fields(self, cid: str, entries: list) -> None:
        """The owner's answers on fields to review: yes takes the value it confirmed while that is still the one
        proposed (else not applied: the page asks again), a value given takes that (both sure, method 'owner'); no
        leaves it to review. Types answered are kept in the chain's tokens (type_tokens: 'defensive minion', a
        minion's 'protective' shield -> defensive_minion), as answered in rulings.json."""
        kind = self.inputs.cards[cid].get("type")
        for e in entries if self.rulings else []:
            if e["kind"] not in ("discord", "old repo"):
                continue
            for key, field in list(e["fields"].items()):
                if key == "ability" and is_sure(field) and "known" in str(field.get("method")):
                    self.keep_transcription(cid, e, key, field)
                if field.get("confidence") != "review":
                    continue
                own = review_id("field", f"{cid}|{e['id']}|{key}")
                proposed = field.get("proposed", self.old_proposed.get(field.get("review") or own))
                for rid in dict.fromkeys(x for x in (field.get("review"), own) if x):
                    r = self.rulings.get(rid) or {}         # a git look's row item, else this entry's own (its
                    value = r.get("value") if r.get("choice") == "value" else None    # answer given before the
                    if r.get("choice") == "yes":                                      # row had an item)
                        if confirms(r, proposed) or key == "types" and isinstance(proposed, list) and \
                                type_tokens(r.get("value"), kind) == type_tokens(proposed, kind):
                            value = r["value"]
                        elif rid not in self.not_applied:   # owner_fields runs twice: counted once
                            self.not_applied.add(rid)
                            self.tally[f"owner: {e['kind']} field {key} yes, its value no longer proposed "
                                       "(not applied)"] += 1
                    if value is not None:
                        value = type_tokens(value, kind) if key == "types" else value     # the chain's tokens
                        e["fields"][key] = sure(value, ruled_by(r), ruled=rid)
                        self.used.add(rid)
                        self.tally[f"owner: {e['kind']} field {key} given"] += 1
                        break

    def keep_transcription(self, cid: str, e: dict, key: str, field: dict) -> None:
        """A text a reader made sure as one of the card's known texts ('ocr+known') keeps the answer given on its
        item before (the entry's own, or its git row's) when that answer has the same words (text_key): the answer
        was typed off the picture, the known text is the PSD's form (spacing, case, punctuation)."""
        row = e["id"][2:] if e["kind"] == "old repo" else ((e.get("look") or {}).get("git") or {}).get("row")
        ids = [review_id("field", f"{cid}|{e['id']}|{key}")] + ([row_item_id(row, key)] if row else [])
        for rid in ids:
            r = self.rulings.get(rid) or {}
            value = r.get("value") if r.get("choice") in ("value", "yes") else None
            if isinstance(value, str) and value != field["value"] and \
                    text_key(dm.strip_tokens(value)) == text_key(dm.strip_tokens(field["value"])):
                e["fields"][key] = sure(value, ruled_by(r), ruled=rid, check=f"same words as {field['method']}")
                self.used.add(rid)
                self.tally[f"owner: {e['kind']} field {key} kept as answered (a known text, same words)"] += 1
                return

    def field_reviews(self, cid: str, entries: list) -> None:
        """Every field to review gets its item (an old row's from oldrows, else the chain's own); types with nothing
        read are 'unread' (nothing to confirm)."""
        for e in entries:
            if e["kind"] not in ("discord", "old repo"):
                continue
            for key, field in e["fields"].items():
                if field.get("confidence") != "review" or field.get("review"):
                    continue
                proposed = field.pop("proposed", None)
                if proposed is None and (key == "types" or e.get("repo") == "first"):
                    field["confidence"] = "unread"             # nothing read (a first-repo PNG with no PSD fetched)
                    continue
                field["review"] = self.add_review(
                    "field", f"{cid}|{e['id']}|{key}", card=cid, entry=e["id"], field=key, date=e.get("date"),
                    proposed=proposed, method=field.get("method"), conf=field.get("conf"), why=field.get("why"))
                self.tally[f"{e['kind']} field to review: {key}"] += 1

    def card(self, cid: str, rows: list, split: list) -> dict:
        card = self.inputs.cards[cid]
        entries = self.discord_entries(cid, rows, split) + self.text_entries(cid) + self.old_entries(cid)
        entries.sort(key=lambda e: (e["date"] is not None, dm.epoch(e["date"]) or 0,
                                    {"discord": 1, "text": 2, "old repo": 3}[e["kind"]], e["id"]))
        weak = {r["id"]: r["no_created"] for r in rows if r.get("no_created")}
        for e in entries:                                # the owner's weak link: shown, sets no Created
            if e["kind"] == "discord" and e.get("pictures") and all(p["id"] in weak for p in e["pictures"]):
                e["no_created"] = weak[e["pictures"][0]["id"]]
        self.no_kinds(cid, entries)
        self.owner_fields(cid, entries)
        self.star_guard(entries)
        self.star_fact_guard(entries)
        self.owner_fields(cid, entries)                  # a star the guards took to review, answered
        self.no_kinds(cid, entries)                      # an answer's types lose an item's kinds too
        entries = self.check_facts(cid, entries)
        self.diff(entries, card.get("type"))
        self.field_reviews(cid, entries)
        stop = None
        if card.get("fs"):
            stop = {"date": dm.iso(card["fs"]), "day": day(card["fs"]),
                    "source": FS_SOURCE.get(card.get("fs_from"), "git")}
            firsts = [v for v in self.inputs.versions.get(cid, []) if abs(v["ts"] - card["fs"]) < 1]
            if card.get("fs_psd"):                        # a PSD added before any PNG (old repo)
                stop.update(commit=card["fs_psd"].get("commit"), path=card["fs_psd"].get("path"), psd=True)
            elif firsts:
                stop.update(commit=firsts[0].get("commit"), path=firsts[0].get("path"))
        out = {"card": cid, "name": card.get("name"), "key": card["key"], "kind": card.get("type"), "stop": stop,
               "entries": entries}
        data = self.inputs.first_data.get(card["key"])
        if stop and stop["source"] == "git" and data:      # the card repo's first version, for the last diff
            fields = data_fields(data)
            last = next((e for e in reversed(entries) if e["kind"] in ("discord", "old repo")
                         and not e.get("meme_copy")), None)
            changes = first_changes(last, fields, card.get("type"), self.inputs.versions.get(cid, []), stop,
                                    data.get("timestamp"))
            if changes.pop("same_title", None):
                self.tally["git_first: no rename (its last look is the first version's, the same printed title)"] += 1
            out["git_first"] = {"from": "card data (its first recorded version)",
                                "date": dm.iso(data.get("timestamp")) if data.get("timestamp") else None,
                                "fields": fields, "changes": changes}
        return out

    # Created
    def created(self, cid: str, timeline: dict, rows_all: list) -> dict:
        """Created = the earliest confident appearance with its source, or "Before 2020" from a physical card photo for
        PHOTO_CREATED only (owner Q6, 11-14 22:20: read only on the review page; dated: the earliest appearance); against
        the file. sort orders "Before 2020" before every day."""
        card, inputs = self.inputs.cards[cid], self.inputs
        found, weak = [], {r["id"] for r in rows_all if r.get("no_created")}     # owner: these set no Created
        for e in timeline["entries"]:
            if e["kind"] == "discord" and not any(p["id"] in weak for p in e["pictures"]):
                found.append((dm.epoch(e["date"]), "discord", {"picture": e["pictures"][0]["id"],
                                                               "channel": e["channel"], "as": e.get("as")}))
            elif e["kind"] == "discord":           # the pictures the owner kept weak date nothing
                for pic in (p for p in e["pictures"] if p["id"] not in weak and p.get("date")):
                    found.append((dm.epoch(pic["date"]), "discord", {"picture": pic["id"], "channel": pic["channel"],
                                                                     "as": e.get("as")}))
            elif e["kind"] == "old repo":           # a first-repo PSD: 'first repo' (psd), else 'old repo (PSD)'
                psd = e.get("psd_only") and e.get("badge") != "first repo"
                found.append((dm.epoch(e["date"]), "old repo (PSD)" if psd else e.get("badge") or "old repo",
                              {"commit": e["commit"], "as": e.get("as"), "psd": True if e.get("psd_only") else None}))
        for row in rows_all:                       # every card-sure picture counts, its look told or not
            if row.get("first") and not row.get("created_review") and \
                    not row.get("joke_draft") and not (row.get("joke") and row["verdict"] != "version_sure") and \
                    not row.get("no_created"):
                found.append((dm.epoch(row["first"]["date"]), "discord", {"picture": row["id"],
                                                                          "channel": row["first"]["channel"],
                                                                          "dates_only": bool(row.get("dates_only"))
                                                                          or None}))
        for row in self.also.get(cid, ()):         # printed plainly on a dates-only picture beside its own card
            if row.get("first") and not row.get("no_created"):
                found.append((dm.epoch(row["first"]["date"]), "discord", {
                    "picture": row["id"], "channel": row["first"]["channel"], "dates_only": True,
                    "beside": row.get("card")}))
        if card["key"] in getattr(inputs, "heads", {}):    # a 2019 list's head, its first picture keeping the name
            for f in inputs.facts:
                if f.get("card") == card["key"] and f.get("source") == "changelog" and \
                        search(r":(img|doc)\d", f.get("line", "")) and \
                        (f.get("confidence") == "sure" or f.get("unit") == "unknown") and dm.epoch(f.get("date")):
                    found.append((dm.epoch(f["date"]), "changelog", {"fact": f["id"], "channel": f.get("channel")}))
        first = (getattr(inputs, "psd_first", None) or {}).get(card["key"])
        if first and first.get("ts"):              # a PSD added before any PNG: a git appearance (S6)
            found.append((first["ts"], "first repo" if first.get("repo") == "first" else "old repo (PSD)",
                          {"commit": first.get("commit"), "psd": True}))
        if card.get("fs"):
            psd = "first repo" if card.get("fs_from") == "first git" else "old repo (PSD)"   # the first repo's: as psd_first
            found.append((card["fs"], psd if card.get("fs_psd") else FS_SOURCE.get(card.get("fs_from"), "git"), {}))
        when, source, detail = min(found, key=lambda f: (f[0], f[1]))
        dated = {"date": dm.iso(when), "day": day(when), "source": source,
                 **{k: v for k, v in detail.items() if v is not None}}
        photo = cid in PHOTO_CREATED              # the owner's ruling: no picture or entry of the photo is used
        ours = {"date": None, "day": BEFORE_DAY, "source": PHOTO_SOURCE, "owner": "Q6"} if photo else dated
        key = BEFORE_SORT if photo else ours["day"]
        file = inputs.created_file.get(card["key"])
        out = {"ours": ours, "file": file}
        if photo:
            out["dated"] = dated                   # the earliest dated appearance, for reference
        if file and file.get("from") == "hand":
            out.update(status="hand", date=file["date"], proposed=False)
        elif not file or not file.get("date"):
            out.update(status="new", date=ours["day"], proposed=True)
        elif ours["day"] == file["date"]:
            out.update(status="same", date=file["date"], proposed=False)
        elif key < file["date"]:
            out.update(status="earlier", date=ours["day"], proposed=True)
        else:
            out.update(status="later", date=file["date"], proposed=False)
        out["sort"] = BEFORE_SORT if out["date"] == BEFORE_DAY else out["date"]
        if out["status"] in ("earlier", "later", "new"):
            hint = None if photo else self.maybe.get(cid)
            hint = {"picture": hint["id"], "date": hint["first"]["date"], "channel": hint["first"]["channel"]} \
                if hint and dm.epoch(hint["first"]["date"]) < when else None
            why = {"earlier": "an earlier confident appearance: proposed (the preview uses it)",
                   "later": "our earliest is later than the file's date: the file's kept (never moved later)",
                   "new": "no date in the file yet: proposed"}[out["status"]]
            if photo and out["status"] != "later":
                why = f"the owner's ruling (Q6, 11-14 22:20): Created '{BEFORE_DAY}' from its physical card photo " \
                      "(the photo itself is in no timeline); the preview uses it"
            elif ours.get("source") == "changelog" and out["status"] != "later":
                why = "a 2019 balance list names it exactly (its first picture keeps that name, by eye): proposed"
            elif ours.get("source") == "first repo" and out["status"] != "later":
                why = "the first repo's file (2022-08..2023-04, refound on GitHub): proposed"
            out["review"] = self.add_review(f"created {out['status']}", cid, card=cid, ours=ours, file=file,
                                            earlier_unsure=hint, why=why, **({"read_only": True, "answer": "Q6"}
                                                                             if photo else {}))
        named = [r for r in rows_all if r.get("created_review") and r.get("first")]
        if named:
            row = min(named, key=lambda r: dm.epoch(r["first"]["date"]))
            if day(dm.epoch(row["first"]["date"])) < key:
                out["named_only_earlier"] = {"day": day(dm.epoch(row["first"]["date"])), "picture": row["id"],
                                             "review": picture_id("named only", row["id"], cid,
                                                                  getattr(self.inputs, "issued", {}))}
        if out["status"] == "hand":                # never touched: no hint to answer
            return out
        pick, base = self.owner_pick(out)
        before = created_sort(base[0])            # a hint only when it is earlier than the Created date in use
        facts = [f for f in inputs.facts if f.get("card") == card["key"] and f.get("source") == "changelog"
                 and f.get("confidence") == "sure" and dm.epoch(f.get("date"))]
        if facts:
            fact = min(facts, key=lambda f: dm.epoch(f["date"]))
            if day(dm.epoch(fact["date"])) < before:
                out["changelog_earlier"] = {"day": day(dm.epoch(fact["date"])), "fact": fact["id"],
                                            "channel": fact.get("channel"),
                                            "review": self.add_review("changelog earlier", cid, card=cid,
                                                                      fact=fact["id"], date=fact["date"],
                                                                      why="a sure changelog line names it before "
                                                                          "its Created date")}
        pushes = [r for r in inputs.first_repo if r.get("card") == card["key"] and dm.epoch(r.get("date"))]
        if pushes:
            push = min(pushes, key=lambda r: dm.epoch(r["date"]))
            if day(dm.epoch(push["date"])) < before:
                out["first_repo_earlier"] = {"day": day(dm.epoch(push["date"])), "row": "Q-" + push["id"],
                                             "review": self.add_review("first repo earlier", cid, card=cid,
                                                                       row="Q-" + push["id"], date=push["date"],
                                                                       why="a first-repo push notice names it "
                                                                           "before its Created date")}
        self.owner_created(cid, out, pick, base)
        return out

    def owner_pick(self, out: dict) -> tuple:
        """(the owner's answer on the card's Created item, (the date it gives, its source)): ours or the file's while
        that is still the day the answer carries, or a date given; the owner's (not the triage's) pick whose day is
        shown no more keeps its day until answered again ('held': never dropped); else the date as built."""
        r = (self.rulings.get(out["review"]) or {}) if out.get("review") else {}
        file, ours = out.get("file") or {}, out["ours"]
        given = {"ours": (ours.get("day"), {**ours, "from": "ours"}),
                 "file": (file.get("date"), {"day": file.get("date"), "source": "website", "from": "file",
                                             "file_from": file.get("from")}),
                 "date": (r.get("value"), {"day": r.get("value"), "source": "owner", "from": "typed"})}
        if r.get("choice") in ("ours", "file") and not confirms(r, given[r["choice"]][0]):
            if not r.get("source") or r["source"] == "owner":       # the owner's day holds until answered again
                self.tally["owner: Created picked, its day no longer the one shown (kept as the owner's day)"] += 1
                held = {"day": r.get("value"), "source": "owner", "from": "held", "pick": r["choice"]}
                return {**r, "held": True}, (r.get("value"), held)
            self.tally["owner: Created picked, its day no longer the one shown (not applied)"] += 1
        elif r.get("choice") in given and given[r["choice"]][0]:
            return r, given[r["choice"]]
        if out["date"] == ours.get("day"):
            return {}, (out["date"], {**ours, "from": "ours"})
        return {}, (out["date"], {"day": out["date"], "source": "website", "from": "file",
                                  "file_from": file.get("from")})

    def owner_created(self, cid: str, out: dict, r: dict, base: tuple) -> None:
        """The owner's answers on Created: a status item's pick (base: ours, the file's, a date given) and the earlier
        hints said yes to (a changelog line, a first-repo push notice, a Morgue entry's picture); the earliest wins and
        its source is kept (owner.source, shaped like ours: the preview's source for status 'owner'). 'hand' is never
        touched; a hint not earlier than base is never one (the review page leaves it out), nor one the triage said
        yes to below the owner's own pick, held or not (owner > triage: Holy Card's 2023-05-31, picked over that very
        picture; Standstill's 2020-07-22, picked before its 2019 hint was asked): that hint is the owner's to answer."""
        if not self.rulings:
            return
        ruled = [out["review"]] if r else []
        picks = [base]
        mine = bool(r) and r.get("source") in (None, "owner")    # the owner's own pick (held too): no triage hint below
        lower = lambda h: mine and h.get("source") not in (None, "owner")
        for k, source, what in (("changelog_earlier", "changelog", "changelog line"),
                                ("first_repo_earlier", "first repo", "first repo push")):
            hint = out.get(k)
            answer = (self.rulings.get(hint["review"]) or {}) if hint else {}
            if lower(answer) and answer.get("choice") == "yes":
                self.tally[f"owner: a triage {what} hint's yes below the owner's day (not applied)"] += 1
                continue
            if answer.get("choice") == "yes" and not confirms(answer, hint["day"]):     # another line's day now
                self.tally[f"owner: a {what} hint's yes, its day changed (not applied)"] += 1
                continue
            if answer.get("choice") in ("yes", "no"):
                ruled.append(hint["review"])
                self.used.add(hint["review"])
            if answer.get("choice") == "yes":
                picks.append((hint["day"], {"day": hint["day"], "source": source, "from": what,
                                            **{x: hint[x] for x in ("fact", "row", "channel") if hint.get(x)}}))
        for hid, h in sorted(self.rulings.items()):
            item = h.get("item") or {}
            if h.get("kind") != "morgue hint" or item.get("card") != cid or item.get("meme"):
                continue
            when = item.get("day") or str(item.get("date") or "")[:10]
            if not when or created_sort(when) >= created_sort(base[0]) or h.get("choice") not in ("yes", "no"):
                continue
            if lower(h) and h.get("choice") == "yes":
                self.tally["owner: a triage Morgue hint's yes below the owner's day (not applied)"] += 1
                continue
            ruled.append(hid)
            self.used.add(hid)
            if h.get("choice") == "yes":
                picks.append((when, {"day": when, "source": "discord",
                                     "from": "morgue picture", "entry": item.get("entry"),
                                     **{x: item[x] for x in ("picture", "channel") if item.get(x)}}))
        date, source = min(picks, key=lambda p: created_sort(p[0]))
        if r or date != out["date"]:
            if r:
                self.used.add(out["review"])
            file = (out.get("file") or {}).get("date")
            out.update(status="owner", date=date, sort=created_sort(date), proposed=date != file,
                       owner={"was": out["status"], "pick": r.get("choice"), "base": base[0],
                              "hints": len(picks) - 1, "source": source, "ruled": ruled,
                              **({"held": r["value"]} if r.get("held") else {})})
            self.tally["owner: Created picked"] += 1


def build(root: str, cards_repo: str = LOCAL_DIR_LOC, log: Callable = print) -> dict:
    root = dm.check_root(root)
    paths = Paths(root)
    inputs = Inputs(root, cards_repo)
    owned = owner_rows(inputs)
    owned["discord: named only on or after the file's Created day (enters)"] = named_on_or_after(inputs)
    chain = Chain(inputs)
    chain.tally.update(owned)
    missing = photo_missing(inputs.cards)
    for m in missing:
        log(f"WARNING: {PHOTO_SOURCE} Created: no card {m} (owner Q6): its 'Before 2020' lapsed")
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
        how = next(({k: i["split"][k] for k in ("halves", "matched_by", "readings", "eye") if k in i["split"]}
                    for i in inputs.old_review if (i.get("split") or {}).get("halves")
                    and i.get("old_id") == group["split_of"]["name"]), None)
        chain.add_review("split", group["id"], name=group["split_of"]["name"], halves=halves, group=group["id"],
                         picked=how,
                         entries={h: sum(1 for e in timelines[h]["entries"] if e.get("as")) for h in halves
                                  if h in timelines},
                         why="an old card that became two (owner Q3): both halves start with its versions, each "
                             "with its own half's stats")
    for q, answer, when, text in OWNER_ANSWERS:
        chain.add_review("owner's answer", q, question=q, answer=answer, ruling=text, read_only=True, when=when,
                         why=f"the owner's answer ({when})")
    for n, a in enumerate(inputs.owner.get("answers") or [], 1):   # later answers (owner.json), as given
        chain.add_review("owner's answer", a["id"], question=a["title"], answer=a["answer"], ruling=a["ruling"],
                         read_only=True, order=n, when=a.get("when"), why=f"the owner's answer ({a.get('when')})")
    dm.write_json(paths.created, {
        "about": "Created dates (STAGE1 S6): ours = the earliest confident appearance (a card-sure Discord picture, "
                 "the first repo, the old repo, the card repo; 'Before 2020' from a physical card photo only for Pep "
                 "Drink and Wham Bam Bone, owner Q6 11-14, dated: their earliest appearance) with its "
                 "source; status against site/Card Creation Dates.json: same, earlier (proposed), later (review only, "
                 "the file's date kept), new (proposed), hand (untouched). date = what the preview uses, sort orders "
                 "'Before 2020' first. Named-only pictures, changelog lines and first-repo push notices before it are "
                 "review only.",
        "generated": dm.now_iso(), "cards": created})
    if inputs.rulings:
        dm.write_json(paths.ruled, {"about": "The review page's answers (derived/review/rulings.json) this build acted "
                                             "on: an answer whose item left the page is 'taken in' only when listed "
                                             "here, else its item changed.",
                                    "generated": dm.now_iso(), "used": sorted(chain.used)})
    elif exists(paths.ruled):
        remove(paths.ruled)
    items = sorted(chain.review.values(), key=lambda i: (i["kind"], i["id"]))
    dm.write_text(paths.review, "".join(dumps(i, ensure_ascii=False) + "\n" for i in items))
    text = stats_text(timelines, created, items, chain.tally, kinds, missing)
    dm.write_text(paths.stats, text)
    log(text)
    return {"timelines": len(timelines), "review": len(items), "entries": dict(kinds), "photo_missing": missing}


def _listdir(path: str) -> list:
    from os import listdir
    return listdir(path) if exists(path) else []


def stats_text(timelines: dict, created: dict, items: list, tally: Counter, kinds: Counter, missing=()) -> str:
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
    firsts = [sum(e.get("repo") == "first" for e in t["entries"]) for t in timelines.values()]
    lines.append(f"  the first repo's {sum(firsts)} (cards {sum(1 for n in firsts if n)})")
    texts = [e for t in timelines.values() for e in t["entries"] if e["kind"] == "text"]
    lines.append(f"text entries {len(texts)}: " + ", ".join(
        f"{k} {n}" for k, n in Counter(e["change"].get("field") or "ability" for e in texts).most_common()) +
        f"; confirmed by a picture {sum(1 for e in texts if e.get('confirmed_by'))}")
    bal = Counter(e["changes"].get("bal") for t in timelines.values() for e in t["entries"] if e.get("changes"))
    lines.append("buff/nerf (any version vs the one before): " + ", ".join(f"{k} {n}" for k, n in bal.items() if k))
    ab = Counter(e["changes"].get("ability_change") for t in timelines.values() for e in t["entries"]
                 if e.get("changes") and e["kind"] == "discord")
    unread = sum(1 for t in timelines.values() for e in t["entries"] for f in (e.get("fields") or {}).values()
                 if f.get("confidence") == "unread")
    lines.append("discord ability changes: " + ", ".join(f"{k} {n}" for k, n in ab.most_common()))
    lines.append(f"fields unread (types, nothing read to confirm): {unread}")
    lines.append("fields not collected (an item's kinds alone): " + str(sum(
        1 for t in timelines.values() for e in t["entries"] for f in (e.get("fields") or {}).values()
        if f.get("confidence") == "not collected")))
    status = Counter(c["status"] for c in created.values())
    sources = Counter(c["ours"]["source"] for c in created.values())
    lines += ["", "CREATED: " + ", ".join(f"{k} {n}" for k, n in status.most_common()),
              "  ours from: " + ", ".join(f"{k} {n}" for k, n in sources.most_common()),
              f"  named-only picture earlier (review): {sum(1 for c in created.values() if c.get('named_only_earlier'))}",
              f"  changelog line earlier (review): {sum(1 for c in created.values() if c.get('changelog_earlier'))}",
              f"  first repo push earlier (review): {sum(1 for c in created.values() if c.get('first_repo_earlier'))}"]
    for key, c in sorted(created.items()):
        if c["ours"]["source"] == PHOTO_SOURCE:
            lines.append(f"  {PHOTO_SOURCE} Created: {key}: {c['status']} {c['date']} (file "
                         f"{(c.get('file') or {}).get('date')} {(c.get('file') or {}).get('from')})")
    lines.append(f"  {PHOTO_SOURCE} Created, owner's cards missing (lapsed): {', '.join(missing) or 'none'}")
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
        line = f"TEXT {e['day']} {e['channel']}: {dumps(e['change'])}" + \
            (f" confirmed by {e['confirmed_by']}" if e.get('confirmed_by') else "")
        draw.text((8, 32 + 18 * i), line[:160], fill="navy", font=small)
    for i, e in enumerate(versions):
        x, y = (i % cols) * tile_w, head + (i // cols) * tile_h
        img = entry_image(e, root)
        if img is not None:
            img = img.convert("RGB")
            img.thumbnail((side, side * 2))
            sheet.paste(img, (x + 4, y + 4))
        when = e.get("day") or e.get("label")
        lines = [f"#{i + 1} {e['kind']} {when} {e.get('channel') or e.get('subject') or ''}"[:60],
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
