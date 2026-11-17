"""
One verdict per Discord picture, from the matcher's scores (commands/discord_match.py score): the card timelines' match
step (docs/Future/CARD_TIMELINE.md, stage 2). Run after `discord_match score`, with a Python that has Pillow and numpy:

    python -m commands.discord_verdicts build  [--root <dir>] [--cards-repo <dir>]
    python -m commands.discord_verdicts checks [--root <dir>] [--seed <n>] [--sure <n>] [--lost <n>]
    python -m commands.discord_verdicts selftest [--root <dir>] [--cards-repo <dir>]
    python -m commands.discord_verdicts eyekeys [--root <dir>] [--cards-repo <dir>]
    python -m commands.discord_verdicts local [--root <dir>] [--cards-repo <dir>] [--workers <n>]
    python -m commands.discord_verdicts localtest [--root <dir>] [--cards-repo <dir>] [--workers <n>] [--sample <n>]

build   A picture is one stored file (its sha256: every post of it, reposts in other channels too; the earliest post
        dates it); a sheet's cells are pictures of their own. Each gets one verdict:
          version_sure    the card, and which of its git looks it shows
          version_unsure  the card; its look can't be told (a screenshot, a crop, looks alike)
          draft           the card, in a look no git version has (a version before git, or a draft)
          lost            a card picture naming no live card (grouped into lost cards)
          morgue          a joke: posted in a meme channel (or the same picture as one) and no live card's
        and outside those: unsure (which card isn't sure: review), art (a card's art only), none (no card).
        Card sure is the matcher's confident verdict at REVIEW or more (a screenshot also needs the card's art or look
        behind it, a look-alike family the card's own title; a card named only, by its title, file name or chat, needs
        its text to fit the card and no card the matcher saw in it to fit better: name_check). A git look needs the
        picture's timing tags and numbers to be that version's, its text exact (any length, hash picks too) if made
        before the look's first git version, and its bars read the same if made more than GRACE_DAYS before
        (look_check). An embed's picture never dates (fetched when archived). A card named only, dated before the
        card's first picture evidence, sets Created only through review (created_review). Joke pictures (owner Q1=C:
        meme-cards, joke-ideas, cursed, idea-graveyard; ai-images and patty-vault are never read) never date a live
        card; one that is a live card stays its card, or goes to review when not sure (named so by its posts outside
        meme channels, or a git version's look and text: MEME_LIVE), never the Morgue; a meme that is no card is
        none. The physical card photos ('Old stuff' thread, owner Q6 11-14) are never read (discord_match.LEFT_OUT).
        Rules pages and app screenshots are never lost cards.
        Lost pictures are grouped into lost cards by title, else by look (ids from the earliest picture's hash and
        cell), each with its first date, channels, whether it was played (port, update or classification channels, an
        update changelog naming it as a card, or the old repo's file; owner Q2=A: Morgue candidates when a card's stats
        or template show), a split of an old card (owner Q3=A: review) and its nearest live cards.
checks  Contact sheets for the by-eye checks: random sure matches, every match of the look-alike families, random lost
        pictures (post | the matched version).
selftest The sure error of these gates on discord_match selftest's degraded copies (cards and looks).
eyekeys Every eye ruling naming a git version beside that version's whole blob (the old repo's, the card repo's when
        stored, else its thumb): the largest local difference and zooms of the 3 most different areas, for the eye
        before a version key is accepted (a word, an icon or a medal hides in a whole-card score). checks/eye-keys/.
local   Each card-sure picture against every stored version of its card (eyekeys' local diff, the nearest two again
        aligned; aspect ratios kept): derived/match/local.jsonl, cached; build's local gate reads it (decide).
localtest The local gate's bars on Discord and small copies of every stored version: derived/match/local_selftest.json.
Outputs under the archive's derived/match/ (never a repo): verdicts.jsonl, lost_groups.json, sheets/ (sheets.json and
each sheet's cells drawn), checks/. Ids are stable: P-<sha256 12> (a cell: -c<n>), LC-<sha256 10> (-c<n>). No message
text, posters or authors anywhere (owner 2026-11-13).
"""
import hashlib
from argparse import ArgumentParser
from collections import Counter, defaultdict
from json import loads
from os import environ, makedirs
from os.path import basename, dirname, join, splitext
from random import Random
from re import compile as re_compile, sub
from sys import stderr
from typing import Callable, Optional

from commands import discord_match as dm
from global_config import DATA_BRANCH

# ========================
# Configuration
# ========================
OUT = ("derived", "match")
CATALOGUE_PICTURES = ("derived", "catalogue", "pictures.jsonl")
OLD_LOST = ("derived", "oldgit", "lost.json")
OLD_FILES = (("derived", "oldgit", "versions.jsonl"), ("derived", "oldgit", "reads.jsonl"))   # the repos' files, looks
OLD_MAP = ("derived", "oldgit", "map.json")
MORGUE_DIR = "MDW/The Morgue/"                   # build_site MORGUE: retired cards
PLAYED_KINDS = {"port", "update"}               # the catalogue's kinds whose pictures were played (owner Q2=A)
PLAYED_NAMES = {"card-classification", "minion classification"}
SHOWN_FIRST = re_compile(r"port|update-\d+-port|card-classification|ideas-discussion|new-cards")   # a look shown
                                                # there before its meme repost is the card's look, not a joke
MORGUE_EYE = ("derived", "morgue", "eye.json")
JOKE_ASKED = {"zero-two": "the Morgue's anime Zero Two", "fireball": "Fireball (Mario)",
              "mr-softener": "the joke edit Mrs. Softener", "nme-stock2": "NME Stock"}   # open owner questions: kept
CHANGELOG_KINDS = {"update"}
CARD_CLASSES = {"ps1200", "gen400", "gen350", "shot", "cardshape", "photo"}   # the catalogue's card picture sizes
SURE_CLASSES = ("version_sure", "version_unsure", "draft")
REVIEW = {"version_unsure": "version", "draft": "a look not in git", "lost": "lost card", "morgue": "morgue",
          "unsure": "card"}
SURE = dm.REVIEW                                # a card's confidence for sure (the matcher's confirm line)
SURE_BITS, SURE_FIT, SURE_NOVEL = 3.0, 0.9, 8   # a git look sure: the whole card this near it; a text pick this
                                                # much of its text read, fewer characters in a row not its
SURE_MISSING = 6                                # ... and fewer characters in a row of its steady text unread
FAR_RUN, FAR_BITS, FAR_FIT = 12, dm.DRAFT_BITS, 0.75   # a draft by text: the whole card this many bits from every
                                                # git version, and this many characters in a row no git look has or
                                                # under this share of its text in the looks left
CLASH_BITS = 1.5                                # a text pick's look this much farther by hash than another's: unsure
GRACE_DAYS = 3                                  # a git look posted more days than this before its first git version:
                                                # sure only with its bars read the same (the hash can't see a bar)
EXACT_RUN = 2                                   # a git look posted before its first git version: fewer characters in
                                                # a row than this its own or that version's, at any length, hash
                                                # picks too (8 of 9 such runs were a typo or a word fixed before the
                                                # commit; Halo, Glued: short texts a hash picked)
STATS = ("hp", "def", "atk", "spd")             # the bars, top to bottom
BARS = ("derived", "readers", "bars", "reads.jsonl")   # the era bar readers' reads: Discord pictures, old repo PNGs
NO_BARS = {"field", "buff", "debuff"}           # card types that print no stat bars (and any card whose data
                                                # stats are all -1: no_stats)
OLD_ERAS = {"A", "B", "C", "A-item", "B-item", "C-item"}   # templates before the card template (2022-06)
OLD_BITS = 8.0                                  # an older template's picture this far from every git version: a draft
EXPORT_SIZES = {"ps1200", "gen400"}             # whole card exports: copies of a git version stay under OLD_BITS of it
CARD_ASPECTS, WHOLE_ASPECT = (1 / 2, 2 / 3, 1.0), 0.03   # card exports' w/h; a cut screenshot is 5%+ off
TEXT_READS = ("derived", "readers", "text", "reads.jsonl")  # the text readers' era and stars per Discord picture
TEXT_TRUTH = ("derived", "readers", "text", "truth.jsonl")
TEXT_CAL, S_STARS_BAR = ("derived", "readers", "text", "calibration.json"), "S/today"   # status cards' star bar
OLD_ROWS = ("derived", "oldrows", "rows.jsonl")     # old repo versions' fields (their stars)
EYE = ("derived", "match", "eye.json")               # pictures told by eye: by_eye
OWNER = ("derived", "review", "owner.json")          # the owner's rulings outside the review page (its pictures:
                                                     # by_eye's shape, applied after the eye; kept apart from eye.json;
                                                     # ruled_out: {picture: {cards}} left out of its maybe list)
EYE_W, EYE_BOX, EYE_BLUR = 600, 15, 1.2              # eyekeys: compared this wide, windows this big, both blurred
LOCAL = ("derived", "match", "local.jsonl")          # card-sure pictures' local differences from their stored versions
LOCAL_SAME, LOCAL_KEEP, LOCAL_DRAFT = 5.0, 10.0, 30.0   # local diff: that version's look / over it a gate's look is
LOCAL_AR = 0.01                                      # unsure / from every stored version a draft; aspect within 1%
LOCAL_WIDE, LOCAL_DRAFT_NARROW = 800, 45.0          # a picture under this wide: drafted from this far (small copies)
LOCAL_ALIGN, LOCAL_MARGIN = 2, 12                    # the nearest versions scored again aligned (a few px cropped or
                                                     # padded: Miles Edgeworth's 598x1198 copy); the border left out
STATS_FILE, OLD_STATS_FILE = dm.STATS_FILE, "db/old_stats.json"
TAG = re_compile(r"\(([0-9/]*[A-Z][A-Z0-9/]{1,5})\)|\(([0-9/]*[A-Z][A-Z0-9/]{1,5})\b|\b([0-9/]*[A-Z][A-Z0-9/]{1,5})\)")
NUMBER = re_compile(r"(?<![a-z0-9])(\d+(?:[.,]\d+)?%?)\s*([a-z]?)")   # a number and the next word's first letter
SHOT_LOOK = 0.5                                 # a screenshot: its hash or art anchor at least this
FAMILY_TITLE = 0.9                              # a look-alike family: the card's own title read this well
LOOKALIKE_BITS = 4.0                            # cards with git versions this near: one family, no twin shortcut
LOST_TITLE = 4                                  # a lost picture's title: at least this many letters
LETTERS = 0.6                                   # a card picture's text: at least this share letters (not digits)
FOREIGN = re_compile(r"(?i)\b(atk|def)\s*/\s*[\d?]|\blink-\d|takahashi|konami|\bpokemon\b|\d+\s*stamina\b")  # other
                                                # games' cards
APP = re_compile(r"(?i)\.(png|psd|jsx|ya?ml|json|lua)\b|pull requests|brush settings|find and replace|visual studio|"
                 r"resolve conflicts|conflicted files|website only")    # screenshots of apps (git, Photoshop, editor)
RULE_CHAPTERS = ("Table of Contents", "Getting Started", "Abbreviations", "Setting Up", "Table", "Shops",
                 "Reading Cards", "Stats", "The Global Turn", "Taking Your Turn", "Attacking", "Abilities", "Reactions",
                 "Items", "Item Types", "Minions", "The Field", "Death", "Field Cards", "Status Effects",
                 "Creature Types", "Damage Types", "Negation", "Timing Tags")   # the rulebook's (db/rulebook.json)
TAG_RULE = re_compile(r"\([A-Z0-9/]{2,6}\)\s*=|\bimpl(y|ies)\b")   # a rules page defining timing tags
STAT_LABEL = re_compile(r"^(health|heaith|hp|defen[cs]e|defemse|def|attack|atk|speed|spd)$")
CARD_ERAS = {"A", "B", "C", "D1", "D2", "D3", "R", "S", "F", "A-item", "B-item", "C-item", "D-item", "bars-other"}
COMMON_WORD = 100                               # a one-word title in more messages than this: chat, not a card name
SPLITS = {"Twin Princes": ("lothric", "lorian")}    # an old card that became two (owner Q3=A: review, never Morgue)
GAME = set("creature creatures attack attacks attacking attacked attacker attackers turn turns equip equipped equips "
           "summon summons summoned damage dmg item items minion minions field ability abilities health defense "
           "defence speed atk def spd hp point points enemy enemies friendly target targeted targets stat stats "
           "piercing inflict inflicts heal heals healing card cards slot intercept protective star stars cd opt oyt "
           "eot aat tpt bogt endt lane debuff buff negate negated".split())
GAME_WORDS = 2                                  # a lost card picture's text: at least this many of the game's words
TEMPLATE_SIZES = {"ps1200", "gen400", "gen350"}  # ... or one of the card templates' export sizes
OLD_NAME = 0.9                                  # a lost title this near an old repo name of a live card: renamed?
GROUP_TITLE = 0.85                              # lost pictures one card: titles this alike
FAMILIES = (("dull-knife", "rusty-knife", "red-knife", "shiny-knife"), ("moss-charger", "massive-moss-charger"))
FAMILY_PREFIXES = ("pod",)                      # PoD Bloons: every card id starting so is one family
FAMILY_TAINTED = "tainted-"                     # the Tainted cards (one frame) and any plain X
NAME_METHODS = frozenset(dm.EVIDENCE["name"])   # a card named only: its title, a line, the file name, chat
NAME_FIT = 0.25                                 # ... sure only with this version fit (all 7 named-only rows of another
                                                # card read under it: Potato, Tentacle, Firebomb)
MEME_LIVE = "a live card's picture in a meme channel (or named as one)"     # review, never a Morgue candidate
LOST_ABOUT = ("Lost cards: lost and joke pictures grouped by title, else by look; id from the earliest picture's "
              "sha256 (and cell). morgue_candidate: a joke card, or played (port, update or classification channels, "
              "an update changelog naming it as a card, the old repo's file; owner Q2=A) with a card's stats or "
              "template (card_evidence); split: an old card that became two (owner Q3=A, review); ideas_never_ported: "
              "the rest (review page). No message text or posters.")


# ========================
# Helpers
# ========================
class Paths:
    def __init__(self, root: str):
        self.root, self.out = root, join(root, *OUT)
        self.verdicts, self.lost = join(self.out, "verdicts.jsonl"), join(self.out, "lost_groups.json")
        self.sheets, self.checks = join(self.out, "sheets"), join(self.out, "checks")
        self.matcher = dm.Paths(root)


def picture_id(sha: str, cell: Optional[int] = None) -> str:
    return f"P-{sha[:12]}" + (f"-c{cell}" if cell is not None else "")


def group_id(sha: str, cell: Optional[int] = None) -> str:
    """A lost card's id from its earliest picture (a sheet's cell: its number too)."""
    return f"LC-{sha[:10]}" + (f"-c{cell}" if cell is not None else "")


def lookalikes(cards: dict, versions) -> dict:
    """{card: [cards]}: unrelated cards with git versions within LOOKALIKE_BITS of each other (the hash can't tell
    them apart: Red and Shiny Knife are 1 bit apart)."""
    found = defaultdict(set)
    refs = [v for v in versions if v.get("whole") and v.get("card") in cards]
    for i, a in enumerate(refs):
        for b in refs[i + 1:]:
            if a["card"] != b["card"] and dm.hash_distance(a["whole"], b["whole"]) <= LOOKALIKE_BITS and \
                    not dm.related(cards, a["card"], b["card"]):
                found[a["card"]].add(b["card"])
                found[b["card"]].add(a["card"])
    return {cid: sorted(others) for cid, others in found.items()}


def families(cards: dict, close: Optional[dict] = None) -> dict:
    """{card: family name}: look-alike cards (FAMILIES, PoD Bloons, the Tainted cards with any plain X, and cards
    whose git versions look alike: lookalikes)."""
    found = {}
    for members in FAMILIES:
        for cid in members:
            if cid in cards:
                found[cid] = members[0]
    for cid, others in sorted((close or {}).items()):
        name = next((found[c] for c in [cid, *others] if c in found), cid)
        for c in [cid, *others]:
            found.setdefault(c, name)
    for cid in cards:
        if any(cid.startswith(p) and cid[len(p):len(p) + 1].isdigit() for p in FAMILY_PREFIXES):
            found[cid] = "pod-bloons"
        if cid.startswith(FAMILY_TAINTED):      # one frame for all; their plain cards too, if any
            found[cid] = "tainted"
            if cid[len(FAMILY_TAINTED):] in cards:
                found[cid[len(FAMILY_TAINTED):]] = "tainted"
    return found


def catalogue_classes(root: str) -> dict:
    """{sha256: size class} from the channel pass."""
    return {splitext(basename(r.get("file") or ""))[0]: r.get("size_class")
            for r in dm.read_jsonl(join(root, *CATALOGUE_PICTURES))}


def channel_kinds(root: str) -> dict:
    """{channel or thread id: catalogue kind}."""
    data = dm.read_json(join(root, *dm.CATALOGUE), {}) or {}
    rows = data.get("channels") if isinstance(data, dict) else data
    return {str(r.get("id")): r.get("kind") for r in rows or [] if isinstance(r, dict)}


def played_place(record: dict, kinds: dict, channels: dict) -> bool:
    """A port, update or classification channel (or thread): its pictures were played."""
    places = [record.get("channel_id"), (record.get("thread") or {}).get("id")]
    return any(kinds.get(p) in PLAYED_KINDS or (channels.get(p) or {}).get("name") in PLAYED_NAMES or
               ((record.get("thread") or {}).get("name") or "").lower() in PLAYED_NAMES for p in places if p)


def post_row(record: dict, morgue: set, kind: Optional[str] = None) -> dict:
    """A post without its text or poster (kind: the picture's, an attachment or an embed)."""
    thread = record.get("thread") or {}
    return {"message": record["id"], "channel_id": record.get("channel_id"), "channel": record.get("channel_name"),
            "thread": thread.get("name"), "date": record.get("timestamp"), "jump_url": record.get("jump_url"),
            "bot": dm.is_bot(record), "morgue": dm.in_morgue(record, morgue), "embed": kind == "embed"}


def dating(post: dict) -> bool:
    """A post that may date a live card: no bot's, no meme channel's, no embed's (fetched when archived: it may be a
    later picture than the post's)."""
    return not (post["bot"] or post["morgue"] or post.get("embed"))


# ========================
# Post verdicts (as discord_match.build)
# ========================
def post_verdicts(images: dict, records: dict, cards: dict, rulings: dict, registry: dict, morgue: set) -> tuple:
    """(verdicts, raw): each post's verdict as the matcher's build gives it (rulings applied, morgue posts set
    apart, twins of sure posts made sure), and its verdict before the morgue rule."""
    verdicts = {key: dm.judge(image, cards) for key, image in images.items() if image["message"] in records}
    by_id = {aid: key for key, aid in (registry.get("ids") or {}).items()}
    for aid, ruling in rulings.items():
        key = by_id.get(aid) or ruling.get("key")
        if key in verdicts:
            verdicts[key].update(status="ruled", cards=[c for c in ruling["cards"] if c in cards], maybe=[],
                                 reason=f"ruled {aid}: {ruling.get('text')}")
    raw = {key: {k: v.get(k) for k in ("status", "cards", "maybe", "scored", "reason")} for key, v in verdicts.items()}
    for key, v in verdicts.items():
        if dm.in_morgue(records[images[key]["message"]], morgue):
            v.update(was=v["status"], looks_like=(v["cards"] or v["maybe"])[:3], status="morgue", cards=[], maybe=[])
    dm.same_image(images, verdicts, cards)
    return verdicts, raw


def scored_of(v: dict, cid: str) -> Optional[dict]:
    return next((c for c in v.get("scored") or [] if c["card"] == cid), None)


def sure_cards(vs: list, cards: dict) -> dict:
    """{card: (confidence, verdict)} of the confident or ruled verdicts, the best per card."""
    found = {}
    for v in vs:
        if v["status"] not in ("confident", "ruled"):
            continue
        for cid in v["cards"]:
            conf = 1.0 if v["status"] == "ruled" else (scored_of(v, cid) or {}).get("conf", 0.0)
            if conf > found.get(cid, (-1.0, None))[0]:
                found[cid] = (conf, v)
    return found


def unrelated(found: dict, cards: dict) -> list:
    """found's cards, a duplicate of a better one dropped (Gust of Trees shares Gust of Wind's picture)."""
    kept = []
    for cid in sorted(found, key=lambda c: -found[c][0]):
        if not any(dm.related(cards, cid, k) for k in kept):
            kept.append(cid)
    return kept


def gate(cid: str, conf: float, v: dict, shape: Optional[str], family: dict, close=(), whole: bool = False) -> \
        Optional[str]:
    """Why a confident card isn't sure (None: sure): under SURE (a whole card export named only goes to name_check
    instead: a title read's weight is under SURE; a cut copy's title can lose a word, Ancient Stopwatch);
    a screenshot with nothing of the card's look;
    a look-alike family's card without its own title (the same image as a sure post will do, unless another card's
    git versions look the same: close)."""
    if v["status"] == "ruled":
        return None
    s = (scored_of(v, cid) or {}).get("s") or {}
    if conf < SURE and not (whole and s and set(s) <= NAME_METHODS):
        return f"confidence {conf:.3f}"
    if not shape and max(s.get("hash", 0), s.get("anchor", 0)) < SHOT_LOOK:
        return "a screenshot: only its text names the card"
    twin = v.get("twin") and v["twin"]["bits"][0] <= dm.TWIN_BITS and cid not in close
    if cid in family and s.get("title", 0) < FAMILY_TITLE and not twin:
        return f"look-alike family {family[cid]}: its title doesn't name it"
    return None


def look_clash(pick: dict, whole: Optional[dict], members: list) -> Optional[str]:
    """A text pick the whole-card hash contradicts: another look of the card nearer by CLASH_BITS, within LOOK_BITS
    (members: the card's versions, {key, look, whole})."""
    keys = set(pick.get("keys") or [])
    if pick.get("by") != "text" or not whole or not keys:
        return None
    near = [(dm.hash_distance(whole, m["whole"]), m["key"] in keys) for m in members if m.get("whole")]
    mine = min((d for d, picked in near if picked), default=99.0)
    other = min((d for d, picked in near if not picked), default=99.0)
    return f"the hash is nearer another look ({other:.2f} vs {mine:.2f} bits)" \
        if other <= dm.LOOK_BITS and mine - other >= CLASH_BITS else None


def when_posted(posts: list, card: Optional[dict]) -> dict:
    """A picture's first post (seconds), the first one a person uploaded (made: no bot's, no embed's; an embed's
    picture was fetched when archived, so it may be a later one), its card's git versions in no local store (their
    times) and whether the card prints no bars."""
    made = [dm.epoch(p["date"]) or 0 for p in posts if not p["bot"] and not p.get("embed")]
    return {"posted": min((dm.epoch(p["date"]) or 0 for p in posts), default=None),
            "made": min(made) if made else None, "unstored": (card or {}).get("unstored") or [],
            "no_bars": prints_no_bars(card)}


def prints_no_bars(card: Optional[dict]) -> bool:
    """A card that prints no stat bars: its type (NO_BARS) or its data's stats all -1 (no_stats: most items)."""
    return (card or {}).get("type") in NO_BARS or bool((card or {}).get("no_stats"))


def card_data(cards_repo: str) -> tuple:
    """({card key} whose data stats are all -1, {card key: every star count its data versions have}) from the
    card repo's data branch (empty without one)."""
    git_dir = dm.git_dir_of(cards_repo) if cards_repo else None
    if not git_dir:
        return set(), {}
    stats, old = ((loads(dm.git_out(git_dir, "show", f"origin/{DATA_BRANCH}:{f}") or b"{}") or {}).get("cards") or {}
                  for f in (STATS_FILE, OLD_STATS_FILE))
    none, stars = set(), defaultdict(set)
    for key, entry in stats.items():
        line = [(entry.get("stats") or {}).get(s) for s in ("hp", "defense", "attack", "speed")]
        if not any(isinstance(v, int) and v >= 0 for v in line):
            none.add(key)
        for e in [entry, *(old.get(key) or [])]:
            if isinstance(e, dict) and isinstance(e.get("stars"), int):
                stars[key].add(e["stars"])
    return none, dict(stars)


def text_reads(root: str) -> tuple:
    """({picture sha256: era}, {picture sha256: star count}) from the text readers: the era of every Discord picture
    they read, its stars when read by eye or sure, or on a status card (era S, whose Discord reads have no truth to be
    sure by) at today's status cards' calibrated bar with no doubt."""
    eras, stars = {}, {}
    eye = defaultdict(dict)                     # an id's fields from every eye file
    for r in dm.read_jsonl(join(root, *TEXT_TRUTH)):
        if str(r.get("id", "")).startswith("d:"):
            eye[r["id"][2:]].update(r.get("fields") or {})
    cal = dm.read_json(join(root, *TEXT_CAL), {}) or {}
    s_bar = ((cal.get("eras") or {}).get(S_STARS_BAR) or {}).get("stars", {}).get("bar")
    for r in dm.read_jsonl(join(root, *TEXT_READS)):
        if r.get("src") != "discord" or "#" in (r.get("file") or ""):     # a sheet's cell: the chain's
            continue
        sha = splitext(basename(r.get("file") or ""))[0]
        eras[sha] = r.get("era")
        read = r.get("stars") or {}
        given = (eye.get(sha) or {}).get("stars")
        if isinstance(given, int):
            stars[sha] = given
        status = r.get("era") == "S" and s_bar is not None and (read.get("conf") or 0) >= s_bar
        if isinstance(read.get("value"), int) and (read.get("sure") or (status and not read.get("doubt"))):
            stars.setdefault(sha, read["value"])
    return eras, stars


def old_stars(root: str) -> dict:
    """{card key: star counts its old repo versions show (any read)}."""
    out = defaultdict(set)
    for r in dm.read_jsonl(join(root, *OLD_ROWS)):
        f = (r.get("fields") or {}).get("stars") or {}
        for v in [f.get("value"), *((f.get("reads") or {}).values())]:
            if isinstance(v, int):
                out[r.get("card")].add(v)
    return dict(out)


def old_eras(root: str) -> dict:
    """{old repo blob[:20]: era} of the era bar readers' old repo PNGs (an older template kept in git: Hellmari)."""
    return {r["blob"][:20]: r.get("era") for r in dm.read_jsonl(join(root, *BARS))
            if r.get("src") == "oldgit" and r.get("variant") == "png" and r.get("blob")}


def marks(lines) -> set:
    """The timing tags and numbers (with the next word's first letter) an OCR read has: what a cooldown or number
    change alters while the text and the hash stay alike."""
    raw = " ".join(line if isinstance(line, str) else line[0] for line in lines or []
                   if isinstance(line, str) or line[1] >= 0.3)
    found = {"t:" + next(g for g in m.groups() if g).lower().translate(dm.CONFUSABLE) for m in TAG.finditer(raw)}
    return found | {f"n:{sub(r'^0+(?=[0-9])', '', n.replace(',', '.'))}:{w}" for n, w in NUMBER.findall(dm.fold(raw))}


def read_marks(lines) -> Optional[set]:
    """A picture's marks, None when it has too little text read to tell (dm.TEXT_MIN)."""
    text = " ".join(line if isinstance(line, str) else line[0] for line in lines or [])
    return marks(lines) if len(dm.alnum(text)) >= dm.TEXT_MIN else None


def version_marks(entry: dict) -> Optional[tuple]:
    """(in any read, in every read) of a version's marks (its reads at several sizes)."""
    reads = [marks(r) for r in [entry.get("lines"), *(entry.get("rereads") or [])] if r]
    return (set().union(*reads), set.intersection(*reads)) if reads else None


def version_reads(entry: dict) -> list:
    """A version's OCR reads (squashed), one per size read."""
    return [t for t in (dm.read_text(r) for r in [entry.get("lines"), *(entry.get("rereads") or [])] if r) if t]


def text_runs(text: str, reads: list) -> Optional[tuple]:
    """(novel, missing) in characters in a row: the picture's text no read of the version has, and the version's
    steady text (in its every read) the picture lacks. None when either has no text read."""
    if not text or not reads:
        return None
    union = set().union(*(dm.grams(t) for t in reads))
    steady = set.intersection(*(dm.grams(t) for t in reads))
    return dm.novel_run(text, union), dm.novel_run(reads[0], dm.grams(text) | (dm.grams(reads[0]) - steady))


def picture_text(image: dict) -> str:
    """A picture's OCR text, squashed (a screenshot's: its lines)."""
    return image.get("text") or dm.squash(" ".join(line if isinstance(line, str) else line[0]
                                                   for line in image.get("lines") or []))


def text_fit(seen: set, members: list) -> Optional[float]:
    """The share of a picture's 4-grams in the nearest of a card's versions (all its reads), None when either has
    none."""
    fits = [len(seen & m["grams"]) / len(seen) for m in members if m.get("grams")]
    return round(max(fits), 3) if seen and fits else None


def name_only(cid: str, v: dict, twin: Optional[dict] = None) -> bool:
    """A card the matcher is sure of by its name alone: its methods (and, made sure by a twin, the twin's) only
    NAME_METHODS, nothing of its look or text."""
    methods = set()
    for verdict in (v, twin):
        methods |= set((scored_of(verdict, cid) or {}).get("s") or {}) if verdict else set()
    return v.get("status") != "ruled" and methods <= NAME_METHODS


def name_check(cid: str, v: dict, text: str, members: dict, cards: dict, twin: Optional[dict] = None) -> \
        Optional[str]:
    """Why a card named only isn't sure (None: it is, or it isn't named only): its text not read or under NAME_FIT
    of the card's (the matcher's version fit; without one, its nearest version's), or a card the matcher saw in its
    text (ocr, body) fitting it better, both by text_fit (Blooper Tentacle's text under Tentacle's name). members:
    {card: its versions with grams}."""
    if not name_only(cid, v, twin):
        return None
    seen = dm.grams(text or "")
    own = text_fit(seen, members.get(cid) or [])
    fit = next((f for f in ((((scored_of(x, cid) or {}).get("why") or {}).get("pick") or {}).get("fit")
                            for x in (v, twin) if x) if f is not None), own)
    rivals = {c["card"] for verdict in (v, twin) if verdict for c in verdict.get("scored") or []
              if not dm.related(cards, c["card"], cid) and (c["s"].get("ocr") or c["s"].get("body"))}
    best, rival = max(((text_fit(seen, members.get(r) or []) or 0.0, r) for r in rivals), default=(0.0, ""))
    better = f"its text reads {rival}'s better ({best:.2f} vs {own or 0.0:.2f})" if best > (own or 0.0) else ""
    if fit is None:
        return "named only, no text to check it by"
    if fit < NAME_FIT:
        return f"named only, its text isn't the card's (fit {fit:.3f})" + (f"; {better}" if better else "")
    return f"named only, {better}" if better else None


def live_look(image: dict, versions, text: str) -> Optional[str]:
    """The live card a joke picture is a git version of: its whole card within dm.LOOK_BITS of that version and
    the version's steady text all in it (text_runs' missing under SURE_MISSING, or no text read): a repost, not a
    joke made on the card's look (Toy Bonnie 'vvvvvvv')."""
    if not image.get("whole"):
        return None
    near = sorted((d, v["key"], v) for v in versions if v.get("whole")
                  for d in [dm.hash_distance(image["whole"], v["whole"])] if d <= dm.LOOK_BITS)
    for _, _, v in near:
        runs = text_runs(text, v.get("reads") or [])
        if runs is None or runs[1] < SURE_MISSING:
            return v["card"]
    return None


def version_grams(v: dict) -> set:
    return set().union(*(dm.grams(t) for t in v.get("reads") or []))


def tag_words(versions) -> frozenset:
    """The timing tags printed on 2+ cards (their versions' every read): real tags, not OCR noise."""
    cards = defaultdict(set)
    for v in versions:
        for m in (v.get("marks") or (set(), set()))[1]:
            if m.startswith("t:"):
                cards[m].add(v.get("card"))
    return frozenset(m for m, owners in cards.items() if len(owners) >= 2)


def mark_diff(seen: set, known: tuple, tags: Optional[frozenset] = None) -> tuple:
    """(novel, lacks): a picture's marks no read of the version has (a number by its value), and the version's
    marks in its every read the picture lacks (a number with its next letter: 3 turns for 2 turns). With tags (the
    real timing tags), a tag OCR misread ("(AAT)" as CART) is read as the tag, or else dropped."""
    union, stable = known

    def tag(m: str) -> Optional[str]:
        if tags is None or m in tags or not m.startswith("t:"):
            return m
        t = m[2:]
        return next((f"t:{x}" for x in (t[1:] if t[:1] == "c" else "", t[:-1] if t[-1:] in "lji" else "")
                     if x and f"t:{x}" in tags), None)
    post = {tag(m) for m in seen} - {None}
    values = {m.split(":")[1] for m in union if m.startswith("n:")}
    novel = sorted(m for m in post if (m.split(":")[1] not in values if m.startswith("n:") else m not in union))
    lacks = sorted(m for m in stable if not _known(m, post) and (tags is None or not m.startswith("t:") or m in tags))
    return novel, lacks


def short_marks(found: set) -> set:
    """Marks without numbers of 3+ characters or with a point (OCR joins '10 154', splits '0 5')."""
    return {m for m in found if not m.startswith("n:") or (len(m.split(":")[1]) <= 2 and "." not in m.split(":")[1])}


def whole_card(image: dict) -> bool:
    """A picture shaped as a whole card export (1:2, 2:3 or square within WHOLE_ASPECT), not a cut screenshot."""
    w, h = image.get("w"), image.get("h")
    return bool(w and h) and any(abs(w / h - r) / r <= WHOLE_ASPECT for r in CARD_ASPECTS)


def _known(mark: str, found: set) -> bool:
    """A mark among others: a number's next letter may be missing on either side."""
    if mark in found or not mark.startswith("n:"):
        return mark in found
    n, w = mark[2:].rsplit(":", 1)
    return any(m.startswith(f"n:{n}:") and (not w or m.endswith(":") or m.endswith(f":{w}")) for m in found)


def sure_bars(read: Optional[dict]) -> Optional[dict]:
    """{stat: value} of a bar read's sure stats, or None."""
    stats = {k: s["v"] for k, s in ((read or {}).get("stats") or {}).items() if isinstance(s, dict) and s.get("sure")}
    return stats or None


def bar_reads(root: str, variant: str = "png") -> tuple:
    """({picture sha256: sure stats}, {blob[:20]: sure stats}, {picture sha256: era}) from the era bar readers'
    reads (BARS): Discord pictures, the old repo's and the card repo's PNGs (variant: as stored, or jpeg400, a
    Discord-like copy)."""
    pictures, blobs, eras = {}, {}, {}
    for r in dm.read_jsonl(join(root, *BARS)):
        stats = sure_bars(r)
        if r.get("src") == "discord" and "#" in (r.get("file") or ""):   # a sheet's cell: the chain's
            continue
        if r.get("src") == "discord":
            sha = splitext(basename(r.get("file") or ""))[0]
            eras[sha] = r.get("era")
            if stats:
                pictures[sha] = stats
        elif r.get("src") in ("oldgit", "tc") and r.get("variant") == variant and r.get("blob") and stats:
            blobs[r["blob"][:20]] = stats
    return pictures, blobs, eras


def look_check(image: dict, members: list, look: int) -> tuple:
    """(why not, checked) for a git look the hash and text pick: the version shown (before the look's first git
    version: that one; after: the latest at the post) must have the picture's timing tags and numbers, and its bars
    (when both are read) the picture's; a picture made more than GRACE_DAYS before the look's first git version
    needs its bars read the same (the hash can't see a bar), as does a look whose versions print different stats;
    one made before it at all, its text read exactly that version's (EXACT_RUN, at any length)."""
    mine = sorted((m for m in members if m.get("look", 0) == look and m.get("ts")), key=lambda m: m["ts"])
    if not mine:
        return None, {}
    made = image.get("made")
    early = (mine[0]["ts"] - made) / 86400 if made is not None and made < mine[0]["ts"] else 0.0
    shown = mine[0] if early else mine[-1] if made is None else [m for m in mine if m["ts"] <= made][-1]
    checked = {"days_before_git": round(early, 1)} if early else {}
    seen, known = image.get("marks"), shown.get("marks")
    if seen is not None and known:
        novel, lacks = mark_diff(seen, known, image.get("tags"))
        if novel or lacks:
            return "numbers or timing tags not that version's (" + ", ".join(
                [*(m[2:] for m in novel[:3]), *("no " + m[2:] for m in lacks[:3])]) + ")", checked
        checked["marks"] = "same"
    bars, ref = image.get("bars"), shown.get("bars")
    both = [k for k in STATS if k in (bars or {}) and k in (ref or {})]
    if both and any(bars[k] != ref[k] for k in both):
        return "its bars read " + "/".join(str(bars[k]) for k in both) + ", that version's " + \
            "/".join(str(ref[k]) for k in both), checked
    if both and set(ref) <= set(bars):
        checked["bars"] = "same"
    no_bars = image.get("no_bars")
    if early > GRACE_DAYS and not no_bars and "bars" not in checked:
        return f"made {early:.0f} days before that look's first git version, its bars not read", checked
    if early > GRACE_DAYS and "marks" not in checked:
        return f"made {early:.0f} days before that look's first git version, its text not read", checked
    if not no_bars and "bars" not in checked and \
            len({tuple(sorted(m["bars"].items())) for m in mine if m.get("bars")}) > 1:
        return "that look's git versions print different stats, its bars not read", checked
    if early:
        runs = text_runs(image.get("text") or "", shown.get("reads") or [])
        if runs is None:
            return "made before that look's first git version, its text not read", checked
        if max(runs) >= EXACT_RUN:
            return f"made before that look's first git version, its text not exactly that version's " \
                   f"({max(runs)} characters in a row)", checked
        checked["text"] = "same"
    return None, checked


def unstored_before(image: dict, members: list) -> Optional[str]:
    """A git version of the card made before the picture was posted is in no local store (2025-26 re-exports): it
    may be the look shown."""
    posted, unstored = image.get("posted"), image.get("unstored") or []
    return "a git version made before the post isn't stored here" if posted and any(t <= posted for t in unstored) \
        else None


def version_of(v: dict, cid: str, image: dict, versions: dict, members: list) -> tuple:
    """(kind, note) of the look a card-shaped image shows: kind git (one look of the card's git versions), draft (a
    look not in git) or None (can't tell)."""
    return decide(((scored_of(v, cid) or {}).get("why") or {}).get("pick") or {}, image, versions, members)


def decide(pick: dict, image: dict, versions: dict, members: list) -> tuple:
    """version_of from the matcher's pick (members: the card's versions, {key, look, whole}). git: one look, the
    whole card within SURE_BITS of it, a text pick reading SURE_FIT of it, the hash saying no other look
    (look_clash). draft: the matcher's draft, or a text pick whose look is FAR_BITS from every git version and whose
    text isn't theirs (FAR_RUN characters in a row, or under FAR_FIT of it). Else None: can't tell. The local gate
    (local.jsonl, eyekeys' score, aspect within LOCAL_AR) comes first: LOCAL_SAME from a stored version is its look;
    a gate's look over LOCAL_KEEP from that look's stored versions isn't told; LOCAL_DRAFT from every one a draft."""
    local = image.get("local") or {}
    best = min(local, key=local.get) if local else None
    if best in versions and local[best] <= LOCAL_SAME:
        same = versions[best].get("look", 0)
        keys = sorted((k for k in local if local[k] <= LOCAL_SAME and versions.get(k, {}).get("look", 0) == same),
                      key=local.get)
        note = dm.version_note(("git", same), {"by": "local", "keys": keys}, versions)
        note.update(keys=keys[:12], checked={"local": local[best]})
        return "git", note
    look, pick = dm.shown_look({"why": {"pick": pick}}, image, versions)
    note = dm.version_note(look, pick, versions)
    note["keys"] = [k for k in pick.get("keys") or [] if k in versions][:12]
    near: dict = {}                             # look -> the whole card's distance to its nearest version
    for m in members if image.get("whole") else ():
        if m.get("whole"):
            near[m.get("look", 0)] = min(near.get(m.get("look", 0), 99.0), dm.hash_distance(image["whole"], m["whole"]))
    fit = pick.get("fit") if pick.get("fit") is not None else 1.0
    if look and look[0] == "git":
        mine = near.get(look[1], 99.0)
        text = pick.get("by") == "text"
        why = look_clash(pick, image.get("whole"), members) or (
            f"the whole card is {mine:.2f} bits from that look" if mine > SURE_BITS else None) or (
            f"its text reads {fit:.2f} of that look's" if text and fit < SURE_FIT else None) or (
            f"{pick.get('novel')} characters in a row not that look's" if text and (pick.get("novel") or 0) >=
            SURE_NOVEL else None) or (
            f"{pick.get('missing')} characters in a row of that look's text not read" if text and
            (pick.get("missing") or 0) >= SURE_MISSING else None) or unstored_before(image, members)
        if not why:
            why, checked = look_check(image, members, look[1])
            if checked:
                note["checked"] = checked
        mine = [local[k] for k in local if versions.get(k, {}).get("look", 0) == look[1]]
        if not why and mine and min(mine) > LOCAL_KEEP:
            why = f"drawn otherwise in one area: {min(mine):.0f} from that look's stored versions (local)"
        if not why:
            return "git", note
        note.update(shows=None, unsure=why, near_look=look[1])
        note.pop("look", None)
    elif look:
        return "draft", note
    far = min(near.values(), default=pick.get("bits") or 0.0)
    if image.get("shape") and pick.get("by") == "text" and far >= FAR_BITS and (
            (pick.get("novel") or 0) >= FAR_RUN or fit < FAR_FIT):
        note.update(shows="draft", far=round(far, 2), commits=[], git_dates=[], keys=[])
        note.pop("unsure", None)
        return "draft", note
    why = local_draft(image, members, pick.get("keys") or ()) or (
        no_git_look(image, members, near) if image.get("shape") and near else None)
    if why:
        note.update(shows="draft", far=round(min(near.values()), 2) if near else None, commits=[], git_dates=[],
                    keys=[], draft=why)
        for k in ("unsure", "near_look", "look"):
            note.pop(k, None)
        return "draft", note
    return None, note


def local_seen(row: Optional[dict], members: list) -> dict:
    """The local gate's view of a picture's local.jsonl row: {key: local diff} of its card's stored versions with its
    aspect ratio (LOCAL_AR), and the times of the versions it wasn't compared with (unstored, not computed)."""
    if not row:
        return {}
    got, keys = row.get("versions") or {}, {m["key"]: m for m in members}
    return {"local": {k: s for k, (s, ar) in got.items() if k in keys and abs(ar - 1) <= LOCAL_AR},
            "local_unread": [keys[k].get("ts") for k in keys if k not in got]}


def local_draft(image: dict, members: list, picked=()) -> Optional[str]:
    """A draft by the local gate: LOCAL_DRAFT from every stored version with its aspect ratio (LOCAL_DRAFT_NARROW
    under LOCAL_WIDE), none made before the post left uncompared, and one of the look the matcher picked (picked: its
    versions) compared (localtest: no Discord or small copy of a git version gets there)."""
    local, posted = image.get("local") or {}, image.get("posted")
    bar = LOCAL_DRAFT if (image.get("w") or 0) >= LOCAL_WIDE else LOCAL_DRAFT_NARROW
    if not local or min(local.values()) < bar or unstored_before(image, members) or \
            any(t is None or posted is None or float(t) <= posted for t in image.get("local_unread") or ()) or \
            (picked and not any(k in local for k in picked)):
        return None
    return f"drawn otherwise: {min(local.values()):.0f} or more from every stored git version (local)"


def no_git_look(image: dict, members: list, near: dict) -> Optional[str]:
    """Why a card picture whose look isn't told can be no git version (None: it may be one): an older template
    (OLD_ERAS) where no git version of the card is one, OLD_BITS from every git version; a whole card (not cut:
    whole_card) printing a number or timing tag no read of any git version has (mark_diff's novel, numbers of 1-2
    digits: OCR joins and splits longer ones); a status card's star count no git version has; a whole card export
    (EXPORT_SIZES) OLD_BITS from every git version. The last three also need no git version unstored before the post,
    the first two of them the whole card over SURE_BITS from every git version. (Selftest: no copy of a git version
    turns into a draft; a version's own marks missing from a cut or small copy did; Discord copies of 4,029
    versions are under 7.5 bits from their own.)"""
    far = min(near.values())
    if image.get("era") in OLD_ERAS and far >= OLD_BITS and not any(m.get("era") in OLD_ERAS for m in members):
        return f"an older template ({image['era']}) than every git version's"
    if far <= SURE_BITS or unstored_before(image, members):
        return None
    seen, known = image.get("marks"), [m["marks"] for m in members if m.get("marks")]
    if seen and known and whole_card(image) and all(
            mark_diff(short_marks(seen), tuple(map(short_marks, k)), image.get("tags"))[0] for k in known):
        return "numbers or timing tags no git version has"
    stars, git = image.get("stars"), image.get("git_stars")
    if image.get("era") == "S" and isinstance(stars, int) and git and stars not in git:
        return f"{stars} stars, no git version's ({', '.join(map(str, sorted(git)))})"
    if image.get("size_class") in EXPORT_SIZES and far >= OLD_BITS:
        return f"a whole card export {far:.1f} bits from every git version"
    return None


# ========================
# Pictures
# ========================
class Run:
    """What judging a picture needs: the matcher's pairs, the posts and their verdicts."""
    def __init__(self, images: dict, records: dict, cards: dict, versions: dict, morgue: set, verdicts: dict,
                 raw: dict, checked: Optional[dict] = None, close: Optional[dict] = None):
        self.images, self.records, self.cards, self.versions, self.morgue = images, records, cards, versions, morgue
        self.verdicts, self.raw, self.close = verdicts, raw, close or {}  # close: cards whose git versions look alike
        self.family = families(cards, self.close)
        self.checked, self.names = checked or {}, dm.Names(cards)     # sheets checked by eye; names for cells
        self.classes: dict = {}                                       # sha256 -> the catalogue's size class
        self.bars: dict = {}                                          # sha256 -> its bars read (sure stats)
        self.eras: dict = {}                                          # sha256 -> its era (bar readers, else text)
        self.stars: dict = {}                                         # sha256 -> its star count (eye or sure)
        self.git_stars: dict = {}                                     # card -> its git versions' star counts
        self.local: dict = {}                                         # picture id -> its local.jsonl row
        self.tags = tag_words(versions.values())                      # the real timing tags
        self.members = defaultdict(list)            # card -> its versions {key, look, whole, ts, marks, bars}
        for v in versions.values():
            self.members[v["card"]].append(v)

    def seen(self, sha: str, image: dict, posts: list, cid: str, cell: bool = False, pid: str = "") -> dict:
        """What a look check needs of a picture (or a sheet's cell: its bars unread)."""
        return {**image, **when_posted(posts, self.cards.get(cid)), "marks": read_marks(image.get("lines")),
                "bars": None if cell else self.bars.get(sha), "tags": self.tags,
                "era": None if cell else self.eras.get(sha), "stars": None if cell else self.stars.get(sha),
                "git_stars": self.git_stars.get(cid), "size_class": None if cell else self.classes.get(sha),
                **local_seen(self.local.get(pid), self.members.get(cid, []))}

    def twin(self, v: dict) -> Optional[dict]:
        """The post a verdict was made sure by (dm.same_image), if any."""
        return self.verdicts.get((v.get("twin") or {}).get("key"))

    def name_why(self, cid: str, v: dict, text: str) -> Optional[str]:
        return name_check(cid, v, text, self.members, self.cards, self.twin(v))

    @staticmethod
    def named_live(kept: list, found: dict, meme_only: bool) -> list:
        """The cards a joke picture's posts outside meme channels name at SURE (sure of the card or not: a name
        check failing still keeps it out of the Morgue)."""
        return [] if meme_only else [c for c in kept if found[c][0] >= SURE]

    def joke_verdict(self, image: dict, text: str, reason: str, named=(), looks=()) -> dict:
        """A joke picture's verdict when it is no sure card: morgue, or review (MEME_LIVE) when it is a live
        card's picture: named so by its posts outside meme channels, or a git version's look and text (live_look)."""
        live = next(iter(named), None)
        if not live:
            live = live_look(image, self.versions.values(), text)
            reason = f"its whole card and text are {live}'s git version" if live else reason
        if live:
            return {"verdict": "unsure", "meme_live": live, "reason": reason,
                    "maybe": list(dict.fromkeys([live, *looks]))[:6]}
        return {"verdict": "morgue", "reason": reason, "looks_like": list(looks)[:3]}


def judge_picture(sha: str, keys: list, joke: bool, run: Run) -> dict:
    """One picture's verdict from its posts' verdicts."""
    images, records, cards, versions, morgue = run.images, run.records, run.cards, run.versions, run.morgue
    verdicts, raw, family, members = run.verdicts, run.raw, run.family, run.members
    image = images[keys[0]]
    posts = [post_row(records[images[k]["message"]], morgue, images[k].get("kind")) for k in keys]
    live = [verdicts[k] for k, p in zip(keys, posts) if not p["morgue"]]
    row = {"id": picture_id(sha), "sha256": sha, "file": image["file"], "thumb": image.get("thumb"),
           "w": image.get("w"), "h": image.get("h"), "shape": image.get("shape"), "posts": posts,
           "joke": joke}
    found = sure_cards(live, cards)
    meme_only = False
    if not found and joke:                     # posted only in meme channels: its own verdict, if a git look
        found, meme_only = sure_cards([raw[k] for k, p in zip(keys, posts) if p["morgue"]], cards), True
    kept = unrelated(found, cards)
    shape, text = image.get("shape"), picture_text(image)
    if kept and not shape:                     # a screenshot: each card it shows, none's look told
        sure = [c for c in kept if gate(c, *found[c], None, family, run.close) is None and not meme_only]
        if sure:
            conf, v = found[sure[0]]
            row.update(verdict="version_unsure", card=sure[0], cards=sure, conf=round(conf, 4),
                       methods=(scored_of(v, sure[0]) or {}).get("s"), reason="a screenshot: " + (v.get("reason") or
                                                                                               "its look isn't told"))
            if joke:
                row["meme_repost"] = True
            return row
        why = gate(kept[0], *found[kept[0]], None, family, run.close) or "posted only in meme channels"
        row.update(maybe=kept[:6], reason=why)
        if joke:
            row.update(run.joke_verdict(image, text, why, run.named_live(kept, found, meme_only), kept))
        else:
            row["verdict"] = "unsure"
        return row
    if len(kept) == 1:
        cid = kept[0]
        conf, v = found[cid]
        named = run.name_why(cid, v, text)
        why = gate(cid, conf, v, shape, family, run.close, whole_card(image)) or named
        kind, note = version_of(v, cid, run.seen(sha, image, posts, cid, pid=row["id"]), versions,
                                members.get(cid, []))
        if meme_only and kind != "git":
            why = why or "posted only in meme channels, in no git look"
        if why is None:
            verdict = "version_sure" if kind == "git" else "draft" if kind == "draft" else "version_unsure"
            row.update(verdict=verdict, card=cid, cards=kept, conf=round(conf, 4),
                       methods=(scored_of(v, cid) or {}).get("s"), version=note, reason=v.get("reason") or "")
            if name_only(cid, v, run.twin(v)):
                row["name_only"] = True
            if joke:
                row["meme_repost"] = True
                if kind == "draft":                     # a joke picture that is also a live card's draft: review
                    row["joke_draft"] = True
            if v.get("twin"):
                row["twin_of"] = v["twin"]["key"]
            return row
        row.update(maybe=kept, conf=round(conf, 4), reason=why)
        if joke:
            row.update(run.joke_verdict(image, text, why, run.named_live(kept, found, meme_only), kept))
        else:
            row["verdict"] = "unsure"
        return row
    if len(kept) > 1:                          # one card's image, several sure cards: a clash
        why = "posts say different cards: " + ", ".join(kept)
        row.update(maybe=kept, reason=why)
        if joke:
            row.update(run.joke_verdict(image, text, why, run.named_live(kept, found, meme_only), kept))
        else:
            row["verdict"] = "unsure"
        return row
    every = live or [raw[k] for k in keys]
    status = Counter(v["status"] for v in every)
    maybe = list(dict.fromkeys(c for v in every for c in v.get("maybe") or []))
    if joke:
        looks = list(dict.fromkeys(c for k in keys for c in (raw[k].get("cards") or raw[k].get("maybe") or [])))
        if joke_card(image) or any(raw[k].get("cards") for k in keys):
            row.update(run.joke_verdict(image, text, "a joke picture: no live card's", (), looks))
        else:                                   # a meme, not a joke card: never a Morgue candidate
            row.update(verdict="none", reason="a meme channel's picture, not a card")
    elif status.get("ambiguous"):
        row.update(verdict="unsure", maybe=maybe[:6], reason=next(v["reason"] for v in every
                                                                  if v["status"] == "ambiguous"))
    elif lost_like(image, run.classes.get(sha)):
        row.update(verdict="lost", reason="a card picture naming no live card", maybe=maybe[:3])
    elif status.get("art"):
        row.update(verdict="art", maybe=list(dict.fromkeys(c for v in every for c in v.get("art") or []))[:3],
                   reason="a card's art only")
    else:
        row.update(verdict="none", reason="no card")
    return row


def lost_like(image: dict, size_class: Optional[str] = None) -> bool:
    """A card picture of ours: card-shaped, a title read as words, card text (or a stat label), mostly letters (no
    table of numbers), no other game's card (FOREIGN), and the game's words (GAME_WORDS of them) or a card
    template's export size (memes, comics and screenshots have neither)."""
    title = image.get("title_block") or next(iter(image.get("title") or []), "")
    text = " ".join(image.get("lines") or [])
    letters, digits = sum(c.isalpha() for c in text), sum(c.isdigit() for c in text)
    ours = len(set(dm.words(text).split()) & GAME) >= GAME_WORDS or size_class in TEMPLATE_SIZES or \
        (image.get("w"), image.get("h")) == (400, 400)
    return bool(image.get("shape")) and dm.wordy(title) and len(dm.alnum(title)) >= LOST_TITLE and \
        dm.card_text(image) and letters >= LETTERS * (letters + digits) and not FOREIGN.search(text) and ours and \
        not APP.search(text) and not rule_page(title, image.get("lines") or [])


def joke_card(image: dict) -> bool:
    """A joke card (Q1: the Morgue's), not a meme: card-shaped, a title read as words, card text, and no other game's
    card, app screen or rules page."""
    title = image.get("title_block") or next(iter(image.get("title") or []), "")
    text = " ".join(image.get("lines") or [])
    return bool(image.get("shape")) and dm.wordy(title) and len(dm.alnum(title)) >= LOST_TITLE and \
        dm.card_text(image) and not FOREIGN.search(text) and not APP.search(text) and \
        not rule_page(title, image.get("lines") or [])


def chapter(text: str) -> Optional[str]:
    """The rulebook chapter a heading names (a page number after it allowed), or None."""
    form = dm.squash(sub(r"\d+", " ", text))
    return next((c for c in RULE_CHAPTERS if form and form == dm.squash(c)), None)


def rule_page(title: str, lines: list) -> bool:
    """A rules page: titled as a rulebook chapter (a one-word one only with a page number or no stat labels; a
    two-word one also in its first lines), a table of contents (2+ chapters in its title), or timing tags defined
    (TAG_RULE twice)."""
    labels = sum(1 for line in lines if STAT_LABEL.match(dm.alnum(line)))
    named = chapter(title)
    if named and (" " in named or any(c.isdigit() for c in title) or labels < 2):
        return True
    if any(" " in (chapter(line) or "") for line in lines[:3]):
        return True
    if sum(1 for c in RULE_CHAPTERS if " " in c and f" {dm.words(c)} " in f" {dm.words(title)} ") >= 2:
        return True
    return len(TAG_RULE.findall(" ".join(lines))) >= 2


def card_evidence(image: dict, size_class: Optional[str], era: Optional[str], cell_checked: bool = False) -> \
        Optional[str]:
    """Why a lost picture is a card of ours, not a page or a screenshot: its stat labels, a card template's export
    size, a card era (the era readers' class), or a cell of a sheet checked by eye. None: no such evidence."""
    labels = sum(1 for line in image.get("lines") or [] if STAT_LABEL.match(dm.alnum(line)))
    return "stat labels" if labels >= 2 else "a template's size" if size_class in TEMPLATE_SIZES or (
        image.get("w"), image.get("h")) in ((400, 400), (1200, 2400), (1200, 1800)) else \
        f"era {era}" if era in CARD_ERAS else "a checked sheet's cell" if cell_checked else None


def judge_cells(sha: str, keys: list, joke: bool, run: Run) -> list:
    """A sheet's cells, each a picture judged as a card (no chat around a cell)."""
    images, records, cards, versions, morgue = run.images, run.records, run.cards, run.versions, run.morgue
    family, members = run.family, run.members
    image = images[keys[0]]
    posts = [post_row(records[images[k]["message"]], morgue, images[k].get("kind")) for k in keys]
    rows = []
    for n, cell in enumerate(image["cells"], 1):
        part = {**cell, "ocr": image.get("ocr")}
        v = dm.judge(part, cards)
        cid = v["cards"][0] if v["status"] == "confident" and len(v["cards"]) == 1 else None
        row = {"id": picture_id(sha, n), "sha256": sha, "cell": n, "box": cell["box"], "file": image["file"],
               "thumb": image.get("thumb"), "w": cell.get("w"), "h": cell.get("h"), "shape": cell.get("shape"),
               "posts": posts, "joke": joke, "sheet": image.get("sheet")}
        conf = (scored_of(v, cid) or {}).get("conf", 0.0) if cid else 0.0
        named = {c: (1.0, None) for t in cell.get("title") or [] for c in run.names.fuzzy(t, dm.TITLE_LINE, limit=1)}
        mixed = len(unrelated(named, cards)) > 1           # the grid cut through cards: other cells' titles in it
        doubt = "other cards' titles inside the cell (the grid may be wrong)" if mixed else None if \
            run.checked.get(sha) else "a sheet found by names only (its cells may be cut wrong)"
        text = picture_text(part)
        named = run.name_why(cid, v, text) if cid else None
        why = (gate(cid, conf, v, cell.get("shape"), family, run.close, whole_card(cell)) or named or
               (doubt if mixed else None)) if cid else None
        live = [cid] if cid and conf >= SURE and any(not p["morgue"] for p in posts) else []
        if cid and why is None:
            kind, note = version_of(v, cid, run.seen(sha, part, posts, cid, cell=True, pid=row["id"]), versions,
                                    members.get(cid, []))
            if doubt:                           # the card, its look not told
                kind, note = None, {**note, "shows": None, "unsure": doubt}
            row.update(verdict="version_sure" if kind == "git" else "draft" if kind == "draft" else "version_unsure",
                       card=cid, cards=[cid], conf=round(conf, 4), methods=(scored_of(v, cid) or {}).get("s"),
                       version=note, reason=v.get("reason") or "")
            if name_only(cid, v):
                row["name_only"] = True
            if joke and kind != "git":
                for k in ("card", "cards", "conf", "methods", "version", "name_only"):
                    row.pop(k, None)
                row.update(run.joke_verdict(part, text, "a joke sheet's cell in no git look", live, [cid]))
        elif joke and (joke_card(part) or v["cards"]):
            row.update(run.joke_verdict(part, text, "a joke sheet's cell", live, v["cards"] or v["maybe"]))
        elif joke:
            row.update(verdict="none", reason="a meme channel's picture, not a card")
        elif cid or v["status"] == "ambiguous" or (doubt and lost_like(part)):
            row.update(verdict="unsure", maybe=(v["cards"] or v["maybe"])[:6], reason=why or v["reason"] or doubt)
        elif lost_like(part):
            row.update(verdict="lost", reason="a card picture naming no live card", maybe=v["maybe"][:3])
        else:
            row.update(verdict="none", reason=v["reason"] or "no card")
        rows.append(row)
    return rows


def joke_pictures(images: dict, by_sha: dict, records: dict, morgue: set) -> set:
    """Pictures posted in a meme channel, and pictures the same as one (whole within TWIN_BITS): the joke flag
    follows the picture, not the channel it was first posted in."""
    found = {sha for sha, keys in by_sha.items() if any(dm.in_morgue(records[images[k]["message"]], morgue)
                                                        for k in keys)}
    sha_of = {k: i["sha256"] for k, i in images.items()}
    for sha, keys in by_sha.items():
        if sha not in found and any(sha_of.get(t) in found and whole <= dm.TWIN_BITS
                                    for t, whole, _ in images[keys[0]].get("twins") or []):
            found.add(sha)
    return found


def meme_dates(images: dict, by_sha: dict, records: dict, morgue: set) -> dict:
    """{sha256: its first post in a meme channel, or a twin's (whole within TWIN_BITS)} (message ids: in time)."""
    own = {}
    for sha, keys in by_sha.items():
        ids = [int(images[k]["message"]) for k in keys if dm.in_morgue(records[images[k]["message"]], morgue)]
        if ids:
            own[sha] = min(ids)
    sha_of = {k: i["sha256"] for k, i in images.items()}
    found = {}
    for sha, keys in by_sha.items():
        near = [own[sha_of[t]] for t, whole, _ in images[keys[0]].get("twins") or []
                if sha_of.get(t) in own and whole <= dm.TWIN_BITS]
        dates = ([own[sha]] if sha in own else []) + near
        if dates:
            found[sha] = min(dates)
    return found


def shown_first(row: dict, image: dict, memes: dict, meme_copy: Callable = lambda row, whole: None) -> None:
    """A card's picture (a draft, or a look not told) shown in a played or card channel (SHOWN_FIRST) before any meme
    repost of it is the card's, dated by its other posts: no joke picture (JOKE_ASKED: kept for the owner's open
    questions; meme_copy: a picture the first repo filed only in a Meme folder stays a joke)."""
    if not row.get("joke") or row["verdict"] not in ("draft", "version_unsure") or not row.get("card") or \
            row["card"] in JOKE_ASKED:
        return
    posts = sorted((p for p in row["posts"] if dating(p)), key=lambda p: int(p["message"]))
    meme = memes.get(row["sha256"])
    if posts and meme and SHOWN_FIRST.fullmatch(posts[0]["channel"] or "") and int(posts[0]["message"]) < meme:
        whole = image.get("whole") if row.get("cell") is None else (image["cells"][row["cell"] - 1] or {}).get("whole")
        found = meme_copy(row, whole)
        if found:
            row["meme_file"] = found
            return
        row.update(joke=False, shown_first={"channel": posts[0]["channel"], "date": posts[0]["date"]})
        row.pop("joke_draft", None)


def meme_files(root: str) -> Callable:
    """meme_copy(row, whole) for shown_first: the first repo's PNG filed only in Meme folders that the picture is (the
    same bytes), or near (whole within TWIN_BITS) when the eye hasn't told the picture (God Tamer's port picture
    hashes 0.2 bits from the joke 'doGma T er': only its title differs); None: neither."""
    paths = defaultdict(set)
    for v in dm.read_jsonl(join(root, *OLD_FILES[0])):
        if v.get("ext") == "png":
            paths[v["blob"]].add(v["path"])
    first = {v["blob"] for v in dm.read_jsonl(join(root, *OLD_FILES[0])) if v.get("repo") == "first"}
    only = {b: sorted(p)[0] for b, p in paths.items() if b in first and all("/Meme/" in x for x in p)}
    looks = [(r["look"]["whole"], only[r["blob"]]) for r in dm.read_jsonl(join(root, *OLD_FILES[1]))
             if r.get("blob") in only and isinstance((r.get("look") or {}).get("whole"), dict)]

    def meme_copy(row: dict, whole) -> Optional[str]:
        if row.get("cell") is None and row.get("file"):
            with open(join(root, row["file"]), "rb") as f:
                data = f.read()
            blob = hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()
            if blob in only:
                return only[blob]
        if isinstance(whole, dict) and not row.get("eye") and not row.get("owner"):
            near = [p for h, p in looks if dm.hash_distance(whole, h) <= dm.TWIN_BITS]
            if near:
                return f"near {near[0]}"
        return None
    return meme_copy


def repost_groups(rows: list, images: dict, by_sha: dict) -> dict:
    """{picture id: the earliest picture id of its reposts}: pictures the same image (whole within TWIN_BITS) that
    show one card's one git look (version_sure both): another version or a look-alike card is never a repost."""
    parent = {}

    def top(x):
        while parent.get(x, x) != x:
            x = parent[x]
        return x
    shows = {r["sha256"]: (r["card"], r["version"]["look"]) for r in rows
             if r.get("cell") is None and r["verdict"] == "version_sure"}
    first = {r["sha256"]: min(int(p["message"]) for p in r["posts"]) for r in rows if r["sha256"] in shows}
    sha_of = {k: i["sha256"] for k, i in images.items()}
    for sha in first:
        for t, whole, _ in images[by_sha[sha][0]].get("twins") or []:
            other = sha_of.get(t)
            if other in first and other != sha and whole <= dm.TWIN_BITS and shows[other] == shows[sha]:
                a, b = top(sha), top(other)
                if a != b:
                    parent[max(a, b, key=lambda s: first[s])] = min(a, b, key=lambda s: first[s])
    return {picture_id(sha): picture_id(top(sha)) for sha in first if top(sha) != sha}


# ========================
# Lost cards
# ========================
def nearest_looks(rows: list, versions: list, cards: dict, limit: int = 3) -> dict:
    """{picture id: [{card, bits}]}: each lost picture's nearest live cards by look (whole or art)."""
    np = dm._np()
    refs = [v for v in versions if v.get("whole")]
    if not refs or not rows:
        return {}
    owner = [v["card"] for v in refs]

    def table(part):
        return {k: np.frombuffer(b"".join(bytes.fromhex(v[part][k]) if v.get(part) else bytes(n) for v in refs),
                                 dtype=np.uint8).reshape(len(refs), n) for k, n in (("p", 8), ("d", 8), ("q", 32))}
    whole, art = table("whole"), table("art")
    has_art = np.array([bool(v.get("art")) for v in refs])

    def dist(arrays, h):
        return sum(np.bitwise_count(arrays[k] ^ np.frombuffer(bytes.fromhex(h[k]), dtype=np.uint8)).sum(axis=1) /
                   (4 if k == "q" else 1) for k in ("p", "d", "q")) / 3
    found = {}
    for row in rows:
        if not row.get("whole"):
            continue
        d = dist(whole, row["whole"])
        if row.get("art"):
            d = np.minimum(d, np.where(has_art, dist(art, row["art"]), np.inf))
        best = {}
        for i in np.argsort(d)[:60]:
            best.setdefault(owner[i], round(float(d[i]), 2))
        found[row["id"]] = [{"card": c, "name": (cards.get(c) or {}).get("name"), "bits": b}
                            for c, b in list(best.items())[:limit]]
    return found


def old_names(root: str, cards: dict, names: "dm.Names") -> dict:
    """{squashed old name: card}: the old repo's names of today's cards (its sure renames, oldgit map.json) that no
    card of today goes by: a lost picture so titled may be the card before its rename (a hint for review, never an
    alias)."""
    data = dm.read_json(join(root, *OLD_MAP), {}) or {}
    by_key = {dm.edit_key(c.get("key")): cid for cid, c in cards.items()}
    found = {}
    for old, row in (data.get("cards") or {}).items():
        if not isinstance(row, dict) or row.get("status") != "live" or row.get("confidence") != "sure":
            continue
        name = sub(r"\s*\([^)]*/[^)]*\)$", "", old).strip()
        cid = by_key.get(dm.edit_key(row.get("key") or ""))
        if cid and len(dm.squash(name)) >= 4 and not names.exact(name):
            found[dm.squash(name)] = cid
    return found


def morgue_names(cards_repo: str) -> dict:
    """{squashed name: name} of the cards already in the website's Morgue (the card repo's MORGUE folder)."""
    git_dir = dm.git_dir_of(cards_repo) if cards_repo else None
    out = dm.git_out(git_dir, "ls-tree", "-r", "--name-only", "-z", f"origin/{dm.CARD_BRANCH}", "--", MORGUE_DIR) \
        if git_dir else None
    names = [dm.file_name(p) for p in (out or b"").decode("utf-8", "replace").split("\0") if p.lower().endswith(".png")]
    return {dm.squash(n): n for n in names if len(dm.squash(n)) >= 4}


def changelog_dates(title: str, changelog: list, live: tuple = (), common: frozenset = frozenset()) -> list:
    """Days an update changelog message names the title as a card's name (its words in a row; dates only, never
    the text): not a word of everyday chat (common), not inside a live card's name (live: names, as words)."""
    phrase = dm.words(title)
    if len(phrase.replace(" ", "")) < 5 or phrase in common:
        return []
    inside = [n for n in live if n != phrase and f" {phrase} " in f" {n} "]
    days = set()
    for r in changelog:
        text = f" {dm.words(dm.content_of(r))} "
        for name in inside:
            text = text.replace(f" {name} ", " | ")
        if f" {phrase} " in text:
            days.add(r["timestamp"][:10])
    return sorted(days)


def common_words(titles: list, records: dict) -> frozenset:
    """The one-word titles in more than COMMON_WORD people's messages: chat words, not card names."""
    single = {dm.words(t) for t in titles if t and " " not in dm.words(t)}
    seen = Counter(w for r in records.values() if not dm.is_bot(r)
                   for w in set(dm.words(dm.content_of(r)).split()) & single)
    return frozenset(w for w, n in seen.items() if n > COMMON_WORD)


def lost_groups(rows: list, images: dict, by_sha: dict, records: dict, cards: dict, versions: list, kinds: dict,
                channels: dict, old_lost: list, root: str, cards_repo: str = "") -> list:
    """Lost and morgue pictures grouped into lost cards: one title (alike), else one look (twins)."""
    pics = sorted((r for r in rows if r["verdict"] in ("lost", "morgue")),
                  key=lambda r: (min(int(p["message"]) for p in r["posts"]), r.get("cell") or 0))
    parent = {r["id"]: r["id"] for r in pics}
    order = {r["id"]: n for n, r in enumerate(pics)}

    def top(x):
        while parent[x] != x:
            x = parent[x]
        return x

    def join_ids(a, b):                         # the earlier picture stays the group's root
        a, b = top(a), top(b)
        if a != b:
            parent[max(a, b, key=order.get)] = min(a, b, key=order.get)
    titles = {}
    for r in pics:
        image = images[by_sha[r["sha256"]][0]]
        title = (image.get("title_block") or next(iter(image.get("title") or []), "")) if r.get("cell") is None \
            else next(iter(image["cells"][r["cell"] - 1].get("title") or []), "")
        r["title_read"] = title if dm.wordy(title) and len(dm.alnum(title)) >= LOST_TITLE else ""
        if r["title_read"]:
            form = dm.squash(r["title_read"])
            same = next((t for t in titles if dm.ratio(form, t) >= GROUP_TITLE), None)
            if same:
                join_ids(titles[same], r["id"])
            else:
                titles[form] = r["id"]
    by_sha_id = {r["sha256"]: r["id"] for r in pics if r.get("cell") is None}
    for r in pics:
        if r.get("cell") is not None:
            continue
        for t, whole, art in images[by_sha[r["sha256"]][0]].get("twins") or []:
            other = by_sha_id.get(images[t]["sha256"]) if t in images else None
            if other and (whole <= dm.HASH_FULL or art <= dm.TWIN_BITS):
                join_ids(other, r["id"])
    groups = defaultdict(list)
    for r in pics:
        groups[top(r["id"])].append(r)
    near = nearest_looks([{**r, "whole": r.get("whole"), "art": r.get("art")} for r in pics], versions, cards)
    names = dm.Names(cards)
    changelog = sorted((r for r in records.values() if not dm.is_bot(r) and
                        (kinds.get(r.get("channel_id")) in CHANGELOG_KINDS)), key=lambda r: int(r["id"]))
    old = {dm.squash(c.get("old_id") or ""): c.get("id") for c in old_lost}
    renamed, morgue_cards = old_names(root, cards, names), morgue_names(cards_repo)
    live = tuple(sorted({dm.words(t) for c in cards.values() for t, _ in dm.card_names(c)} - {""}))
    common = common_words([m["title_read"] for m in pics], records)
    found = []
    for root_id, members in groups.items():
        first = members[0]
        sha = first["sha256"]
        title = next((m["title_read"] for m in members if m["title_read"]), "")
        posts = [p for m in members for p in m["posts"]]
        dated = sorted((p for p in posts if not (p["bot"] or p.get("embed"))),
                       key=lambda p: int(p["message"]))
        when = dated[0] if dated else min(posts, key=lambda p: int(p["message"]))
        played = any(played_place(records[p["message"]], kinds, channels) for p in posts if not p["bot"])
        logged = changelog_dates(title, changelog, live, common) if title else []
        joke = any(m["verdict"] == "morgue" for m in members)
        old_id = next((old[o] for m in members if m["title_read"] for o in old
                       if dm.ratio(dm.squash(m["title_read"]), o) >= GROUP_TITLE), None)
        looks = Counter(c for m in members for c in m.get("looks_like") or [])
        by_look = {}
        for m in members:
            for n in near.get(m["id"]) or []:
                if n["bits"] < by_look.get(n["card"], {"bits": 99})["bits"]:
                    by_look[n["card"]] = n
        by_name = [{"card": c, "name": (cards.get(c) or {}).get("name"), "ratio": round(r, 3)}
                   for c, (r, _) in names.fuzzy(title, 0.6, limit=3).items()] if title else []
        was = next((renamed[o] for m in members if m["title_read"] for o in renamed
                    if dm.ratio(dm.squash(m["title_read"]), o) >= OLD_NAME), None)
        buried = next((morgue_cards[o] for m in members if m["title_read"] for o in morgue_cards
                       if dm.ratio(dm.squash(m["title_read"]), o) >= OLD_NAME), None)
        split = next((name for name in SPLITS for m in members if m["title_read"]
                      and dm.ratio(dm.squash(m["title_read"]), dm.squash(name)) >= OLD_NAME), None)
        known = was or buried or split          # a live card before a rename or a split, or a Morgue card already
        evidence = next((m["card_evidence"] for m in members if m.get("card_evidence")), None) or (
            "the old repo's file" if old_id else None)
        was_played = bool(played or logged or old_id)
        kinds_of = sorted({"joke" if joke else "lost"} | ({"played"} if was_played else {"idea"})
                          | ({"renamed?"} if was else set()) | ({"in the Morgue"} if buried else set())
                          | ({"split"} if split else set())
                          | ({"no card evidence"} if was_played and not joke and not evidence else set()))
        gid = group_id(sha, first.get("cell"))
        found.append({"id": gid, "title": title, "kinds": kinds_of,
                      "morgue_candidate": bool(joke or was_played and evidence) and not known,
                      "ideas_never_ported": not (joke or was_played or known),
                      "old_name_of": was, "morgue_card": buried,
                      "split_of": {"name": split, "into": list(SPLITS[split])} if split else None,
                      "card_evidence": evidence,
                      "first": {"date": when["date"], "channel": when["channel"], "thread": when["thread"],
                                "message": when["message"], "jump_url": when["jump_url"]},
                      "last": max(p["date"] for p in posts), "count": len(members),
                      "channels": dict(Counter(p["channel"] for p in posts)),
                      "pictures": [m["id"] for m in members], "changelog_days": logged, "old_lost": old_id,
                      "looks_like": [c for c, _ in looks.most_common(3)],
                      "nearest": {"look": sorted(by_look.values(), key=lambda n: n["bits"])[:3], "name": by_name},
                      "thumb": first.get("thumb"), "cell": first.get("cell"), "box": first.get("box")})
        for m in members:
            m["lost_group"] = gid
    found.sort(key=lambda g: g["first"]["date"] or "")
    ids = Counter(g["id"] for g in found)
    assert max(ids.values(), default=1) == 1, f"lost card ids not unique: {ids.most_common(3)}"
    return found


# ========================
# Build
# ========================
def build(root: str, log: Callable = print, cards_repo: str = dm.LOCAL_DIR_LOC) -> dict:
    root = dm.check_root(root)
    paths = Paths(root)
    lock = dm.hold(paths.matcher.lock)
    try:
        return _build(paths, log, cards_repo)
    finally:
        lock.close()


def _build(paths: Paths, log: Callable, cards_repo: str = "") -> dict:
    root = paths.root
    pairs = dm.read_json(paths.matcher.pairs, {}) or {}
    if not pairs.get("images"):
        raise SystemExit(f"no {paths.matcher.pairs}: run python -m commands.discord_match score first")
    cards, images = pairs.get("cards") or {}, pairs["images"]
    versions_list = pairs.get("versions") or []
    versions = {v["key"]: v for v in versions_list}
    records, _ = dm.load_posts(root)
    images = {k: i for k, i in images.items() if i["message"] in records}       # left-out channels' posts gone
    channels = {c["id"]: c for c in dm.read_json(join(root, "channels.json"), []) or [] if isinstance(c, dict)}
    morgue = dm.morgue_ids(root, channels) - set(dm.channel_ids(root, dm.LEFT_OUT))
    kinds = channel_kinds(root)
    pictures, blobs, eras = bar_reads(root)
    texts = dm.Cache(paths.matcher.card_cache).entries
    blob_eras = old_eras(root)
    for v in versions_list:                     # each version's marks, bars and era, for the look checks
        feat = v["key"].split("-")[0]
        v.update(marks=version_marks(texts.get(feat) or {}), bars=blobs.get(feat[1:]),
                 reads=version_reads(texts.get(feat) or {}), era=blob_eras.get(feat[1:]))
        v["grams"] = version_grams(v)
    statless, data_stars = card_data(cards_repo)
    git_stars = {k: set(v) for k, v in data_stars.items()}
    for key, found in old_stars(root).items():
        git_stars.setdefault(key, set()).update(found)
    for c in cards.values():
        if c.get("key") in statless:
            c["no_stats"] = True
    close = lookalikes(cards, versions_list)
    family = families(cards, close)
    verdicts, raw = post_verdicts(images, records, cards, dm.read_json(paths.matcher.rulings, {}) or {},
                                  dm.read_json(paths.matcher.ids, {}) or {}, morgue)
    by_sha = defaultdict(list)
    for key in sorted(images, key=lambda k: (int(images[k]["message"]), k)):
        by_sha[images[key]["sha256"]].append(key)
    jokes = joke_pictures(images, by_sha, records, morgue)
    classes = catalogue_classes(root)
    run = Run(images, records, cards, versions, morgue, verdicts, raw, dm.catalogue_sheets(root), close)
    run.classes, run.bars = classes, pictures
    run.local = {r["id"]: r for r in dm.read_jsonl(join(root, *LOCAL)) if not r.get("error")}
    text_eras, run.stars = text_reads(root)
    run.eras = {**text_eras, **{sha: e for sha, e in eras.items() if e}}
    run.git_stars = {cid: git_stars[c["key"]] for cid, c in cards.items() if git_stars.get(c.get("key"))}
    rows = []
    for sha, keys in by_sha.items():
        image = images[keys[0]]
        if image.get("cells"):
            rows += judge_cells(sha, keys, sha in jokes, run)
        else:
            rows.append(judge_picture(sha, keys, sha in jokes, run))
    eye = (dm.read_json(join(root, *EYE), {}) or {}).get("pictures") or {}
    owner = dm.read_json(join(root, *OWNER), {}) or {}
    for row in rows:
        if row["id"] in eye:
            by_eye(row, eye[row["id"]] or {}, versions, cards)
        by_owner(row, owner, versions, cards)
    hand_over(rows, dm.read_json(join(root, *MORGUE_EYE), {}) or {}, cards, {*eye, *(owner.get("pictures") or {})})
    reposts = repost_groups(rows, images, by_sha)
    memes = meme_dates(images, by_sha, records, morgue)
    meme_copy = meme_files(root)
    morgue_eye = {k for k, e in (dm.read_json(join(root, *MORGUE_EYE), {}) or {}).items()   # the Morgue eye's own
                  if isinstance(e, dict) and e.get("status") not in ("renamed", "live")}   # lost cards
    for row in rows:
        image = images[by_sha[row["sha256"]][0]]
        if group_id(row["sha256"], row.get("cell")) not in morgue_eye:      # its eye's entry: stays the Morgue's
            shown_first(row, image, memes, meme_copy)
        row["size_class"] = classes.get(row["sha256"])
        if row.get("cell") is None:
            row.update(whole=image.get("whole"), art=image.get("art"))
            if row["id"] in reposts:
                row["repost_of"] = reposts[row["id"]]
        else:
            cell = image["cells"][row["cell"] - 1]
            row.update(whole=cell.get("whole"), art=cell.get("art"))
        if row["verdict"] in ("lost", "morgue"):
            row["card_evidence"] = card_evidence(image if row.get("cell") is None else cell, row["size_class"],
                                                 eras.get(row["sha256"]), bool(run.checked.get(row["sha256"])))
        dated = sorted((p for p in row["posts"] if dating(p) or row.get("meme_dated") and not (p["bot"] or p.get("embed"))),
                       key=lambda p: int(p["message"]))
        row["first"] = {k: dated[0][k] for k in ("date", "channel", "thread", "message")} if dated else None
        cid = row.get("card")
        if cid and row["first"] and cards.get(cid, {}).get("fs"):
            row["after_git"] = (dm.epoch(row["first"]["date"]) or 0) >= cards[cid]["fs"]
        if row.get("card") and family.get(row["card"]):
            row["family"] = family[row["card"]]
        row["review"] = REVIEW.get(row["verdict"]) if not row.get("joke_draft") else "a joke picture's draft look"
        if row.get("meme_live"):
            row["review"] = MEME_LIVE
    created_reviews(rows, cards)
    old_lost = (dm.read_json(join(root, *OLD_LOST), {}) or {}).get("cards") or []
    groups = lost_groups(rows, images, by_sha, records, cards, versions_list, kinds, channels, old_lost, root,
                         cards_repo)
    makedirs(paths.out, exist_ok=True)
    order = ["id", "verdict", "card", "cards", "conf", "review", "first", "after_git", "name_only", "created_review",
             "eye", "owner", "no_created", "joke", "meme_repost", "meme_live", "meme_dated", "handed", "family", "version",
             "methods", "maybe",
             "looks_like", "lost_group", "reason", "repost_of", "twin_of",
             "sha256", "cell", "box", "sheet", "file", "thumb", "w", "h", "shape", "size_class", "whole", "art",
             "title_read", "card_evidence", "joke_draft", "shown_first", "meme_file", "posts"]
    rows.sort(key=lambda r: (min(int(p["message"]) for p in r["posts"]), r.get("cell") or 0))
    dm.write_text(paths.verdicts, "".join(dm.dumps({k: r[k] for k in order if r.get(k) not in (None, [], "")} |
                                                   {"joke": r["joke"]}, ensure_ascii=False) + "\n" for r in rows))
    dm.write_json(paths.lost, {"generated": dm.now_iso(), "about": LOST_ABOUT, "groups": groups})
    sheets_out(paths, rows, images, by_sha, root)
    tally = summary(rows, groups, images)
    dm.write_json(join(paths.out, "summary.json"), tally)
    log(dm.dumps(tally, indent=1))
    return tally


def by_eye(row: dict, ruling: dict, versions: dict, cards: dict, by: str = "eye") -> None:
    """A picture told by eye (derived/match/eye.json: {pictures: {id: {card, look, why}}}): card a live card's id
    ('none': no card), look 'draft' or the git version key it shows (none: not told). Applied over the gates'
    verdict; never to a joke picture's Morgue verdict. by 'owner': the owner's ruling (derived/review/owner.json,
    same shape; no_created: the picture sets no Created, a weak link the owner kept)."""
    cid, look = ruling.get("card", row.get("card")), ruling.get("look")
    if row["verdict"] in ("morgue", "lost") and cid not in cards:
        return
    why = f"{by}: " + (ruling.get("why") or f"told by {by}")
    if by != "eye" and cid in cards:
        row["owner"] = why
        if ruling.get("no_created"):
            row["no_created"] = ruling["no_created"]
    if cid == "none":
        for k in ("card", "cards", "conf", "methods", "version", "name_only"):
            row.pop(k, None)
        row.update(verdict="none", reason=why)
        return
    if cid not in cards:
        return
    v = versions.get(look) if look not in (None, "draft") else None
    if v is not None and v.get("card") != cid:
        v = None
    note = {"shows": "draft", "by": by, "commits": [], "git_dates": [], "keys": []} if look == "draft" else \
        {"shows": "current" if v.get("look", 0) == 0 else "older", "by": by, "look": v.get("look", 0),
         "keys": [look], "commits": [v["commit"]] if v.get("commit") else [],
         "git_dates": [dm.iso(v["ts"])[:10]] if v.get("ts") else []} if v else \
        {**(row.get("version") or {}), "shows": None, "unsure": "the card told by eye, not its look"}
    row.pop("name_only", None)
    row.update(verdict="draft" if look == "draft" else "version_sure" if v else "version_unsure", card=cid,
               cards=[cid], conf=1.0, version=note, reason=why, **({"eye": why} if by == "eye" else {}))
    for k in ("maybe", "meme_live", "looks_like"):
        row.pop(k, None)
    if ruling.get("meme_dated"):                 # a meme channel's picture told the live card's: dated by those posts
        row.update(joke=False, meme_dated=True)
        row.pop("joke_draft", None)


def hand_over(rows: list, morgue_eye: dict, cards: dict, told: set) -> None:
    """The Morgue eye's 'live' and 'renamed' entries (no candidates: a live card's picture, its early or renamed look)
    hand their pictures (the entry's own, its 'add' list) to that card: card told, look not (its review: 'look not
    told', a 'joke picture' first when posted in meme channels), dated by any post, meme ones too. A picture the eye
    or the owner told keeps that ruling."""
    entries = {k: e for k, e in morgue_eye.items() if isinstance(e, dict) and e.get("status") in ("live", "renamed")
               and e.get("card") in cards}
    for row in rows:
        if row["id"] in told or row["verdict"] not in ("morgue", "lost", "unsure", "none", "art"):
            continue
        gid = group_id(row["sha256"], row.get("cell"))
        key = next((k for k, e in sorted(entries.items()) if row["id"] in (k, *(e.get("add") or ())) or k == gid), None)
        if key:
            cid = entries[key]["card"]
            row.update(verdict="version_unsure", card=cid, cards=[cid], conf=1.0, meme_dated=True, handed=key,
                       version={"shows": None, "unsure": f"the Morgue's eye: {cid}'s picture ({key}), not its look"})
            for k in ("maybe", "meme_live", "looks_like", "name_only"):
                row.pop(k, None)


def by_owner(row: dict, owner: dict, versions: dict, cards: dict) -> None:
    """The owner's rulings on a picture (owner.json): its card and look (pictures, over the eye's) and the cards a
    ruling rules out of an untold picture's maybe list (ruled_out)."""
    if row["id"] in (owner.get("pictures") or {}):
        by_eye(row, owner["pictures"][row["id"]] or {}, versions, cards, by="owner")
    out = (owner.get("ruled_out") or {}).get(row["id"])
    if out and row.get("maybe"):
        row["maybe"] = [c for c in row["maybe"] if c not in out.get("cards", ())]


def created_reviews(rows: list, cards: dict):
    """A card named only (name_only) dated before the card's first picture evidence (a card-sure picture with more
    than its name, or its first git record) sets Created only through review: created_review says so."""
    start = {cid: c["fs"] for cid, c in cards.items() if c.get("fs")}
    for row in rows:
        if row["verdict"] in SURE_CLASSES and row["first"] and not row.get("name_only"):
            when = dm.epoch(row["first"]["date"]) or 0
            start[row["card"]] = min(start.get(row["card"], when), when)
    for row in rows:
        if row.get("name_only") and row["verdict"] in SURE_CLASSES and row["first"] and \
                (dm.epoch(row["first"]["date"]) or 0) < start.get(row["card"], float("inf")):
            evidence = dm.iso(start[row["card"]])[:10] if row["card"] in start else "none"
            row["created_review"] = f"named only, before its first picture evidence ({evidence})"


def summary(rows: list, groups: list, images: dict) -> dict:
    pictures = [r for r in rows if r.get("cell") is None]
    return {"generated": dm.now_iso(), "post_images": len(images), "pictures": len(pictures),
            "sheet_cells": sum(1 for r in rows if r.get("cell") is not None),
            "sheets": len({r["sha256"] for r in rows if r.get("cell") is not None}),
            "verdicts": dict(Counter(r["verdict"] for r in rows)),
            "cards_sure": len({r["card"] for r in rows if r["verdict"] in SURE_CLASSES}),
            "sure_after_git": sum(1 for r in rows if r["verdict"] in SURE_CLASSES and r.get("after_git")),
            "jokes": sum(1 for r in rows if r["joke"]), "meme_reposts": sum(1 for r in rows if r.get("meme_repost")),
            "joke_drafts": sum(1 for r in rows if r.get("joke_draft")),
            "meme_live": sum(1 for r in rows if r.get("meme_live")),
            "name_only": sum(1 for r in rows if r.get("name_only")),
            "created_review": sum(1 for r in rows if r.get("created_review")),
            "families": dict(Counter(r["family"] for r in rows if r.get("family") and r["verdict"] in SURE_CLASSES)),
            "lost_groups": len(groups), "morgue_candidates": sum(1 for g in groups if g["morgue_candidate"]),
            "renamed_hints": sum(1 for g in groups if g.get("old_name_of")),
            "in_the_morgue": sum(1 for g in groups if g.get("morgue_card")),
            "ideas_never_ported": sum(1 for g in groups if g["ideas_never_ported"]),
            "lost_group_kinds": dict(Counter("+".join(g["kinds"]) for g in groups)),
            "thresholds": {**gates(), "matcher": dm.thresholds()}}


def gates() -> dict:
    return {"sure": SURE, "shot_look": SHOT_LOOK, "family_title": FAMILY_TITLE, "clash_bits": CLASH_BITS,
            "sure_bits": SURE_BITS, "sure_fit": SURE_FIT, "sure_novel": SURE_NOVEL, "sure_missing": SURE_MISSING,
            "far": [FAR_RUN, FAR_BITS, FAR_FIT], "group_title": GROUP_TITLE, "grace_days": GRACE_DAYS,
            "exact_run": EXACT_RUN, "lookalike_bits": LOOKALIKE_BITS, "common_word": COMMON_WORD,
            "name_fit": NAME_FIT}


def sheets_out(paths: Paths, rows: list, images: dict, by_sha: dict, root: str):
    """sheets/sheets.json and each sheet with its cells drawn (small webp)."""
    from PIL import ImageDraw
    makedirs(paths.sheets, exist_ok=True)
    found, checked = [], dm.catalogue_sheets(root)
    for sha in sorted({r["sha256"] for r in rows if r.get("cell") is not None}):
        cells = sorted((r for r in rows if r["sha256"] == sha and r.get("cell") is not None), key=lambda r: r["cell"])
        image = images[by_sha[sha][0]]
        name = f"{picture_id(sha)}.webp"
        try:
            img = dm.trim(dm.load_image(join(root, image["file"])))
            img.thumbnail((900, 900))
            draw = ImageDraw.Draw(img)
            for r in cells:
                x0, y0, x1, y1 = r["box"]
                w, h = img.size
                draw.rectangle((x0 * w + 1, y0 * h + 1, x1 * w - 2, y1 * h - 2), outline=(255, 0, 255), width=3)
                draw.text((x0 * w + 6, y0 * h + 6), f"c{r['cell']}", fill=(255, 0, 255))
            img.save(join(paths.sheets, name), "WEBP", quality=70)
        except Exception as e:                  # a picture that won't open: listed without its drawing
            name = f"error: {type(e).__name__}"
        found.append({"id": picture_id(sha), "sha256": sha, "grid": image.get("sheet"), "drawing": name,
                      "found_by": "catalogue, checked by eye" if checked.get(sha) else "names in a row",
                      "first": cells[0]["first"], "cells": [{k: r.get(k) for k in ("id", "cell", "box", "verdict",
                                                                                   "card", "conf", "maybe")}
                                                            for r in cells]})
    dm.write_json(join(paths.sheets, "sheets.json"), {"generated": dm.now_iso(), "sheets": found})


# ========================
# Checks (contact sheets for checking by eye)
# ========================
def _font(size: int):
    from PIL import ImageFont
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def tile(row: dict, root: str, versions: dict, cards: dict, side: int = 260):
    """A post picture (or a sheet cell) beside the version it matched, labelled."""
    from PIL import Image, ImageDraw
    img = dm.trim(dm.load_image(join(root, row["file"])))
    if row.get("box"):
        w, h = img.size
        x0, y0, x1, y1 = row["box"]
        img = img.crop((round(x0 * w), round(y0 * h), round(x1 * w), round(y1 * h)))
    img.thumbnail((side, side * 2))
    ref = None
    keys = (row.get("version") or {}).get("keys") or []
    thumb = (versions.get(keys[0]) or {}).get("thumb") if keys else None
    if not thumb and row.get("card"):
        thumb = dm.card_thumb(versions, row["card"])
    if not thumb and (row.get("maybe") or row.get("looks_like")):
        thumb = dm.card_thumb(versions, (row.get("maybe") or row.get("looks_like"))[0])
    if thumb:
        try:
            ref = Image.open(join(root, *dm.OUT, thumb)).convert("RGB")
            ref.thumbnail((side, side * 2))
        except OSError:
            ref = None
    height = max(img.height, ref.height if ref else 0)
    canvas = Image.new("RGB", (side * 2 + 6, height + 58), (30, 31, 34))
    canvas.paste(img, (0, 0))
    if ref:
        canvas.paste(ref, (side + 6, 0))
    draw = ImageDraw.Draw(canvas)
    card = row.get("card") or ",".join((row.get("maybe") or row.get("looks_like") or [])[:2]) or "-"
    name = (cards.get(card) or {}).get("name") or card
    version = row.get("version") or {}
    draw.text((4, height + 4), f"{row['id']}  {row['verdict']}  {row.get('conf', '')}", fill=(240, 240, 240),
              font=_font(15))
    draw.text((4, height + 22), f"{name[:34]}", fill=(255, 210, 90), font=_font(16))
    draw.text((4, height + 41), f"{(row.get('first') or {}).get('date', '')[:10]} {row['posts'][0]['channel'] or ''}"
              f" {version.get('by') or ''} {','.join(version.get('git_dates') or [])[:24]}", fill=(180, 180, 190),
              font=_font(13))
    return canvas


def contact(rows: list, path: str, root: str, versions: dict, cards: dict, per_row: int = 4):
    from PIL import Image
    tiles = []
    for r in rows:
        try:
            tiles.append(tile(r, root, versions, cards))
        except Exception as e:                  # a file that won't open: a blank tile, named
            from PIL import ImageDraw
            t = Image.new("RGB", (526, 120), (60, 20, 20))
            ImageDraw.Draw(t).text((4, 4), f"{r['id']}: {type(e).__name__}", fill=(255, 255, 255))
            tiles.append(t)
    if not tiles:
        return
    w = max(t.width for t in tiles)
    lines = [tiles[i:i + per_row] for i in range(0, len(tiles), per_row)]
    heights = [max(t.height for t in line) for line in lines]
    sheet = Image.new("RGB", (w * per_row + 8 * (per_row - 1), sum(heights) + 8 * (len(lines) - 1)), (0, 0, 0))
    y = 0
    for line, h in zip(lines, heights):
        for i, t in enumerate(line):
            sheet.paste(t, (i * (w + 8), y))
        y += h + 8
    makedirs(dirname(path), exist_ok=True)
    sheet.save(path, "JPEG", quality=82)


def sheet_contact(sheets: list, path: str, paths: Paths, cards: dict, side: int = 520):
    """Sheets with their cells drawn, each cell's verdict and card under it."""
    from PIL import Image, ImageDraw
    tiles = []
    for sheet in sheets:
        try:
            img = Image.open(join(paths.sheets, sheet["drawing"])).convert("RGB")
        except OSError:
            img = Image.new("RGB", (side, 80), (60, 20, 20))
        img.thumbnail((side, side))
        first = (sheet.get("first") or {}).get("date", "")[:10]
        lines = [f"{sheet['id']} {sheet['grid']} {sheet['found_by'][:9]} {first}"]
        for c in sheet["cells"]:
            name = (cards.get(c.get("card")) or {}).get("name") or ",".join(c.get("maybe") or [])[:30]
            lines.append(f"c{c['cell']} {c['verdict']} {name}")
        tile = Image.new("RGB", (side, img.height + 18 * len(lines) + 6), (30, 31, 34))
        tile.paste(img, (0, 0))
        draw = ImageDraw.Draw(tile)
        for i, line in enumerate(lines):
            draw.text((4, img.height + 3 + 18 * i), line[:70], fill=(240, 240, 240) if i == 0 else (255, 210, 90),
                      font=_font(14))
        tiles.append(tile)
    per_row = 4
    rows_of_tiles = [tiles[i:i + per_row] for i in range(0, len(tiles), per_row)]
    heights = [max(t.height for t in line) for line in rows_of_tiles]
    out = Image.new("RGB", (side * per_row + 8 * (per_row - 1), sum(heights) + 8 * len(heights)), (0, 0, 0))
    y = 0
    for line, h in zip(rows_of_tiles, heights):
        for i, t in enumerate(line):
            out.paste(t, (i * (side + 8), y))
        y += h + 8
    out.save(path, "JPEG", quality=82)


def checks(root: str, seed: int = 1113, sure: int = 50, lost: int = 25, versions_n: int = 24,
           log: Callable = print) -> dict:
    """Contact sheets: sure matches at random, git looks claimed before git at random, every look-alike family's
    sure matches, lost pictures at random, every sheet's cells."""
    root = dm.check_root(root)
    paths = Paths(root)
    pairs = dm.read_json(paths.matcher.pairs, {}) or {}
    cards, versions = pairs.get("cards") or {}, {v["key"]: v for v in pairs.get("versions") or []}
    rows = dm.read_jsonl(paths.verdicts)
    rng = Random(seed)
    makedirs(paths.checks, exist_ok=True)
    picked = {"seed": seed}
    sure_rows = [r for r in rows if r["verdict"] in SURE_CLASSES]
    sample = rng.sample(sure_rows, min(sure, len(sure_rows)))
    picked["sure"] = [r["id"] for r in sample]
    for n in range(0, len(sample), 12):
        contact(sample[n:n + 12], join(paths.checks, f"sure-{n // 12 + 1}.jpg"), root, versions, cards)
    looks = [r for r in sure_rows if r["verdict"] == "version_sure" and r.get("after_git") is False]
    sample = rng.sample(looks, min(versions_n, len(looks)))     # git looks claimed before git: what dates versions
    picked["version_sure"] = [r["id"] for r in sample]
    for n in range(0, len(sample), 12):
        contact(sample[n:n + 12], join(paths.checks, f"looks-{n // 12 + 1}.jpg"), root, versions, cards)
    fam = defaultdict(list)
    for r in sure_rows:
        if r.get("family"):
            fam[r["family"]].append(r)
    picked["families"] = {}
    for name, members in sorted(fam.items()):
        safe = dm.slugify(name)
        picked["families"][name] = [r["id"] for r in members]
        for n in range(0, len(members), 16):
            contact(members[n:n + 16], join(paths.checks, f"family-{safe}-{n // 16 + 1}.jpg"), root, versions, cards)
    lost_rows = [r for r in rows if r["verdict"] == "lost"]
    sample = rng.sample(lost_rows, min(lost, len(lost_rows)))
    picked["lost"] = [r["id"] for r in sample]
    for n in range(0, len(sample), 13):
        contact(sample[n:n + 13], join(paths.checks, f"lost-{n // 13 + 1}.jpg"), root, versions, cards)
    sheets = (dm.read_json(join(paths.sheets, "sheets.json"), {}) or {}).get("sheets") or []
    picked["sheets"] = [s["id"] for s in sheets]
    for n in range(0, len(sheets), 12):
        sheet_contact(sheets[n:n + 12], join(paths.checks, f"sheets-{n // 12 + 1}.jpg"), paths, cards)
    dm.write_json(join(paths.checks, "samples.json"), picked)
    log(f"checks: {len(picked['sure'])} sure, {sum(len(v) for v in picked['families'].values())} in "
        f"{len(picked['families'])} families, {len(picked['lost'])} lost -> {paths.checks}")
    return picked


def local_diff(a, b, box: int = EYE_BOX) -> tuple:
    """(the largest mean of |a - b| over a box x box window, grey levels; that window's (x, y)) of two same-size RGB
    arrays."""
    np = dm._np()
    d = np.abs(a.astype(float) - b.astype(float)).mean(axis=2)
    ii = np.pad(d, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    m = (ii[box:, box:] - ii[:-box, box:] - ii[box:, :-box] + ii[:-box, :-box]) / box ** 2
    y, x = np.unravel_index(int(np.argmax(m)), m.shape)
    return float(m.max()), (int(x), int(y)), m


def eye_keys(root: str, cards_repo: str = dm.LOCAL_DIR_LOC, log: Callable = print) -> list:
    """Every eye ruling naming a git version: picture | version | difference, the 3 most different windows zoomed,
    into checks/eye-keys/ (+ eye_keys.json, largest local difference first)."""
    from io import BytesIO
    from os.path import expanduser
    from PIL import Image, ImageDraw, ImageFilter
    np = dm._np()
    root = dm.check_root(root)
    paths = Paths(root)
    eye = (dm.read_json(join(root, *EYE), {}) or {}).get("pictures") or {}
    keys = {pid: r["look"] for pid, r in eye.items() if r.get("look") not in (None, "draft")}
    versions = {v["key"]: v for v in (dm.read_json(paths.matcher.pairs, {}) or {}).get("versions") or []}
    rows = {r["id"]: r for r in dm.read_jsonl(paths.verdicts) if r["id"] in keys}
    stores = {True: expanduser(dm.old_store()), False: dm.git_dir_of(cards_repo)}
    out = join(paths.checks, "eye-keys")
    makedirs(out, exist_ok=True)
    found = []
    for n, (pid, key) in enumerate(sorted(keys.items())):
        v, row = versions.get(key), rows.get(pid)
        if not v or not row:
            found.append({"id": pid, "key": key, "missing": "version" if not v else "verdict row"})
            continue
        data = dm.git_out(stores[bool(v.get("old"))], "cat-file", "blob", f"{v['commit']}:{v['path']}") \
            if stores[bool(v.get("old"))] else None
        ref = dm.load_image(BytesIO(data) if data else join(root, *dm.OUT, v["thumb"])).convert("RGB")
        pic = dm.load_image(join(root, row["file"])).convert("RGB")
        if row.get("box"):
            w, h = pic.size
            x0, y0, x1, y1 = row["box"]
            pic = pic.crop((round(x0 * w), round(y0 * h), round(x1 * w), round(y1 * h)))
        size = (EYE_W, round(EYE_W * ref.height / ref.width))
        a, b = (im.resize(size, Image.LANCZOS) for im in (pic, ref))
        score, _, m = local_diff(*(np.asarray(im.filter(ImageFilter.GaussianBlur(EYE_BLUR))) for im in (a, b)))
        windows, m = [], m.copy()
        for _ in range(3):
            y, x = np.unravel_index(int(np.argmax(m)), m.shape)
            windows.append((int(x), int(y)))
            m[max(0, y - 80):y + 80, max(0, x - 80):x + 80] = -1
        diff = np.clip(np.abs(np.asarray(a).astype(int) - np.asarray(b).astype(int)).mean(axis=2) * 4, 0, 255)
        z, w, h = 290, *size
        sheet = Image.new("RGB", (3 * w + 20, h + 40 + z), "white")
        for i, im in enumerate((a, b, Image.fromarray(diff.astype("uint8")).convert("RGB"))):
            sheet.paste(im, (i * (w + 10), 30))
        draw = ImageDraw.Draw(sheet)
        draw.text((4, 4), f"{pid} {row.get('card')} {(row.get('first') or {}).get('date', '')[:16]} | {key[:8]} "
                          f"{'old repo' if v.get('old') else 'card repo'} {v['commit'][:7]} "
                          f"({'blob' if data else 'thumb'}) | local {score:.1f} | zooms: picture, version", fill="red")
        for i, (x, y) in enumerate(windows):
            x0 = max(0, min(w - 145, x + EYE_BOX // 2 - 72))
            y0 = max(0, min(h - 145, y + EYE_BOX // 2 - 72))
            draw.rectangle((2 * w + 20 + x0, 30 + y0, 2 * w + 165 + x0, 175 + y0), outline="red", width=2)
            for j, im in enumerate((a, b)):
                sheet.paste(im.crop((x0, y0, x0 + 145, y0 + 145)).resize((z, z), Image.LANCZOS),
                            ((2 * i + j) * (z + 6), h + 36))
        name = f"{n:03d}-{pid}.jpg"
        sheet.save(join(out, name), quality=85)
        found.append({"id": pid, "card": row.get("card"), "key": key, "ref": "blob" if data else "thumb",
                      "local": round(score, 1), "sheet": name})
    found.sort(key=lambda f: -f.get("local", 1e9))
    dm.write_json(join(paths.checks, "eye_keys.json"), found)
    log(f"eyekeys: {len(found)} eye rulings naming a git version ({sum(f.get('ref') == 'thumb' for f in found)} "
        f"against a thumb); largest local differences: " +
        ", ".join(f"{f.get('card')} {f.get('local')}" for f in found[:6]) + f" -> {out}")
    return found


# ========================
# Local differences (eyekeys' score against every stored version of the card)
# ========================
class Store:
    """A git store's blobs through one cat-file --batch (read only, never a lazy fetch); None for one it lacks."""
    def __init__(self, git_dir: str):
        self.git_dir, self.p = git_dir, None

    def get(self, spec: str) -> Optional[bytes]:
        from subprocess import PIPE, Popen
        if self.p is None:
            self.p = Popen(["git", "--git-dir", self.git_dir, "-c", "core.fsmonitor=false", "cat-file", "--batch"],
                           stdin=PIPE, stdout=PIPE, env={**environ, "GIT_NO_LAZY_FETCH": "1"})
        self.p.stdin.write(spec.encode() + b"\n")
        self.p.stdin.flush()
        head = self.p.stdout.readline().split()
        if len(head) != 3 or not head[2].isdigit():
            return None                        # missing or ambiguous
        data = self.p.stdout.read(int(head[2]))
        self.p.stdout.read(1)
        return data if head[1] == b"blob" else None


def eye_array(img, size: tuple):
    """img at size, blurred as eyekeys compares it."""
    from PIL import Image, ImageFilter
    return dm._np().asarray(img.resize(size, Image.LANCZOS).filter(ImageFilter.GaussianBlur(EYE_BLUR)))


def ref_arrays(specs: list) -> tuple:
    """({key: (its size, its eye array at EYE_W)}, [keys no store holds]) of [(key, git dir, 'commit:path')]."""
    from io import BytesIO
    stores, refs, missing = {}, {}, []
    for key, git_dir, spec in specs:
        data = stores.setdefault(git_dir, Store(git_dir)).get(spec) if git_dir else None
        if not data:
            missing.append(key)
            continue
        ref = dm.load_image(BytesIO(data))
        refs[key] = (ref.size, eye_array(ref, (EYE_W, round(EYE_W * ref.height / ref.width))))
    for store in stores.values():
        if store.p:
            store.p.stdin.close()
            store.p.wait()
    return refs, missing


def local_scores(pic, refs: dict, keys=None) -> dict:
    """{key: [local diff, aspect ratio (picture / version)]} of a picture against refs (ref_arrays)."""
    sized, out = {}, {}
    for key in keys if keys is not None else refs:
        if key not in refs:
            continue
        (w, h), arr = refs[key]
        size = arr.shape[1], arr.shape[0]
        if size not in sized:
            sized[size] = eye_array(pic, size)
        out[key] = [round(local_diff(sized[size], arr)[0], 1), round((pic.width / pic.height) / (w / h), 4)]
    return out


def aligned_diff(a, b, margin: int = LOCAL_MARGIN) -> Optional[float]:
    """eyekeys' score after an affine alignment of a onto b (ECC at a quarter of the size; the border's margin left
    out): a copy cropped or padded by a few pixels lines up, a drawn difference stays. None when ECC fails."""
    import cv2
    np = dm._np()
    h, w = b.shape[:2]
    small = (w // 4, h // 4)
    ga, gb = (cv2.resize(x.astype(np.float32).mean(axis=2), small, interpolation=cv2.INTER_AREA) for x in (a, b))
    warp = np.eye(2, 3, dtype=np.float32)
    try:
        _, warp = cv2.findTransformECC(gb, ga, warp, cv2.MOTION_AFFINE,
                                       (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-5), None, 3)
    except cv2.error:
        return None
    warp[:, 2] *= 4
    moved = cv2.warpAffine(a.astype(np.float32), warp, (w, h), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                           borderMode=cv2.BORDER_REPLICATE)
    return local_diff(moved[margin:-margin, margin:-margin], b[margin:-margin, margin:-margin].astype(np.float32))[0]


def local_task(job: dict) -> list:
    """A worker: one card's pictures against the versions they lack; the LOCAL_ALIGN nearest with the picture's
    aspect ratio over LOCAL_SAME scored again aligned (the lower score kept)."""
    refs, missing = ref_arrays(job["versions"])
    rows = []
    for p in job["pictures"]:
        if not any(k in refs for k in [*p["keys"], *p.get("have", {})]):
            rows.append({"id": p["id"], "card": job["card"], "versions": {}, "missing": p["keys"], "aligned": []})
            continue
        try:
            pic = dm.load_image(join(job["root"], p["file"]))
        except Exception as e:
            rows.append({"id": p["id"], "card": job["card"], "error": f"{type(e).__name__}: {str(e)[:120]}"})
            continue
        if p.get("box"):
            w, h = pic.size
            x0, y0, x1, y1 = p["box"]
            pic = pic.crop((round(x0 * w), round(y0 * h), round(x1 * w), round(y1 * h)))
        got = local_scores(pic, refs, p["keys"])
        every = {**p.get("have", {}), **got}
        near = sorted((k for k, (sc, ar) in every.items() if abs(ar - 1) <= LOCAL_AR and sc > LOCAL_SAME and k in refs
                       and k not in p.get("aligned_before", [])), key=lambda k: every[k][0])[:LOCAL_ALIGN]
        for k in near:
            (w, h), arr = refs[k]
            score = aligned_diff(eye_array(pic, (arr.shape[1], arr.shape[0])), arr)
            if score is not None and score < every[k][0]:
                got[k] = [round(score, 1), every[k][1]]
            elif k not in got:
                got[k] = every[k]
        rows.append({"id": p["id"], "card": job["card"], "versions": got,
                     "missing": [k for k in p["keys"] if k in missing], "aligned": near})
    return rows


def local(root: str, cards_repo: str = dm.LOCAL_DIR_LOC, workers: int = 3, log: Callable = print,
          before_git: bool = False) -> dict:
    """local.jsonl: each card-sure picture of verdicts.jsonl against every stored git version of its card (old repo,
    card repo: never a lazy fetch), eyekeys' local diff and the aspect ratio; cached (only new pairs computed, the
    unstored tried again). before_git: only pictures posted before their card's first git record (the timelines'
    part; a picture left with a version uncompared is never a local draft)."""
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from multiprocessing import get_context
    from os.path import expanduser
    root = dm.check_root(root)
    paths = Paths(root)
    members = defaultdict(list)
    pairs = dm.read_json(paths.matcher.pairs, {}) or {}
    for v in pairs.get("versions") or []:
        members[v["card"]].append(v)
    first = {cid: c.get("fs") for cid, c in (pairs.get("cards") or {}).items()}
    stores = {True: expanduser(dm.old_store()), False: dm.git_dir_of(cards_repo)}
    path = join(root, *LOCAL)
    have = {r["id"]: r for r in dm.read_jsonl(path)}
    todo = defaultdict(list)
    for row in dm.read_jsonl(paths.verdicts):
        cid = row.get("card")
        if row["verdict"] not in SURE_CLASSES or not cid:
            continue
        if before_git and first.get(cid) and (dm.epoch((row.get("first") or {}).get("date")) or 0) >= first[cid]:
            continue
        old = have.get(row["id"]) if (have.get(row["id"]) or {}).get("card") == cid else None
        done = set((old or {}).get("versions") or {})
        need = [v for v in members.get(cid, []) if v["key"] not in done]
        if need or (old or {}).get("aligned") is None:
            todo[cid].append((row, need, old or {}))
    jobs = []
    for cid, items in sorted(todo.items()):
        keys = {v["key"]: v for v in members.get(cid, [])}       # every version: the nearest get aligned
        jobs.append({"root": root, "card": cid, "versions": [(k, stores[bool(v.get("old"))],
                                                              f"{v['commit']}:{v['path']}") for k, v in keys.items()],
                     "pictures": [{"id": r["id"], "file": r["file"], "box": r.get("box"),
                                   "keys": [v["key"] for v in need], "have": old.get("versions") or {},
                                   "aligned_before": old.get("aligned") or []} for r, need, old in items]})
    log(f"local: {sum(len(j['pictures']) for j in jobs)} pictures of {len(jobs)} cards to compare")
    done = 0
    with ProcessPoolExecutor(max_workers=max(1, workers), mp_context=get_context("spawn")) as pool:
        futures = [pool.submit(local_task, j) for j in jobs]
        for i, fut in enumerate(as_completed(futures), 1):
            for r in fut.result():
                old = have.get(r["id"]) if (have.get(r["id"]) or {}).get("card") == r["card"] else {}
                if "error" in r:
                    have[r["id"]] = {**old, **r} if old else r
                    continue
                have[r["id"]] = {"id": r["id"], "card": r["card"],
                                 "versions": {**(old.get("versions") or {}), **r["versions"]},
                                 "missing": sorted(r["missing"]),
                                 "aligned": sorted({*(old.get("aligned") or []), *r["aligned"]})}
                done += 1
            if i % 25 == 0 or i == len(jobs):
                log(f"local: {i}/{len(jobs)} cards")
                dm.write_text(path, "".join(dm.dumps(have[k], ensure_ascii=False) + "\n" for k in sorted(have)))
    dm.write_text(path, "".join(dm.dumps(have[k], ensure_ascii=False) + "\n" for k in sorted(have)))
    rows = list(have.values())
    tally = {"pictures": len(rows), "compared now": done, "with a missing version": sum(1 for r in rows if r.get(
        "missing")), "errors": sum(1 for r in rows if r.get("error"))}
    log(f"local: {tally} -> {path}")
    return tally


def localtest(root: str, cards_repo: str = dm.LOCAL_DIR_LOC, workers: int = 3, log: Callable = print,
              sample: int = 0) -> dict:
    """The local gate on discord_match selftest's Discord and small copies (dm.degrade) of every stored version of
    the cards with 2+ looks, against every stored version of the card: how many land at LOCAL_SAME on another look
    (wrong), on their own (right), in between, and at LOCAL_DRAFT from every one (drafted: must be 0); the cut
    copies' aspect check. Into derived/match/local_selftest.json (+ the copies between the bars, for the eye)."""
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from multiprocessing import get_context
    from os.path import expanduser
    root = dm.check_root(root)
    paths = Paths(root)
    pairs = dm.read_json(paths.matcher.pairs, {}) or {}
    members = defaultdict(list)
    for v in pairs.get("versions") or []:
        members[v["card"]].append(v)
    stores = {True: expanduser(dm.old_store()), False: dm.git_dir_of(cards_repo)}
    jobs = [{"card": cid, "looks": {v["key"]: v.get("look", 0) for v in vs},
             "versions": [(v["key"], stores[bool(v.get("old"))], f"{v['commit']}:{v['path']}") for v in vs]}
            for cid, vs in sorted(members.items()) if len({v.get("look", 0) for v in vs}) > 1]
    if sample:                                 # a seeded share of the cards, every copy's row kept
        jobs = Random(1113).sample(jobs, min(sample, len(jobs)))
    out, between, every, worst = {name: Counter() for name in ("discord", "small", "cut")}, [], [], {}
    with ProcessPoolExecutor(max_workers=max(1, workers), mp_context=get_context("spawn")) as pool:
        futures = [pool.submit(localtest_task, j) for j in jobs]
        for i, fut in enumerate(as_completed(futures), 1):
            for r in fut.result():
                out[r["variant"]][r["outcome"]] += 1
                if r["best"] is not None:                # the farthest a copy got from every stored version
                    worst[r["variant"]] = max(worst.get(r["variant"], 0.0), r["best"])
                if r["outcome"] == "between" and r["variant"] != "cut":
                    between.append(r)
                if sample:
                    every.append(r)
            if i % 50 == 0 or i == len(jobs):
                log(f"localtest: {i}/{len(jobs)} cards")
    between.sort(key=lambda r: -r["own"])
    report = {"generated": dm.now_iso(), "bars": {"same": LOCAL_SAME, "keep": LOCAL_KEEP, "draft": LOCAL_DRAFT,
                                                  "draft_narrow": LOCAL_DRAFT_NARROW, "aspect": LOCAL_AR},
              "copies": {k: dict(c) for k, c in out.items()}, "farthest": worst, "between": between[:400]}
    if sample:
        report["every"] = every
    dm.write_json(join(paths.out, "local_selftest" + ("_sample" if sample else "") + ".json"), report)
    log(dm.dumps({"copies": report["copies"], "farthest": worst}, indent=1))
    return report


def localtest_task(job: dict) -> list:
    """A worker: one card's versions as Discord, small and cut copies, each against every stored version."""
    from io import BytesIO
    refs, _ = ref_arrays(job["versions"])
    stores, rows = {}, []
    for key, git_dir, spec in job["versions"]:
        if key not in refs:
            continue
        img = dm.load_image(BytesIO(stores.setdefault(git_dir, Store(git_dir)).get(spec)))
        for name in ("discord", "small", "cut"):
            got = local_scores(dm.degrade(img, **dm.VARIANTS[name]), refs)
            comp = {k: s for k, (s, ar) in got.items() if abs(ar - 1) <= LOCAL_AR}
            own, look = got[key][0], job["looks"][key]
            best = min(comp, key=comp.get) if comp else None
            bar = LOCAL_DRAFT if dm.VARIANTS[name]["width"] >= LOCAL_WIDE else LOCAL_DRAFT_NARROW
            outcome = "no aspect" if not comp else "drafted" if comp[best] >= bar else \
                ("right" if job["looks"][best] == look else "wrong") if comp[best] <= LOCAL_SAME else "between"
            rows.append({"card": job["card"], "key": key, "variant": name, "outcome": outcome, "own": own,
                         "best": comp.get(best), "best_key": best, "w": img.width})
    for store in stores.values():
        if store.p:
            store.p.stdin.close()
            store.p.wait()
    return rows


# ========================
# Self-test (the sure gates on the matcher's degraded copies)
# ========================
def selftest(root: str, cards_repo: str = dm.LOCAL_DIR_LOC, log: Callable = print) -> dict:
    """The verdicts' sure error on discord_match selftest's degraded copies (match/cache/selftest.jsonl; run that
    first): each current card's copies (is the sure card right?) and each version of the cards with 2+ looks (is
    the sure look right?). Into derived/match/selftest.json."""
    root = dm.check_root(root)
    paths = Paths(root)
    cards, found = dm.references(root, cards_repo, "all", bool(dm._vision()), 1, log=log)
    statless = card_data(cards_repo)[0]
    for c in cards.values():
        if c.get("key") in statless:
            c["no_stats"] = True
    index = dm.CardIndex(cards, found)
    looks = index.looks()
    _, blobs, _ = bar_reads(root)
    _, copies, _ = bar_reads(root, "jpeg400")   # a version's Discord-like copy read: a post's bars
    versions = {v["key"]: {**v, "look": looks[v["key"]], "marks": version_marks(v), "reads": version_reads(v),
                           "bars": blobs.get(v["key"].split("-")[0][1:])} for v in index.versions}
    for v in versions.values():
        v["grams"] = version_grams(v)
    members = defaultdict(list)
    for v in versions.values():
        members[v["card"]].append(v)
    close = lookalikes(cards, index.versions)
    family = families(cards, close)
    tags = tag_words(versions.values())
    tests = dm.Cache(join(paths.matcher.cache, "selftest.jsonl"))
    cards_out = {name: Counter() for name in dm.CARD_VARIANTS}
    looks_out = {name: Counter() for name in dm.VERSION_VARIANTS}
    wrong, held = [], defaultdict(Counter)
    many = {cid for cid, vis in index.by_card.items() if len({looks[index.versions[i]["key"]] for i in vis}) > 1}

    def verdict(feat: dict) -> tuple:
        v = dm.judge(dm.as_post(feat, dm.score_image(index, feat)), cards)
        if v["status"] != "confident" or len(v["cards"]) != 1:
            return None, v, v["status"]
        cid = v["cards"][0]
        why = gate(cid, (scored_of(v, cid) or {}).get("conf", 0.0), v, feat.get("shape"), family, close,
                   whole_card(feat)) or name_check(cid, v, dm.read_text(feat.get("lines")), members, cards)
        return (cid, v, "sure") if why is None else (None, v, "gated")

    def held_out(pick: dict, image: dict, others: list) -> tuple:
        """A draft's stand-in (its own look left out): (old gates, made a day before the look it is taken for,
        made 30 days before), each git, draft or None."""
        kinds = [decide(pick, {"shape": image["shape"], "whole": image["whole"], "no_bars": True}, versions,
                        others)[0]]
        kind, note = decide(pick, {**image, "made": None}, versions, others)
        look = note.get("look", note.get("near_look"))
        start = min((m["ts"] for m in others if m.get("look") == look and m.get("ts")), default=None)
        for days in (1, 30):
            made = start - days * 86400 if start is not None else None
            kinds.append(decide(pick, {**image, "made": made}, versions, others)[0] if look is not None else kind)
        return tuple(kinds)
    for v in index.versions:
        variants = (tests.entries.get(v["key"]) or {}).get("variants") or {}
        truth = v["card"]
        for name in dm.CARD_VARIANTS if v.get("current") else ():
            if name not in variants:
                cards_out[name]["unread"] += 1
                continue
            cid, _, status = verdict(variants[name])
            outcome = status if cid is None else "right" if dm.related(cards, cid, truth) else "wrong"
            cards_out[name][outcome] += 1
            if outcome == "wrong":
                wrong.append({"kind": "card", "variant": name, "card": truth, "taken_for": cid})
        for name in dm.VERSION_VARIANTS if truth in many else ():
            if name not in variants:
                looks_out[name]["unread"] += 1
                continue
            feat = variants[name]
            cid, judged, status = verdict(feat)
            if cid is None or not dm.related(cards, cid, truth):
                looks_out[name]["card " + ("wrong" if cid else status)] += 1
                continue
            seen = {"shape": feat.get("shape"), "w": feat.get("w"), "h": feat.get("h"), "whole": feat.get("whole"),
                    "marks": read_marks(feat.get("lines")),
                    "text": dm.read_text(feat.get("lines")),
                    "bars": copies.get(v["key"].split("-")[0][1:]), "made": v.get("ts"), "tags": tags,
                    "no_bars": prints_no_bars(cards.get(truth)),
                    "size_class": "ps1200" if name in ("discord", "small") else None}   # uncut copies: exports
            kind, note = version_of(judged, cid, seen, versions, members[cid])
            outcome = "draft" if kind == "draft" else "unsure" if kind != "git" else \
                "right" if note.get("look") == looks[v["key"]] else "wrong"
            looks_out[name][outcome] += 1
            if name == "discord":               # its own look left out: a draft's stand-in
                own = looks[v["key"]]
                pick = index.pick_version(truth, feat, dm.read_text(feat.get("lines")), own)
                found = held_out(pick, seen, [m for m in members[truth] if m["look"] != own])
                groups = index.groups[truth]
                mine = groups[own][0] - set().union(*(g[1] for other, g in groups.items() if other != own))
                for when, kind in zip(("old gates", "made 1 day before", "made 30 days before"), found):
                    held[when][("far " if len(mine) >= 12 else "near ")
                               + ("draft" if kind == "draft" else "taken" if kind == "git" else "unsure")] += 1
            if outcome == "wrong" and len(wrong) < 80:
                wrong.append({"kind": "look", "variant": name, "card": truth, "version": v["key"],
                              "look": looks[v["key"]], "taken_for": note.get("look"), "by": note.get("by"),
                              "fit": note.get("fit"), "bits": note.get("bits")})
    report = {"generated": dm.now_iso(), "gates": gates(),
              "cards": {name: dict(c) for name, c in cards_out.items()},
              "looks": {name: dict(c) for name, c in looks_out.items()},
              "held_out": {when: dict(c) for when, c in held.items()}, "wrong": wrong}
    for c in report["held_out"].values():      # a draft taken for a git look: the sure error of looks before git
        for side in ("near", "far"):
            n = sum(k for name, k in c.items() if name.startswith(side + " "))
            c[side + " taken share"] = round(c.get(side + " taken", 0) / n, 4) if n else None
    for part in ("cards", "looks"):
        for name, c in report[part].items():
            sure = c.get("right", 0) + c.get("wrong", 0)
            c["sure_error"] = round(c.get("wrong", 0) / sure, 4) if sure else None
    makedirs(paths.out, exist_ok=True)
    dm.write_json(join(paths.out, "selftest.json"), report)
    log(dm.dumps({k: report[k] for k in ("cards", "looks", "held_out")}, indent=1))
    return report


# ========================
# Command line
# ========================
def main(argv: Optional[list] = None) -> int:
    parser = ArgumentParser(prog="python -m commands.discord_verdicts", description="One verdict per Discord picture.")
    subs = parser.add_subparsers(dest="command", required=True)
    for name in ("build", "checks", "selftest", "eyekeys", "local", "localtest"):
        p = subs.add_parser(name)
        p.add_argument("--root", default=environ.get("DISCORD_ARCHIVE", dm.DEFAULT_ROOT))
        if name != "checks":
            p.add_argument("--cards-repo", default=dm.LOCAL_DIR_LOC)
        if name in ("local", "localtest"):
            p.add_argument("--workers", type=int, default=3)
        if name == "local":
            p.add_argument("--before-git", action="store_true", help="only pictures before their card's first git record")
        if name == "localtest":
            p.add_argument("--sample", type=int, default=0, help="this many cards (seeded), every copy kept")
        if name == "checks":
            p.add_argument("--seed", type=int, default=1113)
            p.add_argument("--sure", type=int, default=50)
            p.add_argument("--lost", type=int, default=25)
    args = parser.parse_args(argv)

    def log(text):
        print(text, file=stderr, flush=True)
    if args.command == "build":
        build(args.root, log, args.cards_repo)
    elif args.command == "selftest":
        selftest(args.root, args.cards_repo, log)
    elif args.command == "eyekeys":
        eye_keys(args.root, args.cards_repo, log)
    elif args.command == "local":
        local(args.root, args.cards_repo, args.workers, log, args.before_git)
    elif args.command == "localtest":
        localtest(args.root, args.cards_repo, args.workers, log, args.sample)
    else:
        checks(args.root, args.seed, args.sure, args.lost, log=log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
