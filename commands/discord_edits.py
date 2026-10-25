"""
Discord's card edits as commits to the card repo (docs/Campaigns/DATA_BUILD_ACTION.md section 11, decisions 2 and 3):
/update_metadata, /mass_replace and /alias write TTSCardMaker's site/Card Edits.json and site/Card Authors.txt in one
commit per command; the commit never carries a pickle. It goes through card_bake.land (on main's current head, again from
the new head when main moved) and git_commit (the API with the bot's GIT_TOKEN, else the history clone pushed over SSH: the
path /bake_cards and /update_tts_save commit by). Its subject starts "Discord:", it carries an Edit-by trailer, and each
entry it changes gets _by ("discord:<username>") and _at (one holding PSD fields keeps its stamps when the edit sets
none: their editor keeps the bake credit, and the fields the edit sets go in its _bot with their own credit in _credit);
a value the head already has is no change (no op, no restamp, no commit).
Then the bot applies the change in memory (CardEdits.apply, CardAuthors.resolve, the duplicates' names, the alias registry) and saves stats.pkl, so /query shows it before the next /update_stats or card sync builds it.

Fields: author -> a [cards] line in Card Authors.txt; ability, stars, types, hp/def/atk/spd -> the "psd" kind (they
persist and show "PSD pending" on the website; the next bake prints them, and a stars or types layout it can't follow is
refused for that field alone; a types name no Types/ icon at the head has is refused before the commit); subtype, series, card_type -> subtype, series, type (the "folder" kind); subseries;
del_entry -> hidden: true. /mass_replace author rewrites the right-hand sides in Card Authors.txt where
they stand (old's group renamed) and adds a "[people] old = new" line; its other fields write one entry per card, in one
commit. /alias writes `aliases` on the entry of the card its PNG belongs to. aliases.pkl keeps counting until the
migration sets Card Edits.json's aliases_moved; before that its aliases can't be changed here.

The migration (section 14 step 5, after the parity re-parse): one commit, authored by the bot, of the files `migrate`
writes, aliases.pkl's aliases and the Discord-only values (carried) together. An entry another editor stamped keeps
its _by and _at; the fields the migration adds to it are listed in its _bot, credited to nobody (edit.js and
edits_file drop a field from _bot when they set or unset it):

    python -m commands.discord_edits migrate --clone <full card repo clone> [--ref HEAD] [--aliases aliases.pkl]
        [--stats stats.pkl] --out-dir DIR

Creation dates: the Discord archive's created_dates.json (review_pages.py) as the card repo's site/Card Creation
Dates.json (dates only; "from": "hand" entries kept). Re-run after rulings, review the diff, commit it:

    python -m commands.discord_edits created [--archive ~/Documents/Tools/discord-archive] [--repo <card repo checkout>]
        [--out FILE]

edits_file and authors_file are ports of site/assets/edit.js editsFile and authorsFile.
"""
from argparse import ArgumentParser
from copy import deepcopy
from datetime import datetime, timezone
from difflib import get_close_matches
from json import loads
from os.path import basename
from re import IGNORECASE, compile as re_compile, fullmatch
from subprocess import run as run_process
from traceback import format_exc
from typing import Callable, Optional

from commands.card_edits import ALIASES_MOVED, ORDER, CardAuthors, CardDates, CardEdits, _collate, dump_edits
from commands.card_models import CardType, card_name
from commands.github_api import blob_sha
import global_config                                    # CARD_SYNC, read at call time (the cutover switch)
from global_config import AUTHORS_PATHS, CARD_REPO, CREATED_PATHS, EDITS_PATHS, LOCAL_DIR_LOC

# ========================
# Configuration
# ========================
SUBJECT = "Discord:"                            # every commit's subject starts so (the website's Database edits tab)
EDIT_BY = "Edit-by"                             # the trailer key (card_bake.EDIT_BY, build_site.EDIT_BY)
DISCORD = "discord:"                            # an editor's _by and Edit-by: "discord:<username>" (card_bake.DISCORD)
BOT = "AutoCroissant"                           # the migration's _by (build_site.BOT_AUTHOR: counted for nobody)
BOT_FIELDS = "_bot"                             # an entry's fields the migration set under another editor's _by (edit.js, build_site)
CREDIT = "_credit"                              # {field: {"_by", "_at"}}: who set a _bot field from Discord (changelog, build_site)
FIELDS = frozenset(ORDER)
GROUP = re_compile(r"# --- credited to (.+) \((\d+)\) ---")     # authorsFile's group headers (fullmatch)
RULEBOOK_GROUP = "# --- rulebook pages ---"
NEW_AUTHORS = "[people]\n\n[cards]\n"           # authorsFile's file when there is none
ABOUT = ("Card data fixes from the website's Edit button. Keyed by card file name; each field overrides the PSD. Read by "
         "the website and the card sync.")                             # edit.js ABOUT, for a new file
MOVED_NOTE = "AutoCroissant's aliases.pkl moved into the cards' aliases on {at}: a bot or build that knows this key reads aliases only here"
# /update_metadata's options -> Card Edits.json fields, in the command's order
METADATA = (("ability", "ability"), ("stars", "stars"), ("subtype", "subtype"), ("series", "series"),
            ("subseries", "subseries"), ("hp", "hp"), ("defense", "def"), ("attack", "atk"), ("speed", "spd"),
            ("card_type", "type"), ("types", "types"))
KINDS = {t.value.lower(): t.value for t in CardType}   # a card_type typed in Discord: one of these, any case
SETTABLE = {k: v for k, v in KINDS.items() if v != CardType.UNKNOWN.value}     # edit.js offers no "unknown"
MASS = {"series": "series", "subseries": "subseries", "subtype": "subtype", "card_type": "type"}   # /mass_replace -> field
BODY_LINES = 40                                 # a commit body lists at most this many cards
SHOWN = 60                                      # a value is cut to this many characters in a reply or commit body
ARCHIVE = "~/Documents/Tools/discord-archive"   # the Discord archive (discord_archive.DEFAULT_OUT; never in a repo)
SOURCES = {"discord": "discord", "git-first-upload": "git"}    # created_dates.json source -> the file's "from"
DATES_ABOUT = ("When each card was made: the earlier of its first Discord post and its first upload to this repo (UTC "
               "day); provisional: an open ruling on the Discord archive may still move it. Written by AutoCroissant "
               "(python -m commands.discord_edits created); an entry with \"from\": \"hand\" is set by hand and kept. "
               "Read by the website (the created sort, card pages) and the bot (/update_stats).")


# ========================
# Card Edits.json and Card Authors.txt (edit.js ports)
# ========================
def now() -> str:
    """The _at stamp: edit.js's toISOString() without the milliseconds."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def editor(interaction) -> str:
    """The _by and Edit-by of a Discord edit: "discord:<username>" (one word: a reminder's creator name may have spaces)."""
    user = getattr(interaction, "user", None)
    name = str(getattr(user, "name", "") or getattr(user, "id", "") or "unknown")
    return DISCORD + "_".join(name.split())


def _is_bot(by) -> bool:
    """A _by of the bot (any case), as changes.js isBot."""
    return isinstance(by, str) and by.lower() == BOT.lower()


def edits_file(text: Optional[str], ops: list, at: str, by: str) -> str:
    """
    edit.js editsFile: Card Edits.json with ops applied. A touched entry is matched like the authors file, lists its
    fields in ORDER, then its other keys (_psd kept; _bot less the fields the op sets or unsets, _credit less the fields
    no longer in _bot), then _by and _at
    (replaced); an entry left without keys goes. An entry the bot stamped lists its fields in _bot first, when another
    editor edits it.

    Args:
        text: The file ("" or None: a new one)
        ops: [{"key": card name, "set": {field: value}, "unset": [field]}]
        at: The _at stamp
        by: The _by stamp

    Returns:
        The text (dump_edits); raises ValueError when text isn't a JSON object
    """
    data = loads(text) if text else {"about": ABOUT, "cards": {}}
    if not isinstance(data, dict):
        raise ValueError(f"{EDITS_PATHS[0]} isn't a JSON object")
    if not isinstance(data.get("cards"), dict):
        data["cards"] = {}
    cards = data["cards"]
    for op in ops:
        if not op.get("set") and not op.get("unset"):
            continue
        key = next((k for k in cards if CardAuthors.key(k) == CardAuthors.key(op["key"])), op["key"])
        current = dict(cards[key]) if isinstance(cards.get(key), dict) else {}
        if _is_bot(current.get("_by")) and not _is_bot(by):     # the bot's entry: its fields stay nobody's edit
            current[BOT_FIELDS] = [f for f in ORDER if f in current]
        for name in op.get("unset") or ():
            current.pop(name, None)
        current.update(op.get("set") or {})
        current.pop("_by", None)
        current.pop("_at", None)
        if isinstance(current.get(BOT_FIELDS), list):          # a field this op sets or unsets is no longer the bot's
            touched = set(op.get("set") or ()) | set(op.get("unset") or ())
            left = [f for f in current[BOT_FIELDS] if f not in touched and f in current]
            if left:
                current[BOT_FIELDS] = left
            else:
                current.pop(BOT_FIELDS)
        if CREDIT in current:                                   # a _bot field's Discord editor goes with it (edit.js)
            bot = current.get(BOT_FIELDS) if isinstance(current.get(BOT_FIELDS), list) else []
            was = current[CREDIT] if isinstance(current[CREDIT], dict) else {}
            credit = {f: v for f, v in was.items() if f in bot}
            if credit:
                current[CREDIT] = credit
            else:
                current.pop(CREDIT)
        cards.pop(key, None)
        names = [f for f in ORDER if f in current] + [f for f in current if f not in FIELDS]
        if names:
            cards[key] = {**{f: current[f] for f in names}, "_by": by, "_at": at}
    return dump_edits(data)


def entry_of(text: str, name: str) -> tuple:
    """A card's entry in Card Edits.json, matched like the authors file: (its key or None, its dict or {})."""
    cards = (loads(text).get("cards") or {}) if text else {}
    key = next((k for k in cards if CardAuthors.key(k) == CardAuthors.key(name)), None)
    return key, (cards[key] if key is not None and isinstance(cards[key], dict) else {})


def _entries(text: Optional[str]) -> dict:
    """Card Edits.json's entries by CardAuthors.key (the first of a key, as edits_file matches)."""
    out: dict = {}
    for key, fields in ((loads(text).get("cards") or {}) if text else {}).items():
        if isinstance(fields, dict):
            out.setdefault(CardAuthors.key(key), fields)
    return out


def _stamped(text: Optional[str], ops: list, at: str, by: str) -> str:
    """
    edits_file for a Discord edit. No op: the head's text as it is (never reformatted). An entry holding PSD fields
    keeps its _by and _at when its op sets none of them: the bake credits the entry's _by, and a discord: one gets no
    Edit-by trailer, so an alias or folder edit would take the credit from the PSD fields' editor. The fields the op
    sets there are listed in its _bot (not that editor's) and credited in its _credit; a _credit field out of _bot goes,
    and an entry left with only its _credit goes (edits_file keeps it).
    """
    ops = [op for op in ops if op.get("set") or op.get("unset")]
    if not ops:
        return text
    psd = lambda fields: any(CardEdits.KIND.get(f) == "psd" for f in fields)
    touched = {CardAuthors.key(op["key"]) for op in ops}
    sets: dict = {}
    for op in ops:
        sets.setdefault(CardAuthors.key(op["key"]), set()).update(op.get("set") or ())
    data = loads(edits_file(text, ops, at, by))
    before = _entries(text)
    for key, fields in list(data["cards"].items()):
        k = CardAuthors.key(key)
        if k not in touched:
            continue
        old = before.get(k)
        rest = {f: v for f, v in fields.items() if f not in ("_by", "_at", BOT_FIELDS, CREDIT)}
        if not rest:                                            # only its _credit left: goes
            del data["cards"][key]
            continue
        bot = list(fields.get(BOT_FIELDS) or [])
        was = fields.get(CREDIT) if isinstance(fields.get(CREDIT), dict) else {}
        credit = {f: v for f, v in was.items() if f in bot}
        stamps = {s: fields[s] for s in ("_by", "_at")}
        if not psd(sets[k]) and psd(fields) and isinstance(old, dict) and old.get("_by"):
            stamps = {s: old[s] for s in ("_by", "_at") if s in old}
            if old["_by"] != by:                                # not the stamp's editor: per-field credit
                mine = [f for f in ORDER if f in sets[k] and f in fields]
                bot += [f for f in mine if f not in bot]
                credit.update({f: {"_by": by, "_at": at} for f in mine})
        if bot == (fields.get(BOT_FIELDS) or []) and credit == (fields.get(CREDIT) or {}):
            fields.update(stamps)                               # its layout kept
            continue
        data["cards"][key] = {**rest, **({BOT_FIELDS: bot} if bot else {}), **({CREDIT: credit} if credit else {}), **stamps}
    return dump_edits(data)


def _section(lines: list, name: str) -> Optional[tuple]:
    """authorsFile's section(): (first line, end) of a [name] section's body, None when absent."""
    start = next((i for i, line in enumerate(lines) if line.strip().lower() == f"[{name}]"), None)
    if start is None:
        return None
    end = start + 1
    while end < len(lines) and not fullmatch(r"\s*\[.+\]\s*", lines[end]):
        end += 1
    return start + 1, end


def _bump(lines: list, who: str, by: int) -> int:
    """authorsFile's bump(): add by to who's "credited to" count. Returns its header's index, -1 when it has none."""
    for i, line in enumerate(lines):
        found = GROUP.fullmatch(line.strip())
        if found and found.group(1) == who:
            lines[i] = f"# --- credited to {who} ({max(0, int(found.group(2)) + by)}) ---"
            return i
    return -1


def authors_file(text: Optional[str], ops: list, people: dict) -> str:
    """
    edit.js authorsFile: Card Authors.txt with [people] lines set and [cards] lines moved, as the website writes it.

    Args:
        text: The file (None: a new one)
        ops: [{"key": card name, "author": person ("" removes the card's line)}]
        people: {git name: name to show}

    Returns:
        The text
    """
    lines = (NEW_AUTHORS if text is None else text).split("\n")
    if not _section(lines, "people"):
        lines[0:0] = ["[people]", ""]
    if not _section(lines, "cards"):
        lines += ["", "[cards]"]
    for git, shown in people.items():
        start, end = _section(lines, "people")
        i = next((j for j in range(start, end) if not lines[j].strip().startswith("#")
                  and lines[j].split("=")[0].strip().lower() == git.lower()), -1)
        if i >= 0:
            lines[i] = f"{lines[i].split('=')[0].strip()} = {shown}"
        else:
            j = end
            while j > start and not lines[j - 1].strip():
                j -= 1
            lines.insert(j, f"{git} = {shown}")
    for op in ops:
        if "author" not in op:
            continue
        start, end = _section(lines, "cards")
        i = next((j for j in range(start, end) if not lines[j].strip().startswith("#") and "=" in lines[j]
                  and CardAuthors.key(lines[j].split("=")[0]) == CardAuthors.key(op["key"])), -1)
        if i >= 0:
            _bump(lines, "=".join(lines[i].split("=")[1:]).strip(), -1)
            del lines[i]
            start, end = _section(lines, "cards")
        if not op["author"]:
            continue
        line = f"{op['key']} = {op['author']}"
        group = _bump(lines, op["author"], 1)
        if group < 0:                           # a new group, before the rulebook pages
            at = next((j for j in range(start, end) if lines[j].strip() == RULEBOOK_GROUP), end)
            while at > start and not lines[at - 1].strip():
                at -= 1
            lines[at:at] = ["", f"# --- credited to {op['author']} (1) ---", line]
            continue
        j = group + 1                           # alphabetical within the group
        while (j < len(lines) and lines[j].strip() and not lines[j].strip().startswith("#")
               and _collate(lines[j]) < _collate(line)):
            j += 1
        lines.insert(j, line)
    return "\n".join(lines)


def card_line(text: Optional[str], name: str) -> Optional[str]:
    """The person a card's [cards] line credits (authorsFile's match), None when it has no line."""
    lines = (text or "").split("\n")
    found = _section(lines, "cards")
    for j in range(*found) if found else ():
        if not lines[j].strip().startswith("#") and "=" in lines[j] \
                and CardAuthors.key(lines[j].split("=")[0]) == CardAuthors.key(name):
            return "=".join(lines[j].split("=")[1:]).strip()
    return None


def _group_end(lines: list, header: int) -> int:
    """The line after a "credited to" group's last card line."""
    j = header + 1
    while j < len(lines) and lines[j].strip() and not lines[j].strip().startswith(("#", "[")):
        j += 1
    return j


def replace_author(text: Optional[str], old: str, new: str, known: bool = True) -> tuple:
    """
    /mass_replace author (design section 11: right-hand sides rewritten): every [cards] line crediting old credits new
    where it stands (a rulebook page stays among the rulebook pages), old's "credited to" group is renamed new's (merged
    into new's own group when there is one), every [people] line showing old shows new, and "old = new" joins [people]
    when old has no line there, so cards credited to the git name follow.

    Args:
        known: old is an author the bot knows; when it isn't and no line names old, the file stays as it is

    Returns:
        (the text (text itself when nothing changes), the [cards] names rewritten, the [people] lines set {git name:
        new}); new takes the spelling of a person the file already names in another case
    """
    lines = authors_file(text, [], {}).split("\n")  # the sections there, as authorsFile makes them
    fold = lambda value: " ".join(value.split()).lower()
    same = lambda value: fold(value) == fold(old)

    def entries(name: str) -> list:
        found = _section(lines, name)
        return [j for j in range(*found) if not lines[j].strip().startswith("#") and "=" in lines[j]] if found else []
    if not same(new):                           # someone already named so: their spelling (edit.js spell), one group
        start, end = _section(lines, "cards")
        names = [lines[j].split("=", 1)[1] for j in entries("cards") + entries("people")] \
            + [m.group(1) for m in (GROUP.fullmatch(lines[i].strip()) for i in range(start, end)) if m]
        new = next((" ".join(n.split()) for n in names if fold(n) == fold(new)), new)
    moved = []
    for j in entries("cards"):
        left, right = lines[j].split("=", 1)
        if same(right):
            lines[j] = f"{left.strip()} = {new}"
            moved.append(left.strip())
    people = {lines[j].split("=", 1)[0].strip(): new for j in entries("people") if same(lines[j].split("=", 1)[1])}
    if not moved and not people and not known:
        return text, [], {}
    if not any(same(lines[j].split("=", 1)[0]) for j in entries("people")):
        people[old.strip()] = new
    start, end = _section(lines, "cards")
    for i in range(start, end):                 # old's group headers, and new's in another case: new's
        found = GROUP.fullmatch(lines[i].strip())
        if found and (same(found.group(1)) or fold(found.group(1)) == fold(new)):
            lines[i] = f"# --- credited to {new} ({found.group(2)}) ---"
    while True:                                 # two groups of new's: the later one's lines join the first, in order
        start, end = _section(lines, "cards")
        headers = [i for i in range(start, end) if (GROUP.fullmatch(lines[i].strip()) or [None, None])[1] == new]
        if len(headers) < 2:
            break
        first, later = headers[0], headers[1]
        block = lines[later + 1:_group_end(lines, later)]
        cut = later - 1 if later > 0 and not lines[later - 1].strip() else later
        del lines[cut:_group_end(lines, later)]
        _bump(lines, new, len(block))
        for line in block:
            j = first + 1
            while j < _group_end(lines, first) and _collate(lines[j]) < _collate(line):
                j += 1
            lines.insert(j, line)
    return authors_file("\n".join(lines), [], people), moved, people


# ========================
# The commit
# ========================
def _short(value) -> str:
    text = ", ".join(value) if isinstance(value, list) else str(value)
    text = " ".join(text.split())
    return text if len(text) <= SHOWN else text[:SHOWN - 1] + "…"


def commit(subject, by: str, body, edits: Optional[Callable] = None, authors: Optional[Callable] = None) -> tuple:
    """
    One commit on the card repo's main through card_bake.land: the rewrites run on the files at main's head, and again
    on the new head when main moved meanwhile (up to COMMIT_TRIES times).

    Args:
        subject: The commit subject ("Discord: ..."), or a function giving it once the rewrites ran
        by: The editor: the Edit-by trailer
        body: Lines between the subject and the trailer, or a function giving them once the rewrites ran
        edits: Card Edits.json's text at the head -> the new text (None or the same text: unchanged)
        authors: Card Authors.txt's text at the head (None when absent) -> the new text (None or the same: unchanged)

    Returns:
        (commit sha or None when nothing changed, {"edits": text written, "authors": text written}); raises on a GitHub,
        git or JSON failure
    """
    import commands.card_bake as card_bake      # lazy: the edit pipeline's modules load on the first edit
    written: dict = {}

    def rewrite(name: str, change: Optional[Callable]) -> Callable:
        def run(text):
            new = change(text) if change is not None else None
            written[name] = None if new == text else new
            return written[name]
        return run

    def message(included: list) -> str:
        lines = body() if callable(body) else body
        lines = lines[:BODY_LINES] + ([f"- … {len(lines) - BODY_LINES} more"] if len(lines) > BODY_LINES else [])
        title = subject() if callable(subject) else subject
        return "\n\n".join(part for part in (title, "\n".join(lines), f"{EDIT_BY}: {by}") if part)
    share = card_bake.Share(None, {}, {}, rewrite("edits", edits),
                            authors=rewrite("authors", authors) if authors is not None else None)
    sha = card_bake.land(CARD_REPO, [share], message, False)
    return sha, (written if sha else {})


def _link(sha: str) -> str:
    return f"<https://github.com/{CARD_REPO}/commit/{sha}>"


def refresh(names, edits_text: Optional[str] = None, authors_text: Optional[str] = None) -> list:
    """
    The committed files applied in memory, as the next build applies them: the authors file and Card Edits.json on these
    cards (an edit that changed a card's data keeps its card before as a version: card_history.record_edits), then the
    duplicates copied again and renamed (card_data._sync_duplicates: an alias on a duplicate's PNG names
    it), then the query tables rebuilt and stats.pkl saved (so a restart before the next /update_stats keeps the edit;
    the next build writes the same values). After the cutover nothing is saved: the next card sync's data holds it.

    Args:
        names: Card names (None: every card)
        edits_text: Card Edits.json as committed (None: unchanged)
        authors_text: Card Authors.txt as committed (None: unchanged)

    Returns:
        The names whose card changed
    """
    from commands.card_runner import bake_lock      # lazy: the CLI (migrate) needs no Discord
    if bake_lock.locked():
        print("Discord edit: its in-memory refresh waits for the running /update_stats or bake")
    with bake_lock:                                 # never saves stats.pkl inside a build (repo_traverser holds it)
        return _refresh(names, edits_text, authors_text)


def _refresh(names, edits_text: Optional[str], authors_text: Optional[str]) -> list:
    """refresh's body, under bake_lock."""
    from commands.card_data import CardBuild
    from commands.card_history import record_edits
    from commands.psd_analyzer import stats_db
    from commands.query_card import card_repo
    edits = CardEdits.parse(edits_text) if edits_text is not None else None
    authors = CardAuthors.parse(authors_text) if authors_text is not None else None
    stats = stats_db.stats
    wanted = [n for n in (list(stats) if names is None else names) if n in stats]
    changed, edited = [], {}
    for name in wanted:
        card = stats[name]
        before = repr(card)
        if authors is not None:
            card.author = authors.resolve(name, card.author)
        if edits is not None:
            edited[name] = deepcopy(card)
            edits.apply(name, card)
        if repr(card) != before:
            changed.append(name)
    record_edits(stats_db, {n: edited[n] for n in changed if n in edited}, datetime.now(timezone.utc).timestamp())
    current = edits if edits is not None else card_repo._edits
    moved = edits.aliases_moved if edits is not None else card_repo.aliases_moved()   # no file read: aliases.pkl, as the build
    legacy = {} if moved else card_repo.git_file_alias
    pngs = {card_name(p).lower(): p for p in card_repo.git_files.values()}
    dups: dict = {}
    for name, dup in stats.items():
        original = getattr(dup, "duplicate_of", None)
        png = basename(pngs.get(name.lower(), ""))
        found = fullmatch(r"(.+?)(\d+)\.png", png, IGNORECASE)
        if original in stats and not getattr(stats[original], "duplicate_of", None) and png:
            dups[name] = (original, int(found.group(2)) if found else None, png.lower())   # None: a dup_of card's own PNG
    for name, (original, number, png) in dups.items():  # a duplicate is its original's copy (card_data._sync_duplicates)
        if original not in changed and edits is None:
            continue
        before, src = repr(stats[name]), stats[original]
        copy = deepcopy(src)
        copy.name, copy.duplicate_of, copy.parsed_values, copy.problems, copy.alt_names = name, original, None, [], None
        own = (current.cards.get(CardAuthors.key(name), {}).get("aliases") or []) if current is not None else []
        copy.display_name = (CardBuild._duplicate_name(png, src, legacy, own) if number is not None
                             else CardBuild._duplicate_name(png, src, {}, own, own_name=True))
        if current is not None:
            current.apply(name, copy)
        stats[name] = copy
        if repr(copy) != before:
            changed.append(name)
    if edits is not None:
        card_repo._edits, card_repo._edit_names = edits, (blob_sha(edits_text.encode("utf-8")), edits.names())
        card_repo._moved_hint = False                # the file parsed: it says whether aliases moved
    if not changed and edits is not None:
        card_repo._populate_renames()           # a new alias: autocomplete offers it (prep_dataframes does when changed)
    if changed:
        try:
            card_repo.prep_dataframes()
        except Exception:                       # noqa: BLE001 - the tables catch up on the next /update_stats
            print(f"Query tables not rebuilt:\n{format_exc()}")
        try:
            if not global_config.CARD_SYNC:     # the cutover: the pickles are frozen (Card sync builds the data)
                stats_db.save()
        except Exception:                       # noqa: BLE001 - in memory still; the next /update_stats saves it
            print(f"stats.pkl not saved:\n{format_exc()}")
    return changed


def _failed(interaction, what: str, error: Exception) -> None:
    from commands.utils import say
    print(f"{what} failed:\n{format_exc()}")
    say(interaction, f"{what} not committed: {type(error).__name__}: {error}")


# ========================
# /update_metadata
# ========================
def _already(card, field: str, value) -> bool:
    """The card's data already has this value (a Card Edits.json field)."""
    if card is None:
        return False
    stat = {"hp": "hp", "def": "defense", "atk": "attack", "spd": "speed"}.get(field)
    have = getattr(card.stats, stat) if stat else getattr(card, "card_type" if field == "type" else field, None)
    return have == value and type(have) is type(value)


def type_names(names: list) -> tuple:
    """
    Types as the card bake reads them: the icon vocabulary of the card repo's head (card_bake.vocabulary: Types/ and
    the stoogedings manifest's aliases).

    Returns:
        (the known ones in the vocabulary's spelling, "'fyre' (fire?)" per name that is no type icon)
    """
    import commands.card_bake as card_bake      # lazy, as commit()
    icons = card_bake.vocabulary(card_bake.snapshot_at(CARD_REPO, card_bake.ref_head(CARD_REPO))).icons
    known = sorted({name for name, kind in icons.tokens.values() if kind == "type"})
    hits = [icons.lookup(name) for name in names]
    unknown = [f"'{name}'" + "".join(f" ({close}?)" for close in get_close_matches(name, known, 1))
               for name, hit in zip(names, hits) if hit is None or hit[1] != "type"]
    return [hit[0] for hit in hits if hit is not None and hit[1] == "type"], unknown


def manual_metadata_entry(interaction,
                          query: str,
                          del_entry: bool = False,
                          author: Optional[str] = None,
                          ability: Optional[str] = None,
                          stars: Optional[int] = None,
                          subtype: Optional[str] = None,
                          series: Optional[str] = None,
                          subseries: Optional[str] = None,
                          hp: Optional[int] = None,
                          defense: Optional[int] = None,
                          attack: Optional[int] = None,
                          speed: Optional[int] = None,
                          card_type: Optional[str] = None,
                          types: Optional[str] = None) -> None:
    """
    /update_metadata: a card's fields as one commit to the card repo (author in Card Authors.txt, the rest in Card
    Edits.json), applied in memory at once. No field shows the card's data; an archived card is shown, never edited.

    Args:
        interaction: Discord interaction (or a reminder's)
        query: Card name query
        del_entry: Hide the card (hidden: true)
        author: Card author ([cards] line)
        ability, stars, types, hp, defense, attack, speed: PSD fields (pending until the bake prints them; types
            comma-separated Types/ icon names: an unknown one is refused)
        subtype, series, card_type, subseries: folder and database fields
    """
    from commands.psd_analyzer import _format_metadata, stats_db
    from commands.query_card import card_repo
    from commands.utils import say
    path = card_repo.get_card_path(query)
    if not path:
        say(interaction, f"No card found matching '{query}'.")
        return
    name = basename(path).replace("_", " ")     # path has no extension: card_name would cut at a dot in the name
    values = dict(zip((option for option, _ in METADATA),
                      (ability, stars, subtype, series, subseries, hp, defense, attack, speed, card_type, types)))
    given = {field: values[option] for option, field in METADATA if values[option] is not None}
    given = {f: v.strip() if isinstance(v, str) and f != "ability" else v for f, v in given.items()}
    if "ability" in given:                      # edit.js: \r dropped, the ends trimmed, the spaces inside kept
        given["ability"] = given["ability"].replace("\r", "").strip()
    if "types" in given:
        given["types"] = [t.strip().lower() for t in given["types"].split(",") if t.strip()]
    if "type" in given:
        given["type"] = KINDS.get(given["type"].lower(), given["type"])
    author = " ".join(author.split()) if author is not None else None
    if del_entry:
        given["hidden"] = True
    archived = stats_db.archived().get(name) if name not in stats_db.stats else None
    if archived is not None:                     # no entry: the next run would archive it over the real card
        if not given and author is None:
            say(interaction, f"**{name}** is archived (its PSD was deleted from the card repo):\n"
                             f"```{_format_metadata(archived, name)}```")
        else:
            say(interaction, f"**{name}** is archived (its PSD was deleted from the card repo); its metadata can't be edited.")
        return
    card = stats_db.stats.get(name)
    if not given and author is None:
        if card is None:
            say(interaction, f"**{name}** has a card image but no data yet (its PSD isn't parsed).")
        else:
            say(interaction, f"Current metadata for **{name}**:\n```{_format_metadata(card, name)}```")
        return
    original = getattr(card, "duplicate_of", None)
    if author is not None and original:          # the build and the site credit a duplicate with its original's author
        say(interaction, f"Not committed: **{name}** is a numbered copy of **{original}**'s card, so its author is "
                         f"{original}'s. Set the author on {original}.")
        return
    if given.get("type", "") not in ("", *SETTABLE.values()):
        say(interaction, f"Not committed: card_type must be one of {', '.join(SETTABLE.values())}.")
        return
    bad = [f for f, v in given.items() if not CardEdits.valid(f, v) or v in ("", [])]
    if bad or author == "":
        say(interaction, f"Not committed: {', '.join(bad) or 'author'} must be "
                         + ("a whole number, 0 or more, or some text." if bad else "a name."))
        return
    if "types" in given:                        # a typo would sit in the database until the bake refused it
        try:
            given["types"], unknown = type_names(given["types"])
        except Exception as error:              # noqa: BLE001 - as the commit's failures
            _failed(interaction, f"/update_metadata {name}", error)
            return
        if unknown:
            say(interaction, f"Not committed: no type icon in Types/ is named {', '.join(unknown)}.")
            return
    by, at = editor(interaction), now()
    done: dict = {"fields": {}, "author": False}

    def edit(text):                             # only the fields the head's entry doesn't already have
        entry = entry_of(text, name)[1]         # no entry and the card already so: none (edit.js same(v, pub))
        done["fields"] = {f: v for f, v in given.items() if (f in entry or not _already(card, f, v))
                          and (f not in entry or entry[f] != v or type(entry[f]) is not type(v))}
        return _stamped(text, [{"key": name, "set": done["fields"], "unset": []}], at, by)

    def credit(text):                           # only when the card's [cards] line credits someone else
        line = card_line(text, name)
        done["author"] = line != author and (line is not None or getattr(card, "author", None) != author)
        return authors_file(text, [{"key": name, "author": author}], {}) if done["author"] else text

    def fields() -> list:
        return list(done["fields"]) + (["author"] if done["author"] else [])

    def body() -> list:
        return [f"- {f}: {_short(v)}" for f, v in done["fields"].items()] + ([f"- author: {author}"] if done["author"] else [])
    try:
        sha, written = commit(lambda: f"{SUBJECT} /update_metadata {name} ({', '.join(fields())})", by, body,
                              edit if given else None, credit if author is not None else None)
    except Exception as error:                  # noqa: BLE001 - one line in Discord, the traceback on stdout
        _failed(interaction, f"/update_metadata {name}", error)
        return
    if not sha:
        say(interaction, f"**{name}**: already has those values; nothing committed.")
        return
    refresh([name], written.get("edits"), written.get("authors"))
    set_now = done["fields"]
    same = [f for f in given if f not in set_now] + (["author"] if author is not None and not done["author"] else [])
    psd = [f for f in set_now if CardEdits.KIND.get(f) == "psd"]
    shared = original or next((n for n, c in stats_db.stats.items() if getattr(c, "duplicate_of", None) == name), None)
    unbaked = ("the card bake doesn't print it (numbered PNGs share the card's PSD), so it stays a database edit; "
               "change the PSD to show it on the card image.")
    lines = [f"  • {f}: {_short(v)}" for f, v in set_now.items()] + ([f"  • author: {author}"] if done["author"] else [])
    say(interaction, f"Updated **{name}** in the card repo ({_link(sha)}):\n" + "\n".join(lines)
        + (f"\nAlready so: {', '.join(same)}." if same else "")
        + (f"\n{', '.join(psd)}: {unbaked if shared else 'pending on the card image until the next card bake prints it.'}"
           if psd else "")
        + ("" if card is not None else "\nNo data for this card yet (its PSD isn't parsed): the website shows the edit."))


# ========================
# /mass_replace
# ========================
def _cards(n: int) -> str:
    return f"{n} card" if n == 1 else f"{n} cards"


def mass_replace_field(interaction, field: str = "", old_value: str = "", new_value: str = "") -> None:
    """
    /mass_replace: one value replaced by another on every card, in one commit to the card repo. author rewrites Card
    Authors.txt (replace_author); series, subseries, subtype and card_type write one Card Edits.json entry per card whose
    value is old_value (any case; archived cards and duplicates' copies are left alone).

    Args:
        interaction: Discord interaction (or a reminder's)
        field: author, series, subseries, subtype or card_type
        old_value: The value replaced
        new_value: The new value
    """
    from commands.psd_analyzer import stats_db
    from commands.utils import say
    if field != "author" and field not in MASS:
        say(interaction, f"Invalid field '{field}'. Valid fields are: author, {', '.join(MASS)}")
        return
    old, new = " ".join(old_value.split()), " ".join(new_value.split())
    if not old or not new:
        say(interaction, "Not committed: old_value and new_value can't be empty.")
        return
    if field == "card_type":
        if new.lower() not in SETTABLE:
            say(interaction, f"Not committed: card_type must be one of {', '.join(SETTABLE.values())}.")
            return
        old, new = KINDS.get(old.lower(), old), SETTABLE[new.lower()]
    if old == new:
        say(interaction, f"Not committed: old_value and new_value are both '{old}'.")
        return
    by, at = editor(interaction), now()
    subject = f"{SUBJECT} /mass_replace {field} '{_short(old)}' -> '{_short(new)}'"
    seen: dict = {}
    try:
        if field == "author":
            known = any((c.author or "").strip().lower() == old.lower() for c in stats_db.stats.values())

            def rewrite(text):
                seen["text"], seen["moved"], seen["people"] = replace_author(text, old, new, known)
                return seen["text"]

            def spelled() -> str:              # new as written: an existing person's spelling (replace_author)
                return next(iter(seen.get("people", {}).values()), new)
            sha, written = commit(lambda: f"{SUBJECT} /mass_replace {field} '{_short(old)}' -> '{_short(spelled())}'", by,
                                  lambda: [f"- {n}" for n in seen["moved"]]
                                  + [f"- [people] {g} = {v}" for g, v in seen["people"].items()], authors=rewrite)
            names = None
        else:
            target = MASS[field]
            found = {n: getattr(c, field).strip() for n, c in stats_db.stats.items() if not getattr(c, "duplicate_of", None)
                     and isinstance(getattr(c, field, None), str) and getattr(c, field).strip().lower() == old.lower()}
            if not found:
                say(interaction, f"No card has {field} '{old}': nothing to commit.")
                return
            names = sorted(n for n, value in found.items() if value != new)     # one already so: no override
            if not names:
                say(interaction, f"Nothing to replace: every card with {field} '{old}' already has '{new}'.")
                return

            def rewrite(text):                  # a card whose entry already says new: no op
                entries = _entries(text)
                seen["names"] = [n for n in names if entries.get(CardAuthors.key(n), {}).get(target) != new]
                return _stamped(text, [{"key": n, "set": {target: new}, "unset": []} for n in seen["names"]], at, by)
            sha, written = commit(lambda: f"{subject} ({_cards(len(seen['names']))})", by,
                                  lambda: [f"- {n}" for n in seen["names"]], rewrite)
    except Exception as error:                  # noqa: BLE001
        _failed(interaction, f"/mass_replace {field}", error)
        return
    if not sha:
        if field != "author":
            say(interaction, f"Nothing to replace: every card with {field} '{old}' already has '{new}' in the card repo.")
        elif seen.get("moved") or seen.get("people"):
            say(interaction, f"Card Authors.txt already credits '{spelled()}' wherever it credited '{old}': nothing to commit.")
        else:
            say(interaction, f"No card has author '{old}': nothing to commit.")
        return
    changed = refresh(names, written.get("edits"), written.get("authors"))
    if field == "author":
        new = spelled()
        detail = (f"{len(seen['moved'])} [cards] lines now credit {new}; [people] "
                  + ", ".join(f"{git} = {shown}" for git, shown in seen["people"].items()))
    else:
        detail = f"{_cards(len(seen['names']))} set in Card Edits.json"
    say(interaction, f"Replaced {field}='{old}' with '{new}' ({_link(sha)}): {detail}; {_cards(len(changed))} changed now.")


# ========================
# /alias
# ========================
def alias_key(alias: str) -> str:
    """How a lookup name is registered (query_card._add_lookup): lowercase, spaces as underscores."""
    return alias.strip().lower().replace(" ", "_")


def _legacy(card_repo) -> dict:
    """aliases.pkl's aliases while they still count (Card Edits.json without aliases_moved), keyed like alias_key."""
    return {} if card_repo.aliases_moved() else {alias_key(a): t for a, t in card_repo.git_file_alias.items()}


def _png_path(card_repo, key: str) -> Optional[str]:
    """A registry key's PNG path (key.png, else a folder-prefixed key ending so)."""
    target = f"{key}.png"
    return card_repo.git_files.get(target) or next((v for k, v in card_repo.git_files.items() if k.endswith("/" + target)), None)


def edit_aliases(card_repo) -> list:
    """Card Edits.json's aliases as the bot last read them: [(alias, card name)], sorted."""
    edits = card_repo._edits
    out = []
    for key, fields in (edits.cards.items() if edits is not None else ()):
        path = _png_path(card_repo, key.replace(" ", "_"))
        out += [(alias, card_name(path) if path else key) for alias in fields.get("aliases") or []]
    return sorted(out, key=lambda pair: (alias_key(pair[0]), pair[1]))


def _alias_ops(text: str, alias: str, card: Optional[str]) -> list:
    """
    Card Edits.json ops putting alias on card's entry alone (card None: on no entry). [] when already so: the card
    holding the alias in any spelling (alias_key, as lookups match) keeps its list as it is.
    """
    cards = (loads(text) if text else {}).get("cards") or {}
    ops = []
    norm = alias_key(alias)
    for key, fields in cards.items():
        if not isinstance(fields, dict):
            continue
        current = list(fields.get("aliases") or [])
        kept = [a for a in current if alias_key(a) != norm]
        if card is not None and CardAuthors.key(key) == CardAuthors.key(card):
            kept = current if len(kept) < len(current) else current + [alias]
        if kept != current:
            ops.append({"key": key, "set": {"aliases": kept} if kept else {}, "unset": [] if kept else ["aliases"]})
    if card is not None and not any(CardAuthors.key(k) == CardAuthors.key(card) for k in cards):
        ops.append({"key": card, "set": {"aliases": [alias]}, "unset": []})
    return ops


def manage_alias(interaction, key: Optional[str] = None, val: Optional[str] = None) -> None:
    """
    /alias: no alias lists them; an alias alone removes it; an alias and a card puts it on the card the PNG belongs to
    (taking it off any other card). Each change is one commit to Card Edits.json's `aliases`, registered at once.

    Args:
        interaction: Discord interaction (or a reminder's)
        key: The alias
        val: The card (a PNG file name, or the end of one)
    """
    from commands.query_card import card_repo
    from commands.utils import say
    legacy = _legacy(card_repo)
    if not key or not key.strip():
        pairs = edit_aliases(card_repo)
        text = "\n".join(f"{a:20s} -> {c}" for a, c in pairs)
        if legacy:
            text += ("\n\n" if text else "") + "aliases.pkl (until the alias migration):\n" \
                    + "\n".join(f"{a:20s} -> {t}" for a, t in sorted(legacy.items()))
        say(interaction, f"```\n{text}\n```" if text else "```No aliases defined.```")
        return
    alias = " ".join(key.split())
    norm = alias_key(alias)
    if norm in legacy:
        say(interaction, f"'{alias}' is in aliases.pkl (-> {legacy[norm]}): it moves to Card Edits.json with the alias "
                         "migration; change it after that.")
        return
    by, at = editor(interaction), now()
    if not val or not val.strip():
        holders = [c for a, c in edit_aliases(card_repo) if alias_key(a) == norm]
        try:
            sha, written = commit(f"{SUBJECT} /alias {alias} removed" + (f" ({holders[0]})" if holders else ""), by, [],
                                  lambda text: _stamped(text, _alias_ops(text, alias, None), at, by))
        except Exception as error:              # noqa: BLE001
            _failed(interaction, f"/alias {alias}", error)
            return
        if not sha:
            say(interaction, f"No alias exists for '{alias}'.")
            return
        path = card_repo.git_files.get(f"{norm}.png")
        if path and basename(path).lower() != f"{norm}.png":    # an alias's entry, never a card's own file name
            card_repo.git_files.pop(f"{norm}.png", None)
        refresh([], written.get("edits"))
        card_repo.relink()                      # a name or printed title with that key finds its card again
        say(interaction, f"Deleted alias: {alias} ({_link(sha)})")
        return
    wanted = val.strip().lower().replace(" ", "_")
    wanted = wanted if wanted.endswith(".png") else f"{wanted}.png"
    target = card_repo.git_files.get(wanted) or next((p for f, p in card_repo.git_files.items() if f.lower().endswith(wanted)), None)
    if not target:
        say(interaction, f"No card found matching '{val}'. Alias not created.")
        return
    card = card_name(target)
    own = card_repo.git_files.get(f"{norm}.png")
    if own and basename(own).lower() == f"{norm}.png" and own != target:
        say(interaction, f"'{alias}' is already the name of {card_name(own)}'s card image: alias not created.")
        return
    named: list = []

    def add(text):                              # a reload adds Card Edits names first come: that card would win the key
        paths = [_png_path(card_repo, k.replace(" ", "_")) for k, f in _entries(text).items()
                 if k != CardAuthors.key(card) and isinstance(f.get("name"), str) and alias_key(f["name"]) == norm]
        named[:] = [card_name(p) for p in paths if p]
        return text if named else _stamped(text, _alias_ops(text, alias, card), at, by)
    try:
        sha, written = commit(f"{SUBJECT} /alias {alias} -> {card}", by, [], add)
    except Exception as error:                  # noqa: BLE001
        _failed(interaction, f"/alias {alias}", error)
        return
    if named:
        say(interaction, f"'{alias}' is the Card Edits.json name of another card ({', '.join(named)}): alias not created.")
        return
    if not sha:
        say(interaction, f"'{alias}' already finds {card}.")
        return
    card_repo.git_files[f"{norm}.png"] = target
    refresh([], written.get("edits"))
    say(interaction, f"Created alias: {alias} -> {card} ({_link(sha)})")


# ========================
# The migration (section 14 step 5): aliases.pkl and the Discord-only values, one commit
# ========================
def _bot_edits(text: str, ops: list, at: str) -> dict:
    """
    ops applied with _by AutoCroissant; an entry that was there keeps its editor's stamps (their bake credit), and
    the fields the ops add to another editor's entry are listed in its _bot (BOT_FIELDS): not that editor's edit.
    """
    data = loads(edits_file(text, ops, at, BOT))
    before = _entries(text)
    cards = data["cards"]
    for key, fields in list(cards.items()):
        old = before.get(CardAuthors.key(key))
        if not (isinstance(old, dict) and old.get("_by")):
            continue
        bot = []
        if not _is_bot(old["_by"]):                                 # the bot's fields before, then the ones it adds
            was = old.get(BOT_FIELDS) if isinstance(old.get(BOT_FIELDS), list) else []
            bot = [f for f in was if f in fields] + [f for f in fields if f in FIELDS and f not in old and f not in was]
        stamps = {k: old[k] for k in ("_by", "_at") if k in old}
        if bot == (fields.get(BOT_FIELDS) or []):                  # an entry left as it was keeps its layout
            fields.update(stamps)
        else:
            rest = {k: v for k, v in fields.items() if k not in ("_by", "_at", BOT_FIELDS)}
            cards[key] = {**rest, **({BOT_FIELDS: bot} if bot else {}), **stamps}
    return data


def alias_ops(aliases: dict, text: str, pngs: list) -> tuple:
    """
    aliases.pkl as Card Edits.json ops: each alias on the entry of the card its PNG belongs to, resolved as the bot does
    (the PNG's file name, else the first whose name ends so, in tree order), with spaces for underscores. An alias whose
    PNG is gone is left out; one already on its card (any spelling) is not added twice; one that is its card's own name
    stays (a duplicate is named by it).

    Returns:
        (ops, report lines)
    """
    names: dict = {}
    for path in pngs:
        names.setdefault(basename(path).lower(), path)
    wanted: dict = {}
    lines, left_out = [], 0
    for alias, target in sorted(aliases.items()):
        target = str(target).lower()
        path = names.get(target) or next((p for f, p in names.items() if f.endswith(target)), None)
        spelled = " ".join(str(alias).replace("_", " ").split())
        if path is None:
            lines.append(f"- {alias} -> {target}: no such PNG, left out")
            left_out += 1
        else:
            wanted.setdefault(card_name(path), []).append(spelled)
            lines.append(f"- {alias} -> {card_name(path)} ({path})")
    ops, added = [], 0
    entries = _entries(text)
    for card, spelled in wanted.items():
        current = list(entries.get(CardAuthors.key(card), {}).get("aliases") or [])
        merged = current + [a for i, a in enumerate(spelled) if alias_key(a) not in {alias_key(c) for c in current + spelled[:i]}]
        added += len(merged) - len(current)
        if merged != current:
            ops.append({"key": card, "set": {"aliases": merged}, "unset": []})
    found = sum(len(v) for v in wanted.values())
    head = (f"{len(aliases)} aliases: {added} moved onto {len(ops)} cards, {found - added} already on their card "
            f"(the same alias, any spelling), {left_out} left out")
    return ops, [head] + lines


def migrate_aliases(aliases: dict, text: str, pngs: list, at: str) -> tuple:
    """
    aliases.pkl alone into Card Edits.json (alias_ops), plus ALIASES_MOVED. New entries get _by AutoCroissant; an entry
    already there keeps its _by and _at (the fields added to it in _bot). The migration commit is migrate()'s: this,
    with the carried values.

    Returns:
        (the new text, report lines)
    """
    ops, report = alias_ops(aliases, text, pngs)
    data = _bot_edits(text, ops, at)
    data[ALIASES_MOVED] = MOVED_NOTE.format(at=at[:10])
    return dump_edits(data), report


def carried(stats: dict, text: str, authors_text: Optional[str], first_authors: dict) -> tuple:
    """
    The Discord-only values (section 14 step 5's parity re-parse): what a re-parse keeps from the database (author,
    series, subseries: psd_parser's "preserve") that no card-repo input gives: not the card's folder, nor Card
    Edits.json, nor Card Authors.txt with the first commit's author. /update_metadata and /mass_replace set them before
    they committed to the card repo; a build from the card repo alone would lose them. A field Card Edits.json already
    sets is that value (it wins), so it isn't carried.

    Args:
        stats: stats.pkl
        text: Card Edits.json
        authors_text: Card Authors.txt (None: none)
        first_authors: PSD path -> the author of its first commit (card_history.GitIndex.first)

    Returns:
        (Card Edits.json ops, Card Authors.txt ops, report lines)
    """
    from commands.psd_parser import CardClassifier   # lazy: psd-tools loads only for the migration
    entries = _entries(text)
    authors = CardAuthors.parse(authors_text or "")
    ops, credits, lines = [], [], []
    for name, card in sorted(stats.items()):
        if getattr(card, "duplicate_of", None):     # its original's copy
            continue
        parsed = getattr(card, "parsed_values", None) or {}
        kept = lambda attr: parsed[attr] if attr in parsed else getattr(card, attr, None)
        entry = entries.get(CardAuthors.key(name), {})
        try:
            folder = CardClassifier.classify(card.path or "")
        except (ValueError, IndexError):        # a folder the parser can't read: none given
            folder = {}
        fresh = {"series": folder.get("series"), "subseries": None}
        values = {attr: kept(attr) for attr in ("series", "subseries")
                  if kept(attr) and kept(attr) != fresh[attr] and attr not in entry and CardEdits.valid(attr, kept(attr))}
        if values:
            ops.append({"key": name, "set": values, "unset": []})
            lines += [f"- {name}: {attr} {value!r} (the folder gives {fresh[attr]!r})" for attr, value in values.items()]
        author, given = kept("author"), authors.resolve(name, first_authors.get(card.path))
        if author and authors.resolve(name, author) != given:      # both through [cards] and [people], as a build
            credits.append({"key": name, "author": " ".join(str(author).split())})
            lines.append(f"- {name}: author {author!r} (the card repo gives {given!r})")
    head = (f"{sum(len(op['set']) for op in ops)} Discord-only values on {len(ops)} cards into Card Edits.json, "
            f"{len(credits)} authors into Card Authors.txt")
    return ops, credits, [head] + lines


def migrate(aliases: dict, stats: dict, text: str, authors_text: Optional[str], pngs: list, first_authors: dict,
            at: str) -> tuple:
    """
    The migration commit's files: aliases.pkl's aliases (alias_ops) and the Discord-only values (carried) in Card
    Edits.json with ALIASES_MOVED, and the carried authors as [cards] lines. New entries get _by AutoCroissant; an
    entry already there keeps its _by and _at (the fields added to it in _bot).

    Returns:
        (Card Edits.json, Card Authors.txt (None when unchanged), report lines)
    """
    value_ops, credits, values_report = carried(stats, text, authors_text, first_authors)
    moved, aliases_report = alias_ops(aliases, text, pngs)
    data = _bot_edits(text, value_ops + moved, at)
    data[ALIASES_MOVED] = MOVED_NOTE.format(at=at[:10])
    new_authors = authors_file(authors_text, credits, {}) if credits else None
    return dump_edits(data), new_authors, aliases_report + values_report


# ========================
# Creation dates (site/Card Creation Dates.json)
# ========================
def dates_file(created: dict, text: Optional[str]) -> tuple:
    """
    site/Card Creation Dates.json from the archive's created_dates.json: each dated card's day, from and provisional.
    The file's "from": "hand" entries stay as they are; nothing else of the archive leaves it.

    Args:
        created: created_dates.json (card key -> {date, source, provisional, ...})
        text: The file now ("" or None: none)

    Returns:
        (the text (dump_edits), report lines)
    """
    old = loads(text) if text else {}
    before = old.get("cards") if isinstance(old, dict) and isinstance(old.get("cards"), dict) else {}
    hand = {CardAuthors.key(k): (k, v) for k, v in before.items() if isinstance(v, dict) and v.get("from") == "hand"}
    cards, skipped = {}, []
    for name, entry in created.items():
        if CardAuthors.key(name) in hand:
            continue
        value = {"date": str(entry.get("date") or "")[:10], "from": SOURCES.get(entry.get("source")),
                 **({"provisional": True} if entry.get("provisional") else {})} if isinstance(entry, dict) else None
        if CardDates.valid(value):
            cards[name] = value
        elif not isinstance(entry, dict) or entry.get("date"):
            skipped.append(name)                # a date it can't read; an undated card is just left out
    cards.update(hand.values())
    count = lambda f: sum(1 for v in cards.values() if v["from"] == f)
    lines = [f"{len(cards)} creation dates: {count('discord')} first Discord post, {count('git')} first git upload, "
             f"{count('hand')} by hand; {sum(1 for v in cards.values() if v.get('provisional'))} provisional"]
    if skipped:
        lines.append(f"{len(skipped)} unreadable, left out: {', '.join(sorted(skipped)[:10])}")
    old_by = {CardAuthors.key(k): v for k, v in before.items()}
    new_by = {CardAuthors.key(k): k for k in cards}
    added = sorted(k for key, k in new_by.items() if key not in old_by)
    gone = sorted(k for k in before if CardAuthors.key(k) not in new_by)
    moved = []
    for key, k in sorted(new_by.items()):
        was, now = old_by.get(key), cards[k]
        if key in old_by and was != now:
            was_day = was.get("date") if isinstance(was, dict) else was
            moved.append(f"{k}: {was_day} -> {now['date']}" if was_day != now["date"] else
                         f"{k}: {now['date']} {'provisional' if now.get('provisional') else 'no longer provisional'}")
    if before:
        lines.append(f"vs the file: {len(added)} new, {len(moved)} changed, {len(gone)} gone")
        lines += [f"- {m}" for m in moved[:BODY_LINES]] + [f"- gone: {k}" for k in gone[:BODY_LINES]]
    elif added:
        lines.append(f"a new file: {len(added)} cards")
    return dump_edits({"about": DATES_ABOUT, "cards": cards}), lines


def _created(options) -> int:
    """The created command: the archive's dates into <repo>/site/Card Creation Dates.json (or --out)."""
    from os.path import dirname, expanduser, getmtime, isdir, isfile, join
    archive = expanduser(options.archive)
    source, matches = join(archive, "created_dates.json"), join(archive, "matches.json")
    if not isfile(source):
        print(f"No {source}: run review_pages.py in the Discord archive first.")
        return 1
    if isfile(matches) and getmtime(matches) > getmtime(source):
        print(f"{source} is older than matches.json (rulings since?): run `python3 -I review_pages.py` in "
              f"{archive} first, then this again.")
        return 1
    with open(source, encoding="utf-8") as fh:
        created = loads(fh.read())
    out = expanduser(options.out) if options.out else join(expanduser(options.repo), CREATED_PATHS[0])
    if not isdir(dirname(out) or "."):
        print(f"No {dirname(out)}: give --repo a card repo checkout, or --out a file.")
        return 1
    text = None
    if isfile(out):
        with open(out, encoding="utf-8") as fh:
            text = fh.read()
    try:
        new, report = dates_file(created if isinstance(created, dict) else {}, text)
    except ValueError as e:
        print(f"{out} isn't valid JSON ({e}): fix or delete it, then run this again.")
        return 1
    if new == text:
        report.append(f"Unchanged: {out}")
    else:
        _write(out, new)
        report.append(f"Wrote {out}: review the diff, then commit it (the website and /update_stats read it).")
    print("\n".join(report))
    return 0


def _git(clone: str, *args: str) -> str:
    return run_process(["git", "-C", clone, *args], capture_output=True, check=True, text=True, encoding="utf-8").stdout


def _write(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(text)


def main(args: Optional[list] = None) -> int:
    """The command line: migrate (the migration commit's files) and created (the creation dates); commits nothing."""
    parser = ArgumentParser(prog="python -m commands.discord_edits", description="Discord's card edits (the bot's side).")
    sub = parser.add_subparsers(dest="command", required=True)
    step = sub.add_parser("migrate", help="aliases.pkl and the Discord-only values into the card repo's files "
                                          "(written to --out-dir for one commit, never committed)")
    step.add_argument("--clone", required=True, help="a full (not shallow) clone of the card repo: its tree, history and "
                                                     "files at --ref")
    step.add_argument("--ref", default="HEAD", help="the card repo commit to read (default HEAD)")
    step.add_argument("--aliases", default="aliases.pkl", help="aliases.pkl (default ./aliases.pkl)")
    step.add_argument("--stats", default="stats.pkl", help="stats.pkl the parity re-parse matched (default ./stats.pkl)")
    step.add_argument("--out-dir", required=True, help="where Card Edits.json (and Card Authors.txt when it changes) go")
    dates = sub.add_parser("created", help=f"the Discord archive's creation dates into the card repo's {CREATED_PATHS[0]} "
                                           "(written in place, never committed)")
    dates.add_argument("--archive", default=ARCHIVE, help="the Discord archive holding created_dates.json (default %(default)s)")
    dates.add_argument("--repo", default=LOCAL_DIR_LOC, help="the card repo checkout to write (default %(default)s)")
    dates.add_argument("--out", help=f"write this file instead of <repo>/{CREATED_PATHS[0]}")
    options = parser.parse_args(args)
    if options.command == "created":
        return _created(options)
    from commands.card_db import load_pickle   # the allow-list unpickler
    from commands.card_history import GitIndex
    from os import makedirs
    from os.path import join
    aliases, stats = load_pickle(options.aliases), load_pickle(options.stats)
    tree = [p for p in _git(options.clone, "ls-tree", "-r", "-z", "--name-only", options.ref).split("\0")
            if p.lower().endswith(".png")]
    authors_path = next((p for p in AUTHORS_PATHS if _git(options.clone, "ls-tree", "--name-only", options.ref, "--", p).strip()),
                        None)
    authors_text = _git(options.clone, "show", f"{options.ref}:{authors_path}") if authors_path else None
    first = {path: commit.author for path, commit in GitIndex.from_git(options.clone, options.ref).first.items()}
    text, authors, report = migrate(aliases, stats, _git(options.clone, "show", f"{options.ref}:{EDITS_PATHS[0]}"),
                                    authors_text, tree, first, now())
    makedirs(options.out_dir, exist_ok=True)
    _write(join(options.out_dir, basename(EDITS_PATHS[0])), text)
    if authors is not None:
        _write(join(options.out_dir, basename(authors_path or AUTHORS_PATHS[0])), authors)
    from commands.git_commit import AUTHOR
    print("\n".join(report + [f"Commit them as the bot (build_site counts its commits for nobody): "
                              f"git commit --author '{AUTHOR['name']} <{AUTHOR['email']}>'"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# ========================
# Module Exports
# ========================
__all__ = [
    'alias_key',
    'authors_file',
    'commit',
    'edit_aliases',
    'editor',
    'edits_file',
    'entry_of',
    'main',
    'manage_alias',
    'manual_metadata_entry',
    'mass_replace_field',
    'alias_ops',
    'card_line',
    'carried',
    'dates_file',
    'migrate',
    'migrate_aliases',
    'refresh',
    'replace_author',
    'type_names',
]
