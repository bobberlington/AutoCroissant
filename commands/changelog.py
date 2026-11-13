"""
The card changelog, without Discord, network or psd-tools: per-version field changes (stats, ability, timing tags,
name, types, stars and the other data fields) with a buff/nerf verdict, the build's events from the database before
and after (new, changed, art, reparse, moved, renamed, archived, restored, edited, authors; restore for a rollback),
the push that brought each (the activity list, else consecutive commits by one author), their numbering, and one
Discord message per push within Discord's limits. Design: docs/Campaigns/DATA_BUILD_ACTION.md section 12.1. The bake's
events (baked, bake-refused) ride their push's post; the guards' (guard-png, guard-fix) and every event of a guards
commit's push are the Changes page's alone (owner, 2026-10-25): docs/Finished/session-2026-10-26/CARD_SYNC_VERDICTS.md.
Re-reads too (owner, 2026-11-11).

CLI: python -m commands.changelog post --db <db dir> --site <url> --out <file>
     |  advance --db <db dir> --seq <n> --push <sha>  (after Discord accepted that message)
"""
from argparse import ArgumentParser
from datetime import datetime, timezone
from difflib import SequenceMatcher
from itertools import groupby
from glob import glob
from json import dumps, loads
from os import makedirs
from os.path import isfile, join
from re import compile as re_compile
from typing import Optional
from urllib.parse import quote

from global_config import SITE_URL, TIMEZONE
from commands.card_db import write_text
from commands.card_edits import art_list, as_parsed
from commands.card_history import same_series, unknown
from commands.card_models import CardInfo, picture_only

# ========================
# Configuration
# ========================
STATS = (("hp", "hp", "HP"), ("def", "defense", "DEF"), ("atk", "attack", "ATK"), ("spd", "speed", "SPD"))
FIELDS = ("name", "card_type", "stars", "subtype", "series", "subseries", "types", "hp", "def", "atk", "spd", "creatures",
          "ability", "tags", "author", "note", "art", "hidden")
# timing tags in ability text: the cooldown and uses tags (2026-10-21) and the older per-turn ones
TAG_PATTERN = re_compile(r"\((?:-?\d+/\d+|\d+|X)?(?:GCD|CD|U|PGT|PG|PT)\)|\((?:FT)?E(?:\d+|X)G?T\)"
                         r"|\((?:OYT|AAT|BOT|ENDT|BOGT|ENDGT|OFS|OS|OD|WOF)\)")
COOLDOWN = re_compile(r"\((\d+)(G?CD)\)")              # one number: shorter is stronger
USES = re_compile(r"\((\d+)U\)")                         # more is stronger
KINDS = ("new", "changed", "art", "reparse", "moved", "renamed", "archived", "restored", "edited", "authors", "restore",
         "baked", "bake-refused", "guard-png", "guard-fix")
SILENT = ("reparse", "guard-png", "guard-fix")           # never in a Discord post: the Changes page (and the commit) say them
PLAIN = re_compile(r"\boverflow: |: bad [\d.]+%, mean [\d.]+, max \d+| \([+-]?\d+ px(?:, '(?:[^'\\]|\\.)*'|, \"[^\"]*\")?\)")   # Discord: no codes, no pixel counts (nor the layer they name)
KIND_LABEL = {"creature": "Creature", "item": "Item", "field": "Field card", "minion": "Minion", "nme": "N.M.E. item",
              "aux item": "Auxiliary item", "debuff": "Debuff", "buff": "Buff", "creatures": "Auxiliary creature",
              "unknown": "Other"}                       # site/assets/app.js KIND_LABEL
SECTIONS = ((("new",), "New cards"), (("changed",), "Changed"), (("art",), "New art"), (("moved", "renamed"), "Moved and renamed"),
            (("archived",), "Archived"), (("restored",), "Restored"), (("edited", "authors"), "Database edits"),
            (("baked",), "Baked into the PSDs"), (("bake-refused",), "Bake refused"),
            (("restore",), "Rollback"))
COUNT_WORDS = (("new", "new"), ("changed", "changed"), ("art", "new art"), ("moved", "moved"), ("renamed", "renamed"),
               ("archived", "archived"), ("restored", "restored"), ("edited", "edited"), ("authors", "edited"),
               ("restore", "data rollback"), ("baked", "baked"), ("bake-refused", "bake refused"))   # title: "1 new, 2 changed"
CARD_SYNC = "Card sync"                                  # the push of the run's own commit M
GUARDS = "Card guards"                                   # the push of a guards commit (card_guards.SUBJECT): never posted
SWEEP_TRAILER = "Card-sweep"                             # a trailer: a hand-committed sweep of the guards' fixes
SWEPT = ("art", "changed")                               # a sweep commit's own events of these kinds are silent ("sweep")
MINOR = "minor"                                          # an event's key: "picture" (an art change under png_changes.VISIBLE)
                                                         # or "text" (formatting only): no post, folded on the Changes page
GITHUB = "https://github.com/"
MESSAGES_MAX = 20                                        # per run; the rest wait for the next one (the cursor)
EMBEDS_MAX = 10                                          # Discord: embeds per message
DESCRIPTION_MAX = 4096                                   # characters per embed description
MESSAGE_MAX = 6000                                       # characters per message, all embeds together
TAIL_MAX = 300                                           # kept for the closing line
CHANGELOG_DIR = "changelog"                              # db/changelog/YYYY-MM.jsonl
POSTED_JSON = "posted.json"                              # db/changelog/posted.json: {"seq", "push"} of the last post
BOT = "AutoCroissant"                                    # the bot's _by (build_site.BOT_AUTHOR): no website edit
BOT_FIELDS = "_bot"                                      # an entry's fields that aren't its _by's (build_site.BOT_FIELDS)
EDIT_FIELD = {"card_type": "type", "tags": "ability"}    # a changed field -> its Card Edits.json field
UNKNOWN_FIELD = {"display_name": "name"}                 # card_history.unknown's field -> the changelog's
OWN_FIELDS = ("art",)                                    # a duplicate's own fields (the rest is its original's)
FIELD_WORD = {"name": "title", "art": "artwork sources", "ability_art": "ability artwork source"}   # changes.js FIELD (ability_art: October's lines)
TEXT_SHORT = 60                                          # an ability side this short shows whole; longer ones are cut
TEXT_HUNKS = 2                                           # changed spots shown per ability; the rest "(+N more)"


# ========================
# Field changes
# ========================
def tags_of(ability: Optional[str]) -> list:
    """The timing tags in an ability, in order (repeats kept)."""
    return TAG_PATTERN.findall(ability or "")


def _stat(value) -> Optional[int]:
    return value if isinstance(value, int) and value >= 0 else None


def fields_of(card: CardInfo) -> dict:
    """A card's changelog fields: the shown name, the folder fields, the four stats (None when missing), the auxiliary
    creatures' stats, the ability, its timing tags, author, note, artwork sources and hidden."""
    stats = getattr(card, "stats", None)
    out = {"name": getattr(card, "display_name", None) or card.name, "card_type": card.card_type,
           "stars": getattr(card, "stars", None), "subtype": getattr(card, "subtype", None),
           "series": getattr(card, "series", None), "subseries": getattr(card, "subseries", None),
           "types": list(getattr(card, "types", None) or [])}
    for key, attr, _ in STATS:
        out[key] = _stat(getattr(stats, attr, None)) if stats is not None else None
    creatures = getattr(card, "creatures", None)
    out["creatures"] = [[_stat(getattr(c, attr, None)) for _, attr, _ in STATS] for c in creatures] if creatures else None
    out["ability"] = getattr(card, "ability", None) or None
    out["tags"] = tags_of(out["ability"])
    out["author"], out["note"], out["hidden"] = getattr(card, "author", None), getattr(card, "note", None), bool(getattr(card, "hidden", False))
    out["art"] = art_list(getattr(card, "art", None)) or None   # older records: a string
    return out


def field_changes(old: CardInfo, new: CardInfo) -> dict:
    """{field: [old, new]} for each changelog field that differs (FIELDS order); not a field either reading could not
    know (card_history.unknown: the name and types before parser 2, a field its parse failed on, misread stats: then
    no stat at all), nor a series the 2026-10-20 split moved under a parent (same_series)."""
    a, b = fields_of(old), fields_of(new)
    skip = {UNKNOWN_FIELD.get(key, key) for key in unknown(old) | unknown(new)}
    skip |= {"tags"} if "ability" in skip else set()
    skip |= {key for key, _, _ in STATS} | {"creatures"} if "stats" in skip else set()
    if same_series(a, b):
        skip |= {"series", "subseries"}
    return {key: [a[key], b[key]] for key in FIELDS if a[key] != b[key] and key not in skip}


def split_changes(old: CardInfo, new: CardInfo, changes: dict) -> tuple:
    """
    A re-read card's changes as (the PSD's, Card Edits.json's or the authors file's): a field is the PSD's when its
    parse changed (as_parsed: the edits undone) and no edit sets it now; the author never is (git and Card Authors.txt
    give it).
    """
    psd = field_changes(as_parsed(old), as_parsed(new))
    edited = {key for key in (getattr(new, "parsed_values", None) or {}) if key != "problems"}
    mine = {key: value for key, value in changes.items()
            if key in psd and key != "author" and EDIT_FIELD.get(key, key) not in edited}
    return (mine, {key: value for key, value in changes.items() if key not in mine})


def _tag_steps(old: list, new: list) -> tuple:
    """(buffs, nerfs) from timing tags: a cooldown (nCD, nGCD) shorter, or uses (nU) more, is a buff; the other way a
    nerf. Only a family with one numbered tag on each side counts; an added or removed tag is not judged."""
    buffs, nerfs = [], []
    for family in ("CD", "GCD"):
        a = [int(n) for n, f in COOLDOWN.findall(" ".join(old)) if f == family]
        b = [int(n) for n, f in COOLDOWN.findall(" ".join(new)) if f == family]
        if len(a) == len(b) == 1 and a != b:
            (buffs if b[0] < a[0] else nerfs).append(f"({a[0]}{family}) -> ({b[0]}{family})")
    a, b = [int(n) for n in USES.findall(" ".join(old))], [int(n) for n in USES.findall(" ".join(new))]
    if len(a) == len(b) == 1 and a != b:
        (buffs if b[0] > a[0] else nerfs).append(f"({a[0]}U) -> ({b[0]}U)")
    return (buffs, nerfs)


def classify(changes: dict) -> dict:
    """
    The verdict on a version's changes: each stat up (HP, DEF, ATK, SPD, also an auxiliary creature's) and each
    cooldown shorter or uses more is a buff, the other way a nerf. Stars, ability wording, name and types are listed
    but not judged (a star change moves a card's cost or tier, not plainly its strength).

    Returns:
        {"verdict": "buff" | "nerf" | "mixed" | None, "buffs": [...], "nerfs": [...]}
    """
    buffs, nerfs = [], []
    for key, _, label in STATS:
        if key in changes:
            a, b = changes[key]
            if a is not None and b is not None:
                (buffs if b > a else nerfs).append(f"{label} {a} -> {b}")
    if "creatures" in changes:
        a, b = changes["creatures"]
        if a and b and len(a) == len(b):
            for i, (x, y) in enumerate(zip(a, b)):
                for (_, _, label), p, q in zip(STATS, x, y):
                    if p is not None and q is not None and p != q:
                        (buffs if q > p else nerfs).append(f"creature {i + 1} {label} {p} -> {q}")
    if "tags" in changes:
        more, less = _tag_steps(*changes["tags"])
        buffs += more
        nerfs += less
    verdict = "mixed" if buffs and nerfs else "buff" if buffs else "nerf" if nerfs else None
    return {"verdict": verdict, "buffs": buffs, "nerfs": nerfs}


TOKEN = re_compile(r"\[([0-9A-Za-z_ ]+)(?::[0-9A-Za-z_]+)?\]")   # an icon token ([fire], [light:tint]): its word
PUNCTUATION = re_compile(r"[.,;:!?\"\u201c\u201d\u2026()\[\]{}]|(?<=[^\W\d_])[-\u2010\u2011\u2013\u2014](?=[^\W\d_])")


def text_key(text: Optional[str]) -> str:
    """An ability as its words: icon tokens as their word, letter case, punctuation (a dash between letters too),
    apostrophes and spacing (line breaks too) left out (build_site text_key)."""
    words = TOKEN.sub(lambda m: m.group(1), text or "").casefold()
    return "".join(PUNCTUATION.sub("", words).replace("'", "").replace("\u2019", "").replace("\u2018", "").replace("_", "").split())


def formatting_only(old: Optional[str], new: Optional[str]) -> bool:
    """Two abilities that differ only in spacing, line breaks, punctuation, letter case or an icon token for its word
    (owner 11-09: not a change of the card)."""
    return (old or "") != (new or "") and text_key(old) == text_key(new)


# ========================
# Events
# ========================
def _event(kind: str, name: str, card: CardInfo, **extra) -> dict:
    out = {"kind": kind, "card": name, "title": getattr(card, "display_name", None) or name, "card_type": card.card_type,
           "stars": getattr(card, "stars", None), "path": card.path}
    out.update({k: v for k, v in extra.items() if v is not None})
    return out


def is_bot(by) -> bool:
    """A _by of the bot, any case (changes.js isBot)."""
    return isinstance(by, str) and by.lower() == BOT.lower()


def edit_credit(meta, fields) -> Optional[str]:
    """
    Who an edited event's fields are credited to: its entry's _by; a field in its _bot, the bot (the migration's).
    A mix: the _by when one field is its, else the bot (it wrote them all). meta: the entry's _-keys, or its _by alone.
    """
    if not isinstance(meta, dict):
        return meta
    by = meta.get("_by")
    bot = meta.get(BOT_FIELDS) if isinstance(meta.get(BOT_FIELDS), list) else []

    def of(field: str):
        return BOT if EDIT_FIELD.get(field, field) in bot else by
    credits = {of(f) for f in fields}
    if len(credits) == 1:
        return credits.pop()
    return by if by in credits or not credits else BOT


def events(before: dict, after: dict, archived_before: dict, archived_after: dict, reparsed: Optional[dict] = None,
           edits_by: Optional[dict] = None, index=None, file_commits: Optional[dict] = None) -> list:
    """
    One build's events from the database before and after it.

    Args:
        before: stats before ({name: CardInfo})
        after: stats after
        archived_before: archived cards before ({name: last version}, StatsDatabase.archived())
        archived_after: archived cards after
        reparsed: names this build parsed from their PSD (a data change with the same blob is then a re-read); of a
            re-read card's changes only its parse's are the PSD's (split_changes), an edit beside them is its own event
        edits_by: card name -> its Card Edits.json entry's _-keys (edit_credit), or its _by
        index: card_history.GitIndex: the causing commit and its git author (a new blob's intro, the deletion)
        file_commits: {"edited": the newest commit of Card Edits.json, "authors": of Card Authors.txt} (their events' commit)

    Returns:
        Events in name order, unnumbered (number() gives the seq); a duplicate's change is its original's, but for its
        own artwork sources (OWN_FIELDS)
    """
    reparsed, edits_by, file_commits = reparsed or {}, edits_by or {}, file_commits or {}
    renamed = {(getattr(archived_after.get(n), "archived", None) or {}).get("renamed_to")
               for n in before if n not in after and n in archived_after}           # a rename's new name: no "new" line

    def source(blob: Optional[str] = None, sha: Optional[str] = None, path: Optional[str] = None) -> dict:
        """The causing commit and its author: the path's newest commit (the one that put this blob there: a revert to
        an old blob is this push's), else the blob's intro, else a known commit."""
        found = None
        if index is not None:
            found = (index.newest.get(path) if path else None) or (index.blob_intro(blob) if blob else None) \
                or (index.by_sha.get(sha or "") if sha else None)
        return {"commit": found.sha, "by": found.author} if found is not None else ({"commit": sha} if sha else {})
    out = []
    for name in sorted(set(before) | set(after) | set(archived_after)):
        old, new = before.get(name), after.get(name)
        if any(c is not None and picture_only(c) for c in (old, new)):
            # a PNG-only duplicate (a variant picture's or a marked PSD's events are its own)
            both = old is not None and new is not None
            own = {k: v for k, v in field_changes(old, new).items() if k in OWN_FIELDS} if both else {}
            if own:                                       # a duplicate's own artwork sources: its own edit
                out.append(_event("edited", name, new, fields=own, edit_by=edit_credit(edits_by.get(name), own),
                                  **({"commit": file_commits["edited"]} if file_commits.get("edited") else {})))
            continue
        if old is None and new is not None:
            if name in renamed and name not in archived_before:
                continue
            out.append(_event("restored" if name in archived_before else "new", name, new, by=new.author,
                              commit=source(getattr(new, "blob_sha", None), path=new.path).get("commit")))
        elif old is not None and new is None:
            tag = getattr(archived_after.get(name), "archived", None) or {}
            if name in archived_after:
                out.append(_event("renamed" if tag.get("renamed_to") else "archived", name, old,
                                  renamed_to=tag.get("renamed_to"), moved_to=tag.get("moved_to"),
                                  **source(sha=tag.get("commit"))))
        elif old is not None and new is not None:
            changes = field_changes(old, new)
            blobs = [getattr(old, "blob_sha", None), getattr(new, "blob_sha", None)]
            moved = old.path != new.path
            psd, edits = split_changes(old, new, changes) if changes and name in reparsed else ({}, changes)
            if psd:
                kind = "changed" if blobs[0] != blobs[1] else "moved" if moved else "reparse"   # a move: folder data
                out.append(_event(kind, name, new, fields=psd, blob=blobs if blobs[0] != blobs[1] else None,
                                  from_path=old.path if moved else None, **classify(psd),
                                  **(source(path=new.path) if kind == "moved" else source(blobs[1], path=new.path))))
            elif blobs[0] != blobs[1]:                    # an edit in the same build is its own event (below)
                out.append(_event("art", name, new, blob=blobs, from_path=old.path if moved else None,
                                  **source(blobs[1], path=new.path)))
            elif moved:
                out.append(_event("moved", name, new, from_path=old.path, **source(path=new.path)))
            if edits:
                kind = "authors" if set(edits) == {"author"} else "edited"
                out.append(_event(kind, name, new, fields=edits,
                                  edit_by=edit_credit(edits_by.get(name), edits) if kind == "edited" else None,
                                  **classify(edits), **({"commit": file_commits[kind]} if file_commits.get(kind) else {})))
    return out


def number(new_events: list, last_seq: int, posted_seq: int = 0, at: Optional[float] = None, **common) -> list:
    """Give events their seq (max(the largest seq, posted.seq) + 1 onward: a restore never reuses a number) and the
    build's common keys (at, build, card_commit...)."""
    seq = max(last_seq, posted_seq)
    stamp = at if at is not None else datetime.now(timezone.utc).timestamp()
    out = []
    for event in new_events:
        seq += 1
        out.append({"seq": seq, "at": stamp, **{k: v for k, v in common.items() if v is not None}, **event})
    return out


def assign_pushes(new_events: list, pushes: list, commits: list, base: Optional[str] = None,
                  alone: Optional[set] = None, guards: Optional[set] = None) -> list:
    """
    Each event's push (in place): the activity list's push whose before..after holds the event's commit, else the
    run of consecutive commits by one author holding it. A re-read, or an event whose commit is in neither, gets none
    (it posts with its build).

    Args:
        new_events: One build's events (their "commit" from events())
        pushes: [{"after", "before", "by", "at", "commits": [sha, ...]}], oldest first (the activity list)
        commits: [(sha, at, author)] of the build's new commits, oldest first (the fallback)
        base: The commit before them (the last build's), the first fallback push's before
        alone: Commits that are a push of their own by CARD_SYNC (card_data --own: the main commits the card sync pushed)
        guards: The guards' commits: a push of their own by GUARDS, listed or not

    Returns:
        The events in push order (oldest push first, push-less ones last), ready for number()
    """
    found: dict = {}
    for push in pushes:
        for sha in push.get("commits") or [push["after"]]:
            found.setdefault(sha, {k: push.get(k) for k in ("after", "before", "by", "at")})
            if push["after"] in (guards or ()):
                found[sha]["by"] = GUARDS
    previous = base
    group: list = []
    groups: list = []
    for sha, at, author in commits:                       # consecutive commits by one author: one push
        if sha in found:
            previous = sha
            continue
        if sha in (alone or ()) or sha in (guards or ()):
            found[sha] = {"after": sha, "before": previous, "by": GUARDS if sha in (guards or ()) else CARD_SYNC, "at": at}
            group, previous = [], sha
            continue
        if group and group[0][2] == author and group[len(group) - 1][0] == previous:
            group.append((sha, at, author, previous))
        else:
            group = [(sha, at, author, previous)]
            groups.append(group)
        previous = sha
    for group in groups:
        last = group[len(group) - 1]
        push = {"after": last[0], "before": group[0][3], "by": last[2], "at": last[1]}
        for sha, _, _, _ in group:
            found[sha] = push
    place = {sha: i for i, (sha, _, _) in enumerate(commits)}   # a push's place: its after commit's in the log
    for event in new_events:
        push = found.get(event.get("commit") or "") if event["kind"] != "reparse" else None
        if push is not None:
            event["push"] = dict(push)
    return sorted(new_events, key=lambda e: (e["kind"] == "reparse", "push" not in e,
                                             place.get((e.get("push") or {}).get("after"), len(place))))


# ========================
# db/changelog files
# ========================
def read_events(db: str) -> list:
    """Every event in db/changelog/*.jsonl, in seq order."""
    found = []
    for path in sorted(glob(join(db, CHANGELOG_DIR, "*.jsonl"))):
        with open(path, encoding="utf-8") as fh:
            found += [loads(line) for line in fh if line.strip()]
    return sorted(found, key=lambda e: e.get("seq", 0))


def append_events(db: str, new_events: list) -> None:
    """Append to each event's month file (YYYY-MM.jsonl, by its time in TIMEZONE), one event per line."""
    by_month: dict = {}
    for event in new_events:
        month = datetime.fromtimestamp(event["at"], TIMEZONE).strftime("%Y-%m")
        by_month.setdefault(month, []).append(event)
    folder = join(db, CHANGELOG_DIR)
    makedirs(folder, exist_ok=True)
    for month, items in by_month.items():
        path = join(folder, f"{month}.jsonl")
        text = ""
        if isfile(path):
            with open(path, encoding="utf-8") as fh:
                text = fh.read()
        write_text(path, text + "".join(dumps(e, ensure_ascii=False, sort_keys=True) + "\n" for e in items))


def read_posted(db: str) -> dict:
    path = join(db, CHANGELOG_DIR, POSTED_JSON)
    if not isfile(path):
        return {"seq": 0, "push": None}
    with open(path, encoding="utf-8") as fh:
        return loads(fh.read())


def advance(db: str, seq: int, push: Optional[str] = None) -> None:
    """Move the posted cursor past one accepted message (its last event and push); never backwards."""
    posted = read_posted(db)
    if seq <= posted.get("seq", 0):
        return
    makedirs(join(db, CHANGELOG_DIR), exist_ok=True)
    write_text(join(db, CHANGELOG_DIR, POSTED_JSON), dumps({"seq": seq, "push": push or None}) + "\n")


# ========================
# The posts (one per push)
# ========================
def site_ids(data) -> dict:
    """The website's cards.json -> {card name: share id}, keyed by each card's `key` (the database's name; its `name`
    is the shown title), else its name."""
    cards = data.get("cards", data) if isinstance(data, dict) else data
    return {c.get("key") or c.get("name"): c["id"] for c in cards or [] if isinstance(c, dict) and c.get("id")}


def card_link(name: str, site: str, ids: Optional[dict] = None) -> str:
    """The card's share page (SITE/c/<id>/), else a search for the card."""
    card_id = (ids or {}).get(name)
    if card_id:
        return f"{site}/c/{card_id}/"
    return f"{site}/#/cards?q={quote(name)}"


def _md(text: str) -> str:
    return str(text).replace("[", "(").replace("]", ")")


def _value(value) -> str:
    if value is None or value == "" or value == []:
        return "-"
    return ", ".join(map(str, value)) if isinstance(value, list) else str(value)


def _clip(text: str, size: int = TEXT_SHORT) -> str:
    """Cut at a word, "\u2026" after."""
    if len(text) <= size:
        return text
    cut = text[:size + 1].rsplit(" ", 1)[0] if " " in text[:size + 1] else text[:size]
    return cut.rstrip(" ,;:") + "\u2026"


def _quote(words: list) -> str:
    return "'" + _md(_clip(" ".join(words))).replace("*", "\\*").replace("_", "\\_").replace("~", "\\~").replace("`", "'") + "'"


def text_change(old: Optional[str], new: Optional[str]) -> str:
    """
    "'have +1 DEF.' \u2192 'have +0.5 DEF.'": the words an edit changed with one word around them (at most TEXT_HUNKS
    spots), each side cut at a word; a short text whole; "rewritten: 'the new text\u2026'" when most words changed.
    """
    a, b = (str(t or "").split() for t in (old, new))
    if not b:
        return "removed"
    if not a:
        return f"\u2192 {_quote(b)}"
    if a == b:                                            # whitespace alone
        return "line breaks only" if str(old).count("\n") != str(new).count("\n") else "spacing only"
    if len(" ".join(a)) <= TEXT_SHORT and len(" ".join(b)) <= TEXT_SHORT:
        return f"{_quote(a)} \u2192 {_quote(b)}"
    hunks = [op for op in SequenceMatcher(None, a, b, autojunk=False).get_opcodes() if op[0] != "equal"]
    if not hunks:
        return f"{_quote(a)} \u2192 {_quote(b)}"
    if sum(max(i2 - i1, j2 - j1) for _, i1, i2, j1, j2 in hunks) > max(len(a), len(b)) // 2:
        return f"rewritten: {_quote(b)}"
    spots = [f"{_quote(a[max(0, i1 - 1):i2 + 1])} \u2192 {_quote(b[max(0, j1 - 1):j2 + 1])}"
             for _, i1, i2, j1, j2 in hunks[:TEXT_HUNKS]]
    more = f" (+{len(hunks) - TEXT_HUNKS} more)" if len(hunks) > TEXT_HUNKS else ""
    return ", ".join(spots) + more


def _creatures_change(a, b) -> str:
    """"creature 2 HP 5 \u2192 6", else the new stats (HP/DEF/ATK/SPD) when creatures came, went or changed in number."""
    if not b:
        return "creature stats removed"
    if a and len(a) == len(b):
        steps = [f"creature {i} {label} {_value(p)} \u2192 {_value(q)}"
                 for i, (x, y) in enumerate(zip(a, b), 1) for (_, _, label), p, q in zip(STATS, x, y) if p != q]
        if steps:
            return ", ".join(steps)
    return "creature stats \u2192 " + ", ".join("/".join(_value(v) for v in c) for c in b) + " (HP/DEF/ATK/SPD)"


def _changes(fields: dict) -> str:
    """"HP 5 \u2192 6; ability 'heal 2' \u2192 'heal 3'; title 'kirby' \u2192 'Kirby'; author A \u2192 B; subseries \u2192 Kirby" (an old value
    shown, sources aside); timing tags show through the ability and the verdict."""
    parts = []
    for key, (a, b) in fields.items():
        if key in ("hp", "def", "atk", "spd", "stars"):
            parts.append(f"{key.upper() if key != 'stars' else 'stars'} {_value(a)} \u2192 {_value(b)}")
        elif key == "ability":
            parts.append(f"ability {text_change(a, b)}")
        elif key == "creatures":
            parts.append(_creatures_change(a, b))
        elif key == "hidden":
            parts.append("hidden" if b else "shown again")
        elif key == "name":                               # old and new: most differ in case alone
            parts.append(f"title {text_change(a, b)}")
        elif key != "tags":
            word, was = FIELD_WORD.get(key, key.replace("_", " ")), _value(a)
            old = "" if key in ("art", "ability_art") or was == "-" else f" {_md(was)}"   # sources: long
            parts.append(f"{word}{old} \u2192 {_md(_value(b))}")
    return "; ".join(parts)


def _kind_label(card_type: Optional[str]) -> str:
    """The website's word for a card kind (app.js: KIND_LABEL[type] || type)."""
    return KIND_LABEL.get(card_type or "", card_type or KIND_LABEL["unknown"])


def _line(event: dict, site: str, ids: Optional[dict]) -> str:
    """One event as a line of the post, its card linked."""
    kind = event["kind"]
    if kind == "restore":
        return f"Card data restored from {_md((event.get('restored_from') or '?')[:8])}"
    title = f"[{_md(event.get('title') or event['card'])}]({card_link(event['card'], site, ids)})"
    if kind == "new":
        stars = f", {event['stars']}\u2605" if event.get("stars") else ""
        by = f" by {_md(event['by'])}" if event.get("by") else ""
        return f"{title} ({_kind_label(event.get('card_type'))}{stars}){by}"
    if kind in ("changed", "edited", "authors"):
        verdict = f" ({event['verdict']})" if event.get("verdict") else ""
        by = event.get("edit_by") or ""
        where = f" (by {BOT})" if is_bot(by) else f" (by {_md(by)}, on the website)" if by else ""
        return f"{title}: {_changes(event.get('fields') or {})}{verdict}{where}"
    if kind == "renamed":
        return f"[{_md(event['renamed_to'])}]({card_link(event['renamed_to'], site, ids)}) (renamed from {_md(event['card'])})"
    if kind == "moved":
        return f"{title} moved to {_md((event.get('path') or '').rsplit('/', 1)[0])}"
    if kind == "archived":
        gone = f"moved to {_md(event['moved_to'].rsplit('/', 1)[0])}" if event.get("moved_to") else "PSD deleted"
        return f"{_md(event.get('title') or event['card'])} ({gone})"
    if kind in ("baked", "bake-refused"):
        why, was = event.get("reason") or "", event.get("renamed_from")
        if was:                                           # the rename said once, at the end
            why = _without_rename(why, was, event["card"])
        why = PLAIN.sub("", why)
        png = " (PNG re-made)" if kind == "baked" and event.get("pngs") else ""
        renamed = f" (renamed from {_md(was)})" if was else ""
        return f"{title}: {_md(why)}{png}{renamed}" if why else f"{title}{png}{renamed}"
    return title


def _without_rename(reason: str, old: str, new: str) -> str:
    """A bake reason without its "renamed: Old -> New" (card_bake._describe's last job)."""
    said = f"renamed: {old} -> {new}"
    for cut in (f", {said}", f"{said}; ", said):
        if cut in reason:
            return reason.replace(cut, "", 1)
    return reason


def _when(at: Optional[float]) -> str:
    """"Oct 22 14:05" in TIMEZONE."""
    d = datetime.fromtimestamp(at or 0, TIMEZONE)
    return f"{d.strftime('%b')} {d.day} {d.strftime('%H:%M')}"


def title_of(group: list) -> str:
    """"Zomoroko, Oct 22 14:05: 1 new, 2 changed": who pushed and when (a push-less group: the card sync, at its build)."""
    push = group[0].get("push")
    who, at = (push.get("by") or "Someone", push.get("at")) if push else (CARD_SYNC, group[0].get("at"))
    if push and who == CARD_SYNC:                         # the run's own commit M (card_data --own)
        who = "Website edits baked into the PSDs"
    kinds = [e["kind"] for e in group]
    counts: dict = {}
    for kind, word in COUNT_WORDS:
        if kinds.count(kind):
            counts[word] = counts.get(word, 0) + kinds.count(kind)
    return f"{_md(who)}, {_when(at)}: " + ", ".join(f"{n} {word}" for word, n in counts.items())


def link_of(group: list, repository: str) -> Optional[str]:
    """The push's compare view (before...after), its commit without a before, else the build's run."""
    push = group[0].get("push")
    if push and repository:
        before, after = push.get("before") or "", push["after"]
        return (f"{GITHUB}{repository}/compare/{before[:12]}...{after[:12]}" if before.strip("0")
                else f"{GITHUB}{repository}/commit/{after}")
    build = group[0].get("build") or ""
    return build if build.startswith("http") else None


def payload(group: list, site: str = SITE_URL, ids: Optional[dict] = None, repository: str = "") -> Optional[dict]:
    """
    One push's Discord message: titled title_of(), linked link_of(), one embed per section (SECTIONS order), the
    website's kind words, within Discord's limits (EMBEDS_MAX embeds, DESCRIPTION_MAX each, MESSAGE_MAX in all); what
    doesn't fit becomes "N more on the website". A baked card's new art is its bake line's "(PNG re-made)": no New art
    line; its rename, the line's "(renamed from Old)", and its move, a stars bake's "(moved to 4 Stars)": no Moved and
    renamed line (owner, 2026-11-11).

    Returns:
        The webhook body, or None when there is nothing to post
    """
    if not group:
        return None
    baked = {e["card"] for e in group if e["kind"] == "baked"}
    moved = {e["card"] for e in group if e["kind"] == "baked" and "(moved to " in (e.get("reason") or "")}
    was: dict = {}                                        # a baked card -> its old name
    for e in group:
        if e["kind"] == "renamed" and e.get("renamed_to") in baked:
            was.setdefault(e["renamed_to"], e["card"])
    group = [dict(e, renamed_from=was[e["card"]]) if e["kind"] == "baked" and e["card"] in was else e for e in group
             if not (e["kind"] == "art" and e["card"] in baked)
             and not (e["kind"] == "renamed" and was.get(e.get("renamed_to")) == e["card"])
             and not (e["kind"] == "moved" and e["card"] in moved)]
    title, url = title_of(group), link_of(group, repository)
    embeds, total, hidden = [], len(title), 0
    for kinds, heading in SECTIONS:
        lines = [_line(e, site, ids) for e in group if e["kind"] in kinds]
        if not lines:
            continue
        text, shown = f"**{heading}**", 0
        if len(embeds) < EMBEDS_MAX - 1:                      # the last embed is kept for the closing line
            for line in lines:
                grown = f"{text}\n{line}"
                if len(grown) > DESCRIPTION_MAX or total + len(grown) > MESSAGE_MAX - TAIL_MAX:
                    break
                text, shown = grown, shown + 1
        hidden += len(lines) - shown
        if shown:
            embeds.append({"description": text})
            total += len(text)
    if hidden:
        embeds.append({"description": f"{hidden} more on the website: {site}/#/changes/changelog"})
    if not embeds:                                        # a kind with no section: the title alone
        embeds.append({})
    embeds[0] = {"title": title, **({"url": url} if url else {}), **embeds[0]}
    return {"embeds": embeds, "allowed_mentions": {"parse": []}}


def silent(event: dict) -> bool:
    """An event no Discord post shows: a guard event, any event of a guards commit's push (owner, 2026-10-25), a
    re-read (owner, 2026-11-11), a sweep's own art or text change, rename or move (its "sweep", card_data.mark_sweeps),
    or a minor one (card_data.mark_minor)."""
    return event["kind"] in SILENT or (event.get("push") or {}).get("by") == GUARDS \
        or (event["kind"] in (*SWEPT, "renamed", "moved") and bool(event.get("sweep"))) or bool(event.get(MINOR))


def messages(pending: list, site: str = SITE_URL, ids: Optional[dict] = None, repository: str = "",
             limit: int = MESSAGES_MAX) -> list:
    """
    The posts for the events past the cursor: one per push (per build: a push whose cards wait for a later run posts
    again then), oldest first, at most `limit`; the rest wait for the next run. A group of silent() events alone (a
    build's re-reads) is a message without a body: the cursor moves past it, nothing is sent.

    Returns:
        [{"body": webhook body or None, "cursor": {"seq": its last event, "push": its after sha or None}}]
    """
    out = []
    ordered = sorted(pending, key=lambda e: e.get("seq", 0))
    key = lambda e: (e.get("build"), (e.get("push") or {}).get("after"), e["kind"] == "reparse")
    for _, items in groupby(ordered, key=key):
        group = list(items)
        if sum(1 for m in out if m["body"] is not None) >= limit:
            break
        last = group[len(group) - 1]
        out.append({"body": payload([e for e in group if not silent(e)], site, ids, repository),
                    "cursor": {"seq": last.get("seq", 0), "push": (last.get("push") or {}).get("after")}})
    return out


# ========================
# CLI
# ========================
def main(argv: Optional[list] = None) -> int:
    parser = ArgumentParser(prog="python -m commands.changelog", description="The card changelog.")
    sub = parser.add_subparsers(dest="command", required=True)
    post = sub.add_parser("post", help="the Discord messages (one per push) of the events past the posted cursor")
    post.add_argument("--db", required=True)
    post.add_argument("--site", default=SITE_URL)
    post.add_argument("--ids", help="the website's cards.json (names to share ids)")
    post.add_argument("--repo", default="", help="owner/name of the card repo, for the compare links (default: the manifest's)")
    post.add_argument("--out", required=True, help="the messages to send, each with the cursor to advance after it")
    move = sub.add_parser("advance", help="move the posted cursor once Discord accepted a message")
    move.add_argument("--db", required=True)
    move.add_argument("--seq", type=int, required=True)
    move.add_argument("--push", default="", help="that message's push (its after sha)")
    args = parser.parse_args(argv)
    if args.command == "post":
        posted = read_posted(args.db)
        pending = [e for e in read_events(args.db) if e.get("seq", 0) > posted.get("seq", 0)]
        ids = None
        if args.ids:
            with open(args.ids, encoding="utf-8") as fh:
                data = loads(fh.read())
            ids = site_ids(data)
        repository = args.repo
        if not repository and isfile(join(args.db, "manifest.json")):
            with open(join(args.db, "manifest.json"), encoding="utf-8") as fh:
                repository = loads(fh.read()).get("card_repo") or ""
        out = messages(pending, args.site, ids, repository)
        write_text(args.out, dumps({"messages": out}, ensure_ascii=False, indent=1) + "\n")
        waiting = len(pending) - sum(1 for e in pending if e.get("seq", 0) <= max((m["cursor"]["seq"] for m in out), default=0))
        sent = sum(1 for m in out if m["body"] is not None)
        print(f"{sent} message{'' if sent == 1 else 's'} to post" + (f", {waiting} events wait for the next run." if waiting else ".")
              if out else "Nothing to post.")
        return 0
    advance(args.db, args.seq, args.push)
    print(f"Posted cursor at {args.seq}" + (f" ({args.push[:8]})." if args.push else "."))
    return 0


# ========================
# Module Exports
# ========================
__all__ = [
    'CARD_SYNC',
    'FIELDS',
    'GUARDS',
    'MINOR',
    'SWEEP_TRAILER',
    'SWEPT',
    'KINDS',
    'SILENT',
    'TAG_PATTERN',
    'advance',
    'append_events',
    'assign_pushes',
    'card_link',
    'classify',
    'edit_credit',
    'events',
    'field_changes',
    'fields_of',
    'formatting_only',
    'link_of',
    'main',
    'messages',
    'number',
    'payload',
    'title_of',
    'read_events',
    'read_posted',
    'silent',
    'site_ids',
    'split_changes',
    'tags_of',
    'text_change',
    'text_key',
]


if __name__ == "__main__":
    raise SystemExit(main())
