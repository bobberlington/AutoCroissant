"""
The card changelog, without Discord, network or psd-tools: per-version field changes (stats, ability, timing tags,
name, types, stars and the other data fields) with a buff/nerf verdict, the build's events from the database before
and after (new, changed, art, reparse, moved, renamed, archived, restored, edited, authors; restore for a rollback),
the push that brought each (the activity list, else consecutive commits by one author), their numbering, and one
Discord message per push within Discord's limits. Design: docs/Campaigns/DATA_BUILD_ACTION.md section 12.1.

CLI: python -m commands.changelog history --pickles <dir> <card>  |  post --db <db dir> --site <url> --out <file>
     |  advance --db <db dir> --seq <n> --push <sha>  (after Discord accepted that message)
"""
from argparse import ArgumentParser
from datetime import datetime, timezone
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
from commands.card_models import CardInfo

# ========================
# Configuration
# ========================
STATS = (("hp", "hp", "HP"), ("def", "defense", "DEF"), ("atk", "attack", "ATK"), ("spd", "speed", "SPD"))
FIELDS = ("name", "card_type", "stars", "subtype", "series", "subseries", "types", "hp", "def", "atk", "spd", "creatures",
          "ability", "tags", "author", "note", "hidden")
# timing tags in ability text: the cooldown and uses tags (2026-10-21) and the older per-turn ones
TAG_PATTERN = re_compile(r"\((?:-?\d+/\d+|\d+|X)?(?:GCD|CD|U|PGT|PG|PT)\)|\((?:FT)?E(?:\d+|X)G?T\)"
                         r"|\((?:OYT|AAT|BOT|ENDT|BOGT|ENDGT|OFS|OS|OD|WOF)\)")
COOLDOWN = re_compile(r"\((\d+)(G?CD)\)")              # one number: shorter is stronger
USES = re_compile(r"\((\d+)U\)")                         # more is stronger
KINDS = ("new", "changed", "art", "reparse", "moved", "renamed", "archived", "restored", "edited", "authors", "restore")
KIND_LABEL = {"creature": "Creature", "item": "Item", "field": "Field card", "minion": "Minion", "nme": "N.M.E. item",
              "aux item": "Auxiliary item", "debuff": "Debuff", "buff": "Buff", "creatures": "Auxiliary creature",
              "unknown": "Other"}                        # site/assets/app.js KIND_LABEL
SECTIONS = ((("new",), "New cards"), (("changed",), "Changed"), (("art",), "New art"), (("moved", "renamed"), "Moved and renamed"),
            (("archived",), "Archived"), (("restored",), "Restored"), (("edited", "authors"), "Database edits"),
            (("restore",), "Rollback"))                  # re-reads: one line
COUNT_WORDS = (("new", "new"), ("changed", "changed"), ("art", "new art"), ("moved", "moved"), ("renamed", "renamed"),
               ("archived", "archived"), ("restored", "restored"), ("edited", "edited"), ("authors", "edited"),
               ("reparse", "re-read"), ("restore", "data rollback"))   # the title's counts: "1 new, 2 changed"
CARD_SYNC = "Card sync"                                  # the push of the run's own commit M
GITHUB = "https://github.com/"
MESSAGES_MAX = 20                                        # per run; the rest wait for the next one (the cursor)
EMBEDS_MAX = 10                                          # Discord: embeds per message
DESCRIPTION_MAX = 4096                                   # characters per embed description
MESSAGE_MAX = 6000                                       # characters per message, all embeds together
TAIL_MAX = 300                                           # kept for the closing line
CHANGELOG_DIR = "changelog"                              # db/changelog/YYYY-MM.jsonl
POSTED_JSON = "posted.json"                              # db/changelog/posted.json: {"seq", "push"} of the last post


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
    creatures' stats, the ability, its timing tags, author, note and hidden."""
    stats = getattr(card, "stats", None)
    out = {"name": getattr(card, "display_name", None) or card.name, "card_type": card.card_type,
           "stars": getattr(card, "stars", None), "subtype": getattr(card, "subtype", None),
           "series": getattr(card, "series", None), "subseries": getattr(card, "subseries", None),
           "types": list(getattr(card, "types", None) or [])}
    for key, attr, _ in STATS:
        out[key] = _stat(getattr(stats, attr, None)) if stats is not None else None
    creatures = getattr(card, "creatures", None)
    out["creatures"] = [[_stat(getattr(c, attr, None)) for _, attr, _ in STATS] for c in creatures] if creatures else None
    out["ability"] = getattr(card, "ability", None)
    out["tags"] = tags_of(out["ability"])
    out["author"], out["note"], out["hidden"] = getattr(card, "author", None), getattr(card, "note", None), bool(getattr(card, "hidden", False))
    return out


def field_changes(old: CardInfo, new: CardInfo) -> dict:
    """{field: [old, new]} for each changelog field that differs (FIELDS order)."""
    a, b = fields_of(old), fields_of(new)
    return {key: [a[key], b[key]] for key in FIELDS if a[key] != b[key]}


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


def version_changes(versions: list, current: Optional[CardInfo]) -> list:
    """
    A card's history as steps: each version to the next (the last to the current card).

    Returns:
        [{"from": the newer version's start, "cause": its cause, "changes": {...}, "verdict": ..., "buffs", "nerfs"}],
        oldest first
    """
    chain = list(versions) + ([current] if current is not None else [])
    steps = []
    for old, new in zip(chain, chain[1:]):
        info = getattr(new, "version", None) or {}
        changes = field_changes(old, new)
        steps.append({"from": info.get("from", new.timestamp), "cause": info.get("cause"), "changes": changes, **classify(changes)})
    return steps


# ========================
# Events
# ========================
def _event(kind: str, name: str, card: CardInfo, **extra) -> dict:
    out = {"kind": kind, "card": name, "title": getattr(card, "display_name", None) or name, "card_type": card.card_type,
           "stars": getattr(card, "stars", None), "path": card.path}
    out.update({k: v for k, v in extra.items() if v is not None})
    return out


def events(before: dict, after: dict, archived_before: dict, archived_after: dict, reparsed: Optional[dict] = None,
           edits_by: Optional[dict] = None, index=None, file_commits: Optional[dict] = None) -> list:
    """
    One build's events from the database before and after it.

    Args:
        before: stats before ({name: CardInfo})
        after: stats after
        archived_before: archived cards before ({name: last version}, StatsDatabase.archived())
        archived_after: archived cards after
        reparsed: names this build parsed from their PSD (a data change with the same blob is then a re-read)
        edits_by: card name -> the `_by` of its Card Edits.json entry
        index: card_history.GitIndex: the causing commit and its git author (a new blob's intro, the deletion)
        file_commits: {"edited": the newest commit of Card Edits.json, "authors": of Card Authors.txt} (their events' commit)

    Returns:
        Events in name order, unnumbered (number() gives the seq); a duplicate's change is its original's
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
        if (new is not None and getattr(new, "duplicate_of", None)) or (old is not None and getattr(old, "duplicate_of", None)):
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
                                  renamed_to=tag.get("renamed_to"), **source(sha=tag.get("commit"))))
        elif old is not None and new is not None:
            changes = field_changes(old, new)
            blobs = [getattr(old, "blob_sha", None), getattr(new, "blob_sha", None)]
            moved = old.path != new.path
            if changes:
                if name in reparsed:
                    kind = "changed" if blobs[0] != blobs[1] else "moved" if moved else "reparse"   # a move: folder data
                elif set(changes) == {"author"}:
                    kind = "authors"
                else:
                    kind = "edited"
                out.append(_event(kind, name, new, fields=changes, edit_by=edits_by.get(name) if kind == "edited" else None,
                                  blob=blobs if blobs[0] != blobs[1] else None, from_path=old.path if moved else None,
                                  **classify(changes), **(source(blobs[1], path=new.path) if kind in ("changed", "reparse")
                                                          else source(path=new.path) if kind == "moved"
                                                          else {"commit": file_commits[kind]} if file_commits.get(kind) else {})))
            elif blobs[0] != blobs[1]:
                out.append(_event("art", name, new, blob=blobs, from_path=old.path if moved else None,
                                  **source(blobs[1], path=new.path)))
            elif moved:
                out.append(_event("moved", name, new, from_path=old.path, **source(path=new.path)))
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
                  alone: Optional[set] = None) -> list:
    """
    Each event's push (in place): the activity list's push whose before..after holds the event's commit, else the
    run of consecutive commits by one author holding it. A re-read, or an event whose commit is in neither, gets none
    (it posts with its build).

    Args:
        new_events: One build's events (their "commit" from events())
        pushes: [{"after", "before", "by", "at", "commits": [sha, ...]}], oldest first (the activity list)
        commits: [(sha, at, author)] of the build's new commits, oldest first (the fallback)
        base: The commit before them (the last build's), the first fallback push's before
        alone: Commits that are a push of their own by CARD_SYNC (the run's local bake commit B, never listed)

    Returns:
        The events in push order (oldest push first, push-less ones last), ready for number()
    """
    found: dict = {}
    for push in pushes:
        for sha in push.get("commits") or [push["after"]]:
            found.setdefault(sha, {k: push.get(k) for k in ("after", "before", "by", "at")})
    previous = base
    group: list = []
    groups: list = []
    for sha, at, author in commits:                       # consecutive commits by one author: one push
        if sha in found:
            previous = sha
            continue
        if sha in (alone or ()):
            found[sha] = {"after": sha, "before": previous, "by": CARD_SYNC, "at": at}
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


def rewrite_events(db: str, fix) -> None:
    """Every event through fix(event) -> event, each month file rewritten only when it changed."""
    for path in sorted(glob(join(db, CHANGELOG_DIR, "*.jsonl"))):
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
        new = "".join(dumps(fix(loads(line)), ensure_ascii=False, sort_keys=True) + "\n" for line in text.splitlines() if line.strip())
        if new != text:
            write_text(path, new)


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
def card_link(name: str, site: str, ids: Optional[dict] = None) -> str:
    """The card's share page (SITE/c/<id>/), else a search for it."""
    card_id = (ids or {}).get(name)
    return f"{site}/c/{card_id}/" if card_id else f"{site}/#/cards?q={quote(name)}"


def _md(text: str) -> str:
    return str(text).replace("[", "(").replace("]", ")")


def _value(value) -> str:
    if value is None or value == "" or value == []:
        return "-"
    return ", ".join(map(str, value)) if isinstance(value, list) else str(value)


def _changes(fields: dict) -> str:
    """"HP 5 \u2192 6; ability changed; subseries \u2192 Kirby"; timing tags show through the ability and the verdict."""
    parts = []
    for key, (a, b) in fields.items():
        if key in ("hp", "def", "atk", "spd", "stars"):
            parts.append(f"{key.upper() if key != 'stars' else 'stars'} {_value(a)} \u2192 {_value(b)}")
        elif key == "ability":
            parts.append("ability changed")
        elif key == "creatures":
            parts.append("creature stats changed")
        elif key == "hidden":
            parts.append("hidden" if b else "shown again")
        elif key != "tags":
            parts.append(f"{'title' if key == 'name' else key.replace('_', ' ')} \u2192 {_md(_value(b))}")
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
        where = (f" (by {_md(by.split(':', 1)[1])}, via Discord)" if by.startswith("discord:")
                 else f" (by {_md(by)}, on the website)" if by else "")
        return f"{title}: {_changes(event.get('fields') or {})}{verdict}{where}"
    if kind == "renamed":
        return f"{_md(event['card'])} is now [{_md(event['renamed_to'])}]({card_link(event['renamed_to'], site, ids)})"
    if kind == "moved":
        return f"{title} moved to {_md((event.get('path') or '').rsplit('/', 1)[0])}"
    if kind == "archived":
        return f"{_md(event.get('title') or event['card'])} (PSD deleted)"
    return title


def _when(at: Optional[float]) -> str:
    """"Oct 22 14:05" in TIMEZONE."""
    d = datetime.fromtimestamp(at or 0, TIMEZONE)
    return f"{d.strftime('%b')} {d.day} {d.strftime('%H:%M')}"


def title_of(group: list) -> str:
    """"Zomoroko, Oct 22 14:05: 1 new, 2 changed": who pushed and when (a push-less group: the card sync, at its build)."""
    push = group[0].get("push")
    who, at = (push.get("by") or "Someone", push.get("at")) if push else (CARD_SYNC, group[0].get("at"))
    if push and who == CARD_SYNC:                         # the run's own commit M (rewrite_commit)
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
    doesn't fit becomes "N more on the website", and parser re-reads are one closing line.

    Returns:
        The webhook body, or None when there is nothing to post
    """
    if not group:
        return None
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
    tail = []
    rereads = sum(1 for e in group if e["kind"] == "reparse")
    if rereads:
        tail.append(f"The parser re-read {rereads} card{'' if rereads == 1 else 's'}.")
    if hidden:
        tail.append(f"{hidden} more on the website: {site}/#/changes/changelog")
    if tail:
        embeds.append({"description": " ".join(tail)})
    if not embeds:                                        # a kind with no section: the title alone
        embeds.append({})
    embeds[0] = {"title": title, **({"url": url} if url else {}), **embeds[0]}
    return {"embeds": embeds, "allowed_mentions": {"parse": []}}


def messages(pending: list, site: str = SITE_URL, ids: Optional[dict] = None, repository: str = "",
             limit: int = MESSAGES_MAX) -> list:
    """
    The posts for the events past the cursor: one per push (per build: a push whose cards wait for a later run posts
    again then), oldest first, at most `limit`; the rest wait for the next run.

    Returns:
        [{"body": webhook body, "cursor": {"seq": its last event, "push": its after sha or None}}]
    """
    out = []
    ordered = sorted(pending, key=lambda e: e.get("seq", 0))
    key = lambda e: (e.get("build"), (e.get("push") or {}).get("after"), e["kind"] == "reparse")
    for _, items in groupby(ordered, key=key):
        group = list(items)
        if len(out) >= limit:
            break
        last = group[len(group) - 1]
        out.append({"body": payload(group, site, ids, repository),
                    "cursor": {"seq": last.get("seq", 0), "push": (last.get("push") or {}).get("after")}})
    return out


def size(body: dict) -> int:
    """A message's characters as Discord counts them (embed titles and descriptions)."""
    return sum(len(e.get("title") or "") + len(e.get("description") or "") for e in body.get("embeds", []))


# ========================
# CLI
# ========================
def main(argv: Optional[list] = None) -> int:
    parser = ArgumentParser(prog="python -m commands.changelog", description="The card changelog.")
    sub = parser.add_subparsers(dest="command", required=True)
    hist = sub.add_parser("history", help="a card's versions as field changes with a buff/nerf verdict")
    hist.add_argument("--pickles", required=True, help="a folder holding stats.pkl and old_stats.pkl")
    hist.add_argument("card")
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
    if args.command == "history":
        from commands.card_db import read_pickles
        stats, old = read_pickles(args.pickles)
        versions, current = old.get(args.card) or [], stats.get(args.card)
        if not versions and current is None:
            print(f"No card named {args.card}.")
            return 1
        for step in version_changes(versions, current):
            when = datetime.fromtimestamp(step["from"], TIMEZONE).strftime("%Y-%m-%d") if step["from"] else "?"
            print(f"{when} ({step['cause'] or 'unknown'}): " + (", ".join(step["changes"]) or "no data change")
                  + (f" [{step['verdict']}: {'; '.join(step['buffs'] + step['nerfs'])}]" if step["verdict"] else ""))
        return 0
    if args.command == "post":
        posted = read_posted(args.db)
        pending = [e for e in read_events(args.db) if e.get("seq", 0) > posted.get("seq", 0)]
        ids = None
        if args.ids:
            with open(args.ids, encoding="utf-8") as fh:
                data = loads(fh.read())
            cards = data.get("cards", data) if isinstance(data, dict) else data
            ids = {c.get("name"): c.get("id") for c in cards if isinstance(c, dict) and c.get("id")}
        repository = args.repo
        if not repository and isfile(join(args.db, "manifest.json")):
            with open(join(args.db, "manifest.json"), encoding="utf-8") as fh:
                repository = loads(fh.read()).get("card_repo") or ""
        out = messages(pending, args.site, ids, repository)
        write_text(args.out, dumps({"messages": out}, ensure_ascii=False, indent=1) + "\n")
        waiting = len(pending) - sum(1 for e in pending if e.get("seq", 0) <= max((m["cursor"]["seq"] for m in out), default=0))
        print(f"{len(out)} message{'' if len(out) == 1 else 's'} to post" + (f", {waiting} events wait for the next run." if waiting else ".")
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
    'KINDS',
    'TAG_PATTERN',
    'advance',
    'append_events',
    'assign_pushes',
    'card_link',
    'classify',
    'events',
    'field_changes',
    'fields_of',
    'link_of',
    'main',
    'messages',
    'number',
    'payload',
    'title_of',
    'read_events',
    'read_posted',
    'rewrite_events',
    'size',
    'tags_of',
    'version_changes',
]


if __name__ == "__main__":
    raise SystemExit(main())
