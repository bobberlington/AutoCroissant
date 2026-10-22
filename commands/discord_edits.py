"""
Discord's card edits as commits to the card repo (docs/Campaigns/DATA_BUILD_ACTION.md section 11, decisions 2 and 3):
/update_metadata, /mass_replace and /alias write TTSCardMaker's site/Card Edits.json and site/Card Authors.txt in one
commit per command, never the pickles. The commit goes through card_bake.land (on main's current head, again from the
new head when main moved) and git_commit (the API with the bot's GIT_TOKEN, else the history clone pushed over SSH: the
path /bake_cards and /update_tts_save commit by). Its subject starts "Discord:", it carries an Edit-by trailer, and each
entry it writes gets _by ("discord:<username>") and _at. Then the bot applies the change in memory (CardEdits.apply,
CardAuthors.resolve, the alias registry), so /query shows it before the next /update_stats or card sync builds it.

Fields: author -> a [cards] line in Card Authors.txt; ability, stars, types, hp/def/atk/spd -> the "psd" kind (they persist
and show "PSD pending" on the website; the next bake prints ability and stats on the card, stars and types not yet);
subtype, series, card_type -> subtype, series, type (the "folder" kind); subseries; del_entry -> hidden: true. /mass_replace author rewrites the [cards] lines and the
[people] right-hand sides and adds a "[people] old = new" line; its other fields write one entry per card, in one commit.
/alias writes `aliases` on the entry of the card its PNG belongs to. aliases.pkl keeps counting until the alias migration
(migrate_aliases) sets Card Edits.json's aliases_moved; before that its aliases can't be changed here.

    python -m commands.discord_edits migrate-aliases --clone <card repo clone> [--ref HEAD] [--aliases aliases.pkl] [--out FILE]

edits_file and authors_file are ports of site/assets/edit.js editsFile and authorsFile.
"""
from argparse import ArgumentParser
from copy import deepcopy
from datetime import datetime, timezone
from json import loads
from os.path import basename
from re import compile as re_compile, fullmatch
from subprocess import run as run_process
from sys import stderr
from traceback import format_exc
from typing import Callable, Optional

from commands.card_edits import ALIASES_MOVED, ORDER, CardAuthors, CardEdits, _collate, dump_edits
from commands.card_models import card_name
from commands.github_api import blob_sha
from global_config import CARD_REPO, EDITS_PATHS

# ========================
# Configuration
# ========================
SUBJECT = "Discord:"                            # every commit's subject starts so (the website's Database edits tab)
EDIT_BY = "Edit-by"                             # the trailer key (card_bake.EDIT_BY, build_site.EDIT_BY)
DISCORD = "discord:"                            # an editor's _by and Edit-by: "discord:<username>" (card_bake.DISCORD)
BOT = "AutoCroissant"                           # the migration's _by (build_site.BOT_AUTHOR: counted for nobody)
FIELDS = frozenset(ORDER)
GROUP = re_compile(r"# --- credited to (.+) \((\d+)\) ---")     # authorsFile's group headers (fullmatch)
RULEBOOK_GROUP = "# --- rulebook pages ---"
NEW_AUTHORS = "[people]\n\n[cards]\n"           # authorsFile's file when there is none
ABOUT = ("Card data fixes from the website's Edit button. Keyed by card file name; each field overrides the PSD. Read by "
         "the website and the bot (/update_stats).")                    # edit.js ABOUT, for a new file
MOVED_NOTE = "AutoCroissant's aliases.pkl moved into the cards' aliases on {at}: the bot reads aliases only here"
# /update_metadata's options -> Card Edits.json fields, in the command's order
METADATA = (("ability", "ability"), ("stars", "stars"), ("subtype", "subtype"), ("series", "series"),
            ("subseries", "subseries"), ("hp", "hp"), ("defense", "def"), ("attack", "atk"), ("speed", "spd"),
            ("card_type", "type"), ("types", "types"))
NOT_BAKED = ("stars", "types")                  # PSD fields the bake refuses for now (psd_edit.NOT_YET)
MASS = {"series": "series", "subseries": "subseries", "subtype": "subtype", "card_type": "type"}   # /mass_replace -> field
BODY_LINES = 40                                 # a commit body lists at most this many cards
SHOWN = 60                                      # a value is cut to this many characters in a reply or commit body


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


def edits_file(text: Optional[str], ops: list, at: str, by: str) -> str:
    """
    edit.js editsFile: Card Edits.json with ops applied. A touched entry is matched like the authors file, lists its
    fields in ORDER, then its other keys (_psd kept), then _by and _at (replaced); an entry left without keys goes.

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
        for name in op.get("unset") or ():
            current.pop(name, None)
        current.update(op.get("set") or {})
        current.pop("_by", None)
        current.pop("_at", None)
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


def replace_author(text: Optional[str], old: str, new: str) -> tuple:
    """
    /mass_replace author: every [cards] line crediting old moves to new (authorsFile), every [people] line showing old
    shows new, and "old = new" joins [people] when old has no line there, so cards credited to the git name follow.
    A group left empty loses its header.

    Returns:
        (the text, the [cards] names moved, the [people] lines set {git name: new})
    """
    lines = (NEW_AUTHORS if text is None else text).split("\n")
    same = lambda value: value.strip().lower() == old.strip().lower()

    def entries(name: str) -> list:
        found = _section(lines, name)
        return [lines[j].split("=", 1) for j in range(*found) if not lines[j].strip().startswith("#") and "=" in lines[j]] \
            if found else []
    moved = [left.strip() for left, right in entries("cards") if same(right)]
    people = {left.strip(): new for left, right in entries("people") if same(right)}
    if not any(same(left) for left, _ in entries("people")):
        people[old.strip()] = new
    out = authors_file(text, [{"key": name, "author": new} for name in moved], people).split("\n")
    i = len(out) - 1
    while i >= 0:                               # old's emptied group headers go, with the blank line before them
        found = GROUP.fullmatch(out[i].strip())
        follows = next((line.strip() for line in out[i + 1:] if line.strip()), "")
        if found and same(found.group(1)) and found.group(2) == "0" and (not follows or follows.startswith(("#", "["))):
            del out[i]
            if i > 0 and not out[i - 1].strip():
                del out[i - 1]
                i -= 1
        i -= 1
    return "\n".join(out), moved, people


# ========================
# The commit
# ========================
def _short(value) -> str:
    text = ", ".join(value) if isinstance(value, list) else str(value)
    text = " ".join(text.split())
    return text if len(text) <= SHOWN else text[:SHOWN - 1] + "…"


def commit(subject: str, by: str, body, edits: Optional[Callable] = None, authors: Optional[Callable] = None) -> tuple:
    """
    One commit on the card repo's main through card_bake.land: the rewrites run on the files at main's head, and again
    on the new head when main moved meanwhile (up to COMMIT_TRIES times).

    Args:
        subject: The commit subject ("Discord: ...")
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
        return "\n\n".join(part for part in (subject, "\n".join(lines), f"{EDIT_BY}: {by}") if part)
    share = card_bake.Share(None, {}, {}, rewrite("edits", edits),
                            authors=rewrite("authors", authors) if authors is not None else None)
    sha = card_bake.land(CARD_REPO, [share], message, False)
    return sha, (written if sha else {})


def _link(sha: str) -> str:
    return f"<https://github.com/{CARD_REPO}/commit/{sha}>"


def refresh(names, edits_text: Optional[str] = None, authors_text: Optional[str] = None) -> list:
    """
    The committed files applied in memory, as the next build applies them: the authors file and Card Edits.json on these
    cards, then their duplicates copied again (card_data._sync_duplicates), then the query tables rebuilt.

    Args:
        names: Card names (None: every card)
        edits_text: Card Edits.json as committed (None: unchanged)
        authors_text: Card Authors.txt as committed (None: unchanged)

    Returns:
        The names whose card changed
    """
    from commands.psd_analyzer import stats_db
    from commands.query_card import card_repo
    edits = CardEdits.parse(edits_text) if edits_text is not None else None
    authors = CardAuthors.parse(authors_text) if authors_text is not None else None
    stats = stats_db.stats
    wanted = [n for n in (list(stats) if names is None else names) if n in stats]
    changed = []
    for name in wanted:
        card = stats[name]
        before = repr(card)
        if authors is not None:
            card.author = authors.resolve(name, card.author)
        if edits is not None:
            edits.apply(name, card)
        if repr(card) != before:
            changed.append(name)
    for name, dup in list(stats.items()):           # a duplicate is its original's copy (card_data._sync_duplicates)
        original = getattr(dup, "duplicate_of", None)
        if original in changed and not getattr(stats[original], "duplicate_of", None):
            copy = deepcopy(stats[original])
            copy.name, copy.duplicate_of, copy.parsed_values, copy.problems, copy.alt_names = name, original, None, [], None
            copy.display_name = dup.display_name
            if edits is not None:
                edits.apply(name, copy)
            stats[name] = copy
    if edits is not None:
        card_repo._edits, card_repo._edit_names = edits, (blob_sha(edits_text.encode("utf-8")), edits.names())
    if changed:
        try:
            card_repo.prep_dataframes()
        except Exception:                       # noqa: BLE001 - the tables catch up on the next /update_stats
            print(f"Query tables not rebuilt:\n{format_exc()}")
    return changed


def _failed(interaction, what: str, error: Exception) -> None:
    from commands.utils import say
    print(f"{what} failed:\n{format_exc()}")
    say(interaction, f"{what} not committed: {type(error).__name__}: {error}")


# ========================
# /update_metadata
# ========================
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
        ability, stars, types, hp, defense, attack, speed: PSD fields (pending until the bake prints them; stars and
            types it can't yet)
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
    if "types" in given:
        given["types"] = [t.strip().lower() for t in given["types"].split(",") if t.strip()]
    author = author.strip() if author is not None else None
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
    bad = [f for f, v in given.items() if not CardEdits.valid(f, v) or v == ""]
    if bad or author == "":
        say(interaction, f"Not committed: {', '.join(bad) or 'author'} must be "
                         + ("a whole number, 0 or more, or some text." if bad else "a name."))
        return
    by, at = editor(interaction), now()
    changed = list(given) + (["author"] if author is not None else [])
    body = [f"- {f}: {_short(v)}" for f, v in given.items()] + ([f"- author: {author}"] if author is not None else [])
    try:
        sha, written = commit(f"{SUBJECT} /update_metadata {name} ({', '.join(changed)})", by, body,
                              (lambda text: edits_file(text, [{"key": name, "set": given, "unset": []}], at, by)) if given else None,
                              (lambda text: authors_file(text, [{"key": name, "author": author}], {})) if author is not None else None)
    except Exception as error:                  # noqa: BLE001 - one line in Discord, the traceback on stdout
        _failed(interaction, f"/update_metadata {name}", error)
        return
    if not sha:
        say(interaction, f"**{name}**: the card repo already has those values; nothing committed.")
        return
    refresh([name], written.get("edits"), written.get("authors"))
    psd = [f for f in given if CardEdits.KIND.get(f) == "psd" and f not in NOT_BAKED]
    kept = [f for f in given if f in NOT_BAKED]
    lines = [f"  • {f}: {_short(v)}" for f, v in given.items()] + ([f"  • author: {author}"] if author is not None else [])
    say(interaction, f"Updated **{name}** in the card repo ({_link(sha)}):\n" + "\n".join(lines)
        + (f"\n{', '.join(psd)}: pending on the card image until the next card bake prints it." if psd else "")
        + (f"\n{', '.join(kept)}: the card bake can't print it yet; the edit holds until the PSD is fixed." if kept else "")
        + ("" if card is not None else "\nNo data for this card yet (its PSD isn't parsed): the website shows the edit."))


# ========================
# /mass_replace
# ========================
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
    old, new = old_value.strip(), new_value.strip()
    if not old or not new:
        say(interaction, "Not committed: old_value and new_value can't be empty.")
        return
    if field == "author" and not any((c.author or "").strip().lower() == old.lower() for c in stats_db.stats.values()):
        say(interaction, f"No card has author '{old}': nothing to commit.")
        return
    by, at = editor(interaction), now()
    subject = f"{SUBJECT} /mass_replace {field} '{_short(old)}' -> '{_short(new)}'"
    try:
        if field == "author":
            seen: dict = {}

            def rewrite(text):
                seen["text"], seen["moved"], seen["people"] = replace_author(text, old, new)
                return seen["text"]
            sha, written = commit(subject, by, lambda: [f"- {n}" for n in seen["moved"]]
                                  + [f"- [people] {g} = {v}" for g, v in seen["people"].items()], authors=rewrite)
            names = None
        else:
            target = MASS[field]
            names = sorted(n for n, c in stats_db.stats.items() if not getattr(c, "duplicate_of", None)
                           and isinstance(getattr(c, field, None), str) and getattr(c, field).strip().lower() == old.lower())
            if not names:
                say(interaction, f"No card has {field} '{old}': nothing to commit.")
                return
            sha, written = commit(f"{subject} ({len(names)} cards)", by, [f"- {n}" for n in names],
                                  lambda text: edits_file(text, [{"key": n, "set": {target: new}, "unset": []} for n in names], at, by))
    except Exception as error:                  # noqa: BLE001
        _failed(interaction, f"/mass_replace {field}", error)
        return
    if not sha:
        say(interaction, f"Nothing to replace: no card shows {field} '{old}' in the card repo's files.")
        return
    changed = refresh(names, written.get("edits"), written.get("authors"))
    if field == "author":
        detail = (f"{len(seen['moved'])} [cards] lines moved to {new}; [people] "
                  + ", ".join(f"{git} = {shown}" for git, shown in seen["people"].items()))
    else:
        detail = f"{len(names)} card entries in Card Edits.json"
    say(interaction, f"Replaced {field}='{old}' with '{new}' ({_link(sha)}): {detail}; {len(changed)} cards changed now.")


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
    """Card Edits.json ops putting alias on card's entry alone (card None: on no entry). [] when already so."""
    cards = (loads(text) if text else {}).get("cards") or {}
    ops = []
    for key, fields in cards.items():
        if not isinstance(fields, dict):
            continue
        current = list(fields.get("aliases") or [])
        mine = card is not None and CardAuthors.key(key) == CardAuthors.key(card)
        kept = [a for a in current if alias_key(a) != alias_key(alias)] + ([alias] if mine else [])
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
                                  lambda text: edits_file(text, _alias_ops(text, alias, None), at, by))
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
    try:
        sha, written = commit(f"{SUBJECT} /alias {alias} -> {card}", by, [],
                              lambda text: edits_file(text, _alias_ops(text, alias, card), at, by))
    except Exception as error:                  # noqa: BLE001
        _failed(interaction, f"/alias {alias}", error)
        return
    if not sha:
        say(interaction, f"'{alias}' already finds {card}.")
        return
    card_repo.git_files[f"{norm}.png"] = target
    refresh([], written.get("edits"))
    say(interaction, f"Created alias: {alias} -> {card} ({_link(sha)})")


# ========================
# The alias migration (section 14 step 5)
# ========================
def migrate_aliases(aliases: dict, text: str, pngs: list, at: str) -> tuple:
    """
    aliases.pkl into Card Edits.json: each alias on the entry of the card its PNG belongs to, resolved as the bot does
    (the PNG's file name, else the first whose name ends so, in tree order), with spaces for underscores; plus
    ALIASES_MOVED, so the bot and the build stop reading the pickle. An alias whose PNG is gone is left out and reported;
    one that is its card's own name stays (a duplicate is named by it). New entries get _by AutoCroissant; an entry
    already there keeps its _by and _at.

    Args:
        aliases: aliases.pkl (alias -> PNG file name)
        text: Card Edits.json
        pngs: The card repo's PNG paths, in tree order
        at: The _at stamp

    Returns:
        (the new text, report lines)
    """
    names: dict = {}
    for path in pngs:
        names.setdefault(basename(path).lower(), path)
    wanted: dict = {}
    report = []
    for alias, target in sorted(aliases.items()):
        target = str(target).lower()
        path = names.get(target) or next((p for f, p in names.items() if f.endswith(target)), None)
        spelled = " ".join(str(alias).replace("_", " ").split())
        if path is None:
            report.append(f"- {alias} -> {target}: no such PNG, left out")
        else:
            wanted.setdefault(card_name(path), []).append(spelled)
            report.append(f"- {alias} -> {card_name(path)} ({path})")
    ops = []
    for card, spelled in wanted.items():
        current = list(entry_of(text, card)[1].get("aliases") or [])
        merged = current + [a for i, a in enumerate(spelled) if alias_key(a) not in {alias_key(c) for c in current + spelled[:i]}]
        if merged != current:
            ops.append({"key": card, "set": {"aliases": merged}, "unset": []})
    data = loads(edits_file(text, ops, at, BOT))
    before = (loads(text).get("cards") or {}) if text else {}
    for key, fields in data["cards"].items():   # an entry that was there keeps its editor's stamps (their bake credit)
        if isinstance(before.get(key), dict) and before[key].get("_by"):
            fields.update({k: before[key][k] for k in ("_by", "_at") if k in before[key]})
    data[ALIASES_MOVED] = MOVED_NOTE.format(at=at[:10])
    report.insert(0, f"{len(aliases)} aliases: {sum(len(v) for v in wanted.values())} moved onto {len(wanted)} cards, "
                     f"{len(aliases) - sum(len(v) for v in wanted.values())} left out")
    return dump_edits(data), report


def _git(clone: str, *args: str) -> str:
    return run_process(["git", "-C", clone, *args], capture_output=True, check=True, text=True, encoding="utf-8").stdout


def main(args: Optional[list] = None) -> int:
    """The command line: migrate-aliases (writes the migrated Card Edits.json; commits nothing)."""
    parser = ArgumentParser(prog="python -m commands.discord_edits", description="Discord's card edits (the bot's side).")
    sub = parser.add_subparsers(dest="command", required=True)
    migrate = sub.add_parser("migrate-aliases", help="move aliases.pkl into Card Edits.json (written to --out, never committed)")
    migrate.add_argument("--clone", required=True, help="a clone of the card repo (the tree and Card Edits.json at --ref)")
    migrate.add_argument("--ref", default="HEAD", help="the card repo commit to read (default HEAD)")
    migrate.add_argument("--aliases", default="aliases.pkl", help="aliases.pkl (default ./aliases.pkl)")
    migrate.add_argument("--out", default="-", help="where the new Card Edits.json goes (default stdout)")
    options = parser.parse_args(args)
    from commands.card_db import load_pickle   # the allow-list unpickler: a dict of strings needs no globals
    aliases = load_pickle(options.aliases)
    tree = [p for p in _git(options.clone, "ls-tree", "-r", "-z", "--name-only", options.ref).split("\0")
            if p.lower().endswith(".png")]
    text, report = migrate_aliases(aliases, _git(options.clone, "show", f"{options.ref}:{EDITS_PATHS[0]}"), tree, now())
    if options.out == "-":
        print(text, end="")
    else:
        with open(options.out, "w", encoding="utf-8", newline="") as fh:
            fh.write(text)
    print("\n".join(report), file=stderr)
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
    'migrate_aliases',
    'refresh',
    'replace_author',
]
