"""
Updater for the Tabletop Simulator save of the card table (TTS/Impossibility Simulator.json in the card repo).

    python -m commands.tts_save <save.json>        (from the bot's root; TTSCardMaker's tts-save.yml Action runs this)
    update_save(path) -> report text
    update_repo_save() -> (report text, commit sha or None)    (the bot: the save on the card repo's main)

Pins every repo image link to the commit that last changed the file, follows renames, syncs complete decks with
their rules (TTS/decks.json: missing repo cards added, cards outside the rule removed; a rule matching nothing leaves
its deck alone), sets card Nicknames to the
card browser's names, clears role tags, regenerates the Lua resizeTargets table. When something changed it writes
<stem>.<time>.bak.json and <stem>.report.md next to the save. The shas in the links are the only state.
The bot updates a temp copy of main's save and commits it with git_commit (the GitHub API; when the token can't write:
git objects in the history clone, pushed over SSH); it never writes a checkout.
"""
from copy import deepcopy
from datetime import datetime
from json import dumps, load, loads
from os import remove, replace
from os.path import basename, dirname, exists, expanduser, join, splitext
from re import DOTALL, IGNORECASE, MULTILINE, compile as regex, escape, sub
from secrets import token_hex
from subprocess import CalledProcessError
from sys import argv, exit
from tempfile import TemporaryDirectory
from threading import Lock
from time import sleep
from typing import Callable, Optional
from urllib.parse import quote, unquote

from requests import get

import commands.git_commit as git_commit
import commands.repo_history as repo_history
from commands.card_edits import CardAuthors
from commands.card_models import card_name as _stem
from commands.git_commit import COMMIT_TRIES, commit_files, commit_info, git as _git, plural as _plural, ref_head
from commands.repo_history import pinned_ref, refresh
from global_config import (CARD_BRANCH, CARD_REPO, EDITS_PATHS, HISTORY_CLONE, LOCAL_DIR_LOC, RAW_BASE, RULES_PATH,
                           SITE_URL, TTS_SAVE_PATH)

# ========================
# Configuration
# ========================
RAW = f"{RAW_BASE}{CARD_REPO}/"
MAIN_REFS = (CARD_BRANCH, f"refs/heads/{CARD_BRANCH}")
LINK = regex(escape(RAW) + f"({escape(MAIN_REFS[1])}|{escape(MAIN_REFS[0])}|" + r"[0-9a-f]{7,40})/([^\s\"\\<>]+)")
URL_SAFE = "/'!()*-._~"                     # left as is in the save's links (GitHub raw style; spaces are %20)
SITE_CARDS = "site/dist/data/cards.json"     # local site build: fallback when the website is unreachable
CARD_NAMES = ("Card", "CardCustom")
HAND_NAMED = ("Auxiliary/Rulebook/", "Auxiliary/Markers/")   # page and marker cards keep their Nicknames
LARGE_CARDS = "Auxiliary/Creatures/"         # the Lua resizeTargets table lists these by Nickname
ROLE_TAG = regex(r"(turbo-)?(damage|crippler|cripler|support|stacker|jack|summoner|berserker|consumer|tank|devourer|glass)",
                 IGNORECASE)
LUA_TABLE = regex(r"(resizeTargets\s*=\s*\{\n)(.*?)(\n\})", DOTALL)
LUA_ROW = regex(r'^(\s*)\[\s*"((?:[^"\\]|\\.)*)"\s*\]\s*=\s*(.*?)\s*,?\s*$', MULTILINE)
JSON_ESCAPED = {"\u0085": "\\u0085", "\u2028": "\\u2028", "\u2029": "\\u2029"}   # TTS (Newtonsoft) escapes these
CARD_EXTRAS = ("States", "AttachedDecals", "ChildObjects", "Tags")       # not copied onto new cards
CARD_SCRIPTS = ("LuaScript", "LuaScriptState", "XmlUI", "GMNotes")
KIND_LABEL = {"creatures": "creatures", "items": "items", "fields": "fields", "nme": "N.M.E", "minions": "minions",
              "auxiliary_items": "auxiliary items", "status_effects": "status effects", "morgue": "The Morgue"}
HTTP_TIMEOUT = 30
LAG_PAUSE = 10                                      # s before retrying a history clone behind main
REPORT_FILE = "tts_save_report.md"

# ========================
# Global State
# ========================
_lock = Lock()                                      # one run at a time
_objects: dict = {}                                 # git object sha -> tree entries / blob bytes (immutable)
_used: set = set()                                  # shas read this run; older ones are dropped
_site: tuple = ("", None)                           # (ETag, cards) of the website's cards.json


# ========================
# Helper Functions
# ========================
def _ls_tree(where: str, commit: str, paths: list = ()) -> dict:
    """{path: blob} at commit: the whole tree, or only paths (absent ones are left out)."""
    out = _git(where, "ls-tree", "-r", "-z", commit, "--", *paths).decode("utf-8")
    files = {}
    for entry in out.split("\0"):
        meta, _, path = entry.partition("\t")
        parts = meta.split(" ")
        if len(parts) == 3 and parts[1] == "blob":
            files[path] = parts[2]
    return files


def _encode(path: str) -> str:
    return quote(path, safe=URL_SAFE)


def _norm(text: str) -> str:
    """Loose name key: case, spaces, _ - . ' don't matter (as build_site.py)."""
    return sub(r"[\s_\-.'’]+", "", (text or "").lower())


def _glob(pattern: str):
    """'*' = part of one path segment, '**' = anything."""
    out = ""
    for tok in regex(r"(\*\*|\*|\?)").split(pattern):
        out += {"**": ".*", "*": "[^/]*", "?": "[^/]"}.get(tok, escape(tok))
    return regex(out + r"\Z")


def _rule_cards(rule: dict, pngs: list) -> list:
    """Repo PNGs a deck rule covers: folder globs + include - exclude."""
    globs = [_glob(g) for g in rule.get("folders", [])]
    include, exclude = set(rule.get("include", [])), set(rule.get("exclude", []))
    return [p for p in pngs if p in include or (p not in exclude and any(g.match(p) for g in globs))]


def _rule_text(rule: dict) -> str:
    stars = "/".join(str(s) for s in rule.get("stars") or [])
    return KIND_LABEL.get(rule.get("kind"), rule.get("kind", "?")) + (f" {stars}★" if stars else "")


def _walk(objects, parents=()):
    """Every object with its chain of containers (outermost first), including States."""
    for o in objects or []:
        yield o, parents
        yield from _walk(o.get("ContainedObjects"), parents + (o,))
        for state in (o.get("States") or {}).values():
            yield from _walk([state], parents + (o,))


def _strings(value, fn):
    """Apply fn to every string holding a repo link, in place inside containers."""
    if isinstance(value, dict):
        for k, v in value.items():
            value[k] = _strings(v, fn)
    elif isinstance(value, list):
        for i, v in enumerate(value):
            value[i] = _strings(v, fn)
    elif isinstance(value, str) and RAW in value:
        return fn(value)
    return value


def _entry(card: dict):
    """The card's own CustomDeck entry: under CardID // 100, else the first one (stale key)."""
    cd = card.get("CustomDeck") or {}
    if not cd:
        return None
    return cd.get(str(card.get("CardID", 0) // 100)) or next(iter(cd.values()))


def _face_path(card: dict):
    """Repo path of the card's face, or None."""
    entry = _entry(card)
    m = LINK.fullmatch((entry or {}).get("FaceURL") or "")
    return unquote(m.group(2)) if m else None


def _lua_key(name: str) -> str:
    return '["' + name.replace("\\", "\\\\").replace('"', '\\"') + '"]'


def _dump(value, indent: str = "") -> str:
    """JSON the way TTS writes it: 2-space indent, ": ", UTF-8 as is, floats like 1.0 and 1.00179122E-05."""
    if isinstance(value, dict):
        if not value:
            return "{}"
        inner = indent + "  "
        return "{\n" + ",\n".join(f"{inner}{dumps(k, ensure_ascii=False)}: {_dump(v, inner)}" for k, v in value.items()) \
            + f"\n{indent}}}"
    if isinstance(value, list):
        if not value:
            return "[]"
        inner = indent + "  "
        return "[\n" + ",\n".join(f"{inner}{_dump(v, inner)}" for v in value) + f"\n{indent}]"
    if isinstance(value, str):
        return dumps(value, ensure_ascii=False).translate(str.maketrans(JSON_ESCAPED))
    if value is True or value is False or value is None:
        return "true" if value is True else "false" if value is False else "null"
    if isinstance(value, float):
        return repr(value).replace("e", "E")
    return str(value)


class _Repo:
    """The card repo: current tree (path -> blob) and pinned shas from the history clone, else the checkout's tree."""

    def __init__(self, errors: list):
        self.clone = expanduser(HISTORY_CLONE)
        self.checkout = expanduser(LOCAL_DIR_LOC)
        fresh = refresh()                                   # creates the clone; an old clone still pins its own main
        self.history = fresh or exists(self.clone)
        if self.history and not fresh:
            errors.append("History clone not refreshed (offline?): links pinned against its last fetch.")
        self.tree: dict = {}
        self.head = ""
        self.source = ""
        try:
            if exists(self.clone):
                self.head = _git(self.clone, "rev-parse", CARD_BRANCH).decode().strip()
                self.tree, self.source = _ls_tree(self.clone, CARD_BRANCH), "the history clone"
            elif exists(join(self.checkout, ".git")):
                self.head = _git(self.checkout, "rev-parse", "HEAD").decode().strip()
                self.tree, self.source = _ls_tree(self.checkout, "HEAD"), "the checkout"
                errors.append("No history clone: links and decks stay as they are (a rename would look like a new card); "
                              "the repo tree comes from the checkout.")
            else:
                errors.append("No history clone and no checkout: the repo is unknown, only names were refreshed.")
        except (CalledProcessError, OSError) as e:
            errors.append(f"git failed: {e}")
        self.pngs = sorted(p for p in self.tree if p.lower().endswith(".png"))
        self.by_blob: dict = {}
        for path, blob in self.tree.items():
            self.by_blob.setdefault(blob, []).append(path)
        self.by_lower = {p.lower(): p for p in self.tree}
        self._at: dict = {}                                 # (sha, path) -> blob there, or None

    def sha(self, path: str):
        """Commit that last changed path, or None when history doesn't know it."""
        blob = self.tree.get(path)
        return pinned_ref(CARD_REPO, path, blob) if blob else None

    def url(self, path: str) -> str:
        return f"{RAW}{self.sha(path) or CARD_BRANCH}/{_encode(path)}"

    def blobs_at(self, sha: str, paths: list) -> dict:
        """{path: blob} at an older commit (None: not there, or the commit is unknown)."""
        missing = [p for p in paths if (sha, p) not in self._at]
        if missing and exists(self.clone):
            try:
                found = _ls_tree(self.clone, sha, missing)
            except CalledProcessError:
                found = {}
            for p in missing:
                self._at[(sha, p)] = found.get(p)
        return {p: self._at.get((sha, p)) for p in paths}

    def newest(self, shas: list) -> str:
        """The latest of several commits (topological order), given as they were written."""
        if len(shas) < 2:
            return shas[0]
        try:
            full = _git(self.clone, "rev-list", "--topo-order", "-n", "1", *shas).decode().strip()
        except CalledProcessError:
            full = ""
        return next((s for s in shas if full.startswith(s)), sorted(shas)[0])

    def last_seen(self, paths: list) -> dict:
        """Paths gone from main -> (newest commit that still had them, blob then), walking main's trees."""
        found = {}
        if not paths or not exists(self.clone):
            return found
        try:
            left = set(paths)
            for commit in _git(self.clone, "rev-list", "--topo-order", CARD_BRANCH).decode().split():
                if not left:
                    break
                for path, blob in _ls_tree(self.clone, commit, sorted(left)).items():
                    found[path] = (commit, blob)
                    left.discard(path)
        except CalledProcessError:
            pass
        return found


class _Names:
    """Website names for card images: Card Edits.json > the card browser's cards.json > PSD stem > PNG stem."""

    def __init__(self, repo: _Repo, errors: list, edits: Optional[tuple] = None):
        """edits: (main's Card Edits.json text or None,) for bot runs: the checkout is then not read."""
        global _site
        local = edits is None
        edits = edits[0] if edits else None
        self.repo = repo
        self.site: dict = {}
        self.edits: dict = {}
        self.source = "file stems"
        try:
            etag, known = _site
            reply = get(f"{SITE_URL}/data/cards.json", timeout=HTTP_TIMEOUT,
                        headers={"If-None-Match": etag} if etag and known is not None else {})
            if reply.status_code == 304 and known is not None:     # unchanged since the last run
                cards = known
            else:
                reply.raise_for_status()
                cards = reply.json()["cards"]
                _site = (reply.headers.get("ETag", ""), cards)
            self.source = "the card browser"
        except Exception as e:
            cards = []
            try:
                if local:
                    with open(join(repo.checkout, SITE_CARDS), encoding="utf-8") as f:
                        cards, self.source = load(f)["cards"], "the local site build"
            except (OSError, ValueError, KeyError):
                pass
            errors.append(f"Card browser data not fetched ({e}): names from {self.source}"
                          + ("." if cards else "; Nicknames left as they are, new cards named by file stem."))
        self.site = {c["png"]: c["name"] for c in cards if isinstance(c, dict) and c.get("png") and c.get("name")}
        try:
            if local:
                path = next((p for p in EDITS_PATHS if exists(join(repo.checkout, p))), None)
                if path is not None:
                    with open(join(repo.checkout, path), encoding="utf-8") as f:
                        edits = f.read()
            if edits is not None:
                self.edits = {CardAuthors.key(k): v for k, v in (loads(edits).get("cards") or {}).items() if isinstance(v, dict)}
        except (OSError, ValueError, AttributeError):
            pass

    def _stem(self, path: str) -> str:
        psd = self.repo.by_lower.get(splitext(path)[0].lower() + ".psd")
        return _stem(psd or path)                           # a parsed card is keyed by its PSD stem

    def known(self, path: str):
        """The website's name for the image (Card Edits.json first), or None when it has none."""
        found = self.edits.get(CardAuthors.key(self._stem(path)))
        if found and isinstance(found.get("name"), str) and found["name"].strip():
            return found["name"]
        return self.site.get(path)

    def of(self, path: str) -> str:
        """known(), else the file stem (offline, or a card the website doesn't list)."""
        return self.known(path) or self._stem(path)


class _Report:
    """What changed, for the markdown report."""

    def __init__(self):
        self.pinned: dict = {}          # path -> (old refs, new sha, links)
        self.repinned: dict = {}
        self.unpinned: set = set()
        self.renamed: dict = {}         # old path -> (new path, links)
        self.gone: dict = {}            # path -> (refs, links, note)
        self.added: list = []           # (deck label, [(name, path, copies)])
        self.removed: list = []         # (deck label, [(name, path, copies)]): cards outside the deck's rule
        self.names_real: dict = {}      # (path, old, new) -> objects
        self.names_case: dict = {}
        self.unnamed: dict = {}         # (path, Nickname) -> objects: not on the website, kept
        self.cleared = 0
        self.kept: dict = {}            # description -> cards
        self.stale: list = []
        self.lua_added: list = []
        self.lua_removed: list = []
        self.left: list = []
        self.errors: list = []
        self.first_run = False
        self.changed = False
        self.head = ""                  # main's commit the repo data came from


# ========================
# Update Steps
# ========================
def _pin_links(save: dict, repo: _Repo, report: _Report) -> None:
    """Every repo link -> the commit that last changed its file; renamed files follow; gone files are reported."""
    links: dict = {}                                        # (ref, encoded path) -> occurrences

    def collect(text):
        for m in LINK.finditer(text):
            links[m.groups()] = links.get(m.groups(), 0) + 1
        return text
    _strings(save, collect)
    if not repo.tree or not repo.history:
        return
    report.first_run = all(ref in MAIN_REFS for ref, _ in links)
    check: dict = {}                                        # sha -> paths whose pin differs from the current one
    lost: dict = {}                                         # gone path -> refs
    for ref, enc in links:
        path = unquote(enc)
        if path not in repo.tree:
            lost.setdefault(path, set()).add(ref)
        elif ref not in MAIN_REFS and ref != repo.sha(path):
            check.setdefault(ref, set()).add(path)
    old = {}                                                # (sha, path) -> blob at that sha
    for sha, paths in check.items():
        old.update({(sha, p): b for p, b in repo.blobs_at(sha, sorted(paths)).items()})
    last = repo.last_seen(sorted(lost))                     # gone path -> (last commit on main that had it, blob then)
    for path, refs in lost.items():                         # never on main: the newest pinned link that had it
        had = {sha: repo.blobs_at(sha, [path])[path] for sha in sorted(refs) if sha not in MAIN_REFS and path not in last}
        had = {sha: blob for sha, blob in had.items() if blob}
        if had:
            sha = repo.newest(list(had))
            last[path] = (sha, had[sha])
    rename = {}
    for path, (_, blob) in last.items():
        targets = repo.by_blob.get(blob, [])
        if len(targets) == 1:
            rename[path] = targets[0]

    changes: dict = {}                                      # (ref, enc) -> (new ref, new enc)
    for (ref, enc), n in links.items():
        path = unquote(enc)
        if path in rename:
            new = rename[path]
            changes[(ref, enc)] = (repo.sha(new) or CARD_BRANCH, _encode(new))
            if repo.sha(new) is None:
                report.unpinned.add(new)
            report.renamed[path] = (new, report.renamed.get(path, (new, 0))[1] + n)
        elif path not in repo.tree:
            commit = last.get(path, ("", ""))[0]
            note = ""
            if ref in MAIN_REFS and commit:
                changes[(ref, enc)] = (commit, enc)
                note = "link pinned to the last commit that had it"
            refs, count, _ = report.gone.get(path, (set(), 0, ""))
            report.gone[path] = (refs | {ref}, count + n, note)
        else:
            target = repo.sha(path)
            if target is None:
                report.unpinned.add(path)
            elif ref in MAIN_REFS:
                changes[(ref, enc)] = (target, enc)
                refs, _, count = report.pinned.get(path, (set(), target, 0))
                report.pinned[path] = (refs | {ref}, target, count + n)
            elif ref != target and old.get((ref, path)) != repo.tree[path]:
                changes[(ref, enc)] = (target, enc)
                refs, _, count = report.repinned.get(path, (set(), target, 0))
                report.repinned[path] = (refs | {ref}, target, count + n)

    def apply(text):
        return LINK.sub(lambda m: f"{RAW}{changes[m.groups()][0]}/{changes[m.groups()][1]}" if m.groups() in changes
                        else m.group(0), text)
    if changes:
        _strings(save, apply)


def _sync_decks(save: dict, rules: dict, repo: _Repo, names: _Names, report: _Report) -> None:
    """Complete decks follow their rule: missing repo cards join (at the rule's usual count), cards outside it leave."""
    everything = list(_walk(save.get("ObjectStates")))
    guids = {o.get("GUID") for o, _ in everything}
    top = 0
    for o, _ in everything:
        top = max([top, o.get("CardID", 0) // 100 if isinstance(o.get("CardID"), int) else 0]
                  + [int(k) for k in (o.get("CustomDeck") or {}) if k.isdigit()])
    ids: dict = {}                                          # new image path -> its new CustomDeck id
    for deck, parents in everything:
        if deck.get("Name") != "Deck":
            continue
        guid = deck.get("GUID", "")
        label = f"`{guid}` {deck.get('Nickname') or '(unnamed)'}"
        try:                                                # a malformed entry skips only its own deck
            entry = rules.get("decks", {}).get(guid) or {}
            rule = entry.get("rule")
            label += f", {entry['where']}" if entry.get("where") else ""
            if not rule:
                report.left.append(f"{label}: no rule in {RULES_PATH}")
                continue
            label += f" ({_rule_text(rule)})"
            cards = deck.get("ContainedObjects") or []
            if rule.get("partial"):
                report.left.append(f"{label}: hand-picked, {len(cards)} cards kept as they are")
                continue
            if not cards or not rule.get("count"):
                report.left.append(f"{label}: empty, left empty")
                continue
            wanted = _rule_cards(rule, repo.pngs)
            count = int(rule["count"])
            counts = {p: int(c) for p, c in (entry.get("odd_counts") or {}).items()}
            back = entry.get("back")
            back = back if back in repo.tree else None
        except (AttributeError, KeyError, TypeError, ValueError) as e:
            report.errors.append(f"{label}: bad entry in {RULES_PATH} ({type(e).__name__}: {e}); deck left alone.")
            continue
        if not wanted:                                      # a typo or a renamed folder: never empty the deck
            report.errors.append(f"{label}: its rule matches no repo image; deck left alone.")
            continue
        present = {_face_path(c) for c in cards}
        missing = [p for p in wanted if p not in present]
        outside = [c for c in cards if (_face_path(c) or "").lower().endswith(".png")    # rules only cover PNGs
                   and _face_path(c) in repo.tree and _face_path(c) not in wanted]
        if not missing and not outside:
            continue
        sibling = cards[len(cards) - 1]
        base = _entry(sibling) or {}
        back_url = repo.url(back) if back else base.get("BackURL", "")
        added = []
        for path in missing:
            if path not in ids:
                top += 1
                ids[path] = top
            if repo.sha(path) is None:
                report.unpinned.add(path)
            face = deepcopy(base)
            face.update(FaceURL=repo.url(path), BackURL=back_url, NumWidth=1, NumHeight=1)
            name = names.of(path)
            copies = counts.get(path, count)
            for _ in range(copies):
                card = deepcopy(sibling)
                while True:
                    card["GUID"] = token_hex(3)
                    if card["GUID"] not in guids:
                        break
                guids.add(card["GUID"])
                card.update(Nickname=name, Description="", CardID=ids[path] * 100, CustomDeck={str(ids[path]): deepcopy(face)})
                card.update({k: "" for k in CARD_SCRIPTS if k in card})
                for k in CARD_EXTRAS:
                    card.pop(k, None)
                cards.append(card)
            deck.setdefault("CustomDeck", {})[str(ids[path])] = deepcopy(face)
            added.append((name, path, copies))
        if added:
            report.added.append((label, added))
        if outside:
            gone = {id(c) for c in outside}
            cards = [c for c in cards if id(c) not in gone]
            left: dict = {}
            for c in outside:
                key = (c.get("Nickname") or "", _face_path(c))
                left[key] = left.get(key, 0) + 1
            report.removed.append((label, [(n, p, c) for (n, p), c in left.items()]))
        used = {str(c.get("CardID", 0) // 100) for c in cards}       # keys no card uses go
        deck["CustomDeck"] = {k: v for k, v in (deck.get("CustomDeck") or {}).items() if k in used}
        deck["ContainedObjects"] = cards
        deck["DeckIDs"] = [c.get("CardID", 0) for c in cards]


def _fix_keys(save: dict, report: _Report) -> None:
    """Each card's own CustomDeck map keyed by CardID // 100 (the deck's entry, the one TTS shows, wins)."""
    for card, parents in _walk(save.get("ObjectStates")):
        cd = card.get("CustomDeck") or {}
        key = str(card.get("CardID", 0) // 100)
        if card.get("Name") not in CARD_NAMES or not cd or key in cd:
            continue
        deck = parents[len(parents) - 1] if parents and parents[len(parents) - 1].get("Name") == "Deck" else {}
        own, shown = next(iter(cd.values())), (deck.get("CustomDeck") or {}).get(key)
        card["CustomDeck"] = {key: deepcopy(shown or own)}
        note = "" if not shown or shown.get("FaceURL") == own.get("FaceURL") else "; own face differed, the deck's kept"
        report.stale.append(f"`{card.get('GUID')}` {card.get('Nickname')}: {', '.join(cd)} -> {key}{note}")


def _fix_cards(save: dict, repo: _Repo, names: _Names, report: _Report) -> None:
    """Nickname = website name, role tags cleared."""
    for card, _ in _walk(save.get("ObjectStates")):
        if card.get("Name") not in CARD_NAMES:
            continue
        path = _face_path(card)
        if not path or path.startswith(HAND_NAMED):
            continue
        description = card.get("Description") or ""
        if description:
            if all(ROLE_TAG.fullmatch(w) for w in description.split()):
                card["Description"] = ""
                report.cleared += 1
            else:
                report.kept[description] = report.kept.get(description, 0) + 1
        if path not in repo.tree or not names.site:
            continue                                        # gone image, or no website data: name left alone
        old = card.get("Nickname") or ""
        name = names.known(path)
        if name is None:
            report.unnamed[(path, old)] = report.unnamed.get((path, old), 0) + 1
        elif name != old:
            card["Nickname"] = name
            group = report.names_case if _norm(old) == _norm(name) else report.names_real
            group[(path, old, name)] = group.get((path, old, name), 0) + 1


def _resize_targets(save: dict, repo: _Repo, names: _Names, report: _Report) -> None:
    """The global script's resizeTargets table lists the large creatures by their Nicknames in the save."""
    lua = save.get("LuaScript") or ""
    m = LUA_TABLE.search(lua)
    if not m:
        report.errors.append("LuaScript: no resizeTargets table found.")
        return
    rows = LUA_ROW.findall(m.group(2))
    lines = [line for line in m.group(2).split("\n") if line.strip()]
    if len(rows) != len(lines) or any("--" in line for line in lines):
        report.errors.append("LuaScript: resizeTargets has rows the updater doesn't understand; table left alone.")
        return
    old = {key.replace('\\"', '"').replace("\\\\", "\\"): value for _, key, value in rows}
    nicks: dict = {}                                        # large card path -> its Nicknames in the save
    for card, _ in _walk(save.get("ObjectStates")):
        path = _face_path(card) if card.get("Name") in CARD_NAMES else None
        if path and path.startswith(LARGE_CARDS) and card.get("Nickname"):
            nicks.setdefault(path, set()).add(card["Nickname"])
    wanted = {n for ns in nicks.values() for n in ns}        # cards in the save, even when their image is gone
    wanted |= {names.of(p) for p in repo.pngs if p.startswith(LARGE_CARDS) and p not in nicks}
    wanted = sorted(wanted, key=lambda k: (k.lower(), k))
    if not wanted:
        return
    indent = rows[0][0] if rows else "    "
    width = max(len(_lua_key(k)) for k in wanted)
    body = "\n".join(f"{indent}{_lua_key(k).ljust(width)} = {old.get(k, 'true')}," for k in wanted)
    if body != m.group(2):
        save["LuaScript"] = lua[:m.start(2)] + body + lua[m.end(2):]
    report.lua_added = [k for k in wanted if k not in old]
    report.lua_removed = [k for k in old if k not in wanted]


def _decks(rules):
    """The parsed rules, if their top level is an object with a "decks" object."""
    if not isinstance(rules, dict) or not isinstance(rules.get("decks", {}), dict):
        raise ValueError('no "decks" object')
    return rules


def _load_rules(repo: _Repo, errors: list, given: Optional[tuple] = None):
    """TTS/decks.json: given as (text or None, source) by the bot, else from the checkout, else GitHub (raw, main)."""
    local = join(repo.checkout, RULES_PATH)
    try:
        if given is not None:
            text, source = given
            if text is None:
                raise FileNotFoundError(f"not in {source}")
            return _decks(loads(text)), source
        if exists(local):
            with open(local, encoding="utf-8") as f:
                return _decks(load(f)), "the checkout"
        reply = get(f"{RAW}{CARD_BRANCH}/{_encode(RULES_PATH)}", timeout=HTTP_TIMEOUT)
        reply.raise_for_status()
        return _decks(reply.json()), "GitHub"
    except (OSError, ValueError) as e:
        errors.append(f"{RULES_PATH} not read ({e}): no cards added.")
        return None, "nowhere"


def _render(report: _Report, head: list) -> str:
    lines = ["# TTS save update", ""] + head
    if not report.changed:
        lines += ["", "**No changes.**"]

    def section(title, items):
        if items:
            lines.extend(["", f"## {title} ({len(items)})"] + [f"- {x}" for x in items])

    def shas(refs):
        return ", ".join(r if r in MAIN_REFS else f"`{r[:8]}`" for r in sorted(refs))
    section("Links pinned" if report.first_run else "Links re-pinned (changed files)",
            [f"`{p}`: {shas(refs)} -> `{sha[:8]}` ({n} link{'s' if n != 1 else ''})"
             for p, (refs, sha, n) in sorted({**report.pinned, **report.repinned}.items())])
    section("Links not pinned (history doesn't know the file)", [f"`{p}`" for p in sorted(report.unpinned)])
    section("Renamed images", [f"`{old}` -> `{new}` ({n} links)" for old, (new, n) in sorted(report.renamed.items())])
    section("Cards added", [f"{label}: " + ", ".join(f"{name} `{path}`" + (f" x{c}" if c != 1 else "") for name, path, c in cards)
                            for label, cards in report.added])
    section("Cards removed (outside the deck's rule)",
            [f"{label}: " + ", ".join(f"{name} `{path}`" + (f" x{c}" if c != 1 else "") for name, path, c in cards)
             for label, cards in report.removed])
    section("Nicknames changed", [f"`{p}`: '{old}' -> '{new}' ({n} card{'s' if n != 1 else ''})"
                                  for (p, old, new), n in sorted(report.names_real.items())])
    section("Nicknames changed (case/spacing only)", [f"`{p}`: '{old}' -> '{new}' ({n} card{'s' if n != 1 else ''})"
                                                      for (p, old, new), n in sorted(report.names_case.items())])
    section("Nicknames kept (not on the website)", [f"`{p}`: '{old}' ({n} card{'s' if n != 1 else ''})"
                                                    for (p, old), n in sorted(report.unnamed.items())])
    if report.cleared:
        lines += ["", f"## Descriptions: {report.cleared} role tags cleared"]
        lines += [f"- kept on {n} card{'s' if n != 1 else ''}: {dumps(d)[:80]}" for d, n in sorted(report.kept.items(), key=lambda kv: -kv[1])]
    section("CustomDeck keys fixed", report.stale)
    section("resizeTargets", [f"added `{k}`" for k in report.lua_added] + [f"removed `{k}`" for k in report.lua_removed])
    section("Images gone from the repo (report only)",
            [f"`{p}`: {shas(refs)} ({n} links){'; ' + note if note else ''}" for p, (refs, n, note) in sorted(report.gone.items())])
    section("Decks left alone", report.left)
    section("Errors", report.errors)
    return "\n".join(lines) + "\n"


def _update(save_path: str, rules: Optional[tuple] = None, extras: bool = True, edits: Optional[tuple] = None) -> tuple:
    """update_save, also returning the _Report. rules: see _load_rules; extras: write the backup and report files;
    edits: see _Names."""
    with open(save_path, "rb") as f:
        raw = f.read()
    save = loads(raw.decode("utf-8"))
    report = _Report()
    repo = _Repo(report.errors)
    report.head = repo.head
    rules, rules_from = _load_rules(repo, report.errors, rules) if repo.tree and repo.history else (None, "nowhere")
    names = _Names(repo, report.errors, edits)
    _pin_links(save, repo, report)
    if repo.tree:
        _fix_keys(save, report)
        if rules is not None:
            _sync_decks(save, rules, repo, names, report)
        _fix_cards(save, repo, names, report)
        _resize_targets(save, repo, names, report)

    out = _dump(save).encode("utf-8")
    stem = splitext(save_path)[0]
    backup = f"{stem}.{datetime.now().strftime('%Y%m%d-%H%M%S')}.bak.json"
    report.changed = out != raw
    if report.changed:
        if extras:
            with open(backup, "wb") as f:
                f.write(raw)
        try:
            with open(f"{save_path}.tmp", "wb") as f:
                f.write(out)
            replace(f"{save_path}.tmp", save_path)
        finally:
            if exists(f"{save_path}.tmp"):
                remove(f"{save_path}.tmp")
    kept = f"; backup `{basename(backup)}`" if extras else ""
    head = [f"- Save: `{basename(save_path)}` ({len(raw):,} bytes): "
            + (f"updated ({len(out):,} bytes{kept})" if report.changed else "unchanged"),
            f"- Repo: main `{repo.head[:8]}` from {repo.source or 'nowhere'} ({len(repo.tree)} files, {len(repo.pngs)} PNGs)",
            f"- Rules: `{RULES_PATH}` from {rules_from} ({len((rules or {}).get('decks', {}))} decks)",
            f"- Names: from {names.source} ({len(names.site)} cards)"]
    text = _render(report, head)
    if report.changed and extras:
        with open(f"{stem}.report.md", "w", encoding="utf-8") as f:
            f.write(text)
    return text, report


# ========================
# Bot Flow
# ========================
def _commit_message(report: _Report) -> str:
    """'TTS save: 812 links pinned, 3 cards added', from the report."""
    links = sum(n for _, _, n in report.pinned.values()) + sum(n for _, _, n in report.repinned.values())
    added = sum(len(cards) for _, cards in report.added)
    removed = sum(len(cards) for _, cards in report.removed)
    names = sum(report.names_real.values()) + sum(report.names_case.values())
    parts = [f"{_plural(links, 'link')} pinned" if links else "",
             f"{_plural(added, 'card')} added" if added else "",
             f"{_plural(removed, 'card')} removed" if removed else "",
             f"{_plural(len(report.renamed), 'rename')} followed" if report.renamed else "",
             f"{_plural(names, 'Nickname')} fixed" if names else ""]
    return "TTS save: " + (", ".join(p for p in parts if p) or "updated")


def _api(method: str, path: str, **kwargs):
    """git_commit.api on the card repo (raw: a blob's bytes; check: raise on an error status)."""
    return git_commit.api(CARD_REPO, method, path, **kwargs)


def _object(sha: str, fetch: Callable[[], object]):
    """A git object by sha, fetched once (a sha's content never changes); kept while runs keep reading it."""
    if sha not in _objects:
        _objects[sha] = fetch()
    _used.add(sha)
    return _objects[sha]


def _files_at(tree: str, paths: tuple) -> dict:
    """{path: (mode, blob sha)} of the given paths in a git tree (one small request per folder level)."""
    listed: dict = {}                                   # folder -> {name: entry}
    trees: dict = {}                                    # tree sha -> {name: entry}
    found = {}
    for path in paths:
        folder = dirname(path)
        if folder not in listed:
            sha, entries = tree, {}
            for part in [""] + (folder.split("/") if folder else []):
                if part:
                    entry = entries.get(part)
                    sha = entry["sha"] if entry and entry.get("type") == "tree" else None
                if sha is None:
                    entries = {}
                    break
                if sha not in trees:
                    trees[sha] = _object(sha, lambda: {e["path"]: e for e in _api("GET", f"git/trees/{sha}").json().get("tree", [])})
                entries = trees[sha]
            listed[folder] = entries
        entry = listed[folder].get(basename(path))
        if entry and entry.get("type") == "blob":
            found[path] = (entry["mode"], entry["sha"])
    return found


def _say(interaction, text: str) -> None:
    """A message to the interaction, or the terminal without one."""
    from commands.utils import say  # lazy: standalone use needs no Discord
    say(interaction, text)


def _summary(text: str) -> str:
    """The report's header lines plus its section titles (with their counts)."""
    lines = text.splitlines()
    first = next((i for i, line in enumerate(lines) if line.startswith("## ")), len(lines))
    head = [line for line in lines[:first] if line.strip()]
    if head and head[0].startswith("# "):
        head[0] = f"**{head[0][2:]}**"
    return "\n".join(head + [f"- **{line[3:]}**" for line in lines[first:] if line.startswith("## ")])


def _post(interaction, text: str, sha: Optional[str]) -> None:
    """Short summary with the full report attached, then the players' download link."""
    from io import BytesIO
    from discord import File
    from commands.utils import queue_file, queue_message, split_long_message  # lazy: standalone use needs no Discord
    link = f"{RAW}{sha or CARD_BRANCH}/{_encode(TTS_SAVE_PATH)}"
    if interaction is None:
        print(text)
        print(link)
        return
    chunks = split_long_message(_summary(text))
    for chunk in chunks[:len(chunks) - 1]:
        queue_message(interaction, chunk)
    queue_file(interaction, File(BytesIO(text.encode("utf-8")), filename=REPORT_FILE), content=chunks[len(chunks) - 1])
    # the save itself (5.5 MB) is never attached: players download it from the link
    queue_message(interaction, f"Updated save, committed as `{sha[:8]}`: {link}" if sha
                  else f"No changes: the save on main is current: {link}")


# ========================
# Public API Functions
# ========================
def update_save(save_path: str) -> str:
    """
    Update a TTS save in place and return the report. When it changed: backup and report beside it.

    Args:
        save_path: The save's JSON file

    Returns:
        The markdown change report (also written to <stem>.report.md when the save changed)
    """
    return _update(save_path)[0]


def _blob(sha: str) -> bytes:
    """A blob's bytes from the card repo, by sha."""
    return _object(sha, lambda: _api("GET", f"git/blobs/{sha}", raw=True).content)


def update_repo_save() -> tuple:
    """
    Update the save on the card repo's main and commit it with git_commit.commit_files (blocking; one run at a time):
    the GitHub API, or when the token can't write (401/403/404, or no token) git objects in the history clone pushed over SSH.
    Rules and Card Edits.json come from main too. Only a temp dir is written. When main moves meanwhile (or the
    history clone lags it), the update is redone on the new save, up to COMMIT_TRIES times.

    Returns:
        (markdown report, sha of the new commit, or None when nothing changed)
    """
    with _lock, TemporaryDirectory(prefix="tts_save_") as tmp:
        for sha in set(_objects) - _used:               # only the last run's objects stay cached
            del _objects[sha]
        _used.clear()
        save_path = join(tmp, basename(TTS_SAVE_PATH))
        git_commit.limited = False                      # a rate limit is one run's
        why = "main kept moving"
        for attempt in range(COMMIT_TRIES):
            head = ref_head(CARD_REPO)
            base = commit_info(CARD_REPO, head)["tree"]["sha"]
            files = _files_at(base, (TTS_SAVE_PATH, RULES_PATH) + EDITS_PATHS)
            if TTS_SAVE_PATH not in files:
                raise FileNotFoundError(f"{TTS_SAVE_PATH} is not on main")
            mode, old_blob = files[TTS_SAVE_PATH]
            with open(save_path, "wb") as f:
                f.write(_blob(old_blob))
            edits_path = next((p for p in EDITS_PATHS if p in files), None)
            rules = _blob(files[RULES_PATH][1]).decode("utf-8") if RULES_PATH in files else None
            edits = _blob(files[edits_path][1]).decode("utf-8") if edits_path is not None else None
            text, report = _update(save_path, (rules, f"main `{head[:8]}`"), extras=False, edits=(edits,))
            if head != report.head or head != repo_history._head:   # pins from main's head, even for "no changes"
                why = (f"the history clone is at `{report.head[:8] or 'nothing'}` (links map at "
                       f"`{repo_history._head[:8] or 'nothing'}`), main at `{head[:8]}`")
                if attempt < COMMIT_TRIES - 1:
                    sleep(LAG_PAUSE)
                continue
            if not report.changed:
                return text, None
            with open(save_path, "rb") as f:
                data = f.read()
            commit = commit_files(CARD_REPO, head, base, {TTS_SAVE_PATH: mode}, {TTS_SAVE_PATH: data}, _commit_message(report),
                                  ssh_on_limit=True)    # a throttled API is this run's: the bot's SSH agent pushes now
            if commit:
                return text, commit
            why = "main kept moving (the commit was not a fast forward)"
        raise RuntimeError(f"{why}: the save was not committed after {COMMIT_TRIES} tries")


def update_tts_save(interaction) -> None:
    """/update_tts_save: update the save on main, post the summary, the report file and the download link."""
    try:
        text, sha = update_repo_save()
    except Exception as e:
        _say(interaction, f"TTS save update failed: {e}")
        return
    _post(interaction, text, sha)


def main() -> None:
    if len(argv) != 2:
        print("usage: python -m commands.tts_save <save.json>")
        exit(2)
    print(update_save(argv[1]), end="")


if __name__ == "__main__":
    main()


# ========================
# Module Exports
# ========================
__all__ = [
    'update_save',
    'update_repo_save',
    'update_tts_save',
    'main',
]
