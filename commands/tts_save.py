"""
Updater for the Tabletop Simulator save of the card table (TTS/Impossibility Simulator.json in the card repo).

    python -m commands.tts_save <save.json> [--names db/stats.json]   (from the bot's root; card-sync.yml)
    update_save(path) -> report text

Pins every repo image link to the commit that last changed the file, follows renames (the bake's Renamed trailers too), syncs complete decks with
their rules (TTS/decks.json: missing repo cards added, an empty deck filled, cards outside the rule removed; a rule
matching nothing leaves its deck alone), adds new cards to hand-picked decks whose rule has add_new_since (cards the
save at that commit lacked; a card moved to other stars moves between those decks; no record: one taken out by hand
comes back unless excluded), sets card Nicknames to the card browser's names (or, with --names, the same names made
from a card data build's stats.json), regenerates the Lua resizeTargets table, and puts the rulebook's PDF
(TTS/Rulebook/Rulebook.pdf, commands.rulebook_pages) in the Rulebooks bag as a Custom_PDF object (the Photoshop pages
and their bag taken out, owner 10-30).
An existing card's Description, GMNotes, Memo, Tags and scripts are left to the team, never changed; new cards start
with them empty. When something changed it writes <stem>.<time>.bak.json, <stem>.report.md and <stem>.message.txt
(the commit subject) next to the save. The shas in the links are the only state.
"""
from argparse import ArgumentParser
from copy import deepcopy
from datetime import datetime
from json import dumps, load, loads
from os import remove, replace
from os.path import basename, exists, expanduser, join, splitext
from re import DOTALL, MULTILINE, compile as regex, escape, sub
from secrets import token_hex
from subprocess import CalledProcessError
from typing import Optional
from urllib.parse import quote, unquote

from requests import get

from commands.card_edits import CardAuthors
from commands.card_models import RENAMED, card_name as _stem, rename_trailer
from commands.git_commit import git as _git, plural as _plural
from commands.repo_history import pinned_ref, refresh
from commands.rulebook_pages import PDF_FILE, read_pages
from global_config import (CARD_BRANCH, CARD_REPO, EDITS_PATH, HISTORY_CLONE, LOCAL_DIR_LOC, RAW_BASE, RULEBOOK_PAGES, RULES_PATH,
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
HAND_NAMED = ("Auxiliary/Rulebook/", "Auxiliary/Markers/", RULEBOOK_PAGES + "/")   # page and marker cards: not website names
RULEBOOK_BAG = "Rulebooks"                   # the save's bag of rulebook pages
OLD_RULEBOOK = "Old Rulebook (Photoshop pages)"   # the bag inside it the Photoshop pages went to on 10-29 (taken out, owner 10-30)
OLD_PAGES = "Auxiliary/Rulebook/"
PDF_NAME = "Rulebook"                        # the PDF object's Nickname when made
PDF_SCALE = 3.0                              # its size when made (a 1080x1920 page); the team may resize it
CARD_FOLDERS = ("Creatures/", "Items/", "Field/", "N.M.E/", "Auxiliary/")   # build_site.CARD_FOLDERS
SKIP_PARTS = ("MDW", "Markers")                                             # build_site.SKIP_PARTS
LARGE_CARDS = "Auxiliary/Creatures/"         # the Lua resizeTargets table lists these by Nickname
LUA_TABLE = regex(r"(resizeTargets\s*=\s*\{\n)(.*?)(\n\})", DOTALL)
LUA_ROW = regex(r'^(\s*)\[\s*"((?:[^"\\]|\\.)*)"\s*\]\s*=\s*(.*?)\s*,?\s*$', MULTILINE)
STARS_FOLDER = regex(r"(?<=/)\d+ Stars(?=/)")
JSON_ESCAPED = {"\u0085": "\\u0085", "\u2028": "\\u2028", "\u2029": "\\u2029"}   # TTS (Newtonsoft) escapes these
CARD_EXTRAS = ("States", "AttachedDecals", "ChildObjects", "Tags", "Memo")      # not copied onto new cards
CARD_SCRIPTS = ("LuaScript", "LuaScriptState", "XmlUI", "GMNotes")          # emptied on new cards
TEAM_FIELDS = ("Description", "GMNotes", "Memo", "Tags", "LuaScript", "LuaScriptState", "XmlUI")   # cards: never touched
KIND_LABEL = {"creatures": "creatures", "items": "items", "fields": "fields", "nme": "N.M.E", "minions": "minions",
              "auxiliary_items": "auxiliary items", "status_effects": "status effects", "morgue": "The Morgue"}
HTTP_TIMEOUT = 30

# ========================
# Global State
# ========================
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


def _other_stars(rule: dict, path: str, stars: set) -> bool:
    """Whether rule covers path under another stars folder but not as it is (the card changed stars)."""
    globs = [_glob(g) for g in rule.get("folders", [])]
    include, exclude = set(rule.get("include", [])), set(rule.get("exclude", []))
    m = STARS_FOLDER.search(path)
    if not m or any(g.match(path) for g in globs):          # outside stars folders, or excluded on purpose
        return False
    for folder in stars - {m.group(0)}:
        other = path[:m.start()] + folder + path[m.end():]
        if other in include or (other not in exclude and any(g.match(other) for g in globs)):
            return True
    return False


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
    """Apply fn to every string holding a repo link, in place inside containers (a card's TEAM_FIELDS skipped)."""
    if isinstance(value, dict):
        card = value.get("Name") in CARD_NAMES
        for k, v in value.items():
            if not (card and k in TEAM_FIELDS):
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
        self._carried: dict = {}                            # sha -> carried()
        self._told: Optional[dict] = None                   # old path -> new path: the bake's Renamed trailers (follow())

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

    def carried(self, sha: str):
        """Today's paths of the card images the save had at sha, renames followed (as _pin_links: by blob, the PSD's
        too); None when that save can't be read. Its blob is fetched once into the history clone."""
        if sha in self._carried:
            return self._carried[sha]
        found = None
        if self.history and exists(self.clone):
            try:
                old = loads(_git(self.clone, "cat-file", "blob", f"{sha}:{TTS_SAVE_PATH}").decode("utf-8"))
                found = {_face_path(c) for c, _ in _walk(old.get("ObjectStates")) if c.get("Name") in CARD_NAMES} - {None}
                gone = sorted(p for p in found if p not in self.tree)
                if gone:
                    then = {p.lower(): p for p in _ls_tree(self.clone, sha)}
                    psds = {p: then.get(splitext(p)[0].lower() + ".psd") for p in gone}
                    seen = self.last_seen(sorted(set(gone) | {q for q in psds.values() if q}))
                    for p in gone:
                        if self.follow(p):
                            found.add(self.follow(p))
                            continue
                        for q in (p, psds[p]):
                            targets = self.by_blob.get(seen.get(q, ("", None))[1], [])
                            if len(targets) == 1:
                                png = self.by_lower.get(splitext(targets[0])[0].lower() + ".png")
                                if png:
                                    found.add(png)
            except (CalledProcessError, OSError, ValueError, AttributeError):
                found = None
        self._carried[sha] = found
        return found

    def follow(self, path: str) -> Optional[str]:
        """Where the bake's renames took a gone file (its Renamed trailers: a card renamed and redrawn in one commit), if
        that path is in the tree."""
        if getattr(self, "_told", None) is None:       # a _Repo made without __init__ (tests) has none yet
            self._told = {}
            if exists(getattr(self, "clone", "")):
                try:
                    out = _git(self.clone, "log", CARD_BRANCH, f"--grep=^{RENAMED}: ",
                               f"--format=%x1e%(trailers:key={RENAMED},valueonly,separator=%x1f)").decode("utf-8", "replace")
                except CalledProcessError:
                    out = ""
                for value in out.replace("\x1e", "\x1f").split("\x1f"):     # newest first: the newest rename of a path
                    pair = rename_trailer(value)
                    if pair:
                        self._told.setdefault(*pair)
        seen = {path}
        while path in self._told and self._told[path] not in seen:
            path = self._told[path]
            seen.add(path)
        return path if len(seen) > 1 and path in self.tree else None

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


def db_names(stats_path: str, tree: dict) -> dict:
    """
    cards.json's png -> name from a card data build's stats.json, by build_site's rule: a card's printed title (the
    parser's, under any Card Edits.json name) else its name (a variant picture N: NameN.png), a duplicate PNG its
    display name, then a PSD or PNG the data lacks its file stem; rule pages are left out (build_site lists them
    apart). Card Edits.json names are applied by _Names, as for cards.json.
    """
    from commands.card_db import parse_stats
    with open(stats_path, encoding="utf-8") as fh:
        stats = parse_stats(fh.read(), stats_path)
    in_cards = [p for p in sorted(tree) if p.startswith(CARD_FOLDERS) and not any(x in SKIP_PARTS for x in p.split("/"))]
    by_name: dict = {}
    for path in in_cards:
        if path.lower().endswith(".png"):
            by_name.setdefault(basename(path).lower(), path)

    def png_for(psd: str, name: str, variant=None):
        number = str(variant) if isinstance(variant, int) and variant > 1 else ""     # variant N: NameN.png
        guess = psd[:len(psd) - 4] + number + ".png" if psd.lower().endswith(".psd") else None
        return guess if guess in tree else by_name.get(name.replace(" ", "_").lower() + ".png")
    out, seen, dup_names, known = {}, set(), {}, set()
    for name, card in sorted(stats.items(), key=lambda kv: kv[0].lower()):
        path = getattr(card, "path", "") or ""
        if getattr(card, "duplicate_of", None) and not getattr(card, "variant_of", None):     # a variant: its own card
            if getattr(card, "display_name", None):
                dup_names[name.replace(" ", "_").lower() + ".png"] = card.display_name
            continue
        png = png_for(path, name, getattr(card, "variant", None))
        parsed = getattr(card, "parsed_values", None) or {}
        title = parsed["name"] if "name" in parsed else getattr(card, "display_name", None)
        rule = getattr(card, "card_type", "") == "rulebook" or "/Rulebook/" in path
        shown = (title if not rule else None) or name
        known |= {_norm(shown), _norm(name)}
        if png:
            seen.add(png)
            if not rule:                                    # rule pages: rules.json, not cards.json
                out[png] = shown
    for path in in_cards:                                   # duplicates: numbered PNGs named by the data
        found = dup_names.get(basename(path).lower())
        if found and path.lower().endswith(".png") and path not in seen:
            out[path] = found
            seen.add(path)
            known.add(_norm(found))
    for path in in_cards:                                   # not in the data yet: the file stem
        low = path.lower()
        if not (low.endswith(".psd") or (low.endswith(".png") and path not in seen)):
            continue
        name = splitext(basename(path))[0].replace("_", " ")
        if _norm(name) in known or "/Rulebook/" in path:
            continue
        known.add(_norm(name))
        png = png_for(path, name) if low.endswith(".psd") else path
        if png:
            out.setdefault(png, name)
    return out


class _Names:
    """Website names for card images: Card Edits.json > the card browser's cards.json (or a build's stats.json) > PSD
    stem > PNG stem."""

    def __init__(self, repo: _Repo, errors: list, stats_path: str = ""):
        """stats_path: a card data build's stats.json, read instead of the website's cards.json (Card sync: the run's
        own data)."""
        edits = None
        self.repo = repo
        self.site: dict = {}
        self.edits: dict = {}
        self.source = "file stems"
        if stats_path:
            try:
                self.site, self.source = db_names(stats_path, repo.tree), f"the card data ({basename(stats_path)})"
            except (OSError, ValueError, TypeError) as e:
                errors.append(f"Card data not read ({e}): Nicknames left as they are, new cards named by file stem.")
        else:
            self.site = self._website(repo, errors)
        try:
            path = EDITS_PATH if exists(join(repo.checkout, EDITS_PATH)) else None
            if path is not None:
                with open(join(repo.checkout, path), encoding="utf-8") as f:
                    edits = f.read()
            if edits is not None:
                self.edits = {CardAuthors.key(k): v for k, v in (loads(edits).get("cards") or {}).items() if isinstance(v, dict)}
        except (OSError, ValueError, AttributeError):
            pass

    def _website(self, repo: _Repo, errors: list) -> dict:
        """png -> name from the card browser's cards.json (an ETag 304 reuses the last), else the local site build."""
        global _site
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
                with open(join(repo.checkout, SITE_CARDS), encoding="utf-8") as f:
                    cards, self.source = load(f)["cards"], "the local site build"
            except (OSError, ValueError, KeyError):
                pass
            errors.append(f"Card browser data not fetched ({e}): names from {self.source}"
                          + ("." if cards else "; Nicknames left as they are, new cards named by file stem."))
        return {c["png"]: c["name"] for c in cards if isinstance(c, dict) and c.get("png") and c.get("name")}

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
        self.new: list = []             # the same, hand-picked decks (add_new_since)
        self.removed: list = []         # (deck label, [(name, path, copies)]): cards outside the deck's rule
        self.moved: list = []           # cards moved between add_new_since decks (other stars)
        self.names_real: dict = {}      # (path, old, new) -> objects
        self.names_case: dict = {}
        self.unnamed: dict = {}         # (path, Nickname) -> objects: not on the website, kept
        self.stale: list = []
        self.lua_added: list = []
        self.lua_removed: list = []
        self.rulebook: list = []        # the Rulebooks bag's changes, one line each
        self.left: list = []
        self.errors: list = []
        self.changed = False


# ========================
# Update Steps
# ========================
def _stem_key(path: str) -> tuple:
    """(top folder, card key, extension): one card's file wherever under its top folder it moves (PARTS_EDITOR.md §4.12)."""
    stem, ext = splitext(path)
    return path.split("/")[0].lower(), CardAuthors.key(_stem(path)), ext.lower()


def _stem_index(tree: dict) -> dict:
    out: dict = {}
    for path in tree:
        out.setdefault(_stem_key(path), []).append(path)
    return out


def _pin_links(save: dict, repo: _Repo, report: _Report) -> None:
    """Every repo link -> the commit that last changed its file; renamed files follow (by blob, else a card key found once
    under the same top folder: a card moved and redrawn); gone files are reported."""
    links: dict = {}                                        # (ref, encoded path) -> occurrences

    def collect(text):
        for m in LINK.finditer(text):
            links[m.groups()] = links.get(m.groups(), 0) + 1
        return text
    _strings(save, collect)
    if not repo.tree or not repo.history:
        return
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
    for path in lost:                                       # a card renamed and redrawn (the bake's Renamed trailers)
        if path not in rename and repo.follow(path):
            rename[path] = repo.follow(path)
    stems = _stem_index(repo.tree)
    for path in lost:                                       # a card moved and redrawn (its stars: a new "N Stars" folder)
        targets = stems.get(_stem_key(path), [])
        if path not in rename and len(targets) == 1:
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


def _template(everything: list, deck: dict, back: str):
    """A card to copy into an empty deck: from a deck of its scale and back, else of its scale, else any deck's."""
    scale = (deck.get("Transform") or {}).get("scaleX")
    found = [(o, parents[len(parents) - 1]) for o, parents in everything
             if o.get("Name") in CARD_NAMES and parents and parents[len(parents) - 1].get("Name") == "Deck"]
    same = [o for o, d in found if (d.get("Transform") or {}).get("scaleX") == scale]
    backed = [o for o in same if ((_entry(o) or {}).get("BackURL") or "").endswith("/" + _encode(back))] if back else []
    return next(iter(backed + same + [o for o, _ in found]), None)


def _sync_decks(save: dict, rules: dict, repo: _Repo, names: _Names, report: _Report) -> None:
    """Complete decks follow their rule: missing repo cards join (at the rule's usual count; an empty deck from a
    template card, at the deck's Transform), cards outside it leave. Hand-picked decks with add_new_since gain the
    rule's cards the save at that commit lacked, and a card moved to other stars changes deck (_move_between)."""
    everything = list(_walk(save.get("ObjectStates")))
    guids = {o.get("GUID") for o, _ in everything}
    top = 0
    for o, _ in everything:
        top = max([top, o.get("CardID", 0) // 100 if isinstance(o.get("CardID"), int) else 0]
                  + [int(k) for k in (o.get("CustomDeck") or {}) if k.isdigit()])
    ids: dict = {}                                          # new image path -> its new CustomDeck id
    top = _move_between(everything, rules, repo, report, top)
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
            since = rule.get("add_new_since") if rule.get("partial") else None
            if rule.get("partial") and not since:
                report.left.append(f"{label}: hand-picked, {len(cards)} cards kept as they are")
                continue
            if not rule.get("count"):
                report.left.append(f"{label}: no count in its rule, {len(cards)} cards kept as they are")
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
        if since:
            had = repo.carried(str(since))
            if had is None:
                report.errors.append(f"{label}: the save at `{since}` (add_new_since) can't be read; no new cards added.")
                continue
            missing = [p for p in wanted if p not in present and p not in had]
            outside = []
            if not missing:
                report.left.append(f"{label}: hand-picked, {len(cards)} cards kept, no new cards since `{str(since)[:8]}`")
                continue
        else:
            missing = [p for p in wanted if p not in present]
            outside = [c for c in cards if (_face_path(c) or "").lower().endswith(".png")    # rules only cover PNGs
                       and _face_path(c) in repo.tree and _face_path(c) not in wanted]
        if not missing and not outside:
            continue
        empty = not cards
        sibling = cards[len(cards) - 1] if cards else _template(everything, deck, back or "")
        if sibling is None:
            report.errors.append(f"{label}: empty, and no card in the save to copy; deck left empty.")
            continue
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
                if empty and "Transform" in deck:           # template from another deck: this deck's own
                    card["Transform"] = deepcopy(deck["Transform"])
                cards.append(card)
            if "CustomDeck" not in deck:                    # TTS's place for it: after DeckIDs
                items = list(deck.items())
                at = next((i + 1 for i, (k, _) in enumerate(items) if k == "DeckIDs"), len(items))
                deck.clear()
                deck.update(items[:at] + [("CustomDeck", {})] + items[at:])
            deck["CustomDeck"][str(ids[path])] = deepcopy(face)
            added.append((name, path, copies))
        if added:
            (report.new if since else report.added).append((label, added))
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


def _move_between(everything: list, rules: dict, repo: _Repo, report: _Report, top: int) -> int:
    """A card in an add_new_since deck whose image moved to other stars (its rule covers it only under its old ones)
    moves, object and all, to the one other add_new_since deck whose rule now covers it. Decks whose rule matches
    nothing or whose since save can't be read stay (_sync_decks reports them). A CustomDeck key the target uses for
    another face: the target's key for this face, else a fresh id above top. Returns top."""
    stars = {m.group(0) for m in map(STARS_FOLDER.search, repo.pngs) if m}
    since: dict = {}                                        # guid -> (deck, label, rule, rule cards)
    for deck, _ in everything:
        try:
            entry = rules.get("decks", {}).get(deck.get("GUID", "")) or {}
            rule = entry.get("rule") or {}
            if deck.get("Name") == "Deck" and rule.get("partial") and rule.get("add_new_since") and rule.get("count"):
                wanted = set(_rule_cards(rule, repo.pngs))
                if wanted and repo.carried(str(rule["add_new_since"])) is not None:
                    since[deck["GUID"]] = (deck, f"`{deck['GUID']}` {deck.get('Nickname') or '(unnamed)'}", rule, wanted)
        except (AttributeError, KeyError, TypeError, ValueError):
            continue                                        # _sync_decks reports it
    touched: set = set()
    for guid, (deck, label, rule, wanted) in since.items():
        cards = deck.get("ContainedObjects") or []
        for card in list(cards):
            path = _face_path(card)
            if not path or path not in repo.tree or path in wanted or not _other_stars(rule, path, stars):
                continue
            to = [g for g, (_, _, _, w) in since.items() if g != guid and path in w]
            if len(to) != 1:
                continue
            target, to_label, _, _ = since[to[0]]
            held = target.setdefault("ContainedObjects", [])
            place = held[len(held) - 1].get("Transform") if held else target.get("Transform")
            cards.remove(card)
            if place is not None:
                card["Transform"] = deepcopy(place)
            held.append(card)
            key = str(card.get("CardID", 0) // 100)
            face = (card.get("CustomDeck") or {}).get(key) or (deck.get("CustomDeck") or {}).get(key)
            faces = target.setdefault("CustomDeck", {})
            if face is not None and faces.get(key, face) != face:   # the key shows another image there
                key = next((k for k, v in faces.items() if v == face and k.isdigit()), None)
                if key is None:
                    top += 1
                    key = str(top)
                card["CardID"] = int(key) * 100 + card.get("CardID", 0) % 100
                card["CustomDeck"] = {key: deepcopy(face)}
            if face is not None:
                faces[key] = deepcopy(face)
            report.moved.append(f"{card.get('Nickname') or '?'} `{path}`: {label} -> {to_label}")
            touched.update((guid, to[0]))
    for guid in touched:
        d = since[guid][0]
        held = d.get("ContainedObjects") or []
        used = {str(c.get("CardID", 0) // 100) for c in held}
        d["CustomDeck"] = {k: v for k, v in (d.get("CustomDeck") or {}).items() if k in used}
        d["DeckIDs"] = [c.get("CardID", 0) for c in held]
    return top


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
    """Nickname = website name (nothing else: TEAM_FIELDS are the team's)."""
    for card, _ in _walk(save.get("ObjectStates")):
        if card.get("Name") not in CARD_NAMES:
            continue
        path = _face_path(card)
        if not path or path.startswith(HAND_NAMED):
            continue
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


def _rulebook(save: dict, repo: _Repo, report: _Report) -> None:
    """
    The Rulebooks bag holds the rulebook's PDF as one Custom_PDF object (on top: the bag's last object), made once from
    the bag's settings and linked to TTS/Rulebook/Rulebook.pdf (its link's commit is _pin_links'); page cards of the
    rulebook's pages from before the PDF, the Photoshop pages and their bag (OLD_RULEBOOK) are taken out. Anything else in
    the bag stays, the PDF object's place and settings too once made.
    """
    book = read_pages(repo.checkout)
    if not book.get("pdf"):
        return
    path = f"{RULEBOOK_PAGES}/{book['pdf']}"
    bag = next((o for o in save.get("ObjectStates") or [] if str(o.get("Name", "")).endswith("Bag") and o.get("Nickname") == RULEBOOK_BAG), None)
    if bag is None:
        report.errors.append(f"No bag named {RULEBOOK_BAG}: the rulebook's PDF is not in the save.")
        return
    inside = list(bag.get("ContainedObjects") or [])
    page_of = lambda o: (_face_path(o) or "") if o.get("Name") in CARD_NAMES else ""
    pdf_of = lambda o: (unquote(m.group(2)) if (m := LINK.search((o.get("CustomPDF") or {}).get("PDFUrl") or "")) else "") if o.get("Name") == "Custom_PDF" else ""
    old = [o for o in inside if page_of(o).startswith(OLD_PAGES)]
    pages = [o for o in inside if page_of(o).startswith(RULEBOOK_PAGES + "/")]
    pdfs = [o for o in inside if pdf_of(o) == path]
    taken = {id(o) for o in old + pages + pdfs}
    others = [o for o in inside if id(o) not in taken]
    guids = {o.get("GUID") for o, _ in _walk(save.get("ObjectStates"))}

    def guid() -> str:
        while True:
            g = token_hex(3)
            if g not in guids:
                guids.add(g)
                return g
    nested = [o for o in others if o.get("Nickname") == OLD_RULEBOOK]
    if nested or old:
        n = len(old) + sum(len(o.get("ContainedObjects") or []) for o in nested)
        others = [o for o in others if o not in nested]
        report.rulebook.append(f"the Photoshop rulebook taken out ({_plural(n, 'page')}{', and its bag' if nested else ''})")
    if pages:
        report.rulebook.append(f"{_plural(len(pages), 'page card')} taken out (the rulebook is a PDF now)")
    if pdfs:
        pdf = pdfs[0]
        for extra in pdfs[1:]:
            report.rulebook.append(f"a second PDF object `{extra.get('GUID')}` taken out")
    else:
        template = next((o for o in [*pages, *old, *(c for b in nested for c in b.get("ContainedObjects") or [])] if o.get("Transform")), None)
        pdf = {"GUID": guid(), "Name": "Custom_PDF",
               "Transform": deepcopy((template or bag).get("Transform") or {}), "Nickname": PDF_NAME,
               "Description": f"The rulebook. Flip pages with the arrows; also on the website: {SITE_URL}/#/rules",
               "GMNotes": ""}
        for k in ("AltLookAngle", "ColorDiffuse", "LayoutGroupSortIndex", "Value", "Locked", "Grid", "Snap", "IgnoreFoW",
                  "MeasureMovement", "DragSelectable", "Autoraise", "Sticky", "Tooltip", "GridProjection"):
            if k in bag:
                pdf[k] = deepcopy(bag[k])
        pdf["ColorDiffuse"] = {"r": 1.0, "g": 1.0, "b": 1.0}
        pdf["Transform"].update(scaleX=PDF_SCALE, scaleY=1.0, scaleZ=PDF_SCALE)
        pdf.update(HideWhenFaceDown=False, Hands=False,
                   CustomPDF={"PDFUrl": repo.url(path), "PDFPassword": "", "PDFPage": 0, "PDFPageOffset": 0},
                   LuaScript="", LuaScriptState="", XmlUI="")
        if repo.sha(path) is None:
            report.unpinned.add(path)
        report.rulebook.append(f"added the PDF `{path}` ({_plural(len(book.get('pages') or []), 'page')})")
    contained = others + [pdf]
    if [id(o) for o in contained] != [id(o) for o in inside]:
        bag["ContainedObjects"] = contained


def _decks(rules):
    """The parsed rules, if their top level is an object with a "decks" object."""
    if not isinstance(rules, dict) or not isinstance(rules.get("decks", {}), dict):
        raise ValueError('no "decks" object')
    return rules


def _load_rules(repo: _Repo, errors: list):
    """TTS/decks.json from the checkout, else GitHub (raw, main)."""
    local = join(repo.checkout, RULES_PATH)
    try:
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

    def pins(found):
        return [f"`{p}`: {shas(refs)} -> `{sha[:8]}` ({n} link{'s' if n != 1 else ''})"
                for p, (refs, sha, n) in sorted(found.items())]
    section("Links pinned", pins(report.pinned))
    section("Links re-pinned (changed files)", pins(report.repinned))
    section("Links not pinned (history doesn't know the file)", [f"`{p}`" for p in sorted(report.unpinned)])
    section("Renamed images", [f"`{old}` -> `{new}` ({n} links)" for old, (new, n) in sorted(report.renamed.items())])
    def listed(found):
        return [f"{label}: " + ", ".join(f"{name} `{path}`" + (f" x{c}" if c != 1 else "") for name, path, c in cards)
                for label, cards in found]
    section("Cards added", listed(report.added))
    section("New cards added (hand-picked decks, add_new_since)", listed(report.new))
    section("Cards removed (outside the deck's rule)", listed(report.removed))
    section("Cards moved (other stars, add_new_since decks)", report.moved)
    section("Nicknames changed", [f"`{p}`: '{old}' -> '{new}' ({n} card{'s' if n != 1 else ''})"
                                  for (p, old, new), n in sorted(report.names_real.items())])
    section("Nicknames changed (case/spacing only)", [f"`{p}`: '{old}' -> '{new}' ({n} card{'s' if n != 1 else ''})"
                                                      for (p, old, new), n in sorted(report.names_case.items())])
    section("Nicknames kept (not on the website)", [f"`{p}`: '{old}' ({n} card{'s' if n != 1 else ''})"
                                                    for (p, old), n in sorted(report.unnamed.items())])
    section("CustomDeck keys fixed", report.stale)
    section("resizeTargets", [f"added `{k}`" for k in report.lua_added] + [f"removed `{k}`" for k in report.lua_removed])
    section("Rulebook", report.rulebook)
    section("Images gone from the repo (report only)",
            [f"`{p}`: {shas(refs)} ({n} links){'; ' + note if note else ''}" for p, (refs, n, note) in sorted(report.gone.items())])
    section("Decks left alone", report.left)
    section("Errors", report.errors)
    return "\n".join(lines) + "\n"


def _commit_message(report: _Report) -> str:
    """'TTS save: 812 links pinned, 3 cards added', from the report."""
    links = sum(n for _, _, n in report.pinned.values()) + sum(n for _, _, n in report.repinned.values())
    added = sum(len(cards) for _, cards in report.added + report.new)
    removed = sum(len(cards) for _, cards in report.removed)
    names = len(report.names_real) + len(report.names_case)              # distinct names, not cards
    parts = [f"{_plural(links, 'link')} pinned" if links else "",
             f"{_plural(added, 'card')} added" if added else "",
             f"{_plural(removed, 'card')} removed" if removed else "",
             f"{_plural(len(report.renamed), 'rename')} followed" if report.renamed else "",
             f"{_plural(len(report.moved), 'card')} moved" if report.moved else "",
             f"{_plural(names, 'Nickname')} fixed" if names else "",
             "resizeTargets updated" if report.lua_added or report.lua_removed else "",
             "rulebook pages updated" if report.rulebook else ""]
    return "TTS save: " + (", ".join(p for p in parts if p) or "updated")


def _update(save_path: str, names_from: str = "") -> tuple:
    """update_save, also returning the _Report. names_from: see _Names (its stats_path)."""
    with open(save_path, "rb") as f:
        raw = f.read()
    save = loads(raw.decode("utf-8"))
    report = _Report()
    repo = _Repo(report.errors)
    rules, rules_from = _load_rules(repo, report.errors) if repo.tree and repo.history else (None, "nowhere")
    names = _Names(repo, report.errors, names_from)
    _pin_links(save, repo, report)
    if repo.tree:
        _fix_keys(save, report)
        if rules is not None:
            _sync_decks(save, rules, repo, names, report)
        _fix_cards(save, repo, names, report)
        _resize_targets(save, repo, names, report)
        _rulebook(save, repo, report)

    out = _dump(save).encode("utf-8")
    stem = splitext(save_path)[0]
    backup = f"{stem}.{datetime.now().strftime('%Y%m%d-%H%M%S')}.bak.json"
    report.changed = out != raw
    if report.changed:
        with open(backup, "wb") as f:
            f.write(raw)
        try:
            with open(f"{save_path}.tmp", "wb") as f:
                f.write(out)
            replace(f"{save_path}.tmp", save_path)
        finally:
            if exists(f"{save_path}.tmp"):
                remove(f"{save_path}.tmp")
    head = [f"- Save: `{basename(save_path)}` ({len(raw):,} bytes): "
            + (f"updated ({len(out):,} bytes; backup `{basename(backup)}`)" if report.changed else "unchanged"),
            f"- Repo: main `{repo.head[:8]}` from {repo.source or 'nowhere'} ({len(repo.tree)} files, {len(repo.pngs)} PNGs)",
            f"- Rules: `{RULES_PATH}` from {rules_from} ({len((rules or {}).get('decks', {}))} decks)",
            f"- Names: from {names.source} ({len(names.site)} cards)"]
    text = _render(report, head)
    if report.changed:
        with open(f"{stem}.report.md", "w", encoding="utf-8") as f:
            f.write(text)
        with open(f"{stem}.message.txt", "w", encoding="utf-8") as f:     # card-sync.yml's commit body
            f.write(_commit_message(report) + "\n")
    return text, report


# ========================
# Public API Functions
# ========================
def update_save(save_path: str, names_from: str = "") -> str:
    """
    Update a TTS save in place and return the report. When it changed: backup, report and commit subject beside it.

    Args:
        save_path: The save's JSON file
        names_from: A card data build's stats.json for the Nicknames ("" = the website's cards.json)

    Returns:
        The markdown change report (also written to <stem>.report.md when the save changed)
    """
    return _update(save_path, names_from=names_from)[0]


def main(args: Optional[list] = None) -> int:
    """The command line (card-sync.yml): update_save, the report on stdout."""
    parser = ArgumentParser(prog="python -m commands.tts_save", description="Update the TTS save of the card table in place.")
    parser.add_argument("save", help="the save's JSON file")
    parser.add_argument("--names", default="", help="a card data build's stats.json for the Nicknames (default: the "
                                                    "website's cards.json)")
    options = parser.parse_args(args)
    print(update_save(options.save, options.names), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# ========================
# Module Exports
# ========================
__all__ = [
    'update_save',
    'main',
]
