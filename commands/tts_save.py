"""
Updater for the Tabletop Simulator save of the card table (TTS/Impossibility Simulator.json in the card repo).

    python -m commands.tts_save <save.json>        (from the bot's root)
    update_save(path) -> report text                (for the bot)

Pins every repo image link to the commit that last changed the file, follows renames, syncs complete decks with
their rules (TTS/decks.json: missing repo cards added, cards outside the rule removed; a rule matching nothing leaves
its deck alone), sets card Nicknames to the
card browser's names, clears role tags, regenerates the Lua resizeTargets table. When something changed it writes
<stem>.<time>.bak.json and <stem>.report.md next to the save. The shas in the links are the only state.
"""
from copy import deepcopy
from datetime import datetime
from json import dumps, load, loads
from os import replace
from os.path import basename, exists, expanduser, join, splitext
from re import DOTALL, IGNORECASE, MULTILINE, compile as regex, escape, sub
from secrets import token_hex
from subprocess import CalledProcessError, run
from sys import argv, exit
from urllib.parse import quote, unquote

from requests import get

from commands.repo_history import REPOSITORY, raw_url, refresh
from global_config import HISTORY_CLONE, LOCAL_DIR_LOC, SITE_URL

# ========================
# Configuration
# ========================
RAW = f"https://raw.githubusercontent.com/{REPOSITORY}/"
LINK = regex(escape(RAW) + r"(refs/heads/main|main|[0-9a-f]{7,40})/([^\s\"\\<>]+)")
MAIN_REFS = ("main", "refs/heads/main")
URL_SAFE = "/'!()*-._~"                     # left as is in the save's links (GitHub raw style; spaces are %20)
RULES_PATH = "TTS/decks.json"
EDITS_PATH = "MDW/Card Edits.json"           # website renames (they win)
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
GIT_TIMEOUT = 300
HTTP_TIMEOUT = 30


# ========================
# Helper Functions
# ========================
def _git(where: str, *args: str) -> bytes:
    """git in a repo dir. In the blobless clone only refs and trees (rev-parse, rev-list, ls-tree): never blobs."""
    return run(["git", "-C", where, *args], capture_output=True, check=True, timeout=GIT_TIMEOUT).stdout


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


def _stem(path: str) -> str:
    """File stem with underscores as spaces."""
    return splitext(basename(path))[0].replace("_", " ")


def _card_key(name: str) -> str:
    """Card Edits.json key: case-insensitive, _ = space, whitespace collapsed (as build_site.py)."""
    return " ".join((name or "").replace("_", " ").lower().split())


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
                self.head = _git(self.clone, "rev-parse", "main").decode().strip()
                self.tree, self.source = _ls_tree(self.clone, "main"), "the history clone"
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
        ref = raw_url(REPOSITORY, path, blob)[len(RAW):].split("/", 1)[0] if blob else "main"
        return None if ref == "main" else ref

    def url(self, path: str) -> str:
        return f"{RAW}{self.sha(path) or 'main'}/{_encode(path)}"

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
            for commit in _git(self.clone, "rev-list", "--topo-order", "main").decode().split():
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

    def __init__(self, repo: _Repo, errors: list):
        self.repo = repo
        self.site: dict = {}
        self.edits: dict = {}
        self.source = "file stems"
        try:
            reply = get(f"{SITE_URL}/data/cards.json", timeout=HTTP_TIMEOUT)
            reply.raise_for_status()
            cards, self.source = reply.json()["cards"], "the card browser"
        except Exception as e:
            try:
                with open(join(repo.checkout, SITE_CARDS), encoding="utf-8") as f:
                    cards, self.source = load(f)["cards"], "the local site build"
            except (OSError, ValueError, KeyError):
                cards = []
            errors.append(f"Card browser data not fetched ({e}): names from {self.source}"
                          + ("." if cards else "; Nicknames left as they are, new cards named by file stem."))
        self.site = {c["png"]: c["name"] for c in cards if isinstance(c, dict) and c.get("png") and c.get("name")}
        try:
            with open(join(repo.checkout, EDITS_PATH), encoding="utf-8") as f:
                self.edits = {_card_key(k): v for k, v in (load(f).get("cards") or {}).items() if isinstance(v, dict)}
        except (OSError, ValueError, AttributeError):
            pass

    def _stem(self, path: str) -> str:
        psd = self.repo.by_lower.get(splitext(path)[0].lower() + ".psd")
        return _stem(psd or path)                           # a parsed card is keyed by its PSD stem

    def known(self, path: str):
        """The website's name for the image (Card Edits.json first), or None when it has none."""
        found = self.edits.get(_card_key(self._stem(path)))
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
            changes[(ref, enc)] = (repo.sha(new) or "main", _encode(new))
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
        entry = rules.get("decks", {}).get(guid) or {}
        rule = entry.get("rule")
        label = f"`{guid}` {deck.get('Nickname') or '(unnamed)'}" + (f", {entry['where']}" if entry.get("where") else "")
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
        if not wanted:                                      # a typo or a renamed folder: never empty the deck
            report.errors.append(f"{label}: its rule matches no repo image; deck left alone.")
            continue
        present = {_face_path(c) for c in cards}
        missing = [p for p in wanted if p not in present]
        outside = [c for c in cards if _face_path(c) in repo.tree and _face_path(c) not in wanted]
        if not missing and not outside:
            continue
        counts = {p: int(c) for p, c in (entry.get("odd_counts") or {}).items()}
        sibling = cards[len(cards) - 1]
        base = _entry(sibling) or {}
        back = entry.get("back")
        back_url = repo.url(back) if back in repo.tree else base.get("BackURL", "")
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
            copies = counts.get(path, int(rule["count"]))
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
            used = {str(c.get("CardID", 0) // 100) for c in cards}
            deck["CustomDeck"] = {k: v for k, v in (deck.get("CustomDeck") or {}).items() if k in used}
            left: dict = {}
            for c in outside:
                key = (c.get("Nickname") or "", _face_path(c))
                left[key] = left.get(key, 0) + 1
            report.removed.append((label, [(n, p, c) for (n, p), c in left.items()]))
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


def _load_rules(repo: _Repo, errors: list):
    """TTS/decks.json from the checkout, else from GitHub (raw, main)."""
    local = join(repo.checkout, RULES_PATH)
    try:
        if exists(local):
            with open(local, encoding="utf-8") as f:
                return load(f), "the checkout"
        reply = get(f"{RAW}main/{_encode(RULES_PATH)}", timeout=HTTP_TIMEOUT)
        reply.raise_for_status()
        return reply.json(), "GitHub"
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
    with open(save_path, "rb") as f:
        raw = f.read()
    save = loads(raw.decode("utf-8"))
    report = _Report()
    repo = _Repo(report.errors)
    rules, rules_from = _load_rules(repo, report.errors) if repo.tree and repo.history else (None, "nowhere")
    names = _Names(repo, report.errors)
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
        with open(backup, "wb") as f:
            f.write(raw)
        with open(f"{save_path}.tmp", "wb") as f:
            f.write(out)
        replace(f"{save_path}.tmp", save_path)
    head = [f"- Save: `{basename(save_path)}` ({len(raw):,} bytes): "
            + (f"updated ({len(out):,} bytes; backup `{basename(backup)}`)" if report.changed else "unchanged"),
            f"- Repo: main `{repo.head[:8]}` from {repo.source or 'nowhere'} ({len(repo.tree)} files, {len(repo.pngs)} PNGs)",
            f"- Rules: `{RULES_PATH}` from {rules_from} ({len((rules or {}).get('decks', {}))} decks)",
            f"- Names: from {names.source} ({len(names.site)} cards)"]
    text = _render(report, head)
    if report.changed:
        with open(f"{stem}.report.md", "w", encoding="utf-8") as f:
            f.write(text)
    return text


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
    'main',
]
