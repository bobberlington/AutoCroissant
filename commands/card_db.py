"""
The card database as JSON (db/, the card data build's output): CardInfo <-> plain records, stats.json and
old_stats.json one card per line written atomically, the manifest's sha256 per file, and the historical pickles
(AutoCroissant's until 2026-10-25: the build's seed, the diagnostics) read through an allow-list unpickler. No Discord or
psd-tools.

The bot's sync() is the one network use, imported lazily.

CLI: python -m commands.card_db check <db dir>
"""
from argparse import ArgumentParser
from collections import defaultdict
from dataclasses import MISSING, fields
from hashlib import sha256
from json import dumps, loads
from os import fsync, makedirs, remove, replace
from os.path import dirname, exists, isdir, isfile, join
from pickle import Unpickler, UnpicklingError
from threading import Lock
from typing import Optional

from commands.card_models import CardInfo, CardStats

# ========================
# Configuration
# ========================
SCHEMA = 1
STATS_JSON = "stats.json"
OLD_STATS_JSON = "old_stats.json"
DETAILS_JSON = "psd_details.json"                  # psd_details' own format, copied as is
MANIFEST_JSON = "manifest.json"
DATA_FILES = (STATS_JSON, OLD_STATS_JSON, DETAILS_JSON)
SYNC_JSON = "sync.json"                            # the bot's copy only: the data commit it holds and that request's ETag
SHA_MEDIA = "application/vnd.github.sha"           # Accept: a commit's sha alone (the sync's conditional request)
EXTRA_KEYS = ("version", "created")              # CardInfo attributes outside its dataclass fields
CARD_FIELDS = {f.name: f for f in fields(CardInfo)}
STAT_KEYS = tuple(f.name for f in fields(CardStats))
# what a historical pickle may hold: the two card classes under either module path, and old_stats' defaultdict(list)
ALLOWED = {("commands.psd_analyzer", "CardInfo"): CardInfo, ("commands.psd_analyzer", "CardStats"): CardStats,
           ("commands.card_models", "CardInfo"): CardInfo, ("commands.card_models", "CardStats"): CardStats,
           ("collections", "defaultdict"): defaultdict, ("builtins", "list"): list}

# ========================
# Global State
# ========================
_lock = Lock()                                     # one writer per process
_warned: set = set()                               # record keys CardInfo lacks, warned once each


# ========================
# Records
# ========================
def _default(field):
    """A dataclass field's default value."""
    if field.default is not MISSING:
        return field.default
    if field.default_factory is not MISSING:
        return field.default_factory()
    return None


def _stats_record(stats) -> Optional[dict]:
    return None if stats is None else {key: getattr(stats, key) for key in STAT_KEYS}


def _plain(value, where: str):
    """value when it is JSON data (None, bool, int, float, str, lists, dicts with str keys); else TypeError."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_plain(v, where) for v in value]
    if isinstance(value, dict) and all(isinstance(k, str) for k in value):
        return {k: _plain(v, where) for k, v in value.items()}
    raise TypeError(f"{where}: {type(value).__name__} is not JSON data")


def record(card: CardInfo) -> dict:
    """
    A card as plain data: every CardInfo field (an older pickle's missing one takes its default) plus `version`;
    stats and creatures as CardStats' attribute names.

    Raises:
        TypeError: a value that isn't JSON data
    """
    out = {}
    name = getattr(card, "name", "?")
    for key, field in CARD_FIELDS.items():
        value = getattr(card, key, None) if hasattr(card, key) else _default(field)
        if key == "stats":
            out[key] = _stats_record(value)
        elif key == "creatures":
            out[key] = [_stats_record(c) for c in value] if value is not None else None
        else:
            out[key] = _plain(value, f"{name}.{key}")
    for key in EXTRA_KEYS:
        out[key] = _plain(getattr(card, key, None), f"{name}.{key}")
    return out


def card(rec: dict) -> CardInfo:
    """A record back to a CardInfo: a missing key takes the field's default; an unknown key is ignored (warned once)."""
    values = {}
    for key, value in rec.items():
        if key in CARD_FIELDS or key in EXTRA_KEYS:
            values[key] = value
        elif key not in _warned:
            _warned.add(key)
            print(f"card_db: record key {key!r} is not a CardInfo field; ignored")
    out = CardInfo(name=values.get("name") or "", card_type=values.get("card_type") or "unknown")
    for key, value in values.items():
        if key == "stats":
            value = CardStats(**{k: v for k, v in value.items() if k in STAT_KEYS}) if value is not None else None
        elif key == "creatures":
            value = [CardStats(**{k: v for k, v in c.items() if k in STAT_KEYS}) for c in value] if value is not None else None
        if key in EXTRA_KEYS and value is None:
            continue                               # the class default
        setattr(out, key, value)
    return out


def _line(rec: dict) -> str:
    """One record: its keys sorted; nested dicts (parsed_values, archived, version) keep their order, so a loaded card
    reprs as it did (CardEdits.apply compares reprs)."""
    return dumps({key: rec[key] for key in sorted(rec)}, ensure_ascii=False, separators=(",", ":"))


# ========================
# Files
# ========================
def stats_text(stats: dict) -> str:
    """stats.json: {"schema": 1, "cards": {name: record}}, one card per line, sorted by name."""
    lines = ",\n".join(f"{dumps(name, ensure_ascii=False)}:{_line(record(stats[name]))}" for name in sorted(stats))
    return '{"schema":%d,"cards":{\n%s\n}}\n' % (SCHEMA, lines)


def old_stats_text(old_stats: dict) -> str:
    """old_stats.json: {"schema": 1, "cards": {name: [record, ...]}}, one name per line (history, then archived)."""
    lines = ",\n".join(f"{dumps(name, ensure_ascii=False)}:[{','.join(_line(record(c)) for c in old_stats[name])}]"
                       for name in sorted(old_stats) if old_stats[name])
    return '{"schema":%d,"cards":{\n%s\n}}\n' % (SCHEMA, lines)


def _cards_of(text: str, path: str) -> dict:
    data = loads(text)
    if not isinstance(data, dict) or data.get("schema") != SCHEMA or not isinstance(data.get("cards"), dict):
        raise ValueError(f"{path}: not a schema {SCHEMA} card file")
    return data["cards"]


def _card_of(rec, path: str, name: str) -> CardInfo:
    """card(rec); ValueError for a record that isn't one."""
    try:
        return card(rec)
    except (AttributeError, TypeError) as e:
        raise ValueError(f"{path}: {name}: not a card record ({e})") from e


def parse_stats(text: str, path: str = STATS_JSON) -> dict:
    """stats.json's text -> {name: CardInfo}; ValueError for a bad file or record."""
    return {name: _card_of(rec, path, name) for name, rec in _cards_of(text, path).items()}


def parse_old_stats(text: str, path: str = OLD_STATS_JSON) -> defaultdict:
    """old_stats.json's text -> defaultdict(list) of CardInfo, like old_stats.pkl; ValueError for a bad file or record."""
    out = defaultdict(list)
    for name, recs in _cards_of(text, path).items():
        if not isinstance(recs, list):
            raise ValueError(f"{path}: {name}: not a list of records")
        out[name] = [_card_of(rec, path, name) for rec in recs]
    return out


def write_text(path: str, text: str) -> None:
    """Write to <path>.tmp, fsync, then replace: a kill mid-write leaves the old file whole."""
    with _lock:
        temp_path = path + ".tmp"
        try:
            with open(temp_path, "w", encoding="utf-8", newline="\n") as fh:
                fh.write(text)
                fh.flush()
                fsync(fh.fileno())
            replace(temp_path, path)
        except BaseException:
            try:
                remove(temp_path)
            except OSError:
                pass
            raise


def file_sha256(path: str) -> str:
    with open(path, "rb") as fh:
        return sha256(fh.read()).hexdigest()


def read_db(folder: str, details: bool = True) -> tuple:
    """
    db/'s cards.

    Args:
        folder: The db/ folder
        details: Parse psd_details.json too (the bot's loads keep only the cards: {} then)

    Returns:
        (stats, old_stats, details, manifest); empty ones when db/ is empty (the seed's case)
    """
    def text(name):
        path = join(folder, name)
        if not isfile(path):
            return None
        with open(path, encoding="utf-8") as fh:
            return fh.read()
    stats_t, old_t, manifest_t = (text(n) for n in (STATS_JSON, OLD_STATS_JSON, MANIFEST_JSON))
    details_t = text(DETAILS_JSON) if details else None
    stats = parse_stats(stats_t, join(folder, STATS_JSON)) if stats_t is not None else {}
    old = parse_old_stats(old_t, join(folder, OLD_STATS_JSON)) if old_t is not None else defaultdict(list)
    found = (loads(details_t).get("cards") or {}) if details_t is not None else {}
    manifest = loads(manifest_t) if manifest_t is not None else {}
    return (stats, old, found, manifest)


def write_db(folder: str, stats: dict, old_stats: dict, details_text: str, manifest: dict) -> dict:
    """
    db/'s three data files, then the manifest with each file's sha256 (written last: a reader checks the files
    against it).

    Returns:
        The manifest written
    """
    makedirs(folder, exist_ok=True)
    texts = {STATS_JSON: stats_text(stats), OLD_STATS_JSON: old_stats_text(old_stats), DETAILS_JSON: details_text}
    for name, text in texts.items():
        write_text(join(folder, name), text)
    manifest = {**manifest, "schema": SCHEMA, "files": {name: file_sha256(join(folder, name)) for name in texts}}
    write_text(join(folder, MANIFEST_JSON), dumps(manifest, ensure_ascii=False, indent=1, sort_keys=True) + "\n")
    return manifest


def check_db(folder: str, parse: bool = True) -> list:
    """
    Gate G3's file half: every data file parses and hashes to the manifest's sha256.

    Args:
        folder: The db/ folder
        parse: Parse the files too (the gate); without, the hashes alone (the bot's copy: is it whole?)

    Returns:
        The problems, one line each ([] when db/ is whole)
    """
    problems = []
    path = join(folder, MANIFEST_JSON)
    try:
        with open(path, encoding="utf-8") as fh:
            files = loads(fh.read()).get("files") or {}
    except (OSError, ValueError) as e:
        return [f"{path}: {e}"]
    for name in DATA_FILES:
        target = join(folder, name)
        if not isfile(target):
            problems.append(f"{name}: missing")
        elif files.get(name) != file_sha256(target):
            problems.append(f"{name}: sha256 differs from the manifest")
    try:
        if parse:
            read_db(folder)
    except (ValueError, TypeError) as e:
        problems.append(str(e))
    return problems


def round_trip_problems(stats: dict, old_stats: dict) -> list:
    """Gate G3's record half: names whose record changes through record -> CardInfo -> record."""
    bad = [name for name, c in stats.items() if record(card(record(c))) != record(c)]
    bad += [f"{name} (history)" for name, versions in old_stats.items()
            if any(record(card(record(c))) != record(c) for c in versions)]
    return bad


# ========================
# Historical pickles (read only)
# ========================
class SeedUnpickler(Unpickler):
    """Unpickles only the card classes, defaultdict and list (ALLOWED); any other global is refused."""
    def find_class(self, module, name):
        found = ALLOWED.get((module, name))
        if found is None:
            raise UnpicklingError(f"global {module}.{name} is not allowed in a card pickle")
        return found


def load_pickle(path: str):
    """A card pickle through SeedUnpickler (stats.pkl, old_stats.pkl or aliases.pkl)."""
    with open(path, "rb") as fh:
        return SeedUnpickler(fh).load()


def read_pickles(folder: str) -> tuple:
    """(stats, old_stats) of a folder holding stats.pkl and old_stats.pkl (old_stats empty when it has none)."""
    stats = load_pickle(join(folder, "stats.pkl"))
    old_path = join(folder, "old_stats.pkl")
    old = load_pickle(old_path) if exists(old_path) else defaultdict(list)
    return (stats, old if isinstance(old, defaultdict) else defaultdict(list, old))


def _at_commit(git_dir: str, sha: str, name: str) -> Optional[str]:
    """db/<name> at a data commit (git show), or None when it has none."""
    from subprocess import run
    done = run(["git", "-C", git_dir, "show", f"{sha}:db/{name}"], capture_output=True)
    return done.stdout.decode("utf-8") if done.returncode == 0 else None


def read_cards(source: str, git_dir: str = ".") -> tuple:
    """
    (stats, old_stats) of the card data in any of its forms, for the diagnostics: a db/ folder (or a folder holding
    db/), a folder of pickles, a stats.json or stats.pkl (the old_stats file beside it), or a data commit (db/ at that
    commit of git_dir: origin/data in a TTSCardMaker clone). old_stats is None when there is none to read.

    Raises:
        ValueError: no card data there
    """
    if isdir(source):
        for folder in (source, join(source, "db")):
            if isfile(join(folder, STATS_JSON)):
                return read_cards(join(folder, STATS_JSON))
        if isfile(join(source, "stats.pkl")):
            return read_cards(join(source, "stats.pkl"))
        raise ValueError(f"{source}: no {STATS_JSON}, db/{STATS_JSON} or stats.pkl")
    if isfile(source):
        is_json = source.endswith(".json")
        beside = join(dirname(source), OLD_STATS_JSON if is_json else "old_stats.pkl")
        if is_json:
            with open(source, encoding="utf-8") as fh:
                stats = parse_stats(fh.read(), source)
            old = None
            if isfile(beside):
                with open(beside, encoding="utf-8") as fh:
                    old = parse_old_stats(fh.read(), beside)
            return (stats, old)
        old = load_pickle(beside) if isfile(beside) else None
        return (load_pickle(source), old if old is None or isinstance(old, defaultdict) else defaultdict(list, old))
    stats_t = _at_commit(git_dir, source, STATS_JSON)
    if stats_t is None:
        raise ValueError(f"{source}: no such file or folder, nor a commit of {git_dir} with db/{STATS_JSON}")
    old_t = _at_commit(git_dir, source, OLD_STATS_JSON)
    return (parse_stats(stats_t), parse_old_stats(old_t) if old_t is not None else None)


# ========================
# The bot's copy (sync)
# ========================
def write_bytes(path: str, data: bytes) -> None:
    """write_text's way (temp file, fsync, replace) for exact bytes."""
    with _lock:
        temp_path = path + ".tmp"
        try:
            with open(temp_path, "wb") as fh:
                fh.write(data)
                fh.flush()
                fsync(fh.fileno())
            replace(temp_path, path)
        except BaseException:
            try:
                remove(temp_path)
            except OSError:
                pass
            raise


def _check_text(name: str, data: bytes) -> None:
    """A downloaded data file loads as its kind, every record too; else ValueError (nothing is written then)."""
    text = data.decode("utf-8")
    if name == STATS_JSON:
        parse_stats(text, name)
    elif name == OLD_STATS_JSON:
        parse_old_stats(text, name)
    elif not isinstance(loads(text).get("cards"), dict):
        raise ValueError(f"{name}: not a details file")


def read_json(path: str) -> dict:
    """A JSON object file's data, or {} when it is missing or not one."""
    try:
        with open(path, encoding="utf-8") as fh:
            data = loads(fh.read())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def sync(folder: str, repository: str, branch: str) -> dict:
    """
    A folder brought to a data branch's db/ (the bot's card_data/, design section 11): one conditional request for the
    branch's commit (a 304 is free), then, on a new commit, its manifest and only the data files whose sha256 differ
    from the folder's, raw at that commit (immutable, no lag). Each is checked against the manifest and parsed before
    any is written; then each is replaced atomically, the manifest and SYNC_JSON (commit, ETag) last, so an interrupted
    sync is finished by the next. Nothing is unpickled.

    Args:
        folder: The copy (created when missing)
        repository: owner/name of the card repo
        branch: The data branch

    Returns:
        {"commit", "changed": [file names written], "manifest"}: changed is [] when the folder was already there

    Raises:
        LookupError: the branch does not exist; requests' exceptions or ValueError: a failed or bad download (nothing
        written)
    """
    from commands.github_api import HEADERS, api_url, get_with_retry      # lazy: the build needs no network here
    from global_config import RAW_BASE
    state = read_json(join(folder, SYNC_JSON))
    whole = bool(state.get("commit")) and not check_db(folder, parse=False)     # the hashes: a cut-off sync
    headers = {**HEADERS, "Accept": SHA_MEDIA, **({"If-None-Match": state["etag"]} if whole and state.get("etag") else {})}
    reply = get_with_retry(api_url(repository, f"commits/{branch}"), headers=headers)
    if reply.status_code == 304:
        return {"commit": state["commit"], "changed": [], "manifest": read_json(join(folder, MANIFEST_JSON))}
    if reply.status_code in (404, 422):
        raise LookupError(f"{repository} has no {branch} branch")
    reply.raise_for_status()
    commit, etag = reply.text.strip(), reply.headers.get("ETag", "")
    if whole and commit == state.get("commit"):
        write_text(join(folder, SYNC_JSON), dumps({"commit": commit, "etag": etag}) + "\n")
        return {"commit": commit, "changed": [], "manifest": read_json(join(folder, MANIFEST_JSON))}

    def raw(name: str) -> bytes:
        got = get_with_retry(f"{RAW_BASE}{repository}/{commit}/db/{name}", timeout=60)
        got.raise_for_status()
        return got.content
    manifest_data = raw(MANIFEST_JSON)
    manifest = loads(manifest_data.decode("utf-8"))
    files = manifest.get("files") if isinstance(manifest, dict) else None
    if not isinstance(files, dict):
        raise ValueError(f"{MANIFEST_JSON} at {commit[:8]}: no file hashes")
    wanted = {}
    for name in DATA_FILES:
        if not files.get(name):
            raise ValueError(f"{MANIFEST_JSON} at {commit[:8]} lists no {name}")
        path = join(folder, name)
        if isfile(path) and file_sha256(path) == files[name]:
            continue
        data = raw(name)
        if sha256(data).hexdigest() != files[name]:
            raise ValueError(f"{name} at {commit[:8]}: sha256 differs from the manifest")
        _check_text(name, data)
        wanted[name] = data
    makedirs(folder, exist_ok=True)
    for name, data in wanted.items():
        write_bytes(join(folder, name), data)
    write_bytes(join(folder, MANIFEST_JSON), manifest_data)
    write_text(join(folder, SYNC_JSON), dumps({"commit": commit, "etag": etag}) + "\n")
    return {"commit": commit, "changed": sorted(wanted), "manifest": manifest}


# ========================
# CLI
# ========================
def main(argv: Optional[list] = None) -> int:
    parser = ArgumentParser(prog="python -m commands.card_db", description="The card database's JSON files (db/).")
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check", help="db/'s files against its manifest")
    check.add_argument("db")
    args = parser.parse_args(argv)
    problems = check_db(args.db)
    print("\n".join(problems) if problems else f"{args.db}: whole.")
    return 1 if problems else 0


# ========================
# Module Exports
# ========================
__all__ = [
    'ALLOWED',
    'DATA_FILES',
    'MANIFEST_JSON',
    'OLD_STATS_JSON',
    'SCHEMA',
    'STATS_JSON',
    'SYNC_JSON',
    'SeedUnpickler',
    'card',
    'check_db',
    'file_sha256',
    'load_pickle',
    'main',
    'old_stats_text',
    'parse_old_stats',
    'parse_stats',
    'read_cards',
    'read_db',
    'read_json',
    'read_pickles',
    'record',
    'round_trip_problems',
    'stats_text',
    'sync',
    'write_bytes',
    'write_db',
    'write_text',
]


if __name__ == "__main__":
    raise SystemExit(main())
