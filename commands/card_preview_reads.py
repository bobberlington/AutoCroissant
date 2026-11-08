"""
A card preview's reads, started early: card-preview.yml runs this in the background while its setup runs. The data's
render and spot records (db/ on the data branch) go to a folder (CARD_RECORDS_DIR), and the blobs card_preview reads
first (card_preview.blobs_of: the icon manifest, the card's PSD and PNG, the status chart for an ability or parts edit)
into the clone's object store, where card_bake.fetch_verified looks before REST. The same reads, earlier; a miss only
costs the time (the bake checks every blob's sha and fetches what the clone lacks).

    python3 -S -m commands.card_preview_reads --card <key or PSD path> --edit <json file> --clone <checkout>
                                              --repo <owner/name> --head <sha> --records <dir>

Standard library only (the runner's python3, before the venv): card_edits, card_models and global_config, so a card key
is card_preview.resolve's. One JSON line on stdout: what was read, and when.
"""
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha1
from json import dumps, loads
from os import environ, getpid, makedirs, replace
from os.path import basename, dirname, join
from re import fullmatch
from subprocess import run
from sys import argv
from time import perf_counter
from typing import Optional
from urllib.request import Request, urlopen

from commands.card_edits import CardAuthors
from commands.card_models import card_name
from global_config import API_BASE, DATA_BRANCH, ICON_MANIFEST_PATH

# ========================
# Configuration
# ========================
EXCLUDE_FOLDERS = ("Markers", "MDW")            # card_bake's: not cards
STATUS_CHART = "MDW/Croissant's Special Folder/Satus Effect Colors.psd"     # psd_edit's
RECORDS = ("render_record.json", "spot_record.json")                        # psd_edit's RENDER_RECORD, SPOT_RECORD
TIMEOUT = 60                                    # seconds a read (the preview waits 20 s, then goes on without it)
NO_LAZY = {**environ, "GIT_NO_LAZY_FETCH": "1"}


# ========================
# The card
# ========================
def tree_of(clone: str, head: str) -> dict:
    """{path: blob sha} of the clone at head (git ls-tree; a blobless clone has every tree)."""
    out = run(["git", "-C", clone, "ls-tree", "-r", "-z", head], capture_output=True, env=NO_LAZY, check=True, timeout=60)
    tree = {}
    for line in out.stdout.decode("utf-8").split("\0"):
        meta, _, path = line.partition("\t")
        if path and meta.split(" ")[1] == "blob":
            tree[path] = meta.split(" ")[2]
    return tree


def paths_of(tree: dict, card: str, fields: dict) -> list:
    """
    What card_preview reads first for this edit (resolve, then blobs_of), in the tree.

    Args:
        tree: {path: blob sha}
        card: The dispatch's card: a key or a PSD path
        fields: The edit's fields

    Returns:
        Repo paths; [] for a PSD path the bake takes none at (the preview's input error)
    """
    psds = {}
    for path in tree:
        if path.lower().endswith(".psd") and not any(folder in EXCLUDE_FOLDERS for folder in path.split("/")):
            psds[CardAuthors.key(card_name(path))] = path
    if card.lower().endswith(".psd"):
        rels = list(psds.values())
        rel = card if card in rels else next((p for p in rels if p.lower() == card.lower()), "")
        if not rel:
            return []
    else:
        key = CardAuthors.key(card)
        rel = psds.get(key, "")
        lower = {p.lower() for p in tree} if not rel and fields.get("parts") else ()
        for path in tree if lower else ():      # a picture of a PSD (A_Snack2.png): that PSD
            number = fullmatch(r"(.+?)(\d+)", path[:len(path) - 4]) if path.lower().endswith(".png") else None
            if number and CardAuthors.key(card_name(path)) == key and (number.group(1) + ".psd").lower() in lower:
                rel = number.group(1) + ".psd"
                break
    if not rel:
        return [ICON_MANIFEST_PATH]
    folder, name = dirname(rel), basename(rel[:len(rel) - 4]).lower()
    png = ""
    for path in tree:                           # card_bake Snapshot.png_for's
        if dirname(path) == folder and path.lower().endswith(".png") and basename(path)[:len(basename(path)) - 4].lower() == name:
            png = path
    chart = STATUS_CHART if any(fields.get(f) is not None for f in ("ability", "parts")) else ""   # psd_fields' rule
    return [p for p in (ICON_MANIFEST_PATH, rel, png, chart) if p in tree]


# ========================
# Reads
# ========================
def _get(url: str) -> bytes:
    headers = {"Accept": "application/vnd.github.raw", "User-Agent": "AutoCroissant"}
    if environ.get("GITHUB_TOKEN"):
        headers["Authorization"] = f"token {environ['GITHUB_TOKEN']}"
    with urlopen(Request(url, headers=headers), timeout=TIMEOUT) as reply:
        return reply.read()


def read_record(repository: str, name: str, folder: str) -> None:
    """db/<name> from the data branch to folder/<name> (temp file, then rename); raises when it is not a JSON object."""
    data = _get(f"{API_BASE}{repository}/contents/db/{name}?ref={DATA_BRANCH}")
    if not isinstance(loads(data), dict):
        raise ValueError(f"db/{name} is not a JSON object")
    temp = join(folder, f"{name}.{getpid()}")
    with open(temp, "wb") as fh:
        fh.write(data)
    replace(temp, join(folder, name))


def missing(clone: str, shas: list) -> set:
    """The shas the clone's object store lacks (one cat-file, never a lazy fetch)."""
    out = run(["git", "-C", clone, "cat-file", "--batch-check"], input="".join(s + "\n" for s in shas).encode(),
              capture_output=True, env=NO_LAZY, check=True, timeout=60).stdout.decode()
    return {line.split(" ")[0] for line in out.splitlines() if line.endswith(" missing")}


def read_blob(repository: str, sha: str, clone: str) -> None:
    """A blob by REST into the clone's object store; raises when it does not hash to sha."""
    data = _get(f"{API_BASE}{repository}/git/blobs/{sha}")
    if sha1(b"blob %d\0" % len(data) + data).hexdigest() != sha:
        raise ValueError(f"blob {sha[:8]} does not hash to its sha")
    run(["git", "-C", clone, "hash-object", "-w", "--stdin"], input=data, capture_output=True, env=NO_LAZY, check=True,
        timeout=60)


def reads(card: str, fields: dict, clone: str, repository: str, head: str, records: str) -> dict:
    """The records and the card's blobs side by side. Returns {what: [seconds, "ok" or why not]}; never raises."""
    started, done = perf_counter(), {}

    def one(what: str, read, *args) -> None:
        try:
            read(*args)
            done[what] = [round(perf_counter() - started, 2), "ok"]
        except Exception as error:              # noqa: BLE001 - the preview reads it as before
            done[what] = [round(perf_counter() - started, 2), f"{type(error).__name__}: {error}"[:200]]
    makedirs(records, exist_ok=True)
    with ThreadPoolExecutor(len(RECORDS) + 4) as pool:
        for name in RECORDS:
            pool.submit(one, f"db/{name}", read_record, repository, name, records)
        try:
            tree = tree_of(clone, head)
            paths = paths_of(tree, card, fields)
            lacking = missing(clone, [tree[p] for p in paths]) if paths else set()
        except Exception as error:              # noqa: BLE001
            done["tree"] = [round(perf_counter() - started, 2), f"{type(error).__name__}: {error}"[:200]]
            paths, lacking = [], set()
        for path in paths:
            if tree[path] in lacking:
                pool.submit(one, path, read_blob, repository, tree[path], clone)
            else:
                done[path] = [0, "in the clone"]
    return done


# ========================
# Public API Functions
# ========================
def main(args: Optional[list] = None) -> int:
    """The command line (module docstring)."""
    parser = ArgumentParser(prog="python3 -S -m commands.card_preview_reads",
                            description="Read a card preview's records and blobs early (card-preview.yml's Reads step).")
    for name in ("card", "edit", "clone", "repo", "head", "records"):
        parser.add_argument(f"--{name}", required=True)
    options = parser.parse_args(list(argv[1:] if args is None else args))
    try:
        with open(options.edit, "rb") as fh:
            edit = loads(fh.read())
    except (OSError, ValueError):
        edit = {}
    fields = {k: v for k, v in edit.items() if not str(k).startswith("_")} if isinstance(edit, dict) else {}
    started = perf_counter()
    done = reads(options.card, fields, options.clone, options.repo, options.head, options.records)
    print(dumps({"reads": done, "took_s": round(perf_counter() - started, 2)}, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# ========================
# Module Exports
# ========================
__all__ = [
    'main',
    'paths_of',
    'reads',
]
