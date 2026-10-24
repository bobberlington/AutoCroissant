"""
Card previews for the website (docs/EDIT_PIPELINE_DESIGN.md section 16): one card's PSD-kind edit baked exactly as the
bake would bake it (card_bake.bake_card: the duplicate and blob gates, classify, apply_jobs and every gate, with the bake's
fonts, status chart and icon vocabulary) on a temp copy, the card repo never written; and the result's publication on
TTSCardMaker's orphan branch `previews` (card-preview.yml).

    python -m commands.card_preview --card <key or PSD path> --edit <json file> --out <dir> [--clone <checkout>]
                                    [--request <id>] [--repo <owner/name>] [--head <sha>]
    python -m commands.card_preview publish --out <dir> --request <id> [--git <checkout>] [--remote <name or url>] [--png]

The preview writes <out>/result.json always (a bad input, a refusal and a runner error too), and with an image
preview.png (the PNG the bake would commit, recompressed losslessly) and preview.webp (WEBP_WIDTH wide, for the website).
The edit is the card's Card Edits.json entry as it would be published (any of its fields; the PSD-kind ones are baked,
`creatures` gates the stats as in the bake, `_`-keys are ignored). Not the bake's: the `recent` gate (a preview races
nobody: it commits nothing). Exit 0, 2 for a bad input, 1 for any other error (result.json's `transient`: try again).
publish makes one commit on `previews` with previews/<request>/ (result.json, preview.webp, with --png preview.png) and
index.json (one row per preview), on the branch head, retried on a moved head; when the branch would hold more than KEEP
previews or one older than DAYS, the commit is a fresh orphan with the KEEP_AFTER newest under DAYS_AFTER old (pushed
with a lease). It uses the checkout's git and credentials (actions/checkout's), blobless is enough (--no-thin pushes).
No Discord or config imports; SPARSE_FILES is the import closure card-preview.yml checks out.
"""
from argparse import ArgumentParser
from datetime import datetime, timedelta, timezone
from io import BytesIO
from json import dumps, loads
from os import environ, makedirs, remove, replace
from os.path import basename, exists, getsize, isfile, join
from random import uniform
from re import compile as re_compile
from shutil import rmtree
from subprocess import CalledProcessError, TimeoutExpired
from sys import argv, stderr
from tempfile import mkdtemp
from time import perf_counter, sleep
from traceback import format_exc
from typing import Optional

from PIL import Image
from psd_tools import PSDImage
from requests import RequestException

import commands.card_bake as card_bake
import commands.png_export as png_export
import commands.psd_edit as psd_edit
from commands.card_bake import Card, Snapshot, Transient, local_snapshot, psd_fields, snapshot_at, vocabulary
from commands.card_edits import CardAuthors, CardEdits
from commands.card_models import card_name
from commands.git_commit import GIT_IDENTITY, git, plural, ref_head
from commands.github_api import token_from_env
from commands.repo_history import batch_env
from global_config import CARD_REPO

# ========================
# Configuration
# ========================
SPARSE_FILES = ("commands/card_preview.py", "commands/card_bake.py", "commands/psd_edit.py", "commands/psd_regions.py",
                "commands/psd_parser.py", "commands/card_models.py", "commands/card_edits.py", "commands/png_export.py",
                "commands/github_api.py", "commands/git_commit.py", "commands/repo_history.py", "global_config.py")
REQUEST = re_compile(r"[A-Za-z0-9-]{8,64}")     # the client's id: a path on the branch (card-preview.yml checks it too)
MAX_EDIT = 16384                                # bytes of edit JSON (an ability is a few hundred)
MAX_CARD = 200                                  # characters of --card
RESULT, PNG, WEBP = "result.json", "preview.png", "preview.webp"
WEBP_WIDTH = 720                                # the website's card view (build_site IMAGE_SIZES' 720)
WEBP_QUALITY = 80
STATES = ("baked", "redundant", "refused", "error")
VERSION = 1                                     # result.json's and index.json's "v"
BRANCH = "previews"
DIR = "previews"                                # previews/<request>/ on the branch
INDEX, README = "index.json", "README.md"
LOCAL_REF = "refs/card-preview/previews"        # where publish fetches the branch head (no remote-tracking ref moved)
KEEP, DAYS = 200, timedelta(days=7)             # more previews or an older one: the next commit is a fresh orphan...
KEEP_AFTER, DAYS_AFTER = 150, timedelta(days=6)  # ... with the newest this many under this age (a rewrite about daily)
TRIES = 5                                       # attempts, each from the head then; 1-2 s, 2-4 s, 3-6 s ... between them
BRANCH_README = ("# Card previews\n\nMade by TTSCardMaker's `.github/workflows/card-preview.yml` (AutoCroissant "
                 "`commands/card_preview.py`): `previews/<request>/` holds `result.json`, `preview.webp` and "
                 "`preview.png`; `index.json` lists them. One commit per preview; rewritten as one fresh commit when it "
                 f"holds more than {KEEP} previews or one older than {DAYS.days} days. Never merged into main.\n")


# ========================
# Input
# ========================
class BadInput(ValueError):
    """The dispatch's card, edit or request is not one the preview takes: result.json says why (exit 2)."""


def load_edit(raw: Optional[bytes]) -> dict:
    """
    The edit as card-preview.yml's input carries it: UTF-8 JSON, an object of Card Edits.json fields (CardEdits.TYPES and
    valid(); `_`-keys ignored) with at least one PSD-kind value.

    Args:
        raw: The edit file's bytes (None: it could not be read)

    Returns:
        The fields; raises ValueError with the reason
    """
    if raw is None:
        raise BadInput("the edit file could not be read")
    if len(raw) > MAX_EDIT:
        raise BadInput(f"the edit is over {MAX_EDIT} bytes")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise BadInput("the edit is not UTF-8") from error
    try:
        data = loads(text)
    except ValueError as error:
        raise BadInput(f"the edit is not JSON: {error}") from error
    if not isinstance(data, dict):
        raise BadInput("the edit is not a JSON object")
    fields = {k: v for k, v in data.items() if not str(k).startswith("_")}
    unknown = sorted(k for k in fields if k not in CardEdits.TYPES)
    if unknown:
        raise BadInput(f"unknown field{'s' if len(unknown) > 1 else ''}: {', '.join(unknown)}")
    bad = [k for k, v in fields.items() if not CardEdits.valid(k, v)]
    if bad:
        raise BadInput(f"not a valid {bad[0]}: {fields[bad[0]]!r}")
    if not psd_fields(fields):
        raise BadInput("no PSD-kind field with a value (" + ", ".join(k for k, v in CardEdits.KIND.items() if v == "psd") + ")")
    return fields


def check_card(card: str) -> None:
    """--card: 1-MAX_CARD characters, no control characters; raises BadInput."""
    if not card.strip() or len(card) > MAX_CARD or any(ord(c) < 32 or ord(c) == 127 for c in card):
        raise BadInput(f"the card must be 1-{MAX_CARD} characters, no control characters")


def resolve(snapshot: Snapshot, card: str, fields: dict) -> Card:
    """
    The card the bake would take for this edit: a PSD path in the tree (any case), else a name matched like the edits
    file's keys (CardAuthors.key of the file stem; the website's card key). A name without a PSD keeps rel "" (the bake
    refuses it: duplicate or unsupported-card).

    Raises:
        BadInput: a path ending in .psd that the tree lacks
    """
    if card.lower().endswith(".psd"):
        rel = card if card in snapshot.tree else next((p for p in snapshot.psds.values() if p.lower() == card.lower()), "")
        if not rel:
            raise BadInput(f"no PSD at {card!r} in the card repo at {snapshot.head[:8]}")
        key = CardAuthors.key(card_name(rel))
    else:
        key = CardAuthors.key(card)
        rel = snapshot.psds.get(key, "")
    name = card_name(rel) if rel else " ".join(word[:1].upper() + word[1:] for word in key.split())
    return Card(name, key, psd_fields(fields), rel, snapshot.sha(rel) if rel else "", creatures=bool(fields.get("creatures")))


# ========================
# Result
# ========================
def _plain(value):
    """A value as JSON holds it (numpy numbers, tuples, odd objects as strings)."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _run_url() -> Optional[str]:
    if environ.get("GITHUB_RUN_ID") and environ.get("GITHUB_REPOSITORY"):
        return f"{environ.get('GITHUB_SERVER_URL', 'https://github.com')}/{environ['GITHUB_REPOSITORY']}/actions/runs/{environ['GITHUB_RUN_ID']}"
    return None


def blank(request: str, card: str) -> dict:
    """result.json before anything is known: an error until the preview says otherwise."""
    return {"v": VERSION, "request": request if REQUEST.fullmatch(request or "") else None, "card": card, "path": "", "png": "",
            "base": {"repo": "", "commit": "", "psd_blob": "", "png_blob": ""}, "edit": {}, "state": "error",
            "transient": False, "fields": {}, "reason": "", "changes": [], "notes": [], "parity": {}, "region": None,
            "image": None, "files": {}, "runner": {}, "run": _run_url(), "took_s": 0.0, "at": _now()}


def outcome(card: Card) -> tuple:
    """
    The bake's verdict on the card as the preview tells it.

    Returns:
        (state, transient, reason, {field: baked | redundant | refused | error})
    """
    baked = set(card.baked.jobs) if card.baked is not None and card.refusal is None else set()
    transient = card.transient or (card.refusal is not None and card.refusal.code in card_bake.TRANSIENT)
    if card.refusal is not None:
        state = "error" if card.refusal.code == "error" or transient else "refused"
        reason = f"{card.refusal.code}: {card.refusal.why}"
    elif baked:
        state, reason = "baked", card_bake._describe(card)
    elif card.partial:
        state, reason = "refused", "; ".join(f"{r.code}: {r.why}" for r in card.partial)
    elif card.redundant:
        state, reason = "redundant", "already on the card: " + ", ".join(card.redundant)
    else:
        state, reason = "error", "error: the bake did nothing with the edit"
    other = "error" if state == "error" else "refused"
    fields = {f: "baked" if f in baked else "redundant" if f in card.redundant else other for f in card.fields}
    return state, transient, reason, fields


def notes_of(card: Card) -> list:
    """The bake's notes: pixel misses, FX layers, icon layouts, what the bot's validator would say of the new card, the
    stars or types edits refused beside it, the fields already on the card (when not all of them are)."""
    notes = []
    if card.baked is not None and card.refusal is None:
        notes += psd_edit.miss_notes(card.baked)
        if card.baked.fx:
            notes.append(f"FX layers: {', '.join(card.baked.fx)}")
        notes += [str(note) for note in getattr(card.baked, "fx_notes", [])]     # work/fx-rebuild's (BUG 2)
        notes += [f"{f}: {how}" for f, how in getattr(card.baked, "icons", {}).items() if how]
        notes += [f"validator: {problem}" for problem in card.baked.reparse.get("validator") or []]
    notes += [f"{r.field} not baked: {r.code}: {r.why}" for r in card.partial]
    if card.redundant and set(card.redundant) != set(card.fields):
        notes.append(f"already on the card: {', '.join(card.redundant)}")
    return notes


def write_images(png_path: str, out: str) -> dict:
    """
    preview.png (losslessly recompressed when that is smaller: png_export.recompress, the same pixels and chunks) and
    preview.webp (WEBP_WIDTH wide, as the website's card view makes its images).

    Returns:
        result.json's files, size and webp_size
    """
    with open(png_path, "rb") as fh:
        data = fh.read()
    smaller = png_export.recompress(data)
    with open(join(out, PNG), "wb") as fh:
        fh.write(smaller or data)
    image = Image.open(BytesIO(data))
    image.load()
    small = image.convert("RGBA" if image.mode in ("RGBA", "LA", "P") else "RGB")
    small.thumbnail((WEBP_WIDTH, WEBP_WIDTH * 3), Image.LANCZOS)
    small.save(join(out, WEBP), "WEBP", quality=WEBP_QUALITY, method=4)
    return {"files": {"png": PNG, "webp": WEBP}, "size": list(image.size), "webp_size": list(small.size),
            "bytes": {"png": len(smaller or data), "webp": getsize(join(out, WEBP))}, "recompressed": smaller is not None}


def preview(card: str, edit: Optional[bytes], out: str, repository: str = CARD_REPO, clone: str = "", request: str = "",
            head: str = "") -> dict:
    """
    One card's edit through the bake (bake_card) on a temp work dir; the card repo is only read.

    Args:
        card: The card's key (Card Edits.json's, the file stem) or its PSD path
        edit: The edit file's bytes (load_edit; None: unreadable)
        out: Where result.json and the images go (old ones removed first)
        repository: owner/name of the card repo
        clone: A checkout of it (blobless is enough): the tree and blobs from it, a missing blob by REST; "" = the API
        request: The client's id, echoed (REQUEST)
        head: The commit to preview on ("" = the clone's HEAD, or the branch head)

    Returns:
        result.json's dict (written); never raises for a card's or the runner's failure
    """
    started = perf_counter()
    makedirs(out, exist_ok=True)
    for name in (RESULT, PNG, WEBP):
        if exists(join(out, name)):
            remove(join(out, name))
    result = blank(request, card)
    work = mkdtemp(prefix="card-preview_", dir=environ.get(card_bake.WORK_ENV) or None)
    try:
        if not REQUEST.fullmatch(request or ""):
            raise BadInput("the request must be 8-64 of A-Z a-z 0-9 -")
        check_card(card)
        fields = load_edit(edit)
        result["edit"] = psd_fields(fields)
        if fields.get("creatures"):
            result["edit"]["creatures"] = True
        token_from_env()                        # the Action has no config.py: blob fetches take GITHUB_TOKEN
        snapshot = local_snapshot(repository, clone, head) if clone else snapshot_at(repository, head or ref_head(repository))
        result["base"].update(repo=repository, commit=snapshot.head)
        one = resolve(snapshot, card, fields)
        png_rel = snapshot.png_for(one.rel)[0] if one.rel else ""
        result.update(card=one.name, path=one.rel, png=png_rel)
        result["base"].update(psd_blob=one.psd_sha, png_blob=snapshot.sha(png_rel) if png_rel else "")
        fonts = card_bake._fonts_dir(snapshot)  # the bake's: PSD_TOOLS_FONT_DIRS, the checkout's MDW/Fonts, the cache
        result["runner"] = {"psd_tools": card_bake.PSD_TOOLS_VERSION, "text_engine": psd_edit.text_engine(),
                            "story_sync": psd_edit.story_sync(), "fonts": exists(fonts)}
        parser = vocabulary(snapshot)
        colors: list = []

        def chart() -> Optional[dict]:          # run_report's: the status chart's fills by blob sha, once
            if not colors:
                if psd_edit.STATUS_CHART not in snapshot.tree:
                    colors.append(None)
                else:
                    path = join(work, basename(psd_edit.STATUS_CHART))
                    snapshot.fetch(snapshot.sha(psd_edit.STATUS_CHART), path)
                    colors.append(psd_edit.status_fills(PSDImage.open(path)))
            return colors[0]
        card_bake.bake_card(snapshot, one, parser, fonts, chart, work, lambda: set())   # no `recent` gate: nothing is committed
        state, transient, reason, outcomes = outcome(one)
        result.update(state=state, transient=transient, reason=reason, fields=outcomes, notes=notes_of(one))
        if one.traceback:
            result["traceback"] = one.traceback.strip().splitlines()[-1]
        if one.baked is not None:
            result["changes"] = [{"field": e.field, "old": _plain(e.old), "new": _plain(e.new)} for e in one.baked.edits]
            result["parity"] = {f: {k: round(float(v), 3) for k, v in m.items() if k in ("bad_pct", "mean", "max")}
                                for f, m in one.baked.parity.items()}
            result["region"] = _plain(one.baked.region)
        rendered = join(work, one.name.replace("/", "_"), "new", basename(one.png)) if one.png else ""
        if state == "baked":
            result.update(image="baked", **write_images(one.baked.png_path, out))
        elif state == "refused" and rendered and isfile(rendered):    # a gate after the save: what the render got
            result.update(image="refused", **write_images(rendered, out))
    except BadInput as error:
        result.update(state="error", transient=False, reason=f"input: {error}")
    except Transient as error:
        result.update(state="error", transient=True, reason=f"error: {error}")
    except Exception as error:                  # noqa: BLE001 - result.json is always written
        result.update(state="error", transient=isinstance(error, (RequestException, ImportError, OSError)),
                      reason=f"error: {type(error).__name__}: {error}", traceback=format_exc().strip().splitlines()[-1])
    finally:
        rmtree(work, ignore_errors=True)
    result["took_s"] = round(perf_counter() - started, 1)
    result["at"] = _now()
    with open(join(out, RESULT + ".tmp"), "w", encoding="utf-8") as fh:
        fh.write(dumps(result, indent=1, ensure_ascii=False) + "\n")
    replace(join(out, RESULT + ".tmp"), join(out, RESULT))
    return result


def render(result: dict) -> str:
    """The job summary."""
    base = result.get("base") or {}
    lines = ["# Card preview", "", f"- Request: `{result.get('request')}`"]
    if result.get("path"):
        lines.append(f"- Card: **{result['card']}** `{result['path']}` at `{base.get('commit', '')[:8]}` "
                     f"(PSD `{base.get('psd_blob', '')[:8]}`, PNG `{base.get('png_blob', '')[:8] or 'none'}`)")
    else:
        lines.append(f"- Card: **{result.get('card')}**")
    if result.get("edit"):
        lines.append("- Edit: " + ", ".join(f"{k} {v!r}" for k, v in result["edit"].items()))
    lines.append(f"- **{result['state'].capitalize()}**" + (" (try again)" if result.get("transient") else "")
                 + f": {result.get('reason', '')}")
    if result.get("fields"):
        lines.append("- Fields: " + ", ".join(f"{k} {v}" for k, v in result["fields"].items()))
    lines += [f"- {note}" for note in result.get("notes", [])]
    if result.get("image"):
        sizes = result.get("size") or ["?", "?"]
        lines.append(f"- Image ({result['image']}): `{PNG}` {sizes[0]}x{sizes[1]}, `{WEBP}` {WEBP_WIDTH} px wide")
    lines += ["", f"Took {result.get('took_s', 0):.0f} s."]
    return "\n".join(lines) + "\n"


# ========================
# The previews branch
# ========================
def _git(where: str, *args: str, stdin: Optional[bytes] = None, lazy: bool = False) -> str:
    """git in the checkout as the bot (GIT_IDENTITY), no prompts; lazy: a blobless checkout may fetch a missing blob."""
    env = batch_env(where, **GIT_IDENTITY)
    if lazy:
        env.pop("GIT_NO_LAZY_FETCH", None)
    try:
        return git(where, *args, stdin=stdin, env=env).decode("utf-8")
    except CalledProcessError as error:
        raise Transient(f"git {args[0]} failed: {(error.stderr or b'').decode('utf-8', 'replace').strip()}") from error
    except (TimeoutExpired, OSError) as error:
        raise Transient(f"git {args[0]} failed: {error}") from error


def _fetch(where: str, remote: str) -> str:
    """The branch's head from the remote into LOCAL_REF (one commit, its trees, no blobs); "" when there is no branch."""
    listed = _git(where, "ls-remote", "--heads", remote, f"refs/heads/{BRANCH}").split()
    if not listed:
        return ""
    _git(where, "fetch", "-q", "--no-tags", "--no-write-fetch-head", "--depth", "1", "--filter=blob:none", remote,
         f"+refs/heads/{BRANCH}:{LOCAL_REF}")
    return _git(where, "rev-parse", "--verify", f"{LOCAL_REF}^{{commit}}").strip()


def _listing(where: str, head: str) -> tuple:
    """({request: entry tree sha}, index rows) at the branch head; an unreadable index.json counts as empty."""
    if not head:
        return {}, {}
    root = {line.split("\t", 1)[1]: line.split()[2] for line in _git(where, "ls-tree", "-z", head).split("\0") if "\t" in line}
    entries = {}
    if DIR in root:
        for line in _git(where, "ls-tree", "-z", f"{head}:{DIR}").split("\0"):
            if "\t" in line and line.split()[1] == "tree":
                entries[line.split("\t", 1)[1]] = line.split()[2]
    index = {}
    if INDEX in root:
        try:
            data = loads(_git(where, "cat-file", "blob", root[INDEX], lazy=True))
            index = data.get("previews", {}) if isinstance(data, dict) and isinstance(data.get("previews"), dict) else {}
        except ValueError:
            index = {}
    return entries, index


def _age(at, now: datetime) -> Optional[timedelta]:
    try:
        return now - datetime.strptime(str(at), "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def kept(entries: dict, index: dict, request: str, now: datetime) -> Optional[list]:
    """
    The previews a rewrite keeps, or None when none is due: due when the branch would hold more than KEEP previews, or
    one older than DAYS (or of no known age); it keeps this request and the newest others under DAYS_AFTER, KEEP_AFTER in
    all.
    """
    ages = {r: _age((index.get(r) or {}).get("at"), now) for r in entries if r != request}
    if len(ages) + 1 <= KEEP and all(age is not None and age <= DAYS for age in ages.values()):
        return None
    fresh = sorted((age, r) for r, age in ages.items() if age is not None and age <= DAYS_AFTER)
    return [request] + [r for _, r in fresh[:KEEP_AFTER - 1]]


def _blob(where: str, data: bytes) -> str:
    return _git(where, "hash-object", "-w", "--stdin", stdin=data).strip()


def _tree(where: str, rows: list) -> str:
    """mktree of (mode, kind, sha, name) rows."""
    return _git(where, "mktree", stdin="".join(f"{m} {k} {s}\t{n}\n" for m, k, s, n in sorted(rows, key=lambda r: r[3])).encode()).strip()


def _push(where: str, remote: str, commit: str, lease: Optional[str]) -> None:
    """The commit to the branch: a fast forward, or with lease (the head it was built from) a forced update; raises
    Transient when refused (the branch moved) or failed."""
    args = ["push", "-q", "--no-thin"] + ([f"--force-with-lease=refs/heads/{BRANCH}:{lease}"] if lease else [])
    _git(where, *args, remote, f"{commit}:refs/heads/{BRANCH}")


def _result_of(out: str, request: str) -> dict:
    """The preview's result.json, or the stand-in an early failure leaves (an error: try again)."""
    try:
        with open(join(out, RESULT), encoding="utf-8") as fh:
            result = loads(fh.read())
        if isinstance(result, dict) and result.get("request") == request and result.get("state") in STATES:
            return result
    except (OSError, ValueError):
        pass
    result = blank(request, "")
    result.update(transient=True, reason="error: the preview step left no result (see the run)")
    return result


def publish(out: str, request: str, where: str = ".", remote: str = "origin", png: bool = False,
            now: Optional[datetime] = None) -> dict:
    """
    One commit on the previews branch with previews/<request>/ and its index row, on the branch's head (TRIES times
    from a moved head); a rewrite as a fresh orphan commit when kept() says so, pushed with a lease.

    Args:
        out: The preview's out dir
        request: The client's id (REQUEST)
        where: A checkout of the card repo (its git and credentials; blobless is enough); nothing in it moves but LOCAL_REF
        remote: Its remote name or a URL
        png: Publish preview.png too

    Returns:
        {"commit", "parent" (None for an orphan), "rewritten", "kept", "dropped", "result"}; raises Transient when no
        push landed
    """
    if not REQUEST.fullmatch(request or ""):
        raise ValueError("the request must be 8-64 of A-Z a-z 0-9 -")
    result = _result_of(out, request)
    files = {}
    for name in (WEBP, PNG) if png else (WEBP,):
        if name in (result.get("files") or {}).values() and isfile(join(out, name)):
            with open(join(out, name), "rb") as fh:
                files[name] = fh.read()
    result["files"] = {kind: name for kind, name in (result.get("files") or {}).items() if name in files}   # what is published
    files[RESULT] = (dumps(result, indent=1, ensure_ascii=False) + "\n").encode("utf-8")
    row = {"at": result.get("at"), "card": result.get("card"), "path": result.get("path"), "state": result.get("state"),
           "psd_blob": (result.get("base") or {}).get("psd_blob", ""), "png_blob": (result.get("base") or {}).get("png_blob", "")}
    entry = _tree(where, [("100644", "blob", _blob(where, data), name) for name, data in files.items()])
    readme = _blob(where, BRANCH_README.encode("utf-8"))
    why = ""
    for attempt in range(TRIES):
        if attempt:
            sleep(uniform(1, 2) * attempt)
        try:
            head = _fetch(where, remote)
            entries, index = _listing(where, head)
            entries[request], index[request] = entry, row
            keep = kept(entries, index, request, now or datetime.now(timezone.utc)) if head else None
            dropped = sorted(set(entries) - set(keep)) if keep is not None else []
            entries = {r: t for r, t in entries.items() if r not in dropped}
            index = {r: index[r] for r in sorted(entries) if r in index}
            listing = _blob(where, (dumps({"v": VERSION, "previews": index}, indent=1, sort_keys=True, ensure_ascii=False) + "\n").encode())
            previews = _tree(where, [("040000", "tree", t, r) for r, t in entries.items()])
            root = _tree(where, [("100644", "blob", listing, INDEX), ("100644", "blob", readme, README), ("040000", "tree", previews, DIR)])
            message = f"Preview {request}: {result.get('card') or '?'} ({result.get('state')})"
            if keep is not None:
                message += (f"\n\nRewritten as one commit: {len(entries)} of {len(entries) + len(dropped)} previews kept "
                            f"(more than {KEEP}, or one older than {DAYS.days} days).")
            parent = head if head and keep is None else None
            commit = _git(where, "commit-tree", root, *(["-p", parent] if parent else []), "-m", message).strip()
            _push(where, remote, commit, head if keep is not None else None)
        except Transient as error:              # most often a push refused: the branch moved since the fetch
            why = str(error)
            print(f"card_preview: attempt {attempt + 1} of {TRIES}: {why}", file=stderr)
            continue
        return {"commit": commit, "parent": parent, "rewritten": keep is not None, "kept": len(entries), "dropped": dropped,
                "result": result}
    raise Transient(f"not published after {TRIES} tries: {why}")


# ========================
# Public API Functions
# ========================
def main(args: Optional[list] = None) -> int:
    """The command line (module docstring); the summary on stdout."""
    args = list(argv[1:] if args is None else args)
    if args[:1] == ["publish"]:
        parser = ArgumentParser(prog="python -m commands.card_preview publish",
                                description=f"Publish a preview's out dir on the {BRANCH} branch, one commit.")
        parser.add_argument("--out", required=True, help="the preview's out dir")
        parser.add_argument("--request", required=True, help="the client's id")
        parser.add_argument("--git", default=".", help="a checkout of the card repo (its git and credentials)")
        parser.add_argument("--remote", default="origin", help="its remote (name or URL)")
        parser.add_argument("--png", action="store_true", help=f"publish {PNG} too")
        options = parser.parse_args(args[1:])
        try:
            done = publish(options.out, options.request, options.git, options.remote, options.png)
        except (ValueError, Transient) as error:      # a bad request, or no push landed
            print(f"card_preview: not published: {error}", file=stderr)
            return 1
        how = (f"on `{done['parent'][:8]}`" if done["parent"] else "as the branch's first commit" if not done["rewritten"] else
               f"as one fresh commit ({plural(done['kept'], 'preview')} kept, {len(done['dropped'])} dropped)")
        print(f"Published `{DIR}/{options.request}/` ({done['result']['state']}) in `{done['commit'][:8]}` {how}.")
        return 0
    parser = ArgumentParser(prog="python -m commands.card_preview",
                            description="Bake one card's website edit as the bake would, on a temp copy, and write "
                                        f"{RESULT}, {PNG} and {WEBP} (docs/EDIT_PIPELINE_DESIGN.md section 16).")
    parser.add_argument("--card", required=True, help="the card's key (Card Edits.json's: the file name) or its PSD path")
    parser.add_argument("--edit", required=True, help="a file with the edit's JSON (Card Edits.json fields)")
    parser.add_argument("--out", required=True, help="where the result and images go")
    parser.add_argument("--clone", default="", help="a checkout of the card repo (blobless is enough); default: the API")
    parser.add_argument("--request", default="local-preview", help="the client's id (8-64 of A-Z a-z 0-9 -)")
    parser.add_argument("--repo", default=CARD_REPO, help=f"owner/name of the card repo (default {CARD_REPO})")
    parser.add_argument("--head", default="", help="the commit to preview on (default: the clone's HEAD, or main's head)")
    options = parser.parse_args(args)
    try:
        with open(options.edit, "rb") as fh:
            raw = fh.read(MAX_EDIT + 1)
    except OSError:
        raw = None                              # load_edit refuses it: the result says so
    result = preview(options.card, raw, options.out, options.repo, options.clone, options.request, options.head)
    print(render(result), end="")
    if result["state"] == "error":
        print(f"card_preview: {result['reason']}", file=stderr)
        return 2 if result["reason"].startswith("input:") else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# ========================
# Module Exports
# ========================
__all__ = [
    'BadInput',
    'SPARSE_FILES',
    'blank',
    'check_card',
    'kept',
    'load_edit',
    'main',
    'notes_of',
    'outcome',
    'preview',
    'publish',
    'render',
    'resolve',
    'write_images',
]
