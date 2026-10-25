"""
Card previews for the website (docs/EDIT_PIPELINE_DESIGN.md section 16): one card's PSD-kind edit baked exactly as the
bake would bake it (card_bake.bake_card: the duplicate and blob gates, classify, apply_jobs and every gate, with the bake's
fonts, status chart and icon vocabulary) on a temp copy, the card repo never written. card_preview_branch publishes the
result on TTSCardMaker's orphan branch `previews` (card-preview.yml's second job, standard library only).

    python -m commands.card_preview --card <key or PSD path> --edit <json file> --out <dir> [--clone <checkout>]
                                    [--request <id>] [--repo <owner/name>] [--head <sha>]

It writes <out>/result.json always (a bad input, a refusal and a runner error too), and with an image preview.png (the
PNG the bake would commit, recompressed losslessly where oxipng is installed), preview.webp (WEBP_WIDTH wide, the
card view's width) and before.webp (the card's official PNG made the same way: the site's Diff compares like with like).
The edit is the card's Card Edits.json entry as it would be published (any of its fields; the PSD-kind ones are baked,
`creatures` gates the stats as in the bake, `_`-keys are ignored); result.json's `edit` is it exactly as sent, without
the `_`-keys. Not the bake's: the `recent` gate (a preview races nobody: it commits nothing). Exit 0, 2 for a bad input,
1 for any other error (result.json's `transient`: try again). Every dispatched value printed goes through clean().
No Discord or config imports; SPARSE_FILES is what card-preview.yml checks out: the import closure and
requirements-cards.txt.
"""
from argparse import ArgumentParser
from io import BytesIO
from json import dumps, loads
from os import environ, makedirs, remove, replace
from os.path import abspath, basename, exists, expanduser, getsize, isfile, join
from re import fullmatch
from shutil import rmtree
from sys import argv, stderr
from tempfile import mkdtemp
from time import perf_counter
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
from commands.card_preview_branch import BEFORE, MAX_CARD, PNG, REQUEST, RESULT, WEBP, blank, clean, stamp
from commands.git_commit import ref_head
from commands.github_api import token_from_env
from global_config import CARD_REPO

# ========================
# Configuration
# ========================
SPARSE_FILES = ("commands/card_preview.py", "commands/card_preview_branch.py", "commands/card_bake.py",
                "commands/psd_edit.py", "commands/psd_regions.py", "commands/psd_parser.py", "commands/psd_parts.py", "commands/card_models.py",
                "commands/card_edits.py", "commands/png_export.py", "commands/github_api.py", "commands/git_commit.py",
                "commands/repo_history.py", "global_config.py", "requirements-cards.txt")
MAX_EDIT = 16384                                # bytes of edit JSON (an ability is a few hundred)
WEBP_WIDTH, WEBP_QUALITY = 720, 75              # the card view's width; WebP (the previews contract), the site's thumbs are AVIF
RETRY = card_bake.TRANSIENT + ("blob",)         # try again: the bake's own, and a blob that failed its hash (a download's)
RESULT_V = 2                                    # result.json's "v": 2 adds variant and parts (PARTS_EDITOR.md §5, I4)


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
        raise BadInput(f"unknown field{'s' if len(unknown) > 1 else ''}: {clean(', '.join(unknown), 200)}")
    bad = [k for k, v in fields.items() if not CardEdits.valid(k, v)]
    if bad:
        raise BadInput(f"not a valid {bad[0]}: {clean(repr(fields[bad[0]]), 200)}")
    if not psd_fields(fields):
        raise BadInput("no PSD-kind field with a value (" + ", ".join(k for k, v in CardEdits.KIND.items() if v == "psd") + ")")
    return fields


def check_card(card: str) -> None:
    """--card: 1-MAX_CARD characters, none that clean() escapes (control characters, line separators); raises BadInput."""
    if not card.strip() or len(card) > MAX_CARD or clean(card) != card:
        raise BadInput(f"the card must be 1-{MAX_CARD} characters, no control characters")


def resolve(snapshot: Snapshot, card: str, fields: dict) -> Card:
    """
    The card the bake would take for this edit: one of the bake's card PSDs by path (any case; Snapshot.psds: no
    EXCLUDE_FOLDERS PSD, and of two PSDs with one key the one the bake takes), else a name matched like the edits file's
    keys (CardAuthors.key of the file stem; the website's card key). A name without a PSD keeps rel "" (the bake refuses
    it: duplicate or unsupported-card).

    Raises:
        BadInput: a path ending in .psd that is none of the bake's cards
    """
    if card.lower().endswith(".psd"):
        keys = {path: key for key, path in snapshot.psds.items()}
        rel = card if card in keys else next((p for p in keys if p.lower() == card.lower()), "")
        if not rel:
            raise BadInput(f"no card PSD at {card!r} in the card repo at {snapshot.head[:8]} (the bake takes none there)")
        key = keys[rel]
    else:
        key = CardAuthors.key(card)
        rel = snapshot.psds.get(key, "")
    view = picture_of(snapshot, key) if not rel and fields.get("parts") else None
    if view is not None:                        # a picture of a PSD with variant tags: the PSD, baked as that picture
        rel, view = view
        key = CardAuthors.key(card_name(rel))
    name = card_name(rel) if rel else " ".join(word[:1].upper() + word[1:] for word in key.split())
    return Card(name, key, psd_fields(fields), rel, snapshot.sha(rel) if rel else "", creatures=bool(fields.get("creatures")),
                view=view)


def picture_of(snapshot: Snapshot, key: str) -> Optional[tuple]:
    """(PSD path, picture n) of a numbered PNG keyed so (A_Snack2.png -> A_Snack.psd, 2), else None."""
    original = snapshot.duplicate_of(key)
    for path in snapshot.tree if original else ():
        number = fullmatch(r"(.+?)(\d+)", path[:len(path) - 4]) if path.lower().endswith(".png") else None
        if number and CardAuthors.key(card_name(path)) == key and (number.group(1) + ".psd").lower() == original.lower():
            return original, int(number.group(2))
    return None


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


def outcome(card: Card) -> tuple:
    """
    The bake's verdict on the card as the preview tells it.

    Returns:
        (state, transient, reason, {field: baked | redundant | refused | error})
    """
    baked = set(card.baked.jobs) if card.baked is not None and card.refusal is None else set()
    transient = card.transient or (card.refusal is not None and card.refusal.code in RETRY)
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


def changes_of(edits: list) -> list:
    """result.json's changes: one per field, or per layer of a parts bake (its old and new {layer id: text}; a
    deleted layer's new is "")."""
    out = []
    for e in edits:
        if isinstance(e.old, dict) and isinstance(e.new, dict):
            out += [{"field": e.field, "old": _plain(e.old.get(key)), "new": _plain(e.new.get(key) or "")}
                    for key in e.old]
        else:
            out.append({"field": e.field, "old": _plain(e.old), "new": _plain(e.new)})
    return out


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
    plan = getattr(card, "parts", None)
    if plan is not None:                        # per layer: skipped ones (changed in Photoshop), and what waits
        notes += [f"layer {key} skipped: {skip['why']}" for key, skip in plan.skipped.items()] + list(plan.notes)
    if card.redundant and set(card.redundant) != set(card.fields):
        notes.append(f"already on the card: {', '.join(card.redundant)}")
    return notes


def _small(data: bytes, path: str) -> list:
    """A PNG's bytes as a WEBP_WIDTH wide WebP at WEBP_QUALITY; returns its size."""
    image = Image.open(BytesIO(data))
    image.load()
    small = image.convert("RGBA" if image.mode in ("RGBA", "LA", "P") else "RGB")
    small.thumbnail((WEBP_WIDTH, WEBP_WIDTH * 3), Image.LANCZOS)
    small.save(path, "WEBP", quality=WEBP_QUALITY, method=4)
    return list(small.size)


def write_images(png_path: str, out: str, before: str = "") -> dict:
    """
    preview.png (losslessly recompressed when that is smaller: png_export.recompress, the same pixels and chunks),
    preview.webp (WEBP_WIDTH wide at WEBP_QUALITY: the card view's width) and, from the official PNG `before`,
    before.webp made the same way (the site's Diff: one codec on both sides, so unchanged pixels match).

    Returns:
        result.json's files, size and webp_size
    """
    with open(png_path, "rb") as fh:
        data = fh.read()
    smaller = png_export.recompress(data)
    with open(join(out, PNG), "wb") as fh:
        fh.write(smaller or data)
    with Image.open(BytesIO(data)) as image:
        size = list(image.size)
    files, sizes = {"png": PNG, "webp": WEBP}, {"png": len(smaller or data)}
    webp_size = _small(data, join(out, WEBP))
    sizes["webp"] = getsize(join(out, WEBP))
    if before and isfile(before):
        with open(before, "rb") as fh:
            _small(fh.read(), join(out, BEFORE))
        files["before"], sizes["before"] = BEFORE, getsize(join(out, BEFORE))
    return {"files": files, "size": size, "webp_size": webp_size, "bytes": sizes, "recompressed": smaller is not None}


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
            (Snapshot.fetch: the manifest, the PSD and PNG, the chart for an ability, a types edit's new icons)
        request: The client's id, echoed (REQUEST)
        head: The commit to preview on ("" = the clone's HEAD, or the branch head)

    Returns:
        result.json's dict (written); never raises for a card's or the runner's failure
    """
    started = perf_counter()
    out, clone = expanduser(out), abspath(expanduser(clone)) if clone else ""
    makedirs(out, exist_ok=True)
    for name in (RESULT, PNG, WEBP, BEFORE):
        if exists(join(out, name)):
            remove(join(out, name))
    result = blank(request, card)
    result.update(v=RESULT_V, variant=None, parts={})
    work = mkdtemp(prefix="card-preview_", dir=environ.get(card_bake.WORK_ENV) or None)
    try:
        if not REQUEST.fullmatch(request or ""):
            raise BadInput("the request must be 8-64 of A-Z a-z 0-9 -")
        check_card(card)
        fields = load_edit(edit)
        result["edit"] = fields                 # as sent (no `_`-keys): the site's staleness check compares it
        token_from_env()                        # the Action has no config.py: blob fetches take GITHUB_TOKEN
        snapshot = local_snapshot(repository, clone, head) if clone else snapshot_at(repository, head or ref_head(repository))
        result["base"].update(repo=repository, commit=snapshot.head)
        one = resolve(snapshot, card, fields)
        png_rel = snapshot.png_for(one.rel)[0] if one.rel else ""
        if one.view is not None:                # the picture's own PNG
            png_rel = next((p for p in snapshot.png_for(one.rel)[1] if picture_of(snapshot, CardAuthors.key(card_name(p)))
                            == (one.rel, one.view)), png_rel)
            result["variant"] = one.view
        result.update(card=card_name(png_rel) if one.view is not None and png_rel else one.name, path=one.rel, png=png_rel)
        result["base"].update(psd_blob=one.psd_sha, png_blob=snapshot.sha(png_rel) if png_rel else "")
        fonts = card_bake._fonts_dir(snapshot)  # the bake's: PSD_TOOLS_FONT_DIRS, else the checkout's MDW/Fonts
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
        result["parts"] = dict((one.verdict or {}).get("parts") or {})     # per layer: baked, deleted, skipped...
        if one.baked is not None:
            result["changes"] = changes_of(one.baked.edits)
            result["parity"] = {f: {k: round(float(v), 3) for k, v in m.items() if k in ("bad_pct", "mean", "max")}
                                for f, m in one.baked.parity.items()}
            result["region"] = _plain(one.baked.region)
        folder = join(work, one.name.replace("/", "_"))
        shown = png_rel if one.view is not None else one.png
        rendered = join(folder, "new", basename(shown)) if shown else ""
        official = join(folder, "old", basename(shown)) if shown else ""    # bake_card's fetch of the card's PNG
        if state == "baked":
            result.update(image="baked", **write_images(one.baked.png_path, out, official))
        elif state == "refused" and rendered and isfile(rendered):    # a gate after the save: what the render got
            result.update(image="refused", **write_images(rendered, out, official))
    except BadInput as error:
        result.update(state="error", transient=False, reason=f"input: {error}")
    except Transient as error:
        result.update(state="error", transient=True, reason=f"error: {error}")
    except psd_edit.Refused as refused:         # before the card's bake: the icon manifest's blob, say
        result.update(state="error", transient=refused.code in RETRY, reason=f"{refused.code}: {refused.why}")
    except Exception as error:                  # noqa: BLE001 - result.json is always written
        result.update(state="error", transient=isinstance(error, (RequestException, ImportError, OSError)),
                      reason=f"error: {type(error).__name__}: {error}", traceback=format_exc().strip().splitlines()[-1])
    finally:
        rmtree(work, ignore_errors=True)
    result["took_s"] = round(perf_counter() - started, 1)
    result["at"] = stamp()
    with open(join(out, RESULT + ".tmp"), "w", encoding="utf-8") as fh:
        fh.write(dumps(result, indent=1, ensure_ascii=False) + "\n")
    replace(join(out, RESULT + ".tmp"), join(out, RESULT))
    return result


def render(result: dict) -> str:
    """The job summary (stdout, so the log too): every value through clean(), one line each."""
    base = result.get("base") or {}
    lines = ["# Card preview", "", f"- Request: `{clean(result.get('request'))}`"]
    card = clean(result.get("card") or "", MAX_CARD)
    if result.get("path"):
        lines.append(f"- Card: **{card}** `{clean(result['path'])}` at `{clean(base.get('commit', ''))[:8]}` "
                     f"(PSD `{clean(base.get('psd_blob', ''))[:8]}`, PNG `{clean(base.get('png_blob', ''))[:8] or 'none'}`)")
    else:
        lines.append(f"- Card: **{card}**")
    if result.get("edit"):
        lines.append("- Edit: " + clean(", ".join(f"{k} {v!r}" for k, v in result["edit"].items()), 2000))
    lines.append(f"- **{result['state'].capitalize()}**" + (" (try again)" if result.get("transient") else "")
                 + f": {clean(result.get('reason', ''))}")
    if result.get("fields"):
        lines.append("- Fields: " + clean(", ".join(f"{k} {v}" for k, v in result["fields"].items())))
    lines += [f"- {clean(note)}" for note in result.get("notes", [])]
    if result.get("image"):
        sizes = result.get("size") or ["?", "?"]
        lines.append(f"- Image ({result['image']}): `{PNG}` {sizes[0]}x{sizes[1]}, `{WEBP}` {WEBP_WIDTH} px wide"
                     + (f", `{BEFORE}`" if (result.get("files") or {}).get("before") else ""))
    lines += ["", f"Took {result.get('took_s', 0):.0f} s."]
    return "\n".join(lines) + "\n"


# ========================
# Public API Functions
# ========================
def main(args: Optional[list] = None) -> int:
    """The command line (module docstring); the summary on stdout."""
    args = list(argv[1:] if args is None else args)
    parser = ArgumentParser(prog="python -m commands.card_preview",
                            description="Bake one card's website edit as the bake would, on a temp copy, and write "
                                        f"{RESULT}, {PNG}, {WEBP} and {BEFORE} (docs/EDIT_PIPELINE_DESIGN.md section 16).")
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
        print(f"card_preview: {clean(result['reason'], 2000)}", file=stderr)
        return 2 if result["reason"].startswith("input:") else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# ========================
# Module Exports
# ========================
__all__ = [
    'BadInput',
    'RESULT_V',
    'changes_of',
    'SPARSE_FILES',
    'check_card',
    'load_edit',
    'main',
    'notes_of',
    'outcome',
    'preview',
    'render',
    'resolve',
    'write_images',
]
