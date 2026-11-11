"""
Plates for the website's mock-up: the card drawn without one text layer, cut to that layer's box (its effects' reach
and PAD past it), per title and ability layer of its parts (psd_parts). The mock-up pastes a plate over the old text
instead of guessing the backdrop from the PNG. One file per PSD blob, crops at the site's card view scale (VIEW_W
wide) in one AVIF; a crop whose backdrop differs from the PNG off the layer's ink is left out (the mock-up's old way).
Card checks makes them (check_card's "plate") and publishes them on the `plates` branch as <PSD blob>.json, those of
PSDs gone from main dropped; the website fetches the open card's in edit mode.

    python -m commands.card_plates publish --git <checkout> [--head <sha>] [--remote origin] <run results.json> ...
"""
from argparse import ArgumentParser
from base64 import b64encode
from io import BytesIO
from json import dumps, loads
from math import gcd
from random import uniform
from subprocess import CalledProcessError, TimeoutExpired, run as run_process
from sys import argv, stderr
from time import sleep
from typing import Optional

import numpy as np
from PIL import Image
from psd_tools import PSDImage

import commands.psd_edit as psd_edit
from commands.png_export import FX_EXTRA, like_mode
from commands.psd_parser import EFFECT_LAYER_TAG, variant_view, variants_of
from commands.psd_regions import fx_pad
from commands.repo_history import batch_env

# ========================
# Configuration
# ========================
VERSION = 1
VIEW_W = 720                                # the site's card view width (the mock-up draws on it)
PAD = 16                                    # px past a layer's box and effects
INK = 2                                     # |full - plate| past this: the layer's own pixels
SEAM = 24                                   # |plate - PNG| past this off the ink: a seam
SEAM_SHARE = 0.002                          # of a crop's off-ink pixels: more, and the crop is left out
GAP = 8                                     # px between crops in the sheet (AVIF blocks stay apart)
QUALITY = 60
KINDS = ("title", "ability", "blocks")
BRANCH = "plates"
LOCAL_REF = "refs/card-plates/head"         # where publish fetches the branch head (no remote-tracking ref moved)
TRIES = 5
GIT_TIMEOUT = 300
IDENTITY = {"GIT_AUTHOR_NAME": "AutoCroissant", "GIT_AUTHOR_EMAIL": "autocroissant@gmail.com",
            "GIT_COMMITTER_NAME": "AutoCroissant", "GIT_COMMITTER_EMAIL": "autocroissant@gmail.com"}  # git_commit's


# ========================
# Helper Functions
# ========================
def _snap(box: tuple, step: int, size: tuple) -> tuple:
    """box grown to multiples of step (its scaled edges whole pixels), inside size."""
    x0, y0, x1, y1 = box
    return (max(0, x0 // step * step), max(0, y0 // step * step),
            min(size[0], -(-x1 // step) * step), min(size[1], -(-y1 // step) * step))


def _rgb(image: Image.Image) -> np.ndarray:
    return np.asarray(like_mode(image, "RGB"), np.int16)


def layer_ids(parts: dict) -> list:
    """The title, ability and block layers of a details record's parts, in order."""
    return [p["id"] for k in KINDS for p in (parts or {}).get(k) or [] if isinstance(p, dict) and p.get("id") is not None]


def _padded(layer) -> Optional[tuple]:
    pad = max(PAD, fx_pad(layer) + FX_EXTRA)
    x0, y0, x1, y1 = layer.bbox
    return (x0 - pad, y0 - pad, x1 + pad, y1 + pad) if x1 > x0 and y1 > y0 else None


def _meets(layer, other) -> bool:
    """other drawn where layer's padded box is."""
    a, b = _padded(layer), _padded(other) if other.is_visible() else None
    return bool(a and b and b[0] < a[2] and a[0] < b[2] and b[1] < a[3] and a[1] < b[3])


def crop(psd: PSDImage, layer, png: np.ndarray, icc: bool, step: int, extra=()) -> Optional[tuple]:
    """(box, plate image, seam px, ink box) of one layer (canvas px), or None (not drawn, or its backdrop not the PNG's).
    extra: layers the bake redraws with it (its FX layers, copies), hidden too where they meet its box."""
    if layer.kind != "type" or not layer.is_visible() or not layer.opacity:
        return None
    if _padded(layer) is None:
        return None
    hide = [layer, *(other for other in extra if _meets(layer, other))]
    boxes = [_padded(x) for x in hide]
    box = _snap((min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)), step, psd.size)
    full = psd_edit.render_region(psd, box, icc)
    eyes = [(x, x.visible) for x in hide]
    for x in hide:
        x.visible = False
    try:
        plate = psd_edit.render_region(psd, box, icc)
    finally:
        for x, eye in eyes:
            x.visible = eye
    a, b = _rgb(full), _rgb(plate)
    ink = np.abs(a - b).max(-1) > INK
    if not ink.any():
        return None
    off = ~ink
    seam = int(((np.abs(b - png[box[1]:box[3], box[0]:box[2]]).max(-1) > SEAM) & off).sum())
    if seam > SEAM_SHARE * off.sum():
        return None
    ys, xs = np.nonzero(ink)
    return box, like_mode(plate, "RGB"), seam, (box[0] + int(xs.min()), box[1] + int(ys.min()), box[0] + int(xs.max()) + 1, box[1] + int(ys.max()) + 1)


def plates(psd: PSDImage, png: Image.Image, parts: dict) -> Optional[dict]:
    """
    The plates file of a card: {"v", "w", "crops": [{"id", "box", "ink", "at"}], "img"}, or None when no layer gave one.
    box: the crop, ink: where the layer drew in it (canvas px); at: [x, y, w, h] of the crop in img (a data URL). A
    layer with FX layers on it has a second crop without them too, its "fx" their status words (the mock-up's when
    one leaves the text: the bake moves FX layers with their words).

    Args:
        psd: The card
        png: Its PNG (picture 1)
        parts: Its details record's parts (psd_parts.card_parts)
    """
    if psd.width <= 0 or png.size != psd.size:
        return None
    scale = VIEW_W / psd.width
    step = psd.width // gcd(psd.width, VIEW_W)
    step = step if step <= 20 else 1
    icc = psd_edit.export_icc(psd, png)
    by_id = {layer.layer_id: layer for layer in psd.descendants()}
    truth = _rgb(png)
    follow = [f for f in (parts or {}).get("follow") or [] if isinstance(f, dict)]
    found = []
    with variant_view(psd, 1 if len(variants_of(psd)) > 1 else None):
        for lid in dict.fromkeys(layer_ids(parts)):
            layer = by_id.get(lid)
            if layer is None:
                continue
            copies = [by_id[f["id"]] for f in follow if lid in (f.get("of") or []) and f.get("id") in by_id]
            got = crop(psd, layer, truth, icc, step, copies)
            if got:
                found.append(({"id": lid}, *got))
            fx = [x for x in psd_edit.fx_layers(psd, layer) if _meets(layer, x)]
            got = crop(psd, layer, truth, icc, step, [*copies, *fx]) if fx else None
            if got:
                words = [w.strip() for x in fx for w in x.name.split(EFFECT_LAYER_TAG, 1)[1].split("#")[0].split("|") if w.strip()]
                found.append(({"id": lid, "fx": list(dict.fromkeys(words))}, *got))
    if not found:
        return None
    small = [(key, box, ink, image.resize((max(1, round((box[2] - box[0]) * scale)), max(1, round((box[3] - box[1]) * scale))),
                                          Image.LANCZOS)) for key, box, image, _, ink in found]
    sheet = Image.new("RGB", (max(im.width for *_, im in small), sum(im.height + GAP for *_, im in small) - GAP))
    crops, y = [], 0
    for key, box, ink, im in small:
        sheet.paste(im, (0, y))
        crops.append({**key, "box": list(box), "ink": list(ink), "at": [0, y, im.width, im.height]})
        y += im.height + GAP
    buf = BytesIO()
    sheet.save(buf, "AVIF", quality=QUALITY)
    return {"v": VERSION, "w": psd.width, "crops": crops, "img": "data:image/avif;base64," + b64encode(buf.getvalue()).decode()}


# ========================
# The plates branch
# ========================
class Unpublished(RuntimeError):
    """A git failure: publish tries again from the branch head, TRIES times."""


def _git(where: str, *args: str, stdin: Optional[bytes] = None) -> str:
    try:
        return run_process(["git", "-C", where, *args], input=stdin, capture_output=True, check=True, timeout=GIT_TIMEOUT,
                           env={**batch_env(where), **IDENTITY}).stdout.decode("utf-8")
    except CalledProcessError as error:
        raise Unpublished(f"git {args[0]} failed: {(error.stderr or b'').decode('utf-8', 'replace').strip()}") from error
    except (TimeoutExpired, OSError) as error:
        raise Unpublished(f"git {args[0]} failed: {error}") from error


def _files(where: str, head: str) -> dict:
    """{name: blob sha} at the branch head ({} with none)."""
    if not head:
        return {}
    return {line.split("\t", 1)[1]: line.split()[2] for line in _git(where, "ls-tree", "-z", head).split("\0") if "\t" in line}


def psd_blobs(where: str, head: str) -> set:
    """The PSD blobs of head's tree."""
    return {line.split()[2] for line in _git(where, "ls-tree", "-r", "-z", head).split("\0")
            if "\t" in line and line.split("\t", 1)[1].lower().endswith(".psd")}


def collected(paths: list) -> dict:
    """{PSD blob: plates file} of card_checks run results."""
    out = {}
    for path in paths:
        with open(path, encoding="utf-8") as fh:
            out.update({sha: record["plate"] for sha, record in loads(fh.read()).items()
                        if isinstance(record, dict) and record.get("plate")})
    return out


def publish(found: dict, where: str, head: str = "HEAD", remote: str = "origin") -> dict:
    """
    One commit on the plates branch: found ({PSD blob: plates file}) added, those of PSDs not in head's tree dropped;
    none when nothing changed. Built on the branch head, TRIES times from a moved one.

    Args:
        where: A checkout of the card repo (its git and credentials; blobless is enough: --no-thin pushes)
        head: The commit whose PSDs keep their plates

    Returns:
        {"commit" ("" for none), "added", "dropped", "kept"}; raises Unpublished when no push landed
    """
    keep = psd_blobs(where, head)
    new = {f"{sha}.json": _git(where, "hash-object", "-w", "--stdin", stdin=dumps(plate).encode()).strip()
           for sha, plate in found.items() if sha in keep}
    why = ""
    for attempt in range(TRIES):
        if attempt:
            sleep(uniform(1, 2) * attempt)
        try:
            parent = ""
            if _git(where, "ls-remote", "--heads", remote, f"refs/heads/{BRANCH}").split():
                _git(where, "fetch", "-q", "--no-tags", "--no-write-fetch-head", "--depth", "1", "--filter=blob:none", remote,
                     f"+refs/heads/{BRANCH}:{LOCAL_REF}")
                parent = _git(where, "rev-parse", "--verify", f"{LOCAL_REF}^{{commit}}").strip()
            old = _files(where, parent)
            files = {**{name: sha for name, sha in old.items() if name[:-5] in keep}, **new}
            added, dropped = sum(1 for n, s in new.items() if old.get(n) != s), len(set(old) - set(files))
            if parent and files == old:
                return {"commit": "", "added": 0, "dropped": 0, "kept": len(files)}
            tree = _git(where, "mktree", "--missing", stdin="".join(f"100644 blob {s}\t{n}\n" for n, s in sorted(files.items())).encode()).strip()
            commit = _git(where, "commit-tree", tree, *(["-p", parent] if parent else []), "-m",
                          f"Card plates: {added} added, {dropped} dropped").strip()
            _git(where, "push", "-q", "--no-thin", remote, f"{commit}:refs/heads/{BRANCH}")
        except Unpublished as error:            # most often a push refused: the branch moved since the fetch
            why = str(error)
            print(f"card_plates: attempt {attempt + 1} of {TRIES}: {why}", file=stderr)
            continue
        return {"commit": commit, "added": added, "dropped": dropped, "kept": len(files)}
    raise Unpublished(f"not published after {TRIES} tries: {why}")


def main(args: Optional[list] = None) -> int:
    """The command line (module docstring)."""
    parser = ArgumentParser(prog="python -m commands.card_plates", description="The mock-up's plates")
    parser.add_argument("action", choices=("publish",))
    parser.add_argument("results", nargs="*", help="card_checks run's files")
    parser.add_argument("--git", required=True, help="a checkout of the card repo (blobless is enough)")
    parser.add_argument("--head", default="HEAD", help="the commit whose PSDs keep their plates")
    parser.add_argument("--remote", default="origin")
    options = parser.parse_args(list(argv[1:] if args is None else args))
    done = publish(collected(options.results), options.git, options.head, options.remote)
    print(f"Plates: {done['added']} added, {done['dropped']} dropped, {done['kept']} on the branch"
          + (f" ({done['commit'][:8]})." if done["commit"] else ": no change."))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# ========================
# Module Exports
# ========================
__all__ = [
    'BRANCH',
    'Unpublished',
    'VERSION',
    'VIEW_W',
    'collected',
    'crop',
    'layer_ids',
    'main',
    'plates',
    'psd_blobs',
    'publish',
]
