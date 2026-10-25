"""
Variant tags for the PSDs that print several pictures (docs/Future/PARTS_EDITOR.md §9): a layer or group named "... #N"
(" #2,3") shows only in picture N; untagged ones as their eye says; picture 1 is Name.png, N is NameN.png.

    python tools/variant_tags.py --suggest [--clone DIR] [--json PLAN] [PSD ...]
    python tools/variant_tags.py --apply PLAN --clone DIR        renames in the checkout (gated: names only change)

--suggest re-derives each numbered PNG (NameN.png beside Name.psd, no PSD of its own): from the saved eyes (picture 1),
the layers whose eye flips make NameN.png, greedily (each flip the one that brings the render closest; a hidden layer
inside a hidden group flips with the group). A layer whose eye differs between pictures is tagged with the pictures it
shows in; a tagged text layer in the title region is named Name (the parser's title). The fit per picture: the mean
|diff| at 300 px wide and the share of pixels off by more than 24, the tags applied. Prints the plan; --json writes it.
"""
import sys
from argparse import ArgumentParser
from json import dump, load
from os import listdir, replace
from os.path import abspath, basename, dirname, expanduser, isfile, join
from re import IGNORECASE, match

import numpy as np
from PIL import Image
from psd_tools import PSDImage

sys.path.insert(0, dirname(dirname(abspath(__file__))))

from commands.card_models import card_name  # noqa: E402
from commands.psd_parser import EFFECT_LAYER_TAG, TITLE_REGION_RATIO, PSDParser, layer_tag, variant_view  # noqa: E402

OFF = 24                    # a channel this far off: a pixel that differs
FIT_WIDTH = 300             # the mean |diff| is taken at this width
MOVES = 8                   # flips per picture, at most
GAIN = 0.05                 # a flip must lower the mean |diff| by this much
EXCLUDE = ("MDW", "Markers")


def _rgb(image) -> np.ndarray:
    """An RGB float array, alpha over white (as the card shows)."""
    rgba = np.asarray(image.convert("RGBA"), dtype=np.float32)
    alpha = rgba[:, :, 3:4] / 255.0
    return rgba[:, :, :3] * alpha + 255.0 * (1 - alpha)


def _eye(layer, flips: set) -> bool:
    return bool(layer.visible) != (layer.layer_id in flips)


def _filter(flips: set):
    """The composite's layer filter with these eyes flipped."""
    def shown(layer) -> bool:
        node = layer
        while node is not None and getattr(node, "kind", "psdimage") != "psdimage":
            if not _eye(node, flips):
                return False
            node = node.parent
        return True
    return shown


def _render(psd, box, flips: set) -> np.ndarray:
    return _rgb(psd.composite(viewport=box, ignore_preview=True, apply_icc=True, layer_filter=_filter(flips)))


def _error(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a - b).mean())


def _fit(render: np.ndarray, png: np.ndarray) -> tuple:
    """(mean |diff| at FIT_WIDTH px, share of pixels off by more than OFF)."""
    h = max(1, round(png.shape[0] * FIT_WIDTH / png.shape[1]))
    small = lambda a: np.asarray(Image.fromarray(a.clip(0, 255).astype(np.uint8)).resize((FIT_WIDTH, h), Image.LANCZOS),
                                 dtype=np.float32)
    off = (np.abs(render - png) > OFF).any(axis=2).mean()
    return (round(_error(small(render), small(png)), 3), round(float(off) * 100, 3))


def _diff_box(a: np.ndarray, b: np.ndarray, pad: int = 8):
    ys, xs = (np.abs(a - b) > OFF).any(axis=2).nonzero()
    if not len(xs):
        return None
    h, w = a.shape[:2]
    return (max(0, int(xs.min()) - pad), max(0, int(ys.min()) - pad), min(w, int(xs.max()) + 1 + pad),
            min(h, int(ys.max()) + 1 + pad))


def _ancestors(layer) -> list:
    out, node = [], layer.parent
    while node is not None and getattr(node, "kind", "psdimage") != "psdimage":
        out.append(node)
        node = node.parent
    return out


def _box(layer) -> tuple:
    """The layer's box; a group's is its leaves' (a hidden group's own box is empty)."""
    if not layer.is_group():
        return tuple(layer.bbox)
    boxes = [tuple(leaf.bbox) for leaf in layer.descendants() if not leaf.is_group() and leaf.bbox[2] > leaf.bbox[0]]
    if not boxes:
        return (0, 0, 0, 0)
    return (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))


def _overlap(a: tuple, b: tuple) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _moves(psd, box) -> list:
    """
    Candidate flips: [(layer, {layer ids})], each node over the box (adjustment layers anywhere), a hidden one with its
    hidden groups; and a swap: a hidden node shown with the shown siblings it covers hidden (one title for another).
    """
    out = []
    for layer in psd.descendants():
        if EFFECT_LAYER_TAG in layer.name:
            continue
        own = _box(layer)
        empty = own[2] <= own[0] or own[3] <= own[1]
        if empty and (layer.is_group() or layer.kind in ("pixel", "smartobject", "type", "shape")):
            continue
        if not empty and not _overlap(own, box):
            continue
        ids = {layer.layer_id} | ({a.layer_id for a in _ancestors(layer) if not a.visible} if not layer.visible else set())
        out.append((layer, frozenset(ids)))
        if not layer.visible and not empty:
            covered = {other.layer_id for other in layer.parent if other is not layer and other.visible
                       and other.kind == layer.kind and _overlap(_box(other), own)}
            if covered:
                out.append((layer, frozenset(ids | covered)))
    return out


def suggest_psd(path: str, rel: str, pngs: dict, parser: PSDParser) -> dict:
    """
    One PSD's plan: {"renames": {lyid: new name}, "pictures": {N: {"png", "flips", "fit", "title", "ability", ...}}}.

    Args:
        path: The PSD on disk
        rel: Its repo path
        pngs: {N: PNG path on disk}, 1 = Name.png
        parser: For the per-picture parse (what each shows)
    """
    psd = PSDImage.open(path)
    w, h = psd.size
    whole = (0, 0, w, h)
    targets = {n: _rgb(Image.open(p)) for n, p in pngs.items()}
    base = _render(psd, whole, set())
    flips = {1: set()}
    for n in sorted(targets):
        if n == 1:
            continue
        box = _diff_box(targets[1], targets[n])
        chosen = set()
        if box is not None:
            want = targets[n][box[1]:box[3], box[0]:box[2]]
            err = _error(_render(psd, box, chosen), want)
            moves = _moves(psd, box)
            for _ in range(MOVES):
                best = None
                for layer, ids in moves:
                    trial = chosen ^ ids
                    e = _error(_render(psd, box, trial), want)
                    if best is None or e < best[0]:
                        best = (e, trial)
                if best is None or err - best[0] < GAIN:
                    break
                err, chosen = best
        flips[n] = chosen
    by_id = {layer.layer_id: layer for layer in psd.descendants()}
    count = max(targets)
    renames = {}
    for lyid in sorted(set().union(*flips.values())):
        layer = by_id[lyid]
        shows = [n for n in range(1, count + 1) if _eye(layer, flips.get(n, set()))]
        base_name = layer_tag(layer.name)[0]
        if layer.kind == "type" and layer.bbox[1] < h * TITLE_REGION_RATIO:
            base_name = "Name"                      # a picture's own title: the parser's Name layer
        renames[str(lyid)] = f"{base_name} #{','.join(str(n) for n in shows)}"
    old = {lyid: by_id[int(lyid)].name for lyid in renames}
    for lyid, name in renames.items():
        by_id[int(lyid)].name = name
    pictures = {}
    for n, png in sorted(pngs.items()):
        with variant_view(psd, n):
            render = _render(psd, whole, set())
            card = parser.parse(psd, rel)
        pictures[str(n)] = {"png": basename(png), "flips": sorted(flips.get(n, ())), "fit": _fit(render, targets[n]),
                            "title": card.display_name or card_name(rel), "ability": card.ability, "stars": card.stars,
                            "types": card.types, "shown": sorted(lyid for lyid, name in renames.items()
                                                                  if n in layer_tag(name)[1])}
    baseline = _fit(base, targets[1])
    return {"path": rel, "renames": {lyid: [old[lyid], name] for lyid, name in renames.items()}, "baseline": baseline,
            "pictures": pictures}


def multi_picture(clone: str) -> dict:
    """{PSD repo path: {N: PNG repo path}} for the PSDs beside numbered PNGs with no PSD of their own."""
    from subprocess import run
    paths = run(["git", "-C", clone, "ls-files", "--", "*.png", "*.psd"], capture_output=True, text=True, check=True).stdout.splitlines()
    lower = {p.lower() for p in paths}
    out: dict = {}
    for path in paths:
        if not path.lower().endswith(".png") or path.split("/")[0] in EXCLUDE + ("Types", "site", "TTS") or "Rulebook" in path:
            continue
        if path[:len(path) - 4].lower() + ".psd" in lower:
            continue
        m = match(r"(.+?)(\d+)\.png$", path, IGNORECASE)
        if m and (m.group(1) + ".psd").lower() in lower:
            psd = next(p for p in paths if p.lower() == (m.group(1) + ".psd").lower())
            out.setdefault(psd, {1: psd[:len(psd) - 4] + ".png"})[int(m.group(2))] = path
    return out


def offline(clone: str) -> PSDParser:
    from commands.psd_edit import offline_parser
    types = [p[:len(p) - 4].lower() for p in listdir(join(clone, "Types")) if p.lower().endswith(".png")]
    stars = [p[:len(p) - 4].lower().replace(" ", "_") for p in listdir(join(clone, "Types", "Stars"))
             if p.lower().endswith(".png")]
    with open(join(clone, "MDW/Fonts/stoogedings-src/stoogedings.json"), encoding="utf-8") as fh:
        manifest = load(fh)
    return offline_parser(types, stars, manifest)


def apply(plan: dict, clone: str) -> list:
    """Each PSD's renames in place, gated (card_standards.rename_layers: the names alone change). Returns the lines."""
    from commands.card_standards import rename_layers
    lines = []
    for rel, entry in plan.items():
        path = join(clone, rel)
        tmp = path + ".tags.psd"
        done = rename_layers(path, tmp, {int(lyid): pair[1] for lyid, pair in entry["renames"].items()})
        if done:
            replace(tmp, path)
        lines.append(f"{rel}: {'; '.join(done)}")
    return lines


def main(args=None) -> int:
    opts = ArgumentParser(description=__doc__.split("\n\n")[0])
    opts.add_argument("--suggest", action="store_true")
    opts.add_argument("--apply", metavar="PLAN")
    opts.add_argument("--clone", default=expanduser("~/Desktop/TTSCardMaker"))
    opts.add_argument("--json", metavar="PLAN")
    opts.add_argument("psds", nargs="*", help="repo paths (default: every multi-picture PSD)")
    opts = opts.parse_args(args)
    if opts.apply:
        with open(opts.apply, encoding="utf-8") as fh:
            plan = load(fh)
        print("\n".join(apply(plan, opts.clone)))
        return 0
    if not opts.suggest:
        opts.print_help()
        return 2
    found = multi_picture(opts.clone)
    parser = offline(opts.clone)
    plan = {}
    for rel, pngs in sorted(found.items()):
        if opts.psds and rel not in opts.psds:
            continue
        if not all(isfile(join(opts.clone, p)) for p in [rel, *pngs.values()]):
            print(f"{rel}: not checked out; skipped")
            continue
        plan[rel] = suggest_psd(join(opts.clone, rel), rel, {n: join(opts.clone, p) for n, p in pngs.items()}, parser)
        entry = plan[rel]
        print(f"{rel} (baseline {entry['baseline']})")
        for lyid, (old, new) in entry["renames"].items():
            print(f"  {lyid} {old!r} -> {new!r}")
        for n, pic in entry["pictures"].items():
            print(f"  picture {n} {pic['png']}: fit {pic['fit']}, title {pic['title']!r}, stars {pic['stars']}, "
                  f"types {pic['types']}")
    if opts.json:
        with open(opts.json, "w", encoding="utf-8") as fh:
            dump(plan, fh, indent=1, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
