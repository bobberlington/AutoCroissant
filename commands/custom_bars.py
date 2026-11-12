"""
Croissant's Isaac-style stat bars built for other cards (custombars, 2026-11-11). Each stat becomes one smart object
named Health/Defense/Attack/Speed inside the stat's own group: a 'Dark' row under a 'Light' group with knockout and
Transparency Shapes Layer off, whose unlinked mask the layer comps "0".."10" move, so the light row replaces the dark
one up to the value's cut. The rows are the card's own looks full and at 0, flattened by Photoshop, so every value
draws as the card drew it (comp_render's comp_document path bakes them).

Run by hand on the Mac (Photoshop 2023 for build and finish, on copies under ~/std-work-1107):

    python -m commands.custom_bars plan   OUT card.psd... [--rels=FILE]  rows, cuts, row documents (OUT/<stem>/)
    python -m commands.custom_bars build  OUT              OUT/build.jsx: the smart objects, masks and comps
    python -m commands.custom_bars tag    OUT              knockout and shapes layer off on each Light group
    python -m commands.custom_bars finish OUT              OUT/finish.jsx: Photoshop redraws each bar and saves;
                                                           renders every comp for verify
    python -m commands.custom_bars strip  OUT              final.psd -> card.psd without the merged image (the repo's)
    python -m commands.custom_bars verify OUT              parse, renders, comps against Photoshop's, PNG export
"""
import json
import sys
from io import BytesIO
from os import makedirs, remove
from os.path import abspath, basename, dirname, join, splitext

import numpy as np
from PIL import Image
from psd_tools import PSDImage
from psd_tools.constants import Tag

from commands import psd_edit
from commands.psd_edit import EFFECT_BLOCKS, Refused
from commands.psd_regions import comp_bars

STATS = ("hp", "def", "atk", "spd")
SO_NAMES = {"hp": "Health", "def": "Defense", "atk": "Attack", "spd": "Speed"}
VALUES = range(11)
SEEN = 2                    # a column differs past this premultiplied level (the fork against itself)
PAD = 16                    # px round a stat group's box for its row renders
NEAR = 400                  # px a straight cut may take from the wrong row at a value the card does not show

# Each card: its stat groups (paths from the root), extra layers a row draws (Green Mage's dark overlay bars), layers
# deleted with the old bars, its recipe for the stat at n when AC's own edit is not the card's look, and bands: each
# bar holds everything drawn in its own band of the card (rows whose glows reach the next row, under its dark overlay).
CARDS = {
    "Hive_Knight": {"groups": "Stats Bars/Bars/{Hp,Def,Atk,Spd}"},
    "False_Knight": {"groups": "Stats Bars/Darks/{Hp,DEF,ATK,SPD}", "recipe": "false_knight"},
    "Chris_&_Jordi": {"groups": "Stats Bars/Bars/{hp,def,atk,spd}", "recipe": "chris"},
    "Green_Mage": {"groups": "Stats Bars/Bars/{hp,Def,ATK,SPD}", "bands": True,
                   "extra": "Stats Bars/Darks/{Health,Defense,Attack,Speed}", "empty": ["Stats Bars/Darks"]},
    "Grimora": {"groups": "Text/bars/{hp,def,atk,spd}"},
    "Skeleton_Mate": {"groups": "Text/Darks/{HP,DEF,ATK,SPD}"},
    "Minecraft_Panda": {"groups": "Stats Bars/Darks/{HP,DEF,ATK,SPD}"},
    "Tainted_Keeper": {"groups": "Stats Bars/Bars/{HP,DEF,ATK,SPD}"},
    "Uumuu": {"groups": "Stats Bars/Bars/{hp,Def,ATK,SPD}", "delete": ["Stats Bars/Darks"]},
    "Zigg": {"groups": "Stats Bars/Bars/{hp,Def,ATK,SPD}"},
}


def _paths(spec: str) -> dict:
    """"A/B/{w,x,y,z}" -> {stat: "A/B/w", ...} in STATS order."""
    head, names = spec.split("{")
    return dict(zip(STATS, (head + name for name in names.rstrip("}").split(","))))


def find(psd, path: str, group: bool = True):
    """
    The layer at a name path, each name unique among its siblings of its kind: groups on the way (and at the end
    unless group is False), as the JSX finds them (layerSets, artLayers).
    """
    layer, names = psd, path.split("/")
    for i, name in enumerate(names):
        want = group or i < len(names) - 1
        hits = [child for child in layer if child.name == name and child.is_group() == want]
        if len(hits) != 1:
            raise ValueError(f"{path}: {len(hits)} {'groups' if want else 'layers'} named {name!r}")
        layer = hits[0]
    return layer


def ancestors(layer) -> list:
    out = []
    while layer.parent is not None and layer.parent.parent is not None:
        layer = layer.parent
        out.append(layer)
    return out


def _subtree(layer) -> set:
    return {id(layer), *(id(child) for child in layer.descendants())} if layer.is_group() else {id(layer)}


def _strip(layer) -> None:
    """A layer's effects removed (its rows are drawn without them: the group keeps them on the new bar)."""
    blocks = layer._record.tagged_blocks
    for key in EFFECT_BLOCKS:
        if blocks is not None and key in blocks:
            del blocks[key]


# ========================
# Recipes: the stat at n
# ========================
def _plain(psd, parsed, stat, n, groups):
    if n != parsed["card"][stat]:
        psd_edit.set_stat(psd, parsed, stat, n)


def _false_knight(psd, parsed, stat, n, groups):
    """Cell i lit: its filled sprite in the lit cells' style (the hidden ones carry their overlay at 100%, the shown
    46%), and its empty as now (hidden past the value); unlit: the empty without the Soft Light overlay the empties
    under lit cells carry (the look the PNG shows), filled hidden."""
    from copy import deepcopy
    group = find(psd, groups[stat])
    value = parsed["card"][stat]
    shelf = find(group, "Empty")
    empty, filled = (sorted(find(group, name), key=lambda c: c.bbox[0]) for name in ("Empty", "Filled"))
    shown = [e.visible and shelf.visible for e in empty]
    lit = next(f for f in [*filled, *(c for g in groups.values() for c in find(psd, g + "/Filled"))] if f.visible)
    shelf.visible = True
    for i, (e, f, was) in enumerate(zip(empty, filled, shown), 1):
        if i <= n and not f.visible:
            _strip(f)
            for key in EFFECT_BLOCKS:
                if key in lit._record.tagged_blocks:
                    f._record.tagged_blocks[key] = deepcopy(lit._record.tagged_blocks[key])
        f.visible = i <= n
        e.visible = was and i <= value if i <= n else True
        if i > n:
            _strip(e)


def _chris(psd, parsed, stat, n, groups):
    """AC's edit, and cells 1 and 10's pictures (clipped to the bar base, not the cell) follow them."""
    _plain(psd, parsed, stat, n, groups)
    group = find(psd, groups[stat])
    find(group, "Layer 14", False).visible = n >= 1
    find(group, "Layer 14 copy", False).visible = n >= 10


RECIPES = {"plain": _plain, "false_knight": _false_knight, "chris": _chris}


# ========================
# plan
# ========================
def _row_doc(psd, groups: dict, extra: dict, stats) -> None:
    """psd edited to draw only `stats`' layers: the rest hidden, the stat groups' and their ancestors' effects off."""
    keep, chain = set(), set()
    for stat in stats:
        for layer in [find(psd, groups[stat])] + ([find(psd, extra[stat], False)] if stat in extra else []):
            keep |= _subtree(layer)
            for owner in ancestors(layer):
                chain.add(id(owner))
                _strip(owner)
            if layer.is_group():
                _strip(layer)
    for layer in psd.descendants():
        if id(layer) not in keep and id(layer) not in chain and not any(id(a) in keep for a in ancestors(layer)):
            layer.visible = False


def _bands(psd, groups: dict) -> dict:
    """{stat: (0, y0, width, y1)}: the card cut across halfway between the stat groups' rows, top to bottom."""
    order = sorted(STATS, key=lambda stat: find(psd, groups[stat]).bbox[1])
    cuts = [0] + [(find(psd, groups[a]).bbox[3] + find(psd, groups[b]).bbox[1]) // 2
                  for a, b in zip(order, order[1:])] + [psd.height]
    return {stat: (0, cuts[i], psd.width, cuts[i + 1]) for i, stat in enumerate(order)}


def _render(psd, box) -> np.ndarray:
    """Premultiplied float RGBA of what shows in box over transparency."""
    im = psd.composite(viewport=box, ignore_preview=True, apply_icc=False, color=0.0, alpha=0.0)
    a = np.asarray(im.convert("RGBA"), np.float32)
    return np.concatenate([a[..., :3] * a[..., 3:] / 255.0, a[..., 3:]], axis=2)


def _cuts(rows: dict) -> dict:
    """Per value: the straight cut range [lo, hi] (row-box x) where light left / dark right rebuilds row n, else the
    per-pixel-row ranges ("rows": [[lo, hi] or None])."""
    light, dark = rows[10], rows[0]
    width = light.shape[1]
    out = {}
    for n, row in rows.items():
        left = np.abs(light - row).max(axis=2) > SEEN      # rows x cols: the light row differs from row n
        right = np.abs(dark - row).max(axis=2) > SEEN
        okl = np.concatenate([np.ones((left.shape[0], 1), bool), np.cumsum(left, axis=1) == 0], axis=1)
        okr = np.concatenate([np.cumsum(right[:, ::-1], axis=1)[:, ::-1] == 0, np.ones((right.shape[0], 1), bool)],
                             axis=1)
        whole = okl.all(axis=0) & okr.all(axis=0)
        xs = np.nonzero(whole)[0]
        per = []
        for y in range(left.shape[0]):
            ys = np.nonzero(okl[y] & okr[y])[0]
            per.append([int(ys[0]), int(ys[-1])] if len(ys) else None)
        if len(xs) and xs[-1] - xs[0] == len(xs) - 1:
            out[n] = {"lo": int(xs[0]), "hi": int(xs[-1]), "rows": per}
            continue
        # the best straight cut: fewest pixels from the wrong row
        bad = (np.concatenate([[0], np.cumsum(left.sum(axis=0))])
               + np.concatenate([np.cumsum(right.sum(axis=0)[::-1])[::-1], [0]]))
        best = int(np.argmin(bad))
        out[n] = {"rows": per, "width": width, "best": best, "bad": int(bad[best])}
    return out


def _profile(cuts: dict, height: int, width: int, value: int) -> dict:
    """
    One mask edge for every value: straight when every value has a cut; else a per-row shape s(y) and offsets o_n
    with each row's cut inside its range (_shaped); else straight with all but a few pixels (NEAR) right at values the
    card does not show now.
    """
    if all("lo" in cut for cut in cuts.values()):
        return {"kind": "straight", "x": {n: (cut["lo"] + cut["hi"]) // 2 for n, cut in cuts.items()}}
    try:
        return _shaped(cuts, height, width)
    except Refused:
        if not all("lo" in cut or (n != value and cut["bad"] <= NEAR) for n, cut in cuts.items()):
            raise
    return {"kind": "straight", "x": {n: (cut["lo"] + cut["hi"]) // 2 if "lo" in cut else cut["best"]
                                      for n, cut in cuts.items()},
            "near": {n: cut["bad"] for n, cut in cuts.items() if "lo" not in cut}}


def _shaped(cuts: dict, height: int, width: int) -> dict:
    ranges = {}
    for n, cut in cuts.items():
        per = cut["rows"]
        if any(r is None for r in per):
            raise Refused("unsupported-card", f"value {n}: a pixel row has no cut")
        ranges[n] = per
    inner = [n for n in ranges if 0 < n < 10]

    def tight(n, y):
        return ranges[n][y][1] - ranges[n][y][0] < width - 1
    ref = max(inner, key=lambda n: sum(tight(n, y) for y in range(height)))
    room = {y: list(ranges[ref][y]) for y in range(height) if tight(ref, y)}   # where s(y) may lie
    offsets = {ref: 0}
    for n in sorted(inner, key=lambda n: abs(n - ref)):
        if n == ref:
            continue
        both = [y for y in range(height) if tight(n, y) and y in room]
        lo = max((ranges[n][y][0] - room[y][1] for y in both), default=0)
        hi = min((ranges[n][y][1] - room[y][0] for y in both), default=0)
        if lo > hi:
            raise Refused("unsupported-card", f"no one mask edge for value {n}")
        offsets[n] = d = (lo + hi) // 2
        for y in range(height):
            if tight(n, y):
                a, b = ranges[n][y][0] - d, ranges[n][y][1] - d
                room[y] = [max(a, room[y][0]), min(b, room[y][1])] if y in room else [a, b]
    centre = int(np.median([(a + b) // 2 for a, b in room.values()])) if room else 0
    shape = [min(max(centre, room[y][0]), room[y][1]) if y in room else centre for y in range(height)]
    offsets[0] = min(b - v for (a, b), v in zip(ranges[0], shape))
    offsets[10] = max(a - v for (a, b), v in zip(ranges[10], shape))
    for n, per in ranges.items():
        if not all(a <= v + offsets[n] <= b for (a, b), v in zip(per, shape)):
            raise Refused("unsupported-card", f"value {n}: the one mask edge misses a row")
    base = min(offsets[n] for n in inner)
    shape = [v + base for v in shape]
    offsets = {n: o - base for n, o in offsets.items()}
    reach = max(v + offsets[10] for v in shape) - max(a for a, b in ranges[10])
    return {"kind": "shaped", "shape": shape, "offsets": offsets, "margin": max(1, reach + 2)}


def plan_card(parser, path: str, rel: str, out: str) -> dict:
    stem = splitext(basename(path))[0]
    config = CARDS[stem]
    groups = _paths(config["groups"])
    extra = _paths(config["extra"]) if "extra" in config else {}
    recipe = RECIPES[config.get("recipe", "plain")]
    raw = open(path, "rb").read()
    base = PSDImage.open(BytesIO(raw))
    parsed = psd_edit.parse_card(parser, path, rel, base)
    values = {stat: parsed["card"][stat] for stat in STATS}
    bands = _bands(base, groups) if config.get("bands") else {}
    shown = lambda stat: STATS if bands else [stat]     # a band's rows draw every stat's layers
    folder = join(out, stem)
    makedirs(folder, exist_ok=True)
    with open(join(folder, "T.psd"), "wb") as fh:
        fh.write(raw)
    stats = {}
    for stat in STATS:
        group = find(base, groups[stat])
        layers = [group] + ([find(base, extra[stat], False)] if stat in extra else [])
        x0 = min(l.bbox[0] for l in layers if l.bbox[2] > l.bbox[0])
        y0 = min(l.bbox[1] for l in layers if l.bbox[2] > l.bbox[0])
        x1 = max(l.bbox[2] for l in layers if l.bbox[2] > l.bbox[0])
        y1 = max(l.bbox[3] for l in layers if l.bbox[2] > l.bbox[0])
        box = (max(0, x0 - PAD), max(0, y0 - PAD), min(base.width, x1 + PAD), min(base.height, y1 + PAD))
        if stat in bands:
            box = (box[0], max(box[1], bands[stat][1]), box[2], min(box[3], bands[stat][3]))
        rows = {}
        for n in VALUES:
            psd = PSDImage.open(BytesIO(raw))
            recipe(psd, parsed, stat, n, groups)
            _row_doc(psd, groups, extra, shown(stat))
            rows[n] = _render(psd, box)
        cuts = _cuts(rows)
        edge = _profile(cuts, box[3] - box[1], box[2] - box[0], values[stat])
        if edge["kind"] == "straight":
            edge["x"] = {n: box[0] + x for n, x in edge["x"].items()}
        else:
            edge["offsets"] = {n: box[0] + o for n, o in edge["offsets"].items()}
            edge["top"] = box[1]
        stats[stat] = {"group": groups[stat], "extra": extra.get(stat), "name": SO_NAMES[stat], "value": values[stat],
                       "box": box, "edge": edge, "band": bands.get(stat),
                       "cuts": {n: ({"lo": c["lo"], "hi": c["hi"]} if "lo" in c else {"best": c["best"], "bad": c["bad"]})
                                for n, c in cuts.items()}}
    for state, n in (("F", 10), ("E", 0)):
        for stat in STATS:
            psd = PSDImage.open(BytesIO(raw))
            recipe(psd, parsed, stat, n, groups)
            _row_doc(psd, groups, extra, shown(stat))
            psd.save(join(folder, f"R{state}_{stat}.psd"))
    card = {"stem": stem, "rel": rel, "path": abspath(path), "folder": abspath(folder), "values": values,
            "stats": stats, "delete": config.get("delete", []), "empty": config.get("empty", [])}
    with open(join(folder, "plan.json"), "w") as fh:
        json.dump(card, fh, indent=1)
    return card


# ========================
# build, tag, finish: Photoshop and psd-tools
# ========================
def _cards(out: str) -> list:
    from glob import glob
    return [json.load(open(path)) for path in sorted(glob(join(out, "*", "plan.json")))]


def write_jsx(out: str, step: str, cards: list) -> str:
    """OUT/<step>.jsx from the template beside this file, its log OUT/<step>.log."""
    with open(join(dirname(abspath(__file__)), "custom_bars.jsx")) as fh:
        template = fh.read()
    plan = {"log": join(abspath(out), f"{step}.log"), "step": step, "cards": cards}
    path = join(out, f"{step}.jsx")
    with open(path, "w") as fh:
        fh.write(template.replace("/*PLAN*/", json.dumps(plan)))
    return path


def tag_card(card: dict) -> list:
    """built.psd -> tagged.psd: each bar's Light group knocks out (shallow) with Transparency Shapes Layer off, as
    Croissant set his (Photoshop 2023's scripting does not set them; it opens, draws and keeps these)."""
    psd = PSDImage.open(join(card["folder"], "built.psd"))
    done = []
    for stat, s in card["stats"].items():
        layer = find(psd, f"{s['group']}/{s['name']}", False)
        doc = PSDImage.open(BytesIO(layer.smart_object.data))
        light = find(doc, "Light")
        if not light.is_group():
            raise ValueError(f"{card['stem']} {stat}: 'Light' is not a group")
        light.tagged_blocks.set_data(Tag.KNOCKOUT_SETTING, 1)
        light.tagged_blocks.set_data(Tag.TRANSPARENCY_SHAPES_LAYER, 0)
        buf = BytesIO()
        doc.save(buf)
        layer.smart_object._data.data = buf.getvalue()
        done.append(stat)
    psd.mark_updated()
    psd.save(join(card["folder"], "tagged.psd"))
    back = PSDImage.open(join(card["folder"], "tagged.psd"))
    for stat, s in card["stats"].items():
        light = find(PSDImage.open(BytesIO(find(back, f"{s['group']}/{s['name']}", False).smart_object.data)), "Light")
        if (light.tagged_blocks.get_data(Tag.KNOCKOUT_SETTING, 0), light.tagged_blocks.get_data(
                Tag.TRANSPARENCY_SHAPES_LAYER, 1)) != (1, 0):
            raise ValueError(f"{card['stem']} {stat}: the tags did not stick")
    return done


# ========================
# verify
# ========================
def _rgb(image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), np.int16)


def _measure(a: np.ndarray, b: np.ndarray, box=None) -> dict:
    if box is not None:
        a, b = a[box[1]:box[3], box[0]:box[2]], b[box[1]:box[3], box[0]:box[2]]
    d = np.abs(a - b).max(axis=2)
    return {"max": int(d.max()), "px>2": int((d > 2).sum()), "px>8": int((d > 8).sum())}


def _bars_box(card: dict, psd) -> tuple:
    boxes = [b for bars in comp_bars(psd).values() for _, _, b in bars] + [tuple(s["box"]) for s in card["stats"].values()]
    return (max(0, min(b[0] for b in boxes) - 8), max(0, min(b[1] for b in boxes) - 8),
            min(psd.width, max(b[2] for b in boxes) + 8), min(psd.height, max(b[3] for b in boxes) + 8))


def _draw(psd) -> np.ndarray:
    return _rgb(psd.composite(ignore_preview=True, apply_icc=False))


def verify_card(parser, card: dict, clone: str, name: str = "card.psd", bakes=(0, 3, 10)) -> dict:
    """
    The converted card against the old one and Photoshop: the parse (stats, problems); our render against the old
    card's (whole card and the bars); Photoshop's saved render (ps_saved.png) and each comp's (ps_<n>.png, every bar
    at n) against AC's set_comp at n; a bake (apply_jobs) at each of `bakes` against Photoshop's; png_export's
    verdict on the card's PNG.
    """
    from os.path import exists
    from commands import png_export
    folder, rel = card["folder"], card["rel"]
    old_path, new_path = join(folder, "T.psd"), join(folder, name)
    old, new = PSDImage.open(old_path), PSDImage.open(new_path)
    was, now = psd_edit.parse_card(parser, old_path, rel, old), psd_edit.parse_card(parser, new_path, rel, new)
    out = {"card": card["stem"], "parse": {stat: now["card"][stat] for stat in STATS},
           "parse_same": now["card"] == was["card"], "problems_same": now["validator"] == was["validator"]}
    if not out["parse_same"]:
        out["parse_diff"] = {k: (was["card"].get(k), v) for k, v in now["card"].items() if was["card"].get(k) != v}
    box = _bars_box(card, new)
    mine, before = _draw(new), _draw(old)
    out["vs_old"] = {"card": _measure(mine, before), "bars": _measure(mine, before, box), "bars_box": box}
    shots = {n: join(folder, f"ps_{n}.png") for n in [*VALUES, "saved"]}
    png_path = card["path"][:-4] + ".png"          # the PNG beside the PSD the plan read (main's)
    old_png = Image.open(png_path)
    if exists(shots["saved"]):
        shot = Image.open(shots["saved"])
        out["vs_ps_saved"] = _measure(mine, _rgb(shot))
        exported = png_export.like_mode(png_export.convert(new, shot.convert("RGBA"), True), old_png.mode)
        out["ps_vs_png"] = {"card": _measure(_rgb(exported), _rgb(old_png)),
                            "bars": _measure(_rgb(exported), _rgb(old_png), box)}
    comps = {}
    for n in VALUES:
        if not exists(shots[n]):
            continue
        psd = PSDImage.open(new_path)
        try:
            for stat in STATS:
                psd_edit.set_comp(psd, now, stat, n)
        except Refused as refused:
            comps[n] = {"refused": refused.why}
            continue
        drawn, truth = _draw(psd), _rgb(Image.open(shots[n]))
        comps[n] = {"card": _measure(drawn, truth), "bars": _measure(drawn, truth, box)}
    out["comps"] = comps
    verdict = png_export.export(old, new, old_png, data=True)
    out["png"] = {"outcome": verdict.outcome, "why": verdict.why, "changed_px": verdict.changed_px}
    if verdict.outcome == "re-exported":
        with open(join(folder, "final.png"), "wb") as fh:
            fh.write(verdict.data)
    elif exists(join(folder, "final.png")):
        remove(join(folder, "final.png"))
    out["bakes"] = {}
    for n in bakes:
        if not exists(shots[n]):
            continue
        wanted = {stat: n for stat in STATS}
        plan = psd_edit.classify(now, wanted, new)
        work = join(folder, f"bake_{n}")
        makedirs(work, exist_ok=True)
        try:
            baked = psd_edit.apply_jobs(new_path, png_path, rel, plan.jobs, now, parser, work,
                                        join(clone, "MDW", "Fonts"), psd=new)
        except Refused as refused:
            out["bakes"][n] = {"refused": refused.why, "jobs": plan.jobs}
            continue
        drawn = _draw(PSDImage.open(baked.psd_path))
        out["bakes"][n] = {"jobs": plan.jobs, "reparse": {s: baked.reparse["card"][s] for s in STATS},
                           "vs_ps": _measure(drawn, _rgb(Image.open(shots[n]))),
                           "bars": _measure(drawn, _rgb(Image.open(shots[n])), box)}
    return out


def restore_links(new, old) -> int:
    """The linked files' records (lnkE) as the card had them: Photoshop saved the copy away from the repo, so it wrote
    each link's relPath as a file URL (the guards would repair them); the same uuid, name and size get the old
    descriptor back."""
    from copy import deepcopy
    def items(psd):
        blocks = psd._record.layer_and_mask_information.tagged_blocks
        return (blocks.get_data(Tag.LINKED_LAYER_EXTERNAL) or []) if blocks is not None else []
    was = {str(item.uuid): item for item in items(old)}
    kept = 0
    for item in items(new):
        before = was.get(str(item.uuid))
        if before is not None and (before.filename, before.filesize) == (item.filename, item.filesize) \
                and before.linked_file != item.linked_file:
            item.linked_file = deepcopy(before.linked_file)
            kept += 1
    return kept


FILLS = ("solidcolorfill", "gradientfill", "patternfill")


def restore_fills(new, old) -> list:
    """Fill layers as the card stored them: Photoshop's save gives an untouched one its pixels (Hive Knight's 'Gradient
    Fill 1': 37 KB -> 4 MB) where the card kept none; every reader draws a fill from its settings. The same layer
    (id, name, mask) gets its old rectangle, channels and settings back."""
    from copy import deepcopy
    was = {layer.layer_id: layer for layer in old.descendants() if layer.kind in FILLS}
    size = lambda layer: sum(len(channel.data) for channel in layer._channels)
    kept = []
    for layer in new.descendants():
        before = was.get(layer.layer_id)
        if before is None or (layer.kind, layer.name) != (before.kind, before.name) or size(layer) <= size(before):
            continue
        now, then = ([(r.mask_data.top, r.mask_data.left, r.mask_data.bottom, r.mask_data.right) if r.mask_data else None,
                      [c.id for c in r.channel_info]] for r in (layer._record, before._record))
        if now != then:                         # the same mask and channels
            continue
        record, source = layer._record, before._record
        record.top, record.left, record.bottom, record.right = source.top, source.left, source.bottom, source.right
        record.channel_info = deepcopy(source.channel_info)
        layer._channels[:] = deepcopy(before._channels)     # in place: the document's channel list holds it
        layer._invalidate_bbox()
        for key in (Tag.SOLID_COLOR_SHEET_SETTING, Tag.GRADIENT_FILL_SETTING, Tag.PATTERN_FILL_SETTING):
            if key in before.tagged_blocks and key in layer.tagged_blocks:
                layer.tagged_blocks[key] = deepcopy(before.tagged_blocks[key])
        kept.append(layer.name)
    return kept


def parser_for(clone: str):
    from commands.card_check import clone_vocabulary
    return clone_vocabulary(clone).parser()


def _rel(path: str, clone: str) -> str:
    from os.path import relpath
    return relpath(abspath(path), abspath(clone)) if abspath(path).startswith(abspath(clone)) else path


def main(argv: list) -> int:
    from os import environ
    from os.path import expanduser
    clone = environ.get("AUTOCROISSANT_LOCAL_DIR", expanduser("~/Desktop/TTSCardMaker"))
    command, out = argv[0], argv[1]
    flags = dict(arg[2:].split("=", 1) for arg in argv[2:] if arg.startswith("--"))
    if command == "plan":                       # --rels=FILE: lines ending in a repo path, for copies kept elsewhere
        parser = parser_for(clone)
        rels = {}
        for line in open(flags["rels"]) if "rels" in flags else ():
            rel = line.rstrip("\n").split("\t")[-1]
            rels[basename(rel)] = rel
        for path in [arg for arg in argv[2:] if not arg.startswith("--")]:
            card = plan_card(parser, path, rels.get(basename(path), _rel(path, clone)), out)
            for stat, s in card["stats"].items():
                edge = s["edge"]
                print(card["stem"], stat, s["value"], edge["kind"],
                      edge.get("x") or {n: o for n, o in edge["offsets"].items()})
        return 0
    only = {arg for arg in argv[2:] if not arg.startswith("--")}
    cards = [card for card in _cards(out) if not only or card["stem"] in only]
    if command in ("build", "finish"):
        print(write_jsx(out, command, cards))
        return 0
    if command == "tag":
        for card in cards:
            print(card["stem"], "tagged", tag_card(card))
        return 0
    if command == "strip":                      # the repo's PSDs carry no merged image (owner 2026-11-07)
        for card in cards:
            psd, old = PSDImage.open(join(card["folder"], "final.psd")), PSDImage.open(join(card["folder"], "T.psd"))
            kept, fills = restore_links(psd, old), restore_fills(psd, old)
            psd_edit.save_psd(psd, join(card["folder"], "card.psd"))
            print(card["stem"], "card.psd", f"{kept} link paths as they were", f"fills as they were: {fills}")
        return 0
    if command == "verify":
        parser = parser_for(clone)
        for card in cards:
            print(json.dumps(verify_card(parser, card, clone, flags.get("file", "card.psd"),
                                         tuple(int(v) for v in flags.get("bakes", "0,3,10").split(",") if v))))
        return 0
    raise SystemExit(f"unknown command {command}")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
