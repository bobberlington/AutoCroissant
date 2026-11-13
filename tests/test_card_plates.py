"""
The mock-up's plates (commands/card_plates.py): a real card drawn without its text layers, each crop the PNG's backdrop
off the layer's ink (an FX layer's word in a second crop) with the layer's own lines and letters, and the plates branch:
added, kept, dropped with the PSDs.
"""
import base64
import io
import json
import subprocess

import numpy as np
from PIL import Image
from psd_tools import PSDImage

import commands.card_plates as cp
import commands.psd_edit as psd_edit
from commands.psd_edit import card_parts, _bare_parser

DEDEDE = "Creatures/Kirby/5 Stars/Masked_Dedede.psd"   # Name, Ability and "Ability FX: Blazing"
PUZZLED = "Creatures/Other/3 Stars/Puzzled_Cat.psd"    # her own pictures under the old template's digit rows
PIX = "Creatures/Kirby/3 Stars/Pix.psd"                 # three creatures: their own DEF rows, one HP bar for all


def test_plates_of_a_card(clone):
    psd, png = PSDImage.open(clone / DEDEDE), Image.open(clone / DEDEDE.replace(".psd", ".png"))
    parts = card_parts(psd, DEDEDE, "Masked Dedede", _bare_parser())
    found = cp.plates(psd, png, parts, str(clone / "MDW" / "Fonts"))
    ids = [p["id"] for k in ("title", "ability") for p in parts[k]]
    assert [c["id"] for c in found["crops"] if "fx" not in c] == ids
    assert [c["fx"] for c in found["crops"] if "fx" in c] == [["Blazing"]]
    sheet = np.asarray(Image.open(io.BytesIO(base64.b64decode(found["img"].split(",", 1)[1]))).convert("RGB"), np.int16)
    small = np.asarray(png.convert("RGB").resize((cp.VIEW_W, round(png.height * cp.VIEW_W / png.width)), Image.LANCZOS), np.int16)
    for c in found["crops"]:
        x0, y0, x1, y1 = (round(v * cp.VIEW_W / psd.width) for v in c["box"])
        assert c["at"][2:] == [x1 - x0, y1 - y0] and c["box"][0] <= c["ink"][0] < c["ink"][2] <= c["box"][2]
        crop = sheet[c["at"][1]:c["at"][1] + c["at"][3], :c["at"][2]]
        off = np.ones(crop.shape[:2], bool)
        ix0, iy0, ix1, iy1 = (round(v * cp.VIEW_W / psd.width) for v in c["ink"])
        off[max(0, iy0 - y0 - 1):iy1 - y0 + 1, max(0, ix0 - x0 - 1):ix1 - x0 + 1] = False
        assert np.abs(crop - small[y0:y1, x0:x1]).max(-1)[off].mean() < 8   # the card's own backdrop
        rim = np.zeros(off.shape, bool)
        rim[[0, -1], :] = rim[:, [0, -1]] = True
        assert np.abs(crop - small[y0:y1, x0:x1]).max(-1)[rim & off].mean() < 4   # no seam from the sheet's gaps
        assert np.abs(crop - small[y0:y1, x0:x1]).max(-1)[~off].mean() > 20  # no letters
    for c in (c for c in found["crops"] if "fx" not in c):     # the layer's layout: its own text, line by line
        layer = next(x for x in psd.descendants() if x.layer_id == c["id"])
        assert c["text"].replace("\x03", "\r").strip() == layer.text.replace("\x03", "\r").strip()
        for start, end, base, left, right, top, bottom, _ in c["lines"]:
            xs = [x for i, x, *_ in c["glyphs"] if start <= i < end]
            assert c["ink"][1] - 2 <= top < base <= bottom + 2 <= c["ink"][3] + 4 and xs == sorted(xs) and left - 1 <= xs[0] < right
    assert all("lines" in c for c in found["crops"] if "fx" not in c) and not any("lines" in c for c in found["crops"] if "fx" in c)


def test_bar_pictures_are_the_bakes(clone, tmp_path, monkeypatch):
    """A stat's picture is the bake's PNG of that edit (apply_jobs); the sheet keeps per value the view pixels moving past
    BAR_MOVED: pasted on the card's view, the bake's view there and within BAR_MOVED of it elsewhere; none for 11. A lossy
    sheet has the same cuts and alpha. A stat past 10 is pictured (the cells draw 0-10)."""
    raw = (clone / PUZZLED).read_bytes()
    psd, png = PSDImage.open(io.BytesIO(raw)), Image.open(clone / PUZZLED.replace(".psd", ".png"))
    png.load()
    parser = _bare_parser()
    parsed = psd_edit.parse_card(parser, str(clone / PUZZLED), PUZZLED, psd)
    psd_edit.use_record(psd, PUZZLED)
    icc = psd_edit.export_icc(psd, png)
    ours = cp.bar_value(psd, raw, png, parsed, parser, PUZZLED, "def", 9, icc)
    baked = psd_edit.apply_jobs(str(clone / PUZZLED), str(clone / PUZZLED.replace(".psd", ".png")), PUZZLED, {"def": 9},
                                parsed, parser, str(tmp_path), str(clone / "MDW" / "Fonts"), psd=psd)
    assert np.array_equal(np.asarray(ours), np.asarray(Image.open(baked.png_path)))
    assert cp.bar_value(psd, raw, png, parsed, parser, PUZZLED, "def", 11, icc) is None
    monkeypatch.setattr(cp, "bar_value", lambda *args: ours if args[6:8] == ("def", 9) else None)   # one value baked
    changes = cp.bar_changes(psd, raw, png, parsed, parser, PUZZLED)
    entry = _moved(png, ours)
    assert list(changes) == ["def"] and list(changes["def"]) == [9] and changes["def"][9][:2] == entry[:2]
    assert all(np.array_equal(a, b) for a, b in zip(changes["def"][9][2:], entry[2:]))
    sheet = cp.bar_sheet(changes, parsed["card"])
    assert cp.with_bars(None, psd, raw, png, parsed, parser, PUZZLED, "custom")["bars"] == sheet
    x, y, w, h, sx, sy = sheet["cuts"][sheet["bars"]["def"]["to"]["9"]]
    cut = np.asarray(Image.open(io.BytesIO(base64.b64decode(sheet["img"].split(",", 1)[1]))).convert("RGBA"))[sy:sy + h, sx:sx + w]
    shown, want = np.asarray(cp.view(png).convert("RGBA")).copy(), np.asarray(cp.view(ours).convert("RGBA"))
    held = cut[..., 3] >= 128
    shown[y:y + h, x:x + w][held] = cut[held]
    assert sheet["bars"]["def"]["at"] == 4 and held.sum() > 1000
    assert np.array_equal(shown[y:y + h, x:x + w][held], want[y:y + h, x:x + w][held])
    assert np.abs(shown.astype(int) - want).max() <= cp.BAR_MOVED
    assert cp.pictured(psd, png, {"atk": 12}, "new") and not cp.pictured(psd, png, {}, "custom")
    tall = np.random.default_rng(1).integers(3, 256, (2, 300, 40, 4), np.uint8)      # past `side`: two columns
    two = cp.bar_sheet({"def": {v: (0, 0, tall[v], np.zeros_like(tall[v])) for v in (0, 1)}}, {"def": 4}, side=400)
    img = np.asarray(Image.open(io.BytesIO(base64.b64decode(two["img"].split(",", 1)[1]))).convert("RGBA"))
    assert img.shape == (300, 80, 4) and [c[4:] for c in two["cuts"]] == [[0, 0], [40, 0]]
    assert all(np.array_equal(img[:, 40 * v:40 * v + 40, :3], tall[v][..., :3]) for v in (0, 1))
    monkeypatch.setattr(cp, "BAR_BYTES", 0)
    lossy = cp.bar_sheet(changes, parsed["card"])
    alpha = np.asarray(Image.open(io.BytesIO(base64.b64decode(lossy["img"].split(",", 1)[1]))).convert("RGBA"))[sy:sy + h, sx:sx + w, 3]
    assert lossy["img"].startswith("data:image/avif") and lossy["cuts"] == sheet["cuts"] and np.array_equal(alpha >= 128, held)


def test_star_pictures_are_the_bakes(clone, tmp_path):
    """A stars count's picture is the bake's PNG of that edit (apply_jobs into its "N Stars" folder, as card_bake's
    plan_move); with_stars pictures every count to the top but the card's, at the card's."""
    raw = (clone / PUZZLED).read_bytes()
    psd, png = PSDImage.open(io.BytesIO(raw)), Image.open(clone / PUZZLED.replace(".psd", ".png"))
    png.load()
    parser = _bare_parser()
    parsed = psd_edit.parse_card(parser, str(clone / PUZZLED), PUZZLED, psd)
    psd_edit.use_record(psd, PUZZLED)
    ours = cp.star_value(psd, raw, png, parsed, parser, PUZZLED, 4, psd_edit.export_icc(psd, png))
    baked = psd_edit.apply_jobs(str(clone / PUZZLED), str(clone / PUZZLED.replace(".psd", ".png")),
                                "Creatures/Other/4 Stars/Puzzled_Cat.psd", {"stars": 4}, parsed, parser, str(tmp_path),
                                str(clone / "MDW" / "Fonts"), psd=psd)
    assert np.array_equal(np.asarray(ours), np.asarray(Image.open(baked.png_path)))
    stars = cp.with_stars(None, psd, raw, png, parsed, parser, PUZZLED, 5)["icons"]["bars"]["stars"]
    assert stars["at"] == 3 and sorted(stars["to"]) == ["1", "2", "4", "5"]


def test_creature_pictures_are_the_bakes(clone, tmp_path):
    """Creature 2's own stat picture is the bake's PNG of that `creatures` edit, keyed "<stat>@2" at its value; none
    past what its row prints, nor on a bar creature 1 shares (creature 1's picture)."""
    raw = (clone / PIX).read_bytes()
    psd, png = PSDImage.open(io.BytesIO(raw)), Image.open(clone / PIX.replace(".psd", ".png"))
    png.load()
    parser = _bare_parser()
    parsed = psd_edit.parse_card(parser, str(clone / PIX), PIX, psd)
    psd_edit.use_record(psd, PIX)
    icc = psd_edit.export_icc(psd, png)
    wanted = [dict(one) for one in parsed["card"]["creatures"]]
    wanted[1]["def"] = 9
    job = psd_edit.classify(parsed, {"creatures": wanted}, psd, parser).jobs["creatures"]
    ours = cp.bar_value(psd, raw, png, parsed, parser, PIX, "def", 9, icc, creature=2)
    baked = psd_edit.apply_jobs(str(clone / PIX), str(clone / PIX.replace(".psd", ".png")), PIX, {"creatures": job},
                                parsed, parser, str(tmp_path), str(clone / "MDW" / "Fonts"), psd=psd)
    assert np.array_equal(np.asarray(ours), np.asarray(Image.open(baked.png_path)))
    assert cp.bar_value(psd, raw, png, parsed, parser, PIX, "def", 11, icc, creature=2) is None
    assert cp.bar_value(psd, raw, png, parsed, parser, PIX, "hp", 5, icc, creature=2) is None
    at = {**parsed["card"], **cp._creature_stats(parsed["card"])}
    assert cp.bar_sheet({"def@2": {9: _moved(png, ours)}}, at)["bars"]["def@2"]["at"] == 8 and at["def@3"] == 2


def test_a_comp_bar_at_zero_is_pictured(monkeypatch):
    """A layer-comp stat at 0 is pictured: its lit look is under the darks (the 4 own-look rows, 11-12), the cells could
    only borrow another row's."""
    psd = type("P", (), {"size": (10, 20)})()
    monkeypatch.setattr(cp, "variants_of", lambda psd: [None])
    monkeypatch.setattr(cp, "comp_bars", lambda psd: {"hp": [], "def": [("bar", 0, (0, 0, 1, 1))], "atk": [], "spd": []})
    monkeypatch.setattr(cp, "_own", lambda layer: False)
    monkeypatch.setattr(cp, "mock_misses", lambda *args: [])
    monkeypatch.setattr(cp, "view", lambda png: png)
    monkeypatch.setattr(cp, "_stats", lambda psd: {})
    png = type("I", (), {"size": (10, 20)})()
    assert cp.pictured(psd, png, {"hp": 0, "def": 0}, "new") and not cp.pictured(psd, png, {"hp": 0, "def": 2}, "new")


def test_bar_changes_read_caps_once(clone, monkeypatch):
    """bar_changes reads stat_caps once for all its values; a value's picture is bar_value's own (classify reading them)."""
    raw = (clone / PUZZLED).read_bytes()
    psd, png = PSDImage.open(io.BytesIO(raw)), Image.open(clone / PUZZLED.replace(".psd", ".png"))
    png.load()
    parser = _bare_parser()
    parsed = psd_edit.parse_card(parser, str(clone / PUZZLED), PUZZLED, psd)
    calls, read = [], psd_edit.stat_caps
    monkeypatch.setattr(psd_edit, "stat_caps", lambda p: calls.append(p) or read(p))
    monkeypatch.setattr(cp, "BAR_TOP", 1)
    changes = cp.bar_changes(psd, raw, png, parsed, parser, PUZZLED)
    assert len(calls) == 1 and calls[0] is psd and sorted(changes) == sorted(psd_edit.STAT_FIELDS)
    alone = _moved(png, cp.bar_value(psd, raw, png, parsed, parser, PUZZLED, "def", 0, psd_edit.export_icc(psd, png)))
    assert len(calls) == 2 and all(np.array_equal(a, b) for a, b in zip(changes["def"][0], alone))


def _moved(png, merged):
    """bar_changes' entry of one value."""
    was, now = (np.asarray(cp.view(image).convert("RGBA")) for image in (png, merged))
    ys, xs = np.nonzero((now != was).any(-1))
    x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
    return int(x0), int(y0), now[y0:y1, x0:x1], was[y0:y1, x0:x1]


def test_publish_adds_keeps_and_drops(tmp_path, monkeypatch):
    env = {"GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": "/dev/null", "HOME": str(tmp_path)}
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    git = lambda *args: subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()
    origin, work = tmp_path / "origin.git", tmp_path / "work"
    git("init", "-q", "--bare", "-b", "main", str(origin))
    git("-C", str(origin), "config", "uploadpack.allowFilter", "true")
    seed = tmp_path / "seed"
    git("init", "-q", "-b", "main", str(seed))
    for name in ("A.psd", "B.psd", "C.png"):
        (seed / name).write_bytes(name.encode())
    git("-C", str(seed), "add", "-A")
    git("-C", str(seed), "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "x")
    git("-C", str(seed), "push", "-q", str(origin), "main")
    git("clone", "-q", "--filter=blob:none", f"file://{origin}", str(work))     # blobless, as the job's checkout
    monkeypatch.setenv("GIT_NO_LAZY_FETCH", "1")
    a, b, c = (git("-C", str(work), "rev-parse", f"HEAD:{n}") for n in ("A.psd", "B.psd", "C.png"))
    files = lambda: sorted(git("-C", str(origin), "ls-tree", "--name-only", cp.BRANCH).split())
    plate = {"v": 1, "crops": []}
    assert cp.publish({}, str(work))["commit"] == "" and not git("-C", str(origin), "branch", "--list", cp.BRANCH)
    done = cp.publish({a: plate, b: plate, c: plate, "f" * 40: plate}, str(work))
    assert (done["added"], done["dropped"]) == (2, 0) and files() == sorted([f"{a}.json", f"{b}.json"])
    git("-C", str(seed), "rm", "-q", "B.psd")
    git("-C", str(seed), "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "y")
    git("-C", str(seed), "push", "-q", str(origin), "main")
    later = tmp_path / "later"                                                  # the next run: the branch's blobs not here
    monkeypatch.delenv("GIT_NO_LAZY_FETCH")
    git("clone", "-q", "--filter=blob:none", f"file://{origin}", str(later))
    monkeypatch.setenv("GIT_NO_LAZY_FETCH", "1")
    done = cp.publish({}, str(later))
    assert (done["added"], done["dropped"], done["kept"]) == (0, 1, 1) and files() == [f"{a}.json"]
    assert json.loads(git("-C", str(origin), "show", f"{cp.BRANCH}:{a}.json")) == plate
    assert cp.publish({a: plate}, str(later))["commit"] == ""                                   # nothing changed
    fresh = tmp_path / "fresh"                                                  # a plate replaced, its old blob not here
    monkeypatch.delenv("GIT_NO_LAZY_FETCH")
    git("clone", "-q", "--filter=blob:none", f"file://{origin}", str(fresh))
    monkeypatch.setenv("GIT_NO_LAZY_FETCH", "1")
    run = cp.run_process                                                        # HTTPS pushes thin whatever --no-thin says
    monkeypatch.setattr(cp, "run_process", lambda cmd, **kw: run(["--thin" if a == "--no-thin" else a for a in cmd], **kw))
    newer = {"v": 1, "crops": [[n, n, 9, 9] for n in range(99)]}
    assert cp.publish({a: newer}, str(fresh))["added"] == 1
    assert json.loads(git("-C", str(origin), "show", f"{cp.BRANCH}:{a}.json")) == newer
