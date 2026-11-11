"""
The mock-up's plates (commands/card_plates.py): a real card drawn without its text layers, each crop the PNG's backdrop
off the layer's ink (an FX layer's word in a second crop), and the plates branch: added, kept, dropped with the PSDs.
"""
import base64
import io
import json
import subprocess

import numpy as np
from PIL import Image
from psd_tools import PSDImage

import commands.card_plates as cp
from commands.psd_edit import card_parts, _bare_parser

DEDEDE = "Creatures/Kirby/5 Stars/Masked_Dedede.psd"   # Name, Ability and "Ability FX: Blazing"


def test_plates_of_a_card(clone):
    psd, png = PSDImage.open(clone / DEDEDE), Image.open(clone / DEDEDE.replace(".psd", ".png"))
    parts = card_parts(psd, DEDEDE, "Masked Dedede", _bare_parser())
    found = cp.plates(psd, png, parts)
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
