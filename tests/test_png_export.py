"""
png_export (guard 1 of the push guards): the PNG from the PSD's stored composite, converted as the card was exported,
and its gates. Fixture PSDs (a pixel layer, made here) cover the gates; real commits from the card clone's object store
(read with git cat-file, never written) pin the replay of docs/Campaigns/PUSH_GUARDS.md 2.2: Dirty Bed equals the
author's export, Final Dark Mind Eye is within 4 levels, God Tamer's push moved no pixel, a bot bake is refereed by the
fork, and a script save under a stale composite is refused.
"""
import subprocess
from importlib.util import find_spec
from inspect import signature

import numpy as np
import pytest
from PIL import Image, ImageDraw
from psd_tools import PSDImage
from psd_tools.api.layers import PixelLayer

import commands.png_export as pe
import commands.psd_edit as psd_edit
from conftest import fork_reason, text_engine

# ========================
# Configuration
# ========================
DIRTY_BED = ("cbcd441af0856e1c4750337224def2e2f87051c3", "65d95c60cd60e597bfe84d8833e3f71b450c1dc6",
             "5be89c3efdbb5c835f135d0297e5cae6be3d6a3d", "45c2a2f0c419cdd8f5dc90c7d3085d85f15efd77")   # Zomoroko 7cb4c867
FINAL_EYE = ("861260d8ca12ba463df0370f922959801e15c0c7", "18f3f95f31f983b1f628618ab37705889c3ad2e3",
             "fef4d4e419d60f02632d5840717f60dfe09fa5b2", "189f13a28ad513460de7a62f047b71efa11e5dce")   # 38c55bd3
SPRINGTRAP = ("532cbe005c7a93c98fdc0fd97f24325d83947f92", "d62bea1c719842de02d0d3e5012edd1866e4d66a",
              "65324a959f373b14892e54b6a86028176630bee7", "b224575a3d0a97a3dba11a0be778ff18fb5fddc1")  # the bake c6dc6d08
GOD_TAMER = ("8ca3d785c7c2f28d7f1a7861579b5add3a80a67c", "6b99873e4e0293365181e534aa974f1a2a815bb2",
             "f72ba14a25fd31614c8e20e632d61cd0c612eb78")                                                # dc37f4ad: PNG not pushed
MINI_DOOMER = "Auxiliary/Minions/Mini_Doomer"
# The referee renders the changed layers as the fork does: upstream psd-tools passes the bake's stale-looking pixels as
# "current" and misses Mini Doomer's stale composite. Measured at the Action's pin cards-20261022 and at cards e0ea34f6.
fork = pytest.mark.skipif(not text_engine(), reason="the G6 referee needs the fork's renderer: " + fork_reason())
gradients = pytest.mark.skipif(find_spec("scipy") is None, reason="gradient fills need psd-tools[composite] (scipy), as the Action installs")


# ========================
# Helpers
# ========================
def blob(clone, sha: str) -> bytes:
    done = subprocess.run(["git", "-C", str(clone), "cat-file", "blob", sha], capture_output=True,
                          env={"GIT_NO_LAZY_FETCH": "1", "PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin"})
    if done.returncode:
        pytest.skip(f"the clone lacks blob {sha[:8]}")
    return done.stdout


def pair(clone, shas) -> tuple:
    old, new, old_png = pe.open_psd(blob(clone, shas[0])), pe.open_psd(blob(clone, shas[1])), pe.open_png(blob(clone, shas[2]))
    truth = pe.open_png(blob(clone, shas[3])) if len(shas) > 3 else None
    return old, new, old_png, truth


def card(size=(60, 90), box=(10, 10, 30, 30), colour=(0, 0, 255), base=(200, 30, 30)) -> Image.Image:
    image = Image.new("RGB", size, base)
    ImageDraw.Draw(image).rectangle(box, fill=colour)
    return image


def fixture_psd(tmp_path, image: Image.Image, name: str, square=(10, 10, 31, 31)) -> PSDImage:
    """A PSD with a Base layer (the image's corner colour) and a Square layer (the image's crop at square), saved by
    psd-tools (its composite is the image) and reopened: a change of the square changes one layer record."""
    psd = PSDImage.new("RGB", image.size)
    psd.append(PixelLayer.frompil(Image.new("RGB", image.size, image.getpixel((0, image.height - 1))), psd, "Base"))
    psd.append(PixelLayer.frompil(image.crop(square), psd, "Square", top=square[1], left=square[0]))
    path = tmp_path / f"{name}.psd"
    psd.save(str(path))
    return PSDImage.open(str(path))


# ========================
# Pixels
# ========================
def test_dilate_grows_a_square():
    mask = np.zeros((20, 20), bool)
    mask[10, 10] = True
    grown = pe.dilate(mask, 3)
    assert grown.sum() == 49 and grown[7, 7] and grown[13, 13] and not grown[6, 10] and not grown[10, 14]
    edge = np.zeros((5, 5), bool)
    edge[0, 0] = True
    assert pe.dilate(edge, 2).sum() == 9                # no wrap-around


def test_like_mode_flattens_over_white():
    rgba = Image.new("RGBA", (2, 1), (10, 20, 30, 0))
    assert np.asarray(pe.like_mode(rgba, "RGB")).tolist() == [[[255, 255, 255], [255, 255, 255]]]
    assert pe.like_mode(Image.new("RGB", (1, 1)), "RGBA").getpixel((0, 0))[3] == 255


# ========================
# Fixture PSDs: the gates
# ========================
def test_no_visible_change_is_current(tmp_path):
    old = fixture_psd(tmp_path, card(), "old")
    new = fixture_psd(tmp_path, card(), "new")
    found = pe.export(old, new, card())
    assert found.outcome == "current" and found.changed_px == 0 and not found.data


def test_g2_without_a_composite(tmp_path):
    new = fixture_psd(tmp_path, card(), "new")
    new.has_preview = lambda: False
    found = pe.export(fixture_psd(tmp_path, card(), "old"), new, card())
    assert (found.outcome, found.gate, found.why) == ("refused", "G2", pe.G2)
    assert pe.export_new(new).gate == "G2"


def test_g7_size_changed(tmp_path):
    new = fixture_psd(tmp_path, card(size=(60, 100)), "new")
    found = pe.export(fixture_psd(tmp_path, card(), "old"), new, card())
    assert (found.outcome, found.gate) == ("refused", "G7")


def test_a_tool_save_is_refereed(tmp_path):
    """No XMP: not Photoshop's. The fork renders the pixel layer as the composite shows it: re-exported."""
    old = fixture_psd(tmp_path, card(), "old")
    new = fixture_psd(tmp_path, card(colour=(0, 255, 0)), "new")
    found = pe.export(old, new, card())
    assert found.outcome == "re-exported" and found.writer == "tool" and found.gate == ""
    assert found.changed_px == 21 * 21 and found.box == (10, 10, 31, 31)
    assert found.numbers["referee"]["bad_pct"] == 0 and found.numbers["baseline"]["bad_pct"] == 0
    assert np.asarray(pe.open_png(found.data)).tolist() == np.asarray(card(colour=(0, 255, 0))).tolist()
    assert pe.export(old, new, card(), run_referee=False).gate == "G6"


def test_g5_the_old_png_was_stale_elsewhere(tmp_path):
    """The old PNG disagrees with the old PSD outside the change: no PNG, the author exports it."""
    old = fixture_psd(tmp_path, card(), "old")
    new = fixture_psd(tmp_path, card(colour=(0, 255, 0)), "new")
    stale = card()
    ImageDraw.Draw(stale).rectangle((0, 40, 59, 89), fill=(255, 255, 0))      # half the card
    found = pe.export(old, new, stale)
    assert (found.outcome, found.gate) == ("refused", "G5") and "export it in Photoshop" in found.why


def test_g6_a_composite_that_lies(tmp_path):
    """A tool wrote a composite the layers don't show (the referee's render disagrees): refused."""
    old = fixture_psd(tmp_path, card(), "old")
    new = PSDImage.new("RGB", (60, 90))
    new.append(PixelLayer.frompil(card(), new, "Art"))          # the layers still show the blue square
    path = tmp_path / "liar.psd"
    if "preview" not in signature(PSDImage.save).parameters:
        pytest.skip("this psd-tools has no save(preview=)")
    new.save(str(path), preview=card(colour=(0, 255, 0)))
    found = pe.export(old, PSDImage.open(str(path)), card())
    assert (found.outcome, found.gate) == ("refused", "G6")


def test_numbered_pngs_take_the_change_when_clear_of_their_titles(tmp_path):
    old = fixture_psd(tmp_path, card(), "old")
    new = fixture_psd(tmp_path, card(colour=(0, 255, 0)), "new")
    clear = card()
    ImageDraw.Draw(clear).rectangle((40, 70, 55, 85), fill=(0, 0, 0))       # its own title, far from the change
    touching = card()
    ImageDraw.Draw(touching).rectangle((25, 25, 40, 40), fill=(0, 0, 0))    # overlaps the changed square
    found = pe.export(old, new, card(), {"X2.png": clear, "X3.png": touching})
    data, why = found.numbered["X2.png"]
    assert why == "" and data
    expected = np.asarray(card(colour=(0, 255, 0))).copy()
    expected[70:86, 40:56] = 0
    assert np.asarray(pe.open_png(data)).tolist() == expected.tolist()
    assert found.numbered["X3.png"] == (b"", "X3.png not updated: re-export it in Photoshop")


def test_a_new_card_is_made_and_refereed(tmp_path):
    new = fixture_psd(tmp_path, card(), "new")
    found = pe.export_new(new)
    assert found.outcome == "re-exported" and found.writer == "new" and pe.open_png(found.data).mode == "RGB"
    assert (pe.export_new(new, run_referee=False).gate, pe.export_new(new, run_referee=False).why) == ("G6", pe.G6_NEW)


def test_png_check(tmp_path):
    psd = fixture_psd(tmp_path, card(), "a")
    assert pe.png_check(psd, card())["ok"] is True
    assert pe.png_check(psd, card(colour=(255, 255, 255), box=(0, 0, 59, 60)))["ok"] is False
    assert pe.png_check(psd, card(size=(10, 10)))["ok"] is None


# ========================
# Real commits (the clone's object store, read-only)
# ========================
def test_dirty_bed_equals_the_authors_export(clone):
    old, new, old_png, truth = pair(clone, DIRTY_BED)
    assert pe.photoshop_saved(old, new) and not pe.photoshop_saved(new, new)
    found = pe.export(old, new, old_png)
    assert found.outcome == "re-exported" and found.writer == "photoshop" and found.icc == "as stored"
    assert found.changed_px == 8961
    assert int(pe._diff(found.image, truth).max()) == 0
    assert found.numbers["outside"]["bad_pct"] == 0


def test_final_dark_mind_eye_within_four_levels(clone):
    old, new, old_png, truth = pair(clone, FINAL_EYE)
    found = pe.export(old, new, old_png)
    assert found.outcome == "re-exported" and found.icc == "converted"
    diff = pe._diff(found.image, truth)
    assert int(diff.max()) <= 4 and (diff <= 2).mean() > 0.9999


@gradients
def test_god_tamer_moved_no_pixel(clone):
    old, new, old_png, _ = pair(clone, GOD_TAMER)
    assert pe.changed_layers(old, new)[1] > 0            # the records changed (stars grouped)
    found = pe.export(old, new, old_png)
    assert found.outcome == "current" and found.changed_px == 0 and not found.data
    assert psd_edit._within_bar(found.numbers["png_vs_psd"])


@fork
def test_a_bot_bake_is_refereed(clone):
    old, new, old_png, truth = pair(clone, SPRINGTRAP)
    assert not pe.photoshop_saved(old, new)              # the fork keeps the XMP
    found = pe.export(old, new, old_png)
    assert found.outcome == "re-exported" and found.writer == "tool"
    assert psd_edit._within_bar(found.numbers["referee"]) and psd_edit._within_bar(found.numbers["baseline"])
    assert int(pe._diff(found.image, truth).max()) <= 2


@fork
def test_a_script_save_under_a_stale_composite_is_refused(clone, tmp_path):
    """Mini Doomer's title hidden by the fork, saved without a new preview: the composite still shows it."""
    path = clone / (MINI_DOOMER + ".psd")
    if not path.exists():
        pytest.skip("no Mini Doomer")
    old = PSDImage.open(str(path))
    edited = PSDImage.open(str(path))
    psd_edit.name_layer(edited).visible = False
    out = tmp_path / "Mini_Doomer.psd"
    psd_edit.save_psd(edited, str(out), update_preview=False)
    found = pe.export(old, PSDImage.open(str(out)), pe.open_png((clone / (MINI_DOOMER + ".png")).read_bytes()))
    assert (found.outcome, found.gate, found.why) == ("refused", "G6", pe.G6_STALE)


def test_r10_flags_the_lost_gradient_maps(clone):
    for rel, ok in (("Creatures/Kirby/5 Stars/Dark_Mind_Eye", False), ("Items/Active/2 Stars/Dirty_Bed", True)):
        if not (clone / (rel + ".psd")).exists():
            pytest.skip(f"no {rel}")
        found = pe.png_check(PSDImage.open(str(clone / (rel + ".psd"))), pe.open_png((clone / (rel + ".png")).read_bytes()))
        assert found["ok"] is ok, (rel, found)
