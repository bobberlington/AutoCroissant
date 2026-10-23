"""
png_export (guard 1 of the push guards): the PNG the bake would make (the fork's render of the changed layers in the
card's ICC export mode, merged into the old PNG), gated; Photoshop's stored composite only as a cross-check that can
refuse. Fixture PSDs (a pixel layer, made here) cover the gates; real commits from the card clone's object store (read
with git cat-file, never written) pin the fork-render replay of docs/Campaigns/PUSH_GUARDS.md 2.2 (2026-10-22, pin
cards-20261022b): Dirty Bed and Final Dark Mind Eye within the bar of the author's export, God Tamer's push moved no
pixel, a bot bake drawn, and the refusals: a script save under a stale composite, Restrained's old PNG already stale where
the change is, Defcon Panda's drop shadow the fork draws differently, 20 Creature Types' image changed where no layer did.
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
from conftest import at_rev, fork_reason, text_engine

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
RESTRAINED = ("68ff57d57b6238f8075e790aa4cc1a9fc6800a7d", "bab7f20bb143c50e13786ec212edd25ee35a0a96",
              "709c9739ad353b96d5a05dd6906b8637fd65b20a")                                                # 5c8cb87b
DEFCON_PANDA = ("9ea7b16710613bd532ceb56e09238bf99a86d5e6", "9e8da0ed097e6dd21cc1d061e38de6bdf739672f",
                "a558861e09df8606038cf9b3218694a12076b8b8")                                              # 1b4f36a0
CREATURE_TYPES = ("c0f6d11f570e5cf9fe3adc01e984a5723d10bde0", "cf4f4394cb06986c3ba4150ffbe9cca3a5f2ced0",
                  "e35deab6a8bdc6d760e73b7363998f6d20c17111")                                            # 100511d3
MINI_DOOMER = "Auxiliary/Minions/Mini_Doomer"
# The real cards are drawn as the bake draws them: upstream psd-tools renders type layers and effects otherwise. Measured
# at the Action's pin cards-20261022b (90508839).
fork = pytest.mark.skipif(not text_engine(), reason="guard 1 renders with the fork, as the bake does: " + fork_reason())
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


def test_g2_names_a_16_bit_or_cmyk_psd():
    from types import SimpleNamespace
    from psd_tools.constants import ColorMode
    assert pe._no_composite(SimpleNamespace(color_mode=ColorMode.RGB, depth=16)).why == pe.G2_MODE
    assert pe._no_composite(SimpleNamespace(color_mode=ColorMode.CMYK, depth=8)).why == pe.G2_MODE
    assert pe._no_composite(SimpleNamespace(color_mode=ColorMode.RGB, depth=8)).why == pe.G2


def test_g7_size_changed(tmp_path):
    new = fixture_psd(tmp_path, card(size=(60, 100)), "new")
    found = pe.export(fixture_psd(tmp_path, card(), "old"), new, card())
    assert (found.outcome, found.gate) == ("refused", "G7")


def test_the_png_is_the_forks_render_merged_into_the_old_png(tmp_path, monkeypatch):
    """The changed layer's box is rendered for both PSDs (psd_edit.render_region, the bake's) and the old PNG takes the
    new render where it moved; the stored composite is never the source: a composite that disagrees by a few levels
    leaves the PNG the render's, and the cross-check passes within the bar."""
    old = fixture_psd(tmp_path, card(), "old")
    new = fixture_psd(tmp_path, card(colour=(0, 255, 0)), "new")
    drawn = []
    draw = psd_edit.render_region
    monkeypatch.setattr(psd_edit, "render_region", lambda psd, box, icc=True: drawn.append(box) or draw(psd, box, icc))
    found = pe.export(old, new, card())
    assert found.outcome == "re-exported" and found.writer == "tool" and found.gate == ""
    assert drawn == [(10, 10, 31, 31)] * 2                              # the Square's record box, old then new
    assert found.changed_px == 21 * 21 and found.box == (10, 10, 31, 31) and found.numbers["render_s"] >= 0
    assert found.numbers["baseline"]["bad_pct"] == 0 and found.numbers["cross"]["bad_pct"] == 0
    assert np.asarray(pe.open_png(found.data)).tolist() == np.asarray(card(colour=(0, 255, 0))).tolist()
    real, off = pe.composite, card(colour=(0, 253, 0))                       # Photoshop's image 2 levels off the render
    monkeypatch.setattr(pe, "composite", lambda psd: off if psd is new else real(psd))
    shifted = pe.export(old, new, card())
    assert shifted.outcome == "re-exported" and shifted.data == found.data        # the render, not the composite
    assert shifted.numbers["cross"]["mean"] == 2


@pytest.mark.parametrize("photoshop", [True, False])
def test_a_png_pushed_before_its_psd_is_current(tmp_path, monkeypatch, photoshop):
    """The author pushed the PNG first and the PSD minutes later in its own push: the PNG in the tree already shows the
    pushed PSD, so nothing is made (a Photoshop save was re-exported over the author's export, a tool save refused at the
    referee's baseline). A PNG still showing the old PSD is re-exported, a change under the parity threshold included
    (24 levels: here 15)."""
    if photoshop:
        monkeypatch.setattr(pe, "photoshop_saved", lambda old, new: True)
    old = fixture_psd(tmp_path, card(), "old")
    for colour in ((0, 255, 0), (0, 0, 240)):
        new = fixture_psd(tmp_path, card(colour=colour), f"new{colour[1]}")
        ahead = pe.export(old, new, card(colour=colour))
        assert (ahead.outcome, ahead.changed_px, ahead.data) == ("current", 21 * 21, b""), colour
        assert ahead.numbers["png_ahead"]["new"]["max"] == 0 and ahead.numbers["png_ahead"]["old"]["mean"] > 0
        stale = pe.export(old, new, card())
        assert stale.outcome == "re-exported" and "png_ahead" not in stale.numbers, colour
    half = card(colour=(0, 255, 0))                                       # the change, but stale elsewhere: not "ahead"
    ImageDraw.Draw(half).rectangle((0, 40, 59, 89), fill=(255, 255, 0))
    assert pe.export(old, fixture_psd(tmp_path, card(colour=(0, 255, 0)), "green"), half).gate == "G5"


def test_a_new_psd_beside_its_own_png_is_current(tmp_path, monkeypatch):
    """A new PSD beside the PNG the author pushed first (or a PNG-only card's PSD): the PNG already shows Photoshop's
    image, export noise at most (no pixel past BESIDE_CUT), so nothing is rendered, made or told."""
    monkeypatch.setattr(psd_edit, "render_region", lambda *a, **k: pytest.fail("rendered a PNG that is current"))
    new = fixture_psd(tmp_path, card(), "new")
    found = pe.export(None, new, card())
    assert (found.outcome, found.changed_px, found.data, found.writer) == ("current", 0, b"", "new")
    assert found.numbers["composite_px"] == 0 and "png_ahead" in found.numbers
    noisy = card()
    ImageDraw.Draw(noisy).rectangle((40, 60, 44, 64), fill=(200, 30, 30 + pe.BESIDE_CUT))
    found = pe.export(None, new, noisy)
    assert (found.outcome, found.numbers["composite_px"], found.data) == ("current", 25, b"")


def test_a_small_recolour_beside_its_png_is_rendered(tmp_path):
    """Owner (2026-10-23): past BESIDE_CUT (8 levels) a new PSD beside its PNG is no longer 'current': a 20-level
    recolour gets the fork's render (or a refusal), never a silent pass."""
    new = fixture_psd(tmp_path, card(), "new")
    recoloured = card()
    ImageDraw.Draw(recoloured).rectangle((40, 60, 44, 64), fill=(200, 30, 50))
    found = pe.export(None, new, recoloured)
    assert found.outcome != "current" and "png_ahead" not in found.numbers


def test_a_new_psd_beside_an_older_png_takes_the_render_where_photoshops_image_moved(tmp_path, monkeypatch):
    """The PNG shows an older card: it takes the fork's render only where Photoshop's image differs from it (D); the
    render's own errors elsewhere never reach it. Errors inside D past the bar refuse."""
    new = fixture_psd(tmp_path, card(colour=(0, 255, 0)), "new")
    draw = psd_edit.render_region

    def off(psd, box, icc=True):                     # the fork draws the base 9 levels off, the square right
        image = np.array(draw(psd, box, icc).convert("RGB"))
        image[60:, :] = (209, 39, 39)
        return Image.fromarray(image)

    monkeypatch.setattr(psd_edit, "render_region", off)
    found = pe.export(None, new, card())
    assert found.outcome == "re-exported" and found.changed_px == 21 * 21 and found.box == (10, 10, 31, 31)
    assert found.numbers["cross"]["max"] == 0
    assert np.asarray(pe.open_png(found.data)).tolist() == np.asarray(card(colour=(0, 255, 0))).tolist()
    monkeypatch.setattr(psd_edit, "render_region", lambda psd, box, icc=True: card(colour=(0, 200, 0)))
    refused = pe.export(None, new, card())
    assert (refused.outcome, refused.gate, refused.data) == ("refused", "G6", b"")
    assert refused.why.startswith("PNG not made: the bot's drawing of your change")


def test_g5_the_old_png_was_stale_elsewhere(tmp_path):
    """The old PNG disagrees with the old PSD outside the change: no PNG, the author exports it."""
    old = fixture_psd(tmp_path, card(), "old")
    new = fixture_psd(tmp_path, card(colour=(0, 255, 0)), "new")
    stale = card()
    ImageDraw.Draw(stale).rectangle((0, 40, 59, 89), fill=(255, 255, 0))      # half the card
    found = pe.export(old, new, stale)
    assert (found.outcome, found.gate) == ("refused", "G5") and "export it in Photoshop" in found.why


def test_g6_a_composite_that_lies(tmp_path):
    """A tool wrote a composite the layers don't show: the render moved nothing where Photoshop's image did, refused."""
    old = fixture_psd(tmp_path, card(), "old")
    new = PSDImage.new("RGB", (60, 90))
    new.append(PixelLayer.frompil(card(), new, "Art"))          # the layers still show the blue square
    path = tmp_path / "liar.psd"
    if "preview" not in signature(PSDImage.save).parameters:
        pytest.skip("this psd-tools has no save(preview=)")
    new.save(str(path), preview=card(colour=(0, 255, 0)))
    found = pe.export(old, PSDImage.open(str(path)), card())
    assert (found.outcome, found.gate, found.why) == ("refused", "G6", pe.G6_UNSEEN)


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


def test_a_new_card_is_rendered_and_cross_checked(tmp_path, monkeypatch):
    new = fixture_psd(tmp_path, card(), "new")
    found = pe.export_new(new)
    assert found.outcome == "re-exported" and found.writer == "new" and pe.open_png(found.data).mode == "RGB"
    assert found.numbers["cross"]["bad_pct"] == 0
    monkeypatch.setattr(pe, "composite", lambda psd: card(colour=(255, 255, 0), box=(0, 0, 59, 60)))
    refused = pe.export_new(new)                         # the render disagrees with the PSD's own image: no PNG
    assert refused.gate == "G6" and refused.why.startswith("PNG not made: the bot's drawing of the new PSD")


def test_every_refusal_says_export_it_in_photoshop():
    for line in (pe.G2, pe.G2_MODE, pe.G5, pe.G5_IN, pe.G6, pe.G6_BASE, pe.G6_NEW, pe.G6_STALE, pe.G6_UNSEEN, pe.G7):
        assert line.startswith("PNG not made: ") and line.endswith("export it in Photoshop"), line


def test_png_check(tmp_path):
    psd = fixture_psd(tmp_path, card(), "a")
    assert pe.png_check(psd, card())["ok"] is True
    assert pe.png_check(psd, card(colour=(255, 255, 255), box=(0, 0, 59, 60)))["ok"] is False
    assert pe.png_check(psd, card(size=(10, 10)))["ok"] is None


# ========================
# Real commits (the clone's object store, read-only)
# ========================
def _near_the_authors_export(found, truth) -> np.ndarray:
    diff = pe._diff(found.image, truth)
    assert psd_edit._within_bar(psd_edit._measure(found.image, truth)) and (diff <= 2).mean() > 0.9995
    return diff


@fork
def test_dirty_bed_is_drawn_within_the_bar_of_the_authors_export(clone):
    old, new, old_png, truth = pair(clone, DIRTY_BED)
    assert pe.photoshop_saved(old, new) and not pe.photoshop_saved(new, new)
    found = pe.export(old, new, old_png)
    assert found.outcome == "re-exported" and found.writer == "photoshop" and found.icc == "as stored"
    assert found.numbers["composite_px"] == 8961 and abs(found.changed_px - 8961) < 50 and found.numbers["where"] == 0
    assert found.numbers["outside"]["bad_pct"] == 0 and psd_edit._within_bar(found.numbers["baseline"])
    diff = _near_the_authors_export(found, truth)
    x0, y0, x1, y1 = found.numbers["region"]
    outside = np.ones(diff.shape, bool)
    outside[y0:y1, x0:x1] = False
    assert int(diff[outside].max()) == 0                 # outside the changed layers the PNG is the old one, untouched


@fork
def test_final_dark_mind_eye_within_the_bar(clone):
    old, new, old_png, truth = pair(clone, FINAL_EYE)
    found = pe.export(old, new, old_png)
    assert found.outcome == "re-exported" and found.icc == "converted"
    assert int(_near_the_authors_export(found, truth).max()) <= psd_edit.PARITY_THRESH


@gradients
def test_god_tamer_moved_no_pixel(clone):
    old, new, old_png, _ = pair(clone, GOD_TAMER)
    assert pe.changed_layers(old, new)[1] > 0            # the records changed (stars grouped)
    found = pe.export(old, new, old_png)
    assert found.outcome == "current" and found.changed_px == 0 and found.numbers["composite_px"] == 0 and not found.data
    assert psd_edit._within_bar(found.numbers["png_vs_psd"])


@fork
def test_a_bot_bake_is_drawn(clone):
    old, new, old_png, truth = pair(clone, SPRINGTRAP)
    assert not pe.photoshop_saved(old, new)              # the fork keeps the XMP
    found = pe.export(old, new, old_png)
    assert found.outcome == "re-exported" and found.writer == "tool"
    assert psd_edit._within_bar(found.numbers["baseline"]) and psd_edit._within_bar(found.numbers["cross"])
    _near_the_authors_export(found, truth)


@fork
def test_a_script_save_under_a_stale_composite_is_refused(clone, tmp_path):
    """Mini Doomer's title hidden by the fork, saved without a new preview: the render drops the title, the composite
    still shows it, so the cross-check refuses."""
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


@fork
def test_the_refusals_of_the_replay(clone):
    """Three of the 9 refusals of the 34 replayed commits (PUSH_GUARDS.md 16, rulings): Restrained's old PNG did not
    show its old PSD where the author changed it ("(BOT) -1 Action" in the PSD, "-1 Action" in the PNG); Defcon Panda's
    Ability drop shadow, which the fork draws with another edge (1.4% bad); 20 Creature Types' header, which Photoshop
    redrew from a type layer whose stored raster did not change (the composite cross-check refuses what the render
    cannot see)."""
    for shas, gate, why in ((RESTRAINED, "G5", "PNG not made: the PSD and its old PNG already disagreed where you changed it"),
                            (DEFCON_PANDA, "G6", "PNG not made: the bot cannot draw this part of the card as its old PNG"),
                            (CREATURE_TYPES, "G6", pe.G6_UNSEEN)):
        old, new, old_png, _ = pair(clone, shas)
        found = pe.export(old, new, old_png)
        assert (found.outcome, found.gate) == ("refused", gate) and found.why.startswith(why), (shas, found.why)
        assert not found.data
    # the author's PNG pushed first: Photoshop's image vouches for the change the fork draws otherwise, nothing to tell
    old, new, _, _ = pair(clone, DEFCON_PANDA)
    first = pe.export(old, new, pe.open_png(blob(clone, "60bc9b5a4e2b271dc1c74e8eaad3ce6ef626c482")))
    assert first.outcome == "current" and "png_ahead" in first.numbers


def test_a_new_psd_beside_the_authors_png_is_current(clone, monkeypatch):
    """Review 7: beside the author's own export (pushed first), Dirty Bed and Defcon Panda were refused (the fork's
    drawing errors over the card made D). The PNG shows Photoshop's image: current, nothing rendered."""
    monkeypatch.setattr(psd_edit, "render_region", lambda *a, **k: pytest.fail("rendered a PNG that is current"))
    for psd, png in ((DIRTY_BED[1], DIRTY_BED[3]), (DEFCON_PANDA[1], "60bc9b5a4e2b271dc1c74e8eaad3ce6ef626c482")):
        found = pe.export(None, pe.open_psd(blob(clone, psd)), pe.open_png(blob(clone, png)))
        assert (found.outcome, found.data, found.numbers["composite_px"]) == ("current", b"", 0), psd
        assert "png_ahead" in found.numbers


def test_r10_flags_the_lost_gradient_maps(clone):
    """At 6e50c3ee (2.1), before the gradient maps were restored on the branch."""
    for rel, ok in (("Creatures/Kirby/5 Stars/Dark_Mind_Eye", False), ("Items/Active/2 Stars/Dirty_Bed", True)):
        psd, png = at_rev(clone, "6e50c3ee", rel + ".psd"), at_rev(clone, "6e50c3ee", rel + ".png")
        found = pe.png_check(pe.open_psd(psd), pe.open_png(png))
        assert found["ok"] is ok, (rel, found)
