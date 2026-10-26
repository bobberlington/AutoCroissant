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
from conftest import fork_reason, text_engine
from helpers import _rows_off, art, card, fixture_psd, photoshop_png

# ========================
# Configuration
# ========================
DIRTY_BED = ("cbcd441af0856e1c4750337224def2e2f87051c3", "65d95c60cd60e597bfe84d8833e3f71b450c1dc6",
             "5be89c3efdbb5c835f135d0297e5cae6be3d6a3d", "45c2a2f0c419cdd8f5dc90c7d3085d85f15efd77")   # Zomoroko 7cb4c867
SPRINGTRAP = ("532cbe005c7a93c98fdc0fd97f24325d83947f92", "d62bea1c719842de02d0d3e5012edd1866e4d66a",
              "65324a959f373b14892e54b6a86028176630bee7", "b224575a3d0a97a3dba11a0be778ff18fb5fddc1")  # the bake c6dc6d08
RESTRAINED = ("68ff57d57b6238f8075e790aa4cc1a9fc6800a7d", "bab7f20bb143c50e13786ec212edd25ee35a0a96",
              "709c9739ad353b96d5a05dd6906b8637fd65b20a")                                                # 5c8cb87b
GREEN_MAGE = ("d17ddacd178eb7353dab252651855da1e2d995f7", "cf9e880710d77ad3b10dfa6690a2a285b5d1b9da",
              "09e2dce6f6ed70454f1e52604015cc41678763ff", "41c2439d09d0eb532995338d2646ed7556b52eac")   # 45cf8746: ATK 6, bars only
SANS = ("97b51c821e8057f1c1a00991172484ee3b30887b", "da0f33d4551435e71a7bd3a64a6f3f5fefa5fa15",
        "6df87ab1668293459015269541c448e4dd3235d9")                                                     # 1b4f36a0: stale push
BLOCK_HOPPER = ("923fdc6d0d874beac87904c5fc68f76737ebaf74", "23f5cd1787d1d8d924738b159acfa523f68e5963")  # 0262b237: raw
CLOT_WHITE = ("09dbfff123941024fe7632dc0109d583f655de24", "497eda88775b84cc2ff53e88699a402f7a71b67c",
              "16c15a759a0a361dcbbe3849218b6f35662703d7", "044b16a8a185ee7c402eb864c33233df21157f41")   # d8163103
ABIGAIL = ("2b68e6a182b67d38e9b7720c035cbdc4f3ab42db", "f7fe63fa4cef78833d18c5e321d40644e7c22a35")    # 'Display' profile
MINI_DOOMER = "Auxiliary/Minions/Mini_Doomer"
SHIELD = "e48fd9f26568f711f80d0688a33e41016d7469a4"     # Items/Equip/3 Stars/Shield_of_the_Guilty.psd (TTSCardMaker 3a268cad)
FREEZER = "8505c4f00a5eda92a48a6966749b214989657caa"    # Field/1 Stars/The_Freezer.psd (91fcda38)
SABRE = "becff9f709b1403a8c5ec2d5aa4b7855ec4c5320"      # Auxiliary/Minions/Sabre.psd (3a268cad): an empty Ability layer
# The real cards are drawn as the bake draws them: upstream psd-tools renders type layers and effects otherwise. Measured
# at the Action's pin cards-20261022b (90508839); misses re-measured at cards-20261023 (ecc4b773).
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


# ========================
# Fixture PSDs: the gates
# ========================
def test_no_visible_change_is_current(tmp_path):
    old = fixture_psd(tmp_path, card(), "old")
    new = fixture_psd(tmp_path, card(), "new")
    found = pe.export(old, new, card())
    assert found.outcome == "current" and found.changed_px == 0 and not found.data


def test_g2_renders_anyway_with_the_bakes_own_gates(tmp_path):
    """Owner, 2026-10-24: a PSD saved without Maximize Compatibility (no stored image) gets its PNG anyway: the fork's
    render merged as always, under the bake's own gates (the baseline, the shape and region), nothing cross-checked
    ("cross" never measured), unchecked so the guards tell it. A push that moved nothing is current; a new card is made
    whole; a new PSD beside its PNG is current within the bar of the render, else redrawn where it differs."""
    old = fixture_psd(tmp_path, card(), "old")
    new = fixture_psd(tmp_path, card(colour=(0, 255, 0)), "new")
    new.has_preview = lambda: False
    found = pe.export(old, new, card())
    assert (found.outcome, found.gate, found.unchecked, found.changed_px) == ("re-exported", "", True, 21 * 21)
    assert found.numbers["baseline"]["bad_pct"] == 0 and "cross" not in found.numbers and found.numbers["reread"] == []
    assert np.asarray(pe.open_png(found.data)).tolist() == np.asarray(card(colour=(0, 255, 0))).tolist()
    same = fixture_psd(tmp_path, card(), "same")
    same.has_preview = lambda: False
    assert pe.export(old, same, card()).outcome == "current"
    made = pe.export_new(new)
    assert (made.outcome, made.unchecked, made.writer) == ("re-exported", True, "new") and "cross" not in made.numbers
    assert pe.export(None, new, card(colour=(0, 255, 0))).outcome == "current"              # beside: the PNG shows it
    beside = pe.export(None, new, card())                                                  # beside an older PNG
    assert (beside.outcome, beside.unchecked, beside.changed_px) == ("re-exported", True, 21 * 21)
    assert not beside.misses and "baseline" not in beside.numbers                          # the change is no drawing miss
    assert beside.numbers["png_vs_render"]["bad_pct"] > 0
    big = fixture_psd(tmp_path, card(colour=(0, 255, 0), box=(0, 0, 59, 60)), "big", square=(0, 0, 60, 61))
    big.has_preview = lambda: False                     # new art over 68% of the card: the author's, never a G6
    large = pe.export(None, big, card())
    assert (large.outcome, large.gate, large.changed_px) == ("re-exported", "", 60 * 61) and not large.misses
    assert psd_edit._past_floor(large.numbers["png_vs_render"]) and pe.miss_notes(large) == []
    assert np.asarray(pe.open_png(beside.data)).tolist() == np.asarray(card(colour=(0, 255, 0))).tolist()
    composed = fixture_psd(tmp_path, card(colour=(0, 255, 0)), "composed")
    assert not pe.export(old, composed, card()).unchecked                                   # with a stored image: checked
    # G8 on this path too: a numbered duplicate takes the change when clear of its own title, else is told
    clear, touching = card(), card()
    ImageDraw.Draw(clear).rectangle((40, 70, 55, 85), fill=(0, 0, 0))
    ImageDraw.Draw(touching).rectangle((25, 25, 40, 40), fill=(0, 0, 0))
    found = pe.export(old, new, card(), {"X/A2.png": clear, "X/A3.png": touching})
    assert found.unchecked and set(found.numbered) == {"X/A2.png", "X/A3.png"}
    data, why = found.numbered["X/A2.png"]
    expected = np.asarray(card(colour=(0, 255, 0))).copy()
    expected[70:86, 40:56] = 0
    assert why == "" and np.asarray(pe.open_png(data)).tolist() == expected.tolist()
    assert found.numbered["X/A3.png"] == (b"", "A3.png not updated: re-export it in Photoshop")


def test_g2_refuses_a_change_the_render_does_not_show(tmp_path, monkeypatch):
    """With no stored image the parser's re-read stands in for the cross-check (the bake's blank render): a text or stat
    row it reads differently must be drawn with a change, else no PNG (G2_BLANK); past the floor the baseline refuses."""
    old = fixture_psd(tmp_path, card(), "old")
    new = fixture_psd(tmp_path, card(colour=(0, 255, 0)), "new")
    new.has_preview = lambda: False
    def reads(box, name="Square"):
        return lambda a, b: [("the Ability text", box, b, next(l for l in b.descendants() if l.name == name))]
    monkeypatch.setattr(pe, "_rereads", reads((40, 60, 50, 70)))                 # nothing moved there, nothing above
    refused = pe.export(old, new, card())
    assert (refused.outcome, refused.gate, refused.data) == ("refused", "G2", b"")
    assert refused.why == pe.G2_BLANK.format(what="the Ability text")
    monkeypatch.setattr(pe, "_rereads", reads((12, 12, 20, 20)))                 # inside the square
    assert pe.export(old, new, card()).outcome == "re-exported"
    # a layer under the art (Wolfgang's Name): drawn with the layers above it hidden, so no blank render
    same = fixture_psd(tmp_path, card(), "same")
    same.has_preview = lambda: False
    monkeypatch.setattr(pe, "_rereads", reads((12, 12, 20, 20), "Base"))
    covered = pe.export(old, same, card())
    assert (covered.outcome, covered.gate) == ("current", "") and all(l.visible for l in same.descendants())
    monkeypatch.setattr(pe, "_rereads", lambda a, b: [])
    _rows_off(monkeypatch, 10)
    broken = pe.export(old, new, card())
    assert (broken.outcome, broken.gate) == ("refused", "G6") and broken.why.startswith("PNG not made: the bot cannot draw")


@pytest.mark.parametrize("mode, depth", [("RGB", 16)])
def test_g2_still_refuses_a_16_bit_or_cmyk_psd(tmp_path, mode, depth):
    """Not 8-bit RGB with no stored image: the fork's render is no export of it, refused as before on every path (with
    an old PSD, beside its PNG, a new card); an 8-bit RGB PSD with no stored image is rendered (None: no refusal)."""
    from types import SimpleNamespace
    from psd_tools.constants import ColorMode
    assert pe._no_composite(SimpleNamespace(color_mode=ColorMode.RGB, depth=8)) is None
    psd = PSDImage.new(mode, (60, 90), depth=depth)
    psd.save(str(tmp_path / "odd.psd"))
    new = PSDImage.open(str(tmp_path / "odd.psd"))
    assert (new.color_mode, new.depth) == (getattr(ColorMode, mode), depth)
    new.has_preview = lambda: False
    old = fixture_psd(tmp_path, card(), "old")
    for found in (pe.export(old, new, card()), pe.export(None, new, card()), pe.export_new(new)):
        assert (found.outcome, found.gate, found.why) == ("refused", "G2", pe.G2_MODE)


def test_g2_shape_and_region_gates_refuse(tmp_path, monkeypatch):
    """With no stored image the bake's shape and region gates (G7) still stand on the merged PNG."""
    old = fixture_psd(tmp_path, card(), "old")
    new = fixture_psd(tmp_path, card(colour=(0, 255, 0)), "new")
    new.has_preview = lambda: False

    def refuse(*args):
        raise psd_edit.Refused("region", "outside the region")
    monkeypatch.setattr(psd_edit, "gate_region", refuse)
    assert (lambda f: (f.outcome, f.gate, f.why))(pe.export(old, new, card())) == ("refused", "G7", pe.G7)


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
    monkeypatch.setattr(psd_edit, "render_region", lambda psd, box, icc=True, text="preview": drawn.append(box) or draw(psd, box, icc, text))
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


@pytest.mark.parametrize("photoshop", [False])
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
        new.has_preview = lambda: False                                   # G2: no stored image, the render alone decides
        ahead = pe.export(old, new, card(colour=colour))
        assert (ahead.outcome, ahead.unchecked, ahead.data) == ("current", True, b""), colour
        assert ahead.numbers["png_ahead"]["new"]["max"] == 0 and "baseline" not in ahead.numbers
        stale = pe.export(old, new, card())
        assert (stale.outcome, stale.unchecked) == ("re-exported", True) and "png_ahead" not in stale.numbers, colour
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


def test_g2_keeps_a_png_beside_up_to_kept_px_and_tells(tmp_path, monkeypatch):
    """Default 2026-10-24, owner to confirm: with no stored image, a PNG at most KEPT_PX px past BESIDE_CUT of the fork's
    whole-card render (its own noise beside a current PNG) is kept and told (numbers["kept"], KEPT), never redrawn; past
    KEPT_PX it is redrawn where it differs past the cut, within the bar or not (review of 2026-10-24: the whole-card bar
    kept real edits of up to 13,496 px stale)."""
    new = fixture_psd(tmp_path, card(size=(300, 400)), "new")
    new.has_preview = lambda: False
    monkeypatch.setattr(pe, "_texts_drawn", lambda *a: pytest.fail("checked a PNG it keeps"))
    noisy = card(size=(300, 400))
    ImageDraw.Draw(noisy).rectangle((40, 60, 44, 64), fill=(200, 30, 50))         # 20 levels over 25 px
    kept = pe.export(None, new, noisy)
    assert (kept.outcome, kept.unchecked, kept.data, kept.changed_px) == ("current", True, b"", 0)
    assert kept.numbers["kept"] == 25 and "png_ahead" not in kept.numbers
    assert "png_ahead" in pe.export(None, new, card(size=(300, 400))).numbers  # no pixel past 8: current, nothing told
    monkeypatch.undo()
    for levels in (20, 60):                             # 800 px: within the bar (20 levels), past it (60)
        stale = card(size=(300, 400))
        ImageDraw.Draw(stale).rectangle((40, 60, 59, 99), fill=(200, 30, 30 + levels))
        redrawn = pe.export(None, new, stale)
        assert (redrawn.outcome, redrawn.unchecked, redrawn.changed_px) == ("re-exported", True, 800), levels
        assert "kept" not in redrawn.numbers and pe._within(redrawn.numbers["png_vs_render"]) == (levels == 20)
        assert np.asarray(pe.open_png(redrawn.data)).tolist() == np.asarray(card(size=(300, 400))).tolist(), levels


def test_a_new_psd_beside_an_older_png_takes_the_render_where_photoshops_image_moved(tmp_path, monkeypatch):
    """The PNG shows an older card: it takes the fork's render only where Photoshop's image differs from it (D); the
    render's own errors elsewhere never reach it. Errors inside D past the bar refuse."""
    new = fixture_psd(tmp_path, card(colour=(0, 255, 0)), "new")
    draw = psd_edit.render_region

    def off(psd, box, icc=True, text="preview"):     # the fork draws the base 9 levels off, the square right
        image = np.array(draw(psd, box, icc, text).convert("RGB"))
        image[60:, :] = (209, 39, 39)
        return Image.fromarray(image)

    monkeypatch.setattr(psd_edit, "render_region", off)
    found = pe.export(None, new, card())
    assert found.outcome == "re-exported" and found.changed_px == 21 * 21 and found.box == (10, 10, 31, 31)
    assert found.numbers["cross"]["max"] == 0
    assert np.asarray(pe.open_png(found.data)).tolist() == np.asarray(card(colour=(0, 255, 0))).tolist()
    monkeypatch.setattr(psd_edit, "render_region", lambda psd, box, icc=True, text="preview": card(colour=(0, 200, 0)))
    refused = pe.export(None, new, card())
    assert (refused.outcome, refused.gate, refused.data) == ("refused", "G6", b"")
    assert refused.why.startswith("PNG not made: the bot's drawing of your change")


def test_a_drawing_miss_under_the_floor_is_written_and_told(tmp_path, monkeypatch):
    """Owner (2026-10-23): the bot's drawing missing the bar (one row of the 21x21 box 30 levels off: 4.8% bad) no
    longer refuses: the PNG is made and the miss is told per measure; past the floor (10 rows: 48%) it refuses as before."""
    old = fixture_psd(tmp_path, card(), "old")
    new = fixture_psd(tmp_path, card(colour=(0, 255, 0)), "new")
    _rows_off(monkeypatch, 1)
    found = pe.export(old, new, card())
    assert found.outcome == "re-exported" and found.data and found.numbers["region"] == [10, 10, 31, 31]
    assert pe.miss_notes(found) == [f"made with a pixel difference ({what}): bad 4.76%, mean 1.43, max 30"
                                    for what in ("the old PSD drawn against the old PNG", "against the PSD's own image")]
    _rows_off(monkeypatch, 10)
    refused = pe.export(old, new, card())
    assert (refused.outcome, refused.gate, refused.data, refused.misses) == ("refused", "G6", b"", {})
    assert refused.why.startswith("PNG not made: the bot cannot draw this part of the card as its old PNG shows it (47.6%")


def _hidden(monkeypatch, name: str) -> None:
    """The fork drawing nothing for the layer or group so named (the bake's blank-render simulation): every render hides it."""
    draw = psd_edit.render_region

    def blank(psd, box, icc=True, text="preview"):
        layer = next(layer for layer in psd.descendants() if layer.name.strip() == name)
        was, layer.visible = layer.visible, False
        try:
            return draw(psd, box, icc, text)
        finally:
            layer.visible = was
    monkeypatch.setattr(psd_edit, "render_region", blank)


@fork
def test_a_new_card_drawn_blank_in_a_text_box_is_refused(clone, monkeypatch):
    """Review (2026-10-23): over the whole card a blank text box stays under the floor (Shield of the Guilty's Ability
    drawn blank: 2.3% of the card), so each text and stat box is held to it too, and a missed box whose layer the fork
    draws nothing for refuses (the bake's blank render); beside its PNG likewise. The Freezer's Text group drawn blank:
    10.1% of the card, its Ability box past the floor."""
    shield = pe.open_psd(blob(clone, SHIELD))
    _hidden(monkeypatch, "Ability")
    found = pe.export_new(shield)
    assert (found.outcome, found.gate, found.why, found.data) == ("refused", "G6", pe.G6_BLANK.format(what="the Ability text"), b"")
    assert psd_edit.PARITY_BAD_MAX < found.numbers["cross"]["bad_pct"] < psd_edit.PARITY_FLOOR
    box = found.numbers["boxes"]["the Ability text"]
    assert box["blank"] and 5 < box["bad_pct"] < psd_edit.PARITY_FLOOR and box["ink"]["bad_pct"] > 90   # its letters
    old = np.array(pe.convert(shield, pe.composite(shield), True).convert("RGB"))
    x0, y0, x1, y1 = next(layer for layer in shield.descendants() if layer.name == "Ability").bbox
    old[y0:y1, x0:x1] = 255 - old[y0:y1, x0:x1]         # the PNG beside it shows another ability
    beside = pe.export(None, shield, Image.fromarray(old))
    assert (beside.outcome, beside.why) == ("refused", pe.G6_BLANK.format(what="the Ability text"))
    assert beside.numbers["cross"]["bad_pct"] < psd_edit.PARITY_FLOOR
    freezer = pe.open_psd(blob(clone, FREEZER))
    _hidden(monkeypatch, "Text")
    found = pe.export_new(freezer)
    assert found.outcome == "refused" and found.why.startswith("PNG not made: the bot's drawing of the Ability text does not")
    assert found.numbers["cross"]["bad_pct"] < psd_edit.PARITY_FLOOR < found.numbers["boxes"]["the Ability text"]["bad_pct"]


@fork
def test_a_new_card_with_an_empty_text_layer_is_made(clone):
    """Sabre's Ability layer holds no text (bbox 0 x 0): no box, so hiding it (it moves nothing) is no blank drawing; its
    stat labels' small misses (bad 0%, mean about 2.3) are told. The sweep of 2026-10-23 refused it before this rule."""
    found = pe.export_new(pe.open_psd(blob(clone, SABRE)))
    assert found.outcome == "re-exported" and "the Ability text" not in found.numbers["boxes"]
    assert all(m["bad_pct"] <= psd_edit.PARITY_BAD_MAX and not m.get("blank") for m in found.numbers["boxes"].values())


def test_g5_stays_a_refusal_under_the_floor(tmp_path):
    """A stale old PNG is no drawing miss: Photoshop's own image disagrees with it, so no psd-tools fix makes the PNG
    right; G5 refuses a small stale area too (10 x 10 px outside the change: 2% bad)."""
    old = fixture_psd(tmp_path, card(), "old")
    new = fixture_psd(tmp_path, card(colour=(0, 255, 0)), "new")
    stale = card()
    ImageDraw.Draw(stale).rectangle((40, 60, 49, 69), fill=(255, 255, 0))
    found = pe.export(old, new, stale)
    assert (found.outcome, found.gate) == ("refused", "G5") and 0.5 < found.numbers["outside"]["bad_pct"] < psd_edit.PARITY_FLOOR


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


def test_png_check(tmp_path):
    psd = fixture_psd(tmp_path, card(), "a")
    assert pe.png_check(psd, card())["ok"] is True
    assert pe.png_check(psd, card(colour=(255, 255, 255), box=(0, 0, 59, 60)))["ok"] is False
    assert pe.png_check(psd, card(size=(10, 10)))["ok"] is None


# ========================
# Lossless recompression (owner, 2026-10-23)
# ========================
oxipng = pytest.mark.skipif(pe.tool() is None, reason="needs the oxipng CLI (as the card Actions install) or pyoxipng")


@oxipng
@pytest.mark.parametrize("opaque, icc", [(True, "sRGB"), (False, "sRGB"), (True, "photoshop")])
def test_recompress_keeps_the_image_and_its_colour_chunks(opaque, icc):
    """A Photoshop-sized PNG gets several times smaller and stays the same image to every reader: the pixels, the ICC
    profile (Photoshop's sRGB one too: oxipng would make it an sRGB chunk), gAMA and pHYs; the text (Photoshop's XMP)
    goes, and a fully opaque card becomes RGB (SMALLER_PNG_PUSHES.md)."""
    if icc == "photoshop":                      # a card's own: sRGB IEC61966-2.1
        from conftest import CLONE, pinned_clone
        if not (CLONE / ".git").exists():
            pytest.skip(f"no card repo clone at {CLONE}")
        path = pinned_clone() / "Auxiliary/Minions/Mini_Doomer.png"
        icc = pe.open_png(path.read_bytes()).info["icc_profile"]
    image = art()
    if not opaque:
        image.putpixel((0, 0), (1, 2, 3, 0))
    data = photoshop_png(image, icc)
    out = pe.recompress(data, pe.RECOMPRESS_MIN, pe.RECOMPRESS_BYTES)
    assert out is not None and len(out) * 3 < len(data) and pe.same_png(data, out)
    before, after = pe.open_png(data), pe.open_png(out)
    assert after.mode == ("RGB" if opaque else "RGBA") and after.size == before.size and out[24] == 8
    assert np.array_equal(np.asarray(after.convert("RGBA")), np.asarray(before))
    for key in ("icc_profile", "gamma", "dpi"):
        assert after.info[key] == before.info[key], key
    assert not {"XML:com.adobe.xmp", "Comment", "Software", "srgb"} & set(after.info)
    assert pe.recompress(out, pe.RECOMPRESS_MIN) is None                  # done once: nothing more to save


@oxipng
def test_same_png_tells_a_changed_pixel_or_chunk():
    data = photoshop_png(art())
    out = pe.recompress(data)
    assert pe.same_png(data, out) and pe.same_png(data, data)
    assert pe.same_png(data, photoshop_png(art("RGB")))                     # opaque RGBA and its RGB: one image
    image = art()
    image.putpixel((7, 7), (0, 0, 0, 255))
    assert not pe.same_png(data, photoshop_png(image))                      # one pixel
    image.putpixel((7, 7), (250, 200, 40, 254))
    assert not pe.same_png(photoshop_png(image), photoshop_png(art("RGB")))  # one alpha: not the RGB
    plain = art()
    import io
    buf = io.BytesIO()
    plain.save(buf, "PNG")
    assert not pe.same_png(data, buf.getvalue())                            # the colour chunks dropped
    assert not pe.same_png(data, photoshop_png(art(), icc="XYZ"))           # another ICC profile
    assert not pe.same_png(data, b"not a png")


def test_recompress_leaves_16_bit_and_runs_without_oxipng(monkeypatch):
    import io
    buf = io.BytesIO()
    Image.fromarray(np.arange(64 * 64, dtype=np.uint16).reshape(64, 64) * 16).save(buf, "PNG", compress_level=0)
    assert buf.getvalue()[24] == 16 and pe.recompress(buf.getvalue()) is None    # same_png reads 8 bits: not tried
    monkeypatch.setattr(pe, "_oxipng", lambda: None)
    monkeypatch.setattr(pe, "which", lambda name: None)                   # no CLI either
    data = photoshop_png(art())
    assert pe.recompress(data) is None
    image, like = art(), pe.open_png(data)
    plain = io.BytesIO()
    psd_edit.save_png(image, plain, like)
    assert pe.encode(image, like) == plain.getvalue()                       # the bot's own PNG, as before


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
    """One of the 9 refusals of the 34 replayed commits (PUSH_GUARDS.md 16, rulings): Restrained's old PNG did not show
    its old PSD where the author changed it ("(BOT) -1 Action" in the PSD, "-1 Action" in the PNG)."""
    old, new, old_png, _ = pair(clone, RESTRAINED)
    found = pe.export(old, new, old_png)
    assert (found.outcome, found.gate) == ("refused", "G5") and not found.data
    assert found.why.startswith("PNG not made: the PSD and its old PNG already disagreed where you changed it"), found.why


@fork
def test_g2_refuses_a_changed_colour_profile(clone, tmp_path):
    """Abigail with its 'Display' profile removed: no layer record shows it, so G2 called the
    PNG current, or merged a patch drawn without the profile into a PNG exported with it (review of 2026-10-24). The
    checked path refuses G6_UNSEEN; G2 refuses G2_PROFILE."""
    from psd_tools.constants import Resource
    data, png = blob(clone, ABIGAIL[0]), pe.open_png(blob(clone, ABIGAIL[1]))
    psd = pe.open_psd(data)
    del psd.image_resources[Resource.ICC_PROFILE]
    psd.save(str(tmp_path / "new.psd"))
    new = pe.open_psd((tmp_path / "new.psd").read_bytes())
    new.has_preview = lambda: False
    found = pe.export(pe.open_psd(data), new, png)
    assert (found.outcome, found.gate, found.why, found.data) == ("refused", "G6", pe.G2_PROFILE, b"")


def test_shows_no_anchor_in_a_png_showing_neither(tmp_path):
    """The PNG's square is neither PSD's (a stale export): nearer the anchor's by mean, but past the floor of it."""
    v1, v2 = fixture_psd(tmp_path, card(), "v1"), fixture_psd(tmp_path, card(colour=(0, 255, 0)), "v2")
    assert not pe.shows(card(colour=(0, 60, 200)), v1, v2)

