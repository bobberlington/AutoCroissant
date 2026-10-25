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
from helpers import _rows_off, art, card, fixture_psd, photoshop_png

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
PANDAMONIUM = ("02e54e182501c45c4483b57756ea083321c5fa73", "be1760ebc808be51e51a88853a77b66d33cd43fe",
               "85f8030a8b985e0b739a3e3c5000093ce449dd38")                                           # 38c55bd3: the author's PNG
GREEN_MAGE = ("d17ddacd178eb7353dab252651855da1e2d995f7", "cf9e880710d77ad3b10dfa6690a2a285b5d1b9da",
              "09e2dce6f6ed70454f1e52604015cc41678763ff", "41c2439d09d0eb532995338d2646ed7556b52eac")   # 45cf8746: ATK 6, bars only
THIN_MAN = ("eb2b78e1f1b3948e38b8c091515845a68226102d", "8603c0ca93be42722dbe7459a664ec9258df7340",
            "c97731e5d1d510a92640873280e5c31aedd98b69")                                                 # 59956ce4: atk markers
DOGMA = ("389d8d8d2b37aaa81e7da0928dadea3d8a1030cf", "2811461496e61a4896ca74f6db958099a70294d2",
         "d5438e166a74bd24744b0aa509bed51b0541f591")                                                    # 5a9e0fcf: TySh only
SANS = ("97b51c821e8057f1c1a00991172484ee3b30887b", "da0f33d4551435e71a7bd3a64a6f3f5fefa5fa15",
        "6df87ab1668293459015269541c448e4dd3235d9")                                                     # 1b4f36a0: stale push
BLOCK_HOPPER = ("923fdc6d0d874beac87904c5fc68f76737ebaf74", "23f5cd1787d1d8d924738b159acfa523f68e5963")  # 0262b237: raw
BUZZO = ("1b5291a649238a46b19305c3beff46cc8b5b31e4", "c73dbbb392fe7f1c864c8f025ed7665f285bcbbf")         # converted
CREDIT_CARD = ("1a8fad1238acc6a27aff13f4d063a2f5a7476247", "f28d8f754971f1c04b81f47f17af2ae5bf50ad1f",
               "ab309e65d9ab5856bf3771d62c615ef7381871c3", "d65db67b7cca128aa100695dfac003ad873565eb")  # 5a9e0fcf
TAINTED_JACOB = ("c211fca23ed832ed3dee34ccf9fad32d44aa504e", "16a49559c70775d6adbeabe440e0ad7e2602a55a",
                 "b3e0dab8f5f8f001527c9ddf60a2c7d2cef81eef", "6b78b2b4540e06944043b4ef30a6f1c399e03e1f")  # 6c67b1fa
FORSAKEN = ("52ef33f8c0ae8550ae2624f0211516dc1367bac4", "05f2a47b4259fcb15c6cae6de3bba61bb07bd83d",
            "6967c88a1dc5e35908f757ed85e27bbbc825e879", "5ab76110bad1848e14063f06def7d5303c928328")   # 7fd2aa30
BLOON = ("e23643044d9800bde0f81e9018faeca60b782cea", "81bc14f93d36ddd1c11bc4a85ed0610e408702c4",
         "cb295f801aa1f62436051c3f88a14dd76826295b", "3851d2f3955cd96847767974c140be0067f38c15")  # PoD06Rainbow_Bloon 38c55bd3
CLOT_WHITE = ("09dbfff123941024fe7632dc0109d583f655de24", "497eda88775b84cc2ff53e88699a402f7a71b67c",
              "16c15a759a0a361dcbbe3849218b6f35662703d7", "044b16a8a185ee7c402eb864c33233df21157f41")   # d8163103
WITHERING = ("593e02c85f1e8d53cdd169433be5b6d85232dc95", "0f3353c4fa24fed0d4c3d6ff02ba7312fa2dffa8",
             "8ac85a3e811ac273da8f4bb82b405ed7236663bc", "5a44863baf78cf6f0af0a128f1f24e9d71ec619a")   # 5c8cb87b
NEGATION = ("b768755bf1eb811611e6e711f145a2eb72a907a4", "416a0af1545f153bf48ddd9e4b7b50b21a546597",
            "6ba7a20059d18960f06125a8bfedc20a4c02d9e4", "9f187e7d3fcce1d2cdae3f9c12b04b2a888d650e")   # 100511d3: TySh only
ABIGAIL = ("2b68e6a182b67d38e9b7720c035cbdc4f3ab42db", "f7fe63fa4cef78833d18c5e321d40644e7c22a35")    # 'Display' profile
BESIDE_ICC = (("0f422f8af6d609b5d11c6b41900c14d015dad27c", "269472effe95cb6f5cef38878198d7065d69ee53",
               "50f933aca10cc21b12e58686469cdbb72eda3223"),                                       # Stranger 5a9e0fcf
              ("d51961ebbe3bcc2d462bc7ec0fe69fd8fe32800d", "f1000e1188fd3a417a285ba8289e9bca01c98855",
               "f8f5d7a9a0a702e7a49166af05ba27ffdb8b9af1"))                                       # The_Forbidden_One 6d59c949
COVERED = (("3f8e622fe59e7158234b85b8bbf3f9826a65d57a", "638584800e46ebd9dafc67f665b2a8d97ef18920"),  # Wolfgang (5a9e0fcf)
           ("703fad1daade38cf3d388783624a064be94a28e2", "9f50d035fd2cd1f3e1f2d78f7713de0c46b46c40"))  # Doctor Zomboss
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


def test_g2_a_stale_png_within_the_bar_of_both_renders_is_not_ahead(tmp_path):
    """A 3-level recolour over a PNG a level off the old square: within the bar of both renders, so only the mean (nearer
    the old) keeps the stale PNG from being called current."""
    old = fixture_psd(tmp_path, card(colour=(0, 0, 200)), "old")
    new = fixture_psd(tmp_path, card(colour=(0, 0, 203)), "new")
    new.has_preview = lambda: False
    found = pe.export(old, new, card(colour=(0, 0, 201)))
    assert (found.outcome, found.changed_px) == ("re-exported", 21 * 21) and "png_ahead" not in found.numbers


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


@pytest.mark.parametrize("mode, depth", [("RGB", 16), ("CMYK", 8)])
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


def test_g2_beside_updates_numbered_pngs(tmp_path):
    """G8 beside an older PNG with no stored image: a numbered duplicate takes the change when clear of its own title,
    else is told."""
    new = fixture_psd(tmp_path, card(colour=(0, 255, 0)), "new")
    new.has_preview = lambda: False
    clear, touching = card(), card()
    ImageDraw.Draw(clear).rectangle((40, 70, 55, 85), fill=(0, 0, 0))
    ImageDraw.Draw(touching).rectangle((25, 25, 40, 40), fill=(0, 0, 0))
    found = pe.export(None, new, card(), {"X/A2.png": clear, "X/A3.png": touching})
    assert found.unchecked and found.outcome == "re-exported" and set(found.numbered) == {"X/A2.png", "X/A3.png"}
    data, why = found.numbered["X/A2.png"]
    expected = np.asarray(card(colour=(0, 255, 0))).copy()
    expected[70:86, 40:56] = 0
    assert why == "" and np.asarray(pe.open_png(data)).tolist() == expected.tolist()
    assert found.numbered["X/A3.png"] == (b"", "A3.png not updated: re-export it in Photoshop")


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


def test_a_small_recolour_beside_its_png_is_rendered(tmp_path):
    """Owner (2026-10-23): past BESIDE_CUT (8 levels) a new PSD beside its PNG is no longer 'current': a 20-level
    recolour gets the fork's render (or a refusal), never a silent pass."""
    new = fixture_psd(tmp_path, card(), "new")
    recoloured = card()
    ImageDraw.Draw(recoloured).rectangle((40, 60, 44, 64), fill=(200, 30, 50))
    found = pe.export(None, new, recoloured)
    assert found.outcome != "current" and "png_ahead" not in found.numbers


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


def test_a_new_card_drawn_with_a_miss_is_made(tmp_path, monkeypatch):
    """A new card's whole render a row off (1 of 90: 1.1% bad) is made, told; the beside case likewise."""
    new = fixture_psd(tmp_path, card(), "new")
    _rows_off(monkeypatch, 1, start=12)
    found = pe.export_new(new)
    assert found.outcome == "re-exported" and list(found.misses) == ["cross"]
    assert pe.miss_notes(found) == ["made with a pixel difference (against the PSD's own image): bad 1.11%, mean 0.33, max 30"]
    beside = pe.export(None, new, card(colour=(0, 255, 0)))       # D is the square; the row crosses it
    assert beside.outcome == "re-exported" and list(beside.misses) == ["cross"]


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
def test_a_new_card_tells_its_missed_box(clone, tmp_path):
    """A new card, the whole card within the bar, its title box missing it: made, the miss told for the box. The miss is
    painted (The Freezer with the top tenth of its title box inverted in its stored image, saved by the fork): since
    round 10 every title is within the bar (Red_Knife's outer glow noise, the last miss, is matched)."""
    if "preview" not in signature(PSDImage.save).parameters:
        pytest.skip("this psd-tools has no save(preview=)")
    freezer = pe.open_psd(blob(clone, FREEZER))
    stored = np.array(pe.composite(freezer))
    x0, y0, x1, y1 = next(layer for layer in freezer.descendants() if layer.name == "Name").bbox
    y1 = y0 + (y1 - y0) // 10
    stored[y0:y1, x0:x1, :3] = 255 - stored[y0:y1, x0:x1, :3]
    path = tmp_path / "freezer.psd"
    freezer.save(str(path), preview=Image.fromarray(stored))
    found = pe.export_new(PSDImage.open(str(path)))
    assert found.outcome == "re-exported" and found.data and psd_edit._within_bar(found.numbers["cross"])
    assert list(found.misses) == ["the Name text"] and found.numbers["boxes"]["the Ability text"]["bad_pct"] == 0
    missed = found.misses["the Name text"]
    assert pe.miss_notes(found) == ["made with a pixel difference (the Name text against the PSD's own image): bad %.2f%%, "
                                    "mean %.2f, max %d" % (missed["bad_pct"], missed["mean"], missed["max"])]


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
    for line in (pe.G2_BLANK, pe.G2_MODE, pe.G5, pe.G5_IN, pe.G6, pe.G6_BASE, pe.G6_NEW, pe.G6_STALE, pe.G6_UNSEEN, pe.G7):
        assert line.startswith("PNG not made: ") and line.endswith("export it in Photoshop"), line


def test_png_check(tmp_path):
    psd = fixture_psd(tmp_path, card(), "a")
    assert pe.png_check(psd, card())["ok"] is True
    assert pe.png_check(psd, card(colour=(255, 255, 255), box=(0, 0, 59, 60)))["ok"] is False
    assert pe.png_check(psd, card(size=(10, 10)))["ok"] is None


# ========================
# Lossless recompression (owner, 2026-10-23)
# ========================
oxipng = pytest.mark.skipif(find_spec("oxipng") is None, reason="needs pyoxipng, as card-guards.yml installs (none on 3.14)")


@oxipng
@pytest.mark.parametrize("mode", ["RGBA", "RGB"])
def test_recompress_keeps_the_image_and_its_chunks(mode):
    """A Photoshop-sized PNG gets several times smaller and stays the same file to every reader: the header (an opaque
    RGBA stays RGBA: the bake and guard 1 read RGB and RGBA only), the pixels, the ICC profile, gAMA, pHYs and the text."""
    data = photoshop_png(art(mode))
    out = pe.recompress(data, pe.RECOMPRESS_MIN, pe.RECOMPRESS_BYTES)
    assert out is not None and len(out) * 3 < len(data) and pe.same_png(data, out)
    assert out[:33] == data[:33]                                            # signature and IHDR, byte for byte
    before, after = pe.open_png(data), pe.open_png(out)
    assert after.mode == before.mode == mode and np.array_equal(np.asarray(after), np.asarray(before))
    for key in ("icc_profile", "gamma", "dpi", "XML:com.adobe.xmp", "Comment", "Software"):
        assert after.info[key] == before.info[key], key
    assert pe.recompress(out, pe.RECOMPRESS_MIN) is None                  # done once: nothing more to save


@oxipng
def test_same_png_tells_a_changed_pixel_or_chunk():
    data = photoshop_png(art())
    out = pe.recompress(data)
    assert pe.same_png(data, out) and pe.same_png(data, data)
    image = art()
    image.putpixel((7, 7), (0, 0, 0, 255))
    assert not pe.same_png(data, photoshop_png(image))                      # one pixel
    assert not pe.same_png(data, photoshop_png(art("RGB")))                 # the colour type
    plain = art()
    import io
    buf = io.BytesIO()
    plain.save(buf, "PNG")
    assert not pe.same_png(data, buf.getvalue())                            # the chunks dropped
    assert not pe.same_png(data, photoshop_png(art(), icc="XYZ"))           # another ICC profile
    assert not pe.same_png(data, b"not a png")


@oxipng
def test_recompress_wants_its_saving():
    data = photoshop_png(art(size=(80, 120)))
    out = pe.recompress(data)
    assert out is not None and len(data) - len(out) < pe.RECOMPRESS_BYTES
    assert pe.recompress(data, pe.RECOMPRESS_MIN, pe.RECOMPRESS_BYTES) is None    # too small to be worth a commit
    assert pe.worth(1000, 899, 0.10, 0) and not pe.worth(1000, 900, 0.10, 0) and not pe.worth(1000, 1000, 0, 0)


def test_recompress_leaves_16_bit_and_runs_without_oxipng(monkeypatch):
    import io
    buf = io.BytesIO()
    Image.fromarray(np.arange(64 * 64, dtype=np.uint16).reshape(64, 64) * 16).save(buf, "PNG", compress_level=0)
    assert buf.getvalue()[24] == 16 and pe.recompress(buf.getvalue()) is None    # same_png reads 8 bits: not tried
    monkeypatch.setattr(pe, "_oxipng", lambda: None)
    data = photoshop_png(art())
    assert pe.recompress(data) is None
    image, like = art(), pe.open_png(data)
    plain = io.BytesIO()
    psd_edit.save_png(image, plain, like)
    assert pe.encode(image, like) == plain.getvalue()                       # the bot's own PNG, as before


@oxipng
def test_the_bots_own_pngs_are_recompressed():
    """encode (guard 1's re-exported, new and numbered PNGs): smaller than PIL's deflate, the same image."""
    import io
    image, like = art(), pe.open_png(photoshop_png(art()))
    plain = io.BytesIO()
    psd_edit.save_png(image, plain, like)
    made = pe.encode(image, like)
    assert len(made) < len(plain.getvalue()) and pe.same_png(plain.getvalue(), made)


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
def test_g2_real_cards_reread_the_parser_and_match_the_checked_png(clone):
    """G2 on real pushes (no stored image): the parser's re-read finds the change (Dirty Bed's Ability, Springtrap's
    Name), and the PNG made is the checked export's, pixel for pixel (the composite only ever cross-checks). Beside its
    stale PNG (thousands of px past BESIDE_CUT, within the whole-card bar) the PSD is redrawn within the bar of the
    author's export, never kept stale (review of 2026-10-24)."""
    for shas, what in ((DIRTY_BED, ["the Ability text"]), (SPRINGTRAP, ["the Name text"])):
        old, new, old_png, truth = pair(clone, shas)
        checked = pe.export(old, new, old_png)
        new.has_preview = lambda: False
        found = pe.export(old, new, old_png)
        assert (found.outcome, found.unchecked, found.numbers["reread"]) == ("re-exported", True, what), what
        assert "cross" not in found.numbers and np.array_equal(np.asarray(found.image), np.asarray(checked.image)), what
        beside = pe.export(None, new, old_png)
        assert (beside.outcome, beside.unchecked) == ("re-exported", True) and "kept" not in beside.numbers, what
        assert pe._within(beside.numbers["png_vs_render"]) and beside.changed_px > pe.KEPT_PX, what
        assert psd_edit._within_bar(psd_edit._measure(beside.image, truth)), what


def _blind(monkeypatch, real):
    """The fork's render (real) ignoring layer visibility: a text hidden draws the same (the bake's blank render case)."""
    whole = {}

    def render(psd, box, icc, numbers, *rest):
        if id(psd) not in whole:
            whole[id(psd)] = real(psd, (0, 0, psd.width, psd.height), icc, numbers, *rest)
        return whole[id(psd)].crop(box)
    monkeypatch.setattr(pe, "_render", render)


@fork
def test_g2_refuses_a_text_drawn_blank_on_real_cards(clone, monkeypatch):
    """With no stored image, a text the render does not show refuses G2_BLANK on every path: an old PSD's re-read change
    the render did not move (_unchecked), and a card drawn whole whose text box does not change when the text is hidden
    (_beside_unchecked, export_new)."""
    old, new, old_png, _ = pair(clone, DIRTY_BED)
    new.has_preview = lambda: False
    real = pe._render
    monkeypatch.setattr(pe, "_render", lambda psd, box, icc, numbers, *rest: real(old if psd is new else psd, box, icc, numbers, *rest))
    found = pe.export(old, new, old_png)                                  # the Ability drawn as it was
    assert (found.outcome, found.gate, found.why) == ("refused", "G2", pe.G2_BLANK.format(what="the Ability text"))
    _blind(monkeypatch, real)
    older = old_png.copy()                              # past KEPT_PX, else kept unchecked (default 2026-10-24)
    ImageDraw.Draw(older).rectangle((0, 0, 399, 399), fill=(0, 0, 0))
    for found in (pe.export(None, new, older), pe.export_new(new)):       # beside an older PNG; a new card
        assert (found.outcome, found.gate, found.data) == ("refused", "G2", b"") and found.unchecked
        assert found.why.startswith("PNG not made: the bot draws nothing for the ") and found.why.endswith(pe.EXPORT)


@fork
def test_g2_rereads_only_texts_whose_record_changed(clone):
    """The re-read demands a drawn change only of a text whose record changed (review 2026-10-24): stat digits come from
    marker layers that draw nothing (The Thin Man's 'Atk Dark/8' hidden, '7' shown) and a TySh-only edit draws nothing
    new (Dogma's title case), so both are current, as on the checked path. Green Mage's ATK fix (bars only) and Sans
    pushed stale (its Ability and a def marker) are drawn as the checked path draws them."""
    for shas in (THIN_MAN, DOGMA):
        old, new, old_png, _ = pair(clone, shas)
        new.has_preview = lambda: False
        found = pe.export(old, new, old_png)
        assert (found.outcome, found.unchecked, found.data, found.numbers["reread"]) == ("current", True, b"", []), shas
    for shas, what in ((GREEN_MAGE, []), (SANS, ["the Ability text"])):
        old, new, old_png, _ = pair(clone, shas)
        checked = pe.export(old, new, old_png)
        new.has_preview = lambda: False
        found = pe.export(old, new, old_png)
        assert (found.outcome, found.unchecked, found.numbers["reread"]) == ("re-exported", True, what), shas
        assert np.array_equal(np.asarray(found.image), np.asarray(checked.image)), shas


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
    show its old PSD where the author changed it ("(BOT) -1 Action" in the PSD, "-1 Action" in the PNG); 20 Creature
    Types' header, which Photoshop redrew from a type layer whose stored raster did not change (the composite
    cross-check refuses what the render cannot see). Defcon Panda's Ability drop shadow, refused until the owner's
    ruling of 2026-10-23, then made with the miss told; since cards-20261024 (round 12's text diagonals) the fork draws
    its edge as Photoshop does: made, nothing to tell."""
    for shas, gate, why in ((RESTRAINED, "G5", "PNG not made: the PSD and its old PNG already disagreed where you changed it"),
                            (CREATURE_TYPES, "G6", pe.G6_UNSEEN)):
        old, new, old_png, _ = pair(clone, shas)
        found = pe.export(old, new, old_png)
        assert (found.outcome, found.gate) == ("refused", gate) and found.why.startswith(why), (shas, found.why)
        assert not found.data
    old, new, old_png, _ = pair(clone, DEFCON_PANDA)
    found = pe.export(old, new, old_png)
    assert found.outcome == "re-exported" and found.data and not found.misses
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


@fork
def test_a_png_pushed_before_its_psd_with_no_stored_image_is_current(clone):
    """Pandamonium 38c55bd3 with the author's export pushed first and the new PSD's stored image dropped (G2): the fork
    draws its gradient Ability up to 9 levels off (mean 3.15 over F, no bad pixel), the old render 82% bad, so the PNG
    is current (png_ahead), as on the checked path; it was re-exported over the author's export with a false miss note
    (review of 2026-10-24)."""
    old, new, authors, _ = pair(clone, PANDAMONIUM)
    new.has_preview = lambda: False
    found = pe.export(old, new, authors)
    assert (found.outcome, found.unchecked, found.data, found.misses) == ("current", True, b"", {})
    ahead = found.numbers["png_ahead"]
    assert not pe._within(ahead["new"]) and ahead["new"]["bad_pct"] == 0 and not pe._within(ahead["old"]), ahead


@fork
def test_g2_keeps_the_authors_png_beside_its_psd(clone):
    """The same with no stored image (default 2026-10-24, owner to confirm): the fork's whole-card render passes 8 levels
    beside the author's export over a few px (Defcon Panda's shadow, 33; Dirty Bed's stays within 8), so the PNG is kept
    and told, not redrawn with the fork's noise. Past KEPT_PX (Pandamonium's gradient, 1,589 px) it is redrawn, within
    the bar of the author's export."""
    seen = {}
    for psd, png in ((DIRTY_BED[1], DIRTY_BED[3]), (DEFCON_PANDA[1], "60bc9b5a4e2b271dc1c74e8eaad3ce6ef626c482")):
        new = pe.open_psd(blob(clone, psd))
        new.has_preview = lambda: False
        kept = pe.export(None, new, pe.open_png(blob(clone, png)))
        assert (kept.outcome, kept.unchecked, kept.data) == ("current", True, b""), psd
        seen[psd] = kept.numbers["png_vs_render"]["max"], kept.numbers.get("kept")
    assert seen[DEFCON_PANDA[1]][0] > pe.BESIDE_CUT and 0 < seen[DEFCON_PANDA[1]][1] <= pe.KEPT_PX, seen
    new, authors = pe.open_psd(blob(clone, PANDAMONIUM[1])), pe.open_png(blob(clone, PANDAMONIUM[2]))
    new.has_preview = lambda: False
    redrawn = pe.export(None, new, authors)
    assert (redrawn.outcome, redrawn.unchecked) == ("re-exported", True) and redrawn.changed_px > pe.KEPT_PX
    assert psd_edit._within_bar(psd_edit._measure(redrawn.image, authors))


@fork
def test_g2_draws_a_raw_export_raw(clone):
    """A card exported raw (ICC unconverted, 21 cards) has no vote with no stored image, so G2 drew it converted:
    Block_Hopper beside its own PNG was redrawn over 51% of the card, and over an old PSD the baseline missed (mean
    9.14, told) (review of 2026-10-24). Both modes are drawn and voted as icc_vote votes: raw, current; a converted card
    (Buzzo) stays converted; Speed hidden over the old PSD (neither stored) is drawn raw, as with the stored image's
    vote."""
    for (psd, png), mode in ((BLOCK_HOPPER, "raw"), (BUZZO, "converted")):
        new = pe.open_psd(blob(clone, psd))
        new.has_preview = lambda: False
        found = pe.export(None, new, pe.open_png(blob(clone, png)))
        assert (found.outcome, found.icc, found.data, found.numbers.get("kept")) == ("current", mode, b"", None), psd
        vote = found.numbers["icc_vote"]
        assert vote["px"] >= psd_edit.ICC_MIN_PX and (vote["raw"] >= psd_edit.ICC_RAW_VOTE) is (mode == "raw"), vote
    data, png = blob(clone, BLOCK_HOPPER[0]), pe.open_png(blob(clone, BLOCK_HOPPER[1]))
    found = {}
    for stored in (True, False):
        old, new = pe.open_psd(data), pe.open_psd(data)
        speed = next(l for l in new.descendants() if l.kind == "type" and l.is_visible() and l.name.strip() == "Speed")
        speed.visible = False
        new.has_preview = lambda: False
        if not stored:
            old.has_preview = lambda: False
        found[stored] = pe.export(old, new, png)
        assert (found[stored].outcome, found[stored].icc, found[stored].misses) == ("re-exported", "raw", {}), stored
        assert pe._within(found[stored].numbers["baseline"]), stored
    assert np.array_equal(np.asarray(found[True].image), np.asarray(found[False].image))


@fork
def test_g2_votes_the_icc_mode_as_icc_vote_does(clone):
    """Tainted_Jacob 6c67b1fa pushed stale with neither PSD stored: the two modes' means over the change were a near tie
    (0.647 converted, 0.646 raw), so it was drawn raw, 1,132 px past 24 levels from the author's export (review of
    2026-10-24). No evidence in the box (no pixel the modes set ICC_MOVED apart), so the whole old card votes:
    converted, the checked export pixel for pixel."""
    old, new, old_png, truth = pair(clone, TAINTED_JACOB)
    checked = pe.export(old, new, old_png)
    old.has_preview = new.has_preview = lambda: False
    found = pe.export(old, new, old_png)
    assert (found.outcome, found.icc, checked.icc) == ("re-exported", "converted", "converted")
    assert found.numbers["icc_vote"]["px"] < psd_edit.ICC_MIN_PX <= found.numbers["icc_vote_card"]["px"], found.numbers
    assert np.array_equal(np.asarray(found.image), np.asarray(checked.image))
    assert int((pe._diff(found.image, truth) > psd_edit.PARITY_THRESH).sum()) == 0


@fork
def test_shows_the_anchor_the_png_shows(clone):
    """The burst anchor (review of 2026-10-24): the PNG before Credit_Card's 5a9e0fcf shows the PSD before it (near:
    bad 11.9%, mean 8.7, the fork's text without the preview's blends; far: mean 160), with and without stored images;
    the author's export shows the PSD after it, never before. Pandamonium's author's export shows its own PSD though the
    fork draws the gradient Ability up to 9 levels off (mean 6.2)."""
    for stored in (True, False):
        v1, v2 = pe.open_psd(blob(clone, CREDIT_CARD[0])), pe.open_psd(blob(clone, CREDIT_CARD[1]))
        if not stored:
            v1.has_preview = v2.has_preview = lambda: False
        before, authors = pe.open_png(blob(clone, CREDIT_CARD[2])), pe.open_png(blob(clone, CREDIT_CARD[3]))
        assert pe.shows(before, v1, v2) and not pe.shows(authors, v1, v2) and pe.shows(authors, v2, v1), stored
    v1, v2, authors, _ = pair(clone, PANDAMONIUM)
    v1.has_preview = v2.has_preview = lambda: False
    assert pe.shows(authors, v2, v1) and not pe.shows(authors, v1, v2)


@fork
def test_g2_a_text_under_the_art_is_no_blank_render(clone):
    """Wolfgang's Name under a smart object and Doctor Zomboss's second Name under 'Layer 2': Photoshop draws none of
    either, so hiding it moves no pixel, and G2 refused both new cards and beside a stale PNG (G2_BLANK) where the
    checked path makes them (review of 2026-10-24). With the layers above hidden the text draws: made as with a stored
    image."""
    for psd, png in COVERED:
        checked = pe.export_new(pe.open_psd(blob(clone, psd)))
        new = pe.open_psd(blob(clone, psd))
        new.has_preview = lambda: False
        made = pe.export_new(new)
        assert (checked.outcome, made.outcome) == ("re-exported", "re-exported"), (psd, made.why)
        assert np.array_equal(np.asarray(made.image), np.asarray(checked.image)), psd
        stale = pe.open_png(blob(clone, png)).copy()
        ImageDraw.Draw(stale).rectangle((600, 300, 619, 319), fill=(0, 0, 0))     # 400 px: past KEPT_PX
        beside = pe.export(None, new, stale)
        assert beside.outcome == "re-exported", (psd, beside.why)


@fork
def test_g2_draws_the_new_psd_with_the_old_ones_text_blends(clone):
    """Forsaken_City 7fd2aa30, the author's PNG pushed first, then the PSD saved without its stored image (the old one
    kept): the old PSD was drawn with its stored image's text blends, the new all in the text gamma, so F took unchanged
    text edges and the author's export was overwritten (2,146 px past 8) with a false miss note (review of 2026-10-24).
    The new is drawn with the old's stored image as its text reference (the bake's render): current, nothing told.
    Pushed stale, the PNG made is within 2 levels of the author's. Drawn all in the gamma on both, Withering 5c8cb87b
    lost its edited Withering text's blend change (1,664 px past 24); the edited layers pinned to the gamma, it is the
    checked PNG pixel for pixel."""
    old, new, stale, authors = pair(clone, FORSAKEN)
    new.has_preview = lambda: False
    first = pe.export(old, new, authors)
    assert (first.outcome, first.data, first.misses, pe.unchecked_line(first)) == ("current", b"", {}, "")
    assert "png_ahead" in first.numbers
    old, new, _, _ = pair(clone, FORSAKEN)
    new.has_preview = lambda: False
    made = pe.export(old, new, stale)
    assert made.outcome == "re-exported" and int(pe._diff(made.image, authors).max()) <= pe.TOL
    old, new, stale, authors = pair(clone, WITHERING)
    checked = pe.export(old, new, stale)
    new.has_preview = lambda: False
    made = pe.export(old, new, stale)
    assert made.outcome == "re-exported" and np.array_equal(np.asarray(made.image), np.asarray(checked.image))
    assert int((pe._diff(made.image, authors) > 24).sum()) == 0


@fork
@pytest.mark.parametrize("old_stored", [True, False])
def test_g2_a_png_pushed_first_and_drawn_a_little_off_is_current(clone, old_stored):
    """PoD06Rainbow_Bloon 38c55bd3 and Clot_White d8163103 with the author's PNG pushed first and no stored image: the
    fork draws the change a little off (Bloon's 0.80% bad, past PARITY_BAD_MAX), the old render far, so the PNG is
    current; it was overwritten with a false miss note (review of 2026-10-24)."""
    for shas in (BLOON, CLOT_WHITE):
        old, new, _, authors = pair(clone, shas)
        new.has_preview = lambda: False
        if not old_stored:
            old.has_preview = lambda: False
        found = pe.export(old, new, authors)
        assert (found.outcome, found.data, found.misses) == ("current", b"", {}), (shas[1], found.numbers)
        ahead = found.numbers["png_ahead"]
        assert not psd_edit._past_floor(ahead["new"]) and not pe._within(ahead["old"]), ahead
        if shas is BLOON:
            assert ahead["new"]["bad_pct"] > psd_edit.PARITY_BAD_MAX, ahead


@fork
def test_g2_a_style_edit_keeps_the_png_and_tells(clone):
    """22_Negation 100511d3: ten type layers changed only their TySh (new style runs), Photoshop's image moved over
    4,723 px, the render over none. The checked path refuses (G6_UNSEEN); with no stored image the PNG was called
    current, nothing told (review of 2026-10-24). Kept, told (RESTYLED). Dogma's case edit (TySh only) is told too;
    The Thin Man's markers changed no TySh: current, nothing told."""
    old, new, old_png, _ = pair(clone, NEGATION)
    new.has_preview = lambda: False
    found = pe.export(old, new, old_png)
    assert (found.outcome, found.data, found.changed_px) == ("current", b"", 0)
    assert len(found.numbers["restyled"]) == 10 and pe.unchecked_line(found).startswith("PNG kept as it is: ")
    assert pe.unchecked_line(found).endswith(pe.EXPORT) and " and 9 more, " in pe.unchecked_line(found)
    old, new, _, _ = pair(clone, NEGATION)
    for psd in (old, new):                              # a hidden text's style draws nothing either way
        for layer in psd.descendants():
            if layer.kind == "type":
                layer.visible = False
    assert pe._restyled(old, new) == []
    for shas, told in ((DOGMA, True), (THIN_MAN, False)):
        old, new, old_png, _ = pair(clone, shas)
        new.has_preview = lambda: False
        found = pe.export(old, new, old_png)
        assert found.outcome == "current" and bool(pe.unchecked_line(found)) is told, shas


@fork
@pytest.mark.parametrize("hide", [False, True])
def test_g2_refuses_a_changed_colour_profile(clone, tmp_path, hide):
    """Abigail with its 'Display' profile removed (and its Ability hidden): no layer record shows it, so G2 called the
    PNG current, or merged a patch drawn without the profile into a PNG exported with it (review of 2026-10-24). The
    checked path refuses G6_UNSEEN; G2 refuses G2_PROFILE."""
    from psd_tools.constants import Resource
    data, png = blob(clone, ABIGAIL[0]), pe.open_png(blob(clone, ABIGAIL[1]))
    psd = pe.open_psd(data)
    del psd.image_resources[Resource.ICC_PROFILE]
    if hide:
        next(l for l in psd.descendants() if l.kind == "type" and l.name.strip() == "Ability").visible = False
    psd.save(str(tmp_path / "new.psd"))
    new = pe.open_psd((tmp_path / "new.psd").read_bytes())
    new.has_preview = lambda: False
    found = pe.export(pe.open_psd(data), new, png)
    assert (found.outcome, found.gate, found.why, found.data) == ("refused", "G6", pe.G2_PROFILE, b"")


@fork
def test_g2_votes_over_the_box_when_the_canvas_grew(clone, tmp_path):
    """Abigail's canvas grown 40 px, its Name hidden, the PNG of the grown card pushed first: with too little evidence
    in the box the old card (1200x2400) was voted against the new PNG (1200x2440) and raised (review of 2026-10-24).
    The box's evidence alone: current."""
    data = blob(clone, ABIGAIL[0])
    psd = pe.open_psd(data)
    psd._record.header.height += 40
    next(l for l in psd.descendants() if l.kind == "type" and l.is_visible() and l.name.strip() == "Name").visible = False
    psd.save(str(tmp_path / "grown.psd"))
    grown = pe.open_psd((tmp_path / "grown.psd").read_bytes())
    png = pe.like_mode(psd_edit.render_region(grown, (0, 0, grown.width, grown.height)), "RGBA")
    for old_stored in (True, False):
        old, new = pe.open_psd(data), pe.open_psd((tmp_path / "grown.psd").read_bytes())
        new.has_preview = lambda: False
        if not old_stored:
            old.has_preview = lambda: False
        found = pe.export(old, new, png)
        assert found.outcome == "current" and "icc_vote_card" not in found.numbers, (old_stored, found.numbers)


@fork
def test_g2_beside_votes_by_the_pixels_one_mode_shows(clone):
    """Stranger 5a9e0fcf and The_Forbidden_One 6d59c949 beside the PNG before the edit, no stored image: counting the
    pixels neither mode shows (the author's change) drew them raw, 893 and 1,004 px past 24 levels from the author's
    export (review of 2026-10-24, round 7). Converted, none past 24."""
    for psd, before, authors in BESIDE_ICC:
        new = pe.open_psd(blob(clone, psd))
        new.has_preview = lambda: False
        found = pe.export(None, new, pe.open_png(blob(clone, before)))
        assert (found.outcome, found.icc) == ("re-exported", "converted"), (psd, found.numbers.get("icc_vote"))
        assert int((pe._diff(found.image, pe.open_png(blob(clone, authors))) > 24).sum()) == 0, psd


def test_icc_evidence_counts_the_pixels_one_mode_shows():
    """_icc_evidence over the pixels the modes draw ICC_MOVED apart where one is within ICC_MOVED of the PNG; _raw wants
    ICC_MIN_PX of them."""
    converted, raw = Image.new("RGB", (40, 10), (100, 100, 100)), Image.new("RGB", (40, 10), (140, 140, 140))
    png = np.full((10, 40, 3), 140, np.uint8)                       # 20 columns raw, 10 converted, 10 neither (250)
    png[:, 20:30] = 100
    png[:, 30:] = 250
    share, px = pe._icc_evidence(converted, raw, Image.fromarray(png))
    assert (share, px) == (round(200 / 300, 3), 300)
    assert not pe._raw((1.0, psd_edit.ICC_MIN_PX - 1)) and pe._raw((psd_edit.ICC_RAW_VOTE, psd_edit.ICC_MIN_PX))


def test_shows_a_small_recolour_by_the_mean(tmp_path):
    """Push A the PNG of v2 (the square 15 levels greener), B the PSD v2: the PNG shows v2, not the anchor v1, though it
    is within the floor of both: the mean tells them apart."""
    v1, v2 = fixture_psd(tmp_path, card(), "v1"), fixture_psd(tmp_path, card(colour=(0, 15, 255)), "v2")
    assert not pe.shows(card(colour=(0, 15, 255)), v1, v2) and pe.shows(card(), v1, v2)


def test_shows_no_anchor_in_a_png_showing_neither(tmp_path):
    """The PNG's square is neither PSD's (a stale export): nearer the anchor's by mean, but past the floor of it."""
    v1, v2 = fixture_psd(tmp_path, card(), "v1"), fixture_psd(tmp_path, card(colour=(0, 255, 0)), "v2")
    assert not pe.shows(card(colour=(0, 60, 200)), v1, v2)


def test_g2_a_text_turned_to_opacity_0_is_reread(clone):
    """A type layer at opacity 0 draws nothing: its text is reread as hidden (a change to draw), not as unchanged."""
    old, new = pe.open_psd(blob(clone, DIRTY_BED[1])), pe.open_psd(blob(clone, DIRTY_BED[1]))
    layer = next(l for l in new.descendants() if l.kind == "type" and l.is_visible() and str(l.text or "").strip())
    layer.opacity = 0
    assert pe._type_layers(new)[layer.layer_id][2] is False
    assert [what for what, *_ in pe._rereads(old, new)] == [f"the {layer.name.strip()} text"]


def test_r10_flags_the_lost_gradient_maps(clone):
    """At 6e50c3ee (2.1), before the gradient maps were restored on the branch."""
    for rel, ok in (("Creatures/Kirby/5 Stars/Dark_Mind_Eye", False), ("Items/Active/2 Stars/Dirty_Bed", True)):
        psd, png = at_rev(clone, "6e50c3ee", rel + ".psd"), at_rev(clone, "6e50c3ee", rel + ".png")
        found = pe.png_check(pe.open_psd(psd), pe.open_png(png))
        assert found["ok"] is ok, (rel, found)

