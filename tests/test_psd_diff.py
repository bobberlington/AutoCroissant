"""
psd_diff (what changed inside a card PSD, for the push guards' comment and post): made-up PSDs for each kind of change and
each re-save that must say nothing, and real pushes from the card clone's history (skipped without it) whose Photoshop
re-saves rewrote dozens of layer records around a one-word edit.
"""
from io import BytesIO

import pytest
from PIL import Image
from psd_tools import PSDImage
from psd_tools.api.layers import Group, PixelLayer
from psd_tools.constants import BlendMode, Compression, Tag

import commands.psd_diff as pdf
from conftest import at_rev

# ========================
# Configuration
# ========================
TEMPLE = "Auxiliary/Items/The_Temple_of_Ahn'Qiraj.psd"
GLAIVE = "Creatures/FTL/4 Stars/Glaive_Main_Cannon.psd"
HEAVY_MOLE = "Creatures/Kirby/3 Stars/Heavy_Mole.psd"
LAZARUS = "Creatures/Isaac/3 Stars/Tainted_Lazarus.psd"
COOLDOWNS = ("9f0c9ece", "11dc6114")            # Zomoroko's push of 2026-10-22 05:54 (cooldown tags, 50 PSDs)


# ========================
# Made-up cards
# ========================
def _card(tmp_path, name, change=None, closed=False, compression=Compression.RLE) -> PSDImage:
    """Base, Square and a Stars group of two stars, with layer ids as Photoshop gives them; change(psd) edits it before
    the save. Saved and reopened."""
    psd = PSDImage.new("RGB", (60, 90))
    psd.append(PixelLayer.frompil(Image.new("RGB", (60, 90), (200, 30, 30)), psd, "Base"))
    psd.append(PixelLayer.frompil(Image.new("RGB", (20, 20), (0, 0, 255)), psd, "Square", top=10, left=10,
                                  compression=compression))
    stars = Group.new(psd, "Stars", open_folder=not closed)
    for i in range(2):
        stars.append(PixelLayer.frompil(Image.new("RGB", (8, 8), (255, 215, 0)), psd, "star", top=60, left=10 + 12 * i))
    for n, layer in enumerate(psd.descendants(), 1):
        layer._record.tagged_blocks.set_data(Tag.LAYER_ID, n)
    if change:
        change(psd)
    path = tmp_path / f"{name}.psd"
    psd.save(str(path))
    return PSDImage.open(str(path))


def test_a_resave_says_nothing(tmp_path):
    assert pdf.changes(_card(tmp_path, "a"), _card(tmp_path, "b")) == []


def test_a_group_opened_or_closed_says_nothing(tmp_path):
    """The Layers panel's twisty is in lsct: the record changes, the card does not."""
    a, b = _card(tmp_path, "a"), _card(tmp_path, "b", closed=True)
    blocks = [psd.find("Stars")._record.tagged_blocks[Tag.SECTION_DIVIDER_SETTING].tobytes() for psd in (a, b)]
    assert blocks[0] != blocks[1]
    assert pdf.changes(a, b) == []


def test_equal_pixels_encoded_otherwise_say_nothing(tmp_path):
    """Photoshop re-encodes channels (Glaive's Laser Beam mask: 22,269 -> 23,144 RLE bytes, the same pixels)."""
    a, b = _card(tmp_path, "a"), _card(tmp_path, "b", compression=Compression.ZIP)
    data = [b"".join(c.data for c in psd.find("Square")._channels) for psd in (a, b)]
    assert data[0] != data[1]
    assert pdf.changes(a, b) == []


def test_each_layer_change_is_told(tmp_path):
    def edit(psd):
        square = psd.find("Square")
        square.visible, square.opacity, square.blend_mode = False, 128, BlendMode.MULTIPLY
        list(psd.find("Stars"))[0].name = "Gold {star}"          # braces in a name are text
        psd.append(PixelLayer.frompil(Image.new("RGB", (5, 5)), psd, "Ability FX: Frozen"))
    assert pdf.changes(_card(tmp_path, "a"), _card(tmp_path, "b", edit)) == [
        "added Ability FX: Frozen", "Square hidden", "star renamed 'Gold {star}'", "Square opacity 100% -> 50%",
        "Square blend Normal -> Multiply"]


def test_moves_pixels_and_groups(tmp_path):
    a = _card(tmp_path, "a")

    def moved(psd):
        psd.find("Square").offset = (14, 10)
    assert pdf.changes(a, _card(tmp_path, "moved", moved)) == ["Square moved +4,+0 px"]

    def painted(psd):
        psd.remove(psd.find("Square"))              # painted over: the same name, new pixels
        psd.insert(1, PixelLayer.frompil(Image.new("RGB", (20, 20), (0, 255, 0)), psd, "Square", top=10, left=10))
    assert pdf.changes(a, _card(tmp_path, "painted", painted)) == ["Square pixels changed"]

    def ungrouped(psd):
        psd.remove(psd.find("Stars"))
    assert pdf.changes(a, _card(tmp_path, "ungrouped", ungrouped)) == ["removed Stars group (2 layers)"]

    def both_hidden(psd):
        for star in psd.find("Stars"):
            star.visible = False
    assert pdf.changes(a, _card(tmp_path, "hidden", both_hidden)) == ["star x2 hidden"]


def test_layers_moved_into_a_group_or_restacked(tmp_path):
    a = _card(tmp_path, "a")

    def into(psd):
        square = psd.find("Square")
        psd.remove(square)
        psd.find("Stars").append(square)
    assert pdf.changes(a, _card(tmp_path, "into", into)) == ["Square moved into Stars"]

    def restack(psd):
        base = psd.find("Base")
        psd.remove(base)
        psd.append(base)                            # Base from the bottom to the top
    assert pdf.changes(a, _card(tmp_path, "restack", restack)) == ["Base restacked"]


# ========================
# Text
# ========================
def test_text_change_quotes_short_texts_whole():
    assert pdf.text_change("White CLot", "White Clot") == "'White CLot' -> 'White Clot'"


def test_text_change_quotes_the_edited_words_of_a_long_text():
    old = "Gain 2 ATK when an ally dies. Deal 6 damage to every enemy creature on the field.(1PT)"
    new = "Gain 2 ATK when an ally dies. Deal 7 damage to every enemy creature on the field.(1CD)"
    assert pdf.text_change(old, new) == "'6' -> '7', 'field.(1PT)' -> 'field.(1CD)'"
    assert pdf.text_change(old, old.replace("an ally", "an  ally")) == ""          # spacing alone
    inserted = pdf.text_change(old, old.replace("every enemy", "every [fire] enemy"))
    assert inserted == "'every enemy' -> 'every [fire] enemy'"                  # an insertion keeps a word each side
    assert pdf.text_change(old, "Nothing like it at all, a new ability for a new card entirely.").startswith("rewritten (")


def test_text_change_names_icons_and_drops_invisible_characters():
    assert pdf.text_change("\ufeffWhite CLot", "\ufeffWhite\u200b Clot") == "'White CLot' -> 'White Clot'"
    old, new = "﻿Deal 1 DMG to a target creature on the opposing field of play.", \
        "﻿Deal 1  DMG to a target creature on the opposing field of play."
    assert pdf.text_change(old, new, {"": "fire"}) == "'1 DMG' -> '1 [fire] DMG'"
    assert pdf.text_change(old, new) == "'1 DMG' -> '1 [icon] DMG'"


def test_line_cuts_whole_phrases():
    phrases = ["Name text: 'a' -> 'b'", "added Ability FX: Frozen", "Frozen overlay hidden"]
    assert pdf.line(phrases) == "Name text: 'a' -> 'b'; added Ability FX: Frozen; Frozen overlay hidden"
    assert pdf.line(phrases, 60) == "Name text: 'a' -> 'b'; added Ability FX: Frozen; and 1 more"
    assert pdf.line([]) == "" and len(pdf.line(["x" * 80], 50)) == 50


# ========================
# Real pushes (the card clone's history)
# ========================
def _pair(clone, before: str, after: str, rel: str) -> tuple:
    return tuple(PSDImage.open(BytesIO(at_rev(clone, rev, rel))) for rev in (before, after))


def test_a_name_edit(clone):
    """The bot's own bake of 2026-10-22 (Card edit: Clot White)."""
    assert pdf.changes(*_pair(clone, "7883697e", "d8163103", "Auxiliary/Minions/Clot_White.psd")) == [
        "Name text: 'White CLot' -> 'White Clot'"]


def test_overlay_hidden_under_its_fx_layer(clone):
    """3b06f803: Gasoline's image icons became tokens and its "Blazing" overlay went under an Ability FX layer."""
    changes = pdf.changes(*_pair(clone, "3b06f803^", "3b06f803", "Items/Support/2 Stars/Gasoline.psd"))
    assert changes == ["Ability text: 'a type,' -> 'a [fire] type,', 'a type,' -> 'a [machine] type,' (+3 more)",
                       "added Ability FX: Blazing", "fire x3, machine x2 and Blazing overlay hidden"]


@pytest.mark.parametrize("rel, expected", [
    (TEMPLE, "Ability text: '(-1/1CD)' -> '(-1/1GCD)(OYT)'"),      # 25 records rewritten: shmd, SoLd/PlLd, a gradient fill
    (GLAIVE, "Ability text: '(E2T)' -> '(2CD)'"),                  # a mask re-encoded, a smart object re-embedded
    (HEAVY_MOLE, "Ability text: '(1PT)' -> '(1CD)'"),              # a newer Photoshop adds gs99 to Gradient Overlays
    (LAZARUS, "Laz/Ability text: '(1PGT)(3' -> '(1GCD)(3', 'Flip to to' -> 'Flip to'"),   # b"Nrml" written b"normal"
])
def test_photoshop_resave_noise_is_left_out(clone, rel, expected):
    old, new = _pair(clone, *COOLDOWNS, rel)
    rewritten = sum(1 for a, b in zip(old._record.layer_and_mask_information.layer_info.layer_records,
                                      new._record.layer_and_mask_information.layer_info.layer_records)
                    if a.tagged_blocks.tobytes() != b.tagged_blocks.tobytes())
    assert rewritten >= 10                      # the save touched far more than the edit
    assert pdf.changes(old, new) == [expected]


def test_a_mask_switched_off(clone):
    data = at_rev(clone, COOLDOWNS[1], GLAIVE)
    old, new = PSDImage.open(BytesIO(data)), PSDImage.open(BytesIO(data))
    beam = next(layer for layer in new.descendants() if layer.name == "Laser Beam")
    key = f"id:{beam.layer_id}"
    assert pdf.facts_of(old)[key].mask == pdf.facts_of(new)[key].mask          # the digest is stable
    beam._record.mask_data.flags.mask_disabled = True
    saved = BytesIO()
    new.save(saved)
    assert pdf.changes(old, PSDImage.open(BytesIO(saved.getvalue()))) == ["Laser Beam mask changed"]


def test_a_style_change_names_its_words(clone):
    """The badge batch of 2026-10-22 set the tag in stoogedings and left the text as it was."""
    assert pdf.changes(*_pair(clone, "af826714", "587d38b4", "Auxiliary/Items/Dreamshield.psd")) == [
        "Ability font Perpetua -> stoogedings-Regular on '(ENDT)'"]


def test_a_box_resized(clone):
    changes = pdf.changes(*_pair(clone, "f58418a5", "09131c67", "Creatures/Mario/2 Stars/Koops.psd"))
    assert changes == ["Ability text: '(FTE2T)(3' -> '(1/2CD)(3', '(E2T)(4' -> '(2CD)(4'", "Ability box shorter by 624 px"]
