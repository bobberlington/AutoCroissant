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


def test_each_layer_change_is_told(tmp_path):
    def edit(psd):
        square = psd.find("Square")
        square.visible, square.opacity, square.blend_mode = False, 128, BlendMode.MULTIPLY
        list(psd.find("Stars"))[0].name = "Gold {star}"          # braces in a name are text
        psd.append(PixelLayer.frompil(Image.new("RGB", (5, 5)), psd, "Ability FX: Frozen"))
    assert pdf.changes(_card(tmp_path, "a"), _card(tmp_path, "b", edit)) == [
        "added Ability FX: Frozen", "Square hidden", "star renamed 'Gold {star}'", "Square opacity 100% -> 50%",
        "Square blend Normal -> Multiply"]


# ========================
# Text
# ========================
def test_text_change_quotes_the_edited_words_of_a_long_text():
    old = "Gain 2 ATK when an ally dies. Deal 6 damage to every enemy creature on the field.(1PT)"
    new = "Gain 2 ATK when an ally dies. Deal 7 damage to every enemy creature on the field.(1CD)"
    assert pdf.text_change(old, new) == "'6' -> '7', 'field.(1PT)' -> 'field.(1CD)'"
    assert pdf.text_change(old, old.replace("an ally", "an  ally")) == ""          # spacing alone
    inserted = pdf.text_change(old, old.replace("every enemy", "every [fire] enemy"))
    assert inserted == "'every enemy' -> 'every [fire] enemy'"                  # an insertion keeps a word each side
    assert pdf.text_change(old, "Nothing like it at all, a new ability for a new card entirely.").startswith("rewritten (")


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


@pytest.mark.parametrize("rel, expected", [(TEMPLE, "Ability text: '(-1/1CD)' -> '(-1/1GCD)(OYT)'")])
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


def test_smart_filters_renumbered_by_a_save_say_nothing(clone):
    """Bone's Loans: the save renumbered Layer 2's Puppet Warp (filterID 1002 -> 632), its settings the same. Pep Drink's
    Perspective Warp really moved (49 px), beside a new filterID and validAtPosition."""
    assert pdf.changes(*_pair(clone, "21c1c98f", "7cb4c867", "Items/Equip/1 Stars/Bone's Loans.psd")) == [
        "Ability text: '(1PGT)' -> '(1GCD)'"]
    assert "tumblr_nguxntKncE1u258ivo1_500 smart filters changed" in pdf.changes(
        *_pair(clone, "91fcda38", "56a79f4f", "Items/Heal/2 Stars/Pep_Drink.psd"))
