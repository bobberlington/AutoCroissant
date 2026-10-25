"""
psd_edit and psd_regions: the edit pipeline's stat toggle, text edits, delta merge and gates (EDIT_PIPELINE_DESIGN
sections 5, 6, 10; steps S2 and S3). The fixture is the real card repo (MDW/Templates/Creature_Template.psd is 30 MB,
too big to ship): tests that need a PSD read it from the clone and skip without it; tests that render or edit text
skip without the fork's text engine (psd_tools.composite.text), so the suite stays green under stock psd-tools.
conftest.find_fork picks the fork ($AC_PSD_TOOLS_SRC, else an installed fork, else ~/Documents/Tools/psd-tools/src or
/tmp/wt-cards/src when they have its API); the fork venv has psd-tools[text]:
    ~/.venvs/ac-py314-fork/bin/python -m pytest -q -n auto tests/test_psd_edit.py
"""
import json
from collections import Counter
from copy import deepcopy
from dataclasses import replace
from importlib.util import find_spec
from pathlib import Path
from re import search
from subprocess import CalledProcessError, run
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image
from psd_tools import PSDImage
from psd_tools.constants import Tag
from psd_tools.psd import engine_data

import commands.psd_edit as pe
from commands.card_models import card_name
from commands.psd_regions import PAINT_ORDER, badge_layers, clip, fx_pad, regions, stat_rows
from conftest import (BADGED, PLAIN_DOOMER, at_rev, no_lazy, salt_font, colour_glyphs, fork_reason, painted,
                      saves_preview, story_sync as _story_sync, text_engine as _text_engine)
from helpers import MINI_DOOMER, _card, _typo_fix, stale_copy

# ========================
# Configuration
# ========================
VOCAB = Path(__file__).resolve().parent / "goldens" / "_vocab.json"
HIVE = "Field/4 Stars/The_Hive.psd"                                # stoogedings badges, no overlay over the ability
MORPHO = "Auxiliary/Creatures/Morpho_Knight.psd"                  # badge runs, chart words and unstyled words on FX layers
HP_ROW = (360, 1200, 1173, 1284)                                   # its hp row: the digits' mask union
METTATON = "Auxiliary/Creatures/Mettaton_EX.psd"                  # a GradientOverlay on one digit per row (Atk: on hidden 6)
GREEN_MAGE = "Creatures/Everhood/4 Stars/Green_Mage.psd"          # fixed 2026-10-23: atk 6, its bars on the canvas
OFF_CANVAS_MAGE = "45cf8746^"   # Green Mage before its bars moved onto the card (atk 10: no digit visible, hidden 7 styled)
SHELL_SHOCK = "Auxiliary/Status Effects/Debuffs/3 Stars/Shell_Shock.psd"
MISMATCHED_SHOCK = "8af07bcd^"  # Shell_Shock before its re-save: its PNG another session's than its preview
PURE_VESSEL = "Creatures/Hollow Knight/5 Stars/Pure_Vessel.psd"   # digit masks start on the canvas and reach past it
AZAMOTH_KING = "Creatures/ROTMG/5 Stars/Azamoth_the_Forgotten_King.psd"   # hp 10 with 6, 8 and 9 styled: ambiguous
ANUBISATH = "Auxiliary/Minions/Anubisath_Guardian.psd"            # left-aligned title 11 px short of the first star
NAMED = ("Creatures/World Of Warcraft/5 Stars/The_Lich_King.psd",)   # Text/Name + one visible Text/Ability, fonts in MDW/Fonts
CANNON = "Creatures/FTL/5 Stars/Royal_Eradicator_Main_Cannon.psd"  # the one 10+ card: atk 12 as Top (its 10 marker) + Bottom 2
GASOLINE = "Items/Support/2 Stars/Gasoline.psd"                    # an overlay "Blazing" type layer over the ability (OuterGlow)
RED_BLOON = "Auxiliary/Minions/PoD01Red_Bloon.psd"                 # PNG exported raw (unconverted) from a non-sRGB profile
BLOCK_HOPPER = "Creatures/Mario/2 Stars/Block_Hopper.psd"          # the same, its rows 18% bad when converted
BONE = "Creatures/Kirby/4 Stars/Wham_Bam_Bone.psd"                 # its title layer is named after the card
VIRUS = "Creatures/Kirby/3 Stars/Computer_Virus.psd"               # its Name layer sits at the root
MARX = "Creatures/Kirby/5 Stars/Marx.psd"                          # its Ability sits at the root, 'Ability FX: Glued' over it
EDEN = "Creatures/Isaac/2 Stars/Tainted_Eden.psd"                  # 'Name copy' and 'Name copy 2' print the title again
SOMETHING = "Creatures/Omori/3 Stars/SOMETHING.psd"                # two offset copies print the ability again
AFRAID = "Auxiliary/Status Effects/Debuffs/2 Stars/Afraid.psd"     # no Text group: its title is the top-level layer 'Afraid'
JAWSUM = "Creatures/Omori/3 Stars/Mr_Jawsum.psd"                   # no Text group: its title 'Group 3/Group 2/Mr. Jawsum'
SOUL_MASTER = "Creatures/Hollow Knight/4 Stars/Soul_Master.psd"    # its title 'Soul Master' sits beside the Text group
PAPYRUS = "Creatures/Undertale/3 Stars/Papyrus.psd"                # its title layer 'Text/The Great Papyrus' is named after its text
SANS = "Creatures/Undertale/5 Stars/Sans.psd"                      # likewise 'Text/snas.'
LORIAN = "Auxiliary/Minions/Lorian.psd"                            # smart objects with smart filters
PAINT_WARRIOR = "Auxiliary/Minions/Paint_Warrior.psd"              # a blank Ability whose box runs over the stat bars
KNIGHT = "Creatures/Everhood/2 Stars/Knight_Lost_A_Lot.psd"        # 'Knight', 'Lost', 'A', 'Lot' stacked: two in the title region
GLOBAL_TURN = "Auxiliary/Rulebook/08_Global_Turn.psd"              # a rule page: its heading 'Text/THE GLOBAL TURN'
FROSTMOURNE = "Auxiliary/Items/Frostmourne.psd"                    # four lines of plain ability text: an edit to one keeps three
PIZZA = "Creatures/Other/3 Stars/Pizza.psd"                        # three lines in three colours, each after a badge
COLOURED = "7fd2aa30"       # Pizza after the badges re-run: its first badge 18.66 pt, the others 32 (one badge size)
OVERLAID = {GASOLINE: "3b06f803bb0abf57859297e271177dd82855402b^",    # the last versions with the overlay word: these
            CANNON: "587d38b401102b13d06733a5f62b01ebeb674d3f^"}      # commits hid it under the JSX's FX layer (2026-10-22)
STYLED = ["Become Back Money", "Blazing", "Condemned", "Contracted", "Corrupted Blood", "Curse", "Cursed Inferno", "Frozen",
          "Frozen In Time", "Glued", "Hemorrhaging", "Hexed", "Infected", "Spirit Flame", "Withering"]   # the chart's styled statuses
TEXT_RECORD = ["blocks", "channels", "rect"]                       # what a text edit may change on its layer record
STAT_RECORD = ["blocks", "flags"]                                  # a toggled digit whose effect block moved
GLOW = 18                                                          # the chart's OuterGlow reach (fx_pad) on 13 of the 15


def _fills(layer) -> list:
    return [(layer.text[s:e], tuple(round(v, 3) for v in f)) for s, e, f in pe.run_fills(layer)]


def _ink(layer, psd) -> np.ndarray:
    """A layer's ink (alpha > 0.5) placed on the canvas."""
    out = np.zeros((psd.height, psd.width), bool)
    x0, y0, x1, y1 = layer.bbox
    out[y0:y1, x0:x1] = layer.numpy("shape")[..., 0] > 0.5
    return out


def _overlaid(clone, rel: str, tmp_path) -> str:
    """OVERLAID[rel]'s PSD from the clone's history, in tmp_path/overlaid/."""
    try:
        data = run(["git", "-C", str(clone), "show", f"{OVERLAID[rel]}:{rel}"], capture_output=True, check=True,
                   env=no_lazy()).stdout
    except (CalledProcessError, OSError):
        pytest.skip(f"the clone's history lacks {OVERLAID[rel]}:{rel}")
    path = tmp_path / "overlaid" / Path(rel).name
    path.parent.mkdir()
    path.write_bytes(data)
    return str(path)


def _record(name, flags=b"\x08", channels=((-1, 10), (0, 10), (1, 10), (2, 10)), blocks=(), rect=(0, 0, 1, 1)) -> dict:
    """A layer_records() fingerprint."""
    return {"name": name, "flags": flags, "rect": rect, "channels": tuple(channels), "mask": b"", "blend": (0, 255, 0),
            "blocks": tuple(blocks)}


text_engine = pytest.mark.skipif(not _text_engine(), reason=fork_reason())
# a text edit: the fork's set_text also writes the layer's Txt2 story (Photoshop reads that one)
text_edit = pytest.mark.skipif(not (_text_engine() and _story_sync()),
                               reason=fork_reason() if not _text_engine() else "the fork's text edits leave the Txt2 story stale")
# the copy check composites the card: gradient fills need scipy (psd-tools[composite])
composites = pytest.mark.skipif(find_spec("scipy") is None, reason="gradient fills need psd-tools[composite] (scipy)")


# ========================
# Fixtures
# ========================
@pytest.fixture(scope="module")
def parser():
    vocab = json.loads(VOCAB.read_text(encoding="utf-8"))
    return pe.offline_parser(vocab["all_types"], vocab["stars"], vocab["manifest"])


@pytest.fixture(scope="module")
def chart(clone):
    path = clone / pe.STATUS_CHART
    if not path.exists():
        pytest.skip("no status chart")
    return pe.status_fills(PSDImage.open(str(path)))


def _plain(clone, tmp_path) -> tuple:
    """Mini Doomer without its badge layer (PLAIN_DOOMER, the version ability edits were proven on): PSD and PNG paths."""
    out = tmp_path / "plain"
    out.mkdir(exist_ok=True)
    paths = []
    for ext in (".psd", ".png"):
        rel = MINI_DOOMER[:len(MINI_DOOMER) - 4] + ext
        (out / Path(rel).name).write_bytes(at_rev(clone, PLAIN_DOOMER, rel))
        paths.append(str(out / Path(rel).name))
    return tuple(paths)


def flagged(notes: list, pattern: str) -> str:
    """The note matching pattern (re.search): a content problem baked and told (owner 2026-10-24); fails naming them."""
    hit = [note for note in notes if search(pattern, note)]
    assert hit, notes
    return hit[0]


def _bake(clone, parser, rel, jobs, tmp_path, chart=None, update_preview=False, paths=None):
    psd, png = paths or _card(clone, rel)
    parsed = pe.parse_card(parser, psd, rel)
    assert pe.classify(parsed, jobs, PSDImage.open(psd)).jobs == jobs
    return parsed, pe.apply_jobs(psd, png, rel, jobs, parsed, parser, str(tmp_path), str(clone / pe.FONTS_DIR), chart,
                                 update_preview=update_preview)


# ========================
# Pure functions (any psd-tools)
# ========================
def test_parity_bar():
    """The owner's bar (2026-10-22): mean |diff| <= 2.0 (was 1.0), at most 0.5% of pixels past 24. Missing it is told, not
    refused (2026-10-23); only past the floor, more than 25% of the region's pixels past 24, is a render broken."""
    assert (pe.PARITY_MEAN_MAX, pe.PARITY_BAD_MAX, pe.PARITY_THRESH, pe.PARITY_FLOOR) == (2.0, 0.5, 24, 25.0)
    assert pe._within_bar({"bad_pct": 0.5, "mean": 2.0, "max": 200})
    assert not pe._within_bar({"bad_pct": 0.0, "mean": 2.01, "max": 7})
    assert not pe._within_bar({"bad_pct": 0.51, "mean": 0.5, "max": 7})
    assert not pe._past_floor({"bad_pct": 25.0, "mean": 90.0, "max": 255})
    assert pe._past_floor({"bad_pct": 25.01, "mean": 1.0, "max": 30})


def test_own_pixels():
    """The blank-render check's pixels (review 2026-10-23): the merge's replaced pixels no other edit's region reaches;
    all of them when the others cover the region."""
    replaced = np.zeros((10, 20), bool)
    replaced[0:2, 0:2] = True                                               # another edit's change, in the overlap
    region = (100, 50, 120, 60)
    assert not pe.own_pixels(replaced, region, [(90, 40, 105, 55)]).any()
    assert pe.own_pixels(replaced, region, [(110, 55, 130, 70), (0, 0, 10, 10)]).sum() == 4    # no overlap there
    replaced[9, 19] = True
    assert pe.own_pixels(replaced, region, [(90, 40, 105, 55)]).sum() == 1
    assert pe.own_pixels(replaced, region, [(0, 0, 200, 200)]).sum() == 5   # covered: every replaced pixel counts


def test_codes_are_the_closed_list():
    assert pe.CODES == ("duplicate", "unsupported-card", "stat>10", "recent", "blob", "parity", "styled-word", "font",
                        "overflow", "bytes", "size", "region", "shape", "reparse", "idempotence", "edit-changed",
                        "head-moved", "capped", "error")
    assert PAINT_ORDER[-1] == "ability_fx" and clip((-5, -5, 2000, 10), 1200, 2400) == (0, 0, 1200, 10)


def test_normalize_ability_matches_the_parser():
    layer = "Has stats -2 .\x03(BOT) Die .  \r"
    site = "Has stats -2.\n(BOT) Die."
    assert pe.normalize_ability(layer) == pe.normalize_ability(site) == "Has stats -2.\n(BOT) Die."
    assert pe.normalize_ability("﻿\"Gain\t1 \nhp .\"\n") == "Gain 1 \nhp." and pe.normalize_ability("") == ""
    assert pe.collapse(" Shooting\rStar ") == "Shooting Star" and pe.collapse(None) == ""


def test_keep_ends_puts_back_blank_first_and_last_paragraphs():
    """The website and Discord trim an ability's ends: the layer's blank first and last paragraphs come back (PoD10BFB)."""
    old = "\r(1GCD)(2U) Intercept.\r(2CD) Summon."
    assert pe.keep_ends(old, "(1GCD)(2U) Intercept.\r(2CD) Summon!") == "\r(1GCD)(2U) Intercept.\r(2CD) Summon!"
    assert pe.keep_ends(" \r\x03 Hit.  \r\r", "Hit hard.") == " \r\x03Hit hard.\r\r"     # the paragraphs, not the indent
    assert pe.keep_ends("Hit.  ", "Hit hard.") == "Hit hard." and pe.keep_ends("Hit.", "Hit hard.") == "Hit hard."
    assert pe.keep_ends(old, "\rAlready.") == "\rAlready." and pe.keep_ends("\r", "New.") == "New."
    assert pe.keep_ends(old, "") == ""


def test_new_fx_refuses_a_status_word_the_jsx_cannot_name():
    """BUG 2 review: status_spans reads "Blazing" in "2Blazing" (a digit touches it), Effect Layers.jsx's phrase "Blazing"
    only whole words when there are any: beside a whole-word one, no name reaches it, so new_fx refuses rather than
    leave it coloured without its layer; alone, "Blazing" names it."""
    blazing = pe.StatusColor("Blazing", (1.0, 0.5, 0.0), True)
    chart = {"blazing": blazing}
    text = "Inflict 2Blazing, then Blazing."
    with pytest.raises(pe.Refused, match="'2Blazing,': Effect Layers.jsx cannot name") as refused:
        pe.new_fx(text, pe.status_spans(text, chart, []), [])
    assert refused.value.code == "unsupported-card"
    assert pe.new_fx(text, pe.status_spans(text, chart, []), [(8, 16)]) == [("Blazing", [(23, 30)], blazing)]
    assert pe.new_fx("Inflict 2Blazing.", pe.status_spans("Inflict 2Blazing.", chart, []), []) == [("Blazing", [(9, 16)], blazing)]


def test_gate_bytes_pairs_fx_records(monkeypatch):
    """An ability edit's records: the old FX layer gone, the new one come with the chart's bytes, the overlay's one flag byte,
    the text record's channels; anything undeclared refuses."""
    glow = (b"lfx2", b"chart-block")
    files = {"old": [_record("Text"), _record("Ability", channels=((-1, 5), (0, 5), (1, 5), (2, 5))),
                     _record("Ability FX: Frozen", blocks=[glow]), _record("Blazing")],
             "new": [_record("Text"), _record("Ability"), _record("Ability FX: Blazing", blocks=[glow]), _record("Blazing", flags=b"\x0a")]}
    monkeypatch.setattr(pe, "layer_records", lambda path: [dict(r) for r in files[path]])
    edit = pe.Edit("ability", "a", "b", [], [1], None, hidden=[3], fx=[("Ability FX: Blazing", {b"lfx2": b"chart-block"})],
                   dropped=["Ability FX: Frozen"])
    records, flag_bytes, block_bytes = pe.gate_bytes("old", "new", [edit])
    assert records == [(1, "Ability", ["channels"]), (2, "Ability FX: Frozen", ["removed"]), (2, "Ability FX: Blazing", ["added"]),
                       (3, "Blazing", ["flags"])]
    assert (flag_bytes, block_bytes) == (1, len(b"chart-block"))
    files["new"][2] = _record("Ability FX: Blazing", blocks=[(b"lfx2", b"other")])
    with pytest.raises(pe.Refused, match="not the declared pixel layer"):
        pe.gate_bytes("old", "new", [edit])
    files["new"][2] = _record("Ability FX: Blazing", blocks=[glow], channels=((0, 10), (1, 10), (2, 10)))   # no transparency
    with pytest.raises(pe.Refused, match="not the declared pixel layer"):
        pe.gate_bytes("old", "new", [edit])
    files["new"][2] = _record("Ability FX: Blazing", blocks=[glow])
    files["new"].append(_record("Stray"))
    with pytest.raises(pe.Refused, match="layer count"):
        pe.gate_bytes("old", "new", [edit])
    files["new"].pop()
    with pytest.raises(pe.Refused, match="changed blocks, flags, name"):      # an undeclared drop pairs the old FX with Blazing
        pe.gate_bytes("old", "new", [pe.Edit("ability", "a", "b", [], [1], None, hidden=[3], fx=edit.fx)])
    with pytest.raises(pe.Refused, match="declared removed or added is not"):
        pe.gate_bytes("old", "new", [pe.Edit("ability", "a", "b", [], [1], None, hidden=[3], fx=edit.fx,
                                             dropped=["Ability FX: Frozen", "Ability FX: Glued"])])
    files["new"][3] = _record("Blazing", flags=b"\x08")                        # the overlay did not hide
    with pytest.raises(pe.Refused, match="did not change"):
        pe.gate_bytes("old", "new", [edit])


def test_gate_bytes_keeps_a_rebuilt_fx_records_style(monkeypatch):
    """Owner q6: an FX layer cut again keeps its own style. Its record may change rect and channels (and its unicode name
    when renamed); an effect block, its blending or opacity changed refuses, whatever the chart says. This pins the
    rebuilt record's (index, old name, name) and what may change (the pre-q6 gate refused these too); the q6 guards are
    test_noop_ability_bake_keeps_an_l7_fx_layers_own_style and test_rebuilt_fx_layer_keeps_its_own_style."""
    frozen = "Ability FX: Frozen"
    own = [(b"lfx2", b"own-glow"), (b"luni", b"Frozen")]
    files = {"old": [_record("Ability"), _record(frozen, blocks=own)],
             "new": [_record("Ability"), _record(frozen, blocks=own, rect=(0, 5, 1, 6), channels=((-1, 9), (0, 9), (1, 9), (2, 9)))]}
    monkeypatch.setattr(pe, "layer_records", lambda path: [dict(r) for r in files[path]])
    edit = pe.Edit("ability", "a", "b", [], [0], None, rebuilt=[(1, frozen, frozen)])
    assert pe.gate_bytes("old", "new", [edit]) == ([(1, frozen, ["channels", "rect"])], 0, 0)
    files["new"][1] = _record(frozen, blocks=[(b"lfx2", b"chart-glow"), (b"luni", b"Frozen")])
    with pytest.raises(pe.Refused, match="is not the declared rebuild: blocks lfx2"):
        pe.gate_bytes("old", "new", [edit])
    files["new"][1] = {**_record(frozen, blocks=own), "blend": (0, 200, 0)}
    with pytest.raises(pe.Refused, match="changed blend"):
        pe.gate_bytes("old", "new", [edit])


def test_classify_table():
    parsed = {"card": {"type": "minion", "hp": 5, "def": 2, "atk": 5, "spd": 5, "ability": "Die.\n(BOT) Live."}}
    out = pe.classify(parsed, {"hp": 8, "def": 2, "atk": 11, "spd": None, "stars": 3, "types": ["fire"],
                               "ability": "Die.\n(BOT) Live. ", "name": "X", "note": "n", "series": "mario",
                               "creatures": [{"hp": 1}], "hidden": True})
    assert out.jobs == {"hp": 8}
    assert out.redundant == ["def", "ability"]
    assert [(r.code, r.why.split(":")[0]) for r in out.refusals] == [("stat>10", "atk 11"), ("unsupported-card", "stars"),
                                                                     ("unsupported-card", "types"), ("unsupported-card", "no PSD to find the title layer in")]
    assert out.other == ["spd", "note", "series", "creatures", "hidden"]
    assert out.refusals[0].why == "atk 11: one bar row; a second row is standardization work (ledger L3)"
    two_rows = {"card": {"type": "creature", "atk": 12}}                      # a parse above 10 means Top + Bottom rows (idea 13)
    assert pe.classify(two_rows, {"atk": 15}).jobs == {"atk": 15} and pe.classify(two_rows, {"atk": 12}).redundant == ["atk"]
    assert [(r.code, r.why) for r in pe.classify(two_rows, {"atk": 21}).refusals] == [("stat>10", "atk 21: above 20")]
    assert pe.classify(parsed, {"ability": "Die.\n(BOT) Die."}).jobs == {"ability": "Die.\n(BOT) Die."}
    assert pe.shown_name(parsed) == "" and pe.shown_name({"card": {"display_name": "Y"}}) == "Y"
    with pytest.raises(ValueError):                                 # not a stat: the card's "error", not stat>10
        pe.classify(parsed, {"hp": -1})


def test_merge_png_is_a_pixel_delta():
    png = Image.new("RGBA", (10, 10), (10, 20, 30, 255))
    base = Image.new("RGBA", (4, 4), (10, 20, 30, 255))
    render = base.copy()
    render.putpixel((1, 1), (200, 0, 0, 255))
    render.putpixel((3, 2), (0, 200, 0, 255))
    base.putpixel((0, 0), (11, 20, 30, 255))                  # baseline noise: the render agrees, so the PNG keeps its pixel
    render.putpixel((0, 0), (11, 20, 30, 255))
    merged, changed = pe.merge_png(png, base, render, (2, 2, 6, 6))
    assert changed.shape == (4, 4) and changed.sum() == 2 and merged.mode == "RGBA" and merged.size == (10, 10)
    assert merged.getpixel((3, 3)) == (200, 0, 0, 255) and merged.getpixel((5, 4)) == (0, 200, 0, 255)
    assert merged.getpixel((2, 2)) == (10, 20, 30, 255)
    assert pe.gate_region(png, merged, (2, 2, 6, 6)) == 2
    with pytest.raises(pe.Refused) as refused:
        pe.gate_region(png, merged, (0, 0, 3, 3))
    assert refused.value.code == "region"
    with pytest.raises(pe.Refused) as refused:
        pe.merge_png(png.convert("P"), base, render, (2, 2, 6, 6))
    assert refused.value.code == "shape"


def test_save_png_keeps_the_official_colour_chunks(tmp_path):
    """The merged PNG carries Photoshop's iCCP, gAMA, cHRM and pHYs chunks: only pixels change."""
    from PIL.PngImagePlugin import PngInfo
    chunks = PngInfo()
    chunks.add(b"gAMA", (45455).to_bytes(4, "big"))
    chunks.add(b"cHRM", b"".join(v.to_bytes(4, "big") for v in (31270, 32900, 64000, 33000, 30000, 60000, 15000, 6000)))
    official = tmp_path / "official.png"
    Image.new("RGBA", (4, 4), (1, 2, 3, 255)).save(official, pnginfo=chunks, icc_profile=b"not really a profile", dpi=(72.009, 72.009))
    like = Image.open(official)
    pe.save_png(Image.new("RGBA", (4, 4), (9, 9, 9, 255)), str(tmp_path / "merged.png"), like)
    out = Image.open(tmp_path / "merged.png")
    assert out.info["gamma"] == like.info["gamma"] == 0.45455 and out.info["chromaticity"] == like.info["chromaticity"]
    assert out.info["icc_profile"] == b"not really a profile" and out.info["dpi"] == like.info["dpi"]
    assert out.getpixel((0, 0)) == (9, 9, 9, 255)
    pe.save_png(Image.new("RGBA", (4, 4)), str(tmp_path / "plain.png"), Image.new("RGBA", (4, 4)))   # no chunks: none added
    assert Image.open(tmp_path / "plain.png").info == {}


# ========================
# Layer shapes (the clone, any psd-tools)
# ========================
def test_regions_and_rows_on_mini_doomer(clone, parser):
    path = _card(clone, MINI_DOOMER)[0]
    psd = PSDImage.open(path)
    found = regions(psd, ["wind", "light", "defensive_minion"])
    stars, icons, inline = badge_layers(psd, ["wind", "light", "defensive_minion"])
    assert [tuple(s.bbox) for s in stars] == found["stars"] and len(icons) == 3 and inline == []
    parsed = pe.parse_card(parser, path, MINI_DOOMER)
    with pytest.raises(pe.Refused) as refused:
        pe.set_stat(psd, parsed, "hp", 11)
    assert refused.value.code == "stat>10"
    with pytest.raises(ValueError):
        pe.set_stat(psd, parsed, "hp", -1)
    assert found["stat_hp"] == [(359, 1199, 1174, 1285)] and found["stat_spd"] == [(359, 1577, 1174, 1663)]
    assert found["name"] == [(300, 23, 899, 114)] and found["ability"] == [(27, 1802, 1157, 2343)]   # 2343 since 19db3675's "Destroy"
    assert found["frame"][0] == (0, 1708, 1200, 1748) and found["ability_bg"] == [(0, 1748, 1200, 2400)]
    assert len(found["types"]) == 3 and len(found["stars"]) == 2
    assert {k: [d for _, d, _ in rows] for k, rows in stat_rows(psd).items()} == {"hp": [[5]], "def": [[2]], "atk": [[5]], "spd": [[5]]}
    group, digits, visible = pe.template_row(psd, "hp")
    assert group.name == "Hp Dark" and sorted(digits) == sorted(pe.DIGITS) and visible == [5]


def test_non_template_rows_and_titles_refuse(clone):
    cannon = PSDImage.open(_card(clone, CANNON)[0])
    with pytest.raises(pe.Refused, match="two bar rows") as refused:           # the single-row reader; template_rows takes both
        pe.template_row(cannon, "atk")
    assert refused.value.code == "unsupported-card" and len(pe.template_rows(cannon, "atk")) == 2
    with pytest.raises(pe.Refused, match="spd: 3 bar rows"):                   # Sonic's 20 is a renamed 9 beside two hidden rows
        pe.template_rows(PSDImage.open(_card(clone, "Creatures/Other/4 Stars/Sonic.psd")[0]), "spd")
    twins = PSDImage.open(_card(clone, "Creatures/World Of Warcraft/4 Stars/Twin_Emperors.psd")[0])
    for read in (lambda psd: pe.name_layer(psd, own_text=False), pe.ability_layer):   # no Name, no card name given
        with pytest.raises(pe.Refused) as refused:
            read(twins)
        assert refused.value.code == "unsupported-card"
    assert pe.read_name(twins) == "Twin Emperors"                               # its title is named after its text


def test_title_layer_is_the_one_the_parser_reads(clone, parser):
    """name_layer takes the one layer PSDParser.title_layers reads, wherever the card keeps it (NEXT_IDEAS 5b A): a title
    named after the card under Text, a Name layer at the root. Two Name layers (an epithet) or a Name beside layers
    printing the rest of the title refuse: a bake would write the whole title into one piece."""
    bone = PSDImage.open(_card(clone, BONE)[0])
    with pytest.raises(pe.Refused, match="title layer named 'Wham Bam Bone', not Name or the card's name"):
        pe.name_layer(bone, own_text=False)                                     # without the card's name or its text
    assert pe.name_layer(bone).name == "Wham Bam Bone"                           # named after its text too
    assert pe.name_layer(bone, card_name(BONE)).name == "Wham Bam Bone" and pe.read_name(bone, card_name(BONE)) == "Wham Bam Bone"
    parsed = pe.parse_card(parser, _card(clone, BONE)[0], BONE)
    assert pe.classify(parsed, {"name": "Wham Bam Bones"}, bone).jobs == {"name": "Wham Bam Bones"}
    virus = PSDImage.open(_card(clone, VIRUS)[0])
    assert pe.name_layer(virus).parent is virus and pe.read_name(virus) == "Computer Virus appears!"
    for rel, why in ((AZAMOTH_KING, "2 title layers"), (CANNON, "the title is split over several layers")):
        psd = PSDImage.open(_card(clone, rel)[0])
        with pytest.raises(pe.Refused, match=why) as refused:
            pe.name_layer(psd, card_name(rel))
        assert refused.value.code == "unsupported-card" and pe.read_name(psd, card_name(rel)) is None
        parsed = pe.parse_card(parser, _card(clone, rel)[0], rel)
        assert [r.why for r in pe.classify(parsed, {"name": "Azamoth"}, psd).refusals] == [why]


def test_a_title_named_after_the_card_outside_text(clone, parser):
    """With no title under Text (55 status effects and 3 creatures without a Text group, 3 Hollow Knight titles beside
    it), the title is the visible text layer named after the card in the title region, at the root or in any group;
    letters and digits decide ('Mr. Jawsum' on Mr Jawsum). The parser reads the same layer."""
    for rel, path, title in ((AFRAID, "Afraid", "Afraid"), (JAWSUM, "Group 3/Group 2/Mr. Jawsum", "Mr. Jawsum"),
                             (SOUL_MASTER, "Soul Master", "Soul Master")):
        psd_path = _card(clone, rel)[0]
        psd = PSDImage.open(psd_path)
        layer = pe.name_layer(psd, card_name(rel))
        assert _layer_path(layer) == path and pe.read_name(psd, card_name(rel)) == title
        parsed = pe.parse_card(parser, psd_path, rel)
        assert parsed["card"].get("display_name") == (None if title == card_name(rel) else title)
        assert pe.classify(parsed, {"name": title}, psd).redundant == ["name"]
        assert pe.classify(parsed, {"name": title + "s"}, psd).jobs == {"name": title + "s"}
    with pytest.raises(pe.Refused, match="no Text group"):
        pe.name_layer(PSDImage.open(_card(clone, AFRAID)[0]), own_text=False)  # without the card's name or its text


class _Fake:
    """A layer or group with what the title finder reads (bbox 100 px tall from top)."""
    def __init__(self, name, top=0, kind="type", visible=True, children=(), text=None):
        self.name, self.kind, self.visible, self.text = name, kind, visible, name if text is None else text
        self.bbox, self.parent, self.children, self.height = (0, top, 400, top + 100), None, list(children), 1000
        for child in self.children:
            child.parent = self

    def is_visible(self):
        return self.visible

    def is_group(self):
        return self.kind == "group"

    def __iter__(self):
        return iter(self.children)

    def descendants(self):
        for child in self.children:
            yield child
            if child.is_group():
                yield from child.descendants()


def _layer_path(layer) -> str:
    """"Group/Sub/Layer" by name, a real layer's or a _Fake's."""
    names = []
    while layer.parent is not None:
        names.append(layer.name)
        layer = layer.parent
    return "/".join(reversed(names))


def test_the_title_outside_text_is_the_single_visible_candidate():
    """Outside Text, only visible text layers in the title region named after the card are candidates; two refuse (a
    bake would change one), and a Name or a layer under Text named after the card still comes first."""
    def found(*layers):
        psd = _Fake("", kind="group", children=layers)
        texts = [layer for layer in psd.descendants() if layer.kind == "type"]
        return [_layer_path(layer) for layer in pe.PSDParser.title_layers(psd, texts, "Mr Jawsum")]
    assert found(_Fake("Mr. Jawsum", 20)) == ["Mr. Jawsum"]
    assert found(_Fake("Group", kind="group", children=[_Fake("MR JAWSUM", 20)])) == ["Group/MR JAWSUM"]
    assert found(_Fake("Mr Jawsum", 500)) == []                                 # below the title region
    assert found(_Fake("Mr Jawsum", 20, visible=False), _Fake("Mr Jawsum", 20, kind="pixel")) == []
    assert found(_Fake("Mr Jaws", 20), _Fake("Jawsum", 20)) == []                # named after part of it
    assert found(_Fake("Mr Jawsum", 20), _Fake("Text", kind="group", children=[_Fake("Name", 30)])) == ["Text/Name"]
    assert found(_Fake("Mr Jawsum", 20), _Fake("Text", kind="group", children=[_Fake("Mr Jawsum", 600)])) == ["Text/Mr Jawsum"]
    two = _Fake("", kind="group", children=[_Fake("Mr Jawsum", 20), _Fake("Group", kind="group", children=[_Fake("Mr. Jawsum", 40)])])
    with pytest.raises(pe.Refused, match="2 title layers"):
        pe.name_layer(two, "Mr Jawsum")
    assert pe.read_name(two, "Mr Jawsum") is None


def test_ability_layer_wherever_the_card_keeps_it(clone):
    """ability_layer takes the single visible Ability layer in any group or at the root (NEXT_IDEAS 5b A: 92 cards keep it
    outside the Text group); its FX layers go directly above its top-level group, the layer itself at the root."""
    marx = PSDImage.open(_card(clone, MARX)[0])
    layer = pe.ability_layer(marx)
    assert layer.parent is marx and pe._top(layer) is layer
    assert [fx.name for fx in pe.fx_layers(marx, layer)] == ["Ability FX: Glued"]
    doomer = PSDImage.open(_card(clone, MINI_DOOMER)[0])
    assert pe._top(pe.ability_layer(doomer)) is pe._text_group(doomer)


def test_a_title_named_after_its_own_text(clone, parser):
    """The last rule (owner, 2026-10-23: where the title sits is the developer's): a visible layer in the title region
    named after its own text, as Photoshop names a new text layer, when nothing else is found. Papyrus prints 'The
    Great Papyrus' and Sans 'snas.' (the live bake's two refusals of b22f4cc4): parser and bake read that layer, and
    the website's edits asking for those titles are redundant. Several such layers (Knight Lost A Lot's stacked words)
    read as none; a rule page's heading is no title."""
    for rel, title in ((PAPYRUS, "The Great Papyrus"), (SANS, "snas.")):
        psd_path = _card(clone, rel)[0]
        psd = PSDImage.open(psd_path)
        layer = pe.name_layer(psd, card_name(rel))
        assert _layer_path(layer) == "Text/" + title and pe.read_name(psd, card_name(rel)) == title
        assert pe.own_text_title(layer, card_name(rel))
        parsed = pe.parse_card(parser, psd_path, rel)
        assert parsed["card"]["display_name"] == pe.shown_name(parsed) == title
        assert pe.classify(parsed, {"name": title}, psd).redundant == ["name"]
        assert pe.classify(parsed, {"name": title + "s"}, psd).jobs == {"name": title + "s"}
    psd_path = _card(clone, KNIGHT)[0]
    parsed = pe.parse_card(parser, psd_path, KNIGHT)
    assert parsed["card"].get("display_name") is None and pe.shown_name(parsed) == "Knight Lost A Lot"
    assert [r.why for r in pe.classify(parsed, {"name": "Knight Lost A Lot"}, PSDImage.open(psd_path)).refusals] == []
    assert [r.why for r in pe.classify(parsed, {"name": "Knight Lost"}, PSDImage.open(psd_path)).refusals] == [
        "2 layers in the title region named after their text"]
    psd_path = str(clone / GLOBAL_TURN)
    if not Path(psd_path).exists():
        pytest.skip(f"{GLOBAL_TURN} not in the clone")
    psd = PSDImage.open(psd_path)
    parsed = pe.parse_card(parser, psd_path, GLOBAL_TURN)
    assert parsed["card"].get("display_name") is None and pe.name_layer(psd).name == "THE GLOBAL TURN"
    assert [r.why for r in pe.classify(parsed, {"name": "The Global Turn"}, psd).refusals] == [
        "title layer named 'THE GLOBAL TURN', not Name or the card's name"]
    assert pe.read_name(psd) == "THE GLOBAL TURN" and pe.read_name(psd, own_text=False) is None
    with pytest.raises(pe.Refused, match="the title reads None"):              # the reparse gate reads it as classify
        pe.gate_reparse(parser, parsed, psd_path, GLOBAL_TURN, {"name": "THE GLOBAL TURN"}, psd)
    if pe.story_sync():                                                         # and so does the bake
        with pytest.raises(pe.Refused, match="not Name or the card's name"):
            pe.set_text_layer(psd, "name", "The Global Turn", "", own_text=False)


def test_own_text_is_the_last_rule():
    """Only when no Name and no layer named after the card is found, and only the one visible text layer in the title
    region named after its text, letters and digits alike ('SNAS' printing 'snas.'); not on a rule page."""
    def found(*layers, own_text=True):
        psd = _Fake("", kind="group", children=layers)
        texts = [layer for layer in psd.descendants() if layer.kind == "type"]
        return [_layer_path(layer) for layer in pe.PSDParser.title_layers(psd, texts, "Sans", own_text)]
    assert found(_Fake("Text", kind="group", children=[_Fake("snas.", 20)])) == ["Text/snas."]
    assert found(_Fake("SNAS", 20, text="snas.\r")) == ["SNAS"]
    assert found(_Fake("snas.", 20), own_text=False) == []
    assert found(_Fake("snas.", 500)) == [] and found(_Fake("snas.", 20, visible=False)) == []
    assert found(_Fake("Layer 1", 20, text="snas.")) == [] and found(_Fake("-", 20)) == []    # named otherwise; no letters
    assert found(_Fake("Knight", 20), _Fake("Lost", 60)) == []                  # several: none
    assert found(_Fake("snas.", 20), _Fake("Text", kind="group", children=[_Fake("Name", 30, text="Sans")])) == ["Text/Name"]
    assert found(_Fake("snas.", 20), _Fake("Sans", 40, text="the skeleton")) == ["Sans"]      # named after the card first
    stacked = _Fake("", kind="group", children=[_Fake("Text", kind="group", children=[_Fake("Knight", 20), _Fake("Lost", 60)])])
    with pytest.raises(pe.Refused, match="2 layers in the title region named after their text"):
        pe.name_layer(stacked, "Knight Lost A Lot")


def test_gate_bytes_allows_only_the_declared_rename(monkeypatch):
    """A title named after its text is renamed after the new title (Edit.renamed): that name and no other."""
    files = {"old": [_record("Text"), _record("snas.")],
             "new": [_record("Text"), _record("sans.", channels=((-1, 5), (0, 5), (1, 5), (2, 5)))]}
    monkeypatch.setattr(pe, "layer_records", lambda path: [dict(r) for r in files[path]])
    records, flag_bytes, _ = pe.gate_bytes("old", "new", [pe.Edit("name", "snas.", "sans.", [], [1], None, renamed="sans.")])
    assert records == [(1, "snas.", ["channels", "name"])] and flag_bytes == 0
    with pytest.raises(pe.Refused, match="changed channels, name"):
        pe.gate_bytes("old", "new", [pe.Edit("name", "snas.", "sans.", [], [1], None)])
    with pytest.raises(pe.Refused, match="renamed 'sans.', not 'sans'"):
        pe.gate_bytes("old", "new", [pe.Edit("name", "snas.", "sans.", [], [1], None, renamed="sans")])


@composites
def test_a_title_or_ability_printed_twice_refuses(clone):
    """Another visible text layer printing the same words refuses (Tainted Eden's glitch copies of its title, SOMETHING's
    two offset copies of its ability): a bake would change one and leave the copies (an ability bake edits them too:
    set_ability_layers). An empty layer has no copies (Anubisath's empty Ability beside an empty 'Layer 1')."""
    with pytest.raises(pe.Refused, match="the title is printed again by 'Name copy'") as refused:
        pe.name_layer(PSDImage.open(_card(clone, EDEN)[0]), card_name(EDEN))
    assert refused.value.code == "unsupported-card"
    with pytest.raises(pe.Refused, match="the ability is printed again by 'Attacks are piercing") as refused:
        pe.ability_layer(PSDImage.open(_card(clone, SOMETHING)[0]))
    assert refused.value.code == "unsupported-card"
    assert pe.ability_layer(PSDImage.open(_card(clone, ANUBISATH)[0])).text == ""


def test_name_redundancy_uses_the_printed_title(clone, parser):
    """A name edit is redundant when the bot already shows it (the parser's display_name, else the file name)."""
    path = _card(clone, MINI_DOOMER)[0]
    parsed = pe.parse_card(parser, path, MINI_DOOMER)
    psd = PSDImage.open(path)
    assert parsed["card"].get("display_name") is None and pe.shown_name(parsed) == "Mini Doomer"   # prints its file name
    assert pe.classify(parsed, {"name": "Mini  Doomer\r"}, psd).redundant == ["name"]
    assert pe.classify(parsed, {"name": "Mini Doomer II"}, psd).jobs == {"name": "Mini Doomer II"}
    path = _card(clone, VIRUS)[0]                                          # Name at the root: read and editable
    parsed = pe.parse_card(parser, path, VIRUS)
    assert parsed["card"]["display_name"] == pe.shown_name(parsed) == "Computer Virus appears!"
    assert pe.classify(parsed, {"name": "Computer Virus appears!"}, PSDImage.open(path)).redundant == ["name"]
    assert pe.classify(parsed, {"name": "Computer Virus"}, PSDImage.open(path)).jobs == {"name": "Computer Virus"}


def test_royal_eradicator_two_row_atk(clone, parser, tmp_path):
    """Idea 13 on the one 10+ card. `atk dark` holds Top ("0".."9" plus the parser's "10" marker: visible at opacity 0, so the
    parser adds a full top row; hidden, it reads Bottom alone: 2) and Bottom ("0".."9", 2 visible with ColorOverlay + BevelEmboss).
    12 -> 7 shows Top 7 + Bottom 0 and moves 2's effect onto 0, the two trading fxrp; 12 -> 15 shows Bottom 5; 12 -> 10 Bottom 0;
    20 refuses (no Bottom marker); the parser reads each result; gate bytes sees one flag byte per digit plus the moved blocks;
    hp (one row) refuses 11."""
    path = _card(clone, CANNON)[0]
    psd = PSDImage.open(path)
    parsed = pe.parse_card(parser, path, CANNON)
    assert parsed["card"]["atk"] == 12 and parsed["validator"] == ["STATS TOO HIGH"]
    assert parsed["card"].get("display_name") is None and pe.read_name(psd) == "Main Cannon"
    assert not hasattr(pe.CardValidator, "EXCESSIVE_STAT_EXCLUSIONS")
    assert sorted((group.name.strip(), digits) for group, digits, _ in stat_rows(psd)["atk"]) == [("Bottom", [2]), ("Top", [10])]
    top, bottom = pe.template_rows(psd, "atk")
    assert (top.group.name, top.visible, top.marker.name, top.marker.opacity, top.reads()) == ("Top", [], "10", 0, 10)
    assert (bottom.group.name, bottom.visible, bottom.marker, bottom.reads()) == ("Bottom", [2], None, 2)
    assert pe.stat_value([top, bottom]) == 12 and sorted(top.digits) == sorted(bottom.digits) == sorted(pe.DIGITS)
    assert pe.classify(parsed, {"atk": 15}, psd).jobs == pe.classify(parsed, {"atk": 15}).jobs == {"atk": 15}
    assert pe.classify(parsed, {"atk": 12}, psd).redundant == ["atk"] and pe.classify(parsed, {"atk": 7}).jobs == {"atk": 7}
    assert [(r.code, r.why) for r in pe.classify(parsed, {"atk": 21, "hp": 11}, psd).refusals] == [
        ("stat>10", "atk 21: above 20"), ("stat>10", "hp 11: one bar row; a second row is standardization work (ledger L3)")]
    for value, shows, names in ((7, ([7], [0]), ["7", "0", "2"]), (15, ([], [5]), ["5", "2"]), (10, ([], [0]), ["0", "2"])):
        psd = PSDImage.open(path)
        edit = pe.set_stat(psd, parsed, "atk", value)
        top, bottom = pe.template_rows(psd, "atk")
        assert (top.visible, bottom.visible) == shows and pe.stat_value([top, bottom]) == value and (edit.old, edit.new) == (12, value)
        assert [layer.name for layer in edit.layers] == names
        shown = str(shows[1][0])                                              # the Bottom digit now visible
        two, new_digit = pe._record_index(bottom.digits["2"]), pe._record_index(bottom.digits[shown])
        assert edit.moves == [(two, new_digit, b"lfx2"), (two, new_digit, b"lrFX"), (two, new_digit, b"fxrp"), (new_digit, two, b"fxrp")]
        saved = str(tmp_path / f"cannon_{value}.psd")
        pe.save_psd(psd, saved, update_preview=False)
        records, flag_bytes, block_bytes = pe.gate_bytes(path, saved, [edit])
        assert flag_bytes == len(names) and block_bytes == 6360 + 2 * 28 and sorted(name for _, name, _ in records) == sorted(names)
        assert {name: fields for _, name, fields in records} == {shown: STAT_RECORD, "2": STAT_RECORD, **({"7": ["flags"]} if value == 7 else {})}
        reopened = PSDImage.open(saved)
        reparsed = pe.gate_reparse(parser, parsed, saved, CANNON, {"atk": value}, reopened)
        assert reparsed["card"]["atk"] == value and reparsed["validator"] == (["STATS TOO HIGH"] if value > 10 else [])
        again = pe.template_rows(reopened, "atk")[1]
        assert sorted(pe._enabled_effects(again.digits[shown])) == ["BevelEmboss", "ColorOverlay"] and pe._enabled_effects(again.digits["2"]) == []
        assert pe.classify(reparsed, {"atk": value}, reopened).redundant == ["atk"]
    with pytest.raises(pe.Refused, match="10 marker") as refused:
        pe.set_stat(PSDImage.open(path), parsed, "atk", 20)
    assert refused.value.code == "stat>10"
    with pytest.raises(pe.Refused, match="ledger L3") as refused:
        pe.set_stat(PSDImage.open(path), parsed, "hp", 11)
    assert refused.value.code == "stat>10"
    # the replace path: hp 9 (styled) -> 2 (a block with nothing enabled): 2's block goes, 9's lands, the two trade fxrp
    psd = PSDImage.open(path)
    edit = pe.set_stat(psd, parsed, "hp", 2)
    row = pe.template_rows(psd, "hp")[0]
    two, nine = pe._record_index(row.digits["2"]), pe._record_index(row.digits["9"])
    assert edit.moves == [(two, None, b"lfx2"), (two, None, b"lrFX"), (nine, two, b"lfx2"), (nine, two, b"lrFX"),
                          (nine, two, b"fxrp"), (two, nine, b"fxrp")]
    saved = str(tmp_path / "cannon_hp2.psd")
    pe.save_psd(psd, saved, update_preview=False)
    records, flag_bytes, block_bytes = pe.gate_bytes(path, saved, [edit])
    assert (flag_bytes, block_bytes) == (2, 6360 + 2 * 28) and [(name, fields) for _, name, fields in records] == [("2", STAT_RECORD), ("9", STAT_RECORD)]
    assert sorted(pe._enabled_effects(pe.template_rows(PSDImage.open(saved), "hp")[0].digits["2"])) == ["BevelEmboss", "ColorOverlay"]
    # a raise from <= 10 adds STATS TOO HIGH on two rows: by design, not a new problem (review fix)
    seven = str(tmp_path / "cannon_7.psd")
    at_seven = pe.parse_card(parser, seven, CANNON)
    assert at_seven["card"]["atk"] == 7 and at_seven["validator"] == []
    for value in (15, 11):
        psd = PSDImage.open(seven)
        assert pe.classify(at_seven, {"atk": value}, psd).jobs == {"atk": value}
        edit = pe.set_stat(psd, at_seven, "atk", value)
        raised = str(tmp_path / f"cannon_7_to_{value}.psd")
        pe.save_psd(psd, raised, update_preview=False)
        reparsed = pe.gate_reparse(parser, at_seven, raised, CANNON, {"atk": value}, PSDImage.open(raised), [edit])
        assert reparsed["card"]["atk"] == value and reparsed["validator"] == ["STATS TOO HIGH"]
    # the marker is load-bearing: hidden, the parser reads Bottom alone and the shape refuses
    psd = PSDImage.open(path)
    pe.template_rows(psd, "atk")[0].marker.visible = False
    with pytest.raises(pe.Refused, match="10 marker"):
        pe.template_rows(psd, "atk")
    hidden = str(tmp_path / "cannon_no_marker.psd")
    pe.save_psd(psd, hidden, update_preview=False)
    assert pe.parse_card(parser, hidden, CANNON)["card"]["atk"] == 2


def test_effect_rules_on_styled_rows(clone, tmp_path):
    """Idea 12's rule on real rows: Mettaton atk 7 -> 8 moves nothing (7 is unstyled; the hidden styled 6 stays); 7 -> 6 keeps 6's
    own effect; hp 5 -> 10 moves nothing (5 keeps its block, hidden) and 10 -> 3 then takes it from 5; Green Mage before its fix
    (OFF_CANVAS_MAGE) atk 10 -> 4 takes the one hidden styled digit's (7) (its row's toggle: set_stat refuses its bars, below the
    canvas); the live Green Mage (atk 6, its 6 styled) 6 -> 4 takes 6's; Azamoth the Forgotten King's hp (10; 6, 8 and 9 styled)
    refuses as ambiguous."""
    path = _card(clone, METTATON)[0]
    parsed = {"card": {"hp": 5, "atk": 7}}
    psd = PSDImage.open(path)
    edit = pe.set_stat(psd, parsed, "atk", 8)
    row = pe.template_rows(psd, "atk")[0]
    assert edit.moves == [] and row.visible == [8] and pe._enabled_effects(row.digits["6"]) == ["GradientOverlay"]
    assert pe._enabled_effects(row.digits["8"]) == [] and [layer.name for layer in edit.layers] == ["8", "7"]
    psd = PSDImage.open(path)
    edit = pe.set_stat(psd, parsed, "atk", 6)
    row = pe.template_rows(psd, "atk")[0]
    assert edit.moves == [] and row.visible == [6] and pe._enabled_effects(row.digits["6"]) == ["GradientOverlay"]
    psd = PSDImage.open(path)
    edit = pe.set_stat(psd, parsed, "hp", 10)
    row = pe.template_rows(psd, "hp")[0]
    assert edit.moves == [] and row.visible == [] and [layer.name for layer in edit.layers] == ["5"]
    assert pe._enabled_effects(row.digits["5"]) == ["GradientOverlay"] and pe.stat_value([row]) == 10
    edit = pe.set_stat(psd, {"card": {"hp": 10}}, "hp", 3)
    five, three = pe._record_index(row.digits["5"]), pe._record_index(row.digits["3"])
    assert edit.moves == [(five, three, b"lfx2"), (five, three, b"lrFX")]                 # a gradient tiles nothing: no fxrp trade
    assert [layer.name for layer in edit.layers] == ["3", "5"]
    assert row.visible == [3] and pe._enabled_effects(row.digits["3"]) == ["GradientOverlay"] and pe._enabled_effects(row.digits["5"]) == []
    green = PSDImage.open(_at(clone, tmp_path, OFF_CANVAS_MAGE, GREEN_MAGE)[0])
    row = pe.template_rows(green, "atk")[0]
    changed, moves, _ = pe._toggle_row(row, 4, "atk")
    assert [layer.name for layer in changed] == ["4", "7"] and [bytes(key) for _, _, key in moves] == [b"lfx2", b"lrFX"]
    assert pe._enabled_effects(row.digits["4"]) == ["GradientOverlay"] and pe._enabled_effects(row.digits["7"]) == []
    green = PSDImage.open(_card(clone, GREEN_MAGE)[0])
    edit = pe.set_stat(green, {"card": {"atk": 6}}, "atk", 4)
    row = pe.template_rows(green, "atk")[0]
    assert [layer.name for layer in edit.layers] == ["4", "6"] and [bytes(key) for _, _, key in edit.moves] == [b"lfx2", b"lrFX"]
    assert row.visible == [4] and pe._enabled_effects(row.digits["4"]) == ["GradientOverlay"] and pe._enabled_effects(row.digits["6"]) == []
    with pytest.raises(pe.Refused, match="6, 8, 9") as refused:
        pe.set_stat(PSDImage.open(_card(clone, AZAMOTH_KING)[0]), {"card": {"hp": 10}}, "hp", 3)
    assert refused.value.code == "unsupported-card"


def test_a_bar_outside_the_card_refuses(clone, tmp_path):
    """Green Mage before its fix (OFF_CANVAS_MAGE): its four bars lay below its 2400 px canvas (the printed cells were pixel
    layers): a toggle there would change nothing on the card, so every stat refuses `unsupported-card` before a layer changes
    (it was a ValueError from the empty render region: an `error` that failed the run). The live card's bars are on the canvas:
    every stat toggles. Pure Vessel's masks start on the canvas and reach past it: it toggles (its bake: test_card_bake's
    test_a_bar_outside_the_card_is_a_final_refusal)."""
    path = _at(clone, tmp_path, OFF_CANVAS_MAGE, GREEN_MAGE)[0]
    parsed = {"card": {"hp": 7, "def": 4, "atk": 10, "spd": 7}}
    for stat, value in (("hp", 3), ("def", 5), ("atk", 6), ("spd", 10)):
        psd = PSDImage.open(path)
        before = [layer.visible for layer in pe.template_rows(psd, stat)[0].digits.values()]
        with pytest.raises(pe.Refused, match=rf"^unsupported-card: {stat}: the bar '{stat.capitalize()} Dark' lies outside the "
                                             r"card \(digit masks \(960, \d+, 1773, \d+\) on the 1200x2400 canvas\)"):
            pe.set_stat(psd, parsed, stat, value)
        assert [layer.visible for layer in pe.template_rows(psd, stat)[0].digits.values()] == before
    parsed = {"card": {"hp": 7, "def": 4, "atk": 6, "spd": 7}}
    for stat, value in (("hp", 3), ("def", 5), ("atk", 4), ("spd", 10)):
        psd = PSDImage.open(_card(clone, GREEN_MAGE)[0])
        x0, y0, x1, y1 = pe.set_stat(psd, parsed, stat, value).region
        assert 0 <= x0 < x1 <= psd.width and 0 <= y0 < y1 <= psd.height
    pure = PSDImage.open(_card(clone, PURE_VESSEL)[0])
    edit = pe.set_stat(pure, {"card": {"hp": 7}}, "hp", 3)
    x0, y0, x1, y1 = edit.region
    assert [layer.name for layer in edit.layers] == ["3", "7"] and 0 <= x0 < x1 <= pure.width and 0 <= y0 < y1 <= pure.height


# ========================
# Stats (the fork: viewport renders)
# ========================
def test_export_icc_follows_the_png(clone):
    """The harness's ICC vote over the stored preview: a raw export votes 1.0 (render unconverted), a converted one 0.0, a
    card without a foreign profile has no vote (converted, as before)."""
    for rel, icc in ((RED_BLOON, False), (BLOCK_HOPPER, False), (MINI_DOOMER, True), (METTATON, True)):
        psd_path, png_path = _card(clone, rel)
        psd, png = PSDImage.open(psd_path), Image.open(png_path)
        vote, moved = pe.icc_vote(psd, png)
        assert moved >= pe.ICC_MIN_PX and vote == (0.0 if icc else 1.0) and pe.export_icc(psd, png) is icc
    plain = PSDImage.new("RGB", (4, 4))
    assert pe.foreign_profile(plain) is None and pe.icc_vote(plain, Image.new("RGB", (4, 4))) is None
    assert pe.export_icc(plain, Image.new("RGB", (4, 4)))


@text_engine
def test_mini_doomer_hp_5_to_8(clone, parser, tmp_path):
    """S2's gate: 2 flag bytes, the parser reads 8, 0 pixels differ outside the row mask, a rerun does nothing."""
    parsed, baked = _bake(clone, parser, MINI_DOOMER, {"hp": 8}, tmp_path, update_preview=True)
    assert parsed["card"]["hp"] == 5 and baked.reparse["card"]["hp"] == 8 and baked.reparse["validator"] == []
    assert baked.flag_bytes == 2 and [(name, fields) for _, name, fields in baked.records] == [("5", ["flags"]), ("8", ["flags"])]
    assert baked.block_bytes == 0 and baked.edits[0].moves == []
    assert list(baked.parity) == ["hp"] and baked.parity["hp"]["bad_pct"] == 0 and baked.parity["hp"]["mean"] <= pe.PARITY_MEAN_MAX
    assert clip(baked.region, 1200, 2400) == (763, 1199, 1174, 1285)      # digits 5 and 8 + the Stats Bars stroke
    old = np.asarray(Image.open(_card(clone, MINI_DOOMER)[1]).convert("RGBA"))
    new = np.asarray(Image.open(baked.png_path).convert("RGBA"))
    differs = (old != new).any(axis=2)
    row = np.zeros_like(differs)
    row[HP_ROW[1]:HP_ROW[3], HP_ROW[0]:HP_ROW[2]] = True
    assert int((differs & ~row).sum()) == 0 and 0 < int(differs.sum()) <= baked.changed_pixels   # a replaced pixel may equal the PNG's
    saved = PSDImage.open(baked.psd_path)
    assert saved.has_preview() and {d.name for d in pe.template_row(saved, "hp")[1].values() if d.visible} == {"8"}
    if saves_preview():                 # else save() regenerated it
        old_preview = PSDImage.open(_card(clone, MINI_DOOMER)[0]).topil(apply_icc=False)
        moved = (np.asarray(old_preview) != np.asarray(saved.topil(apply_icc=False))).any(axis=2)
        # Photoshop's preview, the merge's pixels replaced (outside the PNG's changes only where a replaced pixel kept its value)
        assert moved.any() and int((moved & ~row).sum()) == 0
        assert int((moved & ~differs).sum()) <= baked.changed_pixels - int(differs.sum())
    assert Image.open(baked.png_path).info.get("icc_profile") == Image.open(_card(clone, MINI_DOOMER)[1]).info.get("icc_profile")
    again = pe.classify(baked.reparse, {"hp": 8}, saved)
    assert again.jobs == {} and again.redundant == ["hp"]
    print("timings", {k: round(v, 2) for k, v in baked.timings.items()})


# ========================
# Text (the fork: text engine)
# ========================
@text_edit
@pytest.mark.parametrize("rel", NAMED)
def test_name_edit_passes_every_gate(clone, parser, chart, rel, tmp_path):
    psd_path = _card(clone, rel)[0]
    title = pe.read_name(PSDImage.open(psd_path))
    new = _typo_fix(title)
    parsed, baked = _bake(clone, parser, rel, {"name": new}, tmp_path, chart)
    assert [(name, fields) for _, name, fields in baked.records] == [("Name", TEXT_RECORD)] and baked.flag_bytes == 0
    without = {k: v for k, v in parsed["card"].items() if k != "display_name"}
    assert baked.reparse["card"] == {**without, "display_name": new}      # the parser reads the new title
    saved = PSDImage.open(baked.psd_path)
    assert pe.read_name(saved) == new
    assert pe.stories(saved)[pe.story_index(pe.name_layer(saved))] == new + "\r"   # what Photoshop reads
    assert pe.stale_stories(saved) == []
    assert pe.classify(baked.reparse, {"name": new}, saved).redundant == ["name"]
    assert pe.classify(baked.reparse, {"name": title}, saved).jobs == {"name": title}


@text_edit
@pytest.mark.parametrize("rel", [PAPYRUS])
def test_name_edit_on_a_title_named_after_its_text(clone, parser, chart, rel, tmp_path):
    """Papyrus's and Sans's titles bake through every gate; the layer takes the new title as its name, as Photoshop
    renames a layer it named, so the parser reads it back and nothing is left to bake."""
    psd_path = _card(clone, rel)[0]
    title = pe.read_name(PSDImage.open(psd_path), card_name(rel))
    new = _typo_fix(title)
    parsed, baked = _bake(clone, parser, rel, {"name": new}, tmp_path, chart)
    assert [name for _, name, _ in baked.records] == [title] and [e.renamed for e in baked.edits] == [new]
    assert {"blocks", "channels", "name"} <= set(baked.records[0][2]) <= set(TEXT_RECORD + ["name"])
    assert baked.reparse["card"]["display_name"] == new and baked.reparse["validator"] == parsed["validator"]
    saved = PSDImage.open(baked.psd_path)
    assert _layer_path(pe.name_layer(saved, card_name(rel))) == "Text/" + new and pe.stale_stories(saved) == []
    assert pe.classify(baked.reparse, {"name": new}, saved).redundant == ["name"]


@text_edit
def test_a_bake_runs_gate_untouched(clone, parser, chart, tmp_path, monkeypatch):
    """apply_jobs runs gate_untouched: Claw's new FX layer put directly above its Text group, as before round 5, takes
    MagentaSmorg and Exposure 1 off the title; the bake refuses region (review 2026-10-24)."""
    rel = "Auxiliary/Minions/Claw.psd"
    psd_path = _card(clone, rel)[0]
    new = pe.normalize_ability(pe.parse_card(parser, psd_path, rel)["card"]["ability"]) + " Inflict Blazing."
    made = pe._regenerate_fx

    def under_chain(psd, layer, *args):
        out = made(psd, layer, *args)
        group = pe._top(layer)
        for n, fx in enumerate(out):
            fx.move_to(psd, index=psd.index(group) + 1 + n)
        return out
    monkeypatch.setattr(pe, "_regenerate_fx", under_chain)
    with pytest.raises(pe.Refused, match="outside the edited boxes") as refused:
        _bake(clone, parser, rel, {"ability": new}, tmp_path, chart)
    assert refused.value.code == "region"


@text_edit
def test_long_title_is_flagged(clone, chart):
    """Point text has no box: a title that runs off the canvas or onto a star badge bakes, flagged overflow (owner
    2026-10-24: the editor previewed it)."""
    fonts = str(clone / pe.FONTS_DIR)
    psd = PSDImage.open(_card(clone, ANUBISATH)[0])
    title = pe.read_name(psd)
    flagged(pe.set_text_layer(psd, "name", title + " Of The Temple", fonts, chart).notes, "^overflow: name runs off the canvas$")
    psd = PSDImage.open(_card(clone, ANUBISATH)[0])                            # 'Anubisath Guardian II' runs into the first star
    flagged(pe.set_text_layer(psd, "name", title + " II", fonts, chart).notes, "^overflow: name reaches the stars")
    psd = PSDImage.open(_card(clone, MINI_DOOMER)[0])                          # centred: 'Mini Doomer II' ends 3 px short of the star
    edit = pe.set_text_layer(psd, "name", pe.read_name(psd) + " II", fonts, chart)
    assert edit.notes == [] and tuple(pe.name_layer(psd).bbox)[2] < 936


@text_edit
def test_ability_edit_colours_a_status_word(clone, parser, chart, tmp_path):
    paths = _plain(clone, tmp_path)
    old = pe.parse_card(parser, paths[0], MINI_DOOMER)["card"]["ability"]
    parsed, baked = _bake(clone, parser, MINI_DOOMER, {"ability": old + " Inflict Poison."}, tmp_path, chart, paths=paths)
    assert [(name, fields) for _, name, fields in baked.records] == [("Ability", TEXT_RECORD)]
    assert pe.normalize_ability(baked.reparse["card"]["ability"]) == pe.normalize_ability(old + " Inflict Poison.")
    layer = pe.ability_layer(PSDImage.open(baked.psd_path))
    text = layer.text
    start = text.index("Poison")
    fills = [fill for s, e, fill in pe.run_fills(layer) if s <= start < e]
    assert fills == [chart["poison"].fill] and chart["poison"].styled is False
    assert baked.region[1] < 1815 and baked.region[3] > 2245                 # the ability box plus the Text group's halo


def test_fx_kind_names_how_each_fx_layer_is_made_again(chart):
    """BUG 2: an ability edit makes every FX layer again, each keeping its own style (owner q6): badge or token runs
    (none), a styled chart status, another word under a chart status's style (Monkey D Goofy's "1#1" under Become Back
    Money's gradient), and since 2026-10-23 one in a style no chart status has, or none ("own": Smoldering's stroke,
    +2/4/6 ATK's; the owner's ruling, the 2026-10-21 refusal gone). Only an FX layer that is no pixel layer refuses
    (Photoshop would draw a type layer from its text)."""
    base = SimpleNamespace(name="Ability")
    names = ["Ability FX: (1CD) | (3U) | (OYT) | (ENDT)", "Ability FX: [fire]", "Ability FX: Blazing", "Ability FX: Frozen#2",
             "Ability FX: Smoldering", "Ability FX: +2/4/6 ATK", "Ability FX: 1#1"]
    fx = [SimpleNamespace(name=name, kind="pixel") for name in names]
    assert [pe.fx_label(base, layer) for layer in fx] == ["(1CD) | (3U) | (OYT) | (ENDT)", "[fire]", "Blazing", "Frozen",
                                                           "Smoldering", "+2/4/6 ATK", "1"]
    assert [pe.fx_kind(layer, base, chart) for layer in fx] == ["badge", "badge", "status", "status", "own", "own", "own"]
    assert [pe.fx_kind(layer, base, None) for layer in fx] == ["badge", "badge"] + ["own"] * 5
    goofy = SimpleNamespace(name="Ability FX: 1#1", kind="pixel", tagged_blocks=chart["become back money"].blocks)
    assert pe.fx_kind(goofy, base, chart) == "styled" and set(pe.ANCHORED) == {"styled", "own"}
    typed = [SimpleNamespace(name=name, kind="type") for name in names[:4]]
    assert [pe.fx_kind(layer, base, chart) for layer in typed] == [""] * 4
    assert pe.fx_phrases(base, fx[0]) == ["(1CD)", "(3U)", "(OYT)", "(ENDT)"] and pe.fx_phrases(base, fx[3]) == ["Frozen#2"]


def test_phrase_ranges_follow_the_jsx():
    """Effect Layers.jsx's phraseRanges: any case, any blanks between words, whole words first, else every match, #n the
    nth; the parser's EFFECT LAYER WORDS NOT IN TEXT counts alike."""
    goofy = "Attacks inflict either 3T2 Smeckeldorfed or 3T1 Become Back Money.\r(1CD) Draw"
    assert pe.phrase_ranges(goofy, ["1#1"]) == [(46, 47)] and pe.phrase_ranges(goofy, ["become back money"]) == [(48, 65)]
    assert pe.phrase_ranges(goofy, ["1"]) == [(46, 47), (68, 69)] and pe.phrase_ranges(goofy, ["1#3", "Poison"]) == []
    text = "Blazing blazingly blazing\x03Frozen In  Time"
    assert pe.phrase_ranges(text, ["Blazing", "frozen in time", "Blazing#2"]) == [(0, 7), (18, 25), (26, 41), (18, 25)]
    assert pe.phrase_ranges("Blazingly", ["Blazing"]) == [(0, 7)]


def test_follow_fx_refuses_what_would_move_an_anchored_layer(chart):
    """An "own" or "styled" phrase found only inside other words (Mr Krabs' "2T" of "2T1"): a new such match typed would
    join the layer under Update all, and a whole word typed would take the JSX's match off its own: both refuse
    unsupported-card; its own match kept bakes. A numbered status layer on a status word a digit touches ("2Frozen")
    refuses a whole-word one typed, which would take its "#1"."""
    old = "Inflict 2T1 Become Back Money."
    plan = pe.FxPlan(SimpleNamespace(name="Ability FX: 2T"), ["2T"], "own", ranges=[(8, 10)], found=[("2T", [(8, 10)])])
    for new, why in ((old + " Then 2T3 Poison.", "the '2T' of '2T3' would join the layer Ability FX: 2T; edit in Photoshop"),
                     (old + " Then 2T.", "a whole-word '2T' in the ability would move the layer Ability FX: 2T off the "
                                         "'2T' of '2T1'; edit in Photoshop")):
        with pytest.raises(pe.Refused) as refused:
            pe.follow_fx(plan, new, pe.kept_chars(old, new))
        assert (refused.value.code, refused.value.why) == ("unsupported-card", why)
    assert pe.follow_fx(plan, old + " Die.", pe.kept_chars(old, old + " Die."))[0] == ("2T", "2T", [(8, 10)], [(8, 10)], "")
    old = "Inflict 2Frozen."
    plan = pe.FxPlan(SimpleNamespace(name="Ability FX: Frozen#1"), ["Frozen#1"], "status", ranges=[(9, 15)],
                     found=[("Frozen#1", [(9, 15)])], status="frozen")
    new = old + " Then Frozen."
    with pytest.raises(pe.Refused, match="^unsupported-card: a whole-word 'Frozen' in the ability would move the layer "
                                         "Ability FX: Frozen#1 off the 'Frozen' of '2Frozen'; edit in Photoshop$"):
        pe.follow_fx(plan, new, pe.kept_chars(old, new), pe.status_spans(new, chart, []))


@text_edit
def test_ambiguous_fx_layers(clone, chart):
    """On Morpho Knight and Mini Doomer altered in memory: an FX layer whose words now fall in a badge (an own layer
    named "CD", the letters of its "(1CD)": the JSX would cut every badge into it) goes, told; two FX layers that would
    end with one name (+2/4/6 ATK renamed a second Smoldering) and an FX layer that is no pixel layer (a type layer named
    like one: Photoshop would draw it from its text) refuse unsupported-card."""
    fonts = str(clone / pe.FONTS_DIR)

    def morpho(rename: dict):
        psd = PSDImage.open(_card(clone, MORPHO)[0])
        layer = pe.ability_layer(psd)
        for fx in pe.fx_layers(psd, layer):
            fx.name = rename.get(fx.name, fx.name)
        return psd, pe.normalize_ability(layer.text) + " Die."

    psd, text = morpho({"Ability FX: Burning": "Ability FX: CD"})
    edit = pe.set_text_layer(psd, "ability", text, fonts, chart)
    assert "Ability FX: CD deleted: its words now fall in a badge or token" in edit.notes and "Ability FX: CD" in edit.dropped
    psd, text = morpho({"Ability FX: +2/4/6 ATK": "Ability FX: Smoldering"})
    with pytest.raises(pe.Refused, match="^unsupported-card: two FX layers would be named Ability FX: Smoldering; edit in "
                                         "Photoshop$"):
        pe.set_text_layer(psd, "ability", text, fonts, chart)
    psd = PSDImage.open(_card(clone, MINI_DOOMER)[0])
    pe.name_layer(psd).name = "Ability FX: Die"
    layer = pe.ability_layer(psd)
    with pytest.raises(pe.Refused, match="^unsupported-card: FX layers that are no pixel layers: Ability FX: Die; edit in "
                                         "Photoshop$"):
        pe.set_text_layer(psd, "ability", pe.normalize_ability(layer.text) + " Die.", fonts, chart)


@text_edit
def test_own_style_fx_layers_are_made_again_in_their_style(clone, parser, chart, tmp_path):
    """Owner 2026-10-23: FX layers in no chart status's style are rebuilt in their own style, as the others (the 2026-10-21
    refusal went). Morpho Knight's Burning, +2/4/6 ATK and Smoldering strokes bake an appended sentence through every
    gate: each record kept whole or cut again (rect and channels), its style bytes unchanged, no new layer; its badge
    layer goes (retired, owner 2026-10-25). Without the chart the card still refuses (its FX words can't be told apart)."""
    paths = _at(clone, tmp_path, BADGED, MORPHO)
    psd_path = paths[0]
    psd = PSDImage.open(psd_path)
    layer = pe.ability_layer(psd)
    text, fx = pe.normalize_ability(layer.text), [f.name for f in pe.fx_layers(psd, layer)]
    assert [pe.fx_kind(f, layer, chart) for f in pe.fx_layers(psd, layer)] == ["badge", "own", "status", "own", "own"]
    with pytest.raises(pe.Refused, match="status word layers over the ability"):
        pe.set_text_layer(psd, "ability", text + " Die.", str(clone / pe.FONTS_DIR), None)
    old = pe.normalize_ability(pe.parse_card(parser, psd_path, MORPHO)["card"]["ability"])
    parsed, baked = _bake(clone, parser, MORPHO, {"ability": old + " Die."}, _dir(tmp_path, "bake"), chart, paths=paths)
    edit, = baked.edits
    badge, words = fx[0], fx[1:]
    assert edit.fx_names == words and edit.fx == [] and edit.dropped == [badge] and baked.block_bytes == 0
    for name in words:
        assert _style(baked.psd_path, name) == _style(psd_path, name)
    assert all(fields in (["channels", "rect"], ["channels"]) for _, name, fields in baked.records if name in words)   # salt font: the box stays
    assert (badge, ["removed"]) in [(name, fields) for _, name, fields in baked.records]


def _noop(clone, parser, chart, rel, tmp_path, paths=None):
    """apply_jobs with the ability layer's own text (classify would call it redundant): (psd path, png path, Baked)."""
    psd_path, png_path = paths or _card(clone, rel)
    text = pe.ability_layer(PSDImage.open(psd_path)).text.replace("\r", "\n")
    parsed = pe.parse_card(parser, psd_path, rel)
    return psd_path, png_path, pe.apply_jobs(psd_path, png_path, rel, {"ability": text}, parsed, parser, str(tmp_path),
                                             str(clone / pe.FONTS_DIR), chart, update_preview=False)


@text_edit
def test_noop_ability_bake_retires_the_badge_layer(clone, parser, chart, tmp_path):
    """Owner 2026-10-25: badge FX layers are retired. A no-op ability bake on Mini Doomer (a Gradient Overlay on Text)
    deletes 'Ability FX: (BOT)', told, and sets Stylistic Alternates on its "(BOT)", in the Txt2 story only (TySh has no
    such key), every gate passing; the re-parse reads the same text."""
    paths = _at(clone, tmp_path, BADGED, MINI_DOOMER)
    psd_path, png_path, baked = _noop(clone, parser, chart, MINI_DOOMER, tmp_path, paths)
    edit, = baked.edits
    assert edit.fx_names == [] and edit.dropped == ["Ability FX: (BOT)"] and edit.rebuilt == [] and edit.fx == []
    assert baked.fx_notes == ["Ability FX: (BOT) deleted: badge FX layers are retired"]
    assert ("Ability FX: (BOT)", ["removed"]) in [(name, fields) for _, name, fields in baked.records]
    layer = pe.ability_layer(PSDImage.open(baked.psd_path))
    at = layer.text.index("(BOT)")
    assert pe.one_colour(layer) and pe.stylistic_alternates(layer) == ((at, at + 5),)
    assert not any("StylisticAlternates" in run["StyleSheet"]["StyleSheetData"] for run in layer.engine_dict["StyleRun"]["RunArray"])
    assert pe.stale_stories(PSDImage.open(baked.psd_path)) == [] and baked.reparse["validator"] == []


SNOW_BUNNY = "Items/Heal/2 Stars/Snow_Bunny.psd"   # L7: 'Ability FX: Frozen' keeps its old overlay's style, not the chart's


def _style(path: str, name: str) -> tuple:
    """An FX record's style bytes: blending, opacity, clipping, flags and every tagged block (lfx2, lrFX, iOpa...)."""
    record, = [r for r in pe.layer_records(path) if r["name"] == name]
    return record["blend"], record["flags"], record["blocks"]


@text_edit
def test_fx_gates_refuse_a_wrong_cut(clone, parser, chart, tmp_path):
    """BUG 2 review: the gates on a rebuilt FX layer refuse a wrong one. Snow Bunny's Frozen moved by an edit, then its
    FX layer shifted, cut short, or not the Ability's pixels: gate_fx refuses (reparse), and for a faint stripe away
    from its words too when the layer counts as kept (the own-pixels check skipped); its opacity or name changed on the
    saved card: gate_bytes refuses (bytes)."""
    psd_path = _card(clone, SNOW_BUNNY)[0]
    text = pe.normalize_ability(pe.parse_card(parser, psd_path, SNOW_BUNNY)["card"]["ability"]).replace("at least ", "")
    parsed, baked = _bake(clone, parser, SNOW_BUNNY, {"ability": text}, tmp_path, chart)
    fonts, store = str(clone / pe.FONTS_DIR), pe._fx_engine()[6]

    def tampered(how):
        psd = PSDImage.open(baked.psd_path)
        fx, = pe.fx_layers(psd, pe.ability_layer(psd))
        x0, y0, x1, y1 = fx.bbox
        rgba = np.concatenate([fx.numpy("color"), fx.numpy("shape")], axis=2).astype(np.float32)
        if how == "shifted":
            store(fx, rgba, (x0 + 6, y0, x1 + 6, y1))
        elif how == "short":
            store(fx, rgba[:, :rgba.shape[1] // 2].copy(), (x0, y0, x0 + rgba.shape[1] // 2, y1))
        elif how == "repainted":
            rgba[..., :3] = np.where(rgba[..., 3:] > 0, 1.0 - rgba[..., :3], 0.0)
            store(fx, rgba, (x0, y0, x1, y1))
        elif how == "faint":                    # a 10 px stripe at alpha 0.3, 60 px right of the words
            wide = np.zeros((rgba.shape[0], rgba.shape[1] + 70, 4), np.float32)
            wide[:, :rgba.shape[1]] = rgba
            wide[:, rgba.shape[1] + 60:, :] = (1.0, 1.0, 1.0, 0.3)
            store(fx, wide, (x0, y0, x1 + 70, y1))
        elif how == "opacity":
            fx.opacity = 200
        elif how == "renamed":
            fx.name = "Ability FX: Thawed"
        fx._invalidate_bbox()
        return psd

    pe.gate_fx(PSDImage.open(baked.psd_path), baked.edits, fonts)
    for how, why in (("shifted", "inks .* px outside its words' glyph boxes"), ("short", "covers .* of its words'"),
                     ("repainted", "is not the Ability's own pixels")):
        with pytest.raises(pe.Refused, match=r"'Ability FX: Frozen' " + why) as refused:
            pe.gate_fx(tampered(how), baked.edits, fonts)
        assert refused.value.code == "reparse"
    with pytest.raises(pe.Refused, match=r"'Ability FX: Frozen' inks \d+ px outside"):
        pe.gate_fx(tampered("faint"), [replace(baked.edits[0], rebuilt=[])], fonts)
    renamed = r"renamed 'Ability FX: Thawed', not 'Ability FX: Frozen'"     # gate_bytes' name check fires first
    for how, why in (("opacity", "changed .*blend"), ("renamed", renamed)):
        path = tmp_path / f"{how}.psd"
        tampered(how).save(str(path))
        with pytest.raises(pe.Refused, match=why) as refused:
            pe.gate_bytes(psd_path, str(path), baked.edits)
        assert refused.value.code == "bytes"


@text_edit
def test_a_typed_badge_takes_the_alternates_under_an_overlay(clone, parser, chart, tmp_path):
    """Owner 2026-10-23: a timing tag typed anywhere is set in the stoogedings badge font, at the card's own badge style
    (Mini Doomer's 50 pt (BOT)). Under the overlay (owner 2026-10-25) it and the kept "(BOT)" take Stylistic Alternates,
    one-colour pills; the body text typed beside them does not, nor the retired badge layer stay. Every gate passes; the
    parser reads it back."""
    paths = _at(clone, tmp_path, BADGED, MINI_DOOMER)
    old = pe.normalize_ability(pe.parse_card(parser, paths[0], MINI_DOOMER)["card"]["ability"])
    parsed, baked = _bake(clone, parser, MINI_DOOMER, {"ability": old + " Then (bot) wins."}, tmp_path, chart, paths=paths)
    edit, = baked.edits
    assert edit.fx_names == [] and edit.dropped == ["Ability FX: (BOT)"] and edit.fonts == []
    assert baked.reparse["card"]["ability"].endswith("(BOT) Die. Then (bot) wins.") and baked.reparse["validator"] == []
    layer = pe.ability_layer(PSDImage.open(baked.psd_path))
    runs = _runs(layer)
    assert [run[:3] for run in runs if run[1] == pe.ICON_FONT] == [("(BOT)", pe.ICON_FONT, 50.0), ("(bot)", pe.ICON_FONT, 50.0)]
    text = layer.text
    assert [text[a:b] for a, b in pe.stylistic_alternates(layer)] == ["(BOT)", "(bot)"]
    assert not any("Stylistic" in note for note in baked.fx_notes)


@text_edit
def test_typed_icons_that_print_wrong_are_flagged(clone, parser, chart):
    """Baked and told (owner 2026-10-24): a colour icon typed under the Ability's colour or gradient overlay (Cat; a badge
    or sticker takes the one-colour alternates there, owner 2026-10-25), a [type] token on a card whose type icons are image layers in the ability's gaps
    (Nebulous Precipice's darkmatter: the website's text holds their tokens), and a badge replaced by letters the icon
    font draws plainly ("(abc)")."""
    fonts = str(clone / pe.FONTS_DIR)
    for rel, edit, why in (
            ("Auxiliary/Minions/Cat.psd", lambda t: t + " [fire] Die.", r"^unsupported-card: ability: \[fire\]: the Ability's "
                                                                       r"look \(its Color Overlay\) changes the typed icon's "
                                                                       r"colours$"),
            ("Field/3 Stars/Nebulous_Precipice.psd", lambda t: t + " Deal 1 [fire] damage.", r"\[type\] tokens on a card with image icons"),
            (MINI_DOOMER, lambda t: t.replace("(BOT)", "(abc)"), r"'\(abc\)' set in the icon font, which draws no badge")):
        psd = PSDImage.open(_card(clone, rel)[0])
        text = edit(pe.normalize_ability(pe.ability_layer(psd).text))
        flagged(pe.set_text_layer(psd, "ability", text, fonts, chart, parser=parser).notes, why)
    psd = PSDImage.open(_card(clone, "Auxiliary/Minions/Cat.psd")[0])
    text = pe.normalize_ability(pe.ability_layer(psd).text) + " (1CD) [fire:sticker] Die."
    notes = pe.set_text_layer(psd, "ability", text, fonts, chart, parser=parser).notes
    assert not any("look" in note for note in notes)
    layer = pe.ability_layer(psd)
    assert [layer.text[a:b] for a, b in pe.stylistic_alternates(layer)][-2:] == ["(1CD)", "[fire:sticker]"]


TURTLE = "Auxiliary/Minions/Turtle.psd"               # '(1GCD)' under a Color Overlay, 'Ability FX: (1GCD)' over it
BLOOD_BOMB = "Items/Attack/3 Stars/Blood_Bomb.psd"       # Pin Light: its [machine] token bare, no badge FX layer


@text_edit
def test_typed_icons_under_any_tint(clone, parser, chart, tmp_path):
    """Review 2026-10-23: the tint counted only an opaque Normal overlay. Soul of Eden's Pin Light blend printed a typed
    (2CD) white on cyan, Ice Staff's two 69% gradients a pale blue one and Holy Mantle's Pin Light a washed [fire], The
    Patients' 78% opacity a dimmed one, Claw's and Miasma's Exposure 1 clipped to Text a pale one (reviews 2026-10-24):
    _tinted names each, and the edit is flagged. Binge Eater, Pin Light with a badge layer: the layer goes (retired,
    owner 2026-10-25) and a typed badge is told as any (no overlay: no alternates)."""
    fonts = str(clone / pe.FONTS_DIR)
    for rel, why in (("Items/Equip/2 Stars/Soul_of_Eden.psd", "its Pin Light blending"),
                     ("Items/Equip/2 Stars/Ice_Staff.psd", "its Gradient Overlay"),
                     ("Items/Equip/4 Stars/Holy_Mantle.psd", "its Pin Light blending"),
                     ("Creatures/Other/3 Stars/The_Patients.psd", "its opacity or fill below full"),
                     ("Auxiliary/Minions/Claw.psd", "'Exposure 1', clipped to its group 'Text'"),
                     ("Auxiliary/Minions/Miasma.psd", "'Exposure 1', clipped to its group 'Text'")):
        layer = pe.ability_layer(PSDImage.open(_card(clone, rel)[0]))
        assert pe._tinted(layer) == why and not pe._overlaid(layer)
        assert pe.one_colour(layer) == ("Overlay" in why)
    psd = PSDImage.open(_card(clone, "Items/Equip/4 Stars/Holy_Mantle.psd")[0])
    edit = pe.set_text_layer(psd, "ability", pe.normalize_ability(pe.ability_layer(psd).text) + " [fire]", fonts, chart,
                             parser=parser)
    flagged(edit.notes, r"^unsupported-card: ability: \[fire\]: the Ability's look \(its Pin Light blending\) changes the "
                        r"typed icon's colours$")
    assert not pe._tinted(pe.ability_layer(PSDImage.open(_card(clone, "Auxiliary/Creatures/Mettaton_NEO.psd")[0])))
    psd = PSDImage.open(_at(clone, tmp_path, BADGED, "Items/Equip/4 Stars/Binge_Eater.psd")[0])
    text = pe.normalize_ability(pe.ability_layer(psd).text) + " (2CD) Die."
    edit = pe.set_text_layer(psd, "ability", text, fonts, chart, parser=parser)
    assert edit.fx_names == [] and edit.dropped == ["Ability FX: (1CD)"]
    assert "Ability FX: (1CD) deleted: badge FX layers are retired; the Ability's look (its Pin Light blending) now " \
           "changes its badges' colours" in edit.notes
    flagged(edit.notes, r"^unsupported-card: ability: \(2CD\): the Ability's look \(its Pin Light blending\) changes the "
                        r"typed badge's colours$")


@text_edit
def test_a_badge_edited_by_deleting_letters_takes_the_alternates(clone, parser, chart, tmp_path):
    """Review 2026-10-23: Turtle's "(1GCD)" made "(1CD)" keeps every character, so it does not count as typed; under
    the Ability's Color Overlay it takes Stylistic Alternates all the same (every badge does), and the badge layer goes
    (retired, owner 2026-10-25), every gate passing, nothing told about colours."""
    paths = _at(clone, tmp_path, BADGED, TURTLE)
    old = pe.normalize_ability(pe.parse_card(parser, paths[0], TURTLE)["card"]["ability"])
    parsed, baked = _bake(clone, parser, TURTLE, {"ability": old.replace("(1GCD)", "(1CD)")}, tmp_path, chart, paths=paths)
    edit, = baked.edits
    assert edit.fx_names == [] and edit.dropped == ["Ability FX: (1GCD)"] and baked.fx == []
    assert baked.fx_notes == ["Ability FX: (1GCD) deleted: badge FX layers are retired"]
    layer = pe.ability_layer(PSDImage.open(baked.psd_path))
    assert [layer.text[a:b] for a, b in pe.stylistic_alternates(layer)] == ["(1CD)"]


@text_engine
def test_alternates_go_on_badges_and_stickers_only(clone):
    """set_alternates (owner 2026-10-25): Stylistic Alternates on the badges and stickers, off on a typed character that
    took them from the badge before it; a kept one keeps its flag. alternate_spans picks badges and :stickers only."""
    text = "(1CD) Burn [fire] and [fire:sticker]."
    drawn = [(0, 5), (11, 17), (22, 36)]
    assert pe.alternate_spans(text, drawn) == [(0, 5), (22, 36)]
    psd = PSDImage.open(_card(clone, MINI_DOOMER)[0])
    layer = pe.ability_layer(psd)
    at, end = layer.text.index("(BOT)"), len(layer.text)
    layer.set_span_style(0, end, StylisticAlternates=True)
    pe.set_alternates(layer, [(at, at + 5)], [(at + 5, end)])
    assert pe.stylistic_alternates(layer) == ((0, at + 5),)
    pe.set_alternates(layer, [(at, at + 5)], [(0, end)])
    assert pe.stylistic_alternates(layer) == ((at, at + 5),)


@text_engine
def test_restyle_due_names_layers_an_edit_leaves(clone, tmp_path):
    """A layer of a several-layer Ability the edit leaves alone is set all the same when a retired badge FX layer is over
    it (Miracle Matter's first) or its badge under an overlay lacks Stylistic Alternates; the rest are left."""
    psd = PSDImage.open(_at(clone, tmp_path, BADGED, MIRACLE)[0])
    layers = pe.ability_layers(psd).layers
    due = [pe._restyle_due(psd, layer, None) for layer in layers]
    assert due[0] and not any(due[1:])
    psd = PSDImage.open(_at(clone, tmp_path, "19db3675^", "Creatures/Kirby/5 Stars/Butterfly.psd")[0])   # (OFS), (OD) in its body font
    assert not any(pe._restyle_due(psd, layer, None) for layer in pe.ability_layers(psd).layers)


@text_engine
def test_gate_fx_tells_a_bare_badge_under_a_tinted_ability(clone, tmp_path):
    """bare_icons, independent of the rebuild: Turtle's badge under its Color Overlay is in the overlay's colour until
    it takes Stylistic Alternates (owner 2026-10-25: one colour on purpose); gate_fx tells a new one, even for an edit
    that cut no FX layer. Blood Bomb's [machine] (Pin Light) was bare before the edit: it may stay so, not become so."""
    from psd_tools.composite.text import FontMap
    fonts_dir = str(clone / pe.FONTS_DIR)
    fonts = FontMap(dirs=[fonts_dir])
    turtle = _at(clone, tmp_path, BADGED, TURTLE)[0]
    old = PSDImage.open(turtle)
    assert pe.bare_icons(old, fonts) == Counter({"(1gcd)": 1})
    psd = PSDImage.open(turtle)
    layer = pe.ability_layer(psd)
    at = layer.text.index("(1GCD)")
    layer.set_span_style(at, at + 6, StylisticAlternates=True)
    assert pe.bare_icons(psd, fonts) == Counter()
    edit = SimpleNamespace(field="ability", fx_names=[])
    assert pe.gate_fx(psd, [edit], fonts_dir, None) == []
    assert pe.gate_fx(old, [edit], fonts_dir, None) == [
        "unsupported-card: ability: (1gcd): under the Ability's look (its Color Overlay), which changes their colours"]
    bomb = PSDImage.open(_card(clone, BLOOD_BOMB)[0])
    assert pe.bare_icons(bomb, fonts) == Counter({"[machine]": 1})
    assert pe.gate_fx(bomb, [edit], fonts_dir, None, PSDImage.open(_card(clone, BLOOD_BOMB)[0])) == []
    flagged(pe.gate_fx(bomb, [edit], fonts_dir), r"^unsupported-card: ability: \[machine\]: under")


@text_edit
def test_letters_a_deletion_leaves_in_the_icon_font_are_flagged(clone, parser, chart, tmp_path):
    """"(1GCD)" made "(GCD)" (Turtle) left "(GCD)" in the icon font, drawn as plain letters: the check looked only at
    typed characters. Any letter the icon font now draws plainly is told, unless it did before the edit. A "★" typed
    for the badge is told under the overlay (a colour icon; the badge layer goes, retired)."""
    turtle = _at(clone, tmp_path, BADGED, TURTLE)[0]
    psd = PSDImage.open(turtle)
    text = pe.normalize_ability(pe.ability_layer(psd).text)
    edit = pe.set_text_layer(psd, "ability", text.replace("(1GCD)", "(GCD)"), str(clone / pe.FONTS_DIR), chart, parser=parser)
    flagged(edit.notes, r"^unsupported-card: ability: '\(GCD\)' set in the icon font, which draws no badge or icon of it$")
    psd = PSDImage.open(turtle)
    edit = pe.set_text_layer(psd, "ability", text.replace("(1GCD)", "★"), str(clone / pe.FONTS_DIR), chart, parser=parser)
    flagged(edit.notes, r"^Ability FX: \(1GCD\) deleted: badge FX layers are retired$")
    flagged(edit.notes, r"^unsupported-card: ability: ★: the Ability's look \(its Color Overlay\) changes the typed icon's")


@text_engine
def test_gate_fx_tells_letters_in_the_icon_font(clone):
    """lettered_icons, independent of icon_text: "[fire:]" typed right after The Hive's (1GCD) (no overlay), in the badge's
    icon font, shapes as seven letters, no ligature: gate_fx tells it. The card as it is draws none, and its badge and
    token are one cluster each."""
    from psd_tools.composite.text import FontMap
    fonts_dir = str(clone / pe.FONTS_DIR)
    fonts = FontMap(dirs=[fonts_dir])
    old, psd = PSDImage.open(_card(clone, HIVE)[0]), PSDImage.open(_card(clone, HIVE)[0])
    assert pe.lettered_icons(old, fonts) == Counter()
    layer = pe.ability_layer(psd)
    text = layer.text.replace("(1GCD) Summon", "(1GCD)[fire:] Summon")
    pe.splice_text(layer, text, pe.edit_parts(layer.text, text, *pe.changed_range(layer.text, text)))
    layer.rasterize(fonts=fonts)
    assert pe.lettered_icons(psd, fonts) == Counter("[fire:]")
    assert pe.gate_fx(psd, [SimpleNamespace(field="ability", fx_names=[])], fonts_dir, None, old) == [
        "unsupported-card: ability: ':[]efir': the icon font draws them as letters, no badge or icon"]


@text_edit
def test_gate_fx_tells_a_token_not_shaped_as_one_icon(clone, parser):
    """unshaped_icons, the layout's check of icon_text: each badge or token it reads is one icon-font cluster. A "[fire]"
    typed in The Hive's body font (an edit's style there) is six letters: gate_fx tells it; its badge and token as they
    are pass."""
    from psd_tools.composite.text import FontMap
    fonts_dir = str(clone / pe.FONTS_DIR)
    fonts = FontMap(dirs=[fonts_dir])
    old, psd = PSDImage.open(_card(clone, HIVE)[0]), PSDImage.open(_card(clone, HIVE)[0])
    assert pe.unshaped_icons(old, fonts, parser.icons) == Counter()
    layer = pe.ability_layer(psd)
    text = layer.text.replace("onto any", "onto [fire] any")
    pe.splice_text(layer, text, pe.edit_parts(layer.text, text, *pe.changed_range(layer.text, text)))
    layer.rasterize(fonts=fonts)
    assert not any(pe._char_styles(layer)[1][i] for i in range(text.index("[fire]"), text.index(" any")))
    assert pe.unshaped_icons(psd, fonts, parser.icons) == Counter({"[fire]": 1})
    assert pe.gate_fx(psd, [SimpleNamespace(field="ability", fx_names=[])], fonts_dir, None, old) == []   # no vocabulary
    assert pe.gate_fx(psd, [SimpleNamespace(field="ability", fx_names=[])], fonts_dir, None, old, parser.icons) == [
        "unsupported-card: ability: [fire]: not drawn as one badge or icon"]


@text_edit
def test_new_text_under_a_layer_above_it_or_over_a_bar_is_flagged(clone, parser, chart):
    """Review 2026-10-24: text set where a layer above the Ability hides it, or over the stat bars, baked with every gate
    passing: Papyrus' spaghetti 'Layer 10' sits in the gap after "Spaghetti" ("Die." printed under it), Paint Warrior's
    blank Ability box reaches its Health bar. covered_ink tells it (overflow); text clear of both sets untold."""
    fonts_dir = str(clone / pe.FONTS_DIR)
    psd = PSDImage.open(_card(clone, PAPYRUS)[0])
    old = pe.normalize_ability(pe.ability_layer(psd).text)
    flagged(pe.set_text_layer(psd, "ability", old + " Die.", fonts_dir, chart, parser=parser).notes,
            r"^overflow: ability: 'Layer 10' above it would cover \d+ px of the new text$")
    psd = PSDImage.open(_card(clone, PAPYRUS)[0])
    assert pe.set_text_layer(psd, "ability", "Now " + old, fonts_dir, chart, parser=parser).notes == []
    psd = PSDImage.open(_card(clone, PAINT_WARRIOR)[0])
    flagged(pe.set_text_layer(psd, "ability", "Die.", fonts_dir, chart, parser=parser).notes,
            r"^overflow: ability: the new text runs over the hp bar \(\d+ px\)$")
    # titles under their art by design (review 2026-10-24): as much of the new title as of the old may sit under it
    for rel, card, title in ((LESHY, "Leshy", "Leshys"), (DOOPLISS, "DooPliss", "Dooplisss")):
        psd = PSDImage.open(_card(clone, rel)[0])
        assert not [n for n in pe.set_text_layer(psd, "name", title, fonts_dir, chart, card, parser=parser).notes
                    if "above it" in n]
    psd = PSDImage.open(_card(clone, LESHY)[0])
    flagged(pe.set_text_layer(psd, "name", "Lesh", fonts_dir, chart, "Leshy", parser=parser).notes,
            r"^overflow: name: 'Layer 6', 'Layer 7' above it, over the old text too, would cover \d+ px more of the new text$")


def test_icon_text_follows_the_fonts_spellings(parser):
    """What the stoogedings font draws (its GSUB): timing tags of badge_tags() in any case, tokens the vocabulary knows
    with a variant the font has; not "(2 points)", "[fire: no]" (no space by the variant) or "[foo]". canonical_tokens
    reads them as the parser does in the icon font, and classify calls "[Fire]" for "[fire]" redundant."""
    text = "(1cd)(Action)(2 points) [Fire:Sticker] [fire: no] [foo] [warp star] [fire:NO] (2/3GCD)"
    assert [text[a:b] for a, b in pe.icon_text(text, parser.icons)] == [
        "(1cd)", "(Action)", "[Fire:Sticker]", "[warp star]", "[fire:NO]", "(2/3GCD)"]
    assert [text[a:b] for a, b in pe.icon_text(text, None)][2:7] == ["[Fire:Sticker]", "[fire: no]", "[foo]", "[warp star]", "[fire:NO]"]
    assert pe.canonical_tokens(text, parser.icons) == "(1cd)(Action)(2 points) [fire] [fire: no] [foo] [warpstar] [fire:no] (2/3GCD)"
    parsed = {"card": {"ability": "Deal 1 [fire] damage."}}
    assert pe.classify(parsed, {"ability": "Deal 1 [Fire] damage."}, parser=parser).redundant == ["ability"]
    assert pe.classify(parsed, {"ability": "Deal 1 [Fire] damage."}).jobs == {"ability": "Deal 1 [Fire] damage."}
    assert len(pe.BADGE_TAGS) == 341 and pe.BADGE_KEYS["action"] == "Action"


def test_icon_style_takes_the_commonest_size_first(parser, monkeypatch):
    """Owner 2026-10-24: a typed badge takes the card's usual badge size. Three 32 pt badges each with its own baseline
    shift and two alike at 18 pt: 32 pt (the commonest whole style was the 18 pt one), then its commonest style."""
    text = "(1CD)(1CD)(1CD)(2U)(2U) Die."
    layer = SimpleNamespace(text=text)
    style = lambda size, shift: {"Font": pe.ICON_FONT, "FontSize": size, "BaselineShift": shift}
    looks = [style(32.0, 0.0)] * 5 + [style(32.0, 1.0)] * 5 + [style(32.0, 2.0)] * 5 + [style(18.0, 5.0)] * 8
    styles = looks + [{"Font": "Body", "FontSize": 32.0}] * (len(text) - len(looks))
    monkeypatch.setattr(pe, "_char_styles", lambda _: ([None] * len(text), [i < len(looks) for i in range(len(text))],
                                                       styles, [0] * len(text)))
    got = pe.icon_style(layer, True, parser.icons)
    assert (got["FontSize"], got["BaselineShift"]) == (32.0, 0.0)


PALACE = "Field/4 Stars/Palace_of_Shadow.psd"   # [dark:sticker] white on its white first line, [light:sticker] in grey


BENCH = "Field/1 Stars/Bench.psd"           # its title 'Bench', no effects, room for an icon after it
NEO = "Auxiliary/Creatures/Mettaton_NEO.psd"   # its title under a gradient overlay


@text_edit
def test_gate_fx_checks_a_title_edits_icons(clone, parser):
    """A title edit's icons take the Ability's icon checks (owner 2026-10-24, gate_fx's titles): "[fire]" typed in
    Bench's title in its own font is six letters (unshaped_icons), "[fire:]" in the icon font seven (lettered_icons),
    and under Mettaton NEO's gradient overlay an icon is bare (bare_icons); each is told. One set right passes."""
    from psd_tools.composite.text import FontMap
    fonts_dir = str(clone / pe.FONTS_DIR)
    fonts = FontMap(dirs=[fonts_dir])
    edits = [SimpleNamespace(field="name", fx_names=[])]
    for rel, title, icon, why in ((BENCH, "Bench [fire]", False, r"\[fire\]: not drawn as one badge or icon"),
                                  (BENCH, "Bench [fire:]", True, r"the icon font draws them as letters"),
                                  (NEO, "Mettaton NEO [fire]", True, r"\[fire\]: under the title's look \(its Gradient Overlay\)"),
                                  (BENCH, "Bench [fire]", True, None)):
        old, psd = PSDImage.open(_card(clone, rel)[0]), PSDImage.open(_card(clone, rel)[0])
        layer = pe.name_layer(psd, Path(rel).stem.replace("_", " "))
        layer.set_text(title, keep="runs")
        if icon:
            layer.add_font(pe.ICON_FONT)
            layer.set_span_style(title.index("["), len(title), Font=pe.ICON_FONT)
        layer.rasterize(fonts=fonts)
        titles = (layer, pe.name_layer(old, Path(rel).stem.replace("_", " ")))
        assert pe.gate_fx(psd, [SimpleNamespace(field="ability", fx_names=[])], fonts_dir, None, old, parser.icons) == []
        notes = pe.gate_fx(psd, edits, fonts_dir, None, old, parser.icons, titles)
        if why is None:
            assert notes == []
        else:
            flagged(notes, "^unsupported-card: name: .*" + why)


SALT = "Auxiliary/Items/Salt.psd"                               # a white title, no :sticker on the card
LESHY = "Creatures/Other/5 Stars/Leshy.psd"                     # its title under its art ('Layer 6', 'Layer 7')
DOOPLISS = "Creatures/Mario/4 Stars/DooPliss.psd"               # its title under its art ('p')


def test_gate_pairs_checks_each_pairings_filter_and_place(clone):
    """Review 2026-10-24: a copy whose smart object keeps its record but not its filter position, or its left/top,
    renders unlike the card: gate_pairs refuses it as it refuses a lost pairing."""
    old = PSDImage.open(_card(clone, LORIAN)[0])
    pe.gate_pairs(old, pe.edit_copy(old))
    for k in (1, 2, 3):
        new = pe.edit_copy(old)
        key, entry = next((key, entry) for key, entry in new._opened_records.items() if entry[1] is not None)
        new._opened_records[key] = tuple(value + 1 if i == k else value for i, value in enumerate(entry))
        with pytest.raises(pe.Refused, match=r"lost 1 smart object pairings") as refused:
            pe.gate_pairs(old, new)
        assert refused.value.code == "error"


@text_edit
def test_a_glow_disc_on_a_stat_bar_is_flagged(clone, parser, chart, monkeypatch):
    """A typed glow's group effects (a stroked group's disc, Photoshop's look) count on the stat bars: covered_ink takes
    their changed ink with the text's (1ed822c). One pixel of it in a Mini Doomer bar the old text missed is told, the
    text's own ink clear of it; with none there the edit sets untold."""
    fonts = str(clone / pe.FONTS_DIR)
    psd = PSDImage.open(_card(clone, MINI_DOOMER)[0])
    stat, rows = next((stat, rows) for stat, rows in stat_rows(psd).items() if rows and rows[0][2] is not None)
    x0, y0, x1, y1 = clip(rows[0][2], psd.width, psd.height)
    spot = ((y0 + y1) // 2, (x0 + x1) // 2)
    layer = pe.ability_layer(psd)
    canvas = pe._on_canvas(layer, (psd.height, psd.width))
    after, before = np.zeros_like(canvas), np.zeros_like(canvas)
    after[spot] = 255
    assert pe.covered_ink(psd, layer, canvas, []) is None and pe.covered_ink(psd, layer, canvas, [], (before, before)) is None
    assert pe.covered_ink(psd, layer, canvas, [], (after, before)) == f"the new text runs over the {stat} bar (1 px)"
    old = pe.normalize_ability(layer.text)
    for disc in (True, False):
        def halo(psd, layer):
            out = np.zeros((psd.height, psd.width, 4), np.uint8)
            if disc and "[fire:glow]" in layer.text:
                out[spot] = 255
            return out
        monkeypatch.setattr(pe, "_effect_ink", halo)
        psd = PSDImage.open(_card(clone, MINI_DOOMER)[0])
        notes = pe.set_text_layer(psd, "ability", old + " Gain [fire:glow].", fonts, chart, parser=parser).notes
        assert (f"overflow: ability: the new text runs over the {stat} bar (1 px)" in notes) == disc


@text_edit
def test_apply_jobs_checks_its_copy(clone, parser, chart, monkeypatch, tmp_path):
    """Review 2026-10-24: apply_jobs edits a copy, which a plain deepcopy makes without Lorian's smart filters. Its
    pairings are checked first (gate_pairs); a copy that renders unlike the card after them refuses too (gate_copy, on a
    card with smart filters)."""
    hp = pe.parse_card(parser, _card(clone, LORIAN)[0], LORIAN)["card"]["hp"]
    monkeypatch.setattr(pe, "edit_copy", deepcopy)
    with pytest.raises(pe.Refused, match="smart object pairings") as refused:
        _bake(clone, parser, LORIAN, {"hp": hp - 1}, tmp_path, chart)
    assert refused.value.code == "error"
    monkeypatch.setattr(pe, "gate_pairs", lambda psd, copy: None)
    with pytest.raises(pe.Refused, match="renders .* unlike the card itself") as refused:
        _bake(clone, parser, LORIAN, {"hp": hp - 1}, tmp_path, chart)
    assert refused.value.code == "error"


def _runs(layer) -> list:
    """(text, font, size, fill) per style run."""
    return [(layer.text[r.start:r.end], r.style.font_name, round(r.style.font_size, 2),
             tuple(round(v, 3) for v in r.style.fill_color[-3:])) for p in layer.typesetting for r in p.runs]


def _at(clone, tmp_path, rev: str, rel: str) -> tuple:
    """rel's PSD and PNG at rev, in tmp_path/at/."""
    out = tmp_path / "at"
    out.mkdir(exist_ok=True)
    paths = []
    for ext in (".psd", ".png"):
        name = rel[:len(rel) - 4] + ext
        (out / Path(name).name).write_bytes(at_rev(clone, rev, name))
        paths.append(str(out / Path(name).name))
    return tuple(paths)


@text_edit
def test_a_two_site_edit_bakes_its_runs_into_the_story(clone, parser, chart, tmp_path):
    """Through apply_jobs and every gate: the saved PSD's TySh and Txt2 story agree run by run (story_style_drift) and
    keep the badge between the edits."""
    paths = _at(clone, tmp_path, COLOURED, PIZZA)
    old = pe.parse_card(parser, paths[0], PIZZA)["card"]["ability"]
    new = old.replace("add 1 bar", "add 2 bar", 1).replace("one 2 star", "one 3 star", 1)
    _, baked = _bake(clone, parser, PIZZA, {"ability": new}, tmp_path, chart, paths=paths)
    saved = PSDImage.open(baked.psd_path)
    assert pe.story_style_drift(saved) == []
    assert pe.normalize_ability(baked.reparse["card"]["ability"]) == pe.normalize_ability(new)
    badges = [r[0] for r in _runs(pe.ability_layer(saved)) if r[1] == "stoogedings-Regular"]
    assert badges == ["(1GCD)", "(1CD)", "(1GCD)"]


@text_edit
def test_new_text_past_its_design_is_told(clone, parser, chart):
    """Review 2026-10-24: an Ability's added line printed over what lies below its old text (Deep Focus' flourish, Holy
    Card's torn panel border, Autopsy Report off its box) with no note; inks_over passed them, the old ink touching
    those layers elsewhere. It bakes with a note; a line added on the plain panel has none. A title's too."""
    fonts = str(clone / pe.FONTS_DIR)
    for rel, told in (("Items/Equip/3 Stars/Deep_Focus.psd", True), ("Items/Equip/2 Stars/Holy_Card.psd", True),
                      ("Auxiliary/Items/Autopsy_Report.psd", True), ("Auxiliary/Creatures/Mettaton_EX.psd", False)):
        psd_path = _card(clone, rel)[0]
        psd = PSDImage.open(psd_path)
        old = pe.normalize_ability(pe.parse_card(parser, psd_path, rel)["card"]["ability"])
        edit = pe.set_text_layer(psd, "ability", old + " Gain 1 HP. Die.", fonts, chart, parser=parser)
        notes = [note for note in edit.notes if note.startswith("ability: ")]
        assert bool(notes) == told, (rel, notes)
        assert all(note.endswith("; check the card") for note in notes)
    # a longer title past the banner it sits on (Bubble's parchment) or the texture clipped to it (Fire Totem's 'firefx':
    # the new letters print in the plain fill), review 2026-10-24
    for rel, card, why in (("N.M.E/Bubble.psd", "Bubble", r"onto the design below it \(\d+ px, 'space'\)$"),
                           ("Auxiliary/Items/Fire_Totem.psd", "Fire Totem", r"runs past 'firefx' clipped to it \(\d+ px\)$")):
        psd = PSDImage.open(_card(clone, rel)[0])
        edit = pe.set_text_layer(psd, "name", pe.read_name(psd, card) + " II", fonts, chart, card, parser=parser)
        flagged(edit.notes, "^overflow: name: the new text .*" + why)


@text_edit
@pytest.mark.skipif(not (_text_engine() and pe.tab_stops()), reason="the fork lays no tab stops")
def test_tab_stops_bake(clone, parser, chart):
    """Owner 2026-10-24 ("tab stops should be supported"): the fork lays tabs to Photoshop's stops. The website shows
    Michael's Blessing's tabs as spaces: they stay tabs, and an edited tabbed line keeps its '->' column (it collapsed,
    review 2026-10-24)."""
    rel = "N.M.E/Michael's_Blessing.psd"
    psd_path = _card(clone, rel)[0]
    psd = PSDImage.open(psd_path)
    layer = pe.ability_layer(psd)
    old, before = layer.text, _ink(layer, psd)
    text = pe.normalize_ability(pe.parse_card(parser, psd_path, rel)["card"]["ability"])
    assert "\t" in old and "\t" not in text
    pe.set_text_layer(psd, "ability", text.replace("-> (1GCD)", "-> (2GCD)", 1), str(clone / pe.FONTS_DIR), chart,
                      parser=parser)
    assert layer.text == old.replace("(1GCD)", "(2GCD)", 1)
    ys, xs = np.nonzero(_ink(layer, psd) != before)
    assert len(xs) and xs.min() > 760 and 1150 < ys.min() <= ys.max() < 1250     # the OYT row's "2" only


@text_edit
def test_styled_status_word_gets_an_fx_layer(clone, chart, tmp_path):
    """Idea 14 in memory on Mini_Doomer: Blazing is styled on the chart, so the edit cuts "Ability FX: Blazing" from the new
    raster (the ability's own pixels under the word), directly above Text, carrying the chart layer's lfx2/lrFX bytes and a
    fresh layer id; the word keeps the swatch. The next edit makes it again as Effect Layers.jsx's Update all does (BUG 2):
    in place, a repeat cut into the same layer, and a removed word leaves nothing behind."""
    assert sorted(c.name for c in chart.values() if c.styled) == STYLED
    fonts = str(clone / pe.FONTS_DIR)
    psd = PSDImage.open(_plain(clone, tmp_path)[0])
    count = len(list(psd.descendants()))
    white, red = (1.0, 1.0, 1.0), (1.0, 0.0, 0.0)
    edit = pe.set_text_layer(psd, "ability", "Inflict Blazing.", fonts, chart)
    layer, text = pe.ability_layer(psd), pe._text_group(psd)
    assert _fills(layer)[1:] == [("Blazing", red), (".", white)]
    one, = pe.fx_layers(psd, layer)
    assert one.name == "Ability FX: Blazing" and edit.fx[0][0] == one.name and (edit.dropped, edit.hidden) == ([], [])
    assert one.parent is psd and psd.index(one) == psd.index(text) + 1 and one.kind == "pixel" and one.is_visible()
    assert pe._enabled_effects(one) == ["OuterGlow"] and one.layer_id > 0
    assert one.layer_id not in {other.layer_id for other in psd.descendants() if other is not one}
    assert edit.fx[0][1] == {bytes(k): b.tobytes() for k, b in chart["blazing"].blocks.items()} and set(edit.fx[0][1]) == {b"lfx2", b"lrFX"}
    assert {bytes(k): one.tagged_blocks[k].tobytes() for k in pe.EFFECT_BLOCKS if k in one.tagged_blocks} == edit.fx[0][1]
    x0, y0, x1, y1 = one.bbox
    ax0, ay0, ax1, ay1 = layer.bbox
    assert ax0 <= x0 < x1 <= ax1 and ay0 <= y0 < y1 <= ay1 and pe.PSDParser._pixel_coverage(one, layer) == 1.0
    cut, under = one.numpy("shape")[..., 0], layer.numpy("shape")[y0 - ay0:y1 - ay0, x0 - ax0:x1 - ax0, 0]
    assert np.array_equal(cut[cut > 0], under[cut > 0])                        # copied from the raster, not re-rendered
    assert edit.region[0] <= x0 - GLOW and edit.region[2] >= x1 + GLOW and edit.region[3] >= y1 + GLOW
    ink = pe.PSDParser._pixel_coverage(one, layer), int((one.numpy("shape") > 0.5).sum())
    edit = pe.set_text_layer(psd, "ability", "Inflict Blazing. Blazing again.", fonts, chart)
    again, = pe.fx_layers(psd, pe.ability_layer(psd))
    assert again is one and again.name == "Ability FX: Blazing" and (edit.fx, edit.dropped) == ([], [])
    assert [name for _, _, name in edit.rebuilt] == edit.fx_names == ["Ability FX: Blazing"] and psd.index(one) == psd.index(text) + 1
    assert int((one.numpy("shape") > 0.5).sum()) > 1.8 * ink[1] and pe.PSDParser._pixel_coverage(one, pe.ability_layer(psd)) == 1.0
    edit = pe.set_text_layer(psd, "ability", "Inflict Poison.", fonts, chart)
    assert pe.fx_layers(psd, pe.ability_layer(psd)) == [] and edit.fx == [] and edit.dropped == ["Ability FX: Blazing"]
    assert len(list(psd.descendants())) == count


PALACE = "Field/4 Stars/Palace_of_Shadow.psd"   # 'Ability FX: +2 DEF': a Stroke alone over its words' own blue


@text_edit
def test_gate_fx_refuses_an_own_style_layer_over_a_new_colour(clone, chart):
    """gate_fx_colours, independent of set_text_layer's fills: Palace of Shadow's stroke-only '+2 DEF' layer over its
    words in the body's grey (what a typed copy got) refuses reparse, from gate_fx too; the card as it is passes."""
    path = _card(clone, PALACE)[0]
    old, psd = PSDImage.open(path), PSDImage.open(path)
    pe.gate_fx_colours(psd, old, chart)
    layer = pe.ability_layer(psd)
    start = layer.text.index("+2 DEF")
    layer.set_span_style(start + 3, start + 6, fill=(0.2, 0.2, 0.2))
    why = r"^reparse: 'Ability FX: \+2 DEF' covers '\+2 DEF' in a colour its words did not have$"
    with pytest.raises(pe.Refused, match=why):
        pe.gate_fx_colours(psd, old, chart)
    with pytest.raises(pe.Refused, match=why):
        pe.gate_fx(psd, [SimpleNamespace(field="ability", fx_names=[])], str(clone / pe.FONTS_DIR), chart, old)


def test_the_bakes_vocabulary_knows_only_the_fonts_icons():
    """A Types/ icon the font was not rebuilt with ("newtype": in Types/, not in the manifest) is no icon to the bake:
    set in the icon font, its [token] would print the font's letters and read back as the token (review 2026-10-23). It
    stays a plain [token]; without a manifest every Types/ name is one, as before."""
    vocab = json.loads(VOCAB.read_text(encoding="utf-8"))
    text = "Deal [fire] and [newtype] damage."
    fonted = pe.offline_parser(vocab["all_types"] + ["newtype"], vocab["stars"], vocab["manifest"])
    assert [text[a:b] for a, b in pe.icon_text(text, fonted.icons)] == ["[fire]"] and fonted.icons.lookup("newtype") is None
    bare = pe.offline_parser(vocab["all_types"] + ["newtype"], vocab["stars"], None)
    assert [text[a:b] for a, b in pe.icon_text(text, bare.icons)] == ["[fire]", "[newtype]"]


@text_edit
def test_ability_bake_regenerates_fx_layers_through_every_gate(clone, parser, chart, tmp_path):
    """Idea 14 through apply_jobs on Mini_Doomer: adding Blazing and Frozen cuts two FX layers (2 x 5916 block bytes landed),
    within the parity bar, the validator clean, nothing outside the region."""
    psd_path, png_path = _plain(clone, tmp_path)
    old = pe.parse_card(parser, psd_path, MINI_DOOMER)["card"]["ability"]
    one = tmp_path / "one"
    one.mkdir()
    parsed, baked = _bake(clone, parser, MINI_DOOMER, {"ability": old + " Inflict Blazing and Frozen."}, one, chart,
                          paths=(psd_path, png_path))
    assert [(name, fields) for _, name, fields in baked.records] == [("Ability", TEXT_RECORD), ("Ability FX: Blazing", ["added"]),
                                                                     ("Ability FX: Frozen", ["added"])]
    assert baked.fx == ["Ability FX: Blazing", "Ability FX: Frozen"] and (baked.flag_bytes, baked.block_bytes) == (0, 2 * 5916)
    assert baked.reparse["validator"] == [] and pe._within_bar(baked.parity["ability"])
    saved = PSDImage.open(baked.psd_path)
    layer = pe.ability_layer(saved)
    for fx in pe.fx_layers(saved, layer):
        assert pe.PSDParser._pixel_coverage(fx, layer) == 1.0 and pe._enabled_effects(fx) == ["OuterGlow"]
        x0, y0, x1, y1 = fx.bbox
        assert baked.region[0] <= x0 - GLOW and baked.region[2] >= x1 + GLOW and baked.region[3] >= y1 + GLOW
    before = np.asarray(Image.open(png_path).convert("RGBA"))
    after = np.asarray(Image.open(baked.png_path).convert("RGBA"))
    differs = (before != after).any(axis=2)
    inside = np.zeros_like(differs)
    inside[baked.region[1]:baked.region[3], baked.region[0]:baked.region[2]] = True
    assert int((differs & ~inside).sum()) == 0 and 0 < int(differs.sum()) <= baked.changed_pixels   # a replaced pixel may equal the PNG's
    assert pe.classify(baked.reparse, {"ability": old + " Inflict Blazing and Frozen."}, saved).redundant == ["ability"]


@text_edit
def test_missing_and_loose_fonts(clone, parser, tmp_path, monkeypatch):
    """A face no file holds is drawn in the fork's stand-in (owner 2026-10-24: fonts flagged, not refused), whatever this
    machine has installed: Anubisath's title face found only loosely bakes, told; Termination Shock's OCR A Extended made
    to miss redraws the whole title in the stand-in, past the parity floor: refused (a gross render error)."""
    from psd_tools.composite.text import FontMap
    lookup = FontMap._lookup
    monkeypatch.setattr(FontMap, "_lookup", lambda self, name: (lambda hit: None if hit is None else (hit[0], "loose"))(lookup(self, name)))
    psd_path = _card(clone, ANUBISATH)[0]
    parsed, baked = _bake(clone, parser, ANUBISATH, {"name": _typo_fix(pe.read_name(PSDImage.open(psd_path)))}, tmp_path)
    flagged(baked.fx_notes, "^font: name: not resolved exactly, drawn in a stand-in: ")
    monkeypatch.setattr(FontMap, "_lookup", lambda self, name: None if name == "OCRAExtended" else lookup(self, name))
    rel = "N.M.E/Termination_Shock.psd"
    with pytest.raises(pe.Refused) as refused:
        _bake(clone, parser, rel, {"name": pe.read_name(PSDImage.open(_card(clone, rel)[0])) + " II"}, tmp_path)
    assert refused.value.code == "parity" and "past the 25% floor" in refused.value.why


def _painted(clone, tmp_path, rows: float, box: tuple = HP_ROW) -> tuple:
    """Mini Doomer's PSD, and its PNG with the top share of a box (the hp row) inverted (a render broken there), in
    tmp_path/card."""
    psd, png = _card(clone, MINI_DOOMER)
    out = tmp_path / "card"
    out.mkdir()
    (out / Path(png).name).write_bytes(painted(Path(png).read_bytes(), box, rows))
    (out / Path(psd).name).write_bytes(Path(psd).read_bytes())
    return str(out / Path(psd).name), str(out / Path(png).name)


@text_engine
def test_a_broken_render_is_refused_past_the_floor(clone, parser, tmp_path):
    """The safety floor (owner, 2026-10-23): more than 25% of the edited box's pixels past 24 is a broken render, refused
    `parity` before anything is written; a box a tenth broken bakes, told; the re-render is held to the same floor."""
    with pytest.raises(pe.Refused) as refused:
        _bake(clone, parser, MINI_DOOMER, {"hp": 8}, tmp_path, paths=_painted(clone, tmp_path, 1.0))
    assert refused.value.code == "parity" and refused.value.why.endswith("past the 25% floor")
    assert float(search(r"bad ([\d.]+)%", refused.value.why).group(1)) > pe.PARITY_FLOOR
    assert [p.name for p in tmp_path.iterdir()] == ["card"]
    out = tmp_path / "out"
    out.mkdir()
    _, baked = _bake(clone, parser, MINI_DOOMER, {"hp": 8}, out, paths=_painted(clone, out, 0.1))
    assert 5 < baked.parity["hp"]["bad_pct"] < pe.PARITY_FLOOR and baked.reparse["card"]["hp"] == 8
    assert pe.miss_notes(baked)[0].startswith("hp baked with a pixel difference: bad ")
    merged = np.array(Image.open(baked.png_path))
    x0, y0, x1, y1 = HP_ROW
    merged[y0:y1, x0:x1, :3] = 255 - merged[y0:y1, x0:x1, :3]
    with pytest.raises(pe.Refused) as refused:
        pe.gate_idempotence(baked.reparse, {"hp": 8}, PSDImage.open(baked.psd_path), Image.fromarray(merged), baked.edits,
                            baked.icc)
    assert refused.value.code == "idempotence" and refused.value.why.endswith("past the 25% floor")


SHIELD = "Items/Equip/3 Stars/Shield_of_the_Guilty.psd"   # its ability box drawn without text: 10.7% bad, under the floor


def _no_ability(monkeypatch):
    """The fork drawing no ability text: every render hides the visible Text/Ability layer (the baseline's too: no cut
    from the whole render, crop_pair)."""
    monkeypatch.setattr(pe, "crop_pair", lambda *args, **kwargs: None)
    for name in ("render_region", "render_pair"):
        def blank(psd, region, icc=True, text="preview", edited=(), real=getattr(pe, name)):
            layer = pe.ability_layer(psd)
            layer.visible = False
            try:
                return real(psd, region, icc, text, edited)
            finally:
                layer.visible = True
        monkeypatch.setattr(pe, name, blank)


@text_edit
def test_a_blank_render_under_the_floor_is_refused(clone, parser, tmp_path, monkeypatch):
    """The fork drawing no ability text misses only 10.7% of Shield of the Guilty's box, under the floor, and its merge
    replaces no pixel: the PSD would carry the new ability, the PNG the old. Refused `parity` before anything is written
    (review, 2026-10-23)."""
    psd_path, png_path = _card(clone, SHIELD)
    _no_ability(monkeypatch)
    old = pe.parse_card(parser, psd_path, SHIELD)["card"]["ability"]
    new = old[:len(old) - 1] + ("!" if old[len(old) - 1] != "!" else ".")
    with pytest.raises(pe.Refused) as refused:
        _bake(clone, parser, SHIELD, {"ability": new}, tmp_path)
    assert refused.value.code == "parity" and refused.value.why.startswith("ability: the render shows no change, baseline bad ")
    assert 5 < float(search(r"bad ([\d.]+)%", refused.value.why).group(1)) < pe.PARITY_FLOOR
    assert not list(tmp_path.iterdir())


@text_engine
def test_baseline_falls_back_when_the_png_disagrees_with_the_preview(clone, tmp_path):
    """Shell_Shock's Ability before its re-save (MISMATCHED_SHOCK): its preview shows the edge straight, its PNG (another
    session's) in the text gamma. The preview's blend misses the bar and baseline() takes the gamma's instead. The live card
    (re-saved, its PNG re-exported) agrees with its preview: baseline() keeps the preview's."""
    for (psd_path, png_path), text in ((_at(clone, tmp_path, MISMATCHED_SHOCK, SHELL_SHOCK), "gamma"),
                                       (_card(clone, SHELL_SHOCK), "preview")):
        psd, png = PSDImage.open(psd_path), Image.open(png_path).convert("RGBA")
        layer = pe.ability_layer(psd)
        region = pe._pad(tuple(layer.bbox), fx_pad(layer), psd)
        icc = pe.export_icc(psd, png)
        assert pe._within_bar(pe._measure(pe.render_region(psd, region, icc), png.crop(region))) == (text == "preview")
        render, measured = pe.baseline(psd, png, region, icc)
        assert measured["text"] == text and pe._within_bar(measured)
        assert np.array_equal(np.asarray(render), np.asarray(pe.render_region(psd, region, icc, text)))


# ========================
# Txt2 stories (the copy of the text Photoshop reads)
# ========================
@text_edit
@pytest.mark.parametrize("rel", NAMED)
def test_story_reader_agrees_with_the_fork(clone, rel):
    """psd_edit reads Txt2 itself (the gates check the fork): its story per layer is the fork's story_text."""
    psd = PSDImage.open(_card(clone, rel)[0])
    texts = pe.stories(psd)
    layers = [layer for layer in psd.descendants() if layer.kind == "type"]
    assert layers and [texts[pe.story_index(layer)] for layer in layers] == [layer.story_text for layer in layers]
    assert pe.stale_stories(psd) == []


@text_edit
def test_gate_stories_refuses_a_tysh_only_edit(clone, tmp_path):
    """The 2026-10-22 flaw: the title changed in TySh, its story did not; Photoshop would show the old title."""
    title = pe.read_name(PSDImage.open(_card(clone, MINI_DOOMER)[0]))
    path, index = stale_copy(clone, MINI_DOOMER, _typo_fix(title), tmp_path)
    stale = PSDImage.open(path)
    assert pe.stale_stories(stale) == [("Name", title + "\r", _typo_fix(title) + "\r")]
    with pytest.raises(pe.Refused, match="Name's Txt2 story holds") as refused:
        pe.gate_stories(PSDImage.open(_card(clone, MINI_DOOMER)[0]), stale, {index})
    assert refused.value.code == "bytes"


@text_edit
def test_gate_stories_refuses_changes_beyond_the_edit(clone):
    """Another layer's story, Txt2 outside the stories, or another document block: refused."""
    path = _card(clone, MINI_DOOMER)[0]
    old, new = PSDImage.open(path), PSDImage.open(path)
    assert pe.gate_stories(old, new, set()) == []
    ability = pe.story_index(pe.ability_layer(new))
    features = pe._story_list(pe._txt2(new))[2][ability]["0"]["6"]["0"][0]["0"]["0"]["6"]
    features["1"] = engine_data.Float(99.0)                     # the ability's font size, in its story alone
    with pytest.raises(pe.Refused, match=rf"stories \[{ability}\] changed: no edit there"):
        pe.gate_stories(old, new, {pe.story_index(pe.name_layer(new))})
    with pytest.raises(pe.Refused, match="Ability's Txt2 story has other styles: FontSize at 0-72: layer 50.0, story 99.0"):
        pe.gate_stories(old, new, {ability})                    # the layer's own size disagrees: Photoshop shows 99
    pe.ability_layer(new).engine_dict["StyleRun"]["RunArray"][0]["StyleSheet"]["StyleSheetData"]["FontSize"] = \
        engine_data.Float(99.0)
    assert pe.gate_stories(old, new, {ability}) == [ability]
    new = PSDImage.open(path)
    pe._txt2(new)["1"]["0"]["99"] = engine_data.Integer(1)      # DocumentSettings gains a key
    with pytest.raises(pe.Refused, match="outside its stories"):
        pe.gate_stories(old, new, set())
    new = PSDImage.open(path)
    blocks = new._record.layer_and_mask_information.tagged_blocks
    del blocks[next(key for key in blocks.keys() if key != Tag.TEXT_ENGINE_DATA)]
    with pytest.raises(pe.Refused, match="document blocks changed"):
        pe.gate_stories(old, new, set())


@text_edit
def test_icon_font_tokens_and_colour_glyphs(clone, chart):
    """The_Hive is set in stoogedings: a typed [fire] takes the icon font in TySh and the story. Its glyphs are SVG colour
    glyphs and no overlay covers the ability (Mini Doomer's gradient covers its badges): a fork that draws them in colour
    takes the edit, one that draws them in one colour (they would come out white) refuses it."""
    psd = PSDImage.open(_card(clone, HIVE)[0])
    layer = pe.ability_layer(psd)
    assert pe.uses_icon_font(layer) and pe.icon_font(layer) == "stoogedings-Regular"
    assert not pe.uses_icon_font(pe.name_layer(psd)) and not pe._overlaid(layer)
    assert pe._overlaid(pe.ability_layer(PSDImage.open(_card(clone, MINI_DOOMER)[0])))
    text = pe.normalize_ability(layer.text) + " Deal 1 [fire] damage."
    if colour_glyphs():
        assert pe.set_text_layer(psd, "ability", text, str(clone / pe.FONTS_DIR), chart).layers == [layer]
    else:
        with pytest.raises(pe.Refused, match="colour glyphs") as refused:
            pe.set_text_layer(psd, "ability", text, str(clone / pe.FONTS_DIR), chart)
        assert refused.value.code == "font"
    start = layer.text.index("[fire]")
    fonts = [str(font["Name"].value) for font in layer.resource_dict["FontSet"]]
    style, unit = layer.engine_dict["StyleRun"], 0
    for n, run in zip(style["RunLengthArray"], style["RunArray"]):
        if unit <= start < unit + int(n):
            assert fonts[int(run["StyleSheet"]["StyleSheetData"]["Font"])] == "stoogedings-Regular"
            assert unit + int(n) == start + len("[fire]")
        unit += int(n)
    assert pe.stale_stories(psd) == [] and pe.story_style_drift(psd) == []     # the story's font is the same


RED_KNIFE = "Auxiliary/Items/Red_Knife.psd"     # its Ability near the card's foot: a halo's faint tail passes it
SALT = "Auxiliary/Items/Salt.psd"               # its title in a stroked Text group, 'Layer 19' (MISS) below beside it


SWAPPED_FONTS = {"Items/Equip/3 Stars/Shiny_Knife.psd": "OMORI_GAME2",    # its title's pixels in the other Omori font
                 "Auxiliary/Minions/Dark_Tentacle.psd": "OMORI_GAME"}


@text_edit
def test_a_title_drawn_in_another_font_is_flagged(clone, parser, chart):
    """Shiny Knife's title names OMORI_GAME2, Dark Tentacle's OMORI_GAME, but Photoshop drew each in the other: an edit
    reprints every letter in the other typeface (review 2026-10-24). drawn_in_font tells it; another Omori title sets
    untold."""
    fonts = str(clone / pe.FONTS_DIR)
    for rel, font in SWAPPED_FONTS.items():
        psd = PSDImage.open(_card(clone, rel)[0])
        card = Path(rel).stem.replace("_", " ")
        edit = pe.set_text_layer(psd, "name", pe.name_layer(psd, card).text + "s", fonts, chart, card, parser=parser)
        flagged(edit.notes, rf"^font: name: its saved pixels are not drawn in the font it names \({font}; \d+% alike\)$")
    psd = PSDImage.open(_card(clone, RED_KNIFE)[0])     # OMORI_GAME2, drawn in it
    assert not [n for n in pe.set_text_layer(psd, "name", "Red Knifes", fonts, chart, "Red Knife", parser=parser).notes
                if n.startswith("font:")]


@text_edit
def test_bake_repairs_a_stale_story(clone, parser, tmp_path):
    """A pre-fix bake's PSD has its stale Txt2 stories synced (commands.txt2_repair.sync_stories), told, whatever the
    job; gate_stories lets the repaired stories change."""
    title = pe.read_name(PSDImage.open(_card(clone, MINI_DOOMER)[0]))
    path, _ = stale_copy(clone, MINI_DOOMER, _typo_fix(title), tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    parsed = pe.parse_card(parser, path, MINI_DOOMER)
    baked = pe.apply_jobs(path, _card(clone, MINI_DOOMER)[1], MINI_DOOMER, {"hp": 8}, parsed, parser, str(out),
                          str(clone / pe.FONTS_DIR))
    flagged(baked.fx_notes, r"^Txt2 story repaired \(commands.txt2_repair\): ")
    assert pe.stale_stories(PSDImage.open(baked.psd_path)) == []


# ========================
# An ability over several layers (owner 2026-10-24)
# ========================
LIT_BOB = "Auxiliary/Items/Lit_Bob-omb.psd"                        # 'Costs 0 SP to use.' / 'Deal 7 DMG.': two Ability layers
MIRACLE = "Creatures/Kirby/3 Stars/Miracle_Matter.psd"            # five, padded ('\r    Fire: 5/5/8/7'), 'Ability FX: (1GCD)'


def _dir(tmp_path, name: str):
    (tmp_path / name).mkdir()
    return tmp_path / name


def test_ability_split_shares_an_edit_out():
    """Each changed span goes to the layer it lies in (padding kept); a line added at a boundary to the layer above; a
    removed break joins two layers in the one that keeps most; a part removed whole leaves its layer empty; an image
    icon's [type] token the parser adds stays out."""
    raws = ["Costs 0 SP to use.", "\r    Deal 7 DMG.\r   "]
    old = "Costs 0 SP to use.\nDeal 7 DMG."
    split = pe.ability_split(old, "Costs 0 SP to use.\nDeal 9 DMG.", raws)
    assert [(m, r, core, new) for m, r, core, new, _ in split] == [
        ([0], 0, "Costs 0 SP to use.", "Costs 0 SP to use."), ([1], 1, "Deal 7 DMG.", "Deal 9 DMG.")]
    assert pe.project("Deal 7 DMG.", "Deal 9 DMG.", raws[1]) == "\r    Deal 9 DMG.\r   "
    added = pe.ability_split(old, "Costs 0 SP to use.\nDiscard after use.\nDeal 7 DMG.", raws)
    assert [new for *_, new, _ in added] == ["Costs 0 SP to use.\nDiscard after use.", "Deal 7 DMG."]
    joined, = pe.ability_split(old, "Costs 0 SP to use. Deal 7 DMG.", raws)
    assert joined == ([0, 1], 0, "Costs 0 SP to use.", "Costs 0 SP to use. Deal 7 DMG.", [1])
    dropped, = pe.ability_split(old, "Deal 7 DMG.", raws)
    assert dropped == ([0, 1], 1, "Deal 7 DMG.", "Deal 7 DMG.", [])
    tokened = pe.ability_split(old + " [undead]", "Costs 1 SP to use.\nDeal 7 DMG. [undead]", raws)
    assert [new for *_, new, _ in tokened] == ["Costs 1 SP to use.", "Deal 7 DMG."]
    one_line = pe.ability_split(old, "Costs 0 SP. Deal 7 DMG to all.", raws)     # no cut at a space (review)
    assert [members for members, *_ in one_line] == [[0, 1]]
    relaid = pe.ability_split("Costs 0 SP to use. Deal 7 DMG.", "Costs 1 SP to use. Deal 7 DMG.", raws, relaid=True)
    assert [new for *_, new, _ in relaid] == ["Costs 1 SP to use.", "Deal 7 DMG."]
    assert pe.unpadded("Fire: 5/5/8/7\n    \nEarth   \n") == "Fire: 5/5/8/7\nEarth"


@text_edit
def test_an_ability_over_several_layers_bakes(clone, parser, chart, tmp_path):
    """Miracle Matter: an edit across its Fire and Earth layers changes those two records and deletes the retired
    'Ability FX: (1GCD)' (owner 2026-10-25), its other three layers byte for byte; on Lit Bob-omb a removed break joins its two layers in the first, the
    second hidden, told."""
    paths = _at(clone, tmp_path, BADGED, MIRACLE)
    old = pe.parse_card(parser, paths[0], MIRACLE)["card"]["ability"]
    new = old.replace("Fire: 5/5/8/7", "Fire: 5/5/9/7").replace("Earth: 5/8/6/5", "Earth: 5/9/6/5")
    _, baked = _bake(clone, parser, MIRACLE, {"ability": new}, _dir(tmp_path, "mm"), chart, paths=paths)
    texts = 3 if salt_font(clone) else 2        # the salt font redraws the first layer's (1GCD) as an outlined pill
    assert [name for _, name, _ in baked.records] == ["Ability"] * texts + ["Ability FX: (1GCD)"]
    assert all(set(fields) <= set(TEXT_RECORD) for _, _, fields in baked.records[:texts])
    assert baked.records[texts][2] == ["removed"]
    assert baked.reparse["card"]["ability"] == new
    assert baked.fx_notes == ["Ability FX: (1GCD) deleted: badge FX layers are retired"]
    saved = [layer.text for layer in pe.ability_layers(PSDImage.open(baked.psd_path)).layers]
    assert saved[1:3] == ["\r    Fire: 5/5/9/7", "\r    \r    Earth: 5/9/6/5 and heal 2 HP\r   "]
    _, baked = _bake(clone, parser, LIT_BOB, {"ability": "Costs 0 SP to use. Deal 7 DMG."}, _dir(tmp_path, "bob"), chart)
    flagged(baked.fx_notes, r"^ability: the break between the first and second ability layers was removed: their "
                            r"words are joined in the first and the second hidden")
    assert sorted(fields == ["flags"] or set(fields) <= set(TEXT_RECORD) for _, _, fields in baked.records) == [True, True]
    assert ["flags"] in [fields for _, _, fields in baked.records]
    assert baked.reparse["card"]["ability"] == "Costs 0 SP to use. Deal 7 DMG."

