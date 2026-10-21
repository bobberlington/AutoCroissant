"""
psd_edit and psd_regions: the edit pipeline's stat toggle, text edits, delta merge and gates (EDIT_PIPELINE_DESIGN
sections 5, 6, 10; steps S2 and S3). The fixture is the real card repo (MDW/Templates/Creature_Template.psd is 30 MB,
too big to ship): tests that need a PSD read it from the clone and skip without it; tests that render or edit text
skip without the fork's text engine (psd_tools.composite.text), so the suite stays green under stock psd-tools.
    PYTHONPATH=/tmp/wt-pipeline-dev/src ~/Documents/Tools/psd-tools/.venv/bin/python -m pytest -q tests/test_psd_edit.py
"""
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from psd_tools import PSDImage

import commands.psd_edit as pe
from commands.psd_regions import PAINT_ORDER, badge_layers, clip, regions, stat_rows

# ========================
# Configuration
# ========================
VOCAB = Path(__file__).resolve().parent / "goldens" / "_vocab.json"
MINI_DOOMER = "Auxiliary/Minions/Mini_Doomer.psd"                 # template rows, no digit effects (certified golden)
HP_ROW = (360, 1200, 1173, 1284)                                   # its hp row: the digits' mask union
METTATON = "Auxiliary/Creatures/Mettaton_EX.psd"                  # a GradientOverlay on one digit per row
SPD_ROW = (359, 1577, 1174, 1663)                                  # its spd row, padded by the Stats Bars stroke
ANUBISATH = "Auxiliary/Minions/Anubisath_Guardian.psd"            # left-aligned title 11 px short of the first star
ASTR = "Auxiliary/Minions/Astr_Mionion.psd"                       # "Attacks inflict 1T3 Empowered." with Empowered in red
NAMED = (MINI_DOOMER, "Creatures/World Of Warcraft/5 Stars/The_Lich_King.psd", ANUBISATH,
         "Items/Support/2 Stars/Volt_Shroom.psd")                  # Text/Name + one visible Text/Ability, fonts in MDW/Fonts
TEXT_RECORD = ["blocks", "channels", "rect"]                       # what a text edit may change on its layer record


def _typo_fix(title: str) -> str:
    """A rename of the same width: the last letter swapped (the live edits fix typos, they do not lengthen titles)."""
    return title[:len(title) - 1] + ("e" if title[len(title) - 1] != "e" else "a")


def _fills(layer) -> list:
    return [(layer.text[s:e], tuple(round(v, 3) for v in f)) for s, e, f in pe.run_fills(layer)]


def _text_engine() -> bool:
    try:
        import psd_tools.composite.text  # noqa: F401
    except ImportError:
        return False
    return True


text_engine = pytest.mark.skipif(not _text_engine(), reason="no text engine: run under the fork (psd-tools[text])")


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


def _card(clone, rel: str) -> tuple:
    path = clone / rel
    if not path.exists() or not Path(str(path)[:-4] + ".png").exists():
        pytest.skip(f"{rel} not in the clone")
    return str(path), str(path)[:-4] + ".png"


def _bake(clone, parser, rel, jobs, tmp_path, chart=None, update_preview=False):
    psd, png = _card(clone, rel)
    parsed = pe.parse_card(parser, psd, rel)
    assert pe.classify(parsed, jobs, PSDImage.open(psd)).jobs == jobs
    return parsed, pe.apply_jobs(psd, png, rel, jobs, parsed, parser, str(tmp_path), str(clone / pe.FONTS_DIR), chart,
                                 update_preview=update_preview)


# ========================
# Pure functions (any psd-tools)
# ========================
def test_codes_are_the_closed_list():
    assert pe.CODES == ("duplicate", "unsupported-card", "stat>10", "recent", "blob", "parity", "styled-word", "font",
                        "overflow", "bytes", "size", "region", "shape", "reparse", "idempotence", "edit-changed",
                        "head-moved", "error")
    assert PAINT_ORDER[-1] == "ability_fx" and clip((-5, -5, 2000, 10), 1200, 2400) == (0, 0, 1200, 10)


def test_normalise_ability_matches_the_parser():
    layer = "Has stats -2 .\x03(BOT) Die .  \r"
    site = "Has stats -2.\n(BOT) Die."
    assert pe.normalise_ability(layer) == pe.normalise_ability(site) == "Has stats -2.\n(BOT) Die."
    assert pe.normalise_ability("﻿\"Gain\t1 \nhp .\"\n") == "Gain 1 \nhp." and pe.normalise_ability("") == ""
    assert pe.collapse(" Shooting\rStar ") == "Shooting Star" and pe.collapse(None) == ""


def test_changed_range_is_the_engines():
    assert pe.changed_range("Inflict Poison.", "Inflict Poison for 2 turns.") == (14, 26)
    assert pe.changed_range("Inflict Poison.", "Inflict.") == (7, 7)                 # removal only
    assert pe.changed_range("Inflict Poison.", "Inflict Poisoned.") == (14, 16)
    assert pe._word_bounds("Inflict Poisoned.", 14, 16) == (8, 16)
    assert pe.changed_range("Die.", "Die.") == (5, 5) and pe.changed_range("", "Die.") == (0, 4)


def test_status_spans_longest_first_whole_words():
    chart = {"frozen": pe.StatusColour("Frozen", (0.6, 1.0, 0.95), True),
             "frozen in time": pe.StatusColour("Frozen In Time", (0.75, 0.76, 0.95), True),
             "poison": pe.StatusColour("Poison", (0.4, 0.95, 0.3), False)}
    text = "Apply POISON and Frozen in Time, not Frozen. Poisoned ends."
    spans = [(s, e, c.name) for s, e, c in pe.status_spans(text, chart)]
    assert spans == [(6, 12, "Poison"), (17, 31, "Frozen In Time"), (37, 43, "Frozen")]


def test_classify_table():
    parsed = {"card": {"type": "minion", "hp": 5, "def": 2, "atk": 5, "spd": 5, "ability": "Die.\n(BOT) Live."}}
    out = pe.classify(parsed, {"hp": 8, "def": 2, "atk": 11, "spd": None, "stars": 3, "types": ["fire"],
                               "ability": "Die.\n(BOT) Live. ", "name": "X", "note": "n", "series": "mario",
                               "creatures": [{"hp": 1}], "hidden": True})
    assert out.jobs == {"hp": 8}
    assert out.redundant == ["def", "ability"]
    assert [(r.code, r.why.split(":")[0]) for r in out.refusals] == [("stat>10", "atk 11"), ("unsupported-card", "stars"),
                                                                     ("unsupported-card", "types"), ("unsupported-card", "no readable Text/Name layer")]
    assert out.other == ["spd", "note", "series", "creatures", "hidden"]
    assert pe.classify(parsed, {"hp": 8}, excluded=True).refusals[0].code == "stat>10"
    assert pe.classify(parsed, {"ability": "Die.\n(BOT) Die."}).jobs == {"ability": "Die.\n(BOT) Die."}
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
    assert changed == 2 and merged.mode == "RGBA" and merged.size == (10, 10)
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


def test_refused_carries_a_refusal():
    refused = pe.Refused("overflow", "ability does not fit")
    assert refused.refusal == pe.Refusal("overflow", "ability does not fit") and str(refused) == "overflow: ability does not fit"


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
    assert found["name"] == [(300, 23, 899, 114)] and found["ability"] == [(27, 1802, 1157, 2258)]
    assert found["frame"][0] == (0, 1708, 1200, 1748) and found["ability_bg"] == [(0, 1748, 1200, 2400)]
    assert len(found["types"]) == 3 and len(found["stars"]) == 2
    assert {k: [d for _, d, _ in rows] for k, rows in stat_rows(psd).items()} == {"hp": [[5]], "def": [[2]], "atk": [[5]], "spd": [[5]]}
    group, digits, visible = pe.template_row(psd, "hp")
    assert group.name == "Hp Dark" and sorted(digits) == sorted(pe.DIGITS) and visible == [5]


def test_non_template_rows_and_titles_refuse(clone):
    cannon = "Creatures/FTL/5 Stars/Royal_Eradicator_Main_Cannon.psd"
    with pytest.raises(pe.Refused) as refused:
        pe.template_row(PSDImage.open(_card(clone, cannon)[0]), "atk")
    assert refused.value.code == "unsupported-card"
    twins = PSDImage.open(_card(clone, "Creatures/World Of Warcraft/4 Stars/Twin_Emperors.psd")[0])
    for read in (pe.name_layer, pe.ability_layer):
        with pytest.raises(pe.Refused) as refused:
            read(twins)
        assert refused.value.code == "unsupported-card"
    assert pe.read_name(twins) is None
    virus = PSDImage.open(_card(clone, "Creatures/Kirby/3 Stars/Computer_Virus.psd")[0])   # its Name sits at the root
    with pytest.raises(pe.Refused, match="no Text/Name layer"):
        pe.name_layer(virus)
    bone = PSDImage.open(_card(clone, "Creatures/Kirby/4 Stars/Wham_Bam_Bone.psd")[0])      # title named after the card
    with pytest.raises(pe.Refused, match="title layer named 'Wham Bam Bone'"):
        pe.name_layer(bone)


def test_layer_records_fingerprint(clone):
    psd = _card(clone, MINI_DOOMER)[0]
    records = pe.layer_records(psd)
    assert len(records) == 95 and records[93]["name"] == "Name" and records[71]["name"] == "5"
    assert pe.gate_bytes(psd, psd, []) == ([], 0)


# ========================
# Stats (the fork: viewport renders)
# ========================
@text_engine
def test_mini_doomer_hp_5_to_8(clone, parser, tmp_path):
    """S2's gate: 2 flag bytes, the parser reads 8, 0 pixels differ outside the row mask, a rerun does nothing."""
    parsed, baked = _bake(clone, parser, MINI_DOOMER, {"hp": 8}, tmp_path, update_preview=True)
    assert parsed["card"]["hp"] == 5 and baked.reparse["card"]["hp"] == 8 and baked.reparse["validator"] == []
    assert baked.flag_bytes == 2 and [(name, fields) for _, name, fields in baked.records] == [("5", ["flags"]), ("8", ["flags"])]
    assert list(baked.parity) == ["hp"] and baked.parity["hp"]["bad_pct"] == 0 and baked.parity["hp"]["mean"] <= pe.PARITY_MEAN_MAX
    assert clip(baked.region, 1200, 2400) == (763, 1199, 1174, 1285)      # digits 5 and 8 + the Stats Bars stroke
    old = np.asarray(Image.open(_card(clone, MINI_DOOMER)[1]).convert("RGBA"))
    new = np.asarray(Image.open(baked.png_path).convert("RGBA"))
    differs = (old != new).any(axis=2)
    row = np.zeros_like(differs)
    row[HP_ROW[1]:HP_ROW[3], HP_ROW[0]:HP_ROW[2]] = True
    assert int((differs & ~row).sum()) == 0 and int(differs.sum()) == baked.changed_pixels > 0
    saved = PSDImage.open(baked.psd_path)
    assert saved.has_preview() and {d.name for d in pe.template_row(saved, "hp")[1].values() if d.visible} == {"8"}
    assert Image.open(baked.png_path).info.get("icc_profile") == Image.open(_card(clone, MINI_DOOMER)[1]).info.get("icc_profile")
    again = pe.classify(baked.reparse, {"hp": 8}, saved)
    assert again.jobs == {} and again.redundant == ["hp"]
    print("timings", {k: round(v, 2) for k, v in baked.timings.items()})


@text_engine
def test_two_stats_gate_one_region_per_row(clone, parser, tmp_path):
    """Section 5 step 3: hp and spd each get their own baseline, render and merge; the def and atk rows between stay."""
    _, baked = _bake(clone, parser, MINI_DOOMER, {"hp": 8, "spd": 3}, tmp_path)
    assert sorted(baked.parity) == ["hp", "spd"] and all(m["bad_pct"] == 0 for m in baked.parity.values())
    rows = {edit.field: edit.region for edit in baked.edits}
    assert clip(rows["hp"], 1200, 2400)[1:] == (1199, 1174, 1285) and clip(rows["spd"], 1200, 2400)[1:] == (1577, 1174, 1663)
    assert baked.flag_bytes == 4 and sorted(name for _, name, _ in baked.records) == ["3", "5", "5", "8"]
    old = np.asarray(Image.open(_card(clone, MINI_DOOMER)[1]).convert("RGBA"))
    new = np.asarray(Image.open(baked.png_path).convert("RGBA"))
    differs = (old != new).any(axis=2)
    inside = np.zeros_like(differs)
    for x0, y0, x1, y1 in (HP_ROW, SPD_ROW):
        inside[y0:y1, x0:x1] = True
    assert int((differs & ~inside).sum()) == 0 and int(differs.sum()) == baked.changed_pixels > 0
    saved = PSDImage.open(baked.psd_path)
    assert stat_rows(saved)["spd"][0][1] == [3] and stat_rows(saved)["def"][0][1] == [2]


@text_engine
def test_effect_on_a_digit_refuses_parity(clone, parser):
    psd, _ = _card(clone, METTATON)
    parsed = pe.parse_card(parser, psd, METTATON)
    for stat, value in (("atk", 6), ("hp", 8)):           # 6 is hidden and styled; 5 is visible and styled
        with pytest.raises(pe.Refused) as refused:
            pe.set_stat(PSDImage.open(psd), parsed, stat, value)
        assert refused.value.code == "parity" and "GradientOverlay" in refused.value.why


# ========================
# Text (the fork: text engine)
# ========================
@text_engine
@pytest.mark.parametrize("rel", NAMED)
def test_name_edit_passes_every_gate(clone, parser, chart, rel, tmp_path):
    psd_path = _card(clone, rel)[0]
    title = pe.read_name(PSDImage.open(psd_path))
    new = _typo_fix(title)
    parsed, baked = _bake(clone, parser, rel, {"name": new}, tmp_path, chart)
    assert [(name, fields) for _, name, fields in baked.records] == [("Name", TEXT_RECORD)] and baked.flag_bytes == 0
    assert baked.reparse["card"] == parsed["card"]                        # the parser never reads the title
    saved = PSDImage.open(baked.psd_path)
    assert pe.read_name(saved) == new
    assert pe.classify(baked.reparse, {"name": new}, saved).redundant == ["name"]
    assert pe.classify(baked.reparse, {"name": title}, saved).jobs == {"name": title}


@text_engine
def test_long_title_refuses_overflow(clone, chart):
    """Point text has no box: a title that runs off the canvas or onto a star badge refuses overflow (Q5's gate)."""
    fonts = str(clone / pe.FONTS_DIR)
    psd = PSDImage.open(_card(clone, ANUBISATH)[0])
    title = pe.read_name(psd)
    with pytest.raises(pe.Refused, match="off the canvas") as refused:
        pe.set_text_layer(psd, "name", title + " Of The Temple", fonts, chart)
    assert refused.value.code == "overflow"
    psd = PSDImage.open(_card(clone, ANUBISATH)[0])
    with pytest.raises(pe.Refused, match="stars") as refused:                  # 'Anubisath Guardian II' runs into the first star
        pe.set_text_layer(psd, "name", title + " II", fonts, chart)
    assert refused.value.code == "overflow"
    psd = PSDImage.open(_card(clone, MINI_DOOMER)[0])                          # centred: 'Mini Doomer II' ends 3 px short of the star
    edit = pe.set_text_layer(psd, "name", pe.read_name(psd) + " II", fonts, chart)
    assert edit.field == "name" and tuple(pe.name_layer(psd).bbox)[2] < 936


@text_engine
def test_ability_edit_colours_a_status_word(clone, parser, chart, tmp_path):
    psd_path = _card(clone, MINI_DOOMER)[0]
    old = pe.parse_card(parser, psd_path, MINI_DOOMER)["card"]["ability"]
    parsed, baked = _bake(clone, parser, MINI_DOOMER, {"ability": old + " Inflict Poison."}, tmp_path, chart)
    assert [(name, fields) for _, name, fields in baked.records] == [("Ability", TEXT_RECORD)]
    assert pe.normalise_ability(baked.reparse["card"]["ability"]) == pe.normalise_ability(old + " Inflict Poison.")
    layer = pe.ability_layer(PSDImage.open(baked.psd_path))
    text = layer.text
    start = text.index("Poison")
    fills = [fill for s, e, fill in pe.run_fills(layer) if s <= start < e]
    assert fills == [chart["poison"].fill] and chart["poison"].styled is False
    assert baked.region[1] < 1815 and baked.region[3] > 2245                 # the ability box plus the Text group's halo


@text_engine
def test_inserted_text_takes_the_body_fill(clone, chart):
    """set_text(keep="runs") gives inserted text the run before it; after a status word that is the swatch colour."""
    fonts = str(clone / pe.FONTS_DIR)
    psd = PSDImage.open(_card(clone, MINI_DOOMER)[0])
    old = pe.ability_layer(psd).text
    white, green = (1.0, 1.0, 1.0), tuple(round(v, 3) for v in chart["poison"].fill)
    pe.set_text_layer(psd, "ability", old + " Inflict Poison.", fonts, chart)
    pe.set_text_layer(psd, "ability", old + " Inflict Poison for 2 turns.", fonts, chart)
    assert _fills(pe.ability_layer(psd))[-2:] == [("Poison", green), (" for 2 turns.", white)]
    pe.set_text_layer(psd, "ability", old + " Inflict Poisoned for 2 turns.\n", fonts, chart)   # the word widens, the \n goes
    layer = pe.ability_layer(psd)
    assert _fills(layer)[-1][1] == white and layer.text.endswith("Poisoned for 2 turns.")
    assert pe.body_fill(layer, chart) == white


@text_engine
def test_replacing_a_status_word_drops_its_colour(clone, chart):
    path = _card(clone, ASTR)[0]
    fonts = str(clone / pe.FONTS_DIR)
    body = (1.0, 0.996, 0.996)
    for text, tail in (("Attacks inflict 1T3 Burn.", [("Burn.", body)]),
                       ("Attacks inflict 1T3 Empowered and Poison.", [("Empowered", (0.773, 0.0, 0.0)), (" and ", body),
                                                                       ("Poison", tuple(round(v, 3) for v in chart["poison"].fill)), (".", body)])):
        psd = PSDImage.open(path)
        pe.set_text_layer(psd, "ability", text, fonts, chart)
        assert _fills(pe.ability_layer(psd))[-len(tail):] == tail


@text_engine
def test_untouched_styled_word_is_left_alone(clone, chart):
    """styled-word applies to the words the edit touches; a Blazing already on the card keeps its runs."""
    fonts = str(clone / pe.FONTS_DIR)
    psd = PSDImage.open(_card(clone, MINI_DOOMER)[0])
    layer = pe.ability_layer(psd)
    layer.set_text("Inflict Blazing. Die.", keep="runs")
    layer.set_span_style(8, 15, fill=chart["blazing"].fill)
    pe.set_text_layer(psd, "ability", "Inflict Blazing. Live.", fonts, chart)
    assert _fills(pe.ability_layer(psd))[1:] == [("Blazing", (1.0, 0.0, 0.0)), (".", (1.0, 1.0, 1.0)), (" Live.", (1.0, 1.0, 1.0))]
    with pytest.raises(pe.Refused) as refused:                                  # the comma touches the word
        pe.set_text_layer(psd, "ability", "Inflict Blazing, Live.", fonts, chart)
    assert refused.value.code == "styled-word"


@text_engine
def test_too_long_ability_refuses_overflow(clone, chart):
    psd = PSDImage.open(_card(clone, MINI_DOOMER)[0])
    text = pe.normalise_ability(pe.ability_layer(psd).text)
    with pytest.raises(pe.Refused) as refused:
        pe.set_text_layer(psd, "ability", " ".join([text] * 4), str(clone / pe.FONTS_DIR), chart)
    assert refused.value.code == "overflow"


@text_engine
def test_styled_status_word_refuses(clone, chart):
    assert sorted(c.name for c in chart.values() if c.styled) == ["Become Back Money", "Blazing", "Condemned", "Contracted",
        "Corrupted Blood", "Curse", "Cursed Inferno", "Frozen", "Frozen In Time", "Glued", "Hemorrhaging", "Hexed",
        "Infected", "Spirit Flame", "Withering"]
    psd = PSDImage.open(_card(clone, MINI_DOOMER)[0])
    with pytest.raises(pe.Refused) as refused:
        pe.set_text_layer(psd, "ability", "Inflict Blazing.", str(clone / pe.FONTS_DIR), chart)
    assert refused.value.code == "styled-word" and "Blazing" in refused.value.why


@text_engine
def test_missing_font_refuses(clone, parser, tmp_path):
    """Termination Shock's title is set in OCR A Extended, which neither MDW/Fonts nor this machine carries (Q5)."""
    from psd_tools.composite.text import FontMap
    rel = "N.M.E/Termination_Shock.psd"
    psd_path = _card(clone, rel)[0]
    if FontMap(dirs=[str(clone / pe.FONTS_DIR)]).find("OCRAExtended") is not None:
        pytest.skip("OCR A Extended is installed here")
    with pytest.raises(pe.Refused) as refused:
        _bake(clone, parser, rel, {"name": pe.read_name(PSDImage.open(psd_path)) + " II"}, tmp_path)
    assert refused.value.code == "font" and "OCRAExtended" in refused.value.why


@text_engine
def test_baseline_refuses_a_region_the_fork_renders_wrong(clone, parser, tmp_path):
    """The Freezer's title (KrabbyPatty over the Text group's effects) misses the parity bar: refused, never pasted."""
    rel = "Field/1 Stars/The_Freezer.psd"
    psd_path = _card(clone, rel)[0]
    with pytest.raises(pe.Refused) as refused:
        _bake(clone, parser, rel, {"name": _typo_fix(pe.read_name(PSDImage.open(psd_path)))}, tmp_path)
    assert refused.value.code == "parity" and not list(tmp_path.iterdir())
