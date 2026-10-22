"""
psd_edit and psd_regions: the edit pipeline's stat toggle, text edits, delta merge and gates (EDIT_PIPELINE_DESIGN
sections 5, 6, 10; steps S2 and S3). The fixture is the real card repo (MDW/Templates/Creature_Template.psd is 30 MB,
too big to ship): tests that need a PSD read it from the clone and skip without it; tests that render or edit text
skip without the fork's text engine (psd_tools.composite.text), so the suite stays green under stock psd-tools.
conftest.find_fork picks the fork ($AC_PSD_TOOLS_SRC, else an installed fork, else ~/Documents/Tools/psd-tools/src or
/tmp/wt-cards/src when they have its API); the fork venv has psd-tools[text]:
    AC_PSD_TOOLS_SRC=<fork>/src ~/Documents/Tools/psd-tools/.venv/bin/python -m pytest -q tests/test_psd_edit.py
"""
import json
import sys
from copy import deepcopy
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from re import search
from subprocess import CalledProcessError, run

import numpy as np
import pytest
from PIL import Image
from psd_tools import PSDImage
from psd_tools.constants import Resource, Tag
from psd_tools.psd import engine_data

import commands.psd_edit as pe
from commands.psd_regions import PAINT_ORDER, badge_layers, clip, regions, stat_rows
from conftest import bevel_textures, fork_reason, rounds_bytes, saves_preview, story_sync as _story_sync, text_engine as _text_engine

# ========================
# Configuration
# ========================
VOCAB = Path(__file__).resolve().parent / "goldens" / "_vocab.json"
MINI_DOOMER = "Auxiliary/Minions/Mini_Doomer.psd"                 # template rows, no digit effects (certified golden)
HP_ROW = (360, 1200, 1173, 1284)                                   # its hp row: the digits' mask union
METTATON = "Auxiliary/Creatures/Mettaton_EX.psd"                  # a GradientOverlay on one digit per row (Atk: on hidden 6)
SPD_ROW = (359, 1577, 1174, 1663)                                  # its spd row, padded by the Stats Bars stroke
GREEN_MAGE = "Creatures/Everhood/4 Stars/Green_Mage.psd"          # atk 10: no digit visible, hidden 7 styled (the from-10 move)
AZAMOTH_KING = "Creatures/ROTMG/5 Stars/Azamoth_the_Forgotten_King.psd"   # hp 10 with 6, 8 and 9 styled: ambiguous
ANUBISATH = "Auxiliary/Minions/Anubisath_Guardian.psd"            # left-aligned title 11 px short of the first star
ASTR = "Auxiliary/Minions/Astr_Mionion.psd"                       # "Attacks inflict 1T3 Empowered." with Empowered in red
NAMED = ("Creatures/World Of Warcraft/5 Stars/The_Lich_King.psd", ANUBISATH,
         "Items/Support/2 Stars/Volt_Shroom.psd")                  # Text/Name + one visible Text/Ability, fonts in MDW/Fonts
CANNON = "Creatures/FTL/5 Stars/Royal_Eradicator_Main_Cannon.psd"  # the one 10+ card: atk 12 as Top (its 10 marker) + Bottom 2
GASOLINE = "Items/Support/2 Stars/Gasoline.psd"                    # an overlay "Blazing" type layer over the ability (OuterGlow)
FIRE_STAFF = "Items/Equip/2 Stars/Fire_Staff.psd"                  # an overlay "Smoldering": unstyled on the chart, nothing to cut
# the JSX-made FX layers to compare with: Effect Layers - All Cards.jsx's 2026-10-20 trial copies, cut down by
# tests/fixtures/make_jsx_fx.py (the copies: ~/Documents/Tools/psd-research/fixtures/jsx-fx/)
JSX_FX = Path(__file__).resolve().parent / "fixtures" / "jsx_fx"
OVERLAID = {GASOLINE: "3b06f803bb0abf57859297e271177dd82855402b^",    # the last versions with the overlay word: these
            CANNON: "587d38b401102b13d06733a5f62b01ebeb674d3f^"}      # commits hid it under the JSX's FX layer (2026-10-22)
STYLED = ["Become Back Money", "Blazing", "Condemned", "Contracted", "Corrupted Blood", "Curse", "Cursed Inferno", "Frozen",
          "Frozen In Time", "Glued", "Hemorrhaging", "Hexed", "Infected", "Spirit Flame", "Withering"]   # the chart's styled statuses
TEXT_RECORD = ["blocks", "channels", "rect"]                       # what a text edit may change on its layer record
STAT_RECORD = ["blocks", "flags"]                                  # a toggled digit whose effect block moved
GLOW = 18                                                          # the chart's OuterGlow reach (fx_pad) on 13 of the 15


def _typo_fix(title: str) -> str:
    """A rename of the same width: the last letter swapped (the live edits fix typos, they do not lengthen titles)."""
    return title[:len(title) - 1] + ("e" if title[len(title) - 1] != "e" else "a")


def _fills(layer) -> list:
    return [(layer.text[s:e], tuple(round(v, 3) for v in f)) for s, e, f in pe.run_fills(layer)]


def _ink(layer, psd) -> np.ndarray:
    """A layer's ink (alpha > 0.5) placed on the canvas."""
    out = np.zeros((psd.height, psd.width), bool)
    x0, y0, x1, y1 = layer.bbox
    out[y0:y1, x0:x1] = layer.numpy("shape")[..., 0] > 0.5
    return out


def _jsx_ink(rel: str) -> tuple:
    """The JSX FX layer of a trial copy: its name, bbox and ink on the canvas (the fixture of make_jsx_fx.py)."""
    entry = json.loads((JSX_FX / "index.json").read_text(encoding="utf-8"))[rel]
    out = np.zeros(tuple(reversed(entry["canvas"])), bool)
    x0, y0, x1, y1 = entry["bbox"]
    out[y0:y1, x0:x1] = np.asarray(Image.open(JSX_FX / (entry["source"][:-4] + ".png")).convert("L")) > 127
    assert int(out.sum()) == entry["ink"]
    return entry["name"], tuple(entry["bbox"]), out


def _overlaid(clone, rel: str, tmp_path) -> str:
    """OVERLAID[rel]'s PSD from the clone's history, in tmp_path/overlaid/."""
    try:
        data = run(["git", "-C", str(clone), "show", f"{OVERLAID[rel]}:{rel}"], capture_output=True, check=True).stdout
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
# round 4 (tag cards-20261022): rounded bytes and textured bevels
round4 = pytest.mark.skipif(not (_text_engine() and rounds_bytes() and bevel_textures()),
                            reason="the fork lacks round 4's byte rounding or bevel textures")
before_textures = pytest.mark.skipif(not _text_engine() or bevel_textures(), reason="the fork draws bevel textures (round 4)")


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
def test_jsx_fx_fixture_is_vendored_and_small():
    """The comparison's fixture ships in tests/ (no /tmp, no 20 MB trial copies): both cards, ink counts intact."""
    assert sorted(json.loads((JSX_FX / "index.json").read_text(encoding="utf-8"))) == sorted([CANNON, GASOLINE])
    for rel in (GASOLINE, CANNON):
        name, (x0, y0, x1, y1), ink = _jsx_ink(rel)
        assert name == "Ability FX: Blazing" and ink.any() and not ink[:y0].any() and not ink[:, x1:].any()
    assert sum(f.stat().st_size for f in JSX_FX.rglob("*") if f.is_file()) < 64 * 1024


def test_make_jsx_fx_refuses_without_psds(monkeypatch, tmp_path):
    """No trial copies (wiped, other machine): exit non-zero, vendored fixture untouched."""
    spec = spec_from_file_location("make_jsx_fx", JSX_FX.parent / "make_jsx_fx.py")
    script = module_from_spec(spec)
    spec.loader.exec_module(script)
    before = {f.name: f.read_bytes() for f in JSX_FX.iterdir()}
    monkeypatch.setattr(script, "SOURCE", tmp_path)
    with pytest.raises(SystemExit, match="no PSDs"):
        script.main()
    assert {f.name: f.read_bytes() for f in JSX_FX.iterdir()} == before


def test_find_fork_explains_a_skip(monkeypatch, tmp_path):
    """$AC_PSD_TOOLS_SRC wins; without the fork API or psd-tools[text] (or too late to switch) the reason says so."""
    import conftest
    monkeypatch.setenv(conftest.FORK_ENV, str(tmp_path))
    assert conftest.find_fork() == ("", f"{conftest.FORK_ENV}={tmp_path.resolve()} has no {conftest.FORK_API}")
    (tmp_path / conftest.FORK_API).parent.mkdir(parents=True)
    (tmp_path / conftest.FORK_API).write_text("")
    path = list(sys.path)
    for module in conftest.FORK_DEPS:                   # deps missing
        monkeypatch.setattr(conftest, "find_spec", lambda name, real=conftest.find_spec, gone=module:
                            None if name == gone else object() if name in conftest.FORK_DEPS else real(name))
        src, why = conftest.find_fork()
        assert src == "" and "this Python lacks " + conftest.FORK_DEPS[module] in why and sys.path == path
    monkeypatch.setattr(conftest, "find_spec", lambda name: object() if name in conftest.FORK_DEPS else None)
    monkeypatch.setitem(sys.modules, "psd_tools", sys.modules.get("psd_tools", object()))
    assert conftest.find_fork() == ("", f"psd_tools was imported before conftest could put {tmp_path.resolve()} first")
    assert sys.path == path


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


def test_keep_breaks_puts_back_the_layers_breaks():
    """The website shows an ability's \x03 as a newline and a name's breaks as spaces: kept where the edit kept them."""
    layer = "it. \x03(BOT) Die.\rAttacks pierce."
    assert pe.keep_breaks(layer, "it. \r(BOT) Die. Inflict Poison.\rAttacks pierce.", "ability") == \
        "it. \x03(BOT) Die. Inflict Poison.\rAttacks pierce."
    assert pe.keep_breaks(layer, "it.\r(BOT) Die.\rNew.\rAttacks pierce.", "ability") == \
        "it.\x03(BOT) Die.\rNew.\rAttacks pierce."                 # a new paragraph stays one
    assert pe.keep_breaks("Die.\rLive.", "Die.\rLive.\rAgain.", "ability") == "Die.\rLive.\rAgain."
    assert pe.keep_breaks("Unbreakable\x03strength", "Unbreakable Strength", "name") == "Unbreakable\x03Strength"
    assert pe.keep_breaks("Mega\rBee", "Mega Bees", "name") == "Mega\rBees"
    assert pe.keep_breaks("B.F.B", "B.F.B.", "name") == "B.F.B."


def test_changed_range_is_the_engines():
    assert pe.changed_range("Inflict Poison.", "Inflict Poison for 2 turns.") == (14, 26)
    assert pe.changed_range("Inflict Poison.", "Inflict.") == (7, 7)                 # removal only
    assert pe.changed_range("Inflict Poison.", "Inflict Poisoned.") == (14, 16)
    assert pe._word_bounds("Inflict Poisoned.", 14, 16) == (8, 16)
    assert pe.changed_range("Die.", "Die.") == (5, 5) and pe.changed_range("", "Die.") == (0, 4)


def test_status_spans_longest_first_whole_words():
    chart = {"frozen": pe.StatusColor("Frozen", (0.6, 1.0, 0.95), True),
             "frozen in time": pe.StatusColor("Frozen In Time", (0.75, 0.76, 0.95), True),
             "poison": pe.StatusColor("Poison", (0.4, 0.95, 0.3), False)}
    text = "Apply POISON and Frozen in Time, not Frozen. Poisoned ends."
    spans = [(s, e, c.name) for s, e, c in pe.status_spans(text, chart)]
    assert spans == [(6, 12, "Poison"), (17, 31, "Frozen In Time"), (37, 43, "Frozen")]


def test_fx_names_follow_the_jsx():
    assert pe.fx_name("Ability", "Blazing") == "Ability FX: Blazing" and pe.fx_name("Ability", "Frozen In Time", 2) == "Ability FX: Frozen In Time#2"
    assert pe.FX_TAG == pe.EFFECT_LAYER_TAG == " FX: " and pe.ALLOWED_RECORD_FIELDS["overlay"] == {"flags"}


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


def test_merge_preview_takes_the_raw_render():
    """merge_png()'s pixels, in the PSD's own values; an RGB preview takes them over white. One pass for every edit."""
    base = Image.new("RGBA", (4, 4), (10, 20, 30, 255))
    render, raw = base.copy(), Image.new("RGBA", (4, 4), (9, 9, 9, 255))
    render.putpixel((1, 1), (200, 0, 0, 255))
    raw.putpixel((1, 1), (190, 4, 2, 255))
    render.putpixel((3, 2), (0, 200, 0, 51))
    raw.putpixel((3, 2), (0, 190, 0, 51))
    changed = pe.merge_png(Image.new("RGBA", (10, 10)), base, render, (2, 2, 6, 6))[1]
    second = np.zeros((4, 4), bool)
    second[0, 0] = True
    for mode, fill in (("RGBA", (1, 2, 3, 255)), ("RGB", (1, 2, 3))):
        patches = [(changed, raw, (2, 2, 6, 6)), (second, Image.new("RGBA", (4, 4), (7, 8, 9, 255)), (6, 6, 10, 10))]
        preview = pe.merge_preview(Image.new(mode, (10, 10), fill), patches)
        assert preview.mode == mode and preview.size == (10, 10)
        assert preview.getpixel((3, 3))[:3] == (190, 4, 2) and preview.getpixel((2, 2)) == fill
        assert preview.getpixel((5, 4)) == ((0, 190, 0, 51) if mode == "RGBA" else (204, 242, 204))
        assert preview.getpixel((6, 6))[:3] == (7, 8, 9) and preview.getpixel((7, 7)) == fill


def test_save_psd_stores_the_merged_preview(tmp_path):
    """The fork stores stored_preview()'s image as given; stock psd-tools has no save(preview=): None, regenerate."""
    psd = PSDImage.new("RGB", (4, 4), (10, 20, 30))
    if not saves_preview():
        assert pe.stored_preview(psd) is None
        return
    preview = pe.stored_preview(psd)
    assert preview.mode == "RGB" and preview.getpixel((0, 0)) == (10, 20, 30)
    preview.putpixel((1, 1), (200, 0, 0))
    pe.save_psd(psd, str(tmp_path / "x.psd"), True, preview)
    saved = PSDImage.open(str(tmp_path / "x.psd"))
    assert saved.has_preview() and np.array_equal(np.asarray(saved.topil(apply_icc=False)), np.asarray(preview))


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
    assert found["name"] == [(300, 23, 899, 114)] and found["ability"] == [(27, 1802, 1157, 2264)]   # 2258 before the (BOT) badge
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


def test_name_redundancy_uses_the_printed_title(clone, parser):
    """A name edit is redundant when the bot already shows it (the parser's display_name, else the file name)."""
    path = _card(clone, MINI_DOOMER)[0]
    parsed = pe.parse_card(parser, path, MINI_DOOMER)
    psd = PSDImage.open(path)
    assert parsed["card"].get("display_name") is None and pe.shown_name(parsed) == "Mini Doomer"   # prints its file name
    assert pe.classify(parsed, {"name": "Mini  Doomer\r"}, psd).redundant == ["name"]
    assert pe.classify(parsed, {"name": "Mini Doomer II"}, psd).jobs == {"name": "Mini Doomer II"}
    rel = "Creatures/Kirby/3 Stars/Computer_Virus.psd"                     # Name at the root: read, not editable
    path = _card(clone, rel)[0]
    parsed = pe.parse_card(parser, path, rel)
    assert parsed["card"]["display_name"] == pe.shown_name(parsed) == "Computer Virus appears!"
    assert pe.classify(parsed, {"name": "Computer Virus appears!"}, PSDImage.open(path)).redundant == ["name"]
    assert pe.classify(parsed, {"name": "Computer Virus"}, PSDImage.open(path)).refusals[0].why == "no readable Text/Name layer"


def test_royal_eradicator_two_row_atk(clone, parser, tmp_path):
    """Idea 13 on the one 10+ card. `atk dark` holds Top ("0".."9" plus the parser's "10" marker: visible at opacity 0, so the
    parser adds a full top row; hidden, it reads Bottom alone: 2) and Bottom ("0".."9", 2 visible with ColorOverlay + BevelEmboss).
    12 -> 7 shows Top 7 + Bottom 0 and moves 2's effect onto 0; 12 -> 15 shows Bottom 5; 12 -> 10 Bottom 0; 20 refuses (no
    Bottom marker); the parser reads each result; gate bytes sees one flag byte per digit plus the moved blocks; hp (one row) refuses 11."""
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
        assert edit.moves == [(two, new_digit, b"lfx2"), (two, new_digit, b"lrFX")]
        saved = str(tmp_path / f"cannon_{value}.psd")
        pe.save_psd(psd, saved, update_preview=False)
        records, flag_bytes, block_bytes = pe.gate_bytes(path, saved, [edit])
        assert flag_bytes == len(names) and block_bytes == 6360 and sorted(name for _, name, _ in records) == sorted(names)
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
    # the replace path: hp 9 (styled) -> 2 (a block with nothing enabled): 2's block goes, 9's lands
    psd = PSDImage.open(path)
    edit = pe.set_stat(psd, parsed, "hp", 2)
    row = pe.template_rows(psd, "hp")[0]
    two, nine = pe._record_index(row.digits["2"]), pe._record_index(row.digits["9"])
    assert edit.moves == [(two, None, b"lfx2"), (two, None, b"lrFX"), (nine, two, b"lfx2"), (nine, two, b"lrFX")]
    saved = str(tmp_path / "cannon_hp2.psd")
    pe.save_psd(psd, saved, update_preview=False)
    records, flag_bytes, block_bytes = pe.gate_bytes(path, saved, [edit])
    assert (flag_bytes, block_bytes) == (2, 6360) and [(name, fields) for _, name, fields in records] == [("2", STAT_RECORD), ("9", STAT_RECORD)]
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


def test_effect_rules_on_styled_rows(clone):
    """Idea 12's rule on real rows: Mettaton atk 7 -> 8 moves nothing (7 is unstyled; the hidden styled 6 stays); 7 -> 6 keeps 6's
    own effect; hp 5 -> 10 moves nothing (5 keeps its block, hidden) and 10 -> 3 then takes it from 5; Green Mage atk 10 -> 4 takes
    the one hidden styled digit's (7); Azamoth the Forgotten King's hp (10; 6, 8 and 9 styled) refuses as ambiguous."""
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
    assert edit.moves == [(five, three, b"lfx2"), (five, three, b"lrFX")] and [layer.name for layer in edit.layers] == ["3", "5"]
    assert row.visible == [3] and pe._enabled_effects(row.digits["3"]) == ["GradientOverlay"] and pe._enabled_effects(row.digits["5"]) == []
    green = PSDImage.open(_card(clone, GREEN_MAGE)[0])
    edit = pe.set_stat(green, {"card": {"atk": 10}}, "atk", 4)
    row = pe.template_rows(green, "atk")[0]
    assert [layer.name for layer in edit.layers] == ["4", "7"] and len(edit.moves) == 2
    assert pe._enabled_effects(row.digits["4"]) == ["GradientOverlay"] and pe._enabled_effects(row.digits["7"]) == []
    with pytest.raises(pe.Refused, match="6, 8, 9") as refused:
        pe.set_stat(PSDImage.open(_card(clone, AZAMOTH_KING)[0]), {"card": {"hp": 10}}, "hp", 3)
    assert refused.value.code == "unsupported-card"


def test_layer_records_fingerprint(clone):
    psd = _card(clone, MINI_DOOMER)[0]
    records = pe.layer_records(psd)
    assert len(records) == 96 and records[93]["name"] == "Name" and records[71]["name"] == "5"
    assert records[95]["name"] == "Ability FX: (BOT)"
    assert pe.gate_bytes(psd, psd, []) == ([], 0, 0)


# ========================
# Stats (the fork: viewport renders)
# ========================
@text_engine
def test_render_pair_is_render_region_and_its_raw_values(clone):
    """One unconverted render gives render_region()'s image byte for byte, and the preview's values beside it."""
    psd = PSDImage.open(_card(clone, MINI_DOOMER)[0])
    assert Resource.ICC_PROFILE in psd.image_resources                # a non-sRGB profile: the two differ
    box = (763, 1199, 863, 1285)
    render, raw = pe.render_pair(psd, box)
    assert np.array_equal(np.asarray(render), np.asarray(pe.render_region(psd, box)))
    assert not np.array_equal(np.asarray(render), np.asarray(raw))
    assert pe.stored_preview(psd).mode == "RGB" if saves_preview() else pe.stored_preview(psd) is None


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
    assert int((differs & ~inside).sum()) == 0 and 0 < int(differs.sum()) <= baked.changed_pixels
    saved = PSDImage.open(baked.psd_path)
    assert stat_rows(saved)["spd"][0][1] == [3] and stat_rows(saved)["def"][0][1] == [2]


@text_engine
def test_mettaton_hp_5_to_8_moves_the_effect(clone, parser, tmp_path):
    """Idea 12 through every gate: hp 5 -> 8 hides the styled 5 and shows 8 with its GradientOverlay. 2 flag bytes plus the
    lfx2 and lrFX blocks (5884 bytes) leaving record 5 for record 8, nothing else; the dark cells render within the bar
    (mean 0.90, 0% bad, max 7 on 2026-10-21); the parser reads 8; a rerun finds nothing."""
    parsed, baked = _bake(clone, parser, METTATON, {"hp": 8}, tmp_path)
    assert parsed["card"]["hp"] == 5 and baked.reparse["card"]["hp"] == 8 and baked.reparse["validator"] == []
    assert baked.flag_bytes == 2 and baked.block_bytes == 5884
    assert [(name, fields) for _, name, fields in baked.records] == [("5", STAT_RECORD), ("8", STAT_RECORD)]
    five, eight = (index for index, _, _ in baked.records)
    assert baked.edits[0].moves == [(five, eight, b"lfx2"), (five, eight, b"lrFX")]
    assert baked.parity["hp"]["bad_pct"] == 0 and baked.parity["hp"]["mean"] <= pe.PARITY_MEAN_MAX and baked.parity["hp"]["max"] <= 7
    assert clip(baked.region, 1200, 2400) == (763, 1199, 1174, 1285) and baked.changed_pixels > 0
    saved = PSDImage.open(baked.psd_path)
    row = pe.template_rows(saved, "hp")[0]
    assert row.visible == [8] and pe._enabled_effects(row.digits["8"]) == ["GradientOverlay"] and pe._enabled_effects(row.digits["5"]) == []
    assert not any(key in row.digits["5"].tagged_blocks for key in pe.EFFECT_BLOCKS)
    assert pe.classify(baked.reparse, {"hp": 8}, saved).redundant == ["hp"]
    print("timings", {k: round(v, 2) for k, v in baked.timings.items()})


@text_engine
def test_moved_effect_paints_the_new_digit(clone):
    """hp 5 -> 2 in memory: the fork paints the moved GradientOverlay over cells 3-10, so the colour inside digit 2's mask
    matches the official PNG inside digit 5's mask (the lit cells were bright before). With rounded bytes the baseline
    passes parity (mean 0.556, max 7, 0% bad on 2026-10-22); truncated it refuses on the mean (1.03)."""
    psd_path, png_path = _card(clone, METTATON)
    png = Image.open(png_path)
    psd = PSDImage.open(psd_path)
    edit = pe.set_stat(psd, {"card": {"hp": 5}}, "hp", 2)
    row = pe.template_rows(psd, "hp")[0]
    render = np.asarray(pe.render_region(psd, edit.region).convert("RGB")).astype(float)
    official = np.asarray(png.convert("RGB")).astype(float)

    def inside(pixels, layer, origin):
        x0, y0, x1, y1 = layer.mask.bbox
        mask = np.asarray(layer.mask.topil()) > 128
        return pixels[y0 - origin[1]:y1 - origin[1], x0 - origin[0]:x1 - origin[0]][mask].mean(axis=0)
    new = inside(render, row.digits["2"], edit.region[:2])
    assert np.abs(new - inside(official, row.digits["5"], (0, 0))).max() < 3
    assert np.abs(new - inside(official, row.digits["2"], (0, 0))).max() > 20
    if rounds_bytes():
        _, measured = pe.baseline(PSDImage.open(psd_path), png, edit.region)
        assert measured["bad_pct"] == 0 and measured["mean"] <= pe.PARITY_MEAN_MAX and measured["max"] <= 7
    else:
        with pytest.raises(pe.Refused) as refused:
            pe.baseline(PSDImage.open(psd_path), png, edit.region)
        assert refused.value.code == "parity" and refused.value.why.startswith("baseline bad 0.00% mean 1.0")


@round4
def test_cannon_atk_passes_parity_and_hp_refuses_on_the_mean(clone, parser, tmp_path):
    """The Cannon's digits carry ColorOverlay + a textured BevelEmboss. With round 4 every row is 0% bad (2026-10-22):
    the two-row atk passes (mean 0.51) and bakes, the one-row hp refuses on the mean alone (1.06 against the 1.0 bar)."""
    for stat in ("atk", "hp"):
        (tmp_path / stat).mkdir()
    parsed, baked = _bake(clone, parser, CANNON, {"atk": 15}, tmp_path / "atk")
    assert baked.reparse["card"]["atk"] == 15 and baked.reparse["validator"] == parsed["validator"]   # STATS TOO HIGH: atk 12 is
    assert baked.parity["atk"]["bad_pct"] == 0 and baked.parity["atk"]["mean"] <= pe.PARITY_MEAN_MAX
    with pytest.raises(pe.Refused) as refused:
        _bake(clone, parser, CANNON, {"hp": 2}, tmp_path / "hp")
    assert refused.value.code == "parity" and refused.value.why.startswith("baseline bad 0.00% mean 1.")
    assert float(search(r"mean ([\d.]+)", refused.value.why).group(1)) > pe.PARITY_MEAN_MAX
    assert not list((tmp_path / "hp").iterdir())


@before_textures
def test_cannon_rows_refuse_parity_without_bevel_textures(clone, parser, tmp_path):
    """Before round 4 the fork skips the digits' BevelEmboss texture (5-7% wrong, max 110): apply_jobs refuses parity before
    any pixel is pasted, for the two-row atk as for its one-row hp."""
    for stat, value in (("atk", 15), ("hp", 2)):
        with pytest.raises(pe.Refused) as refused:
            _bake(clone, parser, CANNON, {stat: value}, tmp_path)
        assert refused.value.code == "parity" and refused.value.why.startswith("baseline bad")
        assert float(search(r"bad ([\d.]+)%", refused.value.why).group(1)) > pe.PARITY_BAD_MAX
    assert not list(tmp_path.iterdir())


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


@text_edit
def test_ability_edit_colours_a_status_word(clone, parser, chart, tmp_path):
    psd_path = _card(clone, MINI_DOOMER)[0]
    old = pe.parse_card(parser, psd_path, MINI_DOOMER)["card"]["ability"]
    parsed, baked = _bake(clone, parser, MINI_DOOMER, {"ability": old + " Inflict Poison."}, tmp_path, chart)
    assert [(name, fields) for _, name, fields in baked.records] == [("Ability", TEXT_RECORD)]
    assert pe.normalize_ability(baked.reparse["card"]["ability"]) == pe.normalize_ability(old + " Inflict Poison.")
    layer = pe.ability_layer(PSDImage.open(baked.psd_path))
    text = layer.text
    start = text.index("Poison")
    fills = [fill for s, e, fill in pe.run_fills(layer) if s <= start < e]
    assert fills == [chart["poison"].fill] and chart["poison"].styled is False
    assert baked.region[1] < 1815 and baked.region[3] > 2245                 # the ability box plus the Text group's halo


@text_edit
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


@text_edit
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


@text_edit
def test_styled_status_word_gets_an_fx_layer(clone, chart):
    """Idea 14 in memory on Mini_Doomer: Blazing is styled on the chart, so the edit cuts "Ability FX: Blazing" from the new
    raster (the ability's own pixels under the word), directly above Text, carrying the chart layer's lfx2/lrFX bytes and a
    fresh layer id; the word keeps the swatch. FX layers are derived state: the next edit rebuilds them, a repeat is numbered
    and a removed word leaves nothing behind."""
    assert sorted(c.name for c in chart.values() if c.styled) == STYLED
    fonts = str(clone / pe.FONTS_DIR)
    psd = PSDImage.open(_card(clone, MINI_DOOMER)[0])
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
    edit = pe.set_text_layer(psd, "ability", "Inflict Blazing. Blazing again.", fonts, chart)
    names = [fx.name for fx in pe.fx_layers(psd, pe.ability_layer(psd))]
    assert names == ["Ability FX: Blazing#1", "Ability FX: Blazing#2"] == [n for n, _ in edit.fx] and edit.dropped == ["Ability FX: Blazing"]
    assert [psd.index(fx) for fx in pe.fx_layers(psd, pe.ability_layer(psd))] == [psd.index(text) + 1, psd.index(text) + 2]
    edit = pe.set_text_layer(psd, "ability", "Inflict Poison.", fonts, chart)
    assert pe.fx_layers(psd, pe.ability_layer(psd)) == [] and edit.fx == [] and edit.dropped == names
    assert len(list(psd.descendants())) == count


@text_edit
def test_status_words_are_normalized_to_the_chart(clone, chart):
    """The chart is the authority (design 14 q6): a status word the edit leaves alone takes its swatch too (a grey Poison turns
    the chart's green), and an edit touching a styled word no longer refuses (round 1's styled-word)."""
    fonts = str(clone / pe.FONTS_DIR)
    psd = PSDImage.open(_card(clone, MINI_DOOMER)[0])
    layer = pe.ability_layer(psd)
    layer.set_text("Inflict Poison. Die.", keep="runs")
    layer.set_span_style(8, 14, fill=(0.5, 0.5, 0.5))
    pe.set_text_layer(psd, "ability", "Inflict Poison. Live.", fonts, chart)
    green = tuple(round(v, 3) for v in chart["poison"].fill)
    fills = _fills(pe.ability_layer(psd))
    assert fills[1] == ("Poison", green) and all(fill == (1.0, 1.0, 1.0) for _, fill in fills[2:])
    pe.set_text_layer(psd, "ability", "Inflict Blazing, Live.", fonts, chart)
    assert _fills(pe.ability_layer(psd))[1][0] == "Blazing" and [fx.name for fx in pe.fx_layers(psd, pe.ability_layer(psd))] == ["Ability FX: Blazing"]


@text_edit
def test_overlay_word_layer_without_a_chart_style_refuses(clone, chart):
    """Fire Staff's hand-made "Smoldering" overlay carries a Stroke the chart knows nothing about: nothing could stand in for it."""
    fonts = str(clone / pe.FONTS_DIR)
    psd = PSDImage.open(_card(clone, FIRE_STAFF)[0])
    text = pe.ability_layer(psd).text
    with pytest.raises(pe.Refused, match="overlay word layers the chart has no style for: Smoldering") as refused:
        pe.set_text_layer(psd, "ability", text + " Die.", fonts, chart)
    assert refused.value.code == "unsupported-card" and pe.ability_layer(psd).text == text
    with pytest.raises(pe.Refused, match="status word layers over the ability"):
        pe.set_text_layer(psd, "ability", text + " Die.", fonts, None)


@text_edit
def test_fx_without_the_engine_refuses_styled_word(clone, chart, monkeypatch):
    """styled-word now means this psd-tools cannot cut glyph pixels: with the rasterizer gone a styled word refuses before
    anything changes, an unstyled one still bakes."""
    fonts = str(clone / pe.FONTS_DIR)
    psd = PSDImage.open(_card(clone, MINI_DOOMER)[0])
    before = pe.ability_layer(psd).text
    monkeypatch.setitem(sys.modules, "psd_tools.composite.text.raster", None)
    with pytest.raises(pe.Refused, match="cannot cut glyph pixels") as refused:
        pe.set_text_layer(psd, "ability", "Inflict Blazing.", fonts, chart)
    assert refused.value.code == "styled-word" and pe.ability_layer(psd).text == before
    assert pe.set_text_layer(psd, "ability", "Inflict Poison.", fonts, chart).fx == []


@text_edit
@pytest.mark.parametrize("rel, floor", [(GASOLINE, 0.95), (CANNON, 0.9)])
def test_fx_layers_match_the_jsx_output(clone, parser, chart, rel, floor, tmp_path):
    """The proof against Effect Layers.jsx (Convert overlays on the 2026-10-20 trial copies): regenerating the FX layer for
    the unchanged text cuts the same word (IoU 0.966 / 0.937 with the JSX pixel cut, 99.9% / 100% of our ink on its ink: the
    rest is the fork's raster against Photoshop's), hides the overlay, passes the validator's coverage (1.0) and the bytes
    and reopen gates. These two cards keep image-icon gaps, so the token rule refuses their edits in the pipeline. The
    clone's versions before the JSX ran on them (OVERLAID)."""
    psd_path = _overlaid(clone, rel, tmp_path)
    psd = PSDImage.open(psd_path)
    text = pe.ability_layer(psd).text
    assert [pe.collapse(o.text) for o in pe.overlay_words(psd)] == ["Blazing"]
    edit = pe.set_text_layer(psd, "ability", text, str(clone / pe.FONTS_DIR), chart)
    assert [n for n, _ in edit.fx] == ["Ability FX: Blazing"] and edit.dropped == [] and len(edit.hidden) == 1
    saved = str(tmp_path / Path(rel).name)
    pe.save_psd(psd, saved, update_preview=False)
    reopened = PSDImage.open(saved)
    mine, = pe.fx_layers(reopened, pe.ability_layer(reopened))
    name, box, theirs = _jsx_ink(rel)
    ours = _ink(mine, reopened)
    assert name == mine.name and theirs.shape == ours.shape
    assert (ours & theirs).sum() / (ours | theirs).sum() >= floor and (ours & theirs).sum() / ours.sum() >= 0.99
    assert abs(mine.bbox[0] - box[0]) <= 1 and abs(mine.bbox[1] - box[1]) <= 1 and mine.bbox[3] == box[3]
    assert pe.overlay_words(reopened) == [] and pe._enabled_effects(mine) == ["OuterGlow"]
    assert pe.PSDParser._pixel_coverage(mine, pe.ability_layer(reopened)) == 1.0
    old, new = pe.parse_card(parser, psd_path, rel), pe.parse_card(parser, saved, rel)
    assert new["validator"] == old["validator"] and not any(p.startswith("EFFECT LAYER") for p in new["validator"])
    records, flag_bytes, block_bytes = pe.gate_bytes(psd_path, saved, [edit])
    assert [(name, fields) for _, name, fields in records] == [("Ability", ["blocks", "channels"]), ("Ability FX: Blazing", ["added"]),
                                                               ("Blazing", ["flags"])]
    assert flag_bytes == 1 and block_bytes == sum(len(b) for b in edit.fx[0][1].values()) == 5916
    pe.gate_reparse(parser, old, saved, rel, {}, reopened, [edit])


@text_edit
def test_ability_bake_regenerates_fx_layers_through_every_gate(clone, parser, chart, tmp_path):
    """Idea 14 through apply_jobs on Mini_Doomer: adding Blazing and Frozen cuts two FX layers (2 x 5916 block bytes landed),
    within the parity bar, the validator clean, nothing outside the region; a second bake on the result drops Frozen and
    numbers the repeated Blazing, the old layers removed and the new ones added in the record diff."""
    psd_path, png_path = _card(clone, MINI_DOOMER)
    old = pe.parse_card(parser, psd_path, MINI_DOOMER)["card"]["ability"]
    one = tmp_path / "one"
    one.mkdir()
    parsed, baked = _bake(clone, parser, MINI_DOOMER, {"ability": old + " Inflict Blazing and Frozen."}, one, chart)
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
    two = tmp_path / "two"
    two.mkdir()
    text = old + " Inflict Blazing. Blazing twice."
    parsed = pe.parse_card(parser, baked.psd_path, MINI_DOOMER)
    assert pe.classify(parsed, {"ability": text}, saved).jobs == {"ability": text}
    again = pe.apply_jobs(baked.psd_path, baked.png_path, MINI_DOOMER, {"ability": text}, parsed, parser, str(two),
                          str(clone / pe.FONTS_DIR), chart, update_preview=False)
    assert [(name, fields) for _, name, fields in again.records] == [
        ("Ability", TEXT_RECORD), ("Ability FX: Blazing", ["removed"]), ("Ability FX: Frozen", ["removed"]),
        ("Ability FX: Blazing#1", ["added"]), ("Ability FX: Blazing#2", ["added"])]
    assert again.fx == ["Ability FX: Blazing#1", "Ability FX: Blazing#2"] and again.reparse["validator"] == []
    assert again.edits[0].dropped == ["Ability FX: Blazing", "Ability FX: Frozen"]
    assert pe.stale_stories(PSDImage.open(again.psd_path)) == []
    print("timings", {k: round(v, 2) for k, v in baked.timings.items()}, {k: round(v, 2) for k, v in again.timings.items()})


@text_edit
def test_too_long_ability_refuses_overflow(clone, chart):
    psd = PSDImage.open(_card(clone, MINI_DOOMER)[0])
    text = pe.normalize_ability(pe.ability_layer(psd).text)
    with pytest.raises(pe.Refused) as refused:
        pe.set_text_layer(psd, "ability", " ".join([text] * 4), str(clone / pe.FONTS_DIR), chart)
    assert refused.value.code == "overflow"


@text_edit
def test_missing_font_refuses(clone, parser, tmp_path, monkeypatch):
    """Both `font` refusals, whatever this machine has installed: the fork's lookup is made to miss Termination Shock's
    OCR A Extended (Q5), then to find Anubisath's title face only loosely."""
    from psd_tools.composite.text import FontMap
    lookup = FontMap._lookup
    monkeypatch.setattr(FontMap, "_lookup", lambda self, name: None if name == "OCRAExtended" else lookup(self, name))
    rel = "N.M.E/Termination_Shock.psd"
    psd_path = _card(clone, rel)[0]
    with pytest.raises(pe.Refused) as refused:
        _bake(clone, parser, rel, {"name": pe.read_name(PSDImage.open(psd_path)) + " II"}, tmp_path)
    assert refused.value.code == "font" and "OCRAExtended" in refused.value.why
    monkeypatch.setattr(FontMap, "_lookup", lambda self, name: (lambda hit: None if hit is None else (hit[0], "loose"))(lookup(self, name)))
    psd_path = _card(clone, ANUBISATH)[0]
    with pytest.raises(pe.Refused) as refused:
        _bake(clone, parser, ANUBISATH, {"name": _typo_fix(pe.read_name(PSDImage.open(psd_path)))}, tmp_path)
    assert refused.value.code == "font" and refused.value.why.startswith("not resolved exactly: ")


@text_engine
def test_baseline_refuses_a_region_the_fork_renders_wrong(clone, parser, tmp_path):
    """The Freezer's title (KrabbyPatty over the Text group's effects) misses the parity bar: refused, never pasted."""
    rel = "Field/1 Stars/The_Freezer.psd"
    psd_path = _card(clone, rel)[0]
    with pytest.raises(pe.Refused) as refused:
        _bake(clone, parser, rel, {"name": _typo_fix(pe.read_name(PSDImage.open(psd_path)))}, tmp_path)
    assert refused.value.code == "parity" and not list(tmp_path.iterdir())


# ========================
# Txt2 stories (the copy of the text Photoshop reads)
# ========================
def stale_copy(clone, rel: str, title: str, tmp_path) -> tuple:
    """A pre-fix bake's PSD: the title set in TySh alone (the fork's set_text, then the old story put back)."""
    psd = PSDImage.open(_card(clone, rel)[0])
    layer = pe.name_layer(psd)
    index = pe.story_index(layer)
    items = pe._story_list(pe._txt2(psd))[2]
    old = deepcopy(items[index])
    layer.set_text(title)
    items[index] = old
    path = tmp_path / Path(rel).name
    pe.save_psd(psd, str(path), update_preview=False)
    return str(path), index


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
def test_ability_edit_keeps_a_forced_line_break(clone, chart):
    """Mini Doomer's "it. \x03(BOT) Die.": the website's newline there stays a soft break, its paragraph runs as they were."""
    psd = PSDImage.open(_card(clone, MINI_DOOMER)[0])
    layer = pe.ability_layer(psd)
    old = layer.text
    assert "it. \x03(BOT) Die." in old
    pe.set_text_layer(psd, "ability", pe.normalize_ability(old) + " Inflict Poison.", str(clone / pe.FONTS_DIR), chart)
    assert layer.text == old.rstrip() + " Inflict Poison."
    runs = [int(n) for n in layer.engine_dict["ParagraphRun"]["RunLengthArray"]]
    model = pe._story_list(pe._txt2(psd))[2][pe.story_index(layer)]["0"]
    assert runs == [int(run["1"].value) for run in model["5"]["0"]] == [72, len(layer.text) + 1 - 72]
    assert pe.stale_stories(psd) == [] and pe.story_style_drift(psd) == []


@text_edit
def test_icon_font_tokens_and_colour_glyphs(clone, chart):
    """The_Hive is set in stoogedings: a typed [fire] takes the icon font in TySh and the story; its colour glyphs
    would come out white with no overlay over the ability (Mini Doomer's gradient covers them), so it refuses."""
    psd = PSDImage.open(_card(clone, "Field/4 Stars/The_Hive.psd")[0])
    layer = pe.ability_layer(psd)
    assert pe.uses_icon_font(layer) and pe.icon_font(layer) == "stoogedings-Regular"
    assert not pe.uses_icon_font(pe.name_layer(psd)) and not pe._overlaid(layer)
    assert pe._overlaid(pe.ability_layer(PSDImage.open(_card(clone, MINI_DOOMER)[0])))
    text = pe.normalize_ability(layer.text) + " Deal 1 [fire] damage."
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


@text_edit
def test_story_style_drift_catches_a_tysh_only_fill(clone, tmp_path):
    """A fill written to TySh alone: Photoshop would show the story's colour. The bake refuses such a card."""
    path, png = _card(clone, MINI_DOOMER)
    psd = PSDImage.open(path)
    assert pe.story_style_drift(psd) == []
    data = pe.ability_layer(psd).engine_dict["StyleRun"]["RunArray"][0]["StyleSheet"]["StyleSheetData"]
    data["FillColor"]["Values"][1] = engine_data.Float(0.0)
    assert pe.story_style_drift(psd) == [("Ability", "FillColor at 0-72: layer [1.0, 0.0, 1.0, 1.0], story [1.0, 1.0, 1.0, 1.0]")]
    with pytest.raises(pe.Refused, match="Ability's Txt2 story has other styles: FillColor"):
        pe.gate_stories(PSDImage.open(path), psd, set())
    drifted = tmp_path / "drifted.psd"
    pe.save_psd(psd, str(drifted), update_preview=False)
    with pytest.raises(pe.Refused, match="other styles") as refused:      # checked before any job or parse
        pe.apply_jobs(str(drifted), png, MINI_DOOMER, {"hp": 8}, {}, None, str(tmp_path), str(clone / pe.FONTS_DIR))
    assert refused.value.code == "unsupported-card"


@text_edit
def test_gate_stories_refuses_a_dropped_text_index(clone):
    """A layer without its TextIndex leaves its story orphaned (a fork whose edits drop it): refused, not passed."""
    path = _card(clone, MINI_DOOMER)[0]
    old, new = PSDImage.open(path), PSDImage.open(path)
    layer = pe.name_layer(new)
    index = pe.story_index(layer)
    del layer._data.text_data[b"TextIndex"]
    with pytest.raises(pe.Refused, match="TextIndex") as refused:
        pe.gate_stories(old, new, {index})
    assert refused.value.code == "bytes" and str(index) in refused.value.why


@text_edit
def test_bake_refuses_a_card_with_a_stale_story(clone, parser, tmp_path):
    """A pre-fix bake's PSD is refused until txt2_repair has run on it, whatever the job."""
    title = pe.read_name(PSDImage.open(_card(clone, MINI_DOOMER)[0]))
    path, _ = stale_copy(clone, MINI_DOOMER, _typo_fix(title), tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    parsed = pe.parse_card(parser, path, MINI_DOOMER)
    with pytest.raises(pe.Refused, match="Txt2 story is stale") as refused:
        pe.apply_jobs(path, _card(clone, MINI_DOOMER)[1], MINI_DOOMER, {"hp": 8}, parsed, parser, str(out),
                      str(clone / pe.FONTS_DIR))
    assert refused.value.code == "unsupported-card" and not any(out.iterdir())


@text_engine
def test_text_edit_needs_the_story_sync(clone, chart, monkeypatch):
    """A fork whose set_text writes TySh alone never edits a card: ImportError, which the bake reports as transient."""
    monkeypatch.setattr(pe, "story_sync", lambda: False)
    psd = PSDImage.open(_card(clone, MINI_DOOMER)[0])
    with pytest.raises(ImportError, match="Txt2 story stale"):
        pe.set_text_layer(psd, "name", "Mini Doomer II", str(clone / pe.FONTS_DIR), chart)
    assert pe.read_name(psd) == "Mini Doomer"
