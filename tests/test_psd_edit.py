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
from collections import Counter
from copy import deepcopy
from dataclasses import replace
from io import BytesIO
from importlib.util import find_spec, module_from_spec, spec_from_file_location
from pathlib import Path
from re import search
from subprocess import CalledProcessError, run
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image
from psd_tools import PSDImage
from psd_tools.constants import Resource, Tag
from psd_tools.psd import engine_data

import commands.psd_edit as pe
from commands.card_models import card_name
from commands.psd_regions import PAINT_ORDER, badge_layers, clip, fx_pad, regions, stat_rows
from conftest import (PLAIN_DOOMER, at_rev, bevel_textures, colour_glyphs, fork_reason, painted, rounds_bytes,
                      saves_preview, story_sync as _story_sync, text_engine as _text_engine)

# ========================
# Configuration
# ========================
VOCAB = Path(__file__).resolve().parent / "goldens" / "_vocab.json"
MINI_DOOMER = "Auxiliary/Minions/Mini_Doomer.psd"                 # template rows, no digit effects (certified golden)
HIVE = "Field/4 Stars/The_Hive.psd"                                # stoogedings badges, no overlay over the ability
MORPHO = "Auxiliary/Creatures/Morpho_Knight.psd"                  # badge runs, chart words and unstyled words on FX layers
HP_ROW = (360, 1200, 1173, 1284)                                   # its hp row: the digits' mask union
METTATON = "Auxiliary/Creatures/Mettaton_EX.psd"                  # a GradientOverlay on one digit per row (Atk: on hidden 6)
SPD_ROW = (359, 1577, 1174, 1663)                                  # its spd row, padded by the Stats Bars stroke
GREEN_MAGE = "Creatures/Everhood/4 Stars/Green_Mage.psd"          # fixed 2026-10-23: atk 6, its bars on the canvas
OFF_CANVAS_MAGE = "45cf8746^"   # Green Mage before its bars moved onto the card (atk 10: no digit visible, hidden 7 styled)
SHELL_SHOCK = "Auxiliary/Status Effects/Debuffs/3 Stars/Shell_Shock.psd"
MISMATCHED_SHOCK = "8af07bcd^"  # Shell_Shock before its re-save: its PNG another session's than its preview
PURE_VESSEL = "Creatures/Hollow Knight/5 Stars/Pure_Vessel.psd"   # digit masks start on the canvas and reach past it
AZAMOTH_KING = "Creatures/ROTMG/5 Stars/Azamoth_the_Forgotten_King.psd"   # hp 10 with 6, 8 and 9 styled: ambiguous
ANUBISATH = "Auxiliary/Minions/Anubisath_Guardian.psd"            # left-aligned title 11 px short of the first star
ASTR = "Auxiliary/Minions/Astr_Mionion.psd"                       # "Attacks inflict 1T3 Empowered." with Empowered in red
NAMED = ("Creatures/World Of Warcraft/5 Stars/The_Lich_King.psd", ANUBISATH,
         "Items/Support/2 Stars/Volt_Shroom.psd")                  # Text/Name + one visible Text/Ability, fonts in MDW/Fonts
CANNON = "Creatures/FTL/5 Stars/Royal_Eradicator_Main_Cannon.psd"  # the one 10+ card: atk 12 as Top (its 10 marker) + Bottom 2
GASOLINE = "Items/Support/2 Stars/Gasoline.psd"                    # an overlay "Blazing" type layer over the ability (OuterGlow)
FIRE_STAFF = "Items/Equip/2 Stars/Fire_Staff.psd"                  # "Ability FX: Smoldering": unstyled on the chart, nothing to cut
OVERLAY_STAFF = "0c573ed0^"                                        # Fire Staff before L7, its "Smoldering" an overlay text layer
RED_BLOON = "Auxiliary/Minions/PoD01Red_Bloon.psd"                 # PNG exported raw (unconverted) from a non-sRGB profile
BLOCK_HOPPER = "Creatures/Mario/2 Stars/Block_Hopper.psd"          # the same, its rows 18% bad when converted
BONE = "Creatures/Kirby/4 Stars/Wham_Bam_Bone.psd"                 # its title layer is named after the card
VIRUS = "Creatures/Kirby/3 Stars/Computer_Virus.psd"               # its Name layer sits at the root
MARX = "Creatures/Kirby/5 Stars/Marx.psd"                          # its Ability sits at the root, 'Ability FX: Glued' over it
EDEN = "Creatures/Isaac/2 Stars/Tainted_Eden.psd"                  # 'Name copy' and 'Name copy 2' print the title again
SOMETHING = "Creatures/Omori/3 Stars/SOMETHING.psd"                # two offset copies print the ability again
ZOMBOSS = "Creatures/Other/5 Stars/Doctor_Zomboss.psd"             # 'images/Name' prints the title again under the art
MOSLING = "Auxiliary/Minions/Mosling.psd"                          # a faint watermark 'Group 1/Name' prints the title again
AFRAID = "Auxiliary/Status Effects/Debuffs/2 Stars/Afraid.psd"     # no Text group: its title is the top-level layer 'Afraid'
JAWSUM = "Creatures/Omori/3 Stars/Mr_Jawsum.psd"                   # no Text group: its title 'Group 3/Group 2/Mr. Jawsum'
SOUL_MASTER = "Creatures/Hollow Knight/4 Stars/Soul_Master.psd"    # its title 'Soul Master' sits beside the Text group
PAPYRUS = "Creatures/Undertale/3 Stars/Papyrus.psd"                # its title layer 'Text/The Great Papyrus' is named after its text
SANS = "Creatures/Undertale/5 Stars/Sans.psd"                      # likewise 'Text/snas.'
KNIGHT = "Creatures/Everhood/2 Stars/Knight_Lost_A_Lot.psd"        # 'Knight', 'Lost', 'A', 'Lot' stacked: two in the title region
GLOBAL_TURN = "Auxiliary/Rulebook/08_Global_Turn.psd"              # a rule page: its heading 'Text/THE GLOBAL TURN'
FROSTMOURNE = "Auxiliary/Items/Frostmourne.psd"                    # four lines of plain ability text: an edit to one keeps three
PIZZA = "Creatures/Other/3 Stars/Pizza.psd"                        # three lines in three colours, each after a badge
COLOURED = "7fd2aa30"       # Pizza after the badges re-run: its first badge 18.66 pt, the others 32 (one badge size)
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


def _near(ours: np.ndarray, theirs: np.ndarray) -> float:
    """The share of our ink on their ink or its 1 px edge ring: a raster's edges may differ, a leaked badge may not."""
    ring = theirs.copy()
    ring[1:] |= theirs[:-1]
    ring[:-1] |= theirs[1:]
    ring[:, 1:] |= theirs[:, :-1]
    ring[:, :-1] |= theirs[:, 1:]
    return (ours & ring).sum() / ours.sum()


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
# the copy check composites the card: gradient fills need scipy (psd-tools[composite])
composites = pytest.mark.skipif(find_spec("scipy") is None, reason="gradient fills need psd-tools[composite] (scipy)")
# a text edit redraws only what changed (the fork's rasterize(keep=))
spliced = pytest.mark.skipif(not (_text_engine() and _story_sync() and pe.splices()),
                             reason="the fork redraws a text layer whole (no rasterize(keep=))")
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


PLAIN_POD = "71ddebec"      # PoD10BFB before the badges re-run's "Ability FX: (1GCD) | (2U) | (2CD)" layer


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


def _bake(clone, parser, rel, jobs, tmp_path, chart=None, update_preview=False, paths=None):
    psd, png = paths or _card(clone, rel)
    parsed = pe.parse_card(parser, psd, rel)
    assert pe.classify(parsed, jobs, PSDImage.open(psd)).jobs == jobs
    return parsed, pe.apply_jobs(psd, png, rel, jobs, parsed, parser, str(tmp_path), str(clone / pe.FONTS_DIR), chart,
                                 update_preview=update_preview)


# ========================
# Pure functions (any psd-tools)
# ========================
def test_text_blends_follow_the_fork():
    """A fork whose composite takes no per-layer text keywords renders its default and the straight mix, unpinned."""
    class Fork:
        def composite(self, viewport=None, ignore_preview=False, apply_icc=True, text_gamma=1.45):
            return None

    assert pe.text_blends(Fork()) == ["preview", "straight"]
    assert pe.text_kwargs(Fork(), "preview", (7,)) == {}
    assert pe.edited_ids([]) == ()


def test_parity_bar():
    """The owner's bar (2026-10-22): mean |diff| <= 2.0 (was 1.0), at most 0.5% of pixels past 24. Missing it is told, not
    refused (2026-10-23); only past the floor, more than 25% of the region's pixels past 24, is a render broken."""
    assert (pe.PARITY_MEAN_MAX, pe.PARITY_BAD_MAX, pe.PARITY_THRESH, pe.PARITY_FLOOR) == (2.0, 0.5, 24, 25.0)
    assert pe._within_bar({"bad_pct": 0.5, "mean": 2.0, "max": 200})
    assert not pe._within_bar({"bad_pct": 0.0, "mean": 2.01, "max": 7})
    assert not pe._within_bar({"bad_pct": 0.51, "mean": 0.5, "max": 7})
    assert not pe._past_floor({"bad_pct": 25.0, "mean": 90.0, "max": 255})
    assert pe._past_floor({"bad_pct": 25.01, "mean": 1.0, "max": 30})


def test_miss_notes():
    """The owner's note per edited box past the bar (2026-10-23); the re-render's only when the baseline was within it."""
    ok, miss = {"bad_pct": 0.0, "mean": 0.8, "max": 7}, {"bad_pct": 1.9725, "mean": 1.5955, "max": 217}
    note = "baked with a pixel difference: bad 1.97%, mean 1.60, max 217"
    assert pe.miss_notes(SimpleNamespace(parity={"hp": ok}, rerender={"hp": ok})) == []
    assert pe.miss_notes(SimpleNamespace(parity={"name": miss}, rerender={"name": miss})) == [f"name {note}"]
    assert pe.miss_notes(SimpleNamespace(parity={"hp": ok, "spd": ok}, rerender={"spd": miss})) == [
        "spd baked with a pixel difference (re-render): bad 1.97%, mean 1.60, max 217"]
    assert pe.miss_notes(SimpleNamespace(parity={"hp": miss, "spd": ok})) == [f"hp {note}"]     # a Baked without rerender


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


def test_keep_ends_puts_back_blank_first_and_last_paragraphs():
    """The website and Discord trim an ability's ends: the layer's blank first and last paragraphs come back (PoD10BFB)."""
    old = "\r(1GCD)(2U) Intercept.\r(2CD) Summon."
    assert pe.keep_ends(old, "(1GCD)(2U) Intercept.\r(2CD) Summon!") == "\r(1GCD)(2U) Intercept.\r(2CD) Summon!"
    assert pe.keep_ends(" \r\x03 Hit.  \r\r", "Hit hard.") == " \r\x03Hit hard.\r\r"     # the paragraphs, not the indent
    assert pe.keep_ends("Hit.  ", "Hit hard.") == "Hit hard." and pe.keep_ends("Hit.", "Hit hard.") == "Hit hard."
    assert pe.keep_ends(old, "\rAlready.") == "\rAlready." and pe.keep_ends("\r", "New.") == "New."
    assert pe.keep_ends(old, "") == ""


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
    spans = [(s, e, c.name) for s, e, c in pe.status_spans(text, chart, [])]
    assert spans == [(6, 12, "Poison"), (17, 31, "Frozen In Time"), (37, 43, "Frozen")]
    broken = "Inflict Frozen\x03In Time."           # a forced break between its words, as Effect Layers.jsx reads it
    assert [(s, e, c.name) for s, e, c in pe.status_spans(broken, chart, [])] == [(8, 22, "Frozen In Time")]


def test_status_spans_skip_tokens(parser):
    """Review 2026-10-24: "Stuffed" matched inside the token "[stuffed star]" ("[" and "_" are no letters), so the bake set
    the Stuffed swatch on the icon's characters (shown on a :tint or :sticker one). What the icon font draws holds no
    status word; brackets it draws as letters do ("[Blazing]" baked unstyled, where Effect Layers.jsx styles it: review
    2026-10-24)."""
    chart = {"stuffed": pe.StatusColor("Stuffed", (0.9, 0.6, 0.2), False),
             "blazing": pe.StatusColor("Blazing", (1.0, 0.5, 0.0), True)}
    text = "Gain [stuffed star] or [Stuffed_Star:tint], then Stuffed."
    start = text.index("Stuffed.")
    icons = pe.icon_text(text, parser.icons)
    assert len(icons) == 2 and [(s, e, c.name) for s, e, c in pe.status_spans(text, chart, icons)] == [(start, start + 7, "Stuffed")]
    for text in ("Inflict [Blazing].", "Inflict [Blazing, Frozen or Burning]."):
        assert pe.icon_text(text, parser.icons) == []
        assert [(s, e, c.name) for s, e, c in pe.status_spans(text, chart, [])] == [(9, 16, "Blazing")]

def test_fx_names_follow_the_jsx():
    assert pe.fx_name("Ability", "Blazing") == "Ability FX: Blazing" and pe.fx_name("Ability", "Frozen In Time", 2) == "Ability FX: Frozen In Time#2"
    assert pe.FX_TAG == pe.EFFECT_LAYER_TAG == " FX: " and pe.ALLOWED_RECORD_FIELDS["overlay"] == {"flags"}


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


FIT_CHART = {"frozen": pe.StatusColor("Frozen", (0.6, 1.0, 0.95), True),
             "frozen in time": pe.StatusColor("Frozen In Time", (0.75, 0.76, 0.95), True)}


def _status_plan(name: str, old: str) -> pe.FxPlan:
    """A "status" FxPlan as plan_fx makes it, from an FX layer's name and the old text (no PSD)."""
    phrases = [phrase.strip() for phrase in name.split(pe.FX_TAG, 1)[1].split("|")]
    status = pe.collapse(pe._numbered(phrases[0])[0]).lower()
    return pe.FxPlan(SimpleNamespace(name=name), phrases, "status", None, (0, 0, 1, 1), pe.phrase_ranges(old, phrases),
                     [(phrase, pe.phrase_ranges(old, [phrase])) for phrase in phrases], status)


def _follow(name: str, old: str, new: str) -> list:
    return pe.follow_fx(_status_plan(name, old), new, pe.kept_chars(old, new), pe.status_spans(new, FIT_CHART, []))


def test_numbered_status_fx_follows_its_own_word():
    """Review (q6): Standstill's 'Frozen In Time#1' and '#2' keep their own style, so each stays on its own word, as a
    "styled" layer does: "Inflict 2T Frozen In Time." made "Frozen In Time, Frozen In Time." renumbers #2 (line 2, not
    edited) to #3 and #1 to the occurrence it kept; the new one gets a chart-style layer (new_fx). The JSX's literal
    names would move #2 up to line 1 and give line 2's word the chart's style."""
    old = "Inflict 2T Frozen In Time.\rOnly inflicts 1T Frozen In Time to 5 star or higher creatures."
    new = old.replace("Inflict 2T Frozen In Time.", "Frozen In Time, Frozen In Time.")
    first, second, third = pe.phrase_ranges(new, ["Frozen In Time"])
    assert _follow("Ability FX: Frozen In Time#2", old, new) == [
        ("Frozen In Time#2", "Frozen In Time#3", [third], [third], "its Frozen In Time is #3 now")]
    assert _follow("Ability FX: Frozen In Time#1", old, new) == [
        ("Frozen In Time#1", "Frozen In Time#2", [second], [second], "its Frozen In Time is #2 now")]
    fit = FIT_CHART["frozen in time"]
    assert pe.new_fx(new, pe.status_spans(new, FIT_CHART, []), [second, third]) == [("Frozen In Time#1", [first], fit)]


def test_status_fx_lands_only_on_its_status_words():
    """Review: a status layer re-cut wherever the JSX's phrase lands took the "Frozen" of "Frozen In Time" (Snow Bunny's
    "Frozen." made "Frozen In Time.": no Frozen In Time layer, " In Time" left flat) or of "Unfrozen" (a glow on part of
    a word). It lands on its status's own words only: gone from the first ("'Frozen In Time' is no Frozen") so new_fx
    makes the Frozen In Time layer; renamed "Frozen#2" when a Frozen of its own stays; gone from "Unfrozen"."""
    old = "Heal 5 HP, inflict 1T Frozen."
    gone = [("Frozen", None, [], [], "'Frozen In Time' is no Frozen")]
    new = old.replace("Frozen.", "Frozen In Time.")
    assert _follow("Ability FX: Frozen", old, new) == gone
    fit, at = FIT_CHART["frozen in time"], new.index("Frozen")
    assert pe.new_fx(new, pe.status_spans(new, FIT_CHART, []), []) == [("Frozen In Time", [(at, at + 14)], fit)]
    new = old.replace("Frozen.", "Frozen In Time or Frozen.")
    mine = (new.rindex("Frozen"), new.rindex("Frozen") + 6)
    assert _follow("Ability FX: Frozen", old, new) == [("Frozen", "Frozen#2", [mine], [], gone[0][4])]   # typed, not kept
    assert _follow("Ability FX: Frozen", old, old.replace("Frozen.", "Unfrozen.")) == [
        ("Frozen", None, [], [], "'Unfrozen' is no Frozen")]


def test_new_fx_counts_a_word_covered_only_when_held_whole():
    """Review: a styled status word counts as covered only when a rebuilt layer's range holds all of it (a "styled"
    phrase around it, Clot Gold's "1 Become Back Money", or its own layer); one a range overlaps in part refuses, where
    the overlap rule left " In Time" of "Frozen In Time" without a layer."""
    text = "Heal 5 HP, inflict 1T Frozen In Time."
    spans, at = pe.status_spans(text, FIT_CHART, []), text.index("Frozen")
    assert pe.new_fx(text, spans, [(at - 3, at + 14)]) == [] and pe.new_fx(text, spans, [(at, at + 14)]) == []
    with pytest.raises(pe.Refused, match="unsupported-card: 'Frozen In Time' is partly under the FX layer of 'Frozen'") as refused:
        pe.new_fx(text, spans, [(at, at + 6)])
    assert refused.value.code == "unsupported-card"


def test_kept_spans_need_every_character_kept():
    """Review: a range counts as kept (held, _kept) only when every character of it is, in a row: "Gold" retyped "Goxd"
    keeps its ends, not the word."""
    kept = pe.kept_chars("Clot Gold", "Clot Goxd")
    assert pe.kept_spans([(0, 4), (5, 9)], kept) == [(0, 4)]
    plan = pe.FxPlan(None, ["Gold"], "styled", None, (0, 0, 1, 1), [(5, 9)])
    assert not pe._kept(plan, [(5, 9)], None, kept)


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
    two offset copies of its ability): a bake would change one and leave the copies. An empty layer has no copies
    (Anubisath's empty Ability beside an empty 'Layer 1')."""
    with pytest.raises(pe.Refused, match="the title is printed again by 'Name copy'") as refused:
        pe.name_layer(PSDImage.open(_card(clone, EDEN)[0]), card_name(EDEN))
    assert refused.value.code == "unsupported-card"
    with pytest.raises(pe.Refused, match="the ability is printed again by 'Attacks are piercing") as refused:
        pe.ability_layer(PSDImage.open(_card(clone, SOMETHING)[0]))
    assert refused.value.code == "unsupported-card"
    assert pe.ability_layer(PSDImage.open(_card(clone, ANUBISATH)[0])).text == ""


@composites
def test_a_copy_the_art_hides_does_not_refuse(clone):
    """Only a copy people can see refuses: Doctor Zomboss's 'images/Name' sits under the art (the card renders the same
    without it), so its Text/Name bakes; Mosling's faint watermark still shows and refuses."""
    zomboss = PSDImage.open(_card(clone, ZOMBOSS)[0])
    layer = pe.name_layer(zomboss, card_name(ZOMBOSS))
    assert layer.parent is pe._text_group(zomboss) and pe.read_name(zomboss, card_name(ZOMBOSS)) == "Doctor Zomboss"
    hidden = next(other for other in zomboss.descendants() if other.name == "Name" and other.parent.name == "images")
    assert hidden.is_visible() and pe.collapse(hidden.text) == "Doctor Zomboss" and not pe._shows(zomboss, hidden)
    with pytest.raises(pe.Refused, match="the title is printed again by 'Name'"):
        pe.name_layer(PSDImage.open(_card(clone, MOSLING)[0]), card_name(MOSLING))


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


@text_engine
def test_a_digit_outside_the_card_refuses(clone):
    """One digit's mask wholly off the canvas, the rest of the bar on it (no card today): hiding only that digit (5 -> 10) gave
    an empty region (a ValueError at the render, an `error`); it refuses `unsupported-card` naming the digit, before a layer changes.
    Moving the mask needs the fork's Mask.offset setter (upstream's mask has no offset)."""
    psd = PSDImage.open(_card(clone, MINI_DOOMER)[0])
    row = pe.template_rows(psd, "hp")[0]
    five = row.digits["5"]
    five.mask.offset = (five.mask.left, psd.height + 10)          # below the canvas
    with pytest.raises(pe.Refused, match=r"^unsupported-card: hp: digit 5 of the bar 'Hp Dark' lies outside the card \(digit "
                                         r"masks \(764, 2410, 1173, 2494\) on the 1200x2400 canvas\)"):
        pe.set_stat(psd, {"card": {"hp": 5}}, "hp", 10)
    assert row.visible == [5] and five.visible


def test_layer_records_fingerprint(clone):
    psd = _card(clone, MINI_DOOMER)[0]
    records = pe.layer_records(psd)
    assert len(records) == 96 and records[93]["name"] == "Name" and records[71]["name"] == "5"
    assert records[95]["name"] == "Ability FX: (BOT)"
    assert pe.gate_bytes(psd, psd, []) == ([], 0, 0)


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


@round4
def test_raw_export_bakes_unconverted(clone, parser, tmp_path):
    """PoD01Red_Bloon's PNG holds the PSD's own values: converted, its hp row is 12 levels off (mean 12.1, refused); rendered
    raw it passes (mean 0.58, 0% bad on 2026-10-22) and bakes, the region re-rendered raw for idempotence."""
    psd_path, png_path = _card(clone, RED_BLOON)
    parsed, baked = _bake(clone, parser, RED_BLOON, {"hp": 2}, tmp_path)
    assert baked.icc is False and baked.reparse["card"]["hp"] == 2
    assert baked.parity["hp"]["bad_pct"] == 0 and baked.parity["hp"]["mean"] <= pe.PARITY_MEAN_MAX and pe.miss_notes(baked) == []
    _, converted = pe.baseline(PSDImage.open(psd_path), Image.open(png_path), baked.edits[0].region)
    assert converted["mean"] > 10 and not pe._within_bar(converted)    # a miss: it would bake, told


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
    lfx2 and lrFX blocks (5884 bytes) leaving record 5 for record 8, nothing else (a gradient tiles nothing: fxrp stays); the dark cells render within the bar
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
    passes parity (mean 0.556, max 7, 0% bad on 2026-10-22); truncated (mean 1.03) it refused on the mean until the bar
    went to 2.0 that day."""
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
    _, measured = pe.baseline(PSDImage.open(psd_path), png, edit.region)
    assert measured["bad_pct"] == 0 and measured["mean"] <= pe.PARITY_MEAN_MAX and measured["max"] <= 7
    assert (measured["mean"] < 1.0) == bool(rounds_bytes())


@round4
def test_cannon_rows_pass_parity(clone, parser, tmp_path):
    """The Cannon's digits carry ColorOverlay + a textured BevelEmboss. With round 4 every row is 0% bad (2026-10-22):
    the two-row atk bakes (mean 0.51), and so does the one-row hp, refused on the mean alone (1.06) until the owner set the
    bar to 2.0 that day."""
    for stat in ("atk", "hp"):
        (tmp_path / stat).mkdir()
    parsed, baked = _bake(clone, parser, CANNON, {"atk": 15}, tmp_path / "atk")
    assert baked.reparse["card"]["atk"] == 15 and baked.reparse["validator"] == parsed["validator"]   # STATS TOO HIGH: atk 12 is
    assert baked.parity["atk"]["bad_pct"] == 0 and baked.parity["atk"]["mean"] <= pe.PARITY_MEAN_MAX
    parsed, baked = _bake(clone, parser, CANNON, {"hp": 2}, tmp_path / "hp")
    assert baked.reparse["card"]["hp"] == 2 and baked.parity["hp"]["bad_pct"] == 0
    assert 1.0 < baked.parity["hp"]["mean"] <= pe.PARITY_MEAN_MAX


@round4
def test_cannon_toggle_keeps_the_texture_anchor(clone):
    """The bevel texture tiles from the styled digit's fxrp; the Cannon's digits sit 138-160 px apart, off its 64 px pattern.
    hp 9 -> 8 trades fxrp with the effect, so cells 9-10, lit before and after, render as the PNG shows them (0% bad on
    2026-10-22; 8.1% bad, max 128, with 8's own fxrp); 8 -> 9 gives both records their blocks back."""
    psd_path, png_path = _card(clone, CANNON)
    png = Image.open(png_path)
    psd = PSDImage.open(psd_path)
    row = pe.template_rows(psd, "hp")[0]
    before = {name: {bytes(k): v.tobytes() for k, v in row.digits[name].tagged_blocks.items()} for name in ("8", "9")}
    edit = pe.set_stat(psd, {"card": {"hp": 9}}, "hp", 8)
    lit = (row.digits["9"].mask.bbox[0] + 3, *edit.region[1:])     # past the bevel on 9's old left edge (2 px)
    measured = pe._measure(pe.render_region(psd, lit), png.crop(lit))
    assert measured["bad_pct"] == 0 and measured["mean"] <= pe.PARITY_MEAN_MAX
    pe.set_stat(psd, {"card": {"hp": 8}}, "hp", 9)
    assert {name: {bytes(k): v.tobytes() for k, v in row.digits[name].tagged_blocks.items()} for name in ("8", "9")} == before


def test_toggle_round_trip_keeps_block_order(clone, tmp_path):
    """Moved blocks land where they sat on their source and a traded fxrp stays in place: the Cannon's hp 9 -> 8 gives 8
    9's block order, and 8 -> 9 saves both records byte for byte as they were (fxrp used to move after lfx2/lrFX)."""
    path = _card(clone, CANNON)[0]
    psd = PSDImage.open(path)
    row = pe.template_rows(psd, "hp")[0]
    order = {name: [bytes(key) for key in row.digits[name].tagged_blocks.keys()] for name in ("8", "9")}
    assert order["9"] == order["8"][:1] + [b"lfx2", b"lrFX"] + order["8"][1:] and order["8"][-1] == b"fxrp"
    pe.set_stat(psd, {"card": {"hp": 9}}, "hp", 8)
    assert [bytes(key) for key in row.digits["8"].tagged_blocks.keys()] == order["9"]
    assert [bytes(key) for key in row.digits["9"].tagged_blocks.keys()] == order["8"]
    pe.set_stat(psd, {"card": {"hp": 8}}, "hp", 9)
    saved = str(tmp_path / "cannon_round_trip.psd")
    pe.save_psd(psd, saved, update_preview=False)
    assert pe.gate_bytes(path, saved, []) == ([], 0, 0)


@before_textures
def test_cannon_rows_bake_with_a_note_without_bevel_textures(clone, parser, tmp_path):
    """Before round 4 the fork skips the digits' BevelEmboss texture (5-7% wrong, max 110): refused `parity` until
    2026-10-23, now baked under the floor with the miss told, for the two-row atk as for its one-row hp."""
    for stat, value in (("atk", 15), ("hp", 2)):
        (tmp_path / stat).mkdir()
        _, baked = _bake(clone, parser, CANNON, {stat: value}, tmp_path / stat)
        assert pe.PARITY_BAD_MAX < baked.parity[stat]["bad_pct"] <= pe.PARITY_FLOOR and baked.reparse["card"][stat] == value
        assert pe.miss_notes(baked)[0].startswith(f"{stat} baked with a pixel difference: bad ")


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
@pytest.mark.parametrize("rel, record", [(BONE, "Wham Bam Bone"), (VIRUS, "Name"), (ZOMBOSS, "Name"), (AFRAID, "Afraid"),
                                         (JAWSUM, "Mr. Jawsum")])
def test_name_edit_on_the_title_the_parser_reads(clone, parser, chart, rel, record, tmp_path):
    """A title named after the card (Wham Bam Bone; Afraid's and Mr Jawsum's outside a Text group), a Name layer at the
    root (Computer Virus) or a Text/Name with a copy under the art (Doctor Zomboss) bakes through every gate: the parser
    reads the new title back and nothing is left to bake."""
    psd_path = _card(clone, rel)[0]
    new = _typo_fix(pe.read_name(PSDImage.open(psd_path), card_name(rel)))
    parsed, baked = _bake(clone, parser, rel, {"name": new}, tmp_path, chart)
    assert [name for _, name, _ in baked.records] == [record]
    assert {"blocks", "channels"} <= set(baked.records[0][2]) <= set(TEXT_RECORD)     # rect: only when the box moved
    assert baked.reparse["card"]["display_name"] == new
    saved = PSDImage.open(baked.psd_path)
    assert pe.read_name(saved, card_name(rel)) == new and pe.stale_stories(saved) == []
    assert pe.classify(baked.reparse, {"name": new}, saved).redundant == ["name"]


@text_edit
@pytest.mark.parametrize("rel", [PAPYRUS, SANS])
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
def test_ability_bake_at_the_root_keeps_its_fx_layer_above_it(clone, parser, chart, tmp_path):
    """Marx's root Ability bakes through every gate; 'Ability FX: Glued' stays where it is, above the Ability layer and
    its hidden "Glued" overlay at the root (Effect Layers.jsx's place for a root layer, not above Text): its word was not
    redrawn, so its record stays whole."""
    psd_path = _card(clone, MARX)[0]
    new = pe.normalize_ability(pe.parse_card(parser, psd_path, MARX)["card"]["ability"]).replace("Glued", "Glued at once", 1)
    parsed, baked = _bake(clone, parser, MARX, {"ability": new}, tmp_path, chart)
    assert baked.edits[0].fx_names == ["Ability FX: Glued"] and pe.normalize_ability(baked.reparse["card"]["ability"]) == new
    assert baked.fx == [] and "Ability FX: Glued" not in [name for _, name, _ in baked.records]
    saved = PSDImage.open(baked.psd_path)
    layer = pe.ability_layer(saved)
    fx, = pe.fx_layers(saved, layer)
    old = PSDImage.open(psd_path)
    assert layer.parent is saved and saved.index(fx) == old.index(pe.fx_layers(old, pe.ability_layer(old))[0]) == saved.index(layer) + 2


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


def test_effect_layer_numbers_read_as_the_jsx(parser):
    """"#0" is every occurrence, as Effect Layers.jsx's splitOccurrence reads it and _numbered too; the parser's EFFECT
    LAYER WORDS NOT IN TEXT reads it as the first, the same count (psd_parser left as it was, review 2026-10-23)."""
    text = "Inflict Frozen and Frozen."
    assert pe._numbered("Frozen#0") == ("Frozen", 0) and pe.phrase_ranges(text, ["Frozen#0"]) == pe.phrase_ranges(text, ["Frozen"])
    base = SimpleNamespace(name="Ability", bbox=(0, 0, 100, 100), engine_dict={"Editor": {"Text": SimpleNamespace(value=text)}})
    for name, missing in (("Frozen#0", False), ("Frozen#2", False), ("Frozen#3", True), ("Blazing#0", True), ("Blazing", True)):
        card = SimpleNamespace(problems=[])
        parser._check_effect_layer(SimpleNamespace(name="Ability FX: " + name, bbox=(0, 0, 9, 9), is_visible=lambda: True),
                                   [base], card)
        assert card.problems == ([f"EFFECT LAYER WORDS NOT IN TEXT: Ability FX: {name}"] if missing else []), name


def test_phrase_ranges_follow_the_jsx():
    """Effect Layers.jsx's phraseRanges: any case, any blanks between words, whole words first, else every match, #n the
    nth; the parser's EFFECT LAYER WORDS NOT IN TEXT counts alike."""
    goofy = "Attacks inflict either 3T2 Smeckeldorfed or 3T1 Become Back Money.\r(1CD) Draw"
    assert pe.phrase_ranges(goofy, ["1#1"]) == [(46, 47)] and pe.phrase_ranges(goofy, ["become back money"]) == [(48, 65)]
    assert pe.phrase_ranges(goofy, ["1"]) == [(46, 47), (68, 69)] and pe.phrase_ranges(goofy, ["1#3", "Poison"]) == []
    text = "Blazing blazingly blazing\x03Frozen In  Time"
    assert pe.phrase_ranges(text, ["Blazing", "frozen in time", "Blazing#2"]) == [(0, 7), (18, 25), (26, 41), (18, 25)]
    assert pe.phrase_ranges("Blazingly", ["Blazing"]) == [(0, 7)]


def test_held_words_follow_the_layer_not_its_name():
    """Edit.held (review 2026-10-23) by layer: an "own" or "styled" layer renamed onto a deleted one's old name ("2T#2" ->
    "2T#1" once the first "2T" left) is still held to the words it kept (compared by name, it passed for the deleted
    one and gate_fx left it unchecked); a deleted layer and a badge layer are not held."""
    gone, moved, badge = (SimpleNamespace(name=name) for name in ("Ability FX: 2T#1", "Ability FX: 2T#1", "Ability FX: (1CD)"))
    plans = [pe.FxPlan(gone, ["2T#1"], "own", ranges=[(8, 10)]), pe.FxPlan(moved, ["2T#2"], "own", ranges=[(20, 22)]),
             pe.FxPlan(badge, ["(1CD)"], "badge", ranges=[(0, 5)])]
    kept = [-1 if 8 <= i < 12 else i - 4 * (i >= 12) for i in range(30)]
    assert pe.held_words(plans, kept, [moved, badge]) == [("Ability FX: 2T#1", [(16, 18)])]


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
def test_ambiguous_fx_layers_refuse(clone, chart):
    """The refusals no card reaches today, on Morpho Knight and Mini Doomer altered in memory: an FX layer whose words
    now fall in a badge (an own layer named "CD", the letters of its "(1CD)": the JSX would cut every badge into it), two
    FX layers that would end with one name (+2/4/6 ATK renamed a second Smoldering), and an FX layer that is no pixel
    layer (a type layer named like one: Photoshop would draw it from its text). Each unsupported-card."""
    fonts = str(clone / pe.FONTS_DIR)

    def morpho(rename: dict):
        psd = PSDImage.open(_card(clone, MORPHO)[0])
        layer = pe.ability_layer(psd)
        for fx in pe.fx_layers(psd, layer):
            fx.name = rename.get(fx.name, fx.name)
        return psd, pe.normalize_ability(layer.text) + " Die."

    for rename, why in (({"Ability FX: Burning": "Ability FX: CD"}, "Ability FX: CD: its words now fall in a badge or token"),
                        ({"Ability FX: +2/4/6 ATK": "Ability FX: Smoldering"},
                         "two FX layers would be named Ability FX: Smoldering")):
        psd, text = morpho(rename)
        with pytest.raises(pe.Refused, match="^unsupported-card: " + why + "; edit in Photoshop$"):
            pe.set_text_layer(psd, "ability", text, fonts, chart)
    psd = PSDImage.open(_card(clone, MINI_DOOMER)[0])
    pe.name_layer(psd).name = "Ability FX: Die"
    layer = pe.ability_layer(psd)
    with pytest.raises(pe.Refused, match="^unsupported-card: FX layers that are no pixel layers: Ability FX: Die; edit in "
                                         "Photoshop$"):
        pe.set_text_layer(psd, "ability", pe.normalize_ability(layer.text) + " Die.", fonts, chart)


@text_edit
def test_a_typed_status_word_joining_its_layer_is_told(clone, chart):
    """Snow Bunny's own-style 'Ability FX: Frozen' (q6) covers every Frozen of its text, as Update all cuts it: a typed
    Frozen joins it, drawn in that layer's style rather than the chart's, and the report says so (review 2026-10-23)."""
    psd = PSDImage.open(_card(clone, SNOW_BUNNY)[0])
    layer = pe.ability_layer(psd)
    edit = pe.set_text_layer(psd, "ability", layer.text.replace("Heal 5 HP.", "Heal 5 HP, Frozen."),
                             str(clone / pe.FONTS_DIR), chart)
    assert edit.fx_names == [FROZEN] and edit.fx == [] and edit.notes == [f"{FROZEN} also covers the typed 'Frozen', in its own style"]


@text_edit
def test_own_style_fx_layers_are_made_again_in_their_style(clone, parser, chart, tmp_path):
    """Owner 2026-10-23: FX layers in no chart status's style are rebuilt in their own style, as the others (the 2026-10-21
    refusal went). Morpho Knight's Burning, +2/4/6 ATK and Smoldering strokes bake an appended sentence through every
    gate: each record kept whole or cut again (rect and channels), its style bytes unchanged, no new layer. Without the
    chart the card still refuses (its FX words can't be told apart)."""
    psd_path = _card(clone, MORPHO)[0]
    psd = PSDImage.open(psd_path)
    layer = pe.ability_layer(psd)
    text, fx = pe.normalize_ability(layer.text), [f.name for f in pe.fx_layers(psd, layer)]
    assert [pe.fx_kind(f, layer, chart) for f in pe.fx_layers(psd, layer)] == ["badge", "own", "status", "own", "own"]
    with pytest.raises(pe.Refused, match="status word layers over the ability"):
        pe.set_text_layer(psd, "ability", text + " Die.", str(clone / pe.FONTS_DIR), None)
    old = pe.normalize_ability(pe.parse_card(parser, psd_path, MORPHO)["card"]["ability"])
    parsed, baked = _bake(clone, parser, MORPHO, {"ability": old + " Die."}, tmp_path, chart)
    edit, = baked.edits
    assert edit.fx_names == fx and edit.fx == [] and edit.dropped == [] and baked.block_bytes == 0
    for name in fx:
        assert _style(baked.psd_path, name) == _style(psd_path, name)
    assert all(fields in (["channels", "rect"],) for _, name, fields in baked.records if name in fx)


def _noop(clone, parser, chart, rel, tmp_path):
    """apply_jobs with the ability layer's own text (classify would call it redundant): (psd path, png path, Baked)."""
    psd_path, png_path = _card(clone, rel)
    text = pe.ability_layer(PSDImage.open(psd_path)).text.replace("\r", "\n")
    parsed = pe.parse_card(parser, psd_path, rel)
    return psd_path, png_path, pe.apply_jobs(psd_path, png_path, rel, {"ability": text}, parsed, parser, str(tmp_path),
                                             str(clone / pe.FONTS_DIR), chart, update_preview=False)


def _same_png(a, b) -> bool:
    return np.array_equal(np.asarray(Image.open(a).convert("RGBA")), np.asarray(Image.open(b).convert("RGBA")))


@text_edit
def test_noop_ability_bake_keeps_the_badge_layer(clone, parser, chart, tmp_path):
    """BUG 2 on Mini Doomer's gradient badge layer: a no-op ability bake passes every gate (gate_fx too) and keeps
    'Ability FX: (BOT)' byte for byte, none of its words redrawn, the PNG pixel for pixel. Cut again by force from the
    unchanged raster, it is the layer Effect Layers.jsx cut, byte for byte: the key takes the badge's colour glyph on
    purpose (its phrase is a badge run)."""
    psd_path, png_path, baked = _noop(clone, parser, chart, MINI_DOOMER, tmp_path)
    edit, = baked.edits
    assert edit.fx_names == ["Ability FX: (BOT)"] and (edit.rebuilt, edit.dropped, edit.fx, edit.notes) == ([], [], [], [])
    assert "Ability FX: (BOT)" not in [name for _, name, _ in baked.records] and _same_png(png_path, baked.png_path)
    from psd_tools.composite.text import FontMap
    psd = PSDImage.open(psd_path)
    layer = pe.ability_layer(psd)
    old, = pe.fx_layers(psd, layer)
    box, before = tuple(old.bbox), _raster(old, old.bbox)
    plans = pe.plan_fx(layer, [old], chart)
    fonts = FontMap(dirs=[str(clone / pe.FONTS_DIR)])
    kept = pe.kept_chars(layer.text, layer.text)
    assert pe._kept(plans[0], plans[0].ranges, None, kept) and not pe._kept(plans[0], plans[0].ranges, tuple(layer.bbox), kept)
    rebuilt, dropped, covered, notes = pe._rebuild_fx(psd, layer, layer.text, plans, tuple(layer.bbox), kept, fonts,
                                                      pe._fx_engine(), [pe.follow_fx(plans[0], layer.text, kept)])
    assert rebuilt == [(old, "Ability FX: (BOT)")] and dropped == notes == []
    assert covered == [(layer.text.index("(BOT)"), layer.text.index("(BOT)") + 5)]
    assert tuple(old.bbox) == box and np.array_equal(_raster(old, box), before)


@text_edit
def test_ability_bake_moves_the_badge_layer_with_its_badge(clone, parser, chart, tmp_path):
    """BUG 2: Mini Doomer's centred "(BOT) Die." made "(BOT) Then die at once." moves the badge left; its FX layer is cut
    again in place from the new raster (rect and channels, its record otherwise whole) over the badge's new place, every
    gate passing: the parse, the validator's coverage, gate_fx's glyph boxes and the Ability's own pixels."""
    psd_path = _card(clone, MINI_DOOMER)[0]
    old = PSDImage.open(psd_path)
    before, = pe.fx_layers(old, pe.ability_layer(old))
    text = pe.normalize_ability(pe.parse_card(parser, psd_path, MINI_DOOMER)["card"]["ability"]).replace("Die.", "Then die at once.")
    parsed, baked = _bake(clone, parser, MINI_DOOMER, {"ability": text}, tmp_path, chart)
    edit, = baked.edits
    (_, ability, fields), (_, name, fx_fields) = baked.records
    assert ability == "Ability" and {"blocks", "channels"} <= set(fields) <= set(TEXT_RECORD)
    assert (name, fx_fields) == ("Ability FX: (BOT)", ["channels", "rect"])
    assert baked.fx == edit.fx_names == ["Ability FX: (BOT)"] and edit.rebuilt[0][1:] == ("Ability FX: (BOT)", "Ability FX: (BOT)")
    saved = PSDImage.open(baked.psd_path)
    layer = pe.ability_layer(saved)
    fx, = pe.fx_layers(saved, layer)
    assert fx.bbox[0] < before.bbox[0] and fx.bbox[1] == before.bbox[1] and fx.layer_id == before.layer_id
    assert saved.index(fx) == old.index(before) and pe.PSDParser._pixel_coverage(fx, layer) == 1.0
    assert baked.reparse["validator"] == [] and baked.region[0] <= fx.bbox[0] and baked.region[2] >= before.bbox[2]


SNOW_BUNNY = "Items/Heal/2 Stars/Snow_Bunny.psd"   # L7: 'Ability FX: Frozen' keeps its old overlay's style, not the chart's
FROZEN = "Ability FX: Frozen"


def _style(path: str, name: str) -> tuple:
    """An FX record's style bytes: blending, opacity, clipping, flags and every tagged block (lfx2, lrFX, iOpa...)."""
    record, = [r for r in pe.layer_records(path) if r["name"] == name]
    return record["blend"], record["flags"], record["blocks"]


@text_edit
def test_noop_ability_bake_keeps_an_l7_fx_layers_own_style(clone, parser, chart, tmp_path):
    """Owner q6 (unattended call 2026-10-23): an ability bake keeps each FX layer's own style. Snow Bunny's 'Ability FX:
    Frozen' draws its old overlay's style, not the chart's Frozen (L7): a no-op bake keeps its record whole, tells no
    restyle, and leaves the PNG pixel for pixel (restyled to the chart, 21,447 px went past 24)."""
    psd = PSDImage.open(_card(clone, SNOW_BUNNY)[0])
    fx, = pe.fx_layers(psd, pe.ability_layer(psd))
    assert fx.name == FROZEN and pe._style_key(fx.tagged_blocks) != pe._style_key(chart["frozen"].blocks)
    psd_path, png_path, baked = _noop(clone, parser, chart, SNOW_BUNNY, tmp_path)
    edit, = baked.edits
    assert edit.fx_names == [FROZEN] and edit.notes == [] and baked.block_bytes == 0
    assert _style(baked.psd_path, FROZEN) == _style(psd_path, FROZEN) and _same_png(png_path, baked.png_path)


@text_edit
def test_rebuilt_fx_layer_keeps_its_own_style(clone, parser, chart, tmp_path):
    """Owner q6: an FX layer cut again keeps its own style; only a new one takes the chart's. Snow Bunny's "at least "
    deleted moves its Frozen: the layer is cut again at its new place (rect and channels), its style bytes kept, and
    gate_bytes refuses the chart's style put on it (no effect block may change on a rebuilt record)."""
    psd_path = _card(clone, SNOW_BUNNY)[0]
    text = pe.normalize_ability(pe.parse_card(parser, psd_path, SNOW_BUNNY)["card"]["ability"]).replace("at least ", "")
    parsed, baked = _bake(clone, parser, SNOW_BUNNY, {"ability": text}, tmp_path, chart)
    edit, = baked.edits
    assert edit.notes == [] and _style(baked.psd_path, FROZEN) == _style(psd_path, FROZEN) and baked.block_bytes == 0
    assert dict((name, fields) for _, name, fields in baked.records)[FROZEN] == ["channels", "rect"]
    assert [record[1:] for record in edit.rebuilt] == [(FROZEN, FROZEN)]
    saved = PSDImage.open(baked.psd_path)
    fx, = pe.fx_layers(saved, pe.ability_layer(saved))
    for key in pe.EFFECT_BLOCKS:
        if key in fx.tagged_blocks:
            del fx.tagged_blocks[key]
    for key, block in chart["frozen"].blocks.items():
        fx.tagged_blocks[key] = deepcopy(block)
    restyled = tmp_path / "restyled.psd"
    saved.save(str(restyled))
    with pytest.raises(pe.Refused, match="Frozen' is not the declared rebuild: blocks lfx2") as refused:
        pe.gate_bytes(psd_path, str(restyled), baked.edits)
    assert refused.value.code == "bytes"


@text_edit
def test_status_fx_layer_leaves_a_longer_status_word(clone, parser, chart, tmp_path):
    """Review: Snow Bunny's "inflict 1T Frozen." made "inflict 1T Frozen In Time." bakes with its 'Ability FX: Frozen'
    (its own style) deleted, told, and a new 'Ability FX: Frozen In Time' of the chart's style over the whole words (the
    Frozen layer re-cut on "Frozen" left " In Time" flat). gate_status_words refuses a Frozen layer left on them, and
    the words without a layer."""
    psd_path = _card(clone, SNOW_BUNNY)[0]
    text = pe.normalize_ability(pe.parse_card(parser, psd_path, SNOW_BUNNY)["card"]["ability"])
    parsed, baked = _bake(clone, parser, SNOW_BUNNY, {"ability": text.replace("Frozen.", "Frozen In Time.")}, tmp_path, chart)
    edit, = baked.edits
    assert (edit.dropped, [name for name, _ in edit.fx], edit.rebuilt) == ([FROZEN], ["Ability FX: Frozen In Time"], [])
    assert baked.fx_notes == ["Ability FX: Frozen deleted: 'Frozen In Time' is no Frozen"]
    saved = PSDImage.open(baked.psd_path)
    pe.gate_status_words(saved, chart)
    fx, = pe.fx_layers(saved, pe.ability_layer(saved))
    fx.name = FROZEN
    with pytest.raises(pe.Refused, match="'Ability FX: Frozen' covers 'Frozen In Time', no Frozen of its own") as refused:
        pe.gate_status_words(saved, chart)
    assert refused.value.code == "reparse"
    fx.parent.remove(fx)
    with pytest.raises(pe.Refused, match="'Frozen In Time' has no FX layer of its own"):
        pe.gate_status_words(saved, chart)


STANDSTILL = "Items/Infliction/4 Stars/Standstill.psd"   # 'Frozen In Time#1' and '#2' (L7), each its own style


@text_edit
def test_numbered_status_fx_layers_stay_on_their_words(clone, parser, chart, tmp_path):
    """Review (q6): Standstill's "Inflict 2T Frozen In Time." made "Frozen In Time, Frozen In Time.": its '#2' (line 2,
    not edited) stays on its word, renamed '#3', its pixels and the PNG there unchanged; '#1' follows the word it kept,
    renamed '#2'; the typed one gets a new '#1' of the chart's style. Told, every gate passing. Line 2's PNG is the
    no-op bake's (its word takes the chart's fill either way, the 2026-10-21 rule; the JSX's literal names gave it a
    layer of the chart's style)."""
    psd_path, png_path = _card(clone, STANDSTILL)
    old = PSDImage.open(psd_path)
    before = {fx.layer_id: (fx.name, tuple(fx.bbox)) for fx in pe.fx_layers(old, pe.ability_layer(old))}
    text = pe.normalize_ability(pe.parse_card(parser, psd_path, STANDSTILL)["card"]["ability"])
    jobs = {"ability": text.replace("Inflict 2T Frozen In Time.", "Frozen In Time, Frozen In Time.")}
    parsed, baked = _bake(clone, parser, STANDSTILL, jobs, tmp_path, chart)
    fit = "Ability FX: Frozen In Time"
    assert baked.fx_notes == [f"{fit}#1 renamed '{fit}#2': its Frozen In Time is #2 now",
                              f"{fit}#2 renamed '{fit}#3': its Frozen In Time is #3 now"]
    saved = PSDImage.open(baked.psd_path)
    after = {fx.layer_id: (fx.name, tuple(fx.bbox)) for fx in pe.fx_layers(saved, pe.ability_layer(saved))}
    line2 = [i for i, (name, _) in before.items() if name == f"{fit}#2"][0]
    assert after[line2] == (f"{fit}#3", before[line2][1]) and baked.edits[0].fx_names == [f"{fit}#{n}" for n in (1, 2, 3)]
    assert [name for name, _ in baked.edits[0].fx] == [f"{fit}#1"]
    x0, y0, x1, y1 = before[line2][1]
    crop = lambda path: np.asarray(Image.open(path).convert("RGBA"))[y0:y1, x0:x1]
    (tmp_path / "noop").mkdir()
    assert np.array_equal(crop(_noop(clone, parser, chart, STANDSTILL, tmp_path / "noop")[2].png_path), crop(baked.png_path))


@text_edit
def test_fx_gates_refuse_a_wrong_cut(clone, parser, chart, tmp_path):
    """BUG 2 review: the gates on a rebuilt FX layer refuse a wrong one. Mini Doomer's "(BOT)" moved by an edit, then its
    FX layer shifted, cut short, or not the Ability's pixels: gate_fx refuses (reparse), and for a faint stripe away
    from its words too when the layer counts as kept (the own-pixels check skipped); its opacity or name changed on the
    saved card: gate_bytes refuses (bytes)."""
    psd_path = _card(clone, MINI_DOOMER)[0]
    text = pe.normalize_ability(pe.parse_card(parser, psd_path, MINI_DOOMER)["card"]["ability"]).replace("Die.", "Then die at once.")
    parsed, baked = _bake(clone, parser, MINI_DOOMER, {"ability": text}, tmp_path, chart)
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
            fx.name = "Ability FX: (TOP)"
        fx._invalidate_bbox()
        return psd

    pe.gate_fx(PSDImage.open(baked.psd_path), baked.edits, fonts)
    for how, why in (("shifted", "inks .* px outside its words' glyph boxes"), ("short", "covers .* of its words'"),
                     ("repainted", "is not the Ability's own pixels")):
        with pytest.raises(pe.Refused, match=r"'Ability FX: \(BOT\)' " + why) as refused:
            pe.gate_fx(tampered(how), baked.edits, fonts)
        assert refused.value.code == "reparse"
    with pytest.raises(pe.Refused, match=r"'Ability FX: \(BOT\)' inks \d+ px outside"):
        pe.gate_fx(tampered("faint"), [replace(baked.edits[0], rebuilt=[])], fonts)
    renamed = r"renamed 'Ability FX: \(TOP\)', not 'Ability FX: \(BOT\)'"     # gate_bytes' name check fires first
    for how, why in (("opacity", "changed .*blend"), ("renamed", renamed)):
        path = tmp_path / f"{how}.psd"
        tampered(how).save(str(path))
        with pytest.raises(pe.Refused, match=why) as refused:
            pe.gate_bytes(psd_path, str(path), baked.edits)
        assert refused.value.code == "bytes"


@text_edit
def test_ability_bake_drops_the_fx_layer_of_a_word_it_removes(clone, parser, chart, tmp_path):
    """BUG 2: a badge that leaves the text leaves its FX layer's name (PoD11ZOMG's "(1GCD) | (2U) | (2CD)" becomes
    "(2U) | (2CD)", told in the report); a layer whose words all left goes (Mini Doomer's "(BOT)"), told too. A badge
    typed in the icon font joins the name (review: "(2U)" made "(3U)", whose pill the layer draws)."""
    zomg = "Auxiliary/Minions/PoD11ZOMG.psd"
    text = pe.normalize_ability(pe.parse_card(parser, _card(clone, zomg)[0], zomg)["card"]["ability"]).replace("(1GCD)", "", 1)
    one = tmp_path / "one"
    one.mkdir()
    parsed, baked = _bake(clone, parser, zomg, {"ability": text}, one, chart)
    assert [(name, fields) for _, name, fields in baked.records][1:] == [
        ("Ability FX: (1GCD) | (2U) | (2CD)", ["blocks", "channels", "name", "rect"])]
    assert baked.fx == ["Ability FX: (2U) | (2CD)"] and baked.fx_notes == [
        "Ability FX: (1GCD) | (2U) | (2CD) renamed 'Ability FX: (2U) | (2CD)': (1GCD) left the ability"]
    assert baked.reparse["validator"] == []
    two = tmp_path / "two"
    two.mkdir()
    text = pe.normalize_ability(pe.parse_card(parser, _card(clone, MINI_DOOMER)[0], MINI_DOOMER)["card"]["ability"]).replace("(BOT) ", "")
    parsed, baked = _bake(clone, parser, MINI_DOOMER, {"ability": text}, two, chart)
    assert [(name, fields) for _, name, fields in baked.records][1:] == [("Ability FX: (BOT)", ["removed"])]
    assert baked.fx == [] and baked.edits[0].dropped == ["Ability FX: (BOT)"] and baked.edits[0].fx_names == []
    assert baked.fx_notes == ["Ability FX: (BOT) deleted: its words left the ability"]
    three = tmp_path / "three"
    three.mkdir()
    text = pe.normalize_ability(pe.parse_card(parser, _card(clone, zomg)[0], zomg)["card"]["ability"]).replace("(2U)", "(3U)")
    parsed, baked = _bake(clone, parser, zomg, {"ability": text}, three, chart)
    assert baked.fx == ["Ability FX: (1GCD) | (2CD) | (3U)"] and baked.fx_notes == [
        "Ability FX: (1GCD) | (2U) | (2CD) renamed 'Ability FX: (1GCD) | (2CD) | (3U)': (2U) left the ability; (3U) added"]


@text_edit
def test_an_ability_edit_goes_before_a_title_edit(clone, parser, chart, tmp_path):
    """Review 2026-10-24: a title edit stored its layer's record index before the ability edit deleted an FX layer below
    it (Fire Staff's Smoldering sits under its Name in record order), so gate_bytes refused a title it had not changed
    ("layer 15 'Name' changed"). The ability is edited first: both bake."""
    jobs = {"name": "Fire Rod", "ability": "Attacks inflict 2T."}
    parsed, baked = _bake(clone, parser, FIRE_STAFF, jobs, tmp_path, chart)
    assert [edit.field for edit in baked.edits] == ["ability", "name"] and baked.reparse["card"]["display_name"] == "Fire Rod"
    assert baked.reparse["card"]["ability"] == "Attacks inflict 2T." and baked.fx == []

LEAKY = "6e50c3ee"          # The Radiance's 'Ability FX: Infected' with the B1 batch's leaked badge pills (std-batch2 FX.md)


@text_edit
def test_ability_bake_cuts_a_leaky_fx_layer_again(clone, parser, chart, tmp_path):
    """The B1 batch's L7b leaked badge pills into The Radiance's 'Ability FX: Infected' (std-batch2 FX.md; clean on main
    since): an edit that leaves the word alone still cuts the layer again, clean, as today's Effect Layers.jsx would
    (11faa634), and says so; gate_fx would refuse the leaky layer kept."""
    rel = "Creatures/Hollow Knight/5 Stars/The_Radiance.psd"
    leaky = tmp_path / "leaky"
    leaky.mkdir()
    paths = []
    for ext in (".psd", ".png"):
        (leaky / ("The_Radiance" + ext)).write_bytes(at_rev(clone, LEAKY, rel[:-4] + ext))
        paths.append(str(leaky / ("The_Radiance" + ext)))
    old = PSDImage.open(paths[0])
    before, = pe.fx_layers(old, pe.ability_layer(old))
    text = pe.normalize_ability(pe.parse_card(parser, paths[0], rel)["card"]["ability"]) + " Die."
    out = tmp_path / "out"
    out.mkdir()
    parsed, baked = _bake(clone, parser, rel, {"ability": text}, out, chart, paths=paths)
    assert baked.fx_notes == ["Ability FX: Infected cut again: it held pixels outside its words"]
    saved = PSDImage.open(baked.psd_path)
    after, = pe.fx_layers(saved, pe.ability_layer(saved))
    assert after.layer_id == before.layer_id and _ink(after, saved).sum() < 0.3 * _ink(before, old).sum()


GOOFY = "Creatures/Monkey D Folder/4 Stars/Monkey_D_Goofy.psd"    # "Ability FX: 1#1": the "1" of "3T1 Become Back Money"


@text_edit
def test_goofys_styled_one_is_made_again(clone, parser, chart, tmp_path):
    """Monkey D Goofy's "Ability FX: 1#1" (the owner's 20th BUG 2 card) carries the chart's Become Back Money gradient, so an
    ability edit makes it again (kept whole, its "1" not redrawn); once its "1" leaves the text the layer goes, told."""
    psd_path = _card(clone, GOOFY)[0]
    old = pe.normalize_ability(pe.parse_card(parser, psd_path, GOOFY)["card"]["ability"])
    parsed, baked = _bake(clone, parser, GOOFY, {"ability": old + " Die."}, tmp_path, chart)
    assert baked.edits[0].fx_names == ["Ability FX: Become Back Money", "Ability FX: 1#1"] and baked.fx_notes == []
    psd = PSDImage.open(psd_path)
    edit = pe.set_text_layer(psd, "ability", old.replace("3T1", "3T"), str(clone / pe.FONTS_DIR), chart)
    assert edit.fx_names == ["Ability FX: Become Back Money"] and edit.dropped == ["Ability FX: 1#1"]
    assert edit.notes == ["Ability FX: 1#1 deleted: its 1 left the ability"]


@text_edit
def test_goofys_one_stays_on_its_one(clone, parser, chart, tmp_path):
    """BUG 2 review: Goofy's "1#1" is the first "1" of its text, none being a whole word: the "1" of "3T1". A whole-word
    "1" typed anywhere would take the JSX's "1#1" off it ("Draw 1 card."): refused before anything changes. A "1"
    typed before it renumbers the layer ("1#2", told), and that "1" keeps the colour of what it replaces, not the
    layer's gold. gate_fx holds the layer to the "1" it kept (Edit.held): named "1#1" again, it refuses."""
    psd_path = _card(clone, GOOFY)[0]
    old = pe.normalize_ability(pe.parse_card(parser, psd_path, GOOFY)["card"]["ability"])
    fonts = str(clone / pe.FONTS_DIR)
    for text in (old + " Draw 1 card.", "Draw 1 card. " + old, old.replace("a copy of one", "1")):
        psd = PSDImage.open(psd_path)
        layer = pe.ability_layer(psd)
        before = layer.text
        with pytest.raises(pe.Refused, match="unsupported-card: a whole-word '1' in the ability would move the layer Ability "
                                             "FX: 1#1 off the '1' of '3T1'; edit in Photoshop$") as refused:
            pe.set_text_layer(psd, "ability", text, fonts, chart)
        assert refused.value.code == "unsupported-card" and layer.text == before
    parsed, baked = _bake(clone, parser, GOOFY, {"ability": old.replace("3T2", "3T1")}, tmp_path, chart)
    edit, = baked.edits
    assert edit.fx_names == ["Ability FX: Become Back Money", "Ability FX: 1#2"]
    assert baked.fx_notes == ["Ability FX: 1#1 renamed 'Ability FX: 1#2': its 1 is #2 now"]
    mine, typed = old.index("3T1") + 2, old.index("3T2") + 2
    gold = pe._char_styles(pe.ability_layer(PSDImage.open(psd_path)))[0][mine]
    saved = PSDImage.open(baked.psd_path)
    layer = pe.ability_layer(saved)
    fills = pe._char_styles(layer)[0]
    assert edit.held == [("Ability FX: 1#2", [(mine, mine + 1)])] and fills[mine] == gold and fills[typed] != gold
    fx, = [fx for fx in pe.fx_layers(saved, layer) if fx.name == "Ability FX: 1#2"]
    old_psd = PSDImage.open(psd_path)
    assert tuple(fx.bbox) == tuple([f for f in pe.fx_layers(old_psd, pe.ability_layer(old_psd)) if f.name.endswith("1#1")][0].bbox)
    fx.name = "Ability FX: 1#1"
    with pytest.raises(pe.Refused, match="'Ability FX: 1#1' left its words '1'") as refused:
        pe.gate_fx(saved, [replace(edit, held=[("Ability FX: 1#1", edit.held[0][1])])], fonts)
    assert refused.value.code == "reparse"


@text_edit
def test_a_typed_badge_joins_the_badge_layer(clone, parser, chart, tmp_path):
    """Owner 2026-10-23: a timing tag typed anywhere is set in the stoogedings badge font, at the card's own badge style
    (Mini Doomer's 50 pt (BOT)); on a badge card it joins the badge layer as a real badge (review 2026-10-23 refused it:
    typed as body text the JSX's cut would have lifted its letters out of the gradient, white). Every gate passes; the
    parser reads it back."""
    psd_path = _card(clone, MINI_DOOMER)[0]
    old = pe.normalize_ability(pe.parse_card(parser, psd_path, MINI_DOOMER)["card"]["ability"])
    parsed, baked = _bake(clone, parser, MINI_DOOMER, {"ability": old + " Then (bot)."}, tmp_path, chart)
    edit, = baked.edits
    assert edit.fx_names == ["Ability FX: (BOT)"] and baked.fx == ["Ability FX: (BOT)"] and edit.fonts == []
    assert baked.reparse["card"]["ability"].endswith("(BOT) Die. Then (bot).") and baked.reparse["validator"] == []
    runs = _runs(pe.ability_layer(PSDImage.open(baked.psd_path)))
    assert [run[:3] for run in runs if run[1] == pe.ICON_FONT] == [("(BOT)", pe.ICON_FONT, 50.0), ("(bot)", pe.ICON_FONT, 50.0)]


@text_edit
def test_rebuild_fx_refuses_a_badge_in_plain_letters(clone, chart):
    """_rebuild_fx's safety net (set_text_layer sets every typed badge in the icon font, so no bake reaches it): Mini
    Doomer's badge layer whose phrase now also matches a "(BOT)" in the body font refuses rather than cut its letters
    out of the Ability's gradient, white."""
    from psd_tools.composite.text import FontMap
    psd = PSDImage.open(_card(clone, MINI_DOOMER)[0])
    layer = pe.ability_layer(psd)
    old = layer.text
    new = old.replace("Die.", "Die. Then (BOT).")
    plans = pe.plan_fx(layer, pe.fx_layers(psd, layer), chart)
    kept = pe.kept_chars(old, new)
    targets = [pe.follow_fx(plan, new, kept) for plan in plans]
    fonts = FontMap(dirs=[str(clone / pe.FONTS_DIR)])
    layer.set_text(new, keep="runs")
    layer.rasterize(fonts=fonts)
    assert not pe._char_styles(layer)[1][new.rindex("(BOT)")]
    with pytest.raises(pe.Refused, match=r"^unsupported-card: Ability FX: \(BOT\): '\(BOT\)' would show on it as plain "
                                         r"letters"):
        pe._rebuild_fx(psd, layer, new, plans, tuple(layer.bbox), kept, fonts, pe._fx_engine(), targets)


@text_edit
def test_typed_icons_on_a_card_without_the_icon_font(clone, parser, chart, tmp_path):
    """A card whose Ability has no badge or token (Mettaton NEO): a typed timing tag and [type] token are set in the icon
    font at the text's own style, the most common card style, the face added to the layer's and Txt2's FontSets
    (Edit.fonts; gate_stories lets only a declared face in). The re-parse reads "[Fire]" as [fire], as the website
    draws it. Without the parser's types the old rule stands: a layer without the icon font counts as one with image
    icons."""
    rel = "Auxiliary/Creatures/Mettaton_NEO.psd"
    psd_path = _card(clone, rel)[0]
    psd = PSDImage.open(psd_path)
    assert not pe.uses_icon_font(pe.ability_layer(psd)) and pe.image_icons(psd, parser) == []
    old = pe.normalize_ability(pe.parse_card(parser, psd_path, rel)["card"]["ability"])
    with pytest.raises(pe.Refused, match="image icons"):
        pe.set_text_layer(psd, "ability", old + " (1CD) Deal 1 [Fire] damage.", str(clone / pe.FONTS_DIR), chart)
    parsed, baked = _bake(clone, parser, rel, {"ability": old + " (1CD) Deal 1 [Fire] damage."}, tmp_path, chart)
    edit, = baked.edits
    assert edit.fonts == [pe.ICON_FONT] and baked.reparse["card"]["ability"].endswith("(1CD) Deal 1 [fire] damage.")
    saved = PSDImage.open(baked.psd_path)
    runs = _runs(pe.ability_layer(saved))
    body = runs[0][1:3]
    assert [run[:3] for run in runs if run[1] == pe.ICON_FONT] == [("(1CD)", pe.ICON_FONT, body[1]), ("[Fire]", pe.ICON_FONT, body[1])]
    with pytest.raises(pe.Refused, match="Txt2 FontSet gained stoogedings-Regular: no edit added them") as refused:
        pe.gate_stories(PSDImage.open(psd_path), saved, {edit.story})
    assert refused.value.code == "bytes" and pe.gate_stories(PSDImage.open(psd_path), saved, {edit.story}, edit.fonts) == [edit.story]


@text_edit
def test_typed_icons_refuse_where_they_would_print_wrong(clone, parser, chart):
    """Before anything changes: a badge typed under the Ability's colour or gradient overlay with no badge FX layer to
    restore its colours (Cat), a [type] token on a card whose type icons are image layers in the ability's gaps
    (Nebulous Precipice's darkmatter: the website's text holds their tokens), and a badge replaced by letters the icon
    font would draw plainly ("(abc)"). Each unsupported-card."""
    fonts = str(clone / pe.FONTS_DIR)
    for rel, edit, why in (
            ("Auxiliary/Minions/Cat.psd", lambda t: t + " (1CD) Die.", r"\(1CD\): the Ability's overlay, blending or "
                                                                      r"opacity would change the badge's colours and no "
                                                                      r"badge FX layer restores them"),
            ("Field/3 Stars/Nebulous_Precipice.psd", lambda t: t + " Deal 1 [fire] damage.", r"\[type\] tokens on a card with image icons"),
            (MINI_DOOMER, lambda t: t.replace("(BOT)", "(abc)"), r"'\(abc\)' would be set in the icon font, which draws no badge")):
        psd = PSDImage.open(_card(clone, rel)[0])
        layer = pe.ability_layer(psd)
        before = layer.text
        with pytest.raises(pe.Refused, match=why) as refused:
            pe.set_text_layer(psd, "ability", edit(pe.normalize_ability(before)), fonts, chart, parser=parser)
        assert refused.value.code == "unsupported-card"
        assert layer.text == before or rel == MINI_DOOMER          # the stray letters are found once the text is set


TURTLE = "Auxiliary/Minions/Turtle.psd"               # '(1GCD)' under a Color Overlay, 'Ability FX: (1GCD)' over it
BLOOD_BOMB = "Items/Attack/3 Stars/Blood_Bomb.psd"       # Pin Light: its [machine] token bare, no badge FX layer
FLAMEBLADE = "Items/Equip/3 Stars/Flameblade.psd"        # 'Ability FX: Attacks inflict 2T': its own gradient (q6)


@text_edit
def test_typed_icons_refuse_under_any_tint_without_a_badge_layer(clone, parser, chart):
    """Review 2026-10-23: the refusal above counted only an opaque Normal overlay. Soul of Eden's Pin Light blend printed
    a typed (2CD) white on cyan, Ice Staff's two 69% gradients a pale blue one and Holy Mantle's Pin Light a washed [fire],
    every gate passing, and The Patients' 78% opacity a dimmed one (review 2026-10-24): with no badge FX layer to restore
    them they refuse unsupported-card before anything changes.
    Binge Eater, Pin Light with a badge layer, still takes one (the layer names it)."""
    fonts = str(clone / pe.FONTS_DIR)
    for rel, edit in (("Items/Equip/2 Stars/Soul_of_Eden.psd", lambda t: "(2CD) " + t),
                      ("Items/Equip/2 Stars/Ice_Staff.psd", lambda t: "(2CD) " + t),
                      ("Items/Equip/4 Stars/Holy_Mantle.psd", lambda t: t + " [fire]"),
                      ("Creatures/Other/3 Stars/The_Patients.psd", lambda t: "(1CD) " + t)):
        psd = PSDImage.open(_card(clone, rel)[0])
        layer = pe.ability_layer(psd)
        before = layer.text
        assert pe._tinted(layer) and not pe._overlaid(layer)
        with pytest.raises(pe.Refused, match=r"^unsupported-card: .*: the Ability's overlay, blending or opacity would "
                                             r"change the badge's colours and no badge FX layer restores them; edit in "
                                             r"Photoshop$"):
            pe.set_text_layer(psd, "ability", edit(pe.normalize_ability(before)), fonts, chart, parser=parser)
        assert layer.text == before
    assert not pe._tinted(pe.ability_layer(PSDImage.open(_card(clone, "Auxiliary/Creatures/Mettaton_NEO.psd")[0])))
    psd = PSDImage.open(_card(clone, "Items/Equip/4 Stars/Binge_Eater.psd")[0])
    text = pe.normalize_ability(pe.ability_layer(psd).text) + " (2CD) Die."
    assert pe.set_text_layer(psd, "ability", text, fonts, chart, parser=parser).fx_names == ["Ability FX: (1CD) | (2CD)"]


@text_edit
def test_a_badge_edited_by_deleting_letters_keeps_its_badge_layer(clone, parser, chart, tmp_path):
    """Review 2026-10-23: Turtle's "(1GCD)" made "(1CD)" keeps every character, so it did not count as typed: the badge
    layer, its only phrase gone, was deleted ("its words left the ability") and the new badge printed as a blank pill
    under the Ability's Color Overlay, every gate passing. Any icon-font badge no badge layer names joins it: renamed,
    every gate passing."""
    psd_path = _card(clone, TURTLE)[0]
    old = pe.normalize_ability(pe.parse_card(parser, psd_path, TURTLE)["card"]["ability"])
    parsed, baked = _bake(clone, parser, TURTLE, {"ability": old.replace("(1GCD)", "(1CD)")}, tmp_path, chart)
    edit, = baked.edits
    assert edit.fx_names == ["Ability FX: (1CD)"] and edit.dropped == [] and baked.fx == ["Ability FX: (1CD)"]
    assert baked.fx_notes == ["Ability FX: (1GCD) renamed 'Ability FX: (1CD)': (1GCD) left the ability; (1CD) added"]


@text_engine
def test_gate_fx_refuses_a_bare_badge_under_a_tinted_ability(clone):
    """bare_icons, independent of the rebuild: Turtle's badge under its Color Overlay is covered by its badge FX layer;
    with the layer gone (what the deleting-letters bug left) gate_fx refuses, even for an edit that cut no FX layer.
    Blood Bomb's [machine] (Pin Light, no badge layer) was bare before the edit: it may stay so, not become so."""
    from psd_tools.composite.text import FontMap
    fonts_dir = str(clone / pe.FONTS_DIR)
    fonts = FontMap(dirs=[fonts_dir])
    old = PSDImage.open(_card(clone, TURTLE)[0])
    assert pe.bare_icons(old, fonts) == Counter()
    psd = PSDImage.open(_card(clone, TURTLE)[0])
    fx, = pe.fx_layers(psd, pe.ability_layer(psd))
    fx.parent.remove(fx)
    assert pe.bare_icons(psd, fonts) == Counter({"(1gcd)": 1})
    edit = SimpleNamespace(field="ability", fx_names=[])
    with pytest.raises(pe.Refused, match=r"^reparse: \(1gcd\): under the Ability's overlay, blending or opacity with no badge FX "
                                         r"layer over it$"):
        pe.gate_fx(psd, [edit], fonts_dir, None, old)
    bomb = PSDImage.open(_card(clone, BLOOD_BOMB)[0])
    assert pe.bare_icons(bomb, fonts) == Counter({"[machine]": 1})
    pe.gate_fx(bomb, [edit], fonts_dir, None, PSDImage.open(_card(clone, BLOOD_BOMB)[0]))
    with pytest.raises(pe.Refused, match=r"^reparse: \[machine\]: under"):
        pe.gate_fx(bomb, [edit], fonts_dir)


@text_edit
def test_letters_a_deletion_leaves_in_the_icon_font_refuse(clone, parser, chart):
    """"(1GCD)" made "(GCD)" (Turtle) left "(GCD)" in the icon font, drawn as plain letters: the check looked only at
    typed characters. Any letter the icon font now draws plainly refuses, unless it did before the edit. And a badge
    layer whose badges all left while the icon font still draws something no badge layer would name ("★" for the
    badge) refuses rather than go."""
    psd = PSDImage.open(_card(clone, TURTLE)[0])
    text = pe.normalize_ability(pe.ability_layer(psd).text)
    with pytest.raises(pe.Refused, match=r"^unsupported-card: '\(GCD\)' would be set in the icon font, which draws no "
                                         r"badge or icon of it; edit in Photoshop$"):
        pe.set_text_layer(psd, "ability", text.replace("(1GCD)", "(GCD)"), str(clone / pe.FONTS_DIR), chart, parser=parser)
    psd = PSDImage.open(_card(clone, TURTLE)[0])
    with pytest.raises(pe.Refused, match=r"^unsupported-card: Ability FX: \(1GCD\): its badges left the ability but the "
                                         r"icon font still draws something there"):
        pe.set_text_layer(psd, "ability", text.replace("(1GCD)", "★"), str(clone / pe.FONTS_DIR), chart, parser=parser)


@text_edit
def test_a_typed_star_is_set_in_the_icon_font(clone, parser, chart):
    """Ruling C (review 2026-10-23): the website writes "5★"; the stoogedings font draws ★ as its gold star and the parser
    reads it so ([star]), but icon_text knew only "(tag)" and "[token]", so Standstill's "5 star or" made "5★ or" printed
    a missing-glyph box in the body font, every gate passing (the re-parse read "★" back). It is set in the icon font as
    a token is; canonical_tokens reads it as the parser does, but not on the re-parse's side (a "★" outside the font
    stays one), and a typed character its font has no glyph for refuses font."""
    fonts = str(clone / pe.FONTS_DIR)
    psd = PSDImage.open(_card(clone, STANDSTILL)[0])
    text = pe.normalize_ability(pe.ability_layer(psd).text).replace("5 star or", "5★ or")
    edit = pe.set_text_layer(psd, "ability", text, fonts, chart, parser=parser)
    assert [run[:2] for run in _runs(pe.ability_layer(psd)) if run[1] == pe.ICON_FONT] == [("★", pe.ICON_FONT)]
    assert edit.fonts == [pe.ICON_FONT] and [text[a:b] for a, b in pe.icon_text(text, parser.icons)] == ["★"]
    assert pe.canonical_tokens("5★ or", parser.icons) == "5[star] or" and pe.canonical_tokens("5★", parser.icons, False) == "5★"
    psd = PSDImage.open(_card(clone, STANDSTILL)[0])
    with pytest.raises(pe.Refused, match="^font: ability: '☃' has no glyph in its font"):
        pe.set_text_layer(psd, "ability", text.replace("5★", "5☃"), fonts, chart, parser=parser)


@text_edit
def test_a_typed_star_is_cut_into_the_badge_layer(clone, parser, chart):
    """A badge layer takes every colour glyph whatever its name, so a "★" typed far from Scapular's (BOGT) (Pin Light)
    cuts it again: kept whole (none of its glyphs redrawn), the star printed in the blend's colours and gate_fx's
    bare_icons refused it (PoD11ZOMG, Palace of Shadow, Scapular in the review census)."""
    from psd_tools.composite.text import FontMap
    fonts = str(clone / pe.FONTS_DIR)
    psd = PSDImage.open(_card(clone, "Items/Support/3 Stars/Scapular.psd")[0])
    text = pe.normalize_ability(pe.ability_layer(psd).text) + " Costs 5★."
    edit = pe.set_text_layer(psd, "ability", text, fonts, chart, parser=parser)
    assert [name for _, _, name in edit.rebuilt] == ["Ability FX: (BOGT)"]
    assert pe.bare_icons(psd, FontMap(dirs=[fonts])) == Counter()


@text_edit
def test_a_badge_layer_takes_its_icons_red_less_pixels(clone, parser, chart):
    """Review 2026-10-24: a badge layer is cut with the JSX's red key, so a typed [undead]'s black eye sockets and jaw (no
    red) stayed out of Turtle's and printed in its Color Overlay's colour, glowing eyes, every gate passing (gate_fx
    and bare_icons let 1% of a glyph out). The layer takes its colour glyphs' silhouettes too: every opaque pixel of
    the skull is under it, as on the official cuts. And both checks let none out: a hole poked in it refuses."""
    from psd_tools.composite.text import FontMap
    from psd_tools.composite.text.geometry import text_geometry
    fonts_dir = str(clone / pe.FONTS_DIR)
    fonts = FontMap(dirs=[fonts_dir])
    old, psd = PSDImage.open(_card(clone, TURTLE)[0]), PSDImage.open(_card(clone, TURTLE)[0])
    text = pe.normalize_ability(pe.ability_layer(psd).text) + " Immune to [undead]."
    edit = pe.set_text_layer(psd, "ability", text, fonts_dir, chart, parser=parser)
    assert edit.fx_names == ["Ability FX: (1GCD) | [undead]"]
    layer = pe.ability_layer(psd)
    fx, = pe.fx_layers(psd, layer)
    geometry = text_geometry(layer, fonts)
    glyphs = [glyph for glyph in geometry["glyphs"] if glyph["ink"]]
    skull, = [glyph for glyph in glyphs if geometry["text"][glyph["i"]:glyph["end"]] == "[undead]"]
    shape = (psd.height, psd.width)
    body = pe._boxes([glyph for glyph in glyphs if not glyph["font"].lower().startswith("stooge")], 1, shape)
    want = (pe._on_canvas(layer, shape)[..., 3] > 127) & pe._boxes([skull], 0, shape) & ~body
    cover = pe._on_canvas(fx, shape)[..., 3] > 127
    assert want.sum() > 2000 and not (want & ~cover).any()
    pe.gate_fx(psd, [edit], fonts_dir, chart, old)
    ys, xs = (want & cover).nonzero()
    x0, y0 = fx.bbox[:2]
    rgba = np.concatenate([fx.numpy("color"), fx.numpy("shape")], axis=2).astype(np.float32)
    rgba[ys[:6] - y0, xs[:6] - x0] = 0.0        # 6 of its 2,500 px: what the red key left out was 10
    pe._fx_engine()[6](fx, rgba, tuple(fx.bbox))
    fx._invalidate_bbox()
    assert pe.bare_icons(psd, fonts) == Counter({"[undead]": 1})
    with pytest.raises(pe.Refused, match=r"^reparse: 'Ability FX: \(1GCD\) \| \[undead\]' leaves 6 px of its badges and "
                                         r"icons out$"):
        pe.gate_fx(psd, [edit], fonts_dir, chart, old)
    with pytest.raises(pe.Refused, match=r"^reparse: \[undead\]: under the Ability's overlay"):
        pe.gate_fx(psd, [SimpleNamespace(field="ability", fx_names=[])], fonts_dir, chart, old)

def _colour_px(layer, fonts, shape) -> np.ndarray:
    """The text's pixels (any alpha) in its icon font's own colours: renders with every run filled red, then blue, agree."""
    from psd_tools.composite.text.layout import layout
    from psd_tools.composite.text.model import TextModel
    from psd_tools.composite.text.raster import GlyphCache, Options, rasterize
    model, out = TextModel.from_layer(layer), []
    for fill in ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0)):
        key = replace(model, runs=tuple(replace(run, style=replace(run.style, fill=fill)) for run in model.runs))
        rgba, (x0, y0, x1, y1) = rasterize(layout(key, fonts), key.transform, key.antialias, Options(color_glyphs=True),
                                           GlyphCache())
        canvas = np.zeros((*shape, 4), np.float32)
        canvas[int(y0):int(y1), int(x0):int(x1)] = rgba
        out.append(canvas)
    return (np.abs(out[0] - out[1]).max(axis=2) < 1e-3) & (out[0][..., 3] > 0)


@text_edit
def test_a_badge_layer_takes_a_colour_icons_see_through_glow(clone, parser, chart):
    """Reviews 2026-10-24: [support]'s ring has a soft, see-through, red-less glow past its glyf outline (the silhouette
    stops at alpha 110): the badge layer left it out, and it printed in the overlay's colour (maroon on Mini Doomer, olive
    on Turtle), every gate passing (they counted ink past alpha 0.5). The layer takes every pixel of its glyphs, the
    gates every colour-glyph pixel past FAINT_INK: a hole in the glow refuses. A typed "★", which no badge phrase names
    (Turtle: 2 faint px of its glow left out, the gate refused), is the layer's too."""
    from psd_tools.composite.text import FontMap
    from psd_tools.composite.text.geometry import text_geometry
    fonts_dir = str(clone / pe.FONTS_DIR)
    fonts = FontMap(dirs=[fonts_dir])
    for rel, tail in ((MINI_DOOMER, " Gain [support]."), (TURTLE, " Gain 5★."), (TURTLE, " Immune to [support].")):
        old, psd = PSDImage.open(_card(clone, rel)[0]), PSDImage.open(_card(clone, rel)[0])
        edit = pe.set_text_layer(psd, "ability", pe.normalize_ability(pe.ability_layer(psd).text) + tail, fonts_dir, chart,
                                 parser=parser)
        layer = pe.ability_layer(psd)
        fx, = [fx for fx in pe.fx_layers(psd, layer) if pe.BADGE_FX.fullmatch(pe.fx_label(layer, fx))]
        assert fx.name.endswith(" | [support]" if "[" in tail else "(1GCD)") and pe._tinted(layer)
        shape = (psd.height, psd.width)
        alpha = pe._on_canvas(layer, shape)[..., 3]
        colour = _colour_px(layer, fonts, shape) & (alpha > 0)
        held = pe._on_canvas(fx, shape)[..., 3] > 0
        assert (colour & (alpha <= 110)).sum() > (1000 if "[" in tail else 100) and not (colour & ~held).any()
        pe.gate_fx(psd, [edit], fonts_dir, chart, old, parser.icons)
    geometry = text_geometry(layer, fonts)
    ring, = [glyph for glyph in geometry["glyphs"] if geometry["text"][glyph["i"]:glyph["end"]] == "[support]"]
    glow = pe._boxes([ring], 20, shape) & ~pe._boxes([ring], pe.FX_BOX_PAD, shape)
    ys, xs = (colour & held & (alpha > 20) & (alpha < 120) & glow).nonzero()
    x0, y0 = fx.bbox[:2]
    rgba = np.concatenate([fx.numpy("color"), fx.numpy("shape")], axis=2).astype(np.float32)
    rgba[ys[:6] - y0, xs[:6] - x0] = 0.0        # 6 faint px of Turtle's glow, past the ring's glyph box
    pe._fx_engine()[6](fx, rgba, tuple(fx.bbox))
    fx._invalidate_bbox()
    assert pe.bare_icons(psd, fonts) == Counter({"[support]": 1})
    with pytest.raises(pe.Refused, match=r"^reparse: 'Ability FX: \(1GCD\) \| \[support\]' leaves 6 px of its badges and "
                                         r"icons out$"):
        pe.gate_fx(psd, [edit], fonts_dir, chart, old, parser.icons)


@text_edit
def test_a_status_word_in_lettered_brackets_takes_its_fx_layer(clone, parser, chart):
    """Review 2026-10-24: status_spans skipped every bracketed run, so "Inflict [Blazing]." baked unstyled where Effect
    Layers.jsx styles it ("[Blazing]" is no icon: the font draws its letters). Only what the icon font draws is skipped:
    it takes the swatch and its FX layer, every gate passing."""
    fonts_dir = str(clone / pe.FONTS_DIR)
    old, psd = PSDImage.open(_card(clone, HIVE)[0]), PSDImage.open(_card(clone, HIVE)[0])
    text = pe.normalize_ability(pe.ability_layer(psd).text) + " Inflict [Blazing]."
    edit = pe.set_text_layer(psd, "ability", text, fonts_dir, chart, parser=parser)
    assert edit.fx_names == ["Ability FX: Blazing"]
    layer = pe.ability_layer(psd)
    at = layer.text.index("[Blazing]") + 1
    fills = pe._char_styles(layer)[0][at:at + len("Blazing")]
    assert {tuple(round(v, 3) for v in fill) for fill in fills} == {tuple(round(v, 3) for v in chart["blazing"].fill)}
    pe.gate_fx(psd, [edit], fonts_dir, chart, old, parser.icons)


@text_edit
def test_apply_jobs_checks_each_typed_icon_is_one_cluster(clone, parser, monkeypatch, tmp_path):
    """Review 2026-10-24: unshaped_icons runs only given the vocabulary, which apply_jobs passes; nothing failed when it
    did not. Its parse gate reads a token left in the body font first, so the layout here says a typed [fire] is not one
    icon-font cluster: the bake refuses, given the parser's vocabulary."""
    real, seen = pe.unshaped_icons, []

    def unshaped(psd, fonts, vocab):
        seen.append(vocab)
        out = real(psd, fonts, vocab)
        return out + Counter({"[fire]": 1}) if vocab is not None and "[fire]" in pe.ability_layer(psd).text else out

    monkeypatch.setattr(pe, "unshaped_icons", unshaped)
    old = pe.normalize_ability(pe.parse_card(parser, _card(clone, HIVE)[0], HIVE)["card"]["ability"])
    assert "[fire]" not in old
    with pytest.raises(pe.Refused, match=r"^reparse: \[fire\]: not drawn as one badge or icon$"):
        _bake(clone, parser, HIVE, {"ability": old + " Gain [fire]."}, tmp_path)
    assert parser.icons in seen


@text_edit
def test_reparse_reads_a_star_outside_the_icon_font_as_typed(clone, parser, tmp_path):
    """gate_reparse reads a typed "★" as [star] on the edit's side only: a "★" left in the body font (a box on the card)
    reads back as "★" and refuses reparse (read as [star] on both sides, it passed: review 2026-10-23)."""
    psd_path = _card(clone, STANDSTILL)[0]
    parsed = pe.parse_card(parser, psd_path, STANDSTILL)
    psd = PSDImage.open(psd_path)
    layer = pe.ability_layer(psd)
    text = layer.text.replace("5 star or", "5★ or")
    pe.splice_text(layer, text, pe.edit_parts(layer.text, text, *pe.changed_range(layer.text, text)))
    path = str(tmp_path / "Standstill.psd")
    psd.save(path)
    job = {"ability": pe.normalize_ability(text)}
    with pytest.raises(pe.Refused, match="^reparse: fields differ from the expected parse: ability$"):
        pe.gate_reparse(parser, parsed, path, STANDSTILL, job, PSDImage.open(path))


@text_edit
def test_a_token_the_font_draws_as_letters_refuses(clone, parser, chart):
    """Review 2026-10-24: "[fire:]" counted as drawn (ICON_VARIANTS' "" is "[fire]"'s), so it was set in the icon font,
    which printed its letters (no ligature), cut into Mini Doomer's badge layer (white over the gradient) and read back
    as [fire], while the editor showed the fire icon: every gate passed. Typed, made by adding a colon (Palace of
    Shadow's "[dark:sticker]", "[dark]" on the website), or a spelling the website's norm() alone forgives ("[Fire.]"):
    refused font before anything changes."""
    fonts = str(clone / pe.FONTS_DIR)
    for rel, edit in ((MINI_DOOMER, lambda t: t + " Deal 1 [fire:] damage."),
                      (PALACE, lambda t: t.replace("[dark]", "[dark:]", 1)),
                      (TURTLE, lambda t: t + " Deal 1 [Fire.] damage.")):
        psd_path = _card(clone, rel)[0]
        psd = PSDImage.open(psd_path)
        before = pe.ability_layer(psd).text
        text = edit(pe.normalize_ability(pe.parse_card(parser, psd_path, rel)["card"]["ability"]))
        with pytest.raises(pe.Refused, match=r"^font: \[\w+[:.]\]: the website shows an icon, but the icon font has no such "
                                             r"spelling and would print its letters"):
            pe.set_text_layer(psd, "ability", text, fonts, chart, parser=parser)
        assert pe.ability_layer(psd).text == before


@text_edit
def test_reparse_reads_a_token_with_no_variant_as_typed(clone, parser, tmp_path):
    """A "[fire:]" in the icon font (typed right after Mini Doomer's (BOT), whose style it takes) prints its letters: the
    parser reads it as typed, an unknown token (it read [fire], as gate_reparse did: it passed). Refused reparse."""
    psd_path = _card(clone, MINI_DOOMER)[0]
    parsed = pe.parse_card(parser, psd_path, MINI_DOOMER)
    psd = PSDImage.open(psd_path)
    layer = pe.ability_layer(psd)
    text = layer.text.replace("(BOT) Die.", "(BOT)[fire:] Die.")
    pe.splice_text(layer, text, pe.edit_parts(layer.text, text, *pe.changed_range(layer.text, text)))
    assert all(pe._char_styles(layer)[1][i] for i in range(text.index("[fire:]"), text.index(" Die.")))
    path = str(tmp_path / "Mini_Doomer.psd")
    psd.save(path)
    with pytest.raises(pe.Refused, match=r"^reparse: new problems: UNKNOWN ICON TOKEN: \[fire:\]$"):
        pe.gate_reparse(parser, parsed, path, MINI_DOOMER, {"ability": pe.normalize_ability(text)}, PSDImage.open(path))

def test_follow_fx_refuses_an_edit_inside_an_anchored_phrase():
    """Ruling B: an "own" or "styled" layer stays on the characters it covered. "+2 DEF" made "+3 DEF" keeps most of
    them but the phrase no longer matches: the layer was deleted ("its words left the ability", untrue) and its style
    lost (Palace of Shadow's blue, Flameblade's gradient, Mr Krabs' 2T, Morpho Knight's +2/4/6 ATK). Refused
    unsupported-card; a phrase the edit removes whole still leaves (told)."""
    old = "[dark]: +2 DEF\rAfter 4 turns."
    plan = pe.FxPlan(SimpleNamespace(name="Ability FX: +2 DEF"), ["+2 DEF"], "own", ranges=[(8, 14)],
                     found=[("+2 DEF", [(8, 14)])])
    new = old.replace("+2", "+3")
    with pytest.raises(pe.Refused, match=r"^unsupported-card: Ability FX: \+2 DEF: the edit changes its '\+2 DEF' \(now "
                                         r"'\+3 DEF'\), which would lose the layer's style; edit in Photoshop$"):
        pe.follow_fx(plan, new, pe.kept_chars(old, new))
    new = "[dark]: After 4 turns."
    assert pe.follow_fx(plan, new, pe.kept_chars(old, new)) == [("+2 DEF", None, [], [], "+2 DEF left the ability")]


@text_engine
def test_gate_fx_refuses_letters_in_the_icon_font(clone):
    """lettered_icons, independent of icon_text: "[fire:]" typed right after The Hive's (1GCD) (no overlay), in the badge's
    icon font, shapes as seven letters, no ligature: gate_fx refuses reparse. The card as it is draws none, and its
    badge and token are one cluster each."""
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
    with pytest.raises(pe.Refused, match=r"^reparse: ':\[\]efir': the icon font draws them as letters, no badge or icon$"):
        pe.gate_fx(psd, [SimpleNamespace(field="ability", fx_names=[])], fonts_dir, None, old)


@text_edit
def test_gate_fx_refuses_a_token_not_shaped_as_one_icon(clone, parser):
    """unshaped_icons, the layout's check of icon_text: each badge or token it reads is one icon-font cluster. A "[fire]"
    typed in The Hive's body font (an edit's style there) is six letters: gate_fx refuses reparse; its badge and token
    as they are pass."""
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
    pe.gate_fx(psd, [SimpleNamespace(field="ability", fx_names=[])], fonts_dir, None, old)     # no vocabulary: not checked
    with pytest.raises(pe.Refused, match=r"^reparse: \[fire\]: not drawn as one badge or icon$"):
        pe.gate_fx(psd, [SimpleNamespace(field="ability", fx_names=[])], fonts_dir, None, old, parser.icons)


@text_edit
def test_a_blank_ability_on_an_fx_card_refuses_as_blank(clone, parser, chart, tmp_path):
    """Review 2026-10-24: deleting a whole ability on an FX card refused styled-word ('the text layer has no pixels to
    cut') before looking at its layers, though every one leaves with its words. Each is deleted (told), and the blank
    ability refuses as on any card: the parser finds no ability text."""
    psd = PSDImage.open(_card(clone, SNOW_BUNNY)[0])
    edit = pe.set_text_layer(psd, "ability", "", str(clone / pe.FONTS_DIR), chart, parser=parser)
    assert edit.dropped == [FROZEN] and edit.fx_names == [] and edit.notes[0].startswith(f"{FROZEN} deleted")
    with pytest.raises(pe.Refused, match=r"^reparse: new problems: ABILITY TEXT NOT FOUND$"):
        _bake(clone, parser, SNOW_BUNNY, {"ability": ""}, tmp_path, chart)


@text_edit
def test_an_edit_inside_an_own_style_phrase_refuses(clone, chart):
    """Flameblade's "Attacks inflict 2T" (its own gradient) made "3T": refused before anything changes."""
    psd = PSDImage.open(_card(clone, FLAMEBLADE)[0])
    layer = pe.ability_layer(psd)
    before = layer.text
    with pytest.raises(pe.Refused, match=r"^unsupported-card: Ability FX: Attacks inflict 2T: the edit changes its "
                                         r"'Attacks inflict 2T' \(now 'Attacks inflict 3T'\)"):
        pe.set_text_layer(psd, "ability", before.replace("2T", "3T"), str(clone / pe.FONTS_DIR), chart)
    assert layer.text == before


@text_edit
def test_an_fx_phrase_the_ability_lacks_refuses(clone, chart):
    """A hand-named "Ability FX: Frozen In Time #1" (a space before "#": no number to Effect Layers.jsx, so no status
    either) crashed plan_fx (fx_kind called it a Frozen In Time layer, the chart lookup a "frozen in time #1" one); its
    phrase is in no text, so what it covers can't be told: refused unsupported-card."""
    base = SimpleNamespace(name="Ability")
    assert pe.fx_kind(SimpleNamespace(name="Ability FX: Frozen #2", kind="pixel", tagged_blocks=None), base, chart) == "own"
    psd = PSDImage.open(_card(clone, STANDSTILL)[0])
    layer = pe.ability_layer(psd)
    fx = next(fx for fx in pe.fx_layers(psd, layer) if fx.name.endswith("#1"))
    fx.name = "Ability FX: Frozen In Time #1"
    with pytest.raises(pe.Refused, match=r"^unsupported-card: FX layer phrases not in the ability: Frozen In Time #1; "
                                         r"edit in Photoshop$"):
        pe.set_text_layer(psd, "ability", pe.normalize_ability(layer.text) + " Die.", str(clone / pe.FONTS_DIR), chart)


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


def test_a_colon_with_no_variant_is_no_token(parser):
    """Review 2026-10-24: ICON_VARIANTS' "" (the plain "[fire]") let "[fire:]" count as drawn, which the font has no
    ligature for (it prints the letters) and canonical_tokens read as [fire]. token_variant takes only the font's
    spellings; unlettered_icons names what the website draws as an icon (text.js: the base before the first colon,
    norm()ed) and the font does not."""
    assert [pe.token_variant(t) for t in ("[fire]", "[Fire:Sticker]", "[fire:NO]", "[fire:]", "[fire: no]", "[fire:foo]")] == [
        "", "sticker", "no", None, None, None]
    text = "Deal [fire:] and [fire] damage."
    assert [text[a:b] for a, b in pe.icon_text(text, parser.icons)] == ["[fire]"]
    assert pe.canonical_tokens(text, parser.icons) == "Deal [fire:] and [fire] damage."
    text = "[fire:], [fire: no], [Fire.], [warp  star], [fire:no:x], [fire], [warp star], [Fire Staff] and [foo]."
    assert pe.unlettered_icons(text, parser) == ["[fire:]", "[fire: no]", "[Fire.]", "[warp  star]", "[fire:no:x]"]
    assert pe.unlettered_icons(text, None) == []

@text_edit
def test_an_edit_keeps_the_cards_own_token_spellings(clone, parser, chart, tmp_path):
    """The parser reads Palace of Shadow's "[dark:sticker]" and "[light:sticker]" as "[dark]" and "[light]", the website
    sends those back, and setting them as given redrew both stickers as plain icons, every gate passing (review
    2026-10-23: 9 cards on main print a :sticker or :tint). keep_tokens spells a kept icon as the layer does; a token
    the edit types stays as typed. Through every gate the PSD keeps both stickers and the parse reads the same."""
    rel = "Field/4 Stars/Palace_of_Shadow.psd"
    psd_path = _card(clone, rel)[0]
    old = pe.normalize_ability(pe.parse_card(parser, psd_path, rel)["card"]["ability"])
    assert old.startswith("[dark]: +2 DEF") and "[light] creature" in old
    layer = pe.ability_layer(PSDImage.open(psd_path))
    assert pe.keep_tokens(layer, old.replace("\n", "\r") + " [dark] Die.", parser.icons).endswith("[light:sticker] creature with "
        "over 50% of its HP to summon the Shadow Queen in its place. [dark] Die.")
    assert pe.keep_tokens(layer, "[dark]", None) == "[dark]"
    parsed, baked = _bake(clone, parser, rel, {"ability": old + " Die."}, tmp_path, chart)
    text = pe.ability_layer(PSDImage.open(baked.psd_path)).text
    assert text.startswith("[dark:sticker]: +2 DEF") and "[light:sticker] creature" in text and baked.fx == []
    assert pe.normalize_ability(baked.reparse["card"]["ability"]) == old + " Die."


def test_keep_tokens_pairs_whole_tokens(clone, parser):
    """Review 2026-10-24: a diff of characters paired a token typed right after a kept one ("[dark] [fire]:") with the
    kept token's ']' when the text after was the longer, so "[dark:sticker]" went plain (Error Room, Corruption,
    Elemental World, Palace of Shadow). Whole tokens pair; the typed one stays as typed."""
    layer = pe.ability_layer(PSDImage.open(_card(clone, "Field/4 Stars/Palace_of_Shadow.psd")[0]))
    seen = pe.canonical_tokens(layer.text, parser.icons)
    assert seen.startswith("[dark]: +2 DEF")
    for typed in (" [fire]", " [support]", " (1CD)", " foes"):
        got = pe.keep_tokens(layer, seen.replace("[dark]:", "[dark]" + typed + ":", 1), parser.icons)
        assert got.startswith("[dark:sticker]" + typed + ": +2 DEF") and "[light:sticker] creature" in got
    got = pe.keep_tokens(layer, "[fire] " + seen, parser.icons)
    assert got.startswith("[fire] [dark:sticker]: +2 DEF")


def test_icon_style_is_the_cards_commonest(clone, parser, tmp_path):
    """A typed badge takes the card's commonest badge style: Pizza's first badge is 18.66 pt with a baseline shift, its two
    others 32 pt (COLOURED), so 32; a tie goes to the body text's size, then the first (Morpho Knight's tracking 0 and 500,
    two each: 0). None on a layer without the icon font (Mettaton NEO)."""
    pizza = pe.ability_layer(PSDImage.open(_at(clone, tmp_path, COLOURED, PIZZA)[0]))
    style = pe.icon_style(pizza, True, parser.icons)
    assert style["Font"] == pe.ICON_FONT and pe._canon(style["FontSize"]) == 32.0 and pe._canon(style["BaselineShift"]) == 0.0
    assert pe._canon(pe.icon_style(pizza, False, parser.icons)) == pe._canon(style)    # no token: its badges' style
    morpho = pe.ability_layer(PSDImage.open(_card(clone, MORPHO)[0]))
    assert pe._canon(pe.icon_style(morpho, True, parser.icons)["Tracking"]) == 0
    neo = pe.ability_layer(PSDImage.open(_card(clone, "Auxiliary/Creatures/Mettaton_NEO.psd")[0]))
    assert pe.icon_style(neo, True, parser.icons) is None


@text_edit
def test_inserted_text_takes_the_body_fill(clone, chart, tmp_path):
    """set_text(keep="runs") gives inserted text the run before it; after a status word that is the swatch colour."""
    fonts = str(clone / pe.FONTS_DIR)
    psd = PSDImage.open(_plain(clone, tmp_path)[0])
    old = pe.ability_layer(psd).text
    white, green = (1.0, 1.0, 1.0), tuple(round(v, 3) for v in chart["poison"].fill)
    pe.set_text_layer(psd, "ability", old + " Inflict Poison.", fonts, chart)
    pe.set_text_layer(psd, "ability", old + " Inflict Poison for 2 turns.", fonts, chart)
    assert _fills(pe.ability_layer(psd))[-2:] == [("Poison", green), (" for 2 turns.", white)]
    pe.set_text_layer(psd, "ability", old + " Inflict Poisoned for 2 turns.\n", fonts, chart)   # the word widens, the \n goes
    layer = pe.ability_layer(psd)
    assert _fills(layer)[-1][1] == white and layer.text.endswith("Poisoned for 2 turns.")
    assert pe.body_fill(layer, chart) == white


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
def test_an_ability_edit_takes_the_colour_of_what_it_replaces(clone, chart, tmp_path):
    """Pizza's three lines are yellow, cyan and green; body_fill (the commonest) is cyan. An edited word takes the fill
    of the text it replaces, an insertion the line's (the 2026-10-23 check: '1' -> '2' on the yellow line went cyan)."""
    fonts = str(clone / pe.FONTS_DIR)
    psd = PSDImage.open(_at(clone, tmp_path, COLOURED, PIZZA)[0])
    layer = pe.ability_layer(psd)
    yellow, cyan, green = (0.979, 0.945, 0.477), (0.053, 0.825, 0.894), (0.546, 0.829, 0.326)
    assert pe.body_fill(layer, chart) == pytest.approx(cyan, abs=1e-3)
    text = layer.text.replace("add 1 bar", "add 2 bar").replace("another type.", "another type. Yes.")
    pe.set_text_layer(psd, "ability", text, fonts, chart)
    runs = _runs(pe.ability_layer(psd))
    assert [run[3] for run in runs] == [yellow, yellow, cyan, cyan, green, green]
    assert "add 2 bar" in runs[1][0] and runs[5][0].endswith("another type. Yes.")


@text_edit
def test_an_edit_at_both_ends_keeps_the_badges_between(clone, chart, tmp_path):
    """The text goes in one changed part at a time (splice_text), so a badge between two edits keeps its icon font,
    size and colour (in one set_text, Pizza's (1CD) took the body font: its pill drawn as text). Words the engine sets in
    a badge's or token's style (replacing a badge, typed right after a token) take the style of the text they replace;
    a badge typed after a badge stays a badge."""
    fonts = str(clone / pe.FONTS_DIR)
    yellow, cyan = (0.979, 0.945, 0.477), (0.053, 0.825, 0.894)
    icon = "stoogedings-Regular"
    pizza = _at(clone, tmp_path, COLOURED, PIZZA)[0]
    for edits, check in (
            ([("add 1 bar", "add 2 bar"), ("one 2 star", "one 3 star")],
             lambda runs: runs[2] == ("(1CD)", icon, 32.0, cyan) and runs[3][1:] == ("Domigorgon", 32.0, cyan)),
            ([("(1GCD) Remove or add 1", "(2GCD) Remove or add 2")],
             lambda runs: runs[0] == ("(2GCD)", icon, 18.66, yellow) and runs[1][1:] == ("Domigorgon", 32.0, yellow)),
            ([("(1GCD) Remove", "Once per turn, remove")],
             lambda runs: runs[0][0].startswith("Once per turn, remove")
             and runs[0][1:] == ("Domigorgon", 32.0, yellow)),
            ([("(1CD) Can", "(1CD)(2U) Can")],
             lambda runs: runs[2] == ("(1CD)(2U)", icon, 32.0, cyan))):
        psd = PSDImage.open(pizza)
        text = pe.ability_layer(psd).text
        for old, new in edits:
            text = text.replace(old, new, 1)
        pe.set_text_layer(psd, "ability", text, fonts, chart)
        runs = _runs(pe.ability_layer(psd))
        assert check(runs), (edits, runs)
    psd = PSDImage.open(_at(clone, tmp_path, COLOURED, NAMED[0])[0])
    text = pe.ability_layer(psd).text.replace("[undead].", "[undead], always.")
    pe.set_text_layer(psd, "ability", text, fonts, chart)
    runs = _runs(pe.ability_layer(psd))
    assert [r[0] for r in runs if r[1] == icon] == ["(OFS)", "(1PG)(OYT)", "[undead]"]
    assert [r[1] for r in runs if ", always" in r[0]] == ["FrizQuadrataStd"] or \
        [r[1][:7] for r in runs if ", always" in r[0]] == ["FrizQua"]


@text_edit
def test_parentheses_typed_after_a_badge_are_body_text(clone, chart, tmp_path):
    """The engine gives text typed right after a badge the badge's style; only what the icon font draws stays in it
    (icon_text): Pizza's "(1CD)(2U)" keeps both badges, "(1CD)(2 points)" sets "(2 points)" in the body font (it stayed
    in the icon font, drawn as its plain letters, while any parenthesis counted as a badge)."""
    pizza = _at(clone, tmp_path, COLOURED, PIZZA)[0]
    for typed, check in (("(2U)", lambda runs: runs[2][:2] == ("(1CD)(2U)", pe.ICON_FONT)),
                         ("(2 points)", lambda runs: runs[2][:2] == ("(1CD)", pe.ICON_FONT)
                          and runs[3][0].startswith("(2 points) Can") and runs[3][1] == "Domigorgon")):
        psd = PSDImage.open(pizza)
        pe.set_text_layer(psd, "ability", pe.ability_layer(psd).text.replace("(1CD) Can", f"(1CD){typed} Can", 1),
                          str(clone / pe.FONTS_DIR), chart)
        runs = _runs(pe.ability_layer(psd))
        assert check(runs), (typed, runs)


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


def test_edit_parts_split_an_edit_at_what_it_keeps():
    old, new = "(1CD) Deal 1 damage.\r(2U) Heal 2.", "(1CD) Deal 3 damage.\r(2U) Heal 4 HP."
    start, end = pe.changed_range(old, new)
    parts = pe.edit_parts(old, new, start, min(end, len(new)))
    assert [(old[a0:a1], new[b0:b1], kept) for a0, a1, b0, b1, kept in parts] == [
        ("1", "3", False), (" damage.\r(2U) Heal ", " damage.\r(2U) Heal ", True), ("2", "4 HP", False)]
    assert pe.edit_parts("Burn", "Burns", 0, 5) == [(0, 4, 0, 4, True), (4, 4, 4, 5, False)]     # a word widened
    assert pe.edit_parts("a1b", "a2b", 1, 2) == [(1, 2, 1, 2, False)]
    assert pe.edit_parts("ab", "ab", 2, 2) == [] and pe.edit_parts("abcd", "abd", 2, 2) == [(2, 3, 2, 2, False)]


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


@text_edit
def test_status_words_are_normalized_to_the_chart(clone, chart, tmp_path):
    """The chart is the authority (design 14 q6): a status word the edit leaves alone takes its swatch too (a grey Poison turns
    the chart's green), and an edit touching a styled word no longer refuses (round 1's styled-word)."""
    fonts = str(clone / pe.FONTS_DIR)
    psd = PSDImage.open(_plain(clone, tmp_path)[0])
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
    """Fire Staff's hand-made "Smoldering" overlay (before L7) carries a Stroke the chart knows nothing about: nothing could
    stand in for it."""
    fonts = str(clone / pe.FONTS_DIR)
    psd = PSDImage.open(BytesIO(at_rev(clone, OVERLAY_STAFF, FIRE_STAFF)))
    text = pe.ability_layer(psd).text
    with pytest.raises(pe.Refused, match="overlay word layers the chart has no style for: Smoldering") as refused:
        pe.set_text_layer(psd, "ability", text + " Die.", fonts, chart)
    assert refused.value.code == "unsupported-card" and pe.ability_layer(psd).text == text
    with pytest.raises(pe.Refused, match="status word layers over the ability"):
        pe.set_text_layer(psd, "ability", text + " Die.", fonts, None)


SMOLDERING = "Ability FX: Smoldering"


@text_edit
def test_own_style_fx_layer_takes_a_typed_word_in_its_style(clone, parser, chart, tmp_path):
    """L7 made Fire Staff's overlay an "Ability FX: Smoldering" layer (0c573ed0) with a stroke no chart status has: since
    2026-10-23 (owner) an ability edit rebuilds it in that style. A typed Smoldering joins it, as Effect Layers.jsx's
    Update all cuts every whole-word match, and the report says the typed word takes the layer's own style; its
    style bytes stay, the stroke on both words, every gate passing."""
    psd_path = _card(clone, FIRE_STAFF)[0]
    old = pe.normalize_ability(pe.parse_card(parser, psd_path, FIRE_STAFF)["card"]["ability"])
    assert old == "Attacks inflict 2T Smoldering."
    parsed, baked = _bake(clone, parser, FIRE_STAFF, {"ability": "Smoldering: 2T Smoldering."}, tmp_path, chart)
    edit, = baked.edits
    assert edit.fx_names == [SMOLDERING] and edit.fx == [] and _style(baked.psd_path, SMOLDERING) == _style(psd_path, SMOLDERING)
    assert baked.fx_notes == [f"{SMOLDERING} also covers the typed 'Smoldering', in its own style"]
    saved = PSDImage.open(baked.psd_path)
    layer = pe.ability_layer(saved)
    fx, = pe.fx_layers(saved, layer)
    assert len(pe.phrase_ranges(layer.text, pe.fx_phrases(layer, fx))) == 2 and edit.held == [(SMOLDERING, [(15, 25)])]


@text_edit
def test_an_own_style_status_word_keeps_the_charts_fill(clone, chart):
    """An edit beside an own-style layer's words gives them the phrase's colour (its first drawn run, 8308103f), but a
    status word among them keeps the chart's swatch (the owner's 2026-10-23 ruling: off-chart status colours go to the
    chart): Fire Staff's Smoldering painted green before the edit comes out the chart's orange."""
    psd = PSDImage.open(_card(clone, FIRE_STAFF)[0])
    layer = pe.ability_layer(psd)
    start = layer.text.index("Smoldering")
    layer.set_span_style(start, start + len("Smoldering"), fill=(0.0, 1.0, 0.0))
    pe.set_text_layer(psd, "ability", layer.text.replace("Smoldering.", "Smoldering!"), str(clone / pe.FONTS_DIR), chart)
    rounded = lambda fill: tuple(round(v, 3) for v in fill)
    fills = pe._char_styles(pe.ability_layer(psd))[0]
    assert [rounded(f) for f in fills[start:start + len("Smoldering")]] == [rounded(chart["smoldering"].fill)] * len("Smoldering")


PALACE = "Field/4 Stars/Palace_of_Shadow.psd"   # 'Ability FX: +2 DEF': a Stroke alone over its words' own blue
PLUS_DEF = "Ability FX: +2 DEF"
WATCHER = "Creatures/In Sound Mind/3 Stars/The_Watcher.psd"     # 'Ability FX: Despair': its own style
CLOT_GOLD = "Auxiliary/Minions/Clot_Gold.psd"   # 'Ability FX: 1 Become Back Money': the chart's style on "2T1 Become Back Money"


@text_edit
def test_a_typed_copy_of_an_own_style_phrase_takes_its_colour(clone, parser, chart, tmp_path):
    """Review 2026-10-23: a "+2 DEF" typed beside Palace of Shadow's own-style "+2 DEF" joins its layer (Update all cuts
    every whole-word match), whose Stroke draws over the words' own blue: the copy took the body's grey, a colour the
    card never had, every gate passing. It takes the phrase's colour (its first drawn run), through every gate; so do a
    copy of Morpho Knight's red "+2/4/6 ATK" (it printed yellow) and one made by deleting a letter outside the edited
    word ("x+2 DEF" made "+2 DEF")."""
    psd_path = _card(clone, PALACE)[0]
    blue = dict(_fills(pe.ability_layer(PSDImage.open(psd_path))))["+2 DEF"]
    old = pe.normalize_ability(pe.parse_card(parser, psd_path, PALACE)["card"]["ability"])
    parsed, baked = _bake(clone, parser, PALACE, {"ability": old + " Allies get +2 DEF."}, tmp_path, chart)
    assert baked.fx_notes == [f"{PLUS_DEF} also covers the typed '+2 DEF', in its own style"]
    assert [fill for words, fill in _fills(pe.ability_layer(PSDImage.open(baked.psd_path))) if "+2 DEF" in words] == [blue, blue]
    psd = PSDImage.open(_card(clone, MORPHO)[0])
    layer = pe.ability_layer(psd)
    red = dict(_fills(layer))["+2/4/6 ATK"]
    pe.set_text_layer(psd, "ability", layer.text.replace("(ENDT).", "(ENDT). +2/4/6 ATK."), str(clone / pe.FONTS_DIR), chart,
                      parser=parser)
    assert [fill for words, fill in _fills(layer) if "+2/4/6 ATK" in words] == [red, red]
    psd = PSDImage.open(psd_path)
    layer = pe.ability_layer(psd)
    pe.set_text_layer(psd, "ability", layer.text + " Allies get x+2 DEF.", str(clone / pe.FONTS_DIR), chart, parser=parser)
    pe.set_text_layer(psd, "ability", layer.text.replace("x+2", "+2"), str(clone / pe.FONTS_DIR), chart, parser=parser)
    assert [fill for words, fill in _fills(layer) if "+2 DEF" in words] == [blue, blue]


@text_edit
def test_a_typed_copy_of_a_phrase_in_several_colours_refuses(clone, parser, chart):
    """Which colour a copy joining an own-style layer takes can't be told when its words are in more than one (none on
    main: Palace of Shadow's "+2" painted red here): refused unsupported-card before anything changes."""
    psd = PSDImage.open(_card(clone, PALACE)[0])
    layer = pe.ability_layer(psd)
    start = layer.text.index("+2 DEF")
    layer.set_span_style(start, start + 2, fill=(1.0, 0.0, 0.0))
    before = layer.text
    with pytest.raises(pe.Refused, match=r"^unsupported-card: Ability FX: \+2 DEF: its words are in more than one colour, "
                                         r"so the typed '\+2 DEF' has none of its own; edit in Photoshop$"):
        pe.set_text_layer(psd, "ability", before + " Allies get +2 DEF.", str(clone / pe.FONTS_DIR), chart, parser=parser)
    assert layer.text == before


@text_edit
def test_an_edit_in_a_phrase_in_several_colours_refuses(clone, parser, chart):
    """Review 2026-10-24: an edit inside an own-style phrase's characters paints the phrase its colour (plan.fill), which
    recoloured the rest of one in several colours, gate_fx_colours passing (the colour was one its words had): Palace of
    Shadow's "+2" painted red, "+2 DEF" made "+2  DEF" baked with "DEF" red. Refused before anything changes."""
    psd = PSDImage.open(_card(clone, PALACE)[0])
    layer = pe.ability_layer(psd)
    start = layer.text.index("+2 DEF")
    layer.set_span_style(start, start + 2, fill=(1.0, 0.0, 0.0))
    before = layer.text
    with pytest.raises(pe.Refused, match=r"^unsupported-card: Ability FX: \+2 DEF: its words are in more than one colour, "
                                         r"which the edit in '\+2  DEF' would make one; edit in Photoshop$"):
        pe.set_text_layer(psd, "ability", before.replace("+2 DEF", "+2  DEF"), str(clone / pe.FONTS_DIR), chart,
                          parser=parser)
    assert layer.text == before

@text_edit
def test_a_moved_own_style_phrase_refuses(clone, parser, chart):
    """An own-style layer stays on the characters it covered (ruling B). Palace of Shadow's "+2 DEF" deleted from its
    line and typed at the end, or its whole line moved there, kept none of them, yet the layer jumped to the typed copy,
    printed grey, and the note said it "also" covered it. Refused unsupported-card before anything changes."""
    for move in (lambda t: t.replace("+2 DEF\x03", "\x03").replace("its place.", "its place. +2 DEF"),
                 lambda t: t.replace("[dark:sticker]: +2 DEF\x03", "").replace("its place.", "its place.\r[dark]: +2 DEF")):
        psd = PSDImage.open(_card(clone, PALACE)[0])
        layer = pe.ability_layer(psd)
        before = layer.text
        assert move(before) != before
        with pytest.raises(pe.Refused, match=r"^unsupported-card: Ability FX: \+2 DEF: the edit deletes its '\+2 DEF' and "
                                             r"types it again, which would move the layer; edit in Photoshop$"):
            pe.set_text_layer(psd, "ability", move(before), str(clone / pe.FONTS_DIR), chart, parser=parser)
        assert layer.text == before


@text_edit
def test_letters_glued_onto_an_own_style_phrase_refuse(clone, parser, chart):
    """"1T Despair." made "1T Despairing." kept The Watcher's own-style 'Ability FX: Despair' on "Despair" and printed
    "ing" plain, untold: half a word in the layer's style (a status layer there leaves, told). Refused unsupported-card
    before anything changes, and so is a styled phrase glued on where its word ended (Clot Gold's "1 Become Back Money"
    of "2T1 Become Back Money", "Money" made "Moneying"); a word typed after it still bakes."""
    psd = PSDImage.open(_card(clone, CLOT_GOLD)[0])
    layer = pe.ability_layer(psd)
    with pytest.raises(pe.Refused, match=r"^unsupported-card: Ability FX: 1 Become Back Money: the edit makes its '1 Become "
                                         r"Back Money' part of '2T1 Become Back Moneying', half a word in its style"):
        pe.set_text_layer(psd, "ability", layer.text.replace("Back Money", "Back Moneying"), str(clone / pe.FONTS_DIR),
                          chart, parser=parser)
    psd = PSDImage.open(_card(clone, WATCHER)[0])
    layer = pe.ability_layer(psd)
    before = layer.text
    with pytest.raises(pe.Refused, match=r"^unsupported-card: Ability FX: Despair: the edit makes its 'Despair' part of "
                                         r"'Despairing', half a word in its style; edit in Photoshop$"):
        pe.set_text_layer(psd, "ability", before.replace("Despair.", "Despairing."), str(clone / pe.FONTS_DIR), chart,
                          parser=parser)
    assert layer.text == before
    edit = pe.set_text_layer(psd, "ability", before.replace("Despair.", "Despair for free."), str(clone / pe.FONTS_DIR),
                             chart, parser=parser)
    assert edit.fx_names == [fx.name for fx in pe.fx_layers(PSDImage.open(_card(clone, WATCHER)[0]), layer)]


@text_edit
def test_a_status_layer_moved_to_a_typed_word_is_told_so(clone, parser, chart):
    """Doctor Zomboss's own-style (L7) 'Ability FX: Blazing' lands on its status's words: "2T Blazing or 2T Frostbite"
    swapped deletes its Blazing and types one, so the note says the layer moved to the typed word; it said "also"."""
    psd = PSDImage.open(_card(clone, ZOMBOSS)[0])
    layer = pe.ability_layer(psd)
    text = layer.text.replace("2T Blazing or 2T Frostbite", "2T Frostbite or 2T Blazing")
    assert text != layer.text
    edit = pe.set_text_layer(psd, "ability", text, str(clone / pe.FONTS_DIR), chart, parser=parser)
    assert edit.notes == ["Ability FX: Blazing moved to the typed 'Blazing', in its own style"]


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
def test_fx_without_the_engine_refuses_styled_word(clone, chart, monkeypatch, tmp_path):
    """styled-word now means this psd-tools cannot cut glyph pixels: with the rasterizer gone a styled word refuses before
    anything changes, an unstyled one still bakes."""
    fonts = str(clone / pe.FONTS_DIR)
    psd = PSDImage.open(_plain(clone, tmp_path)[0])
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
    the unchanged text cuts the same word (IoU 0.953 / 0.994 with the JSX pixel cut, all of our ink on its ink or its 1 px
    edge ring: the fork's 4 x 4 samples draw Gasoline 0.08 px low, a quarter-pixel more on bottom edges), hides the overlay, passes the validator's coverage (1.0) and the bytes
    and reopen gates. A splice keeps the word's Photoshop pixels and cuts them under the engine's key: the JSX's own cut
    (IoU 1.0). These two cards keep image-icon gaps, so the token rule refuses their edits in the pipeline. The
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
    iou = (ours & theirs).sum() / (ours | theirs).sum()
    assert (iou == 1.0) if pe.splices() else (iou >= floor and _near(ours, theirs) >= 0.99)
    assert abs(mine.bbox[0] - box[0]) <= 1 and abs(mine.bbox[1] - box[1]) <= 1 and mine.bbox[3] == box[3]
    assert pe.overlay_words(reopened) == [] and pe._enabled_effects(mine) == ["OuterGlow"]
    assert pe.PSDParser._pixel_coverage(mine, pe.ability_layer(reopened)) == 1.0
    old, new = pe.parse_card(parser, psd_path, rel), pe.parse_card(parser, saved, rel)
    assert new["validator"] == old["validator"] and not any(p.startswith("EFFECT LAYER") for p in new["validator"])
    records, flag_bytes, block_bytes = pe.gate_bytes(psd_path, saved, [edit])
    ability = ["blocks"] if pe.splices() else ["blocks", "channels"]     # a splice keeps the unchanged text's raster
    assert [(name, fields) for _, name, fields in records] == [("Ability", ability), ("Ability FX: Blazing", ["added"]),
                                                               ("Blazing", ["flags"])]
    assert flag_bytes == 1 and block_bytes == sum(len(b) for b in edit.fx[0][1].values()) == 5916
    pe.gate_reparse(parser, old, saved, rel, {}, reopened, [edit])


@text_edit
@pytest.mark.skipif(not colour_glyphs(), reason="the fork draws SVG colour glyphs in one colour")
def test_fx_layers_leave_colour_glyphs_out(clone, chart):
    """Gasoline's [fire] token keeps its SVG colours in a key copy: the all-black key finds it, and it stays out of the
    regenerated Blazing layer, which matches the clean one Effect Layers.jsx cut before the badges (std-batch2 FX.md).
    Its own Blazing layer taken away first, the edit makes it anew (new_fx)."""
    psd = PSDImage.open(_card(clone, GASOLINE)[0])
    layer = pe.ability_layer(psd)
    assert pe.uses_icon_font(layer) and "[fire]" in layer.text
    old, = pe.fx_layers(psd, layer)
    theirs = _ink(old, psd)
    old.parent.remove(old)
    edit = pe.set_text_layer(psd, "ability", pe.normalize_ability(layer.text), str(clone / pe.FONTS_DIR), chart)
    assert [name for name, _ in edit.fx] == [old.name] == ["Ability FX: Blazing"]
    mine, = pe.fx_layers(psd, pe.ability_layer(psd))
    ours = _ink(mine, psd)
    assert (ours & theirs).sum() / (ours | theirs).sum() >= 0.9 and _near(ours, theirs) >= 0.99


@text_edit
def test_ability_bake_regenerates_fx_layers_through_every_gate(clone, parser, chart, tmp_path):
    """Idea 14 through apply_jobs on Mini_Doomer: adding Blazing and Frozen cuts two FX layers (2 x 5916 block bytes landed),
    within the parity bar, the validator clean, nothing outside the region; a second bake on the result deletes Frozen's
    and cuts the repeated Blazing into its layer again in place (BUG 2), the record diff saying so."""
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
    two = tmp_path / "two"
    two.mkdir()
    text = old + " Inflict Blazing. Blazing twice."
    parsed = pe.parse_card(parser, baked.psd_path, MINI_DOOMER)
    assert pe.classify(parsed, {"ability": text}, saved).jobs == {"ability": text}
    again = pe.apply_jobs(baked.psd_path, baked.png_path, MINI_DOOMER, {"ability": text}, parsed, parser, str(two),
                          str(clone / pe.FONTS_DIR), chart, update_preview=False)
    assert [(name, fields) for _, name, fields in again.records] == [
        ("Ability", TEXT_RECORD), ("Ability FX: Blazing", ["channels", "rect"]), ("Ability FX: Frozen", ["removed"])]
    assert again.fx == ["Ability FX: Blazing"] and again.reparse["validator"] == [] and again.block_bytes == 0
    assert again.edits[0].dropped == ["Ability FX: Frozen"] and again.fx_notes == ["Ability FX: Frozen deleted: its words left the ability"]
    assert pe.stale_stories(PSDImage.open(again.psd_path)) == []
    print("timings", {k: round(v, 2) for k, v in baked.timings.items()}, {k: round(v, 2) for k, v in again.timings.items()})


def _raster(layer, box) -> np.ndarray:
    """The layer's stored RGBA (8-bit values) placed on a canvas box."""
    out = np.zeros((box[3] - box[1], box[2] - box[0], 4), np.int16)
    x0, y0, x1, y1 = layer.bbox
    rgba = np.concatenate([layer.numpy("color"), layer.numpy("shape")], axis=2)
    out[y0 - box[1]:y1 - box[1], x0 - box[0]:x1 - box[0]] = np.rint(rgba * 255)
    return out


def _first_line_edit(layer) -> str:
    """Frostmourne's ability, its first line "+1.5 ATK" made "+2.5 ATK" (as the website sends it)."""
    assert layer.text.startswith("+1.5 ATK\r")
    return ("+2.5" + layer.text[4:]).replace("\r", "\n")


@spliced
def test_ability_edit_keeps_photoshops_raster_on_unchanged_lines(clone, chart):
    """A one-word edit redraws its own line or less: outside that band every pixel of the layer is Photoshop's raster."""
    psd = PSDImage.open(_card(clone, FROSTMOURNE)[0])
    layer = pe.ability_layer(psd)
    old_box = tuple(layer.bbox)
    box = (old_box[0] - 8, old_box[1] - 8, old_box[2] + 8, old_box[3] + 8)
    old = _raster(layer, box)
    edit = pe.set_text_layer(psd, "ability", _first_line_edit(layer), str(clone / pe.FONTS_DIR), chart)
    assert pe.ability_layer(psd).text.startswith("+2.5 ATK\r")
    x0, y0, x1, y1 = edit.band
    assert y1 - y0 < (old_box[3] - old_box[1]) / 3                # the first line, not the layer
    new = _raster(pe.ability_layer(psd), box)
    band = np.zeros(old.shape[:2], bool)
    band[y0 - box[1]:y1 - box[1], x0 - box[0]:x1 - box[0]] = True
    assert np.array_equal(old[~band], new[~band])
    assert (old[band] != new[band]).any()


@spliced
def test_spliced_ability_bake_changes_only_the_edited_line(clone, parser, chart, tmp_path):
    """Through apply_jobs: every pixel the merge changed lies in the redrawn band grown by the effects' reach."""
    psd_path, png_path = _card(clone, FROSTMOURNE)
    text = _first_line_edit(pe.ability_layer(PSDImage.open(psd_path)))
    _, baked = _bake(clone, parser, FROSTMOURNE, {"ability": text}, tmp_path, chart)
    edit, = baked.edits
    layer = pe.ability_layer(PSDImage.open(baked.psd_path))
    x0, y0, x1, y1 = pe._pad(edit.band, fx_pad(layer), layer._psd)
    assert y1 - y0 < (edit.region[3] - edit.region[1]) / 2
    before = np.asarray(Image.open(png_path).convert("RGBA"))
    after = np.asarray(Image.open(baked.png_path).convert("RGBA"))
    differs = (before != after).any(axis=2)
    inside = np.zeros_like(differs)
    inside[y0:y1, x0:x1] = True
    assert differs.any() and not (differs & ~inside).any()


@text_edit
def test_too_long_ability_refuses_overflow(clone, chart, tmp_path):
    psd = PSDImage.open(_plain(clone, tmp_path)[0])
    text = pe.normalize_ability(pe.ability_layer(psd).text).replace("(BOT) ", "")   # typed (BOT)s refuse first: no badge layer
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


@text_edit
def test_baseline_miss_under_the_floor_bakes_with_a_note(clone, parser, tmp_path):
    """A title missing the parity bar under the floor: baked and told, no longer refused (owner, 2026-10-23); the parser
    reads the new title back. The miss is painted (the top tenth of Mini Doomer's title box inverted in its PNG): since
    round 10 every title is within the bar (Red_Knife's outer glow noise, the last miss, is matched)."""
    psd = PSDImage.open(_card(clone, MINI_DOOMER)[0])
    name = _typo_fix(pe.read_name(psd))
    paths = _painted(clone, tmp_path, 0.1, pe.name_layer(psd).bbox)
    _, baked = _bake(clone, parser, MINI_DOOMER, {"name": name}, tmp_path, paths=paths)
    measured = baked.parity["name"]
    assert not pe._within_bar(measured) and not pe._past_floor(measured)
    assert pe.read_name(PSDImage.open(baked.psd_path)) == name
    assert pe.miss_notes(baked) == ["name baked with a pixel difference: bad %.2f%%, mean %.2f, max %d"
                                    % (measured["bad_pct"], measured["mean"], measured["max"])]


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
    """The fork drawing no ability text: every render hides the visible Text/Ability layer."""
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
    (review, 2026-10-23). An edit that draws no change where the render matches the PNG still bakes."""
    psd_path, png_path = _card(clone, SHIELD)
    _no_ability(monkeypatch)
    old = pe.parse_card(parser, psd_path, SHIELD)["card"]["ability"]
    new = old[:len(old) - 1] + ("!" if old[len(old) - 1] != "!" else ".")
    with pytest.raises(pe.Refused) as refused:
        _bake(clone, parser, SHIELD, {"ability": new}, tmp_path)
    assert refused.value.code == "parity" and refused.value.why.startswith("ability: the render shows no change, baseline bad ")
    assert 5 < float(search(r"bad ([\d.]+)%", refused.value.why).group(1)) < pe.PARITY_FLOOR
    assert not list(tmp_path.iterdir())
    region = pe.set_text_layer(PSDImage.open(psd_path), "ability", new, str(clone / pe.FONTS_DIR), None).region
    old_psd, png = PSDImage.open(psd_path), Image.open(png_path)
    png.load()
    drawn = png.copy()
    drawn.paste(pe.render_region(old_psd, region, pe.export_icc(old_psd, png)).convert(png.mode), region[:2])
    out = tmp_path / "drawn"
    out.mkdir()
    drawn.save(out / Path(png_path).name)
    (out / Path(psd_path).name).write_bytes(Path(psd_path).read_bytes())
    (tmp_path / "baked").mkdir()
    _, baked = _bake(clone, parser, SHIELD, {"ability": new}, tmp_path / "baked",
                     paths=(str(out / Path(psd_path).name), str(out / Path(png_path).name)))
    assert baked.changed_pixels == 0 and pe._within_bar(baked.parity["ability"]) and pe.miss_notes(baked) == []
    assert pe.normalize_ability(baked.reparse["card"]["ability"]) == pe.normalize_ability(new)


@text_edit
def test_a_blank_render_is_refused_beside_another_edit(clone, parser, tmp_path, monkeypatch):
    """Another edit's pixels inside an overlapping region are not this edit's drawing (review 2026-10-23): Shield of the
    Guilty retitled and its ability drawn blank, the ability's region stretched over the title, is refused `parity` as
    the blank render it is (it baked before: the title's pixels made the ability's merge non-empty)."""
    psd_path, _ = _card(clone, SHIELD)
    _no_ability(monkeypatch)
    old = pe.parse_card(parser, psd_path, SHIELD)["card"]["ability"]
    new = old[:len(old) - 1] + ("!" if old[len(old) - 1] != "!" else ".")
    real = pe.set_text_layer

    def stretched(psd, which, text, fonts_dir, chart=None, card="", own_text=True, parser=None):
        edit = real(psd, which, text, fonts_dir, chart, card, own_text, parser)
        if which == "ability":
            edit.region = pe.union_box([edit.region, pe.name_layer(psd, card).bbox])
        return edit
    monkeypatch.setattr(pe, "set_text_layer", stretched)
    with pytest.raises(pe.Refused) as refused:
        _bake(clone, parser, SHIELD, {"name": _typo_fix(pe.read_name(PSDImage.open(psd_path))), "ability": new}, tmp_path)
    assert refused.value.code == "parity" and refused.value.why.startswith("ability: the render shows no change, baseline bad ")
    assert not list(tmp_path.iterdir())


@text_engine
def test_baseline_reads_a_kept_raster_off_the_preview(clone):
    """The Freezer's title: Photoshop composited it straight (a raster kept through a session), and only its preview
    says so; the default blend reads that and is within the bar, the text gamma alone is not."""
    psd_path, png_path = _card(clone, "Field/1 Stars/The_Freezer.psd")
    psd, png = PSDImage.open(psd_path), Image.open(png_path).convert("RGBA")
    region = pe._pad(tuple(pe.name_layer(psd).bbox), fx_pad(pe.name_layer(psd)), psd)
    _, measured = pe.baseline(psd, png, region, pe.export_icc(psd, png))
    assert measured["text"] == "preview"
    assert not pe._within_bar(pe._measure(pe.render_region(psd, region, pe.export_icc(psd, png), "gamma"), png.crop(region)))


@text_engine
def test_an_edited_layer_goes_on_in_the_gamma(clone):
    """The Freezer's title, read straight off its preview, is pinned to the text gamma once edited: the preview still
    shows the old text."""
    psd_path, png_path = _card(clone, "Field/1 Stars/The_Freezer.psd")
    psd, png = PSDImage.open(psd_path), Image.open(png_path).convert("RGBA")
    layer = pe.name_layer(psd)
    region = pe._pad(tuple(layer.bbox), fx_pad(layer), psd)
    icc = pe.export_icc(psd, png)
    pinned = np.asarray(pe.render_region(psd, region, icc, "preview", (layer.layer_id,)))
    assert np.array_equal(pinned, np.asarray(pe.render_region(psd, region, icc, "gamma")))
    assert not np.array_equal(pinned, np.asarray(pe.render_region(psd, region, icc, "preview")))
    assert pe.text_kwargs(psd, "straight", (layer.layer_id,)) == {"text_gamma": None}


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
def test_ability_edit_keeps_a_forced_line_break(clone, chart, tmp_path):
    """Mini Doomer's "it. \x03(BOT) Die.": the website's newline there stays a soft break, its paragraph runs as they were."""
    psd = PSDImage.open(_plain(clone, tmp_path)[0])
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
def test_ability_edit_keeps_a_leading_empty_paragraph(clone, chart, tmp_path):
    """PoD10BFB's ability starts with an empty paragraph: an edit (the website's trimmed text) keeps it in TySh and the
    story, so the text stays where it was. The card before its badge layer (PLAIN_POD), whose edit refuses."""
    path = tmp_path / "PoD10BFB.psd"
    path.write_bytes(at_rev(clone, PLAIN_POD, "Auxiliary/Minions/PoD10BFB.psd"))
    psd = PSDImage.open(path)
    layer = pe.ability_layer(psd)
    old = layer.text
    assert old.startswith("\r(1GCD)")
    pe.set_text_layer(psd, "ability", pe.normalize_ability(old).replace("creature.", "creature!"), str(clone / pe.FONTS_DIR), chart)
    assert layer.text == old.replace("creature.", "creature!") and pe.stories(psd)[pe.story_index(layer)] == layer.text + "\r"
    assert [int(n) for n in layer.engine_dict["ParagraphRun"]["RunLengthArray"]][0] == 1


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


@text_edit
def test_glow_tokens_refuse(clone, chart):
    """The fork draws a :glow halo as its glyf outline, a solid block even under an overlay: refused before any change."""
    psd = PSDImage.open(_card(clone, "Field/4 Stars/The_Hive.psd")[0])
    layer = pe.ability_layer(psd)
    old = layer.text
    for token in ("[fire:glow]", "[Fire:DarkGlow]"):
        with pytest.raises(pe.Refused, match="glow halo") as refused:
            pe.set_text_layer(psd, "ability", pe.normalize_ability(old) + f" Deal 1 {token} damage.", str(clone / pe.FONTS_DIR), chart)
        assert refused.value.code == "font" and token in refused.value.why and layer.text == old
    if not colour_glyphs():                                     # :sticker and :tint keep the old rule
        with pytest.raises(pe.Refused, match="colour glyphs"):
            pe.set_text_layer(psd, "ability", pe.normalize_ability(old) + " Deal 1 [fire:sticker] damage.", str(clone / pe.FONTS_DIR), chart)


@text_edit
@pytest.mark.skipif(not colour_glyphs(), reason="the fork draws SVG colour glyphs in one colour")
def test_colour_glyph_ability_bakes_through_every_gate(clone, parser, chart, tmp_path):
    """The_Hive's typed [fire] through apply_jobs: within the parity bar, the validator clean, nothing outside the region,
    and the new badge in its SVG colours (the body text is white on a dark stroke: saturated pixels are the badge's)."""
    old = pe.parse_card(parser, _card(clone, HIVE)[0], HIVE)["card"]["ability"]
    _, baked = _bake(clone, parser, HIVE, {"ability": old + " Deal 1 [fire] damage."}, tmp_path, chart)
    assert pe._within_bar(baked.parity["ability"]) and baked.reparse["validator"] == [] and baked.fx == []
    before = np.asarray(Image.open(_card(clone, HIVE)[1]).convert("RGB")).astype(np.int16)
    after = np.asarray(Image.open(baked.png_path).convert("RGB")).astype(np.int16)
    differs = (before != after).any(axis=2)
    x0, y0, x1, y1 = baked.region
    assert not differs[:y0].any() and not differs[y1:].any() and not differs[:, :x0].any() and not differs[:, x1:].any()
    saturated = after.max(axis=2) - after.min(axis=2) > 96
    assert int((differs & saturated).sum()) > 50


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
