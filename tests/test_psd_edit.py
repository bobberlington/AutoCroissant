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
LORIAN = "Auxiliary/Minions/Lorian.psd"                            # smart objects with smart filters
PAINT_WARRIOR = "Auxiliary/Minions/Paint_Warrior.psd"              # a blank Ability whose box runs over the stat bars
KORO = "Creatures/Other/5 Stars/Koro-Sensei.psd"                   # tags printed in the body font, no icon font
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
    assert pe.miss_notes(SimpleNamespace(parity={"ability": ok}, whole={"ability": miss})) == [
        "ability redrawn whole, its lines laid out again: bad 1.97%, mean 1.60, max 217"]
    assert pe.miss_notes(SimpleNamespace(parity={"ability": ok}, whole={"ability": ok})) == []


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
    two offset copies of its ability): a bake would change one and leave the copies (an ability bake edits them too:
    set_ability_layers). An empty layer has no copies (Anubisath's empty Ability beside an empty 'Layer 1')."""
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
def test_new_fx_layer_goes_above_the_groups_clip_chain(clone, parser, chart, tmp_path):
    """Claw's MagentaSmorg and Exposure 1 are clipped to its Text group: a new FX layer put directly above Text took the
    chain off the group, so the PSD's title and text lost them while the PNG did not, every gate passing (review
    2026-10-24). It goes above the chain; the new PSD renders as the new PNG outside the edit, and gate_untouched
    refuses a change there."""
    rel = "Auxiliary/Minions/Claw.psd"
    psd_path = _card(clone, rel)[0]
    new = pe.normalize_ability(pe.parse_card(parser, psd_path, rel)["card"]["ability"]) + " Inflict Blazing."
    parsed, baked = _bake(clone, parser, rel, {"ability": new}, tmp_path, chart)
    saved = PSDImage.open(baked.psd_path)
    assert [layer.name for layer in saved][-4:] == ["Text", "MagentaSmorg", "Exposure 1", "Ability FX: Blazing"]
    assert [layer.name for layer in pe._clip_chain(saved[len(saved) - 4])] == ["MagentaSmorg", "Exposure 1"]
    render = pe.render_all(saved)[..., :3].astype(int)
    png = np.asarray(Image.open(baked.png_path).convert("RGB")).astype(int)
    off = np.abs(render - png).max(axis=2) > 24
    x0, y0, x1, y1 = baked.region
    off[y0:y1, x0:x1] = False
    assert not off.any()
    old = PSDImage.open(psd_path)
    before = pe.render_all(old)
    old[len(old) - 2].visible = False                                 # MagentaSmorg, over the title
    edit = SimpleNamespace(field="ability", region=baked.region)
    with pytest.raises(pe.Refused, match="outside the edited boxes") as refused:
        pe.gate_untouched(before, old, [edit])
    assert refused.value.code == "region" and refused.value.field is None


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
def test_a_stat_and_the_ability_bake_together(clone, parser, chart, tmp_path):
    """gate_untouched leaves out the stat edits' regions too: hp and the ability in one bake (review 2026-10-24: the hp
    row's digits counted as changed outside the edited boxes)."""
    old = pe.normalize_ability(pe.parse_card(parser, _card(clone, MINI_DOOMER)[0], MINI_DOOMER)["card"]["ability"])
    parsed, baked = _bake(clone, parser, MINI_DOOMER, {"hp": 6, "ability": old + " Die."}, tmp_path, chart)
    assert sorted(edit.field for edit in baked.edits) == ["ability", "hp"]
    assert baked.reparse["card"]["hp"] == 6


@text_edit
def test_the_edit_copy_keeps_smart_filters(clone, parser, chart, tmp_path):
    """apply_jobs edits a deepcopy of the card, and the fork pairs a smart object with its smart filter by id() of its
    record: a plain deepcopy renders Lorian without them (1.5M px), which the PNG took inside an edit's region (review
    2026-10-24). edit_copy renders as the card; gate_copy refuses a copy that does not; an hp bake's PNG is its PSD."""
    psd_path, png_path = _card(clone, LORIAN)
    psd = PSDImage.open(psd_path)
    before = pe.render_all(psd)
    pe.gate_copy(before, pe.edit_copy(psd))
    with pytest.raises(pe.Refused, match="renders") as refused:
        pe.gate_copy(before, deepcopy(psd))
    assert refused.value.code == "error"
    hp = pe.parse_card(parser, psd_path, LORIAN)["card"]["hp"]
    parsed, baked = _bake(clone, parser, LORIAN, {"hp": hp - 1}, tmp_path, chart)
    x0, y0, x1, y1 = baked.region
    saved = PSDImage.open(baked.psd_path)
    render = saved.composite(viewport=baked.region, ignore_preview=True, apply_icc=baked.icc).convert("RGB")
    png = Image.open(baked.png_path).convert("RGB").crop(baked.region)
    assert int((np.abs(np.asarray(render).astype(int) - np.asarray(png).astype(int)).max(axis=2) > 24).sum()) == 0


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
    tells the words without a layer."""
    psd_path = _card(clone, SNOW_BUNNY)[0]
    text = pe.normalize_ability(pe.parse_card(parser, psd_path, SNOW_BUNNY)["card"]["ability"])
    parsed, baked = _bake(clone, parser, SNOW_BUNNY, {"ability": text.replace("Frozen.", "Frozen In Time.")}, tmp_path, chart)
    edit, = baked.edits
    assert (edit.dropped, [name for name, _ in edit.fx], edit.rebuilt) == ([FROZEN], ["Ability FX: Frozen In Time"], [])
    assert baked.fx_notes == ["Ability FX: Frozen deleted: 'Frozen In Time' is no Frozen"]
    saved = PSDImage.open(baked.psd_path)
    assert pe.gate_status_words(saved, chart) == []
    fx, = pe.fx_layers(saved, pe.ability_layer(saved))
    fx.name = FROZEN
    with pytest.raises(pe.Refused, match="'Ability FX: Frozen' covers 'Frozen In Time', no Frozen of its own") as refused:
        pe.gate_status_words(saved, chart)
    assert refused.value.code == "reparse"
    fx.parent.remove(fx)
    assert pe.gate_status_words(saved, chart) == ["styled-word: ability: 'Frozen In Time' has no FX layer of its own"]


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
    "1" typed would take the JSX's "1#1" off it ("Draw 1 card."): the layer goes, told, its "1" printing plain. A "1"
    typed before it renumbers the layer ("1#2", told), and that "1" keeps the colour of what it replaces, not the
    layer's gold. gate_fx holds the layer to the "1" it kept (Edit.held): named "1#1" again, it refuses."""
    psd_path = _card(clone, GOOFY)[0]
    old = pe.normalize_ability(pe.parse_card(parser, psd_path, GOOFY)["card"]["ability"])
    fonts = str(clone / pe.FONTS_DIR)
    edit = pe.set_text_layer(PSDImage.open(psd_path), "ability", old + " Draw 1 card.", fonts, chart)
    assert edit.dropped == ["Ability FX: 1#1"]
    flagged(edit.notes, "^Ability FX: 1#1 deleted: a whole-word '1' in the ability would move the layer Ability FX: 1#1 "
                        "off the '1' of '3T1'$")
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
def test_rebuild_fx_leaves_a_badge_in_plain_letters_off(clone, chart):
    """_rebuild_fx's safety net (set_text_layer sets every typed badge in the icon font): Mini Doomer's badge layer whose
    phrase now also matches a "(BOT)" in the body font leaves it off, told, rather than cut its letters out of the
    Ability's gradient, white."""
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
    at = new.rindex("(BOT)")
    rebuilt, deleted, covered, notes = pe._rebuild_fx(psd, layer, new, plans, tuple(layer.bbox), kept, fonts,
                                                      pe._fx_engine(), targets)
    flagged(notes, r"^styled-word: ability: Ability FX: \(BOT\): '\(BOT\)' is not in the icon font")
    assert (at, at + 5) not in covered


@text_edit
def test_typed_icons_on_a_card_without_the_icon_font(clone, parser, chart, tmp_path):
    """A card whose Ability has no badge or token (Mettaton NEO): a typed timing tag and [type] token are set in the icon
    font at the text's own style, the most common card style, the face added to the layer's and Txt2's FontSets
    (Edit.fonts; gate_stories lets only a declared face in). The re-parse reads "[Fire]" as [fire], as the website
    draws it."""
    rel = "Auxiliary/Creatures/Mettaton_NEO.psd"
    psd_path = _card(clone, rel)[0]
    psd = PSDImage.open(psd_path)
    assert not pe.uses_icon_font(pe.ability_layer(psd)) and pe.image_icons(psd, parser) == []
    old = pe.normalize_ability(pe.parse_card(parser, psd_path, rel)["card"]["ability"])
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
def test_typed_icons_that_print_wrong_are_flagged(clone, parser, chart):
    """Baked and told (owner 2026-10-24): a badge typed under the Ability's colour or gradient overlay with no badge FX
    layer to restore its colours (Cat), a [type] token on a card whose type icons are image layers in the ability's gaps
    (Nebulous Precipice's darkmatter: the website's text holds their tokens), and a badge replaced by letters the icon
    font draws plainly ("(abc)")."""
    fonts = str(clone / pe.FONTS_DIR)
    for rel, edit, why in (
            ("Auxiliary/Minions/Cat.psd", lambda t: t + " (1CD) Die.", r"^unsupported-card: ability: \(1CD\): the Ability's "
                                                                      r"look \(its Color Overlay\) changes the typed badge's "
                                                                      r"colours and no badge FX layer restores them$"),
            ("Field/3 Stars/Nebulous_Precipice.psd", lambda t: t + " Deal 1 [fire] damage.", r"\[type\] tokens on a card with image icons"),
            (MINI_DOOMER, lambda t: t.replace("(BOT)", "(abc)"), r"'\(abc\)' set in the icon font, which draws no badge")):
        psd = PSDImage.open(_card(clone, rel)[0])
        text = edit(pe.normalize_ability(pe.ability_layer(psd).text))
        flagged(pe.set_text_layer(psd, "ability", text, fonts, chart, parser=parser).notes, why)


TURTLE = "Auxiliary/Minions/Turtle.psd"               # '(1GCD)' under a Color Overlay, 'Ability FX: (1GCD)' over it
BLOOD_BOMB = "Items/Attack/3 Stars/Blood_Bomb.psd"       # Pin Light: its [machine] token bare, no badge FX layer
FLAMEBLADE = "Items/Equip/3 Stars/Flameblade.psd"        # 'Ability FX: Attacks inflict 2T': its own gradient (q6)


@text_edit
def test_typed_icons_under_any_tint_without_a_badge_layer(clone, parser, chart):
    """Review 2026-10-23: the tint counted only an opaque Normal overlay. Soul of Eden's Pin Light blend printed a typed
    (2CD) white on cyan, Ice Staff's two 69% gradients a pale blue one and Holy Mantle's Pin Light a washed [fire], The
    Patients' 78% opacity a dimmed one, Claw's and Miasma's Exposure 1 clipped to Text a pale one (reviews 2026-10-24):
    _tinted names each, and with no badge FX layer to restore them the edit is flagged. Binge Eater, Pin Light with a
    badge layer, takes one (the layer names it)."""
    fonts = str(clone / pe.FONTS_DIR)
    for rel, why in (("Items/Equip/2 Stars/Soul_of_Eden.psd", "its Pin Light blending"),
                     ("Items/Equip/2 Stars/Ice_Staff.psd", "its Gradient Overlay"),
                     ("Items/Equip/4 Stars/Holy_Mantle.psd", "its Pin Light blending"),
                     ("Creatures/Other/3 Stars/The_Patients.psd", "its opacity or fill below full"),
                     ("Auxiliary/Minions/Claw.psd", "'Exposure 1', clipped to its group 'Text'"),
                     ("Auxiliary/Minions/Miasma.psd", "'Exposure 1', clipped to its group 'Text'")):
        layer = pe.ability_layer(PSDImage.open(_card(clone, rel)[0]))
        assert pe._tinted(layer) == why and not pe._overlaid(layer)
    psd = PSDImage.open(_card(clone, "Items/Equip/4 Stars/Holy_Mantle.psd")[0])
    edit = pe.set_text_layer(psd, "ability", pe.normalize_ability(pe.ability_layer(psd).text) + " [fire]", fonts, chart,
                             parser=parser)
    flagged(edit.notes, r"^unsupported-card: ability: \[fire\]: the Ability's look \(its Pin Light blending\) changes the "
                        r"typed icon's colours and no badge FX layer restores them$")
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
def test_gate_fx_tells_a_bare_badge_under_a_tinted_ability(clone):
    """bare_icons, independent of the rebuild: Turtle's badge under its Color Overlay is covered by its badge FX layer;
    with the layer gone (what the deleting-letters bug left) gate_fx tells it, even for an edit that cut no FX layer.
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
    assert pe.gate_fx(psd, [edit], fonts_dir, None, old) == [
        "unsupported-card: ability: (1gcd): under the Ability's look (its Color Overlay) with no badge FX layer over it"]
    bomb = PSDImage.open(_card(clone, BLOOD_BOMB)[0])
    assert pe.bare_icons(bomb, fonts) == Counter({"[machine]": 1})
    assert pe.gate_fx(bomb, [edit], fonts_dir, None, PSDImage.open(_card(clone, BLOOD_BOMB)[0])) == []
    flagged(pe.gate_fx(bomb, [edit], fonts_dir), r"^unsupported-card: ability: \[machine\]: under")


@text_edit
def test_letters_a_deletion_leaves_in_the_icon_font_are_flagged(clone, parser, chart):
    """"(1GCD)" made "(GCD)" (Turtle) left "(GCD)" in the icon font, drawn as plain letters: the check looked only at
    typed characters. Any letter the icon font now draws plainly is told, unless it did before the edit. And a badge
    layer whose badges all left while the icon font still draws something no badge layer names ("★" for the badge)
    goes, told."""
    psd = PSDImage.open(_card(clone, TURTLE)[0])
    text = pe.normalize_ability(pe.ability_layer(psd).text)
    edit = pe.set_text_layer(psd, "ability", text.replace("(1GCD)", "(GCD)"), str(clone / pe.FONTS_DIR), chart, parser=parser)
    flagged(edit.notes, r"^unsupported-card: ability: '\(GCD\)' set in the icon font, which draws no badge or icon of it$")
    psd = PSDImage.open(_card(clone, TURTLE)[0])
    edit = pe.set_text_layer(psd, "ability", text.replace("(1GCD)", "★"), str(clone / pe.FONTS_DIR), chart, parser=parser)
    flagged(edit.notes, r"^Ability FX: \(1GCD\) deleted: .*its badges left the ability, the icon font still draws something")


@text_edit
def test_a_typed_star_is_set_in_the_icon_font(clone, parser, chart):
    """Ruling C (review 2026-10-23): the website writes "5★"; the stoogedings font draws ★ as its gold star and the parser
    reads it so ([star]), but icon_text knew only "(tag)" and "[token]", so Standstill's "5 star or" made "5★ or" printed
    a missing-glyph box in the body font, every gate passing (the re-parse read "★" back). It is set in the icon font as
    a token is; canonical_tokens reads it as the parser does, but not on the re-parse's side (a "★" outside the font
    stays one), and a typed character its font has no glyph for is flagged font."""
    fonts = str(clone / pe.FONTS_DIR)
    psd = PSDImage.open(_card(clone, STANDSTILL)[0])
    text = pe.normalize_ability(pe.ability_layer(psd).text).replace("5 star or", "5★ or")
    edit = pe.set_text_layer(psd, "ability", text, fonts, chart, parser=parser)
    assert [run[:2] for run in _runs(pe.ability_layer(psd)) if run[1] == pe.ICON_FONT] == [("★", pe.ICON_FONT)]
    assert edit.fonts == [pe.ICON_FONT] and [text[a:b] for a, b in pe.icon_text(text, parser.icons)] == ["★"]
    assert pe.canonical_tokens("5★ or", parser.icons) == "5[star] or" and pe.canonical_tokens("5★", parser.icons, False) == "5★"
    psd = PSDImage.open(_card(clone, STANDSTILL)[0])
    flagged(pe.set_text_layer(psd, "ability", text.replace("5★", "5☃"), fonts, chart, parser=parser).notes,
            "^font: ability: '☃' has no glyph in its font")


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
    the skull is under it, as on the official cuts. And both checks let none out: a hole poked in it is told."""
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
    assert pe.gate_fx(psd, [edit], fonts_dir, chart, old) == []
    ys, xs = (want & cover).nonzero()
    x0, y0 = fx.bbox[:2]
    rgba = np.concatenate([fx.numpy("color"), fx.numpy("shape")], axis=2).astype(np.float32)
    rgba[ys[:6] - y0, xs[:6] - x0] = 0.0        # 6 of its 2,500 px: what the red key left out was 10
    pe._fx_engine()[6](fx, rgba, tuple(fx.bbox))
    fx._invalidate_bbox()
    assert pe.bare_icons(psd, fonts) == Counter({"[undead]": 1})
    flagged(pe.gate_fx(psd, [edit], fonts_dir, chart, old),
            r"^unsupported-card: ability: 'Ability FX: \(1GCD\) \| \[undead\]' leaves 6 px of its badges and icons out$")
    flagged(pe.gate_fx(psd, [SimpleNamespace(field="ability", fx_names=[])], fonts_dir, chart, old),
            r"^unsupported-card: ability: \[undead\]: under the Ability's look \(its Color Overlay\)")

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
    gates every colour-glyph pixel past FAINT_INK: a hole in the glow is told. A typed "★", which no badge phrase names
    (Turtle: 2 faint px of its glow left out, the gate told it), is the layer's too. A typed [support] is told all
    the same (see_through_icons): the layer over its see-through glow shows the overlay through it (review 2026-10-24,
    third round: Heavy Mole's olive-grey [support])."""
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
        notes = pe.gate_fx(psd, [edit], fonts_dir, chart, old, parser.icons)
        if "[" in tail:             # yet a see-through glow under a badge layer prints in the overlay's colour (review 3)
            flagged(notes, r"^unsupported-card: ability: \[support\]: see-through, under a badge FX layer")
        else:
            assert notes == []
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
    flagged(pe.gate_fx(psd, [edit], fonts_dir, chart, old, parser.icons),
            r"^unsupported-card: ability: 'Ability FX: \(1GCD\) \| \[support\]' leaves 6 px of its badges and icons out$")


@text_edit
def test_a_see_through_icon_under_a_badge_layer_is_flagged(clone, parser, chart, tmp_path):
    """Review 2026-10-24 (third round): a badge FX layer is a copy of the Ability's pixels over it, so a pixel at alpha a
    shows a*(1-a) of the tinted Ability below: [support]'s glow on Mini Doomer printed maroon, every gate passing. A
    typed see-through icon bakes, told (see_through_icons); an opaque one ([fire], (2CD)) bakes untold, its rim like the
    printed badges'."""
    from psd_tools.composite.text import FontMap
    fonts = FontMap(dirs=[str(clone / pe.FONTS_DIR)])
    old = pe.normalize_ability(pe.parse_card(parser, _card(clone, MINI_DOOMER)[0], MINI_DOOMER)["card"]["ability"])
    parsed, baked = _bake(clone, parser, MINI_DOOMER, {"ability": old + " Gain [warpstar]."}, tmp_path, chart)
    flagged(baked.fx_notes, r"^unsupported-card: ability: \[warpstar\]: see-through, under a badge FX layer, which prints "
                            r"its glow twice and in the Ability's overlay or blend colour$")
    parsed, baked = _bake(clone, parser, MINI_DOOMER, {"ability": old + " Gain [fire]. (2CD) Die."}, tmp_path, chart)
    saved = PSDImage.open(baked.psd_path)
    assert pe.see_through_icons(saved, fonts) == Counter() == pe.see_through_icons(PSDImage.open(_card(clone, MINI_DOOMER)[0]), fonts)


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
    icon-font cluster: the bake tells it, given the parser's vocabulary."""
    real, seen = pe.unshaped_icons, []

    def unshaped(psd, fonts, vocab, layer=None):
        seen.append(vocab)
        out = real(psd, fonts, vocab, layer=layer)
        return out + Counter({"[fire]": 1}) if vocab is not None and "[fire]" in pe.ability_layer(psd).text else out

    monkeypatch.setattr(pe, "unshaped_icons", unshaped)
    old = pe.normalize_ability(pe.parse_card(parser, _card(clone, HIVE)[0], HIVE)["card"]["ability"])
    assert "[fire]" not in old
    parsed, baked = _bake(clone, parser, HIVE, {"ability": old + " Gain [fire]."}, tmp_path)
    flagged(baked.fx_notes, r"^unsupported-card: ability: \[fire\]: not drawn as one badge or icon$")
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
def test_a_token_the_font_draws_as_letters_is_flagged(clone, parser, chart):
    """Review 2026-10-24: "[fire:]" counted as drawn (ICON_VARIANTS' "" is "[fire]"'s), so it was set in the icon font,
    which printed its letters (no ligature), cut into Mini Doomer's badge layer (white over the gradient) and read back
    as [fire], while the editor showed the fire icon: every gate passed. Typed, or made by adding a colon (Palace of
    Shadow's "[dark:sticker]", "[dark]" on the website): told as font."""
    fonts = str(clone / pe.FONTS_DIR)
    for rel, edit in ((MINI_DOOMER, lambda t: t + " Deal 1 [fire:] damage."),
                      (PALACE, lambda t: t.replace("[dark]", "[dark:]", 1))):
        psd_path = _card(clone, rel)[0]
        text = edit(pe.normalize_ability(pe.parse_card(parser, psd_path, rel)["card"]["ability"]))
        edit = pe.set_text_layer(PSDImage.open(psd_path), "ability", text, fonts, chart, parser=parser)
        flagged(edit.notes, r"^font: ability: \[\w+:\]: the website shows an icon, but the icon font has no such spelling "
                            r"and prints its letters$")


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
def test_a_plain_cards_body_font_tags_do_not_flag_a_typed_token(clone, parser, chart, tmp_path):
    """Review 2026-10-24: unshaped_icons counted nothing on a card without the icon font, so once a typed token added it
    the new side alone counted the tags Papyrus prints as body text ("(1CD)") and every typed badge or token on 10 such
    cards was refused. Both sides count them now: the typed token bakes untold, one left in the body font is told."""
    from psd_tools.composite.text import FontMap
    fonts_dir = str(clone / pe.FONTS_DIR)
    fonts = FontMap(dirs=[fonts_dir])
    psd_path = _card(clone, KORO)[0]
    old = PSDImage.open(psd_path)
    assert not pe.uses_icon_font(pe.ability_layer(old)) and pe.unshaped_icons(old, fonts, parser.icons) == Counter(
        {"(1CD)": 1, "(1GCD)": 1, "(1PG)": 1, "(ENDGT)": 1})
    new = pe.normalize_ability(pe.parse_card(parser, psd_path, KORO)["card"]["ability"]) + " Deal 1 [fire] damage."
    parsed, baked = _bake(clone, parser, KORO, {"ability": new}, tmp_path, chart)
    assert pe.normalize_ability(baked.reparse["card"]["ability"]) == new
    assert not [note for note in baked.fx_notes if "not drawn as one" in note]
    psd = PSDImage.open(psd_path)
    layer = pe.ability_layer(psd)
    text = layer.text.rstrip("\r") + " [fire]"
    pe.splice_text(layer, text, pe.edit_parts(layer.text, text, *pe.changed_range(layer.text, text)))
    layer.rasterize(fonts=fonts)
    assert not pe.uses_icon_font(layer)
    flagged(pe.gate_fx(psd, [SimpleNamespace(field="ability", fx_names=[])], fonts_dir, None, old, parser.icons),
            r"^unsupported-card: ability: \[fire\]: not drawn as one badge or icon$")


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


@text_edit
def test_a_blank_ability_on_an_fx_card_bakes_flagged(clone, parser, chart, tmp_path):
    """Review 2026-10-24: deleting a whole ability on an FX card refused styled-word ('the text layer has no pixels to
    cut') before looking at its layers, though every one leaves with its words. Each is deleted (told), and the blank
    ability bakes with the parser's new problem told (owner 2026-10-24)."""
    psd = PSDImage.open(_card(clone, SNOW_BUNNY)[0])
    edit = pe.set_text_layer(psd, "ability", "", str(clone / pe.FONTS_DIR), chart, parser=parser)
    assert edit.dropped == [FROZEN] and edit.fx_names == [] and edit.notes[0].startswith(f"{FROZEN} deleted")
    parsed, baked = _bake(clone, parser, SNOW_BUNNY, {"ability": ""}, tmp_path, chart)
    flagged(baked.fx_notes, r"^validator: new problems: ABILITY TEXT NOT FOUND$")


@text_edit
def test_an_edit_inside_an_own_style_phrase_drops_its_layer(clone, chart):
    """Flameblade's "Attacks inflict 2T" (its own gradient) made "3T": the layer can't follow, so it goes, told; its
    words print plain (owner 2026-10-24)."""
    psd = PSDImage.open(_card(clone, FLAMEBLADE)[0])
    edit = pe.set_text_layer(psd, "ability", pe.ability_layer(psd).text.replace("2T", "3T"), str(clone / pe.FONTS_DIR), chart)
    assert edit.dropped == ["Ability FX: Attacks inflict 2T"]
    flagged(edit.notes, r"^Ability FX: Attacks inflict 2T deleted: the edit changes its 'Attacks inflict 2T' \(now "
                        r"'Attacks inflict 3T'\)")


@text_edit
def test_an_fx_phrase_the_ability_lacks_drops_its_layer(clone, chart):
    """A hand-named "Ability FX: Frozen In Time #1" (a space before "#": no number to Effect Layers.jsx, so no status
    either) crashed plan_fx (fx_kind called it a Frozen In Time layer, the chart lookup a "frozen in time #1" one); its
    phrase is in no text, so what it covers can't be told: it goes, told."""
    base = SimpleNamespace(name="Ability")
    assert pe.fx_kind(SimpleNamespace(name="Ability FX: Frozen #2", kind="pixel", tagged_blocks=None), base, chart) == "own"
    psd = PSDImage.open(_card(clone, STANDSTILL)[0])
    layer = pe.ability_layer(psd)
    fx = next(fx for fx in pe.fx_layers(psd, layer) if fx.name.endswith("#1"))
    fx.name = "Ability FX: Frozen In Time #1"
    edit = pe.set_text_layer(psd, "ability", pe.normalize_ability(layer.text) + " Die.", str(clone / pe.FONTS_DIR), chart)
    assert "Ability FX: Frozen In Time #1" in edit.dropped
    flagged(edit.notes, r"^Ability FX: Frozen In Time #1 deleted: its phrases are not in the ability: Frozen In Time #1$")


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


def test_icon_style_breaks_a_size_tie_by_the_body_size(parser, monkeypatch):
    """Owner 2026-10-24: two badge sizes as common, the body text's size wins (Mr Krabs' 17.49 pt and 30 pt badges);
    then the first. The 32 pt badge comes second here."""
    text = "(1CD)(2U) Die."
    layer = SimpleNamespace(text=text)
    style = lambda size: {"Font": pe.ICON_FONT, "FontSize": size}
    for body, want in ((32.0, 32.0), (11.0, 18.0)):
        styles = [style(18.0)] * 5 + [style(32.0)] * 4 + [{"Font": "Body", "FontSize": body}] * (len(text) - 9)
        monkeypatch.setattr(pe, "_char_styles", lambda _: ([None] * len(text), [i < 9 for i in range(len(text))],
                                                           styles, [0] * len(text)))
        assert pe.icon_style(layer, True, parser.icons)["FontSize"] == want


@text_edit
def test_a_typed_badge_takes_the_cards_usual_badge_size(clone, parser, chart, tmp_path):
    """Owner 2026-10-24: a newly typed badge or token takes the card's commonest badge size in that text, not the style
    of the badge beside it the engine gives it (Pizza's odd 18.66 pt first badge: a (2U) typed after it was 18.66 pt);
    an old badge edited in place keeps its own ("(1GCD)" made "(2GCD)": 18.66, its letters one style)."""
    fonts = str(clone / pe.FONTS_DIR)
    pizza = _at(clone, tmp_path, COLOURED, PIZZA)[0]
    big = pe.icon_style(pe.ability_layer(PSDImage.open(pizza)), True, parser.icons)
    for old, new, want in (("(1GCD) Remove", "(1GCD)(2U) Remove", [("(1GCD)", 18.66), ("(2U)", 32.0)]),
                           ("(1GCD) Remove", "(2U) (1GCD) Remove", [("(2U)", 32.0), ("(1GCD)", 18.66)]),
                           ("(1GCD) Remove", "(1GCD)[fire] Remove", [("(1GCD)", 18.66), ("[fire]", 32.0)]),
                           ("(1GCD) Remove", "(2GCD) Remove", [("(2GCD)", 18.66)])):
        psd = _unclipped(PSDImage.open(pizza))
        pe.set_text_layer(psd, "ability", pe.ability_layer(psd).text.replace(old, new, 1), fonts, chart, parser=parser)
        layer = pe.ability_layer(psd)
        runs = [run for run in _runs(layer) if run[1] == pe.ICON_FONT]
        assert [(run[0], run[2]) for run in runs[:len(want)]] == want, (new, runs)
        styles = pe._char_styles(layer)[2]
        assert all(pe._canon(styles[layer.text.index(piece)]) == pe._canon(big) for piece, size in want if size == 32.0)


PALACE = "Field/4 Stars/Palace_of_Shadow.psd"   # [dark:sticker] white on its white first line, [light:sticker] in grey


def test_variant_fill_takes_its_kinds_colour(parser):
    """Owner 2026-10-24: a typed :sticker or :tint takes the colour of its kind on the card, the same icon's first, the
    edited text's before the other's; none: the surrounding text's (None); a tie refuses."""
    white, grey, red = (1.0, 1.0, 1.0), (0.2, 0.2, 0.2), (1.0, 0.0, 0.0)
    palace = [("dark", "sticker", white), ("light", "sticker", grey)]
    assert pe.variant_fill("[Dark:Sticker]", parser.icons, [palace, []]) == white
    assert pe.variant_fill("[light:sticker]", parser.icons, [palace, []]) == grey
    assert pe.variant_fill("[fire:tint]", parser.icons, [palace, [("dark", "tint", red)]]) == red
    assert pe.variant_fill("[fire:tint]", parser.icons, [palace, []]) is None
    assert pe.variant_fill("[fire]", parser.icons, [palace, []]) is pe.variant_fill("[dark:no]", parser.icons, [palace]) is None
    assert pe.variant_fill("[fire:sticker]", parser.icons, [palace + [("ice", "sticker", grey)], []]) == grey
    with pytest.raises(pe.Refused, match=r"^unsupported-card: \[fire:sticker\]: the card's :sticker tokens are in as many"):
        pe.variant_fill("[fire:sticker]", parser.icons, [palace, []])


@text_edit
def test_a_typed_sticker_takes_the_colour_of_its_kind(clone, parser, chart):
    """Owner 2026-10-24: Palace of Shadow's [dark:sticker] is white, its [light:sticker] the body's grey: a [dark:sticker]
    typed in the grey text is white (it was grey), a [light:sticker] grey, a :tint (none on the card) the text's grey,
    and a [fire:sticker] (one white, one grey: which can't be told) takes the first, told."""
    fonts = str(clone / pe.FONTS_DIR)
    white, grey = (1.0, 1.0, 1.0), (0.2, 0.2, 0.2)
    for typed, want in (("[dark:sticker]", white), ("[light:sticker]", grey), ("[fire:tint]", grey)):
        psd = PSDImage.open(_card(clone, PALACE)[0])
        layer = pe.ability_layer(psd)
        pe.set_text_layer(psd, "ability", layer.text + f" Gain {typed}.", fonts, chart, parser=parser)
        layer = pe.ability_layer(psd)
        assert [run[1:] for run in _runs(layer) if run[0] == typed][-1] == (pe.ICON_FONT, 33.33, want), typed
    psd = PSDImage.open(_card(clone, PALACE)[0])
    edit = pe.set_text_layer(psd, "ability", pe.ability_layer(psd).text + " Gain [fire:sticker].", fonts, chart, parser=parser)
    flagged(edit.notes, r"^unsupported-card: ability: \[fire:sticker\]: the card's :sticker tokens are in as many of one "
                        r"colour as of another; the typed one takes the first$")


BENCH = "Field/1 Stars/Bench.psd"           # its title 'Bench', no effects, room for an icon after it
MANGLE = "Creatures/Fnaf/3 Stars/Mangle.psd"   # a stroked title
NEO = "Auxiliary/Creatures/Mettaton_NEO.psd"   # its title under a gradient overlay


@text_edit
def test_a_token_typed_in_the_title_is_set_in_the_icon_font(clone, parser, chart, tmp_path):
    """Owner 2026-10-24: a token typed in the title is set in the icon font too, at the title's own size and colour,
    under the same gates. The parser reads a title's icons as its name, not as a header's types or stars (the re-parse
    read Bench's [fire] as a fire type and refused). Told: a token under the title's gradient overlay (Mettaton NEO);
    a timing badge typed there prints as plain words, as the website shows it; a glow is set too (fork
    cards-20261024b); without the vocabulary a title has no tokens."""
    parsed, baked = _bake(clone, parser, BENCH, {"name": "Bench [fire]"}, tmp_path, chart)
    assert baked.reparse["card"]["display_name"] == "Bench [fire]"
    assert {key: baked.reparse["card"].get(key) for key in ("types", "stars")} == \
        {key: parsed["card"].get(key) for key in ("types", "stars")}
    runs = _runs(pe.name_layer(PSDImage.open(baked.psd_path), "Bench"))
    assert [run[0] for run in runs] == ["Bench ", "[fire]"] and runs[1][1] == pe.ICON_FONT and runs[1][2:] == runs[0][2:]
    fonts = str(clone / pe.FONTS_DIR)
    psd = PSDImage.open(_card(clone, MANGLE)[0])
    pe.set_text_layer(psd, "name", "★ Mangle", fonts, chart, "Mangle", parser=parser)
    runs = _runs(pe.name_layer(psd, "Mangle"))
    assert runs[0][:2] == ("★", pe.ICON_FONT) and runs[0][2:] == runs[1][2:]
    psd = PSDImage.open(_card(clone, NEO)[0])
    flagged(pe.set_text_layer(psd, "name", "Mettaton NEO [fire]", fonts, chart, "Mettaton NEO", parser=parser).notes,
            r"^unsupported-card: name: \[fire\]: the title's look \(its Gradient Overlay\) recolours the typed icon$")
    psd = PSDImage.open(_card(clone, BENCH)[0])
    edit = pe.set_text_layer(psd, "name", "Bench (1CD)", fonts, chart, "Bench", parser=parser)
    flagged(edit.notes, r"^unsupported-card: name: \(1CD\): a timing badge typed in the title, printed as plain words")
    assert pe.ICON_FONT not in [run[1] for run in _runs(pe.name_layer(psd, "Bench"))]
    psd = PSDImage.open(_card(clone, BENCH)[0])
    own = _runs(pe.name_layer(psd, "Bench"))[0][1]
    pe.set_text_layer(psd, "name", "Bench [fire:glow]", fonts, chart, "Bench", parser=parser)
    assert [run[:2] for run in _runs(pe.name_layer(psd, "Bench"))] == [("Bench ", own), ("[fire:glow]", pe.ICON_FONT)]
    psd = PSDImage.open(_card(clone, BENCH)[0])
    pe.set_text_layer(psd, "name", "[ice]", fonts, chart, "Bench")
    assert pe.ICON_FONT not in [run[1] for run in _runs(pe.name_layer(psd, "Bench"))]


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


RADIANCE = "Creatures/Hollow Knight/5 Stars/The_Radiance.psd"   # its title in 'Group 1': a Gradient Overlay, no own one
SPACEBOY = "Creatures/Omori/3 Stars/Captain_Spaceboy.psd"      # its title in 'Stat Txt': a Gradient Overlay
SALT = "Auxiliary/Items/Salt.psd"                               # a white title, no :sticker on the card
OLD_FOXY = "Creatures/Fnaf/4 Stars/Old_Foxy.psd"                # its title 'Old Foxy\r': a blank last paragraph
LESHY = "Creatures/Other/5 Stars/Leshy.psd"                     # its title under its art ('Layer 6', 'Layer 7')
DOOPLISS = "Creatures/Mario/4 Stars/DooPliss.psd"               # its title under its art ('p')


@text_edit
def test_a_token_under_a_groups_overlay_is_flagged(clone, parser, chart):
    """Review 2026-10-24: _tinted read only the title's own overlays, so a [fire] typed in The Radiance's title ('Group 1'
    has a Gradient Overlay) baked as a cream silhouette, Captain Spaceboy's as a cyan one, untold. A group's effects
    count as the layer's own, and so do its Inner Glow, Satin and Inner Shadow (Butterfly's): the edit is told."""
    fonts = str(clone / pe.FONTS_DIR)
    for rel, card in ((SPACEBOY, "Captain Spaceboy"), (RADIANCE, "The Radiance")):
        layer = pe.name_layer(PSDImage.open(_card(clone, rel)[0]), card)
        assert not pe._overlaid(layer) and pe._tinted(layer).startswith("the Gradient Overlay of its group ")
    psd = PSDImage.open(_card(clone, RADIANCE)[0])
    edit = pe.set_text_layer(psd, "name", pe.collapse(pe.name_layer(psd, "The Radiance").text) + " [fire]", fonts, chart,
                             "The Radiance", parser=parser)
    flagged(edit.notes, r"^unsupported-card: name: \[fire\]: the title's look \(the Gradient Overlay of its group ")
    assert pe._tinted(pe.name_layer(PSDImage.open(_card(clone, BUTTERFLY)[0]), "Butterfly")) == "its Inner Glow"


@text_edit
def test_a_title_sticker_takes_the_titles_colour_beside_a_typed_ability_one(clone, parser, chart, tmp_path):
    """Owner 2026-10-24 (ruling 2), review 2026-10-24: the Ability is edited first, so a [dark:sticker] typed in both
    took the Ability's colour in Salt's white title (purple), read off the edited copy as "already on the card". The
    other layer's colours are the card's before the bake: none there, the title's own white."""
    old = pe.normalize_ability(pe.parse_card(parser, _card(clone, SALT)[0], SALT)["card"]["ability"])
    parsed, baked = _bake(clone, parser, SALT, {"name": "Salt [dark:sticker]", "ability": old + " Gain [dark:sticker]."},
                          tmp_path, chart)
    runs = _runs(pe.name_layer(PSDImage.open(baked.psd_path), "Salt"))
    assert [run[1:] for run in runs if run[0] == "[dark:sticker]"] == [(pe.ICON_FONT, runs[0][2], runs[0][3])]
    assert runs[0][3] == (1.0, 1.0, 1.0)


@text_edit
def test_a_title_edit_keeps_its_blank_last_paragraph_off_the_new_words(clone, parser, chart):
    """Review 2026-10-24: keep_breaks paired the space before a word appended to "Old Foxy\r" with its trailing break
    (shown as a space), setting the word on a new line, and keep_ends added the break again. The blank end paragraphs
    are keep_ends' alone."""
    assert pe.keep_breaks("Old Foxy\r", "Old Foxy [fire]", "name") == "Old Foxy [fire]"
    assert pe.keep_ends("Old Foxy\r", "Old Foxy II") == "Old Foxy II\r"
    psd = PSDImage.open(_card(clone, OLD_FOXY)[0])
    pe.set_text_layer(psd, "name", "Old Foxy [fire]", str(clone / pe.FONTS_DIR), chart, "Old Foxy", parser=parser)
    assert pe.name_layer(psd, "Old Foxy").text == "Old Foxy [fire]\r"


@text_edit
def test_words_typed_after_a_title_token_take_the_titles_style(clone, parser, chart):
    """Review 2026-10-24: the engine gives text typed right after a token its icon font, and a title edit had no
    edit_styles: "Bench [fire] II" refused unsupported-card ('II' in the icon font). It takes the title's own style
    (title_styles); "[fire:]", which the font draws as letters and the website shows as letters in a title, stays in
    the title's font without the Ability's font refusal."""
    fonts = str(clone / pe.FONTS_DIR)
    psd = PSDImage.open(_card(clone, BENCH)[0])
    for title in ("Bench [fire]", "Bench [fire] II"):
        pe.set_text_layer(psd, "name", title, fonts, chart, "Bench", parser=parser)
    runs = _runs(pe.name_layer(psd, "Bench"))
    assert [run[:2] for run in runs] == [("Bench ", runs[0][1]), ("[fire]", pe.ICON_FONT), (" II", runs[0][1])]
    assert runs[2][2:] == runs[0][2:]
    psd = PSDImage.open(_card(clone, BENCH)[0])
    pe.set_text_layer(psd, "name", "[fire:]", fonts, chart, "Bench", parser=parser)
    assert [run[:2] for run in _runs(pe.name_layer(psd, "Bench"))] == [("[fire:]", runs[0][1])]


BARRY = "Creatures/Other/2 Stars/Barry_The_Barrel.psd"   # its badges and tokens at tracking 280
GUST = "Items/Support/2 Stars/Gust_of_Wind.psd"            # its title's ligatures off
PEPPERONI = "Items/Heal/2 Stars/Pepperoni_Pizza.psd"       # its title 'Pepperoni\r\rPizza'
PO = "Creatures/Pandas/4 Stars/Po.psd"                     # types ['panda'] as image layers, room after its title
BUTTERFLY = "Creatures/Kirby/5 Stars/Butterfly.psd"        # its title's own Inner Glow, no overlay


def _typed_tokens(layer, fonts_dir: str, vocab) -> dict:
    """{token: (Tracking, Ligatures, spread)} per icon-font token; spread: px between its glyphs' ink centres."""
    from psd_tools.composite.text import FontMap
    from psd_tools.composite.text.geometry import text_geometry
    styles = pe._char_styles(layer)[2]
    glyphs = text_geometry(layer, FontMap(dirs=[fonts_dir]))["glyphs"]
    out = {}
    for a, b in pe.icon_text(layer.text, vocab):
        centres = [(glyph["ink"][0] + glyph["ink"][2]) / 2 for glyph in glyphs if glyph["i"] == a and glyph["ink"]]
        out[layer.text[a:b]] = (pe._canon(styles[a].get("Tracking")), styles[a].get("Ligatures"),
                                max(centres) - min(centres))
    return out


@text_edit
def test_typed_layered_tokens_are_untracked_and_ligated(clone, parser, chart):
    """Review 2026-10-24: a typed :sticker or :no took the card's tracking (Barry's tokens 280, The Hive's title 240),
    which the fork adds between the overlapping glyphs of one cluster: the outline or no-circle slid off its icon, every
    gate passing (no :sticker or :no on main is tracked). They are set at tracking 0, a plain [fire] keeping the
    card's. Every typed token has ligatures on: Gust of Wind's title has them off, under which Photoshop may print the
    letters while the engine (rlig) draws the icon."""
    fonts = str(clone / pe.FONTS_DIR)
    psd = PSDImage.open(_card(clone, BARRY)[0])
    layer = pe.ability_layer(psd)
    assert pe._canon(pe.icon_style(layer, False, parser.icons)["Tracking"]) == 280
    pe.set_text_layer(psd, "ability", pe.normalize_ability(layer.text) + " Gain [dark:sticker], [fire:no] and [fire].",
                      fonts, chart, parser=parser)
    got = _typed_tokens(pe.ability_layer(psd), fonts, parser.icons)
    assert [got[token][:2] for token in ("[dark:sticker]", "[fire:no]", "[fire]")] == [(0, True), (0, True), (280, True)]
    assert got["[dark:sticker]"][2] <= 2 and got["[fire:no]"][2] <= 2
    psd = PSDImage.open(_card(clone, HIVE)[0])
    pe.set_text_layer(psd, "name", "The Hive [fire:no]", fonts, chart, "The Hive", parser=parser)
    tracking, ligatures, spread = _typed_tokens(pe.name_layer(psd, "The Hive"), fonts, parser.icons)["[fire:no]"]
    assert (tracking, ligatures) == (0, True) and spread <= 2
    psd = PSDImage.open(_card(clone, GUST)[0])
    assert not pe._char_styles(pe.name_layer(psd, "Gust of Wind"))[2][0].get("Ligatures")
    pe.set_text_layer(psd, "name", "Gust of Wind [fire]", fonts, chart, "Gust of Wind", parser=parser)
    assert _typed_tokens(pe.name_layer(psd, "Gust of Wind"), fonts, parser.icons)["[fire]"][1]


@text_edit
def test_a_title_edit_keeps_its_runs_of_blanks(clone, parser, chart):
    """Review 2026-10-24: the website shows a title's run of blanks as one space, and keep_breaks paired one break of
    "Pepperoni\r\rPizza" with it: the PIZZA line moved up onto PEPPERONI, every gate passing. Runs pair as runs ("The
    Spire"'s two spaces, a centred line's leading space); spaces typed as two stay as typed."""
    for old, new, want in (("Pepperoni\r\rPizza", "Pepperoni Pizza [fire]", "Pepperoni\r\rPizza [fire]"),
                           ("The  Spire", "The Spire [fire]", "The  Spire [fire]"),
                           ("Spiky Ball\r Trap", "Spiky Ball Trap II", "Spiky Ball\r Trap II"),
                           ("Mega\rBee", "Mega  Bees", "Mega  Bees"), ("The Spire", "The Spire", "The Spire")):
        assert pe.keep_breaks(old, new, "name") == want
    psd = PSDImage.open(_card(clone, PEPPERONI)[0])
    assert pe.name_layer(psd, "Pepperoni Pizza").text.rstrip("\r") == "Pepperoni\r\rPizza"
    pe.set_text_layer(psd, "name", "Pepperoni Pizza [fire]", str(clone / pe.FONTS_DIR), chart, "Pepperoni Pizza",
                      parser=parser)
    assert pe.name_layer(psd, "Pepperoni Pizza").text.rstrip("\r") == "Pepperoni\r\rPizza [fire]"


@text_edit
def test_a_title_token_is_no_header_type_row(clone, parser, chart, tmp_path):
    """Review 2026-10-24: once "Po [fire]" was baked, type_row counted the title's token as a header type token and every
    types edit after it refused ("printed as icon tokens in 'Po'"). The title's icons are the name's, as the parser
    reads them."""
    _, baked = _bake(clone, parser, PO, {"name": "Po [fire]"}, tmp_path, chart)
    psd = PSDImage.open(baked.psd_path)
    parsed = pe.parse_card(parser, baked.psd_path, PO)
    assert parsed["card"]["types"] == ["panda"] and pe.type_row(psd, parser, "Po")[1] == []
    assert pe.plan_types(psd, parsed, ["panda", "ice"], parser).new == ["panda", "ice"]


FOOL_EATER = "Auxiliary/Minions/Fool_Eater.psd"     # its 'Defensive_Minion' type icon at the root, left of the title
KRUSTY_KRAB = "Field/3 Stars/The_Krusty_Krab.psd"    # its 'Field' type icon at the root
SPICE = "Items/Support/2 Stars/Spice.psd"            # its star 'images/Stuffed_Star copy' right of the title
NOSK = "Creatures/Hollow Knight/4 Stars/Nosk.psd"    # its 'Hk star' just left of the title


@text_edit
def test_a_title_token_off_the_header_icons_and_stars(clone, parser, chart, monkeypatch):
    """Review 2026-10-24: a typed title token printed over a header type icon or star, untold: the title check found
    type icons by name only in a *types* group (Fool Eater's and The Krusty Krab's sit at the root) and no star in
    'images' (Spice's). badge_layers takes the vocabulary; and the title's new ink over a visible layer below it is
    told (ink_beyond, inks_over), whatever its name."""
    fonts = str(clone / pe.FONTS_DIR)
    for rel, card, title in ((FOOL_EATER, "Fool Eater", "[fire] Fool Eater"), (KRUSTY_KRAB, "The Krusty Krab", "The Krusty Krab [fire]")):
        psd = PSDImage.open(_card(clone, rel)[0])
        assert badge_layers(psd)[1] == [] and badge_layers(psd, parser.all_types)[1]
        flagged(pe.set_text_layer(psd, "name", title, fonts, chart, card, parser=parser).notes,
                r"^overflow: name reaches the stars or type icons$")
    psd = PSDImage.open(_card(clone, SPICE)[0])
    flagged(pe.set_text_layer(psd, "name", "Spice [fire]", fonts, chart, "Spice", parser=parser).notes,
            r"^overflow: name: .*'Stuffed_Star copy'")
    monkeypatch.setattr(pe, "badge_layers", lambda psd, vocab=(), rulebook=None: ([], [], []))   # by pixels alone
    psd = PSDImage.open(_card(clone, FOOL_EATER)[0])
    flagged(pe.set_text_layer(psd, "name", "[fire] Fool Eater", fonts, chart, "Fool Eater", parser=parser).notes,
            r"^overflow: name: .*'Defensive_Minion'")
    psd = PSDImage.open(_card(clone, BENCH)[0])
    assert pe.set_text_layer(psd, "name", "Bench [fire]", fonts, chart, "Bench", parser=parser).notes == []


def test_star_and_type_plans_see_the_root_type_icons(clone, parser, monkeypatch):
    """Review 2026-10-24: plan_stars' and plan_types' badge_layers calls take the vocabulary, as set_text_layer's."""
    real, seen = pe.badge_layers, []
    monkeypatch.setattr(pe, "badge_layers", lambda psd, vocab=(), rulebook=None: seen.append(list(vocab)) or real(psd, vocab, rulebook))
    psd = PSDImage.open(_card(clone, MINI_DOOMER)[0])
    parsed = pe.parse_card(parser, _card(clone, MINI_DOOMER)[0], MINI_DOOMER)
    pe.plan_stars(psd, parsed, parsed["card"]["stars"] + 1, parser)
    psd = PSDImage.open(_card(clone, PO)[0])
    pe.plan_types(psd, pe.parse_card(parser, _card(clone, PO)[0], PO), ["panda", "ice"], parser)
    assert seen and all(vocab == list(parser.all_types) for vocab in seen)


MACHINE_GUN = "Auxiliary/Items/Machine_Gun.psd"      # a pixel font: its letters a quarter of its em
HOLY_BLADE = "Items/Equip/4 Stars/Holy_Blade.psd"    # ChiQuel: its letters 2.9 times its em
NIIUE = "Creatures/Other/5 Stars/Niiue.psd"          # leading 0.46 of its size
BASIC_SHIELD = "Items/Equip/2 Stars/Basic_Shield.psd"   # its title box text, 224x46
CURSED_IDOL = "N.M.E/The_Evil,_Fricked_Up_Cursed_Idol.psd"


@text_edit
def test_a_typed_icon_far_from_its_letters_is_flagged(clone, parser, chart):
    """Review 2026-10-24: a typed icon at the text's own size is about an em, and some fonts draw their letters far
    from it: Machine Gun's [fire] printed over 'MACHINE' (763 px of its letters), Holy Blade's a speck (0.2 of its
    letters). typed_icon_fit tells past ICON_OVERLAP px over the other glyphs' ink (overflow), and outside ICON_HEIGHT
    of its line's letters (unsupported-card; an icon at the card's own badge style: no height check)."""
    fonts = str(clone / pe.FONTS_DIR)
    for rel, card, text, why in (
            (MACHINE_GUN, "Machine Gun", "MACHINE GUN [fire]",
             r"^overflow: name: \[fire\] would print over the letters beside it \(\d+ px\)$"),
            (HOLY_BLADE, "Holy Blade", "Holy Blade [fire]",
             r"^unsupported-card: name: \[fire\] at the text's font size would be 0\.2\dx the height of its letters")):
        psd = PSDImage.open(_card(clone, rel)[0])
        flagged(pe.set_text_layer(psd, "name", text, fonts, chart, card, parser=parser).notes, why)
    assert pe.ICON_OVERLAP == 16 and pe.ICON_HEIGHT == (0.6, 2.2)


@text_edit
def test_a_title_token_that_wraps_is_flagged(clone, parser, chart, monkeypatch):
    """Review 2026-10-24: Basic Shield's title is box text: "Basic Shield [fire]" wrapped the [fire] onto a second line
    the fork draws past the box, over the art. A title edit on more lines is told (overflow), and so is a line set
    below the old title."""
    fonts = str(clone / pe.FONTS_DIR)
    psd = PSDImage.open(_card(clone, BASIC_SHIELD)[0])
    flagged(pe.set_text_layer(psd, "name", "Basic Shield [fire]", fonts, chart, "Basic Shield", parser=parser).notes,
            r"^overflow: name: it runs onto 2 lines \(it has 1\)$")
    real, calls = pe._title_lines, []          # and no line of it below the old title (a box's last line)
    monkeypatch.setattr(pe, "_title_lines", lambda geometry: [y + 1000 * len(calls) for y in real(geometry)]
                        if calls.append(1) or True else None)
    edit = pe.set_text_layer(PSDImage.open(_card(clone, BENCH)[0]), "name", "Bench [fire]", fonts, chart, "Bench",
                             parser=parser)
    flagged(edit.notes, r"^overflow: name: a line of it moves below the title$")


@text_edit
def test_apply_jobs_runs_the_title_icon_checks(clone, parser, monkeypatch, tmp_path):
    """Review 2026-10-24: apply_jobs gives gate_fx the title of a title edit; nothing failed when it gave None. The
    layout here says the title's typed [fire] is not one icon-font cluster: the bake tells it."""
    real = pe.unshaped_icons

    def unshaped(psd, fonts, vocab, layer=None):
        out = real(psd, fonts, vocab, layer=layer)
        return out + Counter({"[fire]": 1}) if layer is not None and "[fire]" in layer.text else out

    monkeypatch.setattr(pe, "unshaped_icons", unshaped)
    parsed, baked = _bake(clone, parser, BENCH, {"name": "Bench [fire]"}, tmp_path)
    flagged(baked.fx_notes, r"^unsupported-card: name: \[fire\]: not drawn as one badge or icon$")


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
def test_covered_ink_weighs_layers_over_the_old_ink_anywhere(clone):
    """Review 2026-10-24: a layer above the text that lay over the old text's ink is the design, wherever that ink was
    (covered_ink's box takes the old ink with the new): Leshy's art layers over a made-up old title that filled the
    canvas around the real one pass; looked for over the new ink's box alone, they would refuse."""
    psd = PSDImage.open(_card(clone, LESHY)[0])
    layer = pe.name_layer(psd, "Leshy")
    before = np.zeros((psd.height, psd.width, 4), np.uint8)
    before[..., 3] = 255
    x0, y0, x1, y1 = layer.bbox
    before[max(y0, 0):y1, max(x0, 0):x1] = 0
    assert pe.covered_ink(psd, layer, np.zeros_like(before), []) is not None     # the real title under its art
    assert pe.covered_ink(psd, layer, before, []) is None


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


CONFUSION = "Auxiliary/Status Effects/Debuffs/2 Stars/Confusion.psd"     # a stroked title, its stars above it


@text_edit
@pytest.mark.skipif(not pe.glow_halos(), reason="the fork draws no glow halos")
def test_a_glow_disc_under_a_star_is_flagged(clone, parser, chart):
    """Review 2026-10-24: a stroked title's typed glow prints a black disc, which ran 1,364 px under Confusion's 'star
    copy 2' (and Krabby Patty's 'Food' icon) untold: only the title's own ink was checked. Its changed effect ink counts
    on the stars and type icons (a clipped one where its base inks too) and under the layers above (covered_ink); a
    token clear of them sets untold."""
    fonts = str(clone / pe.FONTS_DIR)
    for rel, typed in ((CONFUSION, "{} [ice:darkglow]"), ("Auxiliary/Items/Krabby_Patty.psd", "[fire:glow] {}")):
        path = _card(clone, rel)[0]
        parsed = pe.parse_card(parser, path, rel)
        name, card, own = pe.shown_name(parsed), pe._card_name(parsed), pe._own_text(parsed)
        edit = pe.set_text_layer(PSDImage.open(path), "name", typed.format(name), fonts, chart, card, own, parser,
                                 PSDImage.open(path))
        flagged(edit.notes, "^overflow: name: a glow token's halo reaches the stars or type icons$")
    path = _card(clone, CONFUSION)[0]
    parsed = pe.parse_card(parser, path, CONFUSION)
    name, card, own = pe.shown_name(parsed), pe._card_name(parsed), pe._own_text(parsed)
    psd = PSDImage.open(path)
    layer = pe.name_layer(psd, card, own)
    star = next(other for other in psd.descendants() if other.name == "star copy 2")
    x0, y0 = star.bbox[:2]
    ys, xs = (pe._alpha(star) > 0.9).nonzero()
    canvas = pe._on_canvas(layer, (psd.height, psd.width))
    after, before = np.zeros_like(canvas), np.zeros_like(canvas)
    after[y0 + ys[0], x0 + xs[0]] = 255
    assert pe.covered_ink(psd, layer, canvas, []) is None and pe.covered_ink(psd, layer, canvas, [], (before, before)) is None
    assert pe.covered_ink(psd, layer, canvas, [], (after, before)) == "'star copy 2' above it would cover 1 px of the new text"
    assert not [n for n in pe.set_text_layer(psd, "name", name + " [fire]", fonts, chart, card, own, parser,
                                             PSDImage.open(path)).notes if "stars" in n]
    rel = "Field/3 Stars/Elemental_World.psd"   # its disc under the part of a clipped ice texture its icon hides
    path = _card(clone, rel)[0]
    parsed = pe.parse_card(parser, path, rel)
    edit = pe.set_text_layer(PSDImage.open(path), "name", pe.shown_name(parsed) + " [ice:darkglow]", fonts, chart,
                             pe._card_name(parsed), pe._own_text(parsed), parser, PSDImage.open(path))
    assert not [n for n in edit.notes if "halo" in n]


@text_edit
def test_a_glow_disc_beside_a_layer_composited_with_it_counts(clone, parser, monkeypatch):
    """In its effect scope's group (Crock Pot's stroked Text) a layer above the text hides the halo in the composite
    _effect_ink draws, so the halo's changed ink beside its ink counts (covered_ink), and a star or type icon there
    counts by its box (_meets)."""
    path = _card(clone, "N.M.E/Crock_Pot.psd")[0]
    psd = PSDImage.open(path)
    parsed = pe.parse_card(parser, path, "N.M.E/Crock_Pot.psd")
    title, layer = pe.name_layer(psd, pe._card_name(parsed), pe._own_text(parsed)), pe.ability_layer(psd)
    badge = next(other for other in psd.descendants() if other.name == "NME")
    shared = pe._scoped(layer)
    assert {id(title._record), id(badge._record)} <= shared
    mask = np.zeros((psd.height, psd.width), bool)
    ys, xs = (pe._alpha(badge) <= pe.INK_ALPHA).nonzero()
    mask[badge.bbox[1] + ys[0], badge.bbox[0] + xs[0]] = True
    assert not pe._meets(mask, badge) and pe._meets(mask, badge, whole=True)
    x0, y0, x1, y1 = title.bbox
    drawn = np.asarray(title.composite(viewport=(x0, y0, x1, y1), apply_icc=False).convert("RGBA"))[..., 3] > pe.INK_ALPHA * 255
    edge = drawn & ~np.roll(drawn, 1, axis=1)               # its ink with no ink left of it: the pixel there is beside it
    ys, xs = edge[:, 1:].nonzero()
    canvas = pe._on_canvas(layer, (psd.height, psd.width))
    after, before = np.zeros_like(canvas), np.zeros_like(canvas)
    after[y0 + ys[0], x0 + xs[0]] = 255
    assert search(rf"^'{title.name}' above it would cover \d+ px of the new text$",
                  pe.covered_ink(psd, layer, canvas, [], (after, before)) or "")
    monkeypatch.setattr(pe, "_scoped", lambda layer: set())    # outside the group: only ink on its ink counts
    assert pe.covered_ink(psd, layer, canvas, [], (after, before)) is None


WENDY = "Creatures/DST/3 Stars/Wendy.psd"     # its stored ink outside the engine's layout: a splice falls back


@text_edit
def test_a_text_redrawn_whole_is_measured(clone, parser, chart, tmp_path, monkeypatch):
    """Review 2026-10-24: where the fork's splice falls back (stored ink outside its layout), the engine redraws every
    line: a typo in Wendy's ability moved its kept 2CD badge 6 px and '(3 points)' 8 px, every gate passing (the bar
    sees the stored raster, gate_idempotence the engine's own render). The old text so redrawn is measured against the
    PNG (whole_redraw): a miss of the bar is told, past the floor refused; an edit it splices is not measured."""
    old = pe.normalize_ability(pe.parse_card(parser, _card(clone, WENDY)[0], WENDY)["card"]["ability"])
    jobs = {"ability": old.replace("Resummon", "Resummen", 1)}
    (tmp_path / "a").mkdir()
    _, baked = _bake(clone, parser, WENDY, jobs, tmp_path / "a", chart)
    assert pe._within_bar(baked.parity["ability"]) and baked.edits[0].whole
    assert baked.whole["ability"]["bad_pct"] > 10 and baked.reparse["validator"] == []
    assert any(note.startswith("ability redrawn whole, its lines laid out again: bad ") for note in pe.miss_notes(baked))
    monkeypatch.setattr(pe, "PARITY_FLOOR", 10.0)
    (tmp_path / "b").mkdir()
    with pytest.raises(pe.Refused, match=r"^parity: ability redrawn whole bad [\d.]+% mean [\d.]+ max \d+: past the 10% floor"):
        _bake(clone, parser, WENDY, jobs, tmp_path / "b", chart)
    monkeypatch.undo()
    (tmp_path / "c").mkdir()
    _, baked = _bake(clone, parser, HIVE, {"ability": pe.normalize_ability(pe.parse_card(parser, _card(clone, HIVE)[0], HIVE)[
        "card"]["ability"]) + " Deal 1 damage."}, tmp_path / "c", chart)
    assert not baked.edits[0].whole and baked.whole == {}


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


@text_edit
def test_a_bake_renders_the_card_only_once_an_edit_is_made(clone, parser, chart, monkeypatch, tmp_path):
    """Review 2026-10-24: apply_jobs rendered the whole card twice before trying any edit (gate_copy's pair): a refused
    ability edit cost 4.7 s on Pure Vessel (0.5 s before). A refused edit renders nothing; a stat bake on a card
    without smart filters renders the card before and after, in one RenderCache."""
    real, calls = pe.render_all, []
    monkeypatch.setattr(pe, "render_all", lambda psd: calls.append(psd) or real(psd))
    old = pe.normalize_ability(pe.parse_card(parser, _card(clone, PURE_VESSEL)[0], PURE_VESSEL)["card"]["ability"])
    with monkeypatch.context() as patch:
        patch.setattr(pe, "set_text_layer", lambda *args, **kwargs: (_ for _ in ()).throw(pe.Refused("unsupported-card", "x")))
        with pytest.raises(pe.Refused):
            _bake(clone, parser, PURE_VESSEL, {"ability": old + " Die."}, tmp_path, chart)
    assert calls == []
    hp = pe.parse_card(parser, _card(clone, PURE_VESSEL)[0], PURE_VESSEL)["card"]["hp"]
    parsed, baked = _bake(clone, parser, PURE_VESSEL, {"hp": hp - 1}, tmp_path, chart)
    assert len(calls) == 2 and not pe._filtered(calls[0])


@text_edit
def test_a_bake_cuts_its_region_renders_from_the_whole_ones(clone, parser, chart, monkeypatch, tmp_path):
    """Review 2026-10-24: the baseline rendered each region of the card again (1.3 s on Pure Vessel) where the whole
    render before the edits holds the same pixels: it is cut from it (crop_pair), the gamma and straight fallbacks still
    rendered. The merge's region render and the saved card's re-render of it run in the bake's RenderCache, the second
    resuming the first. The PNG is the one the region renders made."""
    hp = pe.parse_card(parser, _card(clone, MINI_DOOMER)[0], MINI_DOOMER)["card"]["hp"]
    real_region, real_pair, real_cache, depth, calls = pe.render_region, pe.render_pair, pe.render_cache, [0], []

    class Counted:                              # the bake's cache, telling when it is on
        def __init__(self):
            self.cache = real_cache()

        def __enter__(self):
            depth[0] += 1
            return self.cache.__enter__()

        def __exit__(self, *exc):
            depth[0] -= 1
            return self.cache.__exit__(*exc)
    monkeypatch.setattr(pe, "render_cache", Counted)
    monkeypatch.setattr(pe, "render_region", lambda *a, **k: calls.append(("region", depth[0])) or real_region(*a, **k))
    monkeypatch.setattr(pe, "render_pair", lambda *a, **k: calls.append(("pair", depth[0])) or real_pair(*a, **k))
    for out in ("cut", "rendered"):
        (tmp_path / out).mkdir()
    _, got = _bake(clone, parser, MINI_DOOMER, {"hp": hp - 1}, tmp_path / "cut", chart, update_preview=True)
    assert calls == [("pair", 1), ("region", 1)]        # the merge's and the saved card's re-render, in the cache
    calls.clear()
    monkeypatch.setattr(pe, "crop_pair", lambda psd, full, region, icc=True: None)
    _, want = _bake(clone, parser, MINI_DOOMER, {"hp": hp - 1}, tmp_path / "rendered", chart, update_preview=True)
    assert calls == [("region", 0), ("pair", 1), ("region", 1)]
    assert np.array_equal(np.asarray(Image.open(got.png_path)), np.asarray(Image.open(want.png_path)))
    assert got.parity == want.parity


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


def _unclipped(psd: PSDImage) -> PSDImage:
    """Pizza with 'Layer 12', its Multiply layer clipped to the Ability, hidden: a typed badge under it refuses
    (_tinted; review 2026-10-24); these tests are about the runs the edit sets."""
    clipped, = pe._clip_chain(pe.ability_layer(psd))
    assert clipped.name == "Layer 12"
    clipped.visible = False
    return psd


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
        psd = _unclipped(PSDImage.open(pizza))
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
    in the icon font, drawn as its plain letters, while any parenthesis counted as a badge). Under Pizza's own look
    ('Layer 12' clipped to it) a typed badge is told."""
    pizza = _at(clone, tmp_path, COLOURED, PIZZA)[0]
    for typed, check in (("(2U)", lambda runs: runs[2][:2] == ("(1CD)(2U)", pe.ICON_FONT)),
                         ("(2 points)", lambda runs: runs[2][:2] == ("(1CD)", pe.ICON_FONT)
                          and runs[3][0].startswith("(2 points) Can") and runs[3][1] == "Domigorgon")):
        psd = _unclipped(PSDImage.open(pizza))
        pe.set_text_layer(psd, "ability", pe.ability_layer(psd).text.replace("(1CD) Can", f"(1CD){typed} Can", 1),
                          str(clone / pe.FONTS_DIR), chart)
        runs = _runs(pe.ability_layer(psd))
        assert check(runs), (typed, runs)
    psd = PSDImage.open(pizza)
    edit = pe.set_text_layer(psd, "ability", pe.ability_layer(psd).text.replace("(1CD) Can", "(1CD)(2U) Can", 1),
                             str(clone / pe.FONTS_DIR), chart)
    flagged(edit.notes, r"^unsupported-card: ability: \(2U\): the Ability's look \('Layer 12', clipped to it\)")


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


def test_a_deletion_cuts_whole_lines_or_words():
    """Review 2026-10-24: changed_range's prefix is greedy, so deleting a line whose next starts alike ("+2 ATK" before
    "+25% SPD") cut " ATK\r+2" and the kept "+2" kept the deleted line's runs. The deletion slides to whole lines, else
    words or badges, alone (deletion_start) or among an edit's parts (edit_parts)."""
    def cut(old, new):
        start, end = pe.changed_range(old, new)
        start = end = pe.deletion_start(old, new, min(start, end, len(new)))
        return [(old[a0:a1], new[b0:b1], kept) for a0, a1, b0, b1, kept in pe.edit_parts(old, new, start, end)]
    assert cut("A\r+2 ATK\r+25% SPD", "A\r+25% SPD") == [("+2 ATK\r", "", False)]
    assert cut("+2 ATK\r+25% SPD", "+25% SPD") == [("+2 ATK\r", "", False)]
    assert cut("x a.\rb a.", "x a.") == [("\rb a.", "", False)]                 # the last line: its break before it
    assert cut("deal (1CD) (2CD) x", "deal (2CD) x") == [("(1CD) ", "", False)]
    assert cut("Inflict Poison.", "Inflict.") == [(" Poison", "", False)]
    old, new = "(OFS) Gain.\r(WOF) All use. \rW", "(WOF) All use.\rW"           # with another change after it
    parts = pe.edit_parts(old, new, *pe.changed_range(old, new))
    assert [(old[a0:a1], new[b0:b1], kept) for a0, a1, b0, b1, kept in parts] == [
        ("(OFS) Gain.\r", "", False), ("(WOF) All use.", "(WOF) All use.", True), (" ", "", False)]


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
def test_typed_icons_are_unstretched(clone, parser, chart):
    """Review 2026-10-24: a [fire] typed at Tomato's body style (no icon of its own) kept the run's scales (or the layer's
    non-uniform transform: Monster Meat's) and printed 1.72 wide per unit height (a designer's: 0.81). unstretch sets its
    HorizontalScale so x scale x HorizontalScale over y scale x VerticalScale is 1, as designers do."""
    from psd_tools.composite.text.model import TextModel
    psd = PSDImage.open(_card(clone, "Items/Heal/1 Stars/Tomato.psd")[0])    # its body at 1.0 x 0.47
    layer = pe.ability_layer(psd)
    pe.set_text_layer(psd, "ability", pe.normalize_ability(layer.text) + " Gain [fire].", str(clone / pe.FONTS_DIR), chart,
                      parser=parser)
    layer = pe.ability_layer(psd)
    xx, xy, yx, yy = TextModel.from_layer(layer).transform[:4]
    styles = pe._char_styles(layer)[2]

    def aspect(style):
        return np.hypot(xx, xy) * style.get("HorizontalScale", 1.0) / (np.hypot(yx, yy) * style.get("VerticalScale", 1.0))
    assert aspect(styles[0]) > 1.5 and abs(aspect(styles[layer.text.rindex("[fire]")]) - 1) < 0.01


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
def test_tab_stops_refuse_on_an_older_fork(clone, parser, chart, monkeypatch):
    """A psd-tools older than the pin lays each tab as a space: a tabbed text stops before any change, the runner's
    fault (ImportError: retried once the fork is reinstalled)."""
    monkeypatch.setattr(pe, "tab_stops", lambda: False)
    rel = "N.M.E/Michael's_Blessing.psd"
    psd_path = _card(clone, rel)[0]
    psd = PSDImage.open(psd_path)
    old = pe.ability_layer(psd).text
    text = pe.normalize_ability(pe.parse_card(parser, psd_path, rel)["card"]["ability"]).replace("5 Points", "6 Points")
    with pytest.raises(ImportError, match=r"^ability: tab stops"):
        pe.set_text_layer(psd, "ability", text, str(clone / pe.FONTS_DIR), chart, parser=parser)
    assert pe.ability_layer(psd).text == old


@text_edit
def test_deleting_a_line_keeps_the_next_lines_own_style(clone, parser, chart):
    """Review 2026-10-24: Dragon Mount's "+25% SPD" printed "+2" red after its "+2 ATK" line went, Chef Saltbaker's
    kept "(2 points)" took the deleted line's blue, and Spamton NEO's kept "(WOF)" split across two badge sizes (the
    gate refused its letters). Every kept character keeps its own fill and size; the engine's prefix is held to the
    part (splice_text). Salt's kept "As an..." and Nightmare King Grimm's "(OD" took the deleted line's colour without
    deletion_start, Gasoline's next "If" without the word-bound rule (review 2026-10-24)."""
    fonts = str(clone / pe.FONTS_DIR)
    for rel, line in (("Auxiliary/Items/Dragon_Mount.psd", 0), ("Creatures/Cuphead/5 Stars/Chef_Saltbaker.psd", 5),
                      ("Creatures/Undertale/5 Stars/Spamton_Neo.psd", 0), ("Items/Support/2 Stars/Gasoline.psd", 0),
                      ("Auxiliary/Items/Salt.psd", 1), ("Creatures/Hollow Knight/5 Stars/Nightmare_King_Grimm.psd", 1)):
        psd_path = _card(clone, rel)[0]
        psd = PSDImage.open(psd_path)
        layer = pe.ability_layer(psd)
        old = layer.text
        fills, _, styles, _ = pe._char_styles(layer)
        paragraphs = pe.normalize_ability(pe.parse_card(parser, psd_path, rel)["card"]["ability"]).split("\n")
        gone = paragraphs[line]
        a = old.index(gone)
        b = a + len(gone) + 1
        keep = [i for i in range(len(old)) if not a <= i < b and old[i] not in " \r\x03"]
        pe.set_text_layer(psd, "ability", "\n".join(paragraphs[:line] + paragraphs[line + 1:]), fonts, chart,
                          parser=parser)
        after = pe.ability_layer(psd)
        new_fills, _, new_styles, _ = pe._char_styles(after)
        moved = {i: i - (b - a) * (i >= b) for i in keep}
        off = [(i, old[i]) for i in keep if after.text[moved[i]] != old[i]
               or styles[i].get("FontSize") != new_styles[moved[i]].get("FontSize")
               or max(abs(x - y) for x, y in zip(fills[i], new_fills[moved[i]])) > 0.03]    # a status word's swatch
        assert off == [], rel


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
    from psd_tools.composite.text import FontMap
    fonts = str(clone / pe.FONTS_DIR)
    psd = PSDImage.open(_plain(clone, tmp_path)[0])
    layer = pe.ability_layer(psd)
    layer.set_text("Inflict Poison. Die.", keep="runs")
    layer.set_span_style(8, 14, fill=(0.5, 0.5, 0.5))
    layer.rasterize(fonts=FontMap(dirs=[fonts]))     # its pixels the text's (drawn_in_font)
    pe.set_text_layer(psd, "ability", "Inflict Poison. Live.", fonts, chart)
    green = tuple(round(v, 3) for v in chart["poison"].fill)
    fills = _fills(pe.ability_layer(psd))
    assert fills[1] == ("Poison", green) and all(fill == (1.0, 1.0, 1.0) for _, fill in fills[2:])
    pe.set_text_layer(psd, "ability", "Inflict Blazing, Live.", fonts, chart)
    assert _fills(pe.ability_layer(psd))[1][0] == "Blazing" and [fx.name for fx in pe.fx_layers(psd, pe.ability_layer(psd))] == ["Ability FX: Blazing"]


@text_edit
def test_overlay_word_layer_without_a_chart_style_is_hidden(clone, chart):
    """Fire Staff's hand-made "Smoldering" overlay (before L7) carries a Stroke the chart knows nothing about: it is
    hidden, told, the Ability's own word showing (owner 2026-10-24); without the chart nothing is known: refused."""
    fonts = str(clone / pe.FONTS_DIR)
    psd = PSDImage.open(BytesIO(at_rev(clone, OVERLAY_STAFF, FIRE_STAFF)))
    text = pe.ability_layer(psd).text
    with pytest.raises(pe.Refused, match="status word layers over the ability"):
        pe.set_text_layer(psd, "ability", text + " Die.", fonts, None)
    edit = pe.set_text_layer(psd, "ability", text + " Die.", fonts, chart)
    flagged(edit.notes, "^styled-word: ability: overlay word layers the chart has no style for, hidden: Smoldering$")
    assert edit.hidden and not pe.overlay_words(psd)


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
def test_a_phrase_in_several_colours_takes_the_first(clone, parser, chart):
    """Which colour a copy joining an own-style layer takes can't be told when its words are in more than one (none on
    main: Palace of Shadow's "+2" painted red here), and an edit inside such a phrase paints it one colour (plan.fill):
    the first drawn run's, told."""
    for edit, why in ((lambda t: t + " Allies get +2 DEF.", r"the typed '\+2 DEF' takes the first$"),
                      (lambda t: t.replace("+2 DEF", "+2  DEF"), r"which the edit in '\+2  DEF' makes one$")):
        psd = PSDImage.open(_card(clone, PALACE)[0])
        layer = pe.ability_layer(psd)
        start = layer.text.index("+2 DEF")
        layer.set_span_style(start, start + 2, fill=(1.0, 0.0, 0.0))
        notes = pe.set_text_layer(psd, "ability", edit(layer.text), str(clone / pe.FONTS_DIR), chart, parser=parser).notes
        flagged(notes, r"^styled-word: ability: Ability FX: \+2 DEF: its words are in more than one colour.*" + why)


@text_edit
def test_a_moved_own_style_phrase_drops_its_layer(clone, parser, chart):
    """An own-style layer stays on the characters it covered (ruling B). Palace of Shadow's "+2 DEF" deleted from its
    line and typed at the end kept none of them, yet the layer jumped to the typed copy, printed grey, and the note
    said it "also" covered it. The layer goes, told (owner 2026-10-24)."""
    psd = PSDImage.open(_card(clone, PALACE)[0])
    before = pe.ability_layer(psd).text
    text = before.replace("+2 DEF\x03", "\x03").replace("its place.", "its place. +2 DEF")
    assert text != before
    edit = pe.set_text_layer(psd, "ability", text, str(clone / pe.FONTS_DIR), chart, parser=parser)
    flagged(edit.notes, r"^Ability FX: \+2 DEF deleted: the edit deletes its '\+2 DEF' and types it again, which would "
                        r"move the layer$")


@text_edit
def test_letters_glued_onto_an_own_style_phrase_drop_its_layer(clone, parser, chart):
    """"1T Despair." made "1T Despairing." kept The Watcher's own-style 'Ability FX: Despair' on "Despair" and printed
    "ing" plain, untold: half a word in the layer's style. The layer goes, told (owner 2026-10-24); a word typed after
    it still keeps every layer."""
    psd = PSDImage.open(_card(clone, WATCHER)[0])
    layer = pe.ability_layer(psd)
    before = layer.text
    edit = pe.set_text_layer(psd, "ability", before.replace("Despair.", "Despairing."), str(clone / pe.FONTS_DIR), chart,
                             parser=parser)
    flagged(edit.notes, r"^Ability FX: Despair deleted: the edit makes its 'Despair' part of 'Despairing', half a word in "
                        r"its style$")
    psd = PSDImage.open(_card(clone, WATCHER)[0])
    edit = pe.set_text_layer(psd, "ability", before.replace("Despair.", "Despair for free."), str(clone / pe.FONTS_DIR),
                             chart, parser=parser)
    assert edit.fx_names == [fx.name for fx in pe.fx_layers(PSDImage.open(_card(clone, WATCHER)[0]), pe.ability_layer(psd))]


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
def test_too_long_ability_is_flagged(clone, chart, tmp_path):
    psd = PSDImage.open(_plain(clone, tmp_path)[0])
    text = pe.normalize_ability(pe.ability_layer(psd).text).replace("(BOT) ", "")
    flagged(pe.set_text_layer(psd, "ability", " ".join([text] * 4), str(clone / pe.FONTS_DIR), chart).notes, "^overflow: ")


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

    def stretched(psd, which, text, fonts_dir, chart=None, card="", own_text=True, parser=None, original=None):
        edit = real(psd, which, text, fonts_dir, chart, card, own_text, parser, original)
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


RED_KNIFE = "Auxiliary/Items/Red_Knife.psd"     # its Ability near the card's foot: a halo's faint tail passes it
SALT = "Auxiliary/Items/Salt.psd"               # its title in a stroked Text group, 'Layer 19' (MISS) below beside it


def _token_ink(layer, fonts, piece: str) -> tuple:
    """The ink box (canvas px, whole) of the icon of the layer's icon-font cluster drawing piece ('[fire:glow]': its
    smallest glyph, not the halo)."""
    from psd_tools.composite.text import FontMap
    from psd_tools.composite.text.geometry import text_geometry
    geometry = text_geometry(layer, FontMap(dirs=[fonts]))
    boxes = [glyph["ink"] for glyph in geometry["glyphs"] if glyph["ink"] and pe.ICON_FONT_PREFIX in glyph["font"].lower()
             and geometry["text"][glyph["i"]:glyph["end"]] == piece]
    x0, y0, x1, y1 = min(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
    return int(x0), int(y0), int(x1) + 1, int(y1) + 1


@text_edit
@pytest.mark.skipif(not colour_glyphs(), reason="the fork draws SVG colour glyphs in one colour")
def test_glow_tokens_bake_as_photoshop_draws_them(clone, parser, chart, tmp_path):
    """Fork cards-20261024b draws a :glow/:darkglow halo as Photoshop does (it was a solid block, refused): typed ones
    bake through every gate. The_Hive's Text group strokes it 7 px black, so the halo is a black disc around the icon,
    Photoshop's own look (the owner's finding, 2026-10-24), not the golden card behind it."""
    fonts = str(clone / pe.FONTS_DIR)
    old = pe.parse_card(parser, _card(clone, HIVE)[0], HIVE)["card"]["ability"]
    for token in ("[fire:glow]", "[fire:darkglow]"):
        (tmp_path / token[6:-1]).mkdir()
        _, baked = _bake(clone, parser, HIVE, {"ability": old + f" Deal 1 {token} damage."}, tmp_path / token[6:-1], chart)
        assert pe._within_bar(baked.parity["ability"]) and baked.reparse["validator"] == [] and baked.fx == []
        layer = pe.ability_layer(PSDImage.open(baked.psd_path))
        assert token in layer.text and "Deal 1 [fire] damage." in baked.reparse["card"]["ability"]   # read as [fire]
        x0, y0, x1, y1 = _token_ink(layer, fonts, token)
        before = np.asarray(Image.open(_card(clone, HIVE)[1]).convert("RGB")).astype(int)
        after = np.asarray(Image.open(baked.png_path).convert("RGB")).astype(int)
        ring = np.zeros(after.shape[:2], bool)       # above and below the icon, 8-18 px out: the disc
        ring[y0 - 18:y0 - 8, x0:x1] = ring[y1 + 8:y1 + 18, x0:x1] = True
        dark = lambda image: float((image[ring].max(axis=1) < 60).mean())
        assert dark(after) > 0.95 and dark(before) < 0.5, (token, dark(after), dark(before))


@text_edit
def test_glow_tokens_refuse_on_an_older_fork(clone, chart, monkeypatch):
    """A psd-tools older than the pin (a stale install: pip keeps one of the same version) draws a glow halo as a solid
    block: a typed glow stops before any change, the runner's fault (ImportError: no verdict, retried once the fork is
    reinstalled; a `font` verdict outlived it, review 2026-10-24); other tokens do not."""
    monkeypatch.setattr(pe, "glow_halos", lambda: False)
    psd = PSDImage.open(_card(clone, HIVE)[0])
    layer = pe.ability_layer(psd)
    old = layer.text
    for token in ("[fire:glow]", "[Fire:DarkGlow]"):
        with pytest.raises(ImportError, match="older than the pin") as refused:
            pe.set_text_layer(psd, "ability", pe.normalize_ability(old) + f" Deal 1 {token} damage.", str(clone / pe.FONTS_DIR), chart)
        assert token in str(refused.value) and layer.text == old
    pe.set_text_layer(psd, "ability", pe.normalize_ability(old) + " Deal 1 [fire] damage.", str(clone / pe.FONTS_DIR), chart)


@text_edit
def test_a_faint_halo_past_the_canvas_is_no_overflow(clone, parser, chart):
    """Only ink past INK_ALPHA off the canvas refuses: Red_Knife's [ice:glow] halo passes the card's foot by 10 px at
    alpha 0.03 at most (Photoshop clips it the same); its solid ink past a smaller canvas counts."""
    fonts = str(clone / pe.FONTS_DIR)
    psd = PSDImage.open(_card(clone, RED_KNIFE)[0])
    layer = pe.ability_layer(psd)
    old = pe.normalize_ability(layer.text)
    pe.set_text_layer(psd, "ability", old + " Gain [ice:glow].", fonts, chart, parser=parser)
    assert layer.bbox[3] > psd.height and not pe._ink_off_canvas(layer, psd.width, psd.height)
    assert pe._ink_off_canvas(layer, psd.width, layer.bbox[1] + 40) and pe._ink_off_canvas(layer, 300, psd.height)


@text_edit
def test_a_glow_disc_over_a_layer_below_the_title_is_flagged(clone, parser, chart):
    """Salt's Text group strokes a title [fire:glow]'s faint halo into a black disc (Photoshop's look) that covers
    'Layer 19' (MISS) beside the title: inks_over counts the group's changed ink for glow tokens (the title's own ink
    past INK_ALPHA misses it), told; the same token with no glow stays clear of it."""
    fonts = str(clone / pe.FONTS_DIR)
    for title, told in (("Salt [fire:glow]", True), ("Salt [fire]", False)):
        notes = pe.set_text_layer(PSDImage.open(_card(clone, SALT)[0]), "name", title, fonts, chart, "Salt", parser=parser).notes
        assert any("print over 'Layer 19' below it" in note for note in notes) == told, notes


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


@text_engine
def test_text_edit_needs_the_story_sync(clone, chart, monkeypatch):
    """A fork whose set_text writes TySh alone never edits a card: ImportError, which the bake reports as transient."""
    monkeypatch.setattr(pe, "story_sync", lambda: False)
    psd = PSDImage.open(_card(clone, MINI_DOOMER)[0])
    with pytest.raises(ImportError, match="Txt2 story stale"):
        pe.set_text_layer(psd, "name", "Mini Doomer II", str(clone / pe.FONTS_DIR), chart)
    assert pe.read_name(psd) == "Mini Doomer"


# ========================
# An ability over several layers (owner 2026-10-24)
# ========================
LIT_BOB = "Auxiliary/Items/Lit_Bob-omb.psd"                        # 'Costs 0 SP to use.' / 'Deal 7 DMG.': two Ability layers
MIRACLE = "Creatures/Kirby/3 Stars/Miracle_Matter.psd"            # five, padded ('\r    Fire: 5/5/8/7'), 'Ability FX: (1GCD)'
BULBORB = "Auxiliary/Minions/Bulborb Larva.psd"                    # a hidden Ability laid out again in 'text 1'..'text 3'


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
    """Miracle Matter: an edit across its Fire and Earth layers changes those two records alone, its other three and
    their 'Ability FX: (1GCD)' byte for byte; on Lit Bob-omb a removed break joins its two layers in the first, the
    second hidden, told."""
    old = pe.parse_card(parser, _card(clone, MIRACLE)[0], MIRACLE)["card"]["ability"]
    new = old.replace("Fire: 5/5/8/7", "Fire: 5/5/9/7").replace("Earth: 5/8/6/5", "Earth: 5/9/6/5")
    _, baked = _bake(clone, parser, MIRACLE, {"ability": new}, _dir(tmp_path, "mm"), chart)
    assert len(baked.records) == 2 and all(set(fields) <= set(TEXT_RECORD) for _, _, fields in baked.records)
    assert baked.reparse["card"]["ability"] == new and baked.fx_notes == []
    saved = [layer.text for layer in pe.ability_layers(PSDImage.open(baked.psd_path)).layers]
    assert saved[1:3] == ["\r    Fire: 5/5/9/7", "\r    \r    Earth: 5/9/6/5 and heal 2 HP\r   "]
    _, baked = _bake(clone, parser, LIT_BOB, {"ability": "Costs 0 SP to use. Deal 7 DMG."}, _dir(tmp_path, "bob"), chart)
    flagged(baked.fx_notes, r"^ability: the break between the first and second ability layers was removed: their "
                            r"words are joined in the first and the second hidden")
    assert sorted(fields == ["flags"] or set(fields) <= set(TEXT_RECORD) for _, _, fields in baked.records) == [True, True]
    assert ["flags"] in [fields for _, _, fields in baked.records]
    assert baked.reparse["card"]["ability"] == "Costs 0 SP to use. Deal 7 DMG."


@text_edit
def test_several_layers_hidden_part_and_blank_lines(clone, parser, chart, tmp_path):
    """Miracle Matter (review): its first part removed, its FX layer is hidden with it, not given to the last part and
    deleted; a blank line removed between parts is no change (classify), and beside a real change bakes it, told."""
    old = pe.parse_card(parser, _card(clone, MIRACLE)[0], MIRACLE)["card"]["ability"]
    new = old.replace("(1GCD) You may choose one of the following forms:\n", "").replace("whatever", "whichever")
    _, baked = _bake(clone, parser, MIRACLE, {"ability": new}, _dir(tmp_path, "drop"), chart)
    assert baked.reparse["card"]["ability"] == new
    assert baked.fx_notes == ["ability: the words of the first ability layer were removed, so it is hidden"]
    saved = PSDImage.open(baked.psd_path)
    assert [(fx.name, fx.is_visible()) for fx in saved.descendants() if " FX:" in fx.name] == [("Ability FX: (1GCD)", False)]
    tidy = old.replace("5/5/8/7\n    \nEarth", "5/5/8/7\nEarth")
    psd, parsed = PSDImage.open(_card(clone, MIRACLE)[0]), pe.parse_card(parser, _card(clone, MIRACLE)[0], MIRACLE)
    assert pe.classify(parsed, {"ability": tidy}, psd, parser).redundant == ["ability"]
    _, baked = _bake(clone, parser, MIRACLE, {"ability": tidy.replace("heal 2 HP", "heal 3 HP")}, _dir(tmp_path, "tidy"),
                     chart)
    assert baked.reparse["card"]["ability"] == old.replace("heal 2 HP", "heal 3 HP")
    assert baked.fx_notes == ["ability: the blank lines between the ability layers stay as they were: they place the "
                              "layers on the card"]


@text_edit
def test_relaid_and_copied_abilities_bake(clone, parser, chart, tmp_path):
    """Bulborb Larva: the piece holding the edit and the hidden Ability the parser reads take it, the other pieces
    stay; an added line is told. SOMETHING: its two offset copies print the edited words too, its new FX layer over
    them by design (no overflow told)."""
    old = pe.parse_card(parser, _card(clone, BULBORB)[0], BULBORB)["card"]["ability"]
    new = old.replace("next minion.", "next creature.\nCannot be targeted.")
    _, baked = _bake(clone, parser, BULBORB, {"ability": new}, _dir(tmp_path, "larva"), chart)
    assert sorted(name for _, name, _ in baked.records) == ["Ability", "text 3"]
    flagged(baked.fx_notes, r"^ability: a line was added to the third ability layer")
    assert baked.reparse["card"]["ability"] == new
    old = pe.parse_card(parser, _card(clone, SOMETHING)[0], SOMETHING)["card"]["ability"]
    new = old.replace("1T Afraid.", "1T Afraid and Blazing.")
    _, baked = _bake(clone, parser, SOMETHING, {"ability": new}, _dir(tmp_path, "something"), chart)
    assert baked.fx_notes == []
    saved = PSDImage.open(baked.psd_path)
    found = pe.ability_layers(saved)
    assert len(found.copies[0]) == 2 and {layer.text for layer in [*found.layers, *found.copies[0]]} == {
        new.replace("\n", "\r").replace("action.\r", "action. \r")}
