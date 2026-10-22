"""Parser regression gate: fresh parses of golden PSDs must equal tests/goldens/ exactly (ability text byte for byte)."""
import json
import types

import pytest

from conftest import blob_sha
from golden_cards import CARDS, CERTIFIED, golden_file, parse, parser


@pytest.fixture(scope="module")
def psd_parser(clone):
    return parser()


@pytest.mark.parametrize("rel", CARDS)
def test_golden(psd_parser, clone, rel):
    expected = json.loads(golden_file(rel).read_text(encoding="utf-8"))
    if not (clone / rel).exists() or blob_sha(clone / rel) != expected["blob"]:
        pytest.skip(f"{rel} changed upstream since the golden was made (re-pin: python tests/make_goldens.py)")
    got = parse(psd_parser, clone, rel)
    assert got["card"] == expected["card"]
    assert got["alt_names"] == expected["alt_names"]
    assert got["validator"] == expected["validator"]


def test_certified_goldens_are_current(clone):
    """The certified cards are always parsed, never skipped: one edited upstream must be re-certified."""
    stale = [rel for rel in CERTIFIED if not (clone / rel).exists()
             or blob_sha(clone / rel) != json.loads(golden_file(rel).read_text(encoding="utf-8"))["blob"]]
    assert stale == [], "re-pin with python tests/make_goldens.py, then check test_certified_values"


def test_certified_values():
    """The certified goldens (validation-and-qa section 3) as recorded there."""
    doomer, freezer, lich = (json.loads(golden_file(rel).read_text(encoding="utf-8")) for rel in CERTIFIED)
    d = doomer["card"]
    assert (d["type"], d["stars"], d["types"]) == ("minion", 2, ["wind", "defensive_minion", "light"])
    assert (d["hp"], d["def"], d["atk"], d["spd"]) == (5, 2, 5, 5)
    assert d["ability"].endswith("(BOT) Die.") and doomer["validator"] == []
    f = freezer["card"]
    assert (f["type"], f["stars"], f["types"]) == ("field", 1, ["field"])
    assert f["ability"] == ("For friendly creatures:\n[water] Turns into [ice]. [ice]: +1 DEF\n"
                            "(1PGT) Choose a friendly creature to turn into a [ice] and lower its speed by 1.")
    assert lich["card"]["types"] == ["ice", "undead"]
    assert lich["card"]["ability"].splitlines()[-1] == "Minions inherit [undead]."


def test_goldens_cover_every_folder():
    tops = {rel.split("/")[0] for rel in CARDS} | {"/".join(rel.split("/")[:2]) for rel in CARDS if rel.startswith("Auxiliary")}
    assert {"Creatures", "Items", "Field", "N.M.E", "Auxiliary/Minions", "Auxiliary/Items", "Auxiliary/Creatures",
            "Auxiliary/Status Effects", "Auxiliary/Rulebook"} <= tops


# ========================
# Effect layers and exclusions
# ========================
class _Layer:
    """A text or effect layer stand-in: name, bbox, visibility, text; no pixels (coverage skipped)."""
    def __init__(self, name, text=None, visible=True, bbox=(0, 0, 100, 100)):
        self.name, self.bbox, self._visible = name, bbox, visible
        self.engine_dict = {"Editor": {"Text": types.SimpleNamespace(value=text)}} if text is not None else None

    def is_visible(self):
        return self._visible

    def numpy(self, channel=None):
        raise ValueError("no pixels")


def _fx_problems(fx_name, text, visible=True):
    from commands.psd_analyzer import CardInfo, PSDParser, StatsDatabase
    card = CardInfo(name="Card", card_type="item")
    PSDParser([], StatsDatabase())._check_effect_layer(_Layer(fx_name, visible=visible), [_Layer("Ability", text)], card)
    return card.problems


def test_orphaned_effect_layer():
    """Idea 14: an FX layer whose words left the ability is WORDS NOT IN TEXT, a #n past the last match too."""
    text = "(BOT) Die. Inflict Blazing.\rBlazing\x03twice. Frozen In Time"
    assert _fx_problems("Ability FX: Blazing", text) == []
    assert _fx_problems("Ability FX: blazing#2 | frozen  in time", text) == []
    assert _fx_problems("Ability FX: Blazing#3", text) == ["EFFECT LAYER WORDS NOT IN TEXT: Ability FX: Blazing#3"]
    assert _fx_problems("Ability FX: Gone | Blazing", text) == ["EFFECT LAYER WORDS NOT IN TEXT: Ability FX: Gone | Blazing"]
    assert _fx_problems("Ability FX: Blazing #2", text) == ["EFFECT LAYER WORDS NOT IN TEXT: Ability FX: Blazing #2"]  # literal, as in the JSX
    assert _fx_problems("Ability FX: Gone", text, visible=False) == []               # hidden: prints nothing
    assert _fx_problems("Text FX: Blazing", text) == ["EFFECT LAYER WITHOUT TEXT LAYER: Text FX: Blazing"]


def test_occurrences_like_effect_layers_jsx():
    """Whole words when there are any ("Burn" in "Burn, Burning": 1), else every match ("1" in "2T1 x21": 2)."""
    from commands.psd_analyzer import PSDParser
    assert PSDParser._occurrences("burn, burning burn", "burn") == 2
    assert PSDParser._occurrences("2t1 x21", "1") == 2
    assert PSDParser._occurrences("frozen in time", "frozen") == 1
    assert PSDParser._occurrences("anything", "") == 0


def test_shadow_duelist_exclusion(clone):
    """D3 is not stale: the card prints no ability; the parsed text comes from its hidden Ability layer."""
    from psd_tools import PSDImage
    from commands.psd_analyzer import CardValidator
    rel = "Auxiliary/Minions/Shadow_Duelist.psd"
    if not (clone / rel).exists() or blob_sha(clone / rel) != "6c777d46e1c106a1548ba15facb75f218f9327ab":
        pytest.skip(f"{rel} changed upstream: re-check its ABILITY_EXCLUSIONS entry")
    ability = [layer for layer in PSDImage.open(clone / rel).descendants() if layer.name == "Ability"]
    assert [layer.is_visible() for layer in ability] == [False] and ability[0].text.startswith("Heals the creature")
    assert "Shadow Duelist" in CardValidator.ABILITY_EXCLUSIONS
