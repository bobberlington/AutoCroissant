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
    assert d["ability"].endswith("(BOT) Destroy this creature.") and doomer["validator"] == []
    f = freezer["card"]
    assert (f["type"], f["stars"], f["types"]) == ("field", 1, ["field"])
    assert f["ability"] == ("For friendly creatures:\n[water] Turns into [ice]. [ice]: +1 DEF\n"
                            "(1GCD) Choose a friendly creature to turn into a [ice] and lower its speed by 1.")
    assert lich["card"]["types"] == ["ice", "undead"]
    assert lich["card"]["ability"].splitlines()[-1] == "Minions inherit [undead]."


def test_goldens_cover_every_folder():
    tops = {rel.split("/")[0] for rel in CARDS} | {"/".join(rel.split("/")[:2]) for rel in CARDS if rel.startswith("Auxiliary")}
    assert {"Creatures", "Items", "Field", "N.M.E", "Auxiliary/Minions", "Auxiliary/Status Effects", "Auxiliary/Rulebook"} <= tops


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


def test_unfilled_line_gaps_are_padding():
    """A line's leading gap that no type fills is layout padding (Twin Emperors' body beside its label, Warcraftian
    Druid's "Bear:" line): gone. A gap a type fills, and one inside a line, stay as before."""
    from commands.psd_analyzer import BoundingBox, PSDParser, StatsDatabase
    p = PSDParser(["fire"], StatsDatabase())
    texts = [("Vek'nilash:\n", BoundingBox(18, 1951)), ("            Immune.\n(2CD) Hit.\n", BoundingBox(250, 2000)),
             ("Turtle        Cat\n", BoundingBox(20, 2100))]
    assert p._process_abilities(texts, [], 1200) == "Vek'nilash:\nImmune.\n(2CD) Hit.\nTurtle        Cat"
    assert p._process_abilities([("    Burn.\n    Then.\n", BoundingBox(20, 2000))], [("fire", BoundingBox(30, 2010))], 1200) \
        == "[fire] Burn.\nThen."
