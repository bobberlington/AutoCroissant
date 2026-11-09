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
    assert blob_sha(clone / rel) == expected["blob"], "goldens not made at CARDS_REV: python tests/make_goldens.py"
    got = parse(psd_parser, clone, rel)
    assert got["card"] == expected["card"]
    assert got["alt_names"] == expected["alt_names"]
    assert got["validator"] == expected["validator"]


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
    assert {"Creatures", "Items", "Field", "N.M.E", "Auxiliary/Minions", "Auxiliary/Status Effects"} <= tops


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
    from commands.psd_analyzer import CardInfo, StatsDatabase
    from commands.psd_parser import PSDParser
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
    from commands.psd_analyzer import StatsDatabase
    from commands.psd_parser import BoundingBox, PSDParser
    p = PSDParser(["fire"], StatsDatabase())
    texts = [("Vek'nilash:\n", BoundingBox(18, 1951)), ("            Immune.\n(2CD) Hit.\n", BoundingBox(250, 2000)),
             ("Turtle        Cat\n", BoundingBox(20, 2100))]
    assert p._process_abilities(texts, [], 1200) == "Vek'nilash:\nImmune.\n(2CD) Hit.\nTurtle        Cat"
    assert p._process_abilities([("    Burn.\n    Then.\n", BoundingBox(20, 2000))], [("fire", BoundingBox(30, 2010))], 1200) \
        == "[fire] Burn.\nThen."


def test_one_line_label_beside_a_part():
    """A one-line part with another in its row labels that one's first line (Twin Emperors' "Vek'nilash:" at y 1951,
    its body at 1958, 11-09); a label a row above stays on its own line."""
    from commands.psd_analyzer import StatsDatabase
    from commands.psd_parser import BoundingBox, PSDParser
    p = PSDParser([], StatsDatabase())
    texts = [("'Vek'nilash:\n'", BoundingBox(18, 1951)), ("'            Immune.\n(2CD) Hit.\n'", BoundingBox(231, 1958)),
             ("'Vek'lor:\n'", BoundingBox(159, 2157)), ("'Attacks inflict 2T1 Enfeebled.\n'", BoundingBox(364, 2165))]
    assert p._process_abilities(texts, [], 1200) == "Vek'nilash: Immune.\n(2CD) Hit.\nVek'lor: Attacks inflict 2T1 Enfeebled."
    texts = [("'Vek'lor:'", BoundingBox(159, 2100)), ("'Hit.'", BoundingBox(364, 2165))]
    assert p._process_abilities(texts, [], 1200) == "Vek'lor:\nHit."


def test_leftover_icon_only_in_ability_span():
    """A type icon with no gap left is appended only within a line of an Ability layer's box: Tainted Lazarus's
    creature 2 header icon (y 2242, its Ability boxes end at 1382) is no ability's (11-09)."""
    from commands.psd_analyzer import StatsDatabase
    from commands.psd_parser import BoundingBox, PSDParser
    p = PSDParser(["undead"], StatsDatabase())
    texts = [("Attacks inflict 1T Afraid.\n", BoundingBox(12, 1255, 1382))]
    assert p._process_abilities(texts, [("undead", BoundingBox(1006, 2242))], 1200) == "Attacks inflict 1T Afraid."
    assert p._process_abilities(texts, [("undead", BoundingBox(900, 1300))], 1200) == "Attacks inflict 1T Afraid. [undead]"


def test_lowercase_part_continues_the_line():
    """Parts laying one text out line by line (Apex Predator, 11-09): a lowercase start continues the line before it,
    each part without its padding. With none, the old join (Miracle Matter's lines of blanks go: padding)."""
    from commands.psd_analyzer import StatsDatabase
    from commands.psd_parser import BoundingBox, PSDParser
    p = PSDParser(["fire"], StatsDatabase())
    texts = [("'Cannot attack.\n'", BoundingBox(357, 1780)), ("'\n\n \n \n\nWhen next to it, this counts as'", BoundingBox(52, 2248)),
             ("'\n\n \n \n\n\nbeing in its slot.'", BoundingBox(72, 2326))]
    assert p._process_abilities(texts, [], 1200) == "Cannot attack.\nWhen next to it, this counts as being in its slot."
    texts = [("'(1GCD) Assume a form:'", BoundingBox(19, 1762)), ("'\n    Fire: 5/5/8/7'", BoundingBox(121, 1901)),
             ("'\n    \n    Earth: 5/8/6/5\n   '", BoundingBox(124, 1969))]
    assert p._process_abilities(texts, [], 1200) == "(1GCD) Assume a form:\nFire: 5/5/8/7\nEarth: 5/8/6/5"


def test_text_with_both_quotes_parses_without_backslash():
    """str() of an engine string is its repr, which escapes ' as \\' when the text also holds a ": the parse unescapes it
    (the old rule page 03 Setting Up's "they're")."""
    from psd_tools.psd.engine_data import String
    from commands.psd_analyzer import CardInfo, StatsDatabase
    from commands.psd_parser import MutableValue, PSDParser
    layer = types.SimpleNamespace(name="Ability", engine_dict={"Editor": {"Text": String('They\'re "in" now\r')}}, resource_dict={},
                                  bbox=(0, 2000, 100, 2100), is_visible=lambda: True)
    found = []
    PSDParser([], StatsDatabase())._process_text_layer(layer, CardInfo(name="Card", card_type="item"), MutableValue(0), found,
                                                       False, 1200)
    assert found[0][0].strip("'\"\n") == 'They\'re "in" now' and "\\" not in found[0][0]


def test_header_font_icons_give_an_aux_items_type():
    """Icons typed in the stoogedings font in the header are header icons too (_process_text_layer's header): of two
    item-type icons in one text layer the first typed is the item type."""
    from psd_tools.psd.engine_data import String
    from commands.psd_analyzer import CardInfo, StatsDatabase
    from commands.psd_parser import MutableValue, PSDParser
    p = PSDParser([], StatsDatabase())
    p._icon_font_spans, p._flag_misplaced_tokens = (lambda layer, text: [(0, 1)]), (lambda *a: None)
    p._read_icon_tokens = lambda text, spans, card: ("[deploy][fire][food]", [("deploy", "type"), ("fire", "type"), ("food", "type")])
    layer = types.SimpleNamespace(name="Types", engine_dict={"Editor": {"Text": String("x")}}, resource_dict={},
                                  bbox=(10, 20, 300, 120), is_visible=lambda: True)
    card, header = CardInfo(name="Card", card_type="aux item"), []
    p._process_text_layer(layer, card, MutableValue(0), [], False, 1200, header=header)
    assert card.types == [n for n, _ in header] == ["deploy", "fire", "food"] and PSDParser._aux_item_type(header) == "deploy"
