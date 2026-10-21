"""CardEdits (MDW/Card Edits.json): parsing, validation and apply/restore."""
import json
from copy import deepcopy

import pytest

from commands.psd_analyzer import CardEdits, CardInfo, CardStats, CardValidator


def _edits(cards: dict) -> CardEdits:
    return CardEdits.parse(json.dumps({"cards": cards}))


def _card(**kw) -> CardInfo:
    base = dict(name="Foo Bar", card_type="creature", ability="", types=["fire"],
                stats=CardStats(hp=-1, defense=2, attack=3, speed=4), problems=["MISSPELT TYPE: tornado"])
    base.update(kw)
    return CardInfo(**base)


# ========================
# Parsing and validation
# ========================
def test_parse_keeps_valid_fields_only():
    e = _edits({"Foo_Bar": {"stars": 2, "_by": "web", "_at": 1, "bogus": 1, "hp": -1, "types": ["ice"], "ability": 5},
                "Not A Dict": "x", "  Spaced   Name ": {"note": "hi"}})
    assert e.cards == {"foo bar": {"stars": 2, "types": ["ice"]}, "spaced name": {"note": "hi"}}


@pytest.mark.parametrize("text", ["[]", "null", '{"cards": []}', '{"other": 1}'])
def test_parse_odd_shapes_give_no_edits(text):
    assert CardEdits.parse(text).cards == {}


def test_parse_rejects_non_json():
    with pytest.raises(ValueError):
        CardEdits.parse("{not json")


@pytest.mark.parametrize("name,value,ok", [
    ("hp", 3, True), ("hp", 0, True), ("hp", -1, False), ("hp", True, False), ("hp", 2.0, False), ("hp", None, True),
    ("stars", None, True), ("ability", None, False), ("name", None, False), ("ability", "x", True), ("ability", 1, False),
    ("types", ["a", "b"], True), ("types", ["a", 1], False), ("types", "a", False), ("hidden", True, True),
    ("hidden", 1, False), ("aliases", ["x"], True), ("unknown", 1, False),
    ("creatures", [{"hp": 7, "def": None}], True), ("creatures", [{"hp": -1}], False), ("creatures", [{"hp": True}], False),
    ("creatures", [{"mana": 1}], False), ("creatures", [1], False), ("creatures", {}, False),
])
def test_valid(name, value, ok):
    assert CardEdits.valid(name, value) is ok


def test_names():
    e = _edits({"Foo": {"aliases": ["f", "fb"], "name": "Foo Prime"}, "Bar": {"note": "x"}, "Baz": {"name": "Bazz"}})
    assert e.names() == {"foo": ["f", "fb", "Foo Prime"], "baz": ["Bazz"]}


# ========================
# Apply and restore
# ========================
def test_apply_then_restore():
    card = _card()
    original = repr(card)
    e = _edits({"FOO_BAR": {"hp": 5, "ability": "Fixed.", "name": "Foo Prime", "note": "checked"}})
    assert e.apply("Foo Bar", card) is True
    assert (card.stats.hp, card.ability, card.display_name, card.note) == (5, "Fixed.", "Foo Prime", "checked")
    assert card.parsed_values == {"hp": None, "ability": "", "name": None, "note": None,
                                  "problems": ["MISSPELT TYPE: tornado"]}
    assert card.problems == ["MISSPELT TYPE: tornado"]       # not answered by hp/ability
    assert e.apply("Foo Bar", card) is False                 # idempotent
    assert CardEdits().apply("Foo Bar", card) is True        # edit gone: PSD values back
    assert repr(card) == original


def test_apply_clears_answered_problems():
    card = _card(problems=["MISSPELT TYPE: tornado", "HP NOT FOUND", "NO ABILITY LAYER"])
    _edits({"Foo Bar": {"types": ["wind"]}}).apply("Foo Bar", card)
    assert card.types == ["wind"] and card.problems == ["HP NOT FOUND", "NO ABILITY LAYER"]
    _edits({"Foo Bar": {"types": ["wind"], "ability": "A."}}).apply("Foo Bar", card)
    assert card.types == ["wind"] and card.problems == ["HP NOT FOUND"]
    assert card.parsed_values["types"] == ["fire"]           # still the PSD's, not the first edit's


def test_problems_ok_silences_validator():
    card = _card(ability="")
    assert CardValidator.validate(deepcopy(card))
    _edits({"Foo Bar": {"problems_ok": True}}).apply("Foo Bar", card)
    assert card.problems == [] and CardValidator.validate(card) == []


def test_null_stat_edit_unsets_it():
    card = _card(stats=CardStats(hp=6, defense=2, attack=3, speed=4))
    _edits({"Foo Bar": {"hp": None}}).apply("Foo Bar", card)
    assert card.stats.hp == -1 and "hp" not in card.to_dict()


def test_creatures_edit_on_real_card(real_stats):
    """Tainted Lazarus (2 creatures): edit creature 2, then drop the edit."""
    card = deepcopy(real_stats.stats["Tainted Lazarus"])
    if getattr(card, "parsed_values", None):
        pytest.skip("Tainted Lazarus already has website edits")
    original = repr(card)
    creatures = CardEdits().get(card, "creatures")
    assert len(creatures) == 2 and creatures[0] == card.stats.to_dict()
    creatures[1]["def"] = 9
    e = _edits({"Tainted Lazarus": {"creatures": creatures}})
    assert e.apply("Tainted Lazarus", card) is True
    assert card.creatures[1].defense == 9 and card.stats.to_dict() == creatures[0]
    assert not any(p.startswith("SEVERAL CREATURES") for p in card.problems)
    assert CardEdits().apply("Tainted Lazarus", card) is True
    assert repr(card) == original


def test_creatures_then_single_stat():
    """creatures is applied first, so a stat edit still changes creature 1's copy in card.stats."""
    card = _card()
    _edits({"Foo Bar": {"hp": 9, "creatures": [{"hp": 7, "def": 4, "atk": 5, "spd": 5}, {"hp": 5, "def": None}]}}
           ).apply("Foo Bar", card)
    assert [c.to_dict() for c in card.creatures] == [{"hp": 7, "def": 4, "atk": 5, "spd": 5},
                                                    {"hp": 5, "def": -1, "atk": -1, "spd": -1}]
    assert card.stats.to_dict() == {"hp": 9, "def": 4, "atk": 5, "spd": 5}
