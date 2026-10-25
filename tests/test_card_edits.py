"""CardEdits (site/Card Edits.json): parsing, validation and apply/restore."""
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
    ("subseries", "Paper Mario", True), ("subseries", None, True), ("subseries", 1, False),
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


def test_subseries_edit_is_database_only():
    """No PSD value exists, so parsed_values keeps None and dropping the edit clears the field."""
    card = _card(series="mario")
    e = _edits({"Foo Bar": {"subseries": "Paper Mario", "series": "Mario"}})
    assert e.apply("Foo Bar", card) is True
    assert (card.series, card.subseries) == ("Mario", "Paper Mario")
    assert card.parsed_values == {"subseries": None, "series": "mario", "problems": ["MISSPELT TYPE: tornado"]}
    assert card.problems == ["MISSPELT TYPE: tornado"]       # subseries answers no parser problem
    assert e.get(card, "subseries") == "Paper Mario"
    _edits({"Foo Bar": {"subseries": None}}).apply("Foo Bar", card)
    assert card.subseries is None and card.series == "mario"
    assert CardEdits().apply("Foo Bar", card) is True and card.parsed_values is None
    assert card.subseries is None and "subseries" not in card.to_dict()


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


def test_creatures_edit_undone_brings_back_the_stats():
    """A card whose PSD reads no creatures: undoing a creatures edit gives back its own stats, not creature 1's."""
    from commands.card_edits import as_parsed
    card = _card(stats=CardStats(hp=6, defense=2, attack=3, speed=4))
    original = repr(card)
    _edits({"Foo Bar": {"creatures": [{"hp": 7, "def": 4, "atk": 5, "spd": 5}, {"hp": 5}]}}).apply("Foo Bar", card)
    assert card.stats.to_dict() == {"hp": 7, "def": 4, "atk": 5, "spd": 5}
    parsed = as_parsed(card)
    assert parsed.creatures is None and parsed.stats.to_dict() == {"hp": 6, "def": 2, "atk": 3, "spd": 4}
    assert CardEdits().apply("Foo Bar", card) is True and repr(card) == original


# ========================
# Two kinds of edits (EDIT_PIPELINE_DESIGN section 3)
# ========================
KINDS = [("name", "psd"), ("stars", "psd"), ("types", "psd"), ("hp", "psd"), ("def", "psd"), ("atk", "psd"), ("spd", "psd"),
         ("ability", "psd"), ("type", "folder"), ("subtype", "folder"), ("series", "folder"), ("subseries", "database"),
         ("note", "database"), ("aliases", "database"), ("dup_of", "database"), ("hidden", "database"),
         ("problems_ok", "database"), ("allow_high_stats", "database"), ("creatures", "database")]
SAMPLE = {"name": "Foo Prime", "stars": 3, "types": ["ice"], "hp": 7, "def": 7, "atk": 7, "spd": 7, "ability": "New.", "type": "item",
          "subtype": "heal", "series": "Mario", "subseries": "Paper Mario", "note": "n", "aliases": ["f"], "dup_of": "Bar", "hidden": True,
          "problems_ok": True, "allow_high_stats": True,
          "creatures": [{"hp": None, "def": 2, "atk": 3, "spd": 4}, {"hp": 5, "def": 6, "atk": 7, "spd": 8}]}   # creature 1 = the card's stats
PSD_VALUES = {"name": None, "stars": 2, "types": ["fire"], "hp": None, "def": 2, "atk": 3, "spd": 4, "ability": ""}   # what _card(stars=2) read
FOLDER_VALUES = {"type": "creature", "subtype": None, "series": "mario"}


def test_kind_covers_every_field():
    assert CardEdits.KIND.keys() == CardEdits.TYPES.keys() == dict(KINDS).keys()
    assert set(CardEdits.KIND.values()) == {"psd", "folder", "database"}


@pytest.mark.parametrize("name,kind", KINDS)
def test_kind_table(name, kind):
    """Each field's kind; apply saves what the edit replaces (the PSD's value, the folder's, or nothing) and dropping it restores."""
    assert CardEdits.KIND[name] == kind
    card = _card(series="mario", stars=2)
    original = repr(card)
    e = _edits({"Foo Bar": {name: SAMPLE[name]}})
    applied = name in CardEdits.ATTRS or name in CardEdits.STATS or name == "creatures"
    assert e.apply("Foo Bar", card) is applied                # aliases and dup_of never touch the card
    if applied:
        assert e.get(card, name) == SAMPLE[name]
        saved = card.parsed_values[name]
        if kind == "psd":
            assert saved == PSD_VALUES[name]
        elif kind == "folder":
            assert saved == FOLDER_VALUES[name]
        else:
            assert saved in (None, False)                     # database-only: no PSD or folder value exists
    assert CardEdits().apply("Foo Bar", card) is applied
    assert repr(card) == original


def test_psd_meta_rides_beside_by_and_at():
    """_psd (the pipeline's verdict) is kept like _by/_at: out of the edits, readable, writable, ignored by apply()."""
    verdict = {"state": "refused", "for": {"hp": 5}, "psd": "e0b11bab", "reason": "stat>10: not yet", "at": "2026-10-20T00:00:00Z"}
    e = _edits({"Foo_Bar": {"hp": 5, "_by": "web", "_at": "2026-10-20T00:00:00Z", "_psd": verdict},
                "Baz": {"note": "x", "_psd": {"state": "pending"}}, "Qux": {"note": "y"}})
    assert e.cards["foo bar"] == {"hp": 5}
    assert e.meta["foo bar"] == {"_by": "web", "_at": "2026-10-20T00:00:00Z", "_psd": verdict}
    assert e.psd_state("Foo Bar") == verdict
    assert e.psd_state("Baz") is None and e.psd_state("Qux") is None and e.psd_state("Nobody") is None
    card = _card()
    assert e.apply("Foo Bar", card) is True
    assert card.stats.hp == 5 and card.parsed_values == {"hp": None, "problems": ["MISSPELT TYPE: tornado"]}
    e.set_psd_state("FOO BAR", {"state": "baked", "for": {"hp": 5}})
    assert e.psd_state("Foo Bar")["state"] == "baked" and e.meta["foo bar"]["_by"] == "web"
    e.set_psd_state("Foo Bar", None)
    assert e.psd_state("Foo Bar") is None and e.meta["foo bar"] == {"_by": "web", "_at": "2026-10-20T00:00:00Z"}
    with pytest.raises(ValueError):
        e.set_psd_state("Foo Bar", {"state": "done"})
    with pytest.raises(ValueError):
        e.set_psd_state("Foo Bar", {"state": "pending"})       # pending is the website's computation, never written
    e.set_psd_state("New Card", {"state": "baked"})            # a card with no edits yet
    assert e.psd_state("New Card") == {"state": "baked"} and "new card" not in e.cards
    e.set_psd_state("Nobody", None)
    assert "nobody" not in e.meta                              # removing nothing leaves no empty meta behind

