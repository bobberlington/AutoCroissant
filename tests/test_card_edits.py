"""CardEdits (site/Card Edits.json): parsing, validation and apply/restore."""
import json
from copy import deepcopy


from commands.psd_analyzer import CardEdits, CardInfo, CardStats
from commands.psd_parser import CardValidator


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


VALID_CASES = [
    ("hp", 3, True), ("hp", 0, True), ("hp", -1, False), ("hp", True, False), ("hp", 2.0, False), ("hp", None, True),
    ("stars", None, True), ("ability", None, False), ("name", None, False), ("ability", "x", True), ("ability", 1, False),
    ("types", ["a", "b"], True), ("types", ["a", 1], False), ("types", "a", False), ("hidden", True, True),
    ("hidden", 1, False), ("aliases", ["x"], True), ("unknown", 1, False),
    ("subseries", "Paper Mario", True), ("subseries", None, True), ("subseries", 1, False),
    ("creatures", [{"hp": 7, "def": None}], True), ("creatures", [{"hp": -1}], False), ("creatures", [{"hp": True}], False),
    ("creatures", [{"mana": 1}], False), ("creatures", [1], False), ("creatures", {}, False),
]


def test_valid():
    for name, value, ok in VALID_CASES:
        assert CardEdits.valid(name, value) is ok, (name, value)


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


def test_kind_table():
    """Each field's kind; apply saves what the edit replaces (the PSD's value, the folder's, or nothing) and dropping it restores."""
    for name, kind in KINDS:
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
    """_psd (the pipeline's verdict) is kept like _by/_at: out of the edits, ignored by apply()."""
    verdict = {"state": "refused", "for": {"hp": 5}, "psd": "e0b11bab", "reason": "stat>10: not yet", "at": "2026-10-20T00:00:00Z"}
    e = _edits({"Foo_Bar": {"hp": 5, "_by": "web", "_at": "2026-10-20T00:00:00Z", "_psd": verdict},
                "Baz": {"note": "x", "_psd": {"state": "pending"}}, "Qux": {"note": "y"}})
    assert e.cards["foo bar"] == {"hp": 5}
    assert e.meta["foo bar"] == {"_by": "web", "_at": "2026-10-20T00:00:00Z", "_psd": verdict}
    card = _card()
    assert e.apply("Foo Bar", card) is True
    assert card.stats.hp == 5 and card.parsed_values == {"hp": None, "problems": ["MISSPELT TYPE: tornado"]}



# ========================
# Parts (docs/Future/PARTS_EDITOR.md §3, §11 case 10)
# ========================
def test_parts_valid_and_merged_per_layer():
    """`parts` as build_site.valid_parts takes it; the fixture's entries parse whole; merge and unset by layer id."""
    from pathlib import Path
    from commands.card_edits import dump_edits, merge_parts
    fixture = json.loads((Path(__file__).parent / "fixtures" / "parts" / "card_edits.parts.json").read_text(encoding="utf-8"))
    edits = CardEdits.parse(json.dumps(fixture))
    for name, entry in fixture["cards"].items():
        assert edits.cards[name.lower()].get("parts") == entry.get("parts"), name
    part = {"field": "ability", "text": "x", "was": "y", "sha": "0123abcd", "psd": "89abcdef", "layer": "Text/Ability"}
    bad = [{"1": {**part, "field": "note"}}, {"1": {**part, "delete": True}}, {"1": {k: v for k, v in part.items() if k != "text"}},
           {"1": {**part, "sha": "XYZ"}}, {"x1": part}, {"1": {**part, "text": "é" * 2049}}, {}, {str(n): part for n in range(65)},
           {"+ability": {**part, "field": "name"}}, {"+ability": {**part, "delete": True, "text": None}}]
    assert all(not CardEdits.valid("parts", value) for value in bad)
    gone = {k: v for k, v in part.items() if k != "text"} | {"delete": True}
    assert CardEdits.valid("parts", {"1": part, "+ability": {**part, "sha": None, "psd": None}, "2": gone})
    assert CardEdits.KIND["parts"] == "psd" and CardEdits().apply("X", _card()) is False      # never applied to a card
    merged = merge_parts({"1": part, "2": part}, {"2": {**part, "text": "z"}, "3": part}, ["1"])
    assert list(merged) == ["2", "3"] and merged["2"]["text"] == "z" and merge_parts({"1": part}, None, ["1"]) is None
    text = dump_edits({"about": "a", "cards": {"C": {"ability": "x", "parts": merged, "note": "n", "_by": "w"}}})
    assert '"ability":"x","parts":{"2":' in text and json.loads(text)["cards"]["C"]["parts"] == merged
