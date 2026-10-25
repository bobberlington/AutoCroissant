"""Rule pages skip hidden text layers (owner plan 2026-10-22); card pages skip hidden Ability layers (owner, 2026-10-22:
only what is printed); a rule page's pictures named after icons and hidden title layers read as nothing (owner 2026-10-25)."""
import json
import types

import pytest
from psd_tools import PSDImage

from commands.psd_parser import CardValidator
from golden_cards import golden_file, parser

L11 = "100511d37848773bb8e948ba29625397f6ce5216"           # the card repo commit that deleted the H1 hidden text layers


@pytest.fixture(scope="module")
def psd_parser(clone):
    return parser()


def _parse(psd_parser, monkeypatch, psd, rel):
    """psd_parser.parse on an in-memory PSDImage."""
    monkeypatch.setattr("commands.psd_parser.PSDImage", types.SimpleNamespace(open=lambda _path: psd))
    return psd_parser.parse(rel, rel)


def _remove(psd, layers):
    for layer in layers:
        layer.parent.remove(layer)
    return psd


def _named(psd, name):
    return next(layer for layer in psd.descendants() if layer.name == name)


def test_hidden_layer_and_group_skipped(psd_parser, clone, monkeypatch):
    """Hiding a group or a text layer on a rule page reads like deleting it; the rest of the text is unchanged."""
    rel = "Auxiliary/Rulebook/16_The_Field.psd"
    shown = _parse(psd_parser, monkeypatch, PSDImage.open(clone / rel), rel).ability
    psd = PSDImage.open(clone / rel)
    _named(psd, "Secondary Field").visible = False
    _named(psd, "Phase 1 copy").visible = False
    hidden = _parse(psd_parser, monkeypatch, psd, rel)
    psd = PSDImage.open(clone / rel)
    deleted = _parse(psd_parser, monkeypatch, _remove(psd, [_named(psd, "Secondary Field"), _named(psd, "Phase 1 copy")]), rel)
    assert "SECONDARY FIELD" in shown.splitlines() and "SECONDARY FIELD" not in hidden.ability.splitlines()
    assert hidden.to_dict() == deleted.to_dict() and hidden.alt_names == deleted.alt_names


def test_card_page_skips_hidden_ability(psd_parser, clone, monkeypatch):
    """A hidden Ability layer is not printed: no ability (empty, not NO ABILITY LAYER: the layer is there), so the validator
    says ABILITY TEXT NOT FOUND unless the card prints none on purpose (ABILITY_EXCLUSIONS). The rest parses as before."""
    rel = "Auxiliary/Minions/Mini_Doomer.psd"
    golden = json.loads(golden_file(rel).read_text(encoding="utf-8"))["card"]
    psd = PSDImage.open(clone / rel)
    _named(psd, "Ability").visible = False
    card = _parse(psd_parser, monkeypatch, psd, rel)
    assert card.ability == "" and card.problems == [] and CardValidator.validate(card) == ["ABILITY TEXT NOT FOUND"]
    stats = card.stats
    assert (stats.hp, stats.defense, stats.attack, stats.speed) == tuple(golden[k] for k in ("hp", "def", "atk", "spd"))
    card.name = "Shadow Duelist"
    assert CardValidator.validate(card) == []


def test_relaid_ability_counts_while_its_words_are_printed(psd_parser, clone, monkeypatch):
    """Butterfly hides Ability and prints the same words in two visible pieces over it: its ability is the hidden layer's
    text. Hide a piece and the words are no longer all printed: no ability."""
    rel = "Creatures/Kirby/5 Stars/Butterfly.psd"
    if not (clone / rel).exists():
        pytest.skip(f"{rel} not in the clone")
    psd = PSDImage.open(clone / rel)
    hidden = _named(psd, "Ability")
    if hidden.is_visible() or not psd_parser._relaid(hidden, [l for l in psd.descendants() if l.kind == "type"]):
        pytest.skip(f"{rel} no longer lays its ability out again")
    card = _parse(psd_parser, monkeypatch, psd, rel)
    assert card.ability.split() == hidden.text.split() and CardValidator.validate(card) == []
    psd = PSDImage.open(clone / rel)
    _named(psd, "Section 2").visible = False
    card = _parse(psd_parser, monkeypatch, psd, rel)
    assert card.ability == "" and CardValidator.validate(card) == ["ABILITY TEXT NOT FOUND"]

