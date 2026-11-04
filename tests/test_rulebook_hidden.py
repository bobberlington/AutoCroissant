"""Cards skip hidden Ability layers (owner, 2026-10-22: only what is printed). Rule-page parsing removed 2026-11-04."""
import json
import types

import pytest
from psd_tools import PSDImage

from commands.psd_parser import CardValidator
from golden_cards import golden_file, parser



@pytest.fixture(scope="module")
def psd_parser(clone):
    return parser()


def _parse(psd_parser, monkeypatch, psd, rel):
    """psd_parser.parse on an in-memory PSDImage."""
    monkeypatch.setattr("commands.psd_parser.PSDImage", types.SimpleNamespace(open=lambda _path: psd))
    return psd_parser.parse(rel, rel)


def _named(psd, name):
    return next(layer for layer in psd.descendants() if layer.name == name)


def test_card_page_skips_hidden_ability(psd_parser, clone, monkeypatch):
    """A hidden Ability layer is not printed: no ability (empty), and no problem (cards need no ability, owner 10-25). The
    rest parses as before."""
    rel = "Auxiliary/Minions/Mini_Doomer.psd"
    golden = json.loads(golden_file(rel).read_text(encoding="utf-8"))["card"]
    psd = PSDImage.open(clone / rel)
    _named(psd, "Ability").visible = False
    card = _parse(psd_parser, monkeypatch, psd, rel)
    assert card.ability == "" and card.problems == [] and CardValidator.validate(card) == []
    stats = card.stats
    assert (stats.hp, stats.defense, stats.attack, stats.speed) == tuple(golden[k] for k in ("hp", "def", "atk", "spd"))


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
    assert card.ability == "" and CardValidator.validate(card) == []

