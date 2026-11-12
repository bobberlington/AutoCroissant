"""Cards skip hidden Ability layers (owner, 2026-10-22: only what is printed). Rule-page parsing removed 2026-11-04."""
import json
import types

from psd_tools import PSDImage

from commands.psd_edit import remove_layer
from commands.psd_parser import CardValidator
from golden_cards import golden_file, psd_parser  # noqa: F401 (psd_parser: a fixture)


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
    remove_layer(psd, _named(psd, "Ability"))                   # none at all: no ability either (11-09)
    assert psd_parser.parse(psd, rel).to_dict() == card.to_dict()
