"""Rule pages skip hidden text layers (owner plan 2026-10-22); card pages skip hidden Ability layers (owner, 2026-10-22:
only what is printed)."""
import json
import subprocess
import types
from io import BytesIO

import pytest
from psd_tools import PSDImage

from commands.card_models import card_name
from commands.psd_parser import CardValidator
from golden_cards import golden_file, parser

L11 = "100511d37848773bb8e948ba29625397f6ce5216"           # the card repo commit that deleted the H1 hidden text layers
HIDDEN_ABILITY = ("Auxiliary/Minions/Shadow_Duelist.psd", "Auxiliary/Minions/Abyss_Watcher.psd", "Auxiliary/Minions/Warrior_Dee.psd",
                  "Auxiliary/Minions/Bulborb Larva.psd", "Creatures/Kirby/5 Stars/Butterfly.psd",
                  "Creatures/Other/3 Stars/Empress Bulblax.psd", "Creatures/R.E.P.O/2 Stars/Apex_Predator.psd")
H1_PAGES = ("09_TakingYourTurn", "11_Abilities", "12_Reactions2", "22_Negation")


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


def _hidden_text(psd):
    return [layer for layer in psd.descendants() if layer.kind == "type" and not layer.is_visible()]


def _page(clone, rev, rel):
    """A rule page from the clone's history."""
    try:
        data = subprocess.run(["git", "-C", str(clone), "show", f"{rev}:{rel}"], capture_output=True, check=True).stdout
    except (subprocess.CalledProcessError, OSError):
        pytest.skip(f"the clone's history lacks {rev}:{rel}")
    return PSDImage.open(BytesIO(data))


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
    shown = _parse(psd_parser, monkeypatch, PSDImage.open(clone / rel), rel)
    assert shown.ability == golden["ability"]
    assert {**card.to_dict(), "ability": None} == {**shown.to_dict(), "ability": None}
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


def test_clone_cards_with_only_hidden_ability(psd_parser, clone, monkeypatch):
    """The clone's cards whose Ability layers are all hidden (7 on 2026-10-22): the four that print the words in other
    layers keep them; Shadow Duelist, Abyss Watcher and Warrior Dee print none, parse to no ability, and are excluded."""
    found = {}
    for rel in HIDDEN_ABILITY:
        if not (clone / rel).exists():
            continue
        psd = PSDImage.open(clone / rel)
        texts = [layer for layer in psd.descendants() if layer.kind == "type"]
        abilities = [layer for layer in texts if layer.name.lower() == "ability"]
        if not abilities or any(layer.is_visible() for layer in abilities):
            continue                                       # shown or redone since
        card = _parse(psd_parser, monkeypatch, psd, rel)
        found[rel] = (psd_parser._relaid(abilities[0], texts), card.ability, CardValidator.validate(card))
    if not found:
        pytest.skip("no clone card with only a hidden Ability layer")
    for rel, (relaid, ability, problems) in found.items():
        assert bool(ability) == relaid and problems == [], rel
    assert sorted(card_name(rel) for rel, (relaid, _, _) in found.items() if not relaid) == \
        sorted(card_name(rel) for rel in HIDDEN_ABILITY if rel in found and card_name(rel) in CardValidator.ABILITY_EXCLUSIONS)


def test_clone_rule_pages(psd_parser, clone, monkeypatch):
    """Every rule page in the clone parses as if its hidden text layers were deleted (none left: the H1 pages before L11)."""
    pages = [(f"Auxiliary/Rulebook/{page.name}", lambda page=page: PSDImage.open(page))
             for page in sorted((clone / "Auxiliary/Rulebook").glob("*.psd"))]
    with_hidden = [(rel, open_page) for rel, open_page in pages if _hidden_text(open_page())]
    if not with_hidden:
        with_hidden = [(rel, lambda rel=rel: _page(clone, L11 + "^", rel)) for rel in (f"Auxiliary/Rulebook/{p}.psd" for p in H1_PAGES)]
    for rel, open_page in with_hidden:
        psd = open_page()
        assert _hidden_text(psd), rel
        got = _parse(psd_parser, monkeypatch, psd, rel)
        psd = open_page()
        want = _parse(psd_parser, monkeypatch, _remove(psd, _hidden_text(psd)), rel)
        assert got.to_dict() == want.to_dict(), rel


@pytest.mark.parametrize("page", H1_PAGES)
def test_h1_pages(psd_parser, clone, monkeypatch, page):
    """L11 deleted the page's H1 hidden text layers in Photoshop: the page before parses the same as the page after."""
    rel = f"Auxiliary/Rulebook/{page}.psd"
    before, after = _page(clone, L11 + "^", rel), _page(clone, L11, rel)
    assert _hidden_text(before) and not _hidden_text(after)
    got = _parse(psd_parser, monkeypatch, before, rel)
    want = _parse(psd_parser, monkeypatch, after, rel)
    assert got.to_dict() == want.to_dict() and got.alt_names == want.alt_names
