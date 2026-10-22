"""Rule pages skip hidden text layers (owner plan 2026-10-22); card pages still read hidden Ability layers."""
import json
import subprocess
import types
from io import BytesIO

import pytest
from psd_tools import PSDImage

from golden_cards import golden_file, parser

L11 = "100511d37848773bb8e948ba29625397f6ce5216"           # the card repo commit that deleted the H1 hidden text layers
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


def test_card_page_reads_hidden_ability(psd_parser, clone, monkeypatch):
    """Card parsing is unchanged: a hidden Ability layer still counts."""
    rel = "Auxiliary/Minions/Mini_Doomer.psd"
    psd = PSDImage.open(clone / rel)
    _named(psd, "Ability").visible = False
    card = _parse(psd_parser, monkeypatch, psd, rel)
    assert card.ability == json.loads(golden_file(rel).read_text(encoding="utf-8"))["card"]["ability"]


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
