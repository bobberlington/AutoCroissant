"""Rule pages skip hidden text layers (owner plan 2026-10-22); card pages still read hidden Ability layers."""
import json
import subprocess
import types
from pathlib import Path

import pytest
from psd_tools import PSDImage

from conftest import blob_sha
from golden_cards import golden_file, parser

COPIES = Path("/private/tmp/std-queue/rulebook/cards")    # the H1 copies (hidden text deleted), read-only
COPIES_BASE = "75e1f776d79c08132808dd2041181b326bf7e70d"   # the card repo commit they were made from
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
    """Every rule page in the clone parses as if its hidden text layers were deleted."""
    pages = sorted((clone / "Auxiliary/Rulebook").glob("*.psd"))
    with_hidden = [page for page in pages if _hidden_text(PSDImage.open(page))]
    if not with_hidden:
        pytest.skip("no rule page in the clone has hidden text left")
    for page in with_hidden:
        rel = f"Auxiliary/Rulebook/{page.name}"
        psd = PSDImage.open(page)
        got = _parse(psd_parser, monkeypatch, psd, rel)
        psd = PSDImage.open(page)
        want = _parse(psd_parser, monkeypatch, _remove(psd, _hidden_text(psd)), rel)
        assert got.to_dict() == want.to_dict(), page.name


def _base_blob(clone, rel):
    try:
        return subprocess.run(["git", "-C", str(clone), "rev-parse", f"{COPIES_BASE}:{rel}"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, OSError):
        return None


@pytest.mark.parametrize("page", H1_PAGES)
def test_h1_copies(psd_parser, clone, monkeypatch, page):
    """A page that still has the H1 hidden text parses the same as its copy with those layers deleted."""
    rel = f"Auxiliary/Rulebook/{page}.psd"
    copy = COPIES / f"{page}.psd"
    if not copy.exists() or blob_sha(clone / rel) != _base_blob(clone, rel):
        pytest.skip("no H1 copy, or the clone page is no longer the one the copy was made from")
    assert _hidden_text(PSDImage.open(clone / rel)) and not _hidden_text(PSDImage.open(copy))
    got = _parse(psd_parser, monkeypatch, PSDImage.open(clone / rel), rel)
    want = _parse(psd_parser, monkeypatch, PSDImage.open(copy), rel)
    assert got.ability == want.ability
