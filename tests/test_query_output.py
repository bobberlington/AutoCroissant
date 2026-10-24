"""/query_ability output: one message per matching card, its image URL alone (Discord previews it), in order."""
import pytest
from pandas import DataFrame
from types import SimpleNamespace

import commands.query_card as qc
import commands.utils as utils


# ========================
# Fakes
# ========================
@pytest.fixture
def repo(monkeypatch):
    repo = qc.CardRepository()
    repo.cards_dff = DataFrame({"ability": ["fire"]}, index=["Items/Any"])          # not empty: the search runs
    repo.get_card_url = lambda name: f"https://img/{name.replace(' ', '_')}.png"
    monkeypatch.setattr(qc, "card_repo", repo)
    utils.dispatch_queue.clear()
    yield repo
    utils.dispatch_queue.clear()


def _search(repo, n, limit=-1):
    names = [f"Items/Card {i}" for i in range(n)]
    repo.ability_search_engine = lambda ability: DataFrame({"ability": ["fire"] * n}, index=names)
    qc.query_ability(utils.make_simple_fake_interaction(1), "Fire", limit)
    return [kw["content"] for _, kw in utils.dispatch_queue], [kw for _, kw in utils.dispatch_queue]


def _url(i):
    return f"https://img/Card_{i}.png"


# ========================
# Message count and contents
# ========================
@pytest.mark.parametrize("n", [0, 1, 10, 11, 25])
def test_one_message_per_card(repo, n):
    out, kwargs = _search(repo, n)
    assert out[0] == f"Found {n} results for 'Fire'!"
    assert out[1:len(out) - 1] == [_url(i) for i in range(n)]                     # each image URL alone, in order
    assert out[len(out) - 1] == f"Displayed {n} of {n} results for 'Fire'." and "links" in kwargs[len(kwargs) - 1]


def test_limit_caps_the_cards(repo):
    out, _ = _search(repo, 25, limit=3)
    assert out == ["Found 25 results for 'Fire'!", _url(0), _url(1), _url(2), "Displayed 3 of 25 results for 'Fire'."]


def test_a_card_without_an_image_posts_its_name(repo):
    """A match whose PNG is not in the tree: its name, never a content=None message Discord drops."""
    urls = repo.get_card_url
    repo.get_card_url = lambda name: None if name == "Card 1" else urls(name)
    out, _ = _search(repo, 3)
    assert out[1:4] == [_url(0), "Card 1", _url(2)]


def test_a_single_result_without_an_image_posts_its_name(repo):
    repo.get_card_url = lambda name: None
    out, _ = _search(repo, 1)
    assert out == ["Found 1 results for 'Fire'!", "Card 0", "Displayed 1 of 1 results for 'Fire'."]


# ========================
# Printed titles
# ========================
def _card(name, title=None, edit=False):
    return SimpleNamespace(name=name, display_name=title, parsed_values={"name": None} if edit else None)


def test_titles_join_the_name_lookup(repo, monkeypatch):
    """/query finds a card by its printed title; names already taken and cards without a PNG are left alone."""
    repo.git_files = {"clot_black.png": "Auxiliary/Minions/Clot_Black.png", "cabbage.png": "Auxiliary/Items/Cabbage.png"}
    monkeypatch.setattr(qc, "stats_db", SimpleNamespace(stats={
        "Clot Black": _card("Clot Black", "Black Clot"), "Cabbage": _card("Cabbage", "CABBAGE"), "Gone": _card("Gone", "Not Here")}))
    repo._populate_titles()
    assert repo.git_files == {"clot_black.png": "Auxiliary/Minions/Clot_Black.png", "cabbage.png": "Auxiliary/Items/Cabbage.png",
                              "black_clot.png": "Auxiliary/Minions/Clot_Black.png"}


def test_website_names_join_the_name_lookup(repo):
    """Card Edits.json names and aliases find the PNG (registry keys are lowercase; broken until 2026-10-21)."""
    repo.git_files = {"raw_cock.png": "Creatures/Mario/3 Stars/Raw_Cock.png", "creatures/sans.png": "Creatures/Sans.png"}
    repo._tree = {"tree": [{"path": qc.EDITS_PATHS[0], "sha": "sha"}]}
    repo._edit_names = ("sha", {"Raw Cock": ["Raw Kawk"], "Sans": ["snas.", "Skeleton"]})       # already downloaded
    repo._populate_edit_names()
    assert repo.git_files["raw_kawk.png"] == "Creatures/Mario/3 Stars/Raw_Cock.png"
    assert repo.git_files["snas..png"] == repo.git_files["skeleton.png"] == "Creatures/Sans.png"
