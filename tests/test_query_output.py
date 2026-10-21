"""/query_ability output: one match posts its image; more are listed as name and link, 10 per message, in order."""
import pytest
from pandas import DataFrame
from types import SimpleNamespace

import commands.query_card as qc
import commands.utils as utils

PER = qc.RESULTS_PER_MESSAGE


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


def _search(repo, n):
    names = [f"Items/Card {i}" for i in range(n)]
    repo.ability_search_engine = lambda ability: DataFrame({"ability": ["fire"] * n}, index=names)
    qc.query_ability(utils.make_simple_fake_interaction(1), "Fire")
    return [kw["content"] for _, kw in utils.dispatch_queue], [kw for _, kw in utils.dispatch_queue]


def _line(i):
    return f"Card {i} — https://img/Card_{i}.png"


# ========================
# Message count and contents
# ========================
@pytest.mark.parametrize("n, messages", [(0, 2), (1, 3), (10, 3), (11, 4), (25, 5)])
def test_results_are_grouped_ten_per_message(repo, n, messages):
    out, kwargs = _search(repo, n)
    assert len(out) == messages
    assert out[0] == f"Found {n} results for 'Fire'!"
    assert out[len(out) - 1] == f"Displayed {n} of {n} results for 'Fire'." and "links" in kwargs[len(kwargs) - 1]
    body = out[1:len(out) - 1]
    if n == 0:
        assert body == []
    elif n == 1:
        assert body == ["https://img/Card_0.png"]                                  # one card: image preview, as before
    else:
        assert [len(m.split("\n")) for m in body] == [min(PER, n - PER * i) for i in range(len(body))]
        assert "\n".join(body).split("\n") == [_line(i) for i in range(n)]       # every card once, in order
    assert all(len(m) <= 2000 for m in out)


def test_limit_caps_the_list(repo):
    repo.ability_search_engine = lambda ability: DataFrame({"ability": ["fire"] * 25}, index=[f"Items/Card {i}" for i in range(25)])
    utils.dispatch_queue.clear()
    qc.query_ability(utils.make_simple_fake_interaction(1), "Fire", limit=3)
    out = [kw["content"] for _, kw in utils.dispatch_queue]
    assert out == ["Found 25 results for 'Fire'!", "\n".join(_line(i) for i in range(3)), "Displayed 3 of 25 results for 'Fire'."]


def test_single_result_under_limit_keeps_the_preview(repo):
    repo.ability_search_engine = lambda ability: DataFrame({"ability": ["fire"] * 5}, index=[f"Items/Card {i}" for i in range(5)])
    qc.query_ability(utils.make_simple_fake_interaction(1), "Fire", limit=1)
    out = [kw["content"] for _, kw in utils.dispatch_queue]
    assert out == ["Found 5 results for 'Fire'!", "https://img/Card_0.png", "Displayed 1 of 5 results for 'Fire'."]


def test_long_links_split_under_the_discord_limit(repo):
    repo.get_card_url = lambda name: "https://img/" + "x" * 400 + f"/{name.replace(' ', '_')}.png"
    out, _ = _search(repo, 10)
    body = out[1:len(out) - 1]
    assert len(body) > 1 and all(len(m) <= utils.BREAK_LEN for m in body)
    assert "\n".join(body).split("\n") == [f"Card {i} — " + repo.get_card_url(f"Card {i}") for i in range(10)]


def test_a_card_without_an_image_keeps_its_name(repo):
    urls = repo.get_card_url
    repo.get_card_url = lambda name: None if name == "Card 1" else urls(name)
    out, _ = _search(repo, 3)
    assert out[1] == f"{_line(0)}\nCard 1\n{_line(2)}"


def test_single_result_without_an_image_posts_its_name(repo):
    """One match whose PNG is not in the tree: its name, never a content=None message Discord drops."""
    repo.get_card_url = lambda name: None
    out, _ = _search(repo, 1)
    assert out == ["Found 1 results for 'Fire'!", "Card 0", "Displayed 1 of 1 results for 'Fire'."]


# ========================
# Printed titles
# ========================
def _card(name, title=None, edit=False):
    return SimpleNamespace(name=name, display_name=title, parsed_values={"name": None} if edit else None)


def test_printed_title_folds_case_and_spacing():
    """A title differing from the name by letter case or spacing alone shows as the name; a website name as written."""
    assert qc.printed_title(_card("Apex Predator", "A P E X P R E D A T O R")) is None
    assert qc.printed_title(_card("Poorly Drawn Cabbage", "POORLY DRAWN CABBAGE")) is None
    assert qc.printed_title(_card("Clot Black", "Black Clot")) == "Black Clot"
    assert qc.printed_title(_card("Mr Krabs", "Mr. Krabs")) == "Mr. Krabs"
    assert qc.printed_title(_card("Sans", "snas.", edit=True)) == "snas."
    assert qc.printed_title(_card("Fawful", "FAWFUL", edit=True)) == "FAWFUL"
    assert qc.printed_title(_card("Mini Doomer")) is None


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
