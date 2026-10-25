"""/query_ability output: one card as its image URL alone (Discord previews it); more as grid images of 20, each message
listing its cards, in order."""
import pytest
from io import BytesIO
from PIL import Image
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
def test_one_card_is_its_image_link(repo):
    out, kwargs = _search(repo, 1)
    assert out == ["Found 1 results for 'Fire'!", _url(0), "Displayed 1 of 1 results for 'Fire'."]
    assert "links" in kwargs[2]


def test_many_cards_come_as_grids(repo):
    """25 cards: two messages, each one grid image and its cards' list; the header and footer ride along."""
    repo.site_url = lambda key: None if key == "card_1.png" else f"https://site/c/{key}/"
    out, kwargs = _search(repo, 25)
    assert len(out) == 2 and [kw["file"].filename for kw in kwargs] == ["cards-1.jpg", "cards-2.jpg"]
    first, last = out[0].split("\n"), out[1].split("\n")
    assert first[0] == "Found 25 results for 'Fire'!" and len(first) == 21
    assert first[1] == "**1.** [Card 0](<https://site/c/card_0.png/>)" and first[2] == "**2.** Card 1"   # no page: name
    assert last[0].startswith("**21.** [Card 20]") and last[5] == "Displayed 25 of 25 results for 'Fire'."
    assert "links" in kwargs[1] and "links" not in kwargs[0]
    grid = Image.open(BytesIO(kwargs[0]["file"].fp.read()))
    assert grid.format == "JPEG" and grid.width == 4 * 308 + 8                          # 4 columns of 300 px cards


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
    repo._tree = {"tree": [{"path": qc.EDITS_PATH, "sha": "sha"}]}
    repo._edit_names = ("sha", {"Raw Cock": ["Raw Kawk"], "Sans": ["snas.", "Skeleton"]})       # already downloaded
    repo._populate_edit_names()
    assert repo.git_files["raw_kawk.png"] == "Creatures/Mario/3 Stars/Raw_Cock.png"
    assert repo.git_files["snas..png"] == repo.git_files["skeleton.png"] == "Creatures/Sans.png"
