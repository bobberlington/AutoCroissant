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
    return repo


def _search(repo, n):
    names = [f"Items/Card {i}" for i in range(n)]
    repo.ability_search_engine = lambda ability: DataFrame({"ability": ["fire"] * n}, index=names)
    qc.query_ability(utils.make_simple_fake_interaction(1), "Fire", -1)
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


def test_nothing_found_is_one_reply_with_the_websites_guesses(repo, monkeypatch):
    """No results: one message with the website's "did you mean" (search.js suggest), each a Card page if it has one."""
    monkeypatch.setattr(qc, "stats_db", SimpleNamespace(stats={"Water": _card("Water"), "Firefly": _card("Firefly"),
        "Fire Ant": _card("Fire Ant"), "Fire": SimpleNamespace(display_name="Fire", hidden=True)}))
    repo.git_files = {"fire_ant.png": "Creatures/Fire_Ant.png"}
    repo.site_url = lambda key: f"https://site/c/{key[:-4]}/" if key else None
    out, kwargs = _search(repo, 0)
    assert out == ["No cards match 'Fire'.\nDid you mean: [Fire Ant](<https://site/c/fire_ant/>), Firefly?"]
    assert "links" in kwargs[0]


def test_many_cards_come_as_grids(repo):
    """25 cards: two messages, each one grid image and its cards' list; the header and footer ride along."""
    repo.site_url = lambda key: None if key == "card_1.png" else f"https://site/c/{key}/"
    out, kwargs = _search(repo, 25)
    assert len(out) == 2 and [kw["file"].filename for kw in kwargs] == ["cards-1.jpg", "cards-2.jpg"]
    first, last = out[0].split("\n"), out[1].split("\n")
    assert first[0] == "Found 25 results for 'Fire'!" and len(first) == 21     # A-Z as text: Card 0, 1, 10..19, 2, ...
    assert first[1] == "**1.** [Card 0](<https://site/c/card_0.png/>)" and first[2] == "**2.** Card 1"   # no page: name
    assert last[0].startswith("**21.** [Card 5]") and last[5] == "Displayed 25 of 25 results for 'Fire'."
    assert "links" in kwargs[1] and "links" not in kwargs[0]
    grid = Image.open(BytesIO(kwargs[0]["file"].fp.read()))
    assert grid.format == "JPEG" and grid.width == 4 * 308 + 8                          # 4 columns of 300 px cards


def test_grids_go_as_the_website(repo, monkeypatch):
    """Printed titles, the card named like the query first, then A-Z; pictures still found by file name."""
    import commands.card_grid as cg
    seen = []
    monkeypatch.setattr(cg, "grids", lambda cards: seen.extend(cards) or [b"jpg"])
    repo.git_files = {"ant_queen.png": "Items/Ant_Queen.png"}
    repo.ability_search_engine = lambda ability: DataFrame(
        {"ability": ["fire"] * 3, "display_name": ["Zap", None, "Fire Ant"]}, index=["Bolt", "apple", "Ant Queen"])
    qc.query_ability(utils.make_simple_fake_interaction(1), "fire-ANT")
    assert [(caption, url) for caption, url, *_ in seen] == [
        ("1. Fire Ant", "https://img/Ant_Queen.png"), ("2. apple", None), ("3. Zap", None)]
    assert utils.dispatch_queue[0][1]["content"].split("\n")[1:4] == ["**1.** Fire Ant", "**2.** apple", "**3.** Zap"]


def test_grid_caches_only_pinned_pictures(repo, monkeypatch):
    """Each card's website thumb (named by its blob); its raw link is cached only when commit-pinned (CDN lag)."""
    import commands.card_grid as cg
    repo.git_files = {f"card_{i}.png": f"Items/Card_{i}.png" for i in range(2)}
    repo.blobs = {"Items/Card_0.png": "b0", "Items/Card_1.png": "b1"}
    repo.thumb_tag = "t3"
    monkeypatch.setattr(qc, "pinned_ref", lambda _repo, path, blob: "c0" if blob == "b0" else None)
    seen = []
    monkeypatch.setattr(cg, "grids", lambda cards: seen.extend(cards) or [])
    monkeypatch.setattr(cg, "prune", lambda keep: None)
    qc.post_grids(utils.make_simple_fake_interaction(1), ["Card 0", "Card 1"], "", "")
    assert [card[2:] for card in seen] == [("b0", f"{qc.SITE_URL}/thumbs/b0-360-t3.avif", True),
                                           ("b1", f"{qc.SITE_URL}/thumbs/b1-360-t3.avif", False)]


def test_grid_pictures_from_the_websites_thumbs(tmp_path, monkeypatch):
    """The thumb first (cached even beside a branch link); a failed one: the PNG (cached if pinned); no AVIF: the PNG."""
    import commands.card_grid as cg
    from requests import HTTPError
    served, asked = {"thumb": (360, (255, 0, 0)), "png": (1200, (0, 0, 255))}, []

    def get(url, timeout):
        asked.append(url)
        if url not in served:
            return SimpleNamespace(raise_for_status=lambda: (_ for _ in ()).throw(HTTPError("404")))
        out = BytesIO()
        Image.new("RGBA", (served[url][0], served[url][0] * 3 // 2), served[url][1]).save(out, "PNG")
        return SimpleNamespace(raise_for_status=lambda: None, content=out.getvalue())
    monkeypatch.setattr(cg, "get", get)
    monkeypatch.setattr(cg, "CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(cg, "AVIF", True)
    color = lambda pic: pic.getpixel((150, 200))
    assert color(cg.picture("png", "b1", "thumb", pinned=False)) == (255, 0, 0) and asked == ["thumb"]
    assert (tmp_path / "b1.jpg").is_file()
    assert color(cg.picture("png", "b2", "gone", pinned=False)) == (0, 0, 255) and asked[1:] == ["gone", "png"]
    assert not (tmp_path / "b2.jpg").exists()                                  # a branch link's: not cached
    assert color(cg.picture("png", "b2", "gone")) == (0, 0, 255) and (tmp_path / "b2.jpg").is_file()   # pinned
    monkeypatch.setattr(cg, "AVIF", False)
    asked.clear()
    assert color(cg.picture("png", "b3", "thumb")) == (0, 0, 255) and asked == ["png"]


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
