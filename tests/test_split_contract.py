"""The psd_analyzer split: the facade's re-exports, the shared helpers, the classifier guard."""


import commands.github_api as ga
import commands.psd_parser as pp
import commands.repo_history as rh


def test_classify_shallow_paths_are_unknown():
    unknown = {"type": "unknown"}
    assert pp.CardClassifier.classify("Card.psd") == unknown
    assert pp.CardClassifier.classify("Items/Card.psd") == unknown            # read out of bounds when compiled
    assert pp.CardClassifier.classify("Creatures/Card.psd") == unknown
    assert pp.CardClassifier.classify("Items/Heal/2 Stars/B.psd") == {"type": "item", "subtype": "heal", "stars": 2}
    assert pp.CardClassifier.classify("N.M.E/Star_Jar.psd") == {"type": "nme"}
    assert pp.CardClassifier.classify("Auxiliary/Minions/Mini_Doomer.psd")["type"] == "minion"


def test_pinned_ref_and_raw_url_agree(monkeypatch):
    monkeypatch.setattr(rh, "_paths", {"Items/A b.png": ("blob1", "c0ffee")})
    assert rh.pinned_ref(rh.REPOSITORY, "Items/A b.png", "blob1") == "c0ffee"
    assert rh.pinned_ref(rh.REPOSITORY, "Items/A b.png", "blob2") is None
    assert rh.raw_url(rh.REPOSITORY, "Items/A b.png", "blob1").endswith("/c0ffee/Items/A%20b.png")
    assert rh.raw_url("someone/else", "Items/A b.png", "blob1").endswith(f"/{ga.CARD_BRANCH}/Items/A%20b.png")

