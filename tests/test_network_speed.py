"""Network and speed pass: same results with fewer requests (search fast paths, tree ETags)."""
import json
import random
from difflib import SequenceMatcher

from pandas import DataFrame

import commands.github_api as ga
import commands.query_card as qc
import commands.type_emojis as te


# ========================
# Fakes
# ========================
class Reply:
    def __init__(self, status=200, body=None, etag="", content=b""):
        self.status_code, self._body, self.content = status, body, content
        self.headers = {"ETag": etag} if etag else {}

    def json(self):
        return self._body

    @property
    def text(self):
        return json.dumps(self._body) if self._body is not None else ""

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)

    def close(self):
        pass


def _tree(*paths):
    return {"tree": [{"path": p, "sha": s} for p, s in paths]}


# ========================
# Search
# ========================
def _brute_fuzzy(df, ability, ratio):
    scores = df["ability"].apply(lambda x: SequenceMatcher(None, ability, x.lower()).ratio())
    return list(scores[scores > ratio].index)


def test_fuzzy_prefilter_matches_scoring_every_card():
    rnd = random.Random(3)
    words = "deal damage to a creature draw two cards heal one minion gain stars at the end of turn".split()
    texts = [" ".join(rnd.choice(words) for _ in range(rnd.randrange(0, 15))) for _ in range(120)] + ["", "x"]
    repo = qc.CardRepository()
    repo.cards_dff = DataFrame({"ability": texts}, index=[f"Card {i}" for i in range(len(texts))])
    queries = [" ".join(rnd.choice(words) for _ in range(rnd.randrange(1, 12))) for _ in range(30)] + ["", "a" * 120]
    for ratio in (0.0, 0.6, 0.9, 1.0):
        repo.match_ratio = ratio
        for q in queries:
            assert list(repo._fuzzy_labels(q)) == _brute_fuzzy(repo.cards_dff, q, ratio), (ratio, q)


def test_fuzzy_cache_follows_new_dataframes():
    repo = qc.CardRepository()
    repo.cards_dff = DataFrame({"ability": ["deal 2 damage"]}, index=["A"])
    assert list(repo._fuzzy_labels("deal 2 damage")) == ["A"]
    repo.cards_dff = DataFrame({"ability": ["draw a card"]}, index=["B"])        # prep_dataframes replaced it
    assert list(repo._fuzzy_labels("draw a card")) == ["B"]


# ========================
# Tree ETags
# ========================
def _fake_github(monkeypatch, state, log):
    def get(url, headers=None, **kw):
        tree, etag = state["tree"], state["etag"]
        hit = (headers or {}).get("If-None-Match") == etag
        log.append(304 if hit else 200)
        return Reply(304) if hit else Reply(200, tree, etag)
    monkeypatch.setattr(ga, "get", get)
    monkeypatch.setattr(qc.CardRepository, "_populate_site_ids", lambda self: None)
    monkeypatch.setattr(qc.CardRepository, "_populate_edit_names", lambda self: None)


def test_the_card_list_tree_etag_makes_the_next_fetch_a_304(monkeypatch):
    state, log = {"tree": _tree(("Items/A.png", "a1")), "etag": '"1"'}, []
    _fake_github(monkeypatch, state, log)
    repo = qc.CardRepository()
    repo.populate_files()
    state.update(tree=_tree(("Items/A.png", "a2"), ("Items/B.png", "b1")), etag='"2"')
    assert repo.populate_files() == 200 and sorted(repo.git_files) == ["a.png", "b.png"]
    log.clear()
    assert repo.populate_files() == 200 and log == [304]


def test_emoji_listing_reuses_and_seeds_the_card_tree(monkeypatch):
    state, log = {"tree": _tree(("Types/Fire.png", "f1"), ("Items/A.png", "a1")), "etag": '"1"'}, []
    _fake_github(monkeypatch, state, log)
    repo = qc.CardRepository()
    monkeypatch.setattr(qc, "card_repo", repo)
    _, icons = te._list_type_icons()                                                # startup, card list not loaded
    assert set(icons) == {"fire", "fire_no"} and log == [200]
    assert repo.populate_files() == 200 and log == [200, 304] and repo.git_files     # the card list's fetch: 304
    _, again = te._list_type_icons()                                                # /sync_emojis
    assert again == icons and log == [200, 304, 304]


# ========================
# Card Edits.json names
# ========================
EDITS_TEXT = {"e": '{"cards": {"Foo": {"name": "Web Foo"}}}'}


def _edit_names(monkeypatch, fetched, *entries):
    monkeypatch.setattr(qc, "blob_text", lambda repo, sha, **kw: fetched.append(sha) or EDITS_TEXT[sha])
    repo = qc.CardRepository()
    repo.git_files = {"foo.png": "Items/Foo.png"}
    repo._tree = _tree(*entries)
    repo._populate_edit_names()
    return repo


def test_edit_names_read_from_site(monkeypatch):
    """Card Edits.json is read from site/ (EDITS_PATH), once per blob sha; no file, no download."""
    path = qc.EDITS_PATH
    assert path == "site/Card Edits.json"
    fetched = []
    repo = _edit_names(monkeypatch, fetched, (path, "e"))
    assert repo.git_files["web_foo.png"] == "Items/Foo.png" and fetched == ["e"]
    fetched.clear()
    repo = _edit_names(monkeypatch, fetched, ("Items/Foo.png", "x"))
    assert repo.git_files == {"foo.png": "Items/Foo.png"} and fetched == []
    repo._tree = _tree((path, "e"))                                                  # same sha: no second download
    repo._populate_edit_names()
    assert fetched == ["e"] and "web_foo.png" in repo.git_files
    repo._populate_edit_names()
    assert fetched == ["e"]
