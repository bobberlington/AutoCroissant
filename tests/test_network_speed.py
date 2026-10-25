"""Network and speed pass: same results with fewer requests (search fast paths, commit pages, tree ETags)."""
import json
import random
from difflib import SequenceMatcher
from types import SimpleNamespace

import pytest
from pandas import DataFrame

import commands.card_data as cd
import commands.github_api as ga
import commands.psd_analyzer as pa
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


class Commit:
    def __init__(self, n):
        self.commit = SimpleNamespace(committer=SimpleNamespace(date=SimpleNamespace(timestamp=lambda: float(n))),
                                      author=SimpleNamespace(name=f"author{n}", email=f"author{n}@x"))


class Commits:
    """Newest first, like GitHub: n commits, page size per_page."""
    def __init__(self, n, per_page, calls):
        self.items, self.per_page, self.calls = [Commit(i) for i in range(n, 0, -1)], per_page, calls

    def get_page(self, page):
        self.calls.append("page")
        return self.items[page * self.per_page:(page + 1) * self.per_page]

    @property
    def reversed(self):
        self.calls.append("reversed")
        return list(reversed(self.items))


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
# /update_stats commit lookups
# ========================
def test_commit_lookup_is_one_request(monkeypatch):
    calls = []
    repo = SimpleNamespace(get_commits=lambda path: Commits(8, pa.COMMITS_PAGE, calls))
    tr = pa.RepositoryTraverser(pa.StatsDatabase())
    monkeypatch.setattr(tr, "_repo", lambda: repo)
    assert tr._get_remote_timestamp(repo, "Items/New.psd", "New") == (8.0, "author1")      # new: first commit's author
    assert calls == ["page"]
    tr.db.stats["Old"] = SimpleNamespace(author="Someone")
    calls.clear()
    assert tr._get_remote_timestamp(repo, "Items/Old.psd", "Old") == (8.0, None)
    assert calls == ["page"]


def test_commit_lookup_more_than_one_page(monkeypatch):
    calls = []
    monkeypatch.setattr(cd, "COMMITS_PAGE", 3)                                     # moved to card_data with the build
    repo = SimpleNamespace(get_commits=lambda path: Commits(7, 3, calls))
    tr = pa.RepositoryTraverser(pa.StatsDatabase())
    monkeypatch.setattr(tr, "_repo", lambda: repo)
    assert tr._get_remote_timestamp(repo, "Items/New.psd", "New") == (7.0, "author1")
    assert calls == ["page", "reversed"]


def test_commit_lookup_prefetched_and_empty(monkeypatch):
    calls = []
    repo = SimpleNamespace(get_commits=lambda path: Commits(0, pa.COMMITS_PAGE, calls))
    tr = pa.RepositoryTraverser(pa.StatsDatabase())
    monkeypatch.setattr(tr, "_repo", lambda: repo)
    tr._lookups = {"Items/A.psd": SimpleNamespace(result=lambda: [Commit(5)])}
    assert tr._get_remote_timestamp(repo, "Items/A.psd", "A") == (5.0, "author5") and calls == []
    with pytest.raises(LookupError):                         # no commits: the card waits, no made-up date (year 1000
        tr._get_remote_timestamp(repo, "Items/B.psd", "B")   # or the run's clock, kept for good) nor a missing author
    with pytest.raises(LookupError):                         # no repository: the card waits
        tr._get_remote_timestamp(None, "Items/B.psd", "B")


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


def test_update_stats_tree_etag_makes_the_next_fetch_a_304(monkeypatch):
    state, log = {"tree": _tree(("Items/A.png", "a1")), "etag": '"1"'}, []
    _fake_github(monkeypatch, state, log)
    repo = qc.CardRepository()
    repo.populate_files()
    state.update(tree=_tree(("Items/A.png", "a2"), ("Items/B.png", "b1")), etag='"2"')
    repo.populate_files(state["tree"], None, '"2"')                                 # what /update_stats queues
    assert sorted(repo.git_files) == ["a.png", "b.png"]
    log.clear()
    assert repo.populate_files() == 200 and log == [304]
    repo.populate_files(state["tree"])                                              # a tree without its ETag
    log.clear()
    repo.populate_files()
    assert log == [200]                                                             # never a 304 for another tree


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
    """Card Edits.json is read from site/ (EDITS_PATHS), once per blob sha; no file, no download."""
    (path,) = qc.EDITS_PATHS
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


def test_commit_lookups_get_a_client_per_thread(monkeypatch):
    import github
    from concurrent.futures import ThreadPoolExecutor
    made = []
    class FakeGithub:
        def __init__(self, **kw):
            made.append(self)
        def get_repo(self, name, lazy=False):
            assert lazy                                                                 # no request
            return SimpleNamespace(client=self)
    monkeypatch.setattr(github, "Github", FakeGithub)
    tr = pa.RepositoryTraverser(pa.StatsDatabase())
    with ThreadPoolExecutor(pa.LOOKUP_WORKERS) as ex:
        clients = set(ex.map(lambda _: tr._repo().client, range(20)))
    assert len(clients) == len(made) <= pa.LOOKUP_WORKERS
    assert tr._repo() is tr._repo() and tr._repo().client not in clients                # the parse thread's own
