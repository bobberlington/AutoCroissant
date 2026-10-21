"""Network and speed pass: same results with fewer requests (search fast paths, commit pages, tree ETags, TTS caches)."""
import json
import random
from difflib import SequenceMatcher, get_close_matches
from types import SimpleNamespace

import pytest
from pandas import DataFrame

import commands.github_api as ga
import commands.psd_analyzer as pa
import commands.query_card as qc
import commands.repo_traverser as rt
import commands.tts_save as ts
import commands.type_emojis as te
import commands.utils as utils


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


class Commit:
    def __init__(self, n):
        self.commit = SimpleNamespace(committer=SimpleNamespace(date=SimpleNamespace(timestamp=lambda: float(n))),
                                      author=SimpleNamespace(name=f"author{n}"))


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


def test_closest_match_exact_key_fast_path():
    rnd = random.Random(5)
    repo = qc.CardRepository()
    names = ["".join(rnd.choice("abcde_") for _ in range(rnd.randrange(1, 12))) + ".png" for _ in range(200)]
    names.append("q" * 230 + ".png")                                               # autojunk length: no fast path
    repo.git_files = {n: n for n in names}
    for q in names[:50] + ["abc", "zz", "", "q" * 230] + [n[:len(n) - 5] for n in names[50:80]]:
        expected = get_close_matches(repo._normalize_card_name(q), list(repo.git_files), n=1, cutoff=repo.match_ratio)
        assert repo._find_closest_match(q) == (expected[0] if expected else None)


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
    monkeypatch.setattr(rt, "COMMITS_PAGE", 3)
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
    assert tr._get_remote_timestamp(repo, "Items/B.psd", "B")[1] is None                  # no commits: the default


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
# TTS save
# ========================
def test_tts_objects_cached_by_sha(monkeypatch):
    monkeypatch.setattr(ts, "_objects", {})
    monkeypatch.setattr(ts, "_used", set())
    fetched = []
    assert ts._object("s1", lambda: fetched.append(1) or b"x") == b"x"
    assert ts._object("s1", lambda: fetched.append(1) or b"y") == b"x" and fetched == [1]


def _tts_403(monkeypatch, api, headers=None, message="Resource not accessible"):
    head = "h" * 40
    def fake_api(method, path, raw=False, check=True, **kw):
        api.append((method, path))
        if path == "git/ref/heads/main":
            return Reply(200, {"object": {"sha": head}})
        if path.startswith("git/commits/"):
            return Reply(200, {"tree": {"sha": "t1"}})
        if method == "POST":
            reply = Reply(403, {"message": message})
            reply.headers.update(headers or {})
            return reply
        return Reply(200, content=b"{}")
    monkeypatch.setattr(ts, "_api", fake_api)
    monkeypatch.setattr(ts, "GIT_TOKEN", "token")
    monkeypatch.setattr(ts, "_cant_write", False)
    monkeypatch.setattr(ts, "_objects", {})
    monkeypatch.setattr(ts, "_used", set())
    monkeypatch.setattr(ts, "_files_at", lambda tree, paths: {ts.TTS_SAVE_PATH: ("100644", "b1")})
    monkeypatch.setattr(ts, "_update", lambda *a, **kw: ("report", SimpleNamespace(head=head, changed=True)))
    monkeypatch.setattr(ts.repo_history, "_head", head)
    monkeypatch.setattr(ts, "_commit_message", lambda report: "m")
    monkeypatch.setattr(ts, "_git_commit", lambda h, data, message: "c" * 40)


def test_tts_token_that_cant_write_is_remembered(monkeypatch):
    api = []
    _tts_403(monkeypatch, api)
    assert ts.update_repo_save() == ("report", "c" * 40)
    assert ("POST", "git/blobs") in api and ts._cant_write
    api.clear()
    assert ts.update_repo_save() == ("report", "c" * 40)
    assert ("POST", "git/blobs") not in api                                         # no 7.8 MB probe again
    assert ("GET", "git/blobs/b1") not in api                                       # the save blob came from the cache


@pytest.mark.parametrize("headers, message", [({"retry-after": "60"}, "x"),
                                              ({}, "You have exceeded a secondary rate limit.")])
def test_tts_rate_limit_403_is_only_this_runs(monkeypatch, headers, message):
    api = []
    _tts_403(monkeypatch, api, headers, message)
    assert ts.update_repo_save() == ("report", "c" * 40) and not ts._cant_write          # SSH this run only
    api.clear()
    ts.update_repo_save()
    assert ("POST", "git/blobs") in api


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
