"""
git_commit, the commit path card_bake and card_guards share: upload_blob's NO_WRITE / rate-limit split, commit_files'
API-then-SSH order against the card_bake fake GitHub, and git_commit's push outcomes with git faked (no clone, no network).
"""
from subprocess import CalledProcessError, CompletedProcess, TimeoutExpired

import pytest

import commands.git_commit as gc
import commands.github_api as ga
from conftest import blob_sha_bytes as _sha
from helpers import FakeGitHub


@pytest.fixture
def github(monkeypatch):
    fake = FakeGitHub(gc.CARD_REPO, {"a.txt": b"a", "dir/b.txt": b"b"})
    monkeypatch.setattr(ga, "get", fake.get)
    monkeypatch.setattr(ga, "GIT_TOKEN", "")
    monkeypatch.setattr(gc, "request", fake.request)
    monkeypatch.setattr(gc, "cant_write", False)
    monkeypatch.setenv("GITHUB_TOKEN", "t")
    return fake


def _posts(github) -> list:
    return [route for method, route in github.calls if method == "POST"]


def test_commit_files_through_the_api(github):
    head = github.head
    base = github.commits[head]["tree"]["sha"]
    uploaded = set()
    sha = gc.commit_files(gc.CARD_REPO, head, base, {"a.txt": "100644"}, {"a.txt": b"a2", "dir/c.txt": b"c"}, "msg", uploaded)
    assert sha == github.head and github.files_at(sha) == {"a.txt": b"a2", "dir/b.txt": b"b", "dir/c.txt": b"c"}
    info = github.commits[sha]
    assert info["message"] == "msg" and info["author"]["name"] == "AutoCroissant" and [p["sha"] for p in info["parents"]] == [head]
    assert uploaded == {_sha(b"a2"), _sha(b"c")} and _posts(github) == ["git/blobs", "git/blobs", "git/trees", "git/commits"]
    github.conflicts = 1                        # the branch moved: None, the caller retries from the new head; nothing re-uploaded
    assert gc.commit_files(gc.CARD_REPO, github.head, base, {}, {"a.txt": b"a2"}, "again", uploaded) is None
    assert _posts(github)[4:] == ["git/trees", "git/commits"]
    github.stuck = True                         # unmoved: a rule refused the update, not a race
    with pytest.raises(gc.Transient, match=r"GitHub 422: Repository rule violations found \(on PATCH"):
        gc.commit_files(gc.CARD_REPO, github.head, base, {}, {"a.txt": b"a2"}, "again", uploaded)


def test_upload_blob_no_write_and_rate_limit(github):
    github.rate_limited = True
    with pytest.raises(gc.Transient, match="rate limited"):
        gc.upload_blob(gc.CARD_REPO, b"x")
    assert gc.cant_write is False               # this run's, not the token's rights
    github.rate_limited = False
    github.forbidden = True
    assert gc.upload_blob(gc.CARD_REPO, b"x") is None and gc.cant_write is True
    github.forbidden = False
    assert gc.upload_blob(gc.CARD_REPO, b"x") == _sha(b"x")


def test_commit_files_falls_to_ssh(github, monkeypatch):
    github.forbidden = True
    seen = []
    monkeypatch.setattr(gc, "git_commit", lambda repository, head, files, message: seen.append((head, sorted(files), message)) or "c0ffee")
    assert gc.commit_files(gc.CARD_REPO, "h", "t", {}, {"a.txt": b"x"}, "m") == "c0ffee"
    assert seen == [("h", ["a.txt"], "m")] and gc.cant_write is True
    assert gc.commit_files(gc.CARD_REPO, "h", "t", {}, {"a.txt": b"x"}, "m2") == "c0ffee"    # cant_write: the API is not asked
    assert _posts(github) == ["git/blobs"]
    monkeypatch.setattr(gc, "cant_write", False)
    monkeypatch.delenv("GITHUB_TOKEN")
    assert gc.commit_files(gc.CARD_REPO, "h", "t", {}, {"a.txt": b"x"}, "m3") == "c0ffee"    # no token: SSH straight away
    assert _posts(github) == ["git/blobs"] and len(seen) == 3


def test_git_commit_push_outcomes(monkeypatch):
    calls = []

    def git(where, *args, stdin=None, env=None):
        calls.append(args[0])
        return {"hash-object": b"b1\n", "rev-parse": b"t0\n", "ls-tree": b"", "mktree": b"t1\n", "commit-tree": b"c1\n"}[args[0]]
    refreshed = []
    monkeypatch.setattr(gc, "git", git)
    monkeypatch.setattr(gc, "refresh", lambda: refreshed.append(1) or True)
    done = {}
    monkeypatch.setattr(gc, "run_process", lambda cmd, **kw: CompletedProcess(cmd, **done))
    monkeypatch.delenv("SSH_AUTH_SOCK", raising=False)
    done.update(returncode=1, stdout=b"! refs/heads/main:refs/heads/main [rejected] (fetch first)\n", stderr=b"")
    assert gc.git_commit(gc.CARD_REPO, "h", {"dir/x.psd": b"x"}, "m") is None       # not a fast forward: the caller retries
    assert calls == ["hash-object", "rev-parse", "ls-tree", "mktree", "mktree", "commit-tree"] and refreshed == [1]
    done.update(returncode=128, stdout=b"", stderr=b"git@github.com: Permission denied (publickey).\nfatal: Could not read from remote repository.\n")
    with pytest.raises(gc.Transient) as error:
        gc.git_commit(gc.CARD_REPO, "h", {"x": b"x"}, "m")
    assert str(error.value).startswith("git push failed: git@github.com: Permission denied (publickey). / fatal:")
    assert str(error.value).endswith("(no SSH_AUTH_SOCK: no SSH agent)")
    done.update(returncode=0, stdout=b"", stderr=b"")
    refreshed.clear()
    assert gc.git_commit(gc.CARD_REPO, "h", {"x": b"x"}, "m") == "c1" and refreshed == [1, 1]   # before and after the push
    with pytest.raises(gc.Transient, match="mirrors"):
        gc.git_commit("o/other", "h", {}, "m")

    def boom(where, *args, stdin=None, env=None):
        raise CalledProcessError(128, ["git", "-C", where, "hash-object"], stderr=b"fatal: bad object")
    monkeypatch.setattr(gc, "git", boom)
    with pytest.raises(gc.Transient, match="git hash-object in the history clone failed: fatal: bad object"):
        gc.git_commit(gc.CARD_REPO, "h", {"x": b"x"}, "m")

    def slow(where, *args, stdin=None, env=None):
        raise TimeoutExpired(["git", "-C", where, args[0]], gc.GIT_TIMEOUT)
    monkeypatch.setattr(gc, "git", slow)        # a timeout or no git is the runner's: Transient, never a verdict
    with pytest.raises(gc.Transient, match="git in the history clone failed: .* timed out after 300 seconds"):
        gc.git_commit(gc.CARD_REPO, "h", {"x": b"x"}, "m")
    monkeypatch.setattr(gc, "git", git)

    def no_push(cmd, **kw):
        raise FileNotFoundError(2, "No such file or directory", "git")
    monkeypatch.setattr(gc, "run_process", no_push)
    with pytest.raises(gc.Transient, match="git push failed: .*No such file or directory"):
        gc.git_commit(gc.CARD_REPO, "h", {"x": b"x"}, "m")
    monkeypatch.setattr(gc, "run_process", lambda cmd, **kw: (_ for _ in ()).throw(TimeoutExpired(cmd, gc.GIT_TIMEOUT)))
    with pytest.raises(gc.Transient, match="git push failed: .* timed out after 300 seconds"):
        gc.git_commit(gc.CARD_REPO, "h", {"x": b"x"}, "m")


def test_api_raises_transient_with_githubs_message(github):
    with pytest.raises(gc.Transient, match=r"GitHub 404: Not Found \(on GET git/commits/nope\)"):
        gc.api(gc.CARD_REPO, "GET", "git/commits/nope")
    assert gc.api(gc.CARD_REPO, "GET", "git/commits/nope", check=False).status_code == 404
    assert gc.ref_head(gc.CARD_REPO) == github.head and gc.commit_info(gc.CARD_REPO, github.head)["sha"] == github.head
    assert gc.plural(1, "card") == "1 card" and gc.plural(2, "card") == "2 cards"
