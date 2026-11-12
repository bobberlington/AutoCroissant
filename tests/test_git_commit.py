"""
git_commit, the commit path card_bake and card_guards share: upload_blob's NO_WRITE / rate-limit split, commit_files'
API-then-git order (a blob too large for the API too), push_target against the card_bake fake GitHub, and git_commit's push outcomes with git faked (no clone, no network) and its delta push.
"""
from io import StringIO
from subprocess import CalledProcessError, CompletedProcess, TimeoutExpired, run as run_process

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


def test_commit_files_too_large_goes_through_git(github, monkeypatch):
    """A blob past the API's size limit (Quote.psd, 73 MB, 10-31): git takes the whole commit; the API stays usable."""
    github.max_blob = 2
    seen = []
    monkeypatch.setattr(gc, "git_commit", lambda repository, head, files, message: seen.append(sorted(files)) or "c0ffee")
    assert gc.commit_files(gc.CARD_REPO, "h", "t", {}, {"a.txt": b"x", "big.psd": b"xyz"}, "m") == "c0ffee"
    assert seen == [["a.txt", "big.psd"]] and gc.cant_write is False


def test_push_target(monkeypatch):
    """SSH with an agent or without a token; else HTTPS, the token in env config only."""
    monkeypatch.setattr(ga, "GIT_TOKEN", "")
    monkeypatch.setenv("GITHUB_TOKEN", "s3cret")
    monkeypatch.setenv("SSH_AUTH_SOCK", "/tmp/agent")
    assert gc.push_target(gc.CARD_REPO) == (f"git@github.com:{gc.CARD_REPO}.git", {})
    monkeypatch.delenv("SSH_AUTH_SOCK")
    url, env = gc.push_target(gc.CARD_REPO)
    assert url == f"https://github.com/{gc.CARD_REPO}.git" and "s3cret" not in url and env["GIT_CONFIG_KEY_0"] == "http.https://github.com/.extraheader"
    monkeypatch.delenv("GITHUB_TOKEN")
    assert gc.push_target(gc.CARD_REPO)[0].startswith("git@github.com:")


def test_git_commit_push_outcomes(monkeypatch):
    calls = []
    monkeypatch.setattr(ga, "GIT_TOKEN", "")
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)       # no token: SSH

    def git(where, *args, stdin=None, env=None):
        calls.append(args[0])
        if args[0] == "hash-object":            # stored: the push zlibs the blob once
            assert (env["GIT_CONFIG_KEY_0"], env["GIT_CONFIG_VALUE_0"]) == ("core.loosecompression", "0")
        return {"hash-object": b"b1\n", "rev-parse": b"t0\n", "ls-tree": b"", "mktree": b"t1\n", "commit-tree": b"c1\n"}[args[0]]
    refreshed = []
    monkeypatch.setattr(gc, "git", git)
    monkeypatch.setattr(gc, "refresh", lambda: refreshed.append(1) or True)
    done = {}
    pushes = []
    monkeypatch.setattr(gc, "run_process", lambda cmd, **kw: pushes.append(cmd) or CompletedProcess(cmd, **done))
    monkeypatch.delenv("SSH_AUTH_SOCK", raising=False)
    done.update(returncode=1, stdout=b"! refs/heads/main:refs/heads/main [rejected] (fetch first)\n", stderr=b"")
    assert gc.git_commit(gc.CARD_REPO, "h", {"dir/x.psd": b"x"}, "m") is None       # not a fast forward: the caller retries
    assert calls == ["hash-object", "rev-parse", "ls-tree", "mktree", "mktree", "commit-tree"] and refreshed == [1]
    assert pushes[0][3:9] == ["-c", "pack.window=0", "-c", "pack.compression=1", "push", "--porcelain"]
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


def test_git_commit_deltas_against_old_blobs_it_holds(monkeypatch, tmp_path):
    """Head's blobs at the written paths all in the clone (its alternates' too: the guards' checkout): a delta push with
    no lazy fetch, then on a failure other than a rejection the push as before; one missing: only the push as before.
    Real git, the push faked."""
    clone, other = tmp_path / "clone.git", tmp_path / "other.git"
    for repo in (clone, other):
        run_process(["git", "init", "-q", "--bare", str(repo)], check=True)
    old = gc.git(str(other), "hash-object", "-w", "--stdin", stdin=b"old").decode().strip()      # only in other
    sub = gc.git(str(clone), "mktree", "--missing", stdin=f"100644 blob {old}\tx.psd\n".encode()).decode().strip()
    tree = gc.git(str(clone), "mktree", stdin=f"040000 tree {sub}\tdir\n".encode()).decode().strip()
    head = gc.git(str(clone), "commit-tree", tree, "-m", "base", env=gc.git_env()).decode().strip()
    monkeypatch.setattr(gc, "HISTORY_CLONE", str(clone))
    monkeypatch.setattr(gc, "refresh", lambda: True)
    pushes, replies = [], []

    def fake_run(cmd, **kwargs):
        if "push" not in cmd:
            return run_process(cmd, **kwargs)
        delta = "--no-thin" not in cmd
        pushes.append(("delta", kwargs["env"].get("GIT_NO_LAZY_FETCH")) if delta else ("plain", None))
        return CompletedProcess(cmd, *replies.pop(0))
    monkeypatch.setattr(gc, "run_process", fake_run)
    monkeypatch.setattr(gc, "stderr", err := StringIO())
    files = {"dir/x.psd": b"new", "dir/y.png": b"y"}
    replies[:] = [(0, b"", b"")]
    assert gc.git_commit(gc.CARD_REPO, head, files, "m") and pushes == [("plain", None)]   # the old blob missing
    assert not gc.has_blobs([]) and not gc.has_blobs([old], str(tmp_path / "none"))
    (clone / "objects" / "info" / "alternates").write_text(f"{other / 'objects'}\n")
    assert gc.has_blobs([old]) and not gc.has_blobs([old, "1" * 40])
    pushes.clear()
    replies[:] = [(0, b"", b"")]
    assert gc.git_commit(gc.CARD_REPO, head, files, "m") and pushes == [("delta", "1")]
    pushes.clear()
    replies[:] = [(1, b"", b"fatal: could not fetch 8aa81b20 from promisor remote\n"), (0, b"", b"")]
    assert gc.git_commit(gc.CARD_REPO, head, files, "m") and pushes == [("delta", "1"), ("plain", None)]
    assert "git delta push failed, again without: fatal: could not fetch 8aa81b20" in err.getvalue()
    pushes.clear()
    replies[:] = [(1, b"!\trefs/heads/main:refs/heads/main\t[rejected] (fetch first)\n", b"")]
    assert gc.git_commit(gc.CARD_REPO, head, files, "m") is None and pushes == [("delta", "1")]   # the branch moved
