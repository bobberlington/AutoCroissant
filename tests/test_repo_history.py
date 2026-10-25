"""repo_history._walk on a tiny temp git repo: merge, revert, rename, incremental walks."""
import os
import subprocess

import pytest

import commands.repo_history as rh

ENV = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
       "GIT_COMMITTER_EMAIL": "t@t", "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"}


@pytest.fixture
def repo(tmp_path, monkeypatch):
    path = tmp_path / "history"
    path.mkdir()
    monkeypatch.setattr(rh, "HISTORY_CLONE", str(path))
    env = {**ENV, "HOME": str(tmp_path), "XDG_CONFIG_HOME": str(tmp_path)}   # git 2.25 ignores GIT_CONFIG_GLOBAL

    def git(*args) -> str:
        return subprocess.run(["git", "-C", str(path), *args], capture_output=True, encoding="utf-8",
                              check=True, env=env).stdout.strip()

    def commit(message, **files) -> str:
        for name, text in files.items():
            if text is None:
                git("rm", "-q", name)
            else:
                (path / name).write_text(text)
                git("add", name)
        git("commit", "-q", "-m", message)
        return git("rev-parse", "HEAD")

    git("init", "-q")
    git("symbolic-ref", "HEAD", "refs/heads/main")      # git 2.25 has no init -b
    git.commit = commit
    return git


def _blob(git, commit, path):
    return git("rev-parse", f"{commit}:{path}")


def test_merge_revert_rename(repo):
    git = repo
    c1 = repo.commit("one", **{"a.txt": "a1", "b.txt": "b1"})
    git("checkout", "-q", "-b", "side")
    c2 = repo.commit("side edits a", **{"a.txt": "a2"})
    git("checkout", "-q", "main")
    c3 = repo.commit("main edits b", **{"b.txt": "b2"})
    git("merge", "-q", "--no-ff", "-m", "merge", "side")
    c4 = git("rev-parse", "HEAD")
    git("revert", "--no-edit", c2)                                        # a.txt back to a1's blob
    c5 = git("rev-parse", "HEAD")
    git("mv", "b.txt", "c.txt")
    git("commit", "-q", "-m", "rename")
    c6 = git("rev-parse", "HEAD")

    at4 = rh._walk(c4, "", {})
    assert at4 == {"a.txt": (_blob(git, c4, "a.txt"), c2),                # from the merge's second parent
                   "b.txt": (_blob(git, c4, "b.txt"), c3)}
    at6 = rh._walk(c6, "", {})
    assert at6 == {"a.txt": (_blob(git, c1, "a.txt"), c5),                # a revert is a change, even to an old blob
                   "c.txt": (_blob(git, c3, "b.txt"), c6)}                # a rename is new at the rename
    assert rh._walk(c6, c4, at4) == at6                                   # incremental walk = full walk
    assert rh._walk(c4, c3, rh._walk(c3, "", {})) == at4                 # side forked before last: no full walk
    assert rh._walk(c6, c6, at6) is at6                                   # nothing new


def test_raw_url_pins_only_a_matching_blob(monkeypatch):
    monkeypatch.setattr(rh, "_paths", {"Items/A b.png": ("blob1", "c0ffee")})
    base = f"https://raw.githubusercontent.com/{rh.REPOSITORY}/"
    assert rh.raw_url(rh.REPOSITORY, "Items/A b.png", "blob1") == base + "c0ffee/Items/A%20b.png"
    assert rh.raw_url(rh.REPOSITORY, "Items/A b.png", "blob2") == base + "main/Items/A%20b.png"
    assert rh.raw_url("someone/else", "Items/A b.png", "blob1").endswith("/main/Items/A%20b.png")


def _ls_tree_walk(head, last, known):
    """The pre-diff-tree _walk (one ls-tree per commit): the reference the single pass must match."""
    rows = [line.split() for line in rh._git("rev-list", "--parents", "--topo-order", "--reverse",
                                             f"{last}..{head}" if last else head).splitlines()]
    if not rows:
        return known if head == last else None
    new = {row[0] for row in rows}
    maps = {last: known} if last else {}
    for row in rows:
        for parent in row[1:]:
            if parent not in new and parent not in maps:
                maps[parent] = {path: known[path] if path in known and known[path][0] == blob else (blob, parent)
                                for path, blob in rh._tree(parent).items()}
    for commit, *parents in rows:
        olds = [maps[p] for p in parents]
        maps[commit] = {path: next(((blob, old[path][1]) for old in olds if path in old and old[path][0] == blob),
                                   (blob, commit)) for path, blob in rh._tree(commit).items()}
    return maps[head]


def test_single_pass_matches_ls_tree_walk(repo, tmp_path, monkeypatch):
    git = repo
    os.mkdir(os.path.join(rh.HISTORY_CLONE, "d"))
    commits = [repo.commit("root", **{"a.txt": "a1", "b.txt": "b1", "d/e.txt": "e1"})]
    git("checkout", "-q", "-b", "side")
    commits.append(repo.commit("side", **{"a.txt": "a2", "d/f.txt": "f1"}))
    git("checkout", "-q", "main")
    commits.append(repo.commit("main", **{"b.txt": "b2", "d/e.txt": None}))
    git("merge", "-q", "--no-ff", "-m", "merge", "side")
    commits.append(git("rev-parse", "HEAD"))
    git("update-index", "--add", "--cacheinfo", f"160000,{commits[0]},sub")            # a submodule: no blob
    git("update-index", "--chmod=+x", "b.txt")                                         # mode only: same blob
    git("commit", "-q", "-m", "sub, mode")
    commits.append(git("rev-parse", "HEAD"))
    git("checkout", "-q", "--orphan", "other")                                         # a second root
    git("read-tree", "--empty")
    commits.append(repo.commit("root 2", **{"a.txt": "a1", "g.txt": "g1"}))
    git("checkout", "-q", "-f", "main")
    git("merge", "-q", "--allow-unrelated-histories", "--no-ff", "-X", "ours", "-m", "join", "other")
    commits.append(git("rev-parse", "HEAD"))
    git("merge", "-q", "-s", "ours", "-m", "empty", commits[1])                        # no diff to the first parent
    git("revert", "--no-edit", commits[1])
    commits.append(git("rev-parse", "HEAD"))
    git("checkout", "-q", "-b", "o1", commits[2])
    o1 = repo.commit("o1", **{"h.txt": "h1"})
    git("checkout", "-q", "-b", "o2", commits[2])
    o2 = repo.commit("o2", **{"h.txt": "h1", "i.txt": "i1"})
    git("checkout", "-q", "main")
    git("merge", "-q", "--no-ff", "-m", "octopus", o1, o2)
    commits.append(git("rev-parse", "HEAD"))

    clone = tmp_path / "blobless.git"                                                  # as the bot's: no blobs
    git("config", "uploadpack.allowFilter", "true")
    subprocess.run(["git", "clone", "-q", "--bare", "--filter=blob:none", f"file://{rh.HISTORY_CLONE}", str(clone)],
                   check=True, capture_output=True, env=ENV)
    monkeypatch.setattr(rh, "HISTORY_CLONE", str(clone))
    monkeypatch.setenv("GIT_NO_LAZY_FETCH", "1")
    full = {c: _ls_tree_walk(c, "", {}) for c in commits}
    for i, head in enumerate(commits):
        assert rh._walk(head, "", {}) == full[head]
        for last in commits[:i]:
            if rh._is_ancestor(last, head):
                assert rh._walk(head, last, full[last]) == _ls_tree_walk(head, last, full[last]) == full[head]
