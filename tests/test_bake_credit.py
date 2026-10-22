"""
card_bake's bake credit: a bake commit names every editor whose fields it prints, one "Edit-by: <login>" trailer per
distinct login after the unchanged subject (TTSCardMaker build_site.EDIT_BY reads them), over the API path and the SSH
path (git_commit with real git objects in a temp clone; the push is faked). apply_jobs is faked: no fork needed.
"""
from io import BytesIO
from subprocess import CompletedProcess, run as run_process
from types import SimpleNamespace

import pytest
from PIL import Image

pytest.importorskip("requests", reason="card_bake imports requests (absent from this Python)")
import commands.card_bake as cb  # noqa: E402
import commands.git_commit as gc  # noqa: E402
import commands.psd_edit as pe  # noqa: E402
from test_card_bake import AT, DOOMER, _edits, files, github  # noqa: E402,F401  (fixtures)

SUBJECT = "Card edit: Mini Doomer hp 5 -> 8"


def _card(jobs, by, redundant=()):
    return cb.Card("Mini Doomer", "mini doomer", dict(jobs), jobs=dict(jobs), redundant=list(redundant), by=by)


def _trailers(message: str) -> list:
    """What git itself parses as the message's Edit-by trailers."""
    out = run_process(["git", "interpret-trailers", "--parse"], input=message, capture_output=True, text=True, check=True)
    return [line.split(": ", 1)[1] for line in out.stdout.splitlines() if line.startswith(f"{cb.EDIT_BY}: ")]


def test_editors_sorted_distinct_and_only_for_baked_fields():
    assert cb.editors(_card({"hp": 8}, "web")) == ["web"]
    assert cb.editors(_card({"hp": 8, "atk": 2, "def": 1}, {"hp": "zed", "atk": "amy", "def": "zed", "spd": "nope"})) == ["amy", "zed"]
    assert cb.editors(_card({"hp": 8}, None)) == []                     # unknown: no trailer
    assert cb.editors(_card({"hp": 8}, " ")) == [] and cb.editors(_card({"hp": 8}, "two words")) == []
    assert cb.editors(_card({"hp": 8}, ["web"])) == [] and cb.editors(_card({"hp": 8}, {"hp": 3})) == []
    assert cb.editors(_card({}, "web", redundant=["hp"])) == []        # nothing printed: no credit


def test_message_keeps_the_subject_and_git_parses_the_trailers():
    card = _card({"hp": 8}, {"hp": "zed", "atk": "amy"})
    card.jobs, card.fields = {"hp": 8, "atk": 2}, {"hp": 8, "atk": 2}
    message = cb._message(card)
    assert message.splitlines()[0] == cb._subject(card) and message.split("\n\n", 1)[1] == "Edit-by: amy\nEdit-by: zed"
    assert _trailers(message) == ["amy", "zed"]
    assert cb._message(_card({"hp": 8}, None)) == cb._subject(_card({"hp": 8}, None))    # no editor: the subject alone


def test_select_carries_the_entrys_by(github):
    info = github.commits[github.head]
    snapshot = cb.Snapshot(cb.CARD_REPO, github.head, info, cb._tree(cb.CARD_REPO, info["tree"]["sha"]))
    edits = cb.CardEdits.parse(_edits({"Mini Doomer": {"hp": 8, "_by": "amy", "_at": AT}, "Nobody": {"hp": 3}}).decode())
    assert [(c.name, c.by) for c in cb.select(edits, snapshot)] == [("Mini Doomer", "amy"), ("Nobody", None)]


@pytest.fixture
def baking(github, monkeypatch):
    """Mini Doomer hp 5 -> 8 asked by "amy"; apply_jobs writes stand-in PSD/PNG bytes (the commit message is under test)."""
    edits = github.files_at(github.head)
    edits[cb.EDITS_PATHS[0]] = _edits({"Mini Doomer": {"hp": 8, "_by": "amy", "_at": AT}})
    github.commit(edits, "web", "Website edit: Mini Doomer")

    def apply_jobs(old_psd, old_png, rel, jobs, parsed, parser, new_dir, *args, **kwargs):
        paths, png = {}, BytesIO()
        Image.new("RGBA", (1, 1)).save(png, "PNG")
        for ext, data in (("psd", b"new psd"), ("png", png.getvalue())):
            paths[ext] = f"{new_dir}/new.{ext}"
            with open(paths[ext], "wb") as fh:
                fh.write(data)
        return SimpleNamespace(psd_path=paths["psd"], png_path=paths["png"], edits=[SimpleNamespace(field="hp", old=5)],
                               region=None, reparse={"card": {}, "validator": []}, fx=[], parity={}, flag_bytes=0,
                               changed_pixels=0)
    monkeypatch.setattr(pe, "apply_jobs", apply_jobs)
    monkeypatch.setattr(pe, "artifacts", lambda *args: {})
    return github


def test_api_commit_carries_the_trailer(baking):
    text, commits = cb.run(card="Mini Doomer")
    assert len(commits) == 1, text
    message = baking.commits[commits[0]]["message"]
    assert message == f"{SUBJECT}\n\nEdit-by: amy" and _trailers(message) == ["amy"]
    assert baking.files_at(commits[0])[DOOMER] == b"new psd"


def test_ssh_commit_carries_the_trailer(baking, monkeypatch, tmp_path):
    """The SSH path for real: objects and commit-tree in a temp clone (the push faked); git keeps the body whole."""
    clone = tmp_path / "clone"
    run_process(["git", "init", "-q", str(clone)], check=True)
    blob = gc.git(str(clone), "hash-object", "-w", "--stdin", stdin=b"x").decode().strip()
    tree = gc.git(str(clone), "mktree", stdin=f"100644 blob {blob}\tx\n".encode()).decode().strip()
    head = gc.git(str(clone), "commit-tree", tree, "-m", "base", env=gc.git_env()).decode().strip()
    monkeypatch.setattr(gc, "HISTORY_CLONE", str(clone))
    pushed = []

    def fake_run(cmd, **kwargs):
        if "push" in cmd:
            pushed.append(cmd[len(cmd) - 1])
            return CompletedProcess(cmd, 0, b"", b"")
        return run_process(cmd, **kwargs)
    monkeypatch.setattr(gc, "run_process", fake_run)
    sha = gc.git_commit(gc.CARD_REPO, head, {"dir/y.txt": b"y"}, f"{SUBJECT}\n\nEdit-by: amy\nEdit-by: zed")
    assert pushed == [f"{sha}:refs/heads/{cb.CARD_BRANCH}"]
    raw = gc.git(str(clone), "cat-file", "commit", sha).decode()
    assert raw.split("\n\n", 1)[1] == f"{SUBJECT}\n\nEdit-by: amy\nEdit-by: zed\n"       # as written, plus git's final newline
    trailers = gc.git(str(clone), "log", "-1", f"--format=%(trailers:key={cb.EDIT_BY},valueonly,separator=%x1f)", sha)
    assert trailers.decode().strip().split("\x1f") == ["amy", "zed"]      # build_site's HISTORY_FORMAT reads them so

    baking.forbidden = True                                             # and card_bake hands git_commit the whole message
    seen = []
    monkeypatch.setattr(gc, "git_commit", lambda repository, head, files, message: seen.append(message)
                        or baking.commit({**baking.files_at(head), **files}, "AutoCroissant", message))
    text, commits = cb.run(card="Mini Doomer")
    assert len(commits) == 1 and seen == [f"{SUBJECT}\n\nEdit-by: amy"], text
