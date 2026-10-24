"""
card_preview (EDIT_PIPELINE_DESIGN section 16): one card's website edit through the bake on a temp copy, against a local
card repo made from the clone's Mini Doomer (the --clone path; the card repo is never written), and its publication on
the previews branch of a bare repo standing in for GitHub, through a blobless depth-1 clone (the Action's checkout): a
baked stat edit (the same pixels as card_bake's dry run), a refused edit (with and without a render), a redundant edit,
a stars edit refused beside a baked stat, the creatures and duplicate gates, bad inputs, a runner error; one commit per
publish, a re-run, the rewrites by age and count, a moved head and a moved lease. Then card-preview.yml: its shape, no
push trigger elsewhere outside main, the Action-shaped copy of SPARSE_FILES, and its Validate, Preview and Publish steps
run as the runner would (bash) against those repos. The workflows come from $AC_SITE_SRC (a TTSCardMaker checkout) or
the clone; a real bake needs the fork's renderer:
    AUTOCROISSANT_LOCAL_DIR=~/Desktop/TTSCardMaker AC_SITE_SRC=<TTSCardMaker checkout> <fork python> -m pytest -q tests/test_card_preview.py
"""
import json
import re
import shutil
from datetime import datetime, timedelta, timezone
from os import environ
from os.path import expanduser
from pathlib import Path
from subprocess import run as run_process
from sys import executable

import pytest

pytest.importorskip("requests", reason="card_bake imports requests (absent from this Python)")
import commands.card_bake as cb  # noqa: E402
import commands.card_preview as cp  # noqa: E402
import commands.psd_edit as pe  # noqa: E402
from conftest import CLONE, REPO, blob_sha_bytes as _sha, fork_reason, text_engine as _text_engine  # noqa: E402
from golden_cards import VOCAB  # noqa: E402
from test_card_bake import ALLOWED_IMPORTS, third_party_imports  # noqa: E402

DOOMER = "Auxiliary/Minions/Mini_Doomer.psd"
DOOMER_PNG = "Auxiliary/Minions/Mini_Doomer.png"
SNACK = "Items/Heal/2 Stars/A_Snack.psd"
WORKFLOWS = Path(expanduser(environ.get("AC_SITE_SRC") or str(CLONE))) / ".github" / "workflows"
REQUEST = "0f8e7d6c-5b4a-4321-9876-0123456789ab"       # a crypto.randomUUID()

fork = pytest.mark.skipif(not _text_engine(), reason="a bake needs the fork's renderer: " + fork_reason())


# ========================
# Fixtures
# ========================
def _git(where, *args, env=None) -> str:
    done = run_process(["git", "-C", str(where), *args], capture_output=True, text=True, timeout=120,
                       env={**environ, "GIT_CONFIG_GLOBAL": "/dev/null", **(env or {})})
    assert done.returncode == 0, done.stderr
    return done.stdout


def _commit(repo, message="Website edit: test") -> str:
    who = {"GIT_AUTHOR_NAME": "web", "GIT_AUTHOR_EMAIL": "w@x", "GIT_COMMITTER_NAME": "web", "GIT_COMMITTER_EMAIL": "w@x"}
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", message, env=who)
    return _git(repo, "rev-parse", "HEAD").strip()


@pytest.fixture
def cards(tmp_path, clone, monkeypatch):
    """A card repo with Mini Doomer, A Snack (Doomer's PSD) and its numbered A_Snack2.png, Types/ and the manifest."""
    for rel in (DOOMER, DOOMER_PNG):
        if not (clone / rel).exists():
            pytest.skip(f"{rel} not in the clone")
    vocab = json.loads(VOCAB.read_text(encoding="utf-8"))
    files = {f"Types/{t}.png": t.encode() for t in vocab["all_types"]}
    files.update({f"Types/Stars/{s}.png": s.encode() for s in vocab["stars"]})
    files[cb.ICON_MANIFEST_PATH] = json.dumps(vocab["manifest"]).encode()
    files[DOOMER], files[DOOMER_PNG] = (clone / DOOMER).read_bytes(), (clone / DOOMER_PNG).read_bytes()
    files[SNACK], files[SNACK[:len(SNACK) - 4] + ".png"] = files[DOOMER], files[DOOMER_PNG]
    files["Items/Heal/2 Stars/A_Snack2.png"] = b"a duplicate"
    files[".github/workflows/x.yml"] = b"name: x\n"
    repo = tmp_path / "cards"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    for path, data in files.items():
        (repo / path).parent.mkdir(parents=True, exist_ok=True)
        (repo / path).write_bytes(data)
    head = _commit(repo)
    monkeypatch.setenv(cb.FONT_DIRS_ENV, str(clone / pe.FONTS_DIR))
    monkeypatch.setenv(cb.WORK_ENV, str(tmp_path / "runner"))
    (tmp_path / "runner").mkdir()
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_RUN_ID", raising=False)
    return type("Cards", (), {"repo": repo, "head": head, "files": files})


def _state(repo) -> tuple:
    """What a preview must leave as it was: HEAD, every ref, the index and worktree status, the object count."""
    return (_git(repo, "rev-parse", "HEAD"), _git(repo, "for-each-ref"), _git(repo, "status", "--porcelain", "--ignored"),
            _git(repo, "count-objects", "-v"))


def _preview(cards, tmp_path, edit, card="Mini Doomer", request=REQUEST, name="out") -> tuple:
    edit_file = tmp_path / f"{name}.json"
    edit_file.write_bytes(edit if isinstance(edit, bytes) else json.dumps(edit).encode())
    out = tmp_path / name
    code = cp.main(["--card", card, "--edit", str(edit_file), "--out", str(out), "--clone", str(cards.repo),
                    "--request", request])
    return code, json.loads((out / cp.RESULT).read_text(encoding="utf-8")), out


# ========================
# The preview
# ========================
@fork
def test_a_stat_edit_bakes_the_bakes_pixels(cards, tmp_path, monkeypatch):
    """hp 5 -> 8: baked, the PNG the bake's dry run makes (the same pixels), recompressed losslessly, a 720 px WebP; the
    base is the commit and the blobs it read; the card repo is untouched and nothing is left in the runner's temp."""
    from PIL import Image
    before = _state(cards.repo)
    code, result, out = _preview(cards, tmp_path, {"hp": 8, "note": "checked", "_by": "web"})
    assert code == 0 and _state(cards.repo) == before
    assert list((tmp_path / "runner").iterdir()) == []                         # the work dir is removed
    assert (result["state"], result["reason"], result["fields"], result["transient"]) == ("baked", "hp 5 -> 8", {"hp": "baked"}, False)
    assert result["request"] == REQUEST and result["card"] == "Mini Doomer" and result["path"] == DOOMER and result["png"] == DOOMER_PNG
    assert result["base"] == {"repo": cb.CARD_REPO, "commit": cards.head, "psd_blob": _sha(cards.files[DOOMER]),
                              "png_blob": _sha(cards.files[DOOMER_PNG])}
    assert result["edit"] == {"hp": 8} and result["changes"] == [{"field": "hp", "old": 5, "new": 8}]
    assert result["image"] == "baked" and result["files"] == {"png": cp.PNG, "webp": cp.WEBP} and result["parity"]["hp"]["bad_pct"] < 1
    assert len(result["region"]) == 4 and result["runner"]["text_engine"] is True and result["at"].endswith("Z")
    webp = Image.open(out / cp.WEBP)
    assert webp.format == "WEBP" and webp.width == cp.WEBP_WIDTH and list(webp.size) == result["webp_size"]

    # the bake's own dry run of the same edit: the same merged pixels
    (cards.repo / cb.EDITS_PATHS[0]).parent.mkdir(parents=True, exist_ok=True)
    (cards.repo / cb.EDITS_PATHS[0]).write_bytes(cb.dump_edits({"cards": {"Mini Doomer": {"hp": 8}}}).encode())
    _commit(cards.repo, "Website edit: Mini Doomer")
    monkeypatch.setattr(cb, "recent_paths", lambda *a, **k: set())
    report = cb.run_report(dry_run=True, card="Mini Doomer", local=str(cards.repo))
    assert [c.name for c in report.cards if c.baked] == ["Mini Doomer"]
    bake = Image.open(report.cards[0].baked.png_path)
    ours = Image.open(out / cp.PNG)
    assert ours.mode == bake.mode and ours.tobytes() == bake.tobytes()
    if result["recompressed"]:
        assert (out / cp.PNG).stat().st_size < Path(report.cards[0].baked.png_path).stat().st_size


def test_a_refused_edit_writes_its_reason_and_no_image(cards, tmp_path):
    code, result, out = _preview(cards, tmp_path, {"hp": 25})
    assert code == 0 and (result["state"], result["reason"]) == ("refused", "stat>10: hp 25: above 20")
    assert result["fields"] == {"hp": "refused"} and result["image"] is None and result["files"] == {}
    assert not (out / cp.PNG).exists() and not (out / cp.WEBP).exists()
    assert result["base"]["psd_blob"] == _sha(cards.files[DOOMER])


@fork
def test_a_refusal_after_the_render_keeps_what_it_got(cards, tmp_path, monkeypatch):
    """A gate past the save (here the idempotence re-render): refused, and the image of the render it refused."""
    def refuse(*args, **kwargs):
        raise pe.Refused("idempotence", "hp re-render bad 31.00% mean 40.00 max 255")
    monkeypatch.setattr(pe, "gate_idempotence", refuse)
    code, result, out = _preview(cards, tmp_path, {"hp": 8})
    assert code == 0 and result["state"] == "refused" and result["reason"].startswith("idempotence: hp re-render bad 31.00%")
    assert result["image"] == "refused" and (out / cp.PNG).exists() and (out / cp.WEBP).exists()


def test_a_redundant_edit(cards, tmp_path):
    code, result, out = _preview(cards, tmp_path, {"hp": 5}, card=DOOMER)             # by its PSD path
    assert code == 0 and (result["state"], result["reason"], result["fields"]) == ("redundant", "already on the card: hp", {"hp": "redundant"})
    assert result["notes"] == [] and result["image"] is None and not (out / cp.WEBP).exists()


@fork
def test_a_refused_stars_edit_beside_a_baked_stat(cards, tmp_path):
    code, result, _ = _preview(cards, tmp_path, {"stars": 0, "hp": 8})
    assert code == 0 and result["state"] == "baked" and result["fields"] == {"stars": "refused", "hp": "baked"}
    assert "stars not baked: unsupported-card: stars: a card shows at least one" in result["notes"]


def test_the_bakes_gates_hold(cards, tmp_path):
    """The creatures list sets the stats; a numbered PNG's card; a card with no PSD: refused as the bake refuses them."""
    code, result, _ = _preview(cards, tmp_path, {"hp": 8, "creatures": [{"hp": 8}]}, name="creatures")
    assert result["state"] == "refused" and "the entry's creatures list sets the stats" in result["reason"]
    assert result["edit"] == {"hp": 8, "creatures": True}
    _, result, _ = _preview(cards, tmp_path, {"ability": "Breakfast."}, card="A Snack2", name="dup")
    assert result["state"] == "refused" and result["reason"] == f"duplicate: a numbered PNG exported from `{SNACK}`"
    assert result["path"] == "" and result["base"]["psd_blob"] == "" and result["base"]["commit"] == cards.head
    _, result, _ = _preview(cards, tmp_path, {"hp": 3}, card="Nobody", name="nobody")
    assert (result["state"], result["reason"]) == ("refused", "unsupported-card: no PSD in the card repo")


@pytest.mark.parametrize("card, edit, client_id, why", [
    ("Mini Doomer", b"{not json", REQUEST, "input: the edit is not JSON"),
    ("Mini Doomer", b"[1]", REQUEST, "input: the edit is not a JSON object"),
    ("Mini Doomer", b'{"hp": 8, "colour": "red"}', REQUEST, "input: unknown field: colour"),
    ("Mini Doomer", b'{"hp": "8"}', REQUEST, "input: not a valid hp: '8'"),
    ("Mini Doomer", b'{"hp": true}', REQUEST, "input: not a valid hp: True"),
    ("Mini Doomer", b'{"note": "x", "hp": null}', REQUEST, "input: no PSD-kind field with a value"),
    ("Mini Doomer", b'{"ability": "' + b"x" * cp.MAX_EDIT + b'"}', REQUEST, f"input: the edit is over {cp.MAX_EDIT} bytes"),
    ("Mini Doomer", b'{"name": "\xff"}', REQUEST, "input: the edit is not UTF-8"),
    ("Mini Doomer", b'{"hp": 8}', "short", "input: the request must be 8-64"),
    ("Mini Doomer", b'{"hp": 8}', "has a space in it", "input: the request must be 8-64"),
    ("Mini\nDoomer", b'{"hp": 8}', REQUEST, "input: the card must be 1-200 characters"),
    ("Creatures/Nope.psd", b'{"hp": 8}', REQUEST, "input: no PSD at 'Creatures/Nope.psd'"),
])
def test_bad_input_is_an_error_to_fix_not_to_retry(cards, tmp_path, card, edit, client_id, why):
    code, result, out = _preview(cards, tmp_path, edit, card=card, request=client_id)
    assert code == 2 and result["state"] == "error" and result["transient"] is False and result["reason"].startswith(why)
    assert result["request"] == (client_id if cp.REQUEST.fullmatch(client_id) else None) and result["image"] is None


def test_a_runner_error_is_a_try_again(cards, tmp_path, monkeypatch):
    monkeypatch.setattr(cp, "local_snapshot", lambda *a: (_ for _ in ()).throw(cb.Transient("git ls-tree failed")))
    code, result, _ = _preview(cards, tmp_path, {"hp": 8})
    assert code == 1 and (result["state"], result["transient"], result["reason"]) == ("error", True, "error: git ls-tree failed")
    monkeypatch.setattr(cp, "local_snapshot", lambda *a: (_ for _ in ()).throw(KeyError("tree")))
    code, result, _ = _preview(cards, tmp_path, {"hp": 8})                      # a bug: an error, not the input's
    assert (code, result["state"], result["transient"]) == (1, "error", False)
    assert result["reason"] == "error: KeyError: 'tree'" and result["traceback"]


def test_an_old_result_never_survives_a_new_run(cards, tmp_path):
    _, _, out = _preview(cards, tmp_path, {"hp": 25})
    (out / cp.WEBP).write_bytes(b"old")
    _, result, _ = _preview(cards, tmp_path, b"{bad")
    assert result["state"] == "error" and not (out / cp.WEBP).exists()


# ========================
# The previews branch
# ========================
@pytest.fixture
def remote(tmp_path):
    """A bare repo standing in for GitHub (filters and lazy fetches allowed) with main, and the Action's checkout of it:
    blobless and depth 1."""
    seed = tmp_path / "seed"
    seed.mkdir()
    _git(seed, "init", "-q", "-b", "main")
    (seed / "README.md").write_text("cards\n")
    _commit(seed, "seed")
    bare = tmp_path / "origin.git"
    _git(tmp_path, "clone", "-q", "--bare", str(seed), str(bare))
    _git(bare, "config", "uploadpack.allowFilter", "true")
    _git(bare, "config", "uploadpack.allowAnySHA1InWant", "true")
    work = tmp_path / "workspace"
    _git(tmp_path, "clone", "-q", "--depth", "1", "--filter=blob:none", f"file://{bare}", str(work))
    return type("Remote", (), {"bare": bare, "work": work})


def _out(tmp_path, request, state="baked", at=None, image=True, card="Mini Doomer") -> Path:
    out = tmp_path / "outs" / request
    out.mkdir(parents=True)
    result = {**cp.blank(request, card), "state": state, "path": DOOMER, "at": at or cp._now(), "reason": "hp 5 -> 8"}
    if image:
        result["files"] = {"png": cp.PNG, "webp": cp.WEBP}
        (out / cp.PNG).write_bytes(b"png of " + request.encode())
        (out / cp.WEBP).write_bytes(b"webp of " + request.encode())
    (out / cp.RESULT).write_text(json.dumps(result))
    return out


def _branch(bare) -> dict:
    """{path: bytes} on previews."""
    out = {}
    for line in _git(bare, "ls-tree", "-r", "previews").splitlines():
        meta, path = line.split("\t", 1)
        out[path] = run_process(["git", "-C", str(bare), "cat-file", "blob", meta.split()[2]], capture_output=True).stdout
    return out


def _index(bare) -> dict:
    return json.loads(_branch(bare)[cp.INDEX])["previews"]


def test_publish_one_commit_per_preview(remote, tmp_path):
    a, b = "aaaaaaaa-0001", "bbbbbbbb-0002"
    first = cp.publish(str(_out(tmp_path, a)), a, str(remote.work), png=True)
    assert first["parent"] is None and not first["rewritten"]
    files = _branch(remote.bare)
    assert sorted(files) == [cp.README, cp.INDEX, f"previews/{a}/{cp.PNG}", f"previews/{a}/{cp.WEBP}", f"previews/{a}/{cp.RESULT}"]
    assert json.loads(files[f"previews/{a}/{cp.RESULT}"])["request"] == a and files[f"previews/{a}/{cp.WEBP}"] == b"webp of " + a.encode()
    second = cp.publish(str(_out(tmp_path, b, state="refused", image=False)), b, str(remote.work))
    assert second["parent"] == first["commit"] and _git(remote.bare, "rev-parse", "previews").strip() == second["commit"]
    assert sorted(_index(remote.bare)) == [a, b] and _index(remote.bare)[b]["state"] == "refused"
    assert f"previews/{b}/{cp.WEBP}" not in _branch(remote.bare)
    assert _git(remote.bare, "log", "-1", "--format=%an|%s", "previews").strip() == f"AutoCroissant|Preview {b}: Mini Doomer (refused)"
    assert _git(remote.bare, "rev-parse", "main").strip() == _git(remote.work, "rev-parse", "HEAD").strip()   # main untouched
    assert _git(remote.work, "status", "--porcelain") == ""                                                  # nor the checkout
    rerun = tmp_path / "rerun"                      # a re-run of a request replaces its folder
    shutil.copytree(_out(tmp_path, "cccccccc-0003", image=False), rerun)
    result = json.loads((rerun / cp.RESULT).read_text())
    (rerun / cp.RESULT).write_text(json.dumps({**result, "request": a, "state": "error"}))
    cp.publish(str(rerun), a, str(remote.work), png=True)
    files = _branch(remote.bare)
    assert json.loads(files[f"previews/{a}/{cp.RESULT}"])["state"] == "error" and f"previews/{a}/{cp.WEBP}" not in files


def test_publish_without_a_result_says_try_again(remote, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    cp.publish(str(empty), "dddddddd-0004", str(remote.work))
    result = json.loads(_branch(remote.bare)[f"previews/dddddddd-0004/{cp.RESULT}"])
    assert (result["state"], result["transient"]) == ("error", True) and "left no result" in result["reason"]
    with pytest.raises(ValueError):
        cp.publish(str(empty), "../../main", str(remote.work))
    out = _out(tmp_path, "eeeeeeee-0005")              # images on disk, without --png: only the WebP, and result.json says so
    cp.publish(str(out), "eeeeeeee-0005", str(remote.work))
    files = _branch(remote.bare)
    assert f"previews/eeeeeeee-0005/{cp.PNG}" not in files and f"previews/eeeeeeee-0005/{cp.WEBP}" in files
    assert json.loads(files[f"previews/eeeeeeee-0005/{cp.RESULT}"])["files"] == {"webp": cp.WEBP}


def test_publish_gives_up_after_its_tries(remote, tmp_path, monkeypatch):
    monkeypatch.setattr(cp, "sleep", lambda s: None)
    with pytest.raises(cb.Transient, match="not published after 5 tries: git ls-remote failed"):
        cp.publish(str(_out(tmp_path, "ffffffff-0006")), "ffffffff-0006", str(remote.work), remote=str(tmp_path / "nowhere"))
    assert cp.main(["publish", "--out", str(tmp_path / "outs" / "ffffffff-0006"), "--request", "ffffffff-0006",
                    "--git", str(remote.work), "--remote", str(tmp_path / "nowhere")]) == 1


def test_publish_rewrites_old_and_many_previews(remote, tmp_path, monkeypatch):
    now = datetime.now(timezone.utc)
    stamp = lambda days: (now - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")
    for i, days in enumerate((6.5, 5, 1)):
        cp.publish(str(_out(tmp_path, f"old-{i}-0000", at=stamp(days))), f"old-{i}-0000", str(remote.work), now=now)
    assert len(_index(remote.bare)) == 3
    later = now + timedelta(days=0.75)              # old-0 is 7.25 days old: a rewrite, keeping those under 6 days
    done = cp.publish(str(_out(tmp_path, "new-0-0000")), "new-0-0000", str(remote.work), now=later)
    assert done["rewritten"] and done["parent"] is None and done["dropped"] == ["old-0-0000"]
    assert sorted(_index(remote.bare)) == ["new-0-0000", "old-1-0000", "old-2-0000"]
    assert not any(p.startswith("previews/old-0-0000/") for p in _branch(remote.bare))
    assert _git(remote.bare, "rev-list", "--count", "previews").strip() == "1"           # one fresh commit
    monkeypatch.setattr(cp, "KEEP", 3)
    monkeypatch.setattr(cp, "KEEP_AFTER", 2)        # a fourth preview: more than KEEP, the newest KEEP_AFTER stay
    done = cp.publish(str(_out(tmp_path, "new-1-0000")), "new-1-0000", str(remote.work), now=later)
    assert done["rewritten"] and sorted(_index(remote.bare)) == ["new-0-0000", "new-1-0000"]


def test_publish_retries_a_moved_head_and_a_moved_lease(remote, tmp_path, monkeypatch):
    """Another run's commit lands between our fetch and push: ours goes on top of it (both kept); during a rewrite the
    lease refuses the forced push and the retry keeps the other run's preview."""
    other = tmp_path / "other"
    _git(tmp_path, "clone", "-q", "--depth", "1", "--filter=blob:none", f"file://{remote.bare}", str(other))
    real, calls = cp._push, []

    def racing(where, name, commit, lease):
        if not calls:                               # the first push of ours: the other run lands first
            calls.append(None)
            request = f"race-0-{lease is not None:d}-00"
            calls[0] = cp.publish(str(_out(tmp_path, request)), request, str(other))
        real(where, name, commit, lease)
    monkeypatch.setattr(cp, "sleep", lambda s: None)
    cp.publish(str(_out(tmp_path, "base-0-0000")), "base-0-0000", str(remote.work))
    monkeypatch.setattr(cp, "_push", racing)
    done = cp.publish(str(_out(tmp_path, "mine-0-0000")), "mine-0-0000", str(remote.work))
    assert done["parent"] == calls[0]["commit"] and sorted(_index(remote.bare)) == ["base-0-0000", "mine-0-0000", "race-0-0-00"]

    calls.clear()
    monkeypatch.setattr(cp, "KEEP", 3)               # the next one rewrites, pushed with a lease the other run breaks
    done = cp.publish(str(_out(tmp_path, "mine-1-0000")), "mine-1-0000", str(remote.work))
    assert done["rewritten"] and "race-0-1-00" in _index(remote.bare) and "mine-1-0000" in _index(remote.bare)


def test_kept_rules():
    now = datetime(2026, 10, 23, tzinfo=timezone.utc)
    at = lambda days: {"at": (now - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")}
    entries = {r: "t" for r in ("a", "b", "c", "me")}
    assert cp.kept(entries, {"a": at(1), "b": at(6.9), "c": at(0.1)}, "me", now) is None
    assert cp.kept(entries, {"a": at(1), "b": at(7.1), "c": at(0.1)}, "me", now) == ["me", "c", "a"]
    assert cp.kept(entries, {"a": at(1), "b": {"at": "yesterday"}, "c": at(0.1)}, "me", now) == ["me", "c", "a"]   # unknown age: old


# ========================
# card-preview.yml
# ========================
def _workflow() -> tuple:
    path = WORKFLOWS / "card-preview.yml"
    if not path.is_file():
        pytest.skip(f"no {path} (set AC_SITE_SRC to a TTSCardMaker checkout with it)")
    yaml = pytest.importorskip("yaml")
    text = path.read_text(encoding="utf-8")
    return text, yaml.safe_load(text)


def test_workflow_shape():
    """Dispatch only, three required inputs, the request in the run name and the concurrency group, contents: write only,
    15 minutes, the sparse list = SPARSE_FILES, the fork pin shared with card-edits.yml, no secret but GITHUB_TOKEN, no
    Discord, and no input expanded inside a shell script (only in env)."""
    text, doc = _workflow()
    on = doc[True]
    assert list(on) == ["workflow_dispatch"]
    inputs = on["workflow_dispatch"]["inputs"]
    assert sorted(inputs) == ["card", "edit", "request"] and all(i["required"] is True and i["type"] == "string" for i in inputs.values())
    assert "${{ inputs.request }}" in doc["run-name"]
    assert doc["permissions"] == {"contents": "write"}
    assert doc["concurrency"] == {"group": "card-preview-${{ inputs.request }}", "cancel-in-progress": False}
    job = doc["jobs"]["preview"]
    assert job["timeout-minutes"] == 15 and "if" not in job
    steps = job["steps"]
    for step in steps:
        assert "${{" not in step.get("run", ""), step.get("name") or step["run"][:40]    # inputs reach the shell through env only
    bot = next(s for s in steps if s.get("with", {}).get("repository") == "bobberlington/AutoCroissant")
    assert bot["with"]["sparse-checkout"].split() == ["/" + f for f in cp.SPARSE_FILES]
    cards = next(s for s in steps if s.get("uses", "").startswith("actions/checkout") and "repository" not in s.get("with", {}))
    assert cards["with"]["filter"] == "blob:none" and "fetch-depth" not in cards["with"]
    pin = re.search(r"psd-tools@([\w.-]+)\"", text).group(1)
    edits = WORKFLOWS / "card-edits.yml"
    if edits.exists():
        assert re.search(r"psd-tools@([\w.-]+)\"", edits.read_text(encoding="utf-8")).group(1) == pin
    assert re.search(r"pip install [^\n]*pyoxipng==[\d.]+ ", text)
    assert set(re.findall(r"secrets\.(\w+)", text)) == {"GITHUB_TOKEN"} and "discord" not in text.lower().replace("no discord", "")
    names = [s.get("name") for s in steps]
    assert names.index("Validate") < names.index("Preview") < names.index("Publish")
    publish = steps[names.index("Publish")]
    assert publish["if"] == "${{ !cancelled() && steps.validate.outcome == 'success' }}" and "--png" in publish["run"]
    preview = steps[names.index("Preview")]
    assert preview["env"]["CARD"] == "${{ inputs.card }}" and '--card="$CARD"' in preview["run"]
    assert preview["env"]["PSD_TOOLS_FONT_DIRS"] == "${{ github.workspace }}/MDW/Fonts"
    font_cache = next(s for s in steps if s.get("id") == "font-cache")
    assert font_cache["with"]["key"] == "card-fonts-${{ steps.fonts.outputs.sha }}"      # card-edits.yml's cache


def test_no_workflow_runs_on_the_previews_branch():
    """Every push trigger in the card repo is main's alone, and none fires on a branch's creation or deletion."""
    _workflow()
    yaml = pytest.importorskip("yaml")
    for path in sorted(WORKFLOWS.glob("*.yml")):
        on = yaml.safe_load(path.read_text(encoding="utf-8"))[True]
        on = {on: None} if isinstance(on, str) else {k: None for k in on} if isinstance(on, list) else on
        assert not {"create", "delete", "pull_request", "pull_request_target"} & set(on), path.name
        if "push" in on:
            assert on["push"]["branches"] == ["main"] and "branches-ignore" not in on["push"], path.name


def test_action_shaped_copy_imports_and_prints_help(tmp_path):
    """Exactly SPARSE_FILES, no config.py: both commands' --help work and only installed packages are imported."""
    bot = tmp_path / "bot"
    for rel in cp.SPARSE_FILES:
        (bot / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO / rel, bot / rel)
    for args, flags in ((["--help"], {"--card", "--edit", "--out", "--clone", "--request", "--repo", "--head", "--help"}),
                        (["publish", "--help"], {"--out", "--request", "--git", "--remote", "--png", "--help"})):
        done = run_process([executable, "-m", "commands.card_preview", *args], cwd=bot, capture_output=True, text=True, timeout=120)
        assert done.returncode == 0, done.stderr
        assert set(re.findall(r"--[a-z-]+", done.stdout)) == flags
    code = "import sys, commands.card_preview; print(sorted(m for m in ('config', 'discord', 'commands.psd_analyzer') if m in sys.modules))"
    done = run_process([executable, "-c", code], cwd=bot, capture_output=True, text=True, timeout=120)
    assert done.returncode == 0 and done.stdout.strip() == "[]", done.stderr
    extra = third_party_imports("import commands.card_preview", bot) - ALLOWED_IMPORTS - {"oxipng"}
    assert not extra, f"card-preview.yml installs none of these: {sorted(extra)}"


def _expand(value: str, inputs: dict, workspace: Path) -> str:
    """The ${{ }} expressions the steps' env use; any other fails the test (the runner would read it, this would not)."""
    known = {"inputs.card": inputs["card"], "inputs.edit": inputs["edit"], "inputs.request": inputs["request"],
             "secrets.GITHUB_TOKEN": "", "github.workspace": str(workspace)}

    def one(match):
        assert match.group(1) in known, f"the job runner here does not know ${{{{ {match.group(1)} }}}}"
        return known[match.group(1)]
    return re.sub(r"\$\{\{ ([\w.]+) \}\}", one, str(value))


def _run_job(text: str, inputs: dict, workspace: Path, bot: Path, tmp_path: Path, runner: Path) -> dict:
    """The job's Validate, Preview and Publish steps as the runner runs them (bash -e -o pipefail, its env and files): a
    step without `if` only while every earlier one passed, Publish when Validate did (its `if`). Returns {step: exit code}."""
    yaml = pytest.importorskip("yaml")
    steps = {s["name"]: s for s in yaml.safe_load(text)["jobs"]["preview"]["steps"] if s.get("name")}
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    (bin_dir / "python").write_text(f'#!/bin/sh\nexec "{executable}" "$@"\n')
    (bin_dir / "python").chmod(0o755)
    runner.mkdir(exist_ok=True)
    (runner / "env").write_text("")
    env = {"PATH": f"{bin_dir}:{environ['PATH']}", "HOME": environ.get("HOME", ""), "RUNNER_TEMP": str(runner),
           "GITHUB_WORKSPACE": str(workspace), "GITHUB_REPOSITORY": cb.CARD_REPO, "GITHUB_SHA": _git(workspace, "rev-parse", "HEAD").strip(),
           "GITHUB_ENV": str(runner / "env"), "GITHUB_OUTPUT": str(runner / "output"), "GITHUB_STEP_SUMMARY": str(runner / "summary.md"),
           "HTTPS_PROXY": "http://127.0.0.1:9", "GIT_CONFIG_GLOBAL": "/dev/null",      # no network: blobs come from the clone
           cb.FONT_DIRS_ENV: ""}
    codes = {}
    for name in ("Validate", "Preview", "Publish"):
        step = steps[name]
        if "if" not in step and any(codes.values()):       # success(): a failed step skips the rest
            continue
        if "if" in step:
            assert step["if"] == "${{ !cancelled() && steps.validate.outcome == 'success' }}", step["if"]
            if codes.get("Validate") != 0:
                continue
        step_env = {**env, **{k: _expand(v, inputs, workspace) for k, v in step.get("env", {}).items()}}
        done = run_process(["bash", "--noprofile", "--norc", "-eo", "pipefail", "-c", step["run"]], env=step_env,
                           cwd=str(bot if step.get("working-directory") == "bot" else workspace), capture_output=True,
                           text=True, timeout=600)
        codes[name] = done.returncode
        for line in (runner / "env").read_text().splitlines():
            key, _, value = line.partition("=")
            env[key] = value
    return codes


@fork
def test_the_jobs_steps_run_here(cards, remote, tmp_path):
    """card-preview.yml's own shell against a stand-in GitHub: the workspace a sparse clone of the card repo, the bot an
    Action-shaped copy; a stat edit is published on previews, a bad edit too (as an input error, Preview red), a bad
    request fails Validate and publishes nothing; main never moves."""
    text, _ = _workflow()
    _git(remote.bare, "fetch", "-q", str(cards.repo), "main:main", "--force")      # the stand-in holds the card repo
    workspace = tmp_path / "ws"
    _git(tmp_path, "clone", "-q", "--sparse", f"file://{remote.bare}", str(workspace))
    _git(workspace, "sparse-checkout", "set", "--no-cone", "/.github/")
    bot = tmp_path / "bot"
    for rel in cp.SPARSE_FILES:
        (bot / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO / rel, bot / rel)
    main = _git(remote.bare, "rev-parse", "main").strip()

    codes = _run_job(text, {"card": "Mini Doomer", "edit": '{"hp": 8}', "request": REQUEST}, workspace, bot, tmp_path, tmp_path / "run1")
    assert codes == {"Validate": 0, "Preview": 0, "Publish": 0}
    files = _branch(remote.bare)
    result = json.loads(files[f"previews/{REQUEST}/{cp.RESULT}"])
    assert (result["state"], result["reason"], result["base"]["commit"]) == ("baked", "hp 5 -> 8", main)
    assert files[f"previews/{REQUEST}/{cp.WEBP}"][:4] == b"RIFF" and files[f"previews/{REQUEST}/{cp.PNG}"][:4] == b"\x89PNG"
    assert "**Baked**: hp 5 -> 8" in (tmp_path / "run1" / "summary.md").read_text() and "Published `previews/" in (tmp_path / "run1" / "summary.md").read_text()

    bad = "1111aaaa-bad-edit"
    codes = _run_job(text, {"card": "Mini Doomer", "edit": '{"hp": "lots"}', "request": bad}, workspace, bot, tmp_path, tmp_path / "run2")
    assert codes == {"Validate": 0, "Preview": 2, "Publish": 0}
    result = json.loads(_branch(remote.bare)[f"previews/{bad}/{cp.RESULT}"])
    assert result["state"] == "error" and result["reason"] == "input: not a valid hp: 'lots'"

    before = _git(remote.bare, "rev-parse", "previews").strip()
    codes = _run_job(text, {"card": "Mini Doomer", "edit": '{"hp": 8}', "request": "x; rm -rf /"}, workspace, bot, tmp_path, tmp_path / "run3")
    assert codes == {"Validate": 1}
    assert _git(remote.bare, "rev-parse", "previews").strip() == before and _git(remote.bare, "rev-parse", "main").strip() == main
    assert sorted(_index(remote.bare)) == sorted([REQUEST, bad])


def test_the_module_stays_uncompiled():
    """Not in setup.py's INCLUDE_FILES: the Action runs it as pure Python (the change-control skill's step 5)."""
    assert "commands/card_preview.py" not in (REPO / "setup.py").read_text(encoding="utf-8")
