"""
card_preview and card_preview_branch (EDIT_PIPELINE_DESIGN section 16): one card's website edit through the bake on a
temp copy, against a local card repo made from the clone's Mini Doomer (the --clone path; the card repo is never
written; a blobless clone for the REST blob path), and its publication on the previews branch of a bare repo standing in
for GitHub, through a blobless depth-1 clone (the Action's checkout): a baked stat edit (the same pixels as card_bake's
dry run), a refused edit (with and without a render), a redundant edit, a stars edit refused beside a baked stat, the
creatures and duplicate gates, bad inputs, dispatched values kept off the runner's command lines, a runner error and a
bad blob; one commit per publish, a re-run, the rewrites by age and count, a moved head and a moved lease, the publisher
on the standard library alone. Then card-preview.yml: its shape, no trigger elsewhere on the branch, the Action-shaped
copies of both SPARSE_FILES, and its two jobs' Validate, Preview, hand-over and Publish steps run as the runner would
(bash) against those repos, a failed install too. The workflows come from $AC_SITE_SRC (a TTSCardMaker checkout) or the
clone; a real bake needs the fork's renderer:
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
import commands.card_preview_branch as br  # noqa: E402
import commands.github_api as github_api  # noqa: E402
import commands.psd_edit as pe  # noqa: E402
from conftest import CLONE, REPO, blob_sha_bytes as _sha, fork_reason, text_engine as _text_engine  # noqa: E402
from golden_cards import VOCAB  # noqa: E402
from test_card_bake import ALLOWED_IMPORTS, third_party_imports  # noqa: E402

DOOMER = "Auxiliary/Minions/Mini_Doomer.psd"
DOOMER_PNG = "Auxiliary/Minions/Mini_Doomer.png"
SNACK = "Items/Heal/2 Stars/A_Snack.psd"
MDW_DOOMER = "MDW/Mini_Doomer.psd"
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
    """A card repo with Mini Doomer, A Snack (Doomer's PSD) and its numbered A_Snack2.png, Types/, the manifest, and
    Doomer's PSD again in MDW/ (no card: an EXCLUDE_FOLDERS folder)."""
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
    files[MDW_DOOMER] = files[DOOMER]
    files[".github/workflows/x.yml"] = b"name: x\n"
    repo = tmp_path / "cards"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "uploadpack.allowFilter", "true")      # a blobless clone of it: the REST blob path
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


def _preview(cards, tmp_path, edit, card="Mini Doomer", request=REQUEST, name="out", clone=None) -> tuple:
    edit_file = tmp_path / f"{name}.json"
    edit_file.write_bytes(edit if isinstance(edit, bytes) else json.dumps(edit).encode())
    out = tmp_path / name
    code = cp.main(["--card", card, "--edit", str(edit_file), "--out", str(out), "--clone", str(clone or cards.repo),
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
    assert result["edit"] == {"hp": 8, "note": "checked"} and result["changes"] == [{"field": "hp", "old": 5, "new": 8}]
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
    """Refused; the edit echoed exactly as sent (a cleared stat's null too: the site compares it)."""
    code, result, out = _preview(cards, tmp_path, {"hp": 25, "def": None})
    assert code == 0 and (result["state"], result["reason"]) == ("refused", "stat>10: hp 25: above 20")
    assert result["edit"] == {"hp": 25, "def": None}
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
    assert result["edit"] == {"hp": 8, "creatures": [{"hp": 8}]}                 # as sent
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
    ("Mini\u2028Doomer", b'{"hp": 8}', REQUEST, "input: the card must be 1-200 characters"),
    ("Creatures/Nope.psd", b'{"hp": 8}', REQUEST, "input: no card PSD at 'Creatures/Nope.psd'"),
    (MDW_DOOMER, b'{"hp": 8}', REQUEST, f"input: no card PSD at '{MDW_DOOMER}'"),            # not a card the bake takes
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


def test_dispatched_values_never_start_a_log_line(cards, tmp_path, capsys):
    """The runner reads a line starting `::` as a command: a card or an edit key with a newline is printed escaped, on
    one line; result.json holds the card clean and clipped to MAX_CARD; the summary escapes every value."""
    spoof = "\n::error title=spoof::x"
    for card, edit in (("Mini Doomer" + spoof, {"hp": 8}), ("Mini Doomer", {"x" + spoof: 1, "hp": 8}), ("x" * 70000, {"hp": 8})):
        capsys.readouterr()
        code, result, _ = _preview(cards, tmp_path, edit, card=card)
        printed = capsys.readouterr()
        assert code == 2 and result["reason"].startswith("input:"), result["reason"]
        assert not [line for line in (printed.out + printed.err).splitlines() if line.lstrip().startswith("::")]
        assert "\n" not in result["card"] + result["reason"] and len(result["card"]) <= br.MAX_CARD
    assert result["card"] == "x" * (br.MAX_CARD - 1) + "…"
    text = cp.render({**br.blank(REQUEST, ""), "card": "a\n::warning::b", "path": "p\n::c", "edit": {"ability": "d\n::e"},
                      "reason": "f\r\n::g", "notes": ["h ::i"], "fields": {"hp\n::j": "baked"}})
    assert not [line for line in text.splitlines() if line.lstrip().startswith("::")] and "a\\n::warning::b" in text


def test_the_rest_blob_path(cards, tmp_path, monkeypatch):
    """A blobless clone (the runner's checkout): what the preview reads comes by REST, verified: the manifest, the PSD
    and the PNG for a stat edit, three reads. A blob that fails its hash is a try-again error, before the card's bake
    (the manifest) or in it (the PSD)."""
    blobless = tmp_path / "blobless"
    _git(tmp_path, "clone", "-q", "--no-checkout", "--filter=blob:none", f"file://{cards.repo}", str(blobless))
    served, bad = [], set()

    def fetch_blob(repository, sha, timeout=30):
        served.append(sha)
        data = run_process(["git", "-C", str(cards.repo), "cat-file", "blob", sha], capture_output=True, check=True).stdout
        return type("Reply", (), {"content": b"tampered" if sha in bad else data})()
    monkeypatch.setattr(github_api, "fetch_blob", fetch_blob)
    code, result, _ = _preview(cards, tmp_path, {"hp": 5}, clone=blobless)
    assert code == 0 and result["state"] == "redundant"
    assert sorted(served) == sorted(_sha(cards.files[p]) for p in (cb.ICON_MANIFEST_PATH, DOOMER, DOOMER_PNG))
    for path in (cb.ICON_MANIFEST_PATH, DOOMER):
        bad.clear()
        bad.add(_sha(cards.files[path]))
        code, result, _ = _preview(cards, tmp_path, {"hp": 5}, clone=blobless, name=path.split("/")[0])
        assert (code, result["state"], result["transient"]) == (1, "error", True), path
        assert result["reason"].startswith(f"blob: blob {_sha(cards.files[path])[:8]} hashes to "), result["reason"]


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
    result = {**br.blank(request, card), "card": card, "state": state, "path": DOOMER, "at": at or br.stamp(), "reason": "hp 5 -> 8"}
    if image:
        result["files"] = {"png": br.PNG, "webp": br.WEBP}
        (out / br.PNG).write_bytes(b"png of " + request.encode())
        (out / br.WEBP).write_bytes(b"webp of " + request.encode())
    (out / br.RESULT).write_text(json.dumps(result))
    return out


def _branch(bare) -> dict:
    """{path: bytes} on previews."""
    out = {}
    for line in _git(bare, "ls-tree", "-r", "previews").splitlines():
        meta, path = line.split("\t", 1)
        out[path] = run_process(["git", "-C", str(bare), "cat-file", "blob", meta.split()[2]], capture_output=True).stdout
    return out


def _index(bare) -> dict:
    return json.loads(_branch(bare)[br.INDEX])["previews"]


def test_publish_one_commit_per_preview(remote, tmp_path):
    a, b = "aaaaaaaa-0001", "bbbbbbbb-0002"
    first = br.publish(str(_out(tmp_path, a)), a, str(remote.work), png=True)
    assert first["parent"] is None and not first["rewritten"]
    files = _branch(remote.bare)
    assert sorted(files) == [br.README, br.INDEX, f"previews/{a}/{br.PNG}", f"previews/{a}/{br.WEBP}", f"previews/{a}/{br.RESULT}"]
    assert json.loads(files[f"previews/{a}/{br.RESULT}"])["request"] == a and files[f"previews/{a}/{br.WEBP}"] == b"webp of " + a.encode()
    second = br.publish(str(_out(tmp_path, b, state="refused", image=False)), b, str(remote.work))
    assert second["parent"] == first["commit"] and _git(remote.bare, "rev-parse", "previews").strip() == second["commit"]
    assert sorted(_index(remote.bare)) == [a, b] and _index(remote.bare)[b]["state"] == "refused"
    assert f"previews/{b}/{br.WEBP}" not in _branch(remote.bare)
    assert _git(remote.bare, "log", "-1", "--format=%an|%s", "previews").strip() == f"AutoCroissant|Preview {b}: Mini Doomer (refused)"
    assert _git(remote.bare, "rev-parse", "main").strip() == _git(remote.work, "rev-parse", "HEAD").strip()   # main untouched
    assert _git(remote.work, "status", "--porcelain") == ""                                                  # nor the checkout
    rerun = tmp_path / "rerun"                      # a re-run of a request replaces its folder
    shutil.copytree(_out(tmp_path, "cccccccc-0003", image=False), rerun)
    result = json.loads((rerun / br.RESULT).read_text())
    (rerun / br.RESULT).write_text(json.dumps({**result, "request": a, "state": "error"}))
    br.publish(str(rerun), a, str(remote.work), png=True)
    files = _branch(remote.bare)
    assert json.loads(files[f"previews/{a}/{br.RESULT}"])["state"] == "error" and f"previews/{a}/{br.WEBP}" not in files


def test_publish_without_a_result_says_try_again(remote, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    br.publish(str(empty), "dddddddd-0004", str(remote.work))
    result = json.loads(_branch(remote.bare)[f"previews/dddddddd-0004/{br.RESULT}"])
    assert (result["state"], result["transient"]) == ("error", True) and "left no result" in result["reason"]
    with pytest.raises(ValueError):
        br.publish(str(empty), "../../main", str(remote.work))
    out = _out(tmp_path, "eeeeeeee-0005")              # images on disk, without --png: only the WebP, and result.json says so
    br.publish(str(out), "eeeeeeee-0005", str(remote.work))
    files = _branch(remote.bare)
    assert f"previews/eeeeeeee-0005/{br.PNG}" not in files and f"previews/eeeeeeee-0005/{br.WEBP}" in files
    assert json.loads(files[f"previews/eeeeeeee-0005/{br.RESULT}"])["files"] == {"webp": br.WEBP}


def test_publish_gives_up_after_its_tries(remote, tmp_path, monkeypatch):
    monkeypatch.setattr(br, "sleep", lambda s: None)
    with pytest.raises(br.Unpublished, match="not published after 5 tries: git ls-remote failed"):
        br.publish(str(_out(tmp_path, "ffffffff-0006")), "ffffffff-0006", str(remote.work), remote=str(tmp_path / "nowhere"))
    assert br.main(["--out", str(tmp_path / "outs" / "ffffffff-0006"), "--request", "ffffffff-0006",
                    "--git", str(remote.work), "--remote", str(tmp_path / "nowhere")]) == 1


def test_publish_rewrites_old_and_many_previews(remote, tmp_path, monkeypatch):
    now = datetime.now(timezone.utc)
    stamp = lambda days: (now - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")
    for i, days in enumerate((6.5, 5, 1)):
        br.publish(str(_out(tmp_path, f"old-{i}-0000", at=stamp(days))), f"old-{i}-0000", str(remote.work), now=now)
    assert len(_index(remote.bare)) == 3
    later = now + timedelta(days=0.75)              # old-0 is 7.25 days old: a rewrite, keeping those under 6 days
    done = br.publish(str(_out(tmp_path, "new-0-0000")), "new-0-0000", str(remote.work), now=later)
    assert done["rewritten"] and done["parent"] is None and done["dropped"] == ["old-0-0000"]
    assert sorted(_index(remote.bare)) == ["new-0-0000", "old-1-0000", "old-2-0000"]
    assert not any(p.startswith("previews/old-0-0000/") for p in _branch(remote.bare))
    assert _git(remote.bare, "rev-list", "--count", "previews").strip() == "1"           # one fresh commit
    monkeypatch.setattr(br, "KEEP", 3)
    monkeypatch.setattr(br, "KEEP_AFTER", 2)        # a fourth preview: more than KEEP, the newest KEEP_AFTER stay
    done = br.publish(str(_out(tmp_path, "new-1-0000")), "new-1-0000", str(remote.work), now=later)
    assert done["rewritten"] and sorted(_index(remote.bare)) == ["new-0-0000", "new-1-0000"]


def test_publish_retries_a_moved_head_and_a_moved_lease(remote, tmp_path, monkeypatch):
    """Another run's commit lands between our fetch and push: ours goes on top of it (both kept); during a rewrite the
    lease refuses the forced push and the retry keeps the other run's preview."""
    other = tmp_path / "other"
    _git(tmp_path, "clone", "-q", "--depth", "1", "--filter=blob:none", f"file://{remote.bare}", str(other))
    real, calls = br._push, []

    def racing(where, name, commit, lease):
        if not calls:                               # the first push of ours: the other run lands first
            calls.append(None)
            request = f"race-0-{lease is not None:d}-00"
            calls[0] = br.publish(str(_out(tmp_path, request)), request, str(other))
        real(where, name, commit, lease)
    monkeypatch.setattr(br, "sleep", lambda s: None)
    br.publish(str(_out(tmp_path, "base-0-0000")), "base-0-0000", str(remote.work))
    monkeypatch.setattr(br, "_push", racing)
    done = br.publish(str(_out(tmp_path, "mine-0-0000")), "mine-0-0000", str(remote.work))
    assert done["parent"] == calls[0]["commit"] and sorted(_index(remote.bare)) == ["base-0-0000", "mine-0-0000", "race-0-0-00"]

    calls.clear()
    monkeypatch.setattr(br, "KEEP", 3)               # the next one rewrites, pushed with a lease the other run breaks
    done = br.publish(str(_out(tmp_path, "mine-1-0000")), "mine-1-0000", str(remote.work))
    assert done["rewritten"] and "race-0-1-00" in _index(remote.bare) and "mine-1-0000" in _index(remote.bare)


def test_kept_rules():
    now = datetime(2026, 10, 23, tzinfo=timezone.utc)
    at = lambda days: {"at": (now - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")}
    entries = {r: "t" for r in ("a", "b", "c", "me")}
    assert br.kept(entries, {"a": at(1), "b": at(6.9), "c": at(0.1)}, "me", now) is None
    assert br.kept(entries, {"a": at(1), "b": at(7.1), "c": at(0.1)}, "me", now) == ["me", "c", "a"]
    assert br.kept(entries, {"a": at(1), "b": {"at": "yesterday"}, "c": at(0.1)}, "me", now) == ["me", "c", "a"]   # unknown age: old


def test_publish_keeps_the_card_on_one_line(remote, tmp_path):
    """A result.json's card (a raw input) reaches the index row and the commit message clean and clipped."""
    request = "cleancard-0008"
    br.publish(str(_out(tmp_path, request, card="x\n::error::y" + "z" * 500)), request, str(remote.work))
    row = _index(remote.bare)[request]
    assert row["card"].startswith("x\\n::error::yz") and len(row["card"]) == br.MAX_CARD
    message = _git(remote.bare, "log", "-1", "--format=%B", "previews")
    assert message.startswith(f"Preview {request}: x\\n::error::yz") and not [m for m in message.splitlines() if m.startswith("::")]


def test_the_publisher_needs_only_the_standard_library(remote, tmp_path):
    """Exactly card_preview_branch.SPARSE_FILES, Python without site-packages (-S: like the runner's own python3, no
    install): --help and a publish work; the commits' identity is git_commit's."""
    from commands.git_commit import GIT_IDENTITY
    assert br.IDENTITY == GIT_IDENTITY
    bot = tmp_path / "publisher"
    for rel in br.SPARSE_FILES:
        (bot / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO / rel, bot / rel)
    done = run_process([executable, "-S", "-m", "commands.card_preview_branch", "--help"], cwd=bot, capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    assert set(re.findall(r"--[a-z-]+", done.stdout)) == {"--out", "--request", "--git", "--remote", "--png", "--help"}
    request = "stdlib00-0009"
    done = run_process([executable, "-S", "-m", "commands.card_preview_branch", "--out", str(_out(tmp_path, request)), "--request", request,
                        "--git", str(remote.work)], cwd=bot, capture_output=True, text=True, timeout=120, env={**environ, "GIT_CONFIG_GLOBAL": "/dev/null"})
    assert done.returncode == 0 and done.stdout.startswith(f"Published `previews/{request}/` (baked)"), done.stderr
    assert list(_index(remote.bare)) == [request]


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
    """Dispatch only, three required inputs, the request in the run name and the concurrency group. Two jobs: the
    preview (the workflow's contents: read; its bot checkout card_preview.SPARSE_FILES; the fork pin shared with
    card-edits.yml) hands its out dir to the publish job (contents: write; card_preview_branch.SPARSE_FILES; the runner's
    python3, no setup-python or pip; no --png), 15 minutes between them; no secret but GITHUB_TOKEN (the Preview step's
    REST reads), no Discord, and no input expanded inside a shell script (only in env)."""
    text, doc = _workflow()
    on = doc[True]
    assert list(on) == ["workflow_dispatch"]
    inputs = on["workflow_dispatch"]["inputs"]
    assert sorted(inputs) == ["card", "edit", "request"] and all(i["required"] is True and i["type"] == "string" for i in inputs.values())
    assert "${{ inputs.request }}" in doc["run-name"]
    assert doc["permissions"] == {"contents": "read"}
    assert doc["concurrency"] == {"group": "card-preview-${{ inputs.request }}", "cancel-in-progress": False}
    assert list(doc["jobs"]) == ["preview", "publish"]
    preview, publish = doc["jobs"]["preview"], doc["jobs"]["publish"]
    assert "permissions" not in preview and publish["permissions"] == {"contents": "write"}
    assert "if" not in preview and preview["outputs"] == {"ok": "${{ steps.validate.outputs.ok }}"}
    assert publish["needs"] == "preview" and publish["if"] == "${{ !cancelled() && needs.preview.outputs.ok == 'true' }}"
    assert preview["timeout-minutes"] + publish["timeout-minutes"] <= 15
    for step in preview["steps"] + publish["steps"]:
        assert "${{" not in step.get("run", ""), step.get("name") or step["run"][:40]    # inputs reach the shell through env only

    def step(job, name):
        return next(s for s in job["steps"] if s.get("name") == name)
    for job, files in ((preview, cp.SPARSE_FILES), (publish, br.SPARSE_FILES)):
        bot = next(s for s in job["steps"] if s.get("with", {}).get("repository") == "bobberlington/AutoCroissant")
        assert bot["with"]["sparse-checkout"].split() == ["/" + f for f in files]
        cards = next(s for s in job["steps"] if s.get("uses", "").startswith("actions/checkout") and "repository" not in s.get("with", {}))
        assert cards["with"]["filter"] == "blob:none" and "fetch-depth" not in cards["with"]
        assert cards["with"]["sparse-checkout"].split() == ["/.github/"]
    install = re.findall(r"pip install [^\n]+", text)
    edits = WORKFLOWS / "card-edits.yml"
    assert len(install) == 1 and "psd-tools[composite,text] @ git+" in install[0]
    if edits.exists():
        assert re.findall(r"pip install [^\n]+", edits.read_text(encoding="utf-8")) == install       # the bake's install, pin and all
    assert not [s for s in publish["steps"] if "setup-python" in s.get("uses", "") or "pip " in s.get("run", "")]
    assert set(re.findall(r"secrets\.(\w+)", text)) == {"GITHUB_TOKEN"} and "secrets" not in str(publish)
    assert "discord" not in text.lower().replace("no discord", "")
    names = [s.get("name") for s in preview["steps"]]
    assert names.index("Validate") < names.index("Preview") < names.index("Hand over") == len(names) - 1
    hand = step(preview, "Hand over")
    assert hand["if"] == "${{ !cancelled() && steps.validate.outcome == 'success' }}" and hand["uses"].startswith("actions/upload-artifact@")
    assert hand["with"]["path"].strip().splitlines() == ["${{ runner.temp }}/preview/" + f for f in (br.RESULT, br.WEBP)]   # not the PNG
    assert hand["with"]["if-no-files-found"] == "ignore"
    take = step(publish, "Take over")
    assert take["uses"].startswith("actions/download-artifact@") and take["continue-on-error"] is True
    assert take["with"] == {"name": hand["with"]["name"], "path": "${{ runner.temp }}/preview"}
    run = step(publish, "Publish")["run"]
    assert run.startswith("python3 -m commands.card_preview_branch ") and "--png" not in run and '"$RUNNER_TEMP/preview"' in run
    shown = step(preview, "Preview")
    assert shown["env"]["CARD"] == "${{ inputs.card }}" and '--card="$CARD"' in shown["run"] and '"$RUNNER_TEMP/preview"' in shown["run"]
    assert shown["env"]["PSD_TOOLS_FONT_DIRS"] == "${{ github.workspace }}/MDW/Fonts"
    font_cache = next(s for s in preview["steps"] if s.get("id") == "font-cache")
    assert font_cache["with"]["key"] == "card-fonts-${{ steps.fonts.outputs.sha }}"      # card-edits.yml's cache


def test_no_workflow_runs_on_the_previews_branch():
    """Every push trigger in the card repo is main's alone, none fires on a branch's creation or deletion, and none
    follows another workflow's run or checks."""
    _workflow()
    yaml = pytest.importorskip("yaml")
    for path in sorted(WORKFLOWS.glob("*.yml")):
        on = yaml.safe_load(path.read_text(encoding="utf-8"))[True]
        on = {on: None} if isinstance(on, str) else {k: None for k in on} if isinstance(on, list) else on
        assert not {"create", "delete", "pull_request", "pull_request_target", "workflow_run", "check_run", "check_suite"} & set(on), path.name
        if "push" in on:
            assert on["push"]["branches"] == ["main"] and "branches-ignore" not in on["push"], path.name


def test_action_shaped_copy_imports_and_prints_help(tmp_path):
    """Exactly card_preview.SPARSE_FILES, no config.py: --help works and only installed packages are imported."""
    bot = tmp_path / "bot"
    for rel in cp.SPARSE_FILES:
        (bot / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO / rel, bot / rel)
    done = run_process([executable, "-m", "commands.card_preview", "--help"], cwd=bot, capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    assert set(re.findall(r"--[a-z-]+", done.stdout)) == {"--card", "--edit", "--out", "--clone", "--request", "--repo", "--head", "--help"}
    code = "import sys, commands.card_preview; print(sorted(m for m in ('config', 'discord', 'commands.psd_analyzer') if m in sys.modules))"
    done = run_process([executable, "-c", code], cwd=bot, capture_output=True, text=True, timeout=120)
    assert done.returncode == 0 and done.stdout.strip() == "[]", done.stderr
    extra = third_party_imports("import commands.card_preview", bot) - ALLOWED_IMPORTS - {"oxipng"}
    assert not extra, f"card-preview.yml installs none of these: {sorted(extra)}"
    assert set(br.SPARSE_FILES) <= set(cp.SPARSE_FILES)


def _expand(value: str, inputs: dict, workspace: Path) -> str:
    """The ${{ }} expressions the steps' env use; any other fails the test (the runner would read it, this would not)."""
    known = {"inputs.card": inputs["card"], "inputs.edit": inputs["edit"], "inputs.request": inputs["request"],
             "secrets.GITHUB_TOKEN": "", "github.workspace": str(workspace)}

    def one(match):
        assert match.group(1) in known, f"the job runner here does not know ${{{{ {match.group(1)} }}}}"
        return known[match.group(1)]
    return re.sub(r"\$\{\{ ([\w.]+) \}\}", one, str(value))


def _run_jobs(text: str, inputs: dict, workspaces: dict, bots: dict, tmp_path: Path, name: str, installed: bool = True) -> dict:
    """
    The workflow's two jobs as the runner runs them (bash -e -o pipefail; each job its own workspace, RUNNER_TEMP, env and
    outputs). The preview job: Validate, then Preview (skipped, as after a failed install, when not installed), then the
    hand-over when its `if` holds (a copy of its listed files that exist, as upload-artifact with if-no-files-found:
    ignore). The publish
    job when the preview job's `ok` output is "true": the take-over (the copy back; continue-on-error) and Publish, whose
    python3 is this Python without site-packages. Returns {step: exit code, None when skipped}.
    """
    yaml = pytest.importorskip("yaml")
    doc = yaml.safe_load(text)
    steps = {job: {s["name"]: s for s in doc["jobs"][job]["steps"] if s.get("name")} for job in ("preview", "publish")}
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    (bin_dir / "python").write_text(f'#!/bin/sh\nexec "{executable}" "$@"\n')
    (bin_dir / "python3").write_text(f'#!/bin/sh\nexec "{executable}" -S "$@"\n')      # the runner's own: no site-packages
    for shim in ("python", "python3"):
        (bin_dir / shim).chmod(0o755)
    artifact = tmp_path / name / "artifact"
    codes = {}

    def run_step(job: str, step: dict, runner: Path) -> int:
        env = {"PATH": f"{bin_dir}:{environ['PATH']}", "HOME": environ.get("HOME", ""), "RUNNER_TEMP": str(runner),
               "GITHUB_WORKSPACE": str(workspaces[job]), "GITHUB_REPOSITORY": cb.CARD_REPO,
               "GITHUB_SHA": _git(workspaces[job], "rev-parse", "HEAD").strip(), "GITHUB_OUTPUT": str(runner / "output"),
               "GITHUB_STEP_SUMMARY": str(runner / "summary.md"), "HTTPS_PROXY": "http://127.0.0.1:9",   # no network
               "GIT_CONFIG_GLOBAL": "/dev/null"}
        env.update({k: _expand(v, inputs, workspaces[job]) for k, v in step.get("env", {}).items()})
        done = run_process(["bash", "--noprofile", "--norc", "-eo", "pipefail", "-c", step["run"]], env=env,
                           cwd=str(bots[job] if step.get("working-directory") == "bot" else workspaces[job]),
                           capture_output=True, text=True, timeout=600)
        return done.returncode

    runner = tmp_path / name / "preview"
    runner.mkdir(parents=True)
    for step_name in ("Validate", "Preview"):
        skipped = any(codes.values()) or (step_name == "Preview" and not installed)
        codes[step_name] = None if skipped else run_step("preview", steps["preview"][step_name], runner)
    hand = steps["preview"]["Hand over"]
    assert hand["if"] == "${{ !cancelled() && steps.validate.outcome == 'success' }}"
    found = [Path(f.strip().replace("${{ runner.temp }}", str(runner))) for f in hand["with"]["path"].strip().splitlines()] if codes["Validate"] == 0 else []
    for path in [f for f in found if f.is_file()]:
        artifact.mkdir(parents=True, exist_ok=True)
        shutil.copy(path, artifact / path.name)
    outputs = dict(line.partition("=")[::2] for line in (runner / "output").read_text().splitlines()) if (runner / "output").exists() else {}
    assert doc["jobs"]["publish"]["if"] == "${{ !cancelled() && needs.preview.outputs.ok == 'true' }}"
    if outputs.get("ok") == "true":
        runner = tmp_path / name / "publish"
        runner.mkdir(parents=True)
        codes["Take over"] = 0 if artifact.is_dir() else 1
        if artifact.is_dir():
            shutil.copytree(artifact, runner / "preview")
        codes["Publish"] = run_step("publish", steps["publish"]["Publish"], runner)
    return codes


@fork
def test_the_jobs_steps_run_here(cards, remote, clone, tmp_path):
    """card-preview.yml's own shell against a stand-in GitHub: each job's workspace a sparse clone of the card repo (the
    preview's with MDW/Fonts as the font cache restores it), the bots Action-shaped copies. A stat edit is published on
    previews (the WebP, not the PNG); a bad edit too (an input error, Preview red); a failed install publishes a try-again
    error; a bad request fails Validate and publishes nothing; main never moves."""
    text, _ = _workflow()
    _git(remote.bare, "fetch", "-q", str(cards.repo), "main:main", "--force")      # the stand-in holds the card repo
    workspaces = {"preview": tmp_path / "ws", "publish": tmp_path / "ws-publish"}
    _git(tmp_path, "clone", "-q", "--sparse", f"file://{remote.bare}", str(workspaces["preview"]))
    _git(tmp_path, "clone", "-q", "--sparse", "--depth", "1", "--filter=blob:none", f"file://{remote.bare}", str(workspaces["publish"]))
    for workspace in workspaces.values():
        _git(workspace, "sparse-checkout", "set", "--no-cone", "/.github/")
    (workspaces["preview"] / "MDW").mkdir()
    (workspaces["preview"] / pe.FONTS_DIR).symlink_to(clone / pe.FONTS_DIR)
    bots = {}
    for job, files in (("preview", cp.SPARSE_FILES), ("publish", br.SPARSE_FILES)):
        bots[job] = tmp_path / f"bot-{job}"
        for rel in files:
            (bots[job] / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(REPO / rel, bots[job] / rel)
    main = _git(remote.bare, "rev-parse", "main").strip()

    codes = _run_jobs(text, {"card": "Mini Doomer", "edit": '{"hp": 8}', "request": REQUEST}, workspaces, bots, tmp_path, "run1")
    assert codes == {"Validate": 0, "Preview": 0, "Take over": 0, "Publish": 0}
    files = _branch(remote.bare)
    result = json.loads(files[f"previews/{REQUEST}/{br.RESULT}"])
    assert (result["state"], result["reason"], result["base"]["commit"]) == ("baked", "hp 5 -> 8", main)
    assert files[f"previews/{REQUEST}/{br.WEBP}"][:4] == b"RIFF" and f"previews/{REQUEST}/{br.PNG}" not in files
    assert result["files"] == {"webp": br.WEBP} and result["runner"]["fonts"] is True
    assert "**Baked**: hp 5 -> 8" in (tmp_path / "run1" / "preview" / "summary.md").read_text()
    assert "Published `previews/" in (tmp_path / "run1" / "publish" / "summary.md").read_text()

    bad = "1111aaaa-bad-edit"
    codes = _run_jobs(text, {"card": "Mini Doomer", "edit": '{"hp": "lots"}', "request": bad}, workspaces, bots, tmp_path, "run2")
    assert codes == {"Validate": 0, "Preview": 2, "Take over": 0, "Publish": 0}
    result = json.loads(_branch(remote.bare)[f"previews/{bad}/{br.RESULT}"])
    assert result["state"] == "error" and result["reason"] == "input: not a valid hp: 'lots'"

    broken = "2222bbbb-no-install"
    codes = _run_jobs(text, {"card": "Mini Doomer", "edit": '{"hp": 8}', "request": broken}, workspaces, bots, tmp_path, "run3", installed=False)
    assert codes == {"Validate": 0, "Preview": None, "Take over": 1, "Publish": 0}
    result = json.loads(_branch(remote.bare)[f"previews/{broken}/{br.RESULT}"])
    assert (result["state"], result["transient"]) == ("error", True) and "left no result" in result["reason"]

    before = _git(remote.bare, "rev-parse", "previews").strip()
    codes = _run_jobs(text, {"card": "Mini Doomer", "edit": '{"hp": 8}', "request": "x; rm -rf /"}, workspaces, bots, tmp_path, "run4")
    assert codes == {"Validate": 1, "Preview": None}
    assert _git(remote.bare, "rev-parse", "previews").strip() == before and _git(remote.bare, "rev-parse", "main").strip() == main
    assert sorted(_index(remote.bare)) == sorted([REQUEST, bad, broken])


def test_the_modules_stay_uncompiled():
    """Not in setup.py's INCLUDE_FILES: the Action runs them as pure Python (the change-control skill's step 5)."""
    setup = (REPO / "setup.py").read_text(encoding="utf-8")
    assert "commands/card_preview.py" not in setup and "commands/card_preview_branch.py" not in setup
