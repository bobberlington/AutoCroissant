"""
card_preview and card_preview_branch (EDIT_PIPELINE_DESIGN section 16): one card's website edit through the bake on a
temp copy, against a local card repo made from the clone's Mini Doomer (the --clone path; the card repo is never
written; a blobless clone for the REST blob path), and its publication on the previews branch of a bare repo standing in
for GitHub, through a blobless depth-1 clone (the Action's checkout): a baked stat edit (the same pixels as card_bake's
dry run), a refused edit (with and without a render), a redundant edit, a stars edit refused beside a baked stat, the
creatures and duplicate gates, bad inputs, a name or ability its parts hold, dispatched values kept off the runner's
command lines, a runner error and a bad blob, the early reads, fonts and records that come after the wait; one commit
per publish, a re-run, the rewrites by age and count, a moved head and a moved lease, a warm read, the publisher
on the standard library alone. Then card-preview.yml's two jobs' Validate, Reads, Start, Go, Preview, hand-over and
Publish steps run as the runner would (bash) against those repos, from Action-shaped copies of both SPARSE_FILES (the
jobs' bot checkouts). The workflows come from $AC_SITE_SRC
(a TTSCardMaker checkout) or the clone; a real bake needs the fork's renderer:
    AUTOCROISSANT_LOCAL_DIR=~/Desktop/TTSCardMaker AC_SITE_SRC=<TTSCardMaker checkout> <fork python> -m pytest -q tests/test_card_preview.py
"""
import json
import re
import shutil
import threading
from ast import parse as ast_parse
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from os import environ
from pathlib import Path
from subprocess import run as run_process
from sys import executable
from types import SimpleNamespace

import pytest
import yaml

import commands.card_bake as cb
import commands.card_db as card_db
import commands.card_preview as cp
import commands.card_preview_branch as br
import commands.card_preview_reads as early
import commands.github_api as github_api
import commands.psd_edit as pe
from conftest import REPO, SITE_SRC, blob_sha_bytes as _sha
from golden_cards import VOCAB
from helpers import bot_files, clone_tree

DOOMER = "Auxiliary/Minions/Mini_Doomer.psd"
DOOMER_PNG = "Auxiliary/Minions/Mini_Doomer.png"
SNACK = "Items/Heal/2 Stars/A_Snack.psd"
MDW_DOOMER = "MDW/Mini_Doomer.psd"
WORKFLOWS = SITE_SRC / ".github" / "workflows"
REQUEST = "0f8e7d6c-5b4a-4321-9876-0123456789ab"       # a crypto.randomUUID()


# ========================
# Fixtures
# ========================
def _git(where, *args, env=None) -> str:
    done = run_process(["git", "-C", str(where), *args], capture_output=True, text=True, timeout=120,
                       env={**environ, "GIT_CONFIG_GLOBAL": "/dev/null", **(env or {})})
    assert done.returncode == 0, done.stderr
    return done.stdout


@pytest.fixture(autouse=True)
def lane_memory(monkeypatch):
    """The lane's MAX_RSS counts from the test's start (ru_maxrss: the worker's peak so far)."""
    import commands.card_preview_lane as lane_mod
    monkeypatch.setattr(lane_mod, "MAX_RSS", lane_mod._rss() + lane_mod.MAX_RSS)


def _commit(repo, message="Website edit: test") -> str:
    who = {"GIT_AUTHOR_NAME": "web", "GIT_AUTHOR_EMAIL": "w@x", "GIT_COMMITTER_NAME": "web", "GIT_COMMITTER_EMAIL": "w@x"}
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", message, env=who)
    return _git(repo, "rev-parse", "HEAD").strip()


@pytest.fixture(scope="session")
def preview_template(tmp_path_factory, clone):
    """A card repo with Mini Doomer, A Snack (Doomer's PSD) and its numbered A_Snack2.png, Types/, the manifest, and
    Doomer's PSD again in MDW/ (no card: an EXCLUDE_FOLDERS folder); built once per worker."""
    vocab = json.loads(VOCAB.read_text(encoding="utf-8"))
    files = {f"Types/{t}.png": t.encode() for t in vocab["all_types"]}
    files.update({f"Types/Stars/{s}.png": s.encode() for s in vocab["stars"]})
    files[cb.ICON_MANIFEST_PATH] = json.dumps(vocab["manifest"]).encode()
    files[DOOMER], files[DOOMER_PNG] = (clone / DOOMER).read_bytes(), (clone / DOOMER_PNG).read_bytes()
    files[SNACK], files[SNACK[:len(SNACK) - 4] + ".png"] = files[DOOMER], files[DOOMER_PNG]
    files["Items/Heal/2 Stars/A_Snack2.png"] = b"a duplicate"
    files[MDW_DOOMER] = files[DOOMER]
    files[".github/workflows/x.yml"] = b"name: x\n"
    repo = tmp_path_factory.mktemp("preview-template") / "cards"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "uploadpack.allowFilter", "true")      # a blobless clone of it: the REST blob path
    for path, data in files.items():
        (repo / path).parent.mkdir(parents=True, exist_ok=True)
        (repo / path).write_bytes(data)
    return repo, _commit(repo), files


@pytest.fixture
def cards(tmp_path, clone, preview_template, monkeypatch):
    """The template's repo, as a clone_tree copy, and the runner's environment."""
    template, head, files = preview_template
    repo = tmp_path / "cards"
    clone_tree(template, repo)
    monkeypatch.setenv(cb.FONT_DIRS_ENV, str(clone / pe.FONTS_DIR))
    monkeypatch.setenv(cb.WORK_ENV, str(tmp_path / "runner"))
    (tmp_path / "runner").mkdir()
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_RUN_ID", raising=False)
    return type("Cards", (), {"repo": repo, "head": head, "files": dict(files)})


def _state(repo) -> tuple:
    """What a preview must leave as it was: HEAD, every ref, the index and worktree status, the object counts (not sizes:
    they move under load)."""
    counts = [l for l in _git(repo, "count-objects", "-v").splitlines() if l.startswith(("count:", "in-pack:"))]
    return (_git(repo, "rev-parse", "HEAD"), _git(repo, "for-each-ref"), _git(repo, "status", "--porcelain", "--ignored"), counts)


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
def test_a_stat_edit_bakes_the_bakes_pixels(cards, tmp_path, monkeypatch):
    """hp 5 -> 8: baked, the PNG the bake's dry run makes (the same pixels), a 720 px AVIF; the
    base is the commit and the blobs it read; the card repo is untouched and nothing is left in the runner's temp. The
    preview skips what only a commit uses (duplicates, crops, banding) and makes before.avif beside the bake."""
    from PIL import Image
    seen, bands = [], []
    monkeypatch.setattr(cb.png_export, "band", lambda data, old=None: bands.append(len(data)))  # None: kept as baked
    monkeypatch.setattr(cb, "_follow_dups", lambda *a, _f=cb._follow_dups: seen.append("dups") or _f(*a))
    monkeypatch.setattr(pe, "artifacts", lambda *a, _f=pe.artifacts: seen.append("crops") or _f(*a))
    monkeypatch.setattr(cp, "_small_of", lambda image, path, _f=cp._small_of: seen.append(path) or _f(image, path))
    olds = []                                                               # gate_dirty's old box: its own thread
    monkeypatch.setattr(pe, "_uncached", lambda *a, _f=pe._uncached: olds.append(threading.current_thread().name) or _f(*a))
    before = _state(cards.repo)
    code, result, out = _preview(cards, tmp_path, {"hp": 8, "note": "checked", "_by": "web"})
    assert code == 0 and _state(cards.repo) == before and bands == []
    made = {Path(p).name: Path(p).parent for p in seen}                     # the small copies alone: no duplicates or crops
    assert len(seen) == 2 and made[cp.BEFORE].parent == tmp_path / "runner" and (pe.PNG_LEVEL, pe.ON_MERGED) == (6, None)
    assert len(olds) == 1 and olds[0].startswith("old") and not pe.OLD_BESIDE
    assert Path(DOOMER_PNG).name + ".small" in made                         # preview.avif from the merge, beside the gates
    assert list((tmp_path / "runner").iterdir()) == []                         # the work dir is removed
    assert (result["state"], result["reason"], result["fields"], result["transient"]) == ("baked", "hp 5 -> 8", {"hp": "baked"}, False)
    assert result["request"] == REQUEST and result["card"] == "Mini Doomer" and result["path"] == DOOMER and result["png"] == DOOMER_PNG
    assert result["base"] == {"repo": cb.CARD_REPO, "commit": cards.head, "psd_blob": _sha(cards.files[DOOMER]),
                              "png_blob": _sha(cards.files[DOOMER_PNG])}
    assert result["edit"] == {"hp": 8, "note": "checked"} and result["changes"] == [{"field": "hp", "old": 5, "new": 8}]
    assert result["image"] == "baked" and result["files"] == {"webp": cp.SMALL, "before": cp.BEFORE} and result["parity"]["hp"]["bad_pct"] < 1
    assert len(result["region"]) == 4 and result["runner"]["psd_tools"] and result["at"].endswith("Z")
    webp = Image.open(out / cp.SMALL)
    assert webp.format == "AVIF" and webp.width == cp.SMALL_WIDTH
    cp._small((out / cp.PNG).read_bytes(), str(tmp_path / "again.avif"))   # as made from preview.png
    assert (tmp_path / "again.avif").read_bytes() == (out / cp.SMALL).read_bytes()
    with Image.open(out / cp.BEFORE) as before:                                # the official PNG, made the same way (the Diff's)
        assert before.format == "AVIF" and before.size == webp.size
        cp._small(cards.files[DOOMER_PNG], str(tmp_path / "official.avif"))
        assert (tmp_path / "official.avif").read_bytes() == (out / cp.BEFORE).read_bytes()

    # the bake's own dry run of the same edit: the same merged pixels
    (cards.repo / cb.EDITS_PATH).parent.mkdir(parents=True, exist_ok=True)
    (cards.repo / cb.EDITS_PATH).write_bytes(cb.dump_edits({"cards": {"Mini Doomer": {"hp": 8}}}).encode())
    _commit(cards.repo, "Website edit: Mini Doomer")
    report = cb.run_report(dry_run=True, card="Mini Doomer", local=str(cards.repo))
    assert [c.name for c in report.cards if c.baked] == ["Mini Doomer"] and {"dups", "crops"} <= set(seen) and bands
    bake = Image.open(report.cards[0].baked.png_path)
    ours = Image.open(out / cp.PNG)
    assert ours.mode == bake.mode and ours.tobytes() == bake.tobytes()


def test_a_refused_edit_writes_its_reason_and_no_image(cards, tmp_path):
    """Refused (old digit rows: past 10, owner 2026-11-08); the edit echoed exactly as sent (a cleared stat's null too: the
    site compares it)."""
    code, result, out = _preview(cards, tmp_path, {"hp": 25, "def": None})
    assert code == 0 and (result["state"], result["reason"]) == ("refused", "unsupported-card: hp 25: its bars show up to 10")
    assert result["edit"] == {"hp": 25, "def": None}
    assert result["fields"] == {"hp": "refused"} and result["image"] is None and result["files"] == {}
    assert not (out / cp.PNG).exists() and not (out / cp.SMALL).exists() and not (out / cp.BEFORE).exists()
    assert result["base"]["psd_blob"] == _sha(cards.files[DOOMER])


def test_a_duplicates_name_takes_its_psd(cards, tmp_path):
    """A picture-only duplicate's edit (Breakfast.png, dup_of Mini Doomer) goes to its PSD as its picture, as the bake's
    select() sends its name (dup_view: here no 'Breakfast #N' tag, so refused for that), its own PNG the preview's; with
    another field too (the website sends its PSD entry's: owner 11-13, a full picture 2)."""
    dup = "Auxiliary/Minions/Breakfast.png"
    (cards.repo / dup).write_bytes(cards.files[DOOMER_PNG])
    (cards.repo / cb.EDITS_PATH).parent.mkdir(parents=True, exist_ok=True)
    (cards.repo / cb.EDITS_PATH).write_bytes(cb.dump_edits({"cards": {"Breakfast": {"dup_of": "Mini Doomer"}}}).encode())
    _commit(cards.repo)
    code, result, _ = _preview(cards, tmp_path, {"name": "Second Breakfast"}, card="Breakfast")
    assert code == 0 and (result["state"], result["path"], result["png"], result["card"]) == ("refused", DOOMER, dup, "Breakfast")
    assert result["reason"].startswith("unsupported-card: no picture of Mini_Doomer.psd tagged 'Breakfast #N' prints Breakfast.png")
    code, result, _ = _preview(cards, tmp_path, {"name": "Second Breakfast", "hp": 3}, card="Breakfast", name="two")
    assert (result["state"], result["path"], result["png"]) == ("refused", DOOMER, dup) and "'Breakfast #N'" in result["reason"]


BAD_INPUT_IS_AN_ERROR_TO_FIX_NOT_TO_RETRY_CASES = [
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
]


def test_bad_input_is_an_error_to_fix_not_to_retry(cards, tmp_path):
    for card, edit, client_id, why in BAD_INPUT_IS_AN_ERROR_TO_FIX_NOT_TO_RETRY_CASES:
        code, result, out = _preview(cards, tmp_path, edit, card=card, request=client_id)
        assert code == 2 and result["state"] == "error" and result["transient"] is False and result["reason"].startswith(why)
        assert result["request"] == (client_id if cp.REQUEST.fullmatch(client_id) else None) and result["image"] is None


def test_a_name_or_ability_in_the_parts_takes_their_outcome():
    """The parts carry the name or ability they hold (never a job of its own): not 'refused' beside 'parts baked'; a
    field whose own boxes were skipped is not baked."""
    part = lambda f: {"field": f, "text": "t", "was": "w", "sha": "0" * 8, "psd": "0" * 8, "layer": "L"}
    edit = {"name": "A II", "ability": "Heal 3.", "hp": 5, "parts": {"19": part("name"), "23": part("ability")}}
    card = cb.Card("A", "a", edit, jobs={"parts": {}}, redundant=["hp"])
    card.baked, card.verdict = SimpleNamespace(jobs={"parts": {}}, edits=[]), {"parts": {"19": "baked", "23": "baked"}}
    assert cp.outcome(card)[3] == {"name": "baked", "ability": "baked", "hp": "redundant", "parts": "baked"}
    card.verdict["parts"]["23"] = "skipped"
    assert cp.outcome(card)[3] == {"name": "baked", "ability": "refused", "hp": "redundant", "parts": "baked"}
    card.baked, card.refusal = None, pe.Refusal("unsupported-card", "no")
    assert cp.outcome(card)[3] == {"name": "refused", "ability": "refused", "hp": "redundant", "parts": "refused"}


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


def test_the_early_reads_are_the_previews(cards, tmp_path, monkeypatch):
    """card_preview_reads (card-preview.yml's Reads step: the standard library alone) takes what the preview reads first
    (a key, a PSD path in another case, a numbered picture's parts or stars, a picture-only duplicate's any fields, an MDW PSD, an
    unknown card) and Card Edits.json (a duplicate's first: it names the PSD), each file at head (contents; git/blobs on a
    non-200), into a blobless clone or to a file per sha (--blobs), where the bake reads them; a record it cannot read is
    left out. It and its imports run on 3.12."""
    assert (early.EXCLUDE_FOLDERS, early.STATUS_CHART, early.RECORDS) == (cb.EXCLUDE_FOLDERS, pe.STATUS_CHART, (pe.RENDER_RECORD, pe.SPOT_RECORD, cp.VERDICTS_JSON))
    assert early.BLOBS_ENV == cb.BLOBS_ENV
    assert cp.VERDICTS_JSON == card_db.VERDICTS_JSON
    assert early.PICTURE_FIELDS == cp.PICTURE_FIELDS
    for python in filter(None, (executable, shutil.which("python3.12"))):     # the runner's python3: 3.12 (ubuntu-24.04)
        assert run_process([python, "-S", "-m", "commands.card_preview_reads", "--help"], cwd=str(REPO), capture_output=True).returncode == 0, python
    found = run_process([executable, "-S", "-c", "import sys, commands.card_preview_reads; print(*[getattr(m, '__file__', None) or '' for m in list(sys.modules.values())], sep='\\n')"],
                        cwd=str(REPO), capture_output=True, text=True, check=True).stdout.splitlines()
    closure = [Path(f) for f in found if f.startswith(f"{REPO}/")]
    assert REPO / "commands" / "card_preview_reads.py" in closure and len(closure) > 1, closure
    for path in closure:                                                        # 3.12's grammar, with or without a 3.12
        ast_parse(path.read_text(encoding="utf-8"), str(path), feature_version=(3, 12))
    dup, edits = "Auxiliary/Minions/Breakfast.png", {"Breakfast": {"dup_of": "Mini Doomer"}}
    cards.files.update({dup: b"a drawn duplicate", cb.EDITS_PATH: cb.dump_edits({"cards": edits}).encode()})
    for path in (dup, cb.EDITS_PATH):
        (cards.repo / path).parent.mkdir(parents=True, exist_ok=True)
        (cards.repo / path).write_bytes(cards.files[path])
    head = _commit(cards.repo)
    snapshot = cb.local_snapshot(cb.CARD_REPO, str(cards.repo))
    snapshot.tree[pe.STATUS_CHART] = ("100644", "0" * 40)                         # read for an ability, parts or colors edit
    tree = {path: sha for path, (_, sha) in snapshot.tree.items()}
    for card, fields in (("Mini Doomer", {"hp": 8}), (DOOMER.upper(), {"ability": "x"}), ("Mini Doomer", {"ability": None, "hp": 8}),
                         ("A Snack2", {"parts": {"1": {}}}), ("A Snack2", {"stars": 3}), ("A Snack2", {"colors": []}),
                         (MDW_DOOMER, {"hp": 8}), ("Nobody", {"hp": 8}),
                         ("Breakfast", {"name": "x"}), ("Breakfast", {"ability": "x", "hp": 3}),
                         ("Mini Doomer", {"colors": [{"field": "name", "color": "#ff2200"}]})):
        try:
            want = [p for p in cp.blobs_of(snapshot, cp.resolve(snapshot, card, fields)) if p in tree]
        except cp.BadInput:
            want = []
        assert early.paths_of(tree, card, fields, edits) == want, card
    assert early.paths_of(tree, "Breakfast", {"name": "x"}, edits) == [cb.ICON_MANIFEST_PATH, DOOMER, DOOMER_PNG, dup]
    assert early.paths_of(tree, "Mini Doomer", {"colors": [{"field": "name", "color": "#ff2200"}]})[-1] == pe.STATUS_CHART
    views = [cp.resolve(snapshot, "A Snack2", f).view for f in ({"stars": 3}, {"name": "x", "ability": "y"}, {"colors": []})]
    assert views == [2, None, 2]                # its derived text: refused as before; its own colours: as picture 2
    blobless = tmp_path / "blobless"
    _git(tmp_path, "clone", "-q", "--no-checkout", "--filter=blob:none", f"file://{cards.repo}", str(blobless))

    asked = []

    def get(url):
        from urllib.error import HTTPError
        from urllib.parse import unquote
        if "/contents/db/" in url:
            if pe.SPOT_RECORD in url:
                raise OSError("down")
            return b'{"x": {}}'
        if "/contents/" in url:
            path, ref = url.split("/contents/", 1)[1].split("?ref=")
            asked.append(unquote(path))
            if unquote(path) == DOOMER_PNG:
                raise HTTPError(url, 404, "Not Found", None, None)
            spec = f"{ref}:{unquote(path)}"
        else:
            asked.append(spec := url.rsplit("/", 1)[1])
        return run_process(["git", "-C", str(cards.repo), "cat-file", "blob", spec], capture_output=True, check=True).stdout
    monkeypatch.setattr(early, "_get", get)
    done = early.reads("Mini Doomer", {"hp": 8}, str(blobless), cb.CARD_REPO, head, str(tmp_path / "records"))
    blobs = [cb.ICON_MANIFEST_PATH, DOOMER, DOOMER_PNG, cb.EDITS_PATH]
    assert sorted(done) == sorted(blobs + [f"db/{pe.RENDER_RECORD}", f"db/{pe.SPOT_RECORD}", f"db/{cp.VERDICTS_JSON}"])
    assert [done[p][1] for p in blobs] == ["ok"] * 4 and done[f"db/{pe.SPOT_RECORD}"][1] == "OSError: down"
    assert sorted(asked) == sorted(blobs + [_sha(cards.files[DOOMER_PNG])])  # the 404's by sha
    assert sorted(p.name for p in (tmp_path / "records").iterdir()) == [pe.RENDER_RECORD, cp.VERDICTS_JSON]
    done = early.reads("Breakfast", {"name": "x"}, str(blobless), cb.CARD_REPO, head, str(tmp_path / "records"))
    assert {p: done[p][1] for p in blobs + [dup]} == {**dict.fromkeys(blobs, "in the clone"), cb.EDITS_PATH: "ok", dup: "ok"}
    for path in blobs + [dup]:
        assert cb._local_blob(str(blobless), _sha(cards.files[path])) == cards.files[path]
    other, kept = tmp_path / "other", tmp_path / "blobs"                    # --blobs: files the bake reads first
    _git(tmp_path, "clone", "-q", "--no-checkout", "--filter=blob:none", f"file://{cards.repo}", str(other))
    for _ in range(2):
        done = early.reads("Breakfast", {"name": "x"}, str(other), cb.CARD_REPO, head, str(tmp_path / "records"), str(kept))
    assert {p: done[p][1] for p in blobs + [dup]} == {**dict.fromkeys(blobs, "read already"), cb.EDITS_PATH: "ok", dup: "read already"}
    monkeypatch.setenv(cb.BLOBS_ENV, str(kept))
    monkeypatch.setattr(github_api, "fetch_verified", lambda *a, **k: pytest.fail("REST"))
    (kept / _sha(cards.files[DOOMER_PNG])).write_bytes(b"tampered")
    for path in blobs + [dup]:
        sha = _sha(cards.files[path])
        assert cb._local_blob(str(other), sha) is None
        if path == DOOMER_PNG:                  # a bad file: the clone's, then REST
            assert cb._kept_blob(sha) is None
            continue
        assert cb.fetch_verified(cb.CARD_REPO, sha, local=str(other)) == cards.files[path]


def test_the_preview_reads_the_verdict(tmp_path, monkeypatch):
    """r16: the preview bakes from where Card sync's bake would: the data's verdict (read beside the records) on the
    card's PSD gives the moved title and the smaller Ability; another PSD's, or none, nothing."""
    verdict = {"state": "baked", "psd": "abcdef12", "reason": "ability", cb.ANCHOR: -12, cb.FIT: 0.8836}
    (tmp_path / cp.VERDICTS_JSON).write_text(json.dumps({"schema": 1, "cards": {"mini doomer": verdict}}))
    monkeypatch.setenv(pe.RECORDS_DIR, str(tmp_path))
    for sha, want in (("abcdef1234", (-12, 0.8836)), ("99999999aa", (0, 1.0))):
        card = cb.Card("Mini Doomer", "mini doomer", {"ability": "x"}, DOOMER, sha)
        cp.remember(card)
        assert (card.title_anchor, card.ability_fit) == want
    monkeypatch.delenv(pe.RECORDS_DIR)
    cp.remember(card)


def test_fonts_and_records_after_the_wait(cards, clone, tmp_path, monkeypatch):
    """card-preview.yml starts the preview while the fonts restore and the reads run: nothing before ready()'s wait
    (READS_ENV's file) reads the records, nor before the first font's (WAIT_ENV's) the fonts, so ones that appear only
    then give the same preview (a name edit: drawn in MDW/Fonts alone, as on the runner)."""
    _, now, out = _preview(cards, tmp_path, {"name": "Maxi Doomer"}, name="now")
    late, record = tmp_path / "late", {"Elsewhere/A_Card.psd": {}}
    monkeypatch.setenv(cb.FONT_DIRS_ENV, str(late / "fonts"))
    monkeypatch.setenv(pe.RECORDS_DIR, str(late / "records"))
    monkeypatch.setenv(cp.WAIT_ENV, str(late / "go"))
    monkeypatch.setenv(cp.READS_ENV, str(late / "reads.done"))
    monkeypatch.setattr(pe, "_RECORDS", {})
    monkeypatch.setattr(pe, "_record_names", lru_cache(maxsize=2)(pe._record_names.__wrapped__))
    waits = []

    def arrive(seconds):                        # the reads end, then (at the first font) the fonts and the go file
        if not waits:
            assert pe._RECORDS == {} and pe.FONTS_WAIT is None
            (late / "records").mkdir(parents=True)
            (late / "records" / pe.RENDER_RECORD).write_text(json.dumps(record))
            (late / "reads.done").touch()
        elif len(waits) == 1:
            assert pe.FONTS_WAIT is not None
            (late / "fonts").symlink_to(clone / pe.FONTS_DIR)
            (late / "go").touch()
        waits.append(seconds)
    monkeypatch.setattr(cp, "sleep", arrive)
    code, result, late_out = _preview(cards, tmp_path, {"name": "Maxi Doomer"}, name="late")
    assert code == 0 and len(waits) == 2 and pe._RECORDS[pe.RENDER_RECORD] == record and result["runner"]["fonts"] is True
    assert pe.FONTS_WAIT is None
    same = ("state", "reason", "fields", "notes", "changes", "parity", "region", "image", "files", "size")
    assert {k: result.get(k) for k in same} == {k: now.get(k) for k in same} and now["state"] == "baked"
    for name in (cp.PNG, cp.SMALL, cp.BEFORE):
        assert (late_out / name).read_bytes() == (out / name).read_bytes(), name


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


def _out(tmp_path, request, state="baked", at=None, image=True) -> Path:
    out = tmp_path / "outs" / request
    out.mkdir(parents=True)
    result = {**br.blank(request, "Mini Doomer"), "card": "Mini Doomer", "state": state, "path": DOOMER, "at": at or br.stamp(), "reason": "hp 5 -> 8"}
    if image:
        result["files"] = {"png": br.PNG, "webp": br.SMALL, "before": br.BEFORE}
        (out / br.PNG).write_bytes(b"png of " + request.encode())
        (out / br.SMALL).write_bytes(b"webp of " + request.encode())
        (out / br.BEFORE).write_bytes(b"before of " + request.encode())
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


def test_publish_one_commit_per_preview(remote, tmp_path, monkeypatch):
    a, b = "aaaaaaaa-0001", "bbbbbbbb-0002"
    first = br.publish(str(_out(tmp_path, a)), a, str(remote.work))
    assert first["parent"] is None and not first["rewritten"]
    files = _branch(remote.bare)
    assert sorted(files) == sorted([br.README, br.INDEX] + [f"previews/{a}/{f}" for f in (br.BEFORE, br.SMALL, br.RESULT)])   # no PNG
    assert json.loads(files[f"previews/{a}/{br.RESULT}"])["request"] == a and files[f"previews/{a}/{br.SMALL}"] == b"webp of " + a.encode()
    second = br.publish(str(_out(tmp_path, b, state="refused", image=False)), b, str(remote.work))
    assert second["parent"] == first["commit"] and _git(remote.bare, "rev-parse", "previews").strip() == second["commit"]
    assert sorted(_index(remote.bare)) == [a, b] and list(_index(remote.bare)[b]) == ["at"]            # its date alone
    assert f"previews/{b}/{br.SMALL}" not in _branch(remote.bare)
    assert _git(remote.bare, "log", "-1", "--format=%an|%s", "previews").strip() == f"AutoCroissant|Preview {b}: Mini Doomer (refused)"
    assert _git(remote.bare, "rev-parse", "main").strip() == _git(remote.work, "rev-parse", "HEAD").strip()   # main untouched
    assert _git(remote.work, "status", "--porcelain") == ""                                                  # nor the checkout
    rerun = tmp_path / "rerun"                      # a re-run of a request replaces its folder
    shutil.copytree(_out(tmp_path, "cccccccc-0003", image=False), rerun)
    result = json.loads((rerun / br.RESULT).read_text())
    (rerun / br.RESULT).write_text(json.dumps({**result, "request": a, "state": "error"}))
    br.publish(str(rerun), a, str(remote.work))
    files = _branch(remote.bare)
    assert json.loads(files[f"previews/{a}/{br.RESULT}"])["state"] == "error" and f"previews/{a}/{br.SMALL}" not in files
    fresh, again = tmp_path / "fresh", tmp_path / "again"      # a re-run from a checkout without the old blobs
    _git(tmp_path, "clone", "-q", "--depth", "1", "--filter=blob:none", f"file://{remote.bare}", str(fresh))
    shutil.copytree(_out(tmp_path, "dddddddd-0004"), again)
    (again / br.RESULT).write_text(json.dumps({**json.loads((again / br.RESULT).read_text()), "request": a}))
    monkeypatch.setenv("GIT_NO_LAZY_FETCH", "1")
    run = br.run_process                            # HTTPS pushes thin whatever --no-thin says
    monkeypatch.setattr(br, "run_process", lambda cmd, **kw: run([{"--no-thin": "--thin"}.get(x, x) for x in cmd], **kw))
    br.publish(str(again), a, str(fresh))
    assert _branch(remote.bare)[f"previews/{a}/{br.SMALL}"] == b"webp of dddddddd-0004"


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

    def racing(where, commit, lease):
        if not calls:                               # the first push of ours: the other run lands first
            calls.append(None)
            request = f"race-0-{lease is not None:d}-00"
            calls[0] = br.publish(str(_out(tmp_path, request)), request, str(other))
        real(where, commit, lease)
    br.publish(str(_out(tmp_path, "base-0-0000")), "base-0-0000", str(remote.work))
    monkeypatch.setattr(br, "_push", racing)
    done = br.publish(str(_out(tmp_path, "mine-0-0000")), "mine-0-0000", str(remote.work))
    assert done["parent"] == calls[0]["commit"] and sorted(_index(remote.bare)) == ["base-0-0000", "mine-0-0000", "race-0-0-00"]

    calls.clear()
    monkeypatch.setattr(br, "KEEP", 3)               # the next one rewrites, pushed with a lease the other run breaks
    done = br.publish(str(_out(tmp_path, "mine-1-0000")), "mine-1-0000", str(remote.work))
    assert done["rewritten"] and "race-0-1-00" in _index(remote.bare) and "mine-1-0000" in _index(remote.bare)


def test_publish_on_a_warm_read(remote, tmp_path, monkeypatch):
    """--warm reads the branch: publish's first try fetches nothing. A head moved since: refused, then the usual path
    with no sleep; a rewrite's lease on the warm head likewise."""
    other = tmp_path / "other"
    _git(tmp_path, "clone", "-q", "--depth", "1", "--filter=blob:none", f"file://{remote.bare}", str(other))
    head = br.publish(str(_out(tmp_path, "base-0-0000")), "base-0-0000", str(remote.work))["commit"]
    real, fetched, naps = br._fetch, [], []
    monkeypatch.setattr(br, "_fetch", lambda *a: fetched.append(a) or real(*a))
    monkeypatch.setattr(br, "sleep", naps.append)
    assert br.main(["--warm", "--git", str(remote.work)]) == 0
    fetched.clear()
    assert br.publish(str(_out(tmp_path, "mine-0-0000")), "mine-0-0000", str(remote.work))["parent"] == head and not fetched

    br.warm(str(remote.work))
    fetched.clear()
    theirs = br.publish(str(_out(tmp_path, "race-0-0000")), "race-0-0000", str(other))["commit"]
    fetched.clear()
    assert br.publish(str(_out(tmp_path, "mine-1-0000")), "mine-1-0000", str(remote.work))["parent"] == theirs
    assert len(fetched) == 1 and not naps

    br.warm(str(remote.work))
    br.publish(str(_out(tmp_path, "race-1-0000")), "race-1-0000", str(other))
    monkeypatch.setattr(br, "KEEP", 4)              # the warm head rewrites, on a lease the other run broke
    assert br.publish(str(_out(tmp_path, "mine-2-0000")), "mine-2-0000", str(remote.work))["rewritten"]
    assert sorted(_index(remote.bare)) == ["base-0-0000", "mine-0-0000", "mine-1-0000", "mine-2-0000", "race-0-0000", "race-1-0000"] and not naps


def _fake_github(monkeypatch, listings: list, jobs: dict, blob: bytes, digest: str = "") -> list:
    """br's HTTPSConnection as GitHub: the run's artifact listings in turn (then the last), its jobs, the zip's 302 to
    another host, the blob there (its digest listed); a fake clock. Returns the requests, (host, path, headers)."""
    from hashlib import sha256
    row = {"id": 7, "name": br.ARTIFACT, "digest": "sha256:" + (digest or sha256(blob).hexdigest())}
    sent, clock = [], [0.0]

    class Reply:
        def __init__(self, status, body=b"", **headers):
            self.status, self.body, self.headers = status, body, headers

        def getheader(self, name, default=None):
            return self.headers.get(name.replace("-", "_"), default)

        def read(self, limit=None):
            return self.body

    class Connection:
        def __init__(self, host, timeout=None):
            self.host = host

        def request(self, method, path, headers):
            sent.append((self.host, path, headers))
            if self.host == "blob.example":
                self.reply = Reply(200, blob)
            elif path.endswith("/zip"):
                self.reply = Reply(302, Location="https://blob.example/a/b?sig=s")
            elif path.endswith("/jobs"):
                self.reply = Reply(200, json.dumps(jobs).encode())
            else:
                status, tag = listings.pop(0) if len(listings) > 1 else listings[0]
                self.reply = (Reply(304) if tag and headers.get("If-None-Match") == tag else
                              Reply(200, json.dumps({"artifacts": [row] if status else []}).encode(), ETag=tag))

        def getresponse(self):
            return self.reply

        def close(self):
            pass

    monkeypatch.setattr(br, "HTTPSConnection", Connection)
    monkeypatch.setattr(br, "sleep", lambda s: clock.__setitem__(0, clock[0] + s))
    monkeypatch.setattr(br, "monotonic", lambda: clock[0])
    return sent


def test_the_take_over(tmp_path, monkeypatch, capsys):
    """--take: the listing until the hand-over shows (If-None-Match: 304s), its zip through the 302, the blob store
    asked without the token, only HANDED's names written; a job that ended without a hand-over, or never validated;
    a zip past the cap or of another digest left to download-artifact."""
    import zipfile
    from io import BytesIO
    zipped = BytesIO()
    with zipfile.ZipFile(zipped, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in ((br.RESULT, b'{"state": "baked"}'), (br.SMALL, b"avif"), ("../x.avif", b"no"), ("a/" + br.BEFORE, b"no")):
            z.writestr(name, data)
    env = {"GITHUB_REPOSITORY": "o/r", "GITHUB_RUN_ID": "5", "GH_TOKEN": "tok", "GITHUB_API_URL": "https://api.example"}
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    running = {"jobs": [{"name": "preview", "status": "in_progress", "steps": []}]}
    sent = _fake_github(monkeypatch, [(0, '"e1"'), (0, '"e1"'), (1, '"e2"')], running, zipped.getvalue())
    assert br.main(["--take", str(tmp_path / "p")]) == 0
    assert capsys.readouterr().out.split() == ["go=true", "took=true"]
    assert sorted(f.name for f in (tmp_path / "p").iterdir()) == [br.SMALL, br.RESULT]
    assert [h.get("If-None-Match") for host, p, h in sent if "artifacts?" in p] == [None, '"e1"', '"e1"']
    assert [host for host, p, h in sent if "Authorization" in h] == ["api.example"] * 4 and sent[-1][0] == "blob.example"

    ended = {"jobs": [{"name": "preview", "status": "completed", "steps": [{"name": "Validate", "conclusion": "success"}]}]}
    for jobs, cap, digest, out in ((ended, br.ZIP_CAP, "", "go=true"), ({"jobs": [{"name": "preview", "status": "completed", "steps": []}]}, br.ZIP_CAP, "", ""),
                                   (running, 10, "", "go=true"), (running, br.ZIP_CAP, "0" * 64, "go=true")):
        _fake_github(monkeypatch, [(0, '"e1"')] if jobs is not running else [(1, '"e2"')], jobs, zipped.getvalue(), digest)
        monkeypatch.setattr(br, "ZIP_CAP", cap)
        assert br.main(["--take", str(tmp_path / "q")]) == 0 and capsys.readouterr().out.strip() == out
    assert not (tmp_path / "q").exists()


def test_the_direct_hand_over(tmp_path, monkeypatch, capsys):
    """--hand: CreateArtifact before result.json as upload-artifact v7's (proto names, the token's backend ids), then
    HANDED's files zipped, put without the token and finalized with the size and sha256 --take takes; hand.done ok.
    A finalize failing twice: fail; a preview that ended with no result: none. The token never printed."""
    import base64
    import io
    from hashlib import sha256
    claims = base64.urlsafe_b64encode(json.dumps({"scp": "Actions.GenericRead:1 Actions.Results:run-1:job-2"}).encode())
    token = f"h.{claims.decode().rstrip('=')}.sig"
    out = tmp_path / "preview"
    out.mkdir()
    sent, finalize = [], []

    class Connection:
        def __init__(self, host, timeout=None):
            self.host = host

        def connect(self):
            pass

        def request(self, method, path, body=None, headers=None):
            sent.append((self.host, method, path, headers, body, (out / br.RESULT).exists()))

        def getresponse(self):
            host, method, path, *_ = sent[-1]
            status, body = ((201, b"") if host == "blob.example" else
                            (200, {"ok": True, "signed_upload_url": "https://blob.example/c/a.zip?sv=1&sig=s"}) if path.endswith("CreateArtifact") else
                            finalize.pop(0) if finalize else (200, {"ok": True, "artifact_id": "9"}))
            return SimpleNamespace(status=status, read=lambda: body if isinstance(body, bytes) else json.dumps(body).encode(),
                                   getheader=lambda name, default=None: default)

        def close(self):
            pass

    def preview_writes(seconds):                    # the bake ends while the hand-over waits
        (out / br.SMALL).write_bytes(b"avif")
        (out / br.RESULT).write_text('{"state": "baked"}')
    monkeypatch.setattr(br, "HTTPSConnection", Connection)
    monkeypatch.setattr(br, "sleep", preview_writes)
    monkeypatch.setattr(br, "stdin", io.StringIO(f"{token}\nhttps://results.example/\n"))
    assert br.main(["--hand", str(out)]) == 0 and (tmp_path / br.HAND_DONE).read_text() == "ok\n"
    (create, put, done) = sent
    assert [(s[0], s[1], s[5]) for s in sent] == [("results.example", "POST", False), ("blob.example", "PUT", True), ("results.example", "POST", True)]
    body = json.loads(create[4])
    assert {k: body[k] for k in ("workflow_run_backend_id", "workflow_job_run_backend_id", "name", "version", "mime_type")} == {
        "workflow_run_backend_id": "run-1", "workflow_job_run_backend_id": "job-2", "name": br.ARTIFACT, "version": 7, "mime_type": "application/zip"}
    assert create[3]["Authorization"] == f"Bearer {token}" and "Authorization" not in put[3] and put[3]["x-ms-blob-type"] == "BlockBlob"
    assert put[2] == "/c/a.zip?sv=1&sig=s" and json.loads(done[4])["size"] == str(len(put[4]))
    digest = json.loads(done[4])["hash"]
    assert digest == "sha256:" + sha256(put[4]).hexdigest() and token not in capsys.readouterr().err

    running = {"jobs": [{"name": "preview", "status": "in_progress", "steps": []}]}     # --take takes that zip
    _fake_github(monkeypatch, [(1, '"e"')], running, put[4], digest[7:])
    assert br.take_over(str(tmp_path / "taken"), "o/r", "5", "tok") == "taken"
    assert sorted(f.name for f in (tmp_path / "taken").iterdir()) == [br.SMALL, br.RESULT]

    monkeypatch.setattr(br, "HTTPSConnection", Connection)
    for failing, ended, state in (([(500, {"msg": "busy"})] * 2, False, "fail"), ([], True, "none")):
        sent.clear()
        finalize[:] = failing
        for f in out.iterdir():
            f.unlink()
        if ended:
            (tmp_path / "preview.code").write_text("1\n")
        monkeypatch.setattr(br, "sleep", (lambda s: None) if ended else preview_writes)
        monkeypatch.setattr(br, "stdin", io.StringIO(f"{token}\nhttps://results.example/\n"))
        assert br.main(["--hand", str(out)]) == 0 and (tmp_path / br.HAND_DONE).read_text() == f"{state}\n"
        assert [s[1] for s in sent] == (["POST"] if ended else ["POST", "PUT", "POST", "POST"])
    assert token not in capsys.readouterr().err


# ========================
# The warm lane
# ========================
def test_a_lanes_title_and_what_it_takes(monkeypatch):
    """The title names the lane and the press's blob; read_press takes the blob once it is in (a 404 tried again), only
    when it hashes to its sha and names its request; fits() takes only a press named to this run, by its editor, on its
    commit, not done; listen() ends idle or at its time, saying so."""
    import base64
    import textwrap
    import commands.card_preview_lane as lane_mod
    from time import time
    from commands.card_preview_lane import fits, press_blob, press_of, read_press
    api = SimpleNamespace(left=None, get=lambda path, conditional=False: (304, None, b""))
    monkeypatch.setattr(lane_mod, "LANE_IDLE", 0.05)
    monkeypatch.setattr(lane_mod, "POLL", 0.01)
    monkeypatch.setattr(lane_mod, "BLOB_WAIT", 0.2)
    monkeypatch.setattr(lane_mod, "BLOB_POLL", 0.01)
    assert lane_mod.listen(api, 100, "ed", until=time() + 60, repository="o/r", head="h") == "idle 0.05 s"
    assert lane_mod.listen(api, 100, "ed", until=time() - 1, repository="o/r", head="h") == "the job's end is near"
    edit = '{"ability":"Deal 2 {fire} damage.\\nOnce per turn: draw.","hp":7}'
    data, sha = press_blob(REQUEST, "Odd {Card}", edit)
    press = press_of(f"Card preview {REQUEST} lane 100-{sha}")
    assert press == (REQUEST, 100, sha) and press_of(f"Card preview {REQUEST}") is None
    assert press_of(f"Card preview {REQUEST} lane 100-{sha[:16]} Odd {{Card}} {edit}") is None     # the old titles: none
    blobs, asked = {}, []

    def get(path, conditional=False):
        asked.append(path)
        blob = blobs.get(path.rsplit("/", 1)[1])
        return (404, None, b"") if blob is None else (200, None, json.dumps(blob).encode())
    store = SimpleNamespace(left=None, get=get)
    blobs[sha] = {"sha": sha, "size": len(data), "encoding": "base64",
                  "content": "\n".join(textwrap.wrap(base64.b64encode(data).decode(), 60))}    # GitHub's 60-column lines
    assert read_press(store, "o/r", press) == ("Odd {Card}", edit, "")
    assert read_press(store, "o/r", press._replace(request="other-request")) == (None, None, "the press's blob names another request")
    blobs[sha] = {**blobs[sha], "content": base64.b64encode(data.replace(b"7", b"8")).decode()}
    assert read_press(store, "o/r", press) == (None, None, "the press's blob doesn't hash to its sha")
    asked.clear()
    assert read_press(store, "o/r", press._replace(blob="0" * 40)) == (None, None, "the press's blob not read (404)")
    assert len(asked) > 2                                                      # tried again for BLOB_WAIT
    run = {"status": "queued", "triggering_actor": {"login": "ed"}, "head_sha": "h"}
    assert fits(run, press, 100, "ed") == "" == fits({**run, "head_sha": "g"}, press, 100, "ed")    # its commit: follow
    assert [fits({**run, **change}, p, 100, "ed") for change, p in (
        ({}, None), ({}, press._replace(lane=99)), ({"status": "completed"}, press),
        ({"triggering_actor": {"login": "other"}}, press))] == [
        "a newer press named no lane", "a newer press named lane 99", "a newer press was done before the lane saw it",
        "a newer press by another editor"]


def test_the_lanes_doorbell(tmp_path, monkeypatch):
    """A press its editor rang at the lane's doorbell (a commit status at the lane's first commit: '<run> <request>
    <blob>') is taken before its run is listed (a request rung with two blobs, another editor's, another context's or
    another lane's is none), its run counts as served once listed, and a newer run it doesn't take ends the lane as
    ever; a rung press's take checks the card repo's main in place of the run's commit and bakes with no run (on a newer
    main: there, once moved() follows it); its blob is read beside those checks."""
    import base64
    import commands.card_preview_lane as lane_mod
    from time import time
    data, sha = lane_mod.press_blob(REQUEST, "Mini Doomer", '{"hp": 7}')
    bell = lambda text, who="ed", context=lane_mod.CONTEXT: {"context": context, "creator": {"login": who}, "description": text}
    statuses = [bell("100 next abcdef12"), bell(f"100 {REQUEST} {sha}"), bell(f"100 rung-twice-1 {'b' * 40}"),
                bell(f"100 rung-twice-1 {'c' * 40}"), bell(f"1001 other-lane-1 {sha}"), bell(f"100 other-editor {sha}", "eve"),
                bell(f"100 other-context {sha}", context="ci"), bell(f"100 next {'d' * 8}", "eve")]
    runs, took, main, asked, checked = [], [], {"o/r": "1" * 40, "bot/repo": "bot"}, [], threading.Event()

    def get(path, conditional=False):
        if path.endswith("/git/ref/heads/main"):
            checked.set()
        elif path.endswith(sha) and took:
            assert checked.wait(10)             # the blob beside the checks
        body = (statuses if path == "/repos/o/r/commits/h/statuses?per_page=100"
                else {"workflow_runs": runs} if path.endswith(f"/runs?event=workflow_dispatch&actor=ed&per_page={lane_mod.LISTED}")
                else {"sha": sha, "size": len(data), "encoding": "base64", "content": base64.b64encode(data).decode()}
                if path.endswith(sha) else {"object": {"sha": main[path.split("/repos/")[1].split("/git/")[0]]}})
        return 200, None, json.dumps(body).encode()
    api = SimpleNamespace(left=None, get=get)

    def take(api, press, run, **press_kw):
        took.append((press, run))
        asked.append(press_kw["hints"].asked)
        runs[:] = [{"id": 102, "display_title": "Card preview another-press-1"},
                   {"id": 101, "display_title": f"Card preview {REQUEST} lane 100-{sha}"}]
        return ""
    real_take = lane_mod.take
    monkeypatch.setattr(lane_mod, "take", take)
    monkeypatch.setattr(lane_mod, "POLL", 0.01)
    assert lane_mod.listen(api, 100, "ed", until=time() + 60, repository="o/r", head="h") == "a newer press named no lane"
    assert took == [((REQUEST, 100, sha), None)] and asked == [{"abcdef12"}]     # a next card: read beside
    baked = []
    monkeypatch.setattr(lane_mod, "read_card", lambda card, fields, clone, repository, head, blobs, pending=None: {})
    monkeypatch.setattr(lane_mod, "read_records", lambda repository, records: {})
    keys = dict(bake=lambda card, edit, out, repository, **k: baked.append((card, edit, k["run"], k["head"])) or {"state": "baked"},
                lane=str(tmp_path), clone="", repository="o/r", head="1" * 40, kept={}, bot_repo="bot/repo", bot="bot", limit=99,
                sides=(api, api))
    monkeypatch.setattr(lane_mod, "RECORDS", ())
    assert real_take(api, took[0][0], None, **keys) == "" and baked == [("Mini Doomer", b'{"hp": 7}', "", "1" * 40)]
    main["o/r"] = "2" * 40
    assert real_take(api, took[0][0], None, **keys) == "the card repo's main moved" and len(baked) == 1
    moves = []                                  # a newer main followed: baked there
    assert real_take(api, took[0][0], None, **keys, moved=lambda new: moves.append(new) or "") == "" and moves == ["2" * 40]
    assert baked[-1][-1] == "2" * 40
    assert real_take(api, took[0][0], None, **keys, moved=lambda new: "the card repo moved (not fetched)") == "the card repo moved (not fetched)"
    assert len(baked) == 2


def test_the_lane_follows_main(tmp_path, monkeypatch):
    """follow() fetches a newer commit's trees into the lane's shallow blobless clone and takes it when KEPT is the same
    there, not when its fonts moved or it can't be fetched; listen() takes a run on a newer commit there, the lane's
    commit from then on."""
    import commands.card_preview_lane as lane_mod
    from time import time
    origin, lane, heads = tmp_path / "origin", tmp_path / "lane", []
    _git(tmp_path, "init", "-q", "origin")
    for path, data in (("MDW/Fonts/A.ttf", "font"), ("Cards/X.psd", "x2"), ("MDW/Fonts/A.ttf", "font2")):
        (origin / path).parent.mkdir(parents=True, exist_ok=True)
        (origin / path).write_text(data)
        _git(origin, "add", "-A")
        _git(origin, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", path)
        heads.append(_git(origin, "rev-parse", "HEAD").strip())
    _git(origin, "config", "uploadpack.allowFilter", "true")
    _git(origin, "config", "uploadpack.allowAnySHA1InWant", "true")
    _git(tmp_path, "init", "-q", "lane")
    _git(lane, "remote", "add", "origin", f"file://{origin}")
    _git(lane, "fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", heads[0])     # as card-preview.yml's Clone
    a, b, c = heads
    assert lane_mod.follow(str(lane), a, a) == "" == lane_mod.follow(str(lane), a, b)
    assert _git(lane, "rev-parse", f"{b}:Cards", env={"GIT_NO_LAZY_FETCH": "1"}).strip()     # its trees: no blob
    assert lane_mod.follow(str(lane), b, c) == "the card repo moved (its fonts or the preview's workflow)"
    assert lane_mod.follow(str(lane), a, "0" * 40) == "the card repo moved (not fetched)"
    assert lane_mod.follow(str(lane), a, "main") == "the card repo moved"
    runs, took, follows = [{"id": 101, "status": "queued", "triggering_actor": {"login": "ed"}, "head_sha": b,
                            "display_title": f"Card preview {REQUEST} lane 100-{'e' * 40}"}], [], []

    def take(api, press, run, **kw):
        took.append(kw["head"])
        runs.append({"id": 102, "display_title": "Card preview another-press-1"})
        return ""
    monkeypatch.setattr(lane_mod, "take", take)
    monkeypatch.setattr(lane_mod, "follow", lambda clone, head, new: follows.append((head, new)) or "")
    monkeypatch.setattr(lane_mod, "_say", lambda text: None)
    monkeypatch.setattr(lane_mod, "POLL", 0.01)
    paths = []
    api = SimpleNamespace(left=None, get=lambda path, conditional=False: paths.append(path) or (
        200, None, json.dumps({"workflow_runs": runs} if "/runs?" in path else []).encode()))
    assert lane_mod.listen(api, 100, "ed", until=time() + 60, repository="o/r", head=a, clone=str(lane)) == "a newer press named no lane"
    assert follows == [(a, b)] and took == [b]
    assert {p for p in paths if "/statuses" in p} == {f"/repos/o/r/commits/{a}/statuses?per_page=100"}    # its doorbell stays


def test_the_lanes_next_cards(tmp_path, monkeypatch):
    """A card rung as next (a status '<run> next <PSD blob>'; no press) is read in the background, once and
    not while a press is taken: its PSD, PNG and the status chart into the clone; a press whose read is still going
    joins it (each blob read once)."""
    import commands.card_preview_lane as lane_mod
    from time import sleep
    psd, png = "Creatures/Mario/1 Stars/Goomba.psd", "Creatures/Mario/1 Stars/Goomba.png"
    files = {psd: b"psd", png: b"png", early.STATUS_CHART: b"chart", "Creatures/Mario/1 Stars/Koopa.psd": b"other"}
    clone = tmp_path / "clone"
    _git(tmp_path, "init", "-q", "clone")
    for path, data in files.items():
        (clone / path).parent.mkdir(parents=True, exist_ok=True)
        (clone / path).write_bytes(data)
    _git(clone, "add", "-A")
    _git(clone, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "c")
    head, shas = _git(clone, "rev-parse", "HEAD").strip(), {p: _sha(d) for p, d in files.items()}
    for sha in shas.values():                   # a blobless clone's
        (clone / ".git" / "objects" / sha[:2] / sha[2:]).unlink()
    served, gate, data = [], threading.Event(), {_sha(d): d for d in files.values()}
    reading, joining, joined = threading.Event(), threading.Event(), early.joined

    def get(url):
        if "/contents/db/" in url:
            return b"{}"
        served.append(url.rsplit("/", 1)[1])
        if served[-1] == shas[psd]:
            reading.set()
            gate.wait(60)
        return data[served[-1]]
    monkeypatch.setattr(early, "_get", get)
    monkeypatch.setattr(early, "joined", lambda *args: joining.set() or joined(*args))
    monkeypatch.setattr(lane_mod, "_say", lambda text: None)
    body = json.dumps([{"context": lane_mod.CONTEXT, "creator": {"login": "ed"}, "description": text} for text in (
        f"100 {REQUEST} {'a' * 40}", "101 next abcdef12", f"100 next {shas[psd][:8]}")]).encode()
    assert lane_mod.hints_of(body, 100, "ed") == [shas[psd][:8]] and lane_mod.rung(body, 100, "ed") == [(REQUEST, 100, "a" * 40)]
    hints = lane_mod.Warm(str(clone), "o/r", head)
    hints.idle.clear()                          # a press being taken
    hints.ask(lane_mod.hints_of(body, 100, "ed") * 2)
    sleep(0.3)
    assert served == []
    hints.idle.set()
    assert reading.wait(60)                     # events, not fixed times (slow under load)
    done = []
    press = threading.Thread(target=lambda: done.append(early.reads(psd, {"ability": "x"}, str(clone), "o/r", head, str(tmp_path / "r"),
                                                                    pending=hints.pending)))
    press.start()
    joining.wait(60)
    gate.set()
    press.join(60)
    assert joining.is_set() and sorted(served) == sorted(shas[p] for p in (psd, png, early.STATUS_CHART))
    assert early.missing(str(clone), list(shas.values())) == {shas["Creatures/Mario/1 Stars/Koopa.psd"]}
    assert done[0][psd][1] == "ok"


def test_the_warm_lane(cards, tmp_path, monkeypatch):
    """--code then --lane against a stand-in API: the first press's code is written before the lane; the lane takes its
    editor's next press named to it (title, blob, commit, bot's main and records checked; its out dir, then <request>.code),
    bakes it as a cold preview does (the same result less took_s, at and run, the same images; run: the press's own),
    and ends at a newer press that names no lane, whose dir it never makes."""
    import base64
    import commands.card_preview_lane as lane_mod
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from time import time
    bot = run_process(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True,  # as the lane: "" in a copy
                      timeout=30).stdout.strip()
    records = tmp_path / "records"
    records.mkdir()
    for name in early.RECORDS:
        (records / name).write_text("{}\n")
    second, third = "0f8e7d6c-5b4a-4321-9876-00000000aaaa", "0f8e7d6c-5b4a-4321-9876-00000000bbbb"
    edit = '{"hp": 7}'
    data, sha = lane_mod.press_blob(second, "Mini Doomer", edit)
    blob = json.dumps({"sha": sha, "size": len(data), "encoding": "base64", "content": base64.b64encode(data).decode()})
    runs = [{"id": 99, "display_title": "Card preview older-press-1"},
            {"id": 101, "status": "queued", "triggering_actor": {"login": "web-ed"}, "head_sha": cards.head,
             "created_at": "2026-11-16T00:00:00Z", "display_title": f"Card preview {second} lane 100-{sha}"},
            {"id": 102, "status": "queued", "triggering_actor": {"login": "web-ed"}, "head_sha": cards.head,
             "display_title": f"Card preview {third}"}]
    asked = []

    class Api(BaseHTTPRequestHandler):
        def do_GET(self):
            path = self.path.split("?")[0]
            body = (json.dumps({"workflow_runs": runs[::-1]}) if path.endswith("/runs") else
                    json.dumps({"object": {"sha": bot}}) if path.endswith("/git/ref/heads/main") else
                    blob if path.endswith(f"/git/blobs/{sha}") else
                    (records / path.rsplit("/", 1)[1]).read_text() if "/contents/db/" in path else None)
            tag = f'"{hash(body)}"'
            status = 404 if body is None else 304 if self.headers.get("If-None-Match") == tag else 200
            asked.append((path.rsplit("/", 1)[1], status))
            self.send_response(status)
            self.send_header("ETag", tag)
            self.end_headers()
            if status == 200:
                self.wfile.write(body.encode())

        def log_message(self, *args):
            pass
    import _socket
    import socket

    def loopback(self, address, _real=_socket.socket.connect):              # the stand-in alone (offline's refusal else)
        assert address[0] == "127.0.0.1", "network access in an offline test"
        return _real(self, address)
    monkeypatch.setattr(socket.socket, "connect", loopback)
    monkeypatch.setattr(socket, "getaddrinfo", lambda host, *a, **k: _socket.getaddrinfo("127.0.0.1", *a, **k) if host == "127.0.0.1" else loopback(None, (host,)))
    server = ThreadingHTTPServer(("127.0.0.1", 0), Api)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        for key, value in {"GITHUB_RUN_ID": "100", "GITHUB_TRIGGERING_ACTOR": "web-ed", "GITHUB_TOKEN": "tok",
                           "GITHUB_API_URL": base, lane_mod.RECORDS_ENV: str(records)}.items():
            monkeypatch.setenv(key, value)
        monkeypatch.setattr(early, "API", base + "/repos/")
        monkeypatch.setattr(pe, "_RECORDS", {})
        monkeypatch.setattr(lane_mod, "_log_to", lambda path: None)
        monkeypatch.setattr(lane_mod, "POLL", 0.01)
        (tmp_path / "hand.done").write_text("ok\n")         # Hand's first hand-over
        (tmp_path / "first.json").write_text('{"hp": 8}')
        codes = []
        monkeypatch.setattr(lane_mod, "take", lambda *a, _f=lane_mod.take, **k: codes.append((tmp_path / "code").read_text()) or _f(*a, **k))
        code = cp.main(["--card", "Mini Doomer", "--edit", str(tmp_path / "first.json"), "--out", str(tmp_path / "out"),
                        "--clone", str(cards.repo), "--request", REQUEST, "--head", cards.head, "--code", str(tmp_path / "code"),
                        "--lane", str(tmp_path / "lane"), "--lane-until", str(time() + 300)])
    finally:
        server.shutdown()
        server.server_close()
    assert code == 0 and codes == ["0\n"] and (tmp_path / "lane.end").read_text() == "a newer press named no lane\n"
    assert sorted(p.name for p in (tmp_path / "lane").iterdir()) == [second, f"{second}.code"]
    assert (tmp_path / "lane" / f"{second}.code").read_text() == "0\n"
    listed = [a for a in asked if a[0] in ("runs", "main", sha)]     # the press's blob beside the bot's main (its own connection)
    assert listed[:2] + sorted(listed[2:4]) + listed[4:] == [("main", 200), ("runs", 200)] + sorted([(sha, 200), ("main", 200)]) + [("runs", 304)]
    warm = json.loads((tmp_path / "lane" / second / cp.RESULT).read_text())
    assert warm["run"] == f"https://github.com/{cb.CARD_REPO}/actions/runs/101" and warm["reason"] == "hp 5 -> 7"
    monkeypatch.delenv("GITHUB_RUN_ID")
    _, cold, out = _preview(cards, tmp_path, edit.encode(), request=second, name="cold")
    assert {k: v for k, v in warm.items() if k not in ("took_s", "at", "run")} == {k: v for k, v in cold.items() if k not in ("took_s", "at", "run")}
    for name in (cp.SMALL, cp.BEFORE):
        assert (tmp_path / "lane" / second / name).read_bytes() == (out / name).read_bytes()


def test_the_lanes_hand_overs(tmp_path, monkeypatch):
    """--hand ... --lane: after the first hand-over, each press's dir in the lane as card-preview-<request> (created at
    once, finalized once its result.json is in); one the lane skipped (its .code, no result) is never finalized; it ends
    at lane.end, hand.lane saying how many it handed over."""
    import base64
    import io
    claims = base64.urlsafe_b64encode(json.dumps({"scp": "Actions.Results:run-1:job-2"}).encode()).decode().rstrip("=")
    lane, first = tmp_path / "lane", tmp_path / "preview"
    for out in (first, lane / "req-0000a", lane / "req-0000b"):
        out.mkdir(parents=True)
    for out in (first, lane / "req-0000a"):
        (out / br.RESULT).write_text('{"state": "baked"}')
    (lane / "req-0000b.code").write_text("skipped\n")
    (tmp_path / "lane.end").write_text("idle 240 s\n")
    sent = []

    class Connection:
        def __init__(self, host, timeout=None):
            self.host = host

        def connect(self):
            pass

        def request(self, method, path, body=None, headers=None):
            sent.append((self.host, path.rsplit("/", 1)[-1], json.loads(body).get("name") if self.host == "r.example" else None))

        def getresponse(self):
            body = {"ok": True, "signed_upload_url": "https://blob.example/a?sig=s"}
            return SimpleNamespace(status=201 if sent[-1][0] == "blob.example" else 200, read=lambda: json.dumps(body).encode(),
                                   getheader=lambda name, default=None: default)

        def close(self):
            pass
    monkeypatch.setattr(br, "HTTPSConnection", Connection)
    monkeypatch.setattr(br, "stdin", io.StringIO(f"h.{claims}.s\nhttps://r.example/\n"))
    assert br.main(["--hand", str(first), "--lane", str(lane)]) == 0
    assert (tmp_path / br.HAND_DONE).read_text() == "ok\n" and (tmp_path / br.LANE_HANDED).read_text() == "1\n"
    assert [s for s in sent if s[1] == "CreateArtifact"] == [("r.example", "CreateArtifact", n) for n in (
        br.ARTIFACT, br.LANE_ARTIFACT + "req-0000a", br.LANE_ARTIFACT + "req-0000b")]
    assert [s[1] for s in sent].count("FinalizeArtifact") == 2 and [s[0] for s in sent].count("blob.example") == 2


# ========================
# card-preview.yml
# ========================
def _workflow() -> tuple:
    text = (WORKFLOWS / "card-preview.yml").read_text(encoding="utf-8")
    return text, yaml.safe_load(text)


def _expand(value: str, inputs: dict, workspace: Path) -> str:
    """The ${{ }} expressions the steps' env use; any other fails the test (the runner would read it, this would not)."""
    known = {"inputs.card": inputs["card"], "inputs.edit": inputs["edit"], "inputs.request": inputs["request"],
             "inputs.lane": inputs.get("lane", ""), "secrets.GITHUB_TOKEN": "", "github.workspace": str(workspace)}

    def one(match):
        assert match.group(1) in known, f"the job runner here does not know ${{{{ {match.group(1)} }}}}"
        return known[match.group(1)]
    return re.sub(r"\$\{\{ ([\w.]+) \}\}", one, str(value))


def _run_jobs(text: str, inputs: dict, workspaces: dict, bots: dict, tmp_path: Path, name: str) -> dict:
    """
    The workflow's two jobs as the runner runs them (bash -e -o pipefail; each job its own workspace, RUNNER_TEMP, env and
    outputs). The preview job: Validate, Reads, Start, Go (the fonts already in) and Preview, then Handed (Hand never
    ran) and the hand-over when its `if` holds (a copy of its listed files that exist, as upload-artifact with if-no-files-found:
    ignore). The publish
    job when the preview job's `ok` output is "true": the take-over (the copy back; continue-on-error) and Publish (its
    `python -S`: this Python without site-packages). Returns {step: exit code, None when skipped}.
    """
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
    (runner / "cards.code").write_text("0\n")         # Clone's card fetch: the workspace is the card repo
    (runner / "bot.code").write_text("0\n")           # and its bot clone, in bot/
    if not (workspaces["preview"] / "bot").exists():
        (workspaces["preview"] / "bot").symlink_to(bots["preview"])
    for step_name in ("Validate", "Reads", "Start", "Go", "Preview"):
        skipped = any(codes.values())
        codes[step_name] = None if skipped else run_step("preview", steps["preview"][step_name], runner)
    if "Handed" in steps["preview"] and codes["Validate"] == 0:     # Hand (a JavaScript action) never ran: not ok
        assert run_step("preview", steps["preview"]["Handed"], runner) == 0
    hand = steps["preview"]["Hand over"]
    assert hand["if"] in ("${{ !cancelled() && steps.validate.outcome == 'success' }}",
                          "${{ !cancelled() && steps.validate.outcome == 'success' && steps.handed.outputs.done != 'ok' }}")
    found = [Path(f.strip().replace("${{ runner.temp }}", str(runner))) for f in hand["with"]["path"].strip().splitlines()] if codes["Validate"] == 0 else []
    for path in [f for f in found if f.is_file()]:
        artifact.mkdir(parents=True, exist_ok=True)
        shutil.copy(path, artifact / path.name)
    outputs = dict(line.partition("=")[::2] for line in (runner / "output").read_text().splitlines()) if (runner / "output").exists() else {}
    if outputs.get("ok") == "true":                     # the publish job's wait: go once a validated preview ends
        runner = tmp_path / name / "publish"
        runner.mkdir(parents=True)
        codes["Take over"] = 0 if artifact.is_dir() else 1
        if artifact.is_dir():
            shutil.copytree(artifact, runner / "preview")
        codes["Publish"] = run_step("publish", steps["publish"]["Publish"], runner)
    return codes


def test_the_jobs_steps_run_here(cards, remote, clone, tmp_path):
    """card-preview.yml's own shell against a stand-in GitHub: each job's workspace a sparse clone of the card repo (the
    preview's with MDW/Fonts as the font cache restores it), the bots Action-shaped copies. A stat edit is published on
    previews (the WebP, not the PNG); a bad request fails Validate and publishes nothing; main never moves."""
    text, doc = _workflow()
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
        bot = next(s for s in doc["jobs"][job]["steps"] if s.get("with", {}).get("repository") in ("bobberlington/AutoCroissant", "${{ env.BOT_REPO }}"))
        assert bot_files(doc["jobs"][job], bot) == ["/" + f for f in files], job     # the job's checkout: that list
        bots[job] = tmp_path / f"bot-{job}"
        for rel in files:
            (bots[job] / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(REPO / rel, bots[job] / rel)
    main = _git(remote.bare, "rev-parse", "main").strip()

    codes = _run_jobs(text, {"card": "Mini Doomer", "edit": '{"hp": 8}', "request": REQUEST}, workspaces, bots, tmp_path, "run1")
    assert codes == {"Validate": 0, "Reads": 0, "Start": 0, "Go": 0, "Preview": 0, "Take over": 0, "Publish": 0}
    files = _branch(remote.bare)
    result = json.loads(files[f"previews/{REQUEST}/{br.RESULT}"])
    assert (result["state"], result["reason"], result["base"]["commit"]) == ("baked", "hp 5 -> 8", main)
    assert files[f"previews/{REQUEST}/{br.SMALL}"][4:12] == b"ftypavif" and f"previews/{REQUEST}/{br.PNG}" not in files
    assert files[f"previews/{REQUEST}/{br.BEFORE}"][4:12] == b"ftypavif"
    assert result["files"] == {"webp": br.SMALL, "before": br.BEFORE} and result["runner"]["fonts"] is True
    assert "**Baked**: hp 5 -> 8" in (tmp_path / "run1" / "preview" / "summary.md").read_text()
    assert (tmp_path / "run1" / "preview" / "lane.end").read_text() == "not on a runner\n"     # no run id: no lane
    assert "Published `previews/" in (tmp_path / "run1" / "publish" / "summary.md").read_text()

    before = _git(remote.bare, "rev-parse", "previews").strip()
    codes = _run_jobs(text, {"card": "Mini Doomer", "edit": '{"hp": 8}', "request": "x; rm -rf /"}, workspaces, bots, tmp_path, "run4")
    assert codes == {"Validate": 1, "Reads": None, "Start": None, "Go": None, "Preview": None}
    assert _git(remote.bare, "rev-parse", "previews").strip() == before and _git(remote.bare, "rev-parse", "main").strip() == main
    assert sorted(_index(remote.bare)) == [REQUEST]

