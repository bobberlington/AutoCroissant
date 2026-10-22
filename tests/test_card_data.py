"""card_data: the Action's sparse copy (--help, the import closure), the gates G1-G6, the exit codes, and offline builds
on a temporary git repo of golden PSDs (new, changed, moved, archived, an unreadable PSD, the budget, restore)."""
import json
import os
import pickle
import re
import shutil
import subprocess
from subprocess import run as run_process
from sys import executable
from types import SimpleNamespace

import pytest

import commands.card_data as cd
import commands.card_db as cdb
import commands.card_models as cm
import commands.changelog as cl
from conftest import REPO, blob_sha_bytes
from golden_cards import VOCAB, golden_file

DOOMER = "Auxiliary/Minions/Mini_Doomer.psd"
FREEZER = "Field/1 Stars/The_Freezer.psd"
OTHER = "Auxiliary/Minions/Anubisath_Guardian.psd"
BROKEN = b"8BPS" + b"\0" * 30


# ========================
# The Action's sparse copy
# ========================
def test_sparse_copy_prints_help_and_holds_its_imports(tmp_path):
    """Exactly SPARSE_FILES, no config.py: --help works, and the build's modules (lazy ones too) are that list, with no
    Discord, pandas or the bot's modules."""
    bot = tmp_path / "bot"
    for rel in cd.SPARSE_FILES:
        assert (REPO / rel).is_file(), f"{rel} is in SPARSE_FILES but missing here"
        (bot / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO / rel, bot / rel)
    done = run_process([executable, "-m", "commands.card_data", "--help"], cwd=bot, capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    assert done.stdout.startswith("usage: python -m commands.card_data")
    assert set(re.findall(r"--[a-z-]+", done.stdout)) == {
        "--repo", "--checkout", "--head", "--blobs", "--db", "--seed", "--aliases", "--full", "--cards", "--allow-archive",
        "--restore", "--budget", "--run", "--summary", "--message", "--dry-run", "--help"}
    code = ("import sys, commands.card_data, commands.card_history, commands.card_db, commands.changelog, commands.psd_details; "
            "print(sorted(m for m in ('config', 'discord', 'pandas', 'github', 'commands.psd_analyzer', 'commands.repo_traverser', "
            "'commands.query_card', 'commands.utils', 'commands.type_emojis', 'commands.card_runner') if m in sys.modules)); "
            "print(sorted(m for m in sys.modules if m.startswith('commands.') or m == 'global_config'))")
    done = run_process([executable, "-c", code], cwd=bot, capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    bad, closure = done.stdout.splitlines()
    assert bad == "[]"
    modules = {m.replace(".", "/") + ".py" for m in json.loads(closure.replace("'", '"'))}
    assert modules == {f for f in cd.SPARSE_FILES if f.endswith(".py")}                   # card-data.yml copies this list
    from test_card_bake import ALLOWED_IMPORTS, third_party_imports
    extra = third_party_imports("import commands.card_data, commands.card_history, commands.changelog, commands.psd_details", bot)
    assert not extra - ALLOWED_IMPORTS, f"requirements-card-data.txt installs none of these: {sorted(extra - ALLOWED_IMPORTS)}"


def test_requirements_pin_and_not_ignored():
    lines = [x.strip() for x in (REPO / "requirements-card-data.txt").read_text(encoding="utf-8").splitlines()
             if x.strip() and not x.startswith("#")]
    assert lines == ["psd-tools==1.25.0", "fonttools", "requests"]
    status = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--ignored", "requirements-card-data.txt"],
                            capture_output=True, text=True).stdout
    assert not status.startswith("!!"), "requirements-card-data.txt is gitignored (*.txt): .gitignore needs its ! line"


def test_build_modules_stay_uncompiled_and_off_discord():
    """Not in setup.py's INCLUDE_FILES (the jetson runs the same pure Python as the runner); no slash command."""
    setup = (REPO / "setup.py").read_text(encoding="utf-8")
    for module in ("card_data", "card_history", "card_db", "changelog"):
        assert f"commands/{module}.py" not in setup
        assert f"commands.{module}" not in (REPO / "main.py").read_text(encoding="utf-8")


# ========================
# Gates
# ========================
def _cards(n, **kw):
    return {f"C{i}": cm.CardInfo(name=f"C{i}", card_type=kw.get("card_type", "item"), ability=kw.get("ability", "Text."),
                                 stats=kw.get("stats", cm.CardStats(1, 1, 1, 1))) for i in range(n)}


def test_gate_floor():
    refused, _ = cd.gates(_cards(200), _cards(99), 0, 0, [], [], False, set(), False)
    assert "G1" in refused
    refused, _ = cd.gates(_cards(200), _cards(189), 0, 0, [], [], True, set(), True)
    assert "G1" in refused                                                       # nothing lifts it
    assert cd.gates(_cards(200), _cards(190), 0, 0, [], [], False, set(), False) == ({}, {})


def test_gate_archive_ceiling():
    refused, _ = cd.gates(_cards(300), _cards(300), 0, 0, [f"C{i}" for i in range(6)], [], False, set(), False)
    assert refused["G2"].startswith("6 cards archived (at most 5)")
    assert cd.gates(_cards(300), _cards(300), 0, 0, ["C1"] * 6, [], False, set(), True) == ({}, {"G2": "allow_archive"})
    assert cd.gates(_cards(1000), _cards(1000), 0, 0, ["C1"] * 10, [], False, set(), False) == ({}, {})   # 1% of 1000


def test_gate_same_blob_reparses():
    names = [f"C{i}" for i in range(26)]
    refused, _ = cd.gates(_cards(200), _cards(200), 0, 0, [], names, False, set(), False)
    assert "G4" in refused
    assert cd.gates(_cards(200), _cards(200), 0, 0, [], names, True, set(), False) == ({}, {"G4": "full"})
    assert cd.gates(_cards(200), _cards(200), 0, 0, [], names, False, {f"c{i}" for i in range(26)}, False) == ({}, {})


def test_gate_quality():
    before, after = _cards(200), _cards(200)
    for i in range(4):
        after[f"C{i}"].ability = ""
    refused, _ = cd.gates(before, after, 0, 0, [], [], False, set(), False)
    assert refused["G5"].startswith("ability text lost: C0, C1, C2, C3")
    assert cd.gates(before, after, 0, 0, [], [], False, {"c0", "c1", "c2", "c3"}, False) == ({}, {"G5": "cards"})
    refused, _ = cd.gates(_cards(200), _cards(200), 1000, 900, [], [], False, set(), False)
    assert refused["G5"] == "psd_details records 1000 -> 900"
    worse = _cards(200, card_type="unknown")
    refused, _ = cd.gates(_cards(200), worse, 0, 0, [], [], False, set(), False)
    assert refused["G5"].startswith("UNKNOWN type")


# ========================
# Builds on a git repo of golden PSDs
# ========================
def git(repo, *args, env=None):
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True,
                          env={**os.environ, **(env or {})}).stdout


def commit(repo, author, at):
    stamp = f"@{at} +0000"
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", f"by {author}", env={"GIT_AUTHOR_NAME": author, "GIT_AUTHOR_EMAIL": f"{author}@x",
                                                          "GIT_COMMITTER_NAME": author, "GIT_COMMITTER_EMAIL": f"{author}@x",
                                                          "GIT_AUTHOR_DATE": stamp, "GIT_COMMITTER_DATE": stamp})
    return git(repo, "rev-parse", "HEAD").strip()


def put(repo, rel, data: bytes):
    path = repo / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


@pytest.fixture
def cards(tmp_path, clone, monkeypatch):
    """A card repo: the golden vocabulary, the site files, Mini Doomer and The Freezer (alice, t=1000); an empty seed.
    G1's floor drops to 1 card and no share (a real build has hundreds)."""
    monkeypatch.setattr(cd, "MIN_CARDS", 1)
    monkeypatch.setattr(cd, "FLOOR_SHARE", 0)
    psds = {}
    for rel in (DOOMER, FREEZER, OTHER):
        data = (clone / rel).read_bytes()
        if blob_sha_bytes(data) != json.loads(golden_file(rel).read_text(encoding="utf-8"))["blob"]:
            pytest.skip(f"{rel} changed upstream since the goldens were made")
        psds[rel] = data
    vocab = json.loads(VOCAB.read_text(encoding="utf-8"))
    repo = tmp_path / "cards"
    repo.mkdir()
    git(repo, "init", "-q", "-b", "main")
    for t in vocab["all_types"]:
        put(repo, f"Types/{t}.png", t.encode())
    for s in vocab["stars"]:
        put(repo, f"Types/Stars/{s}.png", s.encode())
    put(repo, "MDW/Fonts/stoogedings-src/stoogedings.json", json.dumps(vocab["manifest"]).encode())
    put(repo, "site/Card Authors.txt", b"[people]\nbobgit = Bob\n[cards]\nmini_doomer = bobgit\n")
    put(repo, "site/Card Edits.json", json.dumps({"cards": {"The Freezer": {"note": "checked", "_by": "web"}}}).encode())
    put(repo, DOOMER, psds[DOOMER])
    put(repo, FREEZER, psds[FREEZER])
    first = commit(repo, "alice", 1000)
    seed = tmp_path / "seed"
    seed.mkdir()
    (seed / "stats.pkl").write_bytes(pickle.dumps({}))
    db = tmp_path / "data" / "db"
    db.mkdir(parents=True)
    return SimpleNamespace(repo=repo, psds=psds, seed=seed, db=db, first=first, tmp=tmp_path)


def _build(c, *extra):
    message = c.tmp / "message.txt"
    code = cd.main(["--checkout", str(c.repo), "--db", str(c.db), "--message", str(message), *extra])
    return code, (message.read_text(encoding="utf-8") if message.exists() else "")


def _golden(rel):
    return json.loads(golden_file(rel).read_text(encoding="utf-8"))["card"]


def _strip(d):
    return {k: v for k, v in d.items() if k not in ("timestamp", "author", "note")}


def _read(c):
    stats, old, details, manifest = cdb.read_db(str(c.db))
    return stats, old, details, manifest, cl.read_events(str(c.db))


def test_first_build_then_nothing_to_do(cards):
    code, message = _build(cards, "--seed", str(cards.seed))
    assert code == cd.EXIT_OK and message.startswith(f"Card data: 2 new (main {cards.first[:8]})")
    stats, old, details, manifest, events = _read(cards)
    assert set(stats) == {"Mini Doomer", "The Freezer"} and not old
    assert _strip(stats["Mini Doomer"].to_dict()) == _strip(_golden(DOOMER))
    assert _strip(stats["The Freezer"].to_dict()) == _strip(_golden(FREEZER))
    assert (stats["Mini Doomer"].author, stats["The Freezer"].author, stats["The Freezer"].note) == ("Bob", "alice", "checked")
    assert stats["Mini Doomer"].timestamp == 1000.0 and stats["Mini Doomer"].version["cause"] == "psd"
    assert [(e["seq"], e["kind"], e["card"]) for e in events] == [(1, "new", "Mini Doomer"), (2, "new", "The Freezer")]
    assert manifest["card_commit"] == cards.first and manifest["counts"]["cards"] == 2 and manifest["seed"] == {"pickles": str(cards.seed)}
    assert set(details) == {blob_sha_bytes(cards.psds[DOOMER]), blob_sha_bytes(cards.psds[FREEZER])}
    assert cdb.check_db(str(cards.db)) == []
    times = {p.name: p.stat().st_mtime_ns for p in cards.db.iterdir() if p.is_file()}
    assert _build(cards)[0] == cd.EXIT_OK
    assert {p.name: p.stat().st_mtime_ns for p in cards.db.iterdir() if p.is_file()} == times      # nothing to do: no write


def test_changed_moved_archived(cards):
    _build(cards, "--seed", str(cards.seed))
    put(cards.repo, DOOMER, cards.psds[OTHER])
    c2 = commit(cards.repo, "bob", 2000)
    code, message = _build(cards)
    stats, old, _, _, events = _read(cards)
    assert code == 0 and message.startswith("Card data: 1 changed")
    assert stats["Mini Doomer"].ability == _golden(OTHER)["ability"] and stats["Mini Doomer"].timestamp == 2000.0
    version = old["Mini Doomer"][0]
    assert (version.path, version.blob_sha, version.version["to"], version.version["to_commit"]) == \
        (DOOMER, blob_sha_bytes(cards.psds[DOOMER]), 2000.0, c2)
    assert stats["Mini Doomer"].version["from_commit"] == c2
    assert events[len(events) - 1]["kind"] == "changed" and events[len(events) - 1]["by"] == "bob"

    moved = "Field/2 Stars/The_Freezer.psd"
    (cards.repo / "Field/2 Stars").mkdir()
    git(cards.repo, "mv", FREEZER, moved)
    c3 = commit(cards.repo, "carol", 3000)
    assert _build(cards)[0] == 0
    stats, old, _, _, events = _read(cards)
    assert stats["The Freezer"].stars == 2 and [v.path for v in old["The Freezer"]] == [FREEZER]   # its own path kept
    assert stats["The Freezer"].timestamp == 1000.0                                               # same blob: same date
    assert (stats["The Freezer"].version["cause"], stats["The Freezer"].version["from_commit"]) == ("moved", c3)
    last = events[len(events) - 1]
    assert (last["kind"], last["fields"], last["from_path"]) == ("moved", {"stars": [1, 2]}, FREEZER)

    git(cards.repo, "rm", "-q", moved)
    c4 = commit(cards.repo, "dave", 4000)
    assert _build(cards)[0] == 0
    stats, old, _, manifest, events = _read(cards)
    assert "The Freezer" not in stats and manifest["counts"]["archived"] == 1
    tag = old["The Freezer"][len(old["The Freezer"]) - 1].archived
    assert (tag["at"], tag["commit"], tag["path"]) == (4000.0, c4, moved)
    last = events[len(events) - 1]
    assert (last["kind"], last["card"], last["by"], last["seq"]) == ("archived", "The Freezer", "dave", 5)


def test_unreadable_psd_holds_the_archive(cards):
    _build(cards, "--seed", str(cards.seed))
    put(cards.repo, "Items/Broken_Card.psd", BROKEN)
    git(cards.repo, "rm", "-q", FREEZER)
    commit(cards.repo, "erin", 2000)
    code, _ = _build(cards)
    stats, old, _, manifest, _ = _read(cards)
    assert code == 0 and set(stats) == {"Mini Doomer", "The Freezer"} and not old             # nothing archived that run
    assert [f["path"] for f in manifest["pending"]["failed"]] == ["Items/Broken_Card.psd"] and manifest["counts"]["failed"] == 1


def test_budget_holds_cards_for_the_next_run(cards, monkeypatch):
    _build(cards, "--seed", str(cards.seed))
    put(cards.repo, DOOMER, cards.psds[OTHER])
    commit(cards.repo, "bob", 2000)
    psd = blob_sha_bytes(cards.psds[OTHER])
    real = cd.GitBuild._store
    monkeypatch.setattr(cd.GitBuild, "_store", lambda self, sha: "" if sha == psd else real(self, sha))
    code, _ = _build(cards, "--budget", "0")
    stats, _, _, manifest, _ = _read(cards)
    assert code == 0 and manifest["pending"]["stale"] == [DOOMER] and manifest["pending"]["failed"] == []
    assert stats["Mini Doomer"].blob_sha == blob_sha_bytes(cards.psds[DOOMER])                 # its last version stays


def test_gate_refusal_exits_2_and_writes_nothing(cards, monkeypatch, capsys):
    _build(cards, "--seed", str(cards.seed))
    before = {p.name: p.read_bytes() for p in cards.db.iterdir() if p.is_file()}
    git(cards.repo, "rm", "-q", FREEZER)
    commit(cards.repo, "dave", 2000)
    monkeypatch.setattr(cd, "ARCHIVE_MIN", 0)
    assert _build(cards)[0] == cd.EXIT_GATE
    assert "**G2 refused:** 1 cards archived (at most 0): The Freezer" in capsys.readouterr().out
    assert {p.name: p.read_bytes() for p in cards.db.iterdir() if p.is_file()} == before
    assert _build(cards, "--allow-archive")[0] == cd.EXIT_OK and "The Freezer" not in _read(cards)[0]


def test_dry_run_and_errors(cards, tmp_path):
    assert _build(cards, "--seed", str(cards.seed), "--dry-run")[0] == cd.EXIT_OK and not any(cards.db.iterdir())
    assert _build(cards)[0] == cd.EXIT_ERROR                                                   # empty db/ and no seed
    shallow = tmp_path / "shallow"
    subprocess.run(["git", "clone", "-q", "--depth", "1", f"file://{cards.repo}", str(shallow)], check=True)
    assert cd.main(["--checkout", str(shallow), "--db", str(cards.db), "--seed", str(cards.seed)]) == cd.EXIT_ERROR


def test_data_tip_moved_refuses(cards, monkeypatch):
    heads = iter(["a" * 40, "b" * 40])
    monkeypatch.setattr(cd, "_data_head", lambda folder: next(heads))
    assert _build(cards, "--seed", str(cards.seed))[0] == cd.EXIT_GATE and not any(cards.db.iterdir())


def test_restore_writes_an_older_data_commit_back(cards):
    data = cards.db.parent
    git(data, "init", "-q", "-b", "data")
    _build(cards, "--seed", str(cards.seed))
    good = commit(data, "AutoCroissant", 1500)
    put(cards.repo, DOOMER, cards.psds[OTHER])
    commit(cards.repo, "bob", 2000)
    _build(cards)
    commit(data, "AutoCroissant", 2500)
    assert _read(cards)[0]["Mini Doomer"].ability == _golden(OTHER)["ability"]
    code, message = _build(cards, "--restore", good)
    stats, old, _, manifest, events = _read(cards)
    assert code == 0 and stats["Mini Doomer"].ability == _golden(DOOMER)["ability"] and not old
    assert manifest["restored_from"] == good and [e["kind"] for e in events] == ["new", "new", "changed", "restore"]
    assert events[len(events) - 1]["seq"] == 4                                                # seq goes on, never reused
