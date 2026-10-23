"""card_data: the Action's sparse copy (--help, the import closure), the gates G1-G6, the exit codes, and offline builds
on a temporary git repo of golden PSDs (new, changed, moved, archived, an unreadable PSD, the budget, restore)."""
import json
import os
import pickle
import re
import shutil
import subprocess
import tempfile
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
    """Exactly SPARSE_FILES, no config.py: --help works on the three CLIs card-sync.yml runs (the build, the bake, the
    save), and their modules (lazy ones too) are that list, with no Discord, pandas or the bot's modules."""
    bot = tmp_path / "bot"
    for rel in cd.SPARSE_FILES:
        assert (REPO / rel).is_file(), f"{rel} is in SPARSE_FILES but missing here"
        (bot / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO / rel, bot / rel)
    flags = {"card_data": {"--repo", "--checkout", "--head", "--blobs", "--db", "--seed", "--aliases", "--full", "--cards",
                           "--allow-archive", "--restore", "--budget", "--run", "--builder-ref", "--summary", "--message",
                           "--dry-run", "--baked", "--rewrite", "--help"},
             "card_bake": None, "tts_save": {"--names", "--help"}}
    for cli, expected in flags.items():
        done = run_process([executable, "-m", f"commands.{cli}", "--help"], cwd=bot, capture_output=True, text=True, timeout=120)
        assert done.returncode == 0, done.stderr
        assert done.stdout.startswith(f"usage: python -m commands.{cli}")
        if expected is not None:
            assert set(re.findall(r"--[a-z-]+", done.stdout)) == expected
    code = ("import sys, commands.card_data, commands.card_history, commands.card_db, commands.changelog, commands.psd_details, "
            "commands.card_bake, commands.tts_save; "
            "print(sorted(m for m in ('config', 'discord', 'pandas', 'github', 'commands.psd_analyzer', 'commands.repo_traverser', "
            "'commands.query_card', 'commands.utils', 'commands.type_emojis', 'commands.card_runner') if m in sys.modules)); "
            "print(sorted(m for m in sys.modules if m.startswith('commands.') or m == 'global_config'))")
    done = run_process([executable, "-c", code], cwd=bot, capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    bad, closure = done.stdout.splitlines()
    assert bad == "[]"
    modules = {m.replace(".", "/") + ".py" for m in json.loads(closure.replace("'", '"'))}
    assert modules == {f for f in cd.SPARSE_FILES if f.endswith(".py")}                   # card-sync.yml copies this list
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


def test_subject_counts_every_kind():
    out = cd.Outcome(lines=["authors: A: author", "changed: B: hp", "authors: C: author"])
    assert cd.subject(out, "1a2b3c4d5e") == "Card data: 1 changed, 2 authors (main 1a2b3c4d)"


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
    monkeypatch.setattr(cd, "_activity", lambda repository: [])                  # offline: pushes told by author
    aliases = tmp_path / "main-aliases.pkl"                                      # offline: AutoCroissant main's aliases.pkl
    aliases.write_bytes(pickle.dumps({}))
    monkeypatch.setattr(cd, "ALIASES_REF", str(aliases))
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
    return SimpleNamespace(repo=repo, psds=psds, seed=seed, db=db, first=first, tmp=tmp_path, aliases=aliases)


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
    last_version = old["The Freezer"][len(old["The Freezer"]) - 1]
    tag = last_version.archived
    assert (tag["at"], tag["commit"], tag["path"]) == (4000.0, c4, moved)
    assert (last_version.version["to"], last_version.version["to_commit"]) == (4000.0, c4)       # closed at the deletion
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


def test_failed_psd_waits_for_a_new_blob(cards, monkeypatch):
    """A PSD that could not be read is not read again until its blob changes, or `cards` names it; it stays failed
    (and, a new name, holds the archive) meanwhile."""
    _build(cards, "--seed", str(cards.seed))
    put(cards.repo, "Items/Broken_Card.psd", BROKEN)
    git(cards.repo, "rm", "-q", FREEZER)
    commit(cards.repo, "erin", 2000)
    assert _build(cards)[0] == cd.EXIT_OK
    read = []
    real = cd.GitBuild._download
    monkeypatch.setattr(cd.GitBuild, "_download", lambda self, path, sha: read.append(path) or real(self, path, sha))
    assert _build(cards)[0] == cd.EXIT_OK and read == []
    stats, _, _, manifest, _ = _read(cards)
    assert [f["path"] for f in manifest["pending"]["failed"]] == ["Items/Broken_Card.psd"] and manifest["counts"]["failed"] == 1
    assert "The Freezer" in stats                                                             # still held
    assert _build(cards, "--cards", "Broken Card")[0] == cd.EXIT_OK and read == ["Items/Broken_Card.psd"]
    put(cards.repo, "Items/Broken_Card.psd", BROKEN + b"\0")
    commit(cards.repo, "erin", 3000)
    assert _build(cards)[0] == cd.EXIT_OK and read == ["Items/Broken_Card.psd"] * 2


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


def test_an_exact_rename_passes_g2(cards, monkeypatch):
    """A renamed PSD (the same bytes under another name) archives the old name with renamed_to: no card is lost, so G2
    lets it through at a ceiling of 0 (d64fc8d4 renamed 27 cards at once)."""
    _build(cards, "--seed", str(cards.seed))
    git(cards.repo, "mv", FREEZER, "Field/1 Stars/The_Big_Freezer.psd")
    commit(cards.repo, "dave", 2000)
    monkeypatch.setattr(cd, "ARCHIVE_MIN", 0)
    assert _build(cards)[0] == cd.EXIT_OK
    stats, old = _read(cards)[:2]
    assert "The Big Freezer" in stats and old["The Freezer"][len(old["The Freezer"]) - 1].archived["renamed_to"] == \
        "The Big Freezer"
    assert stats["The Big Freezer"].author == "alice"                   # the card's, not the renamer's


def test_a_new_cards_date_and_author_come_from_its_first_path(tmp_path):
    """The build's lookup for a card new under a renamed path whose parse carried no author: the first commit's author
    at the path its exact rename came from (alice, not the renamer dave), and the date of the commit that put its bytes
    back after a revert there (carol), not their first intro nor the rename."""
    repo = tmp_path / "cards"
    repo.mkdir()
    git(repo, "init", "-q", "-b", "main")
    for author, at, data in (("alice", 1000, b"one"), ("bob", 2000, b"two"), ("carol", 3000, b"one")):
        put(repo, "Items/A.psd", data)
        commit(repo, author, at)
    git(repo, "mv", "Items/A.psd", "Items/B.psd")
    commit(repo, "dave", 4000)
    build = cd.GitBuild(cm.StatsDatabase(), str(repo))
    build.shas = {"Items/B.psd": blob_sha_bytes(b"one")}
    assert build._get_remote_timestamp(None, "Items/B.psd", "B") == (3000.0, "alice")


def test_a_rename_that_edited_the_psd_keeps_the_card_by_its_art(cards):
    """The Freezer renamed while its PSD was edited (another blob), its PNG renamed unchanged: the new card is parsed
    with the old one (its author, not the renamer's), and the archive names it."""
    put(cards.repo, "Field/1 Stars/The_Freezer.png", b"freezer art")
    commit(cards.repo, "alice", 1100)
    _build(cards, "--seed", str(cards.seed))
    big = "Field/1 Stars/The_Big_Freezer"
    git(cards.repo, "mv", "Field/1 Stars/The_Freezer.png", f"{big}.png")
    git(cards.repo, "rm", "-q", FREEZER)
    put(cards.repo, f"{big}.psd", cards.psds[OTHER])
    commit(cards.repo, "dave", 2000)
    assert _build(cards, "--allow-archive")[0] == cd.EXIT_OK
    stats, old = _read(cards)[:2]
    assert stats["The Big Freezer"].author == "alice"
    assert old["The Freezer"][len(old["The Freezer"]) - 1].archived["renamed_to"] == "The Big Freezer"


def test_dry_run_and_errors(cards, tmp_path):
    assert _build(cards, "--seed", str(cards.seed), "--dry-run")[0] == cd.EXIT_OK and not any(cards.db.iterdir())
    assert _build(cards)[0] == cd.EXIT_ERROR                                                   # empty db/ and no seed
    shallow = tmp_path / "shallow"
    subprocess.run(["git", "clone", "-q", "--depth", "1", f"file://{cards.repo}", str(shallow)], check=True)
    assert cd.main(["--checkout", str(shallow), "--db", str(cards.db), "--seed", str(cards.seed)]) == cd.EXIT_ERROR


def test_data_tip_moved_refuses(cards, tmp_path):
    """G6 asks origin (git ls-remote): another writer's push to the data branch since this checkout refuses the write."""
    origin = tmp_path / "origin.git"
    git(tmp_path, "init", "-q", "--bare", "-b", "data", str(origin))
    data = cards.db.parent
    git(data, "init", "-q", "-b", "data")
    git(data, "remote", "add", "origin", str(origin))
    _build(cards, "--seed", str(cards.seed))
    commit(data, "AutoCroissant", 1500)
    git(data, "push", "-q", "origin", "data")
    other = tmp_path / "other"
    git(tmp_path, "clone", "-q", "-b", "data", str(origin), str(other))
    put(other, "db/extra.txt", b"another writer")
    commit(other, "someone", 1600)
    git(other, "push", "-q", "origin", "data")
    put(cards.repo, DOOMER, cards.psds[OTHER])
    commit(cards.repo, "bob", 2000)
    before = {p.name: p.read_bytes() for p in cards.db.iterdir() if p.is_file()}
    code, _ = _build(cards)
    assert code == cd.EXIT_GATE and {p.name: p.read_bytes() for p in cards.db.iterdir() if p.is_file()} == before
    git(data, "pull", "-q", "--ff-only", "origin", "data")                       # the next run starts from the new tip
    assert _build(cards)[0] == cd.EXIT_OK and _read(cards)[0]["Mini Doomer"].ability == _golden(OTHER)["ability"]


def _rest_psds(cards, monkeypatch, parsed=None):
    """Every PSD blob off the local stores, so each is a budgeted REST download (served from the repo's git)."""
    monkeypatch.setattr(cd, "RANGE_READS", False)
    psds = {blob_sha_bytes(data) for data in cards.psds.values()}
    real = cd.GitBuild._store
    monkeypatch.setattr(cd.GitBuild, "_store", lambda self, sha: "" if sha in psds else real(self, sha))

    def download(self, path, sha):
        if parsed is not None:
            parsed.append(path)
        fd, name = tempfile.mkstemp(suffix=".psd")
        with os.fdopen(fd, "wb") as fh:
            fh.write(git_bytes(cards.repo, "cat-file", "blob", sha))
        return name
    monkeypatch.setattr(cd.CardBuild, "_download", download)


def git_bytes(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, check=True).stdout


def test_full_reparse_continues_past_the_budget(cards, monkeypatch):
    """--full over the budget: the held cards (same blob) are read by the next runs (pending.stale), the approval goes
    on in pending.full until none is left."""
    _build(cards, "--seed", str(cards.seed))
    put(cards.repo, OTHER, cards.psds[OTHER])
    commit(cards.repo, "bob", 2000)
    _build(cards)
    parsed = []
    _rest_psds(cards, monkeypatch, parsed)
    assert _build(cards, "--full", "--budget", "1", "--run", "https://x/runs/1")[0] == cd.EXIT_OK
    pending = _read(cards)[3]["pending"]
    assert len(parsed) == 1 and len(pending["stale"]) == 2
    assert pending["full"] == {"parser_version": cd.PARSER_VERSION, "approved_run": "https://x/runs/1"}
    assert _build(cards, "--budget", "1")[0] == cd.EXIT_OK and len(parsed) == 2
    assert len(_read(cards)[3]["pending"]["stale"]) == 1 and _read(cards)[3]["pending"]["full"]["approved_run"] == "https://x/runs/1"
    assert _build(cards, "--budget", "5")[0] == cd.EXIT_OK
    assert sorted(parsed) == sorted([DOOMER, FREEZER, OTHER])
    assert _read(cards)[3]["pending"] == {"stale": [], "failed": [], "full": None, "repin": []}
    times = {p.name: p.stat().st_mtime_ns for p in cards.db.iterdir() if p.is_file()}
    assert _build(cards)[0] == cd.EXIT_OK and len(parsed) == 3                   # done: nothing read, nothing written
    assert {p.name: p.stat().st_mtime_ns for p in cards.db.iterdir() if p.is_file()} == times


def test_parser_bump_approval_carries_over(cards, monkeypatch):
    """A PARSER_VERSION bump: G4 refuses until `full`; the continuation runs keep that approval."""
    import commands.psd_parser as pp
    _build(cards, "--seed", str(cards.seed))
    put(cards.repo, OTHER, cards.psds[OTHER])
    commit(cards.repo, "bob", 2000)
    _build(cards)
    _rest_psds(cards, monkeypatch)
    monkeypatch.setattr(cd, "REPARSE_MAX", 0)
    monkeypatch.setattr(cd, "PARSER_VERSION", pp.PARSER_VERSION + 1)
    monkeypatch.setattr(pp, "PARSER_VERSION", pp.PARSER_VERSION + 1)
    assert _build(cards, "--budget", "5")[0] == cd.EXIT_GATE                     # unapproved
    assert _build(cards, "--full", "--budget", "2", "--run", "https://x/runs/7")[0] == cd.EXIT_OK
    assert len(_read(cards)[3]["pending"]["stale"]) == 1
    assert _build(cards, "--budget", "2")[0] == cd.EXIT_OK                       # 1 same-blob re-parse, approved before
    stats, _, _, manifest, _ = _read(cards)
    assert manifest["gates"] == {"G4": "lifted by full (approved by https://x/runs/7)"}
    assert manifest["pending"]["stale"] == [] and manifest["pending"]["full"] is None
    assert {c.parser_version for c in stats.values()} == {pp.PARSER_VERSION}


def test_download_failure_fails_red_and_retries(cards, monkeypatch):
    """A REST download that fails: exit 1, db/ untouched (card_commit stays), so the next run tries it again."""
    from requests.exceptions import ConnectionError as Down
    _build(cards, "--seed", str(cards.seed))
    before = {p.name: p.read_bytes() for p in cards.db.iterdir() if p.is_file()}
    put(cards.repo, DOOMER, cards.psds[OTHER])
    commit(cards.repo, "bob", 2000)
    psd = blob_sha_bytes(cards.psds[OTHER])
    real = cd.GitBuild._store
    monkeypatch.setattr(cd.GitBuild, "_store", lambda self, sha: "" if sha == psd else real(self, sha))
    down = [True]

    def blob(self, sha, stream=False):
        if down[0]:
            raise Down("network down")
        return _Reply(git_bytes(cards.repo, "cat-file", "blob", sha))
    monkeypatch.setattr(cd.CardBuild, "_blob", blob)
    code, _ = _build(cards)
    assert code == cd.EXIT_ERROR and {p.name: p.read_bytes() for p in cards.db.iterdir() if p.is_file()} == before
    down[0] = False
    assert _build(cards)[0] == cd.EXIT_OK and _read(cards)[0]["Mini Doomer"].blob_sha == psd


class _Reply:
    """fetch_blob's streamed reply."""
    headers: dict = {}

    def __init__(self, data):
        self.data = data

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def iter_content(self, size):
        return [self.data]


def test_events_carry_their_push(cards, monkeypatch):
    """Each event's push: the activity list's (the pusher's login, the compare range), else consecutive commits by one
    author; a re-read has none."""
    _build(cards, "--seed", str(cards.seed))
    put(cards.repo, DOOMER, cards.psds[OTHER])
    c2 = commit(cards.repo, "bob", 2000)
    put(cards.repo, "Items/Anubisath_Guardian.psd", cards.psds[OTHER])
    c3 = commit(cards.repo, "bob", 2100)
    git(cards.repo, "rm", "-q", FREEZER)
    c4 = commit(cards.repo, "dave", 2200)
    _build(cards, "--allow-archive")
    by_card = {e["card"]: e for e in _read(cards)[4] if e["seq"] > 2}
    assert by_card["Mini Doomer"]["push"] == {"after": c3, "before": cards.first, "by": "bob", "at": 2100.0}
    assert by_card["The Freezer"]["push"] == {"after": c4, "before": c3, "by": "dave", "at": 2200.0}
    assert [e["kind"] for e in sorted(by_card.values(), key=lambda e: e["seq"])] == ["new", "changed", "archived"]   # by push, then name

    put(cards.repo, DOOMER, cards.psds[DOOMER])
    c5 = commit(cards.repo, "erin", 3000)
    monkeypatch.setattr(cd, "_activity", lambda repository: [
        {"before": c4, "after": c5, "actor": {"login": "erin-gh"}, "timestamp": "1970-01-01T00:50:10Z"}])
    _build(cards)
    last = _read(cards)[4][len(_read(cards)[4]) - 1]
    assert (last["card"], last["push"]) == ("Mini Doomer", {"after": c5, "before": c4, "by": "erin-gh", "at": 3010.0})


def test_bake_commit_rewritten_to_the_main_commit(cards):
    """Stage 4: the build ran on the local bake commit B; after M (B plus the save) is pushed, --rewrite B M names M
    everywhere B was (dates too) and lists B's PNGs to re-pin."""
    _build(cards, "--seed", str(cards.seed))
    put(cards.repo, DOOMER, cards.psds[OTHER])
    put(cards.repo, "Auxiliary/Minions/Mini_Doomer.png", b"baked png")
    b = commit(cards.repo, "AutoCroissant", 2000)
    assert _build(cards, "--head", b)[0] == cd.EXIT_OK
    put(cards.repo, "TTS/Impossibility Simulator.json", b"{}")
    git(cards.repo, "add", "-A")
    git(cards.repo, "commit", "-q", "--amend", "-m", "Card sync: 1 card baked, TTS save updated",
        env={"GIT_COMMITTER_DATE": "@2500 +0000", "GIT_AUTHOR_DATE": "@2500 +0000"})
    m = git(cards.repo, "rev-parse", "HEAD").strip()
    message = cards.tmp / "message.txt"
    assert b[:8] in message.read_text(encoding="utf-8")
    assert cd.main(["--checkout", str(cards.repo), "--db", str(cards.db), "--message", str(message), "--rewrite", b, m]) == cd.EXIT_OK
    assert m[:8] in message.read_text(encoding="utf-8") and b[:8] not in message.read_text(encoding="utf-8")
    stats, old, _, manifest, events = _read(cards)
    doomer = stats["Mini Doomer"]
    assert (doomer.version["from_commit"], doomer.version["from"], doomer.timestamp) == (m, 2500.0, 2500.0)
    assert old["Mini Doomer"][0].version["to_commit"] == m and old["Mini Doomer"][0].version["to"] == 2500.0
    assert (manifest["card_commit"], manifest["main_commit"], manifest["card_commit_at"]) == (m, m, 2500.0)
    assert manifest["counts"]["baked"] == 1 and manifest["pending"]["repin"] == ["Auxiliary/Minions/Mini_Doomer.png"]
    changed = events[len(events) - 1]
    assert (changed["commit"], changed["card_commit"], changed["push"]["after"], changed["push"]["by"]) == (m, m, m, "Card sync")
    assert b not in (cards.db / "stats.json").read_text() + (cards.db / "old_stats.json").read_text()
    assert cdb.check_db(str(cards.db)) == []


def test_bake_commit_posts_as_its_own_push(cards):
    """--baked: the bake commit B is the card sync's push, not part of the push before it by the same author."""
    _build(cards, "--seed", str(cards.seed))
    put(cards.repo, "site/Card Edits.json", json.dumps({"cards": {"The Freezer": {"note": "via bot", "_by": "discord:mike"}}}).encode())
    c2 = commit(cards.repo, "AutoCroissant", 1500)
    put(cards.repo, DOOMER, cards.psds[OTHER])
    b = commit(cards.repo, "AutoCroissant", 2000)
    assert _build(cards, "--head", b, "--baked")[0] == cd.EXIT_OK
    by_card = {e["card"]: e["push"] for e in _read(cards)[4] if e["seq"] > 2}
    assert by_card["The Freezer"] == {"after": c2, "before": cards.first, "by": "AutoCroissant", "at": 1500.0}
    assert by_card["Mini Doomer"] == {"after": b, "before": c2, "by": cl.CARD_SYNC, "at": 2000.0}


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
    assert message.startswith(f"Card data: restored from {good[:8]} (main {cards.first[:8]})")
    assert manifest["card_commit"] == cards.first                 # the restored data's: the next run re-reads what changed
    out = cards.tmp / "discord.json"
    assert cl.main(["post", "--db", str(cards.db), "--out", str(out)]) == 0
    sent = json.loads(out.read_text(encoding="utf-8"))["messages"]
    assert sent[len(sent) - 1]["cursor"]["seq"] == 4
    assert sent[len(sent) - 1]["body"]["embeds"][0]["title"].endswith(": 1 data rollback")


def test_duplicates_named_by_the_seed_aliases(cards, tmp_path):
    """The seed's aliases.pkl (or --aliases) names a duplicate PNG, as /update_stats does with the bot's aliases."""
    put(cards.repo, "Auxiliary/Minions/Mini_Doomer2.png", b"second art")
    commit(cards.repo, "alice", 1100)
    (cards.seed / "aliases.pkl").write_bytes(pickle.dumps({"tiny doom": "mini_doomer2.png"}))
    assert _build(cards, "--seed", str(cards.seed))[0] == cd.EXIT_OK
    stats = _read(cards)[0]
    assert (stats["Mini Doomer2"].display_name, stats["Mini Doomer2"].duplicate_of) == ("Tiny Doom", "Mini Doomer")
    aliases = tmp_path / "aliases.pkl"
    aliases.write_bytes(pickle.dumps({"late doom": "mini_doomer2.png"}))
    assert _build(cards, "--aliases", str(aliases))[0] == cd.EXIT_OK
    assert _read(cards)[0]["Mini Doomer2"].display_name == "Late Doom"
    cards.aliases.write_bytes(pickle.dumps({"tiny doom": "mini_doomer2.png"}))
    assert _build(cards)[0] == cd.EXIT_OK                                        # no --aliases: AutoCroissant main's
    assert _read(cards)[0]["Mini Doomer2"].display_name == "Tiny Doom"
