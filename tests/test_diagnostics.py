"""The diagnostics skill's diff_stats.py and inspect_pickle.py read the data build's db/ JSON: a db/ folder, a
stats.json, a data commit (the shadow's daily check against origin/data)."""
import subprocess
import sys
from collections import defaultdict

import commands.card_db as cdb
import commands.card_models as cm
from conftest import REPO

SCRIPTS = REPO / ".claude" / "skills" / "autocroissant-diagnostics-and-tooling" / "scripts"


def _card(name, **kw):
    base = dict(name=name, card_type="creature", path=f"Creatures/X/3 Stars/{name}.psd", ability="Hit (1CD).", stars=3,
                stats=cm.CardStats(5, 2, 3, 4), timestamp=1_700_000_000.0, blob_sha="a" * 40, parser_version=2)
    base.update(kw)
    return cm.CardInfo(**base)


def _run(script, *args, cwd=None):
    done = subprocess.run([sys.executable, str(SCRIPTS / script), *map(str, args)], capture_output=True, text=True,
                          cwd=cwd or REPO)
    return done.returncode, done.stdout + done.stderr


def _db(folder, stats, old):
    cdb.write_db(str(folder), stats, old, '{"version": 1, "cards": {\n\n}}\n', {})


def _git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True).stdout


def test_diff_stats_reads_db_folders_and_data_commits(tmp_path):
    a = {"Kirby": _card("Kirby"), "Gone": _card("Gone")}
    b = {"Kirby": _card("Kirby", stats=cm.CardStats(6, 2, 3, 4)), "New": _card("New")}
    b["Kirby"].version = {"from": 1_600_000_000.0, "cause": "legacy"}
    old_a, old_b = defaultdict(list), defaultdict(list, {"Kirby": [_card("Kirby", ability="Before.", timestamp=1.0)]})
    _db(tmp_path / "pk", a, old_a)
    data = tmp_path / "data"
    _git(tmp_path, "init", "-q", "-b", "data", str(data))
    _db(data / "db", b, old_b)
    _git(data, "add", "-A")
    _git(data, "-c", "user.name=t", "-c", "user.email=t@x", "commit", "-q", "-m", "Card data")

    for new in (data / "db", data, data / "db" / "stats.json"):
        code, out = _run("diff_stats.py", tmp_path / "pk" / "stats.json", new)
        assert code == 1, out
        assert "added: 1   removed: 1   modified: 1" in out and "+ New" in out and "- Gone" in out, out
        assert "versions added: 1 (1 names)" in out and "starts moved: 1" in out, out
    code, out = _run("diff_stats.py", tmp_path / "pk", "HEAD", "--git", data)
    assert code == 1 and "modified: 1" in out and "versions added: 1" in out, out
    code, out = _run("diff_stats.py", "HEAD", data / "db", "--git", data)
    assert code == 0 and "added: 0   removed: 0   modified: 0" in out, out
    code, out = _run("diff_stats.py", tmp_path / "nothing", "HEAD", "--git", data)
    assert code == 2 and "ERROR loading the card data" in out, out


def test_diff_stats_matches_re_dated_versions(tmp_path):
    """A version given its commit's blob, or re-dated to its blob's intro, is the same version: no version added or
    dropped; the re-dated one's start moved, a normalized one's data changed."""
    stats = {"Kirby": _card("Kirby")}
    jetson = defaultdict(list, {"Kirby": [_card("Kirby", ability="A.", timestamp=1.0, blob_sha=None),
                                          _card("Kirby", ability="B.", timestamp=2.0, blob_sha="b" * 40),
                                          _card("Kirby", ability="C.", timestamp=3.0, blob_sha="c" * 40, series=None)]})
    built = defaultdict(list, {"Kirby": [_card("Kirby", ability="A.", timestamp=1.0, blob_sha="a" * 40),
                                         _card("Kirby", ability="B.", timestamp=1.5, blob_sha="b" * 40),
                                         _card("Kirby", ability="C.", timestamp=3.0, blob_sha="c" * 40, series="Kirby")]})
    _db(tmp_path / "pk", stats, jetson)
    _db(tmp_path / "db", stats, built)
    code, out = _run("diff_stats.py", tmp_path / "pk" / "stats.json", tmp_path / "db")
    assert code == 1, out
    assert "versions added: 0 (0 names)   dropped: 0 (0 names)" in out, out
    assert "starts moved: 1" in out and "kept versions with other data: 1" in out and "  data Kirby" in out, out

