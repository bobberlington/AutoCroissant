"""card_db: CardInfo records, db/'s JSON files and manifest, the seed's allow-list unpickler, export-pickles."""
import hashlib
import json
import os
import pickle
import subprocess
import sys
from collections import defaultdict
from dataclasses import fields

import pytest

import commands.card_db as cdb
import commands.card_models as cm
from conftest import REPO


def _card(name="A Card", **kw):
    base = dict(name=name, card_type="creature", path=f"Creatures/X/3 Stars/{name.replace(' ', '_')}.psd", ability="Hit (1CD).",
                stars=3, stats=cm.CardStats(5, 2, 3, 4), types=["fire"], timestamp=1_700_000_000.25, blob_sha="a" * 40,
                parser_version=2, author="Bob")
    base.update(kw)
    return cm.CardInfo(**base)


def _full():
    card = _card(creatures=[cm.CardStats(1, 2, 3, 4), cm.CardStats(5, 6, 7, -1)], display_name="Shown", note="n",
                 parsed_values={"series": "X", "subseries": None, "problems": ["HP NOT FOUND"]}, problems=["SEVERAL CREATURES: 2"],
                 alt_names=["B", "C"], archived={"at": 1.5, "path": "p.psd", "blob": None, "commit": "c" * 40})
    card.version = {"from": 1.0, "from_commit": "f" * 40, "to": None, "to_commit": None, "cause": "psd", "png_commit": None,
                    "legacy": False}
    return card


# ========================
# Records
# ========================
def test_record_has_every_field_and_the_stats_by_name():
    rec = cdb.record(_card())
    assert set(rec) == {f.name for f in fields(cm.CardInfo)} | {"version"}
    assert rec["stats"] == {"hp": 5, "defense": 2, "attack": 3, "speed": 4}
    assert rec["creatures"] is None and rec["version"] is None and rec["archived"] is None


def test_round_trip_is_exact():
    card = _full()
    back = cdb.card(cdb.record(card))
    assert back == card and repr(back) == repr(card) and back.version == card.version
    assert cdb.record(back) == cdb.record(card) and json.loads(json.dumps(cdb.record(card))) == cdb.record(card)
    assert isinstance(back.stats, cm.CardStats) and all(isinstance(c, cm.CardStats) for c in back.creatures)
    assert type(back.timestamp) is float and back.timestamp == card.timestamp


def test_version_is_no_field():
    """version is a class default, not a dataclass field: parsed cards pickle exactly as before it existed."""
    assert "version" not in {f.name for f in fields(cm.CardInfo)} and cm.CardInfo.version is None
    card = _card()
    assert "version" not in card.__dict__ and b"\x8c\x07version" not in pickle.dumps(card)   # (parser_version is)


def test_missing_keys_default_and_unknown_keys_warn_once(capsys, monkeypatch):
    monkeypatch.setattr(cdb, "_warned", set())
    card = cdb.card({"name": "B", "card_type": "item", "sparkle": 1})
    cdb.card({"name": "C", "card_type": "item", "sparkle": 2, "glow": 3})
    assert card.types == [] and card.stats == cm.CardStats() and card.version is None and card.parser_version is None
    out = capsys.readouterr().out
    assert out.count("'sparkle'") == 1 and out.count("'glow'") == 1


def test_older_pickle_card_takes_the_defaults():
    """A card unpickled from before a field existed (no attribute in its __dict__) records the field's default."""
    card = _card()
    for key in ("display_name", "subseries", "parser_version", "archived", "creatures", "types"):
        del card.__dict__[key]
    rec = cdb.record(card)
    assert (rec["display_name"], rec["subseries"], rec["parser_version"], rec["archived"], rec["creatures"], rec["types"]) == \
        (None, None, None, None, None, [])


def test_not_json_data_is_refused():
    card = _card(parsed_values={"x": {1, 2}})
    with pytest.raises(TypeError, match="A Card.parsed_values"):
        cdb.record(card)


# ========================
# Files
# ========================
def test_files_are_one_card_per_line_and_match_the_manifest(tmp_path):
    stats = {"B": _card("B"), "A": _full()}
    old = defaultdict(list, {"A": [_card("A", ability="Old.")], "Gone": [_card("Gone")], "Empty": []})
    manifest = cdb.write_db(str(tmp_path), stats, old, '{"version": 1, "cards": {\n\n}}\n', {"card_commit": "h"})
    lines = (tmp_path / "stats.json").read_text(encoding="utf-8").splitlines()
    assert lines[0] == '{"schema":1,"cards":{' and lines[len(lines) - 1] == "}}" and len(lines) == 4
    assert lines[1].startswith('"A":{') and lines[2].startswith('"B":{')
    assert all(json.loads((tmp_path / name).read_text(encoding="utf-8")) for name in cdb.DATA_FILES)
    for name in cdb.DATA_FILES:
        assert manifest["files"][name] == hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
    assert json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8")) == manifest and manifest["card_commit"] == "h"
    assert cdb.check_db(str(tmp_path)) == []
    got_stats, got_old, details, read_manifest = cdb.read_db(str(tmp_path))
    assert got_stats == stats and dict(got_old) == {"A": old["A"], "Gone": old["Gone"]} and isinstance(got_old, defaultdict)
    assert details == {} and read_manifest == manifest and got_stats["A"].version == stats["A"].version
    (tmp_path / "stats.json").write_text(lines[0] + "\n" + lines[2] + "\n}}\n", encoding="utf-8")
    assert cdb.check_db(str(tmp_path)) == ["stats.json: sha256 differs from the manifest"]


def test_nested_order_kept_so_edits_see_no_change():
    """A loaded card reprs as it was (parsed_values keeps its order): CardEdits.apply reports no spurious change."""
    from commands.card_edits import CardEdits
    card = _card(series="Edited", parsed_values={"series": "PSD", "problems": []})
    edits = CardEdits.parse(json.dumps({"cards": {"A Card": {"series": "Edited"}}}))
    edits.apply("A Card", card)
    loaded = cdb.parse_stats(cdb.stats_text({"A Card": card}))["A Card"]
    assert repr(loaded) == repr(card) and not edits.apply("A Card", loaded)


def test_write_is_atomic(tmp_path, monkeypatch):
    target = tmp_path / "stats.json"
    target.write_text("old", encoding="utf-8")

    def broken(src, dst):
        raise OSError("disk full")
    monkeypatch.setattr(cdb, "replace", broken)
    with pytest.raises(OSError):
        cdb.write_text(str(target), "new")
    assert target.read_text(encoding="utf-8") == "old" and not (tmp_path / "stats.json.tmp").exists()


def test_a_foreign_file_is_refused(tmp_path):
    (tmp_path / "stats.json").write_text('{"cards": {}}', encoding="utf-8")
    with pytest.raises(ValueError, match="not a schema 1 card file"):
        cdb.read_db(str(tmp_path))


# ========================
# Pickles
# ========================
class Evil:
    def __reduce__(self):
        return (os.system, ("echo pwned",))


def test_seed_unpickler_takes_the_card_classes_only(tmp_path):
    old = defaultdict(list, {"A": [_card()]})
    good = tmp_path / "old_stats.pkl"
    good.write_bytes(pickle.dumps(old))
    loaded = cdb.load_pickle(str(good))
    assert isinstance(loaded, defaultdict) and loaded == old and type(loaded["A"][0]) is cm.CardInfo
    bad = tmp_path / "bad.pkl"
    bad.write_bytes(pickle.dumps({"x": Evil()}))
    with pytest.raises(pickle.UnpicklingError, match="not allowed"):
        cdb.load_pickle(str(bad))


def test_export_pickles_round_trip(tmp_path):
    stats, old = {"A": _full()}, defaultdict(list, {"A": [_card("A", ability="Old.")]})
    cdb.write_db(str(tmp_path / "db"), stats, old, '{"version": 1, "cards": {\n\n}}\n', {})
    cdb.export_pickles(str(tmp_path / "db"), str(tmp_path / "out"))
    data = (tmp_path / "out" / "stats.pkl").read_bytes()
    assert b"commands.psd_analyzer" in data and pickle.loads(data) == stats
    assert pickle.loads((tmp_path / "out" / "old_stats.pkl").read_bytes()) == old
    assert (tmp_path / "out" / "psd_details.json").read_text(encoding="utf-8").startswith('{"version": 1')


def test_export_needs_no_bot(tmp_path):
    """Without config.py, discord or the facade loaded, the pickles still name commands.psd_analyzer (a stand-in)."""
    cdb.write_db(str(tmp_path / "db"), {"A": _card()}, defaultdict(list), '{"version": 1, "cards": {\n\n}}\n', {})
    code = ("import sys; sys.path.insert(0, sys.argv[1]); import commands.card_db as c; c.export_pickles(sys.argv[2], sys.argv[3]); "
            "print('commands.psd_analyzer' in sys.modules, 'discord' in sys.modules)")
    done = subprocess.run([sys.executable, "-c", code, str(REPO), str(tmp_path / "db"), str(tmp_path / "out")], cwd=tmp_path,
                          capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    assert done.stdout.split() == ["False", "False"]
    assert b"commands.psd_analyzer\x94\x8c\x08CardInfo" in (tmp_path / "out" / "stats.pkl").read_bytes()


def test_cli_from_pickles_and_back(tmp_path):
    folder = tmp_path / "p"
    folder.mkdir()
    stats = {"A": _card()}
    (folder / "stats.pkl").write_bytes(pickle.dumps(stats))
    assert cdb.main(["from-pickles", str(folder), str(tmp_path / "db")]) == 0
    assert cdb.main(["check", str(tmp_path / "db")]) == 0
    assert cdb.main(["export-pickles", str(tmp_path / "db"), str(tmp_path / "out")]) == 0
    assert pickle.loads((tmp_path / "out" / "stats.pkl").read_bytes()) == stats


def test_export_pickles_from_a_data_commit(tmp_path):
    """Rollback step 2: export-pickles <data commit> <out> --git <data checkout> reads db/ at that commit."""
    data = tmp_path / "data"
    cdb.write_db(str(data / "db"), {"A": _card(ability="Good.")}, defaultdict(list), '{"version": 1, "cards": {\n\n}}\n', {})
    env = {**__import__("os").environ, "GIT_AUTHOR_NAME": "a", "GIT_AUTHOR_EMAIL": "a@x", "GIT_COMMITTER_NAME": "a",
           "GIT_COMMITTER_EMAIL": "a@x"}
    for args in (("init", "-q", "-b", "data"), ("add", "-A"), ("commit", "-q", "-m", "good")):
        subprocess.run(["git", "-C", str(data), *args], check=True, env=env, capture_output=True)
    good = subprocess.run(["git", "-C", str(data), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    cdb.write_db(str(data / "db"), {"A": _card(ability="Bad.")}, defaultdict(list), '{"version": 1, "cards": {\n\n}}\n', {})
    assert cdb.main(["export-pickles", good, str(tmp_path / "out"), "--git", str(data)]) == 0
    assert pickle.loads((tmp_path / "out" / "stats.pkl").read_bytes())["A"].ability == "Good."
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == ["old_stats.pkl", "psd_details.json", "stats.pkl"]
    with pytest.raises(ValueError):
        cdb.export_pickles("f" * 40, str(tmp_path / "out2"), str(data))


# ========================
# The live pickles (read only)
# ========================
def test_live_round_trip():
    """Every card and version of the committed pickles survives record -> CardInfo -> record, field for field."""
    if not (REPO / "stats.pkl").exists() or not (REPO / "old_stats.pkl").exists():
        pytest.skip("no pickles")
    stats, old = cdb.read_pickles(str(REPO))
    assert len(stats) > 100 and cdb.round_trip_problems(stats, old) == []
    for card in list(stats.values()) + [v for versions in old.values() for v in versions]:
        back = cdb.card(cdb.record(card))
        assert all(getattr(back, f.name) == (getattr(card, f.name) if hasattr(card, f.name) else back.__dict__[f.name])
                   for f in fields(cm.CardInfo)), card.name
    assert all(repr(cdb.card(cdb.record(c))) == repr(c) for c in stats.values())
