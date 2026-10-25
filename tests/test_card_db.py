"""card_db: CardInfo records, db/'s JSON files and manifest."""
import hashlib
import json
from collections import defaultdict
from dataclasses import fields

import pytest

import commands.card_db as cdb
import commands.card_models as cm


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
    assert set(rec) == {f.name for f in fields(cm.CardInfo)} | {"version", "created"}
    assert rec["stats"] == {"hp": 5, "defense": 2, "attack": 3, "speed": 4}
    assert rec["creatures"] is None and rec["version"] is None and rec["archived"] is None and rec["created"] is None


def test_round_trip_is_exact():
    card = _full()
    back = cdb.card(cdb.record(card))
    assert back == card and repr(back) == repr(card) and back.version == card.version
    assert cdb.record(back) == cdb.record(card) and json.loads(json.dumps(cdb.record(card))) == cdb.record(card)
    assert isinstance(back.stats, cm.CardStats) and all(isinstance(c, cm.CardStats) for c in back.creatures)
    assert type(back.timestamp) is float and back.timestamp == card.timestamp


def test_version_is_no_field():
    """version is a class default, not a dataclass field (card_db's EXTRA_KEYS)."""
    assert "version" not in {f.name for f in fields(cm.CardInfo)} and cm.CardInfo.version is None
    assert "version" not in _card().__dict__


def test_missing_keys_default_and_unknown_keys_warn_once(capsys, monkeypatch):
    monkeypatch.setattr(cdb, "_warned", set())
    card = cdb.card({"name": "B", "card_type": "item", "sparkle": 1})
    cdb.card({"name": "C", "card_type": "item", "sparkle": 2, "glow": 3})
    assert card.types == [] and card.stats == cm.CardStats() and card.version is None and card.parser_version is None
    out = capsys.readouterr().out
    assert out.count("'sparkle'") == 1 and out.count("'glow'") == 1


def test_a_missing_attribute_takes_the_default():
    """A card without a field's attribute in its __dict__ records the field's default."""
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