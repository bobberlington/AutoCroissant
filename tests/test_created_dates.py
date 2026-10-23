"""Creation dates: site/Card Creation Dates.json (CardDates), the cards' created, the data build, /query_ability's
created>=2025 and `python -m commands.discord_edits created`."""
import json
import os
import pickle
from types import SimpleNamespace

import pytest

import commands.card_data as cd
import commands.card_db as cdb
import commands.card_models as cm
import commands.discord_edits as de
import commands.query_card as qc
from commands.card_edits import CardDates
from test_card_data import _build, _read, cards, commit, put     # noqa: F401 (the cards fixture)

FILE = "site/Card Creation Dates.json"


def _dates(cards: dict) -> CardDates:
    return CardDates.parse(json.dumps({"about": "x", "cards": cards}))


def _card(name="Foo Bar", **kw):
    return cm.CardInfo(name=name, card_type="creature", path=f"Creatures/X/2 Stars/{name.replace(' ', '_')}.psd",
                       ability="Does a thing.", **kw)


# ========================
# The file
# ========================
def test_parse_keeps_valid_entries_only():
    d = _dates({"Foo_Bar": {"date": "2023-10-13", "from": "discord", "provisional": True, "extra": 1},
                "Plain": {"date": "2020-01-02", "from": "git", "provisional": False},
                "Hand": {"date": "2019-12-31", "from": "hand"},
                "Bad day": {"date": "2023-02-30", "from": "discord"}, "Short": {"date": "2023-1-3", "from": "git"},
                "Stamp": {"date": "2023-10-13T05:00:00Z", "from": "git"}, "Who": {"date": "2023-10-13", "from": "chat"},
                "Flag": {"date": "2023-10-13", "from": "git", "provisional": "yes"}, "Str": "2023-10-13", "None": None})
    assert d.cards == {"foo bar": {"date": "2023-10-13", "from": "discord", "provisional": True},
                       "plain": {"date": "2020-01-02", "from": "git"}, "hand": {"date": "2019-12-31", "from": "hand"}}


@pytest.mark.parametrize("text", ["[]", "null", '{"cards": []}', '{"other": 1}'])
def test_odd_shapes_give_no_dates(text):
    assert CardDates.parse(text).cards == {}


def test_not_json_raises():
    with pytest.raises(ValueError):
        CardDates.parse("{")


def test_apply_sets_changes_and_clears():
    d, card = _dates({"Foo Bar": {"date": "2023-10-13", "from": "discord", "provisional": True}}), _card()
    assert d.apply("Foo Bar", card) and card.created == {"date": "2023-10-13", "from": "discord", "provisional": True}
    assert not d.apply("foo_bar", card)                          # the same: no change
    d.cards["foo bar"]["date"] = "2023-10-01"
    assert d.apply("Foo Bar", card) and card.created["date"] == "2023-10-01"
    assert d.cards["foo bar"] is not card.created                # a copy
    assert _dates({}).apply("Foo Bar", card) and "created" not in card.__dict__ and card.created is None
    assert not _dates({}).apply("Foo Bar", card)


def test_the_websites_rule_with_the_first_upload():
    """build_site.py apply_created's rule: the entry, unless the first upload is earlier and the entry isn't a hand one;
    no entry: the first upload ("from": "git", never provisional); neither: undated."""
    d = _dates({"Posted": {"date": "2023-10-13", "from": "discord", "provisional": True},
                "Late": {"date": "2024-02-02", "from": "discord", "provisional": True},
                "Hand": {"date": "2025-01-01", "from": "hand"}})
    assert d.created("Posted", "2023-11-05") == {"date": "2023-10-13", "from": "discord", "provisional": True}
    assert d.created("Posted", "2023-10-13")["from"] == "discord" and d.created("Posted")["from"] == "discord"
    assert d.created("Late", "2024-01-30") == {"date": "2024-01-30", "from": "git"}
    assert d.created("Hand", "2020-01-01") == {"date": "2025-01-01", "from": "hand"}
    assert d.created("New", "2026-11-01") == {"date": "2026-11-01", "from": "git"} and d.created("New") is None
    card = _card("New")
    assert d.apply("New", card, "2026-11-01") and card.created == {"date": "2026-11-01", "from": "git"}
    assert not d.apply("New", card, "2026-11-01") and d.apply("New", card) and card.created is None


def test_first_upload_is_the_earliest_version_the_bot_holds():
    """Its old_stats versions' starts and its own (a version record's from, else its timestamp), as a UTC day; a stamp
    before 2000 (no commit found) and a rulebook page give none."""
    from datetime import datetime, timezone
    at = lambda *a: datetime(*a, tzinfo=timezone.utc).timestamp()
    old = [_card(timestamp=at(2025, 3, 1, 23, 30)), _card(timestamp=at(2025, 6, 1))]
    old[1].version = {"from": at(2024, 12, 31, 23), "to": None}
    card = _card(timestamp=at(2026, 1, 5))
    build = SimpleNamespace(db=SimpleNamespace(old_stats={"Foo Bar": old}))
    assert cd.CardBuild._first_upload(build, "Foo Bar", card) == "2024-12-31"
    assert cd.CardBuild._first_upload(build, "Other", card) == "2026-01-05"
    build.replaced = {"Other": _card(timestamp=at(2025, 12, 1))}               # replaced this run, not in old_stats yet
    assert cd.CardBuild._first_upload(build, "Other", card) == "2025-12-01"
    build.replaced = {}
    assert cd.CardBuild._first_upload(build, "Other", _card(timestamp=datetime(1000, 1, 1).timestamp())) is None
    page = cm.CardInfo(name="00 Toc", card_type="rulebook", path="Auxiliary/Rulebook/00_Toc.psd", ability="x",
                       timestamp=at(2025, 1, 1))
    assert cd.CardBuild._first_upload(build, "00 Toc", page) is None


def test_apply_dates_dates_a_card_the_file_misses(capsys):
    from datetime import datetime, timezone
    build = cd.CardBuild.__new__(cd.CardBuild)
    new, old = _card("New Card"), _card("Old Card")
    at = lambda *a: datetime(*a, tzinfo=timezone.utc).timestamp()
    new.timestamp, old.timestamp = at(2026, 11, 1), at(2026, 1, 1)
    build.db = SimpleNamespace(stats={"New Card": new, "Old Card": old}, old_stats={})
    assert build._apply_dates(_dates({"Old Card": {"date": "2020-07-22", "from": "discord"}})) == 2
    assert new.created == {"date": "2026-11-01", "from": "git"} and old.created == {"date": "2020-07-22", "from": "discord"}
    assert "1 cards not in site/Card Creation Dates.json, dated by their first upload" in capsys.readouterr().out
    assert build._apply_dates(None) == 0 and new.created["from"] == "git"           # no file read: as they are


def test_created_is_no_field_and_undated_cards_pickle_as_before():
    from dataclasses import fields
    assert "created" not in {f.name for f in fields(cm.CardInfo)} and cm.CardInfo.created is None
    card = _card()
    assert b"created" not in pickle.dumps(card) and "created" not in card.to_dict()
    card.created = {"date": "2023-10-13", "from": "git"}
    back = pickle.loads(pickle.dumps(card))
    assert back.created == card.created and back.to_dict()["created"] == "2023-10-13"
    assert back == _card() and repr(back) == repr(_card())       # no data: eq and repr skip it (CardEdits.apply, history)


def test_record_round_trip():
    card = _card()
    card.created = {"date": "2023-10-13", "from": "discord", "provisional": True}
    rec = cdb.record(card)
    assert rec["created"] == card.created and cdb.card(rec).created == card.created
    assert cdb.record(_card())["created"] is None and "created" not in cdb.card(cdb.record(_card())).__dict__


# ========================
# The data build
# ========================
def test_build_dates_the_cards_with_no_version_or_event(cards):
    assert _build(cards, "--seed", str(cards.seed))[0] == cd.EXIT_OK
    put(cards.repo, FILE, json.dumps({"cards": {"Mini Doomer": {"date": "2020-07-22", "from": "discord", "provisional": True},
                                                "The_Freezer": {"date": "2023-11-05", "from": "git"},
                                                "Nobody": {"date": "2021-01-01", "from": "git"}}}).encode())
    c2 = commit(cards.repo, "MichaelJSr", 2000)
    code, _ = _build(cards)
    stats, old, _, manifest, events = _read(cards)
    assert code == cd.EXIT_OK and manifest["card_commit"] == c2 and manifest["inputs"]["created_blob"]
    assert stats["Mini Doomer"].created == {"date": "2020-07-22", "from": "discord", "provisional": True}
    assert stats["The Freezer"].created == {"date": "2023-11-05", "from": "git"}
    assert not old and [e["card"] for e in events] == ["Mini Doomer", "The Freezer"]      # the two "new" of the seed build
    put(cards.repo, FILE, json.dumps({"cards": {"The Freezer": {"date": "2023-11-04", "from": "hand"}}}).encode())
    commit(cards.repo, "MichaelJSr", 3000)
    assert _build(cards)[0] == cd.EXIT_OK
    stats, old, _, _, events = _read(cards)
    assert stats["Mini Doomer"].created is None and stats["The Freezer"].created == {"date": "2023-11-04", "from": "hand"}
    assert not old and len(events) == 2


def test_unreadable_file_keeps_the_dates(capsys):
    build = cd.CardBuild.__new__(cd.CardBuild)
    build.tree = {"tree": [{"path": FILE}]}
    build._repo_text = lambda path: "{"
    assert build._dates_from_response(None, "r") is None and "creation dates stay as they are" in capsys.readouterr().out
    build.tree = {"tree": []}
    assert build._dates_from_response(None, "r") is None


# ========================
# /query_ability and /query
# ========================
@pytest.fixture
def repo(monkeypatch):
    def make(stats):
        monkeypatch.setattr(cm.stats_db, "stats", stats)
        r = qc.CardRepository()
        r.prep_dataframes()
        return r
    return make


def _dated(name, day, provisional=False):
    card = _card(name)
    if day:
        card.created = {"date": day, "from": "discord", **({"provisional": True} if provisional else {})}
    return card


def test_created_queries_at_the_precision_given(repo):
    r = repo({"Old": _dated("Old", "2019-02-13"), "Mid": _dated("Mid", "2023-10-13"), "New": _dated("New", "2025-05-05"),
              "Undated": _dated("Undated", None)})
    q = lambda text: sorted(r.ability_search_engine(text).index)
    assert q("created>=2025") == ["New"] and q("created>2023") == ["New"] and q("created<2023") == ["Old"]
    assert q("created==2023-10") == ["Mid"] and q("created==2023-10-13") == ["Mid"] and q("created<=2023-10-12") == ["Old"]
    assert q("CREATED>=2019 & created<2024") == ["Mid", "Old"] and q("created~2025-05") == ["New"]
    assert q("created==2024") == []


def test_spaces_around_the_operator_as_the_website(repo):
    """search.js allows spaces around a condition's operator (created >= 2025, hp > 5, type == creature); so does the bot."""
    old, new = _dated("Old", "2019-02-13"), _dated("New", "2025-05-05")
    old.stats, new.stats, new.series = cm.CardStats(3, 1, 1, 1), cm.CardStats(7, 1, 1, 1), "Kirby"
    r = repo({"Old": old, "New": new})
    q = lambda text: sorted(r.ability_search_engine(text).index)
    assert q("created >= 2025") == ["New"] == q("created>=2025") and q("created == 2019-02") == ["Old"]
    assert q("hp > 5") == ["New"] and q("hp >=3 & created< 2020") == ["Old"]
    assert q("type == creature") == ["New", "Old"] and q("series ~ kir") == ["New"] and q("series == kirby") == ["New"]


def test_created_column_when_no_card_has_one(repo):
    r = repo({"Undated": _dated("Undated", None)})
    found = r._parse_stat_query("created>=2025")
    assert "created" in r.cards_dff.columns and found is not None and found.empty


def test_query_embed_says_created(monkeypatch):
    sent = []
    monkeypatch.setattr(qc, "queue_message", lambda inter, content, **kw: sent.append((content, kw)))
    monkeypatch.setattr(qc.card_repo, "git_files", {"goombella.png": "Creatures/Mario/2 Stars/Goombella.png",
                                                    "kirby.png": "Creatures/Kirby/2 Stars/Kirby.png"})
    monkeypatch.setattr(qc.card_repo, "site_ids", {})
    monkeypatch.setattr(qc.card_repo, "ambiguous_names", {})
    goombella = _dated("Goombella", "2023-10-13", provisional=True)
    goombella.series, goombella.subseries = "mario", "Paper Mario"
    monkeypatch.setattr(cm.stats_db, "stats", {"Goombella": goombella, "Kirby": _dated("Kirby", "2020-07-22")})
    qc.query_name(SimpleNamespace(), "Goombella")
    assert sent[0][1]["embed"].description == "mario · Paper Mario\nCreated about 2023-10-13"
    qc.query_name(SimpleNamespace(), "Kirby")                   # a plain link stays one
    assert "embed" not in sent[1][1]


# ========================
# python -m commands.discord_edits created
# ========================
ARCHIVE = {
    "A Snack": {"date": "2023-10-13", "source": "discord", "confidence": "high", "provisional": True,
                "first_discord": "2023-10-13", "id": "a-snack", "waits_on": ["A173"]},
    "Abigail": {"date": "2020-08-13", "source": "discord", "confidence": "confident", "provisional": False,
                "first_discord": "2020-08-13", "id": "abigail"},
    "Exposed": {"date": "2023-11-06", "source": "git-first-upload", "confidence": "discord-later", "provisional": False,
                "first_discord": "2023-12-08", "id": "exposed"},
    "Nothing": {"date": None, "source": None, "confidence": None, "provisional": False, "first_discord": None, "id": "nothing"},
}


def test_dates_file_carries_dates_only_and_keeps_hand_entries():
    text, report = de.dates_file(ARCHIVE, None)
    data = json.loads(text)
    assert data["cards"] == {"A Snack": {"date": "2023-10-13", "from": "discord", "provisional": True},
                             "Abigail": {"date": "2020-08-13", "from": "discord"}, "Exposed": {"date": "2023-11-06", "from": "git"}}
    assert text.splitlines()[3] == '    "A Snack": {"date":"2023-10-13","from":"discord","provisional":true},'   # dump_edits' layout
    assert report == ["3 creation dates: 2 first Discord post, 1 first git upload, 0 by hand; 1 provisional", "a new file: 3 cards"]
    assert de.dates_file(ARCHIVE, text) == (text, [report[0], "vs the file: 0 new, 0 changed, 0 gone"])
    hand = json.loads(text)
    hand["cards"]["exposed"] = hand["cards"].pop("Exposed") | {"date": "2023-11-01", "from": "hand"}
    hand["cards"]["Gone"] = {"date": "2021-01-01", "from": "git"}
    moved = dict(ARCHIVE, Abigail=dict(ARCHIVE["Abigail"], date="2020-08-01"), **{"A Snack": dict(ARCHIVE["A Snack"], provisional=False)})
    text2, report2 = de.dates_file(moved, json.dumps(hand))
    assert json.loads(text2)["cards"] == {"A Snack": {"date": "2023-10-13", "from": "discord"},
                                          "Abigail": {"date": "2020-08-01", "from": "discord"},
                                          "exposed": {"date": "2023-11-01", "from": "hand"}}
    assert report2[1:] == ["vs the file: 0 new, 2 changed, 1 gone", "- A Snack: 2023-10-13 no longer provisional",
                           "- Abigail: 2020-08-13 -> 2020-08-01", "- gone: Gone"]
    assert "waits_on" not in text and "a-snack" not in text and "confident" not in text


def test_created_command(tmp_path, capsys):
    archive, repo = tmp_path / "archive", tmp_path / "cards"
    archive.mkdir()
    (archive / "created_dates.json").write_text(json.dumps(ARCHIVE), encoding="utf-8")
    assert de.main(["created", "--archive", str(archive), "--repo", str(repo)]) == 1      # no site/ yet: written below
    (repo / "site").mkdir(parents=True)
    assert de.main(["created", "--archive", str(archive), "--repo", str(repo)]) == 0
    out = repo / FILE
    assert "Wrote" in capsys.readouterr().out and CardDates.parse(out.read_text(encoding="utf-8")).cards["a snack"]["provisional"]
    assert de.main(["created", "--archive", str(archive), "--repo", str(repo)]) == 0 and "Unchanged:" in capsys.readouterr().out
    (archive / "matches.json").write_text("{}", encoding="utf-8")
    os.utime(archive / "created_dates.json", (1, 1))
    assert de.main(["created", "--archive", str(archive), "--out", str(tmp_path / "x.json")]) == 1
    assert "review_pages.py" in capsys.readouterr().out and not (tmp_path / "x.json").exists()
    assert de.main(["created", "--archive", str(tmp_path / "none"), "--repo", str(repo)]) == 1
