"""Creation dates: site/Card Creation Dates.json (CardDates), the cards' created, the data build and /query_ability's
created>=2025."""
import json
from types import SimpleNamespace

import pytest

import commands.card_data as cd
import commands.card_models as cm
import commands.query_card as qc
from commands.card_edits import CardDates

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
    d = _dates({"Foo_Bar": {"date": "2023-10-13", "from": "discord", "extra": 1},
                "Plain": {"date": "2020-01-02", "from": "git"},
                "Hand": {"date": "2019-12-31", "from": "hand"},
                "Bad day": {"date": "2023-02-30", "from": "discord"}, "Short": {"date": "2023-1-3", "from": "git"},
                "Stamp": {"date": "2023-10-13T05:00:00Z", "from": "git"}, "Who": {"date": "2023-10-13", "from": "chat"},
                "Str": "2023-10-13", "None": None})
    assert d.cards == {"foo bar": {"date": "2023-10-13", "from": "discord"},
                       "plain": {"date": "2020-01-02", "from": "git"}, "hand": {"date": "2019-12-31", "from": "hand"}}


def test_apply_sets_changes_and_clears():
    d, card = _dates({"Foo Bar": {"date": "2023-10-13", "from": "discord"}}), _card()
    assert d.apply("Foo Bar", card) and card.created == {"date": "2023-10-13", "from": "discord"}
    assert not d.apply("foo_bar", card)                          # the same: no change
    d.cards["foo bar"]["date"] = "2023-10-01"
    assert d.apply("Foo Bar", card) and card.created["date"] == "2023-10-01"
    assert d.cards["foo bar"] is not card.created                # a copy
    assert _dates({}).apply("Foo Bar", card) and "created" not in card.__dict__ and card.created is None
    assert not _dates({}).apply("Foo Bar", card)


def test_the_websites_rule_with_the_first_upload():
    """build_site.py apply_created's rule: the entry, unless the first upload is earlier and the entry isn't a hand one;
    no entry: the first upload ("from": "git"); neither: undated."""
    d = _dates({"Posted": {"date": "2023-10-13", "from": "discord"},
                "Late": {"date": "2024-02-02", "from": "discord"},
                "Hand": {"date": "2025-01-01", "from": "hand"}})
    assert d.created("Posted", "2023-11-05") == {"date": "2023-10-13", "from": "discord"}
    assert d.created("Posted", "2023-10-13")["from"] == "discord" and d.created("Posted")["from"] == "discord"
    assert d.created("Late", "2024-01-30") == {"date": "2024-01-30", "from": "git"}
    assert d.created("Hand", "2020-01-01") == {"date": "2025-01-01", "from": "hand"}
    assert d.created("New", "2026-11-01") == {"date": "2026-11-01", "from": "git"} and d.created("New") is None
    card = _card("New")
    assert d.apply("New", card, "2026-11-01") and card.created == {"date": "2026-11-01", "from": "git"}
    assert not d.apply("New", card, "2026-11-01") and d.apply("New", card) and card.created is None


def test_first_upload_is_the_earliest_version_the_bot_holds():
    """Its old_stats versions' starts and its own (a version record's from, else its timestamp), as a Pacific day (the
    bot's TIMEZONE, owner 11-17): 2026-01-05 00:00 UTC is the evening of the 4th."""
    from datetime import datetime, timezone
    at = lambda *a: datetime(*a, tzinfo=timezone.utc).timestamp()
    old = [_card(timestamp=at(2025, 3, 1, 23, 30)), _card(timestamp=at(2025, 6, 1))]
    old[1].version = {"from": at(2024, 12, 31, 23), "to": None}
    card = _card(timestamp=at(2026, 1, 5))
    build = SimpleNamespace(db=SimpleNamespace(old_stats={"Foo Bar": old}))
    assert cd.CardBuild._first_upload(build, "Foo Bar", card) == "2024-12-31"
    assert cd.CardBuild._first_upload(build, "Other", card) == "2026-01-04"


def test_apply_dates_dates_a_card_the_file_misses(capsys):
    from datetime import datetime, timezone
    build = cd.CardBuild.__new__(cd.CardBuild)
    new, old = _card("New Card"), _card("Old Card")
    at = lambda *a: datetime(*a, tzinfo=timezone.utc).timestamp()
    new.timestamp, old.timestamp = at(2026, 11, 1), at(2026, 1, 1)
    build.db = SimpleNamespace(stats={"New Card": new, "Old Card": old}, old_stats={})
    assert build._apply_dates(_dates({"Old Card": {"date": "2020-07-22", "from": "discord"}})) == 2
    assert new.created == {"date": "2026-10-31", "from": "git"} and old.created == {"date": "2020-07-22", "from": "discord"}
    assert "1 cards not in site/Card Creation Dates.json, dated by their first upload" in capsys.readouterr().out
    assert build._apply_dates(None) == 0 and new.created["from"] == "git"           # no file read: as they are


def test_made_before_a_year():
    """Pep Drink and Wham Bam Bone (owner 11-14): 'before': 2020 beside their first dated appearance; the entry is kept
    over any first upload and the bot says "Created before 2020"."""
    d = _dates({"Pep Drink": {"date": "2020-07-21", "from": "discord", "before": 2020},
                "Odd": {"date": "2020-07-21", "from": "discord", "before": "2020"}})
    assert d.created("Pep Drink", "2019-01-02") == {"date": "2020-07-21", "from": "discord", "before": 2020}
    assert d.cards["odd"] == {"date": "2020-07-21", "from": "discord"}            # a year is a number


def test_created_is_no_field():
    from dataclasses import fields
    assert "created" not in {f.name for f in fields(cm.CardInfo)} and cm.CardInfo.created is None
    card = _card()
    assert "created" not in card.to_dict()
    card.created = {"date": "2023-10-13", "from": "git"}
    assert card.to_dict()["created"] == "2023-10-13"
    assert card == _card() and repr(card) == repr(_card())       # no data: eq and repr skip it (CardEdits.apply, history)


# ========================
# The data build
# ========================
def test_unreadable_file_keeps_the_dates(capsys):
    build = cd.CardBuild.__new__(cd.CardBuild)
    build.tree = {"tree": [{"path": FILE}]}
    build._repo_text = lambda path: "{"
    assert build._dates_from_response() is None and "creation dates stay as they are" in capsys.readouterr().out
    build.tree = {"tree": []}
    assert build._dates_from_response() is None


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


def _dated(name, day):
    card = _card(name)
    if day:
        card.created = {"date": day, "from": "discord"}
    return card


def test_created_queries_at_the_precision_given(repo):
    r = repo({"Old": _dated("Old", "2019-02-13"), "Mid": _dated("Mid", "2023-10-13"), "New": _dated("New", "2025-05-05"),
              "Undated": _dated("Undated", None)})
    q = lambda text: sorted(r.ability_search_engine(text).index)
    assert q("created>=2025") == ["New"] and q("created>2023") == ["New"] and q("created<2023") == ["Old"]
    assert q("created==2023-10") == ["Mid"] and q("created==2023-10-13") == ["Mid"] and q("created<=2023-10-12") == ["Old"]
    assert q("CREATED>=2019 & created<2024") == ["Mid", "Old"] and q("created~2025-05") == ["New"]
    assert q("created==2024") == []
    pep = _dated("Pep", "2020-07-21")
    pep.created["before"] = 2020                                  # "before 2020": only < or <= it surely meets
    r = repo({"Old": _dated("Old", "2019-02-13"), "Pep": pep})
    assert q("created<2020") == ["Old", "Pep"] and q("created<=2019-12") == ["Old", "Pep"] and q("created<2019-06") == ["Old"]
    assert q("created>=2020") == [] and q("created==2020-07") == [] and q("created==2019") == ["Old"]


def test_query_embed_says_created(monkeypatch):
    sent = []
    monkeypatch.setattr(qc, "queue_message", lambda inter, content, **kw: sent.append((content, kw)))
    monkeypatch.setattr(qc.card_repo, "git_files", {"goombella.png": "Creatures/Mario/2 Stars/Goombella.png",
                                                    "kirby.png": "Creatures/Kirby/2 Stars/Kirby.png",
                                                    "nodata.png": "Creatures/Kirby/2 Stars/Nodata.png"})
    monkeypatch.setattr(qc.card_repo, "site_ids", {})
    monkeypatch.setattr(qc.card_repo, "ambiguous_names", {})
    goombella, kirby = _dated("Goombella", "2023-10-13"), _dated("Kirby", "2020-07-22")
    goombella.series, goombella.subseries, kirby.series = "mario", "Paper Mario", "kirby"
    monkeypatch.setattr(cm.stats_db, "stats", {"Goombella": goombella, "Kirby": kirby})
    qc.query_name(SimpleNamespace(), "Goombella")
    assert sent[0][1]["embed"].description == "mario · Paper Mario\nCreated 2023-10-13"
    qc.query_name(SimpleNamespace(), "Kirby")                   # series and created only: an embed too
    assert sent[1][1]["embed"].description == "kirby\nCreated 2020-07-22"
    kirby.created["before"] = 2020                              # a physical card photo's (Pep Drink)
    qc.query_name(SimpleNamespace(), "Kirby")
    assert sent.pop()[1]["embed"].description == "kirby\nCreated before 2020"
    qc._post_card(SimpleNamespace(), "nodata.png")              # no data: the plain link
    assert "embed" not in sent[2][1] and sent[2][0].endswith("/Nodata.png")


def test_metadata_and_exports_show_created(monkeypatch, tmp_path):
    """to_dict's created: the metadata view (/query as_of) lists it after timestamp, both exports
    carry it (CSV: a created column). The exports keep non-ASCII and write creatures as hp/def/atk/spd."""
    import csv
    import commands.psd_analyzer as pa
    goombella, kirby, pep = _dated("Goombella", "2023-10-13"), _dated("Kirby", None), _dated("Pep Drink", "2020-07-21")
    pep.created["before"] = 2020                                # owner 11-14: "before 2020", as the embed says
    twins = _card("Twins", creatures=[cm.CardStats(7, 4, 5, 6), cm.CardStats(7, 10, 4, 4)], types=["light", "undead"],
                  display_name="Twin Emperors")
    twins.ability = "Uses: ∞"
    monkeypatch.setattr(pa, "get_card_aliases", lambda name=None: [] if name else {})
    lines = pa._format_metadata(goombella, "Goombella").split("\n")
    assert lines[lines.index("created: 2023-10-13") - 1].startswith("timestamp: ")
    assert "created: before 2020" in pa._format_metadata(pep, "Pep Drink").split("\n")
    assert pep.to_dict()["created"] == "2020-07-21"             # the gates' to_dict keeps the date
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(pa, "queue_file", lambda inter, f: None)
    monkeypatch.setattr(pa.stats_db, "stats", {"Goombella": goombella, "Kirby": kirby, "Twins": twins, "Pep Drink": pep})
    pa.export_stats_to_file(SimpleNamespace(), as_csv=True)
    rows = {r["name"]: r for r in csv.DictReader(open("stats.csv", encoding="utf-8"))}
    assert rows["Goombella"]["created"] == "2023-10-13" and rows["Kirby"]["created"] == ""
    assert (rows["Twins"]["title"], rows["Twins"]["creatures"], rows["Twins"]["types"]) == \
        ("Twin Emperors", "7/4/5/6; 7/10/4/4", "light, undead")       # the title, every creature, plain types
    assert (rows["Kirby"]["title"], rows["Kirby"]["creatures"]) == ("Kirby", "")
    assert rows["Pep Drink"]["created"] == "before 2020"
    pa.export_stats_to_file(SimpleNamespace(), as_csv=False)
    text = open("stats.txt", encoding="utf-8").read()
    assert "\ncreated: 2023-10-13\n" in text and "\nability: Uses: ∞\n" in text
    assert "\ncreatures: 7/4/5/6; 7/10/4/4\n" in text and "\nTwin Emperors (file: Twins)\n" in text
    assert "\ncreated: before 2020\n" in text and "2020-07-21" not in text
    monkeypatch.setattr(pa.stats_db, "rulebook", [{"num": 2, "title": "Setting Up", "text": "★★★ Deck\nfrom −9 up"}])
    pa.export_rulebook_to_file(SimpleNamespace())
    assert "2. Setting Up\n★★★ Deck\nfrom −9 up\n" in open("rules.txt", encoding="utf-8").read()


def test_text_exports_title_name_fields_and_only_real_aliases(monkeypatch, tmp_path):
    """Both text exports: each card under its printed title (its file name beside it when they differ), sorted by
    it; an Aliases line only for a card that has aliases (a Card Edits entry without them gives none); the full
    export names each field in COLUMN_ORDER, the timestamp as its Pacific day."""
    import commands.psd_analyzer as pa
    from commands.card_edits import CardEdits
    moab = cm.CardInfo(name="PoD08MOAB", card_type="minion", path="Auxiliary/Minions/PoD08MOAB.psd",
                       timestamp=1794381853.0, ability="Pop.", stars=4, types=["wind", "offensive_minion"],
                       stats=cm.CardStats(5, 4, 5, 4), display_name="M.O.A.B.")     # 2026-11-11 07:24 UTC
    totem = cm.CardInfo(name="Air Totem", card_type="minion", path="Auxiliary/Minions/Air_Totem.psd", ability="Gust.")
    nimbus = cm.CardInfo(name="Nimbus", card_type="minion", path="Auxiliary/Minions/Nimbus.psd", ability="")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(pa, "queue_file", lambda inter, f: None)
    monkeypatch.setattr(pa.stats_db, "stats", {"PoD08MOAB": moab, "Air Totem": totem, "Nimbus": nimbus})
    monkeypatch.setattr(qc.card_repo, "git_files", {"pod08moab.png": "Auxiliary/Minions/PoD08MOAB.png",
                                                    "air_totem.png": "Auxiliary/Minions/Air_Totem.png"})
    monkeypatch.setattr(qc.card_repo, "_edits", CardEdits.parse(json.dumps({"cards": {
        "PoD08MOAB": {"aliases": ["moab"]}, "Air Totem": {"series": "World of Warcraft"}}})))
    pa.export_stats_to_file(SimpleNamespace(), only_ability=True, as_csv=False)
    assert open("stats.txt", encoding="utf-8").read() == (
        "=== MINION ===\n\nAir Totem\nGust.\n\nM.O.A.B. (file: PoD08MOAB)\nAliases: moab\nPop.\n\nNimbus\n\n")
    pa.export_stats_to_file(SimpleNamespace(), as_csv=False)
    blocks = open("stats.txt", encoding="utf-8").read().split("\n\n")
    assert blocks[2].split("\n") == [
        "M.O.A.B. (file: PoD08MOAB)", "aliases: moab", "type: minion", "ability: Pop.", "hp: 5", "def: 4", "atk: 5",
        "spd: 4", "types: wind, offensive_minion", "path: Auxiliary/Minions/PoD08MOAB.psd", "timestamp: 2026-11-10",
        "stars: 4"]
    assert "Aliases" not in blocks[1] and "aliases" not in blocks[1] and blocks[3].startswith("Nimbus\ntype: minion\n")
