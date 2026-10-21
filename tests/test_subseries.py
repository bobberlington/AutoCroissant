"""Subseries: a database-only card field (Card Edits.json or /update_metadata, never the PSD)."""
import inspect
import json
import pickle
import shutil
from types import SimpleNamespace

import pytest

import commands.card_models as cm
import commands.psd_analyzer as pa
import commands.query_card as qc
from conftest import REPO
from golden_cards import parser


def _card(name, series=None, subseries=None, card_type="creature", path=None, ability="Does a thing."):
    return cm.CardInfo(name=name, card_type=card_type, path=path or f"Creatures/{series or 'X'}/2 Stars/{name}.psd",
                       ability=ability, series=series, subseries=subseries)


FAKE = {
    "Goombella": _card("Goombella", "mario", "Paper Mario: The Thousand-Year Door"),
    "King Bob-omb": _card("King Bob-omb", "mario", "Super Mario 64"),
    "Kirby": _card("Kirby", "kirby"),
    "A Snack": _card("A Snack", card_type="item", path="Items/Heal/1 Stars/A_Snack.psd", ability="subseries==x heals."),
}


@pytest.fixture
def repo(monkeypatch):
    """A CardRepository whose dataframe comes from FAKE instead of stats.pkl."""
    def make(cards):
        monkeypatch.setattr(cm.stats_db, "stats", cards)
        r = qc.CardRepository()
        r.prep_dataframes()
        return r
    return make


# ========================
# CardInfo and the pickle
# ========================
def test_to_dict_puts_subseries_next_to_series():
    d = _card("Goombella", "mario", "Paper Mario").to_dict()
    keys = list(d)
    assert d["subseries"] == "Paper Mario" and keys[keys.index("series") + 1] == "subseries"
    assert "subseries" not in _card("Kirby", "kirby").to_dict()
    assert "subseries" not in _card("Kirby", "kirby", "").to_dict()


def test_card_pickled_before_the_field_loads_with_the_default():
    card = _card("Old", "kirby")
    del card.__dict__["subseries"]                              # what an older stats.pkl holds
    old = pickle.loads(pickle.dumps(card))
    assert old.subseries is None and "subseries" not in old.to_dict()
    assert pa.CardEdits().get(old, "subseries") is None


def test_live_pickle_copy_loads(tmp_path):
    src = REPO / "stats.pkl"
    if not src.exists():
        pytest.skip("no stats.pkl")
    with open(shutil.copy(src, tmp_path / "stats.pkl"), "rb") as f:
        stats = pickle.load(f)
    assert len(stats) > 100
    for card in stats.values():
        assert card.subseries is None or isinstance(card.subseries, str)
        assert ("subseries" in card.to_dict()) == bool(card.subseries)


def test_reparse_keeps_a_hand_set_subseries(clone):
    """A re-parse (PSD blob changed) keeps author, series and subseries from the stored card; a fresh parse has none."""
    rel = "Creatures/Mario/3 Stars/Sniffle_Thwomp.psd"
    if not (clone / rel).exists():
        pytest.skip(f"no {rel}")
    psd_parser = parser()
    name = cm.card_name(rel)
    assert psd_parser.parse(str(clone / rel), rel).subseries is None          # never parsed
    stored = _card(name, "Paper Mario", "Paper Mario: The Thousand-Year Door")
    stored.author = "Someone"
    psd_parser.db.stats[name] = stored
    card = psd_parser.parse(str(clone / rel), rel)
    assert (card.author, card.series, card.subseries) == ("Someone", "Paper Mario", "Paper Mario: The Thousand-Year Door")
    del stored.__dict__["subseries"]                                           # stored by an older bot
    assert psd_parser.parse(str(clone / rel), rel).subseries is None


def test_reparse_takes_series_and_subseries_from_before_the_edits(clone):
    """A re-parse keeps the pre-edit values (parsed_values), so removing the website edit later still clears it."""
    rel = "Creatures/Mario/3 Stars/Sniffle_Thwomp.psd"
    if not (clone / rel).exists():
        pytest.skip(f"no {rel}")
    psd_parser = parser()
    name = cm.card_name(rel)
    edits = pa.CardEdits.parse(json.dumps({"cards": {name: {"subseries": "Paper Mario", "series": "Paper Mario"}}}))
    stored = _card(name, "mario")
    edits.apply(name, stored)
    assert (stored.series, stored.subseries) == ("Paper Mario", "Paper Mario")
    psd_parser.db.stats[name] = stored
    card = psd_parser.parse(str(clone / rel), rel)
    assert (card.series, card.subseries) == ("mario", None)                    # the PSD side, not the edit
    edits.apply(name, card)
    assert (card.series, card.subseries) == ("Paper Mario", "Paper Mario")
    pa.CardEdits().apply(name, card)                                           # edit removed from the website
    assert (card.series, card.subseries) == ("mario", None)
    stored.parsed_values = {"subseries": "Super Paper Mario"}                  # hand-set before the edit
    assert psd_parser.parse(str(clone / rel), rel).subseries == "Super Paper Mario"


# ========================
# Query DSL
# ========================
def test_dataframe_has_the_column_and_the_dsl_filters_on_it(repo):
    r = repo(FAKE)
    assert "subseries" in r.cards_dff.columns and r.cards_dff["subseries"].isna()["Kirby"]
    assert list(r.ability_search_engine("subseries==paper mario: the thousand-year door").index) == ["Goombella"]
    assert set(r.ability_search_engine("subseries~mario").index) == {"Goombella", "King Bob-omb"}
    assert set(r.ability_search_engine("SUBSERIES~MARIO").index) == {"Goombella", "King Bob-omb"}
    assert list(r.ability_search_engine("series==mario & subseries~door").index) == ["Goombella"]
    assert set(r.ability_search_engine("subseries~door | series==kirby").index) == {"Goombella", "Kirby"}
    assert list(r.ability_search_engine("series==kirby").index) == ["Kirby"]          # unchanged


def test_column_exists_when_no_card_has_one(repo):
    r = repo({k: v for k, v in FAKE.items() if k in ("Kirby", "A Snack")})
    assert "subseries" in r.cards_dff.columns and r.cards_dff["subseries"].isna().all()
    found = r._parse_stat_query("subseries==x")
    assert found is not None and found.empty                     # a filter, not a text search of "subseries==x"
    assert list(r.ability_search_engine('"subseries==x"').index) == ["A Snack"]


@pytest.mark.parametrize("only", [("Kirby", "A Snack"), None])
def test_empty_cells_never_match_as_text(repo, only):
    """No subseries is None/NaN; pandas < 3 turns those into "None"/"nan" text, which must not match."""
    r = repo({k: v for k, v in FAKE.items() if only is None or k in only})
    for query in ("subseries~n", "subseries~an", "subseries~on", "subseries==none", "subseries==nan"):
        assert not set(r.ability_search_engine(query).index) & {"Kirby", "A Snack"}, query


def test_quoted_values_keep_and_and_or(repo):
    cards = dict(FAKE, Luigi=_card("Luigi", "mario", "Mario & Luigi"), Fire=_card("Fire", ability="Fire & ice | wind."))
    r = repo(cards)
    assert list(r.ability_search_engine('subseries=="mario & luigi"').index) == ["Luigi"]
    assert list(r.ability_search_engine('series==mario & subseries~"& luigi"').index) == ["Luigi"]
    assert set(r.ability_search_engine('subseries=="Mario & Luigi" | series==kirby').index) == {"Luigi", "Kirby"}
    assert list(r.ability_search_engine('"fire & ice | wind"').index) == ["Fire"]
    assert r._split_unquoted('a | "b', "|") == ["a ", ' "b']                  # an unclosed quote still splits before it


# ========================
# Commands
# ========================
def test_update_metadata_and_mass_replace(monkeypatch):
    sent = []
    monkeypatch.setattr(pa, "queue_message", lambda inter, text, **kw: sent.append(text))
    monkeypatch.setattr(cm.stats_db, "stats", {"Goombella": _card("Goombella", "mario")})
    monkeypatch.setattr(cm.stats_db, "save", lambda: None)
    monkeypatch.setattr(qc.card_repo, "git_files", {"goombella.png": "Creatures/Mario/2 Stars/Goombella.png"})
    monkeypatch.setattr(qc.card_repo, "git_file_alias", {})
    inter = SimpleNamespace()
    assert "subseries" in inspect.signature(pa.manual_metadata_entry).parameters
    pa.manual_metadata_entry(inter, "Goombella", subseries="Paper Mario")
    card = cm.stats_db.stats["Goombella"]
    assert card.subseries == "Paper Mario" and "subseries: Paper Mario" in sent[len(sent) - 1]
    pa.manual_metadata_entry(inter, "Goombella")
    assert "series: mario\nsubseries: Paper Mario```" in sent[len(sent) - 1]
    pa.mass_replace_field(inter, "subseries", "paper mario", "Paper Mario 64")
    assert card.subseries == "Paper Mario 64" and "Replaced 1 instances" in sent[len(sent) - 1]
    assert 'subseries' in pa.COLUMN_ORDER and pa.COLUMN_ORDER.index("subseries") == pa.COLUMN_ORDER.index("series") + 1


def test_query_embed_shows_series_and_subseries(monkeypatch):
    sent = []
    monkeypatch.setattr(qc, "queue_message", lambda inter, content, **kw: sent.append((content, kw)))
    monkeypatch.setattr(qc.card_repo, "git_files", {"goombella.png": "Creatures/Mario/2 Stars/Goombella.png",
                                                    "kirby.png": "Creatures/Kirby/2 Stars/Kirby.png"})
    monkeypatch.setattr(qc.card_repo, "site_ids", {})
    monkeypatch.setattr(qc.card_repo, "ambiguous_names", {})
    monkeypatch.setattr(cm.stats_db, "stats", {"Goombella": FAKE["Goombella"], "Kirby": FAKE["Kirby"]})
    qc.query_name(SimpleNamespace(), "Goombella")
    assert sent[0][0] is None and sent[0][1]["embed"].description == "mario · Paper Mario: The Thousand-Year Door"
    qc.query_name(SimpleNamespace(), "Kirby")                   # no subseries, no website data: plain link as before
    assert "embed" not in sent[1][1] and sent[1][0].endswith("/Kirby.png")


def test_slash_surface_and_help():
    import main
    import commands.help as help_module
    cmds = {c.name: c for c in main.tree.get_commands()}
    params = {p.name: p.description for p in cmds["update_metadata"].parameters}
    assert "subseries" in inspect.signature(cmds["update_metadata"].callback).parameters
    assert "Database only" in params["subseries"] and all(len(d) <= 100 for d in params.values())
    assert "subseries" in [c.value for c in cmds["mass_replace"].parameters[0].choices]
    assert "subseries" in cmds["mass_replace"].parameters[0].description
    stats = help_module.stats_commands
    assert "[series] [subseries] [hp]" in next(k for k in stats if k.startswith("/update_metadata"))
    assert "subseries" in stats["/mass_replace <field> <old_value> <new_value>:"]
    assert "subseries==" in help_module.card_commands["Query Syntax:"] and "subseries~" in help_module.card_commands["Query Syntax:"]
