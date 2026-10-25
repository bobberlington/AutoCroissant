"""Subseries: a database-only card field (Card Edits.json, never the PSD)."""
from types import SimpleNamespace

import pytest

import commands.card_models as cm
import commands.query_card as qc
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

