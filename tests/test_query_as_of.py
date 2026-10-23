"""/query as_of:<date>: the card at the end of that day in TIMEZONE, from fake histories (old_stats, then the current card;
archived cards from their tag)."""
import inspect
from collections import defaultdict
from datetime import date, datetime
from types import SimpleNamespace

import pytest

import commands.card_models as cm
import commands.help as help_module
import commands.query_card as qc
import commands.utils as utils
from global_config import SITE_URL, TIMEZONE

BETHANY = "Creatures/Isaac/4 Stars/Tainted_Bethany"
ZOEY = "Creatures/Other/5 Stars/ZOEY!!!!!"


def _at(*when) -> float:
    return datetime(*when, tzinfo=TIMEZONE).timestamp()


def _card(name, path, when, ability, blob=None, hp=7, **kw):
    return cm.CardInfo(name=name, card_type="creature", path=f"{path}.psd", timestamp=when, ability=ability, stars=4,
                       stats=cm.CardStats(hp, 4, 5, 5), blob_sha=blob, **kw)


def _bethany():
    """Tainted Bethany's shape on 2026-10-22: 4 old versions (the last two in the same second), then the current card."""
    old = [_card("Tainted Bethany", BETHANY, _at(2026, 1, 30, 16, 5, 18), "First text.", hp=6),
           _card("Tainted Bethany", BETHANY, _at(2026, 2, 7, 20, 0, 34), "Second text."),
           _card("Tainted Bethany", BETHANY, _at(2026, 10, 19, 16, 57, 41), "Third text."),
           _card("Tainted Bethany", BETHANY, _at(2026, 10, 19, 16, 57, 41), "Fourth text.", blob="c" * 40)]
    return old, _card("Tainted Bethany", BETHANY, _at(2026, 10, 21, 22, 39, 53), "Current text.", blob="d" * 40)


def _zoey(**tag):
    old = [_card("ZOEY!!!!!", ZOEY, _at(2026, 1, 30, 22, 40, 50), "Zoey before."),
           _card("ZOEY!!!!!", ZOEY, _at(2026, 2, 7, 20, 11, 26), "Zoey last.")]
    if tag:
        old[1].archived = {"at": _at(2026, 6, 18, 10, 0), "path": f"{ZOEY}.psd", "blob": None, "commit": None, **tag}
    return old


@pytest.fixture
def db(monkeypatch):
    """Bethany listed with history, ZOEY!!!!! archived (PNG gone), a duplicate of Bethany and a type icon PNG."""
    old, current = _bethany()
    repo = qc.card_repo
    monkeypatch.setattr(repo, "git_files", {"tainted_bethany.png": f"{BETHANY}.png", "bethany_copy.png": "Extra/Bethany_Copy.png",
                                            "fire.png": "Types/Fire.png"})
    monkeypatch.setattr(repo, "site_ids", {"tainted_bethany.png": ("tainted-bethany", "")})
    monkeypatch.setattr(repo, "ambiguous_names", {})
    monkeypatch.setattr(repo, "git_file_alias", {})
    monkeypatch.setattr(repo, "match_ratio", 0.6)
    monkeypatch.setattr(repo, "get_card_url", lambda name: f"https://img/{name}")
    copy = _card("Bethany Copy", "Extra/Bethany_Copy", current.timestamp, "Current text.", blob="d" * 40,
                 duplicate_of="Tainted Bethany")
    monkeypatch.setattr(cm.stats_db, "stats", {"Tainted Bethany": current, "Bethany Copy": copy})
    monkeypatch.setattr(cm.stats_db, "old_stats", defaultdict(list, {"Tainted Bethany": old,
                                                                     "ZOEY!!!!!": _zoey(commit="35adb383" + "0" * 32)}))
    utils.dispatch_queue.clear()
    yield
    utils.dispatch_queue.clear()


def _ask(query, as_of):
    utils.dispatch_queue.clear()
    qc.query_name(SimpleNamespace(), query, as_of)
    return [kw for _, kw in utils.dispatch_queue]


def _texts(query, as_of):
    return [kw["content"] for kw in _ask(query, as_of)]


HEAD = f"(end of day, {TIMEZONE.key}): "


# ========================
# Dates
# ========================
def test_a_date_is_the_end_of_that_day_in_the_timezone():
    day, end, why = qc.parse_as_of(" 2026-02-01 ", date(2026, 10, 22))
    assert (day, why) == (date(2026, 2, 1), "") and end == _at(2026, 2, 2)
    assert qc.parse_as_of("2026-3-8", date(2026, 10, 22))[1] == _at(2026, 3, 9)        # spring forward: a 23-hour day
    assert qc.parse_as_of("2026-10-22", date(2026, 10, 22))[2] == ""                   # today is allowed


@pytest.mark.parametrize("text", ["2026-02-30", "2026-13-01", "02/01/2026", "yesterday", "2026-02-01T10:00", "0000-01-01"])
def test_invalid_dates_are_answered_plainly(text):
    day, end, why = qc.parse_as_of(text, date(2026, 10, 22))
    assert day is None and why == "That isn't a date: use YYYY-MM-DD, e.g. 2026-02-01."


def test_future_dates_are_answered_plainly():
    assert qc.parse_as_of("2026-10-23", date(2026, 10, 22)) == (
        None, 0.0, f"2026-10-23 is in the future (today is 2026-10-22, {TIMEZONE.key}).")


def test_the_latest_version_not_after_the_date():
    cards = [SimpleNamespace(timestamp=t) for t in (10.0, 20.0, 20.0, 30.0)]
    assert [qc.version_at(cards, end) for end in (5.0, 10.0, 10.5, 20.5, 31.0)] == [None, None, 0, 2, 3]


def test_a_version_starts_at_its_record():
    """A duplicate dropped into a version (card_history.carry): it starts at the record's from, not its timestamp."""
    cards = [SimpleNamespace(timestamp=10.0), SimpleNamespace(timestamp=30.0, version={"from": 20.0, "from_commit": None})]
    assert [qc.version_at(cards, end) for end in (15.0, 25.0, 31.0)] == [0, 1, 1]


# ========================
# Replies
# ========================
def test_an_old_version_shows_its_data_and_dates(db):
    sent = _ask("Tainted Bethany", "2026-02-01")
    assert len(sent) == 1
    text = sent[0]["content"]
    assert text.startswith(f"**Tainted Bethany** as of 2026-02-01 {HEAD}version 1 of 5, from 2026-01-30 (date approximate) "
                           "until 2026-02-07.\n" + qc.AS_OF_EDITS_NOTE + "\n```")
    assert "ability: First text." in text and "hp: 6" in text and "Current text." not in text and "https://img/" not in text
    assert sent[0]["links"] == [("Card page", f"{SITE_URL}/c/tainted-bethany/")]


def test_the_day_is_read_in_the_timezone(db):
    """16:05 Pacific on Jan 30 is Jan 31 in UTC: Jan 30 already has the first version."""
    assert "version 1 of 5, from 2026-01-30" in _texts("Tainted Bethany", "2026-01-30")[0]


def test_two_versions_in_one_second_show_the_later(db):
    text = _texts("tainted bethany", "2026-10-19")[0]
    assert f"{HEAD}version 4 of 5, from 2026-10-19 until 2026-10-21.\n" in text and "ability: Fourth text." in text


def test_the_current_card_posts_its_image(db):
    sent = _ask("Tainted Bethany", "2026-10-21")
    assert [kw["content"] for kw in sent] == [
        f"**Tainted Bethany** as of 2026-10-21 {HEAD}the current card (version 5 of 5), from 2026-10-21.\n{qc.AS_OF_EDITS_NOTE}",
        "https://img/tainted_bethany.png"]


def test_the_current_card_from_its_carried_start(db):
    """Its last old version had the same data (dropped): the current card answers from that version's date."""
    current = cm.stats_db.stats["Tainted Bethany"]
    current.version = {"from": _at(2026, 10, 19, 16, 57, 41), "from_commit": None, "to": None, "to_commit": None,
                       "cause": "legacy", "png_commit": None, "legacy": False}
    cm.stats_db.old_stats["Tainted Bethany"].pop()
    assert _texts("Tainted Bethany", "2026-10-20")[0].startswith(
        f"**Tainted Bethany** as of 2026-10-20 {HEAD}the current card (version 4 of 4), from 2026-10-19.")
    assert f"version 2 of 4, from 2026-02-07 (date approximate) until 2026-10-19." in _texts("Tainted Bethany", "2026-02-07")[0]


def test_before_the_oldest_version(db):
    assert _texts("Tainted Bethany", "2025-12-31") == [
        f"**Tainted Bethany** as of 2025-12-31 {HEAD}no stored version is that old; the oldest is from 2026-01-30 "
        "(date approximate)."]


def test_archived_after_its_archive(db):
    """Found by name though its PNG is gone; the reply says when and in which commit the PSD was deleted."""
    sent = _ask("zoey", "2026-07-01")
    assert len(sent) == 1 and sent[0]["links"] is None
    text = sent[0]["content"]
    assert text.startswith(f"**ZOEY!!!!!** as of 2026-07-01 {HEAD}archived on 2026-06-18 (PSD deleted in 35adb38). "
                           "Its last version (version 2 of 2), from 2026-02-07 (date approximate):\n")
    assert "ability: Zoey last." in text


def test_archived_before_its_archive(db):
    assert f"{HEAD}version 2 of 2, from 2026-02-07 (date approximate) until it was archived on 2026-06-18 " \
           "(PSD deleted in 35adb38).\n" in _texts("ZOEY!!!!!", "2026-03-01")[0]
    assert f"{HEAD}version 1 of 2, from 2026-01-30 (date approximate) until 2026-02-07." in _texts("ZOEY!!!!!", "2026-02-01")[0]
    assert _texts("ZOEY!!!!!", "2026-06-18")[0].startswith(f"**ZOEY!!!!!** as of 2026-06-18 {HEAD}archived on 2026-06-18")


def test_a_backfilled_archive_says_by(db, monkeypatch):
    """A backfill's date is the first archiving run's: the PSD was deleted by then."""
    monkeypatch.setitem(cm.stats_db.old_stats, "ZOEY!!!!!", _zoey(backfill=True, renamed_to="Zoey"))
    assert f"{HEAD}archived by 2026-06-18 (PSD deleted, renamed to Zoey). Its last" in _texts("ZOEY!!!!!", "2026-10-01")[0]


def test_an_undated_archive_keeps_the_earlier_versions(db, monkeypatch):
    old = _zoey(commit=None)
    old[1].archived["at"] = None
    monkeypatch.setitem(cm.stats_db.old_stats, "ZOEY!!!!!", old)
    assert f"{HEAD}version 1 of 2, from 2026-01-30 (date approximate) until 2026-02-07." in _texts("ZOEY!!!!!", "2026-02-01")[0]
    assert f"{HEAD}archived on an unknown date (PSD deleted). Its last version (version 2 of 2)" in \
        _texts("ZOEY!!!!!", "2026-03-01")[0]


def test_archived_cards_only_with_as_of(db):
    assert _texts("ZOEY!!!!!", None) == ["No card found!"]


def test_a_duplicate_shows_its_originals_history(db):
    text = _texts("Bethany Copy", "2026-02-10")[0]
    assert text.startswith(f"**Bethany Copy** (same card as Tainted Bethany) as of 2026-02-10 {HEAD}version 2 of 5, ")
    assert "ability: Second text." in text


def test_a_png_without_card_data(db):
    assert _texts("fire", "2026-02-01") == ["**Fire** has no stored card data."]


@pytest.mark.parametrize("as_of, reply", [("2026-02-30", "That isn't a date: use YYYY-MM-DD, e.g. 2026-02-01."),
                                          ("2999-01-01", "2999-01-01 is in the future")])
def test_bad_dates_get_one_plain_line(db, as_of, reply):
    out = _texts("Tainted Bethany", as_of)
    assert len(out) == 1 and out[0].startswith(reply)


def test_no_as_of_is_the_plain_query(db):
    assert _texts("Tainted Bethany", None) == ["https://img/tainted_bethany.png"]
    assert _texts("Tainted Bethany", "  ") == ["https://img/tainted_bethany.png"]


def test_unknown_dates_are_said_so(db, monkeypatch):
    old, current = _bethany()
    old[0].timestamp = datetime(1000, 1, 1).timestamp()           # the traverser's no-commit fallback
    monkeypatch.setitem(cm.stats_db.old_stats, "Tainted Bethany", old)
    assert "version 1 of 5, from an unknown date until 2026-02-07" in _texts("Tainted Bethany", "2026-01-01")[0]


# ========================
# Help and describe strings
# ========================
def test_slash_surface_and_help():
    import main
    cmd = {c.name: c for c in main.tree.get_commands()}["query"]
    params = {p.name: p for p in cmd.parameters}
    assert inspect.signature(cmd.callback).parameters["as_of"].default is None and not params["as_of"].required
    assert "YYYY-MM-DD" in params["as_of"].description and params["as_of"].description.endswith("Default = now.")
    assert all(len(p.description) <= 100 for p in cmd.parameters) and len(cmd.description) <= 100
    entry = help_module.card_commands["/query <query> [as_of]:"]
    assert "- as_of: A date, YYYY-MM-DD" in entry and "archived cards too" in entry and "(default: the current card)" in entry


def test_archived_text_says_where_a_retired_psd_went():
    """White Palace moved into MDW/The Morgue (moved_to): archived, not deleted."""
    tag = {"at": 1792658876.0, "commit": "b6330730235d", "moved_to": "MDW/The Morgue/White_Palace.psd"}
    assert "(PSD moved to MDW/The Morgue in b633073)" in qc._archived_text(tag)
    assert "(PSD deleted in b633073)" in qc._archived_text({**tag, "moved_to": None})


def test_a_png_named_in_another_case_finds_its_card(db, monkeypatch):
    """omori.psd's card is "omori", its PNG Omori.png: /query and as_of find the card (they said no stored data)."""
    monkeypatch.setitem(qc.card_repo.git_files, "omori.png", "Creatures/Omori/5 Stars/Omori.png")
    omori = _card("omori", "Creatures/Omori/5 Stars/omori", _at(2026, 10, 21, 22, 57, 50), "Death counter.")
    monkeypatch.setitem(cm.stats_db.stats, "omori", omori)
    assert qc._stats_card("omori.png") is omori and qc._as_of_match("omori") == ("omori.png", "omori")
    assert "**omori** as of 2026-10-22 " in _texts("omori", "2026-10-22")[0]   # once: no stored card data


def test_an_archived_name_wins_over_an_alias(db, monkeypatch):
    """ZOEY!!!!! renamed to Z!!!!!, with an alias zoey!!!!! -> Z!!!!!'s PNG: as_of for ZOEY!!!!! is the archived card's
    history; Z!!!!!'s own name still finds Z!!!!!."""
    monkeypatch.setitem(qc.card_repo.git_files, "zoey!!!!!.png", "Creatures/Other/5 Stars/Z!!!!!.png")
    monkeypatch.setitem(qc.card_repo.git_files, "z!!!!!.png", "Creatures/Other/5 Stars/Z!!!!!.png")
    assert qc._as_of_match("ZOEY!!!!!") == (None, "ZOEY!!!!!")
    assert qc._as_of_match("Z!!!!!") == ("z!!!!!.png", "Z!!!!!")
