"""/query as_of:<date>: the card at the end of that day in TIMEZONE, from fake histories (old_stats, then the current card;
archived cards from their tag)."""
from collections import defaultdict
from datetime import date, datetime
from types import SimpleNamespace

import pytest

import commands.card_models as cm
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


INVALID_DATES_ARE_ANSWERED_PLAINLY_CASES = ["2026-02-30", "2026-13-01", "02/01/2026", "yesterday", "2026-02-01T10:00", "0000-01-01"]


def test_invalid_dates_are_answered_plainly():
    for text in INVALID_DATES_ARE_ANSWERED_PLAINLY_CASES:
        day, end, why = qc.parse_as_of(text, date(2026, 10, 22))
        assert day is None and why == "That isn't a date: use YYYY-MM-DD, e.g. 2026-02-01."


def test_the_latest_version_not_after_the_date():
    cards = [SimpleNamespace(timestamp=t) for t in (10.0, 20.0, 20.0, 30.0)]
    assert [qc.version_at(cards, end) for end in (5.0, 10.0, 10.5, 20.5, 31.0)] == [None, None, 0, 2, 3]


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
    assert "version 2 of 4, from 2026-02-07 (date approximate) until 2026-10-19." in _texts("Tainted Bethany", "2026-02-07")[0]


def test_archived_after_its_archive(db):
    """Found by name though its PNG is gone; the reply says when and in which commit the PSD was deleted."""
    sent = _ask("ZOEY!!!!!", "2026-07-01")
    assert len(sent) == 1 and sent[0]["links"] is None
    text = sent[0]["content"]
    assert text.startswith(f"**ZOEY!!!!!** as of 2026-07-01 {HEAD}archived on 2026-06-18 (PSD deleted in 35adb38). "
                           "Its last version (version 2 of 2), from 2026-02-07 (date approximate):\n")
    assert "ability: Zoey last." in text


def test_a_duplicate_shows_its_originals_history(db):
    text = _texts("Bethany Copy", "2026-02-10")[0]
    assert text.startswith(f"**Bethany Copy** (same card as Tainted Bethany) as of 2026-02-10 {HEAD}version 2 of 5, ")
    assert "ability: Second text." in text


def test_no_as_of_is_the_plain_query(db):
    assert _texts("Tainted Bethany", None) == ["https://img/tainted_bethany.png"]
    assert _texts("Tainted Bethany", "  ") == ["https://img/tainted_bethany.png"]


# ========================
# Help and describe strings
# ========================
def test_a_png_named_in_another_case_finds_its_card(db, monkeypatch):
    """omori.psd's card is "omori", its PNG Omori.png: /query and as_of find the card (they said no stored data)."""
    monkeypatch.setitem(qc.card_repo.git_files, "omori.png", "Creatures/Omori/5 Stars/Omori.png")
    omori = _card("omori", "Creatures/Omori/5 Stars/omori", _at(2026, 10, 21, 22, 57, 50), "Death counter.")
    monkeypatch.setitem(cm.stats_db.stats, "omori", omori)
    assert qc._stats_card("omori.png") is omori and qc.history_match("omori") == ("omori.png", "omori")
    assert "**omori** as of 2026-10-22 " in _texts("omori", "2026-10-22")[0]   # once: no stored card data


def test_an_archived_name_wins_over_an_alias(db, monkeypatch):
    """ZOEY!!!!! renamed to Z!!!!!, with an alias zoey!!!!! -> Z!!!!!'s PNG: as_of for ZOEY!!!!! is the archived card's
    history; Z!!!!!'s own name still finds Z!!!!!."""
    monkeypatch.setitem(qc.card_repo.git_files, "zoey!!!!!.png", "Creatures/Other/5 Stars/Z!!!!!.png")
    monkeypatch.setitem(qc.card_repo.git_files, "z!!!!!.png", "Creatures/Other/5 Stars/Z!!!!!.png")
    assert qc.history_match("ZOEY!!!!!") == (None, "ZOEY!!!!!")
    assert qc.history_match("Z!!!!!") == ("z!!!!!.png", "Z!!!!!")


# ========================
# The image of that day
# ========================
def _png_clone(tmp_path, monkeypatch, path, times):
    """A bare clone of a card repo whose PNG at path changed at each of times; the bot's history clone."""
    import os
    import subprocess
    import commands.repo_history as rh
    repo, shas = tmp_path / "cards", []
    (repo / os.path.dirname(path)).mkdir(parents=True)

    def git(*args, at=None):
        stamp = {} if at is None else {"GIT_AUTHOR_DATE": f"@{int(at)} +0000", "GIT_COMMITTER_DATE": f"@{int(at)} +0000"}
        return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True,
                              env={**os.environ, "GIT_AUTHOR_NAME": "a", "GIT_AUTHOR_EMAIL": "a@x",
                                   "GIT_COMMITTER_NAME": "a", "GIT_COMMITTER_EMAIL": "a@x", **stamp}).stdout
    git("init", "-q", "-b", "main")
    for i, at in enumerate(times):
        (repo / path).write_bytes(b"png %d" % i)
        git("add", "-A")
        git("commit", "-q", "-m", f"png {i}", at=at)
        shas.append(git("rev-parse", "HEAD").strip())
    subprocess.run(["git", "clone", "-q", "--bare", str(repo), str(tmp_path / "history.git")], check=True,
                   capture_output=True)
    monkeypatch.setattr(rh, "HISTORY_CLONE", str(tmp_path / "history.git"))
    return shas


def test_the_image_of_that_day(db, tmp_path, monkeypatch):
    """Hiveblood: the current card's start carried back to 2026-02 while its PNG was re-rendered 2026-10-22. as_of a
    day in between posts the PNG of that day (pinned to its commit), after the re-render today's; an old version links
    its own image of that day. Without the history clone a carried day says today's image is shown."""
    current = cm.stats_db.stats["Tainted Bethany"]
    current.version = {"from": _at(2026, 2, 7, 21, 0), "from_commit": None, "to": None, "to_commit": None,
                       "cause": "legacy", "png_commit": None, "legacy": False}
    cm.stats_db.old_stats["Tainted Bethany"][1:] = []
    assert _ask("Tainted Bethany", "2026-06-01")[0]["content"].endswith("Image: today's (that day's is not known).")
    first, second = _png_clone(tmp_path, monkeypatch, f"{BETHANY}.png", [_at(2026, 1, 30, 17), _at(2026, 10, 22, 7)])
    sent = _ask("Tainted Bethany", "2026-06-01")
    assert sent[0]["content"].endswith("the current card (version 2 of 2), from 2026-02-07.\n"
                                       f"{qc.AS_OF_EDITS_NOTE}\nImage: as of 2026-01-30 (it has changed since).")
    png = BETHANY.replace(" ", "%20") + ".png"
    assert sent[1]["content"] == f"https://raw.githubusercontent.com/{qc.card_repo.repository}/{first}/{png}"
    assert [kw["content"] for kw in _ask("Tainted Bethany", "2026-10-22")][1:] == ["https://img/tainted_bethany.png"]
    assert "Image:" not in _ask("Tainted Bethany", "2026-10-22")[0]["content"]
    old = _ask("Tainted Bethany", "2026-02-01")[0]
    assert old["links"][1] == ("Image (2026-01-30)", f"https://raw.githubusercontent.com/{qc.card_repo.repository}/"
                                                     f"{first}/{png}")
    import commands.repo_history as rh
    import subprocess
    work = tmp_path / "cards"                                              # deleted later: never the deleting commit
    subprocess.run(["git", "-C", str(work), "rm", "-q", f"{BETHANY}.png"], check=True)
    subprocess.run(["git", "-C", str(work), "-c", "user.name=a", "-c", "user.email=a@x", "commit", "-q", "-m", "gone"],
                   check=True, env={**__import__("os").environ, "GIT_COMMITTER_DATE": f"@{int(_at(2026, 10, 22, 8))} +0000"})
    subprocess.run(["git", "-C", str(tmp_path / "history.git"), "fetch", "-q", str(work), "+main:main"], check=True)
    assert rh.commit_before(qc.card_repo.repository, f"{BETHANY}.png", _at(2026, 10, 23))[0] == second


# ========================
# /view_old_metadata finds the same cards
# ========================
def _old(query, version=0):
    import commands.psd_analyzer as pa
    utils.dispatch_queue.clear()
    pa.view_old_metadata(SimpleNamespace(), query, version)
    return [kw["content"] for _, kw in utils.dispatch_queue]


def test_view_old_metadata_reaches_archived_and_case_named_cards(db, monkeypatch):
    """ZOEY!!!!! (archived, an alias of Z!!!!! claims its name) shows its own history and says it is archived; omori
    (omori.psd, Omori.png) its own; a listed card as before."""
    monkeypatch.setitem(qc.card_repo.git_files, "zoey!!!!!.png", "Creatures/Other/5 Stars/Z!!!!!.png")
    monkeypatch.setitem(qc.card_repo.git_files, "z!!!!!.png", "Creatures/Other/5 Stars/Z!!!!!.png")
    monkeypatch.setitem(qc.card_repo.git_files, "omori.png", "Creatures/Omori/5 Stars/Omori.png")
    monkeypatch.setitem(cm.stats_db.stats, "omori", _card("omori", "Creatures/Omori/5 Stars/omori", 2.0, "Now."))
    monkeypatch.setitem(cm.stats_db.old_stats, "omori", [_card("omori", "Creatures/Omori/5 Stars/omori", 1.0, "Then.")])
    zoey = _old("ZOEY!!!!!")[0]
    assert zoey.startswith("Old metadata for **ZOEY!!!!!** (most recent old version of 2), archived on 2026-06-18 "
                           "(PSD deleted in 35adb38):") and "Zoey last." in zoey
    assert _old("omori")[0].startswith("Old metadata for **omori** (most recent old version of 1):")
    assert _old("Tainted Bethany", -1)[0].startswith("Old metadata for **Tainted Bethany** (oldest of 4):")
