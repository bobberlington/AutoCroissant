"""
The website dates a card's versions as before the history cleanup: card_history.dedupe and the legacy same-data re-save
(CardBuild._record_history) drop copies but carry their start in the `version` record, which the card repo's
site/build/analytics.py reads (when). Loads analytics.py from $AC_SITE_SRC (a TTSCardMaker checkout) or the clone;
skips without it, or when that analytics.py predates `when` (TTSCardMaker work/version-dates).
"""
import importlib.util
from collections import defaultdict
from copy import deepcopy
from os import environ
from os.path import expanduser
from pathlib import Path

import pytest

import commands.card_history as ch
import commands.card_models as cm
from commands.card_data import CardBuild
from conftest import CLONE

ANALYTICS = Path(expanduser(environ.get("AC_SITE_SRC") or str(CLONE))) / "site" / "build" / "analytics.py"


@pytest.fixture(scope="module")
def an():
    if not ANALYTICS.is_file():
        pytest.skip(f"no card repo analytics at {ANALYTICS}")
    spec = importlib.util.spec_from_file_location("site_analytics", ANALYTICS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "when"):
        pytest.skip(f"{ANALYTICS} dates versions by timestamp only: land TTSCardMaker work/version-dates")
    return module


def _card(ability="Text.", hp=2, blob_sha=None, timestamp=1.0):
    return cm.CardInfo(name="A", card_type="creature", path="Creatures/A.psd", ability=ability, blob_sha=blob_sha,
                       timestamp=timestamp, stars=1, stats=cm.CardStats(hp=hp, defense=1, attack=1, speed=1))


COMMITS = {(at, "a.psd") for at in (100, 200, 300, 400)}   # the PSD's commits (site.yml's full history has them)


def _site(an, stats, old):
    """What the website shows of A: its change rows (exact or approx), stat trend and first point."""
    card = stats["A"]
    e = {"key": "A", "hp": card.stats.hp, "def": card.stats.defense, "atk": card.stats.attack, "spd": card.stats.speed,
         "stars": card.stars, "type": card.card_type, "ability": card.ability}
    chain = an.page_chain(e, stats, old)
    return an.card_changes(e, chain, COMMITS), an.trend_of(chain), an.point(chain[0])[0]


def test_dedupe_keeps_the_websites_dates(an):
    """[A 100, B 200, B again 300] + current B 400: the cleanup leaves [A] + B, B dated 200 as before."""
    stats = {"A": _card("New.", 3, "b4", 400.0)}
    old = defaultdict(list, {"A": [_card(blob_sha="b1", timestamp=100.0), _card("New.", 3, "b2", 200.0),
                                   _card("New.", 3, "b3", 300.0)]})
    before = _site(an, *deepcopy((stats, old)))
    assert [row["ts"] for row in before[0]] == [200] and before[2] == 100.0
    assert len(ch.dedupe(stats, old)) == 2 and len(old["A"]) == 1
    assert _site(an, stats, old) == before


def test_a_same_data_resave_keeps_the_websites_dates(an):
    """/update_stats: B re-saved with the same data adds no version; the website still dates the change at B."""
    first, changed = _card(blob_sha="b1", timestamp=100.0), _card("New.", 3, "b2", 200.0)
    kept = (defaultdict(list, {"A": [first, deepcopy(changed)]}), _card("New.", 3, "b3", 300.0))   # before 2026-10-22
    db = cm.StatsDatabase()
    db.stats, db.old_stats = {"A": _card("New.", 3, "b3", 300.0)}, defaultdict(list, {"A": [deepcopy(first)]})
    build = CardBuild(db, report=lambda *a: None)
    build.replaced = {"A": changed}
    build._record_history()
    assert len(db.old_stats["A"]) == 1 and ch.start(db.stats["A"]) == (200.0, None)
    assert _site(an, db.stats, db.old_stats) == _site(an, {"A": kept[1]}, kept[0])
    assert db.stats["A"].timestamp == 300.0                        # the card's own date (the site's "updated") stays


def test_a_carried_start_still_folds_a_re_read(an):
    """Bob-ulk: a version read again at the same commit by a newer parser gives way to the newer reading. The cleanup drops
    that reading's copy and the card holds its start; the site folds by that start (when), so the page doesn't change."""
    stats = {"A": _card("New.", 3, "b4", 400.0)}
    old = defaultdict(list, {"A": [_card(blob_sha="b1", timestamp=100.0), _card("New.", 3, "b1", 100.0)]})
    before = _site(an, *deepcopy((stats, old)))
    assert before[0] == [] and before[2] == 100.0
    assert ch.dedupe(stats, old) == [("A", 100.0)] and ch.start(stats["A"]) == (100.0, None)
    assert _site(an, stats, old) == before
