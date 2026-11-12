"""
Card ratings (TTSCardMaker site/Card Ratings.json): the TTS import (commands/tts_ratings.py: the board mapping, the
skips, an idempotent merge that keeps website entries), the file's layout (card_edits.CardRatings, dump_ratings) and the
bot never reading it. The data only: the website shows no ratings since 2026-11-11 (owner). Loads the site from
$AC_SITE_SRC (a TTSCardMaker checkout) or the clone; those tests skip without it.
"""
import json

import pytest

import commands.card_sync as card_sync
import commands.tts_ratings as tr
from commands.card_edits import CardEdits, CardRatings, dump_ratings
from conftest import SITE_SRC
from global_config import RATINGS_PATH

SITE = SITE_SRC
G, S = "group", "TTS session"


def _e(day, scale, value, by=G, source=S):
    return CardRatings.entry(day, scale, value, by, source)


def _row(save, board, x, category, card="Alpha", z=0.0, **kw):
    scale = tr.SCALE_OF[board]
    return {"save": save, "scale": scale, "board": board, "x": x, "z": z, "category": category, "card": card,
            "nick": card or kw.get("archived"), "png": f"Creatures/Other/2 Stars/{card or kw.get('archived')}.png",
            "match": "png", "match_note": "", "card_stars": 2, "card_author": "Maker", **kw}


# ========================
# The import
# ========================
def test_score_bands():
    even = (0, 10, 20, 30, 40)                    # labels centred in their bands: exactly 2i+1.5, halves up
    assert [tr.score(x, even)[1] for x in even] == [2, 4, 6, 8, 10]
    labels = tr.LABELS["centre"]
    assert [tr.score(x, labels) for x in labels] == [(0, 2), (1, 3), (2, 6), (3, 8), (4, 10)]   # 1.5 3.4 5.5 7.6 9.5
    mids = [(labels[i] + labels[i + 1]) / 2 for i in range(4)]
    assert [tr.score(m, labels)[1] for m in mids] == [3, 5, 7, 9]
    assert tr.score(-99, labels) == (0, 1) and tr.score(99, labels) == (4, 10)
    sweep = [tr.score(labels[0] - 6 + i * 0.1, labels)[1] for i in range(int((labels[4] - labels[0] + 12) * 10))]
    assert sweep == sorted(sweep)
    assert tr.score(-1.16, labels) == (2, 6)
    assert [tr.tier_of(v) for v in (1, 2.4, 2.5, 4.5, 6.5, 8.4, 8.5, 10)] == [0, 0, 1, 2, 3, 3, 4, 4]


def test_placements_skips_and_dates():
    raw = {"ratings": [
        _row("TS_Save_3", "like-bottom", 32.76, "Double Pain", card="Double Pain"),       # the joke column
        _row("TS_Save_3", "centre", -1.16, "Balens"),
        _row("TS_Save_3", "centre", -1.16, "Balens", card="Beta"),                       # same stack, same value
        _row("TS_Save_8", "centre", 9.71, "Above Average", z=-13.28),
        _row("TS_Save_10", "centre", 9.9, "Above Average", z=-13.0),                     # left where it was
        _row("TS_Save_8", "centre", 9.71, "Above Average", card="Beta", z=-13.28),
        _row("TS_Save_10", "centre", 4.57, "Balens", card="Beta", z=-13.16),             # moved
        _row("TS_Save_10", "like-top", 42.29, "Dislike", card=None, archived="Cho'Gath"),
    ], "forbidden": [{"save": "TS_Save_3", "card": "Kalecgos"}], "skipped": {}}
    entries, report = tr.placements(raw)
    got = [(e["card"], e["date"], e["scale"], e["value"]) for e in entries]
    assert got == [("Alpha", "2023-04-28", "power", 6), ("Beta", "2023-04-28", "power", 6),
                   ("Alpha", "2025-03-02", "power", 7), ("Beta", "2025-03-02", "power", 7),
                   ("Beta", "2025-04-28", "power", 6), ("Cho'Gath", "2025-04-28", "favorability", 3)]
    assert all(e["by"] == G and e["source"] == S for e in entries)
    assert [r["card"] for r in report["dropped"]] == ["Double Pain"] and len(report["carried"]) == 1
    assert report["forbidden"] == raw["forbidden"] and report["cards"] == 3
    with pytest.raises(ValueError, match="Balens"):
        tr.placements({"ratings": [_row("TS_Save_3", "centre", 9.71, "Balens")]})


def test_merge_is_idempotent_and_keeps_website_entries():
    text = dump_ratings({"about": CardRatings.ABOUT, "cards": {"alpha": [
        _e("2023-04-28", "power", 2, G, S),                     # stale: replaced
        _e("2026-10-24", "power", 6, "someone", "website"),
        _e("2024-01-01", "favorability", 4, G, S)]}})          # another date: kept
    entries = [{"card": "Alpha", **_e("2023-04-28", "power", 5.5)}, {"card": "Gamma", **_e("2025-04-28", "favorability", 7.1)}]
    once = tr.merge(text, entries)
    cards = json.loads(once)["cards"]
    assert list(cards) == ["alpha", "Gamma"]
    assert cards["alpha"] == [_e("2023-04-28", "power", 5.5), _e("2024-01-01", "favorability", 4),
                              _e("2026-10-24", "power", 6, "someone", "website")]
    assert tr.merge(once, entries) == once
    assert json.loads(tr.merge(None, entries))["about"] == CardRatings.ABOUT


def test_dump_ratings_layout():
    data = {"about": "x", "cards": {"Zeta": [_e("2023-04-28", "power", 7.0)],
                                    "Éclair": [_e("2025-04-28", "favorability", 8, "Ünï", "website")]}}
    assert dump_ratings(data) == (
        '{\n  "about": "x",\n  "cards": {\n'
        '    "Éclair": [{"date":"2025-04-28","scale":"favorability","value":8,"by":"Ünï","source":"website"}],\n'
        '    "Zeta": [{"date":"2023-04-28","scale":"power","value":7,"by":"group","source":"TTS session"}]\n  }\n}\n')
    parsed = CardRatings.parse(dump_ratings(data))
    assert parsed.cards["zeta"] == ("Zeta", data["cards"]["Zeta"])
    bad = [{**_e("2023-04-28", "power", 5), **change} for change in
           ({"value": 0}, {"value": 7.5}, {"value": 7.0}, {"value": 10.5}, {"value": True}, {"date": "2023-02-30"},
            {"scale": "taste"}, {"source": "Discord"}, {"by": ""})]
    assert CardRatings.parse(json.dumps({"cards": {"A": bad + [_e("2023-04-28", "power", 7)], "B": "no"}})).cards == \
        {"a": ("A", [_e("2023-04-28", "power", 7)])}                        # whole numbers only (owner, 2026-10-26)
    assert [CardRatings.number(v) for v in (5.5, 9.5, 4.5, 2.5, 7.1, 2.8, 6.49, 7.0)] == [6, 10, 5, 3, 7, 3, 6, 7]   # halves up


def test_ratings_file_is_whole():
    """The card repo's file: whole values only (the 2026-10-26 migration), in dump_ratings' layout."""
    path = SITE / RATINGS_PATH
    if not path.is_file():
        pytest.skip(f"no {path}")
    text = path.read_text(encoding="utf-8")
    data = json.loads(text)
    assert dump_ratings(data) == text
    bad = [(name, e) for name, entries in data["cards"].items() for e in entries if not CardRatings.valid(e)]
    assert not bad, f"{len(bad)} entries not whole 1-10 (pull the card repo?): {bad[:3]}"


def test_bot_ignores_ratings():
    assert RATINGS_PATH not in card_sync.CARD_FILES and not card_sync.card_input(RATINGS_PATH)
    shaped = dump_ratings({"cards": {"Alpha": [_e("2023-04-28", "power", 5)]}})
    assert CardEdits.parse(shaped).cards == {}
    workflows = SITE / ".github" / "workflows"
    if not workflows.is_dir():
        pytest.skip(f"no workflows at {workflows}")
    for name in ("card-sync", "card-guards"):
        assert "Card Ratings.json" not in (workflows / f"{name}.yml").read_text(encoding="utf-8"), name
