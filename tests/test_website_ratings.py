"""
Card ratings (TTSCardMaker site/Card Ratings.json): data only since 2026-11-12 (owner: the TTS-session import deleted);
the file in dump_edits' layout (a card rename rewrites it so, card_bake) and the bot never reading it. Loads the site from
$AC_SITE_SRC (a TTSCardMaker checkout) or the clone.
"""
import json

import commands.card_sync as card_sync
from commands.card_edits import CardEdits, dump_edits
from conftest import SITE_SRC
from global_config import RATINGS_PATH

SITE = SITE_SRC


def test_ratings_file_is_whole():
    """The card repo's file: whole values only (the 2026-10-26 migration), in dump_edits' layout."""
    text = (SITE / RATINGS_PATH).read_text(encoding="utf-8")
    data = json.loads(text)
    assert dump_edits(data) == text
    bad = [(name, e) for name, entries in data["cards"].items() for e in entries
           if not (type(e.get("value")) is int and 1 <= e["value"] <= 10)]                # whole numbers only (owner, 2026-10-26)
    assert not bad, f"{len(bad)} entries not whole 1-10 (pull the card repo?): {bad[:3]}"


def test_bot_ignores_ratings():
    assert RATINGS_PATH not in card_sync.CARD_FILES and not card_sync.card_input(RATINGS_PATH)
    shaped = dump_edits({"cards": {"Alpha": [{"date": "2023-04-28", "scale": "power", "value": 5, "by": "group",
                                              "source": "TTS session"}]}})
    assert CardEdits.parse(shaped).cards == {}
    workflows = SITE / ".github" / "workflows"
    for name in ("card-sync", "card-guards"):
        assert "Card Ratings.json" not in (workflows / f"{name}.yml").read_text(encoding="utf-8"), name
