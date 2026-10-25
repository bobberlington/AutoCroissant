"""
The website build's game.txt (TTSCardMaker build_site.game_text, docs/Future/GAME_TXT.md): rulebook pages in book order,
tokens as words, cards grouped by kind, hidden cards left out, a variant's same text once. Loads build_site.py from $AC_SITE_SRC or the clone.
"""
import importlib.util
from os import environ
from os.path import expanduser
from pathlib import Path

import pytest

from conftest import CLONE

BUILD_SITE = Path(expanduser(environ.get("AC_SITE_SRC") or str(CLONE))) / "site" / "build" / "build_site.py"


@pytest.fixture(scope="module")
def bs():
    if not BUILD_SITE.is_file():
        pytest.skip(f"no card repo build_site at {BUILD_SITE}")
    spec = importlib.util.spec_from_file_location("site_build_site_gametxt", BUILD_SITE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "game_text"):
        pytest.skip(f"{BUILD_SITE} has no game.txt")
    return module


def test_game_txt(bs):
    icons = {"fire": {}, "offensiveminion": {}}
    rules = [{"name": "01 Start", "ability": "Burn with [fire].  \nDon’t panic"}, {"name": "02 Tags", "ability": "(1CD) [X]"}]
    cards = [
        {"id": "zed", "name": "Zed", "type": "creature", "stars": 3, "types": ["fire"], "hp": 5, "def": 2, "atk": 4, "spd": 1,
         "series": "B", "ability": "(1CD) Deal [fire:glow] damage, [fire:no]."},
        {"name": "Amy", "type": "creature", "stars": 2, "series": "B", "subseries": "Sub",
         "creatures": [{"hp": 3, "def": 1, "atk": 2, "spd": 3}, {"hp": 4, "def": 1, "atk": None, "spd": 2}]},
        {"name": "Sword", "type": "item", "subtype": "equip", "stars": 1, "series": "A", "ability": "Summon an [offensive_minion]."},
        {"name": "Zed Alt", "type": "creature", "dup": "zed", "series": "B", "ability": "(1CD) Deal [fire:glow] damage, [fire:no]."},
        {"name": "Secret", "type": "creature", "hidden": True, "ability": "gone"},
    ]
    banlist = {"notes": ["Constructed only."], "groups": [{"title": "Banned", "entries": [{"name": "Zed", "reason": "too strong"}]}]}
    text = bs.game_text(cards, rules, banlist, icons, {"built_at": "2026-10-25T00:00:00Z", "card_commit": "a" * 40})
    lines = text.splitlines()
    assert "4 cards, 2 rulebook pages. Card data built 2026-10-25T00:00:00Z from card commit aaaaaaaaaaaa." in lines[1]
    assert lines.index("## 01 Start") < lines.index("## 02 Tags") < lines.index("# Timing tags") < lines.index("# Creatures (3)")
    assert "Burn with fire." in lines and "Don't panic" in lines and "(1CD) [X]" in lines     # [X] isn't an icon: as printed
    assert "(1CD) Cooldown: 1 turn" in lines and "Zed - too strong" in lines
    assert lines.index("# Creatures (3)") < lines.index("# Items (1)")
    assert "Amy | Creature | 2★ | HP 3 DEF 1 ATK 2 SPD 3; HP 4 DEF 1 ATK ? SPD 2 | B / Sub" in lines
    assert "Zed | Creature | 3★ | fire | HP 5 DEF 2 ATK 4 SPD 1 | B" in lines
    assert lines.count("(1CD) Deal fire damage, not fire.") == 1                  # a variant's same text: once
    assert "Zed Alt | Creature | B | variant of Zed" in lines
    assert "Sword | Item (equip) | 1★ | A" in lines and "Summon an offensive minion." in lines
    assert "Secret" not in text and "’" not in text
