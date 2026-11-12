"""
Site storage blocked (TTSCardMaker site/assets/deck.js, in node): reading localStorage throws a SecurityError, as in a
browser that blocks site data. The hands still load, saving never throws, and the picked hand is kept for the page.
Skips without deck.js (2026-11-11).
"""
import json
from subprocess import run as run_process

import pytest

from conftest import SITE_SRC

DECK_JS = SITE_SRC / "site" / "assets" / "deck.js"
pytestmark = pytest.mark.skipif(not DECK_JS.is_file(), reason=f"no {DECK_JS}")


def test_hands_work_with_storage_blocked(tmp_path):
    script = tmp_path / "probe.mjs"
    script.write_text(f"""Object.defineProperty(globalThis, "localStorage", {{ get() {{ throw new DOMException("insecure", "SecurityError"); }} }});
const D = await import({json.dumps(DECK_JS.as_uri())});
const decks = D.loadDecks(), first = D.activeDeck(decks).id;
const two = {{ id: "h2", name: "Hand 2", cards: [], field: null }};
decks.push(two); D.saveDecks(decks); D.setActive(two);
process.stdout.write(JSON.stringify({{ n: decks.length, first: first === decks[0].id, picked: D.activeDeck(decks).id }}));
""", encoding="utf-8")
    done = run_process(["node", str(script)], capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout) == {"n": 2, "first": True, "picked": "h2"}
