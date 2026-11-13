"""
The website's person pages (TTSCardMaker site/assets/profile.js in node), one smoke test (owner 2026-11-12): the star
mix within a kind. Loads the site from $AC_SITE_SRC (a TTSCardMaker checkout) or the clone; skips without profile.js.
"""
import json
from subprocess import run as run_process

import pytest

from conftest import SITE_SRC

PROFILE_JS = SITE_SRC / "site" / "assets" / "profile.js"


def node(tmp_path, body: str):
    """Run body after importing profile.js as pf; its stdout as JSON."""
    if not PROFILE_JS.is_file():
        pytest.skip(f"no profile.js at {PROFILE_JS}")
    script = tmp_path / "probe.mjs"
    script.write_text(f"const pf = await import({json.dumps(PROFILE_JS.as_uri())});\n"
                      "const out = x => process.stdout.write(JSON.stringify(x));\n"
                      "const text = html => html.replace(/<[^>]+>/g, ' ').replace(/\\s+/g, ' ').trim();\n"
                      f"{body}\n", encoding="utf-8")
    return json.loads(run_process(["node", str(script)], capture_output=True, text=True, check=True).stdout)


def test_star_mix_within_kind(tmp_path):
    """mixOf with a kind: star shares within that kind on both sides; kinds rows unchanged; starKinds in KINDS order."""
    out = node(tmp_path, """
const c = (id, author, type, stars, x = {}) => ({ id, author, type, stars, ...x });
const all = [c('a1', 'A', 'creature', 2), c('a2', 'A', 'creature', 3), c('a3', 'A', 'minion', 1), c('a4', 'A', 'item'),
  c('b1', 'B', 'creature', 2), c('b2', 'B', 'minion', 1), c('b3', 'B', 'minion', 2), c('h', 'B', 'minion', 5, { hidden: true })];
const own = pf.cardsOf('A', all), kinds = [['creature'], ['item'], ['field'], ['minion']];
out({ all: pf.mixOf(own, all).stars, minion: pf.mixOf(own, all, undefined, 'minion'), order: pf.starKinds(own, undefined, kinds) });""")
    assert out["all"] == [[1, 1 / 3, 2 / 6], [2, 1 / 3, 3 / 6], [3, 1 / 3, 1 / 6]]
    assert out["minion"]["stars"] == [[1, 1, 2 / 3]]
    assert out["minion"]["kinds"][0][0] == "creature"
    assert out["order"] == ["creature", "minion"]
