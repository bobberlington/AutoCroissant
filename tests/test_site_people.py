"""
The website's person pages (TTSCardMaker site/build/build_site.py person_extras, site/assets/profile.js in node): image
revisions after the first upload, a bake as its editors' revision, untouched and most revised cards, the strength profile,
buffs and nerfs and who made them, ratings of their cards and as a rater (hidden without per-person ratings), edits by
field, archived cards; the style, rank, career, series and most-referenced helpers and the panels. Loads the site from
$AC_SITE_SRC (a TTSCardMaker checkout) or the clone; skips without person_extras / styleOf, or node.
"""
import importlib.util
import json
import shutil
from os import environ
from os.path import expanduser
from pathlib import Path
from subprocess import run as run_process

import pytest

from conftest import CLONE

SITE = Path(expanduser(environ.get("AC_SITE_SRC") or str(CLONE))) / "site"
BUILD_SITE = SITE / "build" / "build_site.py"
PROFILE_JS = SITE / "assets" / "profile.js"
T = 1_780_000_000


@pytest.fixture(scope="module")
def bs():
    if not BUILD_SITE.is_file():
        pytest.skip(f"no card repo build_site at {BUILD_SITE}")
    spec = importlib.util.spec_from_file_location("site_build_site_people", BUILD_SITE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "person_extras"):
        pytest.skip(f"{BUILD_SITE} has no person stats")
    return module


def node(tmp_path, body: str):
    """Run body after importing profile.js as pf; its stdout as JSON."""
    if not (PROFILE_JS.is_file() and shutil.which("node") and "styleOf" in PROFILE_JS.read_text(encoding="utf-8")):
        pytest.skip(f"no profile.js with styleOf at {PROFILE_JS}, or no node")
    script = tmp_path / "probe.mjs"
    script.write_text(f"const pf = await import({json.dumps(PROFILE_JS.as_uri())});\n"
                      "const out = x => process.stdout.write(JSON.stringify(x));\n"
                      "const text = html => html.replace(/<[^>]+>/g, ' ').replace(/\\s+/g, ' ').trim();\n"
                      f"{body}\n", encoding="utf-8")
    return json.loads(run_process(["node", str(script)], capture_output=True, text=True, check=True).stdout)


def card(cid, author, **kw):
    return {"id": cid, "key": cid, "name": cid.title(), "author": author, **kw}


def commit(sha, ts, name, paths=(), editors=()):
    """A parse_history row: (sha, committer time, author, email, subject, entries, Edit-by values)."""
    return (sha, ts, name, f"{name.lower()}@example.com", "s", [("M", "1" * 40, "2" * 40, p) for p in paths], list(editors))


def rows_of(bs, cards, commits=(), people=None, **kw):
    out = bs.contributions(cards, people or {}, [], {}, list(commits), **kw)
    return {r["key"]: r for r in out["people"]}, out


def test_power_profile(bs):
    """power: own cards with z_total; z = the mean (2 dp), top/bottom = flag counts, dots by z with their flag; site.z: all."""
    balance = {"cards": {"a1": {"z_total": 1.5, "flags": ["top"]}, "a2": {"z_total": -0.25, "flags": []},
                         "a3": {"z_total": -2.0, "flags": ["bottom"]}, "b1": {"z_total": 0.5, "flags": []}}}
    rows, out = rows_of(bs, [card("a1", "A"), card("a2", "A"), card("a3", "A"), card("b1", "B")], balance=balance)
    assert rows["A"]["power"] == {"n": 3, "z": -0.25, "top": 1, "bottom": 1, "dots": [["a3", -2.0, "bottom"], ["a2", -0.25], ["a1", 1.5, "top"]]}
    assert out["site"] == {"z": -0.06}


def rate(value, by="group", day="2025-04-28", scale="favorability", source="TTS session"):
    return {"date": day, "scale": scale, "value": value, "by": by, "source": source}


