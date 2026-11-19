"""
The website's person pages (TTSCardMaker site/assets/profile.js in node), one smoke test (owner 2026-11-12): the star
mix within a kind; and who a website edit credits (build_site.py contributions). Loads the site from $AC_SITE_SRC (a
TTSCardMaker checkout) or the clone.
"""
import importlib.util
import json
from subprocess import run as run_process

from conftest import SITE_SRC

PROFILE_JS = SITE_SRC / "site" / "assets" / "profile.js"
BUILD_SITE = SITE_SRC / "site" / "build" / "build_site.py"


def node(tmp_path, body: str):
    """Run body after importing profile.js as pf; its stdout as JSON."""
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


def test_website_edit_credits_the_login(monkeypatch):
    """A publish GitHub wrote under the editor's own name and email (stamp _by 'login'), its bake (Edit-by: login) and a
    later hand commit under that name: one row, the login's, with the git name among its logins."""
    spec = importlib.util.spec_from_file_location("site_build_people", BUILD_SITE)
    bs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bs)
    blobs = {"e0": '{"cards": {}}', "e1": '{"cards": {"Alpha": {"ability": "New.", "_by": "login", "_at": "2026-11-19T09:59:00Z"}}}'}
    monkeypatch.setattr(bs, "read_blobs", lambda shas: blobs)
    edits, png = "site/Card Edits.json", "Creatures/Other/2 Stars/Alpha.png"
    commits = [("c" * 40, 300, "Name", "x@gmail.com", "Alpha art", [("M", "p1", "p2", png)], []),
               ("b" * 40, 200, "AutoCroissant", "bot@example.com", "Card sync: 1 card baked", [("M", "p0", "p1", png), ("M", "e1", "e0", edits)], ["login"]),
               ("a" * 40, 100, "Name", "x@gmail.com", "Alpha reword", [("M", "e0", "e1", edits)], [])]
    row_of, stamped = bs.login_rows(commits, {}, {}), {}
    bs.edit_batches(commits, {}, {}, lambda n: None, str, 0, row_of, stamped=stamped)
    people = bs.contributions([], {}, [], {}, commits, row_of=row_of, stamped=stamped)["people"]
    assert [(r["key"], r["edits"], r["commits"], r["logins"], r.get("github")) for r in people] == [("login", 1, 2, ["login", "Name"], "login")]
