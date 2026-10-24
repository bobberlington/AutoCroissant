"""
The website's card image previews in edit mode (TTSCardMaker site/assets/preview.js and mockup.js, in node; the build's
data/fonts.json from site/build/build_site.py): the rendered preview's request, store, polling states, staleness and
messages against card-preview.yml's contract (commands/card_preview.py's result.json), and the quick mock-up's layout,
erase, wrapping and stat cells on drawn pixel arrays; the build reads font names as font_check does (fontTools) and maps
the PSD fonts to MDW/Fonts files. Loads the site from $AC_SITE_SRC (a TTSCardMaker checkout) or the clone; skips without
it, node, or a site before preview.js.
"""
import importlib.util
import json
import shutil
from os import environ
from os.path import expanduser
from pathlib import Path
from subprocess import run as run_process

import pytest

from commands.font_check import face_names
from conftest import CLONE

SITE = Path(expanduser(environ.get("AC_SITE_SRC") or str(CLONE)))
ASSETS = SITE / "site" / "assets"
PREVIEW_JS, MOCKUP_JS = ASSETS / "preview.js", ASSETS / "mockup.js"
BUILD_SITE = SITE / "site" / "build" / "build_site.py"
FONTS = SITE / "MDW" / "Fonts"
needs_js = pytest.mark.skipif(not (PREVIEW_JS.is_file() and shutil.which("node")), reason=f"no preview.js at {ASSETS}, or no node")
HOUR = 3600_000


def node(tmp_path, body: str):
    """Run body with preview.js as p and mockup.js as m; its stdout as JSON."""
    script = tmp_path / "probe.mjs"
    script.write_text(f"const p = await import({json.dumps(PREVIEW_JS.as_uri())});\nconst m = await import({json.dumps(MOCKUP_JS.as_uri())});\n"
                      f"const out = x => process.stdout.write(JSON.stringify(x));\n{body}\n", encoding="utf-8")
    done = run_process(["node", str(script)], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


# ---- the rendered preview (preview.js)
@needs_js
def test_the_edit_is_the_entry_as_published(tmp_path):
    """The edit sent is every PSD-kind field of the card's Card Edits entry after the batch (equal to the PSD or not), in
    the edits file's shapes; database fields stay out; creatures only as a list of several. What the image changes by
    compares types in any order and text trimmed, and never counts a stat cleared to nothing."""
    out = node(tmp_path, """
const card = { name: " Mini Doomer EX ", hp: 8, def: 2, types: ["light", "wind"], ability: "Text\\r\\n", note: "x", series: "Kirby", creatures: null };
out([p.entryEdit(["name", "hp", "def", "types", "ability", "note", "series", "creatures"], f => card[f]),
  p.entryEdit(["creatures"], () => [{ hp: 1 }, { hp: 2 }]), p.entryEdit(["creatures"], () => [{ hp: 1 }]),
  p.changedOf({ name: "Mini Doomer", types: ["wind", "light"], hp: null, ability: " A \\n", stars: 2 }, { name: "Mini Doomer", types: ["light", "wind"], hp: 5, ability: "A", stars: 3 })]);""")
    assert out[0] == {"name": "Mini Doomer EX", "types": ["light", "wind"], "hp": 8, "def": 2, "ability": "Text"}
    assert out[1] == {"creatures": [{"hp": 1}, {"hp": 2}]} and out[2] == {}
    assert out[3] == {"stars": 2}


@needs_js
def test_the_same_edit_hashes_the_same(tmp_path):
    """Keys in any order and types in any order are the same edit (the contract compares sorted JSON); a value is not."""
    out = node(tmp_path, """
const a = { hp: 8, types: ["wind", "light"], name: "X" }, b = { name: "X", types: ["light", "wind"], hp: 8 };
out([p.canonical(a), p.hashOf(a), p.hashOf(b), p.hashOf({ ...a, hp: 9 })]);""")
    assert out[0] == '{"hp":8,"name":"X","types":["light","wind"]}'
    assert out[1] == out[2] != out[3] and len(out[1]) == 14


@needs_js
def test_request_ids_fit_the_action(tmp_path):
    """card-preview.yml takes 8-64 of A-Z a-z 0-9 -: any login makes one; the store keeps nothing else."""
    out = node(tmp_path, """
out([["MichaelJSr", "a_b.c", "", "x".repeat(80)].map(l => p.newRequest(l, 1792700000000, 0.5)),
  ["short", "a/b-cdefghij", "x".repeat(65), "ok-request-1"].map(p.validRequest)]);""")
    for r in out[0]:
        assert 8 <= len(r) <= 64 and all(ch.isalnum() or ch == "-" for ch in r)
    assert out[0][0].startswith("michaeljsr-") and out[0][1].startswith("a-b-c-") and out[0][2].startswith("editor-")
    assert out[1] == [False, False, False, True]


@needs_js
def test_result_json_is_read_defensively(tmp_path):
    """card_preview's result.json: states, fields, files and the run link kept; anything unreadable is a transient error;
    a file name with a path, a run link off GitHub and over-long notes are dropped or cut."""
    result = {"v": 1, "state": "baked", "transient": False, "reason": "hp 5 -> 8", "fields": {"hp": "baked", "stars": "refused", "evil": "x"},
              "changes": [{"field": "hp", "old": 5, "new": 8}], "notes": ["n"] * 30 + ["x" * 900], "base": {"psd_blob": "a" * 40, "png_blob": "b" * 40, "n": 3},
              "image": "baked", "files": {"webp": "preview.webp", "png": "../x.png"}, "run": "https://evil.example/run"}
    out = node(tmp_path, f"""out([p.readResult({json.dumps(json.dumps(result))}), p.readResult("not json"), p.readResult({{ state: "done" }}),
  p.readResult({{ state: "error", transient: true, reason: "error: timeout", run: "https://github.com/o/r/actions/runs/1" }})]);""")
    ok = out[0]
    assert ok["state"] == "baked" and ok["fields"] == {"hp": "baked", "stars": "refused"} and ok["changes"] == [{"field": "hp", "old": 5, "new": 8}]
    assert ok["files"] == {"webp": "preview.webp"} and ok["run"] == "" and ok["image"] == "baked"
    assert ok["base"] == {"psd_blob": "a" * 40, "png_blob": "b" * 40} and len(ok["notes"]) == 20
    assert out[1]["state"] == out[2]["state"] == "error" and out[1]["transient"] and "JSON" in out[1]["reason"]
    assert out[3]["transient"] and out[3]["run"] == "https://github.com/o/r/actions/runs/1"


@needs_js
def test_the_store_keeps_the_last_three_per_card(tmp_path):
    """Newest first, KEEP (3) per card, a request replaced in place; a malformed stored entry is dropped; one still starting
    a minute on (the page closed) failed; a running one stays running however old (only a check that finds nothing
    times it out: missed)."""
    out = node(tmp_path, """
let s = {};
for (let i = 0; i < 5; i++) s = p.remember(s, { card: "a", request: `req-0000${i}`, at: i, hash: "h", state: "running" });
s = p.remember(s, { card: "b", request: "req-b-0001", at: 1, hash: "h", state: "running", wait: 1000 });
s = p.remember(s, { card: "d", request: "req-d-0001", at: 1, hash: "h", state: "starting" });
s = p.remember(s, { card: "a", request: "req-00004", at: 4, hash: "h", state: "baked" });
const read = p.readStore(JSON.stringify({ ...s, c: [{ request: "bad/one", at: 1, hash: "h" }, { request: "req-c-0001", at: "x", hash: "h" }] }));
const t = p.expire(s, 25 * 60000);
out([s.a.map(e => e.request), s.a[0].state, Object.keys(read).sort(), p.waiting(t).map(e => e.request), t.b[0].state, p.patch(s, "req-00003", { seen: true }).a[1].seen, p.KEEP, p.MAX_WAIT_MS,
  [t.d[0].state, t.d[0].seen, p.expire(s, 30000).d[0].state]]);""")
    assert out[0] == ["req-00004", "req-00003", "req-00002"] and out[1] == "baked"
    assert out[2] == ["a", "b", "d"] and out[3] == ["req-00003", "req-00002", "req-b-0001"] and out[4] == "running" and out[5] is True
    assert out[6:8] == [3, 16 * 60000]                 # the contract's 16 minutes
    assert out[8] == ["error", True, "starting"]       # a page closed mid-dispatch: not a spinner for ever, after a minute


@needs_js
def test_a_result_that_came_while_the_site_was_closed_is_found(tmp_path):
    """The owner's flow: press Preview image, close the laptop, come back 25 minutes later. The entry is still running, so
    the first check looks for it (a timed-out one too, on the first round after a load or a sign-in); only a check that
    finds nothing times it out (past its wait) or, a day on, gives it up. Checks every 10 s for 3 minutes, then 30 s."""
    out = node(tmp_path, """
const now = 25 * 60000, s = { a: [{ card: "a", request: "req-a-0001", at: 0, hash: "h", state: "running" }],
  b: [{ card: "b", request: "req-b-0001", at: 0, hash: "h", state: "timeout" }], c: [{ card: "c", request: "req-c-0001", at: 0, hash: "h", state: "baked" }] };
const kept = p.expire(s, now);
out([kept.a[0].state, p.due(kept, now).map(e => e.request), p.due(kept, now, true).map(e => e.request),
  p.missed(kept.a[0], now), p.missed(kept.a[0], 60000), p.missed({ ...kept.a[0], wait: 40 * 60000 }, now), p.missed(kept.b[0], now), p.missed(kept.b[0], 2 * 86400000),
  p.pollDelay(kept, 60000), p.pollDelay(kept, now), p.pollDelay({ c: s.c }, now), p.due(kept, 7 * 86400000, true).map(e => e.request)]);""")
    assert out[0] == "running" and out[1] == ["req-a-0001"] and out[2] == ["req-a-0001", "req-b-0001"]
    assert out[3] == {"state": "timeout"} and out[4] is None and out[5] is None
    assert out[6] is None and out[7] == {"state": "expired", "seen": True}
    assert out[8:11] == [10000, 30000, None]
    assert out[11] == ["req-a-0001"]                   # a week on: a running one is looked for once more (then missed gives it up), a timed-out one not


@needs_js
def test_old_previews_expire(tmp_path):
    """The previews branch keeps a preview about a week (rewritten past 200 or 7 days): a result older than SHOW_MS (6
    days) is "expired" (no image: Preview again, no badge), an entry older than 30 days goes; the panel says so and
    names the age in days."""
    out = node(tmp_path, """
const day = 86400000, e = (state, at, req) => ({ card: "x", request: req, at, hash: "h", state, edit: { hp: 8 }, result: { fields: { hp: "baked" }, notes: [], base: {}, files: { webp: "preview.webp" } } });
const s = { x: [e("baked", 39 * day, "req-x-0003"), e("baked", 31 * day, "req-x-0002")], y: [e("baked", 0, "req-y-0001")], z: [e("timeout", 32 * day, "req-z-0001")] };
const t = p.expire(s, 40 * day);
const v = { card: { name: "The Freezer", ts: 1, img: "ab" }, n: 1, hash: "h", now: 40 * day, scopes: "repo", images: ["https://r/p.webp"], current: "thumbs/ab-720.webp", diff: false };
out([t.x.map(e => [e.request, e.state, !!e.seen]), "y" in t, p.hasNews(t, "x"), p.hasNews(t, "z"), t.z[0].state, p.renderedHtml(t.x[1], v), p.ago(8 * day), p.ago(3 * 3600000)]);""")
    assert out[0] == [["req-x-0003", "baked", False], ["req-x-0002", "expired", True]] and out[1] is False   # 40 days old: gone
    assert out[2] is True and out[3] is False and out[4] == "expired"
    out = out[:3] + out[5:]
    assert "Preview expired" in out[3] and "kept about a week" in out[3] and "data-pv-img" not in out[3] and 'data-pv="start"' in out[3] and "9 days ago" in out[3]
    assert out[4] == "8 days ago" and out[5] == "3 h ago"


@needs_js
def test_one_tab_checks_and_a_round_is_one_toast(tmp_path):
    """Several tabs: the one holding the lock checks (the others get results through the storage event); a lapsed lock or
    a malformed one is free. A round's results make one toast; a result already there (or expired) none."""
    out = node(tmp_path, """
const lock = p.takeLock(null, "A", 1000);
out([lock, p.takeLock(JSON.stringify(lock), "B", 2000), p.takeLock(JSON.stringify(lock), "A", 2000).tab, p.takeLock(JSON.stringify(lock), "B", lock.until + 1).tab, p.takeLock("{bad", "B", 0).tab,
  p.finished({ a: [{ request: "r1", state: "running", name: "Abbi" }], b: [{ request: "r2", state: "running", name: "Acro" }], c: [{ request: "r3", state: "baked" }] },
             { a: [{ request: "r1", state: "baked", name: "Abbi" }], b: [{ request: "r2", state: "refused", name: "Acro" }], c: [{ request: "r3", state: "baked" }] }).map(e => e.request),
  p.finishedToast([{ state: "baked", name: "Abbi" }]), p.finishedToast([{ state: "refused", name: "Poison" }]),
  p.finishedToast(["Abbi", "Acro", "Mini Doomer", "Poison"].map(name => ({ state: "baked", name }))), p.finishedToast([{ state: "baked", name: "A" }, { state: "error", name: "B" }]), p.finishedToast([]),
  p.editBytes({ ability: "é" }), JSON.stringify({ ability: "é" }).length]);""")
    assert out[0]["tab"] == "A" and out[0]["until"] > 1000 and out[1] is None and out[2] == "A" and out[3] == "B" and out[4] == "B"
    assert out[5] == ["r1", "r2"]
    assert out[6:11] == ["Preview ready: Abbi", "Preview finished: Poison", "Previews ready: Abbi, Acro, Mini Doomer and Poison", "Previews finished: A and B", ""]
    assert out[11] == out[12] + 1                      # UTF-8 bytes, as card_preview counts them


@needs_js
def test_a_preview_goes_out_of_date(tmp_path):
    """Out of date when the fields the site would send changed, or the card's image moved: the result's PNG blob against
    cards.json img (its first 16 hex digits); before a result, the card's ts and img at the dispatch."""
    out = node(tmp_path, """
const card = { ts: 10, img: "abcdef0123456789" };
const e = { hash: "h1", base: { ts: 10, img: "abcdef0123456789" } };
out([p.staleWhy(e, card, "h1"), p.staleWhy(e, card, "h2"), p.staleWhy(e, { ...card, ts: 11 }, "h1"),
  p.staleWhy({ ...e, result: { base: { png_blob: "abcdef0123456789" + "f".repeat(24) } } }, { ...card, ts: 11 }, "h1"),
  p.staleWhy({ ...e, result: { base: { png_blob: "0".repeat(40) } } }, card, "h2")]);""")
    assert out == [[], ["edit"], ["psd"], [], ["edit", "psd"]]


@needs_js
def test_the_badge_is_for_an_unseen_result(tmp_path):
    """A tile is badged while its card's newest preview has a result nobody has seen; not while it runs, not once seen."""
    out = node(tmp_path, """
const s = { a: [{ state: "baked" }], b: [{ state: "running" }, { state: "baked" }], c: [{ state: "refused", seen: true }], d: [{ state: "error" }] };
out([["a", "b", "c", "d", "e"].map(id => p.hasNews(s, id)), p.badgeTip("baked"), p.badgeTip("error")]);""")
    assert out == [[True, False, False, True, False], "Its preview image is ready: open the card", "Its preview failed: open the card"]


@needs_js
def test_a_run_that_cant_start_says_why(tmp_path):
    """The Action missing on main; a public_repo token (repo scope needed, or Actions: Read and write); an expired token;
    GitHub's refusal; no network."""
    out = node(tmp_path, """
out([p.startProblem(404, "", { installed: false }), p.startProblem(403, "", { scopes: "public_repo, gist" }), p.startProblem(404, "", { scopes: "repo, workflow" }),
  p.startProblem(403, "", { scopes: null }), p.startProblem(401), p.startProblem(422, "Unexpected inputs provided"), p.startProblem(0, "Failed to fetch"),
  [p.scopeOk("public_repo"), p.scopeOk("repo"), p.scopeOk(null)]]);""")
    assert "isn't set up" in out[0] and "card-preview.yml" in out[0]
    assert "only the public_repo scope" in out[1] and "repo scope" in out[1] and "Actions: Read and write" in out[1]
    assert "can't start Actions" in out[2] and "can't start Actions" in out[3]
    assert "sign in again" in out[4] and "Unexpected inputs provided" in out[5] and "Couldn't reach GitHub" in out[6]
    assert out[7] == [False, True, None]


@needs_js
def test_the_panel_for_each_state(tmp_path):
    """Baked: current and preview side by side with Diff and Full size, each field's outcome, notes escaped; refused: the
    reason (with the bake's render when it made one); an input error is the website's bug; a transient one says try
    again; running: how long; no edit: the button disabled. Baked with no image file: said so, not an error; an error
    for an edit since changed: out of date, Preview again instead of Try again."""
    out = node(tmp_path, """
const v = { card: { name: "Mini <Doomer>", ts: 1, img: "ab" }, n: 1, hash: "h", now: 3 * 60000, scopes: "repo", images: ["https://r/p.webp", "https://r/p.png"], current: "thumbs/ab-720.webp", diff: false };
const e = state => ({ request: "req-00001", at: 0, hash: "h", base: { ts: 1, img: "ab" }, edit: { hp: 8 }, state });
const res = (state, extra = {}) => ({ ...e(state), result: { fields: { hp: state }, notes: ["<script>x</script>"], base: {}, files: {}, ...extra } });
out([p.renderedHtml(res("baked"), v), p.renderedHtml(res("refused", { reason: "stat>10: hp 25", image: "refused" }), v),
  p.renderedHtml(res("refused", { reason: "overflow" }), { ...v, images: [] }), p.renderedHtml(res("error", { reason: "input: bad edit" }), v),
  p.renderedHtml(res("error", { reason: "error: timeout", transient: true }), v), p.renderedHtml(e("running"), v), p.renderedHtml(null, { ...v, n: 0 }),
  p.renderedHtml(res("baked"), { ...v, hash: "other" }), p.renderedHtml(res("baked"), { ...v, images: [] }),
  p.renderedHtml(res("error", { reason: "error: timeout", transient: true }), { ...v, hash: "other" })]);""")
    baked, refused_img, refused, bug, transient, running, none, stale, no_file, err_stale = out
    assert 'data-pv-img="current"' in baked and 'src="https://r/p.webp"' in baked and 'href="https://r/p.png"' in baked and 'data-pv="diff"' in baked
    assert "HP baked" in baked and "&lt;script&gt;" in baked and "<script>" not in baked and "Mini &lt;Doomer&gt;" in baked and "3 min ago" in baked
    assert "stat&gt;10: hp 25" in refused_img and "before refusing" in refused_img and "data-pv-img" not in refused
    assert "website bug" in bug and "try again" in transient.lower()
    assert "pv-spin" in running and "1 to 3 minutes" in running and "16 min" in running and "disabled" in none
    assert "Out of date: the edit changed since" in stale
    assert "no image came with the result" in no_file and 'class="err"' not in no_file and "HP baked" in no_file   # not an error
    assert "Out of date" in err_stale and "Try again" not in err_stale and "Preview again" in err_stale   # Try again would send the newer edit


@needs_js
def test_the_panel_says_previews_are_public_and_takes_a_webp_alone(tmp_path):
    """Before the first press: a previewed edit is public (the run's inputs and the previews branch; the server's
    contract since TTSCardMaker a1274af7). A result listing only the WebP (no full-size PNG published): no Full size."""
    out = node(tmp_path, """
const v = { card: { name: "Mini Doomer", ts: 1, img: "ab" }, n: 1, hash: "h", now: 0, scopes: "repo", images: ["https://r/p.webp"], current: "thumbs/ab-720.webp", diff: false };
const res = { request: "req-00001", at: 0, hash: "h", base: { ts: 1, img: "ab" }, edit: { hp: 8 }, state: "baked", result: { fields: { hp: "baked" }, notes: [], base: {}, files: { webp: "preview.webp" } } };
out([p.renderedHtml(null, v), p.renderedHtml(null, { ...v, n: 0 }), p.renderedHtml(res, v)]);""")
    first, nothing, webp = out
    assert "Anyone can see a previewed edit" in first and "Anyone" not in nothing
    assert 'href="https://r/p.webp"' in webp and "Full size</a>" not in webp


@needs_js
def test_the_diff_marks_changed_pixels(tmp_path):
    """Changed pixels red, the rest the current image dimmed; the share of changed pixels."""
    out = node(tmp_path, """
const a = new Uint8ClampedArray([100, 100, 100, 255, 10, 10, 10, 255]), b = new Uint8ClampedArray([100, 104, 100, 255, 200, 10, 10, 255]);
const d = p.diffPixels(a, b); out([[...d.data], d.changed]);""")
    assert out == [[30, 30, 30, 255, 255, 40, 40, 255], 0.5]


# ---- the quick mock-up (mockup.js)
PAINT = """
// a W x H grey image; text(lines): glyph-like stripes (light fill, dark outline) on rows [y0, y1) from x0 to x1
const W = 240, H = 240, img = new Uint8ClampedArray(W * H * 4);
const put = (x, y, v) => { const i = (y * W + x) * 4; img[i] = img[i + 1] = img[i + 2] = v; img[i + 3] = 255; };
for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) put(x, y, 60 + Math.round(y / 8));
const text = lines => { for (const [x0, x1, y0, y1] of lines) for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) { const k = (x - x0) % 6; put(x, y, k < 2 ? 240 : k < 3 ? 0 : 60 + Math.round(y / 8)); } };
"""


@needs_js
def test_the_text_on_the_image_is_found(tmp_path):
    """Three centred lines: one paragraph of three lines, its box, a size from their pitch, centred; left-aligned lines
    read as left; a short weak line between two strong ones stays in the paragraph; an empty region has none."""
    out = node(tmp_path, PAINT + """
const clean0 = new Uint8ClampedArray(img);
text([[40, 200, 150, 162], [52, 188, 170, 182], [46, 194, 190, 202]]);
const L = m.luma(img), a = m.findText(L, W, H, [0, 140, W, H]);
text([[10, 150, 20, 32], [10, 120, 40, 52], [10, 170, 60, 72]]);
const b = m.findText(m.luma(img), W, H, [0, 10, W, 90]);
const c = m.findText(L, W, H, [0, 100, W, 140]);
img.set(clean0);
text([[40, 200, 150, 162], [112, 124, 170, 182], [46, 194, 190, 202]]);       // "it." between two long lines
const d = m.findText(m.luma(img), W, H, [0, 140, W, H]);
out([a.lines.length, a.box, Math.round(a.size), a.align, b.align, c, d.box[1], d.box[3]]);""")
    assert out[0] == 3 and out[3] == "center" and out[4] == "left" and out[5] is None
    x0, y0, x1, y1 = out[1]
    assert 38 <= x0 <= 42 and 148 <= y0 <= 151 and 196 <= x1 <= 202 and 200 <= y1 <= 204
    assert 14 <= out[2] <= 18                      # pitch 20 px / 1.22
    assert 148 <= out[6] <= 151 and 200 <= out[7] <= 204    # the weak line kept the last one in the paragraph


@needs_js
def test_erase_puts_the_background_back(tmp_path):
    """Inside the box the text's pixels take the background around them (a gradient here), outside nothing changes; the
    colours read on the way: a light fill with a dark outline on a dark background."""
    out = node(tmp_path, PAINT + """
const clean = new Uint8ClampedArray(img);
text([[40, 200, 150, 162]]);
const before = new Uint8ClampedArray(img);
const col = m.erase(img, W, H, [30, 140, 210, 172], { reach: 2, cell: 10 });
let worst = 0, outside = 0;
for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) { const i = (y * W + x) * 4, inBox = x >= 30 && x < 210 && y >= 140 && y < 172;
  if (inBox) worst = Math.max(worst, Math.abs(img[i] - clean[i])); else outside = Math.max(outside, Math.abs(img[i] - before[i])); }
out([worst, outside, col.fill, col.stroke]);""")
    worst, outside, fill, stroke = out
    assert worst <= 4 and outside == 0
    assert min(fill) >= 200 and max(stroke) <= 20


@needs_js
def test_tokens_wrap_and_fit(tmp_path):
    """[fire] is an icon (":no" crossed out), an unknown bracket stays text, no break inside "[ice].", a blank line is an
    empty line; a long text shrinks until it fits, a name keeps its line count."""
    out = node(tmp_path, """
const icons = { fire: { file: "icons/fire.webp" }, ice: { file: "icons/ice.webp" } };
const paras = m.tokenize("Inflict [fire] and [nope].\\n\\nTurns into [ice]. [fire:no]x", icons);
const measure = (s, z) => s.length * z * 0.5;
const lines = m.layout(paras, 60, 10, measure).map(l => l.pieces.map(q => q.icon ? `<${q.name}${q.no ? ":no" : ""}>` : q.text).join(""));
const fitted = m.fit(m.tokenize("a ".repeat(20), icons), 100, 40, 20, measure), name = m.fit(m.tokenize("A very long card name", icons), 100, 100, 20, measure, { lines: 1 });
out([lines, fitted.lines.length * fitted.size * m.LINE <= 40.5, fitted.size < 20, name.lines.length]);""")
    assert out[0] == ["Inflict <fire>", "and [nope].", "", "Turns into", "<ice>. <fire:no>x"]   # 5 px a character, icons 11
    assert out[1] is True and out[2] is True and out[3] == 1


@needs_js
def test_what_the_mock_up_redraws(tmp_path):
    """Only fields unlike the PSD; stats above 10, stars and types are named, not drawn. A stat cell lights as the last lit
    one (the bar's rounded first cell: its colour) and darkens as the first dark one (none or the rounded last: a shade).
    Bars only where the template has them and the details call the stats "template"."""
    out = node(tmp_path, """
const psd = { name: "A", ability: "Text", hp: 5, def: 2, atk: 3, spd: 9, stars: 2, types: ["wind", "light"] };
const c = m.changes({ name: " A ", ability: "New", hp: 8, def: 12, atk: 3, spd: 10, stars: 3, types: ["light", "wind"] }, psd);
const card = { hp: 5, def: 2, atk: 3, spd: 9 };
out([c, m.cellPlan(5, 8), m.cellPlan(1, 3), m.cellPlan(10, 8), m.cellPlan(9, 8), m.cellPlan(5, 3),
  m.barRows("tall", 720, 1440, card, { stats: { hp: "template", def: "template", atk: "template", spd: "template" } }).map(r => [r.k, r.box.map(Math.round)]),
  m.barRows("tall", 720, 1440, card, { stats: { hp: "pixel" } }).length, m.barRows("square", 720, 720, card, null).length,
  [m.shapeOf(1200, 2400), m.shapeOf(1200, 1800), m.shapeOf(1200, 1200), m.shapeOf(1080, 1920)]]);""")
    assert out[0] == {"name": None, "ability": "New", "stats": {"hp": [5, 8], "spd": [9, 10]}, "skipped": ["DEF", "stars"]}
    assert out[1] == [{"cell": 5, "as": 4}, {"cell": 6, "as": 4}, {"cell": 7, "as": 4}]
    assert out[2] == [{"cell": 1, "as": "lit"}, {"cell": 2, "as": "lit"}]
    assert out[3] == [{"cell": 8, "as": "dark"}, {"cell": 9, "as": "dark"}] and out[4] == [{"cell": 8, "as": "dark"}]
    assert out[5] == [{"cell": 3, "as": 5}, {"cell": 4, "as": 5}]
    assert out[6] == [["hp", [216, 720, 704, 770]], ["def", [216, 796, 704, 846]], ["atk", [216, 871, 704, 922]], ["spd", [216, 947, 704, 997]]]
    assert out[7:] == [0, 0, ["tall", "item", "square", ""]]


@needs_js
def test_a_bar_is_told_from_art(tmp_path):
    """Ten cells with sharp borders read as a bar; a smooth gradient doesn't."""
    out = node(tmp_path, PAINT + """
for (let y = 100; y < 110; y++) for (let x = 20; x < 220; x++) put(x, y, (x - 20) % 20 < 2 ? 10 : 200);
const L = m.luma(img); out([m.isBar(L, W, H, [20, 100, 220, 110]), m.isBar(L, W, H, [20, 30, 220, 40])]);""")
    assert out == [True, False]


@needs_js
def test_the_block_that_holds_the_old_text_is_chosen(tmp_path):
    """The review's wrong mock-ups (2026-10-23): the heaviest cluster of edges was art (a sword), or text left out. Now
    every run of paragraphs is a candidate and calibrate keeps the one the old text fits: its width at the block's size
    (k) and its line count; art taken for a line fails (too tall, not lined up). A text far longer than the block fits
    nowhere (the box instead). One line: its size from its width."""
    out = node(tmp_path, PAINT + """
let seed = 7; const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
for (let y = 30; y < 92; y++) for (let x = 0; x < W; x++) put(x, y, rnd() < 0.5 ? 20 : 230);      // busy art, heavier than the text
text([[40, 200, 150, 162], [52, 188, 170, 182], [46, 194, 190, 202]]);
const L = m.luma(img), area = { box: [0, 20, W, H], seed: [20, H] }, measure = (s, z) => s.length * z * 0.5;
const old = m.tokenize(Array(11).fill("aaaa").join(" "), {}), best = m.bestBlock(L, W, H, area, old, measure);
const heaviest = m.findText(L, W, H, area.box), long = m.bestBlock(L, W, H, area, m.tokenize(Array(60).fill("aaaa").join(" "), {}), measure);
const one = m.calibrate({ box: [40, 150, 200, 158], lines: [[40, 150, 200, 158]], size: 7, pitch: 0, align: "center" }, m.tokenize("aaaa aaaa aaaa aaaa", {}), measure);
out([best && best.b.box, best && best.b.lines.length, heaviest.box[1] < 100, long, [one.ok, Math.round(one.size)],
  m.lineLike({ box: [0, 0, 200, 100], align: "center", lines: [[50, 0, 150, 12], [0, 20, 200, 80], [60, 90, 140, 100]] }),
  m.lineLike({ box: [0, 0, 200, 60], align: "center", lines: [[50, 0, 150, 12], [40, 20, 160, 32], [55, 40, 145, 52]] })]);""")
    box, lines, heaviest_is_art, long, one, art_line, text_lines = out
    assert lines == 3 and 38 <= box[0] <= 42 and 148 <= box[1] <= 151 and 196 <= box[2] <= 202 and 200 <= box[3] <= 204
    assert heaviest_is_art and long is None
    assert one == [True, 17]                           # 160 px for 19 characters at 0.5 em: 16.8 px, not the run of rows (7)
    assert art_line is False and text_lines is True


@needs_js
def test_a_line_the_block_left_out_is_erased_too(tmp_path):
    """A header in another size or a dim last line right next to the block goes with it (charge's "+2 ATK", grodus-x's
    first line); a stat bar there doesn't (two long straight borders), nor a band across the card."""
    out = node(tmp_path, PAINT + """
text([[40, 200, 150, 162], [52, 188, 170, 182], [46, 194, 190, 202]]);
text([[90, 150, 128, 140]]);                                                    // a short header above
for (let y = 214; y < 228; y++) for (let x = 60; x < 236; x++) put(x, y, y === 214 || y === 227 || (x - 60) % 17 === 0 ? 0 : 200);   // a bar below
const L = m.luma(img), found = m.textBlocks(L, W, H, [0, 145, W, 205], { span: 1 })[0];
out([found.lines.length, m.strayLines(L, W, H, found).map(l => l.map(Math.round)), m.barLike(L, W, H, [60, 210, 236, 230]), m.barLike(L, W, H, [40, 150, 200, 162]),
  m.eraseLines({ align: "center", lines: [[100, 0, 130, 10], [60, 20, 180, 30]] })]);""")
    n, stray, bar, text_bar, even = out
    assert n == 3 and len(stray) == 1
    x0, y0, x1, y1 = stray[0]
    assert 80 <= x0 <= 92 and 126 <= y0 <= 130 and 148 <= x1 <= 154 and 138 <= y1 <= 142
    assert bar is True and text_bar is False
    assert even == [[100, 0, 140, 10], [60, 20, 180, 30]]   # a line read short on one side: made even about the centre (120)


@needs_js
def test_badges_are_drawn_and_their_pills_erased(tmp_path):
    """Timing tags are badges (text.js's tags, any case), not "(BOT)" text; another parenthesis stays text. The erase
    fills short gaps between edges in a line (a pill's inside), so an old badge leaves no empty pill."""
    out = node(tmp_path, PAINT + """
const toks = m.tokenize("(1gcd)(3 points) Bake. (see below) (endgt).", {}).flat(2);
for (let y = 100; y < 116; y++) for (let x = 40; x < 90; x++) put(x, y, x === 40 || x === 89 || y === 100 || y === 115 ? 0 : (x - 50) % 9 < 2 && x > 50 && x < 80 ? 255 : 150);   // a pill with letters
const clean = new Uint8ClampedArray(img); for (let y = 100; y < 116; y++) for (let x = 40; x < 90; x++) { const i = (y * W + x) * 4; clean[i] = clean[i + 1] = clean[i + 2] = 60 + Math.round(y / 8); }
const a = new Uint8ClampedArray(img), b = new Uint8ClampedArray(img);
m.erase(a, W, H, [30, 90, 100, 126], { reach: 2, cell: 8, lines: [[40, 100, 90, 116]], close: 14 });
m.erase(b, W, H, [30, 90, 100, 126], { reach: 2, cell: 8 });
const worst = d => { let w = 0; for (let y = 100; y < 116; y++) for (let x = 40; x < 90; x++) { const i = (y * W + x) * 4; w = Math.max(w, Math.abs(d[i] - clean[i])); } return w; };
out([toks.filter(t => t.badge).map(t => [t.badge, t.group]), toks.filter(t => t.text).map(t => t.text), worst(a), worst(b)]);""")
    assert out[0] == [["1GCD", "any"], ["ENDGT", "forced"]]
    assert "(3" in out[1] and "(see" in out[1] and "below)" in out[1]
    assert out[2] <= 6 and out[3] > 40                  # closed: the pill goes; edges alone: its inside stays


@needs_js
def test_the_ability_layer_s_font_is_not_the_icon_font(tmp_path):
    """A text that starts with a badge has the icon font as its layer's first run: the mock-up takes the layer's other
    font (alex-yiik: Cambria, star-cannon: FOT Pop Happiness). The text area starts below the stat rows and spans the
    card's width."""
    out = node(tmp_path, """
const rec = { template: { ability_layers: ["Text/Ability"] }, fonts: { "stoogedings-Regular": { layers: ["Text/Ability"] }, Cambria: { layers: ["Text/Ability", "Text/Stat"] } },
  text: [{ layer: "Text/Ability", font: "stoogedings-Regular" }] };
out([m.layerOf(rec, "ability").font, m.layerOf({ ...rec, fonts: {} }, "ability").font, m.textArea("tall", 720, 1440, 997).box.map(Math.round), m.textArea("item", 720, 1080, 0).seed.map(Math.round)]);""")
    assert out[0] == "Cambria" and out[1] == ""
    assert out[2] == [0, 1014, 720, 1433] and out[3] == [648, 1075]


@needs_js
def test_layout_from_the_details_record(tmp_path):
    """The record's own text boxes (text[].box, canvas px) scaled to the image when the bot has them; the name and text
    layers from template.name_layer / ability_layers, else by name."""
    out = node(tmp_path, """
const rec = { canvas: { w: 1200, h: 2400 }, template: { name_layer: "Text/Title", ability_layers: ["Text/Ability"] },
  text: [{ layer: "Text/Title", font: "KirbyClassic", box: [240, 20, 960, 140] }, { layer: "Text/Ability", font: "Calibri" }, { layer: "Name", font: "X" }] };
out([m.recordBox(rec, "Text/Title", 720), m.recordBox(rec, "Text/Ability", 720), m.layerOf(rec, "name").font, m.layerOf(rec, "ability").font,
  m.layerOf({ text: [{ layer: "Name", font: "Y" }] }, "name").font, m.layerOf(null, "ability")]);""")
    assert out == [[144, 12, 576, 84], None, "KirbyClassic", "Calibri", "Y", None]


# ---- the build's web fonts (build_site.py)
@pytest.fixture(scope="module")
def bs():
    if not BUILD_SITE.is_file():
        pytest.skip(f"no card repo build_site at {BUILD_SITE}")
    spec = importlib.util.spec_from_file_location("site_build_site_previews", BUILD_SITE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "web_fonts"):
        pytest.skip(f"{BUILD_SITE} makes no data/fonts.json: land TTSCardMaker work/site2-previews")
    return module


def _fonts():
    files = sorted(p for p in FONTS.glob("*") if p.suffix.lower() in (".ttf", ".otf", ".ttc", ".otc")) if FONTS.is_dir() else []
    if not files:
        pytest.skip(f"no MDW/Fonts at {FONTS}")
    return files


def test_the_build_reads_font_names_as_font_check(bs):
    """Every font file in MDW/Fonts: the build's reader (no fontTools) gives font_check.face_names' PostScript names,
    collections face by face."""
    differ = []
    for path in _fonts():
        data = path.read_bytes()
        try:
            want = face_names(data)
        except Exception:                          # noqa: BLE001 - fontTools can't read it: the build must refuse it too
            with pytest.raises(ValueError):
                bs.font_faces(data)
            continue
        if bs.font_faces(data) != want:
            differ.append(path.name)
    assert differ == []


def test_web_fonts_map_the_psd_fonts(bs, tmp_path, monkeypatch):
    """data/fonts.json: a PSD font name to its MDW/Fonts file as font_check decides it (a face with exactly that PostScript
    name; since 2026-10-23 no alias or loose match: "Arial-Regular" isn't Arial's ArialMT); never a collection or a file
    over FONT_WEB_MAX; names read once per blob (cached)."""
    files = {p.name: p for p in _fonts()}
    # the files as named since TTSCardMaker 92b5ae6b (named as the fonts), or before
    want = (("Consolas Bold.ttf", "consolab.ttf"), ("Arial.ttf",), ("Book Antiqua.ttc", "BookAntiqua.ttc"))
    pick = [next((n for n in names if n in files), None) for names in want]
    if None in pick:
        pytest.skip("MDW/Fonts lacks Consolas Bold, Arial or Book Antiqua")
    bold, plain, ttc = pick
    repo = tmp_path / "repo"
    (repo / "MDW" / "Fonts").mkdir(parents=True)
    tree = {}
    for i, name in enumerate(pick):
        shutil.copy(files[name], repo / "MDW" / "Fonts" / name)
        tree[f"MDW/Fonts/{name}"] = f"{i:040x}"
    monkeypatch.setattr(bs, "REPO", str(repo))
    monkeypatch.setattr(bs, "CACHE", str(tmp_path / "cache"))
    faces = {name: bs.font_faces((repo / "MDW" / "Fonts" / name).read_bytes()) for name in pick}
    consolas, arial, antiqua = faces[bold][0], faces[plain][0], faces[ttc][0]
    names = {consolas, arial, f"{arial}-Regular", antiqua, "NoSuchFont"}
    first = bs.web_fonts(tree, names)
    assert first == {consolas: {"file": f"MDW/Fonts/{bold}", "bytes": files[bold].stat().st_size},
                     arial: {"file": f"MDW/Fonts/{plain}", "bytes": files[plain].stat().st_size}}
    monkeypatch.setattr(bs, "font_bytes", lambda path: (_ for _ in ()).throw(AssertionError(f"read {path} again")))
    assert bs.web_fonts(tree, names) == first      # from site/.cache/fonts.json
    monkeypatch.setattr(bs, "FONT_WEB_MAX", 10)
    assert bs.web_fonts(tree, names) == {}
    assert not hasattr(bs, "FONT_ALIASES")            # gone with the bot's (font_check: exact names only)


def test_the_build_supplies_a_font_as_font_check_does(bs):
    """The build's rule is font_check's FontIndex.supplies: the same names supplied, for every MDW/Fonts file."""
    from commands.font_check import FontIndex
    files = _fonts()
    index = FontIndex()
    names = set()
    for path in files:
        try:
            faces = face_names(path.read_bytes())
        except Exception:                          # noqa: BLE001 - unreadable: neither offers it
            continue
        index.add(faces)
        names.update(faces)
    probes = sorted(names) + [f"{n}-Regular" for n in sorted(names)[:20]] + ["NoSuchFont"]
    bs_names = set()
    for path in files:
        try:
            bs_names.update(n for n in bs.font_faces(path.read_bytes()) if n)
        except ValueError:
            pass
    assert {n for n in probes if n in bs_names} == {n for n in probes if index.supplies(n)}
