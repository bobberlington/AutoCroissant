"""
The website's card image previews in edit mode (TTSCardMaker site/assets/preview.js and mockup.js, in node), one smoke
test each (owner 2026-11-12): the rendered preview's panel for each state of card-preview.yml's contract
(commands/card_preview.py's result.json) and the quick mock-up's tokens, wrapping and fit; and the bot against the site:
the mock-up's port of card_plates' stat rows. Loads the site from $AC_SITE_SRC (a TTSCardMaker checkout) or the clone.
"""
import json
from pathlib import Path
from subprocess import run as run_process

import pytest

from conftest import SITE_SRC

SITE = SITE_SRC
ASSETS = SITE / "site" / "assets"
PREVIEW_JS, MOCKUP_JS = ASSETS / "preview.js", ASSETS / "mockup.js"


def has(path: Path, word: str) -> bool:
    """For a skip only while a test's feature is still on a work branch."""
    return path.is_file() and word in path.read_text(encoding="utf-8")


def node(tmp_path, body: str):
    """Run body with preview.js as p and mockup.js as m; its stdout as JSON."""
    script = tmp_path / "probe.mjs"
    script.write_text(f"const p = await import({json.dumps(PREVIEW_JS.as_uri())});\nconst m = await import({json.dumps(MOCKUP_JS.as_uri())});\n"
                      f"const out = x => process.stdout.write(JSON.stringify(x));\n{body}\n", encoding="utf-8")
    done = run_process(["node", str(script)], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


def test_the_panel_for_each_state(tmp_path):
    """Baked: current and preview side by side with Diff, each field's outcome, notes escaped; refused: the
    reason (with the bake's render when it made one); an input error is the website's bug; a transient one says try
    again; running: how long; no edit: the button disabled. Baked with no image file: said so, not an error; an error
    for an edit since changed: out of date, Preview again instead of Try again."""
    out = node(tmp_path, """
const v = { card: { name: "Mini <Doomer>", ts: 1, img: "ab" }, n: 1, hash: "h", now: 3 * 60000, scopes: "repo", images: ["https://r/p.webp"], current: "thumbs/ab-720.avif", diff: false };
const e = state => ({ request: "req-00001", at: 0, hash: "h", base: { ts: 1, img: "ab" }, edit: { hp: 8 }, state });
const res = (state, extra = {}) => ({ ...e(state), result: { fields: { hp: state }, notes: ["<script>x</script>"], base: {}, files: {}, ...extra } });
const html = (x, w) => { const r = p.renderedParts(x, w); return r.stage + r.caption; };
out([html(res("baked"), v), html(res("refused", { reason: "stat>10: hp 25", image: "refused" }), v),
  html(res("refused", { reason: "overflow" }), { ...v, images: [] }), html(res("error", { reason: "input: bad edit" }), v),
  html(res("error", { reason: "error: timeout", transient: true }), v), html(e("running"), v), html(null, { ...v, n: 0 }),
  html(res("baked"), { ...v, hash: "other" }), html(res("baked"), { ...v, images: [] }),
  html(res("error", { reason: "error: timeout", transient: true }), { ...v, hash: "other" })]);""")
    baked, refused_img, refused, bug, transient, running, none, stale, no_file, err_stale = out
    assert 'data-pv-img="current"' in baked and 'src="https://r/p.webp"' in baked and 'href="https://r/p.webp"' in baked and 'data-pv="diff"' in baked
    assert "HP baked" in baked and "&lt;script&gt;" in baked and "<script>" not in baked and "Mini &lt;Doomer&gt;" in baked and "3 min ago" in baked
    assert "stat&gt;10: hp 25" in refused_img and "before refusing" in refused_img and "data-pv-img" not in refused
    assert "website bug" in bug and "try again" in transient.lower()
    assert "pv-spin" in running and "about a minute" in running and "16 min" in running and "disabled" in none
    assert "Out of date: the edit changed since" in stale
    assert "no image came with the result" in no_file and 'class="err"' not in no_file and "HP baked" in no_file   # not an error
    assert "Out of date" in err_stale and "Try again" not in err_stale and "Preview again" in err_stale   # Try again would send the newer edit


@pytest.mark.skipif(not has(PREVIEW_JS, "function unzip"), reason="no hand-over unzip in this preview.js")
def test_the_hand_over_unzips(tmp_path):
    """The preview job's hand-over as upload-artifact zips it (deflated, data descriptors) and a stored entry: the
    checking tab's unzip gives the bytes back."""
    import zipfile
    files = {"result.json": json.dumps({"request": "req-00001", "state": "baked"}).encode() * 40, "preview.avif": bytes(range(256)) * 9}
    sink = type("Sink", (), {"b": bytearray(), "write": lambda s, d: s.b.extend(d) or len(d), "flush": lambda s: None})()   # no seek: data descriptors
    with zipfile.ZipFile(sink, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in files.items():
            z.writestr(name, data)
        z.writestr(zipfile.ZipInfo("before.avif"), b"stored")
    assert [i.flag_bits & 8 for i in z.infolist()] == [8, 8, 8]
    (tmp_path / "a.zip").write_bytes(bytes(sink.b))
    out = node(tmp_path, f"""const b = (await import("fs")).readFileSync({json.dumps(str(tmp_path / "a.zip"))});
const f = await p.unzip(b.buffer.slice(b.byteOffset, b.byteOffset + b.length));
out(Object.fromEntries(Object.entries(f).map(([k, v]) => [k, Buffer.from(v).toString("hex")])));""")
    assert out == {**{k: v.hex() for k, v in files.items()}, "before.avif": b"stored".hex()}


def test_the_warm_lane_is_named_and_checked(tmp_path):
    """The press's lane (owner D22): laneBlob is the lane's press_blob (sha and bytes) and the title card-preview.yml's
    run-name makes names the lane and blob, well under GitHub's 512 characters; laneOf names the run that handed over
    the newest result while it has LANE_SAFE_MS left (its idle end, its start's cap); pollDelay looks for a named lane's
    hand-over from LANE_FIRST_MS every LANE_POLL from a round's start."""
    import base64
    import re
    import yaml
    from commands.card_preview_lane import press_blob, press_of
    request, card, edit = "zomoroko-mw5qp6i1-xk2v", "Odd {Card}", json.dumps({"ability": "Deal 2 {fire} — damage\n" * 40, "hp": 7}, ensure_ascii=False, separators=(",", ":"))
    out = node(tmp_path, f"""const now = 1e12, s = 1000;
const own = {{ request: "r-00000001", at: now - 500 * s, hash: "h", state: "baked", run: {{ id: 101, started: now - 500 * s }}, from: 101, fromAt: now - 480 * s }};
const next = fromAt => ({{ request: "r-00000002", at: now - 70 * s, hash: "h", state: "baked", lane: 101, from: 101, fromAt }});
const lane = (age, extra = {{}}) => ({{ c: [{{ request: "r-00000003", at: now - age, hash: "h", state: "running", lane: 101, ...extra }}] }});
out([await p.laneBlob({json.dumps(request)}, {json.dumps(card)}, {json.dumps(edit)}),
  p.laneOf({{ a: [own, next(now - 60 * s)] }}, now), p.laneOf({{ a: [{{ ...own, run: {{ id: 101, started: now - 1070 * s }} }}, next(now - 60 * s)] }}, now),
  p.laneOf({{ a: [{{ ...own, run: {{ id: 101, started: now - 1000 * s }} }}, next(now - 60 * s)] }}, now),
  p.laneOf({{ a: [own, next(now - 230 * s)] }}, now), p.laneOf({{}}, now),
  p.pollDelay(lane(1 * s), now), p.pollDelay(lane(5 * s), now), p.pollDelay(lane(5 * s), now, () => true, 180), p.pollDelay(lane(5 * s), now, () => true, 400),
  p.pollDelay(lane(5 * s, {{ fromAt: now }}), now), p.pollDelay(lane(16 * s), now)]);""")
    blob, named, capped, long, idle, none, *delays = out
    data, sha = press_blob(request, card, edit)
    assert blob == {"sha": sha, "base64": base64.b64encode(data).decode()} and len(data) > 1024
    run_name = yaml.safe_load((SITE / ".github" / "workflows" / "card-preview.yml").read_text(encoding="utf-8"))["run-name"]
    assert run_name == "Card preview ${{ inputs.request }}${{ inputs.lane && format(' lane {0}', inputs.lane) || '' }}"
    title = re.sub(r"\$\{\{.*\}\}", lambda m: f" lane 101-{sha}", run_name.replace("${{ inputs.request }}", request.ljust(64, "x")))
    assert press_of(title) == (request.ljust(64, "x"), 101, sha) and len(title) < 150
    assert (named, capped, long, idle, none) == (101, None, 101, None, None)     # 18 minutes from its start
    assert delays == [1000, 250, 70, 0, 7000, 1000]


def test_one_wait_with_its_run_known_is_checked_every_250_ms(tmp_path):
    """pollDelay: 250 ms from the last round's start while one preview waits with its run known, nothing handed over
    and the tab reads it directly; else as before (two waiting, no run yet, handed over, not read directly, past a
    minute, before 12 s)."""
    out = node(tmp_path, """const now = 1e12, s = 1000, run = { id: 7 };
const one = (age, x = {}) => ({ c: [{ request: "r-00000001", at: now - age, hash: "h", state: "running", ...x }] });
const two = { c: [{ request: "r-00000001", at: now - 13 * s, hash: "h", state: "running", run }, { request: "r-00000002", at: now - 14 * s, hash: "h", state: "running" }] };
out([p.pollDelay(one(13 * s, { run }), now), p.pollDelay(one(13 * s, { run }), now, () => true, 100), p.pollDelay(one(13 * s, { run }), now, () => true, 400),
  p.pollDelay(two, now), p.pollDelay(one(13 * s), now), p.pollDelay(one(13 * s, { run, fromAt: now }), now), p.pollDelay(one(13 * s, { run }), now, () => false),
  p.pollDelay(one(61 * s, { run }), now), p.pollDelay(one(11 * s, { run }), now)]);""")
    assert out == [250, 150, 0, 1000, 1000, 1000, 1000, 3000, 1000]


@pytest.mark.parametrize("miss", [False, pytest.param(True, marks=pytest.mark.skipif(not has(PREVIEW_JS, "ZIP_404S"), reason="no zip 404s looked past"))])
def test_the_editor_takes_the_lanes_hand_over(tmp_path, miss):
    """createPreviews against a stand-in GitHub on a fake clock: press 1 names no lane and shows its run's hand-over;
    press 2 names that run and its blob (posted beside the dispatch: the lane's press_blob), rings the lane's doorbell
    beside them (a commit status at press 1's result's commit: the lane's CONTEXT and RUNG; not after the blob's answer)
    and shows the lane's card-preview-<request> 2 s after the press; one toast each; a 422 with a lane dispatches again
    without it, and no lane is named after that, nor after a blob not posted. Press 1's hand-over is listed from 12.5 s: its list goes
    If-None-Match the last ETag (a 304: not yet) every 250 ms while it waits alone. miss: each zip's first fetch a 404
    (listed just before it's finalized, run 44465554232): both shown on the next round (250 ms on), press 2 still named
    to the lane."""
    import base64
    from commands.card_preview_lane import CONTEXT, RUNG, press_blob
    out = node(tmp_path, f"const miss = {str(miss).lower()}, zipped = new Set();\n" + """let T = 1e12; const q = []; Date.now = () => T;
globalThis.setTimeout = (fn, ms) => { q.push([T + ms, fn]); return q.length; }; globalThis.clearTimeout = () => {};
const run = async until => { while (q.length) { q.sort((a, b) => a[0] - b[0]); if (q[0][0] > until) break; const [at, fn] = q.shift(); T = Math.max(T, at); await fn(); for (let i = 0; i < 50; i++) await new Promise(r => setImmediate(r)); } T = Math.max(T, until); };
const mem = new Map(); globalThis.localStorage = { getItem: k => mem.get(k) ?? null, setItem: (k, v) => mem.set(k, v), removeItem: k => mem.delete(k) };
globalThis.window = { addEventListener() {} }; globalThis.document = { visibilityState: "visible", addEventListener() {} }; URL.createObjectURL = () => "blob:x";
const zip = text => { const n = Buffer.from("result.json"), d = Buffer.from(text), h = Buffer.alloc(30), c = Buffer.alloc(46), e = Buffer.alloc(22);
  h.writeUInt32LE(0x04034b50, 0); h.writeUInt16LE(n.length, 26); c.writeUInt32LE(0x02014b50, 0); c.writeUInt32LE(d.length, 20); c.writeUInt16LE(n.length, 28);
  e.writeUInt32LE(0x06054b50, 0); e.writeUInt16LE(1, 10); e.writeUInt32LE(30 + n.length + d.length, 16); const b = Buffer.concat([h, n, d, c, n, e]); return b.buffer.slice(b.byteOffset, b.byteOffset + b.length); };
const { createHash } = await import("node:crypto");
const sent = [], toasts = [], listed = [], blobs = [], refs = [], posts = [];
let n304 = 0;
let refuse = false, failBlob = false;
const github = async (path, o = {}) => {
  if (o.method === "POST") posts.push(path.endsWith("/git/blobs") ? "blob" : path.includes("/statuses/") ? "status" : "dispatch");
  if (path.endsWith("/git/blobs")) { blobs.push(o.body); if (failBlob) { const e = new Error("GitHub 403"); e.status = 403; throw e; }
    const raw = Buffer.from(o.body.content, "base64"); return { sha: createHash("sha1").update(Buffer.concat([Buffer.from(`blob ${raw.length}\0`), raw])).digest("hex") }; }
  if (path.includes("/statuses/")) { refs.push([T - 1e12, o.method, path.split("/statuses/")[1], o.body]); return null; }
  if (o.method === "POST") { sent.push({ ...o.body.inputs }); if (refuse && o.body.inputs.lane) { const e = new Error("GitHub 422"); e.status = 422; throw e; } return null; }
  if (path.includes("/contents/")) { const e = new Error("404"); e.status = 404; throw e; }
  if (path.includes("/workflows/")) return { workflow_runs: [{ id: 101, display_title: "Card preview " + sent[0].request, html_url: "https://github.com/o/r/actions/runs/101", status: "in_progress", run_started_at: new Date(1e12).toISOString() }] };
  const z = path.split("/zip/")[1];
  if (z && miss && !zipped.has(z)) { zipped.add(z); const e = new Error("GitHub 404"); e.status = 404; throw e; }
  if (z) return zip(JSON.stringify({ request: z, state: "baked", files: {}, base: { commit: (z === sent[0].request ? "c" : "d").repeat(40) } }));   // the lane followed
  const name = path.split("name=")[1], own = name === "card-preview" ? sent[0].request : name.slice(13);
  listed.push([T - 1e12, name]);
  const list = { artifacts: path.includes("/runs/101/") && ((name === "card-preview" && T - 1e12 >= 12500) || own === sent[1]?.request) ? [{ name, archive_download_url: "https://api.github.com/zip/" + own }] : [] };
  const tag = `W/"${createHash("sha1").update(JSON.stringify(list)).digest("hex")}"`, headers = new Headers({ ETag: tag });
  if (o.etag === tag) { n304++; return { data: null, headers, status: 304 }; }
  return o.full ? { data: list, headers } : list;
};
const app = { S: { data: { repo: "o/r" } }, editor: { me: { login: "ed" }, scopes: "repo", github }, toast: m => toasts.push(m), badges() {} };
const pv = p.createPreviews(app), card = id => ({ id, key: "Mini Doomer", name: "Mini Doomer" });
await pv.start(card("c0"), { hp: 8 }); await run(T + 20000);
const second = T;
await pv.start(card("c1"), { hp: 7 }); await run(T + 20000);
refuse = true; await pv.start(card("c2"), { hp: 6 }); await pv.start(card("c3"), { hp: 5 });
refuse = false; failBlob = true; const pv2 = p.createPreviews(app); await pv2.start(card("c4"), { hp: 4 }); await pv2.start(card("c5"), { hp: 3 });
out({ sent, toasts, blobs, refs, second: second - 1e12, states: ["c0", "c1"].map(c => pv.entries(c)[0].state), from: pv.entries("c1")[0].from, head: pv.entries("c1")[0].fromHead, lane: listed.filter(([t, n]) => n !== "card-preview").map(([t]) => t - (second - 1e12)),
  shown: [pv.entries("c0")[0].fromAt - 1e12, pv.entries("c1")[0].fromAt - second],
  named: ["c1", "c4"].map(c => pv2.entries(c)[0].lane ?? null), n304, posts });""")
    sent = out["sent"]
    data, sha = press_blob(sent[1]["request"], "Mini Doomer", sent[1]["edit"])
    assert "lane" not in sent[0] and sent[1]["lane"] == f"101-{sha}"
    assert out["blobs"][0] == {"content": base64.b64encode(data).decode(), "encoding": "base64"}
    assert out["states"] == ["baked", "baked"] and out["from"] == 101 and out["lane"] == ([2000, 2250] if miss else [2000])   # the lane's, 2 s on
    assert out["shown"] == ([12750, 2250] if miss else [12500, 2000]) and out["n304"] == 1   # 12 s, 12.25 (304), 12.5
    assert out["head"] == "c" * 40                                  # still rung where the lane lists, not where it baked
    at, method, commit, body = out["refs"][0]
    rung = RUNG.fullmatch(body["description"])
    assert rung and rung.groups() == ("101", sent[1]["request"], sha) and (method, commit) == ("POST", "c" * 40)
    assert body["context"] == CONTEXT and body["state"] == "success" and at == out["second"] and len(body["description"]) <= 140
    assert out["posts"][:4] == ["dispatch", "blob", "status", "dispatch"]   # press 2's doorbell beside its blob, not after it
    assert len(out["refs"]) == 3                                    # the 422's press and the blob's rang too (beside), the next ones none
    assert out["toasts"].count("Preview ready: Mini Doomer") == 2
    assert "lane" in sent[2] and "lane" not in sent[3] and "lane" not in sent[4]      # the 422's retry, then none named
    assert "lane" in sent[5] and "lane" not in sent[6] and out["named"] == [101, None]  # a blob not posted: none again


@pytest.mark.skipif(not has(PREVIEW_JS, "laneNext"), reason="no next cards in this preview.js")
def test_the_lane_reads_the_next_card(tmp_path):
    """A PSD-kind change on another card while a lane is up rings it as the lane's next card (a status at the lane's
    commit: card_preview_lane.NEXT), once per card and lane, none for a card the lane baked; its press rings as ever."""
    from commands.card_preview_lane import NEXT, RUNG
    out = node(tmp_path, """let T = 1e12; Date.now = () => T; globalThis.setTimeout = () => 0; globalThis.clearTimeout = () => {};
const was = { card: "c0", request: "ed-00000001", at: T - 60000, hash: "h", state: "baked", base: { blob: "aaaaaaaa" }, from: 101, fromAt: T - 50000, fromHead: "c".repeat(40) };
const mem = new Map([[p.KEY, JSON.stringify({ c0: [was] })]]);
globalThis.localStorage = { getItem: k => mem.get(k) ?? null, setItem: (k, v) => mem.set(k, v), removeItem: k => mem.delete(k) };
globalThis.window = { addEventListener() {} }; globalThis.document = { visibilityState: "visible", addEventListener() {} };
const { createHash } = await import("node:crypto"), refs = [];
const github = async (path, o = {}) => {
  if (path.includes("/statuses/")) refs.push([o.method, o.body.description, path.split("/statuses/")[1]]);
  if (!path.endsWith("/git/blobs")) return null;
  const raw = Buffer.from(o.body.content, "base64"); return { sha: createHash("sha1").update(Buffer.concat([Buffer.from(`blob ${raw.length}\0`), raw])).digest("hex") };
};
const app = { S: { data: { repo: "o/r" } }, editor: { me: { login: "ed" }, scopes: "repo", github }, toast() {}, badges() {} };
const pv = p.createPreviews(app), card = (id, blob) => ({ id, key: id, name: id, blob });
pv.editing(card("c1", "bbbbbbbb")); pv.editing(card("c1", "bbbbbbbb")); pv.editing(card("c0", "aaaaaaaa")); pv.editing(card("c2", "../x"));
await pv.start(card("c1", "bbbbbbbb"), { hp: 7 });
for (let i = 0; i < 20; i++) await new Promise(r => setImmediate(r));
out({ refs, lane: pv.entries("c1")[0].lane ?? null });""")
    refs = out["refs"]
    assert NEXT.fullmatch(refs[0][1]).groups() == ("101", "bbbbbbbb") and refs[0][::2] == ["POST", "c" * 40]
    assert out["lane"] == 101 and len(refs) == 2 and RUNG.fullmatch(refs[1][1]) and refs[1][2] == "c" * 40   # the press's doorbell


@pytest.mark.skipif(not has(PREVIEW_JS, "function claim("), reason="the tab in front doesn't take the check in this preview.js")
def test_the_tab_in_front_shows_the_hand_over(tmp_path):
    """Two tabs, one localStorage, storage events to the other, each its own visibility; the dispatch answers in 0.3 s.
    Press 1 (cold) in B, the person moves to A 2 s later: A takes the check and shows the run's hand-over (listed 12.2 s
    on), B waits for the branch. Back in B, press 2 (warm) takes the check from A; the person moves to A 0.1 s later,
    before the dispatch answers: A reads the lane's hand-over (5.06 s on, run 44465651236) itself, about 5.5 s after the
    press; B stops checking. One toast each, in A."""
    out = node(tmp_path, """const { AsyncLocalStorage } = await import("node:async_hooks"), tab = new AsyncLocalStorage();
let T = 1e12, n = 0; const q = []; Date.now = () => T;
globalThis.setTimeout = (fn, ms = 0) => { const t = tab.getStore(), id = ++n; q.push([T + ms, () => tab.run(t, fn), id]); return id; };
globalThis.clearTimeout = id => { const i = q.findIndex(x => x[2] === id); if (i >= 0) q.splice(i, 1); };
const run = async until => { while (q.length) { q.sort((a, b) => a[0] - b[0]); if (q[0][0] > until) break; const [at, fn] = q.shift(); T = Math.max(T, at); await fn(); for (let i = 0; i < 50; i++) await new Promise(r => setImmediate(r)); } T = Math.max(T, until); };
const mem = new Map(), tabs = [], on = (t, type) => t.on.filter(x => x[0] === type).map(x => x[1]);
const tell = (key, newValue) => { const me = tab.getStore(); for (const t of tabs) if (t !== me) q.push([T + 2, () => tab.run(t, () => on(t, "storage").forEach(f => f({ key, newValue }))), 0]); };
globalThis.localStorage = { getItem: k => mem.get(k) ?? null, setItem: (k, v) => { mem.set(k, String(v)); tell(k, String(v)); }, removeItem: k => { mem.delete(k); tell(k, null); } };
const listen = (type, f) => tab.getStore().on.push([type, f]);
globalThis.window = { addEventListener: listen }; globalThis.document = { get visibilityState() { return tab.getStore().vis; }, addEventListener: listen };
URL.createObjectURL = () => "blob:x";
const show = who => { for (const t of tabs) { t.vis = t === who ? "visible" : "hidden"; tab.run(t, () => on(t, "visibilitychange").forEach(f => f())); } };
const zip = text => { const n = Buffer.from("result.json"), d = Buffer.from(text), h = Buffer.alloc(30), c = Buffer.alloc(46), e = Buffer.alloc(22);
  h.writeUInt32LE(0x04034b50, 0); h.writeUInt16LE(n.length, 26); c.writeUInt32LE(0x02014b50, 0); c.writeUInt32LE(d.length, 20); c.writeUInt16LE(n.length, 28);
  e.writeUInt32LE(0x06054b50, 0); e.writeUInt16LE(1, 10); e.writeUInt32LE(30 + n.length + d.length, 16); const b = Buffer.concat([h, n, d, c, n, e]); return b.buffer.slice(b.byteOffset, b.byteOffset + b.length); };
const { createHash } = await import("node:crypto"), presses = [];
const result = r => JSON.stringify({ request: r, state: "baked", files: {}, base: { commit: "c".repeat(40) } }), none = () => { const e = new Error("GitHub 404"); e.status = 404; throw e; };
const github = me => async (path, o = {}) => {
  if (path.endsWith("/git/blobs")) { const raw = Buffer.from(o.body.content, "base64"); return { sha: createHash("sha1").update(Buffer.concat([Buffer.from(`blob ${raw.length}\\0`), raw])).digest("hex") }; }
  if (o.method === "POST") { const at = T; await new Promise(r => setTimeout(r, 300)); if (!path.includes("/statuses/")) presses.push({ ...o.body.inputs, at, run: 101 + presses.length }); return null; }
  me.gets.push(T);
  const age = x => T - x.at;
  if (path.includes("/contents/")) { const x = presses.find(x => path.includes(x.request)); return x && age(x) >= (x.lane ? 16000 : 15000) ? result(x.request) : none(); }
  if (path.includes("/workflows/")) return { workflow_runs: presses.map(x => ({ id: x.run, display_title: "Card preview " + x.request, html_url: "https://github.com/o/r/actions/runs/" + x.run, status: "in_progress", run_started_at: new Date(x.at).toISOString() })) };
  const z = path.split("/zip/")[1];
  if (z) return zip(result(z));
  const name = path.split("name=")[1], run = +path.split("/runs/")[1].split("/")[0];
  const x = name === "card-preview" ? presses.find(x => x.run === run && age(x) >= 12200) : presses.find(x => x.request === name.slice(13) && String(x.lane).startsWith(run + "-") && age(x) >= 5060);
  const list = { artifacts: x ? [{ name, archive_download_url: "https://api.github.com/zip/" + x.request }] : [] };
  return o.full ? { data: list, headers: new Headers() } : list;
};
const open = name => { const t = { name, on: [], gets: [], toasts: [], vis: "hidden" }; tabs.push(t);
  t.pv = tab.run(t, () => p.createPreviews({ S: { data: { repo: "o/r" } }, editor: { me: { login: "ed" }, scopes: "repo", github: github(t) }, toast: m => t.toasts.push(m), badges() {} }));
  return t; };
const B = open("B"), A = open("A"), card = id => ({ id, key: id, name: id }), states = id => [A, B].map(t => t.pv.entries(id)[0].state);
show(B); const t0 = T;
tab.run(B, () => B.pv.start(card("c0"), { hp: 8 })); await run(t0 + 2000); show(A); await run(t0 + 14000);
const p1 = { at14: states("c0"), shown: A.pv.entries("c0")[0].fromAt - t0, bGets: B.gets.filter(x => x > t0 + 2000).length };
await run(t0 + 39000); show(B); const t1 = t0 + 40000; await run(t1);
tab.run(B, () => B.pv.start(card("c1"), { hp: 7 })); await run(t1 + 100); show(A); await run(t1 + 6000);
const p2 = { lane: presses[1].lane ?? null, at6: states("c1"), shown: A.pv.entries("c1")[0].fromAt - t1 };
await run(t1 + 30000);
out({ p1, p2: { ...p2, after: states("c1"), bGets: B.gets.filter(x => x > t1 + 100).length }, toasts: [A.toasts, B.toasts].map(l => l.filter(m => !m.startsWith("Preview started"))) });""")
    p1, p2 = out["p1"], out["p2"]
    assert p1["at14"] == ["baked", "running"] and 12200 < p1["shown"] <= 14000 and p1["bGets"] == 0   # A shows it, B stopped
    assert p2["lane"] and p2["lane"].startswith("101-")                 # press 2 named press 1's run as its lane
    assert p2["at6"] == ["baked", "running"] and 5000 < p2["shown"] <= 6000   # A read the lane's hand-over itself
    assert p2["after"] == ["baked", "baked"] and p2["bGets"] <= 1
    assert out["toasts"] == [["Preview ready: c0", "Preview ready: c1"], []]


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


def test_typed_letters_the_font_lacks_look_as_the_bake_prints_them(tmp_path):
    """A typed letter the card's font can't draw (fonts.json lacks): a dash or ellipsis plain (owner D13), any other (a
    plain one it lacks too) the font's .notdef, said in the caption; kept letters, typed badges and letters typed into
    another font's run (The Structure's digit) stay; the bake's note in plain words."""
    out = node(tmp_path, """
const D = String.fromCharCode(0x2014), E = String.fromCharCode(0x2026), typed = "Heal 2 " + D + " now" + E + " (1CD)";
const show = r => [[...r.text].map(c => (c.charCodeAt(0) >= 0xfdd0 ? `<${m.glyphRuns(c)[0]}>` : c)).join(""), r.boxes];
out([show(m.glyphPlan("(WOF) +1 Heal (+1 DEF).", "(WOF) +1 Heal.", "+")), show(m.glyphPlan(typed, "Heal 2", D + E)),
  show(m.glyphPlan(typed, "Heal 2", "+.:()-" + D + E)), show(m.glyphPlan("by 2 bars.", "by 1 bar.", "0123456789")),
  p.plainNote("font: name: '()' has no glyph in its font (drawn as a box)"),
  p.mockNote({ noGlyph: [["text", "+", false], ["name", ".", true]] })]);""")
    assert out[:4] == [["(WOF) +1 Heal (<1>1 DEF).", "+"], ["Heal 2 <-> now<...> (1CD)", ""], ["Heal 2 <1> now<3> (1CD)", "-."],
                       ["by 2 bars.", ""]]
    assert out[4] == "The title's font has no '(' or ')': it prints as a box or a gap."
    assert out[5] == "The ability's font has no '+': the bake prints a box there. The title's font has no '.': the bake leaves a gap there."


@pytest.mark.skipif(not has(MOCKUP_JS, "function glyphFont"), reason="no picked fonts in this mockup.js")
def test_typed_letters_the_font_cant_draw_take_the_font_the_bake_picks(tmp_path):
    """Owner D15 rest: a typed letter its font lacks or draws empty (fonts.json lacks, empty) is drawn in the font the
    bake sets it in (psd_edit.glyph_fonts): the one the run it is typed into prints in, else its layer's (details'
    faces), else another shown layer's, else Myriad Pro; with no pick a box (lacked) or as the font draws it (empty); a
    kept one in the font its layer prints it in. The text keeps its letters (a colour alone still matches the card)."""
    out = node(tmp_path, """
const map = { B: { file: "b", lacks: "+", empty: ":%" }, E: { file: "e" }, A: { file: "a", lacks: "+" }, C: { file: "c" },
  H: { file: "h" }, M: { file: "m", lacks: "0123456789" }, I: { file: "i" }, "MyriadPro-Regular": { file: "m" } };
const ab = { layer: "Text/Ability", id: 1, text: "On death: [fire] 5", lf: "B", faces: [["B", "5Oadehnt"], ["E", ":"], ["A", ":"]] };
const rec = { text: [ab, { layer: "Text/Hidden", id: 3, text: "+", font: "E", visible: false }, { layer: "Text/Name", id: 2, text: "A+B", font: "C" }] };
const plan = (t, o, lacks, empty, layer, picks = true) => { const r = m.glyphPlan(t, o, lacks, picks ? (s, from) => m.glyphFont(rec, layer, s[0], map, false, from) : null,
  empty, c => m.glyphFont(rec, layer, c, map, true)); return [r.text, r.boxes, Object.fromEntries(r.fonts)]; };
const show = r => [[...r[0]].map(c => (c.charCodeAt(0) >= 0xfdd0 ? `<${m.glyphRuns(c)[0]}>` : c)).join(""), r[1], r[2]];
out([[":", "+", "%", "["].map(c => m.glyphFont(rec, ab, c, map)), m.glyphFont(rec, ab, "%", { ...map, "MyriadPro-Regular": undefined }),
  plan("On death: gain +1, 5%.", "On death gain 1, 5.", "+", ":%", ab), show(plan("Gain: +1.", "Gain 1.", "+", ":", ab, false)),
  plan("+25% Base", "+25% Base", "", "%", { lf: "B", faces: [["B", "+25Baes"], ["H", "%"]] }),
  plan("by 3 bar", "by 1 bar", "0123456789", "", { lf: "M", faces: [["M", "abry"], ["I", "1"]] }),
  m.glyphRuns("a+b\ufdd3", new Map([["+", "F"], [".", "G"]]))]);""")
    assert out[0] == ["E", "C", "MyriadPro-Regular", "MyriadPro-Regular"] and out[1] == ""
    assert out[2] == ["On death: gain +1, 5%.", "", {":": "E", "+": "C", "%": "MyriadPro-Regular"}]
    assert out[3] == ["Gain: <1>1.", "+", {}]
    assert out[4] == ["+25% Base", "", {"%": "H"}]            # kept: Battle Spear's Chiller '%'; a colour alone matches
    assert out[5] == ["by 3 bar", "", {"3": "I"}]             # typed into the digits' run: The Structure's IRONMAN
    assert out[6] == ["a", {"s": "+", "fam": "F"}, "b", {"s": "...", "fam": "G"}]


@pytest.mark.skipif(not has(MOCKUP_JS, "function lineSplices"), reason="no lineSplices in this mockup.js")
def test_words_changed_on_several_lines_splice_each_line(tmp_path):
    """geoPlan on the card's own lines: changes on two lines or in two paragraphs give one splice per line (a paragraph's
    trailing spaces left out); one line: the single splice; a new line break: neither."""
    out = node(tmp_path, """
const was = "Hit an Evil foe and the Cookie dies.\\nHeal 2.";
const R = { was, lines: [[0, 19, 0], [20, 36, 0], [37, 44, 1]].map(([s, e, para]) => ({ s, e, para })), x: new Map([...was].map((_, i) => [i, i])) };
const plan = t => { const p = m.geoPlan(R, t); return [p.splice && p.splice.run, p.splices && p.splices.map(s => [s.i, s.a, s.b, s.at, s.run])]; };
out(["Hit an evil foe and the cookie dies.\\nHeal 2.", "Hit an Evil fee and the Cookie dies. \\nHeal 3.", "Hit an Evil fee and the Cookie dies.\\nHeal 2.",
  "Hit an Evil fee and the Cookie dies.\\nHeal 2.\\nNew."].map(plan));""")
    assert out == [[None, [[0, 7, 8, 7, "e"], [1, 24, 25, 24, "c"]]], [None, [[0, 13, 14, 13, "e"], [2, 42, 43, 43, "3"]]], ["e", None], [None, None]]


@pytest.mark.skipif(not has(MOCKUP_JS, r"/[()\n]/.test(inner)"), reason="no cost swaps in this mockup.js")
def test_a_cost_badge_is_swapped_alone(tmp_path):
    """A cost (one pill glyph on the card) for another, as a timing tag: spliced alone at the old pill's width times the
    new one's over it, the rest of the line moved by the difference."""
    out = node(tmp_path, """
const text = "Gain 1 (2 points) now.", g = [...text].map((ch, i) => (ch === " " ? [i, 10 + i * 10] : [i, 10 + i * 10, 11 + i * 10, 19 + i * 10]));
const c = { text, lines: [[0, 22, 100, 10, 230, 80, 105, 0]], glyphs: [...g.slice(0, 7), [7, 80, 81, 169], ...g.slice(17)], frame: [0, 300] };
const measure = t => [...t].length * 10; measure.hx = 1;
const R = m.recordLayout(c, text, 1), fit = now => { const f = m.spliceFit(R, m.geoPlan(R, now).splice, { measure, z: 20 }); return f && [f.tag?.at, Math.round(f.tag?.w), Math.round(f.ds)]; };
out(["Gain 1 (3 points) now.", "Gain 1 (1 point) now."].map(fit));""")
    assert out == [[7, 100, 0], [7, 90, -10]]


@pytest.mark.skipif(not has(MOCKUP_JS, "mOf = j => R.fr"), reason="no second-font runs in this mockup.js")
def test_letters_typed_into_a_second_fonts_run(tmp_path):
    """psd_details fr: a digit typed into a line set in another font is spliced at the line's size, measured in that
    font (without fr: shrunk to make the main font fit); a line with a few letters in it measures each run in its own font."""
    out = node(tmp_path, """
const lf = (t, z) => [...t].length * z / 2, f2 = (t, z) => [...t].length * z / 3.4; lf.hx = f2.hx = 1;
const plate = (text, w) => { let x = 10; const g = [...text].map((ch, i) => { const a = x; x += w(i); return ch === " " ? [i, a] : [i, a, a + 1, x - 1]; });
  return { text, lines: [[0, text.length, 100, 10, x, 80, 105, 0]], glyphs: g, frame: [0, 300] }; };
const fit = (text, w, fr, now) => { const R = m.recordLayout(plate(text, w), text, 1), sp = m.geoPlan(R, now).splice;
  if (fr) R.fr = [[...fr, "f2", f2]];
  const f = m.spliceFit(R, sp, { measure: lf, z: 20, font: fr ? R.fr[0] : null }); return f && [+f.z.toFixed(2), +f.k.toFixed(2), +f.hx.toFixed(2)]; };
const cost = "Cost: 4 Points", mixed = "Lose 1 bar.", w2 = 20 / 3.4;
out([fit(cost, () => w2, null, "Cost: 5 Points"), fit(cost, () => w2, [0, 14], "Cost: 5 Points"), fit(mixed, i => (i === 5 ? w2 : 10), [5, 6], "Lose 2 bar.")]);""")
    assert out == [[11.76, 0.59, 1], [20, 1, 1], [20, 1, 1]]


@pytest.mark.skipif(not has(MOCKUP_JS, "export function paintLines"), reason="no exported paintLines in this mockup.js")
def test_a_redrawn_pill_keeps_the_layers_stroke(tmp_path):
    """A badge painted again on a layer with a Stroke is outlined in its colour at its width under the pill (as the card
    prints it); not on a one-colour layer, nor without the record's Stroke width."""
    out = node(tmp_path, """const rec = [], ctx = new Proxy({}, { set: (t, k, v) => (t[k] = v, true), get: (t, k) => k in t ? t[k]
  : k === "measureText" ? () => ({ width: 10, actualBoundingBoxAscent: 14 }) : k === "createLinearGradient" ? () => ({ addColorStop() {} })
  : () => { if (k === "stroke" || k === "fill") rec.push(k === "fill" ? k : `${k} ${t.strokeStyle} ${t.lineWidth}`); } });
const look = { at: () => 0, base: 40, pitch: 0, z: 20, fam: "x", fill: [255, 255, 255], stroke: [1, 2, 3], strokeW: 6, imgs: new Map() };
const run = o => { rec.length = 0; m.paintLines(ctx, [{ pieces: [{ badge: "2GCD", group: "any", w: 60 }] }], { ...look, ...o }); return [...rec]; };
out([run({}), run({ one: true }), run({ strokeW: 0 })]);""")
    ring, one, bare = out
    assert ring[:2] == ["stroke rgb(1,2,3) 6", "fill"]
    assert not any(s.endswith("rgb(1,2,3) 6") for s in one + bare) and bare[0] == "fill"


@pytest.mark.skipif(not has(MOCKUP_JS, "function seeThrough"), reason="no see-through Strokes in this mockup.js")
def test_a_see_through_stroke_is_one_layer_at_its_opacity(tmp_path):
    """A Stroke under 100% (psd_details rim): the letters' outlines drawn opaque on a layer of their own, put on the card
    at its opacity before the fills (where two letters' outlines meet they don't add up); at 100% straight on the card."""
    out = node(tmp_path, """const rec = [], proxy = (t, on) => new Proxy(t, { set: (o, k, v) => (o[k] = v, true), get: (o, k) => k in o ? o[k] : (...a) => on(o, k, a) });
globalThis.document = { createElement: () => { const g = proxy({}, (o, k) => { if (k === "strokeText") rec.push(`layer ${o.strokeStyle}`); }); return { getContext: () => g }; } };
const ctx = proxy({ canvas: { width: 90, height: 60 }, globalAlpha: 1 }, (o, k) => { if (k === "strokeText" || k === "fillText") rec.push(`${k} ${k === "fillText" ? o.fillStyle : o.strokeStyle}`); if (k === "drawImage") rec.push(`layer at ${o.globalAlpha}`); });
const run = strokeA => { rec.length = 0; m.paintLines(ctx, [{ pieces: [{ text: "Hi", w: 20 }] }], { at: () => 0, base: 40, pitch: 0, z: 20, fam: "x", fill: [9, 9, 9], stroke: [1, 2, 3], strokeW: 8, strokeA, imgs: new Map() }); return [...rec]; };
out([run(0.64), run(1)]);""")
    assert out == [["layer rgb(1,2,3)", "layer at 0.64", "fillText rgb(9,9,9)"], ["strokeText rgb(1,2,3)", "fillText rgb(9,9,9)"]]


@pytest.mark.skipif(not has(MOCKUP_JS, "if (im && one === true) paintMask"), reason="no one-colour icons in this mockup.js")
def test_a_typed_icon_under_an_overlay_takes_its_colour(tmp_path):
    """Under the Ability's overlay a typed icon is its shape in the text's fill (the overlay paints over it in the
    bake); elsewhere, or with the overlay off for a whole colour ("badges"), the icon's own picture."""
    out = node(tmp_path, """const rec = [], pic = { pic: 1 }, proxy = (t, on) => new Proxy(t, { set: (o, k, v) => (o[k] = v, true), get: (o, k) => k in o ? o[k] : (...a) => on(o, k, a) });
globalThis.document = { createElement: () => { const g = proxy({}, (o, k) => { if (k === "fillRect") rec.push(`mask ${o.fillStyle}`); }); return { getContext: () => g }; } };
const ctx = proxy({}, (o, k, a) => { if (k === "drawImage") rec.push(a[0] === pic ? "icon" : "canvas"); });
const run = one => { rec.length = 0; m.paintLines(ctx, [{ pieces: [{ icon: "f.webp", a: 1, w: 20 }] }], { at: () => 0, base: 40, pitch: 0, z: 20, fam: "x", fill: [200, 10, 10], stroke: null, imgs: new Map([["f.webp", pic]]), one }); return [...rec]; };
out([run(false), run(true), run("badges")]);""")
    assert out == [["icon"], ["mask rgb(200,10,10)", "canvas"], ["icon"]]


@pytest.mark.skipif(not has(MOCKUP_JS, "kind = OVERLAY, t = null"), reason="no own effects in this mockup.js")
def test_a_layer_sharing_its_path_has_its_own_overlay(tmp_path):
    """Twin Emperors' 5 Ability layers share a path whose effects are a label's Gradient Overlay: a layer's own
    effects (psd_details VERSION 12) answer for it, else the path's; a group's overlay still counts."""
    out = node(tmp_path, """const at = "Text/Ability", rec = { effects: { "Text": ["Stroke"], [at]: ["GradientOverlay"] } };
out([m.oneColour(rec, at, undefined, { layer: at, effects: [] }), m.oneColour(rec, at, undefined, { layer: at, effects: ["GradientOverlay"] }),
  m.oneColour(rec, at, undefined, { layer: at }), m.oneColour(rec, at), m.oneColour({ effects: { "Text": ["ColorOverlay"] } }, at, undefined, { layer: at, effects: [] })]);""")
    assert out == [False, True, True, True, True]


@pytest.mark.skipif(not has(MOCKUP_JS, "const iconEm"), reason="no icon glyph shapes in this mockup.js")
def test_an_icon_swap_keeps_the_line_at_the_glyphs_shape(tmp_path):
    """A typed icon steps as its stoogedings glyph (its picture's width over height, 1 em tall or 1.4 wide, 0.04 em each
    side); one icon for another is spliced alone at the old icon's size (its pen step, or its ink where tracking spaces
    it out), the rest of the line moved by the glyphs' difference."""
    out = node(tmp_path, """
const icons = { machine: { file: "m.webp", a: 1 }, field: { file: "f.webp", a: 0.5 }, wide: { file: "w.webp", a: 2 }, odd: { file: "o.webp" } };
const measure = t => [...t].length * 10; measure.hx = 1;
const widths = m.layout(m.tokenize("[field] [machine] [wide] [odd]", icons), Infinity, 10, measure)[0].pieces.filter(p => p.icon).map(p => +p.w.toFixed(2));
const text = "Deal [machine] now.", g = [...text].map((ch, i) => { const x = i < 5 ? 10 + i * 10 : 114 + (i - 14) * 10; return ch === " " ? [i, x] : [i, x, x + 1, x + 9]; });
const c = ink => ({ text, lines: [[0, 19, 100, 10, 164, 60, 105, 0]], glyphs: [...g.slice(0, 5), [5, 60, ...ink], ...g.slice(14)], frame: [0, 300] });   // the icon 54 px
const fit = ink => { const R = m.recordLayout(c(ink), text, 1), f = m.spliceFit(R, m.geoPlan(R, "Deal [field] now.").splice, { measure, z: 20, icons });
  return f && [f.tag.at, Math.round(f.tag.w), Math.round(f.ds), Math.round(f.tag.ez), f.tag.left]; };
out([widths, fit([62, 112]), fit([67, 107])]);""")
    assert out == [[5.8, 10.8, 14.8, 11], [5, 29, -25, 50, False], [5, 34, -20, 40, True]]


@pytest.mark.skipif(not has(MOCKUP_JS, "VARIANT_PAD"), reason="no icon variants in this mockup.js")
def test_a_typed_icon_variant_steps_as_its_glyph(tmp_path):
    """[x:glow] steps as the font's glyph with its halo spacers (0.26 em each side), [x:sticker] 0.03, [x:tint] and
    [x:no] as the plain icon, each with its picture; a layered one typed on a tracked line is set untracked (psd_edit),
    and an icon the plate spells with a variant steps as that one."""
    out = node(tmp_path, """
const icons = { machine: { file: "m.1.webp", a: 1, v: ["tint", "sticker", "glow", "darkglow"] } };
const measure = t => [...t].length * 10; measure.hx = 1;
const ps = m.tokenize("[machine] [machine:glow] [machine:sticker] [machine:tint] [machine:no]", icons)[0].map(c => c[0]);
const widths = m.layout([ps.map(p => [p])], Infinity, 10, measure)[0].pieces.filter(p => p.icon).map(p => +p.w.toFixed(2));
const text = "Deal [machine] now.", g = [...text].map((ch, i) => { const x = i < 5 ? 10 + i * 10 : 114 + (i - 14) * 10; return ch === " " ? [i, x] : [i, x, x + 1, x + 9]; });
const c = (t, ink) => ({ text: t, lines: [[0, t.length, 100, 10, 164, 60, 105, 0]], glyphs: [...g.slice(0, 5), [5, 60, ...ink], ...g.slice(14).map(([i, ...r]) => [i + t.length - text.length, ...r])], frame: [0, 300] });
const fit = (t, ink, now, z) => { const R = m.recordLayout(c(t, ink), text, 1), f = m.spliceFit(R, m.geoPlan(R, now).splice, { measure, z, icons }); return f && [Math.round(f.tag.w), Math.round(f.ds), Math.round(f.tag.ez)]; };
out([ps.map(p => p.vary || (p.no ? "no" : "")), ps[1].vfile, widths, fit(text, [62, 112], "Deal [machine:glow] now.", 50),
  fit(text, [67, 107], "Deal [machine:glow] now.", 40), fit(text, [67, 107], "Deal [machine:tint] now.", 40),
  fit("Deal [machine:sticker] now.", [60, 114], "Deal [machine:glow] now.", 47)]);""")
    assert out == [["", "glow", "sticker", "tint", "no"], "m.1.glow.webp", [10.8, 16, 11.4, 10.8, 10.8], [80, 26, 50], [64, 10, 40],
                   [54, 0, 40], [76, 22, 47]]


@pytest.mark.skipif(not has(MOCKUP_JS, "back[v.at(j)]"), reason="no variant runs in this mockup.js's tintOld")
def test_a_colour_over_a_variant_ability_tints_in_place(tmp_path):
    """Afraid's record spells [dark:sticker], the website [dark]: a colour on its Ability tints the letters in place, each
    letter's old fill read from the record's runs at its own index (after the icon too), not a retyped Ability."""
    out = node(tmp_path, """
const rec = "Can't affect [dark:sticker] creatures.", old = "Can't affect [dark] creatures.", W = 10 * rec.length + 20, H = 30;
const layer = { text: rec, color: "#ffffff", colors: ["#ffffff", "#2050ff"], runs: [[0, 27, 0], [27, rec.length, 1]] };
const fill = j => (j < 27 ? [255, 255, 255] : [32, 80, 255]), data = new Uint8ClampedArray(W * H * 4);
const glyphs = [...rec].map((ch, j) => [j, 10 + j * 10, 11 + j * 10, 19 + j * 10]);
for (const [j, , a, b] of glyphs) if (rec[j] !== " ") for (let y = 5; y < 25; y++) for (let x = a; x < b; x++) data.set([...fill(j), 255], (y * W + x) * 4);
const plate = { crops: [{ id: 1, box: [0, 0, W, H], text: rec, lines: [[0, rec.length, 20, 10, W - 10, 5, 25, 0]], glyphs }] };
const plan = phrase => m.tintPlan(data, W, plate, [1], 1, old, layer, [{ field: "ability", phrase, color: "#ff2020" }], "ability");
const own = (p, w) => p && [...w].map((_, k) => p.own(old.indexOf(w) + k).join(","));
out([own(plan("Can't"), "Can't"), own(plan("creatures"), "creatures")]);""")
    assert out == [["255,255,255"] * 5, ["32,80,255"] * 9]


@pytest.mark.skipif(not has(MOCKUP_JS, "function mixPlan"), reason="no mixPlan in this mockup.js")
def test_a_whole_field_colour_mixes_the_bakes_looks(tmp_path):
    """A plate's tints (card_plates: the bake's looks of a layer in the 9 probe colours): a whole-field colour mixes them
    where they move (a corner probe: its look, grey: the grey look; luma: through black, grey and white), the rest keeps
    the card; a phrase, another field's colour or looks off the card's pixels: none."""
    out = node(tmp_path, """
const W = 60, H = 1, card = new Uint8ClampedArray(W * H * 4).fill(40), lin = c => c.map(v => 200 * v + 20);
const look = (n, px) => { const a = card.slice(); for (const x of [0, 1, 2]) a.set([...px, 255], x * 4); return a; };
const P = m.PROBES.map(h => [1, 3, 5].map(i => parseInt(h.slice(i, i + 2), 16) / 255));
const looks = P.map((c, n) => look(n, n === 8 ? lin(c).map((v, e) => v + [30, 0, -30][e]) : lin(c)));
const plate = luma => ({ tints: [{ id: 7, box: [0, 0, W, H], at: looks.map((_, n) => [0, n, W, H]), ...(luma ? { luma } : {}) }] });
const read = ({ at }) => looks[at[1]], data = card.slice();
const mix = (color, luma = false, f = "name", phrase) => { const d = data.slice(), p = m.mixPlan(d, W, H, 1, plate(luma), [7], [{ field: f, color, ...(phrase ? { phrase } : {}) }], "name", read);
  if (p) m.paintMix(d, W, p); return p && [...d.slice(0, 16)]; };
const off = card.slice().fill(90);
out([mix("#ff0000"), mix("#808080"), mix("#ffffff", true), mix("#ff0000", false, "name", "Claw"), mix("#ff0000", false, "ability"),
  m.mixPlan(off, W, H, 1, plate(false), [7], [{ field: "name", color: "#ff0000" }], "name", read)]);""")
    red, grey, white, phrase, other, stale = out
    assert red == [220, 20, 20, 40] * 3 + [40] * 4 and grey == [150, 120, 90, 40] * 3 + [40] * 4 and white == [220, 220, 220, 40] * 3 + [40] * 4
    assert phrase is None and other is None and stale is None


@pytest.mark.skipif(not has(MOCKUP_JS, "function firstBase"), reason="no firstBase in this mockup.js")
def test_a_no_layout_ability_starts_on_its_own_first_baseline(tmp_path):
    """An Ability whose plate has no layout, retyped with the same words: line 1 on the record's first baseline, its ink
    top plus the line's tallest piece (capitals, a badge's pill, an icon's glyph); scaled from the frame's top if shrunk."""
    out = node(tmp_path, """
const ctx = { font: "", measureText(t) { const z = parseFloat(this.font); return { actualBoundingBoxAscent: (/[A-Z]/.test(t) ? 0.5 : 0.36) * z, actualBoundingBoxDescent: 0 }; } };
const icons = { wide: { file: "w.webp", a: 2 } }, measure = (t, z) => [...t].length * z * 0.5;
const at = (old, o = {}) => m.firstBase(ctx, m.layout(m.tokenize(old, icons), 400, 40, measure)[0], { top: 100, s0: 40, z: 40, frame: null, fam: "x", ...o });
out([at("After attacking, destroy this creature."), at("heal 2 now"), at("(1CD) heal 2"), at("[wide] heal"), at(""),
  at("After attacking.", { z: 30, frame: [0, 80, 400, 300] })].map(v => v && Math.round(v * 10) / 10));""")
    assert out == [120, 114.4, 123.5, 126.8, None, 110]


@pytest.mark.skipif(not has(MOCKUP_JS, "function ownWrap"), reason="no ownWrap in this mockup.js")
def test_a_rewrapped_ability_moves_the_cards_own_words(tmp_path):
    """ownWrap: a shorter or deleted word pulls the next line's first word up where the bake's greedy wrap puts it; kept
    words are the card's ink moved (cut half-way to their neighbours'), only the typed word drawn. A frame that does not
    give the card's own lines, a break within 1 px of a typed word's, or a blank paragraph (wrap() drops one at the
    changed range's ends): null (the old way)."""
    out = node(tmp_path, """
const card = (was, spans) => { const x = new Map(), ink = new Map();
  const lines = spans.map(([s, e, base, para]) => { for (let i = s; i < e; i++) { x.set(i, 5 * (i - s)); if (was[i] !== " ") ink.set(i, [5 * (i - s), 5 * (i - s) + 4]); }
    return { s, e, base, left: 0, right: 5 * (e - s), top: base - 8, bottom: base + 2, just: 0, para }; });
  return { was, lines, x, ink }; };
const one = card("aaa bbb ccc ddd eee", [[0, 11, 20, 0], [12, 19, 32, 0]]), two = card("aaa bbb ccc ddd eee\\nfff", [[0, 11, 20, 0], [12, 19, 32, 0], [20, 23, 44, 1]]);
const run = (c, now, frame = [0, 68]) => { const R = { ...c, frame }, ow = m.ownWrap(R, m.geoPlan(R, now), now, m.tokenize(now, {}),
    { measure: (s, z) => s.length * z * 0.5, z: 10, fx: [], icons: {}, step: 12, B: [[10, 23], [23, 36], [36, 49]], x0: 0, x1: 100 });
  return ow && [ow.dy, ow.moves.map(([r, dx, dy]) => [r[0], r[2], dx, dy]), ow.paints.map(([l, o]) => [l[0].pieces.map(p => p.text).join(""), o.at(), o.base])]; };
out([run(one, "aaa q ccc ddd eee"), run(one, "aaa ccc ddd eee", [0, 62]), run(one, "aaa q ccc ddd eee", [0, 66]), run(one, "aaa q ccc ddd eee", [0, 100]),
  run(two, "aaa bbb ccc ddd eee\\nfgf"), run(two, "aaa q ccc ddd eee\\n\\nfff"), run(two, "aaa bbb ccc ddd eee\\n\\nfgf"), run(one, "aaa q ccc ddd eee\\n\\nfff")]);""")
    assert out[0] == [0, [[0, 17, 0, 0], [37, 59, -10, 0], [0, 17, 50, -12], [17, 39, -20, 0]], [["q", 20, 20]]]
    assert out[1] == [0, [[0, 17, 0, 0], [37, 59, -20, 0], [0, 17, 40, -12], [17, 39, -20, 0]], []]
    assert out[2] is None and out[3] is None
    assert out[4] == [0, [], [["fgf", 0, 44]]]
    assert out[5:] == [None, None, None]   # a blank line after, before, inside the change


@pytest.mark.skipif(not has(MOCKUP_JS, "function geoRoom"), reason="no geoRoom in this mockup.js")
def test_a_growing_ability_measures_its_room_on_the_cards_lines(tmp_path):
    """geoRoom: the new lines on the card's own baselines (the first gone line's, its step), the lines after moved down by
    the added lines, the floor the inked lines' lowest bottom; a blank line is no ink (low: the lowest inked baseline)."""
    out = node(tmp_path, """
const was = "Hit a foe.\\nHeal 2.\\n", now = "Hit a foe.\\nHeal 2. Gain 1 DEF.\\n", line = (s, e, para, base, ink) => ({ s, e, para, base, top: base - ink, bottom: base + (ink && 5), left: 0, right: 60, just: 0 });
const R = { was, lines: [line(0, 10, 0, 25, 20), line(11, 18, 1, 65, 20), line(19, 19, -1, 105, 0)], x: new Map([...was].map((_, i) => [i, i])), frame: [0, 60] };
const G = m.geoRoom(R, m.geoPlan(R, now), m.tokenize(now, {}), now, 10, (s, z) => s.length * z * 0.5, 40, {});
out([G.floor, G.step, G.paint.map(p => [p.base, p.lines.length]), G.low, G.deep]);""")
    assert out == [70, 40, [[65, 2], [145, 1]], 105, 145]


@pytest.mark.skipif(not has(MOCKUP_JS, "function geoRoom"), reason="no geoRoom in this mockup.js")
def test_a_plates_emoji_counts_as_two_letters(tmp_path):
    """The plate's indexes are code points (Python's); the website's text counts an emoji as 2: the lines after it and
    its letters' pens still start where they do (Cave of the Past's clock)."""
    out = node(tmp_path, """
const c = { text: "A \\u{1F550} clock.\\rNext line.", lines: [[0, 10, 20, 0, 50, 5, 25, 0], [11, 21, 60, 0, 50, 45, 65, 0]], glyphs: [[0, 0], [2, 5], [4, 12], [11, 0]] };
const R = m.recordLayout(c, "A \\u{1F550} clock.\\nNext line.", 1);
out([R.lines.map(l => [l.s, l.e, l.para]), [...R.x.keys()]]);""")
    assert out == [[[0, 11, 0], [12, 22, 1]], [0, 2, 5, 12]]


def test_an_added_ability_goes_where_the_bake_puts_it(tmp_path):
    """card_checks' room as the site's details addRoom (owner 2026-11-13): "+ability" drawn in its box, not the template's
    over the bars; none on a card with no room (the bake refuses it); no fact yet: the template's box."""
    out = node(tmp_path, """
const rec = room => ({ parts: { title: [], ability: [] }, partsTemplate: { ability: { box: [405, 1778, 793, 1856] } }, ...(room === undefined ? {} : { addRoom: room }) });
const jobs = room => m.partJobs(rec(room), { "+ability": { text: "Deal 3 damage." } }).map(j => [j.id, j.box]);
out([jobs([30, 2210, 1170, 2388]), jobs(false), jobs(undefined)]);""")
    assert out == [[["+ability", [30, 2210, 1170, 2388]]], [], [["+ability", [405, 1778, 793, 1856]]]]


@pytest.mark.skipif(not has(MOCKUP_JS, "export function putCover"), reason="no covers in this mockup.js")
def test_a_picture_over_the_text_stays_over_the_new_words(tmp_path):
    """A plate's cover (card_plates.cover_alpha: Papyrus's spaghetti): new letters under it are mixed back to the plate
    by its alpha, as the bake leaves the picture over them, and said past the old lines' ink; the card's own pixels stay;
    moved words never take the picture with them (moveInk)."""
    out = node(tmp_path, """
const W = 10, H = 2, px = (a, n, c) => { for (let i = 0; i < n; i++) a.set(c, i * 4); return a; };
const card = px(new Uint8ClampedArray(W * H * 4), W * H, [0, 0, 0, 255]);
for (const x of [6, 7]) card.set([200, 0, 0, 255], x * 4);                          // the picture, on row 0
const plate = card.slice(), alpha = px(new Uint8ClampedArray(W * H * 4), W * H, [0, 0, 0, 255]);
for (const x of [6, 7]) alpha.set([255, 255, 255, 255], x * 4);
const cover = m.coverCuts({ crops: [{ id: 3, box: [0, 0, W, H] }], covers: [{ id: 3, box: [0, 0, W, H] }] }, () => plate, () => alpha, [3], 1, W, H);
const drawn = card.slice(); for (const x of [4, 5, 6, 7, 8]) drawn.set([255, 255, 255, 255], x * 4);   // new words across it
const ctx = buf => ({ getImageData: (x, y, w, h) => { const d = new Uint8ClampedArray(w * h * 4); for (let r = 0; r < h; r++) d.set(buf.subarray(((y + r) * W + x) * 4, ((y + r) * W + x + w) * 4), r * w * 4); return { data: d }; },
  putImageData: (im, x, y) => { const w = im.data.length / 4 / H; for (let r = 0; r < H; r++) buf.set(im.data.subarray(r * w * 4, (r + 1) * w * 4), ((y + r) * W + x) * 4); } });
const a = drawn.slice(), b = drawn.slice(), row = buf => [...Array(W).keys()].map(x => buf[x * 4]);
const said = [m.putCover(ctx(a), W, cover, card, [[0, 0, 4, 2]], 0), m.putCover(ctx(b), W, cover, card, [[0, 0, W, 2]], 0)];
const wiped = card.slice(); for (const x of [6, 7]) wiped.set([150, 0, 0, 255], x * 4);   // the plate's picture, a little off the card's
const moved = c => { const o = wiped.slice(0, W * 4); m.moveInk(o, wiped.slice(0, W * 4), card, W, [0, 1], [0, 0, 8, 1], 2, 0, c); return row(o); };
out([row(a), said, moved(cover), moved(null)]);""")
    assert out[0] == [0, 0, 0, 0, 255, 255, 200, 200, 255, 0] and out[1] == [True, False]
    assert out[2] == [0, 0, 0, 0, 0, 0, 150, 150, 0, 0] and min(out[3][8:]) > 100   # without it: a ghost of the picture


@pytest.mark.skipif(not has(MOCKUP_JS, "export function stackRows"), reason="no stacked rows in this mockup.js")
def test_a_stacked_box_knows_its_rows(tmp_path):
    """A stacked Ability box (bake fit_below stacked) is checked against its sibling rows (one layer path), grown by
    their Stroke and shadows; a lone layer or another creature's (Tainted Lazarus) has none. An indented first line
    (Miracle Matter's spaces) wraps in the room its spaces leave."""
    out = node(tmp_path, """
const part = (id, layer, box) => ({ id, layer, box });
const rec = { parts: { title: [], ability: [part(1, "Text/Ability", [100, 100, 500, 140]), part(2, "Text/Ability", [100, 190, 500, 230]), part(3, "Laz/Ability", [0, 0, 50, 50])] },
  text: [{ id: 1, stroke: [0, 0, 0, 2], fx: [{ k: "shadow", dx: 3, dy: 8 }] }] };
const lay = indent => m.layout(m.tokenize("aaaa bbbb cccc"), 10, 1, (s, z) => s.length * z, indent).map(l => l.w);
out([m.stackRows(rec, 2, 0.5), m.stackRows(rec, 3, 0.5), m.stackRows({ parts: { ability: [part(1, "Text/Ability", [0, 0, 9, 9])] } }, 1, 1), lay(0), lay(3)]);""")
    assert out == [{"own": [50, 95, 250, 115], "others": [[49, 49, 251.5, 74]]}, None, None, [9, 4], [4, 9]]


def test_plates_port_of_the_rows(tmp_path):
    """card_plates.mock_misses at the mock-up's own thresholds misses what draw() misses (a bar on the template's row, one
    at its label, art, a drawn bar showing another value), and with its margin at least that."""
    from types import SimpleNamespace
    import numpy as np
    from PIL import Image
    import commands.card_plates as cp
    W, H = 360, 720
    img = np.repeat(np.linspace(30, 220, W, dtype=np.uint8)[None, :, None], H, 0).repeat(3, 2)   # art: a smooth sweep
    (b0, b1, b2, b3), step = cp.ROWS["tall"]
    def bar(y0, y1, lit):
        x0, cw = round(b0 * W), (b2 - b0) * W / 10
        for x in range(x0, round(b2 * W)):
            c = int((x - x0) // cw)
            img[y0:y1, x] = 20 if (x - x0) % cw < 2 else (200 if c < lit else 90)
    rows = [(round((b1 + n * step) * H), round((b3 + n * step) * H)) for n in range(4)]
    bar(*rows[0], 4)                                        # hp on its row
    bar(rows[1][0] + 15, rows[1][1] + 15, 2)                # def at its label, below its row
    bar(*rows[3], 6)                                        # spd shows 6, not its 3; atk: art
    raw = tmp_path / "img.raw"
    raw.write_bytes(np.dstack([img, np.full((H, W), 255, np.uint8)]).tobytes())
    label = SimpleNamespace(kind="type", text="Defense", bbox=(50, 2 * (rows[1][0] + 15), 300, 2 * (rows[1][1] + 15)),
                            is_visible=lambda: True)
    psd = SimpleNamespace(width=720, descendants=lambda: iter([label]))
    card, kinds = {"hp": 4, "def": 2, "atk": 5, "spd": 3}, {"hp": "template", "def": "template", "atk": "template", "spd": "pixel"}
    rec = {"canvas": {"w": 720}, "stats": kinds, "text": [{"text": "Defense", "visible": True, "box": list(label.bbox)}]}
    js = node(tmp_path, f"""
const fs = await import("node:fs"), W = {W}, H = {H}, d = new Uint8ClampedArray(fs.readFileSync({json.dumps(str(raw))})), L = m.luma(d);
const rec = {json.dumps(rec)}, card = {json.dumps(card)};
const bars = m.barRows("tall", W, H, card, rec).map(r => (m.isBar(L, W, H, r.box) ? r : m.labelRow(L, W, H, rec, r)))
  .filter(r => r && !(rec.stats[r.k] === "pixel" && !m.rowShows(L, W, H, r, card[r.k])));
out(["hp", "def", "atk", "spd"].filter(k => !bars.some(r => r.k === k)));""")
    image = Image.fromarray(img)
    assert js == ["atk", "spd"] and cp.mock_misses(image, psd, card, kinds, 1.0) == js
    assert set(js) <= set(cp.mock_misses(image, psd, card, kinds))


@pytest.mark.skipif(not has(MOCKUP_JS, "export function followSpecs"), reason="no followSpecs in this mockup.js")
def test_a_retyped_title_rewrites_its_copies_elsewhere(tmp_path):
    """card_plates' crops of follow layers: a copy takes the title's new text, a watermark the joined title; none for a
    copy under a layer the draw wipes too, nor one without a crop (never shown); a deleted title: erased only."""
    out = node(tmp_path, """
const rec = { parts: { title: [{ id: 1, text: "Mosling", layer: "Name" }], ability: [{ id: 2, text: "Hit.", layer: "Ability" }],
  follow: [{ id: 3, of: [1], how: "copy" }, { id: 4, of: [1], how: "watermark" }, { id: 5, of: [1], how: "copy" }] },
  text: [[3, "Art/Name", [500, 1800, 1100, 2300]], [4, "Mark", [400, 1000, 800, 1100]], [5, "Hid", [0, 0, 9, 9]]].map(([id, layer, box]) => ({ id, layer, box, text: "Mosling" })) };
const crop = (id, box, ink, more = {}) => ({ id, box, ink, ...more });
const plate = { crops: [crop(1, [300, 0, 900, 200], [350, 20, 850, 180]), crop(2, [0, 1700, 1200, 2100], [50, 1750, 1150, 2050]),
  crop(3, [480, 1780, 1120, 2320], [500, 1800, 1100, 2300], { over: [2] }), crop(4, [380, 980, 820, 1120], [400, 1000, 800, 1100])] };
const spec = (id, text, old) => ({ part: true, id: String(id), text, old, role: id === 2 ? "ability" : "name", label: id === 2 ? "text" : "name" });
const got = specs => m.followSpecs(rec, plate, specs, 0.6).map(s => [s.id, s.text, s.own]);
out([got([spec(1, "Moslings", "Mosling")]), got([spec(1, "Moslings", "Mosling"), spec(2, "Hit twice.", "Hit.")]), got([spec(1, null, "Mosling")])]);""")
    assert out == [[["3", "Moslings", [300, 1080, 660, 1380]], ["4", "Moslings", [240, 600, 480, 660]]],
                   [["4", "Moslings", [240, 600, 480, 660]]], [["3", None, [300, 1080, 660, 1380]], ["4", None, [240, 600, 480, 660]]]]


@pytest.mark.skipif(not has(MOCKUP_JS, "export function splitWhy"), reason="no splitWhy in this mockup.js")
def test_a_split_titles_part_the_bake_refuses_is_said(tmp_path):
    """psd_edit.split_title_check's refusals, said as the bake words them: off the card (The Bagel), onto another line
    (a box), into another part; a part that fits, titles in separate pictures, or a near miss (a letter past a box or its
    frame, a part above or below: Leeroy Jenkins, The Big Yellow Cat, which bake) say nothing."""
    out = node(tmp_path, """
const ctx = { font: "", letterSpacing: "0px", measureText(t) { return { width: t.length * parseFloat(this.font) / 2 }; } };
const part = (id, text, box, variants = null) => ({ id, text, box, variants });
const fams = layers => layers.map(layer => ({ fam: "x", layer }));
const why = (title, specs, layers) => m.splitWhy(ctx, { parts: { title } }, specs.map(([id, text, old]) => ({ role: "name", part: true, id: String(id), text, old })), fams(layers), 0.6, 720);
const bagel = [part(1, "The Bagel", [268, 49, 954, 174]), part(2, "Or... is it", [537, 16, 643, 36])];
const side = [part(1, "Azamoth", [100, 100, 500, 200]), part(2, "King", [520, 100, 900, 200])];
out([why(bagel, [[1, "The Bagel Or... is it", "The Bagel"]], [{ px: 145 }]), why(bagel, [[1, "The Bagels", "The Bagel"]], [{ px: 145 }]),
  why(side, [[1, "Azamoth Azamoth", "Azamoth"]], [{ px: 100, frame: [100, 0, 600, 0] }]), why(side, [[1, "Azamoth The", "Azamoth"]], [{ px: 100, just: 0 }]),
  why([part(1, "Stubborn King", [100, 100, 500, 200], [1]), part(2, "Stubborn Queen", [520, 100, 900, 200], [2])], [[1, "Stubborn King Prime", "Stubborn King"]], [{ px: 100, just: 0 }]),
  why([side[0], part(2, "King", [540, 100, 900, 200])], [[1, "Azamoths", "Azamoth"]], [{ px: 100, just: 0 }]), why(side, [[1, "Azamoths", "Azamoth"]], [{ px: 100, frame: [100, 0, 540, 0] }]),
  why([side[0], part(2, "King", [520, 160, 900, 300])], [[1, "Azamoth The", "Azamoth"]], [{ px: 100, just: 0 }])]);""")
    off, fits, wraps, into, apart, *near = out
    assert off == "title part 1/2 ('The Bagel Or... is it') would run off the card: shorten it, or change the title in Photoshop"
    assert wraps.endswith("would break onto another line: shorten it, or change the title in Photoshop")
    assert into.startswith("title part 1/2 ('Azamoth The') would run into 'King', another part of the title")
    assert fits == apart == "" and near == ["", "", ""]


@pytest.mark.skipif(not has(MOCKUP_JS, "export function titleFit"), reason="no titleFit in this mockup.js")
def test_a_title_over_an_icon_is_set_smaller(tmp_path):
    """psd_edit fit_title in the mock-up: an icon near the old title is left out (titleIcons); the title steps by 0.94
    to the first scale that clears, never under TITLE_MIN, at 1 when it reaches none or none clears; a title a bake set
    smaller leaves out only icons near the card maker's boxes (keptTitle), as the next bake."""
    out = node(tmp_path, """
const head = { stars: { units: [{ core: [900, 40, 960, 100] }, { core: [600, 40, 660, 100] }] }, types: { row: [{ name: "fire", core: [100, 40, 160, 100] }] } };
const at = need => m.titleFit(s => s > need), was = [[300, 20, 400, 60]];
out([m.titleIcons(head, 0.5, was), at(0.85), at(0.5), at(2), m.keptTitle({ titleFit: 0.88, titleWas: was }),
  m.keptTitle({ titleFit: 1, titleWas: was }), m.keptTitle({ titleFit: 0.88 })]);""")
    assert out[0] == [["star", [450, 20, 480, 50]], ["fire", [50, 20, 80, 50]]]
    assert out[1] == pytest.approx(0.94 ** 3) and out[2] == out[3] == 1
    assert out[4] == [[300, 20, 400, 60]] and out[5] is None and out[6] is None


@pytest.mark.skipif(not has(MOCKUP_JS, "export const keptFit"), reason="no keptFit in this mockup.js")
def test_a_split_titles_part_keeps_its_own_fit(tmp_path):
    """r4-2: a split title's part a bake set smaller starts from the card maker's size and boxes (titleFits), as the
    next bake; any other title the card's titleFit."""
    out = node(tmp_path, """
const was = [[300, 20, 400, 60]], card = { titleFit: 0.9, titleWas: [[1, 2, 3, 4]], titleFits: { 422: [0.78, was] } };
out([m.keptTitle(card, 422), m.keptFit(card, 422), m.keptTitle(card, 7), m.keptFit(card, 7), m.keptTitle(card)]);""")
    assert out == [[[300, 20, 400, 60]], 0.78, [[1, 2, 3, 4]], 0.9, [[1, 2, 3, 4]]]


@pytest.mark.skipif(not has(MOCKUP_JS, "export function fitPart"), reason="no fitPart in this mockup.js")
def test_a_type_icon_fills_its_spot_by_its_opaque_part(tmp_path):
    """A type icon with a glow past its opaque part (light, support): the mock-up draws the file where the bake's
    _icon_image links it, the opaque part filling the spot, not the whole file fitted into it."""
    import numpy as np
    from PIL import Image
    from commands.psd_edit import _icon_image
    a = np.zeros((96, 80), np.uint8)
    a[10:86, 20:62] = 60                                    # the glow
    a[24:70, 28:54] = 255
    rgba = np.dstack([np.full_like(a, 200)] * 3 + [a])
    Image.fromarray(rgba, "RGBA").save(tmp_path / "light.png")
    (tmp_path / "light.raw").write_bytes(rgba.tobytes())
    box = (100, 200, 140, 236)
    quad = _icon_image(str(tmp_path / "light.png"), box, whole=True)[3]
    x, y, w, h = node(tmp_path, f"""const d = new Uint8ClampedArray((await import("node:fs")).readFileSync({json.dumps(str(tmp_path / "light.raw"))}));
out(m.fitPart(80, 96, m.opaquePart(d, 80, 96), {json.dumps(box)}).slice(0, 4));""")
    assert max(abs(p - q) for p, q in zip((x, y, x + w, y + h), (quad[0], quad[1], quad[4], quad[5]))) <= 1


@pytest.mark.skipif(not has(MOCKUP_JS, "export const capBox"), reason="no capBox in this mockup.js")
def test_an_icons_erase_box_stays_near_the_icon(tmp_path):
    """A record's star or type box past half the icon (Holo Defense API, Ice Dragon: the whole card) is erased only
    that far; a real glow box is kept as it is."""
    out = node(tmp_path, """
const star = (core, box) => ({ core, box }), at = { 1: [[1120, 60]] };
const stars = { units: [star([1063, 9, 1178, 111], [1063, 8, 1187, 120]), star([949, 9, 1064, 111], [12, -98, 1214, 2388])], at };
const types = { row: [{ name: "defensive_minion", core: [-5, 20, 153, 130], box: [-5, 20, 153, 130] }, { name: "ice", core: [168, 8, 289, 143], box: [26, -2, 1228, 2400] }] };
out([m.starPlan(stars, 2, 1).erase, m.starPlan(stars, 2, 0).erase, m.typePlan(types, ["defensive_minion"]).erase]);""")
    assert out == [[[891.5, -48.5, 1121.5, 168.5]], [[1063, 8, 1187, 120], [891.5, -48.5, 1121.5, 168.5]], [[100.5, -2, 356.5, 210.5]]]


@pytest.mark.skipif(not has(MOCKUP_JS, "export function creatureSite"), reason="no creatureSite in this mockup.js")
def test_creatures_the_bake_leaves_on_the_website_are_said(tmp_path):
    """Split, + or - Creature and creature 2's stat emptied leave the list on the website (psd_edit._classify_creatures):
    said; a creature's stat edit (the plate's picture), creature 1's emptied stat and no list say nothing."""
    out = node(tmp_path, """
const a = { hp: 7, def: 4, atk: 5, spd: 5 }, b = { hp: 5, def: 4, atk: 6, spd: 4 }, psd = { creatures: [a, b] };
const say = (edit, was = psd) => p.mockNote({ site: [["The creature list", m.creatureSite(edit, was)]].filter(x => x[1]) });
out([say({ creatures: [a, { ...a }] }, { creatures: null }), say({ creatures: [a, b, b] }), say({ creatures: null }),
  say({ creatures: [a, { ...b, hp: null, atk: null }] }), say({ creatures: [a, { ...b, hp: 6 }] }), say({ creatures: [{ ...a, hp: null }, b] }),
  say({}, { creatures: null })]);""")
    assert out[:3] == ["The creature list shows on the website only: each creature's stats print only on a bar per creature."] * 3
    assert out[3] == "The creature list shows on the website only: a bar can't show no value (creature 2's HP)."
    assert out[4:] == ["", "", ""]


@pytest.mark.skipif(not has(ASSETS / "parts.js", '"a card shows at least one"'), reason="no under-1 stars rule in this parts.js")
def test_stars_under_one_are_the_bakes_refusal(tmp_path):
    """Stars to 0: the bake refuses (psd_edit: 'a card shows at least one'), so the mock-up keeps the printed stars and
    says so, and a folder card promises no move to '0 Stars/'. A buff (no stars: psd_details' '*') keeps its own
    reason, checked first in _plan_stars; any other '*' reason still gives way at 0."""
    why, buff = "a card shows at least one", "the parser reads no stars on this card"
    src = (Path(__file__).parents[1] / "commands" / "psd_edit.py").read_text(encoding="utf-8")
    assert f'"{why}"' in src and f'"{buff}"' in src
    out = node(tmp_path, f"""const P = await import({json.dumps((ASSETS / "parts.js").as_uri())});
const rec = {{ units: [{{}}], at: {{ 1: [[0, 0]] }}, refused: {{}} }}, path = "Creatures/Kirby/2 Stars/Acro.psd";
const vigor = {{ units: [], at: {{}}, refused: {{ "*": {json.dumps(buff)} }} }}, odd = {{ units: [], at: {{}}, refused: {{ "*": 'no group named "Stars" draws them' }} }};
out([P.starsKept(rec, 0), P.starsKept(rec, 1), P.starMove(path, 0), P.starMove(path, 3)?.to, p.mockNote({{ refused: [["stars", P.starsKept(null, 0)]] }}),
  P.starsKept(vigor, 0), p.mockNote({{ refused: [["stars", P.starsKept(vigor, 0)]] }}), P.starsKept(odd, 0), !!P.starMove(path, 0, [], true)]);""")
    assert out == [why, None, None, "Creatures/Kirby/3 Stars/", f"The bake would refuse the stars: {why}.",
                   buff, "The bake would refuse the stars: this card has none.", why, True]


@pytest.mark.skipif(not has(MOCKUP_JS, "export function starSheet"), reason="no per-picture stars in this mockup.js")
def test_each_picture_takes_its_own_stars(tmp_path):
    """A PSD of several pictures (card_plates.with_stars): picture 1 the plate's "icons", picture n its "icons@<n>"; a
    plate without that picture's (an older one) none: the mock-up's starPlan."""
    out = node(tmp_path, """
const sheet = img => ({ vw: 720, bars: { stars: { at: 2, to: { 3: 0 } } }, cuts: [[1, 2, 3, 4, 0, 0]], img });
const plate = { v: 1, w: 1200, crops: [], icons: sheet("one"), "icons@2": sheet("two") };
out([1, 2, 3].map(n => m.starSheet(plate, n)?.img ?? null).concat([m.starSheet({ v: 1, icons: sheet("one") }, 2), m.starSheet(null, 1)]));""")
    assert out == ["one", "two", None, None, None]
