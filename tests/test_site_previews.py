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


def test_an_added_ability_goes_where_the_bake_puts_it(tmp_path):
    """card_checks' room as the site's details addRoom (owner 2026-11-13): "+ability" drawn in its box, not the template's
    over the bars; none on a card with no room (the bake refuses it); no fact yet: the template's box."""
    out = node(tmp_path, """
const rec = room => ({ parts: { title: [], ability: [] }, partsTemplate: { ability: { box: [405, 1778, 793, 1856] } }, ...(room === undefined ? {} : { addRoom: room }) });
const jobs = room => m.partJobs(rec(room), { "+ability": { text: "Deal 3 damage." } }).map(j => [j.id, j.box]);
out([jobs([30, 2210, 1170, 2388]), jobs(false), jobs(undefined)]);""")
    assert out == [[["+ability", [30, 2210, 1170, 2388]]], [], [["+ability", [405, 1778, 793, 1856]]]]


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
