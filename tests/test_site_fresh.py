"""
Site fresh.js (TTSCardMaker site/assets/fresh.js, in node): an open page looks for a new deploy every 15 s for 5 minutes
after it opens or its tab shows again, then every 3 minutes; a publish's 15 minutes are kept (2026-11-20).
"""
import json
from subprocess import run as run_process

from conftest import SITE_SRC

FRESH_JS = SITE_SRC / "site" / "assets" / "fresh.js"


def test_checks_every_15_s_after_opening_or_showing(tmp_path):
    script = tmp_path / "probe.mjs"
    script.write_text(f"""let now = 0, run = null, on = null;
const delays = [], flush = () => new Promise(r => setImmediate(r));
Date.now = () => now;
globalThis.setTimeout = (f, ms) => {{ run = f; delays.push(ms); return 1; }};
globalThis.clearTimeout = () => {{}};
globalThis.fetch = async () => ({{ ok: true, text: async () => "x\\nabc\\n" }});
globalThis.document = {{ visibilityState: "visible", querySelector: s => s === 'meta[name="page"]' ? {{ content: "abc" }} : null,
  addEventListener: (e, f) => {{ on = f; }} }};
const F = await import({json.dumps(FRESH_JS.as_uri())});
const at = async m => {{ now = m * 60e3; run(); await flush(); await flush(); return delays.at(-1); }};
const show = async (m, state) => {{ now = m * 60e3; document.visibilityState = state; on(); await flush(); return delays.at(-1); }};
F.soon(0); F.watch(() => {{}});                 // a publish waiting at load
const got = [delays.at(-1), await at(10), await at(16), await show(17, "hidden"), await show(20, "visible"), await at(24), await at(25.1)];
process.stdout.write(JSON.stringify(got));
""", encoding="utf-8")
    done = run_process(["node", str(script)], capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert done.returncode == 0, done.stderr
    # opened: 15 s; the publish's 15 min kept; then 3 min; hidden: nothing new; shown: 15 s for 5 min, then 3 min
    assert json.loads(done.stdout) == [15e3, 15e3, 180e3, 180e3, 15e3, 15e3, 180e3]
