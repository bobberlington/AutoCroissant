"""
The website's password sign-in (TTSCardMaker site/logins/<login>.json): build_site.py checks the files' shape (never
failing the build) and copies them to data/logins/<login in lowercase>.json; site/assets/login.js (in node) refuses the
same shapes, seals files the build accepts, opens them only with the right password, refuses weak passwords, and keeps
this browser's token sealed under a device key (over a fake IndexedDB). Loads both from $AC_SITE_SRC (a TTSCardMaker
checkout) or the clone; skips without it, or when that build_site.py predates read_logins (TTSCardMaker
work/site-password). The node checks skip without node.
"""
import base64
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
BUILD = SITE / "build" / "build_site.py"
LOGIN_JS = SITE / "assets" / "login.js"
B64 = lambda n, fill=b"\x01": base64.b64encode(fill * n).decode()   # noqa: E731


@pytest.fixture(scope="module")
def bs():
    if not BUILD.is_file():
        pytest.skip(f"no card repo build_site.py at {BUILD}")
    spec = importlib.util.spec_from_file_location("site_build_logins", BUILD)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "read_logins"):
        pytest.skip(f"{BUILD} has no password files: land TTSCardMaker work/site-password")
    return module


def good(login="MichaelJSr", **over):
    f = {"v": 1, "login": login, "kdf": "PBKDF2-SHA-256", "iterations": 2_000_000, "salt": B64(16), "iv": B64(12),
         "ct": B64(56), "at": "2026-10-23T01:45:00Z"}
    f.update(over)
    return {k: v for k, v in f.items() if v is not ...}


BAD = {                                           # name -> (file name stem, raw): each skipped
    "not an object": ("MichaelJSr", []),
    "v 2": ("MichaelJSr", good(v=2)),
    "v True": ("MichaelJSr", good(v=True)),
    "no login": ("MichaelJSr", good(login=...)),
    "login not a login": ("-x", good(login="-x")),
    "login another file's": ("MichaelJSr", good(login="Other")),
    "login in another case": ("MichaelJSr", good(login="michaeljsr")),
    "kdf": ("MichaelJSr", good(kdf="PBKDF2")),
    "too few iterations": ("MichaelJSr", good(iterations=999_999)),
    "too many iterations": ("MichaelJSr", good(iterations=10_000_001)),
    "iterations a float": ("MichaelJSr", good(iterations=2e6)),
    "iterations True": ("MichaelJSr", good(iterations=True)),
    "salt 15 bytes": ("MichaelJSr", good(salt=B64(15))),
    "iv 16 bytes": ("MichaelJSr", good(iv=B64(16))),
    "ct only a tag": ("MichaelJSr", good(ct=B64(16))),
    "ct over 1 KB": ("MichaelJSr", good(ct=B64(1025))),
    "ct not base64": ("MichaelJSr", good(ct="%%%")),
}


@pytest.mark.parametrize("name", list(BAD))
def test_clean_login_refuses(bs, name):
    stem, raw = BAD[name]
    entry, notes = bs.clean_login(raw, stem)
    assert entry is None and len(notes) == 1, (name, notes)


def test_clean_login_keeps_the_file_drops_extras(bs):
    entry, notes = bs.clean_login(good(), "MichaelJSr")
    assert entry == good() and not notes
    entry, notes = bs.clean_login(good(at="yesterday", extra=1), "MichaelJSr")
    assert entry == good(at=...) and len(notes) == 2           # a bad at is ignored, an unknown field noted


def test_read_and_write_never_fail(bs, tmp_path, monkeypatch, capsys):
    """Good files read, every bad one skipped with a warning line; data/logins/<lower>.json written."""
    monkeypatch.setattr(bs, "REPO", str(tmp_path))
    assert bs.read_logins() == []                                # no folder
    folder = tmp_path / bs.LOGINS_DIR
    folder.mkdir(parents=True)
    (folder / "MichaelJSr.json").write_text(json.dumps(good()), encoding="utf-8")
    (folder / "Bob.json").write_text(json.dumps(good("Bob", at=...)), encoding="utf-8")
    (folder / "Weak.json").write_text(json.dumps(good("Weak", iterations=1000)), encoding="utf-8")
    (folder / "Thief.json").write_text(json.dumps(good("Bob")), encoding="utf-8")
    (folder / "broken.json").write_text("{ nope", encoding="utf-8")
    (folder / "Big.json").write_text(json.dumps(good("Big", pad="x" * 5000)), encoding="utf-8")
    (folder / "README.md").write_text("# Passwords\n", encoding="utf-8")
    (folder / "dir.json").mkdir()
    logins = bs.read_logins()
    out = capsys.readouterr().out
    assert [e["login"] for e in logins] == ["Bob", "MichaelJSr"]
    for name in ("Weak.json", "Thief.json", "broken.json", "Big.json"):
        assert f"warning: {bs.LOGINS_DIR}/{name}" in out, name
    assert "README" not in out and "dir.json" not in out and "passwords: 2 from site/logins/" in out
    dist = tmp_path / "dist" / "data" / "logins"
    bs.write_logins(str(dist), logins)
    assert sorted(p.name for p in dist.iterdir()) == ["bob.json", "michaeljsr.json"]
    assert json.loads((dist / "michaeljsr.json").read_text(encoding="utf-8")) == good()
    bs.write_logins(str(tmp_path / "none"), [])
    assert not (tmp_path / "none").exists()


def node(tmp_path, body: str):
    """Run body after importing login.js as auth; its stdout as JSON."""
    if not (LOGIN_JS.is_file() and shutil.which("node")):
        pytest.skip(f"no login.js at {LOGIN_JS}, or no node")
    script = tmp_path / "probe.mjs"
    script.write_text(f"const auth = await import({json.dumps(LOGIN_JS.as_uri())});\n{body}\n", encoding="utf-8")
    return json.loads(run_process(["node", str(script)], capture_output=True, text=True, check=True).stdout)


def test_a_sealed_file_passes_the_build(bs, tmp_path):
    """login.js's own file (real PBKDF2 + AES-GCM) passes clean_login unchanged."""
    out = node(tmp_path, """
const f = { ...(await auth.seal(globalThis.crypto.subtle, "velvet trombone gecko sunrise", "MichaelJSr", "ghp_" + "x".repeat(36))), at: "2026-10-23T01:45:00Z" };
process.stdout.write(JSON.stringify({ text: auth.fileText(f), iter: auth.ITER, min: auth.MIN_ITER, max: auth.MAX_ITER }));
""")
    raw = json.loads(out["text"])
    entry, notes = bs.clean_login(raw, "MichaelJSr")
    assert entry == raw and not notes
    assert (out["min"], out["max"]) == bs.LOGIN_ITER and out["min"] <= out["iter"] >= 1_000_000


def test_login_js_refuses_what_the_build_refuses(bs, tmp_path):
    cases = {name: [stem, raw] for name, (stem, raw) in BAD.items() if name != "iterations a float"}   # 2e6 is an int in JSON
    out = node(tmp_path, f"""
const cases = {json.dumps(cases)};
process.stdout.write(JSON.stringify(Object.fromEntries(Object.entries(cases).map(([n, [stem, raw]]) => [n, !!auth.checkFile(raw, stem).file]))));
""")
    assert not any(out.values()), [n for n, ok in out.items() if ok]


def test_round_trip_wrong_password_and_tampering(tmp_path):
    """The right password opens the file (also typed in another Unicode form, the login in any case); a wrong one, a
    copy under another login, or a changed ct, tag, iv or salt doesn't; OpenSSL (node:crypto) reads the README's format."""
    out = node(tmp_path, """
const { pbkdf2Sync, createDecipheriv } = await import("node:crypto");
const subtle = globalThis.crypto.subtle, T = "ghp_" + "x".repeat(36), PW = "café olé brûlée naïve";
const f = await auth.seal(subtle, PW, "MichaelJSr", T, { iterations: auth.MIN_ITER });
const open = (pw, login, file) => auth.unseal(subtle, pw, login, file).then(t => t === T, () => false);
const flip = (b, i = 0) => { const x = Buffer.from(b, "base64"); x[(i + x.length) % x.length] ^= 1; return x.toString("base64"); };
const key = pbkdf2Sync(Buffer.from(PW.normalize("NFC")), Buffer.from(f.salt, "base64"), f.iterations, 32, "sha256");
const ct = Buffer.from(f.ct, "base64"), d = createDecipheriv("aes-256-gcm", key, Buffer.from(f.iv, "base64"));
d.setAAD(Buffer.from("login:michaeljsr")); d.setAuthTag(ct.subarray(ct.length - 16));
process.stdout.write(JSON.stringify({
  right: await open(PW, "MichaelJSr", f), nfd: await open(PW.normalize("NFD"), "michaeljsr", f), clear: !JSON.stringify(f).includes("xxxx"),
  wrong: await open(PW + "!", "MichaelJSr", f), copied: await open(PW, "Other", f), ct: await open(PW, "MichaelJSr", { ...f, ct: flip(f.ct, 3) }),
  tag: await open(PW, "MichaelJSr", { ...f, ct: flip(f.ct, -1) }), iv: await open(PW, "MichaelJSr", { ...f, iv: flip(f.iv) }),
  salt: await open(PW, "MichaelJSr", { ...f, salt: flip(f.salt) }),
  openssl: Buffer.concat([d.update(ct.subarray(0, ct.length - 16)), d.final()]).toString() === T }));
""")
    assert out == {"right": True, "nfd": True, "clear": True, "wrong": False, "copied": False, "ct": False, "tag": False,
                   "iv": False, "salt": False, "openssl": True}


def test_password_rules(tmp_path):
    refused = ["", "short pass", "Password1234!", "P@ssw0rd2024!!", "passwordpassword", "correct horse battery staple",
               "qwertyuiop123", "abcdefghijklmn", "123456789012", "abcabcabcabc", "aabbccddaabbcc", "MichaelJSr-rocks-2026",
               "ghp_" + "x" * 36, "impossibility2026", "@dmin12345678"]
    accepted = ["purple monkey dishwasher lamp", "Tr0mbone-Gecko-Velvet", "zq7#Lm2!pX9vR", "8306 1947 2251 0093", "red fox ate hen"]
    out = node(tmp_path, f"""
const r = pw => auth.checkPassword(pw, "MichaelJSr", "ghp_" + "x".repeat(36));
process.stdout.write(JSON.stringify({{ refused: {json.dumps(refused)}.map(p => [p, r(p).ok, r(p).problems.length]),
  accepted: {json.dumps(accepted)}.map(p => [p, r(p).ok, r(p).problems]) }}));
""")
    assert [p for p, ok, n in out["refused"] if ok or not n] == []         # each refused with a reason
    assert [p for p, ok, why in out["accepted"] if not ok] == []


def test_device_vault(tmp_path):
    """The device key: one non-extractable AES-GCM key in IndexedDB (two tabs racing make one), the token sealed in
    localStorage (kept) or sessionStorage, never in clear; clear() removes both; a blob without its key is dropped."""
    out = node(tmp_path, """
function fakeIdb() {
  const dbs = new Map(), later = f => setTimeout(f, 0);
  return { dbs, open(name) { const r = {}; later(() => {
    let db = dbs.get(name), fresh = !db; if (fresh) dbs.set(name, db = { stores: new Map() });
    r.result = { createObjectStore: s => db.stores.set(s, new Map()), close() {}, transaction(s, mode) {
      const store = db.stores.get(s), t = { n: 0, failed: false };
      const op = f => { t.n++; const q = {}; later(() => { try { q.result = f(); q.onsuccess?.(); } catch (e) { q.error = t.error = e; t.failed = true; q.onerror?.(); }
        later(() => { if (!--t.n) (t.failed ? t.onabort : t.oncomplete)?.(); }); }); return q; };
      t.objectStore = () => ({ get: k => op(() => store.get(k)), delete: k => op(() => store.delete(k)),
        add: (v, k) => op(() => { if (mode !== "readwrite") throw new Error("ReadOnlyError"); if (store.has(k)) throw new Error("ConstraintError"); store.set(k, structuredClone(v)); }) });
      return t; } };
    if (fresh) r.onupgradeneeded?.(); r.onsuccess?.(); }); return r; } };
}
const mem = () => { const m = new Map(); return { getItem: k => m.get(k) ?? null, setItem: (k, v) => m.set(k, String(v)), removeItem: k => m.delete(k), all: () => [...m.values()].join("|") }; };
const T = "ghp_" + "x".repeat(36), idb = fakeIdb(), local = mem(), session = mem(), v = auth.vault({ idb, local, session });
const keys = () => idb.dbs.get("ttscardmaker-signin")?.stores.get("keys");
const out = { empty: !v.has() && (await v.load()) === null };
await v.save(T, true);
const key = keys().get("device");
out.key = key.extractable === false && key.algorithm.name === "AES-GCM" && await globalThis.crypto.subtle.exportKey("raw", key).then(() => false, () => true);
out.kept = !!local.getItem(auth.SEALED) && !session.getItem(auth.SEALED) && !local.all().includes("xxxx") && (await v.load()).token === T;
await v.save(T, false);
const l2 = await v.load();
out.tab = !local.getItem(auth.SEALED) && !!session.getItem(auth.SEALED) && l2.keep === false && l2.token === T;
const idb2 = fakeIdb(), a = auth.vault({ idb: idb2, local: mem(), session: mem() }), b = auth.vault({ idb: idb2, local: mem(), session: mem() });
await Promise.all([a.save("A", true), b.save("B", true)]);
out.race = idb2.dbs.get("ttscardmaker-signin").stores.get("keys").size === 1 && (await a.load()).token === "A" && (await b.load()).token === "B";
await v.clear();
out.cleared = !v.has() && !keys().has("device");
local.setItem(auth.SEALED, JSON.stringify({ v: 1, iv: "AAAAAAAAAAAAAAAA", ct: "AAAAAAAAAAAAAAAAAAAAAAAA" }));
out.dropped = (await v.load()) === null && !local.getItem(auth.SEALED);
const none = auth.vault({ idb: null, local: mem(), session: mem() });
out.noIdb = await none.save(T, true).then(() => false, () => true) && (await none.load()) === null;
process.stdout.write(JSON.stringify(out));
""")
    assert out == {"empty": True, "key": True, "kept": True, "tab": True, "race": True, "cleared": True, "dropped": True, "noIdb": True}
