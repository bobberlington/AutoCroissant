"""
The website's password sign-in (TTSCardMaker site/assets/signin.js in node), one smoke test (owner 2026-11-12): a sealed
file opens only with the right password, and OpenSSL reads its format. Loads it from $AC_SITE_SRC (a TTSCardMaker
checkout) or the clone; skips without it.
"""
import json
from subprocess import run as run_process

import pytest

from conftest import SITE_SRC

SITE = SITE_SRC / "site"
LOGIN_JS = SITE / "assets" / "signin.js"


def node(tmp_path, body: str):
    """Run body after importing signin.js as auth; its stdout as JSON."""
    if not LOGIN_JS.is_file():
        pytest.skip(f"no signin.js at {LOGIN_JS}")
    script = tmp_path / "probe.mjs"
    script.write_text(f"const auth = await import({json.dumps(LOGIN_JS.as_uri())});\n{body}\n", encoding="utf-8")
    return json.loads(run_process(["node", str(script)], capture_output=True, text=True, check=True).stdout)


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
