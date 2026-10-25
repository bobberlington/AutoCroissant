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
    "too many iterations": ("MichaelJSr", good(iterations=5_000_001)),
    "login with a newline": ("MichaelJSr\n", good(login="MichaelJSr\n")),
    "iterations a float": ("MichaelJSr", good(iterations=2e6)),
    "iterations True": ("MichaelJSr", good(iterations=True)),
    "salt 15 bytes": ("MichaelJSr", good(salt=B64(15))),
    "iv 16 bytes": ("MichaelJSr", good(iv=B64(16))),
    "ct only a tag": ("MichaelJSr", good(ct=B64(16))),
    "ct over 1 KB": ("MichaelJSr", good(ct=B64(1025))),
    "ct not base64": ("MichaelJSr", good(ct="%%%")),
}


def node(tmp_path, body: str):
    """Run body after importing login.js as auth; its stdout as JSON."""
    if not (LOGIN_JS.is_file() and shutil.which("node")):
        pytest.skip(f"no login.js at {LOGIN_JS}, or no node")
    script = tmp_path / "probe.mjs"
    script.write_text(f"const auth = await import({json.dumps(LOGIN_JS.as_uri())});\n{body}\n", encoding="utf-8")
    return json.loads(run_process(["node", str(script)], capture_output=True, text=True, check=True).stdout)


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

