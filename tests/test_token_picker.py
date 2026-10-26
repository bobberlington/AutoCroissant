"""
The website's token picker and token variants (TTSCardMaker site/, in node and Python): typing "[" in card text
suggests the icons, "name:" an icon's variants, "(" and a tag's start the timing tags (assets/tokens.js); text.js draws
[x:tint], [x:sticker], [x:glow] and [x:darkglow] with build_site's variant images, which build_icons makes with the
font build's own icon_art. Loads the site from $AC_SITE_SRC (a TTSCardMaker checkout) or the clone; skips without it
or without node.
"""
import importlib.util
import json
import re
import shutil
from os import environ
from os.path import expanduser
from pathlib import Path
from subprocess import run as run_process

import pytest

from conftest import CLONE

SITE = Path(expanduser(environ.get("AC_SITE_SRC") or str(CLONE))) / "site"
TOKENS_JS = SITE / "assets" / "tokens.js"
TEXT_JS = SITE / "assets" / "text.js"
INLINE_JS = SITE / "assets" / "inline.js"
EDIT_JS = SITE / "assets" / "edit.js"
BUILD_SITE = SITE / "build" / "build_site.py"
pytestmark = pytest.mark.skipif(not TOKENS_JS.is_file(),
                                reason=f"no {TOKENS_JS}")
needs_node = pytest.mark.skipif(not shutil.which("node"), reason="needs node")

ALL = ["tint", "sticker", "glow", "darkglow"]


def entry(name: str, kind: str) -> dict:
    return {"file": f"icons/{re.sub(r'[^a-z0-9]+', '-', name)}.webp", "name": name, "kind": kind, "v": ALL}


# cards.json "icons": every normalized spelling (name and aliases) -> its icon
ICONS = {"fire": entry("fire", "type"), "field": entry("field", "type"), "dark": entry("dark", "type"),
         "darkmatter": entry("darkmatter", "type"), "fragileequip": entry("fragile equip", "type"),
         "star": entry("star", "star"), "6star": entry("6star", "star"), "yinyangultra": entry("yinyangultra", "star"),
         "yinyang": entry("yinyangultra", "star"), "thepoorlydrawnbarnstar": entry("the_poorly_drawn_barnstar", "star"),
         "poorlydrawnbarnstar": entry("the_poorly_drawn_barnstar", "star"), "barnstar": entry("the_poorly_drawn_barnstar", "star")}
REACH = {"tint": 0, "sticker": 0.0729, "glow": 0.3021, "darkglow": 0.3021}


def node(tmp_path: Path, body: str):
    """Run body with tokens.js as P, text.js as T and the vocabulary V of ICONS; its stdout as JSON."""
    script = tmp_path / "probe.mjs"
    script.write_text(f"const P = await import({json.dumps(TOKENS_JS.as_uri())});\n"
                      f"const T = await import({json.dumps(TEXT_JS.as_uri())});\n"
                      f"const ICONS = {json.dumps(ICONS)}, REACH = {json.dumps(REACH)}, V = P.vocabulary(ICONS);\n"
                      "const at = (text, caret = text.length) => P.openToken(text, caret);\n"
                      "const tokens = (text, caret) => P.suggest(at(text, caret), V).map(i => i.token);\n"
                      f"{body}\n", encoding="utf-8")
    done = run_process(["node", str(script)], capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


@needs_node
def test_a_choice_replaces_the_partial_token(tmp_path):
    """The canonical token in place of what was typed; inside a token whose "]" or ")" follows, in place of all of it
    (a plain icon keeping the ":variant" after the caret); words after the caret stay; the caret just past it."""
    out = node(tmp_path, """
const choose = (text, caret, token) => { const o = at(text, caret), item = P.suggest(o, V).find(i => i.token === token);
  return item ? P.applyChoice(text, o, item) : null; };
process.stdout.write(JSON.stringify([choose("Heal [fi now", 8, "[fire]"), choose("Heal [fi] now", 8, "[fire]"),
  choose("Heal [fi]", 8, "[fire]"), choose("Heal [firx] now", 8, "[fire]"), choose("[fire:g]", 7, "[fire:glow]"),
  choose("[Dark M", 7, "[darkmatter]"), choose("x (1) y", 4, "(1CD)"), choose("x (1", 4, "(1CD)"),
  choose("x (1c) y", 5, "(1CD)"), choose("Immune to [fiand ice", 13, "[fire]"), choose("Heal 2 (1HP", 9, "(1CD)"),
  choose("Deal 3 [fire damage", 8, "[field]"), choose("[fragile equip] x", 8, "[fragile equip]"), choose("[fi:glow]", 3, "[fire]"),
  choose("[fi:glow]", 3, "[field]"), choose("[fire:glow] x", 8, "[fire:darkglow]"), choose("[fire:glow]", 5, "[fire:tint]"),
  choose("x (-1/2GC) y", 9, "(-1/2GCD)"), choose("[Dark M:Glow] x", 7, "[darkmatter]"), choose("[fi:] x", 3, "[fire]"),
  choose("[fi:glows]", 3, "[fire]")].map(r => r && [r.text, r.caret])));""")
    assert out[:19] == [["Heal [fire] now", 11], ["Heal [fire] now", 11], ["Heal [fire]", 11], ["Heal [fire] now", 11],
                        ["[fire:glow]", 11], ["[darkmatter]", 12], ["x (1CD) y", 7], ["x (1CD)", 7], ["x (1CD) y", 7],
                        ["Immune to [fire]and ice", 16], ["Heal 2 (1CD)HP", 12], ["Deal 3 [field]fire damage", 14],
                        ["[fragile equip] x", 15], ["[fire:glow]", 11], ["[field:glow]", 12], ["[fire:darkglow] x", 15],
                        ["[fire:tint]", 11], ["x (-1/2GCD) y", 11], ["[darkmatter:Glow] x", 17]]
    assert out[19:] == [["[fire] x", 6], ["[fire]", 6]]                    # never "[fire:]": letters in the font


@pytest.fixture(scope="module")
def bs():
    for module in ("numpy", "cv2", "fontTools", "PIL"):
        pytest.importorskip(module)
    spec = importlib.util.spec_from_file_location("site_build_tokens", BUILD_SITE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _icons(bs, monkeypatch, root, tree, out):
    monkeypatch.setattr(bs, "REPO", str(root))
    monkeypatch.setattr(bs, "DIST", str(out / "dist"))
    monkeypatch.setattr(bs, "CACHE", str(out / ".cache"))
    return bs.build_icons(tree)

