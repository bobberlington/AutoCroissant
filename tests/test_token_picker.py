"""
The website's token picker and token variants (TTSCardMaker site/, in node and Python): typing "[" in card text
suggests the icons, "name:" an icon's variants, "(" and a tag's start the timing tags (assets/tokens.js); text.js draws
[x:tint], [x:sticker], [x:glow] and [x:darkglow] with build_site's variant images, which build_icons makes with the
font build's own icon_art. Loads the site from $AC_SITE_SRC (a TTSCardMaker checkout) or the clone; skips without it,
without node, or when the clone is behind (no tokens.js: land TTSCardMaker work/token-picker).
"""
import ast
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
FONT_BUILD = SITE.parent / "MDW" / "Fonts" / "stoogedings-src" / "build_stoogedings.py"
pytestmark = pytest.mark.skipif(not TOKENS_JS.is_file(),
                                reason=f"clone behind: no {TOKENS_JS} (land TTSCardMaker work/token-picker)")
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
def test_the_open_token_at_the_caret(tmp_path):
    """An unclosed "[" (a name, maybe "name:variant") or "(" (a tag's start) before the caret, on its line only; the
    choice ends at the caret, or past the rest of a token the caret is inside when its "]" or ")" follows."""
    out = node(tmp_path, """
const cases = [["Deal 2 ["], ["Deal 2 [fi"], ["[fire:gl"], ["[dark m"], ["[fire] and"], ["[fi\\nre"], ["[fire (1"],
  ["(1"], ["(1cd"], ["(O"], ["(o"], ["(see below)", 4], ["(See below"], ["(1 per turn"], ["Heal [fi] now", 8],
  ["[" + "a".repeat(53)], ["[fire:2"], ["Heal [fi now", 8], ["To [fiand ice", 6], ["(1HP", 2], ["[fragile equip] x", 8],
  ["[fi:glow]", 3], ["[fire:gl] x", 8], ["[fire:gl x]", 8], ["(-1/2GCD"], ["(-1/2GCD) y", 7], ["[fi and [fire]", 3]];
process.stdout.write(JSON.stringify(cases.map(([t, c]) => at(t, c))));""")
    pick = lambda o: o and {k: o[k] for k in ("type", "start", "end", "close", "base", "variant", "text", "tail")
                            if k in o and (k != "tail" or o[k])}
    assert [pick(o) for o in out] == [
        {"type": "icon", "start": 7, "end": 8, "close": 8, "base": "", "variant": None},
        {"type": "icon", "start": 7, "end": 10, "close": 10, "base": "fi", "variant": None},
        {"type": "icon", "start": 0, "end": 8, "close": 8, "base": "fire", "variant": "gl"},
        {"type": "icon", "start": 0, "end": 7, "close": 7, "base": "dark m", "variant": None},
        None,                                                    # closed
        None,                                                    # another line
        {"type": "tag", "start": 6, "end": 8, "close": 8, "text": "1"},
        {"type": "tag", "start": 0, "end": 2, "close": 2, "text": "1"},
        {"type": "tag", "start": 0, "end": 4, "close": 4, "text": "1cd"},
        {"type": "tag", "start": 0, "end": 2, "close": 2, "text": "O"},
        None,                                                    # "(o": words in brackets start in lower case
        None,                                                    # "(see below)"
        None,                                                    # "(See below": words
        None,                                                    # a space: no tag
        {"type": "icon", "start": 5, "end": 8, "close": 9, "base": "fi", "variant": None},   # its "]" follows
        None,                                                    # longer than any token
        None,                                                    # a variant is letters
        {"type": "icon", "start": 5, "end": 8, "close": 8, "base": "fi", "variant": None},   # no "]": the words stay
        {"type": "icon", "start": 3, "end": 6, "close": 6, "base": "fi", "variant": None},   # "and ice" stays
        {"type": "tag", "start": 0, "end": 2, "close": 2, "text": "1"},                      # "HP" stays
        {"type": "icon", "start": 0, "end": 8, "close": 15, "base": "fragile", "variant": None},   # inside a spaced token
        {"type": "icon", "start": 0, "end": 3, "close": 9, "base": "fi", "variant": None, "tail": ":glow]"},
        {"type": "icon", "start": 0, "end": 8, "close": 9, "base": "fire", "variant": "gl"},
        {"type": "icon", "start": 0, "end": 8, "close": 8, "base": "fire", "variant": "gl"},  # a variant is letters
        {"type": "tag", "start": 0, "end": 8, "close": 8, "text": "-1/2GCD"},                # 7 characters, text.js TAG's most
        {"type": "tag", "start": 0, "end": 7, "close": 9, "text": "-1/2GC"},
        {"type": "icon", "start": 0, "end": 3, "close": 3, "base": "fi", "variant": None}]   # the next "[" is another token


@needs_node
def test_icons_rank_by_name_then_alias(tmp_path):
    """Name prefix, alias prefix, then (2+ letters) inside a name or alias, then (3+ letters, nothing else) one slip
    away (two from 6 letters); types before stars, then the most used in the cards, then by name; case, spaces, _ and -
    ignored; an exact name or alias first with its variants right after it; nothing for "(See below"."""
    out = node(tmp_path, """
const rows = P.suggest(at("Burn ["), V), used = P.vocabulary(ICONS, ["Burn [fire] and [Fire:glow].", "[field]", "[ice]"]);
const by = (v, text) => P.suggest(at(text), v).map(i => i.token);
process.stdout.write(JSON.stringify({ all: rows.map(i => i.token), first: rows[0], fi: tokens("[fi"), FI: tokens("[F-i"),
  fire: tokens("[fire"), barn: tokens("[barn"), drawn: tokens("[drawn"), one: tokens("[a"), darkm: tokens("[dark m"),
  yin: tokens("[Yin_Yang"), none: tokens("[zz"), see: tokens("(See below"), uses: used.map(e => [e.name, e.uses]),
  usedFi: by(used, "[fi"), usedAll: by(used, "[").slice(0, 3), fier: tokens("[fier"), firre: tokens("[firre"),
  drak: tokens("[drak"), poorly: tokens("[the poorly drwan bar"), xyz: tokens("[xyz"), fireq: tokens("[firex") }));""")
    assert out["all"] == ["[dark]", "[darkmatter]", "[field]", "[fire]", "[fragile equip]",
                          "[6star]", "[star]", "[the_poorly_drawn_barnstar]", "[yinyangultra]"]
    assert out["first"] == {"token": "[dark]", "label": "dark", "kind": "type", "variant": "", "note": "",
                            "image": "icons/dark.webp"}
    assert out["fi"] == out["FI"] == ["[field]", "[fire]"]
    assert out["fire"] == ["[fire]", "[fire:tint]", "[fire:sticker]", "[fire:no]", "[fire:glow]", "[fire:darkglow]"]
    assert out["barn"] == ["[the_poorly_drawn_barnstar]"]                       # by its alias
    assert out["drawn"] == ["[the_poorly_drawn_barnstar]"]                      # inside the name
    assert out["one"] == []                                                     # one letter: prefixes only
    assert out["darkm"] == ["[darkmatter]"]
    assert out["yin"][0] == "[yinyangultra]" and out["yin"][1:] == [f"[yinyangultra:{v}]" for v in
                                                                   ("tint", "sticker", "no", "glow", "darkglow")]
    assert out["none"] == out["see"] == []
    assert dict(out["uses"])["fire"] == 2 and dict(out["uses"])["field"] == 1 and dict(out["uses"])["dark"] == 0
    assert out["usedFi"] == ["[fire]", "[field]"]                                # the most used first
    assert out["usedAll"] == ["[fire]", "[field]", "[dark]"]
    assert out["fier"] == ["[field]", "[fire]"]                                  # a slip: fie(l)d, fi(er)
    assert out["firre"] == ["[fire]"] and out["drak"] == ["[dark]", "[darkmatter]"]
    assert out["poorly"] == ["[the_poorly_drawn_barnstar]"]                      # two slips from 6 letters
    assert out["xyz"] == [] and out["fireq"] == ["[fire]"]                       # "firex": fire, one too many


@needs_node
def test_variants_after_the_colon(tmp_path):
    """After "name:" the variants the font draws (the bot's ICON_VARIANTS): of that icon when the name is exact, else of
    every match; a prefix first, then inside; rows carry the variant's own picture."""
    from commands.psd_parser import ICON_VARIANTS
    out = node(tmp_path, """
const rows = P.suggest(at("[fire:"), V);
process.stdout.write(JSON.stringify({ variants: P.VARIANTS, fire: rows.map(i => [i.token, i.note, i.image]), g: tokens("[fire:g"),
  dm: tokens("[dark matter:GL"), fi: tokens("[fi:no"), none: tokens("[fire:x") }));""")
    assert out["variants"] == [v for v in ICON_VARIANTS if v]
    assert out["fire"] == [["[fire:tint]", "text colour", "icons/fire.tint.webp"],
                           ["[fire:sticker]", "sticker", "icons/fire.sticker.webp"],
                           ["[fire:no]", "crossed out", "icons/fire.webp"],          # style.css draws the sign
                           ["[fire:glow]", "white glow", "icons/fire.glow.webp"],
                           ["[fire:darkglow]", "black glow", "icons/fire.darkglow.webp"]]
    assert out["g"] == ["[fire:glow]", "[fire:darkglow]"]
    assert out["dm"] == ["[darkmatter:glow]", "[darkmatter:darkglow]"]
    assert out["fi"] == ["[field:no]", "[fire:no]"]
    assert out["none"] == []


def font_tags() -> set:
    """The font build's badge_tags() (run on its own: the script imports cv2 and fontTools)."""
    tree = ast.parse(FONT_BUILD.read_text(encoding="utf-8"))
    fns = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "badge_tags"]
    ns = {}
    exec(compile(ast.Module(fns, []), str(FONT_BUILD), "exec"), ns)
    return {tag for tag, _ in ns["badge_tags"]()}


@needs_node
def test_timing_tags(tmp_path):
    """"(" and a tag's start: the font's tags in any case, the cards' own first and the rulebook's old ones last, each
    with text.js's meaning; ordinary brackets get none."""
    out = node(tmp_path, """
const rows = P.suggest(at("Heal 2 (1"), V);
process.stdout.write(JSON.stringify({ tags: P.TAGS, one: rows.map(i => i.token), first: rows[0], pg: tokens("(1pg"),
  act: tokens("(ACT"), o: tokens("(O"), old: tokens("(E2"), see: tokens("(See below"), only: tokens("(only once"),
  words: tokens("(1 per turn"), long: tokens("(-1/2GCD"), longer: tokens("(-1/2GCDX") }));""")
    if FONT_BUILD.is_file():
        assert set(out["tags"]) == font_tags()
    assert len(out["tags"]) == len(set(out["tags"])) == 341
    assert out["one"][:6] == ["(1CD)", "(1GCD)", "(1U)", "(1PG)", "(1/1CD)", "(1/2CD)"]
    assert out["one"][-2:] == ["(1PT)", "(1PGT)"] and len(out["one"]) == 26
    assert out["first"] == {"token": "(1CD)", "label": "Cooldown: 1 turn", "kind": "timing tag", "image": "", "note": "",
                            "tag": "1CD", "group": "turn", "old": False}
    assert out["pg"] == ["(1PG)", "(1PGT)"]
    assert out["act"] == ["(Action)"]
    assert out["o"] == ["(OYT)", "(OFS)", "(OS)", "(OD)"]
    assert out["old"] == ["(E2T)", "(E2GT)"]
    assert out["see"] == out["only"] == out["words"] == out["longer"] == []
    assert out["long"] == ["(-1/2GCD)"]                                          # the longest tags, fully typed


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
  choose("x (-1/2GC) y", 9, "(-1/2GCD)"), choose("[Dark M:Glow] x", 7, "[darkmatter]")].map(r => r && [r.text, r.caret])));""")
    assert out == [["Heal [fire] now", 11], ["Heal [fire] now", 11], ["Heal [fire]", 11], ["Heal [fire] now", 11],
                   ["[fire:glow]", 11], ["[darkmatter]", 12], ["x (1CD) y", 7], ["x (1CD)", 7], ["x (1CD) y", 7],
                   ["Immune to [fire]and ice", 16], ["Heal 2 (1CD)HP", 12], ["Deal 3 [field]fire damage", 14],
                   ["[fragile equip] x", 15], ["[fire:glow]", 11], ["[field:glow]", 12], ["[fire:darkglow] x", 15],
                   ["[fire:tint]", 11], ["x (-1/2GCD) y", 11], ["[darkmatter:Glow] x", 17]]


@needs_node
def test_card_text_fields_take_the_picker(tmp_path):
    """The ability and note fields, in the full Edit form (cards and rulebook pages) and inline, carry data-tokens."""
    script = tmp_path / "inline.mjs"
    script.write_text(f"const I = await import({json.dumps(INLINE_JS.as_uri())});\n"
                      "process.stdout.write(JSON.stringify(['ability', 'note', 'series', 'name'].map(f => I.control(f, 'x'))));\n",
                      encoding="utf-8")
    done = run_process(["node", str(script)], capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert done.returncode == 0, done.stderr
    assert ["data-tokens" in html for html in json.loads(done.stdout)] == [True, True, False, False]
    fields = re.findall(r'<textarea id="ef-(ability|note)"[^>]*>', EDIT_JS.read_text(encoding="utf-8"))
    assert sorted(fields) == ["ability", "ability", "note", "note"]
    assert all("data-tokens" in tag for tag in re.findall(r'<textarea id="ef-(?:ability|note)"[^>]*>', EDIT_JS.read_text(encoding="utf-8")))


@needs_node
def test_text_js_draws_variants_with_their_images(tmp_path):
    """[x:tint] a text-colour mask, [x:sticker] the outline mask under the icon, [x:glow]/[x:darkglow] their pictures
    sized by cards.json iconVariants; without a variant image (an older build) the plain icon, as the parser reads it."""
    out = node(tmp_path, """
const r = T.makeRenderer({ cards: [], icons: ICONS, variants: REACH });
const old = T.makeRenderer({ cards: [], icons: { fire: { file: "icons/fire.webp", name: "fire", kind: "type" } } });
const html = v => r.render(`a [fire:${v}] b`);
process.stdout.write(JSON.stringify({ tint: html("tint"), sticker: html("sticker"), glow: html("Glow"), dark: html("darkglow"),
  no: html("no"), plain: r.render("a [fire] b"), unknown: html("glows"), spaced: [html(" glow"), html("no ")],
  quiet: r.icon("fire:glow", { tip: false }),
  old: ["tint", "sticker", "glow", "darkglow"].map(v => old.render(`a [fire:${v}] b`) === old.render("a [fire] b")) }));""")
    assert re.search(r'<span class="tokm" role="img" aria-label="fire \(text colour\)" style="-webkit-mask-image:url\(&quot;'
                     r'icons/fire\.tint\.webp&quot;\);mask-image:url\(&quot;icons/fire\.tint\.webp&quot;\)" data-tip="fire \(text colour\)">'
                     r'</span>', out["tint"])
    assert re.search(r'<span class="tokv" role="img" aria-label="fire \(sticker\)" style="--r:0\.0729" data-tip="fire \(sticker\)">'
                     r'<span class="tokm" style="[^"]*icons/fire\.sticker\.webp[^"]*"></span><img src="icons/fire\.webp" alt=""></span>',
                     out["sticker"])
    assert '<img class="tokv" src="icons/fire.glow.webp" alt="fire (white glow)" style="--r:0.3021" data-tip="fire (white glow)">' in out["glow"]
    assert 'src="icons/fire.darkglow.webp"' in out["dark"] and "black glow" in out["dark"]
    assert '<span class="tok-no"><img class="tok" src="icons/fire.webp"' in out["no"]
    assert out["unknown"] == out["plain"]
    assert out["spaced"] == [out["plain"]] * 2                                   # no spaces around it, as in the font
    assert "data-tip" not in out["quiet"]
    assert out["old"] == [True] * 4


@pytest.fixture(scope="module")
def bs():
    for module in ("numpy", "cv2", "fontTools", "PIL"):
        pytest.importorskip(module)
    spec = importlib.util.spec_from_file_location("site_build_tokens", BUILD_SITE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def icon_repo(tmp_path):
    """Types/Fire.png (an orange disc with a dark hole), Types/Stars/Gold Star.png, Types/New.png (not in the font yet),
    the manifest and the font build script."""
    from PIL import Image, ImageDraw
    root = tmp_path / "repo"
    (root / "Types" / "Stars").mkdir(parents=True)
    fire = Image.new("RGBA", (300, 300))
    draw = ImageDraw.Draw(fire)
    draw.ellipse((20, 20, 280, 280), fill=(240, 120, 20, 255))
    draw.ellipse((120, 120, 180, 180), fill=(30, 10, 0, 255))
    fire.save(root / "Types" / "Fire.png")
    star = Image.new("RGBA", (200, 100))
    ImageDraw.Draw(star).polygon([(0, 50), (100, 0), (200, 50), (100, 100)], fill=(250, 210, 40, 255))
    star.save(root / "Types" / "Stars" / "Gold Star.png")
    Image.new("RGBA", (50, 50), (0, 200, 0, 255)).save(root / "Types" / "New.png")
    src = root / "MDW" / "Fonts" / "stoogedings-src"
    src.mkdir(parents=True)
    shutil.copy(FONT_BUILD, src / FONT_BUILD.name)
    manifest = {"icons": {"fire": {"kind": "type", "file": "Types/Fire.png", "pua": "E000", "aliases": [], "tint": "auto"},
                          "gold_star": {"kind": "star", "file": "Types/Stars/Gold Star.png", "pua": "E001",
                                        "aliases": ["gold"], "tint": "silhouette"}}}
    (src / "stoogedings.json").write_text(json.dumps(manifest))
    tree = {"Types/Fire.png": "a" * 40, "Types/Stars/Gold Star.png": "b" * 40, "Types/New.png": "c" * 40}
    return root, tree


def _icons(bs, monkeypatch, root, tree, out):
    monkeypatch.setattr(bs, "REPO", str(root))
    monkeypatch.setattr(bs, "DIST", str(out / "dist"))
    monkeypatch.setattr(bs, "CACHE", str(out / ".cache"))
    return bs.build_icons(tree)


def test_build_site_makes_variant_images(bs, icon_repo, tmp_path, monkeypatch):
    """Per manifest icon and variant a WebP, composed as the font draws it at the plain icon's scale: tint the shape in
    the icon's box, sticker the outline and glow/darkglow the halo with the icon, in a square reaching past the box by
    the font's OUTLINE and GLOW_SIZE; cached by blob; none for an icon the font lacks or without the font's packages."""
    from PIL import Image
    root, tree = icon_repo
    icons, reach = _icons(bs, monkeypatch, root, tree, tmp_path / "one")
    fb = bs.font_builder()
    assert reach == {"tint": 0, "sticker": round(round(96 * fb.OUTLINE / fb.HEIGHT) / 96, 4),
                     "glow": round(round(96 * fb.GLOW_SIZE / fb.HEIGHT) / 96, 4), "darkglow": round(round(96 * fb.GLOW_SIZE / fb.HEIGHT) / 96, 4)}
    assert reach["sticker"] == pytest.approx(0.07, abs=1 / 96) and reach["glow"] == pytest.approx(0.3, abs=1 / 96)
    assert icons["fire"]["v"] == icons["goldstar"]["v"] == icons["gold"]["v"] == ALL
    assert "v" not in icons["new"]
    dist = tmp_path / "one" / "dist" / "icons"
    opened = {v: Image.open(dist / f"fire.{v}.webp").convert("RGBA") for v in ALL}
    plain = Image.open(dist / "fire.webp")
    side = lambda v: 96 + 2 * round(reach[v] * 96)
    assert opened["tint"].size == plain.size and Image.open(dist / "gold-star.tint.webp").size == Image.open(dist / "gold-star.webp").size
    assert {v: opened[v].size for v in ("sticker", "glow", "darkglow")} == {v: (side(v), side(v)) for v in ("sticker", "glow", "darkglow")}
    tint = opened["tint"]
    assert tint.getpixel((48, 10))[:3] == (255, 255, 255) and tint.getpixel((48, 10))[3] > 200      # the disc, a mask
    assert tint.getpixel((48, 48))[3] < 60                                                        # its dark hole cut out
    stk, m = opened["sticker"], round(reach["sticker"] * 96)
    assert stk.getpixel((m // 2, stk.height // 2))[3] > 200 and stk.getpixel((0, 0))[3] == 0     # past the icon's box
    for v, rgb in (("glow", (255, 255, 255)), ("darkglow", (0, 0, 0))):
        im, g = opened[v], round(reach[v] * 96)
        halo = im.getpixel((g - 4, im.height // 2))
        assert halo[3] > 40 and all(abs(a - b) < 8 for a, b in zip(halo[:3], rgb)), (v, halo)
        assert im.getpixel((0, 0))[3] < 10                                                        # it fades out
        r, gr, b, a = im.getpixel((g + 48, g + 10))
        assert a == 255 and r > 200 and 90 < gr < 150 and b < 60, (v, (r, gr, b, a))              # the icon on top

    monkeypatch.setattr(bs, "variant_art", lambda *a: pytest.fail("cached variant images drawn again"))
    again, _ = _icons(bs, monkeypatch, root, tree, tmp_path / "one")
    assert again["fire"]["v"] == ALL

    monkeypatch.setattr(bs, "font_builder", lambda: None)                                        # no cv2 / numpy / fontTools
    bare, no_reach = _icons(bs, monkeypatch, root, tree, tmp_path / "bare")
    assert no_reach == {} and not any("v" in e for e in bare.values())
    assert sorted(p.name for p in (tmp_path / "bare" / "dist" / "icons").iterdir()) == ["fire.webp", "gold-star.webp", "new.webp"]
