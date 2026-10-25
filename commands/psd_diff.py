"""
What changed inside a card PSD between two versions, in a card maker's words (docs/Campaigns/PUSH_GUARDS.md 7.5): "Name
text: 'White CLot' -> 'White Clot'", "added Ability FX: Blazing", "Ability box widened 20 px", "Blazing overlay hidden".

Layers pair by layer id (Photoshop keeps it across saves and renames), else by path. Photoshop's re-save noise is left out:
the blocks it rewrites on every save (shmd, lnsr, SoLd/PlLd ids, the legacy lrFX, ...), the re-rendered pixels of type,
smart object, shape and fill layers (their text, content, transform and paths are compared instead), groups opened or
closed in the Layers panel, and float jitter in text boxes and transforms. Report only: nothing here gates a push.

    changes(old, new, codepoints) -> [str]      one phrase per change, the most telling first
    line(phrases, limit) -> str                  the phrases joined, cut to limit ("and N more")
    python -m commands.psd_diff OLD.psd NEW.psd  prints the phrases

No Discord, network or config.
"""
from collections import Counter
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from hashlib import blake2b
from re import finditer, sub
from sys import argv
from typing import Optional
from unicodedata import category

from psd_tools import PSDImage
from psd_tools.constants import Tag
from psd_tools.terminology import Enum

from commands.psd_regions import overlay_words

# ========================
# Configuration
# ========================
NOISE = {b"shmd", b"lnsr", b"lyvr", b"cinf", b"lspf", b"lclr", b"fxrp", b"lrFX", b"PlLd", b"vowv", b"lyid", b"luni"}
# (rewritten by every save, or read elsewhere: lyid pairs, luni names)
FILTER_NOISE = {b"filterID", b"validAtPosition"}   # smart filter ids and cache flag, renumbered by a save
SEEN = {b"TySh", b"lfx2", b"lfxs", b"lmfx", b"SoLd", b"SoLE", b"lsct", b"iOpa", b"vmsk", b"vsms"}   # read by meaning
BLENDING = {b"clbl": 1, b"infx": 0, b"knko": 0, b"brst": [], b"tsly": 1}     # blending options, defaults
OWN_PIXELS = ("pixel",)                         # kinds whose channels are the art
PX = 1.0                                        # px a box or origin must move to count
SHORT = 40                                      # texts this short are quoted whole
HUNKS = 2                                       # text edits quoted per layer
LIST = 3                                        # subjects named per phrase before "and N more"
EFFECTS = {b"DrSh": "Drop Shadow", b"dropShadowMulti": "Drop Shadow", b"IrSh": "Inner Shadow",
           b"innerShadowMulti": "Inner Shadow", b"OrGl": "Outer Glow", b"IrGl": "Inner Glow", b"ebbl": "Bevel & Emboss",
           b"ChFX": "Satin", b"SoFi": "Color Overlay", b"solidFillMulti": "Color Overlay", b"GrFl": "Gradient Overlay",
           b"gradientFillMulti": "Gradient Overlay", b"patternFill": "Pattern Overlay", b"FrFX": "Stroke",
           b"frameFXMulti": "Stroke"}
STYLE = {"Font": "font", "FontSize": "size", "FillColor": "colour", "FauxBold": "faux bold", "FauxItalic": "faux italic",
         "Tracking": "tracking", "Leading": "leading", "AutoLeading": "leading", "HorizontalScale": "width",
         "VerticalScale": "height", "BaselineShift": "baseline", "FontCaps": "caps", "FontBaseline": "superscript",
         "Underline": "underline", "Strikethrough": "strikethrough", "StrokeFlag": "stroke", "StrokeColor": "stroke",
         "OutlineWidth": "stroke"}
SCALED = {"FontSize", "Leading", "BaselineShift", "OutlineWidth"}   # times the transform's scale
PARAGRAPH = {"Justification": "alignment", "FirstLineIndent": "indent", "StartIndent": "indent", "EndIndent": "indent",
             "SpaceBefore": "spacing", "SpaceAfter": "spacing", "EveryLineComposer": "composer"}
ALIGN = {0: "left", 1: "right", 2: "centre", 3: "justified", 4: "justified", 5: "justified", 6: "justified"}
ENUMS = {member.value: member.name.lower() for member in Enum}   # b"Nrml" and b"normal" alike

# phrase ranks, the most telling first
TEXT, ADDED, SHOWN, RENAMED, GEOMETRY, LOOK, EFFECT, CONTENT, ORDER, OTHER = range(10)


# ========================
# Data
# ========================
@dataclass
class Facts:
    """What a layer is, without Photoshop's re-save noise."""
    key: str
    name: str
    parent: str                                 # the parent's key ("" at the top)
    index: int                                  # among its siblings
    kind: str
    visible: bool
    opacity: int
    fill: int
    blend: str
    clipped: bool
    rect: tuple
    pixels: str = ""                            # the art's channels (pixel layers)
    mask: str = ""                              # the user mask (rect, defaults and channel)
    vector: str = ""                            # the vector mask or shape path
    content: str = ""                           # a smart object's file
    transform: tuple = ()                       # a smart object's corners
    filters: str = ""                           # a smart object's smart filters
    text: Optional[str] = None
    chars: list = field(default_factory=list)   # per character: (style dict, paragraph dict)
    origin: tuple = ()
    scale: tuple = ()
    box: tuple = ()                             # box text's size on the canvas (point text: ())
    effects: dict = field(default_factory=dict) # effect name -> [(on, settings)]
    effects_on: bool = True
    blending: tuple = ()
    other: dict = field(default_factory=dict)   # tag -> digest of blocks not otherwise read
    layer: object = field(default=None, repr=False, compare=False)


# ========================
# Helper Functions
# ========================
def _digest(*parts) -> str:
    h = blake2b(digest_size=8)
    for part in parts:
        h.update(part if isinstance(part, bytes) else repr(part).encode())
    return h.hexdigest()


def _mapping(value) -> bool:
    return callable(getattr(value, "items", None)) and not isinstance(value, (str, bytes))


def plain(value):
    """
    A descriptor as plain data: floats rounded (save jitter), enums by name (b"Nrml" and b"normal" alike), strings without
    their NUL, bytes hashed.
    """
    if _mapping(value):
        return {bytes(k) if isinstance(k, bytes) else str(k): plain(v) for k, v in value.items()
                if k not in (b"showInDialog", "showInDialog")}
    if _listing(value):
        return [plain(v) for v in value]
    if hasattr(value, "enum") and hasattr(value, "typeID"):
        enum = bytes(value.enum) if isinstance(value.enum, bytes) else str(value.enum).encode()
        return ("enum", ENUMS.get(enum, enum.decode("latin-1").strip().lower()))
    if hasattr(value, "unit") and hasattr(value, "value"):
        return (str(value.unit), round(float(value.value), 3))
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        return round(value, 3)
    if isinstance(value, (int, str)):
        return value.rstrip("\x00") if isinstance(value, str) else value
    if isinstance(value, bytes):
        return _digest(value)
    if hasattr(value, "value"):
        return plain(value.value)
    if hasattr(value, "__attrs_attrs__"):
        return {a.name: plain(getattr(value, a.name)) for a in value.__attrs_attrs__}
    return repr(value)


def _blocks(layer) -> dict:
    return layer._record.tagged_blocks if layer._record.tagged_blocks is not None else {}


def _data(layer, key: bytes):
    blocks = _blocks(layer)
    return blocks.get_data(key) if key in blocks else None


def _block(blocks, key) -> str:
    """A tagged block's digest: a descriptor by meaning (floats rounded), anything else by its bytes."""
    data = blocks.get_data(key)
    return _digest(plain(data)) if _mapping(data) else _digest(blocks[key].tobytes())


def _channels(layer, masks: bool) -> str:
    """The digest of a layer's channel data: the art (masks False) or the user masks."""
    out = [bytes(channel.data) for info, channel in zip(layer._record.channel_info, layer._channels)
           if (int(info.id) <= -2) == masks]
    return _digest(*out) if out else ""


def _mask(layer) -> str:
    data = layer._record.mask_data
    if data is None:
        return ""
    return _digest(data.left, data.top, data.right, data.bottom, data.background_color, plain(data.flags),
                   _channels(layer, True))


def _engine(value):
    """Engine data (text) as plain data."""
    if _mapping(value):
        return {str(getattr(k, "value", k)): _engine(v) for k, v in value.items()}
    if _listing(value):
        return [_engine(v) for v in value]
    if hasattr(value, "value"):
        return _engine(value.value)
    return round(value, 3) if isinstance(value, float) else value


def _listing(value) -> bool:
    """A list, or psd-tools' list elements (engine data and descriptor lists)."""
    return isinstance(value, (list, tuple)) or (hasattr(value, "__iter__") and not hasattr(value, "value")
                                                and not _mapping(value) and not isinstance(value, (str, bytes)))


def _type_facts(layer, facts: Facts) -> None:
    """Text, per-character style and paragraph, origin, scale and the box of a type layer."""
    setting = _data(layer, b"TySh")
    engine, resources = _engine(layer.engine_dict), _engine(layer.resource_dict)
    xx, _, _, yy, tx, ty = setting.transform
    facts.origin, facts.scale = (tx, ty), (xx, yy)
    text = str(engine.get("Editor", {}).get("Text", "") or "")
    facts.text = text.rstrip("\r")
    fonts = [str(font.get("Name", "")).rstrip("\x00") for font in resources.get("FontSet", [])]
    sheets = resources.get("StyleSheetSet", [])
    normal = sheets[resources.get("TheNormalStyleSheet", 0)].get("StyleSheetData", {}) if sheets else {}
    paragraphs = resources.get("ParagraphSheetSet", [])
    base = paragraphs[resources.get("TheNormalParagraphSheet", 0)].get("Properties", {}) if paragraphs else {}

    def runs(name: str, sheet: str, data: str, start: dict) -> list:
        run = engine.get(name, {})
        out = []
        for length, item in zip(run.get("RunLengthArray", []), run.get("RunArray", [])):
            merged = {**start, **item.get(sheet, {}).get(data, {})}
            out += [merged] * int(length)
        return out

    styles = runs("StyleRun", "StyleSheet", "StyleSheetData", normal)
    paras = runs("ParagraphRun", "ParagraphSheet", "Properties", base)
    for i in range(len(facts.text)):
        style = styles[i] if i < len(styles) else {}
        para = paras[i] if i < len(paras) else {}
        mine = {}
        for key in STYLE:
            if key in style:
                value = style[key]
                if key == "Font":
                    value = fonts[value] if isinstance(value, int) and value < len(fonts) else value
                elif key in ("FillColor", "StrokeColor"):
                    value = _colour(value)
                elif key in SCALED and isinstance(value, (int, float)):
                    value = round(value * abs(yy), 1)
                mine[key] = value
        facts.chars.append((mine, {key: para[key] for key in PARAGRAPH if key in para}))
    shapes = engine.get("Rendered", {}).get("Shapes", {}).get("Children", [])
    if shapes and shapes[0].get("ShapeType") == 1:
        x0, y0, x1, y1 = shapes[0].get("Cookie", {}).get("Photoshop", {}).get("BoxBounds", [0, 0, 0, 0])
        facts.box = ((x1 - x0) * abs(xx), (y1 - y0) * abs(yy))


def _colour(value) -> str:
    values = value.get("Values", []) if isinstance(value, dict) else []
    if len(values) == 4:
        return "#" + "".join(f"{round(v * 255):02X}" for v in values[1:])
    return repr(values)


def _effects(layer) -> tuple:
    """({effect name: [(on, settings)]}, master switch) from lfx2 (or lfxs, lmfx)."""
    found = None
    for key in (b"lfx2", b"lfxs", b"lmfx"):
        found = _data(layer, key)
        if found is not None:
            break
    if found is None:
        return {}, True
    data = plain(found)
    out: dict = {}
    for key, name in EFFECTS.items():
        items = data.get(key)
        for item in items if isinstance(items, list) else [items] if items else []:
            if isinstance(item, dict) and item.get(b"present", True):
                on = bool(item.get(b"enab", True))
                out.setdefault(name, []).append((on, {k: v for k, v in item.items() if k not in (b"enab", b"present")}))
    return out, bool(data.get(b"masterFXSwitch", True))


def _smart(layer, facts: Facts) -> None:
    """A smart object's content (its embedded or linked file), corners and smart filters; its pixels are a render."""
    data = None
    for key in (b"SoLd", b"SoLE"):
        data = _data(layer, key)
        if data is not None:
            break
    if data is None:
        return
    descriptor = data.data
    corners = list(descriptor.get(b"Trnf", []))
    facts.transform = tuple(round(float(v) / PX) * PX for v in corners)
    filters = descriptor.get(b"filterFX")
    facts.filters = _digest(_without(plain(filters), FILTER_NOISE)) if filters is not None else ""
    try:
        facts.content = _digest(layer.smart_object.data)
    except Exception:                           # a missing link: the descriptor stands for it
        facts.content = _digest(plain(descriptor.get(b"Idnt")))


def _without(value, keys: set):
    """Plain data without the given keys, at any depth."""
    if isinstance(value, dict):
        return {k: _without(v, keys) for k, v in value.items() if k not in keys}
    if isinstance(value, list):
        return [_without(v, keys) for v in value]
    return value


def facts_of(psd: PSDImage) -> dict:
    """{key: Facts} for every layer: keyed by layer id when ids are unique, else by path (with the nth duplicate)."""
    layers = list(psd.descendants())
    ids = Counter(layer.layer_id for layer in layers)
    seen: Counter = Counter()
    keys = {}
    for layer in layers:
        if layer.layer_id is not None and layer.layer_id > 0 and ids[layer.layer_id] == 1:
            keys[id(layer)] = f"id:{layer.layer_id}"
        else:
            path, node = [layer.name], layer.parent
            while node is not None and node is not psd:
                path.insert(0, node.name)
                node = node.parent
            joined = "/".join(path)
            keys[id(layer)] = f"path:{joined}#{seen[joined]}"
            seen[joined] += 1
    out = {}
    for layer in layers:
        parent = layer.parent
        siblings = list(parent)
        blocks = _blocks(layer)
        facts = Facts(
            key=keys[id(layer)], name=layer.name, parent=keys.get(id(parent), ""),
            index=next(i for i, s in enumerate(siblings) if s is layer), kind=layer.kind, visible=bool(layer.visible),
            opacity=int(layer.opacity), fill=int(_data(layer, b"iOpa") if b"iOpa" in blocks else 255),
            blend=getattr(layer.blend_mode, "name", str(layer.blend_mode)), clipped=bool(int(layer._record.clipping)),
            rect=(layer._record.left, layer._record.top, layer._record.right, layer._record.bottom))
        if layer.kind in OWN_PIXELS:
            facts.pixels = _channels(layer, False)
        facts.mask = _mask(layer)
        vector = next((key for key in (b"vmsk", b"vsms") if key in blocks), None)
        facts.vector = _block(blocks, vector) if vector else ""
        if layer.kind == "type":
            _type_facts(layer, facts)
        elif layer.kind == "smartobject":
            _smart(layer, facts)
        facts.effects, facts.effects_on = _effects(layer)
        facts.blending = tuple(_blend_option(blocks, key, default) for key, default in BLENDING.items())
        for key in blocks.keys():
            raw = bytes(key)
            if raw not in NOISE and raw not in SEEN and raw not in BLENDING:
                facts.other[raw] = _block(blocks, key)
        facts.layer = layer
        out[facts.key] = facts
    return out


def _blend_option(blocks, key: bytes, default):
    if key not in blocks:
        return default
    value = blocks.get_data(key)
    return plain(value) if not isinstance(value, (int, bool)) else int(value)


# ========================
# Phrases
# ========================
def _readable(text: str, codepoints: Optional[dict]) -> str:
    """Icons as [name] (private-use glyphs; unknown ones as [icon]), no invisible characters, line breaks and runs of
    spaces as one space."""
    out = []
    for char in text:
        if char in (codepoints or {}):
            out.append(f"[{codepoints[char]}]")
        elif 0xE000 <= ord(char) <= 0xF8FF:
            out.append("[icon]")
        elif not _invisible(char):
            out.append(char)
    return sub(r"\s+", " ", "".join(out)).strip()


def _invisible(char: str) -> bool:
    return category(char) == "Cf"               # BOM, zero-width space


def _quote(words: list) -> str:
    return "'" + " ".join(words) + "'"


def text_change(old: str, new: str, codepoints: Optional[dict] = None) -> str:
    """'a' -> 'b' for a short text, else the changed words with one word of context; "" when only spacing changed."""
    a, b = _readable(old, codepoints), _readable(new, codepoints)
    if a == b:
        return ""
    if len(a) <= SHORT and len(b) <= SHORT:
        return f"'{a}' -> '{b}'"
    wa, wb = a.split(" "), b.split(" ")
    hunks = [op for op in SequenceMatcher(None, wa, wb, autojunk=False).get_opcodes() if op[0] != "equal"]
    if sum(max(i2 - i1, j2 - j1) for _, i1, i2, j1, j2 in hunks) > max(len(wa), len(wb)) // 2:
        return f"rewritten ('{_clip(b)}')"
    out = []
    for _, i1, i2, j1, j2 in hunks[:HUNKS]:
        lo, hi = (1, 1) if i1 == i2 or j1 == j2 else (0, 0)    # insertion or deletion: a word of context
        out.append(f"{_quote(wa[max(0, i1 - lo):i2 + hi])} -> {_quote(wb[max(0, j1 - lo):j2 + hi])}")
    more = f" (+{len(hunks) - HUNKS} more)" if len(hunks) > HUNKS else ""
    return ", ".join(out) + more


def _clip(text: str, size: int = SHORT) -> str:
    return text if len(text) <= size else text[:size - 3].rstrip() + "..."


def _style_change(a: Facts, b: Facts, codepoints: Optional[dict] = None, quiet=()) -> str:
    """
    The text's style and paragraph change in the words both versions share (an edited word's new glyphs, such as a tag
    turned badge, belong to the text change), with the words when only a few changed: "font Perpetua -> stoogedings on
    '(1CD)'". "" when none.
    """
    wa, wb = [list(finditer(r"\S+", f.text or "")) for f in (a, b)]
    found: dict = {}
    words: dict = {}                            # the changed words of b, in order
    blocks = SequenceMatcher(None, [m.group() for m in wa], [m.group() for m in wb], autojunk=False).get_matching_blocks()
    for i, j, size in blocks:
        for x, y in zip(wa[i:i + size], wb[j:j + size]):
            for k in range(min(x.end() - x.start(), len(a.chars) - x.start(), len(b.chars) - y.start())):
                (style_a, para_a), (style_b, para_b) = a.chars[x.start() + k], b.chars[y.start() + k]
                for table, p, q in ((STYLE, style_a, style_b), (PARAGRAPH, para_a, para_b)):
                    for key in table:
                        if key not in quiet and p.get(key) != q.get(key):
                            found.setdefault(table[key], set()).add((key, p.get(key), q.get(key)))
                            words[y.start()] = y.group()
    names = list(dict.fromkeys(found))
    if not names:
        return ""
    if len(names) == 1 and len(found[names[0]]) == 1:
        key, x, y = next(iter(found[names[0]]))
        text = _describe(key, names[0], x, y)
    elif len(names) == 1:
        text = ("font size" if names[0] == "size" else names[0]) + " changed"
    else:
        text = "text style changed (" + ", ".join(names[:4]) + (", ..." if len(names) > 4 else "") + ")"
    if len(words) <= HUNKS and len(words) < len(wb):
        text += " on " + ", ".join(f"'{_readable(word, codepoints)}'" for word in words.values())
    return text


def _describe(key: str, name: str, x, y) -> str:
    if key == "Justification":
        return f"alignment {ALIGN.get(x, x)} -> {ALIGN.get(y, y)}"
    if key == "EveryLineComposer":
        return f"composer {'every-line' if x else 'single-line'} -> {'every-line' if y else 'single-line'}"
    if isinstance(x, bool) or isinstance(y, bool):
        return f"{name} {'on' if y else 'off'}"
    if x is None or y is None:
        return f"{name} changed"
    shown = (lambda v: f"{v:g}") if isinstance(x, (int, float)) and isinstance(y, (int, float)) else str
    label = "font size" if key == "FontSize" else "text colour" if key == "FillColor" else name
    return f"{label} {shown(x)} -> {shown(y)}"


def _unseen(old: str, new: str) -> str:
    """The change of a text that reads the same: invisible characters only, else line breaks or spacing."""
    seen = [[c for c in text if not _invisible(c)] for text in (old, new)]
    if seen[0] != seen[1]:
        return "line breaks or spacing changed"
    count = sum(map(_invisible, new)) - sum(map(_invisible, old))
    return ("invisible character" + ("s" if abs(count) != 1 else "") + " "
            + ("added" if count > 0 else "removed" if count < 0 else "changed"))


def _pct(value: int) -> str:
    return f"{round(value * 100 / 255)}%"


def _blend(name: str) -> str:
    return name.replace("_", " ").title()


def _compare(a: Facts, b: Facts, codepoints: Optional[dict], quiet=()) -> list:
    """[(rank, template)] for one paired layer: the template's {} is the layer's label."""
    out = []
    if a.name != b.name:
        out.append((RENAMED, "{} renamed '" + b.name + "'"))
    if a.kind != b.kind:
        verb = "rasterized" if a.kind == "type" and b.kind == "pixel" else f"turned into a {b.kind} layer"
        out.append((CONTENT, "{} " + verb))
    if a.visible != b.visible:
        out.append((SHOWN, "{} shown" if b.visible else "{} hidden"))
    if a.text is not None and b.text is not None:
        change = text_change(a.text, b.text, codepoints)
        if change:
            out.append((TEXT, "{} text: " + change))
        elif a.text != b.text:
            out.append((TEXT, "{} " + _unseen(a.text, b.text)))
        style = _style_change(a, b, codepoints, quiet)
        if style:
            out.append((LOOK, "{} " + style))
        if a.box and b.box:
            dw, dh = b.box[0] - a.box[0], b.box[1] - a.box[1]
            if abs(dw) >= PX:
                out.append((GEOMETRY, "{} box " + ("widened" if dw > 0 else "narrowed") + f" {abs(dw):.0f} px"))
            if abs(dh) >= PX:
                out.append((GEOMETRY, "{} box " + ("taller" if dh > 0 else "shorter") + f" by {abs(dh):.0f} px"))
        elif bool(a.box) != bool(b.box):
            out.append((GEOMETRY, "{} " + ("now box text" if b.box else "now point text")))
        dx, dy = b.origin[0] - a.origin[0], b.origin[1] - a.origin[1]
        if abs(dx) >= PX or abs(dy) >= PX:
            out.append((GEOMETRY, "{} moved " + f"{dx:+.0f},{dy:+.0f} px"))
        if any(abs(x - y) >= 0.005 for x, y in zip(a.scale, b.scale)):
            out.append((GEOMETRY, "{} scaled " + f"{abs(a.scale[1]) * 100:.0f}% -> {abs(b.scale[1]) * 100:.0f}%"))
    elif a.kind == "smartobject" and b.kind == "smartobject":
        if a.transform != b.transform and a.transform and b.transform:
            out.append((GEOMETRY, "{} " + _moved(_corners(a.transform), _corners(b.transform))))
        if a.content != b.content:
            out.append((CONTENT, "{} smart object content changed"))
        if a.filters != b.filters:
            out.append((CONTENT, "{} smart filters changed"))
    elif a.pixels or b.pixels:
        if not _same(a, b, a.pixels != b.pixels, False):
            out.append((CONTENT, "{} pixels changed"))
        elif a.rect != b.rect:
            out.append((GEOMETRY, "{} " + _moved(a.rect, b.rect)))
    if a.vector != b.vector:
        what = "shape" if b.kind == "shape" else "vector mask"
        out.append((CONTENT, "{} " + (f"{what} changed" if a.vector and b.vector else
                                      f"{what} added" if b.vector else f"{what} removed")))
    if a.mask != b.mask and not (a.mask and b.mask and _same(a, b, True, True)):
        out.append((CONTENT, "{} " + ("mask changed" if a.mask and b.mask else "mask added" if b.mask else "mask removed")))
    if a.opacity != b.opacity:
        out.append((LOOK, "{} opacity " + f"{_pct(a.opacity)} -> {_pct(b.opacity)}"))
    if a.fill != b.fill:
        out.append((LOOK, "{} fill " + f"{_pct(a.fill)} -> {_pct(b.fill)}"))
    if a.blend != b.blend:
        out.append((LOOK, "{} blend " + f"{_blend(a.blend)} -> {_blend(b.blend)}"))
    if a.clipped != b.clipped:
        out.append((LOOK, "{} clipped" if b.clipped else "{} unclipped"))
    if a.blending != b.blending:
        out.append((LOOK, "{} blending options changed"))
    out += _effect_changes(a, b)
    settings = sorted(key for key in set(a.other) | set(b.other) if a.other.get(key) != b.other.get(key))
    if settings and b.kind not in ("pixel", "type", "smartobject", "group"):
        out.append((CONTENT, "{} settings changed"))  # an adjustment's, fill's or shape's own blocks
    else:
        out += [(OTHER, "{} " + _tag(key) + " changed") for key in settings]
    return out


def _same(a: Facts, b: Facts, differ: bool, mask: bool) -> bool:
    """Whether two layers' pixels (or user masks) are equal once decoded: Photoshop may encode equal pixels differently.
    differ False (the encoded channels are equal) is equal without decoding."""
    if not differ:
        return True
    try:
        x, y = (a.layer.mask, b.layer.mask) if mask else (a.layer, b.layer)
        if mask and (x.bbox != y.bbox or x.background_color != y.background_color or x.disabled != y.disabled):
            return False
        one, two = x.topil(), y.topil()
        if one is None or two is None:
            return one is two
        return one.size == two.size and one.mode == two.mode and one.tobytes() == two.tobytes()
    except Exception:                           # cannot decode: the encoded channels decide
        return False


def _corners(transform: tuple) -> tuple:
    xs, ys = transform[0::2], transform[1::2]
    return min(xs), min(ys), max(xs), max(ys)


def _moved(a: tuple, b: tuple) -> str:
    wa, ha, wb, hb = a[2] - a[0], a[3] - a[1], b[2] - b[0], b[3] - b[1]
    if abs(wa - wb) < PX and abs(ha - hb) < PX:
        return f"moved {b[0] - a[0]:+.0f},{b[1] - a[1]:+.0f} px"
    return f"resized {wa:.0f}x{ha:.0f} -> {wb:.0f}x{hb:.0f} px"


def _tag(key: bytes) -> str:
    try:
        return Tag(key).name.lower().replace("_", " ")
    except ValueError:
        return key.decode("latin-1").strip()


def _effect_changes(a: Facts, b: Facts) -> list:
    out = []
    if a.effects_on != b.effects_on and (a.effects or b.effects):
        out.append((EFFECT, "{} effects " + ("on" if b.effects_on else "off")))
    for name in dict.fromkeys(list(a.effects) + list(b.effects)):
        x, y = a.effects.get(name, []), b.effects.get(name, [])
        if not x:
            out.append((EFFECT, name + " added to {}"))
        elif not y:
            out.append((EFFECT, name + " removed from {}"))
        elif [on for on, _ in x] != [on for on, _ in y]:
            out.append((EFFECT, "{} " + name + (" on" if any(on for on, _ in y) else " off")))
        elif not all(_alike(p, q) for (_, p), (_, q) in zip(x, y)) or len(x) != len(y):
            out.append((EFFECT, "{} " + name + " changed"))
    return out


def _alike(x, y) -> bool:
    """Equal settings on the keys both versions write: a newer Photoshop adds keys (gs99, numModifyingFX) to old styles."""
    if isinstance(x, dict) and isinstance(y, dict):
        return all(_alike(x[k], y[k]) for k in x.keys() & y.keys())
    if isinstance(x, list) and isinstance(y, list):
        return len(x) == len(y) and all(_alike(p, q) for p, q in zip(x, y))
    return x == y


def _order(a: dict, b: dict, pairs: dict) -> list:
    """[(rank, template, key)]: layers moved to another group or restacked among the siblings they kept."""
    out, groups = [], {}
    for key_a, key_b in pairs.items():
        x, y = a[key_a], b[key_b]
        parent = pairs.get(x.parent) if x.parent else ""        # old parent now ("": top, None: removed)
        if parent != y.parent:
            out.append((ORDER, "{} moved into " + _name(b[y.parent]) if y.parent else
                        "{} moved out of " + _name(a[x.parent]), key_b))
        else:
            groups.setdefault(y.parent, []).append((x.index, y.index, key_b))
    for siblings in groups.values():
        old = [k for _, _, k in sorted(siblings)]
        new = [k for _, _, k in sorted(siblings, key=lambda s: s[1])]
        if old != new:
            kept = {new[j] for block in SequenceMatcher(None, old, new, autojunk=False).get_matching_blocks()
                    for j in range(block.b, block.b + block.size)}
            out += [(ORDER, "{} restacked", k) for k in new if k not in kept]
    return out


def _labels(tables: tuple, overlays: set) -> dict:
    """{key: label}: the layer name as the first table has it (the new PSD, then the old one for a removed layer), with
    its group's name when another group of that PSD holds the same name, "overlay" after a status word laid over the
    ability text, and "group" after a group's name that does not say so."""
    out = {}
    for table in tables:
        places: dict = {}
        for facts in table.values():
            places.setdefault(facts.name, set()).add(facts.parent)
        for key, facts in table.items():
            label = _name(facts)
            if len(places[facts.name]) > 1 and facts.parent in table:
                label = f"{_name(table[facts.parent])}/{label}"
            if key in overlays:
                label += " overlay"
            elif facts.kind == "group" and "group" not in _name(facts).lower():
                label += " group"
            out.setdefault(key, label)
    return out


def _name(facts: Facts) -> str:
    return facts.name or ("unnamed group" if facts.kind == "group" else "unnamed layer")


def _join(subjects: list) -> str:
    counts = Counter(subjects)
    named = [f"{s} x{n}" if n > 1 else s for s, n in counts.items()]
    if len(named) > LIST + 1:
        named = named[:LIST] + [f"{len(named) - LIST} more"]
    return named[0] if len(named) == 1 else ", ".join(named[:len(named) - 1]) + " and " + named[len(named) - 1]


def changes(old: PSDImage, new: PSDImage, codepoints: Optional[dict] = None, quiet=()) -> list:
    """
    What changed between two versions of a card PSD, Photoshop's re-save noise left out.

    Args:
        old: The PSD before
        new: The PSD after
        codepoints: Private-use icon characters -> icon names (psd_parser.IconVocabulary.codepoints), to quote icons as
            [name] in text changes
        quiet: Paragraph and style keys left out ("EveryLineComposer": the guards fix it unsaid)

    Returns:
        The phrases, the most telling first ([] when nothing but a re-save changed)
    """
    out = []
    if old.size != new.size:
        out.append((GEOMETRY, f"canvas {old.width}x{old.height} -> {new.width}x{new.height}", None))
    if (old.color_mode, old.depth) != (new.color_mode, new.depth):
        out.append((CONTENT, f"colour mode {getattr(old.color_mode, 'name', old.color_mode)} {old.depth}-bit -> "
                             f"{getattr(new.color_mode, 'name', new.color_mode)} {new.depth}-bit", None))
    a, b = facts_of(old), facts_of(new)
    pairs = {key: key for key in a if key in b}
    removed = [key for key in a if key not in b]
    added = [key for key in b if key not in a]
    for key in list(removed):                   # deleted and remade in place: one layer
        twin = next((k for k in added if b[k].name == a[key].name and b[k].kind == a[key].kind
                     and pairs.get(a[key].parent, a[key].parent) == b[k].parent), None)
        if twin is not None:
            pairs[key] = twin
            removed.remove(key)
            added.remove(twin)
    overlays = set()
    for psd, table in ((old, a), (new, b)):
        ids = {f"id:{layer.layer_id}" for layer in overlay_words(psd)}
        overlays |= {key for key in table if key in ids}
    labels, before = _labels((b, a), overlays), _labels((a,), overlays)
    rows = []
    gone, new_keys = set(removed), set(added)
    for key in removed:
        if a[key].parent not in gone:
            rows.append((ADDED, "removed {}", labels[key] + _count(a, key, gone)))
    for key in added:
        if b[key].parent not in new_keys:
            hidden = "" if b[key].visible else " (hidden)"
            rows.append((ADDED, "added {}", labels[key] + _count(b, key, new_keys) + hidden))
    for key_a, key_b in pairs.items():
        rows += [(rank, template, before[key_a] if rank == RENAMED else labels[key_b])
                 for rank, template in _compare(a[key_a], b[key_b], codepoints, quiet)]
    rows += [(rank, template, labels[key]) for rank, template, key in _order(a, b, pairs)]
    out += rows
    merged: dict = {}
    for rank, template, subject in out:
        merged.setdefault((template, rank), []).append(subject)
    phrases = sorted(merged.items(), key=lambda item: (item[0][1], list(merged).index(item[0])))
    return [template.replace("{}", _join(subjects), 1) if subjects[0] is not None else template
            for (template, _), subjects in phrases]


def _count(table: dict, key: str, fresh: set) -> str:
    """" (N layers)" for a group added or removed with its new contents."""
    if table[key].kind != "group":
        return ""
    inside = sum(1 for k, f in table.items() if k in fresh and _within(table, k, key))
    return f" ({inside} layer{'s' if inside != 1 else ''})" if inside else ""


def _within(table: dict, key: str, group: str) -> bool:
    parent = table[key].parent
    while parent:
        if parent == group:
            return True
        parent = table[parent].parent if parent in table else ""
    return False


def line(phrases: list, limit: int = 0) -> str:
    """The phrases joined by "; ", whole phrases dropped from the end to fit limit (0: no limit), "and N more" told."""
    if not phrases:
        return ""
    for kept in range(len(phrases), 0, -1):
        rest = len(phrases) - kept
        text = "; ".join(phrases[:kept]) + (f"; and {rest} more" if rest else "")
        if not limit or len(text) <= limit:
            return text
    return _clip(phrases[0], limit)


def main(args: Optional[list] = None) -> int:
    """python -m commands.psd_diff OLD.psd NEW.psd: one phrase per line (nothing when only a re-save changed)."""
    args = argv[1:] if args is None else args
    if len(args) != 2:
        print("python -m commands.psd_diff OLD.psd NEW.psd")
        return 2
    for phrase in changes(PSDImage.open(args[0]), PSDImage.open(args[1])):
        print(phrase)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# ========================
# Module Exports
# ========================
__all__ = [
    'Facts',
    'NOISE',
    'changes',
    'facts_of',
    'line',
    'main',
    'plain',
    'text_change',
]
