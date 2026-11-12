"""
Icon glyphs drawn again (owner 2026-11-08): once the stoogedings workflow commits the icon font rebuilt for a changed
Types/ icon, Card sync draws the glyphs that build changed again in the text layers that show them, with the new font,
and re-exports their PNGs, in the Linked redraw's commit (psd_links.redraw_linked: one rewrite per card, trailer
Card-sweep: glyphs, silent). A layer the splice cannot take gets a hand flag in site/Card Edits.json instead.

    font_commits(clone, head) -> [sha]            the workflow's font commits, newest first (DAYS, FONTS_BACK)
    changed(old, new) -> set                      glyph names drawn otherwise (advance, outline, SVG picture), new or gone
    due(clone, head, tree, details, edits) -> {PSD path: Due}
                                                  PSDs drawn before such a commit whose texts (psd_details) shape to a
                                                  glyph the workflow changed since, or shape otherwise (HarfBuzz)
    redraw(path, out_path, fonts_dir, found) -> [layer]
                                                  those glyphs spliced in, gated (card_standards' old glyphs way);
                                                  Refused when a layer laid out by the old font lays out otherwise now
    flag(text, rel, blob, note) -> str            Card Edits.json with the card's hand flag

A PSD counts as drawn before a font commit while no commit that has the font commit in its history changed it but the
bot's and sweeps' (they keep its text pixels; a glyph redraw, Card-sweep: glyphs, draws), a pure move aside. Font
commits made otherwise are passed over: only the workflow's own changes are drawn again. No Discord, network or config.
"""
from dataclasses import dataclass, replace
from functools import lru_cache
from io import BytesIO
from json import loads
from os import environ
from re import sub
from subprocess import run as run_process
from time import time
from typing import Optional

from psd_tools import PSDImage

import commands.card_standards as standards
import commands.psd_edit as psd_edit
from commands.card_bake import rewrite_entry
from commands.card_edits import FLAG_MAX, CardAuthors, CardEdits, dump_edits
from commands.card_models import card_name
from commands.psd_edit import Refused

# ========================
# Configuration
# ========================
FONT_FILE = "MDW/Fonts/stoogedings.ttf"
FONT_SUBJECT = "Font: stoogedings rebuilt"      # stoogedings.yml's commit (.github/scripts/stoogedings.py)
BOT = "AutoCroissant"
SWEEP = "glyphs"                                # the redraw's Card-sweep value
FONTS_BACK = 10                                 # the workflow's font commits read at most
DAYS = 14                                       # ... none made longer ago
NOTE = "Card sync could not draw its changed icons in the new icon font ({why}): update the text in Photoshop."
WHY_MAX = 150                                   # characters of a refusal's why in NOTE
MARK = "in the new icon font"                   # NOTE's: a card flagged so at its blob is not due
LAID = 2                                        # decimals of a pt two layouts' glyph places are compared at


@dataclass(frozen=True)
class Due:
    """A PSD due: the glyph names to draw again and the font blob it was drawn with."""
    glyphs: frozenset
    was: str


# ========================
# Git
# ========================
def _git(clone: str, *args: str) -> str:
    return run_process(["git", "-C", clone, *args], capture_output=True, text=True, check=True,
                       env={**environ, "GIT_NO_LAZY_FETCH": "1"}).stdout


def font_commits(clone: str, head: str = "HEAD") -> list:
    """The workflow's font commits (bot, FONT_SUBJECT) on head, newest first, font commits made otherwise passed over:
    FONTS_BACK at most, none made DAYS ago or longer."""
    out = []
    for line in _git(clone, "log", "--format=%H%x09%an%x09%ct%x09%s", head, "--", FONT_FILE).splitlines():
        sha, author, when, subject = (line.split("\t", 3) + ["", "", ""])[:4]
        if time() - int(when or 0) > DAYS * 86400 or len(out) == FONTS_BACK:
            break
        if author == BOT and subject.startswith(FONT_SUBJECT):
            out.append(sha)
    return out


def _drawn(clone: str, oldest: str, head: str) -> tuple:
    """({PSD path at head: {commit}} the commits since oldest^ that drew it: not the bot's or a sweep's (Card-sweep) but
    a glyph redraw's; a pure move (an exact rename) draws nothing, the moved file's commits count for its new path;
    {older path: its path at head} of those moves)."""
    span = f"{oldest}^..{head}" if _git(clone, "rev-list", "--parents", "-1", oldest).split()[1:] else head
    log = _git(clone, "-c", "core.quotePath=false", "log", "--topo-order", "--name-status", "--find-renames=100%",
               "--format=%x00%H%x09%an%x09%(trailers:key=Card-sweep,valueonly,separator=%x2C)", span, "--", ":(icase)*.psd")
    drawn, moved = {}, {}                       # moved: an older path -> its path at head
    for chunk in log.split("\0")[1:]:           # newest first
        lines = [line for line in chunk.splitlines() if line.strip()]
        sha, author, sweeps = (lines[0].split("\t", 2) + ["", ""])[:3]
        kinds = {kind.strip() for kind in sweeps.split(",") if kind.strip()}
        draws = not ((author == BOT or kinds) and SWEEP not in kinds)
        for line in lines[1:]:
            status, *paths = line.split("\t")
            if status.startswith("R"):
                moved[paths[0]] = moved.get(paths[1], paths[1])
            elif draws and not status.startswith("D"):
                drawn.setdefault(moved.get(paths[-1], paths[-1]), set()).add(sha)
    return drawn, moved


def _since(clone: str, commit: str, head: str) -> set:
    """The commits on head that have commit in their history (git rev-list --ancestry-path)."""
    return set(_git(clone, "rev-list", "--ancestry-path", f"{commit}..{head}").split())


def _blob_at(clone: str, commit: str, path: str) -> str:
    try:
        return _git(clone, "rev-parse", f"{commit}:{path}").strip()
    except Exception:                           # not there
        return ""


# ========================
# The fonts
# ========================
@lru_cache(maxsize=8)
def _drawings(data: bytes) -> dict:
    """{glyph name: (advance and bearing, outline bytes, SVG picture)} (stoogedings.py's glyphs: by name, ids move)."""
    from fontTools.ttLib import TTFont
    font = TTFont(BytesIO(data))
    order, glyf, hmtx = font.getGlyphOrder(), font["glyf"], font["hmtx"].metrics
    pictures = {}
    if "SVG " in font:
        for doc in font["SVG "].docList:
            text, lo, hi = doc.data, doc.startGlyphID, doc.endGlyphID
            for gid in range(lo, hi + 1):
                pictures[gid] = sub(r'id="glyph\d+"', 'id="glyph"', text)
    return {name: (hmtx[name], glyf[name].compile(glyf), pictures.get(gid)) for gid, name in enumerate(order)}


def changed(old: bytes, new: bytes) -> set:
    """The glyph names two builds of the icon font draw otherwise, the new and the gone ones too."""
    a, b = _drawings(old), _drawings(new)
    return {name for name in set(a) | set(b) if a.get(name) != b.get(name)}


@lru_cache(maxsize=4)
def _shaper(data: bytes):
    """text -> HarfBuzz's glyphs in the font, (name, advance) each, by default and with salt (Stylistic Alternates)."""
    import uharfbuzz as hb
    font = hb.Font(hb.Face(hb.Blob(data)))

    @lru_cache(maxsize=None)
    def shape(text: str) -> tuple:
        out = []
        for features in ({}, {"salt": True}):
            buf = hb.Buffer()
            buf.add_str(text)
            buf.guess_segment_properties()
            hb.shape(font, buf, features)
            out.append(tuple((font.get_glyph_name(i.codepoint) or "", p.x_advance)
                             for i, p in zip(buf.glyph_infos, buf.glyph_positions)))
        return tuple(out)
    return shape


def _icon_texts(record: dict) -> list:
    """The visible text layers' texts a psd_details record sets in the icon font."""
    fonts = record.get("fonts") if isinstance(record.get("fonts"), dict) else {}
    layers = {path for name, entry in fonts.items() if str(name).lower().startswith(standards.ICON_FONT_PREFIX)
              for path in (entry or {}).get("layers") or []}
    return [t["text"] for t in record.get("text") or [] if isinstance(t, dict) and t.get("visible")
            and t.get("layer") in layers and t.get("text")]


def uses(texts: list, glyphs: set, was: bytes, now: bytes) -> bool:
    """Whether texts shape to one of glyphs in either font, or shape otherwise in the new one."""
    before, after = _shaper(was), _shaper(now)
    for text in texts:
        a, b = before(text), after(text)
        if a != b or any(name in glyphs for way in a + b for name, _ in way):
            return True
    return False


# ========================
# Card sync: what is due
# ========================
def flagged(cards: dict, rel: str, blob: str) -> bool:
    """Whether Card Edits.json's cards (CardEdits.cards) flag the card for a glyph refusal at this PSD blob."""
    found = (cards.get(CardAuthors.key(card_name(rel))) or {}).get("flags") or {}
    return found.get("psd") == blob[:8] and MARK in (found.get("hand") or "")


def due(clone: str, head: str, tree: dict, details: Optional[dict], edits: Optional[bytes] = None,
        fetch: bool = True) -> dict:
    """
    The PSDs whose icon glyphs Card sync draws again: drawn before one of font_commits (the oldest such commit F counts:
    the glyphs it and the later ones changed that head's font still draws otherwise than F's parent), shaping such a
    glyph or shaping otherwise (psd_details' visible texts set in the icon font: the record at the PSD's blob, else its
    path's newest), not flagged for it (flagged).

    Args:
        tree: {repo path: blob} at head
        details: db/psd_details.json
        fetch: The font blobs the clone lacks fetched first (one batch)

    Returns:
        {PSD path: Due}
    """
    commits = font_commits(clone, head)
    if not commits or FONT_FILE not in tree or not details:
        return {}
    from commands.psd_links import _git_bytes, fetch_blobs
    olds = {sha: (_blob_at(clone, f"{sha}^", FONT_FILE), _blob_at(clone, sha, FONT_FILE)) for sha in commits}
    blobs = list(dict.fromkeys([tree[FONT_FILE]] + [b for pair in olds.values() for b in pair if b]))
    if fetch:
        fetch_blobs(clone, blobs)
    got = _git_bytes(clone, blobs)
    now = got.get(tree[FONT_FILE])
    if now is None or not all(got.get(b) for pair in olds.values() for b in pair):
        return {}
    drawn, moved = _drawn(clone, commits[-1], head)
    fonts, own = [], set()                      # (commit, was blob, was bytes, after it, glyphs), oldest first
    for n, sha in enumerate(commits):           # newest first: each one's own changes, and the later ones'
        was, made = olds[sha]
        own |= changed(got[was], got[made])
        fonts.insert(0, (sha, was, got[was], _since(clone, sha, head), frozenset(own & changed(got[was], now))))
    cards = CardEdits.parse(edits.decode("utf-8")).cards if edits else {}
    records, by_path = (details.get("cards") or {}), {}
    for blob, record in records.items():                       # by path at head (a move since followed)
        if isinstance(record, dict) and record.get("path"):
            by_path.setdefault(moved.get(record["path"], record["path"]), []).append((record.get("at") or "", blob))
    out = {}
    for rel in sorted(p for p in tree if p.lower().endswith(".psd")):
        commits = drawn.get(rel, set())         # the oldest font commit none of its drawings has in its history
        since = next((f for f in fonts if not commits & f[3]), None)
        if since is None or not since[4]:
            continue
        record = records.get(tree[rel]) or (records.get(max(by_path[rel])[1]) if rel in by_path else None)
        texts = _icon_texts(record) if isinstance(record, dict) else []
        if texts and uses(texts, since[4], since[2], now) and not flagged(cards, rel, tree[rel]):
            out[rel] = Due(since[4], since[1])
    return out


# ========================
# Card sync: one PSD
# ========================
def _laid(lay) -> tuple:
    """A layout as two icon font builds' compare: each glyph's cluster, name (icon font) or id, place and size, the
    lines' spans."""
    return (tuple((*standards._glyph_key(g), round(g.x, LAID), round(g.y, LAID), round(g.em_w, LAID), round(g.advance, LAID))
                  for g in lay.glyphs), tuple((line.start, line.end) for line in lay.lines))


def _marks(lay, glyphs: frozenset) -> list:
    return [i for i, g in enumerate(lay.glyphs)
            if standards._icon_face(g.face) and (g.face.hb.get_glyph_name(g.gid) or "") in glyphs]


def _relaid(layer, faces, found: Due, was_path: str) -> bool:
    """Whether the layer's pixels show the new font's layout (laid out again since the font commit): its ink beside the
    changed glyphs within card_standards.LAID_OFF of the new font's render and nearer it than the old font's."""
    off = []
    for fonts in (faces, standards._Swapped(faces, was_path)):
        drawn = standards._glyph_offs(layer, fonts)
        off.append(1.0 if drawn is None else standards._off_beside(drawn, set(_marks(drawn[0].layout, found.glyphs))))
    return off[0] <= standards.LAID_OFF and off[0] < off[1]


def redraw(path: str, out_path: str, fonts_dir: str, found: Due, was_path: str) -> list:
    """
    found's glyphs drawn again in the visible icon font layers with the font in fonts_dir, those glyphs alone (a splice,
    refused when it would redraw a layer whole), gated (card_standards._only_pixels). A layer that lays out otherwise in
    the font found.was (was_path) and in the new one (an icon of another width, a spelling that is an icon now or
    letters again) is spliced only when its pixels show the new layout already (_relaid), else Refused (Photoshop
    would rewrap it).

    Returns:
        The layer names redrawn ([]: nothing written); raises Refused or the engine's errors
    """
    from psd_tools.composite.text import capture
    from psd_tools.composite.text.layout import layout
    from psd_tools.composite.text.model import TextModel
    psd = PSDImage.open(path)
    fonts = lambda names: fonts_dir             # noqa: E731
    records, stories, names = set(), [], []
    for layer in standards.icon_layers(psd):
        faces = standards._fonts_of(layer, fonts)
        model = TextModel.from_layer(layer)
        now, then = layout(model, faces), layout(model, standards._Swapped(faces, was_path))
        name = layer.name.strip()
        if _laid(now) != _laid(then) and not _relaid(layer, faces, found, was_path):
            raise Refused("unsupported-card", f"{name} lays out otherwise")
        if not _marks(now, found.glyphs):
            continue
        psd_edit.follow_kerning(psd, layer)     # B12: drawn as Photoshop kerns it (the story)
        keep = capture(layer, faces)
        marks = _marks(keep.layout, found.glyphs)
        glyphs = tuple(replace(g, fill=standards.STALE_FILL) if i in marks else g for i, g in enumerate(keep.layout.glyphs))
        standards._by_glyph(layer, layer.rasterize(fonts=faces, keep=replace(keep, layout=replace(keep.layout, glyphs=glyphs)),
                                                   unit="glyph"))
        records.add(psd_edit._record_index(layer))
        stories.append(psd_edit.story_index(layer))
        names.append(name)
    if not names:
        return []
    psd_edit.save_psd(psd, out_path)
    standards._only_pixels(path, out_path, records, stories)
    return names


def note(error: Exception) -> str:
    """NOTE for a refusal or an engine error."""
    why = getattr(error, "why", None) or f"{type(error).__name__}: {error}"
    return NOTE.format(why=why if len(why) <= WHY_MAX else why[:WHY_MAX - 3] + "...")


def flag(text: str, rel: str, blob: str, note: str, cleared: Optional[dict] = None) -> str:
    """
    Card Edits.json with the card's hand flag at its PSD blob: note first (flagged reads it), then a person's reason
    beside it, cut to FLAG_MAX characters; a person's flag set on another blob is kept beside it unless it went
    (cleared: db/card_checks.json's, {PSD path: flags' psd}). The text as it was when nothing changes.
    """
    data = loads(text)
    cards = data.setdefault("cards", {})
    key = next((k for k in cards if CardAuthors.key(k) == CardAuthors.key(card_name(rel))), card_name(rel))
    entry = dict(cards.get(key) or {})
    flags = dict(entry.get("flags") or {}) if isinstance(entry.get("flags"), dict) else {}
    if flags.get("psd") and flags["psd"] != blob[:8] and (cleared or {}).get(rel) == flags["psd"]:
        flags = {}                              # a person's flag that went
    theirs = " ".join(part for part in sub(r"(?<=[.!?])\s+", "\n", flags.get("hand") or "").splitlines()
                      if part.strip() and MARK not in part)
    room = FLAG_MAX - len(note) - 1
    hand = note[:FLAG_MAX] + (" " + (theirs if len(theirs) <= room else theirs[:max(room - 3, 0)] + "...")
                              if theirs and room > 3 else "")
    new = {**{k: v for k, v in flags.items() if k not in ("hand", "psd")}, "hand": hand, "psd": blob[:8]}
    if new == flags:
        return text
    cards[key] = rewrite_entry({**entry, "flags": new})
    return dump_edits(data)


# ========================
# Module Exports
# ========================
__all__ = [
    'BOT',
    'Due',
    'FONT_FILE',
    'FONT_SUBJECT',
    'MARK',
    'NOTE',
    'SWEEP',
    'changed',
    'due',
    'flag',
    'flagged',
    'font_commits',
    'note',
    'redraw',
    'uses',
]
