"""
Guard 2 of the push guards (docs/Campaigns/PUSH_GUARDS.md section 6): a pushed card PSD against the card standards of
docs/Campaigns/CARD_STANDARDIZATION_CAMPAIGN.md. Rules R1-R9 and R11 run over an open PSD (R10, the PNG against the PSD, is
png_export.png_check). R3 and R4 keep only the real issues (several visible Ability layers, or a hidden one with words
the parser reads nowhere; a hidden or rasterized Name layer the title can't be read from): the template layout they also
checked was dropped by the owner on 2026-10-23.
R9 (Txt2 stories) is fixed by the fork with byte gates, the rest is reported. R11's rename (exact layer names) is built
and gated but reported until the owner says yes (AUTO_FIX). Rule pages are never checked.

    check(psd, rel, context, old=None, added=False) -> [Issue]
    fix(data, rel, issues, work, parser=None) -> Fixed            the 0-pixel fixes, gated (one PSD's new bytes)
    BADGE_TAGS, WAIVED                                            the badge vocabulary and the owner's class waivers

No Discord, network or config.
"""
from dataclasses import dataclass, field
from os import makedirs
from os.path import basename, getsize, join
from re import compile as re_compile
from shutil import rmtree
from tempfile import mkdtemp
from typing import Optional

from psd_tools import PSDImage
from psd_tools.constants import Tag

import commands.psd_edit as psd_edit
from commands.card_edits import CardAuthors
from commands.font_check import FontIndex, _get, _value, check_psd
from commands.psd_edit import Refused
from commands.psd_parser import EFFECT_LAYER_TAG, ICON_FONT_PREFIX, PSDParser
from commands.psd_regions import overlay_words
from global_config import RULEBOOK_DIR

# ========================
# Configuration
# ========================
def badge_tags() -> list:
    """Every timing tag the stoogedings font draws as a badge: TTSCardMaker build_stoogedings.badge_tags(), mirrored."""
    nums, firsts = "123456789X", ["-1", *"0123456789", "X"]
    tags = ["OYT", "AAT", "BOT", "ENDT", "BOGT", "ENDGT", "OFS", "OS", "OD", "WOF", "Action"]
    for n in nums:                              # the old system (rule pages)
        tags += [f"{n}PT", f"E{n}T", f"FTE{n}T", f"{n}PGT", f"E{n}GT", f"{n}PG"]
    for n in nums:                              # cooldowns and uses (2026-10-21)
        tags += [f"{n}CD", f"{n}GCD", f"{n}U"]
    for cd in ("CD", "GCD"):
        tags += [f"{a}/{b}{cd}" for a in firsts for b in nums]
    return tags


BADGE_TAGS = tuple(badge_tags())
BADGE_KEYS = {tag.lower(): tag for tag in BADGE_TAGS}   # the font matches any letter case (TTSCardMaker 6e50c3ee)
PAREN = re_compile(r"\(([^()\r\n\x03]{1,12})\)")
SCOPE = ("creature", "aux creature", "minion", "item", "aux item", "nme")   # the ledger's 651
STARS_KINDS = ("minion", "aux item", "nme")     # stars drawn from a Stars group (L14)
EXACT = {"ability": "Ability", "name": "Name"}  # R11: Text/Ability, Text/Name
FIXABLE = ("R9", "R11")                         # 0-pixel by construction, gated: what fix() can do
AUTO_FIX = ("R9",)                              # what the bot fixes: the owner's "easy fixes only (today only R9 Txt2
                                                # sync)", 2026-10-22; R11 is told until the owner adds it
SIZE_MAX = 0.01                                 # a fixed PSD stays within 1% of the pushed one
NAME_BLOCK = bytes(Tag.UNICODE_LAYER_NAME)      # luni
WAIVED = {                                      # the owner's class waivers (section 6.2): rule -> what and why
    "R2": {"cards": {"tainted lazarus"}, "why": "its header icon (2026-10-22)"},
    "R3": {"cards": {"lit bob-omb", "tainted lazarus", "miracle matter", "twin emperors", "warcraftian druid", "pix"},
           "why": "several-Ability and multi-creature cards stay deviations (question 2, 2026-10-21)"},
    "R4": {"why": "existing Name layers are left as they are (owner, 2026-10-22); new cards and regressions are reported"},
    "R5": {"why": "overlay words with no chart style stay (decision 2026-10-21, L7)"},
    "R7": {"fonts": {"BabaUI"}, "why": "BabaUI is deferred by the owner (L13)"},
}


# ========================
# Data
# ========================
@dataclass
class Issue:
    """One standards finding on one PSD."""
    code: str                                   # R1..R11
    text: str                                   # for card makers: "the Name layer is hidden"
    items: tuple = ()                           # what it is about (tags, layer names): "(new)" compares these
    fix: bool = False                           # AUTO_FIX and fixable here
    waived: str = ""                            # the waiver's reason
    new: bool = False                           # absent from the old blob
    discord: bool = True                        # told in the standards post (R7 too: owner, 2026-10-22); False: comment only

    def line(self) -> str:
        return self.text + (" (new)" if self.new else "")

    def as_dict(self) -> dict:
        return {"code": self.code, "text": self.text, "items": list(self.items), "fix": self.fix, "waived": self.waived,
                "new": self.new}


@dataclass
class Context:
    """What the rules need beside the PSD."""
    types: frozenset = frozenset()              # Types/ stems, lowercased (card_check.Vocabulary.all_types)
    fonts: Optional[FontIndex] = None           # MDW/Fonts (R7 skipped without)
    chart: Optional[dict] = None                # psd_edit.status_fills of the status chart (R5 reports every overlay without)
    story_sync: bool = False                    # the fork can sync Txt2 stories (R9 is a fix, else a report)


@dataclass
class Fixed:
    """fix()'s result for one PSD."""
    data: bytes = b""                           # the fixed PSD ("" when nothing was fixed)
    applied: list = field(default_factory=list) # Issue lines fixed
    refused: list = field(default_factory=list) # (code, why) a gate refused


# ========================
# Helper Functions
# ========================
def kind_of(rel: str) -> str:
    """The card's class from its repo path."""
    if RULEBOOK_DIR in rel.split("/"):
        return "rule"
    for prefix, kind in (("Auxiliary/Status Effects/", "status"), ("Auxiliary/Minions/", "minion"),
                         ("Auxiliary/Items/", "aux item"), ("Auxiliary/Creatures/", "aux creature"), ("N.M.E/", "nme"),
                         ("Field/", "field"), ("Items/", "item"), ("Creatures/", "creature")):
        if rel.startswith(prefix):
            return kind
    return "other"


def card_key(rel: str) -> str:
    """The card's key as Card Authors.txt and Card Edits.json match it."""
    stem = basename(rel)
    return CardAuthors.key(stem[:len(stem) - 4])


def _visible_texts(psd: PSDImage) -> list:
    return [layer for layer in psd.descendants() if layer.kind == "type" and layer.is_visible()]


def run_fonts(layer) -> tuple:
    """(text, [(start, end, PostScript font)]) in str indices over the layer's engine text."""
    resources, engine = layer.resource_dict, layer.engine_dict
    font_set = [str(_value(_get(font, "Name", ""))) for font in _get(resources, "FontSet", ())]
    text = str(_value(engine["Editor"]["Text"]))
    encoded = text.encode("utf-16-le")
    out, unit, pos = [], 0, 0
    runs = _get(engine, "StyleRun")
    for data, length in zip(_get(runs, "RunArray", ()), _get(runs, "RunLengthArray", ())):
        n = int(_value(length))
        piece = encoded[2 * unit:2 * (unit + n)].decode("utf-16-le", "ignore")
        sheet = _get(_get(data, "StyleSheet"), "StyleSheetData")
        index = int(_value(sheet.get("Font"))) if sheet is not None and sheet.get("Font") is not None else 0
        out.append((pos, pos + len(piece), font_set[index] if 0 <= index < len(font_set) else "?"))
        pos += len(piece)
        unit += n
    return text, out


def every_line(layer) -> bool:
    """Whether any paragraph of the layer uses the Every-line Composer."""
    try:
        for run in layer.engine_dict["ParagraphRun"]["RunArray"]:
            value = run["ParagraphSheet"]["Properties"].get("EveryLineComposer")
            if value is not None and bool(getattr(value, "value", value)):
                return True
    except (KeyError, TypeError, AttributeError):
        pass
    return False


def _unique(items) -> tuple:
    return tuple(dict.fromkeys(items))


def _list(items, quote: bool = False) -> str:
    items = [f'"{item}"' if quote else item for item in items]
    return items[0] if len(items) == 1 else ", ".join(items[:len(items) - 1]) + " and " + items[len(items) - 1]


# ========================
# The rules
# ========================
def rule_tags(psd: PSDImage) -> Optional[Issue]:
    """R1: timing tags in a visible Ability layer set in the stoogedings badge font (L1)."""
    bad = []
    for layer in _visible_texts(psd):
        if layer.name.strip().lower() != "ability":
            continue
        try:
            text, runs = run_fonts(layer)
        except (KeyError, IndexError, TypeError, ValueError, AttributeError):
            continue
        for match in PAREN.finditer(text):
            if match.group(1).lower() not in BADGE_KEYS:
                continue
            fonts = {font for start, end, font in runs if start < match.end() and end > match.start()}
            if not fonts or any(not font.lower().startswith(ICON_FONT_PREFIX) for font in fonts):
                bad.append(match.group(0))
    if not bad:
        return None
    tags = _unique(bad)
    return Issue("R1", f"timing tag{'s' if len(tags) > 1 else ''} {_list(tags)} not set in the stoogedings badge font", tags)


def rule_icons(psd: PSDImage, types) -> Optional[Issue]:
    """R2: no Types/ icon image layers below the midline (the parser's inline-icon rule; stat bars skipped)."""
    mid = int(psd.height * 0.5)
    found = []
    for layer in psd.descendants():
        if layer.kind in ("type", "group") or EFFECT_LAYER_TAG in layer.name:
            continue
        if (layer.name.lower() in types and layer.is_visible() and layer.has_pixels()
                and not PSDParser._is_stat_layer(None, layer) and layer.bbox[1] >= mid):
            found.append(layer.name.strip().lower())
    if not found:
        return None
    names = _unique(found)
    return Issue("R2", f"icon image layer{'s' if len(names) > 1 else ''} in the ability ({_list(names)}): use "
                       f"{_list(['[' + n + ']' for n in names])} in the stoogedings font", names)


def _words(layer) -> list:
    """A text layer's words, split as PSDParser._relaid splits them."""
    return (layer.text or "").replace("\x03", " ").replace("\ufeff", "").split()


def rule_ability(psd: PSDImage) -> Optional[Issue]:
    """
    R3: one visible Ability text layer, in any group (what psd_edit.ability_layer bakes). Several are told; none only
    when a hidden Ability layer holds words the parser reads nowhere. An empty one (a card with no ability), one laid
    out again in visible pieces (PSDParser._relaid) or none at all (the ability named otherwise) is the developer's.
    """
    named = [layer for layer in psd.descendants()
             if psd_edit._is_text_layer(layer) and layer.name.strip().lower() == "ability"]
    count = sum(1 for layer in named if layer.is_visible())
    if count > 1:
        return Issue("R3", f"{count} visible Ability layers (one is the standard)", (f"{count} visible Ability layers",))
    if count or not any(_words(layer) for layer in named):
        return None
    texts = [layer for layer in psd.descendants() if psd_edit._is_text_layer(layer) and EFFECT_LAYER_TAG not in layer.name]
    if any(PSDParser._relaid(layer, texts) for layer in named):
        return None
    return Issue("R3", "the Ability layer is hidden (the bot reads no ability)", ("hidden ability layer",))


def title_problem(psd: PSDImage, card: str) -> Optional[str]:
    """
    Why no title can be read, when a Name layer is the cause: "hidden name layer" or "rasterized name layer". None when
    the parser reads a title (PSDParser.title_layers), or when no layer is named Name (the title's layout is the
    developer's, owner 2026-10-23).
    """
    if PSDParser.title_layers(psd, [layer for layer in psd.descendants() if psd_edit._is_text_layer(layer)], card):
        return None
    named = [layer for layer in psd.descendants() if not layer.is_group() and layer.name.strip().lower() == "name"]
    if not named or any(psd_edit._is_text_layer(layer) and layer.is_visible() for layer in named):
        return None
    if any(not psd_edit._is_text_layer(layer) for layer in named):
        return "rasterized name layer"
    return "hidden name layer"


def rule_name(psd: PSDImage, card: str) -> Optional[Issue]:
    """R4: the title is not left in a hidden or rasterized Name layer (no title read)."""
    why = title_problem(psd, card)
    if why is None:
        return None
    return Issue("R4", {"hidden name layer": "the Name layer is hidden",
                        "rasterized name layer": "the Name layer is rasterized"}[why], (why,))


def rule_overlays(psd: PSDImage, chart: Optional[dict]) -> list:
    """R5: status words styled by Ability FX: layers, not overlay text layers (L7); chart-less ones are waived."""
    report, waived = [], []
    for layer in overlay_words(psd):
        word = psd_edit.collapse(layer.text)
        styled = chart is None or getattr(chart.get(word.lower()), "styled", False)
        (report if styled else waived).append(word)
    out = []
    if report:
        words = _unique(report)
        out.append(Issue("R5", f"{_list(words, quote=True)} {'is an overlay text layer' if len(words) == 1 else 'are overlay text layers'}, "
                               f"not an Ability FX: layer", words))
    if waived:
        words = _unique(waived)
        out.append(Issue("R5", f"overlay text {_list(words, quote=True)} (no chart style)", words, waived=WAIVED["R5"]["why"]))
    return out


def rule_stars(psd: PSDImage) -> Optional[Issue]:
    """R6: minions, auxiliary items and N.M.E. items draw their stars from a visible group named ... stars (L14)."""
    if PSDParser._star_groups(psd):
        return None
    return Issue("R6", "the stars are not in a group named Stars", ("no stars group",))


def rule_fonts(psd: PSDImage, fonts: FontIndex) -> Optional[Issue]:
    """R7: every font a visible text layer asks for is in MDW/Fonts (font_check)."""
    missing = check_psd(psd, fonts)
    if not missing:
        return None
    names = tuple(sorted(missing))
    issue = Issue("R7", f"font{'s' if len(names) > 1 else ''} {_list(names)} not in MDW/Fonts", names)
    if set(names) <= WAIVED["R7"]["fonts"]:
        issue.waived = WAIVED["R7"]["why"]
    return issue


def rule_composer(psd: PSDImage) -> Optional[Issue]:
    """R8: the Single-line Composer on visible Ability and Name layers (L10)."""
    layers = _unique(layer.name.strip() for layer in _visible_texts(psd)
                     if layer.name.strip().lower() in ("ability", "name") and every_line(layer))
    if not layers:
        return None
    return Issue("R8", f"{_list(layers)} use{'s' if len(layers) == 1 else ''} the Every-line Composer; set Single-line", layers)


def rule_stories(psd: PSDImage, story_sync: bool) -> Optional[Issue]:
    """R9: every type layer's Txt2 story holds its text (Photoshop shows the story's)."""
    stale = psd_edit.stale_stories(psd)
    if not stale:
        return None
    names = _unique(name for name, _, _ in stale)
    return Issue("R9", f"Photoshop would show old text in {_list(names)} (Txt2 story out of date)", names,
                 fix=story_sync and "R9" in AUTO_FIX)


def inexact_names(psd: PSDImage) -> list:
    """R11's renames: [(layer, exact name)] for Text/Ability, Text/Name and the FX layers keyed on an inexact base."""
    group = psd_edit._text_group(psd)
    out, bases = [], {}
    for layer in (group or []):
        target = EXACT.get(layer.name.strip().lower())
        if layer.kind == "type" and target and layer.name != target:
            out.append((layer, target))
            bases[layer.name] = target
    for layer in psd.descendants():
        base, tag, rest = layer.name.partition(EFFECT_LAYER_TAG)
        if tag and base in bases:
            out.append((layer, bases[base] + tag + rest))
    return out


def rule_names(psd: PSDImage) -> Optional[Issue]:
    """R11: Text/Ability, Text/Name and their FX layers named exactly so (case, outer spaces)."""
    renames = inexact_names(psd)
    if not renames:
        return None
    pairs = tuple(f"{layer.name!r} -> {target!r}" for layer, target in renames)
    return Issue("R11", f"layer names to correct: {', '.join(pairs)}", pairs, fix="R11" in AUTO_FIX)


# ========================
# Public API Functions
# ========================
def check(psd: PSDImage, rel: str, context: Context, old: Optional[PSDImage] = None, added: bool = False) -> list:
    """
    The standards issues of one card PSD.

    Args:
        psd: The pushed PSD
        rel: Its repo path (the class: scope, waivers)
        context: Types/, fonts, chart
        old: The PSD before the push (R4 regressions; "(new)" marks)
        added: A new card (R4 applies in full)

    Returns:
        [Issue] in rule order, waived ones included (waived set); [] for a rule page
    """
    kind = kind_of(rel)
    if kind == "rule":
        return []
    key = card_key(rel)
    out: list = []

    def run(code: str, rule, *args) -> None:
        try:
            found = rule(*args)
        except Exception as error:              # one unreadable part never stops the other rules
            found = Issue("error", f"{code} could not be checked ({type(error).__name__}: {error})", (code,), discord=False)
        out.extend(found if isinstance(found, list) else [found])

    run("R1", rule_tags, psd)
    run("R2", rule_icons, psd, context.types)
    if kind in SCOPE:
        stem = basename(rel)
        card = stem[:len(stem) - 4]
        run("R3", rule_ability, psd)
        run("R4", rule_name, psd, card)
        name = out[len(out) - 1]
        if name is not None and name.code == "R4" and not added:
            try:
                regression = old is not None and rule_name(old, card) is None
            except Exception:                   # an unreadable old blob: the existing card's waiver
                regression = False
            if not regression:
                name.waived = WAIVED["R4"]["why"]
    run("R5", rule_overlays, psd, context.chart)
    if kind in STARS_KINDS:
        run("R6", rule_stars, psd)
    if context.fonts is not None:
        run("R7", rule_fonts, psd, context.fonts)
    run("R8", rule_composer, psd)
    run("R9", rule_stories, psd, context.story_sync)
    if kind in SCOPE:
        run("R11", rule_names, psd)
    issues = [issue for issue in out if issue is not None]
    for issue in issues:
        waiver = WAIVED.get(issue.code, {})
        if not issue.waived and key in waiver.get("cards", ()):
            issue.waived = waiver["why"]
    if old is not None:
        mark_new(issues, check(old, rel, context))
    return issues


def mark_new(issues: list, before: list) -> None:
    """Set new on the issues with an item the old blob's issues of the same rule lack."""
    seen: dict = {}
    for issue in before:
        seen.setdefault(issue.code, set()).update(issue.items)
    for issue in issues:
        issue.new = bool(set(issue.items) - seen.get(issue.code, set()))


def reported(issues: list) -> list:
    """The issues told to the author: not waived, not fixed by the bot."""
    return [issue for issue in issues if not issue.waived and not issue.fix]


def _records_except_names(path: str) -> list:
    """layer_records with the name and luni left out (R11's byte proof)."""
    out = []
    for record in psd_edit.layer_records(path):
        out.append({**{k: v for k, v in record.items() if k != "name"},
                    "blocks": tuple((key, data) for key, data in record["blocks"] if key != NAME_BLOCK)})
    return out


def _same_document(old: PSDImage, new: PSDImage) -> Optional[str]:
    """The part of the document other than the layer records that changed, or None."""
    for part in ("header", "color_mode_data", "image_resources", "image_data"):
        if psd_edit._raw(getattr(old._record, part)) != psd_edit._raw(getattr(new._record, part)):
            return part
    if psd_edit._document_blocks(old) != psd_edit._document_blocks(new):
        return "document blocks"
    return None


def rename(path: str, out_path: str) -> list:
    """
    R11: rename the inexact layers and save without a new preview; gated: only the renamed records' names change.

    Returns:
        The renames ("old -> new"); raises Refused("bytes")
    """
    psd = PSDImage.open(path)
    renames = inexact_names(psd)
    if not renames:
        return []
    indices = set()
    records = psd._record.layer_and_mask_information.layer_info.layer_records
    for layer, target in renames:
        indices.add(next(i for i, record in enumerate(records) if record is layer._record))
        layer.name = target
    psd_edit.save_psd(psd, out_path, update_preview=False)
    old_records, new_records = psd_edit.layer_records(path), psd_edit.layer_records(out_path)
    if len(old_records) != len(new_records):
        raise Refused("bytes", "layer count changed")
    names = [i for i, (a, b) in enumerate(zip(old_records, new_records)) if a["name"] != b["name"]]
    if set(names) != indices:
        raise Refused("bytes", f"renamed records {names}, expected {sorted(indices)}")
    if _records_except_names(path) != _records_except_names(out_path):
        raise Refused("bytes", "a layer record changed beyond its name")
    part = _same_document(PSDImage.open(path), PSDImage.open(out_path))
    if part:
        raise Refused("bytes", f"{part} changed")
    return [f"{a['name']!r} -> {b['name']!r}" for i, (a, b) in enumerate(zip(old_records, new_records)) if i in indices]


def _fix_in(folder: str, data: bytes, rel: str, wanted: list, parser: Optional[PSDParser], result: Fixed) -> None:
    """fix()'s work in its own temporary folder."""
    from commands.txt2_repair import repair
    base = join(folder, "pushed.psd")
    with open(base, "wb") as fh:
        fh.write(data)
    current, codes = base, []
    for issue in sorted(wanted, key=lambda i: FIXABLE.index(i.code)):
        out = join(folder, f"{issue.code}.psd")
        try:
            if issue.code == "R11":
                if not rename(current, out):
                    continue
            else:
                done = repair(current, out)
                if not done.out:
                    continue
        except (Refused, ImportError, ValueError) as error:
            result.refused.append((issue.code, str(error)))
            continue
        current = out
        codes.append(issue.code)
        result.applied.append(issue.line())
    if current == base:
        return
    try:
        if abs(getsize(current) - getsize(base)) > SIZE_MAX * getsize(base):
            raise Refused("size", f"PSD {getsize(base)} -> {getsize(current)} bytes")
        if parser is not None:
            before, after = psd_edit.parse_card(parser, base, rel), psd_edit.parse_card(parser, current, rel)
            if (before["card"], before["alt_names"], before["validator"]) != (after["card"], after["alt_names"], after["validator"]):
                raise Refused("reparse", "the bot reads the fixed PSD differently")
        with open(current, "rb") as fh:
            result.data = fh.read()
    except Refused as refused:
        result.refused += [(code, str(refused)) for code in codes]
        result.applied, result.data = [], b""


def fix(data: bytes, rel: str, issues: list, work: str, parser: Optional[PSDParser] = None) -> Fixed:
    """
    The fixable issues of one PSD (R11 renames, then R9 story syncs), each gated; then the size and the reparse gates
    over the result. A refusal leaves that fix out (and reports it); nothing is fixed on a rule page.

    Args:
        data: The pushed PSD
        rel: Its repo path
        issues: check()'s issues
        work: A folder for the temporary files
        parser: The offline parser for gate_reparse (skipped without)

    Returns:
        Fixed
    """
    result = Fixed()
    wanted = [issue for issue in issues if issue.fix and not issue.waived]
    if not wanted or kind_of(rel) == "rule":
        return result
    makedirs(work, exist_ok=True)
    folder = mkdtemp(prefix="fix-", dir=work)   # one per call: cards of one key in two folders never share a file
    try:
        _fix_in(folder, data, rel, wanted, parser, result)
    finally:
        rmtree(folder, ignore_errors=True)
    return result


# ========================
# Module Exports
# ========================
__all__ = [
    'AUTO_FIX',
    'BADGE_KEYS',
    'BADGE_TAGS',
    'Context',
    'FIXABLE',
    'Fixed',
    'Issue',
    'SCOPE',
    'STARS_KINDS',
    'WAIVED',
    'badge_tags',
    'card_key',
    'check',
    'every_line',
    'fix',
    'inexact_names',
    'kind_of',
    'mark_new',
    'rename',
    'reported',
    'run_fonts',
    'title_problem',
]
