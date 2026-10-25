"""
The PSD parser: what a card's layers say (PSDParser), how its path classifies it (CardClassifier), the icon-font
vocabulary (IconVocabulary) and the validator (CardValidator). No Discord, network or pandas, so the diagnostics
scripts and the tests can use it on its own.
"""
from bisect import bisect_left
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from logging import getLogger, CRITICAL
from os import PathLike
from psd_tools import PSDImage
from re import compile as re_compile, fullmatch, sub
from typing import Any, Optional, Union

from global_config import RULEBOOK_DIR
from commands.card_models import CardInfo, CardStats, CardType, StatsDatabase, card_name

getLogger("psd_tools").setLevel(CRITICAL)

# ========================
# Configuration
# ========================
PARSER_VERSION = 3                # bump when every stored card must be re-parsed (the build re-parses older stamps)
TYPE_REGION_RATIO = 0.5
MISSPELT_CARD_TYPES = ['undread', 'tornado', 'error']
DRAWN_LAYER_KINDS = ("pixel", "smartobject", "group", "type", "shape")   # every other kind is an adjustment layer

# stoogedings is the icon font (TTSCardMaker MDW/Fonts). A token such as [fire] set in it renders as the icon,
# so the text itself says which icon sits where. Cards with image-layer icons keep using gap injection.
ICON_FONT_PREFIX = "stoogedings"                                  # PostScript name: stoogedings-Regular
TITLE_REGION_RATIO = 0.12                                         # text above this: the title
ICON_VARIANTS = ("", "tint", "sticker", "no", "glow", "darkglow")   # all but no read as the plain icon
ICON_SCAN_PATTERN = re_compile(r"\[[^\[\]\r\n\x03]{1,52}\]|[-★]")   # 52: [thunderfury blessed blade of the starseeker:darkglow]
EFFECT_LAYER_TAG = " FX: "        # "Ability FX: Blazing | Frozen In Time", made by MDW/Scripts/Effect Layers.jsx
EFFECT_COVERAGE_MIN = 0.95        # share of an effect layer's pixels that must sit on its text layer's pixels
STAR_CORE_ALPHA = 0.5             # a star's core: its pixels at least this opaque (glows and flares fade out wide)
# folders whose stars are counted on the card (elsewhere the folder says them); Auxiliary/Creatures print 2-5 too
STARS_FROM_PSD = ("Auxiliary/Items", "Auxiliary/Minions", "Auxiliary/Creatures", "N.M.E")
# a variant tag: a layer or group named "... #2" (" #2,3") shows only in those pictures (PARTS_EDITOR.md §9)
TAG = re_compile(r"^(.*\S) #(\d{1,2}(?:,\d{1,2})*)$")
LINE_PAD = re_compile(r"(?m)^[^\S\n]{3,}(?=\S)")             # a line's leading layout padding


def stars_from_psd(relative_path: str) -> bool:
    """Whether a card's stars are counted from its PSD (STARS_FROM_PSD)."""
    return any(folder in relative_path for folder in STARS_FROM_PSD)


def layer_tag(name: str) -> tuple:
    """(base name, frozenset of variants) for a tagged name, else (name, None); FX layers are never tagged."""
    m = None if EFFECT_LAYER_TAG in name else TAG.match(name)
    return (m.group(1), frozenset(int(n) for n in m.group(2).split(","))) if m else (name, None)


def base_name(layer) -> str:
    """The layer's name without its variant tag: what every name rule compares."""
    return layer_tag(layer.name)[0]


def shown_in(layer, n: Optional[int]) -> bool:
    """Whether the layer shows in variant n: a tagged node if n is a tag (its eye ignored), an untagged one by its eye,
    and every ancestor too; n None: is_visible()."""
    if n is None:
        return layer.is_visible()
    while layer is not None and getattr(layer, "kind", "psdimage") != "psdimage":
        tags = layer_tag(layer.name)[1]
        if not (n in tags if tags is not None else layer.visible):
            return False
        layer = layer.parent
    return True


def variants_of(psd) -> list:
    """The PSD's pictures: [1] plus every tag."""
    found = {1}
    for layer in psd.descendants():
        tags = layer_tag(layer.name)[1]
        if tags:
            found |= tags
    return sorted(found)


@contextmanager
def variant_view(psd, n: Optional[int]):
    """Within: every tagged node's eye says whether it shows in variant n (is_visible() is shown_in(., n)); restored
    after. n None: nothing changes."""
    changed = []
    if n is not None:
        for layer in psd.descendants():
            tags = layer_tag(layer.name)[1]
            if tags is not None and layer.visible != (n in tags):
                changed.append((layer, layer.visible))
                layer.visible = n in tags
    try:
        yield psd
    finally:
        for layer, eye in changed:
            layer.visible = eye


def name_key(text: str) -> str:
    """A name's letters and digits, case folded: what a layer named after the card shares with the file stem."""
    return "".join(c for c in text.casefold() if c.isalnum())


def token_variant(piece: str) -> Optional[str]:
    """
    A [token]'s variant as the icon font ligates it, lowercased: "" for "[fire]", "sticker" for "[Fire:Sticker]"; None
    for one the font draws as its letters: an unknown variant, or a colon with none ("[fire:]").
    """
    _, colon, variant = piece[1:len(piece) - 1].partition(":")
    variant = variant.lower()
    return variant if variant in ICON_VARIANTS and (variant or not colon) else None


def title_text(text: Optional[str]) -> str:
    """A title as printed: no byte-order mark, control characters and runs of whitespace as one space, ends trimmed."""
    return sub(r"[\s\x00-\x1f]+", " ", (text or "").replace("\ufeff", "")).strip()


def named_after_text(layer) -> bool:
    """A type layer named after its own text (name_key alike), as Photoshop names a new one: 'snas.' printing 'snas.'."""
    key = name_key(base_name(layer))
    return bool(key) and key == name_key(layer.text or "")


@dataclass
class MutableValue:
    """A simple mutable wrapper for primitives."""
    value: Any


@dataclass
class BarRow:
    """One bar row: its visible digit layers and where it sits."""
    group: int                                   # id of the row's parent (the stat block)
    name: str                                    # row group name: "1", "Light", "Hp Dark"
    block: str = ""                              # stat block name: "hp", "def dark"
    outer: int = 0                               # id of the group around the block
    items: list = field(default_factory=list)    # visible (digit, layer kind)


@dataclass
class StatTracker:
    """Tracks whether a stat was found and its visible digit layers, per bar row (the layers' parent group)."""
    found: bool = False
    value: Optional[int] = None
    rows: dict = field(default_factory=dict)     # id(row group) -> BarRow

    @staticmethod
    def _row_value(items: list) -> tuple[int, bool]:
        """
        One row's value, and whether the row draws one layer per cell.

        - Darkening adjustment layers (the template's 0-9 exposure layers): the lowest visible digit, since that one
          already darkens the rest of the bar (Sans had "1" and "9" visible: 1, not 10).
        - One drawn layer per cell, named 1-10 (Blank Card, Dirty Bed and Tainted Lazarus's hearts): the lit cells
          (pixel layers) visible, or 10 minus the dark cells when only those are visible. Adding up the names made 55.
        - Otherwise the visible digits add up (value layers, plus any "0" decorations: Clots, False Knight).
        """
        digits = [digit for digit, _ in items]
        drawn = [kind for _, kind in items if kind in DRAWN_LAYER_KINDS]
        if len(items) > 1 and not drawn:
            return min(digits), False
        if len(items) > 1 and len(drawn) == len(items) and min(digits) >= 1:
            lit = sum(1 for kind in drawn if kind == "pixel")
            return (lit if lit else max(0, 10 - len(items))), True
        return sum(digits), False

    def total(self, rows: Optional[list] = None) -> Optional[int]:
        """
        The stat's value: rows add up (Royal Eradicator Main Cannon's attack is 10 + 2). Rows of cells drawn twice
        in one block (Tainted Lazarus's "Light" and "Hp Dark" hearts) count once. Empty rows are skipped.
        """
        groups: dict = {}
        for row in self.rows.values() if rows is None else rows:
            if row.items:
                groups.setdefault(row.group, []).append(self._row_value(row.items))
        if not groups:
            return None
        value = 0
        for found in groups.values():
            if len(found) > 1 and all(per_cell for _, per_cell in found):
                value += max(row_value for row_value, _ in found)
            else:
                value += sum(row_value for row_value, _ in found)
        return value

    def by_creature(self) -> dict:
        """Rows per creature (0 = shared): rows named "1", "2", ... (Pix), or a stat block repeated (Tainted Lazarus)."""
        outers: dict = {}
        for row in self.rows.values():
            if row.items and row.outer not in outers.setdefault(row.block.lower(), []):
                outers[row.block.lower()].append(row.outer)
        repeated = any(len(o) > 1 for o in outers.values())
        out: dict = {}
        for row in self.rows.values():
            if row.name.strip().isdigit():
                index = int(row.name)
            elif repeated and row.outer in outers.get(row.block.lower(), []):
                index = outers[row.block.lower()].index(row.outer) + 1
            else:
                index = 0
            out.setdefault(index, []).append(row)
        return out


@dataclass
class StatTrackers:
    """Container for all card stat trackers."""
    hp: StatTracker = field(default_factory=StatTracker)
    defense: StatTracker = field(default_factory=StatTracker)
    attack: StatTracker = field(default_factory=StatTracker)
    speed: StatTracker = field(default_factory=StatTracker)

    def creatures(self) -> Optional[list]:
        """
        Each creature's stats on a multi-creature card, else None. A stat only creature 1 has a value for is
        shared (Pix's HP); an empty row next to filled ones is a full bar (10). Uncertain: check on the website.
        """
        split = {key: getattr(self, key).by_creature() for key in ("hp", "defense", "attack", "speed")}
        filled = lambda rows: [r for r in rows if r.items]
        many = sum(1 for s in split.values() if len([i for i, rows in s.items() if i and filled(rows)]) >= 2)
        count = max((i for s in split.values() for i in s), default=0)
        if count < 2 or many < 2:
            return None
        out = []
        for i in range(1, count + 1):
            values = {}
            for key, s in split.items():
                tracker = getattr(self, key)
                others = [j for j, rows in s.items() if j > 1 and filled(rows)]
                if filled(s.get(i, [])):
                    values[key] = tracker.total(filled(s[i]))
                elif filled(s.get(0, [])):
                    values[key] = tracker.total(filled(s[0]))          # a shared row ("shared health")
                elif filled(s.get(1, [])) and not others:
                    values[key] = tracker.total(filled(s[1]))          # only creature 1 drawn: shared
                else:
                    values[key] = 10 if tracker.found else -1
            out.append(CardStats(**values))
        return out


@dataclass
class BoundingBox:
    """Represents a bounding box with coordinates."""
    x: int
    y: int

    @classmethod
    def from_tuple(cls, coords: tuple[int, int]) -> 'BoundingBox':
        """Create BoundingBox from coordinate tuple."""
        return cls(coords[0], coords[1])


@dataclass
class IconVocabulary:
    """Icon names the stoogedings font understands: Types/ icons, Types/Stars/ icons, their aliases and codepoints."""
    tokens: dict[str, tuple[str, str]] = field(default_factory=dict)   # accepted spelling -> (name, "type" | "star")
    codepoints: dict[str, str] = field(default_factory=dict)           # private-use character -> name

    @staticmethod
    def spellings(text: str) -> set[str]:
        """
        Every spelling the font turns into this icon: words joined by a space, _, - or nothing.
        Matching is exact apart from letter case, so a token the font would draw as plain letters
        ("fragile.equip", "fragile  equip") is not read as an icon.
        """
        words = [w for w in sub(r"[ _\-]+", " ", text.lower()).split(" ") if w]
        return {sep.join(words) for sep in (" ", "_", "-", "")}

    def add(self, name: str, kind: str, aliases: tuple = ()) -> None:
        """Register an icon under its own name and its aliases (first registration wins)."""
        for spelling in (name,) + tuple(aliases):
            for accepted in self.spellings(spelling):
                self.tokens.setdefault(accepted, (name, kind))

    def lookup(self, text: str) -> Optional[tuple[str, str]]:
        """(name, kind) for a token's text, e.g. "Fragile Equip" -> ("fragile equip", "type"); None if unknown."""
        return self.tokens.get(text.lower())

    @classmethod
    def build(cls, all_types: list[str], all_stars: list[str], manifest: Optional[dict]) -> 'IconVocabulary':
        """
        Build the vocabulary.

        Args:
            all_types: Types/ icon names (lowercased file stems, as used in card.types)
            all_stars: Types/Stars/ icon names (lowercased, spaces as underscores)
            manifest: Parsed stoogedings.json (aliases and codepoints), or None when the repo has none

        Returns:
            IconVocabulary
        """
        vocab = cls()
        present = set(all_types) | set(all_stars)
        for name, info in (manifest or {}).get("icons", {}).items():
            if present and name not in present:
                continue    # icon PNG renamed or deleted: the manifest keeps its codepoint, but the font no longer draws it
            vocab.add(name, info.get("kind", "type"), tuple(info.get("aliases", ())))
            if info.get("pua"):
                vocab.codepoints[chr(int(info["pua"], 16))] = name
        for name in all_types:
            vocab.add(name, "type")
        for name in all_stars:
            vocab.add(name, "star")
        star = vocab.lookup("star")
        if star is not None:
            vocab.codepoints.setdefault("★", star[0])   # the font draws ★ as the gold star
        return vocab


class CardClassifier:
    """Classifies cards based on their path."""
    @staticmethod
    def classify(relative_path: str) -> dict:
        """
        Classify card based on its relative path in repository.

        Args:
            relative_path: Relative path to card file

        Returns:
            Dictionary with card classification info (UNKNOWN when the path is too shallow for its folder's rule)

        Raises:
            ValueError: A star folder that doesn't start with a number
        """
        if not relative_path:
            return {"type": CardType.UNKNOWN.value}

        folders = relative_path.split('/')
        folders.pop()                                 # the file name
        if not folders:
            return {"type": CardType.UNKNOWN.value}

        top_folder = folders[0]
        if top_folder in ("Items", "Creatures") and len(folders) < 2:     # needs <subtype|series>/<N Stars>/
            return {"type": CardType.UNKNOWN.value}   # (indexed below; compiled code would read out of bounds)

        # Classification logic
        classifiers = {
            "MDW": lambda: {"type": CardType.MDW.value, "ability": None},
            "Field": lambda: {
                "type": CardType.FIELD.value,
                "stars": CardClassifier._stars(folders[len(folders) - 1]),
            },
            "Items": lambda: {
                "type": CardType.ITEM.value,
                "subtype": folders[len(folders) - 2].lower(),
                "stars": CardClassifier._stars(folders[len(folders) - 1]),
            },
            "Creatures": lambda: CardClassifier._classify_creatures(folders),
            "N.M.E": lambda: {"type": CardType.NME.value},
        }

        if top_folder in classifiers:
            return classifiers[top_folder]()

        if top_folder == "Auxiliary":
            return CardClassifier._classify_auxiliary(folders)

        return {"type": CardType.UNKNOWN.value}

    @staticmethod
    def _stars(folder: str) -> int:
        """The star count a folder name starts with ("2 Stars" -> 2)."""
        words = folder.split()
        if not words:           # a blank folder: words[0] would read out of bounds when compiled
            raise ValueError(f"no star count in folder {folder!r}")
        return int(words[0])

    @staticmethod
    def _classify_creatures(folders: list[str]) -> dict:
        """Classify creature cards, preserving existing series if available."""
        result = {
            "type": CardType.CREATURE.value,
            "stars": CardClassifier._stars(folders[len(folders) - 1]),
            "series": folders[len(folders) - 2].lower(),
            "hp": -1, "def": -1, "atk": -1, "spd": -1,
        }
        return result

    @staticmethod
    def _classify_auxiliary(folders: list[str]) -> dict:
        """Classify auxiliary cards."""
        if len(folders) < 2:
            last_folder = folders[len(folders) - 1] if folders else CardType.UNKNOWN.value
            return {"type": last_folder.lower() if isinstance(last_folder, str) else last_folder}

        aux_type = folders[1]

        if aux_type == "Minions":
            return {
                "type": CardType.MINION.value,
                "hp": -1, "def": -1, "atk": -1, "spd": -1,
            }
        elif aux_type == "Items":
            return {"type": CardType.AUX_ITEM.value}
        elif len(folders) > 2:
            if folders[2] == "Debuffs":
                return {
                    "type": CardType.DEBUFF.value,
                    "stars": CardClassifier._stars(folders[len(folders) - 1]),
                }
            elif folders[2] == "Buffs":
                return {"type": CardType.BUFF.value}

        return {"type": folders[len(folders) - 1].lower()}


class PSDParser:
    """Parses PSD files to extract card information."""
    def __init__(self, all_types: list[str], db: StatsDatabase, icons: Optional[IconVocabulary] = None):
        self.all_types = all_types
        self.db = db
        self.icons = icons if icons is not None else IconVocabulary.build(all_types, [], None)
        self._gap_pattern = re_compile(r'\s{3,}')

    def parse(self, file_path: Union[str, PSDImage], relative_path: str = "", existing: Optional[CardInfo] = None,
              variant: Optional[int] = None) -> CardInfo:
        """
        Parse a PSD file to extract card information.

        Args:
            file_path: Full path to PSD file, or the PSD already open (one open for the parse and psd_details)
            relative_path: Relative path in repository
            existing: The card whose author, series and subseries it keeps (default: its name's in the database; a
                renamed PSD's old card)
            variant: The picture to read (variant_view: tagged layers by their tags); None: the eyes as saved

        Returns:
            CardInfo object with extracted data
        """
        name = card_name(relative_path)

        # Get existing card if it exists
        existing_card = existing if existing is not None else self.db.stats.get(name)

        # Initialize card with classification (passing existing card)
        card_dict = CardClassifier.classify(relative_path)

        card = CardInfo(
            name=name,
            card_type=card_dict.get("type", CardType.UNKNOWN.value),
            path=relative_path,
            stars=card_dict.get("stars"),
            subtype=card_dict.get("subtype"),
            series=card_dict.get("series"),
            parser_version=PARSER_VERSION,
        )

        # Preserve author and series (and subseries) from existing card, as they were before website edits
        if existing_card:
            parsed = getattr(existing_card, "parsed_values", None) or {}
            for attr in ("author", "series", "subseries"):      # getattr: older pickles lack subseries
                value = parsed[attr] if attr in parsed else getattr(existing_card, attr, None)
                if value:
                    setattr(card, attr, value)

        # Set initial stats if present
        if "hp" in card_dict:
            card.stats = CardStats(
                hp=card_dict.get("hp", -1),
                defense=card_dict.get("def", -1),
                attack=card_dict.get("atk", -1),
                speed=card_dict.get("spd", -1),
            )

        # Parse PSD layers
        psd = PSDImage.open(file_path) if isinstance(file_path, (str, bytes, PathLike)) else file_path
        with variant_view(psd, variant):
            self._extract_from_layers(psd, card, relative_path)

        return card

    def _extract_from_layers(self, psd: PSDImage, card: CardInfo, relative_path: str) -> None:
        """Extract information from PSD layers."""
        num_stars = MutableValue(0)
        stat_trackers = StatTrackers()
        type_bboxes: list[tuple[str, BoundingBox]] = []
        abilities: list[tuple[str, BoundingBox]] = []

        card_mid_y = int(psd.height * TYPE_REGION_RATIO)

        is_rulepage = RULEBOOK_DIR in relative_path
        get_stars_from_psd = stars_from_psd(relative_path)

        effect_layers = []
        text_layers = []
        hidden_abilities = []
        titles = self.title_layers(psd, [layer for layer in psd.descendants() if layer.kind == "type"
                                         and EFFECT_LAYER_TAG not in layer.name], card.name, not is_rulepage)
        for layer in psd.descendants():
            if EFFECT_LAYER_TAG in layer.name and layer.kind != "group":
                effect_layers.append(layer)     # a pixel copy of words from a text layer; nothing to parse
                continue
            if is_rulepage and layer.kind == "type" and not layer.is_visible():
                continue                        # hidden rule text isn't printed
            if layer.kind == "type":
                text_layers.append(layer)
                if not is_rulepage and base_name(layer).lower() == "ability" and not layer.is_visible():
                    hidden_abilities.append(layer)      # read once every text layer is known (_relaid)
                    continue
            self._process_layer(
                layer, card, num_stars,
                stat_trackers, type_bboxes, abilities,
                is_rulepage, get_stars_from_psd, card_mid_y, titles)
        for layer in hidden_abilities:
            self._process_text_layer(layer, card, num_stars, abilities, is_rulepage, get_stars_from_psd, card_mid_y,
                                     self._relaid(layer, text_layers))
        for layer in effect_layers:
            self._check_effect_layer(layer, text_layers, card)
        card.display_name = self._printed_title(psd, text_layers, card.name, own_text=not is_rulepage)

        if abilities:
            card.ability = self._process_abilities(abilities, type_bboxes, card_mid_y)
        elif hidden_abilities:
            card.ability = ""                   # its Ability layer is hidden and its words printed nowhere
        elif is_rulepage:
            card.problems.append("NO ABILITY LAYER")        # cards need no ability (owner 10-25); a page needs text
        ids = [layer.layer_id for layer in psd.descendants() if layer.layer_id >= 0]     # -1: none stored
        if len(set(ids)) < len(ids):
            card.problems.append("DUPLICATE LAYER IDS")
        self._process_stats(card, stat_trackers)

        if get_stars_from_psd:
            card.stars = num_stars.value + self._count_stars(psd)    # icon-font stars in the header + drawn stars
            if not card.stars:
                card.problems.append('NO STARS FOUND (put the star layers in a group named "Stars")')

    def _printed_title(self, psd: PSDImage, text_layers: list, name: str, own_text: bool = True) -> Optional[str]:
        """
        The title as the card prints it, when it differs from the file name.

        The title is the visible Text/Name layer (several in the title region: joined in reading order), else the
        visible Name text layers in the title region, else a visible layer under Text named after the card, else one
        named after the card elsewhere in the title region (a status effect's top-level title), else one in the title
        region named after its own text, the only one there (Photoshop's name for a new layer: Papyrus's 'The Great
        Papyrus'). A Name layer holding part of the name beside layers holding the rest (Royal Eradicator Main Cannon:
        "Main Cannon", "Royal", "Eradicator"; WX-78: "wx-" and an image layer named "78") prints the name. Hidden or
        rasterized titles read as nothing.

        Args:
            psd: The open card
            text_layers: Its text layers
            name: The card's name (file stem)
            own_text: Whether the last rule applies (not on a rule page: its headings are named after their text)

        Returns:
            The printed title, or None when it equals the name or can't be read
        """
        found = self.title_layers(psd, text_layers, name, own_text)
        texts = []
        for text, _ in self._sort_by_position([(title_text(layer.text), BoundingBox.from_tuple(layer.bbox[:2])) for layer in found]):
            if text and text not in texts:              # a layer copied for an effect prints its words once
                texts.append(text)
        title = " ".join(texts)
        if not title or title == name:
            return None
        if len(texts) > 1 and sorted(title.lower().split()) == sorted(name.lower().split()):
            return None                                 # the pieces' order is a guess
        if self.split_title(psd, found, title, name):
            return None                                 # the other layers print the rest of the name
        longer = self._longer_piece(psd, found, "".join(title.lower().split()), "".join(name.lower().split()))
        if longer is not None:                          # a piece edited longer than the name: the whole title
            return " ".join(title_text(layer.text) for layer in self.reading(found + [longer]))
        return title

    @staticmethod
    def title_layers(psd: PSDImage, text_layers: list, name: str, own_text: bool = True) -> list:
        """
        The visible text layers _printed_title reads: Text/Name, else the Name layers in the title region, else the
        layers under Text named after the card, else those elsewhere in the title region (a status effect's top-level
        title), else the one layer in the title region named after its own text (named_after_text: Sans's 'snas.';
        several are stacked words, Knight Lost A Lot's, and read as none); of several, those in the title region.
        Named after the card: letters and digits alike ('Mr. Jawsum').

        Args:
            psd: The open card
            text_layers: Its text layers
            name: The card's name (file stem)
            own_text: Whether the last rule applies (False on a rule page)

        Returns:
            The layers, in layer order
        """
        top = psd.height * TITLE_REGION_RATIO
        text_group = PSDParser.text_group(psd)
        under_text = [layer for layer in text_group if layer.is_visible() and layer.kind == "type"] if text_group is not None else []
        key = name_key(name)
        found = [layer for layer in under_text if base_name(layer).strip() == "Name"]
        if not found:
            found = [layer for layer in text_layers if base_name(layer).strip() == "Name" and layer.is_visible() and layer.bbox[1] < top]
        if not found and key:
            found = [layer for layer in under_text if name_key(base_name(layer)) == key]
        if not found and key:
            found = [layer for layer in text_layers if layer.kind == "type" and layer.is_visible() and layer.bbox[1] < top
                     and name_key(base_name(layer)) == key]
        if not found and own_text:
            own = [layer for layer in text_layers if layer.kind == "type" and layer.is_visible() and layer.bbox[1] < top
                   and named_after_text(layer)]
            found = own if len(own) == 1 else []
        if len(found) > 1:
            found = [layer for layer in found if layer.bbox[1] < top]
        return found

    @staticmethod
    def split_title(psd: PSDImage, found: list, title: str, name: str) -> bool:
        """
        Whether the title layers print part of the name and the visible layers beside them under Text, in the title
        region, the rest (Royal Eradicator Main Cannon: "Main Cannon" beside "Royal" and "Eradicator").

        Args:
            psd: The open card
            found: title_layers()
            title: Their text, joined
            name: The card's name (file stem)
        """
        joined, whole = "".join(title.lower().split()), "".join(name.lower().split())
        if joined == whole or joined not in whole:
            return False
        return PSDParser._split_pieces(psd, found, joined, whole) is not None

    @staticmethod
    def _split_pieces(psd: PSDImage, found: list, joined: str, whole: str) -> Optional[list]:
        """split_title's layers printing the rest of the name (longest first), or None when they don't cover it."""
        top = psd.height * TITLE_REGION_RATIO
        text_group = PSDParser.text_group(psd)
        shown = [layer for layer in text_group if layer.is_visible()] if text_group is not None else []
        found_ids = {id(layer) for layer in found}
        pieces = [("".join(title_text(layer.text if layer.kind == "type" else base_name(layer)).lower().split()), layer)
                  for layer in shown if id(layer) not in found_ids and layer.bbox[1] < top]
        rest = whole.replace(joined, "", 1)
        used = []
        for piece, layer in sorted(pieces, key=lambda p: len(p[0]), reverse=True):
            if piece and piece in rest:
                rest = rest.replace(piece, "", 1)
                used.append(layer)
        return used if not rest else None

    @staticmethod
    def _longer_piece(psd: PSDImage, found: list, joined: str, whole: str):
        """A split title's last piece edited longer than the name (Warcraftian Druid's 'Druid ' now 'Druid of the
        Claw'): the one visible text layer under Text in the title region, beside the title layers, starting with the
        rest of the name; None else."""
        if not joined or joined == whole or not whole.startswith(joined):
            return None
        rest, top = whole[len(joined):], psd.height * TITLE_REGION_RATIO
        text_group = PSDParser.text_group(psd)
        found_ids = {id(layer) for layer in found}
        hits = [layer for layer in (text_group if text_group is not None else []) if layer.kind == "type"
                and layer.is_visible() and id(layer) not in found_ids and layer.bbox[1] < top
                and "".join(title_text(layer.text).lower().split()).startswith(rest)]
        return hits[0] if len(hits) == 1 else None

    @staticmethod
    def text_group(psd: PSDImage):
        """The top-level Text group, or None."""
        return next((layer for layer in psd if layer.is_group() and base_name(layer).strip() == "Text"), None)

    def _check_effect_layer(self, layer, text_layers: list, card: CardInfo) -> None:
        """
        Flag an effect layer that no longer matches its text.

        "Ability FX: Blazing" holds the word Blazing cut pixel-for-pixel from the "Ability" text layer, with a layer
        style on top (MDW/Scripts/Effect Layers.jsx in TTSCardMaker). Editing the text without running the script's
        Update all leaves the glow where the word used to be; then its pixels stop landing on the text's pixels.
        A layer whose words left the text (orphaned, or "Frozen#2" with one Frozen left) is WORDS NOT IN TEXT.
        """
        if not layer.is_visible():
            return
        owner, _, words = layer.name.partition(EFFECT_LAYER_TAG)
        bases = [text_layer for text_layer in text_layers if text_layer.name == owner or base_name(text_layer) == owner]
        if not bases:
            self._add_problem(card, f"EFFECT LAYER WITHOUT TEXT LAYER: {layer.name}")
            return
        x0, y0, x1, y1 = layer.bbox
        base = next((b for b in bases if b.bbox[0] < x1 and x0 < b.bbox[2] and b.bbox[1] < y1 and y0 < b.bbox[3]), bases[0])

        text = sub(r"[\s\x03]+", " ", base.engine_dict["Editor"]["Text"].value).lower()
        for word in words.split("|"):
            occurrence = fullmatch(r"(.*\S)#(\d+)", word.strip())     # "Frozen#2": the 2nd Frozen
            phrase = sub(r"\s+", " ", occurrence.group(1) if occurrence else word.strip()).lower()
            nth = max(int(occurrence.group(2)), 1) if occurrence else 1
            if phrase and self._occurrences(text, phrase) < nth:
                self._add_problem(card, f"EFFECT LAYER WORDS NOT IN TEXT: {layer.name}")
                return
        coverage = self._pixel_coverage(layer, base)
        if coverage is not None and coverage < EFFECT_COVERAGE_MIN:
            self._add_problem(card, f"EFFECT LAYER OUT OF DATE: {layer.name} (run Effect Layers > Update all)")

    @staticmethod
    def _occurrences(text: str, phrase: str) -> int:
        """Matches of `phrase` in `text` as Effect Layers.jsx counts them: whole words when there are any, else all."""
        found = whole = 0
        start = text.find(phrase) if phrase else -1
        while start >= 0:
            end = start + len(phrase)
            found += 1
            before = text[start - 1] if start else " "
            after = text[end] if end < len(text) else " "
            if not (before.isascii() and before.isalnum()) and not (after.isascii() and after.isalnum()):
                whole += 1
            start = text.find(phrase, end)
        return whole or found

    @staticmethod
    def _pixel_coverage(layer, base) -> Optional[float]:
        """Share of `layer`'s opaque pixels that sit on `base`'s pixels; None when either has no pixel data."""
        try:
            shape = layer.numpy("shape")
            base_shape = base.numpy("shape")
        except (ValueError, KeyError, IndexError, TypeError, AttributeError):
            return None
        if shape is None or base_shape is None:
            return None
        ys, xs = (shape[:, :, 0] > 0.5).nonzero()
        if not len(xs):
            return None
        xs = xs + (layer.bbox[0] - base.bbox[0])
        ys = ys + (layer.bbox[1] - base.bbox[1])
        inside = (xs >= 0) & (ys >= 0) & (xs < base_shape.shape[1]) & (ys < base_shape.shape[0])
        hits = int((base_shape[ys[inside], xs[inside], 0] > 0.2).sum())
        return hits / len(xs)

    def _process_layer(self, layer, card: CardInfo,
                       num_stars: MutableValue, stat_trackers: StatTrackers,
                       type_bboxes: list, abilities: list, is_rulepage: bool,
                       get_stars_from_psd: bool, card_mid_y: int, titles=()) -> None:
        """Process a single PSD layer (titles: title_layers(), whose icons are no header's)."""
        # Text layers
        if layer.kind == "type":
            self._process_text_layer(layer, card, num_stars, abilities, is_rulepage, get_stars_from_psd, card_mid_y,
                                     title=any(layer is title for title in titles))

        # Stat layers
        elif self._is_stat_layer(layer):
            self._process_stat_layer(layer, stat_trackers)

        # Type layers (a rule page's are pictures: its icons are typed as tokens)
        elif (not is_rulepage and base_name(layer).lower() in self.all_types and layer.is_visible() and layer.has_pixels()):
            bbox = BoundingBox.from_tuple(layer.bbox[:2])
            if bbox.y < card_mid_y:
                card.types.append(base_name(layer).lower())
            else:
                type_bboxes.append((base_name(layer).lower(), bbox))

        # Star layers: counted per star by _count_stars once the walk is done
        elif get_stars_from_psd and self._is_star_layer(layer):
            pass

        # Problematic type layer
        elif (base_name(layer).lower() in MISSPELT_CARD_TYPES and layer.is_visible() and layer.has_pixels()):
            card.problems.append(f"MISSPELT TYPE: {base_name(layer).lower()}")

    def _process_text_layer(self, layer, card: CardInfo, num_stars: MutableValue, abilities: list,
                            is_rulepage: bool, get_stars_from_psd: bool, card_mid_y: int, relaid: bool = False,
                            title: bool = False) -> None:
        """
        Process text layer. A hidden Ability layer is the ability only when relaid (_relaid): only printed text counts.

        Text set in the stoogedings font is read as icons: [fire]-style tokens (any case, spacing or alias) become
        canonical [fire] tokens, and icon characters typed from the Glyphs panel become tokens too. Header text in
        the font also adds to card.types, and to the star count where stars come from the PSD; a title's icons
        (title: one of title_layers()) are part of its name (owner 2026-10-24: a token typed in it is set in the font).
        Layers without the font take the original path unchanged.
        """
        engine_text = layer.engine_dict["Editor"]["Text"]
        is_ability = is_rulepage or (base_name(layer).lower() == "ability" and (relaid or layer.is_visible()))
        icon_spans = self._icon_font_spans(layer, engine_text.value)
        self._flag_misplaced_tokens(engine_text.value, icon_spans, card)
        if icon_spans:
            text, icons = self._read_icon_tokens(engine_text.value, icon_spans, card)
            layer_text = repr(text)  # str() of the engine string is its repr, so the cleanup below is unchanged
            if not is_ability and not title and layer.is_visible() and layer.bbox[1] < card_mid_y:
                for name, kind in icons:
                    if kind == "type":
                        card.types.append(name)
                    elif kind == "star" and get_stars_from_psd:
                        num_stars.value += 1
        else:
            layer_text = str(engine_text)
        layer_text = (layer_text
            .replace('\\r', '\n')
            .replace('\\n', '\n')
            .replace('\\t', ' ')
            .replace('\\x03', '\n')
            .replace('\\ufeff', '')
            .rstrip())

        if is_ability:
            bbox = BoundingBox.from_tuple(layer.bbox[:2])
            abilities.append((layer_text, bbox))

    @staticmethod
    def _relaid(hidden, text_layers: list) -> bool:
        """
        Whether a hidden Ability layer's words are printed: the visible text layers over it, in reading order, say
        exactly them (Butterfly, Bulborb Larva, Empress Bulblax, Apex Predator lay their ability out again in pieces).
        Shadow Duelist prints none of its hidden text: not its ability.
        """
        pieces = PSDParser._pieces(hidden, text_layers)
        words = " ".join((layer.text or "") for layer in pieces).replace("\x03", " ").replace("\ufeff", "").split()
        return bool(words) and words == (hidden.text or "").replace("\x03", " ").replace("\ufeff", "").split()

    @staticmethod
    def _pieces(hidden, text_layers: list) -> list:
        """The visible text layers over a hidden one, in reading order (top, then left): _relaid's."""
        x0, y0, x1, y1 = hidden.bbox
        return sorted((layer for layer in text_layers if layer is not hidden and layer.is_visible()
                       and layer.bbox[0] < x1 and x0 < layer.bbox[2] and layer.bbox[1] < y1 and y0 < layer.bbox[3]),
                      key=lambda layer: (layer.bbox[1], layer.bbox[0]))

    # ------------------------
    # The parts: what the website edits, one box per layer (docs/Future/PARTS_EDITOR.md §2.2)
    # ------------------------
    def layer_text(self, layer, card: Optional[CardInfo] = None) -> str:
        """
        One text layer's reading: icon-font runs as canonical tokens, breaks as \\n, tabs as spaces, no byte-order mark,
        blank ends and line-leading padding (3+ blanks) gone; image icons' tokens are not in it.
        """
        text = layer.engine_dict["Editor"]["Text"].value
        spans = self._icon_font_spans(layer, text)
        if spans:
            text = self._read_icon_tokens(text, spans, card if card is not None else CardInfo(name="", card_type="unknown"))[0]
        text = text.replace("\r", "\n").replace("\x03", "\n").replace("\t", " ").replace("\ufeff", "").strip(" \n")
        return LINE_PAD.sub("", text)

    @staticmethod
    def reading(layers: list) -> list:
        """Layers in reading order (_sort_by_position: top, 40 px rows, then left)."""
        return [layer for layer, _ in PSDParser._sort_by_position([(layer, BoundingBox.from_tuple(layer.bbox[:2])) for layer in layers])]

    @staticmethod
    def words(text: Optional[str]) -> str:
        """Words alike: whitespace runs as one space, case folded (a copy prints its layer's words)."""
        return " ".join((text or "").replace("\x03", " ").replace("\ufeff", "").split()).casefold()

    @staticmethod
    def title_parts(psd: PSDImage, text_layers: list, name: str, own_text: bool = True) -> tuple:
        """
        (parts, copies): title_layers() plus the text layers split_title finds printing the rest of the name (Warcraftian
        Druid's 'Druid '), in reading order; a layer printing an earlier part's words is a copy [(layer, part)].
        """
        found = PSDParser.title_layers(psd, text_layers, name, own_text)
        title = " ".join(title_text(layer.text) for layer in PSDParser.reading(found))
        joined, whole = "".join(title.lower().split()), "".join(name.lower().split())
        pieces = []
        if joined and joined != whole and joined in whole:
            pieces = PSDParser._split_pieces(psd, found, joined, whole)
            longer = PSDParser._longer_piece(psd, found, joined, whole) if pieces is None else None
            pieces = [layer for layer in (pieces or []) + ([longer] if longer is not None else []) if layer.kind == "type"]
        parts, copies = [], []
        for layer in PSDParser.reading(found + pieces):
            twin = next((kept for kept in parts if PSDParser.words(kept.text) == PSDParser.words(layer.text)), None)
            if twin is not None:
                copies.append((layer, twin))
            else:
                parts.append(layer)
        return (parts, copies)

    @staticmethod
    def ability_parts(text_layers: list) -> tuple:
        """
        (parts, hidden): the visible Ability text layers in reading order; with none, the visible layers laying a hidden
        Ability out again (_pieces of the first _relaid one: Butterfly) and that hidden layer, else ([], None).
        """
        texts = [layer for layer in text_layers if layer.kind == "type" and EFFECT_LAYER_TAG not in layer.name]
        shown = [layer for layer in texts if layer.is_visible() and base_name(layer).lower() == "ability"]
        if shown:
            return (PSDParser.reading(shown), None)
        for layer in texts:
            if not layer.is_visible() and base_name(layer).lower() == "ability" and PSDParser._relaid(layer, texts):
                return (PSDParser._pieces(layer, texts), layer)
        return ([], None)

    def join_ability(self, psd: PSDImage, layers: list, hidden=None, rulepage: bool = False) -> str:
        """The parser's ability from ability_parts' layers (or blocks'): their readings (the hidden relaid layer's when
        given) joined by _process_abilities, image icons' tokens in. Equals the parse's ability (the census)."""
        mid = int(psd.height * TYPE_REGION_RATIO)
        card, read = CardInfo(name="", card_type=""), []
        for layer in ([hidden] if hidden is not None else layers):
            self._process_text_layer(layer, card, MutableValue(0), read, rulepage, False, mid, relaid=True)
        icons = [] if rulepage else [
            (base_name(layer).lower(), BoundingBox.from_tuple(layer.bbox[:2])) for layer in psd.descendants()
            if layer.kind != "type" and not self._is_stat_layer(layer) and base_name(layer).lower() in self.all_types
            and layer.is_visible() and layer.has_pixels() and layer.bbox[1] >= mid]
        return self._process_abilities(read, icons, mid) if read else ""

    def blocks(self, text_layers: list) -> list:
        """A rule page's text blocks: [(layer, role)], every visible text layer in reading order; role "header" (named
        Header), "page" (Page Number, else the lowest one holding only digits) or "block"."""
        shown = self.reading([layer for layer in text_layers if layer.kind == "type" and EFFECT_LAYER_TAG not in layer.name
                              and layer.is_visible()])
        page = None
        if not any(base_name(layer).strip() == "Page Number" for layer in shown):
            digits = [layer for layer in shown if self.layer_text(layer).isdigit()]
            page = max(digits, key=lambda layer: layer.bbox[1]) if digits else None
        out = []
        for layer in shown:
            role = base_name(layer).strip()
            out.append((layer, "header" if role == "Header" else "page" if role == "Page Number" or layer is page else "block"))
        return out

    @staticmethod
    def _icon_font_spans(layer, text: str) -> list[tuple[int, int]]:
        """(start, end) index ranges of `text` set in the stoogedings font; empty when the layer doesn't use it."""
        try:
            font_set = layer.resource_dict["FontSet"]
            runs = layer.engine_dict["StyleRun"]["RunArray"]
            run_lengths = layer.engine_dict["StyleRun"]["RunLengthArray"]
            fonts = [font_set[int(run["StyleSheet"]["StyleSheetData"].get("Font", 0))]["Name"].value for run in runs]
        except (KeyError, IndexError, TypeError, AttributeError):
            return []
        if not any(font.lower().startswith(ICON_FONT_PREFIX) for font in fonts):
            return []

        # Run lengths count UTF-16 units; characters outside the BMP (some emoji) take two
        unit_starts = []
        units = 0
        for char in text:
            unit_starts.append(units)
            units += 2 if ord(char) > 0xFFFF else 1

        spans = []
        position = 0
        for font, length in zip(fonts, run_lengths):
            end = position + int(length)
            if font.lower().startswith(ICON_FONT_PREFIX):
                spans.append((bisect_left(unit_starts, position), bisect_left(unit_starts, end)))
            position = end
        return spans

    @staticmethod
    def _covered(start: int, end: int, spans: list[tuple[int, int]]) -> int:
        """How many characters of text[start:end] fall inside the spans."""
        return sum(max(0, min(end, b) - max(start, a)) for a, b in spans)

    @staticmethod
    def _add_problem(card: CardInfo, problem: str) -> None:
        """Record a parse-time problem once."""
        if problem not in card.problems:
            card.problems.append(problem)

    def _read_icon_tokens(self, text: str, spans: list[tuple[int, int]],
                          card: CardInfo) -> tuple[str, list[tuple[str, str]]]:
        """
        Rewrite the icon-font tokens in a layer's text as canonical tokens.

        "[Fire:Sticker]" and "[fire:glow]" -> "[fire]", "[Defensive Minion]" -> "[defensive_minion]", "[fire:no]" stays
        "[fire:no]", and an icon character from the Glyphs panel -> "[fire]". Unknown tokens are left as typed and
        reported, because the font shows them as plain letters on the card.

        Args:
            text: Raw layer text
            spans: Ranges set in the icon font
            card: Card being parsed (for problems)

        Returns:
            (rewritten text, [(icon name, "type" | "star"), ...] in text order)
        """
        out, icons, last = [], [], 0
        for match in ICON_SCAN_PATTERN.finditer(text):
            start, end = match.span()
            if self._covered(start, end, spans) < end - start:
                continue
            piece = match.group(0)
            if piece.startswith("["):
                variant = token_variant(piece)      # the font accepts no spaces around the variant either
                hit = self.icons.lookup(piece[1:len(piece) - 1].partition(":")[0])
                if hit is None or variant is None:
                    self._add_problem(card, f"UNKNOWN ICON TOKEN: {piece}")
                    continue
                name, kind = hit
                token = f"[{name}:no]" if variant == "no" else f"[{name}]"
            else:
                name = self.icons.codepoints.get(piece)
                hit = self.icons.lookup(name) if name else None
                if hit is None:
                    self._add_problem(card, f"UNKNOWN ICON CHARACTER: U+{ord(piece):04X}")
                    continue
                kind = hit[1]
                token = f"[{name}]"
            out.append(text[last:start])
            out.append(token)
            last = end
            icons.append((name, kind))
        out.append(text[last:])
        return "".join(out), icons

    def _flag_misplaced_tokens(self, text: str, spans: list[tuple[int, int]], card: CardInfo) -> None:
        """Report icon tokens that aren't fully in the icon font: on the card they show as literal "[fire]" text."""
        for match in ICON_SCAN_PATTERN.finditer(text):
            piece = match.group(0)
            if not piece.startswith("["):
                continue
            if self.icons.lookup(piece[1:len(piece) - 1].partition(":")[0]) is None or token_variant(piece) is None:
                continue
            covered = self._covered(match.start(), match.end(), spans)
            if covered == 0:
                self._add_problem(card, f"ICON TOKEN NOT IN STOOGEDINGS FONT: {piece}")
            elif covered < len(piece):
                self._add_problem(card, f"ICON TOKEN PARTLY IN STOOGEDINGS FONT: {piece}")

    def _is_stat_layer(self, layer) -> bool:
        """Check if layer is a stat layer."""
        keywords = ["dark", "bars"]
        parent_names = []

        current = layer.parent
        depth = 0
        while current and depth < 3:
            parent_names.append(current.name.lower())
            current = getattr(current, 'parent', None)
            depth += 1

        return any(keyword in name for name in parent_names for keyword in keywords)

    def _process_stat_layer(self, layer, stat_trackers: StatTrackers) -> None:
        """Process stat layer."""
        digits = base_name(layer)
        if not digits.isdigit():
            return
        # A digit-named group of digit layers is a bar row (multi-row bars name rows "1", "2", ...), not a value.
        # Sonic's two speed rows "2" and "1" were added as 3 more speed. An empty one draws nothing (Twin Emperors'
        # "shared health" has an empty "1" group, which made its 7 health 8).
        if layer.is_group() and (len(layer) == 0 or any(base_name(child).isdigit() for child in layer)):
            return

        parent_names = []
        current = layer.parent
        depth = 0
        while current and depth < 2:
            parent_names.append(current.name.lower())
            current = getattr(current, 'parent', None)
            depth += 1

        # Map short names to StatTracker attributes ("health": e.g. Twin Emperors' "shared health" row)
        stat_map = {
            "hp": stat_trackers.hp,
            "def": stat_trackers.defense,
            "atk": stat_trackers.attack,
            "spd": stat_trackers.speed,
            "health": stat_trackers.hp,
        }

        for stat_key, tracker in stat_map.items():
            if any(stat_key in name for name in parent_names):
                tracker.found = True
                parent = layer.parent
                block = getattr(parent, "parent", None)
                row = tracker.rows.setdefault(id(parent), BarRow(id(block), base_name(parent), getattr(block, "name", "") or "",
                                                                 id(getattr(block, "parent", None))))
                if layer.is_visible():
                    row.items.append((int(digits), layer.kind))
                break

    def _count_stars(self, psd: PSDImage) -> int:
        """
        Stars drawn on a card whose star count comes from the PSD.

        Each visible group with "stars" in its name is searched: child groups count recursively, and visible layers
        that overlap (one's core center inside the other's core: a glow, a frame, or the pieces of one star) count
        as one star. A star is often several layers, e.g. Indoctrination's pentagram and four pieces, which counting
        layers made 20 stars instead of 4. With no such group, visible layers named "... Star ..." count the same way.
        """
        groups = self._star_groups(psd)
        if groups:
            return sum(self._stars_in(group) for group in groups)
        named = [layer for layer in psd.descendants()
                 if not layer.is_group() and layer.is_visible() and layer.has_pixels()
                 and "star" in sub(r"[^a-z]+", " ", layer.name.lower()).split()]
        return self._count_clusters(named)

    @staticmethod
    def _star_groups(group) -> list:
        """The outermost visible groups with "stars" in their name."""
        found = []
        for layer in group:
            if not layer.is_group() or not layer.is_visible():
                continue
            if "stars" in layer.name.lower():
                found.append(layer)
            else:
                found.extend(PSDParser._star_groups(layer))
        return found

    def _stars_in(self, group) -> int:
        count, loose = 0, []
        for layer in group:
            if not layer.is_visible():
                continue
            if layer.is_group():
                count += self._stars_in(layer)
            elif layer.has_pixels():
                loose.append(layer)
        return count + self._count_clusters(loose)

    @staticmethod
    def _core_box(layer) -> tuple:
        """Bounding box of the layer's mostly opaque pixels (its whole box if they can't be read)."""
        x0, y0, x1, y1 = layer.bbox
        try:
            shape = layer.numpy("shape")
        except (ValueError, KeyError, IndexError, TypeError, AttributeError):
            shape = None
        if shape is not None:
            ys, xs = (shape[:, :, 0] >= STAR_CORE_ALPHA).nonzero()
            if len(xs):
                return x0 + int(xs.min()), y0 + int(ys.min()), x0 + int(xs.max()) + 1, y0 + int(ys.max()) + 1
        return x0, y0, x1, y1

    @staticmethod
    def _center_inside(a: tuple, b: tuple) -> bool:
        cx, cy = (a[0] + a[2]) / 2, (a[1] + a[3]) / 2
        return b[0] <= cx <= b[2] and b[1] <= cy <= b[3]

    @classmethod
    def _count_clusters(cls, layers: list) -> int:
        """How many separate stars the layers draw: overlapping layers (see _count_stars) are one."""
        boxes = [cls._core_box(layer) for layer in layers]
        parent = list(range(len(boxes)))

        def root(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                if cls._center_inside(boxes[i], boxes[j]) or cls._center_inside(boxes[j], boxes[i]):
                    parent[root(i)] = root(j)
        return len({root(i) for i in range(len(boxes))})

    def _is_star_layer(self, layer) -> bool:
        """Check if layer is a star layer."""
        parent_names = []
        current = layer.parent
        depth = 0
        while current and depth < 3:
            parent_names.append(current.name.lower())
            current = getattr(current, 'parent', None)
            depth += 1

        return any("stars" in name for name in parent_names)

    def _process_abilities(self, abilities: list[tuple[str, BoundingBox]],
                           type_bboxes: list[tuple[str, BoundingBox]], card_mid_y: int) -> Optional[str]:
        """Process and combine ability texts."""
        abilities = self._sort_by_position(abilities)
        ability_text = '\n'.join(text.strip('\'"\n') for text, _ in abilities)

        if ability_text:
            type_bboxes = self._prune_type_bboxes(self._sort_by_position(type_bboxes), card_mid_y)
            ability_text = self._inject_type_names(ability_text.rstrip(' \'"'), type_bboxes)
            ability_text = sub(r'(?m)^[^\S\n]{3,}(?=\S)', '', ability_text)     # a line's unfilled gap: layout padding
            ability_text = sub(r'\s+([:;,\.\?!])', r'\1', ability_text)
        return ability_text.strip(' \'"\n')

    def _process_stats(self, card: CardInfo, stat_trackers: StatTrackers) -> None:
        """Process stat values."""
        if not card.stats:
            card.stats = CardStats()

        # Assign each stat value if found
        for stat_attr, tracker in [
            ("hp", stat_trackers.hp),
            ("defense", stat_trackers.defense),
            ("attack", stat_trackers.attack),
            ("speed", stat_trackers.speed)]:
            if tracker.found:
                value = tracker.total()
                setattr(card.stats, stat_attr, value if value is not None else 10)
        creatures = stat_trackers.creatures()
        if creatures:
            card.creatures = creatures
            card.stats = deepcopy(creatures[0])
            card.problems.append("SEVERAL CREATURES: check each one's stats on the website")

    @staticmethod
    def _sort_by_position(items: list[tuple[str, BoundingBox]], row_threshold: int = 40) -> list[tuple[str, BoundingBox]]:
        """Sort items top-to-bottom, then left-to-right, grouping items into visual rows."""
        if not items:
            return items

        # Sort primarily by Y
        items = sorted(items, key=lambda i: i[1].y)

        rows: list[list[tuple[str, BoundingBox]]] = []

        for item in items:
            placed = False
            for row in rows:
                # Compare Y against first item in the row
                if abs(item[1].y - row[0][1].y) <= row_threshold:
                    row.append(item)
                    placed = True
                    break
            if not placed:
                rows.append([item])

        # Sort each row left-to-right
        for row in rows:
            row.sort(key=lambda i: i[1].x)

        # Flatten rows
        return [item for row in rows for item in row]

    @staticmethod
    def _prune_type_bboxes(bboxes: list[tuple[str, BoundingBox]], card_mid_y: int) -> list[tuple[str, BoundingBox]]:
        """Remove type bboxes that are too high up."""
        if not bboxes:
            return bboxes
        max_height = max(bboxes[len(bboxes) - 1][1].y // 3, card_mid_y)
        return [bbox for bbox in bboxes if bbox[1].y >= max_height]

    def _inject_type_names(self, ability: str, type_bboxes: list[tuple[str, BoundingBox]]) -> str:
        if not ability or not type_bboxes:
            return ability

        # Sort icons top-to-bottom, then left-to-right
        type_bboxes = self._sort_by_position(type_bboxes)
        types = [t for t, _ in type_bboxes]

        lines = [line.rstrip(" '\n") + '\n' for line in ability.splitlines(keepends=True)]
        result_lines = []
        type_index = 0

        for line in lines:
            if type_index >= len(types):
                result_lines.append(line)
                continue

            matches = list(self._gap_pattern.finditer(line))
            if not matches:
                result_lines.append(line)
                continue

            new_line = []
            last_end = 0
            for match in matches:
                if type_index >= len(types):
                    break

                new_line.append(line[last_end:match.start()].rstrip())
                new_line.append(f" [{types[type_index]}] ")

                type_index += 1
                last_end = match.end()

            # remainder of line
            new_line.append(line[last_end:].lstrip())
            result_lines.append(''.join(new_line).lstrip())

        # Append remaining types to the last line
        if type_index < len(types) and result_lines:
            remaining = ' '.join(f"[{t}]" for t in types[type_index:])
            last_line_index = len(result_lines) - 1
            result_lines[last_line_index] = result_lines[last_line_index].rstrip('\n') + f" {remaining}"

        return ''.join(result_lines)


class CardValidator:
    """Validates card information for problems."""
    RULEBOOK = "rulebook"           # a rule page's card_type (its folder's name)
    ABILITY_EXCLUSIONS: set[str] = {        # card_standards' R3 waiver (cards need no ability since 10-25)
        "Crystal Bits",
        "Bugzzy",
        "Electro Probe",
        "Galacta Warrior",
        "God Tamer",
        "Mini Bee",
        "Paint Warrior",
        "PoD01Red Bloon",
        "Sabre",
        "Shadow Duelist",         # prints no ability (its Ability layer is hidden)
        "Sword Knight",
        "Tentacle",
        "Whelp",
        "Warrior Dee",
        "The Master",
        "Panda",
        "Chomp",
        "Abyss Watcher",
        "Anubisath Guardian",     # three minions with an empty Ability layer (user, 2026-10-21)
        "Qiraji Soldier",
        "Silithid",
    }

    @staticmethod
    def validate(card: CardInfo) -> list[str]:
        """
        Validate card and return list of problems.

        Args:
            card: CardInfo to validate

        Returns:
            List of problem descriptions
        """
        problems = []
        if getattr(card, "problems_ok", False):           # checked on the website
            return problems

        # Type problems
        if card.card_type == CardType.UNKNOWN.value:
            problems.append("UNKNOWN TYPE")

        # Ability problems: a card needs no ability (owner 10-25); a rule page needs its text
        rulepage = card.card_type == CardValidator.RULEBOOK
        if rulepage and not card.ability:
            problems.append("ABILITY TEXT NOT FOUND")

        # Stat problems
        if card.card_type in [CardType.CREATURE.value, CardType.MINION.value]:
            problems.extend(CardValidator._validate_stats(card))

        # Add any problems found during parsing
        if card.problems:
            if "NO ABILITY LAYER" in card.problems and not rulepage:
                card.problems.remove("NO ABILITY LAYER")     # older data's: no problem now
            problems.extend(card.problems)

        return problems

    @staticmethod
    def _validate_stats(card: CardInfo) -> list[str]:
        """Validate stats for creatures and minions."""
        problems = []
        stats = card.stats

        # Check for missing stats
        if stats.hp == -1:
            problems.append("HP NOT FOUND")
        if stats.defense == -1:
            problems.append("DEF NOT FOUND")

        # Active types don't need ATK/SPD
        is_active = "active" in card.types if card.types else False

        if not is_active:
            if stats.attack == -1:
                problems.append("ATK NOT FOUND")
            if stats.speed == -1:
                problems.append("SPD NOT FOUND")

        # Above 10 (two bar rows: Royal Eradicator Main Cannon's 10 + 2) unless the website allows it for the card
        if stats.has_excessive_stats() and not getattr(card, "allow_high_stats", False):
            problems.append("STATS TOO HIGH")

        return problems


# ========================
# Module Exports
# ========================
__all__ = [
    'PARSER_VERSION',
    'STARS_FROM_PSD',
    'BarRow',
    'BoundingBox',
    'CardClassifier',
    'CardValidator',
    'IconVocabulary',
    'MutableValue',
    'PSDParser',
    'StatTracker',
    'StatTrackers',
    'TAG',
    'base_name',
    'layer_tag',
    'named_after_text',
    'shown_in',
    'stars_from_psd',
    'title_text',
    'token_variant',
    'variant_view',
    'variants_of',
]
