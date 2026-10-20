from bisect import bisect_left
from collections import defaultdict
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from discord import Interaction, File
from enum import Enum
from github import Github, Repository
from json import loads as json_loads
from logging import getLogger, CRITICAL
from os import fdopen, listdir, remove, walk
from tempfile import mkstemp
from os.path import basename, isdir, isfile, splitext, join as path_join
from pickle import load, dump
from psd_tools import PSDImage
from copy import deepcopy
from re import IGNORECASE, compile as re_compile, match as re_match, sub
from requests import get
from requests.exceptions import RequestException
from typing import Optional, Any
from urllib.parse import quote
import pandas as pd

import config
from global_config import STATS_PKL, OLD_STATS_PKL
from commands.query_card import card_repo
from commands.utils import queue_message, queue_file, queue_edit, queue_command, split_long_message

getLogger("psd_tools").setLevel(CRITICAL)

# ========================
# Configuration
# ========================
UPDATE_RATE = 25
DOWNLOAD_WORKERS = 4                                              # PSDs downloaded ahead of parsing
_tree_etags: dict = {}                                            # repository -> ETag of the last update's tree
_file_cache: dict = {}                                            # repo path -> (blob sha, text) of small files
TYPE_REGION_RATIO = 0.5
EXCLUDE_FOLDERS = ["Markers", "MDW"]
EXPORTED_STATS_NAME = "stats"
EXPORTED_RULES_NAME = "rules"
GIT_TOKEN: str = getattr(config, "GIT_TOKEN", "")

COLUMN_ORDER = [
    'aliases', 'type', 'ability', 'hp', 'def', 'atk', 'spd',
    'types', 'path', 'timestamp', 'author', 'stars',
    'problem', 'series', 'subtype'
]

MISSPELT_CARD_TYPES = ['undread', 'tornado', 'error']

# stoogedings is the icon font (TTSCardMaker MDW/Fonts). A token such as [fire] set in it renders as the icon,
# so the text itself says which icon sits where. Cards with image-layer icons keep using gap injection.
ICON_FONT_PREFIX = "stoogedings"                                  # PostScript name: stoogedings-Regular
ICON_MANIFEST_PATH = "MDW/Fonts/stoogedings-src/stoogedings.json" # aliases + codepoints, written by the font build
AUTHORS_PATH = "MDW/Card Authors.txt"                             # who made each card; wins over the database
EDITS_PATH = "MDW/Card Edits.json"                                # website fixes; win over the PSD
TITLE_REGION_RATIO = 0.12                                         # hidden text above this = a duplicate's title
ICON_VARIANTS = ("", "tint", "sticker", "no")
ICON_SCAN_PATTERN = re_compile(r"\[[^\[\]\r\n\x03]{1,48}\]|[-★]")
EFFECT_LAYER_TAG = " FX: "        # "Ability FX: Blazing | Frozen In Time", made by MDW/Scripts/Effect Layers.jsx
EFFECT_COVERAGE_MIN = 0.95        # share of an effect layer's pixels that must sit on its text layer's pixels
STAR_CORE_ALPHA = 0.5             # a star's core: its pixels at least this opaque (glows and flares fade out wide)


class CardType(Enum):
    """Enum for card types."""
    UNKNOWN = "unknown"
    MDW = "MDW"
    FIELD = "field"
    ITEM = "item"
    CREATURE = "creature"
    MINION = "minion"
    AUX_ITEM = "aux item"
    DEBUFF = "debuff"
    BUFF = "buff"
    NME = "nme"


@dataclass
class MutableValue:
    """A simple mutable wrapper for primitives."""
    value: Any


DRAWN_LAYER_KINDS = ("pixel", "smartobject", "group", "type", "shape")   # every other kind is an adjustment layer


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
class CardStats:
    """Represents card statistics."""
    hp: int = -1
    defense: int = -1
    attack: int = -1
    speed: int = -1

    def to_dict(self) -> dict[str, int]:
        """Convert to dictionary format."""
        return {
            "hp": self.hp,
            "def": self.defense,
            "atk": self.attack,
            "spd": self.speed,
        }

    def is_valid_stat(self, value: int) -> bool:
        """Check if a stat value is valid."""
        return value >= 0

    def has_excessive_stats(self) -> bool:
        """Check if any stat exceeds maximum value of 10."""
        return any(
            stat > 10 for stat in [self.hp, self.defense, self.attack, self.speed]
            if self.is_valid_stat(stat)
        )


@dataclass
class CardAuthors:
    """
    TTSCardMaker's MDW/Card Authors.txt: the team's record of who made each card.

    Two sections of "left = right" lines (# starts a comment):
      [people]  git/GitHub name = name to show for that person
      [cards]   Card Name = person   (names match in any case, with _ and spaces alike)
    A listed card gets its listed person; any other card keeps its author. Either way the result goes through
    [people], so a git name becomes the display name. The website applies the same rules.
    """
    people: dict = field(default_factory=dict)      # lowercase git name -> display name
    cards: dict = field(default_factory=dict)       # normalized card name -> person

    @staticmethod
    def key(name: str) -> str:
        return sub(r"\s+", " ", name.replace("_", " ")).strip().lower()

    @classmethod
    def parse(cls, text: str) -> "CardAuthors":
        authors, section = cls(), None
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("[") and line.endswith("]"):
                section = line.strip("[]").strip().lower()
                continue
            left, sep, right = line.partition("=")
            left, right = left.strip(), right.strip()
            if not sep or not left or not right:
                continue
            if section == "people":
                authors.people[left.lower()] = right
            elif section == "cards":
                authors.cards[cls.key(left)] = right
        return authors

    def resolve(self, card_name: str, current: Optional[str]) -> Optional[str]:
        person = self.cards.get(self.key(card_name), current)
        return self.people.get(person.lower(), person) if person else person


@dataclass
class CardEdits:
    """
    TTSCardMaker's MDW/Card Edits.json: card fixes from the website's Edit button.
    {"cards": {"Card Name": {"field": value, "_by": ..., "_at": ...}}}; names match like the authors file.
    Same fields as the website (site/build/build_site.py). The PSD values are kept in card.parsed_values.
    """
    cards: dict = field(default_factory=dict)       # normalized card name -> {field: value}

    TYPES = {"name": str, "type": str, "subtype": str, "series": str, "stars": int, "types": list,
             "hp": int, "def": int, "atk": int, "spd": int, "ability": str, "note": str, "aliases": list,
             "dup_of": str, "hidden": bool, "problems_ok": bool, "allow_high_stats": bool, "creatures": list}
    NULLABLE = {"subtype", "series", "stars", "hp", "def", "atk", "spd"}
    ATTRS = {"name": "display_name", "type": "card_type", "subtype": "subtype", "series": "series", "stars": "stars",
             "types": "types", "ability": "ability", "note": "note", "hidden": "hidden", "problems_ok": "problems_ok",
             "allow_high_stats": "allow_high_stats"}
    STATS = {"hp": "hp", "def": "defense", "atk": "attack", "spd": "speed"}
    FIXED_BY = {"hp": ("HP NOT FOUND",), "def": ("DEF NOT FOUND",), "atk": ("ATK NOT FOUND",), "spd": ("SPD NOT FOUND",),
                "ability": ("ABILITY TEXT NOT FOUND", "NO ABILITY LAYER"), "stars": ("NO STARS FOUND",),
                "type": ("UNKNOWN TYPE",), "types": ("MISSPELT TYPE",), "allow_high_stats": ("STATS TOO HIGH",),
                "creatures": ("SEVERAL CREATURES",)}

    @classmethod
    def valid(cls, name: str, value) -> bool:
        kind = cls.TYPES.get(name)
        if kind is None:
            return False
        if value is None:
            return name in cls.NULLABLE
        if kind is int:
            return isinstance(value, int) and not isinstance(value, bool) and value >= 0
        if name == "creatures":                 # [{"hp": 7, "def": 4, "atk": 5, "spd": 6}, ...]
            return isinstance(value, list) and all(isinstance(c, dict) and all(
                k in cls.STATS and (v is None or (isinstance(v, int) and not isinstance(v, bool) and v >= 0))
                for k, v in c.items()) for c in value)
        if kind is list:
            return isinstance(value, list) and all(isinstance(x, str) for x in value)
        return isinstance(value, kind)

    @classmethod
    def parse(cls, text: str) -> "CardEdits":
        """Raises ValueError for a file that isn't JSON; bad entries are skipped."""
        data = json_loads(text)
        cards = data.get("cards") if isinstance(data, dict) else None
        edits = cls()
        for name, fields in (cards if isinstance(cards, dict) else {}).items():
            if isinstance(fields, dict):
                edits.cards[CardAuthors.key(name)] = {k: v for k, v in fields.items() if not k.startswith("_") and cls.valid(k, v)}
        return edits

    def get(self, card: CardInfo, name: str):
        if name == "creatures":
            return [{k: (lambda v: v if v >= 0 else None)(getattr(c, a)) for k, a in self.STATS.items()} for c in card.creatures] if getattr(card, "creatures", None) else None
        if name in self.STATS:
            value = getattr(card.stats, self.STATS[name])
            return value if value >= 0 else None
        if name == "problems":
            return list(card.problems)
        value = getattr(card, self.ATTRS[name], None)
        return list(value) if isinstance(value, list) else value

    def set(self, card: CardInfo, name: str, value) -> None:
        if name == "creatures":
            card.creatures = [CardStats(**{a: -1 if c.get(k) is None else c[k] for k, a in self.STATS.items()}) for c in value] if value else None
            if card.creatures:
                card.stats = deepcopy(card.creatures[0])
        elif name in self.STATS:
            setattr(card.stats, self.STATS[name], -1 if value is None else value)
        elif name == "problems":
            card.problems = list(value or [])
        else:
            setattr(card, self.ATTRS[name], list(value) if isinstance(value, list) else value)

    def apply(self, name: str, card: CardInfo) -> bool:
        """Put back the PSD values, then apply this card's edits. Returns whether the card changed."""
        before = repr(card)
        for key, value in (getattr(card, "parsed_values", None) or {}).items():
            self.set(card, key, value)
        fields = {k: v for k, v in self.cards.get(CardAuthors.key(name), {}).items() if k in self.ATTRS or k in self.STATS or k == "creatures"}
        if "creatures" in fields:                   # set first, so a single-stat edit can still change creature 1
            fields = {"creatures": fields.pop("creatures"), **fields}
        saved = {key: self.get(card, key) for key in fields}
        if fields:
            saved["problems"] = list(card.problems)
            answered = tuple(p for key in fields for p in self.FIXED_BY.get(key, ()))
            card.problems = [] if fields.get("problems_ok") else [p for p in card.problems if not answered or not p.startswith(answered)]
        for key, value in fields.items():
            self.set(card, key, value)
        card.parsed_values = saved or None
        return repr(card) != before

    def names(self) -> dict[str, list[str]]:
        """Card name -> website aliases and display name, for name lookups."""
        out = {}
        for key, fields in self.cards.items():
            extra = list(fields.get("aliases") or []) + ([fields["name"]] if fields.get("name") else [])
            if extra:
                out[key] = extra
        return out


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


@dataclass
class CardInfo:
    """Complete card information."""
    name: str
    card_type: str
    path: str = ""
    timestamp: float = 0.0
    ability: Optional[str] = None
    stars: Optional[int] = None
    subtype: Optional[str] = None
    series: Optional[str] = None
    types: list[str] = field(default_factory=list)
    stats: CardStats = field(default_factory=CardStats)
    problems: list[str] = field(default_factory=list)
    author: Optional[str] = None
    # set from Card Edits.json
    display_name: Optional[str] = None
    note: Optional[str] = None
    hidden: bool = False
    problems_ok: bool = False
    allow_high_stats: bool = False
    parsed_values: Optional[dict] = None      # PSD values of edited fields
    creatures: Optional[list] = None          # CardStats per creature (multi-creature cards)
    duplicate_of: Optional[str] = None        # original card (same PSD, another PNG)
    alt_names: Optional[list] = None          # hidden title layers: names of the duplicates
    blob_sha: Optional[str] = None            # git blob of the PSD it was parsed from

    def to_dict(self) -> dict:
        """Convert card info to dictionary."""
        result = {
            "type": self.card_type,
            "path": self.path,
            "timestamp": self.timestamp,
        }

        if self.ability is not None:
            result["ability"] = self.ability
        if self.stars is not None:
            result["stars"] = self.stars
        if self.subtype:
            result["subtype"] = self.subtype
        if self.series:
            result["series"] = self.series
        if self.types:
            result["types"] = self.types
        if self.author:
            result["author"] = self.author

        # Add stats if they're valid
        stats_dict = self.stats.to_dict()
        for key, value in stats_dict.items():
            if self.stats.is_valid_stat(value):
                result[key] = value

        if self.problems:
            result["problem"] = self.problems
        for key in ("display_name", "note", "hidden", "duplicate_of"):
            if getattr(self, key, None):
                result[key] = getattr(self, key)
        if getattr(self, "creatures", None):
            result["creatures"] = [c.to_dict() for c in self.creatures]

        return result

    @classmethod
    def from_dict(cls, name: str, data: dict) -> 'CardInfo':
        """Create CardInfo from dictionary (for loading from pickle)."""
        # Extract stats
        stats = CardStats(
            hp=data.get("hp", -1),
            defense=data.get("def", -1),
            attack=data.get("atk", -1),
            speed=data.get("spd", -1),
        )

        return cls(
            name=name,
            card_type=data.get("type", CardType.UNKNOWN.value),
            path=data.get("path", ""),
            timestamp=data.get("timestamp", 0.0),
            ability=data.get("ability"),
            stars=data.get("stars"),
            subtype=data.get("subtype"),
            series=data.get("series"),
            types=data.get("types", []),
            stats=stats,
            problems=data.get("problem", []) if isinstance(data.get("problem"), list) else [],
            author=data.get("author"))


class StatsDatabase:
    """Manages card statistics database."""
    def __init__(self):
        self.stats: dict[str, CardInfo] = {}
        self.old_stats: defaultdict[str, list[CardInfo]] = defaultdict(list)
        self.all_types: list[str] = []
        self.dirty_files: list[str] = []

        self._headers = {'Authorization': f'token {GIT_TOKEN}'} if GIT_TOKEN else {}
        if GIT_TOKEN:
            print("Git token found, API limited to 5000 requests/hour.")
        else:
            print("No git token in config, API limited to 60 requests/hour.")

    def save(self) -> None:
        """Save all databases to pickle files."""
        with open(STATS_PKL, 'wb') as f:
            dump(self.stats, f)
        with open(OLD_STATS_PKL, 'wb') as f:
            dump(self.old_stats, f)

    def load(self) -> None:
        """Load all databases from pickle files."""
        self._load_stats()
        self._load_old_stats()

    def _load_stats(self) -> None:
        """Load stats from pickle file."""
        print(f"Trying to open {STATS_PKL}")
        try:
            with open(STATS_PKL, 'rb') as f:
                self.stats = load(f)
            print(f"Loaded existing stats from {STATS_PKL}")
        except (EOFError, FileNotFoundError):
            print(f"{STATS_PKL} is empty or doesn't exist, rebuilding...")
            self.stats = {}

    def _load_old_stats(self) -> None:
        """Load old stats from pickle file."""
        print(f"Trying to open {OLD_STATS_PKL}")
        try:
            with open(OLD_STATS_PKL, 'rb') as f:
                self.old_stats = load(f)
            print(f"Loaded existing old stats from {OLD_STATS_PKL}")
        except (EOFError, FileNotFoundError):
            print(f"{OLD_STATS_PKL} is empty or doesn't exist, starting fresh...")
            self.old_stats = defaultdict(list)

    def prune_clean_cards(self) -> None:
        """Move cards not in dirty_files to old_stats."""
        cards_to_remove = []
        for name, card in self.stats.items():
            if name not in self.dirty_files and not getattr(card, "duplicate_of", None):
                self.old_stats[name].append(card)
                cards_to_remove.append(name)

        for name in cards_to_remove:
            self.stats.pop(name)


class CardClassifier:
    """Classifies cards based on their path."""
    @staticmethod
    def classify(relative_path: str) -> dict:
        """
        Classify card based on its relative path in repository.

        Args:
            relative_path: Relative path to card file
            existing_card: Existing CardInfo if card is being updated

        Returns:
            Dictionary with card classification info
        """
        if not relative_path:
            return {"type": CardType.UNKNOWN.value}

        folders = relative_path.split('/')
        if not folders:
            return {"type": CardType.UNKNOWN.value}

        name = folders.pop()
        top_folder = folders[0]

        # Classification logic
        classifiers = {
            "MDW": lambda: {"type": CardType.MDW.value, "ability": None},
            "Field": lambda: {
                "type": CardType.FIELD.value,
                "stars": int(folders[len(folders) - 1].split()[0]),
            },
            "Items": lambda: {
                "type": CardType.ITEM.value,
                "subtype": folders[len(folders) - 2].lower(),
                "stars": int(folders[len(folders) - 1].split()[0]),
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
    def _classify_creatures(folders: list[str]) -> dict:
        """Classify creature cards, preserving existing series if available."""
        result = {
            "type": CardType.CREATURE.value,
            "stars": int(folders[len(folders) - 1].split()[0]),
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
                    "stars": int(folders[len(folders) - 1].split()[0]),
                }
            elif folders[2] == "Buffs":
                return {"type": CardType.BUFF.value}

        return {"type": folders[len(folders) - 1].lower()}


class PSDParser:
    """Parses PSD files to extract card information."""
    def __init__(self, all_types: list[str], stats_db: StatsDatabase, icons: Optional[IconVocabulary] = None):
        self.all_types = all_types
        self.stats_db = stats_db
        self.icons = icons if icons is not None else IconVocabulary.build(all_types, [], None)
        self._gap_pattern = re_compile(r'\s{3,}')

    def parse(self, file_path: str, relative_path: str = "") -> CardInfo:
        """
        Parse a PSD file to extract card information.

        Args:
            file_path: Full path to PSD file
            relative_path: Relative path in repository

        Returns:
            CardInfo object with extracted data
        """
        name = splitext(basename(relative_path))[0].replace('_', ' ')

        # Get existing card if it exists
        existing_card = self.stats_db.stats.get(name)

        # Initialize card with classification (passing existing card)
        card_dict = CardClassifier.classify(relative_path)

        card = CardInfo(
            name=name,
            card_type=card_dict.get("type", CardType.UNKNOWN.value),
            path=relative_path,
            stars=card_dict.get("stars"),
            subtype=card_dict.get("subtype"),
            series=card_dict.get("series"),
        )

        # Preserve author and series from existing card
        if existing_card and existing_card.author:
            card.author = existing_card.author
        if existing_card and existing_card.series:
            card.series = existing_card.series

        # Set initial stats if present
        if "hp" in card_dict:
            card.stats = CardStats(
                hp=card_dict.get("hp", -1),
                defense=card_dict.get("def", -1),
                attack=card_dict.get("atk", -1),
                speed=card_dict.get("spd", -1),
            )

        # Parse PSD layers
        self._extract_from_layers(PSDImage.open(file_path), card, relative_path)

        return card

    def _extract_from_layers(self, psd: PSDImage, card: CardInfo, relative_path: str) -> None:
        """Extract information from PSD layers."""
        num_stars = MutableValue(0)
        stat_trackers = StatTrackers()
        type_bboxes: list[tuple[str, BoundingBox]] = []
        abilities: list[tuple[str, BoundingBox]] = []

        card_mid_y = int(psd.height * TYPE_REGION_RATIO)

        is_rulepage = "Rulebook" in relative_path
        get_stars_from_psd = any(
            folder in relative_path
            for folder in ["Auxiliary/Items", "Auxiliary/Minions", "N.M.E"]
        )

        effect_layers = []
        text_layers = []
        for layer in psd.descendants():
            if EFFECT_LAYER_TAG in layer.name and layer.kind != "group":
                effect_layers.append(layer)     # a pixel copy of words from a text layer; nothing to parse
                continue
            if layer.kind == "type":
                text_layers.append(layer)
            self._process_layer(
                layer, card, num_stars,
                stat_trackers, type_bboxes, abilities,
                is_rulepage, get_stars_from_psd, card_mid_y)
        for layer in effect_layers:
            self._check_effect_layer(layer, text_layers, card)
        titles = [" ".join((layer.text or "").split()) for layer in text_layers
                  if not layer.is_visible() and layer.bbox[1] < psd.height * TITLE_REGION_RATIO]
        card.alt_names = [t for t in titles if t and len(t) <= 40 and t.lower() != card.name.lower()] or None

        if abilities:
            card.ability = self._process_abilities(abilities, type_bboxes, card_mid_y)
        else:
            card.problems.append("NO ABILITY LAYER")
        self._process_stats(card, stat_trackers)

        if get_stars_from_psd:
            card.stars = num_stars.value + self._count_stars(psd)    # icon-font stars in the header + drawn stars
            if not card.stars:
                card.problems.append('NO STARS FOUND (put the star layers in a group named "Stars")')

    def _check_effect_layer(self, layer, text_layers: list, card: CardInfo) -> None:
        """
        Flag an effect layer that no longer matches its text.

        "Ability FX: Blazing" holds the word Blazing cut pixel-for-pixel from the "Ability" text layer, with a layer
        style on top (MDW/Scripts/Effect Layers.jsx in TTSCardMaker). Editing the text without running the script's
        Update all leaves the glow where the word used to be; then its pixels stop landing on the text's pixels.
        """
        if not layer.is_visible():
            return
        base_name, _, words = layer.name.partition(EFFECT_LAYER_TAG)
        bases = [text_layer for text_layer in text_layers if text_layer.name == base_name]
        if not bases:
            self._add_problem(card, f"EFFECT LAYER WITHOUT TEXT LAYER: {layer.name}")
            return
        x0, y0, x1, y1 = layer.bbox
        base = next((b for b in bases if b.bbox[0] < x1 and x0 < b.bbox[2] and b.bbox[1] < y1 and y0 < b.bbox[3]), bases[0])

        text = sub(r"[\s\x03]+", " ", base.engine_dict["Editor"]["Text"].value).lower()
        # "Frozen#2" means the 2nd occurrence of Frozen
        phrases = [sub(r"\s+", " ", sub(r"#\d+$", "", word.strip())).lower() for word in words.split("|")]
        if any(phrase and phrase not in text for phrase in phrases):
            self._add_problem(card, f"EFFECT LAYER WORDS NOT IN TEXT: {layer.name}")
            return
        coverage = self._pixel_coverage(layer, base)
        if coverage is not None and coverage < EFFECT_COVERAGE_MIN:
            self._add_problem(card, f"EFFECT LAYER OUT OF DATE: {layer.name} (run Effect Layers > Update all)")

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
                       get_stars_from_psd: bool, card_mid_y: int) -> None:
        """Process a single PSD layer."""
        # Text layers
        if layer.kind == "type":
            self._process_text_layer(layer, card, num_stars, abilities, is_rulepage, get_stars_from_psd, card_mid_y)

        # Stat layers
        elif self._is_stat_layer(layer):
            self._process_stat_layer(layer, stat_trackers)

        # Type layers
        elif (layer.name.lower() in self.all_types and layer.is_visible() and layer.has_pixels()):
            bbox = BoundingBox.from_tuple(layer.bbox[:2])
            if bbox.y < card_mid_y:
                card.types.append(layer.name.lower())
            else:
                type_bboxes.append((layer.name.lower(), bbox))

        # Star layers: counted per star by _count_stars once the walk is done
        elif get_stars_from_psd and self._is_star_layer(layer):
            pass

        # Problematic type layer
        elif (layer.name.lower() in MISSPELT_CARD_TYPES and layer.is_visible() and layer.has_pixels()):
            card.problems.append(f"MISSPELT TYPE: {layer.name.lower()}")

    def _process_text_layer(self, layer, card: CardInfo, num_stars: MutableValue, abilities: list,
                            is_rulepage: bool, get_stars_from_psd: bool, card_mid_y: int) -> None:
        """
        Process text layer.

        Text set in the stoogedings font is read as icons: [fire]-style tokens (any case, spacing or alias) become
        canonical [fire] tokens, and icon characters typed from the Glyphs panel become tokens too. Header text in
        the font also adds to card.types, and to the star count where stars come from the PSD. Layers without the
        font take the original path unchanged.
        """
        engine_text = layer.engine_dict["Editor"]["Text"]
        is_ability = layer.name.lower() == "ability" or is_rulepage
        icon_spans = self._icon_font_spans(layer, engine_text.value)
        self._flag_misplaced_tokens(engine_text.value, icon_spans, card)
        if icon_spans:
            text, icons = self._read_icon_tokens(engine_text.value, icon_spans, card)
            layer_text = repr(text)  # str() of the engine string is its repr, so the cleanup below is unchanged
            if not is_ability and layer.is_visible() and layer.bbox[1] < card_mid_y:
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

        "[Fire:Sticker]" -> "[fire]", "[Defensive Minion]" -> "[defensive_minion]", "[fire:no]" stays "[fire:no]",
        and an icon character from the Glyphs panel -> "[fire]". Unknown tokens are left as typed and reported,
        because the font shows them as plain letters on the card.

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
                base, _, variant = piece[1:len(piece) - 1].partition(":")
                variant = variant.lower()           # the font accepts no spaces around the variant either
                hit = self.icons.lookup(base)
                if hit is None or variant not in ICON_VARIANTS:
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
            base, _, variant = piece[1:len(piece) - 1].partition(":")
            if self.icons.lookup(base) is None or variant.lower() not in ICON_VARIANTS:
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
        if not layer.name.isdigit():
            return
        # A digit-named group of digit layers is a bar row (multi-row bars name rows "1", "2", ...), not a value.
        # Sonic's two speed rows "2" and "1" were added as 3 more speed. An empty one draws nothing (Twin Emperors'
        # "shared health" has an empty "1" group, which made its 7 health 8).
        if layer.is_group() and (len(layer) == 0 or any(child.name.isdigit() for child in layer)):
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
                row = tracker.rows.setdefault(id(parent), BarRow(id(block), parent.name, getattr(block, "name", "") or "",
                                                                 id(getattr(block, "parent", None))))
                if layer.is_visible():
                    row.items.append((int(layer.name), layer.kind))
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
                           type_bboxes: list[tuple[str, BoundingBox]], card_mid_y: int) -> str | None:
        """Process and combine ability texts."""
        abilities = self._sort_by_position(abilities)
        ability_text = '\n'.join(text.strip('\'"\n') for text, _ in abilities)

        if ability_text:
            type_bboxes = self._prune_type_bboxes(self._sort_by_position(type_bboxes), card_mid_y)
            ability_text = self._inject_type_names(ability_text.rstrip(' \'"'), type_bboxes)
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
    EXCESSIVE_STAT_EXCLUSIONS: set[str] = {
        "Royal Eradicator Main Cannon",
        "Pix",
        "Tainted Lazarus",
        "Sonic",
        "Twin Emperors",
    }
    ABILITY_EXCLUSIONS: set[str] = {
        "Crystal Bits",
        "Bugzzy",
        "Electro Probe",
        "Galacta Warrior",
        "God Tamer",
        "Mini Bee",
        "Paint Warrior",
        "PoD01Red Bloon",
        "Sabre",
        "Shadow Duelist",
        "Sword Knight",
        "Tentacle",
        "Whelp",
        "Warrior Dee",
        "The Master",
        "Panda",
        "Chomp",
        "Abyss Watcher",
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

        # Ability problems
        if (card.card_type != CardType.UNKNOWN.value and
            card.card_type != CardType.MDW.value and
            not card.ability and
            card.name not in CardValidator.ABILITY_EXCLUSIONS):
            problems.append("ABILITY TEXT NOT FOUND")

        # Stat problems
        if card.card_type in [CardType.CREATURE.value, CardType.MINION.value]:
            problems.extend(CardValidator._validate_stats(card))

        # Add any problems found during parsing
        if card.problems:
            if ("NO ABILITY LAYER" in card.problems and
                card.name in CardValidator.ABILITY_EXCLUSIONS):
                card.problems.remove("NO ABILITY LAYER")
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

        # Check for excessive stats — but skip excluded cards
        if (stats.has_excessive_stats() and
            card.name not in CardValidator.EXCESSIVE_STAT_EXCLUSIONS and
            not getattr(card, "allow_high_stats", False)):
            problems.append("STATS TOO HIGH")

        return problems


class RepositoryTraverser:
    """Traverses repository to update card stats."""
    def __init__(self, db: StatsDatabase):
        self.db = db
        self.parser: Optional[PSDParser] = None
        self._github_client: Optional[Github] = None
        self.edits: Optional[CardEdits] = None
        self.changed = 0                         # cards changed without a re-parse (authors, edits, duplicates)
        self.only: set = set()                   # card names to re-parse anyway (/update_stats cards:)
        self.repository = ""
        self.shas: dict = {}                     # repo path -> blob sha, from the tree
        self.tree: Optional[dict] = None         # the tree, reused by card_repo.populate_files
        self._repo_obj = None
        self._recent: Optional[set] = None

    def _update_old_stats_paths(self) -> None:
        """
        Update paths in old_stats to match current paths.
        This ensures historical entries reflect where the card is now located.
        """
        for name in self.db.old_stats:
            if name in self.db.stats:
                current_path = self.db.stats[name].path
                # Update all historical entries to use the current path
                for old_card in self.db.old_stats[name]:
                    old_card.path = current_path

    def _psd_changed(self, name: str, path: str, sha: Optional[str]) -> bool:
        """Changed = new, moved or a different blob. Cards saved before blob shas: committed since the last update."""
        card = self.db.stats.get(name)
        if card is None or card.path != path:
            return True
        if getattr(card, "blob_sha", None):
            return card.blob_sha != sha
        recent = self._committed_since_update()
        if recent is None or path in recent:
            return True
        card.blob_sha = sha                       # unchanged: remember its blob
        self.changed += 1
        return False

    def _committed_since_update(self) -> Optional[set]:
        """Paths committed after the newest stored card timestamp (one pass over those commits), or None."""
        if self._recent is None:
            newest = max((c.timestamp for c in self.db.stats.values() if (c.timestamp or 0) > 0), default=0)
            if not newest:
                return None
            repo = Github(login_or_token=GIT_TOKEN).get_repo(self.repository)
            self._recent = {f.filename for commit in repo.get_commits(since=datetime.fromtimestamp(newest, timezone.utc))
                            for f in commit.files}
        return self._recent

    def _repo(self) -> Repository.Repository:
        if self._repo_obj is None:
            self._github_client = Github(login_or_token=GIT_TOKEN, per_page=1)    # one commit per request
            self._repo_obj = self._github_client.get_repo(self.repository)
        return self._repo_obj

    def _blob(self, sha: str, stream: bool = False):
        """A file's exact content by blob sha (raw.githubusercontent can lag behind the tree)."""
        reply = get(f"https://api.github.com/repos/{self.repository}/git/blobs/{sha}", stream=stream, timeout=120,
                    headers={**self.db._headers, "Accept": "application/vnd.github.raw"})
        reply.raise_for_status()
        return reply

    def _repo_text(self, path: str) -> Optional[str]:
        """A small repo file's text; downloaded again only when its blob changed."""
        sha = self.shas.get(path)
        if sha is None:
            return None
        cached = _file_cache.get(path)
        if not cached or cached[0] != sha:
            reply = self._blob(sha)
            reply.encoding = "utf-8"
            _file_cache[path] = cached = (sha, reply.text)
        return cached[1]

    def _download(self, path: str, sha: str) -> Optional[str]:
        """A PSD into a temp file, or None."""
        try:
            with self._blob(sha, stream=True) as reply:
                fd, name = mkstemp(suffix=".psd")
                with fdopen(fd, "wb") as fh:
                    for chunk in reply.iter_content(1 << 20):
                        fh.write(chunk)
            return name
        except (RequestException, OSError) as e:
            print(f"{path}: download failed ({e})")
            return None

    def traverse_remote(self,
                        repository: str,
                        interaction: Optional[Interaction] = None,
                        output_problematic: bool = True,
                        force_update: bool = False) -> tuple[list[str], int]:
        """
        Traverse remote GitHub repository.

        Args:
            repository: Repository name (owner/repo)
            interaction: Optional Discord interaction for progress updates
            output_problematic: Whether to output problematic cards

        Returns:
            List of formatted problem card messages
        """
        self.repository = repository
        headers = dict(self.db._headers)
        if not force_update and not self.only and _tree_etags.get(repository):
            headers["If-None-Match"] = _tree_etags[repository]       # 304 when nothing changed (free)
        resp = get(f"https://api.github.com/repos/{repository}/git/trees/main?recursive=1", headers=headers, timeout=30)
        if resp.status_code == 304:
            msg = "Nothing changed in the card repo since the last update."
            queue_message(interaction, msg) if interaction else print(msg)
            return ([], 0)
        resp.raise_for_status()
        self.tree = resp.json()
        self.shas = {item.get("path", ""): item.get("sha") for item in self.tree.get("tree", [])}

        self._populate_types_from_response(resp)
        self.parser = PSDParser(self.db.all_types, self.db, self._icons_from_response(resp, repository))

        self.edits = self._edits_from_response(resp, repository)
        problems, num_new = self._process_files_from_response(resp, repository, interaction, output_problematic, force_update)
        self.changed += self._apply_authors(self._authors_from_response(resp, repository))
        self.changed += self._apply_edits()
        self.changed += self._sync_duplicates(resp)
        self.changed += self._apply_edits()             # the duplicates' own edits

        # Update old_stats paths after processing all files
        self._update_old_stats_paths()
        _tree_etags[repository] = resp.headers.get("ETag")

        return (problems, num_new)

    def _populate_types_from_response(self, response) -> None:
        """Populate card types from API response."""
        self.db.all_types.clear()
        for item in response.json().get("tree", []):
            path = item.get("path", "")
            if path.startswith("Types") and "Stars" not in path and '.' in path:
                self.db.all_types.append(splitext(basename(path))[0].lower())

    def _populate_types_from_local(self, local_path: str) -> None:
        """Populate card types from a local clone (offline diagnostics; /update_stats itself only reads GitHub)."""
        self.db.all_types.clear()
        types_dir = path_join(local_path, "Types")
        for folder, _, files in walk(types_dir):
            if folder.endswith("Types"):
                for file in files:
                    if not file.startswith('.'):
                        self.db.all_types.append(splitext(file)[0].lower())

    @staticmethod
    def _star_name(path: str) -> str:
        """Types/Stars/Dst Star.png -> "dst_star" (the name the icon font uses)."""
        return splitext(basename(path))[0].lower().replace(" ", "_")

    def _icons_from_response(self, response, repository: str) -> IconVocabulary:
        """Icon vocabulary from the GitHub tree: Types/ and Types/Stars/ names plus the font's manifest, if any."""
        paths = [item.get("path", "") for item in response.json().get("tree", [])]
        stars = [self._star_name(p) for p in paths if p.startswith("Types/Stars/") and p.lower().endswith(".png")]
        manifest = None
        if ICON_MANIFEST_PATH in paths:
            try:
                manifest = json_loads(self._repo_text(ICON_MANIFEST_PATH))
            except (RequestException, ValueError) as e:
                print(f"Could not read {ICON_MANIFEST_PATH}, icon aliases and codepoints unavailable: {e}")
        return IconVocabulary.build(self.db.all_types, stars, manifest)

    def _authors_from_response(self, response, repository: str) -> Optional[CardAuthors]:
        """The card repo's authors file (AUTHORS_PATH), or None if it's missing or unreadable."""
        paths = [item.get("path", "") for item in response.json().get("tree", [])]
        if AUTHORS_PATH not in paths:
            return None
        try:
            return CardAuthors.parse(self._repo_text(AUTHORS_PATH))
        except RequestException as e:
            print(f"Could not read {AUTHORS_PATH}; authors stay as they are: {e}")
            return None

    def _apply_authors(self, authors: Optional[CardAuthors]) -> int:
        """Set every card's author from the authors file (all cards, not just re-parsed ones). Returns how many changed."""
        if authors is None:
            return 0
        changed = 0
        for name, card in self.db.stats.items():
            author = authors.resolve(name, card.author)
            if author != card.author:
                card.author = author
                changed += 1
        if changed:
            print(f"{changed} authors set from {AUTHORS_PATH}.")
        return changed

    def _edits_from_response(self, response, repository: str) -> Optional[CardEdits]:
        """The card repo's website fixes (EDITS_PATH), or None if missing or unreadable."""
        paths = [item.get("path", "") for item in response.json().get("tree", [])]
        if EDITS_PATH not in paths:
            return None
        try:
            return CardEdits.parse(self._repo_text(EDITS_PATH))
        except (RequestException, ValueError) as e:
            print(f"Could not read {EDITS_PATH}; edits stay as they are: {e}")
            return None

    def _sync_duplicates(self, response) -> int:
        """
        Duplicates: <Name>N.png next to <Name>.psd with no PSD of its own (A_Snack2.png "Breakfast"). Each is
        recorded as a copy of the original's card with duplicate_of set. Returns how many entries changed.
        """
        paths = [item.get("path", "") for item in response.json().get("tree", [])]
        lower = {p.lower() for p in paths}
        found = {}
        for path in paths:
            m = re_match(r"(.+?)(\d+)\.png$", path, IGNORECASE)
            if (not m or any(folder in EXCLUDE_FOLDERS for folder in path.split("/")) or "Rulebook" in path
                    or f"{m.group(1)}{m.group(2)}.psd".lower() in lower or f"{m.group(1)}.psd".lower() not in lower):
                continue
            original = basename(m.group(1)).replace("_", " ")
            if original in self.db.stats and not self.db.stats[original].duplicate_of:
                found[splitext(basename(path))[0].replace("_", " ")] = (original, int(m.group(2)), basename(path).lower())
        before = {n: repr(c) for n, c in self.db.stats.items() if getattr(c, "duplicate_of", None)}
        for name in before:
            if name not in found:
                del self.db.stats[name]
        siblings: dict = {}
        for name, (original, number, _) in found.items():
            siblings.setdefault(original, []).append(number)
        for name, (original, number, png) in found.items():
            src = self.db.stats[original]
            card = deepcopy(src)
            card.name, card.duplicate_of, card.parsed_values, card.problems, card.alt_names = name, original, None, [], None
            card.display_name = self._duplicate_name(png, src, number, sorted(siblings[original]))
            self.db.stats[name] = card
            self.db.dirty_files.append(name)            # not pruned as a missing PSD
        return sum(1 for n in found if before.get(n) != repr(self.db.stats[n])) + sum(1 for n in before if n not in found)

    @staticmethod
    def _duplicate_name(png: str, original: CardInfo, number: int, numbers: list) -> str:
        """The /alias pointing at this PNG, else the PSD's hidden title in order, else the original's name."""
        aliases = [a for a, target in card_repo.git_file_alias.items() if target.lower() == png]
        if aliases:
            return " ".join(w[:1].upper() + w[1:] for w in max(aliases, key=len).replace("_", " ").split())
        titles = getattr(original, "alt_names", None) or []
        if len(titles) == len(numbers):
            return titles[numbers.index(number)]
        return original.display_name or original.name

    def _apply_edits(self) -> int:
        """Apply the website fixes to every card. Returns how many changed."""
        if self.edits is None:
            return 0
        changed = sum(1 for name, card in self.db.stats.items() if self.edits.apply(name, card))
        if changed:
            print(f"{changed} cards changed by {EDITS_PATH}.")
        return changed

    def _icons_from_local(self, local_path: str) -> IconVocabulary:
        """Icon vocabulary from a local clone: Types/ and Types/Stars/ names plus the font's manifest, if any."""
        stars_dir = path_join(local_path, "Types", "Stars")
        stars = [self._star_name(f) for f in listdir(stars_dir) if f.lower().endswith(".png")] if isdir(stars_dir) else []
        manifest = None
        manifest_path = path_join(local_path, ICON_MANIFEST_PATH)
        if isfile(manifest_path):
            try:
                with open(manifest_path, encoding="utf-8") as fh:
                    manifest = json_loads(fh.read())
            except (OSError, ValueError) as e:
                print(f"Could not read {manifest_path}, icon aliases and codepoints unavailable: {e}")
        return IconVocabulary.build(self.db.all_types, stars, manifest)

    def _construct_raw_url(self, repository: str, path: str) -> str:
        """
        Construct a raw GitHub URL from a repository path.

        Args:
            path: Relative path within the repository (e.g., "Creatures/Kirby/5 Star/Magolor.psd")

        Returns:
            Full raw GitHub URL
        """
        return f"https://raw.githubusercontent.com/{repository}/main/{quote(path, safe='/')}"

    def _process_files_from_response(self,
                                     response,
                                     repository: str,
                                     interaction: Optional[Interaction],
                                     output_problematic: bool,
                                     force_update: bool) -> tuple[list[str], int]:
        """Re-parse the PSDs whose blob changed (all with force_update). Unchanged cards cost no requests."""
        problematic_cards = []
        num_updated = num_new = num_old = num_moved = 0
        todo = []
        for item in response.json().get("tree", []):
            path = item.get("path", "")
            if not path.endswith('.psd') or any(folder in EXCLUDE_FOLDERS for folder in path.split('/')):
                continue
            name = splitext(basename(path))[0].replace('_', ' ')
            self.db.dirty_files.append(name)
            if force_update or CardAuthors.key(name) in self.only or self._psd_changed(name, path, item.get("sha")):
                todo.append((path, name, item.get("sha")))
            else:
                num_old += 1

        pool = ThreadPoolExecutor(DOWNLOAD_WORKERS)
        downloads = [pool.submit(self._download, path, sha) for path, _, sha in todo]
        try:
            for (path, name, sha), download in zip(todo, downloads):
                local_file = download.result()
                if local_file is None:
                    continue
                try:
                    timestamp, author = self._get_remote_timestamp(self._repo(), path, name)
                    card = self.parser.parse(local_file, path)
                finally:
                    remove(local_file)
                old = self.db.stats.get(name)
                if old is None:
                    num_new += 1
                else:
                    if old.path != path or getattr(old, "blob_sha", None) != sha:
                        self.db.old_stats[name].append(old)     # history only when the PSD changed or moved
                    if old.path != path:
                        num_moved += 1
                    else:
                        num_new += 1
                card.timestamp = timestamp
                card.blob_sha = sha
                if author and not card.author:
                    card.author = author
                if self.edits:
                    self.edits.apply(name, card)
                self.db.stats[name] = card

                if output_problematic:
                    problems = CardValidator.validate(card)
                    if problems:
                        problematic_cards.append((path, card, problems))

                num_updated += 1
                if num_updated % UPDATE_RATE == 0:
                    self._send_progress(interaction, num_updated)
        finally:
            for download in downloads:
                download.cancel()
            pool.shutdown(wait=True)
            for download in downloads:                          # temp files left by an error
                if download.done() and not download.cancelled() and download.result() and isfile(download.result()):
                    remove(download.result())

        self._send_summary(interaction, num_new, num_old, num_moved)
        return (self._format_problems(problematic_cards), num_new + num_moved)

    def _get_remote_timestamp(self,
                              repo: Optional[Repository.Repository],
                              path: str,
                              name: str) -> tuple[float, str]:
        """
        Get timestamp for a file from GitHub.

        Uses the most recent commit for timestamp, but preserves original author
        from the first commit if not already set.
        """
        if not repo:
            return (datetime(1000, 1, 1).timestamp(), None)

        commits = repo.get_commits(path=path)
        if commits.totalCount == 0:
            return (datetime(1000, 1, 1).timestamp(), None)

        # New card: its author is whoever wrote the FIRST commit (not who committed it). The authors file
        # (AUTHORS_PATH) corrects either afterwards.
        if name not in self.db.stats or not self.db.stats[name].author:
            return (commits[0].commit.committer.date.timestamp(), commits[commits.totalCount - 1].commit.author.name)

        return (commits[0].commit.committer.date.timestamp(), None)

    @staticmethod
    def _send_progress(interaction: Optional[Interaction], count: int) -> None:
        """Send progress update."""
        message = f"{count} cards updated."
        if interaction:
            queue_edit(interaction, content=message)
        else:
            print(message)

    @staticmethod
    def _send_summary(interaction: Optional[Interaction],
                      num_new: int,
                      num_old: int,
                      num_moved: int = 0) -> None:
        """Send summary of update."""
        msg = f"{num_new} cards changed or new, {num_old} unchanged, {num_moved} moved."

        if interaction:
            queue_message(interaction, msg)
        else:
            print(msg)

    @staticmethod
    def _format_problems(
        problematic_cards: list[tuple[str, CardInfo, list[str]]]
    ) -> list[str]:
        """Format problematic cards for display."""
        return [
            f"{path}\n```\n" + '\n'.join(problems) + "\n```"
            for path, _, problems in problematic_cards
        ]


# ========================
# Helper Functions
# ========================
def get_card_aliases(card_name: str = None) -> dict[str, list[str]] | list[str]:
    """
    Build a reverse lookup of cards to aliases.

    Args:
        card_name: Optional specific card name to get aliases for.
                   If None, returns dict mapping all card names to their aliases.

    Returns:
        If card_name provided: list of aliases for that card
        If card_name is None: dict mapping card names to lists of aliases
    """
    card_to_aliases = {}

    # Build reverse lookup using get_card_path
    for alias in card_repo.git_file_alias.keys():
        # Use get_card_path to find the actual card this alias points to
        path = card_repo.get_card_path(alias)
        if path:
            # Extract card name from path
            target_name = basename(path).replace('_', ' ')

            if target_name not in card_to_aliases:
                card_to_aliases[target_name] = []
            card_to_aliases[target_name].append(alias)

    # If specific card requested, return just its aliases
    if card_name is not None:
        return sorted(card_to_aliases.get(card_name, []))

    # Otherwise return the full mapping
    return card_to_aliases


def get_stats_as_dict() -> dict[str, dict]:
    """
    Get the stats database as a dictionary format.
    Useful for pandas DataFrame operations and external integrations.

    Returns:
        Dictionary mapping card names to their dict representations
    """
    return {name: card.to_dict() for name, card in stats_db.stats.items()}


def get_old_stats_as_dict() -> dict[str, list[dict]]:
    """
    Get the old_stats database as a dictionary format.

    Returns:
        Dictionary mapping card names to lists of their historical dict representations
    """
    return {
        name: [card.to_dict() for card in cards]
        for name, cards in stats_db.old_stats.items()
    }


# ========================
# Public API Functions
# ========================
def update_stats(interaction: Optional[Interaction] = None,
                 output_problematic: bool = True,
                 force_update: bool = False,
                 verbose: bool = True,
                 cards: str = "") -> list[str]:
    """
    Update card statistics database from the GitHub repository.

    Local mode (walking the clone on disk) was archived on 2026-10-20 to branch archive/local-update-stats.

    Args:
        interaction: Optional Discord interaction
        output_problematic: Whether to output problematic cards
        force_update: Re-parse every card even if its PSD did not change
        cards: Comma-separated card names to re-parse anyway (e.g. after a parser fix)
        verbose: Whether to display progress messages and problem summaries

    Returns:
        List of formatted problem messages
    """
    def _notify(msg: str):
        """Send a message to Discord or print locally."""
        if interaction:
            queue_message(interaction, msg)
        else:
            print(msg)

    if verbose:
        _notify("Updating card statistics database... This may take a while.")

    stats_db.load()
    stats_db.dirty_files.clear()
    traverser = RepositoryTraverser(stats_db)
    traverser.only = {CardAuthors.key(c) for c in (cards or "").split(",") if c.strip()}

    problem_cards, num_new = traverser.traverse_remote(
        card_repo.repository,
        interaction,
        output_problematic,
        force_update)

    if num_new > 0:
        stats_db.prune_clean_cards()
    if num_new > 0 or traverser.changed:
        stats_db.save()

    if verbose:
        _notify("Card statistics update complete!")

    # Output problems
    if verbose and problem_cards:
        # Detailed problems
        bundle = "".join(problem_cards)
        for chunk in split_long_message(bundle):
            _notify(chunk)

        # Card names only
        for card in problem_cards:
            _notify(card.split('```')[0])

    # Always refresh card_repo stats
    queue_command(card_repo.populate_files, traverser.tree)
    queue_command(card_repo.prep_dataframes)

    return problem_cards


def list_orphans(interaction: Interaction) -> None:
    """
    List all cards without an author.

    Args:
        interaction: Discord interaction
    """
    orphans = [
        name for name, card in stats_db.stats.items()
        if not card.author
    ]

    if not orphans:
        queue_message(interaction, "No orphaned cards found!")
        return

    bundle = '\n'.join(orphans)
    for chunk in split_long_message(bundle):
        queue_message(interaction, chunk)


def mass_replace_field(interaction: Interaction,
                       field: str = "",
                       old_value: str = "",
                       new_value: str = "") -> None:
    """
    Replace all instances of one field value with another.

    Args:
        interaction: Discord interaction
        field: Field name to modify (e.g., 'author', 'series', 'subtype', 'card_type')
        old_value: Value to replace
        new_value: New value
    """
    # Validate field name
    valid_fields = ['author', 'series', 'subtype', 'card_type']
    if field not in valid_fields:
        queue_message(interaction, f"Invalid field '{field}'. Valid fields are: {', '.join(valid_fields)}")
        return

    num_replaced = 0
    for _, card in stats_db.stats.items():
        current_value = getattr(card, field, None)

        # Handle case-insensitive comparison for string fields
        if isinstance(current_value, str) and current_value.lower() == old_value.lower():
            setattr(card, field, new_value)
            num_replaced += 1

    stats_db.save()
    queue_message(interaction, f"Replaced {num_replaced} instances of {field}='{old_value}' with '{new_value}'.")


def manual_metadata_entry(interaction: Interaction,
                          query: str,
                          del_entry: bool = False,
                          author: Optional[str] = None,
                          ability: Optional[str] = None,
                          stars: Optional[int] = None,
                          subtype: Optional[str] = None,
                          series: Optional[str] = None,
                          hp: Optional[int] = None,
                          defense: Optional[int] = None,
                          attack: Optional[int] = None,
                          speed: Optional[int] = None,
                          card_type: Optional[str] = None,
                          types: Optional[str] = None) -> None:
    """
    Manually edit metadata for a card.

    Args:
        interaction: Discord interaction
        query: Card name query
        del_entry: Whether to delete the entire entry
        author: Card author/creator
        ability: Card ability text
        stars: Star count
        subtype: Card subtype
        series: Card series
        hp: HP stat
        defense: Defense stat
        attack: Attack stat
        speed: Speed stat
        card_type: Card type
        types: Comma-separated list of types (e.g., "fire,water")
    """
    path = card_repo.get_card_path(query)
    if not path:
        queue_message(interaction, f"No card found matching '{query}'.")
        return

    name = basename(path).replace('_', ' ')

    # Handle deletion
    if del_entry:
        if name in stats_db.stats:
            del stats_db.stats[name]
            stats_db.save()
        queue_message(interaction, f"Deleted metadata for **{name}**.")
        return

    # Initialize metadata entry if it doesn't exist
    if name not in stats_db.stats:
        stats_db.stats[name] = CardInfo(
            name=name,
            card_type=CardType.UNKNOWN.value,
            path=f"{path}.psd")

    card = stats_db.stats[name]

    # Track what was updated
    updates = []

    # Update all provided fields
    if author is not None:
        card.author = author
        updates.append(f"author: {author}")

    if ability is not None:
        card.ability = ability
        updates.append(f"ability: {ability[:50]}..." if len(ability) > 50 else f"ability: {ability}")

    if stars is not None:
        card.stars = stars
        updates.append(f"stars: {stars}")

    if subtype is not None:
        card.subtype = subtype
        updates.append(f"subtype: {subtype}")

    if series is not None:
        card.series = series
        updates.append(f"series: {series}")

    if hp is not None:
        card.stats.hp = hp
        updates.append(f"hp: {hp}")

    if defense is not None:
        card.stats.defense = defense
        updates.append(f"def: {defense}")

    if attack is not None:
        card.stats.attack = attack
        updates.append(f"atk: {attack}")

    if speed is not None:
        card.stats.speed = speed
        updates.append(f"spd: {speed}")

    if card_type is not None:
        card.card_type = card_type
        updates.append(f"type: {card_type}")

    if types is not None:
        # Parse comma-separated types into a list
        types_list = [t.strip() for t in types.split(',') if t.strip()]
        card.types = types_list
        updates.append(f"types: {types_list}")

    # If no updates were provided, show current metadata
    if not updates:
        metadata_dict = card.to_dict()

        # Get aliases for this card
        aliases = get_card_aliases(name)
        if aliases:
            metadata_dict['aliases'] = ', '.join(aliases)

        # Build sorted metadata string
        sorted_metadata = []
        for key in COLUMN_ORDER:
            if key in metadata_dict:
                sorted_metadata.append(f"{key}: {metadata_dict[key]}")

        # Add any remaining keys not in COLUMN_ORDER
        for key, value in metadata_dict.items():
            if key not in COLUMN_ORDER:
                sorted_metadata.append(f"{key}: {value}")

        pretty_metadata = '\n'.join(sorted_metadata)
        queue_message(interaction, f"Current metadata for **{name}**:\n```{pretty_metadata}```")
        return

    stats_db.save()

    # Format success message
    updates_text = '\n'.join(f"  • {update}" for update in updates)
    queue_message(
        interaction,
        f"Updated metadata for **{name}**:\n{updates_text}"
    )


def view_old_metadata(interaction: Interaction,
                     query: str,
                     version: int = 0) -> None:
    """
    View old metadata for a card from the history.

    Args:
        interaction: Discord interaction
        query: Card name query
        version: Version index (-1 for oldest, 0 for most recent old version,
                1 for second most recent, etc.)
    """
    path = card_repo.get_card_path(query)
    if not path:
        queue_message(interaction, f"No card found matching '{query}'.")
        return

    name = basename(path).replace('_', ' ')

    # Check if card has old versions
    if name not in stats_db.old_stats or not stats_db.old_stats[name]:
        queue_message(interaction, f"No old versions found for **{name}**.")
        return

    old_versions = stats_db.old_stats[name]

    # Handle version indexing
    # -1 means oldest (first in list)
    # 0 means newest old version (last in list)
    # positive numbers count back from most recent
    if version == -1:
        index = 0
    elif version >= 0:
        index = len(old_versions) - 1 - version
    else:
        # Negative numbers other than -1: count from beginning
        index = abs(version) - 1

    # Validate index
    if index < 0 or index >= len(old_versions):
        queue_message(
            interaction,
            f"Invalid version index. **{name}** has {len(old_versions)} old version(s).\n"
            f"Use -1 for oldest, 0 for most recent old version, or 1-{len(old_versions)-1} for older versions."
        )
        return

    card = old_versions[index]
    metadata_dict = card.to_dict()

    # Get aliases for this card
    aliases = get_card_aliases(name)
    if aliases:
        metadata_dict['aliases'] = ', '.join(aliases)

    # Add version info
    version_text = "oldest" if index == 0 else f"version {len(old_versions) - index - 1}"
    if index == len(old_versions) - 1:
        version_text = "most recent old version"

    # Build sorted metadata string
    sorted_metadata = []
    for key in COLUMN_ORDER:
        if key in metadata_dict:
            sorted_metadata.append(f"{key}: {metadata_dict[key]}")

    # Add any remaining keys not in COLUMN_ORDER
    for key, value in metadata_dict.items():
        if key not in COLUMN_ORDER:
            sorted_metadata.append(f"{key}: {value}")

    pretty_metadata = '\n'.join(sorted_metadata)

    queue_message(
        interaction,
        f"Old metadata for **{name}** ({version_text} of {len(old_versions)}):\n```{pretty_metadata}```"
    )


def export_stats_to_file(interaction: Interaction,
                         only_ability: bool = False,
                         as_csv: bool = True) -> None:
    """
    Export card statistics to a file (excluding rulebook entries).

    Args:
        interaction: Discord interaction
        only_ability: Export only ability text
        as_csv: Export as CSV (otherwise text file)
    """
    # Filter out rulebook entries and convert to dict format
    filtered_stats = {
        name: card.to_dict() for name, card in stats_db.stats.items()
        if "Rulebook" not in card.path
    }

    # Get all card aliases using helper function
    card_to_aliases = get_card_aliases()

    # Add aliases to filtered_stats
    for name in filtered_stats:
        if name in card_to_aliases:
            filtered_stats[name]['aliases'] = ', '.join(sorted(card_to_aliases[name]))
        else:
            filtered_stats[name]['aliases'] = ''

    if as_csv:
        df = pd.DataFrame.from_dict(filtered_stats).transpose()
        df_filtered = df[df["ability"].notna()].copy()

        # Reorder columns (only include columns that exist)
        existing_columns = [col for col in COLUMN_ORDER if col in df_filtered.columns]
        df_filtered = df_filtered[existing_columns]

        # Sort alphabetically by path
        df_filtered = df_filtered.sort_values(
            by='path',
            key=lambda x: x.str.lower()
        )

        filename = f"{EXPORTED_STATS_NAME}.csv"
        df_filtered.to_csv(filename)

        with open(filename, 'rb') as f:
            queue_file(interaction, File(f, filename))
        return

    filename = f"{EXPORTED_STATS_NAME}.txt"
    ascii_pattern = re_compile(r'[^\x00-\x7f]')

    # Sort by type, then alphabetically within type
    sorted_cards = sorted(
        stats_db.stats.items(),
        key=lambda x: (x[1].card_type, x[0].lower())
    )

    with open(filename, 'w') as f:
        current_type = None
        for name, card in sorted_cards:
            if "Rulebook" in card.path:
                continue

            card_type = card.card_type

            # Add section header when type changes
            if card_type != current_type:
                f.write(f"=== {card_type.upper()} ===\n\n")
                current_type = card_type

            f.write(f"{name}\n")

            # Add aliases right after name
            if name in card_to_aliases:
                aliases_str = ', '.join(sorted(card_to_aliases[name]))
                f.write(f"Aliases: {aliases_str}\n")

            if only_ability:
                if card.ability:
                    cleaned = ascii_pattern.sub('', card.ability)
                    f.write(f"{cleaned}\n")
            else:
                card_dict = card.to_dict()
                for _, metric in card_dict.items():
                    if isinstance(metric, list):
                        f.write(' '.join(metric) + '\n')
                    else:
                        cleaned = ascii_pattern.sub('', str(metric))
                        f.write(f"{cleaned}\n")
            f.write('\n')

    with open(filename, 'rb') as f:
        queue_file(interaction, File(f, filename))


def export_rulebook_to_file(interaction: Interaction) -> None:
    """
    Export rulebook pages to a text file.

    Args:
        interaction: Discord interaction
    """
    filename = f"{EXPORTED_RULES_NAME}.txt"
    ascii_pattern = re_compile(r'[^\x00-\x7f]')

    with open(filename, 'w') as f:
        for name, card in stats_db.stats.items():
            if "Rulebook" in card.path:
                f.write(f"{name}\n")
                if card.ability:
                    cleaned = ascii_pattern.sub('', card.ability)
                    f.write(f"{cleaned}\n")
                f.write('\n')

    with open(filename, 'rb') as f:
        queue_file(interaction, File(f, filename))


# ========================
# Initialization
# ========================
stats_db = StatsDatabase()
def init_psd() -> None:
    """Initialize the stats database."""
    stats_db.load()
    card_repo.prep_dataframes()
    print("Stats database initialized.")


# ========================
# Module Exports
# ========================
__all__ = [
    'stats_db',
    'get_stats_as_dict',
    'get_old_stats_as_dict',
    'update_stats',
    'list_orphans',
    'mass_replace_field',
    'manual_metadata_entry',
    'view_old_metadata',
    'export_stats_to_file',
    'export_rulebook_to_file',
    'init_psd',
]
