"""
The card database's data: CardType, CardStats and CardInfo (the two pickled classes), StatsDatabase and its singleton
stats_db. Pure data, so anything may import it (no Discord, network, pandas or psd-tools).
"""
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from os.path import basename, splitext
from pickle import load, dump
from typing import Optional

from global_config import STATS_PKL, OLD_STATS_PKL


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
    subseries: Optional[str] = None           # from Card Edits.json or /update_metadata, never the PSD

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
        if getattr(self, "subseries", None):      # older pickles lack the attribute
            result["subseries"] = self.subseries
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


# The pickles store this path (stats.pkl, old_stats.pkl, older checkouts); commands.psd_analyzer re-exports both classes
CardInfo.__module__ = CardStats.__module__ = "commands.psd_analyzer"


class StatsDatabase:
    """Manages card statistics database."""
    def __init__(self):
        self.stats: dict[str, CardInfo] = {}
        self.old_stats: defaultdict[str, list[CardInfo]] = defaultdict(list)
        self.all_types: list[str] = []
        self.seen_cards: list[str] = []            # names in the last tree; the rest are pruned

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
        """Move cards not in seen_cards to old_stats."""
        cards_to_remove = []
        for name, card in self.stats.items():
            if name not in self.seen_cards and not getattr(card, "duplicate_of", None):
                self.old_stats[name].append(card)
                cards_to_remove.append(name)

        for name in cards_to_remove:
            self.stats.pop(name)


# ========================
# Helper Functions
# ========================
def card_name(path: str) -> str:
    """A card's name from its repo path: the file stem with underscores as spaces ("Items/A_Snack.psd" -> "A Snack")."""
    return splitext(basename(path))[0].replace("_", " ")


def get_stats_as_dict() -> dict[str, dict]:
    """
    Get the stats database as a dictionary format.
    Useful for pandas DataFrame operations and external integrations.

    Returns:
        Dictionary mapping card names to their dict representations
    """
    return {name: card.to_dict() for name, card in stats_db.stats.items()}


# ========================
# Initialization
# ========================
stats_db = StatsDatabase()


# ========================
# Module Exports
# ========================
__all__ = [
    'CardInfo',
    'CardStats',
    'CardType',
    'StatsDatabase',
    'card_name',
    'get_stats_as_dict',
    'stats_db',
]
