"""
The card database's data: CardType, CardStats and CardInfo (the two pickled classes), StatsDatabase and its singleton
stats_db. Pure data, so anything may import it (no Discord, network, pandas or psd-tools).
"""
from collections import defaultdict
from copy import copy, deepcopy
from dataclasses import dataclass, field
from enum import Enum
from os import fsync, remove, replace
from os.path import basename, splitext
from pickle import load, dump
from threading import Lock
from time import time
from typing import Optional

from global_config import STATS_PKL, OLD_STATS_PKL

_stats_lock = Lock()                                     # one writer per pickle: /update_stats, /mass_replace...
_old_stats_lock = Lock()


def save_pickle(obj, path: str, lock: Lock) -> None:
    """Pickle to <path>.tmp beside the target, fsync, then replace: a kill mid-write leaves the old file whole."""
    with lock:
        temp_path = f"{path}.tmp"
        try:
            with open(temp_path, 'wb') as f:
                dump(obj, f)
                f.flush()
                fsync(f.fileno())
            replace(temp_path, path)
        except BaseException:
            try:
                remove(temp_path)
            except OSError:
                pass
            raise


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
    display_name: Optional[str] = None        # the printed title when it differs from name; a Card Edits.json name wins
    # set from Card Edits.json
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
    parser_version: Optional[int] = None      # psd_parser.PARSER_VERSION that parsed it; None in older pickles
    archived: Optional[dict] = None           # set when its PSD left the card repo (old_stats only)
    version = None                            # card_history's {from, to, cause...}; no field: parsed cards pickle as before

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
        self.seen_cards: list[str] = []            # names in the last tree; the rest are archived
        self.kept_archives: set = set()            # tagged records add_history kept as history: untagged after the save

    def save(self) -> None:
        """Save all databases to pickle files (atomically, one lock per pickle). old_stats first, and a restored card's
        archived record is dropped or untagged only after stats.pkl holds the card: a kill between saves never leaves
        it unlisted and untagged."""
        save_pickle(self.old_stats, OLD_STATS_PKL, _old_stats_lock)
        save_pickle(self.stats, STATS_PKL, _stats_lock)
        if self.drop_stale_archives():
            save_pickle(self.old_stats, OLD_STATS_PKL, _old_stats_lock)

    def load(self) -> None:
        """Load all databases from pickle files."""
        self.kept_archives = set()
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

    def archived(self) -> dict:
        """Cards whose PSD left the card repo: {name: its last version}, for names not in stats whose last old_stats
        version is tagged `archived`. The one read filter for archived cards."""
        found = {}
        for name, versions in self.old_stats.items():
            if name not in self.stats and self.archive_tag(name):
                found[name] = versions[len(versions) - 1]
        return found

    def archive_tag(self, name: str) -> Optional[dict]:
        """The `archived` tag of name's last old_stats version, or None."""
        versions = self.old_stats.get(name)
        return getattr(versions[len(versions) - 1], "archived", None) if versions else None

    def restore(self, name: str, path: str, blob: Optional[str]) -> bool:
        """Put a copy of an archived card back in stats when its PSD is back: the tag's path or blob (another PSD
        under the name is a new card). The tagged record stays until save() has written stats.pkl. Returns whether
        it was restored."""
        tag = None if name in self.stats else self.archive_tag(name)
        if not tag or (tag.get("path") != path and not (blob and tag.get("blob") == blob)):
            return False
        versions = self.old_stats[name]
        card = deepcopy(versions[len(versions) - 1])
        card.archived = None
        self.stats[name] = card
        return True

    def stale_archives(self) -> list:
        """Listed names whose last old_stats version is tagged (restore(), a kill or a /pull mix)."""
        return [name for name in self.old_stats if name in self.stats and self.archive_tag(name)]

    def drop_stale_archives(self) -> bool:
        """Each stale_archives() record: kept as history untagged when add_history kept it or it is another version,
        else dropped (a copy of the listed card). Returns whether any."""
        names = self.stale_archives()
        for name in names:
            versions = self.old_stats[name]
            last = versions[len(versions) - 1]
            if name in self.kept_archives or not _tagged_copy(last, self.stats[name]):
                last.archived = None
            else:
                versions.pop()
                if not versions:
                    del self.old_stats[name]
        self.kept_archives = set()
        return bool(names)

    def add_history(self, name: str, card: CardInfo) -> None:
        """Append card to name's history. A last version that is its archived copy stays (untagged after the save);
        one equal but for the path (a kill between the saves; paths follow the card) is replaced."""
        versions = self.old_stats.setdefault(name, [])
        if versions:
            last = versions[len(versions) - 1]
            if _tagged_copy(last, card):
                self.kept_archives.add(name)
                return
            if _equal_but(last, card, "path"):
                versions[len(versions) - 1] = card
                return
        versions.append(card)

    def prune_clean_cards(self, at: float) -> tuple:
        """
        Archive the cards not in seen_cards (their PSD left the card repo; duplicates are derived, so just dropped by
        _sync_duplicates): each moves to old_stats behind its earlier versions, tagged `archived`. The same pass tags
        the untagged last versions of names gone from stats before archiving existed (backfill).

        Args:
            at: The run's time (epoch seconds), kept in each tag

        Returns:
            (archived names, backfilled names), each sorted
        """
        seen = set(self.seen_cards)
        archived = [name for name, card in self.stats.items()
                    if name not in seen and not getattr(card, "duplicate_of", None)]
        for name in archived:
            card = self.stats.pop(name)
            card.archived = _archive_tag(card, at)
            versions = self.old_stats.setdefault(name, [])
            if versions and _tagged_copy(versions[len(versions) - 1], card):    # left by a kill between the two saves
                versions[len(versions) - 1] = card
            else:
                versions.append(card)

        backfilled = []
        for name, versions in self.old_stats.items():
            if not versions or name in self.stats or name in seen:
                continue
            last = versions[len(versions) - 1]
            if not self.archive_tag(name) and not getattr(last, "duplicate_of", None):
                last.archived = _archive_tag(last, at)
                last.archived["backfill"] = True
                backfilled.append(name)
        return (sorted(archived), sorted(backfilled))


def _archive_tag(card: CardInfo, at: float) -> dict:
    """A CardInfo.archived record: plain builtins (the website reads the pickles as data)."""
    return {"at": at, "path": card.path, "blob": getattr(card, "blob_sha", None) or None, "commit": None}


def _tagged_copy(old: CardInfo, card: CardInfo) -> bool:
    """Whether old is this card archived already: tagged, same PSD path and blob (no blob: equal but the tag)."""
    if not getattr(old, "archived", None) or any(getattr(old, key, None) != getattr(card, key, None)
                                                 for key in ("path", "blob_sha")):
        return False
    return bool(getattr(old, "blob_sha", None)) or _equal_but(old, card, "archived")


def _equal_but(old: CardInfo, card: CardInfo, key: str) -> bool:
    """Whether old equals card but for one field."""
    a, b = copy(old), copy(card)
    setattr(a, key, None)
    setattr(b, key, None)
    return a == b


def rescue_dropped(merged_stats, merged_old, sides) -> list:           # untyped: old_stats is a defaultdict
    """
    Cards a file-by-file pickle merge dropped (stats.pkl from one side, old_stats.pkl from the other): an unlisted
    name whose merged history is shorter than a side's (its history plus its listed card) gets that side's back,
    the longest, a tagged last first. A listed card comes back tagged (the next /update_stats restores it if its
    PSD is there). Duplicates are derived, never rescued.

    Args:
        merged_stats: The merged stats.pkl
        merged_old: The merged old_stats.pkl (changed in place)
        sides: (stats, old_stats) of each merged commit

    Returns:
        The rescued names, sorted
    """
    best: dict = {}
    for stats, old in sides:
        for name in set(stats) | set(old):
            if name in merged_stats:
                continue
            versions = list(old.get(name) or [])
            card = stats.get(name)
            listed = card is not None and not getattr(card, "duplicate_of", None)
            if listed:
                versions.append(card)
            if not versions:
                continue
            tagged = bool(getattr(versions[len(versions) - 1], "archived", None))
            if name not in best or (len(versions), tagged) > best[name][0]:
                best[name] = ((len(versions), tagged), versions, listed)
    rescued = []
    at = time()
    for name, (_, versions, listed) in best.items():
        if len(versions) > len(merged_old.get(name) or []):
            if listed:
                card = copy(versions[len(versions) - 1])
                card.archived = _archive_tag(card, at)
                versions[len(versions) - 1] = card
            merged_old[name] = versions
            rescued.append(name)
    return sorted(rescued)


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
    'rescue_dropped',
    'save_pickle',
    'stats_db',
]
