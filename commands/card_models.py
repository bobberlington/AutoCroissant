"""
The card database's data: CardType, CardStats and CardInfo, StatsDatabase and its singleton stats_db (loaded from the
card data's JSON by card_sync). Pure data, so anything may import it (no Discord, network, pandas or psd-tools).
"""
from collections import defaultdict
from copy import copy, deepcopy
from dataclasses import dataclass, field
from enum import Enum
from os.path import basename, splitext
from typing import Optional


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
    alt_names: Optional[list] = None          # unused since 2026-10-25 (hidden layers name nothing); older data has it
    blob_sha: Optional[str] = None            # git blob of the PSD it was parsed from
    subseries: Optional[str] = None           # from Card Edits.json, never the PSD
    parser_version: Optional[int] = None      # psd_parser.PARSER_VERSION that parsed it; None in older pickles
    archived: Optional[dict] = None           # set when its PSD left the card repo (old_stats only)
    version = None                            # card_history's {from, to, cause...}; no field: parsed cards pickle as before
    created = None                            # CardDates: {date, from[, provisional]}; no field either

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
        if getattr(self, "created", None):
            result["created"] = self.created["date"]

        return result


class StatsDatabase:
    """Manages card statistics database."""
    def __init__(self):
        self.stats: dict[str, CardInfo] = {}
        self.old_stats: defaultdict[str, list[CardInfo]] = defaultdict(list)
        self.all_types: list[str] = []
        self.seen_cards: list[str] = []            # names in the last tree; the rest are archived
        self.kept_archives: set = set()            # tagged records add_history kept as history: drop_stale_archives untags them

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
        under the name is a new card). The tagged record stays until drop_stale_archives. Returns whether it was
        restored."""
        tag = None if name in self.stats else self.archive_tag(name)
        if not tag or (tag.get("path") != path and not (blob and tag.get("blob") == blob)):
            return False
        versions = self.old_stats[name]
        card = deepcopy(versions[len(versions) - 1])
        card.archived = None
        self.stats[name] = card
        return True

    def stale_archives(self) -> list:
        """Listed names whose last old_stats version is tagged (restore())."""
        return [name for name in self.old_stats if name in self.stats and self.archive_tag(name)]

    def drop_stale_archives(self) -> bool:
        """Each stale_archives() record: kept as history untagged when add_history kept it or it is another version,
        else dropped (a copy of the listed card, or its data again, the listed card knowing all it did:
        card_history.knows; its earlier start goes to the listed card, card_history.carry). Returns whether any."""
        from commands.card_history import carry, knows       # lazy: card_history imports this module
        names = self.stale_archives()
        for name in names:
            versions = self.old_stats[name]
            last = versions[len(versions) - 1]
            listed = self.stats[name]
            if name in self.kept_archives or not (_tagged_copy(last, listed) or knows(listed, last)):
                last.archived = None
                vars(last).pop("created", None)            # plain history now: the listed card keeps its date
            else:
                carry(last, listed)
                versions.pop()
                if not versions:
                    del self.old_stats[name]
        self.kept_archives = set()
        return bool(names)

    def add_history(self, name: str, card: CardInfo) -> None:
        """Append card to name's history, without the card's creation date (the card keeps it). A last version that is
        its archived copy stays (untagged by drop_stale_archives); one equal but for the path (an old kill between the
        pickle saves) is replaced."""
        vars(card).pop("created", None)
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
        Archive the cards not in seen_cards (their PSD left the card repo; an alt-art duplicate whose PNG or original
        left, its tag at its own PNG): each moves to old_stats behind its earlier versions, tagged `archived`. The same
        pass tags the untagged last versions of names gone from stats before archiving existed (backfill).

        Args:
            at: The run's time (epoch seconds), kept in each tag

        Returns:
            (archived names, backfilled names), each sorted
        """
        seen = set(self.seen_cards)
        archived = [name for name in self.stats if name not in seen]
        for name in archived:
            card = self.stats.pop(name)
            card.archived = _archive_tag(card, at)
            if getattr(card, "duplicate_of", None):           # a copy of its original's PSD: its own file is its PNG
                card.archived["path"], card.archived["blob"] = duplicate_png(card), None
            versions = self.old_stats.setdefault(name, [])
            if versions and _tagged_copy(versions[len(versions) - 1], card):    # an old kill between the two pickle saves
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


def duplicate_png(card: CardInfo) -> str:
    """An alt-art duplicate's own PNG path: its name's file beside its original's PSD (A_Snack2.png)."""
    folder = card.path[:card.path.rfind("/") + 1] if "/" in (card.path or "") else ""
    return f"{folder}{card.name.replace(' ', '_')}.png"


def _archive_tag(card: CardInfo, at: float) -> dict:
    """A CardInfo.archived record: plain builtins (db/ JSON)."""
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
    'duplicate_png',
    'get_stats_as_dict',
    'stats_db',
]
