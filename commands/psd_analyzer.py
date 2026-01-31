from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from discord import Interaction, File
from enum import Enum
from github import Github, Repository
from logging import getLogger, CRITICAL
from os import walk
from os.path import getmtime, basename, expanduser, splitext, join as path_join
from pickle import load, dump
from psd_tools import PSDImage
from re import compile as re_compile
from requests import get
from typing import Optional, Any
from urllib.parse import quote
from urllib.request import urlretrieve, HTTPError
import pandas as pd

import config
from global_config import LOCAL_DIR_LOC, STATS_PKL, OLD_STATS_PKL
from commands.query_card import card_repo
from commands.utils import queue_message, queue_file, queue_edit, queue_command, split_long_message

getLogger("psd_tools").setLevel(CRITICAL)

# ========================
# Configuration
# ========================
UPDATE_RATE = 25
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


@dataclass
class StatTracker:
    """Tracks whether a stat was found and its accumulated value."""
    found: bool = False
    value: Optional[int] = None


@dataclass
class StatTrackers:
    """Container for all card stat trackers."""
    hp: StatTracker = field(default_factory=StatTracker)
    defense: StatTracker = field(default_factory=StatTracker)
    attack: StatTracker = field(default_factory=StatTracker)
    speed: StatTracker = field(default_factory=StatTracker)


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
            if name not in self.dirty_files:
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
    def __init__(self, all_types: list[str], stats_db: StatsDatabase):
        self.all_types = all_types
        self.stats_db = stats_db
        self._gap_pattern = re_compile(r'\s{3,}')
        self._spacing_pattern = re_compile(r'\s+([:;,\.\?!])')

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
        longest_text = MutableValue("")
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

        for layer in psd.descendants():
            self._process_layer(
                layer, card, longest_text, num_stars,
                stat_trackers, type_bboxes, abilities,
                is_rulepage, get_stars_from_psd, card_mid_y)

        self._process_abilities(card, abilities, type_bboxes, longest_text.value, card_mid_y)
        self._process_stats(card, stat_trackers)

        if get_stars_from_psd:
            card.stars = num_stars.value

    def _process_layer(self, layer, card: CardInfo, longest_text: MutableValue,
                       num_stars: MutableValue, stat_trackers: StatTrackers,
                       type_bboxes: list, abilities: list, is_rulepage: bool,
                       get_stars_from_psd: bool, card_mid_y: int) -> None:
        """Process a single PSD layer."""
        # Text layers
        if layer.kind == "type":
            self._process_text_layer(layer, longest_text, abilities, is_rulepage, card_mid_y)

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

        # Star layers
        elif get_stars_from_psd and self._is_star_layer(layer):
            if layer.is_visible() and layer.has_pixels():
                num_stars.value += 1

        # Problematic type layer
        elif (layer.name.lower() in MISSPELT_CARD_TYPES and layer.is_visible() and layer.has_pixels()):
            card.problems.append(f"MISSPELT TYPE: {layer.name.lower()}")

    def _process_text_layer(self, layer, longest_text: MutableValue,
                            abilities: list, is_rulepage: bool, card_mid_y: int) -> None:
        """Process text layer."""
        layer_text = str(layer.engine_dict["Editor"]["Text"])
        layer_text = (layer_text
            .replace('\\r', '\n')
            .replace('\\n', '\n')
            .replace('\\t', ' ')
            .replace('\\x03', '\n')
            .replace('\\ufeff', '')
            .rstrip())

        if layer.name.lower() == "ability" or is_rulepage:
            bbox = BoundingBox.from_tuple(layer.bbox[:2])
            abilities.append((layer_text, bbox))
        elif layer.bbox[1] > card_mid_y and len(layer_text) > len(longest_text.value):
            longest_text.value = layer_text

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

        parent_names = []
        current = layer.parent
        depth = 0
        while current and depth < 2:
            parent_names.append(current.name.lower())
            current = getattr(current, 'parent', None)
            depth += 1

        # Map short names to StatTracker attributes
        stat_map = {
            "hp": stat_trackers.hp,
            "def": stat_trackers.defense,
            "atk": stat_trackers.attack,
            "spd": stat_trackers.speed,
        }

        for stat_key, tracker in stat_map.items():
            if any(stat_key in name for name in parent_names):
                tracker.found = True
                if layer.is_visible():
                    if tracker.value is None:
                        tracker.value = 0
                    tracker.value += int(layer.name)
                break

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

    def _process_abilities(self, card: CardInfo, abilities: list[tuple[str, BoundingBox]],
                           type_bboxes: list[tuple[str, BoundingBox]], longest_text: str, card_mid_y: int) -> None:
        """Process and combine ability texts."""
        abilities = self._sort_by_position(abilities)
        ability_text = '\n'.join(text for text, _ in abilities)

        if not ability_text:
            ability_text = longest_text
            if ability_text:
                card.problems.append("NO ABILITY LAYER")

        if ability_text:
            type_bboxes = self._prune_type_bboxes(self._sort_by_position(type_bboxes), card_mid_y)
            ability_text = self._inject_type_names(ability_text.rstrip().rstrip("'\""), type_bboxes, card)
            card.ability = self._spacing_pattern.sub(r'\1', ability_text).strip('\'" ').strip()

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
                value = tracker.value if tracker.value is not None else 10
                setattr(card.stats, stat_attr, value)

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

    def _inject_type_names(self, ability: str,
                           type_bboxes: list[tuple[str, BoundingBox]], card: CardInfo) -> str:
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

            offset = 0
            for match in matches:
                if type_index >= len(types):
                    break

                insert_at = match.start() + offset
                type_name = types[type_index]

                replacement = f" [{type_name}] "
                line = (
                    line[:insert_at] +
                    replacement +
                    line[match.end() + offset:]
                ).lstrip()

                offset += len(replacement) - (match.end() - match.start())
                type_index += 1

            result_lines.append(line)

        # Append remaining types to the end
        if type_index < len(types):
            remaining = ' '.join(f"[{t}]" for t in types[type_index:])
            last_line_index = len(result_lines) - 1
            last_line = result_lines[last_line_index].rstrip('\n')
            result_lines[last_line_index] = f"{last_line} {remaining}"

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
            card.name not in CardValidator.EXCESSIVE_STAT_EXCLUSIONS):
            problems.append("STATS TOO HIGH")

        return problems


class RepositoryTraverser:
    """Traverses repository to update card stats."""
    def __init__(self, db: StatsDatabase):
        self.db = db
        self.parser: Optional[PSDParser] = None
        self._github_client: Optional[Github] = None

    def _check_for_path_change(self, name: str, new_path: str) -> bool:
        """
        Check if a card's path has changed and update if needed.

        Args:
            name: Card name
            new_path: New path for the card

        Returns:
            True if path changed, False otherwise
        """
        if name in self.db.stats:
            old_path = self.db.stats[name].path
            if old_path and old_path != new_path:
                # Archive old version with the old path
                old_card = self.db.stats[name]
                self.db.old_stats[name].append(old_card)
                return True
        return False

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

    def _should_update_card(self,
                            name: str,
                            timestamp: float,
                            new_path: str) -> tuple[bool, bool]:
        """
        Determine if a card should be updated.

        Args:
            name: Card name
            timestamp: New timestamp
            new_path: New path

        Returns:
            Tuple of (should_update, is_new_card)
        """
        if name not in self.db.stats:
            return (True, True)

        # Check for path change
        path_changed = self._check_for_path_change(name, new_path)

        # Check timestamp
        timestamp_newer = self.db.stats[name].timestamp < timestamp

        # Update if path changed OR timestamp is newer
        should_update = path_changed or timestamp_newer

        return (should_update, False)

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
        resp = get(
            f"https://api.github.com/repos/{repository}/git/trees/main?recursive=1",
            headers=self.db._headers,
            timeout=30)

        self._github_client = Github(login_or_token=GIT_TOKEN)
        repo = self._github_client.get_repo(repository)

        self._populate_types_from_response(resp)
        self.parser = PSDParser(self.db.all_types, self.db)

        problems, num_new = self._process_files_from_response(resp, repo, repository, interaction, output_problematic, force_update)

        # Update old_stats paths after processing all files
        self._update_old_stats_paths()

        return (problems, num_new)

    def traverse_local(self,
                       repository: str,
                       local_path: str,
                       interaction: Optional[Interaction] = None,
                       output_problematic: bool = True,
                       use_local_timestamp: bool = True,
                       force_update: bool = False) -> tuple[list[str], int]:
        """
        Traverse local repository directory.

        Args:
            repository: Repository name
            local_path: Path to local repository
            interaction: Optional Discord interaction
            output_problematic: Whether to output problematic cards
            use_local_timestamp: Use local file timestamps vs remote

        Returns:
            List of formatted problem card messages
        """
        if not use_local_timestamp:
            print(f"Warning: Remote timestamps use API requests (limit: {5000 if GIT_TOKEN else 60}/hour)")
            self._github_client = Github(login_or_token=GIT_TOKEN)
            repo = self._github_client.get_repo(repository)
        else:
            repo = None

        self._populate_types_from_local(local_path)
        self.parser = PSDParser(self.db.all_types, self.db)

        problems, num_new = self._process_local_files(local_path, repo, interaction, output_problematic, use_local_timestamp, force_update)

        # Update old_stats paths after processing all files
        self._update_old_stats_paths()

        return (problems, num_new)

    def _populate_types_from_response(self, response) -> None:
        """Populate card types from API response."""
        self.db.all_types.clear()
        for item in response.json().get("tree", []):
            path = item.get("path", "")
            if path.startswith("Types") and "Stars" not in path and '.' in path:
                self.db.all_types.append(splitext(basename(path))[0].lower())

    def _populate_types_from_local(self, local_path: str) -> None:
        """Populate card types from local directory."""
        self.db.all_types.clear()
        types_dir = path_join(local_path, "Types")
        for folder, _, files in walk(types_dir):
            if folder.endswith("Types"):
                for file in files:
                    self.db.all_types.append(splitext(file)[0].lower())

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
                                     repo: Repository.Repository,
                                     repository: str,
                                     interaction: Optional[Interaction],
                                     output_problematic: bool,
                                     force_update: bool) -> tuple[list[str], int]:
        """Process files from API response."""
        problematic_cards = []
        num_updated = 0
        num_new = 0
        num_old = 0
        num_moved = 0

        for item in response.json().get("tree", []):
            path = item.get("path", "")

            # Check if any folder in the path is in EXCLUDE_FOLDERS
            if not path.endswith('.psd') or any(folder in EXCLUDE_FOLDERS for folder in path.split('/')):
                continue

            name = splitext(basename(path))[0].replace('_', ' ')
            self.db.dirty_files.append(name)

            # Get timestamp and author metadata
            timestamp, author = self._get_remote_timestamp(repo, path, name)

            # Check if update needed
            should_update, is_new = self._should_update_card(name, timestamp, path)

            if not force_update and not should_update:
                num_old += 1
            else:
                # Archive old version before updating (if not new)
                if should_update and not is_new and name in self.db.stats:
                    old_card = self.db.stats[name]
                    self.db.old_stats[name].append(old_card)

                if is_new:
                    num_new += 1
                # Check if it's a move vs a content update
                elif name in self.db.stats and self.db.stats[name].path != path:
                    num_moved += 1
                else:
                    num_new += 1

                psd_url = self._construct_raw_url(repository, path)
                try:
                    local_file = urlretrieve(psd_url)[0]
                except HTTPError:
                    print(f"{psd_url} not found.")
                    continue

                card = self.parser.parse(local_file, path)
                card.timestamp = timestamp
                # Set author if it was fetched earlier
                if author and not card.author:
                    card.author = author
                self.db.stats[name] = card

                # Validate
                if output_problematic:
                    problems = CardValidator.validate(card)
                    if problems:
                        problematic_cards.append((path, card, problems))

            num_updated += 1
            if num_updated % UPDATE_RATE == 0:
                self._send_progress(interaction, num_updated)

        self._send_summary(interaction, num_new, num_old, num_moved)
        return (self._format_problems(problematic_cards), num_new)

    def _process_local_files(self,
                             local_path: str,
                             repo: Optional[Repository.Repository],
                             interaction: Optional[Interaction],
                             output_problematic: bool,
                             use_local_timestamp: bool,
                             force_update: bool) -> tuple[list[str], int]:
        """Process files from local directory."""
        problematic_cards = []
        num_updated = 0
        num_new = 0
        num_old = 0
        num_moved = 0

        for folder, _, files in walk(local_path):
            folder = folder.replace('\\', '/')
            # Check if current folder or any parent folder is in EXCLUDE_FOLDERS
            if any(part in EXCLUDE_FOLDERS for part in folder.split('/')):
                continue

            for file in files:
                if not file.endswith('.psd'):
                    continue

                full_path = path_join(folder, file)
                relative_path = full_path.removeprefix("TTSCardMaker").strip('/')
                name = splitext(basename(relative_path))[0].replace('_', ' ')
                self.db.dirty_files.append(name)

                # Get timestamp
                author = None
                if use_local_timestamp:
                    timestamp = getmtime(full_path)
                else:
                    timestamp, author = self._get_remote_timestamp(repo, relative_path, name)

                # Check if update needed
                should_update, is_new = self._should_update_card(name, timestamp, relative_path)

                if not force_update and not should_update:
                    num_old += 1
                else:
                    # Archive old version before updating (if not new)
                    if should_update and not is_new and name in self.db.stats:
                        old_card = self.db.stats[name]
                        self.db.old_stats[name].append(old_card)

                    if is_new:
                        num_new += 1
                    # Check if it's a move vs a content update
                    elif name in self.db.stats and self.db.stats[name].path != relative_path:
                        num_moved += 1
                    else:
                        num_new += 1

                    # Parse local file
                    card = self.parser.parse(full_path, relative_path)
                    card.timestamp = timestamp
                    # Set author if it was fetched earlier
                    if author and not card.author:
                        card.author = author
                    self.db.stats[name] = card

                    # Validate
                    if output_problematic:
                        problems = CardValidator.validate(card)
                        if problems:
                            problematic_cards.append((relative_path, card, problems))

                num_updated += 1
                if num_updated % UPDATE_RATE == 0:
                    self._send_progress(interaction, num_updated)

        self._send_summary(interaction, num_new, num_old, num_moved)
        return (self._format_problems(problematic_cards), num_new)

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

        # Set metadata author from FIRST commit (original author)
        if name not in self.db.stats or not self.db.stats[name].author:
            return (commits[0].commit.committer.date.timestamp(), commits[commits.totalCount - 1].commit.committer.name)

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
        msg = f"{num_new} had newer timestamps or were new.\n{num_old} did not have newer timestamps.\n{num_moved} cards changed location."

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
                 use_local_repo: bool = True,
                 use_local_timestamp: bool = True,
                 force_update: bool = True,
                 verbose: bool = True) -> list[str]:
    """
    Update card statistics database.

    Args:
        interaction: Optional Discord interaction
        output_problematic: Whether to output problematic cards
        use_local_repo: Use local repository vs remote
        use_local_timestamp: Use local timestamps vs remote
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

    if use_local_repo:
        local_path = expanduser(LOCAL_DIR_LOC)
        problem_cards, num_new = traverser.traverse_local(
            card_repo.repository,
            local_path,
            interaction,
            output_problematic,
            use_local_timestamp,
            force_update)
    else:
        problem_cards, num_new = traverser.traverse_remote(
            card_repo.repository,
            interaction,
            output_problematic,
            force_update)

    if (num_new > 0):
        stats_db.prune_clean_cards()
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
    queue_command(card_repo.populate_files)
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
