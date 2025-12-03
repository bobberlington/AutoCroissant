from dataclasses import dataclass
from difflib import get_close_matches, SequenceMatcher
from discord import Interaction
from os.path import basename, splitext
from pickle import load, dump
from requests import get
from typing import Optional
from warnings import filterwarnings
from pandas import DataFrame, concat, merge
from re import compile, escape, IGNORECASE
from urllib.parse import quote

import config
from global_config import ALIAS_PKL, STATS_PKL
from commands.utils import queue_message, split_long_message

filterwarnings('ignore')

# ========================
# Configuration
# ========================
DEFAULT_REPOSITORY = "MichaelJSr/TTSCardMaker"
DEFAULT_MATCH_RATIO = 0.6
GIT_TOKEN: str = getattr(config, "GIT_TOKEN", "")


@dataclass
class CardRepository:
    """Manages card repository state and operations."""

    repository: str = DEFAULT_REPOSITORY
    match_ratio: float = DEFAULT_MATCH_RATIO

    def __post_init__(self):
        self.git_file_alias: dict[str, str] = {}
        self.git_files: dict[str, str] = {}
        self.ambiguous_names: dict[str, tuple[str, ...]] = {}
        self.cards_dff: DataFrame = DataFrame()
        self.rulebook_dff: DataFrame = DataFrame()
        self._headers = {'Authorization': f'token {GIT_TOKEN}'} if GIT_TOKEN else {}

        if GIT_TOKEN:
            print("Git token found, API limited to 5000 requests/hour.")
        else:
            print("No git token in config, API limited to 60 requests/hour.")

        self._stat_pattern = compile(r"^(\w+)(<|<=|==|>=|>)(\d+)$", IGNORECASE)
        self._exact_match_pattern = compile(r'(?:^|\s|$|\b){pattern}(?:^|\s|$|\b)')

    @property
    def git_filenames(self) -> list[str]:
        """Get list of all git filenames."""
        return list(self.git_files.keys())

    def populate_files(self) -> int:
        """
        Fetch and populate file listings from GitHub repository.

        Returns:
            HTTP status code (200 for success)
        """
        self.git_files.clear()
        self.ambiguous_names.clear()

        url = f"https://api.github.com/repos/{self.repository}/git/trees/main?recursive=1"
        response = get(url, headers=self._headers, timeout=30)

        # Process PNG files from repository
        for item in response.json().get("tree", []):
            path: str = item.get("path", "")
            if not path.endswith(".png"):
                continue

            self._add_file_to_registry(path)

        # Add aliases to registry
        self._populate_aliases()

        return response.status_code

    def _add_file_to_registry(self, path: str) -> None:
        """Add a file path to the registry, handling duplicates."""
        png_filename = path[path.rindex("/") + 1:].lower()

        if png_filename in self.git_files:
            # Handle duplicate filenames by prefixing with top-level folder
            new_filename = self._get_prefixed_name(path, png_filename)
            old_path = self.git_files[png_filename]
            old_filename = self._get_prefixed_name(old_path, png_filename)

            self.git_files[old_filename] = old_path
            # In case both filenames are the same, set the new second to make sure something maps to it...
            self.git_files[new_filename] = path

            # Track ambiguous names
            if png_filename not in self.ambiguous_names:
                self.ambiguous_names[png_filename] = (old_filename,)
            if old_filename not in self.ambiguous_names[png_filename]:
                self.ambiguous_names[png_filename] = (*self.ambiguous_names[png_filename], old_filename)
            self.ambiguous_names[png_filename] = (*self.ambiguous_names[png_filename], new_filename)
        else:
            self.git_files[png_filename] = path

    @staticmethod
    def _get_prefixed_name(path: str, filename: str) -> str:
        """Get filename prefixed with top-level folder."""
        folder = path[:path.index('/')]
        return f"{folder}/{filename}".lower()

    def _populate_aliases(self) -> None:
        """Populate aliases into the file registry."""
        for alias, target in self.git_file_alias.items():
            alias_key = f"{alias}.png"

            if target in self.git_files:
                self.git_files[alias_key] = self.git_files[target]
            else:
                # Find files ending with target
                matches = [v for k, v in self.git_files.items() if k.endswith(target)]
                if matches:
                    self.git_files[alias_key] = matches[0]

    def _parse_stat_query(self, ability: str) -> Optional[DataFrame]:
        """
        Parse and execute column-based queries (e.g., 'hp>10', 'type==creature', 'stars<=5').

        Supports:
        - Numeric comparisons: hp>5, def<=10, atk>=3, spd==7, stars<6
        - String exact matches: type==creature, subtype==attack, series==kirby
        - String contains: type~creature (contains 'creature')

        Args:
            ability: Query string

        Returns:
            DataFrame of matching cards or None
        """
        # Try numeric comparison pattern (e.g., hp>5, stars<=3)
        numeric_match = self._stat_pattern.match(ability.lower())
        if numeric_match:
            column, operator, value = numeric_match.groups()
            value = int(value)

            # Check if column exists in dataframe
            if column not in self.cards_dff.columns:
                return None

            # Apply operator
            operators = {
                '<': lambda df, col, val: df[df[col] < val],
                '<=': lambda df, col, val: df[df[col] <= val],
                '==': lambda df, col, val: df[df[col] == val],
                '>=': lambda df, col, val: df[df[col] >= val],
                '>': lambda df, col, val: df[df[col] > val],
            }

            return operators.get(operator, lambda df, col, val: DataFrame())(self.cards_dff, column, value)

        # Try string exact match pattern (e.g., type==creature, subtype==attack)
        string_exact_pattern = compile(r"^(\w+)==(.+)$", IGNORECASE)
        string_match = string_exact_pattern.match(ability)
        if string_match:
            column, value = string_match.groups()
            column = column.lower()
            value = value.strip().lower()

            # Check if column exists
            if column not in self.cards_dff.columns:
                return None

            # Filter by exact match (case-insensitive)
            return self.cards_dff[self.cards_dff[column].astype(str).str.lower() == value]

        # Try string contains pattern (e.g., type~creature)
        string_contains_pattern = compile(r"^(\w+)~(.+)$", IGNORECASE)
        contains_match = string_contains_pattern.match(ability)
        if contains_match:
            column, value = contains_match.groups()
            column = column.lower()
            value = value.strip().lower()

            # Check if column exists
            if column not in self.cards_dff.columns:
                return None

            # Filter by contains (case-insensitive)
            return self.cards_dff[self.cards_dff[column].astype(str).str.lower().str.contains(value, na=False, regex=False)]

        return None

    def ability_search_engine(self, ability: str) -> DataFrame:
        """
        Search for cards by ability text or column values.

        Supports:
        - OR queries: "ability1 | ability2"
        - AND queries: "ability1 & ability2"
        - NOT queries: "!ability"
        - Exact matches: '"exact text"'
        - Numeric comparisons: "hp>5", "stars<=3", "def==10"
        - String exact match: "type==creature", "subtype==attack"
        - String contains: "type~aux", "series~kirby"

        Args:
            ability: Search query string

        Returns:
            DataFrame of matching cards
        """
        ability = ability.strip()

        # Handle OR queries
        if "|" in ability:
            queries = [q.strip() for q in ability.split("|")]
            results = [self.ability_search_engine(q) for q in queries]
            combined = concat(results)
            # Drop duplicates by index (card name) and ability column
            return combined[~combined.index.duplicated(keep='first')]

        # Handle AND queries
        if "&" in ability:
            queries = [q.strip() for q in ability.split("&")]
            results = [self.ability_search_engine(q) for q in queries]
            result = results[0]
            for r in results[1:]:
                result = merge(result, r, left_index=True, right_index=True, how="inner", copy=False)
            return result

        # Handle stat queries
        stat_result = self._parse_stat_query(ability)
        if stat_result is not None:
            return stat_result

        # Handle NOT queries
        opposite = ability.startswith("!")
        if opposite:
            ability = ability[1:].strip()

        # Handle exact matches
        if ability.startswith('"') and ability.endswith('"'):
            pattern = r"(?:^|\s|$|\b)" + escape(ability.strip('"')) + r"(?:^|\s|$|\b)"
            if opposite:
                return self.cards_dff[~self.cards_dff["ability"].str.contains(pattern, na=False)]
            return self.cards_dff[self.cards_dff["ability"].str.contains(pattern, na=False)]

        # Handle substring matches with fuzzy matching
        if opposite:
            return self.cards_dff[~self.cards_dff["ability"].str.contains(escape(ability), na=False)]

        # Exact substring matches
        exact_matches = self.cards_dff[self.cards_dff["ability"].str.contains(escape(ability), na=False)]

        # Fuzzy matches
        scores = self.cards_dff["ability"].apply(
            lambda x: SequenceMatcher(None, ability, x.lower()).ratio()
        )
        fuzzy_matches = self.cards_dff.loc[scores[scores > self.match_ratio].index]

        # Combine and remove duplicates by index
        combined = concat([exact_matches, fuzzy_matches])
        return combined[~combined.index.duplicated(keep='first')]

    def rulebook_search_engine(self, search_text: str) -> DataFrame:
        """
        Search rulebook entries for matching text.

        Args:
            search_text: Text to search for (case-insensitive)

        Returns:
            DataFrame of matching rulebook entries
        """
        search_text = search_text.strip().lower()

        if not search_text:
            return self.rulebook_dff

        # Search in ability text
        matches = self.rulebook_dff[
            self.rulebook_dff["ability"].str.contains(
                escape(search_text),
                na=False,
                case=False
            )
        ]

        return matches

    def load_aliases(self) -> None:
        """Load card aliases from pickle file."""
        print(f"Loading aliases from {ALIAS_PKL}")
        try:
            with open(ALIAS_PKL, 'rb') as f:
                self.git_file_alias = load(f)
        except (EOFError, FileNotFoundError):
            print(f"{ALIAS_PKL} doesn't exist, creating empty file...")
            self.save_aliases()

    def save_aliases(self) -> None:
        """Save card aliases to pickle file."""
        with open(ALIAS_PKL, 'wb') as f:
            dump(self.git_file_alias, f)

    def prep_dataframes(self) -> None:
        """Load card statistics from pickle file."""
        from commands.psd_analyzer import get_stats_as_dict

        cards_df = DataFrame.from_dict(get_stats_as_dict()).transpose()

        if not cards_df.empty:
            # Separate cards and rulebook entries
            has_ability = cards_df["ability"].notna()
            is_rulebook = cards_df["path"].str.contains("Rulebook", na=False)

            # Cards DataFrame (excludes rulebook)
            self.cards_dff = cards_df[has_ability & ~is_rulebook].copy()
            self.cards_dff["ability"] = self.cards_dff["ability"].str.lower()

            # Rulebook DataFrame (only rulebook entries)
            self.rulebook_dff = cards_df[has_ability & is_rulebook].copy()
            self.rulebook_dff["ability"] = self.rulebook_dff["ability"].str.lower()

            print(f"Loaded {len(self.cards_dff)} cards and {len(self.rulebook_dff)} rulebook entries")

    def _normalize_card_name(self, query: str) -> str:
        """Normalize card name for searching."""
        card = query.replace(" ", "_").lower()
        return card if card.endswith(".png") else f"{card}.png"

    def _find_closest_match(self, query: str) -> Optional[str]:
        """Find closest matching card filename."""
        matches = get_close_matches(self._normalize_card_name(query), self.git_filenames, n=1, cutoff=self.match_ratio)
        return matches[0] if matches else None

    def get_card_url(self, filename: str) -> str | None:
        """
        Get full GitHub URL for a card.

        Args:
            filename: Card filename (with or without .png extension)

        Returns:
            Full URL to the card image

        Raises:
            KeyError: If card is not found
        """
        # Normalize filename - replace spaces with underscores
        if not filename.endswith('.png'):
            filename = f"{filename}.png"
        filename = filename.replace(' ', '_').lower()

        if filename not in self.git_files:
            filename = filename.replace('_', ' ')
            if filename not in self.git_files:
                print(f"Card not found: {filename}")
                return

        # Keep slashes unencoded
        return f"https://raw.githubusercontent.com/{self.repository}/main/{quote(self.git_files[filename], safe='/')}"

    def get_card_path(self, query: str) -> Optional[str]:
        """
        Get the path for a card (without extension) given its name.

        Args:
            query: Card name

        Returns:
            Path to card (with spaces, no extension, not URL-encoded) or None if not found
        """
        closest = self._find_closest_match(query)
        if not closest:
            return None
        return splitext(self.git_files[closest])[0]


# ========================
# Discord Command Functions
# ========================
def query_name(interaction: Interaction, query: str) -> None:
    """
    Query and display a card by name.

    Args:
        interaction: Discord interaction
        query: Card name to search for
    """
    closest = card_repo._find_closest_match(query)

    if not closest:
        queue_message(interaction, "No card found!")
        return

    queue_message(interaction, card_repo.get_card_url(closest))

    # Notify if name is ambiguous
    if closest in card_repo.ambiguous_names:
        ambiguous_options = "\n".join(card_repo.ambiguous_names[closest])
        queue_message(interaction, f"Ambiguous name found. If this wasn't the card you wanted, try:\n{ambiguous_options}")


def query_ability(interaction: Interaction,
                  ability: str,
                  limit: int = -1) -> None:
    """
    Query cards by ability text or column values (excludes rulebook entries).

    Supports multiple query types:
    - Text search: "damage", "draw"
    - Numeric comparisons: "hp>5", "stars<=3", "def==10"
    - String exact match: "type==creature", "subtype==attack"
    - String contains: "type~aux", "series~kirby"
    - Combinations: "type==creature & hp>7"

    Args:
        interaction: Discord interaction
        ability: Search query string
        limit: Maximum number of results to display (-1 for all)
        filter_raids: Whether to filter out raid cards

    Note:
        This function does not search rulebook entries.
    """
    if card_repo.cards_dff.empty:
        queue_message(interaction, f"{STATS_PKL} is empty. Run `/update_stats` first.")
        return

    results = card_repo.ability_search_engine(ability.strip().lower())

    total_results = len(results)
    results_to_show = total_results if limit < 0 else min(limit, total_results)

    queue_message(interaction, f"Found {total_results} results for '{ability}'!")

    for idx, (index, _) in enumerate(results.iterrows()):
        if idx >= results_to_show:
            break
        # Index is the full path without .png extension
        queue_message(interaction, card_repo.get_card_url(basename(index)))

    queue_message(interaction, f"Displayed {results_to_show} of {total_results} results for '{ability}'.")


def query_ability_num_occur(interaction: Interaction, ability: str) -> None:
    """
    Query the number of cards matching an ability.

    Args:
        interaction: Discord interaction
        ability: Ability search query
    """
    if card_repo.cards_dff.empty:
        queue_message(interaction, f"{STATS_PKL} is empty. Run `/update_stats` first.")
        return

    results = card_repo.ability_search_engine(ability.strip().lower())
    queue_message(interaction, f"Found {len(results)} cards matching '{ability}'.")


def query_rulebook(interaction: Interaction,
                   search_text: str,
                   limit: int = -1) -> None:
    """
    Search rulebook entries for matching text.

    Args:
        interaction: Discord interaction
        search_text: Text to search for in rulebook
        limit: Maximum number of results to display (-1 for all)
    """
    if card_repo.rulebook_dff.empty:
        queue_message(interaction, f"{STATS_PKL} is empty or has no rulebook entries. Run `/update_stats` first.")
        return

    results = card_repo.rulebook_search_engine(search_text.strip())

    # Sort alphabetically by index (card name)
    results = results.sort_index(key=lambda x: x.str.lower())

    total_results = len(results)
    results_to_show = total_results if limit < 0 else min(limit, total_results)

    queue_message(interaction, f"Found {total_results} rulebook entries matching '{search_text}'!")

    if total_results == 0:
        return

    # Send results
    for idx, (name, row) in enumerate(results.iterrows()):
        if idx >= results_to_show:
            break

        ability = row.get("ability", "No text available")
        for chunk in split_long_message(ability):
            queue_message(interaction, f"**{name}**\n```\n{chunk}\n```")

    if results_to_show < total_results:
        queue_message(interaction, f"Displayed {results_to_show} of {total_results} results. Use limit parameter to see more.")


def print_all_aliases() -> str:
    """Get formatted string of all aliases."""
    if not card_repo.git_file_alias:
        return "```No aliases defined.```"

    lines = ["```"]
    for key, val in sorted(card_repo.git_file_alias.items()):
        lines.append(f"{key:20s} -> {val}")
    lines.append("```")
    return "\n".join(lines)


async def manage_alias(interaction: Interaction, key: Optional[str], val: Optional[str]) -> None:
    """
    Manage aliases for card names.

    Args:
        interaction: Discord interaction
        key: Alias name (for create/delete operations)
        val: Target card name (for create), or None to delete
    """
    # View all aliases if no parameters provided
    if not key:
        if not card_repo.git_file_alias:
            await interaction.response.send_message("```No aliases defined.```")
            return

        lines = ["```"]
        for alias_key, alias_val in sorted(card_repo.git_file_alias.items()):
            lines.append(f"{alias_key:20s} -> {alias_val}")
        lines.append("```")
        await interaction.response.send_message("\n".join(lines))
        return

    key = key.lower()

    # Delete alias if val is None or empty
    if not val:
        if key not in card_repo.git_file_alias:
            await interaction.response.send_message(f"No alias exists for '{key}'.")
            return

        target = card_repo.git_file_alias.pop(key)
        card_repo.git_files.pop(f"{key}.png", None)
        card_repo.save_aliases()
        await interaction.response.send_message(f"Deleted alias: {key} -> {target}")
        return

    # Create alias
    val = val.lower() if val.endswith(".png") else f"{val}.png"

    # Find target in git_files
    target_path = None
    if val in card_repo.git_files:
        target_path = card_repo.git_files[val]
    else:
        for filename, path in card_repo.git_files.items():
            if filename.endswith(val):
                target_path = path
                break

    if not target_path:
        await interaction.response.send_message(f"No card found matching '{val}'. Alias not created.")
        return

    # Create alias
    card_repo.git_files[f"{key}.png"] = target_path
    card_repo.git_file_alias[key] = val
    card_repo.save_aliases()
    await interaction.response.send_message(f"Created alias: {key} -> {val}")


async def set_match_ratio(interaction: Interaction, value: Optional[float] = None) -> None:
    """
    Set or display the fuzzy match ratio.

    Args:
        interaction: Discord interaction
        value: New match ratio (0.0-1.0)
    """
    if value is None:
        await interaction.response.send_message(f"Current match ratio: {card_repo.match_ratio}")
        return

    if not 0.0 <= value <= 1.0:
        await interaction.response.send_message("Match ratio must be between 0.0 and 1.0.")
        return

    card_repo.match_ratio = value
    await interaction.response.send_message(f"Match ratio set to {card_repo.match_ratio}")


async def set_repository(interaction: Interaction, new_repo: Optional[str] = None) -> None:
    """
    Set or display the GitHub repository.

    Args:
        interaction: Discord interaction
        new_repo: New repository name
    """
    if not new_repo:
        await interaction.response.send_message(f"Current repository: {card_repo.repository}")
        return

    card_repo.repository = new_repo
    status = card_repo.populate_files()

    if status != 200:
        await interaction.response.send_message(f"Error {status} when connecting to {new_repo}")
    else:
        await interaction.response.send_message(f"Repository set to {card_repo.repository}")


# ========================
# Initialization
# ========================
card_repo = CardRepository()
def init_query() -> None:
    """Initialize the card repository with saved data."""
    card_repo.load_aliases()
    status = card_repo.populate_files()
    if status != 200:
        print(f"Warning: Failed to populate files from repository (status {status})")


# ========================
# Module Exports
# ========================
__all__ = [
    'card_repo',
    'manage_alias',
    'init_query',
    'query_ability',
    'query_ability_num_occur',
    'query_name',
    'query_rulebook',
    'set_match_ratio',
    'set_repository',
]
