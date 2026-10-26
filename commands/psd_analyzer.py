"""
The card database's commands, and its one import point.

The code lives in commands.card_models (CardType, CardStats, CardInfo, StatsDatabase, stats_db), commands.card_edits
(CardAuthors, CardEdits) and commands.psd_parser (PSDParser, CardClassifier, CardValidator, IconVocabulary). The first two
are re-exported here (main.py and query_card import from this module); the parser is not, so the bot never loads
psd_tools: tests and the diagnostics scripts import it from commands.psd_parser.
"""
from discord import Interaction, File
from os.path import basename, splitext
from re import compile as re_compile
from typing import Optional
import pandas as pd

from global_config import AUTHORS_PATH, EDITS_PATH, ICON_MANIFEST_PATH, RULEBOOK_DIR
from commands.card_edits import CardAuthors, CardEdits
from commands.card_models import CardInfo, CardStats, CardType, StatsDatabase, card_name, get_stats_as_dict, stats_db
from commands.github_api import GIT_TOKEN
from commands.query_card import archived_text, card_repo, history_match, init_query
from commands.utils import queue_message, queue_file, queue_long_message

# ========================
# Configuration
# ========================
EXPORTED_STATS_NAME = "stats"
EXPORTED_RULES_NAME = "rules"

COLUMN_ORDER = [
    'aliases', 'type', 'ability', 'hp', 'def', 'atk', 'spd',
    'types', 'path', 'timestamp', 'created', 'author', 'stars',
    'problem', 'series', 'subseries', 'subtype'
]


# ========================
# Helper Functions
# ========================
def get_card_aliases(name: Optional[str] = None) -> dict[str, list[str]] | list[str]:
    """
    Build a reverse lookup of cards to aliases.

    Args:
        name: Optional specific card name to get aliases for.
              If None, returns dict mapping all card names to their aliases.

    Returns:
        If card_name provided: list of aliases for that card
        If card_name is None: dict mapping card names to lists of aliases
    """
    card_to_aliases = {}

    # Card Edits.json's, on the card whose PNG the entry names (as query_card._add_lookup registers them)
    edits = card_repo._edits
    for key, fields in (edits.cards.items() if edits is not None else ()):
        target = f"{key.replace(' ', '_')}.png"
        path = card_repo.git_files.get(target) or next((v for k, v in card_repo.git_files.items() if k.endswith("/" + target)), None)
        if path:
            have = card_to_aliases.setdefault(splitext(basename(path))[0].replace('_', ' '), [])
            have += [a for a in fields.get("aliases") or [] if a.lower().replace(' ', '_') not in
                     {h.lower().replace(' ', '_') for h in have}]

    # If specific card requested, return just its aliases
    if name is not None:
        return sorted(card_to_aliases.get(name, []))

    # Otherwise return the full mapping
    return card_to_aliases


def _format_metadata(card: CardInfo, name: str) -> str:
    """A card's fields one per line: aliases first, COLUMN_ORDER, then any other keys."""
    metadata = card.to_dict()
    aliases = get_card_aliases(name)
    if aliases:
        metadata['aliases'] = ', '.join(aliases)
    lines = [f"{key}: {metadata[key]}" for key in COLUMN_ORDER if key in metadata]
    lines += [f"{key}: {value}" for key, value in metadata.items() if key not in COLUMN_ORDER]
    return '\n'.join(lines)


# ========================
# Public API Functions
# ========================
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

    queue_long_message(interaction, '\n'.join(orphans))


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
    found = history_match(query)                # archived names too, before an alias that claims them
    if not found:
        queue_message(interaction, f"No card found matching '{query}'.")
        return

    name = found[1]

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

    # Add version info
    version_text = "oldest" if index == 0 else f"version {len(old_versions) - index - 1}"
    if index == len(old_versions) - 1:
        version_text = "most recent old version"
    tag = getattr(card, "archived", None)
    archived = f", {archived_text(tag)}" if tag and name not in stats_db.stats else ""

    queue_message(
        interaction,
        f"Old metadata for **{name}** ({version_text} of {len(old_versions)}){archived}:\n"
        f"```{_format_metadata(card, name)}```"
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
        if RULEBOOK_DIR not in card.path
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
            if RULEBOOK_DIR in card.path:
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
            if RULEBOOK_DIR in card.path:
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
def init_psd() -> None:
    """Initialize the stats database from the synced card data (card_sync): the local copy and the saved card list,
    then a sync and GitHub's tree (init_query)."""
    from commands.card_sync import load_local, sync
    try:
        ready = load_local()
        card_repo.load_saved()                      # the card list from the last tree on disk: no network
        if ready:
            card_repo.prep_dataframes()             # queries work before the sync's network round trip
        loaded = stats_db.stats
        sync(None, output_problematic=False, quiet=True, follow_ups=False)
        if not ready or stats_db.stats is not loaded:   # the sync brought new data
            card_repo.prep_dataframes()
        print("Stats database initialized.")
    finally:
        init_query()                                # after the sync (titles, renames); runs even if it failed


# ========================
# Module Exports
# ========================
__all__ = [
    'stats_db',
    'get_stats_as_dict',
    'list_orphans',
    'view_old_metadata',
    'export_stats_to_file',
    'export_rulebook_to_file',
    'init_psd',
]
