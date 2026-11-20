"""
The card database's commands, and its one import point.

The code lives in commands.card_models (CardType, CardStats, CardInfo, StatsDatabase, stats_db), commands.card_edits
(CardAuthors, CardEdits) and commands.psd_parser (PSDParser, CardClassifier, CardValidator, IconVocabulary). What the bot
takes from here is re-exported (CardInfo, CardEdits, card_name, get_stats_as_dict, stats_db, EDITS_PATH: main.py,
query_card and card_sync import from this module); the parser is not, so the bot never loads psd_tools: tests and the
diagnostics scripts import it from commands.psd_parser, the other card classes from commands.card_models.
"""
from discord import Interaction, File
from os.path import basename, splitext
from typing import Optional
import pandas as pd

from global_config import EDITS_PATH
from commands.card_edits import CardEdits
from commands.card_history import UNKNOWN_IF, unknown
from commands.card_models import CardInfo, card_name, get_stats_as_dict, stats_db
from commands.query_card import BAN_LIST, _day, ban_list_text, card_repo, init_query
from commands.rulebook import label as chapter_label
from commands.utils import queue_message, queue_file, queue_long_message

# ========================
# Configuration
# ========================
EXPORTED_STATS_NAME = "stats"
EXPORTED_RULES_NAME = "rules"

COLUMN_ORDER = [
    'aliases', 'type', 'ability', 'hp', 'def', 'atk', 'spd', 'stats',
    'types', 'path', 'timestamp', 'created', 'author', 'stars',
    'problem', 'series', 'subseries', 'subtype'
]
CSV_COLUMNS = ['title', *COLUMN_ORDER]                     # the CSV: the printed title, each creature's stats
CSV_COLUMNS.insert(CSV_COLUMNS.index('spd') + 1, 'creatures')
BEFORE_PARSER_2 = "read before parser 2, 2026-10-21"     # card_history.unknown: its types


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
        aliases = fields.get("aliases")
        if not aliases:                              # most entries are series or art edits
            continue
        target = f"{key.replace(' ', '_')}.png"
        path = card_repo.git_files.get(target) or next((v for k, v in card_repo.git_files.items() if k.endswith("/" + target)), None)
        if path:
            have = card_to_aliases.setdefault(splitext(basename(path))[0].replace('_', ' '), [])
            have += [a for a in aliases if a.lower().replace(' ', '_') not in {h.lower().replace(' ', '_') for h in have}]

    # If specific card requested, return just its aliases
    if name is not None:
        return sorted(card_to_aliases.get(name, []))

    # Otherwise return the full mapping
    return card_to_aliases


def _shown(card: CardInfo) -> dict:
    """to_dict() as the bot shows it (/query as_of, the exports): a Created made before a year says 'before <year>'
    (Pep Drink, Wham Bam Bone), as the embed and the site; to_dict itself keeps the date (the bake gates compare it)."""
    shown = card.to_dict()
    before = (getattr(card, "created", None) or {}).get("before")
    if before and "created" in shown:
        shown["created"] = f"before {before}"
    return shown


def _creatures(card: CardInfo) -> list[str]:
    """Each creature's hp/def/atk/spd (multi-creature cards)."""
    return [f"{c.hp}/{c.defense}/{c.attack}/{c.speed}" for c in card.creatures or ()]


def _format_metadata(card: CardInfo, name: str) -> str:
    """A stored version's fields one per line: aliases first, COLUMN_ORDER, then any other keys. A field its reading could
    not know (card_history.unknown) says unknown, but a stored ability text (the website's history lists it)."""
    metadata = _shown(card)
    aliases = get_card_aliases(name)
    if aliases:
        metadata['aliases'] = ', '.join(aliases)
    for key in unknown(card) & {"types", "stars", "ability"}:
        if key == "ability" and card.ability:
            continue
        why = next(prefix.lower() for prefix, field in UNKNOWN_IF if field == key)
        if key == "types" and getattr(card, "parser_version", None) is None:
            why = BEFORE_PARSER_2
        metadata[key] = f"unknown ({why})"
    if "stats" in unknown(card):                      # a misread's numbers: none of them
        for key in ("hp", "def", "atk", "spd"):
            metadata.pop(key, None)
        problems = [str(p) for p in getattr(card, "problems", None) or []]
        why = [prefix.lower() for prefix, field in UNKNOWN_IF if field == "stats" and any(p.startswith(prefix) for p in problems)]
        metadata["stats"] = f"unknown ({why[0] if why else 'misread'})"
    return '\n'.join(_field_lines(metadata))


def _field_lines(metadata: dict) -> list[str]:
    """'key: value' lines, COLUMN_ORDER first, then any other keys."""
    keys = [key for key in COLUMN_ORDER if key in metadata] + [key for key in metadata if key not in COLUMN_ORDER]
    return [f"{key}: {metadata[key]}" for key in keys]


def _export_fields(card: CardInfo, aliases: str) -> list[str]:
    """The text export's 'key: value' lines: plain lists, every creature on one line, the timestamp's day."""
    row = _shown(card)
    row.pop("display_name", None)                     # the card's heading
    if aliases:
        row["aliases"] = aliases
    row["timestamp"] = _day(card.timestamp)
    if card.creatures:
        row["creatures"] = '; '.join(_creatures(card))
    return _field_lines({key: ', '.join(map(str, value)) if isinstance(value, list) else value
                         for key, value in row.items()})


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


def export_stats_to_file(interaction: Interaction,
                         only_ability: bool = False,
                         as_csv: bool = True) -> None:
    """
    Export card statistics to a file.

    Args:
        interaction: Discord interaction
        only_ability: Export only ability text
        as_csv: Export as CSV (otherwise text file)
    """
    card_to_aliases = get_card_aliases()
    aliases = {name: ', '.join(sorted(card_to_aliases.get(name, []))) for name in stats_db.stats}

    if as_csv:
        filtered_stats = {name: _shown(card) | {'aliases': aliases[name]} for name, card in stats_db.stats.items()}
        for name, row in filtered_stats.items():          # plain text: a title, every creature, lists joined
            card = stats_db.stats[name]
            row['title'] = card.display_name or name
            row['creatures'] = '; '.join(_creatures(card))
            for key, value in row.items():
                if isinstance(value, list):
                    row[key] = ', '.join(map(str, value))
        df = pd.DataFrame.from_dict(filtered_stats).transpose()
        df_filtered = df[df["ability"].notna()].copy()

        # Reorder columns (only include columns that exist)
        existing_columns = [col for col in CSV_COLUMNS if col in df_filtered.columns]
        df_filtered = df_filtered[existing_columns]

        # Sort alphabetically by path
        df_filtered = df_filtered.sort_values(
            by='path',
            key=lambda x: x.str.lower()
        )

        filename = f"{EXPORTED_STATS_NAME}.csv"
        df_filtered.to_csv(filename, index_label='name')

        with open(filename, 'rb') as f:
            queue_file(interaction, File(f, filename))
        return

    filename = f"{EXPORTED_STATS_NAME}.txt"

    # By type, then each card's printed title
    sorted_cards = sorted(
        ((card.display_name or name, name, card) for name, card in stats_db.stats.items()),
        key=lambda x: (x[2].card_type, x[0].lower(), x[1].lower())
    )

    with open(filename, 'w', encoding='utf-8') as f:
        current_type = None
        for title, name, card in sorted_cards:
            card_type = card.card_type

            # Add section header when type changes
            if card_type != current_type:
                f.write(f"=== {card_type.upper()} ===\n\n")
                current_type = card_type

            f.write(f"{title}\n" if title == name else f"{title} (file: {name})\n")

            if only_ability:
                if aliases[name]:
                    f.write(f"Aliases: {aliases[name]}\n")
                if card.ability:
                    f.write(f"{card.ability}\n")
            else:
                f.write('\n'.join(_export_fields(card, aliases[name])) + '\n')
            f.write('\n')

    with open(filename, 'rb') as f:
        queue_file(interaction, File(f, filename))


def export_rulebook_to_file(interaction: Interaction) -> None:
    """
    Export the rulebook's chapters (db/rulebook.json) to a text file, in book order, then the website's ban list.

    Args:
        interaction: Discord interaction
    """
    filename = f"{EXPORTED_RULES_NAME}.txt"

    with open(filename, 'w', encoding='utf-8') as f:
        for chapter in stats_db.rulebook:
            f.write(f"{chapter_label(chapter)}\n{chapter['text']}\n\n")
        bans = ban_list_text(card_repo.banlist)
        if bans:
            f.write(f"{BAN_LIST}\n{bans}\n\n")

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
    'export_stats_to_file',
    'export_rulebook_to_file',
    'init_psd',
]
