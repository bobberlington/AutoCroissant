"""
/update_stats: commands.card_data's build in legacy mode (the GitHub API, the root pickles and psd_details.json) under
the bake lock, its messages sent to Discord, then the follow-ups (type emojis, image links, the card list).
"""
from discord import Interaction
from typing import Optional

from commands.card_data import (
    COMMITS_PAGE,
    DOWNLOAD_WORKERS,
    EXCLUDE_FOLDERS,
    LOOKUP_WORKERS,
    PROGRESS,
    UPDATE_RATE,
    CardBuild,
    update,
)
from commands.card_edits import CardAuthors
from commands.card_models import stats_db
from commands.card_runner import bake_lock
from commands.query_card import DEFAULT_REPOSITORY, card_repo
from commands.repo_history import refresh as refresh_history
from commands.type_emojis import sync_changed_emojis
from commands.utils import queue_command, queue_edit, say

# ========================
# Global State
# ========================
_tree_etags: dict = {}                                            # repository -> ETag of the last update's tree
_file_cache: dict = {}                                            # repo path -> (blob sha, text) of small files



class RepositoryTraverser(CardBuild):
    """CardBuild as the bot runs it (psd_analyzer, the diagnostics): the /alias names unless given."""
    def __init__(self, db, *args, aliases: Optional[dict] = None, **kwargs):
        super().__init__(db, *args, aliases=card_repo.git_file_alias if aliases is None else aliases, **kwargs)



# ========================
# Helper Functions
# ========================
def _forget_tree(repository: str) -> None:
    """Drop the tree ETag so the next /update_stats refetches and retries."""
    _tree_etags.pop(repository, None)


def _progress(interaction: Optional[Interaction]):
    """The parse count as an edit of the command's reply, or None (printed)."""
    if interaction is None:
        return None
    return lambda count: queue_edit(interaction, content=PROGRESS.format(count=count))


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
    if verbose and bake_lock.locked():
        say(interaction, "Waiting for the running card update, bake or font check to finish...")
    with bake_lock:                                 # never beside a card bake (card_runner)
        return _update_stats(interaction, output_problematic, force_update, verbose, cards)


def _update_stats(interaction: Optional[Interaction],
                  output_problematic: bool,
                  force_update: bool,
                  verbose: bool,
                  cards: str) -> list[str]:
    """card_data.update on the bot's database, the messages and the queued follow-ups; update_stats holds bake_lock."""
    if verbose:
        say(interaction, "Updating card statistics database... This may take a while.")

    stats_db.load()
    stats_db.seen_cards.clear()
    traverser = RepositoryTraverser(stats_db, report=lambda text: say(interaction, text), progress=_progress(interaction),
                                    aliases=card_repo.git_file_alias, etags=_tree_etags, file_cache=_file_cache)
    traverser.only = {CardAuthors.key(c) for c in (cards or "").split(",") if c.strip()}
    problem_cards = update(traverser, card_repo.repository, output_problematic, force_update)

    if verbose:
        say(interaction, "Card statistics update complete!")

    # Output problems
    if verbose and problem_cards:
        say(interaction, "".join(problem_cards))         # detailed problems
        for card in problem_cards:                        # card names only
            say(interaction, card.split('```')[0])

    tree = traverser.tree                           # None on 304; emojis follow the card repo, never a /set_repo one
    if tree is not None and not tree.get("truncated") and traverser.repository == DEFAULT_REPOSITORY:
        queue_command(sync_changed_emojis, interaction, traverser.repository, traverser.shas, _forget_tree)
    queue_command(refresh_history)                  # re-pin image links; a git fetch when nothing changed

    # Always refresh card_repo stats
    queue_command(card_repo.populate_files, traverser.tree, None, traverser.etag)
    queue_command(card_repo.prep_dataframes)

    return problem_cards


# ========================
# Module Exports
# ========================
__all__ = [
    'COMMITS_PAGE',
    'DOWNLOAD_WORKERS',
    'EXCLUDE_FOLDERS',
    'LOOKUP_WORKERS',
    'UPDATE_RATE',
    'RepositoryTraverser',
    'update_stats',
]
