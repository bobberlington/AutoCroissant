"""
ARCHIVED 2026-10-20: /update_stats local mode (walk the TTSCardMaker clone on disk instead of GitHub).

This is the local traversal that used to live in commands/psd_analyzer.py behind the slash options
use_local_repo and use_local_timestamp. Since 2026-10-20 /update_stats always reads GitHub (remote mode) and
those options are gone. Why it was retired:
  - B6: with use_local_timestamp=True (its default), local mode compared file mtimes with the GitHub commit
    dates stored in stats.pkl. A clone's mtimes are checkout times, so every card looked newer: all 813 cards
    would be re-parsed and archived into old_stats.pkl.
  - B1 history: until 2026-10-19 local mode also wrote absolute paths into stats.pkl (the removeprefix bug,
    autocroissant-failure-archaeology Entry 1). It survived for months because production only ran remote mode.
  - Remote mode is the only path production uses, so keeping a second traversal in the bot meant a second set
    of bugs to maintain.

This file lives only on the branch archive/local-update-stats; main doesn't carry it. It was written against
main's commands/psd_analyzer.py as of 2026-10-20, the change that removed local mode (the branch itself starts
from a7a4bb3, before that change). It is not in setup.py's INCLUDE_FILES (plain Python, never compiled).
It still runs, for sandbox and diagnostic sweeps only (autocroissant-validation-and-qa method B). It WRITES
stats.pkl / old_stats.pkl in the current directory when any card is new or newer, so run it in a sandbox copy
of main (/tmp/ac_sandbox), never in the live checkout:

    cd /tmp/ac_sandbox
    mkdir -p archive && git show archive/local-update-stats:archive/local_update_stats.py > archive/local_update_stats.py
    python3 -c "import sys; sys.path.insert(0,'.'); from archive.local_update_stats import update_stats_local; update_stats_local(force_update=True)"

The offline helpers it needs (_populate_types_from_local, _icons_from_local) stay in psd_analyzer.py because
the diagnostics scripts (parse_one.py, gap_trace.py) use them.
"""
from os import walk
from os.path import basename, expanduser, getmtime, relpath, splitext, join as path_join
from typing import Optional

from discord import Interaction
from github import Github, Repository

from global_config import LOCAL_DIR_LOC
from commands.psd_analyzer import (
    EXCLUDE_FOLDERS,
    GIT_TOKEN,
    UPDATE_RATE,
    CardValidator,
    PSDParser,
    RepositoryTraverser,
    stats_db,
)
from commands.query_card import card_repo


def traverse_local(traverser: RepositoryTraverser,
                   repository: str,
                   local_path: str,
                   interaction: Optional[Interaction] = None,
                   output_problematic: bool = True,
                   use_local_timestamp: bool = True,
                   force_update: bool = False) -> tuple[list[str], int]:
    """
    Traverse local repository directory.

    Args:
        traverser: RepositoryTraverser whose db receives the parsed cards
        repository: Repository name (owner/repo), only used for remote timestamps
        local_path: Path to local repository
        interaction: Optional Discord interaction
        output_problematic: Whether to output problematic cards
        use_local_timestamp: Use local file timestamps vs remote (True is what B6 is about)

    Returns:
        (formatted problem card messages, number of new or updated cards)
    """
    if not use_local_timestamp:
        print(f"Warning: Remote timestamps use API requests (limit: {5000 if GIT_TOKEN else 60}/hour)")
        traverser._github_client = Github(login_or_token=GIT_TOKEN)
        repo = traverser._github_client.get_repo(repository)
    else:
        repo = None

    traverser._populate_types_from_local(local_path)
    traverser.parser = PSDParser(traverser.db.all_types, traverser.db, traverser._icons_from_local(local_path))

    problems, num_new = _process_local_files(traverser, local_path, repo, interaction, output_problematic,
                                             use_local_timestamp, force_update)

    # Update old_stats paths after processing all files
    traverser._update_old_stats_paths()

    return (problems, num_new)


def _process_local_files(traverser: RepositoryTraverser,
                         local_path: str,
                         repo: Optional[Repository.Repository],
                         interaction: Optional[Interaction],
                         output_problematic: bool,
                         use_local_timestamp: bool,
                         force_update: bool) -> tuple[list[str], int]:
    """Process files from local directory."""
    db = traverser.db
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
            relative_path = relpath(full_path, local_path).replace('\\', '/')
            name = splitext(basename(relative_path))[0].replace('_', ' ')
            db.dirty_files.append(name)

            # Get timestamp
            author = None
            if use_local_timestamp:
                timestamp = getmtime(full_path)
            else:
                timestamp, author = traverser._get_remote_timestamp(repo, relative_path, name)

            # Check if update needed
            should_update, is_new = traverser._should_update_card(name, timestamp, relative_path)

            if not force_update and not should_update:
                num_old += 1
            else:
                # Archive old version before updating (if not new)
                if should_update and not is_new and name in db.stats:
                    old_card = db.stats[name]
                    db.old_stats[name].append(old_card)

                if is_new:
                    num_new += 1
                # Check if it's a move vs a content update
                elif name in db.stats and db.stats[name].path != relative_path:
                    num_moved += 1
                else:
                    num_new += 1

                # Parse local file
                card = traverser.parser.parse(full_path, relative_path)
                card.timestamp = timestamp
                # Set author if it was fetched earlier
                if author and not card.author:
                    card.author = author
                db.stats[name] = card

                # Validate
                if output_problematic:
                    problems = CardValidator.validate(card)
                    if problems:
                        problematic_cards.append((relative_path, card, problems))

            num_updated += 1
            if num_updated % UPDATE_RATE == 0:
                traverser._send_progress(interaction, num_updated)

    traverser._send_summary(interaction, num_new, num_old, num_moved)
    return (traverser._format_problems(problematic_cards), num_new)


def update_stats_local(output_problematic: bool = True,
                       use_local_timestamp: bool = True,
                       force_update: bool = False,
                       local_path: str = LOCAL_DIR_LOC) -> list[str]:
    """
    The old local branch of update_stats(), without Discord. Saves the pickles when anything was new or newer.

    Args:
        output_problematic: Whether to print problematic cards
        use_local_timestamp: File mtimes (True, B6: every card counts as updated) or GitHub commit dates (False,
                             one API call per card)
        force_update: Re-parse every card even if its timestamp is not newer
        local_path: TTSCardMaker clone to walk

    Returns:
        List of formatted problem messages
    """
    print("Updating card statistics database from the local clone (archived local mode)...")
    stats_db.load()
    stats_db.dirty_files.clear()
    traverser = RepositoryTraverser(stats_db)
    problem_cards, num_new = traverse_local(traverser,
                                            card_repo.repository,
                                            expanduser(local_path),
                                            None,
                                            output_problematic,
                                            use_local_timestamp,
                                            force_update)
    if num_new > 0:
        stats_db.prune_clean_cards()
        stats_db.save()
    print("Card statistics update complete!")
    for problem in problem_cards:
        print(problem)
    return problem_cards
