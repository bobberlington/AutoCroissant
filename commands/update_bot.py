from discord import Interaction
from os import execl, execv
from os.path import isfile
from pickle import loads
from sys import argv, executable
from commands.repo_history import batch_overrides
import global_config                    # CARD_SYNC, read at call time (the cutover switch)
from global_config import ALIAS_PKL, CARD_REPO, DETAILS_JSON, STATS_PKL, OLD_STATS_PKL

NET_TIMEOUT = 120                       # seconds per git network call (fetch, pull, push)
PUSH_RETIRED = ("Nothing to push: the card data is built by the Card sync Action in the card repo every 30 minutes "
                f"(<https://github.com/{CARD_REPO}/actions/workflows/card-sync.yml>), and /update_stats loads it. "
                "The pickles here are frozen since the cutover.")


# GitPython is imported in the functions: startup never loads it
def _git():
    """GitPython's Git on the bot's checkout, non-interactive."""
    from git.cmd import Git
    git = Git('.')
    git.update_environment(**batch_overrides('.'))
    return git


def _repo():
    """GitPython's Repo on the bot's checkout, non-interactive."""
    from git import Repo
    repo = Repo('.')
    repo.git.update_environment(**batch_overrides('.'))
    return repo


def _rescue_merged_cards() -> list:
    """
    After a merge commit: cards the file-by-file pickle merge dropped (stats.pkl from one side, old_stats.pkl from
    the other) go back into old_stats.pkl from the side that had them, committed as PICKLE. Returns their names.
    """
    from commands.card_models import _old_stats_lock, rescue_dropped, save_pickle
    repo = _repo()
    head = repo.head.commit
    if len(head.parents) != 2:
        return []

    def read(commit, path) -> dict:
        try:
            return loads((commit.tree / path).data_stream.read())
        except KeyError:                                # not in that commit
            return {}
    merged_old = read(head, OLD_STATS_PKL)
    names = rescue_dropped(read(head, STATS_PKL), merged_old,
                           [(read(parent, STATS_PKL), read(parent, OLD_STATS_PKL)) for parent in head.parents])
    if names:
        save_pickle(merged_old, OLD_STATS_PKL, _old_stats_lock)
        git = _git()
        git.add(OLD_STATS_PKL)
        git.commit('-m', 'PICKLE')
    return names


async def _rescue_after_merge(interaction: Interaction) -> None:
    """_rescue_merged_cards, told in Discord; a failure never fails the pull."""
    try:
        names = _rescue_merged_cards()
    except Exception as e:
        await interaction.followup.send(f"Could not check the merged pickles for dropped cards: {e}")
        return
    if names:
        await interaction.followup.send("Kept in old_stats (the merge dropped them): " + ", ".join(names))


def restart_bot():
    try:
        execv('./startup.sh', argv)
    except FileNotFoundError:
        execl(executable, executable, *argv)


async def git_push(interaction: Interaction):
    if global_config.CARD_SYNC:                         # the cutover: the bot writes no card data
        await interaction.followup.send(PUSH_RETIRED)
        return
    from git import GitCommandError
    try:
        git = _git()
        git.add(ALIAS_PKL)
        git.add(STATS_PKL)
        git.add(OLD_STATS_PKL)
        if isfile(DETAILS_JSON):                        # the website's PSD details travel with the pickles
            git.add(DETAILS_JSON)                       # (a never-tracked missing path would abort the push)
        try:
            git.commit('-m', 'PICKLE')
        except GitCommandError as e:
            if "nothing to commit" not in str(e):
                raise
        try:
            git.pull('--rebase', '--autostash', 'origin', 'main', kill_after_timeout=NET_TIMEOUT)    # code pushed since the last pull
        except GitCommandError as e:
            try:
                git.rebase('--abort')
            except GitCommandError:
                pass                                    # failed before the rebase began (fetch, timeout)
            raise e
        if git.rev_list('--count', 'origin/main..HEAD').strip() == "0":
            await interaction.followup.send("Pickles are already up to date.")
            return
        git.push(kill_after_timeout=NET_TIMEOUT)
        await interaction.followup.send("Successfully pushed!")
    except GitCommandError as e:
        await interaction.followup.send(f"Push failed: {str(e)}")
        raise


async def git_pull(interaction: Interaction):
    from git import GitCommandError
    try:
        git = _git()
        repo = _repo()

        # First, fetch to see the remote state
        await interaction.followup.send("Fetching from remote...")
        git.fetch(kill_after_timeout=NET_TIMEOUT)

        # Check if there are divergent branches
        local_commit = repo.head.commit.hexsha
        remote_commit = repo.refs['origin/main'].commit.hexsha

        if local_commit != remote_commit:
            # Check if we need to merge
            merge_base = repo.merge_base(local_commit, remote_commit)

            if merge_base and merge_base[0].hexsha != remote_commit:
                await interaction.followup.send("Branches have diverged. Attempting to resolve...")

                # Strategy: Keep our pickle files, accept everything else from remote
                try:
                    # Pull with merge strategy
                    git.pull('origin', 'main', '--no-edit', kill_after_timeout=NET_TIMEOUT)
                    await interaction.followup.send("Merge successful!")
                    if not global_config.CARD_SYNC:     # the cutover: the pickles are frozen
                        await _rescue_after_merge(interaction)
                except GitCommandError as merge_error:
                    if "CONFLICT" in str(merge_error) or repo.index.unmerged_blobs():
                        await interaction.followup.send("Merge conflict detected. Resolving automatically...")

                        # Get list of conflicted files
                        conflicted = [item[0] for item in repo.index.unmerged_blobs().keys()]

                        keep = [] if global_config.CARD_SYNC else [ALIAS_PKL, STATS_PKL, OLD_STATS_PKL, DETAILS_JSON]
                        for file_path in conflicted:
                            # Keep ours for pickle files (until the cutover), keep theirs for everything else
                            if file_path in keep:
                                await interaction.followup.send(f"Keeping local version of {file_path}")
                                git.checkout('--ours', file_path)
                            else:
                                await interaction.followup.send(f"Accepting remote version of {file_path}")
                                git.checkout('--theirs', file_path)

                            # Stage the resolved file
                            git.add(file_path)

                        # Complete the merge
                        if global_config.CARD_SYNC:
                            git.commit('-m', 'AUTO-RESOLVE: Merge conflict resolved (accepted remote)')
                        else:
                            git.commit('-m', 'AUTO-RESOLVE: Merge conflict resolved (kept local pickles, accepted remote code)')
                        await interaction.followup.send("Merge conflict resolved automatically!")
                        if not global_config.CARD_SYNC:
                            await _rescue_after_merge(interaction)
                    else:
                        raise

        # Now do the actual pull (or it's already done from the merge above)
        result = git.pull('origin', 'main', kill_after_timeout=NET_TIMEOUT)
        await interaction.followup.send(f"Pull complete:\n```{result}```")

    except GitCommandError as e:
        error_msg = str(e)
        await interaction.followup.send(f"Git pull failed: {error_msg}")

        # Try to recover
        if "divergent branches" in error_msg.lower():
            await interaction.followup.send("Attempting to reconcile divergent branches...")
            try:
                # Rebase strategy as backup
                git.pull('origin', 'main', '--rebase', kill_after_timeout=NET_TIMEOUT)
                await interaction.followup.send("Rebase successful!")
            except GitCommandError as rebase_error:
                await interaction.followup.send(f"Rebase also failed: {str(rebase_error)}")
                await interaction.followup.send("Manual intervention may be required.")
                raise


async def git_reset_hard(interaction: Interaction):
    """Reset local repository to match remote, discarding all local changes."""
    from git import GitCommandError
    try:
        git = _git()

        # Fetch latest
        git.fetch(kill_after_timeout=NET_TIMEOUT)

        # Reset to remote
        result = git.reset('--hard', 'origin/main')
        await interaction.followup.send(f"Hard reset complete:\n```{result}```")
        await interaction.followup.send("Warning: All local changes have been discarded!")

    except GitCommandError as e:
        await interaction.followup.send(f"Hard reset failed: {str(e)}")
        raise


async def update_bot(interaction: Interaction, force_reset: bool = False):
    """
    Update the bot by pushing pickles (until the cutover) and pulling latest code.

    Args:
        interaction: Discord interaction
        force_reset: If True, does a hard reset instead of trying to merge
    """
    try:
        if force_reset:
            await interaction.followup.send("Force reset mode: Local changes will be discarded!")
            await git_reset_hard(interaction)
        else:
            if not global_config.CARD_SYNC:             # the cutover: no pickles to push
                await interaction.followup.send(f"Pushing {ALIAS_PKL}, {STATS_PKL}, {OLD_STATS_PKL} and {DETAILS_JSON}.")
                await git_push(interaction)

            await interaction.followup.send("Pulling latest changes from remote!")
            await git_pull(interaction)

        await interaction.followup.send("Restarting bot!")
        restart_bot()

    except Exception as e:
        await interaction.followup.send(f"Update failed: {str(e)}")
        await interaction.followup.send("Bot was NOT restarted due to errors.")
        raise


# ========================
# Module Exports
# ========================
__all__ = [
    'restart_bot',
    'git_push',
    'git_pull',
    'git_reset_hard',
    'update_bot',
]
