from discord import Interaction
from os import execl, execv
from sys import argv, executable
from commands.repo_history import batch_overrides

NET_TIMEOUT = 120                       # seconds per git network call (fetch, pull)


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


def restart_bot():
    try:
        execv('./startup.sh', argv)
    except FileNotFoundError:
        execl(executable, executable, *argv)


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

                # Strategy: accept the remote's side of any conflict
                try:
                    # Pull with merge strategy
                    git.pull('origin', 'main', '--no-edit', kill_after_timeout=NET_TIMEOUT)
                    await interaction.followup.send("Merge successful!")
                except GitCommandError as merge_error:
                    if "CONFLICT" in str(merge_error) or repo.index.unmerged_blobs():
                        await interaction.followup.send("Merge conflict detected. Resolving automatically...")

                        # Get list of conflicted files
                        conflicted = [item[0] for item in repo.index.unmerged_blobs().keys()]

                        for file_path in conflicted:
                            await interaction.followup.send(f"Accepting remote version of {file_path}")
                            git.checkout('--theirs', file_path)

                            # Stage the resolved file
                            git.add(file_path)

                        # Complete the merge
                        git.commit('-m', 'AUTO-RESOLVE: Merge conflict resolved (accepted remote)')
                        await interaction.followup.send("Merge conflict resolved automatically!")
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
    Update the bot by pulling the latest code.

    Args:
        interaction: Discord interaction
        force_reset: If True, does a hard reset instead of trying to merge
    """
    try:
        if force_reset:
            await interaction.followup.send("Force reset mode: Local changes will be discarded!")
            await git_reset_hard(interaction)
        else:
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
    'git_pull',
    'git_reset_hard',
    'update_bot',
]
