from discord import Interaction
from git import GitCommandError, Repo
from git.cmd import Git
from os import execl, execv
from sys import argv, executable
from global_config import ALIAS_PKL, STATS_PKL, OLD_STATS_PKL


def restart_bot():
    try:
        execv('./startup.sh', argv)
    except FileNotFoundError:
        execl(executable, executable, *argv)


async def git_push(interaction: Interaction):
    try:
        git = Git('.')
        git.add(ALIAS_PKL)
        git.add(STATS_PKL)
        git.add(OLD_STATS_PKL)
        git.commit('-m', 'PICKLE')
        git.push()
        await interaction.followup.send("Successfully pushed!")
    except GitCommandError as e:
        if "nothing to commit" in str(e) or "up-to-date" in str(e):
            await interaction.followup.send("Pickles are already up to date.")
        else:
            await interaction.followup.send(f"Push failed: {str(e)}")
            raise


async def git_pull(interaction: Interaction):
    try:
        git = Git('.')
        repo = Repo('.')

        # First, fetch to see the remote state
        await interaction.followup.send("Fetching from remote...")
        git.fetch()

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
                    git.pull('origin', 'main', '--no-edit')
                    await interaction.followup.send("Merge successful!")
                except GitCommandError as merge_error:
                    if "CONFLICT" in str(merge_error) or repo.index.unmerged_blobs():
                        await interaction.followup.send("Merge conflict detected. Resolving automatically...")

                        # Get list of conflicted files
                        conflicted = [item[0] for item in repo.index.unmerged_blobs().keys()]

                        for file_path in conflicted:
                            # Keep ours for pickle files, keep theirs for everything else
                            if file_path in [ALIAS_PKL, STATS_PKL, OLD_STATS_PKL]:
                                await interaction.followup.send(f"Keeping local version of {file_path}")
                                git.checkout('--ours', file_path)
                            else:
                                await interaction.followup.send(f"Accepting remote version of {file_path}")
                                git.checkout('--theirs', file_path)

                            # Stage the resolved file
                            git.add(file_path)

                        # Complete the merge
                        git.commit('-m', 'AUTO-RESOLVE: Merge conflict resolved (kept local pickles, accepted remote code)')
                        await interaction.followup.send("Merge conflict resolved automatically!")
                    else:
                        raise

        # Now do the actual pull (or it's already done from the merge above)
        result = git.pull('origin', 'main')
        await interaction.followup.send(f"Pull complete:\n```{result}```")

    except GitCommandError as e:
        error_msg = str(e)
        await interaction.followup.send(f"Git pull failed: {error_msg}")

        # Try to recover
        if "divergent branches" in error_msg.lower():
            await interaction.followup.send("Attempting to reconcile divergent branches...")
            try:
                # Rebase strategy as backup
                git.pull('origin', 'main', '--rebase')
                await interaction.followup.send("Rebase successful!")
            except GitCommandError as rebase_error:
                await interaction.followup.send(f"Rebase also failed: {str(rebase_error)}")
                await interaction.followup.send("Manual intervention may be required.")
                raise


async def git_reset_hard(interaction: Interaction):
    """Reset local repository to match remote, discarding all local changes."""
    try:
        git = Git('.')

        # Fetch latest
        git.fetch()

        # Reset to remote
        result = git.reset('--hard', 'origin/main')
        await interaction.followup.send(f"Hard reset complete:\n```{result}```")
        await interaction.followup.send("Warning: All local changes have been discarded!")

    except GitCommandError as e:
        await interaction.followup.send(f"Hard reset failed: {str(e)}")
        raise


async def update_bot(interaction: Interaction, force_reset: bool = False):
    """
    Update the bot by pushing pickles and pulling latest code.

    Args:
        interaction: Discord interaction
        force_reset: If True, does a hard reset instead of trying to merge
    """
    try:
        if force_reset:
            await interaction.followup.send("Force reset mode: Local changes will be discarded!")
            await git_reset_hard(interaction)
        else:
            await interaction.followup.send(f"Pushing {ALIAS_PKL}, {STATS_PKL}, and {OLD_STATS_PKL}.")
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
