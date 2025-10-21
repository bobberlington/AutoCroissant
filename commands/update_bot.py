from discord import Interaction
from git import GitCommandError
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
        Git(argv).add(ALIAS_PKL)
        Git(argv).add(STATS_PKL)
        Git(argv).add(OLD_STATS_PKL)
        Git(argv).commit('-m', 'PICKLE')
        Git(argv).push()
        await interaction.followup.send("Succesfully pushed!")
    except GitCommandError:
        await interaction.followup.send("Pickles are already up to date.")


async def git_pull(interaction: Interaction):
    await interaction.followup.send(f"{Git(argv).reset('--hard')}")
    await interaction.followup.send(f"{Git(argv).pull()}")


async def update_bot(interaction: Interaction):
    await interaction.followup.send(f"Pushing {ALIAS_PKL}, {STATS_PKL}, and {OLD_STATS_PKL}.")
    await git_push(interaction)
    await interaction.followup.send("Doing a git pull!")
    await git_pull(interaction)
    await interaction.followup.send("Restarting bot!")
    restart_bot()


# ========================
# Module Exports
# ========================
__all__ = [
    'restart_bot',
    'git_push',
    'git_pull',
    'update_bot',
]
