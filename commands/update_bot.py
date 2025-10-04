from discord import Interaction
from git import GitCommandError
from git.cmd import Git
from os import execv, system, getpid
from sys import argv, executable

from global_config import ALIAS_PKL, STATS_PKL, OLD_STATS_PKL, METAD_PKL

def perms_check(interaction: Interaction) -> int:
    try:
        import global_config
        admins = global_config.bot_admin_ids
    except AttributeError:
        return 1

    if interaction.user.id not in admins:
        return 1
    return 0

async def restart_bot(interaction: Interaction):
    await interaction.response.send_message("Restarting bot!")
    try:
        execv('./startup.sh', argv)
    except FileNotFoundError:
        execv(executable, ['python'] + argv)

async def stop_bot(interaction: Interaction):
    if perms_check(interaction) != 0:
        await interaction.response.send_message("You do not have permission to stop the bot.")
        return

    await interaction.response.send_message("Stopping bot!")
    # Mac kill
    system(f"kill {getpid()}")
    # Windows kill
    system(f"taskkill /F /PID {getpid()}")

async def git_push(interaction: Interaction):
    await interaction.response.send_message("Pushing aliases.pkl...")
    try:
        Git(argv).add(ALIAS_PKL)
        Git(argv).add(STATS_PKL)
        Git(argv).add(OLD_STATS_PKL)
        Git(argv).add(METAD_PKL)
        Git(argv).commit('-m', 'PICKLE')
        Git(argv).push()
        await interaction.followup.send("Succesfully pushed!")
    except GitCommandError:
        await interaction.followup.send("Pickles are already up to date.")

async def git_pull(interaction: Interaction):
    await interaction.response.send_message("Doing a git pull!")
    await interaction.followup.send(f"{Git(argv).reset('--hard')}")
    await interaction.followup.send(f"{Git(argv).pull()}")

async def update_bot(interaction: Interaction):
    await git_push(interaction)
    await git_pull(interaction)
    await restart_bot(interaction)

async def purge(interaction: Interaction, limit : int, id: int, bulk = False):
    await interaction.response.defer()
    if perms_check(interaction) != 0:
        await interaction.followup.send("You do not have permission to purge messages.")
        return

    if limit == -1:
        limit = 1000000

    channel = interaction.channel
    await interaction.followup.send(f"Purging messages from user {id}.")
    num_messages = len(await channel.purge(limit=limit, check=lambda message: message.author.id == id, bulk=bulk))
    await channel.send(f"Deleted {num_messages} message(s) from user {id}.")
