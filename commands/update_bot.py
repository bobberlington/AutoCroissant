import sys
import os
import discord
import git

import config

def perms_check(message: discord.Message):
    try:
        admins = config.admins
    except AttributeError:
        admins = []

    if message.author.id not in admins:
        return 1
    return 0

async def restart_bot(message: discord.Message):
    await message.channel.send("Restarting bot!")
    try:
        os.execv('./startup.sh', sys.argv)
    except FileNotFoundError:
        os.execv(sys.executable, ['python'] + sys.argv)

async def stop_bot(message: discord.Message):
    if perms_check(message) != 0:
        await message.channel.send("You do not have permission to stop the bot.")
        return

    await message.channel.send("Stopping bot!")
    # Mac kill
    os.system('kill %d' % os.getpid())
    # Windows kill
    os.system('taskkill /F /PID %d' % os.getpid())

async def git_push(message: discord.Message):
    await message.channel.send("Pushing aliases.pkl...")
    try:
        git.cmd.Git(sys.argv).add('aliases.pkl')
        git.cmd.Git(sys.argv).commit('-m', 'aliases.pkl')
        git.cmd.Git(sys.argv).push()
        await message.channel.send("Succesfully pushed!")
    except git.GitCommandError:
        await message.channel.send("aliases.pkl is already up to date.")

async def git_pull(message: discord.Message):
    await message.channel.send("Doing a git pull!")
    await message.channel.send("%s" % git.cmd.Git(sys.argv).reset('--hard'))
    await message.channel.send("%s" % git.cmd.Git(sys.argv).pull())

async def update_bot(message: discord.Message):
    await git_push(message)
    await git_pull(message)
    await restart_bot(message)

async def purge(message: discord.Message, limit : int, id: str, bulk = False):
    if perms_check(message) != 0:
        await message.channel.send("You do not have permission to stop purge messages.")
        return

    if limit == -1:
        limit = 1000000

    await message.channel.send("Purging messages from user %s." % id)
    await message.channel.send("Deleted %d message(s) from user %s." % (len(await message.channel.purge(limit = limit, check = lambda message : message.author.id == id, bulk = bulk)), id))
