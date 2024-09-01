from discord import Message
from git import GitCommandError
from git.cmd import Git
from os import execv, system, getpid
from sys import argv, executable

import config

def perms_check(message: Message):
    try:
        admins = config.admins
    except AttributeError:
        return 1

    if message.author.id not in admins:
        return 1
    return 0

async def restart_bot(message: Message):
    await message.channel.send("Restarting bot!")
    try:
        execv('./startup.sh', argv)
    except FileNotFoundError:
        execv(executable, ['python'] + argv)

async def stop_bot(message: Message):
    if perms_check(message) != 0:
        await message.channel.send("You do not have permission to stop the bot.")
        return

    await message.channel.send("Stopping bot!")
    # Mac kill
    system('kill %d' % getpid())
    # Windows kill
    system('taskkill /F /PID %d' % getpid())

async def git_push(message: Message):
    await message.channel.send("Pushing aliases.pkl...")
    try:
        Git(argv).add('aliases.pkl')
        Git(argv).commit('-m', 'aliases.pkl')
        Git(argv).push()
        await message.channel.send("Succesfully pushed!")
    except GitCommandError:
        await message.channel.send("aliases.pkl is already up to date.")

async def git_pull(message: Message):
    await message.channel.send("Doing a git pull!")
    await message.channel.send("%s" % Git(argv).reset('--hard'))
    await message.channel.send("%s" % Git(argv).pull())

async def update_bot(message: Message):
    await git_push(message)
    await git_pull(message)
    await restart_bot(message)

async def purge(message: Message, limit : int, id: str, bulk = False):
    if perms_check(message) != 0:
        await message.channel.send("You do not have permission to stop purge messages.")
        return

    if limit == -1:
        limit = 1000000

    await message.channel.send("Purging messages from user %s." % id)
    await message.channel.send("Deleted %d message(s) from user %s." % (len(await message.channel.purge(limit = limit, check = lambda message : message.author.id == id, bulk = bulk)), id))
