import sys
import os
import git

async def restart_bot(message):
    await message.channel.send("Restarting bot!")
    os.execv(sys.executable, ['python'] + sys.argv)

async def stop_bot(message, admins):
    if message.author.id not in admins:
        await message.channel.send("You do not have permission to stop the bot.")
        return
    await message.channel.send("Stopping bot!")
    os.system('kill %d' % os.getpid())

async def git_pull(message):
    await message.channel.send("Doing a git pull!")
    await message.channel.send("%s" % git.cmd.Git(sys.argv).pull())

async def update_bot(message):
    await git_pull(message)
    await restart_bot(message)
