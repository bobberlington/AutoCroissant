# Imports
import discord
from discord import app_commands

import config
from commands.query_card import query_card
from commands.update_bot import restart_bot, stop_bot, git_pull, update_bot

# Intents permissions
intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents=intents)
tree = app_commands.CommandTree(client)
try:
    admins = config.admins
except AttributeError:
    admins = []

# Events
@client.event
async def on_ready():
    print(f'We have logged in as {client.user}')

@client.event
async def on_message(message):
    if message.content.startswith('?'):
        await query_card(message)
    elif message.content.startswith('.'):
        if message.content.endswith('restart'):
            await restart_bot(message)
        elif message.content.endswith('stop'):
            await stop_bot(message, admins)
        elif message.content.endswith('pull'):
            await git_pull(message)
        elif message.content.endswith('update'):
            await update_bot(message)


client.run(config.token)
