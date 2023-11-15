# Imports
import discord
from discord import app_commands

import aliases
import config
from commands.query_card import query_card

# Intents permissions
intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents=intents)
tree = app_commands.CommandTree(client)
commands = aliases.commands

# Events
@client.event
async def on_ready():
    print(f'We have logged in as {client.user}')

@client.event
async def on_message(message):
    if message.content.startswith('?'):
        await query_card(message)
    elif message.content in commands:
        await commands[message.content](message)


client.run(config.token)
