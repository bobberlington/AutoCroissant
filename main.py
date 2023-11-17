# Imports
import discord
from discord import app_commands

import global_config
import config
from commands.update_bot import restart_bot
from commands.query_card import query_card, find_description, howmany_description, set_match_ratio

# Intents permissions
intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents=intents)
tree = app_commands.CommandTree(client)
commands = global_config.commands

# Events
@client.event
async def on_ready():
    print(f'We have logged in as {client.user}')

@client.event
async def on_message(message):
    # This ID is for the GitHub webhook bot from the TTS repo
    if message.author.id == 1011982177023561840:
        await restart_bot(message)
    elif message.content.startswith("?find"):
        await find_description(message)
    elif message.content.startswith("?howmany"):
        await howmany_description(message)
    elif message.content.startswith("?set_ratio"):
        await set_match_ratio(message)
    elif message.content.startswith("?"):
        await query_card(message)
    else:
        for key, val in commands.items():
            if message.content.startswith(key):
                await val(message)
                break


client.run(config.token)
