import config
import discord
from discord.ext import tasks
from threading import Thread

import global_config
from commands.diffusion import init_pipeline
from commands.query_card import try_open_alias, try_open_descriptions, populate_files, query_remote, query_pickle, howmany_description, set_match_ratio, set_repository
from commands.tools import messages, files, commands
from commands.update_bot import restart_bot, purge

# Intents permissions
intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

configured_commands = global_config.commands.items()

# Events
@client.event
async def on_ready():
    print(f'We have logged in as {client.user}')

    try_open_alias()
    status = populate_files()
    if status != 200:
       print(f"Error {status} when requesting github.")
    try_open_descriptions()

    commands.append((("not async",), init_pipeline))
    check_pipeline.start()
    print("Finished initializing.")

@client.event
async def on_message(message: discord.Message):
    # This ID is for the GitHub webhook bot from the TTS repo
    if message.author.id == 1011982177023561840:
        await restart_bot(message)
    elif message.content.startswith("?find"):
        await query_pickle(message)
    elif message.content.startswith("?howmany"):
        await howmany_description(message)
    elif message.content.startswith("?set_ratio"):
        await set_match_ratio(message)
    elif message.content.startswith("?set_repo"):
        await set_repository(message)
    elif message.content.startswith("?"):
        await query_remote(message)
    elif message.content.startswith(".purge"):
        await purge(message, -1, client.user.id)
    elif message.content.startswith(".quickpurge"):
        await purge(message, -1, client.user.id, bulk = True)
    else:
        for key, val in configured_commands:
            if message.content.startswith(key):
                await val(message)
                break

@tasks.loop(seconds=3)
async def check_pipeline():
    while len(messages) > 0:
        id, msg = messages.pop(0)
        await client.get_channel(id).send(msg)
    while len(files) > 0:
        id, file = files.pop(0)
        await client.get_channel(id).send(file=file)
    while len(commands) > 0:
        params, cmd = commands.pop(0)
        if params[0] == "not async":
            params = params[1:]
            if len(params) > 0:
                Thread(target=cmd, daemon=True, args=params).start()
            else:
                Thread(target=cmd, daemon=True).start()
        else:
            if len(params) > 0:
                await Thread(target=cmd, daemon=True, args=params).start()
            else:
                await Thread(target=cmd, daemon=True).start()

client.run(config.token)
