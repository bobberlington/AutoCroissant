from discord import Client, Intents, Message, FFmpegPCMAudio
from discord.ext import tasks
from pathlib import Path
from threading import Thread

from config import token
import global_config
from commands.diffusion import init_pipeline
from commands.query_card import try_open_alias, try_open_descriptions, populate_files, query_remote, query_pickle, howmany_description, set_match_ratio, set_repository
from commands.tools import music, prev_music, messages, files, commands
from commands.update_bot import restart_bot, purge

# Intents permissions
intents = Intents.default()
intents.message_content = True
client = Client(intents=intents)

configured_commands = global_config.commands.items()

# Events
@client.event
async def on_ready():
    print(f'We have logged in as {client.user}')
    check_pipeline.start()

    try_open_alias()
    status = populate_files()
    if status != 200:
       print(f"Error {status} when requesting github.")
    try_open_descriptions()

    commands.append(((), init_pipeline))
    print("Finished initializing.")

@client.event
async def on_message(message: Message):
    # This ID is for the GitHub webhook bot from the TTS repo
    if message.author.id == 1011982177023561840:
        await restart_bot(message)
    elif message.content.startswith("??"):
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

@tasks.loop(seconds=2)
async def check_pipeline():
    if len(music) > 0:
        from commands.music_player import last_channel_id, loop_song, vc
        if vc and not vc.is_playing():
            if loop_song:
                Thread(target=vc.play, daemon=True, kwargs={'source': FFmpegPCMAudio(source=prev_music[-1]), 'bitrate': 320, 'signal_type': "music"}).start()
            else:
                cur_song = music.popleft()
                if Path(cur_song).is_file():
                    prev_music.append(cur_song)
                    if last_channel_id:
                        messages.append((last_channel_id, "Playing next song: " + cur_song))
                    Thread(target=vc.play, daemon=True, kwargs={'source': FFmpegPCMAudio(source=cur_song), 'bitrate': 320, 'signal_type': "music"}).start()
    if len(messages) > 0:
        id, msg = messages.popleft()
        await client.get_channel(id).send(msg)
    if len(files) > 0:
        id, file = files.popleft()
        await client.get_channel(id).send(file=file)
    if len(commands) > 0:
        params, cmd = commands.popleft()
        if not params:
            Thread(target=cmd, daemon=True).start()
        elif params[0] != "await":
            Thread(target=cmd, daemon=True, args=params).start()
        else:
            params = params[1:]
            await cmd(*params)

client.run(token)
