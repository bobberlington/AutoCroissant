from discord import Attachment, Client, Intents, Interaction, Member, Message, FFmpegPCMAudio, Object, app_commands
from discord.ext import tasks
from discord.app_commands import Choice, CommandTree
from pathlib import Path
from threading import Thread
from typing import Optional

from config import token
import global_config
from commands.diffusion import init_pipeline, diffusion, set_lora, set_model, set_device, set_scheduler, get_qsize
from commands.query_card import try_open_alias, try_open_descriptions, populate_files, query_remote, query_pickle, howmany_description, set_match_ratio, set_repository, alias_card, delete_alias
from commands.update_bot import restart_bot_github, stop_bot, git_pull, git_push, update_bot, restart_bot, purge
from commands.utils import music, prev_music, messages, files, commands, to_thread
from commands.help import print_help
from commands.frankenstein import frankenstein
from commands.music_player import play_music, replay_all, replay, skip, loop, list_all_music, set_volume, shuffle_music, print_prev_queue, print_queue, clear_queue, pause, stop, disconnect, play_all

# Intents permissions
intents = Intents.default()
intents.message_content = True
client = Client(intents=intents)
tree = CommandTree(client)


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


# Slash Commands
####################
# GENERAL COMMANDS #
####################
@tree.command(name="help", description="Get bot help.")
@app_commands.describe(
    help_type='The type of command you want help for.')
@app_commands.choices(help_type=[
    Choice(name="cards", value="card"),
    Choice(name="ai", value="ai"),
    Choice(name="music", value="music"),
    Choice(name="general", value="general")
])
async def slash_print_help(interaction: Interaction, help_type:Choice[str]):
    await print_help(interaction, help_type)
@tree.command(name="restart_bot", description="Restart the bot.")

async def slash_restart_bot(interaction: Interaction):
    await restart_bot(interaction)

@tree.command(name="stop_bot", description="Stop the bot.")
async def slash_stop_bot(interaction: Interaction):
    await stop_bot(interaction)

@tree.command(name="pull", description="Does a hard reset, than a git pull, and reports status.")
async def slash_pull(interaction: Interaction):
    await git_pull(interaction)

@tree.command(name="push", description="Does a git push, and reports status.")
async def slash_push(interaction: Interaction):
    await git_push(interaction)

@tree.command(name="update", description="Does a pull, then restarts.")
async def slash_update(interaction: Interaction):
    await update_bot(interaction)

@tree.command(name="purge", description="Deletes messages.")
@app_commands.describe(
    num='The number of messages to delete.',
    user='The user whose messages to delete. Default is the bot\'s own messages.',
    bulk='Enable bulk delete. Deletes faster but needs Manage Messages permission.')
async def slash_purge(interaction: Interaction, num: int, bulk: bool, user: Optional[Member]):
    if not user:
        user = client.user.id
    # Have to add +1 to num because the first message it deletes is the command message itself lol
    await purge(interaction, num if user != client.user.id else num + 1, user, bulk)


####################
#   CARD COMMANDS  #
####################
@tree.command(name="query_card", description="Query a card by name.")
@app_commands.describe(
    query='The name of the card you want to search for')
async def slash_query_remote(interaction: Interaction, query: str):
    await query_remote(interaction, query)

@tree.command(name="query_desc", description="Query a card by description.")
@app_commands.describe(
    query='The description you want to search for.')
async def slash_query_pickle(interaction: Interaction, query: str):
    await query_pickle(interaction, query)

@tree.command(name="howmany", description="Get the number of cards that match the given description.")
@app_commands.describe(
    query='The description you want to search for.')
async def slash_howmany_description(interaction: Interaction, query: str):
    await howmany_description(interaction, query)

@tree.command(name="set_ratio", description="View/set tolerance for unexact matches when searching.")
@app_commands.describe(
    value='If empty, posts the current tolerance. Otherwise sets it to this.')
async def slash_set_ratio(interaction: Interaction, value: Optional[float]):
    await set_match_ratio(interaction, value)

@tree.command(name="set_repo", description="View/set the repo that the bot looks at for cards.")
@app_commands.describe(
    new_repo='If empty, posts the current repository. Otherwise sets it to this.')
async def slash_set_ratio(interaction: Interaction, new_repo: Optional[str]):
    await set_repository(interaction, new_repo)

@tree.command(name="alias", description="View/set the card aliases.")
@app_commands.describe(
    alias='The alias you want to set. If either parameter is empty then print all aliases.',
    original_card='The path of the card you want to give the alias.')
async def slash_alias(interaction: Interaction, alias: Optional[str], original_card: Optional[str]):
    await alias_card(interaction, alias, original_card)

@tree.command(name="del_alias", description="Delete an alias.")
@app_commands.describe(
    alias='The alias you want to delete.')
async def slash_del_alias(interaction: Interaction, alias: str):
    await delete_alias(interaction, alias)

@tree.command(name="frankenstein", description="Tie a bunch of cards together.")
@app_commands.describe(
    cards='The cards that you want to frankenstein together, separated by commas.')
async def slash_frankenstein(interaction: Interaction, cards: str):
    await interaction.response.defer()
    await to_thread(frankenstein)(interaction, cards)


####################
#   AI COMMANDS    #
####################
@tree.command(name="ai", description="Generate AI images. '/help ai' for details.")
@app_commands.describe(
    prompt='A description of the image you want generated.',
    image='Modify this image based off the prompt.',
    mask_image='Control which parts of the image to modify. Black = dont modify, white = modify.',
    url='Modify the image at this url based off the prompt.',
    mask_url='Uses the image at this url as the control mask.',
    steps='How long to spend generating the image. Default = 50.',
    height='How many pixels tall should the image be. Default = 512.',
    width='How many pixels wide should the image be. Default = 512.',
    resize='Multiply the output image\'s size by this value. Default = 1 (1x).',
    cfg='Scalar between creativity and conformity to prompt. Default = 7.',
    strength='How much of the attached image to modify. Default = 0.8 (80%).',
    seed='Used to control output, same seed = same output. Default is randomized.')
async def slash_ai(interaction: Interaction, prompt: str, image: Optional[Attachment], mask_image: Optional[Attachment], url: Optional[str], mask_url: Optional[str],
                   steps: Optional[int] = 50, height: Optional[int] = 512, width: Optional[int] = 512, resize: Optional[float] = 1.0, cfg: Optional[float] = 7.0, strength: Optional[float] = 0.8, seed: Optional[int] = None):
    await interaction.response.defer()
    await to_thread(diffusion)(interaction, prompt, image, mask_image, url, mask_url, steps, height, width, resize, cfg, strength, seed)

@tree.command(name="ai_queue", description="Returns the current ai queue size and the content of each queued request.")
async def slash_ai_queue(interaction: Interaction):
    await get_qsize(interaction)

@tree.command(name="set_scheduler", description="Set the scheduler.")
@app_commands.describe(
    scheduler='The scheduler you want. Leaving blank tells you the current scheduler.')
async def slash_set_schedule(interaction: Interaction, scheduler: Optional[str]):
    await set_scheduler(interaction, scheduler)

@tree.command(name="set_device", description="Set the device.")
@app_commands.describe(
    device='The device number you want to set. Leaving blank tells you the current device.')
async def slash_set_device(interaction: Interaction, device: Optional[int]):
    await set_device(interaction, device)

@tree.command(name="set_model", description="Set the model.")
@app_commands.describe(
    model='The model you want to set. Leaving blank tells you the current model.')
async def slash_set_model(interaction: Interaction, model: Optional[str]):
    await set_model(interaction, model)

@tree.command(name="set_lora", description="Set the lora.")
@app_commands.describe(
    lora='The lora you want to set. Leaving blank tells you the current lora.')
async def slash_set_lora(interaction: Interaction, lora: Optional[str]):
    await set_lora(interaction, lora)


####################
#  MUSIC COMMANDS  #
####################

@tree.command(name="play", description="Adds a song to the back of the queue, either a local file or a url.")
@app_commands.describe(
    song='The song you want to play.',
    play_next='True if you want to make this song play next. False by default.')
async def slash_play(interaction: Interaction, song: str, play_next: Optional[bool] = False):
    await interaction.response.defer()
    await play_music(interaction, song, play_next)

@tree.command(name="play_all", description="Play all the songs in the local music directory.")
async def slash_play_all(interaction: Interaction):
    await interaction.response.defer()
    await play_all(interaction)

@tree.command(name="replay", description="Replay a song.")
@app_commands.describe(
    song='0 is the current song, 1 is the previous song, 2 is 2 songs ago, etc. 0 by default.')
async def slash_replay(interaction: Interaction, song: Optional[int] = 0):
    await interaction.response.defer()
    await to_thread(replay)(interaction, song)

@tree.command(name="replay_all", description="Replay all songs.")
async def slash_replay_all(interaction: Interaction):
    await interaction.response.defer()
    await to_thread(replay_all)(interaction)

@tree.command(name="skip", description="Skip current song.")
async def slash_skip(interaction: Interaction):
    await interaction.response.defer()
    await to_thread(skip)(interaction)

@tree.command(name="loop", description="Loop current song.")
async def slash_loop(interaction: Interaction):
    await interaction.response.defer()
    await to_thread(loop)(interaction)

@tree.command(name="list", description="List all songs in the current directory.")
async def slash_list(interaction: Interaction):
    await interaction.response.defer()
    await to_thread(list_all_music)(interaction)

@tree.command(name="volume", description="Changes the volume.")
@app_commands.describe(
    multiplier='The multiplier of the volume. 0.5 = half as loud, 1 = default, 2 = twice as loud, etc.')
async def slash_volume(interaction: Interaction, multiplier: float = 1.0):
    await interaction.response.defer()
    await to_thread(set_volume)(interaction, multiplier)

@tree.command(name="shuffle", description="Shuffles the queue.")
async def slash_shuffle(interaction: Interaction):
    await interaction.response.defer()
    await to_thread(shuffle_music)(interaction)

@tree.command(name="queue", description="Shows the queue.")
async def slash_queue(interaction: Interaction):
    await interaction.response.defer()
    await to_thread(print_queue)(interaction)

@tree.command(name="prev_queue", description="Shows the previously queued songs.")
async def slash_prev_queue(interaction: Interaction):
    await interaction.response.defer()
    await to_thread(print_prev_queue)(interaction)

@tree.command(name="clear", description="Clears the queue.")
async def slash_clear(interaction: Interaction):
    await interaction.response.defer()
    await to_thread(clear_queue)(interaction)

@tree.command(name="pause", description="Pauses/unpauses the current song.")
async def slash_pause(interaction: Interaction):
    await interaction.response.defer()
    await to_thread(pause)(interaction)

@tree.command(name="stop", description="Stops playing the current song and clears the queue.")
async def slash_stop(interaction: Interaction):
    await interaction.response.defer()
    await to_thread(stop)(interaction)

@tree.command(name="disconnect", description="Disconnects the bot and clears the queue.")
async def slash_disconnect(interaction: Interaction):
    await interaction.response.defer()
    await to_thread(disconnect)(interaction)


@client.event
async def on_message(message: Message):
    # This ID is for the GitHub webhook bot from the TTS repo
    # This isn't a slash command because it really doesn't make sense to be one.
    if message.author.id == 1011982177023561840:
        await restart_bot_github(message)
    # This will sync all slash commands with the guild you post the message in if you're a bot admin.
    # This isn't a slash command because it might lead to an awkward catch-22 momento.
    elif message.content.startswith(".sync_guild") and message.author.id in global_config.bot_admin_ids:
        sent_guild = Object(id=message.guild.id)
        tree.copy_global_to(guild=sent_guild)
        await tree.sync(guild=sent_guild)
        await message.channel.send("Slash commands synced with this guild!")
    # This will sync all slash commands globally.
    # Resources online tells me it will take like a whole hour to sync if you do this, so it is not recommended
    # to use this until you actually plan to deploy the commands.
    elif message.content.startswith(".sync_global") and message.author.id in global_config.bot_admin_ids:
        await tree.sync()
        await message.channel.send("Slash commands synced globally! Might take some time, though...")

@tasks.loop(seconds=2)
async def check_pipeline():
    if len(music) > 0:
        from commands.music_player import last_channel, loop_song, vc
        if vc and not vc.is_playing():
            if loop_song:
                Thread(target=vc.play, daemon=True, kwargs={'source': FFmpegPCMAudio(source=prev_music[-1])}).start()
            else:
                cur_song = music.popleft()
                if Path(cur_song).is_file():
                    prev_music.append(cur_song)
                    if last_channel:
                        await last_channel.send("Playing next song: " + str(cur_song))
                    Thread(target=vc.play, daemon=True, kwargs={'source': FFmpegPCMAudio(source=cur_song)}).start()
    if len(messages) > 0:
        interaction, msg = messages.popleft()
        await interaction.followup.send(msg)
    if len(files) > 0:
        interaction, file = files.popleft()
        await interaction.followup.send(file=file)
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
