from discord import Attachment, Client, FFmpegPCMAudio, Intents, Interaction, Member, Message, Object, app_commands
from discord.app_commands import Choice, CommandTree
from discord.errors import HTTPException
from discord.ext import tasks
from pathlib import Path
from threading import Thread
from typing import Optional

from config import token
import global_config
from commands.diffusion import init_pipeline, diffusion, set_lora, set_model, set_device, set_scheduler, get_qsize
from commands.frankenstein import frankenstein
from commands.help import print_help
from commands.music_player import play_music, replay_all, replay, skip, loop, list_all_music, set_volume, shuffle_music, print_prev_queue, print_queue, clear_queue, pause, stop, disconnect, play_all
from commands.psd_analyzer import manual_update_stats, export_stats_to_file, export_rulebook_to_file, manual_metadata_entry, get_card_stats, mass_replace_author, list_orphans
from commands.query_card import try_open_alias, try_open_stats, populate_files, query_name, query_ability, query_ability_num_occur, set_match_ratio, set_repository, alias_card, delete_alias
from commands.update_bot import stop_bot, git_pull, git_push, update_bot, restart_bot, purge
from commands.utils import music, prev_music, messages, edit_messages, files, commands, to_thread

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

    commands.append(((), try_open_alias))
    commands.append(((), populate_files))
    commands.append(((), try_open_stats))
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
    Choice(name="card", value="card"),
    Choice(name="ai", value="ai"),
    Choice(name="music", value="music"),
    Choice(name="general", value="general")
])
async def slash_print_help(interaction: Interaction, help_type: Choice[str]):
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
    user='The user whose messages to delete. Default is the bot\'s own messages.',
    num='The number of messages to delete. Default = 100.',
    bulk='Enable bulk delete. Deletes faster but needs Manage Messages permission. Default = false.')
async def slash_purge(interaction: Interaction, user: Optional[Member], num: Optional[int] = 100, bulk: Optional[bool] = False):
    if not user:
        user = client.user.id
    # Have to add +1 to num because the first message it deletes is the command message itself lol
    await purge(interaction, num if user != client.user.id else num + 1, user, bulk)


####################
#   CARD COMMANDS  #
####################
@tree.command(name="query", description="Query a card by name.")
@app_commands.describe(
    query='The name of the card you want to search for')
async def slash_query_name(interaction: Interaction, query: str):
    await query_name(interaction, query)

@tree.command(name="query_ability", description="Query a card by ability.")
@app_commands.describe(
    query='The ability text you want to search for.',
    limit='The number of cards to output before stopping.',
    howmany='Instead of returning card images, return the number of matches.',
    filter_raids='Removes 6+ star cards from the results')
async def slash_query_ability(interaction: Interaction, query: str, limit: Optional[int] = -1, howmany: Optional[bool] = False, filter_raids: Optional[bool] = False):
    if howmany:
        await query_ability_num_occur(interaction, query)
    else:
        await query_ability(interaction, query, limit, filter_raids)

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
#   STAT COMMANDS  #
####################
@tree.command(name="update_stats", description="Manual call to update card stats.")
@app_commands.describe(
    output_problematic_cards='Should we output cards with consistency errors?',
    use_local_repo='Should we use the local cloned folder for updates?',
    use_local_timestamp='Should we use the local files modified time for timestamping?')
async def slash_update_stats(interaction: Interaction, output_problematic_cards: Optional[bool] = True, use_local_repo: Optional[bool] = True, use_local_timestamp: Optional[bool] = True):
    await interaction.response.defer()
    await to_thread(manual_update_stats)(interaction, output_problematic_cards, use_local_repo, use_local_timestamp)

@tree.command(name="export_abilities", description="Exports all abilities as a text file.")
@app_commands.describe(
    only_ability='Should we only output the ability of cards?',
    as_csv='Should we output the file as a csv?')
async def slash_export_abilities(interaction: Interaction, only_ability: Optional[bool] = True, as_csv: Optional[bool] = True):
    await interaction.response.defer()
    await export_stats_to_file(interaction, only_ability, as_csv)

@tree.command(name="export_rulebook", description="Exports the rulebook as a text file.")
async def slash_export_rulebook(interaction: Interaction):
    await interaction.response.defer()
    await export_rulebook_to_file(interaction)

@tree.command(name="update_metadata", description="Input metadata for a card.")
@app_commands.describe(
    query='The card to edit the metadata for.',
    del_entry='Should this key entry be deleted.',
    author='The creator of the card.')
async def slash_update_metadata(interaction: Interaction, query: str = "", del_entry: Optional[bool] = False, author: Optional[str] = None):
    await interaction.response.defer()
    await to_thread(manual_metadata_entry)(interaction, query, del_entry, author)

@tree.command(name="query_stats", description="Query stats and metadata specifics for a card.")
@app_commands.describe(
    query='The card to edit the metadata for.')
async def slash_query_stats(interaction: Interaction, query: str = ""):
    await interaction.response.defer()
    await to_thread(get_card_stats)(interaction, query)

@tree.command(name="replace_author", description="Replace all instances of one author with another.")
@app_commands.describe(
    author1='The author being replaced.',
    author2='The author to change author1 to.')
async def slash_update_metadata(interaction: Interaction, author1: str = "", author2: str = ""):
    await interaction.response.defer()
    await to_thread(mass_replace_author)(interaction, author1, author2)

@tree.command(name="list_orphans", description="Output all cards without a listed author.")
async def slash_list_orphans(interaction: Interaction):
    await interaction.response.defer()
    await to_thread(list_orphans)(interaction)


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
                   steps: Optional[int] = 50, height: Optional[int] = 512, width: Optional[int] = 512, resize: Optional[float] = 1.0, cfg: Optional[float] = 7.0, strength: Optional[float] = 0.8, seed: Optional[str] = None):
    if seed:
        # ints are limited to 15 digits or less by discord, so I need to take it as a str and then convert it to an int
        try:
            seed = int(seed)
        except ValueError:
            await interaction.response.send_message("Seed must be an integer.")
            return
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
        commands.append(((), try_open_alias))
        commands.append(((), populate_files))
        commands.append(((None, False, False, False), manual_update_stats))
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
    elif message.content.startswith(".clear_commands") and message.author.id in global_config.bot_admin_ids:
        sent_guild = Object(id=message.guild.id)
        tree.clear_commands(guild=None)
        tree.clear_commands(guild=sent_guild)
        await message.channel.send("Cleared commands! Make sure to resync the commands.")

@tasks.loop(seconds=1)
async def check_pipeline():
    if len(music) > 0:
        from commands.music_player import last_channel, loop_song, vc
        if vc and (not vc.is_playing() and not vc.is_paused()):
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
        try:
            await interaction.followup.send(content=msg)
        except HTTPException:
            await client.get_channel(interaction.channel_id).send(content=msg)
    if len(edit_messages) > 0:
        interaction, msg, attachments = edit_messages.popleft()
        try:
            await interaction.edit_original_response(content=msg, attachments=attachments)
        except HTTPException: # Fallback if we can't edit the message
            await client.get_channel(interaction.channel_id).send(content=msg, files=attachments)
    if len(files) > 0:
        interaction, file = files.popleft()
        try:
            await interaction.followup.send(file=file)
        except HTTPException:
            await client.get_channel(interaction.channel_id).send(file=file)
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
