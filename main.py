from asyncio import create_task
from sys import exit
from discord import Attachment, Client, Intents, Interaction, Member, app_commands
from discord.app_commands import Choice, CommandTree
from discord.errors import HTTPException
from discord.ext import tasks
from typing import Optional

from config import TOKEN
from commands.analytics import (
    check_reminder,
    init_reminder,
    list_reminders,
    remove_reminder,
    set_reminder,
)
from commands.diffusion import (
    diffusion,
    get_qsize,
    init_pipeline,
    set_device,
    set_lora,
    set_model,
    set_scheduler,
)
from commands.frankenstein import frankenstein
from commands.help import print_help
from commands.management import (
    get_channel_messages,
    leave_guild,
    list_guild_channels,
    list_guild_members,
    list_guilds,
    purge,
    sync_commands_global,
)
from commands.music_player import (
    clear_queue,
    delete_all_music,
    delete_song,
    disconnect,
    init_vc,
    list_all_music,
    loop,
    pause,
    play_all,
    play_music,
    print_prev_queue,
    print_queue,
    replay,
    replay_all,
    set_volume,
    shuffle_music,
    skip,
    stop,
)
from commands.psd_analyzer import (
    export_rulebook_to_file,
    export_stats_to_file,
    init_psd,
    list_orphans,
    manual_metadata_entry,
    update_stats,
    mass_replace_author,
)
from commands.query_card import (
    manage_alias,
    init_query,
    query_ability,
    query_ability_num_occur,
    query_name,
    query_rulebook,
    set_match_ratio,
    set_repository,
)
from commands.update_bot import git_pull, git_push, restart_bot, update_bot
from commands.utils import (
    command_queue,
    dispatch_queue,
    edit_queue,
    slash_registry,
    queue_command,
    to_thread,
    to_threadpool,
    perms_check,
)

# Intents permissions
intents = Intents.default()
intents.message_content = True
client = Client(intents=intents)
tree = CommandTree(client)


async def sync_guild_commands(guild):
    """Sync commands to a specific guild."""
    tree.copy_global_to(guild=guild)
    await tree.sync(guild=guild)
    print(f"Synced commands to guild: {guild.name} ({guild.id})")


# Events
@client.event
async def on_ready():
    print(f'We have logged in as {client.user}')

    for cmd in tree.get_commands():
        slash_registry[cmd.name] = cmd.callback
    print(f"Registered {len(slash_registry)} slash commands for reminder system.")

    for guild in client.guilds: # Sync commands to each guild
        create_task(sync_guild_commands(guild))

    queue_command(init_reminder)
    queue_command(init_psd)
    queue_command(init_query)
    queue_command(init_pipeline)

    process_command_queue.start()
    process_dispatch_queue.start()
    process_edit_queue.start()
    print("Bot initialization complete")


# Slash Commands
####################
# GENERAL COMMANDS #
####################
print_help = to_thread(print_help)
@tree.command(name="help", description="Get bot help.")
@app_commands.describe(
    help_type="The type of command you want help for.",
)
@app_commands.choices(help_type=[
    Choice(name="text", value="text"),
    Choice(name="card", value="card"),
    Choice(name="ai", value="ai"),
    Choice(name="music", value="music"),
    Choice(name="stats", value="stats"),
    Choice(name="general", value="general"),
])
async def slash_print_help(interaction: Interaction,
                           help_type: Choice[str]):
    await print_help(interaction, help_type.value)


@tree.command(name="restart_bot", description="Restart the bot.")
async def slash_restart_bot(interaction: Interaction):
    process_command_queue.stop()
    process_dispatch_queue.stop()
    process_edit_queue.stop()
    await interaction.response.send_message("Restarting bot!")
    restart_bot()


@tree.command(name="stop_bot", description="Stop the bot.")
async def slash_stop_bot(interaction: Interaction):
    if perms_check(interaction):
        await interaction.response.send_message("You do not have permission.")
        return
    process_command_queue.stop()
    process_dispatch_queue.stop()
    process_edit_queue.stop()
    await interaction.response.send_message("Stopping bot!")
    await client.close()
    exit(0)


git_pull = to_thread(git_pull)
@tree.command(name="pull", description="Does a hard reset, than a git pull, and reports status.")
async def slash_pull(interaction: Interaction):
    if perms_check(interaction):
        await interaction.response.send_message("You do not have permission.")
        return
    await interaction.response.send_message("Doing a git pull!")
    await git_pull(interaction)


git_push = to_thread(git_push)
@tree.command(name="push", description="Does a git push, and reports status.")
async def slash_push(interaction: Interaction):
    if perms_check(interaction):
        await interaction.response.send_message("You do not have permission.")
        return
    await interaction.response.send_message(f"Doing a git push!")
    await git_push(interaction)


update_bot = to_thread(update_bot)
@tree.command(name="update", description="Does a push, then pull, then restarts.")
async def slash_update(interaction: Interaction):
    if perms_check(interaction):
        await interaction.response.send_message("You do not have permission.")
        return
    process_command_queue.stop()
    process_dispatch_queue.stop()
    process_edit_queue.stop()
    await interaction.response.send_message(f"Doing a complete update of the bot!")
    await update_bot(interaction)


####################
#   TEXT COMMANDS  #
####################
set_reminder = to_thread(set_reminder)
@tree.command(name="set_reminder", description="Set a reminder.")
@app_commands.describe(
    msg="What message to send.",
    when="When to send a message in PST (ex. 13:00 or 1PM).",
    offset="Delay before first run (s/m/h/d/w). Default = none.",
    frequency="How often to repeat (s/m/h/d/w). Default = never.",
    command="Optional bot command to run (e.g. /play_all, /update_stats True True False).",
)
async def slash_set_reminder(interaction: Interaction,
                             msg: Optional[str] = "",
                             when: Optional[str] = "",
                             offset: Optional[str] = "",
                             frequency: Optional[str] = "",
                             command: Optional[str] = ""):
    await set_reminder(interaction, msg, when, offset, frequency, command)


list_reminders = to_thread(list_reminders)
@tree.command(name="list_reminders", description="List all reminders in this channel/server.")
@app_commands.describe(
    all="List all reminders in the server instead of just this channel.",
    hidden="Should the list only be sent to you? Default is false.",
)
async def slash_list_reminders(interaction: Interaction,
                               all: Optional[bool] = False,
                               hidden: Optional[bool] = False):
    await list_reminders(interaction, all, hidden)


remove_reminder = to_thread(remove_reminder)
@tree.command(name="remove_reminder", description="Remove a reminder by its ID.")
@app_commands.describe(
    reminder_id="The ID of the reminder to remove.",
)
async def slash_remove_reminder(interaction: Interaction,
                                reminder_id: str):
    await remove_reminder(interaction, reminder_id)


####################
#   CARD COMMANDS  #
####################
query_name = to_thread(query_name)
@tree.command(name="query", description="Query a card by name.")
@app_commands.describe(
    query='The name of the card you want to search for',
)
async def slash_query_name(interaction: Interaction,
                           query: str):
    await query_name(interaction, query)


query_ability_num_occur = to_thread(query_ability_num_occur)
query_ability = to_thread(query_ability)
@tree.command(name="query_ability", description="Query a card by ability.")
@app_commands.describe(
    query='The ability text you want to search for.',
    limit='The number of cards to output before stopping.',
    howmany='Instead of returning card images, return the number of matches.',
)
async def slash_query_ability(interaction: Interaction,
                              query: str,
                              limit: Optional[int] = -1,
                              howmany: Optional[bool] = False):
    if howmany:
        await query_ability_num_occur(interaction, query)
    else:
        await query_ability(interaction, query, limit)


query_rulebook = to_thread(query_rulebook)
@tree.command(name="query_rulebook", description="Search the rulebook for specific text.")
@app_commands.describe(
    search_text='Text to search for in the rulebook',
    limit='Maximum number of results to show (default: all)',
)
async def slash_query_rulebook(
    interaction: Interaction,
    search_text: str,
    limit: int = -1):
    await interaction.response.defer()
    await query_rulebook(interaction, search_text, limit)


set_match_ratio = to_thread(set_match_ratio)
@tree.command(name="set_ratio", description="View/set tolerance for unexact matches when searching.")
@app_commands.describe(
    value='If empty, posts the current tolerance. Otherwise sets it to this.',
)
async def slash_set_ratio(interaction: Interaction,
                          value: Optional[float]):
    await set_match_ratio(interaction, value)


set_repository = to_thread(set_repository)
@tree.command(name="set_repo", description="View/set the repo that the bot looks at for cards.")
@app_commands.describe(
    new_repo='If empty, posts the current repository. Otherwise sets it to this.',
)
async def slash_set_ratio(interaction: Interaction,
                          new_repo: Optional[str]):
    await set_repository(interaction, new_repo)


manage_alias = to_thread(manage_alias)
@tree.command(name="alias", description="View, create, or delete card aliases.")
@app_commands.describe(
    alias='The alias name. Leave empty to view all aliases, or provide without card name to delete.',
    card='The card to alias. Omit to delete the alias.',
)
async def slash_alias(interaction: Interaction,
                      alias: Optional[str] = None,
                      card: Optional[str] = None):
    await manage_alias(interaction, alias, card)


frankenstein = to_thread(frankenstein)
@tree.command(name="frankenstein", description="Tie a bunch of cards together.")
@app_commands.describe(
    cards='The cards that you want to frankenstein together, separated by commas.',
    blend_mode='How to blend: slice, or average. Default = slice.',
)
@app_commands.choices(blend_mode=[
    Choice(name="slice", value="slice"),
    Choice(name="average", value="average"),
])
async def slash_frankenstein(
    interaction: Interaction,
    cards: str,
    blend_mode: Optional[Choice[str]]):
    await interaction.response.defer()
    await frankenstein(interaction, cards, blend_mode.value if blend_mode else "slice")


####################
#   STAT COMMANDS  #
####################
update_stats = to_thread(update_stats)
@tree.command(name="update_stats", description="Manual call to update card stats.")
@app_commands.describe(
    output_problematic_cards='Should we output cards with consistency errors?',
    use_local_repo='Should we use the local cloned folder for updates?',
    use_local_timestamp='Should we use the local files modified time for timestamping?',
)
async def slash_update_stats(interaction: Interaction,
                             output_problematic_cards: Optional[bool] = True,
                             use_local_repo: Optional[bool] = True,
                             use_local_timestamp: Optional[bool] = True):
    await interaction.response.defer()
    await update_stats(interaction, output_problematic_cards, use_local_repo, use_local_timestamp)


export_stats_to_file = to_thread(export_stats_to_file)
@tree.command(name="export_abilities", description="Exports all abilities as a text file.")
@app_commands.describe(
    only_ability='Should we only output the ability of cards?',
    as_csv='Should we output the file as a csv?',
)
async def slash_export_abilities(interaction: Interaction,
                                 only_ability: Optional[bool] = False,
                                 as_csv: Optional[bool] = True):
    await interaction.response.defer()
    await export_stats_to_file(interaction, only_ability, as_csv)


export_rulebook_to_file = to_thread(export_rulebook_to_file)
@tree.command(name="export_rulebook", description="Exports the rulebook as a text file.")
async def slash_export_rulebook(interaction: Interaction):
    await interaction.response.defer()
    await export_rulebook_to_file(interaction)


manual_metadata_entry = to_thread(manual_metadata_entry)
@tree.command(name="update_metadata", description="Edit metadata for a card.")
@app_commands.describe(
    query='The card to edit the metadata for.',
    del_entry='Delete the entire metadata entry for this card.',
    author='The creator of the card.',
    ability='Card ability text.',
    stars='Star count (e.g., 1-7).',
    subtype='Card subtype.',
    series='Card series.',
    hp='HP stat value.',
    defense='Defense stat value.',
    attack='Attack stat value.',
    speed='Speed stat value.',
    card_type='Card type (e.g., creature, item, field).',
    types='Comma-separated list of types (e.g., "fire,water").',
)
async def slash_update_metadata(
    interaction: Interaction,
    query: str,
    del_entry: Optional[bool] = False,
    author: Optional[str] = None,
    ability: Optional[str] = None,
    stars: Optional[int] = None,
    subtype: Optional[str] = None,
    series: Optional[str] = None,
    hp: Optional[int] = None,
    defense: Optional[int] = None,
    attack: Optional[int] = None,
    speed: Optional[int] = None,
    card_type: Optional[str] = None,
    types: Optional[str] = None):
    await interaction.response.defer()
    await manual_metadata_entry(
        interaction,
        query,
        del_entry,
        author,
        ability,
        stars,
        subtype,
        series,
        hp,
        defense,
        attack,
        speed,
        card_type,
        types,
    )


mass_replace_author = to_thread(mass_replace_author)
@tree.command(name="replace_author", description="Replace all instances of one author with another.")
@app_commands.describe(
    author1='The author being replaced.',
    author2='The author to change author1 to.',
)
async def slash_replace_author(interaction: Interaction,
                               author1: str,
                               author2: str):
    await interaction.response.defer()
    await mass_replace_author(interaction, author1, author2)


list_orphans = to_thread(list_orphans)
@tree.command(name="list_orphans", description="Output all cards without a listed author.")
async def slash_list_orphans(interaction: Interaction):
    await interaction.response.defer()
    await list_orphans(interaction)


####################
#   AI COMMANDS    #
####################
diffusion = to_thread(diffusion)
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
    seed='Used to control output, same seed = same output. Default is randomized.',
)
async def slash_ai(interaction: Interaction,
                   prompt: str,
                   image: Optional[Attachment],
                   mask_image: Optional[Attachment],
                   url: Optional[str],
                   mask_url: Optional[str],
                   steps: Optional[int] = 50,
                   height: Optional[int] = 512,
                   width: Optional[int] = 512,
                   resize: Optional[float] = 1.0,
                   cfg: Optional[float] = 7.0,
                   strength: Optional[float] = 0.8,
                   seed: Optional[str] = None):
    if seed:
        # ints are limited to 15 digits or less by discord, so I need to take it as a str and then convert it to an int
        try:
            seed = int(seed)
        except ValueError:
            await interaction.response.send_message("Seed must be an integer.")
            return
    await interaction.response.defer()
    await diffusion(interaction, prompt, image, mask_image, url, mask_url, steps, height, width, resize, cfg, strength, seed)


get_qsize = to_thread(get_qsize)
@tree.command(name="ai_queue", description="Returns the current ai queue size and the content of each queued request.")
async def slash_ai_queue(interaction: Interaction):
    await get_qsize(interaction)


set_scheduler = to_thread(set_scheduler)
@tree.command(name="set_scheduler", description="Set the scheduler.")
@app_commands.describe(
    scheduler='The scheduler you want. Leaving blank tells you the current scheduler.',
)
async def slash_set_schedule(interaction: Interaction,
                             scheduler: Optional[str]):
    await set_scheduler(interaction, scheduler)


set_device = to_thread(set_device)
@tree.command(name="set_device", description="Set the device.")
@app_commands.describe(
    device='The device number you want to set. Leaving blank tells you the current device.',
)
async def slash_set_device(interaction: Interaction,
                           device: Optional[int]):
    await set_device(interaction, device)


set_model = to_thread(set_model)
@tree.command(name="set_model", description="Set the model.")
@app_commands.describe(
    model='The model you want to set. Leaving blank tells you the current model.',
)
async def slash_set_model(interaction: Interaction,
                          model: Optional[str]):
    await set_model(interaction, model)


set_lora = to_thread(set_lora)
@tree.command(name="set_lora", description="Set the lora.")
@app_commands.describe(
    lora='The lora you want to set. Leaving blank tells you the current lora.',
)
async def slash_set_lora(interaction: Interaction,
                         lora: Optional[str]):
    await set_lora(interaction, lora)


####################
#  MUSIC COMMANDS  #
####################
init_vc = to_thread(init_vc)
play_music = to_thread(play_music)
@tree.command(name="play", description="Adds a song to the back of the queue, either a local file or a url.")
@app_commands.describe(
    song='The song you want to play.',
    play_next='True if you want to make this song play next. False by default.',
)
async def slash_play(interaction: Interaction,
                     song: str,
                     play_next: Optional[bool] = False):
    await interaction.response.defer()
    await init_vc(interaction)
    await play_music(interaction, song, play_next)


play_all = to_thread(play_all)
@tree.command(name="play_all", description="Play all the songs in the local music directory.")
async def slash_play_all(interaction: Interaction):
    await interaction.response.defer()
    await init_vc(interaction)
    await play_all(interaction)


replay = to_thread(replay)
@tree.command(name="replay", description="Replay a song.")
@app_commands.describe(
    song='0 is the current song, 1 is the previous song, 2 is 2 songs ago, etc. 0 by default.',
)
async def slash_replay(interaction: Interaction,
                       song: Optional[int] = 0):
    await replay(interaction, song)


replay_all = to_thread(replay_all)
@tree.command(name="replay_all", description="Replay all songs.")
async def slash_replay_all(interaction: Interaction):
    await replay_all(interaction)


skip = to_thread(skip)
@tree.command(name="skip", description="Skip current song.")
async def slash_skip(interaction: Interaction):
    await skip(interaction)


loop = to_thread(loop)
@tree.command(name="loop", description="Loop current song.")
async def slash_loop(interaction: Interaction):
    await loop(interaction)


list_all_music = to_thread(list_all_music)
@tree.command(name="list", description="List all songs in the current directory.")
async def slash_list(interaction: Interaction):
    await list_all_music(interaction)


set_volume = to_thread(set_volume)
@tree.command(name="volume", description="Changes the volume.")
@app_commands.describe(
    multiplier='The multiplier of the volume. 0.5 = half as loud, 1 = default, 2 = twice as loud, etc.',
)
async def slash_volume(interaction: Interaction,
                       multiplier: float = 1.0):
    await set_volume(interaction, multiplier)


shuffle_music = to_thread(shuffle_music)
@tree.command(name="shuffle", description="Shuffles the queue.")
async def slash_shuffle(interaction: Interaction):
    await shuffle_music(interaction)


print_queue = to_thread(print_queue)
@tree.command(name="queue", description="Shows the queue.")
async def slash_queue(interaction: Interaction):
    await print_queue(interaction)


print_prev_queue = to_thread(print_prev_queue)
@tree.command(name="prev_queue", description="Shows the previously queued songs.")
async def slash_prev_queue(interaction: Interaction):
    await print_prev_queue(interaction)


clear_queue = to_thread(clear_queue)
@tree.command(name="clear", description="Clears the music queue.")
async def slash_clear(interaction: Interaction):
    await clear_queue(interaction)


pause = to_thread(pause)
@tree.command(name="pause", description="Pauses/unpauses the current song.")
async def slash_pause(interaction: Interaction):
    await pause(interaction)


stop = to_thread(stop)
@tree.command(name="stop", description="Stops playing the current song and clears the queue.")
async def slash_stop(interaction: Interaction):
    await stop(interaction)


disconnect = to_thread(disconnect)
@tree.command(name="disconnect", description="Disconnects the bot and clears the queue.")
async def slash_disconnect(interaction: Interaction):
    await disconnect(interaction)


delete_song = to_thread(delete_song)
@tree.command(name="delete_song", description="Delete a song/folder.")
@app_commands.describe(
    song_name='The name of the song or folder to delete.',
)
async def slash_delete_song(interaction: Interaction,
                            song_name: str):
    await delete_song(interaction, song_name)


delete_all_music = to_thread(delete_all_music)
@tree.command(name="delete_all_music", description="Delete all music under the music directory.")
async def slash_delete_song(interaction: Interaction):
    await delete_all_music(interaction)


#########################
#  MANAGEMENT COMMANDS  #
#########################
purge = to_thread(purge)
@tree.command(name="purge", description="Deletes messages.")
@app_commands.describe(
    user='The user whose messages to delete. Default is the bot\'s own messages.',
    num='The number of messages to delete. Default = 100.',
    bulk='Enable bulk delete. Deletes faster but needs Manage Messages permission. Default = false.',
)
async def slash_purge(
    interaction: Interaction,
    user: Optional[Member],
    num: Optional[int] = 100,
    bulk: Optional[bool] = False):
    await interaction.response.defer(ephemeral=True, thinking=False)
    await purge(interaction, num, user if user else client.user.id, bulk)


list_guilds = to_thread(list_guilds)
@tree.command(name="list_guilds", description="Lists all guilds the bot is currently a member of (admin only).")
async def slash_list_guilds(interaction: Interaction):
    if perms_check(interaction):
        await interaction.response.send_message("You do not have permission.")
        return
    await list_guilds(interaction)


leave_guild = to_thread(leave_guild)
@tree.command(name="leave_guild", description="Force the bot to leave a specified guild (admin only).")
@app_commands.describe(
    guild_id="The ID of the guild (server) you want the bot to leave.",
)
async def slash_leave_guild(
    interaction: Interaction,
    guild_id: str):
    if not guild_id.isdigit():
        await interaction.response.send_message("Invalid guild ID (must be a number).")
        return
    if perms_check(interaction):
        await interaction.response.send_message("You do not have permission.")
        return
    await leave_guild(interaction, guild_id)


sync_commands_global = to_thread(sync_commands_global)
@tree.command(name="sync_global", description="Sync all slash commands globally (admin only).")
async def slash_sync_global(interaction: Interaction):
    if perms_check(interaction):
        await interaction.response.send_message("You do not have permission.")
        return
    await sync_commands_global(interaction, tree, client)


list_guild_members = to_thread(list_guild_members)
@tree.command(name="list_guild_members", description="Lists all members in a guild by its ID (admin only).")
@app_commands.describe(
    guild_id="The ID of the guild (server) whose members you want to list.",
)
async def slash_list_guild_members(
    interaction: Interaction,
    guild_id: str):
    if not guild_id.isdigit():
        await interaction.response.send_message("Invalid guild ID (must be a number).")
        return
    if perms_check(interaction):
        await interaction.response.send_message("You do not have permission.")
        return
    await list_guild_members(interaction, guild_id)


list_guild_channels = to_thread(list_guild_channels)
@tree.command(name="list_guild_channels", description="Lists all channels in a guild by its ID (admin only).")
@app_commands.describe(
    guild_id="The ID of the guild (server) whose channels you want to list.",
)
async def slash_list_guild_channels(
    interaction: Interaction,
    guild_id: str):
    if not guild_id.isdigit():
        await interaction.response.send_message("Invalid guild ID (must be a number).")
        return
    if perms_check(interaction):
        await interaction.response.send_message("You do not have permission.")
        return
    await list_guild_channels(interaction, guild_id)


get_channel_messages = to_thread(get_channel_messages)
@tree.command(name="get_channel_messages", description="Get the last X messages from a specific channel in a guild (admin only).")
@app_commands.describe(
    guild_id="The ID of the guild (server).",
    channel_id="The ID of the channel to read from.",
    limit="How many recent messages to fetch (default 50, -1 means infinite).",
)
async def slash_get_channel_messages(
    interaction: Interaction,
    guild_id: str,
    channel_id: str,
    limit: Optional[int] = 50):
    if not guild_id.isdigit():
        await interaction.response.send_message("Invalid guild ID (must be a number).")
        return
    if perms_check(interaction):
        await interaction.response.send_message("You do not have permission.")
        return
    await get_channel_messages(interaction, guild_id, channel_id, limit)


########################
# Queue Handling Loops #
########################
async def run_command(func, args, kwargs):
    """Execute a command."""
    if kwargs.get("executor"):
        await to_threadpool(func, kwargs.pop("executor"))(*args, **kwargs)
    else:
        await to_thread(func)(*args, **kwargs)


# --- COMMAND QUEUE ---
@tasks.loop(seconds=1)
async def process_command_queue():
    queue_command(check_reminder)
    while command_queue:
        (args, kwargs), func = command_queue.popleft()
        create_task(run_command(func, args, kwargs))


# --- DISPATCH QUEUE (messages + files + embeds) ---
@tasks.loop(seconds=1)
async def process_dispatch_queue():
    while dispatch_queue:
        interaction, send_kwargs = dispatch_queue.popleft()
        try:
            if not interaction.response.is_done():
                await interaction.response.send_message(**send_kwargs)
            else:
                if "attachments" in send_kwargs:
                    if "files" not in send_kwargs:
                        send_kwargs["files"] = send_kwargs.pop("attachments")
                    else:
                        send_kwargs["files"].extend(send_kwargs.pop("attachments"))
                await interaction.followup.send(**send_kwargs)
        except (HTTPException, AttributeError):
            send_kwargs.pop("ephemeral", None)
            if "attachments" in send_kwargs:
                if "files" not in send_kwargs:
                    send_kwargs["files"] = send_kwargs.pop("attachments")
                else:
                    send_kwargs["files"].extend(send_kwargs.pop("attachments"))
            # fallback to sending as a new message
            await client.get_channel(interaction.channel_id).send(**send_kwargs)


# --- EDIT MESSAGES ---
@tasks.loop(seconds=1)
async def process_edit_queue():
    while edit_queue:
        interaction, edit_kwargs = edit_queue.popleft()
        try:
            await interaction.edit_original_response(**edit_kwargs)
        except (HTTPException, AttributeError):
            if "attachments" in edit_kwargs:
                if "files" not in edit_kwargs:
                    edit_kwargs["files"] = edit_kwargs.pop("attachments")
                else:
                    edit_kwargs["files"].extend(edit_kwargs.pop("attachments"))
            # fallback to sending as a new message
            await client.get_channel(interaction.channel_id).send(**edit_kwargs)


# ========================
# RUN BOT
# ========================
if __name__ == "__main__":
    try:
        client.run(TOKEN)
    except KeyboardInterrupt:
        print("Bot shutting down via keyboard interrupt")
    except Exception as e:
        print(f"Fatal error: {e}")
        exit(1)
