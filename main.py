from asyncio import create_task
from sys import exit
from discord import ButtonStyle, Client, File, Intents, Interaction, Member, app_commands
from discord.ui import Button, View
from global_config import SITE_URL
from discord.app_commands import Choice, CommandTree
from discord.errors import HTTPException
from discord.ext import tasks
from traceback import format_exc
from typing import Optional

from config import TOKEN
from commands.analytics import (
    bind_reminder_client,
    check_reminder,
    init_reminder,
    list_reminders,
    remove_reminder,
    set_reminder,
)
from commands.frankenstein import frankenstein
from commands.github_api import print_token_status
from commands.help import print_help
from commands.management import (
    get_channel_messages,
    leave_guild,
    list_guild_channels,
    list_guilds,
    purge,
    sync_commands_global,
    add_reactions,
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
    view_old_metadata,
    update_stats,
    mass_replace_field,
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
from commands.repo_history import refresh as refresh_history
from commands.tts_save import update_tts_save
from commands.type_emojis import init_type_emojis, sync_type_emojis
from commands.update_bot import git_pull, git_push, restart_bot, update_bot
from commands.utils import (
    command_queue,
    dispatch_queue,
    edit_queue,
    slash_registry,
    queue_command,
    to_thread,
    to_threadpool,
    lacks_admin,
    split_long_message,
)

print_token_status()

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
    queue_command(refresh_history)                      # commit-pinned card image links
    queue_command(init_type_emojis, client)

    process_reminders.start()
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
    Choice(name="music", value="music"),
    Choice(name="stats", value="stats"),
    Choice(name="general", value="general"),
])
async def slash_print_help(interaction: Interaction,
                           help_type: Choice[str]):
    await print_help(interaction, help_type.value)


@tree.command(name="restart_bot", description="Restart the bot.")
async def slash_restart_bot(interaction: Interaction):
    process_reminders.stop()
    process_command_queue.stop()
    process_dispatch_queue.stop()
    process_edit_queue.stop()
    await interaction.response.send_message("Restarting bot!")
    restart_bot()


@tree.command(name="stop_bot", description="Stop the bot (admin only).")
async def slash_stop_bot(interaction: Interaction):
    if lacks_admin(interaction):
        await interaction.response.send_message("You do not have permission.")
        return
    process_reminders.stop()
    process_command_queue.stop()
    process_dispatch_queue.stop()
    process_edit_queue.stop()
    await interaction.response.send_message("Stopping bot!")
    await client.close()
    exit(0)


git_pull = to_thread(git_pull)
@tree.command(name="pull", description="Fetches and merges the latest code (keeps local pickles), and reports status (admin only).")
async def slash_pull(interaction: Interaction):
    if lacks_admin(interaction):
        await interaction.response.send_message("You do not have permission.")
        return
    await interaction.response.send_message("Doing a git pull!")
    await git_pull(interaction)


git_push = to_thread(git_push)
@tree.command(name="push", description="Commits the pickles, rebases onto new code, pushes, and reports status (admin only).")
async def slash_push(interaction: Interaction):
    if lacks_admin(interaction):
        await interaction.response.send_message("You do not have permission.")
        return
    await interaction.response.send_message("Doing a git push!")
    await git_push(interaction)


update_bot = to_thread(update_bot)
@tree.command(name="update", description="Push pickles, pull latest code, and restart the bot (admin only).")
@app_commands.describe(
    force_reset="Skip the push and pull: hard reset to remote, losing local pickles (dangerous!). Default = False."
)
async def slash_update(interaction: Interaction, force_reset: Optional[bool] = False):
    if lacks_admin(interaction):
        await interaction.response.send_message("You do not have permission.")
        return
    process_reminders.stop()
    process_command_queue.stop()
    process_dispatch_queue.stop()
    process_edit_queue.stop()
    await interaction.response.send_message("Doing a complete update of the bot!")
    await update_bot(interaction, force_reset)


####################
#   TEXT COMMANDS  #
####################
bind_reminder_client(client)  # lets the scheduler skip reminders whose channel is gone
set_reminder = to_thread(set_reminder)
@tree.command(name="set_reminder", description="Set a reminder that posts a message and/or runs a bot command (admin only).")
@app_commands.describe(
    msg="What message to send.",
    when="Pacific time: 13:00, 1PM, sun 3AM, tomorrow 9AM, 2026-12-25 9AM. Past times today fire now.",
    offset="Delay added to the first run (s/m/h/d/w, e.g. 30m or 1d). Default = none.",
    frequency="How often to repeat (s/m/h/d/w, e.g. 1d, 1w or 1d12h). Default = never.",
    command="Bot command to run with your permissions, options as name:value (e.g. /sync_emojis refresh:True).",
)
async def slash_set_reminder(interaction: Interaction,
                             msg: Optional[str] = "",
                             when: Optional[str] = "",
                             offset: Optional[str] = "",
                             frequency: Optional[str] = "",
                             command: Optional[str] = ""):
    if lacks_admin(interaction):
        await interaction.response.send_message("You do not have permission.")
        return
    await set_reminder(interaction, msg, when, offset, frequency, command)


list_reminders = to_thread(list_reminders)
@tree.command(name="list_reminders", description="List all reminders in this channel/server.")
@app_commands.describe(
    all="List all reminders in the server instead of just this channel. Default is false.",
    hidden="Should the list only be sent to you? Default is false.",
)
async def slash_list_reminders(interaction: Interaction,
                               all: Optional[bool] = False,
                               hidden: Optional[bool] = False):
    await list_reminders(interaction, all, hidden)


remove_reminder = to_thread(remove_reminder)
@tree.command(name="remove_reminder", description="Remove a reminder by its ID (admins, or the reminder's creator).")
@app_commands.describe(
    reminder_id="The 8-character ID shown by /list_reminders.",
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


@tree.command(name="website", description="Link to the card browser website.")
async def slash_website(interaction: Interaction):
    await interaction.response.send_message(f"**Card browser:** {SITE_URL}/\nEvery card, the rulebook, a deck builder and stats. "
                                            "Editors fix card data there; the bot picks it up on /update_stats.")


def _check_repo_fonts(interaction: Interaction):
    from commands.font_check import check_repo_fonts    # lazy: fontTools loads only when /fonts runs
    check_repo_fonts(interaction)
check_repo_fonts = to_thread(_check_repo_fonts)
@tree.command(name="fonts", description="Lists the fonts card PSDs ask for that MDW/Fonts in the card repo does not supply.")
async def slash_fonts(interaction: Interaction):
    await interaction.response.defer()
    await check_repo_fonts(interaction)


query_ability_num_occur = to_thread(query_ability_num_occur)
query_ability = to_thread(query_ability)
@tree.command(name="query_ability", description="Query a card by ability.")
@app_commands.describe(
    query='The ability text you want to search for.',
    limit='The number of cards to output before stopping. Default = all.',
    howmany='Instead of returning the cards, return the number of matches. Default = False.',
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
async def slash_set_repo(interaction: Interaction,
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
@tree.command(name="update_stats", description="Updates the card database from GitHub (re-parses changed PSDs).")
@app_commands.describe(
    output_problematic_cards='Should we output cards with consistency errors? Default = True.',
    force_update='Re-parse every card, even unchanged ones (downloads every PSD). Default = False.',
    cards='Card names to re-parse anyway, comma separated (e.g. after a parser fix).',
)
async def slash_update_stats(interaction: Interaction,
                             output_problematic_cards: Optional[bool] = True,
                             force_update: Optional[bool] = False,
                             cards: Optional[str] = ""):
    await interaction.response.defer()
    await update_stats(interaction, output_problematic_cards, force_update, cards=cards)


update_tts_save = to_thread(update_tts_save)
@tree.command(name="update_tts_save", description="Updates the Tabletop Simulator save on the card repo and posts its download link (admin only).")
async def slash_update_tts_save(interaction: Interaction):
    if lacks_admin(interaction):
        await interaction.response.send_message("You do not have permission.")
        return
    await interaction.response.defer()
    await update_tts_save(interaction)


export_stats_to_file = to_thread(export_stats_to_file)
@tree.command(name="export_cards", description="Exports all cards as a csv/txt.")
@app_commands.describe(
    only_ability='Text export only: just names, aliases and ability text. Default = False.',
    as_csv='Should we output the file as a csv? Default = True.',
)
async def slash_export_cards(interaction: Interaction,
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
    del_entry='Delete the entire metadata entry for this card. Default = False.',
    author='The creator of the card.',
    ability='Card ability text.',
    stars='Star count (e.g., 1-7).',
    subtype='Card subtype.',
    series='Card series.',
    subseries='A series within the series (e.g. Paper Mario in Mario). Database only, never in the PSD.',
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
    subseries: Optional[str] = None,
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
        subseries,
        hp,
        defense,
        attack,
        speed,
        card_type,
        types,
    )


view_old_metadata = to_thread(view_old_metadata)
@tree.command(name="view_old_metadata", description="View old metadata versions for a card.")
@app_commands.describe(
    query='The card to view old metadata for.',
    version='Version index: -1 for oldest, 0 for most recent old version, 1+ for older versions. Default = 0.'
)
async def slash_view_old_metadata(
    interaction: Interaction,
    query: str,
    version: int = 0):
    await interaction.response.defer()
    await view_old_metadata(interaction, query, version)


mass_replace_field = to_thread(mass_replace_field)
@tree.command(name="mass_replace", description="Replace all instances of one field value with another.")
@app_commands.describe(
    field='The field to modify (author, series, subseries, subtype, or card_type).',
    old_value='The value being replaced.',
    new_value='The new value to set.',
)
@app_commands.choices(field=[
    Choice(name="author", value="author"),
    Choice(name="series", value="series"),
    Choice(name="subseries", value="subseries"),
    Choice(name="subtype", value="subtype"),
    Choice(name="card_type", value="card_type"),
])
async def slash_replace_field(interaction: Interaction,
                              field: Choice[str],
                              old_value: str,
                              new_value: str):
    await interaction.response.defer()
    await mass_replace_field(interaction, field.value, old_value, new_value)


list_orphans = to_thread(list_orphans)
@tree.command(name="list_orphans", description="Output all cards without a listed author.")
async def slash_list_orphans(interaction: Interaction):
    await interaction.response.defer()
    await list_orphans(interaction)


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
    if not await init_vc(interaction):                  # not in a voice channel: replied, nothing queued
        return
    await play_music(interaction, song, play_next)


play_all = to_thread(play_all)
@tree.command(name="play_all", description="Play all the songs in the local music directory.")
async def slash_play_all(interaction: Interaction):
    await interaction.response.defer()
    if not await init_vc(interaction):                  # not in a voice channel: replied, nothing queued
        return
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
async def slash_delete_all_music(interaction: Interaction):
    await delete_all_music(interaction)


#########################
#  MANAGEMENT COMMANDS  #
#########################
purge = to_thread(purge)
@tree.command(name="purge", description="Deletes messages.")
@app_commands.describe(
    user='The user whose messages to delete. Default is the bot\'s own messages.',
    num='How many recent messages to search (-1 = all). Default = 100.',
    bulk='Enable bulk delete. Deletes faster but needs Manage Messages permission. Default = false.',
)
async def slash_purge(interaction: Interaction,
                      user: Optional[Member],
                      num: Optional[int] = 100,
                      bulk: Optional[bool] = False):
    await interaction.response.defer(ephemeral=True, thinking=False)
    await purge(interaction, num, user.id if user else client.user.id, bulk)


list_guilds = to_thread(list_guilds)
@tree.command(name="list_guilds", description="Lists all guilds the bot is currently a member of (admin only).")
async def slash_list_guilds(interaction: Interaction):
    if lacks_admin(interaction):
        await interaction.response.send_message("You do not have permission.")
        return
    await list_guilds(interaction)


leave_guild = to_thread(leave_guild)
@tree.command(name="leave_guild", description="Force the bot to leave a specified guild (admin only).")
@app_commands.describe(
    guild_id="The ID of the guild (server) you want the bot to leave.",
)
async def slash_leave_guild(interaction: Interaction, guild_id: str):
    if not guild_id.isdigit():
        await interaction.response.send_message("Invalid guild ID (must be a number).")
        return
    if lacks_admin(interaction):
        await interaction.response.send_message("You do not have permission.")
        return
    await leave_guild(interaction, guild_id)


sync_commands_global = to_thread(sync_commands_global)
@tree.command(name="sync_global", description="Sync all slash commands globally (admin only).")
async def slash_sync_global(interaction: Interaction):
    if lacks_admin(interaction):
        await interaction.response.send_message("You do not have permission.")
        return
    await sync_commands_global(interaction, tree, client)


sync_type_emojis = to_thread(sync_type_emojis)
@tree.command(name="sync_emojis", description="Uploads missing and edited Types/ icons as the bot's emojis (admin only).")
@app_commands.describe(
    refresh="Re-upload every icon, even unchanged ones (default False).",
    prune="Delete bot emojis whose icon is no longer in Types/ (default False).",
)
async def slash_sync_emojis(interaction: Interaction,
                            refresh: Optional[bool] = False,
                            prune: Optional[bool] = False):
    if lacks_admin(interaction):
        await interaction.response.send_message("You do not have permission.")
        return
    await interaction.response.send_message("Syncing type emojis...")
    await sync_type_emojis(interaction, client, refresh, prune)


list_guild_channels = to_thread(list_guild_channels)
@tree.command(name="list_guild_channels", description="Lists all channels in a guild by its ID (admin only).")
@app_commands.describe(
    guild_id="The ID of the guild (server) whose channels you want to list.",
)
async def slash_list_guild_channels(interaction: Interaction, guild_id: str):
    if not guild_id.isdigit():
        await interaction.response.send_message("Invalid guild ID (must be a number).")
        return
    if lacks_admin(interaction):
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
async def slash_get_channel_messages(interaction: Interaction,
                                     guild_id: str,
                                     channel_id: str,
                                     limit: Optional[int] = 50):
    if not guild_id.isdigit():
        await interaction.response.send_message("Invalid guild ID (must be a number).")
        return
    if lacks_admin(interaction):
        await interaction.response.send_message("You do not have permission.")
        return
    await get_channel_messages(interaction, guild_id, channel_id, limit)


@tree.command(name="react", description="Add reactions to a message by ID or the last message in the channel.")
@app_commands.describe(
    reactions="Space-separated emojis (e.g. 👍 👎 or <:custom:1234567890>)",
    message_id="Optional message ID to react to. Defaults to last message in channel.",
)
async def slash_react(interaction: Interaction,
                      reactions: str,
                      message_id: Optional[str] = None):
    await interaction.response.defer(ephemeral=True, thinking=False)
    await add_reactions(interaction, reactions, message_id)


########################
# Queue Handling Loops #
########################
MAX_CONTENT = 2000                                      # Discord's message length limit
LAST_ONLY = ("file", "files", "attachments", "embed", "embeds", "view", "links")   # a split message's last chunk only
_running: set = set()                                   # queued commands in flight


async def run_command(func, args, kwargs):
    """Execute a command; a failure is logged with the function's name."""
    try:
        if kwargs.get("executor"):
            await to_threadpool(func, kwargs.pop("executor"))(*args, **kwargs)
        else:
            await to_thread(func)(*args, **kwargs)
    except Exception:
        print(f"Queued command {getattr(func, '__name__', func)} failed:\n{format_exc()}")


def _split_content(queue, interaction, kwargs: dict) -> bool:
    """Over-long content: put it back at the queue's front as Discord-sized messages. True if split."""
    content = kwargs.get("content")
    if not isinstance(content, str) or len(content) <= MAX_CONTENT:
        return False
    chunks = split_long_message(content)
    extra = {k: v for k, v in kwargs.items() if k not in LAST_ONLY}   # ephemeral, allowed_mentions... on every chunk
    items = [(interaction, {**extra, "content": c}) for c in chunks[:len(chunks) - 1]]
    items.append((interaction, {**kwargs, "content": chunks[len(chunks) - 1]}))   # files/embeds ride on the last
    for item in reversed(items):
        queue.appendleft(item)
    return True


def _attachments_to_files(kwargs: dict) -> None:
    """A new message takes files, not edit-style attachments."""
    if "attachments" in kwargs:
        if "files" not in kwargs:
            kwargs["files"] = kwargs.pop("attachments")
        else:
            kwargs["files"].extend(kwargs.pop("attachments"))


async def _fallback_send(interaction, kwargs: dict) -> bool:
    """Send as a plain channel message (the interaction failed or expired). False if the channel is gone."""
    kwargs.pop("ephemeral", None)
    _attachments_to_files(kwargs)
    for f in [kwargs.get("file")] + list(kwargs.get("files") or []):
        if isinstance(f, File):
            f.reset()                                   # the failed send read it to the end
    channel = client.get_channel(interaction.channel_id)
    if channel is None:
        print(f"Dropped a message: channel {interaction.channel_id} not found")
        return False
    await channel.send(**kwargs)
    return True


async def _dispatch(interaction, send_kwargs: dict) -> None:
    links = send_kwargs.pop("links", None)              # [(label, url)] -> link buttons (a View needs the event loop)
    if links:
        view = View()
        for label, url in links:
            view.add_item(Button(label=label, url=url, style=ButtonStyle.link))
        send_kwargs["view"] = view
    try:
        if not interaction.response.is_done():
            await interaction.response.send_message(**send_kwargs)
        else:
            _attachments_to_files(send_kwargs)
            await interaction.followup.send(**send_kwargs)
    except (HTTPException, AttributeError):
        await _fallback_send(interaction, send_kwargs)  # fallback to sending as a new message


async def _edit(interaction, edit_kwargs: dict) -> None:
    content = edit_kwargs.get("content")
    rest = []
    if isinstance(content, str) and len(content) > MAX_CONTENT:
        chunks = split_long_message(content)
        edit_kwargs["content"] = chunks[0]
        rest = chunks[1:]
    extra = {"allowed_mentions": edit_kwargs["allowed_mentions"]} if "allowed_mentions" in edit_kwargs else {}
    try:
        msg = await interaction.edit_original_response(**edit_kwargs)
        if getattr(getattr(msg, "flags", None), "ephemeral", False):
            extra["ephemeral"] = True                   # the rest stays as private as the original
    except (HTTPException, AttributeError):
        if not await _fallback_send(interaction, edit_kwargs):  # fallback to sending as a new message
            return                                      # no start, so no rest
    for c in rest:                                      # after the first chunk, to keep the order
        dispatch_queue.append((interaction, {**extra, "content": c}))


def _log_drop(kind: str, interaction, e: Exception) -> None:
    print(f"{kind}: dropped a message for channel {getattr(interaction, 'channel_id', '?')}: {type(e).__name__}: {e}")


# --- REMINDERS (1 s heartbeat; the drains below tick every 0.1 s) ---
@tasks.loop(seconds=1)
async def process_reminders():
    queue_command(check_reminder)


# --- COMMAND QUEUE ---
@tasks.loop(seconds=0.1)
async def process_command_queue():
    while command_queue:
        try:
            (args, kwargs), func = command_queue.popleft()
            task = create_task(run_command(func, args, kwargs))
            _running.add(task)                          # a strong reference until it finishes
            task.add_done_callback(_running.discard)
        except Exception as e:
            print(f"Command queue: dropped an item: {type(e).__name__}: {e}")


# --- DISPATCH QUEUE (messages + files + embeds) ---
@tasks.loop(seconds=0.1)
async def process_dispatch_queue():
    while dispatch_queue:
        interaction, send_kwargs = dispatch_queue.popleft()
        try:                                            # one bad message must not stop the loop (B5)
            if not _split_content(dispatch_queue, interaction, send_kwargs):
                await _dispatch(interaction, send_kwargs)
        except Exception as e:
            _log_drop("Dispatch", interaction, e)


# --- EDIT MESSAGES ---
@tasks.loop(seconds=0.1)
async def process_edit_queue():
    while edit_queue:
        interaction, edit_kwargs = edit_queue.popleft()
        try:
            await _edit(interaction, edit_kwargs)
        except Exception as e:
            _log_drop("Edit", interaction, e)


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
