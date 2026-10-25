from asyncio import create_task
from collections import deque
from sys import exit
from discord import ButtonStyle, Client, File, Intents, Interaction, InteractionType, Member, app_commands
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
    remove_reminder,
    set_reminder,
)
from commands.card_sync import fonts_text, on_use as card_data_used, update as update_stats
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
    view_old_metadata,
)
from commands.reminder_panel import open_panel
from commands.query_card import (
    card_repo,
    init_query,
    query_ability,
    query_ability_num_occur,
    query_name,
    query_rulebook,
    set_match_ratio,
    set_repository,
)
from commands.repo_history import refresh as refresh_history
from commands.type_emojis import init_type_emojis, sync_type_emojis
from commands.update_bot import git_pull, restart_bot, update_bot
from commands.utils import (
    command_queue,
    command_wake,
    dispatch_queue,
    dispatch_wake,
    edit_queue,
    edit_wake,
    slash_registry,
    queue_command,
    queue_long_message,
    to_thread,
    to_threadpool,
    lacks_admin,
    queue_any,
    split_long_message,
)

print_token_status()

# Intents permissions
intents = Intents.default()
intents.message_content = True
client = Client(intents=intents, guild_ready_timeout=0.25)   # our guilds arrive within ~50 ms; the default waits 2 s
tree = CommandTree(client)

# Global State
_started = False                                        # on_ready's one-time inits have run
_synced_guilds: set = set()                             # guild ids synced or syncing


async def sync_guild_commands(guild):
    """Sync commands to a specific guild; a failure lets the next on_ready retry it."""
    try:
        tree.copy_global_to(guild=guild)
        await tree.sync(guild=guild)
    except BaseException:
        _synced_guilds.discard(guild.id)
        raise
    print(f"Synced commands to guild: {guild.name} ({guild.id})")


# Events
@client.event
async def on_ready():
    """Login, and again after every failed RESUME: inits run once; a repeat syncs new guilds and revives failed loops."""
    global _started
    print(f'We have logged in as {client.user}')

    for guild in client.guilds: # Sync commands to each guild not synced yet
        if guild.id not in _synced_guilds:
            _synced_guilds.add(guild.id)                # before the task runs: a quick repeat skips it
            create_task(sync_guild_commands(guild))

    if _started:
        revived = [loop for loop in _loops() if loop.failed() and _start(loop)]   # stopped ones stay stopped (/update)
        print(f"Reconnected: inits skipped, revived {', '.join(loop.coro.__name__ for loop in revived) or 'no loops'}")
        return

    for cmd in tree.get_commands():
        slash_registry[cmd.name] = cmd.callback
    print(f"Registered {len(slash_registry)} slash commands for reminder system.")

    queue_command(init_reminder)
    queue_command(init_psd)
    queue_command(init_query)
    queue_command(refresh_history)                      # commit-pinned card image links
    queue_command(init_type_emojis, client)

    for loop in _loops():
        _start(loop)
    _started = True
    print("Bot initialization complete")


LOGGED_VALUES = ("query", "search_text", "cards", "as_of")   # options whose values the log shows


@client.event
async def on_interaction(interaction: Interaction):
    """
    One console line per slash command: its name, who, where, its options (values only for searches). A card command
    or an autocomplete (every one is a card name's) queues a look for new card data (card_sync.on_use: no wait).
    """
    if interaction.type == InteractionType.autocomplete:
        card_data_used()
        return
    if interaction.type != InteractionType.application_command:
        return
    data = interaction.data or {}
    card_data_used(data.get("name"))
    options = [f"{o.get('name')}={str(o.get('value'))[:100]!r}" if o.get("name") in LOGGED_VALUES else str(o.get("name"))
               for o in data.get("options") or []]
    print(f"Command /{data.get('name')} by {getattr(interaction.user, 'id', '?')} in {interaction.channel_id}"
          f"{': ' + ', '.join(options) if options else ''}")


async def card_names(interaction: Interaction, current: str) -> list:
    """Autocomplete: card names, printed titles, aliases and renamed cards' old names."""
    return [Choice(name=shown, value=value) for shown, value in card_repo.name_choices(current)]


async def card_list_names(interaction: Interaction, current: str) -> list:
    """Autocomplete for a comma-separated list: the last name typed."""
    head, comma, last = current.rpartition(",")
    head = f"{head}, " if comma else ""
    return [Choice(name=shown, value=head + value) for shown, value in card_repo.name_choices(last)
            if len(head + value) <= 100]


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
    _stop_loops()
    await interaction.response.send_message("Restarting bot!")
    restart_bot()


@tree.command(name="stop_bot", description="Stop the bot (admin only).")
async def slash_stop_bot(interaction: Interaction):
    if lacks_admin(interaction):
        await interaction.response.send_message("You do not have permission.")
        return
    _stop_loops()
    await interaction.response.send_message("Stopping bot!")
    await client.close()
    exit(0)


git_pull = to_thread(git_pull)
@tree.command(name="pull", description="Fetches and merges the latest code and reports status (admin only).")
async def slash_pull(interaction: Interaction):
    if lacks_admin(interaction):
        await interaction.response.send_message("You do not have permission.")
        return
    await interaction.response.send_message("Doing a git pull!")
    await git_pull(interaction)


update_bot = to_thread(update_bot)
@tree.command(name="update", description="Pull latest code and restart the bot (admin only).")
@app_commands.describe(
    force_reset="Skip the pull: hard reset to remote, losing local changes (dangerous!). Default = False."
)
async def slash_update(interaction: Interaction, force_reset: Optional[bool] = False):
    if lacks_admin(interaction):
        await interaction.response.send_message("You do not have permission.")
        return
    _stop_loops()
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


@tree.command(name="reminders", description="Browse reminders: reschedule, turn off/on, delete or add one (only you see it).")
async def slash_reminders(interaction: Interaction):
    await open_panel(interaction)                       # answers directly: a View and modals need the live interaction


remove_reminder = to_thread(remove_reminder)
@tree.command(name="remove_reminder", description="Remove a reminder by its ID (admins, or the reminder's creator).")
@app_commands.describe(
    reminder_id="The 8-character ID shown by /reminders.",
)
async def slash_remove_reminder(interaction: Interaction,
                                reminder_id: str):
    await remove_reminder(interaction, reminder_id)


####################
#   CARD COMMANDS  #
####################
query_name = to_thread(query_name)
@tree.command(name="query", description="Query a card by name, now or as it was on a date.")
@app_commands.describe(
    query='Card name, printed title, alias, or a renamed card\'s old name (suggested as you type).',
    as_of='YYYY-MM-DD (Pacific time): the card as it was that day, archived cards too. Default = now.',
)
@app_commands.autocomplete(query=card_names)
async def slash_query_name(interaction: Interaction,
                           query: str,
                           as_of: Optional[str] = None):
    await query_name(interaction, query, as_of)


@tree.command(name="website", description="Link to the card browser website.")
async def slash_website(interaction: Interaction):
    await interaction.response.send_message(f"**Card browser:** {SITE_URL}/\nEvery card, the rulebook, a deck builder and stats. "
                                            + "Editors fix card data there; the card sync applies it within minutes (PSD edits are baked in about 30).")


def _check_repo_fonts(interaction: Interaction):
    queue_long_message(interaction, fonts_text())
check_repo_fonts = to_thread(_check_repo_fonts)
@tree.command(name="fonts", description="Lists the fonts card PSDs ask for that MDW/Fonts in the card repo does not supply.")
async def slash_fonts(interaction: Interaction):
    await interaction.response.defer()
    await check_repo_fonts(interaction)


query_ability_num_occur = to_thread(query_ability_num_occur)
query_ability = to_thread(query_ability)
@tree.command(name="query_ability", description="Query a card by ability.")
@app_commands.describe(
    query='Text (names, aliases, ability) or conditions: hp>5, type==creature, !die. See /help card.',
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
        await interaction.response.defer()              # grids take a few seconds (pictures fetched once, then cached)
        await query_ability(interaction, query, limit)


query_rulebook = to_thread(query_rulebook)
@tree.command(name="query_rulebook", description="Search the rulebook: the lines that match, with their page names.")
@app_commands.describe(
    search_text='A word or phrase: the reply quotes the rulebook lines holding it.',
    limit='Maximum number of pages to show. Default = all.',
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
@tree.command(name="set_repo", description="View/set the repo that the bot looks at for cards (admin only).")
@app_commands.describe(
    new_repo='If empty, posts the current repository. Otherwise sets it to this.',
)
async def slash_set_repo(interaction: Interaction,
                         new_repo: Optional[str]):
    if lacks_admin(interaction):
        await interaction.response.send_message("You do not have permission.")
        return
    await set_repository(interaction, new_repo)


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
@app_commands.autocomplete(cards=card_list_names)
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
@tree.command(name="update_stats", description="Loads the card data the Card sync Action built.")
@app_commands.describe(
    output_problematic_cards='Should we output cards with consistency errors? Default = True.',
    force_update='Admin only: start a Card sync run re-parsing every card, then load it. Default = False.',
    cards='Admin only: start a Card sync run re-parsing these cards (comma separated), then load it.',
)
async def slash_update_stats(interaction: Interaction,
                             output_problematic_cards: Optional[bool] = True,
                             force_update: Optional[bool] = False,
                             cards: Optional[str] = ""):
    if (force_update or (cards or "").strip()) and lacks_admin(interaction):
        await interaction.response.send_message("You do not have permission.")   # a Card sync run costs API budget
        return
    await interaction.response.defer()
    await update_stats(interaction, output_problematic_cards, force_update, cards=cards)


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


view_old_metadata = to_thread(view_old_metadata)
@tree.command(name="view_old_metadata", description="View old metadata versions for a card.")
@app_commands.describe(
    query='The card to view old metadata for.',
    version='Version index: -1 for oldest, 0 for most recent old version, 1+ for older versions. Default = 0.'
)
@app_commands.autocomplete(query=card_names)
async def slash_view_old_metadata(
    interaction: Interaction,
    query: str,
    version: int = 0):
    await interaction.response.defer()
    await view_old_metadata(interaction, query, version)


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
@tree.command(name="delete_song", description="Delete a song/folder (admin only).")
@app_commands.describe(
    song_name='The name of the song or folder to delete.',
)
async def slash_delete_song(interaction: Interaction,
                            song_name: str):
    if lacks_admin(interaction):
        await interaction.response.send_message("You do not have permission.")
        return
    await delete_song(interaction, song_name)


delete_all_music = to_thread(delete_all_music)
@tree.command(name="delete_all_music", description="Delete all music under the music directory (admin only).")
async def slash_delete_all_music(interaction: Interaction):
    if lacks_admin(interaction):
        await interaction.response.send_message("You do not have permission.")
        return
    await delete_all_music(interaction)


#########################
#  MANAGEMENT COMMANDS  #
#########################
purge = to_thread(purge)
@tree.command(name="purge", description="Deletes messages (admin only).")
@app_commands.describe(
    user='The user whose messages to delete. Default is the bot\'s own messages.',
    num='How many recent messages to search (-1 = all). Default = 100.',
    bulk='Enable bulk delete. Deletes faster but needs Manage Messages permission. Default = false.',
)
async def slash_purge(interaction: Interaction,
                      user: Optional[Member],
                      num: Optional[int] = 100,
                      bulk: Optional[bool] = False):
    if lacks_admin(interaction):
        await interaction.response.send_message("You do not have permission.")
        return
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
        queue_any(interaction, **extra, content=c)


def _log_drop(kind: str, interaction, e: Exception) -> None:
    print(f"{kind}: dropped a message for channel {getattr(interaction, 'channel_id', '?')}: {type(e).__name__}: {e}")


# --- REMINDERS (1 s heartbeat; the drains below wake when something is queued, else every FALLBACK s) ---
@tasks.loop(seconds=1)
async def process_reminders():
    queue_command(check_reminder)


# --- COMMAND QUEUE ---
def _drain_commands() -> None:
    while command_queue:
        try:
            (args, kwargs), func = command_queue.popleft()
            task = create_task(run_command(func, args, kwargs))
            _running.add(task)                          # a strong reference until it finishes
            task.add_done_callback(_running.discard)
        except Exception as e:
            print(f"Command queue: dropped an item: {type(e).__name__}: {e}")


@tasks.loop(seconds=0)                                  # paced by the wait: at once on queue_command
async def process_command_queue():
    _drain_commands()
    await command_wake.wait()


# --- DISPATCH QUEUE (messages + files + embeds): one ordered lane per channel ---
_lanes: dict = {}                                       # channel id -> deque of (interaction, send_kwargs)
_lane_tasks: dict = {}                                  # channel id -> the task draining its lane
_sending = [True]                                       # False once the loop is stopped (/restart_bot, /update)


def _drain_dispatch() -> None:
    """Hand each message to its channel's lane: a big search holds back only its own channel (discord.py keeps
    Discord's rate limits, per route and global)."""
    while dispatch_queue:
        item = dispatch_queue.popleft()
        _lanes.setdefault(getattr(item[0], "channel_id", None), deque()).append(item)
    for channel, lane in _lanes.items():
        task = _lane_tasks.get(channel)
        if lane and (task is None or task.done()):
            _lane_tasks[channel] = create_task(_drain_lane(lane))


@tasks.loop(seconds=0)                                  # paced by the wait: at once on queue_message and the rest
async def process_dispatch_queue():
    _drain_dispatch()
    await dispatch_wake.wait()


@process_dispatch_queue.before_loop
async def _open_lanes():
    _sending[0] = True


@process_dispatch_queue.after_loop
async def _close_lanes():
    _sending[0] = False                                 # what is left stays unsent, as before the lanes


async def _drain_lane(lane: deque) -> None:
    """Send a channel's messages in order."""
    while lane and _sending[0]:
        interaction, send_kwargs = lane.popleft()
        try:                                            # one bad message must not stop the lane (B5)
            if not _split_content(lane, interaction, send_kwargs):
                await _dispatch(interaction, send_kwargs)
        except Exception as e:
            _log_drop("Dispatch", interaction, e)


# --- EDIT MESSAGES ---
async def _drain_edits() -> None:
    while edit_queue:
        interaction, edit_kwargs = edit_queue.popleft()
        try:
            await _edit(interaction, edit_kwargs)
        except Exception as e:
            _log_drop("Edit", interaction, e)


@tasks.loop(seconds=0)                                  # paced by the wait: at once on queue_edit
async def process_edit_queue():
    await _drain_edits()
    await edit_wake.wait()


def _loops() -> tuple:
    """The four loops on_ready starts."""
    return (process_reminders, process_command_queue, process_dispatch_queue, process_edit_queue)


def _stop_loops() -> None:
    """Stop the four loops; waking the drains ends their waits at once."""
    for loop in _loops():
        loop.stop()
    for wake in (command_wake, dispatch_wake, edit_wake):
        wake.set()


def _start(loop) -> bool:
    """Start loop; False when its task is still winding down."""
    try:
        loop.start()
        return True
    except RuntimeError as e:
        print(f"Loop {loop.coro.__name__} not started: {e}")
        return False


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
