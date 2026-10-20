from discord import Interaction

from commands.utils import queue_message, split_long_message


general_commands = {
    "/help <text/card/music/stats/general>:"    : "Prints the help message pertaining to the specified command category.",
    "/restart_bot:"                             : "Restarts the bot.",
    "/stop_bot:"                                : "Stops the bot (admin only).",
    "/pull:"                                    : "Does a hard reset, then a git pull, and reports status (admin only).",
    "/push:"                                    : "Commits the card pickles, rebases onto new code, pushes and reports status (admin only).",
    "/update [force_reset]:"                    : "Does a push, then pull, then restarts (admin only). force_reset: hard reset to origin first (dangerous!).",
    "/purge [user] [num] [bulk]:"               : "Deletes messages. Can specify user, number of messages (default 100), and bulk mode.",
    "/list_guilds:"                             : "Lists all guilds the bot is currently a member of (admin only).",
    "/leave_guild <guild_id>:"                  : "Forces the bot to leave a specified guild (admin only). Provide the guild ID.",
    "/sync_global:"                             : "Syncs all slash commands globally and per guild (admin only).",
    "/react <emojis> [message_id]:"             : "Adds reactions to a message.\n"
                                                  "- emojis: Space-separated emojis (👍 👎 or <:custom:123456>)\n"
                                                  "- message_id: Optional. If omitted, reacts to the most recent message in the channel.",
}

text_commands = {
    "/set_reminder [msg] [when] [offset] [frequency] [command]:" : "Posts a message and/or runs a bot command in this channel at a set time (admin only).\n"
                                                                    "- msg: Message to post\n"
                                                                    "- when: Pacific time (follows daylight saving): 13:00, 1PM, 1:30PM, sun 3AM, tomorrow 9AM or 2026-12-25 9AM. A bare time that already passed today fires right away\n"
                                                                    "- offset: Delay added to the first run, e.g. 30m or 1d (default: none)\n"
                                                                    "- frequency: Repeat interval in s/m/h/d/w, combinable like 1d12h (default: never)\n"
                                                                    "- command: Bot command to run with your permissions, options as name:value, e.g. '/sync_emojis refresh:True'",
    "/list_reminders [all] [hidden]:"           : "Lists reminders with their ID, next run, repeat, command and creator.\n"
                                                  "- all: The whole server instead of this channel (default: False)\n"
                                                  "- hidden: Show the list only to you (default: False)",
    "/remove_reminder <reminder_id>:"           : "Removes a reminder by the 8-character ID from /list_reminders (admins, or the reminder's creator).",
}

card_commands = {
    "/query <query>:"                           : "Posts the requested card image from the repo by name. Uses fuzzy matching. "
                                                  "The Card page button opens it on the website; website names and notes show too.",
    "/website:"                                 : "Link to the card browser website (cards, rulebook, deck builder, stats, editing).",
    "/query_ability <query> [limit] [howmany]:" : "Searches for all cards with the requested text/conditions and posts their images.\n"
                                                   "- limit: Max number of cards to display (default: all)\n"
                                                   "- howmany: If True, only returns the count instead of images\n",
    "/query_rulebook <search_text> [limit]:"    : "Searches the rulebook entries for text matching <search_text>. "
                                                  "Optional limit for max results (default: all).",
    "Query Syntax:"                             : "Flexible query syntax for /query_ability:\n"
                                                  "- Exact text match: Surround with quotes \"like this\"\n"
                                                  "- Exclude matches: Prefix with ! to get cards that don't match\n"
                                                  "- OR queries: Use | to combine, e.g., 'fire | water'\n"
                                                  "- AND queries: Use & to require multiple, e.g., 'fire & flying'\n"
                                                  "- Numeric comparisons: hp>5, def<=10, atk==3, spd>=7, stars<6\n"
                                                  "- String exact: type==creature, subtype==attack, series==kirby\n"
                                                  "- String contains: type~creature, series~kirby (case-insensitive)\n"
                                                  "- Combine conditions: '\"damage\" & type==creature & hp>7'\n"
                                                  "- Fuzzy matching works for non-exact text queries",
    "/sync_emojis [refresh] [prune]:"           : "Uploads every missing or edited icon in Types/ and Types/Stars/ as one of the bot's emojis, so [fire]-style tokens show as icons in replies (admin only). /update_stats does this too when the card repo changed.\n"
                                                  "- refresh: Re-upload every icon, even unchanged ones\n"
                                                  "- prune: Delete bot emojis whose icon is no longer in Types/",
    "/set_ratio [value]:"                       : "View or set the fuzzy match tolerance (0.0-1.0). "
                                                  "No argument shows current ratio. Higher = stricter matching.",
    "/set_repo [new_repo]:"                     : "View or set the GitHub repository (format: USER/REPO). "
                                                  "No argument shows current repo.",
    "/alias [alias] [card]:"                    : "Manage card aliases. Maps a short name (e.g., 'red') to a full card name for improved search accuracy.\n"
                                                  "- No arguments: View all aliases\n"
                                                  "- One argument: Delete the specified alias\n"
                                                  "- Two arguments: Create/update an alias",
    "/frankenstein <cards> [blend_mode]:"       : "Combines multiple cards into a single 'frankensteined' image.\n"
                                                  "- cards: Comma-separated list, e.g., 'Red Bloon, Blue Bloon'\n"
                                                  "- blend_mode: 'slice' (default) or 'average'",
}

stats_commands = {
    "/update_stats [output_problematic_cards] [force_update] [cards]:" : "Updates the card database from GitHub: re-parses only PSDs that changed, and re-uploads edited type emojis. "
                                                               "When card images, TTS/decks.json, MDW/Card Edits.json or the TTS save changed, it also updates the TTS save (as /update_tts_save; posts only when the save changed or the update failed).\n"
                                                               "- output_problematic_cards: Show cards with errors (default: True).\n"
                                                               "- force_update: Re-parse every card, even unchanged ones (downloads every PSD; default: False).\n"
                                                               "- cards: Card names to re-parse anyway, comma separated (e.g. after a parser fix).",
    "/update_tts_save:"                         : "Updates the Tabletop Simulator save on the card repo (TTS/Impossibility Simulator.json) and commits it (admin only): "
                                                  "pins card image links, adds new cards to complete decks (rules in TTS/decks.json), fixes card names. "
                                                  "Posts a summary, the full report as a file, and the save's download link.",
    "/export_cards [only_ability] [as_csv]:"    : "Exports all card data/metadata to a file (excludes rulebook).\n"
                                                   "- only_ability: Export only ability text (default: True)\n"
                                                   "- as_csv: Export as CSV instead of text (default: True)",
    "/export_rulebook:"                         : "Exports the rulebook pages as a text file.",
    "/update_metadata <query> [del_entry] [author] [ability] [stars] [subtype] [series] [hp] [defense] [attack] [speed] [card_type] [types]:" :
                                                  "Edit metadata for a card. Provide query to find card, then any fields to update.\n"
                                                  "- del_entry: Delete entire metadata entry\n"
                                                  "- types: Comma-separated list, e.g., 'fire,water'\n"
                                                  "- author: cards listed in TTSCardMaker's MDW/Card Authors.txt get the author from that file again on the next /update_stats; fix those in the file\n"
                                                  "- fields fixed on the website (MDW/Card Edits.json) are set from that file again on the next /update_stats\n"
                                                  "- No parameters (except query) shows current metadata",
    "/mass_replace <field> <old> <new>:"        : "Replace all instances of one field value with another across the entire database (author, series, subtype, card_type).",
    "/list_orphans:"                            : "Lists all cards without a listed author.",
}

music_commands = {
    "/play <song> [play_next]:"                 : "Adds a song to the queue (local file/folder or URL to song/playlist).\n"
                                                  "- play_next: True to add to front of queue (default: False)",
    "/play_all:"                                : "Queues all songs in the local music directory.",
    "/replay [song]:"                           : "Replays a previously played song.\n"
                                                  "- 0 or empty: current song\n"
                                                  "- 1: previous song\n"
                                                  "- 2: two songs ago, etc.",
    "/replay_all:"                              : "Replays all previously played songs in order.",
    "/skip:"                                    : "Skips the currently playing song.",
    "/loop:"                                    : "Toggles looping of the currently playing song.",
    "/list:"                                    : "Lists all songs in the bot's music directory.",
    "/volume [multiplier]:"                     : "Changes volume by a multiplier (0.5 = half, 1.0 = default, 2.0 = double).",
    "/shuffle:"                                 : "Shuffles the currently queued songs randomly.",
    "/queue:"                                   : "Lists all currently queued songs.",
    "/prev_queue:"                              : "Lists all previously played songs.",
    "/clear:"                                   : "Clears the music queue.",
    "/pause:"                                   : "Pauses/unpauses the current song.",
    "/stop:"                                    : "Stops the currently playing song and clears the queue.",
    "/disconnect:"                              : "Disconnects the bot from voice, stopping playback and clearing all queues.",
    "/delete_song <song_name>:"                 : "Deletes a song or folder from the music directory.",
    "/delete_all_music:"                        : "Deletes all music files under the music directory.",
}

def print_help(interaction: Interaction, help_wanted: str):
    """Print help information for the specified command category."""
    if help_wanted == "text":
        wanted_commands = text_commands
    elif help_wanted == "card":
        wanted_commands = card_commands
    elif help_wanted == "music":
        wanted_commands = music_commands
    elif help_wanted == "stats":
        wanted_commands = stats_commands
    else:
        wanted_commands = general_commands

    help_msg = "```Available commands:\n\n"
    for cmd, desc in wanted_commands.items():
        help_msg += f"{cmd}\n{desc}\n\n"
    help_msg += "```"

    # Send each part after splitting
    for part in split_long_message(help_msg):
        queue_message(interaction, part)


# ========================
# Module Exports
# ========================
__all__ = [
    'print_help',
]
