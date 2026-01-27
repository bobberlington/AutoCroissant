from discord import Interaction

from commands.utils import queue_message, split_long_message


general_commands = {
    "/help <text/card/ai/music/general>:"       : "Prints the help message pertaining to the specified command category.",
    "/restart_bot:"                             : "Restarts the bot.",
    "/stop_bot:"                                : "Stops the bot (admin only).",
    "/pull:"                                    : "Does a hard reset, then a git pull, and reports status (admin only).",
    "/push:"                                    : "Does a git push and reports status (admin only).",
    "/update:"                                  : "Does a push, then pull, then restarts (admin only).",
    "/purge [user] [num] [bulk]:"               : "Deletes messages. Can specify user, number of messages (default 100), and bulk mode.",
    "/list_guilds:"                             : "Lists all guilds the bot is currently a member of (admin only).",
    "/leave_guild <guild_id>:"                  : "Forces the bot to leave a specified guild (admin only). Provide the guild ID.",
    "/sync_global:"                             : "Syncs all slash commands globally and per guild (admin only).",
    "/react <emojis> [message_id]:"             : "Adds reactions to a message.\n"
                                                  "- emojis: Space-separated emojis (👍 👎 or <:custom:123456>)\n"
                                                  "- message_id: Optional. If omitted, reacts to the most recent message in the channel.",
}

text_commands = {
    "/set_reminder [msg] [when] [offset] [frequency] [command]:" : "Set a reminder message to be sent at a specified time (e.g., '13:00' or '1PM' PST). "
                                                                    "Optional: offset delay before first run (s/m/h/d/w), frequency for repeating (s/m/h/d/w), "
                                                                    "and a bot command to execute (e.g., '/play_all').",
    "/list_reminders [all] [hidden]:"           : "List all reminders in this channel (or in the whole server if 'all' is True). "
                                                  "Set 'hidden' to True to show the list only to you.",
    "/remove_reminder <reminder_id>:"           : "Remove a reminder by its unique ID. Use /list_reminders to find the ID.",
}

card_commands = {
    "/query <query>:"                           : "Posts the requested card image from the repo by name. Uses fuzzy matching.",
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
    "/update_stats [output_problematic_cards] [use_local_repo] [use_local_timestamp] [force_update]:" : "Manually updates the card statistics database by scanning PSDs.\n"
                                                                                                    "- output_problematic_cards: Show cards with errors (default: True).\n"
                                                                                                    "- use_local_repo: Use local cloned folder vs remote (default: True).\n"
                                                                                                    "- use_local_timestamp: Use local file times vs GitHub (default: True).\n"
                                                                                                    "- force_update: Will process and update every single card, even if there have not been any changes to it (default: False).",
    "/export_cards [only_ability] [as_csv]:"    : "Exports all card data/metadata to a file (excludes rulebook).\n"
                                                   "- only_ability: Export only ability text (default: True)\n"
                                                   "- as_csv: Export as CSV instead of text (default: True)",
    "/export_rulebook:"                         : "Exports the rulebook pages as a text file.",
    "/update_metadata <query> [del_entry] [author] [ability] [stars] [subtype] [series] [hp] [defense] [attack] [speed] [card_type] [types]:" :
                                                  "Edit metadata for a card. Provide query to find card, then any fields to update.\n"
                                                  "- del_entry: Delete entire metadata entry\n"
                                                  "- types: Comma-separated list, e.g., 'fire,water'\n"
                                                  "- No parameters (except query) shows current metadata",
    "/mass_replace <field> <old> <new>:"        : "Replace all instances of one field value with another across the entire database (author, series, subtype, card_type).",
    "/list_orphans:"                            : "Lists all cards without a listed author.",
}

ai_commands = {
    "/ai <prompt> [image] [mask_image] [url] [mask_url] [steps] [height] [width] [resize] [cfg] [strength] [seed]:" :
                                                  "Generate AI images using text prompts and optional input images.\n"
                                                  "- prompt: Description of desired image\n"
                                                  "- image: Upload image to modify (img2img)\n"
                                                  "- mask_image: Control which parts to modify (black=keep, white=change)\n"
                                                  "- url/mask_url: URLs instead of uploads\n"
                                                  "- steps: Generation steps (default: 50)\n"
                                                  "- height/width: Output dimensions (default: 512x512)\n"
                                                  "- resize: Multiply output size (default: 1.0)\n"
                                                  "- cfg: Creativity vs prompt adherence (default: 7.0)\n"
                                                  "- strength: How much to modify input image (default: 0.8)\n"
                                                  "- seed: For reproducible results (default: random)",
    "/ai_queue:"                                : "Shows the current AI generation queue and details of each queued request.",
    "/set_scheduler [scheduler]:"               : "View or set the diffusion scheduler. No argument shows current and available options.\n"
                                                  "Available: 'dpm++ sde', 'dpm++ sde karras', 'euler a'",
    "/set_device [device]:"                     : "View or set the CUDA device number. No argument shows current device and available count.",
    "/set_model [model]:"                       : "View or set the diffusion model. No argument shows current model and available options.\n"
                                                  "Options include 'flux' (HuggingFace) and local .safetensors files.",
    "/set_lora [lora]:"                         : "View or set the LoRA to use. No argument shows current LoRA and available options.",
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
    elif help_wanted == "ai":
        wanted_commands = ai_commands
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
