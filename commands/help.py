from discord import Interaction

from commands.utils import queue_long_message


# One line per command (Discord's picker shows each option's hint); the query syntax in full: the picker can't show it.
general_commands = {
    "/help <category>":                      "This list, for one category.",
    "/restart_bot":                          "Restarts the bot.",
    "/stop_bot":                             "Stops the bot (admin).",
    "/pull":                                 "Pulls the latest code (admin).",
    "/update [force_reset]":                 "Pulls the latest code and restarts; force_reset discards local changes (admin).",
    "/purge [user] [num] [bulk]":            "Deletes a user's recent messages in this channel (default: the bot's) (admin).",
    "/list_guilds":                          "Lists the bot's servers (admin).",
    "/leave_guild <guild_id>":               "Leaves a server (admin).",
    "/sync_global":                          "Syncs the slash commands everywhere (admin).",
    "/list_guild_channels <guild_id>":       "Lists a server's channels (admin).",
    "/get_channel_messages <guild_id> <channel_id> [limit]": "Lists a channel's recent messages (admin).",
    "/react <reactions> [message_id]":       "Adds reactions to a message (default: the latest).",
}

text_commands = {
    "/set_reminder [msg] [when] [offset] [frequency] [command]": "Posts a message or runs a command here at a time, Pacific, optionally repeating (admin).",
    "/reminders":                            "Your panel of reminders: reschedule, turn off/on, delete, new.",
    "/remove_reminder <reminder_id>":        "Removes a reminder by its ID from /reminders (admins or its creator).",
}

card_commands = {
    "/query <query> [as_of]":                "Posts a card by name, title or alias (closest match if none); as_of: the card on a date.",
    "/website":                              "Links the card website.",
    "/query_ability <query> [limit] [howmany]": "Posts the cards matching text or conditions (syntax below); howmany: just the count.",
    "/query_rulebook <search_text> [limit]": "Quotes the rulebook lines holding the text, by chapter.",
    "/sync_emojis [refresh] [prune]":        "Uploads the icons in Types/ as the bot's emojis (admin).",
    "/set_ratio [value]":                    "Shows or sets the name-match tolerance, 0-1 (higher: stricter).",
    "/set_repo [new_repo]":                  "Shows or sets the card repo, USER/REPO (admin).",
    "/frankenstein <cards> [blend_mode]":    "Combines cards into one image (slice or average).",
    "Query syntax":                          "\n  \"exact text\"  !not  a | b  a & b"
                                             "\n  hp>5  def<=10  atk==3  spd>=7  stars<6  total>=25"
                                             "\n  type==creature  series==kirby  subseries~door  alias~mk  (== exact, ~ contains)"
                                             "\n  created>=2025, 2023-10 or 2021-06-01; values with & or | in quotes"
                                             "\n  damage/DMG, defense/DEF, speed/SPD, health/HP find each other",
}

stats_commands = {
    "/update_stats [output_problematic_cards] [force_update] [cards]": "Loads the card data Card sync built now (card commands load it by "
                                                                       "themselves); force_update or cards re-parse every card or some (admin).",
    "/fonts":                                "Lists the fonts the cards ask for that MDW/Fonts lacks, from the card data.",
    "/export_cards [only_ability] [as_csv]": "Exports every card's data as a file.",
    "/export_rulebook":                      "Exports the rulebook as a text file.",
    "/view_old_metadata <query> [version]":  "Shows an older stored version of a card (0 = the last, -1 = the oldest).",
    "/list_orphans":                         "Lists the cards without an author.",
}

music_commands = {
    "/play <song> [play_next]":              "Queues a song, folder or URL (play_next: at the front).",
    "/play_all":                             "Queues every local song.",
    "/replay [song]":                        "Replays a song (0: this one, 1: the one before, ...).",
    "/replay_all":                           "Replays every played song in order.",
    "/skip":                                 "Skips the song.",
    "/loop":                                 "Loops the song (toggle).",
    "/list":                                 "Lists the local songs.",
    "/volume [multiplier]":                  "Sets the volume (1.0 = normal).",
    "/shuffle":                              "Shuffles the queue.",
    "/queue":                                "Lists the queue.",
    "/prev_queue":                           "Lists the played songs.",
    "/clear":                                "Clears the queue.",
    "/pause":                                "Pauses or resumes.",
    "/stop":                                 "Stops and clears the queue.",
    "/disconnect":                           "Leaves voice and clears the queues.",
    "/delete_song <song_name>":              "Deletes a song or folder (admin).",
    "/delete_all_music":                     "Deletes all local music (admin).",
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

    help_msg = "```" + "\n".join(f"{cmd}: {desc}" for cmd, desc in wanted_commands.items()) + "```"

    queue_long_message(interaction, help_msg)


# ========================
# Module Exports
# ========================
__all__ = [
    'print_help',
]
