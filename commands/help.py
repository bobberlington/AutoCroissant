from discord import Interaction
from discord.errors import InteractionResponded



general_commands = {
    "/help <card/ai/music>:"                    : "Prints the help message pertaining to general/card/ai/music commands.",
    "/restart:"                                 : "Restarts the bot.",
    "/stop:"                                    : "Stops the bot.",
    "/pull:"                                    : "Does a hard reset, than a git pull, and reports status.",
    "/push:"                                    : "Does a git push, and reports status.",
    "/update:"                                  : "Does a pull, then restarts.",
    "/purge:"                                   : "Deletes all messages by the bot.",
}

card_commands = {
    "/query_card <Card_Name>:"                  : "Posts the requested <Card_Name> image from the repo.",
    "/query_desc <Card_Description>:"           : "Searches for all cards with the requested description and posts their image.",
    "/howmany <Card_Description>:"              : "Posts number of cards with requested description.",
    "<Card_Description>:"                       : "If the query is surrounded by quotation \"\" marks, then only exact matches are returned.\n\
                                                    If the query is preceded by an exclamation mark !, then only things that don't match that query are returned.\n\
                                                    If there is a | symbol in the query, it treats both the partition before and the partition after the | symbol as separate queries, and combines their results.\n\
                                                    If there is a & symbol in the query, it treats both the partition before and the partition after the & symbol as seperate queries, and only returns results that are in both.\n\
                                                    Multiple symbols can be combined. for example, \'\"atk=8\" & !\"spd=4\"\' will return all cards with 8 ATK and any SPD value except for 4.",
    "/set_ratio <New_Ratio>:"                   : "If this is called with no arguments, returns the current match ratio. Otherwise, sets the match ratio to the passed in float.",
    "/set_repo <New_Repository>:"               : "If this is called with no arguments, returns the current repository. Otherwise, sets the repository to the passed in string (USER/REPO).",
    "/alias <Card_Name> <Card_Path>:"           : "Permanantly aliases a short name such as \"Red_Bloon\" to a path such as \"PoD01Red_Bloon.png\", increasing accuracy and execution time.\nIf no arguments are passed, prints all existing aliases.",
    "/del_alias <Card_Name>:"                   : "Deletes a previously created alias.",
    "/frankenstein <card1>,<card2>,...:"        : "Takes a list of creatures to mash together, and returns an image of the frankensteined creatures.",
}

ai_commands = {
    "/ai <optional url/img> <prompt>"           : "Takes either an image, url, or nothing except for a prompt, and generates an ai image. Can also take a second image to use as a mask.",
    "/ai_queue"                                 : "Returns the current ai queue size and the content of each queued request.",
    "/set_scheduler"                            : "If this is called with no arguments, returns the current scheduler and a list of possible choices. Otherwise, sets the new scheduler.",
    "/set_device <New_Device#>:"                : "If this is called with no arguments, returns the device#. Otherwise, sets the device#.",
    "/set_model <New_Model>:"                   : "If this is called with no arguments, returns the current model and options. Otherwise, sets the model to the passed in string.",
    "/set_lora <New_Lora>:"                     : "If this is called with no arguments, returns the current lora and options. Otherwise, sets the lora to the passed in string.",
}

music_commands = {
    "/play <song>:"                             : "Plays/adds this song(s) to the back of the queue, either a local file/folder or link to a song/playlist.",
    "/play_all:"                                : "Queues all songs in the music directory.",
    "/replay <index>:"                          : "Replays a song that was previously played, where an index of 0 or nothing is the current song, 1 is the previous song, 2 is 2 songs ago, etc.",
    "/replay_all:"                              : "Replays all previously played songs.",
    "/skip:"                                    : "Skips the currently playing song.",
    "/loop:"                                    : "Loops the currently playing song.",
    "/list:"                                    : "Lists all songs under the bot's music directory.",
    "/volume <value>:"                          : "Increases/decreases volume by a multiplier of the given value (0.5 is half as loud, 2 is 2x as loud).",
    "/shuffle:"                                 : "Shuffles the currently queued songs.",
    "/queue:"                                   : "Lists all currently queued songs.",
    "/prev_queue:"                              : "Lists all previously played songs.",
    "/clear:"                                   : "Clears the queue.",
    "/pause:"                                   : "Pauses/unpauses the current song.",
    "/stop:"                                    : "Stops the currently playing song, and clears the queue.",
    "/disconnect:"                              : "Disconnects the bot, stopping the current song and clearing all queues.",
}

async def print_help(interaction: Interaction, help_wanted):
    if not help_wanted:
        help_wanted = ""
    help_wanted = str(help_wanted)
    wanted_commands = general_commands
    if "card" in help_wanted:
        wanted_commands = card_commands
    elif "ai" in help_wanted:
        wanted_commands = ai_commands
    elif "music" in help_wanted:
        wanted_commands = music_commands

    help_msg = f"```Available commands:\n\n"
    for cmd in wanted_commands:
        help_msg += f"{cmd:15s} {wanted_commands[cmd]}\n\n"
        if len(help_msg) > 1000:
            try:
                await interaction.response.send_message(help_msg + "```")
            except InteractionResponded:
                await interaction.followup.send(help_msg + "```")
            help_msg = "```"
    try:
        await interaction.response.send_message(help_msg + "```")
    except InteractionResponded:
        await interaction.followup.send(help_msg + "```")
