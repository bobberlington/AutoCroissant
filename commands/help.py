from discord import Message


general_commands = {
    ".help <card/music>:"                       : "Prints the help message pertaining to general/card/music commands.",
    ".restart:"                                 : "Restarts the bot.",
    ".stop:"                                    : "Stops the bot.",
    ".pull:"                                    : "Does a hard reset, than a git pull, and reports status.",
    ".push:"                                    : "Does a git push, and reports status.",
    ".update:"                                  : "Does a pull, then restarts.",
    ".purge:"                                   : "Deletes all messages by the bot, doesnt require perms.",
    ".quickpurge:"                              : "Deletes all messages by the bot quickly, requires perms.",
}

card_commands = {
    "?<Card_Name>:"                             : "Posts the requested <Card_Name> image from the repo.",
    "?? <Card_Description>:"                    : "Searches for all cards with the requested description and posts their image.",
    "?howmany <Card_Description>:"              : "Posts number of cards with requested description.",
    "<Card_Description>:"                       : "Additionally, for ?? and ?howmany,\nIf the query is surrounded by quotation \"\" marks, then only exact matches are returned.\n\
                                                    If the query is preceded by an exclamation mark !, then only things that don't match that query are returned.\n\
                                                    If there is a | symbol in the query, it treats both the partition before and the partition after the | symbol as separate queries, and combines their results.\n\
                                                    If there is a & symbol in the query, it treats both the partition before and the partition after the & symbol as seperate queries, and only returns results that are in both.\n\
                                                    Multiple symbols can be combined. for example, \'?? \"atk=8\" & !\"spd=4\"\' will return all cards with 8 ATK and any SPD value except for 4.",
    "?set_ratio <New_Ratio>:"                   : "If this is called with no arguments, returns the current match ratio. Otherwise, sets the match ratio to the passed in float.",
    "?set_repo <New_Repository>:"               : "If this is called with no arguments, returns the current repository. Otherwise, sets the repository to the passed in string (USER/REPO).",
    ".alias <Card_Name> <Card_Path>:"           : "Permanantly aliases a short name such as \"Red_Bloon\" to a path such as \"PoD01Red_Bloon.png\", increasing accuracy and execution time.\n\
                                                    Alternatively, if the first argument is 'del', deletes the second argument from aliases. If no arguments are passed, prints all existing aliases.",
    ".frankenstein <card1>,<card2>,...:"        : "Takes a list of creatures to mash together, and returns an image of the frankensteined creatures.",
    ".ai <optional url/img> <prompt>"           : "Takes either an image, url, or nothing except for a prompt, and generates an ai image. Can also take a second image to use as a mask.",
    ".ai_queue"                                 : "Returns the current ai queue size and the content of each queued request.",
    ".set_scheduler"                            : "If this is called with no arguments, returns the current scheduler and a list of possible choices. Otherwise, sets the new scheduler.",
    ".set_device <New_Device#>:"                : "If this is called with no arguments, returns the device#. Otherwise, sets the device#.",
    ".set_model <New_Model>:"                   : "If this is called with no arguments, returns the current model and options. Otherwise, sets the model to the passed in string.",
    ".set_lora <New_Lora>:"                     : "If this is called with no arguments, returns the current lora and options. Otherwise, sets the lora to the passed in string.",
}

music_commands = {
    "-play <song>:"                             : "Plays/adds this song(s) to a queue, either a local file/folder or link to a song/playlist."
}

async def print_help(message: Message):
    help_wanted = ' '.join(message.content.split()[1:])
    wanted_commands = general_commands
    if help_wanted.find("card") != -1:
        wanted_commands = card_commands
    elif help_wanted.find("music") != -1:
        wanted_commands = music_commands

    help_msg = f"```Available commands:\n\n"
    for cmd in wanted_commands:
        help_msg += f"{cmd:15s} {wanted_commands[cmd]}\n\n"
        if len(help_msg) > 1000:
            await message.channel.send(help_msg + "```")
            help_msg = "```"
    await message.channel.send(help_msg + "```")
