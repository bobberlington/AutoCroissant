import discord


commands = {
    ".help:"                                    : "Prints this help message.",
    ".restart:"                                 : "Restarts the bot.",
    ".stop:"                                    : "Stops the bot.",
    ".pull:"                                    : "Does a hard reset, than a git pull, and reports status.",
    ".push:"                                    : "Does a git push, and reports status.",
    ".update:"                                  : "Does a pull, then restarts.",
    ".purge:"                                   : "Deletes all messages by the bot, doesnt require perms.",
    ".quickpurge:"                              : "Deletes all messages by the bot quickly, requires perms.",
    ".alias <Card_Name> <Card_Path>:"           : "Permanantly aliases a short name such as \"Red_Bloon\" to a path such as \"PoD01Red_Bloon.png\", increasing accuracy and execution time.",
    ".del_alias <Card_Name>:"                   : "Permanantly deletes the alias previously mapped to the name.",
    ".print_aliases:"                           : "Prints all existing aliases.",
    ".frankenstein <card1>,<card2>,...:"        : "Takes a list of creatures to mash together, and returns an image of the frankensteined creatures.",
    ".ai <optional url/img> <prompt>"           : "Takes either an image, url, or nothing except for a prompt, and generates an ai image.",
    ".set_device <New_Device> <New_Device#>:"   : "If this is called with no arguments, returns the device and device#. Otherwise, sets the device and device#.",
    "?find <Card_Description>:"                 : "Searches for all cards with the requested description and posts their image.",
    "?howmany <Card_Description>:"              : "Posts number of cards with requested description.",
    "<Card_Description>:"                       : "Additionally, for ?find and ?howmany,\nIf the query is surrounded by quotation \"\" marks, then only exact matches are returned.\n\
                                                    If the query is preceded by an exclamation mark !, then only things that don't match that query are returned.\n\
                                                    If there is a | symbol in the query, it treats both the partition before and the partition after the | symbol as separate queries, and combines their results.\n\
                                                    If there is a & symbol in the query, it treats both the partition before and the partition after the & symbol as seperate queries, and only returns results that are in both.\n\
                                                    Multiple symbols can be combined. for example, \'?find \"atk=8\" & !\"spd=4\"\' will return all cards with 8 ATK and any SPD value except for 4.",
    "?set_ratio <New_Ratio>:"                   : "If this is called with no arguments, returns the current match ratio. Otherwise, sets the match ratio to the passed in float.",
    "?<Card_Name>:"                             : "Posts the requested <Card_Name> image from the repo.",
} 

async def print_help(message: discord.Message):
    help_msg = f"```Available commands:\n\n"
    for cmd in commands:
        help_msg += f"{cmd:15s} {commands[cmd]}\n\n"
        if len(help_msg) > 1000:
            await message.channel.send(help_msg + "```")
            help_msg = "```"
    await message.channel.send(help_msg + "```")
