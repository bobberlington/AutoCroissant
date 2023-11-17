commands = {
    ".help:"                            : "Prints this help message.",
    ".restart:"                         : "Restarts the bot.",
    ".stop:"                            : "Stops the bot.",
    ".pull:"                            : "Does a git pull, and reports status.",
    ".update:"                          : "Does a pull, then restarts.",
    ".alias <Card_Name> <Card_Path>:"   : "Permanantly aliases a short name such as \"Red_Bloon\" to a path such as \"PoD01Red_Bloon.png\", increasing accuracy and execution time.",
    ".del_alias <Card_Name>:"           : "Permanantly deletes the alias previously mapped to the name.",
    ".print_aliases:"                   : "Prints all existing aliases.",
    "?find <Card_Description>:"         : "Searches for all cards with the requested description and posts their image. If query is in quotation \"\" marks, only returns exact results.",
    "?howmany <Card_Description>:"      : "Posts number of cards with requested description. If query is in quotation \"\" marks, only returns exact results.",
    "?set_ratio <New_Ratio>:"           : "If this is called with no arguments, returns the current match ratio. Otherwise, sets the match ratio to the passed in float.",
    "?<Card_Name>:"                     : "Posts the requested <Card_Name> image from the repo.",
} 

async def print_help(message):
    help_msg = f"```Available commands:\n\n"
    for cmd in commands:
        help_msg += f"{cmd:15s} {commands[cmd]}\n\n"
    await message.channel.send(help_msg + "```")
