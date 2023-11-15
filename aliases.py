from commands.help import print_help
from commands.update_bot import restart_bot, stop_bot, git_pull, update_bot

# fill this with odd filenames
aliases = {
    "Red_Bloon"     : "PoD01Red_Bloon.png",
    "Black_Bloon"   : "PoD02Black_Bloon.png",
    "White_Bloon"   : "PoD03White_Bloon.png",
    "Zebra_Bloon"   : "PoD04Zebra_Bloon.png",
    "Lead_Bloon"    : "PoD05Lead_Bloon.png",
    "Rainbow_Bloon" : "PoD06Rainbow_Bloon.png",
    "Ceramic_Bloon" : "PoD07Ceramic_Bloon.png",
    "MOAB"          : "PoD08MOAB.png",
    "DDT"           : "PoD09DDT.png",
    "BFB"           : "PoD10BFB.png",
    "ZOMG"          : "PoD11ZOMG.png",
    "BAD"           : "PoD12BAD.png",
}

# Insert new commands and their function hooks in here
commands = {
    ".help"     : print_help,
    ".restart"  : restart_bot,
    ".stop"     : stop_bot,
    ".pull"     : git_pull,
    ".update"   : update_bot,
}
