from commands.help import print_help
from commands.update_bot import restart_bot, stop_bot, git_pull, git_push, update_bot
from commands.query_card import alias_card, delete_alias, print_all_aliases
from commands.frankenstein import frankenstein

# Insert new commands and their function hooks in here
commands = {
    ".help"             : print_help,
    ".restart"          : restart_bot,
    ".stop"             : stop_bot,
    ".pull"             : git_pull,
    ".push"             : git_push,
    ".update"           : update_bot,
    ".alias"            : alias_card,
    ".del_alias"        : delete_alias,
    ".print_aliases"    : print_all_aliases,
    ".frankenstein"     : frankenstein,
}

list_of_all_attributes = [
    "active",
    "attack",
    "buff",
    "buffstack",
    "dark",
    "darkmatter",
    "debuff",
    "debuffstack",
    "electric",
    "equip",
    "field",
    "fire",
    "food",
    "fragile equip",
    "ice",
    "light",
    "machine",
    "offense",
    "panda",
    "protective",
    "rock",
    "support",
    "undread",
    "error",
    "water",
    "wind"
]

list_of_all_stars = [
    "warpstar",
    "dst star",
    "omori star",
    "garnet_star",
    "ruby_star",
    "gold_star",
    "emerald_star",
    "diamond_star",
    "sapphire_starold",
    "crystal_star",
    "hk star",
    "undertale",
    "triforce",
    "lol",
    "nebula star",
    "nether_star",
    "guild seal",
    "isaac",
    "elden stars",
    "souls",
    "yingyangultra",
    "patrick star",
    "scrap",
    "btd6",
    "mother",
    "vi1",
    "feather",
    "terraria",
    "bfables"
]
