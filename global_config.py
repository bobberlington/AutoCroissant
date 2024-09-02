from typing import Callable
from commands.diffusion import diffusion, set_scheduler, set_device, set_model, set_lora, get_qsize
from commands.frankenstein import frankenstein
from commands.help import print_help
from commands.query_card import alias_card
from commands.tools import to_thread
from commands.update_bot import restart_bot, stop_bot, git_pull, git_push, update_bot

# Insert new commands and their function hooks in here
commands: dict[str, Callable] = {
    ".help"             : print_help,
    ".restart"          : restart_bot,
    ".stop"             : stop_bot,
    ".pull"             : git_pull,
    ".push"             : git_push,
    ".update"           : update_bot,
    ".alias"            : alias_card,
    ".frankenstein"     : to_thread(frankenstein),
    ".ai_queue"         : get_qsize,
    ".ai"               : to_thread(diffusion),
    ".set_scheduler"    : set_scheduler,
    ".set_device"       : set_device,
    ".set_model"        : set_model,
    ".set_lora"         : set_lora,
}

list_of_all_types = [
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

list_of_all_attributes = [
    "nature",
    "monster",
    "demon",
    "brute",
    "military",
    "leadership",
    "martial Artist",
    "mage",
    "business",
    "performer",
    "engineer",
    "industrial",
    "rogue",
    "trickster",
    "explorer",
    "royalty",
    "prodigy",
    "spiritual",
    "haunted",
    "spectral",
    "infected",
    "mythical",
    "higher Being",
    "god",
    "creator"
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
