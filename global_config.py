from commands.help import print_help
from commands.update_bot import restart_bot, stop_bot, git_pull, update_bot
from commands.query_card import query_card, alias_card, delete_alias, print_all_aliases

# Insert new commands and their function hooks in here
commands = {
    "?"                 : query_card,
    ".help"             : print_help,
    ".restart"          : restart_bot,
    ".stop"             : stop_bot,
    ".pull"             : git_pull,
    ".update"           : update_bot,
    ".alias"            : alias_card,
    ".del_alias"        : delete_alias,
    ".print_aliases"    : print_all_aliases,
}
