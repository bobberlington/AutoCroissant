from typing import Callable
from commands.diffusion import diffusion, set_scheduler, set_device, set_model, set_lora, get_qsize
from commands.frankenstein import frankenstein
from commands.help import print_help
from commands.music_player import list_all_music, play_music, replay, replay_all, set_volume, shuffle_music, skip, loop, pause, print_queue, print_prev_queue, clear_queue, stop, disconnect
from commands.query_card import alias_card
from commands.utils import to_thread
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

    "-play"             : play_music,
    "-replay_all"       : to_thread(replay_all),
    "-replay"           : to_thread(replay),
    "-skip"             : to_thread(skip),
    "-loop"             : to_thread(loop),
    "-list"             : to_thread(list_all_music),
    "-volume"           : to_thread(set_volume),
    "-shuffle"          : to_thread(shuffle_music),
    "-prev_queue"       : to_thread(print_prev_queue),
    "-queue"            : to_thread(print_queue),
    "-clear"            : to_thread(clear_queue),
    "-pause"            : to_thread(pause),
    "-stop"             : to_thread(stop),
    "-disconnect"       : to_thread(disconnect),
}

bot_admin_ids = [80731173726191616]
