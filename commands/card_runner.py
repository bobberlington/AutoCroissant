"""
bake_lock: one card-data writer at a time in the bot (update_stats, the fonts cache refresh, the startup sweep of a
cut-off run's temp PSDs). The card bake itself runs in TTSCardMaker's card-sync.yml Action.
"""
from os import getuid, remove, scandir
from re import compile as re_compile
from stat import S_ISREG
from tempfile import gettempdir
from threading import Lock
from time import time
from typing import Optional

# ========================
# Configuration
# ========================
TEMP_PSD = re_compile(r"tmp[a-z0-9_]{8}\.psd")  # mkstemp(suffix=".psd")'s names (card_data, card_check)
TEMP_AGE = 3600                                 # seconds: an older temp PSD is a cut-off run's

# ========================
# Global State
# ========================
bake_lock = Lock()                              # update_stats, the fonts cache refresh and the sweep never overlap


# ========================
# Public API Functions
# ========================
def sweep_temp_psds(folder: Optional[str] = None, age: float = TEMP_AGE) -> int:
    """
    Delete the temp PSDs a cut-off run left behind (startup: the jetson once held 56, 1.1 GB): only mkstemp's tmp*.psd
    names, regular files of this user older than age, never while update_stats or a font check runs (bake_lock).
    One line when any went.

    Returns:
        The number removed
    """
    if not bake_lock.acquire(blocking=False):
        return 0
    try:
        folder = folder or gettempdir()
        removed, size, cutoff = 0, 0, time() - age
        try:
            entries = list(scandir(folder))
        except OSError:
            return 0
        for entry in entries:
            if not TEMP_PSD.fullmatch(entry.name):
                continue
            try:
                info = entry.stat(follow_symlinks=False)
                if not S_ISREG(info.st_mode) or info.st_mtime > cutoff or info.st_uid != getuid():
                    continue
                remove(entry.path)
            except OSError:
                continue
            removed, size = removed + 1, size + info.st_size
        if removed:
            print(f"Removed {removed} stale temp PSD{'s' if removed != 1 else ''} ({size / 1e6:.0f} MB) from {folder}")
        return removed
    finally:
        bake_lock.release()


# ========================
# Module Exports
# ========================
__all__ = [
    'bake_lock',
    'sweep_temp_psds',
]
