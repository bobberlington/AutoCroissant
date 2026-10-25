"""
reminder.json backed up to this repo on change (owner 2026-10-25). The live file stays untracked (deploys never clash);
a copy is committed to backup/<file> on main through the Git Data API, BACKUP_DELAY seconds after the last save. Only the
production service backs up (systemd sets INVOCATION_ID), so a dev run never overwrites it.

    changed(path)   after a save: schedule the commit (saves in a row make one)
"""
from os import environ
from os.path import exists
from threading import Lock, Timer
from traceback import format_exc

from commands.github_api import blob_sha

# ========================
# Configuration
# ========================
BOT_REPO = "bobberlington/AutoCroissant"
BACKUP_DIR = "backup"                   # backup/reminder.json
BACKUP_DELAY = 10                       # seconds after the last save
COMMIT_TRIES = 3

# ========================
# Global State
# ========================
_lock = Lock()
_timers: dict = {}                      # path -> pending Timer


def changed(path: str) -> None:
    """A live file was saved: commit its copy in BACKUP_DELAY seconds (a later save restarts the wait)."""
    if not environ.get("INVOCATION_ID"):
        return
    with _lock:
        if path in _timers:
            _timers[path].cancel()
        timer = Timer(BACKUP_DELAY, _commit, (path,))
        timer.daemon = True
        _timers[path] = timer
        timer.start()


def _commit(path: str) -> None:
    """Commit path's bytes to BACKUP_DIR/path unless the backup already holds them; failures are printed, never raised."""
    with _lock:
        _timers.pop(path, None)
    try:
        import commands.git_commit as gc
        if not exists(path):
            return
        with open(path, "rb") as fh:
            data = fh.read()
        target = f"{BACKUP_DIR}/{path}"
        for _ in range(COMMIT_TRIES):
            head = gc.ref_head(BOT_REPO)
            tree = gc.commit_info(BOT_REPO, head)["tree"]["sha"]
            reply = gc.api(BOT_REPO, "GET", f"contents/{target}?ref={head}", check=False)
            if reply.ok and reply.json().get("sha") == blob_sha(data):
                return
            blob = gc.api(BOT_REPO, "POST", "git/blobs", json={"content": data.decode("utf-8"), "encoding": "utf-8"}).json()["sha"]
            if gc.api_commit(BOT_REPO, head, tree, {}, {target: blob}, f"Backup: {path}"):
                print(f"Backup: {path} committed to {BOT_REPO}.")
                return
        print(f"Backup: {path} not committed: main kept moving.")
    except Exception:
        print(f"Backup: {path} not committed:\n{format_exc()}")
