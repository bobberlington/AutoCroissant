"""
reminder.json backed up to this repo on change (owner 2026-10-25). The live file stays untracked (deploys never clash);
a copy is committed to backup/<file> on main BACKUP_DELAY seconds after the last save, from a separate shallow clone
pushed over SSH (the service's ssh-agent; the GitHub tokens can't write this repo). Only the production service backs
up (systemd sets INVOCATION_ID), so a dev run never overwrites it.

    changed(path)   after a save: schedule the commit (saves in a row make one)
"""
from os import environ, makedirs
from os.path import abspath, dirname, exists, expanduser, join
from shutil import copyfile
from subprocess import run
from threading import Lock, Timer
from traceback import format_exc

# ========================
# Configuration
# ========================
BACKUP_CLONE = expanduser("~/Documents/AutoCroissant-backup")   # used only for backup commits
BACKUP_DIR = "backup"                   # backup/reminder.json
BACKUP_DELAY = 10                       # seconds after the last save
BRANCH = "main"
TRIES = 3                               # a push refused because main moved: fetch and commit again
GIT_TIMEOUT = 120
IDENTITY = ("-c", "user.name=AutoCroissant", "-c", "user.email=autocroissant@gmail.com")

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


def _git(where: str, *args: str, check: bool = True):
    return run(["git", "-C", where, *args], capture_output=True, timeout=GIT_TIMEOUT, check=check,
               env={**environ, "GIT_SSH_COMMAND": "ssh -o BatchMode=yes"})


def _commit(path: str, clone: str = "") -> None:
    """Commit path's bytes to backup/<path> unless main already holds them; failures are printed, never raised."""
    clone = clone or BACKUP_CLONE
    with _lock:
        _timers.pop(path, None)
    try:
        if not exists(path):
            return
        if not exists(join(clone, ".git")):
            url = _git(dirname(abspath(path)), "remote", "get-url", "origin").stdout.decode().strip()
            run(["git", "clone", "-q", "--depth", "1", "--branch", BRANCH, url, clone], capture_output=True,
                timeout=GIT_TIMEOUT, check=True, env={**environ, "GIT_SSH_COMMAND": "ssh -o BatchMode=yes"})
        target = join(BACKUP_DIR, path)
        for _ in range(TRIES):
            _git(clone, "fetch", "-q", "--depth", "1", "origin", BRANCH)
            _git(clone, "reset", "-q", "--hard", "FETCH_HEAD")
            makedirs(join(clone, BACKUP_DIR), exist_ok=True)
            copyfile(path, join(clone, target))
            _git(clone, "add", target)
            if _git(clone, "diff", "--cached", "--quiet", check=False).returncode == 0:
                return
            _git(clone, *IDENTITY, "commit", "-q", "-m", f"Backup: {path}")
            if _git(clone, "push", "-q", "origin", f"HEAD:{BRANCH}", check=False).returncode == 0:
                print(f"Backup: {path} committed.")
                return
        print(f"Backup: {path} not committed: main kept moving.")
    except Exception:
        print(f"Backup: {path} not committed:\n{format_exc()}")
