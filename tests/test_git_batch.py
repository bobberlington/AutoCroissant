"""Non-interactive git: GIT_TERMINAL_PROMPT=0 and ssh BatchMode on every git call, ssh options kept; lazy GitPython."""
import asyncio
import os
import subprocess
import sys
from subprocess import CompletedProcess
from types import SimpleNamespace

import pytest

import commands.git_commit as gc
import commands.repo_history as rh
import commands.update_bot as ub
from conftest import REPO


# ========================
# Fixtures
# ========================
@pytest.fixture
def clean_git(monkeypatch, tmp_path):
    """No ssh variables, no user or system git config, a fresh core.sshCommand cache."""
    for key in ("GIT_SSH_COMMAND", "GIT_SSH"):
        monkeypatch.delenv(key, raising=False)
    for key, value in {"GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1", "HOME": str(tmp_path),
                       "XDG_CONFIG_HOME": str(tmp_path)}.items():   # git 2.25 ignores GIT_CONFIG_GLOBAL
        monkeypatch.setenv(key, value)
    rh._configured_ssh.cache_clear()
    yield tmp_path
    rh._configured_ssh.cache_clear()


def _repo(path, *config):
    path.mkdir(exist_ok=True)
    subprocess.run(["git", "init", "-q", str(path)], check=True, capture_output=True)
    for key, value in config:
        subprocess.run(["git", "-C", str(path), "config", key, value], check=True, capture_output=True)
    return path


# ========================
# The shared helper
# ========================
def test_plain_ssh_gets_batch_mode(clean_git):
    assert rh.ssh_command() == "ssh -o BatchMode=yes"
    assert rh.batch_overrides() == {"GIT_TERMINAL_PROMPT": "0", "GIT_SSH_COMMAND": "ssh -o BatchMode=yes"}


def test_env_command_keeps_its_options(clean_git, monkeypatch):
    monkeypatch.setenv("GIT_SSH_COMMAND", "ssh -i ~/.ssh/deploy -o IdentitiesOnly=yes")
    assert rh.ssh_command() == "ssh -i ~/.ssh/deploy -o IdentitiesOnly=yes -o BatchMode=yes"
    monkeypatch.setenv("GIT_SSH_COMMAND", "ssh -o batchmode=no")      # set on purpose: left alone
    assert rh.ssh_command() == "ssh -o batchmode=no"


def test_core_ssh_command_is_kept(clean_git):
    repo = _repo(clean_git / "r", ("core.sshCommand", "ssh -i /keys/bot"))
    assert rh.ssh_command(str(repo)) == "ssh -i /keys/bot -o BatchMode=yes"
    assert rh.ssh_command(str(clean_git / "missing")) == "ssh -o BatchMode=yes"      # global config only


def test_git_ssh_program_is_left_alone(clean_git, monkeypatch):
    monkeypatch.setenv("GIT_SSH", "/usr/bin/plink")                    # takes no options, and wins over core.sshCommand
    assert rh.batch_overrides() == {"GIT_TERMINAL_PROMPT": "0"}


def test_env_keeps_the_agent_socket(clean_git, monkeypatch):
    monkeypatch.setenv("SSH_AUTH_SOCK", "/run/agent.sock")
    env = rh.batch_env("", GIT_AUTHOR_NAME="x")
    assert env["SSH_AUTH_SOCK"] == "/run/agent.sock" and env["GIT_TERMINAL_PROMPT"] == "0"
    assert env["GIT_SSH_COMMAND"].endswith("-o BatchMode=yes") and env["GIT_AUTHOR_NAME"] == "x"


# ========================
# Every caller uses it
# ========================
def test_history_clone_calls_are_batch(clean_git, monkeypatch):
    seen = []
    def fake_run(cmd, **kw):
        if "config" in cmd:                                            # the core.sshCommand lookup: unset
            return CompletedProcess(cmd, 1, stdout="")
        seen.append(kw["env"])
        return CompletedProcess(cmd, 0, stdout="")
    monkeypatch.setattr(rh, "run", fake_run)
    monkeypatch.setattr(rh, "HISTORY_CLONE", str(clean_git / "clone.git"))
    monkeypatch.setattr(rh, "HISTORY_JSON", str(clean_git / "h.json"))
    monkeypatch.setattr(rh, "_loaded", True)
    monkeypatch.setattr(rh, "_head", "")
    monkeypatch.setattr(rh, "_paths", {})
    monkeypatch.setattr(rh, "replace", lambda a, b: None)
    rh.refresh()                                                        # clone, then rev-parse
    assert len(seen) >= 2
    assert all(env["GIT_TERMINAL_PROMPT"] == "0" and "BatchMode=yes" in env["GIT_SSH_COMMAND"] for env in seen)


def test_commit_path_is_batch_and_keeps_the_key(clean_git, monkeypatch):
    clone = _repo(clean_git / "clone.git", ("core.sshCommand", "ssh -i /keys/bot"))
    monkeypatch.setattr(gc, "HISTORY_CLONE", str(clone))
    env = gc.git_env()
    assert env["GIT_SSH_COMMAND"] == "ssh -i /keys/bot -o BatchMode=yes" and env["GIT_AUTHOR_NAME"] == gc.AUTHOR["name"]
    seen = []
    monkeypatch.setattr(gc, "run_process", lambda cmd, **kw: seen.append(kw["env"]) or CompletedProcess(cmd, 0, b""))
    gc.git(str(clone), "rev-parse", "HEAD")                            # no env given: still no prompts
    assert seen[0]["GIT_TERMINAL_PROMPT"] == "0" and seen[0]["GIT_SSH_COMMAND"] == "ssh -i /keys/bot -o BatchMode=yes"


def test_pull_runs_ssh_in_batch_mode_with_the_key(clean_git, monkeypatch):
    log = clean_git / "ssh.log"
    recorder = clean_git / "rec.sh"
    recorder.write_text(f'#!/bin/sh\necho "$@" >> "{log}"\nexit 255\n')     # a failing ssh that logs its arguments
    recorder.chmod(0o755)
    repo = _repo(clean_git / "bot", ("core.sshCommand", f"{recorder} -i /keys/bot"))
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q",
                    "--allow-empty", "-m", "x"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(repo), "remote", "add", "origin", "ssh://git@example.invalid/r.git"], check=True)
    monkeypatch.chdir(repo)
    sent = []
    async def send(content):
        sent.append(content)
    asyncio.run(ub.git_pull(SimpleNamespace(followup=SimpleNamespace(send=send))))
    assert sent[0] == "Fetching from remote..." and sent[1].startswith("Git pull failed: ")
    calls = log.read_text()
    assert "-i /keys/bot" in calls and "-o BatchMode=yes" in calls


def test_startup_does_not_import_gitpython():
    code = ("import sys, types\n"
            "try:\n    import config\n"
            "except ImportError:\n"
            "    c = types.ModuleType('config'); c.TOKEN = c.GIT_TOKEN = ''; c.ADMINS = []; c.browser_cookies = False\n"
            "    sys.modules['config'] = c\n"
            "import main\n"
            "assert 'git' not in sys.modules, 'GitPython loaded at startup'\n"
            "import commands.update_bot as ub; ub._git()\n"
            "assert 'git' in sys.modules\n")
    out = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
