"""Non-interactive git: GIT_TERMINAL_PROMPT=0 and ssh BatchMode on every git call, ssh options kept; lazy GitPython."""
import asyncio
import os
import subprocess
from subprocess import CompletedProcess
from types import SimpleNamespace

import pytest

import commands.git_commit as gc
import commands.repo_history as rh
import commands.update_bot as ub


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
    monkeypatch.setattr(rh, "_ssh_cache", {})
    return tmp_path


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


def test_core_ssh_command_is_kept(clean_git):
    repo = _repo(clean_git / "r", ("core.sshCommand", "ssh -i /keys/bot"))
    assert rh.ssh_command(str(repo)) == "ssh -i /keys/bot -o BatchMode=yes"
    assert rh.ssh_command(str(clean_git / "missing")) == "ssh -o BatchMode=yes"      # global config only


def test_failed_lookup_sets_no_ssh_command(clean_git, monkeypatch):
    repo = _repo(clean_git / "r", ("core.sshCommand", "ssh -i /keys/bot"))
    real = rh.run
    def hang(cmd, **kw):
        raise subprocess.TimeoutExpired(cmd, 10)
    monkeypatch.setattr(rh, "run", hang)
    assert rh.ssh_command(str(repo)) is None                            # git keeps its -i key
    assert rh.batch_overrides(str(repo)) == {"GIT_TERMINAL_PROMPT": "0"}
    monkeypatch.setenv("GIT_SSH_COMMAND", "ssh -i /keys/env")           # env wins without a lookup
    assert rh.ssh_command(str(repo)) == "ssh -i /keys/env -o BatchMode=yes"
    monkeypatch.delenv("GIT_SSH_COMMAND")
    monkeypatch.setattr(rh, "run", real)                                # the failure wasn't cached
    assert rh.ssh_command(str(repo)) == "ssh -i /keys/bot -o BatchMode=yes"


# ========================
# Every caller uses it
# ========================
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

