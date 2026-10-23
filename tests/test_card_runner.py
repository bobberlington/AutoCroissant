"""card_runner (EDIT_PIPELINE_DESIGN section 8, jetson): /bake_cards runs card_bake.run under bake_lock and posts the
report; the refusal without the fork's text engine; the command's registration, gate, help and describe strings.
card_bake.run itself is faked here (tests/test_card_bake.py covers it)."""
import ast
import asyncio
import os
import time
from threading import Thread
from types import SimpleNamespace

import pytest
from discord import File

import commands.card_bake as cb
import commands.card_runner as cr
import commands.help as help_module
import commands.psd_edit as pe
import commands.utils as utils
import main
from conftest import REPO
from global_config import CARD_REPO

SHA = "a" * 40
REPORT = ("# Card edits\n\n- Repo: `MichaelJSr/TTSCardMaker` main `abcdef12` (dry run)\n- Work dir: `/tmp/x`\n\n"
          "## Baked (1)\n- **Mini Doomer** `Auxiliary/Minions/Mini_Doomer.psd`: hp 5 -> 8; not committed (dry run)\n\n"
          "<details><summary>verdicts</summary>\n\n```\nTraceback (most recent call last):\n  boom\n```\n</details>\n\n"
          "Took 3 s.\n")


# ========================
# Fixtures
# ========================
@pytest.fixture(autouse=True)
def queues():
    for q in (utils.dispatch_queue, utils.edit_queue, utils.command_queue):
        q.clear()
    yield
    for q in (utils.dispatch_queue, utils.edit_queue, utils.command_queue):
        q.clear()


@pytest.fixture
def fake_run(monkeypatch):
    """card_bake.run recorded, the text engine present."""
    calls = []

    def run(repository, head="", dry_run=False, card=None):
        calls.append((repository, head, dry_run, card))
        return REPORT, [] if dry_run else [SHA]
    monkeypatch.setattr(cb, "run", run)
    monkeypatch.setattr(pe, "text_engine", lambda: True)
    return calls


def _sent() -> list:
    return [kwargs for _, kwargs in utils.dispatch_queue]


def _registered() -> dict:
    return {cmd.name: cmd for cmd in main.tree.get_commands()}


# ========================
# Tests
# ========================
def test_bake_posts_summary_and_report(fake_run):
    fake = utils.make_fake_interaction(1)
    cr.bake_cards(fake, None, True)
    assert fake_run == [(CARD_REPO, "", True, None)]
    sent = _sent()
    assert sent[0]["content"] == "Baking the website's card edits (dry run)..."
    assert isinstance(sent[1]["file"], File) and sent[1]["file"].filename == cr.REPORT_FILE
    assert sent[1]["file"].fp.read().decode("utf-8") == REPORT                      # the whole report, tracebacks included
    summary = sent[1]["content"]
    assert summary.startswith("**Card edits**") and "## Baked (1)" in summary and "hp 5 -> 8" in summary
    assert "Traceback" not in summary and "<details" not in summary
    assert len(sent) == 2                                                             # a dry run has no commits


def test_bake_links_its_commits(fake_run):
    cr.bake_cards(utils.make_fake_interaction(1), "Mini Doomer", False)
    assert fake_run == [(CARD_REPO, "", False, "Mini Doomer")]
    assert _sent()[2]["content"] == f"Commits: <https://github.com/{CARD_REPO}/commit/{SHA}>"


def test_bake_prints_without_an_interaction(fake_run, capsys):
    cr.bake_cards(None)
    assert capsys.readouterr().out.endswith(REPORT) and _sent() == []


def test_bake_refuses_without_the_text_engine(monkeypatch):
    monkeypatch.setattr(pe, "text_engine", lambda: False)
    monkeypatch.setattr(cb, "run", lambda *a, **k: pytest.fail("ran without a text engine"))
    cr.bake_cards(utils.make_fake_interaction(1))
    assert len(_sent()) == 1 and "no text engine" in _sent()[0]["content"]


def test_bake_failure_is_one_line(monkeypatch, capsys):
    def run(*args, **kwargs):
        raise RuntimeError("GitHub 502")
    monkeypatch.setattr(cb, "run", run)
    monkeypatch.setattr(pe, "text_engine", lambda: True)
    cr.bake_cards(utils.make_fake_interaction(1))
    assert _sent()[len(_sent()) - 1]["content"] == "Card bake failed: RuntimeError: GitHub 502"
    assert "Traceback" in capsys.readouterr().out and not cr.bake_lock.locked()


def test_bake_waits_for_the_lock(fake_run):
    """The Lock update_stats holds for its whole run: the bake starts only after it."""
    fake = utils.make_fake_interaction(1)
    cr.bake_lock.acquire()
    try:
        worker = Thread(target=cr.bake_cards, args=(fake,))
        worker.start()
        worker.join(0.3)
        assert worker.is_alive() and fake_run == []
        assert _sent()[0]["content"] == "Waiting for the running card update, bake or font check to finish..."
    finally:
        cr.bake_lock.release()
    worker.join(10)
    assert not worker.is_alive() and len(fake_run) == 1


def test_startup_sweeps_stale_temp_psds(tmp_path, capsys):
    """A cut-off run's tmp*.psd (mkstemp's names) older than an hour go, in one line; young ones, other names and
    folders stay; nothing goes while update_stats, a bake or a font check holds bake_lock."""
    old = time.time() - 2 * cr.TEMP_AGE
    names = ("tmpab_c1234.psd", "tmpzz99yy88.psd", "tmpyoung123.psd", "tmpab_c1234.png", "card.psd", "tmpshort.psd")
    for name in names:
        (tmp_path / name).write_bytes(b"x" * 1000)
        if name != "tmpyoung123.psd":
            os.utime(tmp_path / name, (old, old))
    (tmp_path / "tmpdir12345.psd").mkdir()
    os.utime(tmp_path / "tmpdir12345.psd", (old, old))
    cr.bake_lock.acquire()
    try:
        assert cr.sweep_temp_psds(str(tmp_path)) == 0 and len(list(tmp_path.iterdir())) == 7
    finally:
        cr.bake_lock.release()
    assert cr.sweep_temp_psds(str(tmp_path)) == 2 and not cr.bake_lock.locked()
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(set(names) - {"tmpab_c1234.psd", "tmpzz99yy88.psd"}
                                                                | {"tmpdir12345.psd"})
    assert capsys.readouterr().out == f"Removed 2 stale temp PSDs (0 MB) from {tmp_path}\n"
    assert cr.sweep_temp_psds(str(tmp_path)) == 0 and capsys.readouterr().out == ""


def test_card_bake_is_imported_lazily():
    """The bot starts without the fork's psd-tools: card_runner's top-level imports never reach card_bake or psd_edit."""
    tree = ast.parse((REPO / "commands" / "card_runner.py").read_text(encoding="utf-8"))
    top = {n.module if isinstance(n, ast.ImportFrom) else a.name for n in tree.body
           if isinstance(n, (ast.Import, ast.ImportFrom)) for a in n.names}
    assert not any(name.startswith(("commands.card_bake", "commands.psd_edit", "psd_tools")) for name in top)


def test_command_registered_gated_and_documented():
    """/bake_cards: the 52nd command, admin only, defaults stated the same way in main.py and help.py."""
    cmds = _registered()
    assert len(cmds) == 52 and cmds["bake_cards"].callback.__name__ == "slash_bake_cards"
    assert "(admin only)" in cmds["bake_cards"].description and len(cmds["bake_cards"].description) <= 100
    params = {p.name: p for p in cmds["bake_cards"].parameters}
    assert params["dry_run"].default is True and params["dry_run"].description.endswith("Default = True.")
    assert params["card"].default is None and params["card"].description.endswith("Default = every card with a pending PSD edit.")
    entry = help_module.stats_commands["/bake_cards [card] [dry_run]:"]
    assert "(admin only)" in entry and "/update_stats" in entry
    assert "(default: every card with a pending PSD edit)" in entry.split("\n")[1] and "(default: True)" in entry.split("\n")[2]


def test_command_denies_non_admins():
    sent = []

    async def send_message(content=None, **kwargs):
        sent.append(content)
    interaction = SimpleNamespace(user=SimpleNamespace(id=1), response=SimpleNamespace(send_message=send_message))
    asyncio.run(_registered()["bake_cards"].callback(interaction))
    assert sent == ["You do not have permission."] and utils.command_queue == utils.command_queue.__class__()
