"""
The edit pipeline inside the bot (docs/EDIT_PIPELINE_DESIGN.md section 8, "Jetson"), the fallback by hand: /bake_cards
runs card_bake.run under one Lock shared with update_stats (queued commands run concurrently) and posts its report where
the command ran, a reminder's FakeInteraction included. The live runner is TTSCardMaker's card-edits.yml Action; nothing
here queues a bake on its own. card_bake and the fork's psd-tools are imported inside the bake, so the bot starts without
them and refuses with one line.
"""
from io import BytesIO
from threading import Lock
from traceback import format_exc
from typing import Optional

from discord import File

from commands.utils import queue_file, queue_message, say, split_long_message
from global_config import CARD_REPO

# ========================
# Configuration
# ========================
REPORT_FILE = "card_bake_report.md"             # the full report, attached in Discord

# ========================
# Global State
# ========================
bake_lock = Lock()                              # update_stats and the bake never run together


# ========================
# Helper Functions
# ========================
def _summary(text: str) -> str:
    """The report without its <details> blocks (tracebacks stay in the attached file), the title in bold."""
    lines, hidden = [], False
    for line in text.splitlines():
        if line.startswith("<details"):
            hidden = True
        elif line.startswith("</details>"):
            hidden = False
        elif not hidden:
            lines.append(line)
    if lines and lines[0].startswith("# "):
        lines[0] = f"**{lines[0][2:]}**"
    return "\n".join(lines).strip()


def _post(interaction, text: str, commits: list) -> None:
    """The summary as messages with the full report attached to the last, then the commit links."""
    if interaction is None:
        print(text, end="")
        if commits:
            print("Commits: " + ", ".join(commits))
        return
    chunks = split_long_message(_summary(text))
    for chunk in chunks[:len(chunks) - 1]:
        queue_message(interaction, chunk)
    queue_file(interaction, File(BytesIO(text.encode("utf-8")), filename=REPORT_FILE), content=chunks[len(chunks) - 1])
    if commits:
        queue_message(interaction, "Commits: " + ", ".join(f"<https://github.com/{CARD_REPO}/commit/{sha}>" for sha in commits))


# ========================
# Public API Functions
# ========================
def bake_cards(interaction, card: Optional[str] = None, dry_run: bool = True) -> None:
    """
    Bake the website's pending PSD edits (card_bake.run) under bake_lock and post the report where the command ran.

    Args:
        interaction: Discord interaction or a reminder's FakeInteraction (None prints the report)
        card: Only this card (matched like the authors file), else every card with a PSD-kind edit
        dry_run: Render and check, commit nothing
    """
    try:
        import commands.card_bake as card_bake      # lazy: the fork's psd-tools and requests
        import commands.psd_edit as psd_edit
    except ImportError as error:
        say(interaction, f"Card bake unavailable: {error}")
        return
    if not psd_edit.text_engine():
        say(interaction, "Card bake refused: this Python has no text engine (install the fork's psd-tools[text] in the bot venv).")
        return
    if bake_lock.locked():
        say(interaction, "Waiting for the running card update or bake to finish...")
    with bake_lock:
        say(interaction, f"Baking the website's card edits{' (dry run)' if dry_run else ''}...")
        try:
            text, commits = card_bake.run(CARD_REPO, "", dry_run, card)
        except Exception as error:                  # noqa: BLE001 - one line in Discord, the traceback on stdout
            print(f"Card bake failed:\n{format_exc()}")
            say(interaction, f"Card bake failed: {type(error).__name__}: {error}")
            return
    _post(interaction, text, commits)


# ========================
# Module Exports
# ========================
__all__ = [
    'bake_cards',
    'bake_lock',
]
