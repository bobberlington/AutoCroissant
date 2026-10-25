"""The Discord commands that pushed to the card repo, retired 2026-10-25 (branch archive/discord-edits), and /push (step 10)."""
import pytest

import commands.analytics as an


A_REMINDER_RUNNING_ONE_SAYS_IT_WAS_RETIRED_CASES = ["/bake_cards dry_run:False", "/alias", "/update_metadata query:x", "/push"]


def test_a_reminder_running_one_says_it_was_retired(monkeypatch):
    for command in A_REMINDER_RUNNING_ONE_SAYS_IT_WAS_RETIRED_CASES:
        monkeypatch.setattr(an, "slash_registry", {"query": lambda interaction, query: None})
        with pytest.raises(ValueError, match=r"was retired \("):
            an.resolve_command(command)
        with pytest.raises(ValueError, match="is not a bot command"):
            an.resolve_command("/nope")
