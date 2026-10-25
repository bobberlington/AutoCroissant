"""The Discord commands that pushed to the card repo, retired 2026-10-25 (branch archive/discord-edits)."""
import pytest

import commands.analytics as an
import commands.help as help_module
import main
from commands.card_sync import READERS

RETIRED = ("update_metadata", "mass_replace", "alias", "bake_cards", "update_tts_save")


def test_gone_from_the_tree_help_and_readers():
    names = {cmd.name for cmd in main.tree.get_commands()}
    documented = " ".join(k for d in (help_module.general_commands, help_module.text_commands, help_module.card_commands,
                                      help_module.stats_commands, help_module.music_commands,
                                      help_module.card_sync_commands) for k in d)
    assert len(names) == 47 and not names & set(RETIRED) and not READERS & set(RETIRED)
    assert not any(f"/{name} " in documented or f"/{name}:" in documented for name in RETIRED)
    assert set(an.RETIRED_COMMANDS) == set(RETIRED)


@pytest.mark.parametrize("command", ["/bake_cards dry_run:False", "/alias", "/update_metadata query:x"])
def test_a_reminder_running_one_says_it_was_retired(monkeypatch, command):
    monkeypatch.setattr(an, "slash_registry", {"query": lambda interaction, query: None})
    with pytest.raises(ValueError, match=r"was retired \("):
        an.resolve_command(command)
    with pytest.raises(ValueError, match="is not a bot command"):
        an.resolve_command("/nope")
