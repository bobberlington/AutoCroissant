"""The parse and psd_details share one PSDImage.open per card PSD in the data build and the card check (two opens were
most of a re-parse's CPU): the shared open reads as two separate opens do."""
import json
import pickle
from datetime import datetime, timezone

import commands.card_data as cd
import commands.card_db as cdb
import commands.psd_details as pd
from commands.card_models import card_name
from commands.psd_parser import CardValidator
from golden_cards import parser as golden_parser
from helpers import DOOMER

SAME = (DOOMER,                                                             # the identity cards: golden oddities
        "Creatures/World Of Warcraft/5 Stars/The_Lich_King.psd",           # font tokens
        "Creatures/Kirby/3 Stars/Computer_Virus.psd",                      # no stats; title from a root Name layer
        "Creatures/FTL/5 Stars/Royal_Eradicator_Main_Cannon.psd",          # atk 12; title in 3 layers
        "Items/Heal/2 Stars/A_Snack.psd",                                  # hidden titles
        "N.M.E/Termination_Shock.psd")                                     # drawn stars


class _Frozen(datetime):
    @classmethod
    def now(cls, tz=None):
        return datetime(2026, 10, 23, tzinfo=timezone.utc)


def test_one_open_reads_as_two(clone, monkeypatch):
    """The build's shared open gives the card (pickle, db/ line, notes) and details record that two opens of the file give."""
    monkeypatch.setattr(pd, "datetime", _Frozen)
    psd_parser = golden_parser()
    build = cd.CardBuild(psd_parser.db)
    build.parser = psd_parser
    for rel in SAME:
        path, name = str(clone / rel), card_name(rel)
        card = build._parse(path, rel, name, rel)
        alone = psd_parser.parse(path, rel)
        assert pickle.dumps(card) == pickle.dumps(alone), rel
        assert cdb.stats_text({name: card}) == cdb.stats_text({name: alone}), rel
        assert CardValidator.validate(card) == CardValidator.validate(alone), rel
        record = pd.gather(path, rel, name, None, psd_parser.db.all_types)
        assert json.dumps(build.details.pop(rel)) == json.dumps(record), rel
