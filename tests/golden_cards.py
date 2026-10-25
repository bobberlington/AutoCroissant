"""Parser golden cards: the PSDs, the frozen icon vocabulary and the parse that both the test and make_goldens.py use."""
import json
from pathlib import Path

# ========================
# Configuration
# ========================
GOLDENS = Path(__file__).resolve().parent / "goldens"
VOCAB = GOLDENS / "_vocab.json"          # Types/, Types/Stars/ and the stoogedings manifest when the goldens were made
CERTIFIED = ("Auxiliary/Minions/Mini_Doomer.psd",                       # validation-and-qa section 3
             "Field/1 Stars/The_Freezer.psd",
             "Creatures/World Of Warcraft/5 Stars/The_Lich_King.psd")
CARDS = CERTIFIED + (
    "Creatures/Kirby/3 Stars/Pix.psd",                                  # 3 creatures
    "Creatures/FTL/5 Stars/Royal_Eradicator_Main_Cannon.psd",           # atk 10 + 2 (STATS TOO HIGH); title in 3 layers
    "Items/Heal/2 Stars/A_Snack.psd",                                   # hidden titles of 2 duplicates: no alt_names
    "Auxiliary/Minions/Anubisath_Guardian.psd",                         # empty ability; card_data's edited Mini Doomer
    "Auxiliary/Minions/God_Tamer.psd",                                  # 5 stars in a Stars group (none until 2026-10-22), ability excluded
    "Auxiliary/Status Effects/Debuffs/1 Stars/Poison.psd",              # status-effect folder
    "Auxiliary/Rulebook/20_Creature_Types.psd",                         # rule page: pictures named after icons read as nothing
    "Auxiliary/Rulebook/16_The_Field.psd",                              # row grouping
    "N.M.E/Termination_Shock.psd",                                      # star count (STAR_CORE_ALPHA)
)


# ========================
# Helpers
# ========================
def golden_file(rel: str) -> Path:
    return GOLDENS / (rel[:-4].replace("/", "__").replace(" ", "_") + ".json")


def parser():
    """A PSDParser on an empty database with the frozen vocabulary (fresh parse: no author or series preservation)."""
    from commands.psd_analyzer import IconVocabulary, PSDParser, StatsDatabase
    vocab = json.loads(VOCAB.read_text(encoding="utf-8"))
    db = StatsDatabase()
    db.all_types = list(vocab["all_types"])
    return PSDParser(db.all_types, db, IconVocabulary.build(db.all_types, vocab["stars"], vocab["manifest"]))


def parse(psd_parser, clone: Path, rel: str) -> dict:
    """Everything the golden pins: to_dict(), the hidden titles and the validator's problems."""
    from commands.psd_analyzer import CardValidator
    card = psd_parser.parse(str(clone / rel), rel)
    out = {"card": card.to_dict(), "alt_names": card.alt_names}
    out["validator"] = CardValidator.validate(card)
    return out
