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
    "Creatures/Isaac/3 Stars/Tainted_Lazarus.psd",                      # 2 creatures
    "Creatures/Kirby/3 Stars/Pix.psd",                                  # 3 creatures
    "Creatures/World Of Warcraft/4 Stars/Twin_Emperors.psd",
    "Creatures/Kirby/3 Stars/Computer_Virus.psd",                       # stats not found; display_name from a root Name layer
    "Creatures/FTL/5 Stars/Royal_Eradicator_Main_Cannon.psd",           # atk 10 + 2 (STATS TOO HIGH); title in 3 layers
    "Creatures/DST/3 Stars/WX-78.psd",                                  # title "wx-" + an image layer "78": no display_name
    "Creatures/Kirby/4 Stars/Wham_Bam_Bone.psd",
    "Creatures/Mario/3 Stars/Sniffle_Thwomp.psd",
    "Creatures/Other/2 Stars/Little_Guy.psd",
    "Creatures/Pandas/3 Stars/P.A.N.D.A._Mk_A.psd",
    "Creatures/Hollow Knight/5 Stars/The_Radiance.psd",
    "Items/Heal/2 Stars/A_Snack.psd",                                   # hidden titles of 2 duplicates
    "Items/Heal/2 Stars/Regeneration_Potion.psd",
    "Items/Heal/2 Stars/Watermelon.psd",
    "Items/Active/3 Stars/Black_Light_Bulb.psd",
    "Items/Attack/3 Stars/Shooting_Star.psd",
    "Items/Counter/4 Stars/Return_To_Sender.psd",
    "Items/Deploy/1 Stars/Poison_Circle.psd",
    "Items/Equip/4 Stars/Mom's_Knife.psd",
    "Items/Support/2 Stars/Volt_Shroom.psd",
    "Field/3 Stars/White_Space.psd",
    "Field/4 Stars/The_Maelstrom.psd",
    "Field/2 Stars/Irithyll.psd",                                       # row grouping (_sort_by_position threshold)
    "Auxiliary/Minions/Anubisath_Guardian.psd",                         # empty ability text, in ABILITY_EXCLUSIONS since 2026-10-21
    "Auxiliary/Minions/God_Tamer.psd",                                  # 5 stars in a Stars group (none until 2026-10-22), ability excluded
    "Auxiliary/Minions/Whelp.psd",
    "Auxiliary/Items/Poorly_Drawn_Cabbage.psd",
    "Auxiliary/Creatures/Militron_(No_Armor).psd",
    "Auxiliary/Status Effects/Debuffs/1 Stars/Poison.psd",
    "Auxiliary/Status Effects/Buffs/Enraged.psd",
    "Auxiliary/Rulebook/20_Creature_Types.psd",                         # rule page types (MISSPELT TYPE: tornado until 2026-10-22)
    "Auxiliary/Rulebook/00_Table_of_Contents.psd",
    "Auxiliary/Rulebook/16_The_Field.psd",                              # row grouping
    "N.M.E/Jera.psd",
    "N.M.E/The_Evil,_Fricked_Up_Cursed_Idol.psd",
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
