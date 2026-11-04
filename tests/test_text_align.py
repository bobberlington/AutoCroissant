"""
commands/text_align: an edit lined up by paragraphs, then words (owner 2026-11-03, Pot Full of Spiders!).
"""
from commands import text_align as ta

SPIDERS = ("This creature starts with 2 spider counters and gains 2 spider counters per turn. \n"
           "When damaged, it converts each of its spider counters into Lil Spoods.\nIf it has 4 or more.")
SPIDERS_NEW = ("(OFS)/(BOT) Gain 2 spider counters.\n"
               "When damaged, spend all spider counters to summon an equal amount of Lil Spoods.\nIf it has 4 or more.")


def _changes(old, new):
    return [(old[i1:i2], new[j1:j2]) for tag, i1, i2, j1, j2, _ in ta.segments(old, new) if tag != "equal"]


def test_an_edit_lines_up_by_paragraphs_then_words():
    """The spiders edit: paragraph 2's new words replace its own (never paragraph 1's leftover letters), a kept "to"
    is never the end of "into"; no change crosses a break; "(1GCD)" -> "(1CD)" and "Poison" -> "Poisoned" keep their
    letters."""
    assert _changes(SPIDERS, SPIDERS_NEW) == [
        ("This", "(OFS)/(BOT)"), ("creature starts with", "Gain"), (" and gains 2 spider counters per turn", ""), (" ", ""),
        ("it", "spend"), ("converts each of its", "all"), ("into", "to summon an equal amount of")]
    assert [n for _, _, n in ta.blocks("(1GCD) Poison", "(1CD) Poisoned")[:-1]] == [2, 10]


def test_added_deleted_and_moved_paragraphs_are_changes_of_their_own():
    """A deleted first line goes with its break, apart from the next line's new start (Morpho Knight); a line typed
    after a soft break made a paragraph break stays in its paragraph; swapped lines are moves (Z!!!!!, Azamoth)."""
    old, new = "Gone line here.\n(1CD) Inflict 3 enemies.", "(1CD) Hit 3 enemies."
    assert _changes(old, new) == [("Gone line here.\n", ""), ("Inflict", "Hit")]
    assert _changes("+5 Max HP\x03Cannot be unequipped.", "+5.\r(1CD) Cannot be unequipped.") == [
        (" Max HP", "."), ("\x03", "\r"), ("", "(1CD) ")]
    old, new = "(WOF) Allies gain 1 Marked.\nStats start at 0.\n(1CD) Inflict 1T2 Hyper.", \
        "(1CD) Inflict 1T2 Hyper.\nStats start at 0.\n(WOF) Allies gain 1 Marked."
    moved = [(old[slice(*a)], new[slice(*b)]) for a, b, _ in ta.moves(old, new)]   # one line stays, two moved
    assert len(moved) == 2 and all(a == b for a, b in moved)
