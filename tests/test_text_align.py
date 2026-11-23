"""
commands/text_align: an edit lined up by paragraphs, then words (owner 2026-11-03, Pot Full of Spiders!).
Against the website mock-up's port (TTSCardMaker site/assets/textalign.js, from $AC_SITE_SRC or the clone).
"""
import json
from pathlib import Path
from subprocess import run as run_process

from commands import text_align as ta
from commands.psd_edit import aligned_parts
from conftest import SITE_SRC

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


def test_a_line_join_or_split_is_one_changed_break():
    """Rule 5: a break made a space, or a space a break, is the one change, every letter kept (11-15: The Structure's
    joined line 1 took line 2's font); a join that also types a word lines up by paragraphs as before."""
    old = "+1 ATK\nAttacks inflict 5T1 Poison."
    assert _changes(old, old.replace("\n", " ")) == [("\n", " ")]
    assert _changes(old, old.replace(" inflict", "\rinflict")) == [(" ", "\r")]
    assert ta.blocks(old, old.replace("\n", " "))[:-1] == [(0, 0, 6), (7, 7, 27)]
    assert ta.break_ops(old, old.replace("\n", " and ")) is None


TEXTALIGN_JS = SITE_SRC / "site" / "assets" / "textalign.js"
EDITS = json.loads((Path(__file__).resolve().parent / "fixtures" / "text_align_edits.json").read_text(encoding="utf-8"))["edits"]


def test_the_mock_ups_port_lines_up_as_the_bake(tmp_path):
    """The website mock-up's port (textalign.js) gives real card edits the bake's segments, parts (psd_edit
    aligned_parts) and moves, its UTF-16 offsets aside (emoji)."""
    (tmp_path / "edits.json").write_text(json.dumps(EDITS), encoding="utf-8")
    script = tmp_path / "probe.mjs"
    script.write_text(f"const T = await import({json.dumps(TEXTALIGN_JS.as_uri())});\n"
                      f"const edits = JSON.parse((await import('fs')).readFileSync({json.dumps(str(tmp_path / 'edits.json'))}, 'utf8'));\n"
                      "process.stdout.write(JSON.stringify(edits.map(e => [T.taSegments(e.old, e.new).map(s => s.slice(0, 5)), "
                      "T.taParts(e.old, e.new), T.taMoves(e.old, e.new)])));\n", encoding="utf-8")
    done = run_process(["node", str(script)], capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert done.returncode == 0, done.stderr
    for e, got in zip(EDITS, json.loads(done.stdout), strict=True):
        a, b = _utf16(e["old"]), _utf16(e["new"])
        want = [[[tag, a[i1], a[i2], b[j1], b[j2]] for tag, i1, i2, j1, j2, _ in ta.segments(e["old"], e["new"])],
                [[a[a0], a[a1], b[b0], b[b1], kept] for a0, a1, b0, b1, kept in aligned_parts(e["old"], e["new"])],
                [[[a[s0], a[e0]], [b[t0], b[t1]], [[a[s0 + i] - a[s0], b[t0 + j] - b[t0], a[s0 + i + n] - a[s0 + i]] for i, j, n in kept]]
                 for (s0, e0), (t0, t1), kept in ta.moves(e["old"], e["new"])]]
        assert got == want, (e["card"], e["kind"])
    assert sum(bool(ta.moves(e["old"], e["new"])) for e in EDITS) >= 5


def _utf16(text: str) -> list:
    """Per code point offset of text (and its end): the UTF-16 offset, as JS strings count."""
    out = [0]
    for c in text:
        out.append(out[-1] + (2 if ord(c) > 0xFFFF else 1))
    return out
