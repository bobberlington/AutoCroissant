"""
card_standards (guard 2 of the push guards): the rules on real cards of the clone (as of 6e50c3ee, from its history; the counts of
docs/Campaigns/PUSH_GUARDS.md section 6 per card, R3 and R4 without the template layout dropped 2026-10-23), the owner's
waivers, "(new)" against the old blob, R4's new-card and regression cases, and the two 0-pixel fixes on copies: R11's
rename (told until the owner adds it to AUTO_FIX; only the names change, byte for byte) and R9's Txt2 story sync (Clot White before main's Txt2 repair, from the clone's history). The badge vocabulary is pinned against the
glyph names of MDW/Fonts/stoogedings.ttf.
"""
from io import BytesIO
from pathlib import Path

import pytest
from psd_tools import PSDImage

import commands.card_standards as cs
import commands.psd_edit as psd_edit
from commands.card_check import clone_vocabulary
from conftest import at_rev, story_sync

# ========================
# Configuration
# ========================
MINI_DOOMER = "Auxiliary/Minions/Mini_Doomer.psd"
CLOT_WHITE = "Auxiliary/Minions/Clot_White.psd"
MEASURED = "6e50c3ee"                                        # the whole-tree run of PUSH_GUARDS.md 6 (CASES)
STALE_TXT2 = "d81631039734ba3c90683a68ced5de20f65304a3"   # Clot White's title story out of sync (7b48db05 repaired it)
CASES = (   # card, rules told to the author, rules waived (the 6e50c3ee whole-tree run)
    ("Items/Active/2 Stars/Dirty_Bed.psd", {"R1"}, set()),
    ("Creatures/Kirby/3 Stars/Skullord.psd", {"R5"}, set()),      # its Ability outside Text and title in Stat Txt: layout
    ("Auxiliary/Items/Lit_Bob-omb.psd", set(), {"R3"}),           # 2 visible Ability layers (Q2)
    ("Auxiliary/Minions/Shadow_Duelist.psd", {"R3"}, set()),      # its Ability layer hidden
    ("Creatures/Other/4 Stars/Sonic.psd", set(), {"R4"}),          # hidden Name, the title a pixel layer
    ("Creatures/Pandas/2 Stars/Pandanoko.psd", {"R1"}, {"R4"}),    # rasterized Name
    ("Auxiliary/Items/Nasty_Patty.psd", {"R6"}, set()),
    ("Auxiliary/Status Effects/Debuffs/1 Stars/Blindness.psd", {"R2"}, set()),
    ("Items/Equip/2 Stars/Fire_Staff.psd", set(), {"R5"}),
    ("Creatures/Other/5 Stars/Baba.psd", {"R1"}, {"R4", "R7"}),
    ("Creatures/Isaac/3 Stars/Tainted_Lazarus.psd", set(), {"R2", "R3"}),
)


# ========================
# Fixtures
# ========================
@pytest.fixture(scope="module")
def context(clone):
    from commands.font_check import FontIndex
    chart = psd_edit.status_fills(PSDImage.open(str(clone / psd_edit.STATUS_CHART)))
    return cs.Context(frozenset(clone_vocabulary(str(clone)).all_types), FontIndex(str(clone / "MDW/Fonts")), chart,
                      psd_edit.story_sync())


def _open(clone, rel):
    """The card as measured (MEASURED), not the working tree: later standardization commits change the live files."""
    return PSDImage.open(BytesIO(at_rev(clone, MEASURED, rel)))


def _codes(issues, waived: bool) -> set:
    return {i.code for i in issues if bool(i.waived) == waived and not i.fix}


# ========================
# Vocabulary and classes
# ========================
def test_badge_tags_are_the_fonts_glyphs(clone):
    """badge_<tag>, "/" as "_" and "-" as "m" (build_stoogedings.badge_glyph): reversible, so the font is the source."""
    from fontTools.ttLib import TTFont
    font = clone / "MDW/Fonts/stoogedings.ttf"
    if not font.exists():
        pytest.skip("no stoogedings.ttf")
    glyphs = [n for n in TTFont(str(font)).getGlyphOrder() if n.startswith("badge_")]
    tags = {("-" + n[7:] if n.startswith("badge_m") else n[6:]).replace("_", "/") for n in glyphs}
    assert tags == set(cs.BADGE_TAGS) and len(cs.BADGE_TAGS) == len(set(cs.BADGE_TAGS)) == 341


def test_kinds_and_keys():
    assert [cs.kind_of(p) for p in ("Auxiliary/Rulebook/02_A.psd", "Auxiliary/Creatures/X.psd", "Auxiliary/Items/X.psd",
                                     "Auxiliary/Minions/X.psd", "Auxiliary/Status Effects/Buffs/1 Stars/X.psd", "N.M.E/X.psd",
                                     "Field/1 Stars/X.psd", "Items/Heal/2 Stars/X.psd", "Creatures/Isaac/3 Stars/X.psd")] == [
        "rule", "aux creature", "aux item", "minion", "status", "nme", "field", "item", "creature"]
    assert cs.card_key("Creatures/Isaac/3 Stars/Tainted_Lazarus.psd") == "tainted lazarus"
    assert set(cs.SCOPE) >= set(cs.STARS_KINDS)


def test_every_line_reads_the_paragraph_runs():
    class Layer:
        def __init__(self, value):
            self.engine_dict = {"ParagraphRun": {"RunArray": [{"ParagraphSheet": {"Properties": {"EveryLineComposer": value}}}]}}
    assert cs.every_line(Layer(True)) and not cs.every_line(Layer(False)) and not cs.every_line(object())


def test_rule_pages_are_never_checked(clone, context):
    rel = "Auxiliary/Rulebook/11_Abilities.psd"
    assert cs.check(_open(clone, rel), rel, context) == []


# ========================
# The rules on real cards
# ========================
@pytest.mark.parametrize("rel,told,waived", CASES, ids=[Path(c[0]).stem for c in CASES])
def test_rules_on_real_cards(clone, context, rel, told, waived):
    issues = cs.check(_open(clone, rel), rel, context)
    assert _codes(issues, False) == told and _codes(issues, True) == waived, [(i.code, i.text, i.waived) for i in issues]
    assert all(i.code != "error" for i in issues)


def test_wording(clone, context):
    rel = "Creatures/Kirby/3 Stars/Skullord.psd"
    lines = [i.line() for i in cs.reported(cs.check(_open(clone, rel), rel, context))]
    assert lines == ['"Blazing" is an overlay text layer, not an Ability FX: layer']
    rel = "Items/Active/2 Stars/Dirty_Bed.psd"
    assert [i.line() for i in cs.check(_open(clone, rel), rel, context)] == ["timing tag (1GCD) not set in the stoogedings badge font"]
    for rel, code, line in (("Auxiliary/Minions/Shadow_Duelist.psd", "R3", "no visible Ability text layer (hidden or missing)"),
                            ("Creatures/World Of Warcraft/4 Stars/Twin_Emperors.psd", "R3", "5 visible Ability layers (one is the standard)"),
                            ("Items/Equip/2 Stars/Man_Serpent_Shield.psd", "R4", "the Name layer is hidden"),
                            ("Creatures/Other/5 Stars/Tyler.psd", "R4", "the Name layer is rasterized")):   # a smart object
        assert [i.line() for i in cs.check(_open(clone, rel), rel, context) if i.code == code] == [line], rel


def test_r7_is_told_in_the_standards_post(clone, context, monkeypatch):
    """Fonts are told in the standards channel too (owner, 2026-10-22); BabaUI stays waived (L13)."""
    rel = "Creatures/Other/5 Stars/Baba.psd"
    (font,) = [i for i in cs.check(_open(clone, rel), rel, context) if i.code == "R7"]
    assert font.items == ("BabaUI",) and font.discord and font.waived and font not in cs.reported([font])
    monkeypatch.setattr(cs, "check_psd", lambda psd, fonts: {"BabaUI": ["Hp"], "ComicSansMS": ["Ability"]})
    (font,) = [i for i in cs.check(_open(clone, rel), rel, context) if i.code == "R7"]
    assert font.discord and not font.waived and cs.reported([font]) == [font]
    assert font.line() == "fonts BabaUI and ComicSansMS not in MDW/Fonts"


def test_without_the_chart_every_overlay_is_told(clone, context):
    rel = "Items/Equip/2 Stars/Fire_Staff.psd"
    bare = cs.Context(context.types, None, None)
    assert _codes(cs.check(_open(clone, rel), rel, bare), False) == {"R5"}


def test_new_marks_against_the_old_blob(clone, context):
    rel = "Creatures/Kirby/3 Stars/Skullord.psd"
    psd = _open(clone, rel)
    assert not any(i.new for i in cs.check(psd, rel, context, old=_open(clone, rel)))
    clean = _open(clone, "Creatures/Other/3 Stars/Niko.psd")      # meets every rule
    assert cs.reported(cs.check(clean, "Creatures/Other/3 Stars/Niko.psd", context)) == []
    marked = cs.check(psd, rel, context, old=clean)
    assert {i.code for i in marked if i.new and not i.waived} == {"R5"}
    assert all(i.line().endswith(" (new)") for i in marked if i.new)


def _saved(psd, path) -> PSDImage:
    psd_edit.save_psd(psd, str(path), update_preview=False)
    return PSDImage.open(str(path))


def test_r3_r4_real_issues_new_cards_and_regressions(clone, context, tmp_path):
    """R3 and R4 keep the real issues: no visible Ability layer, and a Name layer the title can't be read from (told for
    a new card or a push that causes it; an existing card's waived). A title named otherwise is the developer's layout."""
    rel = "Creatures/Other/4 Stars/Sonic.psd"
    (name,) = [i for i in cs.check(_open(clone, rel), rel, context, added=True) if i.code == "R4"]
    assert not name.waived and cs.reported([name]) == [name]
    old = _open(clone, MINI_DOOMER)
    hidden = _open(clone, MINI_DOOMER)
    psd_edit.name_layer(hidden).visible = False
    psd_edit.ability_layer(hidden).visible = False
    issues = cs.check(_saved(hidden, tmp_path / "hidden.psd"), MINI_DOOMER, context, old=old)
    assert [(i.code, i.line()) for i in cs.reported(issues) if i.code in ("R3", "R4")] == [
        ("R3", "no visible Ability text layer (hidden or missing) (new)"), ("R4", "the Name layer is hidden (new)")]
    renamed = _open(clone, MINI_DOOMER)
    psd_edit.name_layer(renamed).name = "Title"
    assert psd_edit.read_name(renamed, "Mini_Doomer") is None
    issues = cs.check(_saved(renamed, tmp_path / "renamed.psd"), MINI_DOOMER, context, old=old, added=True)
    assert not [i for i in issues if i.code in ("R3", "R4")]


def test_problems_ok_cards_are_waived_by_class_only(clone, context):
    """Pix, Tainted Lazarus and Twin Emperors (problems_ok in Card Edits.json, for parser problems) report nothing today,
    any issue waived by an owner's class decision: problems_ok itself waives no standard."""
    for rel in ("Creatures/Kirby/3 Stars/Pix.psd", "Creatures/Isaac/3 Stars/Tainted_Lazarus.psd",
                "Creatures/World Of Warcraft/4 Stars/Twin_Emperors.psd"):
        issues = cs.check(_open(clone, rel), rel, context)
        assert cs.reported(issues) == [], rel
        assert {issue.waived for issue in issues} <= {cs.WAIVED[code]["why"] for code in cs.WAIVED}, rel
    assert "cards_ok" not in cs.Context.__dataclass_fields__


# ========================
# The fixes (copies)
# ========================
def test_r11_is_told_until_the_owner_says_yes(clone, context, tmp_path):
    """The owner's rule (2026-10-22): easy fixes only, today only R9's Txt2 sync. R11 is a remaining issue: told, not
    fixed (the rename is built: the tests below run it as AUTO_FIX will once the owner adds it)."""
    assert cs.AUTO_FIX == ("R9",) and set(cs.AUTO_FIX) <= set(cs.FIXABLE)
    edited = _open(clone, MINI_DOOMER)
    psd_edit.name_layer(edited).name = "NAME"
    pushed = tmp_path / "pushed.psd"
    psd_edit.save_psd(edited, str(pushed), update_preview=False)
    issues = cs.check(PSDImage.open(str(pushed)), MINI_DOOMER, context)
    (rule,) = [i for i in issues if i.code == "R11"]
    assert not rule.fix and rule in cs.reported(issues)
    assert cs.fix(pushed.read_bytes(), MINI_DOOMER, issues, str(tmp_path / "work")).data == b""


def test_r11_rename_changes_only_the_names(clone, context, tmp_path, monkeypatch):
    monkeypatch.setattr(cs, "AUTO_FIX", ("R9", "R11"))
    edited = _open(clone, MINI_DOOMER)
    psd_edit.name_layer(edited).name = "NAME"
    pushed = tmp_path / "pushed.psd"
    psd_edit.save_psd(edited, str(pushed), update_preview=False)
    issues = cs.check(PSDImage.open(str(pushed)), MINI_DOOMER, context)
    (rule,) = [i for i in issues if i.code == "R11"]
    assert rule.fix and rule.items == ("'NAME' -> 'Name'",) and rule not in cs.reported(issues)
    parser = clone_vocabulary(str(clone)).parser()
    fixed = cs.fix(pushed.read_bytes(), MINI_DOOMER, issues, str(tmp_path / "work"), parser)
    assert fixed.refused == [] and fixed.applied == [rule.line()] and fixed.data
    out = tmp_path / "fixed.psd"
    out.write_bytes(fixed.data)
    reopened = PSDImage.open(str(out))
    assert cs.inexact_names(reopened) == [] and psd_edit.name_layer(reopened).name == "Name"
    assert reopened._record.image_data.data == PSDImage.open(str(pushed))._record.image_data.data   # the composite kept
    assert cs.rename(str(out), str(tmp_path / "again.psd")) == []                                  # idempotent


def test_r11_refused_when_the_rename_changes_the_bots_reading(clone, context, tmp_path, monkeypatch):
    """The parser misses an "ability " layer: the rename would change the card's data, so the reparse gate keeps it a
    report (the bot never changes data on its own)."""
    monkeypatch.setattr(cs, "AUTO_FIX", ("R9", "R11"))
    edited = _open(clone, MINI_DOOMER)
    psd_edit.ability_layer(edited).name = "ability "
    pushed = tmp_path / "pushed.psd"
    psd_edit.save_psd(edited, str(pushed), update_preview=False)
    issues = cs.check(PSDImage.open(str(pushed)), MINI_DOOMER, context)
    fixed = cs.fix(pushed.read_bytes(), MINI_DOOMER, issues, str(tmp_path / "work"), clone_vocabulary(str(clone)).parser())
    assert fixed.data == b"" and fixed.applied == [] and [code for code, _ in fixed.refused] == ["R11"]
    assert "reparse" in fixed.refused[0][1]


def test_r11_gate_refuses_any_other_change(clone, tmp_path, monkeypatch):
    edited = _open(clone, MINI_DOOMER)
    psd_edit.ability_layer(edited).name = "ability"
    pushed = tmp_path / "pushed.psd"
    psd_edit.save_psd(edited, str(pushed), update_preview=False)
    real = psd_edit.save_psd

    def sneaky(psd, path, update_preview=True, preview=None):
        psd_edit.name_layer(psd).visible = False            # a flag byte beyond the rename
        real(psd, path, update_preview=update_preview, preview=preview)
    monkeypatch.setattr(psd_edit, "save_psd", sneaky)
    with pytest.raises(psd_edit.Refused, match="beyond its name"):
        cs.rename(str(pushed), str(tmp_path / "out.psd"))


@pytest.mark.skipif(not story_sync(), reason="needs the fork's Txt2 story sync (fix/txt2-sync)")
def test_r9_story_sync_on_clot_white(clone, context, tmp_path):
    """The pushed blob, not the working tree: main's Txt2 repair (7b48db05) synced the live file, so the stale story is
    read from the commit before it."""
    rel = CLOT_WHITE
    data = at_rev(clone, STALE_TXT2, rel)
    stale = tmp_path / "Clot_White.psd"
    stale.write_bytes(data)
    issues = cs.check(PSDImage.open(str(stale)), rel, context)
    (rule,) = [i for i in issues if i.code == "R9"]
    assert rule.fix and rule.items == ("Name",) and not cs.reported(issues)
    fixed = cs.fix(data, rel, issues, str(tmp_path / "work"), clone_vocabulary(str(clone)).parser())
    assert fixed.refused == [] and fixed.applied == [rule.line()]
    out = tmp_path / "fixed.psd"
    out.write_bytes(fixed.data)
    assert psd_edit.stale_stories(PSDImage.open(str(out))) == []
    assert not list((tmp_path / "work").iterdir())          # temporaries removed


def test_two_cards_of_one_key_fix_in_separate_files(tmp_path, monkeypatch):
    """The process pool shares out/work: Same.psd in two folders, fixed at once (one inside the other's rename here),
    keep their own bytes and leave no temporary behind."""
    calls = []

    def rename(path, out):
        calls.append(path)
        if len(calls) == 1:                                 # the other card's whole fix while this one is mid-way
            other = cs.fix(b"B" * 1000, "Creatures/Other/Same.psd", [cs.Issue("R11", "x", fix=True)], str(tmp_path / "work"))
            assert other.data == b"B" * 1000 + b"!"
        with open(path, "rb") as fh:
            data = fh.read()
        with open(out, "wb") as fh:
            fh.write(data + b"!")
        return ["renamed"]
    monkeypatch.setattr(cs, "rename", rename)
    fixed = cs.fix(b"A" * 1000, "Items/Heal/2 Stars/Same.psd", [cs.Issue("R11", "x", fix=True)], str(tmp_path / "work"))
    assert fixed.data == b"A" * 1000 + b"!" and fixed.refused == [] and len(set(calls)) == 2
    assert not list((tmp_path / "work").iterdir())


def test_nothing_fixed_on_rule_pages_or_waived(clone):
    issue = cs.Issue("R9", "x", ("Name",), fix=True)
    assert cs.fix(b"", "Auxiliary/Rulebook/02_A.psd", [issue], "unused").data == b""
    assert cs.fix(b"", "Items/X.psd", [cs.Issue("R9", "x", fix=True, waived="w")], "unused").data == b""
