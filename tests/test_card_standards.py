"""
card_standards (guard 2 of the push guards): the rules on real cards of the clone (as of 6e50c3ee, from its history; the counts of
docs/Campaigns/PUSH_GUARDS.md section 6 per card, R3 and R4 without the template layout dropped 2026-10-23), the owner's
waivers, "(new)" against the old blob, R4's new-card and regression cases, and the two 0-pixel fixes on copies: R11's
rename (told until the owner adds it to AUTO_FIX; only the names change, byte for byte) and R9's Txt2 story sync (Clot White before main's Txt2 repair, from the clone's history). The badge vocabulary is pinned against the
glyph names of MDW/Fonts/stoogedings.ttf.
"""
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest
from psd_tools import PSDImage

import commands.card_standards as cs
import commands.psd_edit as psd_edit
from commands.card_check import clone_vocabulary
from conftest import RING, at_rev, story_sync
from helpers import BALLOON_BOY, normalised

# ========================
# Configuration
# ========================
MINI_DOOMER = "Auxiliary/Minions/Mini_Doomer.psd"
CLOT_WHITE = "Auxiliary/Minions/Clot_White.psd"
MEASURED = "6e50c3ee"                                        # the whole-tree run of PUSH_GUARDS.md 6 (CASES)
STALE_TXT2 = "d81631039734ba3c90683a68ced5de20f65304a3"   # Clot White's title story out of sync (7b48db05 repaired it)
CASES = (   # card, rules told to the author, rules waived (the 6e50c3ee whole-tree run)
    ("Items/Active/2 Stars/Dirty_Bed.psd", {"R1"}, set()),
    ("Creatures/Kirby/3 Stars/Skullord.psd", {"R1", "R5"}, set()),   # its Ability outside Text and title in Stat Txt: layout; a plain (2 points)
    ("Auxiliary/Minions/Shadow_Duelist.psd", set(), {"R3"}),      # its Ability layer hidden; waived: in ABILITY_EXCLUSIONS
    ("Auxiliary/Minions/Abyss_Watcher.psd", set(), set()),        # hidden and empty: no ability
    ("Creatures/Other/4 Stars/Sonic.psd", set(), {"R4"}),          # hidden Name, the title a pixel layer
    ("Auxiliary/Items/Nasty_Patty.psd", {"R6"}, set()),
    ("Auxiliary/Status Effects/Debuffs/1 Stars/Blindness.psd", {"R2"}, set()),
)


# ========================
# Fixtures
# ========================
@pytest.fixture(scope="module")
def context(clone):
    chart = psd_edit.status_fills(PSDImage.open(str(clone / psd_edit.STATUS_CHART)))
    return cs.Context(frozenset(clone_vocabulary(str(clone)).all_types), chart, psd_edit.story_sync())


def _open(clone, rel):
    """The card as measured (MEASURED), not the working tree: later standardization commits change the live files."""
    return PSDImage.open(BytesIO(at_rev(clone, MEASURED, rel)))


def _codes(issues, waived: bool) -> set:
    return {i.code for i in issues if bool(i.waived) == waived and not i.fix}


# ========================
# Vocabulary and classes
# ========================
def test_badge_tags_are_the_fonts_glyphs():
    """badge_<tag>, "/" as "_" and "-" as "m" (build_stoogedings.badge_glyph): reversible, so the font is the source.
    The checkout's font (the vocabulary the bake and the guards must match), not the pin's."""
    from fontTools.ttLib import TTFont
    from conftest import CLONE
    font = CLONE / "MDW/Fonts/stoogedings.ttf"
    if not font.exists():
        pytest.skip("no stoogedings.ttf")
    glyphs = [n for n in TTFont(str(font)).getGlyphOrder() if n.startswith("badge_") and "." not in n]   # not .salt
    tags = {("-" + n[7:] if n.startswith("badge_m") else n[6:]).replace("_", "/") for n in glyphs if "_cost_" not in n}
    assert tags == set(cs.BADGE_TAGS) and len(cs.BADGE_TAGS) == len(set(cs.BADGE_TAGS)) == 1139
    costs = {n for n in glyphs if "_cost_" in n}      # badge_cost_<kind>_<number, "." as "p">: one per number and kind
    kinds = {"point": "points", "brick": "bricks", "sp": "sp", "star point": "sp", "soul": "souls", "counter": "counters"}
    assert costs == {f"badge_cost_{kinds[w.lower().rstrip('s')]}_{n.replace('.', 'p')}"
                     for n, w in (t.split(" ", 1) for t in psd_edit.COST_TAGS)} and len(costs) == 310


def test_kinds_and_keys():
    assert [cs.kind_of(p) for p in ("Auxiliary/Creatures/X.psd", "Auxiliary/Items/X.psd",
                                     "Auxiliary/Minions/X.psd", "Auxiliary/Status Effects/Buffs/1 Stars/X.psd", "N.M.E/X.psd",
                                     "Field/1 Stars/X.psd", "Items/Heal/2 Stars/X.psd", "Creatures/Isaac/3 Stars/X.psd")] == [
        "aux creature", "aux item", "minion", "status", "nme", "field", "item", "creature"]
    assert cs.card_key("Creatures/Isaac/3 Stars/Tainted_Lazarus.psd") == "tainted lazarus"
    assert set(cs.SCOPE) >= set(cs.STARS_KINDS)


# ========================
# The rules on real cards
# ========================
@pytest.mark.parametrize("rel,told,waived", CASES, ids=[Path(c[0]).stem for c in CASES])
def test_rules_on_real_cards(clone, context, rel, told, waived):
    issues = cs.check(_open(clone, rel), rel, context)
    assert _codes(issues, False) == told and _codes(issues, True) == waived, [(i.code, i.text, i.waived) for i in issues]
    assert all(i.code != "error" for i in issues)


def test_r1_reads_every_ability_layer(clone, monkeypatch):
    """R1 checks the layers the parser reads as the ability (ability_parts): Empress Bulblax lays its hidden Ability out
    again in "text 1".."text 5", its (3CD) in "text 4" (B2, 2026-10-26)."""
    path = clone / "Creatures/Other/3 Stars/Empress Bulblax.psd"
    if not path.exists():
        pytest.skip(f"{path} not in the clone")
    psd = PSDImage.open(str(path))
    assert [layer.name for layer in cs.ability_layers(psd)] == [f"text {n}" for n in range(1, 6)]
    real = cs.run_fonts

    def text_4_in(font):                                    # "text 4" set in one font (the pin's own may be either)
        def runs(layer):
            text, found = real(layer)
            return (text, [(a, b, font) for a, b, _ in found]) if layer.name == "text 4" else (text, found)
        return runs
    monkeypatch.setattr(cs, "run_fonts", text_4_in("stoogedings-Regular"))
    assert cs.rule_tags(psd) is None
    monkeypatch.setattr(cs, "run_fonts", text_4_in("SeuratPro-B"))
    issue = cs.rule_tags(psd)
    assert issue.code == "R1" and issue.items == ("(3CD)",)


def test_r1_tolerates_spaces_in_the_ability_name(clone):
    """A visible "Ability " or " Ability" layer is checked too, though the parser reads only "Ability"."""
    psd = _open(clone, "Items/Active/2 Stars/Dirty_Bed.psd")
    issue = cs.rule_tags(psd)
    layer = next(layer for layer in psd.descendants() if layer.name == "Ability")
    for name in ("Ability ", " Ability"):
        layer.name = name
        assert cs.ability_layers(psd) == [] and cs.rule_tags(psd).items == issue.items, name


def _saved(psd, path) -> PSDImage:
    psd.save(str(path), update_preview=False)
    return PSDImage.open(str(path))


def test_r3_r4_real_issues_new_cards_and_regressions(clone, context, tmp_path):
    """R3 and R4 keep the real issues: a hidden Ability layer whose words the parser reads nowhere, and a Name layer the
    title can't be read from (told for a new card or a push that causes it; an existing card's waived). An ability or a
    title named otherwise is the developer's layout."""
    rel = "Creatures/Other/4 Stars/Sonic.psd"
    (name,) = [i for i in cs.check(_open(clone, rel), rel, context, added=True) if i.code == "R4"]
    assert not name.waived and cs.reported([name]) == [name]
    old = _open(clone, MINI_DOOMER)
    hidden = _open(clone, MINI_DOOMER)
    psd_edit.name_layer(hidden).visible = False
    psd_edit.ability_layer(hidden).visible = False
    issues = cs.check(_saved(hidden, tmp_path / "hidden.psd"), MINI_DOOMER, context, old=old)
    assert [(i.code, i.line()) for i in cs.reported(issues) if i.code in ("R3", "R4")] == [
        ("R3", "the Ability layer is hidden (the bot reads no ability) (new)"), ("R4", "the Name layer is hidden (new)")]
    renamed = _open(clone, MINI_DOOMER)
    psd_edit.name_layer(renamed).name = "Title"
    psd_edit.ability_layer(renamed).name = "Effect"         # no Ability layer at all: the ability named otherwise
    assert psd_edit.read_name(renamed, "Mini_Doomer") == "Mini Doomer"   # its place says it (title rule P)
    issues = cs.check(_saved(renamed, tmp_path / "renamed.psd"), MINI_DOOMER, context, old=old, added=True)
    assert not [i for i in issues if i.code in ("R3", "R4")]


# ========================
# The fixes (copies)
# ========================
def test_r11_rename_changes_only_the_names(clone, context, tmp_path, monkeypatch):
    monkeypatch.setattr(cs, "AUTO_FIX", ("R9", "R11"))
    edited = _open(clone, MINI_DOOMER)
    psd_edit.name_layer(edited).name = "NAME"
    pushed = tmp_path / "pushed.psd"
    edited.save(str(pushed), update_preview=False)
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
    assert not reopened.has_preview()                   # saved stripped (Maximize Compatibility off, owner 2026-11-07)
    assert cs.rename(str(out), str(tmp_path / "again.psd")) == []                                  # idempotent


def test_r11_refused_when_the_rename_changes_the_bots_reading(clone, context, tmp_path, monkeypatch):
    """The parser misses an "ability " layer: the rename would change the card's data, so the reparse gate keeps it a
    report (the bot never changes data on its own)."""
    monkeypatch.setattr(cs, "AUTO_FIX", ("R9", "R11"))
    edited = _open(clone, MINI_DOOMER)
    psd_edit.ability_layer(edited).name = "ability "
    pushed = tmp_path / "pushed.psd"
    edited.save(str(pushed), update_preview=False)
    issues = cs.check(PSDImage.open(str(pushed)), MINI_DOOMER, context)
    fixed = cs.fix(pushed.read_bytes(), MINI_DOOMER, issues, str(tmp_path / "work"), clone_vocabulary(str(clone)).parser())
    assert fixed.data == b"" and fixed.applied == [] and [code for code, _ in fixed.refused] == ["R11"]
    assert "reparse" in fixed.refused[0][1]


def test_r11_gate_refuses_any_other_change(clone, tmp_path, monkeypatch):
    edited = _open(clone, MINI_DOOMER)
    psd_edit.ability_layer(edited).name = "ability"
    pushed = tmp_path / "pushed.psd"
    edited.save(str(pushed), update_preview=False)
    real = psd_edit.save_psd

    def sneaky(psd, path):
        psd_edit.name_layer(psd).visible = False            # a flag byte beyond the rename
        real(psd, path)
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


def test_nothing_fixed_when_waived(clone):
    assert cs.fix(b"", "Items/X.psd", [cs.Issue("R9", "x", fix=True, waived="w")], "unused").data == b""


# ========================
# R8: the Single-line Composer, set unsaid
# ========================
EVERY_LINE = "86733bd2"         # PoD08MOAB's Name pushed Every-line (point text)


def test_composer_set_single_line(clone, tmp_path, monkeypatch):
    rel = "Auxiliary/Minions/PoD08MOAB.psd"
    data = at_rev(clone, EVERY_LINE, rel)
    assert not any(issue.code == "R8" for issue in cs.check(PSDImage.open(BytesIO(data)), rel, cs.Context()))
    fixed = cs.fix(data, rel, [], str(tmp_path / "work"))
    (tmp_path / "old.psd").write_bytes(data)
    (tmp_path / "new.psd").write_bytes(fixed.data)
    sizes = psd_edit.body_sizes(str(tmp_path / "old.psd"), str(tmp_path / "new.psd"))   # flags flipped; saved stripped
    assert fixed.composer == ["Name"] and fixed.applied == [] and sizes[0] == sizes[1]
    old, new = PSDImage.open(BytesIO(data)), PSDImage.open(BytesIO(fixed.data))
    assert cs.composer_layers(new) == [] and psd_edit.read_name(new) == psd_edit.read_name(old)
    name = [next(layer for layer in psd.descendants() if layer.name == "Name") for psd in (old, new)]
    assert cs.every_line(name[0]) and not cs.every_line(name[1])
    drawn = [psd.composite(ignore_preview=True).tobytes() for psd in (old, new)]
    assert name[0].topil().tobytes() == name[1].topil().tobytes() and drawn[0] == drawn[1]
    fonts = clone / "MDW" / "Fonts"
    if fonts.is_dir():                          # the fork lays it out alike (it reads no composer)
        from psd_tools.composite.text import render
        drawn = [render(layer, fonts=psd_edit.font_map(str(fonts))) for layer in name]
        assert drawn[0].bbox == drawn[1].bbox and drawn[0].rgba.tobytes() == drawn[1].rgba.tobytes()
        # box text: an Ability set Every-line is always set; when a line rewraps it is redrawn, the preview with it
        ability = next(layer for layer in new.descendants() if layer.name == "Ability")
        records = {psd_edit._record_index(ability)}
        for props, key in cs._composer_values(new, records).values():
            props[key].value = True
        new.save(str(tmp_path / "box.psd"), update_preview=False)
        assert cs.single_line(str(tmp_path / "box.psd"), str(tmp_path / "out.psd"), lambda names: str(fonts)) == (["Ability"], [], [])
        monkeypatch.setattr(cs, "LINE_IOU", 1.01)
        names, moves, left = cs.single_line(str(tmp_path / "box.psd"), str(tmp_path / "out2.psd"), lambda names: str(fonts))
        assert names == ["Ability"] and moves[0][0] == "Ability" and "would move" in moves[0][1] and left == []
        with open(tmp_path / "box.psd", "rb") as fh:
            box = fh.read()
        fixed = cs.fix(box, rel, [], str(tmp_path / "work1"), None, lambda names: str(fonts))
        assert fixed.composer == ["Ability"] and fixed.rewrapped == ["Ability"] and fixed.plain and fixed.data != fixed.plain
        drawn = PSDImage.open(BytesIO(fixed.data))
        assert cs.composer_layers(drawn) == [] and drawn.composite().tobytes() != new.composite().tobytes()
        # best effort: a render failure leaves that layer unmeasured and not redrawn (never told), the Name still fixed
        from psd_tools.composite.text.fonts import TextRenderError
        monkeypatch.setattr(cs, "LINE_IOU", 0.9)
        monkeypatch.setattr("psd_tools.composite.text.render", lambda *a, **k: (_ for _ in ()).throw(TextRenderError("glyph")))
        both = new.descendants()
        for layer in both:
            if layer.name == "Name":
                records.add(psd_edit._record_index(layer))
        for props, key in cs._composer_values(new, records).values():
            props[key].value = True
        new.save(str(tmp_path / "both.psd"), update_preview=False)
        with open(tmp_path / "both.psd", "rb") as fh:
            fixed = cs.fix(fh.read(), rel, [], str(tmp_path / "work2"), None, lambda names: str(fonts))
        assert sorted(fixed.composer) == ["Ability", "Name"] and fixed.composer_left == [("Ability", "TextRenderError: glyph")]
        assert fixed.data and not fixed.rewrapped
        monkeypatch.setattr(cs, "single_line", lambda *a: 1 / 0)
        fixed = cs.fix(data, rel, [], str(tmp_path / "work3"))
        assert fixed.composer == [] and fixed.composer_left[0][1].startswith("ZeroDivisionError") and not fixed.data


# ========================
# Pixel fixes: straight apostrophes, old icon font glyphs (silent, owner 2026-10-25)
# ========================
CURLY = "b67b8799"              # main with 42 cards' curly apostrophes (13 titles, 29 abilities and the old rule pages)


def _fonts(clone):
    fonts = clone / "MDW" / "Fonts"
    if not fonts.is_dir():
        pytest.skip("no MDW/Fonts in the clone")
    return lambda names: str(fonts)


@pytest.mark.skipif(not story_sync(), reason="needs the fork's Txt2 story sync (fix/txt2-sync)")
def test_apostrophes_straightened(clone, tmp_path):
    """A title: every U+2019 set as ' in the layer and its story, the glyphs redrawn, the parse the same but for the
    apostrophes."""
    fonts, parser = _fonts(clone), clone_vocabulary(str(clone)).parser()
    rel = "Auxiliary/Items/Papa's_Cookie.psd"
    data = at_rev(clone, CURLY, rel)
    fixed = cs.fix(data, rel, [], str(tmp_path / "work"), parser, fonts)
    assert fixed.redrawn == {"apostrophes": ["Name"]} and fixed.pixel_left == [] and fixed.plain
    new = PSDImage.open(BytesIO(fixed.data))
    name = next(layer for layer in new.descendants() if layer.name == "Name")
    assert cs.curly_layers(new) == [] and name.text == "Papa's Cookie" and name.story_text == "Papa's Cookie\r"
    assert psd_edit.read_name(new) == "Papa's Cookie"
    assert not list((tmp_path / "work").iterdir())
    temple = "Auxiliary/Items/The_Temple_of_Ahn'Qiraj.psd"     # stored ink past the layout: the engine would redraw it whole
    fixed = cs.fix(at_rev(clone, CURLY, temple), temple, [], str(tmp_path / "work"), parser, fonts)
    assert fixed.redrawn == {} and "not redrawn by glyph" in fixed.pixel_left[0][1]


@pytest.mark.skipif(not story_sync(), reason="needs the fork's text engine")
def test_old_icon_glyph_redrawn_alone(clone, tmp_path):
    """A badge whose stored pixels are not the current font's (an older font's, painted here) is found and drawn again,
    that glyph alone: the rest of the layer keeps Photoshop's pixels."""
    import numpy as np
    from psd_tools.composite.text import capture
    from psd_tools.composite.text.splice import _glyph_box
    fonts = _fonts(clone)
    found = psd_edit.font_map(fonts([]))
    psd = PSDImage.open(BytesIO(at_rev(clone, CURLY, MINI_DOOMER)))
    layer = psd_edit.ability_layer(psd)
    assert cs.old_glyphs(layer, found) == []
    before = capture(layer, found)
    index = next(i for i, g in enumerate(before.layout.glyphs) if g.face.ft.postscript_name.startswith(b"stoogedings"))
    x0, y0, x1, y1 = _glyph_box(before.layout.glyphs[index], before.model.transform, 0)
    rgba, (bx, by, _, _) = before.rgba.copy(), before.bbox
    box = (slice(y0 - by + 3, y1 - by - 3), slice(x0 - bx + 3, x1 - bx - 3))
    rgba[box] = (0.0, 0.6, 0.0, 1.0)                        # a flat green pill: no badge of the current font
    layer._store_raster(rgba, before.bbox)
    old = tmp_path / "old.psd"
    psd.save(str(old), update_preview=False)
    assert cs.old_glyphs(psd_edit.ability_layer(PSDImage.open(str(old))), found) == [index]
    assert cs.redraw_old_glyphs(str(old), str(tmp_path / "new.psd"), fonts) == ["Ability"]
    after = capture(psd_edit.ability_layer(PSDImage.open(str(tmp_path / "new.psd"))), found)
    assert cs.old_glyphs(psd_edit.ability_layer(PSDImage.open(str(tmp_path / "new.psd"))), found) == []
    assert after.bbox == before.bbox
    outside = np.ones(rgba.shape[:2], bool)
    outside[y0 - by - 2:y1 - by + 2, x0 - bx - 2:x1 - bx + 2] = False
    assert np.array_equal(after.rgba[outside], rgba[outside])


WOLFGANG = "Creatures/DST/4 Stars/Wolfgang.psd"     # one (BOT) badge in Belisaplumillamanual, subscript like its text


@pytest.mark.skipif(not story_sync(), reason="needs the fork's text engine")
def test_badges_fitted_to_their_text(clone, tmp_path, monkeypatch):
    """Owner 2026-10-25: a badge 20% small and 3 pt low is set back to the computed size (the pill 1.35 times the text's
    capital height) and middle (the capitals' middle), its glyphs redrawn, never told; the check finds nothing left.
    A resize that moves the text is made anyway, its FX layers cut again on the moved words."""
    import numpy as np
    fonts = _fonts(clone)
    psd = PSDImage.open(clone / WOLFGANG)
    layer = psd_edit.ability_layer(psd)
    fit, = psd_edit.badge_fits(layer, psd_edit.font_map(fonts([])))
    layer.set_span_style(fit["start"], fit["end"], FontSize=round(fit["fit_size"] * 0.8, 2), BaselineShift=fit["fit_shift"] - 3)
    psd_edit.save_psd(psd, str(tmp_path / "off.psd"))
    fixed = cs.fix((tmp_path / "off.psd").read_bytes(), WOLFGANG, [], str(tmp_path / "work"), None, fonts, only="badges")
    assert fixed.redrawn == {"badges": ["Ability"]} and fixed.pixel_left == [] and fixed.plain
    (tmp_path / "fit.psd").write_bytes(fixed.data)
    assert cs.badge_plan(PSDImage.open(str(tmp_path / "fit.psd")), fonts) == []
    now, = psd_edit.badge_fits(psd_edit.ability_layer(PSDImage.open(tmp_path / "fit.psd")), psd_edit.font_map(fonts([])))
    assert abs(now["off_px"]) < 0.1 and now["size"] == fit["fit_size"]
    from dataclasses import replace
    from psd_tools.api.layers import TypeLayer
    draw = TypeLayer.rasterize                          # a line pushed out of its box refuses
    monkeypatch.setattr(TypeLayer, "rasterize", lambda *a, **k: replace(draw(*a, **k), warnings=("overflow",)))
    fixed = cs.fix((tmp_path / "off.psd").read_bytes(), WOLFGANG, [], str(tmp_path / "work"), None, fonts, only="badges")
    assert not fixed.redrawn and fixed.pixel_left[0][1].startswith("Refused: overflow")
    monkeypatch.undo()
    radiance = "Creatures/Hollow Knight/5 Stars/The_Radiance.psd"   # resized anyway: its lines move, Infected's FX follows
    fixed = cs.fix(at_rev(clone, "1f07349c", radiance), radiance, [], str(tmp_path / "work"), None, fonts, only="badges")
    assert fixed.redrawn == {"badges": ["Ability"]} and fixed.pixel_left == []
    (tmp_path / "rad.psd").write_bytes(fixed.data)
    assert cs.badge_plan(PSDImage.open(str(tmp_path / "rad.psd")), fonts) == []
    psd = PSDImage.open(tmp_path / "rad.psd")
    text, (fx,) = psd_edit.ability_layer(psd), psd_edit.fx_layers(psd, psd_edit.ability_layer(psd))
    ink, cut = (np.zeros((psd.height, psd.width), bool) for _ in "ab")
    for mask, layer, level in ((ink, text, 0), (cut, fx, 0.5)):
        x0, y0, x1, y1 = layer.bbox
        mask[y0:y1, x0:x1] = layer.numpy("shape")[..., 0] > level
    assert cut.any() and not (cut & ~ink).any()     # the FX layer sits on the words where they are now


RESAVE = "8a913cc4"             # main with the cards the 10-25 sweeps left to a Photoshop re-save


@pytest.mark.skipif(not story_sync(), reason="needs the fork's text engine")
def test_badges_drawn_kerned_as_the_story(clone, tmp_path, monkeypatch):
    """B12: Soul Sanctum's story turns kerning off where its TySh has it on; followed, the splice takes it (unfollowed,
    Photoshop's ink sits off the fork's layout and it refused)."""
    fonts, rel = _fonts(clone), "Field/2 Stars/Soul_Sanctum.psd"
    data = at_rev(clone, RESAVE, rel)
    fixed = cs.fix(data, rel, [], str(tmp_path / "work"), None, fonts, only="badges")
    assert fixed.redrawn == {"badges": ["Ability"]} and fixed.pixel_left == []
    (tmp_path / "fit.psd").write_bytes(fixed.data)
    assert cs.badge_plan(PSDImage.open(str(tmp_path / "fit.psd")), fonts) == []
    monkeypatch.setattr(psd_edit, "follow_kerning", lambda psd, layer: False)
    fixed = cs.fix(data, rel, [], str(tmp_path / "work"), None, fonts, only="badges")
    assert not fixed.redrawn and "not redrawn by glyph" in fixed.pixel_left[0][1]


@pytest.mark.skipif(not story_sync(), reason="needs the fork's text engine")
def test_old_glyph_redrawn_kerned_as_the_story(clone, tmp_path, monkeypatch):
    """B12 in the guards' old-glyph fix: Soul Sanctum (story kerning off, TySh on) with an old badge painted in is
    spliced, that glyph alone, its story and document blocks kept; unfollowed it refused."""
    from psd_tools.composite.text import capture
    from psd_tools.composite.text.splice import _glyph_box
    fonts, rel = _fonts(clone), "Field/2 Stars/Soul_Sanctum.psd"
    found = psd_edit.font_map(fonts([]))
    psd = PSDImage.open(BytesIO(at_rev(clone, RESAVE, rel)))
    layer = psd_edit.ability_layer(psd)
    assert psd_edit.follow_kerning(PSDImage.open(BytesIO(at_rev(clone, RESAVE, rel))),
                                   psd_edit.ability_layer(PSDImage.open(BytesIO(at_rev(clone, RESAVE, rel)))))
    before = capture(layer, found)
    index = next(i for i, g in enumerate(before.layout.glyphs) if g.face.ft.postscript_name.startswith(b"stoogedings"))
    x0, y0, x1, y1 = _glyph_box(before.layout.glyphs[index], before.model.transform, 0)
    rgba, (bx, by, _, _) = before.rgba.copy(), before.bbox
    rgba[y0 - by + 3:y1 - by - 3, x0 - bx + 3:x1 - bx - 3] = (0.0, 0.6, 0.0, 1.0)     # an older font's pill
    layer._store_raster(rgba, before.bbox)
    old = tmp_path / "old.psd"
    psd.save(str(old), update_preview=False)
    fixed = cs.fix(old.read_bytes(), rel, [], str(tmp_path / "work"), None, fonts, only="old glyphs")
    assert fixed.redrawn == {"old glyphs": ["Ability"]} and fixed.pixel_left == []
    new = PSDImage.open(BytesIO(fixed.data))
    assert cs.old_glyphs(psd_edit.ability_layer(new), found) == []
    story = psd_edit.story_index(psd_edit.ability_layer(new))
    assert psd_edit.gate_stories(PSDImage.open(str(old)), new, [story]) == [story]
    monkeypatch.setattr(psd_edit, "follow_kerning", lambda psd, layer: False)
    fixed = cs.fix(old.read_bytes(), rel, [], str(tmp_path / "work"), None, fonts, only="old glyphs")
    assert not fixed.redrawn and "not redrawn by glyph" in fixed.pixel_left[0][1]


@pytest.mark.skipif(not story_sync(), reason="needs the fork's text engine")
def test_badges_whole_only_where_drawn_as_stored(clone, tmp_path, monkeypatch):
    """A sweep's whole: Restrained (the splice misses its pill's 1 px edge) is redrawn whole, its lines kept; drawn off
    its pixels, it refuses. The guards never redraw whole. Wendy, which the fork laid out off Photoshop's pixels until it
    spaced tracked badges as Photoshop (cards-20261104), is spliced by glyph."""
    fonts, faces = _fonts(clone), psd_edit.font_map(_fonts(clone)([]))
    rel = "Auxiliary/Status Effects/Debuffs/2 Stars/Restrained.psd"
    data = at_rev(clone, RESAVE, rel)
    fixed = cs.fix(data, rel, [], str(tmp_path / "work"), None, fonts, only="badges")
    assert not fixed.redrawn and "not redrawn by glyph" in fixed.pixel_left[0][1]
    fixed = cs.fix(data, rel, [], str(tmp_path / "work"), None, fonts, only="badges", whole=True)
    assert fixed.redrawn == {"badges": ["Ability"]} and fixed.pixel_left == []
    (tmp_path / "fit.psd").write_bytes(fixed.data)
    assert cs.badge_plan(PSDImage.open(str(tmp_path / "fit.psd")), fonts) == []
    lines = [cs._lines(psd_edit.ability_layer(PSDImage.open(BytesIO(d))), faces) for d in (data, fixed.data)]
    assert lines[0] == lines[1]
    monkeypatch.setattr(cs, "_ink_off", lambda layer, faces: cs.WHOLE_OFF * 2)
    fixed = cs.fix(data, rel, [], str(tmp_path / "work"), None, fonts, only="badges", whole=True)
    assert not fixed.redrawn and "not redrawn by glyph" in fixed.pixel_left[0][1]
    wendy = "Creatures/DST/3 Stars/Wendy.psd"
    fixed = cs.fix(at_rev(clone, RESAVE, wendy), wendy, [], str(tmp_path / "work"), None, fonts, only="badges")
    assert fixed.redrawn == {"badges": ["Ability"]} and fixed.pixel_left == []


BUTTERFLY = ("2e4c9dd1", "Creatures/Kirby/5 Stars/Butterfly.psd")   # badges in Section layers under a Gradient Overlay


@pytest.mark.skipif(not story_sync(), reason="needs the fork's text engine")
def test_badges_under_an_overlay_set_one_colour(clone, tmp_path):
    """Owner 2026-10-30: Butterfly's (OFS) and (OD), re-saved in Photoshop without the icon font's Stylistic Alternates,
    printed as blobs under their layers' Gradient Overlay: set in the alternates (the story alone), those glyphs redrawn,
    never told; nothing left. Mini Doomer's, set by the bake, are left alone."""
    fonts = _fonts(clone)
    rev, rel = BUTTERFLY
    data = at_rev(clone, rev, rel)
    assert [layer.name for layer in cs.salt_layers(PSDImage.open(BytesIO(data)))] == ["Section 2", "Section 1"]
    fixed = cs.fix(data, rel, [], str(tmp_path / "work"), None, fonts, only="overlay badges")
    assert fixed.redrawn == {"overlay badges": ["Section 2", "Section 1"]} and fixed.pixel_left == [] and fixed.plain
    new = PSDImage.open(BytesIO(fixed.data))
    assert cs.salt_layers(new) == []
    salted = {layer.name: [layer.text[a:b] for a, b in psd_edit.stylistic_alternates(layer)]
              for layer in new.descendants() if layer.kind == "type" and layer.name.startswith("Section")}
    assert salted == {"Section 1": ["(OFS)"], "Section 2": ["(OD)"]}
    assert cs.salt_layers(PSDImage.open(BytesIO(at_rev(clone, rev, MINI_DOOMER)))) == []


NAME_CARDS = (("Silithid", "Silithid", "minion"), ("Stay", "Stay", "creature"), ("Polevaulter Zombie", "Polevaulter Zombie", "minion"),
              ("Burning", "Burning", "debuff"), ("The Gates of Ahn'Qiraj", "The Gates of Ahn'Qiraj", "field"))


def test_card_names_in_another_case():
    """R14 (owner 2026-10-30, report only): names the website would link in any case (text.js: several words, a capital,
    a lowercase one after its verb) but typed in another case, the card's own name too; plain words, [tokens] and badges
    are no mention."""
    text = ("(2GCD)(OYT) summon a silithid on any slot. Stay here, stay calm. Summon a polevaulter zombie.\r"
            "Inflict 2T1 burning. Summon a Silithid. [burning] burning")
    assert cs.miscased_names(text, NAME_CARDS) == [("silithid", "Silithid"), ("polevaulter zombie", "Polevaulter Zombie"),
                                                   ("burning", "Burning")]
    assert cs.miscased_names(text, ()) == []


def test_card_names_case_rule_on_the_gates(clone):
    """The Gates of Ahn'Qiraj's "summon a silithid": an R14 report, told, never fixed; without card names, not checked."""
    rel = "Field/4 Stars/The_Gates_of_Ahn'Qiraj.psd"
    psd = PSDImage.open(clone / rel)
    found = [issue for issue in cs.check(psd, rel, cs.Context(cards=NAME_CARDS)) if issue.code == "R14"]
    assert [issue.items for issue in found] == [("silithid",)] and cs.reported(found) == found and "'silithid' should be 'Silithid'" in found[0].text
    assert not any(issue.code == "R14" for issue in cs.check(psd, rel, cs.Context()))


def test_card_names_case_rule_skips_lowercase_cards(clone):
    """snas. (Sans.psd) is written all in lowercase on purpose, its name too: its 'smeckeldorfed' is no R14 (owner
    2026-11-05); the same mention elsewhere still is, and Kyle (printed hdckighfkvhvgmk, capitalised text) is no exception."""
    cards = NAME_CARDS + (("snas.", "Sans", "creature"), ("Smeckeldorfed", "Smeckeldorfed", "debuff"))
    rel = "Creatures/Undertale/5 Stars/Sans.psd"
    psd = PSDImage.open(clone / rel)
    assert any(f[0] == "smeckeldorfed" for layer in cs.ability_layers(psd) for f in cs.miscased_names(layer.text or "", cards))
    assert not [issue for issue in cs.check(psd, rel, cs.Context(cards=cards)) if issue.code == "R14"]
    assert cs.rule_name_case(psd, cards, "other") is not None
    kyle = PSDImage.open(clone / "Creatures/Other/5 Stars/Kyle.psd")   # a lowercase name, capitalised text: no exception
    texts = [layer.text or "" for layer in cs.ability_layers(kyle)]
    assert cs._lowercase_card([layer.text or "" for layer in cs.ability_layers(psd)], "snas.")
    assert not cs._lowercase_card(texts, "hdckighfkvhvgmk") and not cs._lowercase_card(texts, "Kyle")


def test_plain_timing_tags_set_as_badges(clone, tmp_path):
    """Owner 2026-11-03: Elmadan's "(Action)" and "(2 points)", typed in its body font (Apple-Kid) before the font had
    the badges: the tags kind (the guards' on every push) sets them in the icon font (added to the card), fitted, by
    glyph; R1 has nothing left."""
    rel = "Creatures/Other/3 Stars/Elmadan.psd"
    data = at_rev(clone, "2e4c9dd1", rel)
    assert cs.rule_tags(PSDImage.open(BytesIO(data))).items == ("(Action)", "(2 points)")   # a cost too (2026-11-03)
    assert "tags" in cs.pixel_kinds(data)
    fixed = cs.fix(data, rel, [], str(tmp_path / "work"), None, _fonts(clone), only="tags")
    assert fixed.redrawn == {"tags": ["Ability"]} and fixed.pixel_left == []
    new = PSDImage.open(BytesIO(fixed.data))
    assert cs.rule_tags(new) is None
    layer = next(layer for layer in new.descendants() if layer.kind == "type" and layer.name.strip() == "Ability")
    assert "(Action)(2 points)" in layer.text


# ========================
# Badge widths (owner 2026-11-04: the cooldown ring widens every cooldown badge)
# ========================
DAROACH = "Creatures/Kirby/4 Stars/Dark_Daroach.psd"


@pytest.fixture(scope="module")
def ring_fonts(clone, tmp_path_factory):
    """The pin's MDW/Fonts with RING's icon font: (fonts, the same with .earlier the pin's icon font, before the ring)."""
    from functools import partial
    pinned = clone / "MDW" / "Fonts"
    if not pinned.is_dir():
        pytest.skip("no MDW/Fonts in the clone")
    folder = tmp_path_factory.mktemp("ring-fonts")
    for font in pinned.iterdir():
        if font.is_file() and font.name != "stoogedings.ttf":
            (folder / font.name).symlink_to(font)
    (folder / "stoogedings.ttf").write_bytes(at_rev(clone, RING, "MDW/Fonts/stoogedings.ttf"))
    fonts = lambda names: str(folder)
    before = tmp_path_factory.mktemp("ring-earlier") / "stoogedings.earlier"      # the icon font before the ring's commit
    before.write_bytes(at_rev(clone, f"{RING}^", "MDW/Fonts/stoogedings.ttf"))
    earlier = partial(fonts)
    earlier.earlier = (str(before),)
    return fonts, earlier


def _ability(data: bytes):
    return next(layer for layer in PSDImage.open(BytesIO(data)).descendants()
                if layer.kind == "type" and layer.name.strip() == "Ability")


@pytest.mark.skipif(not story_sync(), reason="needs the fork's text engine")
def test_badge_widths_relaid_from_photoshops_layout(clone, tmp_path, ring_fonts, monkeypatch):
    """The guards' case: Dark Daroach's (1/2CD) and (2CD), drawn at the old widths (Photoshop's view says them), laid out
    again at the ring font's, by glyph; its text and stories untouched; afterwards nothing is left, so it never fires
    again. At that font its ink stays 37 px off the canvas edge: an outline that wide would reach it, refused (review
    2026-11-04, Greenpath); its own 5 px outline is re-laid."""
    fonts, _ = ring_fonts
    data = at_rev(clone, RING, DAROACH)
    assert "badge widths" in cs.pixel_kinds(data)
    monkeypatch.setattr(psd_edit, "_stroke_reach", lambda layer: 37)
    fixed = cs.fix(data, DAROACH, [], str(tmp_path / "work"), None, fonts, only="badge widths")
    assert not fixed.redrawn and fixed.pixel_left[0][1].endswith("37 px from the canvas edge (its outline 37 px)")
    monkeypatch.undo()
    fixed = cs.fix(data, DAROACH, [], str(tmp_path / "work"), None, fonts, only="badge widths")
    assert fixed.redrawn == {"badge widths": ["Ability"]} and fixed.pixel_left == [] and fixed.plain
    old, new = _ability(data), _ability(fixed.data)
    faces = psd_edit.font_map(fonts([]), new)
    assert cs.laid_plan(new, faces) is None and cs.old_glyphs(new, faces) == []
    assert new.text == old.text and psd_edit.stories(new._psd) == psd_edit.stories(old._psd)
    assert cs.fix(fixed.data, DAROACH, [], str(tmp_path / "work"), None, fonts, only="badge widths").redrawn == {}


@pytest.mark.skipif(not story_sync(), reason="needs the fork's text engine")
def test_badge_widths_from_an_earlier_font(clone, tmp_path, ring_fonts):
    """A raster the fork drew: Waxwell's view is older than its pixels, so alone it refuses; with the earlier icon font
    (fonts.earlier, the guards' and a sweep's) its (1CD) is re-laid. Michael's Blessing's seven badges at their new
    widths overflow its box: refused."""
    fonts, earlier = ring_fonts
    rel = "Creatures/DST/3 Stars/Waxwell.psd"
    data = at_rev(clone, RING, rel)
    fixed = cs.fix(data, rel, [], str(tmp_path / "work"), None, fonts, only="badge widths")
    assert not fixed.redrawn and "off its pixels" in fixed.pixel_left[0][1]
    fixed = cs.fix(data, rel, [], str(tmp_path / "work"), None, earlier, only="badge widths")
    assert fixed.redrawn == {"badge widths": ["Ability"]} and fixed.pixel_left == []
    new = _ability(fixed.data)
    assert cs.laid_plan(new, psd_edit.font_map(fonts([]), new), earlier.earlier) is None
    rel = "N.M.E/Michael's_Blessing.psd"
    fixed = cs.fix(at_rev(clone, RING, rel), rel, [], str(tmp_path / "work"), None, earlier, only="badge widths")
    assert not fixed.redrawn and fixed.pixel_left[0][1].startswith("Refused: overflow")


def test_drawn_earlier_by_glyph_name(monkeypatch):
    """_drawn_earlier lays an earlier icon font's layout beside the current one's by glyph name in the icon font (ids
    shift when icons are added), by id in other fonts: a badge at another id counts, another badge or body glyph does
    not."""
    def face(postscript: str, names: dict):
        return SimpleNamespace(ft=SimpleNamespace(postscript_name=postscript),
                               hb=SimpleNamespace(get_glyph_name=names.get))

    def offs(glyphs: list, shares: dict):
        laid = [SimpleNamespace(cluster=i, gid=gid, face=f) for i, (gid, f) in enumerate(glyphs)]
        return SimpleNamespace(layout=SimpleNamespace(glyphs=laid)), None, None, None, shares

    body = face("Arial", {})
    now = offs([(36, body), (714, face("stoogedings-Regular", {714: "badge_1CD"}))], {1: 0.20})
    for glyphs, found in (([(36, body), (700, face("stoogedings-Regular", {700: "badge_1CD"}))], {1}),
                          ([(36, body), (701, face("stoogedings-Regular", {701: "badge_2CD"}))], set()),
                          ([(37, body), (700, face("stoogedings-Regular", {700: "badge_1CD"}))], set())):
        monkeypatch.setattr(cs, "_glyph_offs", lambda layer, faces: offs(glyphs, {1: 0.05}))
        assert cs._drawn_earlier(None, None, ("then.earlier",), now) == found


def test_badge_widths_line_guards():
    """relay_badges' line guards (review 2026-11-04): a new line of badges alone is found (Wigfrid's '(1CD)' cut off from
    its cost), ink nearer the canvas edge than before is noted once inside EDGE_NEAR (Sword of Aeons), a new
    break that cuts a count from its word or a coloured term is noted (Toriel's '4' / 'HP', Warcraftian Shaman's
    'Fire' / 'Totem'), a badge and its full stop alone on a line are found (Warcraftian Paladin's '(ENDT).'), not plain
    bracketed text (Puzzled Cat's); a break inside a card's name (Heavy Mole's 'Mini' / 'Drill.') or after a signed count
    is noted; each canvas side is judged on its own, the outline's and the glow's distance said."""
    line = lambda text, y=0.0: SimpleNamespace(text=text, y=y, drawn=True)
    assert cs._badges_alone([line("(1CD)(3 points) Create a"), line("Battle Spear.")],
                            [line("(1CD)"), line("(3 points) Create a")]) == "(1CD)"
    assert cs._badges_alone([line("(1CD)"), line("Go.")], [line("(1CD)"), line("Go.")]) == ""
    assert cs._badges_alone([line("immune until (ENDT).")], [line("immune until"), line("(ENDT).")]) == "(ENDT)."
    assert cs._badges_alone([line("Go (cannot ignore usage conditions).")],
                            [line("Go"), line("(cannot ignore usage conditions).")]) == ""
    mole = "Summon a Mini Drill. Gain +2 ATK."
    on = lambda a, b: SimpleNamespace(text=mole[a:b], start=a, end=b, drawn=True, y=0.0)
    plain = SimpleNamespace(text=mole, runs=(SimpleNamespace(start=0, end=len(mole), style=SimpleNamespace(fill=(1, 1, 1))),))
    cards = (("Mini Drill", "mini-drill", "minion"),)
    assert cs._split_terms(plain, [on(0, len(mole))], [on(0, 13), on(14, 28), on(29, len(mole))], cards) == [
        ("Mini", "Drill."), ("+2", "ATK.")]
    assert cs._split_terms(plain, [on(0, len(mole))], [on(0, 13), on(14, len(mole))]) == []
    text = "Heal a creature for 4 HP, and inflict 1T Insane. Fire Totem, Air Totem."   # 4 HP and Fire Totem coloured
    at = lambda a, b: SimpleNamespace(text=text[a:b], start=a, end=b, drawn=True, y=0.0)
    paint = lambda a, b, rgb: SimpleNamespace(start=a, end=b, style=SimpleNamespace(fill=rgb))
    white, green = (1.0, 1.0, 1.0), (0.2, 0.9, 0.3)
    model = SimpleNamespace(text=text, runs=(paint(0, 20, white), paint(20, 24, green), paint(24, 49, white),
                                             paint(49, 59, green), paint(59, len(text), white)))
    old = [at(0, 25), at(26, len(text))]
    assert cs._split_terms(model, old, [at(0, 21), at(22, 40), at(41, len(text))]) == [("4", "HP,"), ("1T", "Insane.")]
    assert cs._split_terms(model, old, [at(0, 25), at(26, 53), at(54, len(text))]) == [("Fire", "Totem,")]
    assert cs._split_terms(model, old, old) == []
    assert cs._reflow_notes("Ability", old, [at(0, 21), at(22, len(text))], (90, 0, 99, 9), (90, 0, 99, 9), 200, 200,
                            model) == ["Ability: '4' / 'HP,' split across lines"]
    assert cs._reflow_notes("Ability", [], [], (21, 900, 1179, 1100), (2, 900, 1200, 1100), 1200, 2400) == [
        "Ability: ink 0 px from the canvas edge (was 21)"]
    assert cs._reflow_notes("Ability", [], [], (10, 900, 1190, 1100), (10, 900, 1190, 1100), 1200, 2400) == []
    assert cs._reflow_notes("Ability", [], [], (40, 900, 1190, 2395), (6, 900, 1190, 2395), 1200, 2400, None, (7, 29)) == [
        "Ability: ink 6 px from the canvas edge (was 40), its outline -1 px, its glow -23 px"]   # the bottom 5 px both times


def test_old_glyphs_told_only_from_another_build(monkeypatch):
    """The guards tell old glyphs (owner 2026-11-04) only when they show another build of the icon font: the rest of the
    layer is the current font's layout, or, their advances moved, an earlier layout is; never when the fork lays the
    layer out off Photoshop's pixels (Michael's Blessing re-saved with a 65536 times larger tracking)."""
    drawn = (None, None, None, None, [3])
    off = {"share": 0.02}
    monkeypatch.setattr(cs, "_off_beside", lambda drawn, marks: off["share"])
    monkeypatch.setattr(cs, "laid_plan", lambda layer, faces, earlier: ("laid", {3}, 0.01))
    assert cs._another_build(None, None, (), drawn, False)
    off["share"] = 0.6
    assert not cs._another_build(None, None, (), drawn, False)
    assert cs._another_build(None, None, (), drawn, True)
    off["share"] = 0.0126                       # Michael's Blessing: the rest fits, two badges thrown off by the tracking

    def refused(layer, faces, earlier):
        raise psd_edit.Refused("unsupported-card", "Ability: at its old badge widths the fork's layout is 44% off")
    monkeypatch.setattr(cs, "laid_plan", refused)
    assert not cs._another_build(None, None, (), drawn, True)


@pytest.mark.skipif(not story_sync(), reason="needs the fork's text engine")
def test_old_glyphs_refuse_a_moved_advance(clone, tmp_path, ring_fonts):
    """The safety net: Dark Daroach's badges drawn at the old widths are old glyphs to the ring font, their advances
    moved (Photoshop's view): old glyphs refuses them before any splice (badge widths' to re-lay)."""
    fonts, _ = ring_fonts
    (tmp_path / "in.psd").write_bytes(at_rev(clone, RING, DAROACH))
    with pytest.raises(psd_edit.Refused, match="laid out at another width"):
        cs.redraw_old_glyphs(str(tmp_path / "in.psd"), str(tmp_path / "out.psd"), fonts)


def _typed_curly(data: bytes, fonts, path, earlier=()) -> bytes:
    """Taranza's "creature's" typed with a curly apostrophe before the ring: spliced at the old widths it was laid out
    with (laid_plan's), as Photoshop drew it then."""
    from psd_tools.composite.text import capture
    psd = PSDImage.open(BytesIO(data))
    layer = next(layer for layer in psd.descendants() if layer.kind == "type" and layer.name.strip() == "Ability")
    laid = cs.laid_plan(layer, psd_edit.font_map(fonts([]), layer), earlier)[0]
    keep = capture(layer, laid)
    layer.set_text(layer.text.replace("creature's", "creature’s"), keep="runs")
    layer.rasterize(fonts=laid, keep=keep, unit=psd_edit.SPLICE_UNIT)
    psd.save(str(path), update_preview=False)
    return path.read_bytes()


@pytest.mark.skipif(not story_sync(), reason="needs the fork's text engine")
def test_guards_fixes_compose_with_the_ring_font(clone, tmp_path, ring_fonts):
    """The guards' fixes over a push with the ring font (and the earlier icon font, as the guards read it): Martlet's
    plain (2CD) under its overlay (before the badges) set as a badge in the new font and one colour; Taranza's one-colour
    (2CD) at the old width, a curly apostrophe typed in the same layer, re-laid first and then straightened (straighten
    lays out at the current widths), still one colour."""
    fonts, earlier = ring_fonts
    martlet, taranza = "Creatures/Undertale/3 Stars/Martlet.psd", "Creatures/Kirby/3 Stars/Taranza.psd"
    for rel, data, kinds in ((martlet, at_rev(clone, "1b4f36a0", martlet), ["tags"]),
                             (taranza, _typed_curly(at_rev(clone, RING, taranza), fonts, tmp_path / "curly.psd", earlier.earlier),
                              ["badge widths", "apostrophes"])):
        fixed = cs.fix(data, rel, [], str(tmp_path / "work"), None, earlier)
        assert fixed.redrawn == {kind: ["Ability"] for kind in kinds} and list(fixed.redrawn) == kinds, (rel, fixed.pixel_left)
        new = _ability(fixed.data)
        faces = psd_edit.font_map(fonts([]), new)
        assert cs.salt_layers(new._psd) == [] and [new.text[a:b] for a, b in psd_edit.stylistic_alternates(new)] == ["(2CD)"]
        assert cs.laid_plan(new, faces, earlier.earlier) is None and cs.curly_layers(new._psd) == []
        assert cs.rule_tags(new._psd) is None and cs.old_glyphs(new, faces) == []


# ========================
# R15: text off the card
# ========================
LEAD_BLOON = "Auxiliary/Minions/PoD05Lead_Bloon.psd"   # its Ability's ink runs past the card's edge


def test_text_off_the_card(clone, context):
    """R15 on every card kind, report only: a new card's Ability run off the edge is told; the same text in the old blob
    is waived (as R4); a card inside its edges has none."""
    psd = PSDImage.open(clone / LEAD_BLOON)
    (found,) = [i for i in cs.check(psd, LEAD_BLOON, context, added=True) if i.code == "R15"]
    assert found.items == ("Ability",) and not found.waived and cs.reported([found]) == [found]
    (again,) = [i for i in cs.check(psd, LEAD_BLOON, context, old=PSDImage.open(clone / LEAD_BLOON)) if i.code == "R15"]
    assert again.waived and not again.new
    clean = "Auxiliary/Minions/PoD01Red_Bloon.psd"
    assert not [i for i in cs.check(PSDImage.open(clone / clean), clean, context, added=True) if i.code == "R15"]


# ========================
# Normalise (the big pass, owner 2026-11-07)
# ========================
def test_normalise_sets_text_normal_and_strips(clone, tmp_path):
    """Balloon Boy's sub and superscript runs set normal and its merged image stripped in one gated rewrite (the parse
    the same: fix's reparse gate): every glyph where it was, every type layer's box, the fork's drawing the same pixels;
    a second normalise has nothing to do."""
    import numpy as np
    old, new, _, fonts = normalised(clone, BALLOON_BOY, tmp_path)
    assert cs.script_layers(old) and old.has_preview() and not cs.script_layers(new) and not new.has_preview()
    assert "normalise" in cs.SWEEP_ONLY and not cs.pixel_kinds(old)        # the guards' side waits for G1
    faces = psd_edit.layout_fonts(fonts([]))
    for a, b in zip(old.descendants(), new.descendants()):
        if a.kind == "type":
            assert psd_edit.same_places(psd_edit.glyph_places(a, faces), psd_edit.glyph_places(b, faces),
                                        psd_edit.GLYPH_TOL) and a.bbox == b.bbox, a.name
    draw = lambda psd: np.asarray(psd.composite(ignore_preview=True, apply_icc=False, text_preview=False))
    assert np.array_equal(draw(old), draw(new))
    buf = BytesIO()
    new.save(buf, update_preview=False)
    again = cs.fix(buf.getvalue(), BALLOON_BOY, [], str(tmp_path / "again"), None, fonts, only="normalise")
    assert again.data == b"" and again.normalised == []


def test_a_widened_box_passes_the_gates(clone, tmp_path):
    """K2's redo (fonts.widen, layer ids): Toy Bonnie's Ability box narrowed into its slack (proofs P1 c02: 396 to 394
    pt), its own Txt2 frame with it, through normalise's gates."""
    from psd_tools.composite.text.model import TextModel
    from helpers import _card
    rel = "Creatures/Fnaf/3 Stars/Toy_Bonnie.psd"
    fonts = lambda names: str(clone / "MDW" / "Fonts")     # noqa: E731
    fonts.widen = [321]
    done = cs.normalise(_card(clone, rel)[0], str(tmp_path / "out.psd"), fonts, rel)
    layer = next(x for x in PSDImage.open(tmp_path / "out.psd").descendants() if x.layer_id == 321)
    assert "Ability" in done and not done.left and round(TextModel.from_layer(layer).box[2], 3) == 394.0
