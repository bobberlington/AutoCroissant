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
def test_badge_tags_are_the_fonts_glyphs(clone):
    """badge_<tag>, "/" as "_" and "-" as "m" (build_stoogedings.badge_glyph): reversible, so the font is the source."""
    from fontTools.ttLib import TTFont
    font = clone / "MDW/Fonts/stoogedings.ttf"
    if not font.exists():
        pytest.skip("no stoogedings.ttf")
    glyphs = [n for n in TTFont(str(font)).getGlyphOrder() if n.startswith("badge_") and "." not in n]   # not .salt
    tags = {("-" + n[7:] if n.startswith("badge_m") else n[6:]).replace("_", "/") for n in glyphs}
    assert tags == set(cs.BADGE_TAGS) and len(cs.BADGE_TAGS) == len(set(cs.BADGE_TAGS)) == 341


def test_kinds_and_keys():
    assert [cs.kind_of(p) for p in ("Auxiliary/Rulebook/02_A.psd", "Auxiliary/Creatures/X.psd", "Auxiliary/Items/X.psd",
                                     "Auxiliary/Minions/X.psd", "Auxiliary/Status Effects/Buffs/1 Stars/X.psd", "N.M.E/X.psd",
                                     "Field/1 Stars/X.psd", "Items/Heal/2 Stars/X.psd", "Creatures/Isaac/3 Stars/X.psd")] == [
        "rule", "aux creature", "aux item", "minion", "status", "nme", "field", "item", "creature"]
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


def _saved(psd, path) -> PSDImage:
    psd_edit.save_psd(psd, str(path), update_preview=False)
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


# ========================
# R8: the Single-line Composer, set unsaid
# ========================
EVERY_LINE = "86733bd2"         # PoD08MOAB's Name pushed Every-line (point text)


def test_composer_set_single_line(clone, tmp_path, monkeypatch):
    rel = "Auxiliary/Minions/PoD08MOAB.psd"
    data = at_rev(clone, EVERY_LINE, rel)
    assert not any(issue.code == "R8" for issue in cs.check(PSDImage.open(BytesIO(data)), rel, cs.Context()))
    fixed = cs.fix(data, rel, [], str(tmp_path / "work"))
    assert fixed.composer == ["Name"] and fixed.applied == [] and len(fixed.data) == len(data)
    old, new = PSDImage.open(BytesIO(data)), PSDImage.open(BytesIO(fixed.data))
    assert cs.composer_layers(new) == [] and psd_edit.read_name(new) == psd_edit.read_name(old)
    name = [next(layer for layer in psd.descendants() if layer.name == "Name") for psd in (old, new)]
    assert cs.every_line(name[0]) and not cs.every_line(name[1])
    assert name[0].topil().tobytes() == name[1].topil().tobytes() and old.composite().tobytes() == new.composite().tobytes()
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
        psd_edit.save_psd(new, str(tmp_path / "box.psd"), update_preview=False)
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
        psd_edit.save_psd(new, str(tmp_path / "both.psd"), update_preview=False)
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
CURLY = "b67b8799"              # main with 42 cards' curly apostrophes (13 titles, 29 abilities and rule pages)


def _fonts(clone):
    fonts = clone / "MDW" / "Fonts"
    if not fonts.is_dir():
        pytest.skip("no MDW/Fonts in the clone")
    return lambda names: str(fonts)


@pytest.mark.skipif(not story_sync(), reason="needs the fork's Txt2 story sync (fix/txt2-sync)")
def test_apostrophes_straightened(clone, tmp_path):
    """A title and a rule page: every U+2019 set as ' in the layer and its story, the glyphs redrawn, the parse the same
    but for the apostrophes; nothing else fixed on the rule page."""
    fonts, parser = _fonts(clone), clone_vocabulary(str(clone)).parser()
    rel = "Auxiliary/Items/Papa's_Cookie.psd"
    data = at_rev(clone, CURLY, rel)
    fixed = cs.fix(data, rel, [], str(tmp_path / "work"), parser, fonts)
    assert fixed.redrawn == {"apostrophes": ["Name"]} and fixed.pixel_left == [] and fixed.plain
    new = PSDImage.open(BytesIO(fixed.data))
    name = next(layer for layer in new.descendants() if layer.name == "Name")
    assert cs.curly_layers(new) == [] and name.text == "Papa's Cookie" and name.story_text == "Papa's Cookie\r"
    assert psd_edit.read_name(new) == "Papa's Cookie"
    rule = "Auxiliary/Rulebook/11_Abilities2.psd"
    fixed = cs.fix(at_rev(clone, CURLY, rule), rule, [], str(tmp_path / "work"), parser, fonts)
    assert list(fixed.redrawn) == ["apostrophes"] and fixed.applied == [] and fixed.composer == []
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
    psd_edit.save_psd(psd, str(old), update_preview=False)
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
    Rule pages keep theirs. A resize that moves the text is made anyway, its FX layers cut again on the moved words."""
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
    rule = "Auxiliary/Rulebook/11_Abilities5.psd"
    assert not cs.fix((clone / rule).read_bytes(), rule, [], str(tmp_path / "work"), None, fonts, only="badges").data
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
    psd_edit.save_psd(psd, str(old), update_preview=False)
    fixed = cs.fix(old.read_bytes(), rel, [], str(tmp_path / "work"), None, fonts, only="old glyphs")
    assert fixed.redrawn == {"old glyphs": ["Ability"]} and fixed.pixel_left == []
    new = PSDImage.open(BytesIO(fixed.data))
    assert cs.old_glyphs(psd_edit.ability_layer(new), found) == []
    assert psd_edit.gate_stories(PSDImage.open(str(old)), new, []) == []
    monkeypatch.setattr(psd_edit, "follow_kerning", lambda psd, layer: False)
    fixed = cs.fix(old.read_bytes(), rel, [], str(tmp_path / "work"), None, fonts, only="old glyphs")
    assert not fixed.redrawn and "not redrawn by glyph" in fixed.pixel_left[0][1]


@pytest.mark.skipif(not story_sync(), reason="needs the fork's text engine")
def test_badges_whole_only_where_drawn_as_stored(clone, tmp_path):
    """A sweep's whole: Restrained (the splice misses its pill's 1 px edge) is redrawn whole, its lines kept; Wendy, which
    the fork lays out off Photoshop's pixels, still refuses. The guards never redraw whole."""
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
    wendy = "Creatures/DST/3 Stars/Wendy.psd"
    fixed = cs.fix(at_rev(clone, RESAVE, wendy), wendy, [], str(tmp_path / "work"), None, fonts, only="badges", whole=True)
    assert not fixed.redrawn and "not redrawn by glyph" in fixed.pixel_left[0][1]


def test_text_with_both_quotes_parses_without_backslash(clone, tmp_path):
    """str() of an engine string is its repr, which escapes ' as \\' when the text also holds a ": the parse unescapes it
    (03 Setting Up's "they're")."""
    rel = "Auxiliary/Rulebook/03_Setting_Up.psd"
    (tmp_path / "card.psd").write_bytes(at_rev(clone, RESAVE, rel))
    ability = psd_edit.parse_card(clone_vocabulary(str(clone)).parser(), str(tmp_path / "card.psd"), rel)["card"]["ability"]
    assert "they're" in ability and "\\'" not in ability and '"' in ability
