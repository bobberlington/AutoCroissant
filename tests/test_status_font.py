"""Status font (owner 2026-11-09): scope, the grown glow, a title with no style, the guards' kind."""
import commands.card_standards as cs
import commands.status_font as sf


def test_scope():
    assert sf.in_scope("Auxiliary/Status Effects/Debuffs/2 Stars/Blazing.psd")
    assert sf.in_scope("Auxiliary/Status Effects/Buffs/Marked For Life.psd")
    assert sf.in_scope(sf.TEMPLATE)
    assert not sf.in_scope("Auxiliary/Status Effects/Satus Effect Colors.psd")
    assert not sf.in_scope("Creatures/1 Stars/Frog.psd")


def test_grown_keeps_the_blur():
    # Photoshop dilates by floor(size * spread / 100 + 0.5) px and blurs by the rest: past a 9 px stroke the glow
    # dilates 9 px more and blurs the same
    for size, spread, by in ((18, 42, 9), (13, 19, 9), (18, 27, 9), (2, 0, 8)):
        new, p = sf.grown(size, spread, by)
        pixels = int(size * spread / 100 + 0.5)
        assert new == size + by and int(new * p / 100 + 0.5) == pixels + by and float(p).is_integer()


def test_kind_runs_before_status_words():
    kinds = list(cs.PIXEL_FIXES)
    assert kinds.index("tags") < kinds.index("status font") < kinds.index("status words")
    assert "status font" in cs.PIXEL_LINES and "status font" not in cs.SWEEP_ONLY


def test_title_with_no_style_takes_the_charts():
    from types import SimpleNamespace as NS
    glow = {b"enab": True, b"present": True, b"blur": 18.0}
    status = NS(blocks={sf.LFX2: NS(data={b"masterFXSwitch": True, b"OrGl": glow})})
    block, legacy, kinds, whole = sf._effects(NS(tagged_blocks={}), status)
    assert whole and block.data == status.blocks[sf.LFX2].data and legacy is None and kinds == ["OrGl"]
