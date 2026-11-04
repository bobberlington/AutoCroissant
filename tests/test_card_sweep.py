"""
card_sweep: the kinds it takes and the icon font's earlier versions (card_guards.earlier_fonts) a badge widths sweep and
the guards read from the ref's history (the advances a raster drawn before the cooldown ring may show).
"""
import pytest

from commands.card_guards import EARLIER_FONTS, ICON_FONT_FILE, Source, earlier_fonts
from commands.card_sweep import kinds_of
from conftest import RING, blob_sha


def _advance(path: str, text: str) -> int:
    import uharfbuzz as hb
    buf = hb.Buffer()
    buf.add_str(text)
    buf.guess_segment_properties()
    hb.shape(hb.Font(hb.Face(hb.Blob.from_file_path(path), 0)), buf, {"liga": True})
    return sum(pos.x_advance for pos in buf.glyph_positions)


def test_earlier_icon_fonts(clone, tmp_path):
    """At the ring font's commit: up to EARLIER_FONTS older icon fonts, newest first, none the ref's own, their (1CD) the
    pre-ring width (the ring's 3280 units), named so a FontMap over the folder never indexes them."""
    from psd_tools.composite.text.fonts import font_files
    source = Source(str(clone))
    commit = source.commit(RING) or pytest.skip(f"the clone lacks {RING}")
    tree = source.tree(commit)
    files = earlier_fonts(source, commit, tree, str(tmp_path))
    shas = [blob_sha(path) for path in files]
    assert 1 <= len(files) <= EARLIER_FONTS and len(set(shas)) == len(shas) and tree[ICON_FONT_FILE][1] not in shas
    assert font_files(str(tmp_path)) == []
    ring = tmp_path / "ring.ttf"
    ring.write_bytes(source.blob(tree[ICON_FONT_FILE][1]))
    assert [_advance(path, "(1CD)") for path in files] == [2040] * len(files) and _advance(str(ring), "(1CD)") == 3280
    assert kinds_of("badge-widths") == ["badge widths"]
