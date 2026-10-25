"""
txt2_repair: a stale Txt2 story (a pre-fix bake's PSD, made here: the title set in TySh alone) is rewritten to its
layer's text and nothing else changes, byte for byte outside the Txt2 block; an in-sync PSD writes nothing.
Reads the card clone and needs the fork's story sync (fix/txt2-sync); skips without either. The six titles baked on
2026-10-22 are repaired into tmp from their bake commits (main holds them in sync since 7b48db05 and 5a9e0fcf).
"""
import hashlib
import struct
from pathlib import Path

import pytest
from psd_tools import PSDImage

import commands.psd_edit as pe
import commands.txt2_repair as tr
from conftest import at_rev, fork_reason, story_sync, text_engine
from test_psd_edit import MINI_DOOMER, _card, _typo_fix, stale_copy

# ========================
# Configuration
# ========================
BAKED = (                                       # 2026-10-22's bakes: Name stale at the bake commit
    ("Auxiliary/Minions/Clot_White.psd", "d81631039734ba3c90683a68ced5de20f65304a3"),
    ("Auxiliary/Minions/PoD08MOAB.psd", "2d8cce88b9e936b37b5b4b41131fe439811f6ac3"),
    ("Auxiliary/Minions/PoD09DDT.psd", "705e3d3470412f3c063365755a847740ba78b740"),
    ("Auxiliary/Minions/PoD10BFB.psd", "c9fc1bbd94c060560addff6d7065eeed0955f264"),
    ("Auxiliary/Minions/PoD12BAD.psd", "01d8000592e1d07551e466558f2b5ee36343c233"),
    ("Auxiliary/Minions/Springtrap.psd", "c6dc6d08d5cfbbffe5b51f67a0b4ac9be913ca8f"),
)
pytestmark = pytest.mark.skipif(not (text_engine() and story_sync()), reason="needs the fork's Txt2 story sync: " + fork_reason())


def _outside_txt2(path) -> tuple:
    """The file's bytes before and after its Txt2 block (padding included)."""
    data = Path(path).read_bytes()
    at = data.rfind(b"8BIMTxt2")
    length = struct.unpack(">I", data[at + 8:at + 12])[0]
    return data[:at + 8], data[at + 12 + (length + 3) // 4 * 4:]


def _sha(path) -> str:
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()


# ========================
# Tests
# ========================
def test_repair_rewrites_the_story_and_nothing_else(clone, tmp_path):
    title = pe.read_name(PSDImage.open(_card(clone, MINI_DOOMER)[0]))
    new = _typo_fix(title)
    stale, index = stale_copy(clone, MINI_DOOMER, new, tmp_path)
    out = tmp_path / "out.psd"
    result = tr.repair(stale, str(out))
    assert result.stale == [("Name", title + "\r", new + "\r")] and result.stories == [index] and result.out == str(out)
    repaired = PSDImage.open(str(out))
    assert pe.stale_stories(repaired) == [] and pe.stories(repaired)[index] == new + "\r"
    assert _outside_txt2(out) == _outside_txt2(stale)
    # the story a fixed bake writes: the fork's set_text on the original
    fresh = PSDImage.open(_card(clone, MINI_DOOMER)[0])
    pe.name_layer(fresh).set_text(new)
    story = lambda psd: pe._raw(pe._story_list(pe._txt2(psd))[2][index])
    assert story(repaired) == story(fresh)


def test_in_sync_psd_is_left_alone(clone, tmp_path):
    out = tmp_path / "out.psd"
    result = tr.repair(_card(clone, MINI_DOOMER)[0], str(out))
    assert result.stale == [] and result.out == "" and not out.exists()


def test_verify_refuses_any_other_change(clone, tmp_path):
    title = pe.read_name(PSDImage.open(_card(clone, MINI_DOOMER)[0]))
    stale, index = stale_copy(clone, MINI_DOOMER, _typo_fix(title), tmp_path)
    out = tmp_path / "out.psd"
    tr.repair(stale, str(out))
    psd = PSDImage.open(str(out))
    layer = next(layer for layer in psd.descendants() if layer.kind != "type")
    layer.visible = not layer.visible
    other = tmp_path / "other.psd"
    pe.save_psd(psd, str(other), update_preview=False)
    with pytest.raises(pe.Refused, match="layer record changed"):
        tr.verify(stale, str(other), {index})


@pytest.mark.parametrize("rel, rev", BAKED)
def test_baked_titles_of_2026_10_22(clone, rel, rev, tmp_path):
    path = tmp_path / f"stale_{Path(rel).name}"
    path.write_bytes(at_rev(clone, rev, rel))
    out = tmp_path / Path(rel).name
    result = tr.repair(str(path), str(out))
    assert [name for name, _, _ in result.stale] == ["Name"] and len(result.stories) == 1
    assert pe.stale_stories(PSDImage.open(str(out))) == [] and _outside_txt2(out) == _outside_txt2(path)


def test_failed_check_writes_nothing(clone, tmp_path, monkeypatch):
    title = pe.read_name(PSDImage.open(_card(clone, MINI_DOOMER)[0]))
    stale, _ = stale_copy(clone, MINI_DOOMER, _typo_fix(title), tmp_path)
    monkeypatch.setattr(tr, "verify", lambda *args: (_ for _ in ()).throw(pe.Refused("bytes", "a layer record changed")))
    out = tmp_path / "out"
    out.mkdir()
    assert tr.main(["--out", str(out), stale]) == 2
    assert not any(out.iterdir())
