"""
txt2_repair: a stale Txt2 story (a pre-fix bake's PSD, made here: the title set in TySh alone) is rewritten to its
layer's text and nothing else changes, byte for byte outside the Txt2 block; an in-sync PSD and --check write nothing.
Reads the card clone and needs the fork's story sync (fix/txt2-sync); skips without either. The five titles baked on
2026-10-22 are repaired into tmp while the clone still holds them stale.
"""
import hashlib
import struct
from pathlib import Path

import pytest
from psd_tools import PSDImage

import commands.psd_edit as pe
import commands.txt2_repair as tr
from conftest import fork_reason, story_sync, text_engine
from test_psd_edit import MINI_DOOMER, _card, _typo_fix, stale_copy

# ========================
# Configuration
# ========================
BAKED = ("Auxiliary/Minions/Clot_White.psd", "Auxiliary/Minions/PoD08MOAB.psd", "Auxiliary/Minions/PoD09DDT.psd",
         "Auxiliary/Minions/PoD10BFB.psd", "Auxiliary/Minions/PoD12BAD.psd")     # 2026-10-22's bakes: Name stale
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


def test_check_and_in_place(clone, tmp_path, capsys):
    title = pe.read_name(PSDImage.open(_card(clone, MINI_DOOMER)[0]))
    stale, _ = stale_copy(clone, MINI_DOOMER, _typo_fix(title), tmp_path)
    before = _sha(stale)
    assert tr.main(["--check", stale]) == 1 and _sha(stale) == before
    assert "stale: Name" in capsys.readouterr().out
    assert tr.main(["--in-place", stale]) == 0 and _sha(stale) != before
    assert tr.main(["--check", stale]) == 0 and capsys.readouterr().out.endswith("in sync\n")
    assert not list(tmp_path.glob(".*.repair"))


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


@pytest.mark.parametrize("rel", BAKED)
def test_baked_titles_of_2026_10_22(clone, rel, tmp_path):
    path = clone / rel
    if not path.exists():
        pytest.skip(f"{rel} not in the clone")
    stale = pe.stale_stories(PSDImage.open(str(path)))
    if not stale:
        pytest.skip(f"{rel} is in sync (repaired)")
    out = tmp_path / path.name
    result = tr.repair(str(path), str(out))
    assert [name for name, _, _ in result.stale] == ["Name"] and len(result.stories) == 1
    assert pe.stale_stories(PSDImage.open(str(out))) == [] and _outside_txt2(out) == _outside_txt2(path)
