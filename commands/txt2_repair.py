"""
Repair for the card PSDs a bake left with a stale Txt2 story (the five titles of 2026-10-22): the bake changed a text
layer's own TySh text but not its story in the document's Txt2 block, which is the copy Photoshop reads, so opening
and saving the card in Photoshop put the old text back (B.F.B. -> B.F.B). Per PSD, every type layer whose story lacks
its TySh text gets the story rewritten to that text through the fork's TypeLayer.sync_story (text and run lengths; the
story's styles stay, and Photoshop redoes its glyph layout), and nothing else changes: the PSD is saved without a new
preview and stripped (psd_edit.save_psd), then checked (every story holds its layer's text and styles, only the
repaired stories changed, the rest of Txt2 and the other document blocks are the old bytes: psd_edit.gate_stories;
every layer record and the image resources are the old bytes too).

    repair(path, out_path) -> Repair; verify(old_path, new_path, stories) -> the changed stories

Needs the psd-tools fork with sync_story (fix/txt2-sync); no Discord, network or config.
"""
from dataclasses import dataclass, field
from os import remove, replace
from os.path import basename, dirname, exists, join

from psd_tools import PSDImage

import commands.psd_edit as psd_edit
from commands.psd_edit import Refused


# ========================
# Data
# ========================
@dataclass
class Repair:
    """One PSD's repair."""
    path: str
    out: str = ""                                   # the written PSD; "" when nothing was stale
    stale: list = field(default_factory=list)       # (layer name, story text, layer text) as found
    stories: list = field(default_factory=list)     # indices of the stories rewritten


# ========================
# Public API Functions
# ========================
def sync_stories(psd: PSDImage) -> set:
    """Rewrite every stale story of an open PSD to its layer's text (TypeLayer.sync_story); the indices rewritten.
    psd_edit.apply_jobs runs it on a card it bakes (owner 2026-10-24: repaired, not refused)."""
    return {psd_edit.story_index(layer) for layer in psd.descendants() if layer.kind == "type" and layer.sync_story()}


def verify(old_path: str, new_path: str, stories) -> list:
    """
    Check a repaired PSD against its original: only the given stories changed, every story holds its layer's text,
    and every layer record, the document's other blocks and the image resources are the old bytes (the merged image's
    strip aside: psd_edit.changed_part).

    Args:
        old_path: The PSD as it was
        new_path: The repaired PSD
        stories: The story indices (TextIndex) the repair may change

    Returns:
        The changed story indices, sorted; raises Refused("bytes")
    """
    old, new = PSDImage.open(old_path), PSDImage.open(new_path)
    changed = psd_edit.gate_stories(old, new, stories)
    if psd_edit.layer_records(old_path) != psd_edit.layer_records(new_path):
        raise Refused("bytes", "a layer record changed")
    part = psd_edit.changed_part(old, new)
    if part:
        raise Refused("bytes", f"{part} changed")
    return changed


def repair(path: str, out_path: str) -> Repair:
    """
    Rewrite every stale Txt2 story of a PSD to its layer's TySh text and save the result to out_path (may be path).

    Args:
        path: The PSD
        out_path: Where the repaired PSD goes; not written when nothing is stale

    Returns:
        Repair; raises Refused("bytes") when the check fails (out_path is then not written), ImportError without
        the fork's sync_story
    """
    psd = PSDImage.open(path)
    found = Repair(path, stale=psd_edit.stale_stories(psd))
    if not found.stale:
        return found
    synced = sync_stories(psd)
    temp = join(dirname(out_path) or ".", f".{basename(out_path)}.repair")
    try:
        psd_edit.save_psd(psd, temp)
        found.stories = verify(path, temp, synced)
    except BaseException:
        if exists(temp):
            remove(temp)
        raise
    replace(temp, out_path)
    found.out = out_path
    return found


# ========================
# Module Exports
# ========================
__all__ = [
    'Repair',
    'repair',
    'sync_stories',
    'verify',
]
