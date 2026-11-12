"""
The bake's Txt2 story sync: a text layer whose own TySh text its story in the document's Txt2 block lacks (the copy
Photoshop reads, so opening and saving the card put the old text back: B.F.B. -> B.F.B, 2026-10-22) gets the story
rewritten to that text through the fork's TypeLayer.sync_story (text and run lengths; the story's styles stay, and
Photoshop redoes its glyph layout). psd_edit.apply_jobs runs it and gate_stories checks it. The guards' R9 repair (and
repair/verify) was deleted 2026-11-12 (owner). No Discord, network or config.

    sync_stories(psd) -> the story indices rewritten
"""
from psd_tools import PSDImage

import commands.psd_edit as psd_edit


# ========================
# Public API Functions
# ========================
def sync_stories(psd: PSDImage) -> set:
    """Rewrite every stale story of an open PSD to its layer's text (TypeLayer.sync_story); the indices rewritten.
    psd_edit.apply_jobs runs it on a card it bakes (owner 2026-10-24: repaired, not refused)."""
    return {psd_edit.story_index(layer) for layer in psd.descendants() if layer.kind == "type" and layer.sync_story()}


# ========================
# Module Exports
# ========================
__all__ = [
    'sync_stories',
]
