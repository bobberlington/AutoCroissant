"""
Repair for the card PSDs a bake left with a stale Txt2 story (the five titles of 2026-10-22): the bake changed a text
layer's own TySh text but not its story in the document's Txt2 block, which is the copy Photoshop reads, so opening
and saving the card in Photoshop put the old text back (B.F.B. -> B.F.B). Per PSD, every type layer whose story lacks
its TySh text gets the story rewritten to that text through the fork's TypeLayer.sync_story (text and run lengths; the
story's styles stay, and Photoshop redoes its glyph layout), and nothing else changes: the PSD is saved without a new
preview, then checked (every story holds its layer's text and styles, only the repaired stories changed, the rest of
Txt2 and the other document blocks are the old bytes: psd_edit.gate_stories; every layer record, the image resources
and the stored composite are the old bytes too).

    python -m commands.txt2_repair <psd>... (--out DIR | --in-place | --check)
    repair(path, out_path) -> Repair; verify(old_path, new_path, stories) -> the changed stories

--check writes nothing and exits 1 when a story is stale. Needs the psd-tools fork with sync_story (fix/txt2-sync);
no Discord, network or config.
"""
from argparse import ArgumentParser
from dataclasses import dataclass, field
from os import makedirs, remove, replace
from os.path import basename, dirname, exists, join
from sys import stderr
from typing import Optional

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
    out: str = ""                                   # the written PSD; "" when nothing was stale (or --check)
    stale: list = field(default_factory=list)       # (layer name, story text, layer text) as found
    stories: list = field(default_factory=list)     # indices of the stories rewritten


# ========================
# Helper Functions
# ========================
def _raw(element) -> bytes:
    return element.tobytes() if element is not None else b""


def _show(text: Optional[str]) -> str:
    return repr(psd_edit.collapse(text))


# ========================
# Public API Functions
# ========================
def verify(old_path: str, new_path: str, stories) -> list:
    """
    Check a repaired PSD against its original: only the given stories changed, every story holds its layer's text,
    and every layer record, the document's other blocks, the image resources and the stored composite are the old bytes.

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
    for part in ("header", "color_mode_data", "image_resources", "image_data"):
        if _raw(getattr(old._record, part)) != _raw(getattr(new._record, part)):
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
    if not psd_edit.story_sync():
        raise ImportError("this psd-tools has no TypeLayer.sync_story (the fork's fix/txt2-sync)")
    synced = {psd_edit.story_index(layer) for layer in psd.descendants() if layer.kind == "type" and layer.sync_story()}
    temp = join(dirname(out_path) or ".", f".{basename(out_path)}.repair")
    try:
        psd_edit.save_psd(psd, temp, update_preview=False)
        found.stories = verify(path, temp, synced)
    except BaseException:
        if exists(temp):
            remove(temp)
        raise
    replace(temp, out_path)
    found.out = out_path
    return found


def main(args: Optional[list] = None) -> int:
    """The command line: one line per PSD; exit 1 when --check finds a stale story, 2 when a PSD fails."""
    parser = ArgumentParser(prog="python -m commands.txt2_repair",
                            description="Rewrite the Txt2 stories a bake left stale to their layers' text (Photoshop "
                                        "reads the story); nothing else changes.")
    parser.add_argument("psd", nargs="+", help="PSD files")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--out", metavar="DIR", help="write the repaired PSDs here (same file names)")
    mode.add_argument("--in-place", action="store_true", help="overwrite each PSD (written only when it needs it)")
    mode.add_argument("--check", action="store_true", help="report the stale stories, write nothing")
    options = parser.parse_args(args)
    if options.out:
        makedirs(options.out, exist_ok=True)
    status = 0
    for path in options.psd:
        try:
            if options.check:
                result = Repair(path, stale=psd_edit.stale_stories(PSDImage.open(path)))
            else:
                result = repair(path, path if options.in_place else join(options.out, basename(path)))
        except (Refused, ImportError, OSError, ValueError) as error:
            print(f"{path}: failed: {error}", file=stderr)
            status = 2
            continue
        if not result.stale:
            print(f"{path}: in sync")
            continue
        found = "; ".join(f"{name} {_show(story)} -> {_show(own)}" for name, story, own in result.stale)
        if options.check:
            print(f"{path}: stale: {found}")
            status = max(status, 1)
        else:
            print(f"{path}: repaired {found} (stories {result.stories}) -> {result.out}")
    return status


# ========================
# Module Exports
# ========================
__all__ = [
    'Repair',
    'main',
    'repair',
    'verify',
]


if __name__ == "__main__":
    raise SystemExit(main())
