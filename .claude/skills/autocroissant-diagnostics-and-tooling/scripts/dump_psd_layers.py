#!/usr/bin/env python3
"""Dump the layer tree of a card PSD the way the parser sees it.

Usage:
    python3 .claude/skills/autocroissant-diagnostics-and-tooling/scripts/dump_psd_layers.py <path/to/card.psd> [--text]

Prints one line per layer: depth-indented name, kind, visibility, pixel
presence, and bbox (x1, y1, x2, y2). With --text, also prints the raw
engine_dict text of every text layer ("type" kind), which is exactly the
string commands/psd_analyzer.py starts from before cleanup.

Only needs psd-tools; does NOT import bot code and never writes anything.
"""
import sys

from psd_tools import PSDImage


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    path = sys.argv[1]
    show_text = "--text" in sys.argv

    psd = PSDImage.open(path)
    print(f"PSD: {path}")
    print(f"size: {psd.width} x {psd.height}  (mid-y at ratio 0.5 = {int(psd.height * 0.5)})")
    print("-" * 100)
    print(f"{'layer (indent = depth)':60s} {'kind':10s} {'vis':4s} {'pix':4s} bbox")
    print("-" * 100)

    def walk(layer, depth):
        for child in layer:
            try:
                has_pix = child.has_pixels()
            except Exception:
                has_pix = "?"
            name = ("  " * depth) + child.name
            print(f"{name:60.60s} {child.kind:10s} {str(child.is_visible()):4s} {str(has_pix):4s} {child.bbox}")
            if show_text and child.kind == "type":
                raw = str(child.engine_dict["Editor"]["Text"])
                print(f"{'':60s} TEXT: {raw!r}")
            if child.is_group():
                walk(child, depth + 1)

    walk(psd, 0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
