#!/usr/bin/env python3
"""Trace the type-injection pipeline for one card PSD, step by step.

This is THE tool for debugging wrong [type] placement in ability text (the
project's hardest recurring problem). It shows every intermediate value the
real parser produces: raw text layers, sorted order, type-icon bboxes, the
midline/prune decisions, the 3+ space gaps, and the final injected text.

Usage:
    python3 .claude/skills/autocroissant-diagnostics-and-tooling/scripts/gap_trace.py <path/to/card.psd>

Requires config.py in the repo root. Read-only; nothing is saved.

MAINTENANCE NOTE: the extraction loop below deliberately re-drives the REAL
PSDParser methods (_process_layer, _sort_by_position, _prune_type_bboxes,
_inject_type_names) instead of copying their logic, but the orchestration
mirrors PSDParser._extract_from_layers. If _extract_from_layers changes shape
in commands/psd_analyzer.py, update this script to match (one-line check:
compare against that method's body).
"""
import sys
from os.path import expanduser, relpath
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))
import os
os.chdir(REPO_ROOT)

from psd_tools import PSDImage  # noqa: E402

from global_config import LOCAL_DIR_LOC  # noqa: E402


def visualize_gaps(text: str, gap_pattern) -> str:
    """Make 3+ space runs visible as <GAP:n> markers."""
    def mark(m):
        return f"<GAP:{len(m.group(0))}>"
    return "\n".join(gap_pattern.sub(mark, line) for line in text.splitlines())


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    psd_path = Path(sys.argv[1]).expanduser().resolve()
    clone_root = Path(expanduser(LOCAL_DIR_LOC)).resolve()
    try:
        relative_path = relpath(psd_path, clone_root).replace("\\", "/")
    except ValueError:
        relative_path = psd_path.name

    from commands.psd_analyzer import (
        TYPE_REGION_RATIO,
        MutableValue,
        PSDParser,
        RepositoryTraverser,
        StatsDatabase,
        StatTrackers,
        CardClassifier,
        CardInfo,
        CardType,
    )

    db = StatsDatabase()
    RepositoryTraverser(db)._populate_types_from_local(str(clone_root))
    parser = PSDParser(db.all_types, db)

    psd = PSDImage.open(str(psd_path))
    card_mid_y = int(psd.height * TYPE_REGION_RATIO)
    print(f"PSD {psd.width}x{psd.height}; TYPE_REGION_RATIO={TYPE_REGION_RATIO} -> card_mid_y={card_mid_y}")
    print(f"relative path: {relative_path}")

    # --- mirror PSDParser._extract_from_layers ---
    card_dict = CardClassifier.classify(relative_path)
    card = CardInfo(name=psd_path.stem.replace("_", " "),
                    card_type=card_dict.get("type", CardType.UNKNOWN.value))
    num_stars = MutableValue(0)
    stat_trackers = StatTrackers()
    type_bboxes: list = []
    abilities: list = []
    is_rulepage = "Rulebook" in relative_path
    get_stars_from_psd = any(
        folder in relative_path
        for folder in ["Auxiliary/Items", "Auxiliary/Minions", "N.M.E"]
    )
    for layer in psd.descendants():
        parser._process_layer(layer, card, num_stars, stat_trackers,
                              type_bboxes, abilities, is_rulepage,
                              get_stars_from_psd, card_mid_y)
    # --- end mirror ---

    print(f"\n[1] creature-types collected ABOVE midline: {card.types}")

    print(f"\n[2] ability text layers found: {len(abilities)}")
    for text, bbox in abilities:
        print(f"    at (x={bbox.x}, y={bbox.y}): {text!r}")

    sorted_abilities = parser._sort_by_position(abilities)
    joined = "\n".join(text.strip("'\"\n") for text, _ in sorted_abilities)
    print("\n[3] joined ability text with gaps marked (3+ spaces = injection slots):")
    print(visualize_gaps(joined, parser._gap_pattern))

    print(f"\n[4] type icons BELOW midline (candidate inline types): {len(type_bboxes)}")
    sorted_bboxes = parser._sort_by_position(type_bboxes)
    for t, bbox in sorted_bboxes:
        print(f"    {t:15s} at (x={bbox.x}, y={bbox.y})")

    pruned = parser._prune_type_bboxes(sorted_bboxes, card_mid_y)
    if sorted_bboxes:
        last_y = sorted_bboxes[len(sorted_bboxes) - 1][1].y
        print(f"    prune threshold = max(last_y//3={last_y // 3}, card_mid_y={card_mid_y})")
    dropped = [t for t in sorted_bboxes if t not in pruned]
    print(f"    kept after prune: {[t for t, _ in pruned]}   dropped: {[t for t, _ in dropped]}")

    n_gaps = sum(len(parser._gap_pattern.findall(line)) for line in joined.splitlines())
    print(f"\n[5] gap count = {n_gaps} vs kept types = {len(pruned)}"
          + ("   <-- MISMATCH, leftover types get appended to last line!" if n_gaps != len(pruned) else "   (match)"))

    final = parser._process_abilities(sorted_abilities, list(pruned), card_mid_y)
    print("\n[6] FINAL ability text after injection + punctuation cleanup:")
    print("=" * 40)
    print(final)
    print("=" * 40)
    return 0


if __name__ == "__main__":
    sys.exit(main())
