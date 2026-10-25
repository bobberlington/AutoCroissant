#!/usr/bin/env python3
"""Parse ONE card PSD through the real bot parser, without Discord and without
writing anything. This is the fastest way to see exactly what the card data
build (Card sync) would extract for a single card.

Usage (run from anywhere; repo root is derived from this file's location):
    python3 .claude/skills/autocroissant-diagnostics-and-tooling/scripts/parse_one.py <path/to/card.psd> [--with-db]

    <path>      absolute or relative path to a .psd inside the local
                TTSCardMaker clone (default clone location: ~/Desktop/TTSCardMaker)
    --with-db   load the bot's card_data/ copy first so author/series
                preservation behaves exactly like a real build (default: parse fresh)

Notes:
  * Requires config.py to exist in the repo root (the bot module import chain
    reads it). No network calls are made and nothing is saved.
  * The repo-relative path is computed here with relpath against the
    TTSCardMaker clone root, which gives the same repo-relative paths that
    the build stores from the git tree (the archived local traversal on
    branch archive/local-update-stats used the same construction; story:
    autocroissant-failure-archaeology).
  * The stoogedings icon vocabulary (Types/, Types/Stars/, aliases and
    codepoints from the font's manifest) is read from the clone and matches
    what the build reads from the tree, so cards that use the icon font
    parse exactly as in production.
"""
import sys
from os.path import expanduser, relpath
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

import os
os.chdir(REPO_ROOT)  # card_data/ is repo-relative

from global_config import CARD_DATA_DIR, LOCAL_DIR_LOC


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1

    psd_path = Path(sys.argv[1]).expanduser().resolve()
    if not psd_path.exists():
        print(f"ERROR: file not found: {psd_path}")
        return 1

    clone_root = Path(expanduser(LOCAL_DIR_LOC)).resolve()
    try:
        relative_path = relpath(psd_path, clone_root).replace("\\", "/")
    except ValueError:
        relative_path = psd_path.name
    if relative_path.startswith(".."):
        print(f"WARNING: {psd_path} is not under the clone root {clone_root}; "
              f"classification (which is folder-based) will be UNKNOWN.")
        relative_path = psd_path.name

    from commands.card_data import CardBuild
    from commands.psd_analyzer import StatsDatabase
    from commands.psd_parser import CardValidator, PSDParser

    db = StatsDatabase()  # fresh, empty: no preservation effects
    if "--with-db" in sys.argv:
        from commands.card_db import read_db
        db.stats, db.old_stats, _, _ = read_db(CARD_DATA_DIR, details=False)
        print(f"(loaded {CARD_DATA_DIR}/: {len(db.stats)} cards -- author/series preservation active)")

    traverser = CardBuild(db)
    traverser._populate_types_from_local(str(clone_root))
    print(f"known types ({len(db.all_types)}): {sorted(db.all_types)}")
    print(f"relative path: {relative_path}")
    print("-" * 80)

    parser = PSDParser(db.all_types, db, traverser._icons_from_local(str(clone_root)))
    card = parser.parse(str(psd_path), relative_path)
    problems = CardValidator.validate(card)

    d = card.to_dict()
    ability = d.pop("ability", None)
    for k, v in d.items():
        print(f"{k:12s}: {v}")
    print("ability:")
    print("=" * 40)
    print(ability if ability is not None else "<none>")
    print("=" * 40)
    print(f"validator problems: {problems if problems else 'NONE'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
