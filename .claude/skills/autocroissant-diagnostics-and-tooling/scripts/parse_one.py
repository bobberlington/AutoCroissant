#!/usr/bin/env python3
"""Parse ONE card PSD through the real bot parser, without Discord and without
touching any pickle. This is the fastest way to see exactly what
/update_stats would extract for a single card.

Usage (run from anywhere; repo root is derived from this file's location):
    python3 .claude/skills/autocroissant-diagnostics-and-tooling/scripts/parse_one.py <path/to/card.psd> [--with-db]

    <path>      absolute or relative path to a .psd inside the local
                TTSCardMaker clone (default clone location: ~/Desktop/TTSCardMaker)
    --with-db   load stats.pkl first so author/series preservation behaves
                exactly like a real update run (default: parse fresh)

Notes:
  * Requires config.py to exist in the repo root (the bot module import chain
    reads it). No network calls are made and nothing is saved.
  * The repo-relative path is computed HERE, correctly, relative to the
    TTSCardMaker clone root. The bot's own local traversal
    (_process_local_files) currently computes it WRONG for absolute clone
    locations -- see the autocroissant-failure-archaeology skill ("removeprefix
    bug", introduced in commit f7c915c). This script shows what the parser
    produces when given the CORRECT relative path.
"""
import sys
from os.path import expanduser, relpath
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

import os
os.chdir(REPO_ROOT)  # pickle paths in global_config are repo-relative

from global_config import LOCAL_DIR_LOC


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

    from commands.psd_analyzer import (
        CardValidator,
        PSDParser,
        RepositoryTraverser,
        StatsDatabase,
        stats_db,
    )

    db = stats_db
    if "--with-db" in sys.argv:
        db.load()
        print(f"(loaded stats.pkl: {len(db.stats)} cards -- author/series preservation active)")
    else:
        db = StatsDatabase()  # fresh, empty: no preservation effects

    traverser = RepositoryTraverser(db)
    traverser._populate_types_from_local(str(clone_root))
    print(f"known types ({len(db.all_types)}): {sorted(db.all_types)}")
    print(f"relative path: {relative_path}")
    print("-" * 80)

    parser = PSDParser(db.all_types, db)
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
