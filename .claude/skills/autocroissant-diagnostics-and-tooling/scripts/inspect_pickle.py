#!/usr/bin/env python3
"""Inspect the card database pickles (stats.pkl / old_stats.pkl) read-only.

Usage (run from anywhere; repo root derived from this file's location):
    python3 .claude/skills/autocroissant-diagnostics-and-tooling/scripts/inspect_pickle.py            # summary
    python3 .claude/skills/autocroissant-diagnostics-and-tooling/scripts/inspect_pickle.py --problems # cards with recorded problems
    python3 .claude/skills/autocroissant-diagnostics-and-tooling/scripts/inspect_pickle.py "Card Name"  # one card, full detail + history

Requires config.py in the repo root (import chain). Never writes anything.
"""
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))
import os
os.chdir(REPO_ROOT)

from commands.psd_analyzer import stats_db  # noqa: E402


def show_card(name: str) -> int:
    card = stats_db.stats.get(name)
    if card is None:
        # case-insensitive fallback
        lowered = {k.lower(): k for k in stats_db.stats}
        real = lowered.get(name.lower())
        if real is None:
            candidates = [k for k in stats_db.stats if name.lower() in k.lower()]
            print(f"not found: {name!r}")
            if candidates:
                print("did you mean: " + ", ".join(sorted(candidates)[:10]))
            return 1
        name, card = real, stats_db.stats[real]

    print(f"=== {name} ===")
    for k, v in card.to_dict().items():
        if k == "timestamp":
            v = f"{v} ({datetime.fromtimestamp(v)})"
        print(f"{k:12s}: {v!r}")

    history = stats_db.old_stats.get(name, [])
    print(f"\nold versions: {len(history)}")
    for i, old in enumerate(history):
        ab = (old.ability or "")[:60].replace("\n", "\\n")
        print(f"  [{i}] ts={datetime.fromtimestamp(old.timestamp)} type={old.card_type} ability[:60]={ab!r}")
    return 0


def summary() -> int:
    s = stats_db.stats
    print(f"stats.pkl     : {len(s)} cards")
    print(f"old_stats.pkl : {len(stats_db.old_stats)} cards with history, "
          f"{sum(len(v) for v in stats_db.old_stats.values())} archived versions total")
    print("\ncard_type distribution:")
    for t, n in Counter(c.card_type for c in s.values()).most_common():
        print(f"  {t:12s} {n}")
    n_problems = sum(1 for c in s.values() if c.problems)
    n_no_author = sum(1 for c in s.values() if not c.author)
    n_rulebook = sum(1 for c in s.values() if "Rulebook" in c.path)
    newest = max(s.values(), key=lambda c: c.timestamp)
    print(f"\ncards with recorded parse problems: {n_problems}")
    print(f"cards without author (orphans)    : {n_no_author}")
    print(f"rulebook entries                  : {n_rulebook}")
    print(f"newest timestamp: {newest.name} @ {datetime.fromtimestamp(newest.timestamp)}")
    bad_paths = [c.name for c in s.values()
                 if c.path and c.path.split('/')[0] not in
                 ("Creatures", "Items", "Field", "Auxiliary", "N.M.E", "MDW", "Rulebook", "Types")]
    print(f"entries with suspicious (non repo-relative) paths: {len(bad_paths)}"
          + (f" e.g. {bad_paths[:3]}" if bad_paths else ""))
    return 0


def problems() -> int:
    for name, c in sorted(stats_db.stats.items()):
        if c.problems:
            print(f"{name} [{c.path}]: {c.problems}")
    return 0


if __name__ == "__main__":
    stats_db.load()
    if len(sys.argv) == 1:
        sys.exit(summary())
    elif sys.argv[1] == "--problems":
        sys.exit(problems())
    else:
        sys.exit(show_card(" ".join(sys.argv[1:])))
