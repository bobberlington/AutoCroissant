#!/usr/bin/env python3
"""Inspect the card database read-only: the repo root's pickles (stats.pkl / old_stats.pkl), or with --db the data
build's db/ JSON (or any form commands.card_db.read_cards reads).

Usage (run from anywhere; repo root derived from this file's location):
    python3 .claude/skills/autocroissant-diagnostics-and-tooling/scripts/inspect_pickle.py            # summary
    python3 .claude/skills/autocroissant-diagnostics-and-tooling/scripts/inspect_pickle.py --problems # cards with recorded problems
    python3 .claude/skills/autocroissant-diagnostics-and-tooling/scripts/inspect_pickle.py "Card Name"  # one card, full detail + history

    # the same on the data build's db/: a folder, or a data commit of --git's repo (origin/data in a TTSCardMaker clone):
    python3 .claude/skills/autocroissant-diagnostics-and-tooling/scripts/inspect_pickle.py --db origin/data \
        --git ~/Desktop/TTSCardMaker "Card Name"

No config.py needed (the pickles load through card_db's allow-list unpickler). Never writes anything.
"""
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from commands.card_db import read_cards  # noqa: E402
from commands.card_models import StatsDatabase  # noqa: E402

stats_db = StatsDatabase()


def show_card(name: str) -> int:
    card = stats_db.stats.get(name)
    archived = stats_db.archived()
    if card is None and name in archived:                    # PSD deleted: its last version, tagged
        card = archived[name]
        print(f"archived: {card.archived}")
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
    if getattr(card, "version", None):
        print(f"{'version':12s}: {card.version!r}")

    history = stats_db.old_stats.get(name, [])
    print(f"\nold versions: {len(history)}")
    for i, old in enumerate(history):
        ab = (old.ability or "")[:60].replace("\n", "\\n")
        print(f"  [{i}] ts={datetime.fromtimestamp(old.timestamp)} type={old.card_type} ability[:60]={ab!r}")
    return 0


def summary() -> int:
    s = stats_db.stats
    print(f"stats     : {len(s)} cards")
    print(f"old_stats : {len(stats_db.old_stats)} cards with history, "
          f"{sum(len(v) for v in stats_db.old_stats.values())} archived versions total")
    archived = stats_db.archived()
    print(f"archived cards (PSD deleted): {len(archived)}"
          + (f" ({sum(1 for c in archived.values() if c.archived.get('backfill'))} backfilled)" if archived else ""))
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


def load(args: list) -> list:
    """Fill stats_db from --db (default: the repo root's pickles) and --git; returns the other arguments."""
    source, git = str(REPO_ROOT), "."
    for flag in ("--db", "--git"):
        if flag in args:
            i = args.index(flag)
            value, args = (args[i + 1:i + 2] or [""])[0], args[:i] + args[i + 2:]
            source, git = (value, git) if flag == "--db" else (source, value)
    stats, old = read_cards(source, git)
    stats_db.stats, stats_db.old_stats = stats, old if old is not None else defaultdict(list)
    return args


if __name__ == "__main__":
    try:
        rest = load(sys.argv[1:])
    except Exception as e:  # noqa: BLE001
        print(f"ERROR loading the card data: {e}")
        sys.exit(2)
    if not rest:
        sys.exit(summary())
    elif rest[0] == "--problems":
        sys.exit(problems())
    else:
        sys.exit(show_card(" ".join(rest)))
