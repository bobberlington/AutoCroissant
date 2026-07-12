#!/usr/bin/env python3
"""Diff two stats.pkl snapshots. RUN THIS BEFORE EVERY PICKLE PUSH.

The single most damaging failure class in this project is committing a
corrupted stats.pkl (see autocroissant-failure-archaeology). This tool turns
"the pickle looks fine" into numbers.

Usage (run from the repo root):
    # compare the last committed stats.pkl against the working one:
    git show HEAD:stats.pkl > /tmp/stats_head.pkl
    python3 .claude/skills/autocroissant-diagnostics-and-tooling/scripts/diff_stats.py /tmp/stats_head.pkl stats.pkl

    # or any two snapshots:
    python3 .claude/skills/autocroissant-diagnostics-and-tooling/scripts/diff_stats.py <old.pkl> <new.pkl> [--verbose]

Exit code 0 = no changes; 1 = changes found (added/removed/modified);
2 = usage/load error. --verbose prints per-field diffs for every modified card.

Red flags to look for in the output:
  * "removed" cards you did not delete  -> traversal missed files (bad mode/path)
  * mass "path" changes                 -> the removeprefix local-path bug
  * mass "type -> unknown"              -> classification broke
  * huge "added to old_stats" counts    -> duplicate-archiving regression
"""
import sys
from pathlib import Path
from pickle import load

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))


def norm(entry):
    """Normalize a stats entry (CardInfo dataclass or legacy dict) to a dict."""
    if hasattr(entry, "to_dict"):
        return entry.to_dict()
    return dict(entry)


def load_stats(path):
    with open(path, "rb") as f:
        data = load(f)
    return {name: norm(card) for name, card in data.items()}


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    verbose = "--verbose" in sys.argv
    try:
        old = load_stats(sys.argv[1])
        new = load_stats(sys.argv[2])
    except Exception as e:  # noqa: BLE001
        print(f"ERROR loading pickles: {e}")
        print("(unpickling CardInfo requires running from a checkout whose "
              "commands/psd_analyzer.py still defines it, with config.py present)")
        return 2

    added = sorted(set(new) - set(old))
    removed = sorted(set(old) - set(new))
    modified = {}
    for name in set(old) & set(new):
        fields = {}
        for key in set(old[name]) | set(new[name]):
            if old[name].get(key) != new[name].get(key):
                fields[key] = (old[name].get(key), new[name].get(key))
        if fields:
            modified[name] = fields

    print(f"old: {len(old)} cards   new: {len(new)} cards")
    print(f"added: {len(added)}   removed: {len(removed)}   modified: {len(modified)}")

    if added:
        print("\n-- added --")
        for n in added:
            print(f"  + {n} [{new[n].get('path','?')}]")
    if removed:
        print("\n-- removed --  (unexpected removals = traversal problem!)")
        for n in removed:
            print(f"  - {n} [{old[n].get('path','?')}]")
    if modified:
        print("\n-- modified --")
        # aggregate which fields changed most (mass-change detector)
        from collections import Counter
        field_counts = Counter(f for fields in modified.values() for f in fields)
        print(f"  field change counts: {dict(field_counts.most_common())}")
        for n, fields in sorted(modified.items()):
            if verbose:
                print(f"  * {n}:")
                for f, (a, b) in fields.items():
                    print(f"      {f}: {a!r} -> {b!r}")
            else:
                print(f"  * {n}: {sorted(fields)}")

    return 1 if (added or removed or modified) else 0


if __name__ == "__main__":
    sys.exit(main())
