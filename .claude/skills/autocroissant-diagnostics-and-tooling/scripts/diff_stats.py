#!/usr/bin/env python3
"""Diff two stats.pkl snapshots, and their history. RUN THIS BEFORE EVERY PICKLE PUSH.

The single most damaging failure class in this project is committing a
corrupted stats.pkl (see autocroissant-failure-archaeology). This tool turns
"the pickle looks fine" into numbers.

Usage (run from the repo root):
    # compare the last committed pickles against the working ones:
    git show HEAD:stats.pkl > /tmp/stats_head.pkl && git show HEAD:old_stats.pkl > /tmp/old_stats_head.pkl
    python3 .claude/skills/autocroissant-diagnostics-and-tooling/scripts/diff_stats.py /tmp/stats_head.pkl stats.pkl \
        --history /tmp/old_stats_head.pkl old_stats.pkl

    # or any two snapshots (an old_stats.pkl beside each stats.pkl is read without --history):
    python3 .claude/skills/autocroissant-diagnostics-and-tooling/scripts/diff_stats.py <old.pkl> <new.pkl> [--verbose]

Exit code 0 = no changes; 1 = changes found (added/removed/modified cards, or
history: versions added or dropped, starts moved, archive tags, blobs); 2 =
usage/load error. --verbose prints per-field diffs for every modified card.

Red flags to look for in the output:
  * "removed" cards you did not delete  -> traversal missed files (bad mode/path)
  * mass "path" changes                 -> the removeprefix local-path bug
  * mass "type -> unknown"              -> classification broke
  * huge "versions added" counts        -> duplicate-archiving regression (the 15:41 run: +280)
  * starts moved you did not expect     -> a history cleanup or re-parse moved website dates
"""
import sys
from collections import Counter
from pathlib import Path
from pickle import load

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))
BOOKKEEPING = ("blob_sha", "parser_version")       # not in to_dict(); a version's start is read from `version`


def norm(entry):
    """Normalize a stats entry (CardInfo dataclass or legacy dict) to a dict."""
    if hasattr(entry, "to_dict"):
        return entry.to_dict()
    return dict(entry)


def load_pickle(path):
    with open(path, "rb") as f:
        return load(f)


def start(card):
    """When a version starts: its `version` record's from, else its timestamp (card_history.start)."""
    info = getattr(card, "version", None)
    if isinstance(info, dict) and info.get("from") is not None:
        return info["from"]
    return getattr(card, "timestamp", None) if not isinstance(card, dict) else card.get("timestamp")


def key(card):
    """A stored version's identity: its timestamp and blob."""
    return (getattr(card, "timestamp", None), getattr(card, "blob_sha", None))


def history_diff(old_stats, new_stats, old_hist, new_hist, verbose):
    """Print the history section; returns whether anything changed."""
    def count(hist):
        return sum(len(v) for v in hist.values())
    print(f"\n-- history --\nold: {len(old_hist)} names {count(old_hist)} versions   "
          f"new: {len(new_hist)} names {count(new_hist)} versions")
    added, dropped, starts, tags, data = Counter(), Counter(), [], [], []
    for name in sorted(set(old_hist) | set(new_hist)):
        a, b = list(old_hist.get(name) or []), list(new_hist.get(name) or [])
        ka, kb = Counter(map(key, a)), Counter(map(key, b))
        if kb - ka:
            added[name] = sum((kb - ka).values())
        if ka - kb:
            dropped[name] = sum((ka - kb).values())
        by_key = {key(v): v for v in a}
        for v in b:
            w = by_key.get(key(v))
            if w is None:
                continue
            if start(w) != start(v):
                starts.append((f"{name} (version {key(v)[0]})", start(w), start(v)))
            if norm(w) != norm(v):
                data.append(name)
        last_a, last_b = (getattr(a[len(a) - 1], "archived", None) if a else None,
                          getattr(b[len(b) - 1], "archived", None) if b else None)
        if last_a != last_b:
            tags.append((name, last_a, last_b))
    blobs = []
    for name in sorted(set(old_stats) & set(new_stats)):
        w, v = old_stats[name], new_stats[name]
        if start(w) != start(v):
            starts.append((name, start(w), start(v)))
        changed = [f for f in BOOKKEEPING if getattr(w, f, None) != getattr(v, f, None)]
        if changed:
            blobs.append((name, changed))
    print(f"versions added: {sum(added.values())} ({len(added)} names)   "
          f"dropped: {sum(dropped.values())} ({len(dropped)} names)")
    print(f"starts moved: {len(starts)}   archive tags changed: {len(tags)}   "
          f"kept versions with other data: {len(set(data))}   cards with another blob or parser: {len(blobs)}")
    rows = [("+", n, c) for n, c in sorted(added.items())] + [("-", n, c) for n, c in sorted(dropped.items())]
    for sign, name, n in rows if verbose else rows[:40]:
        print(f"  {sign}{n} {name}")
    for name, a, b in starts if verbose else starts[:40]:
        print(f"  start {name}: {a} -> {b}")
    for name, a, b in tags:
        print(f"  tag {name}: {a} -> {b}")
    for name in sorted(set(data)):
        print(f"  data {name}")
    if verbose:
        for name, fields in blobs:
            print(f"  bookkeeping {name}: {fields}")
    return bool(added or dropped or starts or tags or data or blobs)


def main() -> int:
    args = [a for a in sys.argv[1:] if a != "--verbose"]
    history = None
    if "--history" in args:
        i = args.index("--history")
        history, args = args[i + 1:i + 3], args[:i] + args[i + 3:]
    if len(args) < 2 or (history is not None and len(history) < 2):
        print(__doc__)
        return 2
    verbose = "--verbose" in sys.argv
    if history is None:
        beside = [Path(p).with_name("old_stats.pkl") for p in args[:2]]
        history = [str(p) for p in beside] if all(p.is_file() for p in beside) else None
    try:
        old_raw, new_raw = load_pickle(args[0]), load_pickle(args[1])
        old = {name: norm(card) for name, card in old_raw.items()}
        new = {name: norm(card) for name, card in new_raw.items()}
        hists = [load_pickle(p) for p in history] if history else None
    except Exception as e:  # noqa: BLE001
        print(f"ERROR loading pickles: {e}")
        print("(unpickling CardInfo requires running from a checkout where "
              "commands.psd_analyzer imports cleanly and exports it, with config.py present)")
        return 2

    added = sorted(set(new) - set(old))
    removed = sorted(set(old) - set(new))
    modified = {}
    for name in set(old) & set(new):
        fields = {}
        for k in set(old[name]) | set(new[name]):
            if old[name].get(k) != new[name].get(k):
                fields[k] = (old[name].get(k), new[name].get(k))
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
        field_counts = Counter(f for fields in modified.values() for f in fields)
        print(f"  field change counts: {dict(field_counts.most_common())}")
        for n, fields in sorted(modified.items()):
            if verbose:
                print(f"  * {n}:")
                for f, (a, b) in fields.items():
                    print(f"      {f}: {a!r} -> {b!r}")
            else:
                print(f"  * {n}: {sorted(fields)}")

    changed_history = history_diff(old_raw, new_raw, *hists, verbose) if hists else False
    if not hists:
        print("\n(history not compared: pass --history <old old_stats.pkl> <new old_stats.pkl>)")
    return 1 if (added or removed or modified or changed_history) else 0


if __name__ == "__main__":
    sys.exit(main())
