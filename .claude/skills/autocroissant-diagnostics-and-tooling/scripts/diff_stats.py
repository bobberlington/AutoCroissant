#!/usr/bin/env python3
"""Diff two card databases (stats and their history): the data build's db/ JSON.
Run it on Card sync data commits (the data gate) and for the shadow's daily check of the data branch.

Committing corrupted card data is the most damaging failure class in this project (see
autocroissant-failure-archaeology). This tool turns "the data looks fine" into numbers.

Each side is any form the card data takes (commands.card_db.read_cards): a stats.json (the old_stats.json beside it
is the history), a db/ folder (or a folder holding db/), or a data commit of --git's repo (default: here), such as
origin/data in a TTSCardMaker clone. No config.py is needed.

Usage (run from the repo root):
    # the data branch's last commit against the one before (what a Card sync run changed):
    git -C ~/Desktop/TTSCardMaker fetch -q --depth 2 origin +data:refs/remotes/origin/data
    python3 .claude/skills/autocroissant-diagnostics-and-tooling/scripts/diff_stats.py origin/data~1 origin/data \
        --git ~/Desktop/TTSCardMaker

    # any two (a db/ folder from a hand build, an Actions artifact...), history files named explicitly:
    python3 .claude/skills/autocroissant-diagnostics-and-tooling/scripts/diff_stats.py <old> <new> [--verbose] \
        [--history <old old_stats.json> <new old_stats.json>]

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

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))
from commands.card_db import parse_old_stats, read_cards  # noqa: E402
from commands.card_history import data_of  # noqa: E402

BOOKKEEPING = ("blob_sha", "parser_version")       # not in to_dict(); a version's start is read from `version`


def norm(entry):
    """A stats entry (CardInfo) as a dict."""
    return entry.to_dict()


def load_history(path):
    """An old_stats.json named by --history."""
    with open(path, encoding="utf-8") as fh:
        return parse_old_stats(fh.read(), path)


def start(card):
    """When a version starts: its `version` record's from, else its timestamp (card_history.start)."""
    info = getattr(card, "version", None)
    if isinstance(info, dict) and info.get("from") is not None:
        return info["from"]
    return getattr(card, "timestamp", None)


def key(card):
    """A stored version's identity: its timestamp and blob."""
    return (getattr(card, "timestamp", None), getattr(card, "blob_sha", None))


def pair(a, b):
    """
    Two lists of one name's stored versions matched up: one version when the same timestamp and blob; else the same
    blob (re-dated from git); else, where one side has no blob (a legacy version), the same timestamp, or the same
    data in today's forms.

    Returns:
        ([(a's version, b's version)], a's unmatched, b's unmatched)
    """
    def blob(card):
        return getattr(card, "blob_sha", None)
    rules = (lambda x, y: key(x) == key(y),
             lambda x, y: bool(blob(x)) and blob(x) == blob(y),
             lambda x, y: getattr(x, "timestamp", None) == getattr(y, "timestamp", None) and not (blob(x) and blob(y)),
             lambda x, y: not (blob(x) and blob(y)) and same_data(x, y))
    left, right, pairs = list(range(len(a))), list(range(len(b))), []
    for rule in rules:
        for i in list(left):
            j = next((j for j in right if rule(a[i], b[j])), None)
            if j is not None:
                pairs.append((a[i], b[j]))
                left.remove(i)
                right.remove(j)
    return (pairs, [a[i] for i in left], [b[j] for j in right])


def same_data(a, b):
    """Two versions hold the same data in today's forms (card_history.data_of: no timestamp, path or blob)."""
    return data_of(a) == data_of(b)


def data(card):
    """A version's data as diffed (its timestamp aside: a re-dated version shows as a moved start)."""
    out = norm(card)
    out.pop("timestamp", None)
    return out


def history_diff(old_stats, new_stats, old_hist, new_hist, verbose):
    """Print the history section; returns whether anything changed."""
    def count(hist):
        return sum(len(v) for v in hist.values())
    print(f"\n-- history --\nold: {len(old_hist)} names {count(old_hist)} versions   "
          f"new: {len(new_hist)} names {count(new_hist)} versions")
    added, dropped, starts, tags, other = Counter(), Counter(), [], [], []
    for name in sorted(set(old_hist) | set(new_hist)):
        a, b = list(old_hist.get(name) or []), list(new_hist.get(name) or [])
        pairs, gone, new = pair(a, b)
        if new:
            added[name] = len(new)
        if gone:
            dropped[name] = len(gone)
        for w, v in pairs:
            if start(w) != start(v):
                starts.append((f"{name} (version {key(w)[0]})", start(w), start(v)))
            if data(w) != data(v):
                other.append(name)
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
          f"kept versions with other data: {len(set(other))}   cards with another blob or parser: {len(blobs)}")
    rows = [("+", n, c) for n, c in sorted(added.items())] + [("-", n, c) for n, c in sorted(dropped.items())]
    for sign, name, n in rows if verbose else rows[:40]:
        print(f"  {sign}{n} {name}")
    for name, a, b in starts if verbose else starts[:40]:
        print(f"  start {name}: {a} -> {b}")
    for name, a, b in tags:
        print(f"  tag {name}: {a} -> {b}")
    for name in sorted(set(other)):
        print(f"  data {name}")
    if verbose:
        for name, fields in blobs:
            print(f"  bookkeeping {name}: {fields}")
    return bool(added or dropped or starts or tags or other or blobs)


def main() -> int:
    args = [a for a in sys.argv[1:] if a != "--verbose"]
    history, git = None, "."
    if "--git" in args:
        i = args.index("--git")
        git, args = (args[i + 1:i + 2] or ["."])[0], args[:i] + args[i + 2:]
    if "--history" in args:
        i = args.index("--history")
        history, args = args[i + 1:i + 3], args[:i] + args[i + 3:]
    if len(args) < 2 or (history is not None and len(history) < 2):
        print(__doc__)
        return 2
    verbose = "--verbose" in sys.argv
    try:
        (old_raw, old_hist), (new_raw, new_hist) = read_cards(args[0], git), read_cards(args[1], git)
        old = {name: norm(card) for name, card in old_raw.items()}
        new = {name: norm(card) for name, card in new_raw.items()}
        if history:
            old_hist, new_hist = load_history(history[0]), load_history(history[1])
        hists = (old_hist, new_hist) if old_hist is not None and new_hist is not None else None
    except Exception as e:  # noqa: BLE001
        print(f"ERROR loading the card data: {e}")
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
        print("\n(history not compared: pass --history <old old_stats file> <new old_stats file>)")
    return 1 if (added or removed or modified or changed_history) else 0


if __name__ == "__main__":
    sys.exit(main())
