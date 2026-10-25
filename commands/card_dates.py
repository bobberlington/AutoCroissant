"""
Creation dates: the Discord archive's created_dates.json (review_pages.py) as the card repo's site/Card Creation
Dates.json (dates only; "from": "hand" entries kept). Re-run after rulings, review the diff, commit it:

    python -m commands.card_dates created [--archive ~/Documents/Tools/discord-archive] [--repo <card repo checkout>]
        [--out FILE]
"""
from argparse import ArgumentParser
from json import loads
from typing import Optional

from commands.card_edits import CardAuthors, CardDates, dump_edits
from global_config import CREATED_PATH, LOCAL_DIR_LOC

# ========================
# Configuration
# ========================
BODY_LINES = 40                                 # a report lists at most this many changed cards
ARCHIVE = "~/Documents/Tools/discord-archive"   # the Discord archive (mined once, 2026-10; never in a repo)
SOURCES = {"discord": "discord", "git-first-upload": "git"}    # created_dates.json source -> the file's "from"
DATES_ABOUT = ("When each card was made: the earlier of its first Discord post and its first upload to this repo (UTC "
               "day); provisional: an open ruling on the Discord archive may still move it. Written by AutoCroissant "
               "(python -m commands.card_dates created); an entry with \"from\": \"hand\" is set by hand and kept. "
               "Read by the website (the created sort, card pages) and the bot (/update_stats).")


# ========================
# Creation dates (site/Card Creation Dates.json)
# ========================
def dates_file(created: dict, text: Optional[str]) -> tuple:
    """
    site/Card Creation Dates.json from the archive's created_dates.json: each dated card's day, from and provisional.
    The file's "from": "hand" entries stay as they are; nothing else of the archive leaves it.

    Args:
        created: created_dates.json (card key -> {date, source, provisional, ...})
        text: The file now ("" or None: none)

    Returns:
        (the text (dump_edits), report lines)
    """
    old = loads(text) if text else {}
    before = old.get("cards") if isinstance(old, dict) and isinstance(old.get("cards"), dict) else {}
    hand = {CardAuthors.key(k): (k, v) for k, v in before.items() if isinstance(v, dict) and v.get("from") == "hand"}
    cards, skipped = {}, []
    for name, entry in created.items():
        if CardAuthors.key(name) in hand:
            continue
        value = {"date": str(entry.get("date") or "")[:10], "from": SOURCES.get(entry.get("source")),
                 **({"provisional": True} if entry.get("provisional") else {})} if isinstance(entry, dict) else None
        if CardDates.valid(value):
            cards[name] = value
        elif not isinstance(entry, dict) or entry.get("date"):
            skipped.append(name)                # a date it can't read; an undated card is just left out
    cards.update(hand.values())
    count = lambda f: sum(1 for v in cards.values() if v["from"] == f)
    lines = [f"{len(cards)} creation dates: {count('discord')} first Discord post, {count('git')} first git upload, "
             f"{count('hand')} by hand; {sum(1 for v in cards.values() if v.get('provisional'))} provisional"]
    if skipped:
        lines.append(f"{len(skipped)} unreadable, left out: {', '.join(sorted(skipped)[:10])}")
    old_by = {CardAuthors.key(k): v for k, v in before.items()}
    new_by = {CardAuthors.key(k): k for k in cards}
    added = sorted(k for key, k in new_by.items() if key not in old_by)
    gone = sorted(k for k in before if CardAuthors.key(k) not in new_by)
    moved = []
    for key, k in sorted(new_by.items()):
        was, now = old_by.get(key), cards[k]
        if key in old_by and was != now:
            was_day = was.get("date") if isinstance(was, dict) else was
            moved.append(f"{k}: {was_day} -> {now['date']}" if was_day != now["date"] else
                         f"{k}: {now['date']} {'provisional' if now.get('provisional') else 'no longer provisional'}")
    if before:
        lines.append(f"vs the file: {len(added)} new, {len(moved)} changed, {len(gone)} gone")
        lines += [f"- {m}" for m in moved[:BODY_LINES]] + [f"- gone: {k}" for k in gone[:BODY_LINES]]
    elif added:
        lines.append(f"a new file: {len(added)} cards")
    return dump_edits({"about": DATES_ABOUT, "cards": cards}), lines


def _created(options) -> int:
    """The created command: the archive's dates into <repo>/site/Card Creation Dates.json (or --out)."""
    from os.path import dirname, expanduser, getmtime, isdir, isfile, join
    archive = expanduser(options.archive)
    source, matches = join(archive, "created_dates.json"), join(archive, "matches.json")
    if not isfile(source):
        print(f"No {source}: run review_pages.py in the Discord archive first.")
        return 1
    if isfile(matches) and getmtime(matches) > getmtime(source):
        print(f"{source} is older than matches.json (rulings since?): run `python3 -I review_pages.py` in "
              f"{archive} first, then this again.")
        return 1
    with open(source, encoding="utf-8") as fh:
        created = loads(fh.read())
    out = expanduser(options.out) if options.out else join(expanduser(options.repo), CREATED_PATH)
    if not isdir(dirname(out) or "."):
        print(f"No {dirname(out)}: give --repo a card repo checkout, or --out a file.")
        return 1
    text = None
    if isfile(out):
        with open(out, encoding="utf-8") as fh:
            text = fh.read()
    try:
        new, report = dates_file(created if isinstance(created, dict) else {}, text)
    except ValueError as e:
        print(f"{out} isn't valid JSON ({e}): fix or delete it, then run this again.")
        return 1
    if new == text:
        report.append(f"Unchanged: {out}")
    else:
        _write(out, new)
        report.append(f"Wrote {out}: review the diff, then commit it (the website and /update_stats read it).")
    print("\n".join(report))
    return 0


def _write(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(text)


def main(args: Optional[list] = None) -> int:
    """The command line: created (the creation dates); commits nothing."""
    parser = ArgumentParser(prog="python -m commands.card_dates", description="The card repo's creation dates.")
    sub = parser.add_subparsers(dest="command", required=True)
    dates = sub.add_parser("created", help=f"the Discord archive's creation dates into the card repo's {CREATED_PATH} "
                                           "(written in place, never committed)")
    dates.add_argument("--archive", default=ARCHIVE, help="the Discord archive holding created_dates.json (default %(default)s)")
    dates.add_argument("--repo", default=LOCAL_DIR_LOC, help="the card repo checkout to write (default %(default)s)")
    dates.add_argument("--out", help=f"write this file instead of <repo>/{CREATED_PATH}")
    return _created(parser.parse_args(args))


if __name__ == "__main__":
    raise SystemExit(main())


# ========================
# Module Exports
# ========================
__all__ = [
    'dates_file',
    'main',
]
