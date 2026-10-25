"""
The TTS sessions' card ratings into TTSCardMaker's site/Card Ratings.json (card_edits.CardRatings), run by hand:

    python -m commands.tts_ratings [--raw ratings_raw.json] [--repo <card repo checkout>] [--out FILE]
        [--review PREFIX] [--dry-run]

--raw is the parse of TS_Save_3, 8 and 10 (cards placed on the rating boards, matched to card names). Each board has
five category labels (3DText) at x positions; a card's value is piecewise linear in its x: the midpoint between labels
i and i+1 -> 2i+2.5 (label i about 2i+1.5, exactly when centred), past the end labels the same half-band, clamped to
1-10, one decimal.
Every entry is a group one ("by": "group", "source": "TTS session"), dated the session's evening. Skipped: TS_Save_3's
joke "Double Pain" column, the Forbidden Tier, TS_Save_10 power placements left where TS_Save_8 had them.
Re-running replaces the sessions' group entries and keeps every other one (idempotent). It also writes a review table
(PREFIX.csv, one row per card, and PREFIX.md). The bot never reads the file; the website does.
"""
from argparse import ArgumentParser
from collections import Counter, defaultdict
from csv import writer as csv_writer
from io import StringIO
from json import loads as json_loads
from math import floor
from os import walk
from os.path import dirname, expanduser, isdir, isfile, join, relpath, splitext
from statistics import mean, median
from typing import Optional

from commands.card_edits import CardAuthors, CardRatings, dump_ratings
from global_config import AUTHORS_PATH, LOCAL_DIR_LOC, RATINGS_PATH

# ========================
# Configuration
# ========================
# the saves' 3DText labels, identical in TS_Save_3, 8 and 10
SCALE_OF = {"centre": "power", "power-items": "power", "like-top": "favorability", "like-bottom": "favorability"}
CATEGORIES = {"power": ("Plz buff :(", "Below Average", "Balens", "Above Average", "Too Stronk"),
              "favorability": ("Pain", "Dislike", "Bogstandard", "Like", "Nut on you")}
LABELS = {  # 3DText x positions per board, left to right (saves' ObjectStates, Name "3DText")
    "centre":      (-23.96, -13.99, -1.16, 12.92, 23.59),   # creature power board, labels at z ~17
    "power-items": (-33.85, -20.72, -4.63, 11.31, 23.63),   # TS_Save_3 item power board, z ~-25.4
    "like-top":    (37.70, 44.98, 53.61, 62.96, 69.75),     # upper favorability board, z ~18.7
    "like-bottom": (37.65, 44.43, 53.62, 62.69, 69.67),     # lower favorability board, z ~-23.3
}
SESSIONS = {"TS_Save_3": "2023-04-28", "TS_Save_8": "2025-03-02", "TS_Save_10": "2025-04-28"}   # the session evening (owner)
SKIP_CATEGORIES = {"Double Pain"}                        # TS_Save_3's joke column left of Pain (owner: skip)
CARRY = 1.0                                              # TS_Save_10 power: a placement within 1 unit of its TS_Save_8 one is carried over
CARRIED_SAVE, CARRIED_FROM = "TS_Save_10", "TS_Save_8"
SOURCE = "TTS session"
GROUP = "group"
RAW = "~/Documents/Tools/wt/scratch/site-r3/ratings_raw.json"
CARD_FOLDERS = ("Creatures/", "Items/", "Field/", "N.M.E/", "Auxiliary/")


# ========================
# The mapping
# ========================
def score(x: float, labels: tuple) -> tuple:
    """
    A board position as a rating.

    Args:
        x: The card's (its stack's) x
        labels: The board's five label x positions, left to right

    Returns:
        (band 0-4, value 1-10 with one decimal)
    """
    mids = [(labels[i] + labels[i + 1]) / 2 for i in range(4)]
    i = sum(1 for m in mids if x > m)
    lo = mids[i - 1] if i else labels[0] - (mids[0] - labels[0])
    hi = mids[i] if i < 4 else labels[4] + (labels[4] - mids[3])
    v = 2 * i + 0.5 + 2 * (x - lo) / (hi - lo)
    return i, round(min(10.0, max(1.0, v)) + 1e-9, 1)


def _span(labels: tuple) -> tuple:
    """The x range score() maps onto 1-10 unclamped."""
    return 1.5 * labels[0] - 0.5 * labels[1], 1.5 * labels[4] - 0.5 * labels[3]


def tier_of(value: float) -> int:
    """The category a value falls in, 0-4 (edges 2.5, 4.5, 6.5, 8.5): build_site tier_of, ratings.js tierOf."""
    return max(0, min(4, floor((value - 0.5) / 2)))


def placements(raw: dict) -> tuple:
    """
    The raw parse as entries.

    Args:
        raw: ratings_raw.json ({"ratings": [...], "forbidden": [...], "skipped": {...}})

    Returns:
        (entries [{card, date, scale, value, by, source} + the review fields (category, save, board, x, match,
        match_note, stars, author, png)], report {counts, skip lists})

    Raises:
        ValueError: a value whose category isn't the one the save shows (the mapping is wrong)
    """
    rows = raw.get("ratings") or []
    key_of = lambda r: r.get("card") or r.get("archived")
    before = {key_of(r) or r.get("nick"): r for r in rows if r["save"] == CARRIED_FROM and r["scale"] == "power"}
    entries, dropped, carried, unmatched = [], [], [], []
    for r in rows:
        if r["category"] in SKIP_CATEGORIES:
            dropped.append(r)
            continue
        key = key_of(r)
        if not key:
            unmatched.append(r)
            continue
        if r["save"] == CARRIED_SAVE and r["scale"] == "power":
            a = before.get(key)
            if a and abs(a["x"] - r["x"]) < CARRY and abs(a["z"] - r["z"]) < CARRY:
                carried.append(r)
                continue
        band, value = score(r["x"], LABELS[r["board"]])
        scale = SCALE_OF[r["board"]]
        if scale != r["scale"] or CATEGORIES[scale][tier_of(value)].lower() != r["category"].lower():
            raise ValueError(f"{r['save']} {key}: {value} on {r['board']} is {CATEGORIES[scale][tier_of(value)]!r}, "
                             f"the save shows {r['category']!r}")
        entries.append({"card": key, **CardRatings.entry(SESSIONS[r["save"]], scale, value, GROUP, SOURCE),
                        "category": r["category"], "save": r["save"], "board": r["board"], "x": r["x"],
                        "match": r.get("match"), "match_note": r.get("match_note") or "",
                        "stars": r.get("card_stars"), "author": r.get("card_author"), "png": r.get("png"),
                        "archived": not r.get("card")})
    report = {"entries": len(entries), "cards": len({e["card"] for e in entries}),
              "per": Counter((e["save"], e["scale"]) for e in entries), "dropped": dropped, "carried": carried,
              "unmatched": unmatched, "forbidden": raw.get("forbidden") or [], "skipped": raw.get("skipped") or {},
              "clamped": sum(1 for e in entries if not _span(LABELS[e["board"]])[0] <= e["x"] <= _span(LABELS[e["board"]])[1])}
    return entries, report


# ========================
# The file
# ========================
def merge(text: Optional[str], entries: list) -> str:
    """
    The file with the sessions' group entries replaced by entries; website entries and other dates stay.

    Args:
        text: The file as it is, or None for a new one
        entries: placements()'s entries

    Returns:
        The new text (dump_ratings); merge(merge(t, e), e) == merge(t, e)

    Raises:
        ValueError: text isn't JSON
    """
    data = json_loads(text) if text else {"about": CardRatings.ABOUT, "cards": {}}
    if not isinstance(data, dict):
        raise ValueError("not a JSON object")
    cards = data["cards"] = data.get("cards") if isinstance(data.get("cards"), dict) else {}
    ours = set(SESSIONS.values())
    for name in list(cards):
        if isinstance(cards[name], list):
            cards[name] = [e for e in cards[name] if not (isinstance(e, dict) and e.get("source") == SOURCE
                                                          and e.get("by") == GROUP and e.get("date") in ours)]
    names = {CardAuthors.key(n): n for n in cards}
    for e in entries:
        name = names.setdefault(CardAuthors.key(e["card"]), e["card"])
        lst = cards.get(name)
        cards[name] = (lst if isinstance(lst, list) else []) + [{k: e[k] for k in CardRatings.KEYS}]
    for name in list(cards):
        if isinstance(cards[name], list):
            if cards[name]:
                cards[name] = CardRatings.order(cards[name])
            else:
                del cards[name]
    return dump_ratings(data)


# ========================
# The review table
# ========================
def kinds_of(repo: str) -> dict:
    """Card name (normalized) -> its kind from the PNG's folder: creature, item, field, N.M.E, Auxiliary/<sub>."""
    out = {}
    for root, dirs, files in walk(repo):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for f in files:
            path = relpath(join(root, f), repo).replace("\\", "/")
            if not f.lower().endswith(".png") or not path.startswith(CARD_FOLDERS):
                continue
            top = path.split("/")
            kind = {"Creatures": "creature", "Items": "item", "Field": "field", "N.M.E": "N.M.E"}.get(top[0]) \
                or "/".join(top[:2])
            out.setdefault(CardAuthors.key(splitext(f)[0]), kind)
    return out


def _cell(e: Optional[dict]) -> str:
    return f"{e['value']:.1f} {CATEGORIES[e['scale']][tier_of(e['value'])].replace(' :(', '')}" if e else ""


def _columns() -> list:
    """(scale, date) per review column, favorability first."""
    return [(s, d) for s in CardRatings.SCALES for d in sorted(SESSIONS.values())]


def review(entries: list, report: dict, authors: Optional[CardAuthors] = None, kinds: Optional[dict] = None) -> tuple:
    """
    The owner's review: one row per card, and a markdown summary.

    Args:
        entries: placements()'s entries
        report: placements()'s report
        authors: site/Card Authors.txt (credited names), or None
        kinds: kinds_of(repo), or None (the save's PNG folder then)

    Returns:
        (csv text, markdown text)
    """
    authors, kinds = authors or CardAuthors(), kinds or {}
    by_card = defaultdict(dict)
    for e in entries:
        by_card[e["card"]][(e["scale"], e["date"])] = e
    cols = [c for c in _columns() if any(c in v for v in by_card.values())]

    def facts(card):
        first = next(iter(by_card[card].values()))
        kind = kinds.get(CardAuthors.key(card)) or ("archived" if first["archived"] else (first["png"] or "").split("/")[0])
        author = authors.resolve(card, first["author"]) or ""
        notes = sorted({e["match_note"] for e in by_card[card].values() if e["match_note"]})
        matched = sorted({e["match"] for e in by_card[card].values() if e["match"]})
        return kind, first["stars"] or "", author, "/".join(matched), " | ".join(notes)

    buf = StringIO()
    out = csv_writer(buf, lineterminator="\n")
    out.writerow(["card", "kind", "stars", "author", *[f"{s} {d}" for s, d in cols], "matched by", "note"])
    for card in sorted(by_card, key=str.lower):
        kind, stars, author, matched, note = facts(card)
        out.writerow([card, kind, stars, author, *[_cell(by_card[card].get(c)) for c in cols], matched, note])

    md = ["# Card ratings from the TTS sessions: review", "",
          "Rule: each board has five category labels (0-4) at x positions. The midpoint between labels i and i+1 "
          "maps to 2i+2.5 (label i about 2i+1.5, exactly when centred); past the end labels the same half-band; "
          "clamped to 1-10, one decimal. "
          "A value's category: 1-2.4 / 2.5-4.4 / 4.5-6.4 / 6.5-8.4 / 8.5-10 (tierOf). All entries are group ones "
          "(\"by\": \"group\", \"source\": \"TTS session\").", "",
          "| Board | Scale | " + " | ".join(f"label {i}" for i in range(5)) + " |", "|---|---|---|---|---|---|---|"]
    for board, xs in LABELS.items():
        scale = SCALE_OF[board]
        md.append(f"| {board} | {scale} | " + " | ".join(f"{CATEGORIES[scale][i]} x {x} -> {score(x, xs)[1]}"
                                                           for i, x in enumerate(xs)) + " |")
    md += ["", f"**{report['entries']} entries for {report['cards']} cards** "
               f"({sum(1 for c in by_card if next(iter(by_card[c].values()))['archived'])} archived). "
               f"Carried over (skipped): {len(report['carried'])}. Clamped to 1 or 10: {report['clamped']}. "
               f"Unmatched: {len(report['unmatched'])}.", "",
           "| Session | Date | Power | Favorability |", "|---|---|---|---|"]
    for save, day in SESSIONS.items():
        md.append(f"| {save} | {day} | {report['per'].get((save, 'power'), 0)} | {report['per'].get((save, 'favorability'), 0)} |")

    for save, day in SESSIONS.items():
        for scale in CardRatings.SCALES:
            rows = [e for e in entries if e["save"] == save and e["scale"] == scale]
            if not rows:
                continue
            values = [e["value"] for e in rows]
            md += ["", f"## {save} ({day}): {scale}, {len(rows)} cards", "",
                   f"Min {min(values)}, median {median(values)}, mean {mean(values):.2f}, max {max(values)}.", "",
                   "| Category | Values | Cards |", "|---|---|---|"]
            tiers = Counter(tier_of(v) for v in values)
            for i, label in enumerate(CATEGORIES[scale]):
                md.append(f"| {label} | {('1-2.4', '2.5-4.4', '4.5-6.4', '6.5-8.4', '8.5-10')[i]} | {tiers.get(i, 0)} |")
            exact = Counter(values)
            md += ["", "Values: " + ", ".join(f"{v} x{n}" for v, n in sorted(exact.items())) + "."]
            ranked = sorted(rows, key=lambda e: (-e["value"], e["card"].lower()))
            for title, part in (("Highest", ranked[:10]), ("Lowest", sorted(rows, key=lambda e: (e["value"], e["card"].lower()))[:10])):
                cut = part[len(part) - 1]["value"]
                more = sum(1 for e in rows if e["value"] == cut) - sum(1 for e in part if e["value"] == cut)
                md.append(f"{title} 10: " + ", ".join(f"{e['card']} {e['value']}" for e in part)
                          + (f" (and {more} more at {cut})" if more > 0 else "") + ".")

    def listing(title, test):
        found = {}
        for e in entries:
            if test(e):
                found.setdefault(e["card"], e["match_note"] or e["png"] or "")
        if found:
            md.extend(["", f"## {title} ({len(found)} cards)", ""] + [f"- {c}: {n}" for c, n in sorted(found.items(), key=lambda kv: kv[0].lower())])

    listing("Archived cards (kept under their names)", lambda e: e["archived"])
    listing("Renamed cards (the save's card under today's name)", lambda e: e["match"] == "renamed")
    listing("Matched by name or file name only", lambda e: e["match"] in ("name", "png-stem"))
    moved = sorted({e["card"] for e in entries if e["match"] in ("png-moved", "png-case")}, key=str.lower)
    if moved:
        md += ["", f"## PNG moved since the session ({len(moved)} cards, followed in git)", "", ", ".join(moved) + "."]

    md += ["", "## Skipped", ""]
    for r in report["dropped"]:
        md.append(f"- {r['save']} {r['card'] or r.get('nick')} {r['scale']}: in the joke \"{r['category']}\" column (owner: skip).")
    for r in report["forbidden"]:
        md.append(f"- {r['save']} {r.get('card') or r.get('nick') or r.get('png')}: on the Forbidden Tier board (owner: skip).")
    for r in report["unmatched"]:
        md.append(f"- {r['save']} {r.get('nick') or r.get('png')} {r['scale']}: matches no card.")
    if report["carried"]:
        md.append(f"- {CARRIED_SAVE} power, {len(report['carried'])} cards left where {CARRIED_FROM} had them (a carry-over, "
                  f"not a new rating): " + ", ".join(sorted((r.get("card") or r.get("archived") or r.get("nick"))
                                                            for r in report["carried"])) + ".")
    for save, counts in report["skipped"].items():
        md.append(f"- {save}, not on a rating board (never entries): "
                  + ", ".join(f"{why} {n}" for why, n in sorted(counts.items())) + ".")
    return buf.getvalue(), "\n".join(md) + "\n"


# ========================
# Command line
# ========================
def _read(path: str) -> Optional[str]:
    if not isfile(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def _write(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(text)


def main(args: Optional[list] = None) -> int:
    """The import: writes the ratings file (unless --dry-run) and the review; commits nothing."""
    parser = ArgumentParser(prog="python -m commands.tts_ratings", description="The TTS sessions' card ratings into the card repo.")
    parser.add_argument("--raw", default=RAW, help="the saves' parse (default %(default)s)")
    parser.add_argument("--repo", default=LOCAL_DIR_LOC, help="the card repo checkout (default %(default)s)")
    parser.add_argument("--out", help=f"write this file instead of <repo>/{RATINGS_PATH}")
    parser.add_argument("--review", help="the review's path without .csv/.md (default: ratings-review next to --raw)")
    parser.add_argument("--dry-run", action="store_true", help="print the counts and write only the review")
    options = parser.parse_args(args)
    raw_path, repo = expanduser(options.raw), expanduser(options.repo)
    text = _read(raw_path)
    if text is None:
        print(f"No {raw_path}.")
        return 1
    try:
        entries, report = placements(json_loads(text))
    except ValueError as e:
        print(f"Stopped: {e}")
        return 1
    authors_text = _read(join(repo, AUTHORS_PATH))
    csv_text, md_text = review(entries, report, CardAuthors.parse(authors_text) if authors_text else None,
                               kinds_of(repo) if isdir(repo) else None)
    prefix = expanduser(options.review) if options.review else join(dirname(raw_path), "ratings-review")
    _write(prefix + ".csv", csv_text)
    _write(prefix + ".md", md_text)
    print(f"{report['entries']} entries for {report['cards']} cards; carried over {len(report['carried'])}, "
          f"clamped {report['clamped']}, unmatched {len(report['unmatched'])}.")
    print(f"Review: {prefix}.csv, {prefix}.md")
    if options.dry_run:
        return 0
    out = expanduser(options.out) if options.out else join(repo, RATINGS_PATH)
    if not isdir(dirname(out) or "."):
        print(f"No {dirname(out)}: give --repo a card repo checkout, or --out a file.")
        return 1
    try:
        new = merge(_read(out), entries)
    except ValueError as e:
        print(f"{out} isn't valid JSON ({e}): fix or delete it, then run this again.")
        return 1
    _write(out, new)
    print(f"Wrote {report['entries']} entries for {report['cards']} cards to {out}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# ========================
# Module Exports
# ========================
__all__ = [
    'CATEGORIES',
    'LABELS',
    'SESSIONS',
    'kinds_of',
    'main',
    'merge',
    'placements',
    'review',
    'score',
    'tier_of',
]
