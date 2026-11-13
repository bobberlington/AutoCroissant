#!/usr/bin/env python3
"""
The owner's answers on the card timelines' review page (commands/timeline_review.py), applied. By hand:

    python scripts/apply_rulings.py <export.json> [--root <archive>] [--cards-repo <dir>] [--no-build]
    python scripts/apply_rulings.py --list [--root <archive>]

<export.json> is the page's "Export answers" file. Every answer is checked against the page's items
(derived/review/items.json: its id, that it isn't read only, a choice the item offers, a typed value parsed: a card's
id, or a name only one card has; stats as hp/def/atk/spd, hp/def where the item's stats have only those, 'a | b' for
two creatures; stars, types, a date YYYY-MM-DD or "Before 2020"; a git look is its first version's key; a 'yes'
carries the value, day or card it confirms); one error stores nothing. An answer the same as the one stored is kept
as is, even when its item no longer offers it (the page marks it not applied).
Good answers go into derived/review/rulings.json (a later answer of an id wins; "clear" withdraws one), then the
timelines are rebuilt (python -m commands.discord_chain build reads rulings.json), the Morgue list with the answers
written to derived/review/morgue.json, and the page rebuilt with the answers checked. Never a repo; no message text.
"""
import sys
from argparse import ArgumentParser
from collections import defaultdict
from datetime import date, datetime, timezone
from json import dumps, load
from os.path import abspath, dirname, join
from re import findall, fullmatch

sys.path.insert(0, dirname(dirname(abspath(__file__))))         # AutoCroissant, for commands.*
from commands import discord_match as dm                        # noqa: E402
from commands import timeline_review as tr                      # noqa: E402

PAPER_DAY = "Before 2020"                                       # discord_chain.PAPER_DAY (owner Q6)
FIRST_DAY = date(2018, 1, 1)                                    # no card is older than the server (2019)
STATS = ("hp", "def", "atk", "spd")


def stats_text(value) -> str:
    """A stats value (a stored answer coming back) as typed."""
    if isinstance(value, dict) and value.get("creatures"):
        return " | ".join(stats_text(c) for c in value["creatures"])
    return "/".join(str(value[k]) for k in STATS if k in value) if isinstance(value, dict) else str(value)


def card(text: str, cards: dict) -> str:
    """A card id from its exact id, else a name or slug only one card has (two cards share some names)."""
    if text in cards:
        return text
    forms = (defaultdict(set), defaultdict(set))
    for cid, name in cards.items():
        forms[0][str(name).casefold()].add(cid)
        forms[1][dm.slugify(str(name))].add(cid)
    for found in (forms[0].get(text.casefold()), forms[1].get(dm.slugify(text))):
        if found and len(found) > 1:
            raise ValueError(f"{text!r} names {len(found)} cards ({', '.join(sorted(found))}): type the card's id")
        if found:
            return next(iter(found))
    if dm.slugify(text) in cards:
        return dm.slugify(text)
    raise ValueError(f"no live card named {text!r}")


def stats(text: str, shape: str = None) -> dict:
    """hp/def/atk/spd; hp/def when the item's stats have only those (shape 'hp/def'); 'a | b' for two creatures."""
    parts = text.split("|")
    if len(parts) > 1:
        return {"creatures": [stats(part.strip(), "creature") for part in parts]}
    numbers = findall(r"\d+", text)
    if any(int(n) > 99 for n in numbers):
        raise ValueError(f"{text!r} has a stat over 99")
    if len(numbers) == 4:
        return dict(zip(STATS, map(int, numbers)))
    if len(numbers) == 2 and shape == "hp/def":
        return dict(zip(STATS[:2], map(int, numbers)))
    want = "hp/def (or four stats)" if shape == "hp/def" else "four stats (hp/def/atk/spd)" + \
        ("" if shape == "creature" else ", or 'a | b' for two creatures")
    raise ValueError(f"{text!r} is not {want}")


def parse(kind: str, text, cards: dict, shape: str = None):
    """A typed answer's value, or raise ValueError saying why. shape: the item's stats form (items.json other.shape)."""
    if kind == "stats" and isinstance(text, dict):
        shape = shape or ("hp/def" if set(text) <= {"hp", "def"} else None)
        text = stats_text(text)
    text = str(text if text is not None else "").strip()
    if not text:
        raise ValueError("nothing typed")
    if kind == "card":
        return card(text, cards)
    if kind == "date":
        if text.casefold() == PAPER_DAY.casefold():
            return PAPER_DAY
        try:
            day = datetime.strptime(text, "%Y-%m-%d").date()
        except ValueError:
            raise ValueError(f"{text!r} is not YYYY-MM-DD or {PAPER_DAY!r}") from None
        if not FIRST_DAY <= day <= datetime.now(timezone.utc).date():
            raise ValueError(f"{text} is out of range")
        return day.isoformat()
    if kind == "stars":
        if not fullmatch(r"\d{1,2}", text) or int(text) > 20:
            raise ValueError(f"{text!r} is not a star count (0-20)")
        return int(text)
    if kind == "stats":
        return stats(text, shape)
    if kind == "types":
        return sorted({t.strip().lower() for t in text.split(",") if t.strip()})
    return text                                                 # a name or an ability


def check(answers: dict, items: dict, stored: dict, cards: dict) -> tuple:
    """({id: ruling}, [id, ...] to clear, [error, ...]) for an export's answers."""
    good, clear, errors = {}, [], []
    for rid, answer in sorted(answers.items()):
        it = items.get(rid) or (stored.get(rid) or {}).get("item")
        c = (answer or {}).get("choice") if isinstance(answer, dict) else None
        if c == "clear":
            if rid in stored:
                clear.append(rid)
            continue
        if rid in stored and {k: stored[rid].get(k) for k in ("choice", "value")} == {"choice": c,
                                                                                    "value": answer.get("value")}:
            good[rid] = stored[rid]              # unchanged: kept as stored (its item may no longer offer it)
            continue
        if not it:
            errors.append(f"{rid}: no such item on the page")
            continue
        if it.get("read_only"):
            errors.append(f"{rid}: read only")
            continue
        value = answer.get("value")
        listed = next((x for x in it.get("choices") or [] if x["c"] == c and
                       dumps(x.get("value"), sort_keys=True) == dumps(value, sort_keys=True)), None)
        other = it.get("other") or {}
        if listed:
            got = {"choice": c, **({"value": listed["value"]} if "value" in listed else {})}
        elif other and c == other.get("c"):
            try:
                got = {"choice": c, "value": parse(other["kind"], value, cards, other.get("shape"))}
            except ValueError as e:
                errors.append(f"{rid}: {e}")
                continue
        else:
            errors.append(f"{rid}: {c!r} with {value!r} is not an answer this item takes")
            continue
        good[rid] = {"kind": it["kind"], **got, "item": it}
    return good, clear, errors


def apply(export: dict, root: str, cards_repo: str = None, build: bool = True, log=print) -> int:
    root = dm.check_root(root)
    if export.get("format") != tr.FORMAT:
        log(f"not a review page export (format {export.get('format')!r}, want {tr.FORMAT!r})")
        return 1
    page = dm.read_json(join(root, *tr.OUT, "items.json"), {}) or {}
    items = {i["id"]: i for i in page.get("items") or []}
    path = join(root, *tr.RULINGS)
    store = dm.read_json(path, {}) or {}
    stored = store.get("rulings") or {}
    good, clear, errors = check(export.get("rulings") or {}, items, stored, page.get("cards") or {})
    if errors:
        log(f"{len(errors)} answer(s) refused, nothing stored (fix them and export again):")
        for e in errors:
            log("  " + e)
        return 1
    now = dm.now_iso()
    same = [rid for rid, r in good.items() if rid in stored and
            {k: stored[rid].get(k) for k in ("choice", "value")} == {k: r.get(k) for k in ("choice", "value")}]
    for rid, r in good.items():
        if rid not in same:
            stored[rid] = {**r, "at": now}
    for rid in clear:
        stored.pop(rid, None)
    dm.write_json(path, {"about": "The owner's answers on the card timelines' review page (scripts/apply_rulings.py): "
                                  "id -> kind, choice, value, when, and the item as the page showed it. "
                                  "commands.discord_chain build reads them.",
                         "updated": now, "rulings": dict(sorted(stored.items()))})
    log(f"answers: {len(good) - len(same)} new or changed, {len(same)} unchanged, {len(clear)} cleared; "
        f"{len(stored)} kept in {path}")
    if build:
        from commands import discord_chain
        from global_config import LOCAL_DIR_LOC
        discord_chain.build(root, cards_repo or LOCAL_DIR_LOC, log=lambda text: None)
        morgue = tr.morgue_list(root)
        if morgue is not None:
            log(f"Morgue list: {len(morgue['entries'])} entries -> {join(root, *tr.OUT, 'morgue.json')}")
        tr.build(root, log=log)
    return 0


def main(argv=None) -> int:
    parser = ArgumentParser(prog="python scripts/apply_rulings.py", description=__doc__.split("\n\n")[0])
    parser.add_argument("export", nargs="?", help="the page's exported answers (JSON)")
    parser.add_argument("--root", default=dm.DEFAULT_ROOT)
    parser.add_argument("--cards-repo", default=None)
    parser.add_argument("--no-build", action="store_true", help="store the answers only")
    parser.add_argument("--list", action="store_true", help="the answers stored so far")
    args = parser.parse_args(argv)
    if args.list:
        stored = (dm.read_json(join(dm.check_root(args.root), *tr.RULINGS), {}) or {}).get("rulings") or {}
        for rid, r in stored.items():
            print(f"{rid}  {r.get('kind')}: {r.get('choice')}" + (f" {r['value']!r}" if "value" in r else ""))
        return 0
    if not args.export:
        parser.error("give the exported answers file (or --list)")
    with open(args.export, encoding="utf-8") as f:
        export = load(f)
    return apply(export, args.root, args.cards_repo, not args.no_build)


if __name__ == "__main__":
    raise SystemExit(main())
