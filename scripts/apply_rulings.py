#!/usr/bin/env python3
"""
The owner's answers on the card timelines' review page (commands/timeline_review.py), applied. By hand:

    python scripts/apply_rulings.py <export.json> [--root <archive>] [--cards-repo <dir>] [--no-build] [--page <built>]
    python scripts/apply_rulings.py --list [--root <archive>]

<export.json> is the page's "Export answers" file. Every answer is checked against the page's items
(derived/review/items.json: its id, a choice the item offers, a typed value parsed: a card's id, or a name only one
card has; stats as hp/def/atk/spd, hp/def where the item's stats have only those, 'a | b' for two creatures; stars,
types, a date YYYY-MM-DD or "Before 2020"; a git look is its first version's key; a 'yes' carries the value, day or
card it confirms); one error stores nothing. Listed and skipped, never an error: an answer whose item a rebuild
settled or dropped since (an id the page showed before, derived/review/ids.json: an owner's ruling, a stop that
moved), one on an item now read only (the owner's answer), and, from an export of an older page (its "page", or
--page for a file made without one: the page's build time), one carrying a value the item no longer offers (another
value now, or none: answer it again). An answer the same as the one stored is kept as is, even when its item no
longer offers it (the page marks it not applied). An answer on a picture now dates only (derived/match/eye.json: no
card on its own, owner 11-16), whose item left the page for that, is kept as identification only (the picture stays
off every timeline; it still dates its card).
The file's "source" (the triage's files: 'triage ...'; else the owner) is kept with each answer: an owner's answer is
never replaced by the triage's, nor the triage's by its guesses (owner > triage > triage guess); otherwise a later
answer of an id wins (the same answer from a higher source takes its source); "clear" withdraws one. Good answers go into derived/review/rulings.json, then the timelines are
rebuilt (a changelog line given a card: python -m commands.discord_facts build first; python -m commands.discord_chain
build reads rulings.json), the Morgue list with the answers written to derived/review/morgue.json, and the page
rebuilt with the answers checked. Never a repo; no message text.
"""
import sys
from argparse import ArgumentParser
from collections import defaultdict
from datetime import date, datetime, timezone
from json import dumps, load
from os.path import abspath, dirname, join
from re import findall, fullmatch, sub
from typing import Optional

sys.path.insert(0, dirname(dirname(abspath(__file__))))         # AutoCroissant, for commands.*
from commands import discord_chain as dc                        # noqa: E402
from commands import discord_match as dm                        # noqa: E402
from commands import timeline_review as tr                      # noqa: E402

BEFORE_DAY = "Before 2020"                                      # discord_chain.BEFORE_DAY (owner Q6)
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
        if text.casefold() == BEFORE_DAY.casefold():
            return BEFORE_DAY
        try:
            day = datetime.strptime(text, "%Y-%m-%d").date()
        except ValueError:
            raise ValueError(f"{text!r} is not YYYY-MM-DD or {BEFORE_DAY!r}") from None
        if not FIRST_DAY <= day <= datetime.now(timezone.utc).date():
            raise ValueError(f"{text} is out of range")
        return day.isoformat()
    if kind == "stars":
        if not fullmatch(r"\d{1,2}", text) or int(text) > 20:
            raise ValueError(f"{text!r} is not a star count (0-20)")
        return int(text)
    if kind == "stats":
        return stats(text, shape)
    if kind == "types":                                         # 'a, b' (a JSON-like '["a", "b"]' too)
        return sorted({t.strip().lower() for t in sub(r"[\[\]\"']", "", text).split(",") if t.strip()})
    return text                                                 # a name or an ability


OWNER = "owner"


def rank(source: Optional[str]) -> int:
    """Whose answer wins (lower): the owner's, the triage's (owner-delegated), the triage's guesses."""
    source = str(source or OWNER)
    return 2 if source.startswith("triage guess") else 1 if source.startswith("triage") else 0


SUBS = {"card not told": "card", "look not told": "looks", "named only": "named", "joke picture": "jokes",
        "field": "fields"}                                      # a left item's part of the page (timeline_review)


def off_items(issued: dict, dates_only: dict, thumbs: dict = None) -> dict:
    """{review id: its item} for the ids the page showed about a picture now dates only (match/eye.json: no card on
    its own; owner 11-16): its card ('match:'), a picture item (keyed by the picture, or by it and the card) or a
    field of its version. An answer on one is kept as identification only (owner 11-16 23:10), never an error."""
    out = {}
    for rid, was in (issued or {}).items():
        kind, cid = was.get("kind"), was.get("card")
        for pid in dates_only:
            ids = {f"match:{pid}": None} if kind == "card not told" else \
                {dc.review_id(kind, pid): None, dc.review_id(kind, f"{pid}|{cid}"): None} if kind in dc.PICTURE_KINDS \
                else {dc.review_id("field", f"{cid}|D-{pid}|{k}"): k for k in dc.FIELDS} if kind == "field" else {}
            if rid in ids:
                thumb = (thumbs or {}).get(pid)
                out[rid] = {"id": rid, "kind": kind, "section": "matches", "sub": SUBS.get(kind), "title":
                            was.get("title") or rid, "card": cid, "picture": pid, "entry": f"D-{pid}",
                            **({"field": ids[rid]} if ids[rid] else {}), "meta": [],
                            "pics": [{"src": "../matcher/" + thumb, "cap": "the picture"}] if thumb else [],
                            "choices": [], "other": None, "read_only": False, "identification": True}
                break
    return out


def check(answers: dict, items: dict, stored: dict, cards: dict, issued: dict = None, source: str = OWNER,
          older: bool = False, off: dict = None) -> tuple:
    """({id: ruling}, [id, ...] to clear, [error, ...], [(id, why), ...] skipped) for an export's answers; older: the
    export was made on an older page (a value no radio offers now is stale, not a mistake); off: off_items (a
    dates-only picture's items that left the page: an answer kept as identification only)."""
    good, clear, errors, skipped = {}, [], [], []
    for rid, answer in sorted(answers.items()):
        it = items.get(rid) or (stored.get(rid) or {}).get("item")
        c = (answer or {}).get("choice") if isinstance(answer, dict) else None
        if c == "clear":
            if rid in stored and rank(stored[rid].get("source")) < rank(source):
                skipped.append((rid, f"kept: {stored[rid].get('source') or OWNER}'s answer"))
            elif rid in stored:
                clear.append(rid)
            continue
        if rid in stored and {k: stored[rid].get(k) for k in ("choice", "value")} == {"choice": c,
                                                                                    "value": answer.get("value")}:
            good[rid] = stored[rid]              # unchanged: kept as stored (its item may no longer offer it)
            if rank(source) < rank(stored[rid].get("source")):                  # given again by a higher source
                good[rid] = {k: v for k, v in stored[rid].items() if k not in ("source", "at")}
                good[rid].update({"source": source} if source != OWNER else {})
            continue
        if rid in stored and rank(stored[rid].get("source")) < rank(source):
            skipped.append((rid, f"kept: {stored[rid].get('source') or OWNER}'s answer"))
            continue
        if not it and rid in (off or {}) and c:   # a dates-only picture's: what it shows, identification only
            good[rid] = {"kind": off[rid]["kind"], "choice": c, **({"value": answer["value"]} if "value" in answer
                                                                  else {}),
                         "item": off[rid], **({"source": source} if source != OWNER else {})}
            continue
        if not it:
            was = (issued or {}).get(rid)
            if was:                              # shown before, gone at a rebuild: settled or dropped since
                skipped.append((rid, f"left the page (last shown {was.get('last')}: {was.get('title')})"))
            else:
                errors.append(f"{rid}: no such item on the page")
            continue
        if it.get("read_only"):
            skipped.append((rid, "read only now (the owner's answer)"))
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
        elif older and value is not None:        # the page offered it then: another value now, or none
            again = any(x["c"] == c for x in it.get("choices") or [])
            skipped.append((rid, f"{c!r} now offers {'another value' if again else 'nothing'} (answer it again)"))
            continue
        else:
            errors.append(f"{rid}: {c!r} with {value!r} is not an answer this item takes")
            continue
        good[rid] = {"kind": it["kind"], **got, "item": it, **({"source": source} if source != OWNER else {})}
    return good, clear, errors, skipped


def apply(export: dict, root: str, cards_repo: str = None, build: bool = True, log=print, made_on: str = None) -> int:
    root = dm.check_root(root)
    if export.get("format") != tr.FORMAT:
        log(f"not a review page export (format {export.get('format')!r}, want {tr.FORMAT!r})")
        return 1
    page = dm.read_json(join(root, *tr.OUT, "items.json"), {}) or {}
    items = {i["id"]: i for i in page.get("items") or []}
    path = join(root, *tr.RULINGS)
    store = dm.read_json(path, {}) or {}
    stored = store.get("rulings") or {}
    issued = (dm.read_json(join(root, *tr.ISSUED), {}) or {}).get("ids") or {}
    source = str(export.get("source") or OWNER)
    made_on = export.get("page") or made_on                     # the page the answers were given on
    older = bool(made_on) and made_on != page.get("built")
    only = (dm.read_json(join(root, *dc.MATCH_EYE), {}) or {}).get("dates_only") or {}
    thumbs = {r["id"]: r.get("thumb") for r in dm.read_jsonl(join(root, *tr.VERDICTS)) if r["id"] in only} \
        if only else {}
    off = off_items({rid: was for rid, was in issued.items() if rid not in items}, only, thumbs)
    good, clear, errors, skipped = check(export.get("rulings") or {}, items, stored, page.get("cards") or {}, issued,
                                         source, older, off)
    if errors:
        log(f"{len(errors)} answer(s) refused, nothing stored (fix them and export again):")
        for e in errors:
            log("  " + e)
        return 1
    if skipped:
        log(f"{len(skipped)} answer(s) skipped (nothing to apply them to now):")
        for rid, why in skipped:
            log(f"  {rid}: {why}")
    now = dm.now_iso()
    same = [rid for rid, r in good.items() if rid in stored and
            {k: stored[rid].get(k) for k in ("choice", "value", "source")} ==
            {k: r.get(k) for k in ("choice", "value", "source")}]
    lines = any(r.get("kind") == "ambiguous card" for rid, r in good.items() if rid not in same) or \
        any((stored.get(rid) or {}).get("kind") == "ambiguous card" for rid in clear)    # facts to rebuild
    for rid, r in good.items():
        if rid not in same:
            stored[rid] = {**r, "at": now}
    for rid in clear:
        stored.pop(rid, None)
    dm.write_json(path, {"about": "The owner's answers on the card timelines' review page (scripts/apply_rulings.py): "
                                  "id -> kind, choice, value, when, the item as the page showed it, and source "
                                  "when not the owner's (the triage's owner-delegated answers). "
                                  "commands.discord_chain build reads them.",
                         "updated": now, "rulings": dict(sorted(stored.items()))})
    log(f"answers ({source}): {len(good) - len(same)} new or changed, {len(same)} unchanged, {len(clear)} cleared, "
        f"{len(skipped)} skipped; {len(stored)} kept in {path}")
    if build:
        if lines:
            from commands import discord_facts            # a changelog line given a card: its facts first
            discord_facts.main(["build", "--root", root] + (["--cards-repo", cards_repo] if cards_repo else []))
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
    parser.add_argument("--page", default=None, help="the page's build time a file without 'page' was made on")
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
    return apply(export, args.root, args.cards_repo, not args.no_build, made_on=args.page)


if __name__ == "__main__":
    raise SystemExit(main())
