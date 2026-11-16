"""
Card timelines' review page (docs/Future/CARD_TIMELINE.md step 5; STAGE1 S8): one local HTML page of everything the
timelines keep out for the owner, with stable ids and radio answers that export a JSON for scripts/apply_rulings.py.

    python -m commands.timeline_review build [--root <dir>]

Reads the parts' outputs under the archive's derived/ (chain, match, matcher, oldrows, facts, morgue; never a repo; the
preview's eye.json: its leave_out pictures, no version of the card, are not asked about) and writes derived/review/:
    index.html   the page (open it in a browser: its pictures are the archive's own thumbnails, by relative path)
    items.json   every item with its id, kind, section and the answers it takes (what apply_rulings checks against)
Sections: the owner's answers (Q1-Q6 and the later ones, read only), the unsure old maps, unsure matches and
versions (a picture's card, its look, named-only pictures to confirm in bulk, joke pictures, older templates, unsure
fields, changelog lines), PSD vs picture (the old repo's rows), Created dates (earlier hints only when
earlier than the Created date in use), the Morgue list (candidates, ideas never ported, the rest) and splits. A name
two cards share shows its id; a git look is answered by its first version's key (look numbers shift when a version is
added), and a confirming answer carries what it confirms (a field's value, a Created day, a picture's card): a rerun
proposing something else leaves it not applied. Answers already applied (derived/review/rulings.json) show checked;
one whose item left the page says whether the rerun took it in (derived/chain/ruled.json) or the item changed, and one
no radio takes any more is not applied.
No message text or posters: pictures, dates, channels, read fields, and a Discord link for a changelog line (the page
stays local).
"""
from argparse import ArgumentParser
from collections import Counter, defaultdict
from html import escape
from json import dumps
from os import listdir
from os.path import exists, join
from re import search
from typing import Callable, Optional

from commands import discord_chain as dc
from commands import discord_match as dm

# ========================
# Configuration
# ========================
OUT = ("derived", "review")
CHAIN = ("derived", "chain")
VERDICTS = ("derived", "match", "verdicts.jsonl")
OLD_REVIEW = ("derived", "oldrows", "review.jsonl")
FACTS, FACT_REVIEW = ("derived", "facts", "facts.jsonl"), ("derived", "facts", "review.jsonl")
FIRST_REPO = ("derived", "facts", "firstrepo.jsonl")
MORGUE, IDEAS = ("derived", "morgue", "candidates.json"), ("derived", "morgue", "ideas.json")
RULINGS = ("derived", "review", "rulings.json")
ISSUED = dc.ISSUED                                    # every id the page showed, with its card (ids keep their meaning)
OWNER = ("derived", "review", "owner.json")           # the owner's rulings outside the page (answers shown read only)
EYE = ("derived", "preview", "eye.json")              # the preview's eye: leave_out = pictures that are no version
USED = CHAIN + ("ruled.json",)                        # the answers the chain's last build acted on
RIVAL = r"its text reads (\S+)'s better"               # discord_verdicts.name_check's reason: a card read better
GUILD = "545138546021892096"                          # Impossibility Simulator (a changelog line's Discord link)
FORMAT = "card-timelines-rulings/1"                   # the export's format (apply_rulings checks it)
STORE = "timelines-review"                            # the browser's saved answers (localStorage key prefix)
SECTIONS = (                                          # id, title, what it asks
    ("answers", "The owner's answers (read only)", "Your answers to stage 1's six questions and your later rulings, "
                                                   "as built."),
    ("maps", "Old repo: unsure maps", "Old-repo files whose card isn't sure. Their versions join the card you pick. "
                                      "(The seventh, Twin Princes, is under Splits: your Q3 answer.)"),
    ("matches", "Unsure matches and versions", "Discord pictures kept out of the timelines until you answer."),
    ("psd", "PSD vs picture (old repo)", "Old-repo versions whose PSD and picture (or reads) disagree."),
    ("created", "Created dates", "Where our earliest appearance and the website's date disagree, and earlier hints."),
    ("morgue", "The Morgue list", "Lost cards that were played, joke cards, and ideas never ported."),
    ("splits", "Splits", "Old cards that became two (your Q3 answer), with each half's stats to confirm."))
SUBS = {                                              # section -> its parts in order: id, title, bulk answer
    "matches": (("card", "Which card is it?", None), ("looks", "Which look is it?", None),
                ("named", "Named only: is this the card? (confirm in bulk)", "yes"),
                ("jokes", "Joke pictures", None), ("older", "An older template posted later", None),
                ("fields", "Fields read on Discord versions", None),
                ("facts", "Changelog lines", None)),
    "created": (("earlier", "Earlier than the website's date (proposed: the preview uses ours)", None),
                ("new", "No date on the website yet", None),
                ("later", "Later than the website's date (the website's date kept)", None),
                ("hints", "Earlier hints (each earlier than the Created date in use)", None)),
    "morgue": (("candidates", "Morgue candidates", "yes"), ("ideas", "Ideas never ported (not published)", None),
               ("not", "Not candidates (as decided by eye)", None))}
MORGUE_KINDS = {"candidates": "morgue candidate", "ideas": "morgue idea", "not": "morgue not candidate"}
FIELD_OTHER = {"name": "name", "stars": "stars", "types": "types", "stats": "stats", "ability": "ability"}


# ========================
# Inputs
# ========================
class Inputs:
    def __init__(self, root: str):
        at = lambda parts: join(root, *parts)
        self.root = root
        pairs = dm.read_json(dm.Paths(root).pairs, {}) or {}
        self.cards = pairs.get("cards") or {}
        self.by_key = {c["key"]: cid for cid, c in self.cards.items()}
        self.looks, self.current = defaultdict(dict), {}
        for v in sorted(pairs.get("versions") or [], key=lambda v: v["ts"]):
            self.looks[v["card"]].setdefault(v.get("look"), v)
            if v.get("current"):
                self.current[v["card"]] = v
        self.verdicts = {r["id"]: r for r in dm.read_jsonl(at(VERDICTS))}
        self.chain = dm.read_jsonl(at(CHAIN + ("review.jsonl",)))
        self.created = (dm.read_json(at(CHAIN + ("created.json",)), {}) or {}).get("cards") or {}
        self.timelines, self.entries = {}, {}
        folder = at(CHAIN + ("timelines",))
        for name in sorted(listdir(folder)) if exists(folder) else []:
            t = dm.read_json(join(folder, name), {}) or {}
            self.timelines[t.get("card")] = t
            for e in t.get("entries") or []:
                self.entries[(t.get("card"), e["id"])] = e
        self.old_review = dm.read_jsonl(at(OLD_REVIEW))
        self.facts = {f["id"]: f for f in dm.read_jsonl(at(FACTS))}
        self.fact_review = dm.read_jsonl(at(FACT_REVIEW))
        self.first_repo = dm.read_jsonl(at(FIRST_REPO))
        self.morgue = dm.read_json(at(MORGUE), {}) or {}
        self.ideas = (dm.read_json(at(IDEAS), {}) or {}).get("ideas") or []
        self.rulings = (dm.read_json(at(RULINGS), {}) or {}).get("rulings") or {}
        self.owner = dm.read_json(at(OWNER), {}) or {}
        self.answers = {a["id"]: a for a in self.owner.get("answers") or []}
        self.used = set((dm.read_json(at(USED), {}) or {}).get("used") or [])
        self.left_out = {tuple(k.split("|")) for k in (dm.read_json(at(EYE), {}) or {}).get("leave_out") or {}}
        names = Counter(str(c.get("name") or cid).casefold() for cid, c in self.cards.items())
        self.shared = {n for n, k in names.items() if k > 1}          # a name two cards share: shown with the id

    def name(self, cid: Optional[str]) -> str:
        return (self.cards.get(cid) or {}).get("name") or cid or "?"

    def label(self, cid: Optional[str]) -> str:
        """A card's name, with its id when another card has the same name."""
        name = self.name(cid)
        return f"{name} ({cid})" if name.casefold() in self.shared else name

    def created_before(self, cid: Optional[str]) -> Optional[str]:
        """The sort of the Created date in use for a card (the owner's pick when answered, before any hint): a hint
        counts only when earlier. None for 'hand' dates or no Created."""
        made = self.created.get((self.cards.get(cid) or {}).get("key")) or {}
        if not made or made.get("status") == "hand":
            return None
        base = (made.get("owner") or {}).get("base")
        return created_sort(base) if base else made.get("sort") or made.get("date")

    def cid(self, card: Optional[str]) -> Optional[str]:
        """A card id from an id or a card key (the old rows' and facts' names)."""
        return card if card in self.cards else self.by_key.get(card)


# ========================
# Pictures and values
# ========================
def pic(src: Optional[str], cap: str = "", box=None, ar: Optional[float] = None) -> Optional[dict]:
    return {"src": src, "cap": cap, **({"box": box} if box else {}), **({"ar": ar} if ar else {})} if src else None


def row_pic(row: Optional[dict], cap: str = "") -> Optional[dict]:
    """A Discord picture's thumb (the matcher's), a sheet's cell cropped by its box."""
    if not row or not row.get("thumb"):
        return None
    ar = round(row["w"] / row["h"], 4) if row.get("w") and row.get("h") else None
    return pic("../matcher/" + row["thumb"], cap, row.get("box") if row.get("cell") else None, ar)


def card_pic(inputs: Inputs, cid: Optional[str], cap: Optional[str] = None) -> Optional[dict]:
    v = inputs.current.get(cid)
    return pic("../matcher/" + v["thumb"], cap if cap is not None else f"{inputs.name(cid)} today") if v else None


def entry_pic(inputs: Inputs, cid: str, eid: str, cap: str = "") -> Optional[dict]:
    e = inputs.entries.get((cid, eid)) or {}
    if e.get("pictures"):
        p = e["pictures"][0]
        return row_pic(inputs.verdicts.get(p["id"]) or {**p, "w": None, "h": None}, cap)
    return pic("../" + e["thumb"], cap) if e.get("thumb") else None


def show(value) -> str:
    """A field value as the page prints it."""
    if isinstance(value, dict) and value.get("creatures"):
        return " | ".join(show(c) for c in value["creatures"])
    if isinstance(value, dict):
        return "/".join("?" if value.get(k) is None else str(value[k]) for k in ("hp", "def", "atk", "spd"))
    if isinstance(value, list):
        return ", ".join(str(v) for v in value) or "(none)"
    return "(empty)" if value in (None, "") else str(value)


def when(stamp) -> str:
    return str(stamp or "")[:10] or "no date"


def created_sort(date: Optional[str]) -> str:
    return dc.created_sort(date)


def stats_shape(*values) -> Optional[str]:
    """'creatures' or 'hp/def' when the first stats value given has that shape (a typed answer may then take it)."""
    for v in values:
        if isinstance(v, dict) and v.get("creatures"):
            return "creatures"
        if isinstance(v, dict) and v:
            return "hp/def" if {k for k, x in v.items() if x is not None} <= {"hp", "def"} else None
    return None


def card_stats(inputs, cid: Optional[str]):
    """The card's latest stats value in its timeline (its shape, for an item that read none)."""
    for e in reversed((inputs.timelines.get(cid) or {}).get("entries") or []):
        value = ((e.get("fields") or {}).get("stats") or {}).get("value")
        if isinstance(value, dict):
            return value
    return None


def value_other(field: str, shape: Optional[str]) -> dict:
    """A typed value's answer: stats say the form they take (hp/def only, or two creatures)."""
    form = {"hp/def": " (hp/def)", "creatures": " (hp/def/atk/spd | hp/def/atk/spd)"}.get(shape, " (hp/def/atk/spd)") \
        if field == "stats" else ""
    return {"c": "value", "kind": FIELD_OTHER.get(field, "name"), "label": f"the {field} it shows{form}",
            **({"shape": shape} if field == "stats" and shape else {})}


def item(kind: str, iid: str, section: str, sub: Optional[str], title: str, **extra) -> dict:
    return {"id": iid, "kind": kind, "section": section, "sub": sub, "title": title, "meta": [], "pics": [],
            "choices": [], "other": None, "read_only": False, **extra}


def choice(code: str, label: str, value=None, picture: Optional[dict] = None) -> dict:
    return {"c": code, "label": label, **({"value": value} if value is not None else {}),
            **({"pic": picture} if picture else {})}


# ========================
# Items
# ========================
def owner_answers(inputs: Inputs) -> list:
    out = []
    for n, r in enumerate(sorted((r for r in inputs.chain if r["kind"] == "owner's answer"),
                                 key=lambda r: (r.get("order", 0), r["question"]))):
        it = item(r["kind"], r["id"], "answers", None, f"{r['question']} = {r['answer']}", read_only=True, order=n)
        it["meta"] = [r["ruling"]] + ([f"answered {r['when']}"] if r.get("when") else [])
        out.append(it)
    return out


def old_maps(inputs: Inputs) -> list:
    out = []
    for r in (r for r in inputs.old_review if r["kind"] == "unsure map"):
        cid = inputs.cid(r.get("card"))
        it = item(r["kind"], r["id"], "splits" if r.get("split") else "maps", None,
                  f"Old file '{r.get('old_id')}' -> {r.get('card')} ({r.get('status')})", card=cid,
                  old_id=r.get("old_id"))
        it["meta"] = [f"matched by: {', '.join(r.get('how') or [])}", *(r.get("why") or [])]
        rows = r.get("rows") or []
        it["meta"].append(f"{len(rows)} version{'s' if len(rows) != 1 else ''}: " + "; ".join(
            f"{when(x.get('date'))} {x.get('path')}" + (f" (its PSD: {x['psd_file']})" if x.get("replaces") else
                                                     " (a PSD's own look)" if x.get("psd_only") else "")
            for x in rows[:4]))
        it["pics"] = [pic("../" + x["thumb"], f"old repo {when(x.get('date'))}") for x in rows[:4] if x.get("thumb")]
        it["pics"].append(card_pic(inputs, cid))
        if r.get("split"):
            it["read_only"] = True
            it["meta"].append("Your Q3 answer: both halves start with these versions (see below).")
        elif not cid:                             # an archived card: no live timeline to take them
            it["meta"].append(f"{r.get('card')} is not a live card ({r.get('status')}): it has no timeline, so a "
                              "'yes' would change nothing. Give the live card they belong to, or no.")
            it["choices"] = [choice("no", "no: not a live card's version (leave them out)")]
            it["other"] = {"c": "card", "kind": "card", "label": "a live card"}
        else:
            it["choices"] = [choice("yes", f"yes: {r.get('card')}'s versions", r.get("card")),
                             choice("no", "no: not a live card's version (leave them out)")]
            it["other"] = {"c": "card", "kind": "card", "label": "another card"}
        out.append(it)
    return out


def match_items(inputs: Inputs) -> list:
    """Pictures whose card isn't told (only those that could add history: dated before a maybe card's first git
    record, not taken by the Morgue's eye)."""
    taken = {t["picture"] for t in inputs.morgue.get("taken") or []}
    out = []
    for row in sorted(inputs.verdicts.values(), key=lambda r: ((r.get("first") or {}).get("date") or "", r["id"])):
        if row["verdict"] != "unsure" or not row.get("first") or row["id"] in taken:
            continue
        maybe = [c for c in row.get("maybe") or [] if c in inputs.cards]
        date = dm.epoch(row["first"]["date"])
        if not maybe or not any(not inputs.cards[c].get("fs") or date < inputs.cards[c]["fs"] for c in maybe):
            continue
        rival = search(RIVAL, row.get("reason") or "")
        rival = rival.group(1) if rival and rival.group(1) in inputs.cards and rival.group(1) not in maybe else None
        it = item("card not told", "match:" + row["id"], "matches", "card",
                  f"{when(row['first']['date'])} #{row['first'].get('channel')}", picture=row["id"],
                  date=row["first"]["date"])
        it["meta"] = [row.get("reason") or "", "Named here, the picture's look is asked next (after the rerun)."]
        it["pics"] = [row_pic(row, "the picture")]
        it["choices"] = [choice("card", inputs.label(c), c, card_pic(inputs, c)) for c in maybe[:6]]
        if rival:                                 # the card the reason says its text reads better
            it["choices"].append(choice("card", f"{inputs.label(rival)} (its text reads better)", rival,
                                        card_pic(inputs, rival)))
        it["choices"].append(choice("none", "none of these"))
        it["other"] = {"c": "card", "kind": "card", "label": "another card"}
        out.append(it)
    return out


def chain_items(inputs: Inputs) -> list:
    out = []
    for r in inputs.chain:
        kind, cid = r["kind"], r.get("card")
        if (cid, r.get("entry")) in inputs.left_out:          # no version of the card (the preview's eye): not asked
            continue
        row = inputs.verdicts.get(r.get("picture")) or {}
        head = f"{inputs.label(cid)}: " if cid else ""
        date = f"{when(r.get('date'))} #{r.get('channel')}" if r.get("channel") else when(r.get("date"))
        common = dict(card=cid, picture=r.get("picture"), date=r.get("date"))
        if kind == "look not told":
            it = item(kind, r["id"], "matches", "looks", head + date, **common)
            it["meta"] = [r.get("why") or ""]
            it["pics"] = [row_pic(row, "the picture")]
            looks = inputs.looks.get(cid) or {}
            for look, v in sorted(looks.items(), key=lambda kv: kv[1]["ts"]):     # its first version's key: stable
                near = " (nearest)" if r.get("near") == look else ""
                it["choices"].append(choice("look", f"look {look}{near}: in git {when(dm.iso(v['ts']))}"
                                            f"{' (old repo)' if v.get('old') else ''}", v["key"],
                                            pic("../matcher/" + v["thumb"], f"look {look}")))
            it["choices"] += [choice("draft", "a look not in git (a draft)", cid), choice("no", "not this card", cid)]
        elif kind == "named only":
            it = item(kind, r["id"], "matches", "named", head + date, **common)
            it["meta"] = [r.get("why") or ""]
            it["pics"] = [row_pic(row, "the picture"), card_pic(inputs, cid)]
            it["choices"] = [choice("yes", f"yes, {inputs.name(cid)}", cid), choice("no", "no", cid)]
        elif kind == "joke picture":
            it = item(kind, r["id"], "matches", "jokes", head + date, **common)
            it["meta"] = ["Kept out as a joke picture (a meme channel's, or a joke look; your Q1 answer)",
                          f"why: {r['why']}" if r.get("why") not in (None, "version") else ""]
            it["pics"] = [row_pic(row, "the picture"), card_pic(inputs, cid)]
            it["choices"] = [choice("live", f"{inputs.name(cid)}'s picture: show it in its timeline", cid),
                             choice("joke", "a joke: leave it out")]
        elif kind == "older template later":
            it = item(kind, r["id"], "matches", "older", head + when(r.get("date")), card=cid, entry=r.get("entry"),
                      date=r.get("date"))
            it["meta"] = [f"era {r.get('era')}", r.get("why") or ""]
            it["pics"] = [row_pic(inputs.verdicts.get(p), "the picture") for p in r.get("pictures") or []][:3]
            it["pics"].append(card_pic(inputs, cid))
            it["choices"] = [choice("out", "an old look posted again: leave it out"),
                             choice("show", "a real version: show it")]
        elif kind == "field":
            it = field_item(inputs, r)
        elif kind == "fact vs picture":
            fact = inputs.facts.get(r.get("fact")) or {}
            it = item(kind, r["id"], "matches", "facts", head + f"{when(r.get('date'))} #{fact.get('channel')}",
                      card=cid, fact=r.get("fact"), entry=r.get("entry"), date=r.get("date"))
            to = f"{r['fact_by']:+d} (so {r.get('fact_to')})" if isinstance(r.get("fact_by"), int) else \
                f"-> {r.get('fact_to')}"
            it["meta"] = [f"a changelog line: {r.get('field', '').upper()} {to}; the next picture reads "
                          f"{r.get('picture_reads')}"]
            it["link"] = jump(fact)
            it["pics"] = [entry_pic(inputs, cid, r.get("entry") or "", "the next picture")]
            it["choices"] = [choice("fact", "the changelog is right (keep the line)"),
                             choice("picture", "the picture is right (drop the line)")]
        elif kind.startswith("created ") or kind in ("changelog earlier", "first repo earlier"):
            it = created_item(inputs, r)
        elif kind == "split":
            it = item(kind, r["id"], "splits", None, f"{r.get('name')} -> " + " + ".join(
                inputs.name(h) for h in r.get("halves") or []), read_only=True)
            it["meta"] = [r.get("why") or "", "versions 'as' the old card: " + ", ".join(
                f"{inputs.name(h)} {n}" for h, n in (r.get("entries") or {}).items()), *split_evidence(r.get("picked"))]
            for h in r.get("halves") or []:
                t = inputs.timelines.get(h) or {}
                it["pics"] += [entry_pic(inputs, h, e["id"], f"{inputs.name(h)} as {e.get('as')} {when(e.get('date'))}")
                               for e in (t.get("entries") or []) if e.get("as")][:3]
                it["pics"].append(card_pic(inputs, h))
        else:
            continue
        it["pics"] = [p for p in it["pics"] if p]
        out.append(it)
    return out


def split_evidence(how: Optional[dict]) -> list:
    """What picked a split card's halves (oldgit_rows.split_rows), and the readings that say the other way."""
    if not (how or {}).get("halves"):
        return []
    side = {v: k for k, v in how["halves"].items()}
    out = [f"Top rows {side.get('top')}, bottom rows {side.get('bottom')}. Picked by "
           f"{how.get('matched_by') or 'their order'}'s stats" + (f" and {how['eye']}" if how.get("eye") else "") + "."]
    for name, rd in (how.get("readings") or {}).items():
        if rd.get("top") != side.get("top"):
            days = f" ({', '.join(rd['days'])})" if rd.get("days") else ""
            out.append(f"The {name}{days} reads the other way (top {rd.get('top')}, bottom {rd.get('bottom')}), so "
                       "both timelines show a swap and a swap back at the split.")
    return out


def field_item(inputs: Inputs, r: dict) -> dict:
    cid, eid, field = r.get("card"), r.get("entry") or "", r.get("field")
    entry = inputs.entries.get((cid, eid)) or {}
    old = entry.get("kind") == "old repo"
    section, sub = ("splits", None) if entry.get("as") else ("psd", None) if old else ("matches", "fields")
    it = item("field", r["id"], section, sub, f"{inputs.label(cid)}: {field} on {when(r.get('date'))}"
              + (f" #{entry.get('channel')}" if entry.get("channel") else " (old repo)" if old else ""),
              card=cid, entry=eid, field=field, date=r.get("date"))
    it["meta"] = [f"read: {show(r.get('proposed'))}" if r.get("proposed") is not None else "nothing read",
                  f"how: {r.get('method')}" + (f" ({r['conf']})" if r.get("conf") is not None else ""),
                  r.get("why") or ""]
    it["pics"] = [entry_pic(inputs, cid, eid, "this version"), card_pic(inputs, cid)]
    if r.get("proposed") is not None:
        it["choices"].append(choice("yes", f"yes: {show(r['proposed'])}", r["proposed"]))   # the value confirmed
    it["choices"].append(choice("no", "no, or can't tell (stays out)"))
    it["other"] = value_other(field, stats_shape(r.get("proposed"), card_stats(inputs, cid)) if field == "stats"
                              else None)
    return it


def psd_items(inputs: Inputs) -> list:
    """The old repo rows' fields to review (oldrows): the PSD against the picture. An item card's kinds are not
    asked (owner: not collected; the chain's no_kinds): none when every read is its kinds alone."""
    out = []
    for r in inputs.old_review:
        if not r.get("field"):
            continue
        cid, field = inputs.cid(r.get("card")), r["field"]
        kind = (inputs.cards.get(cid) or {}).get("type")
        if field == "types":
            if dc.old_kinds_only(r, kind):
                continue
            r = {**r, "proposed": None if dc.kinds_only(r.get("proposed"), kind) else
                 dc.without_kinds(r.get("proposed"), kind),
                 "reads": {k: dc.without_kinds(v, kind) for k, v in (r.get("reads") or {}).items()
                           if not dc.kinds_only(v, kind)}}
        it = item("old field", r["id"], "psd", None, f"{r.get('card')}: {field} on {when(r.get('date'))} (old repo)",
                  card=cid, field=field, date=r.get("date"))
        it["meta"] = [r.get("why") or "", f"file {r.get('row', '').split(':', 1)[-1]}"]
        reads = r.get("reads") or {}
        if r.get("recall"):
            it["meta"].append(f"printed words found: {r['recall'].get('words')}")
        it["pics"] = [pic("../" + r["thumb"], "this version") if r.get("thumb") else None, card_pic(inputs, cid)]
        it["pics"] = [p for p in it["pics"] if p]
        if r.get("proposed") is not None:
            it["choices"].append(choice("yes", f"yes: {show(r['proposed'])}", r["proposed"]))
        seen = [dumps(r.get("proposed"), sort_keys=True)]
        for src, value in reads.items():
            if value is None or dumps(value, sort_keys=True) in seen:
                continue
            seen.append(dumps(value, sort_keys=True))
            it["choices"].append(choice("value", f"the {src}'s: {show(value)}", value))
        it["choices"].append(choice("no", "no, or can't tell (stays out)"))
        it["other"] = value_other(field, stats_shape(r.get("proposed"), *reads.values(), card_stats(inputs, cid))
                                  if field == "stats" else None)
        out.append(it)
    return out


def jump(fact: dict) -> Optional[str]:
    return f"https://discord.com/channels/{GUILD}/{fact['channel_id']}/{fact['message']}" \
        if fact.get("channel_id") and fact.get("message") else None


def change_text(fact: dict) -> str:
    if fact.get("kind") == "ability changed":
        return "the ability changed"
    field = str(fact.get("field") or fact.get("kind") or "?").upper()
    parts = [field] + [f"{fact['from']} ->" if fact.get("from") is not None else ""] + \
        [str(fact["to"]) if fact.get("to") is not None else f"{fact['by']:+d}" if isinstance(fact.get("by"), int)
         else ""] + [f"({fact['unit']})" if fact.get("unit") else ""]
    return " ".join(p for p in parts if p)


def fact_items(inputs: Inputs) -> list:
    out = []
    for r in inputs.fact_review:
        if r["kind"] == "unsure fact":
            fact = inputs.facts.get(r.get("fact")) or {}
            cid = inputs.cid(r.get("card"))
            it = item(r["kind"], r["id"], "matches", "facts",
                      f"{inputs.name(cid) if cid else r.get('card')}: {when(r.get('date'))} #{r.get('channel')}",
                      card=cid, fact=r.get("fact"), date=r.get("date"))
            it["meta"] = [f"read as: {change_text(fact)}", f"unsure: {r.get('why')}"]
            it["link"] = jump(fact)
            it["pics"] = [p for p in [card_pic(inputs, cid)] if p]
            if fact.get("unit") == "unknown":
                it["choices"] = [choice("yes", "right, in points", "points"), choice("yes", "right, in bars", "bars")]
            else:
                it["choices"] = [choice("yes", "right: count it")]
            it["choices"].append(choice("no", "wrong: leave it out"))
        elif r["kind"] == "ambiguous card":            # a card picker (owner D9): its facts are the card's
            fact = {"channel_id": r.get("channel_id") or next((f["channel_id"] for f in inputs.facts.values()
                                                               if f.get("message") == r.get("message")), None),
                    "message": r.get("message")}
            it = item(r["kind"], r["id"], "matches", "facts", f"{when(r.get('date'))} #{r.get('channel')}",
                      date=r.get("date"), line=r.get("line"))
            it["meta"] = [f"names: {', '.join(r.get('candidates') or []) or 'no card'}", r.get("why") or "",
                          "Which card is the line about? Its changes (data only) go to that card."]
            it["link"] = jump(fact)
            given = (inputs.owner.get("lines") or {}).get(r["id"])
            answer = inputs.answers.get((given or {}).get("answer"))
            if given and answer:                       # the owner named it: shown, not asked
                entry = next((e for e in inputs.morgue.get("candidates") or () if e["id"] == given.get("morgue")), {})
                to = f"the Morgue's {entry.get('name') or given['morgue']}" if given.get("morgue") else \
                    inputs.label(given.get("card"))
                it.update(read_only=True)
                it["meta"].append(f"The owner's answer ({answer.get('when')}): {to}. {answer['ruling']}")
            else:
                for cid in dict.fromkeys(c for c in (inputs.cid(n) for n in r.get("candidates") or []) if c):
                    it["choices"].append(choice("card", inputs.label(cid), cid, card_pic(inputs, cid)))
                it["choices"].append(choice("none", "none of these: leave the line out"))
                it["other"] = {"c": "card", "kind": "card", "label": "another card"}
        else:
            continue
        out.append(it)
    return out


def created_item(inputs: Inputs, r: dict) -> dict:
    kind, cid = r["kind"], r.get("card")
    made = inputs.created.get((inputs.cards.get(cid) or {}).get("key")) or {}
    file, ours = r.get("file") or made.get("file") or {}, r.get("ours") or made.get("ours") or {}
    sub = {"created earlier": "earlier", "created new": "new", "created later": "later"}.get(kind, "hints")
    it = item(kind, r["id"], "created", sub, inputs.label(cid), card=cid, date=r.get("date") or ours.get("date"))
    where = f"#{ours['channel']}" if ours.get("channel") else ""
    it["meta"] = [f"website: {file.get('date') or 'none'} ({file.get('from') or '-'}); ours: {ours.get('day')} "
                  f"from {ours.get('source')} {where}".rstrip(), r.get("why") or ""]
    if ours.get("picture"):
        it["pics"].append(row_pic(inputs.verdicts.get(ours["picture"]), f"ours {ours.get('day')}"))
    elif ours.get("commit"):
        e = next((e for e in (inputs.timelines.get(cid) or {}).get("entries") or [] if e.get("commit") ==
                  ours["commit"] and e.get("thumb")), None)
        it["pics"].append(pic("../" + e["thumb"], f"ours {ours.get('day')} ({e.get('badge') or 'old repo'})")
                          if e else None)
    hint = r.get("earlier_unsure")
    if hint:
        it["meta"].append(f"an earlier picture the matcher wasn't sure names it: {when(hint.get('date'))} "
                          f"#{hint.get('channel')}")
        it["pics"].append(row_pic(inputs.verdicts.get(hint.get("picture")), f"earlier? {when(hint.get('date'))}"))
    it["pics"].append(card_pic(inputs, cid))
    date_other = {"c": "date", "kind": "date", "label": "another date (YYYY-MM-DD or Before 2020)"}
    if kind == "created earlier":
        it["choices"] = [choice("ours", f"ours: {ours.get('day')}", ours.get("day")),
                         choice("file", f"the website's: {file.get('date')}", file.get("date"))]
        it["other"] = date_other
    elif kind == "created new":
        it["choices"] = [choice("ours", f"ours: {ours.get('day')}", ours.get("day"))]
        it["other"] = date_other
    elif kind == "created later":
        it["choices"] = [choice("file", f"keep the website's: {file.get('date')}", file.get("date")),
                         choice("ours", f"ours: {ours.get('day')}", ours.get("day"))]
        it["other"] = date_other
    else:
        if kind == "changelog earlier":
            fact = inputs.facts.get(r.get("fact")) or {}
            it["meta"].append(f"the line: {change_text(fact)} (#{fact.get('channel')})")
            it["link"] = jump(fact)
        else:
            row = next((q for q in inputs.first_repo if "Q-" + q["id"] == r.get("row")), {})
            it["meta"].append(f"first repo push: {row.get('author')}: {row.get('subject')}")
        it["choices"] = [choice("yes", f"yes: Created {when(r.get('date'))}", when(r.get("date"))), choice("no", "no")]
    answer = next((a for a in dc.OWNER_ANSWERS if a[0] == r.get("answer")), None) if r.get("read_only") else None
    if answer:                                    # the owner's ruling (Q6's three): shown, not asked
        it.update(read_only=True, choices=[], other=None)
        it["meta"].append(f"The owner's answer {answer[0]} ({answer[2]}): {answer[3]}")
    return it


def morgue_groups(morgue: dict, ideas: list) -> tuple:
    """(part, entries, on the list as built)."""
    return (("candidates", morgue.get("candidates") or [], "yes"), ("ideas", ideas, "no"),
            ("not", morgue.get("not_candidates") or [], "no"))


def morgue_list(root: str) -> Optional[dict]:
    """The Morgue list with the owner's answers (derived/review/morgue.json, for the preview): the candidates not
    answered 'no', the ideas and the rest answered 'yes'. None when the Morgue part has no output."""
    morgue = dm.read_json(join(root, *MORGUE), {}) or {}
    if not morgue:
        return None
    ideas = (dm.read_json(join(root, *IDEAS), {}) or {}).get("ideas") or []
    rulings = (dm.read_json(join(root, *RULINGS), {}) or {}).get("rulings") or {}
    entries = []
    for _, group, now in morgue_groups(morgue, ideas):
        for e in group:
            r = rulings.get(f"morgue:{e['id']}") or {}
            if r.get("choice", now) == "yes":
                entries.append({**e, **({"owner": r["choice"]} if r else {})})
    out = {"about": "The Morgue list (owner's answers Q1, Q2) with the review page's answers: never cards[], decks, "
                    "Unused, the default search or Mentioned by.", "built": dm.now_iso(), "entries": entries}
    dm.write_json(join(root, *OUT, "morgue.json"), out)
    return out


def place(x: dict) -> str:
    """'#channel', else '(source)' (the old repo is a source, never a channel)."""
    if x.get("channel") and x["channel"] != "old repo":
        return f"#{x['channel']}"
    source = x.get("source") or x.get("channel")
    return f"({source})" if source else ""


def nearest_text(e: dict) -> str:
    """'nearest live card: X (n bits | by eye)', or ''."""
    n = e.get("nearest") or {}
    if not n:
        return ""
    how = f"{n['bits']} bits" if n.get("bits") is not None else f"by {n['by']}" if n.get("by") else ""
    return f"nearest live card: {n.get('name')}" + (f" ({how})" if how else "")


def morgue_items(inputs: Inputs) -> list:
    out = []
    for sub, entries, now in morgue_groups(inputs.morgue, inputs.ideas):
        for e in entries:
            first = e.get("first") or {}
            it = item(MORGUE_KINDS[sub], f"morgue:{e['id']}", "morgue", sub, e.get("name") or e["id"],
                      entry=e["id"], date=first.get("date"), now=now)
            it["meta"] = [f"{e.get('status')}{(' (' + e['kind'] + ')') if e.get('kind') else ''}; first "
                          f"{first.get('day') or when(first.get('date'))} {place(first)}",
                          "; ".join(e.get("why") or []), nearest_text(e)]
            it["pics"] = [pic("../morgue/" + p["thumb"], f"{when(p.get('date'))} {place(p)}")
                          for p in (e.get("pictures") or [])[:6] if p.get("thumb")]
            label = {"yes": "on the Morgue list", "no": "leave it off"}
            it["choices"] = [choice("yes", label["yes"] + (" (as now)" if now == "yes" else "")),
                             choice("no", label["no"] + (" (as now)" if now == "no" else ""))]
            answer = owner_answer(inputs, it["id"], e)
            if answer:                                # the owner answered it: shown, not asked
                it.update(read_only=True, choices=[])
                it["meta"].append(f"The owner's answer ({answer.get('when')}): {answer['title']} = "
                                  f"{answer['answer']}. {answer['ruling']}")
            out.append(it)
    groups = (inputs.morgue.get("candidates") or [], inputs.ideas, inputs.morgue.get("not_candidates") or [])
    thumbs = {e["id"]: e.get("thumb") for es in groups for e in es}
    pictures = {e["id"]: e.get("pictures") or [] for es in groups for e in es}
    for h in inputs.morgue.get("created_hints") or []:
        cid, day = h.get("card"), h.get("day") or (str(h["date"])[:10] if h.get("date") else None)
        before = inputs.created_before(cid)
        if h.get("meme") or not day or not before or created_sort(day) >= before:
            continue                              # a meme channel's date, or not earlier than the Created in use
        shot = next((q for q in pictures.get(h["entry"]) or [] if h.get("date") and q.get("date") == h["date"]), {})
        it = item("morgue hint", f"hint:{h['entry']}:{cid}", "created", "hints", inputs.label(cid), card=cid,
                  entry=h["entry"], date=h.get("date"), **({"day": h["day"]} if h.get("day") else {}),
                  **({"picture": shot["id"]} if shot.get("id") else {}), channel=h.get("channel"))
        it["meta"] = [f"a Morgue list picture '{h.get('name')}' may be its early look: {day} "
                      f"{place(h)}" + (" (weak)" if h.get("weak") else "")]
        it["pics"] = [p for p in (pic("../morgue/" + thumbs[h["entry"]], "the Morgue picture")
                                  if thumbs.get(h["entry"]) else None, card_pic(inputs, cid)) if p]
        it["choices"] = [choice("yes", f"yes: Created {day}"), choice("no", "no")]
        out.append(it)
    return out


def owner_answer(inputs: Inputs, iid: str, e: Optional[dict] = None) -> Optional[dict]:
    """The owner's answer (owner.json) an item is shown with read only: listed by its id (the joke edits of live
    cards too, each by eye), or an entry the owner's ruling made."""
    aid = (inputs.owner.get("read_only") or {}).get(iid) or (e or {}).get("owner")
    return inputs.answers.get(aid) if aid else None


def owner_notes(inputs: Inputs, items: list) -> None:
    """The owner's answers that bear on an item still open (owner.json notes): a line of its meta."""
    notes = inputs.owner.get("notes") or {}
    for it in items:
        note = notes.get(it["id"])
        answer = inputs.answers.get((note or {}).get("answer"))
        if note and answer:
            it["meta"].append(f"The owner's answer ({answer.get('when')}): {answer['title']} = {answer['answer']}"
                              + (f": {note['note']}" if note.get("note") else ""))


def offered(it: dict, r: dict) -> bool:
    """An answer one of the item's radios takes: a choice with its value, or the typed one."""
    return any(x["c"] == r.get("choice") and x.get("value") == r.get("value") for x in it.get("choices") or []) or \
        r.get("choice") == (it.get("other") or {}).get("c")


def gather(inputs: Inputs) -> list:
    items = owner_answers(inputs) + old_maps(inputs) + match_items(inputs) + chain_items(inputs) + \
        psd_items(inputs) + fact_items(inputs) + morgue_items(inputs)
    for it in items:
        it["pics"] = [p for p in it["pics"] if p]
        answer = None if it.get("read_only") else owner_answer(inputs, it["id"])
        if answer:                                   # an item the owner answered outside the page: shown, not asked
            it.update(read_only=True, choices=[], other=None)
            it["meta"].append(f"The owner's answer ({answer.get('when')}): {answer['title']} = {answer['answer']}. "
                              f"{answer['ruling']}")
        r = inputs.rulings.get(it["id"])
        if r and r.get("source") and r["source"] != "owner":
            it["meta"].append(f"answered by: {r['source']}")
    owner_notes(inputs, items)
    have = {i["id"] for i in items}
    for it in items:                                 # answered, but no radio here takes that answer any more (a
        r = inputs.rulings.get(it["id"])             # look's version gone): not applied
        if r and not it.get("read_only") and not offered(it, r):
            it["stale"] = True
    for rid, r in inputs.rulings.items():            # answered, its item gone: taken in (the chain acted on it) or
        if rid not in have and r.get("item"):        # changed (a regrouped look, a Created status that moved)
            items.append({**{k: v for k, v in r["item"].items() if k != "stale"},
                          "gone": "taken" if rid in inputs.used else "changed"})
    order = {s[0]: n for n, s in enumerate(SECTIONS)}
    subs = {s: {p[0]: n for n, p in enumerate(parts)} for s, parts in SUBS.items()}
    return sorted(items, key=lambda i: (order[i["section"]], subs.get(i["section"], {}).get(i.get("sub"), 0),
                                        f"{i.get('order', 0):04d}" if i["section"] == "answers" else
                                        i.get("title") if i["section"] in ("psd", "created", "maps") or
                                        i.get("sub") == "fields" else "",
                                        str(i.get("date") or ""), i["id"]))


# ========================
# The page
# ========================
STYLE = """
body{font:14px/1.4 system-ui,sans-serif;margin:0 0 6rem;background:#1e1f22;color:#dbdee1}
header{padding:1rem 1.5rem;background:#2b2d31;position:sticky;top:0;z-index:2;box-shadow:0 2px 6px #0008}
h1{margin:0 0 .3rem;font-size:1.3rem}h2{margin:1.6rem 1.5rem .3rem;font-size:1.2rem;color:#f0b232}
h2{scroll-margin-top:10rem}h3{margin:1rem 1.5rem .3rem;font-size:1.05rem}p.about{margin:.2rem 1.5rem;color:#b5bac1}
nav a{color:#00a8fc;margin-right:1rem;text-decoration:none}nav .n{color:#949ba4}
details.sec>summary,details.sub>summary{cursor:pointer;list-style:revert}
.item{margin:.6rem 1.5rem;padding:.7rem 1rem;background:#2b2d31;border-radius:8px;border-left:4px solid #4e5058}
.item.done{border-left-color:#23a55a}.item.ro{border-left-color:#5865f2}.item.gone{opacity:.75}
.item.changed{border-left-color:#f0b232}
.item .id{color:#949ba4;font:12px ui-monospace,monospace;float:right}
.item .t{font-weight:600}.meta{color:#b5bac1;margin:.2rem 0 .4rem}.meta div{margin:.1rem 0}
.pics{display:flex;gap:.6rem;flex-wrap:wrap;align-items:flex-start}
figure{margin:0;width:150px}figure img,figure .crop{width:150px;border-radius:4px;background:#111;display:block}
figure .crop{background-repeat:no-repeat}figcaption{font-size:12px;color:#b5bac1}
.ch{display:flex;gap:.5rem 1.2rem;flex-wrap:wrap;margin-top:.4rem}.ch label{cursor:pointer}
.ch label.pick{display:flex;flex-direction:column;width:150px}.ch input[type=text]{width:16rem;background:#1e1f22;
color:#dbdee1;border:1px solid #4e5058;border-radius:4px;padding:.15rem .3rem}
button{background:#5865f2;color:#fff;border:0;border-radius:4px;padding:.3rem .7rem;cursor:pointer;margin:.2rem}
button.small{background:#4e5058;padding:.1rem .5rem;font-size:12px}a{color:#00a8fc}
footer{position:fixed;bottom:0;left:0;right:0;background:#111214;padding:.5rem 1.5rem;display:flex;gap:1rem;
align-items:center;z-index:2}footer textarea{flex:1;height:2.6rem;background:#2b2d31;color:#dbdee1;border:0}
body.open-only .item.done,body.open-only .item.ro{display:none}
"""

SCRIPT = """
const STORE = PAGE.store, saved = JSON.parse(localStorage.getItem(STORE) || "{}");
function state(id) { return saved[id] || RULED[id] || null; }
function canon(v) {                                            // keys sorted: a value as Python's == sees it
  return JSON.stringify(v ?? null, (k, x) => x && typeof x === "object" && !Array.isArray(x) ?
    Object.fromEntries(Object.keys(x).sort().map(n => [n, x[n]])) : x);
}
function same(a, b) { return canon(a) === canon(b); }
function typedText(v) {                                         // a stored value as typed (stats: hp/def/atk/spd)
  if (Array.isArray(v)) return v.join(", ");
  if (v && typeof v === "object" && v.creatures) return v.creatures.map(typedText).join(" | ");
  if (v && typeof v === "object") return Object.values(v).join("/");
  return v ?? "";
}
function show(id) {
  const box = document.getElementById(id); if (!box) return;
  const s = state(id), radios = [...box.querySelectorAll("input[type=radio]")];
  const live = s && s.choice !== "clear";
  const hit = live && (radios.find(r => r.dataset.other === undefined && r.dataset.c === s.choice &&
                                       same(JSON.parse(r.dataset.v || "null"), s.value ?? null)) ||
                       radios.find(r => r.dataset.other !== undefined && r.dataset.c === s.choice));
  radios.forEach(r => { r.checked = r === hit; });
  const text = box.querySelector("input.other");
  if (text) text.value = hit && hit.dataset.other !== undefined ? typedText(s.value) : "";
  box.classList.toggle("done", !!hit);
}
function pick(r) {
  const box = r.closest(".item"), id = box.id;
  const other = r.dataset.other !== undefined, text = box.querySelector("input.other");
  saved[id] = {choice: r.dataset.c, ...(other ? {value: text.value.trim()} :
                                       r.dataset.v ? {value: JSON.parse(r.dataset.v)} : {})};
  localStorage.setItem(STORE, JSON.stringify(saved)); box.classList.add("done"); counts();
}
function typed(t) { const r = t.closest("label").querySelector("input[type=radio]"); r.checked = true; pick(r); }
function clearOne(b) {
  const id = b.closest(".item").id;
  if (RULED[id]) saved[id] = {choice: "clear"}; else delete saved[id];
  localStorage.setItem(STORE, JSON.stringify(saved)); show(id); counts();
}
function bulk(b) {
  document.querySelectorAll(`.item[data-sub="${b.dataset.sub}"]`).forEach(box => {
    if (box.classList.contains("done")) return;
    const r = box.querySelector(`input[type=radio][data-c="${b.dataset.c}"]`); if (r) { r.checked = true; pick(r); }
  });
}
function counts() {
  document.querySelectorAll("[data-count]").forEach(el => {
    const sel = el.dataset.count, all = document.querySelectorAll(`.item[data-${sel}]:not(.ro)`);
    const done = [...all].filter(x => x.classList.contains("done")).length;
    el.textContent = all.length ? `${done} of ${all.length} answered` : "";
  });
}
function answers() {
  const out = {};
  for (const [id, s] of Object.entries({...RULED, ...saved})) {
    if ((s.choice === "clear" && !RULED[id]) || !document.getElementById(id)) continue;   // gone from the page
    out[id] = {choice: s.choice, ...(s.value !== undefined ? {value: s.value} : {})};
  }
  return {format: PAGE.format, page: PAGE.built, exported: new Date().toISOString(), rulings: out};
}
function exportOut() {
  const text = JSON.stringify(answers(), null, 1), o = document.getElementById("out");
  o.value = text;
  const a = document.createElement("a"), stamp = new Date().toISOString().slice(0, 16).replace(/[-:T]/g, "");
  a.href = URL.createObjectURL(new Blob([text], {type: "application/json"}));
  a.download = `timelines-rulings-${stamp}.json`; document.body.appendChild(a); a.click(); a.remove();
}
function forget() {
  if (!confirm("Forget the answers saved in this browser (not yet applied)?")) return;
  localStorage.removeItem(STORE); for (const k of Object.keys(saved)) delete saved[k];
  document.querySelectorAll(".item").forEach(b => show(b.id)); counts();
}
document.addEventListener("DOMContentLoaded", () => {
  for (const [id, s] of Object.entries(saved))                  // applied since: the page has them now
    if (RULED[id] && s.choice === RULED[id].choice && same(s.value, RULED[id].value)) delete saved[id];
  document.querySelectorAll(".item").forEach(b => show(b.id));
  for (const [id, s] of Object.entries(saved)) {                // picked on an older page, no radio takes it now
    const box = document.getElementById(id);
    if (box && s.choice !== "clear" && !box.classList.contains("done")) { delete saved[id]; show(id); }
  }
  localStorage.setItem(STORE, JSON.stringify(saved)); counts();
  document.getElementById("openonly").addEventListener("change", e =>
    document.body.classList.toggle("open-only", e.target.checked));
});
"""


def pic_html(p: dict) -> str:
    cap = f"<figcaption>{escape(p.get('cap') or '')}</figcaption>"
    src = escape(p["src"], quote=True)
    box = p.get("box")
    if box:
        x0, y0, x1, y1 = box
        w, h = max(x1 - x0, 1e-3), max(y1 - y0, 1e-3)
        px = 0 if w >= 1 else 100 * x0 / (1 - w)
        py = 0 if h >= 1 else 100 * y0 / (1 - h)
        ar = p.get("ar") or 0.5
        return (f"<figure><a href='{src}' target=_blank><div class=crop role=img style=\"aspect-ratio:{ar};"
                f"background-image:url('{src}');background-size:{100 / w:.2f}% {100 / h:.2f}%;"
                f"background-position:{px:.2f}% {py:.2f}%\"></div></a>{cap}</figure>")
    return f"<figure><a href='{src}' target=_blank><img loading=lazy src='{src}'></a>{cap}</figure>"


def item_html(it: dict) -> str:
    cls = ["item"] + (["ro"] if it.get("read_only") else []) + (["gone"] if it.get("gone") else []) + \
        (["changed"] if it.get("gone") == "changed" or it.get("stale") else [])
    head = (f"<div class='{' '.join(cls)}' id='{escape(it['id'], quote=True)}' data-sec='{it['section']}'"
            + (f" data-sub='{it['section']}-{it['sub']}'" if it.get("sub") else "") + ">"
            + f"<span class=id>{escape(it['id'])}</span><div class=t>{escape(it['title'])}</div>")
    meta = "".join(f"<div>{escape(m)}</div>" for m in it.get("meta") or [] if m)
    if it.get("gone") == "changed":
        meta += ("<div><b>Not applied: this item changed or left the page since you answered (a look regrouped, a "
                 "Created date that moved). Answer it again where it is now, or clear this one.</b></div>")
    elif it.get("gone"):
        meta += "<div><b>Answered and taken in by the rerun.</b></div>"
    elif it.get("stale"):
        meta += ("<div><b>Not applied: your answer is no longer one this item offers (it proposes another value, "
                 "date or card now, a look's version left the card's history, a card no longer live). Answer it "
                 "again, or clear it.</b></div>")
    if it.get("link"):
        meta += f"<div><a href='{escape(it['link'], quote=True)}' target=_blank>open the line in Discord</a></div>"
    body = [head, f"<div class=meta>{meta}</div>"]
    if it.get("pics"):
        body.append("<div class=pics>" + "".join(pic_html(p) for p in it["pics"]) + "</div>")
    if not it.get("read_only"):
        name = escape("r-" + it["id"], quote=True)
        opts = []
        for c in it.get("choices") or []:
            value = f" data-v='{escape(dumps(c['value']), quote=True)}'" if "value" in c else ""
            radio = f"<input type=radio name='{name}' data-c='{c['c']}'{value} onchange=pick(this)>"
            if c.get("pic"):
                opts.append(f"<label class=pick>{pic_html(c['pic'])}<span>{radio} {escape(c['label'])}</span></label>")
            else:
                opts.append(f"<label>{radio} {escape(c['label'])}</label>")
        other = it.get("other")
        if other:
            listed = " list=cards" if other["kind"] == "card" else ""
            opts.append(f"<label><input type=radio name='{name}' data-c='{other['c']}' data-other "
                        f"onchange=pick(this)> {escape(other['label'])}: <input type=text class=other "
                        f"data-kind='{other['kind']}'{listed} onchange=typed(this)></label>")
        opts.append("<button class=small onclick=clearOne(this)>clear</button>")
        body.append("<div class=ch>" + "".join(opts) + "</div>")
    return "".join(body) + "</div>"


def page(items: list, inputs: Inputs, built: str) -> str:
    by_sec = defaultdict(list)
    for it in items:
        by_sec[it["section"]].append(it)
    ruled = {rid: {k: r[k] for k in ("choice", "value") if k in r} for rid, r in inputs.rulings.items()}
    data = {"store": f"{STORE}:{inputs.root}", "format": FORMAT, "built": built}
    cards_list = "<datalist id=cards>" + "".join(       # a name two cards share: its id typed, the name shown
        f"<option value='{escape(cid, quote=True)}' label='{escape(inputs.label(cid), quote=True)}'>"
        if (c.get("name") or cid).casefold() in inputs.shared else
        f"<option value='{escape(c.get('name') or cid, quote=True)}'>" for cid, c in sorted(inputs.cards.items())) + \
        "</datalist>"
    open_n = sum(1 for i in items if not i.get("read_only") and (i["id"] not in ruled or i.get("stale")))
    missed = sum(1 for i in items if i.get("stale") or i.get("gone") == "changed")    # answered, not applied
    out = ["<!doctype html><meta charset='utf-8'><title>Card timelines: review</title>",
           f"<style>{STYLE}</style>",
           f"<script>const PAGE = {dumps(data)}; const RULED = {dumps(ruled, ensure_ascii=False)};{SCRIPT}</script>",
           "<header><h1>Card timelines: review</h1>",
           f"<div>{open_n} items open, {len(ruled) - missed} answered and applied"
           + (f", {missed} answered but not applied (marked orange)" if missed else "") + ". Pick an answer per item; "
           "your picks are kept in this browser. Then press <b>Export answers</b> (a JSON file downloads) and run "
           "<code>~/.venvs/ac-py314-fork/bin/python scripts/apply_rulings.py &lt;that file&gt;</code> in AutoCroissant: "
           "it checks every answer, keeps them in derived/review/rulings.json, rebuilds the timelines and this page. "
           "Items left unanswered stay as built. Nothing here is published.</div>",
           "<nav>" + "".join(f"<a href='#{s}'>{escape(t.split(' (')[0])} <span class=n>({len(by_sec[s])})</span></a>"
                             for s, t, _ in SECTIONS) +
           "<label><input type=checkbox id=openonly> only open items</label></nav></header>", cards_list]
    for sec, title, about in SECTIONS:
        its = by_sec.get(sec) or []
        out.append(f"<h2 id={sec}>{escape(title)} <small data-count='sec=\"{sec}\"'></small></h2>"
                   f"<p class=about>{escape(about)}</p>")
        parts = SUBS.get(sec)
        if not parts:
            out += [item_html(i) for i in its]
            continue
        for sub, subtitle, bulk in parts:
            group = [i for i in its if i.get("sub") == sub]
            if not group:
                continue
            key = f"{sec}-{sub}"
            button = (f" <button class=small data-sub='{key}' data-c='{bulk}' onclick=bulk(this)>answer "
                      f"'{bulk}' to every open one here</button>") if bulk else ""
            out.append(f"<details class=sub {'open' if len(group) <= 40 else ''}><summary><h3 style='display:inline'>"
                       f"{escape(subtitle)} ({len(group)}) <small data-count='sub=\"{key}\"'></small></h3>"
                       f"</summary>{button}" + "".join(item_html(i) for i in group) + "</details>")
    out.append("<footer><button onclick=exportOut()>Export answers</button><textarea id=out readonly "
               "placeholder='the exported answers show here too'></textarea><button class=small onclick=forget()>"
               "forget unapplied answers</button></footer>")
    return "\n".join(out) + "\n"


def register(root: str, items: list, built: str) -> dict:
    """derived/review/ids.json: every id the page has shown (first and last build, its card, kind and title as last
    shown; a map's old id): the chain keeps a picture's id while it asks about the same card (dc.picture_id), and
    apply_rulings tells an answer whose item a rebuild settled or dropped from a mistyped id."""
    path = join(root, *ISSUED)
    ids = (dm.read_json(path, {}) or {}).get("ids") or {}
    for it in items:
        was = ids.get(it["id"]) or {"first": built, "card": it.get("card")}
        if it.get("gone") or was.get("card") != it.get("card"):     # an old page's id asking about another card
            continue                                                 # (built before ids.json): its meaning kept
        ids[it["id"]] = {**was, "kind": it["kind"], "title": it["title"], "last": built,
                         **({"old_id": it["old_id"]} if it.get("old_id") else {})}
    dm.write_json(path, {"about": "Every review id the page has shown (timeline_review.register): first and last "
                                  "build, its card (the first), kind, title (as last shown). No message text.",
                         "updated": built, "ids": dict(sorted(ids.items()))}, indent=None)
    return ids


def build(root: str = dm.DEFAULT_ROOT, log: Callable = print) -> dict:
    root = dm.check_root(root)
    inputs = Inputs(root)
    items = gather(inputs)
    ids = Counter(i["id"] for i in items)
    dup = [i for i, n in ids.items() if n > 1]
    if dup:
        raise SystemExit(f"review ids not unique: {dup[:5]}")
    built = dm.now_iso()
    out = join(root, *OUT)
    cards = {cid: c.get("name") or cid for cid, c in inputs.cards.items()}
    dm.write_json(join(out, "items.json"), {"format": FORMAT, "built": built, "cards": cards, "items": items},
                  indent=None)
    dm.write_text(join(out, "index.html"), page(items, inputs, built))
    register(root, items, built)
    tally = Counter((i["section"], i.get("sub") or "") for i in items)
    counts = {f"{s}/{p}" if p else s: n for (s, p), n in sorted(tally.items())}
    open_n = sum(1 for i in items if not i.get("read_only") and (i["id"] not in inputs.rulings or i.get("stale")))
    log(f"review page: {len(items)} items ({open_n} open, {len(inputs.rulings)} answered) -> {join(out, 'index.html')}")
    log("  " + ", ".join(f"{k} {n}" for k, n in counts.items()))
    return {"items": len(items), "counts": counts, "built": built}


def main(argv: Optional[list] = None) -> int:
    parser = ArgumentParser(prog="python -m commands.timeline_review", description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=("build",))
    parser.add_argument("--root", default=dm.DEFAULT_ROOT)
    args = parser.parse_args(argv)
    build(args.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
