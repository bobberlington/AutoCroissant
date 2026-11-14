"""
The card PSDs posted on Discord, read exactly, for the card timelines (docs/Future/CARD_TIMELINE.md, step 3 Read):

    python -m commands.discord_psds parse [--root <dir>] [--cards-repo <dir>]   (a Python with the psd-tools fork)
    python -m commands.discord_psds read  [--root <dir>]                        (a Python with pyobjc's Vision)

parse   Each distinct stored PSD (attachments.jsonl; left-out channels' posts dropped) is opened with psd-tools: its
        composite hashed as the matcher hashes a card (whole, art) and kept as a webp at most PICTURE_W wide (the
        timeline's picture); its card from its printed title, its file name or its composite's nearest git version
        (CARD_BITS); then read by PSDParser at that card's path (its kind): the printed name, the ability (text layers
        named after their text read as it), the types (icon layers). Templates and PSDs naming no live card are
        listed, card None.
read    Each composite's stat bars by the era bar readers (OCR'd labels, the era's reader, calibration.json's bars),
        as for the old repo's PNGs: the printed picture's stats.
-> the archive's derived/psds/psds.jsonl (one row per PSD: posts, card, fields with method 'psd' or 'picture'),
pictures/ (the composites). The chain attaches a PSD's fields to the look posted in its own message, else to the
nearest within ATTACH_BITS that no PSD read otherwise holds, or adds the PSD as its own version, dated by its post.
No message text or posters.
"""
from argparse import ArgumentParser
from json import load, loads
from os import environ, makedirs
from os.path import basename, exists, join, splitext
from re import sub
from sys import stderr
from typing import Optional

from commands import discord_match as dm

# ========================
# Configuration
# ========================
OUT = ("derived", "psds")
VOCAB = ("derived", "oldgit", "vocabulary.json")    # the type and star vocabulary the old repo's reads used
PICTURE_W = 720                                     # the composite kept (the website's card view)
CARD_BITS = 4.0                                     # a composite this near a card's git version: that card
NAME_FIT = 0.9                                      # a printed title or file name this like a card's name
ATTACH_BITS = 3.0                                   # a chain look this near a PSD's composite shows it
FIELDS = ("name", "stars", "types", "ability", "stats")
STAT_WORDS = {"health", "defense", "defence", "attack", "speed", "hp", "def", "atk", "spd"}
NOT_ABILITY = {"name", "title", "attribute", "class", "type", "types", "stars"}    # layer names that are no ability
ITEM_KINDS = {"item", "aux item", "nme"}             # their PSD types are item subtypes
RECALL = 0.8                                        # an ability's words the composite's OCR reads in order: sure
ALIKE = 0.6                                         # an OCR'd word this like a text word: a misread of it


def log(text: str):
    print(text, file=stderr, flush=True)


def psd_posts(root: str) -> dict:
    """{sha256: {'file', 'filename', 'posts': [post rows, oldest first]}} of every stored PSD attachment."""
    from commands.discord_verdicts import post_row
    records, _ = dm.load_posts(root)
    found = {}
    for line in open(join(root, "attachments.jsonl"), encoding="utf-8"):
        a = loads(line)
        name = str(a.get("filename") or "")
        if not name.lower().endswith((".psd", ".psb")) or not a.get("sha256") or a.get("message_id") not in records:
            continue
        entry = found.setdefault(a["sha256"], {"file": a["file"], "filename": name, "posts": []})
        entry["posts"].append(post_row(records[a["message_id"]], set()))
    for entry in found.values():
        entry["posts"].sort(key=lambda p: int(p["message"]))
    return found


def card_of(title: str, stem: str, look: Optional[dict], cards: dict, names, versions: list) -> tuple:
    """(card id, how) from the printed title, the file name, else the composite's nearest git version."""
    for text, how in ((title, "printed title"), (stem, "file name")):
        exact = sorted(names.exact(text or ""))
        if len(exact) == 1:
            return exact[0], how
        hits = names.fuzzy(text or "", NAME_FIT, limit=2) if text and len(dm.alnum(text)) >= 3 else {}
        best = sorted(hits.items(), key=lambda kv: -kv[1][0])
        if best and (len(best) == 1 or best[0][1][0] > best[1][1][0]):
            return best[0][0], how
    if look and look.get("whole"):
        near = sorted((dm.hash_distance(look["whole"], v["whole"]), v["card"]) for v in versions if v.get("whole"))
        if near and near[0][0] <= CARD_BITS:
            return near[0][1], f"composite {near[0][0]:.1f} bits from its git version"
    return None, None


def ability_layers(psd) -> list:
    """Visible text layers in the card's lower part that print no title, stat label, class line or placeholder: the
    ability a card set in a layer named after its text (no 'Ability' layer; scripts/oldgit_rows.py's rule)."""
    from commands.psd_parser import base_name
    w, h = psd.width, psd.height
    found = []
    for layer in psd.descendants():
        if layer.kind != "type" or not layer.is_visible() or base_name(layer).strip().lower() == "ability":
            continue
        try:
            text = layer.text or ""
        except Exception:
            continue
        letters = sub(r"[^0-9a-z]", "", dm.alnum(text))
        x0, y0, x1, y1 = layer.bbox
        if len(letters) < 2 or letters.startswith("loremipsum") or letters in STAT_WORDS or \
                base_name(layer).strip().lower() in NOT_ABILITY:
            continue
        if x1 <= max(0, x0) or y1 <= max(0, y0) or x0 >= w or y0 >= h or y0 < 0.45 * h:
            continue
        if h >= 1.8 * w and y0 > 0.93 * h and len(text.strip()) <= 40:
            continue                                   # a creature's class line ('Nature/Brute')
        found.append((layer, letters))
    return [layer for layer, letters in found if not any(letters != other and letters in other for _, other in found)]


def parse(root: str, cards_repo: str) -> list:
    """The parse step: psds.jsonl without stats read."""
    from PIL import Image
    from psd_tools import PSDImage
    from commands.card_check import clone_vocabulary
    from commands.card_models import StatsDatabase
    from commands.psd_parser import CardValidator, IconVocabulary, PSDParser, variants_of
    Image.MAX_IMAGE_PIXELS = None
    out_dir = join(root, *OUT)
    makedirs(join(out_dir, "pictures"), exist_ok=True)
    pairs = dm.read_json(dm.Paths(root).pairs, {}) or {}
    cards, versions = pairs.get("cards") or {}, pairs.get("versions") or []
    names = dm.Names(cards)
    vocab = load(open(join(root, *VOCAB)))
    db = StatsDatabase()
    db.all_types = list(vocab["types"])
    parser = PSDParser(db.all_types, db, IconVocabulary.build(db.all_types, vocab["stars"],
                                                              clone_vocabulary(cards_repo).manifest))
    rows = []
    for sha, entry in sorted(psd_posts(root).items(), key=lambda kv: int(kv[1]["posts"][0]["message"])):
        row = {"id": f"S-{sha[:12]}", "sha256": sha, "file": entry["file"], "filename": entry["filename"],
               "posts": [{k: p[k] for k in ("message", "channel", "thread", "date", "bot")} for p in entry["posts"]]}
        try:
            psd = PSDImage.open(join(root, entry["file"]))
            comp = psd.topil()
            row["size"] = [psd.width, psd.height]
            if comp is not None:
                flat = Image.new("RGBA", comp.size, dm.BACKGROUND)
                flat.alpha_composite(comp.convert("RGBA"))
                feat = dm.features(flat.convert("RGB"), ocr=False)
                row["look"] = {k: feat[k] for k in ("w", "h", "shape", "whole", "art") if k in feat}
                pic = flat.convert("RGB")
                if pic.width > PICTURE_W:
                    pic = pic.resize((PICTURE_W, max(1, round(PICTURE_W * pic.height / pic.width))), Image.LANCZOS)
                row["picture"] = "/".join(OUT + ("pictures", f"{row['id']}.webp"))
                pic.save(join(root, row["picture"]), "WEBP", quality=85, method=5)
            first = parser.parse(psd, f"Creatures/Unknown/0 Stars/{basename(entry['filename'])}")
            title = first.display_name or first.name
            cid, how = card_of(title, splitext(entry["filename"])[0].replace("_", " "), row.get("look"), cards,
                               names, versions)
            row.update(title=title, card=cid, card_how=how)
            if cid and (cards[cid].get("psd") or cards[cid].get("png")):
                rel = cards[cid].get("psd") or cards[cid]["png"][:-4] + ".psd"
                card = parser.parse(psd, rel, **({"variant": 1} if len(variants_of(psd)) > 1 else {}))
                if not (card.ability or "").strip():          # the text kept in a layer named after it: read again
                    layers = ability_layers(psd)
                    for layer in layers:
                        layer.name = "Ability"
                    if layers:
                        card = parser.parse(psd, rel, **({"variant": 1} if len(variants_of(psd)) > 1 else {}))
                        row["ability_from"] = "text layers named after their text"
                notes = " ".join(str(n) for n in [*(card.problems or []), *(CardValidator.validate(card) or [])])
                row["read"] = {"name": card.display_name or card.name, "kind": card.card_type,
                               "types": sorted(card.types or []), "ability": card.ability,
                               "stars_drawn": parser._count_stars(psd), "notes": notes[:400]}
        except Exception as e:                  # one bad PSD never stops the step
            row["error"] = f"{type(e).__name__}: {str(e)[:200]}"
        rows.append(row)
        log(f"parse: {row['id']} {row.get('card')} ({row.get('card_how')}) {row.get('error') or ''}")
    dm.write_text(join(out_dir, "psds.jsonl"), "".join(dm.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    return rows


def read(root: str) -> list:
    """The read step: each composite's stats by the era bar readers, then every row's fields."""
    from PIL import Image
    from commands import era_bars as eb
    out_dir = join(root, *OUT)
    rows = dm.read_jsonl(join(out_dir, "psds.jsonl"))
    cal = load(open(join(root, *eb.OUT, "calibration.json")))
    for row in rows:
        if not row.get("picture") or not row.get("card"):
            continue
        with Image.open(join(root, row["picture"])) as im:
            img = im.convert("RGB")
        lines = eb._ocr(img)
        got = eb.read_card(img, lines, eb.look(img, row["posts"][0]["date"][:19]), None, eb.title_of(lines),
                           row["card"])
        bar = cal["groups"].get(eb.group_of(got), {}).get("discord_truth", {}).get("bar")
        for st in (got.get("stats") or {}).values():
            st["sure"] = bar is not None and st["conf"] >= bar and not st.get("hold")
        row["bars"] = {k: got.get(k) for k in ("era", "labels", "reader", "stats", "two")}
        ability = (row.get("read") or {}).get("ability") or ""
        if ability.strip():
            seen = set(dm.words(" ".join(line[0] for line in lines)).split())
            mine = [w for w in dm.words(ability).split() if len(w) >= 3]
            row["ability_recall"] = round(sum(w in seen for w in mine) / len(mine), 3) if mine else None
            row["ability_check"] = printed_in_order(ability, [line[0] for line in lines])
    for row in rows:
        row["fields"] = fields_of(row)
    dm.write_text(join(out_dir, "psds.jsonl"), "".join(dm.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    log(f"read: {sum(1 for r in rows if r.get('card'))} of {len(rows)} PSDs name a card")
    return rows


def _alike(word: str, ocr: str) -> bool:
    """An OCR'd word that is this word misread or run together with its neighbours ('caatxopt')."""
    return word in ocr or (len(ocr) >= 2 and ocr in word) or dm.ratio(word, ocr) >= ALIKE


def printed_in_order(ability: str, lines: list) -> dict:
    """The ability's words against the composite's OCR in reading order: order, the share matched in order;
    unprinted, its words with nothing like them opposite (moved or overflowed: Twin Emperors' labels, Old Chica's
    last 'attack'); inner, real words the composite prints inside the matched span that the text lacks there."""
    from difflib import SequenceMatcher
    mine, ocr = dm.words(ability).split(), dm.words(" ".join(lines)).split()
    if not mine:
        return {"order": None, "unprinted": [], "inner": []}
    sm = SequenceMatcher(None, mine, ocr, autojunk=False)
    blocks = [b for b in sm.get_matching_blocks() if b.size]
    first, last = (min(b.b for b in blocks), max(b.b + b.size for b in blocks)) if blocks else (0, 0)
    unprinted, inner = [], []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag in ("delete", "replace"):
            unprinted += [w for w in mine[i1:i2] if not any(_alike(w, o) for o in ocr[j1:j2])]
        if tag in ("insert", "replace") and first <= j1 and j2 <= last:
            inner += [o for o in ocr[j1:j2] if len(o) >= 3 and not any(_alike(w, o) for w in mine[i1:i2])]
    return {"order": round(sum(b.size for b in blocks) / len(mine), 3), "unprinted": unprinted, "inner": inner}


def ability_sure(row: dict) -> bool:
    """A text-layer ability is sure when the composite prints it: RECALL of its words in order, none unprinted,
    nothing else inside it (printed_in_order). A parser-read Ability layer is."""
    if row.get("ability_from") is None:
        return True
    check = row.get("ability_check") or {}
    return (check.get("order") or 0) >= RECALL and not check.get("unprinted") and not check.get("inner")


def fields_of(row: dict) -> dict:
    """A PSD's fields as the chain keeps them: name, ability and types as the PSD has them (method 'psd'; an
    ability from text layers named after their text sure when the composite prints it in that order: ability_sure;
    an item's types are its subtypes: never collected, owner 11-13; 'unknown': review), stats as its composite prints
    them (method 'picture'; every printed row sure). Stars are left out (2022 PSDs keep hidden star layers the
    count takes in: 5 for Golden Freddy's 3). A PSD of the art alone (no ability, stats or types) has none."""
    r = row.get("read") or {}
    bars = row.get("bars") or {}
    printed = {k: v for k, v in (bars.get("stats") or {}).items() if v and v.get("row") != "none"}
    types = [t for t in r.get("types") or []] if r.get("kind") not in ITEM_KINDS else []
    if not r or not ((r.get("ability") or "").strip() or printed or types):
        return {}
    out = {}
    if r.get("name"):
        out["name"] = {"value": r["name"], "method": "psd", "confidence": "sure"}
    if (r.get("ability") or "").strip():
        out["ability"] = {"value": r["ability"], "method": "psd", "confidence": "sure" if ability_sure(row) else
                          "review", "recall": row.get("ability_recall"), "check": row.get("ability_check")}
    if types:
        out["types"] = {"value": sorted(types), "method": "psd", "confidence": "review" if "unknown" in types
                        else "sure"}
    if printed and not bars.get("two") and all(v.get("sure") and v.get("v") is not None for v in printed.values()):
        out["stats"] = {"value": {k: v["v"] for k, v in printed.items()}, "method": "picture",
                        "confidence": "sure", "era": bars.get("era")}
    return out


def main(argv: Optional[list] = None) -> int:
    ap = ArgumentParser(prog="python -m commands.discord_psds", description="The card PSDs posted on Discord.")
    ap.add_argument("step", choices=("parse", "read"))
    ap.add_argument("--root", default=environ.get("DISCORD_ARCHIVE", dm.DEFAULT_ROOT))
    ap.add_argument("--cards-repo", default=dm.LOCAL_DIR_LOC)
    a = ap.parse_args(argv)
    root = dm.check_root(a.root)
    if not exists(join(root, "attachments.jsonl")):
        raise SystemExit(f"no archive at {root}")
    if a.step == "parse":
        parse(root, a.cards_repo)
    else:
        read(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
