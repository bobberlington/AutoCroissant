"""
The parts editor's frozen contract fixtures (docs/Future/PARTS_EDITOR.md §2.3, §2.4, §3, §5, §13), from real cards.

Writes, next to this file:
- mini_doomer.json, miracle_matter.json, warcraftian_druid.json, a_snack_tagged.json, getting_started.json: I1, each
  {"blob", "path", "name", "parts"} with "parts" the details record's VERSION 2 key (card_parts(psd, rel, name, parser));
  a_snack_tagged is A_Snack.psd with §9's tags applied in memory (the card repo stays untagged);
- card_edits.parts.json (I2), stats.variants.json (I3), result.v2.json (I4).

Format: json.dumps(obj, indent=1, ensure_ascii=False) + "\\n", keys in the order written here. Track A regenerates these
with the real psd_parts.card_parts (the selection below is its stand-in, by the design's rules) and keeps the bytes.

    ~/.venvs/ac-py314-fork/bin/python tests/fixtures/parts/make_fixtures.py [--clone ~/Desktop/TTSCardMaker] [--ref origin/main]

Copy the folder to TTSCardMaker site/tests/fixtures/parts/ afterwards.
"""
import sys
from argparse import ArgumentParser
from hashlib import sha1
from io import BytesIO
from json import dumps, loads
from os.path import abspath, dirname, expanduser, join
from re import compile as re_compile, sub
from subprocess import run

ROOT = dirname(dirname(dirname(dirname(abspath(__file__)))))
sys.path.insert(0, ROOT)

from psd_tools import PSDImage  # noqa: E402

from commands.card_models import CardInfo, StatsDatabase  # noqa: E402
from commands.psd_parser import (EFFECT_LAYER_TAG, TITLE_REGION_RATIO, IconVocabulary, PSDParser,  # noqa: E402
                                 name_key, title_text)
from global_config import ICON_MANIFEST_PATH, STARS_DIR, TYPES_DIR  # noqa: E402

HERE = dirname(abspath(__file__))
TAG = re_compile(r"^(.*\S) #(\d{1,2}(?:,\d{1,2})*)$")
CARDS = {      # fixture: (path, card name, {lyid: new name} tags applied in memory)
    "mini_doomer": ("Auxiliary/Minions/Mini_Doomer.psd", "Mini Doomer", {}),
    "miracle_matter": ("Creatures/Kirby/3 Stars/Miracle_Matter.psd", "Miracle Matter", {}),
    "warcraftian_druid": ("Creatures/World Of Warcraft/4 Stars/Warcraftian_Druid.psd", "Warcraftian Druid", {}),
    "a_snack_tagged": ("Items/Heal/2 Stars/A_Snack.psd", "A Snack",
                       {1015: "A Snack #1", 327: "Name #1", 1022: "Breakfast #2", 1023: "Name #2", 1024: "Supper #3", 1025: "Name #3"}),
    "getting_started": ("Auxiliary/Rulebook/01_Getting_Started.psd", "01 Getting Started", {}),
}
TEMPLATES = {"creature": "Creature_Template", "minion": "Creature_Template", "creatures": "Creature_Template",
             "item": "Item_Template", "aux item": "Item_Template", "nme": "Item_Template", "field": "Item_Template",
             "buff": "Status_Effect_Template", "debuff": "Status_Effect_Template"}


def git(clone, *args) -> bytes:
    return run(["git", "-C", clone, *args], capture_output=True, check=True).stdout


def write(name: str, obj) -> None:
    with open(join(HERE, name), "w", encoding="utf-8") as fh:
        fh.write(dumps(obj, indent=1, ensure_ascii=False) + "\n")


# ---- the design's functions (§2.1), stand-ins until Track A lands them
def layer_tag(name: str):
    """(base, frozenset of variants | None); FX layers are never tagged."""
    m = None if EFFECT_LAYER_TAG in name else TAG.match(name)
    return (m.group(1), frozenset(int(n) for n in m.group(2).split(","))) if m else (name, None)


def base(layer) -> str:
    return layer_tag(layer.name)[0].strip()


def shown_in(layer, n) -> bool:
    """A tagged node shows in its variants (its eye ignored), an untagged one by its eye; every ancestor must show."""
    while layer is not None and getattr(layer, "kind", "psdimage") != "psdimage":
        tags = layer_tag(layer.name)[1]
        if not (n in tags if tags is not None and n is not None else layer.visible):
            return False
        layer = layer.parent
    return True


def variants_of(psd) -> list:
    return sorted({1} | {v for layer in psd.descendants() for v in (layer_tag(layer.name)[1] or ())})


def raw_sha(layer) -> str:
    return sha1(layer.text.encode("utf-8")).hexdigest()[:8]


def layer_path(layer) -> str:
    names = []
    while layer is not None and layer.kind != "psdimage":
        names.append(layer.name)
        layer = layer.parent
    return "/".join(reversed(names))


def layer_text(parser: PSDParser, layer) -> str:
    """One layer's reading: icon-font runs as tokens, breaks as \\n, blank ends and line-leading padding gone."""
    text = layer.engine_dict["Editor"]["Text"].value
    spans = parser._icon_font_spans(layer, text)
    if spans:
        text = parser._read_icon_tokens(text, spans, CardInfo(name="fixture", card_type="unknown"))[0]
    text = text.replace("\r", "\n").replace("\x03", "\n").replace("\t", " ").replace("﻿", "").strip()
    return sub(r"(?m)^[^\S\n]{3,}(?=\S)", "", text)


def reading(layers: list) -> list:
    return [layer for _, layer in sorted_pos([(layer, layer) for layer in layers])]


def sorted_pos(items):
    rows = []
    for item in sorted(items, key=lambda i: i[1].bbox[1]):
        row = next((r for r in rows if abs(item[1].bbox[1] - r[0][1].bbox[1]) <= 40), None)
        (row.append(item) if row is not None else rows.append([item]))
    return [i for row in rows for i in sorted(row, key=lambda i: i[1].bbox[0])]


def title_layers(psd, text_layers, name, n, own_text=True) -> list:
    """PSDParser.title_layers with shown_in(·, n) and tag-stripped names."""
    top = psd.height * TITLE_REGION_RATIO
    text_group = next((layer for layer in psd if layer.is_group() and base(layer) == "Text"), None)
    under = [layer for layer in text_group if shown_in(layer, n) and layer.kind == "type"] if text_group is not None else []
    key = name_key(name)
    found = [layer for layer in under if base(layer) == "Name"]
    if not found:
        found = [layer for layer in text_layers if base(layer) == "Name" and shown_in(layer, n) and layer.bbox[1] < top]
    if not found and key:
        found = [layer for layer in under if name_key(base(layer)) == key]
    if not found and key:
        found = [layer for layer in text_layers if shown_in(layer, n) and layer.bbox[1] < top and name_key(base(layer)) == key]
    if not found and own_text:
        own = [layer for layer in text_layers if shown_in(layer, n) and layer.bbox[1] < top
               and name_key(base(layer)) and name_key(base(layer)) == name_key(layer.text or "")]
        found = own if len(own) == 1 else []
    if len(found) > 1:
        found = [layer for layer in found if layer.bbox[1] < top]
    return found


def split_pieces(psd, found, name, n) -> list:
    """The text layers beside the title layers that print the rest of the name (PSDParser.split_title's)."""
    title = " ".join(title_text(layer.text) for layer in reading(found))
    joined, whole = "".join(title.lower().split()), "".join(name.lower().split())
    if not joined or joined == whole or joined not in whole:
        return []
    top = psd.height * TITLE_REGION_RATIO
    text_group = next((layer for layer in psd if layer.is_group() and base(layer) == "Text"), None)
    shown = [layer for layer in text_group if shown_in(layer, n)] if text_group is not None else []
    rest, pieces = whole.replace(joined, "", 1), []
    for layer in sorted((l for l in shown if l not in found and l.bbox[1] < top),
                        key=lambda l: -len("".join(title_text(l.text if l.kind == "type" else l.name).split()))):
        piece = "".join(title_text(layer.text if layer.kind == "type" else layer.name).lower().split())
        if piece and piece in rest:
            rest = rest.replace(piece, "", 1)
            pieces.append(layer)
    return [layer for layer in pieces if layer.kind == "type"] if not rest else []


def words(text: str) -> str:
    return " ".join((text or "").replace("\x03", " ").replace("﻿", "").split()).casefold()


def part(parser, layer, role=None, named=None) -> dict:
    out = {"id": layer.layer_id, "layer": layer_path(layer)}
    if role is not None:
        out["role"] = role
    if named is not None:
        out["named"] = named
    out.update(text=layer_text(parser, layer), sha=raw_sha(layer), box=list(layer.bbox), variants=None)
    return out


def select(psd, rel, name, parser, n) -> tuple:
    """(title layers, ability layers, block layers, followers [(layer, of layers, how)]) of variant n."""
    texts = [layer for layer in psd.descendants() if layer.kind == "type" and EFFECT_LAYER_TAG not in layer.name]
    if "Rulebook" in rel:
        return [], [], reading([layer for layer in texts if shown_in(layer, n)]), []
    found = title_layers(psd, texts, name, n)
    titles = reading(found + split_pieces(psd, found, name, n))
    follow, kept = [], []
    for layer in titles:
        twin = next((k for k in kept if words(k.text) == words(layer.text)), None)
        (follow.append((layer, [twin], "copy")) if twin is not None else kept.append(layer))
    abilities = reading([layer for layer in texts if base(layer).lower() == "ability" and shown_in(layer, n)])
    parts = kept + abilities
    for layer in texts:
        if layer in parts or any(f[0] is layer for f in follow) or not shown_in(layer, n):
            continue
        twin = next((p for p in parts if words(p.text) == words(layer.text)), None)
        if twin is not None:
            follow.append((layer, [twin], "copy"))
        elif base(layer) == "Name" and kept and name_key(layer.text) == name_key(" ".join(title_text(k.text) for k in kept)):
            follow.append((layer, kept, "watermark"))
    return kept, abilities, [], follow


def blocks_of(parser, layers) -> list:
    digits = [layer for layer in layers if layer_text(parser, layer).isdigit()]
    page = max(digits, key=lambda l: l.bbox[1]) if digits and not any(base(l) == "Page Number" for l in layers) else None
    out = []
    for layer in layers:
        if base(layer) == "Header":
            out.append(part(parser, layer, "header", True))
        elif base(layer) == "Page Number" or layer is page:
            out.append(part(parser, layer, "page", base(layer) == "Page Number"))
        else:
            out.append(part(parser, layer, "block"))
    return out


def template_of(clone, ref, kind, need_ability, need_types, vocab):
    from commands.psd_details import _style, _type_header
    stem = TEMPLATES.get(kind)
    if stem is None:
        return None
    rel = f"MDW/Templates/{stem}.psd"
    psd = PSDImage.open(BytesIO(git(clone, "cat-file", "blob", f"{ref}:{rel}")))
    ability = None
    if need_ability:
        layer = next((l for l in psd.descendants() if l.kind == "type" and base(l).lower() == "ability" and l.is_visible()), None)
        if layer is not None:
            font, size, _ = _style(layer)
            ability = {"box": list(layer.bbox), "font": font, "size": size}
    types = None
    if need_types:
        header = _type_header(psd, rel, stem, vocab)
        types = {"spot": header["row"][0]["box"]} if header and header.get("row") else None
    return {"kind": kind, "file": rel, "ability": ability, "types": types}


def card_parts(clone, ref, psd, rel, name, parser, kind, types) -> dict:
    """The details record's "parts" (§2.3), every variant (variant=None)."""
    variants = variants_of(psd)
    tagged = len(variants) > 1
    views = {n: select(psd, rel, name, parser, n) for n in (variants if tagged else [None])}
    out = {"title": [], "ability": [], "blocks": [], "follow": [], "variants": None, "template": None}
    seen = {}
    for n, (titles, abilities, blocks, follow) in views.items():
        for key, layers in (("title", titles), ("ability", abilities)):
            for layer in layers:
                if layer.layer_id not in seen:
                    seen[layer.layer_id] = part(parser, layer, None, base(layer) == ("Name" if key == "title" else "Ability"))
                    out[key].append(seen[layer.layer_id])
                seen[layer.layer_id].setdefault("_in", set()).add(n)
        if blocks and not out["blocks"]:
            out["blocks"] = blocks_of(parser, blocks)
        for layer, of, how in follow:
            if not any(f["id"] == layer.layer_id for f in out["follow"]):
                out["follow"].append({"id": layer.layer_id, "layer": layer_path(layer), "of": [o.layer_id for o in of],
                                      "how": how, "variants": None})
    for p in seen.values():
        shown = sorted(p.pop("_in"))
        p["variants"] = None if not tagged or shown == variants else shown
    if tagged:
        stem = rel.rsplit("/", 1)[1][:-4]
        out["variants"] = {"count": len(variants), "pngs": {str(n): f"{stem}{'' if n == 1 else n}.png" for n in variants},
                           "tagged": [{"id": l.layer_id, "layer": layer_path(l), "in": sorted(layer_tag(l.name)[1])}
                                      for l in psd.descendants() if layer_tag(l.name)[1] is not None]}
    if "Rulebook" not in rel and (not out["ability"] or not types):
        out["template"] = template_of(clone, ref, kind, not out["ability"], not types, parser.all_types)
    return out


def main() -> int:
    args = ArgumentParser()
    args.add_argument("--clone", default=expanduser("~/Desktop/TTSCardMaker"))
    args.add_argument("--ref", default="origin/main")
    args.add_argument("--data", default="origin/data", help="the data branch (db/stats.json)")
    opts = args.parse_args()
    clone, ref = opts.clone, opts.ref
    tree = git(clone, "ls-tree", "-r", "--name-only", ref, TYPES_DIR + "/").decode().splitlines()
    all_types = [p.rsplit("/", 1)[1][:-4].lower() for p in tree if p.count("/") == 1 and p.lower().endswith(".png")]
    stars = [p.rsplit("/", 1)[1][:-4].lower().replace(" ", "_") for p in tree if p.startswith(STARS_DIR + "/") and p.lower().endswith(".png")]
    manifest = loads(git(clone, "show", f"{ref}:{ICON_MANIFEST_PATH}"))
    db = StatsDatabase()
    parser = PSDParser(all_types, db, IconVocabulary.build(all_types, stars, manifest))
    stats = loads(git(clone, "show", f"{opts.data}:db/stats.json"))["cards"]

    made = {}
    for fixture, (rel, name, tags) in CARDS.items():
        blob = git(clone, "rev-parse", f"{ref}:{rel}").decode().strip()
        psd = PSDImage.open(BytesIO(git(clone, "cat-file", "blob", blob)))
        for layer in psd.descendants():
            if layer.layer_id in tags:
                layer.name = tags[layer.layer_id]
        rec = stats[name]
        parts = card_parts(clone, ref, psd, rel, name, parser, rec["card_type"], rec.get("types"))
        made[fixture] = {"blob": blob, "path": rel, "name": name, "parts": parts}
        write(f"{fixture}.json", made[fixture])

    def edit(fixture, lyid, text=None, delete=False):
        p = next(x for k in ("title", "ability", "blocks") for x in made[fixture]["parts"][k] if x["id"] == lyid)
        field = "name" if any(x["id"] == lyid for x in made[fixture]["parts"]["title"]) else "ability"
        out = {"field": field}
        out.update({"delete": True} if delete else {"text": text})
        out.update(was=p["text"], sha=p["sha"], psd=made[fixture]["blob"][:8], layer=p["layer"])
        return out

    def joined(fixture, changes, key="ability", n=None):
        parts = [p for p in made[fixture]["parts"][key if key != "blocks" else "blocks"]
                 if n is None or p["variants"] is None or n in p["variants"]]
        texts = [changes.get(p["id"], p["text"]) for p in parts]
        texts = [t for t in texts if t is not None]
        return " ".join(" ".join(t.split("\n")) for t in texts) if key == "title" else "\n".join(texts)

    mm = {1368: "Earth: 5/8/6/5 and heal 3 HP", 417: None}
    snack = {320: "Heal 4 HP.\nInflict 2T2 Regen and 2T1 Stuffed.", 1023: "Second\nBreakfast"}
    at, by = "2026-10-26T01:00:00Z", "MichaelJSr"
    gs = made["getting_started"]["parts"]["blocks"]
    edits = {"about": "The parts editor's Card Edits.json entries (docs/Future/PARTS_EDITOR.md §3): a fixture.", "cards": {
        "01 Getting Started": {"ability": "\n".join("HOW TO WIN:" if p["id"] == 15 else p["text"] for p in gs),
                               "parts": {"15": edit("getting_started", 15, "HOW TO WIN:")}, "_by": by, "_at": at},
        "A Snack": {"ability": snack[320], "parts": {"320": edit("a_snack_tagged", 320, snack[320]),
                                                     "1023": edit("a_snack_tagged", 1023, snack[1023])}, "_by": by, "_at": at},
        "A Snack2": {"name": joined("a_snack_tagged", snack, "title", 2), "ability": joined("a_snack_tagged", snack, "ability", 2),
                     "_by": by, "_at": at},
        "A Snack3": {"ability": joined("a_snack_tagged", snack, "ability", 3), "_by": by, "_at": at},
        "Miracle Matter": {"stars": 4, "ability": joined("miracle_matter", mm),
                           "parts": {"1368": edit("miracle_matter", 1368, mm[1368]), "417": edit("miracle_matter", 417, delete=True)},
                           "_by": by, "_at": at},
        "Mini Doomer": {"note": "a database-only edit: no parts", "_by": by, "_at": at,
                        "_psd": {"state": "baked", "psd": made["mini_doomer"]["blob"][:8], "at": "2026-10-26T01:40:00Z",
                                 "reason": "1 layer skipped (changed in Photoshop)", "parts": {"2068": "skipped"},
                                 "skipped": {"2068": {"text": "Has stats equal to the base stats -1 of the creature that summoned it. \n(BOT) Die.",
                                                      "now": made["mini_doomer"]["parts"]["ability"][0]["text"],
                                                      "why": "changed in Photoshop since this edit: redo it on the website if still wanted"}}}},
        "Panda": {"ability": "(1CD) Eat bamboo: heal 2 HP.",
                  "parts": {"+ability": {"field": "ability", "text": "(1CD) Eat bamboo: heal 2 HP.", "was": None, "sha": None, "psd": "0123abcd"}},
                  "_by": by, "_at": at},
        "Warcraftian Druid": {"name": "Warcraftian Druid of the Claw",
                              "parts": {"1034": edit("warcraftian_druid", 1034, "Druid of\nthe Claw")}, "_by": by, "_at": at},
    }}
    write("card_edits.parts.json", edits)

    keep = ("ability", "blob_sha", "card_type", "display_name", "duplicate_of", "name", "parser_version", "path", "problems",
            "series", "stars", "stats", "subtype", "types")
    snack_rec = {k: stats["A Snack"][k] for k in keep}
    variant = lambda n, title: {**snack_rec, "name": f"A Snack{n}", "display_name": title, "parser_version": 3,
                                "variant": n, "variant_of": "A Snack"}
    titles = {p["variants"][0]: p["text"] for p in made["a_snack_tagged"]["parts"]["title"]}
    write("stats.variants.json", {"schema": 1, "cards": {
        "A Snack": {**snack_rec, "display_name": titles[1], "parser_version": 3, "variant": 1, "variants": ["A Snack2", "A Snack3"]},
        "A Snack2": variant(2, titles[2]), "A Snack3": variant(3, titles[3]),
        "Mini Doomer": {**{k: stats["Mini Doomer"][k] for k in keep}, "parser_version": 3, "variant": None}}})

    snack_blob = made["a_snack_tagged"]["blob"]
    write("result.v2.json", {
        "v": 2, "request": "michaeljsr-mh7k2c3d-0a1b", "card": "A Snack2", "variant": 2, "path": "Items/Heal/2 Stars/A_Snack.psd",
        "png": "Items/Heal/2 Stars/A_Snack2.png",
        "base": {"repo": "MichaelJSr/TTSCardMaker", "commit": "c8556d541f6aff67b4f3b2f6462ff91980f79fec", "psd_blob": snack_blob,
                 "png_blob": git(clone, "rev-parse", f"{ref}:Items/Heal/2 Stars/A_Snack2.png").decode().strip()},
        "edit": {"ability": snack[320], "name": "Second Breakfast",
                 "parts": edits["cards"]["A Snack"]["parts"]},
        "state": "baked", "transient": False, "fields": {"name": "baked", "ability": "baked"}, "reason": "",
        "parts": {"320": "baked", "1023": "baked"},
        "changes": [{"field": "ability", "old": "Heal 3 HP.\nInflict 2T2 Regen and 2T1 Stuffed.", "new": snack[320]},
                    {"field": "name", "old": "Breakfast", "new": "Second Breakfast"}],
        "notes": ["the ability layer is shared: changes every variant (A snack, Breakfast, Supper)"], "parity": {}, "region": None,
        "image": "baked", "files": {"webp": "preview.webp", "png": "preview.png", "before": "before.webp"}, "runner": {},
        "run": "", "took_s": 0.0, "at": "2026-10-26T01:05:00Z"})
    print("wrote", ", ".join(sorted(made)), "and the I2-I4 fixtures to", HERE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
