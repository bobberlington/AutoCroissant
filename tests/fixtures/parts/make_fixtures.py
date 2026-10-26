"""
The parts editor's frozen contract fixtures (docs/Finished/session-2026-10-26/PARTS_EDITOR.md §2.3, §2.4, §3, §5, §13), from real cards.

Writes, next to this file:
- mini_doomer.json, miracle_matter.json, warcraftian_druid.json, a_snack_tagged.json, getting_started.json: I1, each
  {"blob", "path", "name", "parts"} with "parts" the details record's VERSION 2 key (card_parts(psd, rel, name, parser));
  a_snack_tagged is A_Snack.psd with §9's tags applied in memory (the card repo stays untagged);
- card_edits.parts.json (I2), stats.variants.json (I3), result.v2.json (I4).

Format: json.dumps(obj, indent=1, ensure_ascii=False) + "\\n", keys in the order written here; the parts are
psd_parts.card_parts' (until Track A landed, a stand-in here wrote the same bytes).

    ~/.venvs/ac-py314-fork/bin/python tests/fixtures/parts/make_fixtures.py [--clone ~/Desktop/TTSCardMaker] [--ref origin/main]

Copy the folder to TTSCardMaker site/tests/fixtures/parts/ afterwards.
"""
import sys
from argparse import ArgumentParser
from io import BytesIO
from json import dumps, loads
from os.path import abspath, dirname, expanduser, join
from subprocess import run
from types import SimpleNamespace

ROOT = dirname(dirname(dirname(dirname(abspath(__file__)))))
sys.path.insert(0, ROOT)

from psd_tools import PSDImage  # noqa: E402

from commands import psd_parts  # noqa: E402
from commands.card_models import StatsDatabase  # noqa: E402
from commands.psd_parser import IconVocabulary, PSDParser  # noqa: E402
from commands.psd_parts import Templates  # noqa: E402
from global_config import ICON_MANIFEST_PATH, STARS_DIR, TYPES_DIR  # noqa: E402

HERE = dirname(abspath(__file__))
CARDS = {      # fixture: (path, card name, {lyid: new name} tags applied in memory)
    "mini_doomer": ("Auxiliary/Minions/Mini_Doomer.psd", "Mini Doomer", {}),
    "miracle_matter": ("Creatures/Kirby/3 Stars/Miracle_Matter.psd", "Miracle Matter", {}),
    "warcraftian_druid": ("Creatures/World Of Warcraft/4 Stars/Warcraftian_Druid.psd", "Warcraftian Druid", {}),
    "a_snack_tagged": ("Items/Heal/2 Stars/A_Snack.psd", "A Snack",
                       {1015: "A Snack #1", 327: "Name #1", 1022: "Breakfast #2", 1023: "Name #2", 1024: "Supper #3", 1025: "Name #3"}),
    "getting_started": ("Auxiliary/Rulebook/01_Getting_Started.psd", "01 Getting Started", {}),
}


def git(clone, *args) -> bytes:
    return run(["git", "-C", clone, *args], capture_output=True, check=True).stdout


def write(name: str, obj) -> None:
    with open(join(HERE, name), "w", encoding="utf-8") as fh:
        fh.write(dumps(obj, indent=1, ensure_ascii=False) + "\n")


def card_parts(clone, ref, psd, rel, name, parser, kind, types, subtype) -> dict:
    """psd_parts.card_parts over every picture, the template read at ref."""
    templates = Templates.from_blobs(lambda r: git(clone, "cat-file", "blob", f"{ref}:{r}"))
    return psd_parts.card_parts(psd, rel, name, parser, card=SimpleNamespace(card_type=kind, types=types, subtype=subtype),
                                templates=templates)


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
        parts = card_parts(clone, ref, psd, rel, name, parser, rec["card_type"], rec.get("types"), rec.get("subtype"))
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
    edits = {"about": "The parts editor's Card Edits.json entries (docs/Finished/session-2026-10-26/PARTS_EDITOR.md §3): a fixture.", "cards": {
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
