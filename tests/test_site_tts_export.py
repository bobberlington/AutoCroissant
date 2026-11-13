"""
The deck builder's "Download for TTS" (TTSCardMaker site/assets/deck.js ttsSave, in node) makes each card as the TTS save
holds its own: CardCustom at the size its back is played at (creatures 3, items and field cards 2, x = z), each pile at
its cards' size. Sizes from the save when the checkout has it. Loads the site from $AC_SITE_SRC or the clone.
"""
import json
from collections import Counter
from subprocess import run as run_process

from conftest import SITE_SRC

ROOT = SITE_SRC
DECK_JS = ROOT / "site" / "assets" / "deck.js"
SAVE = ROOT / "TTS" / "Impossibility Simulator.json"


def played_sizes():
    """back file -> the size most of the save's cards with it are played at (a card in a deck: the deck's)."""
    if not SAVE.is_file():
        return {"cardback.png": [3, 1, 3], "cardback_item.png": [2, 1, 2]}
    sizes = {}

    def walk(o, scale):
        t = o.get("Transform") or {}
        scale = scale or [t.get(k, 1) for k in ("scaleX", "scaleY", "scaleZ")]
        if o.get("Name") in ("Card", "CardCustom"):
            for face in (o.get("CustomDeck") or {}).values():
                sizes.setdefault(face["BackURL"].rsplit("/", 1)[-1], Counter())[json.dumps(scale)] += 1
        for k in o.get("ContainedObjects") or []:
            walk(k, scale if o.get("Name") in ("Deck", "DeckCustom") else None)

    for o in json.loads(SAVE.read_text(encoding="utf-8"))["ObjectStates"]:
        walk(o, None)
    return {back: json.loads(c.most_common(1)[0][0]) for back, c in sizes.items()}


def test_cards_are_made_as_the_saves(tmp_path):
    script = tmp_path / "probe.mjs"
    script.write_text(f"""const D = await import({json.dumps(DECK_JS.as_uri())});
const c = (id, type, stars) => ({{ id, name: id, type, stars, png: `X/${{id}}.png`, th: type === "creature" ? 720 : 540, tw: 360 }});
const byId = new Map([c("a", "creature", 5), c("b", "creature", 4), c("i", "item", 4), c("j", "item", 1), c("f", "field", 2)]
  .map(x => [x.id, x]));
const save = D.ttsSave({{ name: "H", cards: ["a", "b", "i", "j"], field: "f" }}, byId, "https://raw/x/main/");
const size = t => [t.scaleX, t.scaleY, t.scaleZ];
process.stdout.write(JSON.stringify(save.ObjectStates.map(o => ({{ name: o.Name, size: size(o.Transform),
  back: Object.values(o.CustomDeck)[0].BackURL.split("/").pop(),
  cards: (o.ContainedObjects || [o]).map(k => [k.Name, size(k.Transform)]) }}))));
""", encoding="utf-8")
    done = run_process(["node", str(script)], capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert done.returncode == 0, done.stderr
    piles = json.loads(done.stdout)
    want = played_sizes()
    assert [(p["name"], p["back"], len(p["cards"])) for p in piles] == [
        ("Deck", "cardback.png", 2), ("Deck", "cardback_item.png", 2), ("CardCustom", "cardback_item.png", 1)]
    for p in piles:
        assert p["size"] == want[p["back"]]
        assert p["cards"] == [["CardCustom", want[p["back"]]]] * len(p["cards"])
