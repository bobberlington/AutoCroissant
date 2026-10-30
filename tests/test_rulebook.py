"""The rulebook written on the website (owner 2026-10-29): its chapters as plain text for db/rulebook.json, the TTS pages
written beside pages.json (same pixels: same file), and the save's Rulebooks bag filled from them, the Photoshop pages
moved into a bag of their own and kept."""
import json
import types
from io import BytesIO

from PIL import Image

from commands import rulebook, rulebook_pages, tts_save

CHAPTER = """# Stats {.x}

## HEALTH {.hp}
![The bars](bars.webp){.right width=30}
- Each bar is **2 HP**; see [“ATTACKING”](#/rules/attacking).

::: row 3 1
> [!eggman top] *“Evil Quote!”*

![[Mushroom]] -> ![[Life Shroom|Shroom]]
:::

| ±0% | -25% |
|:---:|:---:|
| 1 | [0.5]{.red} |
{.grid cols=-,red}

=== page ===
1) Uses [[Toy Bonnie|the bunny]] \\[such as (BOGT)\\].
"""


def test_plain_text_and_chapters():
    """Markup goes, words and timing tags stay; files not named NN-slug.md are not chapters; the label numbers them."""
    assert rulebook.plain_text(CHAPTER).split("\n") == [
        "Stats", "", "HEALTH", "", "- Each bar is 2 HP; see “ATTACKING”.", "", "“Evil Quote!”", "", "Mushroom -> Life Shroom", "",
        "±0% | -25%", "1 | 0.5", "", "1) Uses the bunny [such as (BOGT)]."]
    found = rulebook.chapters({"site/rulebook/07-stats.md": CHAPTER, "site/rulebook/README.md": "# no", "site/x/01-a.md": "# no"})
    assert [(c["id"], c["num"], c["title"], c["file"]) for c in found] == [("stats", "07", "Stats", "07-stats.md")]
    assert rulebook.label(found[0]) == "7. Stats" and rulebook.label({"num": "00", "title": "Table of Contents"}) == "Table of Contents"
    text = rulebook.rulebook_text(found)
    assert rulebook.parse_rulebook(text) == found and text.count("\n") == 3          # one chapter per line


def _png(colour) -> bytes:
    out = BytesIO()
    Image.new("RGB", (4, 6), colour).save(out, "PNG")
    return out.getvalue()


def test_pages_written_only_when_their_pixels_change(tmp_path):
    folder = tmp_path / "TTS" / "Rulebook"
    shots = [({"name": "Table of Contents", "num": "", "ch": "00", "n": 1}, _png("white")),
             ({"name": "Stats", "num": "7", "ch": "07", "n": 1}, _png("blue")), ({"name": "Stats 2", "num": "7.2", "ch": "07", "n": 2}, _png("red"))]
    assert rulebook_pages.write_pages(str(folder), shots) == {"new": ["00-01.png", "07-01.png", "07-02.png"], "redrawn": [], "removed": [], "listed": True}
    first = (folder / "07-01.png").stat().st_mtime_ns
    again = [shots[0], (shots[1][0], _png("blue")), (shots[2][0], _png("green"))]
    assert rulebook_pages.write_pages(str(folder), again) == {"new": [], "redrawn": ["07-02.png"], "removed": [], "listed": False}
    assert (folder / "07-01.png").stat().st_mtime_ns == first                       # same pixels: the file (its pin) stays
    assert rulebook_pages.write_pages(str(folder), again[:2])["removed"] == ["07-02.png"]
    assert rulebook_pages.read_pages(str(tmp_path)) == [{"file": "00-01.png", "name": "Table of Contents", "num": ""},
                                                        {"file": "07-01.png", "name": "Stats", "num": "7"}]


def _card(guid, path, nick, deck_id):
    url = f"{tts_save.RAW}abc1234/{path}"
    return {"GUID": guid, "Name": "CardCustom", "Nickname": nick, "Description": "theirs", "CardID": deck_id * 100,
            "CustomDeck": {str(deck_id): {"FaceURL": url, "BackURL": url, "NumWidth": 1, "NumHeight": 1}}, "LuaScript": "x"}


def test_rulebooks_bag_takes_the_pages(tmp_path):
    """The Photoshop pages move into their own bag (once); the pages go in book order, the first page last as before,
    made from a page card with fresh GUIDs and CustomDeck ids; a gone page leaves; a second run changes nothing."""
    pages = tmp_path / "TTS" / "Rulebook"
    pages.mkdir(parents=True)
    listing = [{"file": "00-01.png", "name": "Table of Contents", "num": ""}, {"file": "07-01.png", "name": "Stats", "num": "7"},
               {"file": "07-02.png", "name": "Stats 2", "num": "7.2"}]
    (pages / "pages.json").write_text(json.dumps({"pages": listing}), encoding="utf-8")
    old = [_card("aaaaaa", "Auxiliary/Rulebook/01_Getting_Started.png", "Page 1: Getting Started", 40),
           _card("bbbbbb", "Auxiliary/Rulebook/00_Table_of_Contents.png", "Page 0: Table of Contents", 41)]
    note = {"GUID": "cccccc", "Name": "Notecard", "Nickname": "House rules"}
    bag = {"GUID": "dddddd", "Name": "Custom_Model_Bag", "Nickname": "Rulebooks", "ContainedObjects": [*old, note]}
    save = {"ObjectStates": [bag, _card("eeeeee", "Items/x.png", "X", 90)]}
    repo = types.SimpleNamespace(checkout=str(tmp_path), url=lambda p: f"{tts_save.RAW}main/{p}", sha=lambda p: None)
    report = tts_save._Report()
    tts_save._rulebook(save, repo, report)
    inside = bag["ContainedObjects"]
    assert inside[0]["Nickname"] == tts_save.OLD_RULEBOOK and inside[0]["ContainedObjects"] == old and inside[1] is note
    assert [c["Nickname"] for c in inside[2:]] == ["Page 7.2: Stats 2", "Page 7: Stats", "Page 0: Table of Contents"]
    made = inside[2:]
    assert sorted(c["CardID"] // 100 for c in made) == [91, 92, 93] and all(list(c["CustomDeck"]) == [str(c["CardID"] // 100)] for c in made)
    assert all(c["Description"] == "" and c["LuaScript"] == "" for c in made)
    assert next(iter(made[2]["CustomDeck"].values()))["FaceURL"].endswith("main/TTS/Rulebook/00-01.png")
    guids = [o.get("GUID") for o, _ in tts_save._walk(save["ObjectStates"])]
    assert len(guids) == len(set(guids))
    assert report.unpinned == {f"TTS/Rulebook/{p['file']}" for p in listing}
    again = tts_save._Report()
    tts_save._rulebook(save, repo, again)
    assert again.rulebook == [] and bag["ContainedObjects"] == inside
    (pages / "pages.json").write_text(json.dumps({"pages": listing[:2]}), encoding="utf-8")
    tts_save._rulebook(save, repo, again)
    assert [c["Nickname"] for c in bag["ContainedObjects"][2:]] == ["Page 7: Stats", "Page 0: Table of Contents"]
    assert again.rulebook == ["removed Page 7.2: Stats 2 `TTS/Rulebook/07-02.png` (no longer a page)"]
