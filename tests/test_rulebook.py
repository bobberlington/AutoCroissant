"""The rulebook written on the website (owner 2026-10-29): its chapters as plain text for db/rulebook.json, its TTS PDF
printed again only when a page's pixels change, and the save's Rulebooks bag holding the PDF."""
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


def test_pdf_printed_only_when_a_page_changes(tmp_path):
    """pages.json keeps each page's pixel hash; the PDF is printed again only when one changes (its bytes carry the print
    time); the bag's cover beside it stays."""
    folder = tmp_path / "TTS" / "Rulebook"
    folder.mkdir(parents=True)
    (folder / "Book_Cover.png").write_bytes(_png("white"))
    printed = []
    pdf = lambda: printed.append(1) or b"%PDF-1.4 fake"
    shots = [({"name": "Cover", "num": ""}, _png("white")), ({"name": "Stats", "num": "7"}, _png("blue")), ({"name": "Stats 2", "num": "7.2"}, _png("red"))]
    done = rulebook_pages.write_pages(str(folder), shots, pdf)
    assert done["printed"] and (folder / "Book_Cover.png").exists() and (folder / "Rulebook.pdf").read_bytes() == b"%PDF-1.4 fake"
    again = [shots[0], (shots[1][0], _png("blue")), shots[2]]                     # same pixels
    assert not rulebook_pages.write_pages(str(folder), again, pdf)["printed"] and len(printed) == 1
    done = rulebook_pages.write_pages(str(folder), [shots[0], shots[1], (shots[2][0], _png("green"))], pdf)
    assert done["printed"] and done["changed"] == ["Stats 2"] and len(printed) == 2
    book = rulebook_pages.read_pages(str(tmp_path))
    assert book["pdf"] == "Rulebook.pdf" and [p["name"] for p in book["pages"]] == ["Cover", "Stats", "Stats 2"]


def _card(guid, path, nick, deck_id):
    url = f"{tts_save.RAW}abc1234/{path}"
    return {"GUID": guid, "Name": "CardCustom", "Nickname": nick, "Description": "theirs", "CardID": deck_id * 100,
            "CustomDeck": {str(deck_id): {"FaceURL": url, "BackURL": url, "NumWidth": 1, "NumHeight": 1}}, "LuaScript": "x",
            "Transform": {"posX": 1.0, "scaleX": 4.0, "scaleY": 1.0, "scaleZ": 4.0}}


def test_rulebooks_bag_takes_the_pdf(tmp_path):
    """One Custom_PDF object is made on top of the Rulebooks bag (linked to the PDF, a fresh GUID), the bag wears
    TTS/Rulebook/Book_Cover.png; a second run changes nothing and keeps the team's changes to the PDF object."""
    folder = tmp_path / "TTS" / "Rulebook"
    folder.mkdir(parents=True)
    (folder / "pages.json").write_text(json.dumps({"pdf": "Rulebook.pdf", "pages": [{"name": "Cover"}, {"name": "Stats"}]}), encoding="utf-8")
    (folder / "Book_Cover.png").write_bytes(_png("black"))
    note = {"GUID": "cccccc", "Name": "Notecard", "Nickname": "House rules"}
    bag = {"GUID": "dddddd", "Name": "Custom_Model_Bag", "Nickname": "Rulebooks", "Locked": False, "Transform": {"posX": 1.0, "scaleX": 5.0},
           "CustomMesh": {"DiffuseURL": f"{tts_save.RAW}abc1234/MDW/Old%20Rules/Rulebook/0000_Cover_Book.png"}, "ContainedObjects": [note]}
    save = {"ObjectStates": [bag, _card("eeeeee", "Items/x.png", "X", 90)]}
    repo = types.SimpleNamespace(checkout=str(tmp_path), url=lambda p: f"{tts_save.RAW}main/{p}", sha=lambda p: None)
    report = tts_save._Report()
    tts_save._rulebook(save, repo, report)
    inside = bag["ContainedObjects"]
    assert len(inside) == 2 and inside[0] is note and inside[1]["Name"] == "Custom_PDF" and inside[1]["Nickname"] == "Rulebook"
    pdf = inside[1]
    assert pdf["CustomPDF"] == {"PDFUrl": f"{tts_save.RAW}main/TTS/Rulebook/Rulebook.pdf", "PDFPassword": "", "PDFPage": 0, "PDFPageOffset": 0}
    assert pdf["Transform"]["scaleX"] == tts_save.PDF_SCALE and pdf["Transform"]["posX"] == 1.0
    assert bag["CustomMesh"]["DiffuseURL"] == f"{tts_save.RAW}main/TTS/Rulebook/Book_Cover.png"
    assert report.unpinned == {"TTS/Rulebook/Rulebook.pdf"}
    guids = [o.get("GUID") for o, _ in tts_save._walk(save["ObjectStates"])]
    assert len(guids) == len(set(guids))
    pdf["Transform"]["scaleX"] = 2.0                       # the team resized it
    pdf["CustomPDF"]["PDFUrl"] = f"{tts_save.RAW}{'1' * 40}/TTS/Rulebook/Rulebook.pdf"   # _pin_links pinned it
    bag["CustomMesh"]["DiffuseURL"] = f"{tts_save.RAW}{'2' * 40}/TTS/Rulebook/Book_Cover.png"
    again = tts_save._Report()
    tts_save._rulebook(save, repo, again)
    assert again.rulebook == [] and bag["ContainedObjects"] == inside and pdf["Transform"]["scaleX"] == 2.0
