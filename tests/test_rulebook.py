"""The rulebook written on the website (owner 2026-10-29): its chapters as plain text for db/rulebook.json, its TTS PDF
printed again only when a page's pixels change, and the save's Rulebooks bag holding the PDF."""
import json
import os
import subprocess
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
    found = rulebook.chapters({"site/rulebook/07-stats.md": CHAPTER, "site/rulebook/README.md": "# no", "site/x/01-a.md": "# no",
                               "site/rulebook/24-font.md": "# Font {.web}\n\nWords.\n"})
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
    time); other files beside it stay; each page's text is kept for the TTS save's search."""
    folder = tmp_path / "TTS" / "Rulebook"
    folder.mkdir(parents=True)
    (folder / "notes.png").write_bytes(_png("white"))
    printed = []
    pdf = lambda: printed.append(1) or b"%PDF-1.4 fake"
    shots = [({"name": "Cover", "num": ""}, _png("white")), ({"name": "Stats", "num": "7", "text": "STATS HP DEF"}, _png("blue")), ({"name": "Stats 2", "num": "7.2"}, _png("red"))]
    done = rulebook_pages.write_pages(str(folder), shots, pdf)
    assert done["printed"] and (folder / "notes.png").exists() and (folder / "Rulebook.pdf").read_bytes() == b"%PDF-1.4 fake"
    again = [shots[0], (shots[1][0], _png("blue")), shots[2]]                     # same pixels
    assert not rulebook_pages.write_pages(str(folder), again, pdf)["printed"] and len(printed) == 1
    done = rulebook_pages.write_pages(str(folder), [shots[0], shots[1], (shots[2][0], _png("green"))], pdf)
    assert done["printed"] and done["changed"] == ["Stats 2"] and len(printed) == 2
    book = rulebook_pages.read_pages(str(tmp_path))
    assert book["pdf"] == "Rulebook.pdf" and [p["name"] for p in book["pages"]] == ["Cover", "Stats", "Stats 2"]
    assert book["pages"][1]["t"] == "STATS HP DEF" and "t" not in book["pages"][0]                   # each page's text: the save's search
    rulebook_pages.write_pages(str(folder), shots, pdf, "k1")                       # draw's key, for the next run's skip
    assert rulebook_pages.read_pages(str(tmp_path))["key"] == "k1" and len(printed) == 3
    assert not rulebook_pages.write_pages(str(folder), shots, pdf, "k2")["printed"]  # a new key, the same pixels, the website behind: the print's key stays
    assert rulebook_pages.read_pages(str(tmp_path))["key"] == "k1" and len(printed) == 3
    assert not rulebook_pages.write_pages(str(folder), shots, pdf, "k2", settled=True)["printed"]   # the website caught up: the new key
    assert rulebook_pages.read_pages(str(tmp_path))["key"] == "k2" and len(printed) == 3
    assert rulebook_pages.key("<sheets>", str(tmp_path)) == rulebook_pages.key("<sheets>", str(tmp_path)) != rulebook_pages.key("<sheet>", str(tmp_path))


def test_site_settled_by_its_build(tmp_path, monkeypatch):
    """The key is kept without a print only when the website's build (the main commit dated cards.json's source) has
    HEAD's rulebook pictures and fonts and MDW/Fonts; a later commit elsewhere doesn't hold it back."""
    def commit(when, rel):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(when)
        env = {**os.environ, "GIT_AUTHOR_DATE": when, "GIT_COMMITTER_DATE": when}
        for args in (["add", "-A"], ["-c", "core.hooksPath=/dev/null", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", rel]):
            subprocess.run(["git", "-C", str(tmp_path), *args], check=True, capture_output=True, env=env)
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    commit("2026-11-08T05:00:00+00:00", "Items/a.png")
    commit("2026-11-08T05:41:00+00:00", "MDW/Fonts/a.ttf")
    commit("2026-11-08T07:00:00+00:00", "Items/b.png")
    built = {}
    monkeypatch.setattr(rulebook_pages, "urlopen", lambda request, timeout: BytesIO(json.dumps(built).encode()))
    for source, branch, settled in (("2026-11-07T21:41:00-08:00", "main", True), ("2026-11-08T07:00:00+00:00", "main", True),
                                    ("2026-11-08T05:00:00+00:00", "main", False), ("2026-11-08T06:00:00+00:00", "main", False),
                                    ("2026-11-08T07:00:00+00:00", "work", False), ("", "main", False)):
        built.update(source=source, branch=branch)
        assert rulebook_pages.site_settled("https://site", str(tmp_path)) is settled, source


def _seen(**cards):
    rules = {"chapters": [{"file": "01-a.md", "source": "# A\n![[Mushroom]] and [[Panda|the panda]]\n"}], "images": {"cover": [1, 2, "s1"]}}
    data = {"icons": {"fire": {"file": "icons/fire.1.webp"}}, "iconVariants": {}, "tagMeanings": {}, "thumbs": {"360": "t1"},
            "cards": [{"id": "mushroom", "name": "Mushroom", "img": "m1"}, {"id": "panda", "name": "Panda", "img": "p1"},
                      {"id": "kirby", "name": "Kirby", "img": "k1", "hp": 3}]}
    for name, fields in cards.items():
        next(c for c in data["cards"] if c["id"] == name).update(fields)
    return {"page": "pg", "index": '<link href="assets/app.js?v=a1"><style>.rb{}</style><meta name="build" content="d1">',
            "rules": json.dumps(rules).encode(), "cards": data}


def test_pre_key_and_the_skip(tmp_path, monkeypatch):
    """The pre-key follows what the print view reads (code, chapters, the named cards' pictures, every card's name), not
    the data stamp or other cards' pictures; kept beside the key, it skips Chrome when PRE_SKIP (shadow: Chrome looks and
    a key change warns; --full looks in any case)."""
    pre = lambda seen: rulebook_pages.pre_key(seen, ["t"], [], "/no/chrome")
    base = pre(_seen())
    same = _seen(kirby={"img": "k2", "hp": 4}, panda={"img": "p2"})            # a picture only linked, another card
    same["index"] = same["index"].replace("d1", "d2")
    assert pre(same) == base
    for seen in (_seen(mushroom={"img": "m2"}), _seen(kirby={"name": "Kirbo"}), dict(_seen(), index='<link href="assets/app.js?v=a2">')):
        assert pre(seen) != base
    assert rulebook_pages.pre_key(_seen(), ["u"], [], "/no/chrome") != base
    (tmp_path / "site" / "rulebook").mkdir(parents=True)
    (tmp_path / "site" / "rulebook" / "01-a.md").write_text("# A\n![[Mushroom]] and [[Panda|the panda]]\n", encoding="utf-8")
    folder = tmp_path / "TTS" / "Rulebook"
    folder.mkdir(parents=True)
    shots = [({"name": "Cover", "num": ""}, _png("white"))]
    rulebook_pages.write_pages(str(folder), shots, lambda: b"%PDF", "k1")
    monkeypatch.setattr(rulebook_pages, "site_inputs", lambda site: _seen())
    monkeypatch.setattr(rulebook_pages, "site_settled", lambda site, checkout, built=None: True)
    drawn = {"key": "k1", "n": 0}

    def draw(site, exe, key_of, known, fonts):
        drawn["n"] += 1
        new = drawn["key"] != known
        return ({"01-a.md": "# A\n![[Mushroom]] and [[Panda|the panda]]\n"}, shots if new else None,
                rulebook_pages.Printer(lambda: b"%PDF", lambda: None), drawn["key"], "pg")
    monkeypatch.setattr(rulebook_pages, "draw", draw)
    run = lambda **kw: rulebook_pages.update_pages(str(tmp_path), exe="/no/chrome", **kw)
    mine = rulebook_pages.pre_key(_seen(), rulebook_pages._trees(str(tmp_path)), [], "/no/chrome")
    assert "pre-key held" not in run() and rulebook_pages.read_pages(str(tmp_path))["pre"] == mine            # stored with the key
    assert "pre-key held" in run() and drawn["n"] == 2                                                         # shadow: Chrome looks
    monkeypatch.setattr(rulebook_pages, "PRE_SKIP", True)
    assert "Chrome not started" in run() and drawn["n"] == 2
    assert "pre-key held" in run(full=True) and drawn["n"] == 3
    monkeypatch.setattr(rulebook_pages, "PRE_SKIP", False)
    drawn["key"] = "k2"                                                                                           # an input pre misses
    assert "::warning::" in run() and rulebook_pages.read_pages(str(tmp_path))["key"] == "k2"


def _card(guid, path, nick, deck_id):
    url = f"{tts_save.RAW}abc1234/{path}"
    return {"GUID": guid, "Name": "CardCustom", "Nickname": nick, "Description": "theirs", "CardID": deck_id * 100,
            "CustomDeck": {str(deck_id): {"FaceURL": url, "BackURL": url, "NumWidth": 1, "NumHeight": 1}}, "LuaScript": "x",
            "Transform": {"posX": 1.0, "scaleX": 4.0, "scaleY": 1.0, "scaleZ": 4.0}}


def test_rulebooks_bag_takes_the_pdf(tmp_path):
    """One Custom_PDF object is made on top of the Rulebooks bag (linked to the PDF, a fresh GUID), the bag wears
    MDW/Assets/Book_Cover.png; a second run changes nothing and keeps the team's changes to the PDF object."""
    folder = tmp_path / "TTS" / "Rulebook"
    folder.mkdir(parents=True)
    (folder / "pages.json").write_text(json.dumps({"pdf": "Rulebook.pdf", "pages": [{"name": "Cover"}, {"name": "Stats"}]}), encoding="utf-8")
    note = {"GUID": "cccccc", "Name": "Notecard", "Nickname": "House rules"}
    bag = {"GUID": "dddddd", "Name": "Custom_Model_Bag", "Nickname": "Rulebooks", "Locked": False, "Transform": {"posX": 1.0, "scaleX": 5.0},
           "CustomMesh": {"DiffuseURL": f"{tts_save.RAW}abc1234/MDW/Old%20Rules/Rulebook/0000_Cover_Book.png"}, "ContainedObjects": [note]}
    save = {"ObjectStates": [bag, _card("eeeeee", "Items/x.png", "X", 90)]}
    repo = types.SimpleNamespace(checkout=str(tmp_path), url=lambda p: f"{tts_save.RAW}main/{p}", sha=lambda p: None,
                                 tree={tts_save.BOOK_COVER: "b"})
    report = tts_save._Report()
    tts_save._rulebook(save, repo, report)
    inside = bag["ContainedObjects"]
    assert len(inside) == 2 and inside[0] is note and inside[1]["Name"] == "Custom_PDF" and inside[1]["Nickname"] == "Rulebook"
    pdf = inside[1]
    assert pdf["CustomPDF"] == {"PDFUrl": f"{tts_save.RAW}main/TTS/Rulebook/Rulebook.pdf", "PDFPassword": "", "PDFPage": 0, "PDFPageOffset": 0}
    assert pdf["Transform"]["scaleX"] == tts_save.PDF_SCALE and pdf["Transform"]["posX"] == 1.0
    assert bag["CustomMesh"]["DiffuseURL"] == f"{tts_save.RAW}main/MDW/Assets/Book_Cover.png"
    assert report.unpinned == {"TTS/Rulebook/Rulebook.pdf"}
    guids = [o.get("GUID") for o, _ in tts_save._walk(save["ObjectStates"])]
    assert len(guids) == len(set(guids))
    pdf["Transform"]["scaleX"] = 2.0                       # the team resized it
    pdf["CustomPDF"]["PDFUrl"] = f"{tts_save.RAW}{'1' * 40}/TTS/Rulebook/Rulebook.pdf"   # _pin_links pinned it
    bag["CustomMesh"]["DiffuseURL"] = f"{tts_save.RAW}{'2' * 40}/MDW/Assets/Book_Cover.png"
    again = tts_save._Report()
    tts_save._rulebook(save, repo, again)
    assert again.rulebook == [] and bag["ContainedObjects"] == inside and pdf["Transform"]["scaleX"] == 2.0
