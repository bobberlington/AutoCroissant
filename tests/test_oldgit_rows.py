"""scripts/oldgit_rows.py (by hand): the old repo's version rows of the card timelines."""
import importlib.util
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "oldgit_rows.py"


@pytest.fixture(scope="module")
def rows():
    spec = importlib.util.spec_from_file_location("oldgit_rows", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    words = lambda t: re.findall(r"[a-z0-9]+", (t or "").lower())
    module._W["readers"] = SimpleNamespace(        # the eras readers' text helpers, as small as the rows use them
        labels=lambda lines: {}, words=words, strip_tokens=lambda t: re.sub(r"\[[^\]]*\]", " ", t or ""),
        ability_of=lambda lines, y0, y1=1.0, drop_class=True: " ".join(l[0] for l in lines if y0 <= l[3] <= y1),
        word_score=lambda truth, got: (sum(w in words(got) for w in words(truth)) / len(words(truth))) if words(truth) else None)
    return module


def stats(hp, de, atk, spd):
    return {"hp": hp, "def": de, "atk": atk, "spd": spd}


def test_the_printed_stats_win_and_two_creatures_never_sum(rows):
    """PSD and picture agree: sure. Hidden digit rows (10/10/10/10) under printed bars: the picture's value, to review.
    Two creatures: no value, each half kept apart. Picture only: sure when both readers agree."""
    printed = stats(7, 4, 4, 3)
    agree = rows.check_stats({"stats": printed}, {"best": printed}, False, False)
    assert (agree["value"], agree["confidence"]) == (printed, "sure")
    hidden = rows.check_stats({"stats": stats(10, 10, 10, 10)}, {"best": printed, "template": printed}, False, False)
    assert (hidden["value"], hidden["confidence"]) == (printed, "review")
    two = rows.check_stats({"stats": stats(9, 6, 10, 10)},
                           {"best": stats(5, 4, 6, 5), "halves": {"top": stats(5, 4, 6, 5), "bottom": stats(4, 2, 4, 5)}},
                           True, False)
    assert two["value"] is None and two["confidence"] == "review"
    assert two["creatures"]["picture"] == [stats(5, 4, 6, 5), stats(4, 2, 4, 5)]
    only = rows.check_stats({"stats": stats(-1, -1, -1, -1)}, {"best": printed, "template": printed}, False, False)
    assert (only["value"], only["confidence"], only["check"]) == (printed, "sure", "second reader")
    lone = rows.check_stats(None, {"best": printed, "template": stats(6, 4, 4, 3)}, False, False)
    assert lone["confidence"] == "review"
    three = rows.check_stats({"stats": stats(10, 4, 6, 8)}, {"best": printed, "halves": two["creatures"]["picture"]}, 3, False)
    assert (three["value"], three["creatures"]["count"], three["creatures"]["picture"]) == (None, 3, None)


def test_an_ability_set_in_a_layer_named_after_its_text(rows):
    """The lower text layers are the ability; title, stat labels, the class line, placeholders, words laid over the
    text and layers off the card are not."""
    layer = lambda name, text, bbox, visible=True: SimpleNamespace(name=name, text=text, bbox=bbox, kind="type",
                                                                   is_visible=lambda: visible)
    text = "Attacks inflict Smoldering. (EOT) Gain one extra action."
    psd = SimpleNamespace(width=1200, height=2400, descendants=lambda: [
        layer("Name", "Project P.A.N.D.A", (274, 32, 892, 120)),
        layer("Health", "Health", (70, 1199, 303, 1285)),
        layer(text[:40], text, (13, 1797, 1179, 2231)),
        layer("Condemned", "Smoldering", (500, 1800, 700, 1850)),
        layer("Attribute", "Military", (452, 2268, 755, 2358)),
        layer("Nature/Brute", "Nature/Brute", (453, 2326, 864, 2382)),
        layer("Lorem Ipsum", "Lorem Ipsum", (1411, 1750, 1476, 1762)),
        layer("old", "An old hidden ability", (13, 1797, 1179, 2231), visible=False)])
    assert [l.text for l in rows.ability_layers(psd)] == [text]


def test_types_are_checked_by_their_icons_pictures(rows):
    """An icon layer's picture names its type (a layer the parser read under another name, one it missed); a read
    type no icon layer shows stays; an icon no Types/ picture matches goes to review; a repeated one is a star."""
    names = {"debuff", "debuffstack", "food", "buff", "undread", "undead", "machine"}
    icon = lambda layer, *near: {"layer": layer, "near": [list(n) for n in near]}
    poison = rows.check_types(["debuff"], [icon("debuff", ("debuffstack", 7.2), ("debuff", 37.7))], True, names)
    assert (poison["value"], poison["confidence"], poison["read"]) == (["debuffstack"], "sure", ["debuff"])
    tomato = rows.check_types([], [icon("tomato", ("food", 10.3), ("buff", 33.4))], True, names)
    assert (tomato["value"], tomato["method"]) == (["food"], "icon pictures")
    kept = rows.check_types(["machine", "undread"], [icon("undread", ("undread", 1.7), ("undead", 1.7))], True, names)
    assert (kept["value"], kept["raw"], kept["confidence"]) == (["undead", "machine"], ["undread", "machine"], "sure")
    sword = [icon("sword", ("equip", 1.1), ("fire", 30.0))]
    assert rows.check_types([], sword, True, names | {"equip"}, "minion")["value"] == ["offensive_minion"]
    assert rows.check_types([], sword, True, names | {"equip"}, "item")["value"] == ["equip"]
    stars = [icon("Pinwheel", ("buff", 40.0), ("food", 41.0)), icon("Pinwheel copy", ("buff", 40.0), ("food", 41.0))]
    assert rows.check_types([], stars, True, names)["confidence"] == "sure"
    unknown = rows.check_types([], [icon("Layer 3", ("debuffstack", 41.0), ("buff", 42.5))], True, names)
    assert (unknown["confidence"], unknown["value"]) == ("review", [])


class FakeData:
    def __init__(self, versions, reads, pictures, shares, ocr, drawn=None):
        self.versions, self.psd, self.rereads, self.pictures, self.shares = versions, reads, {}, pictures, shares
        self.authors, self.png, self._ocr = {"Michael": "MichaelJSr"}, {}, ocr
        self.icons, self.type_names, self.by_commit, self.drawn, self.vocab = {}, set(), {}, drawn or {}, set()
        self.eye_bars, self.eye_bars2 = {}, {}

    def ocr_of(self, blob):
        return self._ocr.get(blob, {})

    def thumb(self, blob):
        import numpy as np
        return np.zeros((64, 32, 3), dtype=np.int16)          # every composite = its PNG


def test_stars_are_the_drawn_ones_checked_by_the_folder(rows):
    """The parser takes most kinds' stars from the folder: the PSD's drawn stars check it (a disagreement goes to
    review, the drawn count proposed); one picture under two star folders goes to review; with no drawn stars the
    folder is sure only when it never changes and today's card has it."""
    assert rows.check_stars(5, 5, True, "creature")["confidence"] == "sure"
    sans = rows.check_stars(6, 5, True, "creature")
    assert (sans["value"], sans["confidence"], sans["reads"]) == (5, "review", {"folder": 6, "drawn": 5})
    assert rows.check_stars(5, 5, False, "creature")["confidence"] == "review"
    assert rows.check_stars(None, 0, True, "buff") is None
    row = lambda blob, folder, drawn: {"blob": blob, "path": f"Creatures/X/{folder} Stars/A.png",
                                       "fields": {"stars": rows.check_stars(folder, drawn, True, "creature")}}
    moved = [row("a", 6, 6), row("a", 5, 5)]
    rows.confirm_stars(moved, {"stars": 5})
    assert [r["fields"]["stars"]["confidence"] for r in moved] == ["review", "review"]
    custom = [row("a", 2, 0), row("b", 2, 0)]
    rows.confirm_stars(custom, {"stars": 2})
    assert [r["fields"]["stars"]["confidence"] for r in custom] == ["sure", "sure"]
    later = [row("a", 1, 0)]
    rows.confirm_stars(later, {"stars": 2})
    assert later[0]["fields"]["stars"]["confidence"] == "review"


def test_the_ability_is_checked_both_ways_and_the_name_as_printed(rows):
    """Printed lines of card words the PSD text lacks (a second text layer) go to review; with the composite off the
    PNG, one printed word or digit the text lacks does too; old type tokens take today's names. The name keeps the
    printed separators; a second picture's file number is dropped."""
    vocab = {"green", "attacks", "from", "cannot", "negated", "inflict", "paralysis", "paralyzed", "turns", "creature"}
    line = lambda text, y: [text, 1.0, 0.1, y, 0.8, 0.03]
    text = "Attacks inflict Paralyzed for 2 turns."
    same = [line("Attacks inflict Paralyzed for 2 turns.", 0.75)]
    assert rows.check_ability({"ability": text}, same, "creature", True, vocab)["confidence"] == "sure"
    assert rows.check_ability({"ability": text}, same, "creature", False, vocab)["check"] == "ocr both ways"
    more = same + [line("Green: attacks from green cannot be negated", 0.8)]
    assert rows.check_ability({"ability": text}, more, "creature", True, vocab)["confidence"] == "review"
    reworded = [line("Attacks inflict paralysis for 2 turns.", 0.75)]
    assert rows.check_ability({"ability": text}, reworded, "creature", False, vocab)["confidence"] == "review"
    assert rows.check_ability({"ability": "[undread] +2 DEF"}, [], "field")["value"] == "[undead] +2 DEF"
    assert rows.check_name("WX-78", "wx-", "Wx78", {"WX-78"}, True)["value"] == "WX-78"
    kit = rows.check_name("ALL-PURPOSE MEDICAL KIT", None, "All Purpose Medical Kit2", (), False)
    assert (kit["value"], kit["confidence"]) == ("All-Purpose Medical Kit", "sure")


def test_rows_fold_minor_looks_merge_formatting_and_show_the_old_name(rows):
    """Three versions of a renamed card: the first; a 1% look change whose text differs only in formatting (minor,
    merged); a stat change (shown). Each row: date, mapped author, subject, badge, no link; the old name shown, today's
    names untouched (no aliases)."""
    path = "Creatures/Other/3 Stars/Plan_Ahead"
    versions, reads, pictures, ocr = [], {}, {}, {}
    texts = ["(OPT) Gain 2 actions.", "(OPT) gain 2 actions", "(OPT) gain 2 actions"]
    lines = [stats(5, 4, 6, 5), stats(5, 4, 6, 5), stats(6, 4, 6, 5)]
    for i, (text, line) in enumerate(zip(texts, lines)):
        commit = f"{i}" * 40
        for ext, blob in (("psd", f"p{i}"), ("png", f"g{i}")):
            versions.append({"id": f"{commit[:12]}:{path}.{ext}", "old_id": "Plan Ahead", "commit": commit, "ts": 100 + i,
                             "date": f"2023-05-0{i + 1}T00:00:00-04:00", "author": "Michael", "subject": f"edit {i}",
                             "path": f"{path}.{ext}", "ext": ext, "blob": blob, "kind": "edit", "new_look": True})
        reads[(f"p{i}", f"{path}.psd")] = {"read": {"type": "creature", "name": "Plan Ahead", "printed_title": "Plan Ahead",
                                                    "stars": 3, "types": ["fire"], "stats": line, "ability": text}}
        pictures[f"g{i}"] = {"best": line, "halves": {"apart": 0}}
        ocr[f"g{i}"] = {"title": [["PLAN AHEAD", 1.0]], "lines": [[text, 1.0, 0.1, 0.8, 0.8, 0.05]]}
    data = FakeData({"Plan Ahead": versions}, reads, pictures, {("g0", "g1"): 0.01, ("g1", "g2"): 0.02}, ocr,
                    {f"p{i}": 3 for i in range(3)})
    today = {"names": {"Tomorrow I'll"}, "creatures": False}
    found = rows.card_rows(data, "Tomorrow I'll", ["Plan Ahead"], today)
    assert [r["minor"] for r in found] == [False, True, False]
    assert [r["ability_change"] for r in found] == ["first", "formatting", "same"]
    assert found[2]["data_change"] == ["stats"]
    first = found[0]
    assert (first["author"], first["subject"], first["badge"], first["link"]) == ("MichaelJSr", "edit 0", "old repo", None)
    assert {r["old_name"] for r in found} == {"Plan Ahead"} and today["names"] == {"Tomorrow I'll"}
    assert all(r["fields"][f]["confidence"] == "sure" for r in found for f in ("name", "stars", "stats", "ability"))
    assert first["psd"]["how"] == "same commit"


def test_a_look_folds_only_when_every_field_is_sure_and_the_same(rows):
    """A small look change folds as minor only when every field is sure on both rows and unchanged: a review or
    unread field keeps the row shown (its text 'unknown', a new text run); a multi-creature card's stats compare
    each creature's full reads; one picture is one card."""
    sure = lambda v: {"value": v, "method": "psd", "confidence": "sure"}
    review = lambda v: {"value": v, "method": "psd", "confidence": "review"}
    pair = lambda a, b, blobs=("a", "b"): ({"blob": blobs[0], "fields": a}, {"blob": blobs[1], "fields": b})
    base = {"name": sure("Kit"), "ability": sure("Heal a creature by 3 HP.")}
    assert rows.settled(base, dict(base)) and rows.ability_change(*pair(base, dict(base))) == "same"
    reworded = {**base, "ability": review("Heal a creature by 3 HP.")}       # the PSD's text, the print's unknown
    assert not rows.settled(base, reworded)
    assert rows.ability_change(*pair(base, reworded)) == "unknown"
    assert rows.ability_change(*pair(base, reworded, ("a", "a"))) == "same"
    assert not rows.settled(base, {**base, "ability": sure(None)})
    assert not rows.settled(base, {"name": sure("Kit")})
    two = lambda top: {"value": None, "confidence": "review", "creatures": {
        "count": 2, "psd": None, "picture": [stats(7, 4, top, 6), stats(7, 10, 4, 4)]}}
    assert rows.settled({**base, "stats": two(6)}, {**base, "stats": two(6)})
    assert not rows.settled({**base, "stats": two(6)}, {**base, "stats": two(5)})
    assert rows.changed({"stats": two(6)}, {"stats": two(5)}) == ["stats"]
    unread = {"value": None, "confidence": "review", "creatures": {"count": 2, "psd": None, "picture": [None, None]}}
    assert not rows.settled({**base, "stats": unread}, {**base, "stats": unread})


def test_an_icon_at_a_line_end_the_psd_text_lacks_goes_to_review(rows):
    """'can't affect' ending the text, or a line ending 'per' before a new sentence: an icon not read; today's text is
    proposed when it is the same words with icons. A line running on is fine."""
    line = lambda text, y: [text, 1.0, 0.1, y, 0.8, 0.03]
    status = rows.check_ability({"ability": "2 DMG, can’t affect"}, [line("2 DMG, can't affect", 0.3)], "debuff", True,
                                set(), "2 DMG, can't affect [fire]")
    assert (status["confidence"], status["value"], status["psd"]) == ("review", "2 DMG, can't affect [fire]",
                                                                      "2 DMG, can’t affect")
    other = rows.check_ability({"ability": "-2 DEF\ncan't affect"}, [], "debuff", True, set(), "-1 DEF can't affect [undead]")
    assert (other["confidence"], other["value"]) == ("review", "-2 DEF\ncan't affect")
    assert rows.dangling("-1 Cost per\nTrickster: deals 2 damage") == ["per"]
    assert rows.dangling("Deal 2 damage to\nevery creature.\n(OPT) Heal the") == ["the"]
    assert rows.dangling("Summon an [undead] minion.") == []


def test_a_line_starting_with_punctuation_goes_to_review(rows):
    """A printed word or icon before ':' or ',' the PSD read lost (God Tamer's 'Beast: top stats', The Freezer's
    '[ice], [water]: +2 DEF'): review, even when the words are found elsewhere in the text."""
    line = lambda text, y: [text, 1.0, 0.1, y, 0.8, 0.03]
    tamer = ": top stats,: bottom stats.\nIf the beast is attacked while the tamer is alive"
    seen = [line("Beast: top stats, Tamer: bottom stats.", 0.72),
            line("If the beast is attacked while the tamer is alive", 0.76)]
    got = rows.check_ability({"ability": tamer}, seen, "creature", True, {"beast", "top", "stats", "tamer", "bottom"})
    assert (got["confidence"], got["value"]) == ("review", tamer) and "':'" in got["why"]
    freezer = rows.check_ability({"ability": ", [ice]: +2 DEF [water]"}, [line("+2 DEF", 0.9)], "field")
    assert freezer["confidence"] == "review"
    assert rows.leading("Heal 2 HP\r/ turn") == "/"
    assert rows.leading("(1CD) Deal 2 damage: twice.\n-1 ATK") is None


def test_the_bar_readers_eye_truth_wins(rows):
    """An eye reading of the PNG's bars wins over the readers and the PSD; one the eye couldn't read goes to review."""
    printed = stats(7, 4, 4, 3)
    eye = rows.check_stats({"stats": stats(10, 10, 10, 10)}, {"best": stats(1, 1, 1, 1)}, False, False,
                           eye=printed)
    assert (eye["value"], eye["method"], eye["confidence"]) == (printed, "eye", "sure")
    blind = rows.check_stats({"stats": printed}, {"best": printed}, False, False, eye=None)
    assert blind["confidence"] == "review"


def test_two_creatures_told_by_eye_are_sure_halves(rows):
    """Both halves of a two-creature card read by eye (stats, stats2): sure creatures, a split half its own rows."""
    top, bottom = stats(5, 4, 6, 5), stats(4, 2, 4, 5)
    got = rows.check_stats(None, {}, True, False, eye=top, eye2=bottom)
    assert got["confidence"] == "sure" and got["value"] == {"creatures": [top, bottom]}
    assert rows.half_stats(got, 1, "Twin Princes")["value"] == bottom
    assert rows.half_stats(got, 1, "Twin Princes")["confidence"] == "sure"
    alone = rows.check_stats(None, {"halves": {"top": top, "bottom": bottom}}, True, False)
    assert alone["confidence"] == "review" and rows.half_stats(alone, 0, "Twin Princes")["confidence"] == "review"
