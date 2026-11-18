"""scripts/oldgit_rows.py (by hand): the old repo's version rows of the card timelines."""
import importlib.util
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "oldgit_rows.py"


_MODULE = {}


def rows_module():
    return _MODULE["rows"]


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
    _MODULE["rows"] = module
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
    era = lambda v, sure: {"era": "D2", "stats": {k: {"v": x, "sure": sure} for k, x in v.items()}}
    pic = {"best": printed, "template": printed}
    only = rows.check_stats({"stats": stats(-1, -1, -1, -1)}, pic, False, False, era=era(printed, True))
    assert (only["value"], only["confidence"], only["check"]) == (printed, "sure", "second reader")
    fooled = rows.check_stats(None, pic, False, False, era=era(printed, False))     # Dogma 2023-04-21's static bars
    assert (fooled["value"], fooled["confidence"]) == (printed, "review")
    assert rows.check_stats(None, pic, False, False, era=era(stats(7, 4, 4, 4), True))["confidence"] == "review"
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
    assert rows.check_types([], stars, True, names, stars=2)["confidence"] == "sure"      # the stars, drawn so
    tomato_icon = [icon("tomato", ("food", 10.3), ("buff", 33.4))]
    assert rows.check_types([], tomato_icon + stars, True, names, stars=5)["confidence"] == "sure"   # art beside
    cubes = rows.check_types([], stars, True, names, "creature", ["unknown", "unknown"], 5)   # Kyle's: no stars
    assert (cubes["confidence"], cubes["value"]) == ("review", [])
    unknown = rows.check_types([], [icon("Layer 3", ("debuffstack", 41.0), ("buff", 42.5))], True, names)
    assert (unknown["confidence"], unknown["value"]) == ("review", [])


def test_one_icon_drawn_twice_and_an_items_art_go_to_review(rows):
    """A type twice that today's card has once: one icon drawn as two layers (Smoldering's stacked debuff arrows);
    twice today too stays (Pandora). An item's read type no header icon shows: art (Paintbrush)."""
    names = {"debuff", "debuffupgrade", "panda", "support", "fire"}
    icon = lambda layer, *near: {"layer": layer, "near": [list(n) for n in near]}
    arrow = lambda layer: icon(layer, ("debuff", 0.8), ("debuffupgrade", 22.5))
    smoldering = rows.check_types(["debuff"], [arrow("debuff"), arrow("debuff copy")], True, names, "debuff",
                                  ["debuffupgrade"])
    assert (smoldering["confidence"], smoldering["value"]) == ("review", ["debuff", "debuff"])
    pandas = [icon("panda", ("panda", 1.0), ("fire", 40.0)), icon("panda copy", ("panda", 1.0), ("fire", 40.0))]
    pandora = rows.check_types(["panda", "panda"], pandas, True, names, "creature", ["panda", "panda"])
    assert pandora["confidence"] == "sure"
    support = [icon("support", ("support", 2.1), ("fire", 52.9))]
    brush = rows.check_types(["support", "fire"], support, True, names, "item", ["support"])
    assert (brush["confidence"], brush["value"]) == ("review", ["support", "fire"])
    assert rows.check_types(["support", "fire"], support, True, names, "creature")["confidence"] == "sure"


class FakeData:
    def __init__(self, versions, reads, pictures, shares, ocr, drawn=None):
        self.versions, self.psd, self.rereads, self.pictures, self.shares = versions, reads, {}, pictures, shares
        self.authors, self.png, self._ocr = {"Michael": "MichaelJSr"}, {}, ocr
        self.icons, self.type_names, self.by_commit, self.drawn, self.vocab = {}, set(), {}, drawn or {}, set()
        self.eye_bars, self.eye_bars2, self.era_reads, self.look_reads = {}, {}, {}, {}
        self.text_raw, self.text_cal, self.header, self.eye_types = {}, {}, {}, {}
        self.folder_firsts = {}

    def text_bar(self, era, name):
        return rows_module().Data.text_bar(self, era, name)

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
    other = {"blob": "c", "path": "Creatures/X/4 Stars/A.png", "fields": {"stars": rows.check_stars(4, 4, False, "creature")}}
    rows.confirm_stars([other], {"stars": 4})     # another look's PSD draws the folder's 4; the picture prints 3 (Nosk)
    assert (other["fields"]["stars"]["value"], other["fields"]["stars"]["confidence"]) == (4, "review")
    off = {"blob": "d", "path": "Creatures/X/2 Stars/A.png", "fields": {"stars": rows.check_stars(2, 1, False, "creature")}}
    rows.confirm_stars([off], {"stars": 2})
    assert off["fields"]["stars"]["confidence"] == "review"


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
    assert rows.check_ability({"ability": "a further 25%% SPD"}, [], "field")["value"] == "a further 25% SPD"
    assert rows.check_name("WX-78", "wx-", "Wx78", {"WX-78"}, True)["value"] == "WX-78"
    kit = rows.check_name("ALL-PURPOSE MEDICAL KIT", None, "All Purpose Medical Kit2", (), False)
    assert (kit["value"], kit["confidence"]) == ("All-Purpose Medical Kit", "sure")
    near = rows.check_name("Himphrey's belly", None, "Hupmphreys Belly", {"Humphrey's Belly"}, False)
    assert near["value"] == "Humphrey's Belly"                    # the nearest name, not the first near one


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
    assert rows.ability_change(*pair(base, {**base, "ability": sure("")})) == "removed"      # a blank box (owner 11-17)
    label = {**base, "ability": sure("Non Protecting Minion")}               # a minion's label alone: no ability
    assert rows.ability_change(*pair(label, {**base, "ability": sure("")})) == "same"
    assert rows.ability_change(*pair(label, {**base, "ability": sure("Protective minion.")})) == "changed"
    assert rows.ability_change(*pair({**base, "ability": sure("Summoned by X.")}, label)) == "changed"
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


def test_first_repo_stats_need_an_eye_era_or_discord_read(rows):
    """A first-repo row with no PSD stats: its two readers alone are not sure (one bias on the old templates); the
    same look's Discord eye read, a calibrated era read or a reader agreeing with a sure Discord read is. HP 0 on a
    creature's picture read: never sure."""
    biased, shown = stats(4, 2, 4, 4), stats(5, 3, 5, 5)
    pic = {"best": biased, "template": biased}
    era = lambda v, sure: {"era": "A", "stats": {k: {"v": x, "sure": sure} for k, x in v.items()}}
    check = lambda **kw: rows.check_stats(None, pic, False, False, first=True, **kw)
    assert check()["confidence"] == "review"
    assert (check(era=era(shown, True))["value"], check(era=era(biased, False))["confidence"]) == (shown, "review")
    assert (check(looks=[("eye", shown)])["value"], check(looks=[("eye", shown)])["method"]) == (shown, "eye")
    assert check(looks=[("read", biased)])["check"] == "the same look's Discord read"
    assert check(looks=[("read", shown)])["confidence"] == "review"
    buffed = stats(5, 3, 6, 5)                    # a one-cell change the hash misses (False Knight 2023-04-04)
    got = check(era=era(buffed, True), looks=[("eye", shown)])
    assert (got["value"], got["method"], got["confidence"]) == (buffed, "picture", "sure")
    part = era(buffed, False)
    part["stats"]["atk"]["sure"] = True
    assert check(era=part, looks=[("eye", shown)])["confidence"] == "review"
    zero = stats(0, 0, 4, 3)
    assert rows.check_stats(None, {"best": zero, "template": zero}, False, False, era=era(zero, True),
                            creature=True)["why"] == "HP 0 on a creature: a reader's miss"


def test_two_creatures_told_by_eye_are_sure_halves(rows):
    """Both halves of a two-creature card read by eye (stats, stats2): sure creatures, a split half its own rows."""
    top, bottom = stats(5, 4, 6, 5), stats(4, 2, 4, 5)
    got = rows.check_stats(None, {}, True, False, eye=top, eye2=bottom)
    assert got["confidence"] == "sure" and got["value"] == {"creatures": [top, bottom]}
    assert rows.half_stats(got, 1, "Twin Princes")["value"] == bottom
    assert rows.half_stats(got, 1, "Twin Princes")["confidence"] == "sure"
    alone = rows.check_stats(None, {"halves": {"top": top, "bottom": bottom}}, True, False)
    assert alone["confidence"] == "review" and rows.half_stats(alone, 0, "Twin Princes")["confidence"] == "review"


def test_a_psd_before_any_png_dates_the_card_and_a_look_of_its_own_is_a_row(rows):
    """S6: a PSD added before the card's first PNG is its first record; its composite, when a look of its own (Tainted
    Lazarus 2023-09-10), is a row (psd_only, its thumb), else it only dates the card."""
    path = "Creatures/Isaac/Lazarus"
    v = lambda ext, blob, ts: {"id": f"{blob}:{path}.{ext}", "old_id": "Lazarus", "commit": blob * 20, "ts": ts,
                               "date": f"2023-09-{ts // 10:02d}T00:00:00-04:00", "author": "Michael", "subject": "s",
                               "path": f"{path}.{ext}", "ext": ext, "blob": blob, "kind": "add", "new_look": True}
    versions = [v("psd", "p0", 100), v("psd", "p1", 200), v("png", "g1", 200)]
    read = {"type": "creature", "name": "Lazarus", "printed_title": "Lazarus", "stars": 3, "types": ["light"],
            "stats": stats(5, 4, 6, 5), "ability": "(OPT) Flip."}
    reads = {(b, f"{path}.psd"): {"read": read} for b in ("p0", "p1")}
    ocr = {"g1": {"title": [["LAZARUS", 1.0]], "lines": [["(OPT) Flip.", 1.0, 0.1, 0.8, 0.8, 0.05]]}}

    class Apart(FakeData):
        def thumb(self, blob):
            import numpy as np
            return np.full((64, 32, 3), 200 if blob == "p0" else 0, dtype=np.int16)
    for kind, blobs in ((FakeData, ["g1"]), (Apart, ["p0", "g1"])):
        data = kind({"Lazarus": versions}, reads, {"g1": {"best": stats(5, 4, 6, 5), "halves": {"apart": 0}}}, {},
                    ocr, {"p0": 3, "p1": 3})
        data.psd_firsts = {}
        found = rows.card_rows(data, "Lazarus", ["Lazarus"], {"names": {"Lazarus"}, "creatures": False})
        assert [r["blob"] for r in found] == blobs and data.psd_firsts["Lazarus"]["commit"] == "p0" * 20
    first = found[0]
    assert first["psd_only"] and first["thumb"] == "oldgit/thumbs/p0.webp"
    assert first["psd"]["how"] == "its own composite"
    assert first["fields"]["stars"]["value"] == 3 and "psd_only" not in found[1]


def test_a_buffs_arrow_on_the_right_is_a_header_icon(rows):
    """Other cards' stars sit right of the title, so their right half is left out; a 2022-23 buff has its arrow there."""
    class Layer:
        def __init__(self, name, bbox):
            self.name, self.bbox, self.kind, self.parent = name, bbox, "pixel", None

        def is_group(self):
            return False

        def is_visible(self):
            return True

    class Psd:
        width = 400

        def descendants(self):
            return [Layer("Layer 39", (334, 0, 398, 57)), Layer("tomato", (10, 5, 70, 60))]
    assert [layer.name for layer in rows.header_icons(Psd())] == ["tomato"]
    assert [layer.name for layer in rows.header_icons(Psd(), right=True)] == ["tomato", "Layer 39"]


def test_a_type_icon_one_row_lower_counts_art_there_not(rows):
    """Nailsage Sly's 2022-23 PSDs draw the tornado under the title (bottom past HEADER, middle above): it counts
    only as a type's picture; art there is left out as before (Layer 14 lies lower still)."""
    class Layer:
        def __init__(self, name, bbox):
            self.name, self.bbox, self.kind, self.parent = name, bbox, "pixel", None

        def is_group(self):
            return False

        def is_visible(self):
            return True

    class Psd:
        width = 1200

        def descendants(self):
            return [Layer("Tornado copy", (110, 270, 244, 411)), Layer("Layer 14", (78, 378, 390, 702))]
    assert [layer.name for layer in rows.header_icons(Psd())] == ["Tornado copy"]
    names = {"tornado", "wind", "fire"}
    tornado = {"layer": "Tornado copy", "low": True, "near": [["tornado", 30.5], ["wind", 40.0]]}
    art = {"layer": "Layer 3", "low": True, "near": [["fire", 27.1], ["wind", 32.7]]}
    assert rows.header_icon_kept(tornado, names) and not rows.header_icon_kept(art, names)
    assert rows.header_icon_kept({**art, "low": False}, names)
    assert rows.check_types([], [tornado], True, names, "creature", ["wind"])["value"] == ["wind"]


def test_an_owner_mapped_psd_alone_is_its_commit_s_psd_or_a_look_of_its_own(rows):
    """Old cards of PSDs alone the owner gave a card (Items/Freeze_Spell.psd, 'sumo outside'): a PSD whose composite
    is the same commit's PNG of the card (its own PSD another look) becomes that row's PSD ('replaces'); one that is
    no PNG's look is a psd_only row of its own; base rows are left alone."""
    v = lambda oid, path, ext, blob, ts: {"id": f"{blob}:{path}.{ext}", "old_id": oid, "commit": f"c{ts}" * 8, "ts": ts,
                                          "date": f"2023-04-{ts // 10:02d}T00:00:00-04:00", "author": "Michael",
                                          "subject": "s", "path": f"{path}.{ext}", "ext": ext, "blob": blob,
                                          "kind": "add", "new_look": True}
    own, mine = "Items/Attack/3 Stars/Spell", "Items/Spell"
    versions = {"Spell": [v("Spell", own, "png", "g1", 100), v("Spell", own, "psd", "s1", 100)],
                "Spell (Items)": [v("Spell (Items)", mine, "psd", "o1", 100), v("Spell (Items)", mine, "psd", "o2", 200)]}
    read = lambda n: {"read": {"type": "item", "name": "Spell", "printed_title": "Spell", "stars": 3, "types": [],
                               "ability": f"Freeze {n}."}}
    reads = {("s1", f"{own}.psd"): read(1), ("o1", f"{mine}.psd"): read(2), ("o2", f"{mine}.psd"): read(3)}
    ocr = {"g1": {"title": [["SPELL", 1.0]], "lines": [["Freeze 2.", 1.0, 0.1, 0.8, 0.8, 0.05]]}}

    class Looks(FakeData):
        def thumb(self, blob):
            import numpy as np
            return np.full((64, 32, 3), {"s1": 90, "o2": 200}.get(blob, 0), dtype=np.int16)
    data = Looks(versions, reads, {}, {}, ocr, {"s1": 3, "o1": 3, "o2": 3})
    data.psd_firsts, data.map = {}, {"cards": {}}
    today = {"names": {"Spell"}, "type": "item"}
    base = rows.card_rows(data, "Spell", ["Spell"], today)
    assert [r["psd"]["blob"] for r in base] == ["s1"] and base[0]["fields"]["ability"]["confidence"] != "sure"
    assert rows.psd_alone(data, "Spell (Items)") and not rows.psd_alone(data, "Spell")
    found = rows.psd_rows(data, "Spell", "Spell (Items)", today, ["Spell"], base)
    assert [(r["blob"], r["psd"]["blob"], r.get("replaces"), bool(r.get("psd_only"))) for r in found] == \
        [("g1", "o1", base[0]["id"], False), ("o2", "o2", None, True)]
    assert found[0]["psd_file"] == f"{mine}.psd" and found[0]["psd"]["how"] == "same commit, the owner's PSD"
    assert found[0]["fields"]["ability"]["value"] == "Freeze 2." and found[0]["fields"]["ability"]["confidence"] == "sure"
    assert found[1]["thumb"] == "oldgit/thumbs/o2.webp" and data.psd_firsts == {}


def test_a_png_with_no_psd_text_takes_the_text_readers_and_its_header(rows):
    """No PSD (the first repo's 2022 files): the ability is the text readers' read, sure when its words are one of
    the card's own texts (today's, its PSDs': 'ocr+known') or at its era's stored bar, else proposed; the types are
    a git version's when the header matches it (header_types rows), proposed when under the bar."""
    path = "Cards/Kirby/3 Stars/Pix"
    v = lambda ext, blob, ts: {"id": f"{blob}:{path}.{ext}", "old_id": "Pix", "commit": blob * 20, "ts": ts,
                               "date": "2022-08-24T00:00:00-07:00", "author": "Michael", "subject": "s",
                               "path": f"{path}.{ext}", "ext": ext, "blob": blob, "kind": "add", "new_look": True,
                               "repo": "first"}
    versions = {"Pix": [v("png", "g1", 100), v("png", "g2", 200)]}
    ocr = {b: {"title": [["PIX", 1.0]], "lines": [["Deal 3 damage.", 1.0, 0.1, 0.8, 0.8, 0.05]]} for b in ("g1", "g2")}
    data = FakeData(versions, {}, {}, {}, ocr)
    data.psd_firsts = {}
    read = lambda text, conf: {"era": "D2", "src": "old", "id": "o:x", "ability_fix": {"text": text, "minconf": conf,
                                                                                      "odd": 0}, "ability_raw": {"text": text}}
    data.text_raw = {"g1": read("Deal 3 damage", 1.0), "g2": read("Deal 4 damoge.", 0.5)}
    data.text_cal = {"D2/old": {"ability": {"how": {"variant": "fix"}, "bar": 1.0}}}
    data.header = {"g1": {"types": ["light"], "sure": True, "version": "bv", "ncc": 0.97},
                   "g2": {"types": ["dark"], "sure": False, "version": "bw", "ncc": 0.5}}
    got = rows.card_rows(data, "Pix", ["Pix"], {"names": {"Pix"}, "type": "creature", "ability": "Deal 3 damage."})
    a, b = (r["fields"] for r in got)
    assert (a["ability"]["value"], a["ability"]["method"], a["ability"]["confidence"]) == ("Deal 3 damage.", "ocr+known",
                                                                                            "sure")
    assert (b["ability"]["value"], b["ability"]["confidence"]) == ("Deal 4 damoge.", "review")
    assert (a["types"]["value"], a["types"]["method"], a["types"]["confidence"]) == (["light"], "git header", "sure")
    assert b["types"]["confidence"] == "review"
    data.text_raw["g1"] = read("Deal +3 damage.", 1.0)                           # its words, printed otherwise
    got = rows.card_rows(data, "Pix", ["Pix"], {"names": {"Pix"}, "type": "creature", "ability": "Deal 3 damage."})
    assert (got[0]["fields"]["ability"]["value"], got[0]["fields"]["ability"]["confidence"]) == ("Deal +3 damage.", "review")
    data.eye_types = {"g2": {"types": ["dark", "fire"]}}                           # its header by eye: over the reader
    got = rows.card_rows(data, "Pix", ["Pix"], {"names": {"Pix"}, "type": "creature", "ability": "Deal 3 damage."})
    t = got[1]["fields"]["types"]
    assert (t["value"], t["method"], t["confidence"]) == (["dark", "fire"], "eye", "sure")


def test_types_by_eye_win_over_a_psd_of_another_look(rows):
    """A PNG whose PSD is another look (types to review) takes its types by eye when told (Mini Bees 2022: the
    tornado only, the PSD's minion sword not printed); a sure PSD read stands."""
    path = "Auxillary/Minions/3 Stars/Mini_Bees"
    v = lambda ext, blob: {"id": f"{blob}:{path}.{ext}", "old_id": "Mini_Bees", "commit": "c" * 40, "ts": 100,
                           "date": "2022-08-24T00:00:00-07:00", "author": "Michael", "subject": "s",
                           "path": f"{path}.{ext}", "ext": ext, "blob": blob, "kind": "add", "new_look": True}
    read = {"type": "minion", "name": "Mini Bees", "printed_title": "Mini Bees", "stars": 3,
            "types": ["offensive_minion", "wind"], "stats": stats(1, 2, 3, 6), "ability": "Non-Protecting Minion"}
    ocr = {"g1": {"title": [["Mini Bees", 1.0]], "lines": [["Non-Protecting Minion", 1.0, 0.1, 0.8, 0.8, 0.05]]}}

    class Apart(FakeData):
        def thumb(self, blob):
            import numpy as np
            return np.full((64, 32, 3), 200 if blob == "p1" else 0, dtype=np.int16)
    today = {"names": {"Mini Bee"}, "type": "minion", "creatures": False}
    for kind, sure in ((Apart, False), (FakeData, True)):
        data = kind({"Mini_Bees": [v("psd", "p1"), v("png", "g1")]}, {("p1", f"{path}.psd"): {"read": read}},
                    {"g1": {"best": stats(1, 2, 3, 6), "halves": {"apart": 0}}}, {}, ocr, {"p1": 3})
        data.psd_firsts = {}
        t = rows.card_rows(data, "Mini_Bees", ["Mini_Bees"], today)[-1]["fields"]["types"]
        assert t["confidence"] == ("sure" if sure else "review")
        data.eye_types = {"g1": {"types": ["wind"]}}
        t = rows.card_rows(data, "Mini_Bees", ["Mini_Bees"], today)[-1]["fields"]["types"]
        assert (t["value"], t["method"]) == ((["offensive_minion", "wind"], "psd") if sure else (["wind"], "eye"))


def test_a_file_name_standing_in_for_an_unread_title_is_no_sure_name(rows):
    """'composite, file name' makes a name sure only as the PSD's title: when the same file's other versions print a
    longer title holding it (Satou: 'Satou Matsuzaka' on two lines), the title layers were not read -> review."""
    sure = lambda v, check: {"value": v, "method": "psd", "confidence": "sure", "check": check}
    got = [{"path": "C/Satou.png", "fields": {"name": rows.field("Satou Matsuzaka", "psd", False, why="x")}},
           {"path": "C/Satou.png", "fields": {"name": sure("Satou", "composite, file name")}},
           {"path": "C/Other.png", "fields": {"name": sure("Other", "composite, file name")}}]
    rows.title_stand_in(got)
    assert (got[1]["fields"]["name"]["value"], got[1]["fields"]["name"]["confidence"]) == ("Satou Matsuzaka", "review")
    assert got[2]["fields"]["name"]["confidence"] == "sure"


def test_mdw_drafts_count_only_before_the_card_folders(rows):
    """s3 gap round 2: an MDW draft (oldgit_index MDW_DRAFTS) is a version only before the card's first card-folder
    record; an MDW PSD with no PNG in its commit is a psd_only row of its own composite."""
    v = lambda ext, blob, ts, path, mdw=False: {"id": f"{blob}:{path}.{ext}", "old_id": "t" if mdw else "T",
                                                "commit": blob * 20, "ts": ts, "date": f"2023-09-0{ts // 100}T00:00:00Z",
                                                "author": "Michael", "subject": "s", "path": f"{path}.{ext}", "ext": ext,
                                                "blob": blob, "kind": "add", "new_look": True, **({"mdw": True} if mdw else {})}
    draft, card = "MDW/Croissant's Special Folder/t", "Creatures/Isaac/3 Stars/T"
    kept = rows.before_folders([v("psd", "m0", 100, draft, True), v("psd", "p0", 300, card), v("png", "g0", 300, card),
                                v("psd", "m1", 300, draft, True), v("psd", "m2", 400, draft, True)])
    assert [x["blob"] for x in kept] == ["m0", "p0", "g0"]
    assert rows.folder_first(kept)["ts"] == 300 and rows.folder_first(kept[1:]) is None   # Discord drafts up to it
    meme = [{**v("png", "j0", 100, "Cards/Meme/4 Stars/T"), "old_id": "j"}, v("png", "g1", 300, card)]   # owner 11-17:
    assert [x["blob"] for x in rows.before_folders(meme)] == ["j0", "g1"]                    # Tyler's Mao card
    assert rows.folder_first(meme, {"j"})["ts"] == 300 and rows.folder_first(meme) is None    # only the owner's copy
    data = SimpleNamespace(map={"cards": {"j": {"how": ["name", "owner"], "paths": ["Cards/Meme/4 Stars/T.png"]},
                                          "A": {"how": ["path"], "paths": ["Cards/Meme/5 Stars/A.png", "Cards/Other/A.png"]}}})
    assert rows.joke_copy(data, "j") and not rows.joke_copy(data, "A") and not rows.joke_copy(data, "x")
    read = {"type": "creature", "name": "T", "printed_title": "T", "stars": 3, "types": [], "stats": stats(5, 4, 6, 5),
            "ability": "(OPT) Flip."}
    reads = {("m0", f"{draft}.psd"): {"read": read}, ("p0", f"{card}.psd"): {"read": read}}

    class Apart(FakeData):
        def thumb(self, blob):
            import numpy as np
            return np.full((64, 32, 3), 200 if blob == "m0" else 0, dtype=np.int16)
    data = Apart({"T": kept[1:], "t": kept[:1]}, reads, {"g0": {"best": stats(5, 4, 6, 5), "halves": {"apart": 0}}},
                 {}, {"g0": {"title": [["T", 1.0]], "lines": []}}, {"m0": 3, "p0": 3})
    data.psd_firsts = {}
    found = rows.card_rows(data, "T", ["T", "t"], {"names": {"T"}, "creatures": False})
    assert [(r["blob"], bool(r.get("psd_only")), bool(r.get("mdw"))) for r in found] == [("m0", True, True),
                                                                                         ("g0", False, False)]
    assert data.folder_firsts["T"]["ts"] == 300


def test_the_eras_readers_come_with_the_tools(rows, monkeypatch):
    """S10 (11-17): the bar and text readers the rows use are scripts/era_readers.py, not the stage-1 scratch copy;
    OLDROWS_READERS naming a folder with no readers.py stops with a clear error."""
    monkeypatch.setattr(rows, "READERS", "")
    got = rows.readers()
    assert got.__name__ == "era_readers" and Path(got.__file__).resolve() == SCRIPT.with_name("era_readers.py")
    assert all(callable(getattr(got, f)) for f in ("labels", "read_best", "split_read", "tell_flats", "ability_of"))
    monkeypatch.setattr(rows, "READERS", "/nonexistent/eras")
    with pytest.raises(SystemExit, match="no readers.py"):
        rows.readers()
