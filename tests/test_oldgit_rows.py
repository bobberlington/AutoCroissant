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
    assert (kept["value"], kept["confidence"]) == (["undread", "machine"], "sure")
    stars = [icon("Pinwheel", ("buff", 40.0), ("food", 41.0)), icon("Pinwheel copy", ("buff", 40.0), ("food", 41.0))]
    assert rows.check_types([], stars, True, names)["confidence"] == "sure"
    unknown = rows.check_types([], [icon("Layer 3", ("debuffstack", 41.0), ("buff", 42.5))], True, names)
    assert (unknown["confidence"], unknown["value"]) == ("review", [])


class FakeData:
    def __init__(self, versions, reads, pictures, shares, ocr):
        self.versions, self.psd, self.rereads, self.pictures, self.shares = versions, reads, {}, pictures, shares
        self.authors, self.png, self._ocr = {"Michael": "MichaelJSr"}, {}, ocr
        self.icons, self.type_names, self.by_commit = {}, set(), {}

    def ocr_of(self, blob):
        return self._ocr.get(blob, {})

    def thumb(self, blob):
        return None


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
    data = FakeData({"Plan Ahead": versions}, reads, pictures, {("g0", "g1"): 0.01, ("g1", "g2"): 0.02}, ocr)
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
