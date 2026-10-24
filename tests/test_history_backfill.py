"""history_backfill: the misread rule (lines, print runs, segments), the pre-2025 template and hidden drawn rows, the
cards that left (fates, qualified keys, authors), the once-merge into old_stats, and the committed file's worked
examples. Dead code with the module once its merge has run in production."""
import json
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import pytest

import commands.card_data as cd
import commands.card_db as cdb
import commands.card_history as ch
import commands.card_models as cm
import commands.history_backfill as hb
from test_card_history import blob, commit, git, write
from test_update_stats import github, psds  # noqa: F401 - the fake GitHub's fixtures

REPO = Path(__file__).resolve().parents[1]
PATH = "Creatures/Other/3 Stars/A.psd"
B1, B2, B3, B4 = "1" * 40, "2" * 40, "3" * 40, "4" * 40


# ========================
# Helpers
# ========================
def reading(at, blob_, line, print_="same", **kw):
    return {"at": at, "commit": f"c{int(at)}", "blob": blob_, "path": PATH, "read": line, "how": "parser",
            "print": print_, **kw}


def entry(readings, allowed=False):
    return {"paths": [PATH], "allowed_high": allowed, "readings": readings, "segments": hb.segments(readings, allowed)}


def version(at, blob_, line, ability="Hits.", cause="psd", path=PATH, **kw):
    card = cm.CardInfo(name="A", card_type="creature", path=path, timestamp=at, ability=ability, stars=3,
                       series="other", stats=cm.CardStats(*line), blob_sha=blob_, **kw)
    card.version = {**ch.meta(None, at, cause), "from_commit": f"c{int(at)}"}
    return card


def data(cards=None, archive=None, authors=None):
    return {"id": hb.FILE_ID, "made": {"readings": 3, "cards": 1}, "cards": cards or {}, "changes": [],
            "archive": {"cards": archive or {}, "authors": authors or {}}}


def lines(old, name="A"):
    return [(ch.start(v)[0], hb._line(v)) for v in old.get(name, [])]


def restored():
    """A database the recovery ran on (a restored version somewhere)."""
    mark = version(10.0, B1, [1, 1, 1, 1])
    mark.name = "Z"
    mark.version["restored_from"] = "abc1234"
    return mark


class Layer:
    """A psd-tools layer stand-in: name, kind, visibility, children."""
    def __init__(self, name, kind="group", visible=True, children=(), bbox=(0, 0, 10, 10)):
        self.name, self.kind, self._visible, self.bbox = name, kind, visible, bbox
        self.children, self.parent = list(children), None
        for child in self.children:
            child.parent = self

    def is_group(self):
        return self.kind == "group"

    def is_visible(self):
        return self._visible

    def __iter__(self):
        return iter(self.children)

    def __len__(self):
        return len(self.children)

    def descendants(self):
        for child in self.children:
            yield child
            yield from child.descendants()


# ========================
# Lines and segments
# ========================
@pytest.mark.parametrize("read,extra,allowed,want", [
    ([5, 2, 5, 5], {}, False, "full"),
    ([5, 2, -1, -1], {}, False, "partial"),
    ([10, 10, 10, 10], {}, False, None),                # every digit hidden: the default, no card prints it
    ([-1, -1, -1, -1], {}, False, None),
    ([7, 2, 3, 20], {}, False, "partial"),              # out of range without an allowance: that stat unread
    ([7, 2, 3, 20], {}, True, "full"),                  # allow_high_stats (Sonic)
    ([110, 110, 110, 110], {}, True, None),             # three digits: never
    ([10, 3, 5, 5], {"hidden": ["hp"]}, False, "partial"),      # a drawn-cell row all hidden: that stat unread
    ([10, 3, 5, 5], {"hidden": ["hp", "def", "atk", "spd"]}, False, None),
    ([0, -1, -1, -1], {}, False, None),                 # no value above 0 (Tainted Keeper's first blob)
    (None, {"err": "classifier: no star count"}, False, None),
    ([0, 0, 0, 0], {}, False, None),                    # drawn hearts read as nothing (Clot Bone)
    ([6, 9, 9, 10], {"split": ["hp", "atk"]}, False, None),     # two creatures' rows summed (God Tamer)
    ([2, 0, 5, 6], {}, False, "full"),                  # a 0 among values is a value (Clot White)
])
def test_status(read, extra, allowed, want):
    assert hb.status({"read": read, **extra}, allowed) == want


def test_stat_values_unread_per_stat():
    assert hb.stat_values({"read": [17, 3, 6, 7]}, False) == [None, 3, 6, 7]
    assert hb.stat_values({"read": [10, 4, 6, 5], "hidden": ["hp"]}, False) == [None, 4, 6, 5]
    assert hb.stat_values({"read": [5, 2, 5, 5], "split": ["hp"]}, False) is None


def test_an_unread_stat_keeps_the_others_read():
    """Michael: HP read 17 (a layering fixed later) on both blobs, ATK 6 -> 4 on a row that printed it: a nerf, HP from
    the reading after. Tainted Magdalene: HP hidden, DEF/ATK 4/6 -> 3/5."""
    readings = [reading(1, B1, [17, 3, 6, 7], "first"), reading(2, B2, [17, 3, 4, 7], "changed", row_px=[0, 0, 11656, 0]),
                reading(3, B3, [7, 3, 4, 7], "changed", row_px=[5000, 0, 0, 0])]
    segs = hb.segments(readings, False)
    assert [(s["from"], s["line"], s.get("change")) for s in segs] == [(1, [7, 3, 6, 7], None), (2, [7, 3, 4, 7], "nerf")]
    readings = [reading(1, B1, [10, 4, 6, 5], "first", hidden=["hp"]),
                reading(2, B2, [7, 3, 5, 5], "changed", row_px=[4000, 4000, 4000, 0])]
    assert [s["line"] for s in hb.segments(readings, False)] == [[7, 4, 6, 5], [7, 3, 5, 5]]


def test_a_stat_no_version_read_is_no_line():
    readings = [reading(1, B1, [17, 3, 6, 7], "first"), reading(2, B2, [18, 3, 6, 7], "changed")]
    assert hb.lines_of(readings, False) == [] and hb.segments(readings, False) == []


def test_a_reading_of_zeros_and_minus_ones_is_unread():
    """Tainted Keeper's first blob reads 0/-1/-1/-1 (its preview prints 5/4/5/5): no version of its own."""
    readings = [reading(1, B1, [0, -1, -1, -1], "first"), reading(2, B2, [5, 4, 5, 5], "changed")]
    assert [(s["from"], s["line"]) for s in hb.segments(readings, False)] == [(1, [5, 4, 5, 5])]


def test_bars_that_printed_the_same_are_one_line_the_newest():
    """A reading that changed while the bars printed the same is a misread (Tainted Isaac's layer rename): one
    segment, whose line is the run's newest reading."""
    segs = hb.segments([reading(1, B1, [5, 3, 5, 5], "first"), reading(2, B2, [5, 3, 5, 5]),
                        reading(3, B3, [6, 3, 5, 5])], False)
    assert [(s["from"], s["line"]) for s in segs] == [(1, [6, 3, 5, 5])] and "change" not in segs[0]


def test_changed_bars_make_a_change():
    segs = hb.segments([reading(1, B1, [7, 2, 3, 20], "first"), reading(2, B2, [7, 4, 3, 20], "changed", px=12168),
                        reading(3, B3, [7, 4, 4, 20], "changed", px=6084), reading(4, B4, [7, 4, 4, 20])], True)
    assert [(s["from"], s["line"], s.get("change")) for s in segs] == [
        (1, [7, 2, 3, 20], None), (2, [7, 4, 3, 20], "buff"), (3, [7, 4, 4, 20], "buff")]


def test_an_unread_run_takes_the_line_before_it_the_first_the_one_after():
    segs = hb.segments([reading(1, B1, [-1, -1, -1, -1], "first"), reading(2, B2, [1, 1, 1, 1], "changed"),
                        reading(3, B3, [10, 10, 10, 10], "changed"), reading(4, B4, [1, 1, 1, 1], "changed")], False)
    assert [(s["from"], s["line"]) for s in segs] == [(1, [1, 1, 1, 1])]


def test_a_stat_changes_only_where_its_own_row_printed_a_change():
    """Tainted Eden: 2023-11-17 changed heart outlines in the HP/DEF rows and read ATK 3, SPD 7 on rows that printed
    the same (a misread: they keep 4 and 4); 2023-11-24 changed HP and ATK; 7d4bce24 later read SPD 4 again."""
    readings = [reading(1, B1, [7, 3, 4, 4], "first"),
                reading(2, B2, [7, 3, 3, 7], "changed", px=9792, row_px=[4000, 5000, 18, 18]),
                reading(3, B3, [6, 3, 6, 7], "changed", px=5220, row_px=[1908, 0, 3312, 0]),
                reading(4, B4, [6, 3, 6, 4], "same", px=0, row_px=[0, 0, 0, 0])]
    segs = hb.segments(readings, False)
    assert [(s["from"], s["line"], s.get("change")) for s in segs] == [(1, [7, 3, 4, 4], None), (3, [6, 3, 6, 4], "buff")]


def test_a_row_change_splits_a_run_the_whole_region_calls_the_same():
    """A one-cell drawn-heart edit (about 1,030 px, under the region's PX_SAME) still ends the run."""
    readings = [reading(1, B1, [2, 1, 6, 5], "first"), reading(2, B2, [2, 1, 5, 5], "changed", px=900, row_px=[0, 0, 900, 0])]
    hb_pair = {"px": 900, "rows": [0, 0, 900, 0]}
    assert hb._print_of({"pair": {(B1, B2): hb_pair}}, {"blob": B1}, {"blob": B2}) == ("changed", 900)
    assert [s["line"] for s in hb.segments(readings, False)] == [[2, 1, 6, 5], [2, 1, 5, 5]]


def test_the_same_bytes_under_a_new_name_print_the_same():
    assert hb._print_of({"pair": {}}, {"blob": B1}, {"blob": B1}) == ("same", 0)


def test_print_edits_lists_rows_that_changed_with_no_reading():
    """Chef Saltbaker: the ATK row printed a change the parser reads as nothing; a restyle of all four rows is not listed."""
    readings = [reading(1000, B1, [8, 4, 6, 6], "first"),
                reading(2000, B2, [8, 4, 6, 6], "changed", px=7547, row_px=[0, 0, 7547, 0]),
                reading(3000, B3, [8, 4, 6, 6], "changed", px=99999, row_px=[9000, 9000, 9000, 9000])]
    got = hb.print_edits("Chef", readings, hb.lines_of(readings, False))
    assert [(e[1], e[3], e[4]) for e in got] == [("Chef", B2[:8], {"atk": 7547})]


@pytest.mark.parametrize("a,b,want", [([5, 5, 5, 5], [6, 5, 5, 5], "buff"), ([6, 5, 5, 5], [5, 5, 5, 5], "nerf"),
                                      ([6, 5, 5, 5], [5, 6, 5, 5], "shift"), ([5, 5, -1, -1], [5, 6, -1, -1], "partial"),
                                      ([20, 20, 2, 2], [20, 20, 3, 2], "off-scale")])
def test_change_kinds(a, b, want):
    assert hb.kind_of(a, b) == want


def test_bars_region_never_the_canvas_and_a_size_change_is_no_comparison():
    np = pytest.importorskip("numpy")
    a = np.zeros((40, 40, 3), dtype=np.int16)
    b = a.copy()
    b[0:5, 0:5] = 255                                  # art outside the bars
    assert hb.bars_moved(a, b, [(20, 20, 30, 30)]) == 0
    b[22:24, 22:24] = 255
    assert hb.bars_moved(a, b, [(20, 20, 30, 30)]) == 4
    assert hb.bars_moved(a, b, []) == 0                # no bars: nothing printed to compare
    assert hb.bars_moved(a, np.zeros((10, 10, 3), dtype=np.int16), [(0, 0, 5, 5)]) is None


def test_bar_boxes_are_the_stat_groups_never_the_canvas():
    hp = Layer("Hp Dark", children=[Layer("5", "adjustment", bbox=(10, 10, 20, 20))], bbox=(5, 5, 30, 30))
    psd = Layer("psd", children=[Layer("Darks", children=[hp], bbox=(0, 0, 40, 40)), Layer("Art", "pixel", bbox=(0, 0, 999, 999))])
    assert hb.bar_boxes(psd) == [(0, 0, 40, 40), (5, 5, 30, 30), (10, 10, 20, 20)]


def test_a_missing_preview_renders_the_region(monkeypatch):
    """A version without a stored preview is compared by rendering the bars' region from its layers."""
    np = pytest.importorskip("numpy")
    drawn = []

    class Psd:
        height = width = 40

    monkeypatch.setattr(hb, "_open", lambda repo, sha: Psd())
    monkeypatch.setattr(hb, "read_psd", lambda parser, psd, path: {"read": [1, 1, 1, 1], "how": "parser", "hidden": [],
                                                                   "boxes": [[2, 2, 12, 12]],
                                                                   "rows": [[[2, 2, 12, 4]], [], [], []]})
    monkeypatch.setattr(hb, "_preview", lambda psd: None)

    def render(psd, box):
        drawn.append(box)
        img = np.zeros((10, 10, 3), dtype=np.int16)
        img[0:2, 0:2] = 255 * (len(drawn) - 1)                 # the second version: 4 pixels of the HP row moved
        return img
    monkeypatch.setattr(hb, "_render", render)
    out = hb._task_lineage("repo", [(B1, PATH), (B2, PATH)], set(), set())
    pair = next(r for r in out if r["kind"] == "pair")
    assert pair["how"] == "render" and pair["px"] == 4 and pair["rows"] == [4, None, None, None]
    assert drawn == [(2, 2, 12, 12), (2, 2, 12, 12)]


# ========================
# The PSD readings (layer stand-ins; a real PSD with the clone)
# ========================
def _darks(*names, visible="3"):
    rows = [Layer(name, children=[Layer(str(d), "brightnesscontrast", visible=str(d) == visible) for d in range(10)])
            for name in names]
    return Layer("psd", children=[Layer("Stats Bars", children=[Layer("Darks", children=rows)])])


def test_the_old_template_is_renamed_by_position():
    psd = _darks("Hp Dark copy", "Hp Dark copy 2", "Hp Dark copy 3", "Hp Dark")
    assert hb.template(psd)
    assert [c.name for c in next(l for l in psd.descendants() if l.name == "Darks")] == ["Spd Dark", "Atk Dark", "Def Dark", "Hp Dark"]


def test_a_card_without_the_template_is_untouched():
    psd = _darks("Spd Dark", "Atk Dark", "Def Dark", "Hp Dark")
    assert not hb.template(psd)
    assert [c.name for c in next(l for l in psd.descendants() if l.name == "Darks")] == ["Spd Dark", "Atk Dark", "Def Dark", "Hp Dark"]


def test_hidden_drawn_cells_are_unread_hidden_darkening_is_ten():
    from commands.psd_parser import PSDParser
    parser = PSDParser([], cm.StatsDatabase())
    hearts = Layer("Hp Dark", children=[Layer(str(d), "pixel", visible=False) for d in range(1, 11)])
    darks = Layer("Def Dark", children=[Layer(str(d), "brightnesscontrast", visible=False) for d in range(10)])
    lit = Layer("Atk Dark", children=[Layer(str(d), "pixel", visible=d < 4) for d in range(1, 11)])
    psd = Layer("psd", children=[Layer("Stats Bars", children=[Layer("Darks", children=[hearts, darks, lit])])])
    assert hb.hidden_rows(parser, psd) == ["hp"]


def test_stat_rows_are_each_stats_drawn_layers():
    psd = Layer("psd", children=[Layer("Stats Bars", children=[
        Layer("Bars", children=[Layer("Hp Top", "pixel", bbox=(0, 0, 50, 5)), Layer("atk bottom", "pixel", bbox=(0, 20, 50, 25))]),
        Layer("Darks", children=[Layer("Def Dark", children=[Layer("3", "pixel", bbox=(10, 10, 15, 15))]),
                                 Layer("Spd Dark", children=[Layer("4", "exposure", bbox=(0, 0, 0, 0))])])]),
        Layer("Text", children=[Layer("Health", "type", bbox=(0, 0, 9, 9))]), Layer("hp art", "pixel", bbox=(0, 0, 99, 99))])
    assert hb.stat_boxes(psd) == [[(0, 0, 50, 5)], [(10, 10, 15, 15)], [(0, 20, 50, 25)], []]
    np = pytest.importorskip("numpy")
    a = np.zeros((30, 60, 3), dtype=np.int16)
    b = a.copy()
    b[0:2, 0:10] = 255
    assert hb.rows_moved(a, b, hb.stat_boxes(psd)) == [20, 0, 0, None]


def test_stat_rows_by_the_labels_position():
    """Groda: its 'def' and 'atk' fills swapped rows, the labels under "Lines" did not; Leshy's 'hp copy' fills and a
    box over every row."""
    labels = Layer("Lines", children=[Layer("Hp", "pixel", bbox=(0, 0, 50, 8)), Layer("Def ", "pixel", bbox=(0, 10, 50, 18)),
                                      Layer("Atk", "pixel", bbox=(0, 20, 50, 28))])
    psd = Layer("psd", children=[Layer("Stats Bars", children=[
        Layer("Bars", children=[Layer("def", "pixel", bbox=(0, 20, 50, 28)), Layer("atk", "pixel", bbox=(0, 10, 50, 18)),
                                Layer("hp copy", "pixel", bbox=(0, 20, 50, 28)), Layer("hp", "pixel", bbox=(0, 0, 50, 8)),
                                Layer("spd", "pixel", bbox=(0, 30, 50, 38))]),
        Layer("Layer 8", "pixel", bbox=(0, 0, 50, 38))]), labels])
    assert hb.stat_boxes(psd) == [[(0, 0, 50, 8)], [(0, 10, 50, 18)], [(0, 20, 50, 28)], [(0, 30, 50, 38)]]


def test_summed_rows_that_are_no_overflow_are_split():
    from commands.psd_parser import StatTracker, StatTrackers

    class Row:
        def __init__(self, group, items):
            self.group, self.items = group, items

    def tracker(*rows):
        t = StatTracker()
        t.rows = {i: Row(i, [(v, "exposure")]) for i, v in enumerate(rows)}
        return t
    card = cm.CardInfo(name="God Tamer", card_type="creature")
    trackers = StatTrackers(tracker(2, 4), tracker(9), tracker(10, 2), tracker(5, 0))
    assert hb.split_rows(trackers, card) == ["hp"]     # 10 + 2 is the overflow encoding; a 0 decoration is none
    card.creatures = [cm.CardStats(1, 1, 1, 1)]
    assert hb.split_rows(trackers, card) == []


def test_a_real_psd_reads_as_the_parser_does(clone):
    """Mini Doomer (a golden): today's parser's line, no template, no hidden rows, its bars' boxes."""
    pytest.importorskip("psd_tools")
    from psd_tools import PSDImage
    from golden_cards import VOCAB
    vocab = json.loads(VOCAB.read_text(encoding="utf-8"))
    hb._init((vocab["all_types"], vocab["stars"], vocab["manifest"]))
    psd = PSDImage.open(clone / "Auxiliary/Minions/Mini_Doomer.psd")
    got = hb.read_psd(hb._PARSER, psd, "Auxiliary/Minions/Mini_Doomer.psd")
    assert got["read"] == [5, 2, 5, 5] and got["how"] == "parser" and got["hidden"] == [] and got["boxes"]
    assert got["split"] == [] and all(got["rows"])         # each stat's own row found
    old = hb.read_psd(hb._PARSER, PSDImage.open(clone / "Auxiliary/Minions/Mini_Doomer.psd"), "Auxillary/Minions/Mini_Doomer.psd")
    assert old["type"] == "minion"                     # the old folder spelling read as today's


# ========================
# Fates (a tiny card repo)
# ========================
@pytest.fixture
def cards(tmp_path):
    repo = tmp_path / "cards"
    repo.mkdir()
    git(repo, "init", "-q", "-b", "main")
    for rel, data_ in (("Items/Equip/2 Stars/Old_Name.psd", b"old"), ("Auxiliary/Minions/Shield.psd", b"shield"),
                       ("Auxiliary/Items/Hammer.psd", b"hammer"), ("Creatures/O/3 Stars/Gone.psd", b"gone"),
                       ("Field/1 Stars/Retire_Me.psd", b"retire"), ("Items/Heal/2 Stars/Pizza.psd", b"pizza item"),
                       ("Creatures/O/3 Stars/Pizza.psd", b"pizza creature"), ("Creatures/O/3 Stars/Mid_A.psd", b"mid"),
                       ("Items/Support/2 Stars/Plan.psd", b"plan"), ("Auxiliary/Rulebook/Page.psd", b"page"),
                       ("MDW/Ech's stuff/Olimar.psd", b"draft")):
        write(repo, rel, data_)
    commit(repo, "alice", 1000)
    git(repo, "rm", "-q", "Items/Equip/2 Stars/Old_Name.psd")
    write(repo, "Items/Equip/2 Stars/New_Name.psd", b"new")
    commit(repo, "bob", 2000)
    git(repo, "mv", "Auxiliary/Minions/Shield.psd", "Auxiliary/Items/Shield.psd")
    git(repo, "rm", "-q", "Auxiliary/Items/Hammer.psd")
    commit(repo, "bob", 3000)
    (repo / "MDW/The Morgue").mkdir(parents=True)
    git(repo, "mv", "Field/1 Stars/Retire_Me.psd", "MDW/The Morgue/Retire_Me.psd")
    git(repo, "rm", "-q", "Creatures/O/3 Stars/Gone.psd", "Auxiliary/Rulebook/Page.psd", "MDW/Ech's stuff/Olimar.psd")
    commit(repo, "carol", 4000)
    git(repo, "rm", "-q", "Items/Heal/2 Stars/Pizza.psd")
    write(repo, "Items/Heal/2 Stars/Pepperoni_Pizza.psd", b"pepperoni")
    write(repo, "Field/1 Stars/Copy_Thing.psd", b"new")
    commit(repo, "dave", 5000)
    git(repo, "rm", "-q", "Field/1 Stars/Copy_Thing.psd")
    git(repo, "mv", "Creatures/O/3 Stars/Mid_A.psd", "Creatures/O/3 Stars/Mid_B.psd")
    commit(repo, "erin", 6000)
    git(repo, "mv", "Creatures/O/3 Stars/Mid_B.psd", "MDW/The Morgue/Mid_B.psd")
    git(repo, "rm", "-q", "Items/Support/2 Stars/Plan.psd")
    write(repo, "Items/Equip/2 Stars/Tomorrow.psd", b"tomorrow")
    commit(repo, "erin", 7000)
    return repo


def test_fates_of_the_cards_that_left(cards, monkeypatch):
    """A new name in the folder is a rename; a card moved in beside the deletion is no rename (Regenerative Shield); a
    move into MDW retires it; an owner-confirmed rename; another card of a live name gets a qualified key; a stray
    copy, a name that only passed through, rule pages and MDW drafts are left out."""
    monkeypatch.setattr(hb, "CONFIRMED", {"Items/Support/2 Stars/Plan.psd": "Tomorrow",
                                          "Items/Heal/2 Stars/Pizza.psd": "Pepperoni Pizza"})
    hist = hb.History.of(ch.GitIndex.from_git(str(cards)))
    mains, others = hb.lineages(hist)
    found = {name: (f.kind, f.target, title) for name, title, _, f, _ in hb.archive_candidates(hist, mains, others)}
    assert found == {"Old Name": ("renamed", "New Name", None), "Hammer": ("deleted", None, None),
                     "Retire Me": ("retired", "MDW/The Morgue/Retire_Me.psd", None), "Gone": ("deleted", None, None),
                     "Pizza (item)": ("renamed", "Pepperoni Pizza", "Pizza"), "Plan": ("renamed", "Tomorrow", None),
                     "Mid A": ("retired", "MDW/The Morgue/Mid_B.psd", None)}
    assert "Shield" in mains and mains["Shield"].alive and len(mains["Shield"].versions) == 1    # a move: one version


def test_authors_by_the_three_rules(cards):
    hist = hb.History.of(ch.GitIndex.from_git(str(cards)))
    from commands.card_edits import CardAuthors
    authors = CardAuthors.parse("[people]\nalice = Alice A\n[cards]\nNew Name = bob\nGone = zed\n")
    renamed = hb.Fate("renamed", None, "New Name")
    assert hb.credit(hist, authors, "Old Name", ["Items/Equip/2 Stars/Old_Name.psd"], renamed) == ["bob", "the card it became"]
    assert hb.credit(hist, authors, "Gone", ["Creatures/O/3 Stars/Gone.psd"], hb.Fate("deleted", None)) == ["zed", "Card Authors.txt"]
    assert hb.credit(hist, authors, "Hammer", ["Auxiliary/Items/Hammer.psd"], hb.Fate("deleted", None)) == \
        ["Alice A", "first git author of Auxiliary/Items/Hammer.psd"]


def test_a_built_archive_record_is_dated_as_later_cleanups_date_it(cards):
    hist = hb.History.of(ch.GitIndex.from_git(str(cards)))
    card = cdb.record(cm.CardInfo(name="Old Name", card_type="item", path="Items/Equip/2 Stars/Old_Name.psd", ability="Old."))
    live = {"new name": "New Name"}
    rec = hb._archive_record(hist, "Old Name", None, "Items/Equip/2 Stars/Old_Name.psd",
                             hb.Fate("renamed", hist.gone("Items/Equip/2 Stars/Old_Name.psd"), "New Name"), card, live)
    tag = rec["archived"]
    assert (tag["at"], tag["renamed_to"], tag["blob"], tag[hb.MARK]) == (2000.0, "New Name", blob(b"old"), hb.FILE_ID)
    assert rec["version"]["from"] == 1000.0 and rec["version"]["to"] == 2000.0
    again = cdb.card(deepcopy(rec))
    report = ch.Refined()
    ch.date_archive({"New Name": None}, {"Old Name": [again]}, hist.index, report, backfill=False)
    assert not report.tags_dated and not report.renamed and again.archived == tag


def test_a_renamed_record_shows_its_own_name_not_the_title_it_shares(cards):
    """Plan Ahead's PSD already printed "Tomorrow I’ll": its record shows "Plan Ahead"; Pandeity Old printed
    "Pandiety", another archived card's name; a title of its own stays."""
    hist = hb.History.of(ch.GitIndex.from_git(str(cards)))
    path = "Items/Equip/2 Stars/Old_Name.psd"
    found = hb.Fate("renamed", hist.gone(path), "New Name")

    def title(printed, names=frozenset()):
        card = cdb.record(cm.CardInfo(name="Old Name", card_type="item", path=path, display_name=printed))
        return hb._archive_record(hist, "Old Name", None, path, found, card, {"new name": "New Name"}, names).get("display_name")
    assert title("NEW name") is None and title("Pandiety", frozenset({"pandiety"})) is None
    assert title("Old Name!") == "Old Name!"
    assert hb._plain("Tomorrow I’ll") == hb._plain("Tomorrow I'll")


def test_a_title_another_spelling_of_a_card_has_is_no_title():
    others = hb._others({"Tainted Esau", "TaintedEsau", "Dark Esau"}, "TaintedEsau", "TaintedEsau")
    assert hb._plain("Tainted ESAU") in others and hb._plain("Dark Esau") in others


def test_build_refuses_without_the_side_files(cards, tmp_path, monkeypatch):
    """Card Authors.txt at head but its blob not in the clone: no file (it would credit every record by git alone)."""
    from global_config import AUTHORS_PATH
    write(cards, AUTHORS_PATH, b"[people]\n[cards]\nGone = zed\n")
    commit(cards, "alice", 8000)
    want = blob(b"[people]\n[cards]\nGone = zed\n")
    monkeypatch.setattr(hb, "_missing", lambda repo, blobs: [b for b in blobs if b == want])
    out = tmp_path / "out.json"
    assert hb.build(str(cards), "HEAD", str(tmp_path / "cache.jsonl"), str(out)) == 1 and not out.exists()
    assert (tmp_path / "missing.txt").read_text() == want + "\n"


def test_renamed_cards_are_read_as_one_card():
    """TaintedJacob's bytes are Tainted Jacob's first: one print run, so the record takes the newest reading too."""
    from types import SimpleNamespace as NS
    c = [NS(order=9 - i, at=float(1000 * (i + 1)), sha=f"c{i}") for i in range(3)]
    old = hb.Lineage("TaintedJacob", ["Creatures/T/3 Stars/TaintedJacob.psd"], None, False,
                     [(c[0], B1, "Creatures/T/3 Stars/TaintedJacob.psd")])
    new = hb.Lineage("Tainted Jacob", ["Creatures/T/3 Stars/Tainted_Jacob.psd"], None, True,
                     [(c[1], B1, "Creatures/T/3 Stars/Tainted_Jacob.psd"), (c[2], B2, "Creatures/T/3 Stars/Tainted_Jacob.psd")])
    stat = {"TaintedJacob": old, "Tainted Jacob": new}
    chains = hb.rename_chains(stat, [("TaintedJacob", None, old, hb.Fate("renamed", c[1], "Tainted Jacob"), "x")])
    joined, owners = chains["Tainted Jacob"]
    assert [b for _, b, _ in joined.versions] == [B1, B1, B2] and owners == ["TaintedJacob", "Tainted Jacob", "Tainted Jacob"]
    cache = {"read": {(B1, old.paths[0]): {"read": [10, 0, 4, 6]}, (B1, new.paths[0]): {"read": [10, 0, 4, 6]},
                      (B2, new.paths[0]): {"read": [5, 4, 7, 4]}},
             "pair": {(B1, B2): {"px": 0, "rows": [0, 0, 0, 0]}}, "card": {}}
    got = hb.assemble(None, stat, [], cache, None, None, {}, chains)
    assert [s["line"] for s in got["cards"]["TaintedJacob"]["segments"]] == [[5, 4, 7, 4]]
    assert [s["line"] for s in got["cards"]["Tainted Jacob"]["segments"]] == [[5, 4, 7, 4]] and not got["changes"]


def test_rehearse_never_writes_a_folder_whose_pickles_git_tracks(tmp_path):
    repo = tmp_path / "bot"
    repo.mkdir()
    git(repo, "init", "-q")
    (repo / "stats.pkl").write_bytes(b"x")
    git(repo, "add", "stats.pkl")
    assert hb.live_folder(str(repo)) and hb.live_folder(hb.BOT_CHECKOUT) and not hb.live_folder(str(tmp_path))
    assert hb.rehearse("none.json", str(tmp_path), "HEAD", str(tmp_path), str(repo)) == 1


# ========================
# The merge: stats
# ========================
def test_a_misread_is_corrected_in_place():
    old = defaultdict(list, A=[version(1000, B1, [1, 10, 1, 1])])
    stats = {"A": version(3000, B2, [1, 1, 1, 1], ability="Later.")}
    report = hb.merge(stats, old, data({"A": entry([reading(1000, B1, [1, 10, 1, 1], "first"), reading(3000, B2, [1, 1, 1, 1])])}),
                      ch.GitIndex())
    v = old["A"][0]
    assert hb._line(v) == [1, 1, 1, 1] and v.version["backfill_was"]["stats"] == [1, 10, 1, 1]
    assert v.version[hb.MARK] == hb.FILE_ID and report.corrected == ["A"]


def test_out_of_range_stats_clause_goes_into_backfill_was():
    v = version(1000, B1, [-1, -1, -1, -1])
    v.problems = ["OUT OF RANGE: stats 110/110/110/110 (unknown)", "SEVERAL CREATURES: check"]
    old = defaultdict(list, A=[v])
    hb.merge({}, old, data({"A": entry([reading(1000, B1, [7, 4, 5, 5], "first")], allowed=True)}), ch.GitIndex())
    assert hb._line(v) == [7, 4, 5, 5] and v.problems == ["SEVERAL CREATURES: check"]
    assert v.version["backfill_was"]["problems"] == ["OUT OF RANGE: stats 110/110/110/110 (unknown)"]


def test_a_change_inside_a_span_splits_it():
    old = defaultdict(list, A=[version(1000, B1, [5, 5, 5, 5])])
    stats = {"A": version(5000, B4, [6, 5, 5, 5], ability="Later.")}
    readings = [reading(1000, B1, [5, 5, 5, 5], "first"), reading(3000, B3, [6, 5, 5, 5], "changed"),
                reading(5000, B4, [6, 5, 5, 5])]
    report = hb.merge(stats, old, data({"A": entry(readings)}), ch.GitIndex())
    assert lines(old) == [(1000, [5, 5, 5, 5]), (3000, [6, 5, 5, 5])]
    added = old["A"][1]
    assert added.blob_sha == B3 and added.version["cause"] == "psd" and added.version["from_commit"] == "c3000"
    assert added.ability == "Hits." and old["A"][0].version["to"] == 3000 and report.added == [("A", 3000, [6, 5, 5, 5])]


def test_versions_before_the_first_only_with_an_earlier_change():
    readings = [reading(1000, B1, [5, 5, 5, 5], "first"), reading(3000, B3, [6, 5, 5, 5], "changed")]
    first = version(3500, B3, [6, 5, 5, 5])
    old = defaultdict(list, A=[first])
    report = hb.merge({}, old, data({"A": entry(readings)}), ch.GitIndex())
    assert lines(old) == [(1000, [5, 5, 5, 5]), (3000, [6, 5, 5, 5])] and report.earlier == [("A", False)]
    same = [reading(1000, B1, [6, 5, 5, 5], "first"), reading(3000, B3, [6, 5, 5, 5], "changed")]
    old = defaultdict(list, A=[version(3500, B3, [6, 5, 5, 5])])
    assert not hb.merge({}, old, data({"A": entry(same)}), ch.GitIndex()).changed() and lines(old) == [(3500, [6, 5, 5, 5])]


def test_a_first_version_starting_at_the_change_gets_the_line_before():
    """Grimmchild: its oldest stored version (or a current card with no history) starts at the change's own commit."""
    readings = [reading(1000, B1, [5, 5, 5, 5], "first"), reading(3000, B3, [6, 5, 5, 5], "changed")]
    old = defaultdict(list, A=[version(3000, B3, [6, 5, 5, 5])])
    report = hb.merge({}, old, data({"A": entry(readings)}), ch.GitIndex())
    assert lines(old) == [(1000, [5, 5, 5, 5]), (3000, [6, 5, 5, 5])] and report.added == [("A", 1000, [5, 5, 5, 5])]
    stats = {"A": version(3000, B3, [6, 5, 5, 5])}
    old = defaultdict(list)
    report = hb.merge(stats, old, data({"A": entry(readings)}), ch.GitIndex())
    assert lines(old) == [(1000, [5, 5, 5, 5])] and ch.start(stats["A"])[0] == 3000 and report.balance == {"buff": 1}
    assert not report.earlier and not report.later


def test_a_start_collapse_carries_into_the_current_card_is_listed():
    """A corrected version equal to the current card folds into it, which keeps the earlier start: listed."""
    readings = [reading(1000, B1, [5, 5, 5, 5], "first"), reading(2000, B2, [6, 5, 5, 5], "changed"),
                reading(3000, B3, [6, 5, 5, 5], "same")]
    old = defaultdict(list, A=[version(1000, B1, [5, 5, 5, 5]), version(2000, B2, [6, 5, 5, 9])])
    stats = {"A": version(3000, B3, [6, 5, 5, 5])}
    report = hb.merge(stats, old, data({"A": entry(readings)}), ch.GitIndex())
    assert lines(old) == [(1000, [5, 5, 5, 5])] and ch.start(stats["A"])[0] == 2000
    assert report.folded == ["A"] and report.earlier == [("A", True)]


def test_an_earlier_lifes_tag_keeps_its_place():
    """A name archived, then a new card under it: the old life's tagged record stays where its start puts it."""
    gone = version(1000, B1, [5, 5, 5, 5])
    gone.archived = {"at": 2000.0, "path": PATH, "blob": B1, "commit": None}
    old = defaultdict(list, A=[gone, version(3000, B3, [6, 5, 5, 5])])
    stats = {"A": version(4000, B4, [6, 5, 5, 5], ability="Later.")}
    readings = [reading(3000, B3, [6, 5, 5, 5], "first"), reading(4000, B4, [6, 5, 5, 5])]
    hb.merge(stats, old, data({"A": entry(readings)}), ch.GitIndex())
    assert [ch.start(v)[0] for v in old["A"]] == [1000, 3000] and old["A"][0].archived


def test_a_carried_start_is_split_and_moved_later():
    """A start recover carried earlier than its reading's segment: the earlier part becomes a copy with that line."""
    readings = [reading(1000, B1, [5, 5, 5, 5], "first"), reading(3000, B3, [6, 5, 5, 5], "changed")]
    old = defaultdict(list, A=[version(1000, B1, [5, 5, 5, 5], ability="One."), version(2000, B3, [6, 5, 5, 5], ability="Two.")])
    report = hb.merge({}, old, data({"A": entry(readings)}), ch.GitIndex())
    assert lines(old) == [(1000, [5, 5, 5, 5]), (2000, [5, 5, 5, 5]), (3000, [6, 5, 5, 5])]
    assert [v.ability for v in old["A"]] == ["One.", "Two.", "Two."] and report.later == [("A", False)]


def test_edit_versions_and_the_current_cards_data_are_left_alone():
    edited = version(2000, B1, [9, 9, 9, 9], cause="edit")
    old = defaultdict(list, A=[version(1000, B1, [5, 5, 5, 5]), edited])
    stats = {"A": version(3000, B1, [8, 8, 8, 8])}
    before = cdb.record(stats["A"])
    report = hb.merge(stats, old, data({"A": entry([reading(1000, B1, [5, 5, 5, 5], "first")])}), ch.GitIndex())
    assert hb._line(edited) == [9, 9, 9, 9] and cdb.record(stats["A"]) == before
    assert report.current == [("A", [8, 8, 8, 8], [5, 5, 5, 5])]
    assert report.waits() == "Backfill waits: A unlike the file, not re-parsed yet (/update_stats cards:A)."


def test_the_current_card_starts_earlier_with_an_earlier_change():
    readings = [reading(1000, B1, [5, 5, 5, 5], "first"), reading(3000, B3, [6, 5, 5, 5], "changed")]
    stats = {"A": version(4000, B3, [6, 5, 5, 5])}
    old = defaultdict(list)
    report = hb.merge(stats, old, data({"A": entry(readings)}), ch.GitIndex())
    assert lines(old) == [(1000, [5, 5, 5, 5])] and stats["A"].version["from"] == 3000
    assert hb._line(stats["A"]) == [6, 5, 5, 5] and report.earlier == [("A", True)]


def test_an_unmatched_legacy_version_never_makes_a_change():
    lone = version(2000, None, [7, 7, 7, 7], path="Creatures/Other/3 Stars/Copy_of_A.psd")
    old = defaultdict(list, A=[version(1000, B1, [5, 5, 5, 5]), lone])
    stats = {"A": version(3000, B1, [5, 5, 5, 5])}
    report = hb.merge(stats, old, data({"A": entry([reading(1000, B1, [5, 5, 5, 5], "first")])}), ch.GitIndex())
    assert report.unmatched == [("A", 2000)] and report.neutral == ["A"]
    assert all(line == [5, 5, 5, 5] for _, line in lines(old) + [(0, hb._line(stats["A"]))])


def test_a_blob_met_twice_matches_the_meeting_at_its_start():
    """A revert: the version of the reverted bytes is the later meeting's segment, not the first's."""
    readings = [reading(1000, B1, [5, 5, 5, 5], "first"), reading(2000, B2, [6, 5, 5, 5], "changed"),
                reading(3000, B1, [5, 5, 5, 5], "changed")]
    assert hb.match(version(3000, B1, [5, 5, 5, 5]), readings, ch.GitIndex()) == (2, B1)
    assert hb.match(version(1000, B1, [5, 5, 5, 5]), readings, ch.GitIndex()) == (0, B1)
    old = defaultdict(list, A=[version(1000, B1, [5, 5, 5, 5]), version(2000, B2, [6, 5, 5, 5])])
    stats = {"A": version(3000, B1, [5, 5, 5, 5])}
    assert not hb.merge(stats, old, data({"A": entry(readings)}), ch.GitIndex()).changed()


def test_a_correction_never_adds_a_change_the_stored_history_did_not_count():
    """Ech: today's parser reads its 2025-03-22 ATK as 0 where the preview prints 4 (the stored reading). Correcting it
    would add a shift and a buff; the stored line stays and is listed."""
    readings = [reading(1000, B1, [2, 2, 3, 5], "first"),
                reading(2000, B2, [2, 5, 0, 5], "changed", row_px=[0, 13668, 4556, 0]),
                reading(3000, B3, [2, 5, 4, 5], "changed", row_px=[9000, 9000, 45431, 9000])]
    old = defaultdict(list, A=[version(2000, B2, [2, 5, 4, 5]), version(3000, B3, [2, 5, 4, 5], ability="Later.")])
    report = hb.merge({}, old, data({"A": entry(readings)}), ch.GitIndex())
    assert lines(old) == [(1000, [2, 2, 3, 5]), (2000, [2, 5, 4, 5]), (3000, [2, 5, 4, 5])]   # the line before: a copy
    assert report.corrected == [] and [(k[0], k[2], k[3], k[4]) for k in report.kept] == [
        ("A", [2, 5, 4, 5], [2, 5, 0, 5], "adds a change")]


def test_a_version_whose_bytes_read_as_nothing_keeps_its_line():
    """God Tamer: its 2025-04-03 bytes sum two creatures' rows (unread), so its honest -1s stay."""
    readings = [reading(1000, B1, [7, 9, 10, 12], "first", split=["hp"]),
                reading(3000, B3, [3, 4, 4, 6], "changed", row_px=[9, 9, 9, 9])]
    old = defaultdict(list, A=[version(1000, B1, [-1, -1, -1, -1])])
    stats = {"A": version(3000, B3, [3, 4, 4, 6])}
    report = hb.merge(stats, old, data({"A": entry(readings)}), ch.GitIndex())
    assert lines(old) == [(1000, [-1, -1, -1, -1])] and report.kept[0][4] == "unread"
    assert not (report.balance["buff"] or report.balance["nerf"] or report.balance["shift"]) and not report.added


def test_versions_newer_than_the_file_are_left_alone():
    """Two PSD edits after the build (6/5/5/5, then 7/5/5/5), each read by an /update_stats before the merge."""
    readings = [reading(1000, B1, [5, 5, 5, 5], "first"), reading(2000, B2, [5, 5, 5, 5])]
    file = data({"A": entry(readings)})
    file["made"]["card_head_at"] = 2500
    old = defaultdict(list, A=[version(2000, B2, [5, 5, 5, 5]), version(3000, B3, [6, 5, 5, 5])])
    stats = {"A": version(4000, B4, [7, 5, 5, 5])}
    report = hb.merge(stats, old, file, ch.GitIndex())
    assert lines(old) == [(2000, [5, 5, 5, 5]), (3000, [6, 5, 5, 5])] and hb._line(stats["A"]) == [7, 5, 5, 5]
    assert report.late == [("A", 3000), ("A", 4000)] and not report.corrected and not report.folded
    old = defaultdict(list, A=[version(2000, B2, [5, 5, 5, 5]), version(3000, B3, [6, 5, 5, 5])])
    held = version(3500, B3, [6, 5, 5, 5], cause="edit", note="x")    # then a database-only edit
    hb.merge({"A": held}, old, file, ch.GitIndex())
    assert lines(old) == [(2000, [5, 5, 5, 5]), (3000, [6, 5, 5, 5])]


def test_balance_counts_current_cards_and_the_line_names_those_that_differ():
    old = defaultdict(list, A=[version(1000, B1, [5, 5, 5, 5])], Gone=[version(1000, B1, [5, 5, 5, 5]),
                                                                    version(2000, B2, [6, 5, 5, 5])])
    old["Gone"][1].archived = {"at": 3000.0, "path": PATH, "blob": B2, "commit": None}
    counts, cards = hb.balance({"A": version(2000, B2, [4, 5, 5, 5])}, old)
    assert counts == {"nerf": 1} and cards == 1


def test_the_merge_twice_and_the_cleanup_after_it_change_nothing():
    readings = [reading(1000, B1, [5, 5, 5, 5], "first"), reading(3000, B3, [6, 5, 5, 5], "changed"),
                reading(5000, B4, [6, 5, 5, 5])]
    db = cm.StatsDatabase()
    db.stats = {"A": version(5000, B4, [6, 5, 5, 5])}
    db.old_stats = defaultdict(list, A=[version(1000, B1, [5, 5, 5, 6])])
    file = data({"A": entry(readings)})
    assert hb.merge(db.stats, db.old_stats, file, ch.GitIndex()).changed()
    again = deepcopy((db.stats, db.old_stats))
    assert not hb.merge(*again, file, ch.GitIndex()).changed()
    assert cdb.old_stats_text(again[1]) == cdb.old_stats_text(db.old_stats)
    assert cd.cleaned(db, ch.GitIndex()) is None


# ========================
# The merge: archive
# ========================
def _record(name, fate, path, title=None, **tag):
    card = cm.CardInfo(name=name, card_type="item", path=path, timestamp=100.0, ability="Was.", blob_sha=B1,
                       display_name=title, author="someone")
    card.version = {**ch.meta(None, 100.0, "psd"), hb.MARK: hb.FILE_ID}
    card.archived = {"at": 200.0, "path": path, "blob": B1, "commit": "c200", **tag, hb.MARK: hb.FILE_ID}
    return {"fate": fate, "how": "", "path": path, "record": cdb.record(card)}


def test_archive_records_retags_and_authors():
    old = defaultdict(list)
    esau = cm.CardInfo(name="Tainted Esau", card_type="minion", path="Auxiliary/Minions/Tainted_Esau.psd", author="x")
    esau.archived = {"at": 5.0, "path": esau.path, "blob": None, "commit": None, "backfill": True}
    ech = cm.CardInfo(name="Ech", card_type="creature", path="Creatures/Other/5 Stars/Ech.psd", author="MichaelJSr")
    ech.archived = {"at": 6.0, "path": ech.path, "blob": None, "commit": None, "moved_to": "MDW/The Morgue/Ech.psd"}
    old["Tainted Esau"], old["Ech"] = [esau], [ech]
    stats = {"Pizza": cm.CardInfo(name="Pizza", card_type="creature"),
             "Dark Esau": cm.CardInfo(name="Dark Esau", card_type="minion", author="TyKim182"),
             "Tomorrow I'll": cm.CardInfo(name="Tomorrow I'll", card_type="item", author="Croissant")}
    archive = {"Plan Ahead": _record("Plan Ahead", "renamed", "Items/Support/2 Stars/Plan_Ahead.psd", renamed_to="Tomorrow I'll"),
               "Pizza (item)": _record("Pizza (item)", "renamed", "Items/Heal/2 Stars/Pizza.psd", "Pizza", renamed_to="Pepperoni Pizza"),
               "Gilded": _record("Gilded", "deleted", "Auxiliary/Gilded.psd"),
               "Tainted Esau": _record("Tainted Esau", "renamed", esau.path, renamed_to="Dark Esau"),
               "Ech": _record("Ech", "retired", ech.path, moved_to="MDW/The Morgue/Ech.psd")}
    moved = cm.CardInfo(name="Militron No Armor", card_type="creatures", path="Auxiliary/Minions/Militron_No_Armor.psd")
    moved.archived = {"at": 7.0, "path": moved.path, "blob": B1, "commit": "c200", "renamed_to": "Militron"}
    old["Militron No Armor"] = [moved]
    archive["Militron No Armor"] = _record("Militron No Armor", "renamed", "Auxiliary/Creatures/Militron_No_Armor.psd",
                                           renamed_to="Militron")
    stats["Militron"] = cm.CardInfo(name="Militron", card_type="creatures", author="Croissant")
    authors = {"Gilded": ["Croissant", "first git author of x"], "Ech": ["TyKim182", "Card Authors.txt"]}
    report = hb.merge(stats, old, data(archive=archive, authors=authors), ch.GitIndex())
    db = cm.StatsDatabase()
    db.stats, db.old_stats = stats, old
    found = db.archived()
    assert set(found) == {"Plan Ahead", "Pizza (item)", "Gilded", "Tainted Esau", "Ech", "Militron No Armor"}
    assert moved.archived["path"] == "Auxiliary/Creatures/Militron_No_Armor.psd" and moved.archived[hb.MARK] == hb.FILE_ID
    assert report.paths == [("Militron No Armor", "Auxiliary/Creatures/Militron_No_Armor.psd")]
    assert sorted(report.archived) == [("Gilded", "deleted"), ("Pizza (item)", "renamed"), ("Plan Ahead", "renamed")]
    assert found["Pizza (item)"].display_name == "Pizza" and "Pizza" in stats and stats["Pizza"].card_type == "creature"
    assert esau.archived["renamed_to"] == "Dark Esau" and report.retagged == [("Tainted Esau", "Dark Esau")]
    assert {n: c.archived.get("author") for n, c in found.items()} == {
        "Plan Ahead": "Croissant", "Pizza (item)": "someone", "Gilded": "Croissant", "Tainted Esau": "TyKim182",
        "Ech": "TyKim182", "Militron No Armor": "Croissant"}
    assert report.authors == 6
    assert not db.restore("Pizza", "Items/Heal/2 Stars/Pizza.psd", B1) and not db.restore("Pizza (item)", "x", None)
    assert not hb.merge(stats, old, data(archive=archive, authors=authors), ch.GitIndex()).changed()


def test_a_renamed_records_author_is_its_cards_credit_this_run():
    """Card Authors.txt as this run applies it (merged after the traversal; a rehearsal passes the file)."""
    from commands.card_edits import CardAuthors
    stats = {"Firebomb": cm.CardInfo(name="Firebomb", card_type="item", author="MichaelJSr")}
    tag = {"renamed_to": "Firebomb"}
    file = data(authors={"Fire Bomb": ["bobberlington", "the card it became"]})
    assert hb._author("Fire Bomb", tag, file, stats, None) == "MichaelJSr"
    authors = CardAuthors.parse("[people]\n[cards]\nFirebomb = bobberlington\n")
    assert hb._author("Fire Bomb", tag, file, stats, None, authors) == "bobberlington"


# ========================
# Once
# ========================
def _file(tmp_path, content):
    path = tmp_path / "history_backfill.json"
    path.write_text(hb.file_text(content), encoding="utf-8")
    return str(path)


def test_merged_waits_runs_once_and_never_half(tmp_path, monkeypatch):
    readings = [reading(1000, B1, [5, 5, 5, 5], "first"), reading(3000, B3, [6, 5, 5, 5], "changed")]
    path = _file(tmp_path, data({"A": entry(readings)}))
    db = cm.StatsDatabase()
    db.stats, db.old_stats = {"A": version(4000, B3, [6, 5, 5, 5])}, defaultdict(list)
    assert hb.merged(db, None, path) == (None, False)                         # no history index: waits quietly
    assert hb.merged(db, ch.GitIndex(), str(tmp_path / "none.json")) == (None, False)
    assert hb.merged(db, ch.GitIndex(), path) == ("Backfill waits for the history recovery.", False)
    db.old_stats["Z"] = [restored()]
    before = cdb.old_stats_text(db.old_stats)
    stale = deepcopy(db)
    stale.stats["A"] = version(3000, B3, [9, 9, 9, 9])                       # not re-parsed: waits, unchanged
    assert hb.merged(stale, ch.GitIndex(), path) == (
        "Backfill waits: A unlike the file, not re-parsed yet (/update_stats cards:A).", False)
    assert cdb.old_stats_text(stale.old_stats) == before and not hb.merged_before(stale.stats, stale.old_stats, hb.FILE_ID)
    monkeypatch.setattr(hb, "merge", lambda *a: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(RuntimeError):
        hb.merged(db, ch.GitIndex(), path)
    assert cdb.old_stats_text(db.old_stats) == before                         # the trial copy only
    monkeypatch.undo()
    line, changed = hb.merged(db, ch.GitIndex(), path)
    assert changed and line.startswith("Backfill: 3 PSD versions of 1 cards read") and "1 stat version added" in line
    assert hb.merged(db, ch.GitIndex(), path) == (None, False)


def test_card_data_imports_the_backfill_lazily():
    """card-sync.yml's sparse checkout has no history_backfill: card_data imports it only inside update()."""
    import ast
    assert "commands/history_backfill.py" not in cd.SPARSE_FILES
    tree = ast.parse((REPO / "commands" / "card_data.py").read_text(encoding="utf-8"))
    top = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    assert not any("history_backfill" in ast.dump(n) for n in top)
    update = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "update")
    assert "history_backfill" in ast.dump(update)
    text = ast.get_source_segment((REPO / "commands" / "card_data.py").read_text(encoding="utf-8"), update)
    assert text.index("traverse_remote(") < text.index("history_backfill")    # after this run's Card Authors.txt
    assert "history_backfill" not in (REPO / "setup.py").read_text(encoding="utf-8")      # never compiled


# ========================
# /update_stats end to end (the fake GitHub)
# ========================
def test_update_stats_merges_once(github, psds, tmp_path, monkeypatch, capsys):
    import test_update_stats as tus
    tus._run()
    doomer = tus._sha(psds[tus.DOOMER])
    readings = [{**reading(tus.TIME - 2000, B1, [4, 2, 5, 5], "first"), "path": tus.DOOMER},
                {**reading(tus.TIME - 1000, doomer, [5, 2, 5, 5], "changed"), "path": tus.DOOMER}]
    file = data({"Mini Doomer": {"paths": [tus.DOOMER], "allowed_high": False, "readings": readings,
                                 "segments": hb.segments(readings, False)}})
    monkeypatch.setattr(hb, "FILE", _file(tmp_path, file))
    monkeypatch.setattr(cd, "history_index", lambda clone=None: ch.GitIndex())
    monkeypatch.setattr(hb, "recovered_before", lambda stats, old: True)
    capsys.readouterr()
    tus._run()
    out = capsys.readouterr().out
    assert out.count("Backfill: ") == 1 and "1 stat version added" in out and "(1 current card)" in out
    stats, old = tus._saved()
    assert [hb._line(v) for v in old["Mini Doomer"]] == [[4, 2, 5, 5]] and stats["Mini Doomer"].version["from"] == tus.TIME - 1000
    tus._run()
    assert "Backfill" not in capsys.readouterr().out
    monkeypatch.setattr(hb, "merged_before", lambda *a: False)
    monkeypatch.setattr(hb, "merge", lambda *a: (_ for _ in ()).throw(RuntimeError("boom")))
    tus._run()
    assert "Backfill skipped (RuntimeError: boom)." in capsys.readouterr().out
    assert tus._saved()[1]["Mini Doomer"] == old["Mini Doomer"]


# ========================
# The committed file
# ========================
@pytest.fixture(scope="module")
def committed():
    data_ = hb.load_file(str(REPO / hb.FILE_NAME))
    if data_ is None:
        pytest.skip("no history_backfill.json")
    return data_


def _changes(data_, name):
    return [(c[0], c[2], c[3], c[4]) for c in data_["changes"] if c[1] == name]


def test_worked_examples(committed):
    """The design's examples: misreads make no change, printed changes stay."""
    for name in ("Sans", "Twin Emperors", "Tainted Isaac", "Tainted Lazarus"):
        assert _changes(committed, name) == [], name
    assert [s["line"] for s in committed["cards"]["Sans"]["segments"]] == [[1, 1, 1, 1]]
    assert [s["line"] for s in committed["cards"]["Twin Emperors"]["segments"]] == [[7, 4, 5, 6]]
    assert [s["line"] for s in committed["cards"]["Tainted Lazarus"]["segments"]] == [[7, 4, 5, 5]]
    assert _changes(committed, "Sonic") == [("2025-02-21", [7, 2, 3, 20], [7, 4, 3, 20], "buff"),
                                            ("2025-03-09", [7, 4, 3, 20], [7, 4, 4, 20], "buff")]
    assert not [c for c in _changes(committed, "Tainted Eden") if c[0] >= "2026"]
    assert _changes(committed, "Tainted Eden") == [("2023-11-24", [7, 3, 4, 4], [6, 3, 6, 4], "buff")]   # one row edit
    for name in ("Clot Bone", "God Tamer"):             # all-0 hearts and summed creatures: unread, no change
        assert _changes(committed, name) == [], name
    assert [s["line"] for s in committed["cards"]["God Tamer"]["segments"]] == [[3, 4, 4, 6]]
    assert ("2025-03-22", [2, 2, 3, 5], [2, 5, 0, 5], "shift") in _changes(committed, "Ech")   # the merge keeps 2/5/4/5
    assert [s["line"] for s in committed["cards"]["TaintedJacob"]["segments"]] == [[5, 4, 7, 4]]   # its successor's
    edits = {(e[0], e[1]): e[4] for e in committed["unread_edits"]}
    assert edits[("2023-11-24", "Chef Saltbaker")] == {"atk": 7547} and ("2025-08-28", "Slugcat") in edits
    assert [(c[0], c[3]) for c in _changes(committed, "Star Dream")] == [("2025-04-29", "buff"), ("2025-08-27", "nerf")]
    assert ("2025-10-30", [8, 6, 4, 4], [7, 7, 7, 7], "buff") in _changes(committed, "Kyle")
    assert [(c[0], c[2][2], c[3]) for c in _changes(committed, "Green Mage")] == [("2025-03-11", 6, "nerf")]


def test_the_archive_in_the_file(committed):
    cards = committed["archive"]["cards"]
    fates = {name: (e["fate"], (e.get("record") or {}).get("archived", {}).get("renamed_to")) for name, e in cards.items()}
    for old, new in (("Plan Ahead", "Tomorrow I'll"), ("Nuts and Bolts", "Bolts and Screws"), ("Pizza (item)", "Pepperoni Pizza"),
                     ("Tainted Esau", "Dark Esau"), ("Sacrifice", "Erasure"), ("TaintedJacob", "Tainted Jacob"),
                     ("Dogma (item)", "Indoctrination"), ("Auto Croissant (minion)", "Mini Auto Croissant")):
        assert fates[old] == ("renamed", new), old
    assert fates["Amy Rose's Hammer"] == ("renamed", "Piko Piko Hammer") and fates["Gilded"][0] == "deleted"   # git -M
    for name in ("Fire Guardian's Stronghold", "Pandeity Old", "Ech"):
        assert fates[name][0] == "retired", name
    assert "Twin Princes" not in cards and "Pandeity" not in cards and "Soul Master - Copy" not in cards
    assert cards["Pizza (item)"]["record"]["display_name"] == "Pizza"
    for name in ("Plan Ahead", "Magalor", "Pandeity Old", "Shef Saltbaker"):     # never the title another card has
        assert not cards[name]["record"].get("display_name"), name
    assert committed["archive"]["authors"]["Ech"][0] == "TyKim182"
    for entry_ in cards.values():
        assert hb.is_card(entry_["path"]) and "Ech's stuff" not in entry_["path"]
    assert len(cards) == 92 and len(committed["archive"]["authors"]) == 92


def test_every_rename_is_a_live_card(committed, real_stats):
    for name, entry_ in committed["archive"]["cards"].items():
        target = (entry_.get("record") or {}).get("archived", {}).get("renamed_to")
        if entry_["fate"] == "renamed":
            assert target in real_stats.stats, (name, target)
