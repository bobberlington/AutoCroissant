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
    ([7, 2, 3, 20], {}, False, None),                   # out of range without an allowance
    ([7, 2, 3, 20], {}, True, "full"),                  # allow_high_stats (Sonic)
    ([110, 110, 110, 110], {}, True, None),             # three digits: never
    ([10, 3, 5, 5], {"hidden": ["hp"]}, False, None),   # a drawn-cell row all hidden
    (None, {"err": "classifier: no star count"}, False, None),
])
def test_status(read, extra, allowed, want):
    assert hb.status({"read": read, **extra}, allowed) == want


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
                                                                   "boxes": [[0, 0, 10, 10]]})
    monkeypatch.setattr(hb, "_preview", lambda psd: None)
    monkeypatch.setattr(hb, "_render", lambda psd, box: drawn.append(box) or np.zeros((10, 10, 3), dtype=np.int16))
    out = hb._task_lineage("repo", [(B1, PATH), (B2, PATH)], set(), set())
    pair = next(r for r in out if r["kind"] == "pair")
    assert pair["how"] == "render" and pair["px"] == 0 and drawn == [(0, 0, 10, 10), (0, 0, 10, 10)]


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
    """A new name in the folder is a rename; a card moved in beside the deletion is no rename (Amy Rose's Hammer); a
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
    segs = hb.segments(readings, False)
    assert hb.match(version(3000, B1, [5, 5, 5, 5]), readings, segs, {PATH}, ch.GitIndex()) == 2
    assert hb.match(version(1000, B1, [5, 5, 5, 5]), readings, segs, {PATH}, ch.GitIndex()) == 0
    old = defaultdict(list, A=[version(1000, B1, [5, 5, 5, 5]), version(2000, B2, [6, 5, 5, 5])])
    stats = {"A": version(3000, B1, [5, 5, 5, 5])}
    assert not hb.merge(stats, old, data({"A": entry(readings)}), ch.GitIndex()).changed()


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
    authors = {"Gilded": ["Croissant", "first git author of x"], "Ech": ["TyKim182", "Card Authors.txt"]}
    report = hb.merge(stats, old, data(archive=archive, authors=authors), ch.GitIndex())
    db = cm.StatsDatabase()
    db.stats, db.old_stats = stats, old
    found = db.archived()
    assert set(found) == {"Plan Ahead", "Pizza (item)", "Gilded", "Tainted Esau", "Ech"}
    assert sorted(report.archived) == [("Gilded", "deleted"), ("Pizza (item)", "renamed"), ("Plan Ahead", "renamed")]
    assert found["Pizza (item)"].display_name == "Pizza" and "Pizza" in stats and stats["Pizza"].card_type == "creature"
    assert esau.archived["renamed_to"] == "Dark Esau" and report.retagged == [("Tainted Esau", "Dark Esau")]
    assert {n: c.archived.get("author") for n, c in found.items()} == {
        "Plan Ahead": "Croissant", "Pizza (item)": "someone", "Gilded": "Croissant", "Tainted Esau": "TyKim182",
        "Ech": "TyKim182"}
    assert report.authors == 5
    assert not db.restore("Pizza", "Items/Heal/2 Stars/Pizza.psd", B1) and not db.restore("Pizza (item)", "x", None)
    assert not hb.merge(stats, old, data(archive=archive, authors=authors), ch.GitIndex()).changed()


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
    assert [c[0] for c in _changes(committed, "Tainted Eden")] == ["2023-11-17", "2023-11-24"]
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
    assert fates["Amy Rose's Hammer"][0] == "deleted" and fates["Gilded"][0] == "deleted"
    for name in ("Fire Guardian's Stronghold", "Pandeity Old", "Ech"):
        assert fates[name][0] == "retired", name
    assert "Twin Princes" not in cards and "Pandeity" not in cards and "Soul Master - Copy" not in cards
    assert cards["Pizza (item)"]["record"]["display_name"] == "Pizza"
    assert committed["archive"]["authors"]["Ech"][0] == "TyKim182"
    for entry_ in cards.values():
        assert hb.is_card(entry_["path"]) and "Ech's stuff" not in entry_["path"]
    assert len(cards) == 92 and len(committed["archive"]["authors"]) == 92


def test_every_rename_is_a_live_card(committed, real_stats):
    for name, entry_ in committed["archive"]["cards"].items():
        target = (entry_.get("record") or {}).get("archived", {}).get("renamed_to")
        if entry_["fate"] == "renamed":
            assert target in real_stats.stats, (name, target)
