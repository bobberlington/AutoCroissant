"""scripts/oldgit_index.py (by hand): the first repo's authors and its cards joined to the old repo's."""
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "oldgit_index.py"


@pytest.fixture(scope="module")
def oi():
    spec = importlib.util.spec_from_file_location("oldgit_index", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_authors_map_by_a_shared_email(oi):
    meta = {"a": {"author": "panda", "email": "p@x"}, "b": {"author": "Kitty", "email": "p@x"},
            "c": {"author": "Stranger", "email": "s@x"}}
    got = oi.mapped_authors(meta, {"_source": "owner", "panda": "Michael"})
    assert got == {"panda": "Michael", "Kitty": "Michael"}             # no e-mail match: unmapped


def test_a_first_repo_card_joins_the_old_card_of_its_last_name(oi):
    events = [{"lineage": "F1", "path": "Cards/Kirby/3 Stars/Magolor.png"},
              {"lineage": "F1", "path": "Creatures/Kirby/3 Stars/Magolor.png"},
              {"lineage": "F2", "path": "Items/Heal/2 Stars/Grapes2.png"},
              {"lineage": "F3", "path": "Items/Attack/2 Stars/Bomb.png"},
              {"lineage": "L1", "path": "Creatures/Kirby/3 Stars/Magolor.png"},
              {"lineage": "L2", "path": "Items/Heal/2 Stars/Grapes.png"},
              {"lineage": "L3", "path": "Items/Attack/2 Stars/Bomb.png"},
              {"lineage": "L4", "path": "Auxillary/Items/2 Stars/Bomb.png"}]
    got = oi.join_first({}, lambda x: x, events, {})
    assert got == {"F1": "L1", "F3": "L3"}                             # a name and its folder; never a near name
    meme = [{"lineage": "F9", "path": "Cards/Meme/3 Stars/Something.png"},
            {"lineage": "L9", "path": "Creatures/Omori/3 Stars/SOMETHING.png"}]
    assert oi.join_first({}, lambda x: x, meme, {}) == {}                # another folder: another card


def test_a_re_upload_in_batches_carries_the_files_it_has_not_sent_yet(oi):
    meta = {c: {"parents": p, "repo": "old"} for c, p in (("A", []), ("R1", ["A"]), ("R2", ["R1"]), ("R3", ["R2"]))}
    trees = {"A": {"C/x.png": "b1", "C/y.png": "b2", "C/z.PNG": "b4"},
             "R1": {"C/x.png": "b1", "C/z.png": "b4"},                 # a batch: y not sent yet; z's case changed
             "R2": {"C/x.png": "b1", "C/y.png": "b3", "C/z.png": "b4"},   # y sent, edited
             "R3": {"C/x.png": "b1", "C/z.png": "b4"}}                   # after the batches: y deleted
    state, lineages, events, deletes = oi.walk(list(meta), meta, trees, carry={"R1", "R2"})
    got = [(e["commit"], e["path"], e["kind"]) for e in events]
    assert got[3:] == [("R1", "C/z.png", "move"), ("R2", "C/y.png", "edit")]   # no add: one lineage each
    assert len(lineages) == 3 and [(d["commit"], d["path"]) for d in deletes] == [("R3", "C/y.png")]
    _, _, events, deletes = oi.walk(list(meta), meta, trees, carry=())
    assert ("R2", "C/y.png", "add") in [(e["commit"], e["path"], e["kind"]) for e in events]
    assert ("R1", "C/y.png") in [(d["commit"], d["path"]) for d in deletes]


def test_lost_commits_come_after_their_parents_a_linked_root_after_its_link(oi):
    found = [("R2", {"parents": ["R1"]}), ("R1", {"parents": ["T"]}), ("T", {"parents": ["O"]})]
    assert [sha for sha, _ in oi.parents_first(found, {"O"})] == ["T", "R1", "R2"]
    with pytest.raises(ValueError):
        oi.parents_first([("X", {"parents": ["Y"]}), ("Y", {"parents": ["X"]})], set())


def test_the_owner_s_old_maps_rule_over_the_evidence(monkeypatch):
    """Owner 11-15: an old card the owner mapped (owner.json maps) is that live card's, whatever its file name says
    (Fire Bomb -> Flame Bomb, not Firebomb) or with nothing behind it (Eldman -> Elmadan); a ruling whose card is
    no longer live is not applied."""
    monkeypatch.syspath_prepend(str(SCRIPT.parent))
    import oldgit_map as om
    stats = {"Flame Bomb": {}, "Firebomb": {}, "Elmadan": {}}
    maps = {"Fire Bomb (Auxillary/Items/4 Stars)": {"key": "Flame Bomb", "answer": "a2000-fire-bomb"},
            "Eldman": {"key": "Elmadan", "answer": "a1940-eldman"}, "Gone": {"key": "Old Card", "answer": "a9"}}
    assert om.owner_map("Fire Bomb (Auxillary/Items/4 Stars)", maps, stats) == (("live", "Flame Bomb"),
                                                                                "a2000-fire-bomb")
    assert om.owner_map("Eldman", maps, stats) == (("live", "Elmadan"), "a1940-eldman")
    assert om.owner_map("Gone", maps, stats) is None and om.owner_map("Fire Bomb", maps, stats) is None
    maps.update({"Giegue (Cards/Meme/5 Stars)": {"key": "Giegue", "answer": "a1117-meme"},    # owner 11-17: archived
                 "Zero Two (Cards/Meme/2 Stars)": {"key": None, "answer": "a1117-meme"}})      # a different card
    archived = {"Giegue": {"path": "Creatures/Other/6 Stars/Giegue.psd"}}
    assert om.owner_map("Giegue (Cards/Meme/5 Stars)", maps, stats, archived) == (("archived", "Giegue"), "a1117-meme")
    assert om.owner_map("Giegue (Cards/Meme/5 Stars)", maps, stats) is None
    assert om.owner_map("Zero Two (Cards/Meme/2 Stars)", maps, stats, archived) == ((None, None), "a1117-meme")
    tie = {"key": "Elmadan", "status": "live"}                 # an eye tie: a lost file, or over unsure evidence
    assert om.tie_applies(tie, None, None, False, stats) and om.tie_applies(tie, ("live", "Firebomb"), "unsure", False, stats)
    assert not om.tie_applies(tie, ("live", "Firebomb"), "sure", False, stats)
    assert not om.tie_applies(tie, None, None, True, stats) and not om.tie_applies({"key": "Gone"}, None, None, False, stats)


def test_a_first_repo_meme_file_joins_its_name_only_in_its_own_folder(oi, tmp_path, monkeypatch):
    """The joke item Items/Meme/2 Stars/pain.png is not the debuff Auxillary/.../Pain.png (another kind's folder: two
    old cards); Alex YIIK's Cards/Meme file moved to Cards/Other stays one card."""
    monkeypatch.setattr(oi, "OUT", str(tmp_path))
    meta = {c: {"parents": p, "repo": "first", "date": f"2023-04-0{n}T00:00:00-07:00", "authored": None, "ts": n,
                "author": "Michael", "subject": "s"} for n, (c, p) in enumerate((("A", []), ("B", ["A"]), ("C", ["B"])), 1)}
    trees = {"A": {"Items/Meme/2 Stars/pain.png": "b1", "Cards/Meme/5 Stars/Alex_YIIK.png": "b2"}, "B": {},
             "C": {"Auxillary/Status Effects/Debuffs/1 Stars/Pain.png": "b3", "Cards/Other/5 Stars/Alex_YIIK.png": "b4"}}
    state, lineages, events, deletes = oi.walk(["A", "B", "C"], meta, trees, carry=())
    assert len(lineages) == 4
    oi.finish(["A", "B", "C"], meta, {}, {}, state, lineages, events, deletes)
    cards = oi.json.load(open(tmp_path / "index.json"))["old_cards"]
    assert {k: v["paths"] for k, v in cards.items()} == {
        "pain": ["Items/Meme/2 Stars/pain.png"], "Pain": ["Auxillary/Status Effects/Debuffs/1 Stars/Pain.png"],
        "Alex YIIK": ["Cards/Meme/5 Stars/Alex_YIIK.png", "Cards/Other/5 Stars/Alex_YIIK.png"]}


def test_mdw_drafts_are_card_files_of_their_own_lineages(oi):
    """s3 gap round 2: MDW/Croissant's Special Folder and MDW/Ech's stuff are read (drafts), other MDW folders not;
    their lineages are M1.. (index_main walks them apart, so no move or copy crosses into a card folder)."""
    assert oi.is_mdw("MDW/Croissant's Special Folder/t beth.psd") and oi.is_card("MDW/Ech's stuff/OlimarCard.psd")
    assert not oi.is_card("MDW/Templates/Creature_Template.psd") and not oi.is_mdw("Creatures/Isaac/3 Stars/T.png")
    meta = {"A": {"parents": [], "repo": "old"}}
    _, lineages, events, _ = oi.walk(["A"], meta, {"A": {"MDW/Croissant's Special Folder/peak.psd": "b"}}, prefix="M")
    assert list(lineages) == ["M1"] and events[0]["kind"] == "add"
