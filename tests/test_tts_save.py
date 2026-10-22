"""tts_save: byte-identical serializer, _sync_decks and _fix_keys on a small forged save; the Action's env overrides."""
from copy import deepcopy
from json import dumps, loads
from os import environ
from subprocess import run
from sys import executable
from types import SimpleNamespace

import pytest

import commands.repo_history as rh
import commands.tts_save as tts
from conftest import CLONE, REPO

RAW = tts.RAW
HEAL = "Items/Heal/2 Stars/"
A, B, C, X, GONE = HEAL + "A.png", HEAL + "B.png", HEAL + "C c.png", "Items/Other/X.png", HEAL + "Gone.png"
BACK = "MDW/cardback.png"


# ========================
# Serializer
# ========================
def test_serializer_round_trip_real_save():
    path = CLONE / tts.TTS_SAVE_PATH
    if not path.exists():
        pytest.skip(f"no {path}")
    raw = path.read_bytes()
    assert tts._dump(loads(raw.decode("utf-8"))).encode("utf-8") == raw


def test_serializer_tts_style():
    assert tts._dump({"a": 1.0, "b": 1.00179122e-05, "c": [True, None], "d": {}, "e": [], "f": "é "}) == (
        '{\n  "a": 1.0,\n  "b": 1.00179122E-05,\n  "c": [\n    true,\n    null\n  ],\n  "d": {},\n  "e": [],\n'
        '  "f": "é\\u2028"\n}')


# ========================
# Forged save
# ========================
class Names:
    def of(self, path):
        return "Name of " + tts._stem(path)


@pytest.fixture
def repo(monkeypatch):
    """A real _Repo built without git: tree from a dict, B pinned through repo_history."""
    tree = {A: "ba", B: "bb", C: "bc", X: "bx", BACK: "bk", "Items/Heal/2 Stars/A.psd": "pa"}
    monkeypatch.setattr(rh, "_paths", {B: ("bb", "c0ffee1"), BACK: ("bk", "bac0001")})
    r = object.__new__(tts._Repo)
    r.clone = r.checkout = "/nonexistent"
    r.history, r.tree, r.head, r.source, r._at = True, tree, "head", "test", {}
    r.pngs = sorted(p for p in tree if p.endswith(".png"))
    r.by_blob = {b: [p] for p, b in tree.items()}
    r.by_lower = {p.lower(): p for p in tree}
    return r


def _card(guid, path, card_id, **extra):
    url = f"{RAW}main/{tts._encode(path)}"
    face = {"FaceURL": url, "BackURL": f"{RAW}main/{BACK}", "NumWidth": 1, "NumHeight": 1, "BackIsHidden": True}
    return {"GUID": guid, "Name": "CardCustom", "Nickname": tts._stem(path), "Description": "damage", "CardID": card_id,
            "CustomDeck": {str(card_id // 100): face}, "LuaScript": "print(1)", "Tags": ["x"], **extra}


def _deck(guid, cards):
    return {"GUID": guid, "Name": "Deck", "Nickname": guid, "ContainedObjects": cards,
            "DeckIDs": [c["CardID"] for c in cards],
            "CustomDeck": {k: v for c in cards for k, v in c["CustomDeck"].items()}}


@pytest.fixture
def save():
    heal = _deck("aaaaaa", [_card("000001", A, 100), _card("000002", X, 200), _card("000003", GONE, 300)])
    return {"ObjectStates": [heal, _deck("bbbbbb", [_card("000004", A, 400)]),
                             _deck("cccccc", [_card("000005", A, 500)]), _deck("dddddd", [_card("000006", A, 600)])]}


RULES = {"decks": {
    "aaaaaa": {"rule": {"kind": "items", "folders": [HEAL + "*.png"], "count": 1}, "odd_counts": {C: 2}, "back": BACK},
    "cccccc": {"rule": {"kind": "items", "folders": [HEAL + "*.png"], "count": 1, "partial": True}},
    "dddddd": {"rule": {"kind": "items", "folders": ["Nowhere/*.png"], "count": 1}}}}


def test_sync_decks(save, repo):
    report = tts._Report()
    untouched = deepcopy(save["ObjectStates"][1:])
    tts._sync_decks(save, RULES, repo, Names(), report)
    deck = save["ObjectStates"][0]
    cards = deck["ContainedObjects"]
    assert [tts._face_path(c) for c in cards] == [A, GONE, B, C, C]      # X left (outside the rule), gone image kept
    new = cards[2:]
    assert [c["Nickname"] for c in new] == ["Name of B", "Name of C c", "Name of C c"]
    assert new[0]["CardID"] == 700 and new[1]["CardID"] == new[2]["CardID"] == 800   # after the highest id (6)
    assert len({c["GUID"] for c in cards}) == 5 and all(len(c["GUID"]) == 6 for c in new)
    assert all(c["Description"] == "" and c["LuaScript"] == "" and "Tags" not in c for c in new)
    assert new[0]["CustomDeck"]["7"]["FaceURL"] == f"{RAW}c0ffee1/Items/Heal/2%20Stars/B.png"     # pinned
    assert new[1]["CustomDeck"]["8"]["FaceURL"] == f"{RAW}main/Items/Heal/2%20Stars/C%20c.png"
    assert new[0]["CustomDeck"]["7"]["BackURL"] == f"{RAW}bac0001/{BACK}"
    assert sorted(deck["CustomDeck"]) == ["1", "3", "7", "8"]
    assert deck["DeckIDs"] == [c["CardID"] for c in cards]
    assert report.unpinned == {C}
    assert [(n, p, k) for _, added in report.added for n, p, k in added] == [("Name of B", B, 1), ("Name of C c", C, 2)]
    assert [(n, p, k) for _, removed in report.removed for n, p, k in removed] == [("X", X, 1)]
    assert save["ObjectStates"][1:] == untouched                          # no rule, hand-picked, matches nothing
    assert any("no rule" in x for x in report.left) and any("hand-picked" in x for x in report.left)
    assert any("matches no repo image" in x for x in report.errors)

    again = deepcopy(save)
    tts._sync_decks(again, RULES, repo, Names(), tts._Report())
    assert again == save                                                  # complete now: nothing to do


def test_sync_decks_keeps_non_png_and_skips_bad_entries(save, repo):
    jpg = HEAL + "J.jpg"
    repo.tree[jpg] = "bj"
    deck = save["ObjectStates"][0]
    deck["ContainedObjects"].append(_card("000007", jpg, 700))
    rules = deepcopy(RULES)
    rules["decks"]["bbbbbb"] = {"rule": "oops"}                           # malformed: only its own deck is skipped
    before = deepcopy(save["ObjectStates"][1])
    report = tts._Report()
    tts._sync_decks(save, rules, repo, Names(), report)
    assert jpg in [tts._face_path(c) for c in deck["ContainedObjects"]]   # rules only cover PNGs: never removed
    assert [(n, p, k) for _, removed in report.removed for n, p, k in removed] == [("X", X, 1)]
    assert save["ObjectStates"][1] == before and any("bad entry" in x for x in report.errors)
    with pytest.raises(ValueError):
        tts._decks([])


def test_sync_decks_drops_unused_keys(repo):
    """A card taken out of a complete deck leaves its key behind; the rewrite drops it (UPD-4)."""
    deck = _deck("eeeeee", [_card("000011", A, 100)])
    deck["CustomDeck"]["5"] = deepcopy(deck["CustomDeck"]["1"])          # the removed card's key
    save = {"ObjectStates": [deck]}
    rules = {"decks": {"eeeeee": {"rule": {"kind": "items", "folders": [HEAL + "*.png"], "count": 1}}}}
    tts._sync_decks(save, rules, repo, Names(), tts._Report())
    assert [tts._face_path(c) for c in deck["ContainedObjects"]] == [A, B, C]
    assert sorted(deck["CustomDeck"]) == ["1", "6", "7"]
    assert set(deck["CustomDeck"]) == {str(c["CardID"] // 100) for c in deck["ContainedObjects"]}


def test_pin_links_reports_pinned_and_repinned(repo):
    """Links on main are 'Links pinned', links on an older sha 'Links re-pinned (changed files)' (UPD-3)."""
    tile = {"Name": "Custom_Tile", "CustomImage": {"ImageURL": f"{RAW}main/{tts._encode(B)}",
                                                   "ImageSecondaryURL": f"{RAW}0dd0001/{BACK}"}}
    save = {"ObjectStates": [tile]}
    report = tts._Report()
    tts._pin_links(save, repo, report)
    assert tile["CustomImage"] == {"ImageURL": f"{RAW}c0ffee1/Items/Heal/2%20Stars/B.png",
                                   "ImageSecondaryURL": f"{RAW}bac0001/{BACK}"}
    text = tts._render(report, [])
    assert f"## Links pinned (1)\n- `{B}`: main -> `c0ffee1` (1 link)" in text
    assert f"## Links re-pinned (changed files) (1)\n- `{BACK}`: `0dd0001` -> `bac0001` (1 link)" in text
    assert tts._commit_message(report) == "TTS save: 2 links pinned"


def test_commit_message_counts_names_and_lua():
    """Distinct names, not cards; a resizeTargets change is named."""
    report = tts._Report()
    report.names_real[("a.png", "Old", "New")] = 3
    report.names_case[("b.png", "chuckya", "Chuckya")] = 2
    report.lua_added.append("New")
    assert tts._commit_message(report) == "TTS save: 2 Nicknames fixed, resizeTargets updated"


def test_update_writes_commit_subject(save, repo, tmp_path, monkeypatch):
    """A changed save gets <stem>.message.txt, _commit_message's line, for tts-save.yml's commit (UPD-2)."""
    names = Names()
    names.source, names.site = "test", {}
    monkeypatch.setattr(tts, "_Repo", lambda errors: repo)
    monkeypatch.setattr(tts, "_Names", lambda repo, errors, edits=None: names)
    monkeypatch.setattr(tts, "_load_rules", lambda repo, errors, given=None: (RULES, "test"))
    path = tmp_path / "save.json"
    path.write_text(tts._dump(save), encoding="utf-8")
    text, report = tts._update(str(path))
    message = tmp_path / "save.message.txt"
    assert report.changed and message.read_text(encoding="utf-8") == tts._commit_message(report) + "\n"
    assert tts._commit_message(report) == "TTS save: 12 links pinned, 2 cards added, 1 card removed"
    assert (tmp_path / "save.report.md").read_text(encoding="utf-8") == text
    message.unlink()
    _, again = tts._update(str(path))
    assert not again.changed and not message.exists()                    # nothing changed: no subject


def test_fix_keys(repo):
    shown = {"FaceURL": f"{RAW}main/{tts._encode(A)}"}
    own = {"FaceURL": f"{RAW}main/{tts._encode(B)}"}
    stale = {"GUID": "e1", "Name": "Card", "Nickname": "A", "CardID": 500, "CustomDeck": {"9": own}}
    fine = {"GUID": "e2", "Name": "Card", "Nickname": "A", "CardID": 500, "CustomDeck": {"5": shown}}
    loose = {"GUID": "e3", "Name": "CardCustom", "Nickname": "B", "CardID": 700, "CustomDeck": {"3": own}}
    deck = {"GUID": "dk", "Name": "Deck", "CustomDeck": {"5": shown}, "ContainedObjects": [stale, fine]}
    other = {"GUID": "o1", "Name": "Custom_Model", "CardID": 100, "CustomDeck": {"9": own}}
    save = {"ObjectStates": [deck, loose, other]}
    report = tts._Report()
    tts._fix_keys(save, report)
    assert stale["CustomDeck"] == {"5": shown} and fine["CustomDeck"] == {"5": shown}   # the deck's entry wins
    assert loose["CustomDeck"] == {"7": own} and other["CustomDeck"] == {"9": own}
    assert report.stale == ["`e1` A: 9 -> 5; own face differed, the deck's kept", "`e3` B: 3 -> 7"]


# ========================
# Environment overrides
# ========================
def _paths(env: dict) -> list:
    code = "import global_config as g; print(g.LOCAL_DIR_LOC); print(g.HISTORY_CLONE)"
    return run([executable, "-c", code], cwd=REPO, env=env, capture_output=True, text=True, check=True).stdout.split("\n")[:2]


def test_env_overrides_for_the_action():
    """TTSCardMaker's tts-save.yml points the checkout and the history clone elsewhere through the environment."""
    env = {k: v for k, v in environ.items() if not k.startswith("AUTOCROISSANT_")}
    assert _paths(env) == ["~/Desktop/TTSCardMaker", "~/Documents/TTSCardMaker-history.git"]
    env.update(AUTOCROISSANT_LOCAL_DIR="/work/cards", AUTOCROISSANT_HISTORY_CLONE="/work/history.git")
    assert _paths(env) == ["/work/cards", "/work/history.git"]


# ========================
# Card Edits.json: read from site/
# ========================
def _checkout(tmp_path, **layouts):
    root = tmp_path / "checkout"
    for folder, name in layouts.items():
        (root / folder).mkdir(parents=True, exist_ok=True)
        (root / folder / "Card Edits.json").write_text(dumps({"cards": {"A": {"name": name}}}), encoding="utf-8")
    return str(root)


def test_names_read_card_edits_from_site(repo, tmp_path):
    assert tts.EDITS_PATHS == ("site/Card Edits.json",)
    repo.checkout = _checkout(tmp_path, site="Alpha on the site")
    names = tts._Names(repo, [])                                      # offline: the website is unreachable
    assert names.known(A) == "Alpha on the site" and names.of(B) == "B"
    repo.checkout = str(tmp_path / "empty")
    assert tts._Names(repo, []).known(A) is None


def test_files_at_lists_the_edits_path(monkeypatch):
    """update_repo_save asks main's tree for every EDITS_PATHS entry and reads the first one found (none without site/)."""
    blob = lambda name, sha: {"path": name, "type": "blob", "mode": "100644", "sha": sha}
    sub = lambda name, sha: {"path": name, "type": "tree", "mode": "040000", "sha": sha}
    trees = {"r1": [sub("TTS", "t1"), sub("site", "s1")], "r2": [sub("TTS", "t1")],
             "t1": [blob("decks.json", "d"), blob("Impossibility Simulator.json", "v")],
             "s1": [blob("Card Edits.json", "e"), blob("index.html", "h")]}
    monkeypatch.setattr(tts, "_api", lambda method, path, **kw: SimpleNamespace(json=lambda: {"tree": trees[path.rsplit("/", 1)[1]]}))
    monkeypatch.setattr(tts, "_objects", {})
    monkeypatch.setattr(tts, "_used", set())
    wanted = (tts.TTS_SAVE_PATH, tts.RULES_PATH) + tts.EDITS_PATHS
    files = tts._files_at("r1", wanted)
    assert {p: s for p, (_, s) in files.items()} == {tts.TTS_SAVE_PATH: "v", tts.RULES_PATH: "d", "site/Card Edits.json": "e"}
    assert next(p for p in tts.EDITS_PATHS if p in files) == "site/Card Edits.json"
    files = tts._files_at("r2", wanted)
    assert {p: s for p, (_, s) in files.items()} == {tts.TTS_SAVE_PATH: "v", tts.RULES_PATH: "d"}
    assert next((p for p in tts.EDITS_PATHS if p in files), None) is None


# ========================
# Empty decks, add_new_since, team fields
# ========================
def test_sync_decks_fills_an_empty_deck(save, repo):
    """An empty deck with a counted rule is filled from another deck's card, at its own Transform, keys after DeckIDs."""
    empty = {"GUID": "ffffff", "Name": "Deck", "Transform": {"posX": 9.0, "scaleX": 2.0}, "Nickname": "Heals",
             "DeckIDs": [], "LuaScript": "", "ContainedObjects": []}
    save["ObjectStates"][0]["ContainedObjects"][0]["Memo"] = "team note"
    save["ObjectStates"].append({"GUID": "bag001", "Name": "Bag", "ContainedObjects": [empty]})
    rules = {"decks": {"ffffff": {"rule": {"kind": "items", "folders": [HEAL + "*.png"], "count": 1}, "back": BACK}}}
    report = tts._Report()
    tts._sync_decks(save, rules, repo, Names(), report)
    cards = empty["ContainedObjects"]
    assert [tts._face_path(c) for c in cards] == [A, B, C]
    assert all(c["Transform"] == {"posX": 9.0, "scaleX": 2.0} for c in cards)
    assert all(c["Description"] == "" and c["LuaScript"] == "" and "Tags" not in c and "Memo" not in c for c in cards)
    assert empty["DeckIDs"] == [c["CardID"] for c in cards] == [700, 800, 900]
    assert list(empty)[list(empty).index("DeckIDs") + 1] == "CustomDeck" and sorted(empty["CustomDeck"]) == ["7", "8", "9"]
    assert all(c["CustomDeck"][str(c["CardID"] // 100)]["BackURL"] == f"{RAW}bac0001/{BACK}" for c in cards)
    assert report.added and not any("ffffff" in x for x in report.left)


def test_new_cards_reach_full_and_developer_decks(repo):
    """A new card joins the complete deck and, once, the hand-picked add_new_since deck; an old missing one only the
    complete deck; the hand-picked deck never loses cards."""
    full = _deck("aaaaaa", [_card("000001", A, 100)])
    dev = _deck("bbbbbb", [_card("000002", A, 200), _card("000003", X, 300)])
    save = {"ObjectStates": [full, dev]}
    repo.carried = lambda sha: {A, B} if sha == "base" else None      # B was in the save at base, C is new
    rules = {"decks": {
        "aaaaaa": {"rule": {"kind": "items", "folders": [HEAL + "*.png"], "count": 1}},
        "bbbbbb": {"rule": {"kind": "items", "folders": [HEAL + "*.png"], "count": 1, "partial": True,
                            "add_new_since": "base"}}}}
    report = tts._Report()
    tts._sync_decks(save, rules, repo, Names(), report)
    assert [tts._face_path(c) for c in full["ContainedObjects"]] == [A, B, C]
    assert [tts._face_path(c) for c in dev["ContainedObjects"]] == [A, X, C]
    assert dev["ContainedObjects"][2]["CardID"] == full["ContainedObjects"][2]["CardID"]   # one key per image
    assert [(p, k) for _, added in report.new for _, p, k in added] == [(C, 1)]
    assert "New cards added (hand-picked decks, add_new_since) (1)" in tts._render(report, [])
    again = deepcopy(save)
    tts._sync_decks(again, rules, repo, Names(), tts._Report())
    assert again == save
    repo.carried = lambda sha: None                                   # that save unreadable: deck left alone
    report = tts._Report()
    tts._sync_decks(deepcopy(save), rules, repo, Names(), report)
    assert any("can't be read" in x for x in report.errors)


def test_star_move_changes_developer_deck(repo):
    """A card moved to other stars leaves its add_new_since deck for the one covering it, object and fields kept."""
    three, four = "Items/Heal/3 Stars/", "Items/Heal/4 Stars/"
    moved = four + "M.png"
    repo.tree[moved] = repo.tree[three + "K.png"] = repo.tree[four + "L.png"] = "bm"
    repo.pngs = sorted(p for p in repo.tree if p.endswith(".png"))
    card = _card("000002", moved, 200, Transform={"posX": 1.0}, GMNotes="team")
    dev3 = _deck("aaaaaa", [_card("000001", three + "K.png", 100), card])
    dev4 = _deck("bbbbbb", [_card("000003", four + "L.png", 300, Transform={"posX": 4.0})])
    save = {"ObjectStates": [dev3, dev4]}
    repo.carried = lambda sha: set(repo.pngs)                         # nothing new
    rule = {"kind": "items", "count": 1, "partial": True, "add_new_since": "base"}
    rules = {"decks": {"aaaaaa": {"rule": {**rule, "folders": [three + "*.png"]}},
                       "bbbbbb": {"rule": {**rule, "folders": [four + "*.png"]}}}}
    report = tts._Report()
    tts._sync_decks(save, rules, repo, Names(), report)
    assert [c["GUID"] for c in dev3["ContainedObjects"]] == ["000001"] and dev3["DeckIDs"] == [100]
    assert sorted(dev3["CustomDeck"]) == ["1"] and sorted(dev4["CustomDeck"]) == ["2", "3"]
    assert dev4["ContainedObjects"][1] is card and dev4["DeckIDs"] == [300, 200]
    assert card["Transform"] == {"posX": 4.0} and card["GMNotes"] == "team" and card["Description"] == "damage"
    assert len(report.moved) == 1 and "1 card moved" in tts._commit_message(report)
    again = deepcopy(save)
    tts._sync_decks(again, rules, repo, Names(), tts._Report())
    assert again == save


def test_star_move_gets_a_fresh_key_when_the_target_uses_it(repo):
    """The target deck shows another image under the moved card's CustomDeck key: the card gets a new id, one for
    both copies."""
    three, four = "Items/Heal/3 Stars/", "Items/Heal/4 Stars/"
    moved = four + "M.png"
    repo.tree[moved] = repo.tree[three + "K.png"] = repo.tree[four + "L.png"] = "bm"
    repo.pngs = sorted(p for p in repo.tree if p.endswith(".png"))
    card, copy = _card("000002", moved, 200), _card("000004", moved, 200)
    face, other = deepcopy(card["CustomDeck"]["2"]), _card("000003", four + "L.png", 200)
    dev3, dev4 = _deck("aaaaaa", [_card("000001", three + "K.png", 100), card, copy]), _deck("bbbbbb", [other])
    save = {"ObjectStates": [dev3, dev4]}
    repo.carried = lambda sha: set(repo.pngs)
    rule = {"kind": "items", "count": 1, "partial": True, "add_new_since": "base"}
    rules = {"decks": {"aaaaaa": {"rule": {**rule, "folders": [three + "*.png"]}},
                       "bbbbbb": {"rule": {**rule, "folders": [four + "*.png"]}}}}
    tts._sync_decks(save, rules, repo, Names(), tts._Report())
    assert card["CardID"] == copy["CardID"] == 300 and card["CustomDeck"] == {"3": face} and dev4["DeckIDs"] == [200, 300, 300]
    assert dev4["CustomDeck"] == {"2": other["CustomDeck"]["2"], "3": face} and sorted(dev3["CustomDeck"]) == ["1"]


@pytest.mark.parametrize("case", ["other folder", "excluded", "rule matches nothing", "save unreadable"])
def test_star_move_leaves_hand_picked_cards(repo, case):
    """Only a card whose stars changed moves: one from another folder or excluded by the rule stays, and so does
    every card of a deck whose rule matches nothing or whose since save can't be read."""
    three, four, fire = "Items/Heal/3 Stars/", "Items/Heal/4 Stars/", "Items/Fire/4 Stars/"
    for p in (three + "K.png", four + "L.png", four + "M.png", fire + "F.png"):
        repo.tree[p] = "b" + p
    if case == "rule matches nothing":
        del repo.tree[three + "K.png"]
    repo.pngs = sorted(p for p in repo.tree if p.endswith(".png"))
    repo.carried = lambda sha: set(repo.pngs) if sha == "base" else None
    picked = fire + "F.png" if case == "other folder" else four + "M.png"
    rule = {"kind": "items", "count": 1, "partial": True, "add_new_since": "base"}
    rule3 = {**rule, "folders": [three + "*.png"]}
    if case == "excluded":
        rule3.update(folders=["Items/Heal/*/*.png"], exclude=[picked])
    if case == "save unreadable":
        rule3["add_new_since"] = "gone"
    dev3 = _deck("aaaaaa", [_card("000001", three + "K.png", 100), _card("000002", picked, 200)])
    save = {"ObjectStates": [dev3, _deck("bbbbbb", [_card("000003", four + "L.png", 300)]),
                             _deck("cccccc", [_card("000004", fire + "F.png", 400)])]}
    rules = {"decks": {"aaaaaa": {"rule": rule3}, "bbbbbb": {"rule": {**rule, "folders": [four + "*.png"]}},
                       "cccccc": {"rule": {**rule, "folders": [fire + "*.png"]}}}}
    report = tts._Report()
    tts._sync_decks(save, rules, repo, Names(), report)
    assert picked in [tts._face_path(c) for c in dev3["ContainedObjects"]] and report.moved == []


def test_carried_follows_renames(tmp_path, monkeypatch):
    """The save's cards at a commit, carried to today's paths through a rename (by blob)."""
    git = lambda *a: run(["git", "-C", str(tmp_path), *a], check=True, capture_output=True)
    git("init", "-q", "-b", tts.CARD_BRANCH)
    git("config", "user.email", "t@t"), git("config", "user.name", "t")
    (tmp_path / "TTS").mkdir()
    (tmp_path / "Old").mkdir()
    (tmp_path / "Old" / "Moved.png").write_bytes(b"moved")
    (tmp_path / "Old" / "Kept.png").write_bytes(b"kept")
    save = {"ObjectStates": [_deck("aaaaaa", [_card("000001", "Old/Moved.png", 100), _card("000002", "Old/Kept.png", 200)])]}
    (tmp_path / tts.TTS_SAVE_PATH).write_text(dumps(save), encoding="utf-8")
    git("add", "-A"), git("commit", "-q", "-m", "base")
    base = git("rev-parse", "HEAD").stdout.decode().strip()
    (tmp_path / "New").mkdir()
    git("mv", "Old/Moved.png", "New/Moved.png")
    (tmp_path / "New" / "Fresh.png").write_bytes(b"fresh")
    git("add", "-A"), git("commit", "-q", "-m", "move")
    monkeypatch.setattr(tts, "HISTORY_CLONE", str(tmp_path))
    monkeypatch.setattr(tts, "LOCAL_DIR_LOC", "/nonexistent")
    monkeypatch.setattr(tts, "refresh", lambda: True)
    repo = tts._Repo([])
    assert repo.carried(base) == {"Old/Moved.png", "New/Moved.png", "Old/Kept.png"}
    assert repo.carried("0" * 40) is None


def test_update_leaves_team_fields(save, repo, tmp_path, monkeypatch):
    """An existing card's Description (role tags too), GMNotes, Memo, Tags and scripts stay, repo links in them too."""
    card = save["ObjectStates"][0]["ContainedObjects"][0]
    card.update(Description="Tank damage", GMNotes=f"see {RAW}main/{tts._encode(B)}", Memo="m", LuaScript="x()")
    kept = {k: deepcopy(card.get(k)) for k in tts.TEAM_FIELDS}
    names = Names()
    names.source, names.site = "test", {}
    monkeypatch.setattr(tts, "_Repo", lambda errors: repo)
    monkeypatch.setattr(tts, "_Names", lambda repo, errors, edits=None: names)
    monkeypatch.setattr(tts, "_load_rules", lambda repo, errors, given=None: (RULES, "test"))
    path = tmp_path / "save.json"
    path.write_text(tts._dump(save), encoding="utf-8")
    text, report = tts._update(str(path), extras=False)
    after = loads(path.read_text(encoding="utf-8"))["ObjectStates"][0]["ContainedObjects"][0]
    assert report.changed and {k: after.get(k) for k in tts.TEAM_FIELDS} == kept
    assert "role tags" not in text
