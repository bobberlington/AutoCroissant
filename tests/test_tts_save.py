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
