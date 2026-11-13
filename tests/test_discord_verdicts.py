"""discord_verdicts on a made-up archive whose pairs.json is given: one verdict per picture, jokes, paper cards, lost
cards, sheets."""
import json
from pathlib import Path

import commands.discord_match as dm
import commands.discord_verdicts as dv

G = "100000000000000001"
T = 1600000000
PORT, NEW, MEME = ("735328279619436666", "port"), ("875055733161488444", "new-cards"), ("835450716947873823",
                                                                                     "meme-cards")
H = {"p": "0" * 16, "d": "0" * 16, "q": "0" * 64}   # one look's hashes
IDEAS, PAPER = ("664256411907391509", "ideas-discussion"), ("1088034453319843840", "record-book")


def sha(n: int) -> str:
    return f"{n:02x}" * 32


def snowflake(seconds: float) -> str:
    return str((int(seconds * 1000) - 1420070400000) << 22)


def seen(card, keys, by="text", **s):
    return [{"card": card, "s": s or {"hash": 1.0, "title": 1.0}, "why": {"title": "a long printed title",
                                                                       "pick": {"keys": keys, "by": by}}}]


def archive(root: Path, posts: list, cards: dict, versions: list, kinds: dict, old_lost=(), old_map=None):
    """posts: (seconds, (channel id, name), picture number, candidates, extra image fields, thread)."""
    for sub in ("messages", "attachments", "derived/matcher/match", "derived/catalogue", "derived/oldgit"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    lines, images = [], {}
    for n, (when, (channel, name), pic, candidates, extra, thread) in enumerate(posts):
        mid = snowflake(when)
        digest = sha(pic)
        lines.append({"guild_id": G, "channel_id": channel, "channel_name": name, "thread": thread, "id": mid,
                      "timestamp": dm.iso(when), "author": {"id": "1", "username": "x", "bot": False},
                      "content": extra.pop("_content", ""), "embeds": [], "referenced_message_id": None,
                      "attachments": [{"id": str(n), "filename": "image.png", "sha256": digest,
                                       "file": f"attachments/{digest}.png"}], "jump_url": f"https://x/{mid}"})
        images[f"{mid}:{n}"] = {"key": f"{mid}:{n}", "message": mid, "media": str(n), "sheet": None,
                                "file": f"attachments/{digest}.png", "sha256": digest, "filename": "image.png",
                                "kind": "attachment", "shape": "2:3", "ocr": True, "whole": H, "text": "",
                                "title": [], "lines": [], "candidates": candidates, **extra}
    for channel in {r["channel_id"] for r in lines}:
        (root / "messages" / f"{channel}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in lines
                                                                    if r["channel_id"] == channel))
    (root / "channels.json").write_text(json.dumps([{"id": r["channel_id"], "name": r["channel_name"]}
                                                    for r in lines]))
    (root / "derived/catalogue/channels.json").write_text(json.dumps({"channels": [
        {"id": i, "kind": k} for i, k in kinds.items()]}))
    (root / "derived/oldgit/lost.json").write_text(json.dumps({"cards": list(old_lost)}))
    (root / "derived/oldgit/map.json").write_text(json.dumps({"cards": old_map or {}}))
    (root / "derived/matcher/match/pairs.json").write_text(json.dumps({"version": dm.PAIRS_VERSION, "cards": cards,
                                                                       "versions": versions, "images": images}))


def rows_of(root: Path) -> dict:
    return {r["id"]: r for r in map(json.loads, (root / "derived/match/verdicts.jsonl").read_text().splitlines())}


def test_one_verdict_a_picture_jokes_follow_the_picture_paper_never_dates(tmp_path):
    cards = {"fang": {"id": "fang", "name": "Fang", "key": "Fang", "fs": T + 400 * 86400}}
    versions = [{"key": "now", "card": "fang", "commit": "c2", "ts": T + 86400, "current": True, "look": 0,
                 "whole": H}, {"key": "old", "card": "fang", "commit": "c1", "ts": T + 86400, "current": False,
                               "look": 1, "whole": H}]
    joke_title = {"title": ["Fang But Funny"], "title_block": "Fang But Funny",
                  "lines": ["Fang But Funny", "Deal 5 damage to every creature on the field and laugh about it"]}
    old_stuff = {"id": "1331913819563360288", "name": "Old stuff thread"}
    posts = [(T + 5 * 86400, NEW, 1, seen("fang", ["old"]), {}, None),      # 1: reposted; the port's post first
             (T, PORT, 1, seen("fang", ["old"]), {}, None),
             (T + 86400, IDEAS, 2, [], dict(joke_title), None),             # 2: a joke first posted in ideas
             (T + 2 * 86400, MEME, 2, [], dict(joke_title), None),
             (T + 3 * 86400, PORT, 3, seen("fang", ["now"], by="hash"),          # 3: a port picture, a meme too;
              {"twins": [[f"{snowflake(T)}:1", 1.0, 1.0]]}, None),              # another look: no repost
             (T + 4 * 86400, MEME, 3, seen("fang", ["now"], by="hash"), {}, None),
             (T - 86400, MEME, 4, seen("fang", [], by="draft"), {}, None),  # 4: a meme of Fang, no git look
             (T + 900 * 86400, PAPER, 5, seen("fang", ["old"]), {}, old_stuff),     # 5: a paper card's photo
             (T + 7 * 86400, MEME, 6, [], {"lines": ["when the"]}, None)]        # 6: a meme, not a card
    root = tmp_path / "archive"
    archive(root, posts, cards, versions, {PORT[0]: "port", MEME[0]: "meme/joke"})
    dv.build(str(root), log=lambda *a: None, cards_repo="")
    rows = rows_of(root)
    one, two, three, four, five = (rows[dv.picture_id(sha(n))] for n in range(1, 6))
    assert (one["verdict"], one["card"], one["first"]["channel"], len(one["posts"])) == ("version_sure", "fang",
                                                                                         "port", 2)
    assert one["first"]["date"].startswith("2020-09-13") and one["version"]["commits"] == ["c1"]
    assert two["verdict"] == "morgue" and two["joke"] and two["review"] == "morgue"   # the ideas post too
    assert three["verdict"] == "version_sure" and three["meme_repost"] and three["first"]["channel"] == "port"
    assert four["verdict"] == "morgue" and four["looks_like"] == ["fang"] and four.get("first") is None
    assert five["verdict"] == "version_sure" and five["paper"] and five.get("first") is None    # never dates
    assert "repost_of" not in three and rows[dv.picture_id(sha(6))]["verdict"] == "none"
    groups = json.loads((root / "derived/match/lost_groups.json").read_text())["groups"]
    joke = next(g for g in groups if g["title"])
    assert joke["id"] == dv.group_id(sha(2)) and joke["kinds"] == ["idea", "joke"] and joke["morgue_candidate"]
    assert joke["first"]["channel"] == "ideas-discussion"
    for row in rows.values():
        assert "content" not in json.dumps(row) and "username" not in json.dumps(row)


def test_lost_cards_group_by_title_and_lookalikes_need_their_title(tmp_path):
    cards = {c: {"id": c, "name": n, "key": n} for c, n in (("dull-knife", "Dull Knife"), ("rusty-knife",
                                                                                         "Rusty Knife"))}
    text = "Equip: the wearer deals 2 more damage on each of its attacks this turn"

    def lost(title):
        return {"title": [title], "title_block": title, "lines": [title, text]}
    posts = [(T, IDEAS, 1, [], lost("Blankett"), None),
             (T, IDEAS, 8, [], lost("DAMAGE TYPES"), None),                     # rules pages and app screens
             (T, IDEAS, 9, [], {"title": ["Sign up"], "lines": ["Sign up", "Pull requests", text]}, None),
             (T + 7 * 86400, NEW, 10, [], lost("TWIN PRINCES"), None),         # the old split card
             (T + 86400, PORT, 2, [], lost("Blanket"), None),
             (T + 2 * 86400, IDEAS, 3, [], lost("Wind Dancer"), None),
             (T + 5 * 86400, IDEAS, 6, [], lost("Old Dull Blade"), None),
             (T + 6 * 86400, IDEAS, 7, [], {"title": ["White Duston"], "title_block": "White Duston",
                                            "lines": ["White Duston", "[FIEND]", text, "ATK/ 0 DEF/1000"]}, None),
             (T + 3 * 86400, NEW, 4, seen("dull-knife", [], by="hash", hash=1.0, ocr=1.0), {}, None),
             (T + 4 * 86400, NEW, 5, seen("dull-knife", [], by="hash", hash=1.0, title=1.0), {}, None)]
    root = tmp_path / "archive"
    archive(root, posts, cards, [], {PORT[0]: "port"}, [{"id": "OL-ddd1c85bc7", "old_id": "Blanket"}],
            {"Old Dull Blade": {"status": "live", "confidence": "sure", "key": "Dull Knife"}})
    dv.build(str(root), log=lambda *a: None, cards_repo="")
    groups = {g["title"]: g for g in json.loads((root / "derived/match/lost_groups.json").read_text())["groups"]}
    blanket, wind = groups["Blankett"], groups["Wind Dancer"]
    assert blanket["id"] == dv.group_id(sha(1)) and blanket["count"] == 2
    assert blanket["old_lost"] == "OL-ddd1c85bc7"
    assert blanket["morgue_candidate"] and "played" in blanket["kinds"]
    assert wind["ideas_never_ported"] and wind["kinds"] == ["idea", "lost"]
    assert "DAMAGE TYPES" not in groups and "Sign up" not in groups
    twin = groups["TWIN PRINCES"]
    assert "split" in twin["kinds"] and not twin["morgue_candidate"] and twin["split_of"]["into"] == ["lothric",
                                                                                                    "lorian"]
    blade = groups["Old Dull Blade"]                                           # renamed? a hint, no Morgue
    assert blade["old_name_of"] == "dull-knife" and not blade["morgue_candidate"] and "White Duston" not in groups
    rows = rows_of(root)
    assert rows[dv.picture_id(sha(4))]["verdict"] == "unsure"           # a look-alike without its title
    assert "look-alike" in rows[dv.picture_id(sha(4))]["reason"]
    assert rows[dv.picture_id(sha(5))]["verdict"] == "version_unsure"


def test_sheet_cells_are_pictures(tmp_path):
    (tmp_path / "derived/catalogue").mkdir(parents=True)
    rows = [{"file": f"attachments/{'a' * 64}.png", "sheet": "pair", "seam_grid": [1, 2]},
            {"file": f"attachments/{'b' * 64}.png", "sheet": "no", "seam_grid": [1, 3]},
            {"file": f"attachments/{'c' * 64}.png", "sheet": None, "seam_grid": [1, 2]}]
    (tmp_path / "derived/catalogue/pictures.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert dm.catalogue_sheets(str(tmp_path)) == {"a" * 64: (1, 2), "b" * 64: None}
    cards = {"fang": {"id": "fang", "name": "Fang", "key": "Fang"}}
    cells = [{"box": [0, 0, 0.5, 1], "shape": "2:3", "w": 300, "h": 450, "title": ["Fang"], "lines": ["Fang"],
              "candidates": seen("fang", [], by="draft")},
             {"box": [0.5, 0, 1, 1], "shape": "2:3", "w": 300, "h": 450, "title": [], "lines": [], "candidates": []}]
    root = tmp_path / "archive"
    archive(root, [(T, NEW, 7, [], {"cells": cells, "sheet": [1, 2], "shape": None}, None)], cards, [], {})
    dv.build(str(root), log=lambda *a: None, cards_repo="")              # cut by names alone: no look claimed
    assert rows_of(root)[dv.picture_id(sha(7), 1)]["verdict"] == "version_unsure"
    (root / "derived/catalogue/pictures.jsonl").write_text(json.dumps(
        {"file": f"attachments/{sha(7)}.png", "sheet": "pair", "seam_grid": [1, 2]}) + "\n")
    dv.build(str(root), log=lambda *a: None, cards_repo="")
    found = rows_of(root)
    assert [found[dv.picture_id(sha(7), n)]["verdict"] for n in (1, 2)] == ["draft", "none"]
    sheets = json.loads((root / "derived/match/sheets/sheets.json").read_text())["sheets"]
    assert [c["id"] for c in sheets[0]["cells"]] == [dv.picture_id(sha(7), 1), dv.picture_id(sha(7), 2)]


def test_a_git_look_needs_the_whole_card_and_its_text():
    far = {"p": "f" * 16, "d": "f" * 16, "q": "f" * 64}
    versions = {"a": {"key": "a", "card": "c", "look": 0, "whole": H, "ts": T},
                "b": {"key": "b", "card": "c", "look": 1, "whole": far, "ts": T}}
    members = list(versions.values())
    same = {"by": "text", "keys": ["a"], "fit": 0.95, "novel": 2}
    assert dv.decide(same, {"shape": "2:3", "whole": H}, versions, members)[0] == "git"
    assert dv.decide({**same, "novel": 20}, {"shape": "2:3", "whole": H}, versions, members)[0] is None
    assert dv.decide({**same, "missing": 15}, {"shape": "2:3", "whole": H}, versions, members)[0] is None
    other = {"p": "0f" * 8, "d": "0f" * 8, "q": "0f" * 32}               # 32 bits from both looks
    assert dv.decide(same, {"shape": "2:3", "whole": other}, versions, members)[0] is None
    tied = {"by": "text", "keys": ["a", "b"], "fit": 0.5, "novel": 30}
    kind, note = dv.decide(tied, {"shape": "2:3", "whole": other}, versions, members)
    assert kind == "draft" and note["far"] >= dv.FAR_BITS
    later = {"shape": "2:3", "whole": H, "posted": T + 10, "unstored": [T + 5]}
    assert dv.decide(same, later, versions, members)[0] is None              # an unstored version came first


def test_a_look_before_git_needs_its_bars_and_numbers():
    day, bars, tags = 86400, {"hp": 7, "def": 6, "atk": 5, "spd": 5}, dv.marks(["(2CD) summon it for 2 turns"])
    v = {"key": "a", "card": "c", "look": 0, "whole": H, "ts": T + 100 * day, "bars": bars, "marks": (tags, tags)}
    pick = {"by": "text", "keys": ["a"], "fit": 1.0, "novel": 0}

    def kind(members=(v,), **image):
        return dv.decide(pick, {"shape": "1:2", "whole": H, **image}, {m["key"]: m for m in members},
                         list(members))[0]
    assert tags == {"t:2cd", "n:2:c", "n:2:t"} and dv.marks(["(OPT)(OYT) for 3 turns"]) == {"t:opt", "t:oyt", "n:3:t"}
    assert kind(made=T, marks=tags, bars=bars) == "git"                        # 100 days early, read the same
    assert kind(made=T, marks=tags, bars={**bars, "def": 5}) is None          # another stat (Kalecgos)
    assert kind(made=T, marks={"t:e2t", "n:2:t"}, bars=bars) is None          # another timing tag (Gates)
    assert kind(made=T, marks={"t:2cd", "n:2:c", "n:3:t"}, bars=bars) is None  # another number
    assert kind(made=T, marks=tags) is None and kind(made=T + 98 * day, marks=tags) == "git"   # bars unread: grace
    assert dv.decide({**pick, "missing": 3}, {"shape": "1:2", "whole": H, "made": T + 98 * day, "marks": tags},
                     {"a": v}, [v])[0] is None                                 # a word fixed before the commit
    assert kind(made=T, marks=tags, no_bars=True) == "git" and kind(made=T, no_bars=True) is None
    later = {**v, "key": "b", "ts": T + 200 * day, "bars": {**bars, "atk": 6}}    # one look, two stat sets
    assert kind((v, later), made=T + 300 * day, marks=tags) is None
    assert kind((v, later), made=T + 300 * day, marks=tags, bars=later["bars"]) == "git"


def test_changelog_days_name_a_card_and_lost_ids_are_unique():
    log = [{"timestamp": "2020-08-01T00:00:00", "content": "Buffed Retro Kirby"},
           {"timestamp": "2020-08-02T00:00:00", "content": "Kirby nerfed, actions added"}]
    assert dv.changelog_dates("KIRBY", log, ("retro kirby",)) == ["2020-08-02"]
    assert dv.changelog_dates("Actions", log, (), frozenset({"actions"})) == []
    assert dv.group_id(sha(3)) != dv.group_id(sha(3), 1) != dv.group_id(sha(3), 2)
