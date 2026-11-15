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
FAR = {"p": "f" * 16, "d": "f" * 16, "q": "f" * 64}  # another's
IDEAS, PAPER = ("664256411907391509", "ideas-discussion"), ("1088034453319843840", "record-book")


def sha(n: int) -> str:
    return f"{n:02x}" * 32


def snowflake(seconds: float) -> str:
    return str((int(seconds * 1000) - 1420070400000) << 22)


def seen(card, keys, by="text", **s):
    return [{"card": card, "s": s or {"hash": 1.0, "title": 1.0}, "why": {"title": "a long printed title",
                                                                       "pick": {"keys": keys, "by": by}}}]


def archive(root: Path, posts: list, cards: dict, versions: list, kinds: dict, old_lost=(), old_map=None,
            texts=None):
    """posts: (seconds, (channel id, name), picture number, candidates, extra image fields, thread); texts: {version
    key: its OCR lines}."""
    for sub in ("messages", "attachments", "derived/matcher/match/cache", "derived/catalogue", "derived/oldgit"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    (root / "derived/matcher/match/cache/cards.jsonl").write_text("".join(
        json.dumps({"key": k, "lines": [[line, 1.0] for line in lines]}) + "\n" for k, lines in (texts or {}).items()))
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
    bites = {"text": dm.read_text([["Fang", 1.0], ["Bite for 2", 1.0]])}
    posts = [(T + 5 * 86400, NEW, 1, seen("fang", ["old"]), dict(bites), None),     # 1: reposted; the port's first
             (T, PORT, 1, seen("fang", ["old"]), dict(bites), None),
             (T + 86400, IDEAS, 2, [], dict(joke_title), None),             # 2: a joke first posted in ideas
             (T + 2 * 86400, MEME, 2, [], dict(joke_title), None),
             (T + 3 * 86400, PORT, 3, seen("fang", ["now"], by="hash"),          # 3: a port picture, a meme too;
              {"twins": [[f"{snowflake(T)}:1", 1.0, 1.0]]}, None),              # another look: no repost
             (T + 4 * 86400, MEME, 3, seen("fang", ["now"], by="hash"), {}, None),
             (T - 86400, MEME, 4, seen("fang", [], by="draft"), {"whole": FAR}, None),  # 4: a meme of Fang, no git look
             (T + 900 * 86400, PAPER, 5, seen("fang", ["old"]), {}, old_stuff),     # 5: a paper card's photo
             (T + 7 * 86400, MEME, 6, [], {"lines": ["when the"]}, None)]        # 6: a meme, not a card
    root = tmp_path / "archive"
    archive(root, posts, cards, versions, {PORT[0]: "port", MEME[0]: "meme/joke"},
            texts={"old": ["Fang", "Bite for 2"], "now": ["Fang", "Bite for 3"]})
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


def test_a_look_before_git_needs_its_bars_numbers_and_exact_text():
    day, bars, tags = 86400, {"hp": 7, "def": 6, "atk": 5, "spd": 5}, dv.marks(["(2CD) summon it for 2 turns"])
    words = dm.read_text([["(2CD) summon it for 2 turns", 1.0]])
    v = {"key": "a", "card": "c", "look": 0, "whole": H, "ts": T + 100 * day, "bars": bars, "marks": (tags, tags),
         "reads": [words, words]}
    pick = {"by": "text", "keys": ["a"], "fit": 1.0, "novel": 0}

    def kind(members=(v,), how=pick, **image):
        return dv.decide(how, {"shape": "1:2", "whole": H, "text": words, **image}, {m["key"]: m for m in members},
                         list(members))[0]
    assert tags == {"t:2cd", "n:2:c", "n:2:t"} and dv.marks(["(OPT)(OYT) for 3 turns"]) == {"t:opt", "t:oyt", "n:3:t"}
    assert kind(made=T, marks=tags, bars=bars) == "git"                        # 100 days early, read the same
    assert kind(made=T, marks=tags, bars={**bars, "def": 5}) is None          # another stat (Kalecgos)
    assert kind(made=T, marks={"t:e2t", "n:2:t"}, bars=bars) is None          # another timing tag (Gates)
    assert kind(made=T, marks={"t:2cd", "n:2:c", "n:3:t"}, bars=bars) is None  # another number
    assert kind(made=T, marks=tags) is None and kind(made=T + 98 * day, marks=tags) == "git"   # bars unread: grace
    short = dm.read_text([["(2CD) summon it for 2", 1.0]])
    assert kind(made=T + 98 * day, marks=tags, text=short) is None             # a word fixed before the commit
    assert kind(made=T + 98 * day, marks=tags, text="") is None                # no text read: can't tell
    assert kind(made=T + 101 * day, marks=tags, text=short) == "git"           # after git: the hash and marks
    halo = {"by": "hash", "keys": ["a"], "bits": 2.3}                          # a hash pick of a short text (Halo,
    assert kind(how=halo, made=T + 99.8 * day, text=short) is None             # Glued): exact at any length too
    assert kind(how=halo, made=T + 99.8 * day) == "git"
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


def test_named_only_needs_its_text_and_live_cards_are_never_morgue(tmp_path):
    far = FAR
    cards = {c: {"id": c, "name": n, "key": n} for c, n in (("potato", "Potato"), ("tentacle", "Tentacle"),
                                                             ("blooper-tentacle", "Blooper Tentacle"))}
    cards["fang"] = {"id": "fang", "name": "Fang", "key": "Fang", "fs": T + 400 * 86400}
    texts = {"pot": ["Potato", "Heal a creature by 2 and gain one apple"],
             "ten": ["Tentacle", "Protects the summoner and deals 3 damage to every lane"],
             "blo": ["Blooper Tentacle", "If more than one of this card is protecting a creature overkill carries"],
             "fng": ["Fang", "Bite for 2 damage on every attack"]}
    versions = [{"key": k, "card": c, "commit": k, "ts": T + 400 * 86400, "current": True, "look": 0,
                 "whole": H if k == "fng" else far} for k, c in (("pot", "potato"), ("ten", "tentacle"),
                                                                 ("blo", "blooper-tentacle"), ("fng", "fang"))]

    def named(card, key, fit, **others):
        return [{"card": card, "s": {"title": 1.0, "file": 1.0}, "why": {"title": "a long printed title",
                                                                         "pick": {"keys": [key], "by": "text",
                                                                                  "fit": fit}}},
                *({"card": c, "s": s, "why": {}} for c, s in others.items())]

    def reads(key, lines=None):
        lines = lines or texts[key]
        return {"text": dm.read_text([[t, 1.0] for t in lines]), "lines": list(lines), "whole": far}
    fang = seen("fang", ["fng"], by="hash")
    posts = [(T + 1, IDEAS, 1, named("potato", "pot", 0.11), reads("pot", ["Potato", "If used by michael all stats are "
                                                                      "quadrupled"]), None),     # a joke Potato
             (T + 2, IDEAS, 2, named("tentacle", "ten", 0.3, **{"blooper-tentacle": {"ocr": 0.8, "title": 0.5,
                                                                    "file": 0.5}}),
              reads("blo"), None),                                      # Blooper Tentacle's text, Tentacle's name
             (T + 3, NEW, 3, named("fang", "fng", 0.9), reads("fng"), None),     # named only: Created through review
             (T + 10 * 86400, NEW, 4, fang, reads("fng"), None),
             (T + 5, MEME, 5, fang, {**reads("fng"), "whole": H}, None),    # a live card's picture in a meme channel
             (T + 6, MEME, 6, fang, {**reads("fng", ["Fang", "vvvvvvv"]), "whole": H}, None),   # a joke on its look
             (T + 7, IDEAS, 7, [{"card": "fang", "s": {"line": 1.0, "file": 1.0}, "why": {}}],   # a screenshot named
              {**reads("fng"), "shape": None, "text": ""}, None),                         # so outside the memes
             (T + 86400, MEME, 7, [], {}, None),
             (T + 86400 + 1, MEME, 1, [], {}, None)]                    # a joke named as a live card: no Morgue
    root = tmp_path / "archive"
    archive(root, posts, cards, versions, {MEME[0]: "meme/joke"}, texts=texts)
    dv.build(str(root), log=lambda *a: None, cards_repo="")
    rows = rows_of(root)
    potato, tentacle, three, four, five, six, seven = (rows[dv.picture_id(sha(n))] for n in range(1, 8))
    assert potato["verdict"] == "unsure" and potato["reason"].startswith("named only, its text isn't")
    assert potato["meme_live"] == "potato"
    assert tentacle["verdict"] == "unsure" and "blooper-tentacle's better" in tentacle["reason"]
    assert three["card"] == "fang" and three["name_only"] and "created_review" in three
    assert not four.get("name_only") and "created_review" not in four
    assert five["verdict"] == "unsure" and five["meme_live"] == "fang" and five["review"] == dv.MEME_LIVE
    assert six["verdict"] == "morgue"
    assert seven["verdict"] == "unsure" and seven["meme_live"] == "fang"
    groups = json.loads((root / "derived/match/lost_groups.json").read_text())["groups"]
    assert [g["pictures"] for g in groups] == [[six["id"]]]


def test_looks_no_git_version_has_are_drafts_and_named_only_picks_reach_the_name_check():
    """An older template than every git version's, far from them; numbers no git version prints on a whole card; a
    status card's star count: drafts. A cut screenshot's or a lost digit's marks never are. A title-only pick under
    SURE goes to the name check on a whole card only. Stat-less cards print no bars. An eye ruling tells card and
    look."""
    tags = dv.marks(["(2CD) summon it for 2 turns"])
    v = {"key": "a", "card": "c", "look": 0, "whole": H, "ts": T, "marks": (tags, tags), "era": "D2"}
    pick = {"by": "hash", "keys": ["a"], "bits": 9.0}
    far = {"p": "0" * 15 + "f", "d": "0" * 14 + "ff", "q": "0" * 62 + "ff"}     # 4-5 bits: not that look

    def kind(**image):
        return dv.decide(pick, {"shape": "1:2", "w": 400, "h": 800, "whole": far, **image}, {"a": v}, [v])
    assert kind()[0] is None
    old = {**FAR}
    got = dv.decide(pick, {"shape": "1:2", "w": 400, "h": 800, "whole": old, "era": "B"}, {"a": v}, [v])
    assert got[0] == "draft" and got[1]["draft"].startswith("an older template")
    assert dv.decide(pick, {"shape": "1:2", "whole": old, "era": "B"}, {"a": v}, [{**v, "era": "C"}])[0] is None
    assert kind(marks={"t:2cd", "n:2:c", "n:3:t"})[0] == "draft"                 # 3 turns: no git version's
    assert kind(marks={"t:2cd", "n:2:c"})[0] is None                             # a number unread: maybe that look
    assert kind(marks={"t:2cd", "n:2:c", "n:234:t"})[0] is None                  # OCR joins digits: never counted
    assert kind(marks={"t:2cd", "n:2:c", "n:3:t"}, w=420, h=760)[0] is None      # cut: a screenshot
    assert kind(era="S", stars=2, git_stars={1})[0] == "draft" and kind(era="S", stars=1, git_stars={1})[0] is None
    assert dv.decide(pick, {"shape": "1:2", "w": 1200, "h": 2400, "whole": old, "size_class": "ps1200"}, {"a": v},
                     [v])[1]["draft"].startswith("a whole card export")                 # 8+ bits from every version
    assert dv.decide(pick, {"shape": "1:2", "w": 389, "h": 790, "whole": old, "size_class": "shot"}, {"a": v},
                     [v])[0] is None                                                     # a screenshot: maybe cut
    assert kind(size_class="ps1200")[0] is None                                         # under 8 bits
    named = {"status": "confident", "cards": ["c"], "scored": [{"card": "c", "conf": 0.93, "s": {"title": 1.0}}]}
    assert dv.gate("c", 0.93, named, "1:2", {}, whole=True) is None
    assert dv.gate("c", 0.93, named, "1:2", {}).startswith("confidence")
    mixed = {**named, "scored": [{"card": "c", "conf": 0.93, "s": {"title": 1.0, "ocr": 0.4}}]}
    assert dv.gate("c", 0.93, mixed, "1:2", {}, whole=True).startswith("confidence")
    assert dv.prints_no_bars({"type": "item", "no_stats": True}) and not dv.prints_no_bars({"type": "item"})
    row = {"id": "P-1", "verdict": "unsure", "maybe": ["c"], "reason": "x"}
    dv.by_eye(row, {"card": "c", "look": "a", "why": "same"}, {"a": {**v, "commit": "k"}}, {"c": {}})
    assert (row["verdict"], row["card"], row["version"]["look"], row["version"]["by"]) == ("version_sure", "c", 0, "eye")
    dv.by_eye(row, {"card": "none"}, {}, {"c": {}})
    assert row["verdict"] == "none" and "card" not in row
    lost = {"id": "P-2", "verdict": "lost", "lost_group": "LC-2"}         # the owner's ruling: apart from the eye's
    dv.by_eye(lost, {"card": "c", "look": "draft", "why": "11-13", "no_created": "weak"}, {}, {"c": {}}, by="owner")
    assert (lost["verdict"], lost["card"], lost["owner"], lost["no_created"]) == ("draft", "c", "owner: 11-13", "weak")
    assert "eye" not in lost and lost["version"]["by"] == "owner"
    row = {"id": "P-3", "verdict": "unsure", "maybe": ["c", "d"]}                # a card the owner's ruling rules out
    dv.by_owner(row, {"ruled_out": {"P-3": {"cards": ["c"]}}}, {}, {"c": {}, "d": {}})
    assert (row["verdict"], row["maybe"]) == ("unsure", ["d"])


def test_an_eye_keys_local_difference_finds_a_small_drawn_change():
    """eyekeys' score: a 12 px icon missing on a 600 px card stands out (a whole-card mean would hide it)."""
    import numpy as np
    a = np.full((900, 600, 3), 90, dtype=np.uint8)
    b = a.copy()
    b[40:52, 30:42] = 250
    score, (x, y), _ = dv.local_diff(a, b)
    assert score > 90 and 27 <= x <= 42 and 37 <= y <= 52
    assert dv.local_diff(a, a.copy())[0] == 0


def test_the_local_gate_tells_a_stored_versions_look_and_drafts_far_ones():
    """local.jsonl (eyekeys' score against every stored version, aspect within 1%): at LOCAL_SAME a version's look
    whatever the gates say; a gate's look over LOCAL_KEEP from its look not told; LOCAL_DRAFT from every stored
    version a draft, unless a version made before the post wasn't compared, or none of the picked look's."""
    versions = {"a": {"key": "a", "card": "c", "look": 0, "whole": H, "ts": T},
                "b": {"key": "b", "card": "c", "look": 1, "whole": FAR, "ts": T}}
    members = list(versions.values())
    pick = {"by": "text", "keys": ["a"], "fit": 0.95, "novel": 2}
    image = {"shape": "2:3", "whole": H, "posted": T + 10, "w": 1200, "h": 1800}

    def kind(scores, unread=(), how=pick):
        local = dv.local_seen({"versions": scores}, members)
        return dv.decide(how, {**image, **local, "local_unread": [*local["local_unread"], *unread]}, versions,
                         members)
    got = kind({"a": [40.0, 1.0], "b": [2.0, 1.0]})
    assert got[0] == "git" and got[1]["look"] == 1 and got[1]["checked"] == {"local": 2.0}
    assert kind({"a": [0.0, 1.0], "b": [99.0, 1.0]})[0] == "git"
    assert kind({"a": [14.0, 1.0], "b": [99.0, 1.0]})[0] is None                    # one area drawn otherwise
    assert kind({"a": [14.0, 1.06], "b": [99.0, 1.06]})[0] == "git"                 # cut: the gates decide
    drawn = kind({"a": [31.0, 1.0], "b": [80.0, 1.0]})
    assert drawn[0] == "draft" and "local" in drawn[1]["draft"]
    assert kind({"a": [31.0, 1.0], "b": [80.0, 1.0]}, unread=[T + 5])[0] is None    # a version not compared
    assert kind({"a": [31.0, 1.0]})[0] is None                                      # b not computed yet
    image["w"] = 400                                                                # a small picture: further
    assert kind({"a": [31.0, 1.0], "b": [80.0, 1.0]})[0] is None and kind({"a": [50.0, 1.0], "b": [80.0, 1.0]})[0] == "draft"
    assert dv.local_seen(None, members) == {}
    versions["c"] = {"key": "c", "card": "c", "look": 2, "whole": FAR, "ts": T + 20}     # unstored, after the post
    members = list(versions.values())
    assert kind({"a": [50.0, 1.0], "b": [80.0, 1.0]}, how=dict(pick, keys=["c"]))[0] is None   # the pick unseen


def test_a_look_shown_in_a_card_channel_before_its_meme_repost_is_no_joke():
    post = lambda mid, channel, meme=False: {"message": mid, "channel": channel, "date": f"2020-08-{mid[-2:]}",
                                             "bot": False, "morgue": meme, "paper": False}
    row = lambda card, posts: {"id": "P-1", "sha256": "a", "verdict": "draft", "card": card, "joke": True,
                               "joke_draft": True, "posts": posts}
    shown = row("wendy", [post("1013", "update-02-port"), post("1025", "cursed", True)])
    dv.shown_first(shown, {}, {"a": 1025})
    assert not shown["joke"] and "joke_draft" not in shown and shown["shown_first"]["channel"] == "update-02-port"
    twin = row("god-tamer", [post("1022", "port")])                   # its meme repost is a twin's, later
    dv.shown_first(twin, {}, {"a": 1030})
    assert not twin["joke"]
    for kept in (row("wendy", [post("1013", "general-discussion"), post("1025", "cursed", True)]),
                 row("wendy", [post("1013", "port")]),             # a twin's meme post came first
                 row("fireball", [post("1013", "new-cards")])):    # an open owner question: as built
        dv.shown_first(kept, {}, {"a": 1001 if kept["card"] == "wendy" and len(kept["posts"]) == 1 else 1025})
        assert kept["joke"] and kept["joke_draft"]
    copy = row("the-puppet", [post("1024", "ideas-discussion"), post("1030", "meme-cards", True)])
    dv.shown_first(copy, {}, {"a": 1030}, lambda r, whole: "Cards/Meme/5 Stars/The_Puppet.png")   # the first repo's
    assert copy["joke"] and copy["meme_file"] == "Cards/Meme/5 Stars/The_Puppet.png"            # meme file: a joke


def test_a_meme_copy_is_the_first_repos_meme_only_file_or_near_and_untold(tmp_path):
    """meme_files: the same bytes as a first-repo PNG filed only under Meme folders, or near one (whole within
    TWIN_BITS) while the eye hasn't told the picture; a blob also in a card folder is no meme copy."""
    import hashlib
    import json
    data, other = b"meme picture", b"card picture"
    blob = lambda b: hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest()
    (tmp_path / "attachments").mkdir()
    (tmp_path / "attachments" / "m.png").write_bytes(data)
    (tmp_path / "attachments" / "c.png").write_bytes(other)
    base = tmp_path / "derived" / "oldgit"
    base.mkdir(parents=True)
    h = {"p": "0" * 16, "d": "0" * 16, "q": "0" * 64}
    versions = [{"blob": blob(data), "path": "Cards/Meme/5 Stars/X.png", "ext": "png", "repo": "first"},
                {"blob": blob(other), "path": "Cards/Meme/5 Stars/Y.png", "ext": "png", "repo": "first"},
                {"blob": blob(other), "path": "Items/Active/2 Stars/Y.png", "ext": "png", "repo": "first"}]
    (base / "versions.jsonl").write_text("".join(json.dumps(v) + "\n" for v in versions))
    (base / "reads.jsonl").write_text(json.dumps({"blob": blob(data), "kind": "png", "look": {"whole": h}}) + "\n")
    copy = dv.meme_files(str(tmp_path))
    assert copy({"file": "attachments/m.png"}, None) == "Cards/Meme/5 Stars/X.png"
    assert copy({"file": "attachments/c.png"}, None) is None
    assert copy({"file": "attachments/c.png"}, dict(h)) == "near Cards/Meme/5 Stars/X.png"
    assert copy({"file": "attachments/c.png", "eye": "eye: its own card"}, dict(h)) is None
