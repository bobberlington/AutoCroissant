"""discord_chain on made-up cards: the buff/nerf rule and ability changes, which pictures enter a timeline, reposts
merged, facts checked by pictures, splits, paper cards and Created against the file."""
from collections import defaultdict
from types import SimpleNamespace

import commands.discord_chain as dc

T = 1600000000.0
DAY = 86400
H = {"p": "0" * 16, "d": "0" * 16, "q": "0" * 64}
FAR = {"p": "f" * 16, "d": "f" * 16, "q": "f" * 64}


def iso(seconds):
    return dc.dm.iso(seconds)


def stats(hp, d, a, s, how="picture"):
    return dc.sure({"hp": hp, "def": d, "atk": a, "spd": s}, how)


def test_compare_counts_the_website_way():
    a = {"stats": stats(5, 2, 3, 4), "ability": dc.sure("(1CD) Heal [fire] 2 HP.", "psd")}
    b = {"stats": stats(5, 2, 4, 4), "ability": dc.sure("(1CD) heal 2 hp", "ocr")}
    got = dc.compare(a, b)
    assert got["bal"] == "buff" and got["buffs"] == ["ATK 3 -> 4"] and got["ability_change"] == "formatting"
    assert dc.compare(b, {"stats": stats(5, 1, 5, 4)})["bal"] == "shift"
    assert "bal" not in dc.compare(b, {"stats": stats(30, 2, 10, 4)})         # off the scale: not judged
    unread = dc.compare(a, {"stats": dc.unsure("picture", "bars under their bar", {"hp": 1}),
                            "ability": dc.sure("Heal 3 HP.", "psd")})
    assert unread["unknown"] == ["stats"] and unread["ability_change"] == "changed" and "bal" not in unread
    assert dc.settled({"stars": dc.sure(2, "eye")}, {"stars": dc.sure(2, "psd")})
    assert not dc.settled({"stars": dc.sure(2, "eye"), "types": dc.unsure("ocr", "x")},
                          {"stars": dc.sure(2, "psd"), "types": dc.unsure("ocr", "x")})


def picture(n, when, verdict="draft", whole=H, card="bolt", **extra):
    first = {"date": iso(when), "channel": "new-cards", "thread": None, "message": str(n)} if when else None
    return {"id": f"P-{n:012d}", "verdict": verdict, "card": card, "cards": [card], "first": first,
            "sha256": f"{n:064d}", "file": f"attachments/{n}.png", "thumb": None, "whole": whole,
            "version": {"look": extra.pop("look", None)} if verdict != "draft" else {"shows": "draft"}, **extra}


def inputs(verdicts, text=None, bars=None, facts=(), old_rows=(), created=None, lost=(), old_review=()):
    cards = {"bolt": {"key": "Bolt", "name": "Bolt", "type": "creature", "fs": T + 100 * DAY, "fs_from": "old git"},
             "pike": {"key": "Pike", "name": "Pike", "type": "creature", "fs": T + 100 * DAY, "fs_from": "old git"},
             "lorn": {"key": "Lorn", "name": "Lorn", "type": "minion", "fs": T + 300 * DAY, "fs_from": "git"}}
    versions = defaultdict(list, {"bolt": [{"key": "b" + "a" * 20, "card": "bolt", "ts": T + 100 * DAY, "look": 1,
                                            "old": True, "commit": "c1", "path": "Creatures/Bolt.png"},
                                           {"key": "b" + "b" * 20, "card": "bolt", "ts": T + 150 * DAY, "look": 0,
                                            "old": True, "commit": "c2", "path": "Creatures/Bolt.png"}]})
    rows = defaultdict(list)
    for row in old_rows:
        rows[row["card"]].append(row)
    return SimpleNamespace(cards=cards, by_key={c["key"]: k for k, c in cards.items()}, versions=versions, images={},
                           verdicts=verdicts, lost=list(lost), text=text or {}, text_truth={}, bars=bars or {},
                           old_rows=rows, old_review=list(old_review), facts=list(facts), switches={},
                           created_file=created or {}, first_data={})


def old_row(card, when, blob, hp):
    return {"id": f"{blob[:12]}:Creatures/{card}.png", "card": card, "ts": when, "author": "MichaelJSr",
            "subject": "cards", "commit": blob, "path": f"Creatures/{card}.png", "blob": blob, "minor": False,
            "ability_change": "first", "fields": {"name": {"value": card, "method": "psd", "confidence": "sure"},
                                                  "stats": {"value": {"hp": hp, "def": 2, "atk": 3, "spd": 4},
                                                            "method": "picture", "confidence": "sure"}}}


def test_which_pictures_enter_and_reposts_merge():
    text = lambda n, hp: {f"attachments/{n}.png": {"title": {"value": "Bolt", "sure": True, "conf": 1.0,
                                                             "method": "ocr+name"}}}
    bars = {f"attachments/{n}.png": {"stats": {k: {"v": hp if k == "hp" else v, "sure": True, "conf": 1.0}
                                               for k, v in (("hp", 0), ("def", 2), ("atk", 3), ("spd", 4))}}
            for n, hp in ((1, 6), (2, 6), (3, 7), (4, 7))}
    verdicts = [picture(1, T), picture(2, T + DAY), picture(3, T + 2 * DAY),       # 3: same hash, HP 7: a change
                picture(4, T + 3 * DAY, "version_sure", look=1),                   # the first git look, pre-git
                picture(5, T + 4 * DAY, "version_sure", look=0),                   # a later git look: the git part's
                picture(6, T + 5 * DAY, "version_unsure"),                         # look not told: review
                picture(7, T - DAY, created_review="named only, before its first picture evidence (x)"),
                picture(8, T + 200 * DAY),                                         # after git
                picture(9, None, paper=True),                                      # a paper card: first, undated
                picture(10, T + 6 * DAY, joke=True)]                               # a joke draft: review
    texts = {}
    for n in (1, 2, 3):
        texts.update(text(n, 0))
    old = [old_row("Bolt", T + 100 * DAY, "a" * 40, 7)]
    facts = [{"id": "f1", "card": "Bolt", "source": "changelog", "confidence": "sure", "date": iso(T + 1.5 * DAY),
              "field": "hp", "to": 7, "channel": "update-01"},
             {"id": "f2", "card": "Bolt", "source": "changelog", "confidence": "sure", "date": iso(T + 2.5 * DAY),
              "field": "hp", "to": 9, "channel": "update-01"},
             {"id": "f3", "card": "Bolt", "source": "changelog", "confidence": "sure", "date": iso(T + 300 * DAY),
              "kind": "ability changed", "channel": "finished-updates"}]
    chain = dc.Chain(inputs(verdicts, texts, bars, facts, old))
    found, split = chain.discord_rows()
    t = chain.card("bolt", found["bolt"], split.get("bolt", []))
    kinds = [(e["kind"], e["id"]) for e in t["entries"]]
    assert kinds == [("paper", "D-P-000000000009"), ("discord", "D-P-000000000001"), ("text", "F-f1"),
                     ("discord", "D-P-000000000003"), ("discord", "D-P-000000000004"),
                     ("old repo", "O-aaaaaaaaaaaa:Creatures/Bolt.png")]
    paper, first, fact, second, look = (e for e in t["entries"][:5])
    assert paper["date"] is None and paper["label"] == dc.PAPER_LABEL
    assert [p["id"] for p in first["pictures"]] == ["P-000000000001", "P-000000000002"]     # a repost merged
    assert second["changes"]["bal"] == "buff" and fact["confirmed_by"] == second["id"]
    assert look["look"]["git"]["source"] == "old repo" and look["fields"]["stats"]["method"] == "git look: picture"
    kinds = {i["kind"] for i in chain.review.values()}
    assert {"look not told", "named only", "joke picture", "fact vs picture"} <= kinds     # f2: HP 9 read as 7
    created = chain.created("bolt", t, [r for r in verdicts if r.get("card") == "bolt"])
    assert created["status"] == "new" and created["ours"]["picture"] == "P-000000000001"   # not the paper, not 7
    assert created["named_only_earlier"]["picture"] == "P-000000000007"


def test_created_against_the_file_and_splits():
    lost = [{"id": "LC-x", "split_of": {"name": "Twin Pike", "into": ["pike", "lorn"]}}]
    verdicts = [dict(picture(1, T + 50 * DAY, "lost", card=None), lost_group="LC-x"),
                picture(2, T + 60 * DAY, card="pike"),
                dict(picture(3, T + 150 * DAY, "lost", FAR, card=None), lost_group="LC-x")]   # after pike's git
    review = [{"id": "oldrows-1", "kind": "unsure map", "old_id": "Twin Pike",
               "split": {"halves": {"Pike": "top", "Lorn": "bottom"},
                         "rows": {"Lorn": [dict(old_row("Twin Pike", T + 80 * DAY, "c" * 40, 3),
                                                card="Lorn", split_from="Twin Pike")]}}}]
    bars = {"attachments/1.png": {"truth": {"stats": {"hp": 6, "def": 4, "atk": 6, "spd": 5},
                                            "stats2": {"hp": 4, "def": 2, "atk": 4, "spd": 5}, "from": "eye"}}}
    files = {"Pike": {"date": dc.day(T + 60 * DAY), "from": "discord"},
             "Lorn": {"date": dc.day(T + 10 * DAY), "from": "discord"},
             "Bolt": {"date": "2001-01-01", "from": "hand"}}
    chain = dc.Chain(inputs(verdicts, bars=bars, created=files, lost=lost, old_review=review))
    found, split = chain.discord_rows()
    pike, lorn = (chain.card(c, found.get(c, []), split.get(c, [])) for c in ("pike", "lorn"))
    assert [e.get("as") for e in lorn["entries"]] == ["Twin Pike", "Twin Pike", "Twin Pike"]
    assert [e.get("as") for e in pike["entries"]] == ["Twin Pike", None]
    proposed = lambda t: chain.review[t["entries"][0]["fields"]["stats"]["review"]]["proposed"]    # each half's own
    assert proposed(lorn) == {"hp": 4, "def": 2, "atk": 4, "spd": 5}
    assert proposed(pike) == {"hp": 6, "def": 4, "atk": 6, "spd": 5}
    made = {c: chain.created(c, t, [r for r in verdicts if r.get("card") == c]) for c, t in
            (("pike", pike), ("lorn", lorn), ("bolt", chain.card("bolt", [], [])))}
    assert made["pike"]["status"] == "earlier" and made["pike"]["ours"]["as"] == "Twin Pike"
    assert made["lorn"]["status"] == "later" and made["lorn"]["date"] == files["Lorn"]["date"]   # never later
    assert made["bolt"]["status"] == "hand" and made["bolt"]["date"] == "2001-01-01"
