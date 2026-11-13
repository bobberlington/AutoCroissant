"""discord_chain on made-up cards: the buff/nerf rule and ability changes, which pictures enter a timeline, reposts
merged (nearest look, same words, same numbers counted, no local edit, no fill a picture didn't read), icon star
changes nothing backs, facts checked by pictures, first-repo rows, splits, paper cards and Created against the file."""
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


def inputs(verdicts, text=None, bars=None, facts=(), old_rows=(), created=None, lost=(), old_review=(), images=None,
           first_repo=(), paper_eye=None, root="/nonexistent"):
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
    return SimpleNamespace(root=str(root), cards=cards, by_key={c["key"]: k for k, c in cards.items()}, versions=versions,
                           images=images or {}, verdicts=verdicts, lost=list(lost), text=text or {}, text_truth={},
                           bars=bars or {}, old_rows=rows, old_review=list(old_review), facts=list(facts),
                           switches={"first_repo_rows": True}, first_repo=list(first_repo),
                           paper_eye=paper_eye or {}, created_file=created or {}, first_data={})


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
                picture(9, None, paper=True),                                      # a paper card: 'Before 2020'
                picture(11, None, "none", card=None, paper=True),                  # named by eye: Bolt's too
                picture(12, None, "none", card=None, paper=True),                  # not named: review
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
    pushes = [{"id": "q1", "card": "Bolt", "date": iso(T + 3.5 * DAY), "author": "Chestnut", "subject": "Update Bolt",
               "message": "9", "commit": "abc"},
              {"id": "q2", "card": "Bolt", "date": iso(T + 101 * DAY), "author": "Chestnut", "subject": "x"}]
    eye = {"P-000000000011": {"card": "bolt", "how": "title"}, "P-000000000012": {"card": None, "title": "Bolt X"}}
    chain = dc.Chain(inputs(verdicts, texts, bars, facts, old, first_repo=pushes, paper_eye=eye))
    found, split = chain.discord_rows()
    t = chain.card("bolt", found["bolt"], split.get("bolt", []))
    kinds = [(e["kind"], e["id"]) for e in t["entries"]]
    assert kinds == [("paper", "D-P-000000000009"), ("discord", "D-P-000000000001"), ("text", "F-f1"),
                     ("discord", "D-P-000000000003"), ("discord", "D-P-000000000004"), ("first repo", "Q-q1"),
                     ("old repo", "O-aaaaaaaaaaaa:Creatures/Bolt.png")]
    paper, first, fact, second, look, push = (e for e in t["entries"][:6])
    assert (paper["date"], paper["label"], [p["id"] for p in paper["pictures"]]) == (
        None, dc.PAPER_DAY, ["P-000000000009", "P-000000000011"])                         # one paper card
    assert paper["fields"] == {"name": dc.sure("Bolt", "matcher")} and first["changes"]["ability_change"] == "first"
    assert push == {"id": "Q-q1", "kind": "first repo", "date": iso(T + 3.5 * DAY), "day": dc.day(T + 3.5 * DAY),
                    "badge": "first repo", "author": "Chestnut", "subject": "Update Bolt", "source": "first repo push"}
    assert [p["id"] for p in first["pictures"]] == ["P-000000000001", "P-000000000002"]     # a repost merged
    assert second["changes"]["bal"] == "buff" and fact["confirmed_by"] == second["id"]
    assert look["look"]["git"]["source"] == "old repo" and look["fields"]["stats"]["method"] == "git look: picture"
    kinds = {i["kind"] for i in chain.review.values()}
    assert {"look not told", "named only", "joke picture", "fact vs picture", "paper photo"} <= kinds   # f2: HP 9
    rows = [r for r in verdicts if r.get("card") == "bolt"]
    created = chain.created("bolt", t, rows)                                                 # owner Q6: the paper
    assert (created["status"], created["date"], created["sort"], created["ours"]["source"]) == (
        "new", dc.PAPER_DAY, dc.PAPER_SORT, "paper card") and created["dated"]["picture"] == "P-000000000001"
    dated = chain.created("bolt", {**t, "entries": t["entries"][1:]}, rows)
    assert dated["ours"]["picture"] == "P-000000000001"                                      # not 7, named only
    assert dated["named_only_earlier"]["picture"] == "P-000000000007"


def test_reposts_join_their_nearest_look_with_the_same_words():
    def whole(p, d=0):
        return {"p": format((1 << p) - 1, "016x"), "d": format((1 << d) - 1, "016x"), "q": "0" * 64}
    old = ["Bolt", "Deal two damage to a creature.", "Heal one ally."]
    new = ["Bolt", "Deal two damage to a friendly creature.", "Heal one ally."]
    bars = lambda hp: {k: {"v": v, "sure": True, "conf": 1.0} for k, v in (("hp", hp), ("def", 2), ("atk", 3),
                                                                             ("spd", 4))}
    verdicts = [dict(picture(1, T, whole=whole(0)), cell=1, box=[0, 0, 0.5, 1]),    # a sheet's cell: nothing read
                picture(2, T + 10 * DAY, whole=whole(1)),                          # its repost: HP 6 read
                picture(3, T + 20 * DAY, whole=whole(3)),                          # HP 7: another look
                picture(4, T + 30 * DAY, whole=whole(3, 1)),                       # near both, nearest to 3
                picture(5, T + 40 * DAY, whole=whole(3))]                          # 'friendly' added: its own
    images = {f"{n:064d}": {"lines": lines} for n, lines in ((2, old), (3, old), (4, old), (5, new))}
    images[f"{1:064d}"] = {"cells": [{"lines": old}]}
    chain = dc.Chain(inputs(verdicts, bars={"attachments/2.png": {"stats": bars(6)},
                                            "attachments/3.png": {"stats": bars(7)}}, images=images))
    found, _ = chain.discord_rows()
    entries = [e for e in chain.card("bolt", found["bolt"], [])["entries"] if e["kind"] == "discord"]
    assert [[p["id"] for p in e["pictures"]] for e in entries] == [
        ["P-000000000001", "P-000000000002"], ["P-000000000003", "P-000000000004"], ["P-000000000005"]]
    cell = entries[0]["fields"]["stats"]                  # never filled from the repost: proposed only
    assert cell["confidence"] == "review" and chain.review[cell["review"]]["proposed"]["hp"] == 6
    assert entries[1]["fields"]["stats"]["value"]["hp"] == 7
    assert chain.tally["discord: a near look kept apart (its words differ)"] == 1


def test_created_against_the_file_and_splits():
    lost = [{"id": "LC-x", "split_of": {"name": "Twin Pike", "into": ["pike", "lorn"]}}]
    verdicts = [dict(picture(1, T + 50 * DAY, "lost", card=None), lost_group="LC-x"),
                picture(2, T + 60 * DAY, card="pike"),
                dict(picture(3, T + 150 * DAY, "lost", FAR, card=None), lost_group="LC-x")]   # after pike's git
    half = dict(old_row("Twin Pike", T + 80 * DAY, "c" * 40, 3), card="Lorn", split_from="Twin Pike")
    half["fields"]["stats"].update(confidence="review", why="bottom half of Twin Pike's rows")
    review = [{"id": "oldrows-1", "kind": "unsure map", "old_id": "Twin Pike",
               "split": {"halves": {"Pike": "top", "Lorn": "bottom"}, "rows": {"Lorn": [half]}}}]
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
    row = next(e for e in lorn["entries"] if e["kind"] == "old repo")                      # its own review item
    assert chain.review[row["fields"]["stats"]["review"]]["proposed"] == {"hp": 3, "def": 2, "atk": 3, "spd": 4}


def test_numbers_local_edits_and_unbacked_star_changes(tmp_path):
    """A swapped number counts (5/7/6/6 -> 5/7/7/6); a picture alike but for one area (an icon added) is its own
    look; an icon-read star count changing with nothing to back it goes to review; a repost's own contrary read
    takes a reader's sure stars to review."""
    import numpy as np
    from PIL import Image
    a, b = ["Mettaton X form (5/7/6/6) for two turns"], ["Mettaton X form (5/7/7/6) for two turns"]
    assert dc.read_marks(a) == dc.read_marks(b) and dc.mark_counts(a) != dc.mark_counts(b)
    (tmp_path / "attachments").mkdir()
    base = np.tile(np.linspace(40, 200, 400, dtype=np.float32), (800, 1))
    edit, noisy, cased = base.copy(), base + np.random.RandomState(1).uniform(-3, 3, base.shape), base.copy()
    edit[20:60, 20:60] = 0                                                             # an icon added
    cased[400:420, 100:130] = 0                                                        # 'deal' -> 'Deal'
    for n, img in ((1, base), (2, base), (3, edit), (4, noisy), (5, cased)):
        Image.fromarray(img.clip(0, 255).astype(np.uint8)).save(tmp_path / f"attachments/{n}.png")
    row = lambda n: {"file": f"attachments/{n}.png"}
    assert dc.edited_cells(str(tmp_path), row(1), row(3))
    assert not dc.edited_cells(str(tmp_path), row(1), row(2)) and not dc.edited_cells(str(tmp_path), row(1), row(4))
    ocr = lambda n, text, *more: dc.dumps({"id": f"d:{n:064d}", "fix": [["Bolt", 1, 0.3, 0.02, 0.4, 0.05, []],
                                                                        [text, 1, 0.2, 0.49, 0.6, 0.04, []], *more]})
    cls = lambda text: [text, 1, 0.35, 0.965, 0.3, 0.03, []]                          # a class line
    (tmp_path / "derived" / "readers" / "text").mkdir(parents=True)
    (tmp_path / "derived" / "readers" / "text" / "ocr.jsonl").write_text("\n".join(
        [ocr(n, "deal two damage.") for n in (1, 2, 3)] + [ocr(5, "Deal two damage!"),
         ocr(6, "deal two damage.", cls("Demonic/God")), ocr(7, "deal two damage.", cls("DEMONIC / god")),
         ocr(8, "deal two damage.", cls("Demon/God"))]) + "\n")
    for n in (6, 7, 8):
        Image.fromarray(base.astype(np.uint8)).save(tmp_path / f"attachments/{n}.png")
    lines = ["Bolt", "Deal two damage to a creature."]
    images = {f"{n:064d}": {"lines": lines} for n in (1, 2, 3, 5, 6, 7, 8)}

    def looks(*ns):
        chain = dc.Chain(inputs([picture(n, T + i * DAY) for i, n in enumerate(ns)], images=images, root=tmp_path))
        found, _ = chain.discord_rows()
        return chain, [[p["id"][-1] for p in e["pictures"]] for e in chain.card("bolt", found["bolt"], [])["entries"]
                       if e["kind"] == "discord"]
    chain, got = looks(1, 2, 3, 5)
    assert got == [["1", "2", "5"], ["3"]]
    assert chain.tally["discord: a near look kept apart (a local edit)"] == 1          # formatting alone: one look
    chain, got = looks(1, 6, 7, 8)              # a class line read on one only tells nothing; its case and spacing
    assert got == [["1", "6", "7"], ["8"]]      # neither; 'Demonic/God' -> 'Demon/God' is its own look
    assert chain.tally["discord: a near look kept apart (its words differ)"] == 1
    assert dc.lone_words({"demonic", "dancer"}, {"demon", "dance"}) == ["dancer", "demonic"]    # letters added
    assert dc.lone_words({"defence"}, {"health"}, {"defence"}) == []   # a stat label misread on its own line
    assert dc.lone_words({"defender"}, {"health"}) == ["defender"]     # ... in the text, a word

    def version(n, when, stars, how="icon+known", kind="discord", era="D2"):
        return {"id": f"E{n}", "kind": kind, "date": iso(T + when * DAY), "era": era,
                "fields": {"stars": dc.sure(stars, how)}}
    fact = {"id": "F1", "kind": "text", "date": iso(T + 45 * DAY), "change": {"field": "stars", "to": 4}}
    cases = [([version(1, 0, 5), version(2, 40, 5), version(3, 50, 1), version(4, 100, 5, "psd", "old repo")], {"E3"}),
             ([version(1, 0, 2), version(2, 100, 3, "psd", "old repo")], {"E1"}),        # an edge-cut star
             ([version(1, 0, 2, "eye"), version(2, 100, 3, "psd", "old repo")], set()),  # eye backs it
             ([version(1, 0, 5), version(2, 40, 5), version(3, 50, 4), fact, version(4, 100, 5, "psd", "old repo")],
              set()),                                                                    # a changelog fact backs it
             ([version(1, 0, 3), version(2, 5, 4)], {"E2"})]                             # nothing after agrees
    for entries, held in cases:
        chain.star_guard(entries)
        assert {e["id"] for e in entries if e["kind"] != "text" and not dc.is_sure(e["fields"]["stars"])} == held
    mine = {"stars": dc.sure(2, "icon+known")}
    other = {"stars": dc.unsure("icon", "read under its bar", 4)}
    assert dc.merge_fields(mine, other, "repost P-1")["stars"]["confidence"] == "review"
    assert dc.is_sure(dc.merge_fields({"stars": dc.sure(2, "eye")}, other, "repost P-1")["stars"])
    assert dc.is_sure(dc.merge_fields(mine, other, "the picture", fill=False)["stars"])


def test_the_owner_s_answers():
    """The review page's answers (rulings.json): a named-only picture confirmed enters and dates Created, a look told,
    'not this card', a card named, a paper photo named, a field given, an unsure fact right, Created picked with its
    source, earlier hints only before the Created date in use, a map to a card that isn't live, the ids acted on."""
    rid = dc.review_id
    text = {"attachments/1.png": {"title": {"value": "Bolt", "sure": False, "conf": 0.5, "method": "ocr+name"}}}
    verdicts = [picture(1, T, created_review="named only, before its first picture evidence (x)"),
                picture(2, T + DAY, "version_unsure"), picture(3, T + 2 * DAY, "version_unsure"),
                dict(picture(4, T + 3 * DAY, "unsure", card=None), cards=None, maybe=["bolt"]),
                picture(5, None, "none", card=None, paper=True)]
    facts = [{"id": "f9", "card": "Bolt", "source": "changelog", "confidence": "unsure", "date": iso(T + 4 * DAY),
              "field": "hp", "by": -2, "unit": "unknown", "channel": "update-01"}]
    rulings = {rid("named only", "P-000000000001"): {"choice": "yes"},
               rid("look not told", "P-000000000002"): {"choice": "look", "value": 1},
               rid("look not told", "P-000000000003"): {"choice": "no"},
               "match:P-000000000004": {"choice": "card", "value": "bolt"},
               rid("paper photo", "P-000000000005"): {"choice": "card", "value": "bolt"},
               rid("field", "bolt|D-P-000000000001|name"): {"choice": "yes"},
               "R1": {"kind": "unsure fact", "choice": "yes", "value": "points", "item": {"fact": "f9"}},
               rid("created earlier", "bolt"): {"choice": "file"}, rid("changelog earlier", "bolt"): {"choice": "yes"},
               "oldrows-m1": {"kind": "unsure map", "choice": "yes"}}
    gone = {"id": "oldrows-m1", "kind": "unsure map", "card": "Zed Old", "status": "archived",
            "rows": [old_row("Zed", T + 50 * DAY, "d" * 40, 5)]}
    got = inputs(verdicts, text, facts=facts, created={"Bolt": {"date": dc.day(T + 9 * DAY), "from": "discord"}},
                 old_review=[gone])
    got.rulings = rulings
    done = dc.owner_rows(got)
    assert done["owner: not this card"] == 1 and "Zed Old" not in got.old_rows                 # no live card: none
    assert got.used == {rid("named only", "P-000000000001"), rid("look not told", "P-000000000002"),
                        rid("look not told", "P-000000000003"), "match:P-000000000004",
                        rid("paper photo", "P-000000000005"), "R1"}
    chain = dc.Chain(got)
    found, split = chain.discord_rows()
    t = chain.card("bolt", found["bolt"], split.get("bolt", []))
    entries = {e["id"]: e for e in t["entries"]}
    assert set(entries) == {"D-P-000000000005", "D-P-000000000001", "D-P-000000000002", "F-f9"}
    assert entries["D-P-000000000001"]["fields"]["name"] == dc.sure("Bolt", "owner", ruled=rid(
        "field", "bolt|D-P-000000000001|name"))
    assert entries["D-P-000000000002"]["look"]["git"]["commit"] == "c1"
    assert entries["F-f9"]["change"]["unit"] == "points"
    assert {i["picture"] for i in chain.review.values() if i["kind"] == "look not told"} == {"P-000000000004"}
    made = chain.created("bolt", {**t, "entries": [e for e in t["entries"] if e["kind"] != "paper"]},
                         [r for r in verdicts if r.get("card") == "bolt"])
    # ours is T; the website's T+9 picked, so the changelog line (T+4) is a hint, answered yes: it wins, with its source
    assert (made["status"], made["date"], made["owner"]["was"], made["owner"]["base"]) == (
        "owner", dc.day(T + 4 * DAY), "earlier", dc.day(T + 9 * DAY)) and made["proposed"]
    assert made["owner"]["source"] == {"day": dc.day(T + 4 * DAY), "source": "changelog", "from": "changelog line",
                                       "fact": "f9", "channel": "update-01"}
    assert {rid("created earlier", "bolt"), rid("changelog earlier", "bolt"),
            rid("field", "bolt|D-P-000000000001|name")} <= chain.used and "oldrows-m1" not in chain.used
    # a 'later' card: a line after the website's date is no hint; answered 'ours', a line before ours is one
    sure_fact = {**facts[0], "id": "f7", "confidence": "sure", "date": iso(T + 2 * DAY)}
    later = inputs([picture(21, T + 3 * DAY, "version_sure", look=1)], facts=[sure_fact],
                   created={"Bolt": {"date": dc.day(T - 5 * DAY), "from": "discord"}})
    for answers, status, hint in (({}, "later", None), ({rid("created later", "bolt"): {"choice": "ours"}}, "owner",
                                                        dc.day(T + 2 * DAY))):
        later.rulings = answers
        chain = dc.Chain(later)
        found, split = chain.discord_rows()
        t = chain.card("bolt", found["bolt"], split.get("bolt", []))
        made = chain.created("bolt", t, later.verdicts)
        assert (made["status"], (made.get("changelog_earlier") or {}).get("day")) == (status, hint)
    assert made["date"] == dc.day(T + 3 * DAY) and made["proposed"] and made["owner"]["source"]["picture"] == \
        "P-000000000021" and made["owner"]["source"]["from"] == "ours"
