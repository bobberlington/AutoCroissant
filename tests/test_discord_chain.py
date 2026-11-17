"""discord_chain on made-up cards: the buff/nerf rule and ability changes, which pictures enter a timeline, reposts
merged (nearest look, same words, same numbers counted, no local edit, no fill a picture didn't read), icon star
changes nothing backs, facts checked by pictures, first-repo push notices (no rows), splits and Created against the
file (the physical card photos' three, a lapsed id told)."""
from collections import Counter, defaultdict
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
    joke = {"stats": stats(5, 9, 9, 4), "ability": dc.sure("Heal 9 HP.", "psd")}         # owner 11-17: a Meme copy
    rows = [{"kind": "old repo", "fields": a}, {"kind": "old repo", "fields": joke, "meme_copy": True},
            {"kind": "old repo", "fields": {**a, "stats": stats(5, 2, 4, 4)}}]
    chain = dc.Chain.__new__(dc.Chain)
    chain.inputs, chain.tally = SimpleNamespace(root="/nonexistent"), Counter()
    chain.diff(rows, "creature")
    assert rows[1]["changes"] == {} and rows[2]["changes"]["buffs"] == ["ATK 3 -> 4"]   # diffed with the card's own
    gone = {"ability": dc.sure("", "eye")}                     # owner 11-17: a blank box after an ability (Panda)
    assert dc.compare(a, gone)["ability_change"] == "removed" and dc.compare(gone, dict(gone))["ability_change"] == "same"
    assert dc.blank({"fields": gone}) and not dc.blank({"fields": {"ability": dc.sure("", "psd")}})
    assert dc.settled({"stars": dc.sure(2, "eye")}, {"stars": dc.sure(2, "psd")})
    assert not dc.settled({"stars": dc.sure(2, "eye"), "types": dc.unsure("ocr", "x")},
                          {"stars": dc.sure(2, "psd"), "types": dc.unsure("ocr", "x")})


def picture(n, when, verdict="draft", whole=H, card="bolt", **extra):
    first = {"date": iso(when), "channel": "new-cards", "thread": None, "message": str(n)} if when else None
    return {"id": f"P-{n:012d}", "verdict": verdict, "card": card, "cards": [card], "first": first,
            "sha256": f"{n:064d}", "file": f"attachments/{n}.png", "thumb": None, "whole": whole,
            "version": {"look": extra.pop("look", None)} if verdict != "draft" else {"shows": "draft"}, **extra}


def inputs(verdicts, text=None, bars=None, facts=(), old_rows=(), created=None, lost=(), old_review=(), images=None,
           first_repo=(), root="/nonexistent"):
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
                           first_repo=list(first_repo), created_file=created or {}, first_data={})


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
    chain = dc.Chain(inputs(verdicts, texts, bars, facts, old, first_repo=pushes))
    found, split = chain.discord_rows()
    t = chain.card("bolt", found["bolt"], split.get("bolt", []))
    kinds = [(e["kind"], e["id"]) for e in t["entries"]]
    assert kinds == [("discord", "D-P-000000000001"), ("text", "F-f1"),
                     ("discord", "D-P-000000000003"), ("discord", "D-P-000000000004"),
                     ("old repo", "O-aaaaaaaaaaaa:Creatures/Bolt.png")]      # push notices: no rows (owner Q5, 11-14)
    first, fact, second, look = t["entries"][:4]
    assert first["changes"]["ability_change"] == "first"
    assert [p["id"] for p in first["pictures"]] == ["P-000000000001", "P-000000000002"]     # a repost merged
    assert second["changes"]["bal"] == "buff" and fact["confirmed_by"] == second["id"]
    assert look["look"]["git"]["source"] == "old repo" and look["fields"]["stats"]["method"] == "git look: picture"
    kinds = {i["kind"] for i in chain.review.values()}
    assert {"look not told", "named only", "joke picture", "fact vs picture"} <= kinds and "paper photo" not in kinds
    rows = [r for r in verdicts if r.get("card") == "bolt"]
    dated = chain.created("bolt", t, rows)
    assert (dated["status"], dated["ours"]["picture"]) == ("new", "P-000000000001") and "dated" not in dated   # not 7
    assert dated["named_only_earlier"]["picture"] == "P-000000000007"
    chain.inputs.first_repo.append({"id": "q0", "card": "Bolt", "date": iso(T - 2 * DAY), "author": "Chestnut",
                                    "subject": "Bolt"})
    hinted = chain.created("bolt", t, rows)
    assert hinted["first_repo_earlier"]["row"] == "Q-q0"                 # a push notice: a Created hint (review) only


def test_a_draft_after_an_mdw_draft_but_before_the_card_folders_enters():
    """s3 gap 2 fix: a card whose first record is an MDW draft (oldrows psd_first.json folders) keeps its Discord drafts
    posted before its first card-folder record (Tainted Bethany 2023-09-09 12:55); git looks and jokes stay out."""
    verdicts = [picture(1, T + 110 * DAY), picture(2, T + 120 * DAY, "version_sure", look=1),
                picture(3, T + 130 * DAY, joke=True), picture(4, T + 210 * DAY)]
    got = inputs(verdicts)
    assert not dc.Chain(got).discord_rows()[0].get("bolt")             # after the first git record (fs)
    got.cards["bolt"]["fs_folder"] = T + 200 * DAY
    found, _ = dc.Chain(got).discord_rows()
    assert [r["id"] for r in found["bolt"]] == ["P-000000000001"]


def test_a_picture_that_is_no_card_on_its_own_dates_only():
    """Owner 11-16 23:10/23:15: a picture the eye found is no card on its own (match/eye.json dates_only: a cut-off
    piece, a screenshot of several cards) enters no timeline, split halves' neither, and asks nothing (a read-only
    'dates only' item shows it), yet still dates Created, also the other cards printed plainly on it (dates_also)."""
    shot = "a phone screenshot of a sheet of two cards, both cut off"
    verdicts = [picture(1, T), picture(2, T - DAY, dates_only=shot), picture(3, T + DAY, "version_unsure",
                                                                             dates_only=shot)]
    got = inputs(verdicts)
    got.dates_also = {"P-000000000002": ["pike", "bolt", "gone"]}
    chain = dc.Chain(got)
    found, split = chain.discord_rows()
    assert [r["id"] for r in found["bolt"]] == ["P-000000000001"] and not found.get("pike")
    items = {i["picture"]: i for i in chain.review.values()}
    assert {p: (i["kind"], i.get("read_only")) for p, i in items.items()} == {
        "P-000000000002": ("dates only", True), "P-000000000003": ("dates only", True)}
    assert items["P-000000000002"]["also"] == ["pike"] and "also" not in items["P-000000000003"]
    t = chain.card("bolt", found["bolt"], [])
    assert [e["id"] for e in t["entries"]] == ["D-P-000000000001"]
    made = chain.created("bolt", t, verdicts)
    assert (made["ours"]["picture"], made["ours"]["dates_only"]) == ("P-000000000002", True)
    pike = chain.created("pike", chain.card("pike", [], []), [])        # printed beside bolt: dated by it too
    assert (pike["ours"]["picture"], pike["ours"]["dates_only"], pike["ours"]["beside"]) == ("P-000000000002", True,
                                                                                            "bolt")
    group = {"id": "LG-1", "split_of": {"name": "Twin", "into": ["bolt", "pike"]}}
    chain = dc.Chain(inputs([]))
    split = defaultdict(list)
    chain.to_halves(group, picture(4, T, dates_only=shot), split)
    assert not split and chain.tally["discord: no card on its own (by eye): dates only, not shown"] == 2


def test_a_crop_by_eye_is_shown_and_a_repo_png_s_eye_fields_win():
    """Owner 11-16 23:15 (2): the whole card cut out of a bigger picture (match/eye.json crops, set on the row as a
    cell's box) enters its timeline as that crop, with a read-only 'cropped' item; an old/first repo PNG's fields by
    eye (readers' eye files, o:<blob>) win over its folder's or PSD's, on its row and on a git look posted before."""
    shot = "a TTS table: the whole card cut out"
    row = old_row("Bolt", T + 100 * DAY, "a" * 40, 5)
    row["fields"]["stars"] = {"value": 3, "method": "folder", "confidence": "sure"}
    got = inputs([picture(1, T, box=[0.2, 0.1, 0.5, 0.9], crop=shot), picture(2, T + DAY, "version_sure", look=1)],
                 old_rows=[row])
    got.old_eye = {"a" * 40: {"stars": 4}}
    chain = dc.Chain(got)
    found, split = chain.discord_rows()
    t = chain.card("bolt", found["bolt"], [])
    first = t["entries"][0]["pictures"][0]
    assert (first["id"], first["box"], first["crop"]) == ("P-000000000001", [0.2, 0.1, 0.5, 0.9], True)
    assert [(i["kind"], i.get("read_only"), i["why"]) for i in chain.review.values() if i["kind"] != "field"] == \
        [("cropped", True, shot)]
    stars = {e["id"]: (e["fields"]["stars"]["value"], e["fields"]["stars"]["method"]) for e in t["entries"]
             if dc.is_sure(e["fields"].get("stars"))}
    assert stars == {"D-P-000000000002": (4, "git look: eye"), "O-" + row["id"]: (4, "eye")}


def test_the_physical_card_photos_date_exactly_three_cards():
    """Owner Q6 (11-14 22:15/22:20): the photos are in no timeline (no paper entry, no paper review item); Created
    'Before 2020' (sorting first, source 'physical card photo', read only) for Pep Drink and Wham Bam Bone, the
    earliest appearance kept as dated; M'uru is the Morgue's (discord_morgue); no other card."""
    assert dc.PHOTO_CREATED == {"pep-drink": "Pep Drink", "wham-bam-bone": "Wham Bam Bone"}
    assert dc.PHOTO_MORGUE == {"LC-1c11be0504": "M'uru"} and dc.PHOTO_SOURCE == "physical card photo"
    assert dict(q[:2] for q in dc.OWNER_ANSWERS)["Q6"] == "photos left out" and "M'uru" in dc.OWNER_ANSWERS[-1][3]
    verdicts = [picture(1, T, card="pep-drink"), picture(2, T + DAY, card="wham-bam-bone"), picture(3, T)]
    got = inputs(verdicts, created={"Pep Drink": {"date": "2020-07-22", "from": "discord"},
                                    "Wham Bam Bone": {"date": "2020-07-22", "from": "discord"},
                                    "Bolt": {"date": dc.day(T), "from": "discord"}})
    for cid, name in dc.PHOTO_CREATED.items():
        got.cards[cid] = {"key": name, "name": name, "type": "item", "fs": T + 100 * DAY, "fs_from": "old git"}
    chain = dc.Chain(got)
    found, split = chain.discord_rows()
    made = {}
    for cid in ("pep-drink", "wham-bam-bone", "bolt"):
        t = chain.card(cid, found[cid], [])
        assert [e["kind"] for e in t["entries"]] == ["discord"]
        made[cid] = chain.created(cid, t, [r for r in verdicts if r["card"] == cid])
    for cid, n in (("pep-drink", 1), ("wham-bam-bone", 2)):
        c = made[cid]
        assert (c["status"], c["date"], c["sort"], c["proposed"]) == ("earlier", dc.BEFORE_DAY, dc.BEFORE_SORT, True)
        assert c["ours"] == {"date": None, "day": "Before 2020", "source": "physical card photo", "owner": "Q6"}
        assert c["dated"]["picture"] == f"P-{n:012d}"
        item = chain.review[c["review"]]
        assert item["read_only"] and item["answer"] == "Q6" and "owner's ruling" in item["why"]
    bolt = made["bolt"]
    assert (bolt["status"], bolt["ours"]["source"], bolt["ours"]["picture"]) == ("same", "discord", "P-000000000003")
    assert not any(i["kind"] == "paper photo" for i in chain.review.values())
    assert sum(1 for i in chain.review.values() if i.get("read_only")) == 2           # only the two
    assert dc.photo_missing(got.cards) == []
    got.cards["pep-drinks"] = got.cards.pop("pep-drink")                              # renamed: the ruling lapses
    assert dc.photo_missing(got.cards) == ["pep-drink (Pep Drink)"]
    text = dc.stats_text({}, {}, [], Counter(), Counter(), dc.photo_missing(got.cards))
    assert "physical card photo Created, owner's cards missing (lapsed): pep-drink (Pep Drink)" in text
    assert dc.photo_missing({"LC-1c11be0504": {"name": "M'uru"}}, dc.PHOTO_MORGUE) == []
    assert dc.photo_missing({"LC-0000000000": {"name": "M'uru"}}, dc.PHOTO_MORGUE) == ["LC-1c11be0504 (M'uru)"]


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
    told = lambda t: t["entries"][0]["fields"]["stats"]          # both halves told by eye: each half's own, sure
    assert told(lorn)["value"] == {"hp": 4, "def": 2, "atk": 4, "spd": 5} and told(lorn)["confidence"] == "sure"
    assert told(pike)["value"] == {"hp": 6, "def": 4, "atk": 6, "spd": 5} and "top half" in told(pike)["half"]
    assert lorn["entries"][1]["fields"]["stats"]["confidence"] == "review"     # the old repo half read by no eye
    made = {c: chain.created(c, t, [r for r in verdicts if r.get("card") == c]) for c, t in
            (("pike", pike), ("lorn", lorn), ("bolt", chain.card("bolt", [], [])))}
    assert made["pike"]["status"] == "earlier" and made["pike"]["ours"]["as"] == "Twin Pike"
    assert made["lorn"]["status"] == "later" and made["lorn"]["date"] == files["Lorn"]["date"]   # never later
    assert made["bolt"]["status"] == "hand" and made["bolt"]["date"] == "2001-01-01"
    row = next(e for e in lorn["entries"] if e["kind"] == "old repo")                      # its own review item
    assert chain.review[row["fields"]["stats"]["review"]]["proposed"] == {"hp": 3, "def": 2, "atk": 3, "spd": 4}


def test_numbers_local_edits_and_unbacked_star_changes(tmp_path):
    """A swapped number counts (5/7/6/6 -> 5/7/7/6); a picture alike but for one area (an icon added) is its own
    look; an icon-read star count changing with nothing to back it goes to review, as does one a later changelog
    line changes the stars to; a repost's own contrary read takes a reader's sure stars to review."""
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
    later = dict(fact, date=iso(T + 400 * DAY), change={"field": "stars", "to": 1})     # Charge: 2 read 1, then 'to 1'
    cases = [([version(1, 0, 1), version(2, 0.01, 1), later], {"E1", "E2"}),
             ([version(1, 0, 1), version(2, 0.01, 1, "eye"), later], set()),            # eye backs the run
             ([version(1, 0, 1), version(2, 200, 2, "icon"), later], set()),            # a version between
             ([version(1, 398, 1), later], set())]                                      # its port picture
    for entries, held in cases:
        for e in entries:
            if e["kind"] == "discord" and e["fields"]["stars"]["method"] == "icon":
                e["fields"]["stars"] = dc.unsure("icon", "read under its bar", 2)
        chain.star_fact_guard(entries)
        assert {e["id"] for e in entries if e["kind"] != "text" and e["fields"]["stars"].get("why", "").startswith(
            "a later changelog")} == held
    mine = {"stars": dc.sure(2, "icon+known")}
    other = {"stars": dc.unsure("icon", "read under its bar", 4)}
    assert dc.merge_fields(mine, other, "repost P-1")["stars"]["confidence"] == "review"
    assert dc.is_sure(dc.merge_fields({"stars": dc.sure(2, "eye")}, other, "repost P-1")["stars"])
    assert dc.is_sure(dc.merge_fields(mine, other, "the picture", fill=False)["stars"])


def test_the_owner_s_answers():
    """The review page's answers (rulings.json): a named-only picture confirmed enters and dates Created, a look told
    (by its version's key: the same look after the numbers shift; gone, not applied), 'not this card', a card named,
    a field given, an unsure fact right, Created picked with its source, earlier hints only
    before the Created date in use, a map to a card that isn't live, the ids acted on; an answer carrying what a rerun
    no longer proposes (a field's value, a picture's card, a Created day, a hint's day, a map's card): not applied."""
    rid = dc.review_id
    text = {"attachments/1.png": {"title": {"value": "Bolt", "sure": False, "conf": 0.5, "method": "ocr+name"}}}
    verdicts = [picture(1, T, created_review="named only, before its first picture evidence (x)"),
                picture(2, T + DAY, "version_unsure"), picture(3, T + 2 * DAY, "version_unsure"),
                dict(picture(4, T + 3 * DAY, "unsure", card=None), cards=None, maybe=["bolt"])]
    facts = [{"id": "f9", "card": "Bolt", "source": "changelog", "confidence": "unsure", "date": iso(T + 4 * DAY),
              "field": "hp", "by": -2, "unit": "unknown", "channel": "update-01"}]
    rulings = {rid("named only", "P-000000000001"): {"choice": "yes", "value": "bolt"},
               rid("look not told", "P-000000000002"): {"choice": "look", "value": "b" + "a" * 20},
               rid("look not told", "P-000000000003"): {"choice": "no", "value": "bolt"},
               "match:P-000000000004": {"choice": "card", "value": "bolt"},
               rid("field", "bolt|D-P-000000000001|name"): {"choice": "yes", "value": "Bolt"},
               "R1": {"kind": "unsure fact", "choice": "yes", "value": "points", "item": {"fact": "f9"}},
               rid("created earlier", "bolt"): {"choice": "file", "value": dc.day(T + 9 * DAY)},
               rid("changelog earlier", "bolt"): {"choice": "yes", "value": dc.day(T + 4 * DAY)},
               "oldrows-m1": {"kind": "unsure map", "choice": "yes", "value": "Zed Old"}}
    gone = {"id": "oldrows-m1", "kind": "unsure map", "card": "Zed Old", "status": "archived",
            "rows": [old_row("Zed", T + 50 * DAY, "d" * 40, 5)]}
    got = inputs(verdicts, text, facts=facts, created={"Bolt": {"date": dc.day(T + 9 * DAY), "from": "discord"}},
                 old_review=[gone])
    got.rulings = rulings
    done = dc.owner_rows(got)
    assert done["owner: not this card"] == 1 and "Zed Old" not in got.old_rows                 # no live card: none
    assert got.used == {rid("named only", "P-000000000001"), rid("look not told", "P-000000000002"),
                        rid("look not told", "P-000000000003"), "match:P-000000000004", "R1"}
    chain = dc.Chain(got)
    found, split = chain.discord_rows()
    t = chain.card("bolt", found["bolt"], split.get("bolt", []))
    entries = {e["id"]: e for e in t["entries"]}
    assert set(entries) == {"D-P-000000000001", "D-P-000000000002", "F-f9"}
    assert entries["D-P-000000000001"]["fields"]["name"] == dc.sure("Bolt", "owner", ruled=rid(
        "field", "bolt|D-P-000000000001|name"))
    assert entries["D-P-000000000002"]["look"]["git"]["commit"] == "c1"
    assert entries["F-f9"]["change"]["unit"] == "points"
    assert {i["picture"] for i in chain.review.values() if i["kind"] == "look not told"} == {"P-000000000004"}
    made = chain.created("bolt", t, [r for r in verdicts if r.get("card") == "bolt"])
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
    ours = {rid("created later", "bolt"): {"choice": "ours", "value": dc.day(T + 3 * DAY)}}
    for answers, status, hint in (({}, "later", None), (ours, "owner", dc.day(T + 2 * DAY))):
        later.rulings = answers
        chain = dc.Chain(later)
        found, split = chain.discord_rows()
        t = chain.card("bolt", found["bolt"], split.get("bolt", []))
        made = chain.created("bolt", t, later.verdicts)
        assert (made["status"], (made.get("changelog_earlier") or {}).get("day")) == (status, hint)
    assert made["date"] == dc.day(T + 3 * DAY) and made["proposed"] and made["owner"]["source"]["picture"] == \
        "P-000000000021" and made["owner"]["source"]["from"] == "ours"
    for n, date in ((2, T + 2 * DAY), (1, T + 3 * DAY)):         # the hint's yes on its own day, then another's
        later.rulings = {**ours, rid("changelog earlier", "bolt"): {"choice": "yes", "value": dc.day(T + n * DAY)}}
        chain = dc.Chain(later)
        found, split = chain.discord_rows()
        made = chain.created("bolt", chain.card("bolt", found["bolt"], split.get("bolt", [])), later.verdicts)
        assert made["date"] == dc.day(date) and (rid("changelog earlier", "bolt") in chain.used) == (n == 2)
    assert chain.tally["owner: a changelog line hint's yes, its day changed (not applied)"] == 1
    later.rulings = {**ours, rid("changelog earlier", "bolt"): {"choice": "yes", "value": dc.day(T + 2 * DAY),
                                                                "source": "triage (x)"}}
    chain = dc.Chain(later)                                    # the triage's yes never goes below the owner's own pick
    found, split = chain.discord_rows()
    made = chain.created("bolt", chain.card("bolt", found["bolt"], split.get("bolt", [])), later.verdicts)
    assert made["date"] == dc.day(T + 3 * DAY) and rid("changelog earlier", "bolt") not in chain.used
    assert chain.tally["owner: a triage changelog line hint's yes below the owner's day (not applied)"] == 1
    # a rerun proposing other than what the answer carries: not applied, not used, the item back on the page
    live = {"id": "oldrows-m2", "kind": "unsure map", "card": "Bolt", "status": "live",
            "rows": [old_row("Bolt", T + 60 * DAY, "e" * 40, 5)]}
    drift = {rid("named only", "P-000000000001"): {"choice": "yes", "value": "bolt"},
             rid("look not told", "P-000000000003"): {"choice": "no", "value": "pike"},          # another card's
             rid("field", "bolt|D-P-000000000001|name"): {"choice": "yes", "value": "Bolt II"},  # read again: 'Bolt'
             rid("created earlier", "bolt"): {"choice": "file", "value": dc.day(T + 8 * DAY)},   # the file says T+9
             "oldrows-m2": {"kind": "unsure map", "choice": "yes", "value": "Pike"}}
    again = inputs([picture(1, T, created_review="named only, before its first picture evidence (x)"),
                    picture(3, T + 2 * DAY, "version_unsure")], text,
                   created={"Bolt": {"date": dc.day(T + 9 * DAY), "from": "discord"}}, old_review=[live])
    again.rulings = drift
    done = dc.owner_rows(again)
    assert again.used == {rid("named only", "P-000000000001")} and not again.old_rows["Bolt"]
    assert done["owner: a look not told answer given on another card (not applied)"] == 1 and \
        done["owner: an unsure old map's yes, its card changed (not applied)"] == 1
    chain = dc.Chain(again)
    found, split = chain.discord_rows()
    t = chain.card("bolt", found["bolt"], split.get("bolt", []))
    name = next(e for e in t["entries"] if e["id"] == "D-P-000000000001")["fields"]["name"]
    assert name["confidence"] == "review" and chain.review[name["review"]]["proposed"] == "Bolt"
    assert chain.tally["owner: discord field name yes, its value no longer proposed (not applied)"] == 1
    assert "P-000000000003" in {i["picture"] for i in chain.review.values() if i["kind"] == "look not told"}
    made = chain.created("bolt", t, again.verdicts)         # the owner's own day held until answered again
    assert (made["status"], made["date"], made["owner"]["held"]) == ("owner", dc.day(T + 8 * DAY), dc.day(T + 8 * DAY))
    assert chain.tally["owner: Created picked, its day no longer the one shown (kept as the owner's day)"] == 1
    chain.rulings["hint:LC-x:bolt"] = {"kind": "morgue hint", "choice": "yes", "source": "triage (x)",
                                       "item": {"card": "bolt", "day": dc.day(T - 3 * DAY), "entry": "LC-x"}}
    assert chain.created("bolt", t, again.verdicts)["date"] == dc.day(T + 8 * DAY)    # no triage hint below it
    assert chain.tally["owner: a triage Morgue hint's yes below the owner's day (not applied)"] == 1
    chain.rulings[rid("created earlier", "bolt")]["source"] = "triage (x)"           # the triage's: not applied
    made = chain.created("bolt", t, again.verdicts)
    assert made["status"] == "owner" and made["date"] == dc.day(T - 3 * DAY) and "held" not in made["owner"]
    assert chain.tally["owner: Created picked, its day no longer the one shown (not applied)"] == 1
    del chain.rulings["hint:LC-x:bolt"]
    named = inputs([picture(1, T, created_review="named only, before its first picture evidence (x)")])
    named.rulings = {rid("named only", "P-000000000001"): {"choice": "yes", "value": "pike"}}
    assert dc.owner_rows(named)["owner: a named only answer given on another card (not applied)"] == 1 and \
        not named.used and named.verdicts[0]["created_review"]
    # a look told by its first version's key: a new version shifts the numbers, the same look; its version gone, or an
    # older page's number: not applied (the picture stays to review)
    told = rid("look not told", "P-000000000002")
    new = {"key": "b" + "c" * 20, "card": "bolt", "ts": T + 200 * DAY, "look": 0, "commit": "c3", "path": "x.png"}
    shift = lambda vs: [new] + [{**v, "look": v["look"] + 1} for v in vs]
    for value, change, commit in (("b" + "a" * 20, shift, "c1"), ("b" + "a" * 20, lambda vs: vs[1:], None),
                                  (1, shift, None)):
        moved = inputs([picture(2, T + DAY, "version_unsure")])
        moved.versions["bolt"] = change(moved.versions["bolt"])
        moved.rulings = {told: {"choice": "look", "value": value}}
        done = dc.owner_rows(moved)
        chain = dc.Chain(moved)
        found, split = chain.discord_rows()
        if commit:
            t = chain.card("bolt", found["bolt"], split.get("bolt", []))
            assert t["entries"][0]["look"]["git"]["commit"] == commit and moved.used == {told}
        else:
            assert not moved.used and done["owner: a git look told, its version gone (not applied)"] == 1 and \
                [i["picture"] for i in chain.review.values() if i["kind"] == "look not told"] == ["P-000000000002"]


def test_an_owner_mapped_psd_gives_its_row_or_counts_apart():
    """An unsure map of PSDs alone (oldgit_rows psd_rows): a 'yes' puts its PSD on the card's row of that commit
    (replaces) or adds its own look; a 'yes' with no rows adds nothing and is never counted as applied."""
    base = old_row("Bolt", T + 60 * DAY, "e" * 40, 5)
    given = {**old_row("Bolt", T + 60 * DAY, "e" * 40, 6), "replaces": base["id"], "psd_file": "Items/Bolt.psd"}
    alone = {**old_row("Bolt", T + 70 * DAY, "f" * 40, 5), "psd_only": True}
    maps = [{"id": "oldrows-p1", "kind": "unsure map", "card": "Bolt", "status": "live", "rows": [given, alone]},
            {"id": "oldrows-p2", "kind": "unsure map", "card": "Bolt", "status": "live", "rows": []}]
    got = inputs([picture(1, T)], old_rows=[base], old_review=maps)
    got.rulings = {"oldrows-p1": {"kind": "unsure map", "choice": "yes", "value": "Bolt"},
                   "oldrows-p2": {"kind": "unsure map", "choice": "yes", "value": "Bolt"}}
    done = dc.owner_rows(got)
    assert [(r["blob"][0], r["fields"]["stats"]["value"]["hp"], r.get("owner")) for r in got.old_rows["Bolt"]] == \
        [("e", 6, True), ("f", 5, True)] and got.used == {"oldrows-p1"}
    assert done["owner: an unsure old map yes"] == 1 and done["owner: an unsure old map yes, no rows (nothing added)"] == 1


def test_a_git_look_s_row_item_or_the_picture_s_own_answer():
    """A git look's field its row leaves to review keeps the row's item (its answers apply); an answer given on the
    picture's own item before the row had one applies too (a first-repo PNG read by the text readers later)."""
    row = old_row("Bolt", T + 100 * DAY, "a" * 40, 5)
    row["fields"]["ability"] = {"value": "Bite 3.", "method": "ocr", "confidence": "review", "why": "under its bar"}
    row["fields"]["name"] = {"value": "Bolt", "method": "psd", "confidence": "review", "why": "x"}
    got = inputs([picture(1, T, "version_sure", look=1)], old_rows=[row],
                 old_review=[{"id": "oldrows-ab", "row": row["id"], "field": "ability", "proposed": "Bite 3."},
                             {"id": "oldrows-nm", "row": row["id"], "field": "name", "proposed": "Bolt"}])
    got.rulings = {dc.review_id("field", "bolt|D-P-000000000001|ability"): {"choice": "value", "value": "Bite 3"},
                   "oldrows-nm": {"choice": "yes", "value": "Bolt"}}
    chain = dc.Chain(got)
    found, split = chain.discord_rows()
    e = next(e for e in chain.card("bolt", found["bolt"], [])["entries"] if e["id"] == "D-P-000000000001")
    assert (e["fields"]["ability"]["value"], e["fields"]["ability"]["ruled"]) == (
        "Bite 3", dc.review_id("field", "bolt|D-P-000000000001|ability"))
    assert (e["fields"]["name"]["value"], e["fields"]["name"]["ruled"]) == ("Bolt", "oldrows-nm")


def test_a_git_look_takes_the_picture_s_own_field_and_keeps_a_typed_answer():
    """A field the git look's row has none of (a PNG with no PSD: no types) is the picture's own sure read; a known
    text (ocr+known) keeps an answer typed off the picture with the same words (its row's item or its own)."""
    row = old_row("Bolt", T + 100 * DAY, "a" * 40, 5)
    row["fields"]["ability"] = {"value": "Bite 3.(OPT)", "method": "ocr+known", "confidence": "sure"}
    got = inputs([picture(1, T, "version_sure", look=1)], old_rows=[row])
    got.text_truth = {"d:" + "1".zfill(64): {"types": ["ice"]}}
    own = dc.review_id("field", "bolt|D-P-000000000001|ability")
    got.rulings = {own: {"choice": "value", "value": "Bite 3. (OPT)", "source": "triage (owner-delegated)"},
                   dc.row_item_id(row["id"], "ability"): {"choice": "value", "value": "Bite 4. (OPT)"}}
    chain = dc.Chain(got)
    found, split = chain.discord_rows()
    e = next(e for e in chain.card("bolt", found["bolt"], [])["entries"] if e["id"] == "D-P-000000000001")
    assert (e["fields"]["types"]["value"], e["fields"]["types"]["method"]) == (["ice"], "eye")
    assert (e["fields"]["ability"]["value"], e["fields"]["ability"]["method"], e["fields"]["ability"]["ruled"]) == (
        "Bite 3. (OPT)", "triage", own)
    o = next(e for e in chain.card("bolt", found["bolt"], [])["entries"] if e["kind"] == "old repo")
    assert o["fields"]["ability"]["value"] == "Bite 3.(OPT)"                       # other words: the known text


def test_a_picture_s_id_keeps_its_card_and_the_triage_s_method():
    """BRIEF 11-15 20:50: a picture-keyed item (look not told, named only, joke picture) keeps its id while the page
    asks about the card it was first shown with (derived/review/ids.json); asked about another card it gets a new id,
    so an answer given for the first card never lands on the second. A field the triage gave is method 'triage'."""
    rid = dc.review_id("look not told", "P-000000000006")
    assert dc.picture_id("look not told", "P-000000000006", "pike", {}) == rid              # never shown: as before
    assert dc.picture_id("look not told", "P-000000000006", "bolt", {rid: {"card": "bolt"}}) == rid
    new = dc.picture_id("look not told", "P-000000000006", "pike", {rid: {"card": "bolt"}})
    assert new not in (rid, dc.review_id("look not told", "P-000000000006|bolt"))
    got = inputs([picture(6, T + 5 * DAY, "version_unsure", card="pike")])
    got.issued = {rid: {"card": "bolt"}}
    got.rulings = {rid: {"choice": "look", "value": "b" + "a" * 20}}                       # given for Bolt's look
    dc.owner_rows(got)
    chain = dc.Chain(got)
    chain.discord_rows()
    assert not got.used and set(chain.review) == {new} and chain.review[new]["card"] == "pike"
    text = {"attachments/1.png": {"title": {"value": "Bolt", "sure": False, "conf": 0.5, "method": "ocr+name"}}}
    field = dc.review_id("field", "bolt|D-P-000000000001|name")
    for source, method in ((None, "owner"), ("triage (owner-delegated 2026-11-16, two judges agreed)", "triage")):
        got = inputs([picture(1, T)], text)
        got.rulings = {field: {"choice": "value", "value": "Bolt", **({"source": source} if source else {})}}
        chain = dc.Chain(got)
        found, split = chain.discord_rows()
        t = chain.card("bolt", found["bolt"], split.get("bolt", []))
        assert t["entries"][0]["fields"]["name"] == dc.sure("Bolt", method, ruled=field)
    got = inputs([picture(1, T)], text)                       # an item's kinds given in an answer: not collected
    got.cards["bolt"]["type"] = "item"
    kinds = {dc.review_id("field", f"bolt|D-P-000000000001|types"): {"choice": "value", "value": ["equip", "fire"]},
             dc.review_id("field", f"bolt|D-P-000000000001|stars"): {"choice": "value", "value": 2}}
    got.rulings = kinds
    chain = dc.Chain(got)
    found, split = chain.discord_rows()
    fields = chain.card("bolt", found["bolt"], split.get("bolt", []))["entries"][0]["fields"]
    assert fields["types"]["value"] == ["fire"] and fields["stars"]["value"] == 2


def test_facts_by_a_step_and_by_the_pictures_around_them():
    chain = dc.Chain(inputs([]))
    at = lambda days: iso(T + days * DAY)
    pic = lambda n, days, stats=None, stars=None: {"id": f"D-{n}", "kind": "discord", "date": at(days), "fields": {
        **({"stats": dc.sure(stats, "picture")} if stats else {}),
        **({"stars": dc.sure(stars, "psd")} if stars else {})}}
    fact = lambda n, days, **change: {"id": f"F-f{n}", "kind": "text", "date": at(days), "change": change}
    entries = [pic(1, 0, {"hp": 8, "def": 5, "atk": 6, "spd": 3}, 2), fact("a", 1, field="atk", by=1),
               fact("d", 2, field="def", by=-1), fact("e", 2, field="hp", to=9, **{"from": 8}),
               fact("f", 2, field="stars", to=1, **{"from": 2}), pic(2, 3, {"atk": 7}), fact("b", 4, field="spd", by=2),
               fact("c", 5, field="spd", by=-2), pic(3, 6, {"spd": 5}), pic(4, 300, stars=2),
               pic(5, 400, {"hp": 9, "def": 4}, 1)]
    kept = {e["id"]: e for e in chain.check_facts("bolt", entries)}
    assert kept["F-fa"]["confirmed_by"] == "D-2"                   # 6 + 1, read 3 days on
    assert "F-fc" not in kept and kept["F-fb"]["confirmed_by"] == "D-3"    # 3 + 2 - 2 read 5: the -2 to review
    item = next(i for i in chain.review.values() if i["kind"] == "fact vs picture")
    assert (item["fact"], item["fact_to"], item["fact_by"], item["picture_reads"]) == ("fc", 3, -2, 5)
    assert kept["F-fd"]["confirmed_by"] == kept["F-fe"]["confirmed_by"] == "D-5"   # far, the move the pictures show
    assert "confirmed_by" not in kept["F-ff"]                       # a picture between still reads 2 stars
    widened = [pic(6, 0, {"hp": 4}), fact("g", 10, field="hp", to=5), fact("h", 11, field="hp", to=7),
               pic(7, 12, {"hp": 9})]
    assert [e["id"] for e in chain.check_facts("bolt", widened)] == ["D-6", "D-7"]    # h gone, g's window reaches D-7


def test_named_only_after_created_psds_and_header_types():
    """A named-only picture on or after the file's Created day enters; an earlier one stays for review. A stat-less
    card's picture reading no bar label has no stats. A posted PSD's exact fields go onto the look posted with it,
    else it is a version of its own. A git header match gives the types, an eye reference's header (eras A-C) too."""
    verdicts = [picture(1, T, created_review="named only, before its first picture evidence (x)"),
                picture(2, T - 30 * DAY, created_review="named only, before its first picture evidence (x)")]
    inp = inputs(verdicts, created={"Bolt": {"date": dc.day(T), "from": "discord"}})
    assert dc.named_on_or_after(inp) == 1 and "created_review" not in verdicts[0] and verdicts[1]["created_review"]
    inp = inputs([picture(3, T)], bars={"attachments/3.png": {"labels": 0, "stats": None}})
    inp.statless = {"Bolt"}
    assert "stats" not in dc.picture_fields(inp, {**inp.verdicts[0]}, "creature")
    inp.statless = set()
    assert dc.picture_fields(inp, {**inp.verdicts[0]}, "creature")["stats"]["confidence"] == "review"
    inp.header = {"P-000000000003": {"card": "bolt", "types": ["fire"], "ncc": 0.97, "version": "k"}}
    inp.text = {"attachments/3.png": {"types": {"sure": False}}}
    assert dc.picture_fields(inp, {**inp.verdicts[0]}, "creature")["types"]["method"] == "git header"
    inp.text_truth = {"d:" + "3".zfill(64): {"ability": "", "title": ""}}   # a blank box by eye: no ability
    got = dc.picture_fields(inp, {**inp.verdicts[0]}, "creature")
    assert (got["ability"]["value"], got["ability"]["method"]) == ("", "eye") and got["name"]["method"] != "eye"
    inp.text_truth = {}
    inp.header = {}
    inp.header_eye = {"P-000000000003": {"card": "bolt", "types": ["rock"], "ncc": 0.9, "ref": "P-9"}}
    got = dc.picture_fields(inp, {**inp.verdicts[0]}, "creature")["types"]
    assert (got["value"], got["method"], got["ref"]) == (["rock"], "eye header", "P-9")
    rows = [{**picture(4, T, posts=[{"message": "44"}]), "whole": H}]
    psd = lambda n, msg, whole: {"id": f"S-{n}", "card": "bolt", "picture": f"derived/psds/pictures/S-{n}.webp",
                                 "look": {"whole": whole}, "posts": [{"message": msg, "date": iso(T + n), "channel":
                                                                      "new-cards", "bot": False}],
                                 "fields": {"name": {"value": "Bolt", "confidence": "sure"},
                                            "ability": {"value": "Zap", "confidence": "sure"}}}
    inp = inputs(rows)
    inp.psds = {"bolt": [psd(1, "44", FAR), psd(2, "99", FAR)]}
    chain = dc.Chain(inp)
    entries = chain.discord_entries("bolt", rows, [])
    assert [e["id"] for e in entries] == ["D-P-000000000004", "D-S-2"]
    assert entries[0]["fields"]["ability"]["method"] == "psd (posted PSD)" and entries[0]["psds"] == ["S-1"]
    assert entries[1]["pictures"][0]["file"].endswith("S-2.webp") and entries[1]["fields"]["stats"]["confidence"] \
        == "review"


def test_a_posted_psd_s_look():
    """A posted PSD goes on its own message's look before a nearer one, by bits only onto a look no PSD read
    otherwise holds; a field another PSD set is never overwritten (two that differ: review)."""
    near = {"p": "0" * 15 + "1", "d": "0" * 16, "q": "0" * 64}
    entries = [{"id": f"D-{n}", "kind": "discord", "fields": {}} for n in (1, 2)]
    seen = {"D-1": ({"11"}, [H]), "D-2": ({"22"}, [near])}
    r = {"id": "S-9", "read": {"name": "Bolt", "types": ["fire"], "ability": "Zap"}}
    assert dc.psd_home(r, {"22"}, H, entries, seen, {}) == (entries[1], "message")
    assert dc.psd_home(r, {"33"}, H, entries, seen, {}) == (entries[0], "bits")
    other = {"id": "S-8", "read": {"name": "Bolt", "types": [], "ability": "Zap"}}
    assert dc.psd_home(r, {"33"}, H, entries, seen, {"D-1": [other], "D-2": [other]}) == (None, None)
    rows = [{**picture(4, T, posts=[{"message": "44"}]), "whole": H}]
    psd = lambda n, ability: {"id": f"S-{n}", "card": "bolt", "picture": f"derived/psds/pictures/S-{n}.webp",
                              "look": {"whole": FAR}, "read": {"name": "Bolt", "ability": ability},
                              "posts": [{"message": "44", "date": iso(T + n), "channel": "new-cards", "bot": False}],
                              "fields": {"name": {"value": "Bolt", "confidence": "sure"},
                                         "ability": {"value": ability, "confidence": "sure"}}}
    inp = inputs(rows)
    inp.psds = {"bolt": [psd(1, "Zap"), psd(2, "Zap twice")]}
    got = dc.Chain(inp).discord_entries("bolt", rows, [])[0]
    assert got["psds"] == ["S-1", "S-2"] and got["fields"]["name"]["psd"] == "S-1"
    assert got["fields"]["ability"]["confidence"] == "review" and got["fields"]["ability"]["proposed"] == "Zap"


def test_header_types_never_cross_to_a_picture_whose_header_misses():
    """A repost whose header icons match no git version keeps a git header's types out of the look (review)."""
    miss = {**dc.unsure("icon+known", "x", ["light"]), "header_miss": True}
    hit = dc.sure(["light"], "git header", 0.95)
    for a, b in ((miss, hit), (hit, miss)):
        got = dc.merge_fields({"types": a}, {"types": b}, "repost P-1")
        assert got["types"]["confidence"] == "review" and got["types"]["header_miss"]
    assert dc.merge_fields({"types": dc.unsure("icon+known", "x", ["light"])}, {"types": hit},
                           "repost P-1")["types"]["confidence"] == "sure"


def test_names_of_other_cards_item_kinds_and_the_first_title():
    """A title read snapped to another live card's name goes to review, unless it is this card's own (alias, old repo
    title). An item's kinds are no types change. git_first keeps no rename when the last look is the first git
    version's and its title read is in that version's printed title."""
    inp = inputs([picture(5, T)], text={"attachments/5.png": {"title": {"sure": True, "value": "Pike", "conf": 1.0,
                                                                         "method": "ocr+name", "ocr": "P1ke"}}})
    got = dc.picture_fields(inp, inp.verdicts[0], "creature")["name"]
    assert got["confidence"] == "review" and got["proposed"] == "Pike"
    inp.old_rows["Bolt"].append({"old_id": "Pike", "fields": {}})
    assert dc.picture_fields(inp, inp.verdicts[0], "creature")["name"]["value"] == "Pike"
    item = lambda *ts: {"types": dc.sure(list(ts), "psd")}
    assert "types" not in dc.compare(item("offense"), item("support"), "item")
    assert dc.compare(item("offense", "nme"), item("support"), "nme")["types"] == {"add": [], "drop": ["nme"]}
    assert dc.compare(item("offense"), item("support"), "creature")["types"]["add"] == ["support"]
    stop = {"commit": "c9", "path": "N.M.E/Bun.png"}
    last = {"kind": "discord", "look": {"git": dict(stop)}, "fields": {"name": dc.sure("Bun", "ocr+name")}}
    data = {"name": dc.sure("Bun Or... is it", "psd (card data)")}
    versions = [{"commit": "c9", "path": "N.M.E/Bun.png", "title": "Or... is it Bun", "ts": T}]
    assert dc.first_changes(last, data, "nme", versions, stop) == {"same_title": True}
    assert dc.first_changes({**last, "look": {"draft": True}}, data, "nme", versions, stop)["name"] == ["Bun",
                                                                                                    "Bun Or... is it"]
    versions = [{"commit": "c9", "path": "N.M.E/Bun.png", "title": "Or... is it Bun", "ts": T},
                {"commit": "d1", "path": "N.M.E/Bun.png", "title": "BAGEL", "ts": T + 9 * DAY}]
    assert dc.first_changes(last, data, "nme", versions, stop, T + 5 * DAY) == {"same_title": True}
    assert "name" in dc.first_changes(last, data, "nme", versions, stop, T + 9 * DAY)     # retitled before git_first


def test_an_item_cards_kinds_are_never_asked():
    """Types read as an item card's kinds alone get no review item (not collected); mixed reads lose the kinds. Sure
    types lose them too (owner 14:30: Hiveblood's 'equip'), compared as no types when nothing else was read."""
    inp = inputs([], old_review=[{"id": "o-1", "kind": "types unsure", "card": "Bolt", "row": "r1", "field": "types",
                                  "proposed": ["food"]}])
    inp.cards["bolt"]["type"] = "item"
    entries = [{"id": "D-1", "kind": "discord", "fields": {"types": dc.unsure("icon", "x", ["food", "heal"])}},
               {"id": "D-2", "kind": "discord", "fields": {"types": dc.unsure("icon", "x", ["nme", "support"])}},
               {"id": "O-r1", "kind": "old repo", "fields": {"types": {"confidence": "review", "review": "o-1"}}}]
    chain = dc.Chain(inp)
    chain.no_kinds("bolt", entries)
    chain.field_reviews("bolt", entries)
    assert [e["fields"]["types"]["confidence"] for e in entries] == ["not collected", "review", "not collected"]
    assert [i["proposed"] for i in chain.review.values()] == [["nme"]]
    sure = [{"id": f"D-{n}", "kind": k, "fields": {"types": dc.sure(v, "git look: psd")}}
            for n, k, v in ((3, "discord", ["equip"]), (4, "old repo", ["active", "field"]), (5, "discord", ["fire"]))]
    chain.no_kinds("bolt", sure)
    assert [e["fields"]["types"].get("value") for e in sure] == [None, ["field"], ["fire"]]
    assert sure[0]["fields"]["types"]["confidence"] == "not collected"
    assert dc.compare(sure[0]["fields"], sure[2]["fields"], "item")["types"] == {"add": ["fire"], "drop": []}
    assert dc.settled(sure[0]["fields"], {"types": {**sure[0]["fields"]["types"]}})
    inp.cards["bolt"]["type"] = "field"                  # an item then (Rock Bottom's 2023 'equip')
    then = [{"id": "D-6", "kind": "discord", "fields": {"types": dc.sure(["equip"], "eye")}},
            {"id": "D-7", "kind": "discord", "fields": {"types": dc.sure(["fire"], "eye")}}]
    chain.no_kinds("bolt", then)
    assert then[0]["fields"]["types"]["confidence"] == "not collected" and then[1]["fields"]["types"]["value"] == ["fire"]
    assert dc.old_kinds_only({"field": "types", "proposed": ["food"], "reads": {"psd": ["support"]}}, "aux item")
    assert not dc.old_kinds_only({"field": "types", "proposed": ["food"]}, "creature")


def test_answered_types_take_the_chain_s_tokens():
    """Types given or confirmed in an answer are the card data's tokens: 'defensive minion' -> defensive_minion; on a
    minion the 2023 shield ('protective') and sword ('equip') are its role (Miasma, Electro Probe; s3-gap1fix3), and
    compare() maps both sides, so no '+offensive_minion -offensive minion' change."""
    assert dc.type_tokens(["Defensive  Minion", "fragile_equip", "tornado"]) == ["defensive_minion", "fragile equip",
                                                                               "wind"]
    assert dc.type_tokens(["dark", "protective"], "minion") == ["dark", "defensive_minion"]
    assert dc.type_tokens(["equip", "machine"], "item") == ["equip", "machine"]          # an item's kind stays
    assert dc.type_tokens(["infection", "wind"]) == ["infected", "wind"]                # owner 11-17: 'infected'
    text = {"attachments/1.png": {"types": {"value": ["offensive minion"], "sure": False, "conf": 0.2,
                                            "method": "icon+known"}},
            "attachments/2.png": {"types": {"value": ["dark"], "sure": False, "conf": 0.2, "method": "icon+known"}},
            "attachments/3.png": {"types": {"value": ["offensive_minion"], "sure": False, "conf": 0.2,
                                            "method": "icon+known"}}}
    got = inputs([picture(1, T, card="lorn"), picture(2, T + DAY, whole=FAR, card="lorn"),
                  picture(3, T + 2 * DAY, whole={"p": "0f" * 8, "d": "0f" * 8, "q": "0f" * 32}, card="lorn")], text)
    yes, given, again = (dc.review_id("field", f"lorn|D-P-00000000000{n}|types") for n in (1, 2, 3))
    got.rulings = {yes: {"choice": "yes", "value": ["offensive minion"], "source": "triage (owner-delegated)"},
                   given: {"choice": "value", "value": ["dark", "protective"], "source": "triage guess"},
                   again: {"choice": "yes", "value": ["offensive minion"]}}    # proposed in the data's form since
    chain = dc.Chain(got)
    found, split = chain.discord_rows()
    t = chain.card("lorn", found["lorn"], split.get("lorn", []))
    types = [e["fields"]["types"] for e in t["entries"] if e["kind"] == "discord"]
    assert [(f["value"], f["method"], f["ruled"]) for f in types] == [(["offensive_minion"], "triage", yes),
                                                                   (["dark", "defensive_minion"], "triage", given),
                                                                   (["offensive_minion"], "owner", again)]
    assert t["entries"][1]["changes"]["types"] == {"add": ["dark", "defensive_minion"], "drop": ["offensive_minion"]}
    spaced, data = {"types": dc.sure(["offensive minion"], "triage")}, {"types": dc.sure(["offensive_minion"], "psd")}
    assert "types" not in dc.compare(spaced, data, "minion") and dc.settled(spaced, data)
    sword = {"types": dc.sure(["equip", "machine"], "triage")}
    assert "types" not in dc.compare(sword, {"types": dc.sure(["machine", "offensive_minion"], "psd")}, "minion")
    assert dc.compare(sword, {"types": dc.sure(["machine"], "psd")}, "item") == {}       # an item's kind: no change


def test_two_creatures_told_by_eye_publish_each_half():
    """A two-creature card's bar truth with both halves (stats, stats2) is sure, never summed; a split card's half
    takes its own half's rows (owner Q3)."""
    from commands import discord_chain as dc
    top, bottom = {"hp": 5, "def": 4, "atk": 6, "spd": 5}, {"hp": 4, "def": 2, "atk": 4, "spd": 5}
    got = dc.picture_stats({"two": True, "truth": {"stats": top, "stats2": bottom, "from": "eye"}}, "creature")
    assert got["confidence"] == "sure" and got["value"] == {"creatures": [top, bottom]}
    half = dc.picture_stats({"two": True, "truth": {"stats": top, "from": "eye"}}, "creature")
    assert half["confidence"] == "review"
    assert dc.read_key({"file": "a.png", "cell": 3}) == "a.png#c3" and dc.read_key({"file": "a.png"}) == "a.png"


def test_weak_pictures_set_no_created_and_a_first_psd_dates_the_card():
    weak, seen = picture(1, T + 5 * DAY, no_created="weak"), picture(2, T + 40 * DAY, whole=FAR)
    files = {"Bolt": {"date": dc.day(T + 40 * DAY), "from": "discord"}}
    got = inputs([weak, seen], created=files)
    chain = dc.Chain(got)
    found, split = chain.discord_rows()
    t = chain.card("bolt", found["bolt"], [])
    assert [e["pictures"][0]["id"] for e in t["entries"] if e["kind"] == "discord"] == [weak["id"], seen["id"]]
    assert t["entries"][0]["no_created"] == "weak" and "no_created" not in t["entries"][1]
    made = chain.created("bolt", t, [weak, seen])                 # the owner's weak link enters, dates nothing
    assert (made["status"], made["ours"]["picture"]) == ("same", seen["id"])
    got.psd_first = {"Bolt": {"ts": T + 2 * DAY, "commit": "p1"}}  # the old repo's PSD before any PNG of it
    made = chain.created("bolt", t, [weak, seen])
    assert (made["status"], made["ours"]["source"], made["ours"]["day"]) == ("earlier", "old repo (PSD)",
                                                                             dc.day(T + 2 * DAY))
    got.psd_first["Bolt"]["repo"] = "first"                       # the first repo's PSD: its own label
    made = chain.created("bolt", t, [weak, seen])
    assert (made["ours"]["source"], made["ours"]["psd"]) == ("first repo", True)
    got.psd_first = {}                                            # its first record a PSD: the stop's label too
    got.cards["bolt"].update(fs=T + DAY, fs_psd={"commit": "p0"}, fs_from="first git")
    assert chain.created("bolt", t, [weak, seen])["ours"]["source"] == "first repo"
    got.cards["bolt"]["fs_from"] = "old git"
    assert chain.created("bolt", t, [weak, seen])["ours"]["source"] == "old repo (PSD)"


def test_a_settled_look_folds_only_near_the_picture_before(tmp_path):
    """A Discord look folds (minor) only when every field is sure and unchanged AND its picture is near the one before
    (owner 11-09: under png_changes.VISIBLE of the card): a smaller JPEG copy cut a few px off or a small icon edit
    folds; new art with the same fields is listed, as is another shape; two PNGs of one size take png_changes' own
    rule (a 4-level shift of every pixel shows); one git look needs no picture."""
    import numpy as np
    from PIL import Image
    (tmp_path / "attachments").mkdir()
    rng = np.random.RandomState(3)
    base = rng.uniform(0, 255, (35, 25, 3)).repeat(16, 0).repeat(16, 1)               # 400 x 560 blocks
    art, icon = base.copy(), base.copy()
    art[40:260, 40:360] = rng.uniform(0, 255, (11, 16, 3)).repeat(20, 0).repeat(20, 1)  # new art
    icon[10:50, 10:50] = 0                                                              # an icon added
    save = lambda n, a, **kw: Image.fromarray(a.clip(0, 255).astype(np.uint8)).save(tmp_path / f"attachments/{n}", **kw)
    save("1.png", base)
    Image.fromarray(base[3:-4, 4:-3].astype(np.uint8)).resize((300, 418)).save(tmp_path / "attachments/2.jpg", quality=70)
    save("3.png", art)
    save("4.png", icon)
    save("5.png", base[:400])                                                           # square: another shape
    save("6.png", base + 4)                                                             # every pixel 4 levels up
    same = {"stars": dc.sure(3, "eye"), "stats": stats(5, 2, 3, 4)}
    entry = lambda n, file, look=None: {"id": f"D-{n}", "kind": "discord", "look": look or {"draft": True},
                                        "fields": dict(same), "pictures": [{"id": f"P-{n}", "file": f"attachments/{file}"}]}
    chain = dc.Chain(inputs([], root=tmp_path))
    entries = [entry(1, "1.png"), entry(2, "2.jpg"), entry(3, "3.png"), entry(4, "1.png"), entry(5, "4.png"),
               entry(6, "5.png")]
    chain.diff(entries, "creature")
    assert [e["minor"] for e in entries] == [False, True, False, False, True, False]
    assert entries[2]["share"] >= 0.05 > entries[1]["share"] and "share" not in entries[5]
    assert chain.tally["discord: settled, kept apart (a new picture)"] == 2
    assert chain.tally["discord: settled, kept apart (pictures not compared)"] == 1
    minor, share = dc.fold(str(tmp_path), entry(1, "1.png"), entry(6, "6.png"))
    picture = lambda n: dc.fold_picture(str(tmp_path / f"attachments/{n}.png"), None)
    assert not minor and share > 0.9 and dc.changed_share(picture(1), picture(6)) < 0.05
    git = {"git": {"row": "c1:Creatures/Bolt.png"}}                                  # one git look: no picture read
    assert dc.fold(None, entry(7, "x.png", git), entry(8, "y.png", git)) == (True, 0.0)
    assert dc.fold(None, entry(7, "x.png"), entry(8, "y.png")) == (False, None)
