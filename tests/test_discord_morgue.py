"""discord_morgue on a made-up archive: candidates (jokes, played, the old repo's), ideas, cards already in MDW or
renamed, the check by eye (names, splits, merges, weak links), twins of rows a live card or the unsure review lists
(taken by eye, ruled another picture, else review), the kin pass (made by eye, ruled not kin, else open), MDW cards'
pictures (kin, taken, clashes), M'uru's 'Before 2020' (owner Q6: its photo never read), created hints' meme flag,
unique slugs, thumbs, no message text."""
import json
from pathlib import Path

from PIL import Image

import commands.discord_morgue as dmo
import commands.discord_verdicts as dv

PORT, IDEAS, MEME = ("735328279619436666", "port"), ("664256411907391509", "ideas-discussion"), \
    ("835450716947873823", "meme-cards")
MURU = "LC-1c11be0504"                          # discord_chain.PHOTO_MORGUE


def sha(n: int) -> str:
    return f"{n:02x}" * 32


def post(n: int, channel: tuple, day: int, meme: bool = False) -> dict:
    return {"message": str(1000 + n), "channel_id": channel[0], "channel": channel[1], "thread": None,
            "date": f"2020-08-{day:02d}T00:00:00+00:00", "bot": False, "morgue": meme, "embed": False}


def look(n: int) -> dict:
    """A hash set far from every other n's (n < 16; n 16..31: a nibble and its complement, 32 bits from all n < 16)."""
    digits = f"{n:x}" * 2 if n < 16 else f"{n % 16:x}{15 - n % 16:x}"
    return {"p": digits * 8, "d": digits * 8, "q": digits * 32}


def group(n: int, title: str, kinds: list, pictures: list, **more) -> dict:
    played, joke = "played" in kinds, "joke" in kinds
    return {"id": dv.group_id(sha(n)), "title": title, "kinds": kinds, "pictures": pictures,
            "morgue_candidate": joke or played, "ideas_never_ported": not (joke or played), "old_name_of": None,
            "morgue_card": None, "split_of": None, "card_evidence": "stat labels", "changelog_days": [],
            "old_lost": None, "nearest": {"look": [{"card": "fang", "name": "Fang", "bits": 21.0}], "name": []},
            **more}


def archive(root: Path) -> None:
    for sub in ("attachments", "derived/match", "derived/oldgit/thumbs", "derived/matcher/match", "derived/catalogue",
                "derived/morgue"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    rows = []
    places = {1: (MEME, True), 2: (PORT, False), 3: (IDEAS, False), 4: (IDEAS, False), 5: (IDEAS, False),
              6: (MEME, True), 7: (IDEAS, False), 8: (IDEAS, False), 9: (PORT, False), 10: (MEME, True),
              11: (IDEAS, False), 12: (MEME, True), 13: (IDEAS, False), 26: (PORT, False)}
    listed = {9: ("version_unsure", {"card": "zero-two"}, 6), 11: ("unsure", {"maybe": ["fang"]}, 10),
              13: ("draft", {"card": "fang"}, 12)}      # n: (verdict, card, the lost picture it is the same as)
    kin = {14: ("none", PORT, False, 14), 15: ("none", MEME, True, 15), 16: ("unsure", IDEAS, False, 0),
           17: ("unsure", IDEAS, False, 9), 18: ("none", IDEAS, False, 11), 19: ("none", PORT, False, 13),
           20: ("none", PORT, False, 13), 21: ("unsure", IDEAS, False, 16), 22: ("none", IDEAS, False, 3),
           23: ("unsure", IDEAS, False, 4), 25: ("unsure", IDEAS, False, 18)}    # n: (verdict, channel, joke, look)
    images = {f"{1000 + n}:{n}": {"message": str(1000 + n), "sha256": sha(n), "title": [], "lines": lines, **more}
              for n, lines, more in ((14, ["Book One", "Health", "Defense", "Attack", "Speed"],
                                      {"title_block": "Book One"}), (16, ["Old Brick"], {"title_block": "Old Brick"}),
                                     (17, ["Zero Two"], {"title_block": "Zero Two"}),
                                     (19, ["Health", "Speed"], {}), (20, ["a chat"], {}),
                                     (25, ["Star Nova"], {"title_block": "Star Nova"}))}
    for n, (verdict, channel, joke, same) in kin.items():
        places[n] = (channel, joke)
        listed[n] = (verdict, {"maybe": ["fang"]} if verdict == "unsure" else {}, same)
    for n, (channel, meme) in places.items():
        Image.new("RGB", (400, 800), (n * 12, 40, 90)).save(root / f"attachments/{sha(n)}.png")
        verdict, card, same = listed.get(n, ("morgue" if meme else "lost", {}, n))
        rows.append({"id": dv.picture_id(sha(n)), "sha256": sha(n), "file": f"attachments/{sha(n)}.png",
                     "verdict": verdict, **card, "w": 400, "h": 800, "posts": [post(n, channel, n, meme)],
                     "whole": look(19 if n == 22 else same), "art": look(same),     # 22: the art alone
                     "title_read": "Fang Gets Funnier" if n == 7 else "", "joke": meme})
    (root / "derived/match/verdicts.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    p = dv.picture_id
    groups = [group(1, "Fang But Funny", ["idea", "joke"], [p(sha(1)), p(sha(7))]),
              group(2, "Old Brick", ["lost", "played"], [p(sha(2))], changelog_days=["2020-08-09"],
                    old_lost="OL-aaaaaaaaaa"),
              group(3, "Never Made", ["idea", "lost"], [p(sha(3))]),
              group(4, "The Developper", ["idea", "lost"], [p(sha(4))]),
              group(5, "Probe", ["lost", "played"], [p(sha(5))]),
              group(6, "Zero Two", ["joke", "played"], [p(sha(6))]),
              group(8, "Never Mad", ["idea", "lost"], [p(sha(8))]),
              group(10, "Fang Joke", ["joke", "lost"], [p(sha(10))]),
              group(12, "Fang Prank", ["joke", "lost"], [p(sha(12))]),
              group(26, "M'uru", ["lost", "played"], [p(sha(26))], id=MURU)]
    (root / "derived/match/lost_groups.json").write_text(json.dumps({"groups": groups}))
    old = [{"id": f"OL-{c * 10}", "old_id": name, "paths": [f"Items/{name}.png"],
            "first": {"commit": "c1", "date": "2023-09-11T21:12:04-04:00"}, "last": None,
            "deleted": {"commit": "c2", "date": "2023-10-13T05:26:45-04:00"}, "versions": {"png": 1},
            "thumb": f"thumbs/{c}.webp", "nearest": [{"key": "Fang", "d": 15.0}], "authors": ["Someone"]}
           for c, name in (("a", "Old Brick"), ("b", "The Battery"))]
    for c in "ab":
        Image.new("RGB", (320, 480), (200, 10, 10)).save(root / f"derived/oldgit/thumbs/{c}.webp")
    (root / "derived/oldgit/lost.json").write_text(json.dumps({"cards": old}))
    cards = {"fang": {"name": "Fang"}, "zero-two": {"name": "Zero Two"}, "electro-probe": {"name": "Electro Probe"}}
    (root / "derived/matcher/match/pairs.json").write_text(json.dumps({"cards": cards, "versions": [],
                                                                        "images": images}))
    (root / "derived/catalogue/channels.json").write_text(json.dumps({"channels": [{"id": PORT[0], "kind": "port"}]}))
    eye = {dv.group_id(sha(1)): {"status": "candidate", "kind": "joke", "name": "Fang But Funny", "split": [
        {"name": "Fang But Funny", "pictures": [p(sha(1))], "status": "idea"},
        {"name": "Fang Gets Funnier", "pictures": [p(sha(7))]}]},
        dv.group_id(sha(5)): {"weak": "no shared picture"},
        dv.group_id(sha(6)): {"status": "candidate", "kind": "joke", "name": "Zero Two", "nearest": "zero-two",
                              "add": [p(sha(9)), p(sha(18))]},
        dv.group_id(sha(8)): {"status": "idea", "merge": dv.group_id(sha(3))},
        dv.group_id(sha(12)): {"status": "candidate", "kind": "joke", "name": "Fang Prank", "not_twin": [p(sha(13))]},
        dv.group_id(sha(4)): {"add": [p(sha(21))]},       # MDW Developper's PNG, unsure for fang
        MURU: {"status": "candidate", "kind": "played", "name": "M'uru", "add": [p(sha(24))]},   # 24: its photo (no row)
        p(sha(14)): {"status": "candidate", "kind": "played", "name": "Book One"},     # a lost card found by kin
        "not_kin": {p(sha(15)): "a meme"}}
    (root / "derived/morgue/eye.json").write_text(json.dumps(eye))


def test_candidates_ideas_and_the_cards_that_are_not(tmp_path):
    root = tmp_path / "archive"
    archive(root)
    known = {"morgue": {"Developper": "MDW/The Morgue/Developper.psd"},
             "unfinished": {"Star Nova": "MDW/Unfinished/Star_Nova.psd"},
             "renamed": {"Probe": "Electro Probe"},
             "looks": [{"place": "morgue", "name": "Developper", "path": "MDW/The Morgue/Developper.png",
                        "whole": look(16), "art": look(16)},
                       {"place": "unfinished", "name": "Star Nova", "path": "MDW/Unfinished/Star_Nova.png",
                        "whole": look(3), "art": look(3)}]}
    counts = dmo._build(dmo.Paths(str(root)), known, log=lambda *a: None)
    p9 = dv.picture_id(sha(9))
    data = json.loads((root / "derived/morgue/candidates.json").read_text())
    ideas = json.loads((root / "derived/morgue/ideas.json").read_text())["ideas"]
    found = {e["id"]: e for e in data["candidates"]}
    split = dv.group_id(sha(7))
    book = dv.group_id(sha(14))
    assert set(found) == {split, dv.group_id(sha(2)), dv.group_id(sha(6)), dv.group_id(sha(12)), "OL-bbbbbbbbbb", book,
                          MURU}
    assert found[book]["made"] and found[book]["kind"] == "played" and found[book]["title_read"] == "Book One" and \
        found[book]["played"] == [{"channel": "port", "date": "2020-08-14"}] and found[book]["card_evidence"] == \
        "stat labels"
    assert counts["kin"] == 6 and counts["kin_open"] == [dv.picture_id(sha(n)) for n in (16, 19, 22, 23, 25)]
    # 15 ruled not kin; 17 titled as a live card; 20 a played channel's with no card evidence; 22 MDW Star Nova's
    # art (another look); 23 the same picture as the Developper entry's; 25 titled like Star Nova
    assert counts["mdw_clash"] == [f"{dv.picture_id(sha(3))} in {dv.group_id(sha(3))}: MDW/Unfinished/Star_Nova.png "
                                   "(0.0 bits)"]          # an idea holding an MDW card's PNG
    assert found[split]["name"] == found[split]["title_read"] == "Fang Gets Funnier" and found[split]["kind"] == \
        "joke" and found[split]["group"] == dv.group_id(sha(1))         # a split part: its title, its parent group
    brick = found[dv.group_id(sha(2))]
    assert brick["kind"] == "played" and brick["old_repo"]["id"] == "OL-aaaaaaaaaa" and brick["first"]["channel"] \
        == "port" and any("port picture" in w for w in brick["why"]) and len(brick["pictures"]) == 2
    assert found["OL-bbbbbbbbbb"]["first"] == {"date": "2023-09-12T01:12:04+00:00", "channel": "old repo",
                                                "thread": None, "source": "old repo"}
    zero = found[dv.group_id(sha(6))]
    assert zero["slug"] == "zero-two-morgue" and zero["nearest"]["card"] == "zero-two"   # never a live card's id
    assert zero["played"] == [{"channel": "port", "date": "2020-08-09"}]
    assert zero["first"]["date"] == "2020-08-06T00:00:00+00:00" and zero["last"] == "2020-08-18T00:00:00+00:00"
    muru = found[MURU]       # owner Q6 (11-14 22:20): Created 'Before 2020', its photo in no Morgue evidence
    assert muru["first"] == {"date": None, "channel": None, "thread": None, "source": "physical card photo",
                             "day": "Before 2020", "dated": {"date": "2020-08-26T00:00:00+00:00", "channel": "port",
                                                             "thread": None, "source": "discord"}}
    assert [x["id"] for x in muru["pictures"]] == [dv.picture_id(sha(26))] and muru["slug"] == "m-uru"
    assert any("Before 2020" in w for w in muru["why"]) and data["candidates"][0]["id"] == MURU   # sorts first
    assert counts["photo_missing"] == []                    # a regrouped M'uru would be listed (its Created lapsed)
    assert all(e["first"].get("day") is None for e in data["candidates"][1:] + data["not_candidates"] + ideas)
    assert data["taken"] == [{"picture": p9, "entry": dv.group_id(sha(6)), "verdict": "version_unsure",
                              "card": "zero-two"},             # its twin taken by eye
                             {"picture": dv.picture_id(sha(21)), "entry": dv.group_id(sha(4)), "verdict": "unsure",
                              "card": "fang"}] and counts["taken"] == 2     # an MDW card's picture: its entry's
    assert counts["twins_open"] == [dv.group_id(sha(10))]                       # a twin not ruled on: review
    slugs = [e["slug"] for e in data["candidates"]]
    assert len(set(slugs)) == len(slugs) and not set(slugs) & {"fang", "zero-two", "electro-probe", "developper"}
    others = {e["id"]: e for e in data["not_candidates"]}
    assert others[dv.group_id(sha(4))]["status"] == "morgue" and others[dv.group_id(sha(5))]["card"] == \
        "electro-probe" and others[dv.group_id(sha(8))]["status"] == "merged"
    twin = others[dv.group_id(sha(10))]
    assert twin["status"] == "twin" and twin["card"] == "fang" and twin["twins"] == [dv.picture_id(sha(11))]
    assert data["created_hints"] == [{"entry": dv.group_id(sha(5)), "name": "Probe", "card": "electro-probe",
                                      "date": "2020-08-05T00:00:00+00:00", "channel": "ideas-discussion",
                                      "weak": True, "meme": False},
                                     {"entry": dv.group_id(sha(10)), "name": "Fang Joke", "card": "fang",
                                      "date": "2020-08-10T00:00:00+00:00", "channel": "meme-cards", "weak": False,
                                      "meme": True}]
    assert [i["id"] for i in ideas] == [dv.group_id(sha(1)), dv.group_id(sha(3))] and \
        len(ideas[1]["pictures"]) == 2              # a split part's own status; a merge
    assert counts["candidates"] == 7 and counts["made"] == 1 and dv.group_id(sha(3)) in counts["unchecked"]
    for e in data["candidates"] + ideas:
        for pic in e["pictures"]:
            with Image.open(root / "derived/morgue" / pic["thumb"]) as im:
                assert im.width <= dmo.THUMB_W
    text = json.dumps(data) + json.dumps(ideas)
    assert all(k not in text for k in ('"content"', '"author', "username", "Someone", "jump_url"))


def test_the_owner_s_entries_over_the_eye_s(tmp_path):
    root = tmp_path / "archive"
    archive(root)
    eye = json.loads((root / "derived/morgue/eye.json").read_text())
    book = eye.pop(dv.picture_id(sha(14)))
    (root / "derived/morgue/eye.json").write_text(json.dumps(eye))
    (root / "derived/review").mkdir(parents=True, exist_ok=True)
    (root / "derived/review/owner.json").write_text(json.dumps({"morgue": {dv.picture_id(sha(14)): {
        **book, "name": "Book Two", "answer": "a1"}}}))
    known = {"morgue": {}, "unfinished": {}, "renamed": {}, "looks": []}
    dmo._build(dmo.Paths(str(root)), known, log=lambda *a: None)
    found = {e["id"]: e for e in json.loads((root / "derived/morgue/candidates.json").read_text())["candidates"]}
    entry = found[dv.group_id(sha(14))]
    assert (entry["name"], entry["owner"], entry["kind"]) == ("Book Two", "a1", "played")
