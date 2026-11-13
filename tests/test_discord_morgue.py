"""discord_morgue on a made-up archive: candidates (jokes, played, the old repo's), ideas, cards already in MDW or
renamed, the check by eye (names, splits, merges), unique slugs, thumbs, no message text."""
import json
from pathlib import Path

from PIL import Image

import commands.discord_morgue as dmo
import commands.discord_verdicts as dv

PORT, IDEAS, MEME = ("735328279619436666", "port"), ("664256411907391509", "ideas-discussion"), \
    ("835450716947873823", "meme-cards")


def sha(n: int) -> str:
    return f"{n:02x}" * 32


def post(n: int, channel: tuple, day: int, meme: bool = False) -> dict:
    return {"message": str(1000 + n), "channel_id": channel[0], "channel": channel[1], "thread": None,
            "date": f"2020-08-{day:02d}T00:00:00+00:00", "bot": False, "morgue": meme, "embed": False,
            "paper": False}


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
              6: (MEME, True), 7: (IDEAS, False), 8: (IDEAS, False)}
    for n, (channel, meme) in places.items():
        Image.new("RGB", (400, 800), (n * 20, 40, 90)).save(root / f"attachments/{sha(n)}.png")
        rows.append({"id": dv.picture_id(sha(n)), "sha256": sha(n), "file": f"attachments/{sha(n)}.png",
                     "verdict": "morgue" if meme else "lost", "w": 400, "h": 800,
                     "posts": [post(n, channel, n, meme)]})
    (root / "derived/match/verdicts.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    p = dv.picture_id
    groups = [group(1, "Fang But Funny", ["idea", "joke"], [p(sha(1)), p(sha(7))]),
              group(2, "Old Brick", ["lost", "played"], [p(sha(2))], changelog_days=["2020-08-09"],
                    old_lost="OL-aaaaaaaaaa"),
              group(3, "Never Made", ["idea", "lost"], [p(sha(3))]),
              group(4, "The Developper", ["idea", "lost"], [p(sha(4))]),
              group(5, "Probe", ["lost", "played"], [p(sha(5))]),
              group(6, "Zero Two", ["joke", "played"], [p(sha(6))]),
              group(8, "Never Mad", ["idea", "lost"], [p(sha(8))])]
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
    (root / "derived/matcher/match/pairs.json").write_text(json.dumps({"cards": cards, "versions": []}))
    (root / "derived/catalogue/channels.json").write_text(json.dumps({"channels": [{"id": PORT[0], "kind": "port"}]}))
    eye = {dv.group_id(sha(1)): {"status": "candidate", "kind": "joke", "name": "Fang But Funny", "split": [
        {"name": "Fang But Funny", "pictures": [p(sha(1))]}, {"name": "Fang Gets Funnier", "pictures": [p(sha(7))]}]},
        dv.group_id(sha(6)): {"status": "candidate", "kind": "joke", "name": "Zero Two", "nearest": "zero-two"},
        dv.group_id(sha(8)): {"status": "idea", "merge": dv.group_id(sha(3))}}
    (root / "derived/morgue/eye.json").write_text(json.dumps(eye))


def test_candidates_ideas_and_the_cards_that_are_not(tmp_path):
    root = tmp_path / "archive"
    archive(root)
    known = {"morgue": {"Developper": "MDW/The Morgue/Developper.psd"}, "unfinished": {},
             "renamed": {"Probe": "Electro Probe"}}
    counts = dmo._build(dmo.Paths(str(root)), known, log=lambda *a: None)
    data = json.loads((root / "derived/morgue/candidates.json").read_text())
    ideas = json.loads((root / "derived/morgue/ideas.json").read_text())["ideas"]
    found = {e["id"]: e for e in data["candidates"]}
    split = dv.group_id(sha(7))
    assert set(found) == {dv.group_id(sha(1)), split, dv.group_id(sha(2)), dv.group_id(sha(6)), "OL-bbbbbbbbbb"}
    assert found[split]["name"] == "Fang Gets Funnier" and found[split]["kind"] == "joke"
    brick = found[dv.group_id(sha(2))]
    assert brick["kind"] == "played" and brick["old_repo"]["id"] == "OL-aaaaaaaaaa" and brick["first"]["channel"] \
        == "port" and any("port picture" in w for w in brick["why"]) and len(brick["pictures"]) == 2
    assert found["OL-bbbbbbbbbb"]["first"] == {"date": "2023-09-12T01:12:04+00:00", "channel": "old repo",
                                                "thread": None, "source": "old repo"}
    zero = found[dv.group_id(sha(6))]
    assert zero["slug"] == "zero-two-morgue" and zero["nearest"]["card"] == "zero-two"   # never a live card's id
    slugs = [e["slug"] for e in data["candidates"]]
    assert len(set(slugs)) == len(slugs) and not set(slugs) & {"fang", "zero-two", "electro-probe", "developper"}
    others = {e["id"]: e for e in data["not_candidates"]}
    assert others[dv.group_id(sha(4))]["status"] == "morgue" and others[dv.group_id(sha(5))]["card"] == \
        "electro-probe" and others[dv.group_id(sha(8))]["status"] == "merged"
    assert [i["id"] for i in ideas] == [dv.group_id(sha(3))] and len(ideas[0]["pictures"]) == 2   # merged in
    assert counts["candidates"] == 5 and dv.group_id(sha(2)) in counts["unchecked"]
    for e in data["candidates"] + ideas:
        for pic in e["pictures"]:
            with Image.open(root / "derived/morgue" / pic["thumb"]) as im:
                assert im.width <= dmo.THUMB_W
    text = json.dumps(data) + json.dumps(ideas)
    assert all(k not in text for k in ('"content"', '"author', "username", "Someone", "jump_url"))
