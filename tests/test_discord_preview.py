"""discord_preview build on a made-up archive: only sure fields are published (value, method, confidence; not one of
the wrong kind or one the eye dropped; a title snapped to an old casing as the picture prints it), no message, poster,
note or review keys; one webp per version at most WEBP_W wide (a sheet's cell cut out, an old-repo blob read from its git
dir); the Morgue's candidates and MDW cards; Created proposals only when earlier or new. Stop at git: a card moved and
redrawn in one commit starts at its first file."""
import json
import subprocess
from os import environ

from PIL import Image

import commands.discord_match as dm
import commands.discord_preview as dp

BAD_KEYS = {"message", "jump_url", "poster", "author_id", "content", "note", "why", "review", "conf", "file", "box"}


def keys(obj):
    if isinstance(obj, dict):
        return set(obj) | {k for v in obj.values() for k in keys(v)}
    return {k for v in obj for k in keys(v)} if isinstance(obj, list) else set()


def test_build(tmp_path):
    root = tmp_path / "archive"
    (root / "attachments").mkdir(parents=True)
    Image.new("RGB", (1000, 2000), (200, 30, 30)).save(root / "attachments" / "a.png")
    sheet = Image.new("RGB", (800, 600), (200, 30, 30))
    sheet.paste((10, 200, 10), (400, 0, 800, 600))              # a 1x2 sheet: cell 2 green
    sheet.paste((20, 20, 220), (400, 0, 420, 20))                # a corner unlike the rest: nothing trimmed
    sheet.save(root / "attachments" / "s.png")
    git = tmp_path / "old.git"
    subprocess.run(["git", "init", "-q", "--bare", str(git)], check=True)
    png = tmp_path / "old.png"
    Image.new("RGBA", (300, 600), (0, 0, 255, 255)).save(png)
    blob = subprocess.run(["git", "--git-dir", str(git), "hash-object", "-w", str(png)], capture_output=True, text=True,
                          check=True).stdout.strip()
    sure = lambda v, how: {"value": v, "method": how, "confidence": "sure", "conf": 0.99}
    pic = lambda pid, f, cell=None, box=None: {"id": pid, "date": "2021-01-01T00:00:00+00:00", "channel": "new-cards",
                                               "file": f"attachments/{f}", "cell": cell, "box": box, "verdict": "draft",
                                               "message": "1", "jump_url": "https://x"}
    t = {"card": "bolt", "key": "Bolt", "name": "Bolt", "kind": "creature", "stop": {"day": "2023-05-01", "source": "old repo"},
         "git_first": None, "created": {}, "entries": [
             {"id": "D-P-a", "kind": "discord", "date": "2021-01-01T00:00:00+00:00", "day": "2021-01-01", "channel": "new-cards",
              "look": {"draft": True}, "era": "D2", "pictures": [pic("P-a", "a.png")], "minor": False,
              "fields": {"name": sure("BOLT", "ocr+name"), "ability": {"method": "ocr", "confidence": "review", "why": "x", "review": "chain-1"}},
              "changes": {"ability_change": "first"}},
             {"id": "D-P-s-c2", "kind": "discord", "date": "2021-02-01T00:00:00+00:00", "day": "2021-02-01", "channel": "ideas",
              "look": {"git": {"commit": "c"}}, "pictures": [pic("P-s-c2", "s.png", 2, [0.5, 0.0, 1.0, 1.0])], "minor": False,
              "fields": {"stars": sure(3, "eye")}, "changes": {"ability_change": "unknown"}},
             {"id": "F-1", "kind": "text", "date": "2022-01-01T00:00:00+00:00", "day": "2022-01-01", "channel": "update-05",
              "source": "changelog", "change": {"field": "atk", "from": 5, "to": 4, "stated": {"from": 5}}, "confidence": "sure",
              "confirmed_by": "O-1"},
             {"id": "O-1", "kind": "old repo", "date": "2023-05-01T00:00:00+00:00", "day": "2023-05-01", "badge": "old repo",
              "author": "Maker", "subject": "Bolt", "commit": "c" * 40, "path": "Creatures/Bolt.png", "blob": blob,
              "thumb": "x", "minor": True, "share": 0.01, "fields": {"stats": sure({"hp": 3, "def": 2, "atk": 4, "spd": 1}, "picture")},
              "changes": {}}]}
    (root / "derived" / "chain" / "timelines").mkdir(parents=True)
    (root / "derived" / "chain" / "timelines" / "bolt.json").write_text(json.dumps(t))
    created = {"cards": {"Bolt": {"status": "earlier", "proposed": True, "date": "2021-01-01", "sort": "2021-01-01",
                                  "ours": {"source": "discord", "picture": "P-a"}, "file": {"date": "2021-05-05"}},
                         "Pike": {"status": "later", "proposed": False, "date": "2020-01-01", "ours": {"source": "discord"}}}}
    (root / "derived" / "chain" / "created.json").write_text(json.dumps(created))
    (root / "derived" / "match").mkdir(parents=True)
    (root / "derived" / "match" / "verdicts.jsonl").write_text(json.dumps({"id": "P-m", "file": "attachments/a.png"}) + "\n")
    (root / "derived" / "morgue").mkdir(parents=True)
    morgue = {"candidates": [{"id": "LC-1", "name": "Joke", "slug": "joke", "kind": "joke", "status": "candidate",
                              "first": {"date": "2020-01-01T00:00:00+00:00", "channel": "meme-cards", "source": "discord"},
                              "why": ["x"], "eye": {"note": "x"},
                              "pictures": [{"id": "P-m", "date": "2020-01-01T00:00:00+00:00", "channel": "meme-cards", "source": "discord"}]}],
              "not_candidates": [{"id": "LC-2", "name": "Old One", "card": "Old One", "status": "morgue", "first": {},
                                  "pictures": [{"id": "P-m", "date": None, "channel": None, "source": "discord", "printed": True}]}]}
    (root / "derived" / "morgue" / "candidates.json").write_text(json.dumps(morgue))
    (root / "derived" / "preview").mkdir(parents=True)
    (root / "derived" / "preview" / "eye.json").write_text(json.dumps({"drop": {"bolt|O-1|stats": "the eye: another value"},
                                                                    "covered": {"bolt|D-P-a": "under another card"}}))
    (root / "derived" / "readers" / "text").mkdir(parents=True)
    (root / "derived" / "readers" / "text" / "reads.jsonl").write_text(json.dumps(
        {"src": "discord", "file": "attachments/a.png", "title": {"ocr": "* Bolt", "value": "BOLT"}}) + "\n")
    t["entries"][1]["fields"]["name"] = sure(["Bolt", "Bolt2"], "ocr+name")   # a value of the wrong kind
    (root / "derived" / "chain" / "timelines" / "bolt.json").write_text(json.dumps(t))
    index = dp.build(str(root), str(git), workers=1, cards_repo=None)
    out = root / "derived" / "preview" / "history"
    got = json.loads((out / "bolt" / "timeline.json").read_text())
    assert [e["kind"] for e in got["entries"]] == ["discord", "discord", "text", "old repo"]
    first, cell, fact, old = got["entries"]
    assert first["fields"] == {"name": {"value": "Bolt", "method": "ocr+name", "confidence": "sure"}}   # the review field left out
    assert first["covered"] and "covered" not in cell and fact["confirmed_by"] == "O-1"
    assert dp.printed_name("DooPliss", ["f"], {"f": "O DooPliss"}, "Doopliss") == "DooPliss"     # printed so: kept
    assert cell["git_look"] and fact["change"] == {"field": "atk", "from": 5, "to": 4} and old["minor"] and old["author"] == "Maker"
    assert "name" not in cell["fields"] and "stats" not in old["fields"]          # the wrong kind; the eye's drop
    assert not keys(got) & BAD_KEYS and not keys(index) & BAD_KEYS
    sizes = {e["image"]: Image.open(out / "bolt" / e["image"]).size for e in got["entries"] if e.get("image")}
    assert sizes["P-a.webp"] == (dp.WEBP_W, 2 * dp.WEBP_W) and sizes[f"O-{blob[:12]}.webp"] == (300, 600)
    assert sizes["P-s-c2.webp"] == (400, 600)                                       # the sheet's cell 2 alone
    assert Image.open(out / "bolt" / "P-s-c2.webp").convert("RGB").getpixel((200, 500))[1] > 150   # green
    assert index["created"] == {"Bolt": {"date": "2021-01-01", "sort": "2021-01-01", "source": "discord", "status": "earlier",
                                         "was": "2021-05-05"}}
    assert [m["name"] for m in index["morgue"]] == ["Joke"] and [m["card"] for m in index["mdw"]] == ["Old One"]
    mdw = json.loads((out / index["mdw"][0]["dir"] / "timeline.json").read_text())
    assert mdw["entries"][0]["date"] is None and mdw["entries"][0]["printed"]       # a printed card's photo: no day
    # a rebuild keeps the webps, and drops one no entry names any more
    (out / "bolt" / "stale.webp").write_bytes(b"x")
    dp.build(str(root), str(git), workers=1, cards_repo=None)
    assert not (out / "bolt" / "stale.webp").exists() and (out / "bolt" / "P-a.webp").exists()


def test_stop_at_git(tmp_path):
    """A card moved and redrawn in one commit (4 Stars -> 3 Stars: the matcher's stop is that commit) starts at its first
    file, as the website follows it: the Discord post after that is left out, the stop moves back, and git_first is
    diffed again, dated by its PSD's commit and approx (an older PSD version came first). An MDW Morgue card's pictures
    end at its first record: the old repo's (oldgit map.json), else its PNG's in the card repo."""
    repo = tmp_path / "cards"
    who = {"GIT_AUTHOR_NAME": "M", "GIT_AUTHOR_EMAIL": "m@x", "GIT_COMMITTER_NAME": "M", "GIT_COMMITTER_EMAIL": "m@x"}
    git = lambda *a, at=0: subprocess.run(["git", "-C", str(repo), *a], capture_output=True, text=True, check=True, env={
        **environ, **who, "GIT_AUTHOR_DATE": f"@{at} +0000", "GIT_COMMITTER_DATE": f"@{at} +0000"}).stdout
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    for n, (folder, at) in enumerate((("4 Stars", 1610000000), ("3 Stars", 1740000000)), 1):
        for f in repo.glob("Bolt/*/*"):
            f.unlink()
        (repo / "Bolt" / folder).mkdir(parents=True, exist_ok=True)
        (repo / "Bolt" / folder / "Bolt.png").write_bytes(b"png %d" % n)
        (repo / "Bolt" / folder / "Bolt.psd").write_bytes(b"psd %d" % n)
        git("add", "-A")
        git("commit", "-q", "-m", folder, at=at)
    git("update-ref", "refs/remotes/origin/main", "HEAD")
    commits = dp.repo_log(str(repo))
    pngs = dp.lineage(commits, {"bolt": "Bolt/3 Stars/Bolt.png"})["bolt"]
    psds = dp.lineage(commits, {"bolt": "Bolt/3 Stars/Bolt.psd"}, ".psd")["bolt"]
    assert [(r[0], r[2]) for r in pngs] == [(1740000000, "Bolt/3 Stars/Bolt.png"), (1610000000, "Bolt/4 Stars/Bolt.png")]
    line = lambda atk: {"value": {"hp": 3, "def": 2, "atk": atk, "spd": 1}, "method": "picture", "confidence": "sure"}
    t = {"stop": {"date": dm.iso(1740000000), "day": "2025-02-19", "source": "git"}, "entries": [
        {"id": "D-1", "kind": "discord", "date": "2020-06-01T00:00:00+00:00", "fields": {"stats": line(5)}},
        {"id": "D-2", "kind": "discord", "date": "2022-01-01T00:00:00+00:00", "fields": {"stats": line(4)}}],
        "git_first": {"date": "2025-03-01T00:00:00+00:00", "fields": {"stats": line(3)}, "changes": {}}}
    done = dp.stop_at_git(t, pngs, psds, {"blob_sha": psds[0][3]})
    assert done == {"stop": "2021-01-07", "was": "2025-02-19", "dropped": ["D-2"], "approx": True}
    assert [e["id"] for e in t["entries"]] == ["D-1"] and t["stop"]["path"] == "Bolt/4 Stars/Bolt.png"
    assert t["git_first"]["date"] == dm.iso(1740000000) and t["git_first"]["changes"]["stats"][0]["atk"] == 5
    (repo / "MDW" / "The Morgue").mkdir(parents=True)
    (repo / "MDW" / "The Morgue" / "Ech.png").write_bytes(b"ech")
    git("add", "-A")
    git("commit", "-q", "-m", "Ech", at=1750000000)
    git("update-ref", "refs/remotes/origin/main", "HEAD")
    root = tmp_path / "archive"
    (root / "derived" / "oldgit").mkdir(parents=True)
    (root / "derived" / "oldgit" / "map.json").write_text(json.dumps({"cards": {
        "White Palace": {"key": "White Palace", "status": "archived", "first": {"date": "2023-10-24T17:43:59-04:00"}},
        "Bolt": {"key": "Bolt", "status": "live", "first": {"date": "2020-01-01T00:00:00+00:00"}}}}))
    firsts = dp.mdw_firsts(str(root), str(repo), dp.repo_log(str(repo)))
    assert firsts == {"whitepalace": dm.epoch("2023-10-24T21:43:59+00:00"), "ech": 1750000000}
    entry = {"pictures": [{"id": "P-1", "source": "discord", "date": "2023-10-24T18:34:59+00:00"},
                          {"id": "P-2", "source": "discord", "date": "2026-06-19T04:06:58+00:00"},
                          {"id": "P-3", "source": "discord", "date": None, "printed": True}]}
    assert dp.mdw_cut(entry, firsts["whitepalace"]) == ["P-2"] and [p["id"] for p in entry["pictures"]] == ["P-1", "P-3"]
