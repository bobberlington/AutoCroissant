"""discord_preview build on a made-up archive: only sure fields are published (value, method, confidence; not one of
the wrong kind or one the eye dropped; a title snapped to an old casing as the picture prints it), no message, poster,
note or review keys; one webp per version at most WEBP_W wide (a sheet's cell cut out, an old-repo blob read from its git
dir); the Morgue's candidates (M'uru's first 'Before 2020', owner Q6) and MDW cards; Created proposals only when
earlier or new (the physical card photo's 'Before 2020'); the eye's leave_out (no version of the card: the next one
diffed again, no Created from it). Stop at git: a card moved and redrawn in one commit starts
at its first file."""
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
             {"id": "D-P-x", "kind": "discord", "date": "2021-03-01T00:00:00+00:00", "day": "2021-03-01", "channel": "random-images",
              "look": {"draft": True}, "era": "other", "pictures": [pic("P-x", "a.png")], "minor": False, "fields": {}, "changes": {}},
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
                         "Pike": {"status": "later", "proposed": False, "date": "2020-01-01", "ours": {"source": "discord"}},
                         "Pep Drink": {"status": "earlier", "proposed": True, "date": "Before 2020", "sort": "0000-00-00",
                                       "ours": {"date": None, "day": "Before 2020", "source": "physical card photo",
                                                "owner": "Q6"}, "file": {"date": "2020-07-22"}}}}
    (root / "derived" / "chain" / "created.json").write_text(json.dumps(created))
    (root / "derived" / "match").mkdir(parents=True)
    (root / "derived" / "match" / "verdicts.jsonl").write_text(json.dumps({"id": "P-m", "file": "attachments/a.png"}) + "\n")
    (root / "derived" / "morgue").mkdir(parents=True)
    morgue = {"candidates": [{"id": "LC-1", "name": "Joke", "slug": "joke", "kind": "joke", "status": "candidate",
                              "first": {"date": "2020-01-01T00:00:00+00:00", "channel": "meme-cards", "source": "discord"},
                              "why": ["x"], "eye": {"note": "x"},
                              "pictures": [{"id": "P-m", "date": "2020-01-01T00:00:00+00:00", "channel": "meme-cards", "source": "discord"}]},
                             {"id": "LC-1c11be0504", "name": "M'uru", "slug": "m-uru", "kind": "played", "status": "candidate",
                              "first": {"date": None, "channel": None, "thread": None, "source": "physical card photo",
                                        "day": "Before 2020", "dated": {"date": "2020-07-18T00:00:00+00:00", "channel": "port"}},
                              "first_repo": {"paths": ["Cards/Muru.png"], "date": "2022-09-01", "mdw": False},
                              "pictures": [{"id": "P-m", "date": "2020-07-18T00:00:00+00:00", "channel": "port", "source": "discord"}]}],
              "not_candidates": [{"id": "LC-2", "name": "Old One", "card": "Old One", "status": "morgue", "first": {},
                                  "pictures": [{"id": "P-m", "date": None, "channel": None, "source": "discord"}]}]}
    (root / "derived" / "morgue" / "candidates.json").write_text(json.dumps(morgue))
    (root / "derived" / "preview").mkdir(parents=True)
    eye = {"drop": {"bolt|O-1|stats": "the eye: another value"}, "leave_out": {"bolt|D-P-x": "an AI restyle"}}
    (root / "derived" / "preview" / "eye.json").write_text(json.dumps(eye))
    (root / "derived" / "readers" / "text").mkdir(parents=True)
    (root / "derived" / "readers" / "text" / "reads.jsonl").write_text(json.dumps(
        {"src": "discord", "file": "attachments/a.png", "title": {"ocr": "* Bolt", "value": "BOLT"}}) + "\n")
    t["entries"][1]["fields"]["name"] = sure(["Bolt", "Bolt2"], "ocr+name")   # a value of the wrong kind
    (root / "derived" / "chain" / "timelines" / "bolt.json").write_text(json.dumps(t))
    index = dp.build(str(root), str(git), workers=1, cards_repo=None)
    out = root / "derived" / "preview" / "history"
    got = json.loads((out / "bolt" / "timeline.json").read_text())
    assert [e["kind"] for e in got["entries"]] == ["discord", "discord", "text", "old repo"]       # D-P-x left out
    assert not (out / "bolt" / "P-x.webp").exists()
    first, cell, fact, old = got["entries"]
    assert first["fields"] == {"name": {"value": "Bolt", "method": "ocr+name", "confidence": "sure"}}   # the review field left out
    assert fact["confirmed_by"] == "O-1" and {e["badge"] for e in got["entries"]} == {"Discord", "changelog", "old repo"}
    assert dp.printed_name("DooPliss", ["f"], {"f": "O DooPliss"}, "Doopliss") == "DooPliss"     # printed so: kept
    assert cell["git_look"] and fact["change"] == {"field": "atk", "from": 5, "to": 4} and old["minor"] and old["author"] == "Maker"
    assert "name" not in cell["fields"] and "stats" not in old["fields"]          # the wrong kind; the eye's drop
    assert not keys(got) & BAD_KEYS and not keys(index) & BAD_KEYS
    sizes = {e["image"]: Image.open(out / "bolt" / e["image"]).size for e in got["entries"] if e.get("image")}
    assert sizes["P-a.webp"] == (dp.WEBP_W, 2 * dp.WEBP_W) and sizes[f"O-{blob[:12]}.webp"] == (300, 600)
    assert sizes["P-s-c2.webp"] == (400, 600)                                       # the sheet's cell 2 alone
    assert Image.open(out / "bolt" / "P-s-c2.webp").convert("RGB").getpixel((200, 500))[1] > 150   # green
    assert index["created"] == {"Bolt": {"date": "2021-01-01", "sort": "2021-01-01", "source": "discord", "status": "earlier",
                                         "was": "2021-05-05"},
                                "Pep Drink": {"date": "Before 2020", "sort": "0000-00-00", "source": "physical card photo",
                                              "status": "earlier", "was": "2020-07-22"}}       # owner Q6: no picture of it
    assert [m["name"] for m in index["morgue"]] == ["Joke", "M'uru"] and [m["card"] for m in index["mdw"]] == ["Old One"]
    assert index["morgue"][1]["first"] == {"date": None, "day": "Before 2020", "channel": None,
                                           "source": "physical card photo"}
    muru = json.loads((out / "_morgue" / "m-uru" / "timeline.json").read_text())
    assert [(e["kind"], e["day"]) for e in muru["entries"]] == [("discord", "2020-07-18")]
    assert muru["git_file"] == "first repo" and "git_file" not in json.loads((out / "_morgue" / "joke" / "timeline.json").read_text())
    mdw = json.loads((out / index["mdw"][0]["dir"] / "timeline.json").read_text())
    assert mdw["entries"][0]["date"] is None and "printed" not in mdw["entries"][0]
    # a rebuild keeps the webps, and drops one no entry names any more
    (out / "bolt" / "stale.webp").write_bytes(b"x")
    dp.build(str(root), str(git), workers=1, cards_repo=None)
    assert not (out / "bolt" / "stale.webp").exists() and (out / "bolt" / "P-a.webp").exists()
    # the eye leaves out the picture Created came from: no proposal from it
    eye["leave_out"]["bolt|D-P-a"] = "not this card"
    (root / "derived" / "preview" / "eye.json").write_text(json.dumps(eye))
    assert set(dp.build(str(root), str(git), workers=1, cards_repo=None)["created"]) == {"Pep Drink"}
    got = json.loads((out / "bolt" / "timeline.json").read_text())
    assert [e["id"] for e in got["entries"]] == ["D-P-s-c2", "F-1", "O-1"] and not (out / "bolt" / "P-a.webp").exists()
    assert "created" not in got
    # the version after the ones left out is diffed again; a fact they confirmed loses it; so does git_first
    line = lambda atk: {"value": {"hp": 3, "def": 2, "atk": atk, "spd": 1}, "method": "picture", "confidence": "sure"}
    t = {"entries": [{"id": "D-1", "kind": "discord", "fields": {"stats": line(5)}, "changes": {}},
                     {"id": "D-X", "kind": "discord", "fields": {}, "changes": {}},
                     {"id": "F-1", "kind": "text", "change": {"field": "atk", "from": 5, "to": 4}, "confirmed_by": "D-X"},
                     {"id": "D-2", "kind": "discord", "fields": {"stats": line(4)}, "changes": {"unknown": ["stats"]}},
                     {"id": "D-Y", "kind": "discord", "fields": {"stats": line(9)}, "changes": {}}],
         "git_first": {"fields": {"stats": line(3)}, "changes": {"stats": [line(9)["value"], line(3)["value"]]}}}
    assert [e["id"] for e in dp.leave_out(t, {"D-X", "D-Y"})] == ["D-X", "D-Y"]
    assert [e["id"] for e in t["entries"]] == ["D-1", "F-1", "D-2"] and "confirmed_by" not in t["entries"][1]
    assert t["entries"][2]["changes"]["nerfs"] == ["ATK 5 -> 4"] and t["entries"][2]["minor"] is False
    assert t["git_first"]["changes"]["nerfs"] == ["ATK 4 -> 3"]


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
                          {"id": "P-3", "source": "discord", "date": None}]}
    assert dp.mdw_cut(entry, firsts["whitepalace"]) == ["P-2"] and [p["id"] for p in entry["pictures"]] == ["P-1", "P-3"]


def test_a_first_psd_s_row_and_created():
    e = {"kind": "old repo", "blob": "b" * 40, "thumb": "oldgit/thumbs/" + "b" * 40 + ".webp", "psd_only": True}
    assert dp.picture_job(e, "/r") == ("file", "/r/derived/oldgit/thumbs/" + "b" * 40 + ".webp", None)
    assert dp.picture_job({**e, "psd_only": False}, "/r") == ("blob", "b" * 40, None)
    laz = {"proposed": True, "status": "earlier", "date": "2023-09-10", "ours": {"source": "old repo (PSD)"},
           "file": {"date": "2023-10-09"}}
    got = dp.created_proposals({"cards": {"Laz": laz}})
    assert got["Laz"]["source"] == "old repo" and got["Laz"]["psd"] and got["Laz"]["was"] == "2023-10-09"
    pick = lambda date, source: {"proposed": True, "status": "owner", "date": date, "sort": date,
                                 "file": {"date": "2020-09-08"}, "owner": {"source": {"day": date, "source": source}}}
    got = dp.created_proposals({"cards": {"A": pick("2020-08-02", "discord"), "B": pick("2020-07-22", "owner"),
                                          "C": pick("2023-11-07", "discord"), "D": pick("2020-09-08", "website")}})
    assert {k: (v["date"], v["source"], v["status"]) for k, v in got.items()} == {    # the page's picks: earlier only
        "A": ("2020-08-02", "discord", "owner"), "B": ("2020-07-22", "owner", "owner")}


def test_the_morgue_list_follows_the_answers_and_takes_its_marks():
    """The review page's Morgue answers reach the preview (a candidate answered 'no' is off the list, an idea or a
    not-candidate answered 'yes' on it; the MDW cards as before), and a changelog line given to a Morgue entry (owner
    D9: 'morgue:<id>') is that entry's mark, data only."""
    morgue = {"candidates": [{"id": "LC-1"}, {"id": "LC-2"}],
              "not_candidates": [{"id": "LC-3", "status": "renamed"}, {"id": "LC-4", "status": "morgue"}]}
    rulings = {"morgue:LC-1": {"choice": "no"}, "morgue:LC-3": {"choice": "yes"}, "morgue:LC-5": {"choice": "yes"},
               "morgue:LC-4": {"choice": "no"}}
    got = dp.morgue_entries(morgue, [{"id": "LC-5"}, {"id": "LC-6"}], rulings)
    assert [(e["id"], kind) for e, kind in got] == [("LC-2", "candidate"), ("LC-5", "candidate"),
                                                     ("LC-3", "candidate"), ("LC-4", "mdw")]
    facts = [{"id": "f1", "card": "morgue:LC-2", "source": "changelog", "confidence": "sure", "kind": "ability changed",
              "date": "2019-03-22T06:33:53+00:00", "channel": "general-discussion", "line": "9:doc4"},
             {"id": "f2", "card": "morgue:LC-2", "source": "changelog", "confidence": "sure", "field": "stars", "to": 4,
              "date": "2019-03-22T06:33:53+00:00", "channel": "general-discussion"},
             {"id": "f3", "card": "Computer Virus", "source": "changelog", "confidence": "sure", "field": "stars",
              "to": 4, "date": "2019-03-22T06:33:53+00:00"}]
    marks = dp.morgue_marks(facts, {"id": "LC-2"})
    assert [(m["id"], m["kind"], m["change"]) for m in marks] == [("F-f1", "text", {"ability": "changed"}),
                                                                 ("F-f2", "text", {"field": "stars", "to": 4})]
    assert not keys(marks) & BAD_KEYS and "line" not in keys(marks)


def test_morgue_entries_show_their_old_and_first_repo_versions():
    """An MDW Morgue card's old repo rows (archived, by its name), a lost old-repo card's rows (they take the place of
    its old-repo picture) and the first repo's files linked to a candidate (picture rows) join its _morgue timeline:
    date, author, subject, badge and only sure fields; no link, poster or commit."""
    row = lambda rid, ts, blob, repo="old", **kw: {"id": rid, "ts": ts, "blob": blob, "repo": repo, "author": "Croissant",
                                                    "subject": "cards", "commit": "c" * 40, "path": "X.png", **kw}
    sure = {"name": {"value": "Clock", "method": "psd", "confidence": "sure"},
            "ability": {"value": "", "method": "psd", "confidence": "sure"},           # no text read: left out
            "stars": {"value": 9, "method": "folder", "confidence": "review"}}
    old = {"card": {"clock": [row("b", 1.7e9, "b" * 40, fields=sure), row("a", 1.6e9, "a" * 40, minor=True)]},
           "lost": {"Battery": [row("c", 1.69e9, "c" * 40, card_kind="item", fields={
               **sure, "types": {"value": ["equip"], "method": "icon pictures", "confidence": "sure"}})]},
           "first": {"Book1": [row("d", 1.66e9, "d" * 40, repo="first", fields={})]}}
    mdw = {"id": "LC-9", "name": "Clock", "card": "Clock", "pictures": []}
    out, jobs = dp.morgue_out(mdw, {}, "mdw", rows=dp.entry_rows(mdw, "mdw", old))
    assert [(e["id"], e["badge"], e["minor"]) for e in out["entries"]] == [("O-a", "old repo", True),
                                                                           ("O-b", "old repo", False)]
    assert out["entries"][1]["fields"] == {"name": {"value": "Clock", "method": "psd", "confidence": "sure"}}
    assert jobs[1] == ("O-bbbbbbbbbbbb.webp", ("blob", "b" * 40, None))
    lost = {"id": "OL-1", "name": "The Battery", "old_repo": {"old_id": "Battery"}, "first_files": ["Book1"],
            "pictures": [{"id": "OL-1", "source": "old repo", "thumb": "thumbs/OL-1.webp", "date": "2023-09-12"},
                         {"id": "P-1", "source": "discord", "thumb": "thumbs/P-1.webp", "date": "2020-01-01T00:00:00+00:00",
                          "channel": "port"}]}
    out, jobs = dp.morgue_out(lost, {}, "candidate", rows=dp.entry_rows(lost, "candidate", old))
    assert [(e["id"], e.get("repo")) for e in out["entries"]] == [("P-1", None), ("O-d", "first"), ("O-c", None)]
    assert out["entries"][1]["badge"] == "first repo" and not keys(out["entries"]) & BAD_KEYS
    assert "types" not in out["entries"][2]["fields"]                          # an item's kind: never collected
    assert "commit" not in keys(out["entries"]) and "path" not in keys(out["entries"])


def test_a_check_cut_short_goes_on_where_it_stopped(tmp_path, monkeypatch):
    """discord_preview check keeps each read in check.partial.jsonl as it goes: --redo reads only what is left, and
    the partial file goes once check.jsonl is written."""
    from concurrent.futures import ThreadPoolExecutor
    import commands.discord_preview_check as pc
    out = tmp_path / "derived" / "preview"
    out.mkdir(parents=True)
    jobs = [{"card": c, "id": "D-1", "image": "P-1.webp", "kind": "discord", "fields": {"stars": 2}} for c in "abc"]
    calls = []

    def one(job):
        calls.append(job["card"])
        if len(calls) == 3:
            raise KeyboardInterrupt                          # the run stopped here
        return {"card": job["card"], "id": job["id"], "image": job["image"], "fields": {"stars": {"r": "same", "read": 2}}}
    monkeypatch.setattr(pc, "jobs_of", lambda root: jobs)
    monkeypatch.setattr(pc, "check_one", one)
    monkeypatch.setattr(pc, "ProcessPoolExecutor", ThreadPoolExecutor)
    monkeypatch.setattr(pc.dm, "check_root", lambda root: root)
    try:
        pc.check(str(tmp_path), workers=1, redo=True)
    except KeyboardInterrupt:
        pass
    assert len((out / "check.partial.jsonl").read_text().splitlines()) == 2
    rows = pc.check(str(tmp_path), workers=1, redo=True)
    assert len(calls) == 4 and calls[3] == calls[2] and [r["card"] for r in rows] == ["a", "b", "c"]
    assert (out / "check.jsonl").exists() and not (out / "check.partial.jsonl").exists()


def test_a_type_list_the_eye_gave_goes_to_the_eye(tmp_path, monkeypatch):
    """A types list from the eye reference that reads the same is listed for the eye (a left-out icon is never read);
    a PSD's stays 'same'; a differing one stays 'differs'."""
    from concurrent.futures import ThreadPoolExecutor
    import commands.discord_preview_check as pc
    (tmp_path / "derived" / "preview").mkdir(parents=True)
    jobs = [{"card": c, "id": "D-1", "image": "P-1.webp", "kind": "discord", "fields": {"types": ["light"]},
             "types_method": m} for c, m in (("gk", "eye header"), ("psd", "psd"), ("pix", "eye"))]
    seen = {"gk": "same", "psd": "same", "pix": "differs"}
    monkeypatch.setattr(pc, "jobs_of", lambda root: jobs)
    monkeypatch.setattr(pc, "check_one", lambda j: {**{k: j[k] for k in ("card", "id", "image")},
                                                     "fields": {"types": {"r": seen[j["card"]], "read": {}}}})
    monkeypatch.setattr(pc, "ProcessPoolExecutor", ThreadPoolExecutor)
    monkeypatch.setattr(pc.dm, "check_root", lambda root: root)
    rows = pc.check(str(tmp_path), workers=1)
    assert [r["fields"]["types"]["r"] for r in rows] == ["eye", "same", "differs"]
    assert "  gk D-1 P-1.webp ['light']" in (tmp_path / "derived" / "preview" / "check.txt").read_text()
