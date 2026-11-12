"""The rulebook written on the website (owner 2026-10-29): its chapters as plain text for db/rulebook.json, its TTS PDF
printed again only when a page's pixels change, and the save's Rulebooks bag holding the PDF."""
import json
import os
import re
import shutil
import subprocess
import types
from io import BytesIO
from time import sleep, time

import pytest
from PIL import Image

from commands import rulebook, rulebook_pages, tts_save
from conftest import SITE_SRC

CHAPTER = """# Stats {.x}

## HEALTH {.hp}
![The bars](bars.webp){.right width=30}
- Each bar is **2 HP**; see [“ATTACKING”](#/rules/attacking).

::: row 3 1
> [!eggman top] *“Evil Quote!”*

![[Mushroom]] -> ![[Life Shroom|Shroom]]
:::

| ±0% | -25% |
|:---:|:---:|
| 1 | [0.5]{.red} |
{.grid cols=-,red}

=== page ===
1) Uses [[Toy Bonnie|the bunny]] \\[such as (BOGT)\\].
"""


def test_plain_text_and_chapters():
    """Markup goes, words and timing tags stay; files not named NN-slug.md are not chapters; the label numbers them."""
    assert rulebook.plain_text(CHAPTER).split("\n") == [
        "Stats", "", "HEALTH", "", "- Each bar is 2 HP; see “ATTACKING”.", "", "“Evil Quote!”", "", "Mushroom -> Life Shroom", "",
        "±0% | -25%", "1 | 0.5", "", "1) Uses the bunny [such as (BOGT)]."]
    found = rulebook.chapters({"site/rulebook/07-stats.md": CHAPTER, "site/rulebook/README.md": "# no", "site/x/01-a.md": "# no",
                               "site/rulebook/24-font.md": "# Font {.web}\n\nWords.\n"})
    assert [(c["id"], c["num"], c["title"], c["file"]) for c in found] == [("stats", "07", "Stats", "07-stats.md")]
    assert rulebook.label(found[0]) == "7. Stats" and rulebook.label({"num": "00", "title": "Table of Contents"}) == "Table of Contents"
    text = rulebook.rulebook_text(found)
    assert rulebook.parse_rulebook(text) == found and text.count("\n") == 3          # one chapter per line


def _png(colour) -> bytes:
    out = BytesIO()
    Image.new("RGB", (4, 6), colour).save(out, "PNG")
    return out.getvalue()


def test_pdf_printed_only_when_a_page_changes(tmp_path):
    """pages.json keeps each page's pixel hash; the PDF is printed again only when one changes (its bytes carry the print
    time); other files beside it stay; each page's text is kept for the TTS save's search."""
    folder = tmp_path / "TTS" / "Rulebook"
    folder.mkdir(parents=True)
    (folder / "notes.png").write_bytes(_png("white"))
    printed = []
    pdf = lambda: printed.append(1) or b"%PDF-1.4 fake"
    shots = [({"name": "Cover", "num": ""}, _png("white")), ({"name": "Stats", "num": "7", "text": "STATS HP DEF"}, _png("blue")), ({"name": "Stats 2", "num": "7.2"}, _png("red"))]
    done = rulebook_pages.write_pages(str(folder), shots, pdf)
    assert done["printed"] and (folder / "notes.png").exists() and (folder / "Rulebook.pdf").read_bytes() == b"%PDF-1.4 fake"
    again = [shots[0], (shots[1][0], _png("blue")), shots[2]]                     # same pixels
    assert not rulebook_pages.write_pages(str(folder), again, pdf)["printed"] and len(printed) == 1
    done = rulebook_pages.write_pages(str(folder), [shots[0], shots[1], (shots[2][0], _png("green"))], pdf)
    assert done["printed"] and done["changed"] == ["Stats 2"] and len(printed) == 2
    book = rulebook_pages.read_pages(str(tmp_path))
    assert book["pdf"] == "Rulebook.pdf" and [p["name"] for p in book["pages"]] == ["Cover", "Stats", "Stats 2"]
    assert book["pages"][1]["t"] == "STATS HP DEF" and "t" not in book["pages"][0]                   # each page's text: the save's search
    rulebook_pages.write_pages(str(folder), shots, pdf, "k1")                       # draw's key, for the next run's skip
    assert rulebook_pages.read_pages(str(tmp_path))["key"] == "k1" and len(printed) == 3
    assert not rulebook_pages.write_pages(str(folder), shots, pdf, "k2")["printed"]  # a new key, the same pixels: kept
    assert rulebook_pages.read_pages(str(tmp_path))["key"] == "k2" and len(printed) == 3
    assert rulebook_pages.key("<sheets>", str(tmp_path)) == rulebook_pages.key("<sheets>", str(tmp_path)) != rulebook_pages.key("<sheet>", str(tmp_path))


def _commit(repo, when, files):
    for rel, data in files.items():
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_bytes(data)
    env = {**os.environ, "GIT_AUTHOR_DATE": when, "GIT_COMMITTER_DATE": when}
    for args in (["add", "-A"], ["-c", "core.hooksPath=/dev/null", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", when]):
        subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, env=env)


def test_site_settled_by_its_build(tmp_path):
    """Settled: the website's build (the main commit dated cards.json's source; of a batch sharing a date, the newest) has
    HEAD's rulebook pictures and web fonts, Types/ icons and their aliases (41ab8074's table map, [buff]) and site code;
    later cards, book fonts, chapters (compared by content) or files no deploy takes (skeptic 6: a README, a non-picture)
    don't hold it back; no commit of that date in HEAD's history: a later main (the whole history looked at: a commit
    dated before its parent, as 1912f245, hides no build)."""
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    _commit(tmp_path, "2026-11-10T20:00:00+00:00", {"site/rulebook/01-a.md": b"# A", "site/rulebook/img/table-map.webp": b"1", "Items/a.png": b"a"})
    _commit(tmp_path, "2026-11-10T20:16:39+00:00", {"site/rulebook/img/table-map.webp": b"2"})
    settled = lambda source, branch="main": rulebook_pages.site_settled(str(tmp_path), {"cards": {"source": source, "branch": branch}})
    assert not settled("2026-11-10T20:00:00+00:00") and settled("2026-11-10T20:16:39+00:00")       # 41ab8074
    _commit(tmp_path, "2026-11-10T20:30:00+00:00", {"Types/Buff.PNG": b"b"})
    for files in ({"Items/b.png": b"b", "MDW/Fonts/a.ttf": b"f"}, {"Types/Stars/S.png": b"s", "site/rulebook/notes.txt": b"n"}):
        _commit(tmp_path, "2026-11-10T21:00:00+00:00", files)                 # a batch: one date
    _commit(tmp_path, "2026-11-10T21:05:00+00:00", {"Items/c.png": b"c", "Types/readme.webp": b"r"})
    for source in ("2026-11-10T21:00:00+00:00", "2026-11-10T21:05:00+00:00", "2026-11-10T22:00:00+00:00"):
        assert settled(source), source
    for source in ("2026-11-10T12:16:39-08:00", "2026-11-10T20:00:00+00:00", "2026-11-10T20:30:00+00:00", ""):
        assert not settled(source), source
    assert not settled("2026-11-10T21:00:00+00:00", "work")
    _commit(tmp_path, "2026-11-10T21:10:00+00:00", {"site/assets/rulebook.js": b"j"})
    assert not settled("2026-11-10T21:05:00+00:00") and settled("2026-11-10T21:10:00+00:00")
    _commit(tmp_path, "2026-11-10T21:20:00+00:00", {"Items/d.png": b"d"})                        # the build
    _commit(tmp_path, "2026-11-10T21:15:00+00:00", {"Items/e.png": b"e"})                        # dated before its parent
    _commit(tmp_path, "2026-11-10T21:25:00+00:00", {"site/rulebook/img/table-map.webp": b"3"})
    assert not settled("2026-11-10T21:20:00+00:00") and settled("2026-11-10T21:25:00+00:00")
    _commit(tmp_path, "2026-11-10T21:30:00+00:00", {"site/rulebook/README.md": b"n", "site/rulebook/25-New Chapter.md": b"# B",
                                                    "site/rulebook/01-a.md": b"# A\r\n", "site/rulebook/img/source.psd": b"8BPS",
                                                    "site/rulebook/img/old/x.webp": b"x"})
    assert settled("2026-11-10T21:25:00+00:00")
    _commit(tmp_path, "2026-11-10T21:35:00+00:00", {"MDW/Fonts/stoogedings-src/stoogedings.json": b"{}"})
    assert not settled("2026-11-10T21:30:00+00:00") and settled("2026-11-10T21:35:00+00:00")
    for files in ({"Items/f.png": b"f"}, {"site/rulebook/img/table-map.webp": b"4"}):
        _commit(tmp_path, "2026-11-10T21:40:00+00:00", files)                 # the website: the batch's newest
    _commit(tmp_path, "2026-11-10T21:45:00+00:00", {"site/rulebook/img/table-map.webp": b"3"})   # as the batch's first
    assert not settled("2026-11-10T21:40:00+00:00")


def _seen(**cards):
    rules = {"chapters": [{"file": "01-a.md", "source": "# A\n![[Mushroom]] and [[Panda|the panda]]\n"}], "images": {"cover": [1, 2, "s1"]}}
    data = {"icons": {"fire": {"file": "icons/fire.1.webp"}}, "iconVariants": {}, "tagMeanings": {}, "thumbs": {"360": "t1"},
            "cards": [{"id": "mushroom", "name": "Mushroom", "img": "m1"}, {"id": "panda", "name": "Panda", "img": "p1"},
                      {"id": "kirby", "name": "Kirby", "img": "k1", "hp": 3}]}
    for name, fields in cards.items():
        next(c for c in data["cards"] if c["id"] == name).update(fields)
    return {"page": "pg", "index": '<link href="assets/app.js?v=a1"><style>.rb{}</style><meta name="build" content="d1">',
            "rules": json.dumps(rules).encode(), "cards": data, "build": "d1 pg\n"}


def test_pre_key_and_the_skip(tmp_path, monkeypatch):
    """The pre-key follows what the print view reads (code, chapters, the named cards' pictures, every card's name), not
    the data stamp or other cards' pictures; kept beside the key, it skips Chrome when PRE_SKIP (on; off, the shadow:
    Chrome looks and a key change warns; --full looks in any case)."""
    pre = lambda seen: rulebook_pages.pre_key(seen, ["t"], [], "/no/chrome")
    base = pre(_seen())
    same = _seen(kirby={"img": "k2", "hp": 4}, panda={"img": "p2"})            # a picture only linked, another card
    same["index"] = same["index"].replace("d1", "d2")
    assert pre(same) == base
    for seen in (_seen(mushroom={"img": "m2"}), _seen(kirby={"name": "Kirbo"}), dict(_seen(), index='<link href="assets/app.js?v=a2">')):
        assert pre(seen) != base
    assert rulebook_pages.pre_key(_seen(), ["u"], [], "/no/chrome") != base
    (tmp_path / "site" / "rulebook").mkdir(parents=True)
    (tmp_path / "site" / "rulebook" / "01-a.md").write_text("# A\n![[Mushroom]] and [[Panda|the panda]]\n", encoding="utf-8")
    folder = tmp_path / "TTS" / "Rulebook"
    folder.mkdir(parents=True)
    shots = [({"name": "Cover", "num": ""}, _png("white"))]
    rulebook_pages.write_pages(str(folder), shots, lambda: b"%PDF", "k1")
    monkeypatch.setattr(rulebook_pages, "site_inputs", lambda site: _seen())
    monkeypatch.setattr(rulebook_pages, "site_settled", lambda checkout, seen: True)
    drawn = {"key": "k1", "n": 0}

    def draw(site, exe, key_of, known, fonts):
        drawn["n"] += 1
        new = drawn["key"] != known
        return ({"01-a.md": "# A\n![[Mushroom]] and [[Panda|the panda]]\n"}, shots if new else None,
                rulebook_pages.Printer(lambda: b"%PDF", lambda: None), drawn["key"], "pg", {})
    monkeypatch.setattr(rulebook_pages, "draw", draw)
    monkeypatch.setattr(rulebook_pages, "find_chrome", lambda: "/no/chrome")
    run = lambda **kw: rulebook_pages.update_pages(str(tmp_path), **kw)
    mine = rulebook_pages.pre_key(_seen(), rulebook_pages._trees(str(tmp_path)), [], "/no/chrome")
    assert "pre-key held" not in run() and rulebook_pages.read_pages(str(tmp_path))["pre"] == mine            # stored with the key
    monkeypatch.setattr(rulebook_pages, "PRE_SKIP", False)
    assert "pre-key held" in run() and drawn["n"] == 2                                                        # shadow: Chrome looks
    monkeypatch.setattr(rulebook_pages, "PRE_SKIP", True)
    assert "Chrome not started" in run() and drawn["n"] == 2
    assert "pre-key held" in run(full=True) and drawn["n"] == 3
    monkeypatch.setattr(rulebook_pages, "PRE_SKIP", False)
    drawn["key"] = "k2"                                                                                           # an input pre misses
    assert "::warning::" in run() and rulebook_pages.read_pages(str(tmp_path))["key"] == "k2"


def test_the_look_waits_for_the_site(tmp_path, monkeypatch, capsys):
    """11-10 (41ab8074): the look waits (up to wait, no Chrome; looked at every SITE_POLL, and at first when build.txt,
    read every SITE_POLL_FIRST, changed) until the website shows HEAD's chapters and is settled, then draws; Chrome drew
    another build: looked at again at once; an older copy of a plain-URL font or picture than HEAD's (the CDN's): drawn
    again after SITE_POLL, in that time (skeptic 6). Else nothing changes and the command line exits 3 (card-sync.yml
    marks the lane)."""
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    _commit(tmp_path, "2026-11-10T20:00:00+00:00", {"site/rulebook/01-a.md": b"# A\n", "site/rulebook/fonts/a b.woff2": b"f2"})
    font = rulebook_pages.heads(str(tmp_path), ["rulebook/fonts/a%20b.woff2", "rulebook/img/gone.webp"])
    assert font["rulebook/img/gone.webp"] is None and len(font["rulebook/fonts/a%20b.woff2"]) == 40
    folder = tmp_path / "TTS" / "Rulebook"
    folder.mkdir(parents=True)
    shots = [({"name": "Cover", "num": ""}, _png("white"))]
    rulebook_pages.write_pages(str(folder), shots, lambda: b"%PDF", "k0")
    monkeypatch.setattr(rulebook_pages, "site_inputs", lambda site: dict(_seen(), rules=json.dumps({"chapters": [{"file": "01-a.md", "source": "# A\n"}]}).encode()))
    monkeypatch.setattr(rulebook_pages, "find_chrome", lambda: "/no/chrome")
    fast, poll, draw_s = rulebook_pages.SITE_POLL_FIRST, rulebook_pages.SITE_POLL, 12
    clock = {"t": 0, "sleeps": []}                         # time passes in sleeps and draws

    def nap(s):
        clock["sleeps"].append(s)
        clock["t"] += s
    monkeypatch.setattr(rulebook_pages, "sleep", nap)
    monkeypatch.setattr(rulebook_pages, "monotonic", lambda: clock["t"])
    monkeypatch.setattr(rulebook_pages, "_build", lambda site: "d1 pg\n" if clock["t"] < 30 else "d2 pg\n")   # a deploy at 30 s
    good = ("pg", dict(font, **{"rulebook/img/gone.webp": None}))
    drawn = {"n": 0, "lag": []}                            # what the next draws meet: (page, plain), then good

    def draw(site, exe, key_of, known, fonts):
        drawn["n"] += 1
        clock["t"] += draw_s
        page, plain = drawn["lag"].pop(0) if drawn["lag"] else good
        return {"01-a.md": "# A\n"}, shots, rulebook_pages.Printer(lambda: b"%PDF", lambda: None), "k1", page, plain
    monkeypatch.setattr(rulebook_pages, "draw", draw)
    run = lambda: (clock["sleeps"].clear(), rulebook_pages.main([str(tmp_path), "--wait", "600"]), capsys.readouterr().out)[1:]
    looks = iter([False, False, True])                     # looked at 0 s, SITE_POLL, the deploy (settled)
    monkeypatch.setattr(rulebook_pages, "site_settled", lambda checkout, seen: next(looks))
    assert run() == (0, "Rulebook PDF: 1 pages, as they were (key k1 kept).\n") and drawn["n"] == 1 and next(looks, None) is None
    assert clock["t"] == 30 + draw_s and clock["sleeps"] == [fast] * 15
    rulebook_pages.write_pages(str(folder), shots, lambda: b"%PDF", "k0")
    monkeypatch.setattr(rulebook_pages, "site_settled", lambda checkout, seen: False)   # not in 10 minutes
    code, out = run()
    assert code == 3 and rulebook_pages.LAGS in out and drawn["n"] == 1
    assert clock["sleeps"] == [fast] * (rulebook_pages.SITE_FAST // fast) + [poll] * ((600 - rulebook_pages.SITE_FAST) // poll)
    monkeypatch.setattr(rulebook_pages, "site_settled", lambda checkout, seen: True)
    for lag in (("pg-old", good[1]), ("pg", dict(good[1], **{"rulebook/fonts/a%20b.woff2": "0" * 40})),
                ("pg", dict(good[1], **{"rulebook/fonts/a%20b.woff2": None}))):
        drawn["lag"], n = [lag] * 1000, drawn["n"]          # past the wait
        code, out = run()
        assert code == 3 and rulebook_pages.LAGS in out and rulebook_pages.read_pages(str(tmp_path))["key"] == "k0", lag
        assert 1 < drawn["n"] - n <= 600 / (draw_s if lag[0] == "pg-old" else poll), lag
        drawn["lag"], n = [lag] * 2, drawn["n"]             # a deploy mid-look, the CDN's copy renewed: drawn again
        assert run() == (0, "Rulebook PDF: 1 pages, as they were (key k1 kept).\n") and drawn["n"] - n == 3, lag
        assert clock["sleeps"] == ([] if lag[0] == "pg-old" else [poll] * 2), lag     # another build: no sleep
        rulebook_pages.write_pages(str(folder), shots, lambda: b"%PDF", "k0")
    out = tmp_path / "out"                                 # Card sync's look in the background: the checkout left alone
    assert rulebook_pages.main([str(tmp_path), "--wait", "0", "--out", str(out)]) == 0
    head = subprocess.run(["git", "-C", str(tmp_path), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout
    assert (out / "inputs").read_text() == "\n".join((head.strip(), *rulebook_pages.INPUTS)) + "\n" and (out / "Rulebook.pdf").is_file()
    assert rulebook_pages.read_pages("", str(out))["key"] == "k1" and rulebook_pages.read_pages(str(tmp_path))["key"] == "k0"


def _gh_bash(script, cwd, env, timeout=60):
    """A workflow step's script as GitHub's shell: bash runs it (expressions dropped)."""
    script = re.sub(r"\$\{\{.*?\}\}", "", script)
    return subprocess.run(["bash", "--noprofile", "--norc", "-eo", "pipefail", "-c", script], cwd=cwd, env=env,
                          capture_output=True, text=True, timeout=timeout)


def test_card_sync_marks_a_look_that_waited_in_vain(tmp_path):
    """card-sync.yml as written: Rulebook PDF's exit 3 (the website lagged past the wait) sets lag and the step after it
    marks the lane, as it does a failed step (exit 1, its time run out); the Gate of the schedule's next run passes over a
    marked lane, so a rulebook picture's commit since the lane before it is card input (GO); unmarked, nothing to do; the
    lane before it marked too (a lag that lasts, skeptic 6): not passed over, nothing to do."""
    yaml = pytest.importorskip("yaml")
    path = SITE_SRC / ".github" / "workflows" / "card-sync.yml"
    if not path.is_file():
        pytest.skip(f"no {path}")
    main = yaml.safe_load(path.read_text(encoding="utf-8"))["jobs"]["main"]
    assert main["defaults"]["run"]["shell"] == "bash"
    steps = {s.get("name"): s for s in main["steps"]}
    names = [s.get("name") for s in main["steps"]]
    pages, gate = steps["Rulebook PDF"], steps["Gate"]
    mark = steps[gate["env"]["UNDRAWN"]]
    assert names.index(mark["name"]) == names.index("Rulebook PDF") + 1
    assert all(c in mark["if"] for c in ("steps.pages.outputs.lag == 'true' || steps.pages.outcome == 'failure'", "env.DRY_RUN != 'true'"))
    assert pages["continue-on-error"] is True and pages["timeout-minutes"] * 60 >= rulebook_pages.SITE_WAIT + 120
    repo, bin_ = tmp_path / "ws", tmp_path / "bin"
    (repo / "bot").mkdir(parents=True)
    bin_.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    _commit(repo, "2026-11-10T20:00:00+00:00", {"TTS/Rulebook/pages.json": b"{}", "site/rulebook/01-a.md": b"# A"})
    old = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    _commit(repo, "2026-11-10T20:16:39+00:00", {"site/rulebook/img/table-map.webp": b"2"})
    head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    venv = tmp_path / "temp" / "venv" / "bin"
    venv.mkdir(parents=True)
    for name, body in {"fc-list": "echo 'Noto Color Emoji'", "python": 'echo "Rulebook PDF: a look"; exit "$CODE"',
                       "gh": 'f=$1; [ "$1" = api ] && f=$(basename "$(dirname "$2")"); jq -r "${@: -1}" "$FIX/$f.json"'}.items():
        for where in (bin_, venv) if name == "python" else (bin_,):
            (where / name).write_text(f"#!/bin/bash\n{body}\n")
            (where / name).chmod(0o755)
    env = {"PATH": f"{bin_}:{os.path.dirname(shutil.which('jq'))}:/usr/bin:/bin", "RUNNER_TEMP": str(tmp_path / "temp"), "GITHUB_WORKSPACE": str(repo),
           "SITE_URL": "https://site", "GITHUB_STEP_SUMMARY": str(tmp_path / "summary"), "HOME": str(tmp_path)}
    for code, ok, lag in (("0", True, "false"), ("3", True, "true"), ("1", False, None)):
        out = tmp_path / f"out{code}"
        out.write_text("")
        done = _gh_bash(pages["run"], repo / "bot", {**env, "CODE": code, "GITHUB_OUTPUT": str(out)})
        assert (done.returncode == 0) is ok, (code, done.stdout, done.stderr)
        assert (f"lag={lag}" in out.read_text().split()) if lag else "lag=" not in out.read_text(), (code, out.read_text())
    fix = tmp_path / "fix"
    fix.mkdir()
    (fix / "run.json").write_text(json.dumps([{"databaseId": 3, "headSha": head, "displayTitle": "Card sync"},
                                              {"databaseId": 2, "headSha": old, "displayTitle": "Card sync"}]))
    job = lambda marked: {"jobs": [{"name": gate["env"]["LANE"], "conclusion": "success", "steps": [
        {"name": "Gate", "conclusion": "success"}, {"name": "Rulebook PDF", "conclusion": "success"},
        {"name": mark["name"], "conclusion": "success" if marked else "skipped"}]}]}
    (fix / "2.json").write_text(json.dumps(job(False)))
    for marked, go in ((True, "true"), (False, "false"), ((True, True), "false")):
        if marked == (True, True):                       # 4, a schedule run's lane at head, marked too
            (fix / "run.json").write_text(json.dumps([{"databaseId": 4, "headSha": head, "displayTitle": "Card sync"},
                                                      *json.loads((fix / "run.json").read_text())]))
            (fix / "4.json").write_text(json.dumps(job(True)))
        (fix / "3.json").write_text(json.dumps(job(bool(marked))))
        envs = tmp_path / "env"
        envs.write_text("")
        done = _gh_bash(gate["run"], repo, {**env, "FIX": str(fix), "GITHUB_EVENT_NAME": "schedule", "GITHUB_REPOSITORY": "o/r",
                                            "GITHUB_OUTPUT": str(tmp_path / "gate-out"), "GITHUB_ENV": str(envs), "DRY_RUN": "false",
                                            "LANE": gate["env"]["LANE"], "UNDRAWN": gate["env"]["UNDRAWN"], "RETRY": "", "CHAIN": "",
                                            "GIT_NO_LAZY_FETCH": "1"})
        assert done.returncode == 0 and f"GO={go}" in envs.read_text().split(), (marked, done.stdout, done.stderr)


def test_card_sync_keeps_the_look_in_the_background(tmp_path):
    """card-sync.yml as written: Rulebook PDF start looks in the background (--out) once main's history is in; Rulebook
    PDF keeps that look when main's head since left its inputs alone (a card), else stops it and looks again (a Types/
    icon), as after a failure; after a lag, once (--wait 0). Rulebook PDF stop ends a look."""
    yaml = pytest.importorskip("yaml")
    path = SITE_SRC / ".github" / "workflows" / "card-sync.yml"
    if not path.is_file():
        pytest.skip(f"no {path}")
    steps = {s.get("name"): s for s in yaml.safe_load(path.read_text(encoding="utf-8"))["jobs"]["main"]["steps"]}
    start, pages, stop = steps["Rulebook PDF start"], steps["Rulebook PDF"], steps["Rulebook PDF stop"]
    assert pages["env"]["STARTED"] == "${{ steps.pdf-start.outcome }}" and "steps.pdf-start.outcome == 'success'" in stop["if"]
    repo, bin_, temp, calls = tmp_path / "ws", tmp_path / "bin", tmp_path / "temp", tmp_path / "calls"
    (repo / "bot").mkdir(parents=True)
    bin_.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    _commit(repo, "2026-11-12T10:00:00+00:00", {"TTS/Rulebook/pages.json": b"{}\n", "site/rulebook/01-a.md": b"# A"})
    venv = temp / "venv" / "bin"
    venv.mkdir(parents=True)
    look = ('out=""; prev=""; for a in "$@"; do [ "$prev" = --out ] && out=$a; prev=$a; done; echo "$*" >> "$CALLS"\n'
            'if [ -n "$out" ]; then mkdir -p "$out"; echo \'{"key": "bg"}\' > "$out/pages.json"\n'
            '  { git -C "$GITHUB_WORKSPACE" rev-parse HEAD; printf "%s\\n" "$INPUTS"; } > "$out/inputs.part"; mv "$out/inputs.part" "$out/inputs"\n'
            '  sleep "${NAP:-0}"; echo "Rulebook PDF: the background\'s"; exit "${CODE:-0}"; fi\n'
            'echo \'{"key": "here"}\' > "$GITHUB_WORKSPACE/TTS/Rulebook/pages.json"; echo "Rulebook PDF: here"')
    for where, name, body in ((venv, "python", look), (bin_, "fc-list", "echo 'Noto Color Emoji'")):
        (where / name).write_text(f"#!/bin/bash\n{body}\n")
        (where / name).chmod(0o755)
    env = {"PATH": f"{bin_}:{os.environ['PATH']}", "RUNNER_TEMP": str(temp), "GITHUB_WORKSPACE": str(repo), "SITE_URL": "https://site",
           "HOME": str(tmp_path), "CALLS": str(calls), "INPUTS": "\n".join(rulebook_pages.INPUTS), "STARTED": "success"}
    (temp / "history.code").write_text("0\n")
    bg = temp / "pdf"

    def begin(**more):
        shutil.rmtree(bg, ignore_errors=True)
        calls.write_text("")
        subprocess.run(["git", "-C", str(repo), "checkout", "-q", "--", "TTS/Rulebook"], check=True)
        assert _gh_bash(start["run"], repo / "bot", {**env, **more}).returncode == 0
        for _ in range(200):
            if (bg / "out" / "inputs").exists():
                return
            sleep(0.05)
        raise AssertionError("no look")

    def ended():                                         # the look's exit code, in 5 s
        for _ in range(100):
            if (bg / "code").exists():
                return (bg / "code").read_text().strip()
            sleep(0.05)

    for when, (moved, more, kept, again) in enumerate((({"Items/x.png": b"x"}, {}, True, None),
                                                       ({"Types/Fire.PNG": b"f"}, {"NAP": "30"}, False, ""),
                                                       (None, {"CODE": "3"}, False, "--wait 0"), (None, {"CODE": "1"}, False, ""))):
        begin(**more)
        if moved:                                        # Newest brought it
            _commit(repo, f"2026-11-12T10:0{when}:00+00:00", moved)
        out = tmp_path / "out"
        out.write_text("")
        t = time()
        done = _gh_bash(pages["run"], repo / "bot", {**env, "GITHUB_OUTPUT": str(out)})
        assert done.returncode == 0 and time() - t < 10 and "lag=false" in out.read_text().split(), (when, done.stdout, done.stderr)
        key = json.loads((repo / "TTS" / "Rulebook" / "pages.json").read_text())["key"]
        looks = calls.read_text().splitlines()
        assert key == ("bg" if kept else "here") and len(looks) == (1 if kept else 2) and "--out" in looks[0], (when, looks)
        assert kept or ("--out" not in looks[1] and ("--wait 0" in looks[1]) == bool(again)), (when, looks)
        assert ended() == more.get("CODE", "143" if moved and not kept else "0"), when      # stopped when looked at again
    begin(NAP="30")
    assert _gh_bash(stop["run"], repo, env).returncode == 0 and ended() == "143"


def test_card_sync_saves_right_after_the_gate(tmp_path):
    """card-sync.yml as written: Early save starts the save and the Linked redraw's count on main's head; TTS save start
    keeps both while HEAD is there, and a count of 0 skips the redraw; after the Bake's B it stops that save, puts the
    file back and starts both again on B, whose count of 1 lets the redraw run."""
    yaml = pytest.importorskip("yaml")
    path = SITE_SRC / ".github" / "workflows" / "card-sync.yml"
    if not path.is_file():
        pytest.skip(f"no {path}")
    main = yaml.safe_load(path.read_text(encoding="utf-8"))["jobs"]["main"]
    names = [s.get("name") for s in main["steps"]]
    steps = {s.get("name"): s for s in main["steps"]}
    early, start, redraw = steps["Early save"], steps["TTS save start"], steps["Linked redraw"]
    assert names.index("Early save") == names.index("Gate") + 1 and early["run"] == start["run"] and early["continue-on-error"] is True
    repo, temp, calls = tmp_path / "ws", tmp_path / "temp", tmp_path / "calls"
    save = repo / "TTS" / "Impossibility Simulator.json"
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    _commit(repo, "2026-11-12T10:00:00+00:00", {"TTS/Impossibility Simulator.json": b"{}\n", "A.psd": b"a"})
    (repo / "bot" / "commands").mkdir(parents=True)
    (repo / "bot" / "commands" / "psd_links.py").write_text("")
    (repo / "carddata" / "db").mkdir(parents=True)
    (repo / "carddata" / "db" / "links.json").write_text("{}")
    venv = temp / "venv" / "bin"
    venv.mkdir(parents=True)
    (venv / "python").write_text('#!/bin/bash\necho "$2 $3 $*" >> "$CALLS"\ncase $2 in\n'
                                 '  commands.tts_save) echo "{\\"by\\": \\"$WHO\\"}" > "$3"; sleep "${NAP:-0}"; echo "TTS save: $WHO" ;;\n'
                                 '  *) [ "$3" = due ] && echo "$DUE" || echo "## Linked redraw"; ;;\nesac\n')
    (venv / "python").chmod(0o755)
    env = {"PATH": os.environ["PATH"], "RUNNER_TEMP": str(temp), "GITHUB_WORKSPACE": str(repo), "HOME": str(tmp_path), "CALLS": str(calls),
           "GITHUB_STEP_SUMMARY": str(tmp_path / "summary"), "START": str(int(time())), "GIT_NO_LAZY_FETCH": "1"}
    head = lambda: subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    main_head = head()

    def done(*files, word=""):
        for _ in range(200):
            if all(f.exists() and f.read_text().strip() and word in f.read_text() for f in files):
                return [f.read_text().strip() for f in files]
            sleep(0.05)
        raise AssertionError(f"not done: {files}")

    def redrawn(**more):
        out = tmp_path / "out"
        out.write_text("")
        run = _gh_bash(redraw["run"], repo / "bot", {**env, "MAIN": main_head, "GITHUB_OUTPUT": str(out), **more})
        assert run.returncode == 0, run.stderr
        return run.stdout, out.read_text().split()

    first = _gh_bash(early["run"], repo / "bot", {**env, "WHO": "early", "DUE": "0"})
    assert first.returncode == 0 and done(temp / "save" / "code", temp / "due" / main_head / "code") == ["0", "0"], first.stderr
    kept = _gh_bash(start["run"], repo / "bot", {**env, "WHO": "again", "DUE": "5"})
    assert kept.returncode == 0 and "kept" in kept.stdout and json.loads(save.read_text()) == {"by": "early"}
    stdout, out = redrawn()
    assert "0 PSDs drawn again" in stdout and "due=0" in out and "psd_links redraw" not in calls.read_text()
    lines = calls.read_text().splitlines()
    assert len(lines) == 2 and lines[0].startswith(f"commands.tts_save {save} ") and lines[1].startswith("commands.psd_links due ")

    shutil.rmtree(temp / "save")
    calls.write_text("")
    assert _gh_bash(early["run"], repo / "bot", {**env, "WHO": "early", "DUE": "0", "NAP": "30"}).returncode == 0
    done(save, word="early")
    _commit(repo, "2026-11-12T10:05:00+00:00", {"A.psd": b"baked"})          # the Bake's B
    again = _gh_bash(start["run"], repo / "bot", {**env, "WHO": "on B", "DUE": "1"})
    assert again.returncode == 0 and "made again" in again.stdout, again.stderr
    assert done(temp / "save" / "code", temp / "due" / head() / "code") == ["0", "0"] and json.loads(save.read_text()) == {"by": "on B"}
    stdout, out = redrawn()
    assert "## Linked redraw" in stdout and "0 PSDs drawn again" not in stdout and [line.split()[1] for line in calls.read_text().splitlines()].count("redraw") == 1


def _card(guid, path, nick, deck_id):
    url = f"{tts_save.RAW}abc1234/{path}"
    return {"GUID": guid, "Name": "CardCustom", "Nickname": nick, "Description": "theirs", "CardID": deck_id * 100,
            "CustomDeck": {str(deck_id): {"FaceURL": url, "BackURL": url, "NumWidth": 1, "NumHeight": 1}}, "LuaScript": "x",
            "Transform": {"posX": 1.0, "scaleX": 4.0, "scaleY": 1.0, "scaleZ": 4.0}}


def test_rulebooks_bag_takes_the_pdf(tmp_path):
    """One Custom_PDF object is made on top of the Rulebooks bag (linked to the PDF, a fresh GUID), the bag wears
    MDW/Assets/Book_Cover.png; a second run changes nothing and keeps the team's changes to the PDF object."""
    folder = tmp_path / "TTS" / "Rulebook"
    folder.mkdir(parents=True)
    (folder / "pages.json").write_text(json.dumps({"pdf": "Rulebook.pdf", "pages": [{"name": "Cover"}, {"name": "Stats"}]}), encoding="utf-8")
    note = {"GUID": "cccccc", "Name": "Notecard", "Nickname": "House rules"}
    bag = {"GUID": "dddddd", "Name": "Custom_Model_Bag", "Nickname": "Rulebooks", "Locked": False, "Transform": {"posX": 1.0, "scaleX": 5.0},
           "CustomMesh": {"DiffuseURL": f"{tts_save.RAW}abc1234/MDW/Old%20Rules/Rulebook/0000_Cover_Book.png"}, "ContainedObjects": [note]}
    save = {"ObjectStates": [bag, _card("eeeeee", "Items/x.png", "X", 90)]}
    repo = types.SimpleNamespace(checkout=str(tmp_path), url=lambda p: f"{tts_save.RAW}main/{p}", sha=lambda p: None,
                                 tree={tts_save.BOOK_COVER: "b"})
    report = tts_save._Report()
    tts_save._rulebook(save, repo, report)
    inside = bag["ContainedObjects"]
    assert len(inside) == 2 and inside[0] is note and inside[1]["Name"] == "Custom_PDF" and inside[1]["Nickname"] == "Rulebook"
    pdf = inside[1]
    assert pdf["CustomPDF"] == {"PDFUrl": f"{tts_save.RAW}main/TTS/Rulebook/Rulebook.pdf", "PDFPassword": "", "PDFPage": 0, "PDFPageOffset": 0}
    assert pdf["Transform"]["scaleX"] == tts_save.PDF_SCALE and pdf["Transform"]["posX"] == 1.0
    assert bag["CustomMesh"]["DiffuseURL"] == f"{tts_save.RAW}main/MDW/Assets/Book_Cover.png"
    assert report.unpinned == {"TTS/Rulebook/Rulebook.pdf"}
    guids = [o.get("GUID") for o, _ in tts_save._walk(save["ObjectStates"])]
    assert len(guids) == len(set(guids))
    pdf["Transform"]["scaleX"] = 2.0                       # the team resized it
    pdf["CustomPDF"]["PDFUrl"] = f"{tts_save.RAW}{'1' * 40}/TTS/Rulebook/Rulebook.pdf"   # _pin_links pinned it
    bag["CustomMesh"]["DiffuseURL"] = f"{tts_save.RAW}{'2' * 40}/MDW/Assets/Book_Cover.png"
    again = tts_save._Report()
    tts_save._rulebook(save, repo, again)
    assert again.rulebook == [] and bag["ContainedObjects"] == inside and pdf["Transform"]["scaleX"] == 2.0
