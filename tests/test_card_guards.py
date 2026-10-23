"""
card_guards (the push guards' driver): a card repo made here with git (fixture PSDs, every case of PUSH_GUARDS.md 4.2,
bot commits left out), check end to end on it (the report, the one Discord post without mentions or previews, the plan),
land against a fake GitHub (superseded cards, a moved head, the quiet wait) and in a local clone, the replay of the real
push 38c55bd3..07589cc7 from the card clone (read-only: God Tamer, Bubble and 12 Reactions2 "PNG still current"), the
Action's sparse copy (SPARSE_FILES), card-guards.yml's shape, and its steps' shell against a fake gh and curl (one post
per push, the failure streak, the refused dispatch, MODE in any case).
"""
import json
import os
import re
import shutil
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from sys import executable

import pytest
from PIL import Image, ImageDraw

import commands.card_guards as cg
import commands.git_commit as gc
from commands.github_api import blob_sha
from test_png_export import card, fixture_psd

# ========================
# Configuration
# ========================
REPO = Path(__file__).resolve().parents[1]
FOLDER = "Items/Heal/2 Stars"
WORKFLOW = Path(os.environ.get("CARD_GUARDS_YML", "")) if os.environ.get("CARD_GUARDS_YML") else None


# ========================
# A card repo
# ========================
def _git(root, *args, author="Base", subject=None):
    env = {**os.environ, "GIT_AUTHOR_NAME": author, "GIT_AUTHOR_EMAIL": f"{author.lower()}@example.com",
           "GIT_COMMITTER_NAME": author, "GIT_COMMITTER_EMAIL": f"{author.lower()}@example.com"}
    if author == "AutoCroissant":
        env.update(GIT_AUTHOR_EMAIL=gc.AUTHOR["email"], GIT_COMMITTER_EMAIL=gc.AUTHOR["email"])
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, check=True, env=env).stdout.strip()


def _commit(root, message, author="Base"):
    _git(root, "add", "-A", author=author)
    _git(root, "commit", "-q", "-m", message, author=author)
    return _git(root, "rev-parse", "HEAD")


def _psd(root, work, rel, image):
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    fixture_psd(work, image, Path(rel).stem)
    shutil.copy(work / f"{Path(rel).stem}.psd", path)


def _png(root, rel, image):
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


@pytest.fixture
def cards(tmp_path):
    """(root, base commit, push commits): the base tree, then Alice's push and two bot commits."""
    root = tmp_path / "cards"
    root.mkdir()
    _git(root, "init", "-q", "-b", "main")
    work = tmp_path / "psds"
    work.mkdir()
    titled = card()
    ImageDraw.Draw(titled).rectangle((40, 70, 55, 85), fill=(0, 0, 0))
    for stem in ("Snack", "Bread", "Gone", "Lonely", "Kept", "Dropped", "Bot_Card"):
        _psd(root, work, f"{FOLDER}/{stem}.psd", card())
        _png(root, f"{FOLDER}/{stem}.png", card())
    _png(root, f"{FOLDER}/Snack2.png", titled)                       # a numbered PNG of Snack.psd
    _psd(root, work, "Creatures/Other/3 Stars/Old_Name.psd", card())
    _png(root, "Creatures/Other/3 Stars/Old_Name.png", card())
    _psd(root, work, "MDW/Template.psd", card())
    _png(root, "Types/fire.png", card(size=(8, 8)))
    base = _commit(root, "base")
    green = card(colour=(0, 255, 0))
    _psd(root, work, f"{FOLDER}/Snack.psd", green)                    # stale: PSD without its PNG
    _psd(root, work, f"{FOLDER}/Bread.psd", green)                    # pair: PSD and PNG together
    _png(root, f"{FOLDER}/Bread.png", green)
    _psd(root, work, f"{FOLDER}/New_Card.psd", green)                 # new: no PNG
    (root / "Creatures/Other/3 Stars/Old_Name.psd").rename(root / "Creatures/Other/3 Stars/New_Name.psd")   # rename
    (root / f"{FOLDER}/Gone.psd").unlink()                            # deleted: its PNG stays
    _png(root, f"{FOLDER}/Lonely.png", green)                         # PNG pushed alone, not matching its PSD
    (root / f"{FOLDER}/Kept.png").unlink()                            # PNG deleted, PSD kept
    _psd(root, work, f"{FOLDER}/Dropped.psd", green)                  # PSD pushed, its PNG deleted: no PNG made
    (root / f"{FOLDER}/Dropped.png").unlink()
    _psd(root, work, "MDW/Template.psd", green)                       # not a card
    alice = _commit(root, "Alice's cards", author="Alice")
    _psd(root, work, f"{FOLDER}/Bot_Card.psd", green)
    bot = _commit(root, "Card edits: 1 baked", author="AutoCroissant")
    return root, base, alice, bot


# ========================
# The push range and the jobs
# ========================
def test_jobs_cover_every_case(cards):
    root, base, alice, bot = cards
    source = cg.Source(str(root))
    jobs, skipped, notes, tree = cg.plan_jobs(source, base, bot, "actor")
    got = {(Path(j.psd).name, j.case, Path(j.png).name if j.png else "") for j in jobs}
    assert got == {("Snack.psd", "stale", "Snack.png"), ("Bread.psd", "pair", "Bread.png"),
                   ("New_Card.psd", "new", "New_Card.png"), ("New_Name.psd", "rename", "New_Name.png"),
                   ("Gone.psd", "deleted", "Gone.png"), ("Lonely.psd", "png-only", "Lonely.png"),
                   ("Kept.psd", "png-deleted", "Kept.png"), ("Dropped.psd", "psd-only", "")}
    assert set(skipped) == {f"{FOLDER}/Bot_Card.psd"} and notes == [f"`{FOLDER}/Dropped.psd` was pushed with its PNG deleted: no PNG is made"]
    snack = next(j for j in jobs if j.case == "stale")
    assert snack.numbered == {f"{FOLDER}/Snack2.png": tree[f"{FOLDER}/Snack2.png"][1]} and snack.login == "Alice"
    assert snack.commit == alice and snack.psd_old and snack.psd_new and snack.png_old == tree[f"{FOLDER}/Snack.png"][1]
    rename = next(j for j in jobs if j.case == "rename")
    assert rename.copy_from == source.tree(base)["Creatures/Other/3 Stars/Old_Name.png"][1]
    assert rename.left == "Creatures/Other/3 Stars/Old_Name.png"
    only = cg.plan_jobs(source, base, bot, "actor", {"snack"})[0]
    assert [j.case for j in only] == ["stale"]


def test_push_range_resolves_before(cards):
    root, base, alice, bot = cards
    source = cg.Source(str(root))
    assert cg.push_range(source, base, bot) == (base, bot, [])
    assert cg.push_range(source, "", bot)[0] == alice                     # a dispatch: the first parent
    assert cg.push_range(source, cg.ZEROS, alice)[:2] == (base, alice)
    _, _, notes = cg.push_range(source, "f" * 40, bot)
    assert cg.push_range(source, "f" * 40, bot)[0] == alice and "lacks" in notes[0]
    assert cg.push_range(source, bot, alice)[0] == base                   # not an ancestor: a force push
    assert cg.push_range(source, "", base)[0] == cg.EMPTY_TREE            # a root commit
    with pytest.raises(gc.Transient):
        cg.push_range(source, base, "e" * 40)


def test_bot_writers():
    assert cg.Writer("x", "AutoCroissant", "a@b", "anything").bot
    assert cg.Writer("x", "Someone", gc.AUTHOR["email"], "anything").bot
    assert cg.Writer("x", "github-actions[bot]", cg.BOT_EMAILS[0], "anything").bot
    for subject in ("Card guards: 1 PNG", "Card edits: 2 baked", "Card edit: Foo", "TTS save: 3 cards", "Card sync: x"):
        assert not cg.Writer("x", "Alice", "a@b", subject).bot          # a person's commit named like a bot's is guarded


def _plumb(root, files, message, author="Alice"):
    """A commit of {path: bytes} on top of HEAD through the index (paths that differ only in case, on any disk)."""
    for path, data in files.items():
        blob = subprocess.run(["git", "-C", str(root), "hash-object", "-w", "--stdin"], input=data, capture_output=True,
                              check=True).stdout.decode().strip()
        _git(root, "update-index", "--add", "--cacheinfo", f"100644,{blob},{path}", author=author)
    _git(root, "commit", "-q", "-m", message, author=author)
    return _git(root, "rev-parse", "HEAD")


def test_a_path_a_person_changed_is_guarded_after_a_bot_commit(cards):
    root, base, alice, bot = cards
    psd = (root / FOLDER / "Bread.psd").read_bytes()
    human = _plumb(root, {f"{FOLDER}/Bot_Card.psd": psd + b"\0"}, "Card edits: by hand", author="Bob")
    later = _plumb(root, {f"{FOLDER}/Bot_Card.psd": psd + b"\0\0"}, "Card edits: 1 baked", author="AutoCroissant")
    source = cg.Source(str(root))
    assert source.writers(alice, later)[f"{FOLDER}/Bot_Card.psd"].commit == human
    jobs, skipped, _, _ = cg.plan_jobs(source, alice, later, "actor")
    assert not skipped and [(Path(j.psd).name, j.login) for j in jobs] == [("Bot_Card.psd", "Bob")]
    assert set(cg.plan_jobs(source, human, later, "actor")[1]) == {f"{FOLDER}/Bot_Card.psd"}   # the bot's alone: left out


def test_png_pairs_with_the_psd_of_its_exact_case(cards):
    root, base, alice, bot = cards
    psd = (root / FOLDER / "Bread.psd").read_bytes()
    two = _plumb(root, {f"{FOLDER}/Dark_matter.psd": psd, f"{FOLDER}/Dark_Matter.psd": psd}, "two cases")
    png = (root / FOLDER / "Bread.png").read_bytes()
    exact = _plumb(root, {f"{FOLDER}/Dark_Matter.png": png}, "the PNG")
    source = cg.Source(str(root))
    jobs, _, notes, _ = cg.plan_jobs(source, two, exact, "actor")
    assert [(j.psd, j.case) for j in jobs] == [(f"{FOLDER}/Dark_Matter.psd", "png-only")] and notes == []
    other = _plumb(root, {f"{FOLDER}/DARK_MATTER.png": png}, "neither case")
    jobs, _, notes, _ = cg.plan_jobs(source, exact, other, "actor")
    assert jobs == [] and notes == [f"`{FOLDER}/DARK_MATTER.png` matches 2 PSDs that differ only in letter case: not checked"]


# ========================
# check, end to end
# ========================
def test_check_end_to_end(cards, tmp_path):
    root, base, alice, bot = cards
    out = tmp_path / "out"
    data = cg.check(str(root), bot, base, "o/cards", "actor", mode="report", out=str(out), workers=1)
    by = {Path(c["psd"]).stem: c for c in data["cards"]}
    snack = by["Snack"]
    assert snack["export"]["outcome"] == "re-exported" and snack["export"]["writer"] == "tool"
    assert [w["path"] for w in snack["writes"]] == [f"{FOLDER}/Snack.png", f"{FOLDER}/Snack2.png"]
    for write in snack["writes"]:
        assert blob_sha((out / write["file"]).read_bytes()) == write["sha"]
    assert snack["expect"] == {f"{FOLDER}/Snack.psd": cg.Source(str(root)).tree(bot)[f"{FOLDER}/Snack.psd"][1],
                               f"{FOLDER}/Snack.png": cg.Source(str(root)).tree(bot)[f"{FOLDER}/Snack.png"][1],
                               f"{FOLDER}/Snack2.png": cg.Source(str(root)).tree(bot)[f"{FOLDER}/Snack2.png"][1]}
    assert by["New_Card"]["export"]["outcome"] == "re-exported" and by["New_Card"]["writes"][0]["path"] == f"{FOLDER}/New_Card.png"
    assert by["New_Name"]["writes"][0]["sha"] == cg.Source(str(root)).tree(base)["Creatures/Other/3 Stars/Old_Name.png"][1]
    assert by["Bread"]["r10"]["ok"] is True and not by["Bread"]["writes"]
    assert by["Lonely"]["r10"]["ok"] is False and by["Lonely"]["told"] == []          # R10: the comment only (TELL_PNG_MISMATCH)
    assert by["Gone"]["told"] == [] and by["Gone"]["notes"] == ["Gone.png is left without its PSD"]     # comment only
    assert by["New_Name"]["told"] == [] and by["New_Name"]["notes"] == ["Old_Name.png is left at the PSD's old name"]
    assert by["Kept"]["told"] == [] and by["Kept"]["notes"] == ["Kept.png was deleted; the PSD stays"]
    assert (out / "crops" / FOLDER / "Snack.diff.png").exists()
    summary = json.loads((out / "summary.json").read_text())
    assert summary == {"after": bot, "cards": 8, "told": summary["told"], "comment": 8, "writes": 4, "errors": 0,
                       "card_errors": 0} and summary["told"] >= 1
    assert not by["Dropped"]["writes"] and not by["Dropped"]["export"]
    report = (out / "report.md").read_text()
    assert report.startswith("## Card guards") and "Report only (CARD_GUARDS=report)" in report and "Bot_Card" in report
    payload = json.loads((out / "discord.json").read_text())
    assert payload["allowed_mentions"] == {"parse": []} and payload["flags"] == 4 and "<@" not in payload["content"]
    lines = payload["content"].splitlines()
    assert all(re.fullmatch(r"\S+ pushed .+ with these issues: .+\. \[[0-9a-f]{8}\]\(https://github\.com/o/cards/commit/[0-9a-f]{12}\)", l)
               for l in lines), lines
    assert "Lonely.png does not match its PSD" in report and not any("Lonely" in l for l in lines)
    assert not any("Snack" in l and "re-exported" in l for l in lines)        # writes are not issues


def test_a_new_psd_pushed_with_its_png_gets_the_new_card_rules(cards, tmp_path):
    """A PSD added with its PNG (case pair, status A) is a new card: R4 is told, not waived, as for one pushed alone."""
    root, base, alice, bot = cards
    psd, png = (root / FOLDER / "Bread.psd").read_bytes(), (root / FOLDER / "Bread.png").read_bytes()
    push = _plumb(root, {f"{FOLDER}/Fresh.psd": psd, f"{FOLDER}/Fresh.png": png, f"{FOLDER}/Alone.psd": psd}, "two new cards")
    data = cg.check(str(root), push, bot, mode="report", out=str(tmp_path / "out"), workers=1)
    by = {c["card"]: c for c in data["cards"]}
    assert by["Fresh"]["case"] == "pair" and by["Alone"]["case"] == "new"
    for name in ("Fresh", "Alone"):
        (r4,) = [i for i in by[name]["issues"] if i["code"] == "R4"]
        assert not r4["waived"] and r4["told"] and r4["line"] in by[name]["told"], by[name]
    assert cg.added(cg.Job("X.psd", "psd-only")) and not cg.added(cg.Job("X.psd", "pair", psd_old="s"))
    assert not cg.added(cg.Job("X.psd", "stale", psd_old="s")) and not cg.added(cg.Job("X.psd", "whole"))


def _dated(root, work, stem, image, when):
    """Alice's commit of FOLDER/<stem>.psd from image, dated when (committer and author)."""
    _psd(root, work, f"{FOLDER}/{stem}.psd", image)
    env = {**os.environ, "GIT_AUTHOR_NAME": "Alice", "GIT_AUTHOR_EMAIL": "alice@example.com", "GIT_COMMITTER_NAME": "Alice",
           "GIT_COMMITTER_EMAIL": "alice@example.com", "GIT_AUTHOR_DATE": when, "GIT_COMMITTER_DATE": when}
    subprocess.run(["git", "-C", str(root), "commit", "-q", "-am", f"{stem} at {when}"], check=True, env=env)
    return _git(root, "rev-parse", "HEAD")


@pytest.mark.parametrize("first, made", [("2026-10-22T10:00:00Z", True), ("2026-10-22T06:00:00Z", False)])
def test_a_burst_of_psd_pushes_still_gets_its_png(cards, tmp_path, first, made):
    """Two PSD pushes 10 minutes apart, no PNG: the first run is superseded, so the second checks against the PSD as of
    the PNG's last change (the base): outside both changes the PNG still matches, and the PNG is made. Four hours apart,
    no anchor: G5 refuses as before."""
    root, base, alice, bot = cards
    work = tmp_path / "burst"
    work.mkdir()
    one = _dated(root, work, "Bread", card(base=(30, 200, 30)), first)             # the Base layer: everywhere
    two = _dated(root, work, "Bread", card(base=(30, 200, 30), colour=(0, 255, 0)), "2026-10-22T10:10:00Z")   # the square
    source = cg.Source(str(root))
    (job,) = cg.plan_jobs(source, one, two, "actor")[0]
    assert job.case == "stale" and bool(job.psd_anchor) is made
    if made:
        assert job.psd_anchor == source.tree(alice)[f"{FOLDER}/Bread.psd"][1] and job.anchor_commit == alice
    data = cg.check(str(root), two, one, mode="on", out=str(tmp_path / "out"), workers=1)
    (entry,) = data["cards"]
    if made:
        assert entry["export"]["outcome"] == "re-exported" and entry["writes"][0]["path"] == f"{FOLDER}/Bread.png"
        assert any("an earlier push changed the PSD without its PNG" in note for note in entry["notes"])
        assert not any(line.startswith("PNG not made") for line in entry["told"])
    else:
        assert entry["export"]["gate"] == "G5" and entry["told"][0].startswith("PNG not made: the PSD and its old PNG already")


def test_png_mismatch_reaches_discord_only_when_switched_on(cards, tmp_path, monkeypatch):
    root, base, alice, bot = cards
    monkeypatch.setattr(cg, "TELL_PNG_MISMATCH", True)
    data = cg.check(str(root), bot, base, "o/cards", mode="report", out=str(tmp_path / "out"), workers=1, cards={"lonely"})
    assert "Lonely.png does not match its PSD" in data["discord"]["content"]


def test_a_png_mismatch_alone_still_gets_its_comment(cards, tmp_path):
    """R10 is comment-only: a PNG-only push whose PNG does not match its PSD tells nothing on Discord, yet summary.json
    asks for the comment (card-guards.yml's Comment step gates on "comment")."""
    root, base, alice, bot = cards
    out = tmp_path / "out"
    data = cg.check(str(root), bot, base, "o/cards", mode="report", out=str(out), workers=1, cards={"lonely"})
    (lonely,) = data["cards"]
    assert lonely["case"] == "png-only" and lonely["r10"]["ok"] is False and lonely["told"] == []
    summary = json.loads((out / "summary.json").read_text())
    assert summary["told"] == 0 and summary["writes"] == 0 and summary["comment"] == 1
    assert "Lonely.png does not match its PSD" in (out / "report.md").read_text() and data["discord"] == {}


def test_a_card_error_is_told_and_counted(cards, tmp_path, monkeypatch):
    """A card whose download or composite crashed: its error is told on Discord (a stale or new PSD: "PNG not made"),
    the others go on, and summary.json's card_errors turns the run red."""
    root, base, alice, bot = cards
    real = cg.png_export.export_new

    def crash(*args, **kwargs):
        raise MemoryError("composite")
    monkeypatch.setattr(cg.png_export, "export_new", crash)
    out = tmp_path / "out"
    data = cg.check(str(root), bot, base, "o/cards", mode="on", out=str(out), workers=1, cards={"new card", "snack"})
    by = {c["card"]: c for c in data["cards"]}
    assert by["New Card"]["error"] == "MemoryError: composite" and not by["New Card"]["writes"]
    assert "the bot could not finish checking it (MemoryError); PNG not made: export it in Photoshop" in by["New Card"]["told"]
    assert by["Snack"]["writes"] and not by["Snack"]["error"]
    summary = json.loads((out / "summary.json").read_text())
    assert summary["card_errors"] == 1 and summary["errors"] == 1 and summary["told"] >= 1
    assert "New Card with these issues: the bot could not finish checking it" in data["discord"]["content"]
    result = cg.Result(cg.Job("X/A.psd", "pair"), error="KeyError: 'x'")
    assert result.told() == ["the bot could not finish checking it (KeyError)"]
    monkeypatch.setattr(cg.png_export, "export_new", real)


def test_report_mode_tells_the_fixable_issues():
    """Nothing is fixed in report mode, so the fixable issues are remaining issues there; in on mode the bot fixes them."""
    issues = [{"line": "the Name layer is named 'NAME'", "told": False, "discord": True, "fix": True, "waived": ""},
              {"line": "waived", "told": False, "discord": True, "fix": True, "waived": "owner"},
              {"line": "R7 fonts", "told": True, "discord": False, "fix": False, "waived": ""}]
    result = cg.Result(cg.Job("X/A.psd", "pair"), issues=issues)
    assert result.told("report") == ["the Name layer is named 'NAME'"]
    assert result.told("on") == [] and result.told("dry") == []


def test_check_fails_when_it_cannot_check(cards, tmp_path, capsys):
    """Not checked (the clone lacks the commit, the API is down): exit 1, so card-guards.yml's failure alert runs."""
    root, base, alice, bot = cards
    assert cg.main(["check", "--clone", str(root), "--after", "e" * 40, "--out", str(tmp_path / "out")]) == 1
    assert "Not checked: the clone has no commit eeeeeeee" in capsys.readouterr().out


def test_whole_tree(cards, tmp_path):
    root, base, alice, bot = cards
    data = cg.check(str(root), bot, whole_tree=True, out=str(tmp_path / "out"), workers=1)
    assert {Path(c["psd"]).stem for c in data["cards"]} == {"Snack", "Bread", "Lonely", "Kept", "Dropped", "Bot_Card", "New_Card",
                                                            "New_Name"}
    assert all(c["case"] == "whole" and not c["writes"] for c in data["cards"])
    assert {Path(c["psd"]).stem for c in data["cards"] if c["r10"].get("ok") is False} == {"Snack", "Lonely", "Bot_Card"}


def test_discord_fits_and_says_nothing_without_issues():
    entry = {"login": "Zomoroko", "card": "Skullord", "told": ["the Ability layer is outside the Text group"], "commit": "7" * 40}
    data = {"after": "a" * 40, "cards": [dict(entry, card=f"Card {i}") for i in range(80)]}
    payload = cg.discord_payload(data, "MichaelJSr/TTSCardMaker")
    assert len(payload["content"]) <= cg.DISCORD_LIMIT and payload["flags"] == 4
    last = payload["content"].splitlines()[-1]
    assert re.fullmatch(r"\.\.\. and \d+ more cards: \[details\]\(https://github\.com/MichaelJSr/TTSCardMaker/commit/a{40}\)", last)
    one = cg.discord_payload({"after": "a" * 40, "cards": [entry]}, "MichaelJSr/TTSCardMaker")["content"]
    assert one == ("Zomoroko pushed Skullord with these issues: the Ability layer is outside the Text group. "
                   "[77777777](https://github.com/MichaelJSr/TTSCardMaker/commit/" + "7" * 12 + ")")
    assert cg.discord_payload({"after": "a" * 40, "cards": [dict(entry, told=[])]}, "o/r") == {}


def test_message():
    landed = [{"login": "Zomoroko", "commit": "7cb4c867" + "0" * 32,
               "writes": [{"path": "Items/X.png", "kind": "png", "line": "re-exported from the PSD's composite (12 px changed)"},
                          {"path": "Items/Y.psd", "kind": "psd", "line": "Txt2 synced"}]}]
    text = cg.message(landed, "f" * 40)
    assert text.splitlines()[0] == f"Card guards: 1 PNG re-exported, 1 PSD fixed (push {'f' * 8})"
    assert "Items/X.png: re-exported from the PSD's composite (12 px changed) (Zomoroko, 7cb4c867)" in text
    assert text.rstrip().endswith(f"Guarded-push: {'f' * 40}") and "Edit-by" not in text


# ========================
# land
# ========================
class FakeHub:
    """snapshot_at / ref_head / recent_paths / commit_files for land."""
    def __init__(self, tree, heads=("h1",), recent=()):
        self.tree, self.heads, self.recent, self.commits, self.calls = dict(tree), list(heads), set(recent), [], 0

    def install(self, monkeypatch):
        monkeypatch.setattr(cg, "ref_head", lambda repo: self.heads[min(self.calls, len(self.heads) - 1)])
        monkeypatch.setattr(cg, "snapshot_at", lambda repo, head: cg.Snapshot(repo, head, {"tree": {"sha": "t" + head}}, self.tree))
        monkeypatch.setattr(cg, "recent_paths", lambda repo, head, now, local="": self.recent)
        monkeypatch.setattr(cg, "commit_files", self.commit)

    def commit(self, repo, head, base_tree, modes, files, message, uploaded=None):
        self.calls += 1
        if self.calls < len(self.heads):
            return None                                 # the head moved
        self.commits.append((head, sorted(files), message))
        return "c" * 40


def _plan(tmp_path, entries):
    out = tmp_path / "plan"
    (out / "files").mkdir(parents=True)
    cards = []
    for name, path, data, expect in entries:
        (out / "files" / name).write_bytes(data)
        cards.append({"card": name, "psd": path[:-4] + ".psd", "case": "stale", "png": path, "login": "Alice",
                      "commit": "a" * 40, "expect": expect, "told": [], "export": {"outcome": "re-exported"},
                      "writes": [{"path": path, "file": f"files/{name}", "sha": blob_sha(data), "kind": "png", "line": "made"}]})
    (out / "plan.json").write_text(json.dumps({"repo": "o/r", "after": "f" * 40, "cards": cards}))
    return str(out / "plan.json")


def test_land_commits_what_still_holds(tmp_path, monkeypatch):
    plan = _plan(tmp_path, [("A", "X/A.png", b"a", {"X/A.psd": "s1", "X/A.png": "p1"}),
                            ("B", "X/B.png", b"b", {"X/B.psd": "s2", "X/B.png": "p2"}),
                            ("C", "X/C.png", b"c", {"X/C.psd": "s3", "X/C.png": ""})])
    hub = FakeHub({"X/A.psd": ("100644", "s1"), "X/A.png": ("100644", "p1"), "X/B.psd": ("100644", "moved"),
                   "X/B.png": ("100644", "p2"), "X/C.psd": ("100644", "s3")}, heads=("h1", "h2"), recent={"X/C.psd"})
    hub.install(monkeypatch)
    now = datetime(2026, 10, 22, 18, 0, tzinfo=timezone.utc)
    report = cg.land(plan, quiet_from="2026-10-22T17:00:00Z", now=lambda: now)          # a push run: later pushes count
    assert report["sha"] == "c" * 40 and report["landed"] == ["A"]
    assert report["dropped"] == [["B", "superseded: X/B.psd changed on the branch", "X/B.png"],
                                 ["C", "superseded: a later push touched X/C.psd", "X/C.png"]]
    assert report["told"] == {"X/A.psd": [], "X/B.psd": [], "X/C.psd": ["PNG not made: a later push touched X/C.psd; export it in Photoshop"]}
    assert report["after"]["X/B.psd"] == ["the PSD changed again on the branch: that push's own check tells"]
    assert len(hub.commits) == 1 and hub.commits[0][:2] == ("h2", ["X/A.png"])            # retried on the new head
    assert hub.commits[0][2].startswith("Card guards: 1 PNG re-exported (push ffffffff)")


def test_land_waits_for_the_quiet_window(tmp_path, monkeypatch):
    plan = _plan(tmp_path, [("A", "X/A.png", b"a", {"X/A.psd": "s1"})])
    FakeHub({"X/A.psd": ("100644", "s1")}).install(monkeypatch)
    now = datetime(2026, 10, 22, 18, 0, tzinfo=timezone.utc)
    waits = []
    report = cg.land(plan, quiet_from="2026-10-22T17:50:00Z", dry_run=True, sleep=waits.append, now=lambda: now)
    assert waits == [(timedelta(minutes=20) + cg.QUIET_MARGIN).total_seconds()] and report["landed"] == ["A"] and report["sha"] == ""


def test_land_refuses_a_tampered_file(tmp_path, monkeypatch):
    plan = _plan(tmp_path, [("A", "X/A.png", b"a", {"X/A.psd": "s1"})])
    (Path(plan).parent / "files" / "A").write_bytes(b"changed")
    hub = FakeHub({"X/A.psd": ("100644", "s1")})
    hub.install(monkeypatch)
    report = cg.land(plan)
    assert "does not hash" in report["error"] and report["sha"] == "" and not hub.commits
    assert report["told"]["X/A.psd"] == ["PNG not made: the bot's commit failed; export it in Photoshop"]


def _entry(name, psd_blob="s1", png_blob="p1", **extra):
    """A plan card with every key markdown reads (no writes)."""
    entry = {"card": name, "psd": f"X/{name}.psd", "case": "stale", "png": f"X/{name}.png", "login": "Alice",
             "commit": "a" * 40, "export": {}, "r10": {}, "issues": [], "fixed": [], "fix_refused": [], "notes": [],
             "told": [], "error": "", "writes": [], "expect": {f"X/{name}.psd": psd_blob, f"X/{name}.png": png_blob}}
    entry.update(extra)
    return entry


def _write(out, entry, path, kind, data, line):
    (out / "files").mkdir(parents=True, exist_ok=True)
    (out / "files" / Path(path).name).write_bytes(data)
    entry["writes"].append({"path": path, "file": f"files/{Path(path).name}", "sha": blob_sha(data), "kind": kind, "line": line})


def _full_plan(tmp_path, entries, mode="on"):
    out = tmp_path / "plan"
    out.mkdir(exist_ok=True)
    (out / "plan.json").write_text(json.dumps({"repo": "o/r", "mode": mode, "before": "e" * 40, "after": "f" * 40,
                                               "report": {"notes": [], "skipped": [], "errors": []}, "cards": entries}))
    return str(out / "plan.json")


def test_a_failed_commit_is_told_and_commented(tmp_path, monkeypatch):
    """land's Transient: exit 1, but the comment's new body and the one post still say what was not made."""
    out = tmp_path / "plan"
    snack = _entry("Snack", export={"outcome": "re-exported", "changed_px": 5, "numbers": {}})
    _write(out, snack, "X/Snack.png", "png", b"png", "re-exported")
    bread = _entry("Bread", fixed=["the Name layer is named 'NAME'"])
    _write(out, bread, "X/Bread.psd", "psd", b"psd", "the Name layer is named 'NAME'")
    plan = _full_plan(tmp_path, [snack, bread])
    FakeHub({"X/Snack.psd": ("100644", "s1"), "X/Snack.png": ("100644", "p1"), "X/Bread.psd": ("100644", "s1"),
             "X/Bread.png": ("100644", "p1")}).install(monkeypatch)

    def down(*args, **kwargs):
        raise gc.Transient("the API is down")
    monkeypatch.setattr(cg, "commit_files", down)
    assert cg.main(["land", "--plan", plan]) == 1
    landed = json.loads((out / "landed.json").read_text())
    assert landed["error"] == "the API is down" and landed["sha"] == "" and landed["landed"] == []
    comment = (out / "report-landed.md").read_text()
    assert "Not landed: the API is down. Nothing was committed." in comment and "The bot will fix" not in comment
    assert "- Not fixed: the Name layer is named 'NAME'" in comment
    assert comment.count("Not landed (the bot's commit failed): nothing above was committed for this card") == 2
    post = json.loads((out / "discord-final.json").read_text())["content"].splitlines()
    assert post[0].startswith("Alice pushed Snack with these issues: PNG not made: the bot's commit failed; export it in Photoshop.")
    assert post[1].startswith("Alice pushed Bread with these issues: the Name layer is named 'NAME' (the bot could not fix it: "
                              "the bot's commit failed).")


def test_the_post_waits_for_the_quiet_window(tmp_path, monkeypatch):
    """A PNG refusal is withdrawn when the author pushed the PNG during the window; a card whose PSD changed again is left
    to that push's check; the other lines stand (report mode: nothing committed, the same post)."""
    refused = "PNG not made: the PSD and its old PNG already disagreed outside your change (3.0% of pixels); export it in Photoshop"
    snack = _entry("Snack", export={"outcome": "refused", "why": refused, "gate": "G5"}, told=[refused])
    bread = _entry("Bread", told=["the Ability layer is outside the Text group"])
    kept = _entry("Kept", told=["the Ability layer is outside the Text group"])
    plan = _full_plan(tmp_path, [snack, bread, kept], mode="report")
    FakeHub({"X/Snack.psd": ("100644", "s1"), "X/Snack.png": ("100644", "pushed"), "X/Bread.psd": ("100644", "again"),
             "X/Bread.png": ("100644", "p1"), "X/Kept.psd": ("100644", "s1"), "X/Kept.png": ("100644", "p1")}).install(monkeypatch)
    now = datetime(2026, 10, 22, 18, 0, tzinfo=timezone.utc)
    waits = []
    report = cg.land(plan, quiet_from="2026-10-22T17:59:00Z", dry_run=True, sleep=waits.append, now=lambda: now)
    assert waits and report["told"] == {"X/Snack.psd": [], "X/Bread.psd": [], "X/Kept.psd": ["the Ability layer is outside the Text group"]}
    post = cg.finish(plan, report)["content"]
    assert post.startswith("Alice pushed Kept with these issues:") and len(post.splitlines()) == 1
    comment = (Path(plan).parent / "report-landed.md").read_text()
    assert "Report only (CARD_GUARDS=report)" in comment and "Landed" not in comment
    assert "- After the quiet window: Snack.png was pushed since: the refusal is withdrawn" in comment
    assert "- After the quiet window: the PSD changed again on the branch" in comment
    assert cg.land(_full_plan(tmp_path, [_entry("Quiet")]), quiet_from="2026-10-22T17:59:00Z", sleep=waits.append,
                   now=lambda: now)["told"] == {}                          # nothing to tell or write: no wait
    assert len(waits) == 1


def test_a_psd_fix_survives_a_later_png_push(tmp_path, monkeypatch):
    """A PSD fix rests on its PSD alone (0 pixels): the author's later PNG push leaves it to land; the card's PNG write is
    dropped silently there (the author replaced it). Partial landings are named in the comment."""
    out = tmp_path / "plan"
    snack = _entry("Snack", fixed=["the Name layer is named 'NAME'"], export={"outcome": "re-exported", "changed_px": 5, "numbers": {}})
    _write(out, snack, "X/Snack.png", "png", b"png", "re-exported")
    _write(out, snack, "X/Snack.psd", "psd", b"psd", "the Name layer is named 'NAME'")
    plan = _full_plan(tmp_path, [snack])
    hub = FakeHub({"X/Snack.psd": ("100644", "s1"), "X/Snack.png": ("100644", "author's")})
    hub.install(monkeypatch)
    report = cg.land(plan)
    assert report["sha"] and report["landed"] == ["Snack"] and hub.commits[0][1] == ["X/Snack.psd"]
    assert report["dropped"] == [["Snack", "superseded: X/Snack.png changed on the branch", "X/Snack.png"]]
    assert report["told"] == {"X/Snack.psd": []}
    comment = cg.landed_markdown(json.loads(Path(plan).read_text()), report)
    assert "- Fixed by the bot: the Name layer" in comment and "- Not landed (superseded: X/Snack.png changed on the branch): `X/Snack.png`" in comment
    assert "- The rest landed in `cccccccc`" in comment


def test_land_never_raises_and_still_settles_the_post(tmp_path, monkeypatch):
    """A requests error (a POST, or a GET past its retries) is the error like a Transient: landed.json, the comment's
    new body and the settled post are written, exit 1, and the post tells the PNG and the fix that were not made."""
    import requests
    out = tmp_path / "plan"
    snack = _entry("Snack", export={"outcome": "re-exported", "changed_px": 5, "numbers": {}})
    _write(out, snack, "X/Snack.png", "png", b"png", "re-exported")
    bread = _entry("Bread", fixed=["the Name layer is named 'NAME'"])
    _write(out, bread, "X/Bread.psd", "psd", b"psd", "the Name layer is named 'NAME'")
    plan = _full_plan(tmp_path, [snack, bread])
    for where in ("ref_head", "commit_files"):
        FakeHub({"X/Snack.psd": ("100644", "s1"), "X/Snack.png": ("100644", "p1"), "X/Bread.psd": ("100644", "s1"),
                 "X/Bread.png": ("100644", "p1")}).install(monkeypatch)

        def down(*args, **kwargs):
            raise requests.ConnectionError("connection reset")
        monkeypatch.setattr(cg, where, down)
        for name in ("landed.json", "report-landed.md", "discord-final.json"):
            (out / name).unlink(missing_ok=True)
        assert cg.main(["land", "--plan", plan]) == 1
        landed = json.loads((out / "landed.json").read_text())
        assert landed["error"] == "ConnectionError: connection reset" and landed["sha"] == "", where
        assert "Not landed: ConnectionError: connection reset. Nothing was committed." in (out / "report-landed.md").read_text()
        post = json.loads((out / "discord-final.json").read_text())["content"]
        assert "PNG not made: the bot's commit failed" in post and "(the bot could not fix it: the bot's commit failed)" in post


def test_a_dispatch_does_not_count_its_own_push_as_a_later_one(tmp_path, monkeypatch):
    """No quiet_from (a dispatch with dry_run false): the last 30 minutes' pushes include the checked push itself, so only
    the blob check decides; recent_paths is not consulted."""
    plan = _plan(tmp_path, [("A", "X/A.png", b"a", {"X/A.psd": "s1", "X/A.png": "p1"})])
    hub = FakeHub({"X/A.psd": ("100644", "s1"), "X/A.png": ("100644", "p1")}, recent={"X/A.psd", "X/A.png"})
    hub.install(monkeypatch)
    asked = []
    monkeypatch.setattr(cg, "recent_paths", lambda *args, **kwargs: asked.append(args) or {"X/A.psd"})
    report = cg.land(plan)
    assert report["landed"] == ["A"] and report["dropped"] == [] and report["sha"] and not asked


def test_a_rerun_after_a_landing_says_landed_not_superseded(tmp_path, monkeypatch):
    """The first attempt landed (the branch holds the bot's PNG and fixed PSD): the re-run writes nothing, drops nothing,
    tells nothing, and the comment keeps saying the writes are on the branch."""
    out = tmp_path / "plan"
    snack = _entry("Snack", fixed=["the Name layer is named 'NAME'"], export={"outcome": "re-exported", "changed_px": 5, "numbers": {}},
                   told=["the Ability layer is outside the Text group"])
    _write(out, snack, "X/Snack.png", "png", b"png", "re-exported")
    _write(out, snack, "X/Snack.psd", "psd", b"psd", "the Name layer is named 'NAME'")
    plan = _full_plan(tmp_path, [snack])
    hub = FakeHub({"X/Snack.psd": ("100644", blob_sha(b"psd")), "X/Snack.png": ("100644", blob_sha(b"png"))})
    hub.install(monkeypatch)
    report = cg.land(plan)
    assert report["sha"] == "" and report["error"] == "" and report["dropped"] == [] and not hub.commits
    assert sorted(report["already"]) == [["Snack", "X/Snack.png"], ["Snack", "X/Snack.psd"]]
    assert report["told"] == {"X/Snack.psd": ["the Ability layer is outside the Text group"]}      # the PSD did not "change again"
    comment = cg.landed_markdown(json.loads(Path(plan).read_text()), report)
    assert "Nothing new to land: 1 card already on the branch" in comment and "Fixed by the bot" in comment
    assert "Every write is already on the branch" in comment and "Not landed" not in comment and "Not fixed" not in comment


def test_a_bot_change_in_the_window_keeps_the_lines(tmp_path, monkeypatch):
    """A PSD a bot commit changed during the window (no run follows it): its lines are still told, and its fix that
    could not land is told too. A person's change leaves the card to that push's run."""
    out = tmp_path / "plan"
    snack = _entry("Snack", told=["the Ability layer is outside the Text group"], fixed=["the Name layer is named 'NAME'"])
    _write(out, snack, "X/Snack.psd", "psd", b"psd", "the Name layer is named 'NAME'")
    plan = _full_plan(tmp_path, [snack])
    FakeHub({"X/Snack.psd": ("100644", "baked"), "X/Snack.png": ("100644", "p1")}).install(monkeypatch)
    for author, told in ((("AutoCroissant", gc.AUTHOR["email"]), ["the Ability layer is outside the Text group",
                                                                    "the Name layer is named 'NAME' (the bot could not fix it: "
                                                                    "X/Snack.psd changed on the branch)"]),
                         (("Bob", "bob@example.com"), [])):
        def commits(repo, method, path, **kwargs):
            assert path == "commits" and kwargs["params"]["path"] == "X/Snack.psd"
            return type("R", (), {"json": lambda self: [{"commit": {"author": {"name": author[0], "email": author[1]}}}]})()
        monkeypatch.setattr(cg, "api", commits)
        report = cg.land(plan)
        assert report["told"] == {"X/Snack.psd": told}, author


def test_bot_changed_in_a_clone(cards):
    root, base, alice, bot = cards
    assert cg._bot_changed("", str(root), bot, f"{FOLDER}/Bot_Card.psd")
    assert not cg._bot_changed("", str(root), bot, f"{FOLDER}/Snack.psd")
    assert not cg._bot_changed("", str(root), bot, "no/such.psd")


def test_a_png_pushed_in_another_case_counts(tmp_path, monkeypatch):
    """A new Foo.psd's PNG pushed as foo.png during the window: the bot adds no Foo.png beside it, and the refusal of
    another card whose PNG came back in another case is withdrawn."""
    out = tmp_path / "plan"
    foo = _entry("Foo", case="new", png="X/Foo.png", export={"outcome": "re-exported", "changed_px": 5, "numbers": {}})
    foo["expect"] = {"X/Foo.psd": "s1", "X/Foo.png": ""}
    _write(out, foo, "X/Foo.png", "png", b"png", "made")
    refused = "PNG not made: the card changed size"
    bar = _entry("Bar", export={"outcome": "refused", "why": refused}, told=[refused])
    plan = _full_plan(tmp_path, [foo, bar])
    hub = FakeHub({"X/Foo.psd": ("100644", "s1"), "X/foo.png": ("100644", "theirs"), "X/Bar.psd": ("100644", "s1"),
                   "X/bar.png": ("100644", "new")})
    hub.install(monkeypatch)
    report = cg.land(plan)
    assert not hub.commits and report["dropped"] == [["Foo", "superseded: X/Foo.png changed on the branch", "X/Foo.png"]]
    assert report["told"] == {"X/Foo.psd": [], "X/Bar.psd": []}
    assert report["after"]["X/Bar.psd"] == ["Bar.png was pushed since: the refusal is withdrawn"]


def test_a_numbered_png_pushed_in_the_window_withdraws_its_refusal(tmp_path, monkeypatch):
    """A numbered PNG's refusal (G8) is withdrawn as the main PNG's is when the author pushed that numbered PNG in the
    window; another numbered PNG's refusal stands. (The main PNG's write rests on the numbered PNGs too: dropped.)"""
    out = tmp_path / "plan"
    one = "PNG not made for Snack2.png: the change touches its own title; re-export it in Photoshop"
    two = "PNG not made for Snack3.png: the change touches its own title; re-export it in Photoshop"
    snack = _entry("Snack", told=[one, two], export={"outcome": "re-exported", "changed_px": 5, "numbers": {},
                                                     "numbered_refused": [[one, "X/Snack2.png"], [two, "X/Snack3.png"]]})
    snack["expect"].update({"X/Snack2.png": "n2", "X/Snack3.png": "n3"})
    _write(out, snack, "X/Snack.png", "png", b"png", "re-exported")
    plan = _full_plan(tmp_path, [snack], mode="report")
    FakeHub({"X/Snack.psd": ("100644", "s1"), "X/Snack.png": ("100644", "p1"), "X/snack2.png": ("100644", "theirs"),
             "X/Snack3.png": ("100644", "n3")}).install(monkeypatch)
    report = cg.land(plan, dry_run=True)
    assert report["told"] == {"X/Snack.psd": [two]}
    assert report["after"]["X/Snack.psd"] == ["Snack2.png was pushed since: the refusal is withdrawn"]


def test_context_errors_alone_ask_no_comment():
    """summary.json's comment counts cards only: a context error (a standing MDW/Fonts index error, a manifest or status
    chart fetch) is in the job summary, and in the comment when there is one, but puts no comment on every push."""
    quiet = _entry("Quiet")
    broken = _entry("Broken", error="KeyError: 'x'")
    errors = ["MDW/Fonts index: Foo.ttf unreadable", "site/stoogedings.json: timeout"]
    assert cg.summary({"after": "f" * 40, "cards": [quiet]}, errors)["comment"] == 0
    assert cg.summary({"after": "f" * 40, "cards": []}, errors) == {"after": "f" * 40, "cards": 0, "told": 0, "comment": 0,
                                                                     "writes": 0, "errors": 2, "card_errors": 0}
    counted = cg.summary({"after": "f" * 40, "cards": [quiet, broken]}, errors + ["X/Broken.psd: KeyError: 'x'"])
    assert counted["comment"] == 1 and counted["card_errors"] == 1 and counted["errors"] == 3


def test_login_falls_back_to_a_github_login(monkeypatch):
    """No GitHub account on the commit: the login in a noreply email, else the pusher's login (github.actor), never the
    git author name while the API is there; a local run without a token names the git author."""
    monkeypatch.setattr(cg, "token", lambda: "t")
    monkeypatch.setattr(cg, "api", lambda *args, **kwargs: type("R", (), {"json": lambda self: {"author": None}})())
    source = cg.Source("", "o/cards")
    assert source.login(cg.Writer("c1", "Zom", "123+Zomoroko@users.noreply.github.com", ""), "MichaelJSr") == "Zomoroko"
    assert source.login(cg.Writer("c2", "Laptop Name", "me@example.com", ""), "MichaelJSr") == "MichaelJSr"
    # a dispatch passes no pusher (whoever ran it did not push the card): the git author name
    assert source.login(cg.Writer("c4", "Laptop Name", "me@example.com", ""), cg.NOBODY) == "Laptop Name"
    monkeypatch.setattr(cg, "token", lambda: "")
    assert cg.Source("", "o/cards").login(cg.Writer("c3", "Laptop Name", "me@example.com", ""), "MichaelJSr") == "Laptop Name"


def test_land_local_commits_in_the_clone(cards, tmp_path):
    root, base, alice, bot = cards
    out = tmp_path / "out"
    cg.check(str(root), bot, base, mode="on", out=str(out), workers=1)
    report = cg.land(str(out / "plan.json"), local=str(root))
    assert report["sha"] and report["sha"] == _git(root, "rev-parse", "HEAD")              # the checkout moved to it
    assert _git(root, "log", "-1", "--format=%an %s", report["sha"]).startswith("AutoCroissant Card guards: 4 PNGs re-exported")
    changed = set(_git(root, "show", "--name-only", "--format=", report["sha"]).splitlines())
    assert changed == {f"{FOLDER}/Snack.png", f"{FOLDER}/Snack2.png", f"{FOLDER}/New_Card.png", "Creatures/Other/3 Stars/New_Name.png"}
    again = cg.check(str(root), report["sha"], bot, mode="on", out=str(tmp_path / "again"), workers=1)
    assert again["cards"] == []                                                             # the bot's commit is never guarded


def test_a_psd_fix_lands_beside_its_pushed_png(cards, tmp_path):
    """A pair (PSD and PNG pushed together): the fix rests on the pushed PNG's blob, so it lands (it was dropped as
    "superseded" when the plan expected no PNG)."""
    root, base, alice, bot = cards
    out = tmp_path / "out"
    data = cg.check(str(root), bot, base, mode="on", out=str(out), workers=1)
    plan = json.loads((out / "plan.json").read_text())
    (bread,) = [c for c in plan["cards"] if c["case"] == "pair"]
    tree = cg.Source(str(root)).tree(bot)
    fixed = b"the fixed PSD"                                                # land only moves bytes
    (out / "fixed.psd").write_bytes(fixed)
    bread["writes"] = [{"path": bread["psd"], "file": "fixed.psd", "sha": blob_sha(fixed), "kind": "psd", "line": "R11 fixed"}]
    bread["expect"] = cg._expect(cg.Job(bread["psd"], "pair", "", tree[bread["psd"]][1], f"{FOLDER}/Bread.png",
                                        png_new=tree[f"{FOLDER}/Bread.png"][1]))
    assert bread["expect"][f"{FOLDER}/Bread.png"] == tree[f"{FOLDER}/Bread.png"][1]
    plan["cards"] = [bread]
    (out / "plan.json").write_text(json.dumps(plan))
    report = cg.land(str(out / "plan.json"), local=str(root))
    assert report["landed"] == ["Bread"] and report["dropped"] == [] and report["sha"]
    assert _git(root, "show", "--name-only", "--format=", report["sha"]).splitlines() == [f"{FOLDER}/Bread.psd"]


def test_a_real_r11_fix_lands_for_a_pair(clone, tmp_path):
    """Mini Doomer pushed with its Name layer typed "NAME" and its PNG re-saved: R11 renames it, land commits the PSD."""
    from PIL import Image
    from psd_tools import PSDImage
    import commands.psd_edit as psd_edit
    rel = "Auxiliary/Minions/Mini_Doomer"
    if not (clone / f"{rel}.psd").exists():
        pytest.skip("no Mini Doomer")
    root = tmp_path / "cards"
    (root / "Auxiliary/Minions").mkdir(parents=True)
    _git(root, "init", "-q", "-b", "main")
    shutil.copy(clone / f"{rel}.psd", root / f"{rel}.psd")
    shutil.copy(clone / f"{rel}.png", root / f"{rel}.png")
    base = _commit(root, "base")
    edited = PSDImage.open(str(clone / f"{rel}.psd"))
    psd_edit.name_layer(edited).name = "NAME"
    psd_edit.save_psd(edited, str(root / f"{rel}.psd"), update_preview=False)
    Image.open(clone / f"{rel}.png").save(root / f"{rel}.png", compress_level=1)   # same pixels, a new blob
    push = _commit(root, "Mini Doomer", author="Alice")
    out = tmp_path / "out"
    data = cg.check(str(root), push, base, mode="on", out=str(out), workers=1)
    (entry,) = data["cards"]
    assert entry["case"] == "pair" and [w["kind"] for w in entry["writes"]] == ["psd"], entry
    assert entry["expect"][f"{rel}.png"] == cg.Source(str(root)).tree(push)[f"{rel}.png"][1]
    assert "The bot will fix: " in data["markdown"]
    assert cg.main(["land", "--plan", str(out / "plan.json"), "--local", str(root)]) == 0
    landed = json.loads((out / "landed.json").read_text())
    assert landed["landed"] == ["Mini Doomer"] and landed["sha"] == _git(root, "rev-parse", "HEAD")
    assert psd_edit.name_layer(PSDImage.open(str(root / f"{rel}.psd"))).name == "Name"
    comment = (out / "report-landed.md").read_text()
    assert f"Landed in `{landed['sha'][:8]}`: 1 card." in comment and "Fixed by the bot: " in comment


def test_the_comment_after_land_names_the_cards_left_out():
    entry = {"card": "A", "psd": "X/A.psd", "case": "pair", "login": "Alice", "commit": "a" * 40, "export": {}, "r10": {},
             "issues": [], "fixed": ["R11 layer names corrected"], "fix_refused": [], "notes": [], "told": [], "error": "",
             "writes": [{"path": "X/A.psd"}]}
    data = {"mode": "on", "after": "f" * 40, "before": "e" * 40, "cards": [entry, dict(entry, card="B", psd="X/B.psd")],
            "report": {"notes": [], "skipped": [], "errors": []}}
    before = cg.markdown(data, [], [], [])
    assert before.count("- The bot will fix: R11") == 2 and "Fixed by the bot" not in before
    after = cg.landed_markdown(data, {"sha": "c" * 40, "landed": ["A"],
                                      "dropped": [["B", "superseded: X/B.psd changed on the branch", "X/B.psd"]]})
    assert "Landed in `cccccccc`: 1 card. 1 card left out: changed again on the branch." in after
    a, b = after.split("**B**")
    assert "- Fixed by the bot: R11" in a and "- Landed in `cccccccc`" in a
    assert "- Not fixed: R11" in b and "- Not landed (superseded: X/B.psd changed on the branch): nothing above was committed" in b
    none = cg.landed_markdown(data, {"sha": "", "landed": [], "dropped": [["A", "x", "X/A.psd"], ["B", "y", "X/B.psd"]]})
    assert "Nothing landed. 2 cards left out" in none and "Fixed by the bot" not in none


# ========================
# The real push of 2026-10-22 15:18 (the clone, read-only)
# ========================
def test_replay_of_the_stale_png_push(clone, tmp_path):
    from importlib.util import find_spec
    if find_spec("scipy") is None:
        pytest.skip("God Tamer's gradient fills need psd-tools[composite] (scipy), as the Action installs")
    source = cg.Source(str(clone))
    if not (source.commit("38c55bd3") and source.commit("07589cc7")):
        pytest.skip("the clone lacks the push")
    data = cg.check(str(clone), "07589cc7", "38c55bd3", out=str(tmp_path / "out"), workers=1)
    by = {c["card"]: c for c in data["cards"]}
    assert len(by) == 19
    assert {name for name, c in by.items() if c["export"].get("outcome") == "current"} == {"God Tamer", "Bubble", "12 Reactions2"}
    assert all(not c["writes"] for c in data["cards"])
    assert data["discord"]["content"].startswith("MichaelJSr pushed God Tamer with these issues: the Ability layer is outside "
                                                 "the Text group. [dc37f4ad]")
    assert len(data["discord"]["content"].splitlines()) == 1


# ========================
# The Action
# ========================
def test_action_shaped_copy(tmp_path):
    """Exactly SPARSE_FILES (card-guards.yml's list), no config.py: --help works and the closure is that list."""
    bot = tmp_path / "bot"
    for rel in cg.SPARSE_FILES:
        assert (REPO / rel).is_file(), f"{rel} is in SPARSE_FILES but missing here"
        (bot / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO / rel, bot / rel)
    for command in ("--help", "check --help", "land --help"):
        done = subprocess.run([executable, "-m", "commands.card_guards", *command.split()], cwd=bot, capture_output=True,
                              text=True, timeout=120)
        assert done.returncode == 0, done.stderr
    code = ("import sys, commands.card_guards, commands.txt2_repair; "
            "print(sorted(m for m in ('config', 'discord', 'commands.psd_analyzer', 'pandas') if m in sys.modules)); "
            "print(sorted(m for m in sys.modules if m.startswith('commands.') or m == 'global_config'))")
    done = subprocess.run([executable, "-c", code], cwd=bot, capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    bad, closure = done.stdout.splitlines()
    assert bad == "[]"
    assert {m.replace(".", "/") + ".py" for m in json.loads(closure.replace("'", '"'))} == set(cg.SPARSE_FILES)
    from test_card_bake import ALLOWED_IMPORTS, third_party_imports
    extra = third_party_imports("import commands.card_guards, commands.txt2_repair", bot) - ALLOWED_IMPORTS
    assert not extra, f"card-guards.yml installs none of these: {sorted(extra)}"


def test_a_pool_worker_takes_the_token(monkeypatch):
    """A spawned worker has no main(): its blob fetches need GITHUB_TOKEN too (else 60 requests/hour)."""
    import commands.github_api as api
    monkeypatch.setattr(api, "GIT_TOKEN", "")
    monkeypatch.setattr(api, "HEADERS", {})
    monkeypatch.setenv("GITHUB_TOKEN", "t0k")
    monkeypatch.setattr(cg, "_worker", {})
    cg._init_worker("clone", "o/r", None, type("Vocab", (), {"parser": lambda self: None})())
    assert api.HEADERS["Authorization"] == "token t0k"


def test_guard_modules_stay_uncompiled():
    """Not in setup.py's INCLUDE_FILES: the Action runs them as pure Python (the change-control skill's step 5)."""
    setup = (REPO / "setup.py").read_text(encoding="utf-8")
    for module in ("card_guards", "png_export", "card_standards"):
        assert f"commands/{module}.py" not in setup


def _workflow(clone) -> Path:
    path = WORKFLOW or (clone / ".github" / "workflows" / "card-guards.yml")
    if not path.exists():
        pytest.skip(f"no {path} (set CARD_GUARDS_YML to a TTSCardMaker checkout's file)")
    return path


def test_workflow_shape(clone):
    """card-guards.yml: push on PSDs and PNGs behind CARD_GUARDS, dry dispatches by default, a run per push, the bot's
    commits skipped, the sparse list = SPARSE_FILES, the fork pin shared with card-edits.yml, secrets by name only, no ping."""
    yaml = pytest.importorskip("yaml")
    path = _workflow(clone)
    text = path.read_text(encoding="utf-8")
    doc = yaml.safe_load(text)
    on = doc[True]
    assert on["push"] == {"branches": ["main"], "paths": ["**/*.psd", "**/*.PSD", "**/*.png", "**/*.PNG"]}
    inputs = on["workflow_dispatch"]["inputs"]
    assert inputs["dry_run"]["default"] is True and inputs["all"]["default"] is False and set(inputs) == {"dry_run", "sha", "cards", "all"}
    assert doc["concurrency"]["group"] == ("${{ github.event_name == 'push' && format('card-guards-{0}', github.sha) || "
                                           "format('card-guards-{0}', github.run_id) }}") and doc["concurrency"]["cancel-in-progress"] is False
    job = doc["jobs"]["guard"]
    assert "github.actor != 'AutoCroissant'" in job["if"] and "github.actor != 'github-actions[bot]'" in job["if"]
    assert "vars.CARD_GUARDS == 'report' || vars.CARD_GUARDS == 'on'" in job["if"]
    assert job["env"]["MODE"] == ("${{ github.event_name == 'push' && (vars.CARD_GUARDS || 'off') || "
                                  "((inputs.dry_run || inputs.all) && 'dry' || 'on') }}")
    steps = job["steps"]
    bot = next(s for s in steps if s.get("with", {}).get("repository") == "bobberlington/AutoCroissant")
    assert bot["with"]["sparse-checkout"].split() == ["/" + f for f in cg.SPARSE_FILES]
    cards = next(s for s in steps if s.get("uses", "").startswith("actions/checkout") and "repository" not in s.get("with", {}))
    assert cards["with"]["filter"] == "blob:none" and cards["with"]["fetch-depth"] == 0
    pin = re.search(r"psd-tools@([\w.-]+)\"", text).group(1)
    edits = path.parent / "card-edits.yml"
    if edits.exists():
        assert re.search(r"psd-tools@([\w.-]+)\"", edits.read_text(encoding="utf-8")).group(1) == pin
    names = [s.get("name") for s in steps]
    assert names.index("Start") < names.index("No push run in its window") < names.index("Check") \
        < names.index("Comment on the commit") < names.index("Land") < names.index("Follow up") < names.index("Tell Discord") \
        < names.index("Fail on card errors") < names.index("Tell Discord the run failed")
    tell = steps[names.index("Tell Discord")]
    assert tell["env"]["WEBHOOK"] == "${{ secrets.DISCORD_STANDARDS_WEBHOOK }}" and "discord-final.json" in tell["run"]
    assert tell["if"].startswith("always() && (env.MODE == 'report' || env.MODE == 'on') && steps.check.outcome == 'success'")
    assert '"$WEBHOOK"' in tell["run"] and "echo \"$WEBHOOK" not in text and "set -x" not in text
    land = steps[names.index("Land")]
    # Land never waits on the Comment step's success (a failed comment must not drop the writes), only on Check's
    assert land["if"] == "!cancelled() && steps.check.outcome == 'success' && (env.MODE == 'report' || env.MODE == 'on')"
    assert "--dry-run" in land["run"]
    assert "discord.json" not in tell["run"].replace("discord-final.json", "")      # never check's early post
    assert "card_errors" in steps[names.index("Fail on card errors")]["run"]
    alert = steps[names.index("Tell Discord the run failed")]
    # a re-run of a push already told (the comment's mark) never alerts: "one post per push"
    assert alert["if"] == ("(failure() || cancelled()) && github.event_name == 'push' && steps.tell.outputs.posted != 'true' && "
                           "steps.comment.outputs.told != 'true'")
    assert alert["env"]["WEBHOOK"] == "${{ secrets.DISCORD_STANDARDS_WEBHOOK }}" and "allowed_mentions: {parse: []}" in alert["run"]
    # the streak's mark: an artifact left only when the alert told (or stayed quiet in a told streak)
    streak = steps[names.index("Tell Discord the run failed") + 1]
    assert streak["uses"].startswith("actions/upload-artifact") and streak["if"] == "always() && steps.alert.outputs.streak == 'true'"
    assert streak["with"]["name"] == "${{ env.STREAK }}" and job["env"]["STREAK"] and alert["id"] == "alert"
    refuse = steps[names.index("No push run in its window")]
    assert refuse["if"] == "github.event_name == 'workflow_dispatch' && env.MODE == 'on'"
    assert "<@&" not in text and "ROLE" not in text and "DISCORD_FONT_WEBHOOK" not in text
    assert set(re.findall(r"secrets\.(\w+)", text)) == {"GITHUB_TOKEN", "DISCORD_STANDARDS_WEBHOOK"}
    check = steps[names.index("Check")]
    for flag in ("--clone", "--repo", "--after", "--author", "--fonts", "--out", "--mode", "--before", "--cards", "--whole-tree"):
        assert flag in check["run"]
    assert check["env"]["AUTHOR"] == "${{ github.event_name == 'push' && github.actor || '' }}"   # never the dispatcher
    # a job's env may read only github, inputs, matrix, needs, secrets, strategy and vars: GitHub rejects the whole file
    assert not any("runner." in str(value) for value in job["env"].values())
    assert 'OUT=$RUNNER_TEMP/guards' in steps[names.index("Start")]["run"]
    follow = steps[names.index("Follow up")]
    assert "report-landed.md" in follow["run"] and follow["if"] == "always() && steps.land.outputs.ran == 'true'"


FAKE_GH = r"""#!/usr/bin/env python3
import json, os, subprocess, sys
state = os.environ["FAKE_STATE"]
db = json.load(open(state)) if os.path.exists(state) else {"comments": {}, "next": 100, "log": []}
args = sys.argv[1:]
db["log"].append(" ".join(args))
out = ""


def jq(data):
    # gh's --jq, by the real jq: the steps' expressions are under test too
    expr = args[args.index("--jq") + 1]
    return subprocess.run(["jq", "-r", expr], input=json.dumps(data), capture_output=True, text=True, check=True).stdout.rstrip("\n")


if os.environ.get("FAKE_FAIL") and os.environ["FAKE_FAIL"] in " ".join(args):
    json.dump(db, open(state, "w"))
    sys.exit(1)
if args[:2] == ["workflow", "run"]:
    pass
elif args[:2] == ["run", "list"]:
    out = jq(json.loads(os.environ["FAKE_RUNS"])) if "FAKE_RUNS" in os.environ else os.environ.get("FAKE_LAST", "")
elif args[0] == "api":
    method = args[args.index("--method") + 1] if "--method" in args else "GET"
    path = next(a for a in args[1:] if a.startswith("repos/"))
    if path.endswith("/artifacts"):
        marked = path.split("/")[-2] in os.environ.get("FAKE_MARKED", "").split(",")
        out = jq({"artifacts": [{"name": "card-guards"}] + ([{"name": os.environ["STREAK"]}] if marked else [])})
    elif method == "POST":
        db["next"] += 1
        db["comments"][str(db["next"])] = json.load(sys.stdin)["body"]
        out = str(db["next"])
    elif method == "PATCH":
        db["comments"][path.rsplit("/", 1)[1]] = json.load(sys.stdin)["body"]
    elif path.endswith("/comments"):
        out = "\n".join(i for i, body in db["comments"].items() if body.startswith("## Card guards"))
    else:
        out = db["comments"][path.rsplit("/", 1)[1]]
json.dump(db, open(state, "w"))
print(out)
"""
FAKE_CURL = r"""#!/bin/sh
for a in "$@"; do case "$prev" in --data) jq -c . "${a#@}" >> "$FAKE_POSTS";; esac; prev="$a"; done
exit "${FAKE_CURL_STATUS:-0}"
"""


def _run_steps(steps, names, env, outputs):
    """Run the named steps' shell as card-guards.yml would (bash -e -o pipefail); outputs[name] = its GITHUB_OUTPUT."""
    for name in names:
        step = steps[name]
        out = Path(env["OUT"]) / f"{name}.output"
        out.write_text("")
        step_env = {**env, **{k: v for k, v in step.get("env", {}).items() if "${{" not in str(v)}, "GITHUB_OUTPUT": str(out)}
        for key, value in step.get("env", {}).items():
            match = re.fullmatch(r"\$\{\{ steps\.(\w+)\.outputs\.(\w+) \}\}", str(value))
            if match:
                step_env[key] = outputs.get(match.group(1), {}).get(match.group(2), "")
        done = subprocess.run(["bash", "--noprofile", "--norc", "-eo", "pipefail", "-c", step["run"]], env=step_env,
                              capture_output=True, text=True, timeout=60)
        assert done.returncode == 0, (name, done.stdout, done.stderr)
        outputs[name.split()[0].lower()] = dict(line.split("=", 1) for line in out.read_text().splitlines() if "=" in line)
    return outputs


def test_workflow_posts_once_and_a_rerun_edits_its_comment(clone, tmp_path):
    """The Comment, Follow up and Tell Discord steps' shell against a fake gh and curl: the first run comments, posts
    the settled post (discord-final.json) and marks the comment; a re-run edits that comment, keeps the mark and does
    not post again."""
    yaml = pytest.importorskip("yaml")
    if not shutil.which("jq"):
        pytest.skip("no jq")
    doc = yaml.safe_load(_workflow(clone).read_text(encoding="utf-8"))
    job = doc["jobs"]["guard"]
    steps = {s["name"]: s for s in job["steps"] if s.get("name")}
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "gh").write_text(FAKE_GH)
    (bin_dir / "curl").write_text(FAKE_CURL)
    for tool in ("gh", "curl"):
        (bin_dir / tool).chmod(0o755)
    out = tmp_path / "guards"
    out.mkdir()
    (out / "summary.json").write_text(json.dumps({"after": "f" * 40, "cards": 1, "told": 1, "comment": 1, "writes": 0,
                                                  "errors": 0, "card_errors": 0}))
    (out / "plan.json").write_text(json.dumps({"after": "f" * 40}))
    (out / "report.md").write_text("## Card guards\n\nAlice's card.\n")
    (out / "report-landed.md").write_text("## Card guards\n\nAfter the window.\n")
    (out / "discord.json").write_text(json.dumps({"content": "early"}))
    (out / "discord-final.json").write_text(json.dumps({"content": "settled"}))
    env = {"PATH": f"{bin_dir}:{os.environ['PATH']}", "OUT": str(out), "TOLD": job["env"]["TOLD"], "MODE": "on",
           "GITHUB_REPOSITORY": "o/cards", "WEBHOOK": "https://discord.invalid/hook", "FAKE_STATE": str(tmp_path / "gh.json"),
           "FAKE_POSTS": str(tmp_path / "posts.txt"), "GITHUB_STEP_SUMMARY": str(tmp_path / "summary.md")}
    order = ["Comment on the commit", "Follow up", "Tell Discord"]
    first = _run_steps(steps, order, env, {"land": {"ran": "true"}})
    state = json.loads((tmp_path / "gh.json").read_text())
    (comment_id, body), = state["comments"].items()
    assert first["comment"] == {"id": comment_id} and first["tell"] == {"posted": "true"}
    assert body.startswith("## Card guards\n\nAfter the window.") and body.rstrip().endswith(job["env"]["TOLD"])
    assert json.loads((tmp_path / "posts.txt").read_text().strip()) == {"content": "settled"}
    again = _run_steps(steps, order[:2], env, {"land": {"ran": "true"}})          # a re-run of the job
    assert again["comment"] == {"told": "true", "id": comment_id}
    assert "steps.comment.outputs.told != 'true'" in steps["Tell Discord"]["if"]     # so Tell Discord is skipped
    state = json.loads((tmp_path / "gh.json").read_text())
    assert list(state["comments"]) == [comment_id] and state["comments"][comment_id].rstrip().endswith(job["env"]["TOLD"])
    assert (tmp_path / "posts.txt").read_text().count("settled") == 1
    assert "WEBHOOK" in steps["Tell Discord the run failed"]["env"]


def test_workflow_failure_streak_counts_told_runs_only(clone, tmp_path):
    """The failure alert's streak is the STREAK artifact, not a red run: quiet only when the last finished push run told
    Discord it failed (or stayed quiet in a told streak); a red run that posted its own lines (a card error, a failed
    comment) leaves no mark, so the next failure is told. A re-run is quiet after an attempt that told. The wording
    says whether the push was checked; a failed webhook call leaves no mark."""
    yaml = pytest.importorskip("yaml")
    if not shutil.which("jq"):
        pytest.skip("no jq")
    job = yaml.safe_load(_workflow(clone).read_text(encoding="utf-8"))["jobs"]["guard"]
    steps = {s["name"]: s for s in job["steps"] if s.get("name")}
    posts, runner = tmp_path / "posts.txt", tmp_path / "runner"
    env = {"PATH": f"{_fake_tools(tmp_path)}:{os.environ['PATH']}", "OUT": str(tmp_path), "GITHUB_REPOSITORY": "o/cards",
           "WEBHOOK": "https://discord.invalid/hook", "FAKE_STATE": str(tmp_path / "gh.json"), "FAKE_POSTS": str(posts),
           "GITHUB_STEP_SUMMARY": str(tmp_path / "summary.md"), "STREAK": job["env"]["STREAK"], "RUNNER_TEMP": str(runner),
           "GITHUB_RUN_ID": "7", "GITHUB_RUN_ATTEMPT": "1", "GITHUB_ACTOR": "Alice", "CHECKED": "failure",
           "RUN": "https://github.com/o/cards/actions/runs/7"}
    def runs(*pairs):
        return json.dumps([{"databaseId": run, "conclusion": conclusion} for run, conclusion in pairs])
    red = runs((5, "failure"))
    cases = (({"FAKE_RUNS": red, "FAKE_MARKED": "5"}, 0, "true"),     # the last push run told the streak
             ({"FAKE_RUNS": runs((7, "failure"), (6, "skipped"), (5, "timed_out")), "FAKE_MARKED": "5"}, 0, "true"),
             ({"FAKE_RUNS": red}, 1, "true"),                          # red after its own post: nobody was told it failed
             ({"FAKE_RUNS": runs((5, "success")), "FAKE_MARKED": "5"}, 1, "true"),   # re-run green: attempt 1's mark is stale
             ({"FAKE_RUNS": runs((5, "cancelled")), "FAKE_MARKED": "5"}, 0, "true"),
             ({"FAKE_RUNS": "[]"}, 1, "true"),                         # the first push run
             ({"FAKE_RUNS": red, "FAKE_MARKED": "7", "GITHUB_RUN_ATTEMPT": "2"}, 0, "true"),   # a re-run: attempt 1 told
             ({"FAKE_RUNS": "[]", "FAKE_CURL_STATUS": "22"}, 1, ""))   # the webhook failed: no mark, the next run tells
    for extra, sent, streak in cases:
        posts.write_text("")
        shutil.rmtree(runner, ignore_errors=True)
        out = _run_steps(steps, ["Tell Discord the run failed"], {**env, **extra}, {})["tell"]
        lines = [json.loads(l) for l in posts.read_text().splitlines() if l.strip()]
        assert len(lines) == sent and out.get("streak", "") == streak, (extra, lines, out)
        assert not streak or (runner / "streak" / "told.txt").read_text().strip() == env["RUN"]      # what STREAK uploads
    for checked, what in (("failure", "was not guarded"), ("success", "was checked, but its run failed")):
        posts.write_text("")
        _run_steps(steps, ["Tell Discord the run failed"], {**env, "FAKE_RUNS": "[]", "CHECKED": checked}, {})
        assert json.loads(posts.read_text()) == {
            "content": f"Card guards: the push by Alice {what} (the next failures stay quiet until a run succeeds). "
                       "https://github.com/o/cards/actions/runs/7", "allowed_mentions": {"parse": []}, "flags": 4}


def test_workflow_tells_only_the_settled_post(clone, tmp_path):
    """Tell Discord posts discord-final.json only: Land ran but settled nothing -> the step fails (the run-failed message
    goes out instead), nothing posted; Land did not run -> nothing to tell. The Comment step goes up for comment-only
    findings (R10 alone: told 0, comment 1). Fail on card errors fails on card_errors only."""
    yaml = pytest.importorskip("yaml")
    if not shutil.which("jq"):
        pytest.skip("no jq")
    job = yaml.safe_load(_workflow(clone).read_text(encoding="utf-8"))["jobs"]["guard"]
    steps = {s["name"]: s for s in job["steps"] if s.get("name")}
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "gh").write_text(FAKE_GH)
    (bin_dir / "curl").write_text(FAKE_CURL)
    for tool in ("gh", "curl"):
        (bin_dir / tool).chmod(0o755)
    out = tmp_path / "guards"
    out.mkdir()
    (out / "summary.json").write_text(json.dumps({"after": "f" * 40, "cards": 1, "told": 0, "comment": 1, "writes": 0,
                                                  "errors": 0, "card_errors": 0}))
    (out / "plan.json").write_text(json.dumps({"after": "f" * 40}))
    (out / "report.md").write_text("## Card guards\n\n- Lonely.png does not match its PSD\n")
    (out / "discord.json").write_text(json.dumps({"content": "early"}))
    posts = tmp_path / "posts.txt"
    env = {"PATH": f"{bin_dir}:{os.environ['PATH']}", "OUT": str(out), "TOLD": job["env"]["TOLD"], "MODE": "on",
           "GITHUB_REPOSITORY": "o/cards", "WEBHOOK": "https://discord.invalid/hook", "FAKE_STATE": str(tmp_path / "gh.json"),
           "FAKE_POSTS": str(posts), "GITHUB_STEP_SUMMARY": str(tmp_path / "summary.md")}
    done = _run_steps(steps, ["Comment on the commit"], env, {})
    assert done["comment"]["id"] and json.loads((tmp_path / "gh.json").read_text())["comments"]
    tell = steps["Tell Discord"]["run"]
    for ran, code in (("true", 1), ("", 0)):
        result = subprocess.run(["bash", "--noprofile", "--norc", "-eo", "pipefail", "-c", tell], capture_output=True, text=True,
                                env={**env, "LAND_RAN": ran, "GITHUB_OUTPUT": str(out / "tell.output")}, timeout=60)
        assert result.returncode == code, (ran, result.stdout, result.stderr)
    assert not posts.exists() or posts.read_text() == ""
    fail = steps["Fail on card errors"]["run"]
    for errors, code in ((0, 0), (2, 1)):
        (out / "summary.json").write_text(json.dumps({"card_errors": errors}))
        result = subprocess.run(["bash", "--noprofile", "--norc", "-eo", "pipefail", "-c", fail], capture_output=True, text=True,
                                env=env, timeout=60)
        assert result.returncode == code, (errors, result.stdout)


def _fake_tools(tmp_path) -> Path:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    (bin_dir / "gh").write_text(FAKE_GH)
    (bin_dir / "curl").write_text(FAKE_CURL)
    for tool in ("gh", "curl"):
        (bin_dir / tool).chmod(0o755)
    return bin_dir


def _shell(step, env, cwd=None):
    out = Path(env["GITHUB_OUTPUT"])
    out.write_text("")
    done = subprocess.run(["bash", "--noprofile", "--norc", "-eo", "pipefail", "-c", step["run"]], env=env, cwd=cwd,
                          capture_output=True, text=True, timeout=60)
    return done, dict(line.split("=", 1) for line in out.read_text().splitlines() if "=" in line)


def test_workflow_comment_lookup_never_posts_twice(clone, tmp_path):
    """The Comment step reads the lookup whole (no pipe into head, which can kill gh under pipefail): with two earlier
    comments it edits the first. A lookup that fails on a re-run, or an earlier comment it cannot read, counts as told:
    no new comment, the step fails, and Tell Discord and the failure alert stay quiet (their `if`). On a first attempt a
    failed lookup still comments (no earlier comment of this run can exist)."""
    yaml = pytest.importorskip("yaml")
    if not shutil.which("jq"):
        pytest.skip("no jq")
    job = yaml.safe_load(_workflow(clone).read_text(encoding="utf-8"))["jobs"]["guard"]
    steps = {s["name"]: s for s in job["steps"] if s.get("name")}
    step = steps["Comment on the commit"]
    out = tmp_path / "guards"
    out.mkdir()
    (out / "summary.json").write_text(json.dumps({"comment": 1}))
    (out / "plan.json").write_text(json.dumps({"after": "f" * 40}))
    (out / "report.md").write_text("## Card guards\n\nAgain.\n")
    state = tmp_path / "gh.json"
    told = job["env"]["TOLD"]
    env = {"PATH": f"{_fake_tools(tmp_path)}:{os.environ['PATH']}", "OUT": str(out), "TOLD": told, "MODE": "on",
           "GITHUB_REPOSITORY": "o/cards", "FAKE_STATE": str(state), "GITHUB_OUTPUT": str(out / "comment.output"),
           "GITHUB_RUN_ATTEMPT": "2"}

    def seed():
        state.write_text(json.dumps({"comments": {"101": f"## Card guards\n\nFirst.\n\n{told}\n", "102": "## Card guards\n\nOld.\n"},
                                     "next": 102, "log": []}))
    seed()
    done, outputs = _shell(step, env)
    assert done.returncode == 0 and outputs == {"told": "true", "id": "101"}, done.stderr
    comments = json.loads(state.read_text())["comments"]
    assert set(comments) == {"101", "102"} and comments["101"].startswith("## Card guards\n\nAgain.") and told in comments["101"]
    for fail in ("--paginate", "comments/101 --jq"):           # the lookup, then the earlier comment's body
        seed()
        done, outputs = _shell(step, {**env, "FAKE_FAIL": fail})
        assert done.returncode == 1 and outputs == {"told": "true"}, (fail, done.stderr)
        assert json.loads(state.read_text())["comments"] == {"101": f"## Card guards\n\nFirst.\n\n{told}\n",
                                                            "102": "## Card guards\n\nOld.\n"}
    for name in ("Tell Discord", "Tell Discord the run failed"):
        assert "steps.comment.outputs.told != 'true'" in steps[name]["if"]
    state.unlink()
    done, outputs = _shell(step, {**env, "FAKE_FAIL": "--paginate", "GITHUB_RUN_ATTEMPT": "1"})
    assert done.returncode == 0 and outputs == {"id": "101"} and list(json.loads(state.read_text())["comments"]) == ["101"]


def test_workflow_refuses_a_landing_dispatch_while_a_push_run_waits(clone, tmp_path):
    """A dispatch with dry_run off on a commit whose push run is unfinished (its head is the commit or after it) would
    post and land first: refused. A push run of an earlier commit, or none, lets it through; an unknown head (pushed
    since the checkout) refuses."""
    yaml = pytest.importorskip("yaml")
    job = yaml.safe_load(_workflow(clone).read_text(encoding="utf-8"))["jobs"]["guard"]
    step = next(s for s in job["steps"] if s.get("name") == "No push run in its window")
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    shas = []
    for n in range(2):
        (repo / "f.txt").write_text(str(n))
        _git(repo, "add", "f.txt")
        _commit(repo, f"c{n}")
        shas.append(_git(repo, "rev-parse", "HEAD"))
    one, two = shas
    env = {"PATH": f"{_fake_tools(tmp_path)}:{os.environ['PATH']}", "GITHUB_REPOSITORY": "o/cards", "GITHUB_SHA": two,
           "FAKE_STATE": str(tmp_path / "gh.json"), "GITHUB_OUTPUT": str(tmp_path / "refuse.output")}
    def runs(*pairs):
        return json.dumps([{"status": status, "headSha": head} for status, head in pairs])
    for sha, pushes, code in (("", runs(), 0), (one, runs(("in_progress", two)), 1), (two, runs(("in_progress", one)), 0),
                              ("", runs(("queued", two)), 1), (one, runs(("completed", two)), 0),
                              (one, runs(("completed", one), ("waiting", one), ("in_progress", two)), 1),
                              (two, runs(("in_progress", "e" * 40)), 1)):
        done, _ = _shell(step, {**env, "SHA": sha, "FAKE_RUNS": pushes}, cwd=repo)
        assert done.returncode == code, (sha, pushes, done.stdout, done.stderr)
        if code:
            assert "is still in its quiet window" in done.stdout


def test_workflow_lowercases_card_guards(clone, tmp_path):
    """The job's `if` matches CARD_GUARDS in any case ("On"), the shell does not: Start lowercases MODE for every later
    step (Land's --dry-run test is exact: a "Report" Check now accepts would otherwise commit)."""
    yaml = pytest.importorskip("yaml")
    job = yaml.safe_load(_workflow(clone).read_text(encoding="utf-8"))["jobs"]["guard"]
    step = next(s for s in job["steps"] if s.get("name") == "Start")
    for typed, mode in (("On", "on"), ("REPORT", "report"), ("dry", "dry")):
        github_env = tmp_path / f"env-{typed}"
        github_env.write_text("")
        done, _ = _shell(step, {"PATH": os.environ["PATH"], "MODE": typed, "RUNNER_TEMP": str(tmp_path),
                                "GITHUB_ENV": str(github_env), "GITHUB_OUTPUT": str(tmp_path / "start.output")})
        assert done.returncode == 0 and f"MODE={mode}" in github_env.read_text().splitlines(), done.stderr
    land = next(s for s in job["steps"] if s.get("name") == "Land")
    assert '[ "$MODE" = report ] && args+=(--dry-run)' in land["run"]


def test_mode_takes_any_case(cards, tmp_path):
    root, base, alice, bot = cards
    out = tmp_path / "out"
    assert cg.main(["check", "--clone", str(root), "--after", bot, "--before", base, "--mode", "Report", "--cards", "bread",
                    "--workers", "1", "--out", str(out)]) == 0
    assert json.loads((out / "plan.json").read_text())["mode"] == "report"


def test_workflow_lints(clone):
    """actionlint (when installed) accepts card-guards.yml: a file GitHub rejects fails every run, pushes included."""
    actionlint = shutil.which("actionlint")
    if not actionlint:
        pytest.skip("no actionlint on PATH")
    done = subprocess.run([actionlint, "-shellcheck=", str(_workflow(clone))], capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stdout + done.stderr
