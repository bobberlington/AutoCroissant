"""
card_guards (the push guards' driver): a card repo made here with git (fixture PSDs, every case of PUSH_GUARDS.md 4.2,
bot commits left out), check end to end on it (the report, the one Discord post without mentions or previews, the plan),
land against a fake GitHub (superseded cards, a moved head, the quiet wait) and in a local clone, the replay of the real
push 38c55bd3..07589cc7 from the card clone (read-only: God Tamer, Bubble and 12 Reactions2 "PNG still current"), the
Action's sparse copy (SPARSE_FILES), card-guards.yml's shape, and its steps' shell against a fake gh and curl (one post
per push, the failure streak, a re-run's lookups and window, the refused dispatch, MODE in any case).
"""
import json
import os
import re
import shutil
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from importlib.util import find_spec
from sys import executable

import pytest
from PIL import Image, ImageDraw

import commands.card_guards as cg
import commands.git_commit as gc
import commands.png_export as pe
from commands.github_api import blob_sha
from test_png_export import _rows_off, art, card, fixture_psd, photoshop_png

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
    # report mode makes nothing, so the PNGs the bot would make are told (with CARD_GUARDS=on they land instead)
    assert by["New_Name"]["told"] == ["PNG not renamed with the PSD: rename it too"]
    assert by["New_Name"]["notes"] == ["Old_Name.png is left at the PSD's old name"]
    assert snack["told"][:2] == ["PNG not exported with the PSD: export it in Photoshop",
                                 "Snack2.png not updated with the PSD: re-export it in Photoshop"]
    assert by["New_Card"]["told"][0] == "no PNG pushed with the new PSD: export it in Photoshop"
    assert by["Kept"]["told"] == [] and by["Kept"]["notes"] == ["Kept.png was deleted; the PSD stays"]
    assert (out / "crops" / FOLDER / "Snack.diff.png").exists()
    summary = json.loads((out / "summary.json").read_text())
    assert summary == {"after": bot, "cards": 8, "told": summary["told"], "comment": 6, "writes": 4, "errors": 0,
                       "card_errors": 0} and summary["told"] >= 1
    assert by["Bread"]["told"] == [] and not by["Bread"]["issues"]      # R3: no Ability layer is the developer's layout
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
    assert "- PNG would be re-exported by the bot's render of the PSD (441 px changed" in report and "with CARD_GUARDS=on" in report
    assert "- PNG re-exported" not in report and "- PNG copied" not in report
    assert "Alice pushed Snack (Square pixels changed) with these issues: PNG not exported with the PSD: export it in " \
           "Photoshop; Snack2.png" in payload["content"]
    # what changed inside each PSD pushed over an older one (psd_diff): a new PSD or a PNG alone has none
    assert snack["changes"] == by["Bread"]["changes"] == by["Dropped"]["changes"] == ["Square pixels changed"]
    assert by["New_Card"]["changes"] == by["Lonely"]["changes"] == by["New_Name"]["changes"] == []
    assert "**Snack** (`Items/Heal/2 Stars/Snack.psd`, Alice, `" in report and "- Changed: Square pixels changed" in report
    assert "Alice pushed New Card with these issues" in payload["content"]              # a new PSD: nothing to compare


def test_what_changed_never_stops_a_card(cards, tmp_path, monkeypatch):
    """psd_diff only tells: when it fails, the card is guarded as before and its lines carry no change."""
    root, base, alice, bot = cards

    def crash(*args, **kwargs):
        raise ValueError("an unreadable layer")
    monkeypatch.setattr(cg.psd_diff, "changes", crash)
    data = cg.check(str(root), bot, base, "o/cards", mode="report", out=str(tmp_path / "out"), workers=1, cards={"snack"})
    (snack,) = data["cards"]
    assert snack["changes"] == [] and not snack["error"] and snack["export"]["outcome"] == "re-exported"
    assert "Alice pushed Snack with these issues: PNG not exported" in data["discord"]["content"]


def test_a_long_change_list_is_cut_to_fit_the_line():
    entry = {"login": "Alice", "card": "Snack", "told": ["an issue"], "commit": "7" * 40,
             "changes": [f"layer {n} hidden" for n in range(40)]}
    content = cg.discord_payload({"after": "a" * 40, "cards": [entry]}, "o/cards")["content"]
    changes = content[len("Alice pushed Snack ("):content.index(") with these issues")]
    assert len(changes) <= cg.CHANGES_DISCORD and changes.endswith("more") and changes.startswith("layer 0 hidden; ")
    assert "- Changed: layer 0 hidden" in cg.markdown({"mode": "on", "after": "a" * 40, "before": "b" * 40, "cards": [
        {**entry, "psd": "Items/Snack.psd", "export": {}, "r10": {}, "issues": [], "fixed": [], "fix_refused": [],
         "notes": [], "writes": []}]}, [], [], [])


def test_changes_give_way_to_issues_on_discord():
    """A long push: the cards lose their changes, the last card first, before any card's issues are left out."""
    cards = [{"login": "Alice", "card": f"Card {n}", "told": ["an issue"], "commit": "7" * 40,
              "changes": [f"layer {n} hidden; " * 8]} for n in range(16)]
    content = cg.discord_payload({"after": "a" * 40, "cards": cards}, "o/cards")["content"]
    lines = content.splitlines()
    assert len(content) <= cg.DISCORD_LIMIT and len(lines) == 16 and "more card" not in content
    assert lines[0].startswith("Alice pushed Card 0 (layer 0 hidden") and lines[15].startswith("Alice pushed Card 15 with")


def test_changes_are_text_in_markdown():
    """Card text and layer names neither format nor link (GitHub comment, Discord post)."""
    entry = {"login": "Alice", "card": "Snack", "told": ["an issue"], "commit": "7" * 40, "psd": "Items/Snack.psd",
             "changes": ["Ability text: '*2* ATK' -> '[fire](1CD) ~~x~~'", "Ability_FX_Fire hidden"], "export": {},
             "r10": {}, "issues": [], "fixed": [], "fix_refused": [], "notes": [], "writes": []}
    escaped = r"Ability text: '\*2\* ATK' -> '\[fire\](1CD) \~\~x\~\~'; Ability\_FX\_Fire hidden"
    assert f"Alice pushed Snack ({escaped}) with" in cg.discord_payload({"after": "a" * 40, "cards": [entry]}, "o/r")["content"]
    assert f"- Changed: {escaped}" in cg.markdown({"mode": "on", "after": "a" * 40, "before": "b" * 40, "cards": [entry]},
                                                  [], [], [])


def test_a_drawing_miss_is_told_in_the_comment_and_the_commit(cards, tmp_path, monkeypatch):
    """Owner (2026-10-23): a PNG the bot drew missing the bar under the floor is made; the miss goes in the comment and
    the bot commit's line per measure, never on Discord."""
    root, base, alice, bot = cards
    _rows_off(monkeypatch, 1)
    data = cg.check(str(root), bot, base, "o/cards", "actor", mode="on", out=str(tmp_path / "out"), workers=1, cards={"snack"})
    (snack,) = data["cards"]
    notes = snack["export"]["misses"]
    assert snack["export"]["outcome"] == "re-exported" and len(notes) == 2
    assert notes[0].startswith("made with a pixel difference (the old PSD drawn against the old PNG): bad ")
    assert snack["writes"][0]["line"].endswith("px changed); " + "; ".join(notes))
    assert data["markdown"].count("; ".join(notes)) == 1 and "; ".join(notes) in cg.message([snack], bot)
    assert not [t for t in snack["told"] if "pixel difference" in t] and "pixel difference" not in json.dumps(data["discord"])


def test_a_new_psd_pushed_with_its_png_gets_the_new_card_rules(cards, tmp_path, monkeypatch):
    """A PSD added with its PNG (case pair, status A) is a new card: R4 (a hidden or rasterized Name layer) is told, not
    waived, as for one pushed alone; an existing card's stays waived."""
    root, base, alice, bot = cards
    monkeypatch.setattr(cg.standards, "title_problem", lambda psd, card: "hidden name layer")   # the fixtures have no Name
    psd, png = (root / FOLDER / "Bread.psd").read_bytes(), (root / FOLDER / "Bread.png").read_bytes()
    push = _plumb(root, {f"{FOLDER}/Fresh.psd": psd, f"{FOLDER}/Fresh.png": png, f"{FOLDER}/Alone.psd": psd,
                         f"{FOLDER}/Lonely.psd": psd}, "two new cards, one old")
    data = cg.check(str(root), push, bot, mode="report", out=str(tmp_path / "out"), workers=1)
    by = {c["card"]: c for c in data["cards"]}
    assert by["Fresh"]["case"] == "pair" and by["Alone"]["case"] == "new" and by["Lonely"]["case"] == "stale"
    for name in ("Fresh", "Alone"):
        (r4,) = [i for i in by[name]["issues"] if i["code"] == "R4"]
        assert not r4["waived"] and r4["told"] and r4["line"] == "the Name layer is hidden" and r4["line"] in by[name]["told"]
    (r4,) = [i for i in by["Lonely"]["issues"] if i["code"] == "R4"]
    assert r4["waived"] and not r4["told"] and r4["line"] not in by["Lonely"]["told"]
    assert cg.added(cg.Job("X.psd", "psd-only")) and not cg.added(cg.Job("X.psd", "pair", psd_old="s"))
    assert not cg.added(cg.Job("X.psd", "stale", psd_old="s")) and not cg.added(cg.Job("X.psd", "whole"))


def test_a_new_psd_beside_its_pushed_png_tells_nothing(cards, tmp_path):
    """A new card's PNG pushed first and its PSD in a later push (case beside): the PNG is the author's export, so the PSD
    is current and nothing is made or told about the PNG (review 7: it was "PNG not made"). Beside an older PNG, the PNG
    takes the render where Photoshop's image moved."""
    root, base, alice, bot = cards
    psd, png = (root / FOLDER / "Bread.psd").read_bytes(), (root / FOLDER / "Bread.png").read_bytes()
    old_png = (root / FOLDER / "Snack2.png").read_bytes()
    first = _plumb(root, {f"{FOLDER}/First.png": png, f"{FOLDER}/Older.png": old_png}, "the PNGs first")
    push = _plumb(root, {f"{FOLDER}/First.psd": psd, f"{FOLDER}/Older.psd": psd}, "then the PSDs")
    data = cg.check(str(root), push, first, mode="on", out=str(tmp_path / "out"), workers=1)
    by = {c["card"]: c for c in data["cards"]}
    assert by["First"]["case"] == by["Older"]["case"] == "beside"
    found = by["First"]["export"]
    assert (found["outcome"], found["changed_px"], by["First"]["writes"]) == ("current", 0, [])
    assert not [t for t in by["First"]["told"] if t.startswith("PNG")]
    assert cg._guard1_line(by["First"]) == "PNG already shows the pushed PSD (exported before the PSD was pushed): nothing to make"
    older = by["Older"]["export"]
    assert older["outcome"] == "re-exported" and older["changed_px"] == 21 * 21 + 16 * 16
    assert [w["path"] for w in by["Older"]["writes"]] == [f"{FOLDER}/Older.png"]


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


def test_a_burst_without_a_stored_image_is_made_against_the_anchor(cards, tmp_path, monkeypatch):
    """G2 has no G5 to refuse, so over push 1's PSD run 2 drew push 2's square alone and called the PNG current: push
    1's Base was left out (review of 2026-10-24, Niko). A G2 export takes the burst anchor: both changes drawn."""
    root, base, alice, bot = cards
    monkeypatch.setattr(cg.png_export, "composite", lambda psd: None)        # saved without Maximize Compatibility
    work = tmp_path / "burst"
    work.mkdir()
    one = _dated(root, work, "Bread", card(base=(30, 200, 30)), "2026-10-22T10:00:00Z")                   # Base, square
    two = _dated(root, work, "Bread", card(base=(30, 200, 30), colour=(0, 255, 0)), "2026-10-22T10:10:00Z")  # square
    out = tmp_path / "out"
    data = cg.check(str(root), two, one, mode="on", out=str(out), workers=1)
    (entry,) = data["cards"]
    assert entry["export"]["outcome"] == "re-exported" and entry["export"]["unchecked"] == pe.UNCHECKED
    assert any("an earlier push changed the PSD without its PNG" in note for note in entry["notes"])
    (write,) = entry["writes"]
    made = Image.open(out / write["file"])
    assert int(pe._diff(made, card(base=(30, 200, 30), colour=(0, 255, 0))).max()) == 0


def test_a_g2_burst_keeps_the_anchors_refusal(cards, tmp_path, monkeypatch):
    """A G2 burst whose anchored export refuses keeps the refusal, never the export over push 1's PSD, which calls the
    PNG current with push 1's Base change left out, nothing told (review of 2026-10-24)."""
    root, base, alice, bot = cards
    monkeypatch.setattr(cg.png_export, "composite", lambda psd: None)
    real = pe.psd_edit.gate_region

    def gate(old_png, merged, region):              # the anchored region (push 1's Base: the whole card) refuses
        if (region[2] - region[0]) * (region[3] - region[1]) > 21 * 21:
            raise pe.psd_edit.Refused("region", "test")
        return real(old_png, merged, region)
    monkeypatch.setattr(pe.psd_edit, "gate_region", gate)
    work = tmp_path / "burst"
    work.mkdir()
    one = _dated(root, work, "Bread", card(base=(30, 200, 30)), "2026-10-22T10:00:00Z")
    two = _dated(root, work, "Bread", card(base=(30, 200, 30), colour=(0, 255, 0)), "2026-10-22T10:10:00Z")
    data = cg.check(str(root), two, one, mode="on", out=str(tmp_path / "out"), workers=1)
    (entry,) = data["cards"]
    assert (entry["export"]["outcome"], entry["export"]["gate"]) == ("refused", "G7"), entry["export"]
    assert not [w for w in entry["writes"] if w["kind"] == "png"] and pe.G7 in entry["told"]


def test_a_g2_burst_refusal_is_not_retried_when_the_png_shows_no_anchor(cards, tmp_path, monkeypatch):
    """A G2 refusal is no checked G5/G6 (Photoshop's image never vouched for it): with the PNG not showing the anchor it
    stands, never retried against the anchor. Push 1 the Base, push 2 the square, the PNG showing neither (shows
    False): over push 1's PSD the baseline is past the floor (G6)."""
    root, base, alice, bot = cards
    monkeypatch.setattr(cg.png_export, "composite", lambda psd: None)
    monkeypatch.setattr(cg.png_export, "shows", lambda png, psd, other: False)
    work = tmp_path / "burst"
    work.mkdir()
    one = _dated(root, work, "Bread", card(base=(30, 200, 30)), "2026-10-22T10:00:00Z")
    two = _dated(root, work, "Bread", card(base=(30, 200, 30), colour=(255, 255, 0)), "2026-10-22T10:10:00Z")
    data = cg.check(str(root), two, one, mode="on", out=str(tmp_path / "out"), workers=1)
    (entry,) = data["cards"]
    assert (entry["export"]["outcome"], entry["export"]["gate"]) == ("refused", "G6"), entry["export"]
    assert entry["export"]["why"].startswith("PNG not made: the bot cannot draw this part of the card")
    assert not any("an earlier push changed the PSD" in note for note in entry["notes"]), entry["notes"]


def test_a_checked_burst_whose_last_push_moved_nothing_keeps_the_first(cards, tmp_path):
    """Push 1 a 4-px dot, push 2 a layer record that draws nothing (the square grown by a column of the base colour),
    stored images kept: over push 1's PSD the PNG was "current" (G5's bar hides 4 px), push 1's dot lost, nothing told
    (review of 2026-10-24: Tainted Lost, Grimora, Dogma). The old PNG shows the anchor (png_export.shows), so the anchor
    decides: the dot drawn."""
    root, base, alice, bot = cards
    work = tmp_path / "burst"
    work.mkdir()
    dotted = card(colour=(0, 255, 0))
    ImageDraw.Draw(dotted).rectangle((12, 12, 13, 13), fill=(255, 255, 0))
    one = _dated(root, work, "Bread", dotted, "2026-10-22T10:00:00Z")
    fixture_psd(work, dotted, "grown", square=(10, 10, 32, 31))
    shutil.copy(work / "grown.psd", root / f"{FOLDER}/Bread.psd")
    env = {**os.environ, "GIT_AUTHOR_DATE": "2026-10-22T10:10:00Z", "GIT_COMMITTER_DATE": "2026-10-22T10:10:00Z"}
    subprocess.run(["git", "-C", str(root), "commit", "-q", "-am", "grown"], check=True, env=env)
    two = _git(root, "rev-parse", "HEAD")
    source = cg.Source(str(root))
    old, new = (pe.open_psd(source.blob(source.tree(c)[f"{FOLDER}/Bread.psd"][1])) for c in (one, two))
    assert pe.export(old, new, card(colour=(0, 255, 0))).outcome == "current"       # over push 1's PSD: the dot lost
    out = tmp_path / "out"
    data = cg.check(str(root), two, one, mode="on", out=str(out), workers=1)
    (entry,) = data["cards"]
    assert (entry["export"]["outcome"], entry["export"]["changed_px"]) == ("re-exported", 4), entry["export"]
    assert any("an earlier push changed the PSD without its PNG" in note for note in entry["notes"])
    (write,) = entry["writes"]
    assert int(pe._diff(Image.open(out / write["file"]), dotted).max()) == 0


@pytest.mark.parametrize("dot", [False, True])
def test_a_png_pushed_before_its_psd_keeps_g2_off_the_anchor(cards, tmp_path, monkeypatch, dot):
    """Push A the PNG of v2 alone, B the PSD v2, C the PSD v3 (no stored image), within the burst window: the anchor is
    v1 (the PSD at A) but the PNG shows v2, so G2 draws v3 over v2 as the checked path does. Over the anchor it refused
    G6 (the square recoloured) or called a 4-px dot current, nothing told (review of 2026-10-24)."""
    root, base, alice, bot = cards
    work = tmp_path / "burst"
    work.mkdir()
    v2, v3 = card(base=(30, 200, 30)), card(base=(30, 200, 30), colour=(0, 255, 0))
    if dot:
        v3 = card(base=(30, 200, 30))
        ImageDraw.Draw(v3).rectangle((12, 12, 13, 13), fill=(255, 255, 0))
    _png(root, f"{FOLDER}/Bread.png", v2)
    env = {**os.environ, "GIT_AUTHOR_DATE": "2026-10-22T10:00:00Z", "GIT_COMMITTER_DATE": "2026-10-22T10:00:00Z"}
    subprocess.run(["git", "-C", str(root), "commit", "-q", "-am", "PNG first"], check=True, env=env)
    one = _dated(root, work, "Bread", v2, "2026-10-22T10:05:00Z")
    two = _dated(root, work, "Bread", v3, "2026-10-22T10:10:00Z")
    source = cg.Source(str(root))
    (job,) = cg.plan_jobs(source, one, two, "actor")[0]
    assert job.case == "stale" and job.psd_anchor                     # the anchor is v1: not what the PNG shows
    monkeypatch.setattr(cg.png_export, "composite", lambda psd: None)
    out = tmp_path / "out"
    data = cg.check(str(root), two, one, mode="on", out=str(out), workers=1)
    (entry,) = data["cards"]
    assert entry["export"]["outcome"] == "re-exported" and entry["export"]["changed_px"] == (4 if dot else 441)
    assert not any("an earlier push changed the PSD" in note for note in entry["notes"]), entry["notes"]
    assert not any(line.startswith("PNG not made") for line in entry["told"])
    (write,) = entry["writes"]
    assert int(pe._diff(Image.open(out / write["file"]), v3).max()) == 0


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
              {"line": "comment only", "told": True, "discord": False, "fix": False, "waived": ""}]
    result = cg.Result(cg.Job("X/A.psd", "pair"), issues=issues)
    assert result.told("report") == ["the Name layer is named 'NAME'"]
    assert result.told("on") == [] and result.told("dry") == []


def test_fonts_are_left_to_font_check(cards, tmp_path, monkeypatch, capsys):
    """Owner, 2026-10-24: missing fonts are reported by font-check.yml only. The guards take no fonts (no --fonts, no
    fonts_dir) and tell none, in the post or the comment."""
    from inspect import signature
    root, base, alice, bot = cards
    monkeypatch.setattr("commands.font_check.FontIndex.supplies", lambda index, font: False)
    assert "fonts_dir" not in signature(cg.check).parameters and "fonts_dir" not in signature(cg.build_context).parameters
    out = tmp_path / "out"
    data = cg.check(str(root), bot, base, "o/cards", mode="report", out=str(out), workers=1, cards={"bread"})
    (bread,) = data["cards"]
    assert not [issue for issue in bread["issues"] if issue["code"] == "R7"] and "ComicSansMS" not in json.dumps(data["cards"])
    assert "ComicSansMS" not in data["markdown"] and "ComicSansMS" not in json.dumps(data["discord"])
    with pytest.raises(SystemExit):
        cg.main(["check", "--help"])
    assert "--fonts" not in capsys.readouterr().out


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
    entry = {"login": "Zomoroko", "card": "Skullord", "told": ["timing tag (1GCD) not set in the stoogedings badge font"], "commit": "7" * 40}
    data = {"after": "a" * 40, "cards": [dict(entry, card=f"Card {i}") for i in range(80)]}
    payload = cg.discord_payload(data, "MichaelJSr/TTSCardMaker")
    assert len(payload["content"]) <= cg.DISCORD_LIMIT and payload["flags"] == 4
    last = payload["content"].splitlines()[-1]
    assert re.fullmatch(r"\.\.\. and \d+ more cards: \[details\]\(https://github\.com/MichaelJSr/TTSCardMaker/commit/a{40}\)", last)
    one = cg.discord_payload({"after": "a" * 40, "cards": [entry]}, "MichaelJSr/TTSCardMaker")["content"]
    assert one == ("Zomoroko pushed Skullord with these issues: timing tag (1GCD) not set in the stoogedings badge font. "
                   "[77777777](https://github.com/MichaelJSr/TTSCardMaker/commit/" + "7" * 12 + ")")
    assert cg.discord_payload({"after": "a" * 40, "cards": [dict(entry, told=[])]}, "o/r") == {}


def test_message():
    landed = [{"login": "Zomoroko", "commit": "7cb4c867" + "0" * 32,
               "writes": [{"path": "Items/X.png", "kind": "png", "line": "re-exported by the bot's render of the PSD (12 px changed)"},
                          {"path": "Items/Y.psd", "kind": "psd", "line": "Txt2 synced"}]}]
    text = cg.message(landed, "f" * 40)
    assert text.splitlines()[0] == f"Card guards: 1 PNG re-exported, 1 PSD fixed (push {'f' * 8})"
    assert "Items/X.png: re-exported by the bot's render of the PSD (12 px changed) (Zomoroko, 7cb4c867)" in text
    assert text.rstrip().endswith(f"Guarded-push: {'f' * 40}") and "Edit-by" not in text
    landed[0]["writes"].append({"path": "Items/Z.png", "kind": "recompress", "line": "recompressed losslessly (9.0 MB -> 2.0 MB)"})
    text = cg.message(landed, "f" * 40)
    assert text.splitlines()[0] == f"Card guards: 1 PNG re-exported, 1 PSD fixed, 1 PNG recompressed (push {'f' * 8})"
    trailers = subprocess.run(["git", "interpret-trailers", "--parse"], input=text, capture_output=True, text=True, check=True).stdout
    assert trailers == f"Recompressed: Items/Z.png\nGuarded-push: {'f' * 40}\n"     # what git (and build_site) reads


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


def test_a_push_during_the_check_still_supersedes(tmp_path, monkeypatch):
    """The environment's timer starts Land 30 minutes after the check job ends: a push made while the check ran (here a
    case-only rename 3 minutes after the PSD, the same blob) is still a later push. The window counts from the push."""
    import commands.card_bake as cb
    plan = _plan(tmp_path, [("Foo", "C/foo.png", b"bot png", {"C/foo.psd": "s1", "C/foo.png": "p1"})])
    hub = FakeHub({"C/foo.psd": ("100644", "s1"), "C/Foo.png": ("100644", "p1")})
    hub.install(monkeypatch)
    monkeypatch.setattr(cg, "recent_paths", cb.recent_paths)                 # the real window, over a faked API
    start = datetime(2026, 10, 24, 12, 0, tzinfo=timezone.utc)
    pushes = [{"timestamp": "2026-10-24T12:03:00Z", "before": "b" * 40, "after": "h1"}]

    def api(repo, method, path, **kwargs):
        reply = {"activity": pushes, f"compare/{'b' * 40}...h1": {"files": [
            {"filename": "C/Foo.png", "previous_filename": "C/foo.png", "status": "renamed"}]}}[path]
        return type("Reply", (), {"ok": True, "json": lambda self: reply})()
    monkeypatch.setattr(cb, "api", api)
    report = cg.land(plan, quiet_from="2026-10-24T12:00:00Z", sleep=lambda s: None,
                     now=lambda: start + timedelta(minutes=40))               # check 10 min, then the 30-minute timer
    assert report["sha"] == "" and report["landed"] == [] and not hub.commits
    assert report["dropped"] == [["Foo", "superseded: a later push touched C/foo.png", "C/foo.png"]]


@pytest.mark.parametrize("pushes, files", [(100, 1), (4, 300)])
def test_a_late_rerun_counts_pushes_from_now(tmp_path, monkeypatch, pushes, files):
    """A re-run 3 hours after the push: from the push its window held the day's pushes (100, or 300 files) and dropped
    every write as "too many pushes to tell" (review of 2026-10-24). It counts from now, as before: none in the last
    30 minutes, the blob check alone."""
    import commands.card_bake as cb
    plan = _plan(tmp_path, [("A", "X/A.png", b"bot png", {"X/A.psd": "s1", "X/A.png": "p1"})])
    hub = FakeHub({"X/A.psd": ("100644", "s1"), "X/A.png": ("100644", "p1")})
    hub.install(monkeypatch)
    monkeypatch.setattr(cg, "recent_paths", cb.recent_paths)
    push = datetime(2026, 10, 24, 9, 0, tzinfo=timezone.utc)
    activity = [{"timestamp": (push + timedelta(minutes=5 + i)).strftime("%Y-%m-%dT%H:%M:%SZ"), "before": f"{i:040d}",
                 "after": f"{i + 1:040d}"} for i in range(pushes)]

    def api(repo, method, path, **kwargs):
        reply = activity if path == "activity" else {"files": [{"filename": f"Y/{i}.png"} for i in range(files)]}
        return type("Reply", (), {"ok": True, "json": lambda self: reply})()
    monkeypatch.setattr(cb, "api", api)
    report = cg.land(plan, quiet_from="2026-10-24T09:00:00Z", rerun=True, sleep=lambda s: None,
                     now=lambda: push + timedelta(hours=3))
    assert report["landed"] == ["A"] and report["dropped"] == [] and hub.commits


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
    bread = _entry("Bread", told=["timing tag (1GCD) not set in the stoogedings badge font"])
    kept = _entry("Kept", told=["timing tag (1GCD) not set in the stoogedings badge font"])
    plan = _full_plan(tmp_path, [snack, bread, kept], mode="report")
    FakeHub({"X/Snack.psd": ("100644", "s1"), "X/Snack.png": ("100644", "pushed"), "X/Bread.psd": ("100644", "again"),
             "X/Bread.png": ("100644", "p1"), "X/Kept.psd": ("100644", "s1"), "X/Kept.png": ("100644", "p1")}).install(monkeypatch)
    now = datetime(2026, 10, 22, 18, 0, tzinfo=timezone.utc)
    waits = []
    report = cg.land(plan, quiet_from="2026-10-22T17:59:00Z", dry_run=True, sleep=waits.append, now=lambda: now)
    assert waits and report["told"] == {"X/Snack.psd": [], "X/Bread.psd": [], "X/Kept.psd": ["timing tag (1GCD) not set in the stoogedings badge font"]}
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
                   told=["timing tag (1GCD) not set in the stoogedings badge font"])
    _write(out, snack, "X/Snack.png", "png", b"png", "re-exported")
    _write(out, snack, "X/Snack.psd", "psd", b"psd", "the Name layer is named 'NAME'")
    plan = _full_plan(tmp_path, [snack])
    hub = FakeHub({"X/Snack.psd": ("100644", blob_sha(b"psd")), "X/Snack.png": ("100644", blob_sha(b"png"))})
    hub.install(monkeypatch)
    report = cg.land(plan)
    assert report["sha"] == "" and report["error"] == "" and report["dropped"] == [] and not hub.commits
    assert sorted(report["already"]) == [["Snack", "X/Snack.png"], ["Snack", "X/Snack.psd"]]
    assert report["told"] == {"X/Snack.psd": ["timing tag (1GCD) not set in the stoogedings badge font"]}      # the PSD did not "change again"
    comment = cg.landed_markdown(json.loads(Path(plan).read_text()), report)
    assert "Nothing new to land: 1 card already on the branch" in comment and "Fixed by the bot" in comment
    assert "Every write is already on the branch" in comment and "Not landed" not in comment and "Not fixed" not in comment


def test_report_mode_tells_the_png_it_would_make(tmp_path, monkeypatch):
    """Report mode commits nothing, so a PSD pushed without its PNG is told ("PNG not exported with the PSD"), once, after
    the window, and the comment says the PNG would be re-exported (CARD_GUARDS=on), never that it was. A PNG the author
    pushed in the window withdraws the line. In on mode the PNG lands instead: nothing told."""
    job = cg.Job("X/Snack.psd", "stale", "s0", "s1", "X/Snack.png", "p1", login="Alice", commit="a" * 40)
    export = {"outcome": "re-exported", "why": "", "gate": "", "changed_px": 1234, "box": [0, 0, 10, 10], "icc": "converted",
              "writer": "photoshop", "vote": None, "numbered_refused": [],
              "numbers": {"baseline": {"bad_pct": 0.0}, "cross": {"bad_pct": 0.01, "mean": 0.3}}}
    result = cg.Result(job, export=export, writes=[{"path": "X/Snack.png", "file": "files/Snack.png", "sha": blob_sha(b"png"),
                                                    "kind": "png", "line": "re-exported"}])
    report = cg.plan([result], "o/r", "e" * 40, "f" * 40, "report")
    (entry,) = report["cards"]
    assert entry["told"] == ["PNG not exported with the PSD: export it in Photoshop"]
    assert cg.discord_payload(report, "o/r")["content"].startswith(
        "Alice pushed Snack with these issues: PNG not exported with the PSD: export it in Photoshop.")
    text = cg.markdown(report, [], [], [])
    assert "Report only (CARD_GUARDS=report): nothing is committed." in text
    assert ("- PNG would be re-exported by the bot's render of the PSD (1,234 px changed in 10x10 at (0, 0); ICC converted; "
            "the old PSD drawn against the old PNG: bad 0.00%; against the PSD's own image: bad 0.01%, mean 0.30; saved by "
            "Photoshop) with CARD_GUARDS=on") in text and "- PNG re-exported" not in text
    assert cg.summary(report, [])["told"] == 1
    on = cg.plan([result], "o/r", "e" * 40, "f" * 40, "on")
    assert on["cards"][0]["told"] == [] and "- PNG re-exported by the bot's render of the PSD (1,234 px" in cg.markdown(on, [], [], [])
    assert cg.plan([result], "o/r", "e" * 40, "f" * 40, "dry")["cards"][0]["told"] == []    # a dry run shows what on would do
    out = tmp_path / "plan"
    (out / "files").mkdir(parents=True)
    (out / "files" / "Snack.png").write_bytes(b"png")
    (out / "plan.json").write_text(json.dumps(report))
    now = datetime(2026, 10, 22, 18, 0, tzinfo=timezone.utc)
    for png, told in (("p1", ["PNG not exported with the PSD: export it in Photoshop"]), ("author's", [])):
        FakeHub({"X/Snack.psd": ("100644", "s1"), "X/Snack.png": ("100644", png)}).install(monkeypatch)
        landed = cg.land(str(out / "plan.json"), quiet_from="2026-10-22T17:00:00Z", dry_run=True, now=lambda: now)
        assert landed["told"] == {"X/Snack.psd": told} and landed["sha"] == "", png
        post = cg.finish(str(out / "plan.json"), landed)
        assert (post.get("content", "").startswith("Alice pushed Snack with these issues: PNG not exported")) is bool(told)
        comment = (out / "report-landed.md").read_text()
        assert ("- After the quiet window: Snack.png was pushed since: not told" in comment) is not told
        assert "would be re-exported" in comment and "Landed" not in comment
    ahead = dict(entry, export={"outcome": "current", "numbers": {"png_ahead": {}}})              # png_export's verdict
    assert cg._guard1_line(ahead, "report") == "PNG already shows the pushed PSD (exported before the PSD was pushed): nothing to make"


def test_a_rerun_that_cannot_read_the_branch_tells_no_write(tmp_path, monkeypatch):
    """Attempt 1 landed its writes and told nothing, then went red elsewhere; the re-run's Land cannot read the head: an
    earlier run may have landed the writes, so none is told as not made (no false "PNG not made" post) and the comment
    says this run committed nothing. A first attempt that cannot read the head did commit nothing: it tells as before."""
    out = tmp_path / "plan"
    snack = _entry("Snack", export={"outcome": "re-exported", "changed_px": 5, "numbers": {}},
                   told=["timing tag (1GCD) not set in the stoogedings badge font"])
    _write(out, snack, "X/Snack.png", "png", b"png", "re-exported")
    _write(out, snack, "X/Snack.psd", "psd", b"psd", "the Name layer is named 'NAME'")
    snack["fixed"] = ["the Name layer is named 'NAME'"]
    plan = _full_plan(tmp_path, [snack])

    def down(*args, **kwargs):
        raise gc.Transient("API 502")
    monkeypatch.setattr(cg, "ref_head", down)
    first = cg.land(plan, quiet_from="2020-01-01T00:00:00Z")
    assert first["error"] == "API 502" and not first["unread"]
    assert first["told"]["X/Snack.psd"][1:] == ["PNG not made: the bot's commit failed; export it in Photoshop",
                                                "the Name layer is named 'NAME' (the bot could not fix it: the bot's commit failed)"]
    for name in ("landed.json", "report-landed.md", "discord-final.json"):
        (out / name).unlink(missing_ok=True)
    assert cg.main(["land", "--plan", plan, "--rerun"]) == 1
    landed = json.loads((out / "landed.json").read_text())
    assert landed["unread"] and landed["told"] == {"X/Snack.psd": ["timing tag (1GCD) not set in the stoogedings badge font"]}
    post = json.loads((out / "discord-final.json").read_text())["content"]
    assert "PNG not made" not in post and "could not fix" not in post and "badge font" in post
    comment = (out / "report-landed.md").read_text()
    assert "Not landed: API 502. Nothing was committed by this run; an earlier run may have landed it" in comment
    assert "- Not committed by this run: an earlier run may have landed it (the branch could not be read)" in comment
    assert "- Not fixed by this run: the Name layer" in comment and "nothing above was committed" not in comment
    for name in ("landed.json", "report-landed.md", "discord-final.json"):
        (out / name).unlink(missing_ok=True)
    assert cg.main(["land", "--plan", plan]) == 1                                          # a first attempt
    assert "PNG not made: the bot's commit failed" in json.loads((out / "discord-final.json").read_text())["content"]


def test_a_bot_change_in_the_window_keeps_the_lines(tmp_path, monkeypatch):
    """A PSD a bot commit changed during the window (no run follows it): its lines are still told, and its fix that
    could not land is told too. A person's change leaves the card to that push's run."""
    out = tmp_path / "plan"
    snack = _entry("Snack", told=["timing tag (1GCD) not set in the stoogedings badge font"], fixed=["the Name layer is named 'NAME'"])
    _write(out, snack, "X/Snack.psd", "psd", b"psd", "the Name layer is named 'NAME'")
    plan = _full_plan(tmp_path, [snack])
    FakeHub({"X/Snack.psd": ("100644", "baked"), "X/Snack.png": ("100644", "p1")}).install(monkeypatch)
    for author, told in ((("AutoCroissant", gc.AUTHOR["email"]), ["timing tag (1GCD) not set in the stoogedings badge font",
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


def test_a_png_kept_beside_a_psd_without_a_stored_image_is_told():
    """Default 2026-10-24, owner to confirm: a new PSD with no stored image beside a PNG at most KEPT_PX px past
    BESIDE_CUT of the fork's render keeps the PNG (png_export's numbers["kept"]); the post tells it (KEPT, with the px past the cut) on Discord
    and as the comment's guard-1 line, nothing written."""
    exported = cg._export_dict(pe.Export("current", unchecked=True, numbers={"kept": 33, "png_vs_render": {"max": 17}}))
    line = pe.KEPT.format(px=33)
    assert exported["unchecked"] == line and line.startswith("PNG kept as it is: ") and " 33 px" in line
    job = cg.Job("X/Snack.psd", "beside", png="X/Snack.png")
    result = cg.Result(job, export=exported)
    assert result.told("on") == [line] and result.told("report") == [line]
    assert cg._guard1_line({"export": exported, "case": "beside"}) == line
    assert cg._export_dict(pe.Export("current", numbers={"kept": 33}))["unchecked"] == ""      # checked: never kept


def test_a_png_made_without_a_stored_image_is_told_not_refused(tmp_path, monkeypatch):
    """Owner, 2026-10-24 (G2 rendered anyway): a PSD saved without Maximize Compatibility gets the bot's PNG, and the post
    says "no stored image to cross-check" instead of refusing (report mode beside the PNG it would make); the comment's
    guard-1 line says so. The line is withdrawn, as a refusal is, when the author pushed the PNG in the window (the
    bot's write is dropped), and when the write did not land it is the "PNG not made" line alone."""
    found = pe.Export("re-exported", changed_px=5, unchecked=True)
    exported = cg._export_dict(found)
    assert exported["unchecked"] == pe.UNCHECKED and "no stored image to cross-check" in pe.UNCHECKED
    assert cg._export_dict(pe.Export("current", unchecked=True))["unchecked"] == ""
    restyled = cg._export_dict(pe.Export("current", unchecked=True, numbers={"restyled": ["the Header text"]}))
    assert restyled["unchecked"] == pe.RESTYLED.format(what="the Header text")
    assert cg._guard1_line({"export": restyled, "case": "stale"}) == restyled["unchecked"]
    job = cg.Job("X/Snack.psd", "stale", png="X/Snack.png")
    result = cg.Result(job, export=exported, writes=[{"path": "X/Snack.png", "kind": "png"}])
    assert result.told("on") == [pe.UNCHECKED]
    assert result.told("report") == [pe.UNCHECKED, "PNG not exported with the PSD: export it in Photoshop"]
    line = cg._guard1_line({"export": {**exported, "numbers": {}, "icc": "as stored"}, "case": "stale"})
    assert line.startswith("PNG re-exported by the bot's render of the PSD (") and "no stored image to cross-check" in line
    beside = cg._guard1_line({"export": {**exported, "numbers": {"png_vs_render": {"bad_pct": 51.9}}, "icc": "as stored",
                                         "misses": []}, "case": "beside"})
    assert "old PSD" not in beside and "made with a pixel difference" not in beside    # the author's change, no miss
    # default 2026-10-24, owner to confirm: the new PSD drawn against its PNG is plan.json's number alone, not posted
    assert "51.90" not in beside and "the new PSD drawn against its PNG" not in beside
    drawn = cg._export_dict(pe.Export("re-exported", changed_px=5, unchecked=True,
                                      numbers={"png_vs_render": {"bad_pct": 51.9}}))
    assert drawn["numbers"]["png_vs_render"]["bad_pct"] == 51.9

    out = tmp_path / "plan"
    snack = _entry("Snack", export=dict(exported, numbers={}), told=[pe.UNCHECKED])
    _write(out, snack, "X/Snack.png", "png", b"png", "re-exported")
    plan = _full_plan(tmp_path, [snack])
    hub = FakeHub({"X/Snack.psd": ("100644", "s1"), "X/Snack.png": ("100644", "p1")})
    hub.install(monkeypatch)
    report = cg.land(plan)
    assert report["sha"] and report["told"] == {"X/Snack.psd": [pe.UNCHECKED]}
    assert cg.finish(plan, report)["content"].startswith(f"Alice pushed Snack with these issues: {pe.UNCHECKED}.")
    FakeHub({"X/Snack.psd": ("100644", "s1"), "X/Snack.png": ("100644", "author's")}).install(monkeypatch)
    assert cg.land(plan)["told"] == {"X/Snack.psd": []}                     # the author pushed the PNG: nothing to tell
    hub.install(monkeypatch)

    def down(*args, **kwargs):
        raise gc.Transient("the API is down")
    monkeypatch.setattr(cg, "commit_files", down)
    assert cg.land(plan)["told"] == {"X/Snack.psd": ["PNG not made: the bot's commit failed; export it in Photoshop"]}


def test_context_errors_alone_ask_no_comment():
    """summary.json's comment counts cards only: a context error (a standing manifest or status chart one) is in the job
    summary, and in the comment when there is one, but puts no comment on every push."""
    quiet = _entry("Quiet")
    broken = _entry("Broken", error="KeyError: 'x'")
    errors = ["MDW/Croissant's Special Folder/Satus Effect Colors.psd: unreadable", "MDW/Fonts/stoogedings-src/stoogedings.json: timeout"]
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


# ========================
# Lossless recompression (owner, 2026-10-23)
# ========================
@pytest.mark.skipif(find_spec("oxipng") is None, reason="needs pyoxipng, as card-guards.yml installs (none on 3.14)")
def test_pushed_pngs_are_recompressed_in_the_bot_commit(cards, tmp_path):
    """A PNG the push brought (a pair's; a numbered one pushed alone) that a lossless recompression makes more than 10%
    smaller is rewritten in the one bot commit, the same image, never told on Discord; an already small one is left;
    the bot's commit is never guarded (no loop)."""
    root, base, alice, bot = cards
    work = tmp_path / "psds"                    # the fixture's PSD folder
    big = art("RGB", (300, 450))
    fixture_psd(work, big, "Fat")
    psd, heavy, numbered = (work / "Fat.psd").read_bytes(), photoshop_png(big), photoshop_png(art("RGB", (300, 450)))
    push = _plumb(root, {f"{FOLDER}/Fat.psd": psd, f"{FOLDER}/Fat.png": heavy, f"{FOLDER}/Thin.psd": psd,
                         f"{FOLDER}/Thin.png": pe.recompress(heavy), f"{FOLDER}/Snack2.png": numbered}, "Photoshop exports")
    _git(root, "reset", "-q", "--hard")         # the checkout land commits in (_plumb left it behind)
    out = tmp_path / "out"
    data = cg.check(str(root), push, bot, mode="on", out=str(out), workers=1)
    by = {c["card"]: c for c in data["cards"]}
    assert (by["Fat"]["case"], by["Snack"]["case"], by["Thin"]["writes"]) == ("pair", "numbered", [])
    assert by["Fat"]["told"] == by["Snack"]["told"] == [] and "ok" in by["Fat"]["r10"]   # R10 still checks the pushed PNG
    (fat,), (snack,) = by["Fat"]["writes"], by["Snack"]["writes"]
    assert (fat["path"], fat["kind"], snack["path"], snack["kind"]) == (f"{FOLDER}/Fat.png", "recompress", f"{FOLDER}/Snack2.png", "recompress")
    assert pe.same_png(heavy, (out / fat["file"]).read_bytes()) and pe.same_png(numbered, (out / snack["file"]).read_bytes())
    assert "ecompress" not in json.dumps(data["discord"])
    report = (out / "report.md").read_text()
    assert re.search(r"- Fat\.png recompressed losslessly \(\d+ KB -> \d+ KB, the same pixels\)\n", report)
    assert "- Snack2.png recompressed losslessly" in report
    landed = cg.land(str(out / "plan.json"), local=str(root))
    assert landed["sha"] and sorted(landed["landed"]) == ["Fat", "Snack"]
    assert _git(root, "log", "-1", "--format=%an %s", landed["sha"]) == f"AutoCroissant Card guards: 2 PNGs recompressed (push {push[:8]})"
    committed = subprocess.run(["git", "-C", str(root), "show", f"{landed['sha']}:{FOLDER}/Fat.png"], capture_output=True,
                               check=True).stdout
    assert len(committed) * 3 < len(heavy) and pe.same_png(heavy, committed)
    assert cg.check(str(root), landed["sha"], push, mode="on", out=str(tmp_path / "again"), workers=1)["cards"] == []
    told = cg.check(str(root), push, bot, mode="report", out=str(tmp_path / "report"), workers=1)
    assert [c["told"] for c in told["cards"] if c["card"] in ("Fat", "Snack")] == [[], []]
    assert "- Fat.png would be recompressed losslessly (" in (tmp_path / "report" / "report.md").read_text()


def test_a_recompression_rests_on_its_png_and_is_never_told(tmp_path, monkeypatch):
    """A recompressed PNG rests on that PNG alone (a later push of the PSD does not drop it) and is never told on
    Discord: not in report mode ("export it in Photoshop" is for a PNG not made), not when it did not land."""
    line = "recompressed losslessly (9.0 MB -> 2.0 MB, the same pixels)"
    job = cg.Job("X/Fat.psd", "pair", "", "s1", "X/Fat.png", png_new="p1", login="Alice", commit="a" * 40)
    result = cg.Result(job, writes=[{"path": "X/Fat.png", "file": "files/Fat.png", "sha": blob_sha(b"small"),
                                     "kind": "recompress", "line": line}])
    for mode in ("report", "on"):
        planned = cg.plan([result], "o/r", "e" * 40, "f" * 40, mode)
        assert planned["cards"][0]["told"] == [] and cg.discord_payload(planned, "o/r") == {} and cg.summary(planned, [])["writes"] == 1
    assert f"- Fat.png would be {line} with CARD_GUARDS=on" in cg.markdown(cg.plan([result], "o/r", "e" * 40, "f" * 40, "report"), [], [], [])
    assert f"- Fat.png {line}\n" in cg.markdown(cg.plan([result], "o/r", "e" * 40, "f" * 40, "on"), [], [], [])
    out = tmp_path / "plan"
    fat = _entry("Fat", case="pair")
    _write(out, fat, "X/Fat.png", "recompress", b"small", line)
    plan = _full_plan(tmp_path, [fat])
    now = datetime(2026, 10, 22, 18, 0, tzinfo=timezone.utc)
    hub = FakeHub({"X/Fat.psd": ("100644", "s1"), "X/Fat.png": ("100644", "p1")}, recent={"X/Fat.psd"})
    hub.install(monkeypatch)
    report = cg.land(plan, quiet_from="2026-10-22T17:00:00Z", now=lambda: now)
    assert report["landed"] == ["Fat"] and hub.commits[0][2].startswith("Card guards: 1 PNG recompressed (push ffffffff)")
    FakeHub({"X/Fat.psd": ("100644", "s1"), "X/Fat.png": ("100644", "theirs")}).install(monkeypatch)
    report = cg.land(plan)
    assert report["dropped"] == [["Fat", "superseded: X/Fat.png changed on the branch", "X/Fat.png"]]
    assert report["told"] == {"X/Fat.psd": []} and cg.finish(plan, report) == {}


def test_a_pushed_numbered_png_holds_back_only_its_recompression(tmp_path, monkeypatch):
    """Snack.psd pushed with a heavy Snack2.png (stale: the bot re-exports Snack.png). The author re-pushes Snack2.png in
    the window: its recompression is dropped, Snack.png's re-export (which never read Snack2.png) still lands."""
    out = tmp_path / "plan"
    snack2 = "X/Snack2.png"
    job = cg.Job("X/Snack.psd", "stale", "s0", "s1", "X/Snack.png", png_old="p1", pushed={snack2: "n2"}, login="Alice",
                 commit="a" * 40)
    result = cg.Result(job, export={"outcome": "re-exported", "changed_px": 5, "numbers": {}})
    entry = {"writes": result.writes}
    _write(out, entry, "X/Snack.png", "png", b"png", "re-exported")
    _write(out, entry, snack2, "recompress", b"small", "recompressed losslessly")
    (planned,) = cg.plan([result], "o/r", "e" * 40, "f" * 40, "on")["cards"]
    assert planned["expect"][snack2] == "n2" and planned["pushed"] == [snack2]
    plan = _full_plan(tmp_path, [planned])
    hub = FakeHub({"X/Snack.psd": ("100644", "s1"), "X/Snack.png": ("100644", "p1"), snack2: ("100644", "theirs")})
    hub.install(monkeypatch)
    report = cg.land(plan)
    assert report["landed"] == ["Snack"] and hub.commits[0][1] == ["X/Snack.png"]
    assert report["dropped"] == [["Snack", f"superseded: {snack2} changed on the branch", snack2]]
    assert report["told"] == {"X/Snack.psd": []}


def test_a_deleted_png_and_its_pushed_numbered_png_are_one_card(cards):
    """A push deleting Snack.png and changing Snack2.png: one job for Snack.psd (png-deleted, Snack2.png recompressed in
    it), not a second "numbered" one whose notes would replace the first's in the comment."""
    root, base, alice, bot = cards
    _git(root, "rm", "-q", f"{FOLDER}/Snack.png")
    _png(root, f"{FOLDER}/Snack2.png", card(colour=(0, 0, 255)))
    push = _commit(root, "Snack's PNGs", author="Alice")
    source = cg.Source(str(root))
    jobs = [job for job in cg.plan_jobs(source, bot, push, "actor")[0] if job.psd == f"{FOLDER}/Snack.psd"]
    assert [(job.case, list(job.pushed)) for job in jobs] == [("png-deleted", [f"{FOLDER}/Snack2.png"])]


def test_a_failed_recompression_is_a_comment_note(monkeypatch, tmp_path):
    """A PNG that cannot be read or recompressed: a comment note, never the card's error (Discord and the run's "card
    errors" would tell it); oxipng raising anything gives no recompression."""
    class Broken:
        def blob(self, sha):
            raise OSError("blob unreadable")
    job = cg.Job("X/Snack.psd", "numbered", "", "s1", login="Alice", commit="a" * 40, pushed={"X/Snack2.png": "n2"})
    result = cg.run_job(Broken(), job, None, None, str(tmp_path), "on")
    assert result.error == "" and not result.writes and result.told("on") == []
    assert result.notes == [("Snack2.png not recompressed (OSError: blob unreadable)", False)]

    class Oxipng:
        class PngError(Exception):
            pass

        class StripChunks:
            none = staticmethod(lambda: None)

        @staticmethod
        def optimize_from_memory(data, **options):
            raise RuntimeError("thread panicked")
    monkeypatch.setattr(pe, "_oxipng", lambda: Oxipng)
    heavy = photoshop_png(art("RGB", (40, 60)))
    assert pe.optimize(heavy) is None and pe.recompress(heavy) is None


def test_a_faint_beside_mismatch_note_names_the_case(cards, tmp_path):
    """Beside, current (no pixel past the threshold) but the mean is outside the bar: the note says the PSD is new and
    gives the mean, not "0.0% of pixels differ" nor "this push moved no pixel"."""
    import io
    root, base, alice, bot = cards
    psd = (root / FOLDER / "Bread.psd").read_bytes()
    image = Image.open(root / FOLDER / "Bread.png")
    faint = Image.eval(image.convert("RGB"), lambda v: max(v - 6, 0)).convert(image.mode)     # within BESIDE_CUT
    buf = io.BytesIO()
    faint.save(buf, "PNG")
    first = _plumb(root, {f"{FOLDER}/Faint.png": buf.getvalue()}, "the PNG first")
    push = _plumb(root, {f"{FOLDER}/Faint.psd": psd}, "then the PSD")
    data = cg.check(str(root), push, first, mode="on", out=str(tmp_path / "out"), workers=1)
    (entry,) = data["cards"]
    assert entry["case"] == "beside" and entry["export"]["outcome"] == "current"
    (note,) = [n for n in entry["notes"] if "does not match the PSD" in n]
    assert "mean difference" in note and "the PSD is new beside it" in note and "moved no pixel" not in note


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


def test_a_real_r11_fix_lands_for_a_pair(clone, tmp_path, monkeypatch):
    """Mini Doomer pushed with its Name layer typed "NAME" and its PNG re-saved: R11 renames it, land commits the PSD.
    (R11 is told until the owner adds it to AUTO_FIX; this pins the fix path a real PSD takes, as it will then.)"""
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
    told = cg.check(str(root), push, base, mode="on", out=str(tmp_path / "told"), workers=1)["cards"][0]
    squeezed = ["recompress"] if find_spec("oxipng") else []      # the re-saved PNG (level 1), by pyoxipng
    assert [w["kind"] for w in told["writes"]] == squeezed
    assert "layer names to correct: 'NAME' -> 'Name' (new)" in told["told"]                 # the owner's rule today
    monkeypatch.setattr(cg.standards, "AUTO_FIX", ("R9", "R11"))
    data = cg.check(str(root), push, base, mode="on", out=str(out), workers=1)
    (entry,) = data["cards"]
    assert entry["case"] == "pair" and [w["kind"] for w in entry["writes"]] == squeezed + ["psd"], entry
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
    if find_spec("oxipng") is None:
        pytest.skip("the 16 recompressions need pyoxipng, as the Action installs")
    source = cg.Source(str(clone))
    if not (source.commit("38c55bd3") and source.commit("07589cc7")):
        pytest.skip("the clone lacks the push")
    data = cg.check(str(clone), "07589cc7", "38c55bd3", out=str(tmp_path / "out"), workers=1)
    by = {c["card"]: c for c in data["cards"]}
    assert len(by) == 19
    assert {name for name, c in by.items() if c["export"].get("outcome") == "current"} == {"God Tamer", "Bubble", "12 Reactions2"}
    # nothing re-exported (the PNGs were pushed); 16 of Photoshop's exports recompressed (owner, 2026-10-23)
    assert all(w["kind"] == "recompress" for c in data["cards"] for w in c["writes"])
    recompressed = {name for name, c in by.items() if c["writes"]}
    assert len(recompressed) == 16 and {"Warcraftian Druid", "22 Negation", "Asteroid Field"} <= recompressed
    assert by["God Tamer"]["changes"] == ["added Star group", "Group 2 renamed 'Stars'", "Star/Delicate Flower moved into Star"]
    assert by["Bubble"]["changes"] == ["added Stars group", "starfish14 moved into Stars"]
    assert by["12 Reactions2"]["changes"] == ["removed Item shop copy"]
    assert data["discord"] == {}            # God Tamer's one issue was R3's layout, dropped by the owner (2026-10-23)


# ========================
# The Action
# ========================
def test_action_shaped_copy(tmp_path):
    """Exactly SPARSE_FILES (card-guards.yml's list), no config.py: --help works and the closure is that list; the card and
    font checks (its check job's since 2026-10-25) run from it too, without Discord."""
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
    modules = {m.replace(".", "/") + ".py" for m in json.loads(closure.replace("'", '"'))}
    assert modules == {f for f in cg.SPARSE_FILES if f.endswith(".py")}
    assert "requirements-cards.txt" in cg.SPARSE_FILES                         # what the Action installs
    from test_card_bake import ALLOWED_IMPORTS, third_party_imports
    extra = third_party_imports("import commands.card_guards, commands.txt2_repair", bot) - ALLOWED_IMPORTS
    assert not extra, f"card-guards.yml installs none of these: {sorted(extra)}"
    for module in ("card_check", "font_check"):
        done = subprocess.run([executable, "-m", f"commands.{module}", "--help"], cwd=bot, capture_output=True, text=True,
                              timeout=120)
        assert done.returncode == 0 and done.stdout.startswith(f"usage: python -m commands.{module}"), done.stderr
        code = f"import sys, commands.{module}; print(sorted(m for m in ('config', 'discord') if m in sys.modules))"
        done = subprocess.run([executable, "-c", code], cwd=bot, capture_output=True, text=True, timeout=120)
        assert done.returncode == 0 and done.stdout.strip() == "[]", done.stderr


def test_a_pool_worker_takes_the_token(monkeypatch):
    """A spawned worker has no main(): its blob fetches need GITHUB_TOKEN too (else 60 requests/hour)."""
    import commands.github_api as api
    monkeypatch.setattr(api, "GIT_TOKEN", "")
    monkeypatch.setattr(api, "HEADERS", {})
    monkeypatch.setenv("GITHUB_TOKEN", "t0k")
    monkeypatch.setattr(cg, "_worker", {})
    cg._init_worker("clone", "o/r", None, type("Vocab", (), {"parser": lambda self: None})(), {}, "")
    assert api.HEADERS["Authorization"] == "token t0k"


def test_guard_modules_stay_uncompiled():
    """Not in setup.py's INCLUDE_FILES: the Action runs them as pure Python (the change-control skill's step 5)."""
    setup = (REPO / "setup.py").read_text(encoding="utf-8")
    for module in ("card_guards", "png_export", "card_standards", "psd_diff"):
        assert f"commands/{module}.py" not in setup


def _workflow(clone) -> Path:
    site = Path(os.path.expanduser(os.environ.get("AC_SITE_SRC") or str(clone)))       # a TTSCardMaker checkout
    path = WORKFLOW or (site / ".github" / "workflows" / "card-guards.yml")
    if not path.exists():
        pytest.skip(f"no {path} (set CARD_GUARDS_YML to a TTSCardMaker checkout's file, or AC_SITE_SRC to the checkout)")
    return path


SYNC = "workflow run card-sync.yml -f dry_run=false"   # Follow up's card sync start (it bakes, builds and saves)
ENVIRONMENT = "card-guards-quiet"               # the post job's environment: its 30-minute wait timer is the quiet window


def _guard_steps(clone) -> tuple:
    """card-guards.yml's three jobs read as one: (doc, every named step by name, the jobs' env merged: TOLD, STREAK)."""
    yaml = pytest.importorskip("yaml")
    doc = yaml.safe_load(_workflow(clone).read_text(encoding="utf-8"))
    jobs = doc["jobs"].values()
    return (doc, {s["name"]: s for job in jobs for s in job.get("steps", []) if s.get("name")},
            {k: v for job in jobs for k, v in job.get("env", {}).items()})


def test_workflow_shape(clone):
    """card-guards.yml: push on PSDs and PNGs behind CARD_GUARDS, dry dispatches by default, a run per push, the bot's
    commits skipped, the sparse list = SPARSE_FILES, requirements-cards.txt's install (the fork's one pin) in both jobs that
    run the guards, as the other card workflows have it (one cache entry), secrets by name only, no ping, no fonts (R7 is
    font-check.yml's). Three jobs (2026-10-24): check (ends after the comment), post (the quiet window: a push's first
    attempt waits in the environment's timer; only with something to land or tell), finish (card errors, the alert)."""
    path = _workflow(clone)
    text = path.read_text(encoding="utf-8")
    doc, steps_by_name, env = _guard_steps(clone)
    on = doc[True]
    assert on["push"] == {"branches": ["main"], "paths": ["**/*.psd", "**/*.PSD", "**/*.png", "**/*.PNG"]}
    inputs = on["workflow_dispatch"]["inputs"]
    assert inputs["dry_run"]["default"] is True and inputs["all"]["default"] is False and set(inputs) == {"dry_run", "sha", "cards", "all"}
    assert doc["concurrency"]["group"] == ("${{ github.event_name == 'push' && format('card-guards-{0}', github.sha) || "
                                           "format('card-guards-{0}', github.run_id) }}") and doc["concurrency"]["cancel-in-progress"] is False
    assert list(doc["jobs"]) == ["check", "post", "finish"]
    job, post, finish = doc["jobs"]["check"], doc["jobs"]["post"], doc["jobs"]["finish"]
    assert "github.actor != 'AutoCroissant'" in job["if"] and "github.actor != 'github-actions[bot]'" in job["if"]
    assert "vars.CARD_GUARDS == 'report' || vars.CARD_GUARDS == 'on'" in job["if"]
    assert job["env"]["MODE"] == ("${{ github.event_name == 'push' && (vars.CARD_GUARDS || 'off') || "
                                  "((inputs.dry_run || inputs.all) && 'dry' || 'on') }}")
    # the quiet window: no runner sleeps for it. A push's first attempt waits in the environment; a dispatch (lands at
    # once) and a re-run (the window counts from the push) take none. Without the timer, Land sleeps it out (--quiet-from).
    assert post["needs"] == "check" and finish["needs"] == ["check", "post"]
    assert post["environment"] == ("${{ (github.event_name == 'push' && github.run_attempt == '1') && '" + ENVIRONMENT
                                   + "' || '' }}")
    assert post["if"] == ("!cancelled() && needs.check.outputs.checked == 'success' && needs.check.outputs.land == 'true' && "
                          "(needs.check.outputs.mode == 'report' || needs.check.outputs.mode == 'on')")
    assert finish["if"] == "always() && needs.check.result != 'skipped'"          # nothing when the guards are off
    assert job["timeout-minutes"] == 100 and post["timeout-minutes"] == 75
    assert "environment" not in job and "environment" not in finish
    assert job["outputs"] == {"mode": "${{ steps.start.outputs.mode }}", "quiet_from": "${{ steps.start.outputs.quiet_from }}",
                              "started": "${{ steps.start.outputs.started }}", "checked": "${{ steps.check.outcome }}",
                              "comment": "${{ steps.comment.outputs.id }}", "told": "${{ steps.comment.outputs.told }}",
                              "land": "${{ steps.plan.outputs.land }}", "card_errors": "${{ steps.plan.outputs.card_errors }}",
                              "bot_sha": "${{ steps.bot.outputs.sha }}"}
    assert post["outputs"] == {"posted": "${{ steps.tell.outputs.posted }}"}
    assert post["env"]["QUIET_FROM"] == "${{ needs.check.outputs.quiet_from }}" and post["env"]["MODE"] == "${{ needs.check.outputs.mode }}"
    assert post["env"]["TOLD"] == job["env"]["TOLD"] == finish["env"]["TOLD"]
    for one in (job, post):
        bot = next(s for s in one["steps"] if s.get("with", {}).get("repository") == "bobberlington/AutoCroissant")
        assert bot["with"]["sparse-checkout"].split() == ["/" + f for f in cg.SPARSE_FILES]
        assert bot["with"].get("ref") == (None if one is job else "${{ needs.check.outputs.bot_sha }}")   # plan and land: one bot
    sha = next(s for s in job["steps"] if s.get("id") == "bot")
    assert sha["run"] == 'echo "sha=$(git -C bot rev-parse HEAD)" >> "$GITHUB_OUTPUT"'
    for one in (job, post, finish):                                            # a copied step runs twice unseen
        named = [s["name"] for s in one["steps"] if s.get("name")]
        assert len(named) == len(set(named)), named
    cards = next(s for s in job["steps"] if s.get("uses", "").startswith("actions/checkout") and "repository" not in s.get("with", {}))
    assert cards["with"]["filter"] == "blob:none" and cards["with"]["fetch-depth"] == 0
    assert not [s for s in post["steps"] + finish["steps"] if s.get("uses", "").startswith("actions/checkout") and "repository" not in s.get("with", {})]
    assert post["env"]["GH_REPO"] == "${{ github.repository }}"               # gh workflow run without the card repo
    from test_card_bake import FORK_KEY, FORK_WORKFLOWS, fork_install
    others = [path.parent / name for name in FORK_WORKFLOWS if name != "card-guards.yml" and (path.parent / name).is_file()]
    keys = {p.name: fork_install(p) for p in [path, *others]}                  # one pin: requirements-cards.txt's
    assert set(keys.values()) == {FORK_KEY}                                    # one entry for the six, on 3.14
    assert "pyoxipng==" in (REPO / "requirements-cards.txt").read_text(encoding="utf-8")   # png_export's recompression
    assert "MDW/Fonts" not in text and "card-fonts-" not in text and "--fonts" not in text   # R7 is font-check.yml's
    names = [s.get("name") for s in job["steps"]]
    assert names.index("Start") < names.index("No push run in its window") < names.index("Check") \
        < names.index("Comment on the commit") < names.index("Plan for the post job")
    plan = job["steps"][names.index("Plan for the post job")]
    assert plan["id"] == "plan" and plan["if"] == "always() && steps.check.outcome == 'success'"
    hand = job["steps"][names.index("Plan for the post job") + 1]
    assert hand["uses"].startswith("actions/upload-artifact") and hand["with"]["name"] == "card-guards" and hand["if"] == "always()"
    post_names = [s.get("name") for s in post["steps"]]
    take = post["steps"][1]
    assert take["uses"].startswith("actions/download-artifact@") and take["with"] == {"name": "card-guards", "path": "${{ runner.temp }}/guards"}
    assert post_names.index("Work dir") < post_names.index("Told before") < post_names.index("Land") < post_names.index("Follow up") \
        < post_names.index("Tell Discord")
    assert [s.get("name") for s in finish["steps"] if s.get("name")] == ["Fail on card errors", "Tell Discord the run failed"]
    steps = steps_by_name
    tell = steps["Tell Discord"]
    assert tell["env"]["WEBHOOK"] == "${{ secrets.DISCORD_STANDARDS_WEBHOOK }}" and "discord-final.json" in tell["run"]
    assert tell["if"] == "always() && steps.told.outputs.told != 'true'"
    assert steps["Told before"]["env"]["CHECK_TOLD"] == "${{ needs.check.outputs.told }}" and "if" not in steps["Told before"]
    assert '"$WEBHOOK"' in tell["run"] and "echo \"$WEBHOOK" not in text and "set -x" not in text
    land = steps["Land"]
    # Land never waits on the Comment step's success (a failed comment must not drop the writes): the post job runs
    # whatever the check job's result, once Check passed
    assert "if" not in land and "--quiet-from" in land["run"] and "--dry-run" in land["run"]
    assert "discord.json" not in tell["run"].replace("discord-final.json", "")      # never check's early post
    assert steps["Fail on card errors"]["env"]["ERRORS"] == "${{ needs.check.outputs.card_errors }}"
    alert = steps["Tell Discord the run failed"]
    # a re-run of a push already told (the comment's mark) never alerts: "one post per push"
    assert alert["if"] == ("(failure() || contains(needs.*.result, 'failure') || contains(needs.*.result, 'cancelled')) && "
                           "github.event_name == 'push' && needs.post.outputs.posted != 'true' && needs.check.outputs.told != 'true'")
    assert alert["env"]["WEBHOOK"] == "${{ secrets.DISCORD_STANDARDS_WEBHOOK }}" and "allowed_mentions: {parse: []}" in alert["run"]
    assert alert["env"]["RESULTS"] == "${{ needs.check.result }} ${{ needs.post.result }}"
    assert '{"check": %d, "post": %d}' % (job["timeout-minutes"], post["timeout-minutes"]) in alert["run"]
    # the streak's mark: an artifact left only when the alert told (or stayed quiet in a told streak)
    streak = finish["steps"][-1]
    assert streak["uses"].startswith("actions/upload-artifact") and streak["if"] == "always() && steps.alert.outputs.streak == 'true'"
    assert streak["with"]["name"] == "${{ env.STREAK }}" and env["STREAK"] and alert["id"] == "alert"
    refuse = steps["No push run in its window"]
    assert refuse["if"] == "github.event_name == 'workflow_dispatch' && env.MODE == 'on'"
    assert "<@&" not in text and "ROLE" not in text and "DISCORD_FONT_WEBHOOK" not in text
    assert set(re.findall(r"secrets\.(\w+)", text)) == {"GITHUB_TOKEN", "DISCORD_STANDARDS_WEBHOOK"}
    check = steps["Check"]
    for flag in ("--clone", "--repo", "--after", "--author", "--out", "--mode", "--before", "--cards", "--whole-tree"):
        assert flag in check["run"]
    assert check["env"]["AUTHOR"] == "${{ github.event_name == 'push' && github.actor || '' }}"   # never the dispatcher
    # a job's env may read only github, inputs, matrix, needs, secrets, strategy and vars: GitHub rejects the whole file
    assert not any("runner." in str(value) for one in (job, post, finish) for value in one["env"].values())
    assert 'OUT=$RUNNER_TEMP/guards' in job["steps"][0]["run"] and 'OUT=$RUNNER_TEMP/guards' in post["steps"][0]["run"]
    follow = steps["Follow up"]
    assert "report-landed.md" in follow["run"] and follow["if"] == "always() && steps.land.outputs.ran == 'true'"
    assert follow["id"] == "follow" and tell["env"]["REFRESH"] == "${{ steps.follow.outputs.refresh }}"
    assert follow["env"]["TOLD_BEFORE"] == "${{ steps.told.outputs.told }}"
    # a re-run uploads again: the earlier attempt's artifact of the same name is replaced, not a 409
    uploads = [s for s in job["steps"] + post["steps"] + finish["steps"] if s.get("uses", "").startswith("actions/upload-artifact")]
    assert len(uploads) == 3 and all(s["with"]["overwrite"] is True for s in uploads)



def _step_if(expr, job_status="success", results=(), **context):
    """A step's `if` as the runner reads it: failure() and cancelled() see this job's own status only (actions/runner
    FailureFunction.cs, CancelledFunction.cs; a job starts Succeeded), needs.*.result the needed jobs'. Context paths
    not given read as null."""
    tokens = re.findall(r"'[^']*'|contains\(needs\.\*\.result, '\w+'\)|\w+\(\)|[\w.]+|&&|\|\||==|!=|!|[()]|\s+", expr)
    assert "".join(tokens) == expr, expr
    python = []
    for tok in (t for t in tokens if not t.isspace()):
        if tok.startswith("'"):
            python.append(repr(tok[1:-1]))
        elif tok.startswith("contains("):
            python.append(f"({tok.split(chr(39))[1]!r} in {list(results)!r})")
        elif tok.endswith("()"):
            python.append(repr({"failure()": job_status == "failure", "cancelled()": job_status == "cancelled",
                                "success()": job_status == "success", "always()": True}[tok]))
        else:
            python.append({"&&": "and", "||": "or", "!": "not"}.get(tok, tok if tok in ("==", "!=", "(", ")")
                                                                      else repr(context.get(tok))))
    return eval(" ".join(python))


def test_failure_alert_reads_the_needed_jobs(clone):
    """The alert is a step of the finish job: its failure() sees only finish's own steps, so a check or post job that
    failed or timed out must be read from needs.*.result (2026-10-24 review: the run-failed alert never fired)."""
    alert = _guard_steps(clone)[1]["Tell Discord the run failed"]["if"]
    push = {"github.event_name": "push"}

    def fires(check, post, job_status="success", **context):
        return _step_if(alert, job_status, (check, post), **{**push, **context})

    assert fires("failure", "skipped")                  # not checked: the post job never ran
    assert fires("success", "failure")                  # the post job broke before (or after) its post
    assert fires("cancelled", "skipped") and fires("success", "cancelled")       # a timeout; the step tells it from a cancel
    assert fires("success", "skipped", "failure")       # this job's own card-errors step
    assert not fires("success", "success") and not fires("success", "skipped")   # green, or nothing to land or tell
    assert not fires("success", "failure", **{"needs.post.outputs.posted": "true"})   # the post went out: it told
    assert not fires("failure", "skipped", **{"needs.check.outputs.told": "true"})    # a re-run of a told push
    assert not fires("failure", "skipped", **{"github.event_name": "workflow_dispatch"})
    # the old condition, read the same way, missed both failures
    old = ("(failure() || cancelled()) && github.event_name == 'push' && needs.post.outputs.posted != 'true' && "
           "needs.check.outputs.told != 'true'")
    assert not _step_if(old, "success", ("failure", "skipped"), **push)
    assert not _step_if(old, "success", ("success", "failure"), **push)

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
    # FAKE_FAIL_TIMES: only the first n matching calls fail (a transient error)
    db["failed"] = db.get("failed", 0) + 1
    if db["failed"] <= int(os.environ.get("FAKE_FAIL_TIMES", "1000")):
        json.dump(db, open(state, "w"))
        sys.exit(1)
if args[:2] == ["workflow", "run"]:
    pass
elif args[:2] == ["run", "list"]:
    out = jq(json.loads(os.environ["FAKE_RUNS"])) if "FAKE_RUNS" in os.environ else os.environ.get("FAKE_LAST", "")
elif args[0] == "api":
    method = args[args.index("--method") + 1] if "--method" in args else "GET"
    path = next(a for a in args[1:] if a.startswith("repos/"))
    if path.endswith("/jobs"):                  # the attempt's jobs: FAKE_JOBS, the API's {"jobs": [...]}
        out = jq(json.loads(os.environ.get("FAKE_JOBS", '{"jobs": []}')))
    elif path.endswith("/artifacts"):
        marked = path.split("/")[-2] in os.environ.get("FAKE_MARKED", "").split(",")
        out = jq({"artifacts": [{"name": "card-guards"}] + ([{"name": os.environ["STREAK"]}] if marked else [])})
    elif method == "POST":
        db["next"] += 1
        db["comments"][str(db["next"])] = json.load(sys.stdin)["body"]
        out = str(db["next"])
    elif method == "PATCH":
        db["comments"][path.rsplit("/", 1)[1]] = json.load(sys.stdin)["body"]
    elif path.endswith("/comments"):
        listed = [{"id": int(i), "user": {"login": "github-actions[bot]"}, "body": body} for i, body in db["comments"].items()]
        out = jq(listed) if "--jq" in args else json.dumps(listed)
    else:
        out = db["comments"][path.rsplit("/", 1)[1]]
json.dump(db, open(state, "w"))
print(out)
"""
FAKE_CURL = r"""#!/bin/sh
for a in "$@"; do case "$prev" in --data) jq -c . "${a#@}" >> "$FAKE_POSTS";; esac; prev="$a"; done
exit "${FAKE_CURL_STATUS:-0}"
"""


def _run_steps(steps, names, env, outputs, jobs=None):
    """Run the named steps' shell as card-guards.yml would (bash -e -o pipefail); outputs[name] = its GITHUB_OUTPUT. A
    `needs.<job>.outputs.<name>` env reads the step output the job's outputs name (jobs: the doc's), when that step ran
    here; else the env given stays."""
    for name in names:
        step = steps[name]
        out = Path(env["OUT"]) / f"{name}.output"
        out.write_text("")
        step_env = {**env, **{k: v for k, v in step.get("env", {}).items() if "${{" not in str(v)}, "GITHUB_OUTPUT": str(out)}
        for key, value in step.get("env", {}).items():
            needed = re.fullmatch(r"\$\{\{ needs\.(\w+)\.outputs\.(\w+) \}\}", str(value))
            if needed and jobs:
                value = jobs[needed.group(1)].get("outputs", {}).get(needed.group(2), "")
                match = re.fullmatch(r"\$\{\{ steps\.(\w+)\.outputs\.(\w+) \}\}", value)
                if match and match.group(1) in outputs:
                    step_env[key] = outputs[match.group(1)].get(match.group(2), "")
                continue
            match = re.fullmatch(r"\$\{\{ steps\.(\w+)\.outputs\.(\w+) \}\}", str(value))
            if match:
                step_env[key] = outputs.get(match.group(1), {}).get(match.group(2), "")
        done = subprocess.run(["bash", "--noprofile", "--norc", "-eo", "pipefail", "-c", step["run"]], env=step_env,
                              capture_output=True, text=True, timeout=60)
        assert done.returncode == 0, (name, done.stdout, done.stderr)
        outputs[name.split()[0].lower()] = dict(line.split("=", 1) for line in out.read_text().splitlines() if "=" in line)
    return outputs


def test_workflow_posts_once_and_a_rerun_edits_its_comment(clone, tmp_path):
    """The Comment, Told before, Follow up and Tell Discord steps' shell against a fake gh and curl: the first run comments,
    posts the settled post (discord-final.json) and marks the comment; a re-run edits that comment, keeps the mark and
    does not post again, and so does a re-run of the post job alone (its Told before looks the mark up itself: the
    check job's outputs are its first attempt's)."""
    yaml = pytest.importorskip("yaml")
    if not shutil.which("jq"):
        pytest.skip("no jq")
    doc, steps, jenv = _guard_steps(clone)
    bin_dir = _fake_tools(tmp_path)
    out = tmp_path / "guards"
    out.mkdir()
    (out / "summary.json").write_text(json.dumps({"after": "f" * 40, "cards": 1, "told": 1, "comment": 1, "writes": 0,
                                                  "errors": 0, "card_errors": 0}))
    (out / "plan.json").write_text(json.dumps({"after": "f" * 40}))
    (out / "report.md").write_text("## Card guards\n\nAlice's card.\n")
    (out / "report-landed.md").write_text("## Card guards\n\nAfter the window.\n")
    (out / "discord.json").write_text(json.dumps({"content": "early"}))
    (out / "discord-final.json").write_text(json.dumps({"content": "settled"}))
    env = {"PATH": f"{bin_dir}:{os.environ['PATH']}", "OUT": str(out), "TOLD": jenv["TOLD"], "MODE": "on",
           "GITHUB_REPOSITORY": "o/cards", "WEBHOOK": "https://discord.invalid/hook", "FAKE_STATE": str(tmp_path / "gh.json"),
           "FAKE_POSTS": str(tmp_path / "posts.txt"), "GITHUB_STEP_SUMMARY": str(tmp_path / "summary.md")}
    order = ["Comment on the commit", "Told before", "Follow up", "Tell Discord"]
    first = _run_steps(steps, order, env, {"land": {"ran": "true"}}, doc["jobs"])
    state = json.loads((tmp_path / "gh.json").read_text())
    (comment_id, body), = state["comments"].items()
    assert first["comment"] == {"id": comment_id} and first["tell"] == {"posted": "true"} and first["told"] == {}
    assert body.startswith("## Card guards\n\nAfter the window.") and body.rstrip().endswith(jenv["TOLD"])
    assert json.loads((tmp_path / "posts.txt").read_text().strip()) == {"content": "settled"}
    again = _run_steps(steps, order[:3], env, {"land": {"ran": "true"}}, doc["jobs"])   # a re-run of every job
    assert again["comment"] == {"told": "true", "id": comment_id} and again["told"] == {"told": "true"}
    assert "steps.told.outputs.told != 'true'" in steps["Tell Discord"]["if"]       # so Tell Discord is skipped
    alone = _run_steps(steps, ["Told before"], {**env, "GITHUB_RUN_ATTEMPT": "2"}, {}, doc["jobs"])   # the post job alone
    assert alone["told"] == {"told": "true"}
    fresh = _run_steps(steps, ["Told before"], {**env, "GITHUB_RUN_ATTEMPT": "1"}, {}, doc["jobs"])   # a first attempt: no look
    assert fresh["told"] == {}
    state = json.loads((tmp_path / "gh.json").read_text())
    assert list(state["comments"]) == [comment_id] and state["comments"][comment_id].rstrip().endswith(jenv["TOLD"])
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
    doc, steps, jenv = _guard_steps(clone)
    posts, runner = tmp_path / "posts.txt", tmp_path / "runner"
    env = {"PATH": f"{_fake_tools(tmp_path)}:{os.environ['PATH']}", "OUT": str(tmp_path), "GITHUB_REPOSITORY": "o/cards",
           "WEBHOOK": "https://discord.invalid/hook", "FAKE_STATE": str(tmp_path / "gh.json"), "FAKE_POSTS": str(posts),
           "GITHUB_STEP_SUMMARY": str(tmp_path / "summary.md"), "STREAK": jenv["STREAK"], "RUNNER_TEMP": str(runner),
           "GITHUB_RUN_ID": "7", "GITHUB_RUN_ATTEMPT": "1", "GITHUB_ACTOR": "Alice", "CHECKED": "failure",
           "RUN": "https://github.com/o/cards/actions/runs/7", "TOLD": jenv["TOLD"], "GITHUB_SHA": "f" * 40}
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
            "content": f"Card guards: the push by Alice {what} (the next failures stay quiet until a push run succeeds or "
                       "posts its issues). "
                       "https://github.com/o/cards/actions/runs/7", "allowed_mentions": {"parse": []}, "flags": 4}


def test_a_rerun_that_fails_in_check_finds_the_told_mark(clone, tmp_path):
    """Attempt 1 posted the card lines, marked the comment and went red (a card error, say); the re-run fails in Check,
    so the Comment step does not run and its `told` is empty. The alert looks the comment's mark up itself: quiet, and no
    streak mark (the push was told, not a failure). This is the review's case: the last push run green, no artifact. It
    is quiet too when the comments stay unreadable (as the Comment step counts them). With no mark it tells as before;
    a first attempt does not look."""
    yaml = pytest.importorskip("yaml")
    if not shutil.which("jq"):
        pytest.skip("no jq")
    doc, steps, jenv = _guard_steps(clone)
    told = jenv["TOLD"]
    posts, runner, state = tmp_path / "posts.txt", tmp_path / "runner", tmp_path / "gh.json"
    env = {"PATH": f"{_fake_tools(tmp_path)}:{os.environ['PATH']}", "OUT": str(tmp_path), "GITHUB_REPOSITORY": "o/cards",
           "WEBHOOK": "https://discord.invalid/hook", "FAKE_STATE": str(state), "FAKE_POSTS": str(posts),
           "GITHUB_STEP_SUMMARY": str(tmp_path / "summary.md"), "STREAK": jenv["STREAK"], "RUNNER_TEMP": str(runner),
           "GITHUB_RUN_ID": "7", "GITHUB_RUN_ATTEMPT": "2", "GITHUB_ACTOR": "alice", "CHECKED": "failure", "TOLD": told,
           "RUN": "https://github.com/o/cards/actions/runs/7", "GITHUB_SHA": "f" * 40, "FAKE_MARKED": "",
           "FAKE_RUNS": json.dumps([{"databaseId": 6, "conclusion": "success"}])}
    marked = {"101": f"## Card guards\n\nAlice's card.\n\n{told}\n"}
    cases = ((marked, {}, 0, ""),                                       # attempt 1 told: quiet, no streak mark
             (marked, {"FAKE_FAIL": "--paginate"}, 0, ""),              # unreadable: counts as told
             ({"101": "## Card guards\n\nAlice's card.\n"}, {}, 1, "true"),   # attempt 1 commented, never told
             ({}, {}, 1, "true"),                                       # attempt 1 failed in Check too
             (marked, {"GITHUB_RUN_ATTEMPT": "1"}, 1, "true"))          # a first attempt does not look
    for comments, extra, sent, streak in cases:
        posts.write_text("")
        shutil.rmtree(runner, ignore_errors=True)
        state.write_text(json.dumps({"comments": comments, "next": 101, "log": []}))
        out = _run_steps(steps, ["Tell Discord the run failed"], {**env, **extra}, {})["tell"]
        lines = [json.loads(l) for l in posts.read_text().splitlines() if l.strip()]
        assert len(lines) == sent and out.get("streak", "") == streak, (comments, extra, lines, out)
        assert not lines or "the push by alice was not guarded" in lines[0]["content"]
        looked = [c for c in json.loads(state.read_text())["log"] if c.endswith("/comments --paginate")]
        assert looked == ([] if extra.get("GITHUB_RUN_ATTEMPT") == "1" else
                          [f"api repos/o/cards/commits/{'f' * 40}/comments --paginate"] * (3 if "FAKE_FAIL" in extra else 1))
    lookups = {name: re.search(r"guards_comments\(\) \{.*?\n *\}\n", steps[name]["run"], re.S).group(0).strip()
               for name in ("Comment on the commit", "Tell Discord the run failed")}
    assert len(set(lookups.values())) == 1                              # one lookup, one fail-safe


def test_workflow_tells_only_the_settled_post(clone, tmp_path):
    """Tell Discord posts discord-final.json only: Land ran but settled nothing, or did not run (the post job runs only
    with something to tell) -> the step fails (the run-failed message goes out instead), nothing posted. The Comment step
    goes up for comment-only findings (R10 alone: told 0, comment 1), and the plan holds the post job back (nothing to
    land or tell). Fail on card errors fails on the check job's card_errors only."""
    yaml = pytest.importorskip("yaml")
    if not shutil.which("jq"):
        pytest.skip("no jq")
    doc, steps, jenv = _guard_steps(clone)
    bin_dir = _fake_tools(tmp_path)
    out = tmp_path / "guards"
    out.mkdir()
    (out / "summary.json").write_text(json.dumps({"after": "f" * 40, "cards": 1, "told": 0, "comment": 1, "writes": 0,
                                                  "errors": 0, "card_errors": 0}))
    (out / "plan.json").write_text(json.dumps({"after": "f" * 40}))
    (out / "report.md").write_text("## Card guards\n\n- Lonely.png does not match its PSD\n")
    (out / "discord.json").write_text(json.dumps({"content": "early"}))
    posts = tmp_path / "posts.txt"
    env = {"PATH": f"{bin_dir}:{os.environ['PATH']}", "OUT": str(out), "TOLD": jenv["TOLD"], "MODE": "on",
           "GITHUB_REPOSITORY": "o/cards", "WEBHOOK": "https://discord.invalid/hook", "FAKE_STATE": str(tmp_path / "gh.json"),
           "FAKE_POSTS": str(posts), "GITHUB_STEP_SUMMARY": str(tmp_path / "summary.md")}
    done = _run_steps(steps, ["Comment on the commit"], env, {})
    assert done["comment"]["id"] and json.loads((tmp_path / "gh.json").read_text())["comments"]
    tell = steps["Tell Discord"]["run"]
    for ran, code in (("true", 1), ("", 1)):
        result = subprocess.run(["bash", "--noprofile", "--norc", "-eo", "pipefail", "-c", tell], capture_output=True, text=True,
                                env={**env, "LAND_RAN": ran, "GITHUB_OUTPUT": str(out / "tell.output")}, timeout=60)
        assert result.returncode == code, (ran, result.stdout, result.stderr)
    assert not posts.exists() or posts.read_text() == ""
    planned = _run_steps(steps, ["Plan for the post job"], env, {})["plan"]
    assert planned == {"card_errors": "0"}                                      # no land=true: no post job
    fail = steps["Fail on card errors"]["run"]
    for errors, code in (("0", 0), ("", 0), ("2", 1)):
        result = subprocess.run(["bash", "--noprofile", "--norc", "-eo", "pipefail", "-c", fail], capture_output=True, text=True,
                                env={**env, "ERRORS": errors}, timeout=60)
        assert result.returncode == code, (errors, result.stdout)


def _fake_tools(tmp_path) -> Path:
    """gh, curl and a sleep that does not wait (the comment lookup's retries)."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    (bin_dir / "gh").write_text(FAKE_GH)
    (bin_dir / "curl").write_text(FAKE_CURL)
    (bin_dir / "sleep").write_text("#!/bin/sh\necho \"$1\" >> \"${FAKE_SLEPT:-/dev/null}\"\n")
    for tool in ("gh", "curl", "sleep"):
        (bin_dir / tool).chmod(0o755)
    return bin_dir


def _shell(step, env, cwd=None):
    out = Path(env["GITHUB_OUTPUT"])
    out.write_text("")
    done = subprocess.run(["bash", "--noprofile", "--norc", "-eo", "pipefail", "-c", step["run"]], env=env, cwd=cwd,
                          capture_output=True, text=True, timeout=60)
    return done, dict(line.split("=", 1) for line in out.read_text().splitlines() if "=" in line)


def test_workflow_comment_lookup_never_posts_twice(clone, tmp_path):
    """The Comment step reads the guards' comments in one lookup, whole (no pipe into head, which can kill gh under
    pipefail), with their bodies: it edits the oldest, and the TOLD mark on any of them counts (Tell Discord's own
    comment). The lookup is tried 3 times. One that still fails on a re-run, or in a dispatch (its commit's push run may
    have told), counts as told: no new comment, the step fails, and Tell Discord and the failure alert stay quiet (their
    `if`). Only a push run's first attempt comments anyway (no earlier comment of its own can exist). The id is out before
    the edit, so a failed edit still leaves the comment to Follow up and the mark."""
    yaml = pytest.importorskip("yaml")
    if not shutil.which("jq"):
        pytest.skip("no jq")
    doc, steps, jenv = _guard_steps(clone)
    step = steps["Comment on the commit"]
    out = tmp_path / "guards"
    out.mkdir()
    (out / "summary.json").write_text(json.dumps({"comment": 1}))
    (out / "plan.json").write_text(json.dumps({"after": "f" * 40}))
    (out / "report.md").write_text("## Card guards\n\nAgain.\n")
    state, slept = tmp_path / "gh.json", tmp_path / "slept.txt"
    told = jenv["TOLD"]
    env = {"PATH": f"{_fake_tools(tmp_path)}:{os.environ['PATH']}", "OUT": str(out), "TOLD": told, "MODE": "on",
           "GITHUB_REPOSITORY": "o/cards", "FAKE_STATE": str(state), "GITHUB_OUTPUT": str(out / "comment.output"),
           "GITHUB_RUN_ATTEMPT": "2", "GITHUB_EVENT_NAME": "push", "FAKE_SLEPT": str(slept)}
    seeded = {"101": f"## Card guards\n\nFirst.\n\n{told}\n", "102": "## Card guards\n\nOld.\n"}

    def seed(comments=seeded):
        state.write_text(json.dumps({"comments": comments, "next": 102, "log": []}))
        slept.write_text("")
    for comments in (seeded, {"101": "## Card guards\n\nFirst.\n", "102": f"## Card guards\n\nTell's own.\n\n{told}\n"}):
        seed(comments)
        done, outputs = _shell(step, env)
        assert done.returncode == 0 and outputs == {"told": "true", "id": "101"}, done.stderr
        after = json.loads(state.read_text())["comments"]
        assert set(after) == {"101", "102"} and after["101"].startswith("## Card guards\n\nAgain.") and told in after["101"]
        assert slept.read_text().split() == ["0"]
    seed({"101": "## Card guards\n\nFirst.\n"})                     # attempt 1 did not tell: no mark, Tell Discord posts
    done, outputs = _shell(step, {**env, "FAKE_FAIL": "--paginate", "FAKE_FAIL_TIMES": "1"})      # read on the second try
    assert done.returncode == 0 and outputs == {"id": "101"} and slept.read_text().split() == ["0", "10"], done.stderr
    for extra in ({}, {"GITHUB_RUN_ATTEMPT": "1", "GITHUB_EVENT_NAME": "workflow_dispatch"}):     # a re-run, a dispatch
        seed()
        done, outputs = _shell(step, {**env, **extra, "FAKE_FAIL": "--paginate"})
        assert done.returncode == 1 and outputs == {"told": "true"}, (extra, done.stderr)
        assert "could not be read" in done.stdout and slept.read_text().split() == ["0", "10", "30"]
        assert json.loads(state.read_text())["comments"] == seeded
    seed()
    done, outputs = _shell(step, {**env, "FAKE_FAIL": "--method PATCH"})                       # the edit fails
    assert done.returncode == 1 and outputs == {"told": "true", "id": "101"}
    assert "needs.check.outputs.told != 'true'" in steps["Tell Discord the run failed"]["if"]
    assert steps["Told before"]["env"]["CHECK_TOLD"] == "${{ needs.check.outputs.told }}"      # Tell Discord's `if` reads it
    assert "steps.told.outputs.told != 'true'" in steps["Tell Discord"]["if"]
    state.unlink()
    done, outputs = _shell(step, {**env, "FAKE_FAIL": "--paginate", "GITHUB_RUN_ATTEMPT": "1"})
    assert done.returncode == 0 and outputs == {"id": "101"} and list(json.loads(state.read_text())["comments"]) == ["101"]


def test_workflow_tell_marks_a_comment_of_its_own(clone, tmp_path):
    """The Comment step failed (no id): Tell Discord still posts, then puts up a comment of its own (the settled body)
    carrying the TOLD mark, which a re-run's lookup finds, so the re-run does not post again. When that comment cannot
    be made either, it warns that a re-run would post again."""
    yaml = pytest.importorskip("yaml")
    if not shutil.which("jq"):
        pytest.skip("no jq")
    doc, steps, jenv = _guard_steps(clone)
    told = jenv["TOLD"]
    out = tmp_path / "guards"
    out.mkdir()
    (out / "summary.json").write_text(json.dumps({"comment": 1, "told": 1, "writes": 0}))
    (out / "plan.json").write_text(json.dumps({"after": "f" * 40}))
    (out / "report.md").write_text("## Card guards\n\nAt check time.\n")
    (out / "report-landed.md").write_text("## Card guards\n\nAfter the window.\n")
    (out / "discord-final.json").write_text(json.dumps({"content": "settled"}))
    state, posts = tmp_path / "gh.json", tmp_path / "posts.txt"
    env = {"PATH": f"{_fake_tools(tmp_path)}:{os.environ['PATH']}", "OUT": str(out), "TOLD": told, "MODE": "on",
           "GITHUB_REPOSITORY": "o/cards", "WEBHOOK": "https://discord.invalid/hook", "FAKE_STATE": str(state),
           "FAKE_POSTS": str(posts), "GITHUB_OUTPUT": str(out / "tell.output"), "COMMENT": "", "LAND_RAN": "true",
           "GITHUB_RUN_ATTEMPT": "2", "GITHUB_EVENT_NAME": "push"}
    done, outputs = _shell(steps["Tell Discord"], env)
    assert done.returncode == 0 and outputs == {"posted": "true"}, done.stderr
    (comment_id, body), = json.loads(state.read_text())["comments"].items()
    assert body == f"## Card guards\n\nAfter the window.\n\n{told}\n"
    done, outputs = _shell(steps["Comment on the commit"], {**env, "GITHUB_OUTPUT": str(out / "comment.output")})
    assert done.returncode == 0 and outputs == {"told": "true", "id": comment_id}       # the re-run: Tell Discord skipped
    state.unlink()
    posts.write_text("")
    done, outputs = _shell(steps["Tell Discord"], {**env, "FAKE_FAIL": "--method POST"})
    assert done.returncode == 0 and outputs == {"posted": "true"} and posts.read_text().count("settled") == 1
    assert "no commit comment carries the told mark: a re-run would post again" in done.stdout


def test_workflow_tells_a_site_build_that_did_not_start(clone, tmp_path):
    """Follow up starts the site build and the card sync after a bot commit; when one does not start, the comment says so,
    the run fails, and the one post gets a line for it (the failure alert stays quiet once the post went out). A post
    with no room for the line goes out as settled."""
    yaml = pytest.importorskip("yaml")
    if not shutil.which("jq"):
        pytest.skip("no jq")
    doc, steps, jenv = _guard_steps(clone)
    told = jenv["TOLD"]
    out = tmp_path / "guards"
    out.mkdir()
    (out / "plan.json").write_text(json.dumps({"after": "f" * 40}))
    state, posts = tmp_path / "gh.json", tmp_path / "posts.txt"
    env = {"PATH": f"{_fake_tools(tmp_path)}:{os.environ['PATH']}", "OUT": str(out), "TOLD": told, "MODE": "on",
           "GITHUB_REPOSITORY": "o/cards", "WEBHOOK": "https://discord.invalid/hook", "FAKE_STATE": str(state),
           "FAKE_POSTS": str(posts), "SHA": "c" * 40, "COMMENT": "101", "TOLD_BEFORE": "", "LAND_RAN": "true"}
    line = "The bot's commit did not start the site build and the card sync"
    for fail, refreshed in (("", True), ("workflow run card-sync.yml", False), ("workflow run site.yml", False)):
        (out / "report-landed.md").write_text("## Card guards\n\nLanded.\n")
        state.write_text(json.dumps({"comments": {"101": "## Card guards\n\nAt check time.\n"}, "next": 101, "log": []}))
        follow, outputs = _shell(steps["Follow up"], {**env, "FAKE_FAIL": fail, "GITHUB_OUTPUT": str(out / "follow.output")})
        db = json.loads(state.read_text())
        assert follow.returncode == (0 if refreshed else 1) and outputs == ({} if refreshed else {"refresh": "failed"}), fail
        assert ("did not start after this commit" in db["comments"]["101"]) is not refreshed
        assert db["comments"]["101"].startswith("## Card guards\n\nLanded.")
        assert [c for c in db["log"] if c.startswith("workflow run")] == (
            ["workflow run site.yml", SYNC] if fail != "workflow run site.yml" else ["workflow run site.yml"])
    for content, extended in (("settled", True), ("x" * 1950, False)):
        posts.write_text("")
        (out / "discord-final.json").write_text(json.dumps({"content": content, "flags": 4}))
        tell, outputs = _shell(steps["Tell Discord"], {**env, "REFRESH": "failed", "GITHUB_OUTPUT": str(out / "tell.output")})
        sent = json.loads(posts.read_text())
        assert tell.returncode == 0 and outputs == {"posted": "true"} and sent["flags"] == 4
        assert sent["content"].startswith(content) and (line in sent["content"]) is extended and len(sent["content"]) <= 2000
    assert "needs.post.outputs.posted != 'true'" in steps["Tell Discord the run failed"]["if"]


FAKE_PYTHON = r"""#!/bin/sh
echo "$*" >> "$FAKE_PYTHON_ARGS"
printf '%s' "$FAKE_LANDED" > "$OUT/landed.json"
exit "${FAKE_LAND_STATUS:-0}"
"""


def test_workflow_land_and_a_rerun_after_a_landing(clone, tmp_path):
    """The post job (its Land) runs only with something to tell in report mode, as its writes are told, never made; on
    mode for a write too: the check job's plan says land=true. A re-run or a dispatch passes --rerun (an unread branch tells no write as not made), and the
    step hands on the sha and the count already on the branch. Follow up starts the site build and the card sync after a
    commit, and again on a re-run that finds the writes landed (the first attempt's may not have started), not in report
    mode and not on a first attempt."""
    yaml = pytest.importorskip("yaml")
    if not shutil.which("jq"):
        pytest.skip("no jq")
    doc, steps, jenv = _guard_steps(clone)
    bin_dir = _fake_tools(tmp_path)
    (bin_dir / "python").write_text(FAKE_PYTHON)
    (bin_dir / "python").chmod(0o755)
    out, args, state = tmp_path / "guards", tmp_path / "args.txt", tmp_path / "gh.json"
    out.mkdir()
    (tmp_path / "bot").mkdir()
    env = {"PATH": f"{bin_dir}:{os.environ['PATH']}", "OUT": str(out), "GITHUB_REPOSITORY": "o/cards", "EVENT": "push",
           "QUIET_FROM": "2026-10-22T17:00:00Z", "GITHUB_STEP_SUMMARY": str(tmp_path / "summary.md"), "FAKE_PYTHON_ARGS": str(args),
           "GITHUB_OUTPUT": str(tmp_path / "land.output"), "FAKE_STATE": str(state), "TOLD": jenv["TOLD"]}
    landed = json.dumps({"sha": "", "already": [["Snack", "X/Snack.png"]], "landed": []})
    for mode, told, writes, attempt, event, ran, rerun in (("report", 0, 2, "1", "push", False, False),
                                                           ("report", 1, 2, "1", "push", True, False),
                                                           ("on", 0, 2, "1", "push", True, False),
                                                           ("on", 0, 2, "2", "push", True, True),
                                                           ("on", 0, 1, "1", "workflow_dispatch", True, True),
                                                           ("on", 0, 0, "1", "push", False, False)):
        args.write_text("")
        (out / "summary.json").write_text(json.dumps({"told": told, "writes": writes}))
        planned, plan_out = _shell(steps["Plan for the post job"], {**env, "MODE": mode})
        assert planned.returncode == 0 and (plan_out.get("land") == "true") is ran, (mode, told, writes, plan_out)
        if ran:
            done, outputs = _shell(steps["Land"], {**env, "MODE": mode, "GITHUB_RUN_ATTEMPT": attempt, "EVENT": event,
                                                   "FAKE_LANDED": landed}, cwd=tmp_path / "bot")
            assert done.returncode == 0, done.stderr
        else:
            outputs = {}
        called = args.read_text()
        assert bool(called) is ran and outputs.get("ran") == ("true" if ran else None), (mode, told, writes)
        assert ("--rerun" in called) is rerun and ("--dry-run" in called) is (mode == "report" and ran), called
        if ran:
            assert outputs["sha"] == "" and outputs["already"] == "1"
    (out / "report-landed.md").write_text("## Card guards\n\nNothing new to land.\n")
    for mode, attempt, sha, already, started in (("on", "2", "", "1", True), ("on", "1", "", "1", False),
                                                 ("report", "2", "", "1", False), ("on", "2", "", "0", False),
                                                 ("on", "1", "c" * 40, "", True), ("on", "2", "", "", False)):
        state.write_text(json.dumps({"comments": {}, "next": 101, "log": []}))
        done, outputs = _shell(steps["Follow up"], {**env, "MODE": mode, "GITHUB_RUN_ATTEMPT": attempt, "SHA": sha,
                                                    "ALREADY": already, "COMMENT": "", "TOLD_BEFORE": ""})
        log = [c for c in json.loads(state.read_text())["log"] if c.startswith("workflow run")]
        assert done.returncode == 0 and log == (["workflow run site.yml", SYNC] if started else []), \
            (mode, attempt, sha, already, log)


def test_workflow_a_cancel_by_hand_sends_no_alert(clone, tmp_path):
    """Cancelling a run by hand (the post job waiting out the window, say) is no failure to tell the channel: the alert
    is quiet and leaves no streak mark. A job cancelled at its timeout (check 100 minutes, post 75; GitHub reads a
    timeout as cancelled) still tells, as a failure does: the attempt's job list says how long the cancelled job ran
    (never started, cancelled while waiting: by hand). Without the list, the old rule: 95 minutes from the check's
    Start."""
    yaml = pytest.importorskip("yaml")
    if not shutil.which("jq"):
        pytest.skip("no jq")
    doc, steps, jenv = _guard_steps(clone)
    assert steps["Tell Discord the run failed"]["env"]["RESULTS"] == "${{ needs.check.result }} ${{ needs.post.result }}"
    assert 'echo "started=$(date +%s)"' in steps["Start"]["run"]
    posts, runner = tmp_path / "posts.txt", tmp_path / "runner"
    now = int(datetime.now(timezone.utc).timestamp())
    env = {"PATH": f"{_fake_tools(tmp_path)}:{os.environ['PATH']}", "OUT": str(tmp_path), "GITHUB_REPOSITORY": "o/cards",
           "WEBHOOK": "https://discord.invalid/hook", "FAKE_STATE": str(tmp_path / "gh.json"), "FAKE_POSTS": str(posts),
           "GITHUB_STEP_SUMMARY": str(tmp_path / "summary.md"), "STREAK": jenv["STREAK"], "RUNNER_TEMP": str(runner),
           "GITHUB_RUN_ID": "7", "GITHUB_RUN_ATTEMPT": "1", "GITHUB_ACTOR": "Alice", "CHECKED": "success", "FAKE_RUNS": "[]",
           "RUN": "https://github.com/o/cards/actions/runs/7", "TOLD": jenv["TOLD"], "GITHUB_SHA": "f" * 40}

    def jobs(*ran):
        def at(t):
            return datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        return json.dumps({"jobs": [{"name": name, "conclusion": conclusion, "started_at": at(now - minutes * 60) if minutes is not None else None,
                                     "completed_at": at(now) if minutes is not None else None} for name, conclusion, minutes in ran]})
    cases = (("success cancelled", {"FAKE_JOBS": jobs(("check", "success", 5), ("post", "cancelled", 3))}, 0),   # by hand in Land
             ("success cancelled", {"FAKE_JOBS": jobs(("check", "success", 5), ("post", "cancelled", None))}, 0),   # while waiting
             ("success cancelled", {"FAKE_JOBS": jobs(("check", "success", 5), ("post", "cancelled", 75))}, 1),    # post's timeout
             ("cancelled skipped", {"FAKE_JOBS": jobs(("check", "cancelled", 20))}, 0),
             ("cancelled skipped", {"FAKE_JOBS": jobs(("check", "cancelled", 100))}, 1),                           # check's timeout
             ("cancelled skipped", {"FAKE_FAIL": "/jobs", "STARTED": str(now - 20 * 60)}, 0),                      # no list: Start
             ("cancelled skipped", {"FAKE_FAIL": "/jobs", "STARTED": str(now - 96 * 60)}, 1),
             ("cancelled skipped", {"FAKE_FAIL": "/jobs"}, 0),
             ("failure skipped", {"FAKE_JOBS": jobs(("check", "failure", 20))}, 1))
    for results, extra, sent in cases:
        posts.write_text("")
        (tmp_path / "summary.md").write_text("")
        out = _run_steps(steps, ["Tell Discord the run failed"], {**env, "RESULTS": results, **extra}, {})["tell"]
        lines = [l for l in posts.read_text().splitlines() if l.strip()]
        assert len(lines) == sent and out.get("streak", "") == ("true" if sent else ""), (results, extra)
        assert sent or "Cancelled by hand" in (tmp_path / "summary.md").read_text()


def test_workflow_a_rerun_counts_the_window_from_the_push(clone, tmp_path):
    """Start: a first attempt's quiet window starts now; a re-run's at the push (the event's pushed_at: Unix seconds in a
    push event, an ISO time taken too), so a re-run hours later does not wait another 31 minutes. Anything else: now.
    (Land passes --quiet-from on a push only.)"""
    yaml = pytest.importorskip("yaml")
    if not shutil.which("jq"):
        pytest.skip("no jq")
    doc, steps, jenv = _guard_steps(clone)
    step = steps["Start"]
    assert step["env"]["PUSHED_AT"] == "${{ github.event.repository.pushed_at }}"
    pushed = 1761130800                                                 # 2025-10-22T11:00:00Z
    for attempt, pushed_at, at_push in (("2", str(pushed), True), ("1", str(pushed), False), ("2", "2025-10-22T11:00:00Z", True),
                                        ("2", "", False), ("2", "2025-10-22 11:00", False), ("2", "1e9; date", False)):
        github_env = tmp_path / "env"
        github_env.write_text("")
        before = datetime.now(timezone.utc).replace(microsecond=0)
        done, _ = _shell(step, {"PATH": os.environ["PATH"], "MODE": "on", "RUNNER_TEMP": str(tmp_path), "PUSHED_AT": pushed_at,
                                "GITHUB_RUN_ATTEMPT": attempt, "GITHUB_ENV": str(github_env),
                                "GITHUB_OUTPUT": str(tmp_path / "start.output")})
        assert done.returncode == 0, done.stderr
        quiet = dict(l.split("=", 1) for l in github_env.read_text().splitlines())["QUIET_FROM"]
        when = datetime.fromisoformat(quiet.replace("Z", "+00:00"))
        if at_push:
            assert quiet == "2025-10-22T11:00:00Z"
        else:
            assert before <= when <= datetime.now(timezone.utc), (attempt, pushed_at, quiet)


def test_land_commits_only_an_on_plan(tmp_path, capsys):
    """`check` defaults to --mode dry and `land` to a real commit (on the Mac the token comes from config.py): a plan not
    checked with --mode on is refused once its mode is read, before anything is written, unless --dry-run or --local."""
    for mode in ("dry", "report"):
        plan = _full_plan(tmp_path, [_entry("Quiet")], mode=mode)
        for name in ("landed.json", "report-landed.md", "discord-final.json"):
            (Path(plan).parent / name).unlink(missing_ok=True)
        with pytest.raises(SystemExit) as stop:
            cg.main(["land", "--plan", plan])
        assert stop.value.code == 2 and f"checked with --mode {mode}" in capsys.readouterr().err
        assert not (Path(plan).parent / "landed.json").exists()
        assert cg.main(["land", "--plan", plan, "--dry-run"]) == 0 and (Path(plan).parent / "landed.json").exists()


def test_workflow_refuses_a_landing_dispatch_while_a_push_run_waits(clone, tmp_path):
    """A dispatch with dry_run off on a commit whose push run is unfinished (its head is the commit or after it) would
    post and land first: refused. A push run of an earlier commit, or none, lets it through; an unknown head (pushed
    since the checkout) refuses."""
    yaml = pytest.importorskip("yaml")
    doc, steps, jenv = _guard_steps(clone)
    step = steps["No push run in its window"]
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
    step (Land's --dry-run test is exact; `land` would also refuse a report plan without it)."""
    yaml = pytest.importorskip("yaml")
    doc, steps, jenv = _guard_steps(clone)
    step = steps["Start"]
    for typed, mode in (("On", "on"), ("REPORT", "report"), ("dry", "dry")):
        github_env = tmp_path / f"env-{typed}"
        github_env.write_text("")
        done, _ = _shell(step, {"PATH": os.environ["PATH"], "MODE": typed, "RUNNER_TEMP": str(tmp_path),
                                "GITHUB_ENV": str(github_env), "GITHUB_OUTPUT": str(tmp_path / "start.output")})
        assert done.returncode == 0 and f"MODE={mode}" in github_env.read_text().splitlines(), done.stderr
    land = steps["Land"]
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


def test_a_composer_fix_is_said_only_in_the_commit(tmp_path, monkeypatch):
    entry = _entry("A", composer=["Name"])
    _write(tmp_path / "plan", entry, "X/A.psd", "psd", b"fixed", "Single-line Composer (Name)")
    entry["writes"][0].update(shown="", silent=True, composer=True)
    plan = _full_plan(tmp_path, [entry])
    data = json.loads(Path(plan).read_text())
    assert cg.summary(data, []) == {**cg.summary(data, []), "told": 0, "comment": 0, "writes": 1}
    assert cg.discord_payload(data, "o/r") == {} and "omposer" not in cg.markdown(data, [], [], [])
    assert cg.message([entry], "f" * 40).startswith(f"Card guards: Single-line Composer on 1 card (push {'f' * 8})")
    FakeHub({"X/A.psd": ("100644", "s1"), "X/A.png": ("100644", "p1")}).install(monkeypatch)
    landed = cg.land(plan)
    assert landed["sha"] == "c" * 40 and landed["told"]["X/A.psd"] == []
    after = cg.landed_markdown(data, landed)
    assert "omposer" not in after and "Landed" not in after
    FakeHub({"X/A.psd": ("100644", "moved"), "X/A.png": ("100644", "p1")}).install(monkeypatch)
    assert cg.land(plan)["told"]["X/A.psd"] == []          # dropped: never told either


def test_a_rewrap_redraw_follows_into_the_png_unsaid(tmp_path, monkeypatch):
    """R8 always switches; box text that rewraps is redrawn and its PNG re-exported through guard 1's export, unsaid (a
    guard 1 write takes the new pixels, the pushed PNG's recompression gives way); a refused export, or no PNG that shows
    the PSD, lands the PSD without the redraw."""
    from types import SimpleNamespace
    fixed = SimpleNamespace(data=b"drawn", plain=b"plain", rewrapped=["Ability"])
    found = pe.Export("re-exported", image=object(), data=b"png2", changed_px=12)
    monkeypatch.setattr(cg.png_export, "open_psd", lambda data: data)
    monkeypatch.setattr(cg.png_export, "export", lambda old, new, png, numbered: found)
    result = cg.Result(cg.Job("X/A.psd", "pair", png="X/A.png"))
    result.writes.append({"path": "X/A.png", "kind": "recompress", "line": "recompressed", "file": "f", "sha": "s"})
    assert cg._rewrapped_pngs(str(tmp_path), result, b"pushed", fixed, {"X/A.png": "image"})
    assert [(w["path"], w["kind"], w["silent"], w["composer"]) for w in result.writes] == [("X/A.png", "png", True, True)]
    assert "Single-line Composer" in result.writes[0]["line"] and result.composer_left == []
    result.writes = [{"path": "X/A.png", "kind": "png", "line": "re-exported by the bot's render", "file": "f", "sha": "s"}]
    assert cg._rewrapped_pngs(str(tmp_path), result, b"pushed", fixed, {"X/A.png": "image"})
    assert len(result.writes) == 1 and not result.writes[0].get("silent") and result.writes[0]["sha"] == blob_sha(b"png2")
    found.outcome, found.why = "refused", "PNG not made: G6"
    assert not cg._rewrapped_pngs(str(tmp_path), result, b"pushed", fixed, {"X/A.png": "image"})
    assert not cg._rewrapped_pngs(str(tmp_path), result, b"pushed", fixed, {})
    assert [why for _, why in result.composer_left] == ["rewraps, not redrawn: PNG not made: G6",
                                                        "rewraps, not redrawn: no PNG that shows the PSD"]


def test_the_checks_report_at_once(clone):
    """The card and the font check (card-check.yml and font-check.yml until 2026-10-25) run in the check job before the
    guards' Check, whatever CARD_GUARDS says, never failing it, and each tells Discord and comments at once: only the
    guards' fixes and post wait the quiet window. A font added starts them too."""
    doc, steps, _ = _guard_steps(clone)
    names = [s.get("name") for s in doc["jobs"]["check"]["steps"]]
    for check in ("Card check", "Font check"):
        assert steps[check]["continue-on-error"] is True and "MODE" not in steps[check].get("if", "")
        assert names.index(check) < names.index("Tell the checks") < names.index("Check")
    tell = steps["Tell the checks"]
    assert "card-check font-check" in tell["run"] and tell["env"]["WEBHOOK"] == "${{ secrets.DISCORD_STANDARDS_WEBHOOK }}"
    assert "MODE" not in tell["if"] and "MODE" not in doc["jobs"]["check"].get("if", "")
    assert "Tell the checks" not in [s.get("name") for s in doc["jobs"]["post"]["steps"]]
    assert "MDW/Fonts/*" in doc[True]["push"]["paths"]


def test_box_text_fonts_fetched_alone(tmp_path, monkeypatch):
    """R8's box text without a clone's MDW/Fonts: only the files its faces look to be in, a request each, capped; the
    pool's workers get the checked tree (review 2026-10-25)."""
    monkeypatch.delenv(cg.FONT_DIRS_ENV, raising=False)
    names = ["Cambria.ttc", "cambriab.ttf", "Arial.ttf", "ANDYB.TTF", "3367-font.ttf"]
    tree = {f"MDW/Fonts/{name}": ("100644", f"sha-{name}") for name in names}
    assert cg.font_files("Cambria-Bold", cg.tree_fonts({p: sha for p, (_, sha) in tree.items()}))[0] == "cambriab.ttf"
    fetched = []
    faces = {"cambriab.ttf": ["Cambria-Bold"], "Cambria.ttc": ["Cambria", "CambriaMath"], "Arial.ttf": ["ArialMT"]}
    monkeypatch.setattr(cg.Source, "blob", lambda self, sha: fetched.append(sha) or sha.encode())
    monkeypatch.setattr(cg, "face_names", lambda data: faces.get(data.decode()[4:], []))
    (tmp_path / "fonts").mkdir()
    source = cg.Source(str(tmp_path / "clone"), "o/r", str(tmp_path / "fonts"))
    assert source.fonts_for(["Cambria-Bold"]) == ""                 # no checked tree: nothing fetched
    source.checked = tree
    assert source.fonts_for(["Cambria-Bold"]) == str(tmp_path / "fonts") and fetched == ["sha-cambriab.ttf"]
    assert source.fonts_for(["Cambria-Bold", "Wingdings"]) == "" and len(fetched) == 1     # no file looks like it
    assert sorted(os.listdir(tmp_path / "fonts")) == ["cambriab.ttf"]
    monkeypatch.setattr(cg, "FONT_FETCH_MAX", 1)
    assert source.fonts_for(["ArialMT"]) == "" and len(fetched) == 1
    monkeypatch.setattr(cg, "_worker", {})
    monkeypatch.setattr(cg, "token_from_env", lambda: None)
    vocab = type("V", (), {"parser": lambda self: None})()
    cg._init_worker(str(tmp_path / "clone"), "o/r", None, vocab, tree, str(tmp_path / "fonts"))
    assert cg._worker["source"].checked == tree and cg._worker["source"].fonts == str(tmp_path / "fonts")
