"""
card_guards (the push guards' driver): a card repo made here with git (fixture PSDs, every case of PUSH_GUARDS.md 4.2,
bot commits left out), check end to end on it (the report, the one Discord post without mentions or previews, the plan),
land against a fake GitHub (superseded cards, a moved head) and in a local clone, the replay of the real
push 38c55bd3..07589cc7 from the card clone (read-only: God Tamer, Bubble and 12 Reactions2 "PNG still current"), the
Action's sparse copy (SPARSE_FILES), card-guards.yml's shape, and its steps' shell against a fake gh and curl (one post
per push, the failure streak, a re-run's lookups and window, the refused dispatch, MODE in any case).
"""
import json
import os
import re
import shutil
import subprocess
import sys
import types
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from sys import executable
from types import SimpleNamespace

import pytest
from PIL import Image, ImageDraw

import commands.card_guards as cg
import commands.font_check as fc
import commands.git_commit as gc
import commands.png_export as pe
from commands.github_api import blob_sha
from conftest import CLONE
from helpers import art, build_font, card, fixture_psd, photoshop_png, sharing


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


@pytest.fixture(scope="session")
def guards_template(tmp_path_factory):
    """(root, base commit, push commits): the base tree, then Alice's push and two bot commits; built once per worker."""
    tmp_path = tmp_path_factory.mktemp("guards-template")
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


@pytest.fixture
def cards(tmp_path, guards_template):
    """The template's repo, cloned --local (hardlinked objects), with an empty PSD folder beside it."""
    root, base, alice, bot = guards_template
    _git(tmp_path, "clone", "-q", "--local", str(root), str(tmp_path / "cards"))
    _git(tmp_path / "cards", "remote", "remove", "origin")
    (tmp_path / "psds").mkdir()
    return tmp_path / "cards", base, alice, bot


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
def test_check_end_to_end(cards, tmp_path, monkeypatch):
    root, base, alice, bot = cards
    monkeypatch.setattr(cg, "token", lambda: "")       # a local run: logins from git, never the API (config.py's token)
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
    assert summary == {"after": bot, "cards": 8, "told": summary["told"], "comment": 6, "writes": 8, "normalise": 4,
                       "errors": 0, "card_errors": 0} and summary["told"] >= 1     # 4 merged images stripped later (Q2)
    assert by["Bread"]["told"] == [] and not by["Bread"]["issues"]      # R3: no Ability layer is the developer's layout
    assert not by["Dropped"]["writes"] and not by["Dropped"]["export"]
    report = (out / "report.md").read_text()
    assert report.startswith("## Card guards") and "Report only (CARD_GUARDS=report)" in report and "Bot_Card" in report
    payload = json.loads((out / "discord.json").read_text())
    assert payload["allowed_mentions"] == {"parse": []} and payload["flags"] == 4 and "<@" not in payload["content"]
    lines = payload["content"].splitlines()
    assert all(re.fullmatch(r"\S+ pushed .+: .+\. \[[0-9a-f]{8}\]\(https://github\.com/o/cards/commit/[0-9a-f]{12}\)", l)
               for l in lines), lines
    assert "Lonely.png does not match its PSD" in report and not any("Lonely" in l for l in lines)
    assert not any("Snack" in l and "re-exported" in l for l in lines)        # writes are not issues
    assert "- PNG would be re-exported by the bot's render of the PSD (441 px changed" in report and "with CARD_GUARDS=on" in report
    assert "- PNG re-exported" not in report and "- PNG copied" not in report
    assert "Alice pushed Snack: PNG not exported with the PSD: export it in Photoshop; Snack2.png" in payload["content"]
    # what changed inside each PSD pushed over an older one (psd_diff): a new PSD or a PNG alone has none
    assert snack["changes"] == by["Bread"]["changes"] == by["Dropped"]["changes"] == ["Square pixels changed"]
    assert by["New_Card"]["changes"] == by["Lonely"]["changes"] == by["New_Name"]["changes"] == []
    assert "**Snack** (`Items/Heal/2 Stars/Snack.psd`, Alice, `" in report and "- Changed: Square pixels changed" in report
    assert "Alice pushed New Card: " in payload["content"]


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


@pytest.mark.parametrize("first, made", [("2099-10-22T10:00:00Z", True), ("2099-10-22T06:00:00Z", False)])
def test_a_burst_of_psd_pushes_still_gets_its_png(cards, tmp_path, first, made):
    """Two PSD pushes 10 minutes apart, no PNG: the first run is superseded, so the second checks against the PSD as of
    the PNG's last change (the base): outside both changes the PNG still matches, and the PNG is made. Four hours apart,
    no anchor: G5 refuses as before (anchor not even asked: one history walk found nothing in its window)."""
    root, base, alice, bot = cards
    work = tmp_path / "burst"
    work.mkdir()
    one = _dated(root, work, "Bread", card(base=(30, 200, 30)), first)             # the Base layer: everywhere
    two = _dated(root, work, "Bread", card(base=(30, 200, 30), colour=(0, 255, 0)), "2099-10-22T10:10:00Z")   # the square
    source, asked = cg.Source(str(root)), []
    anchor = source.anchor
    source.anchor = lambda *args: asked.append(args) or anchor(*args)
    (job,) = cg.plan_jobs(source, one, two, "actor")[0]
    assert job.case == "stale" and bool(job.psd_anchor) is made and len(asked) == made
    if made:
        assert job.psd_anchor == source.tree(alice)[f"{FOLDER}/Bread.psd"][1] and job.anchor_commit == alice
    data = cg.check(str(root), two, one, mode="on", out=str(tmp_path / "out"), workers=1)
    (entry,) = data["cards"]
    assert not entry["error"], entry["error"]
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
    assert not entry["error"], entry["error"]
    assert entry["export"]["outcome"] == "re-exported" and entry["export"]["unchecked"] is True
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
    assert not entry["error"], entry["error"]
    assert (entry["export"]["outcome"], entry["export"]["gate"]) == ("refused", "G7"), entry["export"]
    assert not [w for w in entry["writes"] if w["kind"] == "png"] and pe.G7 in entry["told"]


@pytest.mark.parametrize("dot", [False])
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
    assert not entry["error"], entry["error"]
    assert entry["export"]["outcome"] == "re-exported" and entry["export"]["changed_px"] == (4 if dot else 441)
    assert not any("an earlier push changed the PSD" in note for note in entry["notes"]), entry["notes"]
    assert not any(line.startswith("PNG not made") for line in entry["told"])
    (write,) = entry["writes"]
    assert int(pe._diff(Image.open(out / write["file"]), v3).max()) == 0


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
    assert "New Card: the bot could not finish checking it" in data["discord"]["content"]
    result = cg.Result(cg.Job("X/A.psd", "pair"), error="KeyError: 'x'")
    assert result.told() == ["the bot could not finish checking it (KeyError)"]
    monkeypatch.setattr(cg.png_export, "export_new", real)


def test_check_fails_when_it_cannot_check(cards, tmp_path, capsys):
    """Not checked (the clone lacks the commit, the API is down): exit 1, so card-guards.yml's failure alert runs."""
    root, base, alice, bot = cards
    assert cg.main(["check", "--clone", str(root), "--after", "e" * 40, "--out", str(tmp_path / "out")]) == 1
    assert "Not checked: the clone has no commit eeeeeeee" in capsys.readouterr().out


def test_discord_fits_and_says_nothing_without_issues():
    entry = {"login": "Zomoroko", "card": "Skullord", "told": ["timing tag (1GCD) not set in the stoogedings badge font"], "commit": "7" * 40}
    data = {"after": "a" * 40, "cards": [dict(entry, card=f"Card {i}") for i in range(80)]}
    payload = cg.discord_payload(data, "MichaelJSr/TTSCardMaker")
    assert len(payload["content"]) <= cg.DISCORD_LIMIT and payload["flags"] == 4
    last = payload["content"].splitlines()[-1]
    assert re.fullmatch(r"\.\.\. and \d+ more cards: \[details\]\(https://github\.com/MichaelJSr/TTSCardMaker/commit/a{40}\)", last)
    one = cg.discord_payload({"after": "a" * 40, "cards": [entry]}, "MichaelJSr/TTSCardMaker")["content"]
    assert one == ("Zomoroko pushed Skullord: timing tag (1GCD) not set in the stoogedings badge font. "
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
# The 11-07 pass's conversions and linked files (G1-G4)
# ========================
def _links_module(monkeypatch, unused=(), duplicates=None, links=None) -> list:
    """commands.psd_links as the guards call it (the links lane's module); returns the relinks made."""
    fake, relinked = types.ModuleType("commands.psd_links"), []
    fake.catalogue_order = sorted
    fake.items = lambda psd, vocab=(): []
    fake.TypesIndex = lambda files, history=None: None
    fake.assign = lambda psds, kinds, linked: {"psds": {rel: [] for rel, _ in psds}, "files": {}}
    fake.fallback_names = lambda catalogue, linked, kept=None: None
    fake.links_of = lambda psd, rel, tree: [SimpleNamespace(path=p, name=p.rsplit("/", 1)[-1], uuid="u1", size=3)
                                            for p in (links or {}).get(rel, [])]
    fake.records_of = lambda index: ({k: v for k, v in index.items() if k != "files"}, index.get("files") or {})
    fake.unused = lambda index, tree, pushed=(), read=(): list(unused)
    fake.duplicates = lambda tree, pushed=(): dict(duplicates or {})
    fake.restore = lambda size, history: None
    fake.relink = lambda psd, uuid, rel, path, size, blob: relinked.append((rel, path))
    monkeypatch.setitem(sys.modules, "commands.psd_links", fake)
    monkeypatch.setattr(sys.modules["commands"], "psd_links", fake, raising=False)   # `from commands import psd_links`
    return relinked


def test_housekeeping_reads_no_pushed_psd_its_check_read(cards, tmp_path, monkeypatch):
    """A pushed PSD's links come from its check (read in the pool): housekeeping does not read the PSD again."""
    root, base, alice, bot = cards
    rel = f"{FOLDER}/Snack.psd"
    _psd(root, tmp_path / "psds", rel, card(colour=(0, 0, 200)))
    head = _commit(root, "Alice again", author="Alice")
    _links_module(monkeypatch, links={rel: ["MDW/Linked/art.png"]})
    monkeypatch.setitem(cg.standards.PIXEL_FIXES, cg.NORMALISE, (lambda psd: [], None, None))   # nothing to convert
    fake, inside, reads, seen = sys.modules["commands.psd_links"], [], [], []
    links_of, housekeeping = fake.links_of, cg.housekeeping
    fake.links_of = lambda psd, path, tree: reads.append(bool(inside)) or links_of(psd, path, tree)
    fake.unused = lambda index, tree, pushed=(), read=(): seen.extend((r, [l.path for l in found]) for r, found in read) or []
    monkeypatch.setattr(cg, "housekeeping", lambda *args, **kw: inside.append(1) or housekeeping(*args, **kw))
    monkeypatch.setattr(cg, "links_index", lambda source: {})
    cg.check(str(root), head, bot, out=str(tmp_path / "out"), workers=1)
    assert inside and reads and not any(reads) and seen == [(rel, ["MDW/Linked/art.png"])]


def test_a_later_push_is_converted_silently(cards, tmp_path, monkeypatch):
    """G1 (owner 11-07): a card and a template pushed with what the normalise fix finds are converted after the check
    (--convert later: summary.json counts them, so Card sync's wait ends at Decided), the card on top of guard 1's PNG
    (its redrawn picture re-exported in it), its new MDW/Linked file beside it, the template with nothing else checked;
    one landing, no Discord line, no guard-event for the conversion, the commit message alone says it."""
    root, base, alice, bot = cards
    _psd(root, tmp_path / "psds", f"{FOLDER}/Snack.psd", card(colour=(0, 0, 200)))
    _psd(root, tmp_path / "psds", "MDW/Templates/Item_Template.psd", card())
    head = _commit(root, "Alice again", author="Alice")
    _links_module(monkeypatch)
    monkeypatch.setitem(cg.standards.PIXEL_FIXES, cg.NORMALISE, (lambda psd: [psd], None, None))
    real, converted = cg.standards.fix, []
    fixture_psd(tmp_path / "psds", card(colour=(0, 200, 0)), "Redrawn")   # Snack's picture drawn again (class 3)
    redrawn = (tmp_path / "psds" / "Redrawn.psd").read_bytes()

    def fix(data, rel, issues, work, parser=None, fonts=None, only=None, whole=False, skip=()):
        if only != cg.NORMALISE:
            assert cg.NORMALISE in skip            # never in the check itself
            return real(data, rel, issues, work, parser, fonts, only, whole, skip)
        assert isinstance(fonts.links, cg.LinkContext) and fonts.links.tree[rel]
        converted.append(rel)
        if rel.endswith("Snack.psd"):           # card_standards' shape: the redrawn layers' names
            return SimpleNamespace(data=redrawn, files={"MDW/Linked/Snack - art.png": b"art"},
                                   redrawn={cg.NORMALISE: ["Square"]}, left=[], pixel_left=[])
        return SimpleNamespace(data=data + b"linked", files={}, redrawn={cg.NORMALISE: ["art"]}, left=[], pixel_left=[])
    monkeypatch.setattr(cg.standards, "fix", fix)
    out = tmp_path / "out"
    data = cg.check(str(root), head, bot, mode="on", out=str(out), workers=1, convert="later")
    by = {Path(e["psd"]).name: e for e in data["cards"]}
    assert by["Snack.psd"]["convert"] == [cg.NORMALISE] and by["Item_Template.psd"]["case"] == "convert"
    assert data["summary"]["normalise"] == 2 and data["summary"]["writes"] >= 3 and not converted
    assert "Item_Template" not in data["markdown"] and data["discord"] == {}
    plan = str(out / "plan.json")
    after = cg.convert(plan, str(root), workers=1)
    assert sorted(converted) == [f"{FOLDER}/Snack.psd", "MDW/Templates/Item_Template.psd"] and after["summary"]["normalise"] == 0
    snack = {w["path"]: w for w in next(e for e in after["cards"] if e["psd"].endswith("Snack.psd"))["writes"]}
    assert snack[f"{FOLDER}/Snack.psd"]["silent"] and snack[f"{FOLDER}/Snack.psd"]["pixels"] == [cg.NORMALISE]
    assert snack["MDW/Linked/Snack - art.png"]["kind"] == "file" and snack["MDW/Linked/Snack - art.png"]["silent"]
    report = cg.land(plan, local=str(root))
    assert report["sha"] and not report["error"] and report["told"] == {e["psd"]: [] for e in after["cards"]}
    assert {(e["kind"], e["path"]) for e in report["events"]} == {("guard-png", f"{FOLDER}/Snack.png"),
                                                                 ("guard-png", f"{FOLDER}/Snack2.png")}   # guard 1's alone
    landed = _git(root, "show", "--name-only", "--format=%B", report["sha"])
    assert "1 linked file added" in landed and "MDW/Linked/Snack - art.png" in landed
    assert _git(root, "show", f"{report['sha']}:MDW/Linked/Snack - art.png") == "art"
    png = subprocess.run(["git", "-C", str(root), "show", f"{report['sha']}:{FOLDER}/Snack.png"], capture_output=True, check=True)
    assert Image.open(BytesIO(png.stdout)).convert("RGB").getpixel((20, 20)) == (0, 200, 0)   # the PNG followed
    assert subprocess.run(["git", "-C", str(root), "show", f"{report['sha']}:MDW/Templates/Item_Template.psd"],
                          capture_output=True, check=True).stdout.endswith(b"linked")


def test_unused_and_duplicate_linked_files_go_in_one_silent_commit(cards, tmp_path, monkeypatch):
    """G2 and G3: a pushed MDW/Linked file with another's bytes merged (its user relinked, the duplicate deleted) and an
    unused one deleted, in the one landing, silent; nothing deleted while the links index is missing or lacks a PSD, nor
    once a PSD moved on the branch before the landing (it may link them)."""
    root, base, alice, bot = cards
    for name in ("kept", "dup", "old"):
        (root / "MDW/Linked").mkdir(parents=True, exist_ok=True)
        (root / f"MDW/Linked/{name}.png").write_bytes(b"one" if name != "old" else b"two")
    head = _commit(root, "files", author="Alice")
    tree = {path: sha for path, (_, sha) in cg.Source(str(root)).tree(head).items()}
    user = f"{FOLDER}/Bread.psd"
    relinked = _links_module(monkeypatch, unused=["MDW/Linked/old.png"], duplicates={"MDW/Linked/dup.png": "MDW/Linked/kept.png"},
                             links={user: ["MDW/Linked/dup.png"]})
    monkeypatch.setattr(cg, "links_index", lambda source: None)
    alone = cg.check(str(root), head, bot, mode="on", out=str(tmp_path / "none"), workers=1)
    assert alone["linked"]["writes"] == []      # no index: a user of dup.png may be unknown
    index = {rel: {"blob": sha, "links": []} for rel, sha in tree.items() if rel.endswith(".psd")}
    index[user]["links"] = [{"path": "MDW/Linked/dup.png", "name": "dup.png", "size": 3, "drawn": None}]
    monkeypatch.setattr(cg, "links_index", lambda source: {k: v for k, v in index.items() if k != "MDW/Template.psd"})
    gap = cg.check(str(root), head, bot, mode="on", out=str(tmp_path / "gap"), workers=1)["linked"]["writes"]
    assert [w["path"] for w in gap] == ["MDW/Linked/old.png"]     # the fake's G2 alone: G3 waits for every PSD's links
    monkeypatch.setattr(cg, "links_index", lambda source: index)
    out = tmp_path / "out"
    data = cg.check(str(root), head, bot, mode="on", out=str(out), workers=1)
    assert relinked == [(user, "MDW/Linked/kept.png")]
    writes = {w["path"]: w["kind"] for w in data["linked"]["writes"]}
    assert writes == {"MDW/Linked/old.png": "delete", user: "psd", "MDW/Linked/dup.png": "delete"}
    assert data["summary"]["writes"] == 3 and data["discord"] == {} and "linked" not in data["markdown"].lower()
    other, moved = tmp_path / "other", tmp_path / "moved"
    _git(tmp_path, "clone", "-q", "--local", str(root), str(other))
    shutil.copytree(out, moved)
    _psd(other, tmp_path / "psds", f"{FOLDER}/Snack.psd", card(colour=(1, 2, 3)))
    _commit(other, "a PSD pushed before the landing", author="Alice")
    raced = cg.land(str(moved / "plan.json"), local=str(other))
    assert sorted(path for _, why, path in raced["dropped"] if "a PSD changed" in why) == ["MDW/Linked/dup.png",
                                                                                         "MDW/Linked/old.png"]
    report = cg.land(str(out / "plan.json"), local=str(root))
    assert report["sha"] and report["landed"] == [] and report["events"] == []
    names = _git(root, "ls-tree", "-r", "--name-only", report["sha"]).splitlines()
    assert "MDW/Linked/kept.png" in names and "MDW/Linked/dup.png" not in names and "MDW/Linked/old.png" not in names
    assert "2 linked files deleted" in _git(root, "log", "-1", "--format=%s", report["sha"])


def test_a_deleted_file_still_linked_is_put_back_silently(cards, tmp_path, monkeypatch):
    """G4b (11-08, the Daco_35508 push): an MDW/Linked file the push deleted while a PSD outside it still links it (its
    record, even at another blob) comes back at its blob before the push, silent; one only the pushed PSD linked (it
    dropped the link; its stale record still names it) stays deleted. G4 places a missing link without the tree."""
    import numpy as np
    from psd_tools import PSDImage
    from psd_tools.api.layers import PixelLayer
    import commands.psd_links as pl
    root, base, alice, bot = cards
    snack = f"{FOLDER}/Snack.psd"
    raw = BytesIO()
    card(size=(20, 20)).save(raw, "PNG")
    psd = PSDImage.new("RGB", (40, 40))
    pl.link_pixels(PixelLayer.fromarray(np.full((20, 20, 4), 255, np.uint8), psd, name="art"), snack,
                   "MDW/Linked/gone.png", raw.getvalue(), [0, 0, 20, 0, 20, 20, 0, 20])
    assert cg._missing(psd, snack, cg.RepoTree([])) == [{"path": "MDW/Linked/gone.png", "size": len(raw.getvalue())}]
    assert cg._missing(psd, snack, cg.RepoTree(["MDW/Linked/Gone.png"])) == []
    gone = {"MDW/Linked/a.psb": "1", "MDW/Linked/b.png": "2", "MDW/Linked/c.png": "3"}     # a by name, b by a's record
    records = {"MDW/Linked/a.psb": {"blob": "1", "links": [{"path": "MDW/Linked/b.png", "name": "b.png"}]}}
    assert cg._restored(gone, {"X.psd": [{"path": None, "name": "A.PSB"}]}, records, cg.RepoTree([]), set()) == \
        {"MDW/Linked/a.psb": ("1", 1), "MDW/Linked/b.png": ("2", 1)}
    (root / "MDW/Linked").mkdir(parents=True, exist_ok=True)
    for name in ("shared", "own"):
        (root / f"MDW/Linked/{name}.png").write_bytes(name.encode())
    start = _commit(root, "files", author="Alice")
    for name in ("shared", "own"):
        (root / f"MDW/Linked/{name}.png").unlink()
    _psd(root, tmp_path / "psds", snack, card(colour=(0, 0, 200)))
    head = _commit(root, "picture removed", author="Alice")
    _links_module(monkeypatch)                  # Snack as read: no links
    monkeypatch.setitem(cg.standards.PIXEL_FIXES, cg.NORMALISE, (lambda psd: [], None, None))
    index = {rel: {"blob": "older", "links": []} for rel in cg.Source(str(root)).tree(head) if rel.endswith(".psd")}
    index[snack]["links"] = [{"path": "MDW/Linked/own.png", "name": "own.png"}]
    index[f"{FOLDER}/Bread.psd"]["links"] = [{"path": "MDW/Linked/shared.png", "name": "shared.png"}]
    monkeypatch.setattr(cg, "links_index", lambda source: index)
    out = tmp_path / "out"
    data = cg.check(str(root), head, start, mode="on", out=str(out), workers=1)
    assert [(w["path"], w["kind"], w["silent"], w["line"]) for w in data["linked"]["writes"]] == \
        [("MDW/Linked/shared.png", "file", True, "restored: still linked by 1 PSD")]
    assert "shared.png" not in data["markdown"] and "shared.png" not in json.dumps(data["discord"])
    report = cg.land(str(out / "plan.json"), local=str(root))
    assert _git(root, "show", f"{report['sha']}:MDW/Linked/shared.png") == "shared"
    assert all(e["path"] != "MDW/Linked/shared.png" for e in report["events"])
    assert "MDW/Linked/own.png" not in _git(root, "ls-tree", "-r", "--name-only", report["sha"]).splitlines()
    assert "1 linked file added" in _git(root, "log", "-1", "--format=%s", report["sha"])


def test_a_font_sharing_table_data_lands_with_its_own_copies(cards, tmp_path):
    """A font pushed to MDW/Fonts whose tables share data (Photoshop 2023 sets no type in it) lands with every table its
    own copy, silent: no card, Discord, comment or changelog event. A sound font and one in a subfolder stay as pushed."""
    root, base, alice, bot = cards
    raw = Path(build_font(str(tmp_path / "f.ttf"), "Shared-Regular", "Shared", "Regular")).read_bytes()
    bad, fixed = sharing(raw, b"Xtra", b"glyf"), fc.split_tables(sharing(raw, b"Xtra", b"glyf"))
    (root / "MDW/Fonts/sub").mkdir(parents=True)
    for rel, data in {"MDW/Fonts/Shared.ttf": bad, "MDW/Fonts/Sound.ttf": raw, "MDW/Fonts/sub/Deep.ttf": bad}.items():
        (root / rel).write_bytes(data)
    head = _commit(root, "fonts", author="Alice")
    out = tmp_path / "out"
    data = cg.check(str(root), head, bot, author="alice", mode="on", out=str(out), workers=1)
    assert [(w["path"], w["kind"], w["sha"]) for w in data["fonts"]["writes"]] == [("MDW/Fonts/Shared.ttf", "font", blob_sha(fixed))]
    assert data["cards"] == [] and data["summary"]["writes"] == 1 and data["summary"]["comment"] == 0 and data["discord"] == {}
    report = cg.land(str(out / "plan.json"), local=str(root))
    assert report["sha"] and report["landed"] == [] and report["events"] == []
    assert _git(root, "rev-parse", f"{report['sha']}:MDW/Fonts/Shared.ttf") == blob_sha(fixed)
    assert _git(root, "log", "-1", "--format=%s", report["sha"]) == f"Card guards: 1 font repaired (push {head[:8]})"
    assert cg.check(str(root), report["sha"], head, mode="on", out=str(tmp_path / "again"), workers=1)["fonts"]["writes"] == []


# ========================
# land
# ========================
class FakeHub:
    """snapshot_at / ref_head / commit_files for land."""
    def __init__(self, tree, heads=("h1",)):
        self.tree, self.heads, self.commits, self.calls = dict(tree), list(heads), [], 0

    def install(self, monkeypatch):
        monkeypatch.setattr(cg, "ref_head", lambda repo: self.heads[min(self.calls, len(self.heads) - 1)])
        monkeypatch.setattr(cg, "snapshot_at", lambda repo, head: cg.Snapshot(repo, head, {"tree": {"sha": "t" + head}}, self.tree))
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
                            ("B", "X/B.png", b"b", {"X/B.psd": "s2", "X/B.png": "p2"})])
    hub = FakeHub({"X/A.psd": ("100644", "s1"), "X/A.png": ("100644", "p1"), "X/B.psd": ("100644", "moved"),
                   "X/B.png": ("100644", "p2")}, heads=("h1", "h2"))
    hub.install(monkeypatch)
    report = cg.land(plan)
    assert report["sha"] == "c" * 40 and report["landed"] == ["A"]
    assert report["dropped"] == [["B", "superseded: X/B.psd changed on the branch", "X/B.png"]]
    assert report["told"] == {"X/A.psd": [], "X/B.psd": []}
    assert report["after"]["X/B.psd"] == ["the PSD changed again on the branch: that push's own check tells"]
    assert len(hub.commits) == 1 and hub.commits[0][:2] == ("h2", ["X/A.png"])            # retried on the new head
    assert hub.commits[0][2].startswith("Card guards: 1 PNG re-exported (push ffffffff)")
    assert report["events"] == [{"kind": "guard-png", "card": "A", "path": "X/A.png", "line": "made"}]


def test_land_refuses_a_tampered_file(tmp_path, monkeypatch):
    plan = _plan(tmp_path, [("A", "X/A.png", b"a", {"X/A.psd": "s1"})])
    (Path(plan).parent / "files" / "A").write_bytes(b"changed")
    hub = FakeHub({"X/A.psd": ("100644", "s1")})
    hub.install(monkeypatch)
    report = cg.land(plan)
    assert "does not hash" in report["error"] and report["sha"] == "" and not hub.commits
    assert report["told"]["X/A.psd"] == ["PNG not made: the bot's commit failed; export it in Photoshop"]


def test_land_by_git_past_the_bytes(tmp_path, monkeypatch):
    """Past GIT_LAND_BYTES one git push, else the API; a broken git falls back to the API."""
    plan = _plan(tmp_path, [("A", "X/A.png", b"aaaa", {"X/A.psd": "s1"})])
    hub = FakeHub({"X/A.psd": ("100644", "s1")})
    hub.install(monkeypatch)
    pushed = []
    monkeypatch.setattr(cg, "git_commit", lambda repo, head, files, text: pushed.append(sorted(files)) or "g" * 40)
    monkeypatch.setattr(cg, "GIT_LAND_BYTES", 3)
    assert cg.land(plan)["sha"] == "g" * 40 and pushed == [["X/A.png"]] and not hub.commits
    monkeypatch.setattr(cg, "GIT_LAND_BYTES", 4)
    assert cg.land(plan)["sha"] == "c" * 40 and len(pushed) == 1

    def broken(*args):
        raise gc.Transient("git push failed")
    monkeypatch.setattr(cg, "GIT_LAND_BYTES", 3)
    monkeypatch.setattr(cg, "git_commit", broken)
    assert cg.land(plan)["sha"] == "c" * 40 and len(hub.commits) == 2


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


@pytest.mark.parametrize("png_pushed", [False, True])
def test_r1_is_untold_once_its_tags_fix_lands(tmp_path, monkeypatch, png_pushed):
    """Owner 2026-11-03, the guards set plain timing tags as badges: R1 is not told when the fixed PSD lands; when the
    author pushed the PNG since, the redraw's PNG is dropped, the PSD lands as its plain and R1 stays told."""
    r1 = "timing tag (Action) not set in the stoogedings badge font"
    out = tmp_path / "plan"
    snack = _entry("Snack", told=[r1])
    _write(out, snack, "X/Snack.png", "png", b"png", "re-exported for timing tags set as badges")
    _write(out, snack, "X/Snack.psd", "psd", b"psd", "timing tags set as badges (Ability)")
    _write(out, snack, "X/Snack.psd.plain", "psd", b"plain", "")
    plain = snack["writes"].pop()
    snack["writes"][0].update(composer=True, silent=True)
    snack["writes"][1].update(silent=True, pixels=["tags"], settles=[r1], plain={"file": plain["file"], "sha": plain["sha"], "line": ""})
    plan = _full_plan(tmp_path, [snack])
    FakeHub({"X/Snack.psd": ("100644", "s1"), "X/Snack.png": ("100644", "author's" if png_pushed else "p1")}).install(monkeypatch)
    report = cg.land(plan)
    assert report["sha"] and report["told"] == {"X/Snack.psd": [r1] if png_pushed else []}


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
        "Alice pushed Snack: PNG not exported with the PSD: export it in Photoshop.")
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
    for png, told in (("p1", ["PNG not exported with the PSD: export it in Photoshop"]), ("author's", [])):
        FakeHub({"X/Snack.psd": ("100644", "s1"), "X/Snack.png": ("100644", png)}).install(monkeypatch)
        landed = cg.land(str(out / "plan.json"), dry_run=True)
        assert landed["told"] == {"X/Snack.psd": told} and landed["sha"] == "", png
        post = cg.finish(str(out / "plan.json"), landed)
        assert (post.get("content", "").startswith("Alice pushed Snack: PNG not exported")) is bool(told)
        comment = (out / "report-landed.md").read_text()
        assert ("- After the quiet window: Snack.png was pushed since: not told" in comment) is not told
        assert "would be re-exported" in comment and "Landed" not in comment
    ahead = dict(entry, export={"outcome": "current", "numbers": {"png_ahead": {}}})              # png_export's verdict
    assert cg._guard1_line(ahead, "report") == "PNG already shows the pushed PSD (exported before the PSD was pushed): nothing to make"


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


def test_nothing_is_told_for_a_png_made_or_kept_without_a_stored_image(tmp_path, monkeypatch):
    """Owner 2026-11-07: every card lacks a stored image after the pass, so a PNG the bot makes or keeps without one is
    no issue: nothing told, on Discord or in the comment (the old UNCHECKED, KEPT, RESTYLED and PICTURE lines); the
    numbers stay in plan.json."""
    found = pe.Export("re-exported", changed_px=5, unchecked=True, numbers={"picture": {"drawn": 1, "saved": 2}})
    exported = cg._export_dict(found)
    assert exported["unchecked"] is True
    job = cg.Job("X/Snack.psd", "stale", png="X/Snack.png")
    result = cg.Result(job, export=exported, writes=[{"path": "X/Snack.png", "kind": "png"}])
    assert result.told("on") == []
    assert result.told("report") == ["PNG not exported with the PSD: export it in Photoshop"]
    assert cg._drawn_as(found) == "its picture 1"
    line = cg._guard1_line({"export": {**exported, "numbers": {}, "icc": "as stored"}, "case": "stale"})
    assert line.startswith("PNG re-exported by the bot's render of the PSD (") and "stored image" not in line
    for numbers in ({"kept": 33}, {"restyled": ["the Header text"]}):
        kept = cg._export_dict(pe.Export("current", unchecked=True, numbers=numbers))
        assert cg.Result(cg.Job("X/Snack.psd", "beside", png="X/Snack.png"), export=kept).told("on") == []
        assert cg._guard1_line({"export": kept, "case": "beside"}) == "PNG still current (the push moved no pixel of the card)"
    out = tmp_path / "plan"
    snack = _entry("Snack", export=dict(exported, numbers={}), told=[])
    _write(out, snack, "X/Snack.png", "png", b"png", "re-exported")
    plan = _full_plan(tmp_path, [snack])
    FakeHub({"X/Snack.psd": ("100644", "s1"), "X/Snack.png": ("100644", "p1")}).install(monkeypatch)
    report = cg.land(plan)
    assert report["sha"] and report["told"] == {"X/Snack.psd": []} and cg.finish(plan, report) == {}


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
@pytest.mark.skipif(pe.tool() is None, reason="needs the oxipng CLI (as card-guards.yml installs) or pyoxipng")
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
    assert sorted((e["kind"], e["card"], e["line"]) for e in landed["events"]) == [
        ("guard-png", "Fat", "recompressed losslessly"), ("guard-png", "Snack", "recompressed losslessly")]
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
    hub = FakeHub({"X/Fat.psd": ("100644", "moved"), "X/Fat.png": ("100644", "p1")})
    hub.install(monkeypatch)
    report = cg.land(plan)
    assert report["landed"] == ["Fat"] and hub.commits[0][2].startswith("Card guards: 1 PNG recompressed (push ffffffff)")
    cg.finish(plan, report)                                 # the landed commit's changelog events, for Card sync
    assert json.loads((Path(plan).parent / cg.GUARD_EVENTS).read_text()) == {"schema": 1, "commit": "c" * 40, "push": "f" * 40,
        "events": [{"kind": "guard-png", "card": "Fat", "path": "X/Fat.png", "line": "recompressed losslessly"}]}
    FakeHub({"X/Fat.psd": ("100644", "s1"), "X/Fat.png": ("100644", "theirs")}).install(monkeypatch)
    report = cg.land(plan)
    assert report["dropped"] == [["Fat", "superseded: X/Fat.png changed on the branch", "X/Fat.png"]]
    assert report["told"] == {"X/Fat.psd": []} and cg.finish(plan, report) == {}


def test_a_psd_fix_lands_beside_its_pushed_png(cards, tmp_path):
    """A pair (PSD and PNG pushed together): the fix rests on the pushed PNG's blob, so it lands (it was dropped as
    "superseded" when the plan expected no PNG)."""
    root, base, alice, bot = cards
    out = tmp_path / "out"
    cg.check(str(root), bot, base, mode="on", out=str(out), workers=1)
    plan = json.loads((out / "plan.json").read_text())
    (bread,) = [c for c in plan["cards"] if c["case"] == "pair"]
    tree = cg.Source(str(root)).tree(bot)
    fixed = b"the fixed PSD"                                                # land only moves bytes
    (out / "fixed.psd").write_bytes(fixed)
    bread["writes"] = [{"path": bread["psd"], "file": "fixed.psd", "sha": blob_sha(fixed), "kind": "psd",
                        "line": "R11 fixed; Single-line Composer (Ability)", "shown": "R11 fixed", "composer": True, "pixels": ["apostrophes"]}]
    bread["expect"] = cg._expect(cg.Job(bread["psd"], "pair", "", tree[bread["psd"]][1], f"{FOLDER}/Bread.png",
                                        png_new=tree[f"{FOLDER}/Bread.png"][1]))
    assert bread["expect"][f"{FOLDER}/Bread.png"] == tree[f"{FOLDER}/Bread.png"][1]
    plan["cards"] = [bread]
    (out / "plan.json").write_text(json.dumps(plan))
    report = cg.land(str(out / "plan.json"), local=str(root))
    assert report["landed"] == ["Bread"] and report["dropped"] == [] and report["sha"]
    assert _git(root, "show", "--name-only", "--format=", report["sha"]).splitlines() == [f"{FOLDER}/Bread.psd"]
    assert report["events"] == [{"kind": "guard-fix", "card": "Bread", "path": f"{FOLDER}/Bread.psd",    # never the composer
                                 "line": "R11 fixed; straight apostrophes"}]


def test_a_duplicate_follows_its_psd_silently_and_records_land(cards, tmp_path):
    """Owner 11-07: a picture-only duplicate (dup_of) takes its card's change in the guards' landing, silent; one drawn
    apart from the PSD stays. A landed write hands what it read before the strip to records.json (card_data --records)."""
    root, base, alice, bot = cards
    work = tmp_path / "psds"
    green = (root / FOLDER / "Bread.png").read_bytes()
    (root / FOLDER / "Bread_Copy.png").write_bytes(green)                     # the card's own picture under another name
    _png(root, f"{FOLDER}/Bread_Other.png", card(colour=(255, 255, 0), base=(0, 0, 0)))    # drawn apart
    (root / "site").mkdir()
    (root / "site/Card Edits.json").write_text(json.dumps({"cards": {"Bread Copy": {"dup_of": "Bread"},
                                                                     "Bread Other": {"dup_of": "Bread"}}}))
    start = _commit(root, "duplicates")
    blue = card(colour=(0, 0, 255), box=(30, 40, 50, 60))
    _psd(root, work, f"{FOLDER}/Bread.psd", blue)
    _png(root, f"{FOLDER}/Bread.png", blue)
    pair = _commit(root, "Bread again", author="Alice")
    out = tmp_path / "out"
    cg.check(str(root), pair, start, mode="on", out=str(out), workers=1)
    plan = json.loads((out / "plan.json").read_text())
    (bread,) = [c for c in plan["cards"] if c["card"] == "Bread"]
    (write,) = [w for w in bread["writes"] if w.get("dup")]
    assert write["path"] == f"{FOLDER}/Bread_Copy.png" and write["silent"] and cg.events_of(bread, write, [write]) == []
    assert Image.open(out / write["file"]).convert("RGB").tobytes() == blue.convert("RGB").tobytes()
    fixed = b"the converted PSD"                                            # land only moves bytes
    (out / "fixed.psd").write_bytes(fixed)
    bread["writes"].append({"path": bread["psd"], "file": "fixed.psd", "sha": blob_sha(fixed), "kind": "psd", "line": "",
                            "shown": "", "silent": True, "pixels": ["normalise"]})
    bread["records"] = {"render": {bread["psd"]: {"icc": [0.0, 0]}}}
    (out / "plan.json").write_text(json.dumps(plan))
    landed = cg.land(str(out / "plan.json"), local=str(root))
    assert landed["sha"] and f"{FOLDER}/Bread_Copy.png" in _git(root, "show", "--name-only", "--format=", landed["sha"])
    cg.finish(str(out / "plan.json"), landed)
    assert json.loads((out / "records.json").read_text()) == [bread["records"]]


# ========================
# The real push of 2026-10-22 15:18 (the clone, read-only)
# ========================
def test_replay_of_the_stale_png_push(clone, tmp_path):
    from importlib.util import find_spec
    if find_spec("scipy") is None:
        pytest.skip("God Tamer's gradient fills need psd-tools[composite] (scipy), as the Action installs")
    if pe.tool() is None:
        pytest.skip("the 16 recompressions need the oxipng CLI (as the Action installs) or pyoxipng")
    source = cg.Source(str(clone))
    if not (source.commit("38c55bd3") and source.commit("07589cc7")):
        pytest.skip("the clone lacks the push")
    data = cg.check(str(clone), "07589cc7", "38c55bd3", out=str(tmp_path / "out"), workers=1, cards={"god tamer"})
    by = {c["card"]: c for c in data["cards"]}
    assert list(by) == ["God Tamer"] and by["God Tamer"]["export"].get("outcome") == "current"
    assert all(w["kind"] == "recompress" or w["composer"] and w["silent"] and not w["pixels"]    # nothing re-exported;
               for c in data["cards"] for w in c["writes"])                                     # R8's key written unsaid
    assert by["God Tamer"]["changes"] == ["added Star group", "Group 2 renamed 'Stars'", "Star/Delicate Flower moved into Star"]
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
    code = ("import sys, commands.card_guards, commands.txt2_repair, commands.ability_place; "
            "print(sorted(m for m in ('config', 'discord', 'commands.psd_analyzer', 'pandas') if m in sys.modules)); "
            "print(sorted(m for m in sys.modules if m.startswith('commands.') or m == 'global_config'))")
    done = subprocess.run([executable, "-c", code], cwd=bot, capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    bad, closure = done.stdout.splitlines()
    assert bad == "[]"
    modules = {m.replace(".", "/") + ".py" for m in json.loads(closure.replace("'", '"'))}
    assert modules == {f for f in cg.SPARSE_FILES if f.endswith(".py")}
    assert "requirements-cards.txt" in cg.SPARSE_FILES                         # what the Action installs
    assert "commands/ability_index.json" in cg.SPARSE_FILES                    # +ability's sibling facts
    from helpers import ALLOWED_IMPORTS, third_party_imports
    extra = third_party_imports("import commands.card_guards, commands.txt2_repair", bot) - ALLOWED_IMPORTS
    assert not extra, f"card-guards.yml installs none of these: {sorted(extra)}"
    for module in ("card_check", "font_check"):
        done = subprocess.run([executable, "-m", f"commands.{module}", "--help"], cwd=bot, capture_output=True, text=True,
                              timeout=120)
        assert done.returncode == 0 and done.stdout.startswith(f"usage: python -m commands.{module}"), done.stderr
        code = f"import sys, commands.{module}; print(sorted(m for m in ('config', 'discord') if m in sys.modules))"
        done = subprocess.run([executable, "-c", code], cwd=bot, capture_output=True, text=True, timeout=120)
        assert done.returncode == 0 and done.stdout.strip() == "[]", done.stderr


def test_guard_modules_stay_uncompiled():
    """Not in setup.py's INCLUDE_FILES: the Action runs them as pure Python (the change-control skill's step 5)."""
    setup = (REPO / "setup.py").read_text(encoding="utf-8")
    for module in ("card_guards", "png_export", "card_standards", "psd_diff"):
        assert f"commands/{module}.py" not in setup


def _workflow(clone) -> Path:
    site = Path(os.path.expanduser(os.environ.get("AC_SITE_SRC") or str(CLONE)))       # a TTSCardMaker checkout (code, not cards)
    path = WORKFLOW or (site / ".github" / "workflows" / "card-guards.yml")
    if not path.exists():
        pytest.skip(f"no {path} (set CARD_GUARDS_YML to a TTSCardMaker checkout's file, or AC_SITE_SRC to the checkout)")
    return path


SYNC = "workflow run card-sync.yml -f dry_run=false -f bake=pending"   # Follow up's card sync (its data commit builds the site)



def _guard_steps(clone) -> tuple:
    """card-guards.yml's jobs read as one: (doc, every named step by name, the jobs' env merged: TOLD, STREAK)."""
    yaml = pytest.importorskip("yaml")
    doc = yaml.safe_load(_workflow(clone).read_text(encoding="utf-8"))
    jobs = doc["jobs"].values()
    return (doc, {s["name"]: s for job in jobs for s in job.get("steps", []) if s.get("name")},
            {k: v for job in jobs for k, v in job.get("env", {}).items()})


def test_workflow_shape(clone):
    """card-guards.yml: push on PSDs, PNGs and fonts, the guards behind CARD_GUARDS (the card and font checks always,
    2026-10-25), dry dispatches by default, a run per push, the sparse list = SPARSE_FILES, requirements-cards.txt's
    install (the fork's one pin) in both jobs that run the guards, as the other card workflows have it (one cache entry),
    secrets by name only, no ping, no fonts for the guards' own check (R7 is the font check's). The jobs: check (lands,
    then ends after the comment), checks (a push's two checks beside it, 2026-11-07), post (at once since 2026-11-03: no
    environment, no quiet window, no setup; only with something to land or tell), finish (card errors, the alert)."""
    path = _workflow(clone)
    text = path.read_text(encoding="utf-8")
    doc, steps_by_name, env = _guard_steps(clone)
    on = doc[True]
    assert on["push"] == {"branches": ["main"], "paths": ["**/*.psd", "**/*.PSD", "**/*.png", "**/*.PNG", "MDW/Fonts/*",
                                                   "Types/**", "MDW/Linked/**"]}          # linked files: G2, G3
    inputs = on["workflow_dispatch"]["inputs"]
    assert inputs["dry_run"]["default"] is True and inputs["all"]["default"] is False and set(inputs) == {"dry_run", "sha", "cards", "all"}
    assert doc["concurrency"]["group"] == ("${{ github.event_name == 'push' && format('card-guards-{0}', github.sha) || "
                                           "format('card-guards-{0}', github.run_id) }}") and doc["concurrency"]["cancel-in-progress"] is False
    assert list(doc["jobs"]) == ["check", "checks", "post", "finish", "caches"]  # caches: prunes old wheel and font entries
    job, checks, post, finish = (doc["jobs"][name] for name in ("check", "checks", "post", "finish"))
    assert "if" not in job and steps_by_name["Check"]["if"] == "env.MODE != 'off'"     # the checks run whatever MODE says
    assert job["env"]["MODE"] == ("${{ github.event_name == 'push' && (vars.CARD_GUARDS || 'off') || "
                                  "((inputs.dry_run || inputs.all) && 'dry' || 'on') }}")
    # no quiet window (owner 2026-11-03): the post job takes no environment and Land (the check job's) waits for nothing
    assert post["needs"] == "check" and finish["needs"] == ["check", "checks", "post"]
    # beside check, whatever MODE says; a dispatch's stay in check, so `all` reads each PSD once
    assert "needs" not in checks and checks["if"] == "github.event_name == 'push'" and "env" not in checks
    here = "(github.event_name == 'push') == (github.job == 'checks')"
    assert steps_by_name["Card check"]["if"] == here and steps_by_name["Font check"]["if"].startswith(here + " && ")
    assert steps_by_name["Card check"]["run"] == steps_by_name["Font check"]["run"]     # one script, by $CHECK
    named = [s.get("name") for s in checks["steps"] if s.get("name")]
    assert named == ["Pushed blobs start", "Pushed blobs", "Card check", "Font check", "Tell the checks"]
    assert "Tell the checks" not in [s.get("name") for s in job["steps"]]
    fonts = [next(s for s in one["steps"] if s.get("id") == "font-cache") for one in (job, checks)]
    assert fonts[0]["with"] == fonts[1]["with"] and fonts[1]["uses"].startswith("actions/cache/restore@")   # one saver
    assert "environment" not in post
    assert post["if"] == ("!cancelled() && needs.check.outputs.checked == 'success' && needs.check.outputs.land == 'true' && "
                          "(needs.check.outputs.mode == 'report' || needs.check.outputs.mode == 'on')")
    assert finish["if"] == "always() && needs.check.outputs.mode != 'off'"       # nothing when the guards are off
    assert job["timeout-minutes"] == "${{ inputs.all && 150 || 60 }}" and post["timeout-minutes"] == 10
    assert job["permissions"] == {"contents": "write", "actions": "write"}           # Land's commit, Start Card sync
    assert "environment" not in job and "environment" not in finish
    assert {k: v for k, v in job["outputs"].items() if k != "saved"} == {"mode": "${{ steps.start.outputs.mode }}",
                              "started": "${{ steps.start.outputs.started }}", "checked": "${{ steps.check.outcome }}",
                              "comment": "${{ steps.comment.outputs.id }}", "told": "${{ steps.comment.outputs.told }}",
                              "land": "${{ steps.plan.outputs.land }}", "card_errors": "${{ steps.plan.outputs.card_errors }}",
                              "fixes": "${{ steps.plan.outputs.fixes }}", "early": "${{ steps.early.outputs.started }}",
                              "early_run": "${{ steps.early.outputs.run }}",
                              "land_ran": "${{ steps.land.outputs.ran }}", "land_sha": "${{ steps.land.outputs.sha }}",
                              "land_already": "${{ steps.land.outputs.already }}"}
    assert post["outputs"] == {"posted": "${{ steps.tell.outputs.posted }}"}
    assert post["env"]["MODE"] == "${{ needs.check.outputs.mode }}"
    assert post["env"]["TOLD"] == job["env"]["TOLD"] == finish["env"]["TOLD"]
    for one in (job, checks):
        bot = next(s for s in one["steps"] if s.get("with", {}).get("repository") == "bobberlington/AutoCroissant")
        assert bot["with"]["sparse-checkout"].split() == ["/" + f for f in cg.SPARSE_FILES] and "ref" not in bot["with"]
    assert not [s for s in post["steps"] if "uses" in s and "artifact" not in s["uses"]]     # no checkout, no setup
    for one in (job, checks, post, finish):                                    # a copied step runs twice unseen
        named = [s["name"] for s in one["steps"] if s.get("name")]
        assert len(named) == len(set(named)), named
    cards = next(s for s in job["steps"] if s.get("uses", "").startswith("actions/checkout") and "repository" not in s.get("with", {}))
    assert cards["with"]["filter"] == "blob:none" and cards["with"]["fetch-depth"] == 0
    assert post["env"]["GH_REPO"] == "${{ github.repository }}"               # gh workflow run without the card repo
    from helpers import FORK_KEY, FORK_WORKFLOWS, fork_install
    others = [path.parent / name for name in FORK_WORKFLOWS if name != "card-guards.yml" and (path.parent / name).is_file()]
    keys = {p.name: fork_install(p) for p in [path, *others]}                  # one pin: requirements-cards.txt's
    assert set(keys.values()) == {FORK_KEY}                                    # one entry for the six, on 3.14
    assert "--fonts" not in steps_by_name["Check"]["run"]                     # R7 is the font check's
    names = [s.get("name") for s in job["steps"]]
    assert names.index("Start") < names.index("No push run in its window") < names.index("Check") \
        < names.index("Plan for the post job") < names.index("Fixes to land") < names.index("Decided") \
        < names.index("Types for the conversion") < names.index("Normalise") < names.index("Land") < names.index("Landed") \
        < names.index("Comment on the commit")
    assert names.index("Start Card sync") == names.index("Fixes to land") + 1               # Card sync sets up while the guards land
    types, normalise = steps_by_name["Types for the conversion"], steps_by_name["Normalise"]     # after Decided: Card sync waits for neither
    assert types["env"] == {"GIT_NO_LAZY_FETCH": "0"} and "sparse.expectFilesOutsideOfPatterns true" in types["run"]
    assert normalise["if"] == "${{ !cancelled() && steps.types.outcome == 'success' }}" and normalise["continue-on-error"] is True
    assert "card_guards normalise" in normalise["run"] and "--deadline" in normalise["run"]
    plan = job["steps"][names.index("Plan for the post job")]
    assert plan["id"] == "plan" and plan["if"] == "always() && steps.check.outcome == 'success'"
    assert job["steps"][names.index("Decided")]["if"] == "always()"             # Card sync's wait ends on it, whatever the check did
    hand = job["steps"][names.index("Comment on the commit") + 1]
    assert hand["uses"].startswith("actions/upload-artifact") and hand["with"]["name"] == "card-guards" and hand["if"] == "always() && env.MODE != 'off'"
    assert job["steps"][names.index("Landed")]["if"] == "always()"             # a follow-up's wait ends on it, landed or not
    post_names = [s.get("name") for s in post["steps"]]
    take = post["steps"][1]
    assert take["uses"].startswith("actions/download-artifact@") and take["with"] == {"name": "card-guards", "path": "${{ runner.temp }}/guards"}
    assert post_names == ["Work dir", None, "Told before", "Follow up", "Tell Discord"]
    early = steps_by_name["Start Card sync"]    # with fixes and no Normalise, Card sync sets up while the guards land (its lanes wait for Landed)
    assert early["if"] == "${{ !cancelled() && steps.plan.outputs.fixes == 'true' && steps.plan.outputs.normalise == 0 }}"
    assert early["continue-on-error"] is True and early["env"]["GH_REPO"] and "-F return_run_details=true" in early["run"]
    assert '-f "inputs[guards]=$GITHUB_RUN_ID"' in early["run"] and "card-sync.yml/dispatches" in early["run"]
    assert all(f'-f "inputs[{k}]={v}"' in early["run"] for k, v in (("dry_run", "false"), ("bake", "pending")))   # as SYNC's
    assert steps_by_name["Follow up"]["env"]["EARLY"] == "${{ needs.check.outputs.early }}"
    assert steps_by_name["Follow up"]["env"]["EARLY_RUN"] == "${{ needs.check.outputs.early_run }}"
    assert [s.get("name") for s in finish["steps"] if s.get("name")] == ["Fail on card errors", "Tell Discord the run failed"]
    steps = steps_by_name
    tell = steps["Tell Discord"]
    assert tell["env"]["WEBHOOK"] == "${{ secrets.DISCORD_STANDARDS_WEBHOOK }}" and "discord-final.json" in tell["run"]
    assert tell["if"] == "always() && steps.told.outputs.told != 'true'"
    assert steps["Told before"]["env"]["CHECK_TOLD"] == "${{ needs.check.outputs.told }}" and "if" not in steps["Told before"]
    assert '"$WEBHOOK"' in tell["run"] and "echo \"$WEBHOOK" not in text and "set -x" not in text
    land = steps["Land"]
    # Land goes before the Comment step (a failed comment cannot drop the writes), only with something to land or tell
    assert land["if"] == "${{ !cancelled() && steps.plan.outputs.land == 'true' && (env.MODE == 'report' || env.MODE == 'on') }}"
    assert "--dry-run" in land["run"] and "steps.check.outcome == 'success'" in steps["Comment on the commit"]["if"]
    assert "discord.json" not in tell["run"].replace("discord-final.json", "")      # never check's early post
    assert steps["Fail on card errors"]["env"]["ERRORS"] == "${{ needs.check.outputs.card_errors }}"
    alert = steps["Tell Discord the run failed"]
    # a re-run of a push already told (the comment's mark) never alerts: "one post per push"
    assert alert["if"] == ("(failure() || contains(needs.*.result, 'failure') || contains(needs.*.result, 'cancelled')) && "
                           "github.event_name == 'push' && (needs.checks.result != 'success' || "
                           "(needs.post.outputs.posted != 'true' && needs.check.outputs.told != 'true'))")
    assert alert["env"]["WEBHOOK"] == "${{ secrets.DISCORD_STANDARDS_WEBHOOK }}" and "allowed_mentions: {parse: []}" in alert["run"]
    assert alert["env"]["RESULTS"] == "${{ needs.check.result }} ${{ needs.checks.result }} ${{ needs.post.result }}"
    assert '{"check": 60, "checks": %d, "post": %d}' % (checks["timeout-minutes"], post["timeout-minutes"]) in alert["run"]  # a push's (`all`: 150)
    assert ">= %d ))" % (60 * 60 - 120) in alert["run"]                       # no job list: the check's own
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
    assert "report-landed.md" in follow["run"] \
        and follow["if"] == "always() && (needs.check.outputs.land_ran == 'true' || needs.check.outputs.fixes == 'true')"
    assert follow["id"] == "follow" and tell["env"]["REFRESH"] == "${{ steps.follow.outputs.refresh }}"
    assert follow["env"]["TOLD_BEFORE"] == "${{ steps.told.outputs.told }}"
    # a re-run uploads again: the earlier attempt's artifact of the same name is replaced, not a 409
    uploads = [s for s in job["steps"] + post["steps"] + finish["steps"] if s.get("uses", "").startswith("actions/upload-artifact")]
    assert len(uploads) == 4 and all(s["with"]["overwrite"] is True for s in uploads)
    landed = job["steps"].index(steps["Landed"])                               # a follow-up's wait reads both before it
    events = next(s for s in job["steps"] if s.get("with", {}).get("name") == "guard-events")       # Card sync reads it
    assert job["steps"].index(steps["Land"]) < job["steps"].index(events) < landed and events["if"] == "always() && steps.land.outputs.sha != ''"
    records = next(s for s in job["steps"] if s.get("with", {}).get("name") == "guard-records")     # Card sync merges it
    assert job["steps"].index(records) < landed and records["with"]["path"].endswith("/records.json")



def test_oxipng_cli_rides_in_the_venv(clone, tmp_path):
    """cards-python's oxipng step: a miss fetches the CLI into the venv in every job, a hit fetches nothing, PATH only
    when asked; a fetch that fails fails a job that asks, else leaves the venv unsaved."""
    import hashlib
    import tarfile
    from helpers import FORK_ACTION
    yaml = pytest.importorskip("yaml")
    action = yaml.safe_load((_workflow(clone).parents[2] / FORK_ACTION).read_text(encoding="utf-8"))
    step = next(s for s in action["runs"]["steps"] if s.get("id") == "oxipng")
    member = f"oxipng-{step['env']['VERSION']}-x86_64-unknown-linux-musl/oxipng"
    tgz, shim = tmp_path / "ox.tar.gz", tmp_path / "shim"
    with tarfile.open(tgz, "w:gz") as tar:
        info, body = tarfile.TarInfo(member), b"#!/bin/sh\necho oxipng\n"
        info.size, info.mode = len(body), 0o755
        tar.addfile(info, BytesIO(body))
    shim.mkdir()
    (shim / "curl").write_text('#!/bin/sh\necho "$@" >> "$CURLS"\n[ -z "$FAIL" ] || exit 22\n'
                               'while [ "$1" != -o ]; do shift; done; cp "$TGZ" "$2"\n')
    (shim / "sha256sum").write_text(f"#!{executable}\nimport hashlib, sys\nsha, path = sys.stdin.read().split()\n"
                                    "sys.exit(hashlib.sha256(open(path, 'rb').read()).hexdigest() != sha)\n")
    for one in shim.iterdir():
        one.chmod(0o755)
    venv = tmp_path / "venv"

    def run(want, fail="", sha=hashlib.sha256(tgz.read_bytes()).hexdigest()):
        files = {k: tmp_path / k for k in ("out", "path", "curls")}
        for f in files.values():
            f.write_text("")
        env = {**os.environ, "PATH": f"{shim}{os.pathsep}{os.environ['PATH']}", "RUNNER_OS": "Linux", "RUNNER_ARCH": "X64",
               "RUNNER_TEMP": str(tmp_path), "VERSION": step["env"]["VERSION"], "SHA256": sha, "ASSET": "1", "TOKEN": "t",
               "BIN": str(venv / ".oxipng"), "WANT": want, "FAIL": fail, "TGZ": str(tgz), "CURLS": str(files["curls"]),
               "GITHUB_OUTPUT": str(files["out"]), "GITHUB_PATH": str(files["path"])}
        done = subprocess.run(["bash", "--noprofile", "--norc", "-eo", "pipefail", "-c", step["run"]], env=env,
                              capture_output=True, text=True, timeout=60)
        return done.returncode, len(files["curls"].read_text().splitlines()), files["path"].read_text(), files["out"].read_text()

    assert run("false") == (0, 1, "", "")                                  # a data job's miss: fetched, off PATH
    assert (venv / ".oxipng" / "oxipng").is_file()
    assert run("true", fail="1") == (0, 0, f"{venv / '.oxipng'}\n", "")   # a hit: no network
    shutil.rmtree(venv)
    assert run("false", fail="1") == (0, 2, "", "unsaved=true\n")          # both links down: the entry waits
    assert not (venv / ".oxipng").exists()
    assert run("true", sha="0" * 64)[:2] == (1, 1)                        # a bad file fails a job that writes PNGs


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
    if path.endswith("/jobs"):                  # the attempt's jobs: FAKE_JOBS, the API's {"jobs": [...]}; FAKE_JOBS_LATER from look FAKE_LOOKS on
        late = sum("/jobs --jq" in line for line in db["log"]) > int(os.environ.get("FAKE_LOOKS", "1000"))
        out = jq(json.loads(os.environ["FAKE_JOBS_LATER"] if late else os.environ.get("FAKE_JOBS", '{"jobs": []}')))
    elif "/runs?" in path:                      # a workflow's runs: FAKE_GUARD_RUNS
        out = jq(json.loads(os.environ["FAKE_GUARD_RUNS"]))
    elif "/git/trees/" in path:                 # the PSD sizes: FAKE_TREE
        out = os.environ["FAKE_TREE"]
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
    pytest.importorskip("yaml")
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
    pytest.importorskip("yaml")
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


def test_workflow_alert_tells_a_timeout_from_a_cancel(clone, tmp_path):
    """Every job is in the alert's limits: one missing reads as timed out, so a cancel by hand would tell Discord."""
    pytest.importorskip("yaml")
    if not shutil.which("jq"):
        pytest.skip("no jq")
    doc, steps, jenv = _guard_steps(clone)
    posts = tmp_path / "posts.txt"
    env = {"PATH": f"{_fake_tools(tmp_path)}:{os.environ['PATH']}", "OUT": str(tmp_path), "GITHUB_REPOSITORY": "o/cards",
           "WEBHOOK": "https://discord.invalid/hook", "FAKE_STATE": str(tmp_path / "gh.json"), "FAKE_POSTS": str(posts),
           "GITHUB_STEP_SUMMARY": str(tmp_path / "summary.md"), "STREAK": jenv["STREAK"], "RUNNER_TEMP": str(tmp_path / "runner"),
           "GITHUB_RUN_ID": "7", "GITHUB_RUN_ATTEMPT": "1", "GITHUB_ACTOR": "Alice", "CHECKED": "failure", "FAKE_RUNS": "[]",
           "RUN": "https://github.com/o/cards/actions/runs/7", "TOLD": jenv["TOLD"], "GITHUB_SHA": "f" * 40,
           "RESULTS": "success cancelled skipped"}
    def cancelled(minutes):
        return json.dumps({"jobs": [{"name": "checks", "conclusion": "cancelled", "started_at": "2026-11-07T10:00:00Z",
                                     "completed_at": f"2026-11-07T10:{minutes:02}:00Z"}]})
    for minutes, sent in ((1, 0), (39, 1)):
        posts.write_text("")
        _run_steps(steps, ["Tell Discord the run failed"], {**env, "FAKE_JOBS": cancelled(minutes)}, {})
        assert len(posts.read_text().splitlines()) == sent, minutes


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


FAKE_PYTHON = r"""#!/bin/sh
echo "$*" >> "$FAKE_PYTHON_ARGS"
printf '%s' "$FAKE_LANDED" > "$OUT/landed.json"
exit "${FAKE_LAND_STATUS:-0}"
"""


def test_workflow_land_and_a_rerun_after_a_landing(clone, tmp_path):
    """Land (the check job's) runs only with something to tell in report mode, as its writes are told, never made; on
    mode for a write too: the check job's plan says land=true. A re-run or a dispatch passes --rerun (an unread branch tells no write as not made), and the
    step hands on the sha and the count already on the branch. Follow up starts the card sync after a
    commit, and again on a re-run that finds the writes landed (the first attempt's may not have started), not in report
    mode and not on a first attempt; with fixes (a push in on mode with writes: its Card sync main lane waits for them)
    always, landed or not."""
    pytest.importorskip("yaml")
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
           "GITHUB_STEP_SUMMARY": str(tmp_path / "summary.md"), "FAKE_PYTHON_ARGS": str(args),
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
        planned, plan_out = _shell(steps["Plan for the post job"], {**env, "MODE": mode, "GITHUB_EVENT_NAME": event})
        assert planned.returncode == 0 and (plan_out.get("land") == "true") is ran, (mode, told, writes, plan_out)
        fixes = mode == "on" and event == "push" and writes > 0       # Card sync's push run leaves its main lane to Follow up
        assert (plan_out.get("fixes") == "true") is fixes, (mode, event, writes, plan_out)
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
    for mode, attempt, sha, already, fixes, started in (("on", "2", "", "1", "", True), ("on", "1", "", "1", "", False),
                                                        ("report", "2", "", "1", "", False), ("on", "2", "", "0", "", False),
                                                        ("on", "1", "c" * 40, "", "", True), ("on", "2", "", "", "", False),
                                                        ("on", "1", "", "", "true", True)):   # fixes, nothing landed
        state.write_text(json.dumps({"comments": {}, "next": 101, "log": []}))
        done, outputs = _shell(steps["Follow up"], {**env, "MODE": mode, "GITHUB_RUN_ATTEMPT": attempt, "SHA": sha,
                                                    "ALREADY": already, "COMMENT": "", "TOLD_BEFORE": "", "FIXES": fixes,
                                                    "LAND_RAN": "true"})
        log = [c for c in json.loads(state.read_text())["log"] if c.startswith("workflow run")]
        assert done.returncode == 0 and log == ([SYNC] if started else []), \
            (mode, attempt, sha, already, fixes, log)
    lane = lambda name, conclusion=None: {"name": name, "status": "completed" if conclusion else "queued", "conclusion": conclusion}  # noqa: E731
    for run, jobs, again in (("5", [lane("Card data"), lane("Bake and save")], False),     # Start Card sync did: none again
                             ("5", [lane("Card data", "success"), lane("Bake and save", "cancelled")], True),   # a newer run took its place
                             ("5", [lane("Card data", "cancelled"), lane("Bake and save")], True),
                             ("", [], True)):                                             # no run id: one more
        state.write_text(json.dumps({"comments": {}, "next": 101, "log": []}))
        done, _ = _shell(steps["Follow up"], {**env, "MODE": "on", "GITHUB_RUN_ATTEMPT": "1", "SHA": "c" * 40, "ALREADY": "", "COMMENT": "",
                                              "TOLD_BEFORE": "", "FIXES": "true", "LAND_RAN": "true", "EARLY": "true", "EARLY_RUN": run,
                                              "FAKE_JOBS": json.dumps({"jobs": jobs})})
        log = json.loads(state.read_text())["log"]
        assert done.returncode == 0 and [c for c in log if c.startswith("workflow run")] == ([SYNC] if again else []), (run, jobs, log)
        assert any(c.startswith(f"api repos/o/cards/actions/runs/{run}/jobs") for c in log) is bool(run)


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
    pytest.importorskip("yaml")
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


def test_workflow_fetches_pushed_blobs_in_the_background(clone, tmp_path, monkeypatch):
    """Pushed blobs start, right after the card checkout, fetches the push's blobs in the background and returns at once
    (its output not held); Check starts at once and waits for each listed blob (CARD_BLOBS_PENDING); Pushed blobs, after
    Decided, waits and counts. A fetch that fails: those blobs by REST, the warning and a failed step."""
    import commands.card_bake as cb
    pytest.importorskip("yaml")
    if not shutil.which("timeout"):
        pytest.skip("no timeout (coreutils)")
    doc, steps, _ = _guard_steps(clone)
    check = [s.get("name") or s.get("uses") for s in doc["jobs"]["check"]["steps"]]
    assert check[1:3] == ["actions/checkout@v7", "Pushed blobs start"]
    assert check.index("Pushed blobs") == check.index("Decided") + 1 and steps["Check"]["env"][cb.PENDING_ENV] == "${{ runner.temp }}/blobs"
    rest = []
    monkeypatch.setattr(cb.github_api, "fetch_verified", lambda repo, sha, timeout: rest.append(sha) or subprocess.run(
        ["git", "-C", str(origin), "cat-file", "blob", sha], capture_output=True, check=True).stdout)
    work, origin = tmp_path / "work", tmp_path / "origin.git"
    work.mkdir()
    _git(work, "init", "-q", "-b", "main")
    shas = []
    for side in "ab":
        for name in ("A.psd", "A.png", "B.psd"):
            (work / name).write_bytes(os.urandom(4096))
        shas.append(_commit(work, side))
    subprocess.run(["git", "clone", "-q", "--bare", str(work), str(origin)], check=True)
    _git(origin, "config", "uploadpack.allowFilter", "true")
    _git(origin, "config", "uploadpack.allowAnySHA1InWant", "true")
    for n, (url, code, said) in enumerate(((origin, 0, "6 of 6 pushed blobs in the clone"),
                                           (tmp_path / "gone.git", 1, "::warning::0 of 6 pushed blobs fetched: the rest by REST"))):
        repo, temp = tmp_path / f"clone{n}", tmp_path / f"temp{n}"
        temp.mkdir()
        subprocess.run(["git", "clone", "-q", "--filter=blob:none", "--no-checkout", f"file://{origin}", str(repo)], check=True)
        _git(repo, "remote", "set-url", "origin", f"file://{url}")
        _git(repo, "config", "remote.origin.uploadpack", "sleep 1; git-upload-pack")      # GitHub's time
        env = {"PATH": os.environ["PATH"], "HOME": os.environ.get("HOME", ""), "RUNNER_TEMP": str(temp), "BEFORE": shas[0],
               "GITHUB_SHA": shas[1], "GITHUB_OUTPUT": str(temp / "output")}
        done, _ = _shell(steps["Pushed blobs start"], env, cwd=repo)
        assert done.returncode == 0 and not (temp / "blobs.done").exists(), (done.stdout, done.stderr)
        monkeypatch.setenv(cb.PENDING_ENV, str(temp / "blobs"))
        listed = (temp / "blobs").read_text().split()
        rest.clear()
        assert [blob_sha(cb.fetch_verified("o/r", sha, local=str(repo))) for sha in listed] == listed
        assert len(rest) == (len(listed) if code else 0)            # waited for, from the clone; a failed fetch: REST
        done, _ = _shell(steps["Pushed blobs"], env, cwd=repo)
        assert (done.returncode != 0) == bool(code) and said in " ".join(done.stdout.split()), (done.stdout, done.stderr)
    os.utime(temp / "blobs", (0, 0))                                # no .done: the cap ends the wait (a new process)
    (temp / "blobs.done").unlink()
    monkeypatch.setattr(cb, "_listed", {})
    assert cb._pending_blob(str(repo), listed[0]) is None
    monkeypatch.delenv(cb.PENDING_ENV)                              # the bake: no list, no wait
    assert cb._pending_blob(str(repo), listed[0]) is None


def test_workflow_lowercases_card_guards(clone, tmp_path):
    """The job's `if` matches CARD_GUARDS in any case ("On"), the shell does not: Start lowercases MODE for every later
    step (Land's --dry-run test is exact; `land` would also refuse a report plan without it)."""
    pytest.importorskip("yaml")
    if not shutil.which("jq"):
        pytest.skip("no jq")
    doc, steps, jenv = _guard_steps(clone)
    step = steps["Start"]
    for typed, mode in (("On", "on"), ("REPORT", "report"), ("dry", "dry")):
        github_env = tmp_path / f"env-{typed}"
        github_env.write_text("")
        done, _ = _shell(step, {"PATH": os.environ["PATH"], "MODE": typed, "RUNNER_TEMP": str(tmp_path),
                                  "GITHUB_RUN_ATTEMPT": "1", "GITHUB_EVENT_NAME": "push",
                                  "GITHUB_ENV": str(github_env), "GITHUB_OUTPUT": str(tmp_path / "start.output")})
        assert done.returncode == 0 and f"MODE={mode}" in github_env.read_text().splitlines(), done.stderr
    land = steps["Land"]
    assert '[ "$MODE" = report ] && args+=(--dry-run)' in land["run"]


def test_card_syncs_wait_ends_at_decided(clone, tmp_path):
    """Card sync's "Guards' fixes" against a fake gh: the wait ends at the check job's Decided step or, failing that, its end
    (never at another job's), and the hold rule then reads Fixes to land. It looks every 3 s, the run every 5th look."""
    yaml = pytest.importorskip("yaml")
    if not shutil.which("jq"):
        pytest.skip("no jq")
    doc = yaml.safe_load((_workflow(clone).parent / "card-sync.yml").read_text(encoding="utf-8"))
    guards = next(s for s in doc["jobs"]["main"]["steps"] if s.get("name") == "Guards' fixes")
    assert guards["env"]["DECIDED"] == "Decided"
    bin_dir = _fake_tools(tmp_path)
    naps = tmp_path / "sleeps.txt"
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    state = tmp_path / "gh.json"
    env = {"PATH": f"{bin_dir}:{os.environ['PATH']}", "FAKE_STATE": str(state), "FAKE_SLEPT": str(naps), "GITHUB_REPOSITORY": "o/cards", "SHA": "a" * 40,
           "GITHUB_OUTPUT": str(tmp_path / "out"), "GITHUB_STEP_SUMMARY": str(tmp_path / "summary.md"), "GITHUB_SERVER_URL": "https://x",
           "FAKE_GUARD_RUNS": json.dumps({"workflow_runs": [{"id": 7, "status": "in_progress", "created_at": now}]}),
           **{k: guards["env"][k] for k in ("FIXES", "DECIDED", "FOLLOW")}}

    def jobs(status="in_progress", decided="completed", fixes="skipped", name="check"):
        steps = [{"name": "Fixes to land", "status": "completed", "conclusion": fixes}]
        steps += [{"name": "Decided", "status": decided, "conclusion": "success" if decided == "completed" else None}] if decided else []
        return json.dumps({"jobs": [{"name": name, "status": status, "steps": steps}]})

    for answer, later, sleeps, held in ((jobs(), None, 0, False),
                                        (jobs(fixes="success"), None, 0, True),
                                        (jobs("completed", decided=None), None, 0, False),            # a run from before Decided
                                        (jobs(decided="pending"), jobs(fixes="success"), 7, True),
                                        (jobs(name="checks"), jobs(), 2, False)):                     # the wrong job's step: not it
        state.unlink(missing_ok=True)
        naps.write_text("")
        done, outputs = _shell(guards, {**env, "FAKE_JOBS": answer, **({"FAKE_JOBS_LATER": later, "FAKE_LOOKS": str(sleeps)} if later else {})})
        assert done.returncode == 0, done.stderr
        assert naps.read_text().split() == ["3"] * sleeps and ("run" in outputs) is held, (answer, outputs)
    runs = [line for line in json.loads(state.read_text())["log"] if "/runs?" in line]
    assert len(runs) == 1                                                       # the first read only: 2 looks, none the 5th
    state.unlink()
    done, _ = _shell(guards, {**env, "FAKE_JOBS": jobs(decided="pending"), "FAKE_JOBS_LATER": jobs(), "FAKE_LOOKS": "11"})
    assert done.returncode == 0 and sum("/runs?" in line for line in json.loads(state.read_text())["log"]) == 3    # looks 5 and 10


def test_card_syncs_follow_up_waits_for_land(clone, tmp_path):
    """A follow-up (input guards, started by the guards' check job before Land): both lanes set up, then "Guards' fixes"
    waits until that run's check job reaches Landed or ends (never another job's step), holds nothing, then Newest."""
    yaml = pytest.importorskip("yaml")
    if not shutil.which("jq"):
        pytest.skip("no jq")
    doc = yaml.safe_load((_workflow(clone).parent / "card-sync.yml").read_text(encoding="utf-8"))
    for job, at, fonts, history in (("data", 5, [], ["History wait"]),        # data: Site build start, History start
                                    ("main", 3, ["Fonts first", None, "Fonts first (cache miss)"], [])):
        names = [s.get("name") for s in doc["jobs"][job]["steps"]]
        want = ["Python environment", *fonts, "Guards' fixes", *history, "Newest"]
        assert names[at:at + len(want)] == want, (job, names)
    assert doc[True]["workflow_dispatch"]["inputs"]["guards"]["default"] == ""       # the guards' run id
    guards = next(s for s in doc["jobs"]["data"]["steps"] if s.get("name") == "Guards' fixes")
    naps = tmp_path / "sleeps.txt"
    env = {"PATH": f"{_fake_tools(tmp_path)}:{os.environ['PATH']}", "FAKE_STATE": str(tmp_path / "gh.json"), "FAKE_SLEPT": str(naps),
           "GITHUB_REPOSITORY": "o/cards", "RUN": "7", "SHA": "", "GITHUB_OUTPUT": str(tmp_path / "out"),
           "GITHUB_STEP_SUMMARY": str(tmp_path / "summary.md"), "GITHUB_SERVER_URL": "https://x",
           "LANDED": guards["env"]["LANDED"]}
    assert "LANDED" in guards["run"] and "POST" not in guards["env"]

    def jobs(*pairs, name="check", status="in_progress"):
        return json.dumps({"jobs": [{"name": name, "status": status, "steps": [{"name": n, "status": st} for n, st in pairs]}]})

    land, landed = ("Land", "completed"), ("Landed", "in_progress")
    for answer, later, looks in ((jobs(land, landed), None, 0),
                                 (jobs(("Land", "in_progress"), ("Landed", "queued")), jobs(land, landed), 4),
                                 (jobs(("Land", "completed"), status="completed"), None, 0),          # cancelled or failed before it
                                 (jobs(land, landed, name="post"), jobs(land, ("Landed", "completed")), 2)):   # not its job
        (tmp_path / "gh.json").unlink(missing_ok=True)
        naps.write_text("")
        done, outputs = _shell(guards, {**env, "FAKE_JOBS": answer, **({"FAKE_JOBS_LATER": later, "FAKE_LOOKS": str(looks)} if later else {})})
        assert done.returncode == 0 and "run" not in outputs and naps.read_text().split() == ["3"] * looks, (answer, done.stderr)
        assert not any("card-guards.yml/runs" in line for line in json.loads((tmp_path / "gh.json").read_text())["log"])


def test_site_build_starts_with_the_data_job(clone, tmp_path):
    """A push's data job, a guards' follow-up and the after job dispatch the site build first, named by run and job; their
    Site build dispatches again only when that run is missing, ended, or past its 20-minute Data wait (or unread)."""
    yaml = pytest.importorskip("yaml")
    if not shutil.which("jq"):
        pytest.skip("no jq")
    workflows = _workflow(clone).parent
    sync = yaml.safe_load((workflows / "card-sync.yml").read_text(encoding="utf-8"))
    site = yaml.safe_load((workflows / "site.yml").read_text(encoding="utf-8"))
    steps = sync["jobs"]["data"]["steps"]
    start, build = steps[0], next(s for s in steps if s.get("name") == "Site build")
    assert sync["jobs"]["after"]["steps"] is steps and start["name"] == "Site build start"
    assert start["if"] == "env.DRY_RUN != 'true' && (github.event_name == 'push' || inputs.guards || github.job == 'after')"
    assert site["run-name"] == "${{ inputs.after && format('Card browser (after {0} {1})', inputs.after, inputs.job) || '' }}"
    assert build["env"]["EARLY"] == "Card browser (after ${{ github.run_id }} ${{ github.job }})"
    names = [s.get("name") or s.get("id") or s.get("uses") for s in site["jobs"]["build"]["steps"]]
    assert names.index("og-before") + 1 == names.index("Data wait") == names.index("Setup wait") - 2   # then the data checkout
    wait = site["jobs"]["build"]["steps"][names.index("Data wait")]
    assert wait["env"]["JOB"] == "${{ inputs.job == 'after' && 'Card data after the bake' || 'Card data' }}"
    assert (sync["jobs"]["data"]["name"], sync["jobs"]["after"]["name"]) == ("Card data", "Card data after the bake")
    assert wait["env"]["COMMIT"] in [s.get("name") for s in steps] and site["jobs"]["build"]["permissions"]["actions"] == "read"
    state, temp = tmp_path / "gh.json", tmp_path / "temp"
    temp.mkdir()
    env = {"PATH": f"{_fake_tools(tmp_path)}:{os.environ['PATH']}", "FAKE_STATE": str(state), "GITHUB_REPOSITORY": "o/cards",
           "GITHUB_RUN_ID": "9", "GITHUB_JOB": "data", "RUNNER_TEMP": str(temp), "GITHUB_OUTPUT": str(tmp_path / "out"),
           "GITHUB_STEP_SUMMARY": str(tmp_path / "summary.md"), "EARLY": "Card browser (after 9 data)"}
    log = lambda: json.loads(state.read_text())["log"] if state.exists() else []
    done, _ = _shell(start, env)
    for _ in range(100):
        if (temp / "site-start").exists():
            break
        subprocess.run(["/bin/sleep", "0.05"])
    assert done.returncode == 0 and log() == ["workflow run site.yml --ref main -R o/cards -f after=9 -f job=data"]
    now = int(datetime.now(timezone.utc).timestamp())
    runs = lambda *rows: json.dumps({"workflow_runs": [{"display_title": t, "status": st} for t, st in rows]})
    for at, listed, fail, again in ((now - 30, runs(("Card browser (after 9 data)", "in_progress")), "", False),
                                    (now - 30, runs(("Card browser", "in_progress"), ("Card browser (after 9 data)", "pending")), "", False),
                                    (now - 30, runs(("Card browser (after 9 after)", "in_progress"), ("Card browser (after 9 data)", "completed")), "", True),
                                    (now - 30, runs(("Card browser (after 8 data)", "in_progress")), "", True),       # not found
                                    (now - 30, runs(("Card browser (after 9 data)", "in_progress")), "/runs?", True),  # not read
                                    (now - 1200, runs(("Card browser (after 9 data)", "in_progress")), "", True),     # past its wait
                                    (None, runs(), "", True)):                                                     # never started
        state.unlink(missing_ok=True)
        (temp / "site-start").unlink(missing_ok=True)
        if at:
            (temp / "site-start").write_text(f"{at}\n")
        done, _ = _shell(build, {**env, "FAKE_GUARD_RUNS": listed, "FAKE_FAIL": fail})
        dispatched = [c for c in log() if c.startswith("workflow run")]
        assert done.returncode == 0 and dispatched == (["workflow run site.yml --ref main -R o/cards"] if again else []), (at, listed, log())
        assert sum("/runs?" in c for c in log()) == (at == now - 30)


def test_site_data_wait(clone, tmp_path):
    """site.yml's Data wait: 3 s looks until the named Card sync job's Commit data is done or the job ended (never another
    job's step), then main's head again; a PSD changed since: the PSD sizes again (the older kept when unread)."""
    yaml = pytest.importorskip("yaml")
    if not shutil.which("jq") or not shutil.which("git"):
        pytest.skip("no jq or git")
    site = yaml.safe_load((_workflow(clone).parent / "site.yml").read_text(encoding="utf-8"))
    wait = next(s for s in site["jobs"]["build"]["steps"] if s.get("name") == "Data wait")
    origin, work, temp, naps = tmp_path / "origin", tmp_path / "work", tmp_path / "temp", tmp_path / "sleeps.txt"
    git = lambda *a: subprocess.run(["git", "-C", str(origin), "-c", "user.name=A", "-c", "user.email=a@x", *a],
                                    check=True, capture_output=True, text=True)
    subprocess.run(["git", "init", "-q", "-b", "main", str(origin)], check=True)

    def push(name, data):
        (origin / name).write_bytes(data)
        git("add", "-A")
        git("commit", "-qm", name)

    push("A.psd", b"1")
    subprocess.run(["git", "clone", "-q", str(origin), str(work)], check=True)
    (temp / "setup").mkdir(parents=True)
    (temp / "setup" / "sizes.code").write_text("0\n")
    (temp / "tree.json").write_text("old")
    env = {"PATH": f"{_fake_tools(tmp_path)}:{os.environ['PATH']}", "FAKE_STATE": str(tmp_path / "gh.json"), "FAKE_SLEPT": str(naps),
           "GITHUB_REPOSITORY": "o/cards", "GITHUB_SERVER_URL": "https://x", "GITHUB_OUTPUT": str(tmp_path / "out"),
           "GITHUB_STEP_SUMMARY": str(tmp_path / "summary.md"), "RUNNER_TEMP": str(temp), "RUN": "7", "JOB": "Card data",
           "COMMIT": wait["env"]["COMMIT"], "FAKE_TREE": "new"}

    def jobs(commit, name="Card data", status="in_progress"):
        return json.dumps({"jobs": [{"name": name, "status": status, "steps": [{"name": "Commit data", "status": commit}]}]})

    def run(answer, later=None, looks=0, **more):
        (tmp_path / "gh.json").unlink(missing_ok=True)
        naps.write_text("")
        done, _ = _shell(wait, {**env, "FAKE_JOBS": answer, **({"FAKE_JOBS_LATER": later, "FAKE_LOOKS": str(looks)} if later else {}),
                                **more}, cwd=work)
        for _ in range(100):                                              # the PSD sizes' read, in the background
            if (temp / "setup" / "sizes.code").exists():
                break
            subprocess.run(["/bin/sleep", "0.05"])
        assert done.returncode == 0 and naps.read_text().split() == ["3"] * looks, (answer, done.stderr)
        calls = json.loads((tmp_path / "gh.json").read_text())["log"]
        return [c for c in calls if "/git/trees/" in c]

    assert not run(jobs("completed"))
    assert not run(jobs("pending"), jobs("completed"), 3)
    assert not run(jobs("completed", name="Card data after the bake"), jobs("completed"), 2)    # another job's step
    assert not run(jobs("pending", status="completed"))                                          # ended before it
    push("B.png", b"2")                                                   # main moved, no PSD: the sizes stand
    assert not run(jobs("completed")) and (temp / "tree.json").read_text() == "old"
    (temp / "setup" / "sizes.code").write_text("1\n")
    push("A.psd", b"3")
    head = git("rev-parse", "HEAD").stdout.strip()
    assert run(jobs("completed")) == [f"api repos/o/cards/git/trees/{head}?recursive=1"]
    assert subprocess.run(["git", "-C", str(work), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip() == head
    assert (temp / "tree.json").read_text().strip() == "new" and (temp / "setup" / "sizes.code").read_text().strip() == "0"
    (temp / "tree.json").write_text("older")
    push("A.psd", b"4")
    assert run(jobs("completed"), FAKE_FAIL="/git/trees/") and (temp / "tree.json").read_text() == "older"
    assert (temp / "setup" / "sizes.code").read_text().strip() == "0" and "not read" in (temp / "setup" / "sizes.log").read_text()


def test_card_syncs_lane_sets_up_before_its_guards(clone, tmp_path):
    """Card sync's main lane sets up (checkouts, Python), waits for this push's guards, takes main's and the data's heads
    again, then gates (no Gate when the guards hold it). A push rides along on a lane queued or in that wait; the
    schedule's gate and the failure streak pass over a held lane."""
    yaml = pytest.importorskip("yaml")
    if not shutil.which("jq") or not shutil.which("git"):
        pytest.skip("no jq or git")
    doc = yaml.safe_load((_workflow(clone).parent / "card-sync.yml").read_text(encoding="utf-8"))
    main = doc["jobs"]["main"]
    assert "quiet" not in doc["jobs"] and main["needs"] == "timer"
    steps = {s.get("name"): s for s in main["steps"]}
    assert [s.get("name") for s in main["steps"]][3:10] == ["Python environment", "Fonts first", None, "Fonts first (cache miss)",
                                                           "Guards' fixes", "Newest", "Gate"]
    assert steps["Gate"]["if"] == "${{ !steps.guards.outputs.run }}" and "steps.gate.outcome == 'success'" in steps["Leftover edits"]["if"]
    bin_dir = _fake_tools(tmp_path)
    state, posts = tmp_path / "gh.json", tmp_path / "posts.jsonl"
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    env = {"PATH": f"{bin_dir}:{os.environ['PATH']}", "FAKE_STATE": str(state), "FAKE_POSTS": str(posts), "GITHUB_REPOSITORY": "o/cards",
           "GITHUB_RUN_ID": "9", "GITHUB_JOB": "main", "GITHUB_OUTPUT": str(tmp_path / "out"), "GITHUB_ENV": str(tmp_path / "env"),
           "GITHUB_STEP_SUMMARY": str(tmp_path / "summary.md"), "GITHUB_SERVER_URL": "https://x", "RUN_URL": "https://x/run"}

    def lane(status="completed", conclusion="success", *steps_):
        return {"name": "Bake and save", "status": status, "conclusion": conclusion,
                "steps": [{"name": n, "status": "completed" if c else "in_progress", "conclusion": c} for n, c in steps_]}
    other = next(s for s in doc["jobs"]["timer"]["steps"] if s["name"] == "Another timer")
    for job, rides in ((lane("queued", None), True), (lane("in_progress", None, ("Guards' fixes", None)), True),
                       (lane("in_progress", None, ("Guards' fixes", "success"), ("Gate", None)), False),
                       (lane(), False), ({"name": "Quiet window", "status": "in_progress", "steps": []}, False)):
        state.unlink(missing_ok=True)
        done, out = _shell(other, {**env, **other["env"], "FAKE_RUNS": json.dumps([{"databaseId": 5, "status": "in_progress", "createdAt": now}]),
                                   "FAKE_JOBS": json.dumps({"jobs": [job]})})
        assert done.returncode == 0 and (out.get("run") == "5") is rides, (job, done.stderr)
    held, went = lane("completed", "success", ("Gate", "skipped")), lane("completed", "success", ("Gate", "success"))
    repo = tmp_path / "cards"                   # the schedule's gate: the held push's head is not a synced one
    git = ["git", "-C", str(repo), "-c", "user.name=Alice", "-c", "user.email=a@x"]
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    for name in ("A.psd", "B.psd"):
        (repo / name).write_bytes(name.encode())
        subprocess.run([*git, "add", name], check=True)
        subprocess.run([*git, "commit", "-qm", name], check=True)
    heads = subprocess.run([*git, "rev-list", "HEAD"], capture_output=True, text=True, check=True).stdout.split()
    state.unlink(missing_ok=True)
    gate = steps["Gate"]
    done, out = _shell(gate, {**env, **{k: v for k, v in gate["env"].items() if "${{" not in str(v)}, "GITHUB_EVENT_NAME": "schedule",
                              "DRY_RUN": "false", "FAKE_LOOKS": "1", "FAKE_JOBS": json.dumps({"jobs": [held]}), "FAKE_JOBS_LATER": json.dumps({"jobs": [went]}),
                              "FAKE_RUNS": json.dumps([{"databaseId": 2, "headSha": heads[0], "displayTitle": "Card sync"},
                                                       {"databaseId": 1, "headSha": heads[1], "displayTitle": "Card sync"}])}, cwd=repo)
    assert done.returncode == 0 and out["go"] == "true", done.stdout + done.stderr
    assert f"card inputs changed in 1 commits since {heads[1][:8]}" in done.stdout
    tell = steps["Tell Discord"]                # the streak: the held lane between is passed over, so no second post
    state.unlink(missing_ok=True)
    done, _ = _shell(tell, {**env, "WEBHOOK": "https://hook", "FAKE_LOOKS": "2", "FAKE_JOBS": json.dumps({"jobs": [held]}),
                            "FAKE_JOBS_LATER": json.dumps({"jobs": [lane("completed", "failure", ("Gate", "success"))]}),
                            "FAKE_RUNS": json.dumps([{"databaseId": 8, "displayTitle": "Card sync"}, {"databaseId": 7, "displayTitle": "Card sync"}])})
    assert done.returncode == 0 and "failed in the last live run where it ran too" in done.stdout and not posts.exists(), done.stdout

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


def test_pixel_fixes_are_said_only_in_the_commit(tmp_path, monkeypatch):
    """Straight apostrophes (owner 2026-10-25): the PNG follows through guard 1's export, both writes silent; the commit
    message alone counts them."""
    from types import SimpleNamespace
    fixed = SimpleNamespace(data=b"straight", plain=b"pushed", rewrapped=[], redrawn={"apostrophes": ["Name"]})
    found = pe.Export("re-exported", image=object(), data=b"png2", changed_px=35)
    monkeypatch.setattr(cg.png_export, "open_psd", lambda data, rel="": data)
    monkeypatch.setattr(cg.png_export, "export", lambda old, new, png, numbered, **kw: found)
    result = cg.Result(cg.Job("X/A.psd", "pair", png="X/A.png"))
    assert cg._rewrapped_pngs(str(tmp_path), result, b"pushed", fixed, {"X/A.png": "image"})
    assert [(w["kind"], w["silent"], w["line"]) for w in result.writes] == [
        ("png", True, "re-exported for straight apostrophes (35 px changed)")]
    found.outcome, found.why = "refused", "PNG not made: G6"
    assert not cg._rewrapped_pngs(str(tmp_path), result, b"pushed", fixed, {"X/A.png": "image"})
    assert result.pixel_left == [("apostrophes", "not redrawn: PNG not made: G6")]
    entry = _entry("A")
    _write(tmp_path / "plan", entry, "X/A.psd", "psd", b"straight", "straight apostrophes (Name)")
    entry["writes"][0].update(shown="", silent=True, composer=False, pixels=["apostrophes"])
    data = json.loads(Path(_full_plan(tmp_path, [entry])).read_text())
    assert cg.discord_payload(data, "o/r") == {} and "apostrophe" not in cg.markdown(data, [], [], [])
    assert cg.message([entry], "f" * 40).startswith(f"Card guards: straight apostrophes on 1 card (push {'f' * 8})")


def test_an_old_font_save_is_told():
    """Owner 2026-11-04: badges drawn with another build of the icon font that the PSD keeps (old glyphs refused or
    fallen back, or a redraw whose PNG couldn't follow) are told, on Discord and in the commit comment; one that lands
    never. A landing pixel fix's notes go to plan.json alone."""
    from types import SimpleNamespace
    fixed = lambda stale=(), redrawn=None: SimpleNamespace(stale_glyphs=list(stale), redrawn=redrawn or {})
    relaid = {"badge widths": ["Ability"]}
    assert cg.stale_left(fixed(), {}) == []
    assert cg.stale_left(fixed(["Ability"]), {}) == ["Ability"]                     # refused, or its PNG didn't follow
    assert cg.stale_left(fixed(["Ability"], {"old glyphs": ["Ability"]}), {"old glyphs": ["Ability"]}) == []
    assert cg.stale_left(fixed((), relaid), {}) == ["Ability"]                       # re-laid, its PNG didn't follow
    assert cg.stale_left(fixed(["Name"], relaid), relaid) == ["Name"]
    result = cg.Result(cg.Job("X/A.psd", "pair", png="X/A.png"), notes=[(cg.OLD_FONT, True)],
                       pixel_notes=["Ability: 3 lines became 4"])
    assert result.told() == [cg.OLD_FONT]
    data = cg.plan([result], "o/r", "a" * 40, "b" * 40, "on")
    assert data["cards"][0]["pixel_notes"] == ["Ability: 3 lines became 4"]
    text = cg.markdown(data, [], [], [])
    assert f"- {cg.OLD_FONT}" in text and "became" not in text


def test_a_rewrap_redraw_follows_into_the_png_unsaid(tmp_path, monkeypatch):
    """R8 always switches; box text that rewraps is redrawn and its PNG re-exported through guard 1's export, unsaid (a
    guard 1 write takes the new pixels, the pushed PNG's recompression gives way); a refused export, or no PNG that shows
    the PSD, lands the PSD without the redraw."""
    from types import SimpleNamespace
    fixed = SimpleNamespace(data=b"drawn", plain=b"plain", rewrapped=["Ability"])
    found = pe.Export("re-exported", image=object(), data=b"png2", changed_px=12)
    monkeypatch.setattr(cg.png_export, "open_psd", lambda data, rel="": data)
    monkeypatch.setattr(cg.png_export, "export", lambda old, new, png, numbered, **kw: found)
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
    result.job.numbered = {"X/A 2.png": "n1"}                   # a numbered PNG that would keep the old wrap
    assert not cg._rewrapped_pngs(str(tmp_path), result, b"pushed", fixed, {"X/A.png": "image"})
    assert [why for _, why in result.composer_left] == ["rewraps, not redrawn: PNG not made: G6",
                                                        "rewraps, not redrawn: no PNG that shows the PSD",
                                                        "rewraps, not redrawn: A 2.png does not show the PSD"]


def test_a_redrawn_psd_lands_plain_without_its_png(tmp_path, monkeypatch):
    """The redraw's PSD and PNG land together: when the PNG cannot (the author's own export since), the PSD lands
    switched, not redrawn, never beside an old-wrap PNG."""
    out = tmp_path / "plan"
    entry = _entry("A")
    _write(out, entry, "X/A.psd", "psd", b"drawn", "Single-line Composer (Ability); straight apostrophes (Name)")
    (out / "files" / "A.psd.plain").write_bytes(b"plain")
    entry["writes"][0].update(shown="", silent=True, composer=True, pixels=["apostrophes"],
                              plain={"file": "files/A.psd.plain", "sha": blob_sha(b"plain"),
                                     "line": "Single-line Composer (Ability)"})
    _write(out, entry, "X/A.png", "png", b"png2", "re-exported for the Single-line Composer's rewrap")
    entry["writes"][1].update(shown="", silent=True, composer=True)
    plan = _full_plan(tmp_path, [entry])
    sent = {}
    for png, files in (("p1", {"X/A.psd": b"drawn", "X/A.png": b"png2"}), ("author", {"X/A.psd": b"plain"})):
        FakeHub({"X/A.psd": ("100644", "s1"), "X/A.png": ("100644", png)}).install(monkeypatch)
        monkeypatch.setattr(cg, "commit_files", lambda *a, **k: sent.clear() or sent.update(a[4], text=a[5]) or "c" * 40)
        landed = cg.land(plan)
        assert "apostrophe" in sent.pop("text") if png == "p1" else "apostrophe" not in sent.pop("text")   # plain: no pixel fix said
        assert sent == files, png
    assert landed["included"] == [["A", "X/A.psd"]] and sent == {"X/A.psd": b"plain"}
    assert landed["dropped"] == [["A", "superseded: X/A.png changed on the branch", "X/A.png"]]
    assert landed["told"]["X/A.psd"] == []


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
    cg._init_worker(str(tmp_path / "clone"), "o/r", None, vocab, tree, str(tmp_path / "fonts"), ("old.earlier",))
    assert cg._worker["source"].checked == tree and cg._worker["source"].fonts == str(tmp_path / "fonts")
    assert cg._worker["earlier"] == ("old.earlier",)


def test_pool_size(monkeypatch):
    """K-2: every core on a runner; locally half the cores, at most POOL_MAX; never more than the cards."""
    monkeypatch.setattr(cg, "cpu_count", lambda: 12)
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    assert (cg.pool_size(12), cg.pool_size(3)) == (4, 3)
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setattr(cg, "cpu_count", lambda: 4)
    assert (cg.pool_size(12), cg.pool_size(3)) == (4, 3)
    monkeypatch.setattr(cg, "cpu_count", lambda: 2)
    assert cg.pool_size(12) == 2


def test_status_words_make_no_changelog_event():
    """Owner 11-07: the status words fix lands in the guards' commit silently, no guard-event for it or its PNG."""
    psd = {"kind": "psd", "path": "a.psd", "pixels": ["status words"], "shown": ""}
    png = {"kind": "png", "path": "a.png", "silent": True, "line": "re-exported"}
    assert cg.events_of({"card": "A"}, psd, [psd, png]) == cg.events_of({"card": "A"}, png, [psd, png]) == []
