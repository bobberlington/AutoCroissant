"""
card_guards (the push guards' driver): a card repo made here with git (fixture PSDs, every case of PUSH_GUARDS.md 4.2,
bot commits left out), check end to end on it (the report, the one Discord post without mentions or previews, the plan),
land against a fake GitHub (superseded cards, a moved head) and in a local clone, the replay of the real
push 38c55bd3..07589cc7 from the card clone (read-only: God Tamer, Bubble and 12 Reactions2 "PNG still current"), the
Action's sparse copy (SPARSE_FILES, card-guards.yml's checkouts of it), the steps that run the bot's own code against a
fake gh and curl (one post per push, the pushed blobs and earlier fonts the checks read, the fonts wait), and actionlint.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import types
from io import BytesIO
from pathlib import Path
from sys import executable
from types import SimpleNamespace

import pytest
import yaml
from PIL import Image, ImageDraw

import commands.card_guards as cg
import commands.font_check as fc
import commands.git_commit as gc
import commands.png_export as pe
from commands.github_api import blob_sha
from conftest import SITE_SRC
from helpers import art, bot_files, build_font, card, fixture_psd, photoshop_png, sharing


# ========================
# Configuration
# ========================
REPO = Path(__file__).resolve().parents[1]
FOLDER = "Items/Heal/2 Stars"


# ========================
# A card repo
# ========================
def _git(root, *args, author="Base"):
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
    assert cg.Writer("x", "AutoCroissant", "a@b").bot
    assert cg.Writer("x", "Someone", gc.AUTHOR["email"]).bot
    assert cg.Writer("x", "github-actions[bot]", cg.BOT_EMAILS[0]).bot
    assert not cg.Writer("x", "Alice", "a@b").bot          # a person's commit named like a bot's is guarded


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
    data = cg.check(str(root), bot, base, "o/cards", "actor", mode="dry", out=str(out), workers=1)
    by = {Path(c["psd"]).stem: c for c in data["cards"]}
    snack = by["Snack"]
    assert snack["export"]["outcome"] == "re-exported" and snack["export"]["writer"] == "tool"
    assert [w["path"] for w in snack["writes"]] == [f"{FOLDER}/Snack.png"]     # Snack2.png has no tag: refused
    for write in snack["writes"]:
        assert blob_sha((out / write["file"]).read_bytes()) == write["sha"]
    assert snack["expect"] == {f"{FOLDER}/Snack.psd": cg.Source(str(root)).tree(bot)[f"{FOLDER}/Snack.psd"][1],
                               f"{FOLDER}/Snack.png": cg.Source(str(root)).tree(bot)[f"{FOLDER}/Snack.png"][1],
                               f"{FOLDER}/Snack2.png": cg.Source(str(root)).tree(bot)[f"{FOLDER}/Snack2.png"][1]}
    assert by["New_Card"]["export"]["outcome"] == "re-exported" and by["New_Card"]["writes"][0]["path"] == f"{FOLDER}/New_Card.png"
    assert by["New_Name"]["writes"][0]["sha"] == cg.Source(str(root)).tree(base)["Creatures/Other/3 Stars/Old_Name.png"][1]
    assert by["Bread"]["r10"]["ok"] is True and not by["Bread"]["writes"]
    assert by["Lonely"]["r10"]["ok"] is False and by["Lonely"]["told"] == []          # R10: the comment only
    assert by["Gone"]["told"] == [] and by["Gone"]["notes"] == ["Gone.png is left without its PSD"]     # comment only
    # the PNGs the bot makes are never told (a dry run shows what on would do)
    assert by["New_Name"]["told"] == [] and by["New_Name"]["notes"] == ["Old_Name.png is left at the PSD's old name"]
    assert snack["told"][0] == "Snack2.png not updated: tag its picture in the PSD"
    assert not any("export it in Photoshop" in line for line in snack["told"] + by["New_Card"]["told"])
    assert by["Kept"]["told"] == [] and by["Kept"]["notes"] == ["Kept.png was deleted; the PSD stays"]
    assert (out / "crops" / FOLDER / "Snack.diff.png").exists()
    summary = json.loads((out / "summary.json").read_text())
    assert summary == {"after": bot, "cards": 8, "told": summary["told"], "comment": 6, "writes": 7, "normalise": 4,
                       "errors": 0, "card_errors": 0} and summary["told"] >= 1     # 4 merged images stripped later (Q2)
    assert by["Bread"]["told"] == [] and not by["Bread"]["issues"]      # R3: no Ability layer is the developer's layout
    assert not by["Dropped"]["writes"] and not by["Dropped"]["export"]
    report = (out / "report.md").read_text()
    assert report.startswith("## Card guards") and "Dry run: nothing is posted or committed." in report and "Bot_Card" in report
    payload = json.loads((out / "discord.json").read_text())
    assert payload["allowed_mentions"] == {"parse": []} and payload["flags"] == 4 and "<@" not in payload["content"]
    lines = payload["content"].splitlines()
    assert all(re.fullmatch(r"\S+ pushed .+: .+\. \[[0-9a-f]{8}\]\(https://github\.com/o/cards/commit/[0-9a-f]{12}\)", l)
               for l in lines), lines
    assert "Lonely.png does not match its PSD" in report and not any("Lonely" in l for l in lines)
    assert not any("Snack" in l and "re-exported" in l for l in lines)        # writes are not issues
    assert "- PNG would be re-exported by the bot's render of the PSD (441 px changed" in report and "with CARD_GUARDS=on" in report
    assert "- PNG re-exported" not in report and "- PNG copied" not in report
    assert "Alice pushed Snack: Snack2.png not updated: tag its picture in the PSD" in payload["content"]
    # what changed inside each PSD pushed over an older one (psd_diff): a new PSD or a PNG alone has none
    assert snack["changes"] == by["Bread"]["changes"] == by["Dropped"]["changes"] == ["Square pixels changed"]
    assert by["New_Card"]["changes"] == by["Lonely"]["changes"] == by["New_Name"]["changes"] == []
    assert "**Snack** (`Items/Heal/2 Stars/Snack.psd`, Alice, `" in report and "- Changed: Square pixels changed" in report
    assert "New Card" not in payload["content"]                                 # the PNG it makes is never told


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


def test_a_png_back_to_an_earlier_picture_is_put_back(cards, tmp_path, monkeypatch):
    """R10 with no stored image (11-10, Despair's black stroke): Bread's old blue PNG pushed alone over its green PSD and
    PNG passed the parity floor. The PNG before goes back, silent (a guard-png event, nothing told); a PNG of no earlier
    picture stays as pushed, the comment alone saying it; pushed with a PSD change, guard 1 makes it from the PNG before."""
    root, base, alice, bot = cards
    monkeypatch.setattr(cg.png_export, "composite", lambda psd: None)        # saved without Maximize Compatibility
    green = _git(root, "rev-parse", f"{bot}:{FOLDER}/Bread.png")
    shutil.copy(root / FOLDER / "Snack.png", root / FOLDER / "Bread.png")
    back = _commit(root, "old PNG back", author="Alice")
    out = tmp_path / "out"
    data = cg.check(str(root), back, bot, mode="on", out=str(out), workers=1)
    (entry,) = data["cards"]
    assert entry["case"] == "png-only" and entry["r10"]["back"]["px"] == 21 * 21 and entry["told"] == []
    assert [(w["kind"], w["sha"]) for w in entry["writes"]] == [("png", green)] and data["discord"] == {}
    report = cg.land(str(out / "plan.json"), local=str(root))
    assert _git(root, "rev-parse", f"{report['sha']}:{FOLDER}/Bread.png") == green
    assert [(e["kind"], e["path"]) for e in report["events"]] == [("guard-png", f"{FOLDER}/Bread.png")]
    (root / FOLDER / "Bread.png").write_bytes(photoshop_png(card(colour=(255, 0, 0))))
    red = _commit(root, "a new picture", author="Alice")
    (entry,) = cg.check(str(root), red, report["sha"], mode="on", out=str(tmp_path / "red"), workers=1)["cards"]
    assert entry["r10"]["back"]["older"] == "" and "kept as pushed" in entry["r10"]["line"]
    assert not [w for w in entry["writes"] if w["kind"] == "png"] and entry["told"] == []
    _git(root, "checkout", "-q", report["sha"])                            # with its PSD: the PSD's change made again
    (tmp_path / "w").mkdir()
    _psd(root, tmp_path / "w", f"{FOLDER}/Bread.psd", card(base=(30, 200, 30), colour=(0, 255, 0)))
    shutil.copy(root / FOLDER / "Snack.png", root / FOLDER / "Bread.png")
    pair = _commit(root, "base recoloured, old PNG", author="Alice")
    (entry,) = cg.check(str(root), pair, report["sha"], mode="on", out=str(tmp_path / "pair"), workers=1)["cards"]
    (write,) = [w for w in entry["writes"] if w["kind"] == "png"]
    assert entry["case"] == "pair" and write["back"] and entry["told"] == []
    assert int(pe._diff(Image.open(tmp_path / "pair" / write["file"]), card(base=(30, 200, 30), colour=(0, 255, 0))).max()) == 0


def test_a_png_that_skips_its_psds_change_is_made_again(cards, tmp_path, monkeypatch):
    """R10, a PSD pushed with a PNG that skips its change (11-10: under the parity floor, as Pizza's colour fix):
    Bread's square recoloured, its PNG pushed as the PNG before in another blob. Guard 1's export onto the PNG before
    lands, silent; the PNG that shows the change passes; one changed elsewhere too stays as pushed (a stored image: no
    went-back), the comment saying it. The export's file is encoded only when made again. Photoshop's PNG passes where
    the fork draws a corner otherwise; the PNG before stays as pushed then (the export misses the stored image)."""
    root, base, alice, bot = cards
    monkeypatch.setattr(cg.png_export, "composite", lambda psd: None)        # saved without Maximize Compatibility
    encoded = []
    monkeypatch.setattr(pe, "encode", lambda *a, _f=pe.encode: encoded.append(1) or _f(*a))
    teal = card(colour=(0, 128, 128))
    _psd(root, tmp_path / "psds", f"{FOLDER}/Bread.psd", teal)
    green = Image.open(root / FOLDER / "Bread.png")
    green.save(root / FOLDER / "Bread.png", compress_level=1)              # the same pixels, another blob
    skipped = _commit(root, "square recoloured, PNG before", author="Alice")
    out = tmp_path / "out"
    data = cg.check(str(root), skipped, bot, mode="on", out=str(out), workers=1)
    (entry,) = data["cards"]
    (write,) = [w for w in entry["writes"] if w["kind"] == "png"]
    assert entry["case"] == "pair" and write["skipped"] and entry["r10"]["skipped"]["px"] == 21 * 21
    assert entry["told"] == [] and data["discord"] == {} and "made again" in entry["r10"]["line"]
    assert int(pe._diff(Image.open(out / write["file"]), teal).max()) == 0 and len(encoded) == 1
    elsewhere = card(colour=(0, 255, 0))
    ImageDraw.Draw(elsewhere).rectangle((40, 60, 55, 85), fill=(0, 0, 0))
    for png, kept in ((teal, False), (elsewhere, True)):
        if kept:
            monkeypatch.undo()
        _png(root, f"{FOLDER}/Bread.png", png)
        after = _commit(root, f"PNG {kept}", author="Alice")
        (entry,) = cg.check(str(root), after, bot, mode="on", out=str(tmp_path / f"o{kept}"), workers=1)["cards"]
        assert not [w for w in entry["writes"] if w["kind"] == "png"] and entry["told"] == []
        assert ("skipped" in entry["r10"]) is kept and ("kept as pushed" in entry["r10"]["line"]) is kept
        assert kept or len(encoded) == 1                                     # shows the change: no file encoded
    shop, real = teal.copy(), pe.composite      # Photoshop drew an 8x8 corner of the square apart from the fork
    shop.paste((0, 255, 0), (23, 23, 31, 31))

    def composite(psd):
        image = real(psd)
        new = image is not None and image.convert("RGB").getpixel((15, 15)) == (0, 128, 128)
        return shop.convert(image.mode) if new else image
    monkeypatch.setattr(cg.png_export, "composite", composite)
    (root / FOLDER / "Bread.png").write_bytes(photoshop_png(shop))
    after = _commit(root, "Photoshop's PNG", author="Alice")
    (entry,) = cg.check(str(root), after, bot, mode="on", out=str(tmp_path / "shop"), workers=1)["cards"]
    assert entry["r10"]["ok"] and "skipped" not in entry["r10"]
    assert not [w for w in entry["writes"] if w["kind"] == "png"] and entry["told"] == []
    green.save(root / FOLDER / "Bread.png", compress_level=1)   # skipped, but the export misses Photoshop's: kept
    after = _commit(root, "PNG before again", author="Alice")
    (entry,) = cg.check(str(root), after, bot, mode="on", out=str(tmp_path / "miss"), workers=1)["cards"]
    assert "skipped" in entry["r10"] and "misses the bar itself: kept as pushed" in entry["r10"]["line"]
    assert not [w for w in entry["writes"] if w["kind"] == "png"] and entry["told"] == []


def test_a_png_whose_own_change_contradicts_its_psd_is_kept(cards, tmp_path, monkeypatch):
    """R10's skip (11-11, Tomato: the PSD showed an old layer over the new art its PNG showed): Bread's square recoloured,
    its PNG pushed with half the square painted black instead. Past PARITY_FLOOR of the export inside the PNG's own
    change: kept as pushed, the comment saying it, nothing told."""
    root, base, alice, bot = cards
    monkeypatch.setattr(cg.png_export, "composite", lambda psd: None)
    _psd(root, tmp_path / "psds", f"{FOLDER}/Bread.psd", card(colour=(0, 128, 128)))
    painted = card(colour=(0, 255, 0))
    painted.paste((0, 0, 0), (10, 10, 20, 31))
    _png(root, f"{FOLDER}/Bread.png", painted)
    after = _commit(root, "square recoloured, PNG painted", author="Alice")
    data = cg.check(str(root), after, bot, mode="on", out=str(tmp_path / "out"), workers=1)
    (entry,) = data["cards"]
    assert entry["case"] == "pair" and "skipped" in entry["r10"]
    assert entry["r10"]["line"] == "Bread.png's own change does not match the PSD (210 px): kept as pushed"
    assert not [w for w in entry["writes"] if w["kind"] == "png"] and entry["told"] == [] and data["discord"] == {}


def test_a_skip_needs_pixels_past_the_bar(cards, tmp_path, monkeypatch):
    """R10's skip (11-11, Tainted Jacob): Bread's square recoloured, its PNG pushed as the PNG before with 8 levels of
    shading elsewhere (under PARITY_THRESH: no change elsewhere): made again as the export. A 15-level tint its PNG
    skips (a mean, no pixel past the bar: the fork's drawing) matches."""
    root, base, alice, bot = cards
    monkeypatch.setattr(cg.png_export, "composite", lambda psd: None)
    teal = card(colour=(0, 128, 128))
    _psd(root, tmp_path / "psds", f"{FOLDER}/Bread.psd", teal)
    shaded = card(colour=(0, 255, 0))
    ImageDraw.Draw(shaded).rectangle((40, 60, 55, 85), fill=(208, 30, 30))
    _png(root, f"{FOLDER}/Bread.png", shaded)
    after = _commit(root, "square recoloured, PNG shaded", author="Alice")
    (entry,) = cg.check(str(root), after, bot, mode="on", out=str(tmp_path / "out"), workers=1)["cards"]
    (write,) = [w for w in entry["writes"] if w["kind"] == "png"]
    assert write["skipped"] and "made again" in entry["r10"]["line"] and entry["told"] == []
    assert int(pe._diff(Image.open(tmp_path / "out" / write["file"]), teal).max()) == 0
    _psd(root, tmp_path / "psds", f"{FOLDER}/Bread.psd", card(colour=(0, 240, 0)))
    card(colour=(0, 255, 0)).save(root / FOLDER / "Bread.png", compress_level=1)
    after = _commit(root, "square tinted, PNG before", author="Alice")
    (entry,) = cg.check(str(root), after, bot, mode="on", out=str(tmp_path / "tint"), workers=1)["cards"]
    assert entry["r10"]["ok"] and "skipped" not in entry["r10"] and entry["told"] == []
    assert not [w for w in entry["writes"] if w["kind"] == "png"]


def test_a_png_over_a_broken_one_is_checked(cards, tmp_path, monkeypatch):
    """R10's reads of the PNG before are best effort: a good PNG pushed over a broken one (a git-lfs pointer), alone with
    no stored image or with its PSD, passes, nothing told."""
    root, base, alice, bot = cards
    (root / FOLDER / "Bread.png").write_text("version https://git-lfs.github.com/spec/v1\n")
    broken = _commit(root, "broken PNG", author="Alice")
    (root / FOLDER / "Bread.png").write_bytes(photoshop_png(card(colour=(0, 255, 0))))
    fix = _commit(root, "PNG fixed", author="Alice")
    _psd(root, tmp_path / "psds", f"{FOLDER}/Bread.psd", card(colour=(0, 128, 128)))
    (root / FOLDER / "Bread.png").write_bytes(photoshop_png(card(colour=(0, 128, 128))))
    pair = _commit(root, "square recoloured", author="Alice")
    for after, case, stored in ((fix, "png-only", False), (pair, "pair", True)):
        if not stored:
            monkeypatch.setattr(cg.png_export, "composite", lambda psd: None)
        data = cg.check(str(root), after, broken, mode="on", out=str(tmp_path / case), workers=1)
        monkeypatch.undo()
        (entry,) = data["cards"]
        assert entry["case"] == case and not entry["error"] and entry["r10"]["ok"]
        assert entry["told"] == [] and data["discord"] == {}


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
    real = pe._drawn

    def drawn(result, key):                         # the anchored region (push 1's Base: the whole card) refuses
        x0, y0, x1, y1 = result.numbers.get("region") or (0, 0, 0, 0)
        return not (key == "baseline" and (x1 - x0) * (y1 - y0) > 21 * 21) and real(result, key)
    monkeypatch.setattr(pe, "_drawn", drawn)
    work = tmp_path / "burst"
    work.mkdir()
    one = _dated(root, work, "Bread", card(base=(30, 200, 30)), "2026-10-22T10:00:00Z")
    two = _dated(root, work, "Bread", card(base=(30, 200, 30), colour=(0, 255, 0)), "2026-10-22T10:10:00Z")
    data = cg.check(str(root), two, one, mode="on", out=str(tmp_path / "out"), workers=1)
    (entry,) = data["cards"]
    assert not entry["error"], entry["error"]
    assert (entry["export"]["outcome"], entry["export"]["gate"]) == ("refused", "G6"), entry["export"]
    assert not [w for w in entry["writes"] if w["kind"] == "png"] and entry["export"]["why"] in entry["told"]


def test_a_png_pushed_before_its_psd_keeps_g2_off_the_anchor(cards, tmp_path, monkeypatch):
    """Push A the PNG of v2 alone, B the PSD v2, C the PSD v3 (no stored image), within the burst window: the anchor is
    v1 (the PSD at A) but the PNG shows v2, so G2 draws v3 over v2 as the checked path does. Over the anchor it refused
    G6 (the square recoloured), nothing told (review of 2026-10-24)."""
    root, base, alice, bot = cards
    work = tmp_path / "burst"
    work.mkdir()
    v2, v3 = card(base=(30, 200, 30)), card(base=(30, 200, 30), colour=(0, 255, 0))
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
    assert entry["export"]["outcome"] == "re-exported" and entry["export"]["changed_px"] == 441
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


def test_only_new_issues_are_told():
    """Owner 2026-11-11: of a card pushed over an older PSD only the issues new on it reach Discord (no "(new)": all are),
    whoever pushed; a new card's all; the old ones stay told in the comment."""
    old = {"code": "R11", "text": "parts are not named so", "line": "parts are not named so", "new": False, "told": True,
           "discord": True, "waived": ""}
    new = {"code": "R14", "text": "card names typed in another letter case: 'x' should be 'X'", "new": True, "told": True,
           "discord": True, "waived": ""}
    new["line"] = new["text"] + " (new)"
    result = cg.Result(cg.Job("X/A.psd", "stale", psd_old="s0"), issues=[old, new])
    assert result.told() == [new["text"]]
    result.job.psd_old = ""                                                    # a new card
    assert result.told() == [old["text"], new["text"]]
    entry = _entry("A", issues=[old, new], told=[new["text"]])
    assert "Issues: parts are not named so; card names typed in another letter case: 'x' should be 'X' (new)" in cg.markdown(
        {"mode": "on", "after": "f" * 40, "before": "e" * 40, "cards": [entry]}, [], [], [])


def test_check_fails_when_it_cannot_check(cards, tmp_path, capsys):
    """Not checked (the clone lacks the commit, the API is down): exit 1, so card-guards.yml's failure alert runs."""
    root, base, alice, bot = cards
    assert cg.main(["check", "--clone", str(root), "--after", "e" * 40, "--out", str(tmp_path / "out")]) == 1
    assert "Not checked: the clone has no commit eeeeeeee" in capsys.readouterr().out


def test_discord_fits_and_says_nothing_without_issues():
    entry = {"login": "Zomoroko", "card": "Skullord", "told": ["timing tag (1GCD) not set in the stoogedings badge font"], "commit": "7" * 40}
    data = {"after": "a" * 40, "cards": [dict(entry, card=f"Card {i}") for i in range(80)]}
    payload = cg.discord_payload(data, "MichaelJSr/TTSCardMaker")
    assert len(payload["content"]) <= 2000 and payload["flags"] == 4
    last = payload["content"].splitlines()[-1]
    assert re.fullmatch(r"\.\.\. and \d+ more cards: \[details\]\(https://github\.com/MichaelJSr/TTSCardMaker/commit/a{40}\)", last)
    one = cg.discord_payload({"after": "a" * 40, "cards": [entry]}, "MichaelJSr/TTSCardMaker")["content"]
    assert one == ("Zomoroko pushed Skullord: timing tag (1GCD) not set in the stoogedings badge font. "
                   "[77777777](https://github.com/MichaelJSr/TTSCardMaker/commit/" + "7" * 12 + ")")
    assert cg.discord_payload({"after": "a" * 40, "cards": [dict(entry, told=[])]}, "o/r") == {}


def test_message():
    landed = [{"login": "Zomoroko", "commit": "7cb4c867" + "0" * 32,
               "writes": [{"path": "Items/X.png", "kind": "png", "line": "re-exported by the bot's render of the PSD (12 px changed)"}]}]
    text = cg.message(landed, "f" * 40)
    assert text.splitlines()[0] == f"Card guards: 1 PNG re-exported (push {'f' * 8})"
    assert "Items/X.png: re-exported by the bot's render of the PSD (12 px changed) (Zomoroko, 7cb4c867)" in text
    assert text.rstrip().endswith(f"Guarded-push: {'f' * 40}") and "Edit-by" not in text
    landed[0]["writes"].append({"path": "Items/Z.png", "kind": "recompress", "line": "recompressed losslessly (9.0 MB -> 2.0 MB)"})
    text = cg.message(landed, "f" * 40)
    assert text.splitlines()[0] == f"Card guards: 1 PNG re-exported, 1 PNG recompressed (push {'f' * 8})"
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
    fake.fallback_names = lambda catalogue, linked: None
    fake.links_of = lambda psd, rel, tree: [SimpleNamespace(path=p, name=p.rsplit("/", 1)[-1], uuid="u1", size=3)
                                            for p in (links or {}).get(rel, [])]
    fake.records_of = lambda index: ({k: v for k, v in index.items() if k != "files"}, index.get("files") or {})
    fake.unused = lambda index, tree, pushed=(), read=(): list(unused)
    fake.duplicates = lambda tree, pushed=(): dict(duplicates or {})
    fake.restore = lambda size, history: None
    fake.relink = lambda psd, uuid, rel, path, size, blob: relinked.append((rel, path))
    fake.needs = lambda psd: []
    fake.link = lambda psd, rel, fonts: cg.standards.Normalised()
    fake.ppis_of = lambda read, seen=None: (lambda path: None)
    fake.ink_key = lambda rgba: None
    monkeypatch.setitem(sys.modules, "commands.psd_links", fake)
    monkeypatch.setattr(sys.modules["commands"], "psd_links", fake, raising=False)   # `from commands import psd_links`
    return relinked


def test_hidden_copies_shown_on_other_psds(monkeypatch):
    """clean's leftovers (owner 11-09): db/links.json's PSDs in scope, at their blob, hold each linked file; a pushed
    card's hidden picture another of them shows is a copy (read at after); a card with picture-only duplicates keeps
    every leftover."""
    links = [{"path": "MDW/Linked/X.png"}]
    index = {"files": {}, "pixels": {}, "Items/A.psd": {"blob": "a1", "links": links},
             "Items/B.psd": {"blob": "old", "links": links}, "MDW/Old Rules/C.psd": {"blob": "c1", "links": links},
             "MDW/Templates/T.psd": {"blob": "t1", "links": [{"path": "MDW/Linked/Y.png"}]}}
    tree = {"Items/A.psd": ("100644", "a1"), "Items/B.psd": ("100644", "b2"), "MDW/Old Rules/C.psd": ("100644", "c1"),
            "MDW/Templates/T.psd": ("100644", "t1")}
    art = cg.art_holders(index, tree)
    assert art == {"mdw/linked/x.png": (("Items/A.psd", "a1"),), "mdw/linked/y.png": (("MDW/Templates/T.psd", "t1"),)}
    context = SimpleNamespace(art=art)
    assert cg.hidden_of(None, cg.Job("Items/P.psd", "psd-only", dups={"Items/Q.png": "q"}), None, context).dups
    monkeypatch.setattr(cg.standards, "hidden_links", lambda psd, rel: {"mdw/linked/x.png", "mdw/linked/z.png"})
    monkeypatch.setattr(cg.standards, "shown_links", lambda psd, rel: {"mdw/linked/x.png"})
    monkeypatch.setattr(cg.png_export, "open_psd", lambda data, rel=None: None)
    found = cg.hidden_of(SimpleNamespace(blob=lambda sha: b""), cg.Job("Items/P.psd", "psd-only"), None, context)
    assert found.elsewhere == {"mdw/linked/x.png"} and not found.dups
    assert cg.hidden_of(None, cg.Job("MDW/Templates/T.psd", "psd-only"), None, context) is None


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
    (summary.json counts them, so Card sync's wait ends at Decided), the card on top of guard 1's PNG
    (its redrawn picture re-exported in it), its new MDW/Linked file beside it, the template with nothing else checked;
    one landing, no Discord line, no guard-event for the conversion, the commit message alone says it."""
    root, base, alice, bot = cards
    _psd(root, tmp_path / "psds", f"{FOLDER}/Snack.psd", card(colour=(0, 0, 200)))
    _psd(root, tmp_path / "psds", "MDW/Templates/Item_Template.psd", card())
    (root / FOLDER / "Snack2.png").unlink()     # an untagged NameN.png is refused, never made (owner 11-12)
    head = _commit(root, "Alice again", author="Alice")
    _links_module(monkeypatch)
    monkeypatch.setitem(cg.standards.PIXEL_FIXES, cg.NORMALISE, (lambda psd: [psd], None, None))
    real, converted = cg.standards.fix, []
    fixture_psd(tmp_path / "psds", card(colour=(0, 200, 0)), "Redrawn")   # Snack's picture drawn again (class 3)
    redrawn = (tmp_path / "psds" / "Redrawn.psd").read_bytes()

    def fix(data, rel, work, parser=None, fonts=None, only=None, skip=()):
        if only != cg.NORMALISE:
            assert cg.NORMALISE in skip            # never in the check itself
            return real(data, rel, work, parser, fonts, only, skip)
        assert isinstance(fonts.links, cg.LinkContext) and fonts.links.tree[rel]
        converted.append(rel)
        if rel.endswith("Snack.psd"):           # card_standards' shape: the redrawn layers' names
            return SimpleNamespace(data=redrawn, files={"MDW/Linked/Snack - art.png": b"art"},
                                   redrawn={cg.NORMALISE: ["Square"]}, left=[], pixel_left=[])
        return SimpleNamespace(data=data + b"linked", files={}, redrawn={cg.NORMALISE: ["art"]}, left=[], pixel_left=[])
    monkeypatch.setattr(cg.standards, "fix", fix)
    out = tmp_path / "out"
    data = cg.check(str(root), head, bot, mode="on", out=str(out), workers=1)
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
    assert {(e["kind"], e["path"]) for e in report["events"]} == {("guard-png", f"{FOLDER}/Snack.png")}   # guard 1's alone
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
    """Past GIT_LAND_BYTES, or with the old blobs in the history clone (a delta push), one git push, else the API; a
    broken git falls back to the API."""
    plan = _plan(tmp_path, [("A", "X/A.png", b"aaaa", {"X/A.psd": "s1"})])
    hub = FakeHub({"X/A.psd": ("100644", "s1"), "X/A.png": ("100644", "p0")})
    hub.install(monkeypatch)
    pushed, asked, here = [], [], [False]
    monkeypatch.setattr(cg, "git_commit", lambda repo, head, files, text: pushed.append(sorted(files)) or "g" * 40)
    monkeypatch.setattr(cg, "has_blobs", lambda shas: asked.append(shas) or here[0])
    monkeypatch.setattr(cg, "GIT_LAND_BYTES", 3)
    assert cg.land(plan)["sha"] == "g" * 40 and pushed == [["X/A.png"]] and not hub.commits
    monkeypatch.setattr(cg, "GIT_LAND_BYTES", 4)
    assert cg.land(plan)["sha"] == "c" * 40 and len(pushed) == 1 and asked == [["p0"]]     # the PNG it replaces
    here[0] = True
    assert cg.land(plan)["sha"] == "g" * 40 and len(pushed) == 2

    def broken(*args):
        raise gc.Transient("git push failed")
    monkeypatch.setattr(cg, "GIT_LAND_BYTES", 3)
    monkeypatch.setattr(cg, "git_commit", broken)
    assert cg.land(plan)["sha"] == "c" * 40 and len(hub.commits) == 2


def _entry(name, **extra):
    """A plan card with every key markdown reads (no writes)."""
    entry = {"card": name, "psd": f"X/{name}.psd", "case": "stale", "png": f"X/{name}.png", "login": "Alice",
             "commit": "a" * 40, "export": {}, "r10": {}, "issues": [], "notes": [],
             "told": [], "error": "", "writes": [], "expect": {f"X/{name}.psd": "s1", f"X/{name}.png": "p1"}}
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
    snack = _entry("Snack", export={"outcome": "re-exported", "changed_px": 5, "numbers": {}})
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
    assert "- Not landed (superseded: X/Snack.png changed on the branch): `X/Snack.png`" in comment
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
    new body and the settled post are written, exit 1, and the post tells the PNG that was not made (a PSD fix is silent)."""
    import requests
    out = tmp_path / "plan"
    snack = _entry("Snack", export={"outcome": "re-exported", "changed_px": 5, "numbers": {}})
    _write(out, snack, "X/Snack.png", "png", b"png", "re-exported")
    bread = _entry("Bread")
    _write(out, bread, "X/Bread.psd", "psd", b"psd", "straight apostrophes")
    bread["writes"][0].update(silent=True, pixels=["apostrophes"])
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
        assert "PNG not made: the bot's commit failed" in post and "could not fix" not in post


def test_a_rerun_after_a_landing_says_landed_not_superseded(tmp_path, monkeypatch):
    """The first attempt landed (the branch holds the bot's PNG and fixed PSD): the re-run writes nothing, drops nothing,
    tells nothing, and the comment keeps saying the writes are on the branch."""
    out = tmp_path / "plan"
    snack = _entry("Snack", export={"outcome": "re-exported", "changed_px": 5, "numbers": {}},
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
    assert "Nothing new to land: 1 card already on the branch" in comment
    assert "Every write is already on the branch" in comment and "Not landed" not in comment and "Not fixed" not in comment


def test_a_dry_run_says_what_on_would_make(tmp_path, monkeypatch):
    """A PSD pushed without its PNG: in on mode the PNG lands, nothing told; a dry run tells nothing either and its comment
    says the PNG would be re-exported (CARD_GUARDS=on), never that it was."""
    job = cg.Job("X/Snack.psd", "stale", "s0", "s1", "X/Snack.png", "p1", login="Alice", commit="a" * 40)
    export = {"outcome": "re-exported", "why": "", "gate": "", "changed_px": 1234, "box": [0, 0, 10, 10], "icc": "converted",
              "writer": "photoshop", "vote": None, "numbered_refused": [],
              "numbers": {"baseline": {"bad_pct": 0.0}, "cross": {"bad_pct": 0.01, "mean": 0.3}}}
    result = cg.Result(job, export=export, writes=[{"path": "X/Snack.png", "file": "files/Snack.png", "sha": blob_sha(b"png"),
                                                    "kind": "png", "line": "re-exported"}])
    dry = cg.plan([result], "o/r", "e" * 40, "f" * 40, "dry")
    (entry,) = dry["cards"]
    assert entry["told"] == [] and cg.discord_payload(dry, "o/r") == {}
    text = cg.markdown(dry, [], [], [])
    assert "Dry run: nothing is posted or committed." in text
    assert ("- PNG would be re-exported by the bot's render of the PSD (1,234 px changed in 10x10 at (0, 0); ICC converted; "
            "the old PSD drawn against the old PNG: bad 0.00%; against the PSD's own image: bad 0.01%, mean 0.30; saved by "
            "Photoshop) with CARD_GUARDS=on") in text and "- PNG re-exported" not in text
    on = cg.plan([result], "o/r", "e" * 40, "f" * 40, "on")
    assert on["cards"][0]["told"] == [] and "- PNG re-exported by the bot's render of the PSD (1,234 px" in cg.markdown(on, [], [], [])
    ahead = dict(entry, export={"outcome": "current", "numbers": {"png_ahead": {}}})              # png_export's verdict
    assert cg._guard1_line(ahead, "dry") == "PNG already shows the pushed PSD (exported before the PSD was pushed): nothing to make"


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
    assert result.told() == []
    assert cg._drawn_as(found) == "its picture 1"
    line = cg._guard1_line({"export": {**exported, "numbers": {}, "icc": "as stored"}, "case": "stale"})
    assert line.startswith("PNG re-exported by the bot's render of the PSD (") and "stored image" not in line
    for numbers in ({"kept": 33}, {"restyled": ["the Header text"]}):
        kept = cg._export_dict(pe.Export("current", unchecked=True, numbers=numbers))
        assert cg.Result(cg.Job("X/Snack.psd", "beside", png="X/Snack.png"), export=kept).told() == []
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
    assert _git(root, "log", "-1", "--format=%an %s", report["sha"]).startswith("AutoCroissant Card guards: 3 PNGs re-exported")
    changed = set(_git(root, "show", "--name-only", "--format=", report["sha"]).splitlines())
    assert changed == {f"{FOLDER}/Snack.png", f"{FOLDER}/New_Card.png", "Creatures/Other/3 Stars/New_Name.png"}
    again = cg.check(str(root), report["sha"], bot, mode="on", out=str(tmp_path / "again"), workers=1)
    assert again["cards"] == []                                                             # the bot's commit is never guarded


def test_a_card_file_name_a_file_cant_hold_is_renamed_silently(cards, tmp_path):
    """Owner 11-11: a pushed card PSD whose file name holds . : [ ] ? * " < > | / \\ (Foo.Bar.psd) takes its name's
    file form in the guards' commit with its PNG (the one the guards make too), the files keyed by the card following
    and a Renamed trailer each; nothing told. An MDW file of such a name is no card: never."""
    root, base, alice, bot = cards
    psd = (root / FOLDER / "Bread.psd").read_bytes()
    keyed = {"site/Card Authors.txt": b"[cards]\nFoo.Bar = Alice\n", "TTS/decks.json": f'{{"d": ["{FOLDER}/Foo.Bar.png"]}}'.encode()}
    first = _plumb(root, keyed, "keyed by the card")
    push = _plumb(root, {f"{FOLDER}/Foo.Bar.psd": psd, "MDW/Old.Stuff.psd": psd}, "a card")
    data = cg.check(str(root), push, first, mode="on", out=str(tmp_path / "out"), workers=1)
    new = f"{FOLDER}/FooBar"
    assert data["renames"] == {f"{FOLDER}/Foo.Bar.psd": f"{new}.psd", f"{FOLDER}/Foo.Bar.png": f"{new}.png"}
    assert data["discord"] == {} or not any("renamed" in str(v) for v in data["discord"].values())
    report = cg.land(str(tmp_path / "out" / "plan.json"), local=str(root))
    tree = set(_git(root, "ls-tree", "-r", "--name-only", report["sha"]).splitlines())
    assert {f"{new}.psd", f"{new}.png", "MDW/Old.Stuff.psd"} <= tree and not {f"{FOLDER}/Foo.Bar.psd", f"{FOLDER}/Foo.Bar.png"} & tree
    message = _git(root, "log", "-1", "--format=%B", report["sha"])
    assert f"Renamed: {FOLDER}/Foo.Bar.psd -> {new}.psd" in message and "2 files renamed" in message.splitlines()[0]
    assert _git(root, "rev-parse", f"{report['sha']}:{new}.psd") == _git(root, "rev-parse", f"{push}:{FOLDER}/Foo.Bar.psd")
    assert "FooBar = Alice" in _git(root, "show", f"{report['sha']}:site/Card Authors.txt")
    assert f"{new}.png" in _git(root, "show", f"{report['sha']}:TTS/decks.json")
    assert all(e["path"] != f"{FOLDER}/Foo.Bar.png" for e in report["events"])
    again = cg.land(str(tmp_path / "out" / "plan.json"), local=str(root))
    assert again["renamed"] == {}                                                       # a rerun: made already


# ========================
# Lossless recompression (owner, 2026-10-23)
# ========================
@pytest.mark.oxipng
def test_pushed_pngs_are_recompressed_in_the_bot_commit(cards, tmp_path):
    """A PNG the push brought (a pair's; a numbered one pushed alone) that a lossless recompression makes more than 3%
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
    told = cg.check(str(root), push, bot, mode="dry", out=str(tmp_path / "dry"), workers=1)
    assert [c["told"] for c in told["cards"] if c["card"] in ("Fat", "Snack")] == [[], []]
    assert "- Fat.png would be recompressed losslessly (" in (tmp_path / "dry" / "report.md").read_text()


@pytest.mark.oxipng
def test_recompressions_beside_the_check_write_the_same(cards, tmp_path, monkeypatch):
    """The pushed PNGs recompressed in a process of their own (_overlap) beside the rest of the check: the same writes,
    in the same order, as one by one (workers=1)."""
    root, base, alice, bot = cards
    work = tmp_path / "psds"
    big = art("RGB", (300, 450))
    fixture_psd(work, big, "Fat")
    push = _plumb(root, {f"{FOLDER}/Fat.psd": (work / "Fat.psd").read_bytes(), f"{FOLDER}/Fat.png": photoshop_png(big),
                         f"{FOLDER}/Fat2.png": photoshop_png(art("RGB", (320, 450))),
                         f"{FOLDER}/Snack2.png": photoshop_png(art("RGB", (300, 470)))}, "Photoshop exports")
    early, made = [], cg._made

    def spy(out, path, read, future=None):
        early.append(future is not None and future.exception() is None)
        return made(out, path, read, future)
    monkeypatch.setattr(cg, "_made", spy)
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)        # in this process: fewer cards than POOL_MIN
    writes = [[(c["card"], [(w["path"], w["kind"], w["sha"], w["line"]) for w in c["writes"]]) for c in
               cg.check(str(root), push, bot, mode="on", out=str(tmp_path / str(n)), workers=n)["cards"]] for n in (1, 0)]
    assert writes[0] == writes[1] and early == [False] * 3 + [True] * 3
    assert [(card, [Path(path).name for path, *_ in found]) for card, found in writes[1]] == [
        ("Fat", ["Fat2.png", "Fat.png"]), ("Snack", ["Snack2.png"])]


def test_a_recompression_rests_on_its_png_and_is_never_told(tmp_path, monkeypatch):
    """A recompressed PNG rests on that PNG alone (a later push of the PSD does not drop it) and is never told on
    Discord: not in a dry run, not when it did not land."""
    line = "recompressed losslessly (9.0 MB -> 2.0 MB, the same pixels)"
    job = cg.Job("X/Fat.psd", "pair", "", "s1", "X/Fat.png", png_new="p1", login="Alice", commit="a" * 40)
    result = cg.Result(job, writes=[{"path": "X/Fat.png", "file": "files/Fat.png", "sha": blob_sha(b"small"),
                                     "kind": "recompress", "line": line}])
    for mode in ("dry", "on"):
        planned = cg.plan([result], "o/r", "e" * 40, "f" * 40, mode)
        assert planned["cards"][0]["told"] == [] and cg.discord_payload(planned, "o/r") == {} and cg.summary(planned, [])["writes"] == 1
    assert f"- Fat.png would be {line} with CARD_GUARDS=on" in cg.markdown(cg.plan([result], "o/r", "e" * 40, "f" * 40, "dry"), [], [], [])
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
                        "line": "Single-line Composer (Ability)", "silent": True, "composer": True, "pixels": ["apostrophes"]}]
    bread["expect"] = cg._expect(cg.Job(bread["psd"], "pair", "", tree[bread["psd"]][1], f"{FOLDER}/Bread.png",
                                        png_new=tree[f"{FOLDER}/Bread.png"][1]))
    assert bread["expect"][f"{FOLDER}/Bread.png"] == tree[f"{FOLDER}/Bread.png"][1]
    plan["cards"] = [bread]
    (out / "plan.json").write_text(json.dumps(plan))
    report = cg.land(str(out / "plan.json"), local=str(root))
    assert report["landed"] == ["Bread"] and report["dropped"] == [] and report["sha"]
    assert _git(root, "show", "--name-only", "--format=", report["sha"]).splitlines() == [f"{FOLDER}/Bread.psd"]
    assert report["events"] == [{"kind": "guard-fix", "card": "Bread", "path": f"{FOLDER}/Bread.psd",    # never the composer
                                 "line": "straight apostrophes"}]


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
                            "silent": True, "pixels": ["normalise"]})
    bread["records"] = {"render": {bread["psd"]: {"icc": [0.0, 0]}}}
    (out / "plan.json").write_text(json.dumps(plan))
    landed = cg.land(str(out / "plan.json"), local=str(root))
    assert landed["sha"] and f"{FOLDER}/Bread_Copy.png" in _git(root, "show", "--name-only", "--format=", landed["sha"])
    cg.finish(str(out / "plan.json"), landed)
    assert json.loads((out / "records.json").read_text()) == [bread["records"]]


def test_a_drawn_duplicate_is_its_psds_picture(cards, clone, tmp_path):
    """Owner 11-09: tags pushed on a PSD whose picture 2 is a duplicate's PNG (Breakfast.png, dup_of A Snack: as
    Gust_of_Trees.png) make no A_Snack2.png: picture 2 is drawn onto the duplicate's PNG (current here), never followed."""
    from psd_tools import PSDImage
    sys.path.insert(0, str(Path(__file__).parent / "fixtures" / "parts"))
    from make_fixtures import CARDS
    root, base, alice, bot = cards
    rel, _, tags = CARDS["a_snack_tagged"]
    stem, dup = rel[:-4], f"{FOLDER}/Breakfast.png"
    for path, src in ((rel, rel), (stem + ".png", stem + ".png"), (dup, stem + "2.png"), (stem + "3.png", stem + "3.png")):
        shutil.copy(clone / src, root / path)
    (root / "site").mkdir()
    (root / "site/Card Edits.json").write_text(json.dumps({"cards": {"Breakfast": {"dup_of": "A Snack"}}}))
    start = _commit(root, "A Snack")
    psd = PSDImage.open(str(root / rel))
    for layer in psd.descendants():
        if layer.layer_id in tags:
            layer.name = tags[layer.layer_id]
    psd.save(str(root / rel))
    pushed = _commit(root, "A Snack tagged", author="Alice")
    out = tmp_path / "out"
    cg.check(str(root), pushed, start, mode="on", out=str(out), workers=1)
    (snack,) = [c for c in json.loads((out / "plan.json").read_text())["cards"] if c["card"] == "A Snack"]
    assert snack["dups"] == {} and dup in snack["expect"] and snack["export"]["outcome"] == "current", snack["export"]
    assert [w["path"] for w in snack["writes"] if w["kind"] == "png"] == [] and not snack["export"]["numbered_refused"]
    assert not [i for i in snack["issues"] if i["code"] == "R12"], snack["issues"]


def test_a_persons_drawn_duplicate_is_kept_as_pushed(cards, clone, tmp_path):
    """The 11-10 skeptic's repro: Stubborn King (tagged; picture 2 is Stubborn_Queen.png, dup_of) pushed by a person with
    hp 2 -> 3, Stubborn_King.png and a hand-edited Stubborn_Queen.png: the Queen is kept as pushed (as a pushed
    NameN.png is: recompressed at most), never drawn over; no Stubborn_King2.png."""
    from psd_tools import PSDImage
    root, base, alice, bot = cards
    folder = "Creatures/Other/2 Stars"
    rel, png, dup = f"{folder}/Stubborn_King.psd", f"{folder}/Stubborn_King.png", f"{folder}/Stubborn_Queen.png"
    (root / folder).mkdir(parents=True)
    for path in (png, dup):
        shutil.copy(clone / path, root / path)
    psd = PSDImage.open(str(clone / rel))
    tags = {3454: "Layer 5 #2", 422: "Name #1", 3455: "Name #2"}
    for layer in psd.descendants():
        if layer.layer_id in tags:
            layer.name = tags[layer.layer_id]
    psd.save(str(root / rel))
    (root / "site").mkdir()
    (root / "site/Card Edits.json").write_text(json.dumps({"cards": {"Stubborn Queen": {"dup_of": "Stubborn King"}}}))
    start = _commit(root, "Stubborn King tagged")
    for layer in psd.descendants():             # hp 2 -> 3: both pictures' hp cell
        if layer.layer_id in (479, 480):
            layer.visible = layer.layer_id == 480
    psd.save(str(root / rel))
    Image.open(clone / png).save(root / png)
    queen = Image.open(clone / dup).convert("RGB")
    ImageDraw.Draw(queen).rectangle((600, 1210, 654, 1259), fill=(255, 0, 255))     # the person's own pixels
    queen.save(root / dup)
    pushed = _commit(root, "Stubborn King hp 3", author="Alice")
    out = tmp_path / "out"
    cg.check(str(root), pushed, start, mode="on", out=str(out), workers=1)
    (king,) = [c for c in json.loads((out / "plan.json").read_text())["cards"] if c["card"] == "Stubborn King"]
    assert [w for w in king["writes"] if w["path"] == dup and w["kind"] != "recompress"] == [], king["writes"]
    assert not [w for w in king["writes"] if "Stubborn_King2" in w["path"]], king["writes"]
    assert king["case"] == "pair" and king["pushed"] == [dup] and king["dups"] == {}, king
    assert king["expect"][dup] == cg.Source(str(root)).tree(pushed)[dup][1]


# ========================
# The real push of 2026-10-22 15:18 (the clone, read-only)
# ========================
def test_replay_of_the_stale_png_push(clone, tmp_path):
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
    done = subprocess.run([executable, "-S", "-m", "commands.standards_post", "--help"], cwd=bot, capture_output=True,
                          text=True, timeout=120)                               # the tell job's: the runner's python3, no venv
    assert done.returncode == 0 and done.stdout.startswith("usage: python3 -m commands.standards_post"), done.stderr
    for module in ("card_check", "font_check"):
        done = subprocess.run([executable, "-m", f"commands.{module}", "--help"], cwd=bot, capture_output=True, text=True,
                              timeout=120)
        assert done.returncode == 0 and done.stdout.startswith(f"usage: python -m commands.{module}"), done.stderr
        code = f"import sys, commands.{module}; print(sorted(m for m in ('config', 'discord') if m in sys.modules))"
        done = subprocess.run([executable, "-c", code], cwd=bot, capture_output=True, text=True, timeout=120)
        assert done.returncode == 0 and done.stdout.strip() == "[]", done.stderr
    doc = _guard_steps(None)[0]
    for name in ("check", "checks", "tell"):                                   # its bot checkouts: that list
        job = doc["jobs"][name]
        bot = next(s for s in job["steps"] if s.get("with", {}).get("repository") == "bobberlington/AutoCroissant")
        assert bot_files(job, bot) == ["/" + f for f in cg.SPARSE_FILES], name


def _workflow(clone) -> Path:
    return SITE_SRC / ".github" / "workflows" / "card-guards.yml"


def _guard_steps(clone) -> tuple:
    """card-guards.yml's jobs read as one: (doc, every named step by name, the jobs' env merged: TOLD)."""
    doc = yaml.safe_load(_workflow(clone).read_text(encoding="utf-8"))
    jobs = doc["jobs"].values()
    return (doc, {s["name"]: s for job in jobs for s in job.get("steps", []) if s.get("name")},
            {k: v for job in jobs for k, v in job.get("env", {}).items()})


FAKE_GH = r"""#!/usr/bin/env python3
import fcntl, json, os, subprocess, sys
state = os.environ["FAKE_STATE"]
lock = open(state + ".lock", "w")
fcntl.flock(lock, fcntl.LOCK_EX)   # calls run side by side: one at a time here
db = json.load(open(state)) if os.path.exists(state) else {"comments": {}, "next": 100, "log": []}
args = sys.argv[1:]
db["log"].append(" ".join(args))
out = ""


def jq(data):
    # gh's --jq, by the real jq: the steps' expressions are under test too
    expr = args[args.index("--jq") + 1]
    return subprocess.run(["jq", "-r", expr], input=json.dumps(data), capture_output=True, text=True, check=True).stdout.rstrip("\n")


if args[:2] == ["workflow", "run"]:
    pass
elif args[0] == "api":
    method = args[args.index("--method") + 1] if "--method" in args else "GET"
    path = next(a for a in args[1:] if a.startswith("repos/"))
    if path.endswith("/jobs"):                  # the attempt's jobs: none
        out = jq({"jobs": []}) if "--jq" in args else json.dumps({"jobs": []})
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
exit 0
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


def _tell(steps, env, bot) -> dict:
    """The tell job's Told already and Tell Discord (the real standards_post on this python3, from bot/); their outputs."""
    out = _run_steps(steps, ["Told already"], env, {})
    if out["told"].get("told") == "true":
        return out
    done, said = _shell(steps["Tell Discord"], {**env, **{k: v for k, v in steps["Tell Discord"]["env"].items() if "${{" not in v},
                                                "GITHUB_OUTPUT": str(Path(env["OUT"]) / "tell.output")}, cwd=bot)
    assert done.returncode == 0, (done.stdout, done.stderr)
    return {**out, "tell": said}


def test_workflow_posts_once_and_a_rerun_edits_its_comment(clone, tmp_path):
    """The Comment, Told before, Follow up, Told already and Tell Discord steps' shell against a fake gh and curl
    (standards_post real, on this python3): the first run comments, Follow up rewrites it, the tell job posts ONE message
    (the guards' lines as Land settled them, the card check's new notes, owner 2026-11-11), marks
    the guards' comment and comments the card check's report; a re-run edits that comment, keeps the mark and posts
    nothing, and so does a re-run of the tell job alone (its Told already looks the mark up). With CARD_GUARDS off the
    card check alone posts, its comment carrying the mark."""
    doc, steps, jenv = _guard_steps(clone)
    bin_dir = _fake_tools(tmp_path)
    bot = tmp_path / "bot"
    (bot / "commands").mkdir(parents=True)
    shutil.copy(REPO / "commands" / "standards_post.py", bot / "commands")
    out, checks = tmp_path / "guards", tmp_path / "checks"
    out.mkdir()
    (checks / "card-check").mkdir(parents=True)
    lines = ["PNG not made: G6; export it in Photoshop"]
    (out / "summary.json").write_text(json.dumps({"after": "f" * 40, "cards": 1, "told": 1, "comment": 1, "writes": 0,
                                                  "errors": 0, "card_errors": 0}))
    (out / "plan.json").write_text(json.dumps({"after": "f" * 40, "cards": [
        {"card": "Alice", "psd": "X/Alice.psd", "login": "Alice", "commit": "a" * 40, "told": lines + ["withdrawn"],
         "issues": []}]}))
    (out / "landed.json").write_text(json.dumps({"told": {"X/Alice.psd": lines}}))
    (out / "report.md").write_text("## Card guards\n\nAlice's card.\n")
    (out / "report-landed.md").write_text("## Card guards\n\nAfter the window.\n")
    (checks / "card-check" / "report.md").write_text("## Card check\n\n- `X/Alice.psd`: no stars\n")
    (checks / "card-check" / "report.json").write_text(json.dumps({"cards": [
        {"path": "X/Alice.psd", "author": "Alice", "notes": ['NO STARS FOUND (x)', "STATS TOO HIGH"], "new": ['NO STARS FOUND (x)', "STATS TOO HIGH"]},
        {"path": "X/Bob.psd", "author": "Bob", "notes": ["STATS TOO HIGH"], "new": []}]}))     # Bob's old note: not told
    env = {"PATH": f"{bin_dir}:{os.environ['PATH']}", "OUT": str(out), "TOLD": jenv["TOLD"], "MODE": "on",
           "GITHUB_REPOSITORY": "o/cards", "WEBHOOK": "https://discord.invalid/hook", "FAKE_STATE": str(tmp_path / "gh.json"),
           "FAKE_POSTS": str(tmp_path / "posts.txt"), "GITHUB_STEP_SUMMARY": str(tmp_path / "summary.md"),
           "RUNNER_TEMP": str(tmp_path), "GITHUB_SHA": "f" * 40, "CHECKED": "success", "LAND": "true", "CHECKS": "success"}
    first = _run_steps(steps, ["Comment on the commit", "Told before", "Follow up"], env, {"land": {"ran": "true"}}, doc["jobs"])
    (comment_id, body), = json.loads((tmp_path / "gh.json").read_text())["comments"].items()
    first.update(_tell(steps, {**env, "COMMENT": comment_id}, bot))
    state = json.loads((tmp_path / "gh.json").read_text())
    assert first["comment"] == {"id": comment_id} and first["tell"] == {"told": "true"} and first["told"] == {}
    guards, card_check = state["comments"].values()
    assert guards.startswith("## Card guards\n\nAfter the window.") and guards.rstrip().endswith(jenv["TOLD"])
    assert card_check.startswith("## Card check") and jenv["TOLD"] not in card_check
    link = " [aaaaaaaa](https://github.com/o/cards/commit/aaaaaaaaaaaa)"
    assert json.loads((tmp_path / "posts.txt").read_text().strip()) == {
        "content": 'Alice pushed Alice: PNG not made: G6; export it in Photoshop; no stars (group them in "Stars"); a stat over 20.' + link,
        "allowed_mentions": {"parse": []}, "flags": 4}
    again = _run_steps(steps, ["Comment on the commit", "Told before"], env, {"land": {"ran": "true"}}, doc["jobs"])   # every job
    assert again["comment"] == {"told": "true", "id": comment_id} and again["told"] == {"told": "true"}
    assert _tell(steps, {**env, "CHECK_TOLD": "true"}, bot)["told"] == {"told": "true"}       # so Tell Discord is skipped
    alone = _tell(steps, {**env, "GITHUB_RUN_ATTEMPT": "2"}, bot)                             # the tell job alone
    assert alone["told"] == {"told": "true"} and "tell" not in alone
    state = json.loads((tmp_path / "gh.json").read_text())
    assert len(state["comments"]) == 2 and (tmp_path / "posts.txt").read_text().count("Alice pushed") == 1
    # CARD_GUARDS off: no guards' files; the card check's comment takes the mark, which a re-run finds
    shutil.rmtree(out)
    out.mkdir()
    (tmp_path / "gh.json").unlink()
    off = {**env, "MODE": "off", "CHECKED": "skipped", "LAND": "", "COMMENT": ""}
    assert _tell(steps, off, bot)["tell"] == {"told": "true"}
    (only,) = json.loads((tmp_path / "gh.json").read_text())["comments"].values()
    assert only.startswith("## Card check") and only.rstrip().endswith(jenv["TOLD"])
    assert _tell(steps, {**off, "GITHUB_RUN_ATTEMPT": "2"}, bot)["told"] == {"told": "true"}
    # a Land that settled nothing: the checks' part goes out, not told whole (the alert names the step)
    shutil.rmtree(out)
    out.mkdir()
    assert _tell(steps, {**env, "COMMENT": ""}, bot)["tell"] == {}
    assert "Alice pushed Alice: no stars" in (tmp_path / "posts.txt").read_text().splitlines()[-1]


def _fake_tools(tmp_path) -> Path:
    """gh, curl and a sleep that does not wait (the comment lookup's retries)."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    (bin_dir / "gh").write_text(FAKE_GH)
    (bin_dir / "curl").write_text(FAKE_CURL)
    (bin_dir / "sleep").write_text("#!/bin/sh\nexit 0\n")
    for tool in ("gh", "curl", "sleep"):
        (bin_dir / tool).chmod(0o755)
    return bin_dir


def _shell(step, env, cwd=None):
    out = Path(env["GITHUB_OUTPUT"])
    out.write_text("")
    done = subprocess.run(["bash", "--noprofile", "--norc", "-eo", "pipefail", "-c", step["run"]], env=env, cwd=cwd,
                          capture_output=True, text=True, timeout=60)
    return done, dict(line.split("=", 1) for line in out.read_text().splitlines() if "=" in line)


def test_land_commits_only_an_on_plan(tmp_path, capsys):
    """`check` defaults to --mode dry and `land` to a real commit (on the Mac the token comes from config.py): a plan not
    checked with --mode on is refused once its mode is read, before anything is written, unless --dry-run or --local."""
    for mode in ("dry",):
        plan = _full_plan(tmp_path, [_entry("Quiet")], mode=mode)
        for name in ("landed.json", "report-landed.md", "discord-final.json"):
            (Path(plan).parent / name).unlink(missing_ok=True)
        with pytest.raises(SystemExit) as stop:
            cg.main(["land", "--plan", plan])
        assert stop.value.code == 2 and f"checked with --mode {mode}" in capsys.readouterr().err
        assert not (Path(plan).parent / "landed.json").exists()
        assert cg.main(["land", "--plan", plan, "--dry-run"]) == 0 and (Path(plan).parent / "landed.json").exists()


def test_workflow_fetches_pushed_blobs_in_the_background(clone, tmp_path, monkeypatch):
    """Pushed blobs start fetches the push's blobs in the background and returns at once (its output not held); Check and
    the two checks start at once and wait for each listed blob (CARD_BLOBS_PENDING); Pushed blobs waits and counts. A
    fetch that fails: those blobs by REST, the warning and a failed step."""
    import commands.card_bake as cb
    import commands.github_api as gh
    _, steps, _ = _guard_steps(clone)
    for wait in ("Card and font checks", "Check start"):                       # the bot's wait reads the list
        assert steps[wait]["env"][gh.PENDING_ENV] == "${{ runner.temp }}/blobs"
    rest = []
    monkeypatch.setattr(gh, "fetch_blob", lambda repo, sha, timeout: rest.append(sha) or SimpleNamespace(content=subprocess.run(
        ["git", "-C", str(origin), "cat-file", "blob", sha], capture_output=True, check=True).stdout))
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
        _until(temp / "blobs.ready")
        monkeypatch.setenv(gh.PENDING_ENV, str(temp / "blobs"))
        listed = (temp / "blobs").read_text().split()
        rest.clear()
        read = (lambda sha: gh.fetch_verified("o/r", sha, clone=str(repo)),      # the checks (a REST read is kept)
                lambda sha: cb.fetch_verified("o/r", sha, local=str(repo)))[n]   # the guards
        assert [blob_sha(read(sha)) for sha in listed] == listed
        assert len(rest) == (len(listed) if code else 0)            # waited for, from the clone; a failed fetch: REST
        done, _ = _shell(steps["Pushed blobs"], env, cwd=repo)
        assert (done.returncode != 0) == bool(code) and said in " ".join(done.stdout.split()), (done.stdout, done.stderr)
    rest.clear()                                                    # the two checks, a blob not fetched: one REST read
    fetch = gh.fetch_blob
    monkeypatch.setattr(gh, "fetch_blob", lambda *a, **k: subprocess.run(["/bin/sleep", "0.5"]) and fetch(*a, **k))
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(2) as pool:
        assert {blob_sha(d) for d in pool.map(lambda _: gh.fetch_verified("o/r", listed[-1], clone=str(repo)), "ab")} == {
            listed[-1]} and rest == [listed[-1]]
    os.utime(temp / "blobs", (0, 0))                                # no .done: the cap ends the wait (a new process)
    (temp / "blobs.done").unlink()
    monkeypatch.setattr(gh, "_listed", {})
    assert gh.pending_blob(str(repo), listed[0]) is None
    monkeypatch.delenv(gh.PENDING_ENV)                              # the bake and the bot: no list, no wait
    assert gh.pending_blob(str(repo), listed[0]) is None


def _until(path, seconds=30):
    """Waits for a background step's mark."""
    subprocess.run(["timeout", str(seconds), "sh", "-c", f'until [ -e "{path}" ]; do sleep 0.05; done'], check=True)


FAKE_RAW = r"""#!/bin/sh
cat > "$FAKE_ARGS.config"
echo "$*" >> "$FAKE_ARGS"
while [ $# -gt 0 ]; do case "$1" in -o) out=$2; shift;; https://*) url=$1;; esac; shift; done
rest=${url#https://raw.githubusercontent.com/o/cards/}
git -C "$FAKE_REPO" cat-file blob "${rest%%/*}:${rest#*/}" > "$out"
"""


def test_workflow_fetches_earlier_icon_fonts_in_the_background(clone, tmp_path, monkeypatch):
    """Earlier fonts start lists the icon font, current first, then its past versions and fetches them by commit in the
    background, the token off the command line; Check's earlier_fonts reads them there (CARD_EARLIER_FONTS), no API and
    no wait for the fonts restore; one missing, or not its blob, by the API."""
    _, steps, _ = _guard_steps(clone)
    repo, temp, bin_dir = tmp_path / "repo", tmp_path / "temp", tmp_path / "bin"
    for folder in (repo, temp, bin_dir):
        folder.mkdir()
    (bin_dir / "curl").write_text(FAKE_RAW)
    (bin_dir / "curl").chmod(0o755)
    _git(repo, "init", "-q", "-b", "main")
    (repo / cg.ICON_FONT_FILE).parent.mkdir(parents=True)
    shas = []
    for n in range(3):
        (repo / cg.ICON_FONT_FILE).write_bytes(b"%d" % n)
        head = _commit(repo, f"font {n}")
        shas.append(blob_sha(b"%d" % n))
    env = {"PATH": f"{bin_dir}:{os.environ['PATH']}", "HOME": os.environ.get("HOME", ""), "RUNNER_TEMP": str(temp),
           "AFTER": head, "GITHUB_REPOSITORY": "o/cards", "GH_TOKEN": "s3cret", "GITHUB_ENV": str(temp / "env"),
           "GITHUB_OUTPUT": str(temp / "output"), "FAKE_REPO": str(repo), "FAKE_ARGS": str(tmp_path / "args")}
    done, _ = _shell(steps["Earlier fonts start"], env, cwd=repo)
    folder = temp / "earlier-fonts"
    assert done.returncode == 0 and (temp / "env").read_text() == f"{cg.EARLIER_ENV}={folder}\n", done.stderr
    listed = [line.split() for line in (folder / "list").read_text().splitlines()]
    assert [sha for _, sha in listed] == shas[::-1] and listed[0][0] == head
    wait = " && ".join(f'[ -f "{folder / sha}" ]' for sha in shas)
    wait = f'until {wait}; do sleep 0.1; done'
    subprocess.run(["timeout", "10", "sh", "-c", wait], check=True)
    assert "s3cret" not in (tmp_path / "args").read_text() and "s3cret" in (tmp_path / "args.config").read_text()

    class NoApi(cg.Source):
        def blob(self, sha):
            raise AssertionError(sha)
    monkeypatch.setenv(cg.EARLIER_ENV, str(folder))
    monkeypatch.setenv(cg.FONTS_READY_ENV, str(tmp_path / "never"))         # the restore not in: never waited for
    monkeypatch.setattr(cg, "FONTS_READY_S", 0)
    monkeypatch.setattr(cg, "font_glyphs", lambda data: {"glyph": int(data) + 1})
    monkeypatch.setattr(cg, "_glyphs", {})
    out = tmp_path / "out"
    out.mkdir()
    files = cg.earlier_fonts(NoApi(str(repo)), head, cg.Source(str(repo)).tree(head), str(out))
    assert [Path(path).read_bytes() for path in files] == [b"1", b"0"]
    (folder / shas[1]).write_bytes(b"9")
    (folder / shas[0]).unlink()
    read, source = [], cg.Source(str(repo))
    blob = source.blob
    source.blob = lambda sha: read.append(sha) or blob(sha)
    monkeypatch.setattr(cg, "_glyphs", {})
    (tmp_path / "again").mkdir()
    files = cg.earlier_fonts(source, head, source.tree(head), str(tmp_path / "again"))
    assert [Path(path).read_bytes() for path in files] == [b"1", b"0"] and sorted(read) == sorted(shas[:2])


def test_a_pushs_check_starts_beside_the_fonts_restore(clone, tmp_path, monkeypatch):
    """A push's Check starts at Check start; Check then waits and carries its output (stdout to the summary) and
    exit code over; a dispatch's runs at Check. Its font reads wait for Fonts ready's mark (fonts_for, FONTS_WAIT)."""
    import threading
    import time
    import commands.psd_edit as psd_edit
    _, steps, _ = _guard_steps(clone)
    start, check = steps["Check start"], steps["Check"]
    assert start["env"][cg.FONTS_READY_ENV] == "${{ runner.temp }}/fonts.ready" and cg.FONTS_READY_ENV not in check["env"]
    fake = tmp_path / "bin" / "python"
    fake.parent.mkdir()
    fake.write_text('#!/bin/sh\necho "$*" > "$FAKE_ARGS"\nsleep 0.3\necho report\necho warn >&2\nexit "$FAKE_EXIT"\n')
    fake.chmod(0o755)
    for n, (event, code) in enumerate((("push", 3), ("push", 0), ("dispatch", 1))):
        temp, summary = tmp_path / f"t{n}", tmp_path / f"summary{n}"
        temp.mkdir()
        env = {"PATH": f"{fake.parent}:{os.environ['PATH']}", "RUNNER_TEMP": str(temp), "GITHUB_STEP_SUMMARY": str(summary),
               "GITHUB_OUTPUT": str(temp / "out"), "GITHUB_WORKSPACE": "/w", "GITHUB_REPOSITORY": "o/cards", "OUT": "/o",
               "GITHUB_SHA": "a" * 40, "MODE": "on", "BEFORE": "b" * 40, "AUTHOR": "maker", "FAKE_ARGS": str(tmp_path / "args"),
               "FAKE_EXIT": str(code)}
        if event == "push":
            done, _ = _shell(start, env)
            assert done.returncode == 0 and not summary.exists(), done.stderr
        done, _ = _shell(check, env)
        assert (done.returncode, done.stdout, done.stderr.strip(), summary.read_text()) == (code, "report\n", "warn", "report\n")
        assert (tmp_path / "args").read_text().split() == ["-m", "commands.card_guards", "check", "--clone", "/w", "--repo",
            "o/cards", "--after", "a" * 40, "--out", "/o", "--mode", "on"] + ["--author", "maker", "--before", "b" * 40] * (event == "push")
    root, mark = tmp_path / "clone", tmp_path / "fonts.ready"
    (root / psd_edit.FONTS_DIR).mkdir(parents=True)
    (root / psd_edit.FONTS_DIR / "a.ttf").write_bytes(b"x")
    monkeypatch.setenv(cg.FONTS_READY_ENV, str(mark))
    monkeypatch.setattr(psd_edit, "FONTS_WAIT", None)
    cg.wait_fonts()
    assert psd_edit.FONTS_WAIT is cg.fonts_ready
    timer, began = threading.Timer(0.3, mark.touch), time.monotonic()
    timer.start()
    assert cg.Source(str(root)).fonts_for(["A"]) == str(root / psd_edit.FONTS_DIR) and time.monotonic() - began >= 0.25
    timer.join()


def test_the_bot_clone_and_records_run_here(clone, tmp_path):
    """check's Bot start and Bot against a stand-in GitHub (file://, a fake gh): SPARSE_FILES alone in bot/, no header
    saved; the records read beside it (one not read in 3 tries: a warning), CARD_RECORDS_DIR at once. checks reads no
    records; a failed clone leaves no bot/ (the checkout after it brings one)."""
    doc, steps, jenv = _guard_steps(clone)
    seed, server, bin_dir = tmp_path / "seed", tmp_path / "github", tmp_path / "bin"
    for rel in cg.SPARSE_FILES + ("config.py",):
        (seed / rel).parent.mkdir(parents=True, exist_ok=True)
        (seed / rel).write_text(rel)
    subprocess.run(["git", "init", "-q", "-b", "main", str(seed)], check=True)
    _commit(seed, "bot")
    _git(tmp_path, "clone", "-q", "--bare", str(seed), str(server / steps["Bot start"]["env"]["BOT_REPO"]))
    bin_dir.mkdir()
    (bin_dir / "gh").write_text('#!/bin/sh\necho "$*" >> "$RUNNER_TEMP/gh.log"\n'
                                'case "$*" in *render_record*) echo \'{"a": 1}\';; *) exit 1;; esac\n')
    (bin_dir / "gh").chmod(0o755)
    for name, job, url in (("ok", "check", f"file://{server}"), ("failed", "checks", f"file://{tmp_path}/none")):
        workspace, runner = tmp_path / name / "ws", tmp_path / name / "temp"
        workspace.mkdir(parents=True)
        runner.mkdir()
        env = {"PATH": f"{bin_dir}:{os.environ['PATH']}", "HOME": os.environ.get("HOME", ""), "GIT_CONFIG_GLOBAL": "/dev/null",
               "RUNNER_TEMP": str(runner), "GITHUB_OUTPUT": str(runner / "output"), "GITHUB_ENV": str(runner / "env"),
               "GITHUB_SERVER_URL": url, "GITHUB_REPOSITORY": "o/cards", "GITHUB_JOB": job, "GH_TOKEN": "t0ken",
               "BOT_REPO": steps["Bot start"]["env"]["BOT_REPO"], "BOT_FILES": jenv["BOT_FILES"]}
        started, _ = _shell(steps["Bot start"], env, cwd=workspace)
        later = dict(line.split("=", 1) for line in (runner / "env").read_text().splitlines()) if (runner / "env").is_file() else {}
        done, outputs = _shell(steps["Bot"], {**env, **later}, cwd=workspace)
        assert started.returncode == done.returncode == 0 and "::add-mask::" in started.stdout, done.stdout
        assert (outputs.get("ok") == "true") == (name == "ok") == (workspace / "bot").is_dir()
        assert ("::warning::the bot's clone failed" in done.stdout) != (name == "ok")
        assert later.get("CARD_RECORDS_DIR") == (str(runner / "records") if job == "check" else None)
        assert ("::warning::db/spot_record.json not read" in done.stdout) == (job == "check")
    assert [p.name for p in (tmp_path / "ok" / "temp" / "records").iterdir()] == ["render_record.json"]
    asked = (tmp_path / "ok" / "temp" / "gh.log").read_text()
    assert (asked.count("render_record"), asked.count("spot_record")) == (1, 3) and not (tmp_path / "failed" / "temp" / "gh.log").exists()
    bot = tmp_path / "ok" / "ws" / "bot"
    assert sorted(str(p.relative_to(bot)) for p in bot.rglob("*") if p.is_file() and ".git" not in p.parts) == sorted(cg.SPARSE_FILES)
    assert "extraheader" not in (bot / ".git" / "config").read_text()


def test_workflow_lints(clone):
    """actionlint accepts card-guards.yml: a file GitHub rejects fails every run, pushes included."""
    done = subprocess.run(["actionlint", "-shellcheck=", str(_workflow(clone))], capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stdout + done.stderr


def test_a_composer_fix_is_said_only_in_the_commit(tmp_path, monkeypatch):
    entry = _entry("A", composer=["Name"])
    _write(tmp_path / "plan", entry, "X/A.psd", "psd", b"fixed", "Single-line Composer (Name)")
    entry["writes"][0].update(silent=True, composer=True)
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
    entry["writes"][0].update(silent=True, composer=False, pixels=["apostrophes"])
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


def test_kept_icon_layers_skip_their_layouts(cards, tmp_path, monkeypatch):
    """ICON_LAYOUTS are left out only when the push kept every icon layer byte for byte and the icon font is the one the
    PSD was last changed with; a changed icon layer, or a changed icon font, runs all three."""
    root, base, alice, bot = cards
    rel, psds = f"{FOLDER}/Snack.psd", tmp_path / "psds"
    font = root / cg.ICON_FONT_FILE
    font.parent.mkdir(parents=True, exist_ok=True)
    font.write_bytes(b"font 1")
    _commit(root, "font", author="Alice")
    _psd(root, psds, rel, card(colour=(0, 0, 200)))
    before = _commit(root, "Snack with font 1", author="Alice")
    _psd(root, psds, rel, card(colour=(0, 200, 0)))     # the Square layer changes, Base stays
    head = _commit(root, "Snack again", author="Alice")
    _git(root, "checkout", "-q", "-b", "font", before)
    _psd(root, psds, rel, card(colour=(0, 200, 0)))
    font.write_bytes(b"font 2")
    refont = _commit(root, "Snack and a new font", author="Alice")
    icons, skips = ["Base"], []
    monkeypatch.setattr(cg.standards, "icon_layers", lambda psd: [l for l in psd.descendants() if l.name in icons])
    monkeypatch.setattr(cg.standards, "pixel_kinds", lambda psd, kinds=(), *a, **k: () if kinds else
                        (*cg.ICON_LAYOUTS, "status words"))
    monkeypatch.setattr(cg.standards, "fix", lambda *a, skip=(), **k: skips.append(skip) or cg.standards.Fixed())
    monkeypatch.setattr(cg, "convert_plan", lambda *a, **k: None)
    for after, shown in ((head, "Base"), (head, "Square"), (refont, "Base")):
        icons[:] = [shown]
        cg.check(str(root), after, before, out=str(tmp_path / f"out-{len(skips)}"), workers=1)
    assert skips == [(cg.NORMALISE, *cg.ICON_LAYOUTS), (cg.NORMALISE,), (cg.NORMALISE,)]


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
    entry["writes"][0].update(silent=True, composer=True, pixels=["apostrophes"],
                              plain={"file": "files/A.psd.plain", "sha": blob_sha(b"plain"),
                                     "line": "Single-line Composer (Ability)"})
    _write(out, entry, "X/A.png", "png", b"png2", "re-exported for the Single-line Composer's rewrap")
    entry["writes"][1].update(silent=True, composer=True)
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


def test_box_text_fonts_only_from_the_clone(tmp_path, monkeypatch):
    """R8's box text and the pixel fixes without a clone's MDW/Fonts: none (owner 11-12: no per-file download, the
    redraws are left out); the pool's workers get the checked tree (review 2026-10-25)."""
    monkeypatch.delenv(cg.FONT_DIRS_ENV, raising=False)
    tree = {"MDW/Fonts/cambriab.ttf": ("100644", "sha-cambriab")}
    source = cg.Source(str(tmp_path / "clone"), "o/r")
    source.checked = tree
    assert source.fonts_for(["Cambria-Bold"]) == ""
    (tmp_path / "clone" / cg.psd_edit.FONTS_DIR).mkdir(parents=True)
    (tmp_path / "clone" / cg.psd_edit.FONTS_DIR / "cambriab.ttf").write_bytes(b"font")
    assert source.fonts_for(["Cambria-Bold"]) == str(tmp_path / "clone" / cg.psd_edit.FONTS_DIR)
    monkeypatch.setattr(cg, "_worker", {})
    monkeypatch.setattr(cg, "token_from_env", lambda: None)
    vocab = type("V", (), {"parser": lambda self: None})()
    cg._init_worker(str(tmp_path / "clone"), "o/r", None, vocab, tree, ("old.earlier",))
    assert cg._worker["source"].checked == tree and cg._worker["earlier"] == ("old.earlier",)


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
    """Owner 11-07, 11-09: the status words and status font fixes land in the guards' commit silently, no guard-event
    for them or their PNG."""
    psd = {"kind": "psd", "path": "a.psd", "pixels": ["status font", "status words"]}
    png = {"kind": "png", "path": "a.png", "silent": True, "line": "re-exported"}
    assert cg.events_of({"card": "A"}, psd, [psd, png]) == cg.events_of({"card": "A"}, png, [psd, png]) == []
