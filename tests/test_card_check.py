"""
commands/card_check.py offline: the clone's ground-truth cards through check_files and the CLI, --all on a tmp checkout
with the clone's Types/ names and a synthetic edits file, cards as plain CardInfo objects for the edits and --repo paths
(a fake GitHub), the Discord payload, the Action's sparse copy (SPARSE_FILES, --help without config.py).
"""
import json
import re
import shutil
from pathlib import Path
from subprocess import CalledProcessError, run as run_process
from sys import executable

import pytest

import commands.card_check as cc
import commands.github_api as ga
from commands.card_models import CardInfo, CardStats, card_name
from conftest import REPO, Reply, blob_sha_bytes as _sha
from test_font_check import FakeGitHub

NO_STARS = 'NO STARS FOUND (put the star layers in a group named "Stars")'
SEVERAL = "SEVERAL CREATURES: check each one's stats on the website"
NOTES = {                                       # what /update_stats printed on 2026-10-21 (e4e0a58) for these cards
    "N.M.E/Bubble.psd": [],                                           # fixed upstream 2026-10-22: OLD
    "Auxiliary/Minions/God_Tamer.psd": [],
    "Auxiliary/Rulebook/20_Creature_Types.psd": [],
    "Creatures/Kirby/3 Stars/Computer_Virus.psd": ["HP NOT FOUND", "DEF NOT FOUND", "ATK NOT FOUND", "SPD NOT FOUND"],
    "Creatures/Isaac/3 Stars/Tainted_Lazarus.psd": [],                # SEVERAL checked on the website (D7)
    "Creatures/Kirby/3 Stars/Pix.psd": [],
    "Creatures/World Of Warcraft/4 Stars/Twin_Emperors.psd": [],
    "Auxiliary/Minions/Anubisath_Guardian.psd": [],                   # ABILITY_EXCLUSIONS since e4e0a58
    "Auxiliary/Minions/Qiraji_Soldier.psd": [],
    "Auxiliary/Minions/Silithid.psd": [],
    "Auxiliary/Minions/Mini_Doomer.psd": [],
    "Creatures/FTL/5 Stars/Royal_Eradicator_Main_Cannon.psd": [],     # STATS TOO HIGH allowed by the clone's Card Edits.json
}
OLD = {                                         # the clone's last versions with notes (fixed by these commits' children)
    "N.M.E/Bubble.psd": ("dc37f4adf5309ff0ebacadfbd2f2e07001eec3a5^", [NO_STARS]),          # L14 star groups
    "Auxiliary/Minions/God_Tamer.psd": ("dc37f4adf5309ff0ebacadfbd2f2e07001eec3a5^", [NO_STARS]),
    "Auxiliary/Rulebook/20_Creature_Types.psd": ("100511d37848773bb8e948ba29625397f6ce5216^", ["MISSPELT TYPE: tornado"]),   # L11
}


# ========================
# Helpers
# ========================
def creature(rel: str, hp=-1, defense=-1, attack=-1, speed=-1, problems=(), types=(), ability="x", card_type="creature") -> CardInfo:
    """A CardInfo as the parser would build it for rel."""
    return CardInfo(card_name(rel), card_type, path=rel, ability=ability, types=list(types), problems=list(problems),
                    stats=CardStats(hp, defense, attack, speed))


def edits(cards: dict) -> str:
    return json.dumps({"cards": cards})


def fake_root(root: Path, files: dict, types=("fire.png", "active.png"), edits_text=None) -> Path:
    """A checkout: files {rel: bytes}, Types/ names, the edits file."""
    for rel, data in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(data)
    (root / "Types" / "Stars").mkdir(parents=True, exist_ok=True)
    for name in types:
        (root / "Types" / name).write_bytes(b"")
    (root / "Types" / "Stars" / "Dst Star.png").write_bytes(b"")
    if edits_text is not None:
        (root / "site").mkdir(exist_ok=True)
        (root / "site" / "Card Edits.json").write_text(edits_text, encoding="utf-8")
    return root


def old_blob(clone, rel: str) -> bytes:
    """OLD[rel]'s PSD from the clone's history."""
    try:
        return run_process(["git", "-C", str(clone), "show", f"{OLD[rel][0]}:{rel}"], capture_output=True, check=True).stdout
    except (CalledProcessError, OSError):
        pytest.skip(f"the clone's history lacks {OLD[rel][0]}:{rel}")


def old_checkout(clone, root: Path) -> Path:
    """The OLD PSDs beside the clone's Types/, MDW/ (the manifest) and site/ (the edits file)."""
    for rel in OLD:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(old_blob(clone, rel))
    for top in ("Types", "MDW", "site"):
        (root / top).symlink_to(clone / top)
    return root


@pytest.fixture
def fake_cards(monkeypatch):
    """parse_card answers from a table keyed by the PSD file's bytes (bytes -> rel -> CardInfo), so no PSD is needed."""
    table = {}

    def fake(parser, psd, rel):
        data = Path(psd).read_bytes()
        if data not in table:
            raise ValueError("unreadable")
        return table[data](rel)
    monkeypatch.setattr(cc, "parse_card", fake)
    return table


# ========================
# Vocabulary and edits
# ========================
def test_vocabulary_reads_the_tree_like_the_traverser():
    paths = ["Types/Fire.png", "Types/Fragile Equip.png", "Types/Stars/Dst Star.png", "Types/Stars/x.txt", "Types", "Types/Stars",
             "Typesetting/a.png", "Items/A.psd", "MDW/Fonts/stoogedings-src/stoogedings.json"]
    vocab = cc.vocabulary(paths, b'{"icons": {"fire": {"kind": "type", "pua": "e000"}}}')
    assert vocab.all_types == ["fire", "fragile equip", "a"] and vocab.stars == ["dst_star"]    # the traverser's filter, quirks included
    assert vocab.manifest == {"icons": {"fire": {"kind": "type", "pua": "e000"}}}
    parser = vocab.parser()
    assert parser.all_types == ["fire", "fragile equip", "a"] and parser.icons.lookup("dst_star") == ("dst_star", "star")
    assert cc.vocabulary([]).manifest is None and cc.Vocabulary().parser().all_types == []


def test_clone_vocabulary_and_edits(tmp_path):
    root = fake_root(tmp_path / "repo", {}, types=("Fire.png", "Defensive_Minion.png", ".DS_Store"),
                     edits_text=edits({"Royal_Eradicator Main Cannon": {"allow_high_stats": True, "_by": "x"}}))
    (root / "MDW" / "Fonts" / "stoogedings-src").mkdir(parents=True)
    (root / "MDW" / "Fonts" / "stoogedings-src" / "stoogedings.json").write_text('{"icons": {}}', encoding="utf-8")
    vocab = cc.clone_vocabulary(str(root))
    assert vocab.all_types == ["defensive_minion", "fire"] and vocab.stars == ["dst_star"] and vocab.manifest == {"icons": {}}
    errors = []
    assert cc.clone_edits(str(root), errors).cards == {"royal eradicator main cannon": {"allow_high_stats": True}} and errors == []
    (root / "site" / "Card Edits.json").write_text("not json", encoding="utf-8")
    assert cc.clone_edits(str(root), errors) is None and errors[0].startswith("site/Card Edits.json: could not be read (")
    assert cc.clone_edits(str(tmp_path), []) is None and cc.clone_vocabulary(str(tmp_path)) == cc.Vocabulary()


def test_clone_has_the_live_vocabulary(clone):
    vocab = cc.clone_vocabulary(str(clone))
    assert len(vocab.all_types) == 35 and {"active", "fragile equip", "defensive_minion", "wind"} <= set(vocab.all_types)
    assert "dst_star" in vocab.stars and "icons" in vocab.manifest
    assert cc.clone_edits(str(clone), []).cards["royal eradicator main cannon"]["allow_high_stats"] is True


def test_clone_readers_match_the_traversers(clone, monkeypatch):
    """The bot's own readers (/update_stats) over the clone as a tree: the same types, icons and edits as clone_vocabulary/clone_edits."""
    import commands.repo_traverser as rt
    paths = sorted(p.relative_to(clone).as_posix() for p in clone.rglob("*") if p.is_file() and ".git" not in p.parts)
    traverser = rt.RepositoryTraverser(cc.StatsDatabase())
    traverser.tree = {"tree": [{"path": p, "type": "blob", "sha": p} for p in paths]}
    traverser.shas = {p: p for p in paths}
    monkeypatch.setattr(traverser, "_repo_text", lambda path: (clone / path).read_text(encoding="utf-8"))
    traverser._populate_types_from_response(None)
    icons = traverser._icons_from_response(None, "o/r")
    vocab = cc.clone_vocabulary(str(clone))
    assert vocab.all_types == traverser.db.all_types and vocab.parser().icons == icons
    assert cc.clone_edits(str(clone), []).cards == traverser._edits_from_response(None, "o/r").cards


def test_checkout_root(tmp_path):
    root = fake_root(tmp_path / "repo", {"Items/Heal/1 Stars/A.psd": b""})
    assert cc.checkout_root(str(root / "Items/Heal/1 Stars/A.psd")) == str(root)
    assert cc.checkout_root(str(tmp_path / "lone.psd")) == ""


# ========================
# One card: edits, then the validator
# ========================
def test_card_notes_apply_the_edits_then_validate(fake_cards, tmp_path):
    rel = "Creatures/Kirby/3 Stars/A.psd"
    psd = tmp_path / "a.psd"
    psd.write_bytes(b"a")
    parser = cc.Vocabulary().parser()
    fake_cards[b"a"] = lambda r: creature(r, defense=4, attack=5, speed=6)
    assert cc.card_notes(parser, None, str(psd), rel) == ["HP NOT FOUND"]
    assert cc.card_notes(parser, cc.CardEdits.parse(edits({"A": {"hp": 5}})), str(psd), rel) == []
    assert cc.card_notes(parser, cc.CardEdits.parse(edits({"a": {"problems_ok": True}})), str(psd), rel) == []
    fake_cards[b"a"] = lambda r: creature(r, 7, 4, 5, 6, problems=[SEVERAL])
    assert cc.card_notes(parser, None, str(psd), rel) == [SEVERAL]
    two = edits({"A": {"creatures": [{"hp": 7, "def": 4, "atk": 5, "spd": 6}, {"hp": 1, "def": 1, "atk": 1, "spd": 1}]}})
    assert cc.card_notes(parser, cc.CardEdits.parse(two), str(psd), rel) == []
    fake_cards[b"a"] = lambda r: creature(r, 7, 4, 12, 6)
    assert cc.card_notes(parser, None, str(psd), rel) == ["STATS TOO HIGH"]
    assert cc.card_notes(parser, cc.CardEdits.parse(edits({"A": {"allow_high_stats": True}})), str(psd), rel) == []
    fake_cards[b"a"] = lambda r: creature(r, 7, 4, -1, -1, types=["active"], ability=None)
    assert cc.card_notes(parser, None, str(psd), rel) == ["ABILITY TEXT NOT FOUND"]
    fake_cards[b"a"] = lambda r: creature(r, 7, 4, 5, 6, problems=["NO ABILITY LAYER"], ability=None, card_type="minion")
    assert cc.card_notes(parser, None, str(psd), "Auxiliary/Minions/Whelp.psd") == []      # ABILITY_EXCLUSIONS (validate strips the layer note)
    assert cc.card_notes(parser, None, str(psd), rel) == ["ABILITY TEXT NOT FOUND", "NO ABILITY LAYER"]
    fake_cards[b"a"] = lambda r: creature(r, problems=["NO ABILITY LAYER"], ability=None, card_type="unknown")
    assert cc.card_notes(parser, None, str(psd), "A.psd") == ["UNKNOWN TYPE", "NO ABILITY LAYER"]


# ========================
# The clone's cards
# ========================
def test_clone_cards_match_update_stats(clone):
    report = cc.check_files([str(clone / rel) for rel in NOTES], "tester")
    assert report.checked == len(NOTES) and report.errors == []
    assert {c.path: c.notes for c in report.cards} == {rel: notes for rel, notes in NOTES.items() if notes}
    assert all(c.author == "tester" for c in report.cards)
    assert [c.path for c in report.cards] == [rel for rel, notes in NOTES.items() if notes]     # command-line order


def test_old_clone_cards(clone, tmp_path):
    """NO STARS FOUND and MISSPELT TYPE: tornado on the clone's last versions that had them."""
    root = old_checkout(clone, tmp_path / "repo")
    report = cc.check_files([str(root / rel) for rel in OLD], "tester")
    assert report.checked == len(OLD) and report.errors == []
    assert [(c.path, c.notes) for c in report.cards] == [(rel, notes) for rel, (_, notes) in OLD.items()]   # command-line order


def test_cli_psd_mode(clone, tmp_path, capsys):
    out, disc = tmp_path / "r.json", tmp_path / "d.json"
    old = old_checkout(clone, tmp_path / "repo")                                 # God Tamer's NO STARS FOUND
    psds = [old / "Auxiliary/Minions/God_Tamer.psd", clone / "Auxiliary/Minions/Mini_Doomer.psd",
            clone / "Creatures/Kirby/3 Stars/Computer_Virus.psd"]
    assert cc.main(["--psd", *map(str, psds), "--author", "bobber", "--json", str(out), "--discord", str(disc)]) == 0
    text = capsys.readouterr().out
    assert text.startswith("## Card check\n\n3 card PSDs read.\n\n")
    assert f"- `Auxiliary/Minions/God_Tamer.psd` (bobber): {NO_STARS}\n" in text
    assert "- `Creatures/Kirby/3 Stars/Computer_Virus.psd` (bobber): HP NOT FOUND; DEF NOT FOUND; ATK NOT FOUND; SPD NOT FOUND\n" in text
    assert "Mini_Doomer" not in text and "Errors" not in text
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data == {"checked": 3, "errors": [],
                    "cards": [{"path": "Auxiliary/Minions/God_Tamer.psd", "author": "bobber", "notes": [NO_STARS]},
                              {"path": "Creatures/Kirby/3 Stars/Computer_Virus.psd", "author": "bobber",
                               "notes": ["HP NOT FOUND", "DEF NOT FOUND", "ATK NOT FOUND", "SPD NOT FOUND"]}]}
    payload = json.loads(disc.read_text(encoding="utf-8"))
    assert payload["content"].splitlines() == [
        f"bobber has pushed God Tamer with: {NO_STARS}.",
        "bobber has pushed Computer Virus with: HP NOT FOUND; DEF NOT FOUND; ATK NOT FOUND; SPD NOT FOUND.",
        cc.CLOSING]
    assert payload["allowed_mentions"] == {"parse": []} and len(payload["content"]) < 2000
    assert "http" not in payload["content"] and "token" not in payload["content"].lower()
    assert cc.main(["--psd", str(clone / "Auxiliary/Minions/Mini_Doomer.psd"), "--discord", str(disc)]) == 0
    assert "Nothing to report." in capsys.readouterr().out and json.loads(disc.read_text(encoding="utf-8")) == {}


def test_cli_all_mode_with_the_edits(clone, tmp_path, capsys):
    """Six real cards (two from the clone's history) in a tmp checkout: the edits file answers four of their notes; MDW/, Markers/ and .git/ are skipped."""
    root = tmp_path / "repo"
    cards = ["Auxiliary/Minions/God_Tamer.psd", "Creatures/Kirby/3 Stars/Computer_Virus.psd", "Creatures/Kirby/3 Stars/Pix.psd",
             "Auxiliary/Rulebook/20_Creature_Types.psd", "Auxiliary/Minions/Mini_Doomer.psd",
             "Creatures/FTL/5 Stars/Royal_Eradicator_Main_Cannon.psd"]
    for rel in cards:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        if rel in OLD:                                                          # God Tamer and 20_Creature_Types with their notes
            (root / rel).write_bytes(old_blob(clone, rel))
        else:
            (root / rel).symlink_to(clone / rel)
    names = [p.name for p in (clone / "Types").iterdir() if p.is_file() and not p.name.startswith(".")]
    text = edits({"Computer Virus": {"problems_ok": True},
                  "Pix": {"creatures": [{"hp": 3, "def": 3, "atk": 3, "spd": 3}] * 3},
                  "20_Creature_Types": {"types": ["wind"]},
                  "Royal Eradicator Main Cannon": {"allow_high_stats": True}})
    fake_root(root, {"MDW/Templates/Creature_Template.psd": b"", "Auxiliary/Markers/M.psd": b"", ".git/x.psd": b"", "Items/note.txt": b""},
              types=names, edits_text=text)
    assert cc.card_psds(str(root)) == [(rel, str(root / rel)) for rel in sorted(cards)]
    out = tmp_path / "r.json"
    assert cc.main(["--all", str(root), "--json", str(out)]) == 0
    text = capsys.readouterr().out
    assert "6 card PSDs read." in text and f"- `Auxiliary/Minions/God_Tamer.psd` (Someone): {NO_STARS}\n" in text
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["cards"] == [{"path": "Auxiliary/Minions/God_Tamer.psd", "author": "Someone", "notes": [NO_STARS]}] and data["errors"] == []
    (root / "site" / "Card Edits.json").unlink()                                 # without the edits the four notes come back
    report = cc.check_checkout(str(root))
    assert {c.path: c.notes for c in report.cards} == {
        "Auxiliary/Minions/God_Tamer.psd": [NO_STARS],
        "Auxiliary/Rulebook/20_Creature_Types.psd": ["MISSPELT TYPE: tornado"],
        "Creatures/FTL/5 Stars/Royal_Eradicator_Main_Cannon.psd": ["STATS TOO HIGH"],
        "Creatures/Kirby/3 Stars/Computer_Virus.psd": ["HP NOT FOUND", "DEF NOT FOUND", "ATK NOT FOUND", "SPD NOT FOUND"],
        "Creatures/Kirby/3 Stars/Pix.psd": [SEVERAL]}


def test_check_checkout_pool_matches_sequential(clone, tmp_path, monkeypatch):
    """--all from POOL_MIN PSDs parses in a process pool (a parser per worker): the same report as the sequential path."""
    n = cc.POOL_MIN
    psds = sorted(clone.joinpath("Auxiliary", "Minions").glob("*.psd"), key=lambda p: p.stat().st_size)[:n]
    if len(psds) < n:
        pytest.skip("not enough card PSDs in the clone")
    root = tmp_path / "repo"
    for path in psds:
        link = root / path.relative_to(clone)
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(path)
    shutil.copytree(clone / "Types", root / "Types")
    pooled = cc.check_checkout(str(root))
    monkeypatch.setattr(cc, "POOL_MIN", n + 1)
    sequential = cc.check_checkout(str(root))
    assert pooled.as_dict() == sequential.as_dict() and pooled.checked + len(pooled.errors) == n


def test_unreadable_psd_is_one_line_and_exit_0(tmp_path, capsys):
    root = fake_root(tmp_path / "repo", {"Items/Heal/1 Stars/Bad.psd": b"not a psd"})
    (tmp_path / "Lone.psd").write_bytes(b"not a psd")
    assert cc.main(["--psd", str(root / "Items/Heal/1 Stars/Bad.psd"), str(tmp_path / "Lone.psd")]) == 0
    text = capsys.readouterr().out
    assert "0 card PSDs read." in text and "Nothing to report." in text
    assert re.search(r"^- Items/Heal/1 Stars/Bad\.psd: could not be read \(.+\)$", text, re.M)
    assert re.search(r"^- Lone\.psd: could not be read \(.+\)$", text, re.M)


def test_psd_under_mdw_or_markers_is_no_card(fake_cards, tmp_path, capsys):
    """/update_stats never parses MDW/ or Markers/: a hand run on one is an error line, not a card."""
    root = fake_root(tmp_path / "repo", {"MDW/Templates/T.psd": b"a", "Auxiliary/Markers/M.psd": b"a", "Items/A.psd": b"a"})
    fake_cards[b"a"] = lambda r: creature(r, card_type="item", problems=[NO_STARS])
    assert cc.main(["--psd", str(root / "MDW/Templates/T.psd"), str(root / "Auxiliary/Markers/M.psd"), str(root / "Items/A.psd")]) == 0
    text = capsys.readouterr().out
    assert "1 card PSD read." in text and f"- `Items/A.psd` (Someone): {NO_STARS}\n" in text
    assert "- MDW/Templates/T.psd: not a card PSD (" in text and "- Auxiliary/Markers/M.psd: not a card PSD (" in text


def test_cli_psd_mode_without_a_checkout(fake_cards, tmp_path, capsys):
    """A file outside any Types/-holding folder is an unknown card; one inside is classified by its path from there."""
    root = fake_root(tmp_path / "repo", {"Items/Heal/1 Stars/A.psd": b"a"}, edits_text=edits({"A": {"stars": 1}}))
    (tmp_path / "A.psd").write_bytes(b"a")
    fake_cards[b"a"] = lambda r: creature(r, card_type="item" if "/" in r else "unknown", problems=[NO_STARS])
    report = cc.check_files([str(root / "Items/Heal/1 Stars/A.psd"), str(tmp_path / "A.psd")])
    assert [(c.path, c.notes) for c in report.cards] == [("A.psd", ["UNKNOWN TYPE", NO_STARS])]       # the edits fixed the first


# ========================
# Reports
# ========================
def report() -> cc.Report:
    r = cc.Report(checked=3)
    r.cards = [cc.Card("Auxiliary/Minions/God_Tamer.psd", "bobber", [NO_STARS]),
               cc.Card("Creatures/Kirby/3 Stars/Computer_Virus.psd", "michael", ["HP NOT FOUND", "DEF NOT FOUND"])]
    return r


def test_markdown():
    text = cc.markdown(report())
    assert text.startswith("## Card check\n\n3 card PSDs read.\n\n")
    assert f"- `Auxiliary/Minions/God_Tamer.psd` (bobber): {NO_STARS}\n" in text
    assert "- `Creatures/Kirby/3 Stars/Computer_Virus.psd` (michael): HP NOT FOUND; DEF NOT FOUND\n" in text
    empty = cc.markdown(cc.Report(checked=1, errors=["x.psd: could not be read (boom)"]))
    assert "1 card PSD read." in empty and "Nothing to report." in empty and "\nErrors:\n- x.psd: could not be read (boom)\n" in empty
    assert report().as_dict() == {"checked": 3, "errors": [], "cards": [
        {"path": "Auxiliary/Minions/God_Tamer.psd", "author": "bobber", "notes": [NO_STARS]},
        {"path": "Creatures/Kirby/3 Stars/Computer_Virus.psd", "author": "michael", "notes": ["HP NOT FOUND", "DEF NOT FOUND"]}]}


def test_discord_payload_wording():
    payload = cc.discord_payload(report())
    assert set(payload) == {"content", "allowed_mentions"} and payload["allowed_mentions"] == {"parse": []}
    assert payload["content"] == (f"bobber has pushed God Tamer with: {NO_STARS}.\n"
                                  "michael has pushed Computer Virus with: HP NOT FOUND; DEF NOT FOUND.\n"
                                  "The bot's next /update_stats prints these too; the parser's own notes (no stars found, misspelt type, "
                                  "several creatures) also show on the website as Parser notes.")
    assert "website shows these" not in payload["content"]        # the validator's notes never reach the website's Parser notes
    assert "<@" not in payload["content"]                                       # nobody is pinged (owner, 2026-10-23)
    assert cc.discord_payload(cc.Report(checked=2, errors=["x.psd: could not be read (boom)"])) == {}


def test_discord_payload_fits_one_message():
    r = cc.Report(cards=[cc.Card(f"Items/Heal/1 Stars/Card_{i}.psd", "bobber", ["HP NOT FOUND", "DEF NOT FOUND"]) for i in range(200)])
    content = cc.discord_payload(r)["content"]
    assert len(content) <= 2000 and content.startswith("bobber has pushed Card 0 with: ")
    lines = content.splitlines()
    assert lines[len(lines) - 1] == cc.CLOSING and re.fullmatch(r"\.\.\. and \d+ more cards", lines[len(lines) - 2])
    kept = len(lines) - 2                                  # cards, then "... and N more", then CLOSING
    assert int(lines[len(lines) - 2].split()[2]) == 200 - kept and kept > 20


# ========================
# --repo mode against a fake GitHub
# ========================
def tree_and_blobs(edits_text: str, psds: dict) -> tuple:
    manifest = b'{"icons": {}}'
    text = edits_text.encode("utf-8")
    blobs = {_sha(manifest): manifest, _sha(text): text, **{_sha(d): d for d in psds.values()}}
    entries = [{"path": "Types/fire.png", "type": "blob", "sha": "1" * 40}, {"path": "Types/active.png", "type": "blob", "sha": "2" * 40},
               {"path": "Types/Stars/Dst Star.png", "type": "blob", "sha": "3" * 40}, {"path": "Types", "type": "tree", "sha": "t"},
               {"path": "MDW/Fonts/stoogedings-src/stoogedings.json", "type": "blob", "sha": _sha(manifest)},
               {"path": "site/Card Edits.json", "type": "blob", "sha": _sha(text)},
               {"path": "MDW/Templates/T.psd", "type": "blob", "sha": _sha(psds["a"])}]
    entries += [{"path": path, "type": "blob", "sha": _sha(data)} for path, data in psds.items() if "/" in path]
    return {"tree": entries}, blobs


def test_repo_context(monkeypatch):
    tree, blobs = tree_and_blobs(edits({"A": {"hp": 5}}), {"a": b"psd-a"})
    fake = FakeGitHub(blobs, {}, {}, tree)
    monkeypatch.setattr(ga, "get", fake.get)
    listing = {e["path"]: e["sha"] for e in tree["tree"] if e["type"] == "blob"}
    errors = []
    parser, card_edits = cc.repo_context("o/r", listing, errors)
    assert errors == [] and parser.all_types == ["fire", "active"] and parser.icons.lookup("dst_star") == ("dst_star", "star")
    assert card_edits.cards == {"a": {"hp": 5}} and fake.blob_calls() == 2
    fake.broken.add(_sha(b'{"icons": {}}'))
    parser, card_edits = cc.repo_context("o/r", listing, errors)
    assert len(errors) == 1 and errors[0].startswith("MDW/Fonts/stoogedings-src/stoogedings.json: blob ") and card_edits is not None
    del listing["site/Card Edits.json"]
    assert cc.repo_context("o/r", {k: v for k, v in listing.items() if "stoogedings" not in k}, [])[1] is None


def test_repo_mode_push(fake_cards, monkeypatch, tmp_path, capsys):
    a, b, c = b"psd-a", b"psd-b", b"psd-c"
    fake_cards[a] = lambda r: creature(r, defense=4, attack=5, speed=6)             # HP NOT FOUND, answered by the edits
    fake_cards[b] = lambda r: creature(r, card_type="item", problems=[NO_STARS])
    psds = {"a": a, "Creatures/Kirby/3 Stars/A.psd": a, "Items/Heal/1 Stars/B.psd": b, "N.M.E/C.psd": c}
    tree, blobs = tree_and_blobs(edits({"A": {"hp": 5}}), psds)
    files = [{"filename": "Creatures/Kirby/3 Stars/A.psd", "status": "modified", "sha": _sha(a)},
             {"filename": "Items/Heal/1 Stars/B.psd", "status": "added", "sha": _sha(b)},
             {"filename": "N.M.E/C.psd", "status": "renamed", "sha": _sha(c)},
             {"filename": "MDW/Templates/T.psd", "status": "modified", "sha": _sha(a)},
             {"filename": "Auxiliary/Markers/M.psd", "status": "modified", "sha": _sha(a)},
             {"filename": "Items/Gone.psd", "status": "removed", "sha": "0" * 40},
             {"filename": "site/Card Edits.json", "status": "modified", "sha": "4" * 40}]
    commits = {"c1": {"sha": "c1", "author": {"login": "alice"}, "files": [files[0]]},
               "c2": {"sha": "c2", "author": None, "files": [files[1], files[2], files[6]]}}   # no GitHub account on the email
    fake = FakeGitHub(blobs, {"commits": [commits["c1"], commits["c2"]], "files": files}, commits, tree)
    fake.broken.add(_sha(c))
    monkeypatch.setattr(ga, "get", fake.get)
    out, disc = tmp_path / "r.json", tmp_path / "d.json"
    assert cc.main(["--repo", "o/r", "--before", "b" * 40, "--after", "c2", "--author", "pusher", "--json", str(out), "--discord", str(disc)]) == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["cards"] == [{"path": "Items/Heal/1 Stars/B.psd", "author": "pusher", "notes": [NO_STARS]}]
    assert data["checked"] == 2 and data["errors"] == [f"N.M.E/C.psd: blob {_sha(c)[:8]} hashes to {_sha(b'garbage')[:8]}"]
    assert fake.blob_calls() == 5                                                   # the manifest, the edits file, three card PSDs
    assert any(u.endswith("/git/trees/c2?recursive=1") for u in fake.calls)
    text = capsys.readouterr().out
    assert f"- `Items/Heal/1 Stars/B.psd` (pusher): {NO_STARS}\n" in text and "A.psd" not in text
    payload = json.loads(disc.read_text(encoding="utf-8"))
    assert payload["content"].splitlines()[0] == f"pusher has pushed B with: {NO_STARS}."
    assert "http" not in payload["content"] and "token" not in payload["content"].lower()


def test_repo_mode_whole_tree_and_dispatch(fake_cards, monkeypatch, tmp_path):
    a = b"psd-a"
    fake_cards[a] = lambda r: creature(r, 1, 1, 1, 1, problems=["MISSPELT TYPE: tornado"])
    tree, blobs = tree_and_blobs(edits({}), {"a": a, "Creatures/Kirby/3 Stars/A.psd": a})
    head = {"sha": "h", "author": {"login": "alice"}, "files": [{"filename": "Creatures/Kirby/3 Stars/A.psd", "status": "added", "sha": _sha(a)}]}
    fake = FakeGitHub(blobs, {}, {"h": head}, tree)
    monkeypatch.setattr(ga, "get", fake.get)
    out = tmp_path / "r.json"
    assert cc.main(["--repo", "o/r", "--after", "h", "--whole-tree", "--author", "pusher", "--json", str(out)]) == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data == {"checked": 1, "errors": [], "cards": [{"path": "Creatures/Kirby/3 Stars/A.psd", "author": "pusher", "notes": ["MISSPELT TYPE: tornado"]}]}
    assert not any("compare" in u or "commits/" in u for u in fake.calls)       # the tree alone; MDW/Templates/T.psd is no card
    assert cc.main(["--repo", "o/r", "--after", "h", "--author", "pusher", "--json", str(out)]) == 0   # a dispatch: the head commit's files
    assert json.loads(out.read_text(encoding="utf-8"))["cards"][0]["author"] == "alice"


def test_repo_mode_api_failure_is_one_line_and_exit_0(monkeypatch, tmp_path, capsys):
    """A 5xx after the retries, no network or a truncated tree: one GitHub API error line, {} to Discord, exit 0 (never a red push)."""
    monkeypatch.setattr(ga, "sleep", lambda s: None)                              # get_with_retry's 1 s and 2 s waits
    replies = {"503": lambda url: Reply(503), "truncated": lambda url: Reply(body={"truncated": True, "tree": []})}

    def connection_error(url):
        from requests import ConnectionError
        raise ConnectionError("no network")
    replies["offline"] = connection_error
    out, disc = tmp_path / "r.json", tmp_path / "d.json"
    for case, expected in (("503", "GitHub API: 503"), ("truncated", "GitHub API: the tree listing is truncated"), ("offline", "GitHub API: no network")):
        monkeypatch.setattr(ga, "get", lambda url, **kw: replies[case](url))
        assert cc.main(["--repo", "o/r", "--before", "b" * 40, "--after", "c2", "--json", str(out), "--discord", str(disc)]) == 0
        text = capsys.readouterr().out
        assert "0 card PSDs read." in text and "Nothing to report." in text and f"\nErrors:\n- {expected}" in text, case
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["cards"] == [] and data["checked"] == 0 and data["errors"][0].startswith(expected)
        assert json.loads(disc.read_text(encoding="utf-8")) == {}


# ========================
# The Action's sparse copy
# ========================
def test_action_shaped_copy_imports_and_prints_help(tmp_path):
    """Exactly SPARSE_FILES (what card-check.yml checks out), no config.py: --help works, the closure is that list, no Discord."""
    bot = tmp_path / "bot"
    for rel in cc.SPARSE_FILES:
        assert (REPO / rel).is_file(), f"{rel} is in SPARSE_FILES but missing here"
        (bot / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO / rel, bot / rel)
    done = run_process([executable, "-m", "commands.card_check", "--help"], cwd=bot, capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    assert done.stdout.startswith("usage: python -m commands.card_check")
    assert set(re.findall(r"--[a-z-]+", done.stdout)) == {"--psd", "--all", "--repo", "--before", "--after", "--author", "--whole-tree",
                                                          "--json", "--discord", "--help"}
    code = ("import sys, commands.card_check; "
            "print(sorted(m for m in ('config', 'discord', 'commands.psd_analyzer', 'commands.repo_traverser', 'commands.query_card', "
            "'commands.utils', 'commands.psd_edit', 'commands.psd_regions', 'pandas') if m in sys.modules)); "
            "print(sorted(m for m in sys.modules if m.startswith('commands.') or m == 'global_config'))")
    done = run_process([executable, "-c", code], cwd=bot, capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    bad, closure = done.stdout.splitlines()
    assert bad == "[]"
    assert {m.replace(".", "/") + ".py" for m in json.loads(closure.replace("'", '"'))} == set(cc.SPARSE_FILES)   # the yml copies this list
    from test_card_bake import ALLOWED_IMPORTS, third_party_imports
    extra = third_party_imports("import commands.card_check", bot) - ALLOWED_IMPORTS
    assert not extra, f"card-check.yml installs none of these: {sorted(extra)}"


def test_cli_only_and_uncompiled():
    """No slash command, no help entry, not in setup.py's INCLUDE_FILES; the module imports under the tests' stub config (above)."""
    for rel in ("main.py", "commands/help.py", "setup.py"):
        assert "card_check" not in (REPO / rel).read_text(encoding="utf-8"), f"{rel} names card_check"
    assert callable(cc.main) and cc.SPARSE_FILES[0] == "commands/card_check.py"
