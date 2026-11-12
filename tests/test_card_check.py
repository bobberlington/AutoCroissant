"""
commands/card_check.py offline: the clone's ground-truth cards through check_files and the CLI, --all on a tmp checkout
with the clone's Types/ names and a synthetic edits file, cards as plain CardInfo objects for the edits and --repo paths
(a fake GitHub), the Discord payload (the guards' checkout runs it: test_card_guards.py).
"""
import json
import re
import shutil
from pathlib import Path
from subprocess import run as run_process

import pytest

import commands.card_check as cc
import commands.github_api as ga
from commands.card_models import CardInfo, CardStats, card_name
from conftest import Reply, at_rev, blob_sha_bytes as _sha, smallest
from helpers import ChecksGitHub as FakeGitHub

NO_STARS = 'NO STARS FOUND (put the star layers in a group named "Stars")'
NO_STATS = "NO STATS FOUND: no HP, DEF, ATK or SPD bars read on this creature"
OLD = {                                         # the clone's last versions with notes (fixed by these commits' children)
    "N.M.E/Bubble.psd": ("dc37f4adf5309ff0ebacadfbd2f2e07001eec3a5^", [NO_STARS]),          # L14 star groups
    "Auxiliary/Minions/God_Tamer.psd": ("dc37f4adf5309ff0ebacadfbd2f2e07001eec3a5^", [NO_STARS]),
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
    return at_rev(clone, OLD[rel][0], rel)


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


def test_clone_readers_match_the_traversers(clone, monkeypatch):
    """The bot's own readers (/update_stats) over the clone as a tree: the same types, icons and edits as clone_vocabulary/clone_edits."""
    import commands.card_data as cd
    tracked = run_process(["git", "-C", str(clone), "ls-files", "-z"], capture_output=True, check=True).stdout
    paths = sorted(p for p in tracked.decode().split("\0") if p)
    traverser = cd.CardBuild(cc.StatsDatabase())
    traverser.tree = {"tree": [{"path": p, "type": "blob", "sha": p} for p in paths]}
    traverser.shas = {p: p for p in paths}
    monkeypatch.setattr(traverser, "_repo_text", lambda path: (clone / path).read_text(encoding="utf-8"))
    traverser._populate_types_from_response(None)
    icons = traverser._icons_from_response(None, "o/r")
    vocab = cc.clone_vocabulary(str(clone))
    assert vocab.all_types == traverser.db.all_types and vocab.parser().icons == icons
    assert cc.clone_edits(str(clone), []).cards == traverser._edits_from_response(None, "o/r").cards


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
    fake_cards[b"a"] = lambda r: creature(r, 7, 4, 5, 6, problems=[NO_STATS])
    assert cc.card_notes(parser, None, str(psd), rel) == [NO_STATS]
    two = edits({"A": {"creatures": [{"hp": 7, "def": 4, "atk": 5, "spd": 6}, {"hp": 1, "def": 1, "atk": 1, "spd": 1}]}})
    assert cc.card_notes(parser, cc.CardEdits.parse(two), str(psd), rel) == []
    fake_cards[b"a"] = lambda r: creature(r, 7, 4, 12, 6)
    assert cc.card_notes(parser, None, str(psd), rel) == []                                 # up to 20 (11-06)
    fake_cards[b"a"] = lambda r: creature(r, 7, 4, 22, 6)
    assert cc.card_notes(parser, None, str(psd), rel) == ["STATS TOO HIGH"]
    assert cc.card_notes(parser, cc.CardEdits.parse(edits({"A": {"allow_high_stats": True}})), str(psd), rel) == []
    fake_cards[b"a"] = lambda r: creature(r, 7, 4, -1, -1, types=["active"], ability=None)
    assert cc.card_notes(parser, None, str(psd), rel) == []                                 # cards need no ability (10-25)
    fake_cards[b"a"] = lambda r: creature(r, 7, 4, 5, 6, problems=["NO ABILITY LAYER"], ability=None, card_type="minion")
    assert cc.card_notes(parser, None, str(psd), rel) == []                                 # older data's note: dropped
    fake_cards[b"a"] = lambda r: creature(r, problems=[NO_STATS])          # no bar read: one note, not four
    assert cc.card_notes(parser, None, str(psd), rel) == [NO_STATS]
    assert cc.card_notes(parser, cc.CardEdits.parse(edits({"A": {"hp": 5}})), str(psd), rel) == [NO_STATS]   # as the website
    assert cc.card_notes(parser, cc.CardEdits.parse(edits({"A": {"hp": 25}})), str(psd), rel) == ["STATS TOO HIGH", NO_STATS]
    four = {"hp": 1, "def": 1, "atk": 1, "spd": 1}
    assert cc.card_notes(parser, cc.CardEdits.parse(edits({"A": four})), str(psd), rel) == []           # every stat: answered
    assert cc.card_notes(parser, cc.CardEdits.parse(edits({"A": {"creatures": [four]}})), str(psd), rel) == []


# ========================
# The clone's cards
# ========================
def test_cli_psd_mode(clone, tmp_path, capsys):
    out, disc = tmp_path / "r.json", tmp_path / "d.json"
    old = old_checkout(clone, tmp_path / "repo")                                 # God Tamer's NO STARS FOUND
    psds = [old / "Auxiliary/Minions/God_Tamer.psd", clone / "Auxiliary/Minions/Mini_Doomer.psd",
            clone / "Creatures/Kirby/3 Stars/Computer_Virus.psd"]
    assert cc.main(["--psd", *map(str, psds), "--author", "bobber", "--json", str(out), "--discord", str(disc)]) == 0
    text = capsys.readouterr().out
    assert text.startswith("## Card check\n\n3 card PSDs read.\n\n")
    assert '- `Auxiliary/Minions/God_Tamer.psd` (bobber): no stars (group them in "Stars")\n' in text
    assert "- `Creatures/Kirby/3 Stars/Computer_Virus.psd` (bobber): no stat bars\n" in text
    assert "Mini_Doomer" not in text and "Errors" not in text
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data == {"checked": 3, "errors": [],
                    "cards": [{"path": "Auxiliary/Minions/God_Tamer.psd", "author": "bobber", "notes": [NO_STARS]},
                              {"path": "Creatures/Kirby/3 Stars/Computer_Virus.psd", "author": "bobber",
                               "notes": [NO_STATS]}]}
    payload = json.loads(disc.read_text(encoding="utf-8"))
    assert payload["content"].splitlines() == [
        'bobber pushed God Tamer: no stars (group them in "Stars").',
        "bobber pushed Computer Virus: no stat bars."]
    assert payload["allowed_mentions"] == {"parse": []} and len(payload["content"]) < 2000
    assert "http" not in payload["content"] and "token" not in payload["content"].lower()
    assert cc.main(["--psd", str(clone / "Auxiliary/Minions/Mini_Doomer.psd"), "--discord", str(disc)]) == 0
    assert "Nothing to report." in capsys.readouterr().out and json.loads(disc.read_text(encoding="utf-8")) == {}


def test_cli_all_mode_with_the_edits(clone, tmp_path, capsys):
    """Five real cards (one from the clone's history) in a tmp checkout: the edits file answers Computer Virus's note; MDW/, Markers/ and .git/ are skipped."""
    root = tmp_path / "repo"
    cards = ["Auxiliary/Minions/God_Tamer.psd", "Creatures/Kirby/3 Stars/Computer_Virus.psd", "Creatures/Kirby/3 Stars/Pix.psd",
             "Auxiliary/Minions/Mini_Doomer.psd",
             "Creatures/FTL/5 Stars/Royal_Eradicator_Main_Cannon.psd"]
    for rel in cards:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        if rel in OLD:                                                          # God Tamer with its note
            (root / rel).write_bytes(old_blob(clone, rel))
        else:
            (root / rel).symlink_to(clone / rel)
    names = [p.name for p in (clone / "Types").iterdir() if p.is_file() and not p.name.startswith(".")]
    text = edits({"Computer Virus": {"problems_ok": True}})
    fake_root(root, {"MDW/Templates/Creature_Template.psd": b"", "Auxiliary/Markers/M.psd": b"", ".git/x.psd": b"", "Items/note.txt": b""},
              types=names, edits_text=text)
    assert cc.card_psds(str(root)) == [(rel, str(root / rel)) for rel in sorted(cards)]
    out = tmp_path / "r.json"
    assert cc.main(["--all", str(root), "--json", str(out)]) == 0
    text = capsys.readouterr().out
    assert "5 card PSDs read." in text and '- `Auxiliary/Minions/God_Tamer.psd` (Someone): no stars (group them in "Stars")\n' in text
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["cards"] == [{"path": "Auxiliary/Minions/God_Tamer.psd", "author": "Someone", "notes": [NO_STARS]}] and data["errors"] == []
    (root / "site" / "Card Edits.json").unlink()                                 # without the edits the two notes come back
    report = cc.check_checkout(str(root))
    assert {c.path: c.notes for c in report.cards} == {
        "Auxiliary/Minions/God_Tamer.psd": [NO_STARS],
        "Creatures/Kirby/3 Stars/Computer_Virus.psd": [NO_STATS]}


def test_check_checkout_pool_matches_sequential(clone, tmp_path, monkeypatch):
    """--all from POOL_MIN PSDs parses in a process pool (a parser per worker): the same report as the sequential path."""
    n = cc.POOL_MIN
    psds = smallest(clone, "Auxiliary/Minions", n)
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


# ========================
# Reports
# ========================
def report() -> cc.Report:
    r = cc.Report(checked=3)
    r.cards = [cc.Card("Auxiliary/Minions/God_Tamer.psd", "bobber", [NO_STARS]),
               cc.Card("Creatures/Kirby/3 Stars/Computer_Virus.psd", "michael", ["HP NOT FOUND", "DEF NOT FOUND"])]
    return r


def test_discord_payload_wording():
    payload = cc.discord_payload(report())
    assert set(payload) == {"content", "allowed_mentions", "flags"} and payload["allowed_mentions"] == {"parse": []}
    assert payload["content"] == ('bobber pushed God Tamer: no stars (group them in "Stars").\n'
                                  "michael pushed Computer Virus: no HP bar; no DEF bar.")      # what to do, short (owner 2026-11-06)
    assert "website shows these" not in payload["content"]        # the validator's notes never reach the website's Parser notes
    assert "<@" not in payload["content"]                                       # nobody is pinged (owner, 2026-10-23)
    assert cc.discord_payload(cc.Report(checked=2, errors=["x.psd: could not be read (boom)"])) == {}


def test_one_standards_post_per_push():
    """standards_post (owner 2026-11-11): the guards' lines, the card check's new notes and the font check's new fonts in
    one post, a line per card in path order with its commit (the head's when the guards name none), a note both checks
    tell once (layer ids: the guards' wording, stars: the card check's), the role pinged only for a missing font and kept
    when lines are cut."""
    from commands.standards_post import payload
    ids = "layers share an id (7): Photoshop's Duplicate Layer gives a new one"
    plan = {"cards": [{"card": "A", "psd": "X/A.psd", "login": "netverk", "commit": "1" * 40, "told": [ids, "the stars are not in a group named Stars"],
                       "issues": [{"code": "R13", "text": ids}, {"code": "R6", "text": "the stars are not in a group named Stars"}]}]}
    checks = {"cards": [{"path": "X/A.psd", "author": "netverk", "notes": ["DUPLICATE LAYER IDS", NO_STARS], "new": ["DUPLICATE LAYER IDS", NO_STARS]},
                        {"path": "X/B.psd", "author": "bob", "notes": [NO_STATS], "new": []}]}
    fonts = {"cards": [{"path": "X/C.psd", "author": "bob", "missing": {"Futura": ["Name"]}, "new": ["Futura"]}]}
    post = payload("o/r", "f" * 40, plan=plan, card_check=checks)
    assert post == {"content": f'netverk pushed A: {ids}; no stars (group them in "Stars"). [11111111](https://github.com/o/r/commit/{"1" * 12})',
                    "allowed_mentions": {"parse": []}, "flags": 4}
    post = payload("o/r", "f" * 40, plan=plan, card_check=checks, font_check=fonts)
    assert post["content"].splitlines()[0] == "<@&546427268751294475>" and post["allowed_mentions"] == {"roles": ["546427268751294475"]}
    assert post["content"].splitlines()[2] == ("bob pushed C: the Futura font is not in MDW/Fonts (Name), bite this user. "
                                               f"[ffffffff](https://github.com/o/r/commit/{'f' * 12})")
    many = {"cards": [{"path": f"X/{i:03}.psd", "author": "bob", "missing": {"Futura": ["Name"]}} for i in range(40)]}
    cut = payload("o/r", "f" * 40, font_check=many)["content"].splitlines()
    assert cut[0] == "<@&546427268751294475>" and cut[-1] == f"... and {42 - len(cut)} more cards: [details](https://github.com/o/r/commit/{'f' * 40})"
    assert payload("o/r", "f" * 40, plan={"cards": [dict(plan["cards"][0], told=[])]}, card_check={"cards": [checks["cards"][1]]}) == {}


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


def test_repo_mode_push(fake_cards, monkeypatch, tmp_path, capsys):
    """The pushed PSDs by blob; a card's notes its PSD before the push lacked are its new ones (owner 2026-11-11: only
    those reach the post), every note for a new card."""
    a, b, c, d, d0 = b"psd-a", b"psd-b", b"psd-c", b"psd-d", b"psd-d0"
    fake_cards[a] = lambda r: creature(r, defense=4, attack=5, speed=6)             # HP NOT FOUND, answered by the edits
    fake_cards[b] = lambda r: creature(r, card_type="item", problems=[NO_STARS])
    fake_cards[d] = lambda r: creature(r, 1, 1, 1, 1, problems=["DUPLICATE LAYER IDS", "MISSPELT TYPE: fier"])
    fake_cards[d0] = lambda r: creature(r, 1, 1, 1, 1, problems=["DUPLICATE LAYER IDS"])
    psds = {"a": a, "Creatures/Kirby/3 Stars/A.psd": a, "Items/Heal/1 Stars/B.psd": b, "N.M.E/C.psd": c, "Creatures/D.psd": d}
    tree, blobs = tree_and_blobs(edits({"A": {"hp": 5}}), psds)
    before, _ = tree_and_blobs(edits({}), {"a": a, "Creatures/Kirby/3 Stars/A.psd": b"psd-a0", "Creatures/D.psd": d0})
    blobs[_sha(d0)] = d0
    files = [{"filename": "Creatures/D.psd", "status": "modified", "sha": _sha(d)},
             {"filename": "Creatures/Kirby/3 Stars/A.psd", "status": "modified", "sha": _sha(a)},
             {"filename": "Items/Heal/1 Stars/B.psd", "status": "added", "sha": _sha(b)},
             {"filename": "N.M.E/C.psd", "status": "renamed", "sha": _sha(c)},
             {"filename": "MDW/Templates/T.psd", "status": "modified", "sha": _sha(a)},
             {"filename": "Auxiliary/Markers/M.psd", "status": "modified", "sha": _sha(a)},
             {"filename": "Items/Gone.psd", "status": "removed", "sha": "0" * 40},
             {"filename": "site/Card Edits.json", "status": "modified", "sha": "4" * 40}]
    commits = {"c1": {"sha": "c1", "author": {"login": "alice"}, "files": [files[0], files[1]]},
               "c2": {"sha": "c2", "author": None, "files": [files[2], files[3], files[7]]}}   # no GitHub account on the email
    fake = FakeGitHub(blobs, {"commits": [commits["c1"], commits["c2"]], "files": files}, commits, tree, {"b" * 40: before})
    fake.broken.add(_sha(c))
    monkeypatch.setattr(ga, "get", fake.get)
    out, disc = tmp_path / "r.json", tmp_path / "d.json"
    assert cc.main(["--repo", "o/r", "--before", "b" * 40, "--after", "c2", "--author", "pusher", "--json", str(out), "--discord", str(disc)]) == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["cards"] == [
        {"path": "Creatures/D.psd", "author": "alice", "notes": ["DUPLICATE LAYER IDS", "MISSPELT TYPE: fier"], "new": ["MISSPELT TYPE: fier"]},
        {"path": "Items/Heal/1 Stars/B.psd", "author": "pusher", "notes": [NO_STARS], "new": [NO_STARS]}]
    assert data["checked"] == 3 and data["errors"] == [f"N.M.E/C.psd: blob {_sha(c)[:8]} hashes to {_sha(b'garbage')[:8]}"]
    assert fake.blob_calls() == 7                                   # the manifest, the edits file, four card PSDs, D's old one
    assert any(u.endswith("/git/trees/c2?recursive=1") for u in fake.calls) and any(f"/git/trees/{'b' * 40}?" in u for u in fake.calls)
    text = capsys.readouterr().out
    assert '- `Items/Heal/1 Stars/B.psd` (pusher): no stars (group them in "Stars")\n' in text and "A.psd" not in text
    assert "- `Creatures/D.psd` (alice): duplicate layer IDs; misspelt type fier\n" in text         # the comment: every note
    payload = json.loads(disc.read_text(encoding="utf-8"))
    assert payload["content"].splitlines() == ["alice pushed D: misspelt type fier. [c2](https://github.com/o/r/commit/c2)",
                                               'pusher pushed B: no stars (group them in "Stars"). [c2](https://github.com/o/r/commit/c2)']
    assert "token" not in payload["content"].lower()


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
# CLI only
# ========================