"""
commands/font_check.py offline: fixture fonts built with fontTools, type layers and PSDs as plain objects, a fake GitHub
for --repo mode, one real card PSD when the clone is there (the guards' checkout runs it: test_card_guards.py).
"""
import json
import shutil
import subprocess
from io import BytesIO
from sys import executable
from types import SimpleNamespace

import pytest
from fontTools.ttLib import TTCollection, TTFont
from requests import HTTPError

import commands.font_check as fc
import commands.github_api as ga
from conftest import REPO, Reply, at_rev, blob_sha_bytes as _sha, smallest
from helpers import ChecksGitHub as FakeGitHub, build_font, commit, git, put, sharing  # noqa: F401

BABA = "Creatures/Other/5 Stars/Baba.psd"                  # BabaUI on four visible digit layers and the hidden Name
BABA_REV = "3a268cad"                                       # Baba as this test was written, from the clone's history
BABA_BLOB = "50205a1c46a2e5403a54118ede96b18bc3902f7f"
ROLE = "546427268751294475"
FONTS = ["BabaUI", "AdobeInvisFont", "MyriadPro-Regular"]   # a card's usual FontSet: the face, Photoshop's two extras


# ========================
# Fixture fonts (helpers.build_font)
# ========================
@pytest.fixture(scope="session")
def fonts_dir(tmp_path_factory):
    """A TrueType, a CFF, a two-face .ttc, a text file and a subfolder (which stays out, like stoogedings-src/)."""
    d = tmp_path_factory.mktemp("fonts")
    parts = tmp_path_factory.mktemp("parts")
    build_font(str(d / "PsdToolsFixture-TT.ttf"), "PsdToolsFixture-TT", "PSD Tools Fixture", "TT")
    build_font(str(d / "PsdToolsFixture-CFF.otf"), "PsdToolsFixture-CFF", "PSD Tools Fixture", "CFF", cff=True)
    a = build_font(str(parts / "a.ttf"), "FixtureCollection-A", "Fixture Collection", "A")
    b = build_font(str(parts / "b.ttf"), "FixtureCollection-B", "Fixture Collection", "B")
    collection = TTCollection()
    collection.fonts = [TTFont(a), TTFont(b)]
    collection.save(str(d / "Pair.ttc"))
    (d / "README.txt").write_text("not a font")
    (d / "sub").mkdir()
    shutil.copy(d / "PsdToolsFixture-TT.ttf", d / "sub" / "Deep.ttf")
    return d


@pytest.fixture(scope="session")
def index(fonts_dir):
    return fc.FontIndex(str(fonts_dir))


# ========================
# Names and the index
# ========================
def test_index_reads_every_face(index):
    assert (index.files, index.faces, index.errors) == (3, 4, [])
    assert index.names == {"PsdToolsFixture-TT", "PsdToolsFixture-CFF", "FixtureCollection-A", "FixtureCollection-B"}
    assert index.supplies("FixtureCollection-B")                    # the .ttc's face at index 1


def test_face_names_and_fallbacks(fonts_dir):
    faces = fc.face_names((fonts_dir / "Pair.ttc").read_bytes())
    assert faces == ["FixtureCollection-A", "FixtureCollection-B"]
    font = TTFont(str(fonts_dir / "PsdToolsFixture-TT.ttf"))
    font["name"].removeNames(nameID=6)
    buf = BytesIO()
    font.save(buf)
    assert fc.face_names(buf.getvalue()) == ["PSDToolsFixture"]                 # no ID 6 or 4: the family, FreeType-style
    font["name"].setName("Full Name Here", 4, 3, 1, 0x409)
    buf = BytesIO()
    font.save(buf)
    assert fc.face_names(buf.getvalue()) == ["FullNameHere"]                    # ID 4 before ID 1, as FreeType reports it


def test_tables_sharing_data_get_their_own_copies(fonts_dir):
    """EarthboundZero's flaw (Photoshop 2023 set no type in it, TC ce778210): two records on one table's bytes. Each table
    gets its own copy, every table's bytes and the names as they were, the sums right; a .ttc's faces may share."""
    raw = (fonts_dir / "PsdToolsFixture-TT.ttf").read_bytes()
    bad = sharing(raw, b"Xtra", b"glyf")
    assert fc.shared_tables(raw) == [] and fc.shared_tables(bad) == [("Xtra", "glyf")]
    longer = bytearray(raw)                     # glyf's record past its bytes, over the next table's (Goudy-Bold's)
    at = next(12 + 16 * i for i in range(int.from_bytes(raw[4:6])) if raw[12 + 16 * i:16 + 16 * i] == b"glyf")
    longer[at + 12:at + 16] = (int.from_bytes(raw[at + 12:at + 16]) + 8).to_bytes(4)
    assert fc.shared_tables(bytes(longer)) == []
    fixed = fc.split_tables(bad)
    assert fc.shared_tables(fixed) == [] and fc.split_tables(fixed) == fixed and fc._sum(fixed) == fc.SFNT_SUM
    old, new = TTFont(BytesIO(bad)), TTFont(BytesIO(fixed), checkChecksums=2)

    def body(font, tag):
        data = font.reader[tag]
        return data[:8] + data[12:] if tag == "head" else data
    assert sorted(new.reader.keys()) == sorted(old.reader.keys())
    assert all(body(new, tag) == body(old, tag) for tag in old.reader.keys())
    assert fc.face_names(fixed) == ["PsdToolsFixture-TT"]
    pair = BytesIO()
    TTCollection(str(fonts_dir / "Pair.ttc")).save(pair, shareTables=True)
    assert fc.shared_tables(pair.getvalue()) == [] and fc.split_tables(pair.getvalue()) is None
    assert fc.shared_tables(b"not a font") == []


# ========================
# Type layers and PSDs as plain objects
# ========================
class Val:
    def __init__(self, value):
        self.value = value


def layer(name, font_set, runs, text, normal_font=2, visible=True, kind="type"):
    """A type layer as font_check reads it: runs = [(length in UTF-16 units, Font index or None)]; None = no StyleRun."""
    resources = {"FontSet": [{"Name": Val(f)} for f in font_set], "TheNormalStyleSheet": Val(0),
                 "StyleSheetSet": [{"StyleSheetData": {"Font": Val(normal_font)}}]}
    engine = {"Editor": {"Text": Val(text)}}
    if runs is not None:
        engine["StyleRun"] = {"RunArray": [{"StyleSheet": {"StyleSheetData": {} if f is None else {"Font": Val(f)}}} for _, f in runs],
                              "RunLengthArray": [Val(n) for n, _ in runs]}
    return SimpleNamespace(name=name, kind=kind, visible=visible, is_visible=lambda: visible,
                           resource_dict=resources, engine_dict=engine, text=text)


def psd(*layers):
    return SimpleNamespace(descendants=lambda: list(layers))


def test_layer_fonts_runs_and_normal_sheet():
    assert fc.layer_fonts(layer("6", FONTS, [(2, 0)], "6\r")) == ["BabaUI"]
    assert fc.layer_fonts(layer("x", FONTS, [(0, 1), (3, 0)], "abc")) == ["BabaUI"]           # a 0-length run names nothing
    assert fc.layer_fonts(layer("x", FONTS, [(3, None)], "abc")) == ["MyriadPro-Regular"]     # no Font in the sheet: the Normal sheet's
    assert fc.layer_fonts(layer("x", FONTS, [], "abc")) == ["MyriadPro-Regular"]              # no runs: the Normal sheet's
    assert fc.layer_fonts(layer("x", FONTS, None, "abc")) == ["MyriadPro-Regular"]
    assert fc.layer_fonts(layer("x", FONTS, [(1, 0)], "abc")) == ["BabaUI"]                  # text past the runs keeps the last style
    assert fc.layer_fonts(layer("x", FONTS, [(5, 0), (4, 2)], "abc")) == ["BabaUI"]          # a run past the text names nothing
    assert fc.layer_fonts(layer("x", FONTS, [(1, 0), (1, 2), (1, 0)], "abc")) == ["BabaUI", "MyriadPro-Regular"]   # first use, once
    assert fc.layer_fonts(layer("x", FONTS, [(3, 9)], "abc")) == ["BabaUI"]                   # out of range: FontSet's first
    assert fc.layer_fonts(layer("x", [], [(3, 0)], "abc")) == []
    assert fc.layer_fonts(layer("x", FONTS, [(2, 0), (1, 2)], "\U0001F600a")) == ["BabaUI", "MyriadPro-Regular"]   # UTF-16 units
    assert fc.layer_fonts(layer("x", FONTS, [], "")) == [] == fc.layer_fonts(layer("x", FONTS, [(0, 0)], ""))   # an empty layer asks for nothing
    assert fc.layer_fonts(layer("x", FONTS, [(1, 2), (3, 0)], "\ufeffabc")) == ["BabaUI"]   # a run over only the byte-order mark asks for nothing


def test_psd_fonts_visible_type_layers_only():
    doc = psd(layer("6", FONTS, [(2, 0)], "6\r"), layer("6", FONTS, [(2, 0)], "6\r"),
              layer("Name", FONTS, [(5, 0)], "Baba\r", visible=False),
              layer("Ability", ["Domigorgon", "AdobeInvisFont"], [(4, 0)], "text", normal_font=1),
              layer("art", FONTS, [(1, 0)], "x", kind="pixel"))
    assert fc.psd_fonts(doc) == {"BabaUI": ["6", "6"], "Domigorgon": ["Ability"]}


# ========================
# Reports
# ========================
def report() -> fc.Report:
    r = fc.Report(checked=3, fonts_files=270, fonts_faces=278, fonts_added=["Pristina.ttf"])
    r.cards = [fc.Card("Creatures/Other/5 Stars/Baba.psd", "bobber", {"BabaUI": ["6", "6", "6", "6"]}),
               fc.Card("Field/4 Stars/The_Gates_of_Ahn'Qiraj.psd", "michael", {"Pristina-Regular": ["Name"], "BabaUI": ["Ability"]}),
               fc.Card("N.M.E/Termination_Shock.psd", "bobber", {"OCRAExtended": ["Ability", "Name"]})]
    return r


def test_discord_payload_wording():
    """The role once on the first line, a line per card in the standards post's form (owner 2026-11-11), the owner's
    "bite this user" kept; a card's fonts its PSD before the push asked for too are not told."""
    payload = fc.discord_payload(report(), "o/r", "f" * 40)
    assert set(payload) == {"content", "allowed_mentions", "flags"} and payload["allowed_mentions"] == {"roles": [ROLE]}
    link = " [ffffffff](https://github.com/o/r/commit/ffffffffffff)"
    assert payload["content"] == (
        f"<@&{ROLE}>\n"
        "bobber pushed Baba: the BabaUI font is not in MDW/Fonts (6, 6, 6, 6), bite this user." + link + "\n"
        "michael pushed The Gates of Ahn'Qiraj: the Pristina-Regular font is not in MDW/Fonts (Name), bite this user; "
        "the BabaUI font is not in MDW/Fonts (Ability), bite this user." + link + "\n"
        "bobber pushed Termination Shock: the OCRAExtended font is not in MDW/Fonts (Ability, Name), bite this user." + link)
    assert "<@" not in payload["content"].replace(f"<@&{ROLE}>", "", 1)         # the role once, no user mentions
    assert fc.discord_payload(fc.Report(checked=2)) == {}
    old = report()
    for card in old.cards:
        card.new = []
    old.cards[1].new = ["BabaUI"]
    assert fc.discord_payload(old, "o/r", "f" * 40)["content"] == (
        f"<@&{ROLE}>\nmichael pushed The Gates of Ahn'Qiraj: the BabaUI font is not in MDW/Fonts (Ability), bite this user." + link)
    old.cards[1].new = []
    assert fc.discord_payload(old, "o/r", "f" * 40) == {}                         # nothing new: no ping


# ========================
# Command line
# ========================
@pytest.fixture
def fake_psds(monkeypatch):
    """psd_fonts answers from a table keyed by path or bytes, so no PSD file is needed."""
    table = {}

    def fake(psd):
        key = psd if isinstance(psd, str) else bytes(psd)
        if key not in table:
            raise ValueError("unreadable")
        return table[key]
    monkeypatch.setattr(fc, "psd_fonts", fake)
    return table


def test_cli_psd_mode(fonts_dir, tmp_path, fake_psds, capsys):
    fake_psds["Items/A.psd"] = {"PsdToolsFixture-TT": ["Name"], "JavaneseText": ["Ability"]}
    fake_psds["Items/B.psd"] = {"PsdToolsFixture-CFF": ["Name"]}
    out, disc = tmp_path / "r.json", tmp_path / "d.json"
    code = fc.main(["--fonts", str(fonts_dir), "--psd", "Items/A.psd", "Items/B.psd", "Items/C.psd", "--author", "bobber",
                    "--json", str(out), "--discord", str(disc)])
    assert code == 0                                                            # findings never fail the run
    text = capsys.readouterr().out
    assert text.startswith("## Font check\n") and "- `Items/A.psd` (bobber): **JavaneseText** (Ability)" in text
    assert "- Items/C.psd: unreadable" in text
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["checked"] == 2 and data["missing"] == {"JavaneseText": {"Items/A.psd": ["Ability"]}} and data["fonts"] == {"files": 3, "faces": 4}
    assert data["cards"] == [{"path": "Items/A.psd", "author": "bobber", "missing": {"JavaneseText": ["Ability"]}}]
    payload = json.loads(disc.read_text(encoding="utf-8"))
    assert payload["content"].splitlines()[1] == "bobber pushed A: the JavaneseText font is not in MDW/Fonts (Ability), bite this user."
    assert fc.main(["--fonts", str(fonts_dir), "--psd", "Items/B.psd", "--discord", str(disc)]) == 0
    assert json.loads(disc.read_text(encoding="utf-8")) == {}


def test_token_from_env(monkeypatch):
    monkeypatch.setattr(ga, "GIT_TOKEN", "")
    monkeypatch.setattr(ga, "HEADERS", {})
    monkeypatch.setenv("GITHUB_TOKEN", "ghs_x")
    ga.token_from_env()
    assert ga.HEADERS == {"Authorization": "token ghs_x"}
    monkeypatch.setattr(ga, "GIT_TOKEN", "mine")
    monkeypatch.setattr(ga, "HEADERS", {"Authorization": "token mine"})
    ga.token_from_env()
    assert ga.HEADERS == {"Authorization": "token mine"}                        # the bot's own token wins


# ========================
# --repo mode and the bot's cache against a fake GitHub
# ========================
def test_repo_mode_push(fonts_dir, fake_psds, monkeypatch, tmp_path, capsys):
    a, b, c, font = b"psd-a", b"psd-b", b"psd-c", (fonts_dir / "PsdToolsFixture-TT.ttf").read_bytes()
    fake_psds[a], fake_psds[b], fake_psds[c] = {"JavaneseText": ["Name"]}, {"PsdToolsFixture-TT": ["Ability"]}, {"BabaUI": ["6"]}
    files = [{"filename": "Creatures/Other/5 Stars/A.psd", "status": "modified", "sha": _sha(a)},
             {"filename": "Items/Heal/1 Stars/B.psd", "status": "added", "sha": _sha(b)},
             {"filename": "N.M.E/C.psd", "status": "renamed", "sha": _sha(c)},
             {"filename": "MDW/Templates/T.psd", "status": "modified", "sha": _sha(a)},
             {"filename": "Auxiliary/Markers/M.psd", "status": "modified", "sha": _sha(a)},
             {"filename": "Items/Gone.psd", "status": "removed", "sha": "0" * 40},
             {"filename": "Creatures/Other/5 Stars/A.png", "status": "modified", "sha": "1" * 40},
             {"filename": "MDW/Fonts/Pristina.ttf", "status": "added", "sha": _sha(font)},
             {"filename": "MDW/Fonts/stoogedings-src/x.ttf", "status": "added", "sha": _sha(font)},
             {"filename": "MDW/Fonts/Old.ttf", "status": "modified", "sha": _sha(font)}]
    commits = {"c1": {"sha": "c1", "author": {"login": "alice"}, "files": [files[0], files[7]]},
               "c2": {"sha": "c2", "author": None, "files": [files[1], files[2], files[0]]}}   # no GitHub account on the email
    a0 = b"psd-a0"                                                              # A before the push asked for it too
    fake_psds[a0] = {"JavaneseText": ["Ability"]}
    before = {"tree": [{"path": "Creatures/Other/5 Stars/A.psd", "type": "blob", "sha": _sha(a0)}]}
    fake = FakeGitHub({_sha(a): a, _sha(b): b, _sha(c): c, _sha(a0): a0}, {"commits": [commits["c1"], commits["c2"]], "files": files},
                      commits, None, {"b" * 40: before})
    fake.broken.add(_sha(c))
    monkeypatch.setattr(ga, "get", fake.get)
    out = tmp_path / "r.json"
    assert fc.main(["--fonts", str(fonts_dir), "--repo", "o/r", "--before", "b" * 40, "--after", "c2", "--author", "pusher",
                    "--json", str(out)]) == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["fonts_added"] == ["Pristina.ttf"]
    assert data["cards"] == [{"path": "Creatures/Other/5 Stars/A.psd", "author": "pusher", "missing": {"JavaneseText": ["Name"]},
                              "new": []}]
    assert data["checked"] == 2 and data["errors"] == [f"N.M.E/C.psd: blob {_sha(c)[:8]} hashes to {_sha(b'garbage')[:8]}"]
    assert fake.blob_calls() == 4                                               # the three card PSDs, once each; A's old one
    assert "**Fonts added to MDW/Fonts:** `Pristina.ttf`" in capsys.readouterr().out


def test_repo_mode_api_failure_is_one_line_and_exit_0(fonts_dir, monkeypatch, tmp_path, capsys):
    """A 5xx after the retries, no network or a truncated tree: one GitHub API error line, {} to Discord, exit 0 (never a red push)."""
    replies = {"503": lambda url: Reply(503), "truncated": lambda url: Reply(body={"truncated": True, "tree": []})}

    def connection_error(url):
        from requests import ConnectionError
        raise ConnectionError("no network")
    replies["offline"] = connection_error
    out, disc = tmp_path / "r.json", tmp_path / "d.json"
    for case, expected in (("503", "GitHub API: 503"), ("truncated", "GitHub API: the tree listing is truncated"), ("offline", "GitHub API: no network")):
        monkeypatch.setattr(ga, "get", lambda url, **kw: replies[case](url))
        args = ["--fonts", str(fonts_dir), "--repo", "o/r", "--before", "b" * 40, "--after", "c2", "--json", str(out), "--discord", str(disc)]
        assert fc.main(args + (["--whole-tree"] if case == "truncated" else [])) == 0
        text = capsys.readouterr().out
        assert "0 card PSDs read" in text and f"\nErrors:\n- {expected}" in text, (case, text)
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["cards"] == [] and data["errors"][0].startswith(expected)
        assert json.loads(disc.read_text(encoding="utf-8")) == {}


def test_force_push_falls_back_to_the_head_commit(monkeypatch):
    a = b"psd-a"
    head = {"sha": "h", "author": {"login": "alice"}, "files": [{"filename": "Items/A.psd", "status": "modified", "sha": _sha(a)}]}
    fake = FakeGitHub({}, None, {"h": head})
    monkeypatch.setattr(ga, "get", fake.get)
    assert fc.pushed_files("o/r", "f" * 40, "h", "pusher") == (
        {"Items/A.psd": (_sha(a), "alice")}, [], ["the previous head ffffffff is gone (a force push?); only the head commit's files were checked"])
    assert any("compare/" in u for u in fake.calls)
    fake = FakeGitHub({}, None, {})                                             # any other failure still raises
    monkeypatch.setattr(ga, "get", fake.get)
    with pytest.raises(HTTPError):
        fc.pushed_files("o/r", "f" * 40, "h")


def test_big_push_read_from_the_clone(fonts_dir, fake_psds, monkeypatch, tmp_path):
    """GitHub can't list a big push (the compare's 422; a commit listing no files): --clone's diff-tree does, every file
    credited to the pusher, and the PSDs are read from the clone. Without a clone the 422 still raises."""
    repo = tmp_path / "cards"
    repo.mkdir()
    git(repo, "init", "-q", "-b", "main")
    for rel, data in (("Items/A.psd", b"a1"), ("Items/Gone.psd", b"g"), ("MDW/Fonts/Old.ttf", b"o1")):
        put(repo, rel, data)
    first = commit(repo, "alice", 1000)
    for rel, data in (("Items/A.psd", b"a2"), ("Creatures/Other/5 Stars/B.psd", b"b"), ("MDW/Templates/T.psd", b"t"),
                      ("MDW/Fonts/Old.ttf", b"o2"), ("MDW/Fonts/New.ttf", b"n"), ("MDW/Fonts/stoogedings-src/x.ttf", b"x")):
        put(repo, rel, data)
    (repo / "Items/Gone.psd").unlink()
    after = commit(repo, "bob", 2000)
    fake = FakeGitHub({}, None, {after: {"sha": after, "author": {"login": "bob"}, "files": []}})   # (b) 200, no files
    monkeypatch.setattr(ga, "get", lambda url, **kw: Reply(422) if "/compare/" in url else fake.get(url))   # (a)
    note = "GitHub could not list this push's files; read from the clone, all credited to pusher"
    expected = ({"Items/A.psd": (_sha(b"a2"), "pusher"), "Creatures/Other/5 Stars/B.psd": (_sha(b"b"), "pusher")}, ["New.ttf"], [note])
    assert fc.pushed_files("o/r", first, after, "pusher", str(repo)) == expected
    assert fc.pushed_files("o/r", "", after, "pusher", str(repo)) == expected              # a dispatch: after^..after
    with pytest.raises(HTTPError):
        fc.pushed_files("o/r", first, after, "pusher")
    fake_psds[b"a2"], fake_psds[b"b"] = {"JavaneseText": ["Name"]}, {"PsdToolsFixture-TT": ["Ability"]}
    fake_psds[b"a1"] = {}
    out = tmp_path / "r.json"
    assert fc.main(["--fonts", str(fonts_dir), "--repo", "o/r", "--before", first, "--after", after, "--author", "pusher",
                    "--clone", str(repo), "--json", str(out)]) == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert (data["checked"], data["fonts_added"], data["errors"]) == (2, ["New.ttf"], [note])
    assert data["cards"] == [{"path": "Items/A.psd", "author": "pusher", "missing": {"JavaneseText": ["Name"]},
                              "new": ["JavaneseText"]}]                         # its old blob (the clone's) asked for none
    assert fake.blob_calls() == 0                                               # read from the clone, not the API


# ========================
# One real card
# ========================
def test_real_card_psd(clone, tmp_path):
    data = at_rev(clone, BABA_REV, BABA)
    assert _sha(data) == BABA_BLOB
    path = tmp_path / "Baba.psd"
    path.write_bytes(data)
    fonts = fc.psd_fonts(str(path))
    assert fonts == {"BabaUI": ["6", "6", "6", "6"], "Domigorgon": ["Ability"]}   # the hidden Name layer (BabaUI) is ignored
    assert fc.psd_fonts(data) == fonts
    index = fc.FontIndex(str(clone / "MDW" / "Fonts"))
    assert index.errors == [] and index.files >= 270 and index.supplies("Domigorgon")
    assert index.supplies("BabaUI")                             # Baba UI Regular.ttf, the owner's file (TTSCardMaker 92b5ae6b)
    assert all(index.supplies(font) for font in fonts)
    assert index.supplies("Papyrus-Regular")                    # Papyrus.ttf (the Windows face), no alias since 2026-10-23


def test_check_checkout_pool_matches_sequential(clone, tmp_path, monkeypatch):
    """--all from POOL_MIN PSDs parses in a process pool: the same report as the sequential path, in path order."""
    n = fc.POOL_MIN
    psds = smallest(clone, "Creatures/Other", n)
    root = tmp_path / "repo"
    for path in psds:
        link = root / path.relative_to(clone)
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(path)
    index = fc.FontIndex(str(clone / "MDW" / "Fonts"))
    pooled = fc.check_checkout(str(root), index)
    monkeypatch.setattr(fc, "POOL_MIN", n + 1)
    sequential = fc.check_checkout(str(root), index)
    assert pooled.as_dict() == sequential.as_dict() and pooled.checked + len(pooled.errors) - len(index.errors) == n


@pytest.mark.parametrize("module", ["commands.font_check", "commands.psd_edit"])
def test_freetype_loads_before_harfbuzz(clone, module):
    """uharfbuzz loaded after the module (fontTools' table reads, the guards' icon fonts), then FreeType's autohinter on
    stoogedings: no segfault (macOS bound FreeType's own HarfBuzz to an earlier uharfbuzz: rc 139, 11-10)."""
    probe = (f"import sys, {module}; first = 'freetype' in sys.modules; import uharfbuzz, freetype; "
             "face = freetype.Face(sys.argv[1]); face.set_char_size(48 * 64); "
             "face.load_glyph(1, freetype.FT_LOAD_TARGET_LIGHT | freetype.FT_LOAD_NO_BITMAP); print(first)")
    font = clone / "MDW" / "Fonts" / "stoogedings.ttf"
    done = subprocess.run([executable, "-c", probe, str(font)], cwd=REPO, capture_output=True, text=True, timeout=120)
    assert (done.returncode, done.stdout.strip()) == (0, "True"), done.stderr[-2000:]
