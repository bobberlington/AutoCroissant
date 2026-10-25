"""
commands/font_check.py offline: fixture fonts built with fontTools, type layers and PSDs as plain objects, a fake GitHub
for --repo mode, one real card PSD when the clone is there (the guards' checkout runs it: test_card_guards.py).
"""
import json
import shutil
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest
from requests import HTTPError

import commands.font_check as fc
import commands.github_api as ga
import global_config
from conftest import REPO, Reply, at_rev, blob_sha_bytes as _sha

fontBuilder = pytest.importorskip("fontTools.fontBuilder")
from fontTools.ttLib import TTCollection, TTFont  # noqa: E402
from helpers import ChecksGitHub as FakeGitHub  # noqa: E402,F401

BABA = "Creatures/Other/5 Stars/Baba.psd"                  # BabaUI on four visible digit layers and the hidden Name
BABA_REV = "3a268cad"                                       # Baba as this test was written, from the clone's history
BABA_BLOB = "50205a1c46a2e5403a54118ede96b18bc3902f7f"
ROLE = "546427268751294475"
FONTS = ["BabaUI", "AdobeInvisFont", "MyriadPro-Regular"]   # a card's usual FontSet: the face, Photoshop's two extras


@pytest.fixture(autouse=True)
def no_local_fonts(monkeypatch, tmp_path):
    """No PSD_TOOLS_FONT_DIRS and no checkout: /fonts reads fonts through the fake GitHub only."""
    monkeypatch.delenv(fc.FONT_DIRS_ENV, raising=False)
    monkeypatch.setattr(fc, "LOCAL_DIR_LOC", str(tmp_path / "no-checkout"))


# ========================
# Fixture fonts (the fork's tests/psd_tools/composite/text/conftest.py recipe, cut down)
# ========================
def build_font(path: str, ps: str, family: str, style: str, cff: bool = False) -> str:
    names = [".notdef", "space", "A"]
    fb = fontBuilder.FontBuilder(1000, isTTF=not cff)
    fb.setupGlyphOrder(names)
    fb.setupCharacterMap({32: "space", 65: "A"})
    if cff:
        from fontTools.pens.t2CharStringPen import T2CharStringPen
        charstrings = {}
        for g in names:
            pen = T2CharStringPen(500, None)
            if g == "A":
                pen.moveTo((0, 0)), pen.lineTo((0, 700)), pen.lineTo((500, 700)), pen.lineTo((500, 0)), pen.closePath()
            charstrings[g] = pen.getCharString()
        fb.setupCFF(ps, {"FullName": ps}, charstrings, {})
    else:
        from fontTools.pens.ttGlyphPen import TTGlyphPen
        glyphs = {}
        for g in names:
            pen = TTGlyphPen(None)
            if g == "A":
                pen.moveTo((0, 0)), pen.lineTo((0, 700)), pen.lineTo((500, 700)), pen.lineTo((500, 0)), pen.closePath()
            glyphs[g] = pen.glyph()
        fb.setupGlyf(glyphs)
    fb.setupHorizontalMetrics({g: (500, 0) for g in names})
    fb.setupHorizontalHeader(ascent=800, descent=-200)
    fb.setupNameTable({"familyName": family, "styleName": style, "psName": ps})
    fb.setupOS2(sTypoAscender=800, sTypoDescender=-200, usWinAscent=800, usWinDescent=200)
    fb.setupPost()
    fb.save(path)
    return path


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


def test_ps_name_is_freetypes(tmp_path):
    """FreeType keeps ASCII 33-126 minus []{}()<>/% (measured 2026-10-21 against freetype-py); the fork resolves with its names."""
    assert fc.ps_name("Kung Fu Panda") == "KungFuPanda" and fc.ps_name("A B-c_d.e+f~g") == "AB-c_d.e+f~g"
    assert fc.ps_name("x[y](z){w}<v>/u%t") == "xyzwvut" and fc.ps_name("Caf\u00e9 \u00df") == "Caf"
    assert fc.ps_name("#$&'*,:;=?@^`|") == "#$&'*,:;=?@^`|"
    path = build_font(str(tmp_path / "kfp.ttf"), "Kung Fu Panda", "Kung Fu Panda", "Regular")
    assert fc.face_names(Path(path).read_bytes()) == ["KungFuPanda"]
    index = fc.FontIndex(str(tmp_path))
    assert index.supplies("KungFuPanda") and not index.supplies("Kung-Fu-Panda")   # no aliases (owner, 2026-10-23)


def test_supplies_is_an_exact_name(index):
    """The face's PostScript name exactly: no loose match or alias since the owner's "change the fonts to the exact
    ones and delete the aliases" (2026-10-23)."""
    assert index.supplies("PsdToolsFixture-TT")
    assert not index.supplies("PsdToolsFixture-TT-Regular")         # loosely, -Regular stripped, until 2026-10-23
    assert not index.supplies("psd tools fixture cff")              # loosely, family + style, until 2026-10-23
    assert not index.supplies("psdtoolsfixture-tt") and not index.supplies("JavaneseText")
    assert not hasattr(global_config, "FONT_ALIASES")


def test_unreadable_font_is_an_error_not_a_crash(tmp_path):
    (tmp_path / "Bad.ttf").write_bytes(b"not a font at all")
    shutil.copy(REPO / "global_config.py", tmp_path / "Also.otf")
    index = fc.FontIndex(str(tmp_path))
    assert (index.files, index.faces) == (0, 0) and sorted(e.split(":")[0] for e in index.errors) == ["Also.otf", "Bad.ttf"]


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


def test_markdown_report():
    text = fc.markdown(report())
    assert text.startswith("## Font check\n\n3 card PSDs read; MDW/Fonts: 270 files, 278 faces.\n")
    assert "**Fonts added to MDW/Fonts:** `Pristina.ttf`" in text
    assert "- `Creatures/Other/5 Stars/Baba.psd` (bobber): **BabaUI** (6, 6, 6, 6)\n" in text
    assert "- `Field/4 Stars/The_Gates_of_Ahn'Qiraj.psd` (michael): **Pristina-Regular** (Name); **BabaUI** (Ability)\n" in text
    empty = fc.markdown(fc.Report(checked=1, fonts_files=2, fonts_faces=2, errors=["x.psd: boom"]))
    assert "1 card PSD read" in empty and "Every font the cards ask for is in MDW/Fonts." in empty and "- x.psd: boom" in empty
    assert "Fonts added" not in empty
    assert report().as_dict()["missing"] == {"BabaUI": {"Creatures/Other/5 Stars/Baba.psd": ["6", "6", "6", "6"],
                                                        "Field/4 Stars/The_Gates_of_Ahn'Qiraj.psd": ["Ability"]},
                                             "OCRAExtended": {"N.M.E/Termination_Shock.psd": ["Ability", "Name"]},
                                             "Pristina-Regular": {"Field/4 Stars/The_Gates_of_Ahn'Qiraj.psd": ["Name"]}}


def test_discord_payload_wording():
    payload = fc.discord_payload(report())
    assert set(payload) == {"content", "allowed_mentions"} and payload["allowed_mentions"] == {"roles": [ROLE]}
    assert payload["content"] == (
        f"<@&{ROLE}>\n"
        "bobber has pushed a card without adding the BabaUI font to MDW/Fonts, bite this user.\n"
        "    Baba (6, 6, 6, 6)\n"
        "bobber has pushed a card without adding the OCRAExtended font to MDW/Fonts, bite this user.\n"
        "    Termination Shock (Ability, Name)\n"
        "michael has pushed a card without adding the BabaUI font to MDW/Fonts, bite this user.\n"
        "    The Gates of Ahn'Qiraj (Ability)\n"
        "michael has pushed a card without adding the Pristina-Regular font to MDW/Fonts, bite this user.\n"
        "    The Gates of Ahn'Qiraj (Name)")
    assert "<@" not in payload["content"].replace(f"<@&{ROLE}>", "", 1)         # the role once, no user mentions
    assert fc.discord_payload(fc.Report(checked=2)) == {}


def test_discord_payload_fits_one_message():
    r = fc.Report(cards=[fc.Card(f"Items/Heal/1 Stars/Card_{i}.psd", "bobber", {"BabaUI": ["Ability"]}) for i in range(200)])
    content = fc.discord_payload(r)["content"]
    assert len(content) <= 2000 and content.startswith(f"<@&{ROLE}>\n") and content.endswith("... (the rest is in the commit comment)")


def test_discord_text():
    text = fc.discord_text(report())
    assert text.startswith("MDW/Fonts: 270 files, 278 faces. 3 card PSDs read.\n"
                           "**BabaUI**: Baba (6, 6, 6, 6); The Gates of Ahn'Qiraj (Ability)\n")
    assert fc.discord_text(fc.Report(checked=5)).endswith("Every font the cards ask for is in MDW/Fonts.")
    errors = fc.discord_text(fc.Report(errors=[f"Items/{i}.psd: 403" for i in range(25)])).splitlines()[2]   # a rate-limited run
    assert errors == "Errors: " + "; ".join(f"Items/{i}.psd: 403" for i in range(10)) + "; and 15 more (in the console)"
    assert fc.discord_text(fc.Report(errors=["a: x"])).endswith("\nErrors: a: x")


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
    assert payload["content"].splitlines()[1] == "bobber has pushed a card without adding the JavaneseText font to MDW/Fonts, bite this user."
    assert fc.main(["--fonts", str(fonts_dir), "--psd", "Items/B.psd", "--discord", str(disc)]) == 0
    assert json.loads(disc.read_text(encoding="utf-8")) == {}


def test_cli_all_mode_skips_mdw_and_markers(fonts_dir, tmp_path, fake_psds, capsys):
    root = tmp_path / "repo"
    for rel in ("Creatures/Kirby/3 Stars/Pix.psd", "MDW/Templates/Creature_Template.psd", "Auxiliary/Markers/Marker.psd",
                "Items/note.txt", ".git/x.psd"):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(b"")
    fake_psds[str(root / "Creatures/Kirby/3 Stars/Pix.psd")] = {"BabaUI": ["Name"]}
    assert fc.main(["--fonts", str(fonts_dir), "--all", str(root)]) == 0
    text = capsys.readouterr().out
    assert "1 card PSD read" in text and "- `Creatures/Kirby/3 Stars/Pix.psd` (Someone): **BabaUI** (Name)" in text and "Errors" not in text


def test_cli_needs_fonts(capsys):
    with pytest.raises(SystemExit) as e:
        fc.main(["--psd", "x.psd"])
    assert e.value.code == 2 and "--fonts" in capsys.readouterr().err


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
    fake = FakeGitHub({_sha(a): a, _sha(b): b, _sha(c): c}, {"commits": [commits["c1"], commits["c2"]], "files": files}, commits)
    fake.broken.add(_sha(c))
    monkeypatch.setattr(ga, "get", fake.get)
    out = tmp_path / "r.json"
    assert fc.main(["--fonts", str(fonts_dir), "--repo", "o/r", "--before", "b" * 40, "--after", "c2", "--author", "pusher",
                    "--json", str(out)]) == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["fonts_added"] == ["Pristina.ttf"]
    assert data["cards"] == [{"path": "Creatures/Other/5 Stars/A.psd", "author": "pusher", "missing": {"JavaneseText": ["Name"]}}]
    assert data["checked"] == 2 and data["errors"] == [f"N.M.E/C.psd: blob {_sha(c)[:8]} hashes to {_sha(b'garbage')[:8]}"]
    assert fake.blob_calls() == 3                                               # the three card PSDs, once each
    assert "**Fonts added to MDW/Fonts:** `Pristina.ttf`" in capsys.readouterr().out


def test_repo_mode_api_failure_is_one_line_and_exit_0(fonts_dir, monkeypatch, tmp_path, capsys):
    """A 5xx after the retries, no network or a truncated tree: one GitHub API error line, {} to Discord, exit 0 (never a red push)."""
    monkeypatch.setattr(ga, "sleep", lambda s: None)                              # get_with_retry's waits
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


def test_repo_mode_single_commit_and_whole_tree(fonts_dir, fake_psds, monkeypatch, tmp_path):
    a = b"psd-a"
    fake_psds[a] = {"BabaUI": ["Name"]}
    head = {"sha": "h", "author": {"login": "alice"}, "files": [{"filename": "Items/A.psd", "status": "added", "sha": _sha(a)}]}
    tree = {"tree": [{"path": "Items/A.psd", "type": "blob", "sha": _sha(a)}, {"path": "MDW/Templates/T.psd", "type": "blob", "sha": _sha(a)},
                     {"path": "Items", "type": "tree", "sha": "t"}]}
    fake = FakeGitHub({_sha(a): a}, {}, {"h": head}, tree)
    monkeypatch.setattr(ga, "get", fake.get)
    expected = ({"Items/A.psd": (_sha(a), "alice")}, [], [])
    assert fc.pushed_files("o/r", "", "h", "pusher") == expected                # a dispatch: no before
    assert fc.pushed_files("o/r", "0" * 40, "h", "pusher") == expected          # a new branch
    assert not any("compare" in u for u in fake.calls)
    out = tmp_path / "r.json"
    assert fc.main(["--fonts", str(fonts_dir), "--repo", "o/r", "--after", "h", "--whole-tree", "--author", "pusher", "--json", str(out)]) == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["cards"] == [{"path": "Items/A.psd", "author": "pusher", "missing": {"BabaUI": ["Name"]}}] and data["checked"] == 1
    assert any(u.endswith("/git/trees/h?recursive=1") for u in fake.calls)


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


def test_pushed_files_notes_the_file_cap(monkeypatch):
    files = [{"filename": f"Items/{i}.png", "status": "modified", "sha": "0" * 40} for i in range(300)]
    fake = FakeGitHub({}, {"commits": [{"sha": "h", "author": {"login": "alice"}}], "files": files}, {})
    monkeypatch.setattr(ga, "get", fake.get)
    assert fc.pushed_files("o/r", "a" * 40, "h") == ({}, [], ["the push changed 300 files or more; only the first 300 were checked"])


def test_pushed_files_notes_the_commit_caps(monkeypatch):
    """A commit listing 300 files or more, or a push of 250 commits or more: later files may carry the wrong author, so a note says so."""
    files = [{"filename": f"Items/{i}.psd", "status": "modified", "sha": "0" * 40} for i in range(300)]
    c1 = {"sha": "c1", "author": {"login": "alice"}, "files": files[:1]}
    c2 = {"sha": "c2", "author": {"login": "bob"}, "files": files}
    fake = FakeGitHub({}, {"commits": [c1, c2], "files": files[:2]}, {"c1": c1, "c2": c2})
    monkeypatch.setattr(ga, "get", fake.get)
    psds, fonts, notes = fc.pushed_files("o/r", "a" * 40, "c2")
    assert notes == ["commit c2 lists 300 files or more; later files may be credited to the wrong author"]
    assert psds == {"Items/0.psd": ("0" * 40, "bob"), "Items/1.psd": ("0" * 40, "bob")} and fonts == []
    many = [{"sha": f"c{i}", "author": {"login": "alice"}} for i in range(250)]
    fake = FakeGitHub({}, {"commits": many, "files": files[:1]}, {c["sha"]: {**c, "files": []} for c in many})
    monkeypatch.setattr(ga, "get", fake.get)
    assert fc.pushed_files("o/r", "a" * 40, "c249")[2] == ["the push has 250 commits or more; later files may be credited to the wrong author"]


def test_help_entry():
    import commands.help as help_module
    entry = help_module.stats_commands["/fonts:"]
    assert "MDW/Fonts" in entry and "(admin only)" not in entry


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
    psds = sorted(clone.joinpath("Creatures", "Other").rglob("*.psd"), key=lambda p: p.stat().st_size)[:n]   # the smallest: a fast parse
    if len(psds) < n:
        pytest.skip("not enough card PSDs in the clone")
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


