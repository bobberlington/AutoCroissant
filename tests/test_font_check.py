"""
commands/font_check.py offline: fixture fonts built with fontTools, type layers and PSDs as plain objects, a fake GitHub
for --repo mode and the bot's cache, the Action's sparse copy (--help without config.py), one real card PSD when the
clone is there.
"""
import json
import re
import shutil
from io import BytesIO
from pathlib import Path
from subprocess import run as run_process
from sys import executable
from types import SimpleNamespace

import pytest
from requests import HTTPError

import commands.font_check as fc
import commands.github_api as ga
import global_config
from conftest import CLONE, REPO, Reply, blob_sha, blob_sha_bytes as _sha

fontBuilder = pytest.importorskip("fontTools.fontBuilder")
from fontTools.ttLib import TTCollection, TTFont  # noqa: E402

WORKFLOW = CLONE / ".github" / "workflows" / "font-check.yml"
BABA = "Creatures/Other/5 Stars/Baba.psd"                  # BabaUI on four visible digit layers and the hidden Name
BABA_BLOB = "50205a1c46a2e5403a54118ede96b18bc3902f7f"
ROLE = "546427268751294475"
FONTS = ["BabaUI", "AdobeInvisFont", "MyriadPro-Regular"]   # a card's usual FontSet: the face, Photoshop's two extras


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
def test_norm_and_loose_keys():
    assert fc.norm("Kung Fu Panda") == "kungfupanda" == fc.norm("Kung-Fu-Panda")
    assert fc.loose_keys("Papyrus-Regular") == ("papyrusregular", "papyrus")
    assert fc.loose_keys("PristinaRegular") == ("pristinaregular", "pristina")
    assert fc.loose_keys("Elephant") == ("elephant",)


def test_index_reads_every_face(index):
    assert (index.files, index.faces, index.errors) == (3, 4, [])
    assert index.exact == {"PsdToolsFixture-TT", "PsdToolsFixture-CFF", "FixtureCollection-A", "FixtureCollection-B"}
    assert index.resolve("FixtureCollection-B") == "exact"          # the .ttc's face at index 1
    assert {"psdtoolsfixturett", "psdtoolsfixturecff", "fixturecollectionb"} <= index.loose


def test_face_names_and_fallbacks(fonts_dir):
    faces = fc.face_names((fonts_dir / "Pair.ttc").read_bytes())
    assert faces == [["FixtureCollection-A", ["FixtureCollection-A", "Fixture CollectionA"]],
                     ["FixtureCollection-B", ["FixtureCollection-B", "Fixture CollectionB"]]]
    font = TTFont(str(fonts_dir / "PsdToolsFixture-TT.ttf"))
    font["name"].removeNames(nameID=6)
    buf = BytesIO()
    font.save(buf)
    assert fc.face_names(buf.getvalue())[0][0] == "PSDToolsFixture"             # no ID 6 or 4: the family, FreeType-style
    font["name"].setName("Full Name Here", 4, 3, 1, 0x409)
    buf = BytesIO()
    font.save(buf)
    assert fc.face_names(buf.getvalue())[0][0] == "FullNameHere"                # ID 4 before ID 1, as FreeType reports it


def test_ps_name_is_freetypes(tmp_path):
    """FreeType keeps ASCII 33-126 minus []{}()<>/% (measured 2026-10-21 against freetype-py); the fork resolves with its names."""
    assert fc.ps_name("Kung Fu Panda") == "KungFuPanda" and fc.ps_name("A B-c_d.e+f~g") == "AB-c_d.e+f~g"
    assert fc.ps_name("x[y](z){w}<v>/u%t") == "xyzwvut" and fc.ps_name("Caf\u00e9 \u00df") == "Caf"
    assert fc.ps_name("#$&'*,:;=?@^`|") == "#$&'*,:;=?@^`|"
    path = build_font(str(tmp_path / "kfp.ttf"), "Kung Fu Panda", "Kung Fu Panda", "Regular")
    assert fc.face_names(Path(path).read_bytes()) == [["KungFuPanda", ["KungFuPanda", "Kung Fu Panda", "Kung Fu PandaRegular"]]]
    index = fc.FontIndex(str(tmp_path))
    assert index.resolve("KungFuPanda") == "exact" and index.resolve("Kung-Fu-Panda") == "alias"   # the FONT_ALIASES target is exact


def test_resolve_loose_regular_and_alias(index, monkeypatch):
    assert index.resolve("PsdToolsFixture-TT") == "exact"
    assert index.resolve("PsdToolsFixture-TT-Regular") == "loose"              # -Regular stripped
    assert index.resolve("psd tools fixture cff") == "loose"                    # family + style, any case or punctuation
    assert index.resolve("JavaneseText") is None
    monkeypatch.setattr(fc, "FONT_ALIASES", {"Windows-Name": "PsdToolsFixture-CFF", "Dead-Alias": "Nothing"})
    assert index.resolve("Windows-Name") == "alias" and index.resolve("Dead-Alias") is None


def test_alias_table_is_shared():
    import commands.psd_edit as pe
    assert fc.FONT_ALIASES is global_config.FONT_ALIASES is pe.FONT_ALIASES
    assert global_config.FONT_ALIASES == {"Papyrus-Regular": "Papyrus", "Kung-Fu-Panda": "KungFuPanda"}


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


def test_check_psd(index):
    doc = psd(layer("Name", ["PsdToolsFixture-TT", "JavaneseText"], [(2, 0), (2, 1)], "abcd"), layer("Ability", ["BabaUI"], [(1, 0)], "x"))
    assert fc.check_psd(doc, index) == {"JavaneseText": ["Name"], "BabaUI": ["Ability"]}


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
    r = report()
    r.remaining = 613
    assert "Warming the cache: 613 card PSDs are not read yet" in fc.discord_text(r)
    assert fc.discord_text(fc.Report(checked=5)).endswith("Every font the cards ask for is in MDW/Fonts.")
    errors = fc.discord_text(fc.Report(errors=[f"Items/{i}.psd: 403" for i in range(25)])).splitlines()[2]   # a rate-limited run
    assert errors == "Errors: " + "; ".join(f"Items/{i}.psd: 403" for i in range(10)) + "; and 15 more (in the console)"
    assert fc.discord_text(fc.Report(errors=["a: x"])).endswith("\nErrors: a: x")
    withheld = fc.discord_text(fc.Report(checked=5, fonts_unread=2))
    assert "2 MDW/Fonts files failed to download, so no card is reported missing a font this time" in withheld
    assert "Every font" not in withheld


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
class FakeGitHub:
    """blobs: sha -> bytes; compare: the compare API's body; commits: sha -> the commit API's body; tree: a trees body."""
    def __init__(self, blobs: dict, compare: dict, commits: dict, tree: dict = None):
        self.blobs, self.compare, self.commits, self.tree = blobs, compare, commits, tree
        self.broken: set = set()                 # blobs served with other bytes
        self.calls: list = []

    def get(self, url, headers=None, timeout=None, stream=False):
        self.calls.append(url)
        path = url.split("/repos/", 1)[1].split("/", 2)[2]
        if path.startswith("compare/"):
            return Reply(body=self.compare) if self.compare is not None else Reply(404)   # None: the old head is gone
        if path.startswith("commits/"):
            sha = path.split("/")[1]
            return Reply(body=self.commits[sha]) if sha in self.commits else Reply(404)
        if path.startswith("git/trees/"):
            return Reply(body=self.tree)
        if path.startswith("git/blobs/"):
            sha = path.rsplit("/", 1)[1]
            if sha in self.broken:
                return Reply(data=b"garbage")
            return Reply(data=self.blobs[sha]) if sha in self.blobs else Reply(404)
        return Reply(404)

    def blob_calls(self) -> int:
        return sum("/git/blobs/" in u for u in self.calls)


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


def test_repo_fonts_cache_and_batches(fonts_dir, fake_psds, monkeypatch, tmp_path):
    font, ttc = (fonts_dir / "PsdToolsFixture-TT.ttf").read_bytes(), (fonts_dir / "Pair.ttc").read_bytes()
    a, b = b"psd-a", b"psd-b"
    fake_psds[a], fake_psds[b] = {"JavaneseText": ["Name"]}, {"PsdToolsFixture-TT": ["Ability"]}
    tree = {"tree": [{"path": "MDW/Fonts/TT.ttf", "type": "blob", "sha": _sha(font)}, {"path": "MDW/Fonts/Pair.ttc", "type": "blob", "sha": _sha(ttc)},
                     {"path": "MDW/Fonts/stoogedings-src/x.ttf", "type": "blob", "sha": _sha(b"nope")},
                     {"path": "MDW/Fonts/readme.txt", "type": "blob", "sha": _sha(b"nope")},
                     {"path": "Creatures/A.psd", "type": "blob", "sha": _sha(a)}, {"path": "Items/B.psd", "type": "blob", "sha": _sha(b)},
                     {"path": "MDW/T.psd", "type": "blob", "sha": _sha(a)}]}
    fake = FakeGitHub({_sha(font): font, _sha(ttc): ttc, _sha(a): a, _sha(b): b}, {}, {}, tree)
    monkeypatch.setattr(ga, "get", fake.get)
    cache = tmp_path / "font_names.json"
    first = fc.repo_fonts("o/r", str(cache), batch=1)
    assert (first.fonts_files, first.fonts_faces, first.checked, first.remaining, first.errors) == (2, 3, 1, 1, [])
    assert [c.path for c in first.cards] == ["Creatures/A.psd"] and "Warming the cache" in fc.discord_text(first)
    saved = json.loads(cache.read_text(encoding="utf-8"))
    assert set(saved["fonts"]) == {_sha(font), _sha(ttc)} and saved["psds"] == {_sha(a): {"JavaneseText": ["Name"]}}
    n = fake.blob_calls()                                                       # 2 fonts + 1 PSD
    second = fc.repo_fonts("o/r", str(cache), batch=1)
    assert (second.checked, second.remaining) == (2, 0) and second.cards[0].missing == {"JavaneseText": ["Name"]}
    assert fake.blob_calls() == n + 1                                           # only B was new
    assert fc.repo_fonts("o/r", str(cache), batch=1).checked == 2 and fake.blob_calls() == n + 1
    tree["tree"] = [e for e in tree["tree"] if e["path"] != "Items/B.psd"]
    assert fc.repo_fonts("o/r", str(cache), batch=1).checked == 1
    assert set(json.loads(cache.read_text(encoding="utf-8"))["psds"]) == {_sha(a)}   # a blob that left the tree leaves the cache
    assert not (tmp_path / "font_names.json.tmp").exists()


def test_repo_fonts_font_download_failure_reports_no_card(fonts_dir, fake_psds, monkeypatch, tmp_path):
    """A font that failed to download may be the one a card asks for: no card is reported, the PSDs still land in the cache."""
    font, ttc, a = (fonts_dir / "PsdToolsFixture-TT.ttf").read_bytes(), (fonts_dir / "Pair.ttc").read_bytes(), b"psd-a"
    fake_psds[a] = {"FixtureCollection-B": ["Name"]}                            # a face of Pair.ttc
    tree = {"tree": [{"path": "MDW/Fonts/TT.ttf", "type": "blob", "sha": _sha(font)}, {"path": "MDW/Fonts/Pair.ttc", "type": "blob", "sha": _sha(ttc)},
                     {"path": "Creatures/A.psd", "type": "blob", "sha": _sha(a)}]}
    fake = FakeGitHub({_sha(font): font, _sha(a): a}, {}, {}, tree)            # Pair.ttc answers 404
    monkeypatch.setattr(ga, "get", fake.get)
    cache = tmp_path / "font_names.json"
    first = fc.repo_fonts("o/r", str(cache))
    assert (first.checked, first.cards, first.fonts_unread, first.errors) == (1, [], 1, ["Pair.ttc: 404 Error"])
    saved = json.loads(cache.read_text(encoding="utf-8"))
    assert set(saved["fonts"]) == {_sha(font)} and set(saved["psds"]) == {_sha(a)}   # the failed font is not cached
    fake.blobs[_sha(ttc)] = ttc
    second = fc.repo_fonts("o/r", str(cache))
    assert (second.checked, second.cards, second.fonts_unread, second.errors) == (1, [], 0, [])
    tree["tree"].append({"path": "MDW/Fonts/New.ttf", "type": "blob", "sha": _sha(b"new")})   # 404 again
    tree["tree"][2]["sha"] = _sha(b"psd-b")
    fake.blobs[_sha(b"psd-b")], fake_psds[b"psd-b"] = b"psd-b", {"BabaUI": ["Ability"]}
    third = fc.repo_fonts("o/r", str(cache))
    assert (third.checked, third.cards, third.fonts_unread) == (1, [], 1)       # BabaUI is missing, but New.ttf might supply it


def test_repo_fonts_caches_unreadable_blobs(fonts_dir, fake_psds, monkeypatch, tmp_path):
    """A PSD or font that downloads but can't be read is cached as its error: reported every run, downloaded once."""
    font, bad, a = (fonts_dir / "PsdToolsFixture-TT.ttf").read_bytes(), b"not a font", b"psd-a"
    tree = {"tree": [{"path": "MDW/Fonts/TT.ttf", "type": "blob", "sha": _sha(font)}, {"path": "MDW/Fonts/Bad.ttf", "type": "blob", "sha": _sha(bad)},
                     {"path": "Creatures/A.psd", "type": "blob", "sha": _sha(a)}]}                       # fake_psds can't read psd-a
    fake = FakeGitHub({_sha(font): font, _sha(bad): bad, _sha(a): a}, {}, {}, tree)
    monkeypatch.setattr(ga, "get", fake.get)
    cache = tmp_path / "font_names.json"
    first = fc.repo_fonts("o/r", str(cache))
    assert (first.checked, first.fonts_files, first.fonts_unread) == (0, 1, 0)  # unreadable is not unread: verdicts stand
    assert [e.split(":")[0] for e in first.errors] == ["Bad.ttf", "Creatures/A.psd"] and first.errors[1] == "Creatures/A.psd: unreadable"
    n = fake.blob_calls()
    assert fc.repo_fonts("o/r", str(cache)).errors == first.errors and fake.blob_calls() == n
    saved = json.loads(cache.read_text(encoding="utf-8"))
    assert saved["psds"] == {_sha(a): "unreadable"} and isinstance(saved["fonts"][_sha(bad)], str)


def test_save_cache_failure_keeps_the_old_file(tmp_path):
    cache = tmp_path / "font_names.json"
    fc._save_cache(str(cache), {"fonts": {}, "psds": {}})
    with pytest.raises(TypeError):
        fc._save_cache(str(cache), {"fonts": {}, "psds": {"x": object()}})     # json can't write it, midway through
    assert fc._load_cache(str(cache)) == {"fonts": {}, "psds": {}} and not (tmp_path / "font_names.json.tmp").exists()


def test_check_repo_fonts_replies(monkeypatch):
    import commands.query_card as qc
    import commands.utils as utils
    sent, seen = [], []
    monkeypatch.setattr(utils, "queue_long_message", lambda interaction, text, **kw: sent.append(text))
    monkeypatch.setattr(fc, "repo_fonts", lambda repository: (seen.append(repository), fc.Report(checked=3, fonts_files=2, fonts_faces=2))[1])
    fc.check_repo_fonts("interaction")
    assert seen == [qc.card_repo.repository]
    assert sent == ["MDW/Fonts: 2 files, 2 faces. 3 card PSDs read.\nEvery font the cards ask for is in MDW/Fonts."]

    def boom(repository):
        raise RuntimeError("tree 503")
    monkeypatch.setattr(fc, "repo_fonts", boom)
    fc.check_repo_fonts("interaction")
    assert sent[1] == "Font check failed: tree 503"
    assert fc._lock.acquire(blocking=False)
    try:
        fc.check_repo_fonts("interaction")
    finally:
        fc._lock.release()
    assert sent[2].startswith("A font check is already running")


def test_help_entry():
    import commands.help as help_module
    entry = help_module.stats_commands["/fonts:"]
    assert "MDW/Fonts" in entry and "(admin only)" not in entry


# ========================
# One real card and the Action's sparse copy
# ========================
def test_real_card_psd(clone):
    path = clone / BABA
    if not path.exists() or blob_sha(path) != BABA_BLOB:
        pytest.skip(f"{BABA} changed upstream since this test was written")
    fonts = fc.psd_fonts(str(path))
    assert fonts == {"BabaUI": ["6", "6", "6", "6"], "Domigorgon": ["Ability"]}   # the hidden Name layer (BabaUI) is ignored
    assert fc.psd_fonts(path.read_bytes()) == fonts
    index = fc.FontIndex(str(clone / "MDW" / "Fonts"))
    assert index.errors == [] and index.files >= 270 and index.resolve("Domigorgon") == "exact"
    assert fc.check_psd(str(path), index) == {"BabaUI": ["6", "6", "6", "6"]}
    assert index.resolve("Papyrus-Regular") == "alias" == index.resolve("Kung-Fu-Panda")   # both FONT_ALIASES targets are exact faces


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


def _bot_files_in_workflow() -> list:
    """The AutoCroissant paths font-check.yml checks out (its second checkout step)."""
    if not WORKFLOW.exists():
        pytest.skip(f"no {WORKFLOW}")
    text = WORKFLOW.read_text(encoding="utf-8")
    block = text.split("repository: bobberlington/AutoCroissant", 1)[1].split("sparse-checkout: |", 1)[1]
    paths = []
    for line in block.splitlines()[1:]:
        if not line.startswith(" " * 12) or not line.strip():
            break
        paths.append(line.strip().lstrip("/"))
    return paths


def test_action_shaped_copy_imports_and_prints_help(tmp_path):
    """Exactly the yaml's files, no config.py: `python -m commands.font_check --help` works and imports no Discord."""
    paths = _bot_files_in_workflow()
    assert paths == ["commands/font_check.py", "commands/github_api.py", "global_config.py"]
    bot = tmp_path / "bot"
    for rel in paths:
        assert (REPO / rel).is_file(), f"{rel} is checked out by the Action but missing here"
        (bot / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO / rel, bot / rel)
    done = run_process([executable, "-m", "commands.font_check", "--help"], cwd=bot, capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    assert done.stdout.startswith("usage: python -m commands.font_check")
    assert set(re.findall(r"--[a-z-]+", done.stdout)) == {"--fonts", "--psd", "--all", "--repo", "--before", "--after", "--author",
                                                          "--whole-tree", "--json", "--discord", "--help"}
    code = ("import sys, commands.font_check; "
            "print(sorted(m for m in ('config', 'discord', 'commands.query_card', 'commands.utils') if m in sys.modules))")
    done = run_process([executable, "-c", code], cwd=bot, capture_output=True, text=True, timeout=120)
    assert done.returncode == 0 and done.stdout.strip() == "[]", done.stderr
    from test_card_bake import ALLOWED_IMPORTS, third_party_imports
    extra = third_party_imports("import commands.font_check", bot) - ALLOWED_IMPORTS
    assert not extra, f"font-check.yml installs none of these: {sorted(extra)}"


def test_workflow_shape():
    if not WORKFLOW.exists():
        pytest.skip(f"no {WORKFLOW}")
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "discord.com/api/webhooks" not in text and "secrets.DISCORD_FONT_WEBHOOK" in text and "<@&" not in text   # no URL, no mention in the yaml
    assert '"**/*.psd"' in text and '"MDW/Fonts/**"' in text and "workflow_dispatch" in text
    assert '--repo "$GITHUB_REPOSITORY"' in text and '--after "$GITHUB_SHA"' in text and "--whole-tree" in text and '--before "$BEFORE"' in text
    assert "contents: write" in text and '|| echo "::warning::' in text and "actions/checkout@v7" in text and "actions/cache@v6" in text
    yaml = pytest.importorskip("yaml")
    doc = yaml.safe_load(text)
    assert doc[True]["push"]["paths"] == ["**/*.psd", "MDW/Fonts/**"]          # PyYAML reads the `on` key as True
    assert doc[True]["workflow_dispatch"]["inputs"]["all"]["default"] is False
    assert doc["permissions"] == {"contents": "write"}
    names = [step.get("name", "") for step in doc["jobs"]["check"]["steps"]]
    assert names.index("Tell Discord") < names.index("Comment on the commit")   # the Discord message never waits on the comment
