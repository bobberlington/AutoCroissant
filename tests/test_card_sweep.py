"""
card_sweep: the kinds it takes and the icon font's earlier eras (card_guards.earlier_fonts) a badge widths sweep and
the guards read from the ref's history (the advances a raster drawn before the cooldown ring may show).
"""
import subprocess
from threading import Barrier

import pytest

import commands.card_guards as card_guards
from commands.card_guards import ICON_FONT_FILE, PREFETCH_WORKERS, Refused, Source, earlier_fonts
from commands.card_sweep import kinds_of
from conftest import RING, blob_sha, no_lazy

PRE_RING = "0a7db1c3"   # the batch's icon font (2026-11-04): the last before the ring
GLOW_12 = "0a4bc2eb"    # the glow halos at 12 units per px (2026-10-23): [fire:glow] 1011 units, 1411 since 903e2d6b
GLOW_TOKENS = "3539dc83"  # the :glow variants (2026-10-23): GLOW_12's advances, 182 glyphs drawn otherwise
FIRST = "54278c1e"      # the first build (2026-10-20): the later builds' advances, 70 glyphs drawn otherwise


def _advance(path: str, text: str) -> int:
    import uharfbuzz as hb
    buf = hb.Buffer()
    buf.add_str(text)
    buf.guess_segment_properties()
    hb.shape(hb.Font(hb.Face(hb.Blob.from_file_path(path), 0)), buf, {"liga": True})
    return sum(pos.x_advance for pos in buf.glyph_positions)


def _font_at(source: Source, rev: str) -> str:
    return source.git("rev-parse", f"{rev}:{ICON_FONT_FILE}").strip()


def test_earlier_icon_fonts(clone, tmp_path):
    """At the ring font's commit: one icon font per earlier era, newest first (the pre-ring batch's, the 12 units per px
    glow halos', the :glow variants' and the first build's: the same advances, drawn otherwise), none the ref's own nor
    another version of an era, their (1CD) the pre-ring width (the ring's 2658 units; the first build had no badges), named so a FontMap over the
    folder never indexes them; a second walk reads only the fonts it writes."""
    from psd_tools.composite.text.fonts import font_files
    source = Source(str(clone))
    commit = source.commit(RING)
    tree = source.tree(commit)
    files = earlier_fonts(source, commit, tree, str(tmp_path))
    shas = [blob_sha(path) for path in files]
    assert shas == [_font_at(source, rev) for rev in (PRE_RING, GLOW_12, GLOW_TOKENS, FIRST)]
    assert font_files(str(tmp_path)) == []
    ring = tmp_path / "ring.ttf"
    ring.write_bytes(source.blob(tree[ICON_FONT_FILE][1]))
    assert [_advance(path, "(1CD)") for path in files] == [2040, 2040, 2040, 2383] and _advance(str(ring), "(1CD)") == 2658
    assert [_advance(path, "[fire:glow]") for path in files] == [1411, 1011, 1011, 4653]    # the first: no :glow yet
    read, blob = [], source.blob
    source.blob = lambda sha: read.append(sha) or blob(sha)
    (tmp_path / "again").mkdir()
    again = earlier_fonts(source, commit, tree, str(tmp_path / "again"))
    assert [blob_sha(path) for path in again] == shas and read == shas
    assert kinds_of("badge-widths") == ["badge widths"]


def test_earlier_fonts_read_side_by_side(tmp_path, monkeypatch):
    """The versions are read PREFETCH_WORKERS at a time, the eras still newest first; a failed read is raised."""
    shas = [str(n) * 12 for n in range(1, 7)]
    meet = Barrier(PREFETCH_WORKERS, timeout=10)    # breaks unless that many read at once

    class History(Source):
        def git(self, *args):
            return " ".join(shas) if args[0] == "log" else args[1].split(":")[0]

        def blob(self, sha):
            if sha in shas[:PREFETCH_WORKERS]:
                meet.wait()
            return sha.encode()

    source, tree = History(str(tmp_path)), {ICON_FONT_FILE: ("100644", "now")}
    monkeypatch.setattr(card_guards, "font_glyphs", lambda data: {"glyph": int(data[:1])})
    monkeypatch.setattr(card_guards, "_glyphs", {"now": {"glyph": 0}})
    files = earlier_fonts(source, "head", tree, str(tmp_path))
    assert files == [str(tmp_path / f"stoogedings-{sha}.earlier") for sha in shas]

    def refused(sha):
        raise Refused("blob", sha)
    source.blob = refused
    monkeypatch.setattr(card_guards, "_glyphs", {"now": {"glyph": 0}})
    with pytest.raises(Refused):
        earlier_fonts(source, "head", tree, str(tmp_path))


def test_earlier_eras_past_more_font_changes(clone, tmp_path, monkeypatch):
    """Three font changes past the ring, each moving an advance (an era each): the walk still keeps the pre-ring eras (4
    eras at most dropped them: an old save judged against no pre-ring build); it stops at EARLIER_WALK versions."""
    repo = tmp_path / "repo"
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / ".git" / "objects" / "info" / "alternates").write_text(f"{clone / '.git' / 'objects'}\n")

    def git(*args, data=None) -> str:
        return subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", *args], input=data,
                              capture_output=True, check=True, env=no_lazy()).stdout.decode().strip()
    source = Source(str(repo))
    head = source.commit(RING)
    ring = source.blob(_font_at(source, RING))
    real, laid = card_guards.font_glyphs, card_guards.font_glyphs(ring)
    moved = next(name for name, (advance, _) in laid.items() if advance)

    def glyphs(data):                           # ring + b"#" + n: the ring's glyphs, one advance n units wider
        if data[:-2] != ring:
            return real(data)
        return {**laid, moved: (laid[moved][0] + data[-1], laid[moved][1])}
    monkeypatch.setattr(card_guards, "font_glyphs", glyphs)
    monkeypatch.setattr(card_guards, "_glyphs", {})
    made = []
    for n in (1, 2, 3):
        made.append(git("hash-object", "-w", "--stdin", data=ring + b"#" + bytes([n])))
        git("update-index", "--add", "--cacheinfo", f"100644,{made[-1]},{ICON_FONT_FILE}")
        head = git("commit-tree", git("write-tree"), "-p", head, "-m", f"font {n}")
    tree = source.tree(head)
    for name in ("all", "short"):
        (tmp_path / name).mkdir()
    eras = made[1::-1] + [_font_at(source, rev) for rev in (RING, PRE_RING, GLOW_12, GLOW_TOKENS, FIRST)]
    assert [blob_sha(path) for path in earlier_fonts(source, head, tree, str(tmp_path / "all"))] == eras
    monkeypatch.setattr(card_guards, "EARLIER_WALK", 3)      # the two after the ring and the ring
    assert [blob_sha(path) for path in earlier_fonts(source, head, tree, str(tmp_path / "short"))] == eras[:3]


def test_held_cards_unmoved_since_the_base_take_the_held_version():
    """Big pass 5.3: every held card still at the base is swept from the held branch; one pushed meanwhile keeps its own
    version (owner 11-07); --held-ref needs --held-base."""
    import commands.card_sweep as cw
    from commands.card_guards import Job

    class Trees:
        trees = {"held": {"A.psd": ("100644", "a2"), "B.psd": ("100644", "b2"), "C.psd": ("100644", "c2")},
                 "base": {"A.psd": ("100644", "a1"), "B.psd": ("100644", "b1"), "C.psd": ("100644", "c1")}}

        def commit(self, ref):
            return ref

        def tree(self, ref):
            return self.trees[ref]

    jobs = [Job("A.psd", "whole", "", "a1"), Job("B.psd", "whole", "", "b1"), Job("C.psd", "whole", "", "c3")]
    out, held = cw._held_versions(Trees(), {}, jobs, "held", "base")
    assert held == {"A.psd": "held", "B.psd": "held", "C.psd": "moved"}
    assert [job.psd_new for job in out] == ["a2", "b2", "c3"] and out[1].base == {"B.psd": "b1"}
    with pytest.raises(ValueError, match="together"):
        cw.run("", "normalise", "", held_ref="held")
