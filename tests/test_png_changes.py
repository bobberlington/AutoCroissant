"""png_changes: the per-pixel rule (colour over black or alpha off by more than 2 of 255, inside a 7x7 patch), the card
PNG change pairs git gives (the website's), the record Card sync and the backfill keep (resumable, a missing pair
visible), and card_data's minor art and text events."""
import io
import subprocess
from os import environ
from types import SimpleNamespace

import numpy as np
from PIL import Image

import commands.png_changes as pc
from commands.card_data import mark_minor, picture_pairs


def _png(pixels) -> bytes:
    out = io.BytesIO()
    Image.fromarray(np.asarray(pixels, dtype=np.uint8), "RGBA").save(out, "PNG")
    return out.getvalue()


def _card(shade=100) -> np.ndarray:
    return np.full((100, 100, 4), (shade, 60, 40, 255), dtype=np.uint8)


def test_the_rule_counts_patches_not_noise_or_strokes():
    base = _card()
    noise = base.copy()
    noise[..., 0] += np.random.default_rng(1).integers(0, 3, (100, 100)).astype(np.uint8)    # off by 2 at most
    stroke = base.copy()
    stroke[50, :, 0] = 250                                                                    # a 1 px line
    patch = base.copy()
    patch[10:30, 10:30, 0] = 110                                                              # 20 x 20 off by 10
    clear = base.copy()
    clear[:50, :, 3] = 0                                                                      # see-through half
    hidden = clear.copy()
    hidden[:50, :, :3] = 7                                                                    # colour under alpha 0
    assert pc.changed_share(_png(base), _png(base)) == 0.0
    assert pc.changed_share(_png(base), _png(noise)) == 0.0
    assert pc.changed_share(_png(base), _png(stroke)) == 0.0
    assert pc.changed_share(_png(base), _png(patch)) == 0.04
    assert pc.changed_share(_png(clear), _png(hidden)) == 0.0
    assert pc.changed_share(_png(base), _png(base[:90])) == 1.0


def _commit(repo, message, env=None):
    git = lambda *a: subprocess.run(["git", "-C", str(repo), *a], capture_output=True, text=True, check=True,
                                    env={**environ, "GIT_AUTHOR_NAME": "Maker", "GIT_AUTHOR_EMAIL": "m@x",
                                         "GIT_COMMITTER_NAME": "Maker", "GIT_COMMITTER_EMAIL": "m@x", **(env or {})}).stdout
    git("add", "-A")
    git("commit", "-q", "-m", message)
    return git("rev-parse", "HEAD").strip()


def _blob(repo, rev, path):
    return subprocess.run(["git", "-C", str(repo), "rev-parse", f"{rev}:{path}"], capture_output=True, text=True,
                          check=True).stdout.strip()


def test_pairs_the_record_and_a_resumed_run(tmp_path, monkeypatch):
    """A modified PNG and one moved and changed (same file name) are pairs; an exact move and the bot's recompression
    (its Recompressed trailer) are not. A run measures what the record lacks (beside: a child process; it failed: here);
    the next run nothing more."""
    repo = tmp_path / "repo"
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    one, two = repo / "Creatures/A/1 Stars/One.png", repo / "Items/B.png"
    one.parent.mkdir(parents=True)
    two.parent.mkdir(parents=True)
    one.write_bytes(_png(_card(100)))
    two.write_bytes(_png(_card(100)))
    first = _commit(repo, "Cards")
    one.write_bytes(_png(_card(160)))
    _commit(repo, "Art")
    moved = repo / "Creatures/A/2 Stars/One.png"
    moved.parent.mkdir(parents=True)
    one.rename(moved)
    moved.write_bytes(_png(_card(30)))
    two.rename(repo / "Items/C.png")
    _commit(repo, "Moved")
    data, shrunk = (repo / "Items/C.png").read_bytes(), io.BytesIO()
    Image.open(io.BytesIO(data)).save(shrunk, "PNG", compress_level=1)            # the same pixels, other bytes
    assert shrunk.getvalue() != data
    (repo / "Items/C.png").write_bytes(shrunk.getvalue())
    _commit(repo, "Card PNGs: recompressed\n\nRecompressed: Items/C.png", {"GIT_AUTHOR_NAME": pc.BOT})
    pairs = pc.card_pairs(str(repo), "HEAD")
    assert pairs == [(_blob(repo, first, "Creatures/A/1 Stars/One.png"), _blob(repo, "HEAD~2", "Creatures/A/1 Stars/One.png")),
                     (_blob(repo, "HEAD~2", "Creatures/A/1 Stars/One.png"), _blob(repo, "HEAD", "Creatures/A/2 Stars/One.png"))]
    assert pc.card_pairs(str(repo), "HEAD", "HEAD~1") == []
    record = pc.read_record(str(tmp_path))
    assert pc.visible(record, *pairs[0])                             # not measured: shows
    done, left = pc.Beside(str(repo), pairs, record).join(record, log=lambda _: None)
    assert (done, left) == (2, []) and all(pc.share_of(record, *p) == 1.0 for p in pairs)
    capped = pc.read_record(str(tmp_path))                          # its limit spent: the run's event pair still first
    assert pc.Beside(str(repo), pairs, capped, limit=1).join(capped, [pairs[1]], log=lambda _: None) == (2, [])
    failed, unforked = pc.read_record(str(tmp_path)), pc.read_record(str(tmp_path))
    monkeypatch.setattr(pc, "executable", "false")
    assert pc.Beside(str(repo), pairs, failed).join(failed, log=lambda _: None) == (2, []) and failed == record

    def no_fork(args, real=pc.Popen, **kwargs):                     # the child alone: its readers still start
        if args[0] == pc.executable:
            raise BlockingIOError(35, "Resource temporarily unavailable")
        return real(args, **kwargs)
    monkeypatch.setattr(pc, "Popen", no_fork)
    assert pc.Beside(str(repo), pairs, unforked).join(unforked, log=lambda _: None) == (2, []) and unforked == record
    pc.write_record(str(tmp_path), record)
    again = pc.read_record(str(tmp_path))
    assert pc.measure(str(repo), pairs, again, log=lambda _: None) == (0, []) and again == record


def test_minor_art_and_text_events():
    """An art event whose PNG changed under VISIBLE is minor ("picture"); a changed one with only a formatting change
    of its ability too ("text"), an edited one always; a reworded one, or one whose picture shows, is not."""
    put = SimpleNamespace(order=3)
    index = SimpleNamespace(at_path={"C/A.psd": [(SimpleNamespace(order=1), "p2"), (put, "p1")],
                                     "C/A.png": [(SimpleNamespace(order=2), "g2"), (SimpleNamespace(order=4), "g1")]})
    stats = {"A": SimpleNamespace(variant=None, path="C/A.psd")}
    art = {"kind": "art", "card": "A", "path": "C/A.psd", "blob": ["p1", "p2"]}
    text = {"kind": "changed", "card": "A", "path": "C/A.psd", "blob": ["p1", "p2"],
            "fields": {"ability": ["Deal 2 [fire] damage", "Deal 2 fire damage."], "tags": [[], []]}}
    edited = {"kind": "edited", "card": "A", "fields": {"ability": ["Heal 2.", "heal  2"]}}
    worded = {"kind": "edited", "card": "A", "fields": {"ability": ["Heal 2.", "Heal 3."]}}
    events = [art, text, edited, worded]
    shown = picture_pairs(events, index, {"C/A.png": "g3", "C/A.psd": "p2"}, stats)
    assert shown == {id(art): ("g1", "g3"), id(text): ("g1", "g3")}
    record = {"rule": pc.RULE, "pairs": {pc.key("g1", "g3"): [pc.VISIBLE / 2, pc.RULE]}}
    mark_minor(events, shown, record)
    assert (art.get("minor"), text.get("minor"), edited.get("minor"), worded.get("minor")) == ("picture", "text", "text", None)
    record["pairs"][pc.key("g1", "g3")] = [pc.VISIBLE, pc.RULE]
    for e in events:
        e.pop("minor", None)
    mark_minor(events, shown, record)
    assert [e.get("minor") for e in events] == [None, None, "text", None]
