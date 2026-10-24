"""
png_recompress (the one batch pass over the card PNGs, owner 2026-10-23): in a git checkout made here, a Photoshop-sized
card PNG is rewritten losslessly (the same image as HEAD's blob), a small or already compressed one and a non-card PNG
are left, an edited file is skipped, and nothing is written without --apply or committed with it.
"""
import json
import subprocess
from importlib.util import find_spec

import pytest

import commands.png_export as pe
import commands.png_recompress as batch
from test_png_export import art, photoshop_png

pytestmark = pytest.mark.skipif(find_spec("oxipng") is None, reason="needs pyoxipng (none on 3.14: requirements-cards.txt)")


def _git(root, *args) -> str:
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, check=True).stdout.strip()


@pytest.fixture
def checkout(tmp_path):
    root = tmp_path / "cards"
    files = {"Items/Heal/2 Stars/Fat.png": photoshop_png(art()),
             "Items/Heal/2 Stars/Fat2.png": photoshop_png(art("RGB")),
             "Items/Heal/2 Stars/Small.png": photoshop_png(art(size=(80, 120))),
             "Items/Heal/2 Stars/Done.png": pe.recompress(photoshop_png(art())),
             "Items/Heal/2 Stars/Edited.png": photoshop_png(art()),
             "Types/fire.png": photoshop_png(art()),
             "MDW/Template.png": photoshop_png(art())}
    for path, data in files.items():
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_bytes(data)
    _git(root.parent, "init", "-q", "-b", "main", str(root))
    _git(root, "add", "-A")
    _git(root, "-c", "user.name=Base", "-c", "user.email=base@example.com", "commit", "-q", "-m", "base")
    (root / "Items/Heal/2 Stars/Edited.png").write_bytes(photoshop_png(art("RGB")))
    return root, files


def test_the_batch_rewrites_the_heavy_card_pngs(checkout, tmp_path, capsys):
    root, files = checkout
    head = _git(root, "rev-parse", "HEAD")
    assert batch.main(["--clone", str(root), "--workers", "1"]) == 0
    assert "recompressed: 2 files" in capsys.readouterr().out
    assert all((root / path).read_bytes() == data for path, data in files.items() if "Edited" not in path)   # measured only
    report = tmp_path / "report.json"
    assert batch.main(["--clone", str(root), "--apply", "--workers", "1", "--report", str(report)]) == 0
    rows = {row["path"].split("/")[-1]: row for row in json.loads(report.read_text())}
    assert set(rows) == {"Fat.png", "Fat2.png", "Small.png", "Done.png", "Edited.png"}     # card PNGs only
    assert {name: row["outcome"] for name, row in rows.items()} == {
        "Fat.png": "recompressed", "Fat2.png": "recompressed", "Small.png": "kept", "Done.png": "kept", "Edited.png": "skipped"}
    for name in ("Fat.png", "Fat2.png"):
        path = f"Items/Heal/2 Stars/{name}"
        new = (root / path).read_bytes()
        assert len(new) == rows[name]["after"] < rows[name]["before"] * 0.9 and pe.same_png(files[path], new)
    for path in ("Items/Heal/2 Stars/Small.png", "Items/Heal/2 Stars/Done.png", "Types/fire.png", "MDW/Template.png"):
        assert (root / path).read_bytes() == files[path]
    assert _git(root, "rev-parse", "HEAD") == head and not (root / "Items/Heal/2 Stars/Fat.png.recompress").exists()
    assert set(_git(root, "diff", "--name-only").splitlines()) == {f"Items/Heal/2 Stars/{n}" for n in ("Fat.png", "Fat2.png", "Edited.png")}
    assert batch.main(["--clone", str(root), "--apply", "--workers", "1", "--report", str(report)]) == 0   # rewritten: not HEAD's
    assert [row["outcome"] for row in json.loads(report.read_text())].count("recompressed") == 0


def test_the_batch_message_names_each_png_in_a_trailer(checkout, tmp_path):
    """--message: the batch commit's text, one Recompressed trailer per rewritten PNG (git parses them: TTSCardMaker
    build_site shows no new image version for those)."""
    root, files = checkout
    text = tmp_path / "message.txt"
    assert batch.main(["--clone", str(root), "--apply", "--workers", "1", "--message", str(text)]) == 0
    message = text.read_text()
    assert message.startswith("Card PNGs: 2 recompressed losslessly (0.00 GB -> 0.00 GB, the same pixels)\n\n")
    trailers = subprocess.run(["git", "interpret-trailers", "--parse"], input=message, capture_output=True, text=True,
                              check=True).stdout
    assert trailers == "".join(f"Recompressed: Items/Heal/2 Stars/{name}\n" for name in ("Fat.png", "Fat2.png"))
