"""scripts/repo_growth.py (by hand): what a range adds, raw and packed as git deltas it, by writer."""
import importlib.util
import io
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import commands.png_export as pe

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "repo_growth.py"


@pytest.mark.oxipng
def test_a_sweeps_banded_png_costs_its_changed_band(tmp_path, capsys):
    """A sweep's one-pixel change to a banded PNG: told under its Card-sweep kind, banded, packed a small part of it."""
    spec = importlib.util.spec_from_file_location("repo_growth", SCRIPT)
    growth = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(growth)
    env = {**os.environ, "GIT_AUTHOR_NAME": "Alice", "GIT_AUTHOR_EMAIL": "a@x", "GIT_COMMITTER_NAME": "Alice",
           "GIT_COMMITTER_EMAIL": "a@x"}
    git = lambda *args: subprocess.run(["git", "-C", str(tmp_path), *args], check=True, capture_output=True, env=env)
    git("init", "-q")
    y, x = np.mgrid[:600, :400]
    image = Image.fromarray((np.stack([x * 255 // 400, y * 255 // 600, (x + y) % 256], 2)
                             + np.random.default_rng(1).integers(0, 8, (600, 400, 3))).astype(np.uint8))
    for message in ("Card: A", "Cards: fixed\n\nCard-sweep: test"):
        buf = io.BytesIO()
        image.save(buf, "PNG")
        (tmp_path / "A.png").write_bytes(pe.band(buf.getvalue()))
        git("add", "A.png")
        git("commit", "-q", "-m", message)
        image.putpixel((5, 100), (0, 0, 0))
    growth.range_cost(str(tmp_path), "HEAD~1..HEAD")
    rows = {tuple(line.split()[:3]): line.split()[3:] for line in capsys.readouterr().out.splitlines()[1:]}
    versions, raw, packed, bands = rows[("sweep:", "test", "png")]
    assert (versions, bands) == ("1", "1") and float(packed) * 4 < float(raw)
