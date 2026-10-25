"""
The card repo's workflows in the shadow (DATA_BUILD_ACTION.md step 7): site.yml reads the card sync's data only once its
CARD_DATA switch is on and the branch exists, and keeps its 6-hour schedule; card-sync.yml, once scheduled, runs every
30 minutes with the main commit and the posts off. At the cutover (step 8) these flip: update the asserts then. Loads
the workflows from $AC_SITE_SRC (a TTSCardMaker checkout) or the clone; skips without them or PyYAML.
"""
from os import environ
from os.path import expanduser
from pathlib import Path

import pytest

from conftest import CLONE

WORKFLOWS = Path(expanduser(environ.get("AC_SITE_SRC") or str(CLONE))) / ".github" / "workflows"


def _load(name: str) -> dict:
    path = WORKFLOWS / name
    if not path.is_file():
        pytest.skip(f"no {path}")
    yaml = pytest.importorskip("yaml")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_site_reads_the_data_only_when_switched_on():
    doc = _load("site.yml")
    if "CARD_DATA" not in (doc.get("env") or {}):
        pytest.skip("site.yml before the card sync's data checkout (TTSCardMaker work/site-history)")
    assert doc["env"]["CARD_DATA"] == "off"
    assert doc[True]["schedule"] == [{"cron": "17 */6 * * *"}]                  # PyYAML reads `on` as True
    steps = doc["jobs"]["build"]["steps"]
    branch = next(s for s in steps if s.get("id") == "data")
    assert branch["if"] == "env.CARD_DATA == 'on'" and "refs/heads/data" in branch["run"]
    checkout = next(s for s in steps if s.get("with", {}).get("ref") == "data")
    assert checkout["if"] == "steps.data.outputs.exists == 'true'" and checkout["with"]["path"] == "carddata"
    build = next(s for s in steps if "build_site.py" in s.get("run", ""))
    assert build["env"]["CARD_DATA_DIR"] == "${{ steps.data.outputs.exists == 'true' && 'carddata/db' || '' }}"
    assert steps.index(branch) < steps.index(checkout) < steps.index(build)


def test_card_sync_shadow_schedule_writes_only_data():
    doc = _load("card-sync.yml")
    on = doc[True]
    if "schedule" not in on:
        pytest.skip("card-sync.yml has no schedule yet (dispatch only, step 6)")
    assert on["schedule"] == [{"cron": "4,34 * * * *"}]
    assert doc["env"]["MAIN_COMMIT"] == "off" and doc["env"]["POSTS"] == "off"
    assert on["workflow_dispatch"]["inputs"]["dry_run"]["default"] is True


def test_site_builds_on_what_it_reads_and_always_deploys():
    """A push runs the build only when it touched what the build reads: everything the job checks out, the card files
    and fonts (read through git), this workflow; never the TTS save. Every build deploys (skipping a build equal to
    the live one is the owner's to decide: the Changes page's "Updated" line would stop moving)."""
    doc = _load("site.yml")
    paths = doc[True]["push"].get("paths")
    if paths is None:
        pytest.skip("site.yml before its paths list (TTSCardMaker work/site-perf-back)")
    checkout = next(s for s in doc["jobs"]["build"]["steps"] if s.get("with", {}).get("sparse-checkout"))
    for line in checkout["with"]["sparse-checkout"].split():
        path = line.strip("/")
        assert (path + "/**" if line.endswith("/") else path) in paths, line
    for pattern in ("**.png", "**.PNG", "**.psd", "**.PSD", "MDW/Fonts/*", ".github/workflows/site.yml"):
        assert pattern in paths
    assert paths[len(paths) - 1] == "!TTS/**"
    steps = doc["jobs"]["build"]["steps"]
    upload = next(s for s in steps if "upload-pages-artifact" in s.get("uses", ""))
    assert "if" not in upload and "if" not in doc["jobs"]["deploy"]
