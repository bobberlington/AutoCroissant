"""
The card repo's workflows after the cutover (DATA_BUILD_ACTION.md step 8): site.yml reads the card sync's data (CARD_DATA
on, when the branch exists) with a daily fallback build; card-sync.yml runs live with the main commit and the posts on,
started by a card push's 30-minute timer, the cron a fallback. Loads the workflows from $AC_SITE_SRC (a TTSCardMaker checkout) or the clone; skips without them or
PyYAML.
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


def test_site_reads_the_data_when_switched_on():
    doc = _load("site.yml")
    if "CARD_DATA" not in (doc.get("env") or {}):
        pytest.skip("site.yml before the card sync's data checkout (TTSCardMaker work/site-history)")
    assert doc["env"]["CARD_DATA"] == "on"
    assert doc[True]["schedule"] == [{"cron": "0 12 * * *"}]                   # PyYAML reads `on` as True
    steps = doc["jobs"]["build"]["steps"]
    branch = next(s for s in steps if s.get("id") == "data")
    assert branch["if"] == "env.CARD_DATA == 'on'" and "refs/heads/data" in branch["run"]
    checkout = next(s for s in steps if s.get("with", {}).get("ref") == "data")
    assert checkout["if"] == "steps.data.outputs.exists == 'true'" and checkout["with"]["path"] == "carddata"
    build = next(s for s in steps if "build_site.py" in s.get("run", ""))
    assert build["env"]["CARD_DATA_DIR"] == "${{ steps.data.outputs.exists == 'true' && 'carddata/db' || '' }}"
    assert steps.index(branch) < steps.index(checkout) < steps.index(build)


def test_card_sync_runs_live():
    doc = _load("card-sync.yml")
    on = doc[True]
    if "schedule" not in on:
        pytest.skip("card-sync.yml has no schedule yet (dispatch only, step 6)")
    assert on["schedule"] == [{"cron": "0,30 * * * *"}]
    assert doc["env"]["MAIN_COMMIT"] == "on" and doc["env"]["POSTS"] == "on"
    assert on["workflow_dispatch"]["inputs"]["dry_run"]["default"] is False
    assert not (WORKFLOWS / "card-edits.yml").exists() and not (WORKFLOWS / "tts-save.yml").exists()   # its two writers


def test_card_sync_starts_from_a_push_timer():
    """A card push starts a 30-minute timer (the environment card-sync-quiet), on every path the bot counts as a card input."""
    from commands.card_sync import CARD_FILES
    doc = _load("card-sync.yml")
    push = doc[True].get("push")
    if push is None:
        pytest.skip("card-sync.yml before its push timer (TTSCardMaker work/clock)")
    assert push["branches"] == ["main"] and CARD_FILES <= set(push["paths"])
    assert {"**.[pP][sS][dD]", "**.[pP][nN][gG]", "Types/**", "MDW/Fonts/*"} <= set(push["paths"])
    jobs = doc["jobs"]
    assert "card-sync-quiet" in jobs["quiet"]["environment"] and jobs["quiet"]["needs"] == "timer"
    assert jobs["sync"]["needs"] == ["timer", "quiet"] and jobs["sync"]["concurrency"]["group"] == "card-sync"
    assert "concurrency" not in doc and "concurrency" not in jobs["timer"] and "concurrency" not in jobs["quiet"]


def test_site_builds_on_what_it_reads_and_deploys_only_a_change():
    """A push runs the build only when it touched what the build reads: everything the job checks out, the card files
    and fonts (read through git), this workflow; never the TTS save. A build that made the live site again is not
    deployed (build.txt, the site's digest; owner 2026-10-24: the Changes page's "Site updated" is the last change)."""
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
    changed = next(s for s in steps if s.get("id") == "changed")
    upload = next(s for s in steps if "upload-pages-artifact" in s.get("uses", ""))
    assert "build.txt" in changed["run"] and 'deploy=true' in changed["run"] and upload["if"] == "steps.changed.outputs.deploy == 'true'"
    assert steps.index(changed) < steps.index(upload)
    assert doc["jobs"]["build"]["outputs"]["deploy"] == "${{ steps.changed.outputs.deploy }}"
    assert doc["jobs"]["deploy"]["if"] == "needs.build.outputs.deploy == 'true'"


def test_site_image_budget_keeps_a_partial_cache():
    """A full AVIF re-encode may not fit one run: the build stops making images at IMAGE_BUDGET and fails, and the images
    it made are saved under a key the next run's restore-keys find (actions/cache saves only after a passing job), after
    any failure once the build ran (a budget stop, failed downloads, an error, the step's timeout, a later step), when the
    cache's files changed (a build that stopped before the images saves no copy)."""
    doc = _load("site.yml")
    steps = doc["jobs"]["build"]["steps"]
    build = next(s for s in steps if "build_site.py" in s.get("run", ""))
    if "IMAGE_BUDGET" not in build.get("env", {}):
        pytest.skip("site.yml before AVIF thumbnails (TTSCardMaker work/site-img)")
    budget = int(build["env"]["IMAGE_BUDGET"])
    assert build["id"] == "build" and 0 < budget <= (doc["jobs"]["build"]["timeout-minutes"] - 10) * 60
    cache = next(s for s in steps if s.get("uses", "").startswith("actions/cache@"))
    save = next(s for s in steps if s.get("uses", "").startswith("actions/cache/save@") and s["with"]["path"] == "site/.cache")
    before = next(s for s in steps if s.get("id") == "cache-before")
    after = next(s for s in steps if s.get("id") == "cache-after")
    assert after["if"] == "failure() && (steps.build.outcome == 'failure' || steps.build.outcome == 'success')"
    assert "steps.cache-after.outputs.names != steps.cache-before.outputs.names" in save["if"]   # no copy of an untouched cache
    assert before["run"] == after["run"] and "site/.cache" in after["run"]
    assert steps.index(cache) < steps.index(before) < steps.index(build) < steps.index(after)
    assert steps[-2:] == [after, save]                         # last: a later step's failure is saved too
    assert budget < build["timeout-minutes"] * 60 < doc["jobs"]["build"]["timeout-minutes"] * 60   # the step stops first: the save runs
    prefix = cache["with"]["restore-keys"]
    assert save["with"]["key"].startswith(prefix) and cache["with"]["key"].startswith(prefix) and "github.run_id" in save["with"]["key"]
    assert steps.index(build) < steps.index(save)
