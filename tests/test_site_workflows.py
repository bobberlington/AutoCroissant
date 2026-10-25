"""
The card repo's workflows after the cutover (DATA_BUILD_ACTION.md step 8): site.yml reads the card sync's data (the
branch data, its only source) with a daily fallback build; card-sync.yml runs live with the main commit and the posts on:
a card push's data at once, its bake and save after a 30-minute timer, the cron a fallback. Loads the workflows from
$AC_SITE_SRC (a TTSCardMaker checkout) or the clone; skips without them or PyYAML.
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


def test_site_reads_the_data_branch():
    doc = _load("site.yml")
    if "CARD_DATA" in (doc.get("env") or {}):
        pytest.skip("site.yml with the pickle fallback (before TTSCardMaker work/pickles)")
    assert doc[True]["schedule"] == [{"cron": "0 12 * * *"}]                   # PyYAML reads `on` as True
    steps = doc["jobs"]["build"]["steps"]
    checkout = next(s for s in steps if s.get("with", {}).get("ref") == "data")
    assert "if" not in checkout and checkout["with"]["path"] == "carddata"
    build = next(s for s in steps if "build_site.py" in s.get("run", ""))
    assert build["env"]["CARD_DATA_DIR"] == "carddata/db" and steps.index(checkout) < steps.index(build)


def test_card_sync_runs_live():
    doc = _load("card-sync.yml")
    on = doc[True]
    if "schedule" not in on:
        pytest.skip("card-sync.yml has no schedule yet (dispatch only, step 6)")
    assert on["schedule"] == [{"cron": "0,30 * * * *"}]
    assert not {"MAIN_COMMIT", "POSTS"} & set(doc["env"])                    # no rollback switches (2026-10-25)
    assert on["workflow_dispatch"]["inputs"]["dry_run"]["default"] is False and "seed" not in on["workflow_dispatch"]["inputs"]
    assert not (WORKFLOWS / "card-edits.yml").exists() and not (WORKFLOWS / "tts-save.yml").exists()   # its two writers


def test_card_sync_data_at_once_main_after_a_push_timer():
    """Owner 2026-10-25: only main's writes wait. A card push (every path the bot counts as a card input) runs the data
    lane at once; the main lane (bake, save) waits out the 30-minute timer (the environment card-sync-quiet), then the
    after job re-reads the bake into data. One writer a branch: data and after share card-data, main is card-main."""
    from commands.card_sync import CARD_FILES
    doc = _load("card-sync.yml")
    push = doc[True].get("push")
    if push is None:
        pytest.skip("card-sync.yml before its push timer (TTSCardMaker work/clock)")
    assert push["branches"] == ["main"] and CARD_FILES <= set(push["paths"])
    assert {"**.[pP][sS][dD]", "**.[pP][nN][gG]", "Types/**", "MDW/Fonts/*"} <= set(push["paths"])
    jobs = doc["jobs"]
    if "data" not in jobs:
        pytest.skip("card-sync.yml before the data lane (TTSCardMaker work/split)")
    assert "needs" not in jobs["data"] and "environment" not in jobs["data"]
    assert "card-sync-quiet" in jobs["quiet"]["environment"] and jobs["quiet"]["needs"] == "timer"
    assert jobs["main"]["needs"] == ["timer", "quiet"] and jobs["after"]["needs"] == "main"
    groups = {name: (job.get("concurrency") or {}).get("group") for name, job in jobs.items()}
    assert groups == {"data": "card-data", "timer": None, "quiet": None, "main": "card-main", "after": "card-data", "caches": None}
    assert jobs["after"]["steps"] == jobs["data"]["steps"] and jobs["after"]["env"]["AFTER"] == "${{ needs.main.outputs.m }}"
    pushes = {name: "\n".join(s.get("run", "") for s in job.get("steps", [])) for name, job in jobs.items()}
    assert "refs/heads/data" in pushes["data"] and "refs/heads/main" not in pushes["data"].replace("ls-remote origin refs/heads/main", "")
    assert "refs/heads/main" in pushes["main"] and "HEAD:refs/heads/data" not in pushes["main"]
    assert "concurrency" not in doc


def test_site_builds_on_what_it_reads_and_deploys_only_a_change():
    """A push runs the build only for site code (2026-10-25): the card inputs (PSDs, PNGs, Types/, the fonts, the card
    files in site/) start Card sync, which dispatches the build after its data commit. A build that made the live site
    again is not deployed (build.txt, the site's digest; owner 2026-10-24: the Changes page's "Site updated" is the last
    change)."""
    doc = _load("site.yml")
    paths = doc[True]["push"].get("paths")
    if paths is None:
        pytest.skip("site.yml before its paths list (TTSCardMaker work/site-perf-back)")
    assert "site/**" in paths and ".github/workflows/site.yml" in paths and "!TTS/**" not in paths
    for pattern in ("**.png", "**.PNG", "**.psd", "**.PSD", "Types/**", "MDW/Fonts/*"):
        assert pattern not in paths, pattern
    for card_file in ("site/Card Edits.json", "site/Card Authors.txt", "site/Card Creation Dates.json"):
        assert "!" + card_file in paths, card_file
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


CACHE_PRUNES = {"site.yml": ("build", "card-images-v2- deck-previews-v2-"), "card-sync.yml": (["data", "main"], "cards- card-fonts-"),
                "card-guards.yml": ("check", "cards- card-fonts-"), "card-preview.yml": ("preview", "cards- card-fonts-")}


def test_cache_saves_prune_superseded_entries():
    """A workflow that saves a cache whose key changes over time calls caches.yml after a save (its saving job's `saved`),
    with its key prefixes and actions: write on that call alone; caches.yml keeps the newest two and any entry used in
    the last hour, and never fails the caller."""
    caches = _load("caches.yml")
    prune = caches["jobs"]["prune"]
    assert list(caches[True]) == ["workflow_call"] and caches["permissions"] == {} and prune["permissions"] == {"actions": "write"}
    step = prune["steps"][0]
    assert step["env"]["KEEP"] == "2" and step["env"]["IDLE"] == "3600" and step["run"].rstrip().endswith("exit 0")
    assert '--ref "$GITHUB_REF"' in step["run"] and "gh cache delete" in step["run"]
    for name, (job, prefixes) in CACHE_PRUNES.items():
        doc = _load(name)
        call = doc["jobs"]["caches"]
        assert call["uses"] == "./.github/workflows/caches.yml" and call["with"] == {"prefixes": prefixes}, name
        assert call["needs"] == job and call["permissions"] == {"actions": "write"}, name
        saved = " || ".join(f"needs.{j}.outputs.saved == 'true'" for j in ([job] if isinstance(job, str) else job))
        assert call["if"] == f"${{{{ !cancelled() && {saved if isinstance(job, str) else f'({saved})'} }}}}", name
        saves = [s for j in doc["jobs"].values() for s in j.get("steps", [])
                 if s.get("uses", "").startswith(("actions/cache@", "actions/cache/save@"))]
        keys = {"cards-" if s["with"]["path"] == "~/.cache/card-wheels" else s["with"]["key"].split("$")[0] for s in saves}
        assert keys <= set(prefixes.split()), (name, keys)
        assert all("saved" in doc["jobs"][j]["outputs"] for j in ([job] if isinstance(job, str) else job)), name
    site = _load("site.yml")["jobs"]["build"]["steps"]
    python = next(s for s in site if s.get("uses", "").startswith("actions/setup-python@"))
    assert "cache" not in python["with"]                                     # uv, no pip cache entry per requirements change


def test_root_checkout_first():
    """actions/checkout into the workspace root empties it first: a job's root checkout comes before its bot/ ones."""
    for path in sorted(WORKFLOWS.glob("*.yml")):
        for name, job in (_load(path.name).get("jobs") or {}).items():
            outs = [s.get("with", {}).get("path") for s in job.get("steps", []) if s.get("uses", "").startswith("actions/checkout@")]
            assert None not in outs or outs.index(None) == 0, (path.name, name)
