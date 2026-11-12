"""card-sync.yml's failure alert (owner 11-11): the standards channel, one alert per broken push across Card sync and
Card guards (the marker comment on the push's head commit), the failed steps named, no preview card (flags 4)."""
import json
import os
import subprocess

import pytest

from conftest import SITE_SRC

MARK = "<!-- discord-failure-told -->"
FAKE_GH = r"""#!/usr/bin/env python3
import json, os, subprocess, sys
state = os.environ["FAKE_STATE"]
db = json.load(open(state)) if os.path.exists(state) else {"comments": {}}
args = sys.argv[1:]


def jq(data):
    expr = args[args.index("--jq") + 1]
    return subprocess.run(["jq", "-r", expr], input=json.dumps(data), capture_output=True, text=True, check=True).stdout.rstrip("\n")


out = ""
if args[:2] == ["run", "list"]:
    out = jq(json.loads(os.environ.get("FAKE_RUNS", "[]")))
else:
    path = next(a for a in args[1:] if a.startswith("repos/"))
    if path.endswith("/jobs"):
        out = jq(json.loads(os.environ.get("FAKE_JOBS", '{"jobs": []}')))
    elif path.endswith("/comments") and "POST" in args:
        db["comments"].setdefault(path.split("/")[4], []).append(json.load(sys.stdin)["body"])
    elif path.endswith("/comments"):
        out = jq([{"id": i, "body": b} for i, b in enumerate(db["comments"].get(path.split("/")[4], []))])
    else:                                       # a guards run
        out = jq(json.loads(os.environ["FAKE_GUARDS_RUN"]))
json.dump(db, open(state, "w"))
print(out)
"""
FAKE_CURL = '#!/bin/sh\njq -c . >> "$FAKE_POSTS"\n'


def test_a_broken_push_tells_the_standards_channel_once(tmp_path):
    yaml = pytest.importorskip("yaml")
    path = SITE_SRC / ".github" / "workflows" / "card-sync.yml"
    if not path.is_file():
        pytest.skip(f"no {path}")
    jobs = yaml.safe_load(path.read_text(encoding="utf-8"))["jobs"]
    tell = {j: next(s for s in jobs[j]["steps"] if s.get("name") == "Tell Discord") for j in ("data", "main", "after")}
    assert len({t["run"] for t in tell.values()}) == 1 and "DISCORD_CHANGELOG_WEBHOOK" not in json.dumps(tell)
    assert all(t["env"]["WEBHOOK"] == "${{ secrets.DISCORD_STANDARDS_WEBHOOK }}" for t in tell.values())
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, text in (("gh", FAKE_GH), ("curl", FAKE_CURL), ("sleep", "#!/bin/sh\n")):
        (bin_dir / name).write_text(text)
        (bin_dir / name).chmod(0o755)
    posts, state = tmp_path / "posts.jsonl", tmp_path / "gh.json"
    env = {"PATH": f"{bin_dir}:{os.environ['PATH']}", "FAKE_STATE": str(state), "FAKE_POSTS": str(posts), "WEBHOOK": "https://hook",
           "GITHUB_REPOSITORY": "o/cards", "GITHUB_RUN_ID": "9", "RUN_URL": "https://x/run", "RUNNER_NAME": "GitHub Actions 3",
           "GITHUB_SHA": "ab" * 20, "GITHUB_ACTOR": "Croissant2", "GITHUB_EVENT_NAME": "push", "GITHUB_JOB": "data"}
    me = {"status": "in_progress", "runner_name": "GitHub Actions 3", "steps": [
        {"name": "Gate", "conclusion": "success"}, {"name": "Build", "conclusion": "failure"}, {"name": "Tell Discord", "conclusion": None}]}
    other = {"status": "in_progress", "runner_name": "GitHub Actions 4", "steps": [{"name": "Bake", "conclusion": "failure"}]}

    def tell_once(job="data", **extra):
        done = subprocess.run(["bash", "--noprofile", "--norc", "-eo", "pipefail", "-c", tell[job]["run"]],
                              env={**env, "GITHUB_JOB": job, **extra}, capture_output=True, text=True, timeout=60)
        assert done.returncode == 0, done.stdout + done.stderr
        sent = [json.loads(line) for line in posts.read_text().splitlines() if line] if posts.exists() else []
        posts.unlink(missing_ok=True)
        return sent, done.stdout
    sent, _ = tell_once(FAKE_JOBS=json.dumps({"jobs": [me, other]}))
    assert sent == [{"content": f"Card sync failed on Croissant2's push abababab: Build https://x/run\n"
                                "The website and the bot keep the last good data.", "flags": 4, "allowed_mentions": {"parse": []}}]
    assert json.loads(state.read_text())["comments"] == {"ab" * 20: [f"Failure told in Discord: https://x/run\n{MARK}"]}
    sent, out = tell_once("main", OUTCOMES="Bake=failure\n")                # the same push's other lane: told already
    assert sent == [] and "told in Discord already" in out
    guards = json.dumps({"event": "push", "head_sha": "cd" * 20, "actor": {"login": "netverk"}})
    follow_up = {"GITHUB_EVENT_NAME": "workflow_dispatch", "GITHUB_ACTOR": "github-actions[bot]", "GUARDS_RUN": "77",
                 "FAKE_GUARDS_RUN": guards, "OUTCOMES": "Gate=success\nBake=failure\nCommit main=skipped\n"}
    sent, _ = tell_once("main", **follow_up)                                  # no job list: the outcomes name the step
    assert [s["content"] for s in sent] == ["Card sync failed on netverk's push cdcdcdcd: Bake https://x/run\n"
                                            "Website edits wait for the next bake."]
    state.write_text(json.dumps({"comments": {"ef" * 20: [f"Card guards failed\n{MARK}"]}}))   # Card guards told it
    sent, _ = tell_once(GITHUB_SHA="ef" * 20, FAKE_JOBS=json.dumps({"jobs": [me]}))
    assert sent == []
    sent, _ = tell_once(GITHUB_EVENT_NAME="schedule", GITHUB_SHA="12" * 20, FAKE_RUNS="[]")   # no push, no streak, no steps read
    assert [s["content"].split("\n")[0] for s in sent] == ["Card sync failed on main 12121212: Card data https://x/run"]
