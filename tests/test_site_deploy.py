"""
The Card browser's deploy beside its build (TTSCardMaker .github/workflows/site.yml, .github/scripts/pages_wait.py): the
version job holds github-pages and deploys, the build (no environment: it starts ~2 s sooner) uploads and waits.
Loads both from $AC_SITE_SRC (a TTSCardMaker checkout) or the clone; skips without pages_wait.py (2026-11-17).
"""
import json
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from subprocess import run as run_process

import pytest
import yaml

from conftest import SITE_SRC

WAIT = SITE_SRC / ".github" / "scripts" / "pages_wait.py"
pytestmark = pytest.mark.skipif(not WAIT.is_file(), reason=f"no {WAIT}")


def test_the_version_job_deploys_what_the_build_uploads():
    doc = yaml.safe_load((SITE_SRC / ".github" / "workflows" / "site.yml").read_text(encoding="utf-8"))
    build, version = doc["jobs"]["build"], doc["jobs"]["version"]
    assert "environment" not in build and {"pages", "id-token"}.isdisjoint(build["permissions"])
    assert version["environment"]["name"] == "github-pages" and version["permissions"]["pages"] == "write"
    uploads = {s["name"]: s["with"]["name"].replace("${{ github.run_attempt }}", "$GITHUB_RUN_ATTEMPT")
               for s in build["steps"] if "upload-artifact" in s.get("uses", "") and s.get("name")}
    waits = [re.fullmatch(r'python3 \.github/scripts/pages_wait\.py artifact "([^"]+)" "([^"]+)"', s.get("run", ""))
             for s in version["steps"]]
    assert {m.group(2): m.group(1) for m in waits if m} == uploads and len(uploads) == 2
    fallbacks = [s["with"]["artifact_name"].replace("${{ github.run_attempt }}", "$GITHUB_RUN_ATTEMPT")
                 for s in version["steps"] if "deploy-pages" in s.get("uses", "")]
    assert sorted(fallbacks) == sorted(uploads.values())
    for job in (build, version):                                # the script in each one's checkout
        assert "/.github/scripts/pages_wait.py" in next(s for s in job["steps"] if "checkout" in s.get("uses", ""))["with"]["sparse-checkout"]
    last = build["steps"][-1]
    assert last["name"] == "Deploy wait" and last["run"].endswith("pages_wait.py deployed")


class _Run:
    """A stand-in run: its jobs and artifacts `at` s after its first read (ETags, 304s), the reads counted."""

    def __init__(self, timeline):
        self.timeline, self.start, self.reads = timeline, None, []

    def state(self):
        self.start = self.start or time.monotonic()        # python's start isn't timed
        at = time.monotonic() - self.start
        return [s for t, s in self.timeline if t <= at][-1]

    def serve(self):
        run = self

        class Api(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"           # one connection, as on GitHub

            def do_GET(self):
                jobs, artifacts = run.state()
                body = json.dumps({"jobs": jobs} if "/jobs" in self.path else {"artifacts": artifacts}).encode()
                tag = f'"{hash(body)}"'
                status = 304 if self.headers.get("If-None-Match") == tag else 200
                run.reads.append((self.path.split("?")[0].rsplit("/", 1)[1], status))
                self.send_response(status)
                self.send_header("ETag", tag)
                self.send_header("Content-Length", "0" if status == 304 else str(len(body)))
                self.end_headers()
                if status == 200:
                    self.wfile.write(body)

            def log_message(self, *args):
                pass
        server = ThreadingHTTPServer(("127.0.0.1", 0), Api)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        return server


def _wait(tmp_path, timeline, *args, attempt="1"):
    run = _Run(timeline)
    server = run.serve()
    out = tmp_path / "out"
    out.write_text("")
    env = {"GITHUB_API_URL": f"http://127.0.0.1:{server.server_port}", "GH_TOKEN": "t", "GITHUB_REPOSITORY": "o/r",
           "GITHUB_RUN_ID": "9", "GITHUB_RUN_ATTEMPT": attempt, "GITHUB_OUTPUT": str(out)}
    try:
        began = time.monotonic()
        done = run_process([sys.executable, "-S", str(WAIT), *args], env=env, capture_output=True, text=True, timeout=30)
        return done, time.monotonic() - (run.start or began), out.read_text(), run.reads
    finally:
        server.shutdown()


def _job(name, status, conclusion=None, attempt=1, **steps):
    return {"name": name, "status": status, "conclusion": conclusion, "run_attempt": attempt,
            "steps": [{"name": k.replace("_", " "), "status": "completed", "conclusion": v} for k, v in steps.items()]}


def test_pages_wait_hands_over(tmp_path):
    """The version job takes the upload once listed (0.25 s looks, 304s while nothing changes), or ends with none when
    the upload was skipped; with nothing deployed it fails when the build failed; the build waits for the deploys and
    fails when they failed or the version job was not re-run with it."""
    pages = [{"id": 77, "name": "github-pages-a1", "expired": False}]
    building, uploaded = _job("build", "in_progress"), _job("build", "in_progress", Pages_upload="success")
    done, took, out, reads = _wait(tmp_path, [(0, ([building], [])), (1.0, ([uploaded], pages))],
                                   "artifact", "github-pages-a1", "Pages upload")
    assert done.returncode == 0 and out == "artifact=77\n" and 1.0 <= took < 2.5, (done.stdout, done.stderr, took)
    assert ("artifacts", 304) in reads and reads[-1] == ("artifacts", 200)     # conditional looks till it was listed

    skipped = _job("build", "in_progress", Pages_upload="skipped")
    done, took, out, _ = _wait(tmp_path, [(0, ([skipped], []))], "artifact", "github-pages-a1", "Pages upload")
    assert done.returncode == 0 and out == "" and "No github-pages-a1" in done.stdout
    done, *_ = _wait(tmp_path, [(0, ([_job("build", "completed", "failure")], []))], "end")
    assert done.returncode == 1 and "the build failed" in done.stdout
    done, *_ = _wait(tmp_path, [(0, ([_job("build", "completed", "cancelled")], []))], "end")
    assert done.returncode == 0                                 # replaced while pending: the run stays 'cancelled'

    deploying, deployed = _job("version", "in_progress"), _job("version", "completed", "success")
    done, took, *_ = _wait(tmp_path, [(0, ([deploying], [])), (0.5, ([deployed], []))], "deployed")
    assert done.returncode == 0 and 0.5 <= took < 2, (done.stdout, took)
    done, *_ = _wait(tmp_path, [(0, ([_job("version", "completed", "failure")], []))], "deployed")
    assert done.returncode == 1 and "the deploy failed" in done.stdout
    done, *_ = _wait(tmp_path, [(0, ([_job("version", "completed", "success", attempt=1)], []))], "deployed", attempt="2")
    assert done.returncode == 1 and "did not run in attempt 2" in done.stdout
