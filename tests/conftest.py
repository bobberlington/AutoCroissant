# Run: ~/.venvs/ac-py314-fork/bin/python -m pytest -q -n auto tests   (offline; never writes the reminder store)
# The suite needs the psd-tools fork with psd-tools[text] (that venv; see use_fork); -n 0 runs one test serially.
import hashlib
import json
import re
import shutil
import socket
import sys
import types
from importlib import import_module
from os import cpu_count, environ, getpid
from os.path import expanduser
from pathlib import Path
from shlex import quote

import pytest

# ========================
# Configuration
# ========================
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from global_config import FORUM_JSON, LOCAL_DIR_LOC, REMIND_JSON  # noqa: E402

CLONE = Path(expanduser(LOCAL_DIR_LOC))                  # the card repo checkout: its objects (and its site code)
CARDS_REV = "a4d137bf06cd372649090a32644023abee8719ea"   # the card repo commit real-card tests read (re-pin: tests/make_goldens.py)
HISTORY = Path(expanduser("~/Documents/Tools/wt/scratch/tc-history.git"))   # full history (no blobs) for revs the clone lacks
PINNED = Path(expanduser("~/.cache/autocroissant-tests"))                   # the pinned trees, one per CARDS_REV
PINNED_BASE = ("Types", "MDW/Fonts", "MDW/Croissant's Special Folder/Satus Effect Colors.psd", "site/Card Edits.json")
LIVE = (REMIND_JSON, FORUM_JSON)                         # the checkout's live stores, never written by a test
GIT_CONFIG = {"core.hooksPath": "/dev/null",              # no machine hooks in fixture repos
              "core.looseCompression": "0"}               # fixture repos of real PSDs: no zlib on every add
POOLED = ("commands.card_guards",)                          # process pools capped per worker
QUEUES = ("dispatch_queue", "edit_queue", "command_queue")                         # utils' deques, empty per test
TOOLS = {"node": "node", "jq": "jq", "timeout": "coreutils", "git": "git", "oxipng": "oxipng",   # CLIs the tests run:
         "actionlint": "actionlint"}                                                              # brew formula
FETCH = ("GIT_NO_LAZY_FETCH=1 git -C {clone} -c fetch.negotiationAlgorithm=noop fetch --no-tags --no-write-fetch-head"
         " --filter=blob:none --stdin origin")           # blob ids on stdin, one a line


def _git_config():
    """GIT_CONFIG for every git this run starts (kept user identity, unlike GIT_CONFIG_GLOBAL=/dev/null)."""
    start = int(environ.get("GIT_CONFIG_COUNT", "0"))
    for i, (key, value) in enumerate(GIT_CONFIG.items(), start):
        environ[f"GIT_CONFIG_KEY_{i}"], environ[f"GIT_CONFIG_VALUE_{i}"] = key, value
    environ["GIT_CONFIG_COUNT"] = str(start + len(GIT_CONFIG))


_git_config()
environ["PSD_DIRTY_VERIFY"] = "1"                         # gate_dirty renders whole too: raises on any difference

TC_PATHS = ("AC_SITE_SRC", "ESBUILD")                     # TTSCardMaker paths from the environment
for _key in TC_PATHS:                                     # '~' expanded for build_site and subprocesses too
    if environ.get(_key):
        environ[_key] = expanduser(environ[_key])
SITE_SRC = Path(environ.get("AC_SITE_SRC") or CLONE)     # a TTSCardMaker checkout (code, not cards), else the clone


# ========================
# psd-tools fork
# ========================
FORK_ENV = "AC_PSD_TOOLS_SRC"                           # a fork src dir tested instead of the installed one
FORK_API = Path("psd_tools") / "api" / "text_edit.py"   # what upstream psd-tools lacks
FORK_VENV = "~/.venvs/ac-py314-fork/bin/python"         # the fork at the pin, psd-tools[text]


def use_fork() -> str:
    """$AC_PSD_TOOLS_SRC first on sys.path when set; else the installed psd_tools. Returns why the fork can't be used ("")."""
    chosen = environ.get(FORK_ENV, "")
    if chosen:
        chosen = Path(expanduser(chosen)).resolve()
        if not (chosen / FORK_API).is_file():
            return f"{FORK_ENV}={chosen} has no {FORK_API}"
        sys.path.insert(0, str(chosen))
    try:
        import_module("psd_tools.composite.text")
    except ImportError as error:
        return f"no psd-tools fork with psd-tools[text] ({error}): run under {FORK_VENV}"
    return ""


FORK_WHY = use_fork()

try:                                                    # the real config.py when present (never printed)
    import config  # noqa: F401
except ImportError:
    fake = types.ModuleType("config")
    fake.TOKEN, fake.GIT_TOKEN, fake.ADMINS, fake.browser_cookies = "", "", [], False
    sys.modules["config"] = fake


def pytest_configure(config):
    if FORK_WHY:
        raise pytest.UsageError(FORK_WHY)
    missing = [f"{key}={environ[key]}: no such path" for key in TC_PATHS if environ.get(key) and not Path(environ[key]).exists()]
    if missing:                                           # set but wrong: stop, not skip
        raise pytest.UsageError("\n".join(missing))
    lacking = [f"{tool} (brew install {formula})" for tool, formula in TOOLS.items() if not shutil.which(tool)]
    if lacking:                                           # checked once: no test skips for a tool
        raise pytest.UsageError("not on PATH: " + ", ".join(lacking))
    if not (CLONE / ".git").exists():
        raise pytest.UsageError(f"no card repo clone at {CLONE} (AUTOCROISSANT_LOCAL_DIR)")
    config.addinivalue_line("filterwarnings", "ignore::DeprecationWarning:discord")
    config.addinivalue_line("markers", "oxipng: runs the real oxipng (no_oxipng's stub off)")
    config.addinivalue_line("markers", "header_stars: psd_details' header plans all 10 star counts (few_header_stars' 3 off)")


# ========================
# Helpers
# ========================
def blob_sha_bytes(data: bytes) -> str:
    """git's blob sha of some bytes (independent of the code under test's own helper)."""
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def blob_sha(path) -> str:
    """git's blob sha of a file (what the GitHub tree lists)."""
    return blob_sha_bytes(Path(path).read_bytes())


PLAIN_DOOMER = "38c55bd361b60b99b9521c43cf9577213845a4ce^"   # Mini Doomer before queue A's "Ability FX: (BOT)" badge layer
BADGED = "e51649baa4143fde379f9fa9d860a0003bfb4380"          # main before the 31 badge FX layers were retired (2026-10-25)
RING = "0ade8c3d"       # TTSCardMaker's ring font (2026-11-04, design B, Roboto labels): (1CD) 2040 -> 2658 units (re-pin here)


def no_lazy() -> dict:
    """git's env for reading the blobless Desktop clone: a missing blob fails instead of being fetched from GitHub."""
    return {**environ, "GIT_NO_LAZY_FETCH": "1"}


class Pinned(type(Path())):
    """The card repo at CARDS_REV: `pinned / rel` writes rel (a file or a folder) from the pin on first use."""
    batch = None                                          # this process's git cat-file --batch
    done: set = set()                                     # folders written by this process

    def __truediv__(self, key):
        path = super().__truediv__(key)
        _materialize(self, path.relative_to(self).as_posix())
        return Path(path)


def _blob(root: Path, spec: str):
    """(type, bytes) of rev:path through one cat-file --batch per process; None when the pin has no such path."""
    import subprocess
    if Pinned.batch is None:
        Pinned.batch = subprocess.Popen(["git", "-C", str(root), "cat-file", "--batch"], stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, env=no_lazy())
    pipe = Pinned.batch
    pipe.stdin.write(spec.encode() + b"\n")
    pipe.stdin.flush()
    head = pipe.stdout.readline().split()
    if head[-1] == b"missing":
        sha = subprocess.run(["git", "-C", str(root), "rev-parse", "-q", "--verify", spec], capture_output=True, text=True).stdout.strip()
        if sha:
            raise RuntimeError(f"{spec} is in the pin's tree, but neither {CLONE} nor {HISTORY} has its blob: missing blob {sha}")
        return None
    data = pipe.stdout.read(int(head[2]))
    pipe.stdout.read(1)
    return head[1].decode(), data


def _write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{getpid()}")
    temp.write_bytes(data)
    temp.replace(path)                                    # xdist workers write the same bytes


def _materialize(root: Path, rel: str) -> None:
    import subprocess
    root = Path(root)                                     # plain paths: no recursion
    path = root / rel
    if path.is_file() or rel in Pinned.done or rel in ("", "."):
        return
    found = _blob(root, f"{CARDS_REV}:{rel}")
    if found is None:
        return                                            # not in the pin: the test sees no file
    kind, data = found
    if kind == "blob":
        return _write(path, data)
    listing = subprocess.run(["git", "-C", str(root), "ls-tree", "-r", "-z", "--name-only", CARDS_REV, "--", rel],
                             capture_output=True, check=True, env=no_lazy()).stdout.decode()
    for each in filter(None, listing.split("\0")):
        if not (root / each).is_file():
            _write(root / each, _blob(root, f"{CARDS_REV}:{each}")[1])
    Pinned.done.add(rel)


def pinned_clone() -> Pinned:
    """A git repo on the clone's objects (and the history clone's) at CARDS_REV, files written on first use."""
    import subprocess
    root = Pinned(PINNED / CARDS_REV[:12])
    if not Path(root, ".git", "index").exists():
        temp = PINNED / f".{CARDS_REV[:12]}.{getpid()}"
        shutil.rmtree(temp, ignore_errors=True)
        subprocess.run(["git", "init", "-q", str(temp)], check=True)
        stores = [p / "objects" for p in (CLONE / ".git", HISTORY) if (p / "objects").is_dir()]
        (temp / ".git" / "objects" / "info" / "alternates").write_text("".join(f"{p}\n" for p in stores))
        (temp / ".git" / "HEAD").write_text(CARDS_REV + "\n")
        if subprocess.run(["git", "-C", str(temp), "read-tree", CARDS_REV], capture_output=True, env=no_lazy()).returncode:
            shutil.rmtree(temp, ignore_errors=True)
            raise pytest.UsageError(f"neither {CLONE} nor {HISTORY} has the card repo at {CARDS_REV[:8]}")
        try:
            temp.rename(root)
        except OSError:                                   # another worker made it
            shutil.rmtree(temp, ignore_errors=True)
    for rel in PINNED_BASE:                               # what the code under test reads straight from disk
        _materialize(root, rel)
    return root


def card_blob(clone, spec: str) -> bytes:
    """A card repo object (rev:path or a blob id) from the clone's objects. Missing fails, naming the blob: the run's
    summary gives one fetch for them all."""
    import subprocess
    done = subprocess.run(["git", "-C", str(clone), "cat-file", "blob", spec], capture_output=True, env=no_lazy())
    if done.returncode == 0:
        return done.stdout
    sha = subprocess.run(["git", "-C", str(clone), "rev-parse", "-q", "--verify", spec], capture_output=True, text=True,
                         env=no_lazy()).stdout.strip()
    pytest.fail(f"{CLONE} lacks {spec}: " + (f"missing blob {sha}" if sha else f"not in its history or {HISTORY}"), pytrace=False)


def at_rev(clone, rev: str, rel: str) -> bytes:
    """A card repo file at rev, from the clone's history (card_blob)."""
    return card_blob(clone, f"{rev}:{rel}")


def painted(png: bytes, box: tuple, rows: float) -> bytes:
    """A PNG with the top share (rows) of a box's colours inverted, alpha kept: a render broken there."""
    from io import BytesIO
    import numpy as np
    from PIL import Image
    image = np.array(Image.open(BytesIO(png)))
    x0, y0, x1, y1 = box
    y1 = y0 + int((y1 - y0) * rows)
    image[y0:y1, x0:x1, :3] = 255 - image[y0:y1, x0:x1, :3]
    out = BytesIO()
    Image.fromarray(image).save(out, "PNG")
    return out.getvalue()


class Reply:
    """A requests.Response stand-in for the fake GitHubs: a status, raw bytes or a JSON body, headers."""
    def __init__(self, status=200, data=b"", body=None, headers=None):
        self.status_code, self._body = status, body
        self.content = data if body is None else json.dumps(body).encode("utf-8")
        self.headers = headers or {}
        self.encoding = None

    @property
    def ok(self):
        return self.status_code < 400

    @property
    def text(self):
        return self.content.decode(self.encoding or "utf-8")

    def json(self):
        return self._body if self._body is not None else json.loads(self.content)

    def raise_for_status(self):
        if self.status_code >= 400:
            from requests import HTTPError
            raise HTTPError(f"{self.status_code} Error")

    def iter_content(self, size):
        for i in range(0, len(self.content), size):
            yield self.content[i:i + size]

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _digests() -> dict:
    return {p: hashlib.sha1((REPO / p).read_bytes()).hexdigest() for p in LIVE if (REPO / p).exists()}


# ========================
# Fixtures
# ========================
@pytest.fixture(scope="session", autouse=True)
def live_files_untouched():
    """Fails the run if any test changed the checkout's reminder store."""
    before = _digests()
    yield
    assert _digests() == before, "a test wrote a live store"


@pytest.fixture(autouse=True)
def offline(monkeypatch, tmp_path):
    """No network, cwd in a temp dir so cwd-relative saves (card_data/, json maps) land there, and no history clone (for
    /query as_of's images) unless a test gives one (the machine's own never counts). Also: no retry sleeps, pools of 2,
    and utils' queues empty (tests that count waits override the sleep stub)."""
    def refuse(*args, **kwargs):
        raise RuntimeError("network access in an offline test")
    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)
    monkeypatch.setenv("GIT_ALLOW_PROTOCOL", "file:ssh")                # git: no https (ssh tests fake their ssh)
    monkeypatch.chdir(tmp_path)
    for name in ("commands.repo_history", "commands.git_commit"):        # /query as_of's images, land's has_blobs
        if name in sys.modules:
            monkeypatch.setattr(sys.modules[name], "HISTORY_CLONE", str(tmp_path / "no-history-clone"))
    import commands.github_api as ga                                     # get_with_retry's waits
    monkeypatch.setattr(ga, "sleep", lambda seconds: None)
    if "commands.card_preview_branch" in sys.modules:                    # the publisher's backoff
        monkeypatch.setattr(sys.modules["commands.card_preview_branch"], "sleep", lambda seconds: None)
    for name in POOLED:                                                  # half of 4 cores: 2 processes per pool
        if name in sys.modules:
            monkeypatch.setattr(sys.modules[name], "cpu_count", lambda: 4)
    utils = sys.modules.get("commands.utils")
    queues = [getattr(utils, name) for name in QUEUES] if utils else []
    for queue in queues:                                                 # main imports them by name: clear, never replace
        queue.clear()
    yield
    for queue in queues:
        queue.clear()


@pytest.fixture(autouse=True)
def no_oxipng(request, monkeypatch):
    """png_export.optimize and band give None (PNGs kept as exported) unless the test is marked oxipng: about 0.7 s and
    3 s a card PNG. Child processes still run the real ones."""
    png_export = sys.modules.get("commands.png_export")
    if png_export and not request.node.get_closest_marker("oxipng"):
        monkeypatch.setattr(png_export, "optimize", lambda data: None)
        monkeypatch.setattr(png_export, "band", lambda data: None)


@pytest.fixture(scope="session", autouse=True)
def few_header_stars():
    """psd_details' header plans star counts 1-3 for the run, module fixtures' too (4-10 cost most of a gather).
    Yields the real count."""
    from commands import psd_details
    every = psd_details.HEADER_STARS
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(psd_details, "HEADER_STARS", 3)
        yield every


@pytest.fixture(autouse=True)
def header_stars(request, monkeypatch, few_header_stars):
    """A test marked header_stars plans every count (its own gathers; module fixtures keep 3)."""
    if request.node.get_closest_marker("header_stars"):
        from commands import psd_details
        monkeypatch.setattr(psd_details, "HEADER_STARS", few_header_stars)


@pytest.fixture(scope="session")
def clone():
    """The card repo at CARDS_REV (pinned_clone), so card pushes never move a test."""
    return pinned_clone()


def pytest_terminal_summary(terminalreporter):
    """One batched fetch for every blob the failures name (card_blob, the pin's _blob)."""
    reports = terminalreporter.stats.get("failed", []) + terminalreporter.stats.get("error", [])
    ids = sorted({sha for r in reports for sha in re.findall(r"missing blob ([0-9a-f]{40})", str(getattr(r, "longrepr", "")))})
    if ids:
        terminalreporter.write_line(f"{len(ids)} card blobs missing from {CLONE}; fetch them in one batch:")
        terminalreporter.write_line(f"printf '%s\\n' {' '.join(ids)} | {FETCH.format(clone=quote(str(CLONE)))}")


def pytest_xdist_auto_num_workers(config):
    """-n auto runs at most 6 workers: every worker spawns git on the 16 GB card clone (10-25: 400+ git processes ran the Mac out of memory).
    AC_TEST_WORKERS raises it on a quiet Mac (each worker peaks near 3 GB)."""
    return int(environ.get("AC_TEST_WORKERS") or 0) or min(6, cpu_count() or 1)
