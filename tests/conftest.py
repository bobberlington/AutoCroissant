# Run: ~/.venvs/ac-py314-psd125/bin/python -m pytest -q tests   (offline; never writes the pickles)
# The text-engine tests (psd_edit, card_bake) need the psd-tools fork and psd-tools[text]; see find_fork:
#   AC_PSD_TOOLS_SRC=<fork>/src ~/Documents/Tools/psd-tools/.venv/bin/python -m pytest -q tests/test_psd_edit.py tests/test_card_bake.py
import hashlib
import json
import socket
import sys
import types
from importlib.util import find_spec
from os import environ
from os.path import expanduser
from pathlib import Path

import pytest

# ========================
# Configuration
# ========================
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from global_config import ALIAS_PKL, DETAILS_JSON, LOCAL_DIR_LOC, OLD_STATS_PKL, REMIND_PKL, STATS_PKL  # noqa: E402

CLONE = Path(expanduser(LOCAL_DIR_LOC))                  # PSD tests skip without it
PICKLES = (STATS_PKL, OLD_STATS_PKL, ALIAS_PKL, REMIND_PKL, DETAILS_JSON)   # the details file counts as one


# ========================
# psd-tools fork
# ========================
FORK_ENV = "AC_PSD_TOOLS_SRC"                           # a psd-tools fork's src dir (the one holding psd_tools/)
FORK_DEFAULTS = ("~/Documents/Tools/psd-tools/src", "/tmp/wt-cards/src")   # tried in order, each only with the fork API
FORK_API = Path("psd_tools") / "api" / "text_edit.py"   # what upstream psd-tools lacks
FORK_DEPS = {"uharfbuzz": "uharfbuzz", "freetype": "freetype-py"}   # psd-tools[text]: module -> package
FORK_VENV = "~/Documents/Tools/psd-tools/.venv/bin/python"   # has them


def find_fork() -> tuple:
    """
    Pick the psd_tools the text-engine tests run on, before anything imports it: $AC_PSD_TOOLS_SRC, else the installed
    psd_tools if it has the fork API, else the first FORK_DEFAULTS dir that has it (put first on sys.path).

    Returns:
        (src dir, "") when usable, ("", why not) otherwise; why is the skip reason.
    """
    spec = find_spec("psd_tools")
    installed = Path(spec.origin).resolve().parents[1] if spec and spec.origin else None
    chosen = environ.get(FORK_ENV, "")
    if chosen:
        chosen = Path(expanduser(chosen)).resolve()
        if not (chosen / FORK_API).is_file():
            return "", f"{FORK_ENV}={chosen} has no {FORK_API}"
    elif installed and (installed / FORK_API).is_file():
        chosen = installed
    else:
        chosen = next((Path(expanduser(d)).resolve() for d in FORK_DEFAULTS if (Path(expanduser(d)) / FORK_API).is_file()), None)
        if chosen is None:
            return "", (f"no psd-tools fork: the installed psd_tools and {', '.join(FORK_DEFAULTS)} lack {FORK_API}; "
                        f"set {FORK_ENV} to a fork src dir (a worktree of its cards branch)")
    missing = [package for module, package in FORK_DEPS.items() if find_spec(module) is None]
    if missing:
        return "", f"fork at {chosen}, but this Python lacks {' and '.join(missing)} (psd-tools[text]): run under {FORK_VENV}"
    if chosen != installed:
        if "psd_tools" in sys.modules:
            return "", f"psd_tools was imported before conftest could put {chosen} first"
        sys.path.insert(0, str(chosen))
    return str(chosen), ""


FORK_SRC, FORK_WHY = find_fork()

try:                                                    # the real config.py when present (never printed)
    import config  # noqa: F401
except ImportError:
    fake = types.ModuleType("config")
    fake.TOKEN, fake.GIT_TOKEN, fake.ADMINS, fake.browser_cookies = "", "", [], False
    sys.modules["config"] = fake


def pytest_configure(config):
    config.addinivalue_line("filterwarnings", "ignore::DeprecationWarning:discord")


# ========================
# Helpers
# ========================
def blob_sha_bytes(data: bytes) -> str:
    """git's blob sha of some bytes (independent of the code under test's own helper)."""
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def blob_sha(path) -> str:
    """git's blob sha of a file (what the GitHub tree lists)."""
    return blob_sha_bytes(Path(path).read_bytes())


def text_engine() -> bool:
    """psd_edit's probe (not card_bake's: it imports requests)."""
    from commands.psd_edit import text_engine as probe
    return probe()


def fork_reason() -> str:
    """The skip reason of a test that needs the fork's text engine."""
    return FORK_WHY or f"the fork at {FORK_SRC} found, but psd_tools.composite.text does not import"


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

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _digests() -> dict:
    return {p: hashlib.sha1((REPO / p).read_bytes()).hexdigest() for p in PICKLES if (REPO / p).exists()}


# ========================
# Fixtures
# ========================
@pytest.fixture(scope="session", autouse=True)
def pickles_untouched():
    """Fails the run if any test changed a live pickle."""
    before = _digests()
    yield
    assert _digests() == before, "a test wrote a live pickle"


@pytest.fixture(autouse=True)
def offline(monkeypatch, tmp_path):
    """No network, and cwd in a temp dir so cwd-relative saves (pickles, json maps) land there."""
    def refuse(*args, **kwargs):
        raise RuntimeError("network access in an offline test")
    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)
    monkeypatch.chdir(tmp_path)


@pytest.fixture(scope="session")
def real_stats():
    """The live stats.pkl, loaded read-only through StatsDatabase (never saved)."""
    path = REPO / STATS_PKL
    if not path.exists():
        pytest.skip("no stats.pkl")
    import commands.card_models as cm
    db = cm.StatsDatabase()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(cm, "STATS_PKL", str(path))
        db._load_stats()
    db.save = None                                       # read-only: a save would raise
    return db


@pytest.fixture(scope="session")
def clone():
    if not (CLONE / "Types").is_dir():
        pytest.skip(f"no card repo clone at {CLONE}")
    return CLONE
