# Run: ~/.venvs/ac-py314-psd125/bin/python -m pytest -q tests   (offline; never writes the pickles)
import hashlib
import json
import socket
import sys
import types
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
    """psd_edit's probe (not card_bake's: the fork venv has no requests)."""
    from commands.psd_edit import text_engine as probe
    return probe()


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
