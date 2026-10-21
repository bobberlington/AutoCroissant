# Run: /Users/michaelsrouji/.venvs/ac-py314-psd125/bin/python -m pytest -q tests   (offline; never writes the pickles)
import hashlib
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
CLONE = Path(expanduser("~/Desktop/TTSCardMaker"))      # LOCAL_DIR_LOC; PSD tests skip without it
PICKLES = ("stats.pkl", "old_stats.pkl", "aliases.pkl", "reminder.pkl")

sys.path.insert(0, str(REPO))
try:                                                     # the real config.py when present (never printed)
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
def blob_sha(path) -> str:
    """git's blob sha of a file (what the GitHub tree lists)."""
    data = Path(path).read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


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
    path = REPO / "stats.pkl"
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
