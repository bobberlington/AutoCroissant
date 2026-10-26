# Run: ~/.venvs/ac-py314-fork/bin/python -m pytest -q -n auto tests   (offline; never writes the reminder store)
# The fork venv has psd-tools[text], which the bake tests need (see find_fork); -n 0 runs one test serially.
import hashlib
import json
import socket
import sys
import types
from importlib.util import find_spec
from os import cpu_count, environ, getpid
from os.path import expanduser
from pathlib import Path

import pytest

# ========================
# Configuration
# ========================
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from global_config import LOCAL_DIR_LOC, REMIND_JSON  # noqa: E402

CLONE = Path(expanduser(LOCAL_DIR_LOC))                  # the card repo checkout: its objects (and its site code)
CARDS_REV = "a4d137bf06cd372649090a32644023abee8719ea"   # the card repo commit real-card tests read (re-pin: tests/make_goldens.py)
HISTORY = Path(expanduser("~/Documents/Tools/wt/scratch/tc-history.git"))   # full history (no blobs) for revs the clone lacks
PINNED = Path(expanduser("~/.cache/autocroissant-tests"))                   # the pinned trees, one per CARDS_REV
PINNED_BASE = ("Types", "MDW/Fonts", "MDW/Croissant's Special Folder/Satus Effect Colors.psd", "site/Card Edits.json")
LIVE = (REMIND_JSON,)                                    # the checkout's reminder store, never written by a test
GIT_CONFIG = {"core.hooksPath": "/dev/null",              # no machine hooks in fixture repos
              "core.looseCompression": "0"}               # fixture repos of real PSDs: no zlib on every add
POOLED = ("commands.card_check", "commands.font_check", "commands.card_guards")   # process pools capped per worker
QUEUES = ("dispatch_queue", "edit_queue", "command_queue")                         # utils' deques, empty per test


def _git_config():
    """GIT_CONFIG for every git this run starts (kept user identity, unlike GIT_CONFIG_GLOBAL=/dev/null)."""
    start = int(environ.get("GIT_CONFIG_COUNT", "0"))
    for i, (key, value) in enumerate(GIT_CONFIG.items(), start):
        environ[f"GIT_CONFIG_KEY_{i}"], environ[f"GIT_CONFIG_VALUE_{i}"] = key, value
    environ["GIT_CONFIG_COUNT"] = str(start + len(GIT_CONFIG))


_git_config()
environ["PSD_DIRTY_VERIFY"] = "1"                         # gate_dirty renders whole too: raises on any difference


# ========================
# psd-tools fork
# ========================
FORK_ENV = "AC_PSD_TOOLS_SRC"                           # a psd-tools fork's src dir (the one holding psd_tools/)
FORK_DEFAULTS = ("~/Documents/Tools/psd-tools/src", "/tmp/wt-cards/src")   # tried in order
FORK_API = Path("psd_tools") / "api" / "text_edit.py"   # what upstream psd-tools lacks
FORK_DEPS = {"uharfbuzz": "uharfbuzz", "freetype": "freetype-py"}   # psd-tools[text]: module -> package
FORK_VENV = "~/Documents/Tools/psd-tools/.venv/bin/python"   # has them


def find_fork() -> tuple:
    """
    The fork's src dir: $AC_PSD_TOOLS_SRC, else the installed psd_tools, else FORK_DEFAULTS (put first on sys.path).

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


PLAIN_DOOMER = "38c55bd361b60b99b9521c43cf9577213845a4ce^"   # Mini Doomer before queue A's "Ability FX: (BOT)" badge layer
BADGED = "e51649baa4143fde379f9fa9d860a0003bfb4380"          # main before the 31 badge FX layers were retired (2026-10-25)


def salt_font(clone) -> bool:
    """Whether the clone's stoogedings has the one-colour Stylistic Alternates (TTSCardMaker work/badgealt)."""
    from fontTools.ttLib import TTFont
    font = clone / "MDW/Fonts/stoogedings.ttf"
    return font.exists() and "salt" in {r.FeatureTag for r in TTFont(str(font), lazy=True)["GSUB"].table.FeatureList.FeatureRecord}


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
        if subprocess.run(["git", "-C", str(root), "rev-parse", "-q", "--verify", spec], capture_output=True).returncode == 0:
            raise RuntimeError(f"{spec} is in the pin's tree, but neither {CLONE} nor {HISTORY} has its blob")
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
    import shutil
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
            pytest.skip(f"neither {CLONE} nor {HISTORY} has the card repo at {CARDS_REV[:8]}")
        try:
            temp.rename(root)
        except OSError:                                   # another worker made it
            shutil.rmtree(temp, ignore_errors=True)
    for rel in PINNED_BASE:                               # what the code under test reads straight from disk
        _materialize(root, rel)
    return root


def smallest(clone: Pinned, folder: str, n: int) -> list:
    """The folder's n smallest card PSDs at the pin (written out): a fast parse."""
    import subprocess
    listing = subprocess.run(["git", "-C", str(clone), "ls-tree", "-r", "-l", "-z", CARDS_REV, "--", folder],
                             capture_output=True, check=True, env=no_lazy()).stdout.decode().split("\0")
    sized = sorted((int(line.split()[3]), line.split("\t", 1)[1]) for line in listing if line.endswith(".psd"))
    return [clone / rel for _, rel in sized[:n]]


def at_rev(clone, rev: str, rel: str) -> bytes:
    """A card repo file at rev, from the clone's history; skips the test when the history lacks it."""
    import subprocess
    try:
        return subprocess.run(["git", "-C", str(clone), "show", f"{rev}:{rel}"], capture_output=True, check=True,
                              env=no_lazy()).stdout
    except (subprocess.CalledProcessError, OSError):
        pytest.skip(f"the clone's history lacks {rev}:{rel}")


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


def text_engine() -> bool:
    """psd_edit's probe (not card_bake's: it imports requests)."""
    from commands.psd_edit import text_engine as probe
    return probe()


def story_sync() -> bool:
    """The fork's text edits write the Txt2 story too (fix/txt2-sync): psd_edit's probe."""
    from commands.psd_edit import story_sync as probe
    return probe()


def rounds_bytes() -> bool:
    """The fork rounds composite bytes (round 4's fix/composite-byte-rounding): 3 at opacity 128 over black is 1.506."""
    from PIL import Image
    from psd_tools import PSDImage
    from psd_tools.api.layers import PixelLayer
    psd = PSDImage.new("RGB", (1, 1), color=(0, 0, 0))
    psd.append(PixelLayer.frompil(Image.new("RGB", (1, 1)), psd))
    top = PixelLayer.frompil(Image.new("RGB", (1, 1), (3, 3, 3)), psd)
    top.opacity = 128
    psd.append(top)
    return psd.composite(force=True).getpixel((0, 0))[0] == 2


def bevel_textures() -> bool:
    """The fork draws a bevel's texture (round 4's fix/bevel-texture)."""
    from psd_tools.api.effects import BevelEmboss
    return hasattr(BevelEmboss, "texture_pattern")


def colour_glyphs() -> bool:
    """The fork draws OpenType-SVG colour glyphs (stoogedings' icons) in colour (round 6's fix/r6-color-glyphs)."""
    from dataclasses import fields
    try:
        from psd_tools.composite.text.raster import Options
    except ImportError:                                     # no text engine (stock psd-tools)
        return False
    return "color_glyphs" in {field.name for field in fields(Options)}


def saves_preview() -> bool:
    """The fork stores a given preview (round 3's perf/save-with-preview, tag cards-20261021)."""
    from inspect import signature
    from psd_tools import PSDImage
    return "preview" in signature(PSDImage.save).parameters


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
    assert _digests() == before, "a test wrote a live reminder store"


@pytest.fixture(autouse=True)
def offline(monkeypatch, tmp_path):
    """No network, cwd in a temp dir so cwd-relative saves (card_data/, json maps) land there, and no history clone (for
    /query as_of's images) unless a test gives one (the machine's own never counts). Also: no retry sleeps, pools of 2,
    and utils' queues empty (tests that count waits override the sleep stub)."""
    def refuse(*args, **kwargs):
        raise RuntimeError("network access in an offline test")
    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)
    monkeypatch.chdir(tmp_path)
    if "commands.repo_history" in sys.modules:                           # /query as_of's images
        monkeypatch.setattr(sys.modules["commands.repo_history"], "HISTORY_CLONE", str(tmp_path / "no-history-clone"))
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


@pytest.fixture(scope="session")
def clone():
    """The card repo at CARDS_REV (pinned_clone), so card pushes never move a test."""
    if not (CLONE / ".git").exists():
        pytest.skip(f"no card repo clone at {CLONE}")
    return pinned_clone()


def pytest_xdist_auto_num_workers(config):
    """-n auto runs at most 6 workers: every worker spawns git on the 16 GB card clone (10-25: 400+ git processes ran the Mac out of memory).
    AC_TEST_WORKERS raises it on a quiet Mac (each worker peaks near 3 GB)."""
    return int(environ.get("AC_TEST_WORKERS") or 0) or min(6, cpu_count() or 1)
