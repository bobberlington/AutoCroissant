"""Helpers and fixtures shared by several test modules (a cut never deletes them with a test)."""
import ast
import hashlib
import json
import os
import re
import shutil
import subprocess
from copy import deepcopy
from datetime import datetime, timezone
from functools import lru_cache
from io import BytesIO
from os import environ
from pathlib import Path
from subprocess import run as run_process
from sys import executable
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image, ImageDraw
from psd_tools import PSDImage
from psd_tools.api.layers import PixelLayer

import commands.card_bake as cb
import commands.card_data as cd
import commands.card_db as cdb
import commands.changelog as cl
import commands.git_commit as gc
import commands.github_api as ga
import commands.psd_edit as pe
import commands.type_emojis as te
from conftest import REPO, Reply, blob_sha_bytes
from golden_cards import VOCAB, golden_file


# ========================
# From test_card_bake
# ========================
DOOMER = "Auxiliary/Minions/Mini_Doomer.psd"

FREEZER = "Field/1 Stars/The_Freezer.psd"

SNACK = "Items/Heal/2 Stars/A_Snack.psd"

AT = "2026-10-20T18:41:54Z"

ABOUT = "Card data fixes from the website's Edit button."

TOKEN = "test-token"


class FakeGitHub:
    """A repo with git objects: blobs, trees, commits and one branch. recent / conflicts / broken simulate the races."""

    def __init__(self, repo: str, files: dict, author="web", message="Website edit: test"):
        self.repo = repo
        self.blobs: dict = {}
        self.trees: dict = {}
        self.commits: dict = {}
        self.recent: set = set()            # paths pushed "in the last 30 minutes"
        self.conflicts = 0                  # PATCH ref answers conflict_status this many times, someone else's push landing first
        self.stuck = False                  # PATCH ref answers 422 with the branch unmoved (a ruleset)
        self.conflict_status = 422          # ... 409 or 422 (both mean "not a fast forward")
        self.conflict_on = ""               # ... only for commits whose message starts so ("" = any)
        self.broken: dict = {}              # blob sha -> the bytes served instead
        self.down: set = set()              # blob shas answering 503
        self.rate_limited = False           # POST git/blobs answers a rate-limit 403
        self.forbidden = False              # POST git/blobs answers 403 without rate-limit headers: the token can't write
        self.calls: list = []
        self.head = self.commit(files, author, message)

    def commit(self, files: dict, author: str, message: str, parents=None) -> str:
        """A commit of {path: bytes} on the branch (parents default to the head), returning its sha."""
        tree = {}
        for path, data in files.items():
            self.blobs[blob_sha_bytes(data)] = data
            tree[path] = ("100644", blob_sha_bytes(data))
        tree_sha = self._tree(tree)
        parents = [self.head] if parents is None and self.commits else list(parents or [])
        sha = hashlib.sha1(json.dumps([tree_sha, parents, message, author]).encode()).hexdigest()
        self.commits[sha] = {"sha": sha, "tree": {"sha": tree_sha}, "parents": [{"sha": p} for p in parents],
                             "message": message, "author": {"name": author, "email": f"{author}@x"}}
        self.head = sha
        return sha

    def files_at(self, sha: str) -> dict:
        """{path: bytes} at a commit."""
        return {path: self.blobs[blob] for path, (_, blob) in self.trees[self.commits[sha]["tree"]["sha"]].items()}

    def _tree(self, tree: dict) -> str:
        sha = hashlib.sha1(json.dumps(sorted(tree.items())).encode()).hexdigest()
        self.trees[sha] = dict(tree)
        return sha

    def get(self, url, **kwargs):
        return self.request("GET", url, **kwargs)

    def request(self, method, url, headers=None, timeout=None, json=None, params=None, stream=False):
        route, _, query = url.split(f"/repos/{self.repo}/", 1)[1].partition("?")
        self.calls.append((method, route))
        if method == "GET" and route == "git/ref/heads/main":
            return Reply(body={"object": {"sha": self.head}})
        if method == "GET" and route.startswith("git/commits/"):
            sha = route.rsplit("/", 1)[1]
            return Reply(body=self.commits[sha]) if sha in self.commits else Reply(404, body={"message": "Not Found"})
        if method == "GET" and route.startswith("git/trees/"):
            sha = route.rsplit("/", 1)[1]
            if sha not in self.trees:
                return Reply(404, body={"message": "Not Found"})
            return Reply(body={"sha": sha, "truncated": False, "tree": [
                {"path": p, "mode": mode, "type": "blob", "sha": blob} for p, (mode, blob) in sorted(self.trees[sha].items())]})
        if method == "GET" and route.startswith("git/blobs/"):
            sha = route.rsplit("/", 1)[1]
            if sha not in self.blobs:
                return Reply(404, body={"message": "Not Found"})
            if sha in self.down:
                return Reply(503, body={"message": "Service Unavailable"})
            return Reply(data=self.broken.get(sha, self.blobs[sha]))
        if method == "GET" and route == "activity":
            now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            return Reply(body=[{"before": "b" * 40, "after": self.head, "timestamp": now}] if self.recent else [])
        if method == "GET" and route.startswith("compare/"):
            return Reply(body={"files": [{"filename": path} for path in sorted(self.recent)]})
        if method == "POST" and route == "git/blobs":
            from base64 import b64decode
            if self.rate_limited:
                return Reply(403, body={"message": "API rate limit exceeded"}, headers={"x-ratelimit-remaining": "0"})
            if self.forbidden:
                return Reply(403, body={"message": "Resource not accessible by integration"})
            data = b64decode(json["content"])
            self.blobs[blob_sha_bytes(data)] = data
            return Reply(201, body={"sha": blob_sha_bytes(data)})
        if method == "POST" and route == "git/trees":
            tree = dict(self.trees[json["base_tree"]])
            for entry in json["tree"]:
                if entry["sha"] is None:            # GitHub deletes the path
                    tree.pop(entry["path"], None)
                else:
                    tree[entry["path"]] = (entry["mode"], entry["sha"])
            return Reply(201, body={"sha": self._tree(tree)})
        if method == "POST" and route == "git/commits":
            sha = hashlib.sha1(json_dumps(json).encode()).hexdigest()
            self.commits[sha] = {"sha": sha, "tree": {"sha": json["tree"]}, "parents": [{"sha": p} for p in json["parents"]],
                                 "message": json["message"], "author": dict(json.get("author") or {"name": "token"})}
            return Reply(201, body={"sha": sha})
        if method == "PATCH" and route == "git/refs/heads/main":
            commit = self.commits.get(json["sha"])
            if self.stuck:
                return Reply(422, body={"message": "Repository rule violations found"})
            if self.conflicts and (commit or {}).get("message", "").startswith(self.conflict_on):
                self.conflicts -= 1
                self.commit({**self.files_at(self.head), "elsewhere.txt": b"%d" % self.conflicts}, "web", "Website edit: elsewhere")
                return Reply(self.conflict_status, body={"message": "Update is not a fast forward"})
            if not commit or [p["sha"] for p in commit["parents"]] != [self.head]:
                return Reply(422, body={"message": "Update is not a fast forward"})
            self.head = json["sha"]
            return Reply(body={"object": {"sha": self.head}})
        return Reply(404, body={"message": f"no route for {method} {route}"})


def json_dumps(value) -> str:
    return json.dumps(value, sort_keys=True)


def _card_files(clone, rel: str) -> dict:
    png = rel[:len(rel) - 4] + ".png"
    if not (clone / rel).exists() or not (clone / png).exists():
        pytest.skip(f"{rel} not in the clone")
    return {rel: (clone / rel).read_bytes(), png: (clone / png).read_bytes()}


def _edits(cards: dict) -> bytes:
    return cb.dump_edits({"about": ABOUT, "cards": cards}).encode("utf-8")

EDITS = {
    "Beldam": {"series": "Mario", "_by": "web", "_at": AT},                 # no PSD field: not looked at
    "Mini Doomer": {"hp": 5, "note": "checked", "_by": "web", "_at": AT},   # hp 5 is on the card: redundant, dropped
    "Nobody": {"hp": 3, "_by": "web", "_at": AT},                           # no PSD
    "The Freezer": {"stars": 2, "_by": "web", "_at": AT},                   # starred by its folder: moving it collides
    "A Snack": {"hp": 1, "_by": "web", "_at": AT},                          # numbered PNGs share its PSD
    "A Snack2": {"ability": "Breakfast.", "_by": "web", "_at": AT},         # a numbered PNG itself
}


@pytest.fixture
def files(clone):
    vocab = json.loads(VOCAB.read_text(encoding="utf-8"))
    out = {f"Types/{t}.png": t.encode() for t in vocab["all_types"]}
    out.update({f"Types/Stars/{s}.png": s.encode() for s in vocab["stars"]})
    out[cb.ICON_MANIFEST_PATH] = json.dumps(vocab["manifest"]).encode()
    out.update(_card_files(clone, DOOMER))
    out.update(_card_files(clone, FREEZER))
    out["Field/2 Stars/The_Freezer.png"] = b"another card's PNG"   # in the way of its move into "2 Stars"
    out[SNACK] = out[DOOMER]                    # read by the redundancy check before the duplicate gate (hp 5, not 1)
    out[SNACK[:len(SNACK) - 4] + ".png"] = b"not fetched"
    out["Items/Heal/2 Stars/A_Snack2.png"] = b"a duplicate"
    out["MDW/Templates/Creature_Template.psd"] = b"excluded folder"
    out[cb.EDITS_PATH] = _edits(EDITS)
    return out


@pytest.fixture
def github(monkeypatch, tmp_path, files, clone):
    fake = FakeGitHub(cb.CARD_REPO, files)
    monkeypatch.setattr(ga, "get", fake.get)
    monkeypatch.setattr(ga, "GIT_TOKEN", "")
    monkeypatch.setattr(ga, "HEADERS", {})
    monkeypatch.setattr(gc, "request", fake.request)
    monkeypatch.setattr(gc, "refresh", lambda: True)
    monkeypatch.setenv("GITHUB_TOKEN", TOKEN)
    monkeypatch.setenv(cb.WORK_ENV, str(tmp_path))
    monkeypatch.setenv(cb.FONT_DIRS_ENV, str(clone / pe.FONTS_DIR))
    monkeypatch.setattr(gc, "cant_write", False)
    return fake

CARD_PACKAGES = {"psd-tools": {"psd_tools", "attr", "attrs", "numpy", "PIL", "scipy", "aggdraw", "uharfbuzz", "freetype"},
                 "fonttools": {"fontTools"}, "requests": {"requests", "certifi", "charset_normalizer", "idna", "urllib3"}}
# requirements-cards.txt's packages: what they import (dependencies included); oxipng is the CLI (cards-python)

ALLOWED_IMPORTS = set().union(*CARD_PACKAGES.values())      # what the Actions install

SKIP_IMPORTS = {"commands", "global_config", "sitecustomize", "usercustomize", "cython_runtime"}


def third_party_imports(code: str, cwd) -> set:
    """The non-stdlib top-level modules an import leaves in sys.modules (a copy of the repo at cwd, no config.py)."""
    probe = (f"import sys; {code}; tops = {{m.split('.')[0] for m in sys.modules}}; "
             f"print(sorted(tops - set(sys.stdlib_module_names) - {SKIP_IMPORTS!r} - {{m for m in tops if m.startswith('_')}}))")
    done = run_process([executable, "-c", probe], cwd=cwd, capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    return set(ast.literal_eval(done.stdout.strip()))

FORK_WORKFLOWS = ("card-guards.yml", "card-sync.yml", "card-preview.yml")   # install requirements-cards.txt

FORK_ACTION = ".github/actions/cards-python/action.yml"                 # their one setup (2026-10-25)

FORK_PYTHON = "3.14"                                                    # every Action (owner, 2026-10-24)

FORK_WHEELS = "~/.cache/card-wheels"

OXIPNG_VERSION = "10.2.1"                                               # the CLI the Actions download (SMALLER_PNG_PUSHES.md)

FORK_KEY = ("cards-${{ steps.reqs.outputs.sha }}-${{ runner.os }}-${{ runner.arch }}"
            "-py${{ steps.python.outputs.python-version }}")             # one entry for all

BY_NAME = "sed 's/ @ [^;]*//' bot/requirements-cards.txt > \"$RUNNER_TEMP/cards.txt\""


def card_pin() -> str:
    """The psd-tools fork's tag: requirements-cards.txt's one fork URL, the only pin (no other requirements file has one)."""
    pins = re.findall(r"^psd-tools\[composite,text\] @ git\+https://github\.com/MichaelJSr/psd-tools@(\S+)$",
                      (REPO / "requirements-cards.txt").read_text(encoding="utf-8"), re.M)
    assert len(pins) == 1 and re.fullmatch(r"cards-\d{8}[a-z]*", pins[0]), "one fork pin, a tag (never a branch)"
    for other in REPO.glob("requirements*.txt"):
        assert other.name == "requirements-cards.txt" or "psd-tools" not in other.read_text(encoding="utf-8").lower(), other.name
    return pins[0]


def _uses_setup(step) -> bool:
    return step.get("uses") == "./" + FORK_ACTION.rsplit("/", 1)[0]


def fork_action(workflows) -> str:
    """
    The card workflows' one setup (.github/actions/cards-python): Python 3.14; requirements-cards.txt's wheels restored by
    the file's hash, runner and Python (one entry for all), built only on a miss, installed offline by name with uv (pip
    builds a URL line from git again), saved only after they installed.

    Returns:
        The cache key.
    """
    yaml = pytest.importorskip("yaml")
    steps = yaml.safe_load((workflows.parent.parent / FORK_ACTION).read_text(encoding="utf-8"))["runs"]["steps"]
    python, reqs, restore, uv, install, save, oxipng = steps
    assert reqs["id"] == "reqs" and "sha256sum" in reqs["run"]          # the key: requirement lines, comments stripped
    assert python["uses"].startswith("actions/setup-python@") and python["id"] == "python"
    assert python["with"] == {"python-version": FORK_PYTHON}                   # no pip cache
    assert restore["uses"].startswith("actions/cache/restore@") and restore["id"] == "wheels"
    assert restore["with"] == {"path": FORK_WHEELS, "key": FORK_KEY}
    assert uv["uses"].startswith("astral-sh/setup-uv@") and uv["with"]["enable-cache"] is False
    run = install["run"]
    assert install["env"]["HIT"] == "${{ steps.wheels.outputs.cache-hit }}" and '"$HIT"' in run
    assert run.index("pip wheel -q -f ~/.cache/card-wheels -w ~/.cache/card-wheels -r bot/requirements-cards.txt") < run.index(BY_NAME)
    assert run.index(BY_NAME) < run.index('uv pip install -q --no-index -f ~/.cache/card-wheels -r "$RUNNER_TEMP/cards.txt"')
    assert save["uses"].startswith("actions/cache/save@") and save["if"].startswith("steps.wheels.outputs.cache-hit != 'true'")
    assert save["with"] == {"path": FORK_WHEELS, "key": "${{ steps.wheels.outputs.cache-primary-key }}"}
    assert oxipng["if"] == "inputs.oxipng == 'true'" and oxipng["env"]["VERSION"] == OXIPNG_VERSION    # the CLI, pinned
    assert re.fullmatch(r"[0-9a-f]{64}", oxipng["env"]["SHA256"]) and "sha256sum -c" in oxipng["run"]
    assert oxipng["run"].index("sha256sum -c") < oxipng["run"].index("tar ") < oxipng["run"].index("GITHUB_PATH")
    with TemporaryDirectory() as tmp:          # the copy pip installs, made by the step's own sed
        (Path(tmp) / "bot").mkdir()
        shutil.copy(REPO / "requirements-cards.txt", Path(tmp) / "bot")
        run_process(["bash", "-c", BY_NAME], cwd=tmp, env={**environ, "RUNNER_TEMP": tmp}, check=True, timeout=30)
        copy = [line for line in (Path(tmp) / "cards.txt").read_text(encoding="utf-8").splitlines() if line and line[0] != "#"]
    assert copy[0] == "psd-tools[composite,text]" and " @ " not in "\n".join(copy)
    return restore["with"]["key"]


def fork_install(path) -> str:
    """
    A workflow's install of requirements-cards.txt (the psd-tools fork's one pin): every job that installs uses the shared
    setup (fork_action) after a bot checkout bringing the file; no pin of the workflow's own, no other install
    (card-preview skips the oxipng CLI: it would recompress a PNG it never publishes).

    Returns:
        The cache key.
    """
    text = path.read_text(encoding="utf-8")
    assert "PSD_TOOLS_REF" not in text and "MichaelJSr/psd-tools" not in text, f"{path.name}: the pin is requirements-cards.txt's"
    assert not re.search(r"cards-\d{8}", text), f"{path.name}: no fork tag of its own"
    assert not re.search(r"pip install", text), f"{path.name}: no install of its own"
    yaml = pytest.importorskip("yaml")
    installs = 0
    for job in yaml.safe_load(text)["jobs"].values():
        steps = job.get("steps", [])
        setups = [i for i, s in enumerate(steps) if _uses_setup(s)]
        if not setups:
            continue
        installs += 1
        bot = [i for i, s in enumerate(steps) if s.get("with", {}).get("repository") in ("bobberlington/AutoCroissant", "${{ env.BOT_REPO }}")
               and "/requirements-cards.txt" in s["with"]["sparse-checkout"].split()]
        assert len(bot) == 1 and bot[0] < setups[0], f"{path.name}: the bot checkout brings requirements-cards.txt first"
        assert ((steps[setups[0]].get("with") or {}).get("oxipng") == "false") == (path.name == "card-preview.yml")
    assert installs, f"{path.name}: installs requirements-cards.txt"
    return fork_action(path.parent)


# ========================
# From test_font_check
# ========================
class ChecksGitHub:
    """blobs: sha -> bytes; compare: the compare API's body; commits: sha -> the commit API's body; tree: a trees body."""
    def __init__(self, blobs: dict, compare: dict, commits: dict, tree: dict = None):
        self.blobs, self.compare, self.commits, self.tree = blobs, compare, commits, tree
        self.broken: set = set()                 # blobs served with other bytes
        self.calls: list = []

    def get(self, url, headers=None, timeout=None, stream=False):
        self.calls.append(url)
        path = url.split("/repos/", 1)[1].split("/", 2)[2]
        if path.startswith("compare/"):
            return Reply(body=self.compare) if self.compare is not None else Reply(404)   # None: the old head is gone
        if path.startswith("commits/"):
            sha = path.split("/")[1]
            return Reply(body=self.commits[sha]) if sha in self.commits else Reply(404)
        if path.startswith("git/trees/"):
            return Reply(body=self.tree)
        if path.startswith("git/blobs/"):
            sha = path.rsplit("/", 1)[1]
            if sha in self.broken:
                return Reply(data=b"garbage")
            return Reply(data=self.blobs[sha]) if sha in self.blobs else Reply(404)
        return Reply(404)

    def blob_calls(self) -> int:
        return sum("/git/blobs/" in u for u in self.calls)


# ========================
# From test_png_export
# ========================
def card(size=(60, 90), box=(10, 10, 30, 30), colour=(0, 0, 255), base=(200, 30, 30)) -> Image.Image:
    image = Image.new("RGB", size, base)
    ImageDraw.Draw(image).rectangle(box, fill=colour)
    return image


def fixture_psd(tmp_path, image: Image.Image, name: str, square=(10, 10, 31, 31)) -> PSDImage:
    """A PSD with a Base layer (the image's corner colour) and a Square layer (the image's crop at square), saved by
    psd-tools (its composite is the image) and reopened: a change of the square changes one layer record."""
    psd = PSDImage.new("RGB", image.size)
    psd.append(PixelLayer.frompil(Image.new("RGB", image.size, image.getpixel((0, image.height - 1))), psd, "Base"))
    psd.append(PixelLayer.frompil(image.crop(square), psd, "Square", top=square[1], left=square[0]))
    path = tmp_path / f"{name}.psd"
    psd.save(str(path))
    return PSDImage.open(str(path))


def _rows_off(monkeypatch, rows: int, start: int = 3, levels: int = 30) -> None:
    """The fork draws rows of every box 30 levels off from its row start (a drawing miss, the same on the old and new PSD)."""
    draw = pe.render_region

    def off(psd, box, icc=True, text="preview"):
        image = np.array(draw(psd, box, icc, text).convert("RGB")).astype(int)
        image[start:start + rows] += levels
        return Image.fromarray(np.clip(image, 0, 255).astype(np.uint8))
    monkeypatch.setattr(pe, "render_region", off)


@lru_cache(maxsize=None)
def lcms_profile(name: str) -> bytes:
    """An LCMS profile's bytes, made once: its header stamps the second it was made."""
    from PIL import ImageCms
    return ImageCms.ImageCmsProfile(ImageCms.createProfile(name)).tobytes()


def photoshop_png(image: Image.Image, icc="sRGB") -> bytes:
    """image as Photoshop exports it: no deflate (zlib level 0), with its colour and text chunks (icc: an LCMS profile's
    name, or a profile's bytes)."""
    import io
    from struct import pack
    from PIL import PngImagePlugin
    info = PngImagePlugin.PngInfo()
    info.add(b"gAMA", pack(">I", 45455))
    info.add_itxt("XML:com.adobe.xmp", "<x:xmpmeta xmlns:x='adobe:ns:meta/'>" + "photoshop " * 400 + "</x:xmpmeta>")
    info.add_itxt("Comment", "squeezed " * 50, zip=True)
    info.add_text("Software", "Adobe Photoshop 24.0", zip=True)
    buf = io.BytesIO()
    image.save(buf, "PNG", compress_level=0, pnginfo=info, dpi=(300, 300),
               icc_profile=icc if isinstance(icc, bytes) else lcms_profile(icc))
    return buf.getvalue()


def art(mode="RGBA", size=(400, 600)) -> Image.Image:
    """A card-like picture: gradients and shapes (opaque, so oxipng could drop the alpha if allowed)."""
    x, y = np.meshgrid(np.arange(size[0]), np.arange(size[1]))
    rgb = np.stack([x * 255 // size[0], y * 255 // size[1], (x + y) % 256], axis=2).astype(np.uint8)
    image = Image.fromarray(rgb).convert(mode)
    ImageDraw.Draw(image).ellipse((50, 80, 300, 400), fill=(250, 200, 40) + (() if mode == "RGB" else (255,)))
    return image


# ========================
# From test_card_data
# ========================
OTHER = "Auxiliary/Minions/Anubisath_Guardian.psd"


def git(repo, *args, env=None):
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True,
                          env={**os.environ, **(env or {})}).stdout


def commit(repo, author, at):
    stamp = f"@{at} +0000"
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", f"by {author}", env={"GIT_AUTHOR_NAME": author, "GIT_AUTHOR_EMAIL": f"{author}@x",
                                                          "GIT_COMMITTER_NAME": author, "GIT_COMMITTER_EMAIL": f"{author}@x",
                                                          "GIT_AUTHOR_DATE": stamp, "GIT_COMMITTER_DATE": stamp})
    return git(repo, "rev-parse", "HEAD").strip()


def put(repo, rel, data: bytes):
    path = repo / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


@pytest.fixture(scope="session")
def cards_template(tmp_path_factory, clone):
    """The cards repo, built once per worker (git clone --local per test hardlinks its objects)."""
    psds = {}
    for rel in (DOOMER, FREEZER, OTHER):
        data = (clone / rel).read_bytes()
        assert blob_sha_bytes(data) == json.loads(golden_file(rel).read_text(encoding="utf-8"))["blob"], f"{rel}: goldens not at CARDS_REV"
        psds[rel] = data
    vocab = json.loads(VOCAB.read_text(encoding="utf-8"))
    repo = tmp_path_factory.mktemp("cards-template") / "cards"
    repo.mkdir()
    git(repo, "init", "-q", "-b", "main")
    for t in vocab["all_types"]:
        put(repo, f"Types/{t}.png", t.encode())
    for s in vocab["stars"]:
        put(repo, f"Types/Stars/{s}.png", s.encode())
    put(repo, "MDW/Fonts/stoogedings-src/stoogedings.json", json.dumps(vocab["manifest"]).encode())
    put(repo, "site/Card Authors.txt", b"[people]\nbobgit = Bob\n[cards]\nmini_doomer = bobgit\n")
    put(repo, "site/Card Edits.json", json.dumps({"cards": {"The Freezer": {"note": "checked", "_by": "web"}}}).encode())
    put(repo, DOOMER, psds[DOOMER])
    put(repo, FREEZER, psds[FREEZER])
    first = commit(repo, "alice", 1000)
    return SimpleNamespace(repo=repo, psds=psds, first=first)


@pytest.fixture
def cards(tmp_path, cards_template, monkeypatch):
    """A card repo: the golden vocabulary, the site files, Mini Doomer and The Freezer (alice, t=1000); an empty db/.
    G1's floor drops to 1 card and no share (a real build has hundreds); no font index for the "supplied" flags."""
    from commands import psd_details
    monkeypatch.setattr(cd, "MIN_CARDS", 1)
    monkeypatch.setattr(cd, "FLOOR_SHARE", 0)
    monkeypatch.setattr(cd, "_activity", lambda repository: [])                  # offline: pushes told by author
    monkeypatch.setattr(psd_details, "font_index", lambda tree=None: None)
    repo = tmp_path / "cards"
    git(tmp_path, "clone", "-q", "--local", str(cards_template.repo), str(repo))
    git(repo, "remote", "remove", "origin")                                    # as the template: no remote
    db = tmp_path / "data" / "db"
    db.mkdir(parents=True)
    return SimpleNamespace(repo=repo, psds=dict(cards_template.psds), db=db, first=cards_template.first, tmp=tmp_path)

def _build(c, *extra):
    message = c.tmp / "message.txt"
    code = cd.main(["--checkout", str(c.repo), "--db", str(c.db), "--message", str(message), *extra])
    return code, (message.read_text(encoding="utf-8") if message.exists() else "")


def _read(c):
    stats, old, details, manifest = cdb.read_db(str(c.db))
    return stats, old, details, manifest, cl.read_events(str(c.db))


def git_bytes(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, check=True).stdout


# ========================
# From test_type_emojis
# ========================
class Emoji:
    def __init__(self, client, name):
        self.client, self.name, self.id = client, name, len(client.created) + 100

    async def delete(self):
        self.client.deleted.append(self.name)
        self.client.emojis.remove(self)

    def __str__(self):
        return f"<:{self.name}:{self.id}>"


class Client:
    def __init__(self, names=(), fail_fetch=False):
        self.created, self.deleted, self.fetches, self.fail_fetch = [], [], 0, fail_fetch
        self.emojis = []
        for n in names:
            self.emojis.append(Emoji(self, n))

    async def fetch_application_emojis(self):
        self.fetches += 1
        if self.fail_fetch:
            raise RuntimeError("503 Service Unavailable")
        return list(self.emojis)

    async def create_application_emoji(self, name, image):
        assert Image.open(BytesIO(image)).size == (te.EMOJI_SIZE, te.EMOJI_SIZE)
        emoji = Emoji(self, name)
        self.created.append(name)
        self.emojis.append(emoji)
        return emoji


# ========================
# From test_psd_edit
# ========================
MINI_DOOMER = "Auxiliary/Minions/Mini_Doomer.psd"                 # template rows, no digit effects (certified golden)


def _typo_fix(title: str) -> str:
    """A rename of the same width: the last letter swapped (the live edits fix typos, they do not lengthen titles)."""
    return title[:len(title) - 1] + ("e" if title[len(title) - 1] != "e" else "a")


def _card(clone, rel: str) -> tuple:
    path, png = clone / rel, clone / (rel[:-4] + ".png")
    if not path.exists() or not png.exists():
        pytest.skip(f"{rel} not in the clone")
    return str(path), str(png)


def stale_copy(clone, rel: str, title: str, tmp_path) -> tuple:
    """A pre-fix bake's PSD: the title set in TySh alone (the fork's set_text, then the old story put back)."""
    psd = PSDImage.open(_card(clone, rel)[0])
    layer = pe.name_layer(psd)
    index = pe.story_index(layer)
    items = pe._story_list(pe._txt2(psd))[2]
    old = deepcopy(items[index])
    layer.set_text(title)
    items[index] = old
    path = tmp_path / Path(rel).name
    pe.save_psd(psd, str(path), update_preview=False)
    return str(path), index
