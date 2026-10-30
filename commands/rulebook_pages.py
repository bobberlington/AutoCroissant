"""
The rulebook's TTS pages: the website's #/rules/print pages (1080x1920, the chapters cut as the Photoshop pages were) as
PNGs in the card repo's TTS/Rulebook, and pages.json listing them in book order. Card sync's main lane runs it before
the save; tts_save puts the pages in the save's Rulebooks bag.

    python -m commands.rulebook_pages <card repo checkout> [--site URL]

The website must show the checkout's chapters (the page lists the chapters it was cut from): else nothing changes and
the next run draws them. Headless Chrome ($CHROME, else the first of CHROMES on PATH) over its DevTools pipe; no
package beyond Pillow. A page whose pixels are the same keeps its file (its pinned link too); pages no longer listed
go. Exit 1 when the pages could not be drawn (nothing changed then).
"""
from argparse import ArgumentParser
from base64 import b64decode
from io import BytesIO
from json import dumps, loads
from os import close, environ, listdir, makedirs, pipe, read, remove, write
from os.path import isdir, isfile, join
from platform import system
from select import select
from shutil import rmtree, which
from subprocess import DEVNULL, Popen, TimeoutExpired
from tempfile import mkdtemp
from time import monotonic, sleep
from typing import Optional

from PIL import Image

from commands.rulebook import CHAPTER_FILE
from global_config import RULEBOOK_PAGES, RULEBOOK_SRC, SITE_URL

# ========================
# Configuration
# ========================
CHROMES = ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser", "chrome-headless-shell", "chrome")
VIEW = "#/rules/print"                         # the website's TTS pages (app.js renderSheets)
PAGES_JSON = "pages.json"
SHEET = (1080, 1920)                           # a page, as the Photoshop pages were (rulebook.js SHEET)
WAIT_SECONDS = 180                             # for the website to load and cut its pages
CALL_SECONDS = 60                              # one DevTools call
ABOUT = ("The rulebook's TTS pages, drawn from the website's #/rules/print by AutoCroissant commands/rulebook_pages.py "
         "(Card sync); the chapters are site/rulebook. tts_save puts them in the save's Rulebooks bag, in this order.")


# ========================
# Chrome
# ========================
class Chrome:
    """Headless Chrome over --remote-debugging-pipe: JSON messages ending in NUL on its fds 3 (in) and 4 (out)."""

    def __init__(self, exe: str):
        into, self._send = pipe()
        self._recv, out = pipe()
        self.profile = mkdtemp(prefix="rulebook-chrome-")
        flags = ["--headless", "--remote-debugging-pipe", f"--user-data-dir={self.profile}", "--no-first-run",
                 "--hide-scrollbars", "--force-color-profile=srgb", "--use-mock-keychain", "--password-store=basic"]
        if system() == "Linux":
            flags.append("--no-sandbox")       # GitHub's Ubuntu runners have no user namespaces for its sandbox
        self.proc = Popen(["sh", "-c", f'exec "$0" "$@" 3<&{into} 4>&{out}', exe, *flags, "about:blank"],
                          pass_fds=(into, out), stdout=DEVNULL, stderr=DEVNULL)
        close(into)
        close(out)
        self._buf, self._id = b"", 0

    def call(self, method: str, params: Optional[dict] = None, session: str = "", seconds: float = CALL_SECONDS) -> dict:
        """A DevTools command's result (events in between are dropped); RuntimeError for its error."""
        self._id += 1
        message = {"id": self._id, "method": method, "params": params or {}, **({"sessionId": session} if session else {})}
        data = dumps(message).encode() + b"\0"
        while data:
            data = data[write(self._send, data):]
        until = monotonic() + seconds
        while True:
            while b"\0" not in self._buf:
                left = until - monotonic()
                if left <= 0 or not select([self._recv], [], [], left)[0]:
                    raise TimeoutError(f"Chrome did not answer {method} in {seconds:.0f}s")
                chunk = read(self._recv, 1 << 20)
                if not chunk:
                    raise RuntimeError("Chrome quit")
                self._buf += chunk
            line, self._buf = self._buf.split(b"\0", 1)
            reply = loads(line)
            if reply.get("id") == self._id:
                if "error" in reply:
                    raise RuntimeError(f"{method}: {reply['error'].get('message')}")
                return reply.get("result") or {}

    def close(self) -> None:
        try:
            self.call("Browser.close", seconds=10)
        except (OSError, RuntimeError, TimeoutError):
            pass
        try:
            self.proc.wait(timeout=10)
        except TimeoutExpired:
            self.proc.kill()
        for fd in (self._send, self._recv):
            try:
                close(fd)
            except OSError:
                pass
        rmtree(self.profile, ignore_errors=True)


def find_chrome() -> Optional[str]:
    """$CHROME, else the first of CHROMES on PATH."""
    return environ.get("CHROME") or next((found for name in CHROMES if (found := which(name))), None)


def draw(site: str, exe: str) -> tuple:
    """
    The website's TTS pages.

    Returns:
        ({file: source} of the chapters the website cut them from, [(sheet entry, PNG bytes)] in book order)
    """
    chrome = Chrome(exe)
    try:
        target = chrome.call("Target.createTarget", {"url": "about:blank"})["targetId"]
        session = chrome.call("Target.attachToTarget", {"targetId": target, "flatten": True})["sessionId"]

        def run(expression: str):
            got = chrome.call("Runtime.evaluate", {"expression": expression, "returnByValue": True, "awaitPromise": True}, session)
            if got.get("exceptionDetails"):
                raise RuntimeError(f"the page: {got['exceptionDetails'].get('text')}")
            return (got.get("result") or {}).get("value")
        chrome.call("Emulation.setDeviceMetricsOverride", {"width": 1400, "height": 1000, "deviceScaleFactor": 1, "mobile": False}, session)
        chrome.call("Page.navigate", {"url": f"{site.rstrip('/')}/{VIEW}"}, session)
        until = monotonic() + WAIT_SECONDS
        while not run("document.body?.dataset.sheets || ''"):
            if monotonic() > until:
                raise TimeoutError(f"the website cut no pages in {WAIT_SECONDS}s")
            sleep(0.5)
        listed = loads(run('document.getElementById("rb-sheet-list").textContent'))
        run('document.head.insertAdjacentHTML("beforeend", "<style>.rb-sheets .rb-sheet { zoom: 1 !important; }</style>"); '
            'Promise.all([...document.images].map(i => i.decode().catch(() => 0))).then(() => document.fonts.ready).then(() => 1)')
        boxes = run('[...document.querySelectorAll(".rb-sheets .rb-sheet")].map(e => { const r = e.getBoundingClientRect(); '
                    'return [r.left + scrollX, r.top + scrollY, r.width, r.height]; })')
        if len(boxes) != len(listed["sheets"]):
            raise RuntimeError(f"{len(boxes)} pages on the website, {len(listed['sheets'])} listed")
        shots = []
        for entry, (x, y, w, h) in zip(listed["sheets"], boxes):
            if (round(w), round(h)) != SHEET:
                raise RuntimeError(f"page {entry.get('num') or entry.get('name')} is {w:.0f}x{h:.0f}, not {SHEET[0]}x{SHEET[1]}")
            got = chrome.call("Page.captureScreenshot", {"format": "png", "captureBeyondViewport": True,
                                                         "clip": {"x": x, "y": y, "width": w, "height": h, "scale": 1}}, session)
            shots.append((entry, b64decode(got["data"])))
        return dict(listed["chapters"]), shots
    finally:
        chrome.close()


# ========================
# Files
# ========================
def chapter_sources(checkout: str) -> dict:
    """{file: source} of the checkout's chapters (as the website reads them)."""
    folder = join(checkout, RULEBOOK_SRC)
    found = {}
    for name in sorted(listdir(folder)) if isdir(folder) else []:
        if CHAPTER_FILE.match(name):
            with open(join(folder, name), encoding="utf-8") as fh:
                found[name] = fh.read().replace("\r\n", "\n")
    return found


def page_file(entry: dict) -> str:
    """A page's file: its chapter's number and its page in it ("07-02.png": Stats 2)."""
    return f"{entry['ch']}-{int(entry['n']):02d}.png"


def same_pixels(png: bytes, path: str) -> bool:
    if not isfile(path):
        return False
    with Image.open(path) as old, Image.open(BytesIO(png)) as new:
        return old.size == new.size and old.convert("RGB").tobytes() == new.convert("RGB").tobytes()


def write_pages(folder: str, shots: list) -> dict:
    """
    The pages into folder: a changed one rewritten (Pillow, optimized), the same one left, the ones no longer listed
    removed, pages.json rewritten when it differs.

    Returns:
        {"new": [files], "redrawn": [files], "removed": [files], "listed": bool (pages.json changed)}
    """
    makedirs(folder, exist_ok=True)
    out = {"new": [], "redrawn": [], "removed": [], "listed": False}
    pages = []
    for entry, png in shots:
        name = page_file(entry)
        path = join(folder, name)
        pages.append({"file": name, "name": entry["name"], "num": entry.get("num") or ""})
        if same_pixels(png, path):
            continue
        out["redrawn" if isfile(path) else "new"].append(name)
        with Image.open(BytesIO(png)) as im:
            im.convert("RGB").save(path, "PNG", optimize=True)
    keep = {p["file"] for p in pages}
    for name in sorted(listdir(folder)):
        if name.lower().endswith(".png") and name not in keep:
            remove(join(folder, name))
            out["removed"].append(name)
    text = dumps({"about": ABOUT, "pages": pages}, ensure_ascii=False, indent=1) + "\n"
    listing = join(folder, PAGES_JSON)
    old = None
    if isfile(listing):
        with open(listing, encoding="utf-8") as fh:
            old = fh.read()
    if text != old:
        with open(listing, "w", encoding="utf-8") as fh:
            fh.write(text)
        out["listed"] = True
    return out


def read_pages(checkout: str) -> list:
    """pages.json's pages ([] when missing or unreadable): [{file, name, num}] in book order."""
    path = join(checkout, RULEBOOK_PAGES, PAGES_JSON)
    try:
        with open(path, encoding="utf-8") as fh:
            pages = loads(fh.read()).get("pages")
    except (OSError, ValueError, AttributeError):
        return []
    return [p for p in pages if isinstance(p, dict) and isinstance(p.get("file"), str)] if isinstance(pages, list) else []


# ========================
# Public API Functions
# ========================
def update_pages(checkout: str, site: str = SITE_URL, exe: str = "") -> str:
    """
    The TTS pages brought to the website's rulebook, when the website shows the checkout's chapters.

    Returns:
        The report (one line)

    Raises:
        RuntimeError, TimeoutError, OSError: the pages could not be drawn (nothing written)
    """
    exe = exe or find_chrome()
    if not exe:
        raise RuntimeError(f"no Chrome: set CHROME or install one of {', '.join(CHROMES)}")
    mine = chapter_sources(checkout)
    if not mine:
        return f"Rulebook pages: no chapters in {RULEBOOK_SRC}; pages left as they are."
    theirs, shots = draw(site, exe)
    if theirs != mine:
        return (f"Rulebook pages: the website doesn't show main's chapters yet ({len(set(theirs.items()) ^ set(mine.items()))} "
                "differ); pages left as they are, the next run draws them.")
    done = write_pages(join(checkout, RULEBOOK_PAGES), shots)
    changed = [f"{len(done[k])} {k}" for k in ("new", "redrawn", "removed") if done[k]]
    return f"Rulebook pages: {len(shots)} pages" + (f", {', '.join(changed)}." if changed else ", all as they were.")


def main(args: Optional[list] = None) -> int:
    """The command line (card-sync.yml): update_pages, its report on stdout."""
    parser = ArgumentParser(prog="python -m commands.rulebook_pages", description="Draw the rulebook's TTS pages from the website.")
    parser.add_argument("checkout", help="the card repo's checkout (its site/rulebook; the pages go to TTS/Rulebook)")
    parser.add_argument("--site", default=SITE_URL, help="the website (default: the card browser)")
    options = parser.parse_args(args)
    try:
        print(update_pages(options.checkout, options.site))
    except (RuntimeError, TimeoutError, OSError, ValueError, KeyError) as e:
        print(f"Rulebook pages not drawn: {e}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# ========================
# Module Exports
# ========================
__all__ = [
    'PAGES_JSON',
    'main',
    'read_pages',
    'update_pages',
]
