"""
The rulebook's TTS PDF: the website's #/rules/print pages (1080x1920, the cover then the chapters cut as the Photoshop
pages were) printed by Chrome to the card repo's TTS/Rulebook/Rulebook.pdf (real text: searchable and selectable, one
page per sheet, bookmarks from the headings), and pages.json listing the pages with a hash of each one's pixels. Card
sync's main lane runs it before the save; tts_save puts the PDF in the save's Rulebooks bag (a Custom_PDF object).

    python -m commands.rulebook_pages <card repo checkout> [--site URL]

The website must show the checkout's chapters (the page lists the chapters it was cut from): else nothing changes and
the next run draws them. Headless Chrome ($CHROME, else the first of CHROMES on PATH) over its DevTools pipe; no
package beyond Pillow. Each page is screenshot and hashed: the PDF is printed again only when a hash changed (a PDF's
bytes carry its print time), so an unchanged rulebook keeps its file and its pinned link. Page PNGs left from before the
PDF are removed. Exit 1 when the pages could not be drawn (nothing changed then).
"""
from argparse import ArgumentParser
from base64 import b64decode
from io import BytesIO
from json import dumps, loads
from os import close, environ, listdir, makedirs, pipe, read, remove, write
from hashlib import sha256
from os.path import getsize, isdir, isfile, join
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
PDF_FILE = "Rulebook.pdf"
TITLE = "Impossibility Simulator Rulebook"     # the PDF's title
JPEG_QUALITY = 0.88                            # opaque pictures go in the PDF as JPEG (Chrome keeps them; others go lossless)
TO_JPEG = """Promise.all([...document.querySelectorAll('.rb-sheets img')].map(async im => {
  if (!im.naturalWidth || im.src.startsWith('data:image/jpeg')) return 0;
  const c = document.createElement('canvas'); c.width = im.naturalWidth; c.height = im.naturalHeight;
  const g = c.getContext('2d'); g.drawImage(im, 0, 0);
  const a = g.getImageData(0, 0, c.width, c.height).data;
  for (let i = 3; i < a.length; i += 4) if (a[i] < 255) return 0;     // see-through: as it is
  im.src = c.toDataURL('image/jpeg', %s); await im.decode().catch(() => 0); return 1;
})).then(r => r.reduce((x, y) => x + y, 0))"""
PRINT_CSS = ("@page { size: 1080px 1920px; margin: 0 } "   # one sheet a page, nothing else printed
             "@media print { body > :not(main), main > :not(.rb-sheets) { display: none !important } "
             "main { padding: 0 !important; margin: 0 !important; max-width: none !important } .rb-sheets { display: block !important } "
             ".rb-sheets .rb-sheet { zoom: 1 !important; box-shadow: none !important; break-after: page; margin: 0 !important } }")
SHEET = (1080, 1920)                           # a page, as the Photoshop pages were (rulebook.js SHEET)
WAIT_SECONDS = 180                             # for the website to load and cut its pages
CALL_SECONDS = 60                              # one DevTools call
ABOUT = ("The rulebook's TTS PDF, printed from the website's #/rules/print by AutoCroissant commands/rulebook_pages.py "
         "(Card sync); the chapters are site/rulebook. sha: each page's pixels (the PDF is printed again when one changes). "
         "tts_save puts the PDF in the save's Rulebooks bag.")


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
        ({file: source} of the chapters the website cut them from, [(sheet entry, PNG bytes)] in book order, a Printer:
        its pdf() prints the PDF, its close() quits Chrome (call it in any case))
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
            'const bg = [...new Set([...document.querySelectorAll(".rb-sheets *")].map(e => getComputedStyle(e).backgroundImage)'
            '.flatMap(v => [...v.matchAll(/url\\("?([^")]+)"?\\)/g)].map(m => m[1])))].map(u => Object.assign(new Image(), { src: u }));'   # the ability boxes' texture
            'Promise.all([...document.images, ...bg].map(i => i.decode().catch(() => 0))).then(() => document.fonts.ready)'
            '.then(() => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)))).then(() => 1)')
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

        def pdf() -> bytes:
            run(f"document.title = {dumps(TITLE)}; document.head.insertAdjacentHTML('beforeend', {dumps('<style>' + PRINT_CSS + '</style>')}); 1")
            run(TO_JPEG % JPEG_QUALITY)
            params = {"printBackground": True, "preferCSSPageSize": True, "marginTop": 0, "marginBottom": 0, "marginLeft": 0,
                      "marginRight": 0, "generateDocumentOutline": True, "generateTaggedPDF": True}
            try:
                got = chrome.call("Page.printToPDF", params, session, seconds=240)
            except RuntimeError:                       # an older Chrome without the outline: plain
                got = chrome.call("Page.printToPDF", {k: v for k, v in params.items() if not k.startswith("generate")}, session, seconds=240)
            return b64decode(got["data"])
        return dict(listed["chapters"]), shots, Printer(pdf, chrome.close)
    except BaseException:
        chrome.close()
        raise


class Printer:
    """The drawn website's PDF (pdf()) and Chrome's end (close())."""

    def __init__(self, pdf, close):
        self.pdf, self.close = pdf, close


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


def page_sha(png: bytes) -> str:
    """A page's pixels' hash (its PNG bytes may differ for the same pixels)."""
    with Image.open(BytesIO(png)) as im:
        return sha256(im.convert("RGB").tobytes() + str(im.size).encode()).hexdigest()


def read_pages(checkout: str) -> dict:
    """pages.json ({} when missing or unreadable): {"pdf": file, "pages": [{name, num, sha}]}."""
    path = join(checkout, RULEBOOK_PAGES, PAGES_JSON)
    try:
        with open(path, encoding="utf-8") as fh:
            data = loads(fh.read())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) and isinstance(data.get("pages"), list) else {}


def write_pages(folder: str, shots: list, pdf_maker) -> dict:
    """
    pages.json and the PDF into folder: the PDF printed (pdf_maker(), bytes) when a page's hash changed, the list changed or the
    PDF is missing; page PNGs left from before removed.

    Returns:
        {"pages": n, "changed": [page names], "printed": bool, "removed": [files]}
    """
    makedirs(folder, exist_ok=True)
    pages = [{"name": entry["name"], "num": entry.get("num") or "", "sha": page_sha(png)} for entry, png in shots]
    listing, pdf = join(folder, PAGES_JSON), join(folder, PDF_FILE)
    old = {}
    if isfile(listing):
        try:
            with open(listing, encoding="utf-8") as fh:
                old = loads(fh.read())
        except ValueError:
            old = {}
    before = [(p.get("name"), p.get("sha")) for p in old.get("pages") or [] if isinstance(p, dict)]
    changed = [p["name"] for k, p in enumerate(pages) if k >= len(before) or before[k] != (p["name"], p["sha"])]
    need = bool(changed) or len(before) != len(pages) or not isfile(pdf)
    out = {"pages": len(pages), "changed": changed, "printed": False, "removed": []}
    if need:
        data = pdf_maker()
        with open(pdf, "wb") as fh:
            fh.write(data)
        out["printed"] = True
    for name in sorted(listdir(folder)):
        if name.lower().endswith(".png"):                    # the page images of before the PDF
            remove(join(folder, name))
            out["removed"].append(name)
    text = dumps({"about": ABOUT, "pdf": PDF_FILE, "pages": pages}, ensure_ascii=False, indent=1) + "\n"
    if text != (open(listing, encoding="utf-8").read() if isfile(listing) else None):
        with open(listing, "w", encoding="utf-8") as fh:
            fh.write(text)
    return out


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
    theirs, shots, printer = draw(site, exe)
    try:
        if theirs != mine:
            return (f"Rulebook PDF: the website doesn't show main's chapters yet ({len(set(theirs.items()) ^ set(mine.items()))} "
                    "differ); left as it is, the next run prints it.")
        done = write_pages(join(checkout, RULEBOOK_PAGES), shots, printer.pdf)
    finally:
        printer.close()
    note = f"; {len(done['removed'])} old page images removed" if done["removed"] else ""
    if not done["printed"]:
        return f"Rulebook PDF: {done['pages']} pages, as they were{note}."
    size = getsize(join(checkout, RULEBOOK_PAGES, PDF_FILE))
    what = f"{len(done['changed'])} pages changed" if done["changed"] and len(done["changed"]) < done["pages"] else "printed"
    return f"Rulebook PDF: {done['pages']} pages, {what} ({size / 1e6:.1f} MB){note}."


def main(args: Optional[list] = None) -> int:
    """The command line (card-sync.yml): update_pages, its report on stdout."""
    parser = ArgumentParser(prog="python -m commands.rulebook_pages", description="Print the rulebook's TTS PDF from the website.")
    parser.add_argument("checkout", help="the card repo's checkout (its site/rulebook; the PDF goes to TTS/Rulebook)")
    parser.add_argument("--site", default=SITE_URL, help="the website (default: the card browser)")
    options = parser.parse_args(args)
    try:
        print(update_pages(options.checkout, options.site))
    except (RuntimeError, TimeoutError, OSError, ValueError, KeyError) as e:
        print(f"Rulebook PDF not printed: {e}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# ========================
# Module Exports
# ========================
__all__ = [
    'PAGES_JSON',
    'PDF_FILE',
    'main',
    'read_pages',
    'update_pages',
]
