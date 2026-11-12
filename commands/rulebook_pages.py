"""
The rulebook's TTS PDF: the website's #/rules/print pages (1080x1920, the cover then the chapters cut as the Photoshop
pages were) printed by Chrome to the card repo's TTS/Rulebook/Rulebook.pdf (real text: searchable and selectable, one
page per sheet, bookmarks from the headings), and pages.json listing the pages with a hash of each one's pixels and its text (the TTS save's search). Card
sync's main lane runs it before the save; tts_save puts the PDF in the save's Rulebooks bag (a Custom_PDF object). The
pages are drawn in the website's fonts (BOOK_FONTS, the checkout's MDW/Fonts copies, also where the machine has none:
Card sync's Ubuntu), so their lines break as on the website.

    python -m commands.rulebook_pages <card repo checkout> [--site URL]

The website must show the checkout's chapters (the page lists the chapters it was cut from), pictures, web fonts,
icons and code (site_settled; Chrome's copies of what its CSS loads by plain URL: HEAD's): the command line waits up to
SITE_WAIT for that (polled without Chrome, build.txt every SITE_POLL_FIRST seconds at first; a deploy met by the draw:
looked at again at once; an old CDN copy: drawn again after SITE_POLL; in that time), else nothing changes and it exits
3 (card-sync.yml marks the lane: the schedule's next lane draws again; so does an exit 1). Headless Chrome ($CHROME,
else the first of CHROMES on PATH) over its DevTools pipe, the page past the CDN's copy; no package beyond Pillow.
Each page is screenshot and hashed: the PDF is printed again only when a hash
changed (a PDF's bytes carry its print time), so an unchanged rulebook keeps its file and its pinned link. pages.json's key skips the
screenshots when nothing they draw changed; its pre-key (pre_key: what the page reads, fetched without Chrome) skips
Chrome too when the website, the fonts and Chrome are as at the last kept key (PRE_SKIP; --full: Chrome looks in any case).
Other files in the folder are left alone. Exit 1 when the pages could not be drawn (nothing changed then).
"""
from argparse import ArgumentParser
from base64 import b64decode, b64encode
from datetime import datetime
from io import BytesIO
from json import dumps, loads
from os import close, environ, listdir, makedirs, pipe, read, write
from hashlib import sha256
from http.client import HTTPException
from os.path import getsize, isdir, isfile, join
from platform import system
from re import DOTALL, compile as regex
from select import select
from shutil import rmtree, which
from subprocess import DEVNULL, PIPE, Popen, TimeoutExpired, run as run_process
from tempfile import mkdtemp
from time import monotonic, sleep, time
from typing import Optional
from urllib.parse import unquote
from urllib.request import Request, urlopen

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
SHEET = (1080, 1920)                           # a page, as the Photoshop pages were (rulebook.js draws them)
WAIT_SECONDS = 180                             # for the website to load and cut its pages
# what the pages draw (key()): the sheets' HTML (text, and the pictures' and assets' URLs, which carry their content's
# stamp), the style sheets (an inline <style>'s text: index.html's style.css), the loaded fonts; with UNSTAMPED, the
# folders the CSS reads by plain URL
KEY_JS = ('JSON.stringify([document.querySelector(".rb-sheets")?.outerHTML || "", '
          '[...document.styleSheets].map(s => s.href || s.ownerNode?.textContent || ""), '
          '[...document.fonts].filter(f => f.status === "loaded").map(f => [f.family, f.weight, f.style].join(" ")).sort()])')
UNSTAMPED = ("site/rulebook/fonts", "site/rulebook/img")
FONTS_DIR = "MDW/Fonts"                        # the card repo's fonts (psd_edit.FONTS_DIR)
# the rulebook's fonts on the website (style.css: .rb's first, .rb-grid's tables' first): their files in FONTS_DIR
BOOK_FONTS = ("Consolas", "Century")
# the book fonts' files ([base64, family, weight, style]) as the page's own, before the website cuts its pages
FONT_JS = """for (const [b64, family, weight, style] of %s)
  document.fonts.add(new FontFace(family, Uint8Array.from(atob(b64), c => c.charCodeAt(0)), { weight, style }));"""
SITE_WAIT = 10 * 60                            # for the website's build of HEAD's rulebook (Card sync goes at once; its step's limit is 12)
# what the website draws the print view from besides cards and chapters (compared by content), each a change that deploys:
# the pictures (build_site: those it opens), web fonts (all), icons (and their aliases, tints), code. A change that does
# not (a module nothing imports) waits in vain: card-sync.yml's mark lasts one extra lane
PICTURES = ("webp", "png", "jpg", "jpeg", "gif", "avif")
SITE_PATHS = (*(f":(glob,icase)site/rulebook/img/*.{x}" for x in PICTURES), "site/rulebook/fonts/", ":(glob,icase)Types/*.png",
              ":(glob,icase)Types/Stars/*.png", "MDW/Fonts/stoogedings-src/stoogedings.json", "site/assets/")
LAGS = "the website lags main's rulebook"     # in the report of a look that waited in vain: exit 3
# Chrome's copies of the fonts and pictures the page's CSS loads by plain URL (no stamp: the CDN's, up to 10 minutes
# old; force-cache: those the page drew with): {url: git blob id, null when missing}
PLAIN_JS = r"""Promise.all([...new Set([...document.querySelectorAll("style")].flatMap(s => [...s.textContent.matchAll(/url\(\s*["']?(rulebook\/[^"')?#\s]+)/g)].map(m => m[1])))]
  .map(async u => { const r = await fetch(u, { cache: "force-cache" }).catch(() => null);
    if (!r || !r.ok) return [u, null];
    const b = new Uint8Array(await r.arrayBuffer()), h = new TextEncoder().encode(`blob ${b.length}\0`), all = new Uint8Array(h.length + b.length);
    all.set(h); all.set(b, h.length);
    return [u, [...new Uint8Array(await crypto.subtle.digest("SHA-1", all))].map(x => x.toString(16).padStart(2, "0")).join("")]; }))
  .then(Object.fromEntries)"""
SITE_POLL = 20                                 # seconds between looks at the website (and draws after an old CDN copy)
SITE_POLL_FIRST, SITE_FAST = 2, 120            # a wait's first SITE_FAST s: build.txt read every SITE_POLL_FIRST s
CALL_SECONDS = 60                              # one DevTools call
# the pre-key (pre_key()): what the print view reads, fetched without Chrome; pages.json's "pre", kept with the key
PRE_SKIP = True                                # held: no Chrome (shadow 11-09 to 11-11: no misses); --full looks and warns on a miss
PAGE_STAMP = regex(r'<meta name="page" content="([^"]*)"')   # build_site: index.html's stamp (build.txt's 2nd line)
ASSET_URL = regex(r"assets/[\w.-]+(?:\?v=\w+)?")              # the page's code: bundle, chunks, style sheet (stamped URLs)
INLINE_CSS = regex(r"<style\b[^>]*>(.*?)</style>", DOTALL)     # style.css, inlined
CARD_REF = regex(r"(!?)\[\[([^\][|\n]+)(?:\|[^\]\n]*)?\]\]")      # rulebook.js INLINE: ![[Card]] a picture, [[Card]] a link
LOOSE = regex(r"[\W_]+")                                      # coarser than app.js norm(): the named cards and any look-alike
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


def book_fonts(checkout: str) -> list:
    """BOOK_FONTS' files in the checkout's FONTS_DIR ("Consolas.ttf", "Consolas Bold Italic.ttf"...) as [base64, family,
    weight, style] (FONT_JS's), [] when it has none."""
    folder, found = join(checkout, FONTS_DIR), []
    for name in sorted(listdir(folder)) if isdir(folder) else []:
        stem, _, ext = name.rpartition(".")
        family = next((f for f in BOOK_FONTS if stem == f or stem.startswith(f + " ")), None)
        styles = set(stem[len(family or ""):].split())
        if family and ext.lower() in ("ttf", "otf") and styles <= {"Bold", "Italic"}:   # not "Century Gothic"
            with open(join(folder, name), "rb") as fh:
                found.append([b64encode(fh.read()).decode(), family, "700" if "Bold" in styles else "400", "italic" if "Italic" in styles else "normal"])
    return found


def draw(site: str, exe: str, key_of=None, known: str = "", fonts: Optional[list] = None) -> tuple:
    """
    The website's TTS pages.

    Args:
        key_of: what the pages draw (KEY_JS's text) -> their key; known: the last print's key (no screenshots when the same)
        fonts: book_fonts(), the page's BOOK_FONTS

    Returns:
        ({file: source} of the chapters the website cut them from, [(sheet entry, PNG bytes)] in book order, or None when
        the key is known's, a Printer: its pdf() prints the PDF, its close() quits Chrome (call it in any case), the key,
        the page's stamp, PLAIN_JS's {url: blob id})
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
        if fonts:                                      # (Page.enable: else the script never runs)
            chrome.call("Page.enable", session=session)
            chrome.call("Page.addScriptToEvaluateOnNewDocument", {"source": FONT_JS % dumps(fonts)}, session)
        chrome.call("Page.navigate", {"url": f"{site.rstrip('/')}/?t={int(time())}{VIEW}"}, session)   # the page past the CDN (app.js drops ?t)
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
        key = key_of(run(KEY_JS)) if key_of else ""
        page = run('document.querySelector(\'meta[name="page"]\')?.content || ""')
        plain = run(PLAIN_JS) or {}
        if key and key == known:                             # nothing they draw changed: no screenshots
            return dict(listed["chapters"]), None, Printer(None, chrome.close), key, page, plain
        texts = run('[...document.querySelectorAll(".rb-sheets .rb-sheet")].map(e => e.innerText.replace(/\\s+/g, " ").trim())')
        for entry, text in zip(listed["sheets"], texts or []):
            entry["text"] = text                             # the TTS save's search (Global.lua)
        shots = []
        for entry, (x, y, w, h) in zip(listed["sheets"], boxes):
            if (round(w), round(h)) != SHEET:
                raise RuntimeError(f"page {entry.get('num') or entry.get('name')} is {w:.0f}x{h:.0f}, not {SHEET[0]}x{SHEET[1]}")
            got = chrome.call("Page.captureScreenshot", {"format": "png", "captureBeyondViewport": True,
                                                         "optimizeForSpeed": True,     # a faster PNG, the same pixels
                                                         "clip": {"x": x, "y": y, "width": w, "height": h, "scale": 1}}, session)
            shots.append((entry, b64decode(got["data"])))

        def pdf() -> bytes:
            run(f"document.title = {dumps(TITLE)}; document.head.insertAdjacentHTML('beforeend', {dumps('<style>' + PRINT_CSS + '</style>')}); 1")
            run(TO_JPEG % JPEG_QUALITY)
            params = {"printBackground": True, "preferCSSPageSize": True, "marginTop": 0, "marginBottom": 0, "marginLeft": 0,
                      "marginRight": 0, "generateDocumentOutline": True, "generateTaggedPDF": True}
            got = chrome.call("Page.printToPDF", params, session, seconds=240)
            return b64decode(got["data"])
        return dict(listed["chapters"]), shots, Printer(pdf, chrome.close), key, page, plain
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


def _trees(checkout: str) -> list:
    """The UNSTAMPED and FONTS_DIR git trees at the checkout's HEAD ("" for one it lacks)."""
    trees = []
    for path in (*UNSTAMPED, FONTS_DIR):
        try:
            got = run_process(["git", "-C", checkout, "rev-parse", f"HEAD:{path}"], stdout=PIPE, stderr=DEVNULL, timeout=30)
            trees.append(got.stdout.decode().strip() if got.returncode == 0 else "")
        except (OSError, TimeoutExpired):
            trees.append("")
    return trees


def key(drawn: str, checkout: str, trees: Optional[list] = None) -> str:
    """The print's key: what the pages draw (KEY_JS's text) and the UNSTAMPED and FONTS_DIR git trees in the checkout
    (trees: _trees()'s, else read)."""
    return sha256("\n".join([drawn, *(_trees(checkout) if trees is None else trees)]).encode("utf-8")).hexdigest()


def pre_key(seen: dict, trees: list, fonts: list, exe: str) -> str:
    """
    What the print view reads (site_inputs()'s seen), Chrome not started: index.html's code (its stamped asset URLs and
    inline style sheets), data/rules.json, from data/cards.json the icons, iconVariants, tagMeanings and thumbs, every
    card's name and key in order (app.js cardByName, text.js's renderer) and the id, name and picture of the cards the
    chapters name (any look-alike too); the UNSTAMPED and FONTS_DIR trees (_trees()), the book fonts (book_fonts()) and
    Chrome's version.
    """
    data, index = seen["cards"], seen["index"]
    cards = [c for c in data.get("cards") or [] if isinstance(c, dict)]
    loose = lambda s: LOOSE.sub("", str(s or "").casefold())
    named = {}                                     # loose name: shown as a picture
    for chapter in loads(seen["rules"]).get("chapters") or []:
        for m in CARD_REF.finditer(str(chapter.get("source") or "") if isinstance(chapter, dict) else ""):
            named[loose(m.group(2))] = named.get(loose(m.group(2))) or bool(m.group(1))
    refs = [[c.get("id"), c.get("name"), c.get("img") if any(named.get(loose(k)) for k in (c.get("name"), c.get("key"))) else ""]
            for c in cards if any(loose(k) in named for k in (c.get("name"), c.get("key")))]
    try:
        chrome = run_process([exe, "--version"], stdout=PIPE, stderr=DEVNULL, timeout=30).stdout.decode(errors="replace").strip()
    except (OSError, TimeoutExpired):
        chrome = ""
    parts = ["pre 1", KEY_JS, sorted(set(ASSET_URL.findall(index))), INLINE_CSS.findall(index), sha256(seen["rules"]).hexdigest(),
             {k: data.get(k) for k in ("icons", "iconVariants", "tagMeanings", "thumbs")},
             [[c.get("name"), c.get("key")] for c in cards], refs, trees, sha256(dumps(fonts).encode()).hexdigest(), chrome]
    return sha256(dumps(parts, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def page_sha(png: bytes) -> str:
    """A page's pixels' hash (its PNG bytes may differ for the same pixels)."""
    with Image.open(BytesIO(png)) as im:
        return sha256(im.convert("RGB").tobytes() + str(im.size).encode()).hexdigest()


def read_pages(checkout: str) -> dict:
    """pages.json ({} when missing or unreadable): {"pdf": file, "pages": [{name, num, sha}], "key": the print's key}."""
    path = join(checkout, RULEBOOK_PAGES, PAGES_JSON)
    try:
        with open(path, encoding="utf-8") as fh:
            data = loads(fh.read())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) and isinstance(data.get("pages"), list) else {}


def write_pages(folder: str, shots: list, pdf_maker, key: str = "", pre: str = "") -> dict:
    """
    pages.json and the PDF into folder: the PDF printed (pdf_maker(), bytes) when a page's hash changed, the list changed or the
    PDF is missing; key (draw's, of a settled website: update_pages) kept for the next run, else the last print's when
    not printed. pre (pre_key() of the build drawn) goes with a kept key; the last one stays with the last key.

    Returns:
        {"pages": n, "changed": [page names], "printed": bool}
    """
    makedirs(folder, exist_ok=True)
    pages = [{"name": entry["name"], "num": entry.get("num") or "", "sha": page_sha(png), **({"t": entry["text"]} if entry.get("text") else {})}
             for entry, png in shots]
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
    out = {"pages": len(pages), "changed": changed, "printed": False}
    if need:
        data = pdf_maker()
        with open(pdf, "wb") as fh:
            fh.write(data)
        out["printed"] = True
    k = key if key or out["printed"] else old.get("key", "")
    p = pre if pre and k == key else old.get("pre", "") if k == old.get("key", "") else ""
    _put_listing(listing, {"about": ABOUT, "pdf": PDF_FILE, "key": k, "pre": p, "pages": pages})
    return out


def _put_listing(path: str, data: dict) -> None:
    """pages.json (about, pdf, key and pre when set, pages), written only when its text changed."""
    text = dumps({k: data[k] for k in ("about", "pdf", "key", "pre", "pages") if data.get(k) not in (None, "")},
                 ensure_ascii=False, indent=1) + "\n"
    if text != (open(path, encoding="utf-8").read() if isfile(path) else None):
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)


def _get(site: str, path: str) -> bytes:
    """A website file, past the CDN's copy."""
    with urlopen(Request(f"{site.rstrip('/')}/{path}?t={int(time())}", headers={"User-Agent": "rulebook-pages"}), timeout=30) as reply:
        return reply.read()


def _chapters(raw: bytes) -> dict:
    """{file: source} of a data/rules.json's chapters."""
    return {c["file"]: c["source"].replace("\r\n", "\n") for c in loads(raw).get("chapters") or []
            if isinstance(c, dict) and CHAPTER_FILE.match(c.get("file") or "") and isinstance(c.get("source"), str)}


def site_inputs(site: str) -> Optional[dict]:
    """What pre_key() reads, of one build: {"page": index.html's stamp, "index": its text, "rules": data/rules.json's
    bytes, "cards": data/cards.json, "build": build.txt's text}; None when unreadable or a deploy came between (build.txt's
    stamp after them)."""
    try:
        index = _get(site, "index.html").decode("utf-8")
        rules, cards = _get(site, "data/rules.json"), loads(_get(site, "data/cards.json"))
        _chapters(rules)
        build = _get(site, "build.txt").decode("utf-8")
    except (OSError, ValueError, AttributeError, HTTPException):
        return None
    page = PAGE_STAMP.search(index)
    if not page or not isinstance(cards, dict) or build.split()[1:2] != [page.group(1)]:
        return None
    return {"page": page.group(1), "index": index, "rules": rules, "cards": cards, "build": build}


def _build(site: str) -> Optional[str]:
    """build.txt's text (a deploy's digest and page stamp), None when unreadable."""
    try:
        return _get(site, "build.txt").decode("utf-8")
    except (OSError, ValueError, HTTPException):
        return None


def _git(checkout: str, *args, codes=(0,)) -> tuple:
    """(exit code, stdout) of git in the checkout; OSError for another code."""
    got = run_process(["git", "-C", checkout, *args], stdout=PIPE, stderr=DEVNULL, timeout=30)
    if got.returncode not in codes:
        raise OSError(f"git {args[0]} failed")
    return got.returncode, got.stdout.decode()


def site_settled(checkout: str, seen: dict) -> bool:
    """Whether the website (seen: site_inputs()'s) was built from main with HEAD's SITE_PATHS: the commit of its build's
    date (cards.json's source; of a batch sharing one, the newest) has them, or none in HEAD's history has that date (a
    later main). False when unreadable or another branch's."""
    try:
        if not seen or seen["cards"].get("branch") != "main":
            return False
        at = int(datetime.fromisoformat(seen["cards"]["source"]).timestamp())
        builds = _git(checkout, "rev-list", "--topo-order", f"--since-as-filter=@{at}", f"--min-age={at}", "HEAD")[1].split()   # all history (--max-age stops at an older-dated commit)
        return not builds or _git(checkout, "diff-tree", "--quiet", "-r", builds[0], "HEAD", "--", *SITE_PATHS, codes=(0, 1))[0] == 0
    except (OSError, ValueError, KeyError, TypeError, AttributeError, TimeoutExpired):
        return False


def heads(checkout: str, urls) -> dict:
    """{url: HEAD's blob id of site/<url>, None when HEAD has none} (PLAIN_JS's shape)."""
    paths = {u: f"site/{unquote(u)}" for u in urls}
    if not paths:
        return {}
    out = _git(checkout, "--literal-pathspecs", "ls-tree", "-z", "HEAD", "--", *paths.values())[1]
    blobs = {path: meta.split()[2] for meta, path in (e.split("\t", 1) for e in out.split("\0") if e)}
    return {u: blobs.get(path) for u, path in paths.items()}


def wait_for_site(site: str, checkout: str, mine: dict, wait: float) -> Optional[dict]:
    """The website's build (site_inputs()) once it shows mine and is settled (site_settled()), else None after wait
    seconds (one look when wait <= 0). Looked at every SITE_POLL, and in the first SITE_FAST seconds when build.txt (read
    every SITE_POLL_FIRST) changed: a deploy."""
    start = monotonic()
    until, looked, build = start + wait, None, None
    while True:
        if looked is None or monotonic() - looked >= SITE_POLL or _build(site) not in (None, build):
            seen, looked = site_inputs(site), monotonic()
            if seen and _chapters(seen["rules"]) == mine and site_settled(checkout, seen):
                return seen
            build = seen["build"] if seen else None
        pause = SITE_POLL_FIRST if monotonic() < start + SITE_FAST else SITE_POLL
        if monotonic() + pause > until:
            return None
        sleep(pause)


# ========================
# Public API Functions
# ========================
def update_pages(checkout: str, site: str = SITE_URL, exe: str = "", wait: float = 0, full: bool = False) -> str:
    """
    The TTS pages brought to the website's rulebook, once it shows the checkout's chapters and is settled (site_settled())
    and Chrome drew that build with HEAD's plain-URL fonts and pictures (else, in wait seconds, waited for and drawn
    again); else nothing changes and the report says LAGS. Chrome is not started when pages.json's pre is the website's
    pre_key() and the PDF is there (PRE_SKIP; full: Chrome looks in any case).

    Returns:
        The report (one line; then a ::warning:: line when the pre-key held but the key changed)

    Raises:
        RuntimeError, TimeoutError, OSError: the pages could not be drawn (nothing written)
    """
    exe = exe or find_chrome()
    if not exe:
        raise RuntimeError(f"no Chrome: set CHROME or install one of {', '.join(CHROMES)}")
    mine = chapter_sources(checkout)
    if not mine:
        return f"Rulebook pages: no chapters in {RULEBOOK_SRC}; pages left as they are."
    folder, old, fonts, trees = join(checkout, RULEBOOK_PAGES), read_pages(checkout), book_fonts(checkout), _trees(checkout)
    has_pdf = isfile(join(folder, PDF_FILE))
    known = old.get("key", "") if has_pdf else ""
    until = monotonic() + wait
    while True:
        seen = wait_for_site(site, checkout, mine, until - monotonic())   # before the draw: that build or a later one
        if not seen:
            return (f"Rulebook PDF: {LAGS}'s chapters, pictures, web fonts, icons or code"
                    f"{f' after {wait / 60:.0f} minutes' if wait >= 90 else f' after {wait:.0f} seconds' if wait else ''}; left as it is.")
        pre = pre_key(seen, trees, fonts, exe)
        held = has_pdf and pre == old.get("pre")
        if held and PRE_SKIP and not full:
            return f"Rulebook PDF: {len(old['pages'])} pages, as they were (nothing they read changed: Chrome not started)."
        theirs, shots, printer, drawn, page, plain = draw(site, exe, lambda text: key(text, checkout, trees), known, fonts)
        try:
            met = page != seen["page"]                                   # a deploy mid-look: the new build is live
            lags = met or theirs != mine or plain != heads(checkout, plain)   # the CDN's old copy (same page: its data's)
            if not lags:
                warn = ("\n::warning::Rulebook PDF: the pre-key held but the pages' key changed: pre_key misses an input."
                        if held and drawn != known else "")
                if shots is None:
                    if pre != old.get("pre"):
                        _put_listing(join(folder, PAGES_JSON), {**old, "pre": pre})
                    return f"Rulebook PDF: {len(old['pages'])} pages, as they were (nothing they draw changed{'; pre-key held' if held else ''})."
                done = write_pages(folder, shots, printer.pdf, drawn, pre)
        finally:
            printer.close()
        if not lags:
            break
        if monotonic() + (0 if met else SITE_POLL) >= until:
            return f"Rulebook PDF: {LAGS} (Chrome drew another build, or an older copy of a web font or picture); left as it is."
        if not met:
            sleep(SITE_POLL)                            # the CDN's new copy: drawn again (that build: looked at at once)
    if not done["printed"]:
        return f"Rulebook PDF: {done['pages']} pages, as they were (key {drawn[:8]} kept).{warn}"
    size = getsize(join(folder, PDF_FILE))
    what = f"{len(done['changed'])} pages changed" if done["changed"] and len(done["changed"]) < done["pages"] else "printed"
    return f"Rulebook PDF: {done['pages']} pages, {what} ({size / 1e6:.1f} MB).{warn}"


def main(args: Optional[list] = None) -> int:
    """The command line (card-sync.yml): update_pages, its report on stdout; exit 3 when the website lags (LAGS)."""
    parser = ArgumentParser(prog="python -m commands.rulebook_pages", description="Print the rulebook's TTS PDF from the website.")
    parser.add_argument("checkout", help="the card repo's checkout (its site/rulebook; the PDF goes to TTS/Rulebook)")
    parser.add_argument("--site", default=SITE_URL, help="the website (default: the card browser)")
    parser.add_argument("--wait", type=float, default=SITE_WAIT, help="seconds to wait for the website to show HEAD's rulebook")
    parser.add_argument("--full", action="store_true", help="Chrome looks even when the pre-key held (the schedule's run)")
    options = parser.parse_args(args)
    try:
        report = update_pages(options.checkout, options.site, wait=options.wait, full=options.full)
    except (RuntimeError, TimeoutError, OSError, ValueError, KeyError, HTTPException) as e:
        print(f"Rulebook PDF not printed: {e}")
        return 1
    print(report)
    return 3 if LAGS in report else 0


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
