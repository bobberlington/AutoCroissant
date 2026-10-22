"""
Font check for the card PSDs: the PostScript names a card's visible type layers ask for, and which of them MDW/Fonts in
the card repo does not supply. TTSCardMaker's font-check.yml Action runs it on every push to main (job summary, commit
comment, a Discord message pinging the card makers' role); the bot's /fonts runs it over the whole repo.

    python -m commands.font_check --fonts <dir> [--psd <path>...] [--all <checkout root>]
                                  [--repo <owner/name> --after <sha> [--before <sha>] [--author <login>] [--whole-tree]]
                                  [--json <out>] [--discord <payload out>]
    FontIndex(dir) -> the faces of a fonts folder; psd_fonts(psd) -> {font: [layers]}; check_psd(psd, index) -> the missing ones
    markdown(report), discord_payload(report) -> the job summary / commit comment and the webhook body
    check_repo_fonts(interaction) -> the bot's /fonts (cache font_names.json: blob sha -> the names a blob asks for or supplies,
                                     or why it can't be read)
    find_fonts(repository) -> the fonts dir: PSD_TOOLS_FONT_DIRS, else the checkout's MDW/Fonts, else refresh_cache()

The fonts cache (FONTS_CACHE beside the pickles, gitignored) holds the font files at the top of MDW/Fonts, for the hosts
without a checkout (the jetson): card_bake's bake, psd_details' "supplied" flags and /fonts read it. A refresh lists the
subtree (git/trees/<branch>:MDW/Fonts, conditional on its ETag: a 304 when nothing changed), or takes the caller's tree,
and downloads only the blobs whose sha it doesn't hold (sha1-verified); a renamed file is copied, a gone one deleted.

A notification tool, never a gate: findings exit 0, nothing is reverted or rejected. --repo mode reads the PSDs the push
changed (before..after through the compare API; --whole-tree: every card PSD of the tree at after) by blob sha, each
blob's sha1 verified. --all parses the checkout's PSDs in a process pool (the parse is the whole cost: 0.1 s a PSD).
No Discord or config imports (the token: config through github_api, else GITHUB_TOKEN); stock psd-tools reads the
engine data, fontTools the font names (psd-tools' text engine fork: composite/text/fonts.py).
"""
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from io import BytesIO
from json import dump, dumps, load
from os import cpu_count, environ, fsync, getpid, makedirs, pathsep, remove, replace, walk
from os.path import abspath, basename, dirname, exists, expanduser, isdir, isfile, join, relpath, splitext
from re import sub
from threading import Lock
from traceback import format_exc
from typing import Optional

from fontTools.ttLib import TTCollection, TTFont
from psd_tools import PSDImage

import commands.github_api as github_api
from commands.github_api import api_url, blob_sha, fetch_tree, fetch_verified, get_with_retry, token_from_env
from global_config import CARD_BRANCH, CARD_REPO, DETAILS_JSON, FONT_ALIASES, FONTS_CACHE, LOCAL_DIR_LOC

# ========================
# Configuration
# ========================
FONTS_DIR = "MDW/Fonts"                     # the card repo's fonts folder; top level only (stoogedings-src/ is the icon font's source)
FONT_EXTENSIONS = (".ttf", ".otf", ".ttc", ".otc")
EXCLUDE_FOLDERS = ("MDW", "Markers")        # repo_traverser.EXCLUDE_FOLDERS: PSDs there are not cards
ROLE_ID = "546427268751294475"              # the card makers' role the Discord message pings
BITE = "{name} has pushed a card without adding the {font} font to MDW/Fonts, bite this user."   # the user's wording
DISCORD_LIMIT = 2000                        # characters per Discord message
NOBODY = "Someone"                          # the author when no login is known
CACHE_JSON = "font_names.json"              # the bot's cache: blob sha -> what the blob asks for (PSD) or supplies (font); gitignored
BATCH = 200                                 # PSDs the bot downloads per /fonts (a 15 MB blob took 1.4 s on 2026-10-21: ~6 min a batch)
ERRORS_SHOWN = 10                           # error lines in the /fonts reply; the console gets them all
INVISIBLE = "\ufeff\u200b\u200c\u200d\u2060"            # characters Photoshop keeps in a run but never draws
API_TIMEOUT = 60
FILES_PAGE = 300                            # the compare and commit APIs list at most this many files
COMMITS_PAGE = 250                          # the compare API lists at most this many commits
POOL_MIN = 8                                # --all parses this many PSDs or more in a process pool (half the cores)
POOL_CHUNK = 4                              # PSDs per pool task
FONT_DIRS_ENV = "PSD_TOOLS_FONT_DIRS"       # card_bake.FONT_DIRS_ENV: the fonts dir override (first entry), pathsep-separated
MANIFEST = ".manifest.json"                 # in FONTS_CACHE: {"tree", "etag", "files": {name: blob sha}}
_lock = Lock()                              # one /fonts at a time
_cache_lock = Lock()                        # one cache refresh at a time


# ========================
# Font names
# ========================
def norm(name: str) -> str:
    """Casefolded alphanumerics: the key of a loose match."""
    return sub(r"[^a-z0-9]", "", name.casefold())


def loose_keys(name: str) -> tuple:
    """norm(name) and, when it differs, norm of the name without its -Regular suffix."""
    keys = (norm(name), norm(sub(r"-?Regular$", "", name)))
    return keys if keys[1] != keys[0] else keys[:1]


def ps_name(name: str) -> str:
    """A name as FreeType (the fork's reader) reports PostScript names: ASCII 33-126 without []{}()<>/%."""
    return sub(r"[^!-~]|[\[\]{}()<>/%]", "", name)


def face_names(data: bytes) -> list:
    """
    One [PostScript name, [loose names]] per face of a font file (.ttc/.otc hold several, in index order).

    The PostScript name is name ID 6, else 4 (full name), else 1 (family), as FreeType would report it ("Kung Fu Panda"
    -> KungFuPanda); the loose names add the raw name and family + subfamily.
    """
    fonts = TTCollection(BytesIO(data), lazy=True).fonts if data[:4] == b"ttcf" else [TTFont(BytesIO(data), lazy=True)]
    out = []
    for font in fonts:
        table = font["name"]
        raw = next((n for n in (table.getDebugName(i) for i in (6, 4, 1)) if n), "")
        ps = ps_name(raw)
        family = (table.getDebugName(1) or "") + (table.getDebugName(2) or "")
        out.append([ps, list(dict.fromkeys(n for n in (ps, raw, family) if n))])
    return out


def _font_files(directory: str) -> list:
    """The font files at the top level of directory, sorted."""
    _, _, files = next(walk(directory), (directory, [], []))
    return [join(directory, f) for f in sorted(files) if f.lower().endswith(FONT_EXTENSIONS)]


class FontIndex:
    """The faces a fonts folder supplies, by exact PostScript name and by loose key; resolve() answers for one name."""

    def __init__(self, directory: str = ""):
        self.exact: set = set()
        self.loose: set = set()
        self.files = 0
        self.faces = 0
        self.errors: list = []                  # "file: why" for fonts fontTools can't read
        if directory:
            for path in _font_files(directory):
                try:
                    with open(path, "rb") as fh:
                        self.add(face_names(fh.read()))
                except Exception as e:          # one bad font never stops the check
                    self.errors.append(f"{basename(path)}: {e}")

    def add(self, faces: list) -> None:
        """Index one file's faces (face_names' output, also what the bot's cache holds)."""
        self.files += 1
        for ps, loose in faces:
            self.faces += 1
            if ps:
                self.exact.add(ps)
            for name in loose:
                self.loose.add(norm(name))

    def resolve(self, name: str) -> Optional[str]:
        """How name resolves: "exact", "alias" (FONT_ALIASES), "loose", or None when no face matches."""
        target = FONT_ALIASES.get(name, name)
        if target in self.exact:
            return "exact" if target == name else "alias"
        if any(key in self.loose for key in loose_keys(target)):
            return "loose"
        return None


# ========================
# PSD type layers
# ========================
def _value(item):
    return getattr(item, "value", item)


def _get(mapping, key: str, default=None):
    if mapping is None:
        return default
    value = mapping.get(key)
    return default if value is None else value


def layer_fonts(layer) -> list:
    """
    The PostScript names a type layer's text uses, in order of first use.

    Each StyleRun that covers text (length > 0) names its sheet's Font, else the Normal sheet's; text no run reaches keeps
    the last run's, a layer without runs uses the Normal sheet's, an empty layer asks for nothing. FontSet entries no run
    uses (AdobeInvisFont, Photoshop's default) are not required; a run over only invisible characters (a leading
    byte-order mark) asks for nothing. Run lengths count UTF-16 units.
    """
    resources, engine = layer.resource_dict, layer.engine_dict
    font_set = [str(_value(_get(f, "Name", ""))) for f in _get(resources, "FontSet", ())]
    if not font_set:
        return []
    sheets = _get(resources, "StyleSheetSet")
    normal_index = int(_value(_get(resources, "TheNormalStyleSheet", 0)))
    normal = _get(sheets[normal_index], "StyleSheetData") if sheets is not None and 0 <= normal_index < len(sheets) else None
    text = _get(_get(engine, "Editor"), "Text")
    text = str(_value(text)) if text is not None else layer.text
    units = len(text.encode("utf-16-le")) // 2

    def font_of(sheet) -> str:
        index = 0
        for candidate in (sheet, normal):
            if candidate is not None and candidate.get("Font") is not None:
                index = int(_value(candidate.get("Font")))
                break
        return font_set[index] if 0 <= index < len(font_set) else font_set[0]

    encoded = text.encode("utf-16-le")
    names, unit = [], 0
    style_run = _get(engine, "StyleRun")
    for data, length in zip(_get(style_run, "RunArray", ()), _get(style_run, "RunLengthArray", ())):
        n = int(_value(length))
        covered = encoded[2 * min(unit, units):2 * min(unit + n, units)].decode("utf-16-le", "ignore")
        if covered and covered.strip(INVISIBLE):        # a run over only a byte-order mark or zero-width characters draws nothing
            names.append(font_of(_get(_get(data, "StyleSheet"), "StyleSheetData")))
        unit += n
    if unit < units:
        names.append(names[len(names) - 1] if names else font_of(None))
    return list(dict.fromkeys(names))


def psd_fonts(psd) -> dict:
    """{font: [layer names]} over the visible type layers of a PSDImage, PSD bytes or a path; hidden layers are ignored."""
    if isinstance(psd, (bytes, bytearray)):
        psd = PSDImage.open(BytesIO(psd))
    elif isinstance(psd, str):
        psd = PSDImage.open(psd)
    out: dict = {}
    for layer in psd.descendants():
        if layer.kind == "type" and layer.is_visible():
            for name in layer_fonts(layer):
                out.setdefault(name, []).append(layer.name)
    return out


def check_psd(psd, index: FontIndex) -> dict:
    """
    The fonts one PSD asks for that the index lacks.

    Args:
        psd: A PSDImage, PSD bytes or a path
        index: The fonts folder's faces

    Returns:
        {missing font: [layer names]}
    """
    return {font: layers for font, layers in psd_fonts(psd).items() if index.resolve(font) is None}


# ========================
# Reports
# ========================
@dataclass
class Card:
    """One PSD asking for a font MDW/Fonts lacks."""
    path: str
    author: str
    missing: dict                               # font -> [layer names]


@dataclass
class Report:
    """One check's findings: what was read, which cards ask for a font the index lacks, and the errors."""
    checked: int = 0                            # PSDs read
    cards: list = field(default_factory=list)   # Card, in path order
    fonts_added: list = field(default_factory=list)   # files MDW/Fonts gained in the push
    fonts_files: int = 0
    fonts_faces: int = 0
    errors: list = field(default_factory=list)  # "path: why" (a bad blob, an unreadable PSD or font)
    remaining: int = 0                          # PSDs not read yet (the bot's batches)
    fonts_unread: int = 0                       # MDW/Fonts files that failed to download: cards is left empty

    def by_font(self) -> dict:
        """{font: {path: [layers]}}, fonts sorted."""
        out: dict = {}
        for card in self.cards:
            for font, layers in card.missing.items():
                out.setdefault(font, {})[card.path] = layers
        return {font: out[font] for font in sorted(out)}

    def as_dict(self) -> dict:
        """The --json report."""
        return {"checked": self.checked, "remaining": self.remaining, "fonts": {"files": self.fonts_files, "faces": self.fonts_faces},
                "fonts_added": self.fonts_added, "missing": self.by_font(),
                "cards": [{"path": c.path, "author": c.author, "missing": c.missing} for c in self.cards], "errors": self.errors}


def _card_name(path: str) -> str:
    """The card's name from its PSD path (stem, underscores as spaces): card_models.card_name, copied so the font-check
    Action's closure stays free of pickle and StatsDatabase."""
    return splitext(basename(path))[0].replace("_", " ")


def markdown(report: Report) -> str:
    """The job summary and commit comment: per PSD its missing fonts and layers, the fonts the push added, errors."""
    out = ["## Font check", "",
           f"{report.checked} card PSD{'s' if report.checked != 1 else ''} read; MDW/Fonts: {report.fonts_files} files, {report.fonts_faces} faces."]
    if report.remaining:
        out.append(f"{report.remaining} not read yet.")
    if report.fonts_added:
        out += ["", "**Fonts added to MDW/Fonts:** " + ", ".join(f"`{f}`" for f in report.fonts_added)]
    out.append("")
    if report.cards:
        out.append("Fonts asked for that MDW/Fonts does not supply (card: **font** (layers)):")
        for card in report.cards:
            fonts = "; ".join(f"**{font}** ({', '.join(layers)})" for font, layers in card.missing.items())
            out.append(f"- `{card.path}` ({card.author}): {fonts}")
    else:
        out.append("Every font the cards ask for is in MDW/Fonts.")
    if report.errors:
        out += ["", "Errors:"] + [f"- {e}" for e in report.errors]
    return "\n".join(out) + "\n"


def discord_payload(report: Report) -> dict:
    """
    The webhook body: the role mention once, then one BITE line per (author, font) with the cards and layers indented
    below it; {} when nothing is missing. Only the role may be pinged.
    """
    pairs: dict = {}
    for card in report.cards:
        for font, layers in card.missing.items():
            pairs.setdefault((card.author, font), []).append(f"    {_card_name(card.path)} ({', '.join(layers)})")
    if not pairs:
        return {}
    lines = [f"<@&{ROLE_ID}>"]
    for (author, font), cards in sorted(pairs.items()):
        lines.append(BITE.format(name=author, font=font))
        lines += cards
    content = "\n".join(lines)
    if len(content) > DISCORD_LIMIT:
        tail = "\n... (the rest is in the commit comment)"
        content = content[:DISCORD_LIMIT - len(tail)].rsplit("\n", 1)[0] + tail
    return {"content": content, "allowed_mentions": {"roles": [ROLE_ID]}}


def discord_text(report: Report) -> str:
    """The bot's /fonts reply: missing fonts -> cards (layers), the fonts folder count, the cache state."""
    out = [f"MDW/Fonts: {report.fonts_files} files, {report.fonts_faces} faces. {report.checked} card PSDs read."]
    if report.remaining:
        out.append(f"Warming the cache: {report.remaining} card PSDs are not read yet (at most {BATCH} downloads per run); "
                   "run /fonts again for the rest. Partial results:")
    missing = report.by_font()
    if report.fonts_unread:
        out.append(f"{report.fonts_unread} MDW/Fonts file{'s' if report.fonts_unread != 1 else ''} failed to download, so no "
                   "card is reported missing a font this time; run /fonts again.")
    elif missing:
        for font, cards in missing.items():
            out.append(f"**{font}**: " + "; ".join(f"{_card_name(p)} ({', '.join(layers)})" for p, layers in cards.items()))
    else:
        out.append("Every font the cards ask for is in MDW/Fonts.")
    if report.errors:
        more = len(report.errors) - ERRORS_SHOWN
        out.append("Errors: " + "; ".join(report.errors[:ERRORS_SHOWN]) + (f"; and {more} more (in the console)" if more > 0 else ""))
    return "\n".join(out)


# ========================
# Sources: files, a checkout, the GitHub API
# ========================
def is_card_psd(path: str) -> bool:
    """A PSD under the card folders (not MDW/, not Markers/), by repo path."""
    return path.lower().endswith(".psd") and not any(folder in EXCLUDE_FOLDERS for folder in path.split("/"))


def _record(report: Report, index: FontIndex, path: str, author: str, fonts: dict) -> None:
    """Count one read PSD and list it when a font it asks for is not in the index."""
    report.checked += 1
    missing = {font: layers for font, layers in fonts.items() if index.resolve(font) is None}
    if missing:
        report.cards.append(Card(path, author, missing))


def _check_one(report: Report, index: FontIndex, path: str, author: str, psd) -> None:
    """Read one PSD into the report; an unreadable one is an error line, never a stop."""
    try:
        fonts = psd_fonts(psd)
    except Exception as e:                      # one bad PSD never stops the check
        report.errors.append(f"{path}: {e}")
        return
    _record(report, index, path, author, fonts)


def _fonts_of(path: str) -> tuple:
    """psd_fonts(path) in a pool worker: (fonts, None), or (None, the error text)."""
    try:
        return psd_fonts(path), None
    except Exception as e:                      # one bad PSD never stops the check
        return None, str(e)


def check_files(paths: list, index: FontIndex, author: str = NOBODY) -> Report:
    """
    PSD files given on the command line.

    Args:
        paths: The files
        index: The fonts folder's faces
        author: Who to name for them

    Returns:
        Report
    """
    report = Report(fonts_files=index.files, fonts_faces=index.faces, errors=list(index.errors))
    for path in paths:
        _check_one(report, index, path, author, path)
    return report


def check_checkout(root: str, index: FontIndex) -> Report:
    """
    Every card PSD under a checkout (MDW/ and Markers/ excluded), in path order.

    Args:
        root: The checkout
        index: The fonts folder's faces

    Returns:
        Report; from POOL_MIN PSDs the parses run in a process pool (half the cores), merged in path order
    """
    report = Report(fonts_files=index.files, fonts_faces=index.faces, errors=list(index.errors))
    found = []
    for folder, subdirs, files in walk(root):
        subdirs[:] = sorted(d for d in subdirs if d not in EXCLUDE_FOLDERS and d != ".git")
        for name in sorted(files):
            rel = relpath(join(folder, name), root).replace("\\", "/")
            if is_card_psd(rel):
                found.append((rel, join(folder, name)))
    if len(found) < POOL_MIN:
        for rel, path in found:
            _check_one(report, index, rel, NOBODY, path)
        return report
    workers = max(1, min((cpu_count() or 2) // 2, -(-len(found) // POOL_CHUNK)))
    with ProcessPoolExecutor(workers) as pool:
        for (rel, _), (fonts, error) in zip(found, pool.map(_fonts_of, [path for _, path in found], chunksize=POOL_CHUNK)):
            if error is not None:
                report.errors.append(f"{rel}: {error}")
            else:
                _record(report, index, rel, NOBODY, fonts)
    return report


def _get_json(repository: str, path: str, missing_ok: bool = False):
    """One GET of the repo API as JSON; None on a 404 when missing_ok."""
    reply = get_with_retry(api_url(repository, path), headers=github_api.HEADERS, timeout=API_TIMEOUT)
    if missing_ok and reply.status_code == 404:
        return None
    reply.raise_for_status()
    return reply.json()


def _login(commit: dict, fallback: str) -> str:
    return (commit.get("author") or {}).get("login") or fallback      # null when the commit email is not a GitHub account's


def pushed_files(repository: str, before: str, after: str, author: str = NOBODY) -> tuple:
    """
    What a push changed.

    before..after through the compare API; the files of after alone when before is empty or all zeros (a dispatch, a new
    branch) or no longer exists (a force push). With several commits each is read for who touched which file last. The
    APIs page at FILES_PAGE files and COMMITS_PAGE commits: past them a note says files may be credited to the wrong author.

    Args:
        repository: owner/name
        before: The push's previous head (github.event.before)
        after: The push's head
        author: Who to name when a commit's author has no login (github.actor)

    Returns:
        ({card PSD path: (blob sha at after, author login)}, [fonts MDW/Fonts gained], [notes])
    """
    notes = []
    data = _get_json(repository, f"compare/{before}...{after}", missing_ok=True) if before and set(before) != {"0"} else None
    if data is None:
        if before and set(before) != {"0"}:
            notes.append(f"the previous head {before[:8]} is gone (a force push?); only the head commit's files were checked")
        head = _get_json(repository, f"commits/{after}")
        commits, files = [head], head.get("files", [])
    else:
        commits, files = data.get("commits", []), data.get("files", [])
    if len(files) >= FILES_PAGE:
        notes.append(f"the push changed {FILES_PAGE} files or more; only the first {FILES_PAGE} were checked")
    if len(commits) >= COMMITS_PAGE:
        notes.append(f"the push has {COMMITS_PAGE} commits or more; later files may be credited to the wrong author")
    who: dict = {}
    if len(commits) > 1:
        for commit in commits:                  # oldest first: the last writer wins
            info = _get_json(repository, f"commits/{commit['sha']}")
            if len(info.get("files", [])) >= FILES_PAGE:
                notes.append(f"commit {commit['sha'][:8]} lists {FILES_PAGE} files or more; later files may be credited to the wrong author")
            for f in info.get("files", []):
                who[f["filename"]] = _login(info, author)
    default = _login(commits[0], author) if commits else author
    psds = {f["filename"]: (f["sha"], who.get(f["filename"], default))
            for f in files if f.get("status") != "removed" and is_card_psd(f["filename"])}
    fonts = [basename(f["filename"]) for f in files
             if f["filename"].startswith(FONTS_DIR + "/") and f["filename"].count("/") == 2
             and f["filename"].lower().endswith(FONT_EXTENSIONS) and f.get("status") in ("added", "renamed", "copied")]
    return psds, sorted(fonts), notes


def tree_blobs(repository: str, ref: str = "") -> dict:
    """
    The repo's files by blob sha.

    Args:
        repository: owner/name
        ref: A commit or tree sha; "" for the branch head

    Returns:
        {path: blob sha}; raises RuntimeError when GitHub truncates the listing
    """
    if ref:
        data = _get_json(repository, f"git/trees/{ref}?recursive=1")
    else:
        reply = fetch_tree(repository, timeout=API_TIMEOUT)
        reply.raise_for_status()
        data = reply.json()
    if data.get("truncated"):
        raise RuntimeError("the tree listing is truncated")
    return {e["path"]: e["sha"] for e in data.get("tree", []) if e.get("type") == "blob"}


def check_blobs(repository: str, psds: dict, index: FontIndex, cache: Optional[dict] = None, batch: int = 0) -> Report:
    """
    PSDs read by blob sha through the API, in path order.

    Args:
        repository: owner/name
        psds: {path: (blob sha, author)}
        index: The fonts folder's faces
        cache: {sha: {font: [layers]}, or the error text of a blob psd-tools can't read}: known blobs are not downloaded,
            new ones are added; a failed download is not (None: no cache)
        batch: Cap on the downloads (0: none); the rest count as remaining

    Returns:
        Report
    """
    report = Report(fonts_files=index.files, fonts_faces=index.faces, errors=list(index.errors))
    downloads = 0
    for path in sorted(psds):
        sha, author = psds[path]
        fonts = cache.get(sha) if cache is not None else None
        if fonts is None:
            if batch and downloads >= batch:
                report.remaining += 1
                continue
            downloads += 1
            try:
                data = fetch_verified(repository, sha, timeout=API_TIMEOUT)
            except Exception as e:              # tried again next run
                report.errors.append(f"{path}: {e}")
                continue
            try:
                fonts = psd_fonts(data)
            except Exception as e:              # the same bytes fail the same way: cached
                fonts = str(e)
            if cache is not None:
                cache[sha] = fonts
        if isinstance(fonts, str):
            report.errors.append(f"{path}: {fonts}")
        else:
            _record(report, index, path, author, fonts)
    return report


# ========================
# The fonts cache
# ========================
@dataclass
class FontsCache:
    """Where the fonts are and what the refresh did."""
    directory: str
    source: str = "cache"                       # "env", "checkout" or "cache"
    tree: str = ""                              # the MDW/Fonts tree sha, when listed
    files: dict = field(default_factory=dict)   # name -> blob sha of the font files in directory
    downloaded: int = 0
    copied: int = 0                             # same blob under a new name: no download
    removed: int = 0
    requests: int = 0                           # API requests made (listing and blobs)
    failed: dict = field(default_factory=dict)  # name -> why its current blob is not in directory
    error: str = ""                             # the listing failed: directory is as the last refresh left it


def tree_fonts(blobs: dict) -> dict:
    """{file name: blob sha} of the font files at the top of MDW/Fonts, from {repo path: blob sha}."""
    return {path[len(FONTS_DIR) + 1:]: sha for path, sha in blobs.items()
            if path.startswith(FONTS_DIR + "/") and path.count("/") == 2 and path.lower().endswith(FONT_EXTENSIONS)}


def local_fonts() -> tuple:
    """(dir, "env") from PSD_TOOLS_FONT_DIRS, else (the checkout's MDW/Fonts, "checkout") when it exists, else ("", "")."""
    dirs = environ.get(FONT_DIRS_ENV, "")
    if dirs:
        return dirs.split(pathsep)[0], "env"
    checkout = join(expanduser(LOCAL_DIR_LOC), FONTS_DIR)
    return (checkout, "checkout") if isdir(checkout) else ("", "")


def _load_manifest(directory: str) -> dict:
    try:
        with open(join(directory, MANIFEST), encoding="utf-8") as fh:
            data = load(fh)
        if isinstance(data, dict) and isinstance(data.get("files"), dict):
            return data
    except (OSError, ValueError):
        pass                                    # missing or corrupt: every file is checked again
    return {"tree": "", "etag": "", "files": {}}


def _write(path: str, write) -> None:
    """write(file handle) into a temp file beside path, fsync, replace: readers never see half a file."""
    temp_path = join(dirname(path), f".{basename(path)}.{getpid()}.tmp")
    try:
        with open(temp_path, "wb") as fh:
            write(fh)
            fh.flush()
            fsync(fh.fileno())
        replace(temp_path, path)
    except BaseException:
        try:
            remove(temp_path)
        except OSError:
            pass
        raise


def _list_fonts(repository: str, etag: str) -> Optional[tuple]:
    """(tree sha, {name: blob sha}, etag) of MDW/Fonts on the branch; None on a 304 (etag still current)."""
    headers = {**github_api.HEADERS, **({"If-None-Match": etag} if etag else {})}
    reply = get_with_retry(api_url(repository, f"git/trees/{CARD_BRANCH}:{FONTS_DIR}"), headers=headers, timeout=API_TIMEOUT)
    if reply.status_code == 304:
        return None
    reply.raise_for_status()
    data = reply.json()
    blobs = {f"{FONTS_DIR}/{e['path']}": e["sha"] for e in data.get("tree", []) if e.get("type") == "blob"}
    return data.get("sha", ""), tree_fonts(blobs), reply.headers.get("ETag", "")


def refresh_cache(repository: str = CARD_REPO, files: Optional[dict] = None, directory: str = FONTS_CACHE, tree: str = "") -> FontsCache:
    """
    Bring the fonts cache to the MDW/Fonts of the branch (or of the caller's tree).

    Args:
        repository: owner/name of the card repo
        files: {name: blob sha} from a tree the caller holds (tree_fonts): no listing request; None lists the subtree
        directory: The cache (FONTS_CACHE)
        tree: The MDW/Fonts tree sha, when the caller knows it

    Returns:
        FontsCache; download and listing failures are in failed and error, never raised. Requests: 1 listing (none
        with files; a 304 when the subtree is unchanged), then 1 per blob the cache lacks
    """
    with _cache_lock:
        directory = abspath(directory)
        makedirs(directory, exist_ok=True)
        manifest = _load_manifest(directory)
        held = {name: sha for name, sha in manifest["files"].items() if isinstance(sha, str) and isfile(join(directory, name))}
        complete = held == manifest["files"]
        result = FontsCache(directory, tree=tree)
        etag = ""
        if files is None:
            result.requests += 1
            try:
                listed = _list_fonts(repository, manifest.get("etag", "") if complete else "")
            except Exception as e:              # no network, a 5xx after the retries: the cache stands
                result.error, result.files, result.tree = f"MDW/Fonts listing: {e}", held, manifest.get("tree", "")
                return result
            if listed is None:                  # 304: nothing changed since the last full refresh
                result.files, result.tree = held, manifest.get("tree", "")
                return result
            result.tree, files, etag = listed
        if not files:                           # a tree without fonts never empties the cache
            result.error, result.files = "no font files under MDW/Fonts", held
            return result
        if complete and held == files:
            result.files = held
            if (result.tree and result.tree != manifest.get("tree")) or (etag and etag != manifest.get("etag")):
                _write_manifest(directory, result.tree, etag or manifest.get("etag", ""), held)
            return result
        by_sha = {sha: name for name, sha in held.items()}
        now = {}
        for name in sorted(files):
            sha, path = files[name], join(directory, name)
            if held.get(name) == sha:
                now[name] = sha
                continue
            try:
                data = b""
                if sha in by_sha:               # same blob under another name (checked: that file may be rewritten already)
                    with open(join(directory, by_sha[sha]), "rb") as fh:
                        data = fh.read()
                if data and blob_sha(data) == sha:
                    result.copied += 1
                else:
                    result.requests += 1
                    data = fetch_verified(repository, sha, timeout=API_TIMEOUT)
                    result.downloaded += 1
                _write(path, lambda fh: fh.write(data))
                now[name] = sha
            except Exception as e:              # tried again next refresh; an older file of that name stays
                result.failed[name] = str(e)
                if name in held:
                    now[name] = held[name]
        for path in _font_files(directory):     # gone from the tree, or not the cache's
            name = basename(path)
            if name not in files:
                try:
                    remove(path)
                    result.removed += 1
                except OSError:
                    pass
                now.pop(name, None)
        done = not result.failed and all(now.get(name) == sha for name, sha in files.items())
        _write_manifest(directory, result.tree if done else "", etag if done else "", now)
        result.files = now
        return result


def _write_manifest(directory: str, tree: str, etag: str, files: dict) -> None:
    text = dumps({"tree": tree, "etag": etag, "files": {name: files[name] for name in sorted(files)}}, indent=1)
    _write(join(directory, MANIFEST), lambda fh: fh.write(text.encode("utf-8")))


def find_fonts(repository: str = CARD_REPO, files: Optional[dict] = None, directory: str = FONTS_CACHE) -> FontsCache:
    """
    The fonts dir: PSD_TOOLS_FONT_DIRS's first entry, else the checkout's MDW/Fonts, else the cache, refreshed.

    Args:
        repository: owner/name of the card repo (the cache's source)
        files: {name: blob sha} from the caller's tree (refresh_cache)
        directory: The cache

    Returns:
        FontsCache (source "env" or "checkout": nothing refreshed)
    """
    local, source = local_fonts()
    if local:
        return FontsCache(local, source)
    return refresh_cache(repository, files, directory)


# ========================
# The bot: /fonts
# ========================
def _load_cache(path: str) -> dict:
    if exists(path):
        try:
            with open(path, encoding="utf-8") as fh:
                data = load(fh)
            if isinstance(data, dict) and isinstance(data.get("fonts"), dict) and isinstance(data.get("psds"), dict):
                return data
        except (OSError, ValueError):
            pass                                # a corrupt cache is rebuilt
    return {"fonts": {}, "psds": {}}


def _save_cache(path: str, cache: dict) -> None:
    """card_models.save_pickle's write (tmp, fsync, replace), copied: the Action's closure has no card_models."""
    temp_path = path + ".tmp"
    try:
        with open(temp_path, "w", encoding="utf-8") as fh:
            dump(cache, fh)
            fh.flush()
            fsync(fh.fileno())
        replace(temp_path, path)
    except BaseException:
        try:
            remove(temp_path)
        except OSError:
            pass
        raise


def _details_fonts(path: str) -> dict:
    """{blob sha: {font: [layer names]}} from the details file /update_stats gathers (its layers are paths); {} without it."""
    from commands.psd_details import load as load_details          # lazy: psd_details imports this module
    out: dict = {}
    for sha, record in load_details(path).items():
        fonts = record.get("fonts") if isinstance(record, dict) else None
        if isinstance(fonts, dict) and all(isinstance(entry, dict) for entry in fonts.values()):
            out[sha] = {font: [basename(layer) for layer in entry.get("layers", [])] for font, entry in fonts.items()}
    return out


def repo_fonts(repository: str = CARD_REPO, cache_path: str = CACHE_JSON, batch: int = BATCH, details_path: str = DETAILS_JSON,
               fonts_cache: str = "") -> Report:
    """
    Every card PSD on the card repo's branch against its MDW/Fonts, through the API with a per-blob cache.

    The fonts come from the fonts cache, refreshed against the same tree listing (the first fill downloads 276 files,
    226 MB; later runs only the blobs it lacks). The first run reads up to batch PSDs (the whole repo is ~813 PSDs,
    15.6 GB: about 25 min of downloads at the 11 MB/s measured on 2026-10-21); later runs download only new blobs. Blobs
    the details file already describes (psd_details.json, gathered by /update_stats) are not downloaded either. Blobs no
    longer in the tree leave the cache. When a font file fails to download, any card could be asking for one of its
    faces: the PSDs are still read and cached, but no card is reported (fonts_unread).

    Args:
        repository: owner/name
        cache_path: The cache file (CACHE_JSON), written whatever happens
        batch: PSD downloads per call
        details_path: The details file whose fonts seed the cache
        fonts_cache: The fonts cache ("": FONTS_CACHE beside cache_path)

    Returns:
        Report, with remaining = the PSDs not read yet
    """
    cache = _load_cache(cache_path)
    for sha, fonts in _details_fonts(details_path).items():
        cache["psds"].setdefault(sha, fonts)
    blobs = tree_blobs(repository)
    files = tree_fonts(blobs)
    held = refresh_cache(repository, files, fonts_cache or join(dirname(cache_path), FONTS_CACHE))
    index = FontIndex()
    unread = 0
    try:
        for name in sorted(files):
            sha = files[name]
            faces = cache["fonts"].get(sha)     # face names by blob: a file is parsed once
            if faces is None:
                if held.files.get(name) != sha:
                    index.errors.append(f"{name}: {held.failed.get(name) or held.error or 'not in the fonts cache'}")
                    unread += 1
                    continue
                try:
                    with open(join(held.directory, name), "rb") as fh:
                        faces = face_names(fh.read())
                except Exception as e:          # fontTools can't read it: cached, like a PSD's
                    faces = str(e)
                cache["fonts"][sha] = faces
            if isinstance(faces, str):
                index.errors.append(f"{name}: {faces}")
            else:
                index.add(faces)
        psds = {p: (s, NOBODY) for p, s in blobs.items() if is_card_psd(p)}
        report = check_blobs(repository, psds, index, cache["psds"], batch)
        if unread:
            report.fonts_unread, report.cards = unread, []
        keep = set(files.values()) | {s for s, _ in psds.values()}
        cache["fonts"] = {s: v for s, v in cache["fonts"].items() if s in keep}
        cache["psds"] = {s: v for s, v in cache["psds"].items() if s in keep}
    finally:
        _save_cache(cache_path, cache)          # what was downloaded stays, whatever happened after
    return report


def check_repo_fonts(interaction) -> None:
    """/fonts: the card repo's missing fonts -> cards (layers), BATCH new PSDs per call."""
    from commands.query_card import card_repo                 # lazy: standalone use needs no Discord
    from commands.utils import queue_long_message
    if not _lock.acquire(blocking=False):
        queue_long_message(interaction, "A font check is already running; wait for its reply.")
        return
    try:
        report = repo_fonts(card_repo.repository)
        if len(report.errors) > ERRORS_SHOWN:
            print("Font check errors:\n" + "\n".join(report.errors))
        text = discord_text(report)
    except Exception as e:                      # the cause goes to the console, the one line to Discord
        print(format_exc())
        text = f"Font check failed: {e}"
    finally:
        _lock.release()
    queue_long_message(interaction, text)


# ========================
# Command line
# ========================
def main(args: Optional[list] = None) -> int:
    """The command line: files, a checkout or the API; the markdown on stdout, --json and --discord files when asked."""
    parser = ArgumentParser(prog="python -m commands.font_check",
                            description="Which fonts card PSDs ask for that MDW/Fonts does not supply. Reports, never rejects: "
                                        "findings exit 0.")
    parser.add_argument("--fonts", required=True, metavar="DIR", help="the fonts folder (a checkout's MDW/Fonts)")
    parser.add_argument("--psd", nargs="*", default=[], metavar="PATH", help="PSD files to check")
    parser.add_argument("--all", default="", metavar="ROOT", help="every card PSD under a checkout (MDW/ and Markers/ excluded)")
    parser.add_argument("--repo", default="", metavar="OWNER/NAME", help="read the pushed PSDs by blob sha through the GitHub API")
    parser.add_argument("--before", default="", metavar="SHA", help="the push's previous head (github.event.before); empty: --after's own files")
    parser.add_argument("--after", default="", metavar="SHA", help="the push's head (github.sha)")
    parser.add_argument("--author", default=NOBODY, metavar="LOGIN", help="who to name when a commit's author has no login (github.actor)")
    parser.add_argument("--whole-tree", action="store_true", help="with --repo: every card PSD of the tree at --after (~813 blob downloads, 16 GB)")
    parser.add_argument("--json", default="", metavar="OUT", help="write the report as JSON here")
    parser.add_argument("--discord", default="", metavar="OUT", help="write the Discord webhook payload here ({} when nothing is missing)")
    options = parser.parse_args(args)
    token_from_env()
    index = FontIndex(options.fonts)
    if options.repo:
        try:
            if options.whole_tree:
                psds = {p: (s, options.author) for p, s in tree_blobs(options.repo, options.after).items() if is_card_psd(p)}
                fonts_added, notes = [], []
            else:
                psds, fonts_added, notes = pushed_files(options.repo, options.before, options.after, options.author)
            report = check_blobs(options.repo, psds, index)
        except Exception as e:                  # the API failed (a 5xx after the retries, no network, a truncated tree): one line
            report, fonts_added, notes = Report(), [], [f"GitHub API: {e}"]
        report.fonts_added, report.errors = fonts_added, report.errors + notes
    elif options.all:
        report = check_checkout(options.all, index)
    else:
        report = check_files(options.psd, index, options.author)
    print(markdown(report), end="")
    if options.json:
        with open(options.json, "w", encoding="utf-8") as fh:
            dump(report.as_dict(), fh, indent=1)
    if options.discord:
        with open(options.discord, "w", encoding="utf-8") as fh:
            dump(discord_payload(report), fh)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# ========================
# Module Exports
# ========================
__all__ = [
    'API_TIMEOUT',
    'BATCH',
    'BITE',
    'CACHE_JSON',
    'COMMITS_PAGE',
    'Card',
    'ERRORS_SHOWN',
    'EXCLUDE_FOLDERS',
    'FILES_PAGE',
    'FONTS_DIR',
    'FONT_DIRS_ENV',
    'FontIndex',
    'FontsCache',
    'MANIFEST',
    'POOL_CHUNK',
    'POOL_MIN',
    'ROLE_ID',
    'Report',
    'check_blobs',
    'check_checkout',
    'check_files',
    'check_psd',
    'check_repo_fonts',
    'discord_payload',
    'discord_text',
    'face_names',
    'find_fonts',
    'is_card_psd',
    'layer_fonts',
    'local_fonts',
    'loose_keys',
    'main',
    'markdown',
    'norm',
    'ps_name',
    'psd_fonts',
    'pushed_files',
    'refresh_cache',
    'repo_fonts',
    'tree_blobs',
    'tree_fonts',
]
