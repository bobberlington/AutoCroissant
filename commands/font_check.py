"""
Font check for the card PSDs: the PostScript names a card's visible type layers ask for, and which of them MDW/Fonts in
the card repo does not supply. TTSCardMaker's card-guards.yml runs it in its check job on every push to main that touches a
PSD or MDW/Fonts (font-check.yml until 2026-10-25): job summary, commit comment, and the fonts the push made missing on a
card (its PSD before the push did not ask for them; owner 2026-11-11) in the standards channel's one post, pinging the
card makers' role (standards_post); the bot's /fonts reads the report the card data build writes from psd_details.json
into its manifest (fonts_report here, card_sync.fonts_text).

    python -m commands.font_check --fonts <dir> [--psd <path>...] [--all <checkout root>]
                                  [--repo <owner/name> --after <sha> [--before <sha>] [--author <login>] [--whole-tree]
                                   [--clone <dir>]]
                                  [--json <out>] [--discord <payload out>]
    FontIndex(dir) -> the faces of a fonts folder; psd_fonts(psd) -> {font: [layers]}
    markdown(report), discord_payload(report) -> the job summary / commit comment and this check's part of the post
    old_blobs(repo, before, psds) -> each pushed PSD's blob before the push (the checks' "new" notes)
    local_fonts() -> the fonts dir: PSD_TOOLS_FONT_DIRS, else the checkout's MDW/Fonts
    save_names(dir, path), load_names(path) -> a fonts dir's index as a small JSON file (NAMES_ENV)
    shared_tables(data), split_tables(data) -> a font's records on the same bytes, the font with each its own copy

A notification tool, never a gate: findings exit 0, nothing is reverted or rejected. --repo mode reads the PSDs the push
changed (before..after through the compare API, or --clone's diff-tree when GitHub can't list them; --whole-tree: every
card PSD of the tree at after) by blob sha, each blob's sha1 verified. --all parses the checkout's PSDs in a process pool
(the parse is the whole cost: 0.1 s a PSD).
No Discord or config imports (the token: config through github_api, else GITHUB_TOKEN); stock psd-tools reads the
engine data, fontTools the font names (psd-tools' text engine fork: composite/text/fonts.py).
"""
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from io import BytesIO
from json import dump, load as json_load
from os import cpu_count, environ, pathsep, walk
from os.path import basename, expanduser, isdir, join, relpath
from re import sub
from struct import error as StructError, pack_into, unpack_from
from subprocess import run
from typing import Optional

import freetype  # noqa: F401  before uharfbuzz (fontTools loads it): else macOS binds FreeType's HarfBuzz to it, segfaults
from fontTools.ttLib import TTCollection, TTFont
from psd_tools import PSDImage
from requests import HTTPError

import commands.github_api as github_api
from commands.github_api import api_url, blob_sha, fetch_tree, fetch_verified, get_with_retry, token_from_env
from global_config import FONT_DIRS_ENV, LOCAL_DIR_LOC

# ========================
# Configuration
# ========================
FONTS_DIR = "MDW/Fonts"                     # the card repo's fonts folder; top level only (stoogedings-src/ is the icon font's source)
FONT_EXTENSIONS = (".ttf", ".otf", ".ttc", ".otc")
EXCLUDE_FOLDERS = ("MDW", "Markers")        # card_data.EXCLUDE_FOLDERS: PSDs there are not cards
NOBODY = "Someone"                          # the author when no login is known
INVISIBLE = "\ufeff\u200b\u200c\u200d\u2060"            # characters Photoshop keeps in a run but never draws
API_TIMEOUT = 60
FILES_PAGE = 300                            # the compare and commit APIs list at most this many files
COMMITS_PAGE = 250                          # the compare API lists at most this many commits
POOL_MIN = 8                                # --all parses this many PSDs or more in a process pool (half the cores)
POOL_CHUNK = 4                              # PSDs per pool task
NAMES_ENV = "CARD_FONT_NAMES"               # a save_names() file: the details' index without the fonts (card-sync's data job)
SFNT_SUM = 0xB1B0AFBA                       # what head.checkSumAdjustment makes a whole font file sum to


# ========================
# Font names
# ========================
def ps_name(name: str) -> str:
    """A name as FreeType (the fork's reader) reports PostScript names: ASCII 33-126 without []{}()<>/%."""
    return sub(r"[^!-~]|[\[\]{}()<>/%]", "", name)


def face_names(data: bytes) -> list:
    """
    The PostScript name of each face of a font file (.ttc/.otc hold several, in index order): name ID 6, else 4 (full
    name), else 1 (family), as FreeType would report it ("Kung Fu Panda" -> KungFuPanda).
    """
    fonts = TTCollection(BytesIO(data), lazy=True).fonts if data[:4] == b"ttcf" else [TTFont(BytesIO(data), lazy=True)]
    out = []
    for font in fonts:
        table = font["name"]
        out.append(ps_name(next((n for n in (table.getDebugName(i) for i in (6, 4, 1)) if n), "")))
    return out


def _font_files(directory: str) -> list:
    """The font files at the top level of directory, sorted."""
    _, _, files = next(walk(directory), (directory, [], []))
    return [join(directory, f) for f in sorted(files) if f.lower().endswith(FONT_EXTENSIONS)]


class FontIndex:
    """The faces a fonts folder supplies, by PostScript name; supplies() answers for one name."""

    def __init__(self, directory: str = ""):
        self.names: set = set()
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
        """Index one file's faces (face_names' output)."""
        self.files += 1
        self.faces += len(faces)
        self.names.update(ps for ps in faces if ps)

    def supplies(self, name: str) -> bool:
        """A face has exactly this PostScript name."""
        return name in self.names


# ========================
# Tables sharing data
# ========================
def _directory(data: bytes, at: int = 0) -> list:
    """(tag, checksum, offset, length) of each table record of the face whose offset table is at at."""
    return [unpack_from(">4sIII", data, at + 12 + 16 * i) for i in range(unpack_from(">H", data, at + 4)[0])]


def shared_tables(data: bytes) -> list:
    """
    The tag pairs of one face whose records point at the very same bytes (a .ttc's faces each alone: tables shared
    across faces are the format's), sorted; [] for a file that is not a font. Photoshop 2023 sets no type in such a font
    (EarthboundZero's bdat and bloc on EBDT's and EBLC's, TC ce778210): split_tables cures it. A record over part of
    another's bytes (Goudy-Bold's glyf, too long) is no such case: Photoshop sets it, and draws its split copy 3 pixels
    differently.
    """
    try:
        faces = unpack_from(f">{unpack_from('>I', data, 8)[0]}I", data, 12) if data[:4] == b"ttcf" else (0,)
        found = set()
        for at in faces:
            seen: dict = {}
            for tag, _, offset, length in _directory(data, at):
                if length and (offset, length) in seen:
                    found.add(tuple(sorted((seen[offset, length], tag.decode("latin-1")))))
                seen.setdefault((offset, length), tag.decode("latin-1"))
        return sorted(found)
    except StructError:
        return []


def _sum(data: bytes) -> int:
    """The sfnt checksum: big-endian uint32s, zero-padded."""
    data += b"\0" * (-len(data) % 4)
    return sum(unpack_from(f">{len(data) // 4}I", data)) & 0xFFFFFFFF


def split_tables(data: bytes) -> Optional[bytes]:
    """
    The font with every table its own copy of its bytes, in their order, each 4-byte aligned, the record checksums and
    head's checkSumAdjustment computed again; nothing else changes (EarthboundZero's repair, byte for byte). None for a
    collection, a table past the file's end or a tag twice.
    """
    if data[:4] == b"ttcf":
        return None
    records = _directory(data)
    tables = {tag: data[offset:offset + length] for tag, _, offset, length in records}
    if len(tables) < len(records) or any(offset + length > len(data) for _, _, offset, length in records):
        return None
    if len(tables.get(b"head", b"")) >= 12:
        tables[b"head"] = tables[b"head"][:8] + bytes(4) + tables[b"head"][12:]
    out, at = bytearray(data[:12 + 16 * len(records)]), {}
    for tag, _, offset, _ in sorted(records, key=lambda record: (record[2], record[0])):
        out += bytes(-len(out) % 4)
        at[tag] = len(out)
        out += tables[tag]
    out += bytes(-len(out) % 4)
    for i, (tag, _, _, length) in enumerate(records):
        pack_into(">4sIII", out, 12 + 16 * i, tag, _sum(tables[tag]), at[tag], length)
    if len(tables.get(b"head", b"")) >= 12:
        pack_into(">I", out, at[b"head"] + 8, (SFNT_SUM - _sum(bytes(out))) & 0xFFFFFFFF)
    return bytes(out)


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


# ========================
# Reports
# ========================
@dataclass
class Card:
    """One PSD asking for a font MDW/Fonts lacks."""
    path: str
    author: str
    missing: dict                               # font -> [layer names]
    new: Optional[list] = None                  # of those, the fonts its PSD before the push did not ask for (None: all)


@dataclass
class Report:
    """One check's findings: what was read, which cards ask for a font the index lacks, and the errors."""
    checked: int = 0                            # PSDs read
    cards: list = field(default_factory=list)   # Card, in path order
    fonts_added: list = field(default_factory=list)   # files MDW/Fonts gained in the push
    fonts_files: int = 0
    fonts_faces: int = 0
    errors: list = field(default_factory=list)  # "path: why" (a bad blob, an unreadable PSD or font)

    def by_font(self) -> dict:
        """{font: {path: [layers]}}, fonts sorted."""
        out: dict = {}
        for card in self.cards:
            for font, layers in card.missing.items():
                out.setdefault(font, {})[card.path] = layers
        return {font: out[font] for font in sorted(out)}

    def as_dict(self) -> dict:
        """The --json report."""
        return {"checked": self.checked, "fonts": {"files": self.fonts_files, "faces": self.fonts_faces},
                "fonts_added": self.fonts_added, "missing": self.by_font(),
                "cards": [{"path": c.path, "author": c.author, "missing": c.missing, **({"new": c.new} if c.new is not None else {})}
                          for c in self.cards], "errors": self.errors}


def markdown(report: Report) -> str:
    """The job summary and commit comment: per PSD its missing fonts and layers, the fonts the push added, errors."""
    out = ["## Font check", "",
           f"{report.checked} card PSD{'s' if report.checked != 1 else ''} read; MDW/Fonts: {report.fonts_files} files, {report.fonts_faces} faces."]
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


def discord_payload(report: Report, repository: str = "", after: str = "") -> dict:
    """This check's part of the standards post (standards_post.payload): a line per card with a missing font told (its
    new ones when compared), the card makers' role pinged once; {} when nothing to tell. Imported here: Card sync and
    Card checks check font_check out without it."""
    from commands.standards_post import payload
    return payload(repository, after, font_check=report.as_dict())


def details_report(details: dict, cards: dict) -> Report:
    """
    /fonts: each current card PSD's psd_details record, whose fonts the card data build marked
    supplied or not against MDW/Fonts (psd_details.mark_supplied, the check _record makes). Nothing is downloaded.

    Args:
        details: {blob sha: record} (the card data's psd_details.json)
        cards: {name: CardInfo} (the database; duplicates share their original's PSD)

    Returns:
        Report: a PSD without a record, or with fonts the build could not judge, is an error line
    """
    report = Report()
    paths = {}
    for card in cards.values():
        path = getattr(card, "path", "") or ""
        if not getattr(card, "duplicate_of", None) and is_card_psd(path):
            paths[path] = getattr(card, "blob_sha", None)
    for path in sorted(paths):
        record = details.get(paths[path] or "")
        fonts = record.get("fonts") if isinstance(record, dict) else None
        if not isinstance(fonts, dict):
            report.errors.append(f"{path}: no PSD details in the card data")
            continue
        report.checked += 1
        entries = {font: entry for font, entry in fonts.items() if isinstance(entry, dict)}
        if any(entry.get("supplied") is None for entry in entries.values()):
            report.errors.append(f"{path}: fonts not judged (the build had no MDW/Fonts)")
        missing = {font: [basename(layer) for layer in entry.get("layers", [])]
                   for font, entry in entries.items() if entry.get("supplied") is False}
        if missing:
            report.cards.append(Card(path, NOBODY, missing))
    return report


def fonts_report(details: dict, cards: dict) -> dict:
    """The card data's fonts report (the manifest's "fonts", the bot's /fonts): details_report's PSDs read, the missing
    fonts ({font: {path: [layers]}}) and the errors."""
    report = details_report(details, cards)
    return {"checked": report.checked, "missing": report.by_font(), "errors": report.errors}


# ========================
# Sources: files, a checkout, the GitHub API
# ========================
def is_card_psd(path: str) -> bool:
    """A PSD under the card folders (not MDW/, not Markers/), by repo path."""
    return path.lower().endswith(".psd") and not any(folder in EXCLUDE_FOLDERS for folder in path.split("/"))


def _record(report: Report, index: FontIndex, path: str, author: str, fonts: dict) -> None:
    """Count one read PSD and list it when a font it asks for is not in the index."""
    report.checked += 1
    missing = {font: layers for font, layers in fonts.items() if not index.supplies(font)}
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


def _clone_files(clone: str, base: str, after: str) -> list:
    """base..after's files from git diff-tree in a clone, as the APIs list them (trees only: no blob read)."""
    raw = run(["git", "-C", clone, "diff-tree", "-r", "-z", "--no-renames", base, after], capture_output=True, text=True,
              check=True, env={**environ, "GIT_NO_LAZY_FETCH": "1"}).stdout.split("\0")
    return [{"filename": p, "sha": m.split(" ")[3], "status": {"A": "added", "D": "removed"}.get(m.split(" ")[4], "modified")}
            for m, p in zip(raw[0::2], raw[1::2]) if m]


def pushed_files(repository: str, before: str, after: str, author: str = NOBODY, clone: str = "") -> tuple:
    """
    What a push changed.

    before..after through the compare API; the files of after alone when before is empty or all zeros (a dispatch, a new
    branch) or no longer exists (a force push). With several commits each is read for who touched which file last. The
    APIs page at FILES_PAGE files and COMMITS_PAGE commits: past them a note says files may be credited to the wrong author.
    With clone, a push GitHub can't list (the compare's 422 on ~40+ PSDs, a commit listing no files) is read from it.

    Args:
        repository: owner/name
        before: The push's previous head (github.event.before)
        after: The push's head
        author: Who to name when a commit's author has no login (github.actor)
        clone: A clone holding both heads (blobless is fine); "" for none

    Returns:
        ({card PSD path: (blob sha at after, author login)}, [fonts MDW/Fonts gained], [notes])
    """
    notes, commits, files = [], [], []
    ranged = bool(before) and set(before) != {"0"}
    base = before if ranged else f"{after}^"
    try:
        data = _get_json(repository, f"compare/{before}...{after}", missing_ok=True) if ranged else None
        if data is None:
            if ranged:
                notes.append(f"the previous head {before[:8]} is gone (a force push?); only the head commit's files were checked")
                base = f"{after}^"
            head = _get_json(repository, f"commits/{after}")
            commits, files = [head], head.get("files", [])
        else:
            commits, files = data.get("commits", []), data.get("files", [])
    except HTTPError:
        if not clone:
            raise
    if len(files) >= FILES_PAGE:
        notes.append(f"the push changed {FILES_PAGE} files or more; only the first {FILES_PAGE} were checked")
    if len(commits) >= COMMITS_PAGE:
        notes.append(f"the push has {COMMITS_PAGE} commits or more; later files may be credited to the wrong author")
    if clone and not files:
        files = _clone_files(clone, base, after)
        if files:
            commits = []
            notes.append(f"GitHub could not list this push's files; read from the clone, all credited to {author}")
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


def old_blobs(repository: str, before: str, psds: dict, clone: str = "", notes: Optional[list] = None) -> Optional[dict]:
    """
    Each pushed card PSD's blob before the push, for the checks' new notes: the path's at before, else its own blob
    anywhere there (an exact rename or copy: nothing new); "" for a new card. Before's tree from the clone (a blobless
    one holds it), else the API.

    Args:
        repository: owner/name
        before: The push's previous head; empty or all zeros (a dispatch, a new branch): None
        psds: {path: (blob sha at after, author)} (pushed_files)
        clone: A clone holding before (blobless is fine); "" for none
        notes: Gains a line when before's tree can't be read

    Returns:
        {path: blob or ""}, or None (every note told)
    """
    if not before or set(before) == {"0"} or not psds:
        return None
    tree = None
    if clone:
        listed = run(["git", "-C", clone, "ls-tree", "-r", "-z", before], capture_output=True, text=True,
                     env={**environ, "GIT_NO_LAZY_FETCH": "1"})
        if listed.returncode == 0:
            tree = {line.split("\t", 1)[1]: line.split(" ")[2].split("\t")[0]
                    for line in listed.stdout.split("\0") if line and line.split(" ")[1] == "blob"}
    try:
        tree = tree if tree is not None else tree_blobs(repository, before)
    except Exception as e:                      # a force push's gone head, the API down: every note told
        if notes is not None:
            notes.append(f"the PSDs before the push could not be read ({e}); every note is told")
        return None
    shas = set(tree.values())
    return {path: tree.get(path) or (sha if sha in shas else "") for path, (sha, _) in psds.items()}


def check_blobs(repository: str, psds: dict, index: FontIndex, clone: str = "", olds: Optional[dict] = None) -> Report:
    """
    PSDs read by blob sha through the API, in path order.

    Args:
        repository: owner/name
        psds: {path: (blob sha, author)}
        index: The fonts folder's faces
        clone: A clone read first and kept filled (github_api.fetch_verified)
        olds: {path: its blob before the push, "" for a new card} (old_blobs): each card's new fonts

    Returns:
        Report
    """
    report = Report(fonts_files=index.files, fonts_faces=index.faces, errors=list(index.errors))
    for path in sorted(psds):
        sha, author = psds[path]
        try:
            data = fetch_verified(repository, sha, timeout=API_TIMEOUT, clone=clone)
        except Exception as e:
            report.errors.append(f"{path}: {e}")
            continue
        try:
            fonts = psd_fonts(data)
        except Exception as e:
            report.errors.append(f"{path}: {e}")
            continue
        cards = len(report.cards)
        _record(report, index, path, author, fonts)
        if olds is not None and len(report.cards) > cards:
            card, old = report.cards[-1], olds.get(path, "")
            try:
                asked = set() if not old else set(card.missing) if old == sha else set(
                    psd_fonts(fetch_verified(repository, old, timeout=API_TIMEOUT, clone=clone)))
            except Exception:                   # the old blob unread: every font is told
                asked = set()
            card.new = [font for font in card.missing if font not in asked]
    return report


# ========================
# The fonts folder
# ========================
def tree_fonts(blobs: dict) -> dict:
    """{file name: blob sha} of the font files at the top of MDW/Fonts, from {repo path: blob sha}."""
    return {path[len(FONTS_DIR) + 1:]: sha for path, sha in blobs.items()
            if path.startswith(FONTS_DIR + "/") and path.count("/") == 2 and path.lower().endswith(FONT_EXTENSIONS)}


def dir_fonts(directory: str) -> dict:
    """{file name: blob sha} of the font files at the top of directory."""
    out = {}
    for path in _font_files(directory):
        with open(path, "rb") as fh:
            out[basename(path)] = blob_sha(fh.read())
    return out


def save_names(directory: str, path: str) -> None:
    """The fonts folder's FontIndex and dir_fonts() as JSON at path (load_names reads it)."""
    index = FontIndex(directory)
    data = {"files": dir_fonts(directory), "count": index.files, "faces": index.faces, "names": sorted(index.names),
            "errors": index.errors}
    with open(path, "w", encoding="utf-8") as fh:     # read all first: no half file
        dump(data, fh)


def load_names(path: str) -> tuple:
    """(FontIndex, {file name: blob sha}) from a save_names() file; raises when it is unreadable."""
    with open(path, encoding="utf-8") as fh:
        data = json_load(fh)
    index = FontIndex()
    index.names, index.files, index.faces, index.errors = set(data["names"]), data["count"], data["faces"], list(data["errors"])
    return index, dict(data["files"])


def local_fonts() -> str:
    """The fonts dir: PSD_TOOLS_FONT_DIRS' first, else the checkout's MDW/Fonts when it exists, else ""."""
    dirs = environ.get(FONT_DIRS_ENV, "")
    if dirs:
        return dirs.split(pathsep)[0]
    checkout = join(expanduser(LOCAL_DIR_LOC), FONTS_DIR)
    return checkout if isdir(checkout) else ""


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
    parser.add_argument("--clone", default="", metavar="DIR", help="with --repo: a clone whose blobs are read first; fetched ones are kept in it")
    parser.add_argument("--json", default="", metavar="OUT", help="write the report as JSON here")
    parser.add_argument("--discord", default="", metavar="OUT", help="write this check's part of the standards post here ({} when nothing to tell)")
    options = parser.parse_args(args)
    token_from_env()
    index = FontIndex(options.fonts)
    if options.repo:
        try:
            if options.whole_tree:
                psds = {p: (s, options.author) for p, s in tree_blobs(options.repo, options.after).items() if is_card_psd(p)}
                fonts_added, notes = [], []
            else:
                psds, fonts_added, notes = pushed_files(options.repo, options.before, options.after, options.author, options.clone)
            olds = None if options.whole_tree else old_blobs(options.repo, options.before, psds, options.clone, notes)
            report = check_blobs(options.repo, psds, index, options.clone, olds)
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
            dump(discord_payload(report, options.repo, options.after if options.repo else ""), fh)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# ========================
# Module Exports
# ========================
__all__ = [
    'API_TIMEOUT',
    'COMMITS_PAGE',
    'Card',
    'EXCLUDE_FOLDERS',
    'FILES_PAGE',
    'FONTS_DIR',
    'FONT_DIRS_ENV',
    'FontIndex',
    'NAMES_ENV',
    'POOL_CHUNK',
    'POOL_MIN',
    'Report',
    'check_blobs',
    'check_checkout',
    'check_files',
    'details_report',
    'dir_fonts',
    'discord_payload',
    'face_names',
    'fonts_report',
    'is_card_psd',
    'layer_fonts',
    'load_names',
    'local_fonts',
    'main',
    'old_blobs',
    'markdown',
    'ps_name',
    'psd_fonts',
    'pushed_files',
    'save_names',
    'tree_blobs',
    'tree_fonts',
]
