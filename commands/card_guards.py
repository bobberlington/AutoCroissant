"""
The push guards (docs/Campaigns/PUSH_GUARDS.md): TTSCardMaker's card-guards.yml runs this on every push to main. Guard 1
(png_export) gives a card PSD pushed without its PNG the PNG the bake would make (the fork's render of the changed layers,
merged into the old PNG; Photoshop's stored composite only cross-checks, and a PSD without one, every card since the
11-07 pass, gets the PNG under the bake's own gates, nothing told); guard 2 (card_standards) checks the pushed PSDs against the card
standards (missing fonts are the font check's alone) and fixes only the 0-pixel classes; the Single-line Composer (R8) it
always sets unsaid, box text that rewraps redrawn and its PNG re-exported: only the bot commit's message names it
(owner, 2026-10-25); so are the pixel fixes (card_standards.PIXEL_FIXES, in order: clean, on cards: hidden stars, type
icons, exact copies and leftovers deleted and data that draws nothing removed, every picture and the parse the same, no
PNG (owner 11-09; a hidden picture's linked file another PSD shows: hidden_of, db/links.json's holders read); badges laid out again at their new widths, straight apostrophes, old icon font glyphs, badges fitted, timing tags set as badges: R1 then untold, status
words in the chart's colours and effects and status cards in Krabby Patty: no changelog event either, owner 11-07,
11-09),
the PNG re-exported the same way; and the folders as the templates' (card_standards.standard_folders, owner 11-09:
only where nothing else changes, no PNG); badges drawn with another build of the icon font that the PSD keeps are told
(owner 2026-11-04: OLD_FONT; a re-lay a font explains lands instead, owner 2026-11-10: card_standards.font_cause).
One report per push: the job summary and commit comment (report.md), one Discord post for the standards channel
(discord.json: no mentions, no previews), and a plan of the writes (plan.json) that `land` commits once, right after
the check. Each card pushed over an older PSD carries what changed inside it (psd_diff: "added Ability FX: Blazing",
"Name text: 'a' -> 'b'").
Every card PNG the push brought is recompressed losslessly in that commit (owner, 2026-10-23; png_export.recompress,
the same pixels) when that saves more than 10% and 64 KiB (our thresholds), and so is every PNG the bot writes; a
pushed one in a process of its own, beside the rest of the check (_overlap).

    python -m commands.card_guards check --clone <dir> --after <sha> [--before <sha>] [--repo <owner/name>] [--author <login>]
                                         [--cards <names>] [--whole-tree] [--mode dry|report|on] [--convert now|later|off]
                                         --out <dir>
    python -m commands.card_guards normalise --out <dir> --clone <dir> [--repo <owner/name>] [--minutes <n>] [--deadline <epoch>]
    python -m commands.card_guards land --plan <out>/plan.json [--dry-run] [--local <clone>] [--rerun]

After the 11-07 pass (PLAN 7, all silent: no Discord, comment or changelog event): a pushed card, template or
MDW/Unfinished PSD with an embedded picture, sub or superscript text, a merged image or a link to repair is converted
by card_standards' normalise fix (G1; `normalise`, after the check's Decided, or in the check with --convert now), a
renamed PSD whose folder depth changed has its links repaired (G4); MDW/Linked files no PSD links are deleted (G2,
db/links.json on the data branch), a pushed one with another's bytes merged into it (G3), a link to a deleted one put
back (G4); one landing (git past GIT_LAND files or GIT_LAND_BYTES). A font pushed to MDW/Fonts whose tables share
data (Photoshop 2023 sets no type in it) lands with every table its own copy, the same bytes (font_check.split_tables;
silent). A pushed card PSD or PNG whose file name holds . : [ ] ? * " < > | / \\ takes the file form of its name
(card_models.file_stem) in that commit, its card's other files with it and the files keyed by the card following
(card_bake.follow_renames, Renamed trailers; unsafe_renames, owner 2026-11-11, silent).

A landed commit's changelog events (guard-png: a PNG re-made or recompressed; guard-fix: a fix the guards landed; the
Single-line Composer none) go to guard-events.json beside the plan, which card-guards.yml uploads and Card sync's data
job reads (card_data --guard-events): the Changes page shows them, never Discord (docs/Finished/session-2026-10-26/CARD_SYNC_VERDICTS.md).

check reads trees from a local clone (the Action's blobless checkout with its history; a full clone locally) and blobs
from it or, when it lacks them, by sha through the API (verified). It writes nothing to the card repo, ever: a dry run
and a replay are the same command. land re-reads the branch head, drops a card whose PSD or PNG moved, and makes one
commit (git_commit.commit_files; --local: in a clone, nothing pushed). The guards' own commits and the bake's, the
save's and Card sync's are skipped (the AutoCroissant author or github-actions; a path a person also changed in the range
is still guarded). No Discord or config imports (the token:
config through github_api, else GITHUB_TOKEN); SPARSE_FILES is what card-guards.yml checks out: the import closure (the card
and font checks' too, run in its check job) and requirements-cards.txt.
"""
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from functools import partial
from gzip import decompress as gzip_decompress
from hashlib import blake2b
from io import BytesIO
from json import dump, load, loads
from os import cpu_count, environ, getpid, listdir, makedirs, pathsep, replace
from os.path import basename, dirname, exists, isdir, join, splitext
from re import IGNORECASE, escape, fullmatch, sub
from struct import unpack
from subprocess import CalledProcessError, TimeoutExpired
from sys import stderr
from tempfile import TemporaryDirectory
from time import sleep, time
from traceback import format_exc
from types import SimpleNamespace
from typing import Callable, Iterator, Optional

import numpy as np
from PIL import Image
from psd_tools.api.links import RepoTree
from requests.exceptions import RequestException

import commands.card_standards as standards
import commands.status_font  # noqa: F401  standards' "status font" kind (imported where used): in SPARSE_FILES
import commands.psd_details  # noqa: F401  standards' folders gate (imported where used): in SPARSE_FILES
import commands.png_export as png_export
import commands.psd_diff as psd_diff
import commands.psd_edit as psd_edit
from commands.card_bake import (PREFETCH_WORKERS, RENAME_SIDES, Snapshot, fetch_verified, follow_renames, local_snapshot,
                                rename_trailers, snapshot_at)
from commands.card_check import vocabulary
from commands.card_edits import CardAuthors, dup_pictures
from commands.card_models import NOT_IN_FILE_NAMES, file_stem
from commands.font_check import face_names, is_card_psd, shared_tables, split_tables, tree_fonts
from commands.git_commit import (AUTHOR, COMMIT_TRIES, NO_LAZY, Transient, api, commit_files, git, git_commit, local_commit,
                                 plural, ref_head, token)
from commands.github_api import blob_sha, token_from_env
from commands.png_changes import changed_area
from commands.psd_edit import Refused
from commands.psd_parser import picture_numbers
from global_config import CARD_REPO, EDITS_PATH, FONT_DIRS_ENV, ICON_MANIFEST_PATH, STARS_DIR, TYPES_DIR

# ========================
# Configuration
# ========================
SPARSE_FILES = ("commands/card_guards.py", "commands/png_export.py", "commands/card_standards.py", "commands/txt2_repair.py",
                "commands/psd_diff.py", "commands/card_bake.py", "commands/card_check.py", "commands/psd_edit.py", "commands/psd_regions.py",
                "commands/psd_parser.py", "commands/psd_parts.py", "commands/text_align.py", "commands/card_models.py", "commands/card_edits.py", "commands/font_check.py",
                "commands/github_api.py", "commands/git_commit.py", "commands/repo_history.py", "global_config.py",
                "requirements-cards.txt",
                "commands/ability_place.py", "commands/ability_index.json",
                "commands/psd_links.py", "commands/status_font.py", "commands/psd_details.py", "commands/png_changes.py")
SUBJECT = "Card guards"                         # the bot commit: "Card guards: 2 PNGs re-exported, 1 PSD fixed (push 7cb4c867)"
TRAILER = "Guarded-push"                        # the push the commit guards
RECOMPRESSED = "Recompressed"                   # a trailer per PNG only recompressed: no image version (build_site)
DATA_STATS = "db/stats.json"                    # the card data on the data branch (card_data): R14's card names
BOT_EMAILS = ("41898282+github-actions[bot]@users.noreply.github.com",)   # besides git_commit.AUTHOR: never guarded
EMPTY_TREE = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"
ZEROS = "0" * 40
MODES = ("dry", "report", "on")
LINE = "{login} pushed {card}: {issues}. [{short}]({url})"           # what to do, as short as it goes (owner 2026-11-06)
CHANGES_COMMENT = 600                           # and in the commit comment
MARKDOWN = r"([\\`*_~|<\[\]])"                   # escaped in card text and layer names (GitHub and Discord markdown)
DISCORD_LIMIT = 2000
SUPPRESS_EMBEDS = 4                             # Discord message flag: no link previews
URL_SHA = 12                                    # hex digits of a commit link (GitHub resolves it; 28 characters a line saved)
NOBODY = "Someone"
NOREPLY = r"(?:\d+\+)?([A-Za-z0-9-]+)@users\.noreply\.github\.com"   # a GitHub noreply email names its login
ANCHOR_WINDOW = timedelta(hours=2)              # a burst: PSD commits this close to the push may stand in for its PNG's PSD
POOL_MIN = 8                                    # checks of this many cards or more run in a process pool (2 on a runner)
POOL_MAX = 4                                    # local pools at most this many (each runs git); runners use every core (K-2)
FONT_FILES_EACH = 4                             # R8: font files tried a face, likeliest name first (270 of 275 faces in 3, 10-25)
FONT_FETCH_MAX = 12                             # R8: font files a run (a worker) fetches at most, a request each
CARD_ROOTS = ("Auxiliary/", "Creatures/", "Field/", "Items/", "N.M.E/")   # where card PNGs live (Types/ icons are not cards)
ICON_FONT_FILE = f"{psd_edit.FONTS_DIR}/stoogedings.ttf"
EARLIER_WALK = 40                               # badge widths: versions of the icon font's history earlier_fonts reads at
                                                # most, every era of them kept (10 versions, 5 eras by 2026-11-10)
EARLIER_ENV = "CARD_EARLIER_FONTS"              # card-guards.yml's Earlier fonts folder: <blob sha> files (_font_blob)
FONTS_READY_ENV = "CARD_FONTS_READY"            # card-guards.yml's mark: MDW/Fonts restored (a push's Check starts beside it)
FONTS_READY_S = 600                             # fonts_ready's wait at most
ICON_LAYOUTS = ("badge widths", "old glyphs", "badges")   # skipped when the icon layers and font are as last
                                                # checked (_layouts_kept)
GUARD_EVENTS = "guard-events.json"              # beside the plan: the landed commit's changelog events
GUARD_RECORDS = "records.json"                  # beside the plan: what the landed PSD writes read before the strip, for
                                                # the data's records (card-guards.yml's guard-records; card_data --records)
REDRAW = "; re-exported for "                   # _rewrapped_pngs' line on a guard 1 PNG (the composer's rewrap): not told
QUIET_KEYS = ("EveryLineComposer",)            # fixed unsaid (owner, 2026-10-25): never in the changes either
OLDER_PNGS = 5                                  # R10 went back: the PNG's earlier versions read at most (_went_back)
OLDER_MATCH = 0.1                               # share of the area an earlier version may still differ over and show it
PUT_BACK_REACH = 32                             # px _put_back follows a went-back change past the rule's area
SKIP_REACH = 8                                  # px past the PSD's change a PNG that skipped it may change (_skipped)
TELL_PNG_MISMATCH = False                       # R10 and "current but outside the bar" on Discord too (beyond the owner's
                                                # two rules: the commit comment only until the owner asks)
OLD_FONT = ("old stoogedings font: pull, reinstall MDW/Fonts/stoogedings.ttf, restart Photoshop, "   # told (owner 2026-11-04)
            "Type > Update All Text Layers, save and push")
NORMALISE = "normalise"                         # the 11-07 pass's fix (card_standards): linked pictures, normal text, no
                                                # merged image; the guards run it on every later push, silent (G1)
QUIET_PIXELS = (NORMALISE, "status words", "status font")   # pixel fixes that make no guard-event (owner 11-07, 11-09)
FOLDERS = "folders as the templates'"           # card_standards.standard_folders: said in the commit message alone
CONVERT_ROOTS = ("MDW/Templates/", "MDW/Unfinished/")   # PSDs converted and nothing else (Q3)
CONVERTS = ("now", "later", "off")              # check's --convert: in the check, in the normalise step (after Decided), never
CONVERT_MINUTES = 20                            # no conversion starts past this; the rest stay as pushed (their next push)
LINKS = "db/links.json"                         # the data branch's links (card_data.gather_links): G2-G4's index
LINKED_DIR = "MDW/Linked/"
ART_ROOTS = ("MDW/Templates/", "MDW/Unfinished/", "MDW/Assets/", "MDW/Croissant's Special Folder/")   # with the cards:
                                                # the PSDs whose shown pictures make a card's hidden copy go (clean)
ART_READS = 3                                   # those PSDs read a card at most (clean's Hidden.elsewhere)
GIT_LAND = 40                                   # files past which land commits by git (git_commit: no blob upload each)
GIT_LAND_BYTES = 32 << 20                       # or bytes: one push beats base64 POSTs (6 PSDs, 115 MB: 23 s by REST)
_worker: dict = {}
_early: dict = {}                               # pushed PNG blob sha -> its recompression's Future (_overlap)
_waiting: list = []                             # _recompress's entries left to _collect
_glyphs: dict = {}                              # icon font blob sha -> font_glyphs (earlier_fonts)


# ========================
# Data
# ========================
@dataclass
class Change:
    """One path the push changed: git diff --raw."""
    status: str                                 # A, M or D
    old: str                                    # blob before ("" when added)
    new: str                                    # blob after ("" when deleted)
    path: str


@dataclass
class Writer:
    """The commit that last changed a path in the push."""
    commit: str
    name: str
    email: str
    subject: str
    login: str = ""

    @property
    def bot(self) -> bool:
        """By author only: a person's commit named like a bot's ("Card edits: ...") is still guarded."""
        return self.email in (AUTHOR["email"], *BOT_EMAILS) or self.name == AUTHOR["name"]


@dataclass
class Job:
    """One card's work in the push."""
    psd: str                                    # the PSD path at after (the deleted one's for "deleted")
    case: str                                   # stale, pair, new, beside, rename, psd-only, png-only, deleted, png-deleted, whole,
                                                # numbered (numbered PNGs pushed alone), convert (normalise alone: Q3, G4)
    psd_old: str = ""                           # its blob before the push
    psd_new: str = ""                           # its blob at after
    png: str = ""                               # the PNG path (to check, or to write)
    png_old: str = ""                           # the PNG blob an export rests on
    png_new: str = ""                           # the pushed PNG (R10)
    numbered: dict = field(default_factory=dict)   # numbered PNG path -> blob at after
    copy_from: str = ""                         # rename: the old PNG's blob, copied to png
    psd_anchor: str = ""                        # stale: the PSD as of its PNG's last change, when a burst's earlier push
    anchor_commit: str = ""                     # changed the PSD without the PNG (G5's second try); that commit
    left: str = ""                              # rename: the old PNG, still at its old path
    pushed: dict = field(default_factory=dict)  # numbered or drawn dup PNG -> blob the push brought (recompressed only)
    convert: bool = False                       # rename: its links repaired too (its folder depth changed, G4)
    dups: dict = field(default_factory=dict)    # picture-only duplicate path -> blob at after (card_edits.dup_pictures)
    dups_pushed: dict = field(default_factory=dict)     # of those, the ones the push brought -> that blob
    base_png: str = ""                          # pair: the PNG's blob before the push (its pictures follow from it)
    font_kept: bool = False                     # its last change before the push had this icon font (Source.font_kept)
    before: str = ""                            # the push's before (R10's earlier PNGs: Source.older_pngs)
    login: str = NOBODY
    commit: str = ""                            # the last writer's commit (the Discord link)


@dataclass
class Result:
    """One card's verdicts."""
    job: Job
    export: dict = field(default_factory=dict)  # guard 1: outcome, why, numbers...
    issues: list = field(default_factory=list)  # card_standards.Issue as dicts
    fixed: list = field(default_factory=list)   # the fix lines
    fix_refused: list = field(default_factory=list)   # (code, why)
    r10: dict = field(default_factory=dict)     # png_check of a pushed PNG
    notes: list = field(default_factory=list)   # (text, told on Discord)
    writes: list = field(default_factory=list)  # {"path", "file", "sha", "kind" (png, psd, recompress), "line"; a redraw: "plain"}
    changes: list = field(default_factory=list) # what changed inside a modified PSD (psd_diff phrases)
    composer: list = field(default_factory=list)   # R8: layers set Single-line (the commit message alone says it)
    composer_left: list = field(default_factory=list)   # R8: (layer, why) not redrawn or not set (plan.json alone)
    redrawn: dict = field(default_factory=dict)  # pixel fixes (card_standards.PIXEL_FIXES): kind -> layers (silent)
    pixel_left: list = field(default_factory=list)  # (kind, why) a pixel fix not made (plan.json alone)
    pixel_notes: list = field(default_factory=list)     # what a landing pixel fix did to look at (plan.json alone)
    convert: list = field(default_factory=list) # what the normalise fix finds (convert_plan converts it, silent)
    shown: dict = field(default_factory=dict)   # PNG path -> {"file"} or {"blob"}: the PNGs that show the PSD as checked
    shows: str = "pushed"                       # which PSD they show: "pushed", or "fixed" (guard 2 redrew and they follow)
    links: dict = field(default_factory=dict)   # the PSD as it lands: what housekeeping reads of it (_links)
    error: str = ""

    @property
    def name(self) -> str:
        return card_name(self.job.psd)

    def told(self, mode: str = "on") -> list:
        """The lines the author is told on Discord; in report mode the PNGs the bot would make and the fixable issues
        too (nothing is made or fixed there)."""
        out = []
        if self.export.get("outcome") == "refused":
            out.append(self.export["why"])
        out += [why for why, _ in self.export.get("numbered_refused", [])]
        if mode == "report":
            out += [would_make(self.job.case, self.job.png, w) for w in self.writes if w["kind"] == "png" and not w.get("silent")]
        out += [f"{line} (the bot could not fix it: {why})" for line, why in self.fix_refused]
        if self.r10.get("ok") is False and TELL_PNG_MISMATCH:
            out.append(self.r10["line"])
        out += [issue["line"] for issue in self.issues
                if issue["discord"] and (issue["told"] or (mode == "report" and issue["fix"] and not issue["waived"]))]
        out += [text for text, discord in self.notes if discord]
        if self.error:
            out.append(self.error_line())
        return out

    def error_line(self) -> str:
        """An error, told (the owner's "no PNG and a report instead"): the PNG part when the card needed one and got none."""
        needs = (self.job.case in ("stale", "beside", "new", "rename") and self.export.get("outcome") not in ("current", "refused")
                 and not any(w["kind"] == "png" for w in self.writes))
        kind = self.error.split(":", 1)[0]
        return f"the bot could not finish checking it ({kind})" + ("; PNG not made: export it in Photoshop" if needs else "")


# ========================
# Helper Functions
# ========================
def card_name(path: str) -> str:
    """The card's name from its PSD path (card_models.card_name)."""
    return splitext(basename(path))[0].replace("_", " ")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def would_make(case: str, png: str, write: dict) -> str:
    """Report mode's line for a PNG write the bot would make with CARD_GUARDS=on (nothing is committed there)."""
    if write["path"] != png:
        return f"{basename(write['path'])} not updated with the PSD: re-export it in Photoshop"
    if case == "new":
        return "no PNG pushed with the new PSD: export it in Photoshop"
    if case == "rename":
        return "PNG not renamed with the PSD: rename it too"
    if write.get("back"):
        return "the PNG shows an earlier picture of the card: export it again in Photoshop"
    if write.get("skipped"):
        return "the PNG does not show the PSD's change: export it again in Photoshop"
    return "PNG not exported with the PSD: export it in Photoshop"


def _key(path: str) -> tuple:
    return dirname(path), splitext(basename(path))[0].lower()


def _is_png(path: str) -> bool:
    return path.lower().endswith(".png")


def card_png(path: str) -> bool:
    """A PNG in the card folders (not Types/, site/, TTS/; MDW/ and Markers/ hold no card)."""
    return _is_png(path) and path.startswith(CARD_ROOTS) and is_card_psd(path[:len(path) - 4] + ".psd")


# ========================
# The card repo: a local clone, blobs by sha (local first, else the API)
# ========================
def font_files(name: str, files: dict) -> list:
    """The font files (of {file name: sha}) a face's PostScript name looks to be in, likeliest first (FONT_FILES_EACH)."""
    def norm(text: str) -> str:
        return sub(r"[^a-z0-9]", "", text.lower())
    full, family = norm(name), norm(name.split("-")[0])
    stems = {file: norm(splitext(file)[0]) for file in files}
    found = [file for file, stem in stems.items()
             if len(stem) >= 3 and (stem.startswith(family) or family.startswith(stem) or full.startswith(stem))]
    found.sort(key=lambda file: (stems[file].replace("regular", "") != full.replace("regular", ""),
                                 abs(len(stems[file]) - len(full)), file))
    return found[:FONT_FILES_EACH]


def fonts_ready() -> None:
    """Returns once FONTS_READY_ENV's mark exists (unset: at once); Transient past FONTS_READY_S."""
    mark, end = environ.get(FONTS_READY_ENV, ""), time() + FONTS_READY_S
    while mark and not exists(mark):
        if time() > end:
            raise Transient(f"the fonts were not in after {FONTS_READY_S} s")
        sleep(0.05)


def wait_fonts() -> None:
    """With FONTS_READY_ENV, every FontMap waits for the fonts too (psd_edit.FONTS_WAIT: card_standards' own calls)."""
    if environ.get(FONTS_READY_ENV):
        psd_edit.FONTS_WAIT = fonts_ready


class Source:
    """Trees and history from a local clone (never a lazy fetch), blobs from it or the API."""

    def __init__(self, clone: str, repository: str = "", fonts: str = ""):
        self.clone, self.repository = clone, repository
        self.logins: dict = {}
        self.checked: dict = {}                 # the checked tree (fonts_for)
        self.fonts = fonts                      # where fonts_for puts the font files it fetches ("" fetches none)
        self._faces: set = set()
        self._tried: set = set()

    def fonts_for(self, names: list) -> str:
        """MDW/Fonts for R8's box text in the faces names (PostScript): FONT_DIRS_ENV's, the clone's, else self.fonts
        with only the files those names look to be in (font_files; FONT_FETCH_MAX a run), from the checked tree; "" when
        a face can't be had (the layer stays Every-line). The clone's once in (fonts_ready)."""
        checkout = join(self.clone, psd_edit.FONTS_DIR)
        if environ.get(FONT_DIRS_ENV):
            return environ[FONT_DIRS_ENV].split(pathsep)[0]
        fonts_ready()
        if isdir(checkout) and listdir(checkout):
            return checkout
        if not (self.fonts and self.checked):
            return ""
        files = tree_fonts({path: sha for path, (_, sha) in self.checked.items()})
        for name in names:
            for file in font_files(name, files):
                if name in self._faces or len(self._tried) >= FONT_FETCH_MAX:
                    break
                if file in self._tried:
                    continue
                self._tried.add(file)
                try:
                    data = self.blob(files[file])
                    faces = face_names(data)
                    temp = join(self.fonts, f".{file}.{getpid()}.tmp")     # workers share the folder
                    with open(temp, "wb") as fh:
                        fh.write(data)
                    replace(temp, join(self.fonts, file))
                except Exception:               # that file left out
                    print(format_exc(), file=stderr)
                    continue
                self._faces.update(faces)
        return self.fonts if all(name in self._faces for name in names) else ""

    def git(self, *args: str) -> str:
        try:
            return git(self.clone, *args, env={**environ, **NO_LAZY}).decode("utf-8", "replace")
        except CalledProcessError as error:
            raise Transient(f"git {args[0]} in {self.clone} failed: {(error.stderr or b'').decode('utf-8', 'replace').strip()}") from error
        except (TimeoutExpired, OSError) as error:
            raise Transient(f"git in {self.clone} failed: {error}") from error

    def commit(self, ref: str) -> str:
        """The commit a ref names; "" when the clone lacks it."""
        try:
            return self.git("rev-parse", "-q", "--verify", f"{ref}^{{commit}}").strip()
        except Transient:
            return ""

    def parent(self, sha: str) -> str:
        return self.commit(f"{sha}^1")

    def is_ancestor(self, a: str, b: str) -> bool:
        try:
            self.git("merge-base", "--is-ancestor", a, b)
            return True
        except Transient:
            return False

    def diff(self, before: str, after: str) -> list:
        """[Change] between two commits (or the empty tree), renames off: trees only, no blob read."""
        raw = self.git("diff", "--raw", "--no-renames", "--no-abbrev", "-z", before, after)
        parts, out, i = raw.split("\0"), [], 0
        while i < len(parts) - 1:
            meta = parts[i]
            if not meta.startswith(":"):
                i += 1
                continue
            fields = meta[1:].split(" ")
            status = fields[4][0]
            out.append(Change(status, "" if set(fields[2]) == {"0"} else fields[2],
                              "" if set(fields[3]) == {"0"} else fields[3], parts[i + 1]))
            i += 2
        return out

    def tree(self, commit: str) -> dict:
        """{path: (mode, blob sha)} of a commit's tree ({} for the empty tree)."""
        if commit == EMPTY_TREE:
            return {}
        out = {}
        for line in self.git("ls-tree", "-r", "-z", commit).split("\0"):
            meta, _, path = line.partition("\t")
            if path and meta.split(" ")[1] == "blob":
                out[path] = (meta.split(" ")[0], meta.split(" ")[2])
        return out

    def writers(self, before: str, after: str) -> dict:
        """{path: Writer}: the last person's commit in before..after that changed it, else the last bot commit (a path is
        left out as the bot's only when no person changed it in the range)."""
        span = after if before == EMPTY_TREE else f"{before}..{after}"
        raw = self.git("log", "--no-renames", "--reverse", "--format=%x1e%H%x1f%an%x1f%ae%x1f%s", "--name-only", "-z", span)
        out, people = {}, {}
        for chunk in raw.split("\x1e"):
            if not chunk.strip():
                continue
            head, _, rest = chunk.partition("\0")
            sha, name, email, subject = (head.split("\x1f") + ["", "", ""])[:4]
            writer = Writer(sha, name, email, subject)
            for path in rest.lstrip("\n").split("\0"):
                if path.strip():
                    out[path.strip("\n")] = writer
                    if not writer.bot:
                        people[path.strip("\n")] = writer
        return {**out, **people}

    def anchor(self, png: str, psd: str, before: str, after: str, psd_old: str, window: timedelta = ANCHOR_WINDOW) -> tuple:
        """
        (blob, commit): the PSD as of the commit that last changed its PNG (up to before), when the PSD changed since only
        in commits within window of after (a burst: an earlier push's PNG was not made, its run superseded); else ("", "").
        """
        if before == EMPTY_TREE:
            return "", ""
        try:
            last = self.git("log", "-1", "--format=%H", before, "--", png).strip()
            if not last:
                return "", ""
            blob = self.git("rev-parse", f"{last}:{psd}").strip()
            since = [int(t) for t in self.git("log", "--format=%ct", f"{last}..{before}", "--", psd).split()]
            now = int(self.git("log", "-1", "--format=%ct", after).strip())
        except (Transient, ValueError):         # the PSD did not exist then, or no history: no anchor
            return "", ""
        if not blob or blob == psd_old or not since or any(now - t > window.total_seconds() for t in since):
            return "", ""
        return blob, last

    def recent(self, before: str, after: str, paths: list, window: timedelta = ANCHOR_WINDOW) -> Optional[set]:
        """The paths (lower case) a commit up to before changed within window of after, in one walk (every parent, a
        merge against each): anchor's rule, so the rest have no anchor. A path git reads as a pattern is always in; None
        when unread (anchor asks each)."""
        if before == EMPTY_TREE:
            return set()
        try:
            now = int(self.git("log", "-1", "--format=%ct", after).strip())
            raw = self.git("log", "--full-history", "-m", "--no-renames", "--name-only", "-z", "--format=%x1e%ct",
                           before, "--", *paths)
            out = {p.lower() for p in paths if p.startswith(":") or any(c in p for c in "*?[\\")}
            for chunk in raw.split("\x1e"):
                stamp, _, rest = chunk.partition("\0")
                if stamp.strip() and now - int(stamp) <= window.total_seconds():
                    out.update(p.strip("\n").lower() for p in rest.split("\0"))
        except (Transient, ValueError):
            return None
        return out

    def older_pngs(self, png: str, before: str, most: int = OLDER_PNGS) -> list:
        """The blobs png held before each of its last most changes up to before, newest first (trees only); [] unread."""
        if not before or before == EMPTY_TREE:
            return []
        try:
            raw = self.git("log", "--no-renames", "--raw", "--no-abbrev", "--format=", f"-{most}", before, "--",
                           f":(literal){png}")
        except Transient:
            return []
        fields = [line.split("\t", 1)[0].split() for line in raw.splitlines() if line.startswith(":")]
        return list(dict.fromkeys(f[2] for f in fields if len(f) >= 5 and set(f[2]) != {"0"}))

    def font_kept(self, before: str, after: str, olds: dict) -> set:
        """The paths of olds ({path: blob at before}) whose last change up to before left that blob in a tree with
        after's icon font: one walk from the font's last change up to after, one rev-parse; empty when unread."""
        if before == EMPTY_TREE or not olds:
            return set()
        try:
            last = self.git("log", "-1", "--format=%H", after, "--", ICON_FONT_FILE).strip()
            if not last:
                return set()
            raw = self.git("log", "--no-renames", "--format=%x1e%H", "--name-only", "-z", f"{last}..{before}", "--",
                           *(f":(literal){path}" for path in olds))
            changed: dict = {}                  # path -> its newest commit in the walk
            for chunk in raw.split("\x1e"):
                head, _, rest = chunk.partition("\0")
                for path in rest.lstrip("\n").split("\0"):
                    changed.setdefault(path.strip("\n"), head.strip())
            pairs = [(path, sha) for path, sha in changed.items() if path in olds]
            asked = [f"{after}:{ICON_FONT_FILE}"] + [f"{sha}:{x}" for path, sha in pairs for x in (ICON_FONT_FILE, path)]
            found = self.git("rev-parse", *asked).split()
        except Transient:
            return set()
        if len(found) != len(asked):
            return set()
        return {path for i, (path, _) in enumerate(pairs) if found[1 + 2 * i] == found[0] and found[2 + 2 * i] == olds[path]}

    def paths(self):
        """The checked tree's paths, blind to case (psd_tools' RepoTree), built once."""
        if getattr(self, "_paths", (None,))[0] is not self.checked:
            self._paths = (self.checked, RepoTree(self.checked))
        return self._paths[1]

    def blob(self, sha: str) -> bytes:
        """A blob by sha, verified; raises Refused("blob")."""
        return fetch_verified(self.repository, sha, "", self.clone)

    def login(self, writer: Writer, fallback: str) -> str:
        """The commit author's GitHub login: the commits API (once per commit, when a token and repo are given), else the
        login in a GitHub noreply email; with the API, an author without a GitHub account is named by the pusher's login
        (fallback, github.actor: who pushed the card); without it (a local run) or without a pusher (a dispatch: whoever
        ran it did not push the card), by the git author name."""
        if writer.commit not in self.logins:
            login, online = "", bool(self.repository and token())
            if online:
                try:
                    login = ((api(self.repository, "GET", f"commits/{writer.commit}").json().get("author") or {}).get("login") or "")
                except Exception:               # no login: the fallbacks stand
                    login = ""
            noreply = fullmatch(NOREPLY, writer.email or "")
            pusher = fallback if online and fallback != NOBODY else ""
            self.logins[writer.commit] = login or (noreply.group(1) if noreply else "") or pusher or writer.name or fallback
        return self.logins[writer.commit]


# ========================
# The push range and the jobs (section 4)
# ========================
def push_range(source: Source, before: str, after: str) -> tuple:
    """
    (before, after, notes): before resolved; an empty or all-zero before, one the clone lacks, or one that is not an
    ancestor of after (a force push) is replaced by after's first parent (the empty tree for a root commit).
    """
    notes = []
    head = source.commit(after)
    if not head:
        raise Transient(f"the clone has no commit {after[:8]}")
    base = source.commit(before) if before and before != ZEROS else ""
    if base and not source.is_ancestor(base, head):
        notes.append(f"`{before[:8]}` is not an ancestor of `{head[:8]}` (a force push?): checked against its first parent")
        base = ""
    elif before and before != ZEROS and not base:
        notes.append(f"the clone lacks `{before[:8]}`: checked against the first parent")
    elif not before or before == ZEROS:
        if before == ZEROS:
            notes.append("a new branch: checked against the first parent")
    if not base:
        base = source.parent(head) or EMPTY_TREE
    return base, head, notes


def plan_jobs(source: Source, before: str, after: str, author: str, cards: Optional[set] = None) -> tuple:
    """
    The push's jobs per section 4.2.

    Returns:
        ([Job], {path: Writer} left out as bot commits, [notes], after tree)
    """
    changes = source.diff(before, after)
    writers = source.writers(before, after)
    head_writer = Writer(after, "", "", "")
    tree = source.tree(after)
    old_tree = source.tree(before)
    snapshot = Snapshot(source.repository, after, {}, tree)
    old_snapshot = Snapshot(source.repository, before, {}, old_tree)
    skipped = {c.path: writers[c.path] for c in changes if c.path in writers and writers[c.path].bot}
    changes = [c for c in changes if c.path not in skipped]
    psds = [c for c in changes if is_card_psd(c.path)]
    pngs = {c.path: c for c in changes if card_png(c.path)}
    converts = [c for c in changes if c.status in "AM" and convertible(c.path)]
    deleted: dict = {}                          # blob -> deleted paths: an added path with one of them is a rename
    for c in psds:
        if c.status == "D":
            deleted.setdefault(c.old, []).append(c.path)
    renamed_new = {c.path: deleted[c.new].pop(0) for c in psds if c.status == "A" and deleted.get(c.new)}
    renamed_old = set(renamed_new.values())
    png_deleted = {_key(p) for p, c in pngs.items() if c.status == "D"}
    notes, jobs = [], []

    def who(path: str) -> tuple:
        writer = writers.get(path, head_writer)
        return source.login(writer, author) if writer.commit != after or writer.name else author, writer.commit or after

    def numbered(path: str) -> dict:
        return {p: tree[p][1] for p in snapshot.png_for(path)[1] if p not in pngs and p in tree}

    try:                                        # picture-only duplicates follow their PSD's PNG (owner 11-07, silent)
        edits = loads(source.blob(tree[EDITS_PATH][1])).get("cards") if EDITS_PATH in tree else None
    except Exception:                           # unread: they stay as they are this push
        print(format_exc(), file=stderr)
        edits = None
    every_dup = dup_pictures(list(tree), edits)
    dups = {psd: {p: tree[p][1] for p in found if p not in pngs} for psd, found in every_dup.items()}
    dups_pushed = {psd: {p: pngs[p].new for p in found if p in pngs and pngs[p].status in "AM"} for psd, found in every_dup.items()}
    dup_pngs = {p for found in every_dup.values() for p in found}

    def pushed(path: str) -> dict:
        return {p: pngs[p].new for p in snapshot.png_for(path)[1] if p in pngs and pngs[p].status in "AM"}

    stale: list = []                            # (job, PSD blob before): anchored after the loop
    for change in psds:
        if change.path in renamed_old:
            continue
        png = snapshot.png_for(change.path)[0] if change.status != "D" else ""
        login, commit = who(change.path)
        job = Job(change.path, "", change.old, change.new, login=login, commit=commit, pushed=pushed(change.path))
        if change.status == "D":
            left = snapshot.png_for(change.path)[0]
            if left:
                jobs.append(Job(change.path, "deleted", change.old, "", left, tree[left][1], login=login, commit=commit))
            continue
        if change.path in renamed_new:
            old_path = renamed_new[change.path]
            moved = old_path.count("/") != change.path.count("/")   # its relPaths name other files now (G4)
            old_png = old_snapshot.png_for(old_path)[0] if not png else ""
            if old_png:
                job.case, job.png = "rename", splitext(change.path)[0] + splitext(old_png)[1]
                job.copy_from, job.psd_old = old_tree[old_png][1], change.new
                job.left = old_png if old_png in tree else ""
                job.convert = moved
                jobs.append(job)
            elif moved:
                jobs.append(Job(change.path, "convert", "", change.new, login=login, commit=commit))
            continue
        job.dups, job.dups_pushed = dups.get(change.path, {}), dups_pushed.get(change.path, {})
        if png and png in pngs and pngs[png].status in "AM":
            job.case, job.png, job.png_new, job.numbered = "pair", png, pngs[png].new, numbered(change.path)
            job.base_png = pngs[png].old if pngs[png].status == "M" else ""
        elif png and change.status == "M":
            job.case, job.png, job.png_old, job.numbered = "stale", png, tree[png][1], numbered(change.path)
            stale.append((job, change.old))
        elif png:
            job.case, job.png, job.png_old, job.numbered = "beside", png, tree[png][1], numbered(change.path)
        elif _key(change.path) in png_deleted:      # the author deleted the PNG: the standards only
            job.case = "psd-only"
            notes.append(f"`{change.path}` was pushed with its PNG deleted: no PNG is made")
        else:
            job.case, job.png = "new", splitext(change.path)[0] + ".png"
        jobs.append(job)
    recent = source.recent(before, after, [job.psd for job, _ in stale]) if stale else set()
    for job, old in stale:                      # anchor's git calls only where a commit is in its window
        if recent is None or job.psd.lower() in recent:
            job.psd_anchor, job.anchor_commit = source.anchor(job.png, job.psd, before, after, old)
    pushed_psds = {c.path for c in psds}
    psd_by_key: dict = {}                       # stems matched in any case (L12), so one key can hold several PSDs
    for p in tree:
        if is_card_psd(p):
            psd_by_key.setdefault(_key(p), []).append(p)
    for path, change in sorted(pngs.items()):
        found = psd_by_key.get(_key(path), [])
        exact = [p for p in found if splitext(p)[0] == splitext(path)[0]]
        if len(exact or found) > 1:
            notes.append(f"`{path}` matches {len(found)} PSDs that differ only in letter case: not checked")
            continue
        psd = (exact or found or [None])[0]
        if psd is None or psd in pushed_psds:
            if psd is None and change.status in "AM" and not snapshot.duplicate_of(CardAuthors.key(card_name(path))) \
                    and path not in dup_pngs:   # a picture-only duplicate: its PSD is its dup_of's
                if not any(p for p in pushed_psds if _key(p) == _key(path)):
                    notes.append(f"`{path}` has no PSD beside it")
            continue
        login, commit = who(path)
        if change.status in "AM":
            jobs.append(Job(psd, "png-only", "", tree[psd][1], path, change.old, change.new, login=login, commit=commit,
                            pushed=pushed(psd)))
        elif change.status == "D":
            jobs.append(Job(psd, "png-deleted", "", tree[psd][1], path, change.old, login=login, commit=commit,
                            pushed=pushed(psd)))  # its numbered PNGs too: one entry per PSD
    covered = {path for job in jobs for path in job.pushed}
    alone: dict = {}                            # numbered PNGs pushed without their PSD or its PNG: recompressed only
    for path, change in sorted(pngs.items()):
        psd = snapshot.duplicate_of(CardAuthors.key(card_name(path))) if change.status in "AM" and path not in covered else ""
        if psd in tree and path in snapshot.png_for(psd)[1]:
            alone.setdefault(psd, {})[path] = change.new
    for psd, found in alone.items():
        login, commit = who(sorted(found)[0])
        jobs.append(Job(psd, "numbered", "", tree[psd][1], login=login, commit=commit, pushed=found))
    for change in converts:                     # templates and MDW/Unfinished: converted, nothing else (Q3)
        login, commit = who(change.path)
        jobs.append(Job(change.path, "convert", change.old, change.new, login=login, commit=commit))
    if cards:
        jobs = [job for job in jobs if CardAuthors.key(card_name(job.psd)) in cards]
    olds = {job.psd: job.psd_old for job in jobs if job.case in ("stale", "pair", "psd-only") and job.psd_old}
    kept = source.font_kept(before, after, olds) if olds else set()
    for job in jobs:
        job.font_kept = job.psd in kept and job.psd_old == olds.get(job.psd)
        job.before = before
    return sorted(jobs, key=lambda job: job.psd), skipped, notes, tree


def whole_tree_jobs(tree: dict, author: str, cards: Optional[set] = None) -> list:
    """Every card PSD of the tree: guard 2 and R10 against its PNG."""
    snapshot = Snapshot("", "", {}, tree)
    jobs = []
    for path in sorted(tree):
        if is_card_psd(path) and (not cards or CardAuthors.key(card_name(path)) in cards):
            png, numbered = snapshot.png_for(path)
            jobs.append(Job(path, "whole", "", tree[path][1], png, "", tree[png][1] if png else "", login=author,
                            numbered={p: tree[p][1] for p in numbered if p in tree}))
    return jobs


# ========================
# Context: Types/, the status chart
# ========================
def build_context(source: Source, tree: dict, errors: Optional[list] = None) -> tuple:
    """
    (card_standards.Context, Vocabulary) from the tree at after.

    The manifest and the status chart come by blob; a failed one is an error line and the rules run without it.
    """
    errors = errors if errors is not None else []
    manifest = None
    if ICON_MANIFEST_PATH in tree:
        try:
            manifest = source.blob(tree[ICON_MANIFEST_PATH][1])
        except Exception as error:              # the parser runs without the manifest
            errors.append(f"{ICON_MANIFEST_PATH}: {error}")
    vocab = vocabulary(list(tree), manifest)
    chart = None
    if psd_edit.STATUS_CHART in tree:
        try:
            chart = psd_edit.status_fills(png_export.open_psd(source.blob(tree[psd_edit.STATUS_CHART][1])))
        except Exception as error:              # R5 reports every overlay without the chart
            errors.append(f"{psd_edit.STATUS_CHART}: {error}")
    context = standards.Context(frozenset(vocab.all_types), chart,
                                frozenset(path.lower() for path in tree if path.lower().endswith(".png")),   # R12's pictures
                                shown_cards(source, errors), art_holders(links_index(source), tree))
    return context, vocab


def art_holders(index: Optional[dict], tree: dict) -> Optional[dict]:
    """{linked file, lowercased: ((PSD path, blob), ...)} of the PSDs in scope (cards, ART_ROOTS) whose db/links.json
    record is of their blob at after: where a pushed card's hidden copy may show (Context.art); None without the index."""
    if not index:
        return None
    out: dict = {}
    for path, record in index.items():
        if (not isinstance(record, dict) or not (is_card_psd(path) or path.startswith(ART_ROOTS))
                or path not in tree or tree[path][1] != record.get("blob")):
            continue
        for link in record.get("links") or ():
            if link.get("path"):
                out.setdefault(link["path"].lower(), set()).add((path, record["blob"]))
    return {key: tuple(sorted(found)) for key, found in out.items()}


def hidden_of(source: Source, job: Job, psd, context) -> Optional[standards.Hidden]:
    """clean's leftovers on a pushed card (card_standards.Hidden): the linked files of its hidden pictures another PSD
    shows (ART_READS at most of those Context.art says link them, read at after), none at all with picture-only
    duplicates; None off a card."""
    if not is_card_psd(job.psd):
        return None
    if job.dups:
        return standards.Hidden(dups=True)
    wanted = standards.hidden_links(psd, job.psd) if context.art else set()
    found, read = set(), {}
    for path in sorted(wanted):
        for other, sha in context.art.get(path, ()):
            if path in found or other == job.psd:
                continue
            if other not in read:
                if len(read) >= ART_READS:
                    break
                try:
                    read[other] = standards.shown_links(png_export.open_psd(source.blob(sha), other), other)
                except Exception:               # unread: not shown there
                    print(format_exc(), file=stderr)
                    read[other] = set()
            found |= read[other] & wanted
    return standards.Hidden(frozenset(found))


def shown_cards(source: Source, errors: list) -> Optional[tuple]:
    """((title, key, kind), ...) of the cards the website shows, from the card data (db/stats.json on the data branch):
    R14's names. None without them (R14 not checked; an unreadable file is an error line)."""
    data = source.commit("origin/data") or source.commit("data")
    found = source.tree(data).get(DATA_STATS) if data else None
    if not found:                               # a clone with no card data: nothing to check names against
        return None
    try:
        cards = loads(source.blob(found[1]))["cards"]
    except Exception as error:                  # R14 is left out
        errors.append(f"card names (R14): {error}")
        return None
    return tuple(sorted((str(c.get("display_name") or c.get("name") or key), key, str(c.get("card_type") or ""))
                        for key, c in cards.items() if isinstance(c, dict) and not c.get("hidden")))


# ========================
# One card
# ========================
def _save(out: str, rel: str, data: bytes) -> str:
    """An artifact under out/files/; returns its path relative to out."""
    path = join("files", rel)
    makedirs(dirname(join(out, path)), exist_ok=True)
    with open(join(out, path), "wb") as fh:
        fh.write(data)
    return path


def _crops(out: str, rel: str, before, after, box) -> None:
    if before is None or after is None or not box:
        return
    stem = join(out, "crops", splitext(rel)[0])
    makedirs(dirname(stem), exist_ok=True)
    for name, image in psd_edit.artifacts(before, after, tuple(box)).items():
        image.save(f"{stem}.{name}.png")


def _size(n: int) -> str:
    return f"{n / 1e6:.1f} MB" if n >= 1e6 else f"{n / 1e3:.0f} KB"


def _recompress(out: str, result: Result, path: str, read: Callable[[], bytes], sha: str = "") -> None:
    """A PNG the push brought (read() gives it), recompressed losslessly (owner, 2026-10-23) when more than RECOMPRESS_MIN
    and RECOMPRESS_BYTES smaller (our thresholds): a write of its own kind, resting on that PNG alone and never told on
    Discord. A failure is a comment note, never the card's error (which Discord would tell). One _overlap started (its
    blob sha) is left to _collect, which puts it where it goes here."""
    if sha in _early:
        _waiting.append((result, len(result.writes), len(result.notes), out, path, read, _early[sha]))
        return
    write, note = _made(out, path, read)
    result.writes += [write] if write else []
    result.notes += [note] if note else []


def _made(out: str, path: str, read: Callable[[], bytes], early=None) -> tuple:
    """_recompress's (write, note), either None: early's (size, file) when its process gave them, else made here."""
    done = None
    if early is not None:
        try:
            done = early.result()
        except Exception:                       # a failed or broken process: inline, as before
            print(format_exc(), file=stderr)
    try:
        if done is None:
            data = read()
            done = len(data), png_export.recompress(data, png_export.RECOMPRESS_MIN, png_export.RECOMPRESS_BYTES)
    except Exception as error:                  # it only saves bytes: the card goes on
        print(format_exc(), file=stderr)
        return None, (f"{basename(path)} not recompressed ({type(error).__name__}: {error})", False)
    size, smaller = done
    if smaller is None:
        return None, None
    return {"path": path, "file": _save(out, path, smaller), "sha": blob_sha(smaller), "kind": "recompress",
            "line": f"recompressed losslessly ({_size(size)} -> {_size(len(smaller))}, the same pixels)"}, None


def _collect() -> None:
    """The recompressions _recompress left waiting, each at the place it would have taken (last first: the order holds)."""
    waiting, _waiting[:] = list(_waiting), []
    for result, at, noted, out, path, read, early in reversed(waiting):
        write, note = _made(out, path, read, early)
        if write:
            result.writes.insert(at, write)
        if note:
            result.notes.insert(noted, note)


def _recompressed(clone: str, repository: str, sha: str) -> tuple:
    """_overlap's process: a pushed PNG's size and png_export.recompress's file (None: not worth it)."""
    data = Source(clone, repository).blob(sha)
    return len(data), png_export.recompress(data, png_export.RECOMPRESS_MIN, png_export.RECOMPRESS_BYTES)


def _pngs(job: Job) -> list:
    """The blobs _run recompresses, in its order: the numbered PNGs the push brought, then the pushed PNG (R10)."""
    return [job.pushed[path] for path in sorted(job.pushed)] + (
        [job.png_new] if job.case in ("pair", "png-only") and job.png_new else [])


@contextmanager
def _overlap(source: Source, jobs) -> Iterator[None]:
    """The jobs' pushed PNGs recompressed in one process of their own (_pngs), started now: oxipng runs beside the rest
    of the check. The same writes in the same order; a start that fails
    leaves them inline."""
    shas, pool = list(dict.fromkeys(sha for job in jobs for sha in _pngs(job))), None
    try:
        pool = ProcessPoolExecutor(1, initializer=token_from_env) if shas else None
        for sha in shas:
            _early[sha] = pool.submit(_recompressed, source.clone, source.repository, sha)
    except Exception:                           # inline, as before
        print(format_exc(), file=stderr)
    try:
        yield
    finally:
        _early.clear()
        if pool is not None:
            pool.shutdown(cancel_futures=True)


def _export_dict(found) -> dict:
    """An Export as plan.json keeps it ("unchecked": no stored image to cross-check, G2; never told)."""
    return {"outcome": found.outcome, "why": found.why, "gate": found.gate, "changed_px": found.changed_px,
            "box": list(found.box) if found.box else None, "icc": found.icc, "writer": found.writer,
            "vote": list(found.vote) if found.vote else None, "numbers": found.numbers,
            "misses": png_export.miss_notes(found),
            "unchecked": found.unchecked}


def _r10_line(job: Job, measured: dict) -> str:
    bad = measured.get("bad_pct", 0)
    if job.case == "png-only":
        return f"{basename(job.png)} does not match its PSD ({bad:.1f}% of pixels differ): push the PSD too?"
    return f"{basename(job.png)} does not match the PSD ({bad:.1f}% of pixels differ): export it again"


def added(job: Job) -> bool:
    """A new card PSD (the new-card rules, R4 in full): no PNG yet, beside a PNG, or added with its PNG or with it deleted."""
    return job.case in ("new", "beside") or (job.case in ("pair", "psd-only") and not job.psd_old)


def _icon_prints(psd) -> dict:
    """{layer id: record_print with TySh} of the icon layers; None when two share an id."""
    layers = standards.icon_layers(psd)
    out = {layer.layer_id: png_export.record_print(layer._record, layer._channels, (b"TySh",)) for layer in layers}
    return out if len(out) == len(layers) else None


def _layouts_kept(job: Job, old, new) -> bool:
    """ICON_LAYOUTS left out: the push kept every icon layer (one of old's, byte for byte but its name) and the
    icon font is the one old was last changed with (the guards laid them out then)."""
    if not job.font_kept or old is None:
        return False
    now, before = _icon_prints(new), _icon_prints(old)
    return now is not None and before is not None and all(before.get(key) == value for key, value in now.items())


def _links(source: Source, psd, rel: str, sha: str) -> dict:
    """What housekeeping reads of a PSD (opened, or its bytes; blob sha), in the pool: its links (psd_links.links_of)
    and its links to an MDW/Linked file missing at after (_missing: G4's restore); {} unread."""
    from commands import psd_links
    try:
        psd, paths = png_export.open_psd(psd) if isinstance(psd, bytes) else psd, source.paths()
        links = [{"path": link.path, "name": link.name} for link in psd_links.links_of(psd, rel, paths)]
        return {"sha": sha, "links": links, "restore": _missing(psd, rel, paths)}
    except Exception:                           # never stops the card
        print(format_exc(), file=stderr)
        return {}


def run_job(source: Source, job: Job, context, parser, out: str, mode: str, earlier: tuple = ()) -> Result:
    """
    Both guards on one card; never raises (an error is the result's error line).

    Args:
        source: The card repo
        job: The card's job
        context: card_standards.Context
        parser: The offline parser (the fixes' reparse gate)
        out: The artifacts dir
        mode: dry, report or on (wording only: the writes are planned either way, land commits them)
        earlier: The icon font's earlier versions (earlier_fonts: badge widths)
    """
    result = Result(job)
    try:
        try:
            _run(source, job, context, parser, out, result, earlier)
        finally:
            _collect()                          # _overlap's recompressions, in their places
    except Exception as error:                  # one bad card never stops the others
        result.error = f"{type(error).__name__}: {error}"
        result.notes.append((f"error: {result.error}", False))
        print(format_exc(), file=stderr)
    return result


def _run(source: Source, job: Job, context, parser, out: str, result: Result, earlier: tuple = ()) -> None:
    for path, sha in sorted(job.pushed.items()):
        _recompress(out, result, path, partial(source.blob, sha), sha)
    if job.case == "numbered":
        return
    if job.case == "deleted":
        result.notes.append((f"{basename(job.png)} is left without its PSD", False))
        return
    if job.case == "png-deleted":
        result.notes.append((f"{basename(job.png)} was deleted; the PSD stays", False))
        return
    if job.case == "rename":
        data = source.blob(job.copy_from)
        smaller = png_export.recompress(data, png_export.RECOMPRESS_MIN, png_export.RECOMPRESS_BYTES)
        line = "the PNG copied to the PSD's new name" + (
            f"; recompressed losslessly ({_size(len(data))} -> {_size(len(smaller))})" if smaller else "")
        data = smaller or data
        file = _save(out, job.png, data)
        result.export = {"outcome": "re-exported", "why": "", "copied": True}
        result.writes.append({"path": job.png, "file": file, "sha": blob_sha(data), "kind": "png", "line": line})
        if job.left:
            result.notes.append((f"{basename(job.left)} is left at the PSD's old name", False))
        if job.convert:
            moved = png_export.open_psd(source.blob(job.psd_new))
            moved.repo_path = job.psd           # its links judged from its new folder
            result.convert = list(standards.pixel_kinds(moved, (NORMALISE,)))
        return
    new_bytes = source.blob(job.psd_new)
    new = png_export.open_psd(new_bytes, job.psd)
    new.repo_path = job.psd                     # the link step places and judges its links from here
    drawn = picture_numbers(new, [*job.numbered, *job.pushed, *job.dups, *job.dups_pushed]) \
        if job.dups or job.dups_pushed else {}
    for path in drawn:                          # a picture-only duplicate the PSD draws: a picture of it (owner 11-09)
        if path in job.dups:
            job.numbered[path] = job.dups.pop(path)
        elif path in job.dups_pushed:           # the push's: kept as a pushed NameN.png is
            job.pushed[path] = job.dups_pushed.pop(path)
            _recompress(out, result, path, partial(source.blob, job.pushed[path]), job.pushed[path])
    if job.case not in ("png-only", "whole"):   # a PSD write below reads its own
        result.links = _links(source, new, job.psd, job.psd_new)
    if job.case == "convert":
        result.convert = list(standards.pixel_kinds(new, (NORMALISE,)))
        return
    old = png_export.open_psd(source.blob(job.psd_old), job.psd) if job.psd_old else None
    faces = partial(source.fonts_for)          # a restyle keeping every glyph is laid out in them (psd_edit.same_glyphs)
    if old is not None:
        result.changes = changes(old, new, parser, faces)
    shown: dict = {}                            # the PNGs (path: image) that show the pushed PSD: R8's redraw follows
    # guard 1
    if job.case in ("stale", "beside"):
        old_png = png_export.open_png(source.blob(job.png_old))
        numbered = {path: png_export.open_png(source.blob(sha)) for path, sha in job.numbered.items()}
        found = png_export.export(old if job.case == "stale" else None, new, old_png, numbered, job.png, faces,
                                  list(job.pushed))
        # a burst: when the old PNG shows the anchor rather than the old PSD (not a PNG pushed before its PSD), the
        # anchored verdict, a refusal too (a small push-1 change hides under G5's bar, review 2026-10-24); else, checked,
        # a G5/G6 refusal retried, the anchored export kept unless it refuses
        if job.psd_anchor and old is not None:
            anchor = png_export.open_psd(source.blob(job.psd_anchor), job.psd)
            matched = png_export.shows(old_png, anchor, old)
            if matched or (not found.unchecked and found.outcome == "refused" and found.gate in ("G5", "G6")):
                anchored = png_export.export(anchor, new, old_png, numbered, job.png, faces, list(job.pushed))
                if matched or anchored.outcome != "refused":   # the PNG matched the PSD before the burst
                    found = anchored
                    result.notes.append((f"checked against the PSD as of {basename(job.png)}'s last change "
                                         f"(`{job.anchor_commit[:8]}`): an earlier push changed the PSD without its PNG",
                                         False))
        result.export = _export_dict(found)
        stale = found.numbers.get("png_vs_psd", {})
        if found.outcome == "current" and stale.get("px") and not psd_edit._within_bar(stale):
            off = (f"{stale['bad_pct']:.1f}% of pixels differ" if stale["bad_pct"] > psd_edit.PARITY_BAD_MAX
                   else f"mean difference {stale['mean']:.1f}")
            why = "the PSD is new beside it" if job.case == "beside" else "though this push moved no pixel"
            result.notes.append((f"{basename(job.png)} does not match the PSD ({off}), {why}: export it in Photoshop",
                                 TELL_PNG_MISMATCH))
        elif found.outcome == "current":
            shown.update(numbered)
            shown[job.png] = old_png
        if found.outcome == "re-exported":
            file = _save(out, job.png, found.data)
            result.writes.append({"path": job.png, "file": file, "sha": blob_sha(found.data), "kind": "png",
                                  "line": f"re-exported by the bot's render of {_drawn_as(found)} ({found.changed_px:,} px changed)"
                                          + "".join(f"; {note}" for note in png_export.miss_notes(found))})
            _crops(out, job.png, old_png, found.image, found.box)
            shown.update(numbered)
            shown[job.png] = found.image
        if found.outcome == "re-exported" or found.numbered:    # a tagged PSD's pictures: whatever its first one did
            refused = []
            for path, (data, why) in sorted(found.numbered.items()):
                if data:
                    tagged = path not in numbered or path in drawn or png_export.picture_number(path) in found.pictures
                    result.writes.append({"path": path, "file": _save(out, path, data), "sha": blob_sha(data), "kind": "png",
                                          "line": "made by the bot's render of its picture (variant tags)" if tagged
                                          else "updated with the same change (delta transfer)"})
                    shown[path] = png_export.open_png(data)
                elif data is None:              # its picture unchanged
                    if path in numbered:
                        shown[path] = numbered[path]
                else:
                    refused.append((why, path))
                    shown.pop(path, None)
            result.export["numbered_refused"] = refused
    elif job.case == "new":
        found = png_export.export_new(new)
        result.export = _export_dict(found)
        if found.outcome == "re-exported":
            shown[job.png] = found.image
            file = _save(out, job.png, found.data)
            result.writes.append({"path": job.png, "file": file, "sha": blob_sha(found.data), "kind": "png",
                                  "line": "made by the bot's render of the PSD (new card)"
                                          + "".join(f"; {note}" for note in png_export.miss_notes(found))})
    # R10: a pushed PNG against the PSD
    if job.case in ("pair", "png-only", "whole") and job.png_new:
        data = source.blob(job.png_new)
        if job.case != "whole":
            _recompress(out, result, job.png, lambda: data, job.png_new)
        png, kept = png_export.open_png(data), {}
        measured = png_export.png_check(new, png, kept)
        result.r10 = {**measured, "line": _r10_line(job, measured) if measured.get("ok") is False else ""}
        if measured.get("ok"):
            shown[job.png] = png
        if measured.get("ok") is None:
            result.notes.append((f"{basename(job.png)} could not be checked against the PSD ({measured.get('why')})", False))
        if "render" in kept and job.case != "whole":
            checked = dict(result.r10)
            try:
                _went_back(source, job, result, png, kept["render"], out, shown, old, new, faces)
            except Exception:                   # best effort: R10 stands as checked
                print(format_exc(), file=stderr)
                result.r10 = checked
        if job.case == "pair" and old is not None and job.base_png and "back" not in result.r10:
            _skipped(source, job, result, png, old, new, out, shown, faces)
    if job.case == "pair" and old is not None and job.base_png:
        _pair_pictures(source, job, result, old, new, out, shown, faces)
    skip = (NORMALISE, *ICON_LAYOUTS) if _layouts_kept(job, old, new) else (NORMALISE,)
    hidden = hidden_of(source, job, new, context) if job.case not in ("png-only", "whole") else None
    pixels = tuple(kind for kind in standards.pixel_kinds(new, rel=job.psd, chart=context.chart, hidden=hidden)
                   if kind not in skip) if job.case != "png-only" else ()
    if job.case != "png-only":                  # converted after the check (convert_plan), SWEEP_ONLY or not
        result.convert = list(standards.pixel_kinds(new, (NORMALISE,)))
    if job.case == "png-only":
        return
    # guard 2
    issues = standards.check(new, job.psd, context, old=old, added=added(job))
    told = {id(issue) for issue in standards.reported(issues)}
    result.issues = [{**issue.as_dict(), "line": issue.line(), "told": id(issue) in told, "discord": issue.discord}
                     for issue in issues]
    if (any(issue.fix and not issue.waived for issue in issues) or standards.composer_layers(new) or pixels
            or standards.folder_issues(new, job.psd, parser)):
        fonts = partial(source.fonts_for)
        fonts.earlier = earlier                 # card_standards.relay_badges, redraw_old_glyphs
        fonts.cards = context.cards             # relay_badges' notes: a break inside a card's name
        fonts.chart = context.chart             # status words
        fonts.hidden = hidden                   # clean's leftovers
        fixed = standards.fix(new_bytes, job.psd, issues, join(out, "work"), parser, fonts, skip=skip)
        result.fixed, result.composer, result.composer_left = fixed.applied, fixed.composer, fixed.composer_left
        result.pixel_left = list(fixed.pixel_left)
        lines = {issue.code: issue.line() for issue in issues if issue.fix}
        result.fix_refused = [(lines.get(code, code), why) for code, why in fixed.refused]
        redraws = bool(fixed.rewrapped or fixed.redrawn)
        _collect()                              # before the redraw drops a re-exported PNG's recompression
        redrawn = bool(fixed.data) and redraws and _rewrapped_pngs(out, result, new, fixed, shown, faces)
        data = fixed.plain if redraws and not redrawn else fixed.data
        if data and data != new_bytes:
            file = _save(out, job.psd, data)
            said = "; ".join(fixed.applied)     # what the comment and the post may say: never the composer
            composer = [f"Single-line Composer ({', '.join(fixed.composer)})"] if fixed.composer else []
            composer += [FOLDERS] if fixed.folders else []
            result.redrawn = dict(fixed.redrawn) if redrawn else {}
            result.pixel_notes = list(fixed.notes) if result.redrawn else []
            silent = [f"{standards.PIXEL_LINES[kind]} ({', '.join(names)})" for kind, names in result.redrawn.items()]
            cleaned = fixed.cleaned                 # clean (owner 11-09): silent, no PNG; in plain too (first)
            composer = ([f"{standards.PIXEL_LINES['clean']} ({', '.join(cleaned)})"] if cleaned else []) + composer
            write = {"path": job.psd, "file": file, "sha": blob_sha(data), "kind": "psd",
                     "line": "; ".join(fixed.applied + composer + silent), "shown": said, "silent": not said,
                     "composer": bool(fixed.composer), "folders": bool(fixed.folders),
                     "pixels": sorted(result.redrawn) + (["clean"] if cleaned else [])}
            if "tags" in result.redrawn:        # R1 is told only if this write doesn't land with its tags (after_window)
                write["settles"] = [issue["line"] for issue in result.issues if issue["code"] == "R1" and issue["told"]]
            if redrawn:                         # land swaps in plain when the redraw's PNGs do not land
                write["plain"] = {"file": _save(out, job.psd + ".plain", fixed.plain), "sha": blob_sha(fixed.plain),
                                  "line": "; ".join(fixed.applied + composer),
                                  "pixels": ["clean"] if cleaned else []}   # no redraw: no pixel fix but clean
            result.writes.append(write)
            result.links = _links(source, data, job.psd, write["sha"])
        if stale_left(fixed, result.redrawn):
            result.notes.append((OLD_FONT, True))
    if result.convert:                          # what the conversion's PNGs start from
        for path in shown:
            write = next((w for w in result.writes if w["path"] == path and w["kind"] == "png"), None)
            result.shown[path] = {"file": write["file"]} if write else {"blob": source.checked.get(path, ("", ""))[1]}
        result.shows = "fixed" if any(w.get("composer") for w in result.writes if w["kind"] == "png") else "pushed"


def _rgba(image) -> np.ndarray:
    return np.asarray(image.convert("RGBA"), dtype=np.uint16)


def _back_area(png, prev, render) -> Optional[np.ndarray]:
    """Where a pushed PNG left the PSD's render and the PNG before it showed it: visibly changed against both
    (png_changes' rule), the PNG before not against the render; there past the parity floor against the render, the PNG
    before within the bar. None when nowhere (or another size)."""
    if not png.size == prev.size == render.size:
        return None
    pushed, drawn = _rgba(png), _rgba(render)
    area = changed_area(pushed, drawn)
    if not area.any():
        return None
    before = _rgba(prev)
    area &= changed_area(pushed, before) & ~changed_area(before, drawn)
    if not area.any() or not psd_edit._past_floor(png_export._stats(png_export._diff(png, render)[area])) \
            or not psd_edit._within_bar(png_export._stats(png_export._diff(prev, render)[area])):
        return None
    return area


def _shows_older(png, area: np.ndarray, older) -> bool:
    """png shows an earlier PNG over area: visibly different from it (png_changes' rule) over at most OLDER_MATCH."""
    return older.size == png.size and int((changed_area(_rgba(png), _rgba(older)) & area).sum()) <= OLDER_MATCH * int(area.sum())


def _put_back(png, prev, area: np.ndarray):
    """png taking prev's pixels over area and what it reaches through their raw difference, PUT_BACK_REACH px at most
    (the rule's opening drops a change's edge and thin parts)."""
    moved = (png_export._diff(png, prev) > png_export.TOL) | area
    grown = area
    for _ in range(PUT_BACK_REACH):
        more = png_export.dilate(grown, 1) & moved
        if (more == grown).all():
            break
        grown = more
    image = np.where(grown[..., None], np.asarray(png_export.like_mode(prev, png.mode)), np.asarray(png))
    return Image.fromarray(image.astype(np.uint8), png.mode)


def _went_back(source: Source, job: Job, result: Result, png, render, out: str, shown: dict, old=None, new=None,
               faces=None) -> None:
    """
    R10 with no stored image (11-10): a pushed PNG that went back to an earlier picture where the PNG before it showed the
    PSD (_back_area), as one of the PNG's last OLDER_PNGS versions shows it there, is made again as a stale PNG's
    re-export, silent: alone, the PNG before taken back there (_put_back; all of it when that was its one change);
    with its PSD (a pair), guard 1's export of the PSD's change onto the PNG before. Shown by none, or the export
    refused, it stays as pushed (the comment says it).
    """
    prev_sha = job.png_old if job.case == "png-only" else job.base_png
    if not prev_sha or png.mode not in ("RGB", "RGBA"):
        return
    prev = png_export.open_png(source.blob(prev_sha))
    area = _back_area(png, prev, render)
    if area is None:
        return
    px, name = int(area.sum()), basename(job.png)

    def shows(sha: str) -> bool:
        try:
            return _shows_older(png, area, png_export.open_png(source.blob(sha)))
        except Exception:                       # not fetched or not a PNG: skipped
            return False
    older = next((sha for sha in source.older_pngs(job.png, job.before) if shows(sha)), "")
    result.r10.update(ok=False, back={"px": px, "box": list(png_export._bbox(area)), "older": older})
    if not older:
        result.r10["line"] = f"{name} differs from the PSD where the PNG before it showed the PSD ({px:,} px): kept as pushed"
        return
    if job.case == "pair":
        found = png_export.export(old, new, prev, None, "", faces) if old is not None else None
        if found is None or found.outcome == "refused":
            why = found.why if found is not None else "no PSD before the push"
            result.r10["line"] = f"{name} went back to an earlier picture (`{older[:8]}`, {px:,} px); not made again: {why}"
            return
        image, data = (found.image, found.data) if found.outcome == "re-exported" else (prev, source.blob(prev_sha))
    elif not (changed_area(_rgba(png), _rgba(prev)) & ~area).any():
        image, data = prev, source.blob(prev_sha)       # its one visible change went back: the PNG before, as it was
    else:
        image = _put_back(png, prev, area)
        data = png_export.encode(image, png)
    result.r10["line"] = f"{name} went back to an earlier picture (`{older[:8]}`, {px:,} px): made again"
    _made_again(out, job, result, data, f"made again: it went back to an earlier picture ({px:,} px)", "back")
    shown[job.png] = image


def _made_again(out: str, job: Job, result: Result, data: bytes, line: str, why: str) -> None:
    """R10's write of job's PNG made again (why: the write's key, back or skipped); its recompression gives way."""
    _collect()
    result.writes = [w for w in result.writes if not (w["path"] == job.png and w["kind"] == "recompress")]
    result.writes.append({"path": job.png, "file": _save(out, job.png, data), "sha": blob_sha(data), "kind": "png",
                          "line": line, why: True})


def _skipped(source: Source, job: Job, result: Result, png, old, new, out: str, shown: dict, faces=None) -> None:
    """
    R10, a PSD pushed with a PNG that skips its change (11-10, stored image or not): guard 1's export of the PSD's
    change onto the PNG before shows where the change moves it (png_changes' rule); a pushed PNG that did not change
    there and is past the bar of the export there (and of the PSD's stored image, when it has one) skipped it. Changed
    nowhere else (SKIP_REACH px past the export's change) and the export missing no bar (Export.misses), it is made
    again as the export, silent; else, or no export made, it stays as pushed (the comment says it).
    """
    try:
        prev = png_export.open_png(source.blob(job.base_png))
        if png.mode not in ("RGB", "RGBA") or prev.size != png.size:
            return
        found = png_export.export(old, new, prev, None, "", faces)
    except Exception:                           # best effort: R10 stands as checked
        print(format_exc(), file=stderr)
        return
    if found.outcome != "re-exported":         # nothing drawn moved, the PNG before shows it, or refused
        return
    made, pushed, before = _rgba(found.image), _rgba(png), _rgba(prev)
    changed = changed_area(pushed, before)
    area = changed_area(made, before) & ~changed
    if not area.any() or psd_edit._within_bar(png_export._stats(png_export._diff(png, found.image)[area])):
        return
    truth = png_export.composite(new)           # Photoshop's image: never skipped on the fork's word alone
    if truth is not None:
        truth = png_export.like_mode(png_export.convert(new, truth, psd_edit.export_icc(new, png)), png.mode)
        if truth.size != png.size or psd_edit._within_bar(png_export._stats(png_export._diff(png, truth)[area])):
            return
    px, name = int(area.sum()), basename(job.png)
    result.r10.update(ok=False, skipped={"px": px, "box": list(png_export._bbox(area))})
    if (changed & ~png_export.dilate(png_export._diff(found.image, prev) > png_export.TOL, SKIP_REACH)).any():
        result.r10["line"] = f"{name} does not show the PSD's change ({px:,} px) and changed elsewhere: kept as pushed"
        return
    if found.misses:                            # the export itself missed the bar
        result.r10["line"] = (f"{name} differs from the bot's drawing of the PSD's change ({px:,} px), which misses the "
                              f"bar itself: kept as pushed")
        return
    result.r10["line"] = f"{name} does not show the PSD's change ({px:,} px): made again"
    _made_again(out, job, result, found.data, f"made again: it did not show the PSD's change ({px:,} px)"
                + "".join(f"; {note}" for note in png_export.miss_notes(found)), "skipped")
    _crops(out, job.png, png, found.image, found.box)
    shown[job.png] = found.image


def _pair_pictures(source: Source, job: Job, result: Result, old, new, out: str, shown: dict, faces=None) -> None:
    """A PSD pushed with its PNG but not its numbered pictures (owner 11-07, silent): each drawn again as its picture (a
    tagged PSD's: png_export.render_picture) or taking the PNG's change (follow); one that refuses stays as it is."""
    left = {path: sha for path, sha in job.numbered.items() if path not in job.pushed}
    if not left:
        return
    tags = set(psd_edit.variants_of(new))
    numbers = picture_numbers(new, [*job.numbered, *job.pushed])   # a drawn duplicate's too
    base = png_export.open_png(source.blob(job.base_png))
    remade = any(w.get("back") or w.get("skipped") for w in result.writes)
    pushed = shown[job.png] if remade else png_export.open_png(source.blob(job.png_new))
    for path, sha in sorted(left.items()):
        image = png_export.open_png(source.blob(sha))
        n = numbers.get(path)
        if len(tags) > 1 and n in tags:
            data, _ = png_export.render_picture(old, new, image, n, faces)
        else:
            data = png_export.follow(base, pushed, {path: image}).get(path, b"")
        if data:
            result.writes.append({"path": path, "file": _save(out, path, data), "sha": blob_sha(data), "kind": "png",
                                  "line": "follows the PSD (its picture)", "silent": True, "dup": True})
            shown[path] = png_export.open_png(data)
        elif data is None:                      # it shows the pushed PSD already
            shown[path] = image


def follow_dups(data: dict, source: Source, out: str) -> None:
    """
    Each card's picture-only duplicates (Job.dups) after its PNG's final write, or after the PNG its pair push brought:
    png_export.follow from the PNG they were drawn with (at after, or before a pair push), silent writes in the same
    landing; one drawn apart from the PSD stays as it is (card_checks flags it). Again after the conversions (their PNG
    writes come later). Edits data in place.
    """
    for entry in data["cards"]:
        if not entry.get("dups") or not entry.get("png"):
            continue
        entry["writes"] = [w for w in entry["writes"] if not (w.get("dup") and w["path"] in entry["dups"])]
        write = next((w for w in entry["writes"] if w["kind"] == "png" and w["path"] == entry["png"] and w.get("file")), None)
        try:
            if write is not None:
                with open(join(out, write["file"]), "rb") as fh:
                    final = png_export.open_png(fh.read())
            elif entry["case"] == "pair" and entry.get("base_png"):
                final = png_export.open_png(source.blob(entry["expect"][entry["png"]]))
            else:
                continue
            base = entry.get("base_png") if entry["case"] == "pair" else entry["expect"].get(entry["png"])
            if not base:
                continue
            made = png_export.follow(png_export.open_png(source.blob(base)), final,
                                     {path: png_export.open_png(source.blob(sha)) for path, sha in entry["dups"].items()})
        except Exception:                       # best effort: the duplicates stay as they are
            print(format_exc(), file=stderr)
            continue
        for path, raw in sorted(made.items()):
            entry["expect"][path] = entry["dups"][path]
            entry["writes"].append({"path": path, "file": _save(out, path, raw), "sha": blob_sha(raw), "kind": "png",
                                    "line": f"follows {basename(entry['png'])} (a duplicate)", "silent": True, "dup": True})


def stale_left(fixed, redrawn: dict) -> list:
    """The layers drawn with another build of the icon font (old glyphs' finds, badge widths' re-laid layers) that the
    PSD keeps so: refused, a splice that fell back, or a PNG that couldn't follow (redrawn: what lands). Told (owner
    2026-11-04), unlike the pixel fixes' other refusals."""
    stale = dict.fromkeys([*getattr(fixed, "stale_glyphs", ()), *fixed.redrawn.get("badge widths", ())])
    done = {*redrawn.get("badge widths", ()), *redrawn.get("old glyphs", ())}
    return [name for name in stale if name not in done]


def _drawn_as(found) -> str:
    """What guard 1's main PNG is a render of: a tagged PSD saved showing another picture is drawn as picture 1."""
    return "its picture 1" if found.numbers.get("picture") else "the PSD"


def _rewrapped_pngs(out: str, result: Result, pushed, fixed, shown: dict, faces=None) -> bool:
    """
    R8's redrawn box text (fixed.rewrapped) into the card's PNGs through guard 1's export (pushed PSD -> fixed one, over
    the PNGs that show the pushed PSD): silent writes, a guard 1 write taking the new pixels. Whether the redraw stands
    (else the PSD lands without it: fixed.plain).
    """
    kinds = list(getattr(fixed, "redrawn", None) or {})    # the pixel fixes follow the same way, silent too

    def left(why: str) -> bool:
        result.composer_left += [(name, f"rewraps, not redrawn: {why}") for name in fixed.rewrapped]
        result.pixel_left += [(kind, f"not redrawn: {why}") for kind in kinds]
        return False
    if result.job.png not in shown:
        return left("no PNG that shows the PSD")
    gone = sorted(p for p in (*result.job.numbered, *result.job.pushed) if p not in shown)
    if gone:                                    # a numbered PNG would keep the old wrap
        return left(f"{basename(gone[0])} does not show the PSD")
    numbered = {path: image for path, image in shown.items() if path != result.job.png}
    found = png_export.export(pushed, png_export.open_psd(fixed.data, result.job.psd), shown[result.job.png], numbered,
                              fonts=faces)
    if found.outcome == "current":
        return True
    if found.outcome != "re-exported" or any(data == b"" for data, _ in found.numbered.values()):
        return left(found.why or next((why for data, why in found.numbered.values() if data == b""), ""))
    what = (["the Single-line Composer's rewrap"] if fixed.rewrapped else []) + [standards.PIXEL_LINES[k] for k in kinds]
    line = (f"re-exported for {' and '.join(what)} ({found.changed_px:,} px changed"
            + (", drawn as picture 1" if found.numbers.get("picture") else "") + ")")
    for path, data in [(result.job.png, found.data)] + [(p, d) for p, (d, _) in sorted(found.numbered.items()) if d]:
        write = next((w for w in result.writes if w["path"] == path and w["kind"] == "png"), None)
        result.writes = [w for w in result.writes if not (w["path"] == path and w["kind"] == "recompress")]
        if write is None:
            write = {"path": path, "kind": "png", "line": line, "shown": "", "silent": True}
            result.writes.append(write)
        else:
            write["line"] += f"; {line}"
        write.update(file=_save(out, path, data), sha=blob_sha(data), composer=True)
    return True


def changes(old, new, parser, fonts=None) -> list:
    """What changed inside the PSD (psd_diff), icons named by the parser's vocabulary; [] when psd_diff fails (it only
    tells, so it never stops a card)."""
    try:
        icons = getattr(getattr(parser, "icons", None), "codepoints", None)
        return psd_diff.changes(old, new, icons, QUIET_KEYS, fonts)
    except Exception:                           # report only: the guards go on without it
        print(format_exc(), file=stderr)
        return []


def _init_worker(clone: str, repository: str, context, vocab, tree: dict, fonts: str, earlier: tuple = ()) -> None:
    token_from_env()        # a spawned worker (macOS; forkserver on 3.14) starts without main()'s header
    wait_fonts()
    source = Source(clone, repository, fonts)
    source.checked = tree
    _worker.update(source=source, context=context, parser=vocab.parser(), earlier=tuple(earlier))


def _pooled(item: tuple) -> Result:
    job, out, mode, early = item
    with _overlap(_worker["source"], [job] if early else ()):
        return run_job(_worker["source"], job, _worker["context"], _worker["parser"], out, mode, _worker["earlier"])


# ========================
# Conversions and linked files (the 11-07 pass, PLAN 7: G1-G4; all silent)
# ========================
@dataclass
class LinkContext:
    """
    What the normalise fix's link step reads (fonts.links: psd_links.link_card's entries, read and first): the tree at
    after, the parent's catalogue (psd_links.assign over every PSD the run converts, so a new file shared by two of them
    gets one name, its first user's), the run's new files and the repo's by blob. Every user writes the new files it
    links (first: None), so none lands linking a file whose first user did not convert or land (one write each).
    """
    tree: dict                                  # {repo path: blob} at after
    catalogue: dict                             # psd_links.assign's {"psds": {rel: rows}, "files": {new path: info}}
    read_blob: Callable                         # blob sha -> bytes (Source.blob)
    files: dict = field(default_factory=dict)   # {new path: bytes} written by this run's first users
    redraw: bool = True                         # False: class 3 taken as exact copies (its PNG could not follow)
    ppis: dict = field(default_factory=dict)    # {repo path: ppi} read once (psd_links.ppis_of: each link's Rslt)

    def entries(self, rel: str) -> list:
        return self.catalogue.get("psds", {}).get(rel, [])

    def first(self, rel: str) -> None:
        return None                             # link_card: every new file it uses

    def read(self, path: str) -> Optional[bytes]:
        if path in self.files:
            return self.files[path]
        sha = self.tree.get(path)
        return self.read_blob(sha) if sha else None


def convertible(path: str) -> bool:
    """A PSD the guards convert and do nothing else to (Q3): MDW/Templates, MDW/Unfinished."""
    return path.lower().endswith(".psd") and path.startswith(CONVERT_ROOTS)


def _link_holder(path: str) -> bool:
    return path.lower().endswith((".psd", ".psb"))


def _links_to(link: dict, path: str) -> bool:
    """A link ({"path", "name"}) to path, blind to case; one no rule places by its file name (unused's rule)."""
    if link.get("path"):
        return link["path"].lower() == path.lower()
    return basename(link.get("name") or "").lower() == basename(path).lower()


def _restorable(path: str) -> bool:
    """A file PSDs link: MDW/Linked or Types (G4b puts it back when a push deletes it while still linked)."""
    return path.startswith((LINKED_DIR, TYPES_DIR + "/"))


def _restored(gone: dict, links: dict, records: dict, paths, made: set) -> dict:
    """
    G4b: {path: (blob before the push, PSDs linking it)} of the files the push deleted (gone: {path: blob}) that a PSD
    still links (links: {PSD: [{"path", "name"}]}, the pushed ones as read), a restored PSB's records counted too. Never
    one back in any letter case, or one a conversion writes (made).
    """
    holders, out = dict(links), {}
    gone = {path: blob for path, blob in gone.items() if blob and path not in paths and path not in made}
    while True:
        more = {path: [rel for rel, found in holders.items() if any(_links_to(link, path) for link in found)]
                for path in gone if path not in out}
        more = {path: users for path, users in more.items() if users}
        if not more:
            return out
        out.update({path: (gone[path], len(users)) for path, users in more.items()})
        holders.update({path: (records.get(path) or {}).get("links") or [] for path in more if _link_holder(path)})


def _missing(psd, rel: str, paths) -> list:
    """G4's candidates: [{"path", "size"}] of a PSD's links to MDW/Linked that place nowhere in the tree (paths), each
    at the path the link names (placed without the tree)."""
    from commands.psd_regions import linked_items
    placed = {item["uuid"] for item in linked_items(psd, rel, paths) if item["path"]}
    return [{"path": item["path"], "size": item["size"]} for item in linked_items(psd, rel)
            if item["uuid"] not in placed and (item["path"] or "").startswith(LINKED_DIR)]


def _holders(pairs) -> str:
    """A digest of a tree's PSDs and PSBs ((path, blob) pairs): what a linked-file deletion was decided on."""
    return blob_sha("\n".join(sorted(f"{path}\t{blob}" for path, blob in pairs if _link_holder(path))).encode())


def links_index(source: Source) -> Optional[dict]:
    """db/links.json at the data branch (card_data.gather_links; one blob read), None without it (nothing deleted)."""
    data = source.commit("origin/data") or source.commit("data")
    found = source.tree(data).get(LINKS) if data else None
    if not found:
        return None
    try:
        return loads(source.blob(found[1]))
    except Exception:                           # unread: no housekeeping this run
        print(format_exc(), file=stderr)
        return None


def _checked_links(entry: Optional[dict], tree: dict) -> Optional[dict]:
    """A pushed card's links as its check read them (Result.links), when they are of the PSD as it lands."""
    write = next((w for w in (entry or {}).get("writes", []) if w["kind"] == "psd" and w["path"] == entry["psd"]), None)
    found = (entry or {}).get("links")
    return found if found and found["sha"] == (write["sha"] if write is not None else tree[entry["psd"]]) else None


def _final(out: str, entry: dict, source: Source, tree: dict) -> bytes:
    """A pushed PSD as it lands: its planned PSD write, else its blob at after."""
    write = next((w for w in entry.get("writes", []) if w["kind"] == "psd" and w["path"] == entry["psd"]), None)
    if write is not None:
        with open(join(out, write["file"]), "rb") as fh:
            return fh.read()
    return source.blob(tree[entry["psd"]])


def _items(item: tuple) -> tuple:
    """(rel, psd_links.items of its PSD at after, or None unread): the catalogue's worker."""
    from commands import psd_links
    clone, repository, rel, sha, vocab = item
    try:
        return rel, psd_links.items(png_export.open_psd(Source(clone, repository).blob(sha)), vocab)
    except Exception:                           # unread: that PSD converts with no catalogue rows (left as pushed)
        print(format_exc(), file=stderr)
        return rel, None


def catalogue(source: Source, tree: dict, rels: list, index: Optional[dict], vocab=(), workers: int = 0) -> dict:
    """The parent's catalogue of the PSDs to convert: their items read in the pool (psd_links.items), then named here in
    catalogue order (psd_links.assign: a new file two of them share is named once, after its first user). Types/ from
    the tree (the checkout's /Types/, else by blob), its older blobs from db/links.json's files, MDW/Linked by blob."""
    from commands import psd_links
    jobs = [(source.clone, source.repository, rel, tree[rel], vocab) for rel in psd_links.catalogue_order(rels)]
    if len(jobs) >= POOL_MIN and workers != 1:
        with ProcessPoolExecutor(workers or pool_size(len(jobs)), initializer=token_from_env) as pool:
            read = list(pool.map(_items, jobs, chunksize=1))
    else:
        read = [_items(job) for job in jobs]
    found = [(rel, rows) for rel, rows in read if rows is not None]
    history = {blob: path for path, blobs in ((index or {}).get("files") or {}).items() if path.startswith("Types/")
               for blob in blobs}
    types = psd_links.TypesIndex({path: source.blob(sha) for path, sha in tree.items()
                                  if dirname(path) in (TYPES_DIR, STARS_DIR)}, history)
    linked = {path: sha for path, sha in tree.items() if path.startswith(LINKED_DIR)}
    out = psd_links.assign(found, types, linked)
    psd_links.fallback_names(out, linked)       # a class 2 or 3 refused, or redraw off: its exact copy's name (5.2 step 9)
    return out


def _shown_images(out: str, source: Source, entry: dict) -> dict:
    """{PNG path: image} of the PNGs that show the PSD the conversion starts from (the check's "shown")."""
    images = {}
    for path, found in entry.get("shown", {}).items():
        if found.get("file"):
            with open(join(out, found["file"]), "rb") as fh:
                images[path] = png_export.open_png(fh.read())
        else:
            images[path] = png_export.open_png(source.blob(found["blob"]))
    return images


def _convert_one(source: Source, entry: dict, ctx: LinkContext, parser, out: str, deadline: float) -> dict:
    """One PSD's normalise (card_standards.fix, only normalise) on top of the check's write; a class 3 redraw takes its
    PNGs along (png_export.export over the PNGs the check found showing the PSD), else it is redone as exact copies.
    "records": what the pushed PSD hands the data's records before the strip (psd_edit.records_of), even when the time
    ran out (the check's own write strips it too)."""
    tree, rel = ctx.tree, entry["psd"]
    shown = _shown_images(out, source, entry) if entry.get("png") in entry.get("shown", {}) else {}
    pushed = png_export.open_psd(source.blob(tree[rel]), rel)
    facts = psd_edit.look_facts(pushed, shown.get(entry.get("png"))) if pushed.has_preview() else {}

    def done(result: dict) -> dict:
        found = {"render": {rel: facts}} if facts else {}
        if "psd" in result and parser is not None:  # icons the conversion redrew: their cores before it
            spots = psd_edit.spot_entries(pushed, png_export.open_psd(result["psd"]), parser, rel,
                                          basename(rel)[:len(basename(rel)) - 4])
            found.update({"spot": {rel: spots}} if spots else {})
        return {**result, "records": found} if found else result
    if time() > deadline:
        return done({"left": "not converted: the guards' time ran out (its next push converts it)"})
    base = _final(out, entry, source, tree)
    fonts = partial(source.fonts_for)
    fonts.links = ctx
    for redraw in (True, False):
        ctx.redraw = redraw
        fixed = standards.fix(base, rel, [], join(out, "work"), parser, fonts, only=NORMALISE)
        if not fixed.data:
            return done({"left": "; ".join(why for _, why in fixed.pixel_left) or "nothing to convert"})
        redraws = list(fixed.redrawn.get(NORMALISE) or ())     # card_standards: the layers drawn again
        if not redraws:
            return done({"psd": fixed.data, "files": dict(fixed.files), "pngs": {}, "left": list(fixed.left)})
        if not shown:                           # no PNG shows it (templates, Unfinished): nothing follows
            if entry.get("png") or is_card_psd(rel):
                continue                        # a card's PNG not followed here (a rename): exact copies
            return done({"psd": fixed.data, "files": dict(fixed.files), "pngs": {}, "left": list(fixed.left)})
        start = psd_edit.use_record(png_export.open_psd(base if entry.get("shows") == "fixed" else source.blob(tree[rel])),
                                    rel, facts or None)
        numbered = {path: image for path, image in shown.items() if path != entry["png"]}
        found = png_export.export(start, psd_edit.use_record(png_export.open_psd(fixed.data), rel, facts or None),
                                  shown[entry["png"]], numbered)
        if found.outcome == "refused" or any(data == b"" for data, _ in found.numbered.values()):
            continue                            # the redraw cannot reach its PNGs: exact copies instead
        pngs = {entry["png"]: found.data} if found.outcome == "re-exported" else {}
        pngs.update({path: data for path, (data, _) in found.numbered.items() if data})
        return done({"psd": fixed.data, "files": dict(fixed.files), "pngs": pngs, "left": list(fixed.left), "base": base})
    return done({"left": "not converted: its redrawn pictures' PNGs could not follow"})


def _merge(out: str, entry: dict, done: dict, tree: dict) -> None:
    """A conversion into the card's writes: its PSD write (the check's carries it), its new files, its PNGs; silent."""
    entry["converted"] = "psd" in done
    if done.get("records"):                     # land hands them to the data with the PSD write (records.json)
        entry["records"] = done["records"]
    if done.get("left"):
        entry.setdefault("convert_left", []).extend([done["left"]] if isinstance(done["left"], str) else done["left"])
    if "psd" not in done:
        return
    line = standards.PIXEL_LINES.get(NORMALISE, NORMALISE)
    write = next((w for w in entry["writes"] if w["kind"] == "psd" and w["path"] == entry["psd"]), None)
    if write is None:
        write = {"path": entry["psd"], "kind": "psd", "line": line, "shown": "", "silent": True, "composer": False,
                 "pixels": []}
        entry["writes"].append(write)
    else:
        write["line"] = "; ".join(filter(None, [write.get("line"), line]))
    if done.get("pngs") and not write.get("plain"):     # a PNG that can't land lands the PSD as it was checked
        write["plain"] = {"file": _save(out, entry["psd"] + ".plain", done["base"]), "sha": blob_sha(done["base"]),
                          "line": write["line"], "pixels": list(write.get("pixels", []))}
    write.update(file=_save(out, entry["psd"], done["psd"]), sha=blob_sha(done["psd"]),
                 pixels=sorted({*write.get("pixels", []), NORMALISE}))
    for path, data in sorted(done.get("files", {}).items()):
        entry["expect"][path] = tree.get(path, "")
        if data is None:
            entry["writes"].append({"path": path, "kind": "delete", "file": None, "sha": None, "silent": True,
                                    "line": "linked file deleted"})
        else:
            entry["writes"].append({"path": path, "kind": "file", "file": _save(out, path, data), "sha": blob_sha(data),
                                    "silent": True, "line": "linked file added"})
    for path, data in sorted(done.get("pngs", {}).items()):
        entry["writes"] = [w for w in entry["writes"] if not (w["path"] == path and w["kind"] == "recompress")]
        old = next((w for w in entry["writes"] if w["path"] == path and w["kind"] == "png"), None)
        if old is None:
            old = {"path": path, "kind": "png", "line": f"re-exported for {line}", "shown": "", "silent": True}
            entry["writes"].append(old)
        old.update(file=_save(out, path, data), sha=blob_sha(data), composer=True)


def _init_convert(clone: str, repository: str, ctx: LinkContext, vocab, tree: dict, fonts: str) -> None:
    token_from_env()
    wait_fonts()
    source = Source(clone, repository, fonts)
    source.checked = tree
    ctx.read_blob = source.blob
    _worker.update(source=source, ctx=ctx, parser=vocab.parser() if vocab is not None else None)


def _pooled_convert(item: tuple) -> dict:
    entry, out, deadline = item
    try:
        return _convert_one(_worker["source"], entry, _worker["ctx"], _worker["parser"], out, deadline)
    except Exception as error:                  # one bad PSD never stops the others: left as pushed
        print(format_exc(), file=stderr)
        return {"left": f"not converted ({type(error).__name__}: {error})"}


def housekeeping(data: dict, source: Source, tree: dict, out: str, index: Optional[dict], deadline: float = 0.0) -> dict:
    """
    G2-G4 over what lands (psd_links.unused, duplicates, restore, relink), as plan.json's "linked" entry, silent: the
    MDW/Linked files no PSD links deleted (nothing while any PSD's links are unknown); a pushed MDW/Linked file with the
    bytes of another merged into it (every user relinked, no pixel changed; within the time budget, else both stay); a
    link to an MDW/Linked file missing at after restored from history when its last blob has the link's size (a file
    pushed ahead of its PSD and deleted meanwhile, R13); first, an MDW/Linked or Types file the push deleted put back at
    its blob before the push while any PSD still links it (G4b, _restored: the pushed PSDs as read, the rest by their
    records at any blob; a restored PSB's links stay used, and no duplicate is merged beside it).
    """
    from commands import psd_links
    entry = {"card": "", "psd": "", "case": "linked", "login": SUBJECT, "commit": data["after"], "writes": [],
             "expect": {}, "notes": [], "told": []}
    pushed_linked = [p for p in data.get("pushed_paths", []) if p.startswith(LINKED_DIR)]
    cards = {e["psd"]: e for e in data["cards"] if e.get("psd") in tree}
    finals: dict = {}
    pooled: dict = {}                           # pushed PSDs whose links the check read (not read again here)
    for path in data.get("pushed_paths", []):
        if _link_holder(path) and path in tree:
            if found := _checked_links(cards.get(path), tree):
                pooled[path] = found
                continue
            try:
                finals[path] = _final(out, cards[path], source, tree) if path in cards else source.blob(tree[path])
            except Exception:                   # unread: its links unknown, nothing deleted
                print(format_exc(), file=stderr)
                return entry
    paths = RepoTree(tree)
    entry["holders"] = _holders(tree.items())   # land drops the deletions once any PSD moved (it may link them)
    read = sorted([(rel, psd_links.links_of(png_export.open_psd(raw), rel, paths)) for rel, raw in finals.items()]
                  + [(rel, [SimpleNamespace(**link) for link in found["links"]]) for rel, found in pooled.items()],
                  key=lambda pair: pair[0])
    direct = dict(read)
    records = psd_links.records_of(index)[0] if index is not None else {}
    known = index is not None and all(rel in direct or ((records.get(rel) or {}).get("blob") == blob
                                                        and not records[rel].get("error"))
                                      for rel, blob in tree.items() if _link_holder(rel))   # G2's rule: every user known
    links = {rel: [{"path": link.path, "name": link.name} for link in found] for rel, found in direct.items()}
    links.update({rel: record.get("links") or [] for rel, record in records.items()
                  if rel not in direct and isinstance(record, dict) and rel in tree})
    made = {w["path"] for e in data["cards"] for w in e.get("writes", []) if w["kind"] == "file"}
    restored = _restored(data.get("deleted_linked") or {}, links, records, paths, made)
    unplaced = {basename(link.get("name") or "").lower() for found in links.values() for link in found if not link.get("path")}
    dups = {dup: kept for dup, kept in psd_links.duplicates(tree, pushed_linked).items()
            if basename(dup).lower() not in unplaced} if known and not any(map(_link_holder, restored)) else {}
    for path, (old, count) in sorted(restored.items()):
        try:
            raw_file = source.blob(old)
        except (Transient, Refused):
            continue
        entry["expect"][path] = ""
        entry["writes"].append({"path": path, "kind": "file", "file": _save(out, path, raw_file), "sha": old,
                                "silent": True, "line": f"restored: still linked by {plural(count, 'PSD')}"})
    back = {**tree, **{path: old for path, (old, _) in restored.items()}}    # a restored PSB's links stay used
    for path in psd_links.unused(index, back, pushed_linked, read) if index is not None else ():
        if path in dups.values() or path in restored:   # G3 points its duplicate's users at it
            continue
        entry["expect"][path] = tree[path]
        entry["writes"].append({"path": path, "kind": "delete", "file": None, "sha": None, "silent": True,
                                "line": "unused linked file deleted"})
    deleted = {w["path"] for w in entry["writes"] if w["kind"] == "delete"}
    users: dict = {}                            # PSD -> the duplicates it links
    for rel, found in links.items():
        for link in found:
            if link.get("path") in dups:
                users.setdefault(rel, set()).add(link["path"])
    held = {dup for rel, mine in users.items() if rel in cards and any(w.get("plain") for w in cards[rel]["writes"])
            for dup in mine}                    # a write that may land as its plain (unrelinked): both files stay
    relinked: dict = {}
    for rel in sorted(users):
        mine = users[rel] - held
        if not mine:
            continue
        if deadline and time() > deadline:      # the rest do not fit the time: their files stay (harmless)
            break
        psd = png_export.open_psd(finals.get(rel) or (_final(out, cards[rel], source, tree) if rel in pooled
                                                       else source.blob(tree[rel])))
        for link in psd_links.links_of(psd, rel, paths):
            if link.path in mine:
                psd_links.relink(psd, link.uuid, rel, dups[link.path], link.size, tree[dups[link.path]])
        saved = BytesIO()
        psd_edit.save_psd(psd, saved)           # one save a PSD, every duplicate it links
        relinked[rel] = (saved.getvalue(), sorted({dups[dup] for dup in mine}))
    for rel, (raw, kept) in sorted(relinked.items()):
        owner = cards.get(rel, entry)           # a pushed card: its own write carries the relink
        write = next((w for w in owner["writes"] if w["kind"] == "psd" and w["path"] == rel), None)
        if write is None:
            write = {"path": rel, "kind": "psd", "line": "", "shown": "", "silent": True, "composer": False, "pixels": []}
            owner["writes"].append(write)
            owner["expect"].setdefault(rel, tree[rel])
        write["line"] = "; ".join(filter(None, [write.get("line"), f"relinked to {', '.join(kept)}"]))
        write.update(file=_save(out, rel, raw), sha=blob_sha(raw))
    for dup, kept in sorted(dups.items()):
        if dup in deleted or dup in held or any(dup in mine and rel not in relinked for rel, mine in users.items()):
            continue                            # a user not relinked: both files stay
        entry["expect"][dup] = tree[dup]
        entry["writes"].append({"path": dup, "kind": "delete", "file": None, "sha": None, "silent": True,
                                "line": f"the same file as {kept}"})
    for rel in sorted({*finals, *pooled}):
        items = pooled[rel]["restore"] if rel in pooled else _missing(png_export.open_psd(finals[rel]), rel, paths)
        for item in items:
            path = item["path"] or ""
            if not path.startswith(LINKED_DIR) or path in tree or path in entry["expect"] or path in made:
                continue                        # a conversion's new file is written by its card
            try:
                last = source.git("log", "-1", "--format=%H", "--diff-filter=AM", data["after"], "--", path).strip()
                blob = source.git("rev-parse", f"{last}:{path}").strip() if last else ""
                raw_file = source.blob(blob) if blob else b""
            except (Transient, Refused):
                continue
            if blob and psd_links.restore(item["size"], [(blob, len(raw_file))]):
                entry["expect"][path] = ""
                entry["writes"].append({"path": path, "kind": "file", "file": _save(out, path, raw_file), "sha": blob,
                                        "silent": True, "line": f"restored for {rel}"})
    return entry


def font_fixes(source: Source, pushed: list, author: str, head: str, out: str) -> dict:
    """
    plan.json's "fonts" entry, silent: each font the push added or changed ([Change]) at the top of MDW/Fonts whose
    tables share data (font_check.shared_tables) given its own copies (split_tables), resting on the pushed blob. One
    not read or not split stays as pushed.
    """
    entry = {"card": "", "psd": "", "case": "fonts", "login": author, "commit": head, "writes": [], "expect": {},
             "notes": [], "told": []}
    for change in pushed:
        if not tree_fonts({change.path: change.new}):
            continue
        try:
            raw = source.blob(change.new)
            shared = shared_tables(raw)
            fixed = split_tables(raw) if shared else None
        except Exception:                       # unread: as pushed
            print(format_exc(), file=stderr)
            continue
        if fixed and not shared_tables(fixed):
            entry["expect"][change.path] = change.new
            entry["writes"].append({"path": change.path, "kind": "font", "file": _save(out, change.path, fixed),
                                    "sha": blob_sha(fixed), "silent": True,
                                    "line": "tables given their own data (" + ", ".join("/".join(p) for p in shared) + ")"})
    return entry


def convert_plan(data: dict, source: Source, out: str, minutes: float = CONVERT_MINUTES, workers: int = 0,
                 deadline: float = 0.0) -> dict:
    """
    G1-G4 after the check (the CLI's `normalise`, or check's --convert now): every PSD the check marked (an embedded
    picture, sub or superscript, a merged image, a link to repair, a linked stat bar) converted by the normalise fix
    with the parent's names, then the linked files' housekeeping (data["linked"]). No PSD starts past deadline (else
    `minutes` from now); the rest stay as pushed (their next push converts them). Silent throughout. Edits data in place
    and returns it.
    """
    tree = {path: sha for path, (_, sha) in source.checked.items()}
    index = links_index(source)
    entries = [e for e in data["cards"] if e.get("convert") and not e.get("error") and "converted" not in e]
    deadline = deadline or time() + minutes * 60
    if entries:
        context, vocab = build_context(source, source.checked)
        ctx = LinkContext(tree, catalogue(source, tree, [e["psd"] for e in entries], index,
                                          sorted(vocab.all_types) if vocab is not None else (), workers), source.blob)
        if len(entries) >= POOL_MIN and workers != 1:
            with ProcessPoolExecutor(workers or pool_size(len(entries)), initializer=_init_convert,
                                     initargs=(source.clone, source.repository, ctx, vocab, source.checked, source.fonts)) as pool:
                done = list(pool.map(_pooled_convert, [(e, out, deadline) for e in entries], chunksize=1))
        else:
            parser = vocab.parser() if vocab is not None else None
            done = []
            for entry in entries:
                try:
                    done.append(_convert_one(source, entry, ctx, parser, out, deadline))
                except Exception as error:      # left as pushed
                    print(format_exc(), file=stderr)
                    done.append({"left": f"not converted ({type(error).__name__}: {error})"})
        for entry, result in zip(entries, done):
            _merge(out, entry, result, tree)
        follow_dups(data, source, out)          # a conversion's PNG write: its duplicates follow it instead
    try:
        data["linked"] = housekeeping(data, source, tree, out, index, deadline)
    except Exception:                           # best effort: nothing deleted or merged this run
        print(format_exc(), file=stderr)
        data["linked"] = {"card": "", "psd": "", "case": "linked", "writes": [], "expect": {}, "notes": [], "told": []}
    return data


# ========================
# Card file names a file name can't hold (owner 2026-11-11): renamed silently on push
# ========================
def _unsafe(path: str) -> bool:
    return any(c in NOT_IN_FILE_NAMES for c in splitext(basename(path))[0])


def unsafe_renames(tree: dict, pushed: list, writes=()) -> dict:
    """
    {old path: new path} of the card files a push renames: every card PSD or PNG it brought whose file name holds a
    character of card_models.NOT_IN_FILE_NAMES, with its card's other files (its PSD, PNG and numbered PNGs, and the PNGs
    this run writes: writes), each the stem's file form (file_stem; the numbers kept). A card none is left of, or whose
    new name another file has (any case), keeps its names. MDW/, Markers/ and Types/ files never (no card there).
    """
    snapshot = Snapshot("", "", {}, tree)
    lower = {path.lower(): path for path in tree}
    cards: dict = {}                            # the card's PSD (or a lone PNG) -> its files
    for path in pushed:
        if not _unsafe(path) or not ((is_card_psd(path) and path.startswith(CARD_ROOTS)) or card_png(path)):
            continue
        psd = path
        if _is_png(path):
            stem = path[:len(path) - 4]
            psd = lower.get(stem.lower() + ".psd") or lower.get(stem.rstrip("0123456789").lower() + ".psd") or path
        if psd in cards or (psd != path and psd not in tree):
            continue
        png, numbered = snapshot.png_for(psd) if psd.lower().endswith(".psd") else ("", [])
        stem = basename(psd)[:len(basename(psd)) - 4]
        made = [w for w in writes if _is_png(w) and dirname(w) == dirname(psd)
                and fullmatch(f"{escape(stem)}\\d*", basename(w)[:len(basename(w)) - 4], IGNORECASE)]
        cards[psd] = list(dict.fromkeys(p for p in [psd, png, *numbered, *made] if p))
    out, taken = {}, set()
    for psd, files in cards.items():
        stem = basename(psd)[:len(basename(psd)) - 4]
        new_stem = file_stem(stem)
        if not new_stem:
            continue
        moves = {}
        for old in files:
            name = basename(old)[:len(basename(old)) - 4]
            moves[old] = f"{dirname(old)}/{new_stem}{name[len(stem):]}{old[len(old) - 4:]}"
        mine = {old.lower() for old in files}
        if any(new.lower() in taken or (new.lower() in lower and new.lower() not in mine) for new in moves.values()):
            continue
        taken.update(new.lower() for new in moves.values())
        out.update(moves)
    return out


# ========================
# Reports
# ========================
def _expect(job: Job) -> dict:
    """The blobs a card's writes rest on at the head (G9): the PNG as checked ("" for one the bot makes)."""
    expect = {job.psd: job.psd_new}
    if job.png:
        expect[job.png] = job.png_old if job.case in ("stale", "beside") else job.png_new
    for path, sha in (*job.numbered.items(), *job.pushed.items()):
        expect[path] = sha
    return expect


def plan(results: list, repository: str, before: str, after: str, mode: str) -> dict:
    """plan.json: the writes per card with the blobs they rest on, and the verdicts."""
    cards = []
    for result in results:
        entry = {"card": result.name, "psd": result.job.psd, "case": result.job.case, "login": result.job.login,
                 "commit": result.job.commit, "export": result.export, "r10": result.r10, "issues": result.issues,
                 "fixed": result.fixed, "fix_refused": result.fix_refused, "notes": [t for t, _ in result.notes],
                 "told": result.told(mode), "error": result.error, "writes": result.writes, "png": result.job.png,
                 "changes": result.changes, "expect": _expect(result.job), "pushed": sorted(result.job.pushed),
                 "composer": result.composer, "composer_left": result.composer_left,
                 "redrawn": result.redrawn, "pixel_left": result.pixel_left, "pixel_notes": result.pixel_notes,
                 "convert": result.convert, "shown": result.shown, "shows": result.shows,
                 "dups": result.job.dups, "base_png": result.job.base_png, "links": result.links}
        cards.append(entry)
    return {"repo": repository, "before": before, "after": after, "mode": mode, "checked_at": _now().isoformat(),
            "cards": cards}


def _box(numbers: dict, box) -> str:
    if not box:
        return ""
    x0, y0, x1, y1 = box
    return f" in {x1 - x0}x{y1 - y0} at ({x0}, {y0})"


def _guard1_line(entry: dict, mode: str = "on") -> str:
    found = entry["export"]
    outcome = found.get("outcome")
    if not outcome:
        return ""
    if outcome == "current" and "png_ahead" in found.get("numbers", {}):
        return "PNG already shows the pushed PSD (exported before the PSD was pushed): nothing to make"
    if outcome == "current":
        return "PNG still current (the push moved no pixel of the card)"
    if outcome == "refused":
        return found["why"]
    would = " would be" if mode != "on" else ""
    switch = " with CARD_GUARDS=on" if mode != "on" else ""
    if found.get("copied"):
        return f"PNG{would} copied to the PSD's new name{switch}"
    numbers = found.get("numbers", {})
    parts = [f"{found['changed_px']:,} px changed{_box(numbers, found.get('box'))}", f"ICC {found.get('icc') or 'as stored'}"]
    if "baseline" in numbers:
        parts.append(f"the old PSD drawn against the old PNG: bad {numbers['baseline'].get('bad_pct', 0):.2f}%")
    # png_vs_render (G2 beside) is plan.json's alone: it measures the author's change (default 2026-10-24, owner to confirm)
    if "cross" in numbers:
        parts.append(f"against the PSD's own image: bad {numbers['cross'].get('bad_pct', 0):.2f}%, "
                     f"mean {numbers['cross'].get('mean', 0):.2f}")
    if found.get("writer") in ("photoshop", "tool"):
        parts.append("saved by Photoshop" if found["writer"] == "photoshop" else "not saved by Photoshop")
    verb = "made" if entry["case"] == "new" else "re-exported"
    misses = "".join(f"; {note}" for note in found.get("misses", []))
    return f"PNG{would} {verb} by the bot's render of the PSD ({'; '.join(parts)}){switch}{misses}"


ALREADY = "already on the branch (an earlier run of this push landed it)"
UNREAD = "an earlier run may have landed it (the branch could not be read)"


def _already(landed: dict, card: str) -> set:
    """The paths of a card's writes the branch already held (a re-run after a landing)."""
    return {path for name, path in landed.get("already", []) if name == card}


def _shown(entry: dict) -> list:
    """A card's writes the comment and the post may name: not a composer fix alone (owner, 2026-10-25)."""
    return [write for write in entry.get("writes", []) if not write.get("silent")]


def _unsaid(data: dict, landed: dict) -> dict:
    """landed as the comment tells it: the silent writes left out of what landed, was dropped or was already there."""
    silent = {(entry["card"], write["path"]) for entry in data["cards"] for write in entry.get("writes", [])
              if write.get("silent")}
    if not silent:
        return landed
    included = landed.get("included")
    cards = (list(dict.fromkeys(card for card, path in included if card and (card, path) not in silent)) if included is not None
             else landed.get("landed", []))
    return {**landed, "landed": cards, "dropped": [d for d in landed.get("dropped", []) if (d[0], d[2]) not in silent],
            "already": [a for a in landed.get("already", []) if tuple(a) not in silent]}


def _landed_head(landed: dict) -> str:
    sha, error = landed.get("sha", ""), landed.get("error", "")
    left = len({card for card, *_ in landed.get("dropped", [])} - set(landed.get("landed", [])))
    already = {card for card, _ in landed.get("already", [])} - set(landed.get("landed", []))
    out = (f"Landed in `{sha[:8]}`: {plural(len(landed.get('landed', [])), 'card')}." if sha and landed.get("landed") else
           "" if sha else
           f"Not landed: {error.rstrip('.')}. Nothing was committed by this run; {UNREAD}." if error and landed.get("unread") else
           f"Not landed: {error.rstrip('.')}. Nothing was committed." if error else
           f"Nothing new to land: {plural(len(already), 'card')} {ALREADY}." if already else "Nothing landed.")
    return out + (f" {plural(left, 'card')} left out: changed again on the branch." if left and not error else "")


def _landing(entry: dict, landed: dict) -> list:
    """The comment's lines on what became of a card's writes (the silent ones left out) after the quiet window."""
    entry = {**entry, "writes": _shown(entry)}
    if not entry["writes"]:
        return []
    already = _already(landed, entry["card"])
    if landed.get("error"):
        if already and len(already) == len(entry["writes"]):
            return [f"- Not committed again: {ALREADY}"]
        if landed.get("unread"):
            return [f"- Not committed by this run: {UNREAD}"]
        return ["- Not landed (the bot's commit failed): nothing above was committed for this card"]
    dropped = {path: why for card, why, path in landed.get("dropped", []) if card == entry["card"]}
    if len(dropped) == len(entry["writes"]):
        return [f"- Not landed ({next(iter(dropped.values()))}): nothing above was committed for this card"]
    out = [f"- Not landed ({why}): `{path}`" for path, why in dropped.items()]
    if landed.get("sha") and any(w["path"] not in dropped and w["path"] not in already for w in entry["writes"]):
        out.append(f"- {'The rest landed' if out else 'Landed'} in `{landed['sha'][:8]}`")
    if already:
        out.append(f"- {'The rest is' if out else 'Every write is'} {ALREADY}")
    return out


def markdown(data: dict, notes: list, skipped, errors: list, landed: Optional[dict] = None) -> str:
    """The job summary and commit comment; with land's report (landed.json), the comment as edited after the quiet window."""
    mode = data["mode"]
    head = {"dry": "Dry run: nothing is posted or committed.", "report": "Report only (CARD_GUARDS=report): nothing is committed.",
            "on": "Writes land right after the check, in one bot commit."}[mode]
    final = landed is not None and not landed.get("dry")
    if final:
        landed = _unsaid(data, landed)
        head = _landed_head(landed)
    out = ["## Card guards", "", f"Push `{data['after'][:8]}` (against `{data['before'][:8]}`): "
                                 f"{plural(len(data['cards']), 'card')}." + (f" {head}" if head else "")]
    for note in notes:
        out.append(f"- {note}")
    if skipped:
        out.append(f"- Left out (bot commits): {', '.join(f'`{p}`' for p in sorted(skipped))}")
    out.append("")
    if not data["cards"]:
        out.append("No card PSD or PNG to guard.")
    for entry in data["cards"]:
        if entry["case"] == "convert" and not (entry["error"] or entry["told"]):
            continue                            # converted silently (Q3, G4): nothing to say
        out.append(f"**{entry['card']}** (`{entry['psd']}`, {entry['login']}, `{entry['commit'][:8]}`)")
        if entry.get("changes"):
            out.append(f"- Changed: {_changes_line(entry, CHANGES_COMMENT)}")
        line = _guard1_line(entry, mode)
        if line:
            out.append(f"- {line}")
        for why, path in entry["export"].get("numbered_refused", []):
            out.append(f"- {why}")
        if entry["r10"].get("ok") is False:
            out.append(f"- {entry['r10']['line']}")
        elif entry["r10"].get("ok"):
            out.append(f"- {basename(entry['psd'])[:-4]}.png matches the PSD (mean {entry['r10'].get('mean', 0):.2f})")
        for write in entry.get("writes", []):
            if write.get("kind") == "recompress":
                out.append(f"- {basename(write['path'])} {write['line']}" if mode == "on" else
                           f"- {basename(write['path'])} would be {write['line']} with CARD_GUARDS=on")
        told = [i["line"] for i in entry["issues"] if i["told"]]
        if told:
            out.append("- Issues: " + "; ".join(told))
        if entry["fixed"]:
            psd_landed = final and (entry["psd"] in _already(landed, entry["card"]) or (
                landed.get("sha") and not landed.get("error") and not any(
                    card == entry["card"] and path == entry["psd"] for card, _, path in landed.get("dropped", []))))
            verb = ("The bot would fix (CARD_GUARDS=on)" if mode != "on" else "The bot will fix" if not final else
                    "Fixed by the bot" if psd_landed else "Not fixed by this run" if landed.get("unread") else "Not fixed")
            out.append(f"- {verb}: " + "; ".join(entry["fixed"]))
        for line, why in entry["fix_refused"]:
            out.append(f"- Not fixed: {line} ({why})")
        waived = [f"{i['code']} {i['line']}" for i in entry["issues"] if i["waived"]]
        if waived:
            out.append("- Waived (owner decisions): " + "; ".join(waived))
        for note in entry["notes"]:
            out.append(f"- {note}")
        if final:
            out += _landing(entry, landed)
        for note in (landed or {}).get("after", {}).get(entry["psd"], []):
            out.append(f"- After the quiet window: {note}")
        if not (line or told or entry["fixed"] or entry["fix_refused"] or entry["notes"] or entry["r10"] or _shown(entry)):
            out.append("- Nothing to recompress." if entry.get("case") == "numbered" else "- Meets the standards.")
        out.append("")
    if errors:
        out += ["Errors:"] + [f"- {e}" for e in errors] + [""]
    return "\n".join(out)


def _changes_line(entry: dict, limit: int) -> str:
    """The card's changes (psd_diff) as markdown text: card text and layer names never format or link."""
    return psd_diff.line([sub(MARKDOWN, r"\\\1", phrase) for phrase in entry.get("changes", [])], limit)


def discord_payload(data: dict, repository: str) -> dict:
    """
    The webhook body: one LINE per card with something to tell (a refusal or a remaining issue); whole lines go from the
    end to fit DISCORD_LIMIT; {} when nothing. Nobody is pinged and links get no preview.
    """
    url = f"https://github.com/{repository}/commit/"
    lines = [LINE.format(login=entry["login"], card=entry["card"], issues="; ".join(entry["told"]).rstrip("."),
                         short=(entry["commit"] or data["after"])[:8], url=url + (entry["commit"] or data["after"])[:URL_SHA])
             for entry in data["cards"] if entry["told"]]
    if not lines:
        return {}

    def assemble(kept: int) -> str:
        rest = len(lines) - kept
        more = [f"... and {plural(rest, 'more card')}: [details]({url}{data['after']})"] if rest else []
        return "\n".join(lines[:kept] + more)

    kept = len(lines)
    content = assemble(kept)
    while len(content) > DISCORD_LIMIT and kept > 0:
        kept -= 1
        content = assemble(kept)
    return {"content": content, "allowed_mentions": {"parse": []}, "flags": SUPPRESS_EMBEDS}


# ========================
# check
# ========================
def font_glyphs(data: bytes) -> dict:
    """{glyph name: (advance in font units, drawing)} of a font file's bytes: the drawing its SVG document's digest with
    glyph ids as names (ids shift when icons are added), else its glyf outline's."""
    import uharfbuzz as hb
    face = hb.Face(hb.Blob(data))
    font = hb.Font(face)
    names = [font.get_glyph_name(gid) or str(gid) for gid in range(face.glyph_count)]
    drawn = {**_outlines(face), **_svg_drawings(face, names)}
    return {name: (font.get_glyph_h_advance(gid), drawn.get(gid, "")) for gid, name in enumerate(names)}


def _svg_drawings(face, names: list) -> dict:
    """{glyph id: digest of its SVG document, glyph ids named}; {} without an SVG table."""
    raw = face.reference_table("SVG ").data
    if len(raw) < 10:
        return {}
    index, out = int.from_bytes(raw[2:6], "big"), {}
    named = lambda m: b'id="' + (names[int(m.group(1))].encode() if int(m.group(1)) < len(names) else m.group(1)) + b'"'
    for i in range(int.from_bytes(raw[index:index + 2], "big")):
        first, last, offset, length = unpack(">HHLL", raw[index + 2 + 12 * i:index + 14 + 12 * i])
        doc = raw[index + offset:index + offset + length]
        doc = gzip_decompress(doc) if doc[:2] == b"\x1f\x8b" else doc
        digest = blake2b(sub(rb'id="glyph(\d+)"', named, doc), digest_size=8).hexdigest()
        out.update(dict.fromkeys(range(first, last + 1), digest))
    return out


def _outlines(face) -> dict:
    """{glyph id: digest of its glyf outline}; {} without a glyf table."""
    glyf, loca, head = (face.reference_table(tag).data for tag in ("glyf", "loca", "head"))
    count, wide = face.glyph_count + 1, len(head) > 51 and head[50:52] == b"\0\1"
    if not glyf or len(loca) < count * (4 if wide else 2):
        return {}
    offs = unpack(f">{count}{'L' if wide else 'H'}", loca[:count * (4 if wide else 2)])
    step = 1 if wide else 2
    return {gid: blake2b(glyf[offs[gid] * step:offs[gid + 1] * step], digest_size=8).hexdigest() for gid in range(count - 1)}


def _file_blob(path: str, sha: str) -> Optional[bytes]:
    """path's bytes when they are blob sha, else None."""
    try:
        with open(path, "rb") as fh:
            data = fh.read()
    except OSError:
        return None
    return data if blob_sha(data) == sha else None


def _font_blob(source: Source, sha: str, own: bool = True) -> bytes:
    """An icon font blob: EARLIER_ENV's file, with own the clone's file (the Action's cached MDW/Fonts, once in:
    fonts_ready), when it is that blob; else source.blob."""
    folder = environ.get(EARLIER_ENV, "")
    data = _file_blob(join(folder, sha), sha) if folder else None
    if data is None and own:
        fonts_ready()
        data = _file_blob(join(source.clone, ICON_FONT_FILE), sha)
    return source.blob(sha) if data is None else data


def _alike(older: dict, newer: dict, names: dict) -> bool:
    """Whether newer has every glyph of older that names (the current font's) has, laid out and drawn alike."""
    return all(newer.get(name) == glyph for name, glyph in older.items() if name in names)


def earlier_fonts(source: Source, commit: str, tree: dict, folder: str) -> list:
    """
    The icon font's earlier eras before the one commit's tree holds, from its history, newest first, written into folder
    (named .earlier: a FontMap over folder never takes one for the current font): each era's newest version, an era
    being the versions a newer one lays out and draws alike (_alike: an icon added keeps the era, an advance or a drawing
    changed ends it), none the current font's. Every era of the EARLIER_WALK newest versions, each read once a process
    (_glyphs), PREFETCH_WORKERS at a time. The advances and drawings a raster drawn before may show (badge widths).
    Raises Transient, the API's errors or Refused("blob").
    """
    now = tree.get(ICON_FONT_FILE, ("", ""))[1]
    if not now:
        return []
    if now not in _glyphs:
        _glyphs[now] = font_glyphs(_font_blob(source, now))
    current, kept, seen, files, walk = _glyphs[now], [], {now}, [], []
    for line in source.git("log", "--format=%H", commit, "--", ICON_FONT_FILE).split():
        if len(seen) > EARLIER_WALK:
            break
        try:
            sha = source.git("rev-parse", f"{line}:{ICON_FONT_FILE}").strip()
        except Transient:                       # deleted there
            continue
        if sha not in seen:
            seen.add(sha)
            walk.append(sha)
    with ThreadPoolExecutor(PREFETCH_WORKERS) as pool:
        reads = pool.map(partial(_font_blob, source, own=False), [sha for sha in walk if sha not in _glyphs])
        for sha in walk:
            data = None
            if sha not in _glyphs:
                data = next(reads)
                _glyphs[sha] = font_glyphs(data)
            if any(_alike(_glyphs[sha], newer, current) for newer in [current] + [_glyphs[k] for k in kept]):
                continue
            kept.append(sha)
            files.append(join(folder, f"stoogedings-{sha[:12]}.earlier"))
            with open(files[-1], "wb") as fh:
                fh.write(_font_blob(source, sha, False) if data is None else data)
    return files


def pool_size(jobs: int) -> int:
    """Processes for a pooled check: every core on a GitHub runner (K-2; revert: the local rule there too), else half
    the cores, at most POOL_MAX."""
    cores = cpu_count() or 2
    count = cores if environ.get("GITHUB_ACTIONS") == "true" else min(cores // 2, POOL_MAX)
    return max(1, min(count, jobs))


def check(clone: str, after: str, before: str = "", repository: str = "", author: str = NOBODY, cards: Optional[set] = None,
          whole_tree: bool = False, mode: str = "dry", out: str = "guards", workers: int = 0, convert: str = "later") -> dict:
    """
    Both guards on a push (or a whole tree); writes report.md, discord.json and plan.json (with the artifacts) to out.
    The 11-07 pass's conversions and linked-file housekeeping (convert_plan, G1-G4, silent): in the check (convert
    "now"), left to the normalise step after Decided ("later", card-guards.yml's: summary.json's normalise counts them;
    with none, the housekeeping runs here), or none ("off"; a whole tree never).

    Returns:
        plan.json's data (with "markdown" and "discord")
    """
    makedirs(out, exist_ok=True)
    wait_fonts()
    source = Source(clone, repository)
    errors: list = []
    if whole_tree:
        head = source.commit(after)
        if not head:
            raise Transient(f"the clone has no commit {after[:8]}")
        tree, base, notes, skipped = source.tree(head), "", ["the whole tree: standards and PNG-vs-PSD only"], {}
        jobs = whole_tree_jobs(tree, author, cards)
    else:
        base, head, notes = push_range(source, before, after)
        jobs, skipped, more, tree = plan_jobs(source, base, head, author, cards)
        notes += more
    source.checked = tree
    least = 2 if environ.get("GITHUB_ACTIONS") == "true" else POOL_MIN
    pooled = len(jobs) >= least and workers != 1
    early = workers != 1 and len(jobs) < (cpu_count() or 1)       # a core to spare: the pushed PNGs beside (_overlap)
    with _overlap(source, jobs if early and not pooled else ()), TemporaryDirectory(prefix="guard-fonts-") as fonts:
        context, vocab = build_context(source, tree, errors) if jobs else (standards.Context(), None)
        source.fonts, earlier = fonts, ()       # R8's fetched font files, the earlier icon fonts
        if jobs and repository:                 # badge widths: read once for every worker
            try:
                earlier = tuple(earlier_fonts(source, head, tree, fonts))
            except (Transient, RequestException, Refused):    # best effort: Photoshop's layouts alone
                print(format_exc(), file=stderr)
        if pooled:
            count = workers or pool_size(len(jobs))
            with ProcessPoolExecutor(count, initializer=_init_worker,
                                     initargs=(clone, repository, context, vocab, tree, fonts, earlier)) as pool:
                results = list(pool.map(_pooled, [(job, out, mode, early) for job in jobs], chunksize=1))
        else:
            parser = vocab.parser() if vocab is not None else None
            results = [run_job(source, job, context, parser, out, mode, earlier) for job in jobs]
        data = plan(results, repository, base, head, mode)
        follow_dups(data, source, out)
        changed = [] if whole_tree else source.diff(base, head)
        pushed = [c for c in changed if c.status in "AM"]
        data["pushed_paths"] = [c.path for c in pushed]
        if not whole_tree:                      # names a file name can't hold (owner 2026-11-11), silent
            data["renames"] = {old: new for old, new in unsafe_renames(tree, data["pushed_paths"], [
                w["path"] for e in data["cards"] for w in e["writes"]]).items()
                if not cards or CardAuthors.key(card_name(old)) in cards}
        data["deleted_linked"] = {c.path: c.old for c in changed if c.status == "D" and _restorable(c.path)}   # G4b
        data["fonts"] = font_fixes(source, [] if cards else [c for c in pushed if c.path not in skipped], author, head, out)
        if whole_tree or convert == "off":
            for entry in data["cards"]:
                entry["convert"] = []
        elif convert == "now" or not any(entry["convert"] for entry in data["cards"]):
            convert_plan(data, source, out, workers=workers)    # later, nothing to convert: G2-G4 counted by Decided
    errors += [f"{r.job.psd}: {r.error}" for r in results if r.error]
    data["report"] = {"notes": notes, "skipped": sorted(skipped), "errors": errors}   # landed_markdown re-renders from these
    data["markdown"] = markdown(data, notes, skipped, errors)
    data["discord"] = discord_payload(data, repository or CARD_REPO)
    with open(join(out, "report.md"), "w", encoding="utf-8") as fh:
        fh.write(data["markdown"])
    with open(join(out, "discord.json"), "w", encoding="utf-8") as fh:
        dump(data["discord"], fh)
    _write_plan(data, out)
    return data


def _write_plan(data: dict, out: str) -> None:
    """plan.json and summary.json (data["summary"] set)."""
    with open(join(out, "plan.json"), "w", encoding="utf-8") as fh:
        dump({k: v for k, v in data.items() if k not in ("markdown", "discord", "summary")}, fh, indent=1)
    data["summary"] = summary(data, data.get("report", {}).get("errors", []))
    with open(join(out, "summary.json"), "w", encoding="utf-8") as fh:
        dump(data["summary"], fh)


def convert(plan_path: str, clone: str, repository: str = "", minutes: float = CONVERT_MINUTES, workers: int = 0,
            deadline: float = 0.0) -> dict:
    """
    The normalise step (card-guards.yml, after Decided): a `check --convert later` plan's conversions and linked-file
    housekeeping (convert_plan; no PSD starts past deadline, epoch seconds, else minutes from now), plan.json and
    summary.json written again beside it; report.md and discord.json stand (it is silent). Raises Transient when the
    clone lacks the push.
    """
    out = dirname(plan_path)
    with open(plan_path, encoding="utf-8") as fh:
        data = load(fh)
    source = Source(clone, repository or data.get("repo", ""))
    head = source.commit(data["after"])
    if not head:
        raise Transient(f"the clone has no commit {data['after'][:8]}")
    source.checked = source.tree(head)
    with TemporaryDirectory(prefix="guard-fonts-") as fonts:
        source.fonts = fonts
        convert_plan(data, source, out, minutes, workers, deadline)
    _write_plan(data, out)
    return data


def _for_comment(entry: dict) -> bool:
    """A card the commit comment has something on: told, written, an error, or comment-only lines (R10, a non-Discord
    note, a told issue, a fix or a refused one); never a composer fix alone."""
    return bool(entry["told"] or _shown(entry) or entry["error"] or entry["notes"] or entry["r10"].get("ok") is False
                or entry["fixed"] or entry["fix_refused"] or any(issue["told"] for issue in entry["issues"]))


def summary(data: dict, errors: list) -> dict:
    """summary.json, the workflow's switches: cards checked; cards with something told (Land, the post), for the comment
    (the Comment step: a context error alone, a standing manifest or chart one say, comments on no push) and with an error
    (a red run); writes planned; errors (the cards' and the context's)."""
    cards = data["cards"]
    pending = sum(1 for c in cards if c.get("convert") and "converted" not in c)    # --convert later: the normalise step's
    return {"after": data["after"], "cards": len(cards), "told": sum(1 for c in cards if c["told"]),
            "comment": sum(1 for c in cards if _for_comment(c)),
            "writes": sum(len(c["writes"]) for c in cards) + pending
            + sum(len((data.get(key) or {}).get("writes", [])) for key in ("linked", "fonts")) + len(data.get("renames") or {}),
            "normalise": pending, "errors": len(errors), "card_errors": sum(1 for c in cards if c["error"])}


# ========================
# land
# ========================
def landed_markdown(data: dict, landed: dict) -> str:
    """The commit comment after land: what landed, and the cards left out (their PNG and fixes were not committed)."""
    report = data.get("report", {})
    return markdown(data, report.get("notes", []), report.get("skipped", []), report.get("errors", []), landed)


def message(landed: list, after: str, renamed: Optional[dict] = None) -> str:
    """The bot commit's message: subject with the counts, one line per file, then the trailers: one Recompressed per PNG
    only recompressed, a Renamed per renamed file (renamed: old path -> new), the Guarded-push. The one place a composer
    fix is said (owner, 2026-10-25)."""
    renamed = renamed or {}
    pngs = sum(1 for entry in landed for w in entry["writes"] if w["kind"] == "png")
    psds = sum(1 for entry in landed for w in entry["writes"] if w["kind"] == "psd" and not w.get("silent"))
    composer = sum(1 for entry in landed for w in entry["writes"] if w["kind"] == "psd" and w.get("composer"))
    folders = sum(1 for entry in landed for w in entry["writes"] if w["kind"] == "psd" and w.get("folders"))
    smaller = sum(1 for entry in landed for w in entry["writes"] if w["kind"] == "recompress")
    added = len({w["path"] for entry in landed for w in entry["writes"] if w["kind"] == "file"})    # users share one
    deleted = len({w["path"] for entry in landed for w in entry["writes"] if w["kind"] == "delete"})
    fonts = sum(1 for entry in landed for w in entry["writes"] if w["kind"] == "font")
    pixels = [kind for entry in landed for w in entry["writes"] if w["kind"] == "psd" for kind in w.get("pixels", ())]
    silent = [f"{standards.PIXEL_LINES.get(kind, kind)} on {plural(pixels.count(kind), 'card')}"
              for kind in dict.fromkeys(pixels)]          # the pixel fixes: said here alone, as the composer
    counts = ", ".join(part for part in (plural(pngs, "PNG") + " re-exported" if pngs else "",
                                         plural(psds, "PSD") + " fixed" if psds else "",
                                         "Single-line Composer on " + plural(composer, "card") if composer else "",
                                         f"{FOLDERS} on " + plural(folders, "card") if folders else "",
                                         *silent,
                                         plural(smaller, "PNG") + " recompressed" if smaller else "",
                                         plural(added, "linked file") + " added" if added else "",
                                         plural(deleted, "linked file") + " deleted" if deleted else "",
                                         plural(fonts, "font") + " repaired" if fonts else "",
                                         plural(len(renamed), "file") + " renamed" if renamed else "") if part)
    lines = [f"{renamed.get(w['path'], w['path'])}: {w['line']} ({entry['login']}, {(entry['commit'] or after)[:8]})"
             for entry in landed for w in entry["writes"]]
    lines += [f"{old}: renamed to {basename(new)} (no {' '.join(NOT_IN_FILE_NAMES)} in a file name)" for old, new in sorted(renamed.items())]
    trailers = sorted(f"{RECOMPRESSED}: {renamed.get(w['path'], w['path'])}" for entry in landed for w in entry["writes"]
                      if w["kind"] == "recompress")
    trailers += rename_trailers(renamed)
    trailers.append(f"{TRAILER}: {after}")
    return f"{SUBJECT}: {counts} (push {after[:8]})\n\n" + "\n".join(lines) + "\n\n" + "\n".join(trailers) + "\n"


def _rests_on(entry: dict, write: dict) -> list:
    """The paths a write rests on: a PSD fix (0 pixels, or an R8 redraw: see _drawn) on its PSD alone, a recompressed PNG
    on itself, a PNG on the whole card (PSD, PNG, numbered) but the numbered PNGs the push brought (only their
    recompressions rest on them)."""
    if write["kind"] == "psd":
        return [write["path"]]
    if write["kind"] in ("file", "delete", "font"):     # a linked file with the card whose conversion makes or frees it; a font
        return [write["path"]] + ([entry["psd"]] if entry.get("psd") else [])
    pushed = set(entry.get("pushed", ()))
    return [write["path"]] if write["kind"] == "recompress" else sorted(p for p in entry["expect"] if p not in pushed)


class Head:
    """A branch head's blobs, looked up in any letter case (a teammate's foo.png stands for the bot's Foo.png, L12)."""

    def __init__(self, tip: Snapshot):
        self.tree = tip.tree
        self.lower = {path.lower(): sha for path, (_, sha) in tip.tree.items()}

    def __call__(self, path: str) -> str:
        return self.tree[path][1] if path in self.tree else self.lower.get(path.lower(), "")

    @property
    def holders(self) -> str:
        if not hasattr(self, "_holders"):
            self._holders = _holders((path, sha) for path, (_, sha) in self.tree.items())
        return self._holders


def _ours(entry: dict) -> dict:
    """{path: {blob}} of a card's planned writes (a redraw's plain too)."""
    ours: dict = {}
    for write in entry.get("writes", []):
        ours.setdefault(write["path"], set()).update(w["sha"] for w in (write, write.get("plain")) if w)
    return ours


def _holds(entry: dict, path: str, blob: str) -> bool:
    """The head still holds what a card's check rests on at path: the blob as checked, or the bot's own write of it."""
    return blob == entry.get("expect", {}).get(path, "") or (bool(blob) and blob in _ours(entry).get(path, ()))


def _drawn(entry: dict, here: Head) -> list:
    """A card's writes as they may land: when a PNG re-exported for R8's redraw cannot, the PSD lands switched but not
    redrawn (its plain) and no other such PNG lands, so the PSD and its PNGs never disagree."""
    writes = entry["writes"]
    if not any(w.get("plain") for w in writes):
        return writes
    redraw = [w for w in writes if w["kind"] == "png" and w.get("composer")]
    if not any(_superseded(entry, w, here) for w in redraw):
        return writes
    return [{**w, "pixels": [], **w["plain"], "plain": None} if w.get("plain") else w for w in writes if w not in redraw]


def _superseded(entry: dict, write: dict, here: Head) -> str:
    if write["kind"] == "delete" and entry.get("holders") and here(write["path"]) and entry["holders"] != here.holders:
        return "superseded: a PSD changed on the branch (it may link the file)"     # G2-G3 decided on after's PSDs
    moved = [p for p in _rests_on(entry, write) if not _holds(entry, p, here(p))]
    return f"superseded: {moved[0]} changed on the branch" if moved else ""


def _bot_changed(repository: str, local: str, head: str, path: str) -> bool:
    """The last commit on the branch that changed path is a bot's (no run of ours follows it); False when unknown."""
    try:
        if local:
            name, _, email = git(local, "log", "-1", "--format=%an%x1f%ae", head, "--", path).decode().strip().partition("\x1f")
        else:
            commits = api(repository, "GET", "commits", params={"sha": head, "path": path, "per_page": 1}).json()
            author = commits[0]["commit"]["author"] if commits else {}
            name, email = author.get("name", ""), author.get("email", "")
    except Exception:                           # unknown: a person's, as before (that push's run tells)
        return False
    return bool(name or email) and Writer("", name, email, "").bot


def after_window(data: dict, landed: dict, tip: Optional[Snapshot], bot_moved: frozenset = frozenset()) -> tuple:
    """
    What the one Discord post says after the quiet window (section 7.3), per card PSD: the check's lines, less a card whose
    PSD a person changed again (the later push's own check tells; a bot's change starts no run, so the lines stay) and a
    PNG refusal, or in report mode a PNG the bot would make, whose PNG was pushed since; plus the writes that did not land
    and were not replaced by the author (the bot's commit failed, a later push touched the card). A write the branch
    already holds (a re-run) is not told, nor any write when a re-run could not read the branch (an earlier attempt may
    have landed it). A numbered PNG's refusal (G8) is withdrawn as the main PNG's is.

    Args:
        bot_moved: The PSDs whose last change on the branch is a bot commit's

    Returns:
        ({psd: [Discord lines]}, {psd: [comment notes]})
    """
    told, after = {}, {}
    dropped = {path: why for _, why, path in landed.get("dropped", [])}
    here = Head(tip) if tip is not None else None
    for entry in data["cards"]:
        psd, expect, lines, notes = entry["psd"], entry.get("expect", {}), list(entry.get("told", [])), []
        by_bot = psd in bot_moved
        if here and entry["case"] != "deleted" and psd in expect and not _holds(entry, psd, here(psd)) and not by_bot:
            told[psd], after[psd] = [], ["the PSD changed again on the branch: that push's own check tells"]
            continue
        if by_bot:
            notes.append("a bot commit changed the PSD since: the push's lines stand")
        export, png = entry.get("export", {}), entry.get("png", "")
        refused = [(export.get("why"), png)] if export.get("outcome") == "refused" and png else []
        for why, path in refused + [tuple(item) for item in export.get("numbered_refused", [])]:
            if here and here(path) != expect.get(path, ""):
                lines = [line for line in lines if line != why]
                notes.append(f"{basename(path)} was pushed since: the refusal is withdrawn")
        if data.get("mode") == "report":
            for write in entry.get("writes", []):
                if write["kind"] == "png" and here and not _holds(entry, write["path"], here(write["path"])):
                    lines = [line for line in lines if line != would_make(entry["case"], png, write)]
                    notes.append(f"{basename(write['path'])} was pushed since: not told")
        already = _already(landed, entry["card"])
        for write in [] if landed.get("dry") or landed.get("unread") else entry.get("writes", []):
            why = "the bot's commit failed" if landed.get("error") else dropped.get(write["path"], "")
            if not why or write["path"] in already or write["kind"] == "recompress" or write.get("silent"):
                continue                        # landed (now or by an earlier run); a recompression or composer never told
            if here and not _holds(entry, write["path"], here(write["path"])) and not (by_bot and write["path"] == psd):
                continue                        # the author replaced that file: nothing to tell
            why = why.replace("superseded: ", "")
            if write["kind"] == "psd":
                lines.append(f"{write.get('shown', write['line'])} (the bot could not fix it: {why})")
            elif write["path"] == png:
                lines.append(f"PNG not made: {why}; export it in Photoshop")
            else:
                lines.append(f"{basename(write['path'])} not updated: {why}; re-export it in Photoshop")
        for write in entry.get("writes", []):   # R1's tags set as badges by the bot: not told once that write stands
            if write.get("settles") and _fix_stands(entry, write, landed, here):
                lines = [line for line in lines if line not in write["settles"]]
        told[psd], after[psd] = list(dict.fromkeys(lines)), notes
    return told, after


def _fix_stands(entry: dict, write: dict, landed: dict, here: Optional[Head]) -> bool:
    """A PSD write stands with its pixel fixes: on the branch already, or committed now with its redraw's PNGs (when one
    of them is dropped the PSD lands as its plain, without them)."""
    if here is not None and here(write["path"]) == write["sha"]:
        return True
    if not landed.get("sha") or [entry["card"], write["path"]] not in landed.get("included", []):
        return False
    pngs = {w["path"] for w in entry["writes"] if w["kind"] == "png" and w.get("composer")}
    return not any(card == entry["card"] and path in pngs for card, _, path in landed.get("dropped", []))


def _read_files(out_dir: str, entries: list) -> dict:
    """{file: bytes} of every planned write (a redraw's plain too)."""
    files = {}
    for entry in entries:
        for write in entry["writes"]:
            for one in (write, write.get("plain")):
                if one and one.get("file"):
                    with open(join(out_dir, one["file"]), "rb") as fh:
                        files[one["file"]] = fh.read()
                    if blob_sha(files[one["file"]]) != one["sha"]:
                        raise Transient(f"{one['file']} does not hash to the planned {one['sha'][:8]}")
    return files


def events_of(entry: dict, write: dict, drawn: list) -> list:
    """
    A landed write's changelog events: guard-png for a PNG made, re-exported or recompressed, guard-fix for a PSD fix the
    comment may say or a pixel fix (card_standards.PIXEL_LINES); the Single-line Composer and its rewrap alone, none
    (owner, 2026-10-25). drawn: the card's writes as they landed (_drawn: a redraw's plain has no pixel fix).
    """
    base = {"card": entry["card"], "path": write["path"]}
    if write["kind"] in ("file", "delete", "font") or write.get("dup"):     # linked files, fonts, duplicates: silent
        return []
    if write["kind"] == "recompress":
        return [{"kind": "guard-png", **base, "line": "recompressed losslessly"}]
    psd = next((w for w in drawn if w["kind"] == "psd"), {})
    pixels = [standards.PIXEL_LINES.get(kind, kind) for kind in psd.get("pixels", ()) if kind not in QUIET_PIXELS]
    if write["kind"] == "png":
        if write.get("silent"):                 # a redraw's PNG: told only for the pixel fixes it carries
            return [{"kind": "guard-png", **base, "line": "re-exported for " + " and ".join(pixels)}] if pixels else []
        return [{"kind": "guard-png", **base, "line": write["line"].split(REDRAW, 1)[0]}]
    said = [part for part in (write.get("shown") or "").split("; ") if part] + pixels
    return [{"kind": "guard-fix", **base, "line": "; ".join(said)}] if said else []


def _renames_at(renames: dict, tip: Snapshot, payload: dict) -> dict:
    """The planned renames still to make at the head: the old file there (or written by this commit), the new name free
    (any case). A rerun after a landing finds the new names taken: none."""
    lower = {path.lower() for path in tip.tree}
    return {old: new for old, new in renames.items()
            if (old in tip.tree or payload.get(old) is not None) and new.lower() not in lower}


def _renamed(payload: dict, renames: dict, tip: Snapshot, uploaded: set) -> dict:
    """payload with the renames made (each file's bytes, this commit's write or the head's, under its new name; the old
    path deleted) and the files keyed by the cards following them (card_bake.follow_renames, as at the head)."""
    out = dict(payload)
    for old, new in renames.items():
        data = out.pop(old, None)
        if data is None:
            data = tip.fetch(tip.tree[old][1])
            uploaded.add(tip.tree[old][1])      # the head has the blob: no upload
        out[new] = data
        if old in tip.tree:
            out[old] = None
    texts = {path: (out[path] if out.get(path) is not None else tip.fetch(tip.tree[path][1])).decode("utf-8")
             for path in RENAME_SIDES if path in tip.tree}
    out.update({path: text.encode("utf-8") for path, text in follow_renames(texts, renames).items()})
    return out


def land(plan_path: str, repository: str = "", dry_run: bool = False, local: str = "", rerun: bool = False) -> dict:
    """
    Commit a plan's writes once (G9) and settle what the one Discord post says (after_window). A write is left out when
    its card no longer holds the blobs it rests on (a PSD fix rests on its PSD alone, a PNG on the whole card; any letter
    case; an R8 redraw lands plain without its PNGs); a write the branch already holds (a re-run after a landing) is not
    written again. The rest is one commit on the branch head (retried on a moved head, COMMIT_TRIES times); --local
    commits in that clone instead (nothing pushed). dry_run (report mode) commits nothing. Never raises once the plan is
    read: any failure is the error.
    rerun (a re-run or a dispatch: an earlier run may have landed the writes): when the branch could not be read, no
    write is told as not made ("unread").

    Returns:
        {"sha": the commit or "", "landed": [card], "included": [[card, path]] committed, "dropped": [[card, why, path]],
         "already": [[card, path]], "events": the included writes' changelog events (events_of),
         "error": why nothing was committed, "told": {psd: [Discord lines]}, "after": {psd: [comment notes]}, "dry": dry_run,
         "unread": a rerun that could not read the branch, "renamed": {old path: new} the commit renames}
    """
    out_dir = dirname(plan_path)
    with open(plan_path, encoding="utf-8") as fh:
        data = load(fh)
    repository = repository or data.get("repo") or ("" if local else CARD_REPO)
    entries = [entry for entry in data["cards"] + [data.get("linked") or {}, data.get("fonts") or {}]
               if entry.get("writes")]  # + G2-G4's, the fonts'
    report = {"sha": "", "landed": [], "included": [], "dropped": [], "already": [], "error": "", "told": {}, "after": {},
              "dry": dry_run, "unread": False, "events": [], "renamed": {}}
    if not entries and not any(entry.get("told") for entry in data["cards"]) and not data.get("renames"):
        return report
    tip, bot_moved = None, frozenset()
    try:
        files = _read_files(out_dir, entries)
        uploaded: set = set()
        for _ in range(COMMIT_TRIES if not local else 1):
            tip = local_snapshot(repository, local) if local else snapshot_at(repository, ref_head(repository))
            here = Head(tip)
            included, report["dropped"], report["already"], told = [], [], [], []
            for entry in entries:
                drawn = _drawn(entry, here)
                for write in entry["writes"]:
                    if write not in drawn and write["kind"] == "png":
                        why = _superseded(entry, write, here) or "superseded: the PSD landed without the redraw"
                        report["dropped"].append([entry["card"], why, write["path"]])
                for write in drawn:
                    why = _superseded(entry, write, here)
                    if why:
                        report["dropped"].append([entry["card"], why, write["path"]])
                    elif here(write["path"]) == (write["sha"] or ""):      # a deletion: already gone
                        report["already"].append([entry["card"], write["path"]])
                    else:
                        included.append((entry, write))
                        told += events_of(entry, write, drawn)
            cards = list(dict.fromkeys(entry["card"] for entry, _ in included))
            report["landed"] = [card for card in cards if card]                # the linked files' entry is no card
            report["included"] = [[entry["card"], write["path"]] for entry, write in included]
            payload = {write["path"]: files[write["file"]] if write.get("file") else None for _, write in included}
            renames = _renames_at(data.get("renames") or {}, tip, payload)
            names = {card_name(old): card_name(new) for old, new in renames.items()}
            report["events"] = [{**e, "path": renames.get(e["path"], e["path"]), "card": names.get(e["card"], e["card"])}
                                for e in told]
            report["renamed"] = renames
            if not (included or renames) or dry_run:
                break
            if renames:
                payload = _renamed(payload, renames, tip, uploaded)
            text = message([{**entry, "writes": [w for e, w in included if e is entry]} for entry in entries
                            if entry["card"] in cards], data["after"], renames)
            if local:
                report["sha"] = local_commit(local, tip.head, payload, text)
                break
            modes = {path: tip.tree[path][0] for path in payload if path in tip.tree}
            by_git = len(payload) > GIT_LAND or sum(len(data) for data in payload.values() if data) > GIT_LAND_BYTES
            if by_git:
                try:
                    sha = git_commit(repository, tip.head, payload, text)        # no upload cap
                except Transient as error:          # git broke: the API lands it
                    print(f"git land failed, by the API: {error}", file=stderr)
                    by_git = False
            if not by_git:
                sha = commit_files(repository, tip.head, tip.info["tree"]["sha"], modes, payload, text, uploaded)
            if sha:
                report["sha"] = sha
                break
        else:
            raise Transient(f"the branch kept moving: not committed after {COMMIT_TRIES} tries")
    except Exception as error:                  # the API, git, a bad file: told as not made, never a lost post
        if not isinstance(error, (Transient, Refused)):
            print(format_exc(), file=stderr)
        why = str(error) if isinstance(error, (Transient, Refused)) else f"{type(error).__name__}: {error}"
        report.update(sha="", landed=[], included=[], events=[], renamed={}, error=why or type(error).__name__,
                      unread=rerun and tip is None)
    if tip is not None:
        here = Head(tip)
        bot_moved = frozenset(entry["psd"] for entry in data["cards"]
                              if entry["case"] != "deleted" and entry["psd"] in entry.get("expect", {})
                              and not _holds(entry, entry["psd"], here(entry["psd"]))
                              and _bot_changed(repository, local, tip.head, entry["psd"]))
    report["told"], report["after"] = after_window(data, report, tip, bot_moved)
    landed = {tuple(pair) for pair in report["included"]} if report["sha"] else set()
    report["records"] = [entry["records"] for entry in data["cards"]
                         if entry.get("records") and (entry["card"], entry["psd"]) in landed]
    return report


def finish(plan_path: str, landed: dict) -> dict:
    """land's files beside the plan: landed.json, report-landed.md (the comment's new body), discord-final.json (the
    one post, as after_window settled it) and, for a landed commit, guard-events.json ({"schema", "commit", "push",
    "events"}) and records.json (its PSD writes' records, a list). Returns the post."""
    out_dir = dirname(plan_path)
    with open(plan_path, encoding="utf-8") as fh:
        data = load(fh)
    with open(join(out_dir, "landed.json"), "w", encoding="utf-8") as fh:
        dump(landed, fh)
    if landed.get("sha"):
        with open(join(out_dir, GUARD_EVENTS), "w", encoding="utf-8") as fh:
            dump({"schema": 1, "commit": landed["sha"], "push": data.get("after"), "events": landed.get("events", [])}, fh)
    if landed.get("sha") and landed.get("records"):
        with open(join(out_dir, GUARD_RECORDS), "w", encoding="utf-8") as fh:
            dump(landed["records"], fh)
    with open(join(out_dir, "report-landed.md"), "w", encoding="utf-8") as fh:
        fh.write(landed_markdown(data, landed))
    told = landed.get("told", {})
    final = {**data, "cards": [dict(entry, told=told.get(entry["psd"], entry.get("told", []))) for entry in data["cards"]]}
    post = discord_payload(final, data.get("repo") or CARD_REPO)
    with open(join(out_dir, "discord-final.json"), "w", encoding="utf-8") as fh:
        dump(post, fh)
    return post


# ========================
# Command line
# ========================
def main(args: Optional[list] = None) -> int:
    """check: the report on stdout (exit 0 whatever it finds, 1 when it could not check); convert: one line (exit 1 when
    the clone lacks the push); land: one line per card, exit 1 when the commit failed (its files are written either
    way), exit 2 for a plan not checked with --mode on that would commit to the repo (neither --dry-run nor --local)."""
    cli = ArgumentParser(prog="python -m commands.card_guards",
                         description="The push guards: a forgotten PNG drawn from the PSD by the fork (the bake's render), the card "
                                     "standards checked (0-pixel fixes planned). check never writes to the card repo; land commits a plan once.")
    sub = cli.add_subparsers(dest="command", required=True)
    one = sub.add_parser("check", help="both guards on a push (or the whole tree)")
    one.add_argument("--clone", required=True, metavar="DIR", help="a clone of the card repo with the push's history (blobless is enough)")
    one.add_argument("--after", required=True, metavar="SHA", help="the push's head")
    one.add_argument("--before", default="", metavar="SHA", help="the push's previous head; empty: the first parent")
    one.add_argument("--repo", default="", metavar="OWNER/NAME", help="blobs the clone lacks and the commit logins by API")
    one.add_argument("--author", default=NOBODY, metavar="LOGIN",
                     help="the pusher, named when a login is unknown (a push's github.actor; a dispatch passes none)")
    one.add_argument("--cards", default="", metavar="NAMES", help="only these cards (comma separated, matched like Card Authors.txt)")
    one.add_argument("--whole-tree", action="store_true", help="every card PSD at --after: standards and PNG-vs-PSD")
    one.add_argument("--mode", default="dry", choices=MODES, type=str.lower,
                     help="dry, report or on (CARD_GUARDS, any case): the report's wording")
    one.add_argument("--workers", type=int, default=0, metavar="N", help="processes for 8 cards or more (0: every core on a runner, else half the cores, at most 4)")
    one.add_argument("--out", required=True, metavar="DIR", help="report.md, discord.json, plan.json and the artifacts")
    one.add_argument("--convert", default="later", choices=CONVERTS,
                     help="the pass's conversions and linked files: in the check (now), in the normalise step (later), none (off)")
    three = sub.add_parser("normalise", help="a check --convert later plan's conversions and linked files (silent)")
    three.add_argument("--out", required=True, metavar="DIR", help="check's --out (plan.json and summary.json rewritten)")
    three.add_argument("--clone", required=True, metavar="DIR", help="the clone the check read")
    three.add_argument("--repo", default="", metavar="OWNER/NAME", help="blobs the clone lacks by API (default: the plan's)")
    three.add_argument("--minutes", type=float, default=CONVERT_MINUTES, metavar="N", help="no PSD converted past this")
    three.add_argument("--deadline", type=float, default=0.0, metavar="EPOCH", help="no PSD converted past this time")
    three.add_argument("--workers", type=int, default=0, metavar="N", help="as check's")
    two = sub.add_parser("land", help="commit a plan's writes once")
    two.add_argument("--plan", required=True, metavar="FILE", help="check's plan.json (its files beside it)")
    two.add_argument("--repo", default="", metavar="OWNER/NAME", help="the card repo (default: the plan's)")
    two.add_argument("--dry-run", action="store_true", help="decide what would land, commit nothing (needed for a plan not checked with --mode on)")
    two.add_argument("--local", default="", metavar="DIR", help="commit in this clone (nothing pushed)")
    two.add_argument("--rerun", action="store_true",
                     help="a re-run or a dispatch: an earlier run may have landed the writes, so an unread branch tells none as not made")
    options = cli.parse_args(args)
    if options.command == "land" and not (options.dry_run or options.local):
        with open(options.plan, encoding="utf-8") as fh:
            planned = load(fh).get("mode", "dry")
        if planned != "on":                     # a dry or report check must never reach the branch
            cli.error(f"the plan was checked with --mode {planned}: land commits only an `on` plan (pass --dry-run, or --local DIR)")
    token_from_env()
    if options.command == "check":
        cards = {CardAuthors.key(n) for n in options.cards.split(",") if n.strip()} or None
        try:
            data = check(options.clone, options.after, options.before, options.repo, options.author, cards, options.whole_tree,
                         options.mode, options.out, options.workers, options.convert)
        except (Transient, Refused) as error:   # the clone or the API failed: one line; the run fails (card-guards.yml alerts)
            print(f"## Card guards\n\nNot checked: {error}\n")
            return 1
        print(data["markdown"], end="")
        return 0
    if options.command == "normalise":
        try:
            data = convert(join(options.out, "plan.json"), options.clone, options.repo, options.minutes, options.workers,
                           options.deadline)
        except (Transient, Refused) as error:
            print(f"Not converted: {error}")
            return 1
        done = [entry["psd"] for entry in data["cards"] if entry.get("converted")]
        print(f"Converted {plural(len(done), 'PSD')}; linked files: {plural(len((data.get('linked') or {}).get('writes', [])), 'write')}")
        return 0
    report = land(options.plan, options.repo, options.dry_run, options.local, options.rerun)
    finish(options.plan, report)
    if report["error"]:
        print(f"Not landed: {report['error']}")
    for card in report["landed"]:
        print(f"{card}: {'landed in `' + report['sha'][:8] + '`' if report['sha'] else 'would land'}")
    for card, why, path in report["dropped"]:
        print(f"{card}: {why} ({path})")
    for card, path in report["already"]:
        print(f"{card}: {ALREADY} ({path})")
    return 1 if report["error"] else 0


if __name__ == "__main__":
    raise SystemExit(main())


# ========================
# Module Exports
# ========================
__all__ = [
    'BOT_EMAILS',
    'LinkContext',
    'catalogue',
    'convert',
    'convert_plan',
    'convertible',
    'housekeeping',
    'links_index',
    'Change',
    'EARLIER_WALK',
    'FONTS_READY_ENV',
    'ICON_FONT_FILE',
    'Head',
    'Job',
    'LINE',
    'OLD_FONT',
    'RECOMPRESSED',
    'Result',
    'SPARSE_FILES',
    'Source',
    'TELL_PNG_MISMATCH',
    'Writer',
    'added',
    'build_context',
    'card_png',
    'changes',
    'check',
    'discord_payload',
    'earlier_fonts',
    'after_window',
    'events_of',
    'finish',
    'font_glyphs',
    'fonts_ready',
    'land',
    'landed_markdown',
    'main',
    'markdown',
    'message',
    'plan',
    'plan_jobs',
    'push_range',
    'run_job',
    'stale_left',
    'summary',
    'whole_tree_jobs',
]
