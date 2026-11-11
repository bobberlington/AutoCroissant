"""
The card data build, without Discord, pandas or config: the card repo's tree, the PSDs whose blob changed re-parsed
(one bad PSD fails its card, never the run), then the authors file, Card Edits.json, the dup_of PNGs and the archive
of deleted cards; db/links.json, every PSD's links and the linked files' blobs (gather_links).

The build (`python -m commands.card_data`, run by the card repo's Card sync Action): a local clone's tree and history,
db/ JSON through card_db, versions only on data changes (card_history), changelog events and the gates. CardBuild is the
parse and its REST blob source (the diagnostics use it alone); GitBuild adds the clone. Design:
docs/Campaigns/DATA_BUILD_ACTION.md (sections 7 and 13).
"""
from argparse import ArgumentParser
from collections import deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha1
from io import BytesIO
from json import dumps as json_dumps, loads as json_loads
from multiprocessing import get_context
from os import environ, fdopen, remove, walk
from os.path import abspath, basename, dirname, isfile, join as path_join, splitext
from pickle import dumps as pickle_dumps, loads as pickle_loads
from subprocess import CalledProcessError, run
from sys import version_info
from tempfile import mkstemp
from threading import Lock, active_count, local
from typing import Callable, Optional

from psd_tools import PSDImage
from requests.exceptions import RequestException

from global_config import (AUTHORS_PATH, CARD_BRANCH, CREATED_PATH, EDITS_PATH, ICON_MANIFEST_PATH, RULEBOOK_DIR, RULEBOOK_SRC, RULES_PATH,
                           STARS_DIR, TYPES_DIR)
from commands.card_edits import CardAuthors, CardDates, CardEdits, dup_pictures
from commands import github_api, png_changes, rulebook
from commands.card_models import CardInfo, StatsDatabase, card_name, duplicate_png, straight, straighten_card
from commands.github_api import HEADERS, api_url, blob_text, fetch_blob, get_with_retry, range_reads
from commands.psd_parser import PARSER_VERSION, CardValidator, IconVocabulary, PSDParser, picture_numbers, variants_of
from commands.psd_remote import RangeError, RemotePSD

# ========================
# Configuration
# ========================
UPDATE_RATE = 25
STRAIGHT_KEYS = ("name", "ability", "note")                     # parsed_values' printed text
DOWNLOAD_WORKERS = 4                                              # PSDs fetched and parsed at a time
RANGED_WORKERS = 8                                                # by range reads on a runner (a read mostly waits)
AHEAD = 2 * DOWNLOAD_WORKERS                                      # PSDs read and not yet kept, at most (memory)
RANGED_AHEAD = 2 * RANGED_WORKERS
FORK_WAIT = 300                                                   # s for a forked worker's PSD: past it, threads
RANGE_REFUSALS = 3                                                # PSDs refused range reads (HTTP error, no connection): off
                                                                  # trying them (github_api.RANGE_READS: the off switch)
EXCLUDE_FOLDERS = ["Markers", "MDW"]
PROGRESS = "{count} cards updated."                               # every UPDATE_RATE parses
REAL_DATES = 946684800.0                                          # 2000-01-01 UTC: older stamps are no commit found (year 1000)
LINKS_JSON = "links.json"                                         # db/: every PSD's links (Card sync's redraw, the guards' files,
                                                                  # the card checks' notes); not in the manifest: no bot reads it
LINK_KEYS = ("files", "pixels")                                  # links.json's keys beside the PSD paths
RECORDS = ("render_record.json", "spot_record.json")              # db/: the renders' look facts and the icons' earlier cores
                                                                  # (psd_edit.RENDER_RECORD, SPOT_RECORD); not in the manifest
STATUS_COLORS = "status_colors.json"                              # db/: each chart status's printed colour (the site's
                                                                  # mock-up); not in the manifest
LINKED_DIR = "MDW/Linked"                                         # the cards' linked pictures (the 11-07 pass)
LINK_HOLDERS = (".psd", ".psb")                                   # files whose links count: MDW/Linked's .psb documents link too
FETCH_CHUNK, FETCH_POOL = 8, 4                                    # blobs a git fetch, fetches at a time (the guards' Pushed blobs)


# ========================
# The build
# ========================
@dataclass
class Read:
    """One PSD's parse in a download worker, kept on the build in todo order (CardBuild._keep)."""
    card: Optional[CardInfo] = None             # None: read it whole (range reads failed), or error
    variants: Optional[list] = None             # its variants 2..N (CardInfo each), None without tags
    details: Optional[dict] = None
    links: Optional[dict] = None
    fetched: Optional[int] = None               # bytes by range reads
    error: Optional[Exception] = None           # what the parse raised


_BUILD: Optional["CardBuild"] = None            # a forked download worker's build (the parent's, copied at the fork)


def _forks() -> bool:
    """Download workers as forked processes (one parse per core, not one GIL): on a GitHub runner with no other thread
    alive; elsewhere threads (macOS forks crash in its system libraries)."""
    return environ.get("GITHUB_ACTIONS") == "true" and active_count() == 1


def _fork_init() -> None:
    """A forked worker's own requests Sessions: never the parent's sockets."""
    github_api._local = local()


def _fork_read(path: str, name: str, sha: str, given: dict) -> tuple:
    """_fetch_read in a forked worker with the parent's worker fields (CardBuild._given): (the Read or None, those fields
    after). An error that would not come back through pickle comes back as a RuntimeError of its text."""
    vars(_BUILD).update(given)
    read = _BUILD._fetch_read(path, name, sha)
    if read is not None and read.error is not None:
        try:
            pickle_loads(pickle_dumps(read.error))
        except Exception:
            read.error = RuntimeError(str(read.error))
    print(end="", flush=True)                   # its lines in the log as it goes (a pipe buffers them)
    return (read, {key: getattr(_BUILD, key) for key in given})


class CardBuild:
    """
    One build over the card repo: the parse, its details and the REST blob source (_blob, _repo_text, _download).
    GitBuild reads a local clone for the tree, the dates and the authors; the diagnostics use this class alone for its
    vocabulary and parser.

    Args:
        db: The database it updates in place
        report: One message (default print)
        progress: The running parse count (default: PROGRESS printed)
    """
    def __init__(self, db: StatsDatabase, report: Optional[Callable] = None, progress: Optional[Callable] = None):
        self.db = db
        self.report = report or print
        self.progress = progress or (lambda count: print(PROGRESS.format(count=count)))
        self.file_cache: dict = {}               # repo path -> (blob sha, text) of the small files read by REST
        self.parser: Optional[PSDParser] = None
        self.edits: Optional[CardEdits] = None
        self.edits_unread = False                # Card Edits.json is in the tree but was not read (5xx, bad JSON)
        self.changed = 0                         # cards changed without a re-parse (authors, edits, duplicates)
        self.failed = 0                          # failed downloads and PSDs: retried next run
        self.failed_new = 0                      # of those, PSDs of names not in stats (maybe a rename): no prune
        self.unreadable: list = []               # PSDs that raised while read: {path, blob, error}; their cards keep their data
        self.only: set = set()                   # card names to re-parse anyway (--cards)
        self.again: set = set()                  # paths to re-parse anyway (last run's pending.stale)
        self.lost: list = []                     # files not fetched (downloads, small files): the build fails red
        self.unlooked: list = []                 # PSDs whose date or author lookup raised
        self.repository = ""
        self.shas: dict = {}                     # repo path -> blob sha, from the tree
        self.sizes: dict = {}                    # repo path -> blob size (git's tree has none)
        self.raw_ref = ""                        # the commit for range reads ("": whole downloads this run)
        self.ranged: list = []                   # bytes fetched per PSD parsed from range reads
        self.refused: list = []                  # PSDs whose range read got a 4xx, 5xx or no connection (RANGE_REFUSALS)
        self.tree: Optional[dict] = None         # the tree
        self.details: dict = {}                  # blob sha -> psd_details record (db/psd_details.json)
        self.font_index = None                   # psd_details.font_index(): faces for the records' "supplied" flags, or None
        self.restored: list = []                 # archived cards whose PSD is back
        self.returned: list = []                 # archived names back under another PSD (path and blob): new cards
        self.renamed_from: dict = {}             # new name -> the listed name its PSD renames (parsed with that card)
        self.new_blobs: dict = {}                # blob sha -> name, for cards new or changed this run (a rename's new name)
        self.reparsed: dict = {}                 # name -> (its card before, path, blob) of each card parsed (the history)
        self.variant_cards: dict = {}            # name -> its PSD's variants 2..N parsed this run (CardInfo each)
        self.dup_pngs: dict = {}                 # PSD path -> its picture-only duplicates' PNGs (card_edits.dup_pictures)
        self.templates = None                    # psd_parts.Templates over this tree (the parts' template facts)
        self.links: dict = {}                    # PSD path -> its link record read this run (_linked; gather_links)
        self._repo_tree = None                   # psd_tools' RepoTree of the tree's files (links placed blind to case)
        self.rule_pages: set = set()             # the Photoshop rule pages' names (not cards; their credits stay)

    # ------------------------
    # The source: GitHub's REST blobs
    # ------------------------
    def _blob(self, sha: str, stream: bool = False):
        """A file's exact content by blob sha (raw.githubusercontent can lag behind the tree)."""
        return fetch_blob(self.repository, sha, stream=stream, timeout=120)

    def _repo_text(self, path: str) -> Optional[str]:
        """A small repo file's text; downloaded again only when its blob changed."""
        sha = self.shas.get(path)
        if sha is None:
            return None
        cached = self.file_cache.get(path)
        if not cached or cached[0] != sha:
            self.file_cache[path] = cached = (sha, blob_text(self.repository, sha, timeout=120))
        return cached[1]

    def _download(self, path: str, sha: str) -> Optional[str]:
        """A PSD into a temp file, or None. fetch_blob retries a 5xx; a stream that breaks gets one more try (a 15 MB
        stream breaks now and then)."""
        for last in (False, True):
            try:
                reply = self._blob(sha, stream=True)
            except RequestException as e:                       # tried 3 times already
                print(f"{path}: download failed ({e})")
                return None
            name, done = None, False
            try:
                with reply:
                    fd, name = mkstemp(suffix=".psd")
                    with fdopen(fd, "wb") as fh:
                        for chunk in reply.iter_content(1 << 20):
                            fh.write(chunk)
                done = True
                return name
            except (RequestException, OSError) as e:
                print(f"{path}: download failed ({e})" + ("" if last else "; trying once more"))
            finally:
                if not done and name is not None and isfile(name):      # partial download
                    remove(name)
        return None

    # ------------------------
    # The traversal
    # ------------------------
    def _psd_changed(self, name: str, path: str, sha: Optional[str]) -> bool:
        """Changed = new, moved, an older parser, a different blob, none stored or a details record of another VERSION."""
        card = self.db.stats.get(name)
        if card is None or card.path != path or getattr(card, "parser_version", None) != PARSER_VERSION:
            return True                           # an older parser's card is read again once (no history: same blob)
        if getattr(card, "blob_sha", None) != sha:
            return True
        record = self.details.get(sha)
        if not isinstance(record, dict):
            return False
        from commands.psd_details import VERSION  # lazy: traverse_remote loaded it
        return record.get("version") != VERSION   # gathered before a VERSION bump: read again once, like a parser bump

    def traverse_remote(self,
                        repository: str,
                        output_problematic: bool = True,
                        force_update: bool = False) -> tuple[list[str], int]:
        """
        The tree, the changed PSDs, then authors, edits and duplicates.

        Args:
            repository: Repository name (owner/repo)
            output_problematic: Whether to collect the validator's problem cards
            force_update: Re-parse every card even if its PSD did not change

        Returns:
            (formatted problem card messages, cards changed or moved)
        """
        self.repository = repository
        resp = self._fetch_tree(repository)
        self.tree = resp.json()                         # parsed once; the helpers below read it
        self.shas = {item.get("path", ""): item.get("sha") for item in self.tree.get("tree", [])}
        self.sizes = {item.get("path", ""): item.get("size") for item in self.tree.get("tree", []) if item.get("type") == "blob"}
        from commands import psd_details                # lazy: fontTools loads here
        self.font_index = psd_details.font_index(self.tree)     # this repo's MDW/Fonts

        self._populate_types_from_response(resp)
        self.parser = PSDParser(self.db.all_types, self.db, self._icons_from_response(resp, repository))

        self.edits = self._edits_from_response(resp, repository)
        self.dup_pngs = self._dup_pngs()                # before the parse's workers fork
        problems, num_changed = self._process_files_from_response(resp, repository, output_problematic, force_update)
        authors = self._authors_from_response(resp, repository)
        self.changed += self._apply_authors(authors)
        self.changed += self._apply_edits()
        self.changed += self._variant_duplicates()
        self.changed += self._sync_duplicates(resp)
        self.changed += self._apply_edits()             # the duplicates' own edits
        dates = self._dates_from_response(resp, repository)
        self.changed += self._apply_dates(dates)
        self._unmatched(authors, dates)
        if self.failed:
            self._send_failures()
        return (problems, num_changed)

    def _tree_items(self, response) -> list:
        """The tree's entries: self.tree once traverse_remote parsed it, else the response's."""
        return (self.tree if self.tree is not None else response.json()).get("tree", [])

    def _populate_types_from_response(self, response) -> None:
        """Populate card types from the tree."""
        self.db.all_types.clear()
        for item in self._tree_items(response):
            path = item.get("path", "")
            if path.startswith(TYPES_DIR) and "Stars" not in path and '.' in path:
                self.db.all_types.append(splitext(basename(path))[0].lower())

    def _populate_types_from_local(self, local_path: str) -> None:
        """Populate card types from a local clone (offline diagnostics; the build reads the tree)."""
        self.db.all_types.clear()
        types_dir = path_join(local_path, TYPES_DIR)
        for folder, _, files in walk(types_dir):
            if folder.endswith(TYPES_DIR):
                for file in files:
                    if not file.startswith('.'):
                        self.db.all_types.append(splitext(file)[0].lower())

    @staticmethod
    def _star_name(path: str) -> str:
        """Types/Stars/Dst Star.png -> "dst_star" (the name the icon font uses)."""
        return splitext(basename(path))[0].lower().replace(" ", "_")

    def _icons_from_response(self, response, repository: str) -> IconVocabulary:
        """Icon vocabulary from the GitHub tree: Types/ and Types/Stars/ names plus the font's manifest, if any."""
        paths = [item.get("path", "") for item in self._tree_items(response)]
        stars = [self._star_name(p) for p in paths if p.startswith(STARS_DIR + "/") and p.lower().endswith(".png")]
        manifest = None
        if ICON_MANIFEST_PATH in paths:
            try:
                manifest = json_loads(self._repo_text(ICON_MANIFEST_PATH))
            except (RequestException, ValueError) as e:
                if isinstance(e, RequestException):
                    self.failed += 1
                    self.lost.append(ICON_MANIFEST_PATH)
                print(f"Could not read {ICON_MANIFEST_PATH}, icon aliases and codepoints unavailable: {e}")
        return IconVocabulary.build(self.db.all_types, stars, manifest)

    def _authors_from_response(self, response, repository: str) -> Optional[CardAuthors]:
        """The card repo's authors file (AUTHORS_PATH), or None if missing or unreadable."""
        paths = [item.get("path", "") for item in self._tree_items(response)]
        path = AUTHORS_PATH if AUTHORS_PATH in paths else None
        if path is None:
            return None
        try:
            return CardAuthors.parse(self._repo_text(path))
        except RequestException as e:
            self.failed += 1
            self.lost.append(path)
            print(f"Could not read {path}; authors stay as they are: {e}")
            return None

    def _apply_authors(self, authors: Optional[CardAuthors]) -> int:
        """Set every card's author from the authors file (all cards, not just re-parsed ones). Returns how many changed."""
        if authors is None:
            return 0
        changed = 0
        for name, card in self.db.stats.items():
            author = authors.resolve(name, card.author)
            if author != card.author:
                card.author = author
                changed += 1
        if changed:
            print(f"{changed} authors set from {AUTHORS_PATH}.")
        return changed

    def _edits_from_response(self, response, repository: str) -> Optional[CardEdits]:
        """The card repo's website fixes (EDITS_PATH), or None if missing or unreadable."""
        paths = [item.get("path", "") for item in self._tree_items(response)]
        path = EDITS_PATH if EDITS_PATH in paths else None
        self.edits_unread = False
        if path is None:
            return None
        try:
            return CardEdits.parse(self._repo_text(path))
        except (RequestException, ValueError) as e:
            self.edits_unread = True
            if isinstance(e, RequestException):
                self.failed += 1
                self.lost.append(path)
            print(f"Could not read {path}; edits stay as they are: {e}")
            return None

    def _sync_duplicates(self, response) -> int:
        """
        Duplicates: a card PNG with no PSD whose Card Edits.json entry has dup_of (Stubborn_Queen.png, build_site's
        rule; numbered PNGs are variant pictures since the 5 PSDs were tagged). Each is recorded as a
        copy of the original's card with duplicate_of set. One that is no longer found stays out of seen_cards, so the
        guarded prune archives it like any card; with Card Edits.json unread, one whose PNG and original are still
        there stays as it is (its dup_of can't be read). Returns how many entries changed.
        """
        paths = [item.get("path", "") for item in self._tree_items(response)]
        found = duplicate_pngs(paths, self.db.stats, self.edits)
        before = {n: repr(c) for n, c in self.db.stats.items()
                  if getattr(c, "duplicate_of", None) and not getattr(c, "variant_of", None)}
        if self.edits_unread:
            lower = {p.lower() for p in paths}
            for name in before:
                card = self.db.stats[name]
                original = self.db.stats.get(card.duplicate_of)
                if name not in found and original is not None and not getattr(original, "duplicate_of", None) \
                        and duplicate_png(card).lower() in lower:
                    self.db.seen_cards.append(name)     # kept, not archived for a file that failed to read
        for name, original in found.items():
            src = self.db.stats[original]
            card = deepcopy(src)
            card.name, card.duplicate_of, card.parsed_values, card.problems = name, original, None, []
            card.art = card.ability_art = None          # its own art: its own sources (its entry), never the original's
            own = (self.edits.cards.get(CardAuthors.key(name), {}).get("aliases") or []) if self.edits is not None else []
            card.display_name = self._duplicate_name(own)
            self.db.stats[name] = card
            self.db.seen_cards.append(name)             # not pruned as a missing PSD
        return sum(1 for n in found if before.get(n) != repr(self.db.stats[n]))

    def _variant_duplicates(self) -> int:
        """Each variant picture's duplicate_of (variant_duplicates): the same card, counted once; it keeps its own parse,
        PNG, parts and history. Card Edits.json unread: one already set stays. Returns how many changed."""
        changed = 0
        for name, original in variant_duplicates(self.db.stats, None if self.edits_unread else self.edits).items():
            card = self.db.stats[name]
            if self.edits_unread and getattr(card, "duplicate_of", None) in self.db.stats:
                continue
            changed += getattr(card, "duplicate_of", None) != original
            card.duplicate_of = original
        return changed

    def duplicate_versions(self) -> None:
        """Each duplicate's `version` a copy of its original's, once the original has its own for this run (a re-parsed
        original gets it after _sync_duplicates copied it): the same whether or not the original was re-parsed."""
        for card in self.db.stats.values():
            original = self.db.stats.get(getattr(card, "duplicate_of", None) or "")
            if original is None or original is card or getattr(card, "variant_of", None):     # a variant: its own
                continue
            info = getattr(original, "version", None)
            if info is None:
                vars(card).pop("version", None)         # the class default, as an original without one
            else:
                card.version = deepcopy(info)

    @staticmethod
    def _duplicate_name(own: Optional[list] = None) -> Optional[str]:
        """The longest of own (the duplicate's Card Edits.json aliases), else None: it is titled by its own name."""
        found = [a for a in own or [] if a.strip()]
        if found:
            return " ".join(w[:1].upper() + w[1:] for w in max(found, key=len).replace("_", " ").split())
        return None

    def _apply_edits(self) -> int:
        """Apply the website fixes to every card, then straight apostrophes on every record, re-parsed or not (owner,
        2026-10-25). Returns how many changed."""
        changed = 0
        for name, card in self.db.stats.items():
            before = repr(card)
            if self.edits is not None:
                self.edits.apply(name, card)
            straighten_card(card)
            values = getattr(card, "parsed_values", None)
            if values:                                  # the PSD values the site puts back under an edit
                card.parsed_values = {k: straight(v) if k in STRAIGHT_KEYS else v for k, v in values.items()}
            changed += repr(card) != before
        if changed:
            print(f"{changed} cards changed by {EDITS_PATH} or straight apostrophes.")
        return changed

    def _dates_from_response(self, response, repository: str) -> Optional[CardDates]:
        """The card repo's creation dates (CREATED_PATH), or None if missing or unreadable."""
        paths = [item.get("path", "") for item in self._tree_items(response)]
        path = CREATED_PATH if CREATED_PATH in paths else None
        if path is None:
            return None
        try:
            return CardDates.parse(self._repo_text(path))
        except (RequestException, ValueError) as e:
            if isinstance(e, RequestException):
                self.failed += 1
                self.lost.append(path)
            print(f"Could not read {path}; creation dates stay as they are: {e}")
            return None

    def _first_upload(self, name: str, card: CardInfo) -> Optional[str]:
        """
        The UTC day of the earliest version of the card the database holds: its first upload as far as the bot saw it
        (the website's is its PNG's first commit; the dates file covers the cards from before the bot's history).
        None for no real date.
        """
        from commands.card_history import start
        versions = [*self.db.old_stats.get(name, ()), card]
        known = [at for at in (start(v)[0] for v in versions) if at and at >= REAL_DATES]
        return datetime.fromtimestamp(min(known), timezone.utc).date().isoformat() if known else None

    def _apply_dates(self, dates: Optional[CardDates]) -> int:
        """Set every card's created from the dates file and its first upload (duplicates included). Returns how many
        changed."""
        if dates is None:
            return 0
        uploads = {name: self._first_upload(name, card) for name, card in self.db.stats.items()}
        changed = sum(1 for name, card in self.db.stats.items() if dates.apply(name, card, uploads[name]))
        if changed:
            print(f"{changed} creation dates set from {CREATED_PATH} and first uploads.")
        missing = sorted(n for n, card in self.db.stats.items() if uploads[n] and CardAuthors.key(n) not in dates.cards)
        if missing:
            print(f"{len(missing)} cards not in {CREATED_PATH}, dated by their first upload until it has them "
                  f"(python -m commands.card_dates created): {', '.join(missing[:10])}")
        return changed

    def _card_psds(self, response) -> list:
        """The tree's card PSDs (excluded folders and rule pages left out: the rulebook is site/rulebook's chapters since
        2026-10-29), one per card name: of two PSDs with one name the stored
        card's path stays (else the first in tree order) and the other is ignored, one line each (a clash would
        otherwise flip the card between them and add a version at every run)."""
        items, by_name = [], {}
        for item in self._tree_items(response):
            path = item.get("path", "")
            parts = path.split('/')
            if path.endswith('.psd') and not any(folder in EXCLUDE_FOLDERS or folder == RULEBOOK_DIR for folder in parts):
                by_name.setdefault(card_name(path), []).append(item)
                items.append(item)
            elif path.endswith('.psd') and RULEBOOK_DIR in parts:
                self.rule_pages.add(CardAuthors.key(card_name(path)))
        skipped = set()
        for name, found in by_name.items():
            if len(found) < 2:
                continue
            stored = getattr(self.db.stats.get(name), "path", None)
            keep = next((item for item in found if item.get("path") == stored), found[0])
            others = [item.get("path", "") for item in found if item is not keep]
            skipped.update(others)
            self.report(f"Two PSDs share the card name {name}: {', '.join(item.get('path', '') for item in found)}; "
                        f"{', '.join(others)} ignored until one is renamed.")
        return [item for item in items if item.get("path", "") not in skipped]

    def _todo(self, response, force_update: bool) -> tuple[list, int]:
        """The tree's card PSDs to re-parse (path, name, blob) and how many stay; archived names back are restored, a
        listed card's PSD under a new name is a rename (renamed_from)."""
        todo, num_old = [], 0
        items = self._card_psds(response)
        names = {card_name(item.get("path", "")) for item in items}
        gone: dict = {}                                 # blob -> a listed name gone from the tree
        for listed, card in self.db.stats.items():
            if (listed not in names and getattr(card, "blob_sha", None) and not getattr(card, "duplicate_of", None)
                    and not getattr(card, "variant_of", None)):
                gone.setdefault(card.blob_sha, listed)
        for item in items:
            path = item.get("path", "")
            name = card_name(path)
            self.db.seen_cards.append(name)
            self.db.seen_cards.extend(getattr(self.db.stats.get(name), "variants", None) or [])   # until a re-parse says
            if name not in self.db.stats and self.db.restore(name, path, item.get("sha")):  # same blob: no download
                self.restored.append(name)
                self.changed += 1
            elif name not in self.db.stats and self.db.archive_tag(name):
                self.returned.append(name)              # another PSD under an archived name: a new card, history kept
            if name not in self.db.stats:
                old = gone.get(item.get("sha")) or self._art_renamed(path, names)
                if old:
                    self.renamed_from[name] = old
            if (force_update or CardAuthors.key(name) in self.only or path in self.again
                    or self._psd_changed(name, path, item.get("sha"))):
                todo.append((path, name, item.get("sha")))
            else:
                num_old += 1
        return (todo, num_old)

    def _process_files_from_response(self,
                                     response,
                                     repository: str,
                                     output_problematic: bool,
                                     force_update: bool) -> tuple[list[str], int]:
        """Re-parse the PSDs whose blob changed (all with force_update). Unchanged cards cost no requests."""
        problematic_cards = []
        num_updated = num_changed = num_moved = 0
        todo, num_old = self._todo(response, force_update)

        self.raw_ref = self._raw_ref() if todo and range_reads() else ""
        if todo:
            self._shared()
        global _BUILD
        forked = bool(todo) and _forks()                # after _shared: the workers start with it built
        if forked:
            _BUILD = self
            workers = RANGED_WORKERS if self.raw_ref else DOWNLOAD_WORKERS
            pool = ProcessPoolExecutor(min(workers, len(todo)), mp_context=get_context("fork"), initializer=_fork_init)
        else:
            pool = ThreadPoolExecutor(DOWNLOAD_WORKERS)
        queue, ahead = iter(todo), deque()

        def submit(item: tuple):
            return pool.submit(_fork_read, *item, self._given()) if forked else pool.submit(self._fetch_read, *item)

        def threads(error: Exception) -> None:          # the forked workers broke or hung: the rest in threads
            nonlocal pool, forked
            print(f"Forked reads stopped ({error!r}); the rest in threads")
            pool.kill_workers()
            pool, forked = ThreadPoolExecutor(DOWNLOAD_WORKERS), False
            for _ in range(len(ahead)):
                item, _task = ahead.popleft()
                ahead.append((item, submit(item)))

        def top_up() -> None:                           # keep AHEAD reads going (RANGED_AHEAD forked by ranges)
            while len(ahead) < (RANGED_AHEAD if forked and self.raw_ref else AHEAD):
                item = next(queue, None)
                if item is None:
                    return
                try:
                    task = submit(item)
                except (BrokenProcessPool, OSError) as e:
                    if not forked:
                        raise
                    threads(e)
                    task = submit(item)
                ahead.append((item, task))

        def result(item: tuple, task):
            if not forked:
                return task.result()
            try:
                read, back = task.result(FORK_WAIT)
            except Exception as e:                      # a worker died or hung, or its Read would not pickle: read here
                if isinstance(e, (BrokenProcessPool, TimeoutError)):
                    threads(e)
                return self._fetch_read(*item)
            self._taken(back)
            return read

        try:
            top_up()
            while ahead:                                # kept in todo order
                item, task = ahead.popleft()
                path, name, sha = item
                read = result(item, task)
                top_up()
                if read is None:
                    self._not_fetched(path, name)
                    continue
                try:
                    timestamp, author = self._get_remote_timestamp(path, name)
                except Exception as e:                          # no date: the card waits
                    self._not_looked_up(path, name, e)
                    continue
                try:
                    card = self._keep(read, path, name, sha)
                    if card is None:                            # range reads failed: the whole file decides
                        source = self._download(path, sha)
                        if source is None:
                            self._not_fetched(path, name)
                            continue
                        try:
                            card = self._parse(source, path, name, sha)
                        finally:
                            self._release(source)
                except Exception as e:                          # one bad PSD: its card keeps its data, the run goes on
                    self._unreadable(path, name, sha, e)
                    continue
                old = self.db.stats.get(name)
                if old is None or getattr(old, "blob_sha", None) != sha:
                    self.new_blobs.setdefault(sha, name)
                if old is None:
                    num_changed += 1
                elif old.path != path:
                    num_moved += 1
                else:
                    num_changed += 1
                self.reparsed[name] = (old, path, sha)
                card.timestamp = timestamp
                card.blob_sha = sha
                if author and not card.author:
                    card.author = author
                straighten_card(card)                           # straight apostrophes (owner, 2026-10-25): the
                if self.edits:                                  # parse, then the edits' values
                    self.edits.apply(name, card)
                    straighten_card(card)
                self.db.stats[name] = card

                if output_problematic:
                    problems = CardValidator.validate(card)
                    if problems:
                        problematic_cards.append((path, card, problems))
                self._store_variants(name, old, card, path, sha, timestamp, author, problematic_cards if output_problematic else None)

                num_updated += 1
                if num_updated % UPDATE_RATE == 0:
                    self.progress(num_updated)
        finally:
            for _, task in ahead:
                task.cancel()
            pool.shutdown(wait=True)                            # each task closes its own PSD
            _BUILD = None

        if self.ranged:
            print(f"{len(self.ranged)} PSDs read by range requests at {self.raw_ref[:8]}: "
                  f"{sum(self.ranged) / 1e6:.1f} MB fetched.")
        self.report(f"{num_changed} cards changed or new, {num_old} unchanged, {num_moved} moved.")
        return (self._format_problems(problematic_cards), num_changed + num_moved)

    def _store_variants(self, name: str, old: Optional[CardInfo], card: CardInfo, path: str, sha: str, timestamp: float,
                        author: Optional[str], problematic: Optional[list]) -> None:
        """A tagged PSD's variants 2..N as their own cards (variant_of the card); variants the PSD no longer has go."""
        found = self.variant_cards.pop(name, [])
        dropped = set(getattr(old, "variants", None) or []) - {v.name for v in found}
        self.db.seen_cards[:] = [n for n in self.db.seen_cards if n not in dropped]    # archived like a deleted card
        for variant in found:
            before = self.db.stats.get(variant.name)
            if before is None or getattr(before, "blob_sha", None) != sha:
                self.new_blobs.setdefault(sha, variant.name)
            self.reparsed[variant.name] = (before, path, sha)
            variant.timestamp, variant.blob_sha = timestamp, sha
            variant.author = variant.author or card.author or author
            if self.edits:
                self.edits.apply(variant.name, variant)
            self.db.stats[variant.name] = variant
            self.db.seen_cards.append(variant.name)
            if problematic is not None:
                problems = CardValidator.validate(variant)
                if problems:
                    problematic.append((path, variant, problems))

    def _dup_pngs(self) -> dict:
        """card_edits.dup_pictures of the tree; Card Edits.json unread: the stats' duplicates whose PNG is in the tree."""
        if not self.edits_unread:
            return dup_pictures(list(self.shas), self.edits.cards if self.edits is not None else None)
        out: dict = {}
        for card in self.db.stats.values():
            original = self.db.stats.get(getattr(card, "duplicate_of", None) or "")
            if original is not None and original.path and not getattr(card, "variant_of", None) \
                    and duplicate_png(card) in self.shas:
                out.setdefault(original.path, []).append(duplicate_png(card))
        return out

    def _parse_psd(self, psd: PSDImage, path: str, name: str, existing: Optional[CardInfo]) -> tuple:
        """
        (the card, its variants or None); a PSD with variant tags (PARTS_EDITOR.md §9) is parsed once per picture:
        variant 1 is the card, variants 2..N wait in variant_cards once kept (named after their PNG: "A Snack2"). A
        picture without its PNG in the tree is a problem on its card. A picture a picture-only duplicate's PNG is
        (picture_numbers: Gust_of_Trees.png, owner 11-09) is no variant: the duplicate stays _sync_duplicates' copy.
        """
        pictures = variants_of(psd)
        if len(pictures) < 2:
            return (self.parser.parse(psd, path, existing), None)
        dups = self.dup_pngs.get(path, [])
        if dups:
            folder, stem = path[:path.rfind("/") + 1], splitext(basename(path))[0]
            drawn = picture_numbers(psd, [f"{folder}{stem}{n}.png" for n in pictures[1:]
                                          if f"{folder}{stem}{n}.png" in self.shas] + dups)
            apart = {n for png, n in drawn.items() if png in dups}
            pictures = [n for n in pictures if n not in apart]
        card = self.parser.parse(psd, path, existing, variant=1)
        if len(pictures) < 2:
            return (card, None)
        cards = [card]
        for n in pictures[1:]:
            own = f"{name}{n}"
            variant = self.parser.parse(psd, path, self.db.stats.get(own) or existing, variant=n)
            variant.name, variant.variant, variant.variant_of = own, n, name
            cards.append(variant)
        card.variant, card.variants = 1, [c.name for c in cards[1:]]
        folder, stem = path[:path.rfind("/") + 1], splitext(basename(path))[0]
        for n, each in zip(pictures, cards):
            if self.shas and f"{folder}{stem}{'' if n == 1 else n}.png" not in self.shas:
                each.problems.append(f"VARIANT {n} HAS NO PNG")
        return (card, cards[1:])

    def _file_bytes(self, path: str) -> Optional[bytes]:
        """A repo file's bytes by its tree blob (the templates); None when not in the tree or not fetched."""
        sha = self.shas.get(path)
        if sha is None:
            return None
        try:
            return self._blob(sha).content
        except RequestException as e:
            print(f"{path}: not fetched ({e})")
            return None

    def _fetch_psd(self, path: str, sha: str):
        """
        The PSD to parse (in a download worker): a RemotePSD read by range requests at raw_ref, else the whole blob in a
        temp file (_download); None when neither worked.
        """
        if self.raw_ref:
            try:
                return RemotePSD(self.repository, self.raw_ref, path, self.sizes.get(path))
            except Exception as e:                              # the whole file decides
                if isinstance(e, RequestException) or (isinstance(e, RangeError) and e.status >= 400):
                    self._refuse(path)
                print(f"{path}: range reads failed ({e}); downloading it whole")
        return self._download(path, sha)

    def _refuse(self, path: str) -> None:
        """A PSD refused range reads; RANGE_REFUSALS of them turn them off (e.g. a commit not on GitHub, raw down: each
        PSD tried 3 times): whole downloads from then on. One odd PSD (a 416 when empty) leaves them on."""
        self.refused.append(path)
        if len(self.refused) >= RANGE_REFUSALS:
            self.raw_ref = ""

    def _given(self) -> dict:
        """The fields download workers write, as a forked one starts a PSD with them (raw_ref: a switch-off reaches
        later PSDs)."""
        return {"raw_ref": self.raw_ref, "refused": []}

    def _taken(self, back: dict) -> None:
        """A forked worker's fields after its PSD (_given's), applied here: its range refusals."""
        for path in back["refused"]:
            self._refuse(path)

    @staticmethod
    def _release(source) -> None:
        """Close a RemotePSD or remove a temp file (again is fine)."""
        if isinstance(source, RemotePSD):
            source.close()
        elif source and isfile(source):
            remove(source)

    def _shared(self) -> None:
        """The parse's lazy shared objects, made before the download workers share them."""
        from commands import psd_details
        from commands.psd_parts import Templates
        if self.templates is None:
            self.templates = Templates.from_blobs(self._file_bytes)
        self._make_repo_tree()
        try:
            psd_details._icon_parser(self.db.all_types)
        except Exception:                               # the details say so, card by card
            pass

    def _make_repo_tree(self) -> None:
        """_repo_tree, made once before workers share it."""
        if self._repo_tree is None:
            from psd_tools.api.links import RepoTree
            self._repo_tree = RepoTree(self.sizes)

    def _fetch_read(self, path: str, name: str, sha: str) -> Optional["Read"]:
        """A download worker's PSD: fetched, read (_read) and closed; None when not fetched."""
        source = self._fetch_psd(path, sha)
        if source is None:
            return None
        try:
            return self._read(source, path, name, sha)
        finally:
            self._release(source)

    def _parse(self, source, path: str, name: str, sha: str) -> Optional[CardInfo]:
        """The card from one open of the PSD, its details, links and variants kept (_read, _keep); raises what the
        parse raised."""
        return self._keep(self._read(source, path, name, sha), path, name, sha)

    def _read(self, source, path: str, name: str, sha: str) -> "Read":
        """
        The card, its variants, details and links from one open of the PSD, set on nothing (a download worker's).
        No card for a RemotePSD that raised or whose reads broke: the caller reads the whole file instead, so a failed
        range read never changes a result (a lazy open checks only the payloads it reads).
        """
        existing = self.db.stats.get(self.renamed_from.get(name, ""))
        read = Read()
        if isinstance(source, RemotePSD):
            try:
                card, variants = self._parse_psd(source.psd, path, name, existing)
                details = self._details(source.psd, path, name, card)
                read.links = self._linked(source.psd, path, sha)
            except Exception as e:
                print(f"{path}: range reads failed ({e}); downloading it whole")
                return read
            if source.failed:
                print(f"{path}: a range read failed; downloading it whole")
                return read
            read.card, read.variants, read.details, read.fetched = card, variants, details, source.fetched
            return read
        try:
            psd = PSDImage.open(source)
            read.links = self._linked(psd, path, sha)           # before the parse: one it refuses still has its links
            read.card, read.variants = self._parse_psd(psd, path, name, existing)
            read.details = self._details(psd, path, name, read.card)
        except Exception as e:
            read.card, read.error = None, e
        return read

    def _keep(self, read: "Read", path: str, name: str, sha: str) -> Optional[CardInfo]:
        """A read's links, details, variants and range bytes on the build; its card (None: read it whole). Raises the
        parse's error."""
        if read.links is not None:
            self.links[path] = read.links
        if read.error is not None:
            raise read.error
        if read.card is not None:
            if read.variants is not None:
                self.variant_cards[name] = read.variants
            if read.details is not None:
                self.details[sha] = read.details
            if read.fetched is not None:
                self.ranged.append(read.fetched)
        return read.card

    def _not_fetched(self, path: str, name: str) -> None:
        """A PSD without a file: held back (GitBuild's budget, read next run) or failed (retried next run)."""
        if self._held(path, name):
            return
        self.failed += 1
        self.failed_new += name not in self.db.stats or name in self.restored
        self.lost.append(path)

    def _unmatched(self, authors: Optional[CardAuthors], dates: Optional[CardDates]) -> None:
        """The side files' names that match no card (a typo; Card Edits.json lines of an archived card too: Card
        Authors.txt and Card Creation Dates.json may credit and date archived cards, any of them the rule pages): one
        console line per file."""
        side_names = {path: set(found.cards) for path, found in
                      ((AUTHORS_PATH, authors), (EDITS_PATH, self.edits), (CREATED_PATH, dates)) if found is not None}
        listed = {CardAuthors.key(name) for name in self.db.stats}
        archived = {CardAuthors.key(name) for name, versions in self.db.old_stats.items()
                    if versions and getattr(versions[len(versions) - 1], "archived", None)}   # credits, dates of archived cards
        for path, keys in side_names.items():
            names = sorted(keys - listed - self.rule_pages - (set() if path == EDITS_PATH else archived))
            if names:
                print(f"{basename(path)}: {len(names)} name{'' if len(names) == 1 else 's'} match no card: "
                      f"{', '.join(names[:20])}{', ...' if len(names) > 20 else ''}")

    def _unreadable(self, path: str, name: str, sha: str, error: Exception) -> None:
        """Count a PSD that raised as failed (a new name holds the prune, like a failed download); one console line."""
        print(f"{path}: not read ({error}); its card keeps its last version")
        self.failed += 1
        self.failed_new += name not in self.db.stats or name in self.restored
        self.unreadable.append({"path": path, "blob": sha, "error": str(error)[:300]})

    def _not_looked_up(self, path: str, name: str, error: Exception) -> None:
        """A PSD whose date or author lookup raised: failed like a download (retried next run), its card unchanged."""
        print(f"{path}: date not looked up ({error}); its card keeps its last version")
        self.failed += 1
        self.failed_new += name not in self.db.stats or name in self.restored
        self.lost.append(path)
        self.unlooked.append(path)

    def _details(self, psd: PSDImage, path: str, name: str, card: Optional[CardInfo] = None) -> Optional[dict]:
        """The card's PSD facts for the website (psd_details.json); a failure is one console line (None), never a failed
        update."""
        try:
            from commands import psd_details                # lazy: fontTools (through font_check) loads with the first parse
            if self.templates is None:
                from commands.psd_parts import Templates
                self.templates = Templates.from_blobs(self._file_bytes)
            return psd_details.gather(psd, path, name, self.font_index, self.db.all_types, self.parser, card, self.templates)
        except Exception as e:
            print(f"{path}: details not gathered ({e})")
            return None

    def _linked(self, psd: PSDImage, path: str, sha: str) -> dict:
        """A PSD's link record ({"blob", "items": psd_regions.linked_items}, or {"blob", "error"}: the fork cannot read
        its links, which holds the guards' deletions)."""
        try:
            from commands.psd_regions import linked_items
            if self._repo_tree is None:
                from psd_tools.api.links import RepoTree
                self._repo_tree = RepoTree(self.sizes)
            return {"blob": sha, "items": linked_items(psd, path, self._repo_tree)}
        except Exception as e:                  # recorded: the guards delete nothing while a PSD's links are unknown
            return {"blob": sha, "error": f"{type(e).__name__}: {e}"[:200]}

    def _send_failures(self) -> None:
        """The failed downloads, the failed date lookups, then the PSDs that could not be read (by path)."""
        unread, unlooked = len(self.unreadable), len(self.unlooked)
        downloads = self.failed - unread - unlooked
        if unlooked:
            self.report(f"Could not date {unlooked} PSD{'' if unlooked == 1 else 's'} ("
                        + ", ".join(self.unlooked[:10]) + (", ..." if unlooked > 10 else "") + "); "
                        + ("its card keeps its" if unlooked == 1 else "their cards keep their")
                        + " last version and the next run tries again.")
        if downloads:
            self.report(f"Could not download {downloads} file{'' if downloads == 1 else 's'} from the card repo; "
                        "the next run tries again.")
        if unread:
            self.report(f"Could not read {unread} PSD{'' if unread == 1 else 's'} from the card repo ("
                        + ", ".join(u["path"] for u in self.unreadable) + "); "
                        + ("its card keeps its" if unread == 1 else "their cards keep their")
                        + " last version and the next run tries again.")

    @staticmethod
    def _format_problems(
        problematic_cards: list[tuple[str, CardInfo, list[str]]]
    ) -> list[str]:
        """Format problematic cards for display."""
        return [
            f"{path}\n```\n" + '\n'.join(problems) + "\n```"
            for path, _, problems in problematic_cards
        ]


# ========================
# Helper Functions
# ========================
# ========================
# db/links.json (the 11-07 pass: every PSD's links, the linked files' blobs)
# ========================
def read_links(db_dir: str) -> dict:
    """db/links.json (PLAN A10's shape, psd_links reads it): {PSD path: {"blob", "links": [{"path", "name", "size",
    "drawn"}]} or {"blob", "error"}, "files": {Types/ or MDW/Linked path: {blob: size or None}}, "pixels": {MDW/Linked
    blob: its decoded pixels' sha1, None when unreadable}}; "files" and "pixels" empty without one."""
    path = path_join(db_dir, LINKS_JSON)
    data = {}
    if isfile(path):
        with open(path, encoding="utf-8") as fh:
            data = json_loads(fh.read())
    return {**data, **{key: dict(data.get(key) or {}) for key in LINK_KEYS}}


def links_text(data: dict) -> str:
    return json_dumps(data, ensure_ascii=False, indent=1, sort_keys=True) + "\n"


def read_records(db_dir: str) -> tuple:
    """(render record, spot record) of db/; one db/ lacks: psd_edit's seed copy (none since 11-08), else {}."""
    from commands import psd_edit
    out = []
    for name in RECORDS:
        found = path_join(db_dir, name)
        path = found if isfile(found) else path_join(psd_edit.RECORDS_SEED, name)
        try:
            with open(path, encoding="utf-8") as fh:
                out.append(json_loads(fh.read()))
        except FileNotFoundError:
            out.append({})
    return tuple(out)


def records_text(record: dict) -> str:
    return json_dumps(record, indent=0, sort_keys=True) + "\n"


def status_colors(run_build, before: dict) -> dict:
    """db/status_colors.json at the build's head: {"blob": the status chart's, "colors": psd_edit.status_inks}; the
    chart read again only when its blob changed, before kept when it can't be read."""
    from commands import psd_edit
    sha = run_build.shas.get(psd_edit.STATUS_CHART)
    if sha == before.get("blob"):
        return before
    if sha is None:
        return {"blob": None, "colors": {}}
    data = run_build._file_bytes(psd_edit.STATUS_CHART)
    try:
        return {"blob": sha, "colors": psd_edit.status_inks(PSDImage.open(BytesIO(data)))}
    except Exception as e:                                # not fetched or unreadable: the next build tries again
        print(f"{psd_edit.STATUS_CHART}: status colours not read ({type(e).__name__}: {e})")
        return before


def linked_file(path: str) -> bool:
    """A file cards link: Types/ (stars and types) or MDW/Linked (every other picture)."""
    return path.startswith((f"{TYPES_DIR}/", f"{LINKED_DIR}/"))


def _drawn(item: dict, psd_path: str, files: dict) -> Optional[str]:
    """The file blob a link's raster was drawn from: the one whose contentID (the fork's link_content_id over blob, PSD
    path and item) the record carries; None for a link Photoshop made (no contentID of ours) or an unknown blob."""
    if not item.get("content") or not item.get("path"):
        return None
    from psd_tools.api.links import link_content_id
    return next((blob for blob in files.get(item["path"], {})
                 if link_content_id(blob, psd_path, item["uuid"]) == item["content"]), None)


def _link_record(found: dict, psd_path: str, files: dict) -> dict:
    if "error" in found:
        return {"blob": found["blob"], "error": found["error"]}
    return {"blob": found["blob"], "links": [{"path": item["path"], "name": item["name"], "size": item["size"],
                                              "drawn": _drawn(item, psd_path, files)} for item in found["items"]]}


def _batch(store: str, shas: list) -> dict:
    """{sha: size} of the blobs a local store holds (cat-file --batch-check: no lazy fetch)."""
    if not store or not shas:
        return {}
    done = run(["git", "-C", store, "cat-file", "--batch-check"], input="\n".join(shas).encode() + b"\n",
               capture_output=True, env={**environ, **NO_LAZY})
    out = {}
    for line in done.stdout.decode("utf-8", "replace").splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[1] == "blob":
            out[parts[0]] = int(parts[2])
    return out


def fetch_blobs(checkout: str, shas: list) -> None:
    """The blobs a blobless clone lacks, fetched from origin FETCH_CHUNK a fetch, FETCH_POOL at a time (no REST); a failed
    fetch leaves its blobs out (the caller reads what came)."""
    def one(chunk: list) -> None:
        run(["git", "-C", checkout, "-c", "fetch.negotiationAlgorithm=noop", "fetch", "-q", "--no-tags",
             "--no-write-fetch-head", "--recurse-submodules=no", "--filter=blob:none", "origin", *chunk],
            capture_output=True, env={**environ, "GIT_NO_LAZY_FETCH": "1"})
    missing = [sha for sha in shas if sha not in _batch(checkout, shas)]
    chunks = [missing[i:i + FETCH_CHUNK] for i in range(0, len(missing), FETCH_CHUNK)]
    if chunks:
        with ThreadPoolExecutor(FETCH_POOL) as pool:
            list(pool.map(one, chunks))


def _sizes(build: "GitBuild", paths: list) -> dict:
    """{path: size} of linked files: from the local stores, else one REST tree read per top folder (Types/, MDW/Linked)."""
    shas = {path: build.shas[path] for path in paths}
    have = {**_batch(build.checkout, list(shas.values())), **_batch(build.blobs, list(shas.values()))}
    out = {path: have[sha] for path, sha in shas.items() if sha in have}
    for top in (TYPES_DIR, LINKED_DIR):
        if not build.repository or not any(p.startswith(f"{top}/") and p not in out for p in shas) or not build.shas.get(top):
            continue
        try:
            reply = get_with_retry(api_url(build.repository, f"git/trees/{build.shas[top]}?recursive=1"), headers=HEADERS)
            reply.raise_for_status()
            for entry in reply.json().get("tree", []):
                full = f"{top}/{entry.get('path', '')}"
                if full in shas and entry.get("size") is not None and entry.get("sha") == shas[full]:
                    out[full] = int(entry["size"])
        except (RequestException, ValueError) as e:      # sizes unknown this run: the redraw waits for them
            print(f"{top}: sizes not read ({e})")
    return out


def _history(checkout: str, head: str) -> dict:
    """{path: {blob: None}} of every Types/ blob in head's history (trees only: git log --raw)."""
    out: dict = {}
    try:
        raw = _git(checkout, "log", "--raw", "--no-renames", "--no-abbrev", "--format=", head, "--", f"{TYPES_DIR}/")
    except CalledProcessError:
        return out
    for line in raw.splitlines():
        meta, _, path = line.partition("\t")
        fields = meta.split()
        if len(fields) >= 4 and path:
            for sha in fields[2:4]:
                if set(sha) != {"0"}:
                    out.setdefault(path, {})[sha] = None
    return out


def pixel_hash(data: bytes, path: str) -> str:
    """sha1 of a picture's decoded pixels (mode, size, RGBA): look-alike linked files (the guards flag them)."""
    from PIL import Image
    if path.lower().endswith(LINK_HOLDERS):
        image = PSDImage.open(BytesIO(data)).composite()
    else:
        image = Image.open(BytesIO(data))
        image.load()
    image = image.convert("RGBA")
    return sha1(repr(image.size).encode() + image.tobytes()).hexdigest()


def gather_links(build: "GitBuild", before: dict) -> dict:
    """
    db/links.json after this run: every PSD and .psb of the tree (cards from this run's parse, the others read when
    their blob moved, DOWNLOAD_WORKERS at a time, RANGED_WORKERS by ranges on a runner: range reads, else the local store,
    else REST within the budget; a card not parsed since its blob moved has no record, so the guards delete nothing), the
    linked files' blobs (append-only; seeded once from Types/ history, sizes unknown) and the pixels of MDW/Linked's
    (batched git fetches).
    """
    files = {path: dict(blobs) for path, blobs in before["files"].items()}
    kept_psds = {path: record for path, record in before.items() if path not in LINK_KEYS}
    if not files:
        files = _history(build.checkout, build.head)
    linked = [path for path in build.sizes if linked_file(path)]
    sizes = _sizes(build, [p for p in linked if files.get(p, {}).get(build.shas[p]) is None])   # stored sizes kept
    for path in linked:
        blobs = files.setdefault(path, {})
        blobs[build.shas[path]] = sizes.get(path, blobs.get(build.shas[path]))
    cards = {path for path in build.sizes if path.endswith(".psd")
             and not any(folder in EXCLUDE_FOLDERS or folder == RULEBOOK_DIR for folder in path.split("/"))}
    psds, unread = {}, []
    for path in sorted(p for p in build.sizes if p.lower().endswith(LINK_HOLDERS)):
        sha = build.shas[path]
        found, kept = build.links.get(path), kept_psds.get(path) or {}
        if found is not None and found["blob"] == sha:
            psds[path] = _link_record(found, path, files)
        elif kept.get("blob") == sha:
            psds[path] = kept
        elif path not in cards:
            unread.append(path)
    if unread and range_reads() and not build.raw_ref:
        build.raw_ref = build._raw_ref()
    todo = []
    for path in unread:                             # the budget in path order, before the workers
        if not build._store(build.shas[path]) and not build.raw_ref:
            if build.rest >= build.budget:
                continue                            # unknown this run: read on a later one
            build.rest += 1
        todo.append(path)
    ranged, lock = bool(build.raw_ref), Lock()
    if todo:
        build._make_repo_tree()

    def read(path: str) -> Optional[dict]:         # a download worker's record; None: not read this run
        sha = build.shas[path]
        if ranged and not build.raw_ref and not build._store(sha):
            with lock:                              # range reads went off meanwhile: whole, within the budget
                if build.rest >= build.budget:
                    return None
                build.rest += 1
        source = build._fetch_psd(path, sha)
        if source is None:
            return None
        try:
            psd = source.psd if isinstance(source, RemotePSD) else PSDImage.open(source)
            found = build._linked(psd, path, sha)
            if isinstance(source, RemotePSD) and source.failed:
                return None
            return _link_record(found, path, files)
        except Exception as e:                      # unreadable: recorded, the guards' deletions held
            return {"blob": sha, "error": f"{type(e).__name__}: {e}"[:200]}
        finally:
            build._release(source)

    runner = ranged and environ.get("GITHUB_ACTIONS") == "true"
    with ThreadPoolExecutor(RANGED_WORKERS if runner else DOWNLOAD_WORKERS) as pool:
        for path, record in zip(todo, pool.map(read, todo)):
            if record is not None:
                psds[path] = record
    pixels = {blob: found for blob, found in before["pixels"].items()}
    wanted = {build.shas[p]: p for p in linked if p.startswith(f"{LINKED_DIR}/") and build.shas[p] not in pixels}
    if wanted:
        fetch_blobs(build.checkout, list(wanted))
        for sha, path in wanted.items():
            store = build._store(sha)
            if not store:
                continue
            done = run(["git", "-C", store, "cat-file", "blob", sha], capture_output=True, env={**environ, **NO_LAZY})
            if done.returncode:
                continue
            try:
                pixels[sha] = pixel_hash(done.stdout, path)
            except Exception as e:                  # no look-alike test for it; None: not read again
                pixels[sha] = None
                print(f"{path}: pixels not read ({e})")
    return {**psds, "files": files, "pixels": pixels}


def duplicate_pngs(paths: list, stats: dict, edits: Optional[CardEdits]) -> dict:
    """
    The tree's duplicates: {name: original}: a card PNG with no PSD whose Card Edits.json
    entry names its original in dup_of (Stubborn_Queen.png; build_site's rule). The original must be a listed card, no
    duplicate. A numbered PNG beside a PSD is one of its variant pictures (" #N" layers): variant_duplicates.
    """
    lower = {p.lower() for p in paths}
    by_key = {CardAuthors.key(n): n for n, c in stats.items() if not getattr(c, "duplicate_of", None)}
    found = {}
    for path in paths:
        if (not path.lower().endswith(".png") or any(folder in EXCLUDE_FOLDERS for folder in path.split("/"))
                or RULEBOOK_DIR in path or (splitext(path)[0] + ".psd").lower() in lower):
            continue
        if "/" not in path or path.split("/")[0] in (TYPES_DIR, "site", "TTS"):
            continue                                    # no card folder
        if getattr(stats.get(card_name(path)), "variant_of", None):
            continue                                    # a tagged PSD's picture: its own card
        target = ((edits.cards.get(CardAuthors.key(card_name(path))) or {}).get("dup_of") if edits is not None else None)
        original = by_key.get(CardAuthors.key(target)) if target else None
        if original and original != card_name(path):
            found[card_name(path)] = original
    return found


def variant_duplicates(stats: dict, edits: Optional[CardEdits]) -> dict:
    """
    {variant picture: original} for every variant picture 2..N of a tagged PSD: a duplicate of its picture 1 (owner,
    10-25: one category), or of another listed card its Card Edits.json dup_of names (no duplicate itself).
    """
    pictures = {n: c.variant_of for n, c in stats.items() if getattr(c, "variant_of", None) in stats}
    by_key = {CardAuthors.key(n): n for n, c in stats.items() if n not in pictures and not getattr(c, "duplicate_of", None)}
    out = {}
    for name, first in pictures.items():
        target = (edits.cards.get(CardAuthors.key(name)) or {}).get("dup_of") if edits is not None else None
        out[name] = by_key.get(CardAuthors.key(target), first) if target else first
    return out


def can_prune(build: CardBuild) -> bool:
    """Whether the run saw the whole repo: a tree, not truncated, no failed new-name PSD, at least one card."""
    tree = build.tree
    if tree is None or build.failed_new:
        return False
    if tree.get("truncated") or not build.db.seen_cards:
        print("Card repo tree truncated or without cards; no card archived.")
        return False
    return True


# ========================
# The build (python -m commands.card_data)
# ========================
# what card-sync.yml checks out of AutoCroissant: the build's import closure and requirements-cards.txt, then the bake's
# and the save's
# (tests/test_card_data.py runs --help on the three CLIs with exactly this list)
SPARSE_FILES = ("commands/card_data.py", "commands/card_history.py", "commands/card_db.py", "commands/changelog.py",
                "commands/png_changes.py",
                "commands/card_models.py", "commands/card_edits.py", "commands/psd_parser.py", "commands/psd_details.py",
                "commands/psd_parts.py", "commands/text_align.py", "commands/psd_regions.py", "commands/font_check.py", "commands/github_api.py", "global_config.py",
                "commands/psd_remote.py",
                "requirements-cards.txt",
                "commands/card_bake.py", "commands/psd_edit.py", "commands/png_export.py", "commands/git_commit.py",
                "commands/repo_history.py", "commands/tts_save.py",
                "commands/ability_place.py", "commands/ability_index.json",
                "commands/rulebook.py", "commands/rulebook_pages.py",
                "commands/psd_links.py", "commands/psd_glyphs.py", "commands/card_standards.py")
BOT_REPO = "bobberlington/AutoCroissant"
BUDGET = 400                                                      # REST downloads per run (two runs an hour: 800 of 1,000)
RATE_FLOOR = 150                                                  # no more REST downloads under this many requests left
MIN_CARDS = 100                                                   # G1: build_site's MIN_CARDS
FLOOR_SHARE = 0.95                                                # G1: of the cards before
ARCHIVE_MIN, ARCHIVE_SHARE = 5, 0.01                              # G2: archived per run, at most max(5, 1%)
REPARSE_MAX = 25                                                  # G4: same-blob re-parses without approval
QUALITY_SLACK = 3                                                 # G5: cards that may lose a type, ability or stats
DETAILS_SHARE = 0.05                                              # G5: psd_details records that may go
EXIT_OK, EXIT_ERROR, EXIT_GATE = 0, 1, 2
NO_LAZY = {"GIT_NO_LAZY_FETCH": "1"}                              # a missing blob fails instead of being fetched


def _git(repo: str, *args: str) -> str:
    """git -C repo ..., never fetching a blob; raises CalledProcessError on a failure."""
    done = run(["git", "-C", repo, *args], capture_output=True, env={**environ, **NO_LAZY})
    if done.returncode != 0:
        raise CalledProcessError(done.returncode, ["git", *args], done.stdout, done.stderr)
    return done.stdout.decode("utf-8", "replace")


def ls_tree(repo: str, head: str) -> dict:
    """The recursive tree at head in the API's shape (path, mode, type, sha; directories too). Never truncated, no API."""
    items = []
    for entry in _git(repo, "ls-tree", "-r", "-t", "-z", head).split("\0"):
        if entry:
            info, path = entry.split("\t", 1)
            mode, kind, sha = info.split()
            items.append({"path": path, "mode": mode, "type": kind, "sha": sha})
    return {"sha": _git(repo, "rev-parse", f"{head}^{{tree}}").strip(), "tree": items, "truncated": False}


class _TreeReply:
    """The tree as the traversal's helpers take it (json())."""
    def __init__(self, tree: dict):
        self._tree = tree

    def json(self) -> dict:
        return self._tree


class GitBuild(CardBuild):
    """
    The build's source: a full clone of the card repo (git ls-tree and cat-file, never a lazy fetch) and its GitIndex.
    A blob comes from --blobs, then the clone, then the REST API within the budget; a card's timestamp is its blob's
    intro and a new card's author the first commit's at its path (card_history).
    """
    def __init__(self, db: StatsDatabase, checkout: str, head: str = "HEAD", index=None, blobs: str = "",
                 budget: int = BUDGET, **kwargs):
        from commands.card_history import GitIndex
        super().__init__(db, **kwargs)
        self.checkout, self.blobs, self.budget = checkout, blobs, budget
        self.head = _git(checkout, "rev-parse", head).strip()
        self.at = float(_git(checkout, "log", "-1", "--format=%ct", self.head).strip())   # not show: a merge diffs (blobs)
        self.index = index if index is not None else GitIndex.from_git(checkout, self.head)
        self.pushed = self.head                  # a commit on GitHub with head's other PSDs: range reads there
        self.pushed_blobs = None                 # path -> blob at pushed when that is not head (a bake's parent)
        self.rest = 0                            # REST downloads started
        self.stale: list = []                    # paths held back (budget, rate floor): read on a later run
        self.failed_before: dict = {}            # (path, blob) -> last run's pending.failed record
        self.still_failed: list = []             # of those, skipped this run (same blob): still failed
        self._rate_low = False
        self._rate_held: set = set()
        self._stores: dict = {}                  # blob -> store that has it

    def _fetch_tree(self, repository: str):
        return _TreeReply(ls_tree(self.checkout, self.head))

    def _store(self, sha: str) -> str:
        """The local object store that has this blob ("" when none)."""
        if sha in self._stores:
            return self._stores[sha]
        for store in (self.blobs, self.checkout):
            if store and run(["git", "-C", store, "cat-file", "-e", sha], capture_output=True,
                             env={**environ, **NO_LAZY}).returncode == 0:
                self._stores[sha] = store           # a found blob stays; a miss is asked again
                return store
        return ""

    def _repo_text(self, path: str) -> Optional[str]:
        sha = self.shas.get(path)
        if sha is None:
            return None
        store = self._store(sha)
        return _git(store, "cat-file", "blob", sha) if store else super()._repo_text(path)

    def _file_bytes(self, path: str) -> Optional[bytes]:
        sha = self.shas.get(path)
        store = self._store(sha) if sha else ""
        if store:
            return run(["git", "-C", store, "cat-file", "blob", sha], capture_output=True, env={**environ, **NO_LAZY}).stdout
        return super()._file_bytes(path)

    def _todo(self, response, force_update: bool) -> tuple[list, int]:
        todo, num_old = super()._todo(response, force_update)
        kept = []
        for path, name, sha in todo:
            known = self.failed_before.get((path, sha))
            if (known is not None and known.get("parser_version") == PARSER_VERSION and not force_update
                    and CardAuthors.key(name) not in self.only):
                self.still_failed.append(dict(known))           # unread until its blob, the parser or `cards` changes
                self.failed_new += name not in self.db.stats or name in self.restored
                continue
            if not self._store(sha):
                if self.rest >= self.budget:
                    self._hold(path, name)
                    continue
                self.rest += 1
            kept.append((path, name, sha))
        return (kept, num_old)

    def _given(self) -> dict:
        return {**super()._given(), "_rate_low": self._rate_low, "_rate_held": set()}

    def _taken(self, back: dict) -> None:
        super()._taken(back)
        self._rate_low = self._rate_low or back["_rate_low"]
        self._rate_held |= back["_rate_held"]

    def _hold(self, path: str, name: str) -> None:
        self.stale.append(path)
        self.failed_new += name not in self.db.stats or name in self.restored     # a held rename archives nothing

    def _held(self, path: str, name: str) -> bool:
        if path not in self._rate_held:
            return False
        self._hold(path, name)
        return True

    def _raw_ref(self) -> str:
        if self.pushed != self.head:
            try:
                self.pushed_blobs = {item["path"]: item["sha"] for item in ls_tree(self.checkout, self.pushed)["tree"]}
            except CalledProcessError as e:
                print(f"Range reads off this run: no tree at {self.pushed[:8]} ({e})")
                return ""
        return self.pushed

    def _fetch_psd(self, path: str, sha: str):
        """A blob the clone has is read there; others by range reads at pushed when it has that blob at the path, else
        whole through REST (both within the budget)."""
        if self._store(sha) or (self.pushed_blobs is not None and self.pushed_blobs.get(path) != sha):
            return self._download(path, sha)
        return super()._fetch_psd(path, sha)

    def _blob(self, sha: str, stream: bool = False):
        reply = super()._blob(sha, stream)
        left = (getattr(reply, "headers", None) or {}).get("X-RateLimit-Remaining")
        if left is not None and str(left).isdigit() and int(left) < RATE_FLOOR:
            self._rate_low = True
        return reply

    def _download(self, path: str, sha: str) -> Optional[str]:
        store = self._store(sha)
        if not store:
            if self._rate_low:
                self._rate_held.add(path)
                return None
            return super()._download(path, sha)
        fd, name = mkstemp(suffix=".psd")
        try:
            with fdopen(fd, "wb") as fh:
                done = run(["git", "-C", store, "cat-file", "blob", sha], stdout=fh, env={**environ, **NO_LAZY})
            if done.returncode == 0:
                return name
            print(f"{path}: blob {sha[:8]} not readable from {store}")
        except OSError as e:
            print(f"{path}: {e}")
        remove(name)
        return None

    def _get_remote_timestamp(self, path: str, name: str) -> tuple[float, Optional[str]]:
        """The blob's intro (a move or the same bytes again is no new date; after a revert, the revert: blob_since; a
        rename whose old name the database holds: the rename, intro_of); a new card's author: the first commit's at the
        oldest path its exact renames came from (never the renamer's)."""
        intro = self.index.intro_of(self.shas.get(path), name, set(self.db.stats) | set(self.db.old_stats), path)
        at = intro.at if intro is not None else self.at
        if name not in self.db.stats or not self.db.stats[name].author:
            first = self.index.first.get(self.index.origin(path))
            return (at, first.author if first is not None else None)
        return (at, None)

    def _art_renamed(self, path: str, names: set) -> Optional[str]:
        png = splitext(path)[0] + ".png"
        old = self.index.origin(png, once=True)
        name = card_name(old) if old != png else None
        return name if name in self.db.stats and name not in names else None

    def file_commit(self, path: str):
        """The newest commit of a file at head, as a card_history.Commit (None when it has none)."""
        from commands.card_history import Commit, mailmap_args
        line = _git(self.checkout, *mailmap_args(self.checkout, self.head), "log", "-1", "--format=%H%x09%ct%x09%aN", self.head,
                    "--", path).strip()
        if not line:
            return None
        sha, at, author = (line.split("\t", 2) + ["", ""])[:3]
        known = self.index.by_sha.get(sha)
        return known if known is not None else Commit(len(self.index.commits), sha, float(at), author)


@dataclass
class Outcome:
    """A build's result: the exit code, the counts, the refused gates and the texts for the summary and commit."""
    code: int = EXIT_OK
    counts: dict = field(default_factory=dict)
    refused: dict = field(default_factory=dict)       # gate -> why
    lifted: dict = field(default_factory=dict)        # gate -> the input that lifted it
    lines: list = field(default_factory=list)         # one per card event
    errors: list = field(default_factory=list)        # why the build failed (files not fetched)
    problems: list = field(default_factory=list)      # {card, path, notes} of re-parsed cards
    manifest: dict = field(default_factory=dict)


def _archived(stats: dict, old_stats: dict) -> dict:
    """StatsDatabase.archived() over plain dicts (the build's snapshot)."""
    return {name: versions[len(versions) - 1] for name, versions in old_stats.items()
            if versions and name not in stats and getattr(versions[len(versions) - 1], "archived", None)}


def gates(before: dict, after: dict, details_before: int, details_after: int, archived: list, same_blob: list,
          full: bool, only: set, allow_archive: bool) -> tuple[dict, dict]:
    """
    G1, G2, G4 and G5 (G3 checks the written files, G6 the data tip).

    Returns:
        ({gate: why refused}, {gate: what lifted it})
    """
    refused, lifted = {}, {}
    cards = [n for n, c in after.items() if not getattr(c, "duplicate_of", None)]
    had = [n for n, c in before.items() if not getattr(c, "duplicate_of", None)]
    if len(cards) < MIN_CARDS or (had and len(cards) < FLOOR_SHARE * len(had)):
        refused["G1"] = f"{len(cards)} cards (before {len(had)}; the floor is {MIN_CARDS} and {FLOOR_SHARE:.0%})"
    ceiling = max(ARCHIVE_MIN, int(ARCHIVE_SHARE * len(had)))
    if len(archived) > ceiling:
        if allow_archive:
            lifted["G2"] = "allow_archive"
        else:
            refused["G2"] = f"{len(archived)} cards archived (at most {ceiling}): {', '.join(sorted(archived)[:10])}"
    unapproved = [n for n in same_blob if CardAuthors.key(n) not in only]
    if len(unapproved) > REPARSE_MAX:
        if full:
            lifted["G4"] = "full"
        else:
            refused["G4"] = f"{len(unapproved)} cards re-parsed with an unchanged blob (more than {REPARSE_MAX}): approve with full"

    def worse(test) -> list:
        return sorted(n for n in cards if n in before and not test(before[n]) and test(after[n]))
    lost = {"UNKNOWN type": worse(lambda c: c.card_type == "unknown"),
            "ability text lost": worse(lambda c: not (c.ability or "").strip()),
            "every stat lost": worse(lambda c: all(v < 0 for v in (c.stats.hp, c.stats.defense, c.stats.attack, c.stats.speed)))}
    new_unknown = sum(1 for n in cards if after[n].card_type == "unknown") - sum(1 for n in had if before[n].card_type == "unknown")
    notes = [f"{what}: {', '.join(names[:10])}" for what, names in lost.items()
             if len(names) > QUALITY_SLACK and (what != "UNKNOWN type" or new_unknown > QUALITY_SLACK)]
    if details_before and details_after < (1 - DETAILS_SHARE) * details_before:
        notes.append(f"psd_details records {details_before} -> {details_after}")
    listed = {CardAuthors.key(n) for names in lost.values() for n in names}
    if notes:
        if full or (listed and listed <= only):
            lifted["G5"] = "full" if full else "cards"
        else:
            refused["G5"] = "; ".join(notes)
    return (refused, lifted)


def _restore(db_dir: str, sha: str) -> tuple:
    """A data commit's stats, old_stats, details and manifest (db/ at that commit of its own repo; fetched when a
    shallow checkout lacks it)."""
    from commands.card_db import MANIFEST_JSON, OLD_STATS_JSON, STATS_JSON, parse_old_stats, parse_stats
    if run(["git", "-C", db_dir, "cat-file", "-e", f"{sha}^{{commit}}"], capture_output=True).returncode != 0:
        _git(db_dir, "fetch", "-q", "--depth", "1", "origin", sha)
    prefix = _git(db_dir, "rev-parse", "--show-prefix").strip()
    text = {name: _git(db_dir, "show", f"{sha}:{prefix}{name}") for name in (STATS_JSON, OLD_STATS_JSON, "psd_details.json")}
    details = json_loads(text["psd_details.json"]).get("cards") or {}
    try:
        manifest = json_loads(_git(db_dir, "show", f"{sha}:{prefix}{MANIFEST_JSON}"))
    except CalledProcessError:
        manifest = {}
    return (parse_stats(text[STATS_JSON]), parse_old_stats(text[OLD_STATS_JSON]), details, manifest)


def _data_head(db_dir: str) -> str:
    """The commit of the repo db/ sits in ("" when it is no git work tree or has no commit)."""
    try:
        return _git(db_dir, "rev-parse", "HEAD").strip()
    except (CalledProcessError, OSError):
        return ""


def _data_branch(db_dir: str) -> str:
    """The branch of the repo db/ sits in."""
    try:
        return _git(db_dir, "symbolic-ref", "-q", "--short", "HEAD").strip() or "data"
    except CalledProcessError:
        return "data"                                # a detached checkout of the data branch


def _checks_only(db_dir: str, head: str) -> bool:
    """G6: origin past head by card-checks.yml's commits alone, no other writer: the data commit goes on top of them."""
    from commands.card_db import CHECKS_JSON
    try:
        _git(db_dir, "fetch", "-q", "origin", _data_branch(db_dir))
        return bool(head) and _git(db_dir, "diff", "--name-only", f"{head}...FETCH_HEAD").split() == [f"db/{CHECKS_JSON}"]
    except (CalledProcessError, OSError):
        return False


def _remote_tip(db_dir: str) -> Optional[str]:
    """
    G6: the data branch's tip on origin now (git ls-remote, no API call): "" when the branch isn't there yet, None
    when db/ has no origin (a hand run in a plain folder) and so no other writer to check.
    """
    try:
        _git(db_dir, "remote", "get-url", "origin")
    except (CalledProcessError, OSError):
        return None
    branch = _data_branch(db_dir)
    done = run(["git", "-C", db_dir, "ls-remote", "--exit-code", "origin", f"refs/heads/{branch}"], capture_output=True,
               env={**environ, **NO_LAZY})
    if done.returncode == 2:
        return ""
    if done.returncode != 0:
        raise CalledProcessError(done.returncode, ["git", "ls-remote", "origin", branch], done.stdout, done.stderr)
    return done.stdout.decode().split()[0]


def _builder_commit() -> str:
    try:
        return _git(path_join(dirname(abspath(__file__)), ".."), "rev-parse", "HEAD").strip()
    except (CalledProcessError, OSError):
        return environ.get("BUILDER_SHA", "")


def _commits(checkout: str, base: Optional[str], head: str) -> list:
    """(sha, committer time, author as .mailmap maps it) of base..head, oldest first (head's last 100 without a base)."""
    from commands.card_history import mailmap_args
    span = [f"{base}..{head}"] if base else ["--max-count=100", head]
    out = []
    log = _git(checkout, *mailmap_args(checkout, head), "log", "--reverse", "--format=%H%x09%ct%x09%aN", *span)
    for line in log.splitlines():
        sha, at, author = (line.split("\t", 2) + ["", ""])[:3]
        if sha:
            out.append((sha, float(at or 0), author))
    return out


def _activity(repository: str) -> list:
    """The last pushes to the card branch (one API call), newest first, as GitHub lists them."""
    from commands.github_api import HEADERS, api_url, get_with_retry
    reply = get_with_retry(api_url(repository, "activity"), timeout=30, headers=HEADERS,
                           params={"ref": f"refs/heads/{CARD_BRANCH}", "activity_type": "push", "per_page": 100})
    reply.raise_for_status()
    return reply.json()


def _epoch(stamp: Optional[str]) -> Optional[float]:
    try:
        return datetime.strptime(stamp or "", "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()
    except ValueError:
        return None


def pushes(repository: str, checkout: str, commits: list) -> list:
    """
    The activity list's pushes that brought this build's commits, oldest first, each with the commits it holds
    (git rev-list before..after in the clone); [] when the list can't be read (the fallback groups by author).
    """
    try:
        listed = _activity(repository)
    except (RequestException, ValueError) as e:
        print(f"Activity list not read ({e}); pushes told by consecutive authors")
        return []
    new = {sha for sha, _, _ in commits}
    out = []
    for item in reversed(listed if isinstance(listed, list) else []):
        after, before = item.get("after") or "", item.get("before") or ""
        if after not in new:
            continue
        try:
            held = _git(checkout, "rev-list", f"{before}..{after}" if before.strip("0") else after).split()
        except CalledProcessError:                   # a before git doesn't have (a force push)
            held = [after]
        out.append({"after": after, "before": before or None, "by": (item.get("actor") or {}).get("login"),
                    "at": _epoch(item.get("timestamp")), "commits": held})
    return out


def build(repository: str, checkout: str, db_dir: str, head: str = "HEAD", blobs: str = "", empty: bool = False,
          full: bool = False, cards: str = "", allow_archive: bool = False, budget: int = BUDGET, run_url: str = "",
          dry_run: bool = False, restore: str = "", builder_ref: str = "",
          baked: bool = False, own: str = "", verdicts: Optional[dict] = None, side: Optional[list] = None,
          guard_docs: Optional[list] = None, records: Optional[list] = None) -> Outcome:
    """
    One build (design section 7.1): db/ before, the tree and index from git, the
    changed PSDs (and last run's held ones), authors, edits, duplicates, the archive dated from git, the versions, the
    changelog events with their pushes, the gates, then db/ written with its manifest.

    Args:
        repository: owner/name of the card repo (the REST fallback, the activity list, the manifest)
        checkout: A full clone of it (blobless is enough)
        db_dir: The db/ folder (read, then written)
        head: The card repo commit to build
        blobs: A clone whose objects hold the PSD blobs (the Mac's working clone); tried first
        empty: Start from an empty db/ (new card data; else an empty one is refused)
        full: Re-parse every card (also G4's and G5's approval, kept in pending.full until the re-parse is done)
        cards: Comma-separated names to re-parse
        allow_archive: Lift G2
        budget: REST downloads at most
        run_url: The Actions run (manifest, commit message)
        dry_run: Build and report, write nothing
        restore: A data commit whose db/ becomes the current one again
        builder_ref: The AutoCroissant ref the builder was checked out at (the manifest)
        baked: head is the run's local bake commit B (a push of its own, by the card sync)
        own: Comma-separated main commits the card sync pushed (the bake's M): their pushes are the card sync's
        verdicts: The bake's verdicts for db/verdicts.json ({key: verdict, or None to remove}; card_db.merge_verdicts)
        side: The bake's events (baked, bake-refused), each with its commit (M) when it has one
        guard_docs: The guards' guard-events.json files ({"commit", "push", "events"}): their events, by that commit
        records: What the guards, the bake and the link redraw read off PSDs before rewriting them (psd_edit.records_of,
            oldest first): merged into db/'s render and spot records

    Returns:
        The Outcome (code 0 written or nothing to do, 2 a gate refused, 1 files not fetched)
    """
    from commands import changelog, psd_details
    from commands.card_db import check_db, read_db, round_trip_problems, write_db
    from commands.font_check import fonts_report
    from commands.card_history import (Refined, allows_high, close_archived, date_archive, kept_values, out_of_range,
                                       record_versions, unchecked, unknown_values)
    out = Outcome()
    data_head = _data_head(db_dir)
    stats, old, details, manifest_before = read_db(db_dir)
    links_before = read_links(db_dir)
    links = links_before
    colors_before = json_loads(_text(db_dir, STATUS_COLORS) or "{}")
    colors = colors_before
    from commands.psd_edit import merge_records, set_records
    render, spot = merge_records(*read_records(db_dir), records or [])
    set_records(render, spot)                                 # this build's parses read them (psd_details' type spots)
    if not stats and not empty:
        raise ValueError(f"{db_dir} holds no cards: give --empty to start new card data")
    pending_before = (manifest_before.get("pending") or {}) if not restore else {}
    carried = pending_before.get("full") or None              # a budgeted full re-parse going on
    if carried and carried.get("parser_version") != PARSER_VERSION:
        carried = None                                        # a newer parser needs its own approval
    approved = full or bool(carried)
    db = StatsDatabase()
    db.stats, db.old_stats = stats, old
    run_build = GitBuild(db, checkout, head, blobs=blobs, budget=budget)
    index = run_build.index
    base = manifest_before.get("card_commit")
    if base and run(["git", "-C", checkout, "cat-file", "-e", f"{base}^{{commit}}"], capture_output=True).returncode != 0:
        base = None
    pictures, measured = png_changes.read_record(db_dir), ""
    beside = None if restore else png_changes.Beside(     # the run's picture changes (owner 11-09), beside the PSD reads
        checkout, pictures["pending"] + (png_changes.card_pairs(checkout, run_build.head, base) if base else []), pictures,
        limit=png_changes.RUN_MAX)
    restored_manifest: dict = {}
    if restore:
        db.stats, db.old_stats, details, restored_manifest = _restore(db_dir, restore)
    before, before_old = deepcopy(db.stats), deepcopy(db.old_stats)
    archived_before = _archived(before, before_old)
    details_before = len(details)

    if baked:
        run_build.pushed = _git(checkout, "rev-parse", f"{head}^").strip()   # B is local; its other PSDs are its parent's
    run_build.only = {CardAuthors.key(c) for c in (cards or "").split(",") if c.strip()}
    run_build.again = set(pending_before.get("stale") or [])  # held by the last run's budget: read now
    run_build.failed_before = {(f.get("path"), f.get("blob")): f for f in pending_before.get("failed") or []}
    run_build.details = details
    edits = authors = None
    book = None                                               # rulebook.json's text (None: a restore keeps db/'s)
    if not restore:
        run_build.traverse_remote(repository, True, full)
        found = {p: run_build._repo_text(p) for p in run_build.shas if p.startswith(RULEBOOK_SRC + "/") and p.endswith(".md")}
        book = rulebook.rulebook_text(rulebook.chapters({p: t for p, t in found.items() if t is not None}))
        pruning = can_prune(run_build)
        gone, backfilled = db.prune_clean_cards(run_build.at) if pruning else ([], [])
        if pruning:
            date_archive(db.stats, db.old_stats, index, Refined())
            for name in gone:                               # a new card with its blob this run: a rename
                tag = db.old_stats[name][len(db.old_stats[name]) - 1].archived
                if not tag.get("renamed_to") and run_build.new_blobs.get(tag.get("blob") or ""):
                    tag["renamed_to"] = run_build.new_blobs[tag["blob"]]
            close_archived(db.old_stats, list(gone) + list(backfilled), index)
        edits, authors = run_build.file_commit(EDITS_PATH), run_build.file_commit(AUTHORS_PATH)
        record_versions(before, archived_before, db, run_build.reparsed, run_build.restored, index, edits, authors)
        links = gather_links(run_build, links_before)
        colors = status_colors(run_build, colors_before)
        run_build.duplicate_versions()
        db.drop_stale_archives()
        impossible = unknown_values(db.stats, db.old_stats)        # a replaced misparse's values: unknown in history
        if run_build.font_index is not None:
            psd_details.refill(details, run_build.font_index)
        keep = set(run_build.shas.values()) | {c.archived["blob"] for c in db.archived().values() if c.archived.get("blob")}
        keep |= {c.blob_sha for c in db.stats.values() if getattr(c, "blob_sha", None)}
        psd_details.prune(details, keep)
        render, spot = merge_records(render, spot, [], {p for p in run_build.shas if p.lower().endswith(".psd")})
    else:
        gone, impossible = [], []
    if run_build.lost:                                    # a 5xx or rate limit: red, nothing written, the next run retries
        if beside is not None:
            beside.stop()
        out.code = EXIT_ERROR
        out.errors.append(f"not fetched: {', '.join(run_build.lost[:10])}" + (", ..." if len(run_build.lost) > 10 else ""))
        return out

    edits_by = {}
    if run_build.edits is not None:                      # each entry's _-keys: _by, and _bot's fields credited apart
        edits_by = {n: run_build.edits.meta.get(CardAuthors.key(n), {}) for n in db.stats}
    file_commits = {"edited": edits.sha if edits else None, "authors": authors.sha if authors else None}
    new_events = changelog.events(before, db.stats, archived_before, db.archived(), run_build.reparsed,
                                  {n: m for n, m in edits_by.items() if m.get("_by")}, index, file_commits)   # card changes only
    guards = _guard_commits(checkout, base, run_build.head) | {d.get("commit") for d in guard_docs or [] if d.get("commit")}
    new_events = [e for e in new_events if not (e["kind"] == "art" and e.get("commit") in guards)]   # the guards' events say them
    mark_sweeps(checkout, new_events)                     # a sweep's own art and text changes: silent, still on the website
    if beside is not None:
        shown = picture_pairs(new_events, index, run_build.shas, db.stats)
        done, left = beside.join(pictures, list(shown.values()))   # the event pairs it did not cover: first
        pictures["pending"] = left                        # the next run's first
        if done or left:
            measured = f"Picture changes: {done} measured" + (f", {len(left)} for the next run" if left else "")
        mark_minor(new_events, shown, pictures)
    new_events += side_events(db.stats, db.archived(), side, guard_docs)
    if restore:
        new_events = [{"kind": "restore", "card": "", "restored_from": restore}]
    elif new_events:
        commits = _commits(checkout, base, run_build.head)
        mine = {_git(checkout, "rev-parse", sha).strip() for sha in own.split(",") if sha.strip()}
        listed = pushes(repository, checkout, commits)
        for push in listed:                               # GITHUB_TOKEN's push: listed as github-actions[bot]
            if push["after"] in mine:
                push["by"] = changelog.CARD_SYNC
        new_events = changelog.assign_pushes(new_events, listed, commits, base,
                                             ({run_build.head} if baked else set()) | mine, guards)
    for name in sorted(run_build.reparsed):
        card = db.stats.get(name)
        notes = CardValidator.validate(card) if card is not None else []
        if notes:
            out.problems.append({"card": name, "path": card.path, "notes": notes})

    same_blob = [n for n, (o, _, sha) in run_build.reparsed.items() if o is not None and getattr(o, "blob_sha", None) == sha]
    lost = [n for n in gone if not (db.archive_tag(n) or {}).get("renamed_to")     # an exact rename loses no card,
            and not getattr(db.old_stats[n][len(db.old_stats[n]) - 1], "duplicate_of", None)]   # an alt art no card either
    out.refused, out.lifted = gates(before, db.stats, details_before, len(details), lost, same_blob, approved,
                                    run_build.only, allow_archive)
    if carried and not full:
        out.lifted = {g: f"full (approved by {carried.get('approved_run') or 'an earlier run'})" if why == "full" else why
                      for g, why in out.lifted.items()}
    bad = round_trip_problems(db.stats, db.old_stats)
    if bad:
        out.refused["G3"] = f"records change through a round trip: {', '.join(bad[:10])}"
    wrong = sorted({n for n, versions in db.old_stats.items() for v in versions
                    if unchecked(v) and out_of_range(v, allows_high(n, db.stats.get(n)))})
    if wrong:                                             # the range check on history (unknown_values leaves none)
        out.refused["G3"] = "; ".join(filter(None, [out.refused.get("G3"), f"history values out of range: {', '.join(wrong[:10])}"]))

    kinds = [e["kind"] for e in new_events]
    out.counts = {"cards": len(db.stats), "versions": sum(len(v) for v in db.old_stats.values()),
                  "archived": len(db.archived()), "details": len(details), "changed": kinds.count("changed"),
                  "new": kinds.count("new"), "failed": run_build.failed + len(run_build.still_failed), "baked": 0}
    failed = sorted([{"path": u["path"], "blob": u["blob"], "error": u["error"], "parser_version": PARSER_VERSION}
                     for u in run_build.unreadable] + run_build.still_failed, key=lambda f: f["path"])
    stale = sorted(run_build.stale)
    still = None
    if stale and approved:                                # the approval goes on with the re-parse
        still = carried if carried and not full else {"parser_version": PARSER_VERSION, "approved_run": run_url or None}
    posted = changelog.read_posted(db_dir)
    numbered = changelog.number(new_events, max((e.get("seq", 0) for e in changelog.read_events(db_dir)), default=0),
                                posted.get("seq", 0), build=run_url or None, card_commit=run_build.head)
    out.lines = [_plain_line(e) for e in numbered] + ([measured] if measured else [])
    out.manifest = {
        "built_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "run": run_url or None,
        "card_repo": repository, "card_commit": run_build.head, "card_commit_at": run_build.at, "main_commit": None,
        "builder": {"repo": BOT_REPO, "ref": builder_ref or None, "commit": _builder_commit(), "parser_version": PARSER_VERSION,
                    "details_version": psd_details.VERSION, "psd_tools": _version("psd-tools"),
                    "python": ".".join(map(str, version_info[:2]))},
        "inputs": {key: run_build.shas.get(path) for key, path in (("edits_blob", EDITS_PATH), ("authors_blob", AUTHORS_PATH),
                                                                   ("created_blob", CREATED_PATH),
                                                                   ("manifest_blob", ICON_MANIFEST_PATH), ("decks_blob", RULES_PATH),
                                                                   ("types_tree", TYPES_DIR), ("fonts_tree", "MDW/Fonts"))},
        "counts": out.counts,
        "pending": {"stale": stale, "failed": failed, "full": still, "repin": []},
        "problems": out.problems, "gates": {g: f"lifted by {why}" for g, why in out.lifted.items()},
        "restored_from": restore or None, "fonts": fonts_report(details, db.stats)}       # the bot's /fonts
    if restored_manifest:                                 # its card commit and inputs: the next run re-reads what changed
        for key in ("card_commit", "card_commit_at", "inputs"):
            out.manifest[key] = restored_manifest.get(key, out.manifest[key])
        kept = restored_manifest.get("pending") or {}
        out.manifest["pending"] = {"stale": kept.get("stale") or [], "failed": kept.get("failed") or [], "full": None, "repin": []}
    if impossible:
        out.lines.append(f"History values out of range set to unknown: {len(impossible)} "
                         f"({', '.join(sorted({n for n, _, _ in impossible}))})")
    kept = kept_values(db.stats, db.old_stats)
    if kept:                                              # read by parser 2: kept until an allow_high_stats entry
        out.lines.append(f"History values out of range kept (an allow_high_stats entry allows them): {', '.join(kept)}")
    if out.refused:
        out.code = EXIT_GATE
        return out
    if dry_run:
        return out
    tip = _remote_tip(db_dir)
    if tip is not None and tip != data_head and not _checks_only(db_dir, data_head):
        out.refused["G6"] = (f"the data branch is at {tip[:8] or 'nothing'} on origin, this build started from "
                             f"{data_head[:8] or 'nothing'} (another writer)")
        out.code = EXIT_GATE
        return out
    from commands.card_db import merge_verdicts, old_stats_text, stats_text, write_text
    recorded = merge_verdicts(db_dir, verdicts) if verdicts and not restore else []
    if recorded:
        out.lines.append(f"verdicts: {', '.join(recorded[:20])}" + (", ..." if len(recorded) > 20 else ""))
    unchanged = (not restore and not numbered and manifest_before.get("card_commit") == run_build.head
                 and manifest_before.get("pending") == out.manifest["pending"]
                 and stats_text(db.stats) == _text(db_dir, "stats.json") and old_stats_text(db.old_stats) == _text(db_dir, "old_stats.json")
                 and psd_details.text_of(details) == _text(db_dir, "psd_details.json")
                 and links_text(links) == links_text(read_links(db_dir))
                 and all(records_text(r) == _text(db_dir, n) for n, r in zip(RECORDS, (render, spot)))
                 and png_changes.record_text(pictures) == _text(db_dir, png_changes.RECORD) and colors == colors_before
                 and book in (None, _text(db_dir, "rulebook.json")) and manifest_before.get("fonts") == out.manifest["fonts"])
    if unchanged:
        return out
    write_db(db_dir, db.stats, db.old_stats, psd_details.text_of(details), out.manifest, book)
    write_text(path_join(db_dir, LINKS_JSON), links_text(links))
    for name, record in zip(RECORDS + (STATUS_COLORS,), (render, spot, colors)):
        write_text(path_join(db_dir, name), records_text(record))
    png_changes.write_record(db_dir, pictures)
    if numbered:
        changelog.append_events(db_dir, numbered)
    problems = check_db(db_dir)
    if problems:
        out.refused["G3"] = "; ".join(problems)
        out.code = EXIT_GATE
    return out


def _guard_commits(checkout: str, base: Optional[str], head: str) -> set:
    """The guards' commits in base..head (author AutoCroissant, subject "Card guards: ..."): their pushes are GUARDS'."""
    from commands.changelog import BOT, GUARDS
    span = [f"{base}..{head}"] if base else ["--max-count=100", head]
    try:
        log = _git(checkout, "log", "--format=%H%x09%an%x09%s", *span)
    except CalledProcessError:
        return set()
    rows = [(line.split("\t", 2) + ["", ""])[:3] for line in log.splitlines()]
    return {sha for sha, author, subject in rows if author == BOT and subject.startswith(GUARDS + ":")}


def mark_sweeps(checkout: str, new_events: list) -> None:
    """
    Gives "sweep" (the trailer's value) to each art or changed event (changelog.SWEPT) whose own commit carries a
    Card-sweep trailer (card_sweep's passes, committed by hand; owner: the guards' fixes are silent) and is the card's
    only change since its stored blob (the blob before that commit is the event's old one), and to each renamed event
    of such a commit (owner 2026-11-11: a rename batch). Any push, any build: a card held for a later run still finds its
    commit; new, archived and moved cards always post.
    """
    from commands.changelog import SWEEP_TRAILER, SWEPT
    candidates = [e for e in new_events if e["kind"] in SWEPT and e.get("commit") and e.get("path")
                  and len(e.get("blob") or ()) == 2 and all(e["blob"])]
    renames = [e for e in new_events if e["kind"] == "renamed" and e.get("commit")]
    if not candidates and not renames:
        return
    try:
        log = _git(checkout, "log", "--no-walk=unsorted", f"--format=%H%x09%(trailers:key={SWEEP_TRAILER},valueonly,separator=%x2C)",
                   *sorted({e["commit"] for e in candidates + renames}))
    except CalledProcessError:
        return
    kinds = {sha: value.strip() for sha, _, value in (line.partition("\t") for line in log.splitlines()) if value.strip()}
    for event in renames:
        if event["commit"] in kinds:
            event["sweep"] = kinds[event["commit"]]
    for sha in sorted({e["commit"] for e in candidates if e["commit"] in kinds}):
        mine = [e for e in candidates if e["commit"] == sha]
        try:                                              # the parent's tree entries: no blob is read
            listing = _git(checkout, "--literal-pathspecs", "ls-tree", "-r", "-z", f"{sha}^", "--", *sorted({e["path"] for e in mine}))
        except CalledProcessError:
            continue
        before = {path: meta.split()[2] for meta, _, path in (entry.partition("\t") for entry in listing.split("\0") if entry)}
        for event in mine:
            if before.get(event["path"]) == event["blob"][0]:
                event["sweep"] = kinds[sha]


def picture_pairs(new_events: list, index, shas: dict, stats: dict) -> dict:
    """{id(event): (PNG blob before, PNG blob now)} of each art and changed event: the card's PNG at the commit that put
    the event's old PSD blob there (index.at_path), and in the tree now; left out when either is unknown."""
    lower = {path.lower(): path for path in shas}
    out = {}
    for event in new_events:
        blobs = event.get("blob") or ()
        if event["kind"] not in ("art", "changed") or len(blobs) != 2 or not all(blobs) or not event.get("path"):
            continue
        card = stats.get(event["card"])
        variant = getattr(card, "variant", None) if card is not None else None
        stem = lambda psd: (duplicate_png(card)[:-4] if isinstance(variant, int) and variant > 1 and psd == card.path
                            else splitext(psd)[0])
        old_psd = event.get("from_path") or event["path"]
        put = next((c for c, blob in index.at_path.get(old_psd, ()) if blob == blobs[0]), None)
        old_png = lower.get(f"{stem(old_psd)}.png".lower()) or f"{stem(old_psd)}.png"
        new_png = lower.get(f"{stem(event['path'])}.png".lower())
        if put is None or new_png is None:
            continue
        before = next((blob for c, blob in index.at_path.get(old_png) or index.at_path.get(new_png, ()) if c.order >= put.order), None)
        if before:
            out[id(event)] = (before, shas[new_png])
    return out


def mark_minor(new_events: list, shown: dict, record: dict) -> None:
    """
    Gives "minor" (changelog.MINOR) to an art event whose picture changed under png_changes.VISIBLE ("picture"), and to
    a changed or edited event whose only change is the ability's formatting (changelog.formatting_only; a changed one's
    picture under VISIBLE too: "text"). Minor events post nothing; the Changes page folds them (owner 11-09).
    """
    from commands.changelog import MINOR, formatting_only
    for event in new_events:
        pair = shown.get(id(event))
        still = pair is not None and not png_changes.visible(record, *pair)
        fields = event.get("fields") or {}
        if event["kind"] == "art" and still:
            event[MINOR] = "picture"
        elif event["kind"] in ("changed", "edited") and fields and set(fields) <= {"ability", "tags"} \
                and formatting_only(*(fields.get("ability") or ("", ""))) and (event["kind"] == "edited" or still):
            event[MINOR] = "text"


class _AsOf:
    """An index's at_path (its get) as a past build saw it: the puts no newer than its head (Commit.order or more)."""
    def __init__(self, at_path: dict, order: int):
        self.source, self.order = at_path, order

    def get(self, path, default=None):
        return [(c, blob) for c, blob in self.source.get(path, ()) if c.order >= self.order] or default


def past_pairs(checkout: str, db_dir: str, head: str = "HEAD") -> tuple:
    """
    db/changelog's art, changed and edited events with no "minor" (from before mark_minor) and picture_pairs of them,
    each at its own build's head (card_commit): the index as of then and that commit's PNGs (png_changes backfill --events).

    Returns:
        (those events, {id(event): (PNG blob before, PNG blob then)})
    """
    from types import SimpleNamespace
    from commands import changelog
    from commands.card_db import parse_stats
    from commands.card_history import GitIndex
    events = [e for e in changelog.read_events(db_dir)
              if e["kind"] in ("art", "changed", "edited") and changelog.MINOR not in e]
    with open(path_join(db_dir, "stats.json"), encoding="utf-8") as fh:
        stats = parse_stats(fh.read())
    index, shown = GitIndex.from_git(checkout, head), {}
    by_commit: dict = {}
    for event in events:
        if event["kind"] != "edited" and event.get("card_commit"):
            by_commit.setdefault(event["card_commit"], []).append(event)
    for sha, mine in by_commit.items():
        try:                                              # its newest PSD or PNG commit: where the index stood
            last = index.by_sha.get(_git(checkout, "log", "-1", "--format=%H", sha, "--", "*.psd", "*.png").strip())
            tree = {e["path"]: e["sha"] for e in ls_tree(checkout, sha)["tree"] if e["type"] == "blob"}
        except CalledProcessError:
            continue
        if last is not None:
            shown.update(picture_pairs(mine, SimpleNamespace(at_path=_AsOf(index.at_path, last.order)), tree, stats))
    return events, shown


def mark_past(db_dir: str, events: list, shown: dict, record: dict) -> int:
    """mark_minor on past_pairs' events, their month files rewritten in place (every seq and the posted cursor kept).
    Returns the events marked."""
    from commands import changelog
    mark_minor(events, shown, record)
    marks = {e["seq"]: e[changelog.MINOR] for e in events if e.get(changelog.MINOR)}
    if marks:
        changelog.rewrite_events(db_dir, lambda e: {**e, changelog.MINOR: marks[e["seq"]]}
                                 if e.get("seq") in marks and changelog.MINOR not in e else e)
    return len(marks)


def side_events(stats: dict, archived: dict, side: Optional[list], guard_docs: Optional[list]) -> list:
    """
    The bake's and the guards' events as the changelog's: kind, card, the card's title, kind of card, stars and path
    from the data (an archived card's last version, else the path the event names), and the commit that brought them.
    """
    from commands.changelog import KINDS
    out = []
    raw = [dict(e) for e in side or []]
    for doc in guard_docs or []:
        raw += [{**e, "commit": doc.get("commit")} for e in doc.get("events") or [] if isinstance(e, dict)]
    for e in raw:
        name = e.get("card")
        if e.get("kind") not in KINDS or not isinstance(name, str) or not name:
            continue
        card = stats.get(name) or archived.get(name)
        known = {"title": getattr(card, "display_name", None) or name, "card_type": card.card_type,
                 "stars": getattr(card, "stars", None), "path": card.path} if card is not None else {"title": name}
        out.append({**{k: v for k, v in e.items() if v is not None}, **known, "path": known.get("path") or e.get("path")})
    return out


def rewrite_commit(db_dir: str, checkout: str, old: str, new: str, message: str = "") -> dict:
    """
    Stage 4's B-to-M rewrite: the run's local bake commit B became main commit M (the same card files plus the save),
    so every reference to B in db/ (version starts and ends, PNG pins, archive tags, card dates, the events, their
    push and the manifest) becomes M, B's time M's. The manifest gains main_commit, counts.baked (the PSDs B changed)
    and pending.repin (the PNGs B changed: their save links are pinned by the next run). The data commit's message
    file, if given, names M too.

    Returns:
        The manifest written
    """
    from commands import changelog, psd_details
    from commands.card_db import read_db, write_db
    old, new = _git(checkout, "rev-parse", old).strip(), _git(checkout, "rev-parse", new).strip()
    old_at = float(_git(checkout, "log", "-1", "--format=%ct", old).strip())
    new_at = float(_git(checkout, "log", "-1", "--format=%ct", new).strip())
    stats, old_stats, details, manifest = read_db(db_dir)

    def fix(card) -> None:
        info = getattr(card, "version", None) or {}
        for sha_key, at_key in (("from_commit", "from"), ("to_commit", "to")):
            if info.get(sha_key) == old:
                info[sha_key], info[at_key] = new, new_at
        if info.get("png_commit") == old:
            info["png_commit"] = new
        if info.get("from_commit") == new and card.timestamp == old_at:
            card.timestamp = new_at                  # a card baked in the run is dated by M
        tag = getattr(card, "archived", None)
        if tag and tag.get("commit") == old:
            tag["commit"], tag["at"] = new, new_at
    for card in stats.values():
        fix(card)
    for versions in old_stats.values():
        for card in versions:
            fix(card)
    for card in stats.values():                      # a duplicate keeps its original's date
        original = stats.get(getattr(card, "duplicate_of", None) or "")
        if original is not None:
            card.timestamp = original.timestamp

    def fix_event(event: dict) -> dict:
        for key in ("card_commit", "commit"):
            if event.get(key) == old:
                event[key] = new
        push = event.get("push") or {}
        if push.get("after") == old:
            push.update(after=new, at=new_at, by=changelog.CARD_SYNC)
        return event
    changelog.rewrite_events(db_dir, fix_event)
    changed = [line.split("\t", 1) for line in _git(checkout, "diff-tree", "--no-commit-id", "-r", "--name-status", old).splitlines()]
    if manifest.get("card_commit") == old:
        manifest["card_commit"], manifest["card_commit_at"] = new, new_at
    manifest["main_commit"] = new
    manifest.setdefault("counts", {})["baked"] = sum(1 for _, path in changed if path.lower().endswith(".psd"))
    manifest.setdefault("pending", {})["repin"] = sorted(path for status, path in changed
                                                          if path.lower().endswith(".png") and status != "D")
    if message and isfile(message):
        with open(message, encoding="utf-8") as fh:
            text = fh.read()
        with open(message, "w", encoding="utf-8") as fh:
            fh.write(text.replace(old, new).replace(old[:8], new[:8]))
    return write_db(db_dir, stats, old_stats, psd_details.text_of(details), manifest)


def _text(folder: str, name: str) -> Optional[str]:
    path = path_join(folder, name)
    if not isfile(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def _version(package: str) -> str:
    try:
        from importlib.metadata import version
        return version(package)
    except Exception:
        return ""


def _json_file(path: str):
    """A JSON file's data; ValueError when it is not JSON (the build then fails red)."""
    with open(path, encoding="utf-8") as fh:
        return json_loads(fh.read())


def _plain_line(event: dict) -> str:
    """One event for the commit message and the job summary."""
    line = _event_line(event)
    return f"{line} (minor: {event['minor']})" if event.get("minor") else line


def _event_line(event: dict) -> str:
    kind, name = event["kind"], event.get("card") or ""
    if kind in ("changed", "edited", "reparse", "authors"):
        fields = ", ".join(k for k in (event.get("fields") or {}) if k != "tags")
        verdict = f" ({event['verdict']})" if event.get("verdict") else ""
        return f"{kind}: {name}: {fields}{verdict}"
    if kind == "renamed":
        return f"renamed: {name} -> {event.get('renamed_to')}"
    if kind == "restore":
        return f"restore: db/ of {event.get('restored_from')}"
    if kind in ("baked", "bake-refused", "guard-png", "guard-fix") and (event.get("reason") or event.get("line")):
        return f"{kind}: {name}: {event.get('reason') or event.get('line')}"
    return f"{kind}: {name}"


def subject(out: Outcome, head: str) -> str:
    """The data commit's subject: "Card data: 2 changed, 1 new, 1 archived (main 1a2b3c4d)"; a rollback names its
    data commit."""
    if out.manifest.get("restored_from"):
        return f"Card data: restored from {out.manifest['restored_from'][:8]} (main {head[:8]})"
    kinds = [line.split(":", 1)[0] for line in out.lines if ":" in line]
    parts = [f"{kinds.count(k)} {k}" for k in ("changed", "new", "archived", "renamed", "restored", "edited", "authors", "art",
                                                "moved", "reparse", "baked", "bake-refused", "guard-png", "guard-fix")
             if kinds.count(k)]
    return f"Card data: {', '.join(parts) or 'refreshed'} (main {head[:8]})"


def summary(out: Outcome, head: str) -> str:
    """The job summary (Markdown): counts, refused and lifted gates, one line per card, the validator's notes."""
    rows = " | ".join(str(out.counts.get(k, "")) for k in ("cards", "versions", "archived", "details", "changed", "new", "failed"))
    title = "Card data build failed" if out.errors else subject(out, head)
    text = [f"## {title}", "", "| cards | versions | archived | details | changed | new | failed |",
            "|---|---|---|---|---|---|---|", f"| {rows} |", ""]
    text += [f"- **Error:** {why}; nothing written, the next run tries again" for why in out.errors]
    text += [f"- **{gate} refused:** {why}" for gate, why in out.refused.items()]
    text += [f"- {gate} lifted by {why}" for gate, why in out.lifted.items()]
    if out.lines:
        text += ["", "### Cards", ""] + [f"- {line}" for line in out.lines]
    if out.problems:
        text += ["", "### Parser notes", ""] + [f"- {p['card']}: {', '.join(p['notes'])}" for p in out.problems]
    return "\n".join(text) + "\n"


def main(argv: Optional[list] = None) -> int:
    """The CLI; exit 0 written or nothing to do, 2 a gate refused, 1 an error (a file not fetched included)."""
    parser = ArgumentParser(prog="python -m commands.card_data",
                            description="Build the card data (db/ JSON) from a full clone of the card repo.")
    parser.add_argument("--repo", default="MichaelJSr/TTSCardMaker", help="owner/name of the card repo")
    parser.add_argument("--checkout", required=True, help="a full clone of it (blobless is enough; never shallow)")
    parser.add_argument("--head", default="HEAD", help="the commit to build (default HEAD)")
    parser.add_argument("--blobs", default="", help="a clone holding the PSD blobs, tried first (the Mac's working clone)")
    parser.add_argument("--db", required=True, help="the db/ folder, read then written")
    parser.add_argument("--empty", action="store_true", help="start from an empty db/ (new card data)")
    parser.add_argument("--full", action="store_true", help="re-parse every card (approves gates G4 and G5)")
    parser.add_argument("--cards", default="", help="comma-separated card names to re-parse")
    parser.add_argument("--allow-archive", action="store_true", help="lift gate G2 (more cards archived than usual)")
    parser.add_argument("--restore", default="", help="a data commit whose db/ is written back as the current one")
    parser.add_argument("--budget", type=int, default=BUDGET, help=f"REST downloads at most (default {BUDGET})")
    parser.add_argument("--run", default="", help="the Actions run URL (manifest, commit message)")
    parser.add_argument("--builder-ref", default="", help="the AutoCroissant ref of this builder (manifest)")
    parser.add_argument("--summary", default="", help="write the job summary (Markdown) here")
    parser.add_argument("--message", default="", help="write the data commit's message here (--rewrite: names M in it)")
    parser.add_argument("--dry-run", action="store_true", help="build and report; write no db/")
    parser.add_argument("--baked", action="store_true", help="--head is the run's local bake commit (its own push)")
    parser.add_argument("--own", default="", metavar="SHA", help="main commits the card sync pushed (comma separated): "
                                                                  "their pushes are the card sync's")
    parser.add_argument("--verdicts", default="", metavar="FILE", help="the bake's verdicts ({\"cards\": {key: verdict or "
                                                                        "null}}) merged into db/verdicts.json")
    parser.add_argument("--events", default="", metavar="FILE", help="the bake's events (a JSON list), each with its commit")
    parser.add_argument("--guard-events", default="", metavar="FILE", help="the guards' guard-events.json files (a JSON list)")
    parser.add_argument("--records", default="", metavar="FILE", help="records read off PSDs before a rewrite (a JSON list of "
                                                                       "{\"render\", \"spot\"}, oldest first) for db/")
    parser.add_argument("--rewrite", nargs=2, metavar=("B", "M"), help="no build: db/'s references to the bake commit B "
                                                                        "become main commit M (after the main push)")
    args = parser.parse_args(argv)
    from commands.github_api import token_from_env
    token_from_env()
    try:
        if args.rewrite:
            manifest = rewrite_commit(args.db, args.checkout, args.rewrite[0], args.rewrite[1], args.message)
            print(f"db/ now names {manifest['main_commit'][:8]} for {args.rewrite[0][:8]}; "
                  f"{len(manifest['pending']['repin'])} PNGs to re-pin.")
            return EXIT_OK
        verdicts = (_json_file(args.verdicts).get("cards") or {}) if args.verdicts else None
        side = _json_file(args.events) if args.events else []
        docs = _json_file(args.guard_events) if args.guard_events else []
        found = _json_file(args.records) if args.records else []
        out = build(args.repo, args.checkout, args.db, args.head, args.blobs, args.empty, args.full, args.cards,
                    args.allow_archive, args.budget, args.run, args.dry_run, args.restore, args.builder_ref,
                    args.baked, args.own, verdicts, side.get("events", []) if isinstance(side, dict) else side,
                    docs if isinstance(docs, list) else [docs], found if isinstance(found, list) else [found])
    except (CalledProcessError, OSError, ValueError, RequestException) as e:
        print(f"Card data build failed: {e}")
        return EXIT_ERROR
    head = out.manifest.get("card_commit", "")
    print(summary(out, head), end="")
    if args.summary:
        with open(args.summary, "a", encoding="utf-8") as fh:
            fh.write(summary(out, head))
    if args.message and out.manifest:
        with open(args.message, "w", encoding="utf-8") as fh:
            fh.write("\n".join([subject(out, head), ""] + out.lines + ["", f"Builder: {BOT_REPO}@{out.manifest['builder']['commit']}"]
                               + ([f"Build: {args.run}"] if args.run else [])) + "\n")
    return out.code


# ========================
# Module Exports
# ========================
__all__ = [
    'BUDGET',
    'DOWNLOAD_WORKERS',
    'EXCLUDE_FOLDERS',
    'EXIT_ERROR',
    'EXIT_GATE',
    'EXIT_OK',
    'PROGRESS',
    'RANGED_WORKERS',
    'SPARSE_FILES',
    'UPDATE_RATE',
    'CardBuild',
    'GitBuild',
    'Outcome',
    'build',
    'can_prune',
    'gates',
    'ls_tree',
    'main',
    'fetch_blobs',
    'gather_links',
    'linked_file',
    'links_text',
    'mark_past',
    'mark_sweeps',
    'past_pairs',
    'pixel_hash',
    'read_links',
    'read_records',
    'records_text',
    'pushes',
    'rewrite_commit',
    'side_events',
    'subject',
    'summary',
]


if __name__ == "__main__":
    raise SystemExit(main())
