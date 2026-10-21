"""
/update_stats: RepositoryTraverser reads the card repo's tree on GitHub, re-parses the PSDs whose blob changed, then
applies the authors file, Card Edits.json and the duplicate PNGs; update_stats saves the result and queues the
follow-ups (type emojis, the TTS save, the card list).
"""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
from discord import Interaction
from json import loads as json_loads
from os import fdopen, listdir, remove, walk
from os.path import basename, isdir, isfile, splitext, join as path_join
from re import IGNORECASE, match as re_match
from requests.exceptions import RequestException
from tempfile import mkstemp
from threading import local
from typing import Optional

from global_config import AUTHORS_PATH, EDITS_PATH, ICON_MANIFEST_PATH, RULEBOOK_DIR, STARS_DIR, TYPES_DIR
from commands.card_edits import CardAuthors, CardEdits
from commands.card_models import CardInfo, StatsDatabase, card_name, stats_db
from commands.github_api import GIT_TOKEN, blob_text, fetch_blob, fetch_tree
from commands.psd_parser import CardValidator, IconVocabulary, PSDParser
from commands.query_card import card_repo
from commands.repo_history import refresh as refresh_history
from commands.tts_save import save_affected, update_after_stats
from commands.type_emojis import sync_changed_emojis
from commands.utils import queue_command, queue_edit, say

# ========================
# Configuration
# ========================
UPDATE_RATE = 25
DOWNLOAD_WORKERS = 4                                              # PSDs downloaded ahead of parsing
LOOKUP_WORKERS = 2                                                # commit lookups ahead of parsing
COMMITS_PAGE = 100                                                # commits per request: newest and first in one
EXCLUDE_FOLDERS = ["Markers", "MDW"]

# ========================
# Global State
# ========================
_tree_etags: dict = {}                                            # repository -> ETag of the last update's tree
_file_cache: dict = {}                                            # repo path -> (blob sha, text) of small files


class RepositoryTraverser:
    """Traverses repository to update card stats."""
    def __init__(self, db: StatsDatabase):
        self.db = db
        self.parser: Optional[PSDParser] = None
        self._local = local()                    # a PyGithub client per thread: one client isn't thread-safe
        self._lookups: dict = {}                 # repo path -> Future of its commit page
        self.edits: Optional[CardEdits] = None
        self.changed = 0                         # cards changed without a re-parse (authors, edits, duplicates)
        self.only: set = set()                   # card names to re-parse anyway (/update_stats cards:)
        self.repository = ""
        self.shas: dict = {}                     # repo path -> blob sha, from the tree
        self.tree: Optional[dict] = None         # the tree, reused by card_repo.populate_files
        self.etag = ""                           # its ETag, so card_repo's next fetch can be a 304
        self._recent: Optional[set] = None

    def _update_old_stats_paths(self) -> None:
        """
        Update paths in old_stats to match current paths.
        This ensures historical entries reflect where the card is now located.
        """
        for name in self.db.old_stats:
            if name in self.db.stats:
                current_path = self.db.stats[name].path
                # Update all historical entries to use the current path
                for old_card in self.db.old_stats[name]:
                    old_card.path = current_path

    def _psd_changed(self, name: str, path: str, sha: Optional[str]) -> bool:
        """Changed = new, moved or a different blob. Cards saved before blob shas: committed since the last update."""
        card = self.db.stats.get(name)
        if card is None or card.path != path:
            return True
        if getattr(card, "blob_sha", None):
            return card.blob_sha != sha
        recent = self._committed_since_update()
        if recent is None or path in recent:
            return True
        card.blob_sha = sha                       # unchanged: remember its blob
        self.changed += 1
        return False

    def _committed_since_update(self) -> Optional[set]:
        """Paths committed after the newest stored card timestamp (one pass over those commits), or None."""
        if self._recent is None:
            newest = max((c.timestamp for c in self.db.stats.values() if (c.timestamp or 0) > 0), default=0)
            if not newest:
                return None
            from github import Github
            repo = Github(login_or_token=GIT_TOKEN).get_repo(self.repository)
            self._recent = {f.filename for commit in repo.get_commits(since=datetime.fromtimestamp(newest, timezone.utc))
                            for f in commit.files}
        return self._recent

    def _repo(self):
        """This thread's repo (its own client and connection; lazy: no request)."""
        repo = getattr(self._local, "repo", None)
        if repo is None:
            from github import Github
            repo = Github(login_or_token=GIT_TOKEN, per_page=COMMITS_PAGE).get_repo(self.repository, lazy=True)
            self._local.repo = repo
        return repo

    def _commit_page(self, path: str) -> Optional[list]:
        """A file's newest commits (all of them up to COMMITS_PAGE): one request."""
        repo = self._repo()
        return repo.get_commits(path=path).get_page(0) if repo else None

    def _blob(self, sha: str, stream: bool = False):
        """A file's exact content by blob sha (raw.githubusercontent can lag behind the tree)."""
        return fetch_blob(self.repository, sha, stream=stream, timeout=120)

    def _repo_text(self, path: str) -> Optional[str]:
        """A small repo file's text; downloaded again only when its blob changed."""
        sha = self.shas.get(path)
        if sha is None:
            return None
        cached = _file_cache.get(path)
        if not cached or cached[0] != sha:
            _file_cache[path] = cached = (sha, blob_text(self.repository, sha, timeout=120))
        return cached[1]

    def _download(self, path: str, sha: str) -> Optional[str]:
        """A PSD into a temp file, or None."""
        name, done = None, False
        try:
            with self._blob(sha, stream=True) as reply:
                fd, name = mkstemp(suffix=".psd")
                with fdopen(fd, "wb") as fh:
                    for chunk in reply.iter_content(1 << 20):
                        fh.write(chunk)
            done = True
            return name
        except (RequestException, OSError) as e:
            print(f"{path}: download failed ({e})")
            return None
        finally:
            if not done and name is not None and isfile(name):      # partial download
                remove(name)

    def traverse_remote(self,
                        repository: str,
                        interaction: Optional[Interaction] = None,
                        output_problematic: bool = True,
                        force_update: bool = False) -> tuple[list[str], int]:
        """
        Traverse remote GitHub repository.

        Args:
            repository: Repository name (owner/repo)
            interaction: Optional Discord interaction for progress updates
            output_problematic: Whether to output problematic cards

        Returns:
            List of formatted problem card messages
        """
        self.repository = repository
        etag = _tree_etags.get(repository) if not force_update and not self.only else None
        resp = fetch_tree(repository, etag or "", retry=False)      # 304 when nothing changed (free)
        if resp.status_code == 304:
            say(interaction, "Nothing changed in the card repo since the last update.")
            return ([], 0)
        resp.raise_for_status()
        self.tree = resp.json()                         # parsed once; the helpers below read it
        self.etag = resp.headers.get("ETag") or ""
        self.shas = {item.get("path", ""): item.get("sha") for item in self.tree.get("tree", [])}

        self._populate_types_from_response(resp)
        self.parser = PSDParser(self.db.all_types, self.db, self._icons_from_response(resp, repository))

        self.edits = self._edits_from_response(resp, repository)
        problems, num_changed = self._process_files_from_response(resp, repository, interaction, output_problematic, force_update)
        self.changed += self._apply_authors(self._authors_from_response(resp, repository))
        self.changed += self._apply_edits()
        self.changed += self._sync_duplicates(resp)
        self.changed += self._apply_edits()             # the duplicates' own edits

        # Update old_stats paths after processing all files
        self._update_old_stats_paths()
        _tree_etags[repository] = resp.headers.get("ETag")
        queue_command(_refresh_links, repository)       # tree fetched: re-pin image links

        return (problems, num_changed)

    def _tree_items(self, response) -> list:
        """The tree's entries: self.tree once traverse_remote parsed it, else the response's."""
        return (self.tree if self.tree is not None else response.json()).get("tree", [])

    def _populate_types_from_response(self, response) -> None:
        """Populate card types from API response."""
        self.db.all_types.clear()
        for item in self._tree_items(response):
            path = item.get("path", "")
            if path.startswith(TYPES_DIR) and "Stars" not in path and '.' in path:
                self.db.all_types.append(splitext(basename(path))[0].lower())

    def _populate_types_from_local(self, local_path: str) -> None:
        """Populate card types from a local clone (offline diagnostics; /update_stats itself only reads GitHub)."""
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
                print(f"Could not read {ICON_MANIFEST_PATH}, icon aliases and codepoints unavailable: {e}")
        return IconVocabulary.build(self.db.all_types, stars, manifest)

    def _authors_from_response(self, response, repository: str) -> Optional[CardAuthors]:
        """The card repo's authors file (AUTHORS_PATH), or None if it's missing or unreadable."""
        paths = [item.get("path", "") for item in self._tree_items(response)]
        if AUTHORS_PATH not in paths:
            return None
        try:
            return CardAuthors.parse(self._repo_text(AUTHORS_PATH))
        except RequestException as e:
            print(f"Could not read {AUTHORS_PATH}; authors stay as they are: {e}")
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
        if EDITS_PATH not in paths:
            return None
        try:
            return CardEdits.parse(self._repo_text(EDITS_PATH))
        except (RequestException, ValueError) as e:
            print(f"Could not read {EDITS_PATH}; edits stay as they are: {e}")
            return None

    def _sync_duplicates(self, response) -> int:
        """
        Duplicates: <Name>N.png next to <Name>.psd with no PSD of its own (A_Snack2.png "Breakfast"). Each is
        recorded as a copy of the original's card with duplicate_of set. Returns how many entries changed.
        """
        paths = [item.get("path", "") for item in self._tree_items(response)]
        lower = {p.lower() for p in paths}
        found = {}
        for path in paths:
            m = re_match(r"(.+?)(\d+)\.png$", path, IGNORECASE)
            if (not m or any(folder in EXCLUDE_FOLDERS for folder in path.split("/")) or RULEBOOK_DIR in path
                    or f"{m.group(1)}{m.group(2)}.psd".lower() in lower or f"{m.group(1)}.psd".lower() not in lower):
                continue
            original = basename(m.group(1)).replace("_", " ")
            if original in self.db.stats and not self.db.stats[original].duplicate_of:
                found[card_name(path)] = (original, int(m.group(2)), basename(path).lower())
        before = {n: repr(c) for n, c in self.db.stats.items() if getattr(c, "duplicate_of", None)}
        for name in before:
            if name not in found:
                del self.db.stats[name]
        siblings: dict = {}
        for name, (original, number, _) in found.items():
            siblings.setdefault(original, []).append(number)
        for name, (original, number, png) in found.items():
            src = self.db.stats[original]
            card = deepcopy(src)
            card.name, card.duplicate_of, card.parsed_values, card.problems, card.alt_names = name, original, None, [], None
            card.display_name = self._duplicate_name(png, src, number, sorted(siblings[original]))
            self.db.stats[name] = card
            self.db.seen_cards.append(name)             # not pruned as a missing PSD
        return sum(1 for n in found if before.get(n) != repr(self.db.stats[n])) + sum(1 for n in before if n not in found)

    @staticmethod
    def _duplicate_name(png: str, original: CardInfo, number: int, numbers: list) -> str:
        """The /alias pointing at this PNG, else the PSD's hidden title in order, else the original's name."""
        aliases = [a for a, target in card_repo.git_file_alias.items() if target.lower() == png]
        if aliases:
            return " ".join(w[:1].upper() + w[1:] for w in max(aliases, key=len).replace("_", " ").split())
        titles = getattr(original, "alt_names", None) or []
        if len(titles) == len(numbers):
            return titles[numbers.index(number)]
        return original.display_name or original.name

    def _apply_edits(self) -> int:
        """Apply the website fixes to every card. Returns how many changed."""
        if self.edits is None:
            return 0
        changed = sum(1 for name, card in self.db.stats.items() if self.edits.apply(name, card))
        if changed:
            print(f"{changed} cards changed by {EDITS_PATH}.")
        return changed

    def _icons_from_local(self, local_path: str) -> IconVocabulary:
        """Icon vocabulary from a local clone: Types/ and Types/Stars/ names plus the font's manifest, if any."""
        stars_dir = path_join(local_path, STARS_DIR)
        stars = [self._star_name(f) for f in listdir(stars_dir) if f.lower().endswith(".png")] if isdir(stars_dir) else []
        manifest = None
        manifest_path = path_join(local_path, ICON_MANIFEST_PATH)
        if isfile(manifest_path):
            try:
                with open(manifest_path, encoding="utf-8") as fh:
                    manifest = json_loads(fh.read())
            except (OSError, ValueError) as e:
                print(f"Could not read {manifest_path}, icon aliases and codepoints unavailable: {e}")
        return IconVocabulary.build(self.db.all_types, stars, manifest)

    def _process_files_from_response(self,
                                     response,
                                     repository: str,
                                     interaction: Optional[Interaction],
                                     output_problematic: bool,
                                     force_update: bool) -> tuple[list[str], int]:
        """Re-parse the PSDs whose blob changed (all with force_update). Unchanged cards cost no requests."""
        problematic_cards = []
        num_updated = num_changed = num_old = num_moved = 0
        todo = []
        for item in self._tree_items(response):
            path = item.get("path", "")
            if not path.endswith('.psd') or any(folder in EXCLUDE_FOLDERS for folder in path.split('/')):
                continue
            name = card_name(path)
            self.db.seen_cards.append(name)
            if force_update or CardAuthors.key(name) in self.only or self._psd_changed(name, path, item.get("sha")):
                todo.append((path, name, item.get("sha")))
            else:
                num_old += 1

        pool = ThreadPoolExecutor(DOWNLOAD_WORKERS)
        downloads = [pool.submit(self._download, path, sha) for path, _, sha in todo]
        lookups = ThreadPoolExecutor(LOOKUP_WORKERS)    # commit lookups overlap the downloads too
        self._lookups = {path: lookups.submit(self._commit_page, path) for path, _, _ in todo}
        try:
            for (path, name, sha), download in zip(todo, downloads):
                local_file = download.result()
                if local_file is None:
                    continue
                try:
                    timestamp, author = self._get_remote_timestamp(self._repo(), path, name)
                    card = self.parser.parse(local_file, path)
                finally:
                    remove(local_file)
                old = self.db.stats.get(name)
                if old is None:
                    num_changed += 1
                else:
                    if old.path != path or getattr(old, "blob_sha", None) != sha:
                        self.db.old_stats[name].append(old)     # history only when the PSD changed or moved
                    if old.path != path:
                        num_moved += 1
                    else:
                        num_changed += 1
                card.timestamp = timestamp
                card.blob_sha = sha
                if author and not card.author:
                    card.author = author
                if self.edits:
                    self.edits.apply(name, card)
                self.db.stats[name] = card

                if output_problematic:
                    problems = CardValidator.validate(card)
                    if problems:
                        problematic_cards.append((path, card, problems))

                num_updated += 1
                if num_updated % UPDATE_RATE == 0:
                    self._send_progress(interaction, num_updated)
        finally:
            self._lookups = {}
            lookups.shutdown(wait=False, cancel_futures=True)
            for download in downloads:
                download.cancel()
            pool.shutdown(wait=True)
            for download in downloads:                          # temp files left by an error
                if download.done() and not download.cancelled() and download.result() and isfile(download.result()):
                    remove(download.result())

        self._send_summary(interaction, num_changed, num_old, num_moved)
        return (self._format_problems(problematic_cards), num_changed + num_moved)

    def _get_remote_timestamp(self,
                              repo,
                              path: str,
                              name: str) -> tuple[float, Optional[str]]:
        """
        Get timestamp for a file from GitHub.

        Uses the most recent commit for timestamp, but preserves original author
        from the first commit if not already set.
        """
        if not repo:
            return (datetime(1000, 1, 1).timestamp(), None)

        pending = self._lookups.pop(path, None)                 # started beside the download
        commits = pending.result() if pending is not None else self._commit_page(path)
        if not commits:
            return (datetime(1000, 1, 1).timestamp(), None)

        # New card: its author is whoever wrote the FIRST commit (not who committed it). The authors file
        # (AUTHORS_PATH) corrects either afterwards.
        if name not in self.db.stats or not self.db.stats[name].author:
            first = (commits[len(commits) - 1] if len(commits) < COMMITS_PAGE
                     else repo.get_commits(path=path).reversed[0])     # more than one page: fetch the last
            return (commits[0].commit.committer.date.timestamp(), first.commit.author.name)

        return (commits[0].commit.committer.date.timestamp(), None)

    @staticmethod
    def _send_progress(interaction: Optional[Interaction], count: int) -> None:
        """Send progress update."""
        message = f"{count} cards updated."
        if interaction:
            queue_edit(interaction, content=message)
        else:
            print(message)

    @staticmethod
    def _send_summary(interaction: Optional[Interaction],
                      num_changed: int,
                      num_old: int,
                      num_moved: int = 0) -> None:
        """Send summary of update."""
        say(interaction, f"{num_changed} cards changed or new, {num_old} unchanged, {num_moved} moved.")

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
def _forget_tree(repository: str) -> None:
    """Drop the tree ETag so the next /update_stats refetches and retries."""
    _tree_etags.pop(repository, None)


def _refresh_links(repository: str) -> None:
    """Re-pin image links; on failure drop the tree ETag so the next /update_stats retries."""
    if not refresh_history():
        _forget_tree(repository)


# ========================
# Public API Functions
# ========================
def update_stats(interaction: Optional[Interaction] = None,
                 output_problematic: bool = True,
                 force_update: bool = False,
                 verbose: bool = True,
                 cards: str = "") -> list[str]:
    """
    Update card statistics database from the GitHub repository.

    Local mode (walking the clone on disk) was archived on 2026-10-20 to branch archive/local-update-stats.

    Args:
        interaction: Optional Discord interaction
        output_problematic: Whether to output problematic cards
        force_update: Re-parse every card even if its PSD did not change
        cards: Comma-separated card names to re-parse anyway (e.g. after a parser fix)
        verbose: Whether to display progress messages and problem summaries

    Returns:
        List of formatted problem messages
    """
    if verbose:
        say(interaction, "Updating card statistics database... This may take a while.")

    stats_db.load()
    stats_db.seen_cards.clear()
    traverser = RepositoryTraverser(stats_db)
    traverser.only = {CardAuthors.key(c) for c in (cards or "").split(",") if c.strip()}

    problem_cards, num_changed = traverser.traverse_remote(
        card_repo.repository,
        interaction,
        output_problematic,
        force_update)

    if num_changed > 0:
        stats_db.prune_clean_cards()
    if num_changed > 0 or traverser.changed:
        stats_db.save()

    if verbose:
        say(interaction, "Card statistics update complete!")

    # Output problems
    if verbose and problem_cards:
        say(interaction, "".join(problem_cards))         # detailed problems
        for card in problem_cards:                        # card names only
            say(interaction, card.split('```')[0])

    if traverser.tree is not None:                  # tree changed (not 304): refresh edited type emojis
        queue_command(sync_changed_emojis, interaction, traverser.repository, traverser.shas, _forget_tree)
        if save_affected(traverser.repository, traverser.shas):
            queue_command(update_after_stats, interaction, traverser.repository, _forget_tree)

    # Always refresh card_repo stats
    queue_command(card_repo.populate_files, traverser.tree, None, traverser.etag)
    queue_command(card_repo.prep_dataframes)

    return problem_cards


# ========================
# Module Exports
# ========================
__all__ = [
    'RepositoryTraverser',
    'update_stats',
]
