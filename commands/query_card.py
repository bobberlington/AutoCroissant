from dataclasses import dataclass
from datetime import date, datetime, time as dt_time, timedelta
from difflib import get_close_matches, SequenceMatcher
from discord import Embed, Interaction
from os.path import basename, splitext
from pickle import load
from asyncio import to_thread
from threading import Lock
from requests import get
from requests.exceptions import RequestException
from typing import Optional
from warnings import filterwarnings
from pandas import DataFrame, Index, concat, merge
from re import compile, escape, IGNORECASE
from urllib.parse import quote

from global_config import ALIAS_PKL, CARD_REPO, EDITS_PATHS, RULEBOOK_DIR, SITE_URL, STATS_PKL, TIMEZONE
from commands.card_edits import ALIASES_MOVED
from commands.card_history import start as version_start
from commands.card_models import save_pickle, stats_db
from commands.github_api import blob_text, fetch_tree
from commands.repo_history import commit_before, raw_url, raw_url_at
from commands.type_emojis import render_type_tokens
from commands.utils import queue_long_message, queue_message, split_long_message

filterwarnings('ignore')

# ========================
# Configuration
# ========================
DEFAULT_REPOSITORY = CARD_REPO
DEFAULT_MATCH_RATIO = 0.6
RESULTS_PER_MESSAGE = 10                                 # /query_ability lines per message
_alias_lock = Lock()                                     # one writer for aliases.pkl
AS_OF_PATTERN = compile(r"^(\d{4})-(\d{1,2})-(\d{1,2})$")   # /query as_of: YYYY-MM-DD
UNKNOWN_BEFORE = 946684800.0                             # 2000-01-01 UTC; older stamps: no commit found (year 1000)
DATE_PATTERN = compile(r"^(created)\s*(<=|>=|==|<|>)\s*(\d{4}(?:-\d{2}){0,2})$")   # created>=2025, created == 2023-10
AS_OF_EDITS_NOTE = ("Database edits (website, /update_metadata) are not dated yet: a version shows those it had "
                    "when it was replaced, the current card today's.")


@dataclass
class CardRepository:
    """Manages card repository state and operations."""

    repository: str = DEFAULT_REPOSITORY
    match_ratio: float = DEFAULT_MATCH_RATIO

    def __post_init__(self):
        self.git_file_alias: dict[str, str] = {}
        self.git_files: dict[str, str] = {}
        self.blobs: dict[str, str] = {}                  # png path -> blob sha, for commit-pinned links
        self.site_ids: dict[str, tuple[str, str]] = {}   # png file name -> (website card id, embed color)
        self._tree: Optional[dict] = None                # last tree, reused while GitHub says 304
        self._tree_etag = ""
        self._tree_lock = Lock()                         # held while _tree/_tree_etag change; type emojis wait on it
        self._edit_names: tuple = ("", {})               # (Card Edits.json blob sha, names)
        self._edits = None                               # that file, parsed (CardEdits): /alias reads its aliases
        self._moved_hint = False                         # the latest Card Edits.json didn't parse but named ALIASES_MOVED
        self._site_etag = ""
        self.ambiguous_names: dict[str, tuple[str, ...]] = {}
        self.cards_dff: DataFrame = DataFrame()
        self.rulebook_dff: DataFrame = DataFrame()
        self._lowered: tuple = (None, [])                # (cards_dff, [(label, lowered ability)]) for the fuzzy step
        self._stat_pattern = compile(r"^(\w+)\s*(<|<=|==|>=|>)\s*(\d+)$", IGNORECASE)   # spaces as the website's search

    @property
    def git_filenames(self) -> list[str]:
        """Get list of all git filenames."""
        return list(self.git_files.keys())

    def populate_files(self, tree: Optional[dict] = None, repository: Optional[str] = None,
                       etag: Optional[str] = None) -> int:
        """
        Fetch and populate file listings from GitHub repository.

        Args:
            tree: A tree /update_stats just fetched (saves a request)
            repository: Switch to this repository, only if its tree loads (/set_repo)
            etag: That tree's ETag, so the next fetch can be a 304

        Returns:
            HTTP status code (200 for success)
        """
        status = 200
        repo = repository or self.repository
        with self._tree_lock:                                   # _tree and _tree_etag change together
            if tree is None:
                tree, etag, status, why = self._fetch_tree(repo)
                if tree is None:                                # keep the last good tree (B7)
                    print(f"Card list: GitHub tree fetch for {repo} failed ({why}); "
                          f"keeping the last good list ({len(self.git_files)} entries)")
                    return status
            self.repository, self._tree, self._tree_etag = repo, tree, etag or ""   # an ETag must match _tree
        self.git_files.clear()
        self.blobs.clear()
        self.ambiguous_names.clear()

        # Process PNG files from repository
        for item in tree.get("tree", []):
            path: str = item.get("path", "")
            if not path.endswith(".png"):
                continue

            self.blobs[path] = item.get("sha", "")
            self._add_file_to_registry(path)

        # Add aliases to registry: Card Edits.json's, then aliases.pkl's until they moved there (they win, as before)
        self._populate_edit_names()
        self._populate_aliases()
        self._populate_titles()
        self._populate_site_ids()

        return status

    def _fetch_tree(self, repo: str) -> tuple:
        """
        GET repo's tree, with the cached ETag when it is repo's (hold _tree_lock).

        Returns:
            (tree, its ETag, 200, "") or (None, "", status (0: no reply or not a tree), why)
        """
        etag = self._tree_etag if repo == self.repository and self._tree and self._tree_etag else ""
        try:
            response = fetch_tree(repo, etag)
            if response.status_code == 304:                 # unchanged: free, and no download
                return self._tree, etag, 200, ""
            if response.status_code != 200:
                return None, "", response.status_code, f"HTTP {response.status_code}"
            body = response.json()                          # untyped: tree is a dict under Cython
        except (RequestException, ValueError) as e:         # ValueError: a body that isn't JSON
            return None, "", 0, f"{type(e).__name__}: {e}"
        if not isinstance(body, dict):
            return None, "", 0, "not a tree"
        return body, response.headers.get("ETag", ""), 200, ""

    def current_tree(self) -> tuple:
        """
        The tree for other readers (type emojis), fetched with the card list's ETag. A newer tree is kept for the
        card list (its next fetch is a 304), so the fallback is never older than what a reader last got.

        Returns:
            (tree, "") or, when GitHub failed, (the last good tree, why); raises when there is no tree yet
        """
        with self._tree_lock:
            tree, etag, _, why = self._fetch_tree(self.repository)
            if tree is not None:
                self._tree, self._tree_etag = tree, etag
                return tree, ""
            if self._tree is None:
                raise RuntimeError(f"GitHub tree fetch for {self.repository} failed ({why})")
            return self._tree, why

    def _add_file_to_registry(self, path: str) -> None:
        """Add a file path to the registry, handling duplicates."""
        png_filename = path[path.rindex("/") + 1:].lower()

        if png_filename in self.git_files:
            # Handle duplicate filenames by prefixing with top-level folder
            new_filename = self._get_prefixed_name(path, png_filename)
            old_path = self.git_files[png_filename]
            old_filename = self._get_prefixed_name(old_path, png_filename)

            self.git_files[old_filename] = old_path
            # In case both filenames are the same, set the new second to make sure something maps to it...
            self.git_files[new_filename] = path

            # Track ambiguous names
            if png_filename not in self.ambiguous_names:
                self.ambiguous_names[png_filename] = (old_filename,)
            if old_filename not in self.ambiguous_names[png_filename]:
                self.ambiguous_names[png_filename] = (*self.ambiguous_names[png_filename], old_filename)
            self.ambiguous_names[png_filename] = (*self.ambiguous_names[png_filename], new_filename)
        else:
            self.git_files[png_filename] = path

    @staticmethod
    def _get_prefixed_name(path: str, filename: str) -> str:
        """Get filename prefixed with top-level folder."""
        folder = path[:path.index('/')]
        return f"{folder}/{filename}".lower()

    def aliases_moved(self) -> bool:
        """Card Edits.json says aliases.pkl moved into it (ALIASES_MOVED): the pickle's aliases no longer count."""
        return self._moved_hint or (self._edits is not None and self._edits.aliases_moved)

    def _populate_aliases(self) -> None:
        """Populate aliases.pkl's aliases into the file registry, until they moved into Card Edits.json."""
        if self.aliases_moved():
            return
        for alias, target in self.git_file_alias.items():
            alias_key = f"{alias.lower()}.png"
            target = target.lower()

            if target in self.git_files:
                self.git_files[alias_key] = self.git_files[target]
            else:
                matches = [
                    v for k, v in self.git_files.items()
                    if k.lower().endswith(target)
                ]
                if matches:
                    self.git_files[alias_key] = matches[0]

    def _parse_stat_query(self, ability: str) -> Optional[DataFrame]:
        """
        Parse and execute column-based queries (e.g., 'hp>10', 'type==creature', 'stars<=5').

        Supports:
        - Numeric comparisons: hp>5, def<=10, atk>=3, spd==7, stars<6
        - String exact matches: type==creature, subtype==attack, series==kirby
        - String contains: type~creature (contains 'creature'), subseries~door

        Args:
            ability: Query string

        Returns:
            DataFrame of matching cards or None
        """
        # Dates at the precision given (created>=2025: 2025 or later; created==2023-10: in October 2023)
        date_match = DATE_PATTERN.match(ability.strip().lower())
        if date_match:
            column, operator, value = date_match.groups()
            if column not in self.cards_dff.columns:
                return None
            col = self.cards_dff[column]
            day = col.where(col.notna(), "").astype(str).str[:len(value)]
            kept = {"<": day < value, "<=": day <= value, "==": day == value, ">=": day >= value, ">": day > value}[operator]
            return self.cards_dff[col.notna() & kept]

        # Try numeric comparison pattern (e.g., hp>5, stars<=3)
        numeric_match = self._stat_pattern.match(ability.lower())
        if numeric_match:
            column, operator, value = numeric_match.groups()
            value = int(value)

            # Check if column exists in dataframe
            if column not in self.cards_dff.columns:
                return None

            # Apply operator
            operators = {
                '<': lambda df, col, val: df[df[col] < val],
                '<=': lambda df, col, val: df[df[col] <= val],
                '==': lambda df, col, val: df[df[col] == val],
                '>=': lambda df, col, val: df[df[col] >= val],
                '>': lambda df, col, val: df[df[col] > val],
            }

            return operators.get(operator, lambda df, col, val: DataFrame())(self.cards_dff, column, value)

        # Try string exact match pattern (e.g., type==creature, subtype==attack)
        string_exact_pattern = compile(r"^(\w+)\s*==(.+)$", IGNORECASE)
        string_match = string_exact_pattern.match(ability)
        if string_match:
            column, value = string_match.groups()
            column = column.lower()
            value = value.strip().strip('"').lower()

            # Check if column exists
            if column not in self.cards_dff.columns:
                return None

            # Filter by exact match (case-insensitive); an empty cell never matches ("None"/"nan" as text)
            col = self.cards_dff[column]
            return self.cards_dff[col.notna() & (col.astype(str).str.lower() == value)]

        # Try string contains pattern (e.g., type~creature)
        string_contains_pattern = compile(r"^(\w+)\s*~(.+)$", IGNORECASE)
        contains_match = string_contains_pattern.match(ability)
        if contains_match:
            column, value = contains_match.groups()
            column = column.lower()
            value = value.strip().strip('"').lower()

            # Check if column exists
            if column not in self.cards_dff.columns:
                return None

            # Filter by contains (case-insensitive); an empty cell never matches
            col = self.cards_dff[column]
            return self.cards_dff[col.notna() & col.astype(str).str.lower().str.contains(value, na=False, regex=False)]

        return None

    @staticmethod
    def _split_unquoted(text: str, sep: str) -> list[str]:
        """text split on sep, except inside "double quotes" (subseries=="mario & luigi")."""
        parts, start, quoted = [], 0, False
        for i, ch in enumerate(text):
            if ch == '"':
                quoted = not quoted
            elif ch == sep and not quoted:
                parts.append(text[start:i])
                start = i + 1
        parts.append(text[start:])
        return parts

    def ability_search_engine(self, ability: str) -> DataFrame:
        """
        Search for cards by ability text or column values.

        Supports:
        - OR queries: "ability1 | ability2"
        - AND queries: "ability1 & ability2"
        - NOT queries: "!ability"
        - Exact matches: '"exact text"'
        - Numeric comparisons: "hp>5", "stars<=3", "def==10"
        - String exact match: "type==creature", "subtype==attack"
        - String contains: "type~aux", "series~kirby"
        - Quoted values keep & and |: 'subseries=="mario & luigi"'

        Args:
            ability: Search query string

        Returns:
            DataFrame of matching cards
        """
        ability = ability.strip()

        # Handle OR queries
        queries = self._split_unquoted(ability, "|")
        if len(queries) > 1:
            results = [self.ability_search_engine(q.strip()) for q in queries]
            combined = concat(results)
            # Drop duplicates by index (card name) and ability column
            return combined[~combined.index.duplicated(keep='first')]

        # Handle AND queries
        queries = self._split_unquoted(ability, "&")
        if len(queries) > 1:
            results = [self.ability_search_engine(q.strip()) for q in queries]
            result = results[0]
            for r in results[1:]:
                result = merge(result, r, left_index=True, right_index=True, how="inner", copy=False)
            return result

        # Handle stat queries
        stat_result = self._parse_stat_query(ability)
        if stat_result is not None:
            return stat_result

        # Handle NOT queries
        opposite = ability.startswith("!")
        if opposite:
            ability = ability[1:].strip()

        # Handle exact matches
        if ability.startswith('"') and ability.endswith('"'):
            pattern = r"(?:^|\s|$|\b)" + escape(ability.strip('"')) + r"(?:^|\s|$|\b)"
            if opposite:
                return self.cards_dff[~self.cards_dff["ability"].str.contains(pattern, na=False)]
            return self.cards_dff[self.cards_dff["ability"].str.contains(pattern, na=False)]

        # Handle substring matches with fuzzy matching
        if opposite:
            return self.cards_dff[~self.cards_dff["ability"].str.contains(escape(ability), na=False)]

        # Exact substring matches
        exact_matches = self.cards_dff[self.cards_dff["ability"].str.contains(escape(ability), na=False)]

        # Fuzzy matches
        fuzzy_matches = self.cards_dff.loc[self._fuzzy_labels(ability)]

        # Combine and remove duplicates by index
        combined = concat([exact_matches, fuzzy_matches])
        return combined[~combined.index.duplicated(keep='first')]

    def _fuzzy_labels(self, ability: str) -> Index:
        """Cards whose SequenceMatcher ratio with the query beats match_ratio, in frame order."""
        if self._lowered[0] is not self.cards_dff:            # built once per prep_dataframes
            self._lowered = (self.cards_dff, [(label, text.lower()) for label, text in self.cards_dff["ability"].items()])
        la, cut = len(ability), self.match_ratio
        keep = []
        for label, text in self._lowered[1]:
            lb = len(text)
            if la + lb and 2.0 * min(la, lb) / (la + lb) <= cut:   # ratio()'s upper bound: can't match
                continue
            if SequenceMatcher(None, ability, text).ratio() > cut:
                keep.append(label)
        return Index(keep, dtype=self.cards_dff.index.dtype)

    def rulebook_search_engine(self, search_text: str) -> DataFrame:
        """
        Search rulebook entries for matching text.

        Args:
            search_text: Text to search for (case-insensitive)

        Returns:
            DataFrame of matching rulebook entries
        """
        search_text = search_text.strip().lower()

        if not search_text:
            return self.rulebook_dff

        # Search in ability text
        matches = self.rulebook_dff[
            self.rulebook_dff["ability"].str.contains(
                escape(search_text),
                na=False,
                case=False
            )
        ]

        return matches

    def load_aliases(self) -> None:
        """Load card aliases from pickle file."""
        print(f"Loading aliases from {ALIAS_PKL}")
        try:
            with open(ALIAS_PKL, 'rb') as f:
                raw_aliases = load(f)

            # Normalize aliases to lowercase
            self.git_file_alias = {
                k.lower(): v.lower()
                for k, v in raw_aliases.items()
            }

        except (EOFError, FileNotFoundError):
            print(f"{ALIAS_PKL} doesn't exist, creating empty file...")
            self.git_file_alias = {}
            self.save_aliases()

    def save_aliases(self) -> None:
        """Save card aliases to pickle file (atomically, under the alias lock)."""
        save_pickle(self.git_file_alias, ALIAS_PKL, _alias_lock)

    def _populate_edit_names(self) -> None:
        """Website aliases and names from the card repo's Card Edits.json (the first of EDITS_PATHS in the tree)."""
        from commands.psd_analyzer import CardEdits
        shas = {item.get("path", ""): item.get("sha", "") for item in (self._tree or {}).get("tree", [])}
        path = next((p for p in EDITS_PATHS if shas.get(p)), None)
        if path is None:
            return
        sha = shas[path]
        if self._edit_names[0] != sha:                           # download only when the file changed
            text = None
            try:
                text = blob_text(self.repository, sha)
                edits = CardEdits.parse(text)
            except Exception as e:
                print(f"Could not read {path}: {e}")
                if text is not None:                     # unparsed, but migrated: aliases.pkl stays off
                    self._moved_hint = f'"{ALIASES_MOVED}"' in text
            else:
                self._edits, self._edit_names, self._moved_hint = edits, (sha, edits.names()), False
        # a failed read keeps the last good names: aliases live only here once moved
        for card, extra in self._edit_names[1].items():
            self._add_lookup(card, extra)

    def relink(self) -> None:
        """Card Edits.json's names and aliases, then the printed titles, onto keys the registry lacks (after /alias delete)."""
        for card, extra in self._edit_names[1].items():
            self._add_lookup(card, extra)
        self._populate_titles()

    def _populate_titles(self) -> None:
        """Printed titles the parser read (display_name), so /query finds "Black Clot" as well as Clot_Black."""
        try:
            cards = list(stats_db.stats.items())
        except RuntimeError:                                     # /update_stats writing meanwhile: after it, again
            return
        for name, card in cards:
            title = getattr(card, "display_name", None)
            if title and title != name:
                self._add_lookup(name, [title])

    def _add_lookup(self, card: str, extra: list) -> None:
        """More names that find card's PNG (first come, first kept). Registry keys are lowercase."""
        target = f"{card.replace(' ', '_')}.png".lower()
        path = self.git_files.get(target) or next((v for k, v in self.git_files.items() if k.endswith("/" + target)), None)
        if path:
            for name in extra:
                self.git_files.setdefault(f"{name.strip().lower().replace(' ', '_')}.png", path)

    def _populate_site_ids(self) -> None:
        """Website card ids, for "Card page" links."""
        try:
            reply = get(f"{SITE_URL}/data/cards.json", timeout=30,
                        headers={"If-None-Match": self._site_etag} if self.site_ids and self._site_etag else {})
            if reply.status_code == 304:                         # unchanged since the last look
                return
            reply.raise_for_status()
            cards = reply.json().get("cards", [])
            self._site_etag = reply.headers.get("ETag", "")
        except Exception as e:
            print(f"Could not read the website's cards: {e}")
            return
        self.site_ids = {basename(c["png"]).lower(): (c["id"], c.get("color") or "") for c in cards if c.get("png")}

    def site_url(self, filename: str) -> Optional[str]:
        """The card's page on the website, for a registry file name (aliases included)."""
        path = self.git_files.get(filename.lower())
        found = self.site_ids.get(basename(path).lower()) if path else None
        return f"{SITE_URL}/c/{found[0]}/" if found else None

    def prep_dataframes(self) -> None:
        """Load card statistics from pickle file."""
        from commands.psd_analyzer import get_stats_as_dict

        cards_df = DataFrame.from_dict(get_stats_as_dict()).transpose()

        if not cards_df.empty:
            for column in ("subseries", "created"):     # database-only fields: absent until a card has one
                if column not in cards_df:
                    cards_df = cards_df.assign(**{column: None})
            if "hidden" in cards_df:                    # hidden on the website
                cards_df = cards_df[cards_df["hidden"] != True]
            if "duplicate_of" in cards_df:              # same card as its original
                cards_df = cards_df[cards_df["duplicate_of"].isna()]

            # Separate cards and rulebook entries
            has_ability = cards_df["ability"].notna()
            is_rulebook = cards_df["path"].str.contains(RULEBOOK_DIR, na=False)

            # Cards DataFrame (excludes rulebook)
            self.cards_dff = cards_df[has_ability & ~is_rulebook].copy()
            self.cards_dff["ability"] = self.cards_dff["ability"].str.lower()

            # Rulebook DataFrame (only rulebook entries)
            self.rulebook_dff = cards_df[has_ability & is_rulebook].copy()
            self.rulebook_dff["ability"] = self.rulebook_dff["ability"].str.lower()

            print(f"Loaded {len(self.cards_dff)} cards and {len(self.rulebook_dff)} rulebook entries")

    def _normalize_card_name(self, query: str) -> str:
        """Normalize card name for searching."""
        card = query.replace(" ", "_").lower()
        return card if card.endswith(".png") else f"{card}.png"

    def _find_closest_match(self, query: str) -> Optional[str]:
        """Find closest matching card filename."""
        key = self._normalize_card_name(query)
        if key in self.git_files and len(key) < 200:    # exact: the only 1.0 score (difflib's autojunk starts at 200)
            return key
        matches = get_close_matches(key, self.git_filenames, n=1, cutoff=self.match_ratio)
        return matches[0] if matches else None

    def get_card_url(self, filename: str) -> Optional[str]:
        """
        Get full GitHub URL for a card.

        Args:
            filename: Card filename (with or without .png extension)

        Returns:
            Full URL to the card image

        Raises:
            KeyError: If card is not found
        """
        # Normalize filename - replace spaces with underscores
        if not filename.endswith('.png'):
            filename = f"{filename}.png"
        filename = filename.replace(' ', '_').lower()

        if filename not in self.git_files:
            filename = filename.replace('_', ' ')
            if filename not in self.git_files:
                print(f"Card not found: {filename}")
                return

        path = self.git_files[filename]
        return raw_url(self.repository, path, self.blobs.get(path))     # pinned to the card's last commit

    def get_card_path(self, query: str) -> Optional[str]:
        """
        Get the path for a card (without extension) given its name.

        Args:
            query: Card name

        Returns:
            Path to card (with spaces, no extension, not URL-encoded) or None if not found
        """
        closest = self._find_closest_match(query)
        if not closest:
            return None
        return splitext(self.git_files[closest])[0]


# ========================
# Discord Command Functions
# ========================
def _stats_card(filename: str):
    """The stats entry for a registry file name (aliases resolve to their PNG)."""
    from commands.psd_analyzer import card_name, stats_db
    path = card_repo.git_files.get(filename.lower())
    return stats_db.stats.get(_stats_name(card_name(path))) if path else None


def _stats_name(name: str) -> str:
    """The database's name for a PNG's card name: the PSD's spelling when the two differ only in letter case
    (Grand_Doomer.png, grand_doomer.psd); the name itself otherwise."""
    from commands.psd_analyzer import stats_db
    if name in stats_db.stats or name in stats_db.old_stats:
        return name
    key = name.lower()
    try:
        return next((n for n in [*stats_db.stats, *stats_db.old_stats] if n.lower() == key), name)
    except RuntimeError:                                 # /update_stats writing meanwhile
        return name


def query_name(interaction: Interaction, query: str, as_of: Optional[str] = None) -> None:
    """
    Query and display a card by name.

    Args:
        interaction: Discord interaction
        query: Card name to search for
        as_of: A date (YYYY-MM-DD): the card as it was at the end of that day, from its stored versions
    """
    if as_of and as_of.strip():
        _query_as_of(interaction, query, as_of)
        return

    closest = card_repo._find_closest_match(query)

    if not closest:
        queue_message(interaction, "No card found!")
        return
    _post_card(interaction, closest)


def _post_card(interaction: Interaction, closest: str, url: Optional[str] = None) -> None:
    """The card's image (an embed when it has a website name, subseries, note or original) and any ambiguity; url: an
    image other than today's (/query as_of)."""
    url, page = url or card_repo.get_card_url(closest), card_repo.site_url(closest)
    links = [("Card page", page)] if page else None
    card = _stats_card(closest)
    shown = getattr(card, "display_name", None) if card else None
    subseries = getattr(card, "subseries", None) if card else None
    extra = [f"Same card as {card.duplicate_of}" if card and getattr(card, "duplicate_of", None) else "",
             getattr(card, "note", None) or "" if card else ""]
    if card and (shown or subseries or any(extra)):     # website name, subseries, note or duplicate: an embed says so
        extra.insert(0, " · ".join(s for s in (card.series, subseries) if s))      # "Series · Subseries"
        created = getattr(card, "created", None)
        if created:
            extra.append(f"Created {'about ' if created.get('provisional') else ''}{created['date']}")
        found = card_repo.site_ids.get(basename(card_repo.git_files.get(closest, "")).lower())
        embed = Embed(title=shown or card.name, url=page, description="\n".join(e for e in extra if e) or None,
                      color=int(found[1][1:], 16) if found and found[1] else None)
        embed.set_image(url=url)
        queue_message(interaction, None, embed=embed, links=links)
    else:
        queue_message(interaction, url, links=links)

    # Notify if name is ambiguous
    if closest in card_repo.ambiguous_names:
        ambiguous_options = "\n".join(card_repo.ambiguous_names[closest])
        queue_message(interaction, f"Ambiguous name found. If this wasn't the card you wanted, try:\n{ambiguous_options}")


# ========================
# /query as_of
# ========================
def parse_as_of(text: str, today: date) -> tuple:
    """
    An as_of date, read in TIMEZONE.

    Args:
        text: The date, YYYY-MM-DD
        today: Today in TIMEZONE

    Returns:
        (the day, its end as epoch seconds, "") or (None, 0.0, why not)
    """
    found = AS_OF_PATTERN.match(text.strip())
    try:
        day = date(int(found.group(1)), int(found.group(2)), int(found.group(3))) if found else None
    except ValueError:                                   # 2026-02-30, month 13...
        day = None
    if day is None:
        return (None, 0.0, "That isn't a date: use YYYY-MM-DD, e.g. 2026-02-01.")
    if day > today:
        return (None, 0.0, f"{day.isoformat()} is in the future (today is {today.isoformat()}, {TIMEZONE.key}).")
    return (day, datetime.combine(day + timedelta(days=1), dt_time(0), tzinfo=TIMEZONE).timestamp(), "")


def version_at(versions: list, end: float) -> Optional[int]:
    """Index of the latest version (oldest first) started before end (_since); None when all are later."""
    found = None
    for i, card in enumerate(versions):
        if _since(card) < end:
            found = i
    return found


def _since(card) -> float:
    """When a version starts: its `version` record's from (a duplicate dropped, its data kept: card_history.carry),
    else its timestamp."""
    return version_start(card)[0] or 0.0


def _day(stamp) -> str:
    """A stamp's date in TIMEZONE."""
    if not stamp or stamp < UNKNOWN_BEFORE:
        return "an unknown date"
    return datetime.fromtimestamp(stamp, TIMEZONE).date().isoformat()


def _approx(card) -> str:
    """Said of legacy versions (no blob): their dates can't be checked against git."""
    known = (getattr(card, "timestamp", 0.0) or 0.0) >= UNKNOWN_BEFORE
    return " (date approximate)" if known and not getattr(card, "blob_sha", None) else ""


def archived_text(tag: dict) -> str:
    """'archived on <date> (PSD deleted in <commit>)', or moved to MDW/...; 'by' for a backfill (deleted earlier)."""
    commit = tag.get("commit")
    gone = f"PSD moved to {tag['moved_to'].rsplit('/', 1)[0]}" if tag.get("moved_to") else "PSD deleted"
    deleted = f"{gone} in {commit[:7]}" if commit else gone
    if tag.get("renamed_to"):
        deleted += f", renamed to {tag['renamed_to']}"
    return f"archived {'by' if tag.get('backfill') else 'on'} {_day(tag.get('at'))} ({deleted})"


def history_match(query: str) -> Optional[tuple]:
    """(registry file name or None, card name) for query, archived cards included (their PNG may be gone; their own
    name before an alias or another card's PNG that claims it): /query as_of and /view_old_metadata."""
    from commands.psd_analyzer import card_name, stats_db
    try:
        archived = {card_repo._normalize_card_name(name): name for name in stats_db.archived()}
    except RuntimeError:                                 # /update_stats writing meanwhile
        archived = {}
    key = card_repo._normalize_card_name(query)
    if key in archived and key in card_repo.git_files and \
            card_repo._normalize_card_name(card_name(card_repo.git_files[key])) != key:
        return (None, archived[key])                     # an alias or another card's PNG claims the archived name
    if len(key) >= 200 or (key not in card_repo.git_files and key not in archived):
        found = get_close_matches(key, card_repo.git_filenames + list(archived), n=1, cutoff=card_repo.match_ratio)
        if not found:
            return None
        key = found[0]
    if key in card_repo.git_files:
        return (key, _stats_name(card_name(card_repo.git_files[key])))
    return (None, archived[key])


def _query_as_of(interaction: Interaction, query: str, as_of: str) -> None:
    """
    /query as_of: the card at the end of that day in TIMEZONE. Its versions are old_stats (oldest first) then the
    current card; each runs from its start (_since) to the next one's, the last of an archived card to its archive.

    Args:
        interaction: Discord interaction
        query: Card name to search for
        as_of: The date, YYYY-MM-DD
    """
    from commands.psd_analyzer import _format_metadata, stats_db
    day, end, why = parse_as_of(as_of, datetime.now(TIMEZONE).date())
    if why:
        queue_message(interaction, why)
        return
    found = history_match(query)
    if found is None:
        queue_message(interaction, "No card found!")
        return
    key, name = found
    label = f"**{name}**"
    current = stats_db.stats.get(name)
    if current is not None and getattr(current, "duplicate_of", None):     # a copy: its original's history
        label = f"**{name}** (same card as {current.duplicate_of})"
        name = current.duplicate_of
        current = stats_db.stats.get(name)
    versions = list(stats_db.old_stats.get(name) or [])
    if current is not None:
        versions.append(current)
    if not versions:
        queue_message(interaction, f"{label} has no stored card data.")
        return

    head = f"{label} as of {day.isoformat()} (end of day, {TIMEZONE.key}): "
    index = version_at(versions, end)
    if index is None:
        first = versions[0]
        queue_message(interaction, f"{head}no stored version is that old; the oldest is from {_day(_since(first))}"
                                   f"{_approx(first)}.")
        return
    card = versions[index]
    which = f"version {index + 1} of {len(versions)}"
    since = f"from {_day(_since(card))}{_approx(card)}"
    tag = None if current is not None else getattr(versions[len(versions) - 1], "archived", None)
    if tag and index + 1 == len(versions) and end > (tag.get("at") or 0):     # no at: archived after its last version
        line = f"{head}{archived_text(tag)}. Its last version ({which}), {since}:"
    elif card is current:
        line = f"{head}the current card ({which}), {since}."
    else:
        until = ""
        if index + 1 < len(versions):
            until = f" until {_day(_since(versions[index + 1]))}"
        elif tag:
            until = f" until it was {archived_text(tag)}"
        line = f"{head}{which}, {since}{until}."
    line = f"{line}\n{AS_OF_EDITS_NOTE}"

    if card is current and key:                          # the current card: its image of that day
        url, said = _image_at(card_repo.git_files.get(key), end, card.timestamp)
        queue_message(interaction, f"{line}\n{said}" if said else line)
        _post_card(interaction, key, url)
        return
    page = card_repo.site_url(key) if key else None      # an older version: its stored data, its image of that day
    links = [("Card page", page)] if page else []
    png = splitext(card.path or "")[0] + ".png"
    known = card_repo.git_files.get(basename(png).lower(), "")
    png = known if known.lower() == png.lower() else png     # a PNG named in another letter case (Grand_Doomer.png)
    found = commit_before(card_repo.repository, png, end) if card.path else None
    if found is not None:
        links.append((f"Image ({_day(found[1])})", raw_url_at(card_repo.repository, png, found[0])))
    queue_message(interaction, f"{line}\n```{_format_metadata(card, name)}```", links=links or None)


def _image_at(png: Optional[str], end: float, read: float) -> tuple:
    """
    The current card's image as of end (a carried start reaches back before today's PNG): the newest commit of its PNG
    before end, from the history clone (repo_history.commit_before).

    Args:
        png: The card's PNG path
        end: The as_of moment
        read: When the card's PSD was read (its timestamp): without the clone, a day before it says it can't tell

    Returns:
        (raw url, or None for today's image; a line saying which image it is, or "" when it is today's and that day's)
    """
    found = commit_before(card_repo.repository, png, end) if png else None
    if found is None:
        return (None, "Image: today's (that day's is not known)." if end <= (read or 0.0) else "")
    sha, at, later = found
    if not later:
        return (None, "")
    return (raw_url_at(card_repo.repository, png, sha), f"Image: as of {_day(at)} (it has changed since).")


def query_ability(interaction: Interaction,
                  ability: str,
                  limit: int = -1) -> None:
    """
    Query cards by ability text or column values (excludes rulebook entries).

    Supports multiple query types:
    - Text search: "damage", "draw"
    - Numeric comparisons: "hp>5", "stars<=3", "def==10"
    - String exact match: "type==creature", "subtype==attack"
    - String contains: "type~aux", "series~kirby"
    - Combinations: "type==creature & hp>7"

    Args:
        interaction: Discord interaction
        ability: Search query string
        limit: Maximum number of results to display (-1 for all)
        filter_raids: Whether to filter out raid cards

    Note:
        This function does not search rulebook entries.
    """
    if card_repo.cards_dff.empty:
        queue_message(interaction, f"{STATS_PKL} is empty. Run `/update_stats` first.")
        return

    results = card_repo.ability_search_engine(ability.strip().lower())

    total_results = len(results)
    results_to_show = total_results if limit < 0 else min(limit, total_results)

    queue_message(interaction, f"Found {total_results} results for '{ability}'!")

    names = [basename(index) for index in results.index[:results_to_show]]   # index: the path without .png
    if len(names) == 1:                                  # one card: its image preview (its name when the PNG is not in the tree)
        url = card_repo.get_card_url(names[0])
        queue_message(interaction, url or names[0])
    else:                                                # more: name and image link, RESULTS_PER_MESSAGE per message
        for start in range(0, len(names), RESULTS_PER_MESSAGE):
            lines = []
            for name in names[start:start + RESULTS_PER_MESSAGE]:
                url = card_repo.get_card_url(name)
                lines.append(f"{name} — {url}" if url else name)
            queue_long_message(interaction, "\n".join(lines))

    queue_message(interaction, f"Displayed {results_to_show} of {total_results} results for '{ability}'.",
                  links=[("All results on the website", f"{SITE_URL}/#/cards?q={quote(ability.strip())}")])


def query_ability_num_occur(interaction: Interaction, ability: str) -> None:
    """
    Query the number of cards matching an ability.

    Args:
        interaction: Discord interaction
        ability: Ability search query
    """
    if card_repo.cards_dff.empty:
        queue_message(interaction, f"{STATS_PKL} is empty. Run `/update_stats` first.")
        return

    results = card_repo.ability_search_engine(ability.strip().lower())
    queue_message(interaction, f"Found {len(results)} cards matching '{ability}'.")


def query_rulebook(interaction: Interaction,
                   search_text: str,
                   limit: int = -1) -> None:
    """
    Search rulebook entries for matching text.

    Args:
        interaction: Discord interaction
        search_text: Text to search for in rulebook
        limit: Maximum number of results to display (-1 for all)
    """
    if card_repo.rulebook_dff.empty:
        queue_message(interaction, f"{STATS_PKL} is empty or has no rulebook entries. Run `/update_stats` first.")
        return

    results = card_repo.rulebook_search_engine(search_text.strip())

    # Sort alphabetically by index (card name)
    results = results.sort_index(key=lambda x: x.str.lower())

    total_results = len(results)
    results_to_show = total_results if limit < 0 else min(limit, total_results)

    queue_message(interaction, f"Found {total_results} rulebook entries matching '{search_text}'!")

    if total_results == 0:
        return

    # Send results
    for idx, (name, row) in enumerate(results.iterrows()):
        if idx >= results_to_show:
            break

        ability = row.get("ability", "No text available")
        for chunk in split_long_message(render_type_tokens(ability, escape=True)):
            queue_message(interaction, f"**{name}**\n{chunk}")

    queue_message(interaction, f"Displayed {results_to_show} of {total_results} results." if results_to_show < total_results else "Done.",
                  links=[("Rulebook on the website", f"{SITE_URL}/#/rules?q={quote(search_text.strip())}")])


async def set_match_ratio(interaction: Interaction, value: Optional[float] = None) -> None:
    """
    Set or display the fuzzy match ratio.

    Args:
        interaction: Discord interaction
        value: New match ratio (0.0-1.0)
    """
    if value is None:
        await interaction.response.send_message(f"Current match ratio: {card_repo.match_ratio}")
        return

    if not 0.0 <= value <= 1.0:
        await interaction.response.send_message("Match ratio must be between 0.0 and 1.0.")
        return

    card_repo.match_ratio = value
    await interaction.response.send_message(f"Match ratio set to {card_repo.match_ratio}")


async def set_repository(interaction: Interaction, new_repo: Optional[str] = None) -> None:
    """
    Set or display the GitHub repository.

    Args:
        interaction: Discord interaction
        new_repo: New repository name
    """
    if not new_repo:
        await interaction.response.send_message(f"Current repository: {card_repo.repository}")
        return

    await interaction.response.defer()
    old_repo = card_repo.repository                     # switched only once new_repo's tree loads
    status = await to_thread(card_repo.populate_files, None, new_repo)  # off the event loop (retries can take seconds)

    if status != 200:
        await interaction.followup.send(f"Error {status or 'no response'} when connecting to {new_repo}; still using {old_repo}.")
    else:
        await interaction.followup.send(f"Repository set to {card_repo.repository}")


# ========================
# Initialization
# ========================
card_repo = CardRepository()
def init_query() -> None:
    """Initialize the card repository with saved data."""
    card_repo.load_aliases()
    status = card_repo.populate_files()
    if status != 200:
        print(f"Warning: Failed to populate files from repository (status {status})")


# ========================
# Module Exports
# ========================
__all__ = [
    'archived_text',
    'card_repo',
    'history_match',
    'init_query',
    'query_ability',
    'query_ability_num_occur',
    'query_name',
    'query_rulebook',
    'set_match_ratio',
    'set_repository',
]
