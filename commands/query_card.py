from collections import Counter
from copy import copy
from dataclasses import dataclass
from datetime import date, datetime, time as dt_time, timedelta
from difflib import get_close_matches, SequenceMatcher
from discord import Embed, File, Interaction
from discord.utils import escape_markdown
from hashlib import sha256
from io import BytesIO
from json import dumps, loads
from os.path import basename, splitext
from asyncio import to_thread
from threading import Lock
from requests import get
from requests.exceptions import RequestException
from typing import Optional
from warnings import filterwarnings
from numpy import array, flatnonzero, int32, maximum, minimum, where, zeros
from pandas import DataFrame, Index, Series, concat, to_numeric
from re import compile, escape, IGNORECASE
from urllib.parse import quote

from global_config import CARD_REPO, CARD_TREE_JSON, EDITS_PATH, RULEBOOK_DIR, SITE_URL, TIMEZONE
from commands.card_edits import CardAuthors, art_list
from commands.card_db import write_text
from commands.card_history import start as version_start
from commands.card_models import stats_db
from commands.github_api import blob_text, fetch_tree
from commands.repo_history import commit_before, pinned_ref, raw_url, raw_url_at
from commands.rulebook import label as chapter_label
from commands.type_emojis import render_type_tokens
from commands.utils import BREAK_LEN, queue_file, queue_message

filterwarnings('ignore')

# ========================
# Configuration
# ========================
DEFAULT_REPOSITORY = CARD_REPO
DEFAULT_MATCH_RATIO = 0.6
AS_OF_PATTERN = compile(r"^(\d{4})-(\d{1,2})-(\d{1,2})$")   # /query as_of: YYYY-MM-DD
UNKNOWN_BEFORE = 946684800.0                             # 2000-01-01 UTC; older stamps: no commit found (year 1000)
DATE_PATTERN = compile(r"^(created)\s*(<=|>=|==|<|>)\s*(\d{4}(?:-\d{2}){0,2})$")   # created>=2025, created == 2023-10
FIELD_PATTERN = compile(r"^(\w+)\s*(<=|>=|==|!=|<|>|~)\s*(.*)$")   # a condition: field, operator, value
NUMBER_FIELDS = ("hp", "def", "atk", "spd", "stars", "total")
URL_PATTERN = compile(r"https?://[^\s<>\[\]\"'`]+")    # art.js URL_RE
TEXT_FIELDS = ("type", "subtype", "series", "subseries", "author", "types", "name", "alias", "ability", "path", "created")
FIELD_HINTS = {"attack": "atk", "defense": "def", "health": "hp", "speed": "spd", "star": "stars", "aliases": "alias"}
SYNONYMS = {"damage": "dmg", "dmg": "damage", "defense": "def", "def": "defense", "speed": "spd", "spd": "speed",
            "health": "hp", "hp": "health"}                       # a text term finds both words (the card's own terms)
SYNONYM_WORDS = compile(r"\b(" + "|".join(SYNONYMS) + r")\b")
BULLET = compile(r"^[-•*]\s+")                           # a rulebook line's own bullet: the reply adds one
EMOJI = compile(r"(<a?:\w+:\d+>)")                     # Discord emoji markup: never bolded inside
MAX_CONTENT = 2000                                       # Discord's message length limit
SNIPPET = 200                                            # /query_rulebook: a longer line is cut around the match
AS_OF_EDITS_NOTE = ("Database edits (website, Discord) are versions of their own only since late October "
                    "2026: an older version shows those it had when it was replaced.")
REGISTRY = ("git_files", "blobs", "ambiguous_names", "_spaced", "_edits", "_edit_names", "_edits_text", "_renamed",
            "choices", "site_ids", "_site_etag")         # what populate_files builds aside and swaps in
TREE_VERSION = 1                                         # CARD_TREE_JSON's layout


class QueryError(ValueError):
    """A /query_ability query that can't run; the message says why (an unknown field, a bad value)."""


class _NameIndex:
    """closest()'s view of one key list: stems, base names, and letter counts bounding every name's quick_ratio."""

    def __init__(self, keys: list):
        self.keys = list(keys)                              # a copy: the caller may change its list
        self.stems: dict = {}
        for k in keys:
            self.stems.setdefault(k[:len(k) - 4] if k.endswith(".png") else k, k)
        self.names = list(self.stems)
        self.bases = [(s[s.rfind("/") + 1:], s) for s in self.names]
        self.column = {c: i for i, c in enumerate(sorted({c for s in self.names for c in s}))}
        self.counts = zeros((len(self.names), len(self.column)), dtype=int32)
        for row, s in enumerate(self.names):
            for c, n in Counter(s).items():
                self.counts[row, self.column[c]] = n
        self.lengths = array([len(s) for s in self.names], dtype=int32)

    def closest(self, key: str, ratio: float) -> Optional[str]:
        """CardRepository.closest over these keys."""
        stem = key[:len(key) - 4] if key.endswith(".png") else key
        if len(stem) >= 3:
            hits = [(not base.startswith(stem), len(base), "/" in s, s) for base, s in self.bases
                    if base.startswith(stem) or "_" + stem in base]
            if hits:
                return self.stems[min(hits)[3]]
        wanted = zeros(len(self.column), dtype=int32)
        for c in stem:
            col = self.column.get(c)
            if col is not None:                             # a letter no name has matches nothing
                wanted[col] += 1
        total = self.lengths + len(stem)
        bound = where(total > 0, 2.0 * minimum(self.counts, wanted).sum(axis=1) / maximum(total, 1), 1.0)
        names = self.names                                  # quick_ratio: get_close_matches drops the rest anyway
        best = get_close_matches(stem, [names[i] for i in flatnonzero(bound >= ratio)], n=1, cutoff=ratio)
        if best and SequenceMatcher(None, stem, best[0]).ratio() > ratio:
            return self.stems[best[0]]
        return None


@dataclass
class CardRepository:
    """Manages card repository state and operations."""

    repository: str = DEFAULT_REPOSITORY
    match_ratio: float = DEFAULT_MATCH_RATIO

    def __post_init__(self):
        self.git_files: dict[str, str] = {}
        self.blobs: dict[str, str] = {}                  # png path -> blob sha, for commit-pinned links
        self.site_ids: dict[str, tuple[str, str]] = {}   # png file name -> (website card id, embed color)
        self._tree: Optional[dict] = None                # last tree, reused while GitHub says 304
        self._tree_etag = ""
        self._tree_lock = Lock()                         # held while _tree/_tree_etag change; type emojis wait on it
        self._edit_names: tuple = ("", {})               # (Card Edits.json blob sha, names)
        self._edits = None                               # that file, parsed (CardEdits)
        self._edits_text = ""                            # and its text, for CARD_TREE_JSON
        self._saved = ("", "", "")                       # (repository, ETag, Card Edits.json sha) CARD_TREE_JSON holds
        self._site_etag = ""
        self.ambiguous_names: dict[str, tuple[str, ...]] = {}
        self._spaced: dict[str, str] = {}                # underscore key -> the space-named file it stands for
        self._renamed: dict[str, tuple] = {}             # an old name's key -> (today's key, old name, today's name)
        self.choices: tuple = ()                         # autocomplete: (lowercase, shown, value), replaced whole
        self.cards_dff: DataFrame = DataFrame()
        self.rulebook_dff: DataFrame = DataFrame()
        self._lowered: tuple = (None, [])                # (cards_dff, [(label, length, matcher)]) for the fuzzy step
        self._fuzzy_lock = Lock()                        # the matchers are shared: one search at a time
        self._indexes: tuple = ()                        # closest()'s _NameIndex for the last two key lists
        self._hay: tuple = (None, "", None)             # (cards_dff, Card Edits.json sha, haystack) for text terms

    @property
    def git_filenames(self) -> list[str]:
        """Get list of all git filenames."""
        return list(self.git_files.keys())

    def populate_files(self, repository: Optional[str] = None) -> int:
        """
        Fetch and populate file listings from GitHub repository.

        Args:
            repository: Switch to this repository, only if its tree loads (/set_repo)

        Returns:
            HTTP status code (200 for success)
        """
        repo = repository or self.repository
        with self._tree_lock:                                   # _tree and _tree_etag change together
            tree, etag, status, why = self._fetch_tree(repo)
            if tree is None:                                    # keep the last good tree (B7)
                print(f"Card list: GitHub tree fetch for {repo} failed ({why}); "
                      f"keeping the last good list ({len(self.git_files)} entries)")
                return status
            self.repository, self._tree, self._tree_etag = repo, tree, etag or ""   # an ETag must match _tree
        self._build(tree)
        if repo == DEFAULT_REPOSITORY:                          # never a /set_repo one's
            self._save_tree()
        return status

    def _build(self, tree: dict, site: bool = True) -> None:
        """The registry from tree, built aside and swapped in whole (site: the website's card ids too)."""
        new = copy(self)                                        # built aside: lookups never see a half-built list
        new.git_files, new.blobs, new.ambiguous_names, new._spaced = {}, {}, {}, {}

        # Process PNG files from repository
        for item in tree.get("tree", []):
            path: str = item.get("path", "")
            if not path.endswith(".png"):
                continue

            new.blobs[path] = item.get("sha", "")
            new._add_file_to_registry(path)

        # Add Card Edits.json's names and aliases to the registry
        new._populate_edit_names()
        new._populate_titles()
        new._populate_renames()
        if site:
            new._populate_site_ids()
        for name in REGISTRY:                                   # swapped in whole
            setattr(self, name, getattr(new, name))

    def _save_tree(self) -> None:
        """The tree, its ETag and Card Edits.json's text into CARD_TREE_JSON when one changed (a failed write is logged)."""
        with self._tree_lock:
            tree, etag = self._tree or {}, self._tree_etag
            pairs = [[item.get("path", ""), item.get("sha", "")] for item in tree.get("tree", [])]
        sha = next((s for p, s in pairs if p == EDITS_PATH), "")
        edits = [sha, self._edits_text] if sha and self._edit_names[0] == sha else None
        key = (self.repository, etag, edits[0] if edits else "")
        if key == self._saved:
            return
        body = {"version": TREE_VERSION, "repository": self.repository, "etag": etag,
                "truncated": bool(tree.get("truncated")), "tree": pairs, "edits": edits}
        text = dumps(body, separators=(",", ":"), sort_keys=True)
        try:
            write_text(CARD_TREE_JSON, dumps({"sha256": sha256(text.encode()).hexdigest(), "body": body},
                                             separators=(",", ":")))
            self._saved = key
        except OSError as e:
            print(f"Card list: could not save {CARD_TREE_JSON}: {e}")

    def load_saved(self) -> int:
        """
        The card list from CARD_TREE_JSON, no network (startup, before GitHub's tree): only while no tree is loaded.
        A file that is not this repository's, or does not read or check out, is ignored; the next fetch rewrites it.

        Returns:
            The card list's entries, or 0 when the file was not used
        """
        if self._tree is not None:
            return 0
        try:
            with open(CARD_TREE_JSON, encoding="utf-8") as f:
                saved = loads(f.read())
            body = saved["body"]
            text = dumps(body, separators=(",", ":"), sort_keys=True)
            if sha256(text.encode()).hexdigest() != saved["sha256"] or body["version"] != TREE_VERSION:
                raise ValueError("checksum or version")
            edits, pairs = body["edits"], body["tree"]
            tree = {"tree": [{"path": p, "sha": s} for p, s in pairs], "truncated": bool(body["truncated"])}
            if not all(isinstance(p, str) and isinstance(s, str) for p, s in pairs) \
                    or not isinstance(body["etag"], str) \
                    or edits is not None and not (isinstance(edits[0], str) and isinstance(edits[1], str)):
                raise ValueError("not a tree")
        except FileNotFoundError:
            return 0
        except (OSError, ValueError, KeyError, TypeError, IndexError, AttributeError) as e:
            print(f"Card list: ignoring {CARD_TREE_JSON} ({type(e).__name__}: {e}); GitHub's tree replaces it")
            return 0
        if body["repository"] != self.repository:
            return 0
        parsed = None
        if edits is not None:
            from commands.psd_analyzer import CardEdits
            try:
                parsed = CardEdits.parse(edits[1])
            except ValueError:                                  # read again from GitHub
                edits = None
        with self._tree_lock:
            if self._tree is not None:                          # GitHub's came first
                return 0
            self._tree, self._tree_etag = tree, body["etag"]
            if parsed is not None:
                self._edits, self._edit_names, self._edits_text = parsed, (edits[0], parsed.names()), edits[1]
        self._saved = (body["repository"], body["etag"], edits[0] if edits else "")
        self._build(tree, site=False)                           # the website's ids: with GitHub's tree
        print(f"Card list: {len(self.git_files)} entries from the saved tree")
        return len(self.git_files)

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
        if self._tree is None:                                  # startup: the saved tree's ETag (304: no download)
            self.load_saved()
        with self._tree_lock:
            tree, etag, _, why = self._fetch_tree(self.repository)
            if tree is not None:
                self._tree, self._tree_etag = tree, etag
                return tree, ""
            if self._tree is None:
                raise RuntimeError(f"GitHub tree fetch for {self.repository} failed ({why})")
            return self._tree, why

    def _add_file_to_registry(self, path: str) -> None:
        """Add a file path to the registry, handling duplicates; a name with spaces also under underscores (aliases,
        titles and the Card Edits.json lookups build underscore keys)."""
        png_filename = path[path.rindex("/") + 1:].lower()
        if png_filename in self._spaced and self._spaced[png_filename] == self.git_files.get(png_filename):
            del self.git_files[png_filename]            # a file of that name: no longer a space-named one's lookup
            self._spaced.pop(png_filename, None)

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
        key = png_filename.replace(" ", "_")
        if " " in png_filename and key not in self.git_files:   # a file named with spaces: underscore lookups find it
            self.git_files[key] = self._spaced[key] = path

    @staticmethod
    def _get_prefixed_name(path: str, filename: str) -> str:
        """Get filename prefixed with top-level folder."""
        folder = path[:path.index('/')]
        return f"{folder}/{filename}".lower()

    def _parse_stat_query(self, ability: str) -> Optional[DataFrame]:
        """
        Run a condition on a card field (hp>10, type==creature, series~kirby, created>=2025, total>=25).

        Args:
            ability: Query term (lowercase)

        Returns:
            DataFrame of matching cards, or None when the term is not a condition

        Raises:
            QueryError: An unknown field, or a value the field can't compare
        """
        found = FIELD_PATTERN.match(ability.strip().lower())
        if not found:
            return None
        column, operator, value = found.groups()
        value = value.strip()
        column = "alias" if column == "aliases" else column     # the website takes both
        if operator == "!=":
            raise QueryError(f"For 'not', put ! before the condition: !{column}=={value}.")
        extra = column not in NUMBER_FIELDS and column not in TEXT_FIELDS and column in self.cards_dff.columns
        if column not in NUMBER_FIELDS and column not in TEXT_FIELDS and not extra:
            hint = FIELD_HINTS.get(column) or next(iter(get_close_matches(column, NUMBER_FIELDS + TEXT_FIELDS, 1, 0.6)), "")
            raise QueryError(f"Unknown field '{column}'{f' (did you mean {hint}?)' if hint else ''}. Fields: "
                             f"{', '.join(NUMBER_FIELDS)} (numbers); {', '.join(TEXT_FIELDS)} (== or ~). "
                             f"To search the text itself, put it in quotes.")
        col = self._field(column)
        kept = None
        if column in NUMBER_FIELDS or (extra and operator != "~" and value.lstrip("-").isdigit()):   # any column too
            if operator == "~" or not value.lstrip("-").isdigit():
                raise QueryError(f"{column} takes a number: {column}>5, {column}==3.")
            if col is not None:
                col = to_numeric(col, errors="coerce")
                number = int(value)
                kept = {"<": col < number, "<=": col <= number, "==": col == number, ">=": col >= number, ">": col > number}[operator]
        elif column == "created" and DATE_PATTERN.match(f"created{operator}{value}"):
            if col is not None:                          # at the precision given: created>=2025, created==2023-10
                day = col.where(col.notna(), "").astype(str).str[:len(value)]
                kept = col.notna() & {"<": day < value, "<=": day <= value, "==": day == value, ">=": day >= value,
                                      ">": day > value}[operator]
        elif operator in ("==", "~"):
            value = value.strip('"')
            if col is not None:
                text = col.where(col.notna(), "").astype(str).str.lower()
                kept = col.notna() & ((text == value) if operator == "==" else text.str.contains(value, regex=False))
        else:
            raise QueryError(f"{column} takes == or ~" + (" or a date: created>=2025, created<2021-06." if column == "created" else "."))
        return self.cards_dff[kept] if kept is not None else self.cards_dff.iloc[0:0]

    def _field(self, column: str) -> Optional[Series]:
        """A field over cards_dff: a column, the stat total, or the name players see (printed title, else name)."""
        df = self.cards_dff
        if column == "total":                            # as the website: missing stats count 0, no stats at all: none
            stats = [c for c in ("hp", "def", "atk", "spd") if c in df]
            return df[stats].apply(to_numeric, errors="coerce").sum(axis=1, min_count=1) if stats else None
        if column == "name":
            names = Series(df.index, index=df.index)
            return df["display_name"].where(df["display_name"].notna(), names) if "display_name" in df else names
        if column == "alias":                            # website names and aliases (Card Edits.json)
            names = self._edit_names[1]
            return Series([" ".join(names.get(CardAuthors.key(str(label)), [])) for label in df.index], index=df.index,
                          dtype=object)
        return df[column] if column in df else None

    def _haystack(self) -> Series:
        """What a text term searches, lowercased (as the website): name, printed title, website names and aliases, ability."""
        df, sha = self.cards_dff, self._edit_names[0]
        if self._hay[0] is not df or self._hay[1] != sha:
            names = self._edit_names[1]
            titles = df["display_name"] if "display_name" in df else Series(None, index=df.index, dtype=object)
            rows = []
            for label, title, ability in zip(df.index, titles, df["ability"]):
                parts = [str(label)] + ([title] if isinstance(title, str) else []) + list(names.get(CardAuthors.key(str(label)), []))
                rows.append("\n".join(parts + [ability if isinstance(ability, str) else ""]).lower())
            self._hay = (df, sha, Series(rows, index=df.index, dtype=object))
        return self._hay[2]

    @staticmethod
    def _text_pattern(text: str) -> str:
        """A regex for a text term: the text, with damage also finding DMG, DEF defense, SPD speed, HP health and back."""
        parts = SYNONYM_WORDS.split(text)
        out = []
        for i, part in enumerate(parts):
            if i % 2 == 0:
                out.append(escape(part))
                continue
            other = SYNONYMS[part]
            other = r"\b" + other + r"\b" if len(other) < len(part) else other      # dmg as a word, not in "admg"
            out.append("(?:" + part + "|" + other + ")")
        return "".join(out)

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
            return combined[~combined.index.duplicated(keep='first')]

        # Handle AND queries
        queries = self._split_unquoted(ability, "&")
        if len(queries) > 1:
            result = self.ability_search_engine(queries[0].strip())
            for q in queries[1:]:
                result = result[result.index.isin(self.ability_search_engine(q.strip()).index)]
            return result

        # NOT: any term, conditions too (!type==creature)
        if ability.startswith("!"):
            inner = self.ability_search_engine(ability[1:].strip())
            return self.cards_dff[~self.cards_dff.index.isin(inner.index)]

        # Conditions on fields
        stat_result = self._parse_stat_query(ability)
        if stat_result is not None:
            return stat_result

        # Text: names, aliases and ability (as the website); "quoted" as whole words
        hay = self._haystack()
        if len(ability) > 1 and ability.startswith('"') and ability.endswith('"'):
            pattern = r"(?:^|\s|$|\b)" + escape(ability.strip('"')) + r"(?:^|\s|$|\b)"
            return self.cards_dff[hay.str.contains(pattern, na=False).values]
        exact_matches = self.cards_dff[hay.str.contains(self._text_pattern(ability), na=False).values]

        # Fuzzy matches: near-whole ability texts
        fuzzy_matches = self.cards_dff.loc[self._fuzzy_labels(ability)]

        combined = concat([exact_matches, fuzzy_matches])
        return combined[~combined.index.duplicated(keep='first')]

    def _fuzzy_labels(self, ability: str) -> Index:
        """Cards whose SequenceMatcher ratio with the query beats match_ratio, in frame order."""
        frame = self.cards_dff
        la, cut = len(ability), self.match_ratio
        keep = []
        with self._fuzzy_lock:
            if self._lowered[0] is not frame:                   # one matcher per card (its ability as b), per frame
                self._lowered = (frame, [(label, len(text), SequenceMatcher(None, "", text))
                                         for label, text in ((k, v.lower()) for k, v in frame["ability"].items())])
            for label, lb, matcher in self._lowered[1]:
                if la + lb and 2.0 * min(la, lb) / (la + lb) <= cut:   # ratio()'s upper bound: can't match
                    continue
                matcher.set_seq1(ability)
                if matcher.quick_ratio() > cut and matcher.ratio() > cut:   # quick_ratio bounds ratio
                    keep.append(label)
        return Index(keep, dtype=frame.index.dtype)

    def rulebook_search_engine(self, search_text: str) -> DataFrame:
        """
        Search rulebook entries for matching text.

        Args:
            search_text: Text to search for (case-insensitive)

        Returns:
            DataFrame of matching rulebook entries
        """
        words = search_text.split()
        if not words:
            return self.rulebook_dff
        pattern = r"\s+".join(escape(w) for w in words)           # any spaces or line breaks between the words
        return self.rulebook_dff[self.rulebook_dff["ability"].str.contains(pattern, na=False, case=False)]

    def _populate_edit_names(self) -> None:
        """Website aliases and names from the card repo's Card Edits.json (EDITS_PATH)."""
        from commands.psd_analyzer import CardEdits
        shas = {item.get("path", ""): item.get("sha", "") for item in (self._tree or {}).get("tree", [])}
        path = EDITS_PATH if shas.get(EDITS_PATH) else None
        if path is None:
            return
        sha = shas[path]
        if self._edit_names[0] != sha:                           # download only when the file changed
            try:
                text = blob_text(self.repository, sha)
                edits = CardEdits.parse(text)
            except Exception as e:
                print(f"Could not read {path}: {e}")
            else:
                self._edits, self._edit_names, self._edits_text = edits, (sha, edits.names()), text
        # a failed read keeps the last good names
        for card, extra in self._edit_names[1].items():
            self._add_lookup(card, extra)

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
        key = self._key_of(card)
        if key:
            for name in extra:
                self.git_files.setdefault(f"{name.strip().lower().replace(' ', '_')}.png", self.git_files[key])

    def _key_of(self, card: str) -> Optional[str]:
        """The registry key of a card's PNG (its file name, else a folder-prefixed one), or None."""
        target = f"{card.replace(' ', '_')}.png".lower()
        if target in self.git_files:
            return target
        return next((k for k in self.git_files if k.endswith("/" + target)), None)

    def _populate_renames(self) -> None:
        """Old names of renamed cards (their archive's renamed_to, followed to a card of today) and the autocomplete list."""
        try:
            archived, cards = stats_db.archived(), dict(stats_db.stats)
        except RuntimeError:                                     # /update_stats writing meanwhile: after it, again
            return
        renamed = {}
        for name, record in archived.items():
            target, seen = name, set()
            while target not in cards and target in archived and target not in seen:
                seen.add(target)
                target = (getattr(archived[target], "archived", None) or {}).get("renamed_to") or ""
            key = self._key_of(target) if target in cards else None
            old = getattr(record, "display_name", None) or name
            for spelling in (name, old):                         # Pizza (item) and its printed Pizza
                old_key = self._normalize_card_name(spelling)
                if key and old_key not in self.git_files:        # a name of today wins
                    renamed.setdefault(old_key, (key, old, getattr(cards[target], "display_name", None) or target))
        self._renamed = renamed
        self._populate_choices(cards)

    def _populate_choices(self, cards: dict) -> None:
        """Autocomplete's names: cards (printed titles first), website names and aliases, renamed cards' old names."""
        found, shown_as = {}, {}

        def add(value: str, shown: str) -> None:
            value = value.strip()
            if value and len(value) <= 100 and value.lower() not in found:
                found[value.lower()] = (shown[:100], value)

        for name, card in cards.items():
            title = getattr(card, "display_name", None)
            shown_as[CardAuthors.key(name)] = title or name      # Card Edits.json's keys, as players know the card
            if getattr(card, "hidden", False) or self._key_of(name) is None or (
                    getattr(card, "duplicate_of", None) and not getattr(card, "variant_of", None)):   # a variant: its own title
                continue
            add(title or name, title or name)
            if title and title.lower() != name.lower():
                add(name, f"{name} ({title})")
        for card, extra in self._edit_names[1].items():
            shown = shown_as.get(card, card)
            for alias in extra:
                add(alias, alias if alias.lower() == shown.lower() else f"{alias} ({shown})")
        for _, old, now in self._renamed.values():
            add(old, f"{old} (now {now})")
        self.choices = tuple(sorted((shown.lower(), shown, value) for shown, value in found.values()))

    def name_choices(self, current: str) -> list:
        """
        Autocomplete for a card-name option (on the event loop: reads the choices tuples only).

        Args:
            current: What is typed so far

        Returns:
            Up to 25 (shown, value): names starting with it, then with a word that does, then holding it
        """
        text = " ".join(current.lower().replace("_", " ").split())
        choices = self.choices                                   # one read: a rebuild swaps it whole
        if not text:
            return [(c[1], c[2]) for c in choices[:25]]
        ranked = []
        for low, shown, value in choices:
            at = low.find(text)
            if at >= 0:
                ranked.append((0 if at == 0 else 1 if low[at - 1] in " (-." else 2, len(low), low, shown, value))
        ranked.sort()
        return [(r[3], r[4]) for r in ranked[:25]]

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
        """The query tables from the card database."""
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

            # Cards DataFrame (the Photoshop rule pages, still parsed, left out)
            has_ability = cards_df["ability"].notna()
            is_rulebook = cards_df["path"].str.contains(RULEBOOK_DIR, na=False)
            self.cards_dff = cards_df[has_ability & ~is_rulebook].copy()
            self.cards_dff["ability"] = self.cards_dff["ability"].str.lower()

            print(f"Loaded {len(self.cards_dff)} cards")
        # the rulebook: its chapters (db/rulebook.json), as written: /query_rulebook quotes their lines
        chapters = stats_db.rulebook
        self.rulebook_dff = DataFrame({"ability": [c["text"] for c in chapters]}, index=[chapter_label(c) for c in chapters])
        print(f"Loaded {len(self.rulebook_dff)} rulebook chapters")
        self._populate_renames()                        # renamed cards and autocomplete follow the new stats

    def _normalize_card_name(self, query: str) -> str:
        """Normalize card name for searching."""
        card = query.replace(" ", "_").lower()
        return card if card.endswith(".png") else f"{card}.png"

    def _find_closest_match(self, query: str) -> Optional[str]:
        """Find closest matching card filename (lookup's key)."""
        return self.lookup(query)[0]

    def lookup(self, query: str) -> tuple:
        """
        A card name's registry key, and how it was found.

        Args:
            query: Card name, printed title, alias or a renamed card's old name

        Returns:
            (key, "exact" | "renamed" | "guess", the old name when renamed) or (None, "", "")
        """
        key = self._normalize_card_name(" ".join(query.split()))
        if key in self.git_files:
            return (key, "exact", "")
        renamed = self._renamed.get(key)
        if renamed:
            return (renamed[0], "renamed", renamed[1])
        found = self.closest(key, self.git_filenames)
        return (found, "guess", "") if found else (None, "", "")

    def closest(self, key: str, keys: list) -> Optional[str]:
        """
        The best guess for a key among keys: a name starting with it or with a word that does (3+ letters; the
        shortest, a plain key before its folder-prefixed ones), else the most similar name above match_ratio, compared
        without ".png" (it raised every score).
        """
        for index in self._indexes:                             # one per key list, kept while it is unchanged
            if index.keys == keys:
                return index.closest(key, self.match_ratio)
        index = _NameIndex(keys)
        self._indexes = (index,) + self._indexes[:1]
        return index.closest(key, self.match_ratio)

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

    closest, how, old = card_repo.lookup(query)
    if not closest:
        print(f"Search miss: /query {query!r}")         # misses that could become aliases
        queue_message(interaction, "No card found!")
        return
    note = None
    if how == "renamed":
        note = f"**{_label(closest)}** (renamed from {old})"
    elif how == "guess":
        print(f"Search guess: /query {query!r} -> {_label(closest)}")
        note = f"Closest match: **{_label(closest)}**"
    _post_card(interaction, closest, note=note)


def _label(key: str) -> str:
    """A registry key's card as players know it: its printed title, else its name."""
    card = _stats_card(key)
    if card is not None:
        return getattr(card, "display_name", None) or card.name
    return splitext(basename(card_repo.git_files.get(key, key)))[0].replace("_", " ")


def trim_url(url: str) -> str:
    """A matched URL without its trailing punctuation, or a ")" it doesn't open (art.js trimUrl)."""
    while True:
        cut = url.rstrip(".,;:!?")
        cut = cut[:len(cut) - 1] if cut.endswith(")") and cut.count(")") > cut.count("(") else cut
        if cut == url:
            return url
        url = cut


def art_text(text: str) -> str:
    """An artwork source as one Discord line: each http(s) URL a masked link named by its host, the rest escaped."""
    out, at = [], 0
    for found in URL_PATTERN.finditer(text):
        url = trim_url(found.group(0))
        host = url.split("/")[2] if url.count("/") >= 2 else url
        target = url.replace("(", "%28").replace(")", "%29")
        out.append(escape_markdown(text[at:found.start()]) + f"[{escape_markdown(host.removeprefix('www.'))}]({target})")
        at = found.start() + len(url)
    out.append(escape_markdown(text[at:]))
    return " ".join("".join(out).split())


def _post_card(interaction: Interaction, closest: str, url: Optional[str] = None, note: Optional[str] = None) -> None:
    """The card's image (an embed when it has a website name, subseries, note, artwork source or original) and any
    ambiguity; url: an image other than today's (/query as_of); note: a line above it (a guess, a rename)."""
    url, page = url or card_repo.get_card_url(closest), card_repo.site_url(closest)
    links = [("Card page", page)] if page else None
    card = _stats_card(closest)
    shown = getattr(card, "display_name", None) if card else None
    subseries = getattr(card, "subseries", None) if card else None
    extra = [f"Same card as {card.duplicate_of}" if card and getattr(card, "duplicate_of", None) else "",
             getattr(card, "note", None) or "" if card else ""]
    arts = art_list(getattr(card, "art", None), getattr(card, "ability_art", None)) if card else []   # a card's artwork sources
    extra += [f"Artwork source{f' #{i}' if len(arts) > 1 else ''}: {art_text(source)}" for i, source in enumerate(arts, 1)]
    if card and (shown or subseries or any(extra)):     # website name, subseries, note, art or duplicate: an embed says so
        extra.insert(0, " · ".join(s for s in (card.series, subseries) if s))      # "Series · Subseries"
        created = getattr(card, "created", None)
        if created:
            extra.append(f"Created {'about ' if created.get('provisional') else ''}{created['date']}")
        found = card_repo.site_ids.get(basename(card_repo.git_files.get(closest, "")).lower())
        embed = Embed(title=shown or card.name, url=page, description="\n".join(e for e in extra if e) or None,
                      color=int(found[1][1:], 16) if found and found[1] else None)
        embed.set_image(url=url)
        queue_message(interaction, note, embed=embed, links=links)
    else:
        queue_message(interaction, f"{note}\n{url}" if note else url, links=links)

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
    found = _history_lookup(query)
    return found[:2] if found else None


def _history_lookup(query: str) -> Optional[tuple]:
    """history_match's (key, name), and whether it was a guess."""
    from commands.psd_analyzer import card_name, stats_db
    try:
        archived = {card_repo._normalize_card_name(name): name for name in stats_db.archived()}
    except RuntimeError:                                 # /update_stats writing meanwhile
        archived = {}
    key = card_repo._normalize_card_name(query)
    if key in archived and key in card_repo.git_files and \
            card_repo._normalize_card_name(card_name(card_repo.git_files[key])) != key:
        return (None, archived[key], False)              # an alias or another card's PNG claims the archived name
    guess = key not in card_repo.git_files and key not in archived
    if guess:
        key = card_repo.closest(key, card_repo.git_filenames + list(archived))     # /query's guess
        if not key:
            return None
    if key in card_repo.git_files:
        return (key, _stats_name(card_name(card_repo.git_files[key])), guess)
    return (None, archived[key], guess)


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
    found = _history_lookup(query)
    if found is None:
        print(f"Search miss: /query {query!r} as_of")
        queue_message(interaction, "No card found!")
        return
    key, name, guess = found
    label = f"**{name}**"
    if guess:
        print(f"Search guess: /query {query!r} as_of -> {name}")
    current = stats_db.stats.get(name)
    if current is not None and getattr(current, "duplicate_of", None) and not getattr(current, "variant_of", None):
        # a copy: its original's history (a variant picture has its own)
        label = f"**{name}** (same card as {current.duplicate_of})"
        name = current.duplicate_of
        current = stats_db.stats.get(name)
    label = f"Closest match: {label}" if guess else label
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
        queue_message(interaction, "No card data yet. Run `/update_stats` first.")
        return

    results = _search(interaction, ability)
    if results is None:
        return

    total_results = len(results)
    results_to_show = total_results if limit < 0 else min(limit, total_results)

    found = f"Found {total_results} results for '{ability}'!"
    shown = [basename(index) for index in results.index[:results_to_show]]   # index: the card's name
    done = f"Displayed {results_to_show} of {total_results} results for '{ability}'."
    links = [("All results on the website", f"{SITE_URL}/#/cards?q={quote(ability.strip())}")]
    if len(shown) > 1:
        post_grids(interaction, shown, found, done, links)
        return

    queue_message(interaction, found)
    for name in shown:                                   # one card: its image link, previewed
        queue_message(interaction, card_repo.get_card_url(name) or name)   # its name when the PNG is not in the tree
    queue_message(interaction, done, links=links)


def post_grids(interaction: Interaction, names: list, first: str, last: str, links: Optional[list] = None) -> None:
    """
    Many cards as grid images (card_grid), one per message with its cards listed (number, name, website page): one
    message per PER_GRID cards instead of one per card. first heads the first message, last ends the last (with links).
    """
    from commands.card_grid import PER_GRID, grids, prune

    cards, lines = [], []
    for number, name in enumerate(names, 1):
        key = card_repo._normalize_card_name(name)
        path = card_repo.git_files.get(key) or card_repo.git_files.get(key.replace("_", " "))
        page = card_repo.site_url(key)
        blob = card_repo.blobs.get(path, "") if path else ""
        if blob and not pinned_ref(card_repo.repository, path, blob):
            blob = ""                                    # a branch link can lag (CDN): fetched, not cached
        cards.append((f"{number}. {name}", card_repo.get_card_url(name) if path else None, blob))
        shown = escape_markdown(name).replace("[", "(").replace("]", ")")
        lines.append(f"**{number}.** [{shown}](<{page}>)" if page else f"**{number}.** {shown}")
    for at, image in enumerate(grids(cards)):
        text = "\n".join(lines[at * PER_GRID:(at + 1) * PER_GRID])
        if at == 0:
            text = f"{first}\n{text}"
        end = at == (len(cards) - 1) // PER_GRID
        if end and len(text) + len(last) < MAX_CONTENT:
            text, last = f"{text}\n{last}", ""
        kwargs = {"links": links} if end and links and not last else {}
        queue_file(interaction, File(BytesIO(image), filename=f"cards-{at + 1}.jpg"), content=text, **kwargs)
    if last:
        queue_message(interaction, last, links=links)
    if card_repo.blobs:
        prune(set(card_repo.blobs.values()))


def query_ability_num_occur(interaction: Interaction, ability: str) -> None:
    """
    Query the number of cards matching an ability.

    Args:
        interaction: Discord interaction
        ability: Ability search query
    """
    if card_repo.cards_dff.empty:
        queue_message(interaction, "No card data yet. Run `/update_stats` first.")
        return

    results = _search(interaction, ability)
    if results is not None:
        queue_message(interaction, f"Found {len(results)} cards matching '{ability}'.")


def _search(interaction: Interaction, ability: str) -> Optional[DataFrame]:
    """/query_ability's search; a query it can't run is answered here (None). Misses are logged, for aliases."""
    try:
        results = card_repo.ability_search_engine(ability.strip().lower())
    except QueryError as e:
        print(f"Search error: /query_ability {ability!r}")
        queue_message(interaction, str(e))
        return None
    if results.empty:
        print(f"Search miss: /query_ability {ability!r}")
    return results


def query_rulebook(interaction: Interaction,
                   search_text: str,
                   limit: int = -1) -> None:
    """
    Search the rulebook: the matching lines under their chapter names, chapters with the most matches first.

    Args:
        interaction: Discord interaction
        search_text: Text to search for in rulebook
        limit: Maximum number of chapters to show (-1 for all)
    """
    if card_repo.rulebook_dff.empty:
        queue_message(interaction, "No card data yet, or no rulebook chapters. Run `/update_stats` first.")
        return
    text = " ".join(search_text.split())
    if not text:
        queue_message(interaction, "Give a word or phrase to search the rulebook for.")
        return

    pages = rulebook_lines(card_repo.rulebook_search_engine(text), text)
    links = [("Rulebook on the website", f"{SITE_URL}/#/rules?q={quote(text)}")]
    if not pages:
        print(f"Search miss: /query_rulebook {text!r}")
        queue_message(interaction, f"No rulebook lines match '{text}'.", links=links)
        return
    shown = pages if limit < 0 else pages[:limit]
    count = sum(len(lines) for _, lines in pages)
    head = f"'{text}' is on {count} line{'s' if count != 1 else ''} of {len(pages)} rulebook chapter{'s' if len(pages) != 1 else ''}"
    head += f" (showing {len(shown)} chapters):" if len(shown) < len(pages) else ":"
    blocks = [head] + [f"**{name}**\n" + "\n".join(f"- {_bold(render_type_tokens(line, escape=True), text)}" for line in lines)
                       for name, lines in shown]
    messages = [blocks[0]]
    for block in blocks[1:]:                         # whole chapters per message while they fit
        if len(messages[len(messages) - 1]) + len(block) + 2 <= BREAK_LEN:
            messages[len(messages) - 1] += "\n\n" + block
        else:
            messages.append(block)                   # a page longer than a message: the dispatch splits it
    for i, message in enumerate(messages):
        queue_message(interaction, message, links=links if i == len(messages) - 1 else None)


def rulebook_lines(pages: DataFrame, text: str) -> list:
    """
    The lines of each page holding text (any case), cut to SNIPPET around it.

    Returns:
        [(page name, [lines])], the most lines first, then by name; a match across a line break: that passage
    """
    found, low = [], text.lower()
    for name, ability in pages["ability"].items():
        lines = [BULLET.sub("", " ".join(line.split())) for line in str(ability).split("\n")]
        lines = [line for line in lines if low in line.lower()]
        if not lines:                                # the text runs over a line break
            lines = [" ".join(str(ability).split())]
        found.append((name, [_cut(line, low) for line in lines]))
    found.sort(key=lambda page: (-len(page[1]), str(page[0]).lower()))
    return found


def _cut(line: str, low: str) -> str:
    """line, or SNIPPET characters of it around the first match."""
    if len(line) <= SNIPPET:
        return line
    at = max(line.lower().find(low), 0)
    start = max(0, min(at - SNIPPET // 3, len(line) - SNIPPET))
    return ("…" if start else "") + line[start:start + SNIPPET].strip() + ("…" if start + SNIPPET < len(line) else "")


def _bold(line: str, text: str) -> str:
    """The searched text in bold where it stands as written (escaped markdown or an emoji can hide it), never inside
    an emoji's <:name:id> (minion in <:defensive_minion:id>)."""
    found = compile(escape(text), IGNORECASE)
    parts = EMOJI.split(line)
    return "".join(p if i % 2 else found.sub(lambda m: f"**{m.group(0)}**", p) for i, p in enumerate(parts))


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
    status = await to_thread(card_repo.populate_files, new_repo)  # off the event loop (retries can take seconds)

    if status != 200:
        await interaction.followup.send(f"Error {status or 'no response'} when connecting to {new_repo}; still using {old_repo}.")
    else:
        await interaction.followup.send(f"Repository set to {card_repo.repository}")


# ========================
# Initialization
# ========================
card_repo = CardRepository()
def init_query() -> None:
    """The card list at startup (init_psd, after the card data): GitHub's tree, with the saved tree's ETag (304: kept)."""
    before = card_repo._tree
    status = card_repo.populate_files()
    if status != 200:
        print(f"Warning: Failed to populate files from repository (status {status})")
        if card_repo._tree is not None:             # the saved tree: rebuilt with the loaded data's titles
            card_repo._build(card_repo._tree)
    else:
        fetched = card_repo._tree is not before
        print(f"Card list: {len(card_repo.git_files)} entries, tree {'fetched' if fetched else 'unchanged'}")


# ========================
# Module Exports
# ========================
__all__ = [
    'QueryError',
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
