# AutoCroissant code-health audit (read-only, 2026-10-20)

Scope: main.py and commands/*.py at HEAD 7f6bd05. This was a read-only pass. Nothing in the repo was edited, and the
pickles were only read from copies in /tmp. Interpreter: ac-py314-psd125 (Python 3.14.8). Tools: vulture 2.16,
pyflakes, and grep/AST scripts. Cython 3.3.0 (the production version) was installed into
/tmp/codehealth/cy with `--target`, so the venv was not touched.

Experiments and their scripts are in /tmp/codehealth/ (the reboot clears it):
`refs.py`, `refs2.py` (dead-code reference counts), `cyexp/` (pickle class-path experiment), `hint/` (what Cython does
with annotations), `cmp/` (a HEAD copy built with the real setup.py, parsed with parse_all.py both compiled and
uncompiled).

---

## 0. Ranked actions (value / risk)

| # | Action | Value | Risk | Evidence |
|---|--------|-------|------|----------|
| 1 | Guard `CardClassifier.classify` against shallow paths. In the compiled build (the jetson), `Card.psd` gives **SIGSEGV (exit 139)** and `Items/Card.psd` gives **SIGBUS (exit 138)**; the .py raises IndexError/ValueError | high: a PSD committed at the repo root or straight into `Items/` kills the bot process in /update_stats | low: changes behaviour only for inputs that crash today | §1.4 |
| 2 | Write down: **type hints in INCLUDE_FILES modules are not inert**. Cython enforces and coerces them; measured table in §3.1. Add it to change-control "looks safe but isn't" and to build-and-env | high (it governs every hint change and the split) | none (docs only) | §3.1 |
| 3 | Delete confirmed dead code: `analytics.parse_time_str`, `psd_analyzer.get_old_stats_as_dict` (+ its `__all__` entry), `CardInfo.from_dict`, the never-read `fake.scheduled` and `fake.reminder_id`, the unused `name =` binding in classify (keep the `.pop()`), and 2 placeholder-less f-strings | low-medium | low | §1 |
| 4 | Rename the duplicate handlers: `slash_set_ratio` #2 (main.py:317, /set_repo) to `slash_set_repo`, and `slash_delete_song` #2 (main.py:617, /delete_all_music) to `slash_delete_all_music` (C5) | medium | low: Discord and slash_registry key on `cmd.name` | §1.3 |
| 5 | One GitHub helper module, `commands/github_api.py`, uncompiled. It replaces GIT_TOKEN ×4, auth-header dicts ×5, tree fetch ×3, blob fetch ×3 and the token-status print ×2 (2 identical lines at every import). Per-caller ETags stay separate | medium (B7 retry then lives in one place) | low-medium | §2.1 |
| 6 | Move repo-path constants to global_config.py: card repo, branch, Types/ folders, MDW and TTS paths, the Rulebook marker. `EDITS_PATH` is defined twice today | medium | low (values identical) | §5 |
| 7 | Split psd_analyzer.py into models / edits / parser / traverser, with psd_analyzer.py as the facade. Pin the pickled class paths. Add the 4 new modules to INCLUDE_FILES | high (maintainability) | medium. Gated: parse_all 0 diffs on .py **and** compiled, plus pickle byte-identity | §4 |
| 8 | Small helpers: `queue_long_message` (7 identical loops), `say()` (notify-or-print ×5), `_format_metadata` (a 20-line block copied twice), `card_name(path)` (×9), one card key (`tts_save._card_key` matches `CardAuthors.key`, proven) | low-medium | low | §2.2 to §2.6 |
| 9 | Naming: `perms_check` returns True for a **non-admin** (24 references). `num_new` also counts changed cards. `dirty_files` holds card names. PSDParser's `stats_db` parameter shadows the module global | low-medium | low (mechanical) | §3.2 |
| 10 | Parse the tree JSON once per /update_stats instead of 7 times | low: 12 ms per run measured, so this is code health, not speed | low | §2.1 |

Two numbers worth having before anything else:
- **The compiled and uncompiled parsers agree today**: 813 cards, 0 differences. HEAD was built with the real setup.py
  and parse_all.py run with the 7 .so loaded, then without them. Parse time was 10 s for both, measured twice each,
  so compiling does not speed up parsing (psd-tools dominates). The existing regression gate only runs the .py, so
  this is the first proof that it also covers production.
- **Import cost of `commands.psd_analyzer`: 0.62 to 0.69 s** (5 runs, `-X importtime`). psd_tools alone takes 0.12 s.
  After the split, a parser-only consumer (parse_all workers, parse_one, gap_trace) could import in about 0.12 s.
  That only helps the tooling; the bot imports everything anyway.

---

## 1. Dead code

### 1.1 vulture (min-confidence 0) over main.py + commands/

67 hits. Each was checked by grep across main.py, commands/, the diagnostics scripts, /tmp/psdreg/parse_all.py, both
archive branches and TTSCardMaker.

| Hit | Verdict |
|-----|---------|
| main.py: `on_ready` + 55 `slash_*` functions | **Live.** Registered by `@client.event` / `@tree.command` at def time |
| main.py:89 `intents.message_content` | Live (discord.py reads the attribute) |
| psd_analyzer.py:1427, query_card.py:347 `reply.encoding` | Live (a requests attribute; it changes `.text`) |
| psd_analyzer.py:1498 `_populate_types_from_local`, :1613 `_icons_from_local` | **Live.** Used by parse_one.py, gap_trace.py, parse_all.py and archive/local-update-stats (kept on purpose, archaeology 10.13) |
| analytics.py:75/77 `_ReminderUser.global_name/.bot`, utils.py:125 `FakeUser.bot` | Keep. Duck-typed stand-ins for discord.User; nothing reads them now, but they cost nothing |
| analytics.py:154 **`parse_time_str`** | **Dead.** No caller anywhere, including both archive branches. Archaeology 10.13 already lists it as a candidate |
| analytics.py:641 **`fake.scheduled = True`** (and :642 `fake.reminder_id`) | **Dead.** Set, never read (no `.scheduled`/`reminder_id` attribute reads; the `reminder_id` names elsewhere are locals) |

### 1.2 Found by grep/AST beyond vulture

vulture misses some names because another object uses the same name.
- **`CardInfo.from_dict`** (psd_analyzer.py:510). No call site in the repo, the scripts or the branches. vulture
  counted `DataFrame.from_dict` as a use. It is also lossy: it drops every field after `author` (11 fields,
  impossibility-cards-reference §442). research-frontier:292 mentions it as a hook for a future migration. Delete
  it, or keep it with a note; nothing calls it.
- **`get_old_stats_as_dict`** (psd_analyzer.py:1813). Its only references are the `__all__` entry and the
  definition. Only definitions exist on the branches, and the website does not import bot code. Delete both.
- pyflakes: `main.py:184` and `:200` are f-strings without placeholders. `psd_analyzer.py:616`: `name = folders.pop()`
  never uses `name`, but the pop is needed, so keep `folders.pop()`.
- Every UPPER_CASE constant in every module has at least 2 references (scripted check). No dead constants.
- `__all__` audit: no names listed in `__all__` that don't exist. Two cross-module imports bypass `__all__`:
  `query_card` imports `EDITS_PATH`/`CardEdits` from psd_analyzer, and `tts_save` imports `REPOSITORY` from
  repo_history. utils.py has no `__all__`; that is the documented exception.

### 1.3 Duplicate names (C5)

`slash_set_ratio` is defined at main.py:307 and :317 (the second is /set_repo). `slash_delete_song` is defined at :610
and :617 (the second is /delete_all_music). Renaming them is safe: `slash_registry[cmd.name] = cmd.callback` keys on the
command name, and reminder.pkl stores command names. Update the change-control and architecture skill text that
describes them.

### 1.4 Crash found while auditing: `CardClassifier.classify` under `boundscheck=False`

psd_analyzer.py:616-629. `top_folder = folders[0]` after `folders.pop()`, and `folders[len(folders) - 2]` for
`Items`/`Creatures`. With one folder or none, those index an empty list or index -1. The .py raises; the compiled
module reads out of bounds:

```
compiled  classify("Card.psd")        -> exit 139 (SIGSEGV)
compiled  classify("Items/Card.psd")  -> exit 138 (SIGBUS)
.py       classify("Card.psd")        -> IndexError ; "Items/Card.psd" -> ValueError
```

No such PSD exists today (904 PSDs, none at the root). But `_process_files_from_response` parses any `*.psd` outside
Markers/MDW, so one stray commit would take the jetson bot down mid-/update_stats. parse_one.py's fallback
(`relative_path = psd_path.name`) hits the same crash in any checkout that has the .so. The fix: return UNKNOWN when
`folders` is empty or too shallow for the branch being taken. This is a behaviour change only for inputs that crash
today; it is robustness work (2.5).

---

## 2. Duplicate helpers and copy-pasted blocks

### 2.1 GitHub access (one home: `commands/github_api.py`, uncompiled, imports requests + optional config)

| Duplicate | Sites |
|-----------|-------|
| `GIT_TOKEN: str = getattr(config, "GIT_TOKEN", "")` | psd_analyzer.py:44, query_card.py:26, type_emojis.py:21, tts_save.py:66 |
| Auth-header dict `{'Authorization': f'token {GIT_TOKEN}'}` | psd_analyzer.py:543 (`StatsDatabase._headers`!), query_card.py:48, type_emojis.py:100 and :110, tts_save.py:726 |
| Token-status print | psd_analyzer.py:545-547 and query_card.py:51-53. Importing psd_analyzer prints **2 identical lines** (measured) |
| Tree fetch `.../git/trees/main?recursive=1` | query_card.py:74 (ETag), psd_analyzer.py:1464 (ETag), type_emojis.py:101 (no ETag, no status check before `.json()`, the B7 path) |
| Blob fetch `Accept: application/vnd.github.raw` | psd_analyzer.py:1414 (stream, 120 s), query_card.py:344 (30 s), type_emojis.py:111 (30 s), tts_save.py:724 (`_api`, raw=True) |
| `Github(login_or_token=GIT_TOKEN)` | psd_analyzer.py:1401 and :1408 (two clients, different per_page) |

Proposed API: `GIT_TOKEN`, `HEADERS`, `fetch_tree(repository, etag="") -> (status, tree|None, etag)`,
`fetch_blob(repository, sha, stream=False, timeout=30)`, and `api_base(repository)`. tts_save can keep its own `_api`
(it adds `X-GitHub-Api-Version` and the POST/PATCH calls) and take `HEADERS`/`GIT_TOKEN` from the helper.
**Keep the three ETag caches separate.** `psd_analyzer._tree_etags` means "unchanged since the last /update_stats"
and a 304 makes update_stats return early. If it shared query_card's startup ETag, the first /update_stats after a
restart would wrongly answer "Nothing changed". Share the function, not the cache.
Startup currently downloads the full tree twice: `init_query`, then `init_type_emojis` → `_list_type_icons`. The tree
is 597,873 bytes and 2,379 entries (fetched once, unauthenticated, during this audit). The second fetch could reuse
`card_repo._tree`; that is the network phase's call.
Inside the traverser, `response.json()` re-parses the same tree **7 times** (psd_analyzer.py:1470, 1493, 1515, 1527,
1552, 1566, 1637). Measured: 2.06 ms per parse, so 14.4 ms against 2.1 ms. Pass `self.tree` or a `paths` list to the
`_*_from_response` helpers instead. This is code health, not speed.

### 2.2 Raw URL builders (one home: repo_history.py)

- repo_history.py:156 `raw_url` → `https://raw.githubusercontent.com/{repo}/{ref}/{quote(path, safe='/')}`.
- tts_save.py:45 `RAW` plus `f"{RAW}{sha or 'main'}/{_encode(path)}"` (:249, :438, :859). It encodes with
  `URL_SAFE = "/'!()*-._~"`. **The two encodings differ on purpose**: save links follow GitHub's raw style.
- tts_save.py:245 gets the pinned commit by **slicing raw_url's output**: `raw_url(...)[len(RAW):].split("/", 1)[0]`.
  Proposal: `repo_history.pinned_ref(path, blob) -> Optional[str]`, with `raw_url` built on top of it and tts_save
  calling `pinned_ref` directly. Same values, no string surgery.

### 2.3 `split_long_message` callers (home: `utils.queue_long_message(interaction, text, **kw)`)

13 call sites in 9 modules. 7 are the same two-line loop `for c in split_long_message(x): queue_message(i, c[, kw])`:
help.py:137, analytics.py:676 and :903 (kwargs), psd_analyzer.py:1918, type_emojis.py:327 and :382, tts_save.py:840
(lazy import; keep the lazy import). Keep explicit loops where each chunk gets a prefix or wrapper: query_card.py:598,
music_player.py:166 and :188, psd_analyzer.py:1882 (goes through `_notify`). management.py (4 sites) sends through
`interaction.followup` directly; leave it.

### 2.4 Notify-or-print (home: `utils.say(interaction, text)`)

psd_analyzer.py:1467 (inline conditional), :1722 `_send_progress` (edit), :1731 `_send_summary`, :1849
`update_stats._notify`, and tts_save.py:834 `_say` (which splits too). These are the same idea written 5 ways.

### 2.5 Copy-pasted metadata formatter (psd_analyzer.py)

`manual_metadata_entry` (about :2062-2081) and `view_old_metadata` (about :2145-2163) repeat the same block: aliases
→ COLUMN_ORDER loop → remaining keys → join. Extract `_format_metadata(card, name) -> str`.

### 2.6 Card name / key normalizers

- Path → card name, `splitext(basename(p))[0].replace("_", " ")` or `basename(p).replace(...)`: psd_analyzer.py:701,
  1576, 1641, 1788, 1992, 2112; query_card.py:467; tts_save.py:113; plus :1574 on a regex group. Use one
  `card_name(path)`.
- `tts_save._card_key` (:116) and `CardAuthors.key` (psd_analyzer.py:259) behave the same. Tested on all 1,114,112
  code points embedded in a name, and on the 1,907 PNG/PSD names in the card repo: 0 differences. The only extra is
  that `_card_key` accepts None. After the split, tts_save can import `CardAuthors` from `psd_edits` (light: no
  psd_tools or discord) or keep its copy with a one-line "same as" comment. tts_save must stay standalone.
- **Not duplicates; keep them:** `IconVocabulary.spellings` (the font's matching), `type_emojis.emoji_name` (Discord's
  naming rules), `RepositoryTraverser._star_name` (the font's star names), `tts_save._norm` (mirrors the website's
  `norm`).
- Three definitions of "the Types/ icons" exist. psd_analyzer.py:1495 is `startswith("Types") and "Stars" not in path
  and '.' in path`; :1503 walks local folders ending in "Types"; type_emojis.py:86 requires `folder == "Types"` and
  `.png`. On today's tree all three give the same 35 names (checked). Unify them only behind the parse gate,
  because the list feeds the parser.

---

## 3. Naming and type hints

### 3.1 In compiled modules, type hints change behaviour (measured; Cython 3.3.0 with setup.py's directives)

| Annotation | .py | compiled |
|------------|-----|----------|
| `x: list[str]` / `x: list`, called with a tuple | works | **TypeError** (expected list) |
| `x: dict`, called with `defaultdict` (old_stats is one!) | works | **TypeError** |
| `x: Optional[dict] = None`, called with `defaultdict` | works | **TypeError** |
| `x: tuple[int, int]`, called with a list | works | **TypeError** |
| `x: str`/`int`, called with None (no None default) | returns None | **TypeError** |
| `x: str = None` / `x: int = None` / `x: bool = None` | — | None allowed (a None default disables the check) |
| `x: float`, called with 1 | 1 | **1.0** (C double); None → TypeError |
| `x: bool`, called with None or 1 | None / 1 | **False / True** (C bint) |
| `x: int`, called with True or `np.int64(3)` | True / np.int64 | **1 / 3** (coerced) |
| `-> dict` returning a defaultdict; `-> tuple` returning a list | works | **TypeError** at return |
| `-> tuple[float, str]` with a literal list return | works | **compile error** |
| `x: Mapping`/`Sequence`/unions (`dict \| list`), dataclass field annotations, `-> int` returning None | untouched | untouched |

Consequence for any "type-hint cleanup" in the 7 INCLUDE_FILES (and in the split modules): only annotations with ABCs,
non-builtin classes, unions, `Optional[...]` with a None default, or return types of non-container builtins are inert.
**Adding or changing a builtin-container, `bool`, `float`, `int` or `str` annotation is a behaviour change.** Gate it
with a compiled run. The current code passes (parse gate compiled = .py: 0 diffs). The only annotated `dict`
returns/params near `old_stats` are dict comprehensions, which are safe.

### 3.2 Inconsistencies worth fixing (no behaviour change)

- `utils.perms_check` returns **True when the user is not an admin** (utils.py:249-259). Every caller reads
  `if perms_check(i): deny`. 13 copies in main.py, 1 in analytics.py:931; 24 references including the docs. Rename it
  to `lacks_admin` (or flip it to `is_admin` and invert the callers). The 13 identical
  `if ...: send_message("You do not have permission."); return` blocks could become one helper, but **not** a
  decorator: discord.py introspects the callback signature, so that is riskier.
- `traverse_remote` returns `num_new + num_moved`, and `num_new` also counts changed same-path cards
  (psd_analyzer.py:1662-1669). Rename it to `num_changed` (internal only).
- `StatsDatabase.dirty_files` holds card **names** seen in the tree. Rename it to `seen_cards` (archive branch: grep it
  first).
- `PSDParser.__init__(self, all_types, stats_db, icons)` (:684) shadows the module singleton. Rename the parameter to
  `db`. It only reads `.stats` for author/series carry-over.
- Repo identity is named two ways: `DEFAULT_REPOSITORY` (query_card.py:24, runtime-settable by /set_repo) and
  `REPOSITORY` (repo_history.py:16, fixed). §5 merges them into one constant; the runtime override stays on `card_repo`.
- The `EDITS_PATH` constant exists twice (psd_analyzer.py:59, tts_save.py:50) with different comments.
- Hints that are safe to fix (inert per §3.1): `get_card_aliases(card_name: str = None)` → `Optional[str] = None`
  (:1768); `_get_remote_timestamp -> tuple[float, str]` → `tuple[float, Optional[str]]` (:1697; it returns None
  authors); untyped `response` parameters go away with §2.1's tree-dict refactor; `X | None` (analytics ×2,
  management ×2, music_player ×2, type_emojis ×1, query_card.py:417) → `Optional[X]`, the house majority (psd_analyzer
  51, main 41).
- Coverage, for the record: analytics.py has 54 of 67 arguments unannotated. **Do not bulk-annotate it**; it is
  compiled (§3.1). main.py's 57 handlers have no return hints. That is harmless (main is never compiled) and low
  value.

---

## 4. psd_analyzer.py split (2,318 lines)

### 4.1 Boundaries (current line ranges; sizes approximate)

| New module | Takes | ~Lines | Imports |
|------------|-------|--------|---------|
| `commands/psd_models.py` | `CardType` (68-81), `CardStats` (215-243), `CardInfo` (445-532), `StatsDatabase` (534-592, **without** the token header/print), `get_stats_as_dict` (1802), and the `stats_db = StatsDatabase()` **singleton** | 230 | dataclasses, pickle, collections, global_config (STATS_PKL, OLD_STATS_PKL) |
| `commands/psd_edits.py` | `CardAuthors` (244-286), `CardEdits` (287-386) | 145 | psd_models, re, json, copy |
| `commands/psd_parser.py` | parser constants (TYPE_REGION_RATIO, TITLE_REGION_RATIO, ICON_FONT_PREFIX, ICON_VARIANTS, ICON_SCAN_PATTERN, EFFECT_LAYER_TAG, EFFECT_COVERAGE_MIN, STAR_CORE_ALPHA, MISSPELT_CARD_TYPES, DRAWN_LAYER_KINDS), `MutableValue`, `BarRow`, `StatTracker(s)` (83-200), `BoundingBox` (203-213), `IconVocabulary` (388-443), `CardClassifier` (594-680), `PSDParser` (682-1253), `CardValidator` + exclusion lists (1255-1352), and the `getLogger("psd_tools")` silencing | 990 | psd_models, psd_tools, bisect, re. **No discord, network, pandas or query_card** |
| `commands/psd_traverser.py` | `RepositoryTraverser` (1354-1752), `_tree_etags`, `_file_cache`, UPDATE_RATE, DOWNLOAD_WORKERS, `_forget_tree`/`_refresh_links` (1757-1765), `update_stats` (1829-1899) | 510 | psd_models, psd_parser, psd_edits, github_api, query_card.card_repo, repo_history, tts_save, type_emojis, utils (queue_*) |
| `commands/psd_analyzer.py` (facade) | The slash workers (`list_orphans`, `mass_replace_field`, `manual_metadata_entry`, `view_old_metadata`, `export_stats_to_file`, `export_rulebook_to_file`), `get_card_aliases`, COLUMN_ORDER and EXPORTED_*, `init_psd`, plus re-exports | 470 | all of the above, pandas, discord.File |

Dependency direction, with no cycles:
`psd_models ← psd_edits`, `psd_models ← psd_parser`, `{models, parser, edits, github_api, query_card, ...} ← psd_traverser ← psd_analyzer`.
Existing cycle break: query_card keeps its **lazy** imports and keeps importing **from the facade**
(`commands.psd_analyzer`). Retargeting them to psd_models/psd_edits would remove the cycle, but do that only after
every host has rebuilt (stale .so, §4.4).

### 4.2 What the facade must keep importable

- `__all__` unchanged (11 names), minus `get_old_stats_as_dict` if §1 lands.
- Pickled classes: **`CardInfo`, `CardStats`**. These are the only `commands.*` globals in stats.pkl and
  old_stats.pkl (pickletools scan; aliases.pkl and reminder.pkl hold builtins only). `CardType` is stored as a str.
- main.py: the 8 names it imports today, all from `__all__`.
- query_card: `EDITS_PATH`, `CardEdits`, `get_stats_as_dict`, `stats_db`.
- Diagnostics: parse_one (`CardValidator, PSDParser, RepositoryTraverser, StatsDatabase, stats_db`), gap_trace
  (`TYPE_REGION_RATIO, MutableValue, PSDParser, RepositoryTraverser, StatsDatabase, StatTrackers, CardClassifier,
  CardInfo, CardType`), inspect_pickle (`stats_db`), and parse_all.py (`PSDParser, RepositoryTraverser, StatsDatabase,
  CardValidator`, plus `RepositoryTraverser._populate_types_from_local/_icons_from_local`). Keep both methods on the
  class.
- archive/local-update-stats: `EXCLUDE_FOLDERS, GIT_TOKEN, UPDATE_RATE, CardValidator, PSDParser, RepositoryTraverser,
  stats_db`.
- Website (TTSCardMaker site/build/build_site.py:122-126): its stub `find_class` accepts **any module starting with
  `commands`**, so it is indifferent to the path. Renaming CardInfo *attributes* still blanks website fields.
- Rule: re-export by binding names (`from commands.psd_models import CardInfo, CardStats, stats_db, ...`). Never
  rebind module state in the facade (`_tree_etags = {}` there would fork it).

### 4.3 Pin the pickle class path (required)

Without a pin, the next save writes `commands.psd_models.CardInfo`. A checkout older than the split (the other
machine before it pulls, `git show` time-travel, the archive branches, diff_stats on old commits) then cannot unpickle
it. Pin it in psd_models.py right after the classes:

```python
# pickles store this path (stats.pkl, old_stats.pkl, older checkouts)
CardInfo.__module__ = CardStats.__module__ = "commands.psd_analyzer"
```

Measured in /tmp/codehealth/cyexp with the real CardType/CardStats/CardInfo code: load copies of stats.pkl
(349,987 B) and old_stats.pkl (107,089 B), then re-dump. The result is **byte-identical to the current module's
re-dump** in all 4 combinations: both .py, facade .so, models .so, both .so. `CardInfo is` identity holds through the
facade. Unpinned, the dump contains `commands.psd_models`. This also means the PICKLE diff after the split should be
empty, a strong extra gate.

### 4.4 setup.py and Cython risks

- **INCLUDE_FILES**: add `commands/psd_models.py`, `commands/psd_edits.py`, `commands/psd_parser.py` and
  `commands/psd_traverser.py`, and keep `commands/psd_analyzer.py`, for 11 in total. The reason is semantics, not
  speed (compiled parse = .py parse = 10 s). Moved code then keeps today's compiled annotation behaviour (§3.1).
  Moving it into uncompiled modules would silently drop the bool/float coercions and exact-type checks on code paths
  parse_all does not cover (CardEdits.apply, the traverser). `github_api.py` stays uncompiled like
  repo_history/type_emojis/tts_save.
- **Stale .so**: an old monolithic `psd_analyzer.*.so` shadows the new facade .py. It is self-contained, so the bot
  just runs old code (today's "my edits don't take effect"), **as long as nothing imports the new modules directly**.
  If query_card imported `psd_models.stats_db` while the old .so was loaded, there would be two `stats_db`
  singletons: /query would read one and /update_stats write the other. Hence §4.1's rule: external imports go
  through the facade until all hosts have rebuilt. Deploy is already pull → `python setup.py build_ext --inplace`
  (which runs `clean_artifacts` first) → restart. Check with
  `python -c "import commands.psd_analyzer as m, commands.psd_models as d; print(m.__file__, d.__file__, m.CardInfo is d.CardInfo)"`.
- **Directives**: the moved code is verbatim code that already compiles under `wraparound=False` and
  `boundscheck=False`. Keep `x[len(x)-1]`. Do not re-annotate a declared variable (the Cython 3.3 rule). `cdivision`:
  the module's only `//` and `%` (:1205, :1684) have non-negative operands. `boundscheck=False` is why §1.4
  segfaults, so fix classify in its own commit before or after the split, never inside it.
- **Circular imports** behave the same compiled (partial module init). The direction in §4.1 has none at module
  level. The traverser→query_card→(lazy) facade chain is the same shape as today.
- **Mixed builds** (facade .so, a new module .py) work (measured) but differ in annotation semantics. Gate both.

### 4.5 Gate for the split (all measured methods)

1. `/tmp/psdreg/parse_all.py` on the old snapshot and on the split, then `compare.py`: **0 differences over 813**, with
   .py modules.
2. The same with the split built by the real setup.py in a /tmp copy (method: /tmp/codehealth/cmp). Baseline today:
   compiled vs .py, 0 differences.
3. Pickle byte-identity: load copies of stats.pkl and old_stats.pkl, re-dump with HEAD and with the split, and compare
   bytes (method: /tmp/codehealth/cyexp/check.py).
4. Import smoke: main.py's import list, query_card's lazy imports, parse_one/gap_trace/inspect_pickle on one card, and
   diff_stats.py on two pickle copies.
5. Docs: 159 `psd_analyzer.py:NNN` references in 15 skills plus BUGS/NEXT_SESSION go stale. Also update gap_trace.py's
   MAINTENANCE NOTE, the architecture module map, change-control's CardInfo line, and the docs-and-style skeleton
   exemplar line refs.

---

## 5. Repo-path constants: every literal (home: global_config.py)

global_config.py is already imported by every relevant module, including standalone tts_save. It is uncompiled and
imports only zoneinfo. Proposed names are in **bold**.

| Fact | Literal sites |
|------|---------------|
| **CARD_REPO** `"MichaelJSr/TTSCardMaker"` | query_card.py:24 `DEFAULT_REPOSITORY`; repo_history.py:16 `REPOSITORY` (→ tts_save.py:39, 45, 67, 73) |
| **CARD_BRANCH** `"main"` | psd_analyzer.py:1464, query_card.py:74, type_emojis.py:101 (in the tree URLs); repo_history.py:122, 129, 155; tts_save.py:46-47 (`MAIN_REFS`), 224-225, 245-249, 280, 412, 611, 821, 859, 904, 947-953. (update_bot's `'main'` is the bot's own repo: leave it) |
| **API_BASE** `https://api.github.com/repos/` | psd_analyzer.py:1414, 1464; query_card.py:74, 344; type_emojis.py:101, 111; tts_save.py:67 |
| **RAW_BASE** `https://raw.githubusercontent.com/` | repo_history.py:156; tts_save.py:45 |
| Clone/push URLs | repo_history.py:126 `https://github.com/{REPOSITORY}.git`; tts_save.py:73 `git@github.com:{REPOSITORY}.git` |
| Raw-blob media type `application/vnd.github.raw` | psd_analyzer.py:1415; query_card.py:345; type_emojis.py:112; tts_save.py:724 (→ github_api) |
| **TYPES_DIR / STARS_DIR** `Types`, `Types/Stars` | psd_analyzer.py:1495, 1501, 1503, 1516, 1615; type_emojis.py:24 (`ICON_FOLDERS`), 86 |
| MDW files | psd_analyzer.py:57 `ICON_MANIFEST_PATH`, 58 `AUTHORS_PATH`, 59 `EDITS_PATH`; tts_save.py:50 `EDITS_PATH` (duplicate) |
| Excluded folders `Markers`, `MDW` | psd_analyzer.py:41 `EXCLUDE_FOLDERS` (used at 1571, 1640) |
| TTS files | global_config.py:13 `TTS_SAVE_PATH` (already there); tts_save.py:49 `RULES_PATH` `TTS/decks.json` |
| Site | global_config.py:7 `SITE_URL`; query_card.py:363 `/data/cards.json`, 379 `/c/{id}/`, 602 `/#/rules?q=`; tts_save.py:51 `SITE_CARDS` `site/dist/data/cards.json` (local build) |
| **RULEBOOK** marker `"Rulebook"` (a substring test) | psd_analyzer.py:747, 1571, 2188, 2234, 2281; query_card.py:395 |
| TTS card folders | tts_save.py:53 `HAND_NAMED` (`Auxiliary/Rulebook/`, `Auxiliary/Markers/`), 54 `LARGE_CARDS` (`Auxiliary/Creatures/`) |
| Classification vocabulary | psd_analyzer.py:621-632 (`MDW`, `Field`, `Items`, `Creatures`, `N.M.E`), 638 `Auxiliary`, 663 `Minions`, 668 `Items`, 671 `Debuffs`, 676 `Buffs`, 750 `["Auxiliary/Items", "Auxiliary/Minions", "N.M.E"]`. **Keep these in the parser**: they are classification rules and CardClassifier is their home |
| Local paths | global_config.py:5 `LOCAL_DIR_LOC`, :10 `HISTORY_CLONE` (already there) |

Keep the old names as aliases where code outside the module imports them. `psd_analyzer.EDITS_PATH` is used by
query_card and `repo_history.REPOSITORY` by tts_save. All values stay identical, so behaviour does not change. The
/set_repo override stays on `card_repo.repository`. The history clone and the TTS save keep working only for
CARD_REPO, as today (`repository == REPOSITORY` checks in repo_history.py:154 and tts_save.py:982).
