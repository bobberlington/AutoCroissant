---
name: autocroissant-build-and-env
description: Load this skill whenever the task involves installing, setting up, or rebuilding the AutoCroissant environment or its Cython build — trigger words include install, setup, environment, venv, requirements, pip, dependencies, torch install, CUDA, MPS, new machine, migrate machine, ffmpeg, opus, voice not working after install, cython, compile, build, build_ext, setup.py, stale .so, "my edits don't take effect", ImportError after install, Pillow/PyNaCl missing, "redeclared" / CompileError from Cython 3.3, discord.py>=2.5 / application emojis / AttributeError fetch_application_emojis, "python3 has no psd_tools", which Python to use, jetson deploy. Contains verified prerequisites (Python version and the 2026-10-19 state of the Mac's interpreters, ffmpeg/opus, config.py minimum), main's three layered requirements files (requirements-cards.txt holds the one psd-tools fork pin) with what every package is for (the AI stack and requirements2.txt live only on branch archive/ai-generation), the Pillow/PyNaCl history (transitive until 2026-10-20, declared since), a Mac vs production jetson table (incl. the psd-tools fork venvs), the full Cython build/clean procedure with compiler-directive consequences (all 10 modules compile with Cython 3.3.0; bounds checks and negative indexes on since 2026-10-23 (the x[len(x)-1] idiom stays), C division, Cython 3.3's no-re-annotation rule; type_emojis.py and 30 other modules stay uncompiled), directory conventions (music/, .ssh/), and copy-pasteable from-scratch checklists for dev-only and full-host environments plus an env smoke test.
---

# AutoCroissant: Build and Environment

Recreating the bot's environment from scratch (the owner's Mac or the production jetson),
the Cython build, and the known traps. All facts verified against the repo on 2026-07-11
unless marked otherwise; sections 1, 2, 3 and 5 re-verified 2026-10-19; AI removal, module
counts, cookies and line numbers re-verified 2026-10-20 (main 9fb3093); requirements, PIL imports, the fork venvs,
directives, module counts and the suite size re-verified 2026-10-23 (main 2ada357). All commands assume
`cwd = /Users/michaelsrouji/Desktop/AutoCroissant` (repo root) unless stated.

Quick orientation:

- Three layered requirements files on main (2026-10-23): `requirements-cards.txt` (the card stack, the psd-tools
  fork's one pin), `requirements.txt` (`-r` it, then the bot), `requirements-dev.txt` (`-r requirements.txt`, then the
  tests). AI generation and its `requirements2.txt` live only on branch `archive/ai-generation` (section 2).
- Cython compilation is optional on dev: the Mac's working tree has no `.so` files and runs
  plain `python3 main.py`. Production (the jetson) compiles the 10 modules on every deploy (section 5).
- `config.py` is secret, gitignored, and must be created by hand on every machine.

## 1. Prerequisites

| Prereq | Detail | Verified how (2026-07-11) |
|---|---|---|
| Python 3.10+ | On 2026-07-11, 3.10.20 at `/usr/local/bin/python3` was the verified-working interpreter on the Mac. **On 2026-10-19 that is gone:** `/usr/local/bin/python3` is a symlink to Homebrew `python3.14` (3.14.8) with discord.py 2.7.1, Pillow 12.3.0 and Cython 3.3.0 but **no psd-tools** (and no setuptools), so it cannot import the bot. A durable venv with every dependency now exists: `~/.venvs/ac-py314-psd125` (3.14.8, psd-tools 1.25.0, discord.py 2.7.1, Cython 3.3.0, pytest 9.1.1; `requirements-dev.txt` adds pytest). The old scratch venv in /private/tmp is gone on reboot. A second Mac venv, `~/.venvs/ac-py314-fork`, has the psd-tools fork (tag `cards-20261023` = ecc4b773, with `range_reader`) for the fork-only tests, card_guards and bakes. Production (the jetson) runs Python 3.14.8 from `~/venvs/autocroissant` since 2026-10-20 (its 3.10 was removed the same day); its psd-tools is the fork since 2026-10-22, tag `cards-20261023` since 2026-10-23 10:36 (section 4). Keep code 3.10-compatible. | `python3 --version; python3 -c "import psd_tools"` |
| git | Needed to clone; also the bot's self-update commands drive git via GitPython, imported lazily inside update_bot's functions (`_git`/`_repo`, update_bot.py:19, 27; since 778c2ce, 2026-10-21: startup never loads GitPython). | `git --version` |
| C compiler | For the Cython build (section 5), and on the jetson for installing the psd-tools fork, whose `_distance` extension compiles at install (gcc). Xcode CLT on Mac, gcc on Linux. | `cc --version` |
| ffmpeg on PATH | Required for music playback: the player streams through `FFmpegOpusAudio` (`commands/music_player.py:4`), which spawns the `ffmpeg` binary. The `POSTPROCESS` config flag adds an ffmpeg mp3 extraction step too (music_player.py:62-68). Mac has 8.1.1 via Homebrew (`/opt/homebrew/bin/ffmpeg`). | `which ffmpeg` |
| PyNaCl | Required for Discord voice encryption. Declared in requirements.txt since 2026-10-20 (before that it arrived only through PyGithub: section 3). | `python3 -m pip show PyNaCl` |
| opus library | Standard discord.py voice requirement. This bot's `FFmpegOpusAudio` path delegates opus encoding to ffmpeg, so libopus is only strictly needed if anything switches to PCM audio (the volume-control path suggests exactly that, music_player.py:327). Cheap insurance: `brew install opus` / `apt install libopus0`. Labeled guidance. | `brew list opus` |
| config.py | Created BY HAND in the repo root; gitignored; contains tokens — never commit, never print. Only `TOKEN` is required to boot: `main.py:13` is `from config import TOKEN` and every other field is read with `getattr(config, ..., default)` (e.g. utils.py:19, music_player.py:37). A TOKEN-only config boots, but `ADMINS` then defaults to `[]` so admin-gated commands deny everyone. Field names and semantics: see **autocroissant-config-and-flags**. | `grep -n "from config import" main.py` |
| TTSCardMaker clone (full host only) | The card-source repo is expected at `~/Desktop/TTSCardMaker` (`global_config.py:5`, `LOCAL_DIR_LOC`). The bot works without it: `/update_stats` reads GitHub only since local mode was archived on 2026-10-20 (to branch `archive/local-update-stats`; autocroissant-run-and-operate §7). The clone serves the diagnostics scripts (parse_one.py, gap_trace.py), sandbox sweeps, and the stoogedings icon font build (ttscardmaker-companion). | `ls ~/Desktop/TTSCardMaker` |

## 2. Requirements

Three layered files since 2026-10-23 (owner: "three layered files, one pin"):

| File | Holds | Installed by |
|---|---|---|
| `requirements-cards.txt` | The card code's packages: `psd-tools[composite,text] @ git+https://github.com/MichaelJSr/psd-tools@<tag>` (the fork: THE one pin; the tag is only in the file), fonttools, requests, and `pyoxipng==9.1.1` behind a marker (2026-10-24): Linux x86_64 at any CPython 3.9+ (the Actions, on 3.14, build it from the sdist with the runners' Rust), elsewhere only where 9.1.1 has a wheel (CPython 3.9-3.13 on macOS x86_64/arm64, Linux aarch64 with glibc 2.28+ or musl, Windows x86/x64), so it is skipped on the Mac's and the jetson's 3.14 (the bot never recompresses) | `requirements.txt`; the six card Actions on Python 3.14 (card-edits, card-guards' two jobs, card-sync's one venv, card-preview without pyoxipng: its PNG is never published, card-check, font-check), offline from one wheel cache entry keyed on the file (the re-pin procedure: autocroissant-config-and-flags) |
| `requirements.txt` | `-r requirements-cards.txt`, then the bot's own (the table below) | the jetson, a dev machine |
| `requirements-dev.txt` | `-r requirements.txt`, then pytest and pyyaml (the workflow-shape tests' yaml half skips without it); `pyobjc-framework-Vision` went with `commands/discord_match.py` on 2026-10-25 | a dev machine |

uharfbuzz and freetype-py come with psd-tools' `[text]` extra, scipy and aggdraw with `[composite]`. `.gitignore`
ignores `*.txt` and re-includes `requirements-dev.txt` and `requirements-cards.txt`; `requirements.txt` is tracked from
before that rule. Proven 2026-10-23: `pip install --dry-run -r requirements-dev.txt` in a fresh 3.14 venv on the Mac
resolves (53 packages, all wheels but the fork; pyoxipng skipped by its marker), and `uv pip compile --only-binary :all:
--no-binary psd-tools` resolves requirements-cards.txt for Linux x86_64 3.12 (the Actions then; pyoxipng in), Linux x86_64
3.14 (the Actions since 2026-10-24; pyoxipng built from its sdist) and Linux aarch64 3.14 (the jetson; pyoxipng out), and
requirements.txt for the jetson. Cold install on the Mac (2026-10-24, fresh wheel dir, no pip cache): `pip wheel -r
requirements-cards.txt` 16.4 s on 3.14.8 (19.3 s on 3.12.15), pyoxipng's sdist 19.5 s with every crate downloaded, the
offline install 10.7 s (docs/Campaigns/PUSH_GUARDS.md section 20).
`tests/test_card_data.py::test_requirements_layers_and_pin` holds the layering, the pin and the marker.

AI generation left main on 2026-10-20 (05c643e): its `requirements2.txt` (torch trio, diffusers etc., the CUDA nightly
index line) and the clean-venv proof that core runs without it live only on branch `archive/ai-generation`, with the
doctrine in **autocroissant-ai-boundary** there. The `discord.py>=2.5` floor is committed (53db8ee).

### requirements.txt — core bot (always install)

`pip install -r requirements.txt` — the card stack (marked "cards") plus 12 packages (2026-10-23):

| Package | Why it is in core |
|---|---|
| Cython | Build tooling for the optional `setup.py` compile (section 5). Harmless if you never compile. |
| setuptools | `setup.py` imports it first (on 3.12+ `distutils` is setuptools' copy); the compile only. |
| davey | Optional Discord "DAVE" voice end-to-end-encryption lib (0.1.6 as of 2026-07-11). discord.py imports it inside `try/except ImportError` (verified in installed discord 2.7.1, `gateway.py`), so the bot runs without it — the Mac's current env does not even have it installed. |
| discord.py (`>=2.5` since 2026-10-19) | The Discord API library (2.7.1 on the Mac). Its only hard dependency is aiohttp. The floor exists because `commands/type_emojis.py` uses `Client.fetch_application_emojis` / `create_application_emoji`, both `versionadded:: 2.5`; on an older install the emoji sync fails with "Emoji sync failed: ... has no attribute ..." while everything else works. Upgrade with `pip install -U "discord.py>=2.5"` |
| fonttools (cards) | Font files: `commands/font_check.py:43` (`/fonts`, the missing-fonts audit), reached by `psd_details` and the bake; imported lazily (main.py imports font_check inside `/fonts`, card_data imports psd_details inside its functions), so startup never loads it (2026-10-21). |
| GitPython | The bot's self-update commands (`/pull`, `/update`; `/push` until 2026-10-25), imported lazily inside update_bot's functions (update_bot.py:19, 27; startup never loads GitPython). |
| numpy | Image array plumbing in `commands/utils.py:7` and `commands/frankenstein.py:2`. |
| opencv-python | Image encode/decode/resize, imported lazily inside the functions that need it (2026-10-20; only /frankenstein): `commands/utils.py:109, 117` (`url_to_cv2image`, `cv2discordfile`), `commands/frankenstein.py:65`. |
| pandas | The card query engine's DataFrames: `commands/query_card.py:13`, `commands/psd_analyzer.py:13`. |
| Pillow | Images: top level in `commands/type_emojis.py:12` (emoji images, loaded at startup), `png_export.py:32`, `card_bake.py:47`, `psd_edit.py:23`; lazily in `psd_details.py:130` (ImageCms). Declared since 2026-10-20 (section 3). |
| psd-tools (cards) | The PSD parser (`commands/psd_parser.py:11`), the fork at the one pin: its text engine serves the bake, the push guards and range reads (section 4). Its version does not follow the tag (1.25.0 up to `cards-20261023d`, 1.26.0+cards since `cards-20261024`, also `cards-20261025` and `cards-20261025b`), and pip calls an install of the same version satisfied, so a re-pin needs `--force-reinstall` (section 4). |
| pyoxipng (cards) | png_export's lossless PNG recompression, imported lazily (`png_export._oxipng`); everything runs without it. On the Actions' Linux x86_64 (built from the sdist on 3.14 since 2026-10-24), elsewhere only where it has a wheel (above): not on the Mac's or the jetson's 3.14. So a fresh Mac venv (3.14) has none: `commands/png_recompress.py`, the by-hand batch, cannot run, and the oxipng tests skip (test_png_recompress.py, test_png_export.py's `@oxipng`, test_card_guards' recompression test; `pytest -rs` lists them). For those, `pip install pyoxipng==9.1.1` builds it with cargo (`~/.venvs/ac-py314-fork` has such a build, 2026-10-23). |
| PyGithub | Removed 2026-10-25 (step 10): its only importers, card_data's legacy `_committed_since_update` and `_repo`, went with the pickles. PyNaCl stays declared on its own. |
| PyNaCl | discord.py voice encryption (music is silent without it). Declared since 2026-10-20 (section 3). |
| requests (cards) | HTTP fetches/downloads: `commands/utils.py:10`, `commands/query_card.py:9`, `commands/github_api.py:7`, `commands/tts_save.py:35`, `commands/git_commit.py:19`. |
| yt-dlp | Music downloading (`commands/music_player.py:73`). Deliberately unpinned — updating it (`pip install -U yt-dlp`) is the first move when downloads break (autocroissant-debugging-playbook §6). |

## 3. Transitive-dependency history (Pillow, PyNaCl)

Until 2026-10-20 two packages the bot imports were in no requirements file: they arrived as
dependencies of other packages (pip metadata, 2026-07-11). Both are declared since 2026-10-20:

| Package | Used to arrive via | Who needs it |
|---|---|---|
| Pillow (PIL) | psd-tools (`pip show psd-tools` → Requires: attrs, numpy, **Pillow**, typing-extensions) | Top-level `from PIL import ...` in `commands/type_emojis.py:12` (emoji images; main.py and query_card import that module at load time, so the bot fails at import without Pillow), `png_export.py:32`, `card_bake.py:47`, `psd_edit.py:23`; lazily in `psd_details.py:130` |
| PyNaCl | PyGithub (`pip show PyGithub` → Requires: pyjwt, **pynacl**, requests, typing-extensions, urllib3) | discord.py voice encryption (voice/music is silent-broken without it); discord.py itself only requires aiohttp (and audioop-lts on 3.13+) |

**The rule**: keep Pillow and PyNaCl declared even if psd-tools goes (PyGithub went 2026-10-25). pip
would not warn when the accident of the dependency graph stopped covering them.

## 4. Mac vs production (the jetson)

Same code; the card data comes from TTSCardMaker's `data` branch on both (each syncs its own `card_data/`; the pickles
synced by `/push` until 2026-10-25 — see **autocroissant-run-and-operate**). What differs:

| Axis | Mac (dev) | Jetson (production) |
|---|---|---|
| Python / run | venv (section 1), plain `python3 main.py` | Python 3.14 venv `~/venvs/autocroissant`, systemd `autocroissant.service` (ExecStart = the venv's python), compiled `.so` (section 5) |
| yt-dlp cookies | `browser_cookies = True` in config.py → `cookiesfrombrowser = ('safari', ...)` (`music_player.py:37`, used at :55). Old configs fall back to `vram_usage == "mps"`. | Unset → no browser cookies |
| psd-tools | `~/.venvs/ac-py314-psd125`: stock 1.25.0 (older tests); `~/.venvs/ac-py314-fork`: the fork (`cards-20261025` = 9edbc1fa, tab stops; move it to the pin `cards-20261025b` with the command at the right; with pyoxipng, pytest-xdist and setuptools: the full suite, card_guards, bakes; the one fork venv since 2026-10-25) | The fork at the pin (`cards-20261025b` = c17e0734, 2026-10-25: Stylistic Alternates, the one-colour badges; `cards-20261025` = 9edbc1fa before it, tab stops; `cards-20261024b` = a651cd08 before it, glow halos; check `pip freeze`: `psd-tools @ git+...@c17e0734...`). Move it to the pin with `~/venvs/autocroissant/bin/pip install --force-reinstall --no-deps "$(grep '^psd-tools' requirements-cards.txt)"` (gcc compiles `_distance`; a plain `pip install -r` keeps the old commit), then `pip check`. The Actions (card-edits, card-guards, card-sync, card-preview) install the same file |

The torch/CUDA/MPS differences went with the AI to branch `archive/ai-generation`.
Config field semantics: **autocroissant-config-and-flags**.

## 5. The Cython build (optional)

### Commands

```bash
python3 setup.py build_ext --inplace   # rebuild the stale INCLUDE_FILES to .so next to the .py
python3 setup.py clean                 # remove *.c *.cpp *.so *.pyd *.pyc *.html, __pycache__/, build/
```

Notes verified from `setup.py` (2026-07-11; the build is incremental since 2026-10-24):

- **Incremental build (2026-10-24).** `build_ext --inplace` rebuilds only the modules whose `.so` is missing or whose
  fingerprint changed: sha256 of the `.py` source, its `.pxd` if any and setup.py itself (any setup.py edit rebuilds
  everything), plus the directives, the `-O3` flag, the Cython and Python versions and the ABI suffix, kept in
  `build/ext-stamps.json`. Each fingerprint is taken before compiling and stored only for modules that built, so a `.py`
  saved mid-build rebuilds next time. A bare `touch` does not rebuild; a content change does. Stale modules build 4 at a time (cythonize `nthreads=4`, build_ext `parallel=4`). Safety
  kept: a stale `.so` is deleted BEFORE its rebuild (a failed build leaves the new `.py` running, exit code 1, never an
  old `.so` next to a new `.py`), and every `.so`/`.pyd` that is not a current INCLUDE_FILES build (other Python ABI,
  dropped module) is deleted, and since 2026-10-25 a dropped module's `.c` source in `commands/`, its `build/lib.*` and
  `build/temp.*` objects and its stamp too (`remove_orphans`). No stamps file (first run, after `clean`, after deleting `build/`) means a full rebuild.
  Mac timings 2026-10-24 (M-series, 24 cores, Cython 3.3.0, Python 3.14.8): old from-clean build 75-88 s; new full
  rebuild 16-23 s; nothing changed 0.2 s; one small module edited (help) 1.7 s; psd_parser 11-18 s. Jetson before:
  6 min 44 s every deploy. Only `setup.py clean` deletes `*.html` (rglob) now; the build no longer does.
- `setup.py` without `--inplace` refuses (the stamps describe the in-place `.so` files).
- Compiled modules (`INCLUDE_FILES`, setup.py) — 10 `commands/` modules since 2026-10-25 (repo_traverser.py was
  deleted; `build_ext` removes its `.so` as an orphan): analytics, frankenstein, help, psd_analyzer and the three modules its
  code was split into on 2026-10-20 (card_models, card_edits, psd_parser: same directives, so the moved code keeps its
  compiled semantics), query_card, management, music_player.
- **Not compiled because they are in neither list:** every module not in INCLUDE_FILES or EXCLUDE_FILES (25 of the
  38 on 2026-10-23). In order of arrival: `commands/type_emojis.py` (added 2026-10-19),
  `commands/repo_history.py`, `commands/tts_save.py` and `commands/github_api.py` (2026-10-20), `commands/psd_edit.py`, `psd_regions.py`,
  `card_bake.py`, `font_check.py` and `git_commit.py` (2026-10-21: the Actions check them out alone, so their import closure stays plain Python), the card data build's `card_data.py`, `card_history.py`, `card_db.py` and `changelog.py` (2026-10-22: `card_data` holds the traverser's code, so /update_stats on the jetson runs the same pure Python as the Action; the compiled `repo_traverser` wrapper was deleted 2026-10-25), and since 2026-10-22/23 `card_sync`, `card_dates`, `card_check`, `card_guards`, `png_export`, `card_standards`, `psd_diff`, `txt2_repair` and `psd_details`.
  New modules are uncompiled by default; all were deliberately left out. Note that compiled
  `query_card` imports it at module level, which is fine (compiled code can import plain modules).
- Never compiled (the comment above `INCLUDE_FILES`; the `EXCLUDE_FILES` set that only listed them went on 2026-10-25):
  `main.py` (entry point — keep as Python), `commands/update_bot.py` (`execl` for restart), `commands/utils.py` (core
  utilities, decorators, dynamic behavior), `config.py`/`global_config.py` (configuration).

### Compiler directives and their consequences (setup.py:36-49)

This is where the coding-discipline rules for compiled modules come from. Rules
without their incident get ignored, so each one carries its story:

| Directive | Consequence in the 10 compiled modules |
|---|---|
| `wraparound` / `boundscheck` | Since 2026-10-23 setup.py builds with `boundscheck=True` and `wraparound=True` (owner): a bad or negative index behaves as in plain Python instead of crashing; the `x[len(x)-1]` idiom stays (don't churn it). Before that (`wraparound=False`, `boundscheck=False`): **NEGATIVE INDEXING WAS BANNED.** Compiled code skips Python's negative-index handling; `x[-1]` can read garbage or crash instead of meaning "last element". House idiom: `x[len(x)-1]` — see commit dd800bc (2026-01-30), which changed `bboxes[-1][1].y` to `bboxes[len(bboxes)-1][1].y`, and the live idiom at `music_player.py:132` (`prev_music[len(prev_music)-1]`). The ban has already drawn blood once: the cythonization commit f7c915c mechanically purged a `split("TTSCardMaker")[-1]` and replaced it with a `removeprefix()` that no-ops on absolute paths — the local-mode stats-corruption bug (local mode archived 2026-10-20). Full story: **autocroissant-failure-archaeology**. |
| (history) `boundscheck=False` until 2026-10-23 | Out-of-range indexes were undefined behavior (crash or garbage), not a clean `IndexError`: a blank star folder read out of bounds in compiled code (fixed in ba27fb7, 2026-10-23). Since 9919a14 (setup.py:38-39) they raise `IndexError` with a traceback. |
| `cdivision=True` | C division semantics for negatives: quotient truncates toward zero and `%` takes the dividend's sign. In Python `-7 // 2 == -4` and `-7 % 2 == 1`; in compiled code they become `-3` and `-1`. Any div/mod on possibly-negative ints must be reasoned about. |
| `embedsignature=True` | Function signatures survive into docstrings — `help()` still works on compiled functions. |
| (Cython 3.3 itself, not a directive) | **Declare an annotated variable once.** Cython 3.3 rejects re-annotating a name already declared in the same function (`'target_message' redeclared`), which plain Python allows. `cythonize()` then raises `CompileError` before `setup()` runs, and because `clean_artifacts()` already deleted the old artifacts, the host ended up with NO compiled modules and ran plain Python without any obvious sign (since 2026-10-24 the build exits 1 and only the stale modules lose their `.so`; deploy.sh stops before the restart). This hit `commands/management.py` (`add_reactions`) until 2026-10-19: autocroissant-failure-archaeology Entry 10.5. Since the fix, all INCLUDE_FILES modules compile with Cython 3.3.0 and setup.py's directives (7 on 2026-10-19; all 11 in a scratch build after the 2026-10-20 split). |
| (Cython itself: annotations) | **Type hints are enforced and coerce** in compiled modules (measured with Cython 3.3, 2026-10-20): `x: list` rejects a tuple, `x: dict` rejects a defaultdict, `x: str`/`int` reject None unless the default is None, `x: float` turns 1 into 1.0, `x: bool` turns None into False, `-> dict` rejects a defaultdict return. Untouched: ABCs, non-builtin classes, unions, `Optional[...]` with a None default, dataclass fields, `-> int`. So a hint change in an INCLUDE_FILES module is a behaviour change: gate it with a compiled run (the 2026-10-20 split kept the moved code compiled for exactly this reason). Known case: compiled `analytics.parse_interval(None)` raises TypeError; the .py returns None. |
| Also set | `language_level='3'`, `initializedcheck=False`, `infer_types=True`, `overflowcheck=False`, `profile=False`, `linetrace=False`. |

### Compilation is optional — and the stale-.so trap

- The Mac's working tree has **no** `.so`/`.c` artifacts (2026-10-20) and runs plain Python.
- Production (the jetson, no startup.sh) builds on every deploy: the `jetson-deploy` skill (~/.claude/skills; pull, psd-tools pin,
  `~/venvs/autocroissant/bin/python setup.py build_ext --inplace`, restart; **autocroissant-run-and-operate**). It runs the 11 `.so` files, so the stale-.so trap below
  is live there: rebuild after every pull.
- **The trap**: an import like `from commands import psd_analyzer` picks a compiled
  `commands/psd_analyzer.*.so` OVER `commands/psd_analyzer.py`. A stale `.so` from an
  old build silently shadows your edited `.py`. Symptom: your edits do not take
  effect, no error anywhere. Fix: `python3 setup.py clean`. This is the first check in
  any "my change does nothing" debugging session on a machine that has ever compiled.

Check for artifacts (also useful before trusting any test result):

```bash
find . -name "*.so" -not -path "./.git/*"
```

- Artifacts are gitignored (`*.so`, `*.c`, `*.cpp`, `*.pyd`, `*.dll`, `build/`, `__pycache__/`).
  History of one quirk (verified 2026-07-11 at `.gitignore` lines 18-20, before the fix): the entries
  `*.so      # Linux/Mac`, `*.pyd     # Windows`, `*.dll     # Windows (rare)` carry
  trailing text — gitignore has no trailing comments, so those three patterns are
  literal and match nothing. `*.so` is safely covered by the bare `*.so` on line 7,
  but `*.pyd`/`*.dll` are covered by nothing: if a Windows machine ever compiles
  (setup.py has win32 branches), `.pyd` files would have shown up as untracked and could have been
  committed by a careless `git add .`. **Fixed 2026-10-19** (53db8ee): the comments moved to
  their own lines, so `*.so`, `*.pyd` and `*.dll` (lines 19-22) all match now
  (`git check-ignore -v a.pyd a.dll`).

## 6. Directory conventions

| Path | Convention |
|---|---|
| `models/` | AI checkpoints; branch `archive/ai-generation` only. Unused on main. |
| `music/` | yt-dlp downloads land here (`MUSIC_BASE_DIR = "music/"`, music_player.py:29). Ignored wholesale by its own tracked `music/.gitignore` (`*` except itself). |
| `.ssh/` | The bot's git deploy key pair lives here on deploy machines (untracked, gitignored). **NEVER commit it, never print its contents.** |
| repo root exports | `/export_cards` and `/export_rulebook` write `stats.csv` / `stats.txt` / `rules.txt` into the repo root; covered by the `*.csv` / `*.txt` gitignore rules. |
| venvs | `venv/`, `.venv/`, `env/`, `ENV/` are gitignored. (History: the venv was committed on day one and then deleted in favor of requirements.txt — ed18602. Do not repeat 2023.) |

## 7. From-scratch checklists

### (a) Dev-only environment

Run from the directory that should contain the clone:

```bash
git clone https://github.com/bobberlington/AutoCroissant.git
cd AutoCroissant
python3 -m venv .venv               # 3.14 is what both machines run (2026-10-20); code stays 3.10-compatible
source .venv/bin/activate
pip install -r requirements-dev.txt   # requirements.txt (and requirements-cards.txt) plus pytest, pyyaml
```

Then:

1. Create `config.py` in the repo root by hand. Minimum content is one line defining
   `TOKEN` (the Discord bot token). Field names and what each does:
   **autocroissant-config-and-flags**. Never commit this file.
2. Smoke test (section c). Music playback also needs ffmpeg (section 1).
3. Offline suite: `~/.venvs/ac-py314-fork/bin/python -m pytest -q -n auto tests` (pytest-xdist from requirements-dev.txt; 553 tests on 2026-10-25, about a
   minute; needs no config.py, network or Discord; PSD tests skip without `~/Desktop/TTSCardMaker`; the bake tests
   need the fork venv).
   Owner: **autocroissant-validation-and-qa** §1 row 1c.

### (b) Full host environment (a machine that will actually host the bot)

1. Steps 1-5 of checklist (a).
2. `config.py`: tokens/fields per **autocroissant-config-and-flags**; `browser_cookies = True`
   on the Mac for Safari cookies (section 4).
3. ffmpeg: `brew install ffmpeg` (Mac) / `apt install ffmpeg` (Linux). Optional
   insurance: opus (section 1). Verify: `which ffmpeg`.
4. Clone the card source where the bot expects it:
   `git clone https://github.com/MichaelJSr/TTSCardMaker ~/Desktop/TTSCardMaker`
   (path is `LOCAL_DIR_LOC`, global_config.py:5). The bot works without it: `/update_stats`
   reads GitHub only since local mode was archived on 2026-10-20
   (**autocroissant-run-and-operate** §7). The repo's history is about 25 GB of PSDs; the dev Mac's clone
   is blobless (`--filter=blob:none`) and shallow (depth 1) as of 2026-10-19, so local `git log`
   there has no history (ttscardmaker-companion). Local checkouts also give every PSD a fresh
   mtime, which is why local-timestamp updates misfired and local mode was archived
   (autocroissant-run-and-operate §7).
5. Deploy key: place the bot's SSH key pair in `.ssh/` if this machine will push by hand (the bot itself pushes
   nothing since 2026-10-25; see **autocroissant-run-and-operate** for the self-update flow).
6. Cython build: `python3 setup.py build_ext --inplace` (section 5). Optional, but production
   (the jetson) compiles with `~/venvs/autocroissant/bin/python` and runs under systemd `autocroissant.service`.
7. Smoke test (section c).

### (c) Verifying an environment

```bash
python3 -c "import main" && echo OK
```

- Run from the repo root (imports are repo-root relative).
- REQUIRES `config.py` to exist — without it this fails immediately, which is itself
  a diagnosis.
- Expected noise: one git-token-presence line,
  `Git token found, API limited to 5000 requests/hour.` (or the
  `No git token in config` variant), from `print_token_status()` (main.py; defined in github_api.py).
  That is normal import-time output, not an error.
- It does NOT start the bot or touch Discord: `client.run(TOKEN)` sits under
  `if __name__ == "__main__":` (verified, bottom of main.py).

Additional probes:

```bash
which ffmpeg                                          # music prerequisite
python3 -m pip show PyNaCl | head -2                  # voice encryption present
find . -name "*.so" -not -path "./.git/*"             # empty = no stale compiled artifacts
```

## When NOT to use this skill

- Starting/operating the bot, the self-update flow, multi-machine handoff, where
  artifacts land at runtime → **autocroissant-run-and-operate**.
- AI generation (install, torch, doctrine) → **autocroissant-ai-boundary** on branch
  `archive/ai-generation`.
- What each config.py / global_config.py field means, runtime-settable vs
  restart-required → **autocroissant-config-and-flags**.
- Debugging a running bot's misbehavior → **autocroissant-debugging-playbook**
  (except the stale-.so shadow trap, which is owned here).
- The full story of the f7c915c removeprefix regression, the Cython 3.3 management.py
  failure and other incidents → **autocroissant-failure-archaeology**.
- Building the stoogedings icon font (its own venv and requirements in TTSCardMaker) →
  **ttscardmaker-companion**.

## Provenance and maintenance

Verified 2026-07-11 on the owner's Mac (Python 3.10.20, discord.py 2.7.1,
psd-tools 1.14.2, PyGithub 2.9.1, torch 2.9.0, ffmpeg 8.1.1). Re-verified 2026-10-19:
system python3 is now Homebrew 3.14.8 without psd-tools (section 1); Cython 3.3.0.
Re-verified 2026-10-20 against main 9fb3093 (no AI, 7 compiled modules), and again after that day's optimization pass
(11 compiled modules after the psd_analyzer split, the Mac venv `~/.venvs/ac-py314-psd125`, the jetson on 3.14).
Line numbers drift; re-verify volatile facts with:

| Fact | Re-verify with |
|---|---|
| Package lists (the card stack + 12 core, 2026-10-23) | `cat requirements-cards.txt requirements.txt requirements-dev.txt` |
| Pillow and PyNaCl declared (they once arrived via psd-tools and PyGithub) | `grep -n "Pillow\|PyNaCl" requirements.txt` (8, 4) |
| discord.py needs only aiohttp | `python3 -m pip show discord.py \| grep Requires` |
| PIL imports | `grep -n "from PIL" commands/*.py` (top level: card_bake:47, png_export:32, psd_edit:23-24, type_emojis:12; lazy: psd_details, psd_edit) |
| discord.py floor and why | `grep -n "discord" requirements.txt` (`discord.py>=2.5`); `grep -n "versionadded:: 2.5" $(python3 -c 'import discord,os;print(os.path.dirname(discord.__file__))')/client.py` |
| Which interpreter has the bot's deps | `ls -l /usr/local/bin/python3; python3 -c "import psd_tools"` (fails on 2026-10-19; use ~/.venvs/ac-py314-psd125/bin/python on the Mac) |
| Cython 3.3 compiles all 10 | in a scratch copy with setuptools available: `python3 setup.py build_ext --inplace` (never in the live checkout on a running host); quick single-file check: cythonize `commands/management.py` with setup.py's directives and expect no "redeclared" error |
| Uncompiled by omission | `grep -c "type_emojis\|repo_history\|tts_save\|github_api\|card_data\|card_sync" setup.py` (0) |
| davey optional (try/except in discord.py) | `grep -rn -A2 "import davey" $(python3 -c 'import discord,os;print(os.path.dirname(discord.__file__))')/gateway.py` |
| Never-compiled note | `grep -n "Files that CAN be compiled" setup.py` |
| Include list (10 files of 41 command modules) | `grep -n "INCLUDE_FILES = " -A14 setup.py`; `ls commands/*.py \| grep -v __init__ \| wc -l` (41) |
| Compiler directives (bounds checks and wraparound on) | `grep -n "COMPILER_DIRECTIVES" -A14 setup.py` (`'boundscheck': True`, `'wraparound': True` at :38-39) |
| The fork in each venv | `~/.venvs/ac-py314-fork/bin/pip freeze \| grep psd` (`@1611fc6...` = `cards-20261023d` on 2026-10-24); on the jetson `~/venvs/autocroissant/bin/pip freeze \| grep psd` (`@ecc4b773...` = `cards-20261023` until it is moved to the pin) |
| Offline suite size | `python -m pytest --collect-only -q tests \| tail -1` (1346 on 2026-10-23) |
| No compiled artifacts in tree | `find . -name "*.so" -not -path "./.git/*"` |
| Negative-index idiom commit | `git show dd800bc \| grep -E "^[-+].*bboxes"` |
| Cythonization commit date | `git log --format='%h %ad %s' --date=short -1 f7c915c` |
| TOKEN is the only direct config import | `grep -rn "from config import" main.py commands/` |
| Safari cookies (`browser_cookies`, `vram_usage` fallback) | `grep -n "browser_cookies\|cookiesfrombrowser" commands/music_player.py` |
| Tracked stub in music/; .ssh untracked | `git ls-files music .ssh` |
| Token-presence print line | `grep -rn "git token" -i commands/github_api.py main.py` |
| LOCAL_DIR_LOC card-repo path | `grep -n "LOCAL_DIR_LOC" global_config.py` |
| client.run under __main__ guard | `tail -8 main.py` |
| Python version | `python3 --version` (3.14.8 on the Mac venvs and the jetson, 2026-10-20) |
| Dev requirements | `cat requirements-dev.txt` (`-r requirements.txt`, pytest, pyyaml: the workflow-shape tests' yaml half skips without it); `.gitignore` re-includes it and requirements-cards.txt |
| The layering resolves | in a fresh venv: `pip install --dry-run -r requirements-dev.txt`; per target `uv pip compile --python-platform aarch64-unknown-linux-gnu --python-version 3.14 --only-binary :all: --no-binary psd-tools requirements.txt` |

2026-10-21: main.py line numbers re-pointed (0.1 s queue tick, reminder loop); suite count refreshed. The 11 compiled modules still build with Cython 3.3.0 (0 errors).

2026-10-21 (audit fixes): requests import citations re-pointed (github_api.py:4, tts_save.py:31, the new git_commit.py:17); pyyaml added
to requirements-dev.txt. The 11 compiled modules still build with Cython 3.3.0 (query_card.py changed one line: /query_ability's single
result without a PNG posts the name). Review: the uncompiled list names the five 2026-10-21 modules.

2026-10-21 (S6/S7 gate): main.py, repo_traverser.py and help.py line numbers re-pointed after the /bake_cards block, the bake_lock in update_stats and the /bake_cards help entry; nothing else in this skill changed.

2026-10-21 (details): repo_traverser.py citations re-pointed by the old-to-new line map (+3 from :60, +4 from :367, +13 from :407, +16 from :513, +19 from :524: the `psd_details` attributes, the gather hook and the details load/save in update_stats); update_bot.py +1 from :24; global_config.py +1 from :11.

2026-10-22 (data build): the PyGithub citation re-pointed to commands/card_data.py (the traverser's code moved there,
uncompiled); `requirements-card-data.txt` (psd-tools 1.25.0, fonttools, requests) is the Card sync build's own pin
list, not the bot's.

2026-10-22 (bake --local): git_commit.py requests citation re-pointed (:18).

2026-10-23 (docs drift): 14 packages (fonttools, Pillow, PyNaCl rows); section 3 is history (both declared since
2026-10-20); GitPython imported lazily (778c2ce); the fork venvs (Mac `~/.venvs/ac-py314-fork`, the jetson at tag
`cards-20261023`, gcc for `_distance`); bounds checks and wraparound on since 9919a14 (the `boundscheck=False` row
folded into history, ba27fb7); 11 of 38 modules compiled, the uncompiled list extended; suite 1346 tests;
import citations re-pointed at 2ada357.

2026-10-23 (unbatch): pandas' query_card.py import re-pointed (:13).

2026-10-23 (three layered files, one pin): requirements-card-data.txt became requirements-cards.txt (the fork pin,
fonttools, requests, pyoxipng behind its wheel marker); requirements.txt and requirements-dev.txt `-r` the layer below
(Cython, setuptools and pyobjc-framework-Vision added, uharfbuzz via psd-tools[text]); the import scan of code, scripts,
skills scripts and tests found nothing else missing and nothing unused (davey and PyNaCl: discord.py voice).

2026-10-24 (review): where the docs describe the pin they no longer copy its tag (the file alone holds it; venv states
keep theirs); the fork's version is no longer claimed fixed at 1.25.0; the Mac's 3.14 venv without pyoxipng is written
down (pyoxipng row); the evidence row's venv commit re-pointed (1611fc6).

2026-10-24 (py314, branch work/py314): every card Action on 3.14 with one wheel cache entry (card-sync one venv,
card-check and font-check on the fork); pyoxipng's marker takes Linux x86_64 at any Python (built from its sdist on the
Actions); the Mac's cold-install numbers.

2026-10-25 (step 10, work/pickles): no pickles to sync between machines, no /push; 10 compiled modules (repo_traverser.py deleted).
