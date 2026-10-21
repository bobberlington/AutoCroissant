---
name: autocroissant-build-and-env
description: Load this skill whenever the task involves installing, setting up, or rebuilding the AutoCroissant environment or its Cython build — trigger words include install, setup, environment, venv, requirements, pip, dependencies, torch install, CUDA, MPS, new machine, migrate machine, ffmpeg, opus, voice not working after install, cython, compile, build, build_ext, setup.py, stale .so, "my edits don't take effect", ImportError after install, Pillow/PyNaCl missing, "redeclared" / CompileError from Cython 3.3, discord.py>=2.5 / application emojis / AttributeError fetch_application_emojis, "python3 has no psd_tools", which Python to use, jetson deploy. Contains verified prerequisites (Python version and the 2026-10-19 state of the Mac's interpreters, ffmpeg/opus, config.py minimum), main's single requirements.txt with what every package is for (the AI stack and requirements2.txt live only on branch archive/ai-generation), the Pillow/PyNaCl transitive-dependency traps, a Mac vs production jetson table, the full Cython build/clean procedure with compiler-directive consequences (negative-indexing ban, C division, Cython 3.3's no-re-annotation rule; all 7 modules compile with Cython 3.3.0 since 2026-10-19; type_emojis.py stays uncompiled), directory conventions (music/, .ssh/), and copy-pasteable from-scratch checklists for dev-only and full-host environments plus an env smoke test.
---

# AutoCroissant: Build and Environment

Recreating the bot's environment from scratch (the owner's Mac or the production jetson),
the Cython build, and the known traps. All facts verified against the repo on 2026-07-11
unless marked otherwise; sections 1, 2, 3 and 5 re-verified 2026-10-19; AI removal, module
counts, cookies and line numbers re-verified 2026-10-20 (main 9fb3093). All commands assume
`cwd = /Users/michaelsrouji/Desktop/AutoCroissant` (repo root) unless stated.

Quick orientation:

- One requirements file on main: `requirements.txt`. AI generation and its
  `requirements2.txt` live only on branch `archive/ai-generation` (section 2).
- Cython compilation is optional on dev: the Mac's working tree has no `.so` files and runs
  plain `python3 main.py`. Production (the jetson) compiles the 11 modules on every deploy (section 5).
- `config.py` is secret, gitignored, and must be created by hand on every machine.

## 1. Prerequisites

| Prereq | Detail | Verified how (2026-07-11) |
|---|---|---|
| Python 3.10+ | On 2026-07-11, 3.10.20 at `/usr/local/bin/python3` was the verified-working interpreter on the Mac. **On 2026-10-19 that is gone:** `/usr/local/bin/python3` is a symlink to Homebrew `python3.14` (3.14.8) with discord.py 2.7.1, Pillow 12.3.0 and Cython 3.3.0 but **no psd-tools** (and no setuptools), so it cannot import the bot. A durable venv with every dependency now exists: `~/.venvs/ac-py314-psd125` (3.14.8, psd-tools 1.25.0, discord.py 2.7.1, Cython 3.3.0, pytest 9.1.1; `requirements-dev.txt` adds pytest). The old scratch venv in /private/tmp is gone on reboot. Production (the jetson) runs Python 3.14.8 from `~/venvs/autocroissant` (psd-tools 1.25.0) since 2026-10-20; its 3.10 was removed the same day (NEXT_SESSION_PLAN Task 0). Keep code 3.10-compatible. | `python3 --version; python3 -c "import psd_tools"` |
| git | Needed to clone; also the bot's self-update commands drive git via GitPython (`commands/update_bot.py:2`). | `git --version` |
| C compiler | Only if you will run the Cython build (section 5). Xcode CLT on Mac, gcc on Linux. | `cc --version` |
| ffmpeg on PATH | Required for music playback: the player streams through `FFmpegOpusAudio` (`commands/music_player.py:4`), which spawns the `ffmpeg` binary. The `POSTPROCESS` config flag adds an ffmpeg mp3 extraction step too (music_player.py:62-68). Mac has 8.1.1 via Homebrew (`/opt/homebrew/bin/ffmpeg`). | `which ffmpeg` |
| PyNaCl | Required for Discord voice encryption. You do NOT install it directly — it arrives transitively via PyGithub (see section 3, this is a trap). | `python3 -m pip show PyNaCl` |
| opus library | Standard discord.py voice requirement. This bot's `FFmpegOpusAudio` path delegates opus encoding to ffmpeg, so libopus is only strictly needed if anything switches to PCM audio (the volume-control path suggests exactly that, music_player.py:327). Cheap insurance: `brew install opus` / `apt install libopus0`. Labeled guidance. | `brew list opus` |
| config.py | Created BY HAND in the repo root; gitignored; contains tokens — never commit, never print. Only `TOKEN` is required to boot: `main.py:12` is `from config import TOKEN` and every other field is read with `getattr(config, ..., default)` (e.g. utils.py:19, music_player.py:37). A TOKEN-only config boots, but `ADMINS` then defaults to `[]` so admin-gated commands deny everyone. Field names and semantics: see **autocroissant-config-and-flags**. | `grep -n "from config import" main.py` |
| TTSCardMaker clone (full host only) | The card-source repo is expected at `~/Desktop/TTSCardMaker` (`global_config.py:4`, `LOCAL_DIR_LOC`). The bot works without it: `/update_stats` reads GitHub only since local mode was archived on 2026-10-20 (to branch `archive/local-update-stats`; autocroissant-run-and-operate §7). The clone serves the diagnostics scripts (parse_one.py, gap_trace.py), sandbox sweeps, and the stoogedings icon font build (ttscardmaker-companion). | `ls ~/Desktop/TTSCardMaker` |

## 2. Requirements

Main has one file, `requirements.txt`. AI generation left main on 2026-10-20 (05c643e): its
`requirements2.txt` (torch trio, diffusers etc., the CUDA nightly index line) and the
clean-venv proof that core runs without it live only on branch `archive/ai-generation`,
with the doctrine in **autocroissant-ai-boundary** there. The `discord.py>=2.5` floor is
committed (53db8ee).

### requirements.txt — core bot (always install)

`pip install -r requirements.txt` — 11 packages:

| Package | Why it is in core |
|---|---|
| cython | Build tooling for the optional `setup.py` compile (section 5). Harmless if you never compile. |
| davey | Optional Discord "DAVE" voice end-to-end-encryption lib (0.1.6 as of 2026-07-11). discord.py imports it inside `try/except ImportError` (verified in installed discord 2.7.1, `gateway.py`), so the bot runs without it — the Mac's current env does not even have it installed. |
| discord.py (`>=2.5` since 2026-10-19) | The Discord API library (2.7.1 on the Mac). Its only hard dependency is aiohttp. The floor exists because `commands/type_emojis.py` uses `Client.fetch_application_emojis` / `create_application_emoji`, both `versionadded:: 2.5`; on an older install the emoji sync fails with "Emoji sync failed: ... has no attribute ..." while everything else works. Upgrade with `pip install -U "discord.py>=2.5"` |
| GitPython | The bot's self-update commands (`/push`, `/pull`, `/update`) drive git from `commands/update_bot.py:2`. |
| numpy | Image array plumbing in `commands/utils.py:7` and `commands/frankenstein.py:2`. |
| opencv-python | Image encode/decode/resize, imported lazily inside the functions that need it (2026-10-20; only /frankenstein): `commands/utils.py:109, 117` (`url_to_cv2image`, `cv2discordfile`), `commands/frankenstein.py:65`. |
| pandas | The card query engine's DataFrames: `commands/query_card.py:9`, `commands/psd_analyzer.py:13`. |
| psd-tools | The PSD parser (`commands/psd_parser.py:10`). **Brings Pillow transitively — load-bearing, see section 3.** |
| PyGithub | GitHub API traversal for remote card updates (`commands/repo_traverser.py:104`, imported lazily in `_repo`). **Brings PyNaCl transitively — load-bearing, see section 3.** |
| requests | HTTP fetches/downloads: `commands/utils.py:10`, `commands/query_card.py:8`, `commands/github_api.py:2`, `commands/tts_save.py:32`. |
| yt-dlp | Music downloading (`commands/music_player.py:73`). Deliberately unpinned — updating it (`pip install -U yt-dlp`) is the first move when downloads break (autocroissant-debugging-playbook §6). |

## 3. Transitive-dependency traps (load-bearing)

Two packages the bot's own code imports at top level are NOT in any requirements file —
they ride in as dependencies of other packages. Verified via pip metadata 2026-07-11:

| Hidden dep | Arrives via | Who needs it | Evidence |
|---|---|---|---|
| Pillow (PIL) | psd-tools (`pip show psd-tools` → Requires: attrs, numpy, **Pillow**, typing-extensions) | Top-level `from PIL import Image, ImageDraw` in `commands/type_emojis.py:11` (emoji images), the only PIL import on main; `main.py:75` and `query_card.py:19` import that module at load time | Remove psd-tools and the whole bot fails at import, not just PSD parsing. Declared explicitly since 2026-10-20. |
| PyNaCl | PyGithub (`pip show PyGithub` → Requires: pyjwt, **pynacl**, requests, typing-extensions, urllib3) | discord.py voice encryption (voice/music is silent-broken without it) | discord.py itself only requires aiohttp (and audioop-lts on 3.13+); it does not pull PyNaCl. Declared explicitly since 2026-10-20. |

**The rule**: if psd-tools or PyGithub is ever removed from `requirements.txt`, add
`Pillow` and/or `PyNaCl` explicitly in the same commit. The rationale: nothing in this
repo declares these two directly, so today's env only works because of an accident of
the dependency graph — and pip will not warn you when the accident stops happening.

## 4. Mac vs production (the jetson)

Same code, same pickles (synced via `/push` and the update flow — see
**autocroissant-run-and-operate**). What differs:

| Axis | Mac (dev) | Jetson (production) |
|---|---|---|
| Python / run | venv (section 1), plain `python3 main.py` | Python 3.14 venv `~/venvs/autocroissant`, systemd `autocroissant.service` (ExecStart = the venv's python), compiled `.so` (section 5) |
| yt-dlp cookies | `browser_cookies = True` in config.py → `cookiesfrombrowser = ('safari', ...)` (`music_player.py:37`, used at :55). Old configs fall back to `vram_usage == "mps"`. | Unset → no browser cookies |

The torch/CUDA/MPS differences went with the AI to branch `archive/ai-generation`.
Config field semantics: **autocroissant-config-and-flags**.

## 5. The Cython build (optional)

### Commands

```bash
python3 setup.py build_ext --inplace   # compile the 11 INCLUDE_FILES to .so next to the .py
python3 setup.py clean                 # remove *.c *.cpp *.so *.pyd *.pyc *.html, __pycache__/, build/
```

Notes verified from `setup.py` (2026-07-11):

- The build ALSO runs the artifact cleanup first (`clean_artifacts()` is called at the
  top of `__main__`, setup.py:138), so every build is a from-clean rebuild. Side
  effect: both commands delete ALL `*.html` files anywhere under the repo (rglob).
- Compiled modules (`INCLUDE_FILES`, setup.py:21-33) — 11 of the 17 `commands/` modules
  (2026-10-20): analytics, frankenstein, help, psd_analyzer and the four modules its code was split into that day
  (card_models, card_edits, psd_parser, repo_traverser: same directives, so the moved code keeps its compiled
  semantics), query_card, management, music_player.
- **Not compiled because they are in neither list:** `commands/type_emojis.py` (added 2026-10-19),
  `commands/repo_history.py`, `commands/tts_save.py` and `commands/github_api.py` (2026-10-20). New modules are uncompiled by default; all were deliberately left out. Note that compiled
  `query_card` imports it at module level, which is fine (compiled code can import plain modules).
- Never compiled (`EXCLUDE_FILES`, setup.py:9-18), with the reasons stated in the
  file's comments: `main.py` (entry point — keep as Python), `commands/update_bot.py`
  (uses `execv`/`execl` for restart), `commands/utils.py` (core utilities, decorators,
  dynamic behavior), `config.py`/`global_config.py` (configuration), `setup.py` itself,
  and both `__init__.py` files.

### Compiler directives and their consequences (setup.py:36-48)

This is where the coding-discipline rules for compiled modules come from. Rules
without their incident get ignored, so each one carries its story:

| Directive | Consequence in the 11 compiled modules |
|---|---|
| `wraparound=False` | **NEGATIVE INDEXING IS BANNED.** Compiled code skips Python's negative-index handling; `x[-1]` can read garbage or crash instead of meaning "last element". House idiom: `x[len(x)-1]` — see commit dd800bc (2026-01-30), which changed `bboxes[-1][1].y` to `bboxes[len(bboxes)-1][1].y`, and the live idiom at `music_player.py:132` (`prev_music[len(prev_music)-1]`). The ban has already drawn blood once: the cythonization commit f7c915c mechanically purged a `split("TTSCardMaker")[-1]` and replaced it with a `removeprefix()` that no-ops on absolute paths — the local-mode stats-corruption bug (local mode archived 2026-10-20). Full story: **autocroissant-failure-archaeology**. |
| `boundscheck=False` | Out-of-range indexes are undefined behavior (crash or garbage), not a clean `IndexError`. Check lengths before indexing. |
| `cdivision=True` | C division semantics for negatives: quotient truncates toward zero and `%` takes the dividend's sign. In Python `-7 // 2 == -4` and `-7 % 2 == 1`; in compiled code they become `-3` and `-1`. Any div/mod on possibly-negative ints must be reasoned about. |
| `embedsignature=True` | Function signatures survive into docstrings — `help()` still works on compiled functions. |
| (Cython 3.3 itself, not a directive) | **Declare an annotated variable once.** Cython 3.3 rejects re-annotating a name already declared in the same function (`'target_message' redeclared`), which plain Python allows. `cythonize()` then raises `CompileError` before `setup()` runs, and because `clean_artifacts()` already deleted the old artifacts, the host ends up with NO compiled modules and runs plain Python without any obvious sign. This hit `commands/management.py` (`add_reactions`) until 2026-10-19: autocroissant-failure-archaeology Entry 10.5. Since the fix, all INCLUDE_FILES modules compile with Cython 3.3.0 and setup.py's directives (7 on 2026-10-19; all 11 in a scratch build after the 2026-10-20 split). |
| (Cython itself: annotations) | **Type hints are enforced and coerce** in compiled modules (measured with Cython 3.3, 2026-10-20): `x: list` rejects a tuple, `x: dict` rejects a defaultdict, `x: str`/`int` reject None unless the default is None, `x: float` turns 1 into 1.0, `x: bool` turns None into False, `-> dict` rejects a defaultdict return. Untouched: ABCs, non-builtin classes, unions, `Optional[...]` with a None default, dataclass fields, `-> int`. So a hint change in an INCLUDE_FILES module is a behaviour change: gate it with a compiled run (the 2026-10-20 split kept the moved code compiled for exactly this reason). Known case: compiled `analytics.parse_interval(None)` raises TypeError; the .py returns None. |
| Also set | `language_level='3'`, `initializedcheck=False`, `infer_types=True`, `overflowcheck=False`, `profile=False`, `linetrace=False`. |

### Compilation is optional — and the stale-.so trap

- The Mac's working tree has **no** `.so`/`.c` artifacts (2026-10-20) and runs plain Python.
- Production (the jetson, no startup.sh) compiles on every deploy: `git pull`,
  `~/venvs/autocroissant/bin/python setup.py build_ext --inplace`, `sudo systemctl restart autocroissant`
  (**autocroissant-run-and-operate**). It runs the 11 `.so` files, so the stale-.so trap below
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
pip install -r requirements.txt
pip install -r requirements-dev.txt   # pytest, for the offline suite in tests/ (dev only)
```

Then:

1. Create `config.py` in the repo root by hand. Minimum content is one line defining
   `TOKEN` (the Discord bot token). Field names and what each does:
   **autocroissant-config-and-flags**. Never commit this file.
2. Smoke test (section c). Music playback also needs ffmpeg (section 1).
3. Offline suite: `python -m pytest -q tests` (about 13 s, 220 tests on 2026-10-20; needs no config.py,
   network or Discord, never writes the pickles; PSD tests skip without `~/Desktop/TTSCardMaker`).
   Owner: **autocroissant-validation-and-qa** §1 row 1c.

### (b) Full host environment (a machine that will actually host the bot)

1. Steps 1-5 of checklist (a).
2. `config.py`: tokens/fields per **autocroissant-config-and-flags**; `browser_cookies = True`
   on the Mac for Safari cookies (section 4).
3. ffmpeg: `brew install ffmpeg` (Mac) / `apt install ffmpeg` (Linux). Optional
   insurance: opus (section 1). Verify: `which ffmpeg`.
4. Clone the card source where the bot expects it:
   `git clone https://github.com/MichaelJSr/TTSCardMaker ~/Desktop/TTSCardMaker`
   (path is `LOCAL_DIR_LOC`, global_config.py:4). The bot works without it: `/update_stats`
   reads GitHub only since local mode was archived on 2026-10-20
   (**autocroissant-run-and-operate** §7). The repo's history is about 25 GB of PSDs; the dev Mac's clone
   is blobless (`--filter=blob:none`) and shallow (depth 1) as of 2026-10-19, so local `git log`
   there has no history (ttscardmaker-companion). Local checkouts also give every PSD a fresh
   mtime, which is why local-timestamp updates misfired and local mode was archived
   (autocroissant-run-and-operate §7).
5. Deploy key: place the bot's SSH key pair in `.ssh/` if this machine will run
   `/push` (see **autocroissant-run-and-operate** for the self-update flow).
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
python3 -m pip show PyNaCl | head -2                  # voice encryption present (via PyGithub)
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
| Package list (11 core) | `cat requirements.txt` |
| Pillow arrives via psd-tools | `python3 -m pip show psd-tools \| grep Requires` |
| PyNaCl arrives via PyGithub | `python3 -m pip show PyGithub \| grep Requires` |
| discord.py needs only aiohttp | `python3 -m pip show discord.py \| grep Requires` |
| Top-level PIL imports | `grep -n "from PIL" commands/*.py` (type_emojis:11 only) |
| discord.py floor and why | `grep -n "discord" requirements.txt` (`discord.py>=2.5`); `grep -n "versionadded:: 2.5" $(python3 -c 'import discord,os;print(os.path.dirname(discord.__file__))')/client.py` |
| Which interpreter has the bot's deps | `ls -l /usr/local/bin/python3; python3 -c "import psd_tools"` (fails on 2026-10-19; use ~/.venvs/ac-py314-psd125/bin/python on the Mac) |
| Cython 3.3 compiles all 11 | in a scratch copy with setuptools available: `python3 setup.py build_ext --inplace` (never in the live checkout on a running host); quick single-file check: cythonize `commands/management.py` with setup.py's directives and expect no "redeclared" error |
| Uncompiled by omission | `grep -c "type_emojis\|repo_history\|tts_save\|github_api" setup.py` (0) |
| davey optional (try/except in discord.py) | `grep -rn -A2 "import davey" $(python3 -c 'import discord,os;print(os.path.dirname(discord.__file__))')/gateway.py` |
| Exclude list + reasons | `grep -n "EXCLUDE_FILES" -A12 setup.py` |
| Include list (11 files of 17 command modules) | `grep -n "INCLUDE_FILES = " -A14 setup.py`; `ls commands/*.py \| grep -v __init__ \| wc -l` (17) |
| Compiler directives | `grep -n "COMPILER_DIRECTIVES" -A14 setup.py` |
| No compiled artifacts in tree | `find . -name "*.so" -not -path "./.git/*"` |
| Negative-index idiom commit | `git show dd800bc \| grep -E "^[-+].*bboxes"` |
| Cythonization commit date | `git log --format='%h %ad %s' --date=short -1 f7c915c` |
| TOKEN is the only direct config import | `grep -rn "from config import" main.py commands/` |
| Safari cookies (`browser_cookies`, `vram_usage` fallback) | `grep -n "browser_cookies\|cookiesfrombrowser" commands/music_player.py` |
| Tracked stub in music/; .ssh untracked | `git ls-files music .ssh` |
| Token-presence print line | `grep -rn "git token" -i commands/github_api.py main.py` |
| LOCAL_DIR_LOC card-repo path | `grep -n "LOCAL_DIR_LOC" global_config.py` |
| client.run under __main__ guard | `tail -8 main.py` |
| Python version | `python3 --version` (3.14.8 on the Mac venv and the jetson, 2026-10-20) |
| Dev requirements | `cat requirements-dev.txt` (pytest); `.gitignore` re-includes it (`!requirements-dev.txt`) |
