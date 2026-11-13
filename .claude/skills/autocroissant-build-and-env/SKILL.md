---
name: autocroissant-build-and-env
description: Installing the AutoCroissant environment. Use for install, setup, venv, pip, requirements.txt, requirements-bot.txt (the jetson's pinned bot packages) and requirements-cards.txt (the psd-tools fork pin), a new machine, ffmpeg/opus, which Python to use, "python3 has no psd_tools", ImportError after install, leftover Cython .so files (no build since 2026-11-11: stale .so, my edits don't take effect), AttributeError fetch_application_emojis. Jetson deploys use jetson-deploy.
---

# AutoCroissant: Build and Environment

Recreating the bot's environment from scratch (the owner's Mac or the production jetson)
and the known traps. Plain Python everywhere since 2026-11-11 (nothing compiled; the jetson a bot-only venv). All facts verified against the repo on 2026-07-11
unless marked otherwise; sections 1, 2, 3 and 5 re-verified 2026-10-19; AI removal, module
counts, cookies and line numbers re-verified 2026-10-20 (main 9fb3093); requirements, PIL imports, the fork venvs,
directives, module counts and the suite size re-verified 2026-10-23 (main 2ada357). All commands assume
`cwd = /Users/michaelsrouji/Desktop/AutoCroissant` (repo root) unless stated.

Quick orientation:

- Four requirements files on main (2026-11-11): `requirements-cards.txt` (the card stack, the psd-tools fork's one
  pin), `requirements-bot.txt` (the bot's packages pinned from the jetson's freeze: all the jetson installs),
  `requirements.txt` (`-r` both: the Mac), `requirements-dev.txt` (`-r requirements.txt`, then the tests). AI
  generation and its `requirements2.txt` live only on branch `archive/ai-generation` (section 2).
- Nothing is compiled since 2026-11-11 (setup.py deleted, section 5): every machine runs the `.py` files the suite tests.
- `config.py` is secret, gitignored, and must be created by hand on every machine.

## 1. Prerequisites

| Prereq | Detail | Verified how (2026-07-11) |
|---|---|---|
| Python 3.10+ | On 2026-07-11, 3.10.20 at `/usr/local/bin/python3` was the verified-working interpreter on the Mac. **On 2026-10-19 that is gone:** `/usr/local/bin/python3` is a symlink to Homebrew `python3.14` (3.14.8; 3.14.10 since 2026-11-11) with discord.py 2.7.1, Pillow 12.3.0 and Cython 3.3.0 but **no psd-tools** (and no setuptools), so it cannot import the bot. The Mac venv with every dependency is `~/.venvs/ac-py314-fork` (3.14.10 since 2026-11-11, rebuilt with the same packages; the psd-tools fork at the pin: `cards-20261112c` = 75264966 since 2026-11-12, first `cards-20261023` = ecc4b773; the jetson's package versions since 2026-11-10; `requirements-dev.txt` adds pytest, pytest-xdist and pyyaml): the suite, card_guards and bakes. The stock psd-tools 1.25.0 venv `~/.venvs/ac-py314-psd125` was retired on 2026-11-10 (owner): the suite needs the fork and stops without it. Production (the jetson) runs the hand-built Python 3.14.10 (PGO+LTO, OpenSSL 3.6.5) since 2026-11-10 (3.14.8 from 2026-10-20; its venvs were deleted 2026-11-11): `~/venvs/autocroissant` is a symlink to `autocroissant-bot-3.14.10` since 2026-11-11, built from `requirements-bot.txt` alone (379 MB; no psd-tools fork, scipy, fontTools, uharfbuzz or Cython); `autocroissant-3.14.10` before it held the whole card stack (611 MB; the fork from 2026-10-22) and was deleted once the switch was proven (section 4). Keep code 3.10-compatible. Mac Python upgrades: Homebrew `python@3.14` is `brew pin`ned and the venvs link through `/opt/homebrew/opt/python@3.14`, which the upgrade moves at once: first point each venv's `bin/python3.14` and pyvenv.cfg `home` at the versioned Cellar (`/opt/homebrew/Cellar/python@3.14/<old>/bin`), then `brew unpin python@3.14`, `HOMEBREW_NO_INSTALL_CLEANUP=1 brew upgrade python@3.14`, `brew pin python@3.14`; each venv is rebuilt beside the old one from its `pip freeze` (the fork from requirements-cards.txt's line with `--force-reinstall --no-deps`), swapped in by rename between suites, and the old one deleted once the suite and the TTS checks pass. No `brew cleanup` while a running process or a kept venv uses the old Cellar. | `python3 --version; python3 -c "import psd_tools"` |
| git | Needed to clone; also the bot's self-update commands drive git via GitPython, imported lazily inside update_bot's functions (`_git`/`_repo`, update_bot.py:19, 27; since 778c2ce, 2026-10-21: startup never loads GitPython). | `git --version` |
| C compiler | Only where the psd-tools fork installs (the Mac, the Actions): its Cython modules compile at install. Xcode CLT on the Mac. The bot needs none: the jetson installs wheels only (`--only-binary :all:`). | `cc --version` |
| ffmpeg on PATH | Required for music playback: the player streams through `FFmpegOpusAudio` (`commands/music_player.py:4`), which spawns the `ffmpeg` binary. The `POSTPROCESS` config flag adds an ffmpeg mp3 extraction step too (music_player.py:62-68). Mac has 8.1.1 via Homebrew (`/opt/homebrew/bin/ffmpeg`). | `which ffmpeg` |
| PyNaCl | Required for Discord voice encryption. Declared in requirements.txt since 2026-10-20 (before that it arrived only through PyGithub: section 3). | `python3 -m pip show PyNaCl` |
| opus library | Standard discord.py voice requirement. This bot's `FFmpegOpusAudio` path delegates opus encoding to ffmpeg, so libopus is only strictly needed if anything switches to PCM audio (the volume-control path suggests exactly that, music_player.py:327). Cheap insurance: `brew install opus` / `apt install libopus0`. Labeled guidance. | `brew list opus` |
| config.py | Created BY HAND in the repo root; gitignored; contains tokens — never commit, never print. Only `TOKEN` is required to boot: `main.py:13` is `from config import TOKEN` and every other field is read with `getattr(config, ..., default)` (e.g. utils.py:19, music_player.py:37). A TOKEN-only config boots, but `ADMINS` then defaults to `[]` so admin-gated commands deny everyone. Field names and semantics: see **autocroissant-config-and-flags**. | `grep -n "from config import" main.py` |
| TTSCardMaker clone (full host only) | The card-source repo is expected at `~/Desktop/TTSCardMaker` (`global_config.py:5`, `LOCAL_DIR_LOC`). The bot works without it: `/update_stats` reads GitHub only since local mode was archived on 2026-10-20 (to branch `archive/local-update-stats`; autocroissant-run-and-operate §7). The clone serves the diagnostics scripts (parse_one.py, gap_trace.py), sandbox sweeps, and the stoogedings icon font build (ttscardmaker-companion). | `ls ~/Desktop/TTSCardMaker` |

## 2. Requirements

Three layered files since 2026-10-23 (owner: "three layered files, one pin"):

| File | Holds | Installed by |
|---|---|---|
| `requirements-cards.txt` | The card code's packages: `psd-tools[composite,text] @ git+https://github.com/MichaelJSr/psd-tools@<tag>` (the fork: THE one pin; the tag is only in the file), fonttools, requests (`pyoxipng` behind a marker from 2026-10-24 until 2026-10-25: the Actions run the oxipng CLI since) | `requirements.txt`; the card Actions on Python 3.14 (card-guards' jobs, card-sync's one venv, card-preview), offline from one wheel cache entry keyed on the file (the re-pin procedure: autocroissant-config-and-flags) |
| `requirements-bot.txt` | The bot's packages and their dependencies, pinned `==` from the jetson's `pip freeze` of 2026-11-11 without the card toolchain (the fork, scipy, fontTools, uharfbuzz, freetype-py, aggdraw) and Cython, setuptools, wheel (29 lines; davey 0.1.7: 0.2 breaks voice). `tests/test_bot_env.py` walks main.py's import closure, lazy imports included, and fails if it reaches the toolchain or imports a package with no pin here | the jetson (`pip install --only-binary :all: -r requirements-bot.txt`, every deploy); the Mac through requirements.txt |
| `requirements.txt` | `-r requirements-cards.txt`, `-r requirements-bot.txt` | a dev machine |
| `requirements-dev.txt` | `-r requirements.txt`, then pytest and pyyaml (the workflow tests import it); `pyobjc-framework-Vision` went with `commands/discord_match.py` on 2026-10-25 | a dev machine |

uharfbuzz and freetype-py come with psd-tools' `[text]` extra, scipy and aggdraw with `[composite]`. `.gitignore`
ignores `*.txt` and re-includes `requirements-dev.txt`, `requirements-cards.txt` and `requirements-bot.txt`;
`requirements.txt` is tracked from before that rule. Proven 2026-10-23: `pip install --dry-run -r requirements-dev.txt` in a fresh 3.14 venv on the Mac
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

`pip install -r requirements.txt` — the card stack (marked "cards") plus requirements-bot.txt's pins (2026-11-11; the
top-level ones below, their dependencies pinned beside them):

| Package | Why it is in core |
|---|---|
| Cython, setuptools | Removed 2026-11-11 with setup.py (section 5). |
| davey | Pinned 0.1.7 (0.2's transition API breaks discord.py voice). Optional Discord "DAVE" voice end-to-end-encryption lib. discord.py imports it inside `try/except ImportError` (verified in installed discord 2.7.1, `gateway.py`), so the bot runs without it — the Mac's current env does not even have it installed. |
| discord.py (`>=2.5` since 2026-10-19) | The Discord API library (2.7.1 on the Mac). Its only hard dependency is aiohttp. The floor exists because `commands/type_emojis.py` uses `Client.fetch_application_emojis` / `create_application_emoji`, both `versionadded:: 2.5`; on an older install the emoji sync fails with "Emoji sync failed: ... has no attribute ..." while everything else works. Upgrade with `pip install -U "discord.py>=2.5"` |
| fonttools (cards) | Font files: `commands/font_check.py` (the guards' missing-fonts check, the data build's fonts report), reached by `psd_details` and the bake. The bot never imports it since 2026-11-11: `/fonts` reads the report the data build writes (the manifest's `fonts`). |
| GitPython | The bot's self-update commands (`/pull`, `/update`; `/push` until 2026-10-25), imported lazily inside update_bot's functions (update_bot.py:19, 27; startup never loads GitPython). |
| numpy | The card name index in `commands/query_card.py` (`_NameIndex`: the closest-name guesses); the card code's pixel checks too. |
| opencv-python | Removed 2026-11-12 with /frankenstein, its only user (`commands/frankenstein.py` and utils' `url_to_cv2image`/`cv2discordfile`; owner). TTSCardMaker's font build keeps its own opencv-python-headless. |
| pandas | The card query engine's DataFrames: `commands/query_card.py:13`, `commands/psd_analyzer.py:13`. |
| Pillow | Images: top level in `commands/type_emojis.py:12` (emoji images, loaded at startup), `png_export.py:32`, `card_bake.py:47`, `psd_edit.py:23`, `card_grid.py` (/query_ability's grids; `ImageFont.load_default(size)` wants Pillow >= 10.1 with FreeType: the wheels have it); lazily in `psd_details.py:130` (ImageCms). Declared since 2026-10-20 (section 3). |
| psd-tools (cards) | The PSD parser (`commands/psd_parser.py:11`), the fork at the one pin: its text engine serves the bake, the push guards and range reads (`commands/psd_remote.py`, section 4). Not on the jetson since 2026-11-11. Its version does not follow the tag (1.25.0 up to `cards-20261023d`, 1.26.0+cards since `cards-20261024`, also `cards-20261025`, `cards-20261025b`, `cards-20261026`, `cards-20261104`, `cards-20261107`, `cards-20261110`, `cards-20261111`, `cards-20261112`, `cards-20261112b` and `cards-20261112c`), and pip calls an install of the same version satisfied, so a re-pin needs `--force-reinstall` (section 4). |
| oxipng CLI (cards; not a pip package since 2026-10-25, docs/Finished/session-2026-10-25/SMALLER_PNG_PUSHES.md) | png_export's lossless PNG recompression of card_preview's previews (`png_export.optimize`; since 2026-11-11 the PNGs the guards, the bake and the sweeps commit are banded by `png_export.band`, zlib only, no oxipng): the `oxipng` binary on PATH, 10.2.1 in the Actions (cards-python keeps the release, its pinned sha256 checked, in the cached venv's `.oxipng/`, on PATH only in the jobs that write PNGs: the guards' check and post, Card sync's Bake and save); on the Mac `brew install oxipng` (10.2.1, as the Actions). Without it no recompression (the pyoxipng fallback was dropped on 2026-11-10); everything runs without it. The tests need it (tests/conftest.py TOOLS); the jetson needs none. |
| PyGithub | Removed 2026-10-25 (step 10): its only importers, card_data's legacy `_committed_since_update` and `_repo`, went with the pickles. PyNaCl stays declared on its own. |
| PyNaCl | discord.py voice encryption (music is silent without it). Declared since 2026-10-20 (section 3). |
| requests (cards) | HTTP fetches/downloads: `commands/query_card.py:9`, `commands/github_api.py:7`, `commands/tts_save.py:35`, `commands/git_commit.py:19`. |
| yt-dlp | Music downloading (`commands/music_player.py:73`). Pinned in requirements-bot.txt since 2026-11-11: when downloads break, bump its pin there and deploy (deploy.sh installs the file; autocroissant-debugging-playbook §6). |

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
| Python / run | venv (section 1), plain `python3 main.py` | Python 3.14 venv `~/venvs/autocroissant` (requirements-bot.txt), systemd `autocroissant.service` (ExecStart = the venv's python), plain Python since 2026-11-11 |
| yt-dlp cookies | `browser_cookies = True` in config.py → `cookiesfrombrowser = ('safari', ...)` (`music_player.py:37`, used at :55). | Unset → no browser cookies |
| psd-tools | `~/.venvs/ac-py314-fork`: the fork at the pin (`cards-20261112c` = 75264966 since 2026-11-12, per `pip freeze`; with pytest-xdist and setuptools: the full suite, card_guards, bakes; the one Mac venv since 2026-11-10, when the stock 1.25.0 `ac-py314-psd125` was retired) | None since 2026-11-11 (the bot-only venv). Until then the fork at the pin, re-pinned by every deploy (gcc, about 170 s per tag); now a fork release only re-pins requirements-cards.txt: nothing on the jetson. The Actions (card-edits, card-guards, card-sync, card-preview) install that file |

The torch/CUDA/MPS differences went with the AI to branch `archive/ai-generation`.
Config field semantics: **autocroissant-config-and-flags**.

## 5. No Cython build (since 2026-11-11)

From 2025-12-02 (f7c915c, "cythonized the code") until 2026-11-11, `setup.py` compiled 10 `commands/` modules (analytics,
frankenstein, help, psd_analyzer, card_models, card_edits, psd_parser, query_card, management, music_player) to `.so`
on the jetson at every deploy; the Mac and the suite ran the `.py` files. The owner turned it off (Fable design panel,
design 3): on the jetson 58 real `/query` strings took 22.6-23.3 ms compiled and 21.9-22.0 ms plain, and the same 70
real queries (68 distinct) replayed through both on 11-11 gave identical replies (warm pass 262.5 ms compiled, 257.3 ms plain).
`setup.py`, `tests/test_setup_build.py`, Cython and setuptools went; the story is autocroissant-failure-archaeology
Entry 6.

What that retires (no sweep needed: existing code stays as written):

- The compiled-module rules: the `x[len(x)-1]` idiom, annotation enforcement (compiled `x: list` rejected a tuple),
  Cython 3.3's "declare an annotated variable once", and `cdivision` (C division for negatives). `x[-1]` is plain
  Python now. The 10-23 bounds-check setting (`boundscheck=True`, `wraparound=True`, 9919a14) is moot.
- The build on the jetson (126-166 s incremental, 208 s full, per deploy) and `/update`'s stale `.so`: it pulled without a build, so compiled
  modules kept running old code; now a pull is the whole update.

### The stale-.so trap (old checkouts only)

An import like `from commands import psd_analyzer` picks a compiled `commands/psd_analyzer.*.so` OVER the `.py`. A
leftover `.so` from an old build silently shadows your edited `.py`: your edits do not take effect, no error anywhere.
The jetson-deploy skill's deploy.sh, in a repo with no `setup.py` at its root (it compiles nothing), removes the files
git does not track: every `*.so`, a `*.c` beside a same-named `.py`/`.pyx` and `build/`; tracked files are never
touched, and it fails if any remain. A repo that still has `setup.py` builds as before. Anywhere else:

```bash
find . -name "*.so" -not -path "./.git/*"           # empty = nothing shadows a .py
```

Only Python's caches stay gitignored (`__pycache__/`, `*.py[cod]`; the Cython patterns went 2026-11-12); deploy.sh
removes untracked `.so`, `.c` and `build/` whatever `.gitignore` says.

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
3. Offline suite: `~/.venvs/ac-py314-fork/bin/python -m pytest -q -n auto tests` (pytest-xdist from requirements-dev.txt; 618 tests on 2026-10-25, about a
   minute; needs no config.py, network or Discord; PSD tests skip without `~/Desktop/TTSCardMaker`; the suite needs the fork
   venv and stops without it, 2026-11-10).
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
   is blobless (`--filter=blob:none`) with full history since 2026-11-10 (shallow, depth 1, before), so `git log`
   works there; run git in it with `GIT_NO_LAZY_FETCH=1` and fetch any blobs you need in one batched call
   (ttscardmaker-companion). Local checkouts also give every PSD a fresh
   mtime, which is why local-timestamp updates misfired and local mode was archived
   (autocroissant-run-and-operate §7).
5. Deploy key: place the bot's SSH key pair in `.ssh/` if this machine will push by hand (the bot itself pushes
   nothing since 2026-10-25; see **autocroissant-run-and-operate** for the self-update flow).
6. Nothing to build (section 5). Production (the jetson) runs `~/venvs/autocroissant/bin/python` (requirements-bot.txt)
   under systemd `autocroissant.service`.
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
find . -name "*.so" -not -path "./.git/*"             # empty = no leftover compiled artifacts
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
| Package lists (the card stack; the bot's 29 pins, 2026-11-11) | `cat requirements-cards.txt requirements-bot.txt requirements.txt requirements-dev.txt` |
| Nothing compiled; the bot's closure free of the toolchain | `ls setup.py` (none); `python -m pytest -q tests/test_bot_env.py` |
| The jetson's venv | `ssh -q jetson 'readlink ~/venvs/autocroissant; ~/venvs/autocroissant/bin/pip freeze \| grep -ciE "psd-tools\|scipy\|fonttools\|cython"'` (`autocroissant-bot-3.14.10`, 0) |
| Pillow and PyNaCl declared (they once arrived via psd-tools and PyGithub) | `grep -in "pillow\|PyNaCl" requirements-bot.txt` |
| discord.py needs only aiohttp | `python3 -m pip show discord.py \| grep Requires` |
| PIL imports | `grep -n "from PIL" commands/*.py` (top level: card_bake:47, png_export:32, psd_edit:23-24, type_emojis:12; lazy: psd_details, psd_edit) |
| discord.py floor and why | `grep -n "discord" requirements.txt` (`discord.py>=2.5`); `grep -n "versionadded:: 2.5" $(python3 -c 'import discord,os;print(os.path.dirname(discord.__file__))')/client.py` |
| Which interpreter has the bot's deps | `ls -l /usr/local/bin/python3; python3 -c "import psd_tools"` (fails on 2026-10-19; use ~/.venvs/ac-py314-fork/bin/python on the Mac) |
| davey optional (try/except in discord.py) | `grep -rn -A2 "import davey" $(python3 -c 'import discord,os;print(os.path.dirname(discord.__file__))')/gateway.py` |
| The fork in the Mac venv | `~/.venvs/ac-py314-fork/bin/pip freeze \| grep psd` (`@75264966...` = `cards-20261112c`); the jetson has none since 2026-11-11 |
| Offline suite size | `python -m pytest --collect-only -q tests \| tail -1` (the count lives in autocroissant-validation-and-qa row 1c) |
| No compiled artifacts in tree | `find . -name "*.so" -not -path "./.git/*"` |
| Negative-index idiom commit | `git show dd800bc \| grep -E "^[-+].*bboxes"` |
| Cythonization commit date | `git log --format='%h %ad %s' --date=short -1 f7c915c` |
| TOKEN is the only direct config import | `grep -rn "from config import" main.py commands/` |
| Safari cookies (`browser_cookies`) | `grep -n "browser_cookies\|cookiesfrombrowser" commands/music_player.py` |
| Tracked stub in music/; .ssh untracked | `git ls-files music .ssh` |
| Token-presence print line | `grep -rn "git token" -i commands/github_api.py main.py` |
| LOCAL_DIR_LOC card-repo path | `grep -n "LOCAL_DIR_LOC" global_config.py` |
| client.run under __main__ guard | `tail -8 main.py` |
| Python version | `python3 --version` (3.14.10: the Mac and its venvs since 2026-11-11, the jetson since 2026-11-10) |
| Dev requirements | `cat requirements-dev.txt` (`-r requirements.txt`, pytest, pyyaml: the workflow tests import it); `.gitignore` re-includes it and requirements-cards.txt |
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

2026-11-05 (ring badges): the fork pin `cards-20261104` = 1258fcbd (tracking scaled) in the dependency row, the Mac vs jetson table and the re-verify row, checked by `pip freeze` in `~/.venvs/ac-py314-fork` and on the jetson (deployed at 3641756); the suite-size row points to autocroissant-validation-and-qa row 1c.

2026-11-10 (fork release): the pin `cards-20261110` = 170a3fce in the Python, dependency, venv and re-verify rows.

2026-11-11 (fork release): the pin `cards-20261111` = da069245 in the same rows.

2026-11-11 (fork release): the pin `cards-20261112` = aa6e38ee (knockout pass-through groups) in the same rows.

2026-11-12 (fork release): the pin `cards-20261112b` = e2fee3b1 (smooth inner bevels: a tent across the band) in the same rows.

2026-11-12 (fork release): the pin `cards-20261112c` = 75264966 (session 1110 rounds 5-8: mask feather, 4096-step gradient maps, Strong text, 8-bit effects and alpha) in the same rows.

2026-11-10 (housekeeping, owner): `~/.venvs/ac-py314-psd125` retired (conftest stops a run without the fork); pyoxipng
left png_export and the fork venv (the oxipng CLI: brew on the Mac); the fork venv at the jetson's versions (pandas 3.1.1,
python-dateutil 4.0.0, tzdata 2026.6, fonttools 4.68.0, uharfbuzz 0.57.0, Cython 3.3.1).

2026-11-11 (docs check): no `EXCLUDE_FILES` (43 of 53 modules unlisted); the Mac's TTSCardMaker clone has full history since 2026-11-10.

2026-11-11 (nocython, owner): setup.py and Cython gone (section 5 rewritten: the measurement, what it retires, the
stale-.so trap for old checkouts); requirements-bot.txt (the jetson's pinned bot packages, tests/test_bot_env.py) and
requirements.txt = cards + bot; the jetson's venv `autocroissant-bot-3.14.10` without the card toolchain; davey and
yt-dlp pinned; the setup.py re-verify rows replaced.

2026-11-12 (asks-1112, owner G4): /frankenstein deleted, and opencv-python with it (requirements-bot.txt 28 pins); the numpy row names its bot use.
