---
name: autocroissant-build-and-env
description: Load this skill whenever the task involves installing, setting up, or rebuilding the AutoCroissant environment or its Cython build — trigger words include install, setup, environment, venv, requirements, pip, dependencies, torch install, CUDA, MPS, new machine, migrate machine, ffmpeg, opus, voice not working after install, cython, compile, build, build_ext, setup.py, stale .so, "my edits don't take effect", ImportError after install, Pillow/PyNaCl missing. Contains verified prerequisites (Python version, ffmpeg/opus, config.py minimum), the requirements.txt vs requirements2.txt split (core bot vs AI enablement) with what every package is for, the Pillow/PyNaCl transitive-dependency traps, a Mac (mps) vs CUDA box differences table, the full Cython build/clean procedure with compiler-directive consequences (negative-indexing ban, C division), directory conventions (models/, music/, .ssh/), and copy-pasteable from-scratch checklists for dev-only and full-host environments plus an env smoke test.
---

# AutoCroissant: Build and Environment

Recreating the bot's environment from scratch on either machine type (the owner's Mac
or the CUDA box), the optional Cython build, and the known traps. All facts verified
against the repo on 2026-07-11 unless marked otherwise. All commands assume
`cwd = /Users/michaelsrouji/Desktop/AutoCroissant` (repo root) unless stated.

Quick orientation:

- Two requirements files: `requirements.txt` (core bot, always) and
  `requirements2.txt` (AI enablement, optional). The bot runs fully without the second.
- Cython compilation is OPTIONAL. No `.so` files exist in the working tree today;
  plain `python3 main.py` is how the bot currently runs.
- `config.py` is secret, gitignored, and must be created by hand on every machine.

## 1. Prerequisites

| Prereq | Detail | Verified how (2026-07-11) |
|---|---|---|
| Python 3.10+ | 3.10.20 at `/usr/local/bin/python3` is the verified-working interpreter on the Mac. Dependencies warn that 3.10 is deprecated (discord.py recommends 3.11+). **Guidance, not a verified requirement: prefer 3.11+ for any NEW env.** | `python3 --version` |
| git | Needed to clone; also the bot's self-update commands drive git via GitPython (`commands/update_bot.py:2`). | `git --version` |
| C compiler | Only if you will run the Cython build (section 5). Xcode CLT on Mac, gcc on Linux. | `cc --version` |
| ffmpeg on PATH | Required for music playback: the player streams through `FFmpegOpusAudio` (`commands/music_player.py:4`), which spawns the `ffmpeg` binary. The `POSTPROCESS` config flag adds an ffmpeg mp3 extraction step too (music_player.py:61-63). Mac has 8.1.1 via Homebrew (`/opt/homebrew/bin/ffmpeg`). | `which ffmpeg` |
| PyNaCl | Required for Discord voice encryption. You do NOT install it directly — it arrives transitively via PyGithub (see section 3, this is a trap). | `python3 -m pip show PyNaCl` |
| opus library | Standard discord.py voice requirement. This bot's `FFmpegOpusAudio` path delegates opus encoding to ffmpeg, so libopus is only strictly needed if anything switches to PCM audio (the volume-control path suggests exactly that, music_player.py:314). Cheap insurance: `brew install opus` / `apt install libopus0`. Labeled guidance. | `brew list opus` |
| config.py | Created BY HAND in the repo root; gitignored; contains tokens — never commit, never print. Only `TOKEN` is required to boot: `main.py:9` is `from config import TOKEN` and every other field is read with `getattr(config, ..., default)` (e.g. diffusion.py:24-29, utils.py:21). A TOKEN-only config boots, but `ADMINS` then defaults to `[]` so admin-gated commands deny everyone. Field names and semantics: see **autocroissant-config-and-flags**. | `grep -n "from config import" main.py` |
| TTSCardMaker clone (full host only) | The card-source repo is expected at `~/Desktop/TTSCardMaker` (`global_config.py:4`, `LOCAL_DIR_LOC`). Remote-mode operation works without it — and until the local-path bug is fixed, remote mode (`use_local_repo:False`) is the safe mode anyway (story in **autocroissant-failure-archaeology**). | `ls ~/Desktop/TTSCardMaker` |

## 2. The requirements split

Reorganized 2026-07-11 per the owner's doctrine: AI is an experiment and must stay
separate and toggleable; the bot must run perfectly without it. The doctrine itself is
owned by **autocroissant-ai-boundary** — this section is the install mechanics.

**Status: the reorg is an UNCOMMITTED working-tree change as of 2026-07-11** — HEAD
284d13c still has the AI packages mixed into requirements.txt, so a fresh checkout or a
`force_reset` does NOT have this split. Check with
`git status --short -- requirements.txt requirements2.txt` (` M` on both while
uncommitted; empty once committed — then re-date this section). Committing it is a
change-control decision (**autocroissant-change-control**).

### requirements.txt — core bot (always install)

`pip install -r requirements.txt` — 11 packages:

| Package | Why it is in core |
|---|---|
| cython | Build tooling for the optional `setup.py` compile (section 5). Harmless if you never compile. |
| davey | Optional Discord "DAVE" voice end-to-end-encryption lib (0.1.6 as of 2026-07-11). discord.py imports it inside `try/except ImportError` (verified in installed discord 2.7.1, `gateway.py`), so the bot runs without it — the Mac's current env does not even have it installed. |
| discord.py | The Discord API library (2.7.1 on the Mac). Its only hard dependency is aiohttp. |
| GitPython | The bot's self-update commands (`/push`, `/pull`, `/update`) drive git from `commands/update_bot.py:2`. |
| numpy | Image array plumbing in `commands/utils.py:8` and `commands/frankenstein.py:3`. |
| opencv-python | Image encode/decode/resize: `commands/utils.py:4` (`url_to_cv2image`, `cv2discordfile`), `commands/frankenstein.py:1`. |
| pandas | The card query engine's DataFrames: `commands/query_card.py:9`, `commands/psd_analyzer.py:17`. |
| psd-tools | The PSD parser (`commands/psd_analyzer.py:11`). **Brings Pillow transitively — load-bearing, see section 3.** |
| PyGithub | GitHub API traversal for remote card updates (`commands/psd_analyzer.py:6`). **Brings PyNaCl transitively — load-bearing, see section 3.** |
| requests | HTTP fetches/downloads: `commands/utils.py:12`, `commands/query_card.py:6`, `commands/psd_analyzer.py:13`. |
| yt-dlp | Music downloading (`commands/music_player.py:12`). Deliberately unpinned — updating it (`pip install -U yt-dlp`) is the first move when downloads break (autocroissant-debugging-playbook §6). |

### requirements2.txt — AI enablement (optional)

`pip install -r requirements2.txt` — the torch trio (torch, torchvision, torchaudio)
plus accelerate, bitsandbytes, diffusers, peft, protobuf, sentencepiece, transformers.

Its first line is a **commented** index switch:

```
#-i https://download.pytorch.org/whl/nightly/cu124
```

- **Mac**: install the file as-is. Plain PyPI torch has MPS support built in
  (torch 2.9.0 is what the Mac's env runs today).
- **CUDA box**: uncomment that first line before installing, to pull the CUDA 12.4
  nightly torch build. That is the owner's documented intent for the line.
  UNVERIFIED caveat (cannot be tested from this Mac): `-i` inside a requirements file
  applies to the whole pip run, and the PyTorch nightly index does not host
  transformers/diffusers/etc. If the one-shot install fails to resolve those, use the
  two-step fallback (candidate, verify on the CUDA box):
  `pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124`
  then `pip install -r requirements2.txt` with the line still commented.

**VERIFIED FACT** (clean-venv experiment, 2026-07-11 — do not re-run casually, it
takes minutes): in a fresh venv with ONLY `requirements.txt` installed, `import main`
succeeds and `commands.diffusion.torch_available()` returns `False`. That is the proof
the core/AI split actually holds. (`torch_available` is at `commands/diffusion.py:146`;
torch is imported lazily inside `get_torch()`, diffusion.py:136-140.)

## 3. Transitive-dependency traps (load-bearing)

Two packages the bot's own code imports at top level are NOT in any requirements file —
they ride in as dependencies of other packages. Verified via pip metadata 2026-07-11:

| Hidden dep | Arrives via | Who needs it | Evidence |
|---|---|---|---|
| Pillow (PIL) | psd-tools (`pip show psd-tools` → Requires: attrs, numpy, **Pillow**, typing-extensions) | Top-level `from PIL import Image` in `commands/utils.py:11` AND `from PIL.Image import ...` in `commands/diffusion.py:5` | Remove psd-tools and the whole bot fails at import, not just PSD parsing. |
| PyNaCl | PyGithub (`pip show PyGithub` → Requires: pyjwt, **pynacl**, requests, typing-extensions, urllib3) | discord.py voice encryption (voice/music is silent-broken without it) | discord.py itself only requires aiohttp; it does not pull PyNaCl. |

**The rule**: if psd-tools or PyGithub is ever removed from `requirements.txt`, add
`Pillow` and/or `PyNaCl` explicitly in the same commit. The rationale: nothing in this
repo declares these two directly, so today's env only works because of an accident of
the dependency graph — and pip will not warn you when the accident stops happening.

## 4. Mac vs CUDA box differences

The bot runs on BOTH machines at different times (whichever is on hosts it). Same code,
same pickles (synced via the bot's own push/update flow — see
**autocroissant-run-and-operate**). What differs is `config.py`'s `vram_usage` and the
torch install:

| Axis | Mac (`vram_usage = 'mps'`) | CUDA box (`vram_usage = 'low'/'medium'/'high'/'distributed'`) |
|---|---|---|
| torch install | `pip install -r requirements2.txt` as-is (PyPI torch, MPS backend) | Uncomment the `#-i .../nightly/cu124` first line of requirements2.txt first |
| Generation device (`torch.Generator`) | `mps` | `cuda` (`diffusion.py:697`) |
| Diffusion dtype / quantization / offload | float32, unquantized Flux, `pipe.to('mps')` — no offloading | float16, nf4-quantized Flux, offload varies per mode — the canonical per-mode table (dtype, offload, VAE opts, quantization, incl. the UNVERIFIED `high`-mode pipeline-placement caveat) is owned by **autocroissant-ai-boundary** |
| yt-dlp cookies | `cookiesfrombrowser = ('safari', ...)` — enabled by the check `getattr(config, "vram_usage", False) == "mps"` (`music_player.py:37`, used at :55). This is a Mac-detection hack: `vram_usage` doubles as an OS probe. Flag it whenever you touch either music cookies or vram modes. | No browser cookies passed to yt-dlp |

Mode semantics beyond install (defaults, `/set_model` etc.): **autocroissant-config-and-flags**.

## 5. The Cython build (optional)

### Commands

```bash
python3 setup.py build_ext --inplace   # compile the 8 INCLUDE_FILES to .so next to the .py
python3 setup.py clean                 # remove *.c *.cpp *.so *.pyd *.pyc *.html, __pycache__/, build/
```

Notes verified from `setup.py` (2026-07-11):

- The build ALSO runs the artifact cleanup first (`clean_artifacts()` is called at the
  top of `__main__`, setup.py:139), so every build is a from-clean rebuild. Side
  effect: both commands delete ALL `*.html` files anywhere under the repo (rglob).
- Compiled modules (`INCLUDE_FILES`, setup.py:21-30) — 8 of the 10 `commands/` modules:
  analytics, frankenstein, help, psd_analyzer, query_card, management, music_player,
  diffusion.
- Never compiled (`EXCLUDE_FILES`, setup.py:9-18), with the reasons stated in the
  file's comments: `main.py` (entry point — keep as Python), `commands/update_bot.py`
  (uses `execv`/`execl` for restart), `commands/utils.py` (core utilities, decorators,
  dynamic behavior), `config.py`/`global_config.py` (configuration), `setup.py` itself,
  and both `__init__.py` files.

### Compiler directives and their consequences (setup.py:33-45)

This is where the coding-discipline rules for compiled modules come from. Rules
without their incident get ignored, so each one carries its story:

| Directive | Consequence in the 8 compiled modules |
|---|---|
| `wraparound=False` | **NEGATIVE INDEXING IS BANNED.** Compiled code skips Python's negative-index handling; `x[-1]` can read garbage or crash instead of meaning "last element". House idiom: `x[len(x)-1]` — see commit dd800bc (2026-01-30), which changed `bboxes[-1][1].y` to `bboxes[len(bboxes)-1][1].y`, and the live idiom at `music_player.py:118` (`prev_music[len(prev_music)-1]`). The ban has already drawn blood once: the cythonization commit f7c915c mechanically purged a `split("TTSCardMaker")[-1]` and replaced it with a `removeprefix()` that no-ops on absolute paths — the live local-mode stats-corruption bug. Full story: **autocroissant-failure-archaeology**. |
| `boundscheck=False` | Out-of-range indexes are undefined behavior (crash or garbage), not a clean `IndexError`. Check lengths before indexing. |
| `cdivision=True` | C division semantics for negatives: quotient truncates toward zero and `%` takes the dividend's sign. In Python `-7 // 2 == -4` and `-7 % 2 == 1`; in compiled code they become `-3` and `-1`. Any div/mod on possibly-negative ints must be reasoned about. |
| `embedsignature=True` | Function signatures survive into docstrings — `help()` still works on compiled functions. |
| Also set | `language_level='3'`, `initializedcheck=False`, `infer_types=True`, `overflowcheck=False`, `profile=False`, `linetrace=False`. |

### Compilation is optional — and the stale-.so trap

- As of 2026-07-11 there are **no** `.so`/`.c` artifacts in the working tree; the bot
  runs as plain Python. Deploy machines MAY compile for speed; nothing requires it.
- **The trap**: an import like `from commands import psd_analyzer` picks a compiled
  `commands/psd_analyzer.*.so` OVER `commands/psd_analyzer.py`. A stale `.so` from an
  old build silently shadows your edited `.py`. Symptom: your edits do not take
  effect, no error anywhere. Fix: `python3 setup.py clean`. This is the first check in
  any "my change does nothing" debugging session on a machine that has ever compiled.

Check for artifacts (also useful before trusting any test result):

```bash
find . -name "*.so" -not -path "./.git/*"
```

- Artifacts are gitignored (`*.so`, `*.c`, `*.cpp`, `build/`, `__pycache__/`).
  One quirk verified in `.gitignore` (lines 18-20): the entries
  `*.so      # Linux/Mac`, `*.pyd     # Windows`, `*.dll     # Windows (rare)` carry
  trailing text — gitignore has no trailing comments, so those three patterns are
  literal and match nothing. `*.so` is safely covered by the bare `*.so` on line 7,
  but `*.pyd`/`*.dll` are covered by nothing: if a Windows machine ever compiles
  (setup.py has win32 branches), `.pyd` files would show up as untracked and could get
  committed by a careless `git add .`.

## 6. Directory conventions

| Path | Convention |
|---|---|
| `models/` | Diffusion checkpoints (`*.safetensors`) and `models/loras/` for LoRA files. Gitignored; the only tracked files are the stubs `models/README` ("*.safetensors models go in here") and `models/loras/README`. Model files are downloaded separately, never committed (multi-GB). |
| `music/` | yt-dlp downloads land here (`MUSIC_BASE_DIR = "music/"`, music_player.py:30). Ignored wholesale by its own tracked `music/.gitignore` (`*` except itself). |
| `.ssh/` | The bot's git deploy key pair lives here on deploy machines (untracked, gitignored). **NEVER commit it, never print its contents.** |
| repo root exports | `/export_cards` and `/export_rulebook` write `stats.csv` / `stats.txt` / `rules.txt` into the repo root; covered by the `*.csv` / `*.txt` gitignore rules. |
| venvs | `venv/`, `.venv/`, `env/`, `ENV/` are gitignored. (History: the venv was committed on day one and then deleted in favor of requirements.txt — ed18602. Do not repeat 2023.) |

## 7. From-scratch checklists

### (a) Dev-only environment (core bot, no AI)

Run from the directory that should contain the clone:

```bash
git clone https://github.com/bobberlington/AutoCroissant.git
cd AutoCroissant
python3 -m venv .venv               # use python3.11+ if available (guidance); 3.10.20 is verified
source .venv/bin/activate
pip install -r requirements.txt
```

Then:

1. Create `config.py` in the repo root by hand. Minimum content is one line defining
   `TOKEN` (the Discord bot token). Field names and what each does:
   **autocroissant-config-and-flags**. Never commit this file.
2. Smoke test (section c). Done — this env can run everything except AI image
   generation, which cleanly reports itself unavailable (verified clean-venv fact,
   section 2).

### (b) Full host environment (a machine that will actually host the bot)

1. Steps 1-5 of checklist (a).
2. AI stack: `pip install -r requirements2.txt` — on the CUDA box, first uncomment
   the `#-i .../nightly/cu124` line (section 2 caveats).
3. `config.py`: set the machine's `vram_usage` (`'mps'` on the Mac; a CUDA mode on the
   box) plus tokens/fields per **autocroissant-config-and-flags**.
4. ffmpeg: `brew install ffmpeg` (Mac) / `apt install ffmpeg` (Linux). Optional
   insurance: opus (section 1). Verify: `which ffmpeg`.
5. Clone the card source where the bot expects it:
   `git clone https://github.com/MichaelJSr/TTSCardMaker ~/Desktop/TTSCardMaker`
   (path is `LOCAL_DIR_LOC`, global_config.py:4). Remote-only operation works without
   it, and remote mode is currently the safe update mode regardless
   (**autocroissant-failure-archaeology**).
6. Diffusion models: drop `*.safetensors` into `models/` (LoRAs into `models/loras/`)
   and set the `model` config field. No model configured = no pipeline init, by design.
7. Deploy key: place the bot's SSH key pair in `.ssh/` if this machine will run
   `/push` (see **autocroissant-run-and-operate** for the self-update flow).
8. Optional Cython build: `python3 setup.py build_ext --inplace` (section 5). If you
   skip it, plain Python is the current normal.
9. Smoke test (section c), including the AI check — expect `True` from
   `torch_available()` on a full host.

### (c) Verifying an environment

```bash
python3 -c "import main" && echo OK
```

- Run from the repo root (imports are repo-root relative).
- REQUIRES `config.py` to exist — without it this fails immediately, which is itself
  a diagnosis.
- Expected noise: it prints git-token-presence lines like
  `Git token found, API limited to 5000 requests/hour.` (or the
  `No git token in config` variant) from `query_card.py:43-45` / `psd_analyzer.py:206-208`.
  That is normal import-time output, not an error.
- It does NOT start the bot or touch Discord: `client.run(TOKEN)` sits under
  `if __name__ == "__main__":` (verified, bottom of main.py).

Additional probes:

```bash
python3 -c "import main; from commands.diffusion import torch_available; print(torch_available())"   # False = core-only env, True = AI env
which ffmpeg                                          # music prerequisite
python3 -m pip show PyNaCl | head -2                  # voice encryption present (via PyGithub)
find . -name "*.so" -not -path "./.git/*"             # empty = no stale compiled artifacts
```

## When NOT to use this skill

- Starting/operating the bot, the self-update flow, multi-machine handoff, where
  artifacts land at runtime → **autocroissant-run-and-operate**.
- Why AI must stay separate, lazy-torch mechanics, rules for adding AI features →
  **autocroissant-ai-boundary** (this skill only covers installing that stack).
- What each config.py / global_config.py field means, runtime-settable vs
  restart-required → **autocroissant-config-and-flags**.
- Debugging a running bot's misbehavior → **autocroissant-debugging-playbook**
  (except the stale-.so shadow trap, which is owned here).
- The full story of the f7c915c removeprefix regression and other incidents →
  **autocroissant-failure-archaeology**.

## Provenance and maintenance

Verified 2026-07-11 on the owner's Mac (Python 3.10.20, discord.py 2.7.1,
psd-tools 1.14.2, PyGithub 2.9.1, torch 2.9.0, ffmpeg 8.1.1). Line numbers drift;
re-verify volatile facts with:

| Fact | Re-verify with |
|---|---|
| Package lists (11 core / AI set, commented `-i` line) | `cat requirements.txt requirements2.txt` |
| Pillow arrives via psd-tools | `python3 -m pip show psd-tools \| grep Requires` |
| PyNaCl arrives via PyGithub | `python3 -m pip show PyGithub \| grep Requires` |
| discord.py needs only aiohttp | `python3 -m pip show discord.py \| grep Requires` |
| Top-level PIL imports | `grep -n "from PIL" commands/utils.py commands/diffusion.py` |
| davey optional (try/except in discord.py) | `grep -rn -A2 "import davey" $(python3 -c 'import discord,os;print(os.path.dirname(discord.__file__))')/gateway.py` |
| Exclude list + reasons | `grep -n "EXCLUDE_FILES" -A12 setup.py` |
| Include list (8 files) | `grep -n "INCLUDE_FILES = " -A10 setup.py` |
| Compiler directives | `grep -n "COMPILER_DIRECTIVES" -A14 setup.py` |
| No compiled artifacts in tree | `find . -name "*.so" -not -path "./.git/*"` |
| Negative-index idiom commit | `git show dd800bc \| grep -E "^[-+].*bboxes"` |
| Cythonization commit date | `git log --format='%h %ad %s' --date=short -1 f7c915c` |
| TOKEN is the only direct config import | `grep -rn "from config import" main.py commands/` |
| mps float32 / flux unquantized on mps / offload modes | `grep -n 'VRAM_USAGE' commands/diffusion.py` |
| Safari-cookies Mac hack | `grep -n "cookies_from_browser\|cookiesfrombrowser" commands/music_player.py` |
| torch_available / lazy torch | `grep -n "def torch_available\|def get_torch" commands/diffusion.py` |
| Tracked stubs in models/, music/; .ssh untracked | `git ls-files models music .ssh` |
| Token-presence print lines | `grep -rn "git token" -i commands/query_card.py commands/psd_analyzer.py` |
| LOCAL_DIR_LOC card-repo path | `grep -n "LOCAL_DIR_LOC" global_config.py` |
| client.run under __main__ guard | `tail -8 main.py` |
| Python version guidance | `python3 --version` (3.10.20 verified; 3.11+ preferred for new envs — guidance) |

The clean-venv proof (core-only → `import main` OK, `torch_available()` False) was run
2026-07-11; re-running it takes minutes and downloads packages — only repeat it after
changing the requirements split, and do it in a scratch directory, never against the
live env.
