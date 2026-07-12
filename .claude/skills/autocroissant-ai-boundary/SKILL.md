---
name: autocroissant-ai-boundary
description: >-
  Load this skill whenever a task touches the AI image-generation (diffusion) side of AutoCroissant
  or its isolation from the core bot: enabling or disabling AI on a machine; anything mentioning
  torch, PyTorch, diffusers, transformers, accelerate, bitsandbytes, peft, CUDA, GPU, VRAM, MPS,
  "PyTorch is not installed.", requirements2.txt or the core/AI requirements split; the /ai,
  /ai_queue, /set_model, /set_scheduler, /set_lora, /set_device commands; vram_usage modes
  (low/medium/high/mps/distributed); Flux vs SDXL vs SD1.5 model routing; adding ANY new AI/ML
  feature or heavy dependency to the bot; or moving the bot between the Mac and the CUDA box. It
  contains the owner's toggleable-AI doctrine, the three off-switches and their verified behavior,
  the lazy-import architecture (commit bf9478e) and the hard rules for new code, the VRAM mode
  table, enable/disable runbooks, and the known boundary leaks. It is the rulebook for the
  boundary, not a generation-debugging guide.
---

# AutoCroissant AI boundary: diffusion is an experiment, not a dependency

Date-stamped 2026-07-11. Line numbers cited below are correct as of this date; every volatile fact
has a re-verification one-liner in "Provenance and maintenance" at the bottom.

## The doctrine

Stated by the owner on 2026-07-11:

> The AI (diffusion) capability is an EXPERIMENT. It must stay separate and toggleable. The bot
> must run perfectly without it, with none of its overhead. There are future plans for it.

You are reading the skill that owns this doctrine and its mechanics. Treat it as a hard constraint
on every change you make to this repo.

### Why the doctrine exists

- **Multi-machine reality.** The bot runs on the owner's Mac (`vram_usage='mps'`, Apple-GPU
  backend) and on a CUDA box (NVIDIA GPU), at different times — whichever machine is on hosts it.
  A future host may have no usable GPU at all. Core features (card queries, music, reminders,
  self-update) must be identical everywhere; only the AI layer varies per machine.
- **The AI stack is heavy and fragile.** torch + diffusers + transformers means gigabytes of
  wheels, seconds of import time, hundreds of MB of RAM residency, and version churn. The two
  deploy targets cannot even install the same way: the CUDA box wants nightly cu124 wheels
  (`#-i https://download.pytorch.org/whl/nightly/cu124`, requirements2.txt line 1, commented out
  in ca0b821 on 2025-12-02), while the Mac wants plain PyPI wheels with MPS support. A stack that
  cannot be pinned uniformly must not be a precondition for the bot starting.
- **History says so.** The split has been coming since 2024: requirements2.txt was born 2024-07-13
  (fffc1fd) as a CUDA-wheel pin file (`torchvision==0.13.1+cu113` era); VRAM-reduction work
  followed (b22dd42 2024-07-18, 3f9a96d 2024-08-18). Dead-end variants (requirements-linux*.txt)
  are chronicled in `autocroissant-failure-archaeology`.

## The three switches (verified 2026-07-11)

Three independent levels, from hardest off to lightest:

| # | Switch | Where | Off state | What it buys |
|---|---|---|---|---|
| 1 | Dependency-level (hard off) | Install only requirements.txt, not requirements2.txt | torch not importable | Zero AI overhead of any kind. Only fully polite state. |
| 2 | Config-level (soft off) | `model = ''` in config.py (or field absent) | Pipeline init no-ops | No weights, no diffusers import, no VRAM. torch itself still imports at startup if installed (see ladder below). |
| 3 | Runtime | `/set_model`, `/set_scheduler`, `/set_lora`, `/set_device` | n/a — switching only | Any change clears all pipelines (frees VRAM/RAM) until the next `/ai`. **There is no runtime full-off**: `/set_model` validates against available models and cannot set `''` (diffusion.py:476-481). Disabling requires switch 1 or 2 plus restart. |

### Switch 1: the requirements split

As of 2026-07-11 (verified by `git diff HEAD`, and note: **this split is an uncommitted
working-tree change today** — HEAD 284d13c still has the AI packages mixed into requirements.txt):

- `requirements.txt` (core, 11 packages): cython, davey, discord.py, GitPython, numpy,
  opencv-python, pandas, psd-tools, PyGithub, requests, yt-dlp.
- `requirements2.txt` (AI stack, 10 packages behind the commented cu124 index line): torch,
  torchvision, torchaudio, accelerate, bitsandbytes, diffusers, peft, protobuf, sentencepiece,
  transformers.

torchvision and torchaudio are companion installs only — **no bot code imports them** (verified:
the only `import torch*` statement in the repo is `import torch` inside `get_torch()`,
diffusion.py:140).

**Verified proof of the hard-off state** (run by the lead engineer 2026-07-11 — do NOT casually
re-run; it takes minutes): clean venv + `pip install -r requirements.txt` only, repo on sys.path
(config.py must still exist — the import chain requires it), then `import main` succeeds and
`commands.diffusion.torch_available()` returns False. The bot starts and every core feature works.

Committing the split is a change-control matter — route through `autocroissant-change-control`.

### Switch 2: `model = ''`

`MODEL = getattr(config, "model", "")` (diffusion.py:24). `init_pipeline()` (diffusion.py:193)
guards in this order:

1. `torch = get_torch()` (line 195) — if None, prints `ERROR: Cannot initialize pipeline - Torch
   is not installed` (line 197) and returns.
2. `if not MODEL:` — prints `INFO: No model configured for initialization` (line 201) and returns.

So with `model = ''`, nothing loads: no diffusers import (the `from diffusers import ...` at line
204 sits after both guards), no weights, no VRAM. But note the guard ORDER: `get_torch()` runs
first, so on a machine where torch IS installed, soft-off still imports the torch module at
startup. Only switch 1 removes all overhead.

### The startup-overhead ladder

`on_ready` queues `init_pipeline` unconditionally (main.py:119, via `queue_command`; it runs about
a second later on the command-queue loop). What that costs per state:

| State | Startup overhead | `/ai` behavior |
|---|---|---|
| AI deps absent (switch 1) | None. Console prints the Torch-is-not-installed ERROR line and moves on. | Polite Discord reply: `PyTorch is not installed.` (diffusion.py:659-661) |
| torch installed, `model=''` (switch 2) | torch module import only (seconds, RAM residency; no VRAM, no weights, no diffusers) | **TRAP — see "Known weak points"**: crashes and wedges the AI queue |
| torch installed, model set | Full pipeline init at startup: model load, offload setup, VRAM/RAM residency | Generates |

## The lazy-import architecture (bf9478e)

Commit bf9478e, "lazily import torch and diffusers", 2026-01-26 (the shared timeline's 01-18 date
is off; git is ground truth). Before it, diffusion.py had **top-level** `from diffusers import
(...)`, `from diffusers.utils import load_image`, `from transformers import ...`, and a top-level
`try: import torch` flag. Because main.py imports `commands.diffusion` unconditionally
(main.py:17-25), every bot start on an AI-capable machine paid the full torch+diffusers import
cost even if nobody ever ran `/ai`. bf9478e is the incident behind every rule in this section.

The mechanism today (all in commands/diffusion.py):

- **Memoized torch handle**: `_torch = None` (line 134); `get_torch()` (lines 136-144) imports
  torch on first call, caches it in `_torch`, returns None on ImportError;
  `torch_available() -> bool` (lines 146-147) wraps it.
- **diffusers/transformers imports live INSIDE functions**, after the guards: init_pipeline
  (line 204), _init_flux_pipeline (lines 262-263, the only transformers import), _init_sdxl_pipeline
  (line 330), _init_sd15_pipeline (line 345), _configure_scheduler (line 359), diffusion (line 663).
- **torchvision/torchaudio: never imported** anywhere in bot code.

### The rule for new code (non-negotiable)

1. **Never add a top-level import** of torch, torchvision, torchaudio, diffusers, transformers,
   accelerate, bitsandbytes, or peft to ANY module that main.py imports (which is effectively
   every `commands/` module). Import inside the function that needs it, after a
   `torch_available()` guard, or extend the `get_torch()` memoization pattern.
2. **Never make a core feature call an AI code path.** Core = everything installed by
   requirements.txt: queries, PSD parsing, music, reminders, management, self-update.
3. **Every AI entry point must check `torch_available()`** and degrade to a short Discord message
   (the house string is `"PyTorch is not installed."`), never a traceback.
4. **New AI dependencies go in requirements2.txt** (or a new optional requirements file for a
   distinct experiment), never requirements.txt.

### What counts as "overhead" to keep out of core

Import time; RAM residency of the torch/diffusers modules; VRAM residency; pipeline init time;
heavy packages in requirements.txt (install time, disk, version-resolution risk on machines that
will never run AI); any background GPU state. Also config coupling: core behavior must not depend
on AI config fields (see "Known weak points").

**Already well-behaved** (verified): `init_pipeline` is queued at startup (main.py:119) but no-ops
without torch or without a model (diffusion.py:193-202), so the unconditional queueing is
harmless. `/ai_queue` (`get_qsize`) touches no torch state and works in every configuration.

## VRAM_USAGE modes and model routing

`VRAM_USAGE = getattr(config, "vram_usage", "low")` (diffusion.py:28). Glossary: VRAM = GPU
memory; CPU offload = keeping weights in system RAM and shuttling them to the GPU per-module
(sequential = per-submodule, smallest VRAM, slowest) or per-model (model offload, middle ground);
VAE slicing/tiling = decoding the image in chunks to cap VRAM spikes; bitsandbytes nf4 = 4-bit
quantized weights, and bitsandbytes is effectively CUDA-only (no MPS support) — which is exactly
why the mps path skips it.

Verified against `init_pipeline` (dtype line 210, device_map line 212), `_apply_memory_optimizations`
(lines 375-390), and `_init_flux_pipeline` (lines 260-325). **This is the canonical VRAM mode
table** — autocroissant-config-and-flags and autocroissant-build-and-env carry one-line summaries
that defer here:

| vram_usage | dtype | Offload | VAE slicing+tiling | Notes |
|---|---|---|---|---|
| `low` (default) | float16 | sequential CPU offload (`gpu_id=DEVICE_NO`) | yes | Smallest VRAM, slowest |
| `medium` | float16 | model CPU offload (`gpu_id=DEVICE_NO`) | yes | |
| `high` | float16 | none | **no** | `_apply_memory_optimizations` applies NOTHING in this mode — the intent is everything-resident-on-GPU (fastest, most VRAM), but **UNVERIFIED where the pipeline actually resides at generation time**: no code path explicitly moves the pipe to CUDA (the only pipe `.to(...)` is the mps one at line 378; line 697's `device` only seeds the `torch.Generator`). Verify on the CUDA box before relying on `high` |
| `mps` | **float32** | none — `pipe.to(device='mps')` then early return (line 377-379) | no | Mac path. Flux loads full-precision from HF with NO bitsandbytes quantization (lines 266-267) |
| `distributed` | float16 | none | yes | `device_map='balanced'` only when >1 CUDA GPU (line 212), and only the Flux init receives device_map (line 216) — SDXL/SD1.5 ignore it |
| anything else | float16 | none | yes | Falls through the low/medium checks; behaves like distributed on one GPU |

Two honest wrinkles (as of 2026-07-11): the computed dtype is passed to the SDXL/SD1.5 initializers
but not used there — only the Flux path consumes it (lines 328-354); and `vram_usage` doubles as
the device selector at generation time (`device = "cuda" if VRAM_USAGE != "mps" else "mps"`,
line 697), so `'mps'` is the bot's only "I am on the Mac" signal — the root of the music
side-channel below.

### Model routing by substring (naming-convention trap)

`init_pipeline` routes on the lowercased `model` value (diffusion.py:215-222), checked in order:

| Test | Pipeline | Weights source |
|---|---|---|
| `"flux"` in name | FluxPipeline | Hugging Face download `black-forest-labs/FLUX.1-dev` (needs `HF_TOKEN` in config.py) |
| else `"xl"` in name | StableDiffusionXLPipeline | `models/<name>` via `from_single_file` |
| else | StableDiffusionPipeline (SD1.5) | `models/<name>` via `from_single_file` |

Traps: a local file named `my_xl_model.safetensors` routes to SDXL purely by name; a local file
with "flux" anywhere in its name is never opened — the bot goes to Hugging Face instead; "flux"
wins over "xl". Name local checkpoint files accordingly. On this Mac today, `models/` holds
AnythingXL_xl.safetensors and dreamshaperXL_v21TurboDPMSDE.safetensors (both route SDXL) and
rpg_v5.safetensors (routes SD1.5). `models/` and `models/loras/` are gitignored — weights are
per-machine and never committed.

Scheduler names: `POSSIBLE_SCHEDULERS = ["dpm++ sde", "dpm++ sde karras", "euler a"]`
(diffusion.py:31); the karras variant is skipped for Flux (line 369).

## Runbook: enable AI on a machine

1. Install the AI stack into the bot's venv: `pip install -r requirements2.txt`. On the CUDA box
   first uncomment requirements2.txt line 1 (`#-i https://download.pytorch.org/whl/nightly/cu124`);
   on the Mac leave it commented (plain PyPI wheels include MPS). Install mechanics beyond this
   (venv creation, ffmpeg, Cython) live in `autocroissant-build-and-env`.
2. Provide weights: drop a `.safetensors` checkpoint into `models/` (create the folder if absent;
   it is gitignored), or plan to use `model = 'flux'` which downloads from Hugging Face and
   requires the `HF_TOKEN` config field. Mind the substring routing above when naming files.
3. Edit config.py (fields by name only — values are secret; the full catalog is in
   `autocroissant-config-and-flags`): set `model`, `scheduler_name` (one of the three above),
   `vram_usage` (table above; `'mps'` on the Mac, `'low'` is the safe CUDA default), `device_no`
   (CUDA GPU index), optionally `lora`.
4. Restart the bot. These fields are read once at import into module globals (diffusion.py:24-29);
   runtime `/set_*` changes do not persist across restarts (`autocroissant-config-and-flags`).
5. Verify: `/set_model` with no argument replies with the current model and the available list
   (diffusion.py:476-481). Then a small run: `/ai prompt:test steps:4` — expect the console lines
   `Initializing pipeline with model: ...` then `Pipeline initialization complete`, a progress
   message with previews, and a final image. Generation problems from here on are
   `autocroissant-debugging-playbook` territory.

## Runbook: disable AI on a machine

- **Soft off**: set `model = ''` in config.py and restart. Expect the console line
  `INFO: No model configured for initialization` at startup. No weights or diffusers load. Caveat:
  if torch is installed, torch still imports at startup, and `/ai` is NOT polite in this state
  (see weak point 2) — soft off silences the loading, not the command.
- **Hard off**: run the bot in an env without requirements2.txt — cleanest is a fresh venv with
  only `pip install -r requirements.txt` (this exact path is the verified clean-venv proof above);
  uninstalling the ten requirements2 packages from an existing env works too. `/ai` then degrades
  to `PyTorch is not installed.` and startup prints the Torch ERROR line and continues. Both
  outcomes verified 2026-07-11.

## Degradation matrix and known weak points

Politeness with torch ABSENT, per entry point (as of 2026-07-11):

| Entry point | Behavior | Where |
|---|---|---|
| startup `init_pipeline` | Console ERROR line, bot continues | diffusion.py:195-198 |
| `/ai` | Reply `PyTorch is not installed.` | diffusion.py:659-661 |
| `/ai_queue` | Works normally | diffusion.py:170-187 |
| `/set_device` | Reply `Torch is not available.` | diffusion.py:439-441 |
| `/set_model`, `/set_scheduler`, `/set_lora` with no/invalid arg | Info reply, safe | early returns before mutation |
| `/set_model flux` (or any valid arg to those three) | **HOLE**: raises AttributeError | see weak point 1 |

Known weak points — all traced in code on 2026-07-11, not executed (hard rule: never initialize
pipelines); fixes are CANDIDATES to route through `autocroissant-change-control`:

1. **`_clear_pipelines` dereferences `_torch` without a guard** (diffusion.py:515-521, the
   `_torch.cuda.is_available()` at line 520). On a torch-less machine `_torch` stays None forever,
   and the mutation paths of `/set_model` / `/set_scheduler` / `/set_lora` call `_clear_pipelines()`
   — `/set_model flux` (always a valid choice, diffusion.py:469) would crash with AttributeError
   instead of a message. Candidate fix: `if _torch is not None and _torch.cuda.is_available():`.
2. **The `in_progress` wedge**: `diffusion()` sets `in_progress = True` (line 681) with no
   try/finally. If the pipeline could not init — torch installed but `model=''`, a bad model file,
   or an init exception (init_pipeline catches its own errors and leaves `txt2img_pipe` None,
   lines 250-253) — the call reaches `txt2img_pipe(...)` (line 763) as None, raises TypeError, and
   `in_progress` stays True. Every later `/ai` then replies "Request queued." forever;
   `_clear_pipelines` does not reset `in_progress`; only a restart clears it. Candidate fix: after
   the init attempt, `if not txt2img_pipe: queue_message(...); return`, plus try/finally around
   generation.
3. **The music side-channel (doctrine violation in spirit)**: music_player.py:37 sets
   `cookies_from_browser = getattr(config, "vram_usage", False) == "mps"` and line 55 feeds
   yt-dlp `cookiesfrombrowser=("safari", None, None, None)` when true. An AI tuning field controls
   a CORE feature (yt-dlp's Safari-cookie use for YouTube's bot checks), because `vram_usage=='mps'`
   is the only existing Mac signal. Consequence: change `vram_usage` for AI reasons, or run a
   core-only config without the field, and music download behavior silently changes. Verified:
   these are the only two `vram_usage` references outside diffusion.py. Candidate fix: a dedicated
   config field (e.g. a boolean for browser-cookie use) — field-adding checklist in
   `autocroissant-config-and-flags`, change itself through `autocroissant-change-control`.

## Future AI experiments live behind this same boundary

The owner has plans for the AI side — the frontier ambitions ("AI that knows the game": grounding
a model in the card database and rulebook exports) are owned by `autocroissant-research-frontier`;
go there for the ideas. What THIS skill dictates about any of them: every new AI/ML capability —
an LLM client, embeddings, RAG over stats.csv/rules.txt, anything with heavy deps — must replicate
the diffusion pattern:

- lazy, memoized imports on the `get_torch()` model; zero import-time cost to anything main.py imports;
- a `*_available()` capability check with a one-line polite degradation message;
- dependencies in requirements2.txt or a NEW optional requirements file — never requirements.txt;
- no core feature ever conditioned on the experiment's config fields (learn from weak point 3);
- the bot must still start and pass the clean-venv test with the experiment absent.

## When NOT to use this skill

- Generation is enabled but broken/slow/OOM/stalling → `autocroissant-debugging-playbook`.
- venv/pip/ffmpeg/Cython install mechanics and platform setup → `autocroissant-build-and-env`.
- The full config-field and constants catalog, or which settings persist across restart →
  `autocroissant-config-and-flags`.
- The AI research ideas themselves → `autocroissant-research-frontier`.
- Committing the requirements split, applying any candidate fix above, or any other repo change →
  `autocroissant-change-control`.
- Queue/dispatch architecture that AI commands ride on → `autocroissant-architecture-contract`.

## Provenance and maintenance

All facts above verified 2026-07-11 against the working tree. Line numbers drift; re-verify with
these before relying on any cited number:

| Fact | Re-verification one-liner | Expected (2026-07-11) |
|---|---|---|
| Only lazy torch import | `grep -n "import torch" commands/*.py` | one hit: diffusion.py:140 (inside get_torch; also proves no torchvision/torchaudio imports) |
| diffusers imports all inside functions | `grep -rn "from diffusers" --include="*.py" .` | 6 hits, all in commands/diffusion.py: 204, 262, 330, 345, 359, 663 |
| Only transformers import | `grep -rn "from transformers" --include="*.py" .` | diffusion.py:263 |
| cu124 index commented | `head -1 requirements2.txt` | `#-i https://download.pytorch.org/whl/nightly/cu124` |
| Split still uncommitted? | `git status --short -- requirements.txt requirements2.txt` | ` M` on both while uncommitted; empty once committed (then re-date this skill) |
| Split contents | `git diff HEAD -- requirements.txt requirements2.txt` | 7 AI packages moved out of core |
| Startup init queueing | `grep -n "queue_command(init_pipeline)" main.py` | main.py:119 |
| Soft-off message | `grep -n "No model configured" commands/diffusion.py` | line 201 |
| Polite /ai message | `grep -n "PyTorch is not installed" commands/diffusion.py` | line 660 |
| Lazy-import commit | `git show --stat bf9478e` | 2026-01-26, commands/diffusion.py +54/-49 (analytics.py hunk is an unrelated unused-import removal) |
| VRAM mode branches | `grep -n "VRAM_USAGE" commands/diffusion.py` | lines 28, 210, 212, 266, 377, 382, 387, 389, 697 |
| Model routing | `grep -n 'in MODEL.lower()' commands/diffusion.py` | flux/xl checks at 215/218 (plus RGB/scheduler sites) |
| Music side-channel present | `grep -rn "vram_usage" --include="*.py" . \| grep -v diffusion` | music_player.py:37 only; empty means weak point 3 was fixed — update this skill |
| Local weights inventory | `ls models/` | machine-specific; gitignored |
| Hard-off proof | clean venv + `pip install -r requirements.txt`, then `python3 -c "import main; from commands.diffusion import torch_available; print(torch_available())"` from repo root | `False`, import succeeds. EXPENSIVE (minutes) — rerun only if the boundary is in doubt, never on the live env |
