---
name: autocroissant-research-frontier
description: Load this skill when the task is about what AutoCroissant could become rather than what it does today — "what should I work on next", "new feature ideas", "roadmap", "frontier", "AI features", "AI that knows the game", "question answering", "RAG", "embeddings", "an /ask command", "LoRA", "train a model on the cards", "card art generation", "in-style generation", "perfect extraction research", "round-trip verification", or "replace the pickles / better state sync". Contains the owner's two stated ambitions (game-grounded AI, perfect extraction) plus two smaller candidates, each broken into: why the current state falls short, the specific asset this repo already has, the first three concrete file-level steps, and a falsifiable "you have a result when..." milestone. Everything here is CANDIDATE/OPEN — nothing is promised or scheduled. Execution of extraction work belongs to autocroissant-psd-extraction-campaign; the evidence bar belongs to autocroissant-research-methodology.
---

# AutoCroissant Research Frontier

Status date: 2026-07-11. Owner's stated ambitions (same date): **(A) "AI that knows the
game"** — ground AI answers in the card database and rulebook; **(B) "perfect extraction"** —
100% correct PSD parsing with zero per-card exclusion lists; under constraint **(C) "it's a
hobby — keep it fun"** — every step below is deliberately fun-sized and incremental, no heavy
process, no multi-week commitments.

Every item on this page is labeled **CANDIDATE** (an idea with a plausible first step) or
**OPEN** (a genuinely unsolved sub-problem). Nothing here is a commitment. The rule of this
skill: an idea earns space only by arriving with four fields — *why current state falls
short*, *this repo's specific asset*, *first three steps in this repo*, and a *falsifiable
milestone*. "Falsifiable" means a numeric or byte-level check that can come back negative;
"looks right" never counts (see autocroissant-research-methodology for the full evidence bar).

Jargon used below, defined once:

- **BM25** — a classic keyword-ranking formula (term frequency x inverse document frequency
  with length normalization). Implementable in ~50 lines of stdlib Python; zero dependencies.
- **RAG** — retrieval-augmented generation: retrieve the relevant rule/card text first, then
  have a language model compose an answer from it (instead of from its own memory).
- **LoRA** — a small low-rank adapter fine-tuned on top of a frozen diffusion model; the
  standard cheap way to teach Stable Diffusion a specific art style. Ships as one
  `.safetensors` file.
- **Eval set** — a fixed list of questions with known-correct answers, written BEFORE building
  anything, so every approach is scored against the same target.

---

## Item 1 — Game-grounded question answering (CANDIDATE)

Ambition (A). Goal: a friend asks "can I stack two debuffs on one creature?" and gets a
correct answer *with a citation to the rulebook page or card that proves it*.

### Why the current state falls short

- `/query_rulebook` is a case-insensitive **substring** search over the 48 rulebook pages'
  text and returns whole pages (`rulebook_search_engine`, commands/query_card.py:266 —
  literally one `str.contains` call; wrapper at query_card.py:472, registered main.py:285).
  Ask it a question and it matches nothing unless your words appear verbatim.
- `/query_ability` is a hand-rolled DSL (`ability_search_engine`, query_card.py:194): `|` OR,
  `&` AND, `!` NOT, `"exact"` word-boundary, numeric `hp>5`, `type==creature`, `series~kirby`,
  else substring + difflib fuzzy above `match_ratio` (0.6 default). Powerful for people who
  know the DSL; useless for natural questions.
- Neither composes an answer. Both retrieve documents and stop.

### This repo's specific assets (all verified 2026-07-11)

| Asset | Where | Why it matters |
|---|---|---|
| 813 structured cards | `stats.pkl` (765 cards + 48 rulebook pages) | The knowledge base already exists — no scraping project needed |
| Rulebook as plain text | `export_rulebook_to_file` (commands/psd_analyzer.py:1615) writes `rules.txt`; `/export_rulebook` (main.py:382) | 48 pages of game rules, one function call away |
| Cards as CSV | `export_stats_to_file` (psd_analyzer.py:1520) writes `stats.csv` with an aliases column; `/export_cards` (main.py:369) | Structured stats + ability text in one table |
| Type-annotated ability text | `_inject_type_names` (psd_analyzer.py:593) already embeds `[water]`, `[ice]` etc. inline | Icon semantics are already textual — a retriever can match on them |
| Synonym map | `aliases.pkl` — 60 alias→filename entries (e.g. `red_bloon` → `pod01red_bloon.png`) | Free query-expansion vocabulary |
| A live friend group | the bot's actual users | Real questions, real ground truth, real judges — most hobby QA projects never have this |

### First three steps (in this repo)

**Step 1 — Build the corpus OUTSIDE the bot.** Do not touch bot code yet. Either run
`/export_rulebook` and `/export_cards` from Discord when the bot is up (files land in the repo
root, gitignored via `*.txt`/`*.csv`, .gitignore:62-63), or build offline from the pickle.
This exact snippet was dry-run 2026-07-11 and produced 48 pages / 765 cards (expect two
"Git token found" import-chain print lines — that is NOT the bot starting):

```bash
cd /Users/michaelsrouji/Desktop/AutoCroissant   # unpickling CardInfo needs repo cwd + config.py
mkdir -p ~/frontier_corpus
python3 - <<'EOF'
import pickle, os
db = pickle.load(open('stats.pkl', 'rb'))       # dict[name -> CardInfo]; READ ONLY, never write pickles
out = os.path.expanduser('~/frontier_corpus')
with open(f'{out}/rules.txt', 'w') as f:
    for name, card in db.items():
        if 'Rulebook' in card.path:
            f.write(f"{name}\n{card.ability or ''}\n\n")
with open(f'{out}/cards.tsv', 'w') as f:
    f.write("name\ttype\tseries\tstars\thp\tdef\tatk\tspd\tability\n")
    for name, card in db.items():
        if 'Rulebook' in card.path: continue
        d = card.to_dict()
        row = [name] + [str(d.get(k, '')) for k in ('type','series','stars','hp','def','atk','spd')]
        f.write('\t'.join(row) + '\t' + (d.get('ability') or '').replace('\t',' ').replace('\n','\\n') + '\n')
EOF
```

**Step 2 — Create the eval set FIRST (the durable asset; zero dependencies).** Collect 20
real questions from the friend group (scroll the Discord server; ask them), write the
ground-truth answer for each with the rulebook page or card name that proves it. One TSV/MD
file, e.g. `.claude/skills/autocroissant-research-frontier/eval/rules_qa_v1.md`, format:
`question | correct answer | citation (page/card name)`. This file outlives every model,
library, and approach you will ever try; it is the single most valuable artifact of this whole
item. Committing it routes through autocroissant-change-control like any repo change.

**Step 3 — Baseline WITHOUT new heavy dependencies.** BM25 or plain keyword scoring over
`rules.txt` paragraphs + `cards.tsv` rows, in a standalone script under `~/frontier_corpus/`
(stdlib only, or a tiny pure-Python `rank_bm25` inside a throwaway research venv — nothing
enters any requirements file, nothing imports bot modules). Use `aliases.pkl` terms for query
expansion. Score it against the 20 eval questions ("does the top-3 retrieved text contain the
answer?"). Only if retrieval quality is the bottleneck do you graduate to an
embedding/LLM experiment — still offline. A `/ask` slash command happens LAST, only after an
offline winner exists, and is gated like every AI feature (lazy imports, clean degradation —
autocroissant-ai-boundary owns the rules; the add-a-slash-command checklist is in
autocroissant-change-control).

### You have a result when...

**>= 15 of the 20 eval questions are answered correctly WITH a citation to the right rulebook
page or card name — measured against the eval file, not vibed.** Predict your score before
each run (autocroissant-research-methodology). A baseline that scores 8/20 is a real result
too: it is the number every fancier approach must beat.

### Fence

**No LLM or embedding dependency may ever enter `requirements.txt` (core).** As of 2026-07-11
it holds exactly 11 packages (cython, davey, discord.py, GitPython, numpy, opencv-python,
pandas, psd-tools, PyGithub, requests, yt-dlp) after today's intentional (uncommitted) reorg;
the torch stack lives in `requirements2.txt`. Anything heavy goes to requirements2.txt at
most, behind the lazy-import pattern — boundary doctrine in autocroissant-ai-boundary.

---

## Item 2 — In-style card-art generation (CANDIDATE)

Ambition (A), art flavor. Goal: `/ai` output that looks like an Impossibility Simulator card
instead of generic Stable Diffusion output.

### Why the current state falls short

`/ai` (main.py:485) drives stock SD 1.5 / SDXL / Flux pipelines (commands/diffusion.py).
Nothing anywhere knows the game's art conventions — the frame layout, the flat-color icon
style, the per-series looks. Every generation starts from zero style knowledge.

### This repo's specific assets (verified 2026-07-11)

- **1003 card PNGs** in `~/Desktop/TTSCardMaker` (all sampled files are 1200x2400 full-card
  renders). Creature PNGs are organized by series folder — real counts:

  | Series (Creatures/) | PNGs | | Series | PNGs |
  |---|---|---|---|---|
  | Kirby | 58 | | World Of Warcraft | 13 |
  | Other | 49 | | Fnaf | 12 |
  | Mario | 28 | | Undertale | 10 |
  | Hollow Knight | 24 | | Isaac | 10 |
  | Pandas | 20 | | (14 more series) | 1–9 each |
  | Omori | 20 | | | |

  20–60 images is squarely LoRA-feasible; Kirby (58) is the obvious first target. Isaac (10)
  is too thin to train alone.
- **The LoRA plumbing already works.** `LORAS_FOLDER = "./models/loras/"` (diffusion.py:22-23);
  pipeline init calls `load_lora_weights` when a LoRA is set (diffusion.py:230-232); `/set_lora`
  (main.py:561; `set_lora` diffusion.py:488) lists every `*.safetensors` in the folder, swaps
  the global, and clears pipelines for lazy re-init. Dropping a file into `models/loras/` and
  running `/set_lora` is the ENTIRE integration — zero new code. (Runtime `/set_lora` does not
  persist across restart; the persistent field is `lora` in config.py —
  autocroissant-config-and-flags.)

### First three steps (in this repo)

**Step 1 — Inventory the training pool.** Exact command (produced the table above):

```bash
find ~/Desktop/TTSCardMaker/Creatures -type d -mindepth 1 -maxdepth 1 | \
  while read d; do echo "$(find "$d" -name '*.png' | wc -l) $(basename "$d")"; done | sort -rn
```

Decide whole-card vs cropped-art training: whole 1200x2400 cards teach frame + text layout
too (maybe desirable for "generate a whole card", noisy for "generate card art"). Note the
choice; it is the first experimental variable.

**Step 2 — CANDIDATE: train a LoRA on ONE series' art, OFFLINE and OUTSIDE this repo.**
Training is not this repo's job — use any standard LoRA trainer on another machine/venv, on
the Kirby set first. This repo's role is only: source images (`Creatures/Kirby/**/*.png`) and
the drop-in point (`models/loras/<name>.safetensors`, then `/set_lora` — never committed;
models/ is on the never-commit list, autocroissant-change-control).

**Step 3 — Blind A/B with the friend group.** Show 5 real cards + 5 generated, shuffled;
each friend labels each image real/generated. Pre-register the pass bar before showing anyone
anything (autocroissant-research-methodology).

### You have a result when...

**>= 40% of generated cards are mistaken for real in the blind test.** Honesty note on the
bar: these friends know every card by sight — their expected false-"real" rate on obvious
SD output is near 0%, so 40% is a genuinely strong signal, not near-chance noise. A 5%
result is also a result: it retires the naive approach with evidence
(autocroissant-failure-archaeology).

### Fence

LoRA *training* is outside-repo work. Inside the repo, only: image inventory, the
`models/loras/` drop-in, `/set_lora`, and the eval. No trainer dependencies enter any
requirements file.

---

## Item 3 — Perfect-extraction research residue (OPEN)

Ambition (B). **Execution is OWNED by autocroissant-psd-extraction-campaign** — baselines,
the ranked solution menu, gates, promotion. Do not run extraction work from this page. What
lives HERE is only the genuinely unsolved research residue that survives even if the campaign
executes perfectly.

### Residue 3a — Line-position estimation for bbox-anchored injection (OPEN)

**Why open:** today's type-icon injection (`_inject_type_names`, psd_analyzer.py:593-637)
assigns icons to `\s{3,}` text gaps by global reading order and never checks whether an
icon's y-coordinate matches the line it lands in. The fix everyone wants — anchor each icon
to its text LINE by (x, y) — hits a wall: **psd-tools exposes only the whole text layer's
bbox (`layer.bbox`), not per-line boxes.** The campaign's Option 3 has a first approximation
(uniform `line_height = (y2-y1)/num_lines` with clamping, worked Lich King numbers) and four
labeled hard sub-problems; the hardest is soft-wrap — `num_lines` counted from `\n` in the
extracted text diverges from RENDERED lines when text wraps.

**Asset:** the diagnostics scripts (`dump_psd_layers.py`, `gap_trace.py`, `parse_one.py` in
`.claude/skills/autocroissant-diagnostics-and-tooling/scripts/`) print exactly the icon bboxes
and layer bboxes needed to hand-label ground truth in minutes per card.

**First three steps:** (1) select ~10 multi-line cards whose ability text contains `[type]`
tokens beyond the first line (scan `stats.pkl` abilities for `\n` + `[`); (2) hand-label each
below-midline icon's TRUE line index using `dump_psd_layers.py` (icon y) plus the rendered
PNG — include at least one soft-wrapped card if one exists; (3) score the campaign's
uniform-line-height formula against the labels, per icon.

**You have a result when:** the line model assigns **>= 95% of hand-labeled icons to the
correct line, including the soft-wrap case** — or the failure mode is characterized precisely
enough to kill the uniform model (an equally valid result; archive it with the numbers).

### Residue 3b — Round-trip verification (CANDIDATE, feasibility unknown)

**Idea:** re-render the EXTRACTED ability text (PIL `ImageDraw`, approximated font) and
image-diff it against the ability region cropped from the card's real PNG. Extraction errors
(dropped words, wrong injection position) should spike the diff. This would be a
per-card correctness oracle needing no exclusion lists — exactly ambition (B).

**Unknowns stated plainly:** the game font is unidentified; `[type]` tokens replace icon
images, so the render must mask or substitute them; kerning/leading mismatch may swamp the
signal. Feasibility is genuinely unknown — that is why the first step is one card.

**First three steps:** (1) prototype on The Freezer only (golden card with byte-exact expected
text — autocroissant-validation-and-qa); crop its ability region from
`Field/1 Stars/The_Freezer.png`; (2) render the extracted text with a guessed font and compute
a normalized pixel diff; (3) seed a corruption (delete one word from the extracted text),
re-render, re-diff.

**You have a result when:** on that single card, **the seeded corruption produces a clearly
larger diff than the correct extraction** (state the threshold before running). If font noise
swamps the seeded-error signal, retire the idea to autocroissant-failure-archaeology with the
two diff numbers as evidence.

---

## Item 4 — Cross-machine state without git-pickle friction (CANDIDATE — "do nothing" currently wins)

Small item; the hobby constraint bites hardest here.

### Why the current state (mostly) does NOT fall short

Pickles-in-git WORKS: `/push` / `/pull` / `/update` move `stats.pkl` / `old_stats.pkl` /
`aliases.pkl` between the Mac and the CUDA box and they are the ONLY synced state
(autocroissant-run-and-operate owns the flow). The real pain points, for the record:

1. **Schema coupling** — unpickling requires this repo on `sys.path` with config.py present,
   because `CardInfo`/`CardStats` are dataclasses in commands/psd_analyzer.py:123/94; the
   366c8d9 "Big massive refactor" already broke the pre-Oct-2025 plain-dict format once.
2. **Commit noise** — 23 of 192 commits mention PICKLE (measured 2026-07-11).
3. **Invisible binary regressions** — old_stats.pkl ballooned 3196→12041 bytes once
   (cca0aaf, reverted eb9aa84); a guard landed later (e7befd5) but binary diffs mean a second
   balloon is invisible in review unless `diff_stats.py` is run
   (autocroissant-change-control owns that gate).

### Asset

`CardInfo.to_dict`/`from_dict` (psd_analyzer.py:138/170) already exist — a schema-decoupled
JSON snapshot is nearly free; `StatsDatabase.save` (psd_analyzer.py:210) is the single write
point; `diff_stats.py` is a ready-made zero-loss checker.

### First three steps (in this repo)

1. **Measure the pain before solving it:** `git log --oneline | grep -ci pickle` (23 today)
   and `git log --oneline -- stats.pkl old_stats.pkl | wc -l`. If the numbers still feel
   tolerable, STOP HERE — that is the intended outcome.
2. CANDIDATE: a sidecar `stats.json` written by `StatsDatabase.save` via `to_dict` — human-
   diffable commits and format-break insurance. Cost: dual-write drift. Route through
   autocroissant-change-control; do not build speculatively.
3. CANDIDATE (only if 2 proves value): replace balloon-prone `old_stats.pkl` with append-only
   JSONL. Nothing beyond that — no databases, no cloud, no sync services.

### You have a result when... (the bar any replacement must beat)

**Zero-loss handoff in BOTH directions (Mac → CUDA box → Mac, `diff_stats.py` reporting 0/0/0
after the round trip) with NO new infrastructure to babysit.** Pickles-in-git already meets
this bar. **"Do nothing" is the current best option** — a replacement that merely ties has
lost, because it costs a migration.

---

## How to add a frontier item

An item enters this page only with all four fields filled in:

1. Why the current state falls short (with file:line evidence);
2. This repo's SPECIFIC asset (a thing that exists, verified, not "we could build");
3. First three concrete steps IN THIS REPO (file-level, command-level, each fun-sized);
4. A falsifiable "you have a result when..." milestone (a number or byte-check that can fail).

Label it CANDIDATE or OPEN. When an item is tried and fails, do not delete it — move it to
autocroissant-failure-archaeology WITH the numbers that killed it (the negative result is the
payment for the time spent). When an item ships, it stops being frontier: its facts move to
the owning skill (commands → run-and-operate, config → config-and-flags, etc.).

## When NOT to use this skill

- **Executing extraction work** (baselines, gap_trace runs, fix promotion, the ranked
  solution menu) → **autocroissant-psd-extraction-campaign**. Item 3 here is only the
  research residue.
- **Methodology / evidence-bar questions** ("is this proven?", hypothesis discipline,
  adversarial refutation, idea lifecycle) → **autocroissant-research-methodology**.
- Adding an AI feature or dependency TODAY → **autocroissant-ai-boundary** (rules) +
  **autocroissant-change-control** (gates).
- Understanding what the bot currently does → **autocroissant-architecture-contract** /
  **impossibility-cards-reference**.
- Running or operating the bot → **autocroissant-run-and-operate**.

## Provenance and maintenance

Written 2026-07-11 against working tree @ 284d13c ("PICKLE"). Every claim was verified that
day by reading the cited code, running the read-only diagnostics, or running the shown
commands. Line numbers drift — re-find with the greps below before trusting them.

| Volatile fact | Re-verify with (cwd = repo root) |
|---|---|
| 813 cards / 48 rulebook pages / type distribution | `python3 .claude/skills/autocroissant-diagnostics-and-tooling/scripts/inspect_pickle.py` |
| Rulebook search is substring | `grep -n "str.contains" commands/query_card.py` (inside `rulebook_search_engine`; def: `grep -n "def rulebook_search_engine" commands/query_card.py`) |
| Ability DSL location | `grep -n "def ability_search_engine" commands/query_card.py` |
| Export functions / filenames | `grep -n "def export_" commands/psd_analyzer.py; grep -n "EXPORTED_" commands/psd_analyzer.py` |
| /export, /query, /ai, /set_lora registrations | `grep -n 'name="' main.py` (lists every slash-command registration; scan for export_cards, export_rulebook, query_rulebook, ai, set_lora) |
| Exports gitignored | `grep -n csv .gitignore; grep -n txt .gitignore` |
| aliases.pkl entry count (60) | `python3 -c "import pickle; print(len(pickle.load(open('aliases.pkl','rb'))))"` |
| Total PNGs (1003) | `python3 -c "import glob,os; print(len(glob.glob(os.path.expanduser('~/Desktop/TTSCardMaker/**/*.png'), recursive=True)))"` |
| Per-series counts (Kirby 58, Isaac 10, ...) | the `find ... Creatures -type d` loop in Item 2 Step 1 |
| PNG dimensions 1200x2400 | `find ~/Desktop/TTSCardMaker/Creatures/Kirby -name '*.png' -exec python3 -c "import sys; from PIL import Image; print(Image.open(sys.argv[1]).size)" {} \; -quit` |
| LoRA folder & loading | `grep -n LORAS_FOLDER commands/diffusion.py; grep -n "def set_lora" commands/diffusion.py` |
| requirements.txt is core-only (11 pkgs) | `cat requirements.txt` (reorg of 2026-07-11 is intentionally uncommitted; working tree is truth) |
| Injection is gap-ordinal, not line-anchored | `grep -n _gap_pattern commands/psd_analyzer.py; grep -n "def _inject_type_names" commands/psd_analyzer.py` |
| CardInfo/StatsDatabase coupling points | `grep -n "class CardInfo" commands/psd_analyzer.py; grep -n "class StatsDatabase" commands/psd_analyzer.py; grep -n "def to_dict" commands/psd_analyzer.py` |
| PICKLE commit noise (23/192) | `git rev-list --count -i --grep=pickle HEAD` (23) and `git rev-list --count HEAD` (192) |
| Balloon incident hashes | `git log --oneline -2 eb9aa84` (shows the revert atop cca0aaf) or see autocroissant-failure-archaeology |

Maintenance: when a step is executed, update its item in place (CANDIDATE → tried, with the
measured number) or move it out per "How to add a frontier item". Re-date the header when you
touch anything. Keep the eval file (`eval/rules_qa_v1.md`, once it exists) append-only —
changing old questions invalidates every historical score.
