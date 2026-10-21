# AI-generated cards campaign (2026-10-20)

**Status:** designed 2026-10-20, not scheduled; plan item 18 (`docs/NEXT_SESSION_PLAN.md`, "Ideas 5-23"). Rule citations are rulebook pages as
`.claude/skills/impossibility-simulator-rules/SKILL.md` names them; card numbers are `~/Desktop/TTSCardMaker/site/dist/data/cards.json`
of 2026-10-20 (801 rows; dup rows dropped). Anything not checked on disk or in git is marked "to confirm". Playtest hooks are the
Play tab (`docs/Campaigns/SIMULATOR_CAMPAIGN.md`) and `docs/Campaigns/GAME_REPORTS_CAMPAIGN.md`; both unscheduled.
**2026-10-21:** the user revised D1-D3 (a separate AI sandbox repo with its own website and bot) and will iron the campaign out later;
until then, read the phases' `ai/cards` and `ai-drafts` branches as that sandbox repo.

**Decided (user, 2026-10-20)**

| # | Decision |
|---|---|
| D1 | Generated cards are completely separate from real cards: never in card statistics, never in the real card folders, always visibly labelled GENERATED (image watermark, embed title, website badge, TTS Nickname if one is ever spawned by hand as a custom object; the save `TTS/Impossibility Simulator.json` never carries one). **2026-10-21 revision:** separate by repo: generated cards live in a SEPARATE AI sandbox repo with its own website and bot, never in TTSCardMaker (no folder, no branch) and never in the real card statistics. |
| D2 | Code on a fresh AutoCroissant branch `ai/cards`; main stays AI-free. The retired diffusion feature stays on `archive/ai-generation` (ed21cd2 = faebfc7 plus a two-line header comment in `commands/diffusion.py`; main dropped it in 05c643e): `/ai`, `/ai_queue`, `/set_model|lora|scheduler|device`, `commands/diffusion.py` (SD 1.5 / SDXL / Flux, LoRA from `models/loras/`), `requirements2.txt` (torch stack); `commands/convert_nf4_flux.py` only in history (1efe116 -> deleted f40152e). **2026-10-21 revision:** no `ai/cards` branch: the code lives in the AI sandbox repo beside its own bot; AutoCroissant main stays AI-free and the archive line above stands. Text and stats come from one of two toggleable venues: Claude Opus via the API, or a custom trained model. |
| D3 | Generated PSDs/PNGs go to a TTSCardMaker branch `ai-drafts` (does not exist yet; only `main` does). Promotion to main is a human act into a dedicated folder/series, name open (Q1). **2026-10-21 revision:** no `ai-drafts` branch and no promotion into TTSCardMaker: drafts and kept cards stay in the AI sandbox repo and show on its own website. |
| D4 | Image generation runs on the Mac (Apple M2 Ultra, 24 cores, 192 GB), never on the jetson (4 cores, 6 GB, `docs/NEXT_SESSION_PLAN.md` Task 0). |
| D5 | A generated card is an edit of a blank template: Task 3's edit pipeline fills the template, so every generated PSD must round-trip through the bot's parser with zero problems. |
| D6 | 2026-10-21: the first target is 3★ Kirby creatures (Q2 answered). |

## 1. Goal and non-goals

**Goal.** Draft creatures (first) and items (later) that read like the game's own cards: rulebook timing tags, real status
effects and types, stats drawn from the real per-star distribution, then a rendered PSD + PNG a player cannot tell from a real
card in a blind vote, with a chaos slider from "house style" to "wacky". Every draft is a free, labelled synthetic PSD for the
extraction campaign.
**Non-goals.** No generated card in `stats.pkl`, `cards.json`, the deck builder's pool, the TTS save or game statistics (D1).
No new pickle (autocroissant-failure-archaeology Entry 7: `descriptions.pkl` and `metadata.pkl` retired, bb58387 / 366c8d9).
No rule invented: the model only combines rulebook vocabulary; a contradiction stays the user's to settle in the rulebook.
No AI code on main, not even an HTTP client for the API (D2).

## 2. Isolation by construction

- **2026-10-21:** with the sandbox repo (D1-D3 revised) nothing generated reaches the bot's sweep, the card website or the TTS save at
  all; the fences below stay as a second line.
- **Bot:** the sweep skips any path containing a folder in `EXCLUDE_FOLDERS = ["Markers", "MDW"]` (`commands/repo_traverser.py:36`).
  **Website:** only `CARD_FOLDERS = ("Creatures/", "Items/", "Field/", "N.M.E/", "Auxiliary/")` and not `SKIP_PARTS = ("MDW", "Markers")`
  (`site/build/build_site.py:64, 82`). **TTS save:** decks are filled by folder rules in `TTS/decks.json` (to confirm no rule globs a new
  top-level folder). So the promoted folder must be top-level, outside `CARD_FOLDERS`, added to both skip lists: invisible to all
  three until a separate, labelled surface is decided (Q1).
- Drafts live only on `ai-drafts`; neither the bot (sweeps `main`) nor the site (builds `main`) ever sees that branch.
- Metadata (chaos, seed, model id, prompt, example cards, validator report) is a sidecar `<card>.gen.json` beside the PSD. Never
  `site/Card Edits.json`: unknown fields are dropped by `valid_edit`/`parse_edits` (build_site.py:350-394) and by the bot (`commands/card_edits.py:135`).

## 3. Phases and gates

### Phase 0. Branches and the first target (decision gate: Q1, Q2)

1. `git checkout -b ai/cards` in AutoCroissant; `anthropic` goes in that branch's requirements only. Package `aicards/` (plain
   scripts run by hand; no daemon, no watcher).
2. `git checkout -b ai-drafts` in TTSCardMaker; folder `Generated/` there is a placeholder until Q1 is answered.
3. **You have a result when** both branches exist and `python -m pytest -q tests` on `ai/cards` equals main's count (238 on 2026-10-20).

### Phase 1. Text and stats on the Mac command line

**Retrieval.** For a request (kind, stars, series, chaos) pick 6-8 real cards: same stars and kind first, same series, then same
type (from `cards.json`; names, series, stars, types, stats, ability: ~300 JSON chars each).

**Vocabulary handed to the model (all real lists, re-read at run time):**
- Timing tags with their rulebook meaning: (OYT), (AAT), (XPT), (XPGT), (XPG), (EXT), (FTEXT), (EXGT), (BOT), (ENDT), (BOGT),
  (ENDGT), (OFS), (OS), (OD), (WOF) (02_Abbreviations, 11_Abilities-11_Abilities7). (XCD), (XGCD), (XU) are on cards but in no
  rulebook page (Known gaps): allowed only above chaos 60, flagged in the report.
- The 64 status effects by name, debuff star level (1★-4★; the card text's "tier" is a word the rulebook never defines, Known
  gaps) and stacking kind (19_Status_Effects, 19_Status_Effects2), with the type
  immunities (20_Creature_Types) so a fire creature never "inflicts Burning on itself for value".
- Type tokens: the 35 `kind: type` entries of `MDW/Fonts/stoogedings-src/stoogedings.json` (91 icons: 35 types, 56 stars);
  item types and their rules (14_Item_Types, 14_Item_Types2); the series names in `cards.json` (a card's series is a database
  field, not its folder: `Creatures/Other/` holds 32 series; the 24 `Creatures/<folder>` names only place the file).
- Rules that shape numbers: decks hold 2★-5★ creatures (04_Table); items cost star points equal to their stars, 7 star points per
  global turn (13_Items, 08_Global_Turn); a creature's death pays points equal to its stars (17_Death); 5+★ creatures are immune
  to negation (22_Negation); one bar = 2 HP / 0.5 DEF / 1 ATK / 1 SPD (06_Reading_Cards, 07_Stats1-07_Stats4); rounding to the
  nearest half, up (07_Stats5); minions cannot mine (15_Minions); reactions resolve in reverse order, damage last (12_Reactions3).

**Output schema** (structured outputs, `output_config.format`; never a prefill or forced tool call, both rejected by current
models): `{name, series, stars, types[], hp, def, atk, spd, ability, art_prompt, notes}`. Stats are bar counts, integers 0-10
(>10 is `STATS TOO HIGH`, `psd_parser.py` CardValidator; 0 non-integer stats exist in the data).

**Validator (hard floors at every chaos band):** every `(TAG)` in the tag list; every status word in the 64; every `[type]`
token in the manifest; name unique against `cards.json` names and aliases; stats in range; ability fits the box (Phase 2
measures); series an existing `cards.json` series (a new one only above chaos 60, flagged). A failed draft is re-rolled with the
failure appended; three failures stop the run.

**Stat envelope** (cards.json, 296 of 302 creatures with stats; mean ± standard deviation, [min-max]):

| Stars | n | HP | DEF | ATK | SPD | Sum of bars | Ability chars (median, max) |
|---|---|---|---|---|---|---|---|
| 2★ | 45 | 4.5 ± 1.3 [1-8] | 2.9 ± 2.1 [0-10] | 4.6 ± 1.6 [0-8] | 4.9 ± 1.5 [3-10] | 16.9 ± 2.9 [11-23] | 113, 464 |
| 3★ | 102 | 5.6 ± 1.3 [2-8] | 4.0 ± 1.6 [0-10] | 5.5 ± 1.2 [2-10] | 5.1 ± 1.4 [2-10] | 20.2 ± 2.5 [13-29] | 150, 488 |
| 4★ | 93 | 6.6 ± 1.2 [3-10] | 3.9 ± 1.7 [0-8] | 6.0 ± 1.4 [3-10] | 5.6 ± 2.0 [3-20] | 22.1 ± 2.8 [13-35] | 176, 475 |
| 5★ | 56 | 7.3 ± 1.5 [1-10] | 4.6 ± 1.6 [1-8] | 6.6 ± 1.8 [1-12] | 5.6 ± 1.7 [1-10] | 24.1 ± 4.1 [4-32] | 238, 515 |

Minions (91 of 108 with stats) for later: 1★ sum 10.1 ± 3.0, 2★ 14.0 ± 3.4, 3★ 15.7 ± 2.4, 4★ 14.5 ± 4.5, 5★ 21.3 ± 2.6.
Per creature type the groups are 1-50 cards (rules skill §10 table, counted from stats.pkl; 20_Creature_Types itself gives only
the immunities): too thin for separate envelopes; use the per-star rows
and let the type only steer the ability. Re-compute with the one-liner in §7 whenever cards change.

**Chaos slider** (0-100; the value, seed and model id are recorded in the sidecar):

| Band | Stats (per stat and sum) | Ability | Mechanics unlocked | Series / types | Name, art prompt |
|---|---|---|---|---|---|
| 0-20 house | within 1 SD of the star mean | 1-2 clauses | (OYT) (1PT) (1PGT) (E2T) (BOT) (ENDT) (OFS) (WOF); standard and stacking status effects | own series; one type, common in that series | plain; "series + type + what it does" |
| 21-40 deck | within 2 SD | up to 3 clauses | + (AAT) reactions, (OD), (OS), (XPG); summon an existing minion by name | own series; any one type | light wordplay |
| 41-60 developer deck | within 3 SD; sum may leave its band by 1 SD | up to 4 clauses | + counter damage, MAX HP changes, item interactions, (XCD)/(XGCD)/(XU) flagged | one crossover series allowed; two types allowed | puns; style drift to a neighbouring series |
| 61-80 wacky | anywhere in [0-10]; sum unbounded | up to 5 clauses | + star-point, points and BRICKS manipulation (04_Table2), N.M.E. shop interactions (05_Shops), persistent minions, true negation | several series; odd type pairs | silly; free-form art prompt |
| 81-100 chaos | as above | no clause cap, box must still fit | anything the validator accepts | anything | anything the validator accepts |

**Model and cost** (the claude-api skill's model table, "cached: 2026-10-06"). Per draft about 700 uncached input tokens
(examples), a ~5k-token vocabulary block served from the prompt cache, ~1k output tokens including adaptive thinking (to confirm
with `count_tokens` and `usage` on the first 20 calls):

| Model | Input / output per MTok | About per draft | 1,000 drafts |
|---|---|---|---|
| `claude-sonnet-5-5` | $2 / $10 | 1-2 cents | ~$12 |
| `claude-opus-5-5` | $4 / $20 | 2-3 cents | ~$24 |
| `claude-fable-5-1` | $10 / $50 | 5-6 cents | ~$58 |

Default `claude-opus-5-5`: the task is short and schema-bound, the validator catches vocabulary slips, and the whole month is
dollars at any tier; Fable 5.1's edge is long agentic work, not a 400-token card, so 2.5x the price buys nothing measurable here.
`claude-sonnet-5-5` is the switch if Q3 says "as cheap as possible" (re-rolls cost a cent). One model at a time: caches are
model-scoped. API shape: thinking stays on (control with `output_config.effort`, `low` for drafts), `cache_control` on the
vocabulary block, check `stop_reason` (`refusal` is possible) before reading content.

**You have a result when** 20 drafts at chaos 20 for 3★ Kirby creatures (Kirby: 56 creatures, 28 of them 3★; the largest series) pass the validator
first try at a rate fixed before the run (propose 80%), every stat lands inside 1 SD, and a reader of the 20 finds no word
outside the rulebook's vocabulary.

### Phase 2. Fill the template and round-trip (needs Task 3 step 4; the text engine is in the fork's `cards` since a6363a04)

**Template assets** (`~/Desktop/TTSCardMaker/MDW/Templates/`): `Creature_Template.psd` (1200x2400), `2_Creature_Template.psd`,
`3_Creature_Template.psd` (several-creature cards), `Item_Template.psd` (1200x1800), `ActiveItem_Template.psd`,
`Raid_Boss_Template.psd`, `Status_Effect_Template.psd` (1200x1200), `Tile Bars.psd`. Creature layers (dump 2026-10-20): group
`images` (`Clip Main image to this` 0-1731 = art; `Clip Ability Backround to this` 1721-2400 = ability background; `Ability bg
Exposure`), `Stars` (one smart object per series star: Warpstar, Hk star, Omori star, undertale, ...; visibility = choice),
`Creature Types` (one smart object per type), `Stats Bars > Darks > Hp|Def|Atk|Spd Dark > 0..9` (one visible exposure layer =
the stat; a 10 is no visible layer in that row, as Halberd Reactor's DEF 10 parses), `Text > Name`, `Text > Ability`, `Text > Stat Txt`. The item template has `Item Types` and `images > Food Background |
Basic Item background`; the status effect template `Status Effect Symbols` and a `1 star`-`4 Star` group. The parser already
reads the blank template cleanly: types `['unknown']`, hp 9 def 2 atk 2 spd 1, ability "Bazooble.", validator problems NONE
(`parse_one.py`); its path is skipped in sweeps because `MDW` is excluded.

1. Fill: `TypeLayer.set_text` for Name and Ability (runs re-fit; `'\r'` paragraphs), `set_span_style` to colour status words
   from the status chart, star and type smart objects by visibility, one digit layer per stat row (none for a 10), art pasted into the clip
   layer, a GENERATED watermark layer (D1). Fonts: `FontMap(dirs=["MDW/Fonts"])`; a `MissingFontError` fails the draft.
2. `rasterize()` the text; `report.warnings` containing `overflow` (a line with `drawn` false) rejects the ability: this is the
   "fits the box" check Phase 1 relies on. `verify()` on an unedited template layer first, to know the engine's fidelity floor.
3. Save PSD, export PNG (Task 3's delta merge is not needed: the whole card is new).
4. Gate: `parse_one.py <psd>` returns exactly the draft's name, stars, types, stats and ability and `validator problems: NONE`.
5. **You have a result when** 10 drafts give 10 PSDs that pass step 4 and open in Photoshop on the Mac with no missing font.

### Phase 3. Art

- **v1:** background and frame from the template (series star, type icon, ability background; per-series ability backgrounds, if
  wanted, are lifted from a real card of that series, to confirm the layer names match); only the illustration is generated, on
  the Mac, from `art_prompt`. Pipeline choice is open (the archive branch's Flux path loads float32 on `mps`, diffusion.py:212,
  268-269: fits 192 GB, speed to confirm). Model weights never enter a repo (`models/` was gitignored on the branch).
- **v2:** a LoRA on the card art for house style (autocroissant-research-frontier Item 2: Kirby first, training outside the
  repos, drop-in adapter only).
- **You have a result when** in a blind real-or-generated vote of 5 + 5 cards, at least 40% of generated cards are called real
  (the frontier skill's pre-registered bar). 5% is also a result: it retires v1 art with evidence.

### Phase 4. Gallery, re-roll, promote (Mac)

`python -m aicards gallery` writes `gallery.html` beside the drafts: PNGs with chaos/seed/model, the validator report, the real
examples used, buttons that write a `promote.txt`/`reroll.txt` list (static HTML, no server). `python -m aicards push` commits
drafts to `ai-drafts`; promotion to main (copy into the Q1 folder, sidecar kept, `GENERATED` label stays) is done by hand.
**You have a result when** one re-roll keeps everything but the re-rolled field, and a promoted card still appears nowhere in
`cards.json`, `stats.pkl` or the TTS save after `/update_stats` and a site build.

### Phase 5. Discord, text only (decision gate: Q4)

`/draft_card kind stars series chaos`: Phase 1 through the API, embed titled GENERATED, nothing written to pickles. Runs from
`ai/cards` only (D2: no AI code on main, the API client included); how the jetson runs that branch is Q4. Zero-jetson fallback:
the Mac script posts the drafts to the channel itself.

### Phase 6. Images from Discord, last

The jetson cannot generate (D4). Simplest honest design: the command writes a request file to `ai-drafts`; the Mac script, run
by hand, renders and pushes; the bot posts the PNG when asked. No queue daemon until the Mac is always on.

### Phase 7. Evaluation

Blind vote (Phase 3), envelope fit (share of stats within 1/2/3 SD per band), round-trip rate (Phase 2 gate over every draft),
playtests on the Play tab and in TTS (a generated card is spawned by hand as a custom object, never a card in the save), their
game reports carrying `generated: true` added to the `cards[]` row of `docs/Campaigns/GAME_REPORTS_CAMPAIGN.md` §4 (not there today) so they feed
a separate table, never the real card statistics.

## 4. Known wrong paths (from `archive/ai-generation` and the playbook; do not repeat)

| Wrong path | Evidence | Why |
|---|---|---|
| Generate where the model does not fit (the jetson, 6 GB) | OOM/dtype errors mid-generation wedge `in_progress` (diffusion.py:683 set, :782 cleared, no try/finally); every later `/ai` says "Request queued" forever; only `/restart_bot` clears it (debugging-playbook §4, §7) | One failed run kills the feature until a restart. D4 exists for this |
| Treat "PyTorch is not installed." as a bug and add torch to main | diffusion.py:661-662 is the boundary doing its job; main boots without any AI field (config-and-flags) | Torch on main breaks the zero-overhead rule and the two-machine install story |
| torch installed but `model = ''` | `init_pipeline` no-ops, `/ai` then calls a None pipeline and wedges the queue (playbook §7 TRAP) | Half-enabled is the worst state; enable fully or not at all |
| Vendor library internals for quantization | `commands/convert_nf4_flux.py` 1efe116 -> f40152e, superseded by `BitsAndBytesConfig` (archaeology table) | Dies the day the library ships the feature |
| Per-OS requirements variants | `requirements-linux*.txt` dead ends (archaeology) | One requirements file per branch |
| Put generated metadata in `Card Edits.json`, or generated cards under `Creatures/` | not an incident; §2 | Unknown fields are silently dropped; a card under a real folder is counted the moment it lands on main |

## 5. Open questions for the user

- **Q1** Folder and series name for promoted cards (top-level, outside `CARD_FOLDERS`; e.g. `Generated/`), and whether the
  website shows them in a labelled tab at all.
  2026-10-21: nothing is promoted into TTSCardMaker; the sandbox repo's own website shows them; names there wait for the iron-out.
- **Q2** First target: 3★ Kirby creatures only (proposed), or items too (same template work, different validator rows).
  2026-10-21: 3★ Kirby creatures first (D6).
- **Q3** API budget per month, which decides opus vs sonnet (table above). A hard cap goes in the script, not in prose.
  2026-10-21: two toggleable venues, Claude Opus via the API or a custom trained model (D2); the budget waits for the iron-out.
- **Q4** How the jetson runs `ai/cards` for Phase 5: a second checkout and service beside the main bot, or switching the one
  bot to that branch while drafting.
  2026-10-21: neither: the sandbox repo has its own bot (D1).

## 6. Order

Phase 0 and 1 need nothing from Task 3 and can run now on the Mac. Phase 2 no longer waits for the fork's text engine: it is in
`cards` since a6363a04 (tag cards-20261020; `docs/text.rst` describes `set_text`, `set_span_style`, `rasterize`, `verify`;
2026-10-21). Phase 3 v1 after Phase 2; Phases 4-7 follow; v2 art and Phase 6 only if Phase 3's vote clears its bar.

## 7. Re-verify

- Stat envelope: `python3 -I -c "import json,statistics as s;c=[x for x in json.load(open('$HOME/Desktop/TTSCardMaker/site/dist/data/cards.json'))['cards'] if x['type']=='creature' and not x.get('dup') and x.get('hp') is not None];[print(k,len(r:=[x for x in c if x['stars']==k]),*(f'{m} {s.mean(v:=[x[m] for x in r]):.1f}±{s.pstdev(v):.1f}' for m in ('hp','def','atk','spd'))) for k in (2,3,4,5)]"`
- Template layers and parse: `~/.venvs/ac-py314-psd125/bin/python .claude/skills/autocroissant-diagnostics-and-tooling/scripts/dump_psd_layers.py ~/Desktop/TTSCardMaker/MDW/Templates/Creature_Template.psd` and `.../parse_one.py` on the same file.
- Manifest kinds: `python3 -I -c "import json,collections as c;print(c.Counter(v['kind'] for v in json.load(open('$HOME/Desktop/TTSCardMaker/MDW/Fonts/stoogedings-src/stoogedings.json'))['icons'].values()))"` (type 35, star 56).
- Archive branch: `git ls-tree -r --name-only archive/ai-generation | grep -E 'diffusion|requirements2'`; `git log --all --oneline --diff-filter=AD -- commands/convert_nf4_flux.py`.
- Skip lists: `grep -n EXCLUDE_FOLDERS commands/repo_traverser.py`; `grep -n 'CARD_FOLDERS =\|SKIP_PARTS =' ~/Desktop/TTSCardMaker/site/build/build_site.py`.
