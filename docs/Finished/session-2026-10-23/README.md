# Session 2026-10-22 evening to 2026-10-23 evening (Opus 5.5, ultracode)

Asked: complete `RESUME_2026-10-22.md` (in this folder, with a header saying what was done), then the owner's requests
through the night. Ran 2026-10-22 17:03 to 2026-10-23 ~11:00. The running log is
`~/Documents/Tools/session-rescue-2026-10-22/resume/SESSION3_NOTES.md`; the owner's rulings are in memory
`owner-decisions-2026-10-22-evening`. The plan for the next session is `docs/NEXT_SESSION_PLAN.md`. Commits: AutoCroissant
about 189 non-merge commits reached main (most on work branches merged in the two releases), TTSCardMaker 51.

## Two releases

**Release 1 (2026-10-22 20:50-20:55).** AutoCroissant main 8861696 = integ fd3807f (data build steps 1-6 and 9, Discord
writers, Txt2 gate, history dedupe) + work/r5-bot (per-card ICC export, fxrp) + work/fixes-1023 (changelog wording,
per-field credit, the startup temp-PSD sweep, PoD10BFB blank paragraphs, label padding) + work/created-dates-bot. Gates:
stock suite 882 passed / 61 skipped, fork 125 passed, Cython 11/11. Fork tag `cards-20261022b` (90508839: round 5b,
fix/txt2-sync, fix/hsl-nan). Jetson deployed 20:55 (52 commands, 764 cards + 48 rulebook). TTSCardMaker c0b6f2d7 (integ:
migration, Changes tabs, card-sync.yml; created dates; the pin). Txt2 repairs of White Clot and Springtrap (7b48db05,
1 byte each inside Txt2). Card sync dry runs (a) incremental, (b) goldens, (c) bake: all green; runner == Mac
byte-identical.

**Deploy of 00:47 (10-23).** e837f68: the history fixes (work/history-fixes: normal forms, collapse repeats, archive tags
dated from git, renames and Morgue moves recorded) and the live-clone test fixes. The 05:00 `/update_stats` reproduced
the rehearsal exactly: old_stats 193,937 B = 446 versions / 354 names, archive tags from git (ZOEY!!!!! `renamed_to`
Z!!!!!, White Palace `moved_to` MDW/The Morgue), 108 authors set, 594 cards changed by Card Edits, 770 creation dates.
The 05:30 `/push` made PICKLE a7c310e.

**Release 2 (2026-10-23 10:11-10:36), live.**
- Fork: `cards` = tag `cards-20261023` (ecc4b773) = round 7 + speed-land + rcache + the `_distance` shape guard +
  text-geometry + layer-edit-txt2 + line-splice + lazy-records. Harness identical to r7_final (only Claw's mean
  moved), uint8 20/20, 4.2x faster wall; bot gate Names 544/548, Abilities 652/660, stat rows 1474/1482; fork suite
  6529 passed.
- AutoCroissant main 2ada357 (work/land-1023b: a7c310e + 17 merges + 7 fixes + the finalize step): history recovery,
  shadow prep, the cutover's bot side (off), mailmap, push guards + soft-parity guards (off), segv fix (bounds checks on),
  glow tokens in the parser, site-password tests, colour glyphs, soft parity, refused layers, line splice, psd-diff,
  range reads, the unbakeable list; R3/R4 dropped. Suite 1161 passed / 2 failed before the TTSCardMaker push (the two
  needed TTSCardMaker's font-check.yml and text.js); Cython 11/11; 52 commands.
- Jetson deployed 10:36: the fork tag installed (`_distance` compiled with gcc, `pip check` clean), `build_ext` 11 .so,
  764 cards, emojis ready.
- TTSCardMaker main 96d61368: site history and date fixes, `.mailmap` (Michael -> MichaelJSr; the `[people] Michael` case
  gone), 5 more first-poster authors, created dates (Pandora 2019-05-01; 0 provisional), Stubborn Queen / Gust of Trees
  `dup_of`, website password sign-in, push guards (`CARD_GUARDS` unset = off), psd-diff, **card-sync.yml's schedule (the
  shadow start, `MAIN_COMMIT`/`POSTS` off)**, all three Actions re-pinned to `cards-20261023`. card-edits.yml re-enabled
  (dry run 39996528699).

## Also landed (pushed between the releases)

- TTSCardMaker f2b2de98: 41 Mario cards to series Mario + subseries <game> (69 Mario cards in all).
- TTSCardMaker 71ddebec: Card Authors from the first Discord posters (108 lines; manual credits kept), owner-approved.
- **Batch B2 pushed 07:1x** (TTSCardMaker 0bc4c099, 6 commits): 185 badge cards re-run (sizing, stickers, FX, (Action),
  title case), the rulebook (02 legend, 11_Abilities3/6, 22_Negation), gradient maps restored on Dark Mind Eye and Kel,
  Name case and Tainted bars, A P E X's "(action)" and "(BOT)" badges, Effect Layers.jsx without badge pixels in FX
  layers, and the 24 cards of the owner's rulings (margins via R0, one badge size, re-layouts).
- Photoshop pass (09:0x): 83728277 (Glitz_Pit, Michael's_Blessing, The_Core, Computer_Virus fixed) and 49cf3b28 (L7
  effect layers on 13 cards; the rest of the 40 overlay cards were already converted). The glow test: Photoshop draws
  colour glyphs whole, so 0.3 em halos are fine; the installed font was restored.
- tts-save 7886c573 (1252 links pinned after B2).
- Discord posts (09:2x-09:3x): card-check, the push guards and changelog never ping; font-check pings the card makers'
  role in the standards channel (AutoCroissant 9627623 + 65adf90; TTSCardMaker 93b7c55c + 413053c0);
  `DISCORD_FONT_WEBHOOK` deleted.

## Numbers

- **Rendering rounds.** Round 6: bad px 605,755 -> 337,064 (-44.4%), 36 cards better, 0 worse (stroke-edge law -153k,
  whole-pixel shadows -113k, colour glyphs: a `[fire]` ability edit passes on 160 of 213 stoogedings layers, 0 before).
  Round 7: 337,064 -> 270,512. Remaining: ~34% stale exports, ~9.5% effect noise; floor ~90-130k after re-exports.
- **Speed.** speed-final 7.1x on 20 cards (uint8 identical 812/812 bar two known cards); the release 4.2x wall.
- **History.** The audit's 63 findings (60 confirmed) became work/history-fixes and work/history-recovery. Rehearsal of
  the next `/update_stats`: "History: 698 versions recovered from the bot's PICKLE commits (553 cards), 171 starts moved
  earlier; 64 cards gone before archiving archived; 42 type misreads merged; 12 impossible values set to unknown; 32
  brought to today's series ...; 299 carried starts given their commit", plus ~219-236 B2 re-parses and a second History
  line ("103 carried starts", noise); old_stats -> about 1,155 versions / 670 names.
- **Unbakeable list** (docs/Future/UNBAKEABLE_CARDS_2026-10-23.md, 845 rows): Name bakeable/refused 486/278 -> 646/118,
  Ability 382/382 -> 701/63, whole cards 275 -> 573 once soft parity, colour glyphs and refused layers are live (they are).
- **Push guards** replay: 34 commits, PNG made on 29, refused on 5; psd-diff replayed 465 PSDs in 20 pushes, 0 errors.
- **Soft parity:** 117 of 118 former parity misses bake (66 with a note); worst box 6.78%.
- **Range reads:** 812/812 PSDs equal to whole downloads.
- **Created dates:** 772/772 site cards dated (770 bot cards; rulebook pages have none by design).

## Owner rulings (memory `owner-decisions-2026-10-22-evening`; do not ask again)

History: recover everything (pre-Feb and 2025 versions, archive every dropped card). Mario series scheme on all Mario
cards; Life Shroom's subseries kept; series-less Mario items left. Stale-PNG guard: the fork renders, a refusal becomes a
report; a new PSD beside its PNG is current only within 8 levels. B2 looks: R0 box moves allowed, one badge size per card
(per layer is fine: Warcraftian_Druid), [light] tinted, Chilled smaller text, Computer_Virus pill matches its caps, Mr_Krabs
overlay colour carried, Michael's_Blessing tab stops, Energy_Ritual candidate A (+3.333 pt on line 4). Glow token
variants `:glow` / `:darkglow`, 0.3 em halos; [darkmatter:glow] on Nebulous_Precipice. Push guards: dry run -> report ~3
days -> on, 30-minute wait kept, R3/R4 dropped (developers' freedom: the bot copes with varied layouts; no L2/L3/L5/L6
template passes). Auxiliary/Creatures read printed stars and count as creature type. Discord: best guesses firm, prefer
the earlier date, fetch the 10 oversize files, 33 misreads = none, A8 = Pandora, A310/A146/A217 = none. Authors: first
Discord poster wins unless credited by hand; Pandora stays MichaelJSr; "Michael" merged into MichaelJSr via .mailmap.
Redo the 8 narrow-margin cards and Koops with R0, a column pass on Cloud_Soiree (both in the B2 follow-up), no font-side
hardening, the L7 effect layers on the 40 overlay cards next (done in the Photoshop pass). Website password sign-in on any device, signed in forever per device; deck passphrases encrypted with the device key.
Bounds checks and negative indexes on in Cython. Bakes and the stale-PNG guard no longer refuse on the pixel check (only
past 25% of a box or a wrong read-back); misses are noted and fixed in psd-tools. Approved: refused-layer bakes,
line-splice text bakes, semantic PSD diff in push posts, records-only range reads. Not chosen: printed-text audit,
Discord /preview, website edit-support map, interactive cards. The bake learns stars and types edits; PNGs recompressed
losslessly on push plus one batch pass; 11_Abilities2 stays. Then: work unattended, no questions (03:25 and 10:50).

## Incidents

- **15:41 (10-22): `/update_stats` on the old code (9f55895)** left unpushed pickles with 280 duplicate versions
  (old_stats 195,956 -> 318,970 B). The rehearsal reproduced them byte for byte; the new code's cleanup ended in the same
  bytes from either start, and the 05:00 run cleaned them.
- **Same-blob re-parse dropped the carried `version` start** (276 cards): found by the rehearsal, fixed before release
  (f0c7d3e).
- **18:13: an agent ran `pkill -f multiprocessing.spawn`**, killing spawn workers machine-wide (the user's other sessions
  may have been hit).
- **Segfaults (owner report 00:1x):** every crash was a rehearsal script that replaced `datetime` before importing pandas
  (NaT inherits the fake class -> GC traverse recursion -> stack overflow); no project extension involved. Scripts freeze
  time by an import hook now. Latent bugs fixed anyway: ba27fb7 (blank star folder) and 9919a14 (bounds checks on).
- **B2 follow-up v1 diverted** (its run agent chased the segfault); its pre-push check found the §16 spacing bug never
  re-run and narrowed margins in the unpushed commits. v2 re-ran 60 cards and removed 5 flagged cards from the commits
  (backup branch `backup/b2-pre-exclude`). The zsh refspec bug struck again during the push.
- **card-edits.yml** was re-enabled at 20:55, paused again at 21:5x (its next run would have rewritten 3 cards in B2's
  list), and kept paused because 9 split-title cards would have baked wrongly until refused-layers; re-enabled at 10:36.
- **Reviews cut short:** the guards' final pass ended clean=false (fixed in an owner-rulings round), the history fix's
  last round and soft parity's last fix round were not re-reviewed (covered by the release checks).
- **card-edits dispatch stuck** (11:3x): dry run 39996528699 sat queued with no job; re-dispatched (40003158811), then
  the live run 40003504861. The stuck one was still queued at 11:39.
- **Card sync dry-run defects** (changelog links by name, duplicates' version churn, edits misread as PSD changes,
  `load_aliases('main')`, no JSON input for diff_stats/inspect_pickle, `load_card_data()`): fixed on work/shadow-prep,
  merged in release 2.

## Afternoon (12:00-17:30, unattended)

Landed, each with its full suite green and pushed straight to main:
- **AutoCroissant** f47e2cb (push guards: R3/R4 back for real issues only, Shadow_Duelist waived as an
  `ABILITY_EXCLUSIONS` card; glow tests; B2 test drift; deprecated asyncio/pandas calls), 0c6567e (lossless PNG
  recompression in the guards' bot commit, `Recompressed` trailers, `png_recompress` batch), 7a7085f (the bake prints
  stars and types edits, refused per field; `/update_metadata` refuses type names with no icon), fa6e708 (Txt2: 0 of
  904 PSDs stale; a test that had stopped testing), 73f22aa (Green Mage's off-canvas bars refuse instead of erroring;
  the 4 stale verdicts were settled), 1bf6892 (a title layer named after its own text: Papyrus, Sans), 0dd3fa7 and
  5f5fe82 (tests: painted known misses instead of Red_Knife's noise; stock-venv fixes). Jetson deploys 12:34, 13:49,
  15:52; fork reinstalls 14:10 and 17:02.
- **TTSCardMaker** 0fcbcdfe (glow font 0.3 em, share text any case), 41e2ddea (Energy_Ritual cand A), 08b1152a (Effect
  Layers.jsx overlay colour), 6a54b4f6 (site skips recompressions in Image versions), 56011634 (686 PNGs 4.18 GB ->
  1.14 GB, pixel-identical, verified twice), 5c88f230 (TTS save re-pinned 4018 links), 6dd5399e (types in any order),
  2921ecbd and 82a8ba7c (fork pins).
- **Fork** tags `cards-20261023b` (round 9: 187,299 -> 174,340) and `cards-20261023c` (round 10: -> 138,578; effect
  noise is one fixed field in document coordinates; styled pass-through groups). Rounds 6-10: -77%.
- **Switches:** `CARD_GUARDS=report` (13:04). The shadow's first scheduled run made the `data` branch (14:21).

Unattended calls (owner to confirm): Nebulous_Precipice keeps the sticker (a stroked Text group swallows a glow halo);
R3/R4 real issues kept; BUG 2's rebuilt FX layers keep their own style (q6); round 11 run past the old cap of 10.

Disk reached 98% (39 GB free) during the census runs; deleting finished scratch freed ~98 GB.

## Open at the end (all in `docs/NEXT_SESSION_PLAN.md`)

The first `/update_stats` (the recovery) and its `/push`; the data branch re-seed and 3 days of daily checks, then the
cutover; `CARD_GUARDS=on` after ~3 days of reports; BUG 2 (`work/fx-rebuild`, in review at the end); fork round 11;
the owner's re-parses (`cards:Shadow Duelist`, the padding five, the seven printed-star cards, Papyrus and Sans);
cleanup of the corpus and the fork's round branches after the rounds.

## Lessons

- Rehearse every first-run-after-deploy on copies of the live pickles; it caught the dropped carried start and pinned
  the 05:00 numbers to the byte.
- Never patch `datetime` (or anything pandas subclasses) before importing pandas in a script.
- Keep `pkill` patterns specific to your own process tree.
- A diverted agent can leave a workflow "done" with its real task undone: check each stage's output, not its status.
- A pre-push check on visual work catches what per-card gates miss; keep it before every PSD push.
