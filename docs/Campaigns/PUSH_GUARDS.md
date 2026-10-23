# Push guards: stale PNGs and card standards, checked on every push (2026-10-22)

**Status:** designed and BUILT 2026-10-22 (section 16), not landed: AutoCroissant `work/push-guards`, TTSCardMaker
`work/push-guards`. Two guards in one TTSCardMaker Action, `card-guards.yml` ("Card guards"), run by three new bot modules. Dry run first: a dispatch defaults to `dry_run: true`, and the repo variable `CARD_GUARDS` (unset = off)
switches the push trigger on. Measured read-only on 2026-10-22 at TTSCardMaker origin/main 6e50c3ee (812 card PSDs) with the
psd-tools fork at `cards` da3b0f6d (round 5b + fix/txt2-sync, not tagged yet) and AutoCroissant work/integ-1022.

**The owner's decisions (2026-10-22).**
1. *Stale-PNG guard:* "only re-export a card's PNG if a push pushes the PSD without updating the PNG". A push that changes a card
   PSD but not its PNG, or adds a PSD with no PNG, gets the PNG made by the bot, gated, in one bot commit. When a gate refuses,
   no PNG is written and the refusal is reported.
2. *Standardization guard:* "fix the easy issues, then ping in the channel for the remaining issues (like we do for fonts)".
   The pushed PSDs are checked against the standards of `CARD_STANDARDIZATION_CAMPAIGN.md`. Only what the fork does safely, with
   gates, is fixed (in one bot commit). Everything else is reported in a commit comment and in **one Discord post per push** to
   the standards channel (secret `DISCORD_STANDARDS_WEBHOOK`), worded "<GitHub login> pushed <card> with these issues: ...",
   one line per card, **no role ping**. The stale-PNG guard's refusals go in the same post.

**What the measurements changed in the design (section 2).**
- **The PNG comes from Photoshop's own composite, not from a render.** Every card PSD stores the flattened image Photoshop
  saved ("Maximize Compatibility"): 812 of 812 at origin/main. Converted the way the card was exported (work/r5-bot's ICC
  vote), it equals the official PNG within 2 levels on 753 cards and within the parity bar on 810.
- **Replay on 34 real commits, with the PNG withheld.** The guard's PNG equals the author's own export within 2 levels on 30
  of the 31 Photoshop-saved cards; 9 are pixel-identical, and the worst differs by 4 levels. The cards include colour badges,
  gradients and smart filters, which the fork cannot always render. So the fork is only a referee: its full render costs 5-14
  CPU-seconds a card, against 0.3 s to read the composite.
- **Many stale-PNG pushes need no PNG at all.** The one stale push of 2026-10-22 changed 3 PSDs and 0 pixels (God Tamer,
  Bubble, 12_Reactions2), and 69 of the 115 PSDs still at a stale blob match their PNG (2.4). The guard compares the old and
  new composites before writing anything.
- **The fix-it bucket is small today.** At origin/main the only safe automatic fix that applies is the Txt2 story sync (6
  cards, all already scheduled for repair). Every other open issue changes pixels, so it is reported (section 6).

## 1. Goal and non-goals

**Goal.**
- Every card PNG on main shows its PSD as last pushed: a forgotten export is made for the author, and the guard says so when it
  can't make one.
- Every pushed card PSD is checked against the card standards. The safe fixes land after the author has gone quiet; the rest is
  told to the author once per push, in the standards channel.

**Non-goals.**
- No PSD or PNG write while an author may still be editing: the 30-minute quiet rule (`card_bake.RECENT`, campaign §5).
- No pixel-changing fix, ever, without a new owner decision (§5 "Pixel-changing classes stay report-only").
- No reverting or rejecting a push.
- No role ping.
- No file renames or deletions (ledger L12).
- Rulebook pages get PNG re-exports only, never a standards fix.
- The parser notes stay card-check.yml's and the font pings stay font-check.yml's (section 6, R7).

## 2. What was measured (2026-10-22)

Evidence (persistent): `~/Documents/Tools/psd-research/evidence/push-guards-2026-10-22/` (scripts in `scripts/`; the scratch
copies are in `~/Documents/Tools/wt/scratch/guards-1022/`).
- The machine was heavily loaded by other workflows during the timings (load average 80-110 on 24 cores), so CPU time is
  quoted next to wall time.
- The Desktop clone was read only (`git cat-file`, file reads); nothing in it was written.

**2.1 The stored composite against the official PNG** (`scan_6e50c3ee.jsonl`, `scan_rules.py`, 812 card PSDs, 470 s with 6
processes).

| | Cards |
|---|---|
| Have a stored composite of the canvas size | 812 of 812 |
| Carry a non-sRGB ("Display") profile | 459 |
| Of those, exported unconverted by the ICC vote | 21 (Block_Hopper, PoD01Red_Bloon, Sabre...; the same 21 work/r5-bot names) |
| Composite (converted as voted) vs PNG: max difference 2 levels or less | 753 |
| Within the parity bar (bad <= 0.5%, mean <= 2.0), not within 2 levels | 57: the 5 bot-baked titles; 28 status effects (18 of them at max 36, on 0.1-0.5% of pixels); 24 others |
| Outside the parity bar | 2: Dark_Mind_Eye (2.4% bad), Kel (1.1%): the lost gradient maps the owner asked to restore |

- Reading the composite takes 0.13 s (median); opening a PSD takes 0.57 s (median, max 2.7).
- 13 PSDs have numbered PNGs (A_Snack2/3, All_Purpose_Medical_Kit2, Regeneration_Potion2, Watermelon2, NME_Stock2, and 7 rule
  pages: 04_Table2-5, 11_Abilities2-7 ...).
- No card PSD lacks a PNG.

**2.2 Replay of the stale-PNG guard on real commits** (`replay_sel.jsonl`, `stale_png_proto.py`).
- 34 card commits since 3a268ca that changed a PSD and its PNG together:
  - 16 by Zomoroko (the 2026-10-21 cooldown edits);
  - 15 by MichaelJSr (standardization batch, queue A, rule pages);
  - 3 bot bakes.
- Each was replayed with only the old PSD, the new PSD and the old PNG. The guard's PNG was then compared with the PNG the
  author actually pushed.

| Result | Cards |
|---|---|
| Photoshop-saved commits (new xmpMM:InstanceID and a later xmp:MetadataDate on all 31) | 31 |
| Guard PNG equal to the author's export: max 0 / max <= 2 / worst | 9 / 30 / 4 levels (Final_Dark_Mind_Eye, 99.998% within 2) |
| Outside-the-change parity against the old PNG passed | 31 of 31 |
| Fork referee over the changed box passed | 28; the 3 failures are cards whose baseline (fork on the old PSD vs the old PNG) fails too (Defcon_Panda 2.0% bad, The_Radiance mean 2.12, Clock 1.5%) |
| Bot bakes (InstanceID and MetadataDate unchanged: the fork does not touch XMP) | 3: referee passes on all 3; truth max 49 on 0.01% (PoD09DDT, PoD10BFB: the bake's merged preview vs its merged PNG) |
| Wall time per card (two PSDs from the local object store, composites, record diff, referee and its baseline) | median 9.8 s, under load |

- Photoshop rewrites volatile blocks on every save: `shmd` on every layer, `SoLd`/`PlLd` on every smart object, and `TySh` on
  type layers whose raster did not change. A cooldown edit therefore "changed" 72 of 72 layer records.
- Ignoring those blocks leaves 1-5 changed records per card, which is the real edit.

**2.3 Today's real stale-PNG push.** Commits dc37f4ad and 100511d3 (MichaelJSr, 08:16, pushed together as 07589cc7) changed
God_Tamer.psd, Bubble.psd and 12_Reactions2.psd without their PNGs (`stale_real.jsonl`). The layer records changed (stars grouped, hidden layers deleted) and
0 composite pixels moved. The guard writes nothing: "PNG still current".

**2.4 How often the stale-PNG trigger fires** (`push_history.jsonl`, `follow_up.py`).
- Data: 1,541 pushes to main from the activity API, diffed in the blobless history clone `wt/tc-full.git`. History before
  2023-11 was rewritten, so it cannot be diffed.
- From 2023-11 to 2026-10-22: 430 people-pushes touched card PSDs, and 81 of them (19%) changed 613 PSDs without their PNGs.
- In the last 3 months: 15 PSD pushes, 1 stale, and that one was 0 pixels (2.3).
- Of the 613 stale PSDs:
  - 442 got a PNG later, only 94 of them from the same person. Only 3 came within 30 minutes; the median wait was 57 days;
  - 171 never got one;
  - of the 115 still at the stale blob today: 69 match their PNG within 2 levels, 44 are within the bar, 2 are outside
    (Dark_Mind_Eye, Kel).
- So an author rarely follows up with the PNG within minutes, but the quiet window still guards the rare case (section 4.5).

**2.5 Render time with the fork** (`render_cpu.jsonl`, `render_da3b0f6d.jsonl`, `render_tag.jsonl`; whole card
`composite(ignore_preview=True)`, ability box `viewport=`, single process).

| Card | Size, layers, bytes | Full render CPU / wall | Ability box CPU / wall | Composite read + ICC | Fork full render vs PNG |
|---|---|---|---|---|---|
| Mini_Doomer | 1200x2400, 83, 30.3 MB | 13.7 s / 43-65 s | 2.0 s / 7-11 s | 0.27-0.36 s | da3b0f6d bad 0%, mean 0.14, max 30 (tag cards-20261022: max 117) |
| Gasoline | 1200x1800, 25, 26.2 MB | 4.8 s / 16-24 s | 1.6 s / 5-9 s | 0.29-0.48 s | bad 0.007%, mean 0.05 |
| The_Lich_King | 1200x2400, 83, 32.1 MB | 13.7 s / 54-70 s | 2.9 s / 13-15 s | 0.27-0.33 s | bad 0%, mean 0.08 |

- The tagged fork (cards-20261022) took about the same time.
- The fork draws a type layer from its stored raster. The colour badges in those rasters are Photoshop's, so the referee is
  valid on badged cards. The one-colour glyph problem is only for text the fork sets itself (bakes).

**2.6 What each standards rule flags at origin/main**: section 6's table (`rules_summary.json`).

## 3. Architecture

```
 git push (PSD/PNG paths) ──> card-guards.yml "Card guards" (one job; skipped unless vars.CARD_GUARDS is report/on)
   1 range    before..after: files, renames by blob sha, deletes, the last writer's login per file
   2 fetch    the pushed PSDs (and for stale-PNG cards the old PSD and old PNG) by blob sha, verified
   3 check    commands.card_guards check: guard 1 (png_export) + guard 2 (card_standards) on each card
              -> plan.json (writes, with the blobs they rest on), report.md, comment.md, discord.json, artifacts
   4 tell     the job summary; one commit comment on `after`; one Discord post (standards channel) when issues remain
   5 land     (CARD_GUARDS=on, plan has writes) sleep until 30 min after the push, drop cards a newer push touched,
              then one commit: re-exported PNGs + fixed PSDs, only where the blobs are still as pushed (git_commit)
   6 follow   gh workflow run site.yml (and tts-save.yml until Card sync replaces it); edit the comment: "landed in <sha>"
```

- **Modules (AutoCroissant, Discord-free, no config.py).** Each is runnable as `python -m`, as card_check is.
  - `commands/png_export.py`: guard 1. Composite-first export, the change detection, the gates, delta transfer for numbered
    PNGs.
  - `commands/card_standards.py`: guard 2. The rules over an open PSD (`check`), the safe fixes (`fix`), the badge
    vocabulary and the waivers.
  - `commands/card_guards.py`: the driver. Push range, downloads, plan, the reports and the Discord payload, `land`, `--local`
    (for Card sync), `--whole-tree` (the audit).
  - `SPARSE_FILES` is the import closure the workflow checks out. A test runs `--help` on exactly that list, as
    test_card_check does.
- **Reused:**
  - `font_check.pushed_files`, `tree_blobs`, `is_card_psd`, `FontIndex`;
  - `github_api.fetch_verified`;
  - `card_bake.Snapshot.png_for` / `duplicate_of` (PNG pairing, case-insensitive, numbered duplicates) and `recent_paths`;
  - `git_commit.commit_files`;
  - `psd_edit`: `_measure`, `_within_bar`, `render_region`, `save_png`, `stale_stories`, `ability_layer`, `name_layer`,
    `uses_icon_font`, plus `icc_vote` / `export_icc` / `foreign_profile` from work/r5-bot;
  - `psd_regions`: `overlay_words`, `fx_pad`;
  - `txt2_repair.repair` / `verify`.
- **Fork pin:** the tag after `cards` da3b0f6d (release step 2; it carries `sync_story`). The pin is shared with card-edits.yml
  and card-sync.yml.

## 4. Triggers and change detection

**4.1 The push range.**
- **Paths filter:** `**/*.psd`, `**/*.PSD`, `**/*.png`, `**/*.PNG`. PNG paths are included because a PNG pushed without its PSD
  is checked against the PSD (rule R10).
- **Files:** `git diff --raw --no-renames before after` in a blobless clone (`filter: blob:none`, trees only, as Card sync's
  gate). This has no 300-file cap, unlike the compare API that font_check uses.
- **Logins:** the commits API, once per commit in the range, gives each file's last writer, as `font_check.pushed_files`
  does; `github.actor` is the fallback.
- **A new branch, an all-zero before or a force push** (before is not an ancestor of after): `after` against its first
  parent, with a note in the report (font_check's rule).
- **Card PSDs** are `font_check.is_card_psd`: not under `MDW/` or `Markers/`. Rule pages are cards for guard 1 only.

**4.2 What counts per card (pairing by folder and case-insensitive stem: `Snapshot.png_for`).**

| In the push | Guard 1 | Guard 2 |
|---|---|---|
| PSD modified, PNG not | stale candidate (section 5) | check |
| PSD modified, PNG modified | none | check; R10 (PNG matches the composite) |
| PSD added, no PNG in the tree | new-card export (5.5) | check, with the new-card rules (R4) |
| PSD added, a PNG of that stem already in the tree and not in the push | stale candidate against that PNG | check |
| PSD renamed (one blob deleted and added under another path, same sha), no PNG at the new path | copy the old PNG's bytes to the new path (no new blob); report the old PNG as left behind | none (same blob) |
| PSD renamed with changes (different sha) | a delete plus an add: new-card export; the old PNG is reported as left behind | check |
| PSD deleted, PNG left | report "PNG without PSD" (never deleted) | none |
| PNG modified or added, PSD untouched | none | R10 only: the pushed PNG against the PSD's composite ("push the PSD too?") |
| PNG deleted, PSD kept | report only (deliberate or not, it is the author's) | none |
| Numbered PNGs (A_Snack2.png) of a stale PSD | delta transfer or refuse per file (5.4) | none |
| Case-only stem mismatch (Dark_matter.psd, Dark_Matter.png) | paired; nothing to do (L12) | none |

**4.3 Bot commits and loops.**
- **GitHub itself:** a commit pushed with `GITHUB_TOKEN` triggers no workflow, so the guard's own commits and the bake's, the
  save's and Card sync's never start a run.
- **The job's `if`** also skips `github.actor` AutoCroissant and `github-actions[bot]`.
- **Inside a range**, a file whose last writer is a commit authored `AutoCroissant` (`git_commit.AUTHOR`) is left out. This
  covers the jetson's `/bake_cards` fallback and the Discord writers, which push with a personal token and so do trigger
  workflows; bakes carry their PNGs anyway. Commits whose subject starts with `Card guards`, `Card edits`, `Card edit:`,
  `TTS save` or `Card sync` are left out too.
- **Idempotence** is the last line: `land` re-derives everything at the head. A PNG already current, or a story already in
  sync, writes nothing, so a re-run is harmless.

**4.4 Concurrency.**
- `group: card-guards-<sha>` for a push and `card-guards-<run id>` for a dispatch, with `cancel-in-progress: false`.
- Each push keeps its own run. A shared group, as card-check.yml and font-check.yml use, keeps one pending run, so a burst of
  pushes (the badge batch's 6 pushes in 13 minutes) loses the middle ones' reports.
- Parallel `land` steps are safe:
  - each commit lists the blobs it rests on (`land()`'s `expect`);
  - a 409/422 retries from the new head (`COMMIT_TRIES`);
  - a card whose PSD or PNG moved is dropped as superseded.

**4.5 The quiet window.**
- `land` sleeps until 30 minutes after the push, then runs `card_bake.recent_paths` (2 API calls). A card that a later push
  touched is dropped: that push's run owns it.
- On a public repo the waiting runner costs nothing. Without the sleep:
  - a fix could race an author who is still editing;
  - their next push would be rejected, and a binary PSD or PNG conflict is the worst thing a teammate can meet in GitHub
    Desktop.
- History (2.4) says a quick PNG follow-up is rare (3 of 613), but the PSD fixes need the window.
- **Alternative:** a GitHub Environment with a 30-minute wait timer on the land job holds no runner. The owner must create it
  (Settings, Environments), so it is not the default.
- After the Card sync cutover the wait disappears (section 11).

## 5. Guard 1: the stale-PNG re-export

**5.1 The source is the composite.** Photoshop writes the flattened image into every PSD it saves (2.1: all 812).
- The PNG is that composite converted as the card was exported:
  - when work/r5-bot's `icc_vote` (old composite vs old PNG) votes raw, the raw values;
  - otherwise the embedded profile's sRGB conversion (`post_process`);
  - with no foreign profile, as stored.
- It is written in the old PNG's mode, with its colour chunks (`psd_edit.save_png`).
- The fork renders nothing that is written. It is only the referee where the composite is not provably Photoshop's (G6).
- Measured on 31 real Photoshop commits: the author's own export within 2 levels on 30 (2.2).

**5.2 The gates, per card, in order** (a refusal is final for this push; the reasons are worded for card makers).

| # | Gate | Passes when | Refusal (report line) |
|---|---|---|---|
| G1 | blob | every downloaded PSD and PNG hashes to the tree's sha (`fetch_verified`) | an error line, retried by a dispatch |
| G2 | composite | the new PSD has a stored composite of the PNG's canvas size, 8-bit RGB | "PNG not made: the PSD was saved without Maximize Compatibility; save it with it on, or export the PNG" |
| G3 | visible change | the old and new composites differ by more than 2 levels somewhere (the set D) | no refusal: "PNG still current", nothing written (2.3) |
| G4 | ICC mode | `icc_vote` over the old composite and the old PNG gives a mode (new card: converted, Photoshop's Export As default) | none |
| G5 | outside the change | outside D and the changed layers' boxes (records diffed with Photoshop's volatile blocks `shmd`, `SoLd`, `PlLd`, `TySh`, `lnsr` ignored, each box padded by `fx_pad`), the new PNG matches the old PNG within the parity bar (bad <= 0.5%, mean <= 2.0) | "PNG not made: the PSD and its old PNG already disagreed outside your change (2.4% of pixels); export it in Photoshop" (today: Dark_Mind_Eye, Kel) |
| G6 | writer / referee | Photoshop saved it (a new xmpMM:InstanceID and a later xmp:MetadataDate: all 31 replayed commits; the fork's saves keep both). Otherwise the fork's render of the new PSD over D's box matches the new PNG within the bar, and the fork's baseline (old PSD vs old PNG, same box) is within the bar too | "PNG not made: the PSD was not saved by Photoshop and the bot could not check its stored image; export it in Photoshop" |
| G7 | shape | same size and mode as the old PNG (`gate_shape`) | "PNG not made: the card changed size" |
| G8 | numbered PNGs | per numbered file, the delta transfer's condition (5.4) | "A_Snack2.png not updated: re-export it in Photoshop" (the base PNG still lands) |
| G9 | quiet | at land, no later push touched the PSD or PNG, and their blobs at the head are still the pushed ones | none: dropped as superseded, and the newer push's run decides |

- The record diff of G5 also yields `where`: the share of D outside the changed layers' boxes. It is reported, not gating, on
  Photoshop-saved PSDs.
- In the replay it fired on 2 rule pages. There a type layer's only changed block was `TySh` and its stored raster stayed the
  same, yet the composite moved (the page headers). So `TySh` is not purely volatile, and `where` cannot gate. For a
  tool-written PSD it sends the card to the referee anyway.

**5.3 Outcomes.**
- **re-exported:** in the bot commit, with D's pixel count and box in the comment.
- **current:** no write; job summary only.
- **refused:** report line and Discord.
- **superseded:** summary only.
- **error:** red run, summary, artifact.

**5.4 Numbered PNGs (delta transfer).** A numbered PNG is the base export with other title layers shown (ledger L12).
- The pixels where it differs from the old base PNG by more than 2 levels, dilated by 8 px, are the toggled area.
- When D stays clear of that area, the new numbered PNG is the old one with D's pixels taken from the new base PNG.
- When D touches the toggled area, that file is refused and reported. The base PNG lands regardless.
- The rule pages with numbered PNGs (11_Abilities2-7 and the like) are different pages altogether, so they refuse, as they
  should.

**5.5 A new PSD with no PNG.**
- There is no old PNG, so G4 takes "converted" and G5 has nothing to compare.
- G6 always runs the referee, over the whole card: 5-14 CPU-seconds; round 5b's corpus has 9 cards past 0.5% bad.
- The mode is RGBA when the composite has transparency.
- The PNG is written beside the PSD, with the PSD's stem in its case.

**5.6 What the bake's gates become.**

| Bake gate (`psd_edit`) | Here |
|---|---|
| `blob` | G1 |
| `recent` | G9 (the same 30 min) |
| `duplicate` | G8 (delta transfer instead of a flat refusal) |
| `parity` (baseline) | G5, plus G6's referee baseline |
| `shape` | G7 |
| `region` | the `where` measure |
| `idempotence` | the next run finds the PNG current (G3 on PNG vs composite) |
| `size`, `bytes`, `reparse` | n/a (no PSD is written by guard 1) |

**5.7 Size.** The guard's PNGs for the 31 replayed cards weigh 60.7 MB (PIL's default deflate), against 164.6 MB for the
authors' Photoshop 2023 exports of the same pixels. A bot export adds about a third of what a Photoshop export does. Recompressing
PNGs that authors pushed was rejected: it would add a second blob for every push.

## 6. Guard 2: the standards

**6.1 The rules.**
- Scope "651" = creatures, auxiliary creatures, minions, items, auxiliary items and N.M.E. items (the ledger's 651); "cards" =
  all but rule pages.
- "At 6e50c3ee" counts every card that rule flags in the whole tree today; after waivers is what a push of that card would
  report.
- Lists are in `rules_summary.json`.

| Rule | Standard (ledger) | Class | Scope | At 6e50c3ee | After waivers |
|---|---|---|---|---|---|
| R1 | timing tags in a visible Ability layer are set in stoogedings, so they draw as badges (L1, companion §1.2); any letter case | report (a pixel change; the fork draws badges in one colour until colour glyphs) | cards | 129 cards / 226 tags: 85 creatures, 16 items, 9 field, 8 minions, 5 N.M.E., 4 aux, 1 aux creature, 1 status. Top tags (Action) 35, (1GCD) 33, (1CD) 26, (ENDGT) 18 | 129 (B2's backlog, including the cards held until the fork draws colour glyphs) |
| R2 | no inline icon image layers: `Types/` icons below the midline become `[token]` runs (L1; the parser's own rule, stat-bar layers skipped) | report | cards | 16 cards (7 status, 6 field, Tainted_Lazarus, Zess_T_Deluxe, Electric_Dart) and 6 rule pages | 15 (Tainted_Lazarus waived: its header icon) |
| R3 | exactly one visible `Text/Ability` text layer (L5, `psd_edit.ability_layer`) | report (moving a layer under `Text` adds the group's effects) | 651 | 47: 35 outside `Text`, 7 hidden, 5 with several | 42 (5 several-Ability cards waived, Q2) |
| R4 | a visible `Text/Name` text layer holds the title (L6, `psd_edit.name_layer`) | report | 651, **new cards and regressions only** (owner: "Leave the 45 Name layers") | 150: 84 titled by a card-named layer, 50 no `Text/Name`, 13 no `Text` group, 3 hidden | 0 (every existing deviation is waived; a new PSD, or a push that loses its `Text/Name`, is reported) |
| R5 | status words styled by `Ability FX:` layers, not overlay text layers (L7) | report | cards | 23 cards / 29 layers | 15 cards / 18 layers (the 16 chart-less layers on 10 cards waived, decision 2026-10-21): Ancient_Fuelweaver, Skullord, Doctor_Zomboss, Oryx, Pontiff_Sulyvahn, Undyne, Infected_Crossroads, Pananda Peel, Dark_Ritual, Snow_Bunny, Flower_Of_Decay, Freeze_Spell, Crimson_Flag, Pandora's_Box, Standstill |
| R6 | minions, auxiliary items, N.M.E. items: a visible group whose name holds "stars" (L14; the parser's contract, any case) | report (grouping needs `create_group`, which writes no layer id yet) | those 3 folders (165) | 2: Nasty_Patty, Shell_Shield (stars counted from loose layers) | 2 |
| R7 | every font a visible text layer asks for is in `MDW/Fonts` (rule 1, `font_check`) | report | cards | 1: Baba (BabaUI) | 0 (deferred by the owner); comment only while font-check.yml pings fonts (section 15) |
| R8 | Single-line composer on visible Ability and Name layers (L10) | report | cards | 0 | 0 |
| R9 | every type layer's Txt2 story holds its text (`psd_edit.stale_stories`) | **auto-fix** (`txt2_repair`) | cards | 6: Clot_White, PoD08MOAB, PoD09DDT, PoD10BFB, PoD12BAD, Springtrap | 6 (release step 4 and B2 plan them anyway) |
| R10 | the PNG matches the PSD's composite (converted as voted) within the bar | report | every push that changes a PNG or a PSD with its PNG | 2: Dark_Mind_Eye, Kel | 2 (the owner's restore is pending) |
| R11 | `Text/Ability`, `Text/Name` and the `Ability FX:` layers that key on them are named exactly so (case, outer spaces) | **auto-fix** (rename, 0 pixels) | 651 | 0 | 0 |

**Not rules (decided).**
- 28 star groups named `stars` or `3 stars` are compliant: the parser reads any case, and a rename is churn.
- Old-system tags on cards: 0. `(nPG)` is current (rules skill §3).
- L2 stat rows (27 creatures refused by `template_rows`) and L9 outlines are not in v1. L9 is not a standard (Q4); L2 is
  pending its rebuild.
- R1's tag list is `build_stoogedings.badge_tags()`. The module mirrors it, and a test pins it against the glyph names in
  `MDW/Fonts/stoogedings.ttf`: `badge_<tag>` with `/` as `_` and `-` as `m`, which is reversible, so the font itself is the
  source.

**6.2 Waivers.**
- **Class waivers** are the owner's decisions, kept in code (`card_standards.WAIVED`), each with its date and ledger entry:
  - Q2: several-Ability and multi-creature cards;
  - Tainted_Lazarus's header icon;
  - the 16 chart-less overlays;
  - "leave the Name layers" (R4 existing);
  - BabaUI.
- **Per-card waivers** go in `site/Card Edits.json` `problems_ok` with the rule's code (`STD R3`). card-check already honours
  that field, so a website admin can silence a deliberate deviation without a code change.
- An issue the push introduced (absent from the old blob) is marked "(new)".

**6.3 The fixes and their gates.**
- **What is fixed:** R9 and R11 only, 0-pixel by construction, on the pushed blob, after the quiet window.
- **Never fixed:** rule pages; when `CARD_GUARDS` is not `on`; the cards a hold names.
- **R9:** `txt2_repair.repair` (the fork's `sync_story`) then `txt2_repair.verify`:
  - only the stale stories changed;
  - every story holds its layer's text and styles;
  - every layer record, the other document blocks, the image resources and the stored composite are the old bytes.
- **R11:** the rename sets the Pascal name and `luni`, and renames the `<old> FX: ...` layers with it. Its byte proof is the
  bake's `gate_bytes` with only `name` and `luni` allowed to change.
- **Both:**
  - `gate_reparse`: the bot's parse of the fixed PSD equals the parse of the pushed one, so no data change and no changelog
    event;
  - the composite bytes are unchanged, so guard 1's PNG still holds;
  - `gate_size` within 1%.
- **Repo size:** the fork keeps every untouched byte, so git packs the fixed blob as a delta of a few KB against the pushed
  one.
- **R9 needs the fork tag with `sync_story`** (release step 2). Until it is installed, R9 is report-only, as the bake's
  `story_sync()` check does.

## 7. Outputs

**7.1 The bot commit** (one per push at most, `git_commit.AUTHOR`, through the API with `GITHUB_TOKEN`):
```
Card guards: 2 PNGs re-exported, 1 PSD fixed (push 7cb4c867)

Items/Active/2 Stars/Dirty_Bed.png: re-exported from the PSD's composite (8,961 px changed; Zomoroko, 7cb4c867)
Auxiliary/Minions/Clot_White.psd: Txt2 story synced (Name)
Guarded-push: 7cb4c867363c10cf5f562fcdf660ff7f94a154fe
```
- No `Edit-by` trailer. build_site leaves AutoCroissant's commits out of contributions (`BOT_AUTHOR`), so the author of the
  PSD keeps the credit.
- No pickles, no `site/` files, no `TTS/`.

**7.2 The commit comment** (on `after`, posted at check time; once the commit lands, the comment is edited to add "landed in
`<sha>`").
- `## Card guards`, then per card:
  - its issues;
  - the guard-1 outcome with D's size, the gates' numbers and the ICC mode;
  - the planned fixes.
- Errors last.
- A dry run writes the same text to the job summary only.

**7.3 The Discord post: exactly one per push, and only when some card has a remaining issue or a refusal.**
- **Sent:** at check time. Every verdict is final then; `land` only re-checks blobs.
- **Format:** one line per card, with the masked link to the commit that last changed it:
  `Zomoroko pushed Skullord with these issues: the Ability layer is outside the Text group; "Blazing" is an overlay text layer, not an Ability FX: layer. [7cb4c867](https://github.com/MichaelJSr/TTSCardMaker/commit/7cb4c867...)`
- **Not in the post:** planned fixes and re-exports (they are not issues) and "PNG still current".
- **Payload:** `{"content": ..., "allowed_mentions": {"parse": []}, "flags": 4}`: nobody is pinged, and the links get no
  preview cards. font-check.yml's role mention is not copied.
- **Long pushes:** whole lines are dropped to fit 2,000 characters, ending "... and N more cards: [details](<commit url>)".
- **The webhook:** the URL comes from `DISCORD_STANDARDS_WEBHOOK` into curl's `--data @file` call. It is never echoed,
  logged or written to a file. A failed call is a `::warning::`, and the comment still goes up.

**7.4 Artifacts.**
- For every card: the planned PNGs and fixed PSDs, and the crops of D (before, after, diff).
- `plan.json`, `report.md`, `discord.json` (its content, no URL).

## 8. Dry run and switches

- **Dispatch inputs** (4):
  - `dry_run` (boolean, **default true**): the job summary and artifacts only; no comment, no Discord, no commit;
  - `sha`: check this commit as if pushed, against its first parent (e.g. 7cb4c867, dc37f4ad); empty means main's head;
  - `cards`: only these, matched like `Card Authors.txt`;
  - `all`: every card PSD of the tree; standards and R10 only, job summary only; about 813 downloads and 40 min.
- **Repo variable `CARD_GUARDS`** (Settings, Variables; not a secret):
  - unset or `off`: push runs are skipped at the job's `if`;
  - `report`: push runs check, comment and post, and never commit;
  - `on`: report and land.
- **Rollout:**
  1. Land the code and the workflow with no variable.
  2. Dry dispatches with predicted results:
     - `sha=7cb4c867` (Zomoroko's cooldowns (5), 29 cards): 29 PSD+PNG pairs, so no guard-1 work; R10 on 29 PNGs, predicted
       0 flags (2.2); R1 flags the cards the badge batch has not reached yet;
     - `sha=dc37f4ad`: God_Tamer and Bubble "PNG still current";
     - `sha=100511d3`: 12_Reactions2 current;
     - one `all` run: section 6's numbers to within the pushes since.
  3. `CARD_GUARDS=report` for 3 days or 10 card pushes; read every post.
  4. `on`.
  5. Fold into Card sync at its cutover (section 11).

## 9. Runtime and cost on Actions

| Item | Estimate (Mac numbers from section 2; the runner is ubuntu-latest, 4 vCPU, assumed 1.5-2x slower per core) |
|---|---|
| Setup | blobless checkout of TTSCardMaker (1.1 MB, 2 s, as Card sync measured), sparse bot checkout (2 s), setup-python, the fork from the pip cache (~10 s; a cold build of the git pin ~60-90 s), MDW/Fonts from the cache keyed by its tree sha (215 MB, ~15 s; only the R7 rule needs it): about 40-60 s |
| Per pushed PSD, check only | blob download 1.5-2 s (25-30 MB; font_check measured 1.4 s for 15 MB), open 0.6-1 s, the rules ~1 s, R10's composite compare 0.5 s: about 4 s |
| Per stale card | plus the old PSD and old PNG (~2.5 s), two composites (0.3 s), the record diff (0.2 s), the PNG write (~1 s); the referee only for tool-written PSDs (2-6 s for a box, 10-30 s for a whole new card): about 6-10 s, up to 40 s for a new card |
| Typical push (1-3 PSDs) | about 1-1.5 min; with writes, plus the 30 min sleep and ~20 s to land |
| A batch push (50 PSDs, as the badge batch's) | about 4-6 min |
| Whole-tree audit (`all`) | 813 downloads, 16 GB: 25-30 min (card-check's measured pass), plus checks in 4 processes (~10 min); `timeout-minutes: 75` |
| The 28 GB repo | never checked out: trees only, then blobs by sha for the touched cards. The alternative is the partial clone's lazy fetch of exactly those blobs (`git sparse-checkout set --no-cone <paths>` after a blobless clone), which costs no REST calls, for when the API quota gets tight |
| API (`GITHUB_TOKEN`, 1,000 requests an hour per repo, shared with card-check, font-check and card-sync) | per push: 1 commits read per commit, k blob reads (k = pushed PSDs + 2 per stale card), 1 comment (+1 edit), at land 2 activity/compare reads and n+3 writes for n files |
| Price | free (standard runners on a public repo); the sleeping land step holds one runner for at most 30 min |
| Storage | a re-export adds one PNG blob, about 1/3 of a Photoshop 2023 export (5.7). A fix adds a delta of a few KB. Nothing is written for the common "PNG still current" case (2.3) |

## 10. Failure modes

| Case | What happens | Recovery |
|---|---|---|
| An unreadable PSD, a failed download | an error line for that card; the others go on; exit 0 (check) | a dispatch with `sha` and `cards` |
| A gate refuses | report line, Discord, no write | the author exports in Photoshop, or fixes the PSD |
| The author pushes again within 30 min | the older run drops the card (superseded); the newer run owns it | none |
| The head moved during land (409/422) | retry from the new head; cards whose blobs moved stay out | none |
| The land step fails (API 5xx, a timeout) | red run, the plan in the artifact, nothing half-written (one commit or none) | re-run the job; the idempotent re-check writes only what is still due |
| Webhook down or the secret missing | `::warning::`; the comment stands | none |
| The fork pin missing or broken | `--help` fails before any card: red, nothing posted | fix the pin; card-edits.yml shares it |
| A PSD saved without Maximize Compatibility | G2 refusal for guard 1; guard 2 still runs | save with it on |
| A PSD written by a script without updating the composite | G6 sends it to the referee, which refuses when the composite is stale | export in Photoshop |
| A wrong re-export that passed the gates | visible on the site; the commit lists it | `git revert` of the bot commit (PNGs only), then a fix |
| GitHub schedules late | n/a: push runs are not scheduled; only the Card sync fold depends on the cron | none |
| A burst of pushes | each its own run, so no report is lost (4.4); landing commits serialize by retry | none |

## 11. Folding into Card sync (`DATA_BUILD_ACTION.md` §6)

Card sync (every 30 minutes, at most one main commit) is the guard's natural landing slot. Its interval is the quiet window, so
the sleeping step goes.
- **Push run (stays).** `check`, the comment and the one Discord post, at push time. The plan goes to the artifact only.
- **Card sync stage 0, guard (new, before bake).**
  - `python -m commands.card_guards land --local "$GITHUB_WORKSPACE" --since <manifest.guard_commit>`.
  - It re-derives the plan for the PSDs and PNGs people pushed since its cursor whose last push is at least 30 min old, with
    the same gates, and writes the PNGs and fixes into the run's local commit B.
  - The bake then skips those cards for this run (its `recent` gate would otherwise see them only after the push).
  - `manifest.guard_commit` advances only on a successful run, as `card_commit` does.
- **Effects.**
  - One main commit M per interval carries the bake's, the guard's and the save's files.
  - The guard's PNGs reach the TTS save through `pending.repin` as baked images do.
  - Guard writes are data-neutral (the reparse gate), so they add no changelog events.
- **When:** at Card sync's cutover (its step 8). Until then `card-guards.yml` lands by itself (section 4.5).

## 12. Workflow sketch (TTSCardMaker `.github/workflows/card-guards.yml`)

```yaml
name: Card guards
on:
  push:
    branches: [main]
    paths: ["**/*.psd", "**/*.PSD", "**/*.png", "**/*.PNG"]
  workflow_dispatch:
    inputs:
      dry_run: {type: boolean, default: true, description: "Job summary and artifacts only: no comment, no Discord, no commit"}
      sha: {type: string, default: "", description: "Check this commit as if pushed (against its first parent); empty = main's head"}
      cards: {type: string, default: "", description: "Only these cards (comma separated)"}
      all: {type: boolean, default: false, description: "Every card PSD: standards and PNG-vs-PSD, summary only (~813 downloads)"}
permissions: {contents: write, actions: write}     # commit and comment; gh workflow run site.yml
concurrency:
  group: ${{ github.event_name == 'push' && format('card-guards-{0}', github.sha) || format('card-guards-{0}', github.run_id) }}
  cancel-in-progress: false
jobs:
  guard:
    if: >-
      github.actor != 'AutoCroissant' && github.actor != 'github-actions[bot]' &&
      (github.event_name == 'workflow_dispatch' || vars.CARD_GUARDS == 'report' || vars.CARD_GUARDS == 'on')
    runs-on: ubuntu-latest
    timeout-minutes: 75
    env:
      MODE: ${{ github.event_name == 'push' && (vars.CARD_GUARDS || 'off') || (inputs.dry_run && 'dry' || 'on') }}   # off | report | on | dry
    steps:
      - uses: actions/checkout@v7                       # trees only: the diff, the pairing, no blobs
        with: {fetch-depth: 0, filter: "blob:none", sparse-checkout: "/.github/", sparse-checkout-cone-mode: false}
      - uses: actions/checkout@v7                       # card_guards.SPARSE_FILES
        with: {repository: bobberlington/AutoCroissant, path: bot, sparse-checkout-cone-mode: false, sparse-checkout: "..."}
      - uses: actions/setup-python@v7
        with: {python-version: "3.12"}
      - run: pip install -q requests fonttools "psd-tools[composite,text] @ git+https://github.com/MichaelJSr/psd-tools@<tag>"
      # fonts cache as font-check.yml (key card-fonts-<MDW/Fonts tree sha>)
      - name: Check
        working-directory: bot
        env: {GITHUB_TOKEN: "${{ secrets.GITHUB_TOKEN }}", BEFORE: "${{ github.event.before }}", SHA: "${{ inputs.sha }}"}
        run: python -m commands.card_guards check --repo "$GITHUB_REPOSITORY" --after "${SHA:-$GITHUB_SHA}" ... --out "$RUNNER_TEMP/guards" | tee -a "$GITHUB_STEP_SUMMARY"
      - name: Tell Discord                              # MODE report/on, discord.json non-empty; URL never printed
        env: {WEBHOOK: "${{ secrets.DISCORD_STANDARDS_WEBHOOK }}"}
        run: curl -sS -o /dev/null -f -H "Content-Type: application/json" --data @"$RUNNER_TEMP/guards/discord.json" "$WEBHOOK" || echo "::warning::the Discord webhook call failed"
      - name: Comment on the commit                     # MODE report/on
      - uses: actions/upload-artifact@v4                # always()
      - name: Land                                      # MODE on, plan has writes: sleep to the quiet mark, recent check, one commit
      - name: Rebuild the site                          # a commit landed: gh workflow run site.yml (and tts-save.yml before Card sync)
```

## 13. Code plan

1. `commands/png_export.py`:
   - `composite(psd)`, `render_fingerprints(psd)` (the volatile-block list), `changed_layers(old, new)`,
     `photoshop_saved(old, new)`, `export(old_psd, new_psd, old_png, numbered) -> Export` (outcome, PNG bytes, D, the gates'
     numbers), `delta_transfer`;
   - tests: real pairs from the Desktop clone's object store (`replay_sel.json`: Dirty_Bed pixel-identical, Final_Dark_Mind_Eye
     max 4, God_Tamer current), plus synthetic PSDs for G2 and G7.
2. `commands/card_standards.py`:
   - `check(psd, rel, vocab, fonts, old=None) -> [Issue]`, `fix(path, issues, out) -> Fixed`, `BADGE_TAGS`, `WAIVED`;
   - tests: one card per rule from the clone, R9 on Clot_White (the txt2_repair test's card), and the BADGE_TAGS-vs-font test.
3. `commands/card_guards.py`:
   - `check` / `land` / `--whole-tree` / `--local`, the markdown, the Discord payload (no mentions, `flags: 4`, 2,000-character
     fit), `SPARSE_FILES`;
   - tests: the fake GitHub of test_card_check, the loop rules, superseded cards, the `--help` closure.
4. TTSCardMaker `card-guards.yml`, with `TTS/checks` step-order checks as for card-sync.
5. Order:
   - after release steps 1-3: integ landed, the fork tag, work/r5-bot with `icc_vote`;
   - the companion skill's push-guard paragraph and campaign §5 updated with the shas.

## 14. Ideas found on the way (for the tools and the fork; not scheduled)

- **The fork stamps XMP on save:** a new `xmpMM:InstanceID`, `xmp:MetadataDate`, and a History event "psd-tools". Then every
  writer is identifiable and G6 needs no referee for our own bakes. Today a fork-saved PSD looks like its last Photoshop save.
  It is our fork's change (no upstream PR).
- **A fork `render_fingerprint()` per layer** that ignores Photoshop's volatile blocks: the "what really changed between two
  saves" primitive. The data build's history (skip identical versions), the bake's `recent` and this guard would share it.
- **Composite-first everywhere.** The bake keeps Photoshop's pixels outside its region already. A website or Discord preview of
  a pushed PSD could show its composite at once, before any PNG exists.
- **The PNG encoder:** the bot's PNGs are pixel-equal and about 1/3 the size of Photoshop 2023's (5.7). A standards rule could
  tell authors their export settings bloat the repo (report only).
- **R10 run in reverse catches "PNG pushed, PSD forgotten".** Commit 07589cc7 re-exported Asteroid_Field.png without touching
  its PSD (the L16 stale export). The new PNG matches the composite (max 1 level), so R10 passes it.

## 15. Open points

- Whether font-check.yml's push run (role ping, font channel) should stop once the guard reports R7 in the standards channel.
  Until then, R7 stays out of the post and only in the comment, so a missing font is told once.
- The Environment wait timer (4.5) instead of a sleeping runner: the owner's one-time setting.
- The 57 cards within the bar but past 2 levels (2.1). They are re-exported only when pushed with a visible change, and then
  their whole PNG becomes the composite. 28 are status effects, 18 of them at max 36 on under 0.5% of pixels: probably text
  re-rasterized by a newer Photoshop. Check one in Photoshop before `on`.

## 16. Build (2026-10-22)

**What exists** (nothing pushed, dispatched or posted; the Desktop clone was only read):
- AutoCroissant `work/push-guards`: `commands/png_export.py` (guard 1), `commands/card_standards.py` (guard 2),
  `commands/card_guards.py` (the driver: `check`, `land`, `--whole-tree`, `--local`), tests `test_png_export.py` (17),
  `test_card_standards.py` (23), `test_card_guards.py` (14). work/r5-bot is merged in for `icc_vote` (one conflict:
  `PARITY_MEAN_MAX` stays 2.0, the owner's value, with r5-bot's ICC constants added).
- TTSCardMaker `work/push-guards`: `.github/workflows/card-guards.yml` (section 12's sketch made whole). Its shape is pinned by
  `test_card_guards.test_workflow_shape` (set `CARD_GUARDS_YML` to the file until it is on the clone's main).
- Run: `python -m commands.card_guards check --clone <card repo clone> --after <sha> [--before <sha>] --out <dir>` writes
  `report.md`, `discord.json`, `plan.json`, `summary.json` and the artifacts (`files/`, `crops/`); `land --plan <dir>/plan.json`.
  Scratch scripts and outputs: `~/Documents/Tools/wt/scratch/guards-build/`; evidence:
  `~/Documents/Tools/psd-research/evidence/push-guards-2026-10-22/build-replay-last10/`.

**Measured with the built code** (fork `cards` e0ea34f6, the Mac at load 75-115):
- The 34 replayed commits (2.2): 34 re-exported. The 31 Photoshop saves: 30 within 2 levels of the author's export, 9
  pixel-identical, the worst 4 levels (Final_Dark_Mind_Eye). The 3 bot bakes went to the referee and passed (Springtrap within
  2 levels; PoD09DDT and PoD10BFB max 49 on 0.3%, the bake's merged preview against its merged PNG). The PNGs weigh 66.0 MB
  against the authors' 169.9 MB.
- The whole tree at 6e50c3ee (812 PSDs, 1.7 s a card median): section 6's counts rule for rule. R1 129 cards, R2 15 (+1
  waived), R3 41 (+6 waived), R4 150 all waived, R5 15 (+8 waived), R6 2, R7 1 waived, R8 0, R9 6 to fix, R10 2
  (Dark_Mind_Eye, Kel), R11 0.
- origin/main's last 10 pushes, replayed:

| Push (UTC) | By | Run | Cards | Guard 1 | Writes | Discord lines |
|---|---|---|---|---|---|---|
| 18:13 6ba412ae..6e50c3ee | MichaelJSr | none (no PSD or PNG) | | | | |
| 17:57 c6dc6d08..6ba412ae | github-actions[bot] | none (bot push) | | | | |
| 17:56 7886ac43..c6dc6d08 | github-actions[bot] | none (bot push: the bake) | | | | |
| 17:00 5dd2a408..7886ac43 | MichaelJSr | none (no PSD or PNG) | | | | |
| 16:01 07589cc7..5dd2a408 | github-actions[bot] | none (bot push) | | | | |
| 15:18 38c55bd3..07589cc7 | MichaelJSr | check, 22 s | 19 (15 pairs, 3 stale, 1 PNG alone) | 3 current (God Tamer, Bubble, 12 Reactions2) | 0 | 1 (God Tamer: Ability outside Text) |
| 15:18 5034c4cf..38c55bd3 | MichaelJSr | check, 26 s | 19 pairs | | 0 | 0 |
| 14:44 56a79f4f..5034c4cf | MichaelJSr | none (no PSD or PNG) | | | | |
| 14:41 91fcda38..56a79f4f | MichaelJSr | check, 26 s | 39 pairs | | 0 | 5 (Ability outside Text) |
| 14:39 287e0e5a..91fcda38 | MichaelJSr | check, 36 s | 38 pairs | | 0 | 2 (Lothric, Jevil: "(Action)" not a badge) |

  No pushed PNG failed R10, and nothing would have been written. Every Discord line names a deviation the card already had
  before the push (none marked "(new)").

- The first dry dispatches of section 8, predicted by running them here:
  - `sha=7cb4c867` (Zomoroko's cooldowns (5)): 29 PSD+PNG pairs, no guard-1 work, R10 0 flags of 29 (as predicted). R1 flags all
    29 cards, each "(new)" (the cooldown tags typed in the text font); also R2 3, R3 2, R5 1. The post fits 6 full-sha lines,
    so the links now carry 12 hex digits (GitHub resolves them), about 8 lines, then "... and N more cards: [details](...)".
  - `sha=dc37f4ad`: God Tamer and Bubble "PNG still current"; one line (God Tamer, R3).
  - `sha=100511d3`: 12 Reactions2 current; 12 rule pages' PNGs match (R10); nothing told.

**Changes from the design, found while building:**
1. Per-card waivers: `site/Card Edits.json` `problems_ok: true` (a bool in `CardEdits.TYPES`, so "STD R3" codes would be
   dropped as invalid). It waives the report-only rules R1-R8; R10 is still told and R9/R11 still fixed. Today: Pix,
   Tainted Lazarus, Twin Emperors.
2. The Q2 class waiver includes Pix (a multi-creature card): R3 after waivers is 41, not 42.
3. R11's reparse gate refuses a rename that changes the bot's reading. The parser misses an `ability ` layer (trailing space),
   so renaming it changes the card's data: it stays a report. The bot never changes data on its own.
4. G6 also covers a tool save that moved no composite pixel but changed layers (a script that saved without a new composite):
   the referee renders the changed layers' box and refuses when the composite is stale (test on Mini Doomer).
5. "PNG still current", but the PNG and the composite were already outside the bar: told ("does not match the PSD, though this
   push moved no pixel"). Dark_Mind_Eye and Kel get this line when their PSD is pushed.
6. A new PSD beside an existing PNG takes the ICC mode from that PNG's own vote.
7. A PSD pushed with its PNG deleted: the standards only, no PNG made (the deletion is the author's).
8. A rename (same blob) copies the old PNG to the new name and tells that the old PNG is left at the old name.
9. PNGs under `Types/`, `site/` or `TTS/` are not card PNGs (the paths filter still starts a run, which finds nothing).
10. The fixes run at check time, so dry and report runs carry the fixed PSD in the artifact; `land` only re-checks the blobs.
11. The commit comment goes up only when a card has something told, a write or an error (`summary.json`); the job summary
    always has the report.
12. The fork pin is `cards-20261022`, card-edits.yml's. It has no `sync_story`, so in the Action R9 reports until both pins move
    to the release's tag (step 2); `story_sync()` decides.

**Before `CARD_GUARDS=report`:** merge both branches; the first dry dispatches (section 8); check one of the 57 "within the bar,
past 2 levels" cards in Photoshop (section 15).

