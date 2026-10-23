# Push guards: stale PNGs and card standards, checked on every push (2026-10-22)

**Status:** designed and BUILT 2026-10-22 (section 16), the owner's three rulings of 2026-10-22 ~23:00 built (section 17),
the pixel-miss ruling of 2026-10-23 built (section 18, AutoCroissant `work/soft-parity-guards` on `work/push-guards`),
not landed: AutoCroissant `work/push-guards`, TTSCardMaker `work/push-guards`. Two guards in one TTSCardMaker Action, `card-guards.yml` ("Card guards"), run by three new bot modules. Dry run first: a dispatch defaults to `dry_run: true`, and the repo variable `CARD_GUARDS` (unset = off)
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

**The owner's rulings (2026-10-22 ~23:00), built in section 17.**
1. *The PNG is a fork render*, the same render the bake uses (`psd_edit.render_region` and `merge_png`, the card's ICC
   export mode by `icc_vote`, the fork pinned at `cards-20261023` as card-edits.yml), **not** Photoshop's stored composite.
   Gated as designed (the old PNG outside the PSD's changed layers, the bake's gates). A refusal is told on Discord and in
   the comment ("PNG not made: ...; export it in Photoshop"). The stored composite stays only as a cross-check that can
   refuse, never as the PNG's source.
2. *font-check.yml posts to the standards channel*: `DISCORD_STANDARDS_WEBHOOK` instead of `DISCORD_FONT_WEBHOOK`, its role
   ping and wording kept. Since 2026-10-23 card-check.yml posts there too, with no ping; `DISCORD_FONT_WEBHOOK` is deleted.
3. *The guard tells font issues (R7)* in its standards-channel post too.

**The owner's ruling (2026-10-23): R3 and R4's template layout dropped.** Where the Ability layer sits (inside the Text
group or not) and which layer holds the title (`Text/Name`, one named after the card or, since title-own-text, after its
own text) are the developers' freedom, and
the bake handles those layouts. The ruling was first applied as dropping both rules whole (2ada357); R3 and R4 are back
for the real issues only (section 6.1), with their waivers and `card_guards.added()` (R4's new-card rules): a hidden
Ability layer with words the parser reads nowhere (an empty one, on a card with no ability, or one laid out again in
visible pieces is not told), several visible Ability layers (the bake can't pick one), and a hidden or rasterized Name
layer the title can't be read from. Bringing them back was an unattended call during the release's finalize, **not an
owner ruling**: the owner confirms it, and R3's told list, before `CARD_GUARDS=report` (section 15).

**What the measurements changed in the design (section 2).** (The first two bullets argued for the composite as the PNG's
source; ruling 1 overrules them: the PNG is the fork's render, section 17. They stand as measurements.)
- **Photoshop's own composite.** Every card PSD stores the flattened image Photoshop saved ("Maximize Compatibility"): 812
  of 812 at origin/main. Converted the way the card was exported (work/r5-bot's ICC vote), it equals the official PNG
  within 2 levels on 753 cards and within the parity bar on 810. It is now the cross-check (5.2).
- **Replay on 34 real commits, with the PNG withheld.** The guard's PNG equals the author's own export within 2 levels on 30
  of the 31 Photoshop-saved cards; 9 are pixel-identical, and the worst differs by 4 levels. The cards include colour badges,
  gradients and smart filters, which the fork cannot always render. So the fork is only a referee: its full render costs 5-14
  CPU-seconds a card, against 0.3 s to read the composite.
- **Many stale-PNG pushes need no PNG at all.** The one stale push of 2026-10-22 changed 3 PSDs and 0 pixels (God Tamer,
  Bubble, 12_Reactions2), and 69 of the 115 PSDs still at a stale blob match their PNG (2.4). The guard compares the old and
  new composites before writing anything.
- **The fix-it bucket is small today.** At origin/main the only safe automatic fix that applies is the Txt2 story sync (6
  cards at `6e50c3ee`, all synced since: 0 at `6a54b4f6`, R9 in section 6). Every other open issue changes pixels, so it is
  reported (section 6).

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
- The parser notes stay card-check.yml's. The font pings stay font-check.yml's (its role ping, now in the standards
  channel); the guard tells R7 in its own post too (ruling 3, section 6).

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
   4 comment  the job summary; one commit comment on `after` (a re-run edits it)
   5 land     (report/on, something to tell or write) sleep until 30 min after the push, drop writes a newer push touched,
              then (on) one commit: re-exported PNGs + fixed PSDs, only where the blobs are still as pushed (git_commit);
              settle the post (after_window): discord-final.json, report-landed.md
   6 follow   edit the comment ("landed in <sha>", "not landed"); gh workflow run site.yml (and tts-save.yml until Card sync)
   7 tell     one Discord post (standards channel) when something remains; a failed run: one alert per failure streak
```

- **Modules (AutoCroissant, Discord-free, no config.py).** Each is runnable as `python -m`, as card_check is.
  - `commands/png_export.py`: guard 1. The bake's render of the changed layers merged into the old PNG (ruling 1), the
    change detection, the gates with the composite as a cross-check, delta transfer for numbered PNGs.
  - `commands/card_standards.py`: guard 2. The rules over an open PSD (`check`), the safe fixes (`fix`), the badge
    vocabulary and the waivers.
  - `commands/card_guards.py`: the driver. Push range, downloads, plan, the reports and the Discord payload, `land`, `--local`
    (for Card sync), `--whole-tree` (the audit).
  - `commands/psd_diff.py`: what changed inside a PSD pushed over an older one, for the reports (7.5).
  - `SPARSE_FILES` is the import closure the workflow checks out. A test runs `--help` on exactly that list, as
    test_card_check does.
- **Reused:**
  - `font_check.pushed_files`, `tree_blobs`, `is_card_psd`, `FontIndex`;
  - `github_api.fetch_verified`;
  - `card_bake.Snapshot.png_for` / `duplicate_of` (PNG pairing, case-insensitive, numbered duplicates) and `recent_paths`;
  - `git_commit.commit_files`;
  - `psd_edit`: `_measure`, `_within_bar`, `render_region`, `merge_png`, `gate_shape`, `gate_region`, `save_png`,
    `stale_stories`, `ability_layer`, `name_layer`, `uses_icon_font`, plus `icc_vote` / `export_icc` / `foreign_profile` from
    work/r5-bot;
  - `psd_regions`: `overlay_words`, `fx_pad`;
  - `txt2_repair.repair` / `verify`.
- **Fork pin:** `cards-20261023` (it carries `sync_story`), shared with card-edits.yml and card-sync.yml: guard 1's render
  is the bake's.

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
| PSD modified, PNG not | stale candidate (section 5; G5's burst anchor) | check |
| PSD modified, PNG modified | none | check; R10 (PNG matches the composite; comment only) |
| PSD added, no PNG in the tree | new-card export (5.5) | check, with the new-card rules (R4) |
| PSD added with its PNG (the usual new card), or with its PNG deleted | none | check, with the new-card rules (R4); R10 |
| PSD added, a PNG of that stem already in the tree and not in the push | stale candidate against that PNG | check |
| PSD renamed (one blob deleted and added under another path, same sha), no PNG at the new path | copy the old PNG's bytes to the new path (no new blob); report the old PNG as left behind (comment only) | none (same blob) |
| PSD renamed with changes (different sha) | a delete plus an add: new-card export; the old PNG is reported as left behind | check |
| PSD deleted, PNG left | report "PNG without PSD" (never deleted; comment only, review 3) | none |
| PNG modified or added, PSD untouched | none | R10 only: the pushed PNG against the PSD's composite ("push the PSD too?"; comment only) |
| PNG deleted, PSD kept | report only (deliberate or not, it is the author's) | none |
| Numbered PNGs (A_Snack2.png) of a stale PSD | delta transfer or refuse per file (5.4) | none |
| Case-only stem mismatch (Dark_matter.psd, Dark_Matter.png) | paired; nothing to do (L12) | none |

**4.3 Bot commits and loops.**
- **GitHub itself:** a commit pushed with `GITHUB_TOKEN` triggers no workflow, so the guard's own commits and the bake's, the
  save's and Card sync's never start a run.
- **The job's `if`** also skips `github.actor` AutoCroissant and `github-actions[bot]`.
- **Inside a range**, a file that only commits authored `AutoCroissant` (`git_commit.AUTHOR`) or `github-actions[bot]`
  changed is left out. This covers the jetson's `/bake_cards` fallback and the Discord writers, which push with a personal
  token and so do trigger workflows; bakes carry their PNGs anyway. A file a person also changed in the range is guarded
  (the person's commit is its writer). By author only (review fix, 2026-10-22): a person's commit whose subject starts
  `Card edits` or `TTS save` (the owner's workflow commits are named so) is guarded like any other.
- **Idempotence** is the last line: `land` re-derives everything at the head. A PNG already current, or a story already in
  sync, writes nothing, so a re-run is harmless.

**4.4 Concurrency.**
- `group: card-guards-<sha>` for a push and `card-guards-<run id>` for a dispatch, with `cancel-in-progress: false`.
- Each push keeps its own run. A shared group, as card-check.yml and font-check.yml use, keeps one pending run, so a burst of
  pushes (the badge batch's 6 pushes in 13 minutes) loses the middle ones' reports.
- Parallel `land` steps are safe:
  - each commit lists the blobs it rests on (`land()`'s `expect`);
  - a 409/422 retries from the new head (`COMMIT_TRIES`);
  - a write whose blobs moved is dropped as superseded: a PNG rests on the whole card (PSD, PNG, numbered PNGs), a PSD fix
    (0 pixels) on its PSD alone, so an author's later PNG push does not drop a pending fix.

**4.5 The quiet window.**
- `land` sleeps until 30 minutes after the push, then runs `card_bake.recent_paths` (2 API calls). A write that a later push
  touched is dropped: that push's run owns it.
- Every write also rests on its blobs (G9), looked up in any letter case: a teammate's `foo.png` pushed in the window for a
  new `Foo.psd` drops the bot's `Foo.png` and withdraws a refusal (review 3). A blob that is the bot's own write (a re-run
  after a landing) still holds: the re-run writes nothing and the comment says "already on the branch".
- **A dispatch with `dry_run` off** has no quiet window: the last 30 minutes' pushes would include the checked push
  itself, so every write would be dropped as "a later push touched it" (review 3). It lands at once on the blob check
  alone, which sees every change since the checked commit. So it is refused (the step "No push run in its window",
  review 4) while a push run whose head is that commit, or comes after it, is unfinished: that run's comment has no told
  mark yet, so both would post. A head the checkout lacks (pushed since) counts too. A dry dispatch is never refused.
- **A card whose PSD changed again** is left to the later push's run, unless the last commit that changed it is a bot's
  (bake, Card sync, another guard run): no run follows that one, so the push's lines are told anyway (`_bot_changed`, one
  API read per such card).
- **The Discord post waits too** (review 2, 2026-10-22), in report mode as well (`land --dry-run`). After the window, a PNG
  refusal, or in report mode a PNG the bot would make (review 6), is withdrawn when the author pushed that PNG meanwhile, a card whose PSD changed again is left to that push's own
  post, and a write that did not land without the author replacing it is told ("PNG not made: ...; export it in Photoshop",
  "... (the bot could not fix it: ...)"). `card_guards.after_window`.
- On a public repo the waiting runner costs nothing. Without the sleep:
  - a fix could race an author who is still editing;
  - their next push would be rejected, and a binary PSD or PNG conflict is the worst thing a teammate can meet in GitHub
    Desktop.
- History (2.4) says a quick PNG follow-up is rare (3 of 613), but the PSD fixes need the window.
- **A re-run** counts the window from the push (the push event's `repository.pushed_at`, Unix seconds or an ISO time),
  not from its own Start (the fallback for anything else), so a re-run hours later lands at once on the blob check and
  `recent_paths` (review 5: it used to wait another 31 minutes).
- **Alternative:** a GitHub Environment with a 30-minute wait timer on the land job holds no runner. The owner must create it
  (Settings, Environments), so it is not the default.
- After the Card sync cutover the wait disappears (section 11).

## 5. Guard 1: the stale-PNG re-export

**5.1 The source is the bake's render** (ruling 1, 2026-10-22; until then the stored composite was written, section 16).
- R, the changed layers' box: the layer records that differ between the old and the new PSD (Photoshop's volatile blocks
  ignored, 2.2), each padded by its effects' reach (`fx_pad`).
- The fork pinned at `cards-20261023` renders R for the old and the new PSD (`psd_edit.render_region`, never the stored
  preview) in the card's ICC export mode: work/r5-bot's `icc_vote` over the old PSD's composite and the old PNG (raw when it
  votes raw, else converted by the embedded profile).
- The old PNG takes the new render's pixels where it differs from the old render (`psd_edit.merge_png`, the bake's
  delta merge); every other pixel stays as Photoshop exported it. F is where the render moved past 2 levels.
- It is written in the old PNG's mode, with its colour chunks (`psd_edit.save_png`).
- The fork draws a type layer from the raster Photoshop stored, so colour badges and Photoshop's text come out as
  Photoshop drew them; the one-colour glyphs are only for text the fork sets itself (the bake's edits), not here.
- **Photoshop's stored composite is never written.** It is a cross-check that can only refuse (G2, G5, G6), converted the
  same way: D is where the old and new composites differ past 2 levels.

**5.2 The gates, per card, in order** (a refusal is final for this push; the reasons are worded for card makers and every
one ends "export it in Photoshop").

| # | Gate | Passes when | Refusal (report line) |
|---|---|---|---|
| G1 | blob | every downloaded PSD and PNG hashes to the tree's sha (`fetch_verified`) | an error line, retried by a dispatch |
| G7 | shape | the old PNG is the canvas's size, RGB or RGBA (`gate_shape` and `gate_region` again after the merge) | "PNG not made: the card changed size; export it in Photoshop" |
| G2 | composite | the new PSD has a stored composite of the canvas size, 8-bit RGB (the cross-check needs it) | "PNG not made: the PSD was saved without Maximize Compatibility, so the bot cannot check its drawing; save it with it on, or export it in Photoshop"; not 8-bit RGB (CMYK, 16-bit): "PNG not made: the PSD is not 8-bit RGB, so the bot cannot check its drawing; export it in Photoshop" |
| G3 | visible change | the render moved past 2 levels (F) or Photoshop's image did (D) | no refusal: "PNG still current", nothing written (2.3) |
| G4 | ICC mode | `icc_vote` over the old composite and the old PNG gives a mode | none |
| G5 | the old PNG | (a G5 or G6 refusal is retried once against the **burst anchor**: the PSD as of the PNG's last change, when every PSD commit since is within 2 hours of the push, `ANCHOR_WINDOW`; so two quick PSD pushes, the first superseded, still get their PNG) outside R and D, the old PNG matches the new PSD's composite within the parity bar (bad <= 0.5%, mean <= 2.0): the merge keeps the old PNG there. Inside R, when the baseline (G6) fails and the old composite disagrees with the old PNG there too, the PNG was stale already | "PNG not made: the PSD and its old PNG already disagreed outside your change (2.4% of pixels); export it in Photoshop" (today: Dark_Mind_Eye, Kel); "... already disagreed where you changed it (...)" (Restrained in the replay) |
| G6 | the bot's drawing | (a) the bake's parity gate: the fork's render of the old PSD over R matches the old PNG within the bar; (b) the cross-check: wherever the render or Photoshop's image moved (F and D), the new PNG matches the new composite within the bar; F empty while D is not: refused. Since 2026-10-23 (a) and (b) refuse only past the floor (more than 25% bad); a miss of the bar under it is written and told (section 18). D outside every changed layer that disagrees still refuses | (a) "PNG not made: the bot cannot draw this part of the card as its old PNG shows it (1.4% of pixels differ); export it in Photoshop"; (b) "PNG not made: the bot's drawing of your change does not match the PSD's own image (...)"; D outside every changed layer: "PNG not made: the PSD's own image changed where the bot sees no layer change; export it in Photoshop"; a tool save whose composite is stale: "PNG not made: the PSD's stored image no longer shows its layers (it was not saved by Photoshop); save it in Photoshop with Maximize Compatibility, or export it in Photoshop" |
| G8 | numbered PNGs | per numbered file, the delta transfer's condition (5.4) | "A_Snack2.png not updated: re-export it in Photoshop" (the base PNG still lands) |
| G9 | quiet | at land, no later push touched the PSD or PNG, and their blobs at the head are still the pushed ones | none: dropped as superseded, and the newer push's run decides |

- `writer` ("photoshop" when the PSD has a new xmpMM:InstanceID and a later xmp:MetadataDate, else "tool") only words a G6
  (b) refusal now; every PSD is rendered.
- `where`, the share of D outside R, is reported. In the replay it is the rule pages whose type layer changed only its
  `TySh` (its stored raster stayed, Photoshop's image moved): the render cannot see that change, so G6 refuses.

**5.3 Outcomes.**
- **re-exported:** in the bot commit, with F's pixel count and box, the baseline and the cross-check in the comment. A
  measure that missed the bar under the floor adds "made with a pixel difference (...): bad X%, mean Y, max Z" to the
  comment's line and the bot commit's line (section 18).
- **current:** no write; job summary only. Also when the PNG in the tree already shows the pushed PSD (review 6): the
  author pushed the PNG first and the PSD minutes later in a push of its own. Over F the PNG is nearer the new render than
  the old, within the bar there, and within the bar of Photoshop's image over the card (`png_ahead`), so the bot does not
  re-export over the author's export. A new PSD beside its PNG (5.5) is current when no pixel of the PNG is more than
  BESIDE_CUT (8 levels; owner 2026-10-23, author exports differ by at most 4) from Photoshop's image: nothing is rendered.
- **refused:** report line and Discord ("PNG not made: ...; export it in Photoshop").
- **superseded:** summary only.
- **error:** the card's line on Discord ("the bot could not finish checking it (MemoryError); PNG not made: export it in
  Photoshop" when it needed a PNG), the error in the comment, the other cards go on, and the step "Fail on card errors"
  turns the run red (`summary.json` `card_errors`; review 3). Errors outside the cards (an unreadable font, the manifest)
  are in the comment only.

**5.4 Numbered PNGs (delta transfer).** A numbered PNG is the base export with other title layers shown (ledger L12).
- The pixels where it differs from the old base PNG by more than 2 levels, dilated by 8 px, are the toggled area.
- When D stays clear of that area, the new numbered PNG is the old one with D's pixels taken from the new base PNG.
- When D touches the toggled area, that file is refused and reported. The base PNG lands regardless.
- The rule pages with numbered PNGs (11_Abilities2-7 and the like) are different pages altogether, so they refuse, as they
  should.

**5.5 A new PSD with no PNG.**
- There is no old PNG or old PSD: the fork renders the whole card, converted (Photoshop's Export As default), RGBA when it
  has transparency, and that render is the PNG.
- G6 (b) is the whole card: the render matches the PSD's composite within the bar, else "PNG not made: the bot's drawing
  of the new PSD does not match its own image (...); export it in Photoshop". No composite: G2.
- A new PSD beside an existing PNG (no old PSD, so no baseline; its PNG pushed first, or a PNG-only card getting its
  PSD): D is where Photoshop's image differs from that PNG past 2 levels. No pixel past the parity threshold: current
  (5.3). Else the whole card is rendered, the PNG takes the render over D only (elsewhere it already shows the PSD), then
  G6 (b) over D (review 7, section 17).
- Since 2026-10-23 both are also held per text and stat box (`png_export._boxes_drawn`; section 18): every visible,
  non-empty type layer drawn at some opacity and every stat row, padded as R is. A box past the floor refuses ("PNG not
  made: the bot's drawing of the Ability text does not match the PSD's own image (34.9% of pixels there); export it in
  Photoshop"), and so does a text box drawn blank: more than 25% of its letters' own pixels (Photoshop's raster of the
  layer, alpha past 0.5) off, and hiding the layer moves no pixel of the render ("PNG not made: the bot draws nothing for
  the Ability text, which the PSD's own image shows; export it in Photoshop").
- The PNG is written beside the PSD, with the PSD's stem in its case.

**5.6 What the bake's gates become.**

| Bake gate (`psd_edit`) | Here |
|---|---|
| `blob` | G1 |
| `recent` | G9 (the same 30 min) |
| `duplicate` | G8 (delta transfer instead of a flat refusal) |
| `parity` (baseline) | G6 (a), over R, the same render and bar; both refuse only past the floor since 2026-10-23 (section 18) |
| `shape`, `region` | G7 (`gate_shape`, `gate_region` on the merge) |
| `idempotence` | G6 (b): the merge against Photoshop's own image of the new PSD (the bake re-renders its saved PSD) |
| `size`, `bytes`, `reparse`, `stories` | n/a (no PSD is written by guard 1) |

**5.7 Size.** The guard's PNGs for the 25 replayed cards it makes with the fork render weigh 52.5 MB (PIL's default
deflate), against 121.3 MB for the authors' Photoshop exports of the same cards (the composite build: 60.7 against 164.6
MB for 31). A bot export adds well under half of what a Photoshop export does. Recompressing
PNGs that authors pushed was rejected: it would add a second blob for every push.

## 6. Guard 2: the standards

**6.1 The rules.**
- Scope "651 cards" = creatures, auxiliary creatures, minions, items, auxiliary items and N.M.E. items (the ledger's 651);
  "cards" = all but rule pages.
- "At 6e50c3ee" counts every card that rule flags in the whole tree today; after waivers is what a push of that card would
  report.
- Lists are in `rules_summary.json`.
- R3 and R4 checked the template layout too (L5: the Ability layer under `Text`; L6: the title in `Text/Name`), 47 and
  150 cards at 6e50c3ee. The owner dropped the layout on 2026-10-23 (the developers' freedom; the bake handles those
  layouts); the rows below are what stays, the real issues.

| Rule | Standard (ledger) | Class | Scope | At 6e50c3ee | After waivers |
|---|---|---|---|---|---|
| R1 | timing tags in a visible Ability layer are set in stoogedings, so they draw as badges (L1, companion §1.2); any letter case | report (a pixel change; the fork draws badges in one colour until colour glyphs) | cards | 129 cards / 226 tags: 85 creatures, 16 items, 9 field, 8 minions, 5 N.M.E., 4 aux, 1 aux creature, 1 status. Top tags (Action) 35, (1GCD) 33, (1CD) 26, (ENDGT) 18 | 129 (B2's backlog, including the cards held until the fork draws colour glyphs) |
| R2 | no inline icon image layers: `Types/` icons below the midline become `[token]` runs (L1; the parser's own rule, stat-bar layers skipped) | report | cards | 16 cards (7 status, 6 field, Tainted_Lazarus, Zess_T_Deluxe, Electric_Dart) and 6 rule pages | 15 (Tainted_Lazarus waived: its header icon) |
| R3 | at most one visible text layer named `Ability`, in any group (`card_standards.rule_ability`; what `psd_edit.ability_layer` bakes), and none hidden with words the parser reads nowhere: several visible are told, and none visible only when a hidden one holds words ("the Ability layer is hidden (the bot reads no ability)"); a hidden empty one (Abyss_Watcher, Warrior_Dee: no ability), one laid out again in visible pieces (`PSDParser._relaid`: Bulborb Larva, Butterfly, Empress Bulblax, Apex_Predator) or none at all (the ability named otherwise; the parser's NO ABILITY LAYER goes out through card-check) is the developer's | report | 651 cards | 6: hidden Shadow_Duelist; several Lit_Bob-omb, Tainted_Lazarus, Miracle_Matter, Twin_Emperors, Warcraftian_Druid; the same at 96d61368 | 0: Shadow_Duelist is waived (in the parser's `ABILITY_EXCLUSIONS` since 2026-01-18: "prints no ability"); the 5 several-Ability cards are waived (Q2) |
| R4 | the title is not lost in a hidden or rasterized `Name` layer (`card_standards.title_problem`: told only when the parser reads no title, `PSDParser.title_layers`, and a layer named Name is hidden or not a text layer; a title named otherwise is layout) | report | 651 cards, **new cards and regressions only** (owner: "Leave the 45 Name layers") | 5: hidden Sonic, Baba, Man_Serpent_Shield; rasterized Tyler (smart object), Pandanoko; the same at 96d61368 | 0 (every existing one is waived; a new PSD, or a push that hides or rasterizes its title's Name layer, is reported) |
| R5 | status words styled by `Ability FX:` layers, not overlay text layers (L7) | report | cards | 23 cards / 29 layers | 15 cards / 18 layers (the 16 chart-less layers on 10 cards waived, decision 2026-10-21): Ancient_Fuelweaver, Skullord, Doctor_Zomboss, Oryx, Pontiff_Sulyvahn, Undyne, Infected_Crossroads, Pananda Peel, Dark_Ritual, Snow_Bunny, Flower_Of_Decay, Freeze_Spell, Crimson_Flag, Pandora's_Box, Standstill |
| R6 | minions, auxiliary items, N.M.E. items: a visible group whose name holds "stars" (L14; the parser's contract, any case) | report (grouping needs `create_group`, which writes no layer id yet) | those 3 folders (165) | 2: Nasty_Patty, Shell_Shield (stars counted from loose layers) | 2 |
| R7 | every font a visible text layer asks for is in `MDW/Fonts` (rule 1, `font_check`) | report, **told in the standards post** (ruling 3, 2026-10-22; font-check.yml tells it in the same channel with its role ping) | cards | 1: Baba (BabaUI) | 0 (deferred by the owner) |
| R8 | Single-line composer on visible Ability and Name layers (L10) | report | cards | 0 | 0 |
| R9 | every type layer's Txt2 story holds its text (`psd_edit.stale_stories`) | **auto-fix** (`txt2_repair`) | cards | 6: Clot_White, PoD08MOAB, PoD09DDT, PoD10BFB, PoD12BAD, Springtrap | 6 (release step 4 and B2 plan them anyway) |
| R10 | the PNG matches the PSD's composite (converted as voted) within the bar | report, **commit comment only** (`TELL_PNG_MISMATCH = False`: beyond the owner's two rules, review 2) | every push that changes a PNG or a PSD with its PNG | 2: Dark_Mind_Eye, Kel | 2 (the owner's restore is pending) |
| R11 | `Text/Ability`, `Text/Name` and the `Ability FX:` layers that key on them are named exactly so (case, outer spaces) | report (the rename is built and gated, 0 pixels; `card_standards.AUTO_FIX` adds it once the owner says yes, review 6) | 651 cards | 0 | 0 |

Since 2026-10-23, R9 flags 0 cards: the 6 were synced by `7b48db05` (Clot White, Springtrap) and B2's `5a9e0fcf` (the
four PoD minions); `txt2_repair --check` over all 904 PSDs at `6a54b4f6`: 0 stale.

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
  - Q2: several-Ability and multi-creature cards (R3);
  - cards in the parser's `ABILITY_EXCLUSIONS` (R3: Shadow_Duelist prints no ability; unattended, 2026-10-23);
  - Tainted_Lazarus's header icon;
  - the 16 chart-less overlays;
  - "leave the Name layers" (R4 existing);
  - BabaUI.
- **Per-card waivers:** none yet (open point, section 15). The design put them in `site/Card Edits.json` `problems_ok` with
  the rule's code (`STD R5`), but `problems_ok` is a bool that the owner approved for parser problems only; it waives no
  standard.
- An issue the push introduced (absent from the old blob) is marked "(new)".

**6.3 The fixes and their gates.**
- **What is fixed:** R9 only (`card_standards.AUTO_FIX`, the owner's "easy fixes only (today only R9 Txt2 sync)"),
  0-pixel by construction, on the pushed blob, after the quiet window. R11's rename is built and gated below, but it is
  told like any other issue until the owner adds it (review 6: it had been fixed, beyond the owner's rule).
- **Never fixed:** rule pages; when `CARD_GUARDS` is not `on`; the cards a hold names.
- **R9:** `txt2_repair.repair` (the fork's `sync_story`) then `txt2_repair.verify`:
  - only the stale stories changed;
  - every story holds its layer's text and styles;
  - every layer record, the other document blocks, the image resources and the stored composite are the old bytes.
- **R11 (once switched on):** the rename sets the Pascal name and `luni`, and renames the `<old> FX: ...` layers with it. Its byte proof is the
  bake's `gate_bytes` with only `name` and `luni` allowed to change.
- **Both:**
  - `gate_reparse`: the bot's parse of the fixed PSD equals the parse of the pushed one, so no data change and no changelog
    event;
  - the composite bytes are unchanged, so guard 1's PNG still holds;
  - `gate_size` within 1%.
- **Repo size:** the fork keeps every untouched byte, so git packs the fixed blob as a delta of a few KB against the pushed
  one.
- **R9 needs the fork tag with `sync_story`** (`cards-20261023`, pinned). Without it R9 is report-only, as the bake's
  `story_sync()` check does.
- **Plainly, for the owner:** "fix the easy issues" means R9 (Txt2 sync) only. Everything else is told, R11 included
  (0 cases in the tree today).
- **Report mode tells the fixable issues and the PNGs the bot would make** (nothing is made or fixed there: "PNG not
  exported with the PSD: export it in Photoshop", "no PNG pushed with the new PSD", "PNG not renamed with the PSD",
  "A_Snack2.png not updated with the PSD"; review 6); in `on` they are made and not told, unless the write does not
  land (7.3).

## 7. Outputs

**7.1 The bot commit** (one per push at most, `git_commit.AUTHOR`, through the API with `GITHUB_TOKEN`):
```
Card guards: 2 PNGs re-exported, 1 PSD fixed (push 7cb4c867)

Items/Active/2 Stars/Dirty_Bed.png: re-exported by the bot's render of the PSD (8,957 px changed) (Zomoroko, 7cb4c867)
Auxiliary/Minions/Clot_White.psd: Txt2 story synced (Name)

Guarded-push: 7cb4c867363c10cf5f562fcdf660ff7f94a154fe
```
- The trailers are their own paragraph (git parses them): one `Recompressed: <path>` per PNG only recompressed (§19),
  then `Guarded-push`. No `Edit-by` trailer. build_site leaves AutoCroissant's commits out of contributions
  (`BOT_AUTHOR`), so the author of the PSD keeps the credit.
- No pickles, no `site/` files, no `TTS/`.

**7.2 The commit comment** (on `after`, posted at check time when `summary.json`'s `comment` count is not 0: a card with
something told, a write, a card's error, or comment-only lines (R10, an R7 issue, "PNG left without its PSD", "deleted; the PSD
stays"...), so a push whose only finding is a PNG that does not match its PSD gets its comment (review 3); an error outside
the cards (a standing MDW/Fonts index error, a manifest or status chart fetch) asks for none, so it is in the job summary
and in a comment that goes up anyway (review 4); saying "The bot
will fix"; after `land` it is rewritten from
`report-landed.md`: "Landed in `<sha>`" per landed card, "Fixed by the bot", for a card left out "Not landed (superseded:
...): nothing above was committed for this card", for a failed commit "Not landed: <why>. Nothing was committed.", and
"After the quiet window: ..." notes, and on a re-run after a landing "already on the branch". A broken Land step still
rewrites it: "The bot could not finish this push". A re-run edits the same comment (the oldest guards comment): the
lookup is one call read whole with the bodies (no pipe into `head`, which can kill `gh` under pipefail; the TOLD mark on any
guards comment counts), tried 3 times (0, 10, 30 s); a re-run or a dispatch that still cannot read it counts the push as
told (reviews 4 and 5: a dispatch's commit may have been told by its push run). The comment's id is out before the edit, so
a failed edit still leaves it to Follow up and the mark. A failed comment call fails the run but never stops Land: its
`if` asks only for Check's success.)
- `## Card guards`, then per card:
  - what changed inside its PSD (7.5), as "- Changed: ...", up to 600 characters;
  - its issues;
  - the guard-1 outcome with D's size, the gates' numbers and the ICC mode;
  - the planned fixes.
- Errors last.
- A dry run writes the same text to the job summary only.

**7.3 The Discord post: exactly one per push, and only when some card has a remaining issue, a refusal, an error or a write that did
not land.**
- **Sent:** after the quiet window (4.5), from `discord-final.json` only (`land` settles it and never raises). Check's
  `discord.json` is never posted: it would tell refusals since withdrawn and miss the writes that did not land. When Land
  ran but settled nothing (killed, timed out), Tell Discord fails and the run-failed message goes out instead. Once sent,
  the comment carries the mark `<!-- card-guards: told on Discord -->`, so a re-run never posts again (nor sends the
  failure alert, review 4; a re-run that fails in Check looks the mark up in the alert step itself, review 5). When the
  Comment step failed, Tell Discord puts up a comment of its own carrying the mark; when that fails too it warns that a
  re-run would post again (review 5).
- **The site build and the TTS save:** a bot commit's run starts them (a `GITHUB_TOKEN` push triggers no workflow). When
  one does not start, the comment says so, the run fails, and the post ends "The bot's commit did not start the site
  build and the TTS save: the site catches up within 6 hours, the TTS save the next day." when it fits in 2,000
  characters (review 5); with nothing else to tell, the failure alert goes out instead.
- **Not sent:** R10 and "current, but outside the bar" (comment only, `TELL_PNG_MISMATCH`);
  "X.png is left without its PSD" and "left at the PSD's old name" (neither a standard nor a PNG refusal; comment only,
  review 3).
- **The login:** the commits API's login of the commit's author; an author with no GitHub account: the login in a
  GitHub noreply email, else the pusher's login (`github.actor`, who pushed the card), never the git author name while the
  API answers (review 3). A dispatch passes no pusher (whoever ran it did not push the card), so there it is the git
  author name (review 4).
- **Format:** one line per card, with what changed inside its PSD (7.5, up to 120 characters, whole phrases) and the
  masked link to the commit that last changed it (beyond the brief's literal line: section 15, item 6):
  `Zomoroko pushed Skullord (Ability text: '(1PT)' -> '(1CD)'; added Ability FX: Blazing) with these issues: timing tag (1GCD) not set in the stoogedings badge font. [7cb4c867](https://github.com/MichaelJSr/TTSCardMaker/commit/7cb4c867...)`
  A new PSD, a PNG alone or a PSD whose push changed nothing but Photoshop's re-save noise has no parenthesis.
- **Not in the post:** planned fixes and re-exports in `on` (they are not issues; report mode tells them, as it makes
  nothing, 6.3) and "PNG still current".
- **Payload:** `{"content": ..., "allowed_mentions": {"parse": []}, "flags": 4}`: nobody is pinged, and the links get no
  preview cards. font-check.yml's role mention is not copied.
- **Long pushes:** to fit 2,000 characters the cards lose their changes first (the last card first), then whole lines
  are dropped, ending "... and N more cards: [details](<commit url>)".
- **The webhook:** the URL comes from `DISCORD_STANDARDS_WEBHOOK` into curl's `--data @file` call. It is never echoed,
  logged or written to a file. A failed call is a `::warning::`, and the comment still goes up.

**7.4 Artifacts.**
- For every card: the planned PNGs and fixed PSDs, and the crops of D (before, after, diff).
- `plan.json`, `report.md`, `discord.json` (its content, no URL); after Land `landed.json`, `report-landed.md`,
  `discord-final.json`.

**7.5 What changed inside the PSD** (`commands/psd_diff.py`, 2026-10-23; NEXT_IDEAS_2026-10-22 section 5, idea 5).
- **For** every card PSD pushed over an older version (cases stale, pair, psd-only): both PSDs are open for the guards
  anyway, so it costs no download; 0.2-0.6 CPU-seconds a card. Plan entries carry the phrases as `changes`.
- **Phrases**, most telling first, the same phrase for several layers merged ("fire x3, machine x2 and Blazing overlay
  hidden"): text edits ("Name text: 'White CLot' -> 'White Clot'", the edited words of a long text with one word of
  context, icons as [name]), layers added or removed (a group with its count), shown or hidden, renamed, moved into or
  out of a group, restacked; a text box widened, narrowed, taller or shorter, a layer moved or resized; font, size,
  colour, leading or alignment on the words that kept their text ("Ability font Perpetua -> stoogedings-Regular on
  '(ENDT)'"); layer styles added, removed, on, off or changed; opacity, fill, blend, clipping; pixels, masks, shapes,
  smart object content and filters, adjustment settings; a text that reads the same: "invisible character removed" (a
  BOM: Cake Mix, Cortez) or "line breaks or spacing changed". An overlay status word (`psd_regions.overlay_words`) is
  named "<word> overlay". The comment and the post escape the phrases' markdown (card text and layer names).
- **Layers pair** by layer id (Photoshop keeps it across saves and renames), else by path; a layer deleted and made again
  under the same group with its name and kind pairs with its old self.
- **Photoshop's re-save noise is left out**, measured on the 465 PSDs of the last 20 PSD pushes (19,801 layer records
  rewritten, most of them by nothing but the save):
  - blocks it rewrites on every save: `shmd` (every layer), `lnsr`, `lyvr`, `cinf`, `lspf`, `lclr`, `fxrp`, `vowv`
    (a version marker newer Photoshops add), the legacy `lrFX` and `PlLd` copies;
  - re-rendered pixels of type, smart object, shape and fill layers (their text, content, transform and paths decide);
  - equal pixels encoded differently (Glaive's Laser Beam mask: 22,269 then 23,144 RLE bytes, the same mask): channels
    whose bytes differ are decoded and compared;
  - a group opened or closed in the Layers panel (`lsct`);
  - smart filter ids and `validAtPosition`, renumbered by a save (Bone's Loans's Puppet Warp: filterID 1002 then 632);
  - descriptors read by meaning: floats rounded, b"Nrml" and b"normal" alike (Tainted Lazarus's 41 styles), and in a
    layer style only the keys both versions write (a newer Photoshop adds `gs99` to old Gradient Overlays: Heavy Mole);
  - text compared as text, style per character, boxes and origins within 1 px.
- **Not read:** the document's own blocks (Txt2, lnk2, cinf, GenI...). Txt2 changes on 49 of 56 plain saves, so a push
  that only rewrote Txt2 stories (7b48db05, Clot White and Springtrap) shows no change.
- **Report only.** A failure leaves the card's line without changes (`card_guards.changes`); nothing is gated on it. The
  comment and the post still go up only when something else is told (the owner's rule, 7.2 and 7.3): a card with changes
  and nothing else is in the job summary (and in the comment when the push has one), not on Discord. Telling every
  push's changes would be a new owner decision.
- `python -m commands.psd_diff OLD.psd NEW.psd` prints the phrases.

## 8. Dry run and switches

- **Dispatch inputs** (4):
  - `dry_run` (boolean, **default true**): the job summary and artifacts only; no comment, no Discord, no commit; off, it
    is refused while a push run of the commit, or of a later one, is unfinished (4.5);
  - `sha`: check this commit as if pushed, against its first parent (e.g. 7cb4c867, dc37f4ad); empty means main's head;
  - `cards`: only these, matched like `Card Authors.txt`;
  - `all`: every card PSD of the tree; standards and R10 only, job summary only; about 813 downloads and 40 min.
- **Repo variable `CARD_GUARDS`** (Settings, Variables; not a secret; any letter case: the job's `if` matches `On` too,
  so the Start step lowercases `MODE` for the shell and `--mode`, review 4):
  - unset or `off`: push runs are skipped at the job's `if`;
  - `report`: push runs check, comment and post (after the quiet window), and never commit;
  - `on`: report and land.
- **Rollout:**
  1. Land the code and the workflow with no variable.
  2. Dry dispatches with predicted results:
     - `sha=7cb4c867` (Zomoroko's cooldowns (5), 29 cards): 29 PSD+PNG pairs, so no guard-1 work; R10 on 29 PNGs, predicted
       0 flags (2.2); R1 flags the cards the badge batch has not reached yet;
     - `sha=dc37f4ad`: God_Tamer and Bubble "PNG still current";
     - `sha=100511d3`: 12_Reactions2 current;
     - one `all` run: section 6's numbers to within the pushes since.
  3. `CARD_GUARDS=report` for 3 days or 10 card pushes; read every post (a forgotten PNG shows as "PNG not exported
     with the PSD"; the comment says what `on` would re-export).
  4. `on`.
  5. Fold into Card sync at its cutover (section 11).
- **`CARD_GUARDS` switches push runs only.** A dispatch with `dry_run` off comments, posts and lands whatever the variable
  says, unset and `report` included: testers in the `report` phase dispatch dry (the default) to stay commit-free.

## 9. Runtime and cost on Actions

| Item | Estimate (Mac numbers from section 2; the runner is ubuntu-latest, 4 vCPU, assumed 1.5-2x slower per core) |
|---|---|
| Setup | blobless checkout of TTSCardMaker (1.1 MB, 2 s, as Card sync measured), sparse bot checkout (2 s), setup-python, the fork from the pip cache (~10 s; a cold build of the git pin ~60-90 s), MDW/Fonts from the cache keyed by its tree sha (215 MB, ~15 s; only the R7 rule needs it): about 40-60 s |
| Per pushed PSD, check only | blob download 1.5-2 s (25-30 MB; font_check measured 1.4 s for 15 MB), open 0.6-1 s, the rules ~1 s, R10's composite compare 0.5 s: about 4 s |
| Per stale card | plus the old PSD and old PNG (~2.5 s), the fork's two renders of the changed layers' box (measured on the 34 replayed commits, section 17: 3.5 CPU-s median, 25 max on the Mac, so about 5-7 s median and 40-50 s for a whole-card box on the runner), the composites and record diff (0.5 s), the PNG write (~1 s): about 10-15 s, up to a minute; a new card renders the whole card once (10-30 s) |
| Typical push (1-3 PSDs) | about 1-2 min; with writes or anything to tell, plus the 30 min sleep and ~20 s to land |
| A batch push (50 PSDs, as the badge batch's) | about 4-6 min with their PNGs; 50 stale PSDs (the PNGs forgotten): about 10-25 min in 2 processes, inside Check's 90 |
| Whole-tree audit (`all`) | 813 downloads, 16 GB: 25-30 min (card-check's measured pass), plus checks in 4 processes (~10 min); the job's `timeout-minutes: 150` (Check 90, Land 60) |
| The 28 GB repo | never checked out: trees only, then blobs by sha for the touched cards. The alternative is the partial clone's lazy fetch of exactly those blobs (`git sparse-checkout set --no-cone <paths>` after a blobless clone), which costs no REST calls, for when the API quota gets tight |
| API (`GITHUB_TOKEN`, 1,000 requests an hour per repo, shared with card-check, font-check and card-sync) | per push: 1 commits read per commit, k blob reads (k = pushed PSDs + 2 per stale card), 1 comment (+1 edit), at land 2 activity/compare reads and n+3 writes for n files |
| Price | free: TTSCardMaker is public (checked 2026-10-22, `visibility: public`), so standard runners cost nothing; the sleeping land step holds one runner for about 31 min on every push with anything to tell or write, report mode included. Were the repo made private, that is about 31 billed minutes a push (2,000 free a month) |
| Storage | a re-export adds one PNG blob, under half a Photoshop export (5.7). A fix adds a delta of a few KB. Nothing is written for the common "PNG still current" case (2.3) |

## 10. Failure modes

| Case | What happens | Recovery |
|---|---|---|
| An unreadable PSD, a failed download, a crashed composite | the card's error told on Discord ("PNG not made" when it needed one) and in the comment; the others go on; check exits 0 so the post and the other writes still go out, then "Fail on card errors" turns the run red (its post went out: no failure alert, and no streak mark, so the next failure is told) | a dispatch with `sha` and `cards` |
| Not checked (the clone lacks the commit, the API is down) | exit 1, red run; "Tell Discord the run failed": one message to the standards channel per failure streak, no ping. The streak is the `card-guards-failure-told` artifact a run leaves when it sent the message or stayed quiet in a told streak: quiet only when the last finished push run left it and is not green (a re-run that passed keeps its first attempt's artifact; review 4: counting red runs hid a push after a red run that had posted its own lines) | re-run the job |
| A gate refuses | report line, Discord, no write | the author exports in Photoshop, or fixes the PSD |
| The author pushes again within 30 min | the older run drops the card's writes and its post line (superseded); the newer run owns it, and G5 retries against the burst anchor | none |
| The head moved during land (409/422) | retry from the new head; cards whose blobs moved stay out | none |
| The land step fails (API 5xx, a requests error such as a reset connection or a GET past its 3 tries, a bad file) | land never raises (review 3: any exception is the error), so landed.json, the comment's "Not landed: ..." and the settled post are written; the post tells each PNG not made and fix not made; red run, the plan in the artifact, nothing half-written (one commit or none) | re-run the job; the idempotent re-check writes only what is still due |
| The Land step is killed (job timeout, a runner lost) | no settled post: Tell Discord fails and the run-failed message goes out (check's early post is never sent) | re-run the job |
| A run cancelled by hand (during Land's wait, say) | no message: the alert is quiet for a cancel before the job's 150-minute timeout (`job.status` cancelled, under 145 minutes since Start's `STARTED`), and leaves no streak mark; the comment says "The bot could not finish this push" (review 6) | re-run it, or dispatch |
| The comment call fails | red run; Land, the post and the writes go on (Land waits on Check only); a failed POST: Tell Discord's own comment carries the told mark, or it warns that a re-run would post again; a failed edit: the id is already out, so the mark is still set; with nothing to tell, the alert says the push "was checked, but its run failed"; the lookup is tried 3 times, then on a re-run or a dispatch counts as told (never a second post; the step's error says the push may be untold) | re-run once GitHub answers |
| The site build or the TTS save does not start after a bot commit | red run; the comment says so, and the post gets a line for it when it fits (with nothing else to tell, the alert goes out); site.yml catches up within 6 hours, the TTS save the next day | re-run the job (it finds the writes landed and starts both again; review 6), run them by hand, or wait |
| A re-run (or a landing dispatch) cannot read the branch head | an earlier run may have landed the writes, so none is told as "not made" (Land's `--rerun`): the post keeps the card's other lines, the comment says "Nothing was committed by this run; an earlier run may have landed it" (review 6; a first attempt that cannot read it tells "PNG not made" as before) | re-run once GitHub answers |
| The author pushes the PNG first and the PSD a few minutes later | the PSD's push is a stale case whose PNG already shows the PSD: "PNG already shows the pushed PSD", nothing written (review 6) | none |
| Webhook down or the secret missing | `::warning::`; the comment stands | none |
| The fork pin missing or broken | `--help` fails before any card: red, nothing posted | fix the pin; card-edits.yml shares it |
| A PSD saved without Maximize Compatibility | G2 refusal for guard 1; guard 2 still runs | save with it on |
| A PSD written by a script without updating the composite | the render shows the new layers, the stale composite does not: G6 (b) refuses past the floor (a hidden title: the whole change disagrees); since 2026-10-23 a smaller disagreement is made and told (section 18) | export in Photoshop |
| The fork draws a changed area otherwise than Photoshop (a layer effect such as a drop shadow's edge, an outer glow, a gradient overlay) | since 2026-10-23 the PNG is made and the miss told in the comment and the bot commit ("made with a pixel difference ..."); past 25% bad G6 (a) or (b) refuses: no PNG, "export it in Photoshop" (section 18) | a fork fix; export in Photoshop when the miss shows |
| A wrong re-export that passed the gates | visible on the site; the commit lists it | `git revert` of the bot commit (PNGs only), then a fix |
| GitHub schedules late | n/a: push runs are not scheduled; only the Card sync fold depends on the cron | none |
| A burst of pushes | each its own run, so no report is lost (4.4); landing commits serialize by retry | none |
| A run re-run by hand | edits its earlier comment; posts to Discord only if the first run had not (the comment's told mark); after a landing, the bot's own blobs count as held: "already on the branch", nothing written or told again; its failure alert stays quiet once the push was told (the mark, looked up by the alert itself when Check failed and the Comment step did not run, or an earlier attempt's streak artifact); its quiet window counts from the push, not from the re-run | none |
| A dispatch with `dry_run` off while the commit's push run waits | refused before Check: a red dispatch run, nothing commented, posted or landed | dispatch again once the push run has finished, or dry |
| `CARD_GUARDS` typed `On` or `Report` | the job's `if` matches it in any case; Start lowercases `MODE` (`--mode` takes any case since review 4; Land's `[ "$MODE" = report ]` test is exact, and `land` refuses to commit a plan not checked with `--mode on`) | none |
| A bot commit (bake, Card sync, another guard run) changes a card's PSD in the window | no run follows it, so the card's lines are still told; its PSD fix is told as not made | none |

## 11. Folding into Card sync (`DATA_BUILD_ACTION.md` §6)

Card sync (every 30 minutes, at most one main commit) is the guard's natural landing slot. Its interval is the quiet window, so
the sleeping step goes.
- **Push run (stays).** `check` and the comment at push time. The plan goes to the artifact only. Without the sleeping
  step the one Discord post goes out at check time again (as before review 2), or Card sync's stage 0 posts it: to decide at
  the fold.
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
    timeout-minutes: 150
    env:   # github, inputs, vars... only: OUT ($RUNNER_TEMP/guards) is set in the first step
      MODE: ${{ github.event_name == 'push' && (vars.CARD_GUARDS || 'off') || ((inputs.dry_run || inputs.all) && 'dry' || 'on') }}   # off | report | on | dry
    steps:
      - uses: actions/checkout@v7                       # trees only: the diff, the pairing, no blobs
        with: {fetch-depth: 0, filter: "blob:none", sparse-checkout: "/.github/", sparse-checkout-cone-mode: false}
      - name: No push run in its window                 # a dispatch with dry_run off: refused while a push run of the commit (or a later one) is unfinished
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
      - name: Comment on the commit                     # MODE report/on; a re-run edits its own comment
      - name: Land                                      # !cancelled(), Check passed, MODE report (--dry-run, something to tell)/on (or a
                                                        # write): sleep to the quiet mark, recent check, one commit, discord-final.json;
                                                        # --rerun on a re-run or a dispatch
      - name: Follow up                                 # always(): a commit, or a re-run finding it landed: gh workflow run site.yml,
                                                        # tts-save.yml (not started: said in the comment and the post, red); the
                                                        # comment rewritten
      - name: Tell Discord                              # always(), Check passed, not told before: discord-final.json only; URL never printed
        env: {WEBHOOK: "${{ secrets.DISCORD_STANDARDS_WEBHOOK }}"}
        run: curl -sS -o /dev/null -f -H "Content-Type: application/json" --data @"$OUT/discord-final.json" "$WEBHOOK" || echo "::warning::..."
      - name: Fail on card errors                       # summary.json card_errors > 0: red (the errors were told)
      - name: Tell Discord the run failed               # a failed push run whose post did not go out, not told before (the comment's
                                                        # mark, looked up here on a re-run), not cancelled by hand: once per failure
                                                        # streak (the last push run's STREAK artifact)
      - uses: actions/upload-artifact@v4                # STREAK (card-guards-failure-told), when the alert told or stayed quiet in a streak
      - uses: actions/upload-artifact@v4                # always()
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
- **Composite-first previews.** The bake keeps Photoshop's pixels outside its region already. A website or Discord preview of
  a pushed PSD could show its composite at once, before any PNG exists (a preview only: guard 1's PNG is the fork's render,
  ruling 1).
- **Fork fixes that would lift guard 1's refusals** (section 17): the drop shadow's outer edge, the outer glow and the
  gradient overlay on type layers drawn as Photoshop draws them.
- **The PNG encoder:** the bot's PNGs are pixel-equal and about 1/3 the size of Photoshop 2023's (5.7). A standards rule could
  tell authors their export settings bloat the repo (report only).
- **R10 run in reverse catches "PNG pushed, PSD forgotten".** Commit 07589cc7 re-exported Asteroid_Field.png without touching
  its PSD (the L16 stale export). The new PNG matches the composite (max 1 level), so R10 passes it.

## 15. Open points

**For the owner to confirm before `CARD_GUARDS=on`** (each goes beyond the literal brief; everything else follows it):
1. **Decided (ruling 1, 2026-10-22): the PNG is the bake's fork render**, the composite only a cross-check that can
   refuse (section 17). Was: the stored composite written, the fork only a referee.
2. **The Discord post waits out the 30-minute quiet window, in report mode too.** font-check.yml, the model the owner named
   ("like we do for fonts"), posts at once. The wait lets a refusal be withdrawn when the author pushes the PNG, leaves a
   card pushed again to the later push, and tells the writes that did not land. Cost: a runner held about 31 minutes per
   push with anything to tell, free while TTSCardMaker is public (it is). The alternative is the post at check time.
3. **"Fix the easy issues" is R9 only** (6.3): the pin `cards-20261023` has `sync_story`. R11's exact-name rename is
   built and gated but told, not fixed, until the owner says yes (`card_standards.AUTO_FIX`; 0 cases in the tree).
   Report mode tells everything, the PNGs the bot would make included.
4. **Decided (rulings 2 and 3, 2026-10-22): fonts in the standards channel.** font-check.yml posts there
   (`DISCORD_STANDARDS_WEBHOOK`, its role ping and wording kept) and the guard tells R7 in its own post too, so a missing
   font is told twice in that channel: at once by font-check (with the role ping) and after the quiet window by the guard
   (no ping). card-check.yml posts in the standards channel too, no ping (owner, 2026-10-23); the parser's notes stay card-check's.
5. **The login** in "<login> pushed <card>": the commit author's GitHub login; for an author with no GitHub account, the
   pusher's login (7.3).
6. **The commit link.** The brief's line is "<GitHub login> pushed <card> with these issues: ..."; each line also ends with
   a masked link to the card's commit (`[7cb4c867](url)`), and a post too long for Discord ends "... and N more cards:
   [details](url)". Nobody is pinged and previews are off. Dropping the link is one change to `card_guards.LINE` (and the
   "more" line).

- **Questions on the rulings (2026-10-22, section 17)** for the owner:
  1. A missing font is told twice in the standards channel (font-check at once with its role ping, the guard after the
     window without): keep both, or should the guard leave R7 to font-check (or font-check stop posting on pushes)?
  2. The fork render refuses 9 of the 34 replayed commits (25 made), where the composite made all 34: accept, or start fork
     fixes for the drop shadow's edge, the outer glow and the gradient overlay first?
  3. Photoshop's image is used beyond refusals in one place: a PNG pushed before its PSD counts as "current" when it
     matches Photoshop's image where the fork draws the change otherwise (4 of 34; else "PNG not made" would be told for a
     PNG already exported). Keep?
  4. A PSD saved without Maximize Compatibility refuses (G2): nothing could cross-check the render. Keep, or render it
     with the bake's gates alone?
  5. "Parity vs the old PNG outside the PSD's changed layers" is built as the old PNG against Photoshop's image of the
     pushed PSD there (the merge keeps the old PNG outside R). The alternative, a whole-card fork render compared with the
     old PNG outside R, costs a full render (2.5, 5-14 CPU-s on the Mac) and refuses every card the fork draws otherwise
     anywhere.
- The Environment wait timer (4.5) instead of a sleeping runner: the owner's one-time setting. Until then a push with writes
  holds a runner up to about 31 min (free on a public repo).
- Per-card standards waivers: `problems_ok` waives no standard since the review (16, change 1). Today it changes nothing (Pix,
  Tainted Lazarus and Twin Emperors report nothing, every issue under a class waiver). A per-rule field (e.g. `standards_ok:
  ["R5"]` in `CardEdits.TYPES`, the editor and build_site) is the owner's call.
- **R3 and R4's real issues, before `CARD_GUARDS=report`** (an unattended call on 2026-10-23, after the owner dropped their
  template layout; section 6.1): keep them as reports? Shadow_Duelist (a hidden Ability layer the card prints nowhere)
  is waived as one of the parser's `ABILITY_EXCLUSIONS`, so R3 tells only a card with several visible Ability layers or
  a hidden one with unread words; R4 tells only new cards and regressions. Dropping them is the R3/R4 block of
  `card_standards.check`.
- Choices beyond the literal brief, to confirm before `CARD_GUARDS=on`: numbered PNGs (A_Snack2.png) are re-made by
  copying the changed pixels into them (G8); R10 now stays in the comment (`TELL_PNG_MISMATCH = True` would post it: 0
  flags in the last 10 pushes, 2 cards in the tree, Dark_Mind_Eye and Kel, whose restore is pending).
- The 57 cards within the bar but past 2 levels (2.1). With the fork render only the changed layers' box is replaced, so
  their other pixels stay as exported; G6 (b) still compares the change with Photoshop's image.

## 16. Build (2026-10-22)

**What exists** (nothing pushed, dispatched or posted; the Desktop clone was only read):
- AutoCroissant `work/push-guards`: `commands/png_export.py` (guard 1), `commands/card_standards.py` (guard 2),
  `commands/card_guards.py` (the driver: `check`, `land`, `--whole-tree`, `--local`), tests `test_png_export.py` (17),
  `test_card_standards.py` (24 with the parametrized ones), `test_card_guards.py` (31 after the second review). work/r5-bot is merged in for `icc_vote` (one conflict:
  `PARITY_MEAN_MAX` stays 2.0, the owner's value, with r5-bot's ICC constants added).
- TTSCardMaker `work/push-guards`: `.github/workflows/card-guards.yml` (section 12's sketch made whole). Its shape is pinned by
  `test_card_guards.test_workflow_shape` (set `CARD_GUARDS_YML` to the file until it is on the clone's main).
- Run: `python -m commands.card_guards check --clone <card repo clone> --after <sha> [--before <sha>] --out <dir>` writes
  `report.md`, `discord.json`, `plan.json`, `summary.json` and the artifacts (`files/`, `crops/`); `land --plan <dir>/plan.json`
  (commits only a plan checked with `--mode on`; otherwise pass `--dry-run` or `--local <clone>`).
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
1. Per-card waivers: none. The build first read `problems_ok: true` as a blanket waiver of R1-R8; the owner approved that flag
   for parser problems only, so the review removed it (it changed nothing today: the three cards' issues are all
   class-waived, `test_problems_ok_cards_are_waived_by_class_only`). Open point in 15.
2. The Q2 class waiver includes Pix (a multi-creature card): R3 after waivers is 41, not 42.
3. R11's reparse gate refuses a rename that changes the bot's reading. The parser misses an `ability ` layer (trailing space),
   so renaming it changes the card's data: it stays a report. The bot never changes data on its own.
4. G6 also covers a tool save that moved no composite pixel but changed layers (a script that saved without a new composite):
   the referee renders the changed layers' box and refuses when the composite is stale (test on Mini Doomer).
5. "PNG still current", but the PNG and the composite were already outside the bar: told ("does not match the PSD, though this
   push moved no pixel"). Dark_Mind_Eye and Kel get this line when their PSD is pushed. A new PSD beside its PNG says "the PSD
   is new beside it", and gives the mean difference when no pixel passes the threshold.
6. A new PSD beside an existing PNG takes the ICC mode from that PNG's own vote.
7. A PSD pushed with its PNG deleted: the standards only, no PNG made (the deletion is the author's).
8. A rename (same blob) copies the old PNG to the new name and tells that the old PNG is left at the old name.
9. PNGs under `Types/`, `site/` or `TTS/` are not card PNGs (the paths filter still starts a run, which finds nothing).
10. The fixes run at check time, so dry and report runs carry the fixed PSD in the artifact; `land` only re-checks the blobs.
11. The commit comment goes up only when a card has something told, a write or an error (`summary.json`); the job summary
    always has the report.
12. The fork pin is `cards-20261023` (with `sync_story`), as card-edits.yml and card-sync.yml; R9 is a fix in `on`. Replay
    after the release (2026-10-22, main 7b48db05, blobless clone, `--repo`): of the last 10 pushes the Txt2 repair push
    (2 PSDs without PNGs) checks both as current (0 pixels moved, nothing re-exported, no R9) and tells one R1 issue
    (Springtrap's (OD) is set in the card's font); the pin push and 3 others are skipped by the paths filter, 3 bot pushes
    by the job's `if`; 07589cc7 tells God Tamer only, 38c55bd3 nothing. Evidence:
    `~/Documents/Tools/wt/scratch/guards-final/replay_on.log`. The replay found pool workers fetching blobs without the
    token (spawned processes skip main()); `_init_worker` now calls `token_from_env`. Earlier, measured at `cards-20261022`
    (0302b08b, review 2026-10-22): the 34 replayed commits give the same outcome, writer, changed pixels and truth max as at
    e0ea34f6 (34 of 34); the 3 bot bakes pass the referee (PoD09DDT and PoD10BFB max 0, Springtrap max 32 on 0.002%);
    test_png_export 17 of 17. Evidence:
    `~/Documents/Tools/wt/scratch/guards-fix/replay_at_pin.jsonl`.

**Review fixes (2026-10-22):**
- card-guards.yml set `OUT: ${{ runner.temp }}/guards` in the job's `env`, where `runner` is not allowed: GitHub rejects the
  whole file, so every push touching a PSD or PNG would show a failed run even with `CARD_GUARDS` unset. OUT is set in the
  Start step; `actionlint` passes (`test_workflow_lints`).
- A pair (PSD and PNG pushed together) planned its fixes against "no PNG", so `land` always dropped them as superseded.
  `_expect` now takes the pushed PNG's blob (`test_a_psd_fix_lands_beside_its_pushed_png`, and a real R11 fix of Mini Doomer
  landed end to end, `test_a_real_r11_fix_lands_for_a_pair`).
- The comment said "Fixed by the bot" before landing and was never corrected for cards left out; see 7.2.
- Bot commits by author only, a person's change in the range wins (4.3); a pushed PNG pairs with the PSD of its exact case
  first, and one that matches PSDs differing only in case is noted and not checked.
- The two png_export tests that need the fork's renderer skip without it; the change-control skill names card-guards.yml and
  its `SPARSE_FILES`; timeouts 120/90/60.

**Review fixes, second pass (2026-10-22):**
- A PSD added with its PNG (case pair, no old blob) got R4 waived as an existing card; `card_guards.added()` gives every
  added PSD the new-card rules (removed with R4 by 2ada357, back with R4's real issues the same day) (`test_a_new_psd_pushed_with_its_png_gets_the_new_card_rules`, measured first with God_Tamer's
  layers on a scratch repo).
- `card_standards.fix` named its temporaries by card key in a shared folder: two cards of one key in different folders,
  fixed at once in the process pool, could swap bytes. Each call now works in its own `mkdtemp` folder
  (`test_two_cards_of_one_key_fix_in_separate_files`).
- The Discord post moved from check time to after the quiet window (4.5, 7.3), and `land` never raises: a failed commit
  writes `landed.json`, `report-landed.md` (Not landed) and `discord-final.json` (what was not made), then exits 1. Follow up
  and Tell Discord run `always()`. A fix that does not land is told; a PSD fix rests on its PSD alone.
- `check` exits 1 when it cannot check, and a failed push run tells the standards channel once per failure streak.
- A re-run edits its comment and does not post twice (the comment's told mark); the steps' shell is run against a fake `gh`
  and `curl` in `test_workflow_posts_once_and_a_rerun_edits_its_comment`.
- Two quick PSD pushes without a PNG: G5 retries against the burst anchor (5.2), so the PNG is still made
  (`test_a_burst_of_psd_pushes_still_gets_its_png`; 4 hours apart it refuses as before).
- R10 and "current, but outside the bar" stay in the comment (`TELL_PNG_MISMATCH`); report mode tells the fixable issues.

**Review fixes, third pass (2026-10-22):**
- `land` caught only `Transient` and `Refused`; a requests error (the commit POSTs, a GET past its 3 tries) escaped, no
  landed.json or settled post was written, and Tell Discord fell back to check's early post (refusals not withdrawn, the
  writes that never landed untold, the failure alert suppressed). It now catches every exception as the error, and Tell
  Discord posts `discord-final.json` only (`test_land_never_raises_and_still_settles_the_post`,
  `test_workflow_tells_only_the_settled_post`).
- A failed comment call skipped Land (no status function: an implicit `success()`); Land now runs on `!cancelled()` and
  Check's success.
- A card's error is told on Discord and turns the run red ("Fail on card errors"; `test_a_card_error_is_told_and_counted`).
- R10 alone (a PNG-only push not matching its PSD) never reached the comment: `summary.json` counts `comment` separately and
  the Comment step gates on it (`test_a_png_mismatch_alone_still_gets_its_comment`).
- A dispatch with `dry_run` off counted its own push as "a later push" and dropped every write; it now lands on the blob
  check alone (`test_a_dispatch_does_not_count_its_own_push_as_a_later_one`).
- A re-run after a landing rewrote the comment to "Not landed"; the bot's own blobs now count as held
  (`test_a_rerun_after_a_landing_says_landed_not_superseded`).
- A card's lines were dropped when a bot commit changed its PSD in the window; they now stand
  (`test_a_bot_change_in_the_window_keeps_the_lines`, `test_bot_changed_in_a_clone`).
- Head blobs are looked up in any letter case (`test_a_png_pushed_in_another_case_counts`).
- "Left without its PSD" and "left at the PSD's old name" are comment only, like R10.
- The login fallback is the pusher's GitHub login, not the git author name (`test_login_falls_back_to_a_github_login`).
- Section 9's timeout is 150, as card-guards.yml has it; the owner's items to confirm are listed at the top of section 15;
  the run-and-operate skill has a card-guards section.

**Review fixes, fourth pass (2026-10-22):**
- The failure alert's streak counted red runs, but four kinds of red run (a card error, a failed comment call, a failed
  commit, a failed follow-up dispatch) post their own lines and never send the alert: the next run that failed in Check
  was quiet and its push went unguarded with nobody told. The streak is now the `card-guards-failure-told` artifact, left
  by a run that sent the alert or stayed quiet inside a told streak, on a last push run that is not green (a re-run that
  passed keeps its first attempt's artifact; `test_workflow_failure_streak_counts_told_runs_only`, with the steps' `--jq`
  run by the real jq).
- A re-run of a red run whose push was told (a card error, say) sent "was not guarded" as a second post: the alert now
  skips a push the comment marks as told, and a re-run of a run that failed in Check finds its earlier attempt's artifact.
- The alert says "was checked, but its run failed" once Check passed (a failed comment call with nothing to tell).
- The Comment step piped the lookup into `head -n 1`: under pipefail a killed `gh` read as "no comment", so a re-run
  posted a new comment and told Discord again. The lookup is read whole; a re-run that cannot read it, or the earlier
  comment, counts the push as told (`test_workflow_comment_lookup_never_posts_twice`).
- A numbered PNG's refusal (G8) is withdrawn when the author pushes that PNG in the window, as the main PNG's is
  (`test_a_numbered_png_pushed_in_the_window_withdraws_its_refusal`).
- `summary.json`'s `comment` counted the context errors, so a standing font-index error would comment on every push; it
  counts cards only (`test_context_errors_alone_ask_no_comment`).
- A dispatch with `dry_run` off on a commit whose push run was still waiting posted and landed first, then the push run
  posted again: it is refused while such a run is unfinished
  (`test_workflow_refuses_a_landing_dispatch_while_a_push_run_waits`).
- `CARD_GUARDS=On` passed the job's `if` (expressions ignore case) and then failed argparse on every push: Start
  lowercases `MODE` (so Land's exact `--dry-run` test holds for `Report`), and `--mode` takes any case
  (`test_workflow_lowercases_card_guards`, `test_mode_takes_any_case`).
- A dispatch named an author with no GitHub account after the dispatcher: a dispatch passes no `--author`, so the git
  author name stands.
- The commit link in each Discord line is listed for the owner (15, item 6).
- `test_card_standards.test_r9_story_sync_on_clot_white` read the live clone, whose Clot White main's Txt2 repair
  (7b48db05) synced: it reads the blob at d8163103, before the repair.

**Review fixes, fifth pass (2026-10-22):**
- A re-run that failed in Check after an earlier attempt had told the push (the TOLD mark set, then red on a card error or
  a failed follow-up) sent "the push ... was not guarded" as a second message: the Comment step does not run after a
  failed Check, so the alert's `told` was empty. On a re-run the alert now looks up the comment's mark itself, with the
  Comment step's lookup and fail-safe (`test_a_rerun_that_fails_in_check_finds_the_told_mark`; the review's simulation,
  re-run with the mark seeded, stays quiet: `~/Documents/Tools/wt/scratch/guards-final/r5/sim.py`).
- The lookup is one call with the bodies (the mark on any guards comment counts), tried 3 times; a dispatch that cannot
  read it counts as told like a re-run (its commit's push run may have posted); the error says the push may be untold
  (`test_workflow_comment_lookup_never_posts_twice`).
- A failed comment POST left the post unmarked without a word: Tell Discord puts up a comment of its own with the mark, or
  warns; a failed edit still hands its id on (`test_workflow_tell_marks_a_comment_of_its_own`).
- A site build or TTS save that did not start after the bot's commit was told nowhere once the post went out: the comment
  and the post say so (`test_workflow_tells_a_site_build_that_did_not_start`).
- A re-run counts the quiet window from the push (`test_workflow_a_rerun_counts_the_window_from_the_push`).
- `land` refuses a plan not checked with `--mode on` unless `--dry-run` or `--local`: it reads only the plan's mode, then
  exits 2 with nothing written (on the Mac its token is config.py's: `check` then a bare `land` would have committed to
  TTSCardMaker; `test_land_commits_only_an_on_plan`).
- The alert's text says "until a push run succeeds or posts its issues" (a red run that posted its own lines ends the
  streak too); section 8 says `CARD_GUARDS` does not bind a dispatch; section 10's case row is exact.
- The `card-guards` artifact upload overwrites, as the streak's does: an artifact name is taken once per run, so a re-run's
  upload would otherwise fail with a conflict.

**Review fixes, sixth pass (2026-10-22):**
- Report mode never told a forgotten PNG: `told()` added only a refused export, so a PSD pushed without its PNG (export
  passing) told nothing, Land held a runner 31 minutes and posted nothing, and the comment said "PNG re-exported" although
  report mode commits nothing. Report mode now tells the PNG the bot would make ("PNG not exported with the PSD: export it
  in Photoshop"; a new card's, a rename's, a numbered PNG's), withdraws it when the author pushes the PNG in the window,
  and the comment says "PNG would be re-exported ... with CARD_GUARDS=on". Land runs in report mode only with something to
  tell (`test_report_mode_tells_the_png_it_would_make`, `test_check_end_to_end`).
- A re-run whose Land could not read the branch head told every write as "PNG not made: the bot's commit failed" and
  rewrote the comment to "nothing above was committed", though attempt 1 had committed them. Land's `--rerun` (a re-run
  or a dispatch) tells no write as not made when the head is unread, and the comment says this run committed nothing
  (`test_a_rerun_that_cannot_read_the_branch_tells_no_write`; the review's `guards-final/rv6/sim.py` case).
- A re-run never retried a site build or TTS save that did not start: Follow up starts them again when a re-run finds
  the writes already landed (`test_workflow_land_and_a_rerun_after_a_landing`).
- Cancelling a run by hand sent "was checked, but its run failed": the alert is quiet for a cancel before the job's
  timeout (`test_workflow_a_cancel_by_hand_sends_no_alert`).
- A PNG pushed first and its PSD a few minutes later in a push of its own: the bot re-exported the composite over the
  author's PNG (G5 passed, the PNG already held the change). `png_export.export` now finds the PNG current when it
  already shows the pushed PSD (`test_a_png_pushed_before_its_psd_is_current`, Photoshop-saved and tool-saved). Replayed
  on the 34 commits of 2.2: with the PNG forgotten, 34 of 34 still re-export; with the author's PNG pushed first, 32 of
  34 are current, and the 2 others are the bot bakes PoD09DDT and PoD10BFB (their merged PNG differs from the merged
  preview, 2.2), re-exported as before; bot commits are never guarded anyway. Evidence:
  `~/Documents/Tools/wt/scratch/guards-final/rv6fix/replay34.log`, `png_first.log`.
- R11 was fixed automatically, beyond the owner's "easy fixes only (today only R9 Txt2 sync)": it is told until the owner
  says yes (`card_standards.AUTO_FIX = ("R9",)`; `test_r11_is_told_until_the_owner_says_yes`; the rename's own tests and
  `test_a_real_r11_fix_lands_for_a_pair` switch it on).
- Section 12's sketch has the real MODE line (an `all` dispatch is always dry).
- Not this branch: `tests/test_metadata_edits.py::test_live_aliases_find_the_same_card_image_after_the_migration` fails on
  live data (KeyError 'grape juice.png'); the file is not in main...work/push-guards.

## 17. The owner's rulings, built (2026-10-22 ~23:00)

Nothing pushed, dispatched or posted; the Desktop clone was only read. AutoCroissant `work/push-guards` and TTSCardMaker
`work/push-guards` (origin/main merged into both first). Evidence and scripts:
`~/Documents/Tools/wt/scratch/guards-rulings/` (`replay34.py` / `replay34.jsonl`: the built code on the 34 commits;
`proto.py`, `why.py` and `why/`: the per-card diagnosis with crops; `png_first4.py`).

**Ruling 1: the PNG is a fork render** (5.1, 5.2, 5.5, 5.6).
- `png_export.export` renders the changed layers' box R of the old and the new PSD with `psd_edit.render_region` (the fork
  pin `cards-20261023`, the card's `icc_vote` mode) and merges the new render into the old PNG with `psd_edit.merge_png`;
  `gate_shape` and `gate_region` follow. The bake's parity gate is G6 (a). `export_new` writes the fork's whole-card render.
- The stored composite never reaches a PNG. It refuses only: G2 (none to check against), G5 (the old PNG against
  Photoshop's image of the new PSD outside R; inside R when the baseline fails and the old composite disagrees too), G6 (b)
  (the new PNG against Photoshop's image wherever either moved), and it keeps a PNG pushed first "current" (5.3).
- Every refusal line starts "PNG not made:" and ends "export it in Photoshop" (`png_export.EXPORT`); it is told on Discord
  and in the comment as before. A G5 or G6 refusal is retried against the burst anchor.
- The referee (`png_export.referee`, `run_referee`) is gone: every PSD is rendered.

**Measured on the 34 replayed commits (2.2)** with the built code, the PNG withheld (`replay34.jsonl`; the Mac under load
115-138 on 24 cores, numpy single-threaded, 2 processes as the Action's pool):

| | Cards |
|---|---|
| PNG made (re-exported) | 25 of 34: 24 of the 31 Photoshop saves, 1 of the 3 bot bakes (Springtrap) |
| Against the author's own export | all 25 within the parity bar; 99.957% to 100% of pixels within 2 levels; max 0-2 on 6, under 24 on 23, worst 68 (Dirty Bed, a few pixels) and 32 (Springtrap). The composite build: 30 of 31 within 2 levels, worst 4 |
| Refused | 9 (below) |
| The author's PNG pushed first (in the tree) | 34 of 34 current, nothing told (4 of them by Photoshop's image, the change being one the fork draws otherwise) |
| Render CPU (both boxes) | median 3.5 s, max 25.2 s (Zero Two: a 2.6-megapixel box); the whole export median 5.7 s, 90th percentile 20.8 s, max 27.9 s; opening the PSDs 0.2-1.0 s |
| On the runner (ubuntu-latest, 4 vCPU, assumed 1.5-2x slower a core) | about 9-11 s CPU a stale card median, 40-55 s for a whole-card box |
| Bytes | 52.5 MB for the 25 PNGs, against 121.3 MB for the authors' exports |

Why the 9 refuse (diagnosed with `why.py`, the crops in `why/`):

| Card (commit) | Gate | Cause |
|---|---|---|
| Defcon Panda (1b4f36a0) | G6 (a), baseline 1.35% bad | the Ability layer's drop shadow: the fork draws its outer edge otherwise (a thin ring round every glyph) |
| Clock (a90761aa) | G6 (b), 3.6% bad | the same drop-shadow edge on the Ability text, and the changed smart object |
| The Radiance (bb44ec9e) | G6 (b), mean 3.84 | the title's outer glow drawn otherwise (14 records changed, the box is the whole card) |
| Pandamonium (38c55bd3) | G6 (b), mean 3.15 | the Ability text's gradient overlay, up to 9 levels off on every glyph pixel |
| Restrained (5c8cb87b) | G5, where changed (7.1% bad) | the old PNG was stale: the old PSD (its composite and the render) reads "(BOT) -1 Action", the old PNG "-1 Action" |
| 20 Creature Types, 11 Abilities5 (100511d3) | G6, "changed where the bot sees no layer change" | a type layer whose only changed block was `TySh`: its stored raster stayed, Photoshop's image moved (2,612 and 3,487 px outside R) |
| PoD09DDT (705e3d34), PoD10BFB (c9fc1bbd) | the same | bot bakes (never guarded): the bake patched the stored preview beyond the changed records' box |

- **Not the one-colour badge glyphs.** The fork draws a type layer from the raster Photoshop stored, so badges come out in
  Photoshop's colours: 15 of the 34 changed a type layer set in stoogedings, and every refusal among them has a cause
  above. The colour-glyph fork matters for the bake's own text, not here.
- So today about 1 in 4 forgotten PNGs would be "PNG not made: export it in Photoshop", where the composite build made
  all 34. Fork fixes for the drop shadow's edge, the outer glow and the gradient overlay would lift 4 of the 9.

**Ruling 2: font-check.yml posts to the standards channel.** Its Tell Discord step reads `DISCORD_STANDARDS_WEBHOOK`; the
role ping and the wording are `font_check.discord_payload`'s, unchanged. card-check.yml moved there too on 2026-10-23, no ping.
`test_font_check.test_workflow_shape` pins the secret (`FONT_CHECK_YML` points it at the branch's file until it lands).

**Ruling 3: R7 is told in the guard's post.** `card_standards.rule_fonts` no longer marks R7 comment-only; BabaUI stays
waived. A missing font is therefore told twice in the standards channel: by font-check at once (role ping) and by the
guard after the quiet window (no ping).

**Tests** (fork venv; `CARD_GUARDS_YML` and `FONT_CHECK_YML` at the TTSCardMaker branch's files):
`test_the_png_is_the_forks_render_merged_into_the_old_png`, `test_a_new_card_is_rendered_and_cross_checked`,
`test_every_refusal_says_export_it_in_photoshop`, `test_the_refusals_of_the_replay` (Restrained G5, Defcon Panda G6 (a),
20 Creature Types G6, Defcon Panda's PNG pushed first current), `test_font_issues_are_told_in_the_standards_post`,
`test_r7_is_told_in_the_standards_post`; the real-card tests re-pinned to the render's numbers. The standards tests and
R10's read the cards at 6e50c3ee from the clone's history: the Desktop clone's later standardization commits changed
Dirty Bed, Blindness and Dark Mind Eye.

**Review fix (2026-10-23): a new PSD beside its PNG** (5.3, 5.5). `_beside` (a PSD added where its PNG is already in the
tree: the PNG pushed first, or a PNG-only card getting its PSD) pasted the fork's render wherever the render differed
from the PNG, so the fork's own drawing errors (drop shadow edges, glows) were the change, and a PNG that already was
the author's export was refused "PNG not made: the bot's drawing of your change does not match the PSD's own image", or
overwritten with the fork's pixels when within the bar. The composite build had found these current. Now D (where
Photoshop's image differs from the PNG past 2 levels) decides: no pixel past the parity threshold is current with
nothing rendered, and otherwise the PNG takes the render over D only, then G6 (b) over D. The review found 7 of 8 live
cards whose R10 passes refused this way (Defcon Panda, Mini Doomer, Militron, Poison Circle, Zero, Dirty Bed, Niko).
Measured with the fixed code (fork `cards-20261022b`, `scratch/guards-rulings/beside/`): those 7 are current, and so are
32 of the 34 replayed new PSDs beside the author's own PNG. Final Dark Mind Eye is one of the 32 (45 px of export noise,
max 4). The 2 left are the bot bakes PoD09DDT and PoD10BFB, refused G6 (their PNG differs from the patched preview by up
to 49; bot commits are never guarded). Beside the old PNG (a PNG-only card getting a newer PSD), 26 of 34 are
re-exported, all within the bar of the author's export (max 0-55, bad at most 0.0004%), and 8 refuse G6 for the causes
of the table above (Defcon Panda, The Radiance, Clock, Pandamonium 38c55bd3, 20 Creature Types, 11 Abilities5, the two
bot bakes). Whole-card render 1-16 s CPU a card (the Mac under load). Tests:
`test_a_new_psd_beside_its_own_png_is_current`,
`test_a_new_psd_beside_an_older_png_takes_the_render_where_photoshops_image_moved`,
`test_a_new_psd_beside_the_authors_png_is_current` (Dirty Bed and Defcon Panda, real blobs), and
`test_a_new_psd_beside_its_pushed_png_tells_nothing` (card_guards end to end). Left open: a tool save beside a PNG whose
stored composite was not updated reads as current (no old PSD to compare its layers with; the composite build did the
same).

**Before `CARD_GUARDS=report`:** merge both branches; the first dry dispatches (section 8); check one of the 57 "within the bar,
past 2 levels" cards in Photoshop (section 15).


## 18. The owner's ruling on pixel misses, built (2026-10-23)

The owner (2026-10-23 ~03:15): "let's not refuse bakes for failing the pixel check anymore. rather, we should improve
psd-tools and fix these as we come across them." For guard 1 the same rule as the bake: write the fork's PNG unless it is
grossly wrong, and tell the miss in the report and the comment. Built on AutoCroissant `work/soft-parity-guards` (from
`work/push-guards`); the bake's side is `work/soft-parity` (EDIT_PIPELINE_DESIGN.md §5 step 3). Nothing pushed.

- **Softened: the checks with the bot's drawing on one side.** G6 (a) the baseline, G6 (b) the cross-check, the new
  card's whole render (G6, 5.5) and the beside case's G6 (b) refuse only past `psd_edit.PARITY_FLOOR`: more than 25% of
  their pixels past 24. A miss of the bar under it is written; `Export.misses` keeps the measure and
  `png_export.miss_notes` words it: "made with a pixel difference (the old PSD drawn against the old PNG): bad 1.35%,
  mean 1.00, max 79", "(against the PSD's own image)" for G6 (b). The line goes on the card's guard-1 line in the comment
  and on the PNG's line in the bot commit, never on Discord.
- **A card drawn whole is held per box** (review 2026-10-23). Over the whole card a blank text box stays far under the
  floor: on real cards with the bake's hidden-layer simulation, Shield of the Guilty's Ability drawn blank was 2.32% of
  the card (mean 4.88), its Text group 3.01%; The Freezer's 7.67% and 10.11%; each was made with only a whole-card note
  (the old bar refused them all). So `export_new` and the beside case also measure each text and stat box (5.5): past the
  floor refuses (The Freezer's Text blank: its Ability box 34.9% bad), and a text box drawn blank refuses as the bake's
  blank render does (Shield's Ability: 10.57% of its box, 100% of its letters' pixels, refused; beside a PNG showing
  another ability, 11.65% over D, refused). Blank means both: the floor counted over the layer's letters (Photoshop's own
  raster of it), and no rendered pixel moves when the layer is hidden. Hiding alone misfired in the first sweep: Sabre's
  Ability layer holds no text and moves nothing, so empty layers are no box; a layer covered by a copy moves nothing
  either, but its letters show, so the letters' count keeps it a miss. A box missing the bar under the floor is told per
  box ("made with a pixel difference (the Name text against the PSD's own image): bad 1.92%, mean 1.55, max 217", The
  Freezer as a new card, whose whole card is within the bar); the whole-card note only when no box missed. The extra
  render (the box with the layer hidden) runs only for a box whose letters are past the floor. Like the bake's, the
  blank-box refusal goes past the owner's literal floor and is the owner's to keep or drop; a box drawn shifted or in a
  wrong font under the floor is still made and told. Measured on 114 real cards pushed as new cards (the bot-gate scan's
  80 parity misses and every fifth card it refused for structure; fork pin `cards-20261022b`,
  `scratch/soft-parity/newcard/sweep_{old,new}.jsonl`): all 114 made before and after; 55 now carry a box note (5 had a
  whole-card note); 98 of 712 boxes miss the bar, the worst 6.78% bad (PoD06Rainbow_Bloon's Health label), the worst
  letters 22.05% (the same box, drawn, so no blank); no hidden render ran; 23.8 s per card median both ways (the Mac
  under load).
- **Still refused, under the floor too.** G5 (outside R, and inside R when Photoshop's old image disagrees with the old
  PNG too): the old PNG was already stale, Photoshop's own image says so, and no psd-tools fix makes the PNG right. G6
  "changed where the bot sees no layer change": the PNG would not show the author's change (a type layer whose stored
  raster stayed), the guard's read-back floor. G2, G7 and G8 as before; the burst anchor still retries a G5 or G6
  refusal. Softening G5 as well, if the owner wants it, is one `_drawn` call in `export`.
- **Measured on the 34 replayed commits** (fork pin `cards-20261022b` at 90508839, `scratch/soft-parity/replay34_soft.jsonl`,
  the Mac under load): 29 PNGs made (25 before), 5 refused (9 before). The 4 that moved are the drawing misses of
  section 17, each now within the parity bar of the author's own export: Defcon Panda (baseline 1.35% bad, cross 12.79%;
  max 79 against the author's PNG), Clock (cross 3.6%; max 107), The Radiance (cross mean 3.84; max 48), Pandamonium
  (cross mean 3.15; max 9). Still refused: Restrained (G5, 7.1%), 20 Creature Types and 11 Abilities5 (G6, unseen), and
  the bot bakes PoD09DDT and PoD10BFB (G6, unseen; bot commits are never guarded). PNG pushed first: 34 of 34 current.
- **Tests** (fail without the change): `test_a_drawing_miss_under_the_floor_is_written_and_told` (one row 30 levels off:
  4.76% bad, made and told; ten rows: 47.6%, refused), `test_a_new_card_drawn_with_a_miss_is_made` (new and beside),
  `test_a_new_card_drawn_blank_in_a_text_box_is_refused` (Shield new and beside, The Freezer's Text group),
  `test_a_new_card_tells_its_missed_box` (The Freezer's title), `test_a_new_card_with_an_empty_text_layer_is_made`
  (Sabre),
  `test_g5_stays_a_refusal_under_the_floor`, `test_the_refusals_of_the_replay` (Defcon Panda now made, its baseline miss
  told), `test_a_drawing_miss_is_told_in_the_comment_and_the_commit` (card_guards: the comment, the bot commit's line,
  nothing on Discord).
- **A blank drawing** (review 2026-10-23; the bake refuses a missed box whose render shows no change, `work/soft-parity`):
  in `export` F empty while Photoshop's image moved is G6 "changed where the bot sees no layer change", refused at any
  floor; a card drawn whole (new, or beside its PNG) has no old render, so its boxes are checked instead (above).
  Not caught, unmeasured: one push that changes a layer the fork draws and a text layer it draws blank in the same box;
  F is the first layer's, and the cross-check over F and D can stay under the floor, so the PNG is made with the old text
  and the miss told. A tool save with a stale composite and a blank drawing has no F and no D: "PNG still current"
  (as before the ruling).
- **Open, not this ruling's:** `test_a_faint_beside_mismatch_note_names_the_case` fails on `work/push-guards` at 0b101e9
  already (BESIDE_CUT 8 makes its 10-level PNG re-exported, not current).

## 19. Lossless PNG recompression (owner, 2026-10-23)

The owner (2026-10-23 ~10:00): recompress the card PNGs losslessly on push (in the guards' bot commit) and once over the
existing ones (Photoshop's exports measured 4.8x too big, below). The thresholds (10%, 64 KiB) are ours; the owner set
none. Built on AutoCroissant `work/png-optimize` and TTSCardMaker `work/png-optimize` (card-guards.yml installs
`pyoxipng==9.1.1`; build_site skips recompressions in Image versions). The two land together: on a TTSCardMaker main
without it, `test_workflow_shape` and `test_a_recompression_is_no_new_image_version` fail. Nothing pushed.

- **oxipng** through `pyoxipng` (wheels for Python 3.9-3.13 on macOS and manylinux, so the Action's 3.12 takes a wheel;
  3.14 builds it with cargo). `png_export.optimize`: level 2, colour type, bit depth and palette never reduced (the bake
  and guard 1 read RGB and RGBA only; an opaque RGBA stays RGBA), no chunk stripped, 120 s at most per PNG. Pillow's
  `optimize=True` saved less (78.1% against 79.3%) and dropped pHYs; level 4 adds 0.3% for 1.8x the time.
- **Proof, per file, before any write** (`png_export.same_png`): IHDR byte for byte, the pixels PIL decodes in the file's
  own mode, and every other chunk in order: iCCP by its profile (oxipng re-deflates it and names it "icc", a label no
  reader uses), zTXt and compressed iTXt by their text, the rest byte for byte (pHYs, gAMA, cHRM, iTXt XMP, eXIf).
  8-bit PNGs only (all card PNGs are).
- **Measured** on the 30 latest card PNGs (Photoshop exports, zlib header 78 01: the deflate stored, IDAT as big as the
  raw pixels): 257.2 MB -> 53.1 MB (79.3%, 4.8x), 30 of 30 the same image; 440 of the 830 card-folder PNGs at origin/main
  96d61368 carry that header.
- **The guards** (`card_guards`): every card PNG a push brought by a person (a pair's PNG, a PNG pushed alone, the
  numbered PNGs, a numbered PNG pushed alone: case `numbered`, or in the `png-deleted` entry when the push deleted the
  main PNG: one entry per PSD) is a write of kind `recompress` when it shrinks by more than `RECOMPRESS_MIN` (10%) and
  `RECOMPRESS_BYTES` (64 KiB); it rests on that PNG alone (and only it rests on a pushed numbered PNG: the main PNG's
  re-export never reads those), lands in the one bot commit ("Card guards: 2 PNGs recompressed (push ...)", one
  `Recompressed: <path>` trailer each), is in the comment ("Fat.png recompressed losslessly (8.7 MB -> 1.8 MB, the same
  pixels)") and never on Discord, nor when it did not land. A failure (a blob unreadable, oxipng raising anything) is a
  comment note ("Fat.png not recompressed (...)"), never the card's error. The PNGs the bot writes (`png_export.encode`, a
  renamed PNG's copy) are recompressed too. Bot commits stay unguarded, so nothing loops. Replay of the push
  38c55bd3..07589cc7: 16 recompress writes (12 rulebook pages 6.2 MB -> 107-801 KB, 4 cards 8.6-11.6 MB -> 1.8-3.4 MB),
  Discord unchanged. oxipng takes about 4 s a card PNG on the Mac (the Action unmeasured), inside Check.
- **The batch** (`python -m commands.png_recompress --clone <checkout> [--apply] [--report <json>] [--message <file>]`,
  nothing committed; `--message` writes the commit's text, one `Recompressed` trailer per PNG):
  run on a sparse copy of origin/main 96d61368 (scratch `followups/png/tc-batch`): 820 card PNGs, 4.47 GB; 686
  recompressed, 4.18 GB -> 1.14 GB (3.04 GB saved, 68% of all card PNGs); 134 kept (under 10% or 64 KiB, 23 MB
  possible); 0 not the same. An independent check (PIL RGBA and raw arrays, mode, size and info of every rewritten file
  against HEAD's blob) found 686 of 686 identical. 25 minutes on the Mac under load.
- **Landing the batch**: `--apply --message <file>`, then `git -c user.name=AutoCroissant -c
  user.email=autocroissant@gmail.com commit -a -F <file>` (the guards skip it). The trailers keep the website's Image
  versions as they are (TTSCardMaker build_site, site/NOTES.md §18): without them each of the 686 cards would get an
  AutoCroissant version identical to the one before, and a card with one version would start showing the block. The
  TTS save re-pins the 686 links and the site rebuilds. card_history's type-misread merge compares PNG blobs across a
  version's start, so a misread pair spanning the batch commit is no longer merged. The bake (`psd_edit.save_png`,
  card-edits.yml) still writes PIL's deflate.
- **Tests** (fail without the change): `test_recompress_keeps_the_image_and_its_chunks`, `test_same_png_tells_a_changed_pixel_or_chunk`,
  `test_recompress_wants_its_saving`, `test_recompress_leaves_16_bit_and_runs_without_oxipng`,
  `test_the_bots_own_pngs_are_recompressed`, `test_pushed_pngs_are_recompressed_in_the_bot_commit`,
  `test_a_recompression_rests_on_its_png_and_is_never_told`, `test_message` (the trailers), `test_workflow_shape`
  (pyoxipng), `test_the_batch_rewrites_the_heavy_card_pngs`, `test_the_batch_message_names_each_png_in_a_trailer`,
  `test_a_pushed_numbered_png_holds_back_only_its_recompression`, `test_a_deleted_png_and_its_pushed_numbered_png_are_one_card`,
  `test_a_failed_recompression_is_a_comment_note`, test_site_build's `test_a_recompression_is_no_new_image_version`;
  the replay and Mini Doomer's R11 now expect their recompress writes.
