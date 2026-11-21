# Next session plan (after 2026-11-21)

## Where to start

Start in `~/Desktop/AutoCroissant`. Read memories `resource-limits`, `secrets-never-printed`,
`maintenance-pushes-silent`, `photoshop-no-interrupt`, `never-pkill-shell-snapshot`, `owner-decisions-2026-11-19` (and
`-11-15`, `-11-12`, `-11-10`), `research-rounds-future`, `research-loop-records`, `card-app-project`,
`card-timelines-project`, `jetson-server`, `jetson-reboot-hang`, `headless-browser-keychain`, `storage-space`,
`persistent-work-paths`, `agent-model-preference`, `no-sendmessage-to-workflow-agents`. What landed:
`docs/Finished/session-2026-11-10-21/README.md` (research rounds, card fixes, the card timelines, the Morgue,
CardBrowser; the 422 verified ideas and every accepted and rejected title); earlier `session-2026-11-07-10/`. This
session's binding lines are in `~/Documents/Tools/wt/scratch/s1110/BRIEF.md`; the ones that last are under Standing
rules below.

The Desktop AutoCroissant clone was left untouched by rule at the wrap-up. Its checkout is e35327a4 (11-13), 594
commits behind main ea0db694, with one untracked file (`caches_now.tsv`). Read this plan with `git -C
~/Desktop/AutoCroissant show origin/main:docs/NEXT_SESSION_PLAN.md` (after a `git fetch`), or fast-forward the checkout
first if the owner's tree is still clean (`git status` shows only `caches_now.tsv`). Wave 2 left nothing half-done:
every kept item landed, every run was green, and the jetson runs AC 0d2de792 (the same bot modules as main).

## Order (owner, 11-20 and 11-21)

1. **Research rounds**: about 8 rounds per lane, one ship every 4 rounds, then stop (owner 11-21; don't ask for a budget
   again). Lanes as this session: Actions + bake + previews (`actbake`), card data + TTS + website + bot + tests
   (`cards`). The fork lane went dry on 11-16 (rounds 21-24 found nothing); start it only for the owner's open lens
   below (linked pictures that can't be redrawn). Seed the rounds with the research items in "Waiting from this
   session". Don't re-propose a title the report lists as accepted or rejected.
2. **Then finish everything the rounds found**, as 11-20/21 did: build every waiting idea, and give each rejected idea
   whose reviewer confirmed a real gain one more try (two judges first: still needed, no ruling against it, the
   regression fixable). All of it lands before the wrap-up.
3. **The other open tasks** below, beside the rounds where their ground doesn't clash (at most 2 workflows).
4. **Card timelines**: live since 11-17. More work on them only when the owner asks.

## Waiting from this session

### Research items

Wave 2 landed every kept item on 11-21: wave 2a cards (11 items: r58-3, f11, x1, x6, x9, x16, x17, x21, x25, x28,
x32) at 08:07, AC ab9c7349 (deployed), TC 68961a8b + save 6950590a; wave 2a actbake (12: f5, f9, x10, x13, x14, x18,
x19, x20, x22, x24, x29 and the early save r60-1) at 09:23, AC f0f4c7c1, TC 0911ce04; wave 2b cards (x35, x36, x39,
x40, x41) at 10:54, AC 0d2de792 (deployed), TC f769e0fc + save 33c6ce3c; wave 2b actbake (x12, x23, x33, x34, x38) at
12:27, AC ea0db694, TC 0cb95c33. Only r60-2 was left out:

- [ ] **AWS warm-up** (actbake r60-2, finish item n=4; proved, not shipped): card jobs on GitHub's AWS runners read node
      and Python into memory in the background first. The builder's proof on the fork's origin (11-21): first step to
      the end of the venv check, network wait included, 8.52 s cold against 3.60 s warmed on AWS (+4.92 s, medians of 10
      and 8 probes, p=0.017; the venv check step alone 1.67 -> 0.06 s); no change on Azure. Wave 2a's verifier ran its
      own proof (180 probes, 19 on AWS): first step to venv checked 6.61 -> 3.66 s, +2.95 s at the median (p=0.0006, 95%
      CI 2.2-5.2 s); both proofs pooled (37 AWS probes) +3.7 to +4.1 s; Azure 3.04 -> 3.22 s (noise). About 1 job in 5
      runs on AWS (25 of 142 real jobs 11-19 to 11-21; Card data 3 of 19). Not shipped: the gain is real but smaller
      than built and clears the 11-16 research bar only on its narrowest reading (owner 11-21 ~01:50: when unsure, ship
      nothing doubtful). Nothing regressed. Decide it against the bar with these numbers. TC abb071f5
      (`work/actbake-1110-r60-2` in the TC bundle; its AC side has no commits).
- [ ] **Early save, watch item (d)** (r60-1, landed in wave 2a: TC d01f81cd, notes (a)-(c) fixed in 79d79682, checks
      0911ce04): the guards' Land onto a main that a Card sync save has already moved has never run on a real push.
      The early save is proven only by the next guarded push that re-exports a PNG: watch Land retrying, the
      follow-up's re-pin, and the early 'Card sync: TTS save updated' commit with its one extra #github-updates post.

Not kept after their one retry (11-20/21). Each gain was reproduced; each is blocked by a named regression. Retry only
with the fix. Ids as in `scratch/s1110/final-1120/final_a_results.json`. Every branch named in this plan (except the
fork's `work/research-1110`) was deleted at the wrap-up cleanup (11-21) after being recorded in
`~/Documents/Tools/wt/scratch/s1110/wrap/branch_record.tsv` and bundled in `~/Documents/Tools/wt/scratch/s1110/bundles/`
(`ac-unlanded-1121.bundle`, `tc-unlanded-1121.bundle`, `fork-unlanded-1121.bundle`, each verified). The retries are bundle refs `refs/wrap/<sha>`, the round branches
`refs/heads/work/<lane>-r<n>-<m>`; get one back with `git fetch <bundle> <ref>:refs/heads/<name>` (in TC with
`GIT_NO_LAZY_FETCH=1`). The SHAs are unchanged.

- [ ] **D15 rest, owner-directed (11-17 ~00:30)** (x31, from r51-1): a typed character the line's font can't draw keeps
      the font the card prints it in (else Myriad Pro), so an edit never erases it. Not kept: the step back to the body
      font changes only the Font, so typed letters keep a stray run's size and baseline shift (Minecraft Panda's '2'
      drops like a subscript; Spice's new line prints huge). Fix: give such a letter the body run's whole style, or skip
      the step when size, shift or scale differ; re-run the census and the Panda round trip. AC fa13c489, TC d20c1105
      (was `work/actbake-1110-x31`).
- [ ] x4: Mock-up: an edit the card's font draws the same keeps the card's own pixels. Not kept: Dr. Robotnik's
      Roboticizer's apostrophe is in another font, so the bake restyles it and the build hides that (about 50 layers mix
      fonts, sizes or shifts on one line). Fix: per-letter style in psd_details, or no shortcut when a layer has more
      than one font. AC c6e4ec99, TC d05bf5c0.
- [ ] x26: Rulebook and Stats icons load as they come near (-22% bytes on the Rulebook's first open). Not kept: icons
      show blank in rulebook search results (0.3-0.7 s) and same-book jumps (up to 1.44 s on a 3G phone). Reviewer's
      suggestion: ship 494d9cba's wake() alone, without the lazy icons. AC e7e6e3ee, TC 494d9cba.
- [ ] x27: Bake: a re-edited Ability is fitted against the card maker's own text. Not kept: the stored ink (about
      10 KB a verdict) breaks Card sync's merge of bake records on Linux (one jq argument over 128 KB) once about 13
      Ability bakes wait. Fix: pass the verdicts as files (`--slurpfile`), test a 200 KB artifact, ship it with the AC
      change. AC 129df760, TC 824bb993.
- [ ] x30: Card dialogs make the page under them inert on the first Tab (open and close about 40% faster). Not kept:
      opening the profile editor from a dialog puts focus on the header instead of the form. Fix: run a pending seal
      before any view render. AC ac82a0e7, TC 2b11af0f.
- [ ] x37: Rendered preview: the tab where a preview is pressed does the checking. Not kept: a move to another tab in
      the first second still waits about 18 s. Fix (tried in scratch): count "starting" presses in claim(), then a fresh
      browser check. AC 6109c566, TC 6e439e89.

Dropped by the judges and never built: x3, x7, x8 and x11 (all done since) and x15 (against the exact mock-up, owner
11-21). Two were split, so the next judges look again:

- [ ] x5 (actbake r10-0): a guarded push's own Bake and save waits for the guards' Landed and saves; the follow-up's
      main lane then skips. Dropped 2-1. Two judges: gone on today's mains (the kept push lane saves sooner; on the 8
      follow-ups since 11-12 the follow-up was ready by Landed). The third: still missing on pushes whose fixes change
      a picture (b156a96d, 11-18: save at +79 s, the push lane ready since +13 s). Check it on the next such push,
      beside the early save (r60-1). AC 5598470f, TC 8496a843.
- [ ] x2 (actbake r7-0): the mock-up matches kept Ability lines by their letters, not their colour runs. Dropped 2-1.
      Two judges: done since (drawText tries the plate's own line layout first; keptLines runs only when that doesn't
      draw). The third: retry parts (a) and (b) without the dy move (76 better, 0 worse when the reviewer measured
      them). AC a35ce841, TC ae380cfa.

Rejected in the rounds and not retried (`rejected.md` section 1 of the report has all 21). These 15 were turned down
for a named regression: 13 had their gain reproduced; r7-2's pixel scores reproduce, but its regression is in the very
edits it targets. For the other 6 the gain was not reproduced or did not hold up, or the change broke a site contract
(r15-3). Two judges first: some may be done since.

- [ ] actbake r1-1: a redrawn Ability sits on the old text's own baseline, measured per font. Regression: a shrunk new
      text sits on the old, larger baseline (pushed down, once off the card). AC 137bdd7e, TC bd3d3fcc.
- [ ] actbake r2-1: an Ability shrinks only when it needs more lines than the old text. Regression: Life Link's text
      runs off the card again; Sans, Poorly Drawn Cabbage and Mangle worse. TC adf4f5e5.
- [ ] actbake r7-2: in the one line an Ability edit changes, the mock-up keeps the card's own letters around the change
      and draws only the changed ones (one-number edits: wrong px median 3656 -> 318). Regression: about 1 one-number
      edit in 5 shows bits of the old digit or the wrong text, which the pixel score counts as better. AC 168c2502,
      TC 25695371.
- [ ] actbake r23-0: Card sync's Commit main stops waiting for the Rulebook PDF (TTS save 12-21% sooner, modelled
      from logs). Regression: its new PDF step starts before the book fonts are in the checkout. AC 4db5c3ce, TC
      040ea6f2.
- [ ] actbake r39-0: a new type icon copies only the clipped layers that cover the whole reference icon (32 cards).
      Regression: the new icon looks worse on 6 of them (White Space most). AC 03a1248e, TC c662cfeb.
- [ ] actbake r40-1: going back to the maker's star count after an option-B shrink brings back the maker's stars
      (finishing D14; 66 round trips on 23 cards). Regression: a new helper `_offset` replaced psd_edit's existing one
      and broke the comp bakes: rename it. AC dd18a196, TC 9566bd52.
- [ ] actbake r42-0: the first preview press is warm too (the site starts a lane when an editor changes a printed
      field). Regression: a late warm-up hand-over can point the next press at a lane that has ended. AC aec4ed01, TC
      6199cb8d.
- [ ] actbake r52-1: a title edit that would print over the stars or type icons shrinks until it clears them (13 of 13
      fixed). Regression: a second edit of a shrunk title prints at full size over the icons, with no note. AC
      6480d339, TC fb14d299.
- [ ] actbake r54-1: an Ability an earlier bake lifted goes back to the maker's place on its next edit (22 of 22
      lifted status Abilities fixed). The AC bake part holds; the TC mock-up part made a follow-up preview worse: ship
      the bake alone or fix the mock-up. AC 61be6f6a, TC 33eb406a.
- [ ] cards r10-0: card pages' related strips load their pictures once in view (desktop -15% bytes). Regression: some
      strips stay blank on desktop (29 tiles on 7 of 41 pages). AC a1af2a3a, TC e4961e73.
- [ ] cards r10-1: the card dialog reads a small per-card history file (desktop -12% bytes). Regression: positions
      shift between builds, so a deploy while a page is open shows the wrong history. versions.json is gone since 11-19:
      check what is left. AC a074e520, TC fc828639.
- [ ] cards r21-0: grid tiles pick their picture by drawn width (`sizes="auto"`; first screen -13 to -21%).
      Regression: the 64 status cards' square pictures get visibly softer. TC 9a9ba84f.
- [ ] cards r22-0: smaller rulebook copies of card pictures, the cover and the table map (whole book -17% bytes).
      Regression: the ban list's `<img>` lost its shape under `sizes="auto"`: give it width and height. AC 98dffe2b,
      TC 2cfa5dd6.
- [ ] cards r25-0: grid tiles load one screen ahead, not Chrome's ~1250 px (first screen -17 to -23%). Regression:
      more blank tiles while scrolling on slow networks. AC 2f5642a1, TC ea7d3d01.
- [ ] cards r44-2: a Stats chart's table is drawn on its first Table press, not built hidden on every visit (10,746 of
      16,861 elements; load -9%, filter changes -10% on a phone). Regression: it shows the wrong table (Series while
      Subseries is pressed). AC 3017ef00, TC f12a757e (branches deleted; the Desktop clones still hold both commits).

The research4 backlog (28 ideas, `scratch/s1110/research4/args.json`) is all built and judged; r54-1 above is the only
one rejected and not retried. Of the 11 finish items, n=3 and n=10 were already covered on main and n=6 by r60-3.

### Card fixes

- [x] Energy Ritual's final '.' in the Ability's font (TC 8b0d4a6a), Exposed's and Grapes' Ability leading (31e9b979,
      258ac03f), all silent; the printed typos Suffocation "Can't affect" (6de68813) and Alex YIIK "At the end"
      (bd06b214), owner-approved, no trailer (11-19). Also Grapes' doubled title (parser rule, AC 781c9931) and Gasoline
      in Cuphead Memphis (variant E as one rewrite, TC f52f4a53).

### Other open tasks

- [ ] **Fork dead code** (owner 11-12, group G8 in memory `owner-decisions-2026-11-12`): the delete list in
      `scratch/s1110/deadcode/fork-delete.json` and the answered items in `scratch/s1110/asks/fork.json`; keep the
      owner's KEEP picks (unused public API, contours and bevels, Angle/pattern stroke/precise glow, unused adjustments).
      One fork release, then re-pin AC's requirements-cards.txt (no jetson deploy: the jetson has no fork). Not
      released at the wrap-up.
- [ ] **Linked pictures that can't be redrawn** (owner lens 11-13, fork): data a2e263cb still lists 46 linked pictures
      on 40 cards (43 with smart filters, 3 with a warp; `db/card_checks.json` field `unredrawable`), so Card sync
      can't redraw those cards when their linked file changes. The fork learned Puppet Warp and Invert (11-14). Bar:
      looks the same by eye against the card's PNG.
- [ ] **Fork r15-2**: colour glyphs in Smooth and Sharp drawn as Strong (fork `work/research-1110` be32812f, kept
      locally and in the fork bundle; held since round 15 because The Great Papyrus' ability-edit mock-up fell back and
      looked worse). The AC fix it waited for, ef5359da "Plates: layout gate leaves icon glyphs out" (round 18-0), is
      on main as 2f7e22fe (actbake research3 ship 1, 11-16), and x10 (wave 2a, AC fcc96acc) now keeps a layout past 5%
      off when its lines match. Nobody has checked The Great Papyrus with r15-2 on today's AC main: make its plates
      that way first; if its layout is kept, r15-2 goes in the next fork release.
- [ ] Details memo (AC `work/r3-1` 22e09b0c, `work/r4-3` 485dd38b: star pieces and core boxes read once per gather;
      the fork lane measured -54% of details CPU on its sample, round 23): rebase onto AC main and measure against the
      research bar before building on it.
- [ ] **Retire game reports** (owner 11-13: after the timelines): delete `docs/Campaigns/GAME_REPORTS_CAMPAIGN.md` and
      the `/report_game` lines in `docs/Campaigns/SIMULATOR_CAMPAIGN.md` and `docs/Future/SIMULATOR.md`.
- [ ] **D24**: put the stoogedings build number in the font's name at the next font build (no build since 11-15).
- [ ] **TTS Saves folder** (owner 11-13): delete the old test save 'AC test 11-11 tooltips 2'. The folder's copy of the
      save is from 11-18: copy main's in again after the last save change.
- [ ] Photoshop 2023's JSX PNG saves have come out untagged and up to 55 levels off the card PNGs since 11-16 ~04:12,
      on untouched PSDs too. Cause not found. Open tests are fine; don't use new Photoshop exports as pixel truth until
      it is found.
- [x] **Desktop TC promisor duplicates**: the wrap-up deduped pack-185d66d5, 4080ec91, 239227ea and c94ca1fd (clean
      dry runs, `fsck --connectivity-only` passed) and removed a 0.91 GiB tmp pack: TC packs 32.3 -> 28.2 GiB.
- [ ] **Two TC packs of unknown origin**: pack-333ff011 (8.47 GiB; its dry run keeps 2,266 of 3,351 objects, 6.2 GiB:
      the commits and trees plus what is found nowhere else) and pack-8cfce261 (0.98 GiB; it keeps all 191 objects).
      Kept: their dry runs were not clean. Find what they hold, at a quiet time, before dropping anything.
- [ ] Wrap-up leftovers (11-21 cleanup, `~/Documents/Tools/wt/scratch/s1110/wrap/run-1121/`):
      - The Desktop AC clone was not fast-forwarded (hard rule): see "Where to start".
      - The fork's Desktop clone `~/Documents/Tools/psd-tools` was not fast-forwarded either: da069245
        (`cards-20261111`), 105 commits behind fork main 6d384060, clean.
      - `scratch/s1109/wrap/ac-actions-1108.bundle` and `tc-actions-1108.bundle` (42 KiB) kept: their heads are not
        ancestors of main (the work landed rebased).
      - About 11 GiB stays in `scratch/s1110`, mostly small .json/.md files the slimming rules keep; a second pass
        could thin it. `s1110/v-x13` and `s1110/x23-merge` (about 8 MB) were skipped as too recent.
      - Memory `card-timelines-project` still points to the deleted review pages focus, final and meme (their
        generators `scratch/s1110/timelines/focus/make_*.py` remain).
      - Done: every work/scratch branch and worktree recorded, bundled and removed (117 worktrees -> 1, 177 local
        branches deleted, none on GitHub), the review pages final, focus and meme 1-4 deleted (the server was not
        running), the 288 Photoshop relink references deleted, 3 leaked x13 workers killed.

### CardBrowser (owner 11-20/21)

- [ ] **The owner's first trial** of v0.1.0 (`MichaelJSr/CardBrowser`, public, a quiet GitHub pre-release: macOS arm64
      and x64, Windows x64). A read-only window around the live site, with Open in Photoshop, Show in Finder/Explorer
      and Copy PNG. CI's self-test passes 19/19 on all three. Never tried: macOS 11-14, a real Windows desktop, a real
      Open in Photoshop. No website link or announcement until the owner has tried it; the owner tells the makers.
      The release is at 5d9f2bf; main 3b14621 adds only a README line (offline on Windows). The site hook it uses
      landed 11-21: TC 016872be, AC 4149cf25 (tests).
- [ ] Linux: only if a maker asks (its code stays, untested and unadvertised).

### From the 11-10 plan

- [x] AC tests' old card blobs restored (11-10, `scratch/s1110/blobs.ok`); the suite has run green since.
- [x] First card workflow run on the fork pin: the re-pins since (cards-20261111 to cards-20261114) rebuilt the fork's
      wheel on the runner with f80772da; runs green.
- [x] `work/actions-1108` landed with its two fixes (11-10: AC fc8357f5, TC 83495964; version-commit fallback, data
      fetched again in 3 tries).
- [x] Card browser deploys stuck in "syncing_files": the first Pages artifact is deflated again (TC 23d54f9f), and
      deploy-pages runs after a failed retry (83495964).
- [x] Guards gap R10 (an old PNG put back, or a PNG that skips its PSD's change): AC f1b03b33, 67daeef0 (11-10/11).
- [ ] The 11-08 Single-line batches and card fixes without a `Card-sweep` trailer (TC 66efec76, 2ae8384c, 12bc1331,
      ac57d748, 380b856e): whether Card sync posted them as New art was never checked. Nothing to undo.
- [x] Card fixes left on 11-10: Water Totem's stray title character (TC 9fa96d17, 11-13); Man-Serpent's Shield's title
      baked to the website name (c05e5d4a, 11-11).
- [x] The 17-card Card checks re-read (run 42833874576): green.

### Waiting on events

- [ ] The first live `+ability` bake: check its placement and font (12 of the 18 minions that lost their empty Ability
      layers can take one; 6 show "no room").
- [ ] Croissant's guarded pushes: the delta-free Land, the kept push lane, PNGs recompressed after Decided, the
      Follow-up build window past Landed, the preview reads at the commit. His push 50ecfca4 (11-20 15:46 PST, the
      guards re-exported 1 PNG) came after all of them shipped: check its runs for them, else his next push.
      Then also the early save (r60-1, wave 2a): proven only by the next guarded push that re-exports a PNG (watch
      item (d) above), and x5's open question beside it.
- [ ] The next Card checks run: x12's threaded WebP encode, plates byte for byte as before.
- [ ] The next bake whose Ability edit moves an effect-layer line: x33 keeps the line's look.
- [ ] The next Preview image press on the website: x20's 250 ms hand-over checks. x34's caption ("The text runs past
      its place on the card.") was not found by string in the live chunks; it and x23 rest on the prepare's checks.
- [x] A psd_details VERSION bump passes gate G4: the 11 -> 13 re-read (wave 2a, Card sync 44928841136) passed it.
- [ ] The next named-cards Card sync dispatch: its site build and Card checks start with its data job.
- [ ] The next runner Python patch: the cached venv is re-pointed, not rebuilt.
- [ ] The next website edit that bakes: one Card sync, not a second one taking over the after-bake job.
- [ ] Next bakes of those cards: Mosling, Moose Goose and Wickerbottom title copies on their arc; a text edit whose
      changed lines miss the parity bar (Tainted Jacob).
- [ ] The push-lane Rulebook PDF on its own Pages artifact (`--local`); a hand-over zip's 404 looked for again; a
      `--baked` save that takes a new name.
- [x] The first card maker push after the font's A1 note: superseded. The guards re-lay text an older font build or a
      substituted font explains, silently (owner 11-10, AC 4007d9d7).
- [x] The proposal forum's first real day-6 ping (11-16): the posts were not closed (no Manage Threads); fixed and
      deployed (AC b00003c5); the owner gave the role Manage Threads server-wide and both posts closed.

## Standing rules

- Work unattended and ask no questions unless the owner says "ask me questions" ("work unattended and dont ask
  questions": 11-18, 11-19, 11-21). When unsure: two independent judges; both certain and agreeing = do it; else take the
  most conservative choice that publishes nothing and list it in the session report. When the owner does ask for
  questions: only the uncertain ones, pictures embedded in the page (11-15). Agents on Opus or lower (the 11-10 Fable
  trial and design panel were one-offs); a single Agent for a build. Owner picks are never dropped: a review may only
  flag them.
- Commit to main in both repos; `work/*` only while in flight, then delete. Very short commit messages; no Claude
  attribution (plumbing bypasses the hook: `git log --format=%B | grep -i co-authored` before any push). Never
  `git stash` (one stash for every worktree, 11-13). Persistent paths only (`~/Documents/Tools/wt/`, `~/std-work-*`).
- Deploy with the `jetson-deploy` skill when a file the bot loads changed (a static walk from main.py); deploys always
  restart. The deploy lock is shared with OJWikibot: let a deploy wait, never kill or force it. The jetson has a
  bot-only venv since 11-11 (no psd-tools fork, no Cython): a fork release only re-pins `requirements-cards.txt`.
  Restart the jetson only with `ssh jetson sudo sync`, then `~/.local/bin/jetson-powercycle cycle` (never `sudo
  reboot`: it hung the board on 11-10 and took the LAN's DNS with it). OJWikibot is the owner's OJWikibot session's
  ground (11-11); `deploy.sh` stays ours.
- Tests: `AUTOCROISSANT_LOCAL_DIR=~/Desktop/TTSCardMaker AC_SITE_SRC=<tc tree>
  ESBUILD=~/Documents/Tools/wt/scratch/esbuild/node_modules/.bin/esbuild ~/.venvs/ac-py314-fork/bin/python -m pytest
  -n 6 -q -p no:cacheprovider tests` (`-n 3` while the owner games).
  1289 passed at the final heads (AC ea0db694 + TC 0cb95c33, 11-21); the TC tree must be at or after 0cb95c33 (with an
  older one, 4 test_site_previews tests skip). Two timing tests flake under `taskpolicy -b` at load 40-60 and pass
  under plain nice (test_card_preview::test_the_lanes_next_cards,
  test_range_reads::test_other_psds_links_read_in_the_workers_as_one_at_a_time): re-run them before calling a red
  real. Point `AC_SITE_SRC` at a sparse TC checkout at or after TC main (`.github`,
  `site`, `TTS/Global.lua`, `Global.xml`, `decks.json`); the workflow tests read `card-guards.yml` from it too.
  Targeted files unless a full run is needed; at most a few new tests per change.
- Resource limits (memory `resource-limits`): xdist at most 6, process pools at most 4, one full suite at a time
  (`~/Documents/Tools/wt/scratch/suite-wait.sh`), at most 2 workflows (a third only when the owner allows), at most 8
  git processes, never a partial clone of the local card repo, no repack or gc of the big clones while agents run.
  The git watchdog (`~/Documents/Tools/wt/scratch/git-watchdog.sh`) runs under Monitor during long unattended work.
  While `pgrep -x RotMGExalt.real` finds the owner's game: heavy work under `taskpolicy -b nice -n 15`, xdist at most 3,
  pools at most 2. Disk: at least 100 GB available counting purgeable and 50 GB truly free
  (`~/Documents/Tools/wt/scratch/s1110/tools/diskfree.sh`); delete bulky outputs when a step ends.
- The Desktop TTSCardMaker clone is blobless from GitHub (full history): every git call with `GIT_NO_LAZY_FETCH=1`; a
  missing blob comes in ONE batched fetch, never one `git show` at a time; `git write-tree` needs `--missing-ok`.
  Refresh TC's local `origin/data` with `git fetch origin +data:refs/remotes/origin/data`. A fetch on top of local
  commits copies them into a promisor pack: dedup it at a quiet time (memory `partial-clone-fetch-packs`). TC
  worktrees are sparse (a full checkout is 16+ GB).
- No test branches on TTSCardMaker's GitHub (every push posts in #github-updates, owner 11-11): test Actions by dry
  dispatches on main or locally. A ship's prepare agent may prove a change on a scratch branch of the fork's origin
  and deletes it after.
- zsh arrays start at 1 and zsh doesn't split unquoted variables: multi-step shell as a `set -euo pipefail` bash
  script. `gh run list -c` needs the FULL sha.
- Secrets (memory `secrets-never-printed`): never print config.py values, tokens or webhook URLs; a token goes only to
  its own service; hooks: name, active flag and host only. Never print other processes' command lines or environments.
- PSD pushes: run the guards dry first, with `--repo`: `python -m commands.card_guards check --clone
  ~/Desktop/TTSCardMaker --repo MichaelJSr/TTSCardMaker --before origin/main --after <head> --mode dry --out <dir>`;
  discord.json must be `{}`. Every scripted TC commit that changes card PSDs or PNGs without a real card change
  carries `Card-sweep: <kind>`. A real wording fix goes in its own commit without the trailer (Drawcia 11-04;
  Suffocation and Alex YIIK 11-19). Right before a card push, compare the card's blob on origin/main with the one
  your rewrite started from: never overwrite a card maker's newer Photoshop work (11-11). A font rollout's sweep and
  batch pushes: `ttscardmaker-companion` §1.10 (card rollouts §1.10-1.11).
- PSDs our code rewrote: one rewrite per card per pass, and a Photoshop open test of every one before the push
  (`~/Documents/Tools/wt/scratch/s1110/ps/open_all.sh`, copies under `~/std-work-1107/`).
- Photoshop 2023 only on copies under `~/std-work-1107/`, under `ps.lock`, `open -g`, the alert watcher the whole
  time, dialogs off, every document closed without saving; quit the Photoshop you started. It may run while the owner
  games (11-11). The JSX and its log live under `~/std-work-1107/` (one under ~/Documents hangs on a privacy prompt).
  Never answer or click a permission prompt: stop what raised it and report. It draws with the fonts installed in
  `~/Library/Fonts` (MDW's stoogedings ring build, Futura-Bold, Lucida Grande Bold, Rockwell Bold, Elephant Italic,
  Myriad Arabic, Roboto Black): keep stoogedings.ttf identical to main's.
- Card fixes found by research (owner 11-19): when two independent judges agree a fix is invisible (looks the same by
  eye: fonts, leading, stray styles) and safe (no maker's recent work at risk), do it: Photoshop open test, blob
  re-check, one commit per card, `Card-sweep` trailer. Printed wording changes always wait for the owner. List every
  fix in the session report.
- The bar for the bot's drawings (guards re-exports, bakes, previews, fork renders) is "looks the same by eye" (owner
  11-13): fix what someone would notice at normal size; level-by-level drift is fine; no card sweeps only for parity.
- Research (owner 11-16 ~04:40): a new idea must be something a person would notice: a wrong card, picture, preview or
  bot answer fixed (name them), a real failure fixed (name the runs), or at least 5 s or 10% off something a person
  waits for. Finders return an empty list when nothing clears the bar (never a "NO BUILD"/"NO CHANGE" record; filter
  them in code; memory `research-loop-records`). A skeptic per build; a ship every 4 rounds under the locks.
- P-2 (owner 10-25: one main lane per guarded push) is amended twice: the kept push lane (11-19: a guarded push whose
  fixes keep every picture saves the TTS save from its own lane, one more save commit and #github-updates post) and the
  early save (11-21: a guarded push's own lane saves once its data is in; when the guards then re-export a PNG, one
  more save commit and post is accepted).
- No guides or checklists for the owner. Problem cards are the website's hand and bars flags (owner 11-07).
- The TTS save's global script is `TTS/Global.lua`; never edit the save by hand except as a deliberate edit (then
  move drift.py's BASELINE). No Lua patterns (MoonSharp). Test a Lua change with `~/.venvs/tts-checks/bin/python
  TTS/checks/lua_mock.py "TTS/Impossibility Simulator.json" TTS/Global.lua`; the whole suite:
  `TTS_HISTORY=~/Documents/Tools/wt/scratch/tts/history.git ~/.venvs/tts-checks/bin/python TTS/checks/run_all.py`
  (13 checks).
- "Rebuild the save now" (owner, per request only): cancel Card sync's waiting run, then `gh workflow run card-sync.yml
  --ref main -f bake=pending`; first check that the site deploy holding any rulebook change has finished.
- Push locks: `mkdir ~/Documents/Tools/wt/scratch/s1107/ac-push.lock` / `tc-push.lock` / `ps.lock` (rmdir after). AC:
  fast-forward only onto a fresh origin/main, tests green. A TC push holds its lock until every run it started has
  finished (`~/Documents/Tools/wt/scratch/s1108/watch_runs.sh`; since 11-12 it keeps the lock and exits 1 on a failed
  run: fix forward or revert under the lock). Never wait for a lock more than 90 minutes: report instead.
- Headless browser: only `~/std-work-1107/chrome/chrome-headless-shell` (the firewall asks about every new program
  path), cwd `~/std-work-1107`, its own `--user-data-dir ~/std-work-1107/chrome-<you>`, `--use-mock-keychain`, pages
  over http://localhost only; kill it and its children by PID in a finally block (memory `headless-browser-keychain`).
- Processes: kill exact PIDs only (never `pkill -f` a shell path); stop what you start.
- Workflows: never resume a multi-lane workflow with edited caps (memory `workflow-resume-multilane`); never
  SendMessage a workflow agent (it starts a second copy); pass long lists through a file and compare counts.
- CardBrowser (memory `card-app-project`): never start the app with a visible window on this Mac (`--self-test` or the
  hidden test mode only); run built apps only from `~/std-work-1107/cardapp-bin/`; no secrets, personal paths,
  Discord text or card files in the public repo; the app stays read-only; the site hook's tests
  (`tests/test_site_app_hook.py`) must stay green.
- Discord archive and card timelines: never copy message text into any repo; derived outputs live outside the repos.
  Answered review items go in with `apply_rulings.py`, a rebuild and `scripts/history_branch.py --push`.
  `~/Documents/Tools/firstrepo.git` is kept (never delete it).
- Dead and deprecated code is deleted (owner 10-23, 11-12), except the owner's KEEP picks (memory
  `owner-decisions-2026-11-12`).

## State (2026-11-21)

- AutoCroissant main ea0db694 (11-21 12:27; then this wrap-up's skills and docs commits), deployed on the jetson at
  0d2de792 (11-21 10:54, 42 commands): the same bot modules as main (ea0db694 changed only card_checks, card_plates,
  psd_edit, tests, a skill and a doc). Fork pin `cards-20261114` = 6d384060 (fork main; 8 fork releases this session,
  cards-20261111 to cards-20261114).
- TTSCardMaker main 0cb95c33 (11-21 12:27); data e111f436 (psd_details VERSION 13, 769 current records); data moves
  with Card sync; branches `history` (orphan, the timelines' pictures; head 680dbfba since 11-19), `plates` (5e1e786f),
  `previews`. Last whole-tree Card checks (11-21): 769 records, 0 hand, 33 bars, 1 name (Beholster).
- CardBrowser: `MichaelJSr/CardBrowser` main 3b14621, pre-release v0.1.0; source `~/Documents/Tools/wt/cardapp`.
- After the wrap-up cleanup: AC keeps only the wrap-up's own worktree `~/Documents/Tools/wt/wrap-1121` (branch
  `work/wrap-1121`) and the local `archive/ai-generation`, `archive/discord-edits`, `archive/local-update-stats`; TC
  has no worktree and no local branch but main; the fork keeps `work/research-1110` (be32812f, r15-2). Everything
  else is in the bundles (above). The Desktop TC clone is at main 0cb95c33 (an untracked `data/` from 11-19 left as
  found); the Desktop AC clone was left untouched by rule (e35327a4, 594 commits behind); the fork's Desktop clone is
  at da069245 (105 behind fork main).
  GitHub has only main, data, history, plates and previews (TC) and the archive/* branches (AC, fork).

## Owner

- [ ] CardBrowser's first trial (above).
- [ ] Check `MDW/Templates/Raid_Boss_Template.psd` (from 11-06; 20-cell split bars, 0-40). Raid bosses (6+ stars,
      `card_models.RAID_STARS`) have no stat limit.
- [ ] `~/Documents/Tools/firstrepo.git` (51.8 GB, self-contained since 11-17): the owner moves it to an external disk.
      The old pre-wipe folder `~/Downloads/Old TTSCardMaker` is no longer needed: the owner may delete it.
- [ ] Choices made unattended since 11-18, for a later look: the report's section "Conservative choices taken while
      unattended".
- [x] Manual pass on Needs handling: empty since 11-13 (0 hand flags; Tainted Lazarus' flag removed by the owner).

## Jetson

- [ ] **davey 0.2.0 held:** its DAVE API needs prepare/execute_transition, which discord.py 2.7.1 never calls (voice
      would break). Upgrade with a discord.py release that supports it.
- [x] **NVMe unsafe shutdowns** (11-10): an orderly shutdown keeps the count (107 before and after the one planned
      reboot); only plug cuts raise it, accepted. Restarts use the Shelly plug (Standing rules).
- [x] **Python 3.14.10** (11-10) on both bots; the 3.14.8 venvs, build trees, Docker rollback images and
      `~/maint-20261105` were deleted 11-11 (owner); recipes in `~/build/python` kept.
- [x] **Bot-only venv** (11-11): no fork, no Cython; `deploy.sh` builds nothing for a repo without setup.py and deletes
      untracked build output. opencv-python left with /frankenstein (11-12).
- [x] OJWikibot's shutdown Traceback: the owner's OJWikibot session's ground since 11-11.
- Notes: Jellyfin keeps running (D25). ojwikibot is enabled at boot; both bot units use RestartSec=900. The units'
  ssh-agent drop-ins stay. smartd watches the NVMe into the journal. OJWikibot's game sync units
  (`ojw-game-sync.{service,timer}`, daily 05:00 UTC) are OJWikibot's: never touch them.

## Open owner decisions

None open. Settled since 11-10: the old pre-wipe clone (mined for the timelines, published 11-17); Futura and old-font
re-lays (the guards fix them silently, AC 4007d9d7); the light theme's `--muted` #686661; no Google indexing; the 10-26
looks kept; Discord pending-updates left; no TTS JPEG copies (11-07; banded PNGs landed instead, 11-11).

## Owner checks (nothing is blocked on them)

- [ ] The ring badges on real cards: the website, the TTS save and the rulebook PDF; deep starts; the one-colour
      version under overlays.
- [ ] TTS, first real-game checks (tested only in the Lua emulator): right-click Card details (in hand too; drag,
      A-/A+, Rules buttons, "+ Name" spawns), Find a card, the rulebook's Contents and search, hand zones following
      seats, turn markers, The Gates of Ahn'Qiraj's "+ Silithid", any Lua error. Since 11-11: hover help on status
      effects, timing tags and types inside the ability (those sections are gone from Card details), '+ <status>'
      tooltips, the lighter status colours, teammates' hand cards.
- [ ] Website, first real use: History "Revert to this version"; the rulebook (its look, TTS pages cut from it, the
      editor's cut marks); the preview trash buttons; Mentions and Unused; card timelines, the Morgue and the compare
      modes (2-up, Swipe, Onion skin, Difference) on a phone. Safari never ran.
- [ ] Rulebook, look over: see-through pictures wrap text to their edges, pictures hang out of their panels, `out=` on
      Getting Started's Kirby; Make space between list items and lines.
- [ ] Try the website editor's drags (anchor grid, Make space, blank space, snapping, undo) and report what feels off.
- [ ] Photoshop: Computer Virus' stat bars (added 11-06 by netverk, beside the 1200 px canvas, x 1245-1982: no stat
      edit can show; several creatures).
- [x] Open in Photoshop once: the smart-filter star move and the title moved back (X2, 11-10). Photoshop matched;
      Polevaulter Zombie's and Anubisath Guardian's small gaps came from their colour profile, converted to sRGB.
- Eventually, never blockers: Photoshop standardizing (one Name and one Ability layer per card, rulebook Header/Page
  Number layers; Mr Jawsum's name layer); the team's rulings draft; the team's split of Tainted Lazarus into two
  creatures (its types read light + undead until then).

## Developer

### Follow-ups from the final waves (11-20/21)

- [ ] The two timing tests that flake under `taskpolicy -b` at load 40-60 (Standing rules, Tests): make them hold
      under load or widen their timing.
- [ ] Beholster's 'name' flag (since data f0df1a6f, 11-20): the card prints 'The Beholster', its file is
      Beholster.psd. Either rename the file (the 11-11 file-name rule, a silent 'Card-sweep: rename' batch with its
      followers) or mark the difference intended. Left as found unattended.
- [ ] On a 390 px phone the editor's title box cuts off a long title (LEEROOOOOOOOY); on main before wave 2.
- [ ] x25 (closing a card dialog goes Back) needs the Navigation API: Safari before 26.2 and Firefox before 147 keep the
      old behaviour (a close adds a history entry). Only Chrome was checked.
- [ ] x23's known limits (not regressions): its mock-up is worse on Mystery Gift, Warcraftian Mage and Papa Chip; on
      Miracle Matter's full-size stacked box 'Ice:'/'Fire:' can be cut at the card's left edge (as on main).
- [ ] test_card_guards::test_a_steps_background_jobs_end_with_the_test: its loop ends by itself after 60 s since AC
      4a4dda5f (11-19); the try/finally cleanup the 11-20 note asked for was not added.

### Follow-ups from the earlier research rounds

- [x] Rulebook deep links on phones settle after the fonts (TC 95508fe8).
- [x] Earlier fonts window: the guards read every earlier icon-font era (AC 3353fcf0).
- [ ] Font builds look the same to a card maker: the build number goes in the name at the next build (D24, above).
- [x] The mock-up's erase memo (TC 529ab83f).
- [x] Rulebook PDF waits for its pictured cards (A12: AC 94d72b13, TC c400e039).
- [ ] The card-sync.yml main-lane push retry's warning (round 3 pipe2-push-retry): check that today's lines ("the push
      failed with main unmoved", "git push failed, but main is at ...") say what happened; reword if not.
- [ ] Only if they show up: one API call per Quiet poll (if the token runs short); `MALLOC_ARENA_MAX` on the jetson
      (if memory pressure appears); 403s on the guards' reads.
- [x] Small website items (11-10/11): search reuses chapter HTML (TC 3e7ecb1d), the card dialog's history and versions
      load when needed (f60f0a90), far pushes skip layout (c2bb5997), focus into the dialog (9fe8934b), the table map
      opaque (41ab8074), the profile editor's author counts (6bd3e115), small type icons from a 68 px copy
      (3903a5e3). Left: the type emojis' 3-line swap (bad3-2), only if type_emojis.py is touched.
- Revisit with their event: published deck buttons with the first public deck (wss-1); SEC-2's `pull_request` filter
  (`startswith`) if PR runs ever appear.

### Older items

- [x] Types: an edit that removes and adds at once closes the row (AC 20ceb81c).
- [x] Stat bars E2: a series hidden in every cell never shows (AC 25bc2bc3).
- [x] `docs/BUGS_AND_IMPROVEMENTS.md` live checks V1 and V4: verified offline 11-10. D6 (icon-font adoption) is ongoing.
- [ ] Fork, low priority: a tracking change breaks kerning (Quick Slash, Ominous Red Circle), tracked badge ligatures,
      Photoshop's underline box (Golden Metronome).
- [x] OJWikibot (`ojwiki_helpers.py:263-268`'s mwparserfromhell fallback, the backup search missing `Backup:
      pending.json`, O-6, All_Cards_Event read once for a lone "===Permanent Rotation===" heading, the 10-21 mixer
      gap): the owner's OJWikibot session's ground since 11-11 ~09:15 (handover), like its shutdown Traceback.

### Kept by ruling (no action)

The 14 old rulebook PSDs keep their old badges. Future Niiue's ringed line ends 2 px from the edge. OJWikibot (owner
10-26): `/update`'s diverged-checkout handling and the field-events snapshot. Cards: 42 glow and gradient FX layers; 6
outline FX layers that change colour; the clock emoji; Miracle Matter's form pictures and the 5 card-art inline
pictures; PSDs over 50 MB left alone (11-09). Stars option B: the 50% floor and two-wide columns (11-10). Card
Authors.txt keeps the old rule-page credits (11-12). Hero Cap keeps 3 stars (D6); Ze'mer's preview flower bits (D16);
the guards keep rewriting makers' link paths (D17). Ancient Fuelweaver, Wickerbottom, Crock Pot and Holo Defense API keep
their stray-font characters (11-17). Hidden-bar leftovers on rows of their own colour stay unflagged; Baba's stats bake
"even if it looks wrong" (11-13). Comp-bar cards at 0 keep the exact mock-up (11-21). Searches: exact words only (D20);
apostrophe forms kept; a series name in /query answers its series (11-20). Several creatures: searches and the Stats
bars match any creature; previews and the table show creature 1 (11-18). The forum's role keeps Manage Threads
server-wide (11-16). The jetson's autocroissant.service on disk changed 10-20 and was never daemon-reloaded (owner's
call).

## Queue (`docs/Future/`)

| File | What is left |
|---|---|
| `CARD_CREATOR.md` | In-browser card creator: designed, not queued (owner 11-10: not yet) |
| `SIMULATOR.md` | Simulator against the AI or by the numbers: designed, not scheduled (owner 11-10: not yet) |
| `TTS_CARD_FORMS.md` | Stage 0 passed; parked (owner 11-13) |
| `CARD_TIMELINE.md` | Built and live 11-17; more only when the owner asks |

## Not chosen

Answered no by the owner: 24 first-screen tiles, an opaque top bar (11-04); editor ideas, idea 14's orphaned-FX check, a
TTS Workshop uploader, Google indexing (11-10); TTS JPEG copies (11-07); the merge guard (11-12: not unless it happens
again); old names from Discord as aliases (11-13); the 152-card Strong-type text redraw (11-13: later bakes redo it); a
startup check, admin alert or next-post check for the forum votes (11-16); linking 'die', 'defeated' and 'kill' in
search (D20); the comp-bar zero-row picture speed-up (11-21); a local bake in CardBrowser and a "use my local folder"
site feature (11-20). Research ideas accepted and rejected this session: titles in the report's `accepted.md` and
`rejected.md` (don't re-propose). Older: the `dropped` lists of `opt-round3/plan.json`, `opt-round4/plan.json`,
`opt-1104/round2-plan.json`, `~/Documents/Tools/wt/scratch/opt/PLAN.md` §(b); parked ideas:
`docs/Finished/session-2026-10-24/README.md`.
