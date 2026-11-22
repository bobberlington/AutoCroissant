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

The Desktop AutoCroissant checkout (add5194d), the fork's `~/Documents/Tools/psd-tools` (977e4067) and the Desktop
TTSCardMaker clone (5f22cef0) were fast-forwarded to main at 21:25 on 11-21 (the stray `caches_now.tsv` deleted
earlier). If a checkout is behind, `git pull --ff-only` first (TC with `GIT_NO_LAZY_FETCH=1`). The session ended at
21:22 on the owner's "Wrap up now, defer all 5": nothing is half-landed, five built items wait in bundles (first under
Waiting), and the jetson runs AC add5194d (main). `~/Documents/Tools/firstrepo.git` (verified intact) and
`~/Documents/Tools/discord-archive` move to the owner's external SSD (11-21 night): when a task needs the pre-wipe
history or the archive, ask the owner to plug the SSD in. `~/Downloads/Old TTSCardMaker` is gone: it was moved to the
Trash through Finder at 15:02 on 11-21, before the owner's 16:54 "delete it", and the Trash emptied (who did it is not
recorded); firstrepo.git holds its history.

## Starting prompt

```
ultracode. Work from ~/Desktop/AutoCroissant. Start with the plan: git -C ~/Desktop/AutoCroissant pull --ff-only, then read docs/NEXT_SESSION_PLAN.md. Read the memories its "Where to start" lists, then follow its Order:
1. Research rounds: about 8 per lane (actbake, cards), ship every 4 rounds, seeded from "Waiting from this session"; never re-propose a title the last report lists as accepted or rejected; filter NO BUILD / NO CHANGE records.
2. Then finish everything the rounds found: build every waiting idea, and give each rejected idea whose reviewer confirmed a real gain one more try (two judges, rebuild with the objection fixed, a fresh skeptic). All of it lands before the wrap-up.
3. The plan's other open tasks beside the rounds, at most 2 workflows at a time.
Ship under the push locks with the full suite and Card-sweep trailers on silent card rewrites; deploy to the jetson whenever bot-loaded code changes. Keep load low (I may be gaming). Work unattended: don't ask me questions; when unsure take the conservative choice and list it in the session report. At the end, wrap up: docs/NEXT_SESSION_PLAN.md, the session report under docs/Finished/, skills and memory; adjudicate and clean worktrees, branches (record SHAs and bundle anything unlanded first), scratch, processes and review pages; then give me the next session's starting prompt.
```

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
4. **Card timelines**: CLOSED (owner 11-21 21:45: "i dont plan to do timelines work again"). The live site keeps the
   `history` branch b8e865a8; per-card image versions keep updating from main in Actions. If ever needed: plug in the
   SSD and set `DISCORD_ARCHIVE` and `TIMELINES_FIRST_GIT` to its paths.

The five deferred items (first under Waiting) are built: they need only a rebase, the verify for r44-2 and x37, the
full suite and a land.

## Waiting from this session

### Deferred from 11-21 (do these first)

Owner 11-21 21:22: "Wrap up now, defer all 5". Workflow ships2-1121 built all five (owner 16:54: retry x31, r52-1,
r44-2 and x37; the Papyrus mock-up "Fix it now") and was stopped before any landed. Every tip lives only in
`~/Documents/Tools/wt/scratch/s1110/bundles/s2-deferred-1121-ac.bundle` and `s2-deferred-1121-tc.bundle` as
`refs/heads/s2/<item>` (full SHAs in that folder's `INDEX.tsv` and `wrap/branch_record.tsv`; prerequisites AC 847b1fe0,
b9aa05e3 and TC b62f36de, f66c280c, ee16083b, all on main). Each needs a rebase onto today's mains (AC add5194d, TC
5f22cef0), the full suite and a ship under the locks (TC first where the AC tests read the site). Get one back:

```
B=~/Documents/Tools/wt/scratch/s1110/bundles
git -C ~/Desktop/AutoCroissant fetch $B/s2-deferred-1121-ac.bundle refs/heads/s2/x31:refs/heads/work/x31
GIT_NO_LAZY_FETCH=1 git -C ~/Desktop/TTSCardMaker fetch $B/s2-deferred-1121-tc.bundle refs/heads/s2/x31:refs/heads/work/x31
```

- [ ] **x31, the D15 rest** (owner-directed 11-17 ~00:30; from r51-1): a typed character the line's font can't draw
      keeps the font the card prints it in, so an edit never erases it. Verified keep: the old objection is fixed
      (Minecraft Panda's round trip 0 px, its '2' keeps Minecraftia and its shift); round trips Wickerbottom 23,161 ->
      480 px, Ancient Fuelweaver 21,516 -> 104, Crock Pot 13,956 -> 117; 329 targeted tests. Landing bumps psd_details
      VERSION 13 -> 14 (71 cards gain "faces"): one Card sync re-read of every record. The verifier flagged for the
      owner (not blockers): NiseSegaSonic's own '"' is a huge 2-shaped glyph, so a typed curly quote prints like 'CALL
      IT 2BOB2' and Sonic's Ability re-lays about 17% smaller (D13 + D15; main prints the same for a straight quote);
      the mock-up then shows that font's '%' on Sonic's redrawn line 2; `site/NOTES.md` still says kept letters follow
      Battle Spear's Chiller '%'. `s2/x31`: AC 16ab7835 (8 commits on 847b1fe0), TC bb16d3d0 (4 on b62f36de). Left:
      rebase, fix the NOTES.md line, full suite, land, watch the VERSION 14 re-read.
- [ ] **r52-1, titles over the stars** (actbake): a title edit that would print over the stars or type icons is set
      smaller until it clears them (13 of 13; the rest byte-identical to main). Verified keep: the round's objection
      (a second edit of a shrunk title printed full size over the icons) is fixed, checked with chained bakes on 11
      cards, together with r40-1's stars undo. `s2/r52-1`: AC 8d7c1440 (2 on 847b1fe0), TC a48251a2 (1 on f66c280c).
      Left: rebase, full suite, land.
- [ ] **Papyrus mock-up** (owner "Fix it now", 16:54): on The Great Papyrus the mock-up drew added words over the
      spaghetti picture beside the Ability text; now pictures over a text layer stay over the new words, as the bake
      lays them. Verified keep. `s2/papyrus-mockup`: AC 605ebbd3 (1 on 847b1fe0, "Plates: pictures over a text layer
      as covers"), TC b445f691 (1 on f66c280c). Left: rebase (TC applies with `git am -3` over ee16083b's mockup.js),
      full suite, land.
- [ ] **r44-2, the Stats table** (cards): a Stats chart's table is drawn on its first Table press, not built hidden on
      every visit (#/stats 18,929 -> 6,769 elements); the round's objection (Series shown while Subseries is pressed) is
      fixed in the build. Built, verify unfinished (ships2 stopped). Two variants: `s2/r44-2` (builder: AC ac4deb7e on
      847b1fe0, TC d02ae198 on b62f36de) and `s2/r44-2-v` (the verifier's rebase: AC 3151073b on b9aa05e3, TC ae131385
      on ee16083b). The AC test needs the TC change landed first. Left: finish the verify, rebase, full suite, land.
- [ ] **x37, the Preview tab** (actbake): the tab where a preview is pressed does the checking, and a "starting" press
      now counts for the tab in front (the retry's objection: a move to another tab before the dispatch answers waited
      about 18 s). Built, verify unfinished. `s2/x37`: AC 6fce267f (3 on b9aa05e3), TC 85f8c0d6 (3 on ee16083b). Left:
      finish the verify, rebase, full suite, land.

### Research items

Owner 11-21 15:01 on the held items: "retry the most impactful ones, to finish within ~3 hours. drop the rest";
on the 7 stranded keeps: "Ship now, re-verified". Workflow held-1121 did both. Landed on 11-21 evening:

- [x] x30: the page under a card dialog goes inert after the paint (TC f66c280c, AC tests 5e6080b4; 18:51).
- [x] actbake r40-1: a stars shrink comes back to the maker's stars on an undo (AC b9aa05e3, TC ee16083b; 19:23), with
      fork r11-0 (template bar pixels read raw) in the same AC push.
- [x] Fable fork r1-0 (CPU parity nudge), r2-0 (history_pairs tool), r2-1 (Group.new() pass-through + same_picture
      tool): fork main 619b5f4b, tag `cards-20261121b`, AC pin 1492173c (20:38).
- [x] The bot's /query skips the website's and the TTS save's PNGs (header-logo skeptic): AC f4c08596, deployed 21:05.
- [x] Website favicon and header logo (owner's icon B1; ringless logo in the text colour): TC 92e0e24 (21:05).
- [x] Timelines owner answers 3 (16:54): history b8e865a8, TC 5f22cef0, AC add5194d, deployed 21:05.
- [x] Fork CI fix (owner "Fix the tests now"): fork main 977e4067, Tests 19/19 green (run 44987185766), the first
      green run since 11-15; test-only, no release.
- [ ] **Early save, watch item (d)** (r60-1, landed in wave 2a: TC d01f81cd, notes (a)-(c) fixed in 79d79682, checks
      0911ce04): the guards' Land onto a main that a Card sync save has already moved has never run on a real push.
      The early save is proven only by the next guarded push that re-exports a PNG: watch Land retrying, the
      follow-up's re-pin, and the early 'Card sync: TTS save updated' commit with its one extra #github-updates post.

Dropped (owner 15:01 "drop the rest"; held-1121's two judges each; don't retry unless the owner asks). Tips kept in
the bundles (below):

- actbake r1-1 (Ability on the old baseline): covered since; geoDraw draws on the plate's own baselines.
- x5 / actbake r10-0 (guarded push waits for Landed): gain gone (kept push lane) and against the early save r60-1.
- actbake r23-0 (Commit main skips the PDF wait): 3 pushes instead of 2 on some lanes, beyond the P-2 amendments.
- actbake r39-0 (new type icon from covering layers): worse on 6 of 32 cards, against the 11-09 icon-look ruling.
- actbake r42-0 (first preview press warm): race regressions on code that moved a lot since 11-16.
- actbake r54-1 (lifted Ability back to the maker's place): the mock-up half makes follow-up previews worse.
- x2 / actbake r7-0 (kept lines by letters): done since (geoDraw / geoPlan).
- actbake r7-2 (keep the card's letters in a one-line edit): covered since by geoPlan's splice.
- actbake r2-1 (shrink only with more lines): superseded by roomBelow (D12, x23, x34).
- actbake r60-2 (AWS warm-up): +3-4 s on AWS runners only (about 1 job in 5), under 1 s a job on average.
- cards r10-0 (related strips load in view): blank strips; the fix costs most of the byte gain.
- cards r10-1 (small per-card history file): done since (per-card versions files, 11-18).
- cards r21-0 (`sizes="auto"` grid tiles): bytes on 1x desktops only; status cards softer.
- cards r22-0 (smaller rulebook copies): bytes on 1x-1.5x desktops only; clashes with x16/x17.
- cards r25-0 (one screen look-ahead): no fix keeps the gain.
- x4 (mock-up keeps the card's pixels for same-looking edits): needs per-letter style data.
- x26 (lazy rulebook icons): blank icons are built in; wake() alone would be a new idea.
- x27 (Ability fit against the maker's text): its ink breaks Card sync's jq merge (E2BIG) once ~13 bakes wait.
- Details memo, fork r3-1 and r4-3: Actions CPU only (3-4% on unpicked cards), below the bar.

Stranded keeps not kept on re-verify: AC r11-1 (Plates: a merge's view redrawn on its rows only) superseded by
dc77c7b8; AC r9-2 (Bars: art fill reads the alpha alone), its gain mostly covered by 7790f7e3; Fable fork r1-1 (one
cards CI workflow) would delete the test workflow the owner asked to fix.

Bundles (rebuilt 11-21 21:23-21:33 per the owner's 15:01 rulings: landed and superseded tips out, rejected and dropped
tips kept): `ac-kept-1121.bundle` (43 refs), `tc-kept-1121.bundle` (36), `fork-kept-1121.bundle` (7), plus the two
s2-deferred bundles above, all in `~/Documents/Tools/wt/scratch/s1110/bundles/`. Refs keep their original names
(`refs/heads/work/<lane>-r<n>-<m>`, `refs/wrap/<sha8>`, `refs/heads/cards-1110-r44-2`, ...); `INDEX.tsv` lists all 98
(bundle, ref, sha, item, verdict; "(was <old>.bundle)" maps the old names), `DROPPED-1121.tsv` the 108 tips left out.
The SHAs in the report and branches.md are unchanged. Get one back with `git fetch <bundle> <ref>:refs/heads/<name>`
(in TC with `GIT_NO_LAZY_FETCH=1`).

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
- [x] **Fork r15-2**: colour glyphs in Smooth and Sharp drawn as Strong. Landed 11-21 15:41 (owner: "do it now"):
      fork main 82be96d2, tag cards-20261121; AC dcf0b6f9 'Pin cards-20261121 (fork r15-2)' (full suite 1289). Fork
      Tests 44962115008 red on the same 17 tests as the old pin's run 43690498999 (16x
      test_gradient_lookup::test_the_lookup_is_interp1ds on windows/3.11, test_fonts::test_platform_dirs_linux):
      there since 11-15, none new. The run deleted the fork's `work/research-1110` (be32812f; left out of the
      rebuilt bundles as landed, SHA in `wrap/branch_record.tsv`).
- [x] Details memo (AC `work/r3-1` 22e09b0c, `work/r4-3` 485dd38b): dropped 11-21 (owner 15:01; held-1121's judges:
      Actions CPU only, below the bar). Tips in `ac-kept-1121.bundle`.
- [ ] **Retire game reports** (owner 11-13: after the timelines, which closed 11-21): delete
      `docs/Campaigns/GAME_REPORTS_CAMPAIGN.md` and the `/report_game` lines in `docs/Campaigns/SIMULATOR_CAMPAIGN.md`
      and `docs/Future/SIMULATOR.md`.
- [ ] **D24**: put the stoogedings build number in the font's name at the next font build (no build since 11-15).
- [ ] **TTS Saves folder** (owner 11-13): delete the old test save 'AC test 11-11 tooltips 2'. The folder's copy of the
      save is from 11-18: copy main's in again after the last save change.
- [ ] Photoshop 2023's JSX PNG saves have come out untagged and up to 55 levels off the card PNGs since 11-16 ~04:12,
      on untouched PSDs too. Cause not found. Open tests are fine; don't use new Photoshop exports as pixel truth until
      it is found.
- [x] **Desktop TC promisor duplicates**: the wrap-up deduped pack-185d66d5, 4080ec91, 239227ea and c94ca1fd (clean
      dry runs, `fsck --connectivity-only` passed) and removed a 0.91 GiB tmp pack: TC packs 32.3 -> 28.2 GiB. The
      second cleanup (11-21 afternoon) deduped 12 more pure copy packs: 28.2 -> 25.9 GiB (fsck ok, 0 objects missing).
- [ ] **Desktop TC pack files**: the clone holds 1,178 pack files (11-21). A careful promisor repack at a quiet time
      (no agents running; every object and promisor flag kept; `fsck --connectivity-only` after); never a gc or fetch
      that lazy-fetches.
- [x] **Two TC packs of unknown origin**: found and kept. pack-333ff011 (8.47 GiB) is the fresh blobless clone's
      checkout of 11-10: 2,575 blobs found nowhere else, all reachable (2,109 in today's checkout, 466 older versions
      on main; 343 in the AC tests' pin tree). pack-8cfce261 (0.98 GiB) is the timelines lane's blob fetch by id of
      11-15: 191 older card image versions, all reachable. Dropping either would lazy-fetch or break the clone.
- [x] Cleanup, third pass (11-21 21:23-21:40): the second pass's 25 vetoed `scratch/s1110` actions done (rebuildable
      bulk deleted, 13 archives packed into `scratch/s1110/archive/`), old scratch packed with the cited files kept
      (opt-round3/4's `plan.json`, opt-round4's `skeptic-cm-a1-tools/proposed2.py`, `s1107/status/census.*`;
      `alt-template/` left whole), the APFS clone dedup run (the 71 PSDs were already clones). `scratch/s1110` 11.74 ->
      2.31 GiB, about 10 GiB freed. Every action: `scratch/s1110/wrap/cleanup3_actions.tsv`; plan and what was held
      back in `wrap/cleanup3/` (`plan3.tsv.gz`, `excluded3.tsv.gz`). Left in place on purpose: `timelines/`, `s2/`,
      `h1121/`, the deferred items' folders (any path naming x31, x37, r44-2, r52-1 or papyrus), r9-2/r11-1/Fable r1-1,
      `strand-a/b/c`, `t1-land`, the `ojw-*` round folders (`ojw-handover/rounds.md` points at them), `x23-merge`,
      `v-x13`. `~/std-work-1107` holds only `cardapp-bin` and `chrome`. `land1121/suite.sh` was copied back out of its
      pack for the final docs suite.
- [ ] Cleanup leftovers, small: `scratch/s1109/wrap/ac-actions-1108.bundle` and `tc-actions-1108.bundle` (42 KiB)
      kept (landed rebased); `scratch/s1109/wrap/tc-unlanded-1110.bundle` (the previous session's record) not
      adjudicated: a fetch into a temp repo failed. `scratch/s1109/wrap/fetchblobs.sh` points at the deleted
      `tcx-repin` clone. Memory `card-timelines-project` still names the deleted review pages (timelines closed: no
      action needed). The Discord archive's local preview site moves with the archive to the owner's SSD.

### CardBrowser (owner 11-20/21)

- [ ] **The owner's first trial** of v0.1.1 (`MichaelJSr/CardBrowser`, public, a quiet GitHub pre-release: macOS arm64
      and x64, Windows x64). A read-only window around the live site, with Open in Photoshop, Show in Finder/Explorer
      and Copy PNG. CI's self-test passes 19/19 on all three. Never tried: macOS 11-14, a real Windows desktop, a real
      Open in Photoshop. No website link or announcement until the owner has tried it; the owner tells the makers.
      v0.1.0 is at 5d9f2bf; v0.1.1 (main 6437f11, 11-21 14:51) adds a README line (offline on Windows) and the app
      icon (the owner's logo pick). The site hook it uses landed 11-21: TC 016872be, AC 4149cf25 (tests).
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
      item (d) above).
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
  1294 passed at the final heads (AC add5194d + TC 5f22cef0, 11-21 21:01); the TC tree must be at or after 0cb95c33 (with an
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
  `~/Documents/Tools/firstrepo.git` is kept (never delete it). It and the Discord archive live on the owner's external
  SSD since 11-21 night: when a task needs them, ask the owner to plug the SSD in (`DISCORD_ARCHIVE`,
  `TIMELINES_FIRST_GIT` point the tools at its paths).
- Dead and deprecated code is deleted (owner 10-23, 11-12), except the owner's KEEP picks (memory
  `owner-decisions-2026-11-12`).

## State (2026-11-21, final)

Heads at the end (11-21 21:45); nothing in flight.

- AutoCroissant main add5194d (11-21 21:05, the timelines owner answers 3, after the /query fix f4c08596 and the
  `cards-20261121b` pin 1492173c), deployed on the jetson at add5194d (42 commands). Fork pin `cards-20261121b` =
  619b5f4b (10 fork releases this session, cards-20261111 to cards-20261121b); fork main 977e4067 (the CI fix on top,
  test-only, Tests 19/19 green).
- TTSCardMaker main 5f22cef0 (11-21 21:05: timelines answers on 92e0e24, the favicon and header logo); data ea50731e
  (psd_details VERSION 13); data moves with Card sync; branches `history` (orphan, the timelines' pictures; head
  b8e865a8), `plates`, `previews`. Last whole-tree Card checks (11-21): 769 records, 0 hand, 33 bars, 0 name flags.
- CardBrowser: `MichaelJSr/CardBrowser` main 6437f11 = pre-release v0.1.1 (v0.1.0 at 5d9f2bf); source
  `~/Documents/Tools/wt/cardapp`.
- Branches: every ref adjudicated in `docs/Finished/session-2026-11-10-21/branches.md` (and its 11-21 evening
  addendum); every tip recorded in `~/Documents/Tools/wt/scratch/s1110/wrap/branch_record.tsv` (244 lines) or, for the
  54 landed item branches the waves deleted, `scratch/s1110/final-1120/wave2a_shas.txt` and `wave2b_shas.txt`;
  unlanded tips only in the bundles (above). Local branches besides main: AC `archive/ai-generation`,
  `archive/discord-edits`, `archive/local-update-stats` (= GitHub), plus this docs run's `work/final-1121` until the
  orchestrator lands it; none in TC or the fork.
- Desktop clones: AC (add5194d), the fork's `~/Documents/Tools/psd-tools` (977e4067) and TC (5f22cef0)
  fast-forwarded to main at 21:25 (TC with `GIT_NO_LAZY_FETCH=1`; an untracked `data/` from 11-19 left as found).
- GitHub has only main, data, history, plates and previews (TC), main and the archive/* branches (AC, fork) and the
  fork's cards-* tags (newest `cards-20261121b`; plus about 125 upstream `v*`/`0.x`/`1.x` tags there, not ours).

## Owner

- [ ] CardBrowser's first trial (above).
- [ ] Check `MDW/Templates/Raid_Boss_Template.psd` (from 11-06; 20-cell split bars, 0-40). Raid bosses (6+ stars,
      `card_models.RAID_STARS`) have no stat limit.
- [x] `~/Documents/Tools/firstrepo.git` (48.4 GiB, self-contained since 11-17, verified intact 11-21) and
      `~/Documents/Tools/discord-archive`: the owner moves both to his external SSD (11-21 night); ask him to plug it
      in when a task needs them. `~/Downloads/Old TTSCardMaker` is gone (Finder Trash 15:02, emptied).
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

### Follow-ups from the 11-21 evening lands

- [ ] x30's kept trade-off: a press right after a dialog closes, with no pointer move first, can be lost (about 100 ms
      on a desktop); phones were not measured.
- [ ] r40-1 (stars undo after a shrink) has had no real round trip yet: check the first website stars edit that undoes
      a shrink.
- [ ] Fable fork r1-0's CPU parity nudge is untested on AVX-512 CPUs (test_cpu_parity passed on GitHub's ubuntu and
      windows x86 runners and on macOS).
- [ ] x31's flagged side effects (see its item under Deferred): the owner's look at NiseSegaSonic's quote and '%'.
- [ ] The /query fix (f4c08596) skips `site/` and `TTS/` PNGs (`card_edits.NO_CARD_FOLDERS`); open question: should
      `MDW/` and `Types/` PNGs (`query_card.NOT_CARDS`) also stop being found by exact name?
- Not planned (timelines closed, owner 11-21 21:45): 12 other Morgue entries have byte-identical repeat pictures; the
  owner ruled only on Fireball (Mario), shown once.

### Follow-ups from the final waves (11-20/21)

- [ ] The two timing tests that flake under `taskpolicy -b` at load 40-60 (Standing rules, Tests): make them hold
      under load or widen their timing.
- [x] Beholster's 'name' flag (since data f0df1a6f, 11-20; the card prints 'The Beholster' from Beholster.psd): the
      owner marked the names intended (TC b62f36de, 11-21 15:12); 0 name flags since.
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
| `CARD_TIMELINE.md` | Built and live 11-17; closed 11-21 (owner: no more timelines work) |

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
