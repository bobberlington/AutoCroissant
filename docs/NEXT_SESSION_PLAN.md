# Next session plan (after 2026-11-23)

## Where to start

Start in `~/Desktop/AutoCroissant`. Read memories `owner-decisions-2026-11-22` (and `-11-19`, `-11-15`, `-11-12`,
`-11-10`), `resource-limits`, `secrets-never-printed`, `maintenance-pushes-silent`, `photoshop-no-interrupt`,
`never-pkill-shell-snapshot`, `research-rounds-future`, `research-loop-records`, `workflow-bookkeeping-from-git`,
`land-agents-owner-approval`, `eta-and-long-tail-verifies`, `card-app-project`, `jetson-server`, `jetson-reboot-hang`,
`headless-browser-keychain`, `storage-space`, `persistent-work-paths`, `agent-model-preference`,
`no-sendmessage-to-workflow-agents`. What landed 11-21 23:40 to 11-23: `docs/Finished/session-2026-11-22/` (README,
accepted.md, rejected.md, branches.md); the session before: `session-2026-11-10-21/`. That session's binding lines
are in `~/Documents/Tools/wt/scratch/s1122/BRIEF.md`; the lasting ones are under Standing rules.

The Desktop TC clone (93e5892f, 06:20) and the fork's `~/Documents/Tools/psd-tools` (bb8c62fc, 07:17) were
fast-forwarded to main on 11-23; the Desktop AC checkout follows the wrap-up docs' push. The last check is in
`~/Documents/Tools/wt/scratch/s1122/wrap/final_state.md`. If one is behind, `git pull --ff-only` first (TC with
`GIT_NO_LAZY_FETCH=1`). Nothing is in flight or half-landed: both ship3s landed (cards 00:36, actbake 01:12) and every
owner answer of 11-23 too. `~/Documents/Tools/firstrepo.git` and the Discord archive are on the owner's external SSD:
ask the owner to plug it in when a task needs them.

## Starting prompt

```
ultracode. Work from ~/Desktop/AutoCroissant. Start with the plan: git -C ~/Desktop/AutoCroissant pull --ff-only, then read docs/NEXT_SESSION_PLAN.md. Read the memories its "Where to start" lists, then follow its Order:
1. Research rounds: about 8 per lane (actbake, cards), ship every 4 rounds from git (git cherry against main), seeded from the plan's research notes; never re-propose a title the last two reports list as accepted or rejected; filter NO BUILD / NO CHANGE records.
2. Then finish everything the rounds found: build every waiting idea, and give each rejected idea whose reviewer confirmed a real gain one more try (two judges, rebuild with the objection fixed, a fresh skeptic). All of it lands before the wrap-up.
3. The plan's other open tasks and the 11-22 follow-ups beside the rounds, at most 2 workflows at a time.
Ship under the push locks with the full suite and Card-sweep trailers on silent card rewrites; deploy to the jetson whenever bot-loaded code changes. Keep load low (I may be gaming). Work unattended: don't ask me questions; when unsure take the conservative choice and list it in the session report. At the end, wrap up: docs/NEXT_SESSION_PLAN.md, the session report under docs/Finished/, skills and memory; adjudicate and clean worktrees, branches (record SHAs and bundle anything unlanded first), scratch, processes and review pages; then give me the next session's starting prompt.
```

Research again at about 8 per lane: owner answer 19 (11-23).

## Order

0. Nothing is deferred: every verified item of 11-22/23 landed.
1. **Research rounds**: about 8 per lane, a ship every 4 rounds, then stop (owner 11-21 and 11-23 answer 19; don't
   ask for a budget). Lanes `actbake` (Actions, bake, previews, PNG export, the mock-up) and `cards` (parser and card
   data, TTS, the website, the bot's queries, the tests). No ideas wait; seed from "Research" below. The fork lane
   only if the owner asks (its lens is done: no linked picture is left that Card sync can't redraw).
2. **Then finish everything the rounds found**: every waiting idea built; each real-gain reject one more try (two
   judges first: still needed, no ruling against it, fixable). All of it lands before the wrap-up.
3. **The other open tasks** below, beside the rounds where their ground doesn't clash (at most 2 workflows).
4. **Card timelines**: CLOSED (owner 11-21). The live site keeps `history` b8e865a8; image versions update from main.

## Waiting from this session

### Research

- No ideas wait. Kept and landed 11-22/23: actbake 18, cards 33 (the finish phase w0, w1, x0-x7 and actbake x2 in
  the s3 ships); titles in the report's accepted.md.
- Not kept after one more try (rejected twice: don't retry): actbake x0 (Mock-up rows that start with spaces: Spur's
  second paragraph laid one space in), x1 (a growing Ability set smaller from the plate: same-length edits on The
  Ringed City, Whacka Bump, Papa Chip preview smaller). Dropped by both judges: cards r8-3 (PSD_DIRTY_VERIFY only in
  the edit tests: owner G3 keeps it on everywhere).
- Seeds from the 11-22 notes: the first maker guarded push after TC cbd4bb1d (see Waiting on events); x3's website
  half (the editor's series "PSD value" vs what a revert restores; x3's parser half landed); the text-colour
  boundaries where the bake and the mock-up still differ (Chris & Jordi, Pix, Grimmchild; report, Watch items); two
  linked-picture updates Card sync still refuses with the "could not draw it again" note (Retro Kirby's Mosaic at 2x
  above the canvas: needs a Photoshop oracle; warps under a trim: Elemental World x2, Mol D Truffy; fork lane only if
  the owner asks).
- Bundles of unlanded tips: `~/Documents/Tools/wt/scratch/s1122/bundles/` (`INDEX.tsv`: the 11-22 rejects, the fork
  lens rejects, the lanes' final branches); the report's branches.md. Older:
  `~/Documents/Tools/wt/scratch/s1110/bundles/` (the `s2-deferred-1121` tips all landed 11-22).

### Other open tasks

Finished 11-22/23 (see the report): the five deferred items (x31, r52-1, Papyrus mock-up, x37, r44-2), fork dead code
and linked pictures (`cards-20261122`, then Reduce Noise in `cards-20261123`: unredrawable 46 on 40 cards -> 0), game
reports retired, the Photoshop PNG export "issue" (no fault), the 11-08 trailer check (nothing to undo), Beholster's
series and Smoldering (owner 12-13), the 11 rulebook typos (owner 18, 23), the fork's composite docs (owner 21-22), the
fork CI cut to 2 jobs (owner 25; fork bb8c62fc, Tests 45203571689 green, so 1b970405's red run needs no rerun), the
Desktop TC repack (owner 6: 1189 packs -> 2, fsck ok), the s1109 bundles, CardBrowser v0.1.2.

- [ ] **TTS Saves folder**: the 11-23 01:40 copy is main's save at 25077c28; main's save changed since (93e5892f, the
      rulebook PDF's address): copy main's in again unless `final_state.md` says the cleanup did.
- [ ] **D24**: the stoogedings build number in the font's name at the next real font build (owner 11-22: no build
      just for it).
- [ ] **CardBrowser** v0.1.2 (main 4c1bd38): quiet (owner 11-22); the owner tells the makers; Linux only if a maker
      asks.

### Waiting on events

Proven 11-22 (report): the delta-free Land, PNGs recompressed after Decided, the Follow-up window past Landed (runs
44839860535, 44839860345), x12's WebP encode (45082969963), the push-lane Rulebook PDF on its own artifact
(44987581999), psd_details VERSION 13 -> 14 -> 15 through gate G4.

- [ ] **The first maker guarded push after TC cbd4bb1d** (r4-0: Follow up and the post in the check job): run
      `~/Documents/Tools/wt/scratch/s1122/research5/actbake-1122-s1-prep/land/first_push_check.sh` (kept, like
      `scratch/s1122/tasks/jsx-png/`). If its guards re-export a PNG it also proves the early save (r60-1: Land
      retrying, the follow-up's re-pin, the early 'Card sync: TTS save updated' commit and its one extra post). The
      kept push lane needs a push whose fixes keep every picture. The check job's 60 min is far off (longest real push
      445 s).
- [ ] **The first real Preview press since 11-19**: the preview reads at the commit, x20's 250 ms hand-over, x34's
      caption, x37 (the tab in front checks), r5-1 (the blob post takes about 0.4 s: read the lane log).
- [ ] **The first website edit that bakes** (none since 11-19): one Card sync, not a second taking over the after-bake
      job; the first live `+ability` bake (12 of the 18 minions have room); x33 (a moved effect-layer line keeps its
      look); Mosling, Moose Goose and Wickerbottom title copies on their arc; Tainted Jacob's parity; r40-1 (a stars
      edit that undoes a shrink).
- [ ] The next named-cards Card sync dispatch: its site build and Card checks start with its data job.
- [ ] The next runner Python patch (still 3.14.10): the cached venv re-pointed, not rebuilt.
- [ ] Rulebook PDF: the next zip 404 (r5-0 asks again; the 404 hit run 45071425967 before it landed); the next push
      that changes the TTS pages' text (r6-1 ends the push's look); a `--baked` save that takes a new name.
- [ ] The first real change of a linked file on a card the fork releases made redrawable (`cards-20261122`, and
      Burly Bomb's Reduce Noise in `cards-20261123`): Card sync redraws it (no "could not draw it again" note).

## Standing rules

- **Unattended**: ask nothing unless the owner says "ask me questions". When unsure: two independent judges, both
  certain and agreeing = do it; else the most conservative choice that publishes nothing, listed in the session report.
  Asked for questions: only the uncertain ones, pictures in the page. Agents on Opus or lower; one Agent per build.
  Owner picks are never dropped: a review may only flag them. A land agent needs the owner's words quoted, or land
  inline.
- **Commits**: to main in both repos; `work/*` only in flight, deleted after (SHAs recorded, unlanded tips bundled).
  Very short messages, author MichaelJSr, no Claude attribution (`git log --format=%B <range> | grep -ic co-authored`
  prints 0 before any push; plumbing bypasses the hook). Never `git stash`, never force-push a main. Persistent paths
  only (`~/Documents/Tools/wt/`, `~/std-work-1107/`). Never run a script with a Desktop checkout as its working folder;
  outputs to absolute paths under the session's scratch (11-22: a script wrote into the Desktop AC tree).
- **Load** (owner 11-22: "Always keep it low"): heavy work (tests, bakes, renders, builds, Chrome) under `nice -n 15`;
  while `pgrep -x RotMGExalt.real` finds the game, `taskpolicy -c utility nice -n 15`, never `taskpolicy -b` (it pins
  work to the E-cores the other sessions keep busy: a full suite took 2:05 h with false timeout reds; 29 tests 711 s
  vs 22 s under utility). pytest `-n 3`, pools at most 2, at most 8 git processes, at most 2 workflows, one full AC
  suite on the machine at a time (`~/Documents/Tools/wt/scratch/suite-wait.sh` first). Never a partial clone of the
  local card repo; no gc or repack of big clones while agents run; the git watchdog
  (`~/Documents/Tools/wt/scratch/git-watchdog.sh`) under Monitor in long unattended work. Disk: at least 100 GB
  available counting purgeable and 50 GB truly free (`scratch/s1110/tools/diskfree.sh`); delete bulky outputs.
- **Processes and secrets**: kill exact PIDs only (never `pkill -f`/`killall` by a shell path or a broad name); stop
  what you start. Never print config.py values, tokens, keys, webhook URLs, or other processes' command lines or
  environments; a token goes only to its own service; repo hooks: name, active flag and host only.
- **Tests**: from your AC worktree, `nice -n 15 env AUTOCROISSANT_LOCAL_DIR=~/Desktop/TTSCardMaker AC_SITE_SRC=<tc tree>
  ESBUILD=~/Documents/Tools/wt/scratch/esbuild/node_modules/.bin/esbuild ~/.venvs/ac-py314-fork/bin/python -m pytest
  -n 3 -q -p no:cacheprovider tests`, never with GIT_NO_LAZY_FETCH exported. `AC_SITE_SRC`: a sparse TC tree at or
  after TC main (`.github site TTS/Global.lua TTS/checks Global.xml decks.json`; without `TTS/checks` the run stops,
  conftest `SITE_NEEDS`). 1350 passed at the final heads (AC 541feb9c + TC daca7927, 11-23). Flaky under heavy
  load, re-run alone before calling a red real:
  `test_card_guards::test_the_checks_reports_come_once_that_job_ends` (reads checks.result as soon as the file exists)
  and `::test_the_bot_clone_and_records_run_here` (waits 15 s for a clone); the two timing tests of 11-21 hold since AC
  5700e58e. Targeted files unless a full run is needed; at most a few new tests per change. TTS: a Lua change with
  `~/.venvs/tts-checks/bin/python TTS/checks/lua_mock.py "TTS/Impossibility Simulator.json" TTS/Global.lua`; the
  whole suite `TTS_HISTORY=~/Documents/Tools/wt/scratch/tts/history.git ~/.venvs/tts-checks/bin/python
  TTS/checks/run_all.py` (13 checks).
- **Research**: a new idea must be something a person would notice: a wrong card, picture, preview or bot answer
  fixed (name them), a real failure fixed (name the runs), or at least 5 s or 10% off a wait (owner 11-16). Measure
  first. Finders return an empty list when nothing clears the bar (never a NO BUILD / NO CHANGE record; filter them in
  code). A finder returns within about 60 minutes, measures on a sample of cards, runs or pages, and never runs a full
  AC suite (the tests lens reads the ships' suite logs or times single files); what it could not finish goes in its
  notes. A skeptic per build; a ship every 4 rounds.
- **Workflows: ship from git.** Key items by the branch name the script gave the builder, never an agent's free-text
  field; decide what ships with `git cherry origin/main work/<lane>` in both repos inside the ship step and log a skip
  loudly; never trust a script's own accepted list; before the wrap-up diff every integration branch against main by
  patch-id (11-22: research5 skipped ships s2 and s3 as "nothing to ship" with 15+ verified commits on each lane).
  Never resume a multi-lane workflow with edited caps; never SendMessage a workflow agent; pass long lists through a
  file and compare counts.
- **TC work clone**: each session makes its own blobless GitHub clone, full history, no checkout
  (`git clone -q --filter=blob:none --no-checkout git@github.com:MichaelJSr/TTSCardMaker.git
  ~/Documents/Tools/wt/tc-<mmdd>`); TC worktrees from it are sparse (`worktree add --no-checkout <wt> -b work/<name>
  origin/main`, `sparse-checkout set --no-cone <paths>`, `checkout`; a full checkout is 16+ GB). Missing blobs come from
  GitHub in batches. The Desktop TC clone is read-only for agents (tests' `AUTOCROISSANT_LOCAL_DIR`, the guards'
  `--clone`, reading blobs): every git call with `GIT_NO_LAZY_FETCH=1`, never a fetch, commit, checkout or worktree
  there; a blob it lacks is read from the work clone. Only a land step moves it: `fetch origin main` and `merge
  --ff-only` with lazy fetch off, when clean, the new blobs copied in from the work clone (`git hash-object -w`).
  `git write-tree` needs `--missing-ok`; refresh `origin/data` with `git fetch origin +data:refs/remotes/origin/data`;
  a fetch on top of local commits makes promisor duplicates (dedup at a quiet time, memory
  `partial-clone-fetch-packs`). The 11-23 final cleanup handled `tc-1122` (`scratch/s1122/wrap/final_state.md`).
- **Shipping**: locks `mkdir ~/Documents/Tools/wt/scratch/s1107/ac-push.lock` / `tc-push.lock` (`|| exit 1`; trap the
  rmdir once held; never wait more than 90 minutes: report). TC first when the AC tests read the changed site or
  workflows; a TC push holds its lock until every run it started has finished (`scratch/s1108/watch_runs.sh`, exit 1
  on a failed run: fix forward or revert under the lock). AC: fast-forward onto a fresh origin/main with the full suite
  green at the final head, then the `jetson-deploy` skill when a file the bot loads changed (a static walk from
  main.py); deploys always restart; the deploy lock is shared with OJWikibot (let it wait, never kill or force it).
  After an AC push that changes what Card sync writes, dispatch one Card sync (11-22: the TTS rulebook headings waited
  31 min for the schedule). Then fast-forward the Desktop checkouts if clean, and append one LANDED line to the BRIEF.
- **Fork releases**: fork main fast-forward, tag `cards-<yyyymmdd>[b]` on the fork's origin (never upstream, no PRs),
  AC `requirements-cards.txt` re-pinned (no jetson deploy: the jetson has no fork); right after that AC push one Card
  checks dispatch on TC main that must end green (re-pin step 5: it builds the runners' venv once); then the shared
  venv `~/.venvs/ac-py314-fork` re-pinned. A release that changes what Card sync writes gets a release-level Photoshop
  2023 open test of the PSDs it reaches before any push (11-22: 133 copies on 75 cards). After a re-pin, AC code from
  before it can fail on removed psd_tools parameters (TypeErrors in card_standards / psd_edit): builders and merge
  agents rebase onto AC origin/main first; such a TypeError in a verify means a stale base, not a defect. Fork CI: 2
  jobs, Linux + macOS on Python 3.14 (owner 11-23; was 19 jobs); no per-round changelog entries (owner fork-r6-1).
- **No test branches on TTSCardMaker's GitHub** (every push posts in #github-updates): Actions by dry dispatches on
  main or locally; a prepare step may prove a change on a `scratch/<lane>-<n>` branch of the fork's origin, deleted
  after. `gh run list -c` needs the FULL sha. Multi-step shell as a `set -euo pipefail` bash script (zsh arrays start
  at 1 and zsh doesn't split words).
- **Card pushes**: the guards dry first (`python -m commands.card_guards check --clone ~/Desktop/TTSCardMaker --repo
  MichaelJSr/TTSCardMaker --before origin/main --after <head> --mode dry --out <dir>`; discord.json must be `{}`).
  Every scripted TC commit that changes card PSDs or PNGs without a real card change carries `Card-sweep: <kind>`.
  Printed wording waits for the owner; an owner-approved wording fix goes in its own commit without the trailer.
  Research card fixes (owner 11-19): only when two independent judges agree the fix is invisible (looks the same by
  eye) and safe; one rewrite per card per pass, a Photoshop open test of every rewritten PSD
  (`scratch/s1110/ps/open_all.sh`), one commit per card, and the card's blob on origin/main re-checked right before the
  push (never overwrite a maker's newer work). Font rollouts: `ttscardmaker-companion` §1.10-1.11.
- **Photoshop 2023**: only on copies under `~/std-work-1107/`, under `mkdir ~/Documents/Tools/wt/scratch/s1107/ps.lock`,
  `open -g`, the alert watcher the whole time, dialogs off, every document closed unsaved; quit the Photoshop you
  started. It may run while the owner games. The JSX and its log live under `~/std-work-1107/`. Never answer a
  permission prompt: stop what raised it and report. Its fonts in `~/Library/Fonts`: keep stoogedings.ttf identical to
  main's. PNG exports: JSX `saveAs(PNGSaveOptions)` writes the document's raw values untagged; a card whose PSD has the
  'Display' profile has a repo PNG converted to sRGB, so convert before comparing
  (`scratch/s1122/tasks/jsx-png/card_png.jsxinc` `cardPng(d, path, convert)`, or `py/psconv.py` there); converted
  exports are pixel truth (11-22: nothing changed on 11-16).
- **The bar for the bot's drawings** (guards re-exports, bakes, previews, fork renders): looks the same by eye at
  normal size (owner 11-13); level-by-level drift is fine; no card sweeps only for parity.
- **P-2** (one main lane per guarded push) amended twice: the kept push lane (11-19) and the early save (11-21), each
  one more save commit and #github-updates post.
- **No guides or checklists** for the owner; problem cards are the website's hand and bars flags (owner 11-07).
- **TTS**: the save's script is `TTS/Global.lua`; never edit the save by hand except as a deliberate edit (then move
  drift.py's BASELINE); no Lua patterns (MoonSharp). "Rebuild the save now" (per request only): cancel Card sync's
  waiting run, then `gh workflow run card-sync.yml --ref main -f bake=pending`, after any rulebook site deploy ends.
- **Headless browser**: only `~/std-work-1107/chrome/chrome-headless-shell`, cwd `~/std-work-1107`, its own
  `--user-data-dir ~/std-work-1107/chrome-<you>`, `--use-mock-keychain --password-store=basic
  --disable-background-networking --disable-component-update --no-pings`, pages over http://localhost only; kill it
  and its children by PID in a finally block.
- **Jetson**: restart only with `ssh jetson sudo sync`, then `~/.local/bin/jetson-powercycle cycle` (never `sudo
  reboot`). OJWikibot is the owner's OJWikibot session's ground; `deploy.sh` stays ours.
- **CardBrowser**: never a visible window on this Mac (`--self-test` or the hidden test mode), built apps only from
  `~/std-work-1107/cardapp-bin/`; no secrets, personal paths, Discord text or card files in the public repo; read-only;
  `tests/test_site_app_hook.py` stays green; its docs as short as possible (owner 11-22).
- **Discord archive and timelines**: never copy message text into a repo; `firstrepo.git` is kept (never deleted).
- **Dead code** is deleted (owner 10-23, 11-12), except the owner's KEEP picks (`owner-decisions-2026-11-12`).

## State (2026-11-23, final)

- AutoCroissant main 541feb9c + the wrap-up docs (skills, this plan, the report), deployed on the jetson at 5553f8da
  (42 commands, 777 cards; nothing the bot loads changed after it). Fork pin `cards-20261123` = baa9a901 (11-22's
  `cards-20261122` = e5915747: dead code out, linked pictures with smart filters and warps redrawn; 11-23: Reduce
  Noise); fork main bb8c62fc (docs and the 2-job CI since the tag); shared venv @baa9a901.
- TTSCardMaker main 93e5892f; data 9fc74401 (11-23 06:12: psd_details VERSION 15; card_checks 769 records, 0 hand,
  33 bars, 0 name, 0 unredrawable); branches `history` b8e865a8, `plates` 01e56f95, `previews` ce568f04.
- CardBrowser main 4c1bd38 = pre-release v0.1.2.
- Branches: every ref adjudicated in `docs/Finished/session-2026-11-22/branches.md` (SHAs in
  `scratch/s1122/wrap/branch_record.tsv`). GitHub holds only main, data, history, plates, previews (TC), main and
  `archive/*` (AC), and the fork's main, `archive/*` and `cards-*` tags (newest `cards-20261123`).

## Owner

- [ ] Check `MDW/Templates/Raid_Boss_Template.psd` (from 11-06; 20-cell split bars, 0-40). Raid bosses (6+ stars) have
      no stat limit.
- [ ] Beholster's copied ability: tell Croissant (it prints The Puppet's Ability word for word; owner 11-22: left
      for Croissant).
- [ ] Choices made unattended on 11-22/23, for a later look: the report's section "Conservative choices".

## Jetson

- [ ] **davey 0.2.0 held**: its DAVE API needs prepare/execute_transition, which discord.py 2.7.1 never calls (voice
      would break). Upgrade with a discord.py release that supports it.
- Notes: bot-only venv, no fork or Cython (11-11); Python 3.14.10 on both bots; Jellyfin keeps running (D25);
  ojwikibot enabled at boot; both units RestartSec=900; the units' ssh-agent drop-ins stay; smartd watches the NVMe;
  OJWikibot's `ojw-game-sync.{service,timer}` are OJWikibot's (never touch them). The jetson's autocroissant.service on
  disk changed 10-20 and was never daemon-reloaded (owner's call).

## Open owner decisions

- [ ] **Fork: a missing `composite`** (left 11-23: answer 22 covered docs and docstrings only): `PSDImage.save()`
      still keeps the stored preview with a warning when `composite` can't be imported. Make it an error, now that
      `composite` is required?

## Owner checks (nothing is blocked on them)

- [ ] The ring badges on real cards: the website, the TTS save and the rulebook PDF; deep starts; the one-colour
      version under overlays.
- [ ] TTS, first real-game checks (tested only in the Lua emulator): right-click Card details (in hand too; drag,
      A-/A+, Rules buttons incl. 11-22's Minions, Death, Attacking and The Field, "+ Name" spawns, deck buttons
      (The Beholster's too), Find a card (by an ability word too), the rulebook's Contents and search
      (section titles, status durations), hand zones following seats, turn markers, The Gates of Ahn'Qiraj's
      "+ Silithid", hover help on status effects, timing tags and types, any Lua error.
- [ ] Website, first real use: History "Revert to this version"; the rulebook (its look, TTS pages cut from it, the
      editor's cut marks); preview trash buttons; Mentions and Unused; the Morgue and the compare modes on a phone; a
      turned phone keeps your place (11-22). Safari never ran.
- [ ] Rulebook look-over: see-through pictures wrap text to their edges, pictures hang out of their panels, `out=` on
      Getting Started's Kirby; Make space between list items and lines.
- [ ] The website editor's drags (anchor grid, Make space, blank space, snapping, undo): report what feels off.
- [ ] Photoshop: Computer Virus' stat bars (beside the 1200 px canvas, x 1245-1982: no stat edit can show).
- Eventually, never blockers: Photoshop standardizing (one Name and one Ability layer per card, rulebook Header/Page
  Number layers; Mr Jawsum's name layer); the team's rulings draft; the team's split of Tainted Lazarus.

## Developer

### Follow-ups from 11-22

- [ ] The two load-flaky guards tests (Standing rules, Tests): `test_the_checks_reports_come_once_that_job_ends` waits
      for a non-empty checks.result; `test_the_bot_clone_and_records_run_here`'s 15 s clone wait.
- [ ] The guards read `requirements-cards.txt` through the API at ref main, apart from the bot clone: a race at a pin
      change (there before 11-22).
- [ ] Fork-r9-4 kept setup.py's `use_limited_api` on 3.13+ (deliberate, the build identical).
- [ ] r8-2's mock-up on Martlet Prime: the larger line runs close to or just over its space (the skeptic: no blocker).
- [ ] Text colours at a boundary, the bake vs the mock-up (on main before actbake s3): Chris & Jordi 'DEF| lost' (bake
      #1b32bd, mock-up #4381f8), Pix 'Marked For Life| until' (wraps differently too), Grimmchild '+2 ATK| to parent'
      (bake pink, mock-up red).
- [ ] x3's limit: a new card in a folder whose cards share one series takes it even from a related game (1 of 302:
      Godskin Noble would get Dark Souls III); the editor's reset label shows the folder name lowercased.
- [ ] Rulebook search: 'alot' suggests 'slot'; a card's name there draws an unrelated "Did you mean" (Frog -> 'from').
      The renamed headings' old links (`#/rules/abilities/when-abilites-can-be-used`, `basic-abilites`) open the
      rulebook at its top (nothing linked them).

### Still open from 11-21

Closed 11-23 by cards s3: the phone title box (x0) and x25's Navigation API need (x1).

- [ ] x30's trade-off: a press right after a dialog closes, with no pointer move first, can be lost (~100 ms on a
      desktop); phones not measured.
- [ ] Fable fork r1-0's CPU parity nudge is untested on AVX-512 CPUs.
- [ ] x23's known limits (not regressions): its mock-up is worse on Mystery Gift, Warcraftian Mage and Papa Chip;
      Miracle Matter's stacked box 'Ice:'/'Fire:' can be cut at the card's left edge (as on main).
- [ ] Only if they show up: one API call per Quiet poll (if the token runs short); `MALLOC_ARENA_MAX` on the jetson (if
      memory pressure appears); 403s on the guards' reads.
- [ ] The type emojis' 3-line swap (bad3-2), only if type_emojis.py is touched.
- Revisit with their event: published deck buttons with the first public deck (wss-1); SEC-2's `pull_request` filter
  (`startswith`) if PR runs ever appear.

### Older items

- [ ] Fork, low priority: a tracking change breaks kerning (Quick Slash, Ominous Red Circle), tracked badge ligatures,
      Photoshop's underline box (Golden Metronome).

### Kept by ruling (no action)

The 14 old rulebook PSDs keep their old badges. Future Niiue's ringed line ends 2 px from the edge. OJWikibot (owner
10-26): `/update`'s diverged-checkout handling and the field-events snapshot. Cards: 42 glow and gradient FX layers; 6
outline FX layers that change colour; the clock emoji; Miracle Matter's form pictures and the 5 card-art inline
pictures; PSDs over 50 MB left alone (11-09). Stars option B: the 50% floor and two-wide columns (11-10). Card
Authors.txt keeps the old rule-page credits (11-12). Hero Cap keeps 3 stars (D6); Ze'mer's preview flower bits (D16);
the guards keep rewriting makers' link paths (D17). Ancient Fuelweaver, Wickerbottom, Crock Pot and Holo Defense API
keep their stray-font characters (11-17). Hidden-bar leftovers on rows of their own colour stay unflagged; Baba's stats
bake "even if it looks wrong" (11-13). Comp-bar cards at 0 keep the exact mock-up (11-21). Searches: exact words only
(D20); apostrophe forms kept; a series name in /query answers its series (11-20); /query keeps finding `MDW/` and
`Types/` PNGs by exact name (11-22). Several creatures: searches and the Stats bars match any creature; previews and
the table show creature 1 (11-18). The forum's role keeps Manage Threads server-wide (11-16). 11-22: x31's side effect
on NiseSegaSonic (its own huge '"' glyph, Ability ~17% smaller) accepted; PSD_DIRTY_VERIFY stays on in every test
(G3); the rulebook picture `status-panda.webp` keeps the old 'Smouldering' card; Beholster's printed Ability (The
Puppet's, word for word) is left for Croissant. The fork CI runs 2 jobs, Linux + macOS on 3.14 (11-23: Windows, other
Pythons and the no-composite jobs dropped).

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
again); old names from Discord as aliases (11-13); the 152-card Strong-type text redraw (11-13); a startup check, admin
alert or next-post check for the forum votes (11-16); linking 'die', 'defeated' and 'kill' in search (D20); the
comp-bar zero-row picture speed-up (11-21); a local bake in CardBrowser and a "use my local folder" site feature
(11-20); changing /query's exact finds of `MDW/` and `Types/` PNGs (11-22); a font build only for D24 (11-22). Research
ideas accepted and rejected: the titles in `docs/Finished/session-2026-11-22/` and `session-2026-11-10-21/`
(accepted.md, rejected.md with its dropped rows; don't re-propose). Older:
the `dropped` lists of `opt-round3/plan.json`, `opt-round4/plan.json`, `opt-1104/round2-plan.json`,
`~/Documents/Tools/wt/scratch/opt/PLAN.md` §(b); parked ideas: `docs/Finished/session-2026-10-24/README.md`.
