# Next session plan (after 2026-11-10)

## Where to start

Start in `~/Desktop/AutoCroissant`. Read memories `resource-limits`, `secrets-never-printed`, `maintenance-pushes-silent`,
`photoshop-no-interrupt`, `never-pkill-shell-snapshot`, `owner-decisions-2026-11-09` (and `-11-07`),
`ttscardmaker-integration`, `jetson-server`, `persistent-work-paths`, `agent-model-preference`. What landed:
`docs/Finished/session-2026-11-07-10/README.md` (linked pictures, automatic flags, clean cards, card history, status
cards, the proposal forum); earlier `session-2026-11-06/`.

## Order (owner, 11-10 ~05:20)

1. **Research rounds again, at once**: psd-tools (fork, a new worktree off fork main = `cards-20261110`; the old
   research worktree is gone), Actions + bake, card data + website. Rounds run until dry (two rounds in a row with nothing new),
   no caps; one ship every 4-5 rounds under the locks.
2. **Every unfinished task** in "Waiting from this session" below (first: ship AC/TC `work/actions-1108` with its two
   fixes, and check the first card workflow run on the new fork pin), then the Developer list's open bugs.
3. **Card timelines from Discord** (`docs/Future/CARD_TIMELINE.md`): only when the owner asks.

## Waiting from this session

- [ ] **FIRST: restore the AC tests' old card blobs.** The test cache (`~/.cache/autocroissant-tests`) was deleted at the
      wrap-up and the fresh clone lacks some pinned blobs it hid, so the AC suite fails (134 failed, 83 errors on 11-10)
      until they are back: run `bash ~/Documents/Tools/wt/scratch/s1109/wrap/fetchblobs.sh` (runs the suite, batch-fetches
      from GitHub every pinned blob the tests report missing, repeats; one batched fetch per pass, never lazy). Done when
      the suite is 925 passed, 1 skipped (esbuild).

- [x] First card workflow run on the new fork pin: Card sync 42830695509 built and cached the fork's wheels and venv on the
      runner, green. A Card checks run that missed the cache at the same moment failed: pip clones the fork blobless and
      the job's `GIT_NO_LAZY_FETCH=1` broke its checkout; fixed in TC f80772da (`env -u GIT_NO_LAZY_FETCH pip wheel` in
      cards-python), proven only at the next cache miss (the next re-pin): check it then.

- [ ] **AC + TC `work/actions-1108`** (rounds 21-25, not on main): its ship was refused three times on 11-10. Fix
      before landing: (a) TC 704baadb "Site build: each deploy under a commit of its own" was never tried against real
      Pages: TRY 2 should fall back to GITHUB_SHA when Pages refuses a version commit (or hold 704baadb and AC a89266d1
      out); (b) TC 4236cbb3's data re-fetch after the wait gets one try and fails the build: retry it 3 times like
      actions/checkout did. The rest (AC d5534c00 range-read workers, 17780f89 + TC aad1c358 guards' Check beside the
      fonts restore, a778792a folders read once, f43ffe48 + 3006c795 picture changes fetched ahead, TC 2b8e1479 each
      lane checks only its CLIs) passed its checks. Both sides are bundles now: AC `~/Documents/Tools/wt/scratch/s1109/wrap/ac-actions-1108.bundle`, TC
      `tc-actions-1108.bundle` beside it (`git fetch <bundle> work/actions-1108:work/actions-1108`).
- [ ] The Card browser deploy timed out twice on 11-09 (deploy-pages stuck in "syncing_files", runs 42622998192,
      42753587574); the next push redeployed. If it repeats, that is (a) above's motivation.
- [ ] Guards gap (11-10 skeptic): a push that puts back only an old PNG (e.g. the black-stroke Despair PNG) passes R10
      when the PSD has no stored image (parity floor only).
- [ ] The 11-08 Single-line batches and some 11-08 card fixes (TC 66efec76, 2ae8384c, 12bc1331, ac57d748, 380b856e)
      have no `Card-sweep` trailer: check whether Card sync posted them as New art (Discord changelog), nothing to undo.
- [x] **Card fixes and psd round 15's strikes sweep** (11-10, TC 11e932ec + b3273ec9, one rewrite per card, Photoshop
      opened all 126): typos (Duel Flag, Z!!!!!, Flip, Ancient Fuelweaver, Crock Pot, Wigfrid's watermark), Clot Gold's
      20 spare HP layers and unused link record (`Card-sweep: clean`), Bulborb Larva's hand flag on, Zote's and The Last
      Cabbage Merchant's off (owner), stale strikes on 119 PSDs (`Card-sweep: strikes`). Left: Water Totem's U+0010 (Photoshop
      spaces it differently without), Man-Serpent's Shield name vs title (sheet first).
- [x] Card data ship's dispatches (owner 11-10): Card sync re-read the abilities (data 45e8268b, "4 reparse"); the 17-card
      Card checks re-read was re-dispatched as run 42833874576 after the first one hit the wheel bug: check it is green.

## Standing rules

- Work unattended unless the owner says "ask me questions" (asked again on 11-04). Agents on Opus or lower; a single
  Agent for a build. Owner picks are never dropped without asking: a review may only flag them.
- Commit to main in both repos; `work/*` only while in flight, then delete. Very short commit messages; no Claude
  attribution. Persistent paths only (`~/Documents/Tools/wt/`, `~/std-work-*`).
- Deploy with the `jetson-deploy` skill: push main and deploy in one sitting.
- Tests: `AUTOCROISSANT_LOCAL_DIR=~/Desktop/TTSCardMaker AC_SITE_SRC=<tc tree>
  ESBUILD=~/Documents/Tools/wt/scratch/esbuild/node_modules/.bin/esbuild ~/.venvs/ac-py314-fork/bin/python -m pytest
  -n auto -q -p no:cacheprovider tests` (925 on 11-10, about 1.5 minutes at `-n 6`; they read old card blobs from the clone, skipping when missing: the fresh clone has them). Never
  with `GIT_NO_LAZY_FETCH=1` exported (10 tests' own fixture repos then fail; conftest sets it for the clone). Targeted
  files unless a full run is needed.
- Resource limits (memory `resource-limits`): never a partial (`blob:none`) clone of the local card repo; xdist at
  most 6; one full suite at a time (`~/Documents/Tools/wt/scratch/suite-wait.sh`); at most 2 workflows; process pools
  at most 4; the git watchdog (`~/Documents/Tools/wt/scratch/git-watchdog.sh`) under Monitor during long unattended
  work; no repack or gc of the big clones while agents run.
- The Desktop TTSCardMaker clone is blobless from GitHub (fresh on 11-10, full history, no longer shallow): every git call with `GIT_NO_LAZY_FETCH=1`, never a fetch of
  many blobs; `git write-tree` needs `--missing-ok` there. Refresh TC's local `origin/data` with
  `git fetch origin +data:refs/remotes/origin/data` (or check `git ls-remote origin data`) before reading it.
- zsh arrays start at 1 and zsh doesn't split unquoted variables: loops over arrays under `bash -c`. `gh run list -c`
  needs the FULL sha.
- Secrets (memory `secrets-never-printed`): never print config.py values, tokens or webhook URLs; a token goes only to
  its own service; hooks: name, active flag and host only.
- PSD pushes: run the guards dry first, with `--repo` (without it the earlier icon fonts are not read):
  `python -m commands.card_guards check --clone ~/Desktop/TTSCardMaker --repo MichaelJSr/TTSCardMaker --before
  origin/main --after <head> --mode dry --out <dir>`; discord.json must be `{}`. Sweep fixes carry `Card-sweep: <kind>`
  (silent per commit). A font rollout's sweep and batch pushes: `ttscardmaker-companion` §1.10.
- PSDs our code rewrote: open every one in Photoshop by script before pushing (a leftover embedded smart-object
  document makes Photoshop refuse a file; `~/Documents/Tools/wt/scratch/alt-template/push/open_all.sh`, it reads only
  from /private/tmp). Card rollouts: `ttscardmaker-companion` §1.10-1.11.
- Photoshop 2023 may be used on copies under `~/std-work-*`: `open -a "Adobe Photoshop 2023" script.jsx` plus a result
  file (osascript is blocked), dialogs off, only documents we opened. It draws with the fonts installed here: MDW's
  stoogedings (the ring build), Futura-Bold, Lucida Grande Bold, Rockwell Bold, Elephant Italic, Myriad Arabic and
  Roboto Black are in `~/Library/Fonts`; keep stoogedings.ttf identical to main's.
- No guides or checklists for the owner. Problem cards are the website's hand and bars flags (owner 11-07; the CSV is gone).
- The TTS save's global script is `TTS/Global.lua` (+ `Global.xml`); never edit the save by hand except as a deliberate
  edit (then move drift.py's BASELINE). No Lua patterns (MoonSharp). Test a Lua change with
  `~/.venvs/tts-checks/bin/python TTS/checks/lua_mock.py "TTS/Impossibility Simulator.json" TTS/Global.lua`; the whole
  suite: `TTS_HISTORY=~/Documents/Tools/wt/scratch/tts/history.git ~/.venvs/tts-checks/bin/python
  TTS/checks/run_all.py` (24 checks, 4 at a time, about 1 minute).
- "Rebuild the save now" (owner, per request only): cancel Card sync's waiting run, then `gh workflow run card-sync.yml
  --ref main -f bake=pending`; first check that the site deploy holding any rulebook change has finished.
- Every scripted TC commit that changes card PSDs or PNGs carries `Card-sweep: <kind>` (memory `maintenance-pushes-silent`).
- Push locks: `mkdir ~/Documents/Tools/wt/scratch/s1107/ac-push.lock` / `tc-push.lock` / `ps.lock` (rmdir after); a TC
  push holds its lock until every run it started has finished (`~/Documents/Tools/wt/scratch/s1108/watch_runs.sh`).
- Photoshop: `open -g`, the alert watcher (`scratch/s1107/cstar/`), nothing left open (memory `photoshop-no-interrupt`).
- Processes: kill exact PIDs only (never `pkill -f` a shell path); stop what you start; never print other apps' command
  lines or environments. While the owner plays: lower load (memory `resource-limits`, 11-09 note).
- Workflows: never resume a multi-lane workflow with edited caps (memory `workflow-resume-multilane`). Multi-step shell:
  a `set -euo pipefail` bash script, not a zsh one-liner (zsh: no word splitting; `$C:r` eats `:refs`).
- AC tests that read TC's workflows or site from a stale checkout: point `AC_SITE_SRC` at a shallow
  sparse clone of TC main (`.github`, `site`, `TTS/Global.lua`, `Global.xml`, `decks.json`).

## State (2026-11-10 ~05:00, all pushed)

- AutoCroissant main cff351a4 (+ the wrap-up docs), deployed on the jetson (47 commands; fork @170a3fce). Fork pin
  `cards-20261110` = 170a3fce (fork main; the fork clone `~/Documents/Tools/psd-tools` has only main and its tags; harness `~/Documents/Tools/wt/scratch/s1107/fork/`, its ROUNDS.md has the release entry).
- TTSCardMaker main 581b61c1; data 28e57faf at the wrap-up (moves with Card sync). The Desktop clone was replaced on
  11-10 (old clone's unlanded branches: `scratch/s1109/wrap/tc-actions-1108.bundle`, `tc-unlanded-1110.bundle`).
- No worktrees or work/* branches in AC, TC or the fork (unshipped work is in the bundles above); GitHub has only main, data, previews (TC) and the archive/* branches.
- The proposal forum module runs on the jetson (forum 1569539619471499324): the first real day-6 ping is due about
  2026-11-16 02:33 UTC (post 1569540573461745755); check the journal for "Forum votes:" lines then.

## Owner (from 11-06, still open)

- [ ] The TTS Saves folder copy (`~/Library/Tabletop Simulator/Saves/Impossibility Simulator.json`) is from 11-06; copy
      main's save when you want (owner 11-10: leave it for now).

- [ ] Check `MDW/Templates/Raid_Boss_Template.psd` (20-cell split bars, 0-40; built unattended). Raid bosses
      (6+ stars, `card_models.RAID_STARS`) have no stat limit; their bars bake up to 40.
- [ ] Manual pass on the cards in the website's Needs handling (hand-drawn bars, several creatures; each with its note).
- [ ] Tell card makers about the new templates (the Alt ones took the old paths on 11-07; the bot reads both kinds of bars).
- [ ] Post the card makers' ring font note and re-upload the TTS Workshop item (from 11-05, still open).

## Jetson (maintenance 11-05; rollback files in `~/maint-20261105`)

- [ ] **Ubuntu Pro / ESM:** 20.04's standard support is over; ~357 security updates wait (openssl, libc, systemd...).
      Needs the owner's own token: `sudo pro attach` in a planned window (dpkg snapshot, install, power-cycle).
- [x] **Python 3.14.10** (11-10): PGO+LTO, both bots on it (`~/venvs/<bot>` -> `<bot>-3.14.10`; 3.14.8 kept as
      `<bot>-3.14.8`). Back: `ssh -q jetson 'bash ~/build/python/rollback-3.14.8.sh start'`.
- [x] **Bot packages** (11-05, tested on the Mac first: both suites and 376 badge cards' renders identical): Cython
      3.3.1 (full rebuild in both venvs), pandas 3.1.1, fonttools 4.68.0, uharfbuzz 0.57.0, python-dateutil 4.0.0 (+
      tzdata). Undo: `~/maint-20261105/held-upgrades.txt`.
- [ ] **davey 0.2.0 held:** its DAVE API needs prepare/execute_transition, which discord.py 2.7.1 never calls (voice
      would break). Upgrade with a discord.py release that supports it. The Mac venv `~/.venvs/ac-py314-fork` has the
      jetson's package versions since 11-10.
- [ ] **Network:** a DHCP reservation for 192.168.4.30 (AdGuard listens only there; `ssh jetson` uses it); no host
      firewall: the AdGuard UI (:3000), SMB and Jellyfin answer on the global IPv6 addresses: confirm the router blocks
      inbound IPv6, or bind them to the LAN. Jellyfin's library is empty (~435 MB RAM): keep it or stop it?
- [ ] After a week of normal running: `sudo docker rmi adguard/adguardhome:rollback-v0.107.79
      jellyfin/jellyfin:rollback-12.1` (~970 MB) and remove `~/maint-20261105` (its tarballs hold password hashes and
      keys: never print them).
- [ ] **NVMe unsafe shutdowns:** 107 of ~140 power cycles, and the ordinary 03:20 reboot counted as one too: L4T's
      reboot/poweroff may cut the drive before it flushes, not only the Shelly. No media errors, 0% wear.
- [ ] OJWikibot logs a harmless shutdown Traceback (`login_manager.py:443` shutdown_handler: the wiki worker gets
      SIGTERM twice and raises SystemExit inside urllib3's atexit): ignore a repeated signal.
- Notes: ojwikibot is now enabled at boot (`sudo systemctl disable ojwikibot` if that was deliberate); both bot units
  use RestartSec=900 (a failed boot start waits 15 minutes). The units' ssh-agent warning is fixed by the drop-ins
  `/etc/systemd/system/<bot>.service.d/ssh-agent.conf` (the bot runs as the agent's command; main PID still python).
  smartd watches the NVMe (health, error log, 70/80 C) into the journal only; no NVMe self-tests (smartmontools 7.1).

## Open owner decisions

- [ ] **A developer's old TTSCardMaker clone from before the history wipe** (owner 11-10): GitHub's history starts at
      2023-11-05 ("init" commits). If they send the folder (zipped as is, `.git` included; no pull/fetch/gc first, GitHub
      Desktop closed), recover what it holds in a scratch copy (`git log --all`, `git reflog`, `git fsck --lost-found`).
      Publish only with the owner's OK: an `archive/history-before-2023-11` branch, or a separate archive repo if old PSD
      versions make it too big; it would also feed the card timelines.
- [ ] **Futura on Macs.** MDW's Futura-Bold differs from macOS's own Futura in every glyph (macOS's 2.8% wider);
      Photoshop draws with macOS's unless MDW's is installed, and the guards then refuse its re-lays (Koro-Sensei). No
      swap helps (Jost 89%, League Spartan 90%, macOS's 69% off; MDW's fits). Keep MDW's file (recommended); should
      card makers on Macs install MDW's Futura-Bold, as this Mac now does? Rockwell Bold (8.3% wider) and Lucida Grande
      Bold differ from macOS's too. Files: `~/Documents/Tools/wt/scratch/futura-research/`.
- [x] **Mantis Lords** (R14): case fixed on 11-05 (TC 6383bbca); R14 finds nothing.
- [ ] **Old-font note for refused re-lays** (round 4 cm-a1-offpixels): tell OLD_FONT when an earlier font explains a
      refused re-lay. The tightened rule (every glyph old to the current font must be drawn by an earlier one) is in
      `opt-round4/skeptic-cm-a1-tools/proposed2.py`: 15 of 16 told, Koro-Sensei silent. Not voted on; a guards change.
- [x] **Traitor Lord:** the owner kept the 8.4 px gap (tracking 0) on 11-05.
- [ ] Discord pending-updates: the 8 card messages (State) are undone card work; pick which to do.
- [ ] Rules left by the 11-01 batch: seven cards count lanes as turns (Delicate Flower, The Last Cabbage Merchant, Time
      Riddle, Asteroid Field, Forget Me Now, NME Salesman, Lost Kin), and pair turns are undefined with uneven slots
      (White Palace). Wording written from discussion, no official text: prime creature, base stats. Full list:
      `impossibility-simulator-rules` Known gaps.
- [ ] TTS-sized JPEG card copies (the save loads 839 files, 1.5 GB of 1200 px PNGs; half-size JPEGs cut it about 10x):
      600 or 800 px wide, or not. Also the holder pictures (Creature_Deck.png 6.7 MB).
- [ ] Looks chosen unattended on 10-26 (sheets in `~/Documents/Tools/wt/scratch/layouts/*/sheets/`; each is one small
      rule to change): Leshy at 4 hides its bottom-left drop and at 1 keeps it alone; banner rows tighten to 0.85 of a
      star; die faces at 1-3; Crowned Magolor's funnel; closing a type row (the last icon fills the hole); the two-row
      grid on adds (Masher, Spamton Neo and Defcon Panda's new icon comes out smaller); Clot Black's and Minecraft
      Panda's card fixes.
- Offered by the research rounds, not yet put to the owner: a light-theme-only `--muted: #686661` (round 3 WQ5, a look
  change); "Should the card website show up in Google?" (round 4 wss-3: only the root and `c/` pages if yes).

## Owner checks (nothing is blocked on them)

- [ ] The ring badges on real cards: the website, the TTS save (1200 px PNGs) and the rulebook PDF; deep starts (more
      than a whole ring below zero: a thinner band, lap rings inside); the one-colour version under overlays.
- [ ] TTS, first real-game checks of the 10-30 save (tested only in a Lua emulator): right-click Card details (in hand
      too; three panels; drag, A-/A+, Rules buttons, "+ Name" into the hand face down at the right size); Find a card;
      the rulebook's Contents and search (does Book.setPage land on the right page, 0-based assumed?); hand zones
      following seats; turn markers (F or the menu count; physics tumbles don't flip them); the panels' build about 2 s
      after the data loads and after someone sits; The Gates of Ahn'Qiraj's "+ Silithid"; whether TTS's PDF viewer
      searches the text; any Lua error. Since 10-31: spawn buttons on up to 2 rows, long names shortened with "..."
      and whole as a `tooltip`; one `UI.setAttributes` call per element; the first spawn without a stall; a card naming
      itself has its own "+ Name". Since 11-03: the mirrored table, brown Bricks counters, SP counters, the table map.
- [ ] Website, first real use: History "Revert to this version" on a chapter, the ban list and a card; the rulebook (its
      look as before 11-03, TTS pages cut from it, the editor's cut marks); the preview trash button and Trash all; the
      editor's "On the site now · File" and "These boxes make" lines; Mentions and Unused; the round 3 and 4 changes on
      a phone (deck builder in 3 columns, the card picker, hand links). Safari never ran (also the many-boxes edge
      scroll).
- [ ] Rulebook, look over: see-through pictures wrap text to their edges, pictures hang out of their panels, `out=` on
      Getting Started's Kirby; Make space between list items and lines.
- [ ] Try the website editor's drags (anchor grid, Make space, blank space, snapping, undo) and report what feels off.
- [ ] Photoshop: Computer Virus' stat bars (added 11-06 by netverk, but beside the 1200 px canvas, x 1245-1982: no
      stat edit can show; several creatures, on the manual list).
- [ ] Big PSDs: GitHub warns past 50 MB (Quote.psd 69.4 MB on batch 2's push; The Panda Future 54,114,387 bytes).
- [ ] Open in Photoshop once: a smart-filter star move (lane 3, S1) and a title moved back onto the card (lane 2); both
      were checked only in psd-tools renders.
- Eventually, never blockers: Photoshop standardizing (one Name and one Ability layer per card, rulebook Header/Page
  Number layers; Mr Jawsum's name layer); the team's rulings draft; developers resetting clones after the 10-24 rewrite.

## Developer

### Follow-ups from the research rounds (not done)

- [ ] Phones at 389-390 px: a rulebook deep link (e.g. #/rules/timing-tags/reading-the-ring) lands with its title 22 px
      under the sticky bar, because the rulebook web fonts swap in after the scroll (Inconsolata/EB Garamond,
      font-display: swap). Re-scroll once `document.fonts.ready` settles, or preload those two fonts (audit 11-05).

From `~/Documents/Tools/wt/scratch/opt-round3/plan.json`, `opt-round4/plan.json` and `opt-1104/round2-plan.json`
(their `dropped` lists say why each waited).

- [ ] **Earlier fonts window** (round 3 ag3-earlier-fonts-window): the guards read the icon font's 3 earlier builds, so
      3 font commits after the ring the pre-ring build drops out and an old save is judged against none. A real fix
      keeps font eras: match `_drawn_earlier` by glyph name (ids shift when icons are added), normalise SVG ids, cap
      the walk. Due before the third font commit after 0ade8c3d.
- [ ] Font builds look the same to a card maker (name ID 5, uniqueID, fontRevision, head.modified): only the file size
      tells a stale stoogedings (round 2 automation gap; AG3, a build id in the version name, was dropped as a separate
      commit: put it in the next font build if wanted).
- [ ] The mock-up's erase memo (round 3 W1, ~25 ms on long texts, editors only): its collision with the ring
      mockup.js work is gone (the key needs bfam; a fresh measure must set `.label`).
- [ ] Rulebook PDF waits for its pictured cards (round 2 A12, part a only; optional: push order covers it today).
- [ ] The card-sync.yml main-lane push retry's misleading warning (round 3 pipe2-push-retry): reword it.
- [ ] Only if they show up: one API call per Quiet poll (pipe2-quiet-poll, if the token runs short); `MALLOC_ARENA_MAX`
      on the jetson (bot-malloc-arenas, if memory pressure appears); 403s on the guards' reads.
- [ ] Small website items, low priority: shared 64 px icons (web-first-load-1, ~55 KB, with other build work); the
      rulebook chapter HTML cache in a WeakMap (W2, ~5 ms a search render); lazy versions.json and history.json in the
      card dialog (WFL2-4 + wr3-2, ~34 + 57 KB on phones); `.cl-push{content-visibility:auto}` once the Edits tab
      grows (wr2-changes-pager); a 3-line focus fix for the card dialog (WQ3, with a later accessibility pass);
      `table-map.webp`'s rounded corners filled so the PDF's JPEG step saves ~2.2 MB a print (P4); the profile
      editor's author list counts duplicates (edit.js, web4-profile-autolist); the type emojis' 3-line swap if
      type_emojis.py is touched (bad3-2).
- Revisit with their event: published deck buttons with the first public deck (wss-1); SEC-2's `pull_request` filter
  (`startswith`) if PR runs ever appear.

### Older items

- [ ] Editor ideas offered, not chosen: replace a picture by dropping a file on it, Ctrl/Cmd+D duplicate, a diff before
      Publish, a zoom control for the to-scale preview. Deck sheets for TTS: not recommended.
- [ ] Card Authors.txt still credits the 48 old rule pages (kept; card_data no longer warns about them).
- [ ] `docs/BUGS_AND_IMPROVEMENTS.md`: live checks V1 (`/sync_emojis refresh:True`) and V4 (`/query Raw Kawk`,
      `/query Black Clot`) still open. D6 (icon-font adoption) is ongoing.
- [x] `psd_parser` ABILITY_EXCLUSIONS gone (11-09, 21 -> 0): no live card has a hidden Ability layer left, so the
      guards' R3 waiver had nothing to waive.
- [ ] Idea 14's orphaned-FX validator was never built; the bake already deletes such layers when it edits a card.
- [ ] Stars that did not fit their banner: option B (owner 11-10): every star shrinks evenly, down to 50%, until the
      added ones fit, told (Thunder Thighs 3 at 95%, Consolation Prize 3 at 65%, Bubble 2 at 60%, Forsaken City 3 at
      70%, The Core 3 at 65%; 66 star counts on 23 cards); the mock-up draws the same. Forgotten Crossroads and Spiky
      Ball Trap's backdrop fixed (AC 4d148771). Owner to decide: the 50% floor (78 counts, mostly 5+ stars, still off
      the banner, told; a 20% floor fits 72 of them at 20-45%) and the two-wide columns at 50-70% (Blood Bomb 8, steam
      sale 8, Holy Mantle 8, Bucket Of Lard 6 and 8, Charm Of The Vampire 6).
- [ ] Types: in an edit that removes and adds at once, the added icons keep today's free spot instead of going through
      the closing loop.
- [ ] Stat bars: Clot Gold's spare hidden HP layers still shown at low values (E2 in
      `~/Documents/Tools/wt/scratch/layouts/bars/DESIGN.md`). Card fix landed (TC 5b066112, 11-10).
      The bake rule (a series hidden in every cell is never shown) stays open.
- [x] Corruption and Dumplings, left out of the 10-26 font sweep: done by later passes (checked 11-09). Both Abilities
      set their icons in stoogedings, no stale Txt2 story, PNGs current.
- [x] Housekeeping (owner 11-10): `ac-py314-psd125` retired with the older-fork checks (the tests' skips, the
      code's fork probes); the pyoxipng fallback dropped (brew oxipng on the Mac); `alt_names` and Card Edits.json's
      `aliases_moved` dropped; `inspect_pickle.py` is `inspect_card_data.py`.
- [ ] Fork, low priority: a tracking change breaks kerning (Quick Slash, Ominous Red Circle), tracked badge ligatures,
      Photoshop's underline box (Golden Metronome: round 6's underline placement is in `cards-20261110`; re-check). The fork scales tracking with the horizontal scale since `cards-20261104`.
- [ ] OJWikibot: `ojwiki_helpers.py:263-268` optional mwparserfromhell fallback; the backup search misses `Backup:
      pending.json`; O-6; read All_Cards_Event once for a lone "===Permanent Rotation===" heading; check the 10-21
      mixer gap in backup/history/mixers.json against that day's wiki page.

### Waiting on events

- [ ] The first live `+ability` bake: check its placement and font.
- [ ] The first card maker push after the font: an old-font save gets the A1 note (`OLD_FONT`); a guards re-lay of a
      Photoshop-finished card stays silent.

### Kept by ruling (no action)

The 14 old rulebook PSDs in `MDW/Old Rules/Rulebook/` keep their old badges. Future Niiue's ringed line ends 2 px
from the edge (as made). OJWikibot (owner 10-26): `/update`'s diverged-checkout handling (both bots) and the
field-events snapshot. Cards: 42 glow and gradient FX layers; 6 outline FX layers that change colour; the clock emoji;
Miracle Matter's form pictures and the 5 card-art inline pictures. The jetson's autocroissant.service on disk changed
10-20 and was never daemon-reloaded (owner's call).

## Queue (`docs/Future/`)

| File | What is left |
|---|---|
| `CARD_CREATOR.md` | In-browser card creator: designed, not queued |
| `SIMULATOR.md` | Simulator against the AI or by the numbers: designed, not scheduled |

## Not chosen

Answered no by the owner (11-04): 24 first-screen tiles, an opaque top bar, phone chart labels. Research items dropped
with reasons: the `dropped` lists of `opt-round3/plan.json`, `opt-round4/plan.json` and `opt-1104/round2-plan.json`
(the ones worth revisiting are under Follow-ups). Older: `~/Documents/Tools/wt/scratch/opt/PLAN.md` §(b) (S-5, S-6,
B-4, B-5, P-3, P-4, P-6, K-4, K-6, O-7, O-8, O-9). Parked ideas: `docs/Finished/session-2026-10-24/README.md`.
