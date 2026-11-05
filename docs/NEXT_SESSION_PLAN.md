# Next session plan (after 2026-11-05)

## Where to start

Start in `~/Desktop/AutoCroissant`. Read memories `resource-limits`, `secrets-never-printed`, `ttscardmaker-integration`
(its 11-03 and 11-04 lines hold this session's rulings), `lean-tests-and-verification`, `no-guides-csv-only`,
`edits-two-kinds`, `jetson-server`, `persistent-work-paths`, `agent-model-preference`, `rulebook-on-website`,
`discord-pending-updates`. What landed: `docs/Finished/session-2026-11-03-05/README.md` (cost badges, the cooldown ring
font, the card rollout); earlier `session-2026-11-01/`, `-10-30c/`, `-10-30b/`, `-10-30/`, `-10-26/`.

First: check that the batch chain finished (In flight), then the go-day leftovers in order, then the owner's decisions.

## Standing rules

- Work unattended unless the owner says "ask me questions" (asked again on 11-04). Agents on Opus or lower; a single
  Agent for a build. Owner picks are never dropped without asking: a review may only flag them.
- Commit to main in both repos; `work/*` only while in flight, then delete. Very short commit messages; no Claude
  attribution. Persistent paths only (`~/Documents/Tools/wt/`, `~/std-work-*`).
- Deploy with the `jetson-deploy` skill: push main and deploy in one sitting.
- Tests: `AUTOCROISSANT_LOCAL_DIR=~/Desktop/TTSCardMaker AC_SITE_SRC=<tc tree>
  ESBUILD=~/Documents/Tools/wt/scratch/esbuild/node_modules/.bin/esbuild ~/.venvs/ac-py314-fork/bin/python -m pytest
  -n auto -q -p no:cacheprovider tests` (689 on 11-04, about 3.5 minutes). Targeted files unless a full run is needed.
- Resource limits (memory `resource-limits`): never a partial (`blob:none`) clone of the local card repo; xdist at
  most 6; one full suite at a time (`~/Documents/Tools/wt/scratch/suite-wait.sh`); at most 2 workflows; process pools
  at most 4; the git watchdog (`~/Documents/Tools/wt/scratch/git-watchdog.sh`) under Monitor during long unattended
  work; no repack or gc of the big clones while agents run.
- The Desktop TTSCardMaker clone is blobless from GitHub: every git call with `GIT_NO_LAZY_FETCH=1`, never a fetch of
  many blobs; `git write-tree` needs `--missing-ok` there. TC's local `origin/data` is stale (679b1309): use `git
  ls-remote origin data` or `git fetch origin +data:refs/remotes/origin/data`.
- zsh arrays start at 1 and zsh doesn't split unquoted variables: loops over arrays under `bash -c`. `gh run list -c`
  needs the FULL sha.
- Secrets (memory `secrets-never-printed`): never print config.py values, tokens or webhook URLs; a token goes only to
  its own service; hooks: name, active flag and host only.
- PSD pushes: run the guards dry first, with `--repo` (without it the earlier icon fonts are not read):
  `python -m commands.card_guards check --clone ~/Desktop/TTSCardMaker --repo MichaelJSr/TTSCardMaker --before
  origin/main --after <head> --mode dry --out <dir>`; discord.json must be `{}`. Sweep fixes carry `Card-sweep: <kind>`
  (silent per commit). A font rollout's sweep and batch pushes: `ttscardmaker-companion` §1.10.
- Photoshop 2023 may be used on copies under `~/std-work-*`: `open -a "Adobe Photoshop 2023" script.jsx` plus a result
  file (osascript is blocked), dialogs off, only documents we opened. It draws with the fonts installed here: MDW's
  stoogedings (the ring build), Futura-Bold, Lucida Grande Bold, Rockwell Bold, Elephant Italic, Myriad Arabic and
  Roboto Black are in `~/Library/Fonts`; keep stoogedings.ttf identical to main's.
- No guides or checklists for the owner: problem lists are a CSV plus its `.md` (memory `no-guides-csv-only`).
- The TTS save's global script is `TTS/Global.lua` (+ `Global.xml`); never edit the save by hand except as a deliberate
  edit (then move drift.py's BASELINE). No Lua patterns (MoonSharp). Test a Lua change with
  `~/.venvs/tts-checks/bin/python TTS/checks/lua_mock.py "TTS/Impossibility Simulator.json" TTS/Global.lua`; the whole
  suite: `TTS_HISTORY=~/Documents/Tools/wt/scratch/tts/history.git ~/.venvs/tts-checks/bin/python
  TTS/checks/run_all.py` (24 checks, about 4 minutes).
- "Rebuild the save now" (owner, per request only): cancel Card sync's waiting run, then `gh workflow run card-sync.yml
  --ref main -f bake=pending`; first check that the site deploy holding any rulebook change has finished.

## State (2026-11-05 ~02:45, all pushed)

- AutoCroissant main 855de4d, pushed and deployed on the jetson (11-05). psd-tools fork `cards-20261104` = 1258fcbd on
  both machines and on GitHub. No work branches or extra worktrees left in either repo.
- TTSCardMaker origin/main 8e040018: the ring font 0ade8c3d (blob 2c0f8b1a, 17,416,116 bytes, the same in
  `~/Library/Fonts`), the 20 site commits, the 9 card commits (tags + 7 batches + the Drawcia/NME Salesman wording), Card
  sync's commits after each, then site.yml without `/Types/`. Completeness checked: all 753 card files (376 PSDs, 377
  PNGs with Stubborn_Queen.png) on origin/main exactly as checked locally. The 14 `MDW/Old Rules/Rulebook` PSDs keep
  their old badges by design.
- Data: moves with Card sync (`git ls-remote origin data`). Drift baseline 0a7db1c3 (f26dcd9c).
- The TTS Saves folder copy is the pushed save (0d88ea9a, 11-05); the old one is in
  `~/Documents/Tools/wt/scratch/tts-saves-bak/`.
- Discord `pending-updates`: the 8 card messages of 11-01 (crystal stars twice, Tainted Lazarus split, The Puppet to The
  Beholster, Earth Totem, the type 2 death card list, the equip stat formatting poll and its result). Earth Totem's PSD
  changed 11-02 ("earth totem nerf", 03119c90, hdckighfkvhvgmk): its message may be done.
- OJWikibot: the jetson runs 8ed817f; nothing changed this session.
- `+ability` is live; no live `+ability` bake yet.

## Go-day: done (11-05)

- [x] Batches 1-7 pushed, each after the previous push's runs settled green (logs `ring-sweep-final/go/<batch>/`).
- [x] Completeness: 753 of 753 files as checked. [x] work/big-push landed (855de4d) and deployed. [x] site.yml without
      `/Types/` (8e040018): the next build logged "icons: 91", no warnings. [x] Clean up: branches, worktrees, scratch
      (11 GB left), Photoshop copies, the clone's 8 duplicate packs (~46 GB; memory `partial-clone-fetch-packs`).
- [ ] **Whole-tree guards dry run** (dispatched 11-05 10:16 UTC, run 42009352040): read its summary and artifact. No
      card should show `badge widths` or `old glyphs` in `redrawn`, `pixel_left` or its writes' `pixels`; what it tells
      is the next problem list.
6. [ ] **Owner:** post the card makers' font note (the owner's ruling; facts in `opt-round4/plan.json` go_day cm-r4-2: pull,
       install `MDW/Fonts/stoogedings.ttf` (17,416,116 bytes) over the old one, restart Photoshop, Update All Text
       Layers on locally saved badge cards; the pull is up to ~10 GB of card files). Re-upload the TTS Workshop item by
       hand (memory `tts-workshop-research`). Re-copy the save into the Saves folder.

## Open owner decisions

- [ ] **Futura on Macs.** MDW's Futura-Bold differs from macOS's own Futura in every glyph (macOS's 2.8% wider);
      Photoshop draws with macOS's unless MDW's is installed, and the guards then refuse its re-lays (Koro-Sensei). No
      swap helps (Jost 89%, League Spartan 90%, macOS's 69% off; MDW's fits). Keep MDW's file (recommended); should
      card makers on Macs install MDW's Futura-Bold, as this Mac now does? Rockwell Bold (8.3% wider) and Lucida Grande
      Bold differ from macOS's too. Files: `~/Documents/Tools/wt/scratch/futura-research/`.
- [ ] **Mantis Lords** (R14): its 'mantis lord' (a plain noun) is still told on a push of it; snas.'s 'smeckeldorfed'
      no longer is (5ae271f: a card written in lowercase on purpose; Kyle/hdckighfkvhvgmk is not one). Fix the PSD's
      case or leave it.
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
- [ ] Photoshop: Computer Virus' stat bars (its NO STATS FOUND stays until then); Ze'mer's full-width flower layers;
      Terry Hintz and Joel Miller's "Stars" groups and typed tags (prepared in `~/std-work-1026/stargroups/`; Photoshop
      2023 can't re-save their EarthboundZero text, nor Elmadan's); Papyrus' typed tag.
- [ ] Big PSDs: GitHub warns past 50 MB (Quote.psd 69.4 MB on batch 2's push; The Panda Future 54,114,387 bytes).
- [ ] Open in Photoshop once: a smart-filter star move (lane 3, S1) and a title moved back onto the card (lane 2); both
      were checked only in psd-tools renders.
- Eventually, never blockers: Photoshop standardizing (one Name and one Ability layer per card, rulebook Header/Page
  Number layers; Mr Jawsum's name layer); the team's rulings draft; developers resetting clones after the 10-24 rewrite.

## Developer

### Follow-ups from the research rounds (not done)

From `~/Documents/Tools/wt/scratch/opt-round3/plan.json`, `opt-round4/plan.json` and `opt-1104/round2-plan.json`
(their `dropped` lists say why each waited).

- [ ] **Earlier fonts window** (round 3 ag3-earlier-fonts-window): the guards read the icon font's 3 earlier builds, so
      3 font commits after the ring the pre-ring build drops out and an old save is judged against none. A real fix
      keeps font eras: match `_drawn_earlier` by glyph name (ids shift when icons are added), normalise SVG ids, cap
      the walk. Due before the third font commit after 0ade8c3d.
- [ ] Font builds look the same to a card maker (name ID 5, uniqueID, fontRevision, head.modified): only the file size
      tells a stale stoogedings (round 2 automation gap; AG3, a build id in the version name, was dropped as a separate
      commit: put it in the next font build if wanted).
- [ ] PNG-only duplicates (Stubborn Queen) get no guards job and aren't re-exported when the original's PSD changes
      (round 2 automation gap; the ring rollout moved it by hand).
- [ ] /update_stats notes hidden by a compare 422 (round 4 bot-r2-1): give the compare its own try in `notes()`.
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

- [ ] The website's global Escape closes an open dialog even while typing in it (app.js keydown): the rulebook editor
      keeps the draft, but it is surprising.
- [ ] Editor ideas offered, not chosen: replace a picture by dropping a file on it, Ctrl/Cmd+D duplicate, a diff before
      Publish, a zoom control for the to-scale preview. Deck sheets for TTS: not recommended.
- [ ] Card Authors.txt still credits the 48 old rule pages (kept; card_data no longer warns about them).
- [ ] `docs/BUGS_AND_IMPROVEMENTS.md`: live checks V1 (`/sync_emojis refresh:True`) and V4 (`/query Raw Kawk`,
      `/query Black Clot`) still open. D6 (icon-font adoption) is ongoing.
- [ ] `psd_parser` ABILITY_EXCLUSIONS still holds 21 names (only the guards' R3 waiver reads it); target 0.
- [ ] Idea 14's orphaned-FX validator was never built; the bake already deletes such layers when it edits a card.
- [ ] Regenerate the problem-cards list after the whole-tree dry run (`docs/Future/PROBLEM_CARDS_2026-10-25b.*` is
      stale). Deliver the CSV and its `.md`.
- [ ] Stars still refused: Thunder Thighs 3, Consolation Prize, Bubble, Forsaken City and The Core (option B, shrinking
      every star to fit the banner, is the larger fix); Forgotten Crossroads 3-6 and Spiky Ball Trap 5-6 wrongly act as
      the stars' backdrop; Yin Yarn 6, Paint Warrior 4-6 and Paint Bomb 7-10 miss the texture clipped onto their group.
- [ ] Types: in an edit that removes and adds at once, the added icons keep today's free spot instead of going through
      the closing loop.
- [ ] Stat bars: Clot Gold's spare hidden HP layers still shown at low values (E2 in
      `~/Documents/Tools/wt/scratch/layouts/bars/DESIGN.md`).
- [ ] Corruption and Dumplings: the 10-26 font sweep left them out (a PNG bar; a Txt2 reorder).
- [ ] Kept on purpose until their condition holds: the pyoxipng fallback (until `brew install oxipng` everywhere; the
      Mac still uses it), the older-fork checks (retire the `ac-py314-psd125` venv first), the `alt_names` field, Card
      Edits.json's `aliases_moved` key, the `inspect_pickle.py` name.
- [ ] Fork, low priority: a tracking change breaks kerning (Quick Slash, Ominous Red Circle), tracked badge ligatures,
      Photoshop's underline box (Golden Metronome). `cards-20261104` now scales tracking with the horizontal scale.
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
| `PROBLEM_CARDS_2026-10-25b.csv` / `.md` | The last problem-cards list (54 cards); stale, regenerate after the whole-tree dry run |
| `PROBLEM_CARDS_2026-10-25.csv` | The 10-25 morning census |

## Not chosen

Answered no by the owner (11-04): 24 first-screen tiles, an opaque top bar, phone chart labels. Research items dropped
with reasons: the `dropped` lists of `opt-round3/plan.json`, `opt-round4/plan.json` and `opt-1104/round2-plan.json`
(the ones worth revisiting are under Follow-ups). Older: `~/Documents/Tools/wt/scratch/opt/PLAN.md` §(b) (S-5, S-6,
B-4, B-5, P-3, P-4, P-6, K-4, K-6, O-7, O-8, O-9). Parked ideas: `docs/Finished/session-2026-10-24/README.md`.
