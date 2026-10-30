# Next session plan (after 2026-10-30)

## Where to start

Start in `~/Desktop/AutoCroissant`. Read memories `resource-limits`, `secrets-never-printed`,
`owner-decisions-2026-10-24-night` (its 10-25 and 10-26 lines), `lean-tests-and-verification`, `no-guides-csv-only`,
`edits-two-kinds`, `jetson-server`, `persistent-work-paths`, `agent-model-preference`, `rulebook-on-website`. What
landed: `docs/Finished/session-2026-10-30/README.md` (the rulebook on the website, its editor, the TTS save's panels;
10-26's in `session-2026-10-26/`). `docs/Future/` holds only the problem-cards lists; nothing is in flight.

First: the owner's checks of the new TTS panels in a real game (below), then the problem-cards list, then the campaign
tasks in the order the owner picks.

## Standing rules

- Work unattended unless the owner says "ask me questions". Agents on Opus or lower; a single Agent for a build (two
  workflow agents answered the owner's latest question instead of building: pin "your job is the code build").
- Commit to main in both repos; `work/*` only while in flight, then delete. Very short commit messages; no Claude
  attribution. Persistent paths only (`~/Documents/Tools/wt/`, `~/std-work-*`).
- Deploy with the `jetson-deploy` skill: push main and deploy in one sitting.
- Tests: `AUTOCROISSANT_LOCAL_DIR=~/Desktop/TTSCardMaker AC_SITE_SRC=<tc tree>
  ESBUILD=~/Documents/Tools/wt/scratch/esbuild/node_modules/.bin/esbuild ~/.venvs/ac-py314-fork/bin/python -m pytest
  -n auto -q -p no:cacheprovider tests` (660 on 10-30, about 3 minutes). Targeted files unless a full run is needed.
- Resource limits (memory `resource-limits`): never a partial (`blob:none`) clone of the local card repo; xdist at most
  6; one full suite at a time (`~/Documents/Tools/wt/scratch/suite-wait.sh`); at most 2 workflows; process pools at
  most 4; the git watchdog (`~/Documents/Tools/wt/scratch/git-watchdog.sh`) under Monitor during long unattended work;
  no repack or gc of the big clones while agents run.
- `gh run list -c` needs the FULL sha: `git rev-parse` first. zsh doesn't split unquoted variables.
- Secrets (memory `secrets-never-printed`): never print config.py values, tokens or webhook URLs; a token goes only to
  its own service; hooks: name, active flag and host only.
- PSD pushes: run the guards dry first (`python -m commands.card_guards check --clone <wt> --after <head> --before
  origin/main --repo MichaelJSr/TTSCardMaker --mode on --workers 2 --out <dir>`), discord.json must be `{}`; sweep
  fixes carry `Card-sweep: <kind>` (silent per commit).
- Photoshop 2023 may be used on copies under `~/std-work-*`: `open -a "Adobe Photoshop 2023" script.jsx` plus a result
  file (osascript is blocked), dialogs off, only documents we opened. Runners: `~/std-work-1026/*/tools/`.
- No guides or checklists for the owner: problem lists are a CSV plus its `.md` (memory `no-guides-csv-only`).
- The TTS save's global script is `TTS/Global.lua` (+ `Global.xml`); never edit the save's script by hand. No Lua
  patterns (MoonSharp); test a change with `~/.venvs/tts-checks/bin/python TTS/checks/lua_mock.py "TTS/Impossibility
  Simulator.json" TTS/Global.lua` (the whole suite: `TTS_HISTORY=~/Documents/Tools/wt/scratch/tts/history.git
  ~/.venvs/tts-checks/bin/python TTS/checks/run_all.py`, 24 checks, about 4 minutes). A deliberate hand edit of the save moves
  drift.py's BASELINE to that commit.
- "Rebuild the save now" (owner, per request only): cancel Card sync's waiting run, then `gh workflow run card-sync.yml
  --ref main -f bake=pending`; check first that the website deploy holding any rulebook change has finished (the PDF is
  printed from the live site).

## State (2026-10-30 ~04:55 PDT)

- AutoCroissant main d86fe30 (plus this wrap-up's docs); the jetson runs 17601aa: the later bot commits touch only Card
  sync's modules (card_data, rulebook_pages, tts_save), tests and skills, so no deploy is owed (deploy any time).
- TTSCardMaker main 3b32375e (the TTS save rebuilt 10-30 04:44 PDT with everything below). Desktop clones level with main.
- Data f417aaf9: parser 5, 772 cards (the 48 Photoshop rule pages are no longer cards); `db/rulebook.json` holds the
  24 chapters.
- psd-tools fork pin `cards-20261026` = e156be36 on both machines (`requirements-cards.txt`).
- OJWikibot main 363ddc9, deployed. One production host: the jetson (leo retired 10-26).
- `+ability` is live; no live `+ability` bake yet (the first comes from the website): its flag note names the box, font
  and source.

## Campaign tasks

### Owner

- [ ] TTS, first real-game checks of the 10-30 save (all were tested only in a Lua emulator): right-click Card details
      (in hand too; three panels; drag, A-/A+, Rules buttons, "+ Name" into the hand face down at the right size); Find
      a card (right-click the table); Contents and search (right-click the rulebook or its bag): does Book.setPage land
      on the right page (0-based assumed)?; hand zones following seats (alone on a side you hold them all; a teammate
      sitting gets theirs back); turn markers (flips with F or the menu count, physics tumbles don't); the first open of
      a panel per player (a short build; it re-lays out other players' open panels, so their dragged spots reset);
      whether TTS's PDF viewer searches the text; any Lua error.
- [ ] Decide: TTS-sized JPEG card copies (the save loads 839 files, 1.5 GB of 1200 px PNGs; half-size JPEGs cut it about
      10x): 600 or 800 px wide, or not. Also the holder pictures (Creature_Deck.png 6.7 MB).
- [ ] Try the website editor's new drags (anchor grid, Make space, blank space, snapping, undo) and report what feels
      off; Safari never ran.
- [ ] Photoshop: Computer Virus' stat bars; Ze'mer's full-width flower layers; Terry Hintz and Joel Miller's "Stars"
      groups and typed tags (prepared in `~/std-work-1026/stargroups/`; Photoshop 2023 can't re-save their
      EarthboundZero text, nor Elmadan's); Papyrus' typed tag (Photoshop's Papyrus font doesn't match the card's).
- [ ] The Panda Future's PSD is 51.6 MB (GitHub warns past 50 MB).
- [ ] Safari: the many-boxes edge scroll (headless checks never ran Safari).
- [ ] Open in Photoshop once: a smart-filter star move (lane 3, S1) and a title moved back onto the card (lane 2); both
      were checked only in psd-tools renders.
- [ ] Looks chosen unattended on 10-26 (sheets in `~/Documents/Tools/wt/scratch/layouts/*/sheets/`; each is one small
      rule to change): Leshy at 4 hides its bottom-left drop and at 1 keeps it alone; banner rows tighten to 0.85 of a
      star; die faces at 1-3; Crowned Magolor's funnel; closing a type row (the last icon fills the hole); the
      two-row grid on adds (Masher, Spamton Neo and Defcon Panda's new icon comes out smaller); Clot Black's and
      Minecraft Panda's card fixes.
- Eventually, never blockers: Photoshop standardizing (one Name and one Ability layer per card, rulebook Header/Page
  Number layers; Mr Jawsum's name layer); the team's rulings draft; developers resetting clones after the 10-24 rewrite
  and reinstalling stoogedings.

### Developer

- [ ] Delete the bot's now-dead Photoshop rule-page handling (psd_parser rule pages, psd_details/psd_parts rule cases,
      the query and export filters, the changelog's rulebook link): no card PSD is a rule page since 10-30.
- [ ] The website's global Escape closes an open dialog even while typing in it (app.js keydown): the rulebook editor
      keeps the draft, but it is surprising.
- [ ] Editor ideas offered, not chosen: replace a picture by dropping a file on it, Ctrl/Cmd+D duplicate, a diff before
      Publish. Deck sheets for TTS: not recommended (every card edit rebuilds a sheet).
- [ ] Card Authors.txt still credits the 48 old rule pages (kept; card_data no longer warns about them).
- [ ] `docs/BUGS_AND_IMPROVEMENTS.md`: live checks V1 (`/sync_emojis refresh:True`) and V4 (`/query Raw Kawk`) are
      still open (Discord commands on the live bot).
- [ ] Regenerate the problem-cards list (`docs/Future/PROBLEM_CARDS_2026-10-25b.*` predates the 10-26 lanes and fixes).
      Deliver the CSV and its `.md`.
- [ ] Stars still refused: Thunder Thighs 3, Consolation Prize, Bubble, Forsaken City and The Core (option B, shrinking
      every star to fit the banner, is the larger fix); Forgotten Crossroads 3-6 (a black gradient) and Spiky Ball Trap
      5-6 (a brick wall) wrongly act as the stars' backdrop; Yin Yarn 6, Paint Warrior 4-6 and Paint Bomb 7-10 are
      refused because a new star would miss the texture clipped onto its group (a copy of the texture would bake them).
- [ ] Types: in an edit that removes and adds at once, the added icons keep today's free spot instead of going through
      the closing loop (a made icon landing on a kept one is refused).
- [ ] Stat bars: Clot Gold's spare hidden HP layers are still shown at low values (E2 in
      `~/Documents/Tools/wt/scratch/layouts/bars/DESIGN.md`).
- [ ] Computer Virus' NO STATS FOUND reaches the data at its next re-parse (or never, once its stat bars exist).
- [ ] Corruption and Dumplings: the font sweep left them out (a PNG bar; a Txt2 reorder).
- [ ] Kept on purpose until their condition holds: the pyoxipng fallback (until `brew install oxipng` everywhere), the
      older-fork checks (retire the `ac-py314-psd125` venv first), the `alt_names` field (removing it rewrites every
      record), Card Edits.json's `aliases_moved` key (a data edit), the `inspect_pickle.py` name.
- [ ] OJWikibot: `ojwiki_helpers.py:263-268` optional mwparserfromhell fallback; the backup search misses
      `Backup: pending.json`.
- [ ] Fork, low priority: a tracking change breaks kerning (Quick Slash, Ominous Red Circle), tracked badge ligatures,
      Photoshop's underline box (Golden Metronome). The cards are fixed in Photoshop; re-bakes of those lines differ.

### Waiting on events

- [ ] OJWikibot's wiki API answers HTTP 500 since 10-23 (wiki/host side); O-6 waits for it. When it's back: read
      All_Cards_Event once for a lone "===Permanent Rotation===" heading (the heal is deleted; remove it by hand if
      there), and check the 2026-10-21 mixer gap in backup/history/mixers.json against that day's wiki page.
- [ ] The first live `+ability` bake: check its placement and font.

### Kept by ruling (no action)

OJWikibot (owner 10-26): `/update`'s diverged-checkout handling (both bots) and the field-events snapshot (a use is
planned). Cards: 42 glow and gradient FX layers; 6 outline FX layers that change colour; the clock emoji; Miracle Matter's form
pictures and the 5 card-art inline pictures; MDW/Old Rules pages unswept.

## Queue (`docs/Future/`)

| File | What is left |
|---|---|
| `PROBLEM_CARDS_2026-10-25b.csv` / `.md` | The last problem-cards list (54 cards); stale, regenerate first |
| `PROBLEM_CARDS_2026-10-25.csv` | The 10-25 morning census |

## Not chosen

Optimization questions answered no or not now: `~/Documents/Tools/wt/scratch/opt/PLAN.md` §(b) (S-5, S-6, B-4, B-5,
P-3, P-4, P-6, K-4, K-6, O-7, O-8, O-9). Parked ideas: `docs/Finished/session-2026-10-24/README.md`.
