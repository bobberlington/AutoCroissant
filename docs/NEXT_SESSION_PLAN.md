# Next session plan (after 2026-10-26)

## Where to start

Start in `~/Desktop/AutoCroissant`. Read memories `resource-limits`, `secrets-never-printed`,
`owner-decisions-2026-10-24-night` (its 10-25 and 10-26 lines), `lean-tests-and-verification`, `no-guides-csv-only`,
`edits-two-kinds`, `jetson-server`, `persistent-work-paths`, `agent-model-preference`. What landed:
`docs/Finished/session-2026-10-26/README.md` (and `session-2026-10-25/`). `docs/Future/` holds only the problem-cards
lists; nothing is in flight.

First: regenerate the problem-cards list (below), then the campaign tasks in the order the owner picks.

## Standing rules

- Work unattended unless the owner says "ask me questions". Agents on Opus or lower; a single Agent for a build (two
  workflow agents answered the owner's latest question instead of building: pin "your job is the code build").
- Commit to main in both repos; `work/*` only while in flight, then delete. Very short commit messages; no Claude
  attribution. Persistent paths only (`~/Documents/Tools/wt/`, `~/std-work-*`).
- Deploy with the `jetson-deploy` skill: push main and deploy in one sitting.
- Tests: `AUTOCROISSANT_LOCAL_DIR=~/Desktop/TTSCardMaker AC_SITE_SRC=<tc tree>
  ESBUILD=~/Documents/Tools/wt/scratch/esbuild/node_modules/.bin/esbuild ~/.venvs/ac-py314-fork/bin/python -m pytest
  -n auto -q -p no:cacheprovider tests` (641 on 10-26, about 3 minutes). Targeted files unless a full run is needed.
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

## State (2026-10-26 ~09:15 PDT)

- AutoCroissant main = deployed a07f822 plus docs and skills commits only.
- TTSCardMaker main 429e724c (plus Card sync's TTS save commits). Desktop clones of all three repos level with main.
- Data 6b8f9050: parser 5 (builder 234b1b0), 820 cards; `db/verdicts.json` holds the verdicts (no `_psd` left).
  Grimora and Skeleton Mate re-parse in the Card sync run after 80466bbe.
- psd-tools fork pin `cards-20261026` = e156be36 on both machines (`requirements-cards.txt`).
- OJWikibot main 77def4b, deployed.
- `+ability` is live; no live `+ability` bake yet (the first comes from the website): its flag note names the box, font
  and source.

## Campaign tasks

### Owner

- [ ] Reset the production Discord TOKEN (jetson's config.py; sent to api.github.com on 10-25; still valid on 10-26),
      then put the new one in the jetson's config.py and restart. Regenerate TTSCardMaker's Discord webhook (its URL was
      printed on 10-26).
- [ ] Photoshop: Computer Virus' stat bars; Ze'mer's full-width flower layers; Terry Hintz and Joel Miller's "Stars"
      groups and typed tags (prepared in `~/std-work-1026/stargroups/`; Photoshop 2023 can't re-save their
      EarthboundZero text, nor Elmadan's); Papyrus' typed tag (Photoshop's Papyrus font doesn't match the card's).
- [ ] The Panda Future's PSD is 51.6 MB (GitHub warns past 50 MB).
- [ ] Safari: the many-boxes edge scroll (headless checks never ran Safari).
- [ ] OJWikibot questions: is the leo host retired; `/update` in both bots; the field-events snapshot; the stray-heading
      fix.

### Developer

- [ ] Regenerate the problem-cards list (`docs/Future/PROBLEM_CARDS_2026-10-25b.*` predates lanes 1-3, the numbered
      cells and the star groups; Twin Emperors was a false alarm; `+ability` cases are handled). Deliver the CSV and its
      `.md`.
- [ ] Star counter: Yin Yarn and Leshy (overlapping and shadowed stars).
- [ ] Star layouts: Bolts and Screws 2's false banner refusal; the 8 off-banner star edits refused (smarter layouts);
      a dice 5 -> 2 rule; Crowned Magolor's 6 look.
- [ ] Types: Nasus' type removal (a close-the-row rule); Masher's "dark" under the shell.
- [ ] Stat bars: Twin Emperors' 2-px seam after 5 -> 10 -> 2 -> 7 (cell spacing 925 vs 921); a numbered-layer 0 refuses
      with "no dark picture" wording (reword); Sonic's SPD 11-19 are website-only.
- [ ] Corruption and Dumplings: the font sweep left them out (a PNG bar; a Txt2 reorder).
- [ ] Kept on purpose until their condition holds: the pyoxipng fallback (until `brew install oxipng` everywhere), the
      older-fork checks (retire the `ac-py314-psd125` venv first), the `alt_names` field (removing it rewrites every
      record), Card Edits.json's `aliases_moved` key (a data edit), the `inspect_pickle.py` name.
- [ ] OJWikibot: `ojwiki_helpers.py:263-268` optional mwparserfromhell fallback; the backup search misses
      `Backup: pending.json`.

### Waiting on events

- [ ] OJWikibot's wiki API answers HTTP 500 since 10-23 (wiki/host side); O-6 waits for it.
- [ ] The first live `+ability` bake: check its placement and font.

### Kept by ruling (no action)

42 glow and gradient FX layers; 6 outline FX layers that change colour; the clock emoji; Miracle Matter's form
pictures and the 5 card-art inline pictures; MDW/Old Rules pages unswept.

## Queue (`docs/Future/`)

| File | What is left |
|---|---|
| `PROBLEM_CARDS_2026-10-25b.csv` / `.md` | The last problem-cards list (54 cards); stale, regenerate first |
| `PROBLEM_CARDS_2026-10-25.csv` | The 10-25 morning census |

## Not chosen

Optimization questions answered no or not now: `~/Documents/Tools/wt/scratch/opt/PLAN.md` §(b) (S-5, S-6, B-4, B-5,
P-3, P-4, P-6, K-4, K-6, O-7, O-8, O-9). Parked ideas: `docs/Finished/session-2026-10-24/README.md`.
