# Session 2026-10-26

Continued from 10-25 after midnight. With the owner until ~08:00 ("ask me questions" mode), then unattended. Everything
below is on main in all repos. The jetson runs AutoCroissant 47c79e5 and OJWikibot 363ddc9; the psd-tools fork pin is
`cards-20261026` (e156be36) on both machines. Times are PDT.

## Landed

| What | Where | Notes |
|---|---|---|
| `+ability` live | TC 8a913cc4, AC d7a6641 | Owner approved `plus_ability_2` (~02:00). Placement and font: 10-25 README, after midnight |
| People star levels by kind | TC 7834c622 | The Style panel's Star levels mix has a kind picker (All, Creatures, Minions, Items, Field cards), kept in the address |
| Ratings whole numbers | TC 98c4b087, AC 9d3619d | Entry and storage 1-10; stored values rounded half up; averages keep decimals |
| Hover notes | TC 4cb07fad, 38cc061b, bcb2b42d, 63045a17, 18970a1c | `data-tip` notes on the stat tables (People, Stats, Ratings, Details counts) |
| 18 badge and apostrophe cards | TC bd97c4bb, cc8fb017, cb6c64cf, 48f4b326, dac2f826; AC ffa3ef3, 955c03e, 0f6f133, d036301 | 5 by a text re-render, the owner's 6 apostrophe files, 7 badge cards re-saved in Photoshop 2023 (driven by `open -a … script.jsx`), Quick Slash's Ability box narrowed. The parser reads `\'` as `'`; a `Card-sweep:` trailer makes its own commit silent (per commit). The owner's 8 no-op badge re-saves were reverted (f0668650). 03 Setting Up re-read without its backslash |
| Nebulous Precipice | TC 7460688b | Option C: a `[darkmatter:sticker]` token; its picture layer and glow removed |
| Picture-only duplicates | TC 410eb8d8, 80c35f53; AC 45db8bc, 21bf7ae | Stubborn Queen and Gust of Trees show the original's data with their own name, picture and artwork sources. Font chips toggle off and stay listed; the first-screen tiles keep links; a ratings "about" line |
| Every grid tile a link | TC f937dcf4, AC 04bcb05 | Outside edit mode; middle-click opens a tab |
| Profile subseries | TC 2bb57c28, bc8ca5a6; AC ce49031 | Profiles take an optional subseries; MichaelJSr's "Don't Starve Gourmet Expansion" split into series plus subseries |
| Verdicts migration | data 0680843, TC fec70c94 | No `_psd` blocks left in Card Edits.json; the fallback code is deleted (dead code, below) |
| Font sweep | TC 1ad6a0e4, 7e6cedef, 383fd2ed, 0bd3b5be | Typed timing tags on 5 cards, inline and heading icons as stoogedings tokens on 9 PSDs, 127 hidden leftover layers removed on 81 cards, 1 outline-only FX layer removed |
| Star groups | TC f5bbcd45 | Star layers grouped as "Stars" on 8 cards |
| Lane 1: star counter, titles, parser 5 | AC 67b0fba (9d9eff2, 89c8e70, f99600a, 13a0f65, 51ce44f) | One star counter for parse and bake (R1); rule P finds a title by its place; dice keep their shape, no shadowed moves, stars stay on the banner. `PARSER_VERSION` 5: full re-parse (Living Bread 2 stars, Massive Moss Charger 4, Thunder Thighs' display) |
| Lane 2: titles, bars | AC 234b1b0 (c7f0f0f, a213208), TC a80a32aa (b83e51d2, c364cfd7, 86288f55) | Omori titles redrawn in the face the pixels show and the PSD's font name fixed; title box anchor; picture titles are website-only name edits with an editor note; stat bars edit again (split cells keep their look; numbered layers); multi-creature stats bake creature 1 only; two-row bars; Ze'mer refused; Slugcat SPD 5 |
| Lane 3: stars and icons | AC a10faa6 (670bd4a, d94a8a6, b5495f7), fork `cards-20261026` | Tainted Lazarus' two star sets alike, centred rows, rows of two, smart-filter star layouts (S1), made icons (S2, S3), stars under art refused (`_covered`) |
| Numbered stat layers | TC 80466bbe | Grimora and Skeleton Mate split into numbered cells (1-10 lit over 1-10 dark per stat) in Photoshop; 80/80 bakes |
| Dead code | AC 42170d4..a07f822, TC 3c33b34b, 311821b1, d87fd3b1, 429e724c | Owner ~08:00: the verdicts `_psd` fallback and card-sync.yml's older-builder probes; `_joined_edit`; dead constants and helpers; the reminder migration and `LEGACY_RUNS_AS_BOT` (a reminder with no creator runs as an unknown non-admin); the retired badge FX branch; pickle leftovers; `tools/variant_tags.py`; the `vram_usage` cookie fallback; stale test skips; the site's dead fallbacks (old thumbs cache, WebP meta, ratings rounding, numbered-PNG scan, jsDelivr fallback) and the plain-text sign-in migrations (plain-text secrets of before 10-23 are dropped at load). Kept: (1PGT)/(E2T), the first screen, pyoxipng, the older-fork checks, `alt_names`. Suite 641 passed; deployed |
| OJWikibot dead code | OJWikibot 77def4b | 23 dead rows; deployed |

## After 10:00 (the owner's answers, then unattended)

The owner answered four questions (~10:00), then asked for unattended work. Where a look was still open, the
coordinator took the diagnosis' recommendation (marked "chosen" below); the design docs with the option sheets are in
`~/Documents/Tools/wt/scratch/layouts/{stars,types,bars}/DESIGN.md`.

| What | Where | Notes |
|---|---|---|
| Jetson tidy | jetson | `~/build` trimmed to its recipes (1.2 GB of ffmpeg build tree, the OpenSSL 3.6.0 rollback archive and logs gone); `~/Documents/Tools` (one timestamp file) removed; stale `.pyc` files of deleted modules removed. The `-backup` clones stay: the bots commit their backups through them |
| OJWikibot cleanup | OJWikibot 7297ecb..119f27b, 363ddc9 | Leo retired: its units and the 3.10 alias gone (Python 3.11+); the 09-09 stray-heading heal deleted; /update's diverged-checkout handling and the field-events snapshot kept on purpose (noted in change-control); stale comments reworded; README/skills drift fixed; store_heal dry-runs on backup/ off the jetson and refuses `--apply` there. Deployed 119f27b |
| Three bugs | AC 97b1e73, 3b43659, d287148, 2918732; TC 579eef91, 2b0d3fb3 | A refused card's website-only fields still move to `_site` (an edits-only main commit); R1 checks every ability layer the parser reads (0 cards newly flagged); a creature or minion with no stat bars read gets NO STATS FOUND (Computer Virus at its next re-parse), answered by all four stats or a creatures edit |
| Follow-ups | AC d376c3f, 86dce67, 97eb6d1, fa85665, 74c7704; TC 773c73c2, 06a8bde9 | The 3.10 alias gone in AutoCroissant too; Auxiliary/Creatures in the stat checks; R1 tolerates a spaced "Ability" name; docs |
| Stars | AC caffa2f, 34c3e6e, 380f228, 3068bef; TC 1536d831, fc6784fe | A star's core is over half opaque (Leshy counts 5); a hide may not bare a kept star's shadow (Leshy at 4 hides the bottom-left drop, chosen C); a moved star keeps a texture clipped onto its group (Yin Yarn 6, Paint Warrior 4-6, Paint Bomb 7-10 refused); a banner must be plain where the stars sit (Bolts and Screws bakes as a column); a row centred and tightened on its banner, tried last (chosen A: 10 more plans bake); die faces for dice-5 cards at 1-3 (chosen B); Crowned Magolor's V point becomes a pair at 6 (chosen B). Leshy's and Yin Yarn's "Group 1" renamed "Stars" with the psd-tools fork (the prepared Photoshop copies also re-rendered text) |
| Types | AC 69167cf, a7ec2ac, d143c61, 3d2f9ac | Type icons in a Stars group are no stars (Pike Beam, Dark Matter bake); a closed row makes room (chosen B: the last icon fills the hole, the others slide only to clear); two-row headers take a grid on adds (chosen: the mock-up picked 10-26 for Masher), falling back to today's spot; made icons checked against kept ones |
| Stat bars | AC a4ccc9e, 01a2d40; TC 22484244, e401c98c | The bar's own cell edges in both plans (Twin Emperors' seam and its HP 1); one clear reason for a 0 that shows nothing; Clot Black's HP row gets hidden dark hearts 1-2 and Minecraft Panda's dark strips move into their rows (chosen D and C: both bake every value; PNGs unchanged); Sonic's SPD 11-19 stay website-only by design |
| Lands | AC 47c79e5 (merge of 4 lanes, 654 tests), TC 06a8bde9 (one push; guards dry: 4 cards, 0 told, 0 writes) | Deployed AutoCroissant 47c79e5; OJWikibot skills 363ddc9. Data c1389ce ("4 art", 0 posts: Leshy and Yin Yarn read 5 stars, stats unchanged), then a `cards:` dispatch of the 72 cards whose stored star/type headers the new rules move: data 98faaf0 (psd_details only, stats unchanged) |

`~/Documents/Tools/wt/scratch/suite-wait.sh` never saw a running suite (macOS cuts `ps -o comm` to 16 characters); fixed
to match the executable in `args`.

## The owner's rulings of the day

From memory `owner-decisions-2026-10-24-night` (its 10-26 lines):

- `+ability` live; template alignment kept, centring not adopted.
- Ratings are whole numbers 1-10; existing values round half up; averages keep decimals.
- Paragraph rewraps from a badge resize are fine. Photoshop 2023 may be used for card fixes, on copies, never on
  documents we didn't open.
- No FX layers where a stoogedings token can do it (Nebulous Precipice: option C). Keep all glow and gradient FX layers;
  delete outline-only FX layers and hidden leftover layers.
- Problem cards: picture titles become website-only name edits; multi-creature stats bake creature 1 only; painted
  bars move to numbered layers (cells named 1-10); two-row bars decrease on the top row, increases past the rows are
  website-only; Ze'mer waits for full-width flower layers; Slugcat SPD 5; the 12 unfindable-star cards get a "Stars"
  group in Photoshop (done on 8).
- Twin Emperors is right (Vek'nilash DEF 4 is creature 1). Computer Virus: the owner adds stat bars.
- TyKim182's write access removed.
- ~08:00: work unattended, delete deprecated and fallback code, clean up fully, list future campaign tasks.

## Incidents

- **A webhook URL printed.** Listing TTSCardMaker's repo hooks printed the Discord webhook URL (the secret is in its
  path). The owner should regenerate it. Rule (memory `secrets-never-printed`): print a hook's name, active flag and host
  only.

## Cleanup

Open items and the future campaign tasks: `docs/NEXT_SESSION_PLAN.md`.

No worktrees or `work/*` branches left in AutoCroissant, TTSCardMaker, OJWikibot or the psd-tools fork (the fork's
filteritem branch deleted). Scratch: the landed lanes' folders (about 29 GB) deleted; their top-level scripts kept in
`~/Documents/Tools/wt/scratch/lane-tools/`.
