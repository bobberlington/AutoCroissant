# Session 2026-10-26

Continued from 10-25 after midnight. With the owner until ~08:00 ("ask me questions" mode), then unattended. Everything
below is on main in all repos. The jetson runs AutoCroissant a07f822 and OJWikibot 77def4b; the psd-tools fork pin is
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

No worktrees or `work/*` branches left in AutoCroissant, TTSCardMaker, OJWikibot or the psd-tools fork (the fork's
filteritem branch deleted). Scratch: the landed lanes' folders (about 29 GB) deleted; their top-level scripts kept in
`~/Documents/Tools/wt/scratch/lane-tools/`.
