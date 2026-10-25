# Session 2026-10-25 (unattended, owner asleep from ~07:45 PDT)

Everything below landed on main in both repos and is deployed (AutoCroissant be39f86 on the jetson; OJWikibot
5135280). Designs that shipped are in this folder; `docs/Future/` keeps the two with a step left.

## Landed

| What | Where | Notes |
|---|---|---|
| Code cleanup | AC 8198291, TC 2770b982 | Pickle-era code, Discord edit credit, finished tools, dead code, the fonts cache, path tuples; card-check and font-check folded into the guards' check job; 223 duplicate fonts removed (KungFuPandaRegular-xYjV.ttf kept: an old copy that matches no current file) |
| Card data on a merge head | AC abc5ec9 | `git show -s` diffs a merge (needs blobs under GIT_NO_LAZY_FETCH): `git log -1` |
| Website small fixes, game.txt | TC d0dfed64, 56651852 | `WEBSITE_SMALL_FIXES.md`, `GAME_TXT.md` |
| Test cut | AC f32dfa7 | `TEST_SUITE_CUT.md`: 1,563 -> 553 tests, about a minute; `-n auto` capped at 6 (e2f287d) |
| Leftovers | TC b67b8799, AC 032ede9, 8a1e629 | TTS/audit.py and AUDIT.md removed (last at 843a4a99); `discord_edits.py` -> `card_dates.py`; reminder retirement shims removed |
| Backups | AC be604a5, OJWikibot 5135280 | Live reminder/history files stay untracked; the bot commits a copy to `backup/` on change over SSH from a separate clone (neither GitHub token can write the bot repos) |
| Optimization clear wins | AC d3f622c, TC b77998c0 | `/query` fuzzy p95 129 -> 6.5 ms, the `/query` refresh race fixed, faster restarts, single-pass git walks, opaque-pixel fast path, font-name index, inlined CSS + esbuild minify (836 -> 520 KB), wheel cache on requirement lines. Plan and questions: `~/Documents/Tools/wt/scratch/opt/PLAN.md` |
| Guard and bake pack | AC 4d4574c, TC 1c717c8b | Straight apostrophes everywhere; badges sized 1.35x the text's drawn cap height and centred by Baseline Shift; old stoogedings glyphs redrawn; all silent (commit messages only) |
| Sweeps | TC 218a122d, 6778c592 | Badges on 101 cards, apostrophes on 34; guards on both pushes green and in agreement (byte-identical fixes) |
| Parts editor, data + bake | AC aae7233, TC abf70775 | `docs/Future/PARTS_EDITOR.md`: parser 3, details v2 with parts, per-layer bake, variants, star folder moves, first type icon; full re-parse: 1 field change (Ms Mowz's display name) |
| PNGs, layout, verdicts | TC a4d137bf, AC be39f86 | `SMALLER_PNG_PUSHES.md`, `LAYOUT_SUPPORT.md`, `docs/Future/CARD_SYNC_VERDICTS.md` (migration left) |

## Incidents

- **Out of memory, 07:17.** 429 git processes (~226 GB). The likely cause, reproduced at 15:05 and stopped: a
  measurement agent made a blob-less clone of the local 16 GB card repo, so every blob read became a lazy fetch with
  its own `upload-pack` and `pack-objects`. Rules now: never partial-clone the local repo, xdist capped at 6, one full
  suite at a time (`suite-wait.sh`), a git watchdog; memory `resource-limits`.
- **Wait loops that matched themselves.** `until ! pgrep -f "pytest.*-n"` finds its own shell; replaced by
  `~/Documents/Tools/wt/scratch/suite-wait.sh`.

## Flags for the owner

- Michael's Blessing: the badge sweep shrank its badge table; the "->" arrows no longer line up.
- Not fixable glyph by glyph (a Photoshop re-save fixes them): badges on Restrained, WX-78, Wendy, Wigfrid, Soul
  Master, Traitor Lord, Greenpath, Soul Sanctum, Ominous Red Circle, Quick Slash, Golden Metronome; apostrophes on The
  Temple/Gates of Ahn'Qiraj titles, 04_Table5, 06_Reading_Cards, 12_Reactions2, 10_Attacking, 17_Death; the
  MDW/Old Rules pages were never swept.
- Layout support still refuses: Grimora and Skeleton Mate (bars painted in one layer), Slugcat's spd, three
  smart-filter star layouts, PoD04 Zebra Bloon +1, Living Bread.
- OJWikibot's wiki API has answered HTTP 500 since 10-23 (wiki/host side).
- The Card sync schedule fires about 8% of the time (GitHub drops scheduled runs): opt PLAN question P-5.
