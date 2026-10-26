# Session 2026-10-25 (unattended, owner asleep from ~07:45 PDT)

Everything in Landed is on main in both repos and deployed (AutoCroissant be39f86 on the jetson; OJWikibot
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

## Afternoon (owner back, ~13:30-17:10)

On main; bot code after be39f86 is not deployed.

| What | Where | Notes |
|---|---|---|
| Variant tags | TC 572680d5 | ' #N' tags on the 5 multi-picture PSDs (names only) |
| Michael's Blessing | TC b6b22965 | text sized to its badges, arrows aligned (the badge sweep's flag) |
| Guards window from the push (P-1) | AC fc420bb, TC 594050bf | `card-guards-quiet` timer 25 min (set), Land sleeps to 30 min after the push |
| Dirty-box gate (K-1) | AC 4047e4b | the untouched gate renders only the dirty box; `PSD_DIRTY_VERIFY` (full compare) on in tests |
| Card sync chain (P-5) | TC 3ca08fda, AC 8690917, e45c127 | a run with work left dispatches the next; one safety cron at :17 (`0,30` gone) |
| Instant replies (B-1) | AC d1725ce, 9694021; OJWikibot a70877d | drains wake on enqueue, 1 s fallback |
| OJWikibot lane 3 | OJWikibot 8ea1c7c..174f62b | worker reply slices, early stop, no blocking start, C twister |
| Site: bundle, WOFF2, lazy chunks (S-1, S-4, S-3) | TC c31e67a3, 7fb7b355, 0a1bfbe3; AC 6a87179, 95b5f20 | one esbuild bundle with a source map; WOFF2 copies of the mock-up's fonts; editor, Stats, People on first use |
| First screen in index.html (S-2) | TC c31e67a3, removed 54395592; AC 4a4ad4c | Safari redrew its pictures (the flash); removed the same evening |

## Incidents

- **Out of memory, 07:17.** 429 git processes (~226 GB). The likely cause, reproduced at 15:05 and stopped: a
  measurement agent made a blob-less clone of the local 16 GB card repo, so every blob read became a lazy fetch with
  its own `upload-pack` and `pack-objects`. Rules now: never partial-clone the local repo, xdist capped at 6, one full
  suite at a time (`suite-wait.sh`), a git watchdog; memory `resource-limits`.
- **Wait loops that matched themselves.** `until ! pgrep -f "pytest.*-n"` finds its own shell; replaced by
  `~/Documents/Tools/wt/scratch/suite-wait.sh`.

## Flags for the owner

- Michael's Blessing: the badge sweep shrank its badge table (fixed in the afternoon, TC b6b22965).
- Not fixable glyph by glyph (a Photoshop re-save fixes them): badges on Restrained, WX-78, Wendy, Wigfrid, Soul
  Master, Traitor Lord, Greenpath, Soul Sanctum, Ominous Red Circle, Quick Slash, Golden Metronome; apostrophes on The
  Temple/Gates of Ahn'Qiraj titles, 04_Table5, 06_Reading_Cards, 12_Reactions2, 10_Attacking, 17_Death; the
  MDW/Old Rules pages were never swept.
- Layout support still refuses: Grimora and Skeleton Mate (bars painted in one layer), Slugcat's spd, three
  smart-filter star layouts, PoD04 Zebra Bloon +1, Living Bread.
- OJWikibot's wiki API has answered HTTP 500 since 10-23 (wiki/host side).
- The Card sync schedule fired about 8% of the time: answered by the chain and the :17 safety cron (afternoon).
