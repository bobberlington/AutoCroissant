# docs/Future

Designs for work that is decided but not built, one short doc each (Status, goal, the owner's rulings, design,
verification, open questions only if real), and the live problem-cards list. The order is the plan's queue
(`docs/NEXT_SESSION_PLAN.md`).

| # | File | What it is |
|---|---|---|
| 1 | `PARTS_EDITOR.md` | The website parts editor: one text box per PSD layer, variant tags, stars folder moves, first ability/type from the template; three build tracks. Designed, build next |
| 2 | `TEST_SUITE_CUT.md` | xdist and the cut (1,923 collected -> about 635), the 15 site smoke tests, discord_archive/discord_match retired |
| 3 | `SMALLER_PNG_PUSHES.md` | The oxipng CLI, XMP strip and opaque-as-RGB for every PNG the bot writes; no rewrite of existing PNGs |
| 4 | `WEBSITE_SMALL_FIXES.md` | #15: phone tooltips, exact name first, deck export with pinned links, rulebook page links |
| 5 | `LAYOUT_SUPPORT.md` | Bakes for the 31 non-template stat bars and the 21 unusual star layouts |
| 6 | `CARD_SYNC_VERDICTS.md` | Bake verdicts to `db/verdicts.json`; bake and guard events in the changelog |
| 7 | `GAME_TXT.md` | One plain-text file of every card and rulebook page (later) |
| - | `PROBLEM_CARDS_2026-10-25.csv` | The 10-25 census (card, folder, kind, reason): 124 of 764 cards with an edit that can't bake, 4 data problems, 308 bakes with a note. Re-run when the parts editor and layout support land |

When an item ships, its doc moves to the session's folder in `docs/Finished/`.
