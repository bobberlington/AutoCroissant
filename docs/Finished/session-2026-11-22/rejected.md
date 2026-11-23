# Rejected and dropped ideas, session 2026-11-22

Every idea rejected, dropped or not kept, with the verifier's or judges' reason in one line. Full texts: the workflow
journals (research5 wf_28f2b897-21a, tasks-1122 wf_ff5043c3-240) and `scratch/s1122/research5/`.

Retries (two judges, then a rebuild with the objection fixed and a fresh verifier): actbake 3 judged, 3 rebuilt,
1 kept; cards 9 judged, 8 rebuilt, 8 kept, 1 dropped. Not kept in the end: 5 (actbake x0, x1; cards x8; actbake r1-3,
r5-2), plus cards r6-3 part 2.

## actbake

| Round | Idea | Why not | Retry |
|---|---|---|---|
| r1-2 | Mock-up: Ability rows that start with spaces, and stacked boxes, drawn where the bake sets them (Miracle Matter, Warcraftian Druid, Twin Emperors) | Rule 3 made Hive Knight's redrawn lines paler and bolder than the bake | x0, rules 1-2 only: NOT KEPT. On Spur, a re-wrap draws the second paragraph one space right (the verifier names a 4-line fix) |
| r7-0 | Mock-up: a growing Ability set smaller where the bake sets it smaller (Papa Chip, Warcraftian Mage, Whispy Woods...) | Shrinks Abilities the bake keeps full size (Nefarious Chip, Godskin Apostle) | x1: NOT KEPT. Same-length edits now preview smaller (The Ringed City '+2' -> '+3', Whacka Bump, The Evil, Fricked Up Cursed Idol) |
| r7-1 | Bake and mock-up: words typed at a coloured paragraph's end take its colour | Yin Yarn: the mock-up drew tan, the bake lavender | x2: KEPT (accepted.md) |
| r1-3 | Card sync: the main-lane push retry warnings say what happened | Not built: already on main (TC 8ae326d1, 11-06) | - |
| r5-2 | Rulebook PDF: the push's Pages zip downloaded again after a 404 | Kept, but the same fix as r5-0: dropped at merge | - |

## cards

| Round | Idea | Why not | Retry |
|---|---|---|---|
| r1-2 | Editor: a long title fits the title box on a 390 px phone | Words that showed whole now broke mid-word ('Thalassophobi/a', 'Frostmourn/e') | x0: KEPT |
| r2-2 | Card dialogs: closing one goes Back without the Navigation API (old Safari, Firefox ESR) | Two history calls per card step: Safari's 100-per-10-s limit jams the dialog twice as soon | x1: KEPT |
| r3-2 | Card history: an older reading the card's picture contradicts is no change | Hero Cap and Slugcat pages lost the rows that balanced their timelines' first record | x2: KEPT (3 cards only) |
| r4-3 | New creature cards: the series their folder's cards share (Beholster showed 'other') | The editor half disagreed with what the bot restores (43 cards); a typed-back series could be lost | x3: KEPT (parser half only) |
| r6-0 | Search: a misspelled word suggests the printed word ('peircing' -> piercing) | Dark theme: the rulebook "Did you mean" link at 2.05:1 contrast (owner D1: AA) | x4: KEPT |
| r6-1 | TTS Find a card: an ability word finds its cards | Words matched across line breaks (Isma for 'reaction') | x5: KEPT |
| r7-2 | Rulebook search: an abbreviation typed alone (SP, OS, DEF) finds it as a word | The website hid sections whose first line starts with it (Damage Types' LIFE STEAL for 'DMG') | x6: KEPT |
| r8-2 | /query: a typed word goes to the card with it as a whole word ('rogue' -> Warcraftian Rogue) | 1-2 letter autocomplete reordered ('d' lists D.D.T. first; 24 of 25 names gone) | x7: KEPT |
| r8-3 | Tests: PSD_DIRTY_VERIFY only in the edit tests (~6% off a suite) | Against the owner's 11-12 G3 keep | x8: DROPPED by both judges (owner pick) |
| r6-3 part 2 | Tests: the caps test without PSD_DIRTY_VERIFY | Unset the owner's G3 keep in one test (cards s2 skeptic) | Backed out at cards s2 (43ebb2c0); part 1 shipped |

## fork

| Round | Idea | Why not | Later |
|---|---|---|---|
| r3-0 | Smart filters: Perspective Warp and Mosaic | The trim refusal didn't cover Perspective Warp | Split: Mosaic (r5-0) and upright Perspective Warp (r6-0), both kept |
| r4-0 | A warpCustom from a file of another pixel size (Photoshop's Relink rule) | Card sync updates, it doesn't relink: a visibly wrong card when a warped file changes size | Redone as Update Modified Content (r7-1), kept |

## Card fixes (cardfix judges, 2 passes)

0 cleared both judges. Beholster's copied Ability and the 'Smouldering' spelling went to the owner (answers 12, 13).
The 9 rulebook typos they found were fixed on owner answer 18 (TC 876f5b70); 2 more, found later, on answer 23 (daca7927).
