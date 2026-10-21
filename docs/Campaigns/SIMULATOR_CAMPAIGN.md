# Simulator campaign (2026-10-20)

**Status:** designed 2026-10-20, not scheduled; plan item 16 (`docs/NEXT_SESSION_PLAN.md`, "Ideas 5-22"). The user's idea
"card game win and loss stats" (pasted 2026-10-20 as item 1 of a second list; item 17 is
`docs/Campaigns/GAME_REPORTS_CAMPAIGN.md`, the home of the shared game report schema); starts after the deck browser
(public decks as repo files, private decks encrypted in the repo; sign-in top right; opt-in profiles
`site/profiles/<login>.json`, not yet created; contributions and a leaderboard). The tab is **Play** ("Table" was the
reverted TTS-save viewer, TTSCardMaker `75692fb`). Rule citations are rulebook pages as
`.claude/skills/impossibility-simulator-rules/SKILL.md` names them; card counts are
`~/Desktop/TTSCardMaker/site/dist/data/cards.json` of 2026-10-20, over the 767 distinct cards unless a count says rows.
Where the rulebook is silent this document says so; nothing below invents a rule.

## 1. Goal and non-goals

**Goal.** Let a player test a constructed hand on the card browser in under a minute (which items the 7 star points
per global turn can pay for, who goes first, what an attack does), play a full game in the browser with the rules
kept by the page instead of by memory, and have every finished game, in the browser or in Tabletop Simulator, land as
a report file that feeds card and player statistics. Eventually: simulate many games between two hands and report
win rates and card performance. Measured at each layer by the "you have a result when" line.

**Non-goals.** Not a replacement for Tabletop Simulator: the save `~/Desktop/TTSCardMaker/TTS/Impossibility
Simulator.json` (kept by `~/Desktop/AutoCroissant/commands/tts_save.py` and TTSCardMaker's
`.github/workflows/tts-save.yml`, rules `TTS/decks.json`) stays the real table. No server: the site stays static GitHub
Pages built by `site/build/build_site.py` from repo files; no backend accounts: sign-in is the existing GitHub token of
`edit.js` (`public_repo` scope and push permission on the repo, so only collaborators can publish from the browser;
everyone else files reports through the bot, §4); no real-time relay we host. No AI play. No ruling made in code: a
contradiction in the rulebook is settled by the user in the rulebook, then encoded; where Layer 1 must pick a page to
compute from, it shows the assumption beside the number and decides nothing.

## 2. Why layered

Abilities are free text. `cards.json` has 801 rows (767 distinct cards after dropping the 6 `dup` rows and deduping
on path and name): 302 creatures, 195 items, 108 minions, 52 debuffs, 50 field cards, 49 auxiliary items, 26 N.M.E.
items, 12 buffs, 7 `Auxiliary/Creatures` cards. 751 rows carry an `ability` string (median 113 characters, longest
515); 50 are blank (34 minions, 7 creatures, 7 items, 2 auxiliary items). Nothing in the data says what an ability
does; the parser only knows where its text and `[type]` icons are. The constructed-legal pool (unbanned creatures 2★
to 5★ and items 1★ to 4★; 53 banned ids) is 432 distinct cards (269 creatures, 163 items; 448 counting every row, the
extra 16 being second rows of the same PSD), 236 with a timing tag, 150 naming one of the 64 status effect cards
(word-boundary match on the card names), 4 with no text (Panda, The Master, Stubborn Queen, Gust of Trees). A full
engine therefore means writing and reviewing about 430 ability scripts on top of the structural rules, and settling
the rulebook's open contradictions first: a campaign, not a feature. Yet most of the testing value needs no engine:
star-point budgets, shop odds, turn order and attack arithmetic are closed formulas on the rulebook's pages, and a
sandbox table with a log replaces paper. So five layers, each useful alone, each a gate for the next.

## 3. The five layers

File names in the headings are new unless they already exist in `site/assets/`; the Play tab is `play.js`.

### Layer 1. Hand test stats (`site/assets/odds.js`, shown in the Play tab `site/assets/play.js`)

- **Ships.** For the active hand of the deck builder: star-cost and item-type curves (14 Item Types), creature-type
  and stat totals (`total` from `search.js`); the **star-point plan**: each global turn resets a team to 7 star points
  (08 Global Turn, 13 Items) and an item costs its stars (13 Items), so the page shows the fewest global turns that
  play every item, the items that never fit together in one global turn, and the free star points per plan; the
  **BRICKS plan**: +1 BRICK per team at the end of each global turn (08 Global Turn), a mining creature generates
  "BRICKS equal to its STAR LEVEL" (09 TakingYourTurn), ROCK +1 (20 Creature Types), a field card costs 2x stars (18
  Field Cards): global turns until each field card is affordable for a chosen set of mining creatures (starting
  BRICKS: rulebook silent; the TTS save's BRICKS counters start at 0); **shop odds**: the item shop lays out 2 items
  per star level from the item deck, reshuffles
  once per global turn for 1 point and restocks the same star level (05 Shops), so P(a wanted item is revealed at
  setup or after k reshuffles) over the pool of that star level; N.M.E. shop 4 random of 26 (05 Shops); field shop 2
  per level, 1 revealed, the rest at the end of global turn 1 (05 Shops); **random item draws** "from a copy of the
  full deck until an eligible item is found" (13 Items) for the 15 legal cards that draw; a **turn-order preview**
  for 3 chosen creatures against 3 of the opponent's (most speed > least health lost > highest star level > RNG, 08
  Global Turn); an **attack calculator**: ATK - DEF as 07 Stats3 and 10 Attacking state it and 10 Attacking's worked
  examples compute it (7 - 1 = 6), x1.5 at double speed, x2 at triple (10 Attacking), rounding to the nearest half
  upward, additives then multiplicatives (07 Stats5), damage rounds up (10 Attacking2), defense mode doubles DEF (09
  TakingYourTurn). Two assumptions shown beside the result, like the shop pool's below: 21 Damage Types' NORMAL "DMG -
  (1/2*DEF)" is not used (Layer 3 ruling 1; the rules skill's Known gaps read it as an old-rules leftover) and the
  page shows both values until the ruling; sneak attack "25% of their attack stat +1" piercing (10 Attacking) is
  shown unrounded and without combo damage, both marked unruled (Layer 3 ruling 8).
- **Not shipped, on purpose.** Opening-hand and mulligan odds and "odds of drawing X by turn N": the rulebook has no
  draw step and no mulligan. The starting hand is the whole hand, 7 creatures and 10 items (03 Setting Up; "You
  start with only 10 of them", 13 Items), and of the first 3 creatures the page says only "Each team fills each of
  the 3 slots with a creature face down to start" (03 Setting Up): no draw is described. Only the "simple version"
  decks for new players are drawn from (04 Table5); that is not constructed. Playing a hand against empty slots
  needs a table: it arrives with Layer 2.
- **Data.** `cards.json` as today; the ban list (`site/Constructed Ban List.txt` via `banlist` in `cards.json`).
  Constructed's shop pool is undefined (the ban list file only restricts the hand and lets each player seed one
  field card of 3★ or less): the page states its assumption (all cards of the level) and lets the user exclude both
  hands.
- **You have a result when** the user checks one real hand and every number matches a hand calculation from 03, 05,
  07 Stats5, 08, 09, 10, 13 and 18; the deck builder's share link `d/` opens the same hand in Play.
- **Size.** 2 to 4 days.

### Layer 2. Sandbox table (`site/assets/game.js` state and log; UI in `play.js`)

- **Ships.** The table as the rulebook lays it out, nothing automated but bookkeeping. Per team: the hand (03 Setting
  Up), 3 slots in 3 lanes, each slot a primary and a secondary field (16 The Field), a facedown item holder behind
  each slot (03 Setting Up, 04 Table3), HP/DEF/ATK/SPD counters beside each slot (04 Table3, 04 Table4), status
  effect markers on the slot (04 Table3), minion holders and active item holders (04 Table4), the graveyard (04
  Table3, 17 Death), one field card slot (04 Table3; one in play per team, 18 Field Cards), counters for star points,
  BRICKS and points (04 Table2). Shared: the discard pile (04 Table3, 13 Items), the global turn counter starting at
  1 (04 Table2), the item shop (8 slots), field shop (8) and N.M.E. shop (4) (04 Table2, 05 Shops), the auxiliary
  items and minions decks (04 Table4), creature type tokens (04 Table5), a die and a coin for RNG (the Redirection
  cube and Order Determinator, 04 Table4; "flip a dice, or roll a coin", 08 Global Turn). Drag, flip (creatures start
  face down and flip when both are ready, 03 Setting Up), turn 90° for defense mode (09 TakingYourTurn), draw from
  the auxiliary decks and the shops, stat counters in 0.5 steps (07 Stats1, 04 Table4), status effect tokens coloured
  from the status chart (`MDW/Status Effects.aco`, 66 status colours among its 79 swatches; authority
  `MDW/Croissant's Special Folder/Satus Effect Colors.psd`) with the status card's own text on hover (the rulebook defines them only by example, 19 Status
  Effects), a turn and phase tracker (phase 1 (BOT), phase 2 (OYT)/items/actions, phase 3 (ENDT), phase 4 status
  damage and ticks, 09 TakingYourTurn; (BOGT) and (ENDGT) at the global turn's ends, 02 Abbreviations, 11
  Abilities3) with the tag
  badges of `text.js` (`tagInfo`, `badgeHtml`), an append-only log, undo. **Stages:** hot-seat (one browser); then
  play-by-link: the whole state encoded in a share link `g/` like `d/` (deck.js `encode`/`decode`), each player
  pasting the next link in Discord; real-time peer play later (see open questions).
- **Data.** `cards.json`; status colours exported by the build into `cards.json` (`statusColor` per buff/debuff; the
  `color` field today is the art's accent, not the chart); `rules.json` for the hover texts of the phases.
- **Rulebook encoded.** Only names, zones and counters: 03, 04 Table2 to 04 Table5, 09, 13, 16, 17.
- **You have a result when** two people finish a hot-seat game with the log readable afterwards; the rulebook's own
  turn (the 08 Global Turn order example and the 10 Attacking C-vs-D hit, rules skill §12) replays exactly; a
  play-by-link game survives a browser restart; the end of either produces a report (§4) with no retyping.
- **Size.** Hot-seat 1 to 2 weeks; play-by-link 2 to 3 days more; real-time play 1 week or more, after a decision.

### Layer 3. Rules automation (`site/assets/engine.js`; abilities still applied by hand)

- **Ships.** The structural rules, each with its page in a comment: lane and slot order (08 Global Turn); dead
  slots skipped but counted (16 The Field); the four phases and one action per creature (09 TakingYourTurn); every
  minion and parent acting in the owner's order (15 Minions); attack arithmetic, combo damage, sneak attack,
  multi-hit as separate attacks, left-to-right multi-target (10 Attacking, 10 Attacking2); rounding and MAX HP (07
  Stats1, 07 Stats2, 07 Stats5); defense mode until the next turn begins (09); star points to 7 and +1 BRICK per
  global turn (08); item costs and the discard after resolving, multi-use items back facedown or to hand (13 Items);
  shop prices (05 Shops, 18 Field Cards); swap out for 2 points, 1 less for WIND (08, 20 Creature Types); points on
  death equal to the creature's stars to its owner, none for minions, +1 for GIRL (17 Death, 15 Minions, 20);
  immediate refill of an empty slot (16); status effects: standard once per creature, stacking, longest duration on
  reapply, the three upgrade chains, Frozen/Blazing/Plague blocking the lower debuff (19 Status Effects2), damage
  and ticks at the end of the turn (19 Status Effects), type immunities (20); negation and 5+ star immunity (22
  Negation); the reaction stack resolving in reverse with damage last and the original play last (12 Reactions3, 12
  Reactions4); hand items only on your turn (12 Reactions5); limits: 3 facedown items (13), one field card in play
  (18), no healing items on machine creatures, no equips or healing on active items (14 Item Types, 14 Item Types2).
- **Encoding forces rulings.** Each must be written into the rulebook (TTSCardMaker content) before the engine
  encodes it; from `docs/NEXT_SESSION_PLAN.md` "Rulebook data issues" (campaign: `docs/Campaigns/RULEBOOK_CONSISTENCY_CAMPAIGN.md`)
  and the rules skill's Known gaps:
  1. 21 Damage Types NORMAL "DMG - (1/2*DEF)" vs "ATK - DEF" (07 Stats3, 10 Attacking).
  2. 12 Reactions3 "5 stacks of hardened, giving it 3 defense" vs Hardened +0.5 per stack (6 stacks).
  3. 07 Stats3 text "-1 DEF BAR --> 1" vs its picture (1.5).
  4. Status damage and ticks at (ENDT) (19 Status Effects) vs phase 4, after (ENDT) (09 TakingYourTurn).
  5. Hidden drafts still in `rules.json`: 12 Reactions2 order 4>2>1>3 vs 4>2>3>1 (12 Reactions3); 22 Negation's
     older ITEMS block. Parser rule to add: skip hidden text layers on rulebook pages.
  6. Tags used on cards and defined nowhere: (XCD) 27 occurrences, (XGCD) 24, (XU) 8; by distinct cards (1GCD) 20,
     (2CD) 14, (1CD) 13, (2U) 6, (2GCD) 2, (3U) 1, (4U) 1.
  7. Undefined words: "prime creature", "tier", "proc", "instance", "cleanse", "innate actions", "N.M.E.".
  8. Silent: which creature's star level breaks a lane tie (08); whether combo damage applies to sneak attacks or
     ability DMG and how "25% +1" rounds (10); "one ACTION" (09) vs "2 innate actions" (11 Abilities7) and Inspired
     (+1 Action); whether creatures still in hand must be defeated to win (01 Getting Started, 15 Minions); hand
     limits and summoning from hand beyond refilling a slot (16); starting BRICKS; whether (OFS)/(OS)/(OD) can be
     reacted to (12 Reactions5 names only time based effects); unspent star points (implied lost).
- **Data.** Layer 2's plus, per card, nothing new: abilities stay text the players apply by hand through Layer 2's
  counters; the engine applies what it can and logs "ability: by hand".
- **You have a result when** the engine replays the rulebook's examples byte-for-byte in the log (08 Global Turn
  order, 10 Attacking's two hits, 12 Reactions' Pebble/Hardened and 12 Reactions3's 4>2>3>1 chain, 07 Stats5's 120
  rounding cells, 07 Stats2's Mushroom 16 -> 19.5) and every item in the list above has a page to cite.
- **Size.** 2 to 4 weeks of code; the rulings are the user's and gate it.

### Layer 4. Ability scripts (`site/assets/scripts.js`; field `script` in `site/Card Edits.json`)

- **Ships.** A small language for the common patterns, interpreted by `scripts.js` on top of `engine.js`. Pattern
  counts over the 432 legal cards (case-insensitive regex on the ability text, cards matched): `attack` 193, `inflict`
  159, `damage|\bDMG\b` 114, stat change `[+-]\s?\d+(\.5)?%?\s*(HP|DEF|ATK|SPD|MAX HP)` 96, `\bpoints?\b` 69, `heal` 67,
  `summon` 52, `equip` 49, `\bdies?\b|death|\(OD\)` 42, `negat` 28, `\(action\)` 25, `redirect|intercept` 16,
  `\bdraws?\b` 15, `random` 11, `swap` 7, `\bmin(e|ing)\b|brick` 4. Statements
  like `(1PT) inflict 2T1 Poison target:enemy`, `(WOF) friendly +1 DEF`, `(OD) summon "Mini Bee"`, with the tags of
  02 Abbreviations and the `XTY` syntax of 19 Status Effects2; anything else stays `manual`. Stored as a
  **database-only edit** `script` (nullable string) beside `note`, `aliases`, `subseries` in `EDIT_FIELDS` and
  `EDIT_NULLABLE` of `site/build/build_site.py`, so the editor's database-only group, `Card Edits.json`, the bot's
  `CardEdits` and the PSD pipeline (which must never bake it) all treat it the same way. Coverage is measured
  (`scripted / legal`, and weighted by plays from §4 reports); most-played cards first; AI-drafted scripts, every one
  reviewed by a human in the editor before publish.
- **You have a result when** 90% of the legal pool and 100% of the 50 most played cards run with no `manual` step
  in a full game, and a game of two fully scripted hands plays end to end with hand input only for choices.
- **Size.** Language and interpreter 2 weeks; scripts 3 to 6 weeks of review at 20 to 40 cards a day.

### Layer 5. Batch simulation (`site/sim/batch.mjs`, run by hand with node; results committed under `site/sim/`)

- **Ships.** The same `engine.js` and `scripts.js` headless; policies (random legal, greedy damage, scripted
  openings); N games per hand pair; win rates, average global turns, per-card death and play rates; the build turns
  `site/sim/*.json` into a Stats section. **Distance, honestly:** reactions, facedown items and 17-card hands make a
  good policy a search problem; results mean nothing for a hand with any `manual` card, and nothing about balance
  until policies beat a random one by a wide margin. Treat Layer 5 as research with the methodology skill's bar.
- **You have a result when** 1,000 games between two fully scripted hands run in under a minute and a human
  reviewer finds no rule error in a sample of 20 logs.
- **Size.** 2 weeks for the runner and dumb policies; good policies open-ended.

## 4. Game report

The schema's home is `docs/Campaigns/GAME_REPORTS_CAMPAIGN.md` §4 (its decision D4: one schema for Tabletop Simulator
and the browser); this document keeps no copy. Files `site/games/<id>.json`, read by `build_site.py` into
`dist/data/games.json` (its P4): card pages get plays, deaths and win rate when in a winning hand; Stats gets most
played and deadliest; profiles get wins and losses.

**The browser fills** `source: browser`, `players[].color: null`, `started`/`ended`, the counters at the end, and every
`cards[]` count exactly from Layer 2's event list (`log[]`); the Play tab validates the file and commits it with the
`edit.js` sign-in. That sign-in is a GitHub token with `public_repo` scope and push permission on the repo, so only
collaborators can file or confirm from the browser; a non-collaborator login is a free name. Everyone else files
through the bot: `/report_game` (the home's P3), pasted text or a `.json` attachment (Discord caps string options at
6000 characters; a report with a 17-card hand per team and a `cards[]` row per minion and item approaches that).

**The Lua reporter** (the home's P1 and P2) needs scripting zones the save does not have today (0 `ScriptingTrigger`,
16 hand zones, 43 counters; tiles nicknamed Creature Holder x6, Minion Holder x2, Item Holder x6, Field Item Holder
x2, Graveyard x2, Discard Pile x1; the "Active item holder" of 04 Table4 is 2 tokens nicknamed "Active Holder"). A
tile object cannot list the cards resting on it, so zones go over the 6 slots, both minion holders, both active
holders, the graveyards, the discard pile, the field card slots and the shops, and `zone.getObjects()` gives every
card's end position. Graveyard and discard zones give `died` (17 Death) and `used` (13 Items "discarded after
resolving"; equips, actives and deploys are not discarded, 14 Item Types2, and a negated item returns where it was, 22
Negation, 12 Reactions5: both undercount, said in the file). `played` is exact only from `onObjectEnterZone` on the
slot and minion zones (a swap to hand and a second summon re-enter them); without that log the reporter writes
`played: 1` for every card outside the hand at the end and says so (the home's snapshot limit), never a guess. Lua
cannot write files: the JSON goes to a Notebook tab and the host copies it out. The reporter suggests the winner from
"Defeat all of your opponent's creatures" (01 Getting Started) and a lone persistent minion (15 Minions); creatures
still in hand are a silent point, so the host confirms (the home's D3).

**Naming proposals for the home** (to settle under its open question 6): `teams[].field` means the seeded field card
(ban list: "active in the shop at the start of the game", not in play) while `cards[].end: "field"` means on the
primary field, and `end` is also the report-level result; proposed `teams[].field_card`, `cards[].end: "primary"`,
report-level `result`. `seat` is not a game word (03 Setting Up says team; the save says Team 1 and Team 2): proposed
`team`.

## 5. Order, gates, reuse, open questions

**Order.** Layer 1 -> Layer 2 hot-seat -> report schema (§4) and the Lua reporter -> Layer 2 play-by-link -> rulings
(Layer 3's list) -> Layer 3 -> Layer 4 -> Layer 5. **Gates** (take the branch, do not rationalize): G1 Layer 1's
numbers match hand calculations on one real hand, else fix before any UI work. G2 the rulebook's worked turn replays
in the hot-seat table, else the zones or counters are wrong. G3 every ruling in Layer 3's list is in the rulebook
before `engine.js` encodes it; an unruled item stays `manual`. G4 Layer 4 reaches its coverage line before any batch
run. G5 Layer 5 only on fully scripted hands.

**Reusable from the deck builder and the site.** `deck.js`: `HAND`, `FIELD_MAX_STARS`, `validate`, `encode`/`decode`
(the `d/` share code, same scheme for `g/`), `loadDecks`/`saveDecks` (localStorage), `toText`/`fromText`, `randomFill`
(random legal hands for empty-slot practice and batch), `ttsSave` (export a sandbox position as a Tabletop Simulator
saved object). `search.js`: `compile` for the card picker, `total`. `text.js`: `tagInfo`, `badgeHtml`, `makeRenderer`
(ability text with icons in the table). `stats.js`: exports only `renderStats`; its chart helpers (`hbar`, `columns`,
`stacked`, `scatter`, `table`, `card`) are module-private and must be exported first. `edit.js`: `createEditor` sign-in
and batched commits for publishing reports and scripts (collaborators only, §4). `build_site.py`: `parse_banlist`,
`slugify`, `EDIT_FIELDS`. `app.js`: routes and tabs (`deck` -> add `play`).

**Known wrong paths (fenced off).**
- A backend, accounts or a hosted relay: contradicts the non-goals; GitHub Pages and repo files only.
- Reading effects out of the free text with regexes "for now": free text is not a schema (the 2026-02-08 whitespace
  saga is the cost of treating text as structure); Layer 4 scripts are explicit and reviewed.
- Settling a contradiction in code: the rulebook is the authority; `engine.js` cites, never decides.
- Inventing draw or mulligan rules for Layer 1: the rulebook has none (03 Setting Up, 13 Items).
- Keeping game state in the TTS save: the save is the published table (tts-save.yml rewrites it); reports are files.
- Letting the reporter alone declare a winner while creatures remain in hand: the rulebook is silent.
- Naming the tab or a module "Table": that name belongs to the reverted viewer (`table.js`, `tts.js`).
- Batch results from hands with `manual` cards: not evidence, by G5.

**Open questions for the user.**
1. Who writes the rulings for Layer 3's list and in what order; do unruled items stay "by hand" indefinitely?
2. Peer play without a server: paste-a-code signalling (clumsy, no server) or a free public signalling service (is
   that "a server")? Or stop at play-by-link?
3. Reports: GitHub logins only, or free names too (free names cannot be claimed by a profile)? Only collaborators can
   file or confirm from the browser (§4): may an admin confirm on a player's behalf, and does the leaderboard count a
   report confirmed by one team? Private reports encrypted like private decks, or all public?
4. The Lua reporter: inside the published save (`tts_save.py` keeps it) or a separate object players spawn? (The
   home's open question 2.)
5. Constructed's shop pool: the whole level, or minus both hands?
6. The `script` field in `Card Edits.json` (one JSON, like every other edit) or one file per card under
   `site/scripts/`? The former matches the pipeline's two-kinds-of-edits rule; the latter diffs better.
7. Minions and `Auxiliary/Creatures` cards in reports: per-card rows (they go to the graveyard, 15 Minions) or
   folded into their parent? (The home's open question 5; one answer for both.)
