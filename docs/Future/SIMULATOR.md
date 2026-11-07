# Simulator: against the AI or by the numbers

**Status:** designed 2026-11-07, not scheduled. Nothing is built: `site/assets/` has no `odds.js`, `engine.js`, `scripts.js`
or `play.js`, and `site/sim/` and `site/scripts/` do not exist. Numbers are from TTSCardMaker main `60c53f0d`, data
`d9ecf01d`. `(08)` is a rulebook chapter. "Legal" means allowed in a constructed hand. Scripts behind the numbers:
`~/Documents/Tools/wt/scratch/s1107/sim/`.

**What changes from `docs/Campaigns/SIMULATOR_CAMPAIGN.md`.** Its rule-by-rule lists still stand; where the two differ, this
doc wins.
- Its non-goal "No AI play" is reversed (owner, 11-07).
- Its hot-seat, play-by-link and real-time play (Layer 2) are out of scope.
- Its report layer (§4: report files, the Lua reporter, `/report_game`) is out of scope.

**Goal.** Say how strong a hand or a card is: closed formulas first, then many games between hands with the AI on both
sides, and one person testing a hand against the AI in the browser.

**Owner's rulings**
- Runs only against AI or mathematically, never player against player (11-07).
- The team writes the rulings; a contradiction is settled in the rulebook, never in code (10-24; campaign non-goals).
- Constructed's shop is the whole level, cards in either hand included (10-24).
- One script file per card under `site/scripts/`, folded into `site/cards/` when the web editor's migration makes it (10-24).
- Also kept from the campaign: no server, and not a replacement for Tabletop Simulator.

## Why

The only test of a hand today is a real game, and the website's balance numbers only rank stat lines. Their `total`
(`search.js:12`, `analytics.py`) adds bars, though a bar is 2 HP, 0.5 DEF, 1 ATK or 1 SPD (07).

Stat lines also say little about strength. A trial: each of the 271 legal creatures fought every other of its star level,
with no abilities, by the arithmetic of (07) and (10). Its score against players' `power` ratings (`site/Card Ratings.json`):

| Compared with `power` | Creatures | Rank correlation |
|---|---|---|
| Duel score | 234 rated | -0.08 |
| `total` | 234 rated | -0.05 |
| Duel score | 43 unchanged since their rating | -0.20 |

Creatures rated 5 or lower even duel slightly better than those rated 7 or higher (+0.09 against -0.14 standard deviations).
Sans is 5★ at 1/1/1/1 and loses every duel: its strength is its ability text. The ratings are coarse (155 of the 234 are a 6)
and date from spring 2025, so this proves nothing about one card. It does mean that a simulator without abilities says
nothing about balance. Abilities are the point and the cost.

## How it fits what exists

| Exists | Use |
|---|---|
| `site/rulebook/` | The authority; every rule in the engine cites its chapter |
| `cards.json`, `db/stats.json` | Stars, stats in bars, types, item types; `creatures` holds each creature of a card with several |
| `deck.js` | `HAND`, `validate`, `randomFill`; `randomFill` uses `Math.random` (line 121) and needs an rng argument |
| `text.js` `tagInfo` | Already reads every timing tag exactly, so timing is not scripted |
| `analytics.py`, `balance.js` `balanceBlock` | Where results show; `data/analytics.json` is the pattern for `data/sim.json` |
| `site/Card Ratings.json` | 569 power ratings on 383 cards, all from TTS sessions: the check on results |
| `TTS/decks.json`, Baseline Creatures | 47 cards "considered balanced" (04), 44 legal: the yardstick |
| `game.txt` | The whole game as plain text: what an AI session reads to draft scripts |

Code goes in TTSCardMaker `site/`. The browser needs it, and the site's modules run in node with no DOM (`deck.js`,
`search.js`, `text.js`, `stats.js` and `balance.js` all import headless, tested; `site/build/check_first.mjs` already does
this), so one engine serves both. AutoCroissant and the bot do not change.

## Design

### The numbers (no engine)

- **Star point budget.** Every legal hand costs the same. Its items add to 25 stars (`deck.js:4`) at 7 SP a global turn (08):
  4 global turns at the least (packed 7, 7, 7, 4), and the two 4★ items never share one. Its creatures add to 24 stars, so
  the points a team can gain from defeats are fixed too (17). These are constants, shown once; the campaign's "fewest global
  turns" is not a per-hand number. What differs per hand is the 18 legal cards that name SP and the 54 with a cost badge.
- **Attack arithmetic.** Attacks to defeat, for any two stat lines: ATK - DEF, x1.5 or x2 for speed, round up to 0.5, a sneak
  attack when that is better, DEF doubled in defense mode (09, 10). Over the 73,170 pairs of legal creatures: median 4 attacks
  (2 to 6 from the 10th to the 90th percentile), sneak attack better in 14.2%, combo damage in 7.6%.
- **Unchanged from the campaign's Layer 1:** shop odds, the BRICKS plan, the turn-order preview.
- **Win rates over many games.** These need the engine, but how many games is arithmetic. ±3 points at 95% takes 1,068 games,
  ±1 takes 9,604. With random legal hands the slowest pool is the 49 5★ creatures, one per team: about 26,000 games give every
  legal card ±3. At the campaign's 1,000 games a minute (a target, not measured) that is under half an hour.

### The engine

`site/assets/engine.js`: the structural rules of the campaign's Layer 3 as plain functions, `legalMoves(state)` and
`apply(state, move)`. Every choice is a move, including each reaction window a play opens (12): 56 legal cards carry (AAT) or
a global cooldown, and every facedown item can react. One seeded generator, so a seed replays a game. A stats-only mode (no
abilities, no items) tests the engine and the AI; its results are never balance data.

### The AI opponent

"AI" here is a plain game-playing program in JavaScript: no model, no torch, no API key. That suits a static site with no
server and keeps to the AI boundary. It loads by `import()` on first use, like the editor and Stats (`app.js:118`), so the rest
of the site pays nothing.

1. **Random:** legal moves at random. Finds engine bugs; the floor the others must beat.
2. **Greedy:** scores each move with a hand-written evaluation (HP and stars on the field, points, charge). Fast; plays the
   batch games.
3. **Search:** for each move, samples the hidden cards, plays the game out with greedy, takes the move that wins most. Slow, so
   it runs in a worker with a time limit per move: the opponent for a person, and a spot check on greedy's results.

The AI's view of the game leaves out the opponent's hand and facedown items (03); it never reads them. The campaign's warning
stands: nothing here means anything about balance until the AI beats random by a wide margin.

### Abilities

"About 430 scripts" holds for starting hands only. A full game reaches more:

| Pool | Cards with text | Running total |
|---|---|---|
| Legal hands (271 creatures, 162 items) | 431 | 431 |
| Status effect cards | 64 | 495 |
| Minions, auxiliary items and creatures those hands name (by name, one level deep) | 97 | 592 |
| Shops: field cards 49, N.M.E. items 26, banned items 20 | 95 | 687 |

The campaign's "small language for the common patterns" also covers less than it hoped. The legal pool has 1,074 ability
lines. With numbers, status effects, types, tags, card names and filler words taken out, 789 are still different and 725 of
those appear once. The 100 commonest cover 35% of lines, and only 73 of 431 cards have every line shared with another card.
Two wordings can mean one effect, so the real tail is shorter, but most cards are one-offs. So:

- **Timing comes from `tagInfo`,** not scripts. Only the effect is encoded.
- **Verbs for what is shared** (damage, heal, stat change, inflict, summon, equip, negate, redirect, gain points), with targets
  and conditions, as data.
- **`custom`, a small named function, for what verbs cannot say.** This replaces the campaign's `manual`: "by hand" needs a
  second person, and there is none. A card with no script is not in the pool, and the hand picker shows only scripted cards.
- **Every choice in a script is a move,** or the AI cannot play the card.
- **A script stores a hash of its card's ability text.** Changed text makes the script stale and takes the card out of the pool
  until it is checked. This will happen often: legal cards took 86 wording changes in October and November with the same parser
  on both sides ("dies" to "is defeated", "Equip" to "Auto-equip"). The build lists stale scripts and scripts of renamed cards.
- **Free text is never read at run time** (the campaign's fenced wrong path). Scripts are drafted by AI from `game.txt`, then
  checked (open question 1).

**Order.** "Most played first" has no source: reports are out of scope and `site/decks/` holds no published deck. Instead:
1. The 64 status effects: 150 legal cards name one.
2. A pilot pool of about 100: the 44 legal Baseline creatures, plus enough 5★ and 2★ creatures (Baseline has only 3★ and 4★)
   and items for legal hands, and the minions they name. Shops off, said on every result.
3. The rest of the legal pool, the extremes of `power` first.
4. Field cards, N.M.E. items and banned items, when shops go on.

### Rules to settle first

From `impossibility-simulator-rules` Known gaps, checked against the chapters at `60c53f0d`. Of the campaign's eight Layer 3
rulings, 1 to 6 are settled; 7 and 8 are partly open, below. The 10-24 answer "unruled items stay by hand" cannot carry over,
so an unruled item is either a switch or keeps its cards out of the pool.

**To rule first.** No honest default exists:

| Gap | Where | Reach |
|---|---|---|
| Pair turns with uneven slots | 08, 16 | Nearly every game: a team down to its last creatures has fewer slots. 6 cards count "pair turns" |
| "One ACTION" against "2 innate actions" and +1 Action | 09, 11 | 17 legal cards state or change the count, beside Inspired and Restrained |
| Health lost and star level of a slot with several creatures | 08 | Turn-order ties once minions are out; 55 legal cards summon |
| Reacting to (OFS), (OS), (OD) | 12 | 60 legal cards; decides where reaction windows open |
| Must creatures in hand be defeated to win? | 01, 15 | Narrow, as (16) forces a refill, but the engine needs one end test |
| Hand limits; summoning from hand except into an empty slot | 16 | Bounds the legal moves |
| "Cleanse", "Auto-equip" | cards only | 10 legal cards each, no definition in any chapter |
| `Auxiliary/Creatures/` cards | none | 7 cards with no rule; legal abilities name 5 |
| Lost Kin "always go first in its lane" | card text | The last card counting lanes as turns |

Known gaps still lists seven lane cards; the other six now say "pair turn".

**Can run as switches.** Each is a named option, and a result carries the switches it ran under, so no ruling sits in code:

| Assumed | Basis | Reach |
|---|---|---|
| ATK - DEF stops at 0 | Unwritten; (10) implies it | 3.5% of pairs |
| Twice and triple speed mean at least that | A reading, not chapter text | 3.6% of pairs |
| No combo damage on sneak attacks or ability DMG | Silent (10) | Every sneak attack |
| Starting BRICKS are 0 | Silent; the TTS save starts at 0 | Field card timing |
| Unspent SP are lost | (08) "set to 7" | Item timing |

### Balance data it would feed

`site/sim/batch.mjs`, run by hand on the Mac (at most 4 workers), writes `site/sim/*.json`; the build makes `data/sim.json`,
loaded on first use like `data/analytics.json`. Each file records commit, AI level, switches, pool, games and seed.

- **Per card:** win rate with it in hand, with ± and games played; its lift over its star level; play and defeated rates.
- **What its ability is worth:** the same card with its script off. The website has nothing like this today.
- **Beside `total` in `balanceBlock`:** attacks to defeat, given and taken.
- **Per pool:** first-move value from mirror games, game length in global turns, NO STALLING damage (08).

What it is for: a review list of cards where results and `power` disagree; a banned card's win rate if let in (power only: 9 bans
are for copying and 9 name "antifun", which no win rate measures); stars checked against results; and a what-if, running a stat or
text change before it is pushed. Nothing changes by itself: a card, a ban or a rating moves only by a person. Results stay
labelled simulated, apart from ratings.

## Stages

Each is useful alone. Sizes are the campaign's where it has one.

| # | Ships | You have a result when | Size |
|---|---|---|---|
| 1 | The numbers: `odds.js` | One real hand matches a hand calculation from 03, 05, 07, 08, 09, 10, 13, 18 | 2 to 4 days |
| 2 | The rulings above, in the rulebook | Each has a chapter to cite | The team; pair turns gate 3, the rest gate 5 |
| 3 | Engine, seeded, stats-only games | The rulebook's examples replay exactly (Verification) | 2 to 4 weeks |
| 4 | Random and greedy AI, `batch.mjs` | Greedy beats random in 9 games of 10 | 2 weeks |
| 5 | Verbs, `custom`, status effects, pilot pool | Two pilot hands play out with no skipped line; a person finds no rule error in 20 logs | 2 weeks, then 4 to 8 days of checking |
| 6 | Win rates: `data/sim.json`, Balance section | 1,000 games in under a minute; Baseline creatures land mid-star-level | Days |
| 7 | Search AI and the Play tab: one person against it, the campaign's zones without hot-seat | A person finishes a game and can read the log | 1 to 2 weeks; good AI open-ended |
| 8 | Rest of the pool, then shops | 90% of the legal pool scripted | 13 to 26 days of checking |

The campaign built the table before the rules, for two people to use. With no second person a table is no use until the engine
and the AI exist, so it comes late.

## Verification

- **The rulebook's examples, exactly, in the log:** C > B > A > D > E > F (08); the hits for 6 and for 9 (10); 4>2>3>1 and
  5>4>2>3>1 (12); the rounding table and Mushroom 16 to 19.5 (07).
- **Same seed, same log.** A hand against itself lands within ± of 50%, apart from first-move value.
- **Evidence only from a clean run:** a run counts only if no card in it has a stale script or a skipped line.
- **Data errors show as results.** Monkey D Letion reads 10/10/10/10 at `d9ecf01d` (should be 8/1/7/6) and tops its star level
  in the duels. A card with a picture-against-data flag stays out of any claim.
- **Check against people:** Baseline creatures should land mid-star-level, and results should rank with `power` above the
  stat-only -0.08. Failing that sends a person to read logs; it does not fail the run, since the ratings are coarse and old.
- **Greedy against search:** on sampled hand pairs the two should rank cards alike. If they do not, greedy's numbers are about
  greedy.

## Open questions

1. Are scripts checked by a person every time (the campaign's rule), or only where two independent AI drafts disagree? At 20
   to 40 cards a day, 592 scripts are 15 to 30 days of checking. This is the largest cost in the plan.
2. Does "the whole level" put banned items in the shop? The ban list only restricts hands. It decides 20 scripts.
3. Do results show on public card pages, or only in the Stats section? A win rate on a card page reads like a verdict.
