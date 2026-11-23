# Accepted ideas, session 2026-11-22

Accepted = an independent verifier reproduced the gain and found no regression. Sources: the lanes' accepted.tsv,
the ship records and the LANDED lines (research5's own accepted lists were wrong; see README, Incidents).

Ships:

| Ship | Landed | Heads |
|---|---|---|
| actbake s1 | 11-22 15:26 | AC f279dbe1, TC cbd4bb1d |
| actbake s2 | 11-22 22:46 | AC e28a8944, TC 9f808c28 |
| actbake s3 | 11-23 01:12 | AC fdc718af, TC 2838c3e8 |
| cards s1 | 11-22 13:24 | AC 8523b665, TC 1f230bea |
| cards s2 | 11-22 23:50 | AC 43ebb2c0, TC 6903dba6 |
| cards s3 | 11-23 00:36 | AC 5553f8da, TC f24c52e0 (save 7858e49d) |
| fork release | 11-22 20:42 | fork e5915747 (`cards-20261122`), AC 5e489de4 |
| fork Reduce Noise (owner 20) | 11-23 04:01 | fork baa9a901 (`cards-20261123`), AC 541feb9c |

## actbake: 18

| Round | Idea | Ship |
|---|---|---|
| r1-0 | Fork re-pin: one Card checks run builds the runners' card venv once (docs) | s1 |
| r1-1 | Stars: after an option-B shrink, a later stars edit starts from the maker's stars | s1 |
| r3-0 | Bake: a stars or types edit on Beholster, Pandora, P.A.N.D.A. Mk F leaves no ghost of the old icon | s1 |
| r3-1 | Mock-up: a stars edit on the 7 PSDs with several pictures (15 card pages) shows the bake's own stars | s1 |
| r4-0 | Card guards: the standards post and Follow up go out from the check job (post/tell jobs gone) | s1 |
| r4-1 | Mock-up: see-through, centred and gradient Strokes drawn as the bake does (~50 layers, 35 cards; VERSION 15) | s1 |
| r4-2 | Bake: box, split and moved titles shrink to clear the stars and type icons (26 cards) | s1 |
| r4-3 | Text colours: a whole-title colour bakes on Bulborb Larva and Empress Bulblax | s1 |
| r5-0 | Rulebook PDF: the push's Pages zip is asked again after a 404 | s2 |
| r5-1 | Warm lane: a preview press shows ~0.6-0.8 s sooner | s2 |
| r5-3 | Text colours: a whole colour reaches Tainted Eden's title and SOMETHING's Ability (offset copies too) | s2 |
| r6-0 | Mock-up: titles and Abilities under their own Gradient Overlay keep their colours (no dark tops) | s2 |
| r6-1 | Rulebook PDF: a push that changes the TTS pages' text skips the 45-screenshot look | s2 |
| r7-2 | Mock-up: a word or line colour on Twin Emperors' three plain Ability boxes is drawn, not refused | s2 |
| r8-0 | Text colours: Pin Light Abilities (22 items) and Michael's Blessing's Linear Dodge title take the picked colour | s2 |
| r8-1 | Text colours: a whole-Ability colour on Ice Staff prints (its 69% overlays go off) | s2 |
| r8-2 | Mock-up: a title line set over twice the title's size keeps its size (Zenith Martlet, The Evil, Fricked Up Cursed Idol) | s2 |
| x2 (retry of r7-1) | Bake and mock-up: words typed at the end of a coloured paragraph take its colour (52 paragraphs, 42 cards); line colours counted by 8-bit value | s3 |

## cards: 33

| Round | Idea | Ship |
|---|---|---|
| r1-0 | Tests: the utility clamp while the owner games, not `-b` (docs) | s1 |
| r1-1 | Tests: the two load-flaky timing tests wait on events | s1 |
| r1-3 | Tests: the guards background-jobs test cleans up in a finally | s1 |
| r2-0 | Card dialog: image versions follow renames (11 cards get 25 older pictures; Man-Serpent's Shield 1 -> 9) | s1 |
| r2-1 | TTS rulebook search: a section title lists its page first | s1 |
| r2-3 | Search: types==fire finds every card with the fire type (bot and website) | s1 |
| r3-0 | Tests: a TC test tree without TTS/checks stops the run (no silent skips) | s1 |
| r3-1 | Rulebook links on phones and tablets land below the pinned bar | s1 |
| r3-3 | TTS checks: the data checkout waits for Cards start's fetch | s1 |
| r4-0 | Card text changes shown as whole phrases (History, Changelog, Recently updated) | s1 |
| r4-1 | TTS checks: History waits for its fetch, not a fixed 3 s | s1 |
| r4-2 | Search: a type's name lists that type's cards (bot /query_ability and website) | s1 |
| r5-0 | Site: turning a phone sideways keeps your place | s2 |
| r5-1 | Rulebook search: a status duration typed as cards print it ('2T1', '1T1 Restrained') finds the [X]T[Y] rule | s2 |
| r5-2 | TTS checks: run_all's parts even again, lua_mock starts early (~41 -> 32 s) | s2 |
| r5-3 | /query_rulebook: a miss on a card's name points to the card (41 of 64 status cards) | s2 |
| r6-2 | TTS Card details: Rules buttons for Minions, Death, The Field and Attacking (126 cards) | s2 |
| r6-3 | Tests: the slowest test 30-75 s -> ~1 s (part 1 only; part 2 backed out, owner G3) | s2 |
| r7-0 | Ban list: the Constructed ban list answers to its own name (bot and website) | s2 |
| r7-1 | Search: types== and type== take the names players see (Dark Matter, Field card, Status effect...) | s2 |
| r7-3 | Tests: has() goes; mock-up and preview checks fail instead of skipping on a TC tree without the feature | s2 |
| r8-0 | Card text search: an icon's name typed as a word finds its cards ('heal a machine') | s2 |
| r8-1 | Card text search: 'attack' after a number finds ATK ('+1 attack') | s2 |
| w0 (waiting) | Rulebook search: an icon's name in words finds its lines ('defensive minion') | s3 |
| w1 (waiting) | TTS Card details: The Beholster gets The Puppet's deck buttons | s3 |
| x0 (retry of r1-2) | Editor: long titles fit the phone preview, no mid-word breaks | s3 |
| x1 (retry of r2-2) | Card dialogs: closing one goes Back without the Navigation API (one history call per step) | s3 |
| x2 (retry of r3-2) | Card history: misreads set to their pictures' values (Retro Kirby, Living Bread, Shell Shield) | s3 |
| x3 (retry of r4-3) | Parser: a new creature takes its folder's shared series (parser half only) | s3 |
| x4 (retry of r6-0) | Search: "Did you mean" for a misspelled word (bot and website), link in the site's link colour | s3 |
| x5 (retry of r6-1) | TTS Find a card: an ability word finds its cards (line breaks read as spaces) | s3 |
| x6 (retry of r7-2) | Rulebook search: an abbreviation typed alone (SP, OS, DEF) finds it as a word | s3 |
| x7 (retry of r8-2) | /query: a typed word goes first to the card with it as a whole word ('rogue' -> Warcraftian Rogue) | s3 |

## fork: 11 + dead code (`cards-20261122`), + Reduce Noise (`cards-20261123`)

| Round | Idea |
|---|---|
| dead code | Owner G8 deletions (20 fork + 3 AC commits); the 19-job CI kept (owner 15; 2 jobs since 11-23, owner 25) |
| r1-0 | Smart filters: Gaussian and Motion Blur; a painted mask no longer refuses |
| r1-1 | Card sync redraws Puppet Warp, Invert and all-off filters |
| r2-0 | Smart filters: Hue/Saturation, Black & White, Selective Color |
| r2-1 | A warpCustom picture redrawn through its Bezier patch |
| r3-1 | Motion Blur at any distance (1-256 px) |
| r5-0 | Mosaic (split out of r3-0) |
| r6-0 | Perspective Warp on an upright quad; a trim refused |
| r7-0 | Perspective Warp on a turned quad, placed as Photoshop does |
| r7-1 | A warpCustom from a new-size file as Photoshop's Update Modified Content |
| r8-0 | Card sync takes a trimmed or padded file under a Puppet or Perspective Warp |
| r8-1 | A turned Perspective Warp updated to a new-size file |
| owner 20 | Reduce Noise at Strength 10: Burly Bomb, the last picture Card sync couldn't redraw (unredrawable 1 -> 0) |
