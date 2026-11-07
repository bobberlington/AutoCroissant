# TTS card forms

**Status:** idea from the owner, 2026-11-07. Stage 0 passed in TTS (owner, 2026-11-07): a three-state Pix switched with 1-3 on the
table and in a hand, and kept its states through a deck, a bag, a counter and moves. The test save had no real Card details; the
owner wants Card details per form (Stage 3).

## Goal

A card with several forms is one object in Tabletop Simulator that shows one form at a time. A player hovers it and presses
1, 2 or 3 to switch. Card details shows the whole card and every form together.

The owner's example is Pix: each of its 3 forms becomes its own card picture, with the forms switched on the table. The table
then shows which form is active, which players have to remember today.

## Which cards

Forms the card switches between, one showing at a time:

| Card | Forms | How the card switches |
|---|---|---|
| Pix | Green, Blue, Red | "(OS)/(BOGT) Assume a form." |
| Computer Virus | Slime, Puppet, Magician | "switches between on death" |
| Tainted Lazarus | 2 creatures | "Flip to the other creature on this card" |

Twin Emperors is not one: its ability says it is "2 entities that share HP", so both are in play at once. It stays one picture.

## How TTS does it

- **States:** any object can hold several states, and any player can pick which one shows. In the save, an object lists its
  other states under `States`. No object in our save uses states today.
- **Number keys:** the knowledge base describes number keys switching states; a Steam forum post says this works while hovering.
  On a plain card a number key draws instead.
- **Scripted control:** `onObjectNumberTyped(object, player_color, number, alt)` fires when a player types a number over an object.
  Returning `true` stops the default action. `setState` switches. Together they give exact behaviour for these cards alone and
  leave every other card as it is.
- **Switching swaps the object:** `onObjectStateChange(object, old_state_guid)` hands over the new one, so its GUID changes.

Sources: <https://kb.tabletopsimulator.com/host-guides/creating-states/>, <https://api.tabletopsimulator.com/object/>,
<https://api.tabletopsimulator.com/events/>, <https://steamcommunity.com/app/286160/discussions/0/1736633196169226801>.

## Stage 0: prove it in TTS

The docs say nothing about states on cards in decks or hands. Make a small test save with a three-state Pix, then check:

1. Does a stated card keep its states when put in a deck, shuffled and drawn?
2. Does it keep them when it goes into a bag and comes back?
3. Do number keys switch it on the table, and from a hand?
4. Does a counter resting on the card stay when the form changes?
5. Does the card keep its size, rotation and place on the table?
6. Do `Nickname` and the right-click menu behave on each state?

Everything below assumes 1 and 3 pass. If 1 fails, see Fallback.

## Stage 1: a picture per form

Each form needs its own PNG. The variant pictures are the natural fit: one PSD gives several PNGs (`psd_parser.variants_of`,
`card_data._variant_duplicates`), and a variant counts once, as "duplicate of" picture 1. The combined picture stays the card's
main picture on the website and in Card details.

Each form's picture needs its own layers shown in the PSD, which is a job for the card's maker or a one-off pass on these
three cards. The 2 and 3 creature templates (session 2026-11-07) give each creature its own bars, so stats per form already
have a home.

## Stage 2: the save

- **`commands/tts_save.py`:** `_walk` follows `ContainedObjects` only, so link pinning, names and deck rules never see a state.
  It must follow `States` too. `_sync_decks` must keep a stated card whole, counted as one card.
- **`TTS/Global.lua`:** `onObjectNumberTyped` for cards with forms; `onObjectStateChange` calls `addMenu` on the new object,
  because `menued` is keyed by object and the old one is gone.
- **Spawning:** `like()` and the two `spawnObjectData` paths ("+ Name" and the deck buttons) must carry `States`, so a spawned
  Pix can switch too.
- **Finding a card:** "Find a card" and every lookup keyed by PNG path must treat all of a card's forms as that card.

## Stage 3: Card details per form (owner, 2026-11-07)

The panel follows the form that shows: its picture, name, stats and its own ability lines, the lines every form shares, and a small
button per other form that switches the panel (not the card) to that form.

- **Data:** each form needs its own entry. A variant picture per form gives one for free: a variant keeps its own parse (name, text,
  stats) and counts once as "duplicate of" picture 1, and `data/tts.json` is keyed by PNG path, so a form's state finds its entry.
  The shared lines are the text every form's parse has in common.
- **Global.lua:** `showDetails` reads the entry of the object's current face (its state's PNG path), and the other forms' entries
  through the card's states (`getStates()`).
- **Website:** unchanged; picture 1 stays the card's main picture.

## Fallback

If cards lose their states in a deck, keep a plain card in decks and add the states when it comes out. Global.lua already acts on
cards leaving decks and bags (`addMenu`). This is untested. If that fails too, `setCustomObject` with `reload` could swap the
card's face. Otherwise the idea stops at Card details: the combined view alone is still worth having.

## Verification

- `TTS/checks/lua_mock.py` learns `setState`, `onObjectNumberTyped` and `onObjectStateChange`.
- `TTS/checks/run_all.py` passes, with `drift.py`'s BASELINE moved for the deliberate change and `updater_idempotent` proving a
  second save update changes nothing.
- A real game in TTS, since the emulator cannot prove Stage 0's answers stay true.

## Open questions

- Does the chosen form reset when a card returns to a deck, or stay as it was?
- Should the website show the forms as well, or only the combined picture?
- Should the first state be form 1, or the combined picture, so a player sees the whole card before picking?
