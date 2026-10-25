# game.txt: the whole game in one text file

**Status:** built and landed 2026-10-25 (TTSCardMaker 56651852: fixed fields with a legend, hidden originals handled; linked from the Rulebook).

**Goal.** One plain-text file with every card and every rulebook page, small enough (about 50k tokens) to hand to any
AI or read in one go. No AI code in either repo.

**Rulings that bind it:** under TTS/ or site/, either (owner; `site/` chosen below); private Discord archive numbers
never appear (owner, 10-23); game words only (creatures, minions, items, field cards, N.M.E. items, status effects,
stars, types, abilities).

## Design

- **Built by** `site/build/build_site.py` (TTSCardMaker) from what it already loads: the cards and the rulebook pages
  from the `data` branch, `site/Constructed Ban List.txt`. About 60 lines; no new inputs.
- **Written to** `site/dist/game.txt`, served at `<site>/game.txt` with each deploy. Nothing is committed, so the file
  costs the repo nothing and is never stale against the site. A copy committed under `TTS/` only if the owner asks.
- **Layout** (UTF-8, `\n`):
  1. a header: build date, card count, the data commit;
  2. the rulebook pages in book order: `## <page name>` then its text as the site shows it (tokens as words:
     `[fire]` -> `fire`, badges as their tag, e.g. `(1CD)`);
  3. the timing tags with the meanings the site shows (`site/assets/text.js` `tagInfo`, ported), the ban list, and the
     team's confirmed rulings once they exist (the rulings draft, ideas sweep item 9; unconfirmed rows never);
  4. one block per card, grouped by kind then series, sorted by name:
     `<name> | <kind> | <stars>★ | <types> | HP <hp> DEF <def> ATK <atk> SPD <spd> | <series>` then the ability on the
     next lines; every field kept, `-` when empty, and a one-line legend in the header; hidden and archived cards left
     out; variants as their own cards.
- Stats as the data holds them (bars, as printed). The sweep's game units (HP x2, DEF x0.5) are left out, so the file
  never restates a rule; the rulebook pages in it explain the bars.

## Verification (lean)

Build once locally; check the size (target under 60k tokens, measured with a word count x 1.3), that every card and
page appears once, and that no `[token]` or HTML is left. No test file.
