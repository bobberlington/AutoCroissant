# Session 2026-11-01: the Discord rulebook batch

The owner asked for every pending rulebook change in Discord `pending-updates` to land in one push on the website, then be forwarded to `finished-updates` and deleted. Procedure for next time: memory `discord-pending-updates`.

## Landed

TTSCardMaker:
- `c3eb51a6`: the 32 📖 messages in `site/rulebook/`: pair turns replace lane turns (a lane is still a place), swapping out cannot change turn order, death vs defeat (ch 17), "destroyed" = true damage equal to current HP, prime creature and base stats defined, charge rules for (XCD) (XGCD) (Y/XCD) (Y/XGCD) (XU), multi-hit is one attack, active items in a slot on the primary field, 0 ATK cannot attack, minion sneak attacks, targeted-for-negation rule, Max HP decrease, field cards are not items, actions only (OYT), Toy Chica + Chef Saltbaker, ban list "field card", game.txt timing tags.
- `2fecc66d`, `c6b1e144`: TTS save tiles and decks.json labels say field card; drift baseline 2fecc66d. Checks 24 of 24.
- `ae6e4dac`: minions follow their parent creature, equips are separate from innate abilities.
- `b110781c`: (XU) under its own USES group; sneak attack "25% of their attack stat, plus 1".
- `87797b2b`: (Y/XCD) and (Y/XGCD) badges; ch 02 reorganized into two-column rows.
- `b2ae3c02`: ch 15's offensive minions follow the parent creature (found by the wrap-up check). Card sync forced by the owner: PDF `09357e06`, save `bf67cd8e`.

AutoCroissant (skills and docs only, no deploy): `abd040b`, `78a5d6f`, `396d2d2` and the wrap-up commit. `impossibility-simulator-rules` now cites `site/rulebook`; its verbatim Photoshop extract and build scripts were deleted.

Discord: 32 messages forwarded by the production bot and deleted. 8 card messages stay in pending-updates.

## Owner rulings

- Damage dealt when something dies IS counter damage (the poll's result was lost).
- Minions are tied to the creature they are on (the parent creature), not their summoner, who can occasionally summon onto another creature. A defeated parent's minions die unless persistent.
- A creature's innate abilities and its equip's are separate (Sword of Aeons grants lifesteal; the creature has none innately). Negating a creature does not negate its equipped items.
- TTS save says field cards too. (XU) moved; sneak attack's +1 is added after the 25%.

## Verified

Live site data after each push; `tests/test_rulebook.py`, `test_site_build.py`, `test_token_picker.py`, `test_changelog.py` pass; TTS checks 24 of 24 at c6b1e144; local render of the TTS pages shows no overflow; nothing has been played in TTS itself.

## Left open

Cards counting lanes as turns, uneven slots in pair turns: `docs/NEXT_SESSION_PLAN.md`.
