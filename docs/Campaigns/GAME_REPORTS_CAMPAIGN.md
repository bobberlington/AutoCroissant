# Game reports campaign (2026-10-20)

**Status:** designed 2026-10-20, not scheduled; plan item 17 (`docs/NEXT_SESSION_PLAN.md`, "Ideas 5-23"). Rule citations are rulebook pages as
`.claude/skills/impossibility-simulator-rules/SKILL.md` names them; where the rulebook is silent this document says so. Save facts
come from a scan of `~/Desktop/TTSCardMaker/TTS/Impossibility Simulator.json` (TTS v14.2.2, 5.9 MB, 228 top-level objects, Hands
enabled) and its Lua on 2026-10-20 (hands and Lua re-checked 2026-10-22 against TTSCardMaker c9873ddf). TTS Lua API names not seen in that Lua are marked "to confirm" (against the TTS API docs).
This file is the home of the report schema (§4); `docs/Campaigns/SIMULATOR_CAMPAIGN.md` §4 (the browser Play tab) points here and
keeps no copy, and both name the bot command `/report_game`. Its naming proposals are open question 6.

**Decided (user, 2026-10-20)**

| # | Decision |
|---|---|
| D1 | Reports travel by paste into a bot command (text or a `.json` attachment), never by a webhook baked into the public save (anyone with the save could spam it). A host-typed webhook URL per session is the only acceptable one-click variant, later (P5). |
| D2 | One JSON file per game under TTSCardMaker `site/games/`, committed by the bot. No new pickle (the costliest pickle incident family, the old_stats.pkl balloons and the reverted PICKLE: autocroissant-failure-archaeology Entry 3). |
| D3 | The winner is suggested by the script from the death rule and confirmed by the host; never inferred silently. |
| D4 | One schema for Tabletop Simulator and the browser Play tab. |
| D5 | Related, not here: opt-in profiles `site/profiles/<login>.json`, a leaderboard, public decks under `site/decks/` (none of the three folders exists yet). **2026-10-21:** all three landed: opt-in profiles `site/profiles/` (TTSCardMaker 4473dd1), the People tab with its leaderboard (41b929b), decks under `site/decks/` with private ones encrypted and shown only on the owner's person page (305812c). |

## 1. Goal and non-goals

**Goal.** Card and player win/loss statistics from real games played in Tabletop Simulator: per card games/wins/deaths/uses, per
player W/L, per deck W/L, shown on the website and answered by the bot.
**Non-goals.** No rules enforcement in TTS (the reporter reads the table, it never moves a card or stops a play). No network from
the save unless the host typed a URL this session (D1, P5). No new pickle (D2). No Steam names or ids without opt-in (§5). No ruling made in code: where the rulebook is silent
the host decides and the report records that it was the host.

## 2. What the save has, what is missing

The save is rebuilt by `commands/tts_save.py` (`_update`: `_pin_links` -> `_fix_keys` -> `_sync_decks` -> `_fix_cards` ->
`_resize_targets`; `_dump` writes TTS-style JSON byte for byte) from `TTS/decks.json` (keys `about`, `generated`,
`decks` (57), `bags` (15), `objects` (56), `large_cards`; `repo_head` and the per-deck `cards` counts dropped 2026-10-22) and the
card PNGs; run by TTSCardMaker's `tts-save.yml` (once a day when the PNGs, `TTS/decks.json`, `site/Card Edits.json`, the save or
the card browser's names changed; by hand) and by `/update_tts_save`. The bot commits through the GitHub API
(`update_repo_save`: blob -> tree -> commit -> `PATCH git/refs/heads/main`, 3 tries when main moves) and, when the token can't
write (POST git/blobs 403: the jetson's case), through git objects in the history clone pushed over SSH (`_git_commit`; the path is
the hard-coded `TTS_SAVE_PATH` and its tree entry must already exist).

Has (verified in the JSON and its Lua):
- **Hands:** 16 `HandTrigger` zones. 10 seat hands (`scaleX` 20) at z = +-35: Hearts Red/Orange/Green (z < 0), Clubs
  Blue/Purple/Pink (TTS teams, set by the global Lua's `TEAMS`); 6 small ones (`scaleX` 4) over the "Item Holder" tiles at z =
  +-23.8, `FogColor` Red (z < 0) and Blue: the facedown item holders (13_Items). 03_Setting_Up says only your side sees them; a
  TTS hand zone hides its cards from every colour but its own and its teammates' (TTS knowledge base, Teams), so the whole side
  sees them (not verified with the Red or Blue seat empty). P1's gate reads these zones as they are. The global Lua sets
  `Player[color].team` on a seat change; since 2026-10-22 (TTSCardMaker c9873ddf) zones keep their saved owner (the old
  `HAND_ZONE_GUIDS` claim through `onObjectEnterZone` and a `spawnObjectJSON` respawn is gone).
- **Counters:** 43 `Counter` objects, CURRENT values (06_Reading_Cards): per team "Star Points" (7), "Bricks" (0), "Points" (2) and
  one "Global Turn" (0; 04_Table2 says it starts at 1) at x = -64; HP/DEF/ATK/SPD beside each of the 6 "Creature Holder" tiles (the
  slots, x = -13/18/49, z = +-13), beside the 2 "Minion Holder" tiles (x = 95, z = +-10) and HP/DEF beside the 2 "Active Holder" tokens.
- **Turn tracking:** 6 `BlockSquare` tokens, one per slot, cycling "Has not taken its turn" / "Is taking its turn" / "Has taken its
  turn" when flipped (own Lua, `setName`); TTS's own `Turns` are disabled. The "Global Turn" counter is the only turn number.
- **Named tiles** (`Custom_Tile`): Creature Holder x6, Item Holder x6, Minion Holder x2, Field Item Holder x2 (x = 74, z = +-5.5;
  one field card in play per team, 18_Field_Cards), Graveyard x2 (x = 86.5, z = +-5.5; "on your side", 17_Death), Discard Pile x1
  (85.2, -0.1; 13_Items), Item/Field/N.M.E. Shop (x = -43..-58), deck tiles. 49 top-level `Deck` objects (57 with the 8 in the
  "Specific Item Decks" bag): creatures and items by stars, "N.M.E. Deck" x2, "Status Effects" x3 (64), "Minions" x3 (99),
  "Auxiliary Items" x3 (60), developer/baseline/simple sets.
- **Lua:** global 48 lines since 2026-10-22 (c9873ddf; 8.3k chars with the zone claiming before): teams Hearts/Clubs and the
  large-card resize (`resizeTargets`, kept by the updater, to 3,1,3 on leaving a container); "Flex Table Control"
  (`Chinese_Checkers_Piece` bd69bd, 16k chars): a stock table-surface and table-scale panel gated by `permissionCheck`
  (`Player[color].host or .promoted`). Seen in the save's Lua and reusable. Global Lua: `onLoad`, `onPlayerChangeColor`,
  `onObjectLeaveContainer`, `getSeatedPlayers`, `Player[color].seated`/`.team`/`.steam_name`, `broadcastToAll`,
  `obj.type`/`getName`/`isDestroyed`/`setScale`, `Wait.frames`/`Wait.time`. Flex script: `onSave`/`onload`,
  `createButton`/`createInput`/`editButton`/`getInputs`, `Player[color].host`/`.promoted`, `getHandTransform(i)`/`setHandTransform`,
  `JSON.encode`/`JSON.decode`, `getObjectFromGUID`, `broadcastToAll`.
- **Notebook:** 12 `TabStates` (Rules + one per colour), all empty. `XmlUI` is TTS's default placeholder comment (no UI).

Missing: no scripting zone (`ScriptingTrigger`: none; only hand zones fire `onObjectEnterZone`), so nothing knows which card sits in
a slot, a holder, a graveyard or the discard pile; no game id, start time or player names; nothing exports anything. `decks.json` has
no section for zones or scripted objects. `_sync_decks` only ever adds cards inside decks (copies of the deck's last card with a new
6-hex `GUID`, `Nickname` = website name, `CardID`, `CustomDeck`, inherited `Transform`); no step adds a top-level object.

## 3. Phases

### P1. Zones in the save (tts_save.py)
1. `TTS/decks.json` gets `"zones"`: per zone `guid`, `name`, `team` (1, 2 or null), `kind`, `position`, `rotation`, `scale`. Per team:
   `slot` x3 (over the Creature Holder, tall enough for primary and secondary field, 16_The_Field), `facedown` x3 (over the Item
   Holders; or read the small hand zones instead, see the gate), `minions`, `active`, `graveyard`, `field`; shared: `discard`,
   `shop_items`, `shop_fields`, `shop_nme`. Names `T1 slot 1` ... `shared discard`; `GMNotes` = the entry as JSON for the Lua.
2. New step `_sync_zones(save, rules, repo, names, report)` (the siblings' signature) after `_resize_targets` (no-op when the key is absent): a zone whose `guid` is missing
   is appended to `ObjectStates` as a `ScriptingTrigger` (field set copied from a `HandTrigger`, `Locked: true`; the exact field set
   of a `ScriptingTrigger` is to confirm against a save that has one); present ones get `Transform`/`GMNotes` refreshed. Report "Zones".
3. Tests beside `tests/test_tts_save.py`: a second run changes nothing; `test_serializer_round_trip_real_save` still byte-identical.
- **Gate:** can Lua list a hand zone's cards (`zone.getObjects()` on a `HandTrigger`, to confirm)? If yes, no `facedown` zones.
- **You have a result when** `python -m commands.tts_save` adds the zones once, the Action commits them, and a card dropped on a slot
  shows up in a test `onObjectEnterZone` print inside TTS.

### P2. Lua reporter (a new scripted tile, kept by tts_save.py)
1. `commands/tts/reporter.lua` in the bot repo (add it to `tts-save.yml`'s sparse checkout); `_sync_zones` writes it into the
   `LuaScript` of a "Game Report" tile listed in `zones`, as `_resize_targets` rewrites the global table. Not inside the Flex Table
   Control: that is a stock script.
2. Buttons: "Start game" (host only, `Player[color].host`): `started` = `os.date("!%Y-%m-%dT%H:%M:%SZ")` (UTC ISO 8601 as §4
   wants; MoonSharp's `os.date` to confirm, else write `os.time()` and P3 converts epoch seconds), id `<date>-<8 hex>`, seated
   colours and teams (`getSeatedPlayers()`, `Player[c].team`); one `createInput` per seat for an opt-in name or GitHub login (blank =
   colour only). Start also snapshots each team's 17 cards as `teams[].hand` and remembers each card's team: 03_Setting_Up puts the 3
   starting creatures face down on the slots and 3 facedown items behind them, so the slot zones, the small facedown zones and the
   seat hand zones (4 creatures, 7 items) together hold the whole starting hand. At End a card in neither set is a shop purchase:
   `seat` = the team of the zone it sits in; on the shared discard pile its `seat` is unknown (key absent, §4) until P5's log.
   "End game" (host only).
3. End game takes a snapshot: the cards in each zone (`zone.getObjects()`, to confirm; `obj.getName()`; face path from `obj.getJSON()`
   -> `CustomDeck[...].FaceURL`), every counter (`Counter.value` in the JSON; `getValue()` to confirm), the Global Turn counter ->
   `global_turns`. Classification by zone and repo folder: a `Creatures/` or `Auxiliary/Creatures/` card in a graveyard = died
   (17_Death); a minion there = died, no points (15_Minions); an item in the discard pile = used (13_Items "discarded after
   resolving"; equips, actives and deploys "do not get discarded immediately", 14_Item_Types2: a fragile equip goes when its creature
   dies, an active at 0 HP, a plain equip returns to hand; so `used` undercounts them: said in the file).
4. Winner suggestion = the death rule only: a team with no creature and no minion left in its 3 slots is defeated ("Defeat all of
   your opponent's creatures", 01_Getting_Started; a lone persistent minion still counts, 15_Minions). Both standing -> `null` (an
   absent key in a TTS report, §4).
   Creatures still in hand: the rulebook is silent (rules skill, Known gaps), and the Notebook text says so. A card on a slot beside
   an HP counter at 0 is flagged (17_Death: it should be in the graveyard). The host confirms or picks `winner` and `end` with
   buttons; `suggested_winner` stays as computed.
5. Output: `JSON.encode(report)` into a Notebook tab "Game report <id>" (`Notes.addNotebookTab`/`editNotebookTab`, to confirm);
   players copy it out (Lua has no file access, to confirm). `confirmed_by` = the host's typed login if any, else `[]` (the browser
   fills it from sign-in). `onSave` keeps the id, start, hand sets and typed names only, never a URL.
- **Gate:** P1 done (a slot zone lists its cards from Lua); `os.date` and `Notes.addNotebookTab` confirmed or their fallbacks chosen
  (epoch seconds; `printToAll` of the JSON).
- **You have a result when** a real game ends with a tab holding JSON that `/report_game` accepts unchanged, and the suggestion
  matched the table in 5 of 5 test games.

### P3. `/report_game` (bot)
1. `commands/game_reports.py`, a sync worker replying through `queue_message`/`queue_file` (architecture contract invariants 1-2;
   registration `func = to_thread(func)` then `@tree.command`; a help.py entry; change-control's add-a-command checklist). Options:
   `report: Optional[str]` (Discord caps string options at 6000 characters; a full report runs 3-6 KB), `file: Optional[discord.Attachment]`
   (`.json`; no command uses an attachment option yet), `allow_unknown: bool = False`.
2. Validation: `v == 1`; `id` matches `^\d{4}-\d{2}-\d{2}-[0-9a-f]{8}$` and `site/games/<id>.json` is not on main (dedupe, a
   `_files_at`-style tree lookup); each card `name` -> the bot's card list (`card_repo._find_closest_match`, exact normalized key
   first) -> the website id (`card_repo.site_ids`, cards.json `id`) written as `card`; unknown names refused and listed, or kept with
   `card: null` under `allow_unknown`; counts non-negative; a missing key reads as null (a Lua table cannot hold nil and TTS's
   `JSON.encode` has no null, so a TTS report omits `winner`, `suggested_winner`, `teams[].field`, `cards[].card`, `cards[].seat` or
   `players[].color` when they would be null); `started`/`ended` accepted as ISO 8601 or epoch seconds, stored as ISO 8601;
   `players[].login` kept only when `site/profiles/<login>.json` exists (else moved to `name`, §5); `reported_by` = the Discord user id.
3. Commit `site/games/<id>.json` like the save: factor the blob -> tree -> commit -> ref block of `update_repo_save` and `_git_commit`
   into `commit_file(path, data, message)` that can ADD a path (today `_git_commit` raises when the entry is missing). Reply: id,
   players, winner, flagged cards, the raw link. `site.yml` checks out `/site/` sparsely, so that push rebuilds the website.
- **Gate:** P2 done (one real report in a Notebook tab); open question 3 (who may run it) answered.
- **You have a result when** the same report pasted twice yields one file and one "already reported <id>"; a misspelt name is listed.
- **2026-10-21 (questions 3 and 4 answered):** `/report_game` is open to anyone in the guild; a report naming an unknown card is stored
  with that card unresolved (`card: null`) and flagged, never refused, so `allow_unknown` is not needed.

### P4. Statistics
1. `site/build/build_site.py` reads `site/games/*.json` -> `dist/data/games.json`; per card `games`, `wins` (in a winning team's
   hand), `deaths`, `uses`; per login W/L; per deck: a report's `teams[].hand` as a set equals a deck file's card set under
   `site/decks/` (D5) -> that deck's W/L. Card pages and Stats show them; nothing shown under 3 games.
2. Bot: `/game_stats [card] [player]` reads `SITE_URL/data/games.json` with an ETag like `_populate_site_ids`; no pickle.
- **Gate:** P3 done and at least 3 reports under `site/games/` on main.
- **2026-10-21 (question 7 answered):** no floor: statistics show from the first game. Per-deck stats can only match public decks (a
  private deck file is encrypted).
- **You have a result when** the website's numbers for one card equal a hand count over the files.

### P5. Event log and the host-typed webhook (optional)
1. `onObjectEnterZone`/`onObjectLeaveZone` (the latter to confirm) on P1 zones append `{t, card, from, to, by}` to `log`: `t` = the
   Global Turn counter plus the slot whose token reads "Is taking its turn"; `by` = that slot's team. Kill attribution by "whose
   turn" is a guess (reactions, 12_Reactions, and status damage at (ENDT), 19_Status_Effects, break it) and is labelled a guess.
2. Webhook: an input on the tile; `WebRequest.post(url, json)` (to confirm) to a Discord webhook URL the host typed this session,
   into a channel the bot reads through a new `on_message` handler that feeds P3's validation (the bot has none today, and no HTTP
   listener to post to: architecture contract); `onSave` never writes the URL.
- **Gate:** 20 real reports through P3 first; events only if snapshots prove too coarse.
- **You have a result when** one game's log replays to the same `played`/`died`/`used` as its snapshot, and a webhook report lands
  in `site/games/` with the save holding no URL after a save and reload.

## 4. Report schema (v 1; this file is its home)

| Field | Meaning |
|---|---|
| `v` | schema version, 1 |
| `id` | `<date>-<8 hex>`, also the file name under `site/games/` |
| `source` | `tts` or `browser` |
| `started`, `ended` | UTC ISO 8601 (`ended` replaces the simulator draft's `duration_s`) |
| `global_turns` | the Global Turn counter at the end (04_Table2) |
| `players[]` | `seat` (team 1 or 2), `color` (TTS seat colour; null in the browser), `name` or `login` (opt-in; only logins reach profiles) |
| `teams[]` | `seat`; `hand`: card ids (17 in constructed: 7 creatures, 10 items, 03_Setting_Up); `field`: field card id or null; `points`, `bricks`, `star_points` at the end (04_Table2) |
| `winner`, `suggested_winner` | seat or null; `suggested_winner` is the script's death-rule verdict, `winner` the host's |
| `end` | `all creatures defeated`, `conceded`, `unfinished` |
| `decided` | `host confirmed suggestion`, `host overrode suggestion`, `host chose (no suggestion)` |
| `cards[]` | `card` id (null if unknown), `name` as printed in the save, `seat`, `played`, `died` (17_Death), `used` (13_Items), `end`: `field`, `secondary`, `hand`, `facedown`, `equipped`, `minions`, `active`, `graveyard`, `discard`, `shop`, `unknown` |
| `log[]` | optional (P5, or browser games): `{t, card, from, to, by}` |
| `confirmed_by`, `reported_by` | logins who confirmed (TTS: the host's typed login, else `[]`; browser: from sign-in); the Discord user id that ran `/report_game` |

Null keys: a Lua table cannot hold nil and TTS's `JSON.encode` has no null, so a TTS report omits any key whose value would be null
(`winner`, `suggested_winner`, `teams[].field`, `cards[].card`, `cards[].seat`, `players[].color`); a missing key reads as null
(P3.2). The browser may write `null` outright, as the example does.

```json
{"v": 1, "id": "2026-10-20-3f9a1c2d", "source": "tts", "started": "2026-10-20T19:02:11Z", "ended": "2026-10-20T19:47:21Z",
 "global_turns": 6, "winner": 1, "suggested_winner": 1, "end": "all creatures defeated", "decided": "host confirmed suggestion",
 "players": [{"seat": 1, "color": "Red", "login": "MichaelJSr"}, {"seat": 2, "color": "Blue", "name": "Ech"}],
 "teams": [{"seat": 1, "hand": ["the-freezer", "mini-doomer", "..."], "field": "cave-of-the-past", "points": 9, "bricks": 3, "star_points": 2},
           {"seat": 2, "hand": ["..."], "field": null, "points": 14, "bricks": 0, "star_points": 7}],
 "cards": [{"card": "the-freezer", "name": "The Freezer", "seat": 1, "played": 1, "died": 0, "used": 0, "end": "field"},
           {"card": "mighty-pebble", "name": "Mighty Pebble", "seat": 2, "played": 2, "died": 0, "used": 2, "end": "discard"},
           {"card": "mini-doomer", "name": "Mini Doomer", "seat": 1, "played": 1, "died": 1, "used": 0, "end": "graveyard"}],
 "log": [], "confirmed_by": ["MichaelJSr"], "reported_by": "discord:123456789012345678"}
```

Snapshot limits (P2): `played` is 1 for a card outside the hand at the end and 0 for `end: hand` or `end: facedown` (a facedown item
is revealed only when played, 03_Setting_Up, 13_Items); a card still in a shop gets no `cards[]` row from TTS (nobody's card, no
`seat`; `end: shop` stays in the list for the browser table); `died` counts graveyard entries (a revived creature that dies again
counts twice; 17_Death gives points rules, not counts), `used` counts discards; P5's log makes all three exact.

## 5. Privacy and hygiene
- Names are opt-in: the reporter writes seat colours; a player types a name or login; `Player[c].steam_name` and Steam ids are never
  written. Logins count only with an opt-in profile (D5); otherwise they are plain names.
- Host-only buttons in TTS (`Player[color].host`); `/report_game` open to the guild but rate-limited (one per user per 10 min),
  deduped by id, `reported_by` recorded; an admin retracts a report by deleting its file in the repo.
- The save never holds a URL or a report (it is a public file rebuilt by the Action); `onSave` state is id, start, hand sets and typed names.
- Reports are public repo files: no private games until encrypted decks exist (`docs/Campaigns/SIMULATOR_CAMPAIGN.md` open question 3).
  2026-10-21: encrypted private decks exist (TTSCardMaker 305812c: AES-GCM, key from a passphrase via PBKDF2); a private game would
  reuse that scheme, which is not designed yet, so reports stay public files until it is.

## 6. Open questions for the user
1. Zones per team (the save is two sides of up to 3 colours each) rather than per seat: right?
2. Reporter as a new tile, or inside the Flex Table Control (stock script, already host-gated)?
3. `/report_game` for anyone in the guild, or admins only at first?
   2026-10-21: anyone in the guild (the user).
4. Unknown card names: refuse, or store with `card: null` and flag?
   2026-10-21: store with the card unresolved (`card: null`) and flag (the user).
5. Minions and `Auxiliary/Creatures` cards: their own `cards[]` rows (they go to the graveyard, 15_Minions) or folded into the parent?
6. The simulator's naming proposals (`docs/Campaigns/SIMULATOR_CAMPAIGN.md` §4): `teams[].field_card` for `teams[].field`,
   `cards[].end: primary` for `end: field`, report-level `result` for `end`, `team` for `seat`: accept any?
7. Fewest games before a number shows (3?); per-deck stats for public decks only?
   2026-10-21: no floor, numbers show from the first game (the user); per-deck stats match public decks only (private ones are encrypted).
