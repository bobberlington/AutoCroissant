# Session 2026-10-21 evening to 2026-10-22 morning (Opus 5.5, ultracode)

Asked: complete `NEXT_SESSION_PLAN.md` (now `PLAN_AS_OF_2026-10-22.md` here), then many owner requests during the night.
The plan for the next session is `docs/NEXT_SESSION_PLAN.md`; it lists the four workflows still running at the end and how
to resume them. Commits: AutoCroissant 51 (non-merge, PICKLEs aside), TTSCardMaker 105 (70 by hand or agents, 31 by the
Actions, 4 by Zomoroko's cooldown update).

## Landed

**Fork (psd-tools).** Round 3 integrated (`cards` a9f71783, tag `cards-20261021`); round 4 (smart-filter mask, composite
byte rounding, bevel texture) integrated with the user's round-3 review fixes: `cards` 0302b08b, tag `cards-20261022`
(>2% cards 15 -> 5, bad pixels 3.93M -> 2.11M). Installed on the jetson and in `~/.venvs/ac-py314-fork`.

**Bot (AutoCroissant).**
- update_stats: B8 prune then archive (deleted PSDs kept in old_stats `archived`, restored on re-add; 1f1d8c4, 7b781f1,
  9ae19f9, a587bd2); B9 retry/ETag, Session per thread (0166948); no prune on failed or truncated runs (f50f0a2); fonts cache
  by MDW/Fonts sha (66cfe9e, 7621de9); git without prompts, ssh BatchMode, lazy GitPython, on_ready once (778c2ce, b6ddf41,
  171c84b); review fixes 086a5a4.
- Bake: Edit-by trailer (c99ceb5), save with merged preview (681c32d, 7f65344), one commit per run + pending gate +
  `--pending` (3a61372, e0f7fff, bb900fb).
- Admin-only /purge, /update_metadata, /mass_replace, /alias, /set_repo, /delete_song, /delete_all_music (784f718, 673d183).
- TTS save updater: report and subject (57b0005, 7e0daa1, ecbdd8a), unused CustomDeck keys dropped (4fb22ec), item decks
  filled, `add_new_since`, team fields left alone (a35b6de, cae1f2f).
- Skills: frontmatter quoted (79d05e7), cooldown/uses tags (39da9ed), rules skill refresh (8ca6234), (Action) parity test
  (64d88a6, cf3a0be). Goldens re-pinned after the cooldown update (0fef159, 33b1360).
- Deploys to the jetson: 2ce53bf (21:19), 9f355f1 (01:42), dfdf953 archive (03:11), 3bea369 + fork tag (04:05), a587bd2
  (04:52), e9154d2 (05:45).

**TTSCardMaker.**
- Website: series/subseries/authors from the filter bar (idea 25: 7c0544ba, 9f0c9ece, 8d8f51dc); People avatars, share,
  activity chart, bake credit (4ae3e4f0, 08b153e7); deck search, decks using a card, montage (57fa1388, 73cd86c6); Details on
  rulebook pages (8a3c0596, 021e055d); shared deck picker (0d3616e3, 3fa98d71); caches and allowlisted pickles (62f13f94,
  259a606e, 972a34ed); bot data read at main's commit (4069607c); cooldown + uses badges (8ec1b925, c3f9d685, 73904dbd);
  sticky headers (8e83bfc3, 7f60ef76); created sort and card mentions (3dc07ca5, 159ff008, f8cc1d3d, 0fa6f8c6, 8a1f47a4);
  (Action) badge (f6ce500e, b686d0cd, 950f4e56).
- TTS save: daily gate (2911a8ce, 075e6c43, 843a4a99); teams Hearts/Clubs (24181aae); 48-line Lua, no zone claiming
  (c9873ddf); TTS/checks suite (b374a22b and later); deck rules and 98 cards added, 131 names fixed (cbff5f51, 7074dd06);
  Maxwell the cat (7ca5e30c); White Palace to the Morgue (b6330730, e6597924); snap points (83640cea, e17f38c6); mats and
  Militron (a93461fa, 0d33adc8). drift.py BASELINE moves with every hand edit (now 0d33adc8; merge, don't rebase, hand edits).
- Card edits: Twin Emperors second DEF (98e85f18), Pix and Tainted Lazarus notes (fa2006d2); five title reprints baked live
  (26 commits, one per card: why the bake now batches); card-edits.yml live every 30 min (6c321989, adb87632, 23a4237f), pinned
  `cards-20261022` (75933ad5).
- test.pdf removed (571c0505).

**Research and data.** Standardization rehearsal (663f940); new badge demo without header text
(`~/Documents/Tools/psd-research/demo-2026-10-22/`); data build design (`docs/Campaigns/DATA_BUILD_ACTION.md`, winner:
data on an orphan `data` branch, the bot as a reader, a 30-min Card sync); Discord archive miner (GET only) running over all
channels into `~/Documents/Tools/discord-archive/`.

## Lessons

- A teammate's push makes raw.githubusercontent lag: read the bot's files at main's sha (4069607c).
- A verifier overwrote the file under review (TTS deep audit): give reviewers read-only copies or check the file hash.
- One commit per baked card buries main: a bake run is one commit.
- Rebasing a hand edit of the save breaks drift.py's BASELINE: merge instead, or re-baseline after.
- `pgrep -f` over ssh matches its own command line.
- The jetson checkout is `~/Documents/AutoCroissant` (a deploy to the wrong path restarted the old code once, 05:40).
- Removing a worktree before the user's review loses nothing only if the branch is pushed or kept: keep branches until the
  review is collected.
