# Session 2026-10-25

Unattended from ~07:45 PDT while the owner slept, then with the owner from ~13:30 to midnight. Everything below is on
main in both repos. The jetson runs AutoCroissant 8b0ce02 (AC main's later commits, to 4b29d55, are tests and skills
only). OJWikibot's deployed commit was not re-checked at wrap-up (5135280 in the morning). Designs that shipped are in
this folder; `docs/Future/` keeps the two with a step left.

## Landed

| What | Where | Notes |
|---|---|---|
| Code cleanup | AC 8198291, TC 2770b982 | Pickle-era code, Discord edit credit, finished tools, dead code, the fonts cache, path tuples; card-check and font-check folded into the guards' check job; 223 duplicate fonts removed (KungFuPandaRegular-xYjV.ttf kept: an old copy that matches no current file) |
| Card data on a merge head | AC abc5ec9 | `git show -s` diffs a merge (needs blobs under GIT_NO_LAZY_FETCH): `git log -1` |
| Website small fixes, game.txt | TC d0dfed64, 56651852 | `WEBSITE_SMALL_FIXES.md`, `GAME_TXT.md` |
| Test cut | AC f32dfa7 | `TEST_SUITE_CUT.md`: 1,563 -> 553 tests, about a minute; `-n auto` capped at 6 (e2f287d) |
| Leftovers | TC b67b8799, AC 032ede9, 8a1e629 | TTS/audit.py and AUDIT.md removed (last at 843a4a99); `discord_edits.py` -> `card_dates.py`; reminder retirement shims removed |
| Backups | AC be604a5, OJWikibot 5135280 | Live reminder/history files stay untracked; the bot commits a copy to `backup/` on change over SSH from a separate clone (neither GitHub token can write the bot repos) |
| Optimization clear wins | AC d3f622c, TC b77998c0 | `/query` fuzzy p95 129 -> 6.5 ms, the `/query` refresh race fixed, faster restarts, single-pass git walks, opaque-pixel fast path, font-name index, inlined CSS + esbuild minify (836 -> 520 KB), wheel cache on requirement lines. Plan and questions: `~/Documents/Tools/wt/scratch/opt/PLAN.md` |
| Guard and bake pack | AC 4d4574c, TC 1c717c8b | Straight apostrophes everywhere; badges sized 1.35x the text's drawn cap height and centred by Baseline Shift; old stoogedings glyphs redrawn; all silent (commit messages only) |
| Sweeps | TC 218a122d, 6778c592 | Badges on 101 cards, apostrophes on 34; guards on both pushes green and in agreement (byte-identical fixes) |
| Parts editor, data + bake | AC aae7233, TC abf70775 | `docs/Future/PARTS_EDITOR.md`: parser 3, details v2 with parts, per-layer bake, variants, star folder moves, first type icon; full re-parse: 1 field change (Ms Mowz's display name) |
| PNGs, layout, verdicts | TC a4d137bf, AC be39f86 | `SMALLER_PNG_PUSHES.md`, `LAYOUT_SUPPORT.md`, `docs/Future/CARD_SYNC_VERDICTS.md` (migration left) |

## Afternoon (owner back, ~13:30-17:10)

On main; deployed with the evening's 8b0ce02.

| What | Where | Notes |
|---|---|---|
| Variant tags | TC 572680d5 | ' #N' tags on the 5 multi-picture PSDs (names only) |
| Michael's Blessing | TC b6b22965 | text sized to its badges, arrows aligned (the badge sweep's flag) |
| Guards window from the push (P-1) | AC fc420bb, TC 594050bf | `card-guards-quiet` timer 25 min (set), Land sleeps to 30 min after the push |
| Dirty-box gate (K-1) | AC 4047e4b | the untouched gate renders only the dirty box; `PSD_DIRTY_VERIFY` (full compare) on in tests |
| Card sync chain (P-5) | TC 3ca08fda, AC 8690917, e45c127 | a run with work left dispatches the next; one safety cron at :17 (`0,30` gone) |
| Instant replies (B-1) | AC d1725ce, 9694021; OJWikibot a70877d | drains wake on enqueue, 1 s fallback |
| OJWikibot lane 3 | OJWikibot 8ea1c7c..174f62b | worker reply slices, early stop, no blocking start, C twister |
| Site: bundle, WOFF2, lazy chunks (S-1, S-4, S-3) | TC c31e67a3, 7fb7b355, 0a1bfbe3; AC 6a87179, 95b5f20 | one esbuild bundle with a source map; WOFF2 copies of the mock-up's fonts; editor, Stats, People on first use |
| First screen in index.html (S-2) | TC c31e67a3, removed 54395592; AC 4a4ad4c | Safari redrew its pictures (the flash); removed the same evening (back at 19:09, Evening table) |

## Evening (~17:10-24:00)

On main in both repos; the bot code is deployed (8b0ce02). Times are PDT; data commits are stamped in UTC.

| What | Where | Notes |
|---|---|---|
| OJWikibot outage lane (O-1 to O-5) | OJWikibot cdf6f03..8924db4 | Landed 16:14-16:37, left out of the afternoon table: live game checks opt-in (`OJ_LIVE_GAME`), throttle 2 s, status commands answer during an outage, `basetimestamp` from the caller's read, retries without respawns, durable edits, append-only plans, own-edit conflicts, paced re-runs |
| Lanes, land3 (17:20) | AC 18223d8, 4f89a2e, 7ba8e4f, 5482254, a3a20a9, d10aca2; TC a58b241a, 1f6e1136, 70bf91f4 | /query_ability as grid images (B-2); the card list saved at startup (B-3a, `card_tree.json`); one main lane per guarded push (P-2); bake copies reopen the card's bytes, one preview decode (K-3, K-5); guards pool at most 4, `AC_TEST_WORKERS`, old Kung Fu Panda copy dropped; badges resized on 43 cards with rewraps allowed |
| Lanes, land4 (17:53) | AC c9697c0, c3c8d72, c577935, 46e6065 | Real-card tests at one pinned commit (T-2) and guards on every runner core (K-2); numbered-PNG duplicate scan dropped, joined ability splice deleted (ab132b2), `+ability` built and off (f92fb5e); bake bugs (body colour, kept characters, fit, kerning); docs |
| Link previews | TC 89929404 | The link-preview image URL carries its content stamp |
| Website-only values | AC a198191, TC 9af79e9b | `_site` in Card Edits; a per-field "Website only (don't change the card image)" checkbox; never baked; a quiet "Website only" tag |
| File rename | AC 8682e1e, TC d5ae5296 | "Also rename the card's files" (`_rename`); everything keyed by the file follows; old links redirect |
| Related-card headings, Missing filter | TC 1d979aee, AC bf913c5 | Headings put the count first; Missing filter (no series, subseries, types, ability, author, stars, stats, item type or ratings, and artwork sources with the artwork fields); missing values sort last both ways |
| Card stays open | TC cc5f238a | An edit that makes the open card leave the filter keeps it open |
| Sidebar scroll | TC b9b4228a | The filters keep their scroll on a click |
| Exclusion filters | TC 6102df89 | Right-click or long-press a filter to exclude it, ≠ on selects, `x`-params |
| Subseries | AC 5bb28a1, TC 8f2b3367 | One shared group across series; needs no series |
| MyriadArabic-Regular | TC 73edc8a4 | Added to MDW/Fonts (900 glyphs; the Regular face extracted via CoreText) |
| Problem cards re-run | AC eadbd36, ba0b91f | Cards with an edit that can't bake: 122 -> 54. Owner copies: `~/Downloads/Problem cards 2026-10-25 (after fixes).csv` and `.md` |
| My filters | TC cf412892 (via 82ad8103), AC 2151196 | Saved searches, browser-only (localStorage), `cf=`/`xcf=` params |
| Artwork sources | AC ae22a6e, TC 82ad8103 | Database-only fields `art` and `ability_art`, per entry (duplicates and variants too); Missing options; `/query` shows them as masked links. Deployed at ae22a6e |
| Bucket drop fix | TC 9b756b02 (via 82ad8103) | A drop on a pinned box's ✕ or rename button lands on the box |
| Safari flash, root fix | TC a7ed9744, AC 229ff79 | The grid stays in the page: tiles reconciled in place, sign-in reuses tiles, S/M/L in place. The phone first screen is back (S-2): its 8 pre-written tiles are adopted in place, and their href is dropped once adopted, so middle-click on those 8 opens no tab. Live check: 0 tiles removed on a redraw |
| Census bug fixes, parser 4 | AC 8b0ce02 (land6; epithet fix d304f3b), TC d76e6c51 | Debuff star folder moves (`STAR_PATH`: any "N Stars" folder except stars-from-PSD folders; the site's parts.js too); copied smart-object star IDs derived (multi-picture star edits bake); a first type icon touching the title bakes, flagged; split titles keep their parts after an edit (`_named_pieces`); type icons in a Stars group are no stars (Toy Freddy 3); hidden giant "star test" layers skipped. `PARSER_VERSION` 4: full re-parse by Card sync runs 40262608396 and 40262751459 (data 122f9065, 7c52c2b5) changed only Pike Beam's stars (3 -> 2) and Leeroy Jenkins' title. Deployed |
| Duplicates and variants: one category | AC 8b0ce02, TC 33f0063f, d76e6c51 | Every variant picture (2+) is a duplicate of picture 1: counted once, bot "Same card as X"; it keeps its own id, name, art, text, parts, author and ratings. A hand-set `dup_of` for a different card is honoured. TTS save names fixed per picture (A_Snack.png was "Supper"). Data: `duplicate_of` on the 6 pictures. Duplicates are marked by hand on the website |
| Website edits published | TC 98131685 | "Website edits: 104 changes" (data 1cf84da1) |
| Version commit links | TC e319f741, AC 3cfd908 | Each card version's optional "commit" link (History texts, Recorded changes, Image versions). Text versions map by PSD blob to the commit that made those bytes, renames followed; no link without proof. Coverage: older texts 1113/1120, current 584/615, recorded changes 1083/1090, image versions 4281/4281 |
| More than 9 pinned boxes | TC 1ccfe437, AC 04b0228 | Cap 100, rows and columns; the tray is height-capped with its own scroll and edge-hover scroll while dragging; keys 1-9 on the first nine; "Skip to cards" |
| Font filter | TC 3853b688, AC 4b29d55 | A sidebar text box with autofill; counts within the search; several fonts = any; exclusions; `font=`/`xfont=`; `data/cardfonts.json` loaded on first use |

In flight at midnight: the `+ability` placement and font work (AC worktree `~/Documents/Tools/wt/ac-plusplace`, branch
`work/plusplace`; scratch `~/Documents/Tools/wt/scratch/plusability2/`). See the plan.

### The owner's rulings of the day

From memory `owner-decisions-2026-10-24-night` (the 10-25 lines), in order:

- Card sync: chain runs that leave work, plus one safety cron at :17. Only main writes wait 30 minutes.
- Discord's card-edit commands retired; all card edits go through the website. Pickles, psd_details.json and /push retired.
- Cards need neither types nor an ability. A first ability or type comes from the card kind's template (creatures:
  Creature_Template; items, fields, N.M.E.: Item_Template), flagged. A stars edit moves a folder-starred card, creating
  the folder.
- Variant pictures: ` #N` layer tags in one PSD.
- Badges: outlined pill under overlays; size 1.35x the text's drawn cap height, centred by Baseline Shift; the 58
  shift-only badges resized anyway.
- Apostrophes are always straight. The guards fix Every-line Composer, old stoogedings glyphs, badges and apostrophes
  silently; Discord gets bake posts only.
- Backups: the bots commit their live files to their repos on change; config.py stays private.
- Optimization: build the clear wins; phone first paint built; instant bot replies in both bots; /query_ability as
  grid images only. The guards' quiet-window timer at 25 minutes (~14:30). OJWikibot never drops wiki edits or
  scheduled commands; no Cloudflare (~15:20).
- ~16:45: remove the pre-written first-screen cards (Safari kept flashing). They came back at 19:09 with the root fix
  (TC de98ccd4); no ruling on that is recorded (plan: owner item).
- Website-only values (~17:30): a card-image field can be marked "Website only (don't change the card image)" per
  field; never baked; a quiet "Website only" tag.
- File rename (~18:00): a name edit renames the card's files only when "Also rename the card's files" is ticked (off by
  default); website-only names never rename.
- Website: related-card counts first; the Missing filter; the open card never closes on an edit.
- A subseries is one group across series and needs no series. Right-click or long-press excludes a filter.
- Artwork source and Ability artwork source: database-only, per entry.
- My filters: saved searches, browser-only.
- Variant and duplicate are one category, "duplicate"; duplicates are marked by hand.
- Each card version gets an optional commit link, mapped exactly or not at all.
- Pinned boxes past 9 (cap 100).
- A font filter, a card's fonts being exactly its Details panel's list.
- `+ability`: one placement rule (top of the card's free text area, flowing down, never off the card) and the card's own
  font when it exists, the template's only as the fallback. New test files go to `plus_ability_2/`.

## Incidents

- **Out of memory, 07:17.** 429 git processes (~226 GB). The likely cause, reproduced at 15:05 and stopped: a
  measurement agent made a blob-less clone of the local 16 GB card repo, so every blob read became a lazy fetch with
  its own `upload-pack` and `pack-objects`. Rules now: never partial-clone the local repo, xdist capped at 6, one full
  suite at a time (`suite-wait.sh`), a git watchdog; memory `resource-limits`.
- **Wait loops that matched themselves.** `until ! pgrep -f "pytest.*-n"` finds its own shell; replaced by
  `~/Documents/Tools/wt/scratch/suite-wait.sh`.

- **Wait loops on short shas (evening).** `gh run list -c <sha>` matches only the FULL sha; loops given a short sha
  never saw their run and hung until killed. Use `git rev-parse` first.
- **A lazy-fetch near-miss (evening).** During the duplicates work a scratch blob-less clone started lazy fetches; it
  was killed after about 2 minutes, before the git count climbed. Same rule as the 07:17 crash.
- **Tokens sent to the wrong host.** Earlier on 10-25 the Discord token and HF_TOKEN were sent to api.github.com by
  mistake; GitHub answered 401. Both need rotating (flags below). Rule: send only GitHub tokens to GitHub.

## Flags for the owner

- Michael's Blessing: the badge sweep shrank its badge table (fixed in the afternoon, TC b6b22965).
- Not fixable glyph by glyph (a Photoshop re-save fixes them): badges on Restrained, WX-78, Wendy, Wigfrid, Soul
  Master, Traitor Lord, Greenpath, Soul Sanctum, Ominous Red Circle, Quick Slash, Golden Metronome; apostrophes on The
  Temple/Gates of Ahn'Qiraj titles, 04_Table5, 06_Reading_Cards, 12_Reactions2, 10_Attacking, 17_Death; the
  MDW/Old Rules pages were never swept.
- Layout support still refuses: Grimora and Skeleton Mate (bars painted in one layer), Slugcat's spd, three
  smart-filter star layouts, PoD04 Zebra Bloon +1, Living Bread.
- OJWikibot's wiki API has answered HTTP 500 since 10-23 (wiki/host side).
- The Card sync schedule fired about 8% of the time: answered by the chain and the :17 safety cron (afternoon). Both
  proven by night: run 40262751459 was dispatched by github-actions[bot] after 40262608396; the :17 cron fired at
  21:19, 00:25 and 06:18 UTC.

Evening (open at midnight):

- Rotate the Discord TOKEN and HF_TOKEN (see Incidents).
- Open the `+ability` test files in `~/std-work-1025/plusability/<card>/plus_ability_2/` when they land; approve or
  correct them (then `ADD_ABILITY_LIVE` is flipped). Possible option from that work: the card family's own centring
  and colours.
- Photoshop re-saves: the 11 badge and 7 apostrophe cards above, and the 54 problem cards
  (`~/Downloads/Problem cards 2026-10-25 (after fixes).csv` and `.md`).
- Safari: check the grid flash and the many-boxes edge scroll (headless checks never ran Safari).
- Mark the PNG-only duplicates on the website.
- The Desktop TTSCardMaker clone is 32 commits behind origin/main (clean): pull it.
- `site/profiles/MichaelJSr.json` names the series "Don't Starve Gourmet Expansion", which isn't a known series.
  "Gourmet Expansion" is a subseries of Don't Starve (1 card): probably series "Don't Starve" plus that subseries.
