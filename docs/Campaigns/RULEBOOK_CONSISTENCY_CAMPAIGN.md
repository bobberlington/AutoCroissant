# Rulebook consistency campaign (2026-10-20)

**Status:** designed 2026-10-20, not scheduled; plan item 19 (`docs/NEXT_SESSION_PLAN.md`, "Ideas 5-23"). User request of
2026-10-20: "one last campaign to find rulebook inconsistencies".
The user is the rules owner and makes every ruling; this campaign finds, classifies, logs and guards, it never decides.
Page citations are the PSD stems of `~/Desktop/TTSCardMaker/Auxiliary/Rulebook/` (48 PSD + PNG pairs) as
`.claude/skills/impossibility-simulator-rules/SKILL.md` names them, e.g. (08_Global_Turn). Card and page text is
`~/Desktop/TTSCardMaker/site/dist/data/cards.json` (801 rows, 772 distinct names; 767 cards counted one per `psd|name`,
the tags.py rule every number below uses) and `rules.json` (48 pages; `site/build/build_site.py` slices them out of the
bot's stats.pkl, so they equal its 48 `rulebook` entries) of 2026-10-20. Game words only: creatures, minions, items, item types, field
cards, N.M.E. items, status effects (buffs, debuffs), stars, types, abilities, timing tags, global turn, hand, graveyard.

## 1. Goal and non-goals

**Goal.** Every timing tag, status effect name and term used on a card or a page has exactly one definition that a
player can see on a page; every contradiction between pages has a written decision; a checker keeps both true on push.
**Non-goals.** No ruling made in code, in a skill or by an agent (`docs/Campaigns/SIMULATOR_CAMPAIGN.md` §1; its proposed
`site/assets/engine.js` "cites, never decides", §5). No balance or wording rewrite of abilities beyond what a decision needs. No hand edits of
rules.json, stats.pkl or the pickles (the pipeline republishes them). `MDW/Old Rules/` (22 PSD+PNG pairs) is history.

## 2. Sources and the scripts that exist today

| Source | Path | Used for |
|---|---|---|
| Rules skill, Known gaps | `.claude/skills/impossibility-simulator-rules/SKILL.md` | contradictions, undefined terms, silences, per-page citations |
| Verbatim pages with `[hidden ...]`, `[off-canvas ...]`, `[overflow ...]` markers | `.claude/skills/impossibility-simulator-rules/reference/rulebook.md` (built by `scripts/build_rulebook.py` + `notes.py`) | H class below |
| Glossary: every abbreviation and defined term with its page, plus undefined terms | `reference/glossary.md` | U class |
| Card-data counts (tags §4, status effects §3) | `reference/data-facts.md` | baseline numbers |
| `scripts/tags.py <cards.json>` | tag occurrences per card kind, other parenthesised words | P1 seed |
| `scripts/mentions.py <cards.json>` | how many other cards name each status effect | unused status effects |
| `scripts/layers.py <dir> <page.psd>...`, `alllayers.py <psd>` (psd-tools venv `~/.venvs/ac-py314-psd125`) | hidden and off-canvas text layers | H class |
| Plan section "Rulebook data issues" | `docs/NEXT_SESSION_PLAN.md` (Queued) | the first list of issues |
| Badge tag list the website renders | TTSCardMaker `MDW/Fonts/stoogedings-src/build_stoogedings.py` `badge_tags()`/`BADGE_GROUPS`; `site/assets/text.js` `tagInfo()` | tags the site knows (16 patterns, same as 02_Abbreviations) |

## 3. Seed inventory (measured 2026-10-20; the first rows of the decision log)

Classes: **C** contradiction between pages, **U** undefined term or tag, **M** rulebook vs card text mismatch,
**T** typo, **G** timing tag misuse, **H** text players cannot see, **V** stray visible text (a decision), **X** parser
slip, **S** silence (a decision, not an inconsistency). Rank: C and U first (they block the simulator's Layer 3), then
M, H, X, T, G, V, S.

| Id | Class | Where | Finding |
|---|---|---|---|
| C1 | C | (21_Damage_Types) vs (07_Stats3), (10_Attacking) | NORMAL "DMG - (1/2*DEF)" vs "damage = ATK-DEF" (07) and "Attacks deal ATK - DEF" (10); an old-rules leftover (`MDW/Old Rules/17_Damage_Types`) |
| C2 | C | (12_Reactions3) vs Hardened card, (12_Reactions) | "5 stacks of hardened, giving it 3 defense"; Hardened is +0.5 DEF per stack, so 6 (12_Reactions says 6) |
| C3 | C | (07_Stats3) | text "-1 DEF BAR --> 1"; its picture shows 1.5 |
| C4 | C | (19_Status_Effects) vs (09_TakingYourTurn) | status DMG/heal "procs at (ENDT)" vs phase 4 "Status Effects deal damage/heal, duration ticks down by 1", after phase 3's (ENDT) |
| C5 | C+H | (12_Reactions2) hidden vs (12_Reactions3) | hidden older draft resolves 4>2>1>3 ("Attacks also resolve forward"); the visible rule is 4>2>3>1 |
| C6 | C+H | (22_Negation) hidden | older ITEMS block: "returns to the hand", "Equip and active items" (no Deploy), "are still expended" |
| U1 | U | 44 cards | tags defined nowhere: (1GCD) 22, (2CD) 14, (1CD) 13, (2U) 6, (2GCD) 2, (3U) 1, (4U) 1 = 59 occurrences on 44 distinct cards; plus "(1/2CD)" on Chris & Jordi. (02_Abbreviations) defines 16 patterns; all 16 are used on cards; the website renders the 7 as plain text |
| U2 | U | pages | "prime creature" (15_Minions, 16_The_Field2, 18_Field_Cards), "tier" (11_Abilities), "proc"/"instance" (19_Status_Effects), "on-hit" (10_Attacking, 22_Negation; "on hit" 12_Reactions4), "innate actions" (11_Abilities7), "cleanse" (Hexed card), "N.M.E." (05_Shops), the "Don't be stupid" rule (01_Getting_Started) |
| M1 | M | (19_Status_Effects), (19_Status_Effects2) vs cards | "Smouldering" 5 times on pages (+1 lowercase on 19) vs the card "Smoldering" (20_Creature_Types spells it the card's way); "Regeneration" (19) vs the card "Regen" |
| M2 | M | pages vs 64 status effect cards | the pages name 19 of 64 status effects (case-sensitive whole word; SKILL.md Known gaps says "~22", stale, P1 refreshes it) and define none; the cards are the only definitions |
| M3 | M | cards | 10 status effects no other card names: Reinforced, Chained, Empty, Exposed, Loony, Cheap, Corrupted Blood, Contracted, Curse, Recursion (website flag "Unused status effects") |
| T1 | T | cards | Suffocation "Can't Effect [water]"; "imprisoned" and "smeckeldorfed" lowercased once each on other cards |
| T2 | T | pages | "occours" (12_Reactions), "abbrviations" (11_Abilities), "Abilites" (02_Abbreviations), "inflcited" (19_Status_Effects), "seperate" (10_Attacking2); the checker's spell pass lists the rest |
| G1 | G | cards | (OFS) abilities "activate when a creature first comes out of the hand onto the field" (11_Abilities5), yet (OFS) is on field cards Pandora's Gym and White Palace, which are bought and played, not summoned; (WOF) on the auxiliary item Earth Totem. Misuse or an intended reading: to confirm |
| H1 | H | (09_TakingYourTurn) x2, (11_Abilities) x2, (12_Reactions2), (22_Negation) | 6 hidden text layers on 4 pages (layers.py, 2026-10-20): ATTACK and DEFENSE MODE duplicates; PASSIVES heading + paragraph; the two drafts of C5/C6. All are in rules.json, stats.pkl and /query_rulebook because the parser treats every text layer of a rule page as ability text (`commands/psd_parser.py:499`, no visibility check before `:523`) |
| H2 | H | (11_Abilities6); (12_Reactions), (14_Item_Types), (20_Creature_Types) | off-canvas "ABILITIES 7"; overflow text not drawn: "of 5.", "if you find this text dont delete it", the GIRL sentence "This may only be an innate type in special cases ..." |
| V1 | V | (16_The_Field) | layer "Secondary Field/Phase 1 copy 2" reads "He pissed on my fucking wife" in a visible lavender box (PNG cropped at bbox 672-1013 x 1611-1665, 2026-10-20); not hidden text but page content the user keeps or deletes |
| X1 | X | (14_Item_Types) | "May not be used on [support]": the page shows the machine icon (icon above the midline went to the page's types; the gap took the next icon) |
| X2 | X | (14_Item_Types) "[infliction]", (14_Item_Types2) "[fragile equip] [active] [deploy]", (15_Minions), (16_The_Field) "[defensive_minion]", (19_Status_Effects2) "[debuff]", (20_Creature_Types) 6 types | leftover icons appended after the page number |
| X3 | X | (08_Global_Turn), (15_Minions) | 08's six example creatures A-F read "SPD: 5  : 3": the star icons are smart-object layers (Text/L1-L3/Top, Bottom), never text, and are not injected into the gap (rulebook.md restores "[icon: ★]"); 15's two minion-icon gaps were swallowed ("TYPES:Offensive") |
| S1 | S | see SKILL.md Known gaps "Silent" | lane tie by star level (08); combo damage on sneak attacks and ability DMG, "25% +1" rounding (10); "one ACTION" (09) vs "2 innate actions" (11_Abilities7) and Inspired; creatures in hand and winning (01, 15); hand limits, summoning beyond refilling a slot (16); starting BRICKS; unspent star points; "5+ star" vs decks ending at 5★ (22); COUNTER item type (14) vs damage type (21); the Auxiliary/Creatures cards. Outside that list: reacting to (OFS)/(OS)/(OD) abilities ("nothing is stated", SKILL.md §3); Constructed's shop pool (`docs/Campaigns/SIMULATOR_CAMPAIGN.md` Layer 1) |

**Numbers measured today** (commands in §9): 604 tag occurrences of 35 distinct tags in 351 of 767 cards; 7 distinct
undefined tags, 59 occurrences, 44 cards; 0 defined-but-unused tag patterns; 0 status effect defined twice; 19 of 64
status effects named on pages; 10 named by no other card; 2 spelling drifts (Smouldering, Regeneration); 6 hidden
text layers on 4 pages; 1 off-canvas and 3 overflow texts (rulebook.md markers); 6 pages with icons after the page number.

## 4. Phases

### P0. Decide the frame (user, 10 minutes)

Confirm: the classes above; the decision log lives at `docs/Campaigns/rulebook-decisions.md` (proposed, not created);
fixes today are Photoshop edits of the page PSDs; the checker runs by hand first. **Gate:** no P1 code before this.

### P1. Checker: one script from tags.py + mentions.py (1 day)

`python3 -I site/build/check_rules.py`, no flags (simple-tooling preference): fixed inputs `site/dist/data/cards.json`
and `rules.json`, `Auxiliary/Rulebook/*.psd` when present (section 4 skips when the folder is absent), the baseline
`site/build/rules_baseline.json`. Proposed home TTSCardMaker `site/build/` beside `build_site.py` (sections 1-3 and 5
need nothing beyond the standard library; `requirements.txt` is `pillow`); a copy in the rules skill's `scripts/` for
hand runs: to confirm. Offline, read-only, prints a report and exits 1 on anything not in the baseline. Sections:
1. **Tags.** Every `(TAG)` on cards and pages; its pattern (digits to X); defined on (02_Abbreviations) and in
   `badge_tags()`; undefined; defined but unused; variants that the strict regex misses (`(1/2CD)`, lowercase, spaces).
   False positive to drop: "(DMG)" on (10_Attacking) is an abbreviation in prose, not a tag.
2. **Status effects.** The 64 card names; each page and card that names one (case-sensitive whole word: "empty slot"
   on 16_The_Field is not the debuff Empty); near-spellings (lowercase match or one edit away: Smouldering, Regen);
   defined twice; named by no card; `[X]T[Y] Word` where Word is not a card, taking names up to three words ("2T1
   Become Back Money" on Gold Clot and Mr Krabs, "1T1 Marked For Death" on Salt) and the bare "1 Marked For Life"
   of Pix and Z!!!!!.
3. **Types and keywords.** `[type]` tokens on cards and pages vs `Types/` and `stoogedings.json`; Types with no card
   (lightshadow, null per rulebook.md); glossary terms used but undefined (U2 list, grown by hand).
4. **Page hygiene** (needs psd-tools): hidden and off-canvas text layers; text past its box (overflow) when psd-tools
   exposes it, else the rulebook.md marker list; icons after the page number; 3+ space runs (today only 00_Table_of_Contents).
5. **Spelling** over pages and cards with a word list seeded from the glossary and card names; findings are T rows.
**You have a result when** one command reproduces today's numbers (§3 last paragraph) and lists the same 44 cards.

### P2. Decision log (user decides; an agent only drafts rows)

`docs/Campaigns/rulebook-decisions.md`: one row per finding: id, class, pages, verbatim quote, options, decision, date,
status open / decided / fixed / verified. Seed it with §3. **Gate:** every C and U row is decided before P3 touches its
page; S rows may stay open (the simulator leaves them "by hand", `docs/Campaigns/SIMULATOR_CAMPAIGN.md` Layer 3 G3).

### P3. Fixes

- **Pages, today:** Photoshop on `Auxiliary/Rulebook/<page>.psd`, export the PNG. Delete the hidden layers of H1 (a
  hidden layer is still parsed; deleting is the fix), enlarge or cut the overflow boxes of H2, decide V1. Then, in
  this order: push PSD+PNG to TTSCardMaker main; `/update_stats` re-parses the 48 pages into stats.pkl; `/push` lands
  the PICKLE commit on AutoCroissant main; the next `site.yml` run (push, cron every 6 h, or dispatch) rebuilds
  rules.json from that pickle (`build_site.py` `load_bot_pickle("stats.pkl")`; its checkout has no Auxiliary/Rulebook,
  so a page push alone changes nothing on the site); rebuild `reference/rulebook.md` with
  `python scripts/build_rulebook.py` (no `-I`, it imports `notes.py`) and re-check Known gaps.
- **Pages, later:** a text-first rulebook (the user's idea; `docs/NEXT_SESSION_PLAN.md` item 15) or Task 3's
  psd-tools text edits make a page fix a text edit; the checker is unchanged.
- **Cards:** U1 is one decision with two outcomes: define (XCD), (XGCD), (XU) on (02_Abbreviations) and add them to
  `badge_tags()`/`tagInfo()` (the Old Rules' "(CD)" meant counter damage, now CDMG: say so on the page), or rewrite the
  44 cards onto existing tags. Card text edits go through the edit pipeline (PSD + database edit in `site/Card
  Edits.json`, `docs/NEXT_SESSION_PLAN.md` "Two kinds of card edits"); today Photoshop. T1 and G1 likewise.
- **Parser:** skip hidden text layers on rule pages (`psd_parser.py:499-523`; change-control, goldens, a sandboxed
  sweep). Predicted effect: the texts of 09, 11, 12.2 and 22 shrink, 4 rulebook entries change in stats.pkl, 0 cards.
  Not a substitute for deleting the layers. X1-X3 slips are the extraction campaign's (gap vs icon order, star glyphs).
**You have a result when** the checker's C, U, M, H and X sections are empty and `rulebook.md` carries no markers.

### P4. Guard

By hand after every fix (`python3 -I site/build/check_rules.py`). Later a step in TTSCardMaker `site.yml` after
`build_site.py` (push to main, cron every 6 h, dispatch) running sections 1-3 and 5 only (the sparse checkout has no
Auxiliary/Rulebook and no psd-tools) and failing on anything outside `rules_baseline.json`; the baseline starts as
today's inventory and shrinks to empty, so no fix is undone silently. The step goes in its own job or is
`continue-on-error: true`: `deploy` has `needs: build`, and a red check must not stop the site publishing (`docs/Campaigns/GAME_REPORTS_CAMPAIGN.md`
P3 relies on that). Fits the later card standardization campaign (`docs/Campaigns/CARD_STANDARDIZATION_CAMPAIGN.md`; same script, more sections). Plain script first, no
watcher (simple-tooling preference). **You have a result when** a hand run after a page fix is green against the
shrunken baseline and the Action step reports the same numbers as the hand run.

### P5. Semantic pass (one afternoon, human-reviewed)

One Claude API call for the whole book (`claude-opus-5-5`; rules.json is 67 KB, SKILL.md 41 KB, so it fits one
request; a Message Batch halves the cost if repeated): input = the page texts plus the rules skill; output = candidate
contradictions and ambiguities, each with two verbatim page quotes. Every candidate becomes a decision-log row with
status "proposed" or is rejected with a reason; nothing is applied by the model. **You have a result when** every
candidate is a row or a rejection, and at least the six C rows above were found independently (else the prompt is weak).

### P6. Forcing function: the Play tab's rules automation

`docs/Campaigns/SIMULATOR_CAMPAIGN.md` Layer 3 (the proposed `site/assets/engine.js`) encodes only decided rules (its gate G3); every
rule it must leave "by hand" is a decision-log row, so the simulator work drains S1 and finds what reading missed. The
log is the hand-off between the two campaigns. **You have a result when** every `manual` item in Layer 3's list has
a decision-log row and none is decided in code.

## 5. Order and gates

P0 -> P1 -> P2 (seeded) -> P3 pages (C1-C6, H1-H2, V1) -> P3 cards (U1, T1, G1) -> P3 parser -> P4 -> P5 -> P6.
Gates (take the branch, do not rationalize): **G1** P1 reproduces §3's numbers, else fix the checker before any log
row. **G2** every C/U row decided before its page is edited. **G3** after each page fix, once `/update_stats`, `/push` and a `site.yml` run have
landed it, the checker's count for that page drops and no other page changes (diff rules.json before/after; right
after the PSD push rules.json is unchanged, so do not diff then). **G4** the Action is added only when the hand run
has been green on a real push once. **G5** P5 findings never bypass P2.

## 6. You have a result when

Zero undefined tags on cards (today 59 on 44); every status effect defined exactly once and every page spelling equal
to its card (today 2 drifts); every C row has a decision and the page shows it; zero hidden, off-canvas or overflow
text in rules.json (today 6 + 1 + 3); the checker passes on push to TTSCardMaker main; `/query_rulebook` returns only
text a player can see.

## 7. Known wrong paths (fenced off)

- Settling a contradiction in `engine.js`, in the rules skill or in this document: the page is the authority.
- Editing rules.json, stats.pkl or any pickle by hand: `/update_stats` rebuilds the pickles from the PSDs and
  `site.yml` rebuilds rules.json from stats.pkl.
- Hiding a layer instead of deleting it: hidden text is parsed today (`psd_parser.py:499`); H1 exists because of this.
- Case-insensitive status matching: "empty slot" (16_The_Field) is not the debuff Empty; 20 vs 19 names.
- Giving (XCD)/(XGCD)/(XU) meanings in `badge_tags()` or the skill before (02_Abbreviations) defines them.
- Reading effects out of ability text with regexes as if text were a schema (`docs/Campaigns/SIMULATOR_CAMPAIGN.md` Known wrong paths;
  the 2026-02-08 whitespace saga). The checker counts words; it does not interpret them.
- `MDW/Old Rules/` as a source for a decision: history only (C1 came from there).
- Patching SKILL.md in place with one-shot scripts (the ones that shaped it "were not kept", SKILL.md provenance):
  regenerate rulebook.md with `build_rulebook.py`, edit the skill by hand.
- Letting the P5 model or any agent write a decision; letting a GitHub Action edit a page.
- A watcher, flag or daemon for the checker: a script run by hand, then one Action step.

## 8. Open questions for the user

1. U1: define the three tags or rewrite the 44 cards? If defined, is CD "turns of this creature" and GCD "global turns"?
2. V1: delete the stray text box on (16_The_Field), or is it a deliberate joke the printed page keeps?
3. M1: "Smouldering" or "Smoldering"? The card file name follows the choice (rename = card edit).
4. C4: do status effects tick in phase 3 with (ENDT) effects or in phase 4? 09 and 19 need one answer.
5. Where does the checker live (TTSCardMaker `site/build/` or the rules skill's `scripts/`), and is a failing Action
   acceptable while the baseline still has rows?

## 9. Provenance (2026-10-20, read-only)

| Fact | Command |
|---|---|
| 604 / 35 / 351; 7 undefined tags, 59 occurrences, 44 cards | `python3 -I .claude/skills/impossibility-simulator-rules/scripts/tags.py ~/Desktop/TTSCardMaker/site/dist/data/cards.json` (+ a regex pass over (02_Abbreviations) with digits mapped to X) |
| status mentions, 10 unused | `python3 -I .claude/skills/impossibility-simulator-rules/scripts/mentions.py <cards.json>` |
| 6 hidden text layers on 4 pages | `~/.venvs/ac-py314-psd125/bin/python -I .claude/skills/impossibility-simulator-rules/scripts/layers.py ~/Desktop/TTSCardMaker/Auxiliary/Rulebook/ $(ls ~/Desktop/TTSCardMaker/Auxiliary/Rulebook/ \| grep '\.psd$')` |
| V1 layer, opacity, bbox | `... alllayers.py ~/Desktop/TTSCardMaker/Auxiliary/Rulebook/16_The_Field.psd` |
| V1 visible on the page | Pillow crop of `Auxiliary/Rulebook/16_The_Field.png` at (600,1560,1080,1720), viewed |
| 08 star icons are smart objects | psd-tools walk of `08_Global_Turn.psd`: `Text/L1-L3/Top` and `Bottom`, kind smartobject, visible |
| rules.json comes from stats.pkl; the Action's checkout | `sed -n '54p;127,132p;601p' ~/Desktop/TTSCardMaker/site/build/build_site.py`; `.github/workflows/site.yml` (sparse-checkout, `needs: build`) |
| 772 names / 767 cards; Gold Clot, Mr Krabs, Salt, Pix, Z!!!!! quotes | `python3 -I -c` over `cards.json` (`cards` list; names set; `tags.py` dedupe) |
| Quotes of C1-C4, H2, X1, X2 | `python3 -I -c` over `rules.json` (ids `21-damage-types`, `12-reactions3`, `07-stats3`, `19-status-effects`, `09-takingyourturn`, `14-item-types`, `20-creature-types`, page tails) |
| Parser treats rule-page text layers as ability text | `grep -n "is_rulepage" commands/psd_parser.py` (line 499) |
