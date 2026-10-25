---
name: ttscardmaker-companion
description: "Load for anything on the TTSCardMaker side (the card-art repo at ~/Desktop/TTSCardMaker) that AutoCroissant depends on. Trigger words include stoogedings, icon font, \"[fire]\" tokens, [x:tint], [x:sticker], [x:no], [x:glow], [x:darkglow], glow halo, ★ star glyph, Private Use Area / PUA codepoints (U+E000…), stoogedings.json manifest, build_stoogedings.py, rebuild the font, add/rename an icon in Types/ or Types/Stars/, timing tags and badges such as (1CD), (1GCD), (1/2CD), (2U), (OYT), (Action), the old (1PGT) / (E2T), badge size, type emojis, application emojis, the _no crossed-out emojis, /sync_emojis naming, the website, card browser, site/, build_site.py, site.yml, GitHub Pages, deck builder, Tabletop Simulator export, card-preview.yml / website card previews / the previews branch, the token picker (tokens.js, typing \"[\" or \"(\" in card text), variant pictures, the TTS save (card-sync.yml's step since 2026-10-24; tts-save.yml until then: TTS/decks.json, its daily gate, the save's Lua and teams, the TTS/checks suite, AUTOCROISSANT_LOCAL_DIR / AUTOCROISSANT_HISTORY_CLONE), share links /c/<slug>/, Open Graph previews, \"the website is stale\", missing fonts, font audit, \"Photoshop says fonts are missing\", Photoshop scripting, JSX, ExtendScript, ActionManager, migrating PSDs from image-layer icons to tokens, status-effect word effects (glow/stroke on \"Blazing\"), overlay text layers, effect layers \"Ability FX: …\", Effect Layers.jsx, Effect Layers - All Cards.jsx, Single-line vs Every-line Composer reflow, and the MDW folder's .asl/.aco/templates. Contains the font's token/badge grammar, manifest, codepoints and rebuild checklist, a cross-repo rename-an-icon table, the website's data flow and Pages setup, the missing-fonts audit, Photoshop recipes and the PSD migration status."
---

# TTSCardMaker companion: icon font, card website, fonts, Photoshop automation

AutoCroissant (the Discord bot) reads card PSDs from **TTSCardMaker** (`MichaelJSr/TTSCardMaker`, cloned at
`~/Desktop/TTSCardMaker`; a blobless partial clone that is also shallow, so `git log` there shows only recent
history; use GitHub, or `git fetch --unshallow --filter=blob:none`, for older commits). On 2026-10-19 three things
were added on the TTSCardMaker side that the bot now depends on or feeds: the **stoogedings** icon font, a static **card-browser website**, and a
**missing-fonts audit**. Photoshop scripting recipes were worked out along the way. This skill is their operational
home. Committed and pushed to TTSCardMaker `main`: 54278c1 (stoogedings font + `stoogedings-src/`, Effect Layers
scripts, missing card fonts), c3e9f66 (effect layers), 3ef3776 (`site/`, `.github/workflows/site.yml`,
`MDW/Card Authors.txt`), 81ca0ea..027fd0d (site features), 9fbe683..81dfc2a (site README; TTS save, `TTS/decks.json`
and tts-save.yml; printed card names; the Table tab 0d49893 was reverted by 75692fb), 15a16be (subseries), 13ac680
(website files `MDW/` -> `site/`, Action env overrides; head on 2026-10-20). The site is live (§2).

The parser side of tokens (how `psd_parser.py` reads a layer set in the font, plus the four new problems) is owned
by **impossibility-cards-reference §4.1**. This skill only summarizes it.

---

## 0. What lives where

| Thing | Path (TTSCardMaker unless noted) | Generated or source |
|---|---|---|
| The font | `MDW/Fonts/stoogedings.ttf` (5,721,356 bytes since 8ec1b92, 2026-10-21; 5,136,060 before; 5,736,204 with any-case badges; 6,966,396 with `:glow` (0.3 em halos) on branch work/glow-tokens, §1.8; 7,269,696 with the `salt` shapes, §1.9. PostScript name `stoogedings-Regular`, family `stoogedings`) | Generated, but committed |
| Build script | `MDW/Fonts/stoogedings-src/build_stoogedings.py` | Source |
| Manifest | `MDW/Fonts/stoogedings-src/stoogedings.json` (icon names, aliases, codepoints, tint modes) | Rewritten by every build. Hand edits survive |
| Build deps | `MDW/Fonts/stoogedings-src/requirements.txt` (fonttools, opencv-python-headless, numpy, pillow) | Source |
| Build venv | `MDW/Fonts/stoogedings-src/.venv/` (gitignored by `stoogedings-src/.gitignore`) | Local only |
| Icon sources | `Types/*.png` (35 "type" icons), `Types/Stars/*.png` (56 "star" icons) | Source art |
| Letter / badge fonts the build embeds | `MDW/Fonts/Bree Serif Regular.ttf` (OFL letters), `MDW/Fonts/Changa One.ttf` (badge labels) | Inputs |
| Website source | `site/index.html`, `site/assets/{app,search,deck,stats,text,edit,profile,pick,inline,details,decks,balance}.js`, `site/assets/style.css`, `site/build/build_site.py`, `site/build/analytics.py`, `site/build/requirements.txt`, `site/README.md` | Source |
| Website output | `site/dist/`, thumbnail cache `site/.cache/` (both gitignored by `site/.gitignore`) | Generated |
| Pages workflow | `.github/workflows/site.yml` | Source |
| TTS save | card-sync.yml's "TTS save" step since the cutover (2026-10-24): keeps `TTS/Impossibility Simulator.json` current with the bot's updater (`--names` from the new data), rules `TTS/decks.json`, the save committed in Card sync's main commit M by its main lane (30 minutes after the first card push of a burst, card-sync.yml's push timer since 2026-10-24; the data lane runs at once since 2026-10-25), the baked PNGs' links re-pinned by a second commit right after; `.github/workflows/tts-save.yml` did it until then and is deleted (§2.6, its history) | Source |
| Font check | a step of `card-guards.yml`'s check job since 2026-10-25 (`.github/workflows/font-check.yml` before; on a push changing a PSD or `MDW/Fonts/`: the fonts the pushed cards ask for that `MDW/Fonts` lacks, as a job summary, a commit comment and a Discord role ping in the standards channel, secret `DISCORD_STANDARDS_WEBHOOK` since the owner's ruling of 2026-10-22 (`DISCORD_FONT_WEBHOOK` before, now deleted); the only role ping of the checks' posts; never reverts; §3) | Source |
| Card check | a step of `card-guards.yml`'s check job since 2026-10-25 (`.github/workflows/card-check.yml` before, 2026-10-21, push guard rule 2: on a push changing a PSD (its `site/Card Edits.json` and `Types/` triggers dropped: it checks only pushed PSDs), the bot's parser and validator notes for the pushed PSDs, as a job summary, a commit comment and a Discord post at once with plain wording and no ping (owner, 2026-10-23); AutoCroissant `commands/card_check.py`, from the guards' sparse list `card_guards.SPARSE_FILES`, the `DISCORD_STANDARDS_WEBHOOK` secret (the standards channel); never reverts; §3) | Source |
| The psd-tools fork in three workflows | `card-guards.yml` (its check and post jobs; the card and font checks run in its check job since 2026-10-25; card-edits.yml too until it was deleted at the cutover, 2026-10-24), `card-sync.yml` (one venv; until 2026-10-24 the builder's 3.14 and the baker's 3.12), `card-preview.yml` (and card-check.yml and font-check.yml from 2026-10-24 until they were folded into the guards on 2026-10-25), one local composite action `.github/actions/cards-python` for the setup since 2026-10-25, all on Python 3.14 like every Action: since 2026-10-23 they install AutoCroissant `requirements-cards.txt` (the one pin: the fork's tag is only there; fonttools, requests, pyoxipng, built from its sdist on the runners' Linux x86_64 3.14, but not installed in card-preview.yml, whose PNG is never published), checked out of the bot repo in each one's sparse list; no pin of their own (TTSCardMaker d3b744c7 had env `PSD_TOOLS_REF`). The packages install offline from wheels in the Actions cache (`~/.cache/card-wheels`, key `cards-<hashFiles('bot/requirements-cards.txt')>-<OS>-<arch>-py<Python>`, one entry for the six since 2026-10-24; card-sync's named both its Pythons before), built by `pip wheel -r` only on a miss and saved once they installed; the install goes by name (`sed 's/ @ [^;]*//'`: pip builds a URL line from git again, offline or not), so a re-pin in AutoCroissant refreshes every entry. The install was 29-35 s a run on 2026-10-23 (the git clone and the C build ~25 s of it); a hit skips both. The two repos' changes land together. Re-pin: autocroissant-config-and-flags; shape test: AutoCroissant `tests/test_card_bake.py` `fork_install` | Source |
| Card guards workflow | `.github/workflows/card-guards.yml` (landed 2026-10-23 in TTSCardMaker a7d1fe56; the repo variable `CARD_GUARDS` is `report` since 2026-10-23 13:04 after a full dry run (812 cards, 16 told, 0 errors), `on` after ~3 days (owner); unset or off = the guards' steps skip, the card and font checks still run): on a push changing a PSD or PNG, a PSD pushed without its PNG gets the PNG the bake would make (owner's ruling 2026-10-22: the fork at `cards-20261023` renders the changed layers in the card's ICC export mode, merged into the old PNG; Photoshop's stored composite only cross-checks and never is the PNG; a refusal is told "PNG not made: ...; export it in Photoshop"; since the owner's ruling of 2026-10-23 the fork's drawing refuses only past 25% bad (a card drawn whole, new or beside its PNG, per text and stat box, and a missed box drawn blank refuses too), a smaller miss is made and told in the comment and the bot commit; 29 of the 34 replayed commits get their PNG, the rest refuse on stale old PNGs and type rasters Photoshop redrew), and the pushed PSDs are checked against the card standards (fonts, R7, dropped 2026-10-24: font-check.yml's alone; a PSD with no stored image is rendered anyway and told "no stored image to cross-check", branch work/py314) (Txt2 sync fixed, the owner's one easy fix; the rest reported, exact layer names too until the owner says yes; R3 and R4 without their template layout, dropped by the owner on 2026-10-23: several visible Ability layers, a hidden one with words the parser reads nowhere, a hidden or rasterized Name layer; kept unattended, for the owner to confirm; the parser's `ABILITY_EXCLUSIONS` cards, Shadow_Duelist, are waived); one Discord post per push to the standards channel (`DISCORD_STANDARDS_WEBHOOK`, no role ping, no previews), a commit comment, writes landed in one bot commit after 30 quiet minutes (since 2026-10-24, branch work/py314: a post job waiting in the environment `card-guards-quiet`'s 30-minute timer, no runner sleeping; three jobs: check, post, finish; since 2026-10-25 the Single-line Composer (R8) is always set, unsaid, box text that rewraps redrawn and its PNG re-exported, and a landing's Follow up starts only `card-sync.yml -f bake=pending`, whose data commit builds the site). AutoCroissant `commands/card_guards.py`, `png_export.py`, `card_standards.py`; sparse list `card_guards.SPARSE_FILES`; AutoCroissant `docs/Campaigns/PUSH_GUARDS.md` | Source |
| Card preview workflow | `.github/workflows/card-preview.yml` (2026-10-23, landed in TTSCardMaker 2150d0ed and AutoCroissant 8983719: dispatch only, inputs `card`, `edit`, `request`): one card's website edit baked as the bake would (AutoCroissant `commands/card_preview.py`, `card_bake.bake_card`, in a read-only job), nothing written to main; a second job (`commands/card_preview_branch.py`, standard library, the only one that writes) puts the result on the orphan branch `previews` (`previews/<request>/result.json`, `preview.webp`, `before.webp`, `index.json`), rewritten past 200 previews or 7 days; no secret beyond GITHUB_TOKEN, no Discord; public like the repo; the website's contract is §2.7 | Source |
| Installed copy of the font on the dev Mac | `~/Library/Fonts/stoogedings.ttf` (identical to the repo build on 2026-10-19) | Install |
| Bot code that depends on the font | AutoCroissant `commands/psd_parser.py` (`ICON_FONT_PREFIX`, `IconVocabulary`), `global_config.py` (`ICON_MANIFEST_PATH`), `commands/type_emojis.py` | See §1.7 |
| Card authors file | `site/Card Authors.txt` (in `MDW/` until 2026-10-20, like `Card Edits.json`, `Website Admins.txt` and `Constructed Ban List.txt`; moved by 13ac680, read from `site/` by the build and the bot): `[people]` git name = display name, `[cards]` Card = person; one person's several git names are merged first by the root `.mailmap` (2026-10-23: Michael -> MichaelJSr, its old `[people]` line gone; build_site reads `%aN`/`%aE`, the bot `%aN` with head's .mailmap blob). The bot (every `/update_stats`) and the website build both apply it; it wins over the database. Seeded 2026-10-20 from stats.pkl | Source, committed (3ef3776) |
| Effect-layer scripts | `MDW/Scripts/Effect Layers.jsx` (one card: Add / Update all / Convert overlays), `MDW/Scripts/Effect Layers - All Cards.jsx` (every card) | Source, committed (54278c1, last changed c3e9f66) (§6) |
| Status styling sources | **Authority:** `MDW/Croissant's Special Folder/Satus Effect Colors.psd` (66 statuses, 15 styled; the scripts read colors and copy styles from it). Fallbacks: `MDW/Debuff And Attributes Layer Effects.asl` (13 styles: 10 status, 3 attribute glows), `MDW/Status Effects.aco` (66 status text colors), `MDW/Croissant's Special Folder/Satus Effect Colors.psd` (chart of all 66, incl. 5 styles missing from the .asl) | Source art |
| Scratch work: audit outputs, Photoshop test JSX, migration prototype | `/private/tmp/fontexplore/` (**volatile: macOS wipes /tmp on reboot; nothing there is committed**) | Scratch |

## 1. The stoogedings icon font

### 1.1 What authors type

Set the text in the **stoogedings** font in Photoshop. These ASCII sequences turn into pictures through ligatures:

| Typed (in stoogedings) | Renders as | Parser stores (impossibility-cards-reference §4.1) |
|---|---|---|
| `[fire]` | The icon in its own colors (an OpenType-SVG glyph with the PNG embedded, 256 px long side) | `[fire]` |
| `[fire:tint]` | One-color silhouette that takes the **text color** | `[fire]` |
| `[fire:sticker]` | The color icon on an outline in the text color | `[fire]` |
| `[fire:no]` | The icon with a red prohibition sign (`#e01b22`) over it | `[fire:no]` (means "non-fire") |
| `[fire:glow]`, `[fire:darkglow]` (branch work/glow-tokens, §1.8) | The color icon on a soft white or black halo, 0.3 em wide | `[fire]` |
| `★` (U+2605) | The gold `star` icon | `[star]` |
| A PUA character (U+E000–U+E05A) inserted from the Glyphs panel | That icon | `[name]` |
| `(1GCD)`, `(1/2CD)`, `(OYT)`, `(Action)` and the other tags (341, §1.2), in any letter case | A colored pill badge, the tag's own one (`(1gcd)` draws (1GCD)). The text is still what was typed | The text, unchanged (`(1gcd)` stays `(1gcd)`) |
| A badge or `[fire:sticker]` with **Stylistic Alternates** on (Character panel, OpenType; §1.9) | One colour, in the text color: a pill outline with solid letters; a thin outline round the `:tint` icon | Unchanged |

Matching rules inside the font (build_stoogedings.py `spellings()` and the MARK lookup):
- Matching is **case-insensitive only inside brackets**. After `[`, every character maps to a lowercase `.tk` copy, so
  capitals elsewhere in the text stay capitals.
- Space, `_`, `-` and no separator are interchangeable: `[Defensive Minion]`, `[defensive_minion]`,
  `[defensive-minion]` and `[defensiveminion]` all work.
- **Dots and apostrophes are NOT ignored by the font.** They only match through an alias (e.g. `n.m.e`).
- Only ASCII 0x20–0x7E is available (the Bree Serif subset). A token that needs another character is skipped at build
  time with "unsupported character in token".
- An unrecognized token, such as `[fier]` or `[X]`, falls back to plain Bree Serif letters. That fallback is
  deliberate, so a typo is visible on the card.
- **Badges match in any letter case** (since 2026-10-22, the CASE lookup, §1.2): `(1gcd)`, `(Bogt)`, `(action)` and
  `(ACTION)` draw the `(1GCD)`, `(BOGT)` and `(Action)` badges. Only a whole tag counts: `(Actions)`, `(Action` and
  `(x cd)` stay letters, as typed. Before that date only the exact case became a badge.
- A tag inside an unclosed token still becomes a badge: `[fire(1CD)]` shapes as `bracketleft f.tk i.tk r.tk e.tk
  badge_1CD bracketright` (checked 2026-10-21; the legacy tags behaved the same before; `[fire(1cd)]` too since 2026-10-22).

**Specimen QA (2026-10-20, Photoshop 2023):** sheets at `/private/tmp/fontexplore/spec/` (scratch). No defects found:
- **Variants:** all 91 icons render in all four variants; all 70 badges render in their group colors (the 270 cards'
  tags of 2026-10-21 are not checked in Photoshop yet).
- **Body fonts:** they sit well next to Century, FOT-PopHappiness, Perpetua, EB Garamond and PixelMplus at those
  fonts' card sizes.
- **Edge cases that work:**
  - ligatures off: tokens still form, because LAYERS and BADGES are also in `rlig`;
  - faux bold and tracking 200;
  - a token split across two color runs (`[pan` + `da]`);
  - any case and every separator (`[Fragile_Equip]`, `[fragile-equip]`, `[fragileequip]`) and aliases (`[dark matter]`, `[N.M.E]`);
  - sizes 24–192 px;
  - auto and fixed leading.
- **Not tokens:** `[fier]`, `(1PG`, `(ABC)`, `[fire:bogus]` stay text.
- **Scaling:** a text layer with a 3× transform renders icons exactly like an unscaled one (1 em tall).

Notes:
- **Icon size:** about 7% smaller than typical hand-placed icons on Gasoline (flame 85 px vs 91 px at 115 px text).
  It was kept, because cards use tight fixed leading (about 0.74 em) and bigger icons would hit the lines around them.
- **lightshadow:** `Types/lightshadow.png` differs from `light.png` only by a faint dark halo; that's the source art.
- **Tags without badges:** the commonest is `(n points)` (68). It isn't a rulebook abbreviation, so it gets no badge
  unless the owner wants one. `(Action)` (36) has had one since 2026-10-22 (owner decision, §1.2).

### 1.2 Timing-tag badges (341: the rulebook's 70 + the cards' 270 + `(Action)`)

**Badges are part of the font going forward** (the user adopted them on 2026-10-19). `badge_tags()` in
build_stoogedings.py defines two vocabularies and one extra tag:
- **Rulebook (old system), 70:** the tags of page "02 Abbreviations", 10 fixed tags plus 6 patterns × 10 values of n
  (1–9 and X). They stay: on the cards they were replaced on 2026-10-21, but the pages 02_Abbreviations,
  11_Abilities2, 11_Abilities4 and 22_Negation still use them.
- **Cards (2026-10-21), 270:** the team's cooldown update (TTSCardMaker df02b3b..7cb4c86; tags and counts in
  impossibility-simulator-rules §3). `nCD`, `nGCD`, `nU` over the same 10 values of n = 30, plus `a/bCD` and `a/bGCD`
  with a in -1, 0–9, X (12) and b in 1–9, X (10) = 240. The a/b range covers every two-number tag on the cards:
  `(1/2CD)` (15 PSDs) and `(-1/1GCD)` (The Temple of Ahn'Qiraj).
- **`(Action)`, 1 (owner decision 2026-10-22):** on 28 cards. Any case draws it (below); `(Actions)` stays text.
  Not a timing tag, so it has its own grey `neutral` group; label "Action" in mixed case, 3130 units wide
  (about the width of `badge_ENDGT`, 3116), readable at that width.

Size of the 340 (committed in TTSCardMaker 8ec1b92, 2026-10-21): 340 `badge_*` glyphs, 522 SVG documents,
5,721,356 bytes ("5.7 MB") vs 5,136,060 with 70 badges (+585,296 bytes, +11.4%), build about 15 s. The /tmp trial
before it was 6,021,116 bytes; the shipped build rounds SVG path numbers to 0.1 unit (`num()`), which saves 300 KB.
That is not unreasonable, so the a/b range was not narrowed. Widest badges (1000 units per em): `badge_m1_1GCD`…`badge_m1_9GCD`
4133, `badge_X_XGCD` 3575; widest old one `badge_ENDGT` 3116. Wider, as expected, not a defect.
With `(Action)` (TTSCardMaker f6ce500, 2026-10-22): 341 badges, 523 SVG documents, 5,723,428 bytes (+2,072), cmap and
icon codepoints unchanged, rebuild byte-identical.

**Any letter case (owner request 2026-10-22, TTSCardMaker branch work/badges-any-case):** every tag also matches in
any case and draws the same badge: `(action)` and `(ACTION)` are (Action), `(1cd)` is (1CD), `(-1/xGcD)` is (-1/XGCD).
The text stays as typed. A CASE lookup (chaining contextual) runs just before BADGES in `liga` and `rlig`. It has one
rule per tag from `badge_tags()`: `(`, then each letter as `[own case, other case]`, then `)`. Each letter goes through
FOLD_UP or FOLD_LOW (single substitutions, applied only from CASE) into the tag's own case, and BADGES then forms the
ligature as before. A rule matches only a whole `(tag)`, so other text in parentheses is never re-cased, and an
exact-case tag goes through unchanged. Each rule is its own subtable (`subtable;`), so it is Format 3 like MARK's;
left alone, feaLib would merge them into one Format 2. The build asserts that no two tags differ only in case.
Size: 5,736,204 bytes (+12,776). GSUB grew from 49,724 to 62,498 bytes; every other table is byte-identical, glyphs,
SVG, cmap and name included.
Proof, 2026-10-22, offline on the built font (scripts in /tmp/anycase-work, scratch):
- HarfBuzz: 341 tags × every letter-case combination = 2530 spellings, each under default features, `liga` off and
  `rlig` off, all shape to their canonical badge glyph with its advance. The 341 canonical tags shape byte-identically
  to the committed font. 17,739 non-tags (every spelling without one paren, with a `z` or a space added, in `[ ]`)
  draw no badge and shape as before.
- Every ability, note and name in stats.pkl (1616 texts), plus `Card Edits.json`, shapes exactly as before except
  A P E X P R E D A T O R's ability. Its `(action)` is the only other-case tag on today's cards.
- CoreText (a Swift tool loads the file, no install): the same 2530/2530, 12,670 near misses unchanged.
- Non-ASCII letters that Unicode folds to a tag (`(oſ)`, `(Actıon)`, `(Actİon)`) draw no badge: the case folding is ASCII only.
- Short tags in prose: any case now badges `(os)`, `(od)`, `(Os)`, `(xu)`, `(e1t)`, `(1u)` and the like wherever they are
  typed, as asked. None occur in today's cards or rulebook text; keep it in mind when writing card text.
- Photoshop: **not confirmed yet** (another workflow owned Photoshop). Run `~/Documents/Tools/psd-research/std/case_badge_check.jsx`
  with the new font installed and Photoshop restarted. It writes `case_badge_check.log` and `.png` next to itself.
  A CoreText simulation of its rules passes on the new font and fails exactly the other-case rows on the old one.
  If those rows FAIL in Photoshop, Photoshop ignores CASE: apply `std/case_canonical_tag_fallback.patch`. The walk
  then writes the tag in its badge's case, which changes the card's text (a PSD + database edit).

Website: `tagInfo(typed)` returns `{tag, ...}` with `tag` in the canonical case (`canon`: upper case, or the `FIXED` key
for Action). `badgeHtml` labels and tips the badge with it, and the Stats page counts under it. The walk: still
exact-case until `std/case_insensitive_tag.patch` is applied (not applied on 2026-10-22, while std-batch2 was running).
Its header lists the driver lines a later pass needs (`std_driver.py` TAG and `badge_adv`, `std_card.jsx` STD_TAG).

Glyph names: `badge_` + the tag with `/` → `_` and `-` → `m` (`badge_1_2CD`, `badge_m1_1GCD`; `badge_glyph()`).
The label draws `-` as a real minus sign U+2212 (`BADGE_LABEL`); the website does the same. `(OPT)` (Bobberlington, Michael) is in neither vocabulary
and stays text.

The colors are the rulebook's groups (`BADGE_GROUPS`, page "02 Abbreviations"). The cooldowns sit beside the old
tags they replaced; `nU` (which replaced the n of `nPT`/`nPGT` on some cards) sits beside `nPG`, the other counted
tag, in the trigger group (orchestrator decision 2026-10-21; the owner may revise):

| Group (color) | Rulebook tags (old system) | Cards' tags (2026-10-21) |
|---|---|---|
| turn, on your turn (blue `#4f86f7`→`#2563eb`) | `OYT`, `nPT`, `EnT`, `FTEnT` | `nCD`, `a/bCD` |
| any, at any time (purple `#9f67ff`→`#7c3aed`) | `AAT`, `nPGT`, `EnGT` | `nGCD`, `a/bGCD` |
| forced (red `#ef5050`→`#dc2626`) | `BOT`, `ENDT`, `BOGT`, `ENDGT` | — |
| trigger, per game and triggers (orange `#e08a1e`→`#b45309`) | `OFS`, `OS`, `OD`, `WOF`, `nPG` | `nU` |
| neutral, not a timing tag (slate grey `#64748b`→`#475569`) | — | `Action` |

Grey for `(Action)` because it says what kind of ability it is, not when it can be used: a fifth hue would read as a
fifth time frame, and grey stays clear of the four group colors (white label contrast about 7.6:1 on `#475569`).

Website tooltips (`tagInfo()` in text.js) follow the letters only, no invented rules: "Cooldown: 2 turns", "Global
cooldown: 2 global turns", "Cooldown: 1 turn, then 2", the same in global turns for `a/bGCD`, "2 uses" (X stays
"X"). `count()` gives the singular for 1 and -1 ("Cooldown: 1 turn", "1 use", "Global cooldown: -1 global turn,
then 1"). Some read oddly because they are literal ("0 turns, then 1", "X global turns, then X"). The old tags keep
their meanings, prefixed "Old system: ". `(Action)`'s tooltip is "Action"; the rules glossary lists it under
"Other tags" (`GROUP_NAMES.neutral`). The site reads ability text from stats.pkl, so the new badges appear there only after an
`/update_stats` + `/push` and a site rebuild (2026-10-21: stats.pkl was PICKLE 56b43b9, before the update; 175 of
the 178 changed cards still showed old tags).

Shape (revised 2026-10-19, confirmed in Photoshop 2023 on 2026-10-20 with the reinstalled font): a pill about
0.95 em tall with label capitals about 0.55 em, the label centred, color only — no extra symbol on multi-turn
tags, in the font or on the website (an earlier extra mark read as an "I" and was dropped). The exact sizes
and spacing are the `BADGE_*` constants in build_stoogedings.py (pill top/bottom, capital height, padding,
`BADGE_TRACKING` letter spacing, border, side); read them there rather than quoting numbers, and rebuild (§1.4)
after changing them. Centring pitfall fixed on 2026-10-20: tracking must go between letters only — adding it
after the last letter pushes the label off centre.

**The tag vocabulary is defined in three places that must change together**, plus the batch copies below. All three
carry the same 341 since 2026-10-22 (TTSCardMaker f6ce500); `tests/test_badge_parity.py` checks them, the walk part only where it exists,
and every case of every tag: all 2^letters combinations, 2530 spellings (the font by HarfBuzz: `uharfbuzz` where importable,
e.g. `~/.venvs/ac-py314-fork`, else the `hb-shape` tool, so it runs in `ac-py314-psd125` too; the walk
test skips "still exact-case" until the patch is applied, and "clone behind" for an any-case walk with an older font):
- `badge_tags()`/`BADGE_GROUPS` in build_stoogedings.py
- `FIXED`/`tagInfo()` in `site/assets/text.js`, plus the `--b-*` colors and `.badge` styles in `site/assets/style.css`.
  Candidates come from text.js's exported `TAG` (`/\((-?[0-9A-Za-z/]{2,6})\)/` since 2026-10-22, any case; no /g flag since
  73904db so it holds no `lastIndex`); the renderer's `PIECE` is built from `TAG.source` and stats.js iterates `new RegExp(TAG.source, "g")`
- the `TAG` regex of the migration walk (§5; `~/Documents/Tools/psd-research/std/std_l1.jsx:9`, also `std.jsx:8` and
  `l1_check.py:14`), which includes `(?:-1|[0-9X])\/[1-9X]G?CD|[1-9X]G?CD|[1-9X]U` and `Action`; not in git, pre-`Action` copies in
  `std/pre-action-2026-10-22/`; any case once `std/case_insensitive_tag.patch` is applied (`/i`, and `(?ai:...)` in l1_check.py: ASCII case only, like
  JS `/i` and the font; plain `(?i:...)` would also take `(oſ)`, `(Actıon)`, `(Actİon)`, which the test probes)

A **fourth copy** lives in each std batch folder (`~/Documents/Tools/psd-research/std-batch-*/`: `scripts/std_card.jsx`
`STD_TAG` and `tools/std_driver.py` `TAG`). `test_batches_match_font` checks it, allowing `BATCH_LAG` (std-batch-2026-10-22
started before `Action` and lacks it). Both walk tests skip, saying "clone behind", when the walk has a tag the clone's font
lacks (the clone is `AUTOCROISSANT_LOCAL_DIR`, default ~/Desktop/TTSCardMaker): pull it, or point the variable at a newer checkout.
**Operating rule:** run the walk (`std.jsx`, `std_l1.jsx`) only with a ~/Library/Fonts/stoogedings.ttf that has every tag in
its `TAG`. With an older font, `(Action)` is still set in stoogedings but draws as plain letters, not a badge.
The same goes for an any-case walk with a font from before CASE: `(action)` is set in stoogedings and drawn as letters.
The website's Stats page "Timing tags" chart (stats.js) leaves out the `neutral` group, so `(Action)` is not counted there.

### 1.3 Manifest schema and codepoints

`stoogedings.json` = `{"icons": {<key>: {kind, file, pua, aliases, tint}}}`, sorted by codepoint. There are 91 entries
as of 2026-10-19 (35 `type` + 56 `star`), using codepoints **U+E000–U+E05A, contiguous**.

| Field | Meaning |
|---|---|
| key | For Types/, the lowercased file stem (spaces kept: `fragile equip`). For Types/Stars/, the stem lowercased with spaces → `_` (`dst_star`). These are the same keys AutoCroissant uses (`all_types`; `_star_name`). |
| `kind` | `"type"` or `"star"` |
| `file` | The repo-relative PNG path. It is refreshed on every build. |
| `pua` | A fixed hex codepoint. **New PNGs get max(used)+1, and existing codepoints never move** (`load_config`, :95). |
| `aliases` | Extra spellings. They are seeded from `DEFAULT_ALIASES` (:56) **only the first time a key is discovered**. After that, edit the JSON. 15 icons have aliases (e.g. `thunderfury`, `barnstar`, `yin yang`, `n.m.e`, `dark matter`). |
| `tint` | How the `:tint` silhouette is cut: `"silhouette"`, `"auto"`, or `{"mode":"detail","cut":"dark"}`. Current split: 61 auto, 26 silhouette, 4 detail. Seeded from `DEFAULT_TINT` (:72). |

Entries are **never deleted**. If a PNG disappears, the build prints "missing file, codepoint kept reserved" and
skips the glyph, but the entry stays.

### 1.4 Rebuild and install checklist

Do this after any change to Types/ or Types/Stars/, or to aliases or tint modes:

1. One-time setup: `python3 -m venv MDW/Fonts/stoogedings-src/.venv && MDW/Fonts/stoogedings-src/.venv/bin/pip install -r MDW/Fonts/stoogedings-src/requirements.txt`
2. Build (no arguments; it finds the repo from its own location; about 13 s, with `:glow` about 32 s of CPU):
   `MDW/Fonts/stoogedings-src/.venv/bin/python MDW/Fonts/stoogedings-src/build_stoogedings.py`
   Expected output today: `built …/stoogedings.ttf: 5.7 MB, 91 icons, 1024 token spellings` (with `:glow`:
   `7.0 MB, 91 icons, 1536 token spellings`).
   Read any "token collision" or "missing file" lines.
3. Rebuilds are **byte-identical** (`recalcTimestamp = False`, :514), so `git status` shows the font as changed only
   when something really changed. Verified 2026-10-19: a rebuild of a /tmp copy matched both the committed TTF and
   the JSON byte for byte. Why `/tmp/fontexplore/venv` once packed GSUB differently from `~/.venvs/ac-py314-psd125`
   (2026-10-21): fontTools uses HarfBuzz's repacker whenever `uharfbuzz` is importable, and only that venv has it.
   Since 2026-10-22 the build pins the pure-Python packer (`font.cfg[USE_HARFBUZZ_REPACKER] = False`), so every venv
   gives the same bytes: both venvs rebuilt f6ce500's TTF exactly, and the any-case build identically. The write is
   atomic (`.tmp` + `os.replace`).
4. Install: copy the TTF to `~/Library/Fonts/` (or open it in Font Book), replacing the old copy. Photoshop pitfalls
   are in §4.
5. In TTSCardMaker, commit the PNG(s), `stoogedings.ttf` and `stoogedings.json` together (the script's docstring
   says to). After that, the bot side needs no code change. The parser reads the manifest from GitHub on each
   `/update_stats` (`_icons_from_response` via `_repo_text`, re-downloaded only when its blob changes; the offline
   diagnostics scripts read the clone's copy with `_icons_from_local`). The next `/update_stats` uploads emojis for new icons
   (so do startup and `/sync_emojis`).

### 1.5 How it is built, and the renderer facts behind the design

GSUB is registered under **both `liga` and `rlig`**. There are eight lookups, in this order (FOLD_UP, FOLD_LOW and CASE since
2026-10-22; ten with GLOW_TOKENS and GLOW_LAYERS on branch work/glow-tokens); TO_TK and the FOLDs run only from the
contextual lookups:

| Lookup | Type | Job |
|---|---|---|
| TO_TK | Single | char → `.tk` copy (capitals → lowercase copy) |
| MARK | Chaining contextual | applies TO_TK after `[` |
| TOKENS | Ligature | `[`+`.tk`…+`]` → icon glyph (longest first) |
| LAYERS | Multiple | `:sticker` → [outline][color icon][spacer]; `:no` → [color icon][zero-width red sign] |
| FOLD_UP, FOLD_LOW | Single | letter → its other case (only the letters the tags use) |
| CASE | Chaining contextual | `(` + a tag in any case + `)`: each letter through FOLD_UP/FOLD_LOW into the tag's own case; 341 Format-3 subtables |
| BADGES | Ligature | `(` tag `)` → badge |
| GLOW_TOKENS, GLOW_LAYERS (§1.8) | Ligature, Multiple (Extension type 7) | TOKENS and LAYERS for `:glow` / `:darkglow`. Last, so the eight above keep their bytes |
| SALT_BADGES, SALT_STICKERS (§1.9) | Single, Ligature | Under `salt` only: badge → `badge_*.salt`; a `:sticker`'s [outline][icon][spacer] → `icon_*.sticker.salt` |

Every icon also gets a plain TrueType outline in `glyf` as a fallback for renderers without SVG support. The SVG table
holds 523 documents (91 icons + 91 no-signs + 341 badges; 522 with 340 before 2026-10-22, 252 with 70 before 2026-10-21;
705 with the 182 halos of §1.8).

Renderer facts. These were tested 2026-10-19 in Photoshop 2023 and macOS CoreText. They are also recorded in the
build script's docstring:

| Fact | Design consequence |
|---|---|
| Photoshop renders OpenType-SVG glyphs, including embedded PNGs | Color icons are SVG glyphs, each embedding its own PNG |
| Photoshop paints SVG `currentColor`, `context-fill` and CPAL `var()` **black**. Only plain outlines take the text color | `:tint` and the `:sticker` outline are plain TrueType outlines, not SVG |
| Photoshop draws overlapping glyphs in text order | `:sticker` = outline glyph then icon (both through GSUB multiple substitution). No PNG is stored twice |
| GSUB single, ligature, multiple and chaining-contextual lookups all work in Photoshop under `liga` (CASE's multi-letter contexts: not checked yet, `case_badge_check.jsx`) | The whole design relies on liga. Leave ligatures on in the Character panel |
| CoreText (Apple apps) draws plain outlines **above** SVG glyphs | In Preview, Pages and similar apps, `:sticker` shows only the outline. Photoshop is the target, so this is accepted |
| CoreText ignores an SVG `<use>` that points at an `<image>` | Each color glyph embeds its PNG directly |
| CoreText draws an SVG glyph whole, past the line box (2026-10-23, the 0.3 em halos: clipping to the line box changes 0 pixels) | Photoshop draws it whole too (glow test 2026-10-23), so nothing clips the halo of §1.8; its `glyf` (the whole halo) only serves non-SVG renderers |

### 1.6 Four consumers, four matching rules

The same token is interpreted in four places. Since 2026-10-20 the font and the parser accept exactly the
same spellings; the website is still more lenient:

| Consumer | Normalization | Aliases | Notes |
|---|---|---|---|
| Font (GSUB) | Case-insensitive inside `[]`. Space, `_`, `-` or nothing interchangeable | Baked in at build | Strictest |
| AutoCroissant parser, `IconVocabulary.spellings` / `lookup` (psd_parser.py) | Same as the font (2026-10-20): the name's or alias's words joined by one space, `_`, `-` or nothing; exact lowercase match; no spaces around `:variant`; the variants are `ICON_VARIANTS` (`glow`, `darkglow` on branch work/glow-tokens), all but `no` read as the plain icon | Read from the manifest at parse time | Anything the font would draw as letters (`[fragile.equip]`, `[ fire]`, `[fire: no]`) is an `UNKNOWN ICON TOKEN` problem. Before 2026-10-20 the parser also dropped `.`, `'` and extra spaces (more lenient than the font) |
| AutoCroissant emojis, `type_emojis.emoji_name` | Lowercase. Drops `.`, `'` and `’`. Other non-alphanumerics → `_`. Max 32 characters, cut at a word boundary | **None**: it relies on the parser having already canonicalized the token in stats.pkl | `[x:no]` → `x_no` emoji, else 🚫 + emoji. `[X]`, `[Status]` and other unknown brackets stay as text |
| Website, `norm()` in build_site.py:102 and text.js:4 | Lowercase. Drops whitespace, `_`, `-`, `.`, `'` and `’` (the parser's pre-2026-10-20 rule) | Read from the manifest at build time | `[x:no]` → icon wrapped in `.tok-no`; `:tint`, `:sticker`, `:glow` and `:darkglow` are drawn with build_site's variant pictures (branch work/token-picker, 2026-10-23, §2.7; the plain icon without them, as before), and `share_text` (build_site.py) flattens `no|tint|sticker|glow|darkglow` in any ASCII case (2026-10-23). More lenient than parser and font on purpose (its docstring says so); harmless for display, since it only reads parser output |

Emoji names differ from manifest keys in exactly two cases: `fragile equip` → `fragile_equip`, and
`thunderfury_blessed_blade_of_the_starseeker` → `thunderfury_blessed_blade_of_the`. The 126 emojis are
35 type + 56 star + 35 `_no` (only Types/ icons get `_no`). This was reproduced 2026-10-19 by running the
`emoji_name` logic on the live listing. The emoji constants are owned by **autocroissant-config-and-flags**.
Application emojis belong to whichever bot identity is logged in, so each identity (dev Mac vs production) syncs its
own set. Live Discord upload is untested as of 2026-10-19.

### 1.7 Cross-repo changes: what breaks

| Change in TTSCardMaker | Font | Bot parser | Bot emojis | Website |
|---|---|---|---|---|
| **Add** a PNG | New codepoint after rebuild | Picked up automatically (Types/ listing + manifest) | Uploaded by the next `/update_stats`, startup or `/sync_emojis` | Picked up on next build |
| **Rename** a PNG (e.g. `fire.png` → `flame.png`) | New key and new codepoint. The old key stays, with its glyph skipped. Existing PSDs typed `[fire]` **show letters** | Old `[fire]` tokens become `UNKNOWN ICON TOKEN` (since 2026-10-20 `IconVocabulary.build` ignores manifest entries whose PNG is gone), matching the letters the card now shows. `card.types` names change for image-layer icons | Old `fire`/`fire_no` emojis are deleted by the next `/update_stats` (older ones not in `type_emojis.json` need `/sync_emojis prune:True`). Until then stored `[fire]` tokens keep rendering with the old emoji | `[fire]` in ability text loses its icon (icons are built from the tree, not the manifest) |
| **Delete** a PNG | Glyph skipped, codepoint reserved | Its tokens are flagged `UNKNOWN ICON TOKEN` | Deleted by the next `/update_stats` | Icon gone |
| Add an **alias** (edit the JSON, rebuild) | New spellings | Accepted at next `/update_stats` | n/a (works on canonical names) | Search/render accept it |
| Add a **timing tag** | Edit `badge_tags` | Nothing: text stays `(TAG)` | n/a | Edit `text.js` `tagInfo` and `style.css` |

Rule: **never rename or delete an icon PNG to "fix" a name. Add an alias instead.** If you must rename, the change
spans TTSCardMaker (PNG, rebuilt font and manifest, every PSD that uses the old token or layer name) and the bot
(`/update_stats`, which also deletes the old emojis). Class (d) gates apply: autocroissant-change-control.

### 1.8 Glow variants `:glow` and `:darkglow` (owner ruling 2026-10-22; branch work/glow-tokens, 2026-10-23)

`[x:glow]` is the color icon on a soft white halo, `[x:darkglow]` on a soft black one, for all 91 icons. Same grammar
as `:sticker`: any case, every separator, aliases. The parser reads both as plain `[x]` (§1.6); the website draws them
(and `:tint`, `:sticker`) with pictures made by the font build's own code since branch work/token-picker (§2.8).
Merged 2026-10-23 (TTSCardMaker and AutoCroissant branches `work/glow-tokens`).

**The model.** The owner's reference is Nebulous_Precipice's darkmatter icon layer, at origin/main blob 81bde4f7
(before the standardization hid it). Its effect is a Drop Shadow: white, Normal, opacity 80%, distance 0, spread 20%,
size 120 px, linear contour, layer knocks out shadow. The icon is 91×93 px and the text 95 px per em. On the card
(original PNG against the standardized one, where the layer is hidden) the glow's alpha is 0.46 at the icon's edge,
0.33 at 20 px, 0.20 at 40 px, 0.10 at 60 px and 0 by about 110 px. This emulation fits it with rmse 0.002: grow the
matte by spread × size, blur with a Gaussian of sigma 0.42 × (1 − spread) × size, multiply by the opacity.

**The size: 0.3 em (owner, 2026-10-23).** At token size the card's glow reaches 1.26 em. The first build cut it to
0.1 em (`GLOW_SIZE` 100), on the belief that Photoshop clips SVG glyphs to the line box. The Photoshop glow test of
2026-10-23 (`~/Documents/Tools/wt/scratch/rulings-0300/ps-results-1023/`: `glow_big_check`, `glow_1263_check`) drew
0.3 em and 1.26 em halos whole, past the line box. The owner then chose 0.3 em: `GLOW_SIZE` 300 units, the same spread,
opacity and model. The vertical metrics stay as they were.
`~/Documents/Tools/wt/scratch/rulings/glow/glow_size_comparison.png` compares the card, 0.1, 0.3 and 1.26 em.

**The glyphs.** Per icon: `icon_x.glow`, `icon_x.glow_bg`, `icon_x.darkglow`, `icon_x.darkglow_bg`, and one shared
`glow_space`. A token draws [halo, advance `GLOW_SIZE` − `SIDE`][color icon][`glow_space`]. It is 520 units wider than
the plain icon, so the halo stays inside its own advance. Each halo is a PNG (`GLOW_UPP` 12 units per px, RGBA; owner 2026-10-23 chose 12 over 8 from an estimate; measured after: 8 cost 0.53 MB more and looked the same) in an OT-SVG
`<image>`, one per icon and color. Its `glyf` is the whole halo's outline, so the non-SVG fallback is a blob, like
`:sticker`'s.
- **Nothing old changes.** The new glyphs come after the `.tk` glyphs and the two new lookups go last. Every old glyph
  id, glyf record, advance, SVG document, cmap entry and the eight old lookups stay byte-identical. The vertical
  metrics are unchanged; `head` yMin/yMax grow to −478/1120 (layout does not use them: the first baseline follows
  the `d` glyph).
- **Extension lookups.** The packer makes the two new lookups Extension (type 7): GSUB grows from 62 to 88 KB. They are
  separate because TOKENS alone (41 KB, one subtable) would have grown to 65 KB, near the 64 KB subtable limit.
- **Size:** 6,966,396 bytes (+1,230,192, +21%), build about 32 s of CPU; wall time follows the machine's load (75-110 s
  on a busy Mac, 2026-10-23).

**Proof, 2026-10-23** (old font 38a19d81 vs the 0.3 em font 3f6d3317, after the 0.1 em one 8a0d9d46 passed the same;
scripts `proof_tables.py`, `proof_hb.py`, `proof_ct.py`, `halo_geometry.py` in
`~/Documents/Tools/wt/scratch/followups/glow/scripts/`; the halo reach is `test_glow_halo_reach` in
tests/test_badge_parity.py):
- **HarfBuzz** (uharfbuzz 0.56.3), under default features, `liga` off and `rlig` off:
  - all 2560 glow spellings (every name and alias, 4 separators, 5 case mixes, both variants: 182 icon × variant
    pairs) shape to [halo, icon, `glow_space`], alone and in a sentence;
  - 1080 old tokens, 2530 badge spellings, 20,240 badge near misses, 8,654 glow near misses (`[x:glo]`, `[x: glow]`,
    `[x:dark glow]`, `[x:glow:no]`…), 6,129 card text lines (stats.pkl, old_stats.pkl, Card Edits.json, the
    standardization runs' texts) and 599
    random mixes shape exactly as before;
  - 4,401 random mixes with a glow token shape as before outside the tokens, and as the triple inside.
- **CoreText** (a Swift tool that loads the file, no install), `liga` on and off: the same sets, all pass.
- **Renders:** CoreText draws the halo whole, and clipping to the line box changes 0 pixels. Samples on four
  backgrounds: `scratch/rulings/glow/samples/`; 8 vs 12 units per px side by side (max 13/255 per channel at 100 px,
  5/255 at 30 px): `scratch/rulings-0300/samples/`.
- **Photoshop** draws the 0.3 em halo whole (installed 2026-10-23; `[darkmatter:glow]` at 100 px: 158 px wide, 112 px
  up and 48 px down from the baseline).
- **Strokes swallow it.** A layer or group Stroke outlines every bit of a halo's alpha, and 652 of 765 cards stroke
  their `Text` group. On Nebulous_Precipice (7 px black stroke) `[darkmatter:glow]` became a black disc with a thin
  white ring, nothing like the old icon layer's white glow, so the card was held (2026-10-23;
  `~/std-work-1022/glow/`, `review/three_way.png`). `:glow` suits unstroked text; the std driver's token gate fails a
  glow token in a stroked layer.
- `[thunderfury blessed blade of the starseeker:darkglow]` has 52 characters inside the brackets, the longest token:
  `ICON_SCAN_PATTERN`, `psd_edit.TOKEN`, `type_emojis.TOKEN_PATTERN` and the site's `text.js` read 1-52 since
  2026-10-23 (48 before; `test_token_limit_fits_every_icon` keeps them in step with the manifest).
- **Bakes set glow tokens** since the fork's `cards-20261024b` (2026-10-24; before, it drew the halo as a solid block and
  the bake refused). A stroked Text group strokes the halo into a black disc, as Photoshop does. Under a tint a glow
  is told as any recoloured icon, and a halo onto the stars or a layer is told too (owner
  2026-10-24: content problems bake, told); on a fork without the halos (`psd_edit.glow_halos`) it is a runner error
  (no verdict; retried once the fork is reinstalled).

### 1.9 One-colour badges and stickers under overlays (owner 2026-10-25; branches `work/badgealt`, not landed)

A Color, Gradient or Pattern Overlay repaints every pixel of the layer, so a colour badge or `:sticker` printed as a
blob. The font now has a one-colour shape for each, picked by OpenType **Stylistic Alternates** (`salt`; Photoshop:
Character panel menu > OpenType > Stylistic Alternates). The text stays as typed, so parsers and searches see no change.
- Badge (341): `badge_*.salt`, a pill outline (`BADGE_LINE` 50 units) with solid letters, same advance. The owner picked
  this look (B) over a cut-out pill (A); the trial `ss01`/`ss02` are gone.
- `:sticker` (91): `icon_*.sticker.salt`, one plain glyph: an outline (`STICKER_LINE` 40 units, the outer edge of the
  sticker's outline) and the `:tint` icon inside, with a gap between. Same advance as the sticker.
- Both take the text color, so the overlay colors them like the letters. The timing colors are lost there (accepted).
- Default glyphs, SVG documents, cmap and default shaping are byte-identical; the 432 new glyphs and the two lookups come
  last. `stoogedings.ttf` 7,269,696 bytes.
- Photoshop keeps the flag only in the document's Txt2 story (feature key `28`, or `/StylisticAlternates`), never in
  the layer's TySh. The fork (branch `badgealt`) reads it there, shapes those runs with `salt`, and its
  `set_span_style(..., StylisticAlternates=True)` writes the story alone.
- The bake (AutoCroissant `psd_edit.one_colour`, `set_alternates`): an Ability whose layer or group has an enabled
  overlay (the layer's Effects switch on too) gets the flag on every badge and `:sticker` set in the icon font, typed
  or kept, and a typed character next to one loses it. A badge typed in a body font (Butterfly's (OD)) is left alone.
- The badge FX layers (`Ability FX: (1CD)`, 31 cards, std-batch2 and queue A) are retired: an ability edit deletes
  one and says so. Word FX layers (`Ability FX: Burning`) stay as they were. TTSCardMaker `work/badgealt` drops the 31
  (one no-op ability bake each with the new font, on copies; 27 have an overlay and take the alternates). Binge Eater,
  Scapular, Black Hole and How To Jump have a blend and no overlay: without the layer the blend recolours their colour
  badges (Binge Eater's blue (1CD) pale cyan, the red (BOGT)/(ENDGT) pink); the bake's note says so; owner to accept. Their parses are unchanged;
  the goldens of Mini Doomer, Wham Bam Bone and Black Light Bulb need a blob re-pin (`tests/make_goldens.py`).
- The website's mock-up draws the same shapes when the details record shows such an overlay (`mockup.js oneColour`;
  `psd_details` marks every effect " (off)" when the layer's Effects switch is off: Springtrap, The Puppet).
- `MDW/Scripts/Effect Layers.jsx`: Add and Update all refuse a phrase that takes in a badge or icon (no new badge
  layers; Update all lists an old one as failed).
- Landing order: (1) fork: fast-forward `main` to `badgealt` (c17e0734), tag `cards-20261025b`, push both; (2)
  AutoCroissant: `requirements-cards.txt` pins `cards-20261025b` (an older fork raises ImportError in every bake that
  meets a badge under an overlay); merge, push, deploy the jetson with the re-pin; (3) TTSCardMaker `work/badgealt`.
- Developers must reinstall the font and restart Photoshop, or Photoshop shows the old glyphs.

## 2. The card-browser website

**Live since 2026-10-20 at https://michaeljsr.github.io/TTSCardMaker/** (Pages source: "GitHub Actions").
- The `github-pages` environment allows deployments only from `main` (custom branch policy). The temporary
  `card-browser-demo` review branch was removed from the policy and deleted after the merge (memory `commit-to-main`).

### 2.1 Data flow (site/build/build_site.py)

| Input | Source | Code |
|---|---|---|
| Card text, stats, authors, timestamps, problems; text history | The card sync's data: `CARD_DATA_DIR` (site.yml checks out branch `data` into carddata/ and sets `carddata/db`; a local build needs one: `git worktree add /tmp/carddata origin/data`): `stats.json`, `old_stats.json`, read into stand-ins of the bot's classes. The only source since 2026-10-25 (AutoCroissant's pickles `stats.pkl`, `old_stats.pkl`, `aliases.pkl` and site.yml's `CARD_DATA` switch are gone); no CARD_DATA_DIR or no readable stats.json stops the build | `load_card_data`, `card_of` |
| Aliases | `site/Card Edits.json` entries' `aliases` (`aliases.pkl` moved in on 2026-10-22) | `main`, `edit_aliases` |
| **PSD details** (2026-10-21) | `db/psd_details.json` in the same folder (`{"version": 1, "cards": {blob sha: record}}`, written by the card sync's build). Joined to a card by its PSD's blob at HEAD, else the blob the bot parsed, else the newest record for the path; `details: true` on the card and one `data/details/<id>.json` per card (the record plus the card's id and name). A missing or odd file logs "no PSD details" and the build goes on without the dropdown. A record's `header` (2026-10-24: the header's star and type icon boxes and where the bake would put them) feeds the mock-up's stars and types; records gathered before have none | `load_card_data`, `psd_records`, `join_details` |
| **Authors** | `site/Card Authors.txt`. `[people]` git name = shown name; `[cards]` Card Name = person. Card names match case-insensitively, `_` = space, whitespace collapsed; `[people]` lookups are case-insensitive. Listed card: `people.get(cards[name], cards[name])`; else `people.get(bot_author, bot_author)`. Same rule as AutoCroissant. Git names reach it through `.mailmap` (`%aN`). A missing file or malformed line only warns | `parse_authors`, `card_key`, `person` |
| File list + blob hashes | `git ls-tree -r -z HEAD` (works in a sparse, blobless checkout) | `repo_tree` |
| **Card images** | Each card PNG becomes 3 AVIF files named `<blob16>-<w>-<tag>.avif` (w 240/360/720 at quality 60/55/50, 4:4:4, speed 6; `thumb_tag(w)` hashes a width's settings, so changing them re-encodes that width under new names; the tags are cards.json `thumbs`; AVIF only since 2026-10-24, no WebP fallback). 240 and 360 are the grid at 1x/2x, 720 is the card view. Cached in `site/.cache/img` by git blob hash, unused files pruned; settings in `IMAGE_SIZES` and `AVIF`; `IMAGE_BUDGET` (site.yml 1800 s) splits a full re-encode over runs (partial cache saved when its files changed, the build fails until done). No service worker (owner 2026-10-24: a trial one cost ~8 ms desktop, ~24 ms 4x phone of first paint on each return visit for bytes only; site/NOTES.md §28); the HTTP cache reuses an unchanged thumb, a changed card gets a new name. PNG bytes come from the local file, else `raw.githubusercontent.com/MichaelJSr/TTSCardMaker/<commit>/` in Actions (`main/` elsewhere or when that fails), checked against the blob sha (`image_bytes`: raw main is a CDN copy up to 5 minutes old) | `make_images` |
| Icons | `Types/`, `Types/Stars/` → 96 px WebP (q90, lossless alpha; shown at ≤ 34 px). Names and aliases from the stoogedings manifest. Branch work/token-picker (2026-10-23): each manifest icon's `:tint`, `:sticker`, `:glow`, `:darkglow` pictures too (`icons/<name>.<stamp>.<variant>.webp`, by the font build's `icon_art`; needs its packages, else a warning and none), `v` per icon and `iconVariants` in cards.json (§2.8) | `build_icons`, `variant_art`, `font_builder` |
| **Web fonts** (2026-10-23) | The fonts the PSD details' text layers name, mapped to their `MDW/Fonts` file as font_check decides them (exactly that PostScript name; no aliases or loose matches since 89039eb, on branch work/site-round-2 too) → `data/fonts.json` `{"fonts": {name: {file, bytes}}}`, .ttf/.otf up to 8 MB only. Name tables read without fontTools (`font_faces`, equal to `font_check.face_names` on every file), cached by blob in `site/.cache/fonts.json`; a file not checked out is downloaded once from raw (8 at a time). For the edit mock-up only; never fails the build | `web_fonts`, `font_faces`, `record_fonts` |
| **Image versions** | One `git log --raw --no-renames` pass over all history, commit metadata only, no blobs, dated by committer time (`%ct`, as the bot dates versions; since 2026-10-23) (0.05 s for 520 commits). For each card PNG it lists the commits that changed its content (not a bot commit's lossless recompressions, named in its `Recompressed: <path>` trailers: the push guards, the PNG batch; 2026-10-23). Exact renames (same blob deleted and added in one commit, i.e. folder moves such as the `Auxillary/` typo fix) are followed; a rename that also edits the image starts a new history. Inexact rename detection (`-M`) is deliberately not used: in a blobless clone it would download blobs. A shallow clone gets none and warns | `image_versions` |
| Ban list | `site/Constructed Ban List.txt`, matched on letters and digits (`alnum`) incl. aliases | `parse_banlist` |
| **Duplicates** | A numbered PNG with no PSD of its own next to a card's PSD (`A_Snack.psd` → `A_Snack2.png`, `A_Snack3.png`). The team's rule: duplicates always share the original's PSD. Each becomes its own card with the original's data, `dup` = original id (the original gets `dups`). Its name is its Card Edits entry's alias (longest, capitalised: "breakfast" → Breakfast), else the original's name. Its author, text history and `since` are the original's. Duplicates don't count as mentioning cards and are left out of the Stats page. The other names are hidden title layers in the PSD (Breakfast, Supper; Grape Juice; Inconspicuous Watermelon), but which PNG gets which name is only recorded by the aliases | `main`, `dupsOf` (app.js) |
| Unused status effects | Debuffs/buffs that no card's text names get `_unused` (Unused flag, filter, and a list on the Stats page) | app.js `init` |
| **Website admins** | `site/Website Admins.txt` (GitHub logins; the repo owner is always one). A personal repo gives collaborators only write, so admin is a site rule: ban list, hidden cards, people, this list | `parse_admins`; edit.js `isAdmin` |
| **Creatures** | stats.json `creatures` (bot split) or the `creatures` edit → `entry.creatures` [{hp, def, atk, spd}]; card stats = creature 1 | `main`, `apply_edits` |
| Unparsed cards | PSDs/PNGs under Creatures/, Items/, Field/, N.M.E/ and Auxiliary/ that aren't in stats.json: image only, classified by folder | `main`, `classify` |
| **Decks** (2026-10-21) | `site/decks/<owner>/<slug>.json`, hands published from the website's deck builder with the edit batch (format: `site/decks/README.md`). Public: resolved against visible cards (unknown ids dropped with a warning, counted) → `data/decks.json` + a `decks/<owner>/<slug>/` share page; private: PBKDF2 → AES-GCM made in the browser, shapes checked, passed through as they are, no page. Bad files skipped with a warning, never a failed build; `decks: {public, private}` and `files.decks` in cards.json | `read_decks`, `clean_deck`, `deck_text` |
| **Balance analytics** (2026-10-22) | `site/build/analytics.py` (no PSDs, no network), called by `main` after the image-version pass and before the history moves to history.json: `analytics.build(cards, stats, old_stats, history=commits is not None, psd_commits=psd_commits)`. `psd_commits` is `{(commit unix time, PSD file name in lower case)}` from the same `git_history()` pass (non-delete `.psd` entries; `None` in a shallow clone): an old_stats version without a `blob_sha` counts as exactly dated (`exact`) when a commit touched a PSD of its file name at that second, else its change row is marked `approx`. Cohorts are creatures and minions by star level (under 5 cards fold into `<kind>\|all`), per-card pct/z/flags, change rows (`page_chain`, re-reads folded by when since 7a87d91a) and the monthly series (full history only). Writes `data/analytics.json` and `balance: true` on each card with an entry; an exception is a `warn()` and an empty file, never a failed build; no tz database also warns (UTC instead of Pacific). build_site puts its own folder on `sys.path` before `import analytics`, so file-path loads (TTS/checks/tts_save_gate.py) still work | `analytics.build`, `analytics.summary`; `site/assets/balance.js` |

The records are read **without AutoCroissant code** (plain JSON into stand-ins). Renaming `CardInfo` is harmless;
renaming attributes (`card_type`, `ability`, `stats.hp`, `timestamp`) silently blanks those fields (a warning names
them). Until 2026-10-25 the build unpickled AutoCroissant's pickles through an allow-list (`_Unpickler`, gone).

**Output (`site/dist/`):**
- **Data files:**
  - `data/cards.json` (everything the first screen needs; ~84 KB gzipped). History counts `hv`/`iv` instead of the
    data; `parsed` only when false; `ts` = the PSD's last commit, `since` = when the current text started (the bot's
    carried `version.from`, or earlier versions with the same text; only when it differs from `ts`).
  - `data/details/<id>.json` (one per card with a PSD record, 0.9-16.7 KB; fetched when the card page's Details
    dropdown is first opened, rendered by `site/assets/details.js`; in the data stamp).
  - `data/analytics.json` (balance analytics, plus since round 3 `revisions` and `archived` rows for Stats' History;
    ~130 KB; loaded only when the Stats page or a card's Balance block opens, rendered by `site/assets/balance.js` and
    `insights.js`; in `files`, so in the build version).
  - `data/contributions.json` (the People tab and person pages: per person cards, edits, commits, and since round 3
    `people[]` keys `power`, `bal`, `revisions`, `rework`, `ratings`, `editFields`, `archived` (P1-P21; `selfBias`,
    `rater`, `powerVsStats` only once someone has ratings of their own); site/NOTES.md §8).
  - `data/changelog.json` (the Changes page: the card sync's events, edit batches, and `archived` rows with their tag:
    Renamed, Retired, Deleted).
  - `data/rules.json`, `data/history.json`, `data/versions.json` (loaded only when the rulebook, a card or the
    changes feed needs them). `versions.json` holds a commits table plus per-card `[[commit index, path then if
    different], …]`, newest first; ~27 KB gzipped, 3,211 versions of 725 cards from 356 commits since 2023-11-05.
- **Pages:** `index.html` (`<base href="./">`), `404.html` (`<base href="<SITE_URL or ''>/">`), and one share page
  per card at `c/<slug>/` with OG tags. Kind labels use the game's words; tokens are flattened, `:tint` / `:sticker`
  included.
- **Cache-busting:** every script/style/data URL carries `?v=`: an asset's from its content and its imports' stamps,
  the data's from its content without the build time (since 2026-10-24). `index.html`'s preload hints and modules'
  `from "./x.js"` imports are stamped too, so a deploy never mixes old scripts with new data, and one that changed
  nothing keeps every URL. `build.txt` digests the site's files (not build_site.py's text); a build equal to the live one is not deployed (owner 2026-10-24; the Changes page's "Site updated" is the last build that changed the site). Its 2nd line is the page's stamp (`<meta name="page">`): fresh.js polls build.txt (no-store) on load, on tab return, every 3 min (15 s for 15 min after a publish) and reloads to a new deploy, or shows a Refresh bar while a form is open; a publish is live once cards.json's `source` (the built commit's date) reaches the publish commit's date (NOTES.md §29). Icons are `icons/<name>.<stamp>.webp` (the stamp in the name, not a query: an old URL gets its own picture or a 404).
- **Size:** about 117 MB in total, mostly 2,463 card images.

**Build behaviour:**
- **Fails** (exit 1, nothing deployed) without `CARD_DATA_DIR` or a readable `stats.json`, when it has fewer than
  `MIN_CARDS` (100) entries, or when more than max(5, 2%) of card images fail.
- **Degrades** (logs and continues) for missing history, aliases, manifest, authors file or a few images.
- **Timing:** the CI cold run (2026-10-20) made 2,463 images in about 100 s. Warm runs take 2–3 s.

### 2.2 Workflow (.github/workflows/site.yml)

- **Triggers:** push to `main` touching site code or site-only files (since 2026-10-25 not the card inputs: PSDs, PNGs,
  `Types/`, the fonts and the card files in `site/` start Card sync, which dispatches this build after its data commit),
  a daily cron `0 12 * * *` (the fallback), and `workflow_dispatch`.
- **Checkout:** `actions/checkout@v7`, `fetch-depth: 0` + `filter: blob:none` (every commit and tree, about 25 MB, no
  contents), plus a non-cone sparse checkout of `/site/` (which holds the four hand-kept website files since
  2026-10-20), `/Types/` and the font manifest.
- **Steps:**
  - `setup-python@v7` (3.12, pip cache; 3.14 since 2026-10-24, branch work/py314) → `pip install -r site/build/requirements.txt` (pillow) → `configure-pages@v6`
  - a cache-key step hashing `git ls-tree -r HEAD | grep -i '\.png$'`, then `actions/cache@v6` on `site/.cache` with key
    `card-images-v2-<hashFiles(build_site.py)>-<png list hash>` and restore-key `card-images-v2-`. The cache is
    re-saved only when card images or the image settings change.
  - `build_site.py` with `SITE_URL` from configure-pages → `upload-pages-artifact@v5` → deploy job
    `deploy-pages@v5`. `error_count: 10` in its log is just that action's default input.
- **Settings:** `concurrency: pages` with `cancel-in-progress: false` (GitHub's recommendation: never cut a deployment
  off) and `timeout-minutes: 45`.
- **Verified locally 2026-10-20** with a fresh `git clone --filter=blob:none` plus the same sparse paths:
  - it builds identically, the history pass downloads no blobs, and removed cache entries are re-fetched from
    raw.githubusercontent;
  - the failure modes behave as above.

### 2.3 Local preview

```bash
cd ~/Desktop/TTSCardMaker
pip install -r site/build/requirements.txt         # pillow (+ the font build's numpy, opencv, fonttools on work/token-picker)
python site/build/build_site.py                    # writes site/dist; uses local PNGs; caches images in site/.cache/img
python -m http.server -d site/dist 8000            # open http://localhost:8000
```

- **Image versions need full history.** The dev Mac's clone is shallow, so the build warns and the card page has no
  version viewer. `git fetch --unshallow --filter=blob:none` gets it (about 25 MB).
- **`SITE_URL`:** empty locally, so share pages have no `og:url` and `404.html` uses `<base href="/">`. Set
  `SITE_URL=https://michaeljsr.github.io/TTSCardMaker` to test previews.
- **Forks:** `CARD_REPO` / `BOT_REPO` point the build at forks.
- **Browser tests:** the scratch harness is `/private/tmp/sitetest/`:
  - `func.mjs`: 31 functional checks at desktop and 375 px;
  - `versions.mjs`;
  - `lh.sh` (Lighthouse);
  - `serve.py`, a gzip server that behaves like Pages.

  It needs `npm i playwright lighthouse` and `npx playwright install chromium`, with `PLAYWRIGHT_BROWSERS_PATH` under
  /tmp. Results 2026-10-20 (mobile, throttled):
  - Performance: home 89–90, card page 90, stats 99–100, rulebook 99.
  - Accessibility, best practices and SEO: 100 everywhere.
  - Weight: card page 3.1 MB → ~0.7 MB.

### 2.4 Feature → file to change

| Feature | File |
|---|---|
| Routing, card grid/table and filters, card modal (stat bars, similar, mentions, text history, **image version viewer** with side-by-side compare and ↑/↓ keys; old images from `raw.githubusercontent.com/<repo>/<sha>/<path>`, falling back to jsDelivr), rulebook, changes feed, compare, shortcuts, lazy data loading (`loadRules`/`loadHistory`/`loadVersions`), responsive images (`imgAttrs`) | `site/assets/app.js` |
| Search language (same as `/query_ability` plus names/aliases; flags `is:unparsed` `has:history` `has:problems` `has:alias`; `has:details` removed 2026-10-24) | `site/assets/search.js` |
| Token → icon (variants with their pictures), `(TAG)` → badge, card-name links, word diffs | `site/assets/text.js` |
| **Token picker** (branch work/token-picker, 2026-10-23): `[` in card text suggests icons with pictures, `name:` the variants, `(` + a tag's start the timing tags (§2.8) | `site/assets/tokens.js` (fields marked `data-tokens` in `edit.js`, `inline.js`) |
| The card page's History block, the image versions' "Text then" (`textAt`, the current text from `since`), Recently updated's diff rule, and the site's dates (`date`, `month`, `monthKey`: Pacific time, `TZ`) | `site/assets/history.js` |
| Deck builder: the constructed hand from rulebook "03 Setting Up" (`HAND`; 17 cards; field ≤ 3★), saved decks, share code, import/export, random fill, print, TTS export. **The only place that shows "banned"** (picker markers, checks, random fill), besides the rulebook's ban list page and the Stats chart about constructed bans | `site/assets/deck.js` (+ `tileHtml(…, { constructed: true })` in app.js) |
| Stats: counts, **"Starting-hand choices"** (legal, unbanned cards per hand tier vs how many the hand needs; replaced the stacked star-level bars on 2026-10-20), averages heat table (OKLab ramp, text always ≥ 4.5:1: `heatScale`), distributions, outlier scatter, table views | `site/assets/stats.js` |
| **Balance** (2026-10-22): the Stats page's Balance section (cohort bands, strongest/weakest, average total over time, buffs and nerfs, recorded changes by month, most reworked; round 3: power creep, series strength, buffs and nerfs by stat; the cooldown update chart is gone) and each card's Balance line and Recorded changes timeline, from `data/analytics.json` | `site/assets/balance.js` (imports `chartCard`, `hbar`, `table`, `stacked`, `fmt`, `trunc` from stats.js), `site/build/analytics.py` |
| **Profiles and People** (2026-10-21): sign-in top right, opt-in `site/profiles/<login>.json` through the publish flow, People tab (leaderboard from `data/contributions.json`: cards, edits as publish batches, commits; plain titles), person pages `#/u/<key>` (profile, contributions, Cards panel, Decks panel), full-page profile editor `#/u/<login>/edit` with the visual card picker | `site/assets/profile.js`, `site/assets/pick.js` (reusable: `data-choose`), `edit.js`, `build_site.py` (`read_profiles`, `contributions`, `u/<login>/` pages) |
| **Ratings** (2026-10-24): `site/Card Ratings.json` (dated favorability and power, 1-10, `by` "group" for the TTS sessions' boards (AutoCroissant `python -m commands.tts_ratings`) or a login for the website's Rate it; read by the build and the website's editor, never the bot) as cards.json `ratings`; the card page's Ratings block (per scale the current value, its tier, a chart; Rate it in edit mode) | `site/assets/ratings.js`, `edit.js` (`ratingsFile`, `op.rate`), `build_site.py` (`parse_ratings`, `apply_ratings`); site/NOTES.md §25 |
| **Stats round 3** (2026-10-24): distributions 0-10 with a pager past 10, cooldowns and uses by star level, the Ratings section (rated power vs stats, favourites, most divisive, risers, power tier vs ban list, Too Stronk nerfed?) and History section (who changes whose cards, most revised, created per year, text freshness, retired and deleted) | `site/assets/stats.js`, `site/assets/insights.js`; site/NOTES.md §26 |
| **Archive tab** (2026-10-24): the Changes page's retired, deleted and renamed cards with a tag each and chips per tag (`?tag=`); profiles list a person's archived cards | `site/assets/changes.js` (`archiveHtml`, `archiveTagHtml`), `build_site.py` (`archived_cards`) |
| **Person stats** (2026-10-24, P1-P18 and P21; P5/P7/P8 hidden until per-person ratings; no P19 or P20): Balance, Ratings, Revisions, Style, Archived cards panels, ranks, edited fields | `site/assets/profile.js`, `build_site.py` (`person_extras`); site/NOTES.md §8 |
| **Pinned tray** (2026-10-24, edit mode, Grid): boxes pinned per browser (series, subseries, author) to drop cards on: drag, tap with a selection, or keys 1-9 | `site/assets/tray.js` (wired in app.js, inline.js `dropOn`); site/NOTES.md §24 |
| **Inline editing** (2026-10-21): in edit mode the card page is the editor (stat cells, ability, name, chips in place; drag cards onto Author or Series; one batch bar; the form stays as All fields) | `site/assets/inline.js` |
| **Editor previews, drafts, stars and types** (2026-10-23): the Edit dialog's live preview pane (the card as it will look once published, on every key; a sticky bar on narrow screens), unsaved-edit drafts in localStorage `edit-drafts` with "Restore your unsaved edit?" (cleared only on save, discard, publish: an editor closed without saving puts the draft back, `snap`/`back`; typing over the dialog's offer joins it; never offers what the batch has; says when the card changed since; floats at the bottom of a phone screen), stars as a slider and types from an icon palette with search on the card page in edit mode (`palette.pickerSearch` adapts the token picker's search: one line in app.js, NOTES §21; the type row is a button), the tooltip's hide rules (the stuck "Click to edit" fixed) | `site/assets/live.js`, `drafts.js`, `stars.js`, `palette.js`, `tip.js` (wired in edit.js, inline.js, app.js); site/NOTES.md §21; AutoCroissant `tests/test_site_editor.py` (16, `AC_SITE_SRC`; one skips without tokens.js) |
| **Image previews** (2026-10-23, edit mode; the viewer 2026-10-24): the card image's tabs Original / Mock-up / Rendered, in place on the card page and at the top of the Edit dialog's live Preview pane. **Mock-up**: the new name, text (icons as icons, the card's font from `data/fonts.json`) and stat cells drawn over the 720 px thumb in a canvas, the old text erased with the background around it (timing tags as badges); psd_details has no text boxes (a `text[].box` would be used), so of the edge-detected blocks of lines in the template's area (below the stat rows) the one the old text fits (`calibrate`: its width at the block's size and its line count; `bestBlock`), plus lines next to it it left out (`strayLines`, never a stat bar); when none fits, the new text in a dark box over the area ("Text in a box"), never over a half-erased old one; stars and type icons where the bake would put them (the record's `header`: `starPlan`, `typePlan`, `fillBox`), else in a box, a refusal named. **Rendered**: dispatches `card-preview.yml` (inputs card, edit = the card's Card Edits entry as published, request) with the editor's token (classic `repo` scope or fine-grained Actions: Read and write), checks `previews/<request>/result.json` on branch `previews` at the contract's pace (10 s for 3 min, then 30 s, up to 16 min; one tab for all, one toast a round) from any page, and on the next load for one still running (a result that came while the site was closed), badges the tile, shows the preview in the image's place with a diff, or the refusal; out of date when the edit or the card's PNG changed; the last 3 per card in localStorage `card-previews`, "expired" after 6 days or an image 404 a day on (the branch keeps about a week). Before the first press it says a previewed edit is public (the run's inputs, the `previews` branch). Contract: `site/NOTES.md` §19 "Card previews" (§2.7 here); the website side: §22, §23 | `site/assets/preview.js` (`CONTRACT` is the one adapter), `site/assets/mockup.js`, `build_site.py` (`web_fonts`); wiring in `app.js` and `edit.js` (`github`, `scopes`, `psdOf`, `entryFields`, `formOpened`) |
| **Details dropdown** (2026-10-21): the PSD's structure at the bottom of the card dialog from `data/details/<id>.json` (fonts with layers and a supplied flag, text layers, effects, canvas, icon style, stat rows, FX/overlay layers, template deviations, layer tree) | `site/assets/details.js`, `build_site.py` (`load_bot_json`, `join_details`) |
| **Published decks** (2026-10-21): `#/decks` browser (public only), `#/decks/<owner>/<slug>` pages, publishing from the Deck builder through the batch, private decks encrypted in the browser and listed only on the owner's page | `site/assets/decks.js`, `edit.js`, `profile.js`, `build_site.py` (`read_decks`) |
| Theme, badge colors, version viewer layout | `site/assets/style.css` |
| Data shape, images, versions, authors, OG pages, ban list, unparsed cards, cache-busting, `Card Edits.json` (`apply_edits`) | `site/build/build_site.py` |
| **Edit button** (sign-in with a GitHub token, edit form, batch tray, one-commit publish via the Git Data API, Card Authors / ban list line edits, rulebook pages, per-creature stats, admins list) | `site/assets/edit.js` |
| **Sharing**: address bar shows `c/<id>/` (cards) and `d/#/deck/…` (hands) via `syncUrl` (base pinned at load); `c` copies the card link (no card Share dialog, Full image or PSD on GitHub link since 2026-10-23); deck and person pages keep their Share; a hand's Discord message links cards as `[name](<url>)` | `site/assets/app.js` |
| **Link previews**: per-card `c/<id>/index.html` with og tags, `og:image` = 480 px JPEG `thumbs/<blob16>-og.jpg` (from the 720 px AVIF), `theme-color` = card accent color; `og.jpg` banner; `d/` page | `site/build/build_site.py` (`og_tags`, `share_text`, `accent_color`, `site_banner`) |
| **Speed pass** (2026-10-24): one `route()` per navigation (a card closes without a redraw unless `viewSig()` changed or the view under it is still `.loading`), cached date formatters (`dateFmt`), the grid reuses its tile nodes (`gridTiles`) and draws the window's tiles first, Table rows in chunks (a collapsed `tbody.sizer` of the rest's widest rows, measured on a canvas, holds the column widths), Back/Forward draws at once (`history.state.drawn`), a card opened from a link draws no grid behind it, lazy type icons, preload hints (the grid's first pictures on index.html: `grid_preload`; the 720 px picture on `c/<id>/`); build: `?v=` stamps by content, site.yml `paths`, `build.txt` (no deploy gate: owner's call). The rules code must keep: TTSCardMaker `site/NOTES.md` §27 (its AutoCroissant tests went with the 2026-10-25 test cut) | `site/assets/app.js`, `text.js`, `stats.js`, `site/build/build_site.py` |

**Vocabulary** (the user's rule, 2026-10-20): use the game's own terms, per the rulebook, never invented ones.
- **Card groups:** creatures, minions ("for most purposes considered creatures"), items, **field cards** (the field is
  the play area), **N.M.E. items**, **auxiliary** items/creatures, status effects / buffs / debuffs.
- **Types:** the creature and item types on a card.
- **Other terms:** stars, abilities, global turn.
- **"Slot"** means one of the 3 field positions, so the hand builder never calls its places slots.
- **"Kind"** is kept as the UI word for card group: the rulebook has none, and "type" already means types.
- **Copy style** (the user's rule, 2026-10-20): headings are a few words; descriptions are one short sentence at most
  and are cut where the heading or chart already says it; no filler ("This chart shows…"). Keep what a player needs
  (e.g. what a chart's numbers mean).
- **Mobile layout:** single-column grids that hold horizontal scrollers (card strips) must use `minmax(0, 1fr)`, not
  `1fr`, or the strip widens the column. The card page did this (a 2,870 px column at 375 px) until 2026-10-20.

### 2.5 What this means for AutoCroissant operations

**Every Card sync data commit updates the public website.** The site shows the `data` branch's `db/` (Card sync
dispatches site.yml after each data commit; its daily cron is the fallback). Until 2026-10-25 it showed AutoCroissant
main's pickles, so every `/push` reached it.
- **Consequence:** bad data that passes the build's gates becomes public at the next site build: the data gate
  (autocroissant-change-control) and a `restore` dispatch (AutoCroissant DATA_BUILD_ACTION.md §13) are the remedy.
- **Card edits:** pushed card content changes images immediately, text and stats once Card sync re-parses (its data
  lane, then the site build).
- **Authors:** both sides read `site/Card Authors.txt` with the same rule, so authors agree whatever the bot recorded.
- **Duplicates:** the card data build records `<Name>N.png` (no PSD of its own) as a copy of `<Name>`'s card with
  `duplicate_of`; name = its Card Edits aliases > the original's (hidden title layers name nothing since 2026-10-25). Search skips them.
- **Bot ↔ site:** `/query` replies have a Card page button (ids from the site's cards.json) and show website
  names, notes and series/subseries; searches link to the site; `/website`. `SITE_URL` is in global_config.py.
- **Website edits:** `site/Card Edits.json` (card file name → fields). The site applies it on every build (a publish
  triggers one); the card data build too, keeping PSD values in `parsed_values`. Field list is in three places:
  `EDIT_FIELDS` (build_site.py), `FIELDS` (edit.js), `CardEdits.TYPES` (card_edits.py); so is each field's kind
  (printed on the card image / from the folder / database-only): `PSD_FIELDS`, `FROM`, `CardEdits.KIND` (2026-10-20,
  docs/EDIT_PIPELINE_DESIGN.md §3 and §9). cards.json carries `pending` (image fields still to be baked by the edit
  pipeline, `name` excepted), `refused` (the pipeline's `_psd` verdict while the edit and PSD are unchanged) and
  `reparsePending` (the PSD's tree blob differs from the bot's `blob_sha`); app.js shows them as "PSD pending: …",
  "Image update refused: …" and "Re-parse pending" on the card page, tiles, table rows and the Changes tab; the
  `pending` flag filters pending and refused cards; the editor marks pending rows. A malformed `_psd` is logged and skipped
  like any wrong-typed field; a refusal shows only while its `for` values and `psd` blob prefix still match (review fix 2026-10-20).
  A pixel miss is no refusal since 2026-10-23 (owner): the edit is baked and the bake commit's line says "name baked
  with a pixel difference: bad 1.97%, mean 1.60, max 217"; only a broken render (more than 25% of the box's pixels off,
  or none of the edit drawn) or a read-back that is not the edit is refused. Since 2026-10-24 (owner; AutoCroissant
  work/fx-rebuild) no content problem refuses either: text that won't fit, typed badges and tokens, styled words and
  font stand-ins bake, each told in the commit line and the baked `_psd` verdict's `reason` ("name 'A' -> 'B'; overflow:
  name: it runs onto 2 lines (it has 1)") and in the preview's `notes`. A flagged bake keeps its baked verdict, alone
  (`{"_psd": ...}`) once no field is left (card_bake.rewrite_entry `keep`); TTSCardMaker branch work/fx-rebuild (not on
  main yet) has build_site put a baked verdict's reason in cards.json's `baked` while `psd` matches the tree's blob, and
  the card page show it ("Image baked: ..."). A refused verdict carries `rules` (card_bake.RULES); one without is tried
  again by `--pending`. Pixel refusals written under the old bar
  (Masked Dedede, PoD11ZOMG, Zero Override) show as refused until the next run bakes them: `build_site.py` (about line
  855) knows nothing of the floor, so the window is the next Card sync run after the change lands (card-edits' 30-minute tick until the cutover,
  2026-10-24; a dispatch closes it at once).

The data gate is owned by **autocroissant-change-control**; /update_stats (the bot's sync) by
**autocroissant-run-and-operate**.

### 2.6 TTS save workflow (.github/workflows/tts-save.yml, 2026-10-20; daily gate 2026-10-21, bot data 2026-10-22; deleted 2026-10-24)

**Retired at the cutover (2026-10-24):** tts-save.yml is deleted. Card sync (`.github/workflows/card-sync.yml`, its "TTS save"
step) runs the same updater on every run with work, with the same two env overrides and `--names` from the new data, and
commits the save in its one main commit M ("Card sync: … TTS save updated"); card-guards' refresh dispatches card-sync.yml.
What follows is how tts-save.yml ran until then: the team fields, deck rules, the save's Lua and the checks still hold.

Keeps `TTS/Impossibility Simulator.json` current with AutoCroissant's updater `commands/tts_save.py` (what it
changes: autocroissant-run-and-operate §7 "TTS save"; the bot's manual `/update_tts_save` was retired 2026-10-25). Until 2026-10-20 the
bot ran the updater itself after `/update_stats`; this Action replaced that.

- **Team fields:** a card's Description, GMNotes, Memo, Tags and scripts (LuaScript, LuaScriptState, XmlUI) are
  left to the team (custom fields, descriptions: 2026-10-22): the updater never changes them on an existing card
  (links in them are not re-pinned) and new cards start with them empty. The one-time role-tag clearing is retired.
- **Deck rules** (`TTS/decks.json`, 2026-10-22): the 7 "Specific Item Decks" are complete rules (count 1, back
  `MDW/cardback_item.png`, 94 cards); Piko Piko Hammer is in the 3★ item decks too; the four "Creatures Developer
  Version" decks have `add_new_since: 318001ea` (the save's first commit): each gains every creature of its
  stars the save then lacked (renames followed by blob; a card moved to other stars moves deck with it, nothing else
  moves; no record is
  kept, so a card taken out by hand comes back unless the rule excludes it). `TTS/audit.py` keeps `add_new_since` and
  every complete counted rule when it rewrites the file.
- **Triggers:** cron `41 11 * * *` and `workflow_dispatch` (input `force`, default true). No push trigger since 2911a8c.
- **Daily gate** (the owner's, 075e6c43, 2026-10-22; `actions: read`): step "Card browser names" downloads the live
  `cards.json` (also the updater's `site/dist` fallback) and hashes its png -> name map; `actions/cache/restore`
  (`lookup-only`) looks for `tts-save-names-<hash>`, saved after each run that ran with new names. Step "Changes since
  the last run" runs when forced, when that cache entry is missing (a data commit renamed cards; the bot's PICKLE 3888392
  did, once site.yml rebuilt the site), when there is no earlier successful run on this branch (`gh run list --branch`;
  a failed lookup also runs), or when a commit not made by the Action itself (`AutoCroissant|TTS save:`) touched
  `*.png`, `TTS/decks.json`, `site/Card Edits.json` or the save (`git log last..HEAD`; full history, `filter:
  blob:none`). A bot commit that changes no name (docs) changes no hash and runs nothing. Our time-window version
  (bot pickle commits within 7 h of the last run, 75ce18cc) was dropped in the rebase: the hash covers it. Not watched:
  the updater's own code and build_site's name logic (dispatch by hand after changing them; owner's call).
- **Checkout:** this repo, non-cone sparse `/TTS/` + `/site/Card Edits.json`; the bot (`bobberlington/AutoCroissant`
  main) into `bot/`, sparse `commands/tts_save.py`, `git_commit.py` (the shared commit path, 2026-10-21), `repo_history.py`,
  `card_models.py`, `card_edits.py`, `github_api.py` and `global_config.py` (everything the updater imports; `commands/` is a namespace package, no
  `__init__.py`; no config.py: github_api reads it in a try/except).
- **Run:** `setup-python@v7` (3.12; 3.14 since 2026-10-24, branch work/py314), `pip install requests`, then in `bot/`
  `python -m commands.tts_save "$GITHUB_WORKSPACE/TTS/Impossibility Simulator.json"` (the report is the job log) with
  `AUTOCROISSANT_LOCAL_DIR=${{ github.workspace }}` (the updater reads `TTS/decks.json` and `site/Card Edits.json`
  there) and `AUTOCROISSANT_HISTORY_CLONE=~/TTSCardMaker-history.git` (its bare blobless clone, made on each run;
  `expanduser` resolves the `~`). Both are read by AutoCroissant's global_config.py (`environ.get`). The first
  version symlinked the workspace to `~/Desktop/TTSCardMaker` instead; the overrides removed that step.
- **Commit:** only when the save changed: identity `AutoCroissant <autocroissant@gmail.com>` (the bot's PICKLE
  identity), message = the first line of `TTS/Impossibility Simulator.message.txt`, the updater's own
  `_commit_message` (as the bot's commits: "TTS save: 29 links pinned"; fallback "TTS save: updated"; until 2026-10-22
  the first four report headings, informational ones included; Nicknames counted by name, not card, and "resizeTargets
  updated" added, since 2026-10-22: before, 131 names read "398 Nicknames fixed"), `git push`. `permissions: contents: write`,
  `concurrency: tts-save` (no cancel), `timeout-minutes: 20`. The `.bak.json`, `.report.md` and `.message.txt` the
  updater writes beside the save stay on the runner (and are git-ignored).
- **The save's Lua** (TTSCardMaker c9873ddf, 48 lines): `TEAMS` puts Red, Orange and Green on the TTS team Hearts and
  Blue, Purple and Pink on Clubs (other seats None; Grey and Black untouched), on each seat change and once after load;
  the large cards in `resizeTargets` go to scale 3,1,3 three frames after they leave a container. The updater rewrites
  that table (one `["Nickname"] = true,` row per line, `}` at column 0, no comments in it; any other shape is an
  Errors line and the table is left alone). No zone claiming any more: every hand zone keeps its saved owner, and a
  TTS team lets teammates see each other's hand zones (TTS knowledge base, "Teams"), so the facedown item holders owned
  by Red and Blue show to their whole side (not verified with that seat empty).
- **Checks:** `TTS/checks/` (its README: what each checks, interpreters, variables). `run_all.py` runs all 24 on the
  save in about 2 minutes: structure, serializer, CustomDeck collisions, drift against c9873ddf, deck contents, names,
  backs, rules, the Lua (lupa mock of the TTS API), link pins and HTTP, the updater run as the Action runs it, its
  report and commit subject, and the daily gate against real history (names hashed from the bot's stats.pkl at
  3888392 and 39da9ed). Drift fails on any change to an existing card's Description, GMNotes, Memo, Tags or scripts and
  on the global Lua outside resizeTargets. Any Python with `requests` and `Pillow`; the
  Lua checks need `lupa`. The names checks follow the live website, so between a data commit and the next run they fail
  (`contents_names`, `final_spot`, `updater_idempotent`; 24 of 24 after the 2026-10-22 update).
  `AUTOCROISSANT` defaults to `~/Desktop/AutoCroissant`; `updater_report_decks.py` needs that checkout to have the
  updater as on main since 2026-10-22 (pull first, or point `AUTOCROISSANT` at a clone that has it).

### 2.7 Card previews: the website's contract (card-preview.yml, 2026-10-23)

The owner's ask (2026-10-23): press "preview" to dry-run bake a card, work on another card, come back to see it. Server
side done (AutoCroissant `commands/card_preview.py` and `commands/card_preview_branch.py`, TTSCardMaker
`.github/workflows/card-preview.yml`; design: `docs/EDIT_PIPELINE_DESIGN.md` §16 with the first-dispatch test plan); the UI
is branch work/site-round-2 (TTSCardMaker `site/NOTES.md` §22 there). The same contract, with more detail, is `site/NOTES.md` §19.

| Step | What the website does |
|---|---|
| Tell | the repo is public: a previewed edit is too (the run's inputs, the `previews` branch). Say so before an editor's first press. |
| Dispatch | `POST https://api.github.com/repos/MichaelJSr/TTSCardMaker/actions/workflows/card-preview.yml/dispatches`, body `{"ref": "main", "inputs": {"card": c.key, "edit": JSON.stringify(fields), "request": crypto.randomUUID()}}` -> 204. `fields` = the card's PSD-kind fields (name, stars, types, hp, def, atk, spd, ability; + creatures if the entry has them) as Publish would leave the entry (committed entry with the staged op applied), at most 16 KB. A new request id per press. `result.edit` echoes `fields` exactly as sent (no `_`-keys). |
| Token | a write collaborator's; classic: `repo` scope (GitHub's docs for this endpoint; `public_repo`, which sign-in accepts, to confirm live); fine-grained: TTSCardMaker with Actions: Read and write (+ Contents: Read and write for publishing edits). 403/404 = scope; 422 = inputs or the file not on main. |
| Poll | `GET https://api.github.com/repos/MichaelJSr/TTSCardMaker/contents/previews/<request>/result.json?ref=previews`, `Accept: application/vnd.github.raw+json`, `cache: "no-store"`: 404 until published; every 10 s for 3 min then 30 s, give up at 16 min (jobs' timeouts 11 + 3; the run's `display_title` is `Card preview <request>`: once it is `completed` and result.json still 404s a minute later, nothing will come: link the run). Never poll raw.githubusercontent.com (`max-age=300`, 404s too). |
| Image | when `result.image` is set: `https://raw.githubusercontent.com/MichaelJSr/TTSCardMaker/previews/previews/<request>/preview.webp` (720 px WebP, quality 75; the path is new per request) or the contents URL with `Accept: application/vnd.github.raw` as a blob; beside it `before.webp` (`files.before`): the card's official PNG made the same way, which the Diff compares with (the site's thumbs are AVIF: against them codec noise marks 1-5% of an unchanged card). The full-size PNG is not published. |
| Show | `state`: `baked` (image, `reason` "hp 5 -> 8", `notes`), `redundant` (no image: already on the card), `refused` (`reason`; an image only when `image` is `refused`: what the render got), `error`: `transient` true = try again (link `run`); `reason` "input: ..." = a site bug; any other = the preview can't run on this card or repo as it is (show `reason` and `run`). `fields` per field, `changes`, `region` (the edited box), `took_s`, `at`. `esc()` every string. |
| Stale | current while `base.psd_blob` / `base.png_blob` are the card's blobs on main (cheap: `base.png_blob.startsWith(c.img)`; exact: `GET .../contents/<folder>?ref=main` lists both shas) and the fields the site would send now equal the edit it stored at dispatch (or `result.edit`), both as canonical JSON (keys sorted). A stale one stays shown, marked, until a new press. |
| Come back | keep `{card id: {request, edit, at}}` in localStorage; a preview stays at least 6 days unless 150 newer ones came; `index.json` on the branch lists all `{request: {at, card, path, state, psd_blob, png_blob}}`. |
| Latency | about 1-2 min for a stat, 1.5-3 for an ability or name (pip 27 s, checkouts 4 s, bake 6-26 s, the publish job's start about 10-20 s, estimated). |

What it never does: write main (PSD, PNG, edits file), post to Discord, start another workflow (every push trigger is
`branches: [main]`; a GITHUB_TOKEN push starts none; AutoCroissant `tests/test_card_preview.py` checks every workflow), or
hold a write token while it parses the PSD and the edit (only the publish job writes). Budget: 3 REST blob reads a stat
preview (manifest, PSD, PNG; + the chart for an ability, + new Types/ icons), against GITHUB_TOKEN's 1,000 an hour for the
repo, shared with card-sync (and card-edits until 2026-10-24). Size: about 60 KB a preview.

### 2.8 Token picker and variant pictures (owner request 2026-10-23 ~18:00; branch `work/token-picker`, both repos, not merged)

Owner: "typing a type in stoogedings (which starts with a [ symbol) will auto suggest types with the visual next to the
suggestion and narrow the selection as the person types", so nobody has to memorize the tokens, variants included.
Full notes: TTSCardMaker `site/NOTES.md` §19; user docs: `site/README.md` Editing.
- **Where it works:** the ability and note fields, in the Edit dialog (`#ef-ability`, `#ef-note`, cards and rulebook
  pages) and inline (`.inline-ability`, `.inline-note`), marked `data-tokens`.
- **What it suggests** (`tokens.js`, pure helpers `openToken`, `suggest`, `applyChoice`, `vocabulary`): after `[`, the
  icons by `norm` (the site's lenient key, §1.6): exact name or alias, name prefix, alias prefix, then inside a name
  or alias (2+ letters), and only when none of those match (3+ letters) a typo's (a start one edit away, two from 6
  letters: `[fier` → fire, field); types before stars, then the most used in the cards' text, then by name; an exact
  name is followed by its variants. After `name:`, `VARIANTS` (= the bot's `ICON_VARIANTS`: tint, sticker, no, glow,
  darkglow). After `(`, the font's 341 badge tags (`badge_tags`, the cards' own first, the rulebook's old ones last;
  up to 7 characters, `(-1/2GCD)`) starting with what was typed; text that starts in lower case or holds a space
  never opens it ("(see below)"); "(Only once)" opens at "(O" and closes at "(On". Accepting writes the canonical
  `[name]`, `[name:variant]` or `(TAG)` over what was typed; text after the caret stays (`[fi|and ice` → `[fire]and
  ice`) unless the caret is inside a token closed by `]`/`)` on the line, which is replaced whole, a plain icon
  keeping its `:variant` (`[fi|re:glow]` → `[fire:glow]`). Until the 2026-10-23 review the rest of the word after the
  caret was replaced too, even with no `]` (it deleted words).
- **Keys:** ↑ ↓, Enter or Tab accept, Esc closes the list only (a capture listener stops it, so inline editing's Esc
  and the dialog's come on the next press); click or tap accepts (mousedown prevented: the field and the phone
  keyboard keep the focus); `execCommand("insertText")` keeps undo and fires the `input` the live preview listens to.
  ARIA: listbox/option rows, `aria-autocomplete="list"`, `aria-expanded`, `aria-controls` (while open),
  `aria-activedescendant` on the field (no `role=combobox`: it would hide a textarea's textbox role), a live count;
  the list sits inside the field's `[role=dialog]`. IME: no refresh while composing; keyCode 229 never accepts.
- **Variant pictures** (`build_site.py`): `build_icons` calls `variant_art`, which draws each with the font build's
  `icon_art` (the per-icon part of `build_stoogedings.build()` moved into a function the font build calls too; the font
  rebuilt byte-identical). tint and sticker take the text colour in the font, so their pictures are white masks
  style.css paints with `currentColor` (`.tokm`); glow and darkglow are the font's halo PNG with the icon on it
  (`.tokv`). Each sits in a square around the icon's box reaching `iconVariants` past it (0.0729 = OUTLINE/HEIGHT,
  0.3021 = GLOW_SIZE/HEIGHT, whole pixels at 96 px), with negative margins above and below, as the font's halo spills
  past the line. Cached by blob + a hash of the build script and tint mode; cold 12 s for 91 icons, warm 0.1 s.
  Needs numpy, opencv-python-headless and fonttools (the font build's requirements; `site/build/requirements.txt`
  includes them and site.yml sparse-checks out the script on that branch, and hashes it into the image cache's key);
  without them, one warning and the plain icon for every variant, never a failed build. text.js reads a variant
  exactly (`splitToken`, also the mock-up's): one the font has no ligature for (`[fire:]`, `[fire:glows]`,
  `[fire: glow]`) is letters, as on the card (2026-10-24; the parser still reads `[fire:]` as `[fire]`).
- **Tests:** AutoCroissant `tests/test_token_picker.py` (8; `AC_SITE_SRC` points it at a checkout; skips while the clone
  has no `tokens.js`; covers kept text after the caret, kept variants, typos, use order, 7-character tags); `tests/test_badge_parity.py::test_site_reads_variants_as_the_plain_icon` now covers data
  without pictures. Browser check: WebKit (WKWebView, offscreen) driven by
  `~/Documents/Tools/wt/scratch/picker/browser/shoot.swift`; screenshots in `.../picker/shots/`. **The harness hides
  IndexedDB on purpose:** WebKit wraps CryptoKeys stored there (the site's sealed sign-in) with a login-keychain
  master key, and a recompiled harness binary is a new keychain identity, so each run then waits on a keychain
  prompt (this happened 2026-10-23: item "shoot WebCrypto Master Key").

## 3. Missing-fonts audit (2026-10-19, re-checked 2026-10-20)

**Status 2026-10-23: no font is missing.** The owner added BabaUI, the last one (`MDW/Fonts/Baba UI Regular.ttf`, his
file, TTSCardMaker 92b5ae6b), after the Windows/Office faces of 2026-10-21 (a88b232). `python -m commands.font_check
--fonts MDW/Fonts --all .` (AutoCroissant, the fork venv) reads 812 card PSDs against 274 files / 282 faces and reports
"Every font the cards ask for is in MDW/Fonts." The audit below is the 2026-10-20 state.

**File names (owner, 2026-10-23, 92b5ae6b):** each file in `MDW/Fonts` is named after the font's full name
(`Cuphead Memphis Medium.ttf`, `Consolas Bold.ttf`); files of one font keep a version or kind suffix
(`Perpetua Bold 1.00.ttf` / `Perpetua Bold 1.75.ttf`, `EB Garamond Regular (Variable).ttf`). The same commit dropped 3
byte-identical copies.

**Status 2026-10-20 (later):** 7 PostScript names used by cards are still missing on the dev Mac. The CoreText check
(`swift /private/tmp/fontexplore/scripts/installed_fonts.swift`) and a scan of the PostScript name inside every
font file on the Mac (1,780 files, `find_fonts_by_psname.py`) agree.

**Added to `MDW/Fonts/` and installed that day (committed in 54278c1):**
- `Noto Color Emoji SVG.otf`: adobe-fonts/noto-emoji-svg 2.100.
- `Consolas.ttf`, `Consolas Bold.ttf`, `Consolas Italic.ttf`, `Consolas Bold Italic.ttf` (Consolas 5.22, four styles) and `Cambria.ttc` /
  `Cambria Bold.ttf`, `Cambria Italic.ttf`, `Cambria Bold Italic.ttf` (Cambria 5.96):
  - Extracted with macOS `bsdtar` from Microsoft's free PowerPoint Viewer (`PowerPointViewer.exe` → `ppviewer.cab`).
  - Fetched from the Internet Archive copy of the original download.microsoft.com URL; SHA-256 `249473568eba7a1e4f95498acba594e0f42e6581add4dead70c1dfb908a09423`.
  - The cabinet also holds Calibri (already in MDW), Candara, Constantia, Corbel and Meiryo, which no card uses.
- `MyriadPro-Regular.otf` (2.106): the copy bundled with Photoshop 2023. The Civilization VI copies are byte-identical.

**Still missing then** (visible runs; all added since, the last on 2026-10-23):

| Font | Visible runs | Where it could come from |
|---|---|---|
| Javanese Text | 15 | Windows |
| Pristina | 10 | Office |
| Papyrus-Regular | 5 | Windows Office version; the Mac's own Papyrus is PostScript `Papyrus` |
| OCR A Extended | 5 | Office |
| BabaUI | 4 | Unknown then; the owner's file, added 2026-10-23 (92b5ae6b) |
| Elephant | 1 | Office |
| MyriadArabic | 0 | Adobe Fonts. Apple's private 26-glyph subset was deliberately not added: installed as a user font it could be picked for real Arabic text and show empty boxes |

A missing font only matters when someone edits that text layer; the stored rendering stays.

**Method.** Scripts are in `/private/tmp/fontexplore/scripts/` (scratch); results are in `build/missing_fonts.json`,
`missing_detail.json` and `missing_fonts_by_card.md`.
1. For every PSD outside `MDW/` and `Markers/`, open it with psd-tools.
2. For each text layer, collect the **exact** PostScript name of every style run:
   `layer.resource_dict["FontSet"][run.StyleSheet.StyleSheetData.Font]["Name"].value`. Use `.value`, not `str()`.
   The first pass used `str()`, mis-parsed names containing apostrophes, and reported 27 instead of 25.
3. Compare against the PostScript names (name ID 6) of every font file in `MDW/Fonts/`.

A quick byte-scan alternative (`all_fontset_names.py`, regex over `/FontSet … /Name (þÿ…)`) found 158 distinct names,
including `AdobeInvisFont`. **25 PostScript names are used in card text but not supplied by MDW/Fonts:**

| Where to get it | Fonts (cards affected) |
|---|---|
| Photoshop bundles it privately (fine in Photoshop, absent elsewhere) | `MyriadPro-Regular` (18): Photoshop 2023 ships `Contents/Required/PDFL/Resource/Fonts/MyriadPro-Regular.otf`; it is also on Adobe Fonts |
| Adobe Fonts (activate in Creative Cloud; verified listings 2026-10-20) | `Consolas`, `Consolas-Bold`, `Consolas-Italic` (46/46/37: **the whole rulebook**), `Cambria` (1), `JavaneseText` (4: the Croissant cards, Magnum Opus), `MyriadArabic-Regular` (6; only on invisible characters such as a leading U+FEFF, so harmless) |
| Free download, **installed 2026-10-20** | `NotoColorEmoji-SVG` (1 card + 2 rulebook pages): adobe-fonts/noto-emoji-svg release 2.100, Apache-2.0, in `~/Library/Fonts` and copied to `MDW/Fonts/` (committed in 54278c1) |
| Microsoft Office for Windows only (not on Adobe Fonts) | `Pristina-Regular` (2), `Elephant-Regular` (1, a single apostrophe), `OCRAExtended` (1). Copy PRISTINA.TTF, ELEPHNT.TTF and OCRAEXT.TTF from `C:\Windows\Fonts` on a PC with Office (done 2026-10-21, a88b232; in the repo as `Pristina.ttf`, `Elephant.ttf`, `OCR A Extended.ttf`). `Century` was found and installed on 2026-10-19 |
| Font Book can download and activate | `BookAntiqua` (1), `CenturyGothic-Bold` (1) |
| macOS ships it (`/System/Library/Fonts/Supplemental`) | `ArialMT` (4), `ArialNarrow` (1), `ArialNarrow-Bold` (1), `ComicSansMS` (4), `ComicSansMS-Bold` (2), `Impact` (3), `TimesNewRomanPSMT` (7), `MicrosoftSansSerif` (1) |
| Near-miss | `Papyrus-Regular` (1: Papyrus): macOS's Papyrus.ttc is named `Papyrus`, so Photoshop still reports it missing. `Kung-Fu-Panda` (4): very likely the repo's `Kung Fu Panda Regular.ttf`, whose PostScript name is `Kung Fu Panda`, so it does not satisfy the lookup. 2026-10-23: renamed to `Kung-Fu-Panda` (work/kfp-name be8c96ff), and the Windows `Papyrus.ttf` is `Papyrus-Regular`, so both resolve exactly and the aliases are gone |
| Unknown source | `BabaUI` (1: Baba). Nothing by that name online (2026-10-20). It is presumably extracted from the game *Baba Is You*, which isn't in the dev Mac's Steam library. The closest public font is the itch.io "Unofficial BIY font", which has a different name, so Photoshop wouldn't match it. Added 2026-10-23: the owner's own file, `MDW/Fonts/Baba UI Regular.ttf` (92b5ae6b) |

Recommendations:
- (a) Before editing rulebook PSDs, activate Consolas through Adobe Fonts, or Photoshop will substitute and reflow.
- (b) The owner's call (2026-10-20): "for any fonts you find, add them to the fonts folder, there isnt any issue with
  having them in that folder." So found fonts, commercial ones included, go into `MDW/Fonts/` (and get installed).
  Note: the card-browser workflow checks out only `site/` (with the four hand-kept website files), `Types/` and the
  font manifest, and it publishes
  only `site/dist`, so `MDW/Fonts/` never goes onto the website. It is in the GitHub repo.
- (c) The audit only reads text layers, so the parser is unaffected by missing fonts. psd-tools reads stored text,
  not rendered glyphs.

**Push guard (2026-10-21).** Since 2026-10-25 the font check and the card check below are steps of `card-guards.yml`'s
check job (one install, one PSD download), whatever `CARD_GUARDS` says, each still posting its own report at once (only the
guards' fixes and post wait the quiet window); what follows describes them as `font-check.yml` and `card-check.yml` ran
them until then. `.github/workflows/font-check.yml` ran AutoCroissant's `commands/font_check.py` on every
push to main that changes a PSD or `MDW/Fonts/` (and by hand; `all` scans every card by blob, ~813 downloads): the PSDs
the push changed come by blob sha through the API, the fonts folder is a sparse checkout cached by its tree sha, the bot
files come from `bobberlington/AutoCroissant` (`font_check.py`, `github_api.py`, `global_config.py`; pip `psd-tools
fonttools requests`; since 2026-10-24, branch work/py314: `requirements-cards.txt` too, the fork from the card workflows'
wheel cache on Python 3.14). It reads the visible type layers' StyleRun fonts (plus the Normal sheet's default when no run covers
the text; an empty layer asks for nothing; hidden layers are ignored) and resolves them against the PostScript names (name
ID 6, else 4, else 1, as FreeType reports them: `Kung Fu Panda` -> KungFuPanda) of every top-level font file, exactly
(no aliases or loose matches since 2026-10-23, owner: "change the fonts to the exact ones and delete the aliases"). Findings exit 0 and go to the job summary, then,
when the repo secret `DISCORD_STANDARDS_WEBHOOK` is set (the standards channel since the owner's ruling of 2026-10-22;
`DISCORD_FONT_WEBHOOK`, the font channel, before, now deleted; the URL is in no file), one Discord
message, then a comment on the
commit (`permissions: contents: write`). The message: `<@&546427268751294475>` once, then per (author, font) "<login> has pushed a card
without adding the <font> font to MDW/Fonts, bite this user." with the cards and layers indented below; `allowed_mentions`
is that role only (the only role ping of the checks' posts; card-check and card-guards ping nobody, owner 2026-10-23). It never reverts or rejects; a failed webhook call only warns; a force push whose old head is gone checks the head commit's files. No developer-to-Discord mapping: the
author is the commit's GitHub login, else the pusher. The bot's `/fonts` runs the same check over the whole repo
(autocroissant-run-and-operate §7). State on 2026-10-21 (`--all` over the checkout, 133 s, 813 PSDs): six faces (BabaUI,
Elephant-Regular, JavaneseText, MyriadArabic-Regular, OCRAExtended, Pristina-Regular) on 15 cards, 29 visible layers (30
with Baba's hidden `Name`); the ten Mac-only faces are in `MDW/Fonts` since fb37995 + ec3a6f1 and Papyrus-Regular /
Kung-Fu-Panda resolved through aliases until 2026-10-23. State on 2026-10-23 (TTSCardMaker 92b5ae6b, BabaUI added, plus
work/kfp-name be8c96ff: `Kung Fu Panda Regular.ttf` renamed to PostScript `Kung-Fu-Panda`, the name the 4 cards ask for;
`Papyrus.ttf` is `Papyrus-Regular` already): exact names only, 812 PSDs against 274 files / 282 faces, nothing missing
(7 s with the process pool). Later standardization rules (AutoCroissant
docs/Campaigns/CARD_STANDARDIZATION_CAMPAIGN.md section 5) take this Action's shape. **Rule 2, the card check (2026-10-21):**
`.github/workflows/card-check.yml` runs AutoCroissant's `commands/card_check.py` on every push changing a PSD, `site/Card Edits.json`
or `Types/` (and by hand with `all`): the pushed PSDs by blob sha, Types/, the icon manifest and the edits file from the pushed tree,
no card-repo checkout at all; the bot files are `card_check.SPARSE_FILES` (card_check, font_check, github_api, psd_parser,
card_models, card_edits, global_config; pip `psd-tools fonttools requests`; since 2026-10-24, branch work/py314:
requirements-cards.txt in the list and the fork from the card workflows' wheel cache on Python 3.14). It reports what `/update_stats` prints for each card
(the parser's own notes such as NO STARS FOUND, MISSPELT TYPE, SEVERAL CREATURES are also the website's Parser notes; the
validator's HP/DEF/ATK/SPD NOT FOUND, STATS TOO HIGH, ABILITY TEXT NOT FOUND, UNKNOWN TYPE are printed only), after the card's
edits entry and `ABILITY_EXCLUSIONS`. Same three outputs and the same secret, but no ping; the Discord line is plain: "<login> has pushed <card>
with: <note>; <note>." An unreadable PSD or a failed API call is one error line, exit 0 (both checkers since 2026-10-21).
The first whole-repo dispatch ran on 2026-10-21 (run 39690899334; expected: Bubble and God_Tamer NO STARS FOUND, 20_Creature_Types
MISSPELT TYPE: tornado, Tainted_Lazarus, Pix, Twin_Emperors SEVERAL CREATURES, Computer_Virus HP/DEF/ATK/SPD NOT FOUND).

## 4. Photoshop automation (macOS, Photoshop 2023)

| Recipe or pitfall | Detail |
|---|---|
| Run a JSX **without Apple Events / Automation permission** | `open -a "Adobe Photoshop 2023" /path/script.jsx`. Photoshop executes it. There is no stdout, so the script must write its own log file (`new File(...).open("w")`) and you poll that file. Start scripts with `#target photoshop` and `app.displayDialogs = DialogModes.NO`. Save and restore `app.preferences.rulerUnits`. |
| Liveness check | A 2-line JSX that writes `app.documents.length` to a log (`/private/tmp/fontexplore/ps/ping.jsx`) |
| **ExtendScript reads script files as MacRoman** | Escape every non-ASCII character in generated JSX as `\uXXXX`. Prototype generators do `"".join(c if ord(c)<128 else "\\u%04X" % ord(c) ...)` and write with `encoding="ascii"`. Otherwise ★, typographic quotes and PUA icons are garbled |
| Create a text layer with **per-run fonts** (ActionManager) | `executeAction(charIDToTypeID("Mk  "), d)` with a `TxLr` descriptor holding `textKey` (the full string) and a `Txtt` list of `TxtS` ranges (`From`, `T   `). Each range sets `fontPostScriptName` (`stoogedings-Regular` for token runs, the body font otherwise), `Sz  ` (points = px·72/resolution) and `Clr ` (RGBC). Worked example: `/private/tmp/fontexplore/ps/make_edge_card.jsx` |
| **Restyle an existing** text layer's runs | `executeActionGet` on the target layer, then `getObjectValue(sID("textKey"))`. Read `textKey` plus the `textStyleRange` list (`from`/`to`) and the `paragraphStyleRange` list. Build new lists: copy each original style with `fromStream(...toStream())`, then override `fontPostScriptName`/`fontName`/`fontStyleName`. Remap paragraph ranges. Apply with `setd` on `TxLr`. **Photoshop's last style range runs one character past the text length**, so carry that extra over. Worked example: `/private/tmp/fontexplore/ps/convert_demo.jsx` |
| Fonts installed mid-session | `app.fonts.getByName(...)` does **not** refresh while Photoshop runs (a test logged "FONT NOT FOUND" right after install). Text layers that name the new PostScript font **do** resolve and render it. To see the font in the menu or `app.fonts`, restart Photoshop. After replacing `~/Library/Fonts/stoogedings.ttf` with a rebuild, restart Photoshop before trusting what it renders (reported 2026-10-19, not re-verified). |
| **A modal dialog blocks everything** | Missing-font, "update text layers" and save dialogs stop the queue. Scripts sent with `open -a` wait silently behind it until a human clicks. `DialogModes.NO` helps for script actions, but not for dialogs Photoshop raises on open. Check the screen when a log never appears |
| **Never quit Photoshop while the user has documents open** | Automation must not destroy unsaved work. The safe quit (`/private/tmp/fontexplore/ps/quit.jsx`) only runs `executeAction(charIDToTypeID("quit"))` when `app.documents.length === 0`, and otherwise logs and leaves it running. Scripts close their own documents with `SaveOptions.DONOTSAVECHANGES` and write outputs with `saveAs(..., asCopy=true)` |
| Verify output | Export a PNG copy per test and inspect it. psd-tools reading the saved PSD is the parser-side check (owned by autocroissant-validation-and-qa) |
| **Scripts can't read `~/Desktop` (macOS privacy)** | `$.evalFile(new File("~/Desktop/..."))` and `app.open` on Desktop paths fail with `IOError: I/O error` when the script is started with `open -a`. A human picking the file (File > Scripts > Browse) grants access. For automated tests, copy the script and PSDs to `/private/tmp` first (2026-10-20) |
| **ExtendScript is ES3: future reserved words are errors** | `var short = ...` fails with "Error 9: Illegal use of reserved word 'short'" as a modal dialog that blocks every queued script (2026-10-20). `node --check` does NOT catch this. Never name things `short`, `int`, `long`, `char`, `byte`, `float`, `double`, `boolean`, `class`, `const`, `enum`, `export`, `extends`, `final`, `goto`, `implements`, `import`, `interface`, `native`, `package`, `private`, `protected`, `public`, `static`, `super`, `synchronized`, `throws`, `transient`, `volatile`. Also avoid a global `var name`: it is the application's read-only `name` (assignments are ignored, so every file came out as "Adobe Photoshop__…") |
| Script prompts suppressed on the dev Mac (2026-10-20, the user's request) | `~/Library/Preferences/Adobe Photoshop 2023 Settings/PSUserConfig.txt` contains `WarnRunningScripts 0`, so `open -a … script.jsx` runs without the "Are you sure you want to run this script?" prompt. It applies after a restart |
| Load a script's functions without running its UI | `var EFFECT_LAYERS_NO_UI = true; $.evalFile(new File(copy))`. `Effect Layers.jsx` only calls `main()` when that variable is undefined |
| OT-SVG glyph support in Photoshop 2023 (probe 2026-10-20) | **Rendered:** `feGaussianBlur`, group `opacity`, linear/radial gradients, `clipPath`, embedded PNG. **Ignored:** `mask`. **Whole glyph dropped:** `feDropShadow`, and a filter chain with `feMorphology`/`feMerge`. The probe found glyph drawing **clipped to the line box** (font ascent/descent and the run's advance); the 2026-10-23 glow test drew embedded-PNG halos of 0.3 and 1.26 em whole, past it (§1.8) |
| Text-style limits found while hiding runs | `baselineShift` needs `impliedBaselineShift` set too, and is capped at about 6× the font size. `verticalScale` minimum is 1% (0 is ignored), and 1% still leaves hairlines on the baseline. The engine's `FillFlag`/`StrokeFlag` are **not** in the ActionManager `textStyle`: setting `fill`/`fillFlag` does nothing |
| Selections | Subtracting a soft selection multiplies (`a·(1−b)`), so identical anti-aliased pixels leave about 25% residue. Harden it: store the selection in a channel, `Thrs` level 128 on that channel, reload. `Modify > Expand` is a max filter. Selections stop at the canvas edge: use **Layer via Copy** (`CpTL`) to take only selected on-canvas pixels (no clipboard) |
| **Composite channels are stale inside a script** | `doc.selection.load(doc.channels[0])` (the red component) returned the card as it was before the script's layer changes, even after `app.refresh()`: 1.96M px selected for a 90k px layer. To read one layer's pixels as a mask, use **Apply Image** into an alpha channel: `AppI` with `With` = `Clcl` {`T   ` = ref(`Chnl` `Rd  `, `Lyr ` by name), `PrsT` false}, with the alpha channel as the only active channel |
| Fills and bounds | `selection.fill()` fills the **active** layer; set `doc.activeLayer` first. `layer.bounds` **includes layer effects** (a glow widens it); use `boundsNoEffects` for the text's own box |
| One undo step | `doc.suspendHistory("Effect Layers", "doJob()")` evaluates its string in global scope (pass data through a global). Check with `doc.historyStates[length-1].name`, not the count: the count stops growing at the 50-state cap |
| Clear a layer's style | `executeAction(sID("disableLayerStyle"))` is **Clear Layer Style**. It throws "The command Clear Styles is not currently available" when the layer has no style, so wrap it in try/catch |
| Copy a layer style | `copyEffects` on the source, then `pasteEffects` (with `allowPasteFXOnLayerSet`) on the target. This replaces the target's style. Apply a Styles-panel preset by name: `ASty` with a `Styl` name reference. List presets: `executeActionGet` on the application's `presetManager`, class `StyC` |
| What a Photoshop 2023 save changes (seen in the 2026-10-22 batches) | It rewrites the text composer to World-Ready, resets Indigo layer labels (the rulebook pages use them; a same-size `lclr` byte patch by layer id puts them back), and may re-render smart objects and blur filters outside the edited box (at most 24 levels: invisible, the owner accepts this Mac's exports). Its PNG exports come out 3-4x larger and without an ICC profile. Some PSDs hold empty Gradient Map layers that their PNGs predate (Dark_Mind_Eye, Kel: restored in batch B2) |

## 5. Migrating PSDs from image-layer icons to tokens

Status: **prototype only, not committed.** It exists only in volatile scratch:
- generator: `/private/tmp/fontexplore/ps/convert_demo.py` → `convert_demo.jsx`
- input: `/private/tmp/fontexplore/demo/spec.json`
- outputs: `demo/*_font.psd|png`, `compare_*.png`, `convert.log`

Copy it somewhere permanent before a reboot if it matters.

What it does for each card's `Ability` text layer:
1. Walks the text. Each run of 3+ spaces or tabs (a gap) becomes ` [token] ` in stoogedings. The tokens come from
   `spec.json`, precomputed **in the parser's gap order**, so the result matches what gap injection stored.
2. Sets each timing tag `(…)` matching the badge regex in stoogedings, so it becomes a badge. This prototype's regex
   (convert_demo.py:15) still holds the 70 old tags; the fixed walk `std/std_l1.jsx` already has all 341 (§1.2), so
   copy its `TAG` if the prototype is ever used for a batch.
3. Rebuilds the style and paragraph ranges (§4 recipe).
4. Hides the now-redundant inline icon layers: visible layers named like a Types/ icon whose top is at or below the
   midline.
5. Saves `*_font.psd` and `*_font.png` as copies.

Results (convert.log, 2026-10-19):

| Card | Badges | Tokens | Icon layers hidden |
|---|---|---|---|
| The Lich King | 3 | 1/1 | 2 |
| The Freezer | 1 | 4/4 | 4 |
| Gasoline | 0 | 5/5 | 5 |

**2026-10-20 version** (`/private/tmp/fontexplore/std/std.jsx`; scratch, not committed). This is the same walk, rebuilt
on `Effect Layers.jsx`'s helpers:
- it sends paragraphs back with the calibrated `textEveryLineComposer` (`layoutMode` + `withParagraphs`);
- it drops a leading U+FEFF (`withoutBom`).

All three demo cards converted with their Single-line Composer kept:
- The Lich King: 3 badges, 1 token;
- The Freezer: 1 badge, 4 tokens;
- Gasoline: 5 tokens, multicolor runs kept.

Lines still re-break, but only because font icons are narrower than the space gaps they replace. Before/after
sheets: `std/compare_*.png`.

**Don't migrate yet:** the owner is waiting for teammates to push their card updates first (2026-10-20).
Answered 2026-10-21 (fixed walk `~/Documents/Tools/psd-research/std/std_l1.jsx`; `l1_check.py` beside it re-parses
the simulated result offline: 41/41 equal to the parser, the old walk 27/41):
- **Spacing:** write what gap injection stores. One space each side of a token; none at a line start (text start,
  `\r` or soft return `\x03`), before `. , : ; ! ?`, or at a line end. A slash keeps both spaces: `[fire] / [machine]`
  (Gasoline, Electric Dart: the only slash cards). The old walk also put a space after Elemental World's soft return.
- **Token order:** icon order by line, then x, equals the parser's order on all 42 L1 cards (71 layers), so
  `spec.json` and the manifest's `std_jsx_tokens` are safe. 15 cards have no gap the parser counts, so it appends
  their tokens to the last line (a gap is `\s{3,}` within a line; a line's trailing spaces are stripped first):
  Blindness, Afraid, Paralyzed, Suffocation, Sickness and Plague (two icons on each of the last two) end in a stripped gap; Burning,
  Poison, Smoldering, Stuffed, Blazing, Chilled, Engorged, Frostbite and The Farlands have none. The old walk dropped
  the tokens on all but the first four; the fixed one appends on all 15 (its gap class is the parser's `\s` minus
  line breaks).
- Leave **Tainted Lazarus** out: its below-midline `undead` is the flipped half's header icon, not inline.
- The old hide rule also hid The Lich King's `Attack` stat label (a text layer); text layers are now skipped.

**Caveat: reflow strands overlays.** Font icons are narrower than the hand-placed icon images, so lines re-break.
Text that a card author placed as a separate layer over the ability text, such as **Gasoline's glowing red "Blazing"
layer**, stays put while the word it decorated moves. `compare_Gasoline.png` shows the stray "Blazing" floating at
the end of the last line. Any batch migration needs a per-card visual check, and either a list of overlay layers to
move or manual fixes.

Remaining scope:
- On 2026-10-19 **no card PSD used stoogedings yet**: 0 of 813 PSDs have a text layer in the font. Visible icon
  layers named like a Types/ icon, with their top at or below the midline, were measured two ways with psd-tools
  (scan took about 2 min):
  - **45 PSDs / 81 layers** counting smart-object layers only. This matches the earlier survey (`survey.py`) and is
    the "inline icon" number usually quoted.
  - **61 PSDs / 148 layers** counting any layer kind, including rasterized pixel layers. Some of these are not inline
    with the ability text (e.g. footer or decoration icons), so check each one before hiding it.
- After migrating a card, its `/update_stats` output must match the pre-migration stats.pkl except for intended token
  changes. Use the parser-regression method in autocroissant-validation-and-qa.

## 6. Effect layers: word effects that keep the card's font (2026-10-20)

**Problem.** Status words such as Gasoline's "Blazing" got their glow from a hand-placed overlay: a second text
layer holding a copy of the word, with a layer style. Overlays go stale silently when the text is edited.
Survey: 53 overlays on 40 cards (impossibility-cards-reference §4.2).

**Rejected, measured in Photoshop 2023:**
- **Effects baked into a font (an "effect twin" of the body font):**
  - 652 of 765 cards outline their text with Stroke and Drop Shadow on the `Text` group, so a baked-in glow gets a black outline.
  - OT-SVG glyphs are clipped to the line box (2026-10-20 probe; embedded-PNG halos are not, §1.8).
  - It would need one font per body font and effect, and 95 body fonts are in use.
- **Ghost fonts** (metric-identical, invisible): the version with empty outlines shifted 17 px. The version with an empty SVG table lined up exactly, but needs every body font copied onto every machine.
- **Text-style tricks:** baseline shift is capped, vertical scale bottoms out at 1% (which leaves hairlines), and the fill flag can't be scripted (§4).

**Adopted: pixel-cut effect layers.** `MDW/Scripts/Effect Layers.jsx` builds them:
1. **Key copy:** a duplicate of the text layer with the chosen words recolored white and every other run black. Only color changes, so the layout is the text's own.
   - An earlier version squashed the other runs to 1% vertical scale and subtracted an all-squashed copy.
   - That moved centered lines by about 2 px wherever kerning depends on glyph shape (Monkey D Goofy).
2. **Mask** (`selectKeyed`): the key copy is merged onto a black layer, so it is opaque everywhere.
   - **Apply Image** (`AppI`, source = that layer's red channel) copies it into a scratch alpha channel.
   - That channel is loaded as the selection and kept wherever it is above 0 (`hardenSelection(doc, 1)`).
   - If the mask has more pixels than the key layer itself, it throws instead of building a wrong layer.
3. **Pixels:** **Layer via Copy** (on-canvas pixels only) from an untouched rasterized duplicate gives a pixel layer named `<text layer> FX: <words>`. Its pixels are the card's own.
4. **Placement:** the layer goes above the text's top group, outside the group's stroke and shadow. The words keep the card's font, size and color.
   - **The edit pipeline makes the same layers (AutoCroissant `commands/psd_edit.py` `set_text_layer`, design §5 step 4 and §7 idea 14):** since BUG 2 (2026-10-23, branch `work/fx-rebuild`, not landed) an ability edit runs this script's **Update all** on the new text: every `<layer> FX: …` layer is cut again in place (same key copy, phrase matching and badge rule; its record, style and place kept), a phrase gone from the text leaves the name, a layer with none left goes, every layer keeps its own style (q6, unattended call 2026-10-23, owner to confirm: the chart's style only on a new layer), and a styled status word no layer covers gets one above `Text` (the **Add** naming). Layers carrying a chart status's style (Monkey D Goofy's `1#1`) and, since 2026-10-23 (owner), layers in any other style (Fire Staff's Smoldering stroke) are rebuilt, each in its own style. A layer in a style of its own stays on its own words (review: a "1" typed before Goofy's renames it `1#2`, which this script's Update all cuts on the same "1"; a whole-word "1" anywhere refuses, since `1#1` would move to it). A typed badge or icon token is set in stoogedings at the card's own badge style (owner 2026-10-23). Badge layers (queue A's gradient cards) are retired since 2026-10-25: the edit deletes one, and this script's Add and Update all refuse a phrase that takes in a badge or icon (§1.9). Checked against this script's Update all run in Photoshop on 8 baked copies: same boxes, style and place, IoU 0.955-1.0, no differing pixel on the shared ink (2026-10-21: Convert overlays, IoU 0.966 Gasoline / 0.937 Royal Eradicator Main Cannon).
5. **Swatch colors:** `readAco` parses `MDW/Status Effects.aco` and `MDW/Attributes.aco` (101 RGB swatches) in ExtendScript.
   - **Add** (checkbox, on by default) and the batch's `AUTO_STYLES` color the words with the swatch of the same name first (`colorWords`). Two-tone statuses use "<name> 1"; their `.asl` gradient draws the rest.
   - **Convert overlays** keeps the overlay's own color, and notes when it differs from the swatch.
   - **Survey (2026-10-20):** status words already use their swatch about 95% of the time (e.g. Enfeebled 20/20, Smoldering 20/23). Stat words (ATK/HP/DEF/SPD) match only about half the time, so the batch never recolors those.
6. The style comes from a Styles-panel preset (the 13 in `MDW/Debuff And Attributes Layer Effects.asl` were loaded on the dev Mac), from another layer, or none.

**Names:**
- `|` separates phrases.
- Matching ignores case and whitespace (line breaks included) and wants whole words. When a phrase only occurs inside other text ("2T1 Become…") it falls back to plain substring matches.
- `word#n` means only the nth occurrence.
- The parser understands both rules.

**Buttons:**
- **Add**
- **Update all:** rebuilds every effect layer from the current text, keeping style, opacity, fill and blend mode, as one history step.
- **Convert overlays:** shown when the card has overlays.

**The round-trip trap, the reason for `layoutMode`.**
- Photoshop reads paragraph styles back *without the line composer*, and resets it to **Every-line** when they are sent back with `setd`. Cards set with **Single-line Composer** then re-break every line: 17 of 40 in the trial. Found by diffing the engine data (`EveryLineComposer False → True`).
- Leaving `paragraphStyleRange` out keeps a no-op round trip exact, but loses alignment once style runs are split (Monkey D Goofy).
- So `layoutMode` tries `textEveryLineComposer` false, then true, on a hidden copy. It keeps the better match if it is within max(50 px, 1% of the text's pixels), and every `setd` uses it.
- The result is cached per document and layer (`doc.id:layer.id`). A cache keyed by layer id alone leaked between documents in a batch, because layer ids repeat across files.
- A leading U+FEFF is dropped by `withoutBom` whenever text is sent back. Otherwise Photoshop gives it its own run in fallback MyriadArabic, which moved Dark Nebula's whole paragraph up 12 px. The parser already ignores the character.
- The textStyle key `autoKern` is also missing from the read-back on many runs, and `AutoKerning False` runs come back as metrics. Calibration catches the cases where that changes pixels.
- If neither matches (the text was last laid out with other font versions), it refuses with "edit the text once by hand, then run again".

**Status styles (15, since 2026-10-20).** `MDW/Debuff And Attributes Layer Effects.asl` holds the original 13 styles
(10 status styles, plus Higher Being, God and Creator glows). Five more were appended, copied byte-for-byte as effect
descriptors from Croissant's chart `MDW/Croissant's Special Folder/Satus Effect Colors.psd`:
- Cursed Inferno: outer glow
- Glued: outer glow
- Curse: drop shadow + 1 px stroke
- Become Back Money: gradient
- Corrupted Blood: gradient

How the file was edited:
- **Kept as-is:** the 13 original entries (byte-identical) and the `8BIMphry` preset hierarchy. The hierarchy is one
  group, "Debuff And Attributes fx", followed by positional `preset` items and a `groupEnd`; five presets were added.
- **The writer:** `/private/tmp/fontexplore/asl/build_asl.py` (scratch). psd-tools `Descriptor.read`/`write`
  round-trip every entry exactly. Entries are padded to 4 bytes.
- **The chart's effect blocks:** they list disabled placeholders under the `*Multi` keys, while the enabled effects
  sit under the single keys (`DrSh`, `FrFX`, `GrFl`, …).

Getting them into Photoshop:
- On the dev Mac, a 5-style `.asl` was loaded with `app.load`, as the group "Debuff And Attributes fx (new)", so the
  13 already in the panel weren't duplicated.
- Teammates load the repo's `.asl` once.

The batch's `STATUS_STYLES` lists all 15, longest first. Whole-word matching keeps "Curse" from matching "Cursed".
Auto styling still changes only 3 cards: the 5 new statuses appear only on cards whose overlays already style them.

**Converting an overlay** (`findOverlays` / `convertOverlays`):
- **What counts:** a visible, short text layer (≤ 40 characters), at least 80% inside the visible `Ability` box, whose text is in the Ability text and whose name isn't a stat label. The Dark Nebula "ATTACK" stat label was caught by the earlier overlap-only rule.
- **What it does:**
  - copies the overlay's color onto those words in the text;
  - builds the effect layer at the overlay's place in the stack, with its style, opacity and blend mode;
  - deletes the overlay.
- **Occurrences:** when a phrase has more occurrences than overlays, it builds probes per occurrence and keeps the ones the overlays sit on.
- **Hidden-`Ability` cards are skipped:** Bulborb Larva, Butterfly, Empress Bulblax and Apex Predator re-lay their text out in separate layers.

**Batch: `MDW/Scripts/Effect Layers - All Cards.jsx`.** `AUTO_STYLES` is **on by default** (the owner's choice, 2026-10-20). On that day it would add effects to exactly 3 cards after overlay conversion: the Hexed status card and Panseida (Hexed ×2 each), Credit Card (Frozen ×1). It also applies to every future card that mentions one of the 10 styled statuses. It opens every card PSD (skipping MDW, Markers, Rulebook, Types and site) and, for each:
1. converts overlays;
2. rebuilds stale effect layers (`isStale`: over 5% of the layer's pixels left after subtracting the text's, the same test as the parser);
3. `AUTO_STYLES` (on by default; set it to false to only convert and update): adds effect layers, with swatch colors, for status words that have a same-named preset, using only occurrences not already covered.

Changed cards are saved, and their PNG is re-exported. Cards with a non-sRGB profile (often "Display") are converted to sRGB first, which matches how the repo PNGs were made: about 0.9% of pixels differ by rounding, and cards without a profile match exactly. The summary goes to `effect-layers-all-cards.log` in the temp folder. For tests, `EFFECT_LAYERS_ALL_NO_UI` plus `runAll(root, progress)` runs without the progress window or alert.

**Trial on copies of the 40 overlay cards** (final run 2026-10-20, through the script's own `convertOverlays`; `/private/tmp/fontexplore/fx/mig/`, scratch):
- AutoCroissant parses all 40 identically before and after, and reports no effect-layer problems.
- Only the effect words change: at most 7,679 changed pixels per card (Standstill), most under 3,000.
- **Batch test** on a 13-card copy repo (overlay cards, styled-word cards, a stale effect layer, plain and hidden-Ability cards, an MDW PSD and two rulebook pages, which must be skipped): run 1 changed 9, run 2 changed **0**.
- Single-card harness T1–T7 pass (`/private/tmp/fontexplore/fx/exp7b.jsx`), including one "Effect Layers" history step and swatch recoloring.
- Showcase sheets: `mig/showcase/effect_layers_1..4.png`.

## When NOT to use this skill

| Task | Use instead |
|---|---|
| How the parser reads font tokens, canonical forms, the four ICON problems, gap injection | impossibility-cards-reference (§4.1, §4) |
| Values and defaults of `ICON_FONT_PREFIX`, `ICON_MANIFEST_PATH`, `ICON_VARIANTS`, and the type_emojis constants | autocroissant-config-and-flags |
| Running `/update_stats`, `/sync_emojis`, reminders | autocroissant-run-and-operate |
| Gating a card data commit (also a website publish) | autocroissant-change-control |
| Proving a parser change is neutral (old-vs-new parser over all PSDs, sandboxed sweep) | autocroissant-validation-and-qa |
| Discord-side symptoms (emojis not rendering, command missing) | autocroissant-debugging-playbook |

## Provenance and maintenance

All facts verified 2026-10-19 (TTSCardMaker HEAD 3a268ca); §1.2 badges and §1.6 matching rules updated
2026-10-20; commit status, Pages, §2.2 workflow and sparse paths re-verified 2026-10-20 against 027fd0d (commit list above extended to 13ac680 the same day from the GitHub API); §2.6 (tts-save.yml,
env overrides instead of the symlink) written 2026-10-20 against the uncommitted working tree on 15a16be; the website-edits
bullet's kinds and badge keys (`PSD_FIELDS`, `pending`/`refused`/`reparsePending`) written 2026-10-20 against the uncommitted
working tree on d7c6c15 (`grep -n "PSD_FIELDS\|valid_verdict\|reparsePending" site/build/build_site.py`); `norm()` re-cited at :102 the same day; §0's font-check row and §3's push-guard paragraph written 2026-10-21 against the
uncommitted font-check.yml on 8393b3b (that day's `--all` numbers; step order, FreeType-style names and the force-push fallback re-checked the same day, review); §2.6's bot checkout list gained
`commands/git_commit.py` 2026-10-21 against the uncommitted tts-save.yml on f5a0e50 (review: tts_save.py:33,37 import it); §6 step 4's
pipeline bullet written 2026-10-21 against AutoCroissant's uncommitted `commands/psd_edit.py` (`fx_name`, `set_text_layer`) on ac3ae57 and the
JSX trial copies under `/private/tmp/fontexplore/fx/mig/out/`. §0's module list, §2.4's four website rows (profiles/People, inline editing, Details dropdown, published decks), §0's card-check row and §3's rule 2 written 2026-10-21 against TTSCardMaker a3451c8 and AutoCroissant df4a439. Run the commands from
`~/Desktop/TTSCardMaker` unless they say otherwise. §2.6 rewritten 2026-10-22 (gate with the bot's data, commit subject, the
save's Lua, `TTS/checks/`) against tts-save.yml and TTS/checks as committed that day ("TTS save: daily gate also counts the bot's pickle
pushes", "TTS checks: the audit's checks, re-runnable"). §0's card-preview row and §2.7 written 2026-10-23 against TTSCardMaker work/card-preview (card-preview.yml, NOTES §19) on 060b24a3 and AutoCroissant work/card-preview; revised the same evening after review (two jobs, `edit` echoed as sent, no PNG on the branch, privacy, error kinds, budget). §2.8 and the token-picker notes in §1.6, §1.8, §2.1, §2.3 and §2.4 written 2026-10-23 on branch work/token-picker of both repos (TTSCardMaker a38003f9, AutoCroissant d9ebf7c), re-checked after that evening's review fixes; site/NOTES.md §20 is the picker's. §3's 2026-10-23 status, the file-name rule and every `MDW/Fonts` file name
re-checked 2026-10-23 against TTSCardMaker 92b5ae6b (`font_check --all`: 812 PSDs, 274 files, 282 faces, 0 missing). §2.4's editor row (previews, drafts, stars and types) written 2026-10-23 against
TTSCardMaker branch work/site2-editor; §2.1's web fonts row and §2.4's image previews row written 2026-10-23
against TTSCardMaker branch work/site2-previews (`grep -n "def web_fonts\|def font_faces" site/build/build_site.py`;
`site/assets/preview.js`, `mockup.js`; AutoCroissant `tests/test_site_previews.py`); both merged on branch
work/site-round-2 (not on main yet; its NOTES sections are 21 and 22 since main took 19 and 20); both rows updated
2026-10-23 evening for that branch's review fixes (tc-site2 c8a432c7: `grep -n "export function bestBlock\|export function strayLines" site/assets/mockup.js`). Refresh the
date-stamps whenever you touch this file.

```bash
# Font identity, icon count, codepoint range, badges, GSUB lookup types (expect: stoogedings-Regular, 0xe000..0xe05a, ★ mapped, 341 badges, [1, 6, 4, 2, 1, 1, 6, 4], 523 SVG docs; with :glow [1, 6, 4, 2, 1, 1, 6, 4, 7, 7] and 705; lookups [1, 6, 4, 2, 4] before the any-case build, 340 / 522 before 2026-10-22, 70 / 252 before 2026-10-21)
MDW/Fonts/stoogedings-src/.venv/bin/python -I -c "from fontTools.ttLib import TTFont; f=TTFont('MDW/Fonts/stoogedings.ttf'); c=f.getBestCmap(); print(f['name'].getDebugName(6), hex(min(x for x in c if x>=0xE000)), hex(max(c)), 0x2605 in c, sum(n.startswith('badge_') for n in f.getGlyphOrder()), [l.LookupType for l in f['GSUB'].table.LookupList.Lookup], len(f['SVG '].docList))"
# Manifest: 91 icons (35 type / 56 star), contiguous PUA, 15 with aliases
python3 -c "import json,collections as C; m=json.load(open('MDW/Fonts/stoogedings-src/stoogedings.json'))['icons']; print(len(m), C.Counter(v['kind'] for v in m.values()), sum(bool(v['aliases']) for v in m.values()))"
# Byte-identical rebuild: copy Types/, the two input fonts and stoogedings-src/ to /tmp/x/ (same layout), build there, then cmp both outputs (expect no output from cmp)
# Badge vocabulary (expect 10 fixed + Action + 9 patterns over "123456789X" + a/bCD, a/bGCD over a in -1, 0-9, X and b in 1-9, X) and size constants
grep -n "def badge_tags" -A14 MDW/Fonts/stoogedings-src/build_stoogedings.py
grep -n "^BADGE_" MDW/Fonts/stoogedings-src/build_stoogedings.py
# The website's badges carry no extra multi-turn symbol (expect no output)
grep -n "multi::after" site/assets/style.css
# Ignore rules
git check-ignore -v MDW/Fonts/stoogedings-src/.venv site/dist site/.cache          # three hits
# Workflow triggers and sparse paths
grep -n "cron\|workflow_dispatch\|branches\|sparse-checkout:\|filter:" .github/workflows/site.yml
# TTS save (card-sync.yml's step since 2026-10-24; tts-save.yml is gone, expect "No such file"): the two env overrides, --names, the message
ls .github/workflows/tts-save.yml; grep -n "AUTOCROISSANT_\|commands.tts_save\|message.txt" .github/workflows/card-sync.yml
# TTS save checks: all pass once Card sync has run since the last name change (TTS/checks/README.md)
python3 TTS/checks/run_all.py
# Card and font checks: steps of card-guards.yml's check job since 2026-10-25 (font-check.yml and card-check.yml are gone, expect "No such file"; no "discord.com" hit)
ls .github/workflows/font-check.yml .github/workflows/card-check.yml; grep -n "commands.font_check\|commands.card_check\|DISCORD_STANDARDS_WEBHOOK\|discord.com" .github/workflows/card-guards.yml
# Site data source and deck rules
grep -n "^BOT_REPO\|^CARD_REPO" site/build/build_site.py; grep -n "tiers\|FIELD_MAX_STARS =" site/assets/deck.js
# Card data reachable (the site build reads the data branch's db/ via CARD_DATA_DIR; expect one hit)
git ls-remote origin refs/heads/data
# AutoCroissant side (run in ~/Desktop/AutoCroissant)
grep -n "^ICON_FONT_PREFIX" commands/psd_parser.py; grep -n "^ICON_MANIFEST_PATH" global_config.py   # 28, 25 (split 2026-10-20; 25 re-checked 2026-10-21)
grep -n "^MAX_NAME_LEN\|^NO_SUFFIX\|^ICON_FOLDERS" commands/type_emojis.py     # 22, 24, 23
# Missing-font audit inputs still present? (scratch; may be gone after reboot)
ls /private/tmp/fontexplore/build/missing_fonts.json /private/tmp/fontexplore/ps/convert_demo.jsx
```

Maintenance rules:
- If the website's `norm()` is aligned with the parser, update the §1.6 website row.
- If the migration prototype is moved into the repo, replace every `/private/tmp/fontexplore` path.
- When a tag or icon is added, re-run the first two provenance lines and update the counts (91 / 341 / 126 /
  U+E05A).
2026-10-21 (details): section 2.1 data-flow row and the `data/details/<id>.json` output line for the card Details dropdown (plan item 23; the bot side is autocroissant-run-and-operate's "PSD details").
2026-10-21 (cooldowns): §1.1, §1.2 (340 badges, the cards' tags, tooltips, size and widths, the three places), §1.4's expected output and byte-equality note, §1.5's SVG count, §5 step 2 and the provenance expectations re-verified against TTSCardMaker 8ec1b92 (font), c3f9d68 and 73904db (site); font line above run on the committed TTF (340, [1, 6, 4, 2, 4], 522); shaping `/tmp/cdverify/shape.py` and site `/tmp/cdverify/harness.mjs` 0 problems.
2026-10-22 (Action): §1.1, §1.2 (341, the `neutral` group, tooltip, the three places), §1.5's SVG count, §5 step 2 and the provenance expectations against TTSCardMaker f6ce500 (font, site) and 6e7fc0c (notes); font line above run on the committed TTF (341, [1, 6, 4, 2, 4], 523); the three-places parity is `tests/test_badge_parity.py`. Later the same day: the batch copies as a fourth place (`test_batches_match_font`), the "clone behind" skip, the walk's font rule and the Stats chart's `neutral` exclusion (TTSCardMaker b686d0c).
2026-10-22 (any case): §1.1, §1.2 (the CASE lookup, its proof, the Photoshop check and fallback, the three places), §1.4 step 3 (the packer pin), §1.5's lookups and the provenance expectations, against TTSCardMaker branch work/badges-any-case.
2026-10-23 (glow): §0's font row, §1.1, §1.4's output, §1.5 (lookups, SVG count, the CoreText row), §1.6 and §1.8 against TTSCardMaker branch work/glow-tokens (3539dc83) and AutoCroissant branch work/glow-tokens; font line below run on its TTF.

2026-10-23 (docs drift): the balance analytics (TTSCardMaker af826714, 2026-10-22) in §0's website-source row, the §2.1 data-flow table, the data files and the feature table (the draft in session-rescue-2026-10-22/resume/companion_analytics_line.md, verified at af826714; `page_chain` folds re-reads by when since 7a87d91a). §4's Photoshop 2023 save row from the 2026-10-22 session notes (SESSION2_NOTES.md 07:3x, 08:3x) and the archived RESUME_2026-10-22.md.

2026-10-23 (BUG 2, AutoCroissant work/fx-rebuild): §4's edit-pipeline bullet says the bake runs Update all on the new text (in place, badge and chart-style layers rebuilt, others refuse), checked against this script's Update all in Photoshop on 8 baked copies.
2026-10-23 (BUG 2 review): §4's bullet adds the follow rule for chart-style layers (Goofy's `1#2`, checked with this script's Update all in Photoshop: IoU 1.0, as for PoD11ZOMG's renamed badge layer) and the typed-badge refusal.
2026-10-24 (round 3): §2.1 output data files (analytics.json ~130 KB with revisions and archived; contributions.json and changelog.json keys) and §2.4 rows (search without has:details, Balance without the cooldown update, Ratings, Stats round 3, Archive tab, person stats, pinned tray), against TTSCardMaker branch work/r3-integ.

2026-10-24 (py314, branch work/py314 of both repos): every Action on Python 3.14; the fork in six workflows (card-check
and font-check join), one wheel cache entry, card-sync one venv; the card guards without R7, G2 rendered anyway, the
quiet window an environment (AutoCroissant docs/Campaigns/PUSH_GUARDS.md section 20).

2026-10-24 (speed pass): §2.4's Speed pass row, against TTSCardMaker branch work/site-perf (front end, stamps by content, site.yml `paths`, `build.txt`); §2.1 cache-busting.

2026-10-24 (BUG 2 phase 6, AutoCroissant work/fx-rebuild): §1.8's bake bullet and §2's Website edits bullet say content problems bake, told; what the site lacks to show a baked verdict's notes.

2026-10-24 (pictures): §2.4 Card images (AVIF, IMAGE_BUDGET, no service worker), §2.1 cache-busting (deploy gate back, icon stamps) and the link-preview row, against TTSCardMaker branch work/site-img.

2026-10-25 (retire): /update_tts_save retired (branch archive/discord-edits).

2026-10-25 (step 10, AutoCroissant and TTSCardMaker work/pickles): §2.1's inputs are the data branch only (no pickles, `aliases.pkl`, `load_bot_*` or `CARD_DATA` switch), the build's failure rule and §2.5 (the data commit reaches the site, not a /push) rewritten.

2026-10-25 (clean, work/clean): the card and font checks as card-guards.yml check-job steps (font-check.yml and card-check.yml folded in), the shared cards-python composite action, site.yml on site code only, the guards' R8 always switching and their Follow up (card sync, bake=pending).
