---
name: ttscardmaker-companion
description: Load this skill for anything on the TTSCardMaker side of the project (the card-art repo at ~/Desktop/TTSCardMaker) that AutoCroissant depends on. Trigger words include stoogedings, icon font, "[fire]" tokens, [x:tint], [x:sticker], [x:no], ★ star glyph, Private Use Area / PUA codepoints (U+E000…), stoogedings.json manifest, build_stoogedings.py, rebuild the font, add/rename an icon in Types/ or Types/Stars/, timing tags and badges such as (1PGT), (OYT), (E2T), badge size, type emojis, application emojis, the _no crossed-out emojis, /sync_emojis naming, the website, card browser, site/, build_site.py, site.yml, GitHub Pages, deck builder, Tabletop Simulator export, share links /c/<slug>/, Open Graph previews, "the website is stale", missing fonts, font audit, "Photoshop says fonts are missing", Photoshop scripting, JSX, ExtendScript, ActionManager, `open -a "Adobe Photoshop 2023"`, migrating PSDs from image-layer icons to tokens, status-effect word effects (glow/stroke on "Blazing"), overlay text layers, effect layers "Ability FX: …", Effect Layers.jsx, Effect Layers - All Cards.jsx, Single-line vs Every-line Composer reflow, PSUserConfig WarnRunningScripts, and the MDW folder's .asl/.aco/templates. Contains the font's token and badge grammar, manifest schema, codepoint rules, rebuild/install checklist, tested Photoshop/CoreText renderer facts, a font-vs-parser-vs-emoji-vs-website matching table, a cross-repo "what breaks if you rename an icon" table, the website's data flow, workflow, Pages setup, local preview and feature-to-file map, the 25-font missing-fonts audit with where to get each, Photoshop automation recipes and pitfalls, and the status of the PSD migration prototype.
---

# TTSCardMaker companion: icon font, card website, fonts, Photoshop automation

AutoCroissant (the Discord bot) reads card PSDs from **TTSCardMaker** (`MichaelJSr/TTSCardMaker`, cloned at
`~/Desktop/TTSCardMaker`; a blobless partial clone that is also shallow — depth 1, so `git log` there shows only
the latest commit (3a268ca on 2026-10-19); use GitHub for history). On 2026-10-19 three things were added on the TTSCardMaker side
that the bot now depends on or feeds: the **stoogedings** icon font, a static **card-browser website**, and a
**missing-fonts audit**. Photoshop scripting recipes were worked out along the way. This skill is their operational
home. Facts verified 2026-10-19 against TTSCardMaker HEAD `3a268ca`. **Everything described here is still
uncommitted in TTSCardMaker** (`MDW/Fonts/stoogedings.ttf`, `MDW/Fonts/stoogedings-src/`, `site/`, `.github/`).
Nothing is pushed yet, and GitHub Pages is not enabled.

The parser side of tokens (how `psd_analyzer.py` reads a layer set in the font, plus the four new problems) is owned
by **impossibility-cards-reference §4.1**. This skill only summarizes it.

---

## 0. What lives where

| Thing | Path (TTSCardMaker unless noted) | Generated or source |
|---|---|---|
| The font | `MDW/Fonts/stoogedings.ttf` (5,142,544 bytes, PostScript name `stoogedings-Regular`, family `stoogedings`) | Generated, but committed |
| Build script | `MDW/Fonts/stoogedings-src/build_stoogedings.py` | Source |
| Manifest | `MDW/Fonts/stoogedings-src/stoogedings.json` (icon names, aliases, codepoints, tint modes) | Rewritten by every build. Hand edits survive |
| Build deps | `MDW/Fonts/stoogedings-src/requirements.txt` (fonttools, opencv-python-headless, numpy, pillow) | Source |
| Build venv | `MDW/Fonts/stoogedings-src/.venv/` (gitignored by `stoogedings-src/.gitignore`) | Local only |
| Icon sources | `Types/*.png` (35 "type" icons), `Types/Stars/*.png` (56 "star" icons) | Source art |
| Letter / badge fonts the build embeds | `MDW/Fonts/BreeSerif-Regular.ttf` (OFL letters), `MDW/Fonts/ChangaOne-Regular.ttf` (badge labels) | Inputs |
| Website source | `site/index.html`, `site/assets/{app,search,deck,stats,text}.js`, `site/assets/style.css`, `site/build/build_site.py`, `site/README.md` | Source |
| Website output | `site/dist/`, thumbnail cache `site/.cache/` (both gitignored by `site/.gitignore`) | Generated |
| Pages workflow | `.github/workflows/site.yml` | Source |
| Installed copy of the font on the dev Mac | `~/Library/Fonts/stoogedings.ttf` (identical to the repo build on 2026-10-19) | Install |
| Bot code that depends on the font | AutoCroissant `commands/psd_analyzer.py` (`ICON_FONT_PREFIX`, `ICON_MANIFEST_PATH`, `IconVocabulary`), `commands/type_emojis.py` | See §1.7 |
| Card authors file | `MDW/Card Authors.txt`: `[people]` git name = display name, `[cards]` Card = person. The bot (every `/update_stats`) and the website build both apply it; it wins over the database. Seeded 2026-10-20 from stats.pkl | Source, uncommitted |
| Effect-layer scripts | `MDW/Scripts/Effect Layers.jsx` (one card: Add / Update all / Convert overlays), `MDW/Scripts/Effect Layers - All Cards.jsx` (every card) | Source, uncommitted (§6) |
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
| `★` (U+2605) | The gold `star` icon | `[star]` |
| A PUA character (U+E000–U+E05A) inserted from the Glyphs panel | That icon | `[name]` |
| `(1PGT)` and the other 69 timing tags | A colored pill badge. The text is still `(1PGT)` | `(1PGT)`, unchanged |

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
- **Badges are case-sensitive.** Only the exact uppercase `(1PGT)` becomes a badge.

**Specimen QA (2026-10-20, Photoshop 2023):** sheets at `/private/tmp/fontexplore/spec/` (scratch). No defects found:
- **Variants:** all 91 icons render in all four variants; all 70 badges render in their group colors.
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
- **Tags without badges:** the commonest are `(n points)` (68) and `(Action)` (36). They aren't rulebook
  abbreviations, so they get no badge unless the owner wants one.

### 1.2 Timing-tag badges (70)

**Badges are part of the font going forward** (the user adopted them on 2026-10-19). `badge_tags()` in
build_stoogedings.py defines 10 fixed tags plus 6 patterns × 10 values of n (1–9 and X). That makes **70**,
all verified in the built font as `badge_*` glyphs. The colors come from the rulebook page "02
Abbreviations" (`BADGE_GROUPS`):

| Group (color) | Tags |
|---|---|
| turn, on your turn (blue `#4f86f7`→`#2563eb`) | `OYT`, `nPT`, `EnT`, `FTEnT` |
| any, at any time (purple `#9f67ff`→`#7c3aed`) | `AAT`, `nPGT`, `EnGT` |
| forced (red `#ef5050`→`#dc2626`) | `BOT`, `ENDT`, `BOGT`, `ENDGT` |
| trigger, per game and triggers (orange `#e08a1e`→`#b45309`) | `OFS`, `OS`, `OD`, `WOF`, `nPG` |

Shape (revised 2026-10-19, confirmed in Photoshop 2023 on 2026-10-20 with the reinstalled font): a pill about
0.95 em tall with label capitals about 0.55 em, the label centred, color only — no extra symbol on multi-turn
tags, in the font or on the website (an earlier extra mark read as an "I" and was dropped). The exact sizes
and spacing are the `BADGE_*` constants in build_stoogedings.py (pill top/bottom, capital height, padding,
`BADGE_TRACKING` letter spacing, border, side); read them there rather than quoting numbers, and rebuild (§1.4)
after changing them. Centring pitfall fixed on 2026-10-20: tracking must go between letters only — adding it
after the last letter pushes the label off centre.

**The tag vocabulary is defined in three places that must change together:**
- `badge_tags()`/`BADGE_GROUPS` in build_stoogedings.py
- `FIXED`/`tagInfo()` in `site/assets/text.js`, plus the `--b-*` colors and `.badge` styles in `site/assets/style.css`
- the `TAG` regex in the migration prototype (§5)

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
2. Build (no arguments; it finds the repo from its own location; about 13 s):
   `MDW/Fonts/stoogedings-src/.venv/bin/python MDW/Fonts/stoogedings-src/build_stoogedings.py`
   Expected output today: `built …/stoogedings.ttf: 5.1 MB, 91 icons, 1024 token spellings`.
   Read any "token collision" or "missing file" lines.
3. Rebuilds are **byte-identical** (`recalcTimestamp = False`, :514), so `git status` shows the font as changed only
   when something really changed. Verified 2026-10-19: a rebuild of a /tmp copy matched both the committed TTF and
   the JSON byte for byte. The write is atomic (`.tmp` + `os.replace`).
4. Install: copy the TTF to `~/Library/Fonts/` (or open it in Font Book), replacing the old copy. Photoshop pitfalls
   are in §4.
5. In TTSCardMaker, commit the PNG(s), `stoogedings.ttf` and `stoogedings.json` together (the script's docstring
   says to). After that, the bot side needs no code change. The parser reads the manifest from GitHub on every
   `/update_stats` (`_icons_from_response`; the offline diagnostics scripts read the clone's copy with
   `_icons_from_local`). Run `/sync_emojis` so new icons get emojis
   (the bot also uploads missing ones at startup).

### 1.5 How it is built, and the renderer facts behind the design

GSUB is registered under **both `liga` and `rlig`**. There are five lookups, in this order:

| Lookup | Type | Job |
|---|---|---|
| TO_TK | Single | char → `.tk` copy (capitals → lowercase copy) |
| MARK | Chaining contextual | applies TO_TK after `[` |
| TOKENS | Ligature | `[`+`.tk`…+`]` → icon glyph (longest first) |
| LAYERS | Multiple | `:sticker` → [outline][color icon][spacer]; `:no` → [color icon][zero-width red sign] |
| BADGES | Ligature | `(` tag `)` → badge |

Every icon also gets a plain TrueType outline in `glyf` as a fallback for renderers without SVG support. The SVG table
holds 252 documents (91 icons + 91 no-signs + 70 badges).

Renderer facts. These were tested 2026-10-19 in Photoshop 2023 and macOS CoreText. They are also recorded in the
build script's docstring:

| Fact | Design consequence |
|---|---|
| Photoshop renders OpenType-SVG glyphs, including embedded PNGs | Color icons are SVG glyphs, each embedding its own PNG |
| Photoshop paints SVG `currentColor`, `context-fill` and CPAL `var()` **black**. Only plain outlines take the text color | `:tint` and the `:sticker` outline are plain TrueType outlines, not SVG |
| Photoshop draws overlapping glyphs in text order | `:sticker` = outline glyph then icon (both through GSUB multiple substitution). No PNG is stored twice |
| GSUB single, ligature, multiple and chaining-contextual lookups all work in Photoshop under `liga` | The whole design relies on liga. Leave ligatures on in the Character panel |
| CoreText (Apple apps) draws plain outlines **above** SVG glyphs | In Preview, Pages and similar apps, `:sticker` shows only the outline. Photoshop is the target, so this is accepted |
| CoreText ignores an SVG `<use>` that points at an `<image>` | Each color glyph embeds its PNG directly |

### 1.6 Four consumers, four matching rules

The same token is interpreted in four places. Since 2026-10-20 the font and the parser accept exactly the
same spellings; the website is still more lenient:

| Consumer | Normalization | Aliases | Notes |
|---|---|---|---|
| Font (GSUB) | Case-insensitive inside `[]`. Space, `_`, `-` or nothing interchangeable | Baked in at build | Strictest |
| AutoCroissant parser, `IconVocabulary.spellings` / `lookup` (psd_analyzer.py) | Same as the font (2026-10-20): the name's or alias's words joined by one space, `_`, `-` or nothing; exact lowercase match; no spaces around `:variant` | Read from the manifest at parse time | Anything the font would draw as letters (`[fragile.equip]`, `[ fire]`, `[fire: no]`) is an `UNKNOWN ICON TOKEN` problem. Before 2026-10-20 the parser also dropped `.`, `'` and extra spaces (more lenient than the font) |
| AutoCroissant emojis, `type_emojis.emoji_name` | Lowercase. Drops `.`, `'` and `’`. Other non-alphanumerics → `_`. Max 32 characters, cut at a word boundary | **None**: it relies on the parser having already canonicalized the token in stats.pkl | `[x:no]` → `x_no` emoji, else 🚫 + emoji. `[X]`, `[Status]` and other unknown brackets stay as text |
| Website, `norm()` in build_site.py:60 and text.js:4 | Lowercase. Drops whitespace, `_`, `-`, `.`, `'` and `’` (the parser's pre-2026-10-20 rule) | Read from the manifest at build time | `[x:no]` → icon wrapped in `.tok-no`. More lenient than parser and font; harmless for display (it only reads parser output), but its docstring "shared with the parser and the font" is out of date since 2026-10-20 |

Emoji names differ from manifest keys in exactly two cases: `fragile equip` → `fragile_equip`, and
`thunderfury_blessed_blade_of_the_starseeker` → `thunderfury_blessed_blade_of_the`. The 126 emojis are
35 type + 56 star + 35 `_no` (only Types/ icons get `_no`). This was reproduced 2026-10-19 by running the
`emoji_name` logic on the live listing. The emoji constants are owned by **autocroissant-config-and-flags**.
Application emojis belong to whichever bot identity is logged in, so each identity (dev Mac vs production) syncs its
own set. Live Discord upload is untested as of 2026-10-19.

### 1.7 Cross-repo changes: what breaks

| Change in TTSCardMaker | Font | Bot parser | Bot emojis | Website |
|---|---|---|---|---|
| **Add** a PNG | New codepoint after rebuild | Picked up automatically (Types/ listing + manifest) | Uploaded at next startup or `/sync_emojis` | Picked up on next build |
| **Rename** a PNG (e.g. `fire.png` → `flame.png`) | New key and new codepoint. The old key stays, with its glyph skipped. Existing PSDs typed `[fire]` **show letters** | Old `[fire]` tokens become `UNKNOWN ICON TOKEN` (since 2026-10-20 `IconVocabulary.build` ignores manifest entries whose PNG is gone), matching the letters the card now shows. `card.types` names change for image-layer icons | Old `fire`/`fire_no` emojis stay until `/sync_emojis prune:True`. Stored `[fire]` tokens keep rendering with the old emoji | `[fire]` in ability text loses its icon (icons are built from the tree, not the manifest) |
| **Delete** a PNG | Glyph skipped, codepoint reserved | Its tokens are flagged `UNKNOWN ICON TOKEN` | Stale emoji until pruned | Icon gone |
| Add an **alias** (edit the JSON, rebuild) | New spellings | Accepted at next `/update_stats` | n/a (works on canonical names) | Search/render accept it |
| Add a **timing tag** | Edit `badge_tags` | Nothing: text stays `(TAG)` | n/a | Edit `text.js` `tagInfo` and `style.css` |

Rule: **never rename or delete an icon PNG to "fix" a name. Add an alias instead.** If you must rename, the change
spans TTSCardMaker (PNG, rebuilt font and manifest, every PSD that uses the old token or layer name) and the bot
(`/sync_emojis prune:True`, then `/update_stats`). Class (d) gates apply: autocroissant-change-control.

## 2. The card-browser website

**Live since 2026-10-20 at https://michaeljsr.github.io/TTSCardMaker/** (Pages source: "GitHub Actions").
- The `github-pages` environment allows deployments from `main` and from `card-browser-demo`. That temporary demo
  branch was pushed for the user's review before the site was merged to `main`.
- Once `site/` and `.github/` are on `main`, remove `card-browser-demo` from the environment's deployment branches and
  delete the branch (the user wants no spare branches; memory `commit-to-main`).

### 2.1 Data flow (site/build/build_site.py)

| Input | Source | Code |
|---|---|---|
| Card text, stats, authors, timestamps, problems | `stats.pkl` from `https://raw.githubusercontent.com/bobberlington/AutoCroissant/main/` (`BOT_REPO` env) | `load_bot_pickle` |
| Text history | `old_stats.pkl` (optional) | `main` |
| Aliases | `aliases.pkl` (alias → PNG file name) | `main` |
| **Authors** | `MDW/Card Authors.txt`. `[people]` git name = shown name; `[cards]` Card Name = person. Card names match case-insensitively, `_` = space, whitespace collapsed; `[people]` lookups are case-insensitive. Listed card: `people.get(cards[name], cards[name])`; else `people.get(bot_author, bot_author)`. Same rule as AutoCroissant. A missing file or malformed line only warns | `parse_authors`, `card_key`, `person` |
| File list + blob hashes | `git ls-tree -r -z HEAD` (works in a sparse, blobless checkout) | `repo_tree` |
| **Card images** | Each card PNG becomes 3 WebP files named `<blob16>-240/-360/-720.webp`. 240 and 360 are the grid at 1x/2x, 720 is the card view. Cached in `site/.cache/img` by git blob hash, unused files pruned; settings in `IMAGE_SIZES`. PNG bytes come from the local file, else `raw.githubusercontent.com/MichaelJSr/TTSCardMaker/main/` | `make_images` |
| Icons | `Types/`, `Types/Stars/` → 96 px WebP (q90, lossless alpha; shown at ≤ 34 px). Names and aliases from the stoogedings manifest | `build_icons` |
| **Image versions** | One `git log --raw --no-renames` pass over all history, commit metadata only, no blobs (0.05 s for 520 commits). For each card PNG it lists the commits that changed its content. Exact renames (same blob deleted and added in one commit, i.e. folder moves such as the `Auxillary/` typo fix) are followed; a rename that also edits the image starts a new history. Inexact rename detection (`-M`) is deliberately not used: in a blobless clone it would download blobs. A shallow clone gets none and warns | `image_versions` |
| Ban list | `MDW/Constructed Ban List.txt`, matched on letters and digits (`alnum`) incl. aliases | `parse_banlist` |
| **Duplicates** | A numbered PNG with no PSD of its own next to a card's PSD (`A_Snack.psd` → `A_Snack2.png`, `A_Snack3.png`). The team's rule: duplicates always share the original's PSD. Each becomes its own card with the original's data, `dup` = original id (the original gets `dups`). Its name is the alias pointing at that PNG (longest, capitalised: "breakfast" → Breakfast), else the original's name. Its author is the original's. Duplicates don't count as mentioning cards and are left out of the Stats page. The other names are hidden title layers in the PSD (Breakfast, Supper; Grape Juice; Inconspicuous Watermelon), but which PNG gets which name is only recorded by `/alias` | `main`, `dupsOf` (app.js) |
| Unused status effects | Debuffs/buffs that no card's text names get `_unused` (Unused flag, filter, and a list on the Stats page) | app.js `init` |
| Unparsed cards | PSDs/PNGs under Creatures/, Items/, Field/, N.M.E/ and Auxiliary/ that aren't in stats.pkl: image only, classified by folder | `main`, `classify` |

The pickles are loaded **without AutoCroissant code** (`_Unpickler` stubs every `commands.*` class). Renaming
`CardInfo` is harmless; renaming attributes (`card_type`, `ability`, `stats.hp`, `timestamp`) silently blanks those
fields. Unpickling still trusts non-`commands` globals: anyone who can push to AutoCroissant main can run code in the
TTSCardMaker Actions runner (same trusted circle today).

**Output (`site/dist/`):**
- **Data files:**
  - `data/cards.json` (everything the first screen needs; ~84 KB gzipped). History counts `hv`/`iv` instead of the
    data; `parsed` only when false.
  - `data/rules.json`, `data/history.json`, `data/versions.json` (loaded only when the rulebook, a card or the
    changes feed needs them). `versions.json` holds a commits table plus per-card `[[commit index, path then if
    different], …]`, newest first; ~27 KB gzipped, 3,211 versions of 725 cards from 356 commits since 2023-11-05.
- **Pages:** `index.html` (`<base href="./">`), `404.html` (`<base href="<SITE_URL or ''>/">`), and one share page
  per card at `c/<slug>/` with OG tags. Kind labels use the game's words; tokens are flattened, `:tint` / `:sticker`
  included.
- **Cache-busting:** every script/style/data URL carries `?v=<build>`. `index.html`'s preload hints are rewritten too,
  and modules' `from "./x.js"` imports get the same stamp, so a deploy never mixes old scripts with new data.
- **Size:** about 117 MB in total, mostly 2,463 card images.

**Build behaviour:**
- **Fails** (exit 1, nothing deployed) when `stats.pkl` can't be fetched, when it has fewer than `MIN_CARDS` (100)
  entries, or when more than max(5, 2%) of card images fail.
- **Degrades** (logs and continues) for missing history, aliases, manifest, authors file or a few images.
- **Timing:** the CI cold run (2026-10-20) made 2,463 images in about 100 s. Warm runs take 2–3 s.

### 2.2 Workflow (.github/workflows/site.yml)

- **Triggers:** push to `main`, cron `17 */6 * * *`, and `workflow_dispatch`. Only on the demo branch, the push trigger
  also lists `card-browser-demo`.
- **Checkout:** `actions/checkout@v7`, `fetch-depth: 0` + `filter: blob:none` (every commit and tree, about 25 MB, no
  contents), plus a non-cone sparse checkout of `/site/`, `/Types/`, the manifest, the ban list and
  `/MDW/Card Authors.txt`.
- **Steps:**
  - `setup-python@v7` (3.12, pip cache) → `pip install -r site/build/requirements.txt` (pillow) → `configure-pages@v6`
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
pip install -r site/build/requirements.txt         # pillow
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
| Search language (same as `/query_ability` plus names/aliases; flags `is:unparsed` `has:history` `has:problems` `has:alias`) | `site/assets/search.js` |
| Token → icon, `(TAG)` → badge, card-name links, word diffs | `site/assets/text.js` |
| Deck builder: the constructed hand from rulebook "03 Setting Up" (`HAND`; 17 cards; field ≤ 3★), saved decks, share code, import/export, random fill, print, TTS export. **The only place that shows "banned"** (picker markers, checks, random fill), besides the rulebook's ban list page and the Stats chart about constructed bans | `site/assets/deck.js` (+ `tileHtml(…, { constructed: true })` in app.js) |
| Stats: counts, **"Starting-hand choices"** (legal, unbanned cards per hand tier vs how many the hand needs; replaced the stacked star-level bars on 2026-10-20), averages heat table (OKLab ramp, text always ≥ 4.5:1: `heatScale`), distributions, outlier scatter, table views | `site/assets/stats.js` |
| Theme, badge colors, version viewer layout | `site/assets/style.css` |
| Data shape, images, versions, authors, OG pages, ban list, unparsed cards, cache-busting, `Card Edits.json` (`apply_edits`) | `site/build/build_site.py` |
| **Edit button** (sign-in with a GitHub token, edit form, batch tray, one-commit publish via the Git Data API, Card Authors / ban list line edits) | `site/assets/edit.js` |

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

**Every pickle push from the bot updates the public website.** The site shows exactly what is on AutoCroissant `main`.
- **Timing:** a pushed `stats.pkl` reaches the site at the next 6-hourly run, or sooner on any TTSCardMaker push or
  manual run (raw.githubusercontent adds up to 5 min of CDN cache).
- **Consequence:** a bad PICKLE push becomes public, and the site is only as fresh as the last `/update_stats` + `/push`.
- **Card edits:** pushed card content changes images immediately, but text and stats only after the bot re-parses
  and pushes.
- **Authors:** both sides read `MDW/Card Authors.txt` with the same rule, so authors agree whatever the bot recorded.
- **Website edits:** `MDW/Card Edits.json` (card file name → fields). The site applies it on every build (a publish
  triggers one); the bot on `/update_stats`, keeping PSD values in `parsed_values`. Field list is in three places:
  `EDIT_FIELDS` (build_site.py), `FIELDS` (edit.js), `CardEdits.TYPES` (psd_analyzer.py).

The diff-gate before pushes is owned by **autocroissant-change-control**; /update_stats and /push by
**autocroissant-run-and-operate**.

## 3. Missing-fonts audit (2026-10-19, re-checked 2026-10-20)

**Status 2026-10-20 (later):** 7 PostScript names used by cards are still missing on the dev Mac. The CoreText check
(`swift /private/tmp/fontexplore/scripts/installed_fonts.swift`) and a scan of the PostScript name inside every
font file on the Mac (1,780 files, `find_fonts_by_psname.py`) agree.

**Added to `MDW/Fonts/` and installed that day (uncommitted):**
- `NotoColorEmoji-SVG.otf`: adobe-fonts/noto-emoji-svg 2.100.
- `consola.ttf`, `consolab.ttf`, `consolai.ttf`, `consolaz.ttf` (Consolas 5.22, four styles) and `cambria.ttc` / `cambriab|i|z.ttf` (Cambria 5.96):
  - Extracted with macOS `bsdtar` from Microsoft's free PowerPoint Viewer (`PowerPointViewer.exe` → `ppviewer.cab`).
  - Fetched from the Internet Archive copy of the original download.microsoft.com URL; SHA-256 `249473568eba7a1e4f95498acba594e0f42e6581add4dead70c1dfb908a09423`.
  - The cabinet also holds Calibri (already in MDW), Candara, Constantia, Corbel and Meiryo, which no card uses.
- `MyriadPro-Regular.otf` (2.106): the copy bundled with Photoshop 2023. The Civilization VI copies are byte-identical.

**Still missing** (visible runs):

| Font | Visible runs | Where it could come from |
|---|---|---|
| Javanese Text | 15 | Windows |
| Pristina | 10 | Office |
| Papyrus-Regular | 5 | Windows Office version; the Mac's own Papyrus is PostScript `Papyrus` |
| OCR A Extended | 5 | Office |
| BabaUI | 4 | Unknown |
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
| Free download, **installed 2026-10-20** | `NotoColorEmoji-SVG` (1 card + 2 rulebook pages): adobe-fonts/noto-emoji-svg release 2.100, Apache-2.0, in `~/Library/Fonts` and copied to `MDW/Fonts/` (uncommitted) |
| Microsoft Office for Windows only (not on Adobe Fonts) | `Pristina-Regular` (2), `Elephant-Regular` (1, a single apostrophe), `OCRAExtended` (1). Copy PRISTINA.TTF, ELEPHNT.TTF and OCRAEXT.TTF from `C:\Windows\Fonts` on a PC with Office. `Century` was found and installed on 2026-10-19 |
| Font Book can download and activate | `BookAntiqua` (1), `CenturyGothic-Bold` (1) |
| macOS ships it (`/System/Library/Fonts/Supplemental`) | `ArialMT` (4), `ArialNarrow` (1), `ArialNarrow-Bold` (1), `ComicSansMS` (4), `ComicSansMS-Bold` (2), `Impact` (3), `TimesNewRomanPSMT` (7), `MicrosoftSansSerif` (1) |
| Near-miss | `Papyrus-Regular` (1: Papyrus): macOS's Papyrus.ttc is named `Papyrus`, so Photoshop still reports it missing. `Kung-Fu-Panda` (4): very likely the repo's `KungFuPandaRegular-xYjV.ttf`, whose PostScript name is `Kung Fu Panda`, so it does not satisfy the lookup |
| Unknown source | `BabaUI` (1: Baba). Nothing by that name online (2026-10-20). It is presumably extracted from the game *Baba Is You*, which isn't in the dev Mac's Steam library. The closest public font is the itch.io "Unofficial BIY font", which has a different name, so Photoshop wouldn't match it |

Recommendations:
- (a) Before editing rulebook PSDs, activate Consolas through Adobe Fonts, or Photoshop will substitute and reflow.
- (b) The owner's call (2026-10-20): "for any fonts you find, add them to the fonts folder, there isnt any issue with
  having them in that folder." So found fonts, commercial ones included, go into `MDW/Fonts/` (and get installed).
  Note: the card-browser workflow checks out only `site/`, `Types/`, the font manifest and the ban list, and it publishes
  only `site/dist`, so `MDW/Fonts/` never goes onto the website. It does go into the GitHub repo if committed.
- (c) The audit only reads text layers, so the parser is unaffected by missing fonts. psd-tools reads stored text,
  not rendered glyphs.

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
| OT-SVG glyph support in Photoshop 2023 (probe 2026-10-20) | **Rendered:** `feGaussianBlur`, group `opacity`, linear/radial gradients, `clipPath`, embedded PNG. **Ignored:** `mask`. **Whole glyph dropped:** `feDropShadow`, and a filter chain with `feMorphology`/`feMerge`. Glyph drawing is **clipped to the line box** (font ascent/descent and the run's advance), so glows baked into glyphs get cut off |
| Text-style limits found while hiding runs | `baselineShift` needs `impliedBaselineShift` set too, and is capped at about 6× the font size. `verticalScale` minimum is 1% (0 is ignored), and 1% still leaves hairlines on the baseline. The engine's `FillFlag`/`StrokeFlag` are **not** in the ActionManager `textStyle`: setting `fill`/`fillFlag` does nothing |
| Selections | Subtracting a soft selection multiplies (`a·(1−b)`), so identical anti-aliased pixels leave about 25% residue. Harden it: store the selection in a channel, `Thrs` level 128 on that channel, reload. `Modify > Expand` is a max filter. Selections stop at the canvas edge: use **Layer via Copy** (`CpTL`) to take only selected on-canvas pixels (no clipboard) |
| **Composite channels are stale inside a script** | `doc.selection.load(doc.channels[0])` (the red component) returned the card as it was before the script's layer changes, even after `app.refresh()`: 1.96M px selected for a 90k px layer. To read one layer's pixels as a mask, use **Apply Image** into an alpha channel: `AppI` with `With` = `Clcl` {`T   ` = ref(`Chnl` `Rd  `, `Lyr ` by name), `PrsT` false}, with the alpha channel as the only active channel |
| Fills and bounds | `selection.fill()` fills the **active** layer; set `doc.activeLayer` first. `layer.bounds` **includes layer effects** (a glow widens it); use `boundsNoEffects` for the text's own box |
| One undo step | `doc.suspendHistory("Effect Layers", "doJob()")` evaluates its string in global scope (pass data through a global). Check with `doc.historyStates[length-1].name`, not the count: the count stops growing at the 50-state cap |
| Clear a layer's style | `executeAction(sID("disableLayerStyle"))` is **Clear Layer Style**. It throws "The command Clear Styles is not currently available" when the layer has no style, so wrap it in try/catch |
| Copy a layer style | `copyEffects` on the source, then `pasteEffects` (with `allowPasteFXOnLayerSet`) on the target. This replaces the target's style. Apply a Styles-panel preset by name: `ASty` with a `Styl` name reference. List presets: `executeActionGet` on the application's `presetManager`, class `StyC` |

## 5. Migrating PSDs from image-layer icons to tokens

Status: **prototype only, not committed.** It exists only in volatile scratch:
- generator: `/private/tmp/fontexplore/ps/convert_demo.py` → `convert_demo.jsx`
- input: `/private/tmp/fontexplore/demo/spec.json`
- outputs: `demo/*_font.psd|png`, `compare_*.png`, `convert.log`

Copy it somewhere permanent before a reboot if it matters.

What it does for each card's `Ability` text layer:
1. Walks the text. Each run of 3+ spaces or tabs (a gap) becomes ` [token] ` in stoogedings. The tokens come from
   `spec.json`, precomputed **in the parser's gap order**, so the result matches what gap injection stored.
2. Sets each timing tag `(…)` matching the 70-tag regex in stoogedings, so it becomes a badge.
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

**Don't migrate yet:** the owner is waiting for teammates to push their card updates first (2026-10-20). Still open
before a batch run:
- spacing around a slash between two icons comes out as `[fire]/ [machine]`; it should match the original
  `[fire] / [machine]`;
- re-check that the token order matches gap injection on every card.

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
  - OT-SVG glyphs are clipped to the line box.
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
| Running `/update_stats`, `/push`, `/sync_emojis`, reminders such as the weekly emoji sync | autocroissant-run-and-operate |
| Gating a pickle push (now also a website publish) | autocroissant-change-control |
| Proving a parser change is neutral (old-vs-new parser over all PSDs, sandboxed sweep) | autocroissant-validation-and-qa |
| Discord-side symptoms (emojis not rendering, command missing) | autocroissant-debugging-playbook |

## Provenance and maintenance

All facts verified 2026-10-19 (TTSCardMaker HEAD 3a268ca, everything above uncommitted); §1.2 badges and
§1.6 matching rules updated 2026-10-20. Run the commands from
`~/Desktop/TTSCardMaker` unless they say otherwise. Refresh the date-stamps whenever you touch this file.

```bash
# Font identity, icon count, codepoint range, badges, GSUB lookup types (expect: stoogedings-Regular, 0xe000..0xe05a, ★ mapped, 70 badges, [1, 6, 4, 2, 4], 252 SVG docs)
MDW/Fonts/stoogedings-src/.venv/bin/python -I -c "from fontTools.ttLib import TTFont; f=TTFont('MDW/Fonts/stoogedings.ttf'); c=f.getBestCmap(); print(f['name'].getDebugName(6), hex(min(x for x in c if x>=0xE000)), hex(max(c)), 0x2605 in c, sum(n.startswith('badge_') for n in f.getGlyphOrder()), [l.LookupType for l in f['GSUB'].table.LookupList.Lookup], len(f['SVG '].docList))"
# Manifest: 91 icons (35 type / 56 star), contiguous PUA, 15 with aliases
python3 -c "import json,collections as C; m=json.load(open('MDW/Fonts/stoogedings-src/stoogedings.json'))['icons']; print(len(m), C.Counter(v['kind'] for v in m.values()), sum(bool(v['aliases']) for v in m.values()))"
# Byte-identical rebuild: copy Types/, the two input fonts and stoogedings-src/ to /tmp/x/ (same layout), build there, then cmp both outputs (expect no output from cmp)
# Badge vocabulary (expect 10 fixed + 6 patterns over "123456789X") and size constants
grep -n "def badge_tags" -A8 MDW/Fonts/stoogedings-src/build_stoogedings.py
grep -n "^BADGE_" MDW/Fonts/stoogedings-src/build_stoogedings.py
# The website's badges carry no extra multi-turn symbol (expect no output)
grep -n "multi::after" site/assets/style.css
# Ignore rules
git check-ignore -v MDW/Fonts/stoogedings-src/.venv site/dist site/.cache          # three hits
# Workflow triggers and sparse paths
grep -n "cron\|workflow_dispatch\|branches\|sparse-checkout:\|filter:" .github/workflows/site.yml
# Site data source and deck rules
grep -n "^BOT_REPO\|^CARD_REPO" site/build/build_site.py; grep -n "tiers\|FIELD_MAX_STARS =" site/assets/deck.js
# Bot pickle reachable (site build depends on it; expect HTTP/2 200)
curl -sI https://raw.githubusercontent.com/bobberlington/AutoCroissant/main/stats.pkl | head -1
# AutoCroissant side (run in ~/Desktop/AutoCroissant)
grep -n "^ICON_FONT_PREFIX\|^ICON_MANIFEST_PATH" commands/psd_analyzer.py      # 49, 50
grep -n "^MAX_NAME_LEN\|^NO_SUFFIX\|^ICON_FOLDERS" commands/type_emojis.py     # 19, 21, 20
# Missing-font audit inputs still present? (scratch; may be gone after reboot)
ls /private/tmp/fontexplore/build/missing_fonts.json /private/tmp/fontexplore/ps/convert_demo.jsx
```

Maintenance rules:
- If the website's `norm()` is aligned with the parser, update the §1.6 website row.
- Once the TTSCardMaker work is committed, replace "uncommitted" with the commit hash. Once Pages is enabled, update
  §2.2 and the website URL.
- If the migration prototype is moved into the repo, replace every `/private/tmp/fontexplore` path.
- When a tag or icon is added, re-run the first two provenance lines and update the counts (91 / 70 / 126 /
  U+E05A).
