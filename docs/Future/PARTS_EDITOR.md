# Website parts editor: one text box per PSD layer

**Status:** built and landed 2026-10-25. Track C (site) TTSCardMaker f6927047; Tracks A and B (data, bake) AutoCroissant aae7233 with TTSCardMaker abf70775; the full re-parse ran (data d547daef, parser 3, details v2, 812 records with parts). The 5 variant PSDs were tagged (TC 572680d5; Card sync links them as variants), then (work/partsfollow) the numbered-PNG duplicate scan and the §8 joined splice were deleted and `+ability` built (`psd_edit.insert_ability`/`add_ability`, the fork's `copy_to`); it stays off (`psd_edit.ADD_ABILITY_LIVE = False`, the entry waits). The owner opened the first test PSDs (they open fine) and ruled a new placement and the card's own font (10-25 ~23:50, §4.7); that work (`work/plusplace`) writes new test files to `plus_ability_2/`, and `+ability` stays off until he approves them.

**Goal.** Website edits of titles, abilities and rulebook text land exactly in the layer they came from, on any card
layout, and never overwrite Photoshop work.

**Rulings that bind it** (memory `owner-decisions-2026-10-24-night`):
- One box per layer the parser reads as a title or ability part; an extra box can be deleted (deletes its layer). Enter
  in a title box is a line break; the card's name stays one line. Rulebook pages: one box per existing text block.
- Identical title copies and name watermarks follow the main title (no box, a note).
- A layer changed in Photoshop since the edit is skipped and flagged ("redo it on the website if still wanted"); the
  other boxes still bake.
- Cards need neither types nor an ability. A first ability or type comes from the card kind's template (N.M.E. and Field:
  `Item_Template`), flagged; no empty layers in PSDs.
- A stars edit on a folder-starred card moves the PSD/PNG into the matching `N Stars` folder, creating it (flagged).
- Variant pictures: a layer or group named `... #N` (`#2,3`) shows only in variant N; variant N = `NameN.png`; on the
  site variants stay separate, linked cards. The 5 PSDs are tagged by script, one commit, plan shown to the owner first.
- Content never refuses (flag it); only waits, impossible cases and integrity gates refuse.

This design was judged and merged from Design A (data-first) and Design B (editor-first). 
The judge ran its own checks (one-off scripts, not kept):

- **Duplicate layer ids: none.** All 904 PSDs in the TC tree were checked.
- **`keep_ends` drops a first-line indent.** It keeps a layer's leading blanks only up to the last break before the text (commands/psd_edit.py:1799). So Miracle Matter's `'\r    \r    Earth…'` would lose its 4-space indent if the box text were simply put back between its ends.
- **The parser strips line-leading layout padding** (commands/psd_parser.py:976, `^[^\S\n]{3,}`).
- **`tts_save._pin_links` follows a renamed file by blob only** (commands/tts_save.py:~500, `repo.by_blob`).
- **What exists in `MDW/Templates`:** Creature, 2_Creature, 3_Creature, Item, ActiveItem, Status_Effect and Raid_Boss templates, plus Tile Bars. There is no Field template and no N.M.E. template.
- **NME_Stock layers.** `stocks are up` (1881) and `stock are down` (1890) are both full-canvas smart objects, with "down" above "up".

## 0. Verdict: which design wins at each point

| # | Point | Choice | Why |
|---|---|---|---|
| 1 | Where the parts are computed | **Merged.** The selection rules stay in `psd_parser` (they already are: `title_layers`, `_relaid`, `_pieces`, `_sort_by_position`), and the parser builds `display_name`/`ability` from them. A new uncompiled `psd_parts.py` assembles the record. | A's "cannot drift" (the parser's fields come from the same selection) together with B's small separate module that Track A owns. Putting `card_parts` inside the Cython parser would make Track B's import heavier, and a `psd_parts` that the parser imports would be a cycle. |
| 2 | The text a box shows | **A.** It is the parser's reading of that one layer (`layer_text`), and the bake puts edits back with `project(was, text, raw)`. | B's version (raw text minus blank ends, put back with `keep_ends`) loses first-line indents. Verified: `keep_ends` keeps only up to the last leading break. A's version also keeps the invariant "the parser's ability is the join of the parts". |
| 3 | The Photoshop-changed check | **A.** `sha` is `sha1(raw layer text)[:8]`, stored per part in the data; `was` is kept for people. | No blob fetch at bake time, and it is independent of normalization or parser version. B's base-blob refetch is heavier for the same safety. |
| 4 | Data older than the PSD | **B.** Warn on the boxes; don't make them read-only. | With a per-layer `sha` an edit made on stale data is safe by construction: at worst it is skipped. Read-only boxes would block editing for up to 30 minutes after every push. |
| 5 | Part key | **A.** The lyid as a string. Duplicate ids are an integrity refusal. | 0 duplicate ids in 904 PSDs (measured), and the fork's `_issue_id` gives copies fresh ids. B's `lyid.k` fallback is code with no case to serve. |
| 6 | Key for an added ability | **B.** `+ability`, plus B's validation (at most 64 keys, text at most 4 KB). | It can't collide with a digit key, and the regex is simple. |
| 7 | Derived `name`/`ability` in the entry | **A.** Keep writing them. | `CardEdits.apply`, `build_site.apply_edits` and `card_check` keep working unchanged, and the database shows the pending text. B dropped them, which would hide pending edits from every reader. |
| 8 | Skipped edits | **B.** `_psd.skipped` keeps the website's text, and the site offers Redo. | The owner's flag says "redo it on the website". Keeping the text makes that one click, and removing the part from the entry stops the gate from reopening. |
| 9 | Followers (copies, watermarks, the relaid hidden layer) | **A.** A `follow` list with `how`. | B's per-part `copies` can't express a watermark of a joined multi-part title (Wickerbottom, Mosling, Moose Goose), nor flag a watermark that doesn't match (WX-78). |
| 10 | Variant parse | **B.** Each variant gets its own full parse with `shown_in(·, N)`. Field edits are routed as in A. | The ruling says the bot "parses every variant", and a tagged stars or type layer would otherwise be wrong. Field edits from a variant page go to the PSD's entry, and a field whose layers differ by variant is impossible (A). |
| 11 | Variant tag regex | **A's strict form, with B's 1 to 2 digit bound:** `^(.*\S) #(\d{1,2}(?:,\d{1,2})*)$`. Never on names containing ` FX: `. | The owner wrote `' #2,3'` with no spaces. The bound keeps out `#10607`-style names. |
| 12 | Tag visibility rule | **Same in both; B's wording.** `shown_in(layer, N)`: a tagged node shows if N is in its tags (its eye is ignored); an untagged node uses its eye; every ancestor must show. | It reads more clearly. |
| 13 | What build_site passes | **B.** Per-card `details/<id>.json` holds the parts already filtered to that variant, `shared`, and `partState`, all computed by build_site. | It keeps the editor thin, and build_site already joins Card Edits. |
| 14 | The joined splice | **A.** Keep `project`, `_align` and `_opcodes`; delete the rest of the joined path. | B deletes `project`, but the per-part splice needs it (point 2), and it also updates the relaid hidden Ability. |
| 15 | Emptying the last part | **Neither as written:** owner, §14 item 2: emptying the last Ability box deletes the layer, flagged; a title keeps text. | A's "leave an empty layer" goes against "no empty layers in PSDs". B refused the case. |
| 16 | Templates | **B's kind map** (the files are verified); Field and N.M.E. use `Item_Template` (§14 item 3). | |
| 17 | NME Stock tags | **B.** `stocks are up #1` and `NME Salesman #1`; `stock are down #2` and group `mad #2`. | A leaves `stocks are up` untagged. That only matches because "down" covers it fully. An explicit tag survives a transparent edit. |
| 18 | The star move and TTS links | **A.** Add a TTS stem-follow. | Verified: `_pin_links` follows by blob only, and a move that also redraws the stars changes the blob. B's "re-pins" was assumed, not checked. |
| 19 | Removing a layer's text story | **B's primary plan** (remove the story and renumber `TextIndex`). A's "leave it orphaned" is the fallback, used only if Photoshop accepts it. | Both need a test in Photoshop. |
| 20 | Rollout order | **B.** The bot learns `parts` before the site writes it. | Card sync checks out AC main, so a site that writes `parts` first would have its edits dropped silently (`valid` rejects unknown shapes). |
| 21 | Which files the tracks own | **A's split by repo.** Track C owns all of TC, build_site included. | No file has two owners. Track C builds against fixture JSON. |
| 22 | Owner questions | Kept: A Q1 (title layers not named Name), A Q2 / B's template gap, A Q3 (how to tag), B Q1 (the last part). Dropped: A Q4 (it folds into Q1), A Q5 and B Q3 (the ruling "existing blocks only, one box per text block" already answers them), and B Q2 (no such case today; the default is in §4.7). | Keep only real decisions. |

## 1. Facts this rests on (measured)

- **Layer ids.** Every layer has a Photoshop lyid (`layer.layer_id`) that stays the same across saves. There are 0 duplicates across 904 PSDs. A layer remade in Photoshop (pasted, re-created) gets a new id.
- **Reading order** is `PSDParser._sort_by_position`: bbox top-left, 40 px rows, then x. For example, Miracle Matter's abilities read 1367, 1369, 1368, 417, 1366.
- **Several visible Ability layers:** 5 cards. Miracle Matter 5, Twin Emperors 5, Warcraftian Druid 2, Lit Bob-omb 2, Tainted Lazarus 2.
- **Several Name layers:** 17 cards, about 11 of them real multi-part titles, the largest The Big Yellow Cat with 4. The rest are copies or watermarks.
- **Watermarks:** `images/Group 1/Name` on Wickerbottom, Mosling and Moose Goose; `images/Name` on Doctor Zomboss. WX-78's watermark does not equal its title.
- **Rule pages:** 48 pages with 1 to 31 visible text layers each, typically 8 to 12.
- **Tags:** no layer name ends in ` #N` today. FX names do use `#n` (`Frozen#2`, `Frozen In Time #1`), so FX layers are excluded from tags.
- **Card Edits.json:** 633 entries, none with a pending `name`/`ability`/stat field, so the file needs no migration. Old open site tabs and localStorage drafts still have to be handled.
- **Parser padding:** it strips line-leading padding (commands/psd_parser.py:976).
- **`keep_ends`** cannot restore a first-line indent (commands/psd_edit.py:1799).
- **Folder moves:** `git_commit.commit_files` can't delete paths, and `tts_save._pin_links` follows renames by blob only.
- **Counts in `stats.json`:** 21 cards with no ability, 75 creatures with no type icon. Cards starred by folder: 297 creatures, 182 items, 49 fields.
- **The five multi-picture PSDs** are saved showing variant 1. Every numbered PNG is reproduced by some visibility set (§9).

## 2. Data contract (Track A produces it; frozen as fixtures on day 1)

### 2.1 Functions

In `commands/psd_parser.py` (Cython; small pure functions):

```python
TAG = r"^(.*\S) #(\d{1,2}(?:,\d{1,2})*)$"
def layer_tag(name) -> (base, frozenset | None)    # None: untagged; never applies to names containing " FX: "
def shown_in(layer, n: int | None) -> bool         # n=None: today's is_visible()
def variants_of(psd) -> list[int]                   # sorted({1} | all tags)
PSDParser.layer_text(layer) -> str                  # factored out of _process_text_layer: one layer's reading
PSDParser.parse(psd, rel, existing=None, variant=None)   # every is_visible() -> shown_in(., variant)
PSDParser.ability_parts(psd, texts, variant) / title_parts(...) / blocks(...)   # the selection; parse() joins them
```

In `commands/psd_parts.py` (new, uncompiled, no Discord or config):

```python
PARTS_VERSION = 1
def raw_sha(layer) -> str                                   # sha1(layer.text.encode("utf-8"))[:8]
def card_parts(psd, rel, name, parser, variant=None) -> dict    # the "parts" record below
def lyid_index(psd) -> dict                                 # id -> layer; raises Refused("integrity") on a duplicate
```

### 2.2 Selection rules

- **Every name rule** (Name, Ability, Header, Page Number, type icons, Stars, stat bars, `named_after_text`, R11) compares the tag-stripped base name.
- **Variant membership.** A layer is in variant N when `shown_in(layer, N)`. Its `variants` is `null` when it is in every variant.
- **Title parts.** These are the layers today's `title_layers` reads, plus the pieces `split_title` finds (Warcraftian Druid's `Druid `), minus identical copies, in reading order. `named: false` marks a layer whose base name isn't `Name` (§14 item 1).
- **Ability parts.** Visible text layers whose base name is `Ability` (no FX layers), in reading order. When none is visible and a hidden Ability layer is relaid (`_relaid`: Butterfly, Bulborb Larva, Empress Bulblax, Apex Predator), the visible pieces (`_pieces`) are the parts and the hidden layer is a `relaid` follower.
- **Rule-page blocks.** Every visible text layer, in reading order. `role` is `header` (base name `Header`), `page` (`Page Number`, or as a fallback the bottom-most layer that is only digits) or `block`. Rule pages have no title or ability parts.
- **Followers (no box).**
  - `copy`: a visible text layer whose words equal a part's (whitespace collapsed, casefold; `_copies`) and which shows on the card.
  - `watermark`: a visible layer with base name `Name` that is not a part and whose `name_key` equals the joined title's.
  - `relaid`: as above.
- **Part text** is `layer_text(layer)`:
  - icon-font runs become canonical tokens;
  - `\r` and `\x03` become `\n`; the BOM goes; tabs become spaces;
  - blank ends (`PAD`) and line-leading padding of 3 or more blanks are stripped;
  - injected image-icon `[type]` tokens are left out.
- **Joins.**
  - `display_name` is the title parts joined with spaces, each `\n` read as a space (`title_text`).
  - `ability` is `_process_abilities` over the ability parts. That is the parser's existing join.
- **Invariant**, checked by a one-off census and asserted in a test: on an untagged PSD, every parse output is unchanged except the expected deltas:
  - split titles gain a `display_name` (Warcraftian Druid, which is null today);
  - the "NO ABILITY LAYER" and "ABILITY TEXT NOT FOUND" problems are dropped for cards.

### 2.3 `db/psd_details.json`, record VERSION 2 (keyed by PSD blob, as today)

New keys:

```json
"parts": {
  "title":   [Part...], "ability": [Part...], "blocks": [Part...],
  "follow":  [{"id": 1031, "layer": "Text/Name copy", "of": [422], "how": "copy|watermark|relaid", "variants": null}],
  "variants": null | {"count": 3, "pngs": {"1": "A_Snack.png", "2": "A_Snack2.png", "3": "A_Snack3.png"},
                      "tagged": [{"id": 1015, "layer": "images/A Snack #1", "in": [1]}, ...]},
  "template": {"kind": "item", "file": "MDW/Templates/Item_Template.psd",
               "ability": {"box": [x0,y0,x1,y1], "font": "...", "size": 32} | null,
               "types": {"spot": [x0,y0,x1,y1]} | null} | null
}
Part = {"id": 1368, "layer": "Text/Stat Txt/ability/Ability", "named": true, "role": "block|header|page"?,
        "text": "Earth: 5/8/6/5 and heal 2 HP", "sha": "bdc91d46", "box": [124,1969,1035,2024], "variants": null|[2]}
```

- `text[]` entries gain `id` and `box` (mockup.js already reads `text[].box`).
- `template` is filled only when the card has no ability part or no type icon.
- A VERSION 1 record is regathered by Card sync, as it already does for a missing record.

Examples. Texts are verbatim, and these become the fixtures:

```json
// Mini Doomer (blob 197c3e76): single layers, looks as today
{"title":[{"id":422,"layer":"Text/Name","named":true,"text":"Mini Doomer","sha":"97e2eb03","box":[313,36,886,101],"variants":null}],
 "ability":[{"id":2068,"layer":"Text/Ability","named":true,"text":"Has stats equal to the base stats -2 of the creature that summoned it. \n(BOT) Die.","sha":"e1f4c753","box":[40,1815,1144,2251],"variants":null}],
 "blocks":[],"follow":[],"variants":null,"template":null}

// Miracle Matter (a46deb90): five Ability layers; raw of 1368 is '\r    \r    Earth: 5/8/6/5 and heal 2 HP\r   '
{"title":[{"id":422,"layer":"Text/Name","named":true,"text":"Miracle\nMatter","sha":"9b6f3069","box":[381,12,821,221],"variants":null}],
 "ability":[
  {"id":1367,"layer":"Text/Stat Txt/ability/Ability","named":true,"text":"(1GCD) You may choose one of the following forms:","sha":"855a83b2","box":[18,1761,1074,1891],"variants":null},
  {"id":1369,"layer":"Text/Stat Txt/ability/Ability","named":true,"text":"Fire: 5/5/8/7","sha":"32ee7ac6","box":[121,1901,493,1956],"variants":null},
  {"id":1368,"layer":"Text/Stat Txt/ability/Ability","named":true,"text":"Earth: 5/8/6/5 and heal 2 HP","sha":"bdc91d46","box":[124,1969,1035,2024],"variants":null},
  {"id":417,"layer":"Text/Stat Txt/ability/Ability","named":true,"text":"Ice: 5/6/7/6 and heal 1 HP","sha":"e8e12acf","box":[125,2031,958,2086],"variants":null},
  {"id":1366,"layer":"Text/Stat Txt/ability/Ability","named":true,"text":"This creature gains the type of whatever form it is in until the end of the turn.","sha":"80ae66fc","box":[19,2122,1134,2303],"variants":null}],
 "blocks":[],"follow":[],"variants":null,"template":null}

// Warcraftian Druid (1643ac4d): split title; display_name becomes "Warcraftian Druid"
{"title":[
  {"id":422,"layer":"Text/Name","named":true,"text":"Warcraftian","sha":"be58e62b","box":[8,19,506,104],"variants":null},
  {"id":1034,"layer":"Text/Druid ","named":false,"text":"Druid","sha":"6d194ea1","box":[9,138,226,218],"variants":null}],
 "ability":[ {"id":1247,...}, {"id":417,...} ], "blocks":[],"follow":[],"variants":null,"template":null}

// A Snack (ba3ca7ce), after the §9 tags
{"title":[
  {"id":327,"layer":"Text/Name #1","named":true,"text":"A snack","sha":"0def9122","box":[368,58,838,134],"variants":[1]},
  {"id":1023,"layer":"Text/Name #2","named":true,"text":"Breakfast","sha":"364653cd","box":[275,54,930,136],"variants":[2]},
  {"id":1025,"layer":"Text/Name #3","named":true,"text":"Supper","sha":"166d7a8a","box":[397,56,809,136],"variants":[3]}],
 "ability":[{"id":320,"layer":"Text/Ability","named":true,"text":"Heal 3 HP.\nInflict 2T2 Regen and 2T1 Stuffed.","sha":"e64944f4","box":[176,1215,1023,1466],"variants":null}],
 "blocks":[],"follow":[],
 "variants":{"count":3,"pngs":{"1":"A_Snack.png","2":"A_Snack2.png","3":"A_Snack3.png"},"tagged":[...]},"template":null}

// 01 Getting Started (10263758): one box per visible text layer
{"title":[],"ability":[],"follow":[],"variants":null,"template":null,"blocks":[
  {"id":3,"layer":"Text/Header","role":"header","named":true,"text":"GETTING STARTED","sha":"ea74752c","box":[101,72,976,141],"variants":null},
  {"id":10,"layer":"Text/Welcome","role":"block","text":"WELCOME TO IMPOSSIBILITY SIM!\n…","sha":"d1aeb11c","box":[483,286,1032,572],"variants":null},
  {"id":15,"layer":"Text/TO WIN:","role":"block","text":"TO WIN:","sha":"2954c2b6","box":[94,724,296,760],"variants":null},
  … ,
  {"id":76,"layer":"Text/1","role":"page","named":false,"text":"1","sha":"356a192b","box":[27,1874,44,1897],"variants":null}]}
```

### 2.4 `db/stats.json` (card_db)

- **One record per variant.** Variant N's key is its PNG stem with spaces ("A Snack2"), with `path` = the PSD, `variant: N` and `variant_of: "A Snack"`. Variant 1 has `variant: 1` and `variants: ["A Snack2", "A Snack3"]`. Single-picture cards have `variant: null`.
- **Each variant is parsed in full** with `shown_in(·, N)`.
- `duplicate_of` stays only for untagged numbered PNGs (the 5 until they are tagged) and for database duplicates (`dup_of`).
- `PARSER_VERSION` becomes 3, so a full re-parse runs.

### 2.5 How build_site passes it on (Track C)

- **`cards.json`** gains only small keys: `variant`, `variantOf` (the id of variant 1) and `variants` (ids, on variant 1). For tagged PSDs these replace `dup`/`dups`. The numbered-PNG duplicate scan stays for untagged PSDs only, and is deleted once the 5 are tagged.
- **`data/details/<id>.json`** (fetched when the editor opens) gains:
  - `psdBlob`: the full blob sha the record was read from.
  - `parts`: filtered to this card's variant, each part with `shared` (the names of the other variants that show it) and `follow`.
  - `variants` and `template`.
  - `partState`: `{lyid: "pending" | "stale" | "gone"}`, computed from Card Edits.json against the parts. `pending` means the entry's `sha` equals the part's `sha`. `stale` means the `sha` differs (the bake will skip it). `gone` means the lyid is no longer a part.
  - `skipped`: from `_psd.skipped`, shown while its `psd` is current (`apply_edits`' existing `current` rule).
- **Stale data.** When `reparsePending` is set or the record's blob isn't the PSD's blob at HEAD, the boxes stay editable with the warning "the card changed since this data was built: an edit made now may be skipped". This is safe by construction (§4.3).

## 3. `Card Edits.json`: per-part edits

```json
"Miracle Matter": {"stars": 4,
  "ability": "(1GCD) You may choose one of the following forms:\nFire: 5/5/8/7\nEarth: 5/8/6/5 and heal 3 HP\nThis creature gains the type of whatever form it is in until the end of the turn.",
  "parts": {
    "1368": {"field":"ability","text":"Earth: 5/8/6/5 and heal 3 HP","was":"Earth: 5/8/6/5 and heal 2 HP","sha":"bdc91d46","psd":"a46deb90","layer":"Text/Stat Txt/ability/Ability"},
    "417":  {"field":"ability","delete":true,"was":"Ice: 5/6/7/6 and heal 1 HP","sha":"e8e12acf","psd":"a46deb90","layer":"Text/Stat Txt/ability/Ability"}},
  "_by":"MichaelJSr","_at":"2026-10-26T01:00:00Z"}

"Warcraftian Druid": {"name":"Warcraftian Druid of the Claw",
  "parts":{"1034":{"field":"name","text":"Druid of\nthe Claw","was":"Druid","sha":"6d194ea1","psd":"1643ac4d","layer":"Text/Druid "}}, "_by":"…","_at":"…"}

"A Snack":  {"parts":{"320":{"field":"ability","text":"Heal 4 HP.\nInflict 2T2 Regen and 2T1 Stuffed.","was":"Heal 3 HP.\n…","sha":"e64944f4","psd":"ba3ca7ce","layer":"Text/Ability"},
                      "1023":{"field":"name","text":"Second\nBreakfast","was":"Breakfast","sha":"364653cd","psd":"ba3ca7ce","layer":"Text/Name #2"}},
             "ability":"Heal 4 HP.\nInflict 2T2 Regen and 2T1 Stuffed.","_by":"…","_at":"…"}
"A Snack2": {"name":"Second Breakfast","ability":"Heal 4 HP.\nInflict 2T2 Regen and 2T1 Stuffed.","_by":"…","_at":"…"}

"01 Getting Started": {"parts":{"15":{"field":"ability","text":"HOW TO WIN:","was":"TO WIN:","sha":"2954c2b6","psd":"10263758","layer":"Text/TO WIN:"}},
                       "ability":"GETTING STARTED\n…HOW TO WIN:\n…","_by":"…","_at":"…"}

"Panda": {"parts":{"+ability":{"field":"ability","text":"(1CD) Eat bamboo: heal 2 HP.","was":null,"sha":null,"psd":"…"}},
          "ability":"(1CD) Eat bamboo: heal 2 HP.","_by":"…","_at":"…"}
```

**Keys and values**

- **Keys** are the lyid as a string, or `+ability` (at most one per card, and never on rule pages).
- **Values:**
  - `field` is `name` or `ability`. Rule-page blocks use `ability`, because the parser reads the page text as ability.
  - Exactly one of `text` or `delete: true`.
  - `was` is the starting text, for people and for `project`.
  - `sha` is the raw-text hash and decides the check.
  - `psd` is the blob8 the data was read from.
  - `layer` is for people only.

**What `was` and `sha` hold**

- They come from the data, never from a published-but-unbaked value.
- **Re-editing.** If a pending part's `sha` still equals the data's `sha`, the entry's `was`/`sha` are kept. Otherwise the site marks the edit stale and the box starts from the PSD's text.

**Line breaks and deletion**

- **`text`** uses `\n` for line breaks. In a title, `\n` is a line break on the card and a space in the name.
- **`delete`** is offered only on a card field with more than one part. An extra box emptied on submit becomes a delete. The last part: §14 item 2.

**Derived fields**

- `name` (title parts joined with spaces, `\n` read as a space) and `ability` (parts joined with `\n`) are still written, so the database readers work unchanged.
- When `parts` holds a field, the bake treats the plain field as derived and never bakes it on its own.
- The derived field goes when no part of that field is left in the entry.

**Variants**

- Every part edit for a PSD lives in the entry of the PSD's own key (variant 1), whichever variant page made it. So each lyid has exactly one entry.
- Variant entries carry only their derived `name`/`ability` and database-only fields.
- `stars`, `types` and stats edits made on any variant go to the PSD's entry and change every variant.
- If that field's layers are tagged differently per variant (no card today), the field is refused as impossible: "edit it in Photoshop".

**`stars`, `types` and stats**

- They keep today's shapes. A stars edit on a folder-starred card is a move (§4.6).

**Validation in `CardEdits.TYPES` / `valid`**

- `parts` is a dict of at most 64 keys, each matching `^\d+$` or `+ability`.
- `text` is at most 4 KB, `field` is `name` or `ability`, and `sha`/`psd` are 8 hex characters (`sha` may be null only for `+ability`).
- A wrong shape drops the field.
- In `ORDER`, edit.js `FIELDS` and `build_site.EDIT_FIELDS`, `parts` comes right after `ability`.
- `dump_edits` and `editsFile` merge `parts` per lyid; an `unset` names lyids.

**Verdict `_psd`** (written by the bake only):

```json
"_psd": {"state":"baked", "psd":"<blob8>", "at":"…", "reason":"…",
         "parts": {"1368":"baked", "417":"deleted", "1366":"skipped"},
         "skipped": {"1366": {"text":"<website text>", "now":"<layer text now or null>",
                              "why":"changed in Photoshop since this edit: redo it on the website if still wanted"}},
         "moved": "Creatures/Kirby/4 Stars", "variants": [1,2,3]}
```

**Backward compatibility**

| Entry | Behaviour |
|---|---|
| Plain `name`/`ability`, no `parts`, on a field with one part | Baked into that part as today, with no `sha` check (there is none). This is today's behaviour exactly. |
| The same on a multi-part field | Refused as impossible: "a joined edit on a card printed over N layers: edit each box on the website". None exists today, and no client writes one after Discord's retirement. |
| `name` on a card with no title part | Applied to the database only, flagged "not printed: the card has no title layer". Today this is a refusal. |
| Old bot code meeting `parts` | Dropped (`valid` is False), hence rollout order §10. |
| Old site drafts of `name`/`ability` on multi-part cards | Dropped with a toast. On single-part cards they become a part edit, using the data's `was`/`sha`. |

## 4. Bake algorithm (`card_bake.bake_card` and `psd_edit.apply_jobs`)

The variant entries of a PSD are grouped into the PSD's own job: one open, one PSD write, every affected PNG. The job key is (edits blob, PSD blob, PSD-kind fields with `parts`).

**4.1 Waits and gates (as today).**

- Blob gate, `recent` (30 minutes, main only), `capped`/budget, creatures gate.
- The `duplicate` refusal stays only for untagged PSDs with numbered PNGs. For those, image fields are impossible: "tag the variant layers".

**4.2 Locate.**

- Open P1, the PSD at head.
- Build `lyid_index` (a duplicate id refuses the card as an integrity case, and the entry is kept).
- Run `card_parts(…, variant=N)` for every N.

**4.3 Classify each part edit.**

| Case | Result |
|---|---|
| The lyid is missing | Skipped: "the layer is gone from the PSD (deleted or remade in Photoshop since this edit): redo it on the website if still wanted" |
| No longer a text layer, or no longer a part of `field` in any variant (renamed, hidden, re-tagged) | Skipped: "changed in Photoshop since this edit: redo it on the website if still wanted" |
| `raw_sha(L) != sha` and `layer_text(L) == text` | Redundant (done in Photoshop) |
| `raw_sha(L) != sha` otherwise | Skipped with the ruling's words |
| `+ability` when the PSD now has an ability part in that variant | Skipped: "the card has an Ability layer now (added in Photoshop): redo it on the website" |
| `layer_text(L) == text` | Redundant |
| Anything else | A job |

Skipped and redundant parts leave the entry in this run's commit. Every other part still bakes. A sha mismatch only ever skips, so Photoshop work is never overwritten.

**4.4 Text jobs**, all on one edit copy. Ability parts go first (their FX records shift indices), then titles, then blocks. `_renumber` runs after each layer.

```
raw' = keep_breaks(raw, project(was, text, raw), field)      # project keeps blank ends, indents, breaks between
set_text_layer(psd, field, raw', fonts_dir, chart, layer=L, beside=parts+followers)
```

- A new `\n` takes the layer's own break kind: `\x03` if the layer has one, else `\r`.
- A title part found by its own text is renamed as today (`Edit.renamed`). Tag suffixes are kept.

**4.5 Followers.**

- Each `copy` gets the same `raw'` (the `copy=True` path).
- Each `watermark` whose text equals the old joined title gets the new joined title. Otherwise it is left alone and flagged: "the watermark 'X' doesn't match the title: not changed".
- A `relaid` hidden Ability gets `project(old_joined, new_joined, hidden.text)`.
- Notes go on the card page, for example "also printed by 2 copies; they follow".

**4.6 Deletion** (`remove_layer(psd, L)`, new).

- Removes L's record and channels, its visible FX layers (`fx_layers`) and its copies.
- In the fork: remove L's Txt2 story and renumber later `TextIndex` values. The fallback, if Photoshop accepts it, is to leave the story orphaned.
- The region is the union of the removed boxes plus `fx_pad`.
- Note: "Ice layer deleted: the text below doesn't move up."

**4.7 Added ability** (`+ability`, scheduled last).

- **Template by kind:**

  | Kind | Template |
  |---|---|
  | Creature, minion, aux creature | `Creature_Template` (all creature cards, owner 10-25) |
  | Item, aux item, N.M.E., field card | `Item_Template` (owner, 10-25) |
  | Active item | `ActiveItem_Template` |
  | Status effect | `Status_Effect_Template` |
  | Raid boss | `Raid_Boss_Template` |

  Rulebook pages get no added ability (existing blocks only).
- **Import.** Bring the template's visible Ability type layer in with fork `import_layer`: TySh with its box, font and size, the Txt2 story and the FontSet merge. Scale it when the canvas sizes differ.
- **Placement.** Put it at the top of the card's `Text` group (the root when there is none), named `Ability`, with a fresh lyid.
- **Variant tag.** Untagged, unless the card's other ability layers are tagged; then it is tagged with the editing variant.
- **Text.** Then the normal `set_text_layer`. Flag: "ability layer created from <template>: check its place".

**4.8 Stars and types** (first, as today).

- **Folder-starred card** (`Creatures/*/N Stars`, `Items/*/N Stars`, `Field/N Stars`). A stars edit is a move:
  - Replace the `N Stars` path segment. An existing folder is matched case-insensitively; a missing one is created and flagged ("created the folder 'Items/Heal/3 Stars'").
  - The PSD and every variant PNG move together.
  - Printed stars are redrawn with `plan_stars` where the PSD draws them. An unknown layout is flagged, but the move still happens.
  - A file of the same name at the target is impossible: refused.
- **First type icon.** Take the template's type spot (cached per template). Make the icon from `Types/<name>.png`, styled like the template's icon (`_make_icon`), and put it in the card's type group or the root. Flag: "first type icon placed at the template's spot: check it". This replaces `plan_types`' "no type icon … to line a new one up with" refusal.

**4.9 Variants: render and merge.**

- The pictures to update are each N in which a changed, deleted or added layer, one of its followers or one of its FX layers shows. Untagged layers count for every N.
- For each such N, inside `variant(psd, N)` (an in-memory visibility override, restored afterwards; the saved file keeps its eye flags):
  - baseline against `PNG_N`;
  - render the region with `layer_filter=lambda l: shown_in(l, N)`;
  - delta-merge onto `PNG_N`;
  - run `gate_region`, `gate_shape` and the parity notes.
- Every content gate that renders (`covered_ink`, `ink_beyond`, `gate_fx`, `gate_untouched`, `gate_outside`) runs inside a variant that shows the edited layer.
- Other pictures' PNGs stay byte-identical.
- Parity misses are reported per picture: "ability (Breakfast) baked with a pixel difference …".

**4.10 Integrity gates.**

- `gate_bytes` and `gate_stories` accept exactly the declared changed, removed and inserted records and stories. Every remaining `TextIndex` must point at its own text.
- `gate_size` and `gate_idempotence` as today.
- `gate_reparse`, per affected variant:
  - every job lyid now reads its `text`;
  - every other part is unchanged;
  - removed and added parts match what was declared;
  - fields that weren't edited are unchanged.

  This replaces the joined `ability`/`display_name` comparison and `unpadded`.
- Content problems are notes, never refusals.

**4.11 Commit** (one per run, `land`).

- `files` may map a path to `None` (a delete): `api_commit` writes `"sha": null` and the SSH path runs `git rm`.
- `expect` holds the old blobs at the old paths and nothing at the new ones.
- The entry is rewritten:
  - baked, redundant, skipped and deleted parts are removed;
  - derived fields are dropped once no part of that field is left;
  - variant entries lose their derived fields along with the PSD entry's parts;
  - `_psd` gets `parts` and `skipped` (`rewrite_entry(keep=True)` for flagged verdicts).
- Example commit line: `- Miracle Matter: ability (3rd layer) 'heal 2 HP' -> 'heal 3 HP'; 4th ability layer deleted; stars 3 -> 4 (moved to 4 Stars)`.

**4.12 After the commit.**

- Card sync parses every variant.
- The TTS save gains a stem-follow: a lost path whose card key appears exactly once under the same top folder counts as a rename, even though the blob changed.
- History and Card Edits/Authors keys are card names (stems), so they survive a move.

**What refuses:**

- **Waits:** `recent`, `head-moved`, `edit-changed`, `capped`.
- **Impossible:**
  - a legacy joined edit on a multi-part field;
  - no template for the kind;
  - the target file already exists;
  - a stars/types/stats field whose layers differ by variant;
  - an untagged PSD with numbered PNGs;
  - emptying a lone Title.
- **Integrity:** blob, duplicate lyids, bytes, stories, read-back.
- Nothing refuses for content.

## 5. Site UI (TC)

**Edit dialog, Text group** (`edit.js`, using `details/<id>.json`):

- **Title.**
  - One auto-height textarea per title part. Enter inserts a newline, shown as a line break, and the name preview below reads it as a space: `Name on the site: "Warcraftian Druid of the Claw"`.
  - Labels: "Title" for a single part; "Title 1/2 · Text/Name" and "2/2 · Text/Druid  (not named Name)" for several.
- **Ability.**
  - One textarea per part ("Ability 3/5"), each with the token bar, the `[`/`(` picker and live badges.
  - A read-only line underneath, "On the site: …", shows the parser-style join.
- **Delete.**
  - "Delete layer" appears on each box while the field has more than one part.
  - A deleted box collapses to "will be DELETED from the PSD, with 'Ability FX: …' [Undo]".
- **No ability yet.** "+ Add an ability" opens a new box with the note "new layer 'Ability' from Item_Template (box, font, size); the bake flags it for a look [Remove]". Rulebook pages have no such button.
- **Notes.**
  - "Also printed by Text/Name copy, Text/Name copy 2 (they follow)."
  - "The watermark images/Group 1/Name follows the title."
- **Types.** With no icon: "the first goes at the template's spot (flagged)".
- **Stars on folder cards:** "moves the PSD and its PNGs to 'Items/Heal/3 Stars/' (created) and redraws the 2 printed stars".

**Variants.**

- Each variant has its own card page and dialog: "Edit Breakfast · a variant of A snack · picture 2 of 3 [A snack][Breakfast][Supper]".
- A box shows "only in this picture" or "⚠ shared: changes every variant (A snack, Breakfast, Supper)".
- Card pages show "A variant of A snack" or "Variants: Breakfast · Supper".
- Grid, search and decks keep them as separate cards.

**Rule pages.**

- One compact textarea per block, in reading order, labelled by role or layer name ("Header", "Page number", "Welcome").
- A jump list sits above; focusing a box highlights its block on the image.
- No add, no delete. Title and Note stay database-only.

**Conflicts (F10).**

- **Before the bake** (`partState: stale`): "⚠ Photoshop changed this layer after your edit: it will be skipped. Photoshop now: '…' Yours: '…' [Start from Photoshop's text] [Redo mine over it]".
- **After the bake** (from `_psd.skipped`): "⚠ Not applied (10-25 14:30): Ability 2/2 changed in Photoshop since this edit. Redo it on the website if still wanted. [Redo]".
- **Redo** stages a new part edit with `was`/`sha`/`psd` from the current data and the saved text.

**Readers and writers.**

- `readForm` builds `op.set.parts` (changed boxes only) plus the derived `name`/`ability`.
- `editsFile` merges per lyid. On a 422 retry, if the file's value for a staged lyid changed, it warns "someone else edited this box" and shows both.
- `drafts.js` keys drafts as `part:<lyid>`.
- `inline.js` keeps inline editing for single-part fields. On a multi-part card or a rule page it opens the dialog at the clicked block.

**Previews.**

- **Mock-up** (`mockup.js`):
  - each changed part is erased and redrawn inside its own `box`, in the part's font, size and colour;
  - a deleted part's box is erased;
  - a title `\n` gives two lines;
  - an added ability is drawn in `template.ability.box`;
  - variants draw on their own picture.
- **Rendered** (`preview.js` `entryEdit`):
  - sends the PSD entry's PSD-kind fields with `parts`, plus the variant entry's derived fields, with `card` = the variant key ("A Snack2");
  - `card_preview` renders that variant instead of refusing `duplicate`, so card-preview.yml needs no new input;
  - `result.json` v2 adds `variant` and `parts: {lyid: baked|redundant|skipped|deleted|added}`, and the caption lists skipped parts.

**Badges.** `apply_edits` marks name/ability `pending` when `parts` holds parts of that field. `edited`/`orig` work as today. The card page shows the `_psd.parts` flags and "moved to 4 Stars".

## 6. Parser changes (Track A)

1. **Tags.** Add `layer_tag`, `shown_in` and `variants_of`. Every name rule uses the base name; FX layers are never tagged.
2. **Per-variant parse.** `parse(…, variant=N)` uses `shown_in` everywhere it uses `is_visible()` today (`_extract_from_layers`, `title_layers`, star and type walks). With `None` it follows today's path.
3. **`layer_text(layer)`**, factored out of `_process_text_layer`.
4. **Title, ability and block selections** become explicit, with identical copies and watermarks folded out. `_printed_title` and the ability join are rebuilt on them, and split titles gain a `display_name`.
5. **`CardValidator` changes:**
   - drop NO ABILITY LAYER and ABILITY TEXT NOT FOUND for cards; no types is not a problem;
   - add `DUPLICATE LAYER IDS` and `VARIANT N HAS NO PNG`.
6. **Versions.** `PARSER_VERSION = 3`; `psd_details` VERSION 2; the new `psd_parts.py`.
7. **`card_data` / `repo_traverser`.** One `CardInfo` per variant (`variant`, `variant_of`). A numbered PNG that matches a tag becomes a variant, not a duplicate.

## 7. Guards and checks

- **Guard 1** (`card_guards`/`png_export`, Track B):
  - Exports and verifies each picture with `shown_in(·, N)`. A tagged PSD pushed without `NameN.png` gets that PNG rendered.
  - The stale-PNG check (R10) runs per variant.
  - The stored-composite cross-check runs only for the picture that matches the saved eye state.
  - A numbered PNG with no tag set is reported "variant PNG with no tagged layers: tag the PSD" (no write).
- **Guard 2** (`card_standards`, Track A, report only):
  - R3/R4 become notes: "not standardized yet: N title/ability layers".
  - R11 exact names accept a ` #N` suffix.
  - New notes:
    - a part not named Name/Ability, with a rename suggestion through R11;
    - tag syntax (`#0`, a tag beyond the PNG count plus one, a tag on an FX layer, which is ignored);
    - a tag N without `NameN.png`;
    - saved visibility that differs from variant 1;
    - duplicate lyids.
- **`card_check`.** Notes per picture ("A Snack2: …"); the Card Edits validation knows `parts`.
- **`card_bake.pending`.** A `parts` entry opens the gate; skipped parts leave the file.
- **`psd_diff`.** Tag renames show as renames.

## 8. Tonight's joined-text splice (Discord card edits retired)

**Done 2026-10-25 (work/partsfollow):** deleted with 0 legacy joined entries in Card Edits.json (none of any plain name or ability). Kept: `name_layer`'s refusals (the legacy name path and its read-back gates still use it) and `Edit.several` (now the mark of a `set_parts` Edit). A plain ability on one part with copies or a relaid layer goes through `set_parts`.

**Delete** (one commit, after rollout step 9):

- In `psd_edit`:
  - `ability_split` (:910);
  - `set_ability_layers` (:1006) and its wording helpers (`ORDINALS`, `nth`, `both`, `where`) and joined-only flags ("a line was added to …", "the break between … was removed", "blank lines between the ability layers stay", "words … removed, so hidden");
  - `unpadded` (:976) and its uses in `classify` (:1907) and `gate_reparse` (:5358);
  - `several_layers` (:981) and `Edit.several` (:265);
  - `_ability_reading` (:839);
  - `set_text_layer`'s fallback to `set_ability_layers` (:3334);
  - `name_layer`'s refusals for several, split or copied titles;
  - `keep_breaks`' name branch that pairs space runs (the boxes show breaks now).
- `AbilityLayers`/`ability_layers`: replaced by `card_parts`; the relaid handling moves there.
- In tests: `test_ability_split_shares_an_edit_out` (tests/test_psd_edit.py:4321), `test_an_ability_over_several_layers_bakes` (:4347), `test_several_layers_hidden_part_and_blank_lines`, `test_relaid_and_copied_abilities_bake`.

**Keep (the per-layer splice primitive):**

- `set_text_layer(layer=, beside=, copy=)` with `edit_parts`, `splice_text`, `changed_range`, `deletion_start`, `keep_ends`, the ability branch of `keep_breaks`, `keep_tokens`, `edit_styles`, `title_styles`, the icon and FX follow/rebuild, `rasterize(keep=)` and every gate.
- `project`, `_align` and `_opcodes`: they put a box back over its raw layer and update the relaid hidden layer.
- `_copies`, `_shows`, `fx_layers` and `PAD`.
- `_record_ids`, `_renumber`, `record_map` and `_issue_id`.

**New:** `set_parts(psd, jobs)` (about 80 lines), `remove_layer`, `insert_ability` and `variant()`. The net change is about 250 lines out of `psd_edit`.

## 9. Variant tags for the 5 multi-picture PSDs (a design note; Track A's `variant_tags --suggest` re-derives them)

Both designs agree except on NME Stock, where B's explicit tags win. Fit is the mean |diff| at 300 px (A) and the share of pixels off by more than 24 (B).

| PSD | Tags (lyid: layer → new name) | Fit |
|---|---|---|
| A_Snack | 1015 `A Snack` → `A Snack #1`; 327 `Snack` → `Name #1`; 1022 `Breakfast` → `Breakfast #2`; 1023 `Breakfast` (text) → `Name #2`; 1024 `Layer 14` → `Supper #3`; 1025 `Supper` (text) → `Name #3` | v1 0.02; v2 0.03 / 0.005%; v3 0.02 / 0.001% |
| Watermelon | 1046 `Name` → `Name #1`; 939 `Inconspicuous Name` → `Name #2`; 1014 `Layer 6` → `Layer 6 #2` | 0.00 / 0.0% |
| Regeneration_Potion | 939 `Name` → `Name #1`; 1052 `Grape Juice` → `Name #2`; 1045 `Hue/Saturation 1` → `Hue/Saturation 1 #2` | 0.08 / 0.0% |
| All_Purpose_Medical_Kit | 955 `Layer 2` → `Layer 2 #1`; 957 `Layer 2 copy` → `Layer 2 copy #2` (title shared) | 0.27 (the PSD's own baseline) / 0.0% |
| NME_Stock | 1881 `stocks are up` → `stocks are up #1`; 1913 `NME Salesman` → `NME Salesman #1`; 1890 `stock are down` → `stock are down #2`; group 1904 `mad` → `mad #2` (group `happy` and `originals/*` stay untagged and hidden) | 0.00 / 0.0% |

- All five share their Ability, so its box reads "changes every variant".
- Every picture except the Medical Kit's has its own title box.
- The renames are 0-pixel. Apply them with R11's gated rename: one commit for all 5, quiet tree, rehearsed on copies, the plan shown to the owner first.
- Parser tag awareness must be live before any rename.

## 10. Migration and rollout order

0. **Day 0: freeze the contract.** Fixtures go in `AutoCroissant/tests/fixtures/parts/` and are copied into `TTSCardMaker/site/tests/fixtures/parts/`:
   - the 5 details records from §2.3;
   - `card_edits.parts.json` (§3);
   - `result.v2.json`;
   - `stats.variants.json`.

   Track A also lands the API with a stub `card_parts` (current `title_layers` + `ability_layers`, no variants).
1. **Track A ships:** parser 3, `psd_parts`, details v2, variant records, validator and standards notes.
   - Run the census: old-vs-new parse on every PSD; the only deltas allowed are §2.2's.
   - Dispatch a full Card sync (`full=true`) right after AC main and TC bbe22593 land. Parser 3 re-parses all ~812
     cards with unchanged blobs, so G4 (`REPARSE_MAX` 25) refuses every run until `full` approves it. No earlier
     approval carries, because `pending.full` is stamped with the parser version. Until that run lands, the card sync's
     runs commit nothing (the bake's after-reads too), and the site gets no `parts` records.
   - Check the data commit: `db/manifest.json` `builder.parser_version` is 3 and `db/psd_details.json` `"version"` is 2.
2. **Track B text part ships:** `parts` in `card_edits`, per-part classify and splice, sha skip, delete, `_psd.parts`/`skipped`, legacy translation, `card_preview` v2. This must be on AC main before any client writes `parts`.
3. **Track C build_site ships:** details pass-through, `partState`, variant link keys. Deploy; no UI change.
4. **Track C editor ships.** From here the site writes only `parts` for printed text.
5. **Track B variants:** `variant()`, multi-PNG merge, per-variant guard 1. Then the 5 PSDs are tagged (§14 item 4), Card sync shows them as linked variants, and the numbered-PNG duplicate scan is deleted (done 10-25: `card_data.duplicate_pngs` keeps only `dup_of`).
6. **Track B extras:**
   - the folder move, after `git_commit` deletions and the TTS stem-follow, with people counts and history checked on the first live move;
   - the first type icon;
   - `+ability` last, after the fork's `import_layer` is proven in Photoshop.
7. ~~Retire Discord's card-edit commands~~: done 10-25 (AC 6e8f8e3, e7e7879; code on `archive/discord-edits`).
8. **Wait one Card sync cycle** with zero legacy joined multi-part entries.
9. **Delete the joined splice (§8).** One commit; cut the tests.
10. **Ongoing.** As the owner standardizes, a card's boxes collapse to one per field. No data migration is needed.

## 11. Test plan (lean: xdist, one light review pass, real PSDs from the card clone, skipped without it)

**AutoCroissant (11):**

1. `layer_tag` table: `Name #2`, `Layer 6 #2,3`, `Hornet fight - atlas0 #10607 (1)` (no tag), `Ability FX: Frozen In Time #1` (FX: no tag).
2. `card_parts` on the 5 fixture PSDs equals the fixtures: order, padding stripped, copies folded, Warcraftian Druid reads "Warcraftian Druid". The joins equal `display_name`/`ability`.
3. A tagged copy of A Snack: three per-variant parses; renders within 0.1 of the PNGs.
4. Miracle Matter bake:
   - 1368 changes only its record, and the indent stays (`project`);
   - 417 is deleted with exactly its records and story gone and `TextIndex` renumbered;
   - 1366 with a wrong `sha` is skipped, flagged and kept in `_psd.skipped` while the others bake.
5. A redundant part (done in Photoshop), and a gone lyid.
6. Warcraftian Druid title Enter: a 2-line layer; the name reads one line.
7. A Snack: the shared Ability writes 3 PNGs; a `Name #2` edit writes only `A_Snack2.png`, and the others stay byte-identical.
8. Star move: delete plus add in the commit, folder created and flagged, a collision refused. TTS stem-follow on a moved, redrawn PNG.
9. A legacy single `ability` bakes; a legacy joined edit on a multi-part card is refused.
10. `CardEdits.valid` and `dump_edits` per-lyid merge and unset.
11. `+ability` on a fixture without one (lands with step 6).

**TTSCardMaker (4, node and pytest):**

1. `editsFile` parts merge, unset and the 422 conflict.
2. `readForm` boxes: delete, Enter, the derived join.
3. `build_site`: `partState`/`skipped`, variant keys, `apply_edits` pending.
4. `mockup.js` per-part erase and draw (existing harness).

**One-off census (not in the suite):**

- old-vs-new parse on all PSDs;
- joins equal fields on all cards;
- duplicate lyids, expected 0.

**Manual on the Mac:** open one baked file with a deleted layer and one with an inserted layer in Photoshop. Expect no "update text layers" or missing-font prompt, the Layers panel correct, and the text editable. Also run one live Rendered preview of a variant.

## 12. Risks (most severe first)

1. **Fork: removing a type layer's story, and importing a template layer across documents** (story, FontSet, styles). Both need proof in Photoshop. Delete has the orphan-story fallback; `+ability` ships last.
2. **Gates under variants.** Every rendering gate needs the right `variant()` context, and an edit to a layer hidden in the saved file must still render. The A Snack rehearsal covers this.
3. **Star move plus redraw** changes the blob. Without the stem-follow, TTS links break. Check the first live move.
4. **False "changed in Photoshop".**
   - A re-edit on data that predates an earlier bake of the same layer is skipped. This is safe, and the stale-data warning shows it in advance.
   - A layer remade in Photoshop reads as "gone".

   Both are conservative.
5. **Tag collisions.** A future stray ` #N` name would make a variant. The standards note "tag without PNG" catches it. FX names are excluded.
6. **Near-copies.** An outline copy with stale text becomes its own box instead of following. The standards report surfaces it.
7. **Derived values** can differ slightly from the parser's join. They are only shown until the bake; `pending` keys on `parts`, never on text.
8. **Bake time.** A shared part re-renders N PNGs (5 PSDs, at most 3 each). One PNG failing integrity refuses the card.
9. **Two editors on one box.** The last publish wins after the 422 warning.

## 13. Build plan: three parallel tracks

**Shared contract (frozen on day 0, changed only by agreement of all three):**

- **I1, the details record `parts` (§2.3) and the Python API** (§2.1: `layer_tag`, `shown_in`, `variants_of`, `layer_text`, `card_parts`, `raw_sha`, `lyid_index`, `parse(variant=)`).
  - Fixtures: `tests/fixtures/parts/{mini_doomer,miracle_matter,warcraftian_druid,a_snack_tagged,getting_started}.json`.
  - The tagged A Snack copy is generated by a fixture script, not committed to the card repo.
- **I2, the Card Edits `parts` entry and the `_psd.parts` / `_psd.skipped` verdict** (§3). Fixture: `card_edits.parts.json`.
- **I3, `stats.json` variant fields** (`variant`, `variant_of`, `variants`). Fixture: `stats.variants.json`.
- **I4, card-preview `result.json` v2** (`variant`, `parts`). Fixture: `result.v2.json`.

| | Track A: data and parser (AC) | Track B: bake (AC and the fork) | Track C: site (TC) |
|---|---|---|---|
| Owns | `commands/psd_parser.py`, `commands/psd_parts.py` (new), `commands/psd_details.py`, `commands/card_data.py`, `commands/card_models.py`, `commands/repo_traverser.py`, `commands/card_check.py`, `commands/card_standards.py`, `tools/variant_tags.py` (new, `--suggest`), census scripts, `tests/test_psd_parts.py`, `tests/fixtures/parts/` | `commands/card_edits.py`, `commands/card_bake.py`, `commands/psd_edit.py`, `commands/git_commit.py`, `commands/card_preview.py`, `commands/card_guards.py`, `commands/png_export.py`, `commands/tts_save.py`, fork (`remove_layer` story removal, `import_layer`), `tests/test_psd_edit.py` (the part cases), `tests/test_card_bake.py` | `site/build/build_site.py`, `site/assets/edit.js`, `drafts.js`, `preview.js`, `mockup.js`, `inline.js`, `app.js`, `style.css`, `site/NOTES.md`, `site/tests/` |
| Exposes | I1, I3 | I2 (bake semantics), I4 | none |
| Consumes | none | I1 (imports `psd_parts`; a JSON-backed stub until A lands) | I1 via details, I2, I3, I4 (all as fixtures) |
| Internal order | Tags and `layer_text`, then selections and `card_parts`, then per-variant parse and records, then standards | Text, delete and the sha check, then variants and guard 1, then the folder move and first icon, then `+ability` | build_site, then the editor boxes, delete and Enter, then conflicts and Redo, then variants and rule blocks, then previews |
| Size | ~450 lines | ~800 lines plus fork | ~650 lines JS/py |

**Merge order onto main:**

1. A (API and stub).
2. A full, with a Card sync re-parse.
3. B text.
4. C build_site.
5. C editor.
6. B variants, then the 5 PSDs tagged by script (one commit, plan shown to the owner first).
7. B extras.
8. The §8 deletion.

B and C rebase on A's frozen fixtures and never edit each other's files. The only cross-repo edits are the fixture copies, owned by A.

## 14. Owner's answers (2026-10-25 ~03:50) and the one open question

1. **Title layers not named `Name`** get a box for exactly what the parser reads as the title, labelled "(not named
   Name)"; guard 2 suggests the rename. Text the parser doesn't read stays Photoshop-only.
2. **Emptying the last box:** the last Ability box deletes the layer, flagged (the card then has no ability, which is
   allowed). A title box can't be emptied.
3. **Cards without a template:** N.M.E. and Field cards take a first ability or type from `Item_Template`.
4. **Variant tags:** by script, one commit for all 5 PSDs, the plan shown first.

5. **Creature template (10-25 ~06:30):** every creature card, two- and three-creature cards included, takes a first
   ability or type from `Creature_Template` (flagged).
