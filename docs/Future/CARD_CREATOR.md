# In-browser card creator (2026-11-07)

**Status:** designed, not built, not queued (owner asked for this doc, 2026-11-07). It builds on
`docs/Campaigns/WEB_EDITOR_CAMPAIGN.md` ("the campaign") and does not wait for it. Code read at AutoCroissant 6babe2f and
TTSCardMaker 60c53f0d. The owner's template rulings of 11-07 had not landed, so the spike ran on the `Alt_` file names.

## What it is

A signed-in card maker makes a new card on the website with no Photoshop: pick a template, add the art, type the name and
ability, set stats, stars and types, watch it live, press Publish. The Actions make the PSD and PNG. From then on it is an
ordinary card, editable on the website or in Photoshop like any other.

## Why

- The website can change a card but not start one: `edit.js` `openForm(id)` needs a card, and the bake refuses a name with
  no PSD ("no PSD in the card repo", `card_bake.bake_card`).
- Making a card today takes Photoshop, every font in `MDW/Fonts`, and a reinstall and restart after each icon font rollout
  (`card_guards.OLD_FONT`).
- A card made from a template is on the new bars and reads clean from the start: nothing for the hand flag to catch.

## How it fits the campaign

The campaign is about existing cards: one data file per card, PSDs regenerated, tiers and gates. Its only words on this are
two rows of §6 (Art, Template); it has no new-card flow. The creator needs none of its data files, because the first bake
is enough: a template copy is a PSD, and the bake already edits PSDs. If cards become data-native later, the request below
becomes the data file: §2's fields, with `picture` for §2's `art` (artwork sources took that name). Nothing is thrown away.
`AI_CARDS_CAMPAIGN.md` Phase 2 is the same fill; the creator uses no AI.

Rulings it stands on: no backend, static site, the browser never writes a PSD or PNG (campaign §1, §11); tier C stays
PSD-first; field cards use the item template; today's two previews are enough (campaign §10, owner 10-24); the 11-07
templates.

## What exists, what is missing

| Step | Exists | Missing for a new card |
|---|---|---|
| Sign-in, publish | `edit.js` `signIn` (push permission), `commitFiles`: one commit, takes `{base64}` pictures (the rulebook editor's `upload` uses it), `authorsFile` | An op that starts a card |
| Form | `tokens.js` picker, `palette.js` types, `stars.js`, `live.js` pane, `drafts.js` | A page that holds them with no card behind it |
| Mock-up | `mockup.js` `draw()`: paints the edit over a card's thumb (`src`), erasing the old text it finds there | A picture to paint on; art; stats above 10 (`changes` skips them, `barRows` is 10 cells); raid boss (2400x2400 reads as `square` in `shapeOf`) |
| Rendered preview | `preview.js`, `card-preview.yml`, `card_preview.preview`, the real bake, 30-55 s a run | Inputs are `card`, `edit` (16 KB), `request`: no template, no art; `resolve` finds no PSD |
| Bake | `psd_edit.classify`, `apply_jobs`: name, ability, stats, stars, types on any PSD + PNG pair | `card_bake.select` only walks `Card Edits.json` entries; nothing makes the pair |
| Whole PNG | `png_export.export_new` (the guards' `new` case), `psd_edit.render_region` | Nothing |
| Art | `PixelLayer.fromarray`, `move_to` (`psd_edit._make_icon`) | An art step. The fork's `SmartObject` reads and exports its content, it cannot replace it (campaign §3.4, still open) |
| Folder, file name | `CardClassifier.classify`, `star_target`, `file_stem`, `plan_rename`'s name check | Nothing |
| After landing | Card sync's data lane, the Changes page's New cards, the TTS save's rules | Nothing |

## Design

**Publish writes a request, never a PSD.** One `commitFiles` commit: `site/new-cards/<Stem>.json`, its pictures beside it,
and the card's `[cards]` line in `site/Card Authors.txt` (a new card's author is the first commit's at its path, here the
bot's; the authors file wins). The request has `template`, `folder`, `picture` (file and placement), `star_style`, and the
bake's own fields as `Card Edits.json` holds them. Database-only fields (artwork sources, subseries, aliases, note) ride
along and land as the card's entry. Not in `Card Edits.json` itself: a request is no edit, and its name matches no card
until it lands. Not in `site/cards/`: that folder is for data files that stay, and a request goes when the card lands, so
the PSD stays the one truth. Who may publish: anyone with push, as with edits.

**The make step** (`commands/card_new.py`, in Card sync's main lane before the bake):

1. Refuse early, as data: a name another card's file holds, a template or folder that does not exist, a missing picture.
2. Copy the template. Put the picture in its art slot, in place of the sample art, keeping the clip. The slot comes from a
   small table: the creature's is the pixel layer `Layer 1`, the item's a smart object `Layer 1`, the active item's a smart
   object `Layer 10`. Convert the picture to the template's profile first (seven of the nine card templates carry
   "Display"). A smart object slot means removing the
   layer and dropping its `lnk2` entry, or Photoshop refuses the file (`card_standards._orphan_links` is the gate; nothing
   prunes yet). Show the chosen star style and hide the others, so the stars step copies that one.
3. Render the whole card, store it as the preview, use it as the PNG baseline.
4. `classify` and `apply_jobs`, unchanged. Stars: parse at the count the template prints and bake to the one asked for (a
   stars edit is redundant if the folder already says it, and the template's picture would stay).
5. Gate: the re-parse equals the request, the validator says nothing, no orphaned smart object data.
6. Land the PSD and PNG in Card sync's commit M and delete the request. A `GITHUB_TOKEN` commit starts no guards run, so
   step 5 is the check. A refusal stays on the request and shows on the maker's list, since no card row exists yet.

**Live view.** A script run by hand writes, per template, into `site/templates/`: the render with the art and its clip
base hidden (a frame with a hole), its `psd_details` record, its sample values. The page flattens picture and frame into
one image and calls `draw()` with those as `src`, `rec` and `psd`, so a template is just a card with text to replace. Only
dragging and zooming the art is new drawing. Rendered preview comes later: new `template` and `picture` inputs on
`card-preview.yml`, the picture as a blob from `POST git/blobs`, as `commitFiles` makes one. It stays public, as previews
are. The owner ruled the mock-up and the dry-run bake enough, so no third preview.

**After landing** the request is gone and the card is a PSD like any other: same editor, previews, bake, flags. Changing
the art of an existing card has no path yet; step 2 as an edit would be it (campaign §6, Art).

## Beside Photoshop

- Nothing in the push path changes: guards, card and font checks, Card sync. Cards made in Photoshop go on as they do.
- A creator card opens in Photoshop (below); finish it there and push, and it is simply a pushed PSD.
- The creator offers only what a template offers. Custom bars, overlays and hand-laid text stay Photoshop work (tier C is
  PSD-first forever, owner 10-24).
- With no requests the make step does nothing, and the page is one route: leaving it out costs nothing.

## Stages

0. The 11-07 template rulings land (`Creature_Template`, `Raid_Boss_Template`, the 2 and 3 creature templates on new bars).
1. The make step as a plain script: `python -m commands.card_new <request> --out <dir>`, with the art slot table and star
   style. Done when every template gives a PSD and PNG that pass step 5 and open in Photoshop.
2. Card sync: `site/new-cards/**` in `card-sync.yml`'s paths and its pending gate, landing, refusals. Done when a request
   committed by hand becomes a card on the site. Check who the Discord post names: `--own` marks M's pushes as Card sync's.
3. The page: `#/new`, template pictures, mock-up, Publish, single-creature templates only. Done when someone other than the
   owner makes a card without Photoshop.
4. Rendered preview before publish.
5. Several creatures (once they bake, 11-07 ruling); changing a card's art.

## Proven so far

A throwaway spike at the commits above, on copies, nothing landed; the art was `Types/fire.png`.

- All 8 templates tried parse with no validator problem and render whole through `export_new` at 0.0% bad pixels against
  their stored Photoshop image (worst pixel off by 5), in 4-15 s. Tried: both creature templates, Alt raid boss, 2 and 3
  creature, item, active item, status effect.
- `classify` and `apply_jobs` unchanged, on a template copy with the render as its PNG. Creature: name, ability, four stats
  (ATK 12), types, stars 4 to 2, on new art: baked in 26 s. Item: name, ability, types, on new art: 15 s. Status effect, no
  art: name and ability, 5 s. Each re-parse matched and the validator was empty.
- Photoshop 2023 opened the creature and item PSDs with no error; the item had one orphaned `lnk2` entry, pruned. Photoshop's
  render against our PNG: item 0.000% bad, creature 0.221% once read through the PSD's Display profile; the bar is 0.5%.
- `psd_details.gather` reads the creature template (stats `template`, header found), and with the art and clip base hidden
  it renders as a frame with a clean hole.
- PSDs came out at 31 MB (creature) and 58 MB (item): the templates' hidden sample layers stay.

Not proven: the Actions path, art converted to the profile, a chosen star style, the item slot's mask and effects (the spike
dropped them).

## Open

1. **Ability background.** The creature template's is a sample picture, and its art layer is the whole background (a
   transparent picture shows the white clip base). Second upload, a plain panel, or both?
2. **Star style.** The template shows four different stars of about 20. Pick per card, or follow the series?
3. **Look.** Name and ability font and colours come from the template, though series have their own. Enough, or a "same
   look as this card" choice? It decides how much of a card's look the creator must copy.
4. **Blob read-back.** Whether a run can read a blob no commit points to, for the Rendered preview: to confirm live.
