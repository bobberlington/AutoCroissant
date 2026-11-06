# Session 2026-11-06: TTS deck buttons, the Alt template, stats to 20

From the 11-05 wrap-up (AutoCroissant a09e4c3, TTSCardMaker 474c642c) through the Alt template rollout. With the owner
("ask me questions") until ~05:00 PST, then unattended.

Heads at the end: AutoCroissant dc9f674 plus this wrap-up's docs (deployed), TTSCardMaker d460db0e. What is left: `docs/NEXT_SESSION_PLAN.md`.

## Landed

| What | Where | Notes |
|---|---|---|
| TTS deck buttons | TC befdf769, 93a58a1d, 37dcf02a, 287a903f, da14929e, 152b7d70; AC 288371a | Card details get a button per specific item deck (13 cards with Tainted Cain); the Specific Item Decks bag and the Deck of Rejects are gone. Decks spawn beside the card, clear of every hand zone. The rulebook is left alone (owner) |
| Website | TC 113e3a41 | A card's arrows keep the grid's filters (under `<base>`, a hash link on `c/<id>/` reloaded the site root) |
| Alt template support | AC 1d7ff1d (deployed) | Layer-comp stat bars read (SoLd `comp`, comp names as values), baked (`set_comp`) and shown as "template" bars |
| 0-20 bars | TC 9878d123 | The Alt template's bar doc in Photoshop: comps 11-20 are a full top row plus a bottom half row |
| Full conversion | AC 9ac5236, 0190d2e; TC batches 1-8 (c79cc4a2, c63597c3, bd8745fc, 117664d7, f4b2f573, d1a8e80c, d65536ba, 23865bd8) | `card_sweep --kind alt-template`: 356 creatures and minions. Darks as the template's bars on each card's rows (half rows above 10, resized bars), plus where the render moves at most 24 levels: bar art as vector shapes, the text layout ("Stat Text and Name"), the clip bases. Text drop shadows kept (owner). Every PSD opened in Photoshop before pushing |
| Raid boss template | TC 4406bd0c | `Alt_Raid_Boss_Template.psd`: the same bars on 20 cells, comps 0-40 |
| Stats to 20 | AC 9ac5236 | STATS TOO HIGH only above 20 (`card_models.STAT_LIMIT`; history `STAT_MAX` 20); the bake's one-row limit stays 10 |
| Reports | AC 9ac5236 | Only actionable, as short as possible: `"{name} pushed {card}: {notes}."`, no closing line, no changes list |
| Website preview | AC 0ee2d10 | A stars edit that only moves a card to its "N Stars" folder (printed stars not redrawn) previewed as "the bake did nothing" (the real bake moved it): now baked, shown with the card's own picture. 148 census edits on 86 cards. A card maker's stat bar beside the canvas is refused with its reason, not a crash (Computer Virus) |
| Stats past 10 | AC dc9f674; TC d460db0e | The bake's planner sent 11-20 on the Alt template to website only (it checked the old rows before the layer comps; the census tried only ±1): now a job up to the bar's top comp (0-20, raid boss 0-40), website only past it. Raid bosses (6+ stars, `card_models.RAID_STARS`) have no stat limit. The website's stat slider edits any stat in place: a full row opens an empty one below (10 opens 11-20), up to 99; Royal Eradicator Main Cannon's too |
| Dead code | AC 3bf99d7 | `card_data._icons_from_local`, `card_standards._edge`, `query_card.get_card_path` |
| Problem cards | AC docs/Future/PROBLEM_CARDS_2026-11-06.csv + .md | Census of 13 website edits on every card after the rollout, the cards not on the Alt template, the validator |

## Findings

- **Photoshop 2023 refuses a PSD holding an embedded smart-object document no layer uses** ("program error"; JSX:
  "the open options are incorrect"). psd-tools' `remove()` of a smart object leaves it. A graft of the 0-20 bars into
  the committed template hit it; Photoshop's own save went in instead. `card_standards._orphan_links` now gates the
  conversion (memory `photoshop-orphan-linked-data`).
- **The renderer shifts Igor Roballtowski's pattern overlay** (its Unknown type icon) in a whole-card render whenever
  any layer changes, though the layer alone renders the same: a fork quirk in the whole-pass render, not the card.
  Igor got the bars only; its PNG is unaffected (the export redraws only the bars).
- **The sweep's workers raced on the template file** (each wrote it at start): two cards read it half written.
  Written whole, then renamed (0190d2e).
- The 5 cards with split titles (two layers named Name) need every Name layer moved together.
- Census after the rollout (5,640 dry bakes of 13 edits on 768 cards, plus single stats on Computer Virus and Ze'mer): 32 cards have an edit that can't bake, none caused by the Alt template; 44 creatures and minions are not on it (39 hand-drawn or custom bars, 4 with several creatures, Igor partly). Whole-field title and ability edits on 36 multi-layer cards were left out: the website edits each box.

## Owner

- Check `MDW/Templates/Alt_Raid_Boss_Template.psd` (raid bosses, 6+ stars, have no stat limit).
- The hand-drawn and several-creature cards are on the problem list for a manual pass.
- Tell card makers about the Alt template (owner's call).
