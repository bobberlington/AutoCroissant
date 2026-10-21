# Card standardization campaign (2026-10-20)

**Status:** idea 2026-10-20, not scheduled; plan item 21 (`docs/NEXT_SESSION_PLAN.md`, "Ideas 5-21"). Nothing measured yet.

## 1. Goal

Every card PSD matches its template's layer structure and names (`~/Desktop/TTSCardMaker/MDW/Templates/`: `Creature_Template.psd`,
`2_Creature_Template.psd`, `3_Creature_Template.psd`, `Item_Template.psd`, `ActiveItem_Template.psd`, `Raid_Boss_Template.psd`,
`Status_Effect_Template.psd`), so the edit pipeline (Task 3 step 4), the parser (`commands/psd_parser.py`) and the renderer can rely
on it: a stat row is `Stats Bars > Darks > <Stat> Dark > 0..9`, the name `Text > Name`, the ability `Text > Ability`, types and stars
smart objects under `Creature Types` and `Stars`. Non-goal: no change to what a card says or shows; the PNG stays the same.

## 2. Inputs

- The templates above: the authority for structure and names.
- The parser's expectations: `.claude/skills/impossibility-cards-reference/SKILL.md` §3 (PSD anatomy, annotated layer dump) and §5
  (validation); `commands/psd_parser.py` (`CardValidator`, `ABILITY_EXCLUSIONS`, `EXCESSIVE_STAT_EXCLUSIONS`). Tools:
  `.claude/skills/autocroissant-diagnostics-and-tooling/scripts/dump_psd_layers.py`, `parse_one.py`; the fork's layer edits (Task 3).
- The standardization rules, to be agreed with the user: which deviations are errors, which are allowed variants (several-creature
  cards, raid bosses, status effect cards, decoration layers).

## 3. Phases

- **P1. Inventory.** A read-only script over all ~813 PSDs: per card, every deviation from its template (missing, extra, renamed or
  misplaced layers; text where the template has a smart object; a stat row with two visible digits; stats above 10, item 13).
  Report by deviation kind with counts, then per card. Gate: it reproduces the known cases (the exclusion-list cards, Sonic, the
  June 2026 WoW batch with blank ability layers) before any rule is written.
- **P2. Rules.** The user picks, per deviation kind: fix, allow (and the parser learns it), or leave. Written here as a table.
- **P3. Fix.** By the edit pipeline on the fork, one commit per card (PSD + re-exported PNG), gated like any edit: the re-parse
  gives the same name, stats, types and ability, and the PNG is unchanged within the harness's per-region bar. Cards the pipeline
  cannot fix are listed for Photoshop.
- **P4. Guard.** A GitHub Action on push to TTSCardMaker runs P1 on the changed PSDs and reports (same job shape as the rulebook
  checker, `docs/Campaigns/RULEBOOK_CONSISTENCY_CAMPAIGN.md` P4); later, once P3 has proven a fix kind on every card, it fixes.

## 4. Dependencies and gates

Needs Task 3 steps 1-4 (harness, rendering, layer edits, the edit pipeline) and the two-kinds-of-edits rule (plan, Queued). Gates:
G1 P1 before P2; G2 no fix without an agreed rule; G3 a fix never changes the parsed card or the PNG beyond the bar; G4 report-only for a month.

## 5. Open questions for the user

1. Are several-creature and raid-boss cards their own standards, or deviations from `Creature_Template.psd`?
2. Do decoration layers (art overlays, `Ability FX: ...` effect layers) count, or only the layers the parser reads?
3. Does the Action ever fix on push, or only report (PSDs cannot be merged; the author may still be editing)?
