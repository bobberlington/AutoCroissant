# Session 2026-10-22 (morning to afternoon; Opus 5.5, ultracode)

Asked: resume the 2026-10-22 plan (`docs/Finished/session-2026-10-22/PLAN_AS_OF_2026-10-22.md`) after the previous session
hit its usage limit at 05:51, then the owner's many requests during the day. Ran 05:51 to about 17:00, with two power cuts
(14:49 and ~15:50). The running log is `~/Documents/Tools/session-rescue-2026-10-22/resume/SESSION2_NOTES.md`; what was left
went into `RESUME_2026-10-22.md` (now `docs/Finished/session-2026-10-23/`), and the evening session carried it on
(`docs/Finished/session-2026-10-23/README.md`). Commits: AutoCroissant about 36 on main in the window (PICKLEs aside),
TTSCardMaker 64.

## Landed (pushed)

**TTSCardMaker.**
- Standardization batch: 242 cards with stoogedings tokens and badges, 28 commits pushed in 6 chunks 07:29-07:42
  (origin 56a79f4f). The first gate run had landed 293 cards without most visual reviews (a usage-limit burst killed
  14 agents); its 36 commits were moved to `std/batch-landed-unreviewed` and main reset before anything was pushed.
  The re-run held 106 cards; a pre-push check dropped Afraid.
- Queue A (5 commits, origin 07589cc7): 19 gradient-Ability cards get a style-less `Ability FX: <tags> | <tokens>` badge
  layer above the Text group, 3 multi-layer cards, 13 rulebook pages (6 hidden layers deleted, `tornado` -> `wind`,
  badges), God Tamer 5 stars and Bubble 1 star (L14), the Asteroid Field PNG.
- Title case fixes in Card Edits.json (5034c4cf: Springtrap, Dark matter, Metal General, Masked Dedede, Star Dream, Zero
  Override). Zote and Knight Lost A Lot need no entry (no Name layer: the file stem shows).
- Balance analytics (af826714), the badge colour legend on the rulebook viewer and card pages (cd77aad7, 7886ac43), badges
  in any letter case (6e50c3ee: a GSUB-only font change; only A P E X P R E D A T O R's "(action)" changes today).
- The tts-save Action ran (5dd2a408: 1584 links pinned, 6 Nicknames fixed). card-edits.yml fired its first scheduled run
  (c6dc6d08: 1 baked, 8 refused): Springtrap baked with the old TySh-only writer, which left a stale Txt2 story.

**AutoCroissant.**
- `/query as_of:` (0905447), deployed 07:18.
- Discord archive miner and matcher, squash-merged without mined data (7690c84); the matcher dates 754 of 772 site cards
  (645 high, 109 confident) from 87,317 messages and 9,889 images; `cb8fe54` (no alias proposals).
- Rule pages skip hidden text layers (work/rule-hidden, merged 6e84d63); goldens re-pinned after the batch (9f55895:
  Freezer and Lich King parse-identical, only the blob; 14 goldens re-pinned with proofs in tests-follow, suite 704 passed).
- Any-case badge parity in the bot's tests and docs.
- Jetson deploys: 0905447 (07:18), 9f55895 (08:36).

**psd-tools fork.** Round 5 integrated (`cards` 592aa335: curves, blend channels, drop-shadow knockout, the user's
round-4 review); round 5b (bricon, group stroke, gradient bars; `cards` 2d7d1a30, then da3b0f6d with fix/txt2-sync):
bad px 1,915,557 -> 605,754 on that day's corpus, median 0.030% -> 0.000%, >0.5% 29 -> 9, >2% 3 -> 1 (max 2.417),
0 cards worse. Pushed, not tagged. Research prototypes banked on `research/speed-proto`, `research/fidelity-exp`,
`research/speed-bench`.

## Built, not landed by the end (landed in the evening session)

- AutoCroissant `work/integ-1022` (data build steps 1-6 and 9: card_data/card_history/card_db/changelog, Discord writers,
  Txt2 story gate, history skip-identical and one-time duplicate collapse, only-printed Ability text, parity mean 2.0,
  badge-layer refusal, `_bot` provenance). Parity gate 1: legacy /update_stats byte-identical; gate 2: a full re-parse of
  812 PSDs, 0 data-field differences.
- TTSCardMaker `work/integ-1022` (alias and Discord-only series migration: 60 aliases, 596 series; Changes page tabs;
  via-Discord credit; card-sync.yml dispatch only), fork `fix/txt2-sync` (title bakes update the Txt2 story), the B2
  Photoshop queue (reconstructed after the reboot).

## Owner rulings (do not ask again)

Parity bar mean 2.0 (0.5% bad, max 24 kept; superseded 2026-10-23: misses no longer refuse). Badges scale to the body
text on small-glyph fonts. `[dark:sticker]` in black-stroked layers. Dark Mind Eye and Kel: restore the gradient maps.
Discord-only series move into Card Edits.json. EarthboundZero cards badged with the fork (blocked until colour glyphs).
11_Abilities3 markers moved; 22_Negation reworded ("Cooldown abilities such as (2CD) still count down even while they are
negated."), re-laid and badged. psd-tools rounds until the gains dry up (< ~2% of the remaining bad pixels). Points stay
plain text (point badges reversed at ~10:00). Badges in any letter case. A badge colour legend on 02_Abbreviations and the
website, one line per colour, badges only. History: skip identical versions and clean the stored duplicates. Only printed
Ability text counts. Title case: only clear typos, Title Case with small words lower, partial caps and the 25 ALL CAPS
kept, with the next Photoshop pass. Repo size: cut churn (one rewrite per card per pass), no history rewrite. Discord:
best guesses accepted where they cannot move a date; the website uses Discord creation dates; no Discord aliases;
authors reviewed by the owner. Shadow as designed (3 days or 10 pushes); the owner tells bobberlington. Computer Virus
left flagged. Changes page tabs at the top. psd-tools next: the stale-PNG guard (re-export only when a push changes the PSD
without its PNG), the Photoshop oracle loop, a standardization guard posting to a new standards channel without a ping,
render cache, template fill, per-layer geometry. `ACTIONS_TOKEN` is in the jetson's config.py (~12:25). Then: work
unattended, no questions.

## Incidents

- **Usage-limit bursts** killed agents twice (05:51 and 06:20): the batch gates' first run landed 293 cards with few
  visual reviews. Backed out before any push; the rule since: a dead reviewer fails the workflow.
- **Pre-push checks found real defects:** The_Puppet's badges twice the cap height, overlaps on Unbread_Twins and omori,
  Crock_Pot's squashed pills (run 1); Afraid's lost icon halo (run 2, dropped); queue A held Sweetheart, Papa_Chip,
  Black_Light_Bulb, Buzzo, Palace_of_Shadow and 11_Abilities3.
- **BAKE BUG (title edits):** card_bake set only the layer's TySh text, not the document Txt2 story Photoshop reads, so
  a Photoshop save reverted the bake. Fixed on fork `fix/txt2-sync` and bot `work/bake-txt2`; the 6 stale stories were
  repaired later with `commands/txt2_repair.py`.
- **BUG 2:** an ability bake deletes `Ability FX: ...` badge layers; the bot refuses those bakes until the FX layers can
  be rebuilt (still open: BUGS D7).
- **Two power cuts.** The 14:49 reboot wiped `/private/tmp`: queue B1's fixed driver, B2's pilot fixes, queue A's gradient
  tooling and every /tmp worktree. Recovery moved all work to `~/Documents/Tools/wt/`, `~/Documents/Tools/psd-research/
  std-batch2/` and `~/std-work-1022/`; fsck clean; bundles of unpushed branches in
  `~/Documents/Tools/session-rescue-2026-10-22/bundles/`.
- **card-edits.yml disabled at ~16:50:** main's bake had neither the Txt2 gate nor the colour-glyph refusal.

## Lessons

- `/tmp` dies with a reboot: persistent paths only, and bank WIP to branches often.
- A dead reviewer must fail the workflow, never count as "no findings".
- Run reviewers in small groups (burst usage limits).
- Check every agent's result before pushing; keep a pre-push visual check for anything visual.
- zsh mangles `git push origin <sha>:main`: quote the refspec, `"${c}:refs/heads/main"`.
