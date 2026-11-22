# Branches and bundles (11-21)

Every branch, bundle ref and git store of the 11-10..21 session, adjudicated by the second cleanup (11-21 14:28-14:54)
and condensed from `~/Documents/Tools/wt/scratch/s1110/wrap/adjudicate/branches.tsv` (290 rows; the full evidence per
row is there). Refs at the same tip share one line here. Every tip's full SHA is also in
`~/Documents/Tools/wt/scratch/s1110/wrap/branch_record.tsv` (200 rows). Updated to 11-21 ~16:00: r15-2 landed, cards
r44-2 rescued into bundles, the owner's rulings below.

Counts (branches.tsv): SUPERSEDED 87, HELD 60, LANDED-REBASED 60, REJECTED 36, POLICY-KEEP 22, LANDED 17, UNKNOWN 8.
Since then: the fork's `work/research-1110` went HELD -> LANDED-REBASED (r15-2), the 7 stranded keeps went UNKNOWN ->
STRANDED KEEP (owner ruling 4), and `work/pin-1121` and the fork's `work/fork-r15-2` went POLICY-KEEP -> LANDED. One
UNKNOWN is left: the previous session's `scratch/s1109/wrap/tc-unlanded-1110.bundle` (below).

## Owner rulings (11-21 15:01)

1. LANDED, LANDED-REBASED, SUPERSEDED tips leave the bundles when the orchestrator rebuilds them (end of 11-21);
   branch_record.tsv keeps every SHA.
2. REJECTED tips stay in the bundles.
3. HELD: "retry the most impactful ones, to finish within ~3 hours. drop the rest." Workflow held-1121 does it (target
   ~18:00); at this writing it is building actbake r40-1, x30 and the stranded keeps (r11-0, r11-1 so far). What it
   lands gets ticked in `docs/NEXT_SESSION_PLAN.md`.
4. The 7 stranded keeps (verified, never shipped): "Ship now, re-verified" (held-1121).

## Bundles

In `~/Documents/Tools/wt/scratch/s1110/bundles/` (persistent; all pass `git bundle verify`):

| Bundle | Refs | Made |
|---|---|---|
| `ac-unlanded-1121.bundle` | 85: `refs/heads/work/*`, `refs/heads/backup/*`, `refs/wrap/<sha8>` | 12:38 |
| `tc-unlanded-1121.bundle` | 73, same kinds | 12:38 |
| `fork-unlanded-1121.bundle` | 21 `refs/heads/work/*` | 12:38 |
| `ac-r44-2-1121.bundle` | `refs/heads/cards-1110-r44-2` (3017ef00; needs 91ddced3) | 14:57, from the Desktop clone's objects |
| `tc-r44-2-1121.bundle` | `refs/heads/cards-1110-r44-2` (f12a757e; needs aab4fffc) | 14:57, from the Desktop clone's objects |

The Bundle column below names them without `-1121.bundle`. Ruling 1's rebuild may rename or merge them: `git bundle
list-heads <bundle>` shows what a bundle holds. Get a tip back (TC with `GIT_NO_LAZY_FETCH=1`):

```
B=~/Documents/Tools/wt/scratch/s1110/bundles
git -C ~/Desktop/AutoCroissant bundle list-heads $B/ac-unlanded-1121.bundle | grep r40-1
git -C ~/Desktop/AutoCroissant fetch $B/ac-unlanded-1121.bundle refs/heads/work/actbake-1110-r40-1:refs/heads/work/r40-1
GIT_NO_LAZY_FETCH=1 git -C ~/Desktop/TTSCardMaker fetch $B/tc-unlanded-1121.bundle refs/wrap/2b11af0f:refs/heads/work/x30
git -C ~/Desktop/AutoCroissant fetch $B/ac-r44-2-1121.bundle refs/heads/cards-1110-r44-2:refs/heads/work/r44-2
git -C ~/Documents/Tools/psd-tools fetch $B/fork-unlanded-1121.bundle refs/heads/work/r5-3:refs/heads/work/r5-3
```

General form: `git fetch <bundle> <ref>:<local>`. The SHAs are unchanged. Work on the fetched branch in your own
worktree (`~/Documents/Tools/wt/<name>`), never in a Desktop clone's working tree.

## AutoCroissant

### HELD (23)

| SHA | Item | Ref in the bundle | Bundle | Reason |
|---|---|---|---|---|
| 137bdd7e | actbake r1-1 | `work/actbake-1110-r1-1` | ac-unlanded | a shrunk new Ability sits on the old, larger baseline (pushed down, once off the card) |
| 5598470f | actbake r10-0 -> x5 | `work/actbake-1110-r10-0` | ac-unlanded | x5 split 2-1 by the judges; check on the next guarded push whose fixes change a picture |
| 4db5c3ce | actbake r23-0 | `work/actbake-1110-r23-0` | ac-unlanded | its new PDF step starts before the book fonts are in the checkout |
| 03a1248e | actbake r39-0 | `work/actbake-1110-r39-0` | ac-unlanded | the new type icon looks worse on 6 of its 32 cards (White Space most) |
| dd18a196 | actbake r40-1 | `work/actbake-1110-r40-1` | ac-unlanded | a new helper `_offset` replaced psd_edit's own and broke the comp bakes (rename it) |
| aec4ed01 | actbake r42-0 | `work/actbake-1110-r42-0` | ac-unlanded | a late warm-up hand-over can point the next press at an ended lane |
| 6480d339 | actbake r52-1 | `work/actbake-1110-r52-1` | ac-unlanded | a second edit of a shrunk title prints at full size over the icons |
| 61be6f6a | actbake r54-1 | `work/actbake-1110-r54-1` | ac-unlanded | the AC bake part holds; the TC mock-up part made a follow-up preview worse |
| a35ce841 | actbake r7-0 -> x2 | `work/actbake-1110-r7-0` | ac-unlanded | x2 split 2-1; retry parts (a) and (b) without the dy move |
| 168c2502 | actbake r7-2 | `work/actbake-1110-r7-2` | ac-unlanded | about 1 one-number edit in 5 shows bits of the old digit or the wrong text |
| a1af2a3a | cards r10-0 | `work/cards-1110-r10-0` | ac-unlanded | some related strips stay blank on desktop (29 tiles on 7 of 41 pages) |
| a074e520 | cards r10-1 | `work/cards-1110-r10-1` | ac-unlanded | history positions shift between builds (wrong history after a deploy) |
| 98dffe2b | cards r22-0 | `work/cards-1110-r22-0` | ac-unlanded | the ban list's `<img>` lost its shape under `sizes="auto"` |
| 2f5642a1 | cards r25-0 | `work/cards-1110-r25-0` | ac-unlanded | more blank tiles while scrolling on slow networks |
| 3017ef00 | cards r44-2 | `cards-1110-r44-2` | ac-r44-2 | shows the wrong table (Series while Subseries is pressed) |
| 22e09b0c | details memo (fork r3-1) | `work/r3-1` | ac-unlanded | rebase onto AC main and measure against the research bar (fork lane: -40% gather CPU) |
| 485dd38b | details memo (fork r4-3, carries r3-1) | `work/r4-3` | ac-unlanded | same item, carries r3-1 (-54% details CPU on its sample, round 23) |
| e7e6e3ee | x26 | `refs/wrap/e7e6e3ee`, `work/cards-1110-x26` | ac-unlanded | icons blank in rulebook search results and same-book jumps; ship the wake() alone |
| 129df760 | x27 | `refs/wrap/129df760`, `work/actbake-1110-x27` | ac-unlanded | the stored ink breaks Card sync's merge of bake records (jq argument over 128 KB) |
| ac82a0e7 | x30 | `refs/wrap/ac82a0e7`, `work/cards-1110-x30` | ac-unlanded | opening the profile editor from a dialog puts focus on the header |
| fa13c489 | x31 | `refs/wrap/fa13c489`, `work/actbake-1110-x31` | ac-unlanded | typed letters keep a stray run's size and baseline shift (D15 rest, owner-directed) |
| 6109c566 | x37 | `refs/wrap/6109c566`, `work/actbake-1110-x37` | ac-unlanded | a move to another tab in the first second still waits about 18 s |
| c6e4ec99 | x4 | `refs/wrap/c6e4ec99`, `work/actbake-1110-x4` | ac-unlanded | Dr. Robotnik's Roboticizer's apostrophe (another font) gets restyled |

### STRANDED KEEP (3)

| SHA | Item | Ref in the bundle | Bundle | Reason |
|---|---|---|---|---|
| ef95dcbf | fork r11-0 template bar pixels raw | `work/r11-0` | ac-unlanded | Empress Bulblax's bars drawn as Photoshop draws them (with r11-1: Card checks -9.9%); owner 15:01: ship now, re-verified (held-1121) |
| 031fb786 | fork r11-1 a merge's view redrawn on its rows | `work/r11-1` | ac-unlanded | a merge's view redrawn on its rows only (Card checks 152.3 -> 137.3 s on 13 cards, with r11-0); owner 15:01: ship now, re-verified (held-1121) |
| 6a2f49b6 | fork r9-2 art fill alpha | `work/r9-2` | ac-unlanded | art fill reads the alpha alone; owner 15:01: ship now, re-verified (held-1121) |

### REJECTED (17)

| SHA | Item | Ref in the bundle | Bundle | Reason |
|---|---|---|---|---|
| 519a182c | Fable actbake r2-1 | `work/fable-actbake-r2-1` | ac-unlanded | fable verify keep=false: gain small and only modelled, three things worse |
| cf5328bd | Fable cards r2-0 | `work/fable-cards-r2-0` | ac-unlanded | fable verify keep=false: a fresh save gets set back to older pictures |
| 26558715 | Fable cards r2-1 | `work/fable-cards-r2-1` | ac-unlanded | fable verify keep=false: every GitHub run would fail (commit with no author on the runner) |
| 748fd25c | actbake r1-3 | `work/actbake-1110-r1-3` | ac-unlanded | gain does not hold on real push runs; adds a way for a run to stall minutes |
| a994a101 | actbake r12-0 | `work/actbake-1110-r12-0` | ac-unlanded | saving and correctness check out, but no gain over what main serves now |
| 79ee48f9 | actbake r15-3 | `work/actbake-1110-r15-3` | ac-unlanded | breaks a site contract with an owner (cards.json `img`) |
| 900a12c7 | actbake r24-2 | `work/actbake-1110-r24-2` | ac-unlanded | not built (builder): Card checks runs with oxipng off on GitHub, so no gain where it matters |
| a4284a5f | actbake r25-0 -> x15 | `work/actbake-1110-r25-0` | ac-unlanded | x15 dropped by judges: against the exact mock-up (owner 11-21 ~01:45, stays dropped) |
| 4a988af3 | actbake r31-0 | `work/actbake-1110-r31-0` | ac-unlanded | correct and safe, but the gain was never measured (evidence confounded) |
| 05a4a3c2 | actbake r47-1 | `work/actbake-1110-r47-1` | ac-unlanded | projected gain rests on a stall model the real runs contradict |
| c4566ab7 | cards r4-0 | `work/cards-1110-r4-0` | ac-unlanded | round 4 reject (lazy icons far down); same idea later x26, not kept |
| ca4b47db | f6 (finish n6) | `refs/wrap/ca4b47db`, `work/actbake-1110-f6` | ac-unlanded | not kept: moved a Warcraftian Hunter line visibly; r60-3 (shipped 11-20) is the same idea and covers it |
| 790f1d4d | fork r10-0 stars sheet WebP method 4 | `work/r10-0` | ac-unlanded | fork verify keep=false: every starred card's plate download much bigger |
| c324a802 | fork r11-2 bars with no comp applied | `work/r11-2` | ac-unlanded | not built (builder built=false); the actbake r20-2 fix covers Pandanoko/Retro Kirby |
| c07080b2 | fork r19-1 bakes band the image straight | `work/r19-1` | ac-unlanded | fork verify keep=false: gain about 3% on its base, needs rework |
| 10cf7a47 | fork r9-0 link paths | `work/r9-0` | ac-unlanded | fork verify keep=false: dropping LINK_DATE goes against owner ruling 11-07; BRIEF 11-15 'r9-0 (rejected; the owner decides link paths)' |
| 1173d8d3 | fork r9-1 recorded ICC vote | `work/r9-1` | ac-unlanded | fork verify keep=false: Paint_Warrior's mock-up plate becomes a wrong render |

### SUPERSEDED (41)

| SHA | Item | Ref | Reason |
|---|---|---|---|
| a07a0bd2 | T1 actions-1108 variant | `work/actions-1108-nover` | no-version-commit alternative; T1 landed work/actions-1108 with the version-commit fallback (AC fc8357f5, TC 83495964, 11-10) |
| 91173c50 | actbake r15-2 -> x7 | `work/actbake-1110-r15-2` | x7 dropped by judges: done since: mockbars-1111 draws custom-bar stat edits |
| 67ea9a6d | actbake r18-0 -> x10 | `work/actbake-1110-r18-0` | first try; retry x10 landed in wave 2a actbake |
| 1ff3d424 | actbake r20-1 | `work/actbake-1110-r20-1` | verifier keep; r20 merge applied the leaner r20-2 alone (same Pandanoko and Retro Kirby gain) |
| cda507ac | actbake r20-3 -> x11 | `work/actbake-1110-r20-3` | x11 dropped by judges: done since: TC cce549f7 (--help half); fonts rarely on the save path |
| 4c85ab92 | actbake r24-0 -> x12 | `work/actbake-1110-r24-0` | first try; retry x12 landed in wave 2b actbake |
| b8d942b9 | actbake r24-1 -> x13 | `work/actbake-1110-r24-1` | first try; retry x13 landed in wave 2a actbake |
| 8bcbce16 | actbake r30-1 -> x18 | `work/actbake-1110-r30-1` | first try; retry x18 landed in wave 2a actbake |
| 8d218b92 | actbake r32-2 -> x19 | `work/actbake-1110-r32-2` | first try; retry x19 landed in wave 2a actbake |
| 2044678a | actbake r32-3 -> x20 | `work/actbake-1110-r32-3` | first try; retry x20 landed in wave 2a actbake |
| a2067b0f | actbake r34-1 -> x22 | `work/actbake-1110-r34-1` | first try; retry x22 landed in wave 2a actbake |
| e9007d4a | actbake r34-2 | `work/actbake-1110-r34-2` | verifier keep; left out as the competing design of r34-0, which shipped (ships.txt l.408) |
| 7e3f0338 | actbake r36-3 -> x23 | `work/actbake-1110-r36-3` | first try; retry x23 landed in wave 2b actbake |
| 204ccd8d | actbake r44-1 -> x27 | `work/actbake-1110-r44-1` | first try; retry x27 not kept (its build is HELD as refs/wrap/<sha>) |
| 32e161d6 | actbake r47-3 -> x29 | `work/actbake-1110-r47-3` | first try; retry x29 landed in wave 2a actbake |
| 4799ea03 | actbake r51-1 -> x31 | `work/actbake-1110-r51-1` | first try; retry x31 not kept (its build is HELD as refs/wrap/<sha>) |
| 2f2f215f | actbake r53-0 -> x33 | `work/actbake-1110-r53-0` | first try; retry x33 landed in wave 2b actbake |
| fff68cf1 | actbake r53-2 -> x34 | `work/actbake-1110-r53-2` | first try; retry x34 landed in wave 2b actbake |
| 1dc972d4 | actbake r56-3 -> x37 | `work/actbake-1110-r56-3` | first try; retry x37 not kept (its build is HELD as refs/wrap/<sha>) |
| dd811875 | actbake r58-3 -> x38 | `work/actbake-1110-r58-3` | first try; retry x38 landed in wave 2b actbake |
| ff6ed73d | actbake r7-3 | `work/actbake-1110-r7-3` | verifier keep; rival fix of r7-1, backed out at the r7 merge; r7-1 landed |
| bb9b8c16 | actbake r9-2 -> x4 | `work/actbake-1110-r9-2` | first try; retry x4 not kept (its build is HELD as refs/wrap/<sha>) |
| f9571c4b | cards r10-3 -> x6 | `work/cards-1110-r10-3` | first try; retry x6 landed in wave 2a cards |
| 4065996d | cards r14-2 -> x8 | `work/cards-1110-r14-2` | x8 dropped by judges: done since: TC c35c13c0 + 055cdc12 (Changes head file) |
| d90f88c7 | cards r15-1 -> x9 | `work/cards-1110-r15-1` | first try; retry x9 landed in wave 2a cards |
| 07db5547 | cards r21-1 -> x16 | `work/cards-1110-r21-1` | first try; retry x16 landed in wave 2a cards |
| 111a63a2 | cards r3-2 -> x1 | `work/cards-1110-r3-2` | first try; retry x1 landed in wave 2a cards |
| 048e0406 | cards r38-0 -> x26 | `work/cards-1110-r38-0` | first try; retry x26 not kept (its build is HELD as refs/wrap/<sha>) |
| b0264da7 | cards r45-0 -> x28 | `work/cards-1110-r45-0` | first try; retry x28 landed in wave 2a cards |
| b3287031 | cards r48-2 -> x32 | `work/cards-1110-r48-2` | first try; retry x32 landed in wave 2a cards |
| 44cef300 | cards r50-1 -> x35 | `work/cards-1110-r50-1` | first try; retry x35 landed in wave 2b cards |
| dc439a20 | cards r52-1 -> x36 | `work/cards-1110-r52-1` | first try; retry x36 landed in wave 2b cards |
| cf22e277 | cards r55-1 -> x40 | `work/cards-1110-r55-1` | first try; retry x40 landed in wave 2b cards |
| 9557435b | cards r55-2 -> x39 | `work/cards-1110-r55-2` | first try; retry x39 landed in wave 2b cards |
| 35d46f5b | cards r7 drawn bars | `backup/cards-1110-ship2-drawnbars` | parked accepted r7 change 'Parser: stat bars drawn as pictures' (ships.txt l.61): breaks test_lost_bars |
| 32e93eff | cards r8-0 -> x3 | `work/cards-1110-r8-0` | x3 dropped by judges: done since: AC b4b1b03a (11-11, same 10% ink rule) |
| a70c5b8a | cards ship 4 pre-state | `backup/cards-1110-ship4-pre` | pre-rebase backup of cards ship 4 (landed); 6/10 subjects on main, rest are test commits reworded in the ship |
| 4fd1c392 | cards ship 6 pre-state | `backup/cards-1110-ship6-pre` | pre-rebase backup of cards ship 6 (landed); 5/7 subjects on main |
| c1d40a71 | cbland-old detached head | `refs/wrap/c1d40a71` | 'Pin cards-20261112' superseded by later pins (main pins cards-20261114); its other 4 commits on main (cherry -) |
| 57c09abc | fork r12-0 PSB in card profile drawn raw | `work/r12-0` | BRIEF 11-15 'skipping r12-0 (superseded)'; main psd_links reads an RGB psd/psb with no ICC step (11-13: 3d60259c, 1c1c8780, 8bce0334) |
| 4dc712ad | held3 pre-state | `backup/cards-1110-r3s4-pre` | pre-rebase backup of cards held3; held3 landed 11-19; 12/12 subjects on main (cherry 11 of 12 -) |

### LANDED-REBASED (30)

| SHA | Item | Ref | Reason |
|---|---|---|---|
| ef5359da | actbake r18-0 first build | `refs/wrap/ef5359da` | 'Plates: layout gate leaves icon glyphs out' on main as 2f7e22fe (11-16, research3 ship 1); 31/35 added lines on main |
| 0c414935 | actbake r55-0 | `work/actbake-1110-r55-0` | verifier keep; rebuilt on r55-3 and landed in research4 ship 1 (11-19 14:22: AC 95f750f2, TC 7b37f7d9); its 'Re-pin waits 60 s' part never ported |
| 0e30a400 | f11 | `work/cards-1110-f11` | landed in wave 2a cards (AC ab9c7349, TC 68961a8b) |
| 96fcbdc2 | f5 | `work/actbake-1110-f5` | landed in wave 2a actbake (AC f0f4c7c1, TC 0911ce04) |
| f31a9b85 | f9 | `work/actbake-1110-f9` | landed in wave 2a actbake (AC f0f4c7c1, TC 0911ce04) |
| c3b801f0 | r57-3 | `work/actbake-1110-r57-3` | landed in wave 2a actbake (AC f0f4c7c1, TC 0911ce04) |
| 5c3d973d | r58-3 | `work/cards-1110-r58-3` | landed in wave 2a cards (AC ab9c7349, TC 68961a8b) |
| c0808277 | x1 | `work/cards-1110-x1` | landed in wave 2a cards (AC ab9c7349, TC 68961a8b) |
| 2d2dd92b | x10 | `work/actbake-1110-x10` | landed in wave 2a actbake (AC f0f4c7c1, TC 0911ce04) |
| ce971e75 | x12 | `work/actbake-1110-x12` | landed in wave 2b actbake (AC ea0db694, TC 0cb95c33) |
| 78eebe74 | x13 | `work/actbake-1110-x13` | landed in wave 2a actbake (AC f0f4c7c1, TC 0911ce04) |
| 67fdb0c8 | x16 | `work/cards-1110-x16` | landed in wave 2a cards (AC ab9c7349, TC 68961a8b) |
| 103d3863 | x17 | `work/cards-1110-x17` | landed in wave 2a cards (AC ab9c7349, TC 68961a8b) |
| 366ddee7 | x18 | `work/actbake-1110-x18` | landed in wave 2a actbake (AC f0f4c7c1, TC 0911ce04) |
| 4e4fa030 | x19 | `work/actbake-1110-x19` | landed in wave 2a actbake (AC f0f4c7c1, TC 0911ce04) |
| cf462abb | x20 | `work/actbake-1110-x20` | landed in wave 2a actbake (AC f0f4c7c1, TC 0911ce04) |
| 172f837d | x22 | `work/actbake-1110-x22` | landed in wave 2a actbake (AC f0f4c7c1, TC 0911ce04) |
| 2bd50aea | x23 | `work/actbake-1110-x23` | landed in wave 2b actbake (AC ea0db694, TC 0cb95c33) |
| 7ea8f2a8 | x28 | `work/cards-1110-x28` | landed in wave 2a cards (AC ab9c7349, TC 68961a8b) |
| aeb94b86 | x29 | `work/actbake-1110-x29` | landed in wave 2a actbake (AC f0f4c7c1, TC 0911ce04) |
| a466c998 | x32 | `work/cards-1110-x32` | landed in wave 2a cards (AC ab9c7349, TC 68961a8b) |
| 635c08c6 | x33 | `work/actbake-1110-x33` | landed in wave 2b actbake (AC ea0db694, TC 0cb95c33) |
| e0dbaa63 | x34 | `work/actbake-1110-x34` | landed in wave 2b actbake (AC ea0db694, TC 0cb95c33) |
| 5aa9a97a | x35 | `work/cards-1110-x35` | landed in wave 2b cards (AC 0d2de792, TC f769e0fc) |
| 6258195d | x36 | `work/cards-1110-x36` | landed in wave 2b cards (AC 0d2de792, TC f769e0fc) |
| 39b23fe1 | x38 | `work/actbake-1110-x38` | landed in wave 2b actbake (AC ea0db694, TC 0cb95c33) |
| 1ac19968 | x39 | `work/cards-1110-x39` | landed in wave 2b cards (AC 0d2de792, TC f769e0fc) |
| 325eeb1c | x40 | `work/cards-1110-x40` | landed in wave 2b cards (AC 0d2de792, TC f769e0fc) |
| 6df3dd13 | x6 | `work/cards-1110-x6` | landed in wave 2a cards (AC ab9c7349, TC 68961a8b) |
| 87b499e4 | x9 | `work/cards-1110-x9` | landed in wave 2a cards (AC ab9c7349, TC 68961a8b) |

## TTSCardMaker

### HELD (24)

| SHA | Item | Ref in the bundle | Bundle | Reason |
|---|---|---|---|---|
| bd3d3fcc | actbake r1-1 | `work/actbake-1110-r1-1` | tc-unlanded | a shrunk new Ability sits on the old, larger baseline (pushed down, once off the card) |
| 8496a843 | actbake r10-0 -> x5 | `work/actbake-1110-r10-0` | tc-unlanded | x5 split 2-1 by the judges; check on the next guarded push whose fixes change a picture |
| adf4f5e5 | actbake r2-1 | `work/actbake-1110-r2-1` | tc-unlanded | Life Link's text runs off the card again; Sans, Poorly Drawn Cabbage, Mangle worse |
| 040ea6f2 | actbake r23-0 | `work/actbake-1110-r23-0` | tc-unlanded | its new PDF step starts before the book fonts are in the checkout |
| c662cfeb | actbake r39-0 | `work/actbake-1110-r39-0` | tc-unlanded | the new type icon looks worse on 6 of its 32 cards (White Space most) |
| 9566bd52 | actbake r40-1 | `work/actbake-1110-r40-1` | tc-unlanded | a new helper `_offset` replaced psd_edit's own and broke the comp bakes (rename it) |
| 6199cb8d | actbake r42-0 | `work/actbake-1110-r42-0` | tc-unlanded | a late warm-up hand-over can point the next press at an ended lane |
| fb14d299 | actbake r52-1 | `work/actbake-1110-r52-1` | tc-unlanded | a second edit of a shrunk title prints at full size over the icons |
| 33eb406a | actbake r54-1 | `work/actbake-1110-r54-1` | tc-unlanded | the AC bake part holds; the TC mock-up part made a follow-up preview worse |
| abb071f5 | actbake r60-2 (AWS warm-up) | `work/actbake-1110-r60-2` | tc-unlanded | proved (+3-5 s on AWS jobs), not shipped: decide against the research bar; AC side on main |
| ae380cfa | actbake r7-0 -> x2 | `work/actbake-1110-r7-0` | tc-unlanded | x2 split 2-1; retry parts (a) and (b) without the dy move |
| 25695371 | actbake r7-2 | `work/actbake-1110-r7-2` | tc-unlanded | about 1 one-number edit in 5 shows bits of the old digit or the wrong text |
| e4961e73 | cards r10-0 | `work/cards-1110-r10-0` | tc-unlanded | some related strips stay blank on desktop (29 tiles on 7 of 41 pages) |
| fc828639 | cards r10-1 | `work/cards-1110-r10-1` | tc-unlanded | history positions shift between builds (wrong history after a deploy) |
| 9a9ba84f | cards r21-0 | `work/cards-1110-r21-0` | tc-unlanded | the 64 status cards' square pictures get visibly softer |
| 2cfa5dd6 | cards r22-0 | `work/cards-1110-r22-0` | tc-unlanded | the ban list's `<img>` lost its shape under `sizes="auto"` |
| ea7d3d01 | cards r25-0 | `work/cards-1110-r25-0` | tc-unlanded | more blank tiles while scrolling on slow networks |
| f12a757e | cards r44-2 | `cards-1110-r44-2` | tc-r44-2 | shows the wrong table (Series while Subseries is pressed) |
| 494d9cba | x26 | `refs/wrap/494d9cba`, `work/cards-1110-x26` | tc-unlanded | icons blank in rulebook search results and same-book jumps; ship the wake() alone |
| 824bb993 | x27 | `refs/wrap/824bb993`, `work/actbake-1110-x27` | tc-unlanded | the stored ink breaks Card sync's merge of bake records (jq argument over 128 KB) |
| 2b11af0f | x30 | `refs/wrap/2b11af0f`, `work/cards-1110-x30` | tc-unlanded | opening the profile editor from a dialog puts focus on the header |
| d20c1105 | x31 | `refs/wrap/d20c1105`, `work/actbake-1110-x31` | tc-unlanded | typed letters keep a stray run's size and baseline shift (D15 rest, owner-directed) |
| 6e439e89 | x37 | `refs/wrap/6e439e89`, `work/actbake-1110-x37` | tc-unlanded | a move to another tab in the first second still waits about 18 s |
| d05bf5c0 | x4 | `refs/wrap/d05bf5c0`, `work/actbake-1110-x4` | tc-unlanded | Dr. Robotnik's Roboticizer's apostrophe (another font) gets restyled |

### REJECTED (14)

| SHA | Item | Ref in the bundle | Bundle | Reason |
|---|---|---|---|---|
| 25abffc8 | D2 (light theme AA) | `work/cards-1110-hold-aa` | tc-unlanded | bronze-gold light theme the owner declined (D2, 11-15); its D1/D4 landed |
| b3ca9158 | Fable actbake r2-1 | `work/fable-actbake-r2-1` | tc-unlanded | fable verify keep=false: gain small and only modelled, three things worse |
| 12db1594 | Fable cards r2-0 | `work/fable-cards-r2-0` | tc-unlanded | fable verify keep=false: a fresh save gets set back to older pictures |
| 4ff4a4ad | Fable cards r2-1 | `work/fable-cards-r2-1` | tc-unlanded | fable verify keep=false: every GitHub run would fail |
| 1e6a7c48 | actbake r1-3 | `work/actbake-1110-r1-3` | tc-unlanded | gain does not hold on real push runs; adds a way for a run to stall minutes |
| 8e52c509 | actbake r12-0 | `work/actbake-1110-r12-0` | tc-unlanded | saving and correctness check out, but no gain over what main serves now |
| e9fe455d | actbake r15-3 | `work/actbake-1110-r15-3` | tc-unlanded | breaks a site contract with an owner (cards.json `img`) |
| 56069edc | actbake r23-2 | `work/actbake-1110-r23-2` | tc-unlanded | claimed gain does not hold for real edits judged by eye |
| 6d1cb2d0 | actbake r25-0 -> x15 | `work/actbake-1110-r25-0` | tc-unlanded | x15 dropped by judges: against the exact mock-up (owner 11-21 ~01:45, stays dropped) |
| 743b8a6d | actbake r31-0 | `work/actbake-1110-r31-0` | tc-unlanded | correct and safe, but the gain was never measured (evidence confounded) |
| 15678aae | cards r29-0 | `work/cards-1110-r29-0` | tc-unlanded | not built (builder): 'reference only, do not ship' (the split no longer fits) |
| bf3558ed | cards r34-2 | `work/cards-1110-r34-2` | tc-unlanded | not built (builder): cards.json as columns already took most of the gain (-4% gzip) |
| 721c162f | cards r4-0 | `work/cards-1110-r4-0` | tc-unlanded | round 4 reject (lazy icons far down); same idea later x26, not kept |
| e1228102 | cards r4-1 | `work/cards-1110-r4-1` | tc-unlanded | round 4 reject (accessibility toggles, brand, underlines) |

### SUPERSEDED (33)

| SHA | Item | Ref | Reason |
|---|---|---|---|
| 369559a3 | T1 actions-1108 variant | `work/actions-1108-nover` | no-version-commit alternative; T1 landed with the version-commit fallback (TC 83495964); also in s1110/t1-land/tc-actions-1108-nover.bundle |
| a6f1e0d5 | actbake r15-2 -> x7 | `work/actbake-1110-r15-2` | x7 dropped by judges: done since: mockbars-1111 draws custom-bar stat edits |
| 3ae16ee7 | actbake r20-3 -> x11 | `work/actbake-1110-r20-3` | x11 dropped by judges: done since: TC cce549f7 (--help half); fonts rarely on the save path |
| 2d574dce | actbake r24-1 -> x13 | `work/actbake-1110-r24-1` | first try; retry x13 landed in wave 2a actbake |
| 9559391e | actbake r25-3 -> x14 | `work/actbake-1110-r25-3` | first try; retry x14 landed in wave 2a actbake |
| 9c832daa | actbake r30-1 -> x18 | `work/actbake-1110-r30-1` | first try; retry x18 landed in wave 2a actbake |
| 5e1b8616 | actbake r32-2 -> x19 | `work/actbake-1110-r32-2` | first try; retry x19 landed in wave 2a actbake |
| 38bd8829 | actbake r32-3 -> x20 | `work/actbake-1110-r32-3` | first try; retry x20 landed in wave 2a actbake |
| 2633f763 | actbake r34-1 -> x22 | `work/actbake-1110-r34-1` | first try; retry x22 landed in wave 2a actbake |
| f1fe3c90 | actbake r34-2 | `work/actbake-1110-r34-2` | verifier keep; left out as the competing design of r34-0, which shipped (ships.txt l.408) |
| dd8d2738 | actbake r36-3 -> x23 | `work/actbake-1110-r36-3` | first try; retry x23 landed in wave 2b actbake |
| aa4b2b29 | actbake r37-0 -> x24 | `work/actbake-1110-r37-0` | first try; retry x24 landed in wave 2a actbake |
| 29551ee6 | actbake r44-1 -> x27 | `work/actbake-1110-r44-1` | first try; retry x27 not kept (its build is HELD as refs/wrap/<sha>) |
| 77f14667 | actbake r47-3 -> x29 | `work/actbake-1110-r47-3` | first try; retry x29 landed in wave 2a actbake |
| 7ce3144d | actbake r51-1 -> x31 | `work/actbake-1110-r51-1` | first try; retry x31 not kept (its build is HELD as refs/wrap/<sha>) |
| fa1370d6 | actbake r53-2 -> x34 | `work/actbake-1110-r53-2` | first try; retry x34 landed in wave 2b actbake |
| dd3f14a1 | actbake r56-3 -> x37 | `work/actbake-1110-r56-3` | first try; retry x37 not kept (its build is HELD as refs/wrap/<sha>) |
| 0ba549d0 | actbake r58-3 -> x38 | `work/actbake-1110-r58-3` | first try; retry x38 landed in wave 2b actbake |
| d65d10ac | actbake r9-2 -> x4 | `work/actbake-1110-r9-2` | first try; retry x4 not kept (its build is HELD as refs/wrap/<sha>) |
| 5e971739 | cards r14-2 -> x8 | `work/cards-1110-r14-2` | x8 dropped by judges: done since: TC c35c13c0 + 055cdc12 (Changes head file) |
| 052723ca | cards r15-1 -> x9 | `work/cards-1110-r15-1` | first try; retry x9 landed in wave 2a cards |
| a900c5fc | cards r21-1 -> x16 | `work/cards-1110-r21-1` | first try; retry x16 landed in wave 2a cards |
| 14c603a2 | cards r23-1 -> x17 | `work/cards-1110-r23-1` | first try; retry x17 landed in wave 2a cards |
| c5227c5a | cards r26-0 -> x21 | `work/cards-1110-r26-0` | first try; retry x21 landed in wave 2a cards |
| 239c0997 | cards r34-0 -> x25 | `work/cards-1110-r34-0` | first try; retry x25 landed in wave 2a cards |
| 59ee8435 | cards r38-0 -> x26 | `work/cards-1110-r38-0` | first try; retry x26 not kept (its build is HELD as refs/wrap/<sha>) |
| 5131d7b6 | cards r48-2 -> x32 | `work/cards-1110-r48-2` | first try; retry x32 landed in wave 2a cards |
| fd3928a4 | cards r55-1 -> x40 | `work/cards-1110-r55-1` | first try; retry x40 landed in wave 2b cards |
| ee41991e | cards r55-2 -> x39 | `work/cards-1110-r55-2` | first try; retry x39 landed in wave 2b cards |
| f4f930e7 | cards r56-0 = x41 first try | `work/cards-1110-r56-0` | x41 retry landed (wave 2b cards, TC f769e0fc, same subject) |
| ada1ebff | cards ship 4 pre-state | `backup/cards-1110-ship4-pre` | pre-rebase backup of cards ship 4 (landed); 12/12 subjects on main |
| 0fa1a7ee | cards ship 6 pre-state | `backup/cards-1110-ship6-pre` | pre-rebase backup of cards ship 6 (landed); 12/12 subjects on main |
| be76fa9f | held3 pre-state | `backup/cards-1110-r3s4-pre` | pre-rebase backup of cards held3; held3 landed 11-19; 8/8 subjects on main |

### LANDED-REBASED (28)

| SHA | Item | Ref | Reason |
|---|---|---|---|
| b4dd2c97 | actbake r4 merge (detached r4-tc) | `refs/wrap/b4dd2c97` | 7/7 commit subjects on main (b8b1f6f4, 868dfba8, 78e43b0d, 139ad049, be2c7ec4, 475a3b9c, 23d54f9f) |
| 7d299557 | actbake r55-0 | `work/actbake-1110-r55-0` | verifier keep; rebuilt on r55-3 and landed in research4 ship 1 (11-19 14:22: AC 95f750f2, TC 7b37f7d9); its 'Re-pin waits 60 s' part never ported |
| 5ab7b532 | f11 | `work/cards-1110-f11` | landed in wave 2a cards (AC ab9c7349, TC 68961a8b) |
| 0ce2d127 | f5 | `work/actbake-1110-f5` | landed in wave 2a actbake (AC f0f4c7c1, TC 0911ce04) |
| d0e7a173 | f9 | `work/actbake-1110-f9` | landed in wave 2a actbake (AC f0f4c7c1, TC 0911ce04) |
| 00796967 | fork r2-3 TC sweep | `work/fork-1110-r2-3` | 'Cards: layer link lists' rebuilt and landed as TC 500f1826 (11-16, cardfix links); type-icon ppi (1/2, 2/2): 104 of 106 new blobs found on main |
| 0e54031d | r57-3 | `work/actbake-1110-r57-3` | landed in wave 2a actbake (AC f0f4c7c1, TC 0911ce04) |
| 41408a91 | r60-1 | `work/actbake-1110-r60-1` | landed in wave 2a actbake (AC f0f4c7c1, TC 0911ce04) |
| ecdbbc0d | x13 | `work/actbake-1110-x13` | landed in wave 2a actbake (AC f0f4c7c1, TC 0911ce04) |
| 5af2a5a1 | x14 | `work/actbake-1110-x14` | landed in wave 2a actbake (AC f0f4c7c1, TC 0911ce04) |
| 00676a06 | x16 | `work/cards-1110-x16` | landed in wave 2a cards (AC ab9c7349, TC 68961a8b) |
| e6a9cad3 | x17 | `work/cards-1110-x17` | landed in wave 2a cards (AC ab9c7349, TC 68961a8b) |
| 85255a38 | x18 | `work/actbake-1110-x18` | landed in wave 2a actbake (AC f0f4c7c1, TC 0911ce04) |
| d85d919a | x19 | `work/actbake-1110-x19` | landed in wave 2a actbake (AC f0f4c7c1, TC 0911ce04) |
| 9fbb6522 | x20 | `work/actbake-1110-x20` | landed in wave 2a actbake (AC f0f4c7c1, TC 0911ce04) |
| b55ec3b6 | x21 | `work/cards-1110-x21` | landed in wave 2a cards (AC ab9c7349, TC 68961a8b) |
| c4a532b4 | x22 | `work/actbake-1110-x22` | landed in wave 2a actbake (AC f0f4c7c1, TC 0911ce04) |
| ca54b0b7 | x23 | `work/actbake-1110-x23` | landed in wave 2b actbake (AC ea0db694, TC 0cb95c33) |
| 4e461a72 | x24 | `work/actbake-1110-x24` | landed in wave 2a actbake (AC f0f4c7c1, TC 0911ce04) |
| f487c8c2 | x25 | `work/cards-1110-x25` | landed in wave 2a cards (AC ab9c7349, TC 68961a8b) |
| c003a055 | x29 | `work/actbake-1110-x29` | landed in wave 2a actbake (AC f0f4c7c1, TC 0911ce04) |
| c183a58f | x32 | `work/cards-1110-x32` | landed in wave 2a cards (AC ab9c7349, TC 68961a8b) |
| e061bf49 | x34 | `work/actbake-1110-x34` | landed in wave 2b actbake (AC ea0db694, TC 0cb95c33) |
| 4183bd84 | x38 | `work/actbake-1110-x38` | landed in wave 2b actbake (AC ea0db694, TC 0cb95c33) |
| d1549ed6 | x39 | `work/cards-1110-x39` | landed in wave 2b cards (AC 0d2de792, TC f769e0fc) |
| 7ca7a3fa | x40 | `work/cards-1110-x40` | landed in wave 2b cards (AC 0d2de792, TC f769e0fc) |
| 551493fb | x41 | `work/cards-1110-x41` | landed in wave 2b cards (AC 0d2de792, TC f769e0fc) |
| 0de56fbc | x9 | `work/cards-1110-x9` | landed in wave 2a cards (AC ab9c7349, TC 68961a8b) |

## psd-tools fork

### STRANDED KEEP (4)

| SHA | Item | Ref in the bundle | Bundle | Reason |
|---|---|---|---|---|
| 0d663338 | Fable fork r1-0 | `work/fable-fork-r1-0` | fork-unlanded | CPU parity nudge (proven only on CPUs without AVX-512); owner 15:01: ship now, re-verified (held-1121) |
| f7ec40a0 | Fable fork r1-1 | `work/fable-fork-r1-1` | fork-unlanded | one cards CI workflow in place of upstream's four; owner 15:01: ship now, re-verified (held-1121) |
| bd0aa15b | Fable fork r2-0 | `work/fable-fork-r2-0` | fork-unlanded | history_pairs tool (tools/ and tests/ only); owner 15:01: ship now, re-verified (held-1121) |
| 3d14ac57 | Fable fork r2-1 | `work/fable-fork-r2-1` | fork-unlanded | Group.new pass-through and a same_picture tool; owner 15:01: ship now, re-verified (held-1121) |

### REJECTED (4)

| SHA | Item | Ref in the bundle | Bundle | Reason |
|---|---|---|---|---|
| fff52a96 | fork r12-1 Mask.offset | `work/r12-1` | fork-unlanded | not built (builder built=false: no gain against the Photoshop oracle); ROUNDS.md "not in this round's list" |
| d2a5b851 | fork r18-2 mask feather slabs | `work/r18-2` | fork-unlanded | fork verify keep=false: correct but gain too narrow (marginal no); other 2 unique commits = r15-2 + bevel (landed as 1584b529) |
| e4b7c760 | fork r3-0 ICC colour table | `work/r3-0` | fork-unlanded | fork verify keep=false: drops ICC metadata on the table path, changes PNG bytes AC writes |
| 18326922 | fork r5-3 compiled layer records | `work/r5-3` | fork-unlanded | fork verify keep=false as built: one small bug (channels-fix.patch ready); never rebuilt |

### SUPERSEDED (3)

| SHA | Item | Ref | Reason |
|---|---|---|---|
| 5f0de139 | fork r17-2 (dry) | `work/r17-2` | no commits of its own: = 5f0de139, an earlier tip of work/research-1110 (r15-2 merge) |
| 1d9950e6 | fork r20-2 (dry) | `work/r20-2` | no commits of its own: = 1d9950e6, research-1110 at round 18 (its fork changes released in cards-20261114 except r15-2) |
| be32812f | fork r21-r24 (dry) | `work/r21-0`, `work/r21-1` and 8 more | dry rounds 21-24 (fork lane went dry 11-15/16): no commits of their own, = work/research-1110 |

### LANDED-REBASED (1)

| SHA | Item | Ref | Reason |
|---|---|---|---|
| be32812f | fork r15-2 (lane branch) | `work/research-1110` | r15-2 landed rebased 11-21 15:41 as fork 82be96d2 (tag cards-20261121, AC pin dcf0b6f9); r17-3, r18-3, r20-1 earlier; branch deleted by the r15-2 run |

## Landed lane branches (17 + 2)

Ancestors of main; deleted at the wrap-up (11-21 12:35-12:56) or by their own run. AC: `work/actbake-1110` ea0db694,
`work/actbake-1110-r5-0`..`r5-2` d09ce6fa (r5 worktree diffs in `scratch/s1110/wrap/patches/`),
`work/actbake-1110-r60-2` ab5bc666, `work/actbake-1110-r60-1` 4149cf25, `work/cards-1110` 0d2de792,
`work/skilldocs-1110` 1a728b17, `work/timelines-1113` 8ccc8d7a, `work/wrap-1121` 8113496b, `work/pin-1121` dcf0b6f9
(r15-2 pin, 15:41). TC: `work/actbake-1110` 0cb95c33, `work/actbake-1110-r5-1`..`r5-3` 135aa1de, `work/cards-1110`
33c6ce3c, `work/timelines2-tc` c318e3fa, `work/ttstips-1111-proto` 6304101c. Fork: `work/fork-r15-2` 82be96d2 (fork
main, 15:41).

## Other refs and stores (kept by policy)

| What | Head | Why kept |
|---|---|---|
| Desktop AC clone `main` | e35327a4 | untouched by the cleanup (hard rule); the orchestrator fast-forwards it to main after the run (owner 14:40) and deletes the stray `caches_now.tsv` |
| AC `archive/ai-generation`, `archive/discord-edits`, `archive/local-update-stats` | ed21cd22, 52ed1fa9, e2835a7a | = GitHub; retired features live on archive branches |
| Desktop TC clone `main` | 0cb95c33 | blobless clone at its origin/main; its `origin/plates` (55beaaa6) is a stale tracking ref (GitHub 5e1e786f), harmless |
| psd-tools clone `main` | da069245 | = tag cards-20261111; the orchestrator fast-forwards it after the run (owner 14:40) |
| Fork tags `cards-20261020`..`cards-20261121` | 82be96d2 (newest) | release pins (AC pins cards-20261121 since dcf0b6f9); upstream's `v*`/`0.x`/`1.x` tags are not ours |
| CardBrowser `main`, tags v0.1.0, v0.1.1 | 6437f11c (= v0.1.1), 5d9f2bf1 (v0.1.0) | both quiet pre-releases (v0.1.1: 11-21 14:51, the new icon); source `~/Documents/Tools/wt/cardapp` |
| `~/Documents/Tools/history.git` `history` | 680dbfba | the timelines' picture store (= GitHub TC `history`) |
| `~/Documents/Tools/firstrepo.git` `refs/first/*` (5), `refs/lost2023/*` (94), `refs/old/*` | - | the pre-wipe record: never delete; may move to the owner's external SSD |
| `scratch/tc-history.git` | a7ed9744 | AC tests' HISTORY store (`tests/conftest.py`) |
| `scratch/tts/history.git`, `scratch/tts/autocroissant.git` | 0911ce04, f0f4c7c1 | TTS checks fixtures (`TTS_HISTORY`, `checks_env.py`) |
| GitHub AC | main dcf0b6f9 + the 3 archive branches | nothing stray, no tags |
| GitHub TC | main, data, history, plates, previews | no work/scratch branch, no tag |
| GitHub fork | main 82be96d2 + 25 `archive/*` + the cards-* tags | no work/scratch branch |

## Older scratch bundles

| Bundle | Head | Verdict |
|---|---|---|
| `scratch/s1110/t1-land/{ac,tc}-actions-1108-nover.bundle` | a07a0bd2 / 369559a3 | SUPERSEDED: same tips as `work/actions-1108-nover` in the 1121 bundles |
| `scratch/s1109/wrap/ac-actions-1108.bundle` | 3006c795 | LANDED-REBASED: all 8 commits on AC main (`git cherry`); kept as the plan says |
| `scratch/s1109/wrap/tc-actions-1108.bundle` | 2b8e1479 | LANDED-REBASED: its 4 subjects on TC main; kept as the plan says |
| `scratch/s1109/wrap/tc-unlanded-1110.bundle` (`work/links-1107`, `work/stoogedings-linux`) | 78ec3257 | UNKNOWN: the previous session's (11-07..10) record. Its tips are not in the Desktop TC; a fetch into a temp repo failed (blobless bases: 6 unresolved deltas), so it was not adjudicated. Kept as is |

## In flight at this writing (11-21 ~16:00)

Not adjudicated: each run removes its own branches and worktrees when it lands.

- held-1121: AC `work/h1121-*` with worktrees `~/Documents/Tools/wt/h1121-*` (TC copies there are its own sparse
  clones); scratch `s1110/h1121/`.
- timelines-apply: AC and TC `work/tl-1121`, worktrees `wt/tl-ac`, `wt/tl-tc`.
- favicon + header logo: TC `work/favicon-1121` (9b774c81), worktree `wt/favicon-tc`; AC `wt/favicon-ac2` (detached).
- This docs run: AC `work/cleanup2-1121`, worktree `wt/cleanup2-1121`.
