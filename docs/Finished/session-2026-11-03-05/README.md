# Session 2026-11-03 to 11-05: cost badges, the cooldown ring font, the card rollout

From the 11-01 wrap-up (AutoCroissant 548a34c, TTSCardMaker bf67cd8e) to the go-day pushes in the night of 11-04 to
11-05. Times are PST. With the owner ("ask me questions" again from 11-04 15:20) until his go (~23:00 11-04); the
batches ran unattended after it.

Heads when this was written (11-05 ~00:30): AutoCroissant main 3641756, pushed, and the jetson runs it (deployed 23:34;
psd-tools fork `cards-20261104` = 1258fcbd on both machines, the tag on GitHub). TTSCardMaker origin/main e71b3610:
batches 1 and 2 of 7 pushed, 3-7 still to come (`git log origin/main`). The data moves with each batch (`git
ls-remote origin data`). What is left: `docs/NEXT_SESSION_PLAN.md`.

## Landed

### 11-01 night to 11-03: cost badges, tags to 20

| What | Where | Notes |
|---|---|---|
| Names in any letter case | TC 1f661e7d, AC 20b7c2e | `EXACT_CASE` back to false (owner); guard R14 stays |
| Card makers | TC 21cf1b5c, c16c784e, 03119c90, 3cabfe12 | Rulebook: Minions; Defeat Update and Earth Totem's nerf (hdckighfkvhvgmk); Zomoroko published 91 queued ability edits, baked by Card sync in ab85122a, 4c03ef72, b9ab34c7 |
| Plain tags as badges | AC 4d49607, a77c719; TC a8ec8a1b | The guards set a timing tag typed in a plain font as a badge on push; a sweep-only tags fix; the one-word name rule as the site has it |
| Bake alignment | AC 0fad316 | Line edits are matched up by paragraphs, then words |
| Website | TC 682d0bc6, 3247ff32, cd918712, 82c7eded, a2b50d1f | Cost badges, mention and unpublished flags, the Rendered tab first, Trash all previews, star points badges; text_align in the bot checkouts |
| The 11-03 batch | TC 0a7db1c3 + f26dcd9c (drift baseline 0a7db1c3); AC aadaaca, af52f78 | Cost badges in stoogedings ((2 points) green coin, (2 bricks) brown brick, (2 SP) yellow star, souls and counters teal); tags to 20, a/b from -9; 62 cards swept (`Card-sweep: tags`); "star points" as SP; the table map in ch 04 (`TTS/table_map.py`); the TTS table mirrored, Bricks counters brown; no 30-minute waits. Rebuilt on the 91 baked edits: 21 overlapping cards swept again, the save merged |

### 11-04: speed, clean-up, rulebook pages, Actions

| What | Where | Notes |
|---|---|---|
| Speed pass | AC efc7808, f8b4301; TC 5fb0eb39, 2efb23e0 | Unchanged rulebook prints skipped; guards pooled from 2 cards; pushes without delta search (`pack.window=0`); Card sync waits for the push's guards; no PSDs in the artifact; a newer site build replaces an older one; card wheels cached per Python minor, the venv per patch; the bot's reloads parse once, tables first. Deployed efc7808 |
| Clean-up | AC b91a648, 694a150, 2752556; TC f38e2deb, 952053c7 | The recent-push grace deleted (`RECENT`, `recent_paths`, `due`, the quiet window); the parser's rule-page code deleted (764 cards parse the same) and the parts editor's rule-page path; /help one line per command. Deployed 2752556 |
| Rulebook | TC 4aaf54fe; AC 5b82a71, 8737edd | The website's look as before 11-03 (owner); the TTS pages cut from it at its proportions (45 pages), the editor's marks redrawn; the PDF printed in MDW/Fonts' Consolas; pages.json pinned with its PDF; the print key sees the inline CSS |
| Actions (research rounds 2 and 3) | TC c562c8af, 535071b2; AC 08930e5 | Bake records only from this repo's runs; the data push and dispatches retried; dry runs locked apart; per-job token scopes; site packages pinned; build-only data commits don't redeploy. Then: the guards fetch a push's card blobs in one fetch (P1), the site build dispatch tried 3 times (P2), artifact actions v7/v8 (P6), the Pages deploy rides out 15 status errors (wbd3-1) |
| Repo settings | GitHub | Outside PRs need approval; the default token is read-only |

### The cooldown ring

| What | Where | Notes |
|---|---|---|
| Ring on the website | TC 675e4bda | First a ring beside CD/GCD badges, website only (card text, rulebook, token picker, Stats, Balance); cards held |
| The final badge font | TC 0ade8c3d | Design B: the ring is the pill's round left end, its rim in the group colour, a segment per turn (the charge green clockwise, below zero red counter-clockwise, the rest hollow). Deep starts (more than a whole ring below zero): a thinner red band and up to 2 thin lap rings inside it. The number in the middle is green when ready (0 or 1), red below zero, grey otherwise. Labels and numbers in Roboto Black 900 (cost badges too); no glow; a one-colour version of each; the trigger orange unchanged. Website, mock-up and rulebook draw it the same; rulebook ability panels' badges centred. Skill: `ttscardmaker-companion` §1.2 |
| Guards for the new font | AC c850328, 83e136d, 498cbb5, 57604ea, bb8cd5f | Badge widths first (the 3 earlier icon fonts read once a push, judged by gain; each card edge checked on its own; split terms noted); R15 text off the card; a reflow whose last line falls below its box refused (AG5), told on a bake (AG2-1); re-layout notes in plan.json (`pixel_notes`, AG2-2); the old-font note A1 (`OLD_FONT`, "pull first") when a badge layer drawn with an older font is left; fork `cards-20261104` (tracking scales with the horizontal scale, as in Photoshop); tests pinned to the ring font (`RING = 0ade8c3d`) |

### Website research rounds 3 and 4 (TC, 20 commits)

- Round 3 (dfd06e3f, b1d6354b, c9697c28, f9c39530, f96134da, 89d33b2c, 4fe8d510, b7cd0d19; tests AC 082d37b): a shared
  card page draws the card before the scripts; share and rulebook pages preload their chunks; steady grid placeholders;
  sections open at their top; reloads keep the reader's place; phone rulebook, 16 px text fields, long links wrap; table
  pictures; touch screens hide the shortcut hints and "+ Deck"; rulebook green #006b0a; keyboard and screen-reader
  fixes; text typed before the scripts load is kept; sized feed pictures; huge commits link a card's own history; word
  diffs cached; load errors retry when back online; the badge font preloaded on card pages with badges.
- Round 4 (87d5263d, 8b35f8c0, 83c75804, d7fd9db5, 16668b33, af45d3a5, de71cf0e, 9e7b944d, 730380d5, 104edd2b,
  d2eaf956, 8367cec2): a hand holds one of each card (duplicate pictures, renamed ids); a broken hand link says so;
  another tab's hands and staged edits are kept; person pages' titles and Cards links; the phone card picker and the
  hand slot's ✕; a hand's cards open from the keyboard; card names link to `c/<id>/`; the Discord copy of a hand
  without the banner preview; quoted search finds timing tags and `[type]` tokens; the phone deck builder in 3 columns
  (up to 435 px).

### Bot (AC, research round 4, deployed 3641756)

bb8cd5f the old-font note says pull first; 9a40494 Discord bake lines without codes and pixel counts; f17c629
/query_ability finds every card with a quoted timing tag; 216b65a /query_rulebook puts chapters with the word at a
word start first ("ring" opens Timing Tags); 3641756 the /website reply in plain words, no 30-minute bake.

### The card sweep and batches

- Census at 0ade8c3d: 764 card PSDs; 390 PSDs with badges: 376 cards and 14 old rulebook pages in
  `MDW/Old Rules/Rulebook/` (left with the old badges by design).
- `card_sweep --kind badge-widths` fixed 334 and refused 42. 37 were fixed in Photoshop on copies (Metal General and
  Puzzled Cat as Photoshop versions too); the 5 left by the owner's rulings: Drawcia and NME Salesman worded shorter
  (their own commit, no sweep trailer, so Discord tells the wording), Michael's Blessing's table re-spaced, Traitor
  Lord's tracked space set to 0, Koro-Sensei from the card maker's own 10-20 save (MDW's Futura) with the guards'
  tags-as-badges fix. Two sweep fixes
  held for the owner: Future Niiue as made, Computer Virus in batch 1. Stubborn_Queen.png follows Stubborn King
  (batch 4).
- Local main, on 8367cec2: 9407d03a tags (Moose Goose, WX-78, Nefarious Chip), 206cd02e batch 1 (49 cards), af82957a
  2 (53), 759a9923 3 (53), 99daffca 4 (54 and Stubborn_Queen.png), d61df7e8 5 (53), 948a47fe the wording (2), f3d964ce
  6 (54), e43b7f3c 7 (55): 376 PSDs and 377 PNGs, at most 1.32 GB a commit (e43b7f3c 1,315,322,925 bytes; 99daffca
  1.30 GB, f3d964ce 1.29 GB). Each was checked once as built (guards dry run with `--repo`, card_check, font_check,
  the font blob: nothing to tell; `ring-sweep-final/batches/checks/`) and is re-parented onto origin/main at push
  time, so the landed hashes differ; the push re-runs only the guards' dry check against the new parent. Procedure:
  `ttscardmaker-companion` §1.10.

### Go day (the owner's go ~23:00 11-04)

1. The fork tag `cards-20261104` pushed.
2. AC 08930e5..3641756 pushed; the jetson deployed at 3641756 (23:34).
3. TC 535071b2..8367cec2 (the font and the 20 site commits) pushed 23:35. Card sync printed the rulebook PDF (74f2df0f)
   and re-pinned the save (24240ba6).
4. Batch 1 (the tags commit and batch 1: 52 cards, 1.23 GB) pushed 00:05 as 7d5782ca + 955e780d. Guards 6.5 min (all
   208 pushed blobs prefetched in about 3 min; the post job posted the allowed R14 lines for Sans and Mantis Lords; Card
   check and Font check had nothing to report on GitHub, where a push past ~40 PSDs can't be listed: the local runs
   were the checks), Card sync 7.3 min, data 36411905 ("52 art"), save and PDF 714b9791 + 7610ca42.
5. Batches 2-7 (the wording commit in batch 5's push) chained from 00:14 (`chain.sh`); batch 2 pushed 00:24 as
   e71b3610 (GitHub warned that Quote.psd is 69.4 MB). The rest: `git log origin/main`.

## Owner rulings

From memory `ttscardmaker-integration` (its 11-03 and 11-04 lines):

- Names link in any letter case (11-01); R14 stays.
- 11-03: the bake aligns edits by paragraphs, then words; the guards set plain timing tags as badges; the Rendered tab
  first; the cost badge colours above; tags to 20; "star points" as SP; no 30-minute waits.
- 11-04: the table map in ch 04. The ring over a battery or a pie, website first. Rulebook: the website's look as before
  0a7db1c3; TTS pages cut from it at its proportions, never shrunk; the editor's cut marks kept; the 14 removed page
  breaks stay out.
- The full ring rollout, cards included, design B. All badge labels in Roboto Black 900 (the [unknown token] fallback
  stays Bree Serif; Changa One's capital height). No glow anywhere. The number green only when ready (0 or 1), red
  below zero, light grey otherwise. Deep starts: C2 (a thinner red band and up to 2 thin lap rings).
- The full sweep and batches committed locally, nothing pushed until the go. Sweep refusals: try Photoshop on copies,
  timeboxed, no code bloat, no card made unlike the rest; whatever stays hard goes on a list. A1 yes (the one told
  pixel-fix refusal; reverses the 10-25 silence rule for this note only). Keep `~/Library/Fonts/stoogedings.ttf`
  identical to the final build.
- Website round 3: keep "+ Deck" hidden on touch, rulebook green #006b0a; no 24 first-screen tiles, opaque top bar or
  phone chart labels. Round 4: the website and bot items yes, the phone deck builder in 3 columns, Card check and Font
  check reading a big push from the clone (pushed after the last batch).
- The owner posts the card makers' font note himself.
- Cards: Drawcia "(1/2CD) Either: summon..." and NME Salesman "...can hit this)" (their own commit, no trailer);
  Michael's Blessing ring badges and a re-spaced table; Traitor Lord do it (the tracked space, +170, was the cause);
  Future Niiue as is; Computer Virus in batch 1 (its NO STATS FOUND post allowed once); research a Futura-Bold swap, no
  change without a go.
- Outside PRs need approval; the default token is read-only.
- The go: push everything when finished: fork tag, AC, jetson deploy, TC font and website (not waiting for the note),
  batches as each push's runs settle (no fixed 30 minutes; watch the API budget), then big-push, the Types/ drop, the
  completeness check and the whole-tree guards dry run.

## Measurements

- Font: 17,416,116 bytes (11,372,244 with the cost badges, 5,721,356 on 10-21); 1,449 badge glyphs (1,139 timing tags,
  1,036 of them cooldowns; 310 costs), each with a one-colour version; built in about 34 s. (1CD) is 2,658 units wide
  (2,040 before).
- Room: 66 of 377 multi-line badge layers had under one line to spare; 138 of 648 ability layers have no room for
  another line.
- Pushes are upload-bound: batch 1 (1,228,532,202 bytes, 1.23 GB) took about 22 minutes from its local check to
  landing, batches 2 and 3 (1.27 GB each) about 6 (`go/b2/`, `go/b3/` push.log). Batch 1's runs: guards 6.5 min, Card
  sync 7.3 min.
- GitHub's compare API gives up (422) past about 40 PSDs (43 and 50 failed, 29 worked).
- API: about 230-350 REST calls a card push against GITHUB_TOKEN's 1,000 an hour; P1's prefetch saves about 165 on a
  55-card batch (guards ~225 -> ~60).
- Futura: MDW's Futura-Bold.otf and macOS's Futura.ttc differ in 95 of 95 glyphs, macOS's 2.8% wider ("o" 700 against
  662 units). On Koro-Sensei the guards' layout is 69% off its pixels with macOS's Futura, 89% with Jost, 90% with
  League Spartan, and fits with MDW's (IoU 0.998): no swap helps. Rockwell Bold (95 of 95, 8.3% wider) and Lucida Grande
  Bold (17 of 95) differ from macOS's too. 480 card PSDs were last saved on a Mac, 284 on Windows. Files:
  `~/Documents/Tools/wt/scratch/futura-research/`.
- Website round 3 (research measurements, headless Chrome): a shared card shows ~0.42 s sooner on fast 4G (LCP 968 ->
  544 ms) and ~0.94 s on slow 4G; rulebook links ~0.44 s sooner on slow 4G; a phone's first screen ~0.42 s sooner on
  slow 4G.
- Tests: 689 passed (full suite, about 3.5 min, 6 workers); work/big-push 689 passed, 1 skipped.
- The TTSCardMaker clone holds 7.09 GiB of loose objects (gc.auto 0) and 47.2 GiB of packs in 82 (11-05 01:00; 35.4
  GiB before the batches): from batch 2 on, each batch's `git fetch` wrote a ~6.3 GB promisor pack holding the last
  pushed batch's card folders whole, most of it already in other packs (git's `index-pack --promisor` repacks the local
  objects a fetched pack points to; the plan's Clean up). The disk had 22 GiB free at 01:00; scratch is 52 GB.

## Lessons

Stories in `autocroissant-failure-archaeology`.

- A review workflow dropped an owner pick (the lap arcs) without asking. Owner picks are never dropped without asking;
  a review may only flag them.
- On the partial clone, `git write-tree` needs `--missing-ok` when the index holds blobs only the remote has (Card
  sync's).
- zsh arrays start at 1: run array loops under `bash -c`.
- Batch pushes are upload-bound (6-22 minutes per 1.2-1.3 GB: batch 1 22, batches 2 and 3 about 6): plan a night by
  that.
- The local guards check needs `--repo` to read the earlier icon fonts as the runner does (Traitor Lord passed
  without it and failed with it).
- Photoshop on this Mac drew Koro-Sensei with macOS's Futura until MDW's Futura-Bold was installed. Now installed, with
  Lucida Grande Bold, Rockwell Bold, Elephant Italic, Myriad Arabic and Roboto Black. A Photoshop fix is only as good as
  the fonts it draws with.
- On 11-03 Zomoroko published 91 edits in the middle of a batch: the batch was rebuilt on top and the 21 overlapping
  cards swept again (`card_sweep moved` finds them).
- On 11-04 morning GitHub refused the Mac's SSH keys for a while; the HTTPS recipe is in memory `commit-to-main`.

## Left open

The rest of the batches, the go-day leftovers, owner decisions and the research rounds' follow-ups:
`docs/NEXT_SESSION_PLAN.md`.
