# Card timelines from Discord

**Status:** stage 2 in progress (2026-11-13; 11-14: the first repo found on GitHub, its commits picture rows): mined, old repo indexed, matcher run over every
post, fields read, chains and Created built, review page built; a local preview built (`discord_preview`, the
website's `HISTORY_DIR`). Nothing published (preview first, owner 11-13).

## Goal

Every card gets a timeline of its versions from its first appearance in the team's Discord up to its first git commit, each
version read off the card itself (name, stars, types, stats, ability text) with as much metadata as can be found, and shown in
the card's Image versions on the website before the git versions. Cards that never reached git get timelines too.

## The owner's rulings (2026-11-08)

- **Public:** Discord-era versions show on the website like git versions, published on an orphan `history` branch of
  TTSCardMaker. No credit on them (no poster, no author line). Message text itself is never shown; raw messages stay off every
  repo (10-22 ruling), only the pictures and the fields read from them are published.
- **Images:** the orphan `history` branch (like `previews` and `data`): never on main, so clones stay small.
- **Review:** exact PSD reads and confident picture matches publish; low-confidence matches and conflicting dates go to a review
  page for the owner (like 10-22's `ambiguous_short.html`).
- **Text-only changes:** a message or ruling that describes a change with no picture ("nerfed Sans ATK to 4") becomes a dated
  text entry, linked to the version it changes when there is one; unsure ones go to review.
- **Lost cards:** cards scrapped, merged or renamed before git get timelines, listed apart from live cards (like The Morgue); a
  rename is joined to its live card.
- **Fields:** everything: name, stars, types, stats and ability text (OCR, icons matched to `Types/`), so versions diff like git
  ones; unsure readings are marked.
- **Stop at git:** nothing from Discord after a card's first git record.

## Physical card photos (Q6, final: owner 2026-11-14 22:15 and 22:20)

The 16 JPGs of the 'Old stuff' thread (#record-book, thread 1331913819563360288, 2025-01-23) are the only physical card
photos; nothing else is one. They are in no timeline, review item or Morgue evidence and are never cropped: the
matcher never reads the thread (`discord_match.LEFT_OUT`, as ai-images and patty-vault). Created = "Before 2020" (no
day, sorting first, source 'physical card photo') for exactly Pep Drink and Wham Bam Bone (`discord_chain.PHOTO_CREATED`;
their earliest appearance kept as `dated`; the review page shows the item read only) and M'uru, a Morgue candidate
(`PHOTO_MORGUE`, LC-1c11be0504, slug m-uru: its entry's first); no other Created comes from the photos. The ids are
fixed: an id that no longer names its card (a rename, a regrouped Morgue entry) is a WARNING in the chain's stats and
the Morgue's counts (`photo_missing`), so a lapsed "Before 2020" never goes unseen.

## Pipeline

0. **The first repo** (2022-08-24..2023-04-21): the same GitHub repo's history before the old .git, cut by the
   force-pushes of 2023-04-02 and 2023-04-21, still served by GitHub by commit id. Fetched read only (blobless, then its
   card PNGs in one batched fetch) into a bare repo outside every repo (`~/Documents/Tools/firstrepo.git`; its
   alternates read the old .git's objects, so one store holds both; its remote URL disabled: no lazy fetch; one path, `discord_match.FIRST_GIT`, env `TIMELINES_FIRST_GIT` after a
   move: a step that reads it stops with an error when its `refs/first` (or the lost 2023 refs) are missing, never
   the old .git alone).
   `scripts/oldgit_index.py` indexes its main history before the old repo's as one history (its rows: repo 'first';
   'Initial No PSD's' of 2023-04-02 has the tree before it as its parent; the side branch the_restoration_project,
   2020 port looks re-committed in 2022, is left out), its creatures under `Cards/`; a first-repo card joins the old
   card its last file name became (the folder decides between two of one name); authors by authors.json, else by a
   shared commit e-mail. Its card PSDs are fetched too (11-15, one batched read-only fetch: 1,133 blobs): its rows
   read their fields from the PSD of the same commit (else the nearest) checked against the PNG, as the old repo's
   do, a disagreement to review (stars a PSD of another look draws stay in review even when the folder agrees: the
   picture may be older, Nosk prints 3 where PSD and folder say 4; a buff's arrow, at the top right on 2022-23
   cards, is a header icon, and so is a type's picture one row lower, its middle in the header: Nailsage Sly's
   tornado under the title (art there is not); a type twice that today's card has once is one icon drawn as two layers, to review; an
   item's read type no header icon shows is art, to review; a repeated layer no Types/ picture names is art beside a
   type icon or the stars drawn as many times, else to review; '%%' reads '%': a font with no % glyph); a row whose PSD is
   missing or unread reads its fields off the picture (its
   stats with no PSD are sure only by eye, by the era bar readers' calibrated read of the PNG, or when a reader agrees
   with the same look's sure Discord read: its two readers share a bias on the 2020-22 templates; a Discord eye read
   its own sure era read contradicts is not used, as a one-cell bar change keeps the hash; a creature's HP 0 read off
   a picture is never sure; an old repo PNG with no PSD stats is sure by its two readers only when its era read is
   sure and the same on every stat: static-noise bars fool all three; its ability, with no PSD text, is the text
   readers' read of the PNG, `timeline_text read-ability`: sure when it prints one of the card's own texts, today's or
   its PSDs' (its words and their form: a '+3' for '3' or a misread bracket goes to review), method 'ocr+known' with
   that text as written, or at its era's stored bar, never fitted again, else proposed for review; a git look's
   Discord picture takes an answer given on its row's item or on its own, and an answer typed off the picture keeps
   its form over a known text of the same words (text_key); a field its row has none of is the picture's own sure read;
   its types, with no PSD, by eye (`derived/readers/types/eye_rows.json`, by PNG blob: the eras A-C, items and fields
   read off zoomed headers with each era's icon bank; an item's kind icons are no type), else a git version's when
   its header matches that version's, `header_types rows` on D1-D3 as for Discord pictures, else proposed; a name
   sure only as the file name its PSD's composite carries goes to review when another version of the same file
   prints a longer title holding it (the title layers unread: Satou's 'Satou Matsuzaka'), and an empty ability is
   never sure with no OCR of the picture).
   Its rows carry the badge "first repo" (owner Q5, 11-14: picture rows, no text-only rows; its push notices in
   github-updates are evidence only: Created hints); each card's timeline stops at its first first-repo record.
   Its card files of no live card (and its `MDW/` files) are evidence that a Morgue candidate of that name was played
   (`lost.json` `first_repo`, never a candidate of their own; second-picture files like `Grapes2` join their card, never
   one in a Meme folder: a joke on it; a Meme folder's file joins a file of its name only in its own top folder: Alex
   YIIK's Cards/Meme -> Cards/Other, while the joke item Items/Meme/pain.png and the debuff Pain are two old cards). A
   lost first-repo or old-repo file, or one only unsure evidence maps (the debuff Pain -> Thinned, its picture alone), whose picture is a live card's card-sure
   Discord picture joins it by eye (`derived/oldgit/eye.json` `ties`: Blooper -> Gooper Blooper, its file under an old
   name; or a 'second picture', an older look kept as a numbered file), so it is never a Morgue candidate too. An old
   card of PSDs alone the owner gave a card (an owner map, or the page's 'yes' on its unsure map: 'sumo outside',
   Items/Freeze_Spell.psd) gives that card its PSD: the card's PSD of its commit when its composite is that commit's
   PNG and the PNG's own PSD is another look (Freeze Spell 2023-04-11), else a psd_only row of its composite; an unsure
   map's rows wait in its item (`replaces`: the row it re-reads) and a 'yes' with no rows is counted apart, never as
   applied. The
   owner's map rulings (`derived/review/owner.json` `maps`, read only on the review page) come over all the evidence:
   sure, how 'owner', the old name shown on the rows, never an alias (11-15: Eldman -> Elmadan, its first file under a
   misspelt title, the same PSD document; Dark_Matter.png of Cards/Kirby/4 Stars -> Dark Nebula; Fire_Bomb.png of
   Auxillary/Items/4 Stars -> Flame Bomb; Humphreys Belly, Gasoline, Magolor and Hemorrhaging -> their live cards).
   A map change moves a card's first record (`fs`: `discord_match score`'s references), so its stop and Created
   follow only after the rerun: map, `oldgit_rows rows`, `discord_match score`, `discord_verdicts local
   --before-git`, then step 2's verdicts on (verdicts, Morgue, facts, chain, Morgue, chain, review page), the preview
   build, its check and the site.
   **The lost 2023 history** (`refs/lost2023/*` in the same store, fetched by id from GitHub's activity log): the old
   repo's own main heads pushed 2023-10-26..11-05 and lost by the force-pushes of 11-03 and 11-05 (104 commits in no
   other local repo). `oldgit_index` reads them as the old repo (rows 'old', badge "old repo", `lost: true`): 78
   commits continue its main 268c2b1; then an unrelated history whose first commits '1'..'8' re-upload a43fa582's
   tree in batches (`LOST_LINKS`: '1' follows a43fa582; `UPLOADS`: a file not re-uploaded yet is carried, never a
   deletion and a new card), on to c76646e2 just before the wipe. Old card ids never change for it (a card's id is
   its last name before the lost history; a card first seen there is named after the others), and a card the lost
   history deleted keeps its path evidence from the old repo's own end (`at_own_end`: The Claw's first file). A
   card's chain then stops at its earliest git record of the three repos (a first-repo PSD before any PNG: 'first
   repo', as Created labels it).
1. **Mine** (`commands/discord_archive.py`, restored from AC history at `070820f^`, with its tests): GET-only REST reads with Waxwell
   (the Mac's bot token, not production; owner 11-13), resumable, into `~/Documents/Tools/discord-archive/` (never a repo). Messages and
   attachments (images, PSDs) of every channel the bot can read, threads included. The 10-22 run reached 8.5 GB with images:
   check free disk first. `--files <json>` stores named non-image attachments of a few messages (documents, Paint.NET
   files, zips), each message read again for a fresh URL. `commands/discord_documents.py` turns the card documents into
   pictures dated by their post (a docx's page 1 drawn by QuickLook in `~/std-work-1107`, cut into its card cells as
   `derived/documents/boxes.json` says; a .pdn's own composite), posts the matcher reads like any other (kind
   'document').
2. **Match** (`commands/discord_match.py`, restored 2026-11-13): each picture to a card by perceptual hashes of the card and
   its art against every version (`refs`: today's cards from the data branch, every git version of their PNGs in
   TTSCardMaker's history and the old pre-wipe repo's versions from `derived/oldgit`, else TTSCardMaker's alone), OCR'd title
   (macOS Vision), OCR'd text against the ability, file name, chat text and aliases; sheets cut into cells (ties in
   name ratio or candidate score go by card id, never by a set's hash order: a rerun gives the same cards). Each card's chain
   stops at its first git record (the old repo's, else TTSCardMaker's). The morgue channels (meme, joke and scrapped cards)
   feed lost-card candidates (`lost.json`), never live cards, as do card-shaped posts naming no card. Outputs under the
   archive's `derived/matcher/`. Then `commands/discord_verdicts.py` gives each picture (a sheet's cells each) one
   verdict: version sure, version unsure, draft (a look not in git), lost or morgue (else unsure, art, none), its
   earliest post dating it (never an embed's: embeds were fetched when archived, so they may show a later picture); lost
   pictures grouped into lost cards; contact sheets for the checks by eye. A git look is claimed only when the picture's
   timing tags and numbers are that version's; made before the look's first git version, its text exactly that
   version's at any length, whatever picked the look (no text read: unsure); more than 3 days before, its bars read
   (`derived/readers/bars`) the same too (the hash can't see a bar or a typo; a card whose data stats are all -1
   prints none; the git versions' bars: the old repo's PNGs and items and the card repo's stored PNGs, `era_bars
   versions`). A look the gates can't tell is a draft when no git version can be it: an older template (eras A-C
   and their items) than every git version's, 8+ bits from all; a whole card export (not a cut screenshot) printing
   a number (1-2 digits) or timing tag no read of any git version has, over 3 bits from all; a status card's star
   count no git version has; a whole card export (the 1200 px and 400 px template sizes) 8+ bits from every git
   version (Discord copies of git versions stay under 7.5). The last three need no git version unstored before the
   post. The selftest's held-out check holds: no copy of a git version becomes a draft. A local gate comes before
   all of these (`discord_verdicts local`: each card-sure picture against every stored version of its card, eyekeys'
   score, the two nearest scored again after an affine alignment, cached in `derived/match/local.jsonl`; only
   versions with the picture's aspect ratio within 1% count): 5 or less from a version is that version's look; a
   gate-made look over 10 from that look's stored versions isn't told (one area drawn otherwise: the eye decides);
   30 or more from every stored version (45 under 800 px wide), none made before the post left out and one of the
   look the matcher picked compared, is a draft.
   (`--before-git`: only pictures posted before their card's first git record, the timelines' part; one left with a
   version uncompared is never a local draft.) `discord_verdicts localtest` sets those bars on every stored version's Discord (800 px) and small (360 px)
   copies: none that far from every version, none at 5 from another look.
   Pictures told by eye (`derived/match/eye.json`: card and look; gap round 1's status pairs and port pictures, gap
   round 2's pictures on a card's Created day, single-candidate pictures before git and near-git looks of the cards
   with no Discord entry; `titled`: a renamed first look's printed title) override the gates. The owner's rulings
   given outside the review page (`derived/review/owner.json`, kept apart from the eye's files; 11-13: the 2020
   'PIZZA' item is Pepperoni Pizza's draft, the 2020 item 'Frozen in time' Standstill's, Wind's Blessing's five
   pictures Second Wind's, which set no Created: `no_created`; `ruled_out`: cards a ruling rules out of an untold
   picture's maybe list, e.g. the creature Pizza for a 2023 Pepperoni Pizza look) come after the eye (D8, 11-15:
   the 2020-08-24 flying-trees 'Gust of Wind' is Gust of Trees' first version: its row shows the title it prints, an
   eye truth, never an alias; the first repo's Gust_of_Wind4.png, the same picture, is Gust of Trees' second picture).
   A version key is taken only when the picture is that version by eye:
   `discord_verdicts eyekeys` puts each beside the version's whole blob (its thumb when the clone lacks it) with the
   3 most different areas zoomed; a picture drawn otherwise in any way (a word, the title's spelling or colour, an
   icon, a medal, the text's layout or outline, the art) is a draft.
   A card named only (its title, file name or chat; nothing of its look or text) is sure only when the picture's text
   fits it (version fit 0.25; a title read alone, weight 0.93, goes to this check on a whole card export) and no
   card the matcher saw in that text fits better (a reused name: 2019's joke Potato,
   Blooper Tentacle's and Flame Bomb's first versions under Tentacle's and Firebomb's names); one dated before the card's
   first picture evidence sets Created only through review (the chain lets one dated on or after the file's Created
   day in: it can't move Created). A joke picture that is a live card's (named so by its posts
   outside meme channels, or a git version's look and text) goes to review, never the Morgue candidates.
   Reposts link only pictures of one card's one look. A card-sure draft first posted in a card channel (port,
   update-NN-port, card-classification, ideas-discussion, new-cards) before any meme repost of it (its own or a
   twin's) is the card's look, not a joke picture (`shown_first`; never a picture the first repo filed only in a Meme
   folder, the same bytes, or near one while the eye hasn't told it; the pictures behind open owner questions stay as
   built). Outputs under `derived/match/`.
   Then `commands/discord_morgue.py` sorts the lost cards and the old repo's into Morgue candidates (joke cards and
   cards really played: the owner's answers Q1, Q2), ideas never ported (review only) and the rest (already in
   MDW/The Morgue or MDW/Unfinished, an old name of a live card, a split, no card), each checked by eye
   (`derived/morgue/eye.json`; the owner's entries in `owner.json` over it: Pananda, a played creature, 11-13);
   outputs `derived/morgue/candidates.json`, `ideas.json` and webp thumbs. A candidate's
   picture that is the same picture (whole and art within 3 bits) as a row given to a live card or the unsure review
   stays only when the eye takes that row (listed in `taken`: the review page drops it there) or rules it another
   picture; else the candidate goes to review. A kin pass lists the rows no entry holds that may still be a lost card's
   (a none, art or unsure row titled like a candidate or idea, nearer than any live card's name; a none row that is a
   joke picture, or a played channel's with card evidence) until the eye adds each to an entry, makes an entry of it
   (keyed by its picture id: Wickerbottom's five books, Warcraftian Kel) or rules it not a lost card's; `kin_open`
   must be empty. The kin pass also lists the rows that may be a card already in MDW/The Morgue or MDW/Unfinished
   (the whole within 5 bits of its PNG or the art within 3, a title like its name, the same picture as such an
   entry's), so those pictures leave the live cards' review too (`taken`); an entry holding an MDW card's PNG but
   ruled another card is listed in `mdw_clash`, which must be empty. Renamed and live entries' first dates are
   `created_hints` for Created (weak links only by the owner; `meme`: dated only by meme channels, so never a live
   card's Created). M'uru's first is "Before 2020" (source 'physical card photo', sorting first; its earliest post
   kept as `dated`; Q6). Candidates go only to the website's Morgue list (cards.json morgue),
   never cards[], decks, Unused, the default search or Mentioned by.
3. **Read:** a posted PSD through `PSDParser` (exact: `commands/discord_psds.py`, its name, ability and types, stats
   off its composite; stars left out, 2022 PSDs keep hidden star layers; an ability kept in layers named after their
   text is sure only when the composite prints its words in that order, none left out: a moved label or an overflowed
   word goes to review); the chain puts its fields on the look posted in its own message, else on the nearest look
   within 3 bits that no PSD read otherwise holds (a field another PSD set is never overwritten), or adds it as a
   version dated by its post. Card-sure pictures
   the card channels' pass missed are bar-read too (`era_bars add`: era from the text readers, sure by the
   calibration as it stands). The text readers take the bar readers' era for a picture no rule of theirs fits
   ('other'), read again with it, the bars not fitted again; a field card (the bar readers' era F, 2021-12..2023-05)
   is read in a text era of its own (every paragraph under the art, calibrated on eye truths), and a 1200 px export
   whose labels sit off the template's ('bars-other') by its size as the alt template (D3) or the item template
   (D-item). A card-sure sheet
   cell is a picture of its own to both readers (cut by its box; its key in their outputs ends '#c<n>'). A
   two-creature card whose two halves' bars are told (eye truth: stats, stats2) publishes both, never summed; a split
   card's halves (owner Q3) each take their own rows (Twin Princes: Lorian's orange top rows, Lothric's blue bottom
   ones, as their cards' bars today; the old repo's first split files of 2023-10-16 read the other way, which the
   review page's split item says). On the template (D1-D3) a picture's types are its git version's when the header's icon
   band matches that version's (`commands/header_types.py`: NCC per icon cell, the bar calibrated on old repo PNGs'
   400 px JPEGs against the card's versions with other icons; method 'git header'; icons the version's PSD puts
   below the band, under a two-line title, are compared in their own boxes, and a version with types whose icon
   boxes aren't known never matches; a repost whose header matches no git version keeps them off the look). On eras
   A-C (no git version prints them) one picture per card and era is typed by eye (`derived/readers/types/eye.json`)
   and carried to the card's other pictures of that era by the same score (`header_types eye`, method 'eye header';
   the bar: the highest pair with other types, the eye's checks and other cards' references, + 0.02). Entries whose
   types no reader made sure are typed by eye from their header beside the icons' pictures (`derived/readers/text/eye/
   val_gap1_types.json`, gap round 1: [] where no icon is printed; an item kind's icon is no type; a held-out file, never
   fitted). Picked by-eye bar
   readings (a truth file with `"calibrate": false`, gap round 5's reads in review) are their pictures' truth and never
   fit a bar. Item and
   status eras count stars as yellow blobs in a row that fill their box as stars do, calibrated on eye truths (a
   yellow wall or backdrop: no count; the truths publish as 'eye'. On pictures with no truth the readers alone are
   sure on few of the status and B-item cards: 11-13, S 4 of 97, B-item 8 of 94, C-item 0 of 3, A-item 174 of
   228). A picture by era: today's template through the card checks' bar reader
   (`card_checks.read_picture`); older templates (digit rows, earlier bars) need their own readers, calibrated on the git versions
   of each era; ability text by OCR with stoogedings icons matched to `Types/` and badges to the font's badge set (an item's or
   status card's ability: the bottom block of evenly spaced lines, a short last line too; each Discord era sure at
   the 0.81 OCR floor only on 18-22 eye truths of its own, 11-13: A-item, B-item, C-item, S at 0.81; the eras
   whose abilities print icons and styled digits (D2, D3, D-item, F) pool their eye bar apart, 1.0: their 0.81 reads
   lose an icon or a digit); stars and types
   by icon match (a count is never sure on shuriken stars, with a copy on a title letter, or with a star cut by the
   picture's right edge: `scripts/timeline_text.py edges`, nor a blob count under its strong copies, a star in a
   second row: Spice 2023-04-10; a count of 1 never in A, C, B-item and F, whose copies
   read overlapped or unlike stars as one; a star bar with under 8 truths at or above it is none: F, D3. B-item's
   overlapped clusters, 93 eye truths: no bar, its stars publish only as 'eye'). Every field keeps its method (psd, picture, ocr, text) and a
   confidence.
4. **Chain** (`commands/discord_chain.py build`, outputs under `derived/chain/`; the owner's answers of 11-13 and 11-14, Q1-Q6):
   per live card, the Discord looks posted before its first git record (a git look, or a draft; a later git look posted
   early is the git part's; an older template posted after a newer one goes to review). Reposts of one look merge, the
   earliest post dating it: a draft joins the nearest look (whole-card hash) with the same numbers and timing tags
   (counted: a swapped digit is a new version) whose every picture prints the same words (the same class line where
   both read one, the abilities' OCR words alike, no real word on one picture missing from the other, a word with
   letters added at one end counting as missing: a reworded ability or a renamed class line is a new version) and
   isn't alike but for one area (an icon, stars, a bar or a word edited: a grid cell far apart); a field a
   picture didn't read (a sheet's cell, unread bars) is never filled from its repost, only proposed, and a repost's own
   contrary star or stat read takes a reader's sure value to review. An icon-read star count that changes with nothing
   to back it (no eye or PSD read, no second read apart, no changelog fact, the versions around it say otherwise) goes
   to review, as do icon-read counts in the run of versions just before a sure changelog line that changes the stars
   to that count (none of them eye or PSD, the last over 3 days before the line). Then the sure changelog facts before it (Q4; item kinds
   never), the old repo's rows (the first repo's among them, badge "first repo"), a split card's versions "as <old name>" on
   both halves (Q3); never a physical card photo (Q6). Only card- and look-sure pictures enter
   (never one the Morgue's eye took for a joke or lost card: `taken`); a stat-less card's picture reading no bar
   label has no stats field (not printed);
   each field keeps its method and confidence and only sure values stay (a git look's own fields first, the readers'
   eye truth, then their sure reads; a title read snapped to another live card's name, not one of this card's own
   names, aliases or old repo titles, goes to review: P.A.N.D.A. Mk A read 'Panda'), the rest name a review item
   (types with nothing read: unread). An item card's kinds (its subtypes) are never kept: read alone, sure or not,
   they are not collected (no item; a sure read compares as no types; so is another card's version read sure as
   kinds alone, an item then: Rock Bottom's 2023 equip); its other types stay, or go to review, without them. Each version is
   diffed with the one before on fields sure on both sides (the website's buff/nerf rule: the stat total, a total over
   40 not judged; formatting-only text is no change; an item's kinds, its subtypes, are no types change). A Discord
   look folds under "minor changes" only when every field is sure and unchanged AND its picture is near the one
   before (owner 11-09, as the old rows fold by png_changes' share): the same git look, or under 5% of the card
   changed, both trimmed and aligned (ECC, affine) at 160 px wide, a pixel changed when a channel is off by more than
   32 levels inside a 3 px square after a light blur (JPEG noise, resizing and a few pixels of crop aside; Discord
   copies can't take png_changes' 2-level rule); pictures of another shape, or none read, never fold. A fact a later picture contradicts goes to review (a stated step,
   'ATK +1', on the value before it: the last sure picture's moved by the kept facts since; checked again when a fact
   that bounded its window goes), and a fact whose move the pictures around it show (the last sure value before, the
   next after, no other fact on that field between) is confirmed by that picture (counted once). The changelog facts
   (`commands/discord_facts.py`) also read the 2019 balance-change lists posted as pictures (general-discussion
   2019-03-31 and 2019-05-14: OCR'd, each 'Name (Buff/Nerf/Rework)' entry a line `<message>:img<n>`, no text kept; a
   head that is no card's name, or one the eye doubts, goes to review) and name cards by the old names the timelines
   settled too (the Morgue's renamed entries, eye.json `titled`, owner.json `old_names`; 'Frozen in time' meant
   Standstill until Standstill's first picture) and a line's short form told by eye (`derived/facts/eye_lines.json`
   `lines`: 'Maxwell', 'PoD', the Warcraftian class prefixes): evidence only, never aliases. A line naming no sure
   card ('ambiguous card', owner D9) takes the card it is given on the review page (a card picker, or none: left out),
   the owner's `owner.json` `lines` over it (2019 'Lightning' = Lightning Bolt, 'Frozen in Time' = Standstill, 2023
   'Dark Matter' = Dark Nebula; 'Computer virus 2.0' = the Morgue's Computer Virus 2.0, `morgue:<entry id>`: its
   marks go to that entry, never to the live Computer Virus). The 2019 February balance
   list (a docx, `CHANGELOG_DOCUMENTS`) is read like the pictures' lists (`<message>:doc<n>`, no text kept). HP/DEF units come from the card's
   sure bar reads near the line (git looks and drafts, the old repo's rows). **Created:**
   the earliest confident appearance (a card-sure picture, the first repo, the old repo, the card repo; "Before
   2020" from a physical card photo for Pep Drink and Wham Bam Bone only, Q6, sorting before every day; an exact head of a 2019 balance list whose
   card's first picture keeps that name by eye (`eye_lines.json` `heads`, source 'changelog'); a PSD the old repo added before any PNG of the card, 'old
   repo (PSD)', the first repo's 'first repo' (psd): `derived/oldrows/psd_first.json`; its composite, when a look of its own, is the card's first old repo
   row, `psd_only`, and its stop: Tainted Lazarus 2023-09-10; pictures the owner kept weak, `no_created`, enter but
   date nothing) against `site/Card Creation Dates.json`: same, earlier
   (proposed), later (review only, never moved later), new, hand (untouched); named-only pictures, changelog lines and
   first-repo push notices before it are review only. `discord_chain checks` draws random cards' whole timelines for the by-eye
   check.
5. **Review page** (`python -m commands.timeline_review build` -> the archive's `derived/review/index.html` + `items.json`;
   local, never published): the owner's answers Q1-Q6 and the later ones (`owner.json` answers; read only; Morgue
   items the owner answered show read only, listed by id, the joke edits of live cards each by eye; an open item a
   later answer bears on says so), the unsure old maps,
   unsure matches and versions (a picture's card, its look, named-only pictures confirmed in bulk, joke pictures, older
   templates, unsure fields, changelog lines), the old repo's PSD-vs-picture fields (an item card's kinds
   never asked), Created dates (earlier,
   new, later, and earlier hints: only those earlier than the Created date in use), the Morgue list (candidates, ideas
   never ported, the rest) and splits; stable ids, the archive's thumbnails, radio answers kept in the browser and
   exported as JSON. An id keeps its meaning: `derived/review/ids.json` lists every id the page has shown with the card
   it was first shown for, and a picture's item (look not told, named only, joke picture) asked about another card
   gets a new id; any item `owner.json` `read_only` lists shows the owner's answer, not asked.
   `scripts/apply_rulings.py <export>` checks every answer against `items.json` (one error stores nothing; a typed card
   is its id or a name only one card has; listed and skipped, never an error: an answer whose item left the page at a
   rebuild, one now read only, and, from an older page's export (its `page`, or `--page` for a file made without
   one), one carrying a value the item no longer offers), keeps them in `derived/review/rulings.json` with the file's
   source (owner > triage > triage guess: a lower one never replaces or clears a higher; a field so given is method
   'owner' or 'triage', an item card's kinds stripped from given types as from read ones), reruns the facts when a
   line was given a card, the chain (which honours them: method 'owner' for a field given, status 'owner' for a Created picked, with `owner.source`
   shaped like `ours`: where the date came from; the ids it acted on in `derived/chain/ruled.json`), writes the Morgue
   list with the answers (`derived/review/morgue.json`) and rebuilds the page with the answers checked. An answer whose
   item left the page shows as taken in (in `ruled.json`) or as not applied (its item changed: answer it again). A git
   look is answered by its first version's key, not its number (numbers shift when a version is added), and a
   confirming answer carries what it confirms (a field's 'yes' its value, a Created pick or hint its day, a picture's
   or a map's answer its card): an answer no radio takes any more (that version gone, a rerun proposing another value,
   day or card) is not applied either. The APPLY ORDER (11-15 ~21:50): the owner's page exports, the BRIEF owner lines
   (owner.json), the triage's two-judge answers, then its guesses.
6. **Publish** (`commands/discord_preview.py build`, the preview first, owner 11-13): `derived/preview/history/` in the
   `history` branch's layout: `index.json` (each card's folder; the Morgue's entries from Discord, owner Q1/Q2; the MDW
   Morgue cards' Discord pictures; the Created dates the chain proposes, earlier or new only, "Before 2020" from a
   physical card photo, Q6), `<card id>/timeline.json` (entries oldest first: discord, text, old repo; only sure fields,
   each with its method and confidence; no message text, posters, ids or links) and one webp per version, at most 720 px
   wide (the website's card view), `_morgue/<slug>/` the same (the Morgue list with the review page's answers: a
   candidate answered 'no' off it, an idea or the rest answered 'yes' on it; an entry's changelog marks, data only; its
   old and first repo versions as 'old repo'/'first repo' entries: an MDW Morgue card's rows by its name, a lost
   old-repo card's rows, the first repo's files linked by name or by eye, `derived/morgue/eye.json` `first_files`:
   Wickerbottom's Book1..5; a picture row when the file has no rows: date, mapped author, subject). Stop at git holds here too: a card's entries end at its
   first record in the card repo with its PNG's moves followed as the website follows them (a move that redrew the
   card, Elemental World's 4 -> 3 Stars), and its last diff (`git_first`) is dated when its card-data version was
   committed, approximate when the card had versions before that one (no rename when the last look is that first
   git version and its title read is in the printed title of every version from it up to the card-data version's
   date: the data's name is its PSD layers', The Bagel's 'Or... is it'; Grape Juice, retitled in between, keeps
   its rename); an MDW Morgue card's Discord pictures end at its
   first record in the old repo (oldgit `map.json`) or the card repo. A confirmed changelog fact names its version
   (`confirmed_by`); a title read snapped to an old PSD's casing takes today's name when the picture's own OCR spells it
   so. The eye's rulings (`derived/preview/eye.json`) drop a wrong field and leave out a
   picture that is no version of the card (OMORI's 16 AI restyles; the next version is diffed again, the review page
   doesn't ask about it). The `history` branch itself (one commit per run, a script run by hand, no Action) waits for the owner's OK.
7. **Website** (TTSCardMaker `site/build/build_site.py` `HISTORY_DIR`; unset, the site builds as before): Image versions
   lists the versions before the repo after its own (old repo: date, mapped author, commit subject, badge "old repo",
   the first repo's "first repo"; Discord: date and channel), each compared with the current card
   as today (its own sure ability as "Text then"), never with a link or Revert, and never at or after the card's first
   version there; their changes and the changelog facts (data only, owner Q4; a type fact's "remove" a removal; a fact
   its confirming version also records counts once, as that version's) join Recorded changes, the trend and the balance charts; the Morgue lists the
   joke and lost cards from Discord (picture only, never in cards[]; never on a live or renamed card's page: `-morgue`;
   first posts as Pacific days; a played one with a file in an older repo, `git_file`, "lost before this repo", not
   "before git"), and an MDW Morgue card's Discord pictures follow its PNG's newest version (its current
   one, no Revert); the proposed Created dates apply (never later, never over hand; "Before 2020", a physical card
   photo's, counts in a "Before 2020" year of its own; M'uru's on its Morgue page).

## Verification

- `discord_match selftest` (degraded copies of every PNG and version) before trusting a new era reader; a reader's accuracy on
  that era's git versions reported per field.
- Every published version's fields re-read from its webp agree with timeline.json (`discord_preview check`: the
  timelines' own readers on the webp; what they can't read again is counted apart, a disagreement goes to the eye, and
  so does every version published with no types: an absent icon can't be read again, a recoloured one matching no
  `Types/` picture; and every types list the eye gave (method 'eye...': the A-C eye reference and its header copies),
  whose left-out icon nothing reads: 19 of 421 left one out, now drops). It runs after every preview build (`--redo` reads only versions published otherwise since, in a
  seeded random order, each read kept as it goes: a run cut short goes on where it stopped and has read a fair sample),
  and its new disagreements are seen by eye before a publish (a wrong field: a drop in `derived/preview/eye.json`).
- The website build and its checks pass with and without the `history` branch.

## Open questions (for the start of the task)

- Which channels to mine (all readable, or the card channels first), and whether image downloads have a size cap.
- Whether lost cards' pictures show in the card browser or only on their timeline list.
