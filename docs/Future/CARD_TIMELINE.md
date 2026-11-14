# Card timelines from Discord

**Status:** stage 2 in progress (2026-11-13): mined, old repo indexed, matcher run over every post, fields read, chains
and Created built, review page built; a local preview built (`discord_preview`, the website's `HISTORY_DIR`).
Nothing published (preview first, owner 11-13).

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

## Pipeline

1. **Mine** (`commands/discord_archive.py`, restored from AC history at `070820f^`, with its tests): GET-only REST reads with Waxwell
   (the Mac's bot token, not production; owner 11-13), resumable, into `~/Documents/Tools/discord-archive/` (never a repo). Messages and
   attachments (images, PSDs) of every channel the bot can read, threads included. The 10-22 run reached 8.5 GB with images:
   check free disk first.
2. **Match** (`commands/discord_match.py`, restored 2026-11-13): each picture to a card by perceptual hashes of the card and
   its art against every version (`refs`: today's cards from the data branch, every git version of their PNGs in
   TTSCardMaker's history and the old pre-wipe repo's versions from `derived/oldgit`, else TTSCardMaker's alone), OCR'd title
   (macOS Vision), OCR'd text against the ability, file name, chat text and aliases; sheets cut into cells. Each card's chain
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
   post. The selftest's held-out check holds: no copy of a git version becomes a draft.
   Pictures told by eye (`derived/match/eye.json`: card and look; gap round 1's status pairs and port pictures, gap
   round 2's pictures on a card's Created day, single-candidate pictures before git and near-git looks of the cards
   with no Discord entry) override the gates. A version key is taken only when the picture is that version by eye:
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
   Reposts link only pictures of one card's one look. Outputs under `derived/match/`.
   Then `commands/discord_morgue.py` sorts the lost cards and the old repo's into Morgue candidates (joke cards and
   cards really played: the owner's answers Q1, Q2), ideas never ported (review only) and the rest (already in
   MDW/The Morgue or MDW/Unfinished, an old name of a live card, a split, no card), each checked by eye
   (`derived/morgue/eye.json`); outputs `derived/morgue/candidates.json`, `ideas.json` and webp thumbs. A candidate's
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
   card's Created). A paper card's photo (owner Q6) is dated "Before 2020" and comes first; a later photo of a
   printed card (eye `not_paper`) has no day. Candidates go only to the website's Morgue list (cards.json morgue),
   never cards[], decks, Unused, the default search or Mentioned by.
3. **Read:** a posted PSD through `PSDParser` (exact: `commands/discord_psds.py`, its name, ability and types, stats
   off its composite; stars left out, 2022 PSDs keep hidden star layers; an ability kept in layers named after their
   text is sure only when the composite prints its words in that order, none left out: a moved label or an overflowed
   word goes to review); the chain puts its fields on the look posted in its own message, else on the nearest look
   within 3 bits that no PSD read otherwise holds (a field another PSD set is never overwritten), or adds it as a
   version dated by its post. Card-sure pictures
   the card channels' pass missed are bar-read too (`era_bars add`: era from the text readers, sure by the
   calibration as it stands). The text readers take the bar readers' era for a picture no rule of theirs fits
   ('other'), read again with it, the bars not fitted again. On the template (D1-D3) a picture's types are its git version's when the header's icon
   band matches that version's (`commands/header_types.py`: NCC per icon cell, the bar calibrated on old repo PNGs'
   400 px JPEGs against the card's versions with other icons; method 'git header'; icons the version's PSD puts
   below the band, under a two-line title, are compared in their own boxes, and a version with types whose icon
   boxes aren't known never matches; a repost whose header matches no git version keeps them off the look). On eras
   A-C (no git version prints them) one picture per card and era is typed by eye (`derived/readers/types/eye.json`)
   and carried to the card's other pictures of that era by the same score (`header_types eye`, method 'eye header';
   the bar: the highest pair with other types, the eye's checks and other cards' references, + 0.02). Item and
   status eras count stars as yellow blobs in a row that fill their box as stars do, calibrated on eye truths (a
   yellow wall or backdrop: no count; the truths publish as 'eye'. On pictures with no truth the readers alone are
   sure on few of the status and B-item cards: 11-13, S 4 of 97, B-item 8 of 94, C-item 0 of 3, A-item 174 of
   228). A picture by era: today's template through the card checks' bar reader
   (`card_checks.read_picture`); older templates (digit rows, earlier bars) need their own readers, calibrated on the git versions
   of each era; ability text by OCR with stoogedings icons matched to `Types/` and badges to the font's badge set; stars and types
   by icon match (a count is never sure on shuriken stars, with a copy on a title letter, or with a star cut by the
   picture's right edge: `scripts/timeline_text.py edges`). Every field keeps its method (psd, picture, ocr, text) and a
   confidence.
4. **Chain** (`commands/discord_chain.py build`, outputs under `derived/chain/`; the owner's answers of 11-13, Q1-Q6):
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
   to review. Then the sure changelog facts before it (Q4; item kinds
   never), the first repo's push notices naming the card alone as text-only rows (Q5: date, mapped author, commit
   message, badge "first repo"; no picture or link), the old repo's rows, a split card's versions "as <old name>" on
   both halves (Q3), and first a paper card's photo named as the card (by the matcher or by eye,
   `derived/readers/paper/eye.json`), dated "Before 2020" with no day (Q6). Only card- and look-sure pictures enter
   (never one the Morgue's eye took for a joke or lost card: `taken`); a stat-less card's picture reading no bar
   label has no stats field (not printed);
   each field keeps its method and confidence and only sure values stay (a git look's own fields first, the readers'
   eye truth, then their sure reads; a title read snapped to another live card's name, not one of this card's own
   names, aliases or old repo titles, goes to review: P.A.N.D.A. Mk A read 'Panda'), the rest name a review item
   (types with nothing read: unread). An item card's kinds (its subtypes) are never kept: read alone, sure or not,
   they are not collected (no item; a sure read compares as no types; so is another card's version read sure as
   kinds alone, an item then: Rock Bottom's 2023 equip); its other types stay, or go to review, without them. Each version is
   diffed with the one before on fields sure on both sides (the website's buff/nerf rule: the stat total, a total over
   40 not judged; formatting-only text is no change; an item's kinds, its subtypes, are no types change); a fact a later picture contradicts goes to review (a stated step,
   'ATK +1', on the value before it: the last sure picture's moved by the kept facts since; checked again when a fact
   that bounded its window goes), and a fact whose move the pictures around it show (the last sure value before, the
   next after, no other fact on that field between) is confirmed by that picture (counted once). **Created:**
   the earliest confident appearance (a named paper photo: "Before 2020", source paper card, sorting before every day;
   else a card-sure picture, the old repo, the card repo) against `site/Card Creation Dates.json`: same, earlier
   (proposed), later (review only, never moved later), new, hand (untouched); named-only pictures, changelog lines and
   first-repo rows before it are review only. `discord_chain checks` draws random cards' whole timelines for the by-eye
   check.
5. **Review page** (`python -m commands.timeline_review build` -> the archive's `derived/review/index.html` + `items.json`;
   local, never published): the owner's answers Q1-Q6 (read only, Q5's first-repo rows listed), the unsure old maps,
   unsure matches and versions (a picture's card, its look, named-only pictures confirmed in bulk, joke pictures, older
   templates, paper photos, unsure fields, changelog lines), the old repo's PSD-vs-picture fields (an item card's kinds
   never asked), Created dates (earlier,
   new, later, and earlier hints: only those earlier than the Created date in use), the Morgue list (candidates, ideas
   never ported, the rest) and splits; stable ids, the archive's thumbnails, radio answers kept in the browser and
   exported as JSON. `scripts/apply_rulings.py <export>` checks every answer against `items.json` (one error stores
   nothing; a typed card is its id or a name only one card has), keeps them in `derived/review/rulings.json`, reruns the
   chain (which honours them: method 'owner' for a field given, status 'owner' for a Created picked, with `owner.source`
   shaped like `ours`: where the date came from; the ids it acted on in `derived/chain/ruled.json`), writes the Morgue
   list with the answers (`derived/review/morgue.json`) and rebuilds the page with the answers checked. An answer whose
   item left the page shows as taken in (in `ruled.json`) or as not applied (its item changed: answer it again). A git
   look is answered by its first version's key, not its number (numbers shift when a version is added), and a
   confirming answer carries what it confirms (a field's 'yes' its value, a Created pick or hint its day, a picture's
   or a map's answer its card): an answer no radio takes any more (that version gone, a rerun proposing another value,
   day or card) is not applied either.
6. **Publish** (`commands/discord_preview.py build`, the preview first, owner 11-13): `derived/preview/history/` in the
   `history` branch's layout: `index.json` (each card's folder; the Morgue's entries from Discord, owner Q1/Q2; the MDW
   Morgue cards' Discord pictures; the Created dates the chain proposes, earlier or new only, "Before 2020" for a paper
   card), `<card id>/timeline.json` (entries oldest first: paper, discord, first repo, text, old repo; only sure fields,
   each with its method and confidence; no message text, posters, ids or links) and one webp per version, at most 720 px
   wide (the website's card view), `_morgue/<slug>/` the same. Stop at git holds here too: a card's entries end at its
   first record in the card repo with its PNG's moves followed as the website follows them (a move that redrew the
   card, Elemental World's 4 -> 3 Stars), and its last diff (`git_first`) is dated when its card-data version was
   committed, approximate when the card had versions before that one (no rename when the last look is that first
   git version and its title read is in the printed title of every version from it up to the card-data version's
   date: the data's name is its PSD layers', The Bagel's 'Or... is it'; Grape Juice, retitled in between, keeps
   its rename); an MDW Morgue card's Discord pictures end at its
   first record in the old repo (oldgit `map.json`) or the card repo. A confirmed changelog fact names its version
   (`confirmed_by`); a title read snapped to an old PSD's casing takes today's name when the picture's own OCR spells it
   so. The eye's rulings (`derived/preview/eye.json`) drop a wrong field, caption a covered photo, and leave out a
   picture that is no version of the card (OMORI's 16 AI restyles; the next version is diffed again, the review page
   doesn't ask about it). The `history` branch itself (one commit per run, a script run by hand, no Action) waits for the owner's OK.
7. **Website** (TTSCardMaker `site/build/build_site.py` `HISTORY_DIR`; unset, the site builds as before): Image versions
   lists the versions before the repo after its own (old repo: date, mapped author, commit subject, badge "old repo";
   Discord: date and channel; a paper card "Before 2020"; first repo: text only), each compared with the current card
   as today (its own sure ability as "Text then"), never with a link or Revert, and never at or after the card's first
   version there; their changes and the changelog facts (data only, owner Q4; a type fact's "remove" a removal; a fact
   its confirming version also records counts once, as that version's) join Recorded changes, the trend and the balance charts; the Morgue lists the
   joke and lost cards from Discord (picture only, never in cards[]; never on a live or renamed card's page: `-morgue`;
   first posts as Pacific days), and an MDW Morgue card's Discord pictures follow its PNG's newest version (its current
   one, no Revert); the proposed Created dates apply (never later, never over hand; "Before 2020" counts in
   a "Before 2020" year of its own).

## Verification

- `discord_match selftest` (degraded copies of every PNG and version) before trusting a new era reader; a reader's accuracy on
  that era's git versions reported per field.
- Every published version's fields re-read from its webp agree with timeline.json (`discord_preview check`: the
  timelines' own readers on the webp; what they can't read again is counted apart, a disagreement goes to the eye, and
  so does every version published with no types: an absent icon can't be read again, a recoloured one matching no
  `Types/` picture).
- The website build and its checks pass with and without the `history` branch.

## Open questions (for the start of the task)

- Which channels to mine (all readable, or the card channels first), and whether image downloads have a size cap.
- Whether lost cards' pictures show in the card browser or only on their timeline list.
