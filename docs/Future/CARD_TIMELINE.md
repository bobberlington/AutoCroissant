# Card timelines from Discord

**Status:** stage 2 in progress (2026-11-13): mined, old repo indexed, matcher run over every post, fields read, chains
and Created built; review page and a local preview next. Nothing published (preview first, owner 11-13).

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
   (`derived/readers/bars`) the same too (the hash can't see a bar or a typo).
   A card named only (its title, file name or chat; nothing of its look or text) is sure only when the picture's text
   fits it (version fit 0.25) and no card the matcher saw in that text fits better (a reused name: 2019's joke Potato,
   Blooper Tentacle's and Flame Bomb's first versions under Tentacle's and Firebomb's names); one dated before the card's
   first picture evidence sets Created only through review. A joke picture that is a live card's (named so by its posts
   outside meme channels, or a git version's look and text) goes to review, never the Morgue candidates.
   Reposts link only pictures of one card's one look. Outputs under `derived/match/`.
3. **Read:** a posted PSD through `PSDParser` (exact). A picture by era: today's template through the card checks' bar reader
   (`card_checks.read_picture`); older templates (digit rows, earlier bars) need their own readers, calibrated on the git versions
   of each era; ability text by OCR with stoogedings icons matched to `Types/` and badges to the font's badge set; stars and types
   by icon match. Every field keeps its method (psd, picture, ocr, text) and a confidence.
4. **Chain** (`commands/discord_chain.py build`, outputs under `derived/chain/`): per live card, the Discord looks posted
   before its first git record (a git look, or a draft; reposts of one look merged, the earliest post dating it; a later
   git look posted early is the git part's; an older template posted after a newer one goes to review), the sure
   changelog facts before it (Q4=B; item kinds never), the old repo's rows, a paper photo first and undated (Q6=A), a
   split card's versions "as <old name>" on both halves (Q3=A). Only card- and look-sure pictures enter; each field keeps
   its method and confidence and only sure values stay (a git look's own fields first, the readers' eye truth, then
   their sure reads), the rest name a review item. Each version is diffed with the one before on fields sure on both
   sides (the website's buff/nerf rule: the stat total, a total over 40 not judged; formatting-only text is no change);
   a fact a later picture contradicts goes to review. **Created:** the earliest confident appearance (a card-sure
   picture, the old repo, the card repo) against `site/Card Creation Dates.json`: same, earlier (proposed), later
   (review only, never moved later), new, hand (untouched); named-only pictures and changelog lines before it are
   review only. `discord_chain checks` draws random cards' whole timelines for the by-eye check.
5. **Review page:** the unsure matches, readings and dates, with stable ids and the owner's answers applied by a rerun (as
   10-22's `apply_rulings.py`).
6. **Publish:** the `history` branch holds per card `history/<card id>/timeline.json` (versions: date, fields, method and
   confidence per field, image name, text entries) and one webp per version (the card's PNG size class), plus an index. Written
   by a script run by hand (no Action), one commit per run.
7. **Website:** Image versions reads the index and shows Discord-era versions first, marked "from Discord" with their date and,
   for text entries, the described change; stat history and the balance charts take the reconstructed stats; lost cards get a
   list of their own.

## Verification

- `discord_match selftest` (degraded copies of every PNG and version) before trusting a new era reader; a reader's accuracy on
  that era's git versions reported per field.
- Every published version's fields re-read from its webp agree with timeline.json.
- The website build and its checks pass with and without the `history` branch.

## Open questions (for the start of the task)

- Which channels to mine (all readable, or the card channels first), and whether image downloads have a size cap.
- Whether lost cards' pictures show in the card browser or only on their timeline list.
