# Card timelines from Discord

**Status:** planned with the owner, 2026-11-08; not started. Mining waits until the task starts (owner).

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

1. **Mine** (`commands/discord_archive.py`, restored from AC history at `070820f^`, with its tests): GET-only REST reads beside the
   production bot on the same token, resumable, into `~/Documents/Tools/discord-archive/` (never a repo). Messages and
   attachments (images, PSDs) of every channel the bot can read, threads included. The 10-22 run reached 8.5 GB with images:
   check free disk first.
2. **Match** (`commands/discord_match.py`, same commit): each picture to a card by perceptual hashes of the card and its art
   against every git version, OCR'd title (macOS Vision), OCR'd text against the ability, file name, chat text and aliases; sheets
   cut into cells. New: a match against no live card starts a lost card (grouped by title and art).
3. **Read:** a posted PSD through `PSDParser` (exact). A picture by era: today's template through the card checks' bar reader
   (`card_checks.read_picture`); older templates (digit rows, earlier bars) need their own readers, calibrated on the git versions
   of each era; ability text by OCR with stoogedings icons matched to `Types/` and badges to the font's badge set; stars and types
   by icon match. Every field keeps its method (psd, picture, ocr, text) and a confidence.
4. **Chain:** reposts of one look merged (first appearance dates it); versions ordered by date and diffed with the one before
   (buff/nerf counting as the 10-23 rule); text-only entries attached; the chain ends at the card's first git commit.
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
