# Impossibility Simulator rulebook: verbatim extract

Extracted 2026-10-20 from `TTSCardMaker/site/dist/data/rules.json` (built locally that day). Checks:
- rules.json has 48 entries, and `Auxiliary/Rulebook/` has 48 PSDs with 48 PNGs, one-to-one.
- AutoCroissant `stats.pkl` holds the same 48 `rulebook` entries. Their `ability` strings are byte-identical to rules.json (compared that day).
- Every page PNG was looked at. Every PSD was scanned with psd-tools for hidden text layers.

How to read this file:
- Each page is `## <rules.json name>`, its PSD path, then the page text in a `text` block, verbatim from rules.json. The only changes are the bracketed markers below.
- `[icon: X]` is an icon drawn on the page where the text has a gap or lost glyph. X is the `TTSCardMaker/Types/` file stem, or a description if it is not a type icon.
- `[hidden ... omitted]`, `[off-canvas ... omitted]` and `[overflow, not shown on page: ...]` mark text that is in the PSD (and so in rules.json, stats.pkl and /query_rulebook answers) but that a reader of the page never sees. Hidden blocks that differ from the visible text are reproduced under the page for reference.
- Trailing ` [type] [type]` tokens after a page number are the bot's leftover-icon append (type injection, see the impossibility-cards-reference skill §4). They are removed from the text and listed under "Icons:".
- Stray digits in the text (e.g. `1 2 3`, `16 2 4 6`) are callout numbers or counter values from the page's pictures. The "Diagram:" note explains them.
- The last line of most pages is the printed page number (`1`, `4.2`, `11.6`, ...). Pages 12.3-12.5 are printed "123", "124", "125".
- Page order is book order: the two cover images, then table of contents, then 01 to 22. rules.json itself lists `00 Table of Contents` before the covers. The book's own numbering (Table of Contents) is 1. Getting Started ... 22. Negation. Several chapters span more pages (04 x5, 06 x2, 07 x5, 10 x2, 11 x7, 12 x5, 14 x2, 16 x2, 19 x2).
- Old rulebook versions are in `TTSCardMaker/MDW/Old Rules/` (22 PSDs, history only; today's pages win). They are cited here only where they explain a leftover in today's text.
