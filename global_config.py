from os import environ
from zoneinfo import ZoneInfo


LOCAL_DIR_LOC   = environ.get("AUTOCROISSANT_LOCAL_DIR", "~/Desktop/TTSCardMaker")   # the card repo checkout (env: TTSCardMaker's tts-save Action)
ALIAS_PKL       = "aliases.pkl"
STATS_PKL       = "stats.pkl"
SITE_URL        = "https://michaeljsr.github.io/TTSCardMaker"   # card browser website
OLD_STATS_PKL   = "old_stats.pkl"
REMIND_PKL      = "reminder.pkl"
DETAILS_JSON    = "psd_details.json"          # PSD facts per card blob for the website's Details dropdown (committed, pushed with the pickles)
HISTORY_JSON    = "repo_history.json"          # repo path -> commit that last changed it (derived, gitignored)
HISTORY_CLONE   = environ.get("AUTOCROISSANT_HISTORY_CLONE", "~/Documents/TTSCardMaker-history.git")   # bare blobless clone of the card repo (env: the Action)
BOT_CLONE       = environ.get("AUTOCROISSANT_BOT_CLONE", "")   # this bot's own full clone, for its PICKLE commits ("": the checkout holding this file)
EMOJI_JSON      = "type_emojis.json"           # bot application id -> {type emoji name: icon blob} (gitignored)
FONTS_CACHE     = "fonts_cache"                # MDW/Fonts' font files, refreshed by tree sha, where there is no checkout (font_check; gitignored)
FONT_DIRS_ENV   = "PSD_TOOLS_FONT_DIRS"        # the fonts dir override (first entry), pathsep-separated: card_bake, font_check
TTS_SAVE_PATH   = "TTS/Impossibility Simulator.json"   # the TTS save in the card repo, updated by /update_tts_save
TIMEZONE        = ZoneInfo("America/Los_Angeles")

# The cutover switch (docs/Campaigns/DATA_BUILD_ACTION.md step 8, on since 2026-10-24): the card data comes from the Card
# sync Action's data branch (commands/card_sync.py) and /update_stats syncs it; False builds the pickles here (rollback).
CARD_SYNC       = True
DATA_BRANCH     = "data"                       # the card repo's branch holding db/ (written by Card sync only)
CARD_DATA_DIR   = "card_data"                  # the bot's copy of that db/ (gitignored)

# The card repo (TTSCardMaker) and where things live in it. /set_repo overrides CARD_REPO at runtime (card_repo.repository).
CARD_REPO       = "MichaelJSr/TTSCardMaker"
CARD_BRANCH     = "main"
API_BASE        = "https://api.github.com/repos/"
RAW_BASE        = "https://raw.githubusercontent.com/"
TYPES_DIR       = "Types"                      # type icons (PNG stems are the type names)
STARS_DIR       = "Types/Stars"                # star icons
RULEBOOK_DIR    = "Rulebook"                   # rule pages: any card path containing it
ICON_MANIFEST_PATH = "MDW/Fonts/stoogedings-src/stoogedings.json"   # icon aliases + codepoints, written by the font build
# The website's hand-kept files live in the card repo's site/ (moved from MDW/ on 2026-10-20); readers take the first path present.
AUTHORS_PATHS   = ("site/Card Authors.txt",)   # who made each card; wins over the database
EDITS_PATHS     = ("site/Card Edits.json",)    # website fixes; win over the PSD
AUTHORS_PATH    = AUTHORS_PATHS[0]
EDITS_PATH      = EDITS_PATHS[0]
CREATED_PATHS   = ("site/Card Creation Dates.json",)   # when each card was made (python -m commands.discord_edits created)
CREATED_PATH    = CREATED_PATHS[0]
RATINGS_PATHS   = ("site/Card Ratings.json",)   # card ratings 1-10 (python -m commands.tts_ratings; the website adds entries); the bot never reads it
RATINGS_PATH    = RATINGS_PATHS[0]
MAILMAP_PATH    = ".mailmap"                   # git's: one person's several git names as one (%aN; the API path: card_edits.Mailmap)
RULES_PATH      = "TTS/decks.json"             # deck rules for the TTS save
