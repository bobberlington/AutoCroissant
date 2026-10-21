from os import environ
from zoneinfo import ZoneInfo


LOCAL_DIR_LOC   = environ.get("AUTOCROISSANT_LOCAL_DIR", "~/Desktop/TTSCardMaker")   # the card repo checkout (env: TTSCardMaker's tts-save Action)
ALIAS_PKL       = "aliases.pkl"
STATS_PKL       = "stats.pkl"
SITE_URL        = "https://michaeljsr.github.io/TTSCardMaker"   # card browser website
OLD_STATS_PKL   = "old_stats.pkl"
REMIND_PKL      = "reminder.pkl"
HISTORY_JSON    = "repo_history.json"          # repo path -> commit that last changed it (derived, gitignored)
HISTORY_CLONE   = environ.get("AUTOCROISSANT_HISTORY_CLONE", "~/Documents/TTSCardMaker-history.git")   # bare blobless clone of the card repo (env: the Action)
EMOJI_JSON      = "type_emojis.json"           # bot application id -> {type emoji name: icon blob} (gitignored)
TTS_SAVE_PATH   = "TTS/Impossibility Simulator.json"   # the TTS save in the card repo, updated by /update_tts_save
TIMEZONE        = ZoneInfo("America/Los_Angeles")

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
RULES_PATH      = "TTS/decks.json"             # deck rules for the TTS save
