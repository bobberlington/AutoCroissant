from zoneinfo import ZoneInfo


LOCAL_DIR_LOC   = "~/Desktop/TTSCardMaker"
ALIAS_PKL       = "aliases.pkl"
STATS_PKL       = "stats.pkl"
SITE_URL        = "https://michaeljsr.github.io/TTSCardMaker"   # card browser website
OLD_STATS_PKL   = "old_stats.pkl"
REMIND_PKL      = "reminder.pkl"
HISTORY_JSON    = "repo_history.json"          # repo path -> commit that last changed it (derived, gitignored)
HISTORY_CLONE   = "~/Documents/TTSCardMaker-history.git"   # bare blobless clone of the card repo
EMOJI_JSON      = "type_emojis.json"           # type emoji name -> icon blob it was made from (gitignored)
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
AUTHORS_PATH    = "MDW/Card Authors.txt"       # who made each card; wins over the database
EDITS_PATH      = "MDW/Card Edits.json"        # website fixes; win over the PSD
RULES_PATH      = "TTS/decks.json"             # deck rules for the TTS save
