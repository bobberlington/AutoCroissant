from os import environ
from zoneinfo import ZoneInfo


LOCAL_DIR_LOC   = environ.get("AUTOCROISSANT_LOCAL_DIR", "~/Desktop/TTSCardMaker")   # the card repo checkout (env: the Card sync Action)
SITE_URL        = "https://michaeljsr.github.io/TTSCardMaker"   # card browser website
REMIND_JSON     = "reminder.json"               # reminders (commands/analytics.py; gitignored, per machine)
HISTORY_JSON    = "repo_history.json"          # repo path -> commit that last changed it (derived, gitignored)
HISTORY_CLONE   = environ.get("AUTOCROISSANT_HISTORY_CLONE", "~/Documents/TTSCardMaker-history.git")   # bare blobless clone of the card repo (env: the Action)
EMOJI_JSON      = "type_emojis.json"           # bot application id -> {type emoji name: icon blob} (gitignored)
FONT_DIRS_ENV   = "PSD_TOOLS_FONT_DIRS"        # the fonts dir override (first entry), pathsep-separated: card_bake, font_check
TTS_SAVE_PATH   = "TTS/Impossibility Simulator.json"   # the TTS save in the card repo, updated by Card sync
TIMEZONE        = ZoneInfo("America/Los_Angeles")

# The card data: the Card sync Action's data branch (commands/card_sync.py; the only source since 2026-10-25)
DATA_BRANCH     = "data"                       # the card repo's branch holding db/ (written by Card sync only)
CARD_DATA_DIR   = "card_data"                  # the bot's copy of that db/ (gitignored)
CARD_TREE_JSON  = "card_tree.json"             # the card repo's last tree, its ETag and Card Edits.json (gitignored)

# The card repo (TTSCardMaker) and where things live in it.
CARD_REPO       = "MichaelJSr/TTSCardMaker"
CARD_BRANCH     = "main"
API_BASE        = "https://api.github.com/repos/"
RAW_BASE        = "https://raw.githubusercontent.com/"
TYPES_DIR       = "Types"                      # type icons (PNG stems are the type names)
STARS_DIR       = "Types/Stars"                # star icons
RULEBOOK_DIR    = "Rulebook"                   # the old Photoshop rule pages' folder (under MDW: card_data keeps their credits)
RULEBOOK_SRC    = "site/rulebook"              # the rulebook's chapters (NN-slug.md), written on the website
RULEBOOK_PAGES  = "TTS/Rulebook"               # their TTS pages (PNGs and pages.json: python -m commands.rulebook_pages)
ICON_MANIFEST_PATH = "MDW/Fonts/stoogedings-src/stoogedings.json"   # icon aliases + codepoints, written by the font build
# The website's hand-kept files in the card repo's site/
AUTHORS_PATH    = "site/Card Authors.txt"      # who made each card; wins over the database
EDITS_PATH      = "site/Card Edits.json"       # website fixes; win over the PSD
CREATED_PATH    = "site/Card Creation Dates.json"   # when each card was made (kept by hand)
RATINGS_PATH    = "site/Card Ratings.json"     # card ratings 1-10 from the TTS sessions: data only (the import deleted 2026-11-12); a card rename moves its entries; the bot never reads it
MAILMAP_PATH    = ".mailmap"                   # git's: one person's several git names as one (%aN)
RULES_PATH      = "TTS/decks.json"             # deck rules for the TTS save

# The proposal forum's votes (commands/forum_votes.py): defaults; /forum_votes saves changes in FORUM_JSON
FORUM_JSON      = "forum_votes.json"           # its settings and the steps done per post (gitignored, per machine)
FORUM_VOTES_ON  = True
FORUM_ID        = 1569539619471499324          # the forum of proposed changes
FORUM_ROLE_ID   = 546427268751294475           # asked to vote on day 6
FORUM_RESULTS_ID = 1087579710428565526         # a won post's winner message goes here
