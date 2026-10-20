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
TIMEZONE        = ZoneInfo("America/Los_Angeles")
