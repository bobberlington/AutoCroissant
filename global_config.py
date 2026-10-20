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
