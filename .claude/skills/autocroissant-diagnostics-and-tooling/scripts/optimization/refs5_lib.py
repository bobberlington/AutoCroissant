import re, os
os.chdir('/Users/michaelsrouji/Desktop/AutoCroissant')
CMDS = {f for f in os.listdir('commands') if f.endswith('.py')}
def resolve_name(name):
    b = os.path.basename(name)
    if name.startswith("archive/") or "scripts/" in name: return None
    if b == "main.py": return "main.py"
    if b in CMDS: return "commands/" + b
    if b in ("global_config.py", "setup.py"): return b
    return None
FILE = r"((?:commands/)?[A-Za-z_]+\.py)"
NUMS = r"(~?\d{1,4}(?:[-–]\d{1,4})?(?:, ?~?\d{1,4}(?:[-–]\d{1,4})?)*)"
PAT = re.compile(FILE + r"(?::| \(lines | lines | line | ~|, line )" + NUMS)
BARE = re.compile(r"(?<![\w.:])\(?:(\d{2,4}(?:[-–]\d{1,4})?)\b")
GENERIC = {"None","True","False","commands","cards","file","stats","only","local","mode","sets","update","ABILITY","LAYER","HIGH","STATS","ICON","TOKEN","UNKNOWN","contains","grep","archive","folder","files","version","appends","leftovers","random","shuffle","pull","push","rules","psd_analyzer","query_card","utils","update_bot","music_player","global_config","frankenstein","main","card_models","card_edits","psd_parser","repo_traverser","github_api","type_emojis","tts_save","repo_history","analytics","management","help","setup"}
def toks_of(win):
    t = set()
    for s in re.findall(r"`([^`]+)`", win):
        for x in re.findall(r"[A-Za-z_][A-Za-z_0-9]{3,}", s):
            if x not in GENERIC: t.add(x)
    for x in re.findall(r"(?<![\w.])/?([a-z]+_[a-z_]+|[A-Z][A-Z_]{4,})\b", win):
        if x not in GENERIC and not x.endswith("_py"): t.add(x)
    return t
