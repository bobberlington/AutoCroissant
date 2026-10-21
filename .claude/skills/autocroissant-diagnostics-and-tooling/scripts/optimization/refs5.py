"""Skill file:line audit. Anchor each ref to the commit where its symbol sat on that line, map to HEAD."""
import re, subprocess, json, os, difflib
from functools import lru_cache
from collections import Counter
os.chdir("/Users/michaelsrouji/Desktop/AutoCroissant")
CMDS = {f for f in os.listdir("commands") if f.endswith(".py")}
def git(*a): return subprocess.run(["git", *a], capture_output=True, text=True).stdout
def resolve(name):
    b = os.path.basename(name)
    if name.startswith("archive/") or "scripts/" in name: return None
    if b == "main.py": return "main.py"
    if b in CMDS: return "commands/" + b
    if b in ("global_config.py", "setup.py"): return b
    return None
@lru_cache(None)
def head(p): return open(p).read().split("\n")
@lru_cache(None)
def at(c, p):
    o = subprocess.run(["git", "show", f"{c}:{p}"], capture_output=True, text=True)
    return tuple(o.stdout.split("\n")) if o.returncode == 0 else None
@lru_cache(None)
def mapping(c, p):
    old = at(c, p); new = head(p)
    if old is None: return None
    m = {}
    for a, b, n in difflib.SequenceMatcher(None, old, new, autojunk=False).get_matching_blocks():
        for k in range(n): m[a+k+1] = b+k+1
    return m
@lru_cache(None)
def blame(sk):
    res = {}
    for line in git("blame", "--line-porcelain", sk).split("\n"):
        mm = re.match(r"^([0-9a-f]{40}) \d+ (\d+)", line)
        if mm: res[int(mm.group(2))] = mm.group(1)
    return res
@lru_cache(None)
def file_commits(p, upto): return tuple(git("log", "--format=%H", upto, "--", p).split())
FILE = r"((?:commands/)?[A-Za-z_]+\.py)"
NUMS = r"(~?\d{1,4}(?:[-–]\d{1,4})?(?:, ?~?\d{1,4}(?:[-–]\d{1,4})?)*)"
PAT = re.compile(FILE + r"(?::| \(lines | lines | line | ~|, line )" + NUMS)
BARE = re.compile(r"(?<![\w.:])\(?:(\d{2,4}(?:[-–]\d{1,4})?)\b")
GENERIC = {"None","True","False","commands","cards","file","stats","only","local","mode","sets","update","ABILITY","LAYER","HIGH","STATS","ICON","TOKEN","UNKNOWN","contains","grep","archive","folder","files","version","appends","leftovers","random","shuffle","pull","push","rules","psd_analyzer","query_card","utils","update_bot","music_player","global_config","frankenstein","main"}
def toks_of(win):
    t = set()
    for s in re.findall(r"`([^`]+)`", win):
        for x in re.findall(r"[A-Za-z_][A-Za-z_0-9]{3,}", s):
            if x not in GENERIC: t.add(x)
    for x in re.findall(r"(?<![\w.])/?([a-z]+_[a-z_]+|[A-Z][A-Z_]{4,})\b", win):
        if x not in GENERIC and not x.endswith("_py"): t.add(x)
    return t
def seg(lines, a, b): return "\n".join(lines[max(0, a-4): b+3])
rows = []
for sk in sorted(git("ls-files", ".claude/skills/*/SKILL.md").split()):
    text = open(sk).read().split("\n"); bl = blame(sk)
    for i, line in enumerate(text, 1):
        spans = [(m.start(), m.end(), m.group(1), m.group(2)) for m in PAT.finditer(line)]
        extra = []
        for m in BARE.finditer(line):
            if any(s <= m.start() < e for s, e, _, _ in spans): continue
            prev = [s for s in spans if s[0] < m.start()]
            if prev and m.start() - prev[-1][1] < 220: extra.append((m.start(), m.end(), prev[-1][2], m.group(1)))
        for s, e, f, nums in spans + extra:
            for part in re.split(r", ?", nums):
                part = part.lstrip("~"); a, *b = re.split(r"[-–]", part)
                a = int(a); b = int(b[0]) if b else a
                if b < a: b = int(str(a)[:len(str(a))-len(str(b))] + str(b))
                rows.append(dict(skill=sk.split("/")[2], sline=i, file=f, path=resolve(f), start=a, end=b,
                                 pos=s, blame=bl.get(i), ctx=line, raw=line[s:e]))
for r in rows:
    p = r["path"]; ctx = r["ctx"]
    after = ctx[r["pos"]: r["pos"]+90]
    dm = re.search(r"\b(on|as of|until|before|in|at|was)\b[^|;)]{0,12}?(20\d\d-\d\d-\d\d)", after)
    r["date_tag"] = dm.group(2) if dm else None
    r["hist"] = bool(dm and dm.group(2) != "2026-10-20") or "on the archive branch" in after or "was psd_analyzer" in ctx[r["pos"]-20:r["pos"]]
    if p is None or not os.path.exists(p): r["status"] = "not-on-main"; continue
    toks = toks_of(ctx[max(0, r["pos"]-130): r["pos"]+40]); r["syms"] = sorted(toks)
    anchor = r["blame"]; how = "blame"
    bm = mapping(anchor, p) or {}
    bns = bm.get(r["start"]); bne = bm.get(r["end"]) or bns
    blame_ok = bool(toks) and bns is not None and any(t in "\n".join(head(p)[max(0, bns-9): bne+8]) for t in toks)
    if blame_ok: how = "symbol"
    if toks and not blame_ok:
        for c in file_commits(p, r["blame"]) or ():
            o = at(c, p)
            if o and r["start"] <= len(o) and any(t in seg(o, r["start"], r["end"]) for t in toks):
                anchor = c; how = "symbol"; break
        else: how = "blame-nosym"
    r["anchor"], r["how"] = anchor, how
    m = mapping(anchor, p); old = at(anchor, p)
    if not old or r["start"] > len(old): r["status"] = "beyond-eof"; continue
    r["old_text"] = old[r["start"]-1].strip()
    ns, ne = m.get(r["start"]), m.get(r["end"])
    if ns is None:
        ks = [k for k in m if k <= r["start"]]; r["approx"] = True
        if ks: k = max(ks); ns = m[k] + (r["start"]-k)
    if ne is None and ns: ne = ns + (r["end"]-r["start"])
    r["new_start"], r["new_end"] = ns, ne
    L = head(p)
    if ns is None: r["status"] = "changed"; r["verified"] = False; continue
    r["verified"] = bool(toks) and any(t in "\n".join(L[max(0, ns-9): ne+8]) for t in toks)
    r["now_ok"] = bool(toks) and any(t in seg(L, r["start"], r["end"]) for t in toks)
    if ns == r["start"] and ne == r["end"]: r["status"] = "ok"
    elif r.get("approx"): r["status"] = "changed"
    else: r["status"] = "stale"
json.dump(rows, open("/Users/michaelsrouji/Documents/ac-optimization/docs_refaudit/refs5.json", "w"), indent=1)
print(len(rows), Counter(r["status"] for r in rows))
print(Counter((r["status"], r.get("how"), r.get("verified")) for r in rows if r["status"] != "not-on-main"))
print("hist", Counter((r["status"], r["hist"]) for r in rows))
