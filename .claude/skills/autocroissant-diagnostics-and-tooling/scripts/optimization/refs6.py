"""Skill file:line refresh against the WORKING TREE, following code moved across files (psd_analyzer split).
Anchor: blame commit of the skill line (uncommitted lines: the matching committed line's blame, else the
working tree). Map anchor:path -> every working-tree .py by difflib blocks; verify by a nearby symbol."""
import re, subprocess, json, os, difflib, sys
from functools import lru_cache
from collections import Counter
REPO = "/Users/michaelsrouji/Desktop/AutoCroissant"
OUT = "/Users/michaelsrouji/Documents/ac-optimization/docs_refaudit"
os.chdir(REPO)
sys.path.insert(0, OUT)
from refs5_lib import PAT, BARE, toks_of, resolve_name
ZERO = "0" * 40
WT_FILES = ["main.py", "global_config.py", "setup.py"] + sorted("commands/" + f for f in os.listdir("commands") if f.endswith(".py"))
def git(*a): return subprocess.run(["git", *a], capture_output=True, text=True).stdout
@lru_cache(None)
def wt(p): return tuple(open(p).read().split("\n")) if os.path.exists(p) else None
@lru_cache(None)
def at(c, p):
    if c == "WT": return wt(p)
    o = subprocess.run(["git", "show", f"{c}:{p}"], capture_output=True, text=True)
    return tuple(o.stdout.split("\n")) if o.returncode == 0 else None
def norm(l): return l.strip()
@lru_cache(None)
def blocks(c, p, q):
    """{old line: (new line, block size)} from anchor c:p to working-tree q."""
    old, new = at(c, p), wt(q)
    if old is None or new is None: return {}
    sm = difflib.SequenceMatcher(None, [norm(x) for x in old], [norm(x) for x in new], autojunk=False)
    m = {}
    for a, b, n in sm.get_matching_blocks():
        real = sum(1 for k in range(n) if len(norm(old[a+k])) > 6)
        if q != p and real < 3: continue
        for k in range(n): m[a+k+1] = (b+k+1, real)
    return m
@lru_cache(None)
def text_index():
    idx = {}
    for f in WT_FILES:
        for i, l in enumerate(wt(f), 1): idx.setdefault(norm(l), []).append((f, i))
    return idx
def locate(c, p, line):
    """Best (file, line, exact) for anchor c:p:line in the working tree."""
    files = [p] + [f for f in WT_FILES if f != p]
    m = blocks(c, p, p)
    if line in m and m[line][1] >= 3: return (p, m[line][0], True)
    t = norm(at(c, p)[line-1])
    hits = text_index().get(t, [])
    if len(t) > 10 and len(hits) == 1: return (hits[0][0], hits[0][1], True)
    best = None
    for q in files:
        mq = blocks(c, p, q)
        if line in mq and (best is None or mq[line][1] > best[3]): best = (q, mq[line][0], True, mq[line][1])
    if best: return best[:3]
    for d in range(1, 7):                       # nearest mapped neighbour
        for k in (line - d, line + d):
            for q in files:
                mq = blocks(c, p, q)
                if k in mq and mq[k][1] >= 3:
                    return (q, mq[k][0] + (line - k), False)
    return None
@lru_cache(None)
def blame(sk, rev=None):
    res = {}
    args = ["blame", "--line-porcelain"] + ([rev] if rev else []) + ["--", sk]
    for l in git(*args).split("\n"):
        mm = re.match(r"^([0-9a-f]{40}) \d+ (\d+)", l)
        if mm: res[int(mm.group(2))] = mm.group(1)
    return res
@lru_cache(None)
def committed(sk): return git("show", f"HEAD:{sk}").split("\n")
def anchor_for(sk, i, line, raw):
    b = blame(sk).get(i)
    if b and b != ZERO: return b
    old = committed(sk); bh = blame(sk, "HEAD")
    cands = [(difflib.SequenceMatcher(None, line, o).ratio(), j) for j, o in enumerate(old, 1) if raw in o]
    if cands:
        r, j = max(cands)
        if r > 0.5: return bh.get(j, "WT")
    return "WT"
def near(lines, a, b, toks, pad=8):
    return bool(toks) and any(t in "\n".join(lines[max(0, a-1-pad): b+pad]) for t in toks)
@lru_cache(None)
def file_commits(p, upto): return tuple(git("log", "--format=%H", upto, "--", p).split())
rows = []
for sk in sorted(git("ls-files", ".claude/skills/*/SKILL.md").split()):
    text = open(sk).read().split("\n")
    for i, line in enumerate(text, 1):
        spans = [(m.start(), m.end(), m.group(1), m.group(2)) for m in PAT.finditer(line)]
        extra = []
        for m in BARE.finditer(line):
            if any(s <= m.start() < e for s, e, _, _ in spans): continue
            prev = [s for s in spans if s[0] < m.start()]
            if prev and m.start() - prev[-1][1] < 220: extra.append((m.start(), m.end(), prev[-1][2], m.group(1)))
        for s, e, f, nums in spans + extra:
            for part in re.split(r", ?", nums):
                part = part.lstrip("~"); a, *bb = re.split(r"[-–]", part)
                a = int(a); b = int(bb[0]) if bb else a
                if b < a: b = int(str(a)[:len(str(a))-len(str(b))] + str(b))
                rows.append(dict(skill=sk.split("/")[2], sk=sk, sline=i, file=f, path=resolve_name(f), start=a, end=b,
                                 pos=s, ctx=line, raw=line[s:e], part=part))
AUD = json.load(open(f"{OUT}/final.json"))
AUDK = {}
for x in AUD:
    AUDK.setdefault((x["skill"], x["sline"]), []).append(x)
@lru_cache(None)
def skill_map(sk):
    """working-tree skill line -> committed skill line (equal lines, then the most similar)."""
    old, new = committed(sk), open(sk).read().split("\n")
    m = {}
    for a, b, n in difflib.SequenceMatcher(None, old, new, autojunk=False).get_matching_blocks():
        for k in range(n): m[b+k+1] = a+k+1
    return m, old
def audited(r):
    m, old = skill_map(r["sk"])
    j = m.get(r["sline"])
    if j is None:
        cands = [(difflib.SequenceMatcher(None, r["ctx"], o).ratio(), k) for k, o in enumerate(old, 1) if r["raw"] in o]
        if cands and max(cands)[0] > 0.6: j = max(cands)[1]
    if j is None: return None
    for x in AUDK.get((r["skill"], j), []):
        if x["raw"] == r["raw"] and x["start"] == r["start"] and x["end"] == r["end"]: return x
    return None
for r in rows:
    p, ctx = r["path"], r["ctx"]
    after = ctx[r["pos"]: r["pos"]+90]
    dm = re.search(r"\b(on|as of|until|before|in|at|was)\b[^|;)]{0,12}?(20\d\d-\d\d-\d\d)", after)
    r["hist"] = bool(dm and dm.group(2) != "2026-10-20") or "archive branch" in after
    if p is None: r["status"] = "not-on-main"; continue
    toks = sorted(toks_of(ctx[max(0, r["pos"]-130): r["pos"]+40])); r["syms"] = toks
    W = wt(p)
    x = audited(r)
    r["aud"] = x and x.get("conf")
    if x and x.get("conf") in ("historical", "gone"): r["status"] = "hist"; continue
    if r["hist"]: r["status"] = "hist"; continue
    if W and r["end"] <= len(W) and near(W, r["start"], r["end"], toks, 3):
        r["status"] = "ok"; r["new"] = (p, r["start"], r["end"]); continue
    if x and x.get("final") and x["final"][0]:
        hs, he = x["final"]
        r["how"] = "audit-" + str(x.get("conf"))
    elif blame(r["sk"]).get(r["sline"]) == ZERO:
        r["status"] = "wt-new"; continue
    else:
        r["status"] = "unaudited"; continue
    r["head"] = (hs, he)
    H = at("HEAD", p)
    if he > len(H): r["status"] = "beyond-eof"; continue
    r["old_text"] = H[hs-1].strip()
    ls, le = locate("HEAD", p, hs), locate("HEAD", p, he)
    if ls is None: r["status"] = "lost"; continue
    if le is None or le[0] != ls[0]: le = (ls[0], ls[1] + (he - hs), False)
    q = ls[0]; L = wt(q)
    r["new"] = (q, ls[1], le[1]); r["exact"] = ls[2] and le[2]
    r["textsame"] = norm(L[ls[1]-1]) == norm(H[hs-1]) if ls[1] <= len(L) else False
    r["verified"] = near(L, ls[1], le[1], toks, 8) or (r["exact"] and r["textsame"])
    r["status"] = "same" if r["new"] == (p, r["start"], r["end"]) else "stale"
json.dump(rows, open(f"{OUT}/refs6.json", "w"), indent=1)
print(len(rows), Counter(r["status"] for r in rows))
print(Counter((r["status"], r.get("verified"), r.get("exact")) for r in rows if r["status"] in ("stale", "same")))
