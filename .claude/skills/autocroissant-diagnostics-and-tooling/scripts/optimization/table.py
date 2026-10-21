import json, os, re
from collections import OrderedDict
os.chdir("/Users/michaelsrouji/Desktop/AutoCroissant")
rows = json.load(open("/Users/michaelsrouji/Documents/ac-optimization/docs_refaudit/final.json"))
H = {}
def hl(p):
    if p not in H: H[p] = open(p).read().split("\n")
    return H[p]
groups = OrderedDict()
for r in rows:
    groups.setdefault((r["skill"], r["sline"], r["pos"]), []).append(r)
edits, hist, branch = [], [], []
for (sk, sl, pos), rs in groups.items():
    raw = rs[0]["raw"]
    if rs[0]["status"] == "not-on-main":
        branch.append((sk, sl, raw)); continue
    new = raw; changed = False; confs = set(); anchors = []
    # replace number parts right-to-left
    parts = list(re.finditer(r"~?\d{1,4}(?:[-–]\d{1,4})?", raw))
    # map rows to parts in order (rows were appended in order of parts)
    rs_num = [r for r in rs]
    if len(parts) != len(rs_num):
        parts = parts[len(parts)-len(rs_num):]
    for m, r in reversed(list(zip(parts, rs_num))):
        if r.get("conf") in ("historical",):
            hist.append((sk, sl, raw, r["start"])); continue
        if r.get("conf") == "gone":
            hist.append((sk, sl, raw + " [code left main]", r["start"])); continue
        a, b = r["final"]
        if a is None: continue
        if (a, b) == (r["start"], r["end"]): 
            continue
        tilde = "~" if m.group(0).startswith("~") else ""
        txt = f"{tilde}{a}" + (f"-{b}" if b != a or "-" in m.group(0) or "–" in m.group(0) else "")
        new = new[:m.start()] + txt + new[m.end():]; changed = True
        confs.add(r["conf"]); anchors.append(hl(r["path"])[a-1].strip()[:60])
    if changed:
        edits.append((sk, sl, raw, new, "/".join(sorted(confs)), anchors[0] if anchors else "", pos))
json.dump(dict(edits=edits, hist=hist, branch=branch), open("/Users/michaelsrouji/Documents/ac-optimization/docs_refaudit/table.json", "w"), indent=1)
print(len(edits), len(hist), len(branch))
for e in edits:
    if e[0]=="autocroissant-architecture-contract" and e[1] in (106,): print(e)
