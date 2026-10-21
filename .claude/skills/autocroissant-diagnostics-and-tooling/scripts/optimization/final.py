import json, os, re
os.chdir("/Users/michaelsrouji/Desktop/AutoCroissant")
rows = json.load(open("/Users/michaelsrouji/Documents/ac-optimization/docs_refaudit/refs5.json"))
H = {}
def hl(p):
    if p not in H: H[p] = open(p).read().split("\n")
    return H[p]
def deflookup(p, toks):
    hits = set()
    for t in toks:
        rx = re.compile(r"^\s*(?:async def|def|class)\s+" + re.escape(t) + r"\b|^\s*" + re.escape(t) + r"\s*(?::[^=]*)?=")
        h = [i+1 for i, l in enumerate(hl(p)) if rx.search(l)]
        if len(h) == 1: hits.add(h[0])
    return hits
OVR = json.load(open("/Users/michaelsrouji/Documents/ac-optimization/docs_refaudit/overrides.json")) if os.path.exists("/Users/michaelsrouji/Documents/ac-optimization/docs_refaudit/overrides.json") else {}
for r in rows:
    if r["status"] == "not-on-main": continue
    key = f'{r["skill"]}:{r["sline"]}:{r["start"]}'
    r["final"] = (r.get("new_start"), r.get("new_end")); r["conf"] = "symbol" if r.get("verified") else "blame"
    if not r.get("verified"):
        d = deflookup(r["path"], r.get("syms", []))
        if len(d) == 1:
            n = d.pop(); r["final"] = (n, n + (r["end"] - r["start"])); r["conf"] = "def-lookup"
    if key in OVR:
        v = OVR[key]
        if v == "hist": r["conf"] = "historical"; r["final"] = (r["start"], r["end"])
        elif v == "gone": r["conf"] = "gone"
        else: a, b = (v + "-" + v).split("-")[:2] if "-" not in v else v.split("-"); r["final"] = (int(a), int(b)); r["conf"] = "manual"
    if r["hist"] and r["conf"] not in ("manual", "gone"): r["conf"] = "historical"; r["final"] = (r["start"], r["end"])
    r["stale"] = r["final"] != (r["start"], r["end"]) and r["conf"] != "historical"
json.dump(rows, open("/Users/michaelsrouji/Documents/ac-optimization/docs_refaudit/final.json", "w"), indent=1)
from collections import Counter
print(Counter((r.get("conf"), r.get("stale")) for r in rows))
