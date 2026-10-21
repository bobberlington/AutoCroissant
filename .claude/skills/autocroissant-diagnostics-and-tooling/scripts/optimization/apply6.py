"""Apply refs6.json to the skills: verified+exact stale refs, plus OVR (skill, sline, part) -> (file, a, b) or None (skip)."""
import json, re, os, sys
from collections import OrderedDict
D = "/Users/michaelsrouji/Documents/ac-optimization/docs_refaudit"
R = "/Users/michaelsrouji/Desktop/AutoCroissant/"
rows = json.load(open(f"{D}/refs6.json"))
OVR = {tuple(k.split("|")): v for k, v in json.load(open(f"{D}/overrides6.json")).items()} if os.path.exists(f"{D}/overrides6.json") else {}
dry = "--dry" in sys.argv
groups = OrderedDict()
for r in rows:
    groups.setdefault((r["sk"], r["sline"], r["pos"]), []).append(r)
edits, skipped = [], []
for (sk, sl, pos), rs in groups.items():
    raw = rs[0]["raw"]
    if rs[0]["status"] in ("not-on-main", "hist"): continue
    parts = list(re.finditer(r"~?\d{1,4}(?:[-–]\d{1,4})?", raw))
    parts = parts[len(parts) - len(rs):]
    targets = []
    for m, r in zip(parts, rs):
        key = (r["skill"], str(sl), r["part"])
        if key in OVR:
            v = OVR[key]
            targets.append(None if v is None else (R and v[0], v[1], v[2]))
        elif r["status"] == "stale" and r.get("verified") and r.get("exact"):
            targets.append(tuple(r["new"]))
        elif r["status"] in ("ok", "same", "wt-new"):
            targets.append((r["path"], r["start"], r["end"]))
        else:
            targets.append(None)
    if all(t is None or (t[0] == rs[i]["path"] and t[1] == rs[i]["start"] and t[2] == rs[i]["end"]) for i, t in enumerate(targets)):
        if any(t is None for t in targets) and any(r["status"] not in ("ok", "same", "wt-new") for r in rs):
            skipped.append((sk, sl, raw))
        continue
    if any(t is None for t in targets):
        skipped.append((sk, sl, raw, "partial")); continue
    files = {t[0] for t in targets}
    fm = re.match(r"((?:commands/)?)([A-Za-z_]+\.py)", raw)
    if len(files) > 1 or not fm:
        if fm and len(files) > 1:
            pre = fm.group(1)
            new = ", ".join(f"{pre}{os.path.basename(t[0])}:{t[1]}" + (f"-{t[2]}" if t[2] != t[1] else "") for t in targets)
            if raw[fm.end():fm.end()+1] != ":": skipped.append((sk, sl, raw, "multi-file")); continue
        else:
            skipped.append((sk, sl, raw, "bare-move")); continue
    else:
        new = raw
        for m, t in reversed(list(zip(parts, targets))):
            tilde = "~" if m.group(0).startswith("~") else ""
            txt = f"{tilde}{t[1]}" + (f"-{t[2]}" if t[2] != t[1] or "-" in m.group(0) or "–" in m.group(0) else "")
            new = new[:m.start()] + txt + new[m.end():]
        nf = os.path.basename(files.pop())
        new = new[:fm.start(2)] + nf + new[fm.end(2):]
    if new != raw: edits.append((sk, sl, pos, raw, new))
by = {}
for sk, sl, pos, raw, new in edits: by.setdefault(sk, []).append((sl, pos, raw, new))
n = 0
for sk, es in by.items():
    lines = open(R + sk).read().split("\n")
    for sl, pos, raw, new in sorted(es, key=lambda e: (e[0], -e[1])):
        line = lines[sl - 1]
        assert line[pos:pos + len(raw)] == raw, (sk, sl, raw)
        lines[sl - 1] = line[:pos] + new + line[pos + len(raw):]; n += 1
        if dry: print(f"{sk.split('/')[2][:28]:28} {sl:4} {raw!r} -> {new!r}")
    if not dry: open(R + sk, "w").write("\n".join(lines))
print(n, "edits;", len(skipped), "skipped")
for s in skipped: print("SKIP", s)
