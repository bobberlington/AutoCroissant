"""Render table.json as markdown (one table per skill)."""
import json, sys
D = "/Users/michaelsrouji/Documents/ac-optimization/docs_refaudit"
t = json.load(open(f"{D}/table.json"))
by = {}
for sk, sl, old, new, conf, anchor, _pos in t["edits"]:
    by.setdefault(sk, []).append((sl, old, new, conf, anchor))
out = []
for sk in sorted(by):
    rs = sorted(by[sk])
    out.append(f"\n#### .claude/skills/{sk}/SKILL.md ({len(rs)} refs)\n")
    out.append("| line | as written | replace with | check | code now at the new line |")
    out.append("|---|---|---|---|---|")
    for sl, old, new, conf, anchor in rs:
        a = anchor.replace("|", "\\|").replace("`", "'")
        out.append(f"| {sl} | `{old}` | `{new}` | {conf} | {a} |")
print("\n".join(out))
