"""Apply table.json's line-ref edits to the skills in place (re-run refs5/final/table first)."""
import json
D = "/Users/michaelsrouji/Documents/ac-optimization/docs_refaudit"
R = "/Users/michaelsrouji/Desktop/AutoCroissant/.claude/skills"
t = json.load(open(f"{D}/table.json"))
by = {}
for sk, sl, old, new, conf, anchor, pos in t["edits"]:
    by.setdefault(sk, []).append((sl, pos, old, new))
n = 0
for sk, es in by.items():
    path = f"{R}/{sk}/SKILL.md"
    lines = open(path).read().split("\n")
    for sl, pos, old, new in sorted(es, key=lambda e: (e[0], -e[1])):   # right to left within a line
        line = lines[sl - 1]
        assert line[pos:pos + len(old)] == old, (sk, sl, old)
        lines[sl - 1] = line[:pos] + new + line[pos + len(old):]; n += 1
    open(path, "w").write("\n".join(lines))
print(n, "refs updated")
