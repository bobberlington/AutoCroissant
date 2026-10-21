"""For `grep -n ...` commands followed by expected line numbers, flag numbers the grep doesn't hit."""
import re, subprocess, os, glob
os.chdir("/Users/michaelsrouji/Desktop/AutoCroissant")
for sk in sorted(glob.glob(".claude/skills/*/SKILL.md")):
    for i, line in enumerate(open(sk).read().split("\n"), 1):
        for m in re.finditer(r"`(grep -n[^`]*)`\s*\(([^)]*)\)", line):
            cmd, exp = m.group(1), m.group(2)
            if "git " in cmd or "$" in cmd or "~" in cmd: continue
            nums = [int(x) for x in re.findall(r"(?<![\w.:-])(\d{1,4})(?![\w.%])", exp)]
            rng = re.findall(r"(\d{1,4})-(\d{1,4})", exp)
            if not nums: continue
            try: out = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=20).stdout
            except Exception: continue
            hits = set()
            for l in out.split("\n"):
                mm = re.match(r"(?:[^:]+:)?(\d+)[:-]", l)
                if mm: hits.add(int(mm.group(1)))
            miss = [n for n in nums if n not in hits and n > 3]
            if miss and hits:
                print(f"{sk.split('/')[2][14:]}:{i} miss={miss} hits={sorted(hits)[:12]} | {cmd[:90]} ({exp[:60]})")
