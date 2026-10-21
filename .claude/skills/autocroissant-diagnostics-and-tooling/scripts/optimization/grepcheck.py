"""Run every read-only grep command quoted in the skills; list the ones that now print nothing."""
import re, subprocess, os, glob
os.chdir("/Users/michaelsrouji/Desktop/AutoCroissant")
bad = []
for sk in sorted(glob.glob(".claude/skills/*/SKILL.md")):
    text = open(sk).read().split("\n")
    incode = False
    for i, line in enumerate(text, 1):
        if line.strip().startswith("```"): incode = not incode; continue
        cmds = []
        if incode and re.match(r"\s*(grep|git grep)", line):
            cmds.append(re.split(r"\s+#", line)[0].strip())
        for m in re.finditer(r"`([^`]*\bgrep\b[^`]*)`", line):
            cmds.append(m.group(1).replace("\\|", "|") if "\\|" in m.group(1) and "| grep" not in m.group(1) and "grep -" in m.group(1) and False else m.group(1))
        for c in cmds:
            if re.search(r"(^|\s)(rm|mv|python|python3|>)\s", c) or ">" in c or "$" in c or "~" in c or "<" in c: continue
            if not re.match(r"\s*(grep|git grep|git show|git log)", c): continue
            # in markdown tables `\|` is an escaped pipe
            c2 = c
            try:
                out = subprocess.run(c2, shell=True, capture_output=True, text=True, timeout=20)
            except subprocess.TimeoutExpired:
                continue
            o = out.stdout.strip()
            if o == "" or o == "0" and "grep -c" not in c2 and "wc -l" not in c2:
                bad.append((sk.split("/")[2], i, c2[:160], out.returncode))
for b in bad: print(*b)
print(len(bad))
