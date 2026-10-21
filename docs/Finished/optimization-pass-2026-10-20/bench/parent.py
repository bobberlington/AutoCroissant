import sys
rows=[]
for l in open(sys.argv[1]):
    if not l.startswith("import time:") or "self [us]" in l: continue
    _, rest = l.split(":",1); s,c,n = rest.split("|"); n=n.rstrip("\n")
    rows.append((int(s),int(c),len(n)-len(n.lstrip()),n.strip()))
# importtime prints children before parents; parent = next row with smaller indent
for i,(s,c,ind,n) in enumerate(rows):
    if n in sys.argv[2:]:
        for j in range(i+1,len(rows)):
            if rows[j][2]<ind: print(f"{n:15s} cum {c/1000:6.1f} ms  imported by {rows[j][3]}"); break
