import sys, collections
rows=[]
for l in open(sys.argv[1]):
    if not l.startswith("import time:") or "self [us]" in l: continue
    _, rest = l.split(":",1); s,c,n = rest.split("|"); rows.append((int(s),int(c),n.rstrip("\n")))
agg=collections.Counter()
for s,c,n in rows: agg[n.strip().split(".")[0]]+=s
tot=sum(agg.values())
print("total self us", tot)
for k,v in agg.most_common(25): print(f"{v/1000:8.1f} ms  {k}")
