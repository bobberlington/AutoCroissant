"""compare.py a.json b.json: field-by-field differences."""
import json, sys
a, b = (json.load(open(p)) for p in sys.argv[1:3])
keys = sorted(set(a) | set(b)); diffs = 0; byfield = {}
for k in keys:
    if k not in a or k not in b: print("ONLY IN", "a" if k in a else "b", k); diffs += 1; continue
    da, db = a[k], b[k]
    for f in sorted(set(da) | set(db)):
        if da.get(f) != db.get(f):
            diffs += 1; byfield[f] = byfield.get(f, 0) + 1
            print(f"DIFF {k} [{f}]\n  a: {str(da.get(f))[:300]}\n  b: {str(db.get(f))[:300]}")
print(f"{len(keys)} cards, {diffs} differences, by field: {byfield}")
