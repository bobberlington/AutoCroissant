"""Time the parser on N PSDs (serial, parse_one style). Usage: parse50.py [N]"""
import sys, os, time, statistics, json
from os.path import expanduser, relpath
from pathlib import Path
ROOT = "/tmp/acprof/sb"; os.chdir(ROOT); sys.path.insert(0, ROOT)
from global_config import LOCAL_DIR_LOC
CLONE = Path(expanduser(LOCAL_DIR_LOC)).resolve()
from commands.psd_analyzer import PSDParser, RepositoryTraverser, StatsDatabase, CardValidator
from psd_tools import PSDImage
t0 = time.perf_counter()
db = StatsDatabase(); tr = RepositoryTraverser(db); tr._populate_types_from_local(str(CLONE))
parser = PSDParser(db.all_types, db, tr._icons_from_local(str(CLONE)))
t_setup = time.perf_counter() - t0
paths = sorted(str(p) for p in CLONE.rglob("*.psd") if "/MDW/" not in str(p) and "/Markers/" not in str(p))
n = int(sys.argv[1]) if len(sys.argv) > 1 else 50
step = len(paths) / n
pick = [paths[int(i * step)] for i in range(n)]
rows = []
for p in pick:
    rel = relpath(p, CLONE)
    a = time.perf_counter(); psd = PSDImage.open(p); b = time.perf_counter()
    nl = sum(1 for _ in psd.descendants()); c = time.perf_counter()
    card = parser.parse(p, rel); d = time.perf_counter()
    CardValidator.validate(card); e = time.perf_counter()
    rows.append(dict(rel=rel, mb=os.path.getsize(p)/1e6, layers=nl, open=b-a, walk=c-b, parse=d-c, validate=e-d, type=card.card_type))
par = [r["parse"] for r in rows]; op = [r["open"] for r in rows]
print(f"setup {t_setup*1e3:.0f} ms; {len(paths)} PSDs total; {n} sampled")
print(f"parse(): total {sum(par):.2f}s mean {statistics.mean(par)*1e3:.0f} ms median {statistics.median(par)*1e3:.0f} ms max {max(par)*1e3:.0f} ms")
print(f"PSDImage.open alone: total {sum(op):.2f}s mean {statistics.mean(op)*1e3:.0f} ms; descendants() walk mean {statistics.mean(r['walk'] for r in rows)*1e3:.1f} ms; validate mean {statistics.mean(r['validate'] for r in rows)*1e6:.0f} us")
print(f"size MB mean {statistics.mean(r['mb'] for r in rows):.1f} max {max(r['mb'] for r in rows):.1f}; layers mean {statistics.mean(r['layers'] for r in rows):.0f}")
for r in sorted(rows, key=lambda r: -r["parse"])[:8]:
    print(f"  {r['parse']*1e3:6.0f} ms  open {r['open']*1e3:5.0f}  {r['mb']:5.1f} MB {r['layers']:4d} layers  {r['type']:10s} {r['rel']}")
json.dump(rows, open("/tmp/acprof/parse50.json", "w"), indent=1)
