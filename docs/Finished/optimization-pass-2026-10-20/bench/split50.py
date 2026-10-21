import sys, os, time, statistics
from os.path import expanduser, relpath
from pathlib import Path
ROOT = "/tmp/acprof/sb"; os.chdir(ROOT); sys.path.insert(0, ROOT)
from global_config import LOCAL_DIR_LOC
CLONE = Path(expanduser(LOCAL_DIR_LOC)).resolve()
import commands.psd_analyzer as pa
from psd_tools import PSDImage
db = pa.StatsDatabase(); tr = pa.RepositoryTraverser(db); tr._populate_types_from_local(str(CLONE))
parser = pa.PSDParser(db.all_types, db, tr._icons_from_local(str(CLONE)))
paths = sorted(str(p) for p in CLONE.rglob("*.psd") if "/MDW/" not in str(p) and "/Markers/" not in str(p))
n=50; step=len(paths)/n; pick=[paths[int(i*step)] for i in range(n)]
# count helper calls
import functools
calls = {"numpy": 0, "numpy_t": 0.0}
orig_core, orig_cov = pa.PSDParser._core_box, pa.PSDParser._pixel_coverage
def timed(f, key):
    @functools.wraps(f)
    def w(*a, **k):
        t=time.perf_counter()
        try: return f(*a, **k)
        finally: calls[key+"_t"] = calls.get(key+"_t",0)+time.perf_counter()-t; calls[key]=calls.get(key,0)+1
    return w
pa.PSDParser._core_box = staticmethod(timed(orig_core, "core_box"))
pa.PSDParser._pixel_coverage = staticmethod(timed(orig_cov, "coverage"))
T_open=T_ext=T_read=0.0
for p in pick:
    rel = relpath(p, CLONE)
    a=time.perf_counter(); data=open(p,"rb").read(); b=time.perf_counter()
    psd = PSDImage.open(p); c=time.perf_counter()
    card = pa.CardInfo(name="x", card_type="creature", path=rel)
    parser._extract_from_layers(psd, card, rel); d=time.perf_counter()
    T_read+=b-a; T_open+=c-b; T_ext+=d-c
print(f"50 PSDs: raw file read {T_read:.2f}s, PSDImage.open {T_open:.2f}s, parser's own layer work {T_ext:.2f}s")
print({k:(round(v,3) if isinstance(v,float) else v) for k,v in calls.items()})
