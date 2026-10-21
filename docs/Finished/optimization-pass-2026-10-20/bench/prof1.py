import sys, os, time, cProfile, pstats, io
from os.path import expanduser, relpath
from pathlib import Path
ROOT = "/tmp/acprof/sb"; os.chdir(ROOT); sys.path.insert(0, ROOT)
from global_config import LOCAL_DIR_LOC
CLONE = Path(expanduser(LOCAL_DIR_LOC)).resolve()
from commands.psd_analyzer import PSDParser, RepositoryTraverser, StatsDatabase, CardValidator
db = StatsDatabase(); tr = RepositoryTraverser(db); tr._populate_types_from_local(str(CLONE))
parser = PSDParser(db.all_types, db, tr._icons_from_local(str(CLONE)))
p = str(CLONE / sys.argv[1]); rel = sys.argv[1]
parser.parse(p, rel)  # warm
ts=[]
for _ in range(10):
    a=time.perf_counter(); parser.parse(p, rel); ts.append(time.perf_counter()-a)
print("warm parse x10: min %.0f ms median %.0f ms" % (min(ts)*1e3, sorted(ts)[5]*1e3))
pr = cProfile.Profile(); pr.enable()
for _ in range(5): parser.parse(p, rel)
pr.disable()
s = io.StringIO(); st = pstats.Stats(pr, stream=s); st.sort_stats("cumulative").print_stats(35); print(s.getvalue()[:9000])
s = io.StringIO(); st = pstats.Stats(pr, stream=s); st.sort_stats("tottime").print_stats(20); print(s.getvalue()[:6000])
