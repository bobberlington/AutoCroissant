import os, sys, gc, subprocess, resource
ROOT = "/tmp/acprof/sb"; os.chdir(ROOT); sys.path.insert(0, ROOT)
def rss(): return int(subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())], capture_output=True, text=True).stdout) / 1024
def peak(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
print(f"bare interpreter        RSS {rss():6.1f} MB")
import main
print(f"after import main       RSS {rss():6.1f} MB")
import tracemalloc; tracemalloc.start()
from commands.psd_analyzer import stats_db, init_psd
from commands.analytics import init_reminder
init_reminder()
a = tracemalloc.get_traced_memory()[0]
stats_db.load(); b = tracemalloc.get_traced_memory()[0]
from commands.query_card import card_repo
card_repo.prep_dataframes(); c = tracemalloc.get_traced_memory()[0]
tracemalloc.stop()
print(f"after pickles+dataframes RSS {rss():6.1f} MB  (python heap: pickles {(b-a)/1e6:.1f} MB, dataframes {(c-b)/1e6:.1f} MB; stats {len(stats_db.stats)}, old_stats {sum(map(len, stats_db.old_stats.values()))} entries)")
sys.path.insert(0, "/tmp/acprof"); import search_bench as sbm
sbm.setup(); ab, rb, nm = sbm.queries()
[card_repo.ability_search_engine(q) for q in ab]; [card_repo._find_closest_match(q) for q in nm]
gc.collect()
print(f"after registry + 200 queries RSS {rss():6.1f} MB, peak {peak():6.1f} MB")
from os.path import expanduser, relpath
from pathlib import Path
from commands.psd_analyzer import PSDParser, RepositoryTraverser, StatsDatabase
CLONE = Path(expanduser("~/Desktop/TTSCardMaker")).resolve()
db = StatsDatabase(); tr = RepositoryTraverser(db); tr._populate_types_from_local(str(CLONE))
parser = PSDParser(db.all_types, db, tr._icons_from_local(str(CLONE)))
paths = sorted(str(p) for p in CLONE.rglob("*.psd") if "/MDW/" not in str(p) and "/Markers/" not in str(p))
big = max(paths, key=os.path.getsize)
parser.parse(big, relpath(big, CLONE)); gc.collect()
print(f"after parsing largest PSD ({os.path.getsize(big)/1e6:.0f} MB) RSS {rss():6.1f} MB, peak {peak():6.1f} MB")
step = len(paths)/50
for i in range(50): p = paths[int(i*step)]; parser.parse(p, relpath(p, CLONE))
gc.collect()
print(f"after 50 more parses    RSS {rss():6.1f} MB, peak {peak():6.1f} MB")
