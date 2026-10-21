import os, sys, gc, subprocess, tracemalloc
ROOT = "/tmp/acprof/sb"; os.chdir(ROOT); sys.path.insert(0, ROOT)
def rss(): return int(subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())], capture_output=True, text=True).stdout) / 1024
from os.path import expanduser, relpath
from pathlib import Path
from commands.psd_analyzer import PSDParser, RepositoryTraverser, StatsDatabase
CLONE = Path(expanduser("~/Desktop/TTSCardMaker")).resolve()
db = StatsDatabase(); tr = RepositoryTraverser(db); tr._populate_types_from_local(str(CLONE))
parser = PSDParser(db.all_types, db, tr._icons_from_local(str(CLONE)))
paths = sorted(str(p) for p in CLONE.rglob("*.psd") if "/MDW/" not in str(p) and "/Markers/" not in str(p))
tracemalloc.start()
base = tracemalloc.get_traced_memory()[0]
print(f"start RSS {rss():.0f} MB")
cards = []
for i, p in enumerate(paths[:300], 1):
    cards.append(parser.parse(p, relpath(p, CLONE)))
    if i % 50 == 0:
        gc.collect(); cur, pk = tracemalloc.get_traced_memory()
        print(f"{i:4d} parses: RSS {rss():5.0f} MB; live python heap {(cur-base)/1e6:6.1f} MB (incl. {i} CardInfo kept); heap peak {pk/1e6:6.0f} MB")
