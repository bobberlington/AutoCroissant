"""Live commit lookups (_get_remote_timestamp) on a sandbox: requests, time, results. Usage: commits_ns.py ROOT OUT.json [page]
page: also check the more-than-one-page path with COMMITS_PAGE set to that (new code only). Read-only GitHub GETs."""
import json, os, sys, time
ROOT, OUT = sys.argv[1], sys.argv[2]
os.chdir(ROOT); sys.path.insert(0, ROOT)
import requests.adapters
calls = []
_send = requests.adapters.HTTPAdapter.send
def send(self, req, **kw):
    calls.append(req.url.split("?")[0].replace("https://api.github.com/repos/", ""))
    return _send(self, req, **kw)
requests.adapters.HTTPAdapter.send = send
import commands.psd_analyzer as pa
from commands.query_card import card_repo
pa.stats_db.load()
PATHS = ["Creatures/Other/5 Stars/Z!!!!!.psd", "Auxiliary/Minions/Mini_Doomer.psd", "Field/1 Stars/The_Freezer.psd",
         "Auxiliary/Minions/Anubisath_Guardian.psd", "Items/Red_Bloon.psd"]
if len(sys.argv) > 3:
    pa.COMMITS_PAGE = int(sys.argv[3])
tr = pa.RepositoryTraverser(pa.stats_db); tr.repository = card_repo.repository
calls.clear(); a = time.perf_counter(); repo = tr._repo(); setup = (len(calls), time.perf_counter() - a)
rows = []
for path in PATHS:
    name = os.path.splitext(os.path.basename(path))[0].replace("_", " ")
    for kind, nm in (("existing", name), ("new", "__not_a_card__ " + name)):
        if kind == "existing" and not (nm in pa.stats_db.stats and pa.stats_db.stats[nm].author):
            continue
        calls.clear(); a = time.perf_counter()
        ts, author = tr._get_remote_timestamp(repo, path, nm)
        rows.append({"path": path, "kind": kind, "timestamp": ts, "author": author, "requests": len(calls),
                     "ms": round((time.perf_counter() - a) * 1e3)})
        print(rows[len(rows) - 1])
print(f"client + get_repo: {setup[0]} requests, {setup[1]*1e3:.0f} ms")
json.dump(rows, open(OUT, "w"), indent=1)
