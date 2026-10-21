"""/update_stats cards:<6 cards> on a throwaway copy of a sandbox (live GitHub reads; only the copy's pickles change).
Two of the cards are deleted first, so they take the new-card path. Usage: e2e_update_ns.py ROOT OUT.json"""
import json, os, shutil, sys, time
from collections import Counter
SRC, OUT = sys.argv[1], sys.argv[2]
ROOT = SRC.rstrip("/") + "_e2e"
shutil.rmtree(ROOT, ignore_errors=True); shutil.copytree(SRC, ROOT, ignore=shutil.ignore_patterns("__pycache__"))
os.chdir(ROOT); sys.path.insert(0, ROOT)
import requests, requests.adapters
calls = Counter(); bytes_in = Counter()
_send = requests.adapters.HTTPAdapter.send
def kind(url):
    u = url.split("?")[0]
    if "/git/trees/" in u: return "tree"
    if "/git/blobs/" in u: return "blob"
    if "/commits" in u: return "commits"
    if u.rstrip("/").endswith("TTSCardMaker"): return "get_repo"
    return u
def send(self, req, **kw):
    r = _send(self, req, **kw); k = kind(req.url); calls[k] += 1
    return r
requests.adapters.HTTPAdapter.send = send
import commands.psd_analyzer as pa
from commands.query_card import card_repo
import commands.utils as utils
CARDS = ["Mini Doomer", "The Freezer", "Anubisath Guardian", "Z!!!!!", "Zigg", "The Watcher"]
NEW = ["The Freezer", "Zigg"]
db = pa.stats_db; db.load()
tree = requests.get(f"https://api.github.com/repos/{card_repo.repository}/git/trees/main?recursive=1",
                    headers=db._headers, timeout=30).json()
shas = {i["path"]: i["sha"] for i in tree["tree"]}
for c in db.stats.values():                       # pin everything else as unchanged
    if c.path in shas: c.blob_sha = shas[c.path]
for n in NEW: db.stats.pop(n, None)
db.save(); calls.clear()
a = time.perf_counter()
problems = pa.update_stats(None, verbose=False, cards=",".join(CARDS))
wall = time.perf_counter() - a
db.load()
cards = {n: db.stats[n].to_dict() for n in CARDS if n in db.stats}
print(f"wall {wall:.1f} s; requests {dict(calls)}; cards {len(cards)}")
json.dump({"wall_s": round(wall, 2), "requests": dict(calls), "cards": cards}, open(OUT, "w"), indent=1, default=str, sort_keys=True)
