"""Offline: after a changed /update_stats, does card_repo's next tree fetch get a 304? Fake GitHub, throwaway copy.
Usage: etag_handoff_ns.py ROOT"""
import hashlib, json, os, shutil, sys
SRC = sys.argv[1]; ROOT = SRC.rstrip("/") + "_etag"
shutil.rmtree(ROOT, ignore_errors=True); shutil.copytree(SRC, ROOT, ignore=shutil.ignore_patterns("__pycache__"))
os.chdir(ROOT); sys.path.insert(0, ROOT)
import commands.psd_analyzer as pa, commands.query_card as qc, commands.utils as utils
class Reply:
    def __init__(self, status, tree=None, etag=""):
        self.status_code, self._tree, self.headers, self.content, self.encoding = status, tree, {"ETag": etag} if etag else {}, b"", None
    def json(self):
        parses.append(1); return self._tree
    def raise_for_status(self):
        if self.status_code >= 400: raise RuntimeError(self.status_code)
state = {"tree": {"tree": [{"path": "Items/A.png", "sha": "a1"}]}}
log, parses = [], []
def get(url, headers=None, **kw):
    tree = state["tree"]; etag = '"%s"' % hashlib.sha1(json.dumps(tree).encode()).hexdigest()
    if "/git/trees/" not in url: return Reply(404)
    hit = (headers or {}).get("If-None-Match") == etag
    log.append(304 if hit else 200)
    return Reply(304) if hit else Reply(200, tree, etag)
pa.get = utils.get = qc.get = get
qc.CardRepository._populate_site_ids = lambda self: None
repo = qc.card_repo
repo.populate_files(); log.clear()                                   # startup: tree v1
state["tree"] = {"tree": [{"path": "Items/A.png", "sha": "a2"}, {"path": "Items/B.png", "sha": "b1"}]}
utils.command_queue.clear(); parses.clear()
pa.update_stats(None, verbose=False)                                 # changed tree: v2
print("tree JSON parses in update_stats:", len(parses))
for (args, kwargs), func in list(utils.command_queue):
    if func.__name__ == "populate_files": func(*args, **kwargs)
print("update_stats tree GET:", log, "| card list after it:", sorted(repo.git_files))
log.clear(); repo.populate_files()                                   # next card-list fetch, tree still v2
print("next card-list tree GET:", log)
