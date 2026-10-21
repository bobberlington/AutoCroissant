"""Tree requests at startup (init_query and the type-emoji listing in parallel, as on_ready queues them), then one
/sync_emojis listing. Live, read-only GitHub GETs. Usage: startup_ns.py ROOT"""
import os, sys, threading, time
ROOT = sys.argv[1]
os.chdir(ROOT); sys.path.insert(0, ROOT)
import requests.adapters
log = []
_send = requests.adapters.HTTPAdapter.send
def send(self, req, **kw):
    r = _send(self, req, **kw)
    if "/git/trees/" in req.url:
        log.append((threading.current_thread().name, r.status_code, int(r.headers.get("Content-Length") or len(r.content))))
    return r
requests.adapters.HTTPAdapter.send = send
import commands.query_card as qc
import commands.type_emojis as te
qc.CardRepository._populate_site_ids = lambda self: None          # not a tree request; keep the run short
def emojis():
    te._load_blobs()
    a = time.perf_counter(); te._list_type_icons(); done["emoji listing"] = time.perf_counter() - a
done = {}
a = time.perf_counter()
ts = [threading.Thread(target=qc.init_query, name="init_query"), threading.Thread(target=emojis, name="init_type_emojis")]
for t in ts: t.start()
for t in ts: t.join()
done["startup"] = time.perf_counter() - a
print("startup tree GETs:", log, {k: round(v * 1e3) for k, v in done.items()}, "ms")
log.clear(); a = time.perf_counter(); te._list_type_icons()
print("/sync_emojis tree GETs:", log, round((time.perf_counter() - a) * 1e3), "ms")
