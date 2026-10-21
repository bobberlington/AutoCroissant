"""The GitHub reads of two back-to-back no-change /update_tts_save runs (ref, commit, trees, 3 blobs) plus the
website's cards.json, live and read-only (no POST/PATCH is made). Usage: tts_ns.py ROOT"""
import os, sys
ROOT = sys.argv[1]
os.chdir(ROOT); sys.path.insert(0, ROOT)
import requests.adapters
log = []
_send = requests.adapters.HTTPAdapter.send
def send(self, req, **kw):
    assert req.method == "GET", req.method
    r = _send(self, req, **kw); log.append((r.status_code, len(r.content)))
    return r
requests.adapters.HTTPAdapter.send = send
import commands.tts_save as ts
def reads():                                            # update_repo_save's read sequence, one attempt
    head = ts._api("GET", "git/ref/heads/main").json()["object"]["sha"]
    base = ts._api("GET", f"git/commits/{head}").json()["tree"]["sha"]
    files = ts._files_at(base, (ts.TTS_SAVE_PATH, ts.RULES_PATH, ts.EDITS_PATH))
    blob = getattr(ts, "_blob", None) or (lambda sha: ts._api("GET", f"git/blobs/{sha}", raw=True).content)
    return [blob(files[p][1]) for p in (ts.TTS_SAVE_PATH, ts.RULES_PATH, ts.EDITS_PATH) if p in files]
def site():
    etag, known = getattr(ts, "_site", ("", None))
    r = ts.get(f"{ts.SITE_URL}/data/cards.json", timeout=30, headers={"If-None-Match": etag} if etag and known is not None else {})
    if hasattr(ts, "_site") and r.status_code == 200:
        ts._site = (r.headers.get("ETag", ""), r.json()["cards"])
for run in (1, 2):
    for sha in set(getattr(ts, "_objects", {})) - getattr(ts, "_used", set()):
        del ts._objects[sha]
    getattr(ts, "_used", set()).clear()
    log.clear(); data = reads(); site()
    print(f"run {run}: {len(log)} requests, {sum(n for _, n in log) / 1e6:.2f} MB, statuses {[s for s, _ in log]}, "
          f"blob bytes {[len(d) for d in data]}")
