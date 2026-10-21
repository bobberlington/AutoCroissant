# race_ns.py N SWITCH [shared]: commit lookups across threads on the real _repo() against a local echo server; "shared" = one client for all threads (the bug)
import json, sys, threading
from functools import partial
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, "/Users/michaelsrouji/Desktop/AutoCroissant")

class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def log_message(self, *a): pass
    def do_GET(self):
        u = urlparse(self.path)
        p = parse_qs(u.query)["path"][0]
        body = [{"sha": p, "commit": {"author": {"name": p, "date": "2020-01-01T00:00:00Z"},
                                     "committer": {"name": p, "date": "2020-01-01T00:00:00Z"}}}]
        data = json.dumps(body).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)

srv = ThreadingHTTPServer(("127.0.0.1", 0), H); PORT = srv.server_address[1]
threading.Thread(target=srv.serve_forever, daemon=True).start()

import github
github.Github = partial(github.Github, base_url=f"http://127.0.0.1:{PORT}")
import commands.psd_analyzer as pa
t = pa.RepositoryTraverser(pa.stats_db)
t.repository = "o/r"
if len(sys.argv) > 3: shared = t._repo(); t._repo = lambda: shared   # control: one client for all threads

stop = False
sys.setswitchinterval(float(sys.argv[2]))
def burn():
    x = 0
    while not stop:
        x += 1
threading.Thread(target=burn, daemon=True).start()

paths = [f"Cards/C{i}.psd" for i in range(int(sys.argv[1]))]
with ThreadPoolExecutor(pa.LOOKUP_WORKERS) as ex:
    futs = {p: ex.submit(t._commit_page, p) for p in paths}
    main = {f"Main/M{i}.psd": t._commit_page(f"Main/M{i}.psd") for i in range(len(paths) // 4)}   # fallback/.reversed stand-in
    bad = [(p, f.result()[0].sha) for p, f in futs.items() if f.result()[0].sha != p]
bad += [(p, c[0].sha) for p, c in main.items() if c[0].sha != p]
stop = True
print(f"{len(bad)} of {len(paths) + len(main)} lookups got another file's commits; e.g. {bad[:3]}")
