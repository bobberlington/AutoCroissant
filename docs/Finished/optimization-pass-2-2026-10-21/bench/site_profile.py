"""build_site.py on the /tmp shared clone: phases timed, pickles served from /tmp copies, no network."""
import sys, os, time, io, cProfile, pstats, importlib.util, json
SITE = "/tmp/acperf/tts_site/site"
spec = importlib.util.spec_from_file_location("build_site", SITE + "/build/build_site.py")
bs = importlib.util.module_from_spec(spec); spec.loader.exec_module(bs)
assert bs.REPO == "/tmp/acperf/tts_site", bs.REPO
def fetch(url, tries=3):
    if url.startswith(bs.BOT_RAW):
        with open("/tmp/acperf/pickles/" + url[len(bs.BOT_RAW):], "rb") as fh: return fh.read()
    raise RuntimeError("network fetch in profile: " + url)
bs.fetch = fetch
acc = {}
def timed(name, fn):
    def w(*a, **k):
        t = time.perf_counter()
        try: return fn(*a, **k)
        finally:
            c, s = acc.get(name, (0, 0.0)); acc[name] = (c + 1, s + time.perf_counter() - t)
    return w
for n in ["repo_tree", "load_bot_pickle", "parse_edits", "apply_edits", "make_images", "accent_color", "build_icons",
          "image_versions", "site_banner", "share_text", "og_size", "parse_authors", "parse_banlist", "image_bytes"]:
    setattr(bs, n, timed(n, getattr(bs, n)))
_dumps = json.dumps; bs.json.dumps = timed("json.dumps", _dumps)
t = time.perf_counter(); pr = cProfile.Profile(); pr.enable(); bs.main(); pr.disable(); wall = time.perf_counter() - t
print(f"\n== build_site.main: {wall:.2f} s wall (cache {'warm' if len(sys.argv) < 2 else sys.argv[1]})")
for name, (c, s) in sorted(acc.items(), key=lambda kv: -kv[1][1]): print(f"   {s:7.3f} s  x{c:<5d} {name}")
s = io.StringIO(); pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(16); print("\n".join(s.getvalue().splitlines()[6:30]))
pages = sum(len(d) for _, d, _ in os.walk(bs.DIST + "/c")); print("share pages:", pages)
