"""psd_fonts over N PSDs: sequential vs a process pool (the --all path's CPU-bound parse)."""
import os, sys, time, random
sys.path.insert(0, "/Users/michaelsrouji/Desktop/AutoCroissant")
import types
fake = types.ModuleType("config"); fake.TOKEN = fake.GIT_TOKEN = ""; fake.ADMINS = []; fake.browser_cookies = False
sys.modules["config"] = fake
import commands.font_check as fc
from concurrent.futures import ProcessPoolExecutor
ROOT = "/Users/michaelsrouji/Desktop/TTSCardMaker"
def all_paths():
    out = []
    for folder, subdirs, files in os.walk(ROOT):
        subdirs[:] = sorted(d for d in subdirs if d not in fc.EXCLUDE_FOLDERS and d != ".git")
        for name in sorted(files):
            rel = os.path.relpath(os.path.join(folder, name), ROOT)
            if fc.is_card_psd(rel): out.append(os.path.join(folder, name))
    return out
def one(p):
    try: return p, fc.psd_fonts(p)
    except Exception as e: return p, {"error": str(e)}
if __name__ == "__main__":
    N, W = int(sys.argv[1]), int(sys.argv[2])
    random.seed(7); paths = random.sample(all_paths(), N)
    os.system("uptime")
    t = time.perf_counter(); seq = [one(p) for p in paths]; ts = time.perf_counter() - t
    t = time.perf_counter()
    with ProcessPoolExecutor(W) as ex: par = list(ex.map(one, paths, chunksize=4))
    tp = time.perf_counter() - t
    print(f"{N} PSDs ({sum(os.path.getsize(p) for p in paths)/1e6:.0f} MB): sequential {ts:.1f} s ({ts/N:.3f}/PSD), pool({W}) {tp:.1f} s -> {ts/tp:.1f}x; same results: {seq == par}")
