"""Where font_check --all spends its time: per PSD, file read vs PSDImage parse vs engine-data/font walk; FontIndex once."""
import os, sys, time, random, io, cProfile, pstats
sys.path.insert(0, "/Users/michaelsrouji/Desktop/AutoCroissant")
import types
fake = types.ModuleType("config"); fake.TOKEN = fake.GIT_TOKEN = ""; fake.ADMINS = []; fake.browser_cookies = False
sys.modules["config"] = fake
from psd_tools import PSDImage
import commands.font_check as fc

ROOT = "/Users/michaelsrouji/Desktop/TTSCardMaker"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
paths = []
for folder, subdirs, files in os.walk(ROOT):
    subdirs[:] = sorted(d for d in subdirs if d not in fc.EXCLUDE_FOLDERS and d != ".git")
    for name in sorted(files):
        rel = os.path.relpath(os.path.join(folder, name), ROOT)
        if fc.is_card_psd(rel):
            paths.append(os.path.join(folder, name))
print("card PSDs:", len(paths), "total bytes: %.1f GB" % (sum(os.path.getsize(p) for p in paths) / 1e9))
random.seed(1); sample = random.sample(paths, N)

t = time.perf_counter(); idx = fc.FontIndex(os.path.join(ROOT, fc.FONTS_DIR)); t_idx = time.perf_counter() - t
print("FontIndex: %.2f s, %d files, %d faces, %d errors, %.0f MB" % (t_idx, idx.files, idx.faces, len(idx.errors),
      sum(os.path.getsize(p) for p in fc.font_files(os.path.join(ROOT, fc.FONTS_DIR))) / 1e6))

tot = dict(read=0, parse=0, walk=0, bytes=0, layers=0, types=0)
cpu0, wall0 = time.process_time(), time.perf_counter()
for p in sample:
    t = time.perf_counter()
    with open(p, "rb") as fh: data = fh.read()
    tot["read"] += time.perf_counter() - t; tot["bytes"] += len(data)
    t = time.perf_counter(); psd = PSDImage.open(io.BytesIO(data)); tot["parse"] += time.perf_counter() - t
    t = time.perf_counter(); fonts = fc.psd_fonts(psd); tot["walk"] += time.perf_counter() - t
    layers = list(psd.descendants()); tot["layers"] += len(layers); tot["types"] += sum(1 for l in layers if l.kind == "type")
cpu, wall = time.process_time() - cpu0, time.perf_counter() - wall0
print("sample %d PSDs, %.0f MB: read %.2f s, parse %.2f s, font walk %.2f s; wall %.2f cpu %.2f; layers %d, type layers %d"
      % (N, tot["bytes"] / 1e6, tot["read"], tot["parse"], tot["walk"], wall, cpu, tot["layers"], tot["types"]))
print("per PSD: read %.3f parse %.3f walk %.3f -> x813 = %.0f s read, %.0f s parse, %.0f s walk"
      % (tot["read"] / N, tot["parse"] / N, tot["walk"] / N, 813 * tot["read"] / N, 813 * tot["parse"] / N, 813 * tot["walk"] / N))
# second pass over the same files: page cache warm
t = time.perf_counter()
for p in sample[:10]:
    with open(p, "rb") as fh: fh.read()
print("warm re-read of 10: %.2f s" % (time.perf_counter() - t))
# profile parse+walk on 8 files
pr = cProfile.Profile(); pr.enable()
for p in sample[:8]:
    fc.psd_fonts(p)
pr.disable()
s = io.StringIO(); pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(28); print(s.getvalue()[:6000])
