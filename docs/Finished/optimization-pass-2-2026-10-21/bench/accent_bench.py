import os, time, glob, importlib.util
import numpy as np
from PIL import Image
spec = importlib.util.spec_from_file_location("build_site", "/Users/michaelsrouji/Desktop/TTSCardMaker/site/build/build_site.py")
bs = importlib.util.module_from_spec(spec); spec.loader.exec_module(bs)
files = sorted(glob.glob("/Users/michaelsrouji/Desktop/TTSCardMaker/site/.cache/img/*-240.webp"))
def accent_np(path):
    with Image.open(path) as im:
        a = np.asarray(im.convert("RGB").resize((40, 80)), np.int16).reshape(-1, 3)
    mx, mn = a.max(1), a.min(1)
    vivid = a[(mx >= 70) & (mx - mn >= 60)]
    if not len(vivid): return None
    keys = (vivid // 40)
    _, inv, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    top = vivid[inv.ravel() == counts.argmax()]
    return "#%02x%02x%02x" % tuple(int(v) for v in top.sum(0) // len(top))
t = time.perf_counter(); a = [bs.accent_color(f) for f in files]; t1 = time.perf_counter() - t
t = time.perf_counter(); b = [accent_np(f) for f in files]; t2 = time.perf_counter() - t
t = time.perf_counter()
for f in files:
    with Image.open(f) as im: im.convert("RGB").resize((40, 80))
t3 = time.perf_counter() - t
print(f"{len(files)} thumbs: accent_color {t1:.2f} s; numpy version {t2:.2f} s (same: {a == b}, {sum(x != y for x, y in zip(a, b))} differ); decode+resize alone {t3:.2f} s")
