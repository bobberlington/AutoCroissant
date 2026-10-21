"""Time PSDImage.open broken down by tagged-block key and by channel image data, over N sampled PSDs."""
import sys, os, time, collections
from pathlib import Path
from psd_tools import PSDImage
import psd_tools.psd.tagged_blocks as tb
import psd_tools.psd.layer_and_mask as lm
CLONE = Path(os.path.expanduser("~/Desktop/TTSCardMaker"))
paths = sorted(str(p) for p in CLONE.rglob("*.psd") if "/MDW/" not in str(p) and "/Markers/" not in str(p))
n = int(sys.argv[1]); step = len(paths) / n; pick = [paths[int(i * step)] for i in range(n)]
acc = collections.Counter(); cnt = collections.Counter(); depth = [0]
orig = tb.TaggedBlock.read.__func__
def read(cls, fp, version=1, padding=1):
    depth[0] += 1; a = time.perf_counter()
    try:
        r = orig(cls, fp, version, padding)
    finally:
        depth[0] -= 1
    if depth[0] == 0:
        k = getattr(r.key, "name", str(r.key)); acc[k] += time.perf_counter() - a; cnt[k] += 1
    return r
tb.TaggedBlock.read = classmethod(read)
cid = lm.ChannelImageData.read.__func__
def cread(cls, *a, **k):
    t = time.perf_counter(); r = cid(cls, *a, **k); acc["<channel image data>"] += time.perf_counter() - t; cnt["<channel image data>"] += 1; return r
lm.ChannelImageData.read = classmethod(cread)
total = 0.0
for p in pick:
    a = time.perf_counter(); PSDImage.open(p); total += time.perf_counter() - a
print(f"{n} PSDs: PSDImage.open total {total:.2f}s (instrumented)")
for k, v in acc.most_common(14):
    print(f"  {v:6.2f}s {100*v/total:5.1f}%  {cnt[k]:5d}x  {k}")
