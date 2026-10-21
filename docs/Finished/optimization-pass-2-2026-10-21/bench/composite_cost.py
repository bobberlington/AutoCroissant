import time, sys
from psd_tools import PSDImage
p = "/tmp/acperf/doomer/Mini_Doomer.psd"
psd = PSDImage.open(p)
print("canvas", psd.width, psd.height, "layers", len(list(psd.descendants())))
for label, kw in (("viewport hp row 411x86", dict(viewport=(763, 1199, 1174, 1285))),
                  ("viewport ability 1130x456", dict(viewport=(27, 1802, 1157, 2258))),
                  ("full canvas", {})):
    t = time.perf_counter(); im = psd.composite(ignore_preview=True, **kw); print(f"{label}: {time.perf_counter() - t:.2f} s, {im.size}")
