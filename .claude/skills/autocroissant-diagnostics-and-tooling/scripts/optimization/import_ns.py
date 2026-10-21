"""`import main` wall time, RSS and heavy modules loaded, best of N fresh processes. Usage: import_ns.py ROOT [N]"""
import subprocess, sys, statistics
ROOT, N = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 5
CHILD = r"""
import sys, time, resource
t = time.perf_counter(); import main; t = time.perf_counter() - t
rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1 << 20 if sys.platform == "darwin" else 1 << 10)
heavy = [m for m in ("cv2", "yt_dlp", "github", "git", "numpy", "pandas", "psd_tools") if m in sys.modules]
print(f"{t*1e3:.1f} {rss:.1f} {','.join(heavy)}")
"""
runs = []
for _ in range(N):
    out = subprocess.run([sys.executable, "-c", CHILD], cwd=ROOT, capture_output=True, text=True)
    last = out.stdout.strip().splitlines()[-1]
    ms, rss, heavy = last.split(" ", 2)
    runs.append((float(ms), float(rss), heavy))
ms = [r[0] for r in runs]; rss = [r[1] for r in runs]
print(f"import main: best {min(ms):.0f} ms, median {statistics.median(ms):.0f} ms; peak RSS median {statistics.median(rss):.1f} MB; loaded: {runs[0][2]}")
