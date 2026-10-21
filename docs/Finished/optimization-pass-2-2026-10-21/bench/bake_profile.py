"""card_bake --dry-run --card 'Mini Doomer' (hp 5 -> 8) against the tests' fake GitHub, per step; then apply_jobs alone."""
import sys, os, time, io, json, types, cProfile, pstats
AC = "/Users/michaelsrouji/Desktop/AutoCroissant"
TTS = "/Users/michaelsrouji/Desktop/TTSCardMaker"
sys.path[:0] = [AC, AC + "/tests"]
try:
    import requests  # noqa
except ImportError:
    class _Exc(Exception): pass
    def _no(*a, **k): raise RuntimeError("network in profile")
    m = types.ModuleType("requests"); m.RequestException = m.HTTPError = m.ConnectionError = m.Timeout = _Exc
    m.get = m.request = m.post = _no
    m.__getattr__ = lambda name: _Exc
    ex = types.ModuleType("requests.exceptions"); ex.__getattr__ = lambda name: _Exc
    m.exceptions = ex; sys.modules["requests"] = m; sys.modules["requests.exceptions"] = ex
fake = types.ModuleType("config"); fake.TOKEN = fake.GIT_TOKEN = ""; fake.ADMINS = []; fake.browser_cookies = False
sys.modules["config"] = fake
os.environ["GITHUB_TOKEN"] = "t"; os.environ["RUNNER_TEMP"] = "/tmp/acperf/work"; os.environ["PSD_TOOLS_FONT_DIRS"] = TTS + "/MDW/Fonts"

import test_card_bake as tb
import commands.card_bake as cb, commands.psd_edit as pe, commands.github_api as ga
from psd_tools import PSDImage, __version__
print("psd-tools", __version__, "text engine:", cb._text_engine(), "python", sys.version.split()[0])

acc = {}
def timed(name, fn):
    def w(*a, **k):
        t = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            c, s = acc.get(name, (0, 0.0)); acc[name] = (c + 1, s + time.perf_counter() - t)
    return w
for mod, names in ((cb, ["fetch_verified", "_tree", "_recent", "vocabulary", "land", "_commit_info", "_ref_head"]),
                   (pe, ["parse_card", "classify", "set_stat", "set_text_layer", "baseline", "render_region", "merge_png",
                         "gate_shape", "gate_region", "gate_size", "gate_bytes", "gate_reparse", "gate_idempotence",
                         "save_psd", "save_png", "artifacts", "layer_records", "status_fills"])):
    for n in names: setattr(mod, n, timed(f"{mod.__name__.split('.')[-1]}.{n}", getattr(mod, n)))
baked = []
_apply = pe.apply_jobs
def apply_jobs(*a, **k):
    out = _apply(*a, **k); baked.append(out); return out
pe.apply_jobs = timed("psd_edit.apply_jobs", apply_jobs)
_open = PSDImage.open
PSDImage.open = staticmethod(timed("PSDImage.open", _open))

from golden_cards import VOCAB
clone = tb.CLONE
vocab = json.loads(VOCAB.read_text(encoding="utf-8"))
files = {f"Types/{t}.png": t.encode() for t in vocab["all_types"]}
files.update({f"Types/Stars/{s}.png": s.encode() for s in vocab["stars"]})
files[cb.ICON_MANIFEST_PATH] = json.dumps(vocab["manifest"]).encode()
files.update(tb._card_files(clone, tb.DOOMER))
files[cb.EDITS_PATHS[0]] = tb._edits({"Mini Doomer": {"hp": 8, "note": "checked", "_by": "web", "_at": tb.AT}})
print("Doomer PSD %.1f MB, PNG %.1f MB" % (len(files[tb.DOOMER]) / 1e6, len(files[tb.DOOMER[:-4] + '.png']) / 1e6))

def setup():
    fake_gh = tb.FakeGitHub(cb.CARD_REPO, files)
    ga.get, ga.GIT_TOKEN, ga.HEADERS = fake_gh.get, "", {}
    cb.request, cb.refresh, cb._cant_write = fake_gh.request, (lambda: True), False
    return fake_gh

for label in ("run 1 (cold imports)", "run 2 (warm)"):
    acc.clear(); baked.clear(); setup()
    t = time.perf_counter(); text, commits = cb.run(dry_run=True, card="Mini Doomer"); wall = time.perf_counter() - t
    print(f"\n== {label}: cb.run dry-run --card 'Mini Doomer': {wall:.2f} s wall; commits {commits}")
    print("   report line:", [l for l in text.splitlines() if l.startswith("- **")][0][:300])
    for name, (c, s) in sorted(acc.items(), key=lambda kv: -kv[1][1]):
        print(f"   {s:7.3f} s  x{c:<3d} {name}")
    if baked: print("   Baked.timings:", {k: round(v, 3) for k, v in baked[0].timings.items()})

# cProfile of one run
acc.clear(); baked.clear(); setup()
pr = cProfile.Profile(); pr.enable(); cb.run(dry_run=True, card="Mini Doomer"); pr.disable()
s = io.StringIO(); st = pstats.Stats(pr, stream=s); st.sort_stats("cumulative").print_stats(r"psd_tools|commands", 45)
print("\n== cProfile (cumulative, psd_tools + commands):"); print("\n".join(s.getvalue().splitlines()[6:60]))
s = io.StringIO(); st = pstats.Stats(pr, stream=s); st.sort_stats("tottime").print_stats(18)
print("\n== cProfile (tottime):"); print("\n".join(s.getvalue().splitlines()[6:32]))

# apply_jobs alone: hp toggle with and without preview; ability edit
psd, png = "/tmp/acperf/doomer/Mini_Doomer.psd", "/tmp/acperf/doomer/Mini_Doomer.png"
parser = pe.offline_parser(vocab["all_types"], vocab["stars"], vocab["manifest"])
parsed = pe.parse_card(parser, psd, tb.DOOMER)
print("\nDoomer ability:", repr(parsed["card"].get("ability"))[:200])
for update_preview in (True, False):
    acc.clear(); out = f"/tmp/acperf/work/direct_{update_preview}"; os.makedirs(out, exist_ok=True)
    t = time.perf_counter()
    b = _apply(psd, png, tb.DOOMER, {"hp": 8}, parsed, parser, out, TTS + "/MDW/Fonts", None, update_preview=update_preview)
    print(f"\n== apply_jobs hp 5->8, update_preview={update_preview}: {time.perf_counter() - t:.2f} s; timings {dict((k, round(v, 3)) for k, v in b.timings.items())}; "
          f"new PSD {os.path.getsize(b.psd_path) / 1e6:.1f} MB; region {b.region}")
    for name, (c, s) in sorted(acc.items(), key=lambda kv: -kv[1][1])[:12]: print(f"   {s:7.3f} s  x{c:<3d} {name}")
chart_path = os.path.join(TTS, pe.STATUS_CHART)
chart = pe.status_fills(_open(chart_path)) if os.path.exists(chart_path) else None
print("\nchart statuses:", len(chart or {}))
old = parsed["card"].get("ability") or ""
new_text = old.replace("the", "this", 1) if "the" in old else old + " Now."
acc.clear(); out = "/tmp/acperf/work/direct_ability"; os.makedirs(out, exist_ok=True)
t = time.perf_counter()
try:
    b = _apply(psd, png, tb.DOOMER, {"ability": new_text}, parsed, parser, out, TTS + "/MDW/Fonts", chart, update_preview=True)
    print(f"\n== apply_jobs ability edit: {time.perf_counter() - t:.2f} s; timings {dict((k, round(v, 3)) for k, v in b.timings.items())}; region {b.region}; parity {b.parity}")
except Exception as e:
    print(f"\n== apply_jobs ability edit: {time.perf_counter() - t:.2f} s -> {type(e).__name__}: {e}")
for name, (c, s) in sorted(acc.items(), key=lambda kv: -kv[1][1])[:14]: print(f"   {s:7.3f} s  x{c:<3d} {name}")
