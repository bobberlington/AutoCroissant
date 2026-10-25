"""
The website's pictures (TTSCardMaker site/build/build_site.py, site/sw.js, owner 2026-10-24): card images are AVIF only
(4:4:4, by PNG blob, encoded once), a build out of IMAGE_BUDGET leaves the rest to the next run, icon URLs carry a
stamp, and the service worker keeps only those versioned pictures (cache first) and lets pages, data and scripts reach
the network. Loads the site from $AC_SITE_SRC (a TTSCardMaker checkout) or the clone; skips before AVIF thumbnails.
"""
import json
import re
import shutil
from concurrent.futures import ThreadPoolExecutor
from subprocess import run as run_process

import pytest

from test_site_build import ALPHA, _build, _commit, _png, bs, repo  # noqa: F401  (fixtures)
from test_site_speed import ASSETS, SITE

SW = SITE / "sw.js"
pytestmark = pytest.mark.skipif(not SW.is_file(), reason=f"no {SW} (TTSCardMaker before work/site-img)")


def test_card_images_are_avif_made_once(bs, repo, tmp_path, monkeypatch):  # noqa: F811
    """Three AVIF widths and the link-preview JPEG per PNG blob, no WebP; a second build makes nothing again."""
    from PIL import Image
    assert bs.AVIF["subsampling"] == "4:4:4" and set(bs.IMAGE_SIZES) == {240, 360, 720}
    data = _build(bs, monkeypatch, repo, tmp_path / "s")
    thumbs = tmp_path / "s" / "dist" / "thumbs"
    imgs = {c["img"] for c in data["cards.json"]["cards"] if c.get("img")}
    assert imgs and not list(thumbs.glob("*.webp"))
    for img in imgs:
        for w in (240, 360, 720):
            with Image.open(thumbs / f"{img}-{w}.avif") as im:
                assert im.format == "AVIF" and im.width == min(w, 60)      # the fixture's PNGs are 60 px wide
        assert (thumbs / f"{img}-og.jpg").is_file()
    cache = tmp_path / "s" / ".cache" / "img"
    before = {p.name: p.stat().st_mtime_ns for p in cache.iterdir()}
    monkeypatch.setattr(bs.Image.Image, "save", lambda *a, **k: pytest.fail("an image made again"))
    _build(bs, monkeypatch, repo, tmp_path / "s")
    assert {p.name: p.stat().st_mtime_ns for p in cache.iterdir()} == before


def test_a_build_over_the_image_budget_leaves_the_rest(bs, repo, tmp_path, monkeypatch):  # noqa: F811
    """Over IMAGE_BUDGET only the images not cached yet wait: the build fails (site.yml then saves the cache), and the
    next run makes just those."""
    _build(bs, monkeypatch, repo, tmp_path / "s")
    cache = tmp_path / "s" / ".cache" / "img"
    made = {p.name for p in cache.iterdir()}
    _png(repo["root"] / f"{ALPHA}.png", 99)                    # one card's image changes
    _commit(repo["root"], "New art", 1_790_000_000)
    monkeypatch.setattr(bs, "IMAGE_BUDGET", 1e-9)
    with pytest.raises(SystemExit, match=r"1 of \d+ card images not made within IMAGE_BUDGET"):
        _build(bs, monkeypatch, repo, tmp_path / "s")
    assert {p.name for p in cache.iterdir()} == made
    monkeypatch.setattr(bs, "IMAGE_BUDGET", 0)
    _build(bs, monkeypatch, repo, tmp_path / "s")
    new = {p.name for p in cache.iterdir()} - made
    assert len(new) == 4 and all(re.fullmatch(r"[0-9a-f]{16}-(240|360|720)\.avif|[0-9a-f]{16}-og\.jpg", n) for n in new)


def test_thumbs_carry_no_colour_profile(bs, repo, tmp_path, monkeypatch):  # noqa: F811
    """Pillow's AVIF writer copies the PNG's ICC profile (about 3 KB, +34% at 240 px); the thumbs drop it, as WebP did."""
    from PIL import Image, ImageCms
    srgb = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    Image.new("RGB", (60, 84), (120, 40, 90)).save(repo["root"] / f"{ALPHA}.png", icc_profile=srgb)
    _commit(repo["root"], "Tagged art", 1_790_000_000)
    with Image.open(repo["root"] / f"{ALPHA}.png") as im:
        assert im.info.get("icc_profile")
    _build(bs, monkeypatch, repo, tmp_path / "s")
    thumbs = sorted((tmp_path / "s" / "dist" / "thumbs").glob("*.avif"))
    assert thumbs
    for path in thumbs:
        with Image.open(path) as im:
            assert not im.info.get("icc_profile"), path.name


def test_a_stopped_image_phase_keeps_what_it_made(bs, repo, tmp_path, monkeypatch):  # noqa: F811
    """Whatever stops the image pool (here an interrupt), images.json is written for what was made, so the cache site.yml
    saves after a failed build is a full head start."""
    made, real = [], bs.make_images
    def once(path, sha, known=None):
        if made:
            raise KeyboardInterrupt
        made.append(sha)
        return real(path, sha, known)
    monkeypatch.setattr(bs, "make_images", once)
    monkeypatch.setattr(bs, "ThreadPoolExecutor", lambda n: ThreadPoolExecutor(1))
    with pytest.raises(KeyboardInterrupt):
        _build(bs, monkeypatch, repo, tmp_path / "s")
    meta = json.loads((tmp_path / "s" / ".cache" / bs.IMAGE_META).read_text(encoding="utf-8"))
    assert list(meta["images"]) == made


def test_the_worker_lists_the_pictures_in_use(bs, repo, tmp_path, monkeypatch):  # noqa: F811
    """dist/sw.js: site/sw.js with every card and rulebook image's blob (its first 10 digits) and icon stamp, as one
    string; the pages preload AVIF only."""
    data = _build(bs, monkeypatch, repo, tmp_path / "s")
    dist = tmp_path / "s" / "dist"
    text = (dist / "sw.js").read_text(encoding="utf-8")
    joined = json.loads(re.search(r'const IN_USE = new Set\(("[0-9a-f]*")\.match', text).group(1))
    listed = [joined[i:i + 10] for i in range(0, len(joined), 10)]
    icons = {i["file"].split("?v=")[1] for i in data["cards.json"]["icons"].values()}
    assert set(listed) == {c["img"][:10] for c in data["cards.json"]["cards"] if c.get("img")} | icons and listed == sorted(set(listed))
    assert "/*PICTURES*/" not in text
    for page in [dist / "index.html", *dist.glob("c/*/index.html")]:
        assert ".webp" not in page.read_text(encoding="utf-8"), page


def test_icon_urls_carry_their_pictures_stamp(tmp_path, monkeypatch):
    """icons/<name>.webp?v=<stamp>: the stamp follows the icon's blob (and its variants' drawing), so the worker can keep
    the URL for good; the files keep their names."""
    import importlib.util
    import test_token_picker as tp
    for name in ("numpy", "cv2", "fontTools", "PIL"):
        pytest.importorskip(name)
    spec = importlib.util.spec_from_file_location("site_build_icons", tp.BUILD_SITE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root, tree = _icon_repo(tp, tmp_path)
    icons, _ = tp._icons(module, monkeypatch, root, tree, tmp_path / "a")
    stamp = re.fullmatch(r"icons/fire\.webp\?v=([0-9a-f]{10})", icons["fire"]["file"]).group(1)
    assert (tmp_path / "a" / "dist" / "icons" / "fire.webp").is_file()
    again, _ = tp._icons(module, monkeypatch, root, tree, tmp_path / "a")
    assert again["fire"]["file"] == icons["fire"]["file"]
    moved, _ = tp._icons(module, monkeypatch, root, {**tree, "Types/Fire.png": "d" * 40}, tmp_path / "b")
    assert moved["fire"]["file"] != icons["fire"]["file"] and stamp not in moved["fire"]["file"]
    assert moved["new"]["file"] == icons["new"]["file"]


def _icon_repo(tp, tmp_path):
    """test_token_picker's icon_repo fixture body (pytest fixtures can't be called directly)."""
    from PIL import Image
    root = tmp_path / "repo"
    (root / "Types" / "Stars").mkdir(parents=True)
    Image.new("RGBA", (60, 60), (240, 120, 20, 255)).save(root / "Types" / "Fire.png")
    Image.new("RGBA", (60, 30), (250, 210, 40, 255)).save(root / "Types" / "Stars" / "Gold Star.png")
    Image.new("RGBA", (50, 50), (0, 200, 0, 255)).save(root / "Types" / "New.png")
    src = root / "MDW" / "Fonts" / "stoogedings-src"
    src.mkdir(parents=True)
    shutil.copy(tp.FONT_BUILD, src / tp.FONT_BUILD.name)
    manifest = {"icons": {"fire": {"kind": "type", "file": "Types/Fire.png", "pua": "E000", "aliases": [], "tint": "auto"}}}
    (src / "stoogedings.json").write_text(json.dumps(manifest))
    return root, {"Types/Fire.png": "a" * 40, "Types/Stars/Gold Star.png": "b" * 40, "Types/New.png": "c" * 40}


def node(tmp_path, body: str):
    script = tmp_path / "probe.mjs"
    script.write_text(body, encoding="utf-8")
    done = run_process(["node", str(script)], capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


MISSING = "https://x.test/site/icons/missing.webp?v=0123456789"
SW_HARNESS = r"""
import fs from "node:fs";
const IN_USE = __IN_USE__;
const src = fs.readFileSync(__SW__, "utf8").replace('/*PICTURES*/""', JSON.stringify(IN_USE.map(n => n.slice(0, 10)).join("")));
const store = new Map(), fetched = [];
let broken = false;
const cacheOf = m => ({ match: async u => m.get(u), put: async (u, r) => { m.set(u, r); }, keys: async () => [...m.keys()].map(url => ({ url })),
  delete: async r => m.delete(r.url ?? r) });
const caches = { open: async n => { if (broken) throw new Error("no storage"); if (!store.has(n)) store.set(n, new Map()); return cacheOf(store.get(n)); },
  keys: async () => [...store.keys()], delete: async n => store.delete(n) };
const fetch = async r => { const url = typeof r === "string" ? r : r.url; fetched.push(url);
  return { status: url.includes("missing") ? 404 : 200, type: "basic", url, body: `net:${url}`, clone() { return { ...this, body: `copy:${url}` }; } }; };
const on = {};
const self = { registration: { scope: "https://x.test/site/" }, location: new URL("https://x.test/site/sw.js"), addEventListener: (t, f) => { on[t] = f; },
  skipWaiting: () => { self.skipped = true; }, clients: { claim: async () => { self.claimed = true; } } };
new Function("self", "caches", "fetch", src)(self, caches, fetch);
const event = extra => { const waits = []; return { waits, waitUntil: p => waits.push(p), ...extra }; };
async function get(url, method = "GET", headers = {}) {
  const e = event({ request: { url, method, headers: new Map(Object.entries(headers)) } });
  let res = null; e.respondWith = p => { res = p; };
  on.fetch(e);
  if (!res) return "network";
  const r = await res; await Promise.all(e.waits); return r.body;
}
const T = "https://x.test/site/thumbs/", out = {};
store.set("pictures-0", new Map([["old", 1]])); store.set("other-app", new Map());
store.set("pictures-1", new Map([[T + "ffffffffffffffff-240.avif", { body: "stale" }], [T + "aaaaaaaaaaaaaaaa-240.avif", { body: "kept" }]]));
on.install(event({}));
const a = event({}); on.activate(a); await Promise.all(a.waits);
out.activated = { caches: [...store.keys()].sort(), entries: [...store.get("pictures-1").keys()], skipped: !!self.skipped, claimed: !!self.claimed };
out.first = await get(T + "bbbbbbbbbbbbbbbb-720.avif");
out.second = await get(T + "bbbbbbbbbbbbbbbb-720.avif");
out.kept = await get(T + "aaaaaaaaaaaaaaaa-240.avif");
out.unused = [await get(T + "cccccccccccccccc-360.avif"), await get(T + "cccccccccccccccc-360.avif")];
out.icon = [await get("https://x.test/site/icons/fire.glow.webp?v=0123456789"), await get("https://x.test/site/icons/fire.glow.webp?v=0123456789")];
out.missing = [await get(T + "dddddddddddddddd-missing.avif"), await get("https://x.test/site/icons/missing.webp?v=0123456789")];
out.passed = await Promise.all(["https://x.test/site/", "https://x.test/site/index.html", "https://x.test/site/data/cards.json?v=abc",
  "https://x.test/site/assets/app.js?v=abc", "https://x.test/site/sw.js", "https://x.test/site/c/hornet/", "https://x.test/site/icons/fire.webp",
  T + "bbbbbbbbbbbbbbbb-720.webp", T + "bbbbbbbbbbbbbbbb-720.avif?x=1", T + "bbbbbbbbbbbbbbbb-og.jpg", "https://y.test/site/thumbs/bbbbbbbbbbbbbbbb-720.avif",
  "https://x.test/other/thumbs/bbbbbbbbbbbbbbbb-720.avif"].map(u => get(u)));
out.post = await get(T + "eeeeeeeeeeeeeeee-720.avif", "POST");
out.range = await get(T + "eeeeeeeeeeeeeeee-720.avif", "GET", { range: "bytes=0-9" });
const m = event({ data: { seen: [T + "eeeeeeeeeeeeeeee-240.avif", T + "bbbbbbbbbbbbbbbb-720.avif", T + "cccccccccccccccc-240.avif", "https://x.test/site/data/cards.json", "not a url"] } });
fetched.length = 0; on.message(m); await Promise.all(m.waits);
out.seen = { fetched: [...fetched], cached: [...store.get("pictures-1").keys()].sort() };
broken = true;
out.noStorage = await get(T + "eeeeeeeeeeeeeeee-360.avif");
process.stdout.write(JSON.stringify(out));
"""


def test_the_worker_keeps_versioned_pictures_and_nothing_else(tmp_path):
    """Cache first for thumbs/<blob>-<w>.avif and icons/<name>.webp?v=<stamp> the site uses; pages, data, scripts, sw.js,
    unstamped or foreign URLs, POST and Range requests never reach it. Activation deletes older picture caches and
    pictures no longer used (never another cache); seen pictures are put in; no storage means the network."""
    if not shutil.which("node"):
        pytest.skip("no node")
    in_use = ["aaaaaaaaaaaaaaaa", "bbbbbbbbbbbbbbbb", "eeeeeeeeeeeeeeee", "dddddddddddddddd", "0123456789"]
    out = node(tmp_path, SW_HARNESS.replace("__IN_USE__", json.dumps(in_use)).replace("__SW__", json.dumps(str(SW))))
    t = "https://x.test/site/thumbs/"
    assert out["activated"] == {"caches": ["other-app", "pictures-1"], "entries": [t + "aaaaaaaaaaaaaaaa-240.avif"], "skipped": True, "claimed": True}
    assert out["first"] == f"net:{t}bbbbbbbbbbbbbbbb-720.avif" and out["second"] == f"copy:{t}bbbbbbbbbbbbbbbb-720.avif"
    assert out["kept"] == "kept"
    assert out["unused"] == [f"net:{t}cccccccccccccccc-360.avif"] * 2                       # served, not kept: not in use
    icon = "https://x.test/site/icons/fire.glow.webp?v=0123456789"
    assert out["icon"] == [f"net:{icon}", f"copy:{icon}"]
    assert out["missing"] == ["network", f"net:{MISSING}"] and MISSING not in out["seen"]["cached"]   # a 404 is never kept
    assert out["passed"] == ["network"] * 12 and out["post"] == "network" and out["range"] == "network"
    assert out["seen"]["fetched"] == [t + "eeeeeeeeeeeeeeee-240.avif"]                         # in use, not cached yet
    assert t + "eeeeeeeeeeeeeeee-240.avif" in out["seen"]["cached"] and t + "cccccccccccccccc-240.avif" not in out["seen"]["cached"]
    assert out["noStorage"] == f"net:{t}eeeeeeeeeeeeeeee-360.avif"


def test_the_site_asks_for_avif_and_starts_the_worker():
    """No thumbnail is named .webp anywhere in the front end; the variant pictures keep the icon's stamp; app.js registers
    sw.js at the site root (share pages included: their <base> climbs to it); the Changes line says what its time is."""
    for js in ASSETS.glob("*.js"):
        text = js.read_text(encoding="utf-8")
        assert not re.search(r"thumbs/[^\"'`]*\.webp|-(240|360|720)\\?\.webp", text), js.name
    app = (ASSETS / "app.js").read_text(encoding="utf-8")
    assert 'navigator.serviceWorker.register(new URL("sw.js", ROOT).href)' in app and "startWorker();" in app
    assert "Site updated ${" in app and "the last build that changed it" in app
    if not shutil.which("node"):
        return
    script = f"""const T = await import({json.dumps((ASSETS / 'text.js').as_uri())});
process.stdout.write(JSON.stringify([T.variantFile({{ file: "icons/fire.webp?v=0a1b2c3d4e" }}, "glow"), T.variantFile({{ file: "icons/fire.webp" }}, "tint")]));"""
    done = run_process(["node", "--input-type=module", "-e", script], capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout) == ["icons/fire.glow.webp?v=0a1b2c3d4e", "icons/fire.tint.webp"]
