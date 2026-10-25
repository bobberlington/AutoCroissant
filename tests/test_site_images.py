"""
The website's pictures (TTSCardMaker site/build/build_site.py, owner 2026-10-24): card images are AVIF only (4:4:4, by
PNG blob and encode settings, encoded once), a build out of IMAGE_BUDGET leaves the rest to the next run, icon file names
carry a stamp, build.txt digests the site's files only, and there is no service worker. Loads the site from $AC_SITE_SRC
(a TTSCardMaker checkout) or the clone; skips before AVIF thumbnails.
"""
import json
import re
import shutil
from concurrent.futures import ThreadPoolExecutor
from subprocess import run as run_process

import pytest

from test_site_build import ALPHA, _build, _commit, _png, bs, repo  # noqa: F401  (fixtures)
from test_site_speed import ASSETS, SITE

BUILD = SITE / "build" / "build_site.py"
pytestmark = pytest.mark.skipif(not BUILD.is_file() or "def thumb_tag(" not in BUILD.read_text(encoding="utf-8"),
                                reason=f"no AVIF thumbs in {BUILD} (TTSCardMaker before work/site-img)")


def test_card_images_are_avif_made_once(bs, repo, tmp_path, monkeypatch):  # noqa: F811
    """Three AVIF widths (named <blob16>-<w>-<settings tag>, the tags in cards.json) and the link-preview JPEG per PNG
    blob, no WebP; a second build makes nothing again."""
    from PIL import Image
    assert bs.AVIF["subsampling"] == "4:4:4" and set(bs.IMAGE_SIZES) == {240, 360, 720}
    data = _build(bs, monkeypatch, repo, tmp_path / "s")
    thumbs = tmp_path / "s" / "dist" / "thumbs"
    imgs = {c["img"] for c in data["cards.json"]["cards"] if c.get("img")}
    tags = data["cards.json"]["thumbs"]
    assert tags == {str(w): bs.thumb_tag(w) for w in (240, 360, 720)} and len(set(tags.values())) == 3
    assert imgs and not list(thumbs.glob("*.webp"))
    for img in imgs:
        for w in (240, 360, 720):
            with Image.open(thumbs / f"{img}-{w}-{tags[str(w)]}.avif") as im:
                assert im.format == "AVIF" and im.width == min(w, 60)      # the fixture's PNGs are 60 px wide
        assert (thumbs / f"{img}-og.jpg").is_file()
    cache = tmp_path / "s" / ".cache" / "img"
    before = {p.name: p.stat().st_mtime_ns for p in cache.iterdir()}
    monkeypatch.setattr(bs.Image.Image, "save", lambda *a, **k: pytest.fail("an image made again"))
    _build(bs, monkeypatch, repo, tmp_path / "s")
    assert {p.name: p.stat().st_mtime_ns for p in cache.iterdir()} == before


def test_a_downloaded_png_must_be_its_blob(bs, tmp_path, monkeypatch):  # noqa: F811
    """A PNG not in the checkout is read at the build's commit (raw main is a CDN copy minutes old), then main, and
    must hash to the blob the thumbs are named by: never another version's picture under its name."""
    import hashlib
    old, new = b"old picture", b"new picture"
    blob = hashlib.sha1(b"blob %d\0" % len(new) + new).hexdigest()
    assert bs.blob_sha(new) == blob
    monkeypatch.setattr(bs, "REPO", str(tmp_path))
    monkeypatch.setattr(bs, "_card_raw", "https://raw.test/abc/")
    got = []
    def fetch(url, **_):
        got.append(url)
        return served["main" if url.startswith(bs.RAW) else "abc"]
    monkeypatch.setattr(bs, "fetch", fetch)
    served = {"abc": new, "main": old}
    assert bs.image_bytes("A/B c.png", blob) == new and got == ["https://raw.test/abc/A/B%20c.png"]
    served = {"abc": old, "main": old}
    with pytest.raises(ValueError, match="is not blob"):
        bs.image_bytes("A/B c.png", blob)
    assert got[1:] == ["https://raw.test/abc/A/B%20c.png", bs.RAW + "A/B%20c.png"]


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
    assert len(new) == 4 and all(re.fullmatch(r"[0-9a-f]{16}-(240|360|720)-[0-9a-f]{6}\.avif|[0-9a-f]{16}-og\.jpg", n) for n in new)


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


def test_new_encode_settings_make_new_files(bs, repo, tmp_path, monkeypatch):  # noqa: F811
    """A width's quality is in its tag (file name): changing it re-encodes that width only, under new names (so no cache
    keeps the old bytes), and prunes the old ones; the accent colour comes from the PNG, so the
    codec doesn't move it."""
    data = _build(bs, monkeypatch, repo, tmp_path / "s")
    cache = tmp_path / "s" / ".cache" / "img"
    before = {p.name: p.stat().st_mtime_ns for p in cache.iterdir()}
    monkeypatch.setitem(bs.IMAGE_SIZES, 240, bs.IMAGE_SIZES[240] + 7)
    monkeypatch.setattr(bs, "IMAGE_META_VERSION", bs.IMAGE_META_VERSION + "x")   # the colours are computed again too
    again = _build(bs, monkeypatch, repo, tmp_path / "s")
    after = {p.name: p.stat().st_mtime_ns for p in cache.iterdir()}
    old, new = data["cards.json"]["thumbs"], again["cards.json"]["thumbs"]
    assert old["240"] != new["240"] and old["360"] == new["360"] and old["720"] == new["720"]
    assert not any(f"-240-{old['240']}." in n for n in after) and sum(f"-240-{new['240']}." in n for n in after) == sum(f"-240-{old['240']}." in n for n in before)
    assert {n: t for n, t in after.items() if "-240-" not in n} == {n: t for n, t in before.items() if "-240-" not in n}   # not made again
    colour = lambda d: {c["id"]: c.get("color") for c in d["cards.json"]["cards"]}
    assert colour(again) == colour(data)


def test_the_webp_builds_colours_are_kept(bs, repo, tmp_path, monkeypatch):  # noqa: F811
    """images.json rows from the WebP builds (IMAGE_META_WEBP) keep each blob's accent colour (theme-color, embed bars)
    while its AVIF files are made; rows of any other version are dropped."""
    data = _build(bs, monkeypatch, repo, tmp_path / "s")
    shutil.rmtree(tmp_path / "s" / ".cache" / "img")
    path = tmp_path / "s" / ".cache" / bs.IMAGE_META
    meta = json.loads(path.read_text(encoding="utf-8"))
    rows = {sha: [w, h, "#123456"] for sha, (w, h, _) in meta["images"].items()}
    for version, want in ((bs.IMAGE_META_WEBP, "#123456"), ("0:360:80", None)):
        (tmp_path / "s" / ".cache" / "img").mkdir(exist_ok=True)
        path.write_text(json.dumps({"version": version, "images": rows}), encoding="utf-8")
        again = _build(bs, monkeypatch, repo, tmp_path / "s")
        colours = {c["id"]: c.get("color") for c in again["cards.json"]["cards"] if c.get("img")}
        assert colours and all(v == (want or {c["id"]: c.get("color") for c in data["cards.json"]["cards"]}[k]) for k, v in colours.items()), version
        assert json.loads(path.read_text(encoding="utf-8"))["version"] == bs.IMAGE_META_VERSION
        shutil.rmtree(tmp_path / "s" / ".cache" / "img")


def test_build_txt_digests_the_site_not_the_script(bs, tmp_path, monkeypatch):  # noqa: F811
    """build.txt (site.yml's deploy gate) changes with any file's bytes, a thumb's too (same name and size), never with
    build_site.py's own text."""
    import importlib.util
    dist = tmp_path / "dist"
    (dist / "thumbs").mkdir(parents=True)
    (dist / "thumbs" / "a-240-aaaaaa.avif").write_bytes(b"one")
    (dist / "index.html").write_text("<p>hi</p>")
    monkeypatch.setattr(bs, "DIST", str(dist))
    first = bs.dist_digest({})
    edited = tmp_path / "build_site.py"
    edited.write_text(open(bs.__file__, encoding="utf-8").read() + "\n# a comment\n", encoding="utf-8")
    spec = importlib.util.spec_from_file_location("site_build_edited", edited)
    other = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(other)
    other.DIST = str(dist)
    assert other.dist_digest({}) == first
    (dist / "thumbs" / "a-240-aaaaaa.avif").write_bytes(b"two")
    assert bs.dist_digest({}) != first


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


def test_the_pages_preload_avif_and_no_worker(bs, repo, tmp_path, monkeypatch):  # noqa: F811
    """The pages preload AVIF only, and the build writes no service worker (owner 2026-10-24: it cost first paint)."""
    _build(bs, monkeypatch, repo, tmp_path / "s")
    dist = tmp_path / "s" / "dist"
    assert not (dist / "sw.js").exists() and not (SITE / "sw.js").exists()
    for page in [dist / "index.html", *dist.glob("c/*/index.html")]:
        assert ".webp" not in page.read_text(encoding="utf-8"), page


def test_icon_urls_carry_their_pictures_stamp(tmp_path, monkeypatch):
    """icons/<name>.<stamp>.webp: the stamp follows the icon's blob (and its variants' drawing) and is in the file's name,
    so an old page's URL gets that picture or a 404, never another deploy's."""
    import importlib.util
    import test_token_picker as tp
    for name in ("numpy", "cv2", "fontTools", "PIL"):
        pytest.importorskip(name)
    spec = importlib.util.spec_from_file_location("site_build_icons", tp.BUILD_SITE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root, tree = _icon_repo(tp, tmp_path)
    icons, _ = tp._icons(module, monkeypatch, root, tree, tmp_path / "a")
    stamp = re.fullmatch(r"icons/fire\.([0-9a-f]{10})\.webp", icons["fire"]["file"]).group(1)
    assert (tmp_path / "a" / "dist" / icons["fire"]["file"]).is_file() and not (tmp_path / "a" / "dist" / "icons" / "fire.webp").exists()
    assert {p.name for p in (tmp_path / "a" / "dist" / "icons").glob(f"fire.{stamp}.*.webp")} == {f"fire.{stamp}.{v}.webp" for v in tp.ALL}
    again, _ = tp._icons(module, monkeypatch, root, tree, tmp_path / "a")
    assert again["fire"]["file"] == icons["fire"]["file"]
    moved, _ = tp._icons(module, monkeypatch, root, {**tree, "Types/Fire.png": "d" * 40}, tmp_path / "b")
    assert moved["fire"]["file"] != icons["fire"]["file"] and stamp not in moved["fire"]["file"]
    assert not list((tmp_path / "b" / "dist" / "icons").glob(f"*{stamp}*"))                  # the old stamp's URL: a 404
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


def test_the_site_asks_for_avif():
    """No thumbnail is named .webp anywhere in the front end; the variant pictures keep the icon's stamp; app.js registers
    no service worker; the Changes line says what its time is."""
    for js in ASSETS.glob("*.js"):
        text = js.read_text(encoding="utf-8")
        assert not re.search(r"thumbs/[^\"'`]*\.webp|-(240|360|720)\\?\.webp", text), js.name
    app = (ASSETS / "app.js").read_text(encoding="utf-8")
    assert "serviceWorker" not in app and "sw.js" not in app
    assert "Site updated ${" in app and "the last build that changed it" in app
    if not shutil.which("node"):
        return
    script = f"""const T = await import({json.dumps((ASSETS / 'text.js').as_uri())});
process.stdout.write(JSON.stringify([T.variantFile({{ file: "icons/fire.0a1b2c3d4e.webp" }}, "glow"), T.variantFile({{ file: "icons/fire.webp" }}, "tint")]));"""
    done = run_process(["node", "--input-type=module", "-e", script], capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout) == ["icons/fire.0a1b2c3d4e.glow.webp", "icons/fire.tint.webp"]
