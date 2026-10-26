"""
Card grids: many cards' pictures as a few composite images, each card numbered and named under it (Discord previews
only a few image links per message). Card pictures come from their raw image links, downloaded once per PNG blob and
kept small in CACHE_DIR. Runs in worker threads; no Discord here.
"""
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from os import getpid, listdir, makedirs, remove, replace
from os.path import isfile, join
from typing import Optional

from PIL import Image, ImageDraw, ImageFont
from requests import get

# ========================
# Configuration
# ========================
COLUMNS, ROWS = 4, 5                                     # 20 cards per grid: readable on a phone once opened
PER_GRID = COLUMNS * ROWS
CARD_W = 300                                             # px; cards are 1200 wide, 2:3 or 1:2
LABEL_H, GAP, FONT_SIZE = 34, 8, 20
BACKGROUND, TEXT = (43, 45, 49), (242, 243, 245)        # Discord's dark theme
QUALITY = 88                                             # grid JPEG (4:4:4, so card text stays sharp)
CACHE_DIR = "card_images"                                # gitignored: <blob>.jpg, each card at CARD_W
POOL = 4                                                 # parallel downloads


def _font():
    try:
        return ImageFont.load_default(FONT_SIZE)         # FreeType's scalable default (Pillow >= 10.1)
    except (TypeError, OSError, ImportError):
        return ImageFont.load_default()


FONT = _font()


# ========================
# Card pictures
# ========================
def picture(url: Optional[str], blob: str = "") -> Optional[Image.Image]:
    """A card's picture at CARD_W: from the cache by its PNG blob, else downloaded (and cached). None if unavailable."""
    path = join(CACHE_DIR, f"{blob}.jpg") if blob else ""
    if path and isfile(path):
        try:
            with Image.open(path) as cached:
                return cached.convert("RGB")
        except OSError:
            pass                                         # a bad file: fetched again
    if not url:
        return None
    try:
        reply = get(url, timeout=30)
        reply.raise_for_status()
        with Image.open(BytesIO(reply.content)) as full:
            small = _shrink(full)
    except Exception as e:
        print(f"Card grid: no picture from {url}: {type(e).__name__}: {e}")
        return None
    if path:
        try:
            makedirs(CACHE_DIR, exist_ok=True)
            tmp = f"{path}.{getpid()}.tmp"
            small.save(tmp, "JPEG", quality=92)
            replace(tmp, path)
        except OSError as e:
            print(f"Card grid: not cached ({e})")
    return small


def _shrink(image: Image.Image) -> Image.Image:
    """image at CARD_W wide, RGB, transparency over the background."""
    factor = image.width // (CARD_W * 2)
    if factor > 1:
        image = image.reduce(factor)                     # fast integer step before the smooth resize
    if image.mode != "RGB":
        rgba = image.convert("RGBA")
        image = Image.new("RGB", rgba.size, BACKGROUND)
        image.paste(rgba, mask=rgba.getchannel("A"))
    return image.resize((CARD_W, max(1, round(image.height * CARD_W / image.width))), Image.LANCZOS)


def prune(keep: set) -> None:
    """Drop cached pictures whose blob is not in keep (cards changed or gone)."""
    try:
        names = listdir(CACHE_DIR)
    except OSError:
        return
    for name in names:
        if name.endswith(".jpg") and name[:len(name) - 4] not in keep:
            try:
                remove(join(CACHE_DIR, name))
            except OSError:
                pass


# ========================
# Grids
# ========================
def _label(draw: ImageDraw.ImageDraw, text: str, width: int) -> str:
    """text cut to width with an ellipsis."""
    if draw.textlength(text, font=FONT) <= width:
        return text
    while text and draw.textlength(text + "…", font=FONT) > width:
        text = text[:len(text) - 1]
    return text + "…"


def render(cards: list) -> bytes:
    """
    One grid image.

    Args:
        cards: up to PER_GRID (label, picture or None) in reading order

    Returns:
        JPEG bytes
    """
    rows = [cards[i:i + COLUMNS] for i in range(0, len(cards), COLUMNS)]
    heights = [max((p.height if p else CARD_W * 3 // 2) for _, p in row) for row in rows]
    columns = min(COLUMNS, len(cards))
    grid = Image.new("RGB", (columns * (CARD_W + GAP) + GAP,
                             sum(h + LABEL_H + GAP for h in heights) + GAP), BACKGROUND)
    draw = ImageDraw.Draw(grid)
    y = GAP
    for row, height in zip(rows, heights):
        for col, (label, pic) in enumerate(row):
            x = GAP + col * (CARD_W + GAP)
            if pic:
                grid.paste(pic, (x, y))
            else:                                        # no picture: an outlined space
                draw.rectangle((x, y, x + CARD_W - 1, y + height - 1), outline=TEXT, width=2)
            draw.text((x + CARD_W // 2, y + height + LABEL_H // 2), _label(draw, label, CARD_W - 4),
                      font=FONT, fill=TEXT, anchor="mm")
        y += height + LABEL_H + GAP
    out = BytesIO()
    grid.save(out, "JPEG", quality=QUALITY, subsampling=0, optimize=True)
    return out.getvalue()


def grids(cards: list):
    """
    Grid images, PER_GRID cards each, yielded as each is ready (pictures downloaded POOL at a time, in order).

    Args:
        cards: (label, image url or None, PNG blob or "")

    Yields:
        JPEG bytes per grid
    """
    with ThreadPoolExecutor(POOL) as pool:
        pictures = pool.map(lambda card: picture(card[1], card[2]), cards)
        chunk = []
        for (label, _, _), pic in zip(cards, pictures):
            chunk.append((label, pic))
            if len(chunk) == PER_GRID:
                yield render(chunk)
                chunk = []
        if chunk:
            yield render(chunk)
