"""
Where things sit on a card PSD: the stat rows, name, ability, stars, type icons, art and frame as canvas bboxes
(x0, y0, x1, y1; x1/y1 exclusive) and as exclusive pixel masks. Port of psd-research score/regions.py; the edit
pipeline uses stat_rows, badge_layers and _fx_pad to find the pixels an edit can reach. No Discord, network or config.
"""
from re import IGNORECASE, fullmatch, search, sub

import numpy as np
from psd_tools import PSDImage

# ========================
# Configuration
# ========================
STATS = {"hp": r"hp|health", "def": r"def|defen[cs]e|defemse", "atk": r"atk|attack", "spd": r"spd|speed"}
STAT_LABELS = {"health", "defense", "attack", "speed"}
NAME_MAX_TOP = 0.15          # a title starts in the top 15% of the canvas (Wigfrid's at y=1784 is the one miss)
TYPE_REGION_RATIO = 0.5      # icons above the midline are header badges, below they sit in the ability (the parser's rule)
ROW_ANCESTORS = 4            # a row names its stat on itself or within 3 enclosing groups ("Darks" > "hp" > "1")
RULEBOOK_WIDTH = 1080        # rule pages are 1080 wide; every card is 1200
CARD_WIDTH = 1200
BAR_X = (360, 1173)          # the template bar's x range on 1200-wide cards
BAR_HALF_HEIGHT = 42         # template rows are 84 px tall, centred on the stat label
STAR_CORNER = (0.6, 0.2)     # loose stars: drawn in the top-right corner (x0 >= 60% W, y1 <= 20% H)
OVERLAY_INSIDE = 0.8         # an overlay word layer has 80% of its bbox inside the ability box
OVERLAY_MAX_CHARS = 40       # and is a short text (the longest status name is 17)
# later keys win when regions overlap (region_masks)
PAINT_ORDER = ["art", "ability_bg", "frame", "stat_labels", "stat_hp", "stat_def", "stat_atk", "stat_spd",
               "stars", "types", "inline_icons", "name", "ability", "ability_fx"]


# ========================
# Helper Functions
# ========================
def _ancestors(layer):
    """The enclosing groups, innermost first."""
    parent = layer.parent
    while parent is not None and not isinstance(parent, PSDImage):
        yield parent
        parent = parent.parent


def _fx_pad(layer) -> int:
    """
    Pixels an effect stack (the layer's and its groups') can reach beyond the layer bbox.

    Args:
        layer: Any layer

    Returns:
        Outside strokes added up, plus the largest shadow reach (distance + size) or glow size
    """
    stroke = shadow = 0
    for owner in (layer, *_ancestors(layer)):
        try:
            effects = list(owner.effects)
        except Exception:
            effects = []
        for effect in effects:
            if not effect.enabled:
                continue
            kind = type(effect).__name__
            if kind == "Stroke":
                position = str(getattr(effect, "position", ""))
                if "INSIDE" not in position:
                    stroke += int(effect.size if "OUTSIDE" in position else effect.size / 2 + 0.5)
            elif kind == "DropShadow":
                shadow = max(shadow, int(effect.distance + effect.size + 0.5))
            elif kind == "OuterGlow":
                shadow = max(shadow, int(effect.size + 0.5))
    return stroke + shadow


def _box(layer, width: int, height: int, pad: bool = True, bbox=None):
    """A layer's bbox (or `bbox`) padded by its effects and clipped to the canvas; None when empty."""
    x0, y0, x1, y1 = bbox or layer.bbox
    reach = _fx_pad(layer) if pad else 0
    box = (max(0, x0 - reach), max(0, y0 - reach), min(width, x1 + reach), min(height, y1 + reach))
    return box if box[2] > box[0] and box[3] > box[1] else None


def _union(boxes):
    boxes = [b for b in boxes if b]
    if not boxes:
        return None
    return (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))


def _leaves(group):
    for layer in group.descendants():
        if not layer.is_group() and layer.is_visible() and layer.bbox[2] > layer.bbox[0]:
            yield layer


def clip(box, width: int, height: int) -> tuple:
    """A bbox clipped to the canvas."""
    return (max(0, box[0]), max(0, box[1]), min(width, box[2]), min(height, box[3]))


# ========================
# Public API Functions
# ========================
def stat_rows(psd: PSDImage) -> dict:
    """
    The bar rows per stat.

    A row is a group with digit-named leaf children ("0".."9" exposure layers in the template) whose own name or an
    ancestor's names a stat ("Hp Dark", "hp" > "1", "shared health") under a *dark*/*bars* group.

    Args:
        psd: An open card

    Returns:
        {stat: [(row group, visible digits sorted, row bbox or None)]}; the bbox is the union of the digit layers'
        mask bboxes (the template's "0" mask covers the whole bar), else of their own bboxes
    """
    out = {key: [] for key in STATS}
    for group in psd.descendants():
        if not group.is_group():
            continue
        digits = [child for child in group if not child.is_group() and child.name.strip().isdigit()]
        if not digits:
            continue
        chain = [group, *_ancestors(group)][:ROW_ANCESTORS]
        if not any(search(r"dark|bars", owner.name, IGNORECASE) for owner in chain):
            continue
        key = next((k for owner in chain for k, pattern in STATS.items()
                    if search(rf"\b({pattern})\b", owner.name, IGNORECASE)), None)
        if key is None:
            continue
        masks = [tuple(c.mask.bbox) for c in digits if c.has_mask() and c.mask.bbox[2] > c.mask.bbox[0]]
        pixels = [tuple(c.bbox) for c in digits if c.bbox[2] > c.bbox[0]]
        out[key].append((group, sorted(int(c.name) for c in digits if c.is_visible()), _union(masks or pixels)))
    return out


def regions(psd: PSDImage, vocab=(), rulebook=None) -> dict:
    """
    Every region of a card as a list of bboxes.

    Args:
        psd: An open card
        vocab: Type icon names (lowercase Types/ stems); a drawn layer so named is an icon
        rulebook: Whether this is a rule page; None guesses (RULEBOOK_WIDTH or a "Page Number" layer)

    Returns:
        {region key: [bbox, ...]} over PAINT_ORDER's keys
    """
    width, height = psd.width, psd.height
    found = {key: [] for key in PAINT_ORDER}
    texts = [layer for layer in psd.descendants() if layer.kind == "type" and layer.is_visible()]
    if rulebook is None:
        rulebook = width == RULEBOOK_WIDTH or any(layer.name == "Page Number" for layer in psd)
    if rulebook:
        for layer in texts:
            key = "name" if layer.name.strip().lower() == "header" else "ability"
            found[key].append(_box(layer, width, height))

    abilities = []
    if not rulebook:
        # name: visible layer(s) named "Name" near the top, else top text layers that aren't labels or the ability
        named = [layer for layer in psd.descendants() if not layer.is_group() and layer.is_visible()
                 and layer.name.strip().lower() == "name" and layer.bbox[1] < NAME_MAX_TOP * height]
        if not named:
            named = [layer for layer in texts if layer.bbox[1] < NAME_MAX_TOP * height
                     and layer.name.strip().lower() not in STAT_LABELS | {"ability"}]
        found["name"] += [_box(layer, width, height) for layer in named]
        abilities = [layer for layer in texts if layer.name.strip().lower() == "ability"]
        if not abilities:   # a hidden "Ability" re-laid out in separate text layers (Butterfly, Bulborb Larva)
            abilities = [layer for layer in texts if layer.bbox[1] >= TYPE_REGION_RATIO * height
                         and layer.name.strip().lower() not in STAT_LABELS | {"name"}]
        found["ability"] += [_box(layer, width, height) for layer in abilities]
        # ability FX: pixel cuts "<text layer> FX: <words>", and short overlay text layers on the ability box
        for layer in psd.descendants():
            if " FX: " in layer.name and not layer.is_group() and layer.is_visible():
                found["ability_fx"].append(_box(layer, width, height))
        for layer in overlay_words(psd, abilities, texts):
            found["ability_fx"].append(_box(layer, width, height))
        found["stat_labels"] += [_box(layer, width, height) for layer in texts
                                 if layer.name.strip().lower() in STAT_LABELS]

    # stats: one bbox per bar row. 1) the digit layers' mask union; 2) visible leaves of a group named like the stat
    # under a *bar*/*stat* group (Isaac hearts, custom bars); 3) a layer named like the stat (+ "bar") inside "Bars";
    # 4) bars painted into the art: the template row centred on the stat label
    rows = stat_rows(psd)
    for key, pattern in STATS.items():
        boxes = [_box(group, width, height, bbox=b) for group, _, b in rows[key] if b]
        if not boxes:
            for group in psd.descendants():
                if (group.is_group() and fullmatch(rf"\s*({pattern})\s*(bars?)?\s*", group.name, IGNORECASE)
                        and any(search(r"bar|stat", owner.name, IGNORECASE) for owner in _ancestors(group))):
                    boxes.append(_union([_box(layer, width, height) for layer in _leaves(group)]))
        if not boxes:
            for layer in psd.descendants():
                parent = layer.parent
                if (not layer.is_group() and layer.is_visible() and parent is not None
                        and not isinstance(parent, PSDImage) and parent.name.strip().lower() == "bars"
                        and fullmatch(rf"\s*({pattern})\s*(bar)?\s*", layer.name, IGNORECASE)):
                    boxes.append(_box(layer, width, height))
        if not any(boxes) and width == CARD_WIDTH:
            for layer in texts:
                if fullmatch(pattern, layer.name.strip(), IGNORECASE) and layer.name.strip().lower() in STAT_LABELS:
                    centre = (layer.bbox[1] + layer.bbox[3]) // 2
                    boxes.append((BAR_X[0], centre - BAR_HALF_HEIGHT, BAR_X[1], centre + BAR_HALF_HEIGHT))
        found["stat_" + key] += [b for b in boxes if b]

    stars, icons, inline = badge_layers(psd, vocab, rulebook)
    found["stars"] += [_box(layer, width, height) for layer in stars]
    found["types"] += [_box(layer, width, height) for layer in icons]
    found["inline_icons"] += [_box(layer, width, height) for layer in inline]

    # art / frame: the LINE separator splits the art (above) from the ability panel (below) on 1200x2400 cards
    images = [layer for layer in psd if layer.is_group() and layer.name.strip().lower() == "images" and layer.is_visible()]
    line = next((layer for group in images for layer in _leaves(group) if layer.name.strip().lower() == "line"), None)
    art = _union([_box(layer, width, height, pad=False) for group in images for layer in _leaves(group)
                  if layer is not line])
    if line is not None:
        y0, y1 = max(0, line.bbox[1]), min(height, line.bbox[3])
        found["frame"].append((0, y0, width, y1))
        if art:
            found["art"].append((art[0], art[1], art[2], min(art[3], y0)))
        found["ability_bg"].append((0, y1, width, height))
    elif art:
        found["art"].append(art)
    elif not rulebook:
        found["art"].append((0, 0, width, height))
    for group in psd.descendants():
        if group.is_group() and group.name.strip().lower() == "lines" and group.is_visible():
            found["frame"] += [_box(layer, width, height) for layer in _leaves(group)]
    return {key: [b for b in boxes if b] for key, boxes in found.items()}


def badge_layers(psd: PSDImage, vocab=(), rulebook=None) -> tuple:
    """
    The drawn star and type icon layers.

    Stars: visible leaves in the outermost visible groups named *stars*, else leaves named with the word star
    (Warpstar, Patrick Star, star copy 2), else anything drawn in the top-right corner outside the images/types
    groups. Type icons: vocabulary names, or any drawn leaf of a group named *types*; above the midline they are
    header badges, below they sit in the ability.

    Args:
        psd: An open card
        vocab: Type icon names (lowercase Types/ stems)
        rulebook: Whether this is a rule page (no corner stars); None guesses as regions()

    Returns:
        (star layers, type icon layers, inline icon layers)
    """
    width, height = psd.width, psd.height
    vocab = {name.lower() for name in vocab}
    if rulebook is None:
        rulebook = width == RULEBOOK_WIDTH or any(layer.name == "Page Number" for layer in psd)

    def star_groups(group):
        for layer in group:
            if layer.is_group() and layer.is_visible():
                if "stars" in layer.name.lower():
                    yield layer
                else:
                    yield from star_groups(layer)
    in_images = lambda layer: any(owner.name.strip().lower() == "images" for owner in _ancestors(layer))
    in_types = lambda layer: any(search(r"types", owner.name, IGNORECASE) for owner in _ancestors(layer))
    drawn = [layer for layer in psd.descendants() if not layer.is_group() and layer.is_visible()
             and layer.has_pixels() and layer.kind != "type" and layer.bbox[2] > layer.bbox[0]]
    stars = [layer for group in star_groups(psd) for layer in _leaves(group) if layer.has_pixels()]
    if not stars:
        stars = [layer for layer in drawn if "star" in layer.name.lower() and not in_images(layer)]
    if not stars and not rulebook:
        stars = [layer for layer in drawn if layer.bbox[0] >= STAR_CORNER[0] * width
                 and layer.bbox[3] <= STAR_CORNER[1] * height and not in_images(layer) and not in_types(layer)]

    star_ids = {id(layer) for layer in stars}
    icons, inline = [], []
    for layer in drawn:
        if id(layer) in star_ids:
            continue
        base = sub(r"\s+copy(\s+\d+)?$", "", layer.name.strip().lower())
        if base in vocab or (in_types(layer) and layer.kind in ("pixel", "smartobject", "shape")):
            (icons if layer.bbox[1] < TYPE_REGION_RATIO * height else inline).append(layer)
    return stars, icons, inline


def overlay_words(psd: PSDImage, abilities=None, texts=None) -> list:
    """
    The old-style status word layers: short visible texts laid over an ability box (Gasoline's "Blazing").

    Args:
        psd: An open card
        abilities: The visible Ability text layers (found when None)
        texts: All visible text layers (found when None)

    Returns:
        The overlay text layers
    """
    if texts is None:
        texts = [layer for layer in psd.descendants() if layer.kind == "type" and layer.is_visible()]
    if abilities is None:
        abilities = [layer for layer in texts if layer.name.strip().lower() == "ability"]
    out = []
    for layer in texts:
        text = (layer.text or "").strip()
        name = layer.name.strip().lower()
        if layer in abilities or name in STAT_LABELS | {"name"} or not text or len(text) > OVERLAY_MAX_CHARS:
            continue
        x0, y0, x1, y1 = layer.bbox
        for ability in abilities:
            ax0, ay0, ax1, ay1 = ability.bbox
            inter = max(0, min(x1, ax1) - max(x0, ax0)) * max(0, min(y1, ay1) - max(y0, ay0))
            if inter >= OVERLAY_INSIDE * max(1, (x1 - x0) * (y1 - y0)):
                out.append(layer)
                break
    return out


def region_masks(psd: PSDImage, found=None, vocab=(), rulebook=None) -> dict:
    """
    Exclusive bool masks (H, W): each pixel goes to the last region in PAINT_ORDER covering it.

    Args:
        psd: An open card
        found: regions(psd) when already computed
        vocab: As regions()
        rulebook: As regions()

    Returns:
        {region key: mask} plus "other" for the pixels no region claims
    """
    found = found or regions(psd, vocab, rulebook)
    owner = np.full((psd.height, psd.width), -1, np.int16)
    for i, key in enumerate(PAINT_ORDER):
        for x0, y0, x1, y1 in found.get(key, []):
            owner[y0:y1, x0:x1] = i
    out = {key: owner == i for i, key in enumerate(PAINT_ORDER)}
    out["other"] = owner == -1
    return out


# ========================
# Module Exports
# ========================
__all__ = [
    'PAINT_ORDER',
    'STATS',
    'badge_layers',
    'clip',
    'overlay_words',
    'region_masks',
    'regions',
    'stat_rows',
]
