from asyncio import to_thread
from io import BytesIO
from re import compile, sub
from urllib.parse import quote

from discord import Client, HTTPException, Interaction
from discord.utils import escape_markdown
from PIL import Image, ImageDraw
from requests import get

import config
from commands.utils import queue_message, split_long_message

# ========================
# Configuration
# ========================
GIT_TOKEN: str = getattr(config, "GIT_TOKEN", "")
EMOJI_SIZE = 128                                        # Discord shows emojis at 48 px or less; 128 stays sharp when enlarged
MAX_NAME_LEN = 32                                       # Discord emoji names: 2-32 characters of a-z, 0-9 and _
ICON_FOLDERS = ("Types", "Types/Stars")
NO_SUFFIX = "_no"                                       # crossed-out copy of every Types/ icon: [fire:no] -> :fire_no:
NO_RED = (224, 27, 34, 255)                             # same red as the stoogedings font's :no sign
TOKEN_PATTERN = compile(r"\[([^\[\]\n]{1,48})\]")       # [fire], [Fragile Equip], [fire:tint]; [X] and [Status] stay text

# Emoji name -> "<:name:id>" for every application emoji the bot owns. Filled by sync_type_emojis.
type_emojis: dict[str, str] = {}


# ========================
# Helper Functions
# ========================
def emoji_name(text: str) -> str:
    """
    Turn an icon file stem or a token into a Discord emoji name.

    "Fragile Equip", "fragile-equip" and "Fragile_Equip" all become "fragile_equip", matching how the
    stoogedings font treats tokens (case-insensitive; space, _ and - interchangeable). Dots and
    apostrophes are dropped ("N.M.E" -> "nme").

    Args:
        text: File stem (e.g. "Dst Star") or token text (e.g. "Defensive Minion")

    Returns:
        Lowercase name of at most 32 characters, cut at a word boundary when too long
    """
    name = sub(r"[^a-z0-9]+", "_", sub(r"[.'’]", "", text.lower())).strip("_")
    while len(name) > MAX_NAME_LEN and "_" in name:
        name = name.rsplit("_", 1)[0]
    return name[:MAX_NAME_LEN]


def _list_type_icons() -> tuple[str, dict[str, tuple[str, bool]]]:
    """
    List the icon PNGs in Types/ and Types/Stars/ of the card repository on GitHub, plus a crossed-out
    variant for every Types/ icon.

    Returns:
        (repository, {emoji name: (repo path, crossed out)})
    """
    from commands.query_card import card_repo  # lazy, like query_card's own late imports

    headers = {'Authorization': f'token {GIT_TOKEN}'} if GIT_TOKEN else {}
    response = get(f"https://api.github.com/repos/{card_repo.repository}/git/trees/main?recursive=1",
                   headers=headers, timeout=30)
    response.raise_for_status()
    icons: dict[str, tuple[str, bool]] = {}
    for item in response.json().get("tree", []):
        folder, _, filename = item.get("path", "").rpartition("/")
        if folder in ICON_FOLDERS and filename.lower().endswith(".png"):
            name = emoji_name(filename.rsplit(".", 1)[0])
            icons.setdefault(name, (item["path"], False))
            if folder == "Types":
                icons.setdefault(name[:MAX_NAME_LEN - len(NO_SUFFIX)] + NO_SUFFIX, (item["path"], True))
    return card_repo.repository, icons


def _cross_out(canvas: Image.Image) -> Image.Image:
    """Draw the red prohibition sign (ring + slash) over an emoji, antialiased by drawing at 4x."""
    big = EMOJI_SIZE * 4
    sign = Image.new("RGBA", (big, big))
    draw = ImageDraw.Draw(sign)
    width = round(big * 0.075)
    margin = width // 2 + 2
    draw.ellipse((margin, margin, big - margin, big - margin), outline=NO_RED, width=width)
    inset = round(big * 0.15)
    draw.line((inset, inset, big - inset, big - inset), fill=NO_RED, width=width)
    return Image.alpha_composite(canvas, sign.resize((EMOJI_SIZE, EMOJI_SIZE), Image.LANCZOS))


def _emoji_image(repository: str, path: str, crossed_out: bool = False) -> bytes:
    """
    Download an icon and fit it into a centered, transparent 128x128 PNG (Discord's limit is 256 KB).
    Small pixel-art icons are enlarged with nearest-neighbor so they stay crisp.

    Args:
        repository: GitHub repository (USER/REPO)
        path: Path of the PNG inside the repository
        crossed_out: Draw the red prohibition sign over it (the :no variant)

    Returns:
        PNG bytes
    """
    url = f"https://raw.githubusercontent.com/{repository}/main/{quote(path, safe='/')}"
    image = Image.open(BytesIO(get(url, timeout=30).content)).convert("RGBA")
    bbox = image.getchannel("A").point(lambda a: 255 if a > 8 else 0).getbbox()
    if bbox:
        image = image.crop(bbox)
    scale = EMOJI_SIZE / max(image.size)
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    image = image.resize(size, Image.NEAREST if scale > 1 else Image.LANCZOS)
    canvas = Image.new("RGBA", (EMOJI_SIZE, EMOJI_SIZE))
    canvas.paste(image, ((EMOJI_SIZE - image.width) // 2, (EMOJI_SIZE - image.height) // 2))
    if crossed_out:
        canvas = _cross_out(canvas)
    out = BytesIO()
    canvas.save(out, "PNG", optimize=True)
    return out.getvalue()


# ========================
# Public API Functions
# ========================
def render_type_tokens(text: str, escape: bool = False) -> str:
    """
    Replace [fire]-style tokens with the bot's type emojis. Brackets that aren't icons ([X], [Status]) stay as text.

    Args:
        text: Ability or rulebook text
        escape: Also escape Discord markdown in the surrounding text (for text that used to sit in a code block)

    Returns:
        The text with known tokens swapped for emojis; [fire:no] uses the crossed-out emoji
        (or 🚫 + the icon if that emoji doesn't exist yet)
    """
    parts = TOKEN_PATTERN.split(text)
    out = []
    for i, part in enumerate(parts):
        if i % 2 == 0:
            out.append(escape_markdown(part) if escape else part)
            continue
        base, _, variant = part.partition(":")
        name = emoji_name(base)
        emoji = type_emojis.get(name)
        crossed = variant.strip().lower() == "no"
        if emoji is None:
            out.append(escape_markdown(f"[{part}]") if escape else f"[{part}]")
        elif crossed:
            out.append(type_emojis.get(name[:MAX_NAME_LEN - len(NO_SUFFIX)] + NO_SUFFIX) or "🚫" + emoji)
        else:
            out.append(emoji)
    return "".join(out)


async def sync_type_emojis(interaction: Interaction | None, client: Client,
                           refresh: bool = False, prune: bool = False) -> None:
    """
    Upload every Types/ and Types/Stars/ icon as one of the bot's application emojis, then reload type_emojis.

    Application emojis belong to whichever bot identity is logged in, so each identity syncs its own set.
    Only missing icons are uploaded unless refresh is set.

    Args:
        interaction: Discord interaction to report to, or None (startup)
        client: Discord client instance
        refresh: Re-upload every icon, e.g. after an icon was edited
        prune: Delete bot emojis whose icon is no longer in Types/
    """
    try:
        repository, icons = await to_thread(_list_type_icons)
        existing = {emoji.name: emoji for emoji in await client.fetch_application_emojis()}
    except Exception as e:
        report = f"Emoji sync failed: {e}"
    else:
        uploaded, removed, failed = [], [], []
        for name, (path, crossed_out) in icons.items():
            current = existing.get(name)
            if current is not None and not refresh:
                continue
            try:
                image = await to_thread(_emoji_image, repository, path, crossed_out)
                if current is not None:
                    await current.delete()
                existing[name] = await client.create_application_emoji(name=name, image=image)
                uploaded.append(name)
            except (HTTPException, OSError) as e:
                existing.pop(name, None)
                failed.append(f"{name} ({e})")
        stale = sorted(set(existing) - set(icons))
        if prune:
            for name in stale:
                try:
                    await existing.pop(name).delete()
                    removed.append(name)
                except HTTPException as e:
                    failed.append(f"{name} ({e})")
        type_emojis.clear()
        type_emojis.update({name: str(emoji) for name, emoji in existing.items()})

        report = (f"Type emojis: {len(icons)} icons and crossed-out variants in {repository}, "
                  f"{len(uploaded)} uploaded, {len(type_emojis)} ready.")
        if removed:
            report += f"\nRemoved: {', '.join(removed)}"
        elif stale:
            report += f"\nNo longer in Types/ (kept, use prune:True to delete): {', '.join(stale)}"
        if failed:
            report += f"\nFailed: {', '.join(failed)}"

    if interaction is None:
        print(report)
        return
    for chunk in split_long_message(report):
        queue_message(interaction, chunk)


# ========================
# Initialization
# ========================
async def init_type_emojis(client: Client) -> None:
    """Called from on_ready via queue_command: upload any missing type emojis and load the token map."""
    await sync_type_emojis(None, client)


# ========================
# Module Exports
# ========================
__all__ = [
    'emoji_name',
    'init_type_emojis',
    'render_type_tokens',
    'sync_type_emojis',
    'type_emojis',
]
