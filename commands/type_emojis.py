from asyncio import Lock, to_thread
from io import BytesIO
from json import dump, load
from os import remove, replace
from os.path import exists
from re import compile, sub
from typing import Callable, Optional

from discord import Client, Interaction
from discord.utils import escape_markdown
from PIL import Image, ImageDraw
from requests.exceptions import RequestException

from global_config import EMOJI_JSON, STARS_DIR, TYPES_DIR
from commands.github_api import fetch_blob, fetch_tree
from commands.utils import queue_long_message

# ========================
# Configuration
# ========================
EMOJI_SIZE = 128                                        # Discord shows emojis at 48 px or less; 128 stays sharp when enlarged
MAX_NAME_LEN = 32                                       # Discord emoji names: 2-32 characters of a-z, 0-9 and _
ICON_FOLDERS = (TYPES_DIR, STARS_DIR)
NO_SUFFIX = "_no"                                       # crossed-out copy of every Types/ icon: [fire:no] -> :fire_no:
NO_RED = (224, 27, 34, 255)                             # same red as the stoogedings font's :no sign
TOKEN_PATTERN = compile(r"\[([^\[\]\n]{1,48})\]")       # [fire], [Fragile Equip], [fire:tint]; [X] and [Status] stay text

# Emoji name -> "<:name:id>" for every application emoji the bot owns. Filled by sync_type_emojis.
type_emojis: dict[str, str] = {}

# ========================
# Global State
# ========================
_blobs: dict[str, str] = {}                 # emoji name -> icon blob sha it was made from (EMOJI_JSON)
_blobs_loaded = False
_client: Optional[Client] = None            # set at startup, used by the /update_stats sync
_lock = Lock()                              # one sync at a time


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


def _no_name(name: str) -> str:
    """Name of an icon's crossed-out emoji."""
    return name[:MAX_NAME_LEN - len(NO_SUFFIX)] + NO_SUFFIX


def _icons(shas: dict) -> dict[str, tuple[str, str, bool]]:
    """
    Pick the icon PNGs in Types/ and Types/Stars/ out of a repo tree, plus a crossed-out variant for
    every Types/ icon.

    Args:
        shas: {repo path: blob sha} of the whole tree

    Returns:
        {emoji name: (repo path, blob sha, crossed out)}
    """
    icons: dict[str, tuple[str, str, bool]] = {}
    for path in sorted(shas):                   # tree order: the first icon with a name wins
        folder, _, filename = path.rpartition("/")
        if folder in ICON_FOLDERS and filename.lower().endswith(".png"):
            name = emoji_name(filename.rsplit(".", 1)[0])
            icons.setdefault(name, (path, shas[path], False))
            if folder == TYPES_DIR:
                icons.setdefault(_no_name(name), (path, shas[path], True))
    return icons


def _list_type_icons() -> tuple[str, dict[str, tuple[str, str, bool]]]:
    """
    List the type icons in the card repository on GitHub.

    Returns:
        (repository, {emoji name: (repo path, blob sha, crossed out)})
    """
    from commands.query_card import card_repo  # lazy, like query_card's own late imports

    with card_repo._tree_lock:                  # one tree fetch at a time: at startup, the card list's or this one
        known, etag = card_repo._tree, card_repo._tree_etag
        try:                                    # with the card list's ETag: 304 = its tree is current
            response = fetch_tree(card_repo.repository, etag if isinstance(known, dict) and etag else "")
            if response.status_code == 304 and isinstance(known, dict):
                tree = known
            else:
                response.raise_for_status()
                tree = response.json()
            if not isinstance(tree, dict):
                raise ValueError("not a tree")
            if known is None and response.headers.get("ETag"):    # card list not loaded yet: its fetch can 304
                card_repo._tree, card_repo._tree_etag = tree, response.headers["ETag"]
        except (RequestException, ValueError) as e:
            if not isinstance(card_repo._tree, dict):
                raise
            print(f"Type emojis: GitHub tree listing failed ({e}); using the last good card tree")
            tree = card_repo._tree
    shas = {item.get("path", ""): item.get("sha", "") for item in tree.get("tree", [])}
    return card_repo.repository, _icons(shas)


def _icon_png(repository: str, sha: str) -> bytes:
    """An icon's exact bytes by blob sha (raw main can lag behind the tree)."""
    return fetch_blob(repository, sha).content


def _load_blobs() -> None:
    """Read the saved emoji blob map once."""
    global _blobs_loaded
    _blobs_loaded = True
    if not exists(EMOJI_JSON):
        return
    try:
        with open(EMOJI_JSON, encoding="utf-8") as f:
            _blobs.update({str(k): str(v) for k, v in load(f).items()})
    except (OSError, ValueError, AttributeError) as e:
        print(f"Could not read {EMOJI_JSON}, existing type emojis count as current: {e}")


def _save_blobs() -> None:
    """Write the emoji blob map atomically."""
    tmp = f"{EMOJI_JSON}.tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            dump(_blobs, f, indent=0, sort_keys=True)
        replace(tmp, EMOJI_JSON)
    finally:
        if exists(tmp):
            remove(tmp)


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


def _emoji_image(png: bytes, crossed_out: bool = False) -> bytes:
    """
    Fit an icon into a centered, transparent 128x128 PNG (Discord's limit is 256 KB).
    Small pixel-art icons are enlarged with nearest-neighbor so they stay crisp.

    Args:
        png: The icon file's bytes
        crossed_out: Draw the red prohibition sign over it (the :no variant)

    Returns:
        PNG bytes
    """
    image = Image.open(BytesIO(png)).convert("RGBA")
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
            out.append(type_emojis.get(_no_name(name)) or "🚫" + emoji)
        else:
            out.append(emoji)
    return "".join(out)


async def _upload(client: Client, repository: str, icons: dict, names: list, existing: dict) -> tuple[list, list]:
    """
    Upload the named icons as emojis, replacing the ones that exist, and remember each one's blob.

    Args:
        client: Discord client instance
        repository: GitHub repository (USER/REPO)
        icons: {emoji name: (repo path, blob sha, crossed out)}
        names: Emoji names to upload
        existing: {emoji name: emoji}, updated in place

    Returns:
        (uploaded names, failure notes)
    """
    uploaded, failed, pngs = [], [], {}
    for name in names:
        _, sha, crossed_out = icons[name]
        current = existing.get(name)
        try:
            if sha not in pngs:
                pngs[sha] = await to_thread(_icon_png, repository, sha)
            image = await to_thread(_emoji_image, pngs[sha], crossed_out)
            if current is not None:
                await current.delete()
                existing.pop(name)
                _blobs.pop(name, None)
                type_emojis.pop(name, None)
            existing[name] = await client.create_application_emoji(name=name, image=image)
            _blobs[name] = sha
            type_emojis[name] = str(existing[name])
            uploaded.append(name)
        except Exception as e:                  # aiohttp errors too; the next sync retries
            failed.append(f"{name} ({e})")
    return uploaded, failed


async def _remove(names: list, existing: dict) -> tuple[list, list]:
    """Delete the named emojis and forget their blobs. Returns (removed names, failure notes)."""
    removed, failed = [], []
    for name in names:
        try:
            if name in existing:
                await existing[name].delete()
                existing.pop(name)
                type_emojis.pop(name, None)
                removed.append(name)
            _blobs.pop(name, None)
        except Exception as e:
            failed.append(f"{name} ({e})")
    return removed, failed


async def _publish(existing: dict) -> None:
    """Reload the token map and save the blob map."""
    type_emojis.clear()
    type_emojis.update({name: str(emoji) for name, emoji in existing.items()})
    try:
        await to_thread(_save_blobs)
    except OSError as e:
        print(f"Could not save {EMOJI_JSON}: {e}")


async def sync_type_emojis(interaction: Optional[Interaction], client: Client,
                           refresh: bool = False, prune: bool = False, edited: bool = True) -> None:
    """
    Upload every Types/ and Types/Stars/ icon as one of the bot's application emojis, then reload type_emojis.

    Application emojis belong to whichever bot identity is logged in, so each identity syncs its own set.
    Uploads missing icons and icons whose blob changed since their emoji was made; everything with refresh.

    Args:
        interaction: Discord interaction to report to, or None (startup)
        client: Discord client instance
        refresh: Re-upload every icon
        prune: Delete bot emojis whose icon is no longer in Types/
        edited: Also replace edited icons (startup uploads only missing ones and adopts unmapped existing ones)
    """
    global _client
    _client = client
    async with _lock:
        try:
            if not _blobs_loaded:
                await to_thread(_load_blobs)
            repository, icons = await to_thread(_list_type_icons)
            existing = {emoji.name: emoji for emoji in await client.fetch_application_emojis()}
        except Exception as e:
            report = f"Emoji sync failed: {e}"
            if not type_emojis:                 # still show the emojis the bot already has
                try:
                    type_emojis.update({emoji.name: str(emoji) for emoji in await client.fetch_application_emojis()})
                    report += f" (using the {len(type_emojis)} existing emojis)"
                except Exception:
                    pass
        else:
            if not edited:                      # first run without a map: trust existing emojis, don't replace all
                for name, (_, sha, _) in icons.items():
                    if name in existing and name not in _blobs:
                        _blobs[name] = sha
            names = [name for name, (_, sha, _) in icons.items()
                     if refresh or name not in existing or (edited and _blobs.get(name) != sha)]
            uploaded, failed = await _upload(client, repository, icons, names, existing)
            stale = sorted(set(existing) - set(icons))
            removed = []
            if prune:
                removed, failures = await _remove(stale, existing)
                failed += failures
            await _publish(existing)

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
    queue_long_message(interaction, report)


async def sync_changed_emojis(interaction: Optional[Interaction], repository: str, shas: dict,
                              on_fail: Optional[Callable[[str], None]] = None) -> None:
    """
    Queued by /update_stats after it fetched a changed tree: upload or replace the emojis whose icon blob is
    new or changed, and delete the ones whose icon is gone. Says one line when something changed.

    Args:
        interaction: Discord interaction to report to, or None
        repository: GitHub repository (USER/REPO) the tree is from
        shas: {repo path: blob sha} of the tree
        on_fail: Called with repository when something failed, so the next /update_stats retries
    """
    client = _client
    if client is None:                          # startup sync not run yet
        if on_fail:
            on_fail(repository)
        return
    async with _lock:
        if not _blobs_loaded:
            await to_thread(_load_blobs)
        icons = _icons(shas)
        gone = [name for name in _blobs if name not in icons]
        if not gone and all(_blobs.get(name) == sha and name in type_emojis
                            for name, (_, sha, _) in icons.items()):
            return                              # nothing to do: no Discord call
        try:
            existing = {emoji.name: emoji for emoji in await client.fetch_application_emojis()}
        except Exception as e:
            report = f"Type emoji sync failed: {e} (next /update_stats retries)"
            failed = [str(e)]
        else:
            names = [name for name, (_, sha, _) in icons.items() if _blobs.get(name) != sha or name not in existing]
            before = set(existing)
            uploaded, failed = await _upload(client, repository, icons, names, existing)
            removed, failures = await _remove(gone, existing)
            failed += failures
            await _publish(existing)
            if not uploaded and not removed and not failed:
                return
            new = sum(name not in before for name in uploaded)
            parts = [f"{len(uploaded) - new} updated"] if len(uploaded) > new else []
            parts += [f"{new} new"] if new else []
            parts += [f"{len(removed)} removed"] if removed else []
            parts += [f"{len(failed)} failed: {', '.join(failed)} (next /update_stats retries)"] if failed else []
            report = "Type emojis: " + ", ".join(parts)
        if failed and on_fail:
            on_fail(repository)

    if interaction is None:
        print(report)
        return
    queue_long_message(interaction, report)


# ========================
# Initialization
# ========================
async def init_type_emojis(client: Client) -> None:
    """Called from on_ready via queue_command: upload any missing type emojis and load the token map."""
    await sync_type_emojis(None, client, edited=False)


# ========================
# Module Exports
# ========================
__all__ = [
    'emoji_name',
    'init_type_emojis',
    'render_type_tokens',
    'sync_changed_emojis',
    'sync_type_emojis',
    'type_emojis',
]
