from asyncio import iscoroutinefunction, get_running_loop, to_thread as asyncio_to_thread
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from cv2 import imdecode, imencode, cvtColor, IMREAD_COLOR, COLOR_BGR2RGB, COLOR_RGB2BGR
from discord import File, Interaction, Object
from functools import lru_cache, wraps
from io import BytesIO
from numpy import ndarray, array, asarray
from os import listdir
from os.path import join, isdir
from PIL import Image
from requests import get
from typing import Callable, TypeVar, Awaitable, Any, Optional

import config

# ========================
# Configurable Parameters
# ========================
BREAK_LEN = 1950
ADMINS = getattr(config, "ADMINS", [])


# ===================================================================
# Queues for async message and command dispatch
# ===================================================================
dispatch_queue: deque[tuple[Interaction, dict[str, Any]]] = deque()
edit_queue: deque[tuple[Interaction, dict[str, Any]]] = deque()
command_queue: deque[tuple[tuple, dict[str, Any], Callable[..., Any]]] = deque()

slash_registry = {}
T = TypeVar("T")


# ===================================================================
# Queue wrappers
# ===================================================================
def queue_any(interaction: Interaction, **kwargs):
    """Queue any send you want."""
    dispatch_queue.append((interaction, kwargs))


def queue_message(interaction: Interaction, content: str, **kwargs):
    """Queue a simple message send."""
    kwargs["content"] = content
    dispatch_queue.append((interaction, kwargs))


def queue_file(interaction: Interaction, file: File, **kwargs):
    """Queue a file send (can include content or embeds too)."""
    kwargs["file"] = file
    dispatch_queue.append((interaction, kwargs))


def queue_edit(interaction: Interaction, **kwargs):
    """Queue an edit to an interaction’s original response."""
    edit_queue.append((interaction, kwargs))


def queue_command(func: Callable[..., Any], *args, **kwargs):
    """Queue a function (async or sync) for deferred execution."""
    command_queue.append(((args, kwargs), func))


# ===================================================================
# Utility decorators
# ===================================================================
def to_thread(func: Callable[..., T]) -> Callable[..., Awaitable[T]]:
    """Decorator to run a blocking function in a separate thread asynchronously."""
    if iscoroutinefunction(func):
        # No need to wrap coroutine functions — they already run asynchronously.
        return func

    @wraps(func)
    async def wrapper(*args, **kwargs) -> T:
        return await asyncio_to_thread(func, *args, **kwargs)
    return wrapper


def to_threadpool(func: Callable[..., T], executor: ThreadPoolExecutor | None = None):
    """Decorator to run a blocking function in a separate threadpool asynchronously."""
    if iscoroutinefunction(func):
        # No need to wrap coroutine functions — they already run asynchronously.
        return func

    @wraps(func)
    async def wrapper(*args, **kwargs) -> T:
        return await get_running_loop().run_in_executor(executor, func, *args, **kwargs)
    return wrapper


# ===================================================================
# Image utilities
# ===================================================================
def url_to_cv2image(url: str, readFlag=IMREAD_COLOR) -> ndarray:
    """Fetch an image from a URL and convert it to a cv2 image array."""
    return imdecode(asarray(bytearray(get(url, timeout=30).content), dtype="uint8"), readFlag)


def cv2discordfile(img: ndarray, filename: str = "image.png") -> File:
    """Convert a cv2 image (ndarray) to a Discord File."""
    return File(BytesIO(array(imencode('.png', img)[1]).tobytes()), filename=filename)


def url_to_pilimage(url: str) -> Image.Image:
    """Fetch an image from a URL and return it as a PIL Image."""
    return Image.open(get(url, stream=True, timeout=30).raw).convert("RGBA")


def pildiscordfile(img: Image.Image, filename: str = "image.png") -> File:
    """Convert a PIL image to a Discord File."""
    buffer = BytesIO()
    img.save(buffer, 'PNG')
    buffer.seek(0)
    return File(buffer, filename=filename)


def cv2_to_pil(cv2_img: ndarray) -> Image.Image:
    """Convert an OpenCV image (BGR) to a PIL Image (RGB)."""
    return Image.fromarray(cvtColor(cv2_img, COLOR_BGR2RGB))


def pil_to_cv2(pil_img: Image.Image) -> ndarray:
    """Convert a PIL Image (RGB) to an OpenCV image (BGR)."""
    return cvtColor(array(pil_img), COLOR_RGB2BGR)


# ===================================================================
# Fake Discord interaction generator
# ===================================================================
def make_fake_interaction(channel_id: int, guild_id: Optional[int] = None):
    """
    Create a minimal interaction-like object for queuing messages.

    Use this when you need to send messages without an original interaction,
    such as in scheduled tasks or background processes.

    Args:
        channel_id: Discord channel ID
        guild_id: Optional Discord guild ID
    """

    class FakeUser:
        """Minimal user object representing the bot."""
        def __init__(self, id: int = 1011264680683110433, name: str = "AutoCroissant", discriminator: str = "0225"):
            self.id = id
            self.name = name
            self.discriminator = discriminator
            self.bot = True

    class FakeResponse:
        def __init__(self, channel_id: int):
            self.channel_id = channel_id
            self._done = False

        async def send_message(self, content: Optional[str] = None, **kwargs: Any) -> None:
            if content:
                kwargs["content"] = content
            fake_interaction = Object(id=self.channel_id)
            fake_interaction.channel_id = self.channel_id
            queue_any(fake_interaction, **kwargs)
            self._done = True

        def is_done(self) -> bool:
            return self._done

        async def defer(self, **kwargs: Any) -> None:
            pass

    class FakeFollowup:
        def __init__(self, channel_id: int):
            self.channel_id = channel_id

        async def send(self, content: Optional[str] = None, **kwargs: Any) -> None:
            if content:
                kwargs["content"] = content
            fake_interaction = Object(id=self.channel_id)
            fake_interaction.channel_id = self.channel_id
            queue_any(fake_interaction, **kwargs)

    class FakeInteraction:
        def __init__(self, channel_id: int, guild_id: Optional[int] = None):
            self.channel_id = channel_id
            self.guild_id = guild_id
            self.response = FakeResponse(channel_id)
            self.followup = FakeFollowup(channel_id)
            self.user = FakeUser()
            self.guild = Object(id=guild_id) if guild_id else None
            self.channel = Object(id=channel_id)

        async def edit_original_response(self, content: Optional[str] = None, **kwargs: Any) -> None:
            if content:
                kwargs["content"] = content
            fake_obj = Object(id=self.channel_id)
            fake_obj.channel_id = self.channel_id
            queue_edit(fake_obj, **kwargs)

    return FakeInteraction(channel_id, guild_id)


def make_simple_fake_interaction(channel_id: int, guild_id: Optional[int] = None):
    """
    Simplified version that just wraps a Discord Object with required attributes.
    Use this if you only need queue_message() compatibility.

    Args:
        channel_id: Discord channel ID
        guild_id: Optional Discord guild ID
    """
    fake = Object(id=channel_id)
    fake.channel_id = channel_id
    fake.guild_id = guild_id
    return fake


# ===================================================================
# Misc
# ===================================================================
@lru_cache(maxsize=128)
def convert_value(value: str) -> Any:
    """Convert a string to int, float, or bool if possible, otherwise keep as string."""
    value = value.strip()

    if value.lower() in {"true", "false"}:
        return value == "true"

    for convert in (int, float):
        try:
            return convert(value)
        except ValueError:
            continue

    return value


def split_long_message(msg: str, max_length: int = BREAK_LEN) -> list[str]:
    """Split long text into Discord-safe chunks while preserving code blocks."""
    parts = []
    code_fence = "```"
    in_code = False

    while msg:
        if len(msg) <= max_length:
            chunk = msg
            msg = ""
        else:
            idx = msg.rfind("\n", 0, max_length)
            if idx == -1:
                idx = msg.rfind(" ", 0, max_length)
            if idx == -1:
                idx = max_length

            chunk = msg[:idx]
            msg = msg[idx:].lstrip()

        # Count code fences in this chunk
        fence_count = chunk.count(code_fence)
        if fence_count % 2 == 1:
            in_code = not in_code

        # If chunk ends inside a code block, close it
        if in_code:
            chunk += f"\n{code_fence}"

        parts.append(chunk)

        # If next chunk continues inside a code block, reopen it
        if in_code:
            msg = f"{code_fence}\n{msg}"

    return parts


def parse_named_args(parts):
    """
    Split a list like ['prompt:sunset', 'height:512'] into args=[], kwargs={...},
    converting numeric and boolean values automatically.
    """
    args = []
    kwargs = {}
    for p in parts:
        if ":" in p:
            key, value = p.split(":", 1)
            kwargs[key] = convert_value(value)
        else:
            args.append(convert_value(p))
    return args, kwargs


def recursively_traverse(directory: str, prefix: str = "") -> str:
    """Recursively list all files in a directory."""
    lines = []
    for item in listdir(directory):
        path = join(directory, item)
        if isdir(path):
            lines.append(f"{prefix}{item}/")
            lines.append(recursively_traverse(path, prefix + "\t"))
        else:
            lines.append(f"{prefix}{item}")
    return "\n".join(lines)


def perms_check(interaction: Interaction) -> bool:
    """
    Check if user has admin permissions.

    Args:
        interaction: Discord interaction

    Returns:
        True if user lacks permissions, False if user is admin
    """
    return interaction.user.id not in ADMINS
