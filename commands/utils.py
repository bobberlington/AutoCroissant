import asyncio
from collections import deque
from cv2 import imdecode, imencode, cvtColor, IMREAD_COLOR, COLOR_BGR2RGB, COLOR_RGB2BGR
from discord import File, Interaction
from functools import wraps
from io import BytesIO
from numpy import ndarray, array, asarray
from os import listdir
from os.path import join, isdir
from PIL import Image
from requests import get
from types import SimpleNamespace
from typing import Callable, TypeVar, Awaitable, Any

# ========================
# Configurable Parameters
# ========================
BREAK_LEN: int = 1950

# ===================================================================
# Queues for async message and command dispatch
# ===================================================================
music_queue: deque[str] = deque()
prev_music: deque[str] = deque()

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
    if asyncio.iscoroutinefunction(func):
        # No need to wrap coroutine functions — they already run asynchronously.
        return func

    @wraps(func)
    async def wrapper(*args, **kwargs) -> T:
        return await asyncio.to_thread(func, *args, **kwargs)
    return wrapper


# ===================================================================
# Image utilities
# ===================================================================
def url_to_cv2image(url: str, readFlag=IMREAD_COLOR) -> ndarray:
    """Fetch an image from a URL and convert it to a cv2 image array."""
    return imdecode(asarray(bytearray(get(url, timeout=10).content), dtype="uint8"), readFlag)


def cv2discordfile(img: ndarray, filename: str = "image.png") -> File:
    """Convert a cv2 image (ndarray) to a Discord File."""
    return File(BytesIO(array(imencode('.png', img)[1]).tobytes()), filename=filename)


def url_to_pilimage(url: str) -> Image.Image:
    """Fetch an image from a URL and return it as a PIL Image."""
    return Image.open(get(url, stream=True, timeout=10).raw).convert("RGBA")


def pildiscordfile(img: Image.Image, filename: str = "image.png") -> File:
    """Convert a PIL image to a Discord File."""
    with BytesIO() as bin:
        img.save(bin, 'png')
        bin.seek(0)
        return File(bin, filename=filename)


def cv2_to_pil(cv2_img: ndarray) -> Image.Image:
    """Convert an OpenCV image (BGR) to a PIL Image (RGB)."""
    return Image.fromarray(cvtColor(cv2_img, COLOR_BGR2RGB))


def pil_to_cv2(pil_img: Image.Image) -> ndarray:
    """Convert a PIL Image (RGB) to an OpenCV image (BGR)."""
    return cvtColor(array(pil_img), COLOR_RGB2BGR)


# ===================================================================
# Fake Discord interaction generator
# ===================================================================
def make_fake_interaction(channel_id: int, guild_id: int):
    """Create a fake interaction object for when the original is lost/old."""

    async def send(content: str | None = None, **kwargs: Any):
        """Generic sender that queues any outgoing message."""
        if content:
            kwargs["content"] = content
        queue_any(SimpleNamespace(channel_id=channel_id), **kwargs)

    async def edit_original_response(content: str | None = None, **kwargs: Any):
        """Simulate editing the original response."""
        if content:
            kwargs["content"] = content
        queue_edit(SimpleNamespace(channel_id=channel_id), **kwargs)

    async def defer():
        pass

    fake_response = SimpleNamespace(
        send_message=send,
        defer=defer,
    )
    fake_followup = SimpleNamespace(send=send)

    # The fake Interaction-like object
    return SimpleNamespace(
        guild_id=guild_id,
        channel_id=channel_id,
        response=fake_response,
        followup=fake_followup,
        edit_original_response=edit_original_response,
        user=None,
    )


# ===================================================================
# Misc
# ===================================================================
def convert_value(value: str) -> Any:
    """Convert a string to int, float, or bool if possible, otherwise keep as string."""
    value = value.strip().lower()

    if value in {"true", "false"}:
        return value == "true"

    for convert in (float, int):
        try:
            return convert(value)
        except ValueError:
            continue

    return value


def split_long_message(msg: str):
    """Split long text into Discord-safe chunks while preserving code blocks."""
    parts = []
    code_prefix = "```"
    code_suffix = "```"
    inside_code = msg.startswith(code_prefix) and msg.endswith(code_suffix)

    if inside_code:
        msg = msg[len(code_prefix):-len(code_suffix)]

    while len(msg) > BREAK_LEN:
        idx = msg.rfind("\n", 0, BREAK_LEN)
        if idx == -1:
            idx = BREAK_LEN
        parts.append(msg[:idx])
        msg = msg[idx:]

    if msg:
        parts.append(msg)

    if inside_code:
        parts = [f"{code_prefix}{p}{code_suffix}" for p in parts]

    return parts


def recursively_traverse(directory: str, prefix: str = "") -> str:
    """Recursively list all files in a directory."""
    lines = []
    for item in listdir(directory):
        path = join(directory, item)
        if isdir(path):
            lines.append(prefix + item + "/")
            lines.append(recursively_traverse(path, prefix + "\t"))
        else:
            lines.append(prefix + item)
    return "\n".join(lines)
