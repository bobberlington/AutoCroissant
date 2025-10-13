import asyncio
from collections import deque
from cv2 import imdecode, imencode, cvtColor, IMREAD_COLOR, COLOR_BGR2RGB, COLOR_RGB2BGR
from discord import File, Interaction, Embed
from functools import wraps
from io import BytesIO
from numpy import ndarray, array, asarray
from PIL import Image
from requests import get
from types import SimpleNamespace
from typing import Callable, TypeVar, Awaitable, Optional, Sequence, Any

# -------------------------------------------------------------------
# Queues for async message and file dispatch
# -------------------------------------------------------------------
music_queue: deque[str] = deque()
prev_music: deque[str] = deque()
message_queue: deque[tuple[Interaction, str]] = deque()
edit_messages: deque[tuple[Interaction, str, tuple[File, ...]]] = deque()
file_queue: deque[tuple[Interaction, File]] = deque()
command_queue: deque[tuple[tuple, Callable]] = deque()

slash_registry = {}
T = TypeVar("T")


# -------------------------------------------------------------------
# Utility decorators
# -------------------------------------------------------------------
def to_thread(func: Callable[..., T]) -> Callable[..., Awaitable[T]]:
    """Decorator to run a blocking function in a separate thread asynchronously."""
    if asyncio.iscoroutinefunction(func):
        # No need to wrap coroutine functions — they already run asynchronously.
        return func

    @wraps(func)
    async def wrapper(*args, **kwargs) -> T:
        return await asyncio.to_thread(func, *args, **kwargs)
    return wrapper


# -------------------------------------------------------------------
# Image utilities
# -------------------------------------------------------------------
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


# -------------------------------------------------------------------
# Fake Discord interaction generator
# -------------------------------------------------------------------
def make_fake_interaction(channel_id: int, guild_id: int):
    """
    Create a lightweight fake Interaction object that mimics Discord.py's behavior.
    It queues messages, embeds, and files so that your pipeline loop can process them.
    """

    async def fake_send(
        content: Optional[str] = None,
        *,
        embeds: Optional[Sequence[Embed]] = None,
        file: Optional[File] = None,
        files: Optional[Sequence[File]] = None,
        attachments: Optional[Sequence[File]] = None,
        ephemeral: bool = False,
        **kwargs: Any
    ):
        """Queue a fake message for the pipeline to process."""
        fake_channel = SimpleNamespace(channel_id=channel_id)

        # If embeds are provided, combine them into a readable representation
        if embeds:
            embed_descriptions = []
            for e in embeds:
                desc = e.title or e.description or "(embed)"
                embed_descriptions.append(f"[EMBED: {desc}]")
            embed_text = "\n".join(embed_descriptions)
            if content:
                content = f"{content}\n{embed_text}"
            else:
                content = embed_text

        # Queue text messages
        if content:
            message_queue.append((fake_channel, content))

        # Queue single file
        if file:
            file_queue.append((fake_channel, file))

        # Queue multiple files
        if files:
            for f in files:
                file_queue.append((fake_channel, f))

        # Queue attachments
        if attachments:
            for a in attachments:
                file_queue.append((fake_channel, a))

    class FakeResponse:
        async def send_message(self, content=None, **kwargs):
            await fake_send(content, **kwargs)

        async def defer(self):
            """Simulate a deferred response (does nothing)."""
            pass

    class FakeFollowup:
        async def send(self, content=None, **kwargs):
            await fake_send(content, **kwargs)

    async def edit_original_response(content=None, attachments: Optional[Sequence[File]] = None, **kwargs):
        """Simulate editing an original response by queuing it for edit_messages."""
        fake_channel = SimpleNamespace(channel_id=channel_id)
        edit_messages.append((fake_channel, content or "", tuple(attachments or ())))

    fake_interaction = SimpleNamespace(
        guild_id=guild_id,
        channel_id=channel_id,
        response=FakeResponse(),
        followup=FakeFollowup(),
        edit_original_response=edit_original_response,
        user=None,
    )

    return fake_interaction


# -------------------------------------------------------------------
# Type conversion
# -------------------------------------------------------------------
def convert_value(value: str) -> Any:
    """Convert a string to int, float, or bool if possible, otherwise keep as string."""
    value = value.strip().lower()

    if value in {"true", "false"}:
        return value == "true"

    for convert in (int, float):
        try:
            return convert(value)
        except ValueError:
            continue

    return value
