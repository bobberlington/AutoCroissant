import asyncio
from collections import deque
from cv2 import imdecode, imencode, cvtColor, IMREAD_COLOR, COLOR_BGR2RGB, COLOR_RGB2BGR
from discord import File, Interaction
from functools import wraps
from io import BytesIO
from numpy import ndarray, array, asarray
from PIL import Image
from requests import get
from typing import Callable


# async messages and files to send
music: deque[str] = deque()
prev_music: deque[str] = deque()
messages: deque[tuple[Interaction, str]] = deque()
edit_messages: deque[tuple[Interaction, str, tuple[File, ...]]] = deque()
files: deque[tuple[Interaction, File]] = deque()
commands: deque[tuple[tuple, Callable]] = deque()

def to_thread(func: Callable):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        return await asyncio.to_thread(func, *args, **kwargs)
    return wrapper

def url_to_cv2image(url: str, readFlag=IMREAD_COLOR) -> ndarray:
    return imdecode(asarray(bytearray(get(url).content), dtype="uint8"), readFlag)

def cv2discordfile(img: ndarray) -> File:
    return File(BytesIO(array(imencode('.png', img)[1]).tobytes()), filename='image.png')

def url_to_pilimage(url: str) -> Image.Image:
    return Image.open(get(url, stream=True).raw)

def pildiscordfile(img: Image.Image) -> File:
    with BytesIO() as bin:
        img.save(bin, 'png')
        bin.seek(0)
        return File(bin, filename='image.png')

def cv2_to_pil(cv2_img: ndarray) -> Image.Image:
    return Image.fromarray(cvtColor(cv2_img, COLOR_BGR2RGB))

def pil_to_cv2(pil_img: Image.Image) -> ndarray:
    return cvtColor(array(pil_img), COLOR_RGB2BGR)
