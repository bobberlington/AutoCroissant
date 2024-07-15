import asyncio
import cv2
import discord
import functools
from io import BytesIO
import numpy as np
from PIL import Image
import typing
from urllib.request import urlopen


# async messages and files to send
messages = []
files = []

def to_thread(func: typing.Callable) -> typing.Coroutine:
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        return await asyncio.to_thread(func, *args, **kwargs)
    return wrapper

def url_to_cv2image(url: str, readFlag=cv2.IMREAD_COLOR) -> np.ndarray:
    resp = urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, readFlag)
    return image

def cv2discordfile(img: np.ndarray) -> discord.File:
    img_encode = cv2.imencode('.png', img)[1]
    data_encode = np.array(img_encode)
    byte_encode = data_encode.tobytes()
    byteImage = BytesIO(byte_encode)
    return discord.File(byteImage, filename='image.png')

def url_to_pilimage(url: str) -> Image:
    resp = urlopen(url)
    return Image.open(resp)

def pildiscordfile(img: Image) -> discord.File:
    with BytesIO() as bin:
        img.save(bin, 'png')
        bin.seek(0)
        return discord.File(bin, filename='image.png')
