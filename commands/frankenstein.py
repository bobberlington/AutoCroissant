import difflib
import cv2
import discord
from urllib.request import urlopen
import numpy as np
from io import BytesIO

from commands.query_card import populate_files, try_open_alias

file_alias = {}
files = {}
ambiguous_names = {}
filenames = []

def url_to_image(url, readFlag=cv2.IMREAD_COLOR):
    resp = urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, readFlag)

    return image

def cv2discordfile(img):
    img_encode = cv2.imencode('.png', img)[1]
    data_encode = np.array(img_encode)
    byte_encode = data_encode.tobytes()
    byteImage = BytesIO(byte_encode)
    image = discord.File(byteImage, filename='image.png')
    return image

async def frankenstein(message):
    global file_alias, files, filenames, ambiguous_names
    from commands.query_card import file_alias, files, filenames, ambiguous_names, match_ratio

    if not file_alias:
        await try_open_alias(message)
        from commands.query_card import file_alias
    if not files:
        status = populate_files()
        if status != 200:
            await message.channel.send(f"Error {status} when requesting github.")
            return
        from commands.query_card import files, ambiguous_names
    if not filenames:
        filenames = files.keys()

    parts = message.content.split(".frankenstein")[1].strip().lower().split(",")
    images = []
    for part in parts:
        creature = part.strip().replace(" ", "_").lower()
        if not creature.endswith(".png"):
            creature += ".png"
        try:
            closest = difflib.get_close_matches(creature, filenames, n=1, cutoff=match_ratio)[0]
        except IndexError:
            await message.channel.send("No card found for query %s!" % creature)
            return
        images.append(url_to_image(f"https://raw.githubusercontent.com/MichaelJSr/TTSCardMaker/main/{files[closest]}"))

        # If the filename was ambiguous, make a note of that.
        if closest in ambiguous_names:
            ambiguous_message = f"Ambiguous name found for {closest}. If this wasn't the card you wanted, try typing: \n"
            for i in ambiguous_names[closest]:
                ambiguous_message += f"{i}\n"
            await message.channel.send(ambiguous_message)
    
    frankensteins_monster = images[-1]
    total_images = len(images)
    cur_image = total_images
    for image in reversed(images):
        height = image.shape[0]
        height_part = int(height / total_images)
        frankensteins_monster[0:height_part * cur_image] = image[0:height_part * cur_image]
        cur_image -= 1

    await message.channel.send(file=cv2discordfile(frankensteins_monster))
