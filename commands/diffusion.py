import discord
import torch
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline

from commands.tools import to_thread, url_to_pilimage, pildiscordfile, messages, files

mfolder = "./models/"
model = "rpg_v5.safetensors"
negative_prompt = "nsfw, lowres, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, blurry"
txt2img_pipe = img2img_pipe = None

def parse_msg_image(message: discord.Message):
    image = None
    if len(message.attachments) > 0 and message.attachments[0].content_type.startswith("image"):
        image = url_to_pilimage(message.attachments[0].url)
    else:
        query = message.content.split(".ai")[1].strip().lower()
        if query.startswith("http"):
            image = url_to_pilimage(query.split()[0])
    
    return image

def init_pipeline():
    txt2img_pipe = StableDiffusionPipeline.from_single_file(mfolder+model, torch_dtype=torch.float16, safety_checker=None, use_safetensors=True)
    if torch.cuda.is_available():
        txt2img_pipe.to("cuda")
    txt2img_pipe.enable_attention_slicing()

    img2img_pipe = StableDiffusionImg2ImgPipeline.from_pipe(txt2img_pipe)
    return txt2img_pipe, img2img_pipe

@to_thread
def diffusion(message: discord.Message):
    global txt2img_pipe, img2img_pipe
    if txt2img_pipe is None:
        messages.append((message.channel.id, "Initializing pipeline, this will take a while..."))
        txt2img_pipe, img2img_pipe = init_pipeline()

    prompt = ""
    steps = 50
    height = 512
    width = 512
    guidance = 7
    image = parse_msg_image(message)

    query = message.content.split(".ai")[1].replace(",", "").strip().lower().split()
    if len(query) > 0 and query[0].startswith("http"):
        query.pop(0)

    for word in query.copy():
        if word.startswith("steps="):
            steps = int(word[len("steps="):])
            query.remove(word)
        elif word.startswith("height="):
            height = int(word[len("height="):])
            query.remove(word)
        elif word.startswith("width="):
            width = int(word[len("width="):])
            query.remove(word)
        elif word.startswith("guidance="):
            guidance = int(word[len("guidance="):])
            query.remove(word)
    prompt = " ".join(query)
    messages.append((message.channel.id, f"steps={steps}, height={height}, width={width}, guidance={guidance}, prompt={prompt}."))

    if image is None:
        files.append((message.channel.id, pildiscordfile(txt2img_pipe(prompt=prompt, negative_prompt=negative_prompt, num_inference_steps=steps, height=height, width=width, guidance_scale=guidance).images[0])))
