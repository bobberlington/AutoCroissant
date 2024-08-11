from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
import discord
import torch

from commands.tools import to_thread, url_to_pilimage, pildiscordfile, messages, files

mfolder = "./models/"
model = "rpg_v5.safetensors"
device = "gpu"
device_no = "1"
negative_prompt = "nsfw, lowres, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, blurry"
txt2img_pipe = img2img_pipe = None

def parse_msg_image(message: discord.Message):
    if len(message.attachments) > 0 and message.attachments[0].content_type.startswith("image"):
        return url_to_pilimage(message.attachments[0].url)
    else:
        query = message.content.split(".ai")[1].strip()
        if query.startswith("http"):
            return url_to_pilimage(query.split()[0])

def init_pipeline():
    dtype = torch.float16
    if device == "cpu":
        dtype = torch.float32
    txt2img_pipe = StableDiffusionPipeline.from_single_file(mfolder+model, torch_dtype=dtype, safety_checker=None, use_safetensors=True)
    if device == "gpu" and torch.cuda.is_available():
        txt2img_pipe.to(f"cuda:{device_no}")
    txt2img_pipe.enable_attention_slicing()

    img2img_pipe = StableDiffusionImg2ImgPipeline.from_pipe(txt2img_pipe)
    torch.cuda.empty_cache()
    return txt2img_pipe, img2img_pipe

async def set_device(message: discord.Message):
    global device, device_no

    if len(message.content.split()) < 3:
        await message.channel.send("Must specify exactly two arguments, the new device and device #. The old device is %s and the old device# is %s." % (device, device_no))
        return

    device, device_no = message.content.split()[1:]

    await message.channel.send("New device of %s and device# of %s set!" % (device, device_no))

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
    resize = 1
    guidance = 7
    strength = 0.8
    image = parse_msg_image(message)

    query = message.content.split(".ai")[1].replace(",", " ").replace("  ", " ").strip().lower().split()
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
        elif word.startswith("resize="):
            resize = float(word[len("resize="):])
            query.remove(word)
        elif word.startswith("guidance="):
            guidance = int(word[len("guidance="):])
            query.remove(word)
        elif word.startswith("strength="):
            strength = float(word[len("strength="):])
            query.remove(word)
    prompt = " ".join(query)
    messages.append((message.channel.id, f"steps={steps}, height={height}, width={width}, resize={resize}, guidance={guidance}, strength={strength}, prompt={prompt}."))

    if image is None:
        files.append((message.channel.id, pildiscordfile(txt2img_pipe(prompt=prompt, negative_prompt=negative_prompt, num_inference_steps=steps, height=height, width=width, guidance_scale=guidance).images[0])))
    else:
        width, height = image.size
        image = image.convert("RGB").resize((int(width * resize), int(height * resize)))
        files.append((message.channel.id, pildiscordfile(img2img_pipe(image=image, strength=strength, prompt=prompt, negative_prompt=negative_prompt, num_inference_steps=steps, guidance_scale=guidance).images[0])))
