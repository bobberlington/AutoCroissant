from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig, DPMSolverSinglestepScheduler, EulerAncestralDiscreteScheduler, AutoPipelineForInpainting, AutoPipelineForImage2Image, StableDiffusionPipeline, StableDiffusionXLPipeline, FluxTransformer2DModel, FluxPipeline, FlowMatchEulerDiscreteScheduler
from diffusers.utils import load_image
from discord import Message
from gc import collect
from os import listdir
import torch
from transformers import BitsAndBytesConfig as BitsAndBytesConfig, T5EncoderModel, CLIPTokenizer
from queue import Queue

from commands.utils import pildiscordfile, messages, files
import config

mfolder = "./models/"
lfolder = f"{mfolder}loras/"
model = ""
lora = ""
device_no = 0
scheduler_name = ""
vram_usage = "low"
try:
    model = config.model
    lora = config.lora
    device_no = config.device_no
    scheduler_name = config.scheduler_name
    vram_usage = config.vram_usage
except AttributeError:
    print("No diffusion params in config, skipping.")

txt2img_pipe = img2img_pipe = inpaint_pipe = None
in_progress = False
possible_schedulers = ["dpm++ sde", "dpm++ sde karras", "euler a"]
queue: Queue[Message] = Queue()

def parse_msg_image(message: Message):
    image = mask_image = None
    split_query = message.content.split()
    if len(split_query) > 1 and split_query[1].startswith("http"):
        image = load_image(split_query[1])
        if len(split_query) > 2 and split_query[2].startswith("http"):
            mask_image = load_image(split_query[2])
    if len(message.attachments) > 0 and message.attachments[0].content_type.startswith("image"):
        if not image:
            image = load_image(message.attachments[0].url)
        elif not mask_image:
            mask_image = load_image(message.attachments[0].url)
        if len(message.attachments) > 1 and message.attachments[1].content_type.startswith("image"):
            if not mask_image:
                mask_image = load_image(message.attachments[1].url)
    return image, mask_image

async def get_qsize(message: Message):
    await message.channel.send(f"Current ai queue size: {queue.qsize()}")
    index = 1
    for q in queue.queue:
        await message.channel.send(f"#{index}: {q.content}")
        index += 1

def init_pipeline():
    if model == "":
        print("No model to initialize, finished.")
        return

    global txt2img_pipe, img2img_pipe, inpaint_pipe, in_progress
    in_progress = True
    dtype = torch.float16 if vram_usage != "mps" else torch.float32
    scheduler = None
    device_map = 'balanced' if vram_usage == "distributed" and torch.cuda.device_count() > 1 else None

    if model.lower().find("flux") != -1:
        bfl_repo = "black-forest-labs/FLUX.1-dev"
        if vram_usage == "mps":
            txt2img_pipe = FluxPipeline.from_pretrained(bfl_repo, torch_dtype=dtype, token=config.hf_token)
        else:
            quant_config = DiffusersBitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
            transformer = FluxTransformer2DModel.from_pretrained(
                bfl_repo,
                subfolder="transformer",
                quantization_config=quant_config,
                torch_dtype=dtype,
                token=config.hf_token,
            )
            quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
            text_encoder = T5EncoderModel.from_pretrained(
                bfl_repo,
                subfolder="text_encoder_2",
                quantization_config=quant_config,
                torch_dtype=dtype,
                device_map=device_map,
                token=config.hf_token,
            )
            tokenizer = CLIPTokenizer.from_pretrained(
                "openai/clip-vit-large-patch14",
                quantization_config=quant_config,
                torch_dtype=dtype,
                clean_up_tokenization_spaces=True,
                device_map=device_map,
                token=config.hf_token,
            )
            scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
                bfl_repo,
                subfolder="scheduler",
                device_map=device_map,
                token=config.hf_token,
            )
            collect()
            torch.cuda.empty_cache()
            txt2img_pipe = FluxPipeline.from_pretrained(
                bfl_repo,
                text_encoder_2=text_encoder,
                transformer=transformer,
                tokenizer=tokenizer,
                torch_dtype=dtype,
                device_map=device_map,
                token=config.hf_token,
            )
    elif model.lower().find("xl") != -1:
        txt2img_pipe = StableDiffusionXLPipeline.from_single_file(mfolder+model, torch_dtype=dtype, safety_checker=None, use_safetensors=True, add_watermarker=False)
    else:
        txt2img_pipe = StableDiffusionPipeline.from_single_file(mfolder+model, torch_dtype=dtype, safety_checker=None, use_safetensors=True)

    if vram_usage == "mps":
        txt2img_pipe.to(device='mps')

    if not scheduler and scheduler_name.startswith("dpm++ sde"):
        scheduler = DPMSolverSinglestepScheduler.from_config(txt2img_pipe.scheduler.config)
        scheduler.config.lower_order_final = True
    elif not scheduler and scheduler_name.startswith("euler a"):
        scheduler = EulerAncestralDiscreteScheduler.from_config(txt2img_pipe.scheduler.config)
    if scheduler_name.endswith("karras") and model.lower().find("flux") == -1:
        scheduler.config.use_karras_sigmas = True
    if scheduler:
        txt2img_pipe.scheduler = scheduler

    if lora:
        txt2img_pipe.load_lora_weights(lfolder+lora)

    if vram_usage != "high":
        txt2img_pipe.vae.enable_slicing()
        txt2img_pipe.vae.enable_tiling()
    if vram_usage == "medium":
        txt2img_pipe.enable_model_cpu_offload(gpu_id=device_no)
    elif vram_usage == "low":
        txt2img_pipe.enable_sequential_cpu_offload(gpu_id=device_no)

    img2img_pipe = AutoPipelineForImage2Image.from_pipe(txt2img_pipe)
    inpaint_pipe = AutoPipelineForInpainting.from_pipe(txt2img_pipe)

    collect()
    torch.cuda.empty_cache()
    in_progress = False
    print("Finished initializing the pipeline.")
    if queue.qsize() > 0:
        diffusion(queue.get_nowait())

async def set_scheduler(message: Message):
    global scheduler_name, txt2img_pipe, img2img_pipe, inpaint_pipe
    if len(message.content.split()) < 2 or not ' '.join(message.content.split()[1:]) in possible_schedulers:
        await message.channel.send("Must specify exactly one argument, the new scheduler. The old scheduler was:\n%s\nPossible choices are:\n%s" % (scheduler_name, '\n'.join(possible_schedulers)))
        return

    scheduler_name = ' '.join(message.content.split()[1:])
    txt2img_pipe = img2img_pipe = inpaint_pipe = None
    collect()
    torch.cuda.empty_cache()
    await message.channel.send("New scheduler of %s set!" % scheduler_name)

async def set_device(message: Message):
    global device_no, txt2img_pipe, img2img_pipe, inpaint_pipe
    if len(message.content.split()) < 2:
        await message.channel.send("Must specify exactly one argument, the new device#. The old device# was: %s" % device_no)
        return

    try:
        device_no = int(message.content.split()[1])
    except:
        await message.channel.send("Invalid device#.")
        return
    txt2img_pipe = img2img_pipe = inpaint_pipe = None
    collect()
    torch.cuda.empty_cache()
    await message.channel.send("New device# of %s set!" % device_no)

async def set_model(message: Message):
    global model, txt2img_pipe, img2img_pipe, inpaint_pipe
    if len(message.content.split()) < 2:
        safetensors = []
        for m in listdir(mfolder):
            if m.endswith('.safetensors'):
                safetensors.append(m)
        await message.channel.send("Must specify exactly one argument, the new model. The old model was:\n%s\nPossible choices are:\n%s" % (model, '\n'.join(safetensors)))
        return

    model = message.content.split()[1]
    txt2img_pipe = img2img_pipe = inpaint_pipe = None
    collect()
    torch.cuda.empty_cache()
    await message.channel.send("New model of %s set!" % model)

async def set_lora(message: Message):
    global lora, txt2img_pipe, img2img_pipe, inpaint_pipe
    if len(message.content.split()) < 2:
        safetensors = []
        for m in listdir(lfolder):
            if m.endswith('.safetensors'):
                safetensors.append(m)
        await message.channel.send("Must specify exactly one argument, the new lora. The old lora was:\n%s\nPossible choices are:\n%s" % (lora, '\n'.join(safetensors)))
        return

    lora = message.content.split()[1]
    txt2img_pipe = img2img_pipe = inpaint_pipe = None
    collect()
    torch.cuda.empty_cache()
    await message.channel.send("New model of %s set!" % lora)

def diffusion(message: Message):
    global in_progress
    if in_progress:
        messages.append((message.channel.id, "Request queued after the current generation."))
        queue.put_nowait(message)
        return

    if not txt2img_pipe:
        messages.append((message.channel.id, "Initializing pipeline, this will take a while..."))
        init_pipeline()

    in_progress = True
    prompt = ""
    steps = 50
    height = 512
    width = 512
    resize = 1
    cfg = 7
    strength = 0.8
    image, mask_image = parse_msg_image(message)

    query = message.content.split(".ai")[1].replace(",", " ").lower().split()
    while len(query) > 0 and query[0].startswith("http"):
        query.pop(0)

    generator = torch.Generator("cuda" if vram_usage != "mps" else "mps")
    generator.seed()
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
        elif word.startswith("cfg="):
            cfg = float(word[len("cfg="):])
            query.remove(word)
        elif word.startswith("strength="):
            strength = float(word[len("strength="):])
            query.remove(word)
        elif word.startswith("seed="):
            generator = generator.manual_seed(int(word[len("seed="):]))
            query.remove(word)
    prompt = " ".join(query)
    if image and not (width and height):
        width, height = image.size
        width = int(width * resize)
        height = int(height * resize)
    messages.append((message.channel.id, f"steps={steps}, height={height}, width={width}, resize={resize}, cfg={cfg}, strength={strength}, seed={generator.initial_seed()}, scheduler={scheduler_name}, prompt={prompt}."))

    collect()
    torch.cuda.empty_cache()
    if mask_image:
        files.append((message.channel.id, pildiscordfile(inpaint_pipe(image=image, mask_image=mask_image, height=height, width=width, strength=strength, prompt=prompt, num_inference_steps=steps, guidance_scale=cfg, generator=generator).images[0])))
    elif image:
        files.append((message.channel.id, pildiscordfile(img2img_pipe(image=image, height=height, width=width, strength=strength, prompt=prompt, num_inference_steps=steps, guidance_scale=cfg, generator=generator).images[0])))
    else:
        files.append((message.channel.id, pildiscordfile(txt2img_pipe(prompt=prompt, num_inference_steps=steps, height=height, width=width, guidance_scale=cfg, generator=generator).images[0])))
    in_progress = False
    if queue.qsize() > 0:
        diffusion(queue.get_nowait())
