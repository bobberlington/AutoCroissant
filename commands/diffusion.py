import discord
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig, DPMSolverSinglestepScheduler, EulerAncestralDiscreteScheduler, AutoPipelineForInpainting, AutoPipelineForImage2Image, StableDiffusionPipeline, StableDiffusionXLPipeline, FluxTransformer2DModel, FluxPipeline, FlowMatchEulerDiscreteScheduler
from diffusers.utils import load_image
from discord import Message, Interaction
from gc import collect
from os import listdir
import torch
from transformers import BitsAndBytesConfig as BitsAndBytesConfig, T5EncoderModel, CLIPTokenizer
from queue import Queue

from commands.utils import pildiscordfile, messages, files

mfolder = "./models/"
lfolder = f"{mfolder}loras/"
model = ""
lora = ""
device_no = 0
scheduler_name = ""
vram_usage = "low"
try:
    import config
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
queue: Queue[Interaction] = Queue()

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

async def get_qsize(interaction: Interaction):
    await interaction.response.send_message(f"Current ai queue size: {queue.qsize()}")
    index = 1
    for q in queue.queue:
        await interaction.followup.send(f"#{index}: {q.content}")
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

async def set_scheduler(interaction: Interaction, new_scheduler: str):
    global scheduler_name, txt2img_pipe, img2img_pipe, inpaint_pipe
    if not new_scheduler or new_scheduler not in possible_schedulers:
        await interaction.response.send_message("The old scheduler was:\n%s\nPossible choices are:\n%s" % (scheduler_name, '\n'.join(possible_schedulers)))
        return

    scheduler_name = new_scheduler
    txt2img_pipe = img2img_pipe = inpaint_pipe = None
    collect()
    torch.cuda.empty_cache()
    await interaction.followup.send("New scheduler of %s set!" % scheduler_name)

async def set_device(interaction: Interaction, new_device: int):
    global device_no, txt2img_pipe, img2img_pipe, inpaint_pipe
    if not new_device:
        await interaction.response.send_message("The current device is %d" % device_no)
        return

    try:
        device_no = new_device
    except:
        await interaction.response.send_message("Invalid device#.")
        return
    txt2img_pipe = img2img_pipe = inpaint_pipe = None
    collect()
    torch.cuda.empty_cache()
    await interaction.response.send_message("New device# of %d set!" % device_no)

async def set_model(interaction: Interaction, new_model: str):
    global model, txt2img_pipe, img2img_pipe, inpaint_pipe
    safetensors = []
    for m in listdir(mfolder):
        if m.endswith('.safetensors'):
            safetensors.append(m)

    if not new_model or new_model not in safetensors:
        await interaction.response.send_message("The old model was:\n%s\nPossible choices are:\n%s" % (model, '\n'.join(safetensors)))
        return

    model = new_model
    txt2img_pipe = img2img_pipe = inpaint_pipe = None
    collect()
    torch.cuda.empty_cache()
    await interaction.response.send_message("New model of %s set!" % model)

async def set_lora(interaction: Interaction, new_lora: str):
    global lora, txt2img_pipe, img2img_pipe, inpaint_pipe
    safetensors = []
    for m in listdir(lfolder):
        if m.endswith('.safetensors'):
            safetensors.append(m)
    if not new_lora or new_lora not in safetensors:
        await interaction.response.send_message("The old lora was:\n%s\nPossible choices are:\n%s" % (lora, '\n'.join(safetensors)))
        return

    lora = new_lora
    txt2img_pipe = img2img_pipe = inpaint_pipe = None
    collect()
    torch.cuda.empty_cache()
    await interaction.response.send_message("New model of %s set!" % lora)

def diffusion(interaction: Interaction, image_param: discord.Attachment, mask_image_param: discord.Attachment, url: str, mask_url: str,prompt: str):
    global in_progress
    if in_progress:
        messages.append((interaction, "Request queued after the current generation."))
        queue.put_nowait(interaction)
        return

    if not txt2img_pipe:
        messages.append((interaction, "Initializing pipeline, this will take a while..."))
        init_pipeline()

    in_progress = True
    steps = 50
    height = 512
    width = 512
    resize = 1
    cfg = 7
    strength = 0.8
    if image_param:
        image = load_image(image_param.url)
    elif url:
        image = load_image(url)
    else:
        image = None
    if mask_image_param:
        mask_image = load_image(mask_image_param.url)
    elif mask_url:
        mask_image = load_image(url)
    else:
        image = None

    query = prompt.replace(",", " ").lower().split()
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
    messages.append((interaction, f"steps={steps}, height={height}, width={width}, resize={resize}, cfg={cfg}, strength={strength}, seed={generator.initial_seed()}, scheduler={scheduler_name}, prompt={prompt}."))

    collect()
    torch.cuda.empty_cache()
    if mask_image:
        files.append((interaction, pildiscordfile(inpaint_pipe(image=image, mask_image=mask_image, height=height, width=width, strength=strength, prompt=prompt, num_inference_steps=steps, guidance_scale=cfg, generator=generator).images[0])))
    elif image:
        files.append((interaction, pildiscordfile(img2img_pipe(image=image, height=height, width=width, strength=strength, prompt=prompt, num_inference_steps=steps, guidance_scale=cfg, generator=generator).images[0])))
    else:
        files.append((interaction, pildiscordfile(txt2img_pipe(prompt=prompt, num_inference_steps=steps, height=height, width=width, guidance_scale=cfg, generator=generator).images[0])))
    in_progress = False
    if queue.qsize() > 0:
        diffusion(queue.get_nowait())
