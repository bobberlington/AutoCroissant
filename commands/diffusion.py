from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from commands.convert_nf4_flux import _replace_with_bnb_linear, create_quantized_param, check_quantized_param
from diffusers import DPMSolverSinglestepScheduler, EulerAncestralDiscreteScheduler, AutoPipelineForInpainting, AutoPipelineForImage2Image, StableDiffusionPipeline, StableDiffusionXLPipeline, FluxTransformer2DModel, FluxPipeline, FlowMatchEulerDiscreteScheduler, AutoencoderKL
import discord
from gc import collect
from huggingface_hub import hf_hub_download
from os import listdir
import safetensors.torch
import torch
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
from queue import Queue

from commands.tools import url_to_pilimage, pildiscordfile, messages, files
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
queue: Queue[discord.Message] = Queue()

def parse_msg_image(message: discord.Message):
    image = mask_image = None
    split_query = message.content.split()
    if len(split_query) > 1 and split_query[1].startswith("http"):
        image = url_to_pilimage(split_query[1])
        if len(split_query) > 2 and split_query[2].startswith("http"):
            mask_image = url_to_pilimage(split_query[2])
    if len(message.attachments) > 0 and message.attachments[0].content_type.startswith("image"):
        if not image:
            image = url_to_pilimage(message.attachments[0].url)
        elif not mask_image:
            mask_image = url_to_pilimage(message.attachments[0].url)
        if len(message.attachments) > 1 and message.attachments[1].content_type.startswith("image"):
            if not mask_image:
                mask_image = url_to_pilimage(message.attachments[1].url)
    return image, mask_image

async def get_qsize(message: discord.Message):
    await message.channel.send(f"Current ai queue size: {queue.qsize()}")
    index = 1
    for q in queue.queue:
        await message.channel.send(f"#{index}: {q.content}")
        index += 1

def init_pipeline():
    global txt2img_pipe, img2img_pipe, inpaint_pipe, in_progress
    in_progress = True
    dtype = torch.float16
    scheduler = None

    if model == "flux":
        bfl_repo = "black-forest-labs/FLUX.1-dev"
        device_map = 'balanced' if torch.cuda.device_count() > 1 else None
        is_torch_e4m3fn_available = hasattr(torch, "float8_e4m3fn")
        ckpt_path = hf_hub_download("sayakpaul/flux.1-dev-nf4", filename="diffusion_pytorch_model.safetensors")
        original_state_dict = safetensors.torch.load_file(ckpt_path)
        with init_empty_weights():
            flux_model = FluxTransformer2DModel.from_config(FluxTransformer2DModel.load_config("sayakpaul/flux.1-dev-nf4"), device_map=device_map).to(dtype)
            expected_state_dict_keys = list(flux_model.state_dict().keys())
        _replace_with_bnb_linear(flux_model, "nf4")
        for param_name, param in original_state_dict.items():
            if param_name not in expected_state_dict_keys:
                continue
            is_param_float8_e4m3fn = is_torch_e4m3fn_available and param.dtype == torch.float8_e4m3fn
            if torch.is_floating_point(param) and not is_param_float8_e4m3fn:
                param = param.to(dtype)
            if not check_quantized_param(flux_model, param_name):
                set_module_tensor_to_device(flux_model, param_name, device=device_no, value=param)
            else:
                create_quantized_param(flux_model, param, param_name, target_device=device_no, state_dict=original_state_dict, pre_quantized=True)
        del original_state_dict
        collect()
        torch.cuda.empty_cache()
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(bfl_repo, subfolder="scheduler", token=config.hf_token, device_map=device_map)
        text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=dtype, token=config.hf_token, device_map=device_map)
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=dtype, clean_up_tokenization_spaces=True, token=config.hf_token, device_map=device_map)
        #tokenizer_2 = T5TokenizerFast.from_pretrained(bfl_repo, subfolder="tokenizer_2", torch_dtype=dtype, clean_up_tokenization_spaces=True, token=config.hf_token, device_map=device_map)
        #text_encoder_2 = T5EncoderModel.from_pretrained(bfl_repo, subfolder="text_encoder_2", torch_dtype=dtype, token=config.hf_token, device_map=device_map)
        #vae = AutoencoderKL.from_pretrained(bfl_repo, subfolder="vae", torch_dtype=dtype, token=config.hf_token, device_map=device_map)
        txt2img_pipe = FluxPipeline.from_pretrained(bfl_repo, scheduler=scheduler, text_encoder=text_encoder, tokenizer=tokenizer, transformer=flux_model, torch_dtype=dtype, token=config.hf_token, device_map=device_map)
    elif model.lower().find("xl") != -1:
        txt2img_pipe = StableDiffusionXLPipeline.from_single_file(mfolder+model, torch_dtype=dtype, safety_checker=None, use_safetensors=True, add_watermarker=False)
    else:
        txt2img_pipe = StableDiffusionPipeline.from_single_file(mfolder+model, torch_dtype=dtype, safety_checker=None, use_safetensors=True)

    if scheduler_name.startswith("dpm++ sde"):
        scheduler = DPMSolverSinglestepScheduler.from_config(txt2img_pipe.scheduler.config)
        scheduler.config.lower_order_final = True
    elif scheduler_name.startswith("euler a"):
        scheduler = EulerAncestralDiscreteScheduler.from_config(txt2img_pipe.scheduler.config)
    if scheduler_name.endswith("karras"):
        scheduler.config.use_karras_sigmas = True
    if scheduler and model != "flux":
        txt2img_pipe.scheduler = scheduler

    if lora:
        txt2img_pipe.load_lora_weights(lfolder+lora)

    if vram_usage != "high":
        txt2img_pipe.vae.enable_slicing()
        txt2img_pipe.vae.enable_tiling()
    if vram_usage == "medium" and (model != "flux" or torch.cuda.device_count() == 1):
        txt2img_pipe.enable_model_cpu_offload(gpu_id=device_no)
    elif vram_usage == "low" and model != "flux":
        txt2img_pipe.enable_sequential_cpu_offload(gpu_id=device_no)
    if model != "flux":
        img2img_pipe = AutoPipelineForImage2Image.from_pipe(txt2img_pipe)
        inpaint_pipe = AutoPipelineForInpainting.from_pipe(txt2img_pipe)

    collect()
    torch.cuda.empty_cache()
    in_progress = False
    print("Finished initializing the pipeline.")
    if queue.qsize() > 0:
        diffusion(queue.get_nowait())

async def set_scheduler(message: discord.Message):
    global scheduler_name, txt2img_pipe, img2img_pipe, inpaint_pipe
    if len(message.content.split()) < 2 or not ' '.join(message.content.split()[1:]) in possible_schedulers:
        await message.channel.send("Must specify exactly one argument, the new scheduler. The old scheduler was:\n%s\nPossible choices are:\n%s" % (scheduler_name, '\n'.join(possible_schedulers)))
        return

    scheduler_name = ' '.join(message.content.split()[1:])
    txt2img_pipe = img2img_pipe = inpaint_pipe = None
    collect()
    torch.cuda.empty_cache()
    await message.channel.send("New scheduler of %s set!" % scheduler_name)

async def set_device(message: discord.Message):
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

async def set_model(message: discord.Message):
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

async def set_lora(message: discord.Message):
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

def diffusion(message: discord.Message):
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

    generator = torch.Generator("cuda")
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
    if image:
        width, height = image.size
        width = int(width * resize)
        height = int(height * resize)
    messages.append((message.channel.id, f"steps={steps}, height={height}, width={width}, resize={resize}, cfg={cfg}, strength={strength}, seed={generator.initial_seed()}, scheduler={scheduler_name}, prompt={prompt}."))

    collect()
    torch.cuda.empty_cache()
    if mask_image:
        files.append((message.channel.id, pildiscordfile(inpaint_pipe(image=image.convert("RGB").resize((width, height)), mask_image=mask_image.resize((width, height)), strength=strength, prompt=prompt, num_inference_steps=steps, guidance_scale=cfg, generator=generator).images[0])))
    elif image:
        files.append((message.channel.id, pildiscordfile(img2img_pipe(image=image.convert("RGB").resize((width, height)), strength=strength, prompt=prompt, num_inference_steps=steps, guidance_scale=cfg, generator=generator).images[0])))
    else:
        files.append((message.channel.id, pildiscordfile(txt2img_pipe(prompt=prompt, num_inference_steps=steps, height=height, width=width, guidance_scale=cfg, generator=generator).images[0])))
    in_progress = False
    if queue.qsize() > 0:
        diffusion(queue.get_nowait())
