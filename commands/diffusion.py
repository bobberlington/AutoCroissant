from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig, DPMSolverSinglestepScheduler, EulerAncestralDiscreteScheduler, AutoPipelineForInpainting, AutoPipelineForImage2Image, StableDiffusionPipeline, StableDiffusionXLPipeline, FluxTransformer2DModel, FluxPipeline, FlowMatchEulerDiscreteScheduler
from diffusers.utils import load_image
from discord import Attachment, Interaction
from gc import collect
from os import listdir
import torch
from transformers import BitsAndBytesConfig as BitsAndBytesConfig, T5EncoderModel, CLIPTokenizer
from typing import Optional
from queue import Queue

from commands.utils import pildiscordfile, messages, edit_messages, files

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
queue: Queue[tuple[Interaction, str, Optional[Attachment], Optional[Attachment], Optional[str], Optional[str], Optional[int], Optional[int], Optional[int], Optional[float], Optional[float], Optional[float], Optional[int]]] = Queue()

async def get_qsize(interaction: Interaction):
    index = 0
    queued_prompts = ""
    for q in queue.queue:
        index += 1
        queued_prompts += f"#{index}: steps={q[6]}, height={q[7]}, width={q[8]}, resize={q[9]}, cfg={q[10]}, strength={q[11]}, seed={q[12]}, scheduler={scheduler_name}, prompt={q[1]}\n\n"
    await interaction.response.send_message("```Queue Size: " + str(index) + "\n" + queued_prompts + "```")

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
        diffusion(*queue.get_nowait())

async def set_scheduler(interaction: Interaction, new_scheduler: str):
    global scheduler_name, txt2img_pipe, img2img_pipe, inpaint_pipe
    if not new_scheduler or new_scheduler not in possible_schedulers:
        await interaction.response.send_message("The old scheduler was:\n%s\nPossible choices are:\n%s" % (scheduler_name, '\n'.join(possible_schedulers)))
        return

    scheduler_name = new_scheduler
    txt2img_pipe = img2img_pipe = inpaint_pipe = None
    collect()
    torch.cuda.empty_cache()
    await interaction.response.send_message("New scheduler of %s set!" % scheduler_name)

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
    safetensors = ["flux"]
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

def progress_check(interaction: Interaction, total_steps: int, seed: int, pipe: StableDiffusionPipeline | StableDiffusionXLPipeline | FluxPipeline, step: int, timestep: torch.Tensor, callback_kwargs: dict):
    # If this does not equal 0, return (thus we only print progress every 10% of the way done)
    if step % (total_steps // 10):
        return callback_kwargs
    edit_messages.append((interaction, f"Seed= {seed}\nProgress= {round(step / total_steps * 100.0)}%", []))
    return callback_kwargs

def diffusion(interaction: Interaction, prompt: str = None, image_param: Attachment = None, mask_image_param: Attachment = None, url: str = None, mask_url: str = None, steps: int = 50, height: int = 512, width: int = 512, resize: float = 1.0, cfg: float = 7.0, strength: float = 0.8, seed: int = None):
    global in_progress
    if in_progress:
        messages.append((interaction, "Request queued after the current generation."))
        queue.put_nowait((interaction, prompt, image_param, mask_image_param, url, mask_url, steps, height, width, resize, cfg, strength, seed))
        return

    if not txt2img_pipe:
        messages.append((interaction, "Initializing pipeline, this will take a while..."))
        init_pipeline()
    in_progress = True

    image = None
    if image_param:
        image = load_image(image_param.url)
    elif url:
        image = load_image(url)
    mask_image = None
    if mask_image_param:
        mask_image = load_image(mask_image_param.url)
    elif mask_url:
        mask_image = load_image(url)

    generator = torch.Generator("cuda" if vram_usage != "mps" else "mps")
    if seed:
        generator = generator.manual_seed(seed)
    else:
        seed = generator.seed()
    if image and not (width and height):
        width, height = image.size
        width = int(width * resize)
        height = int(height * resize)
    messages.append((interaction, f"steps={steps}, height={height}, width={width}, resize={resize}, cfg={cfg}, strength={strength}, seed={generator.initial_seed()}, scheduler={scheduler_name}, prompt={prompt}"))

    collect()
    torch.cuda.empty_cache()
    if mask_image:
        files.append((interaction, pildiscordfile(inpaint_pipe(image=image, mask_image=mask_image, height=height, width=width, strength=strength, prompt=prompt, num_inference_steps=steps,
                                                               guidance_scale=cfg, generator=generator, callback_on_step_end=(lambda *args: progress_check(interaction, steps, seed, *args)), callback_on_step_end_tensor_inputs=["latents"]).images[0])))
    elif image:
        files.append((interaction, pildiscordfile(img2img_pipe(image=image, height=height, width=width, strength=strength, prompt=prompt, num_inference_steps=steps,
                                                               guidance_scale=cfg, generator=generator, callback_on_step_end=(lambda *args: progress_check(interaction, steps, seed, *args)), callback_on_step_end_tensor_inputs=["latents"]).images[0])))
    else:
        files.append((interaction, pildiscordfile(txt2img_pipe(prompt=prompt, num_inference_steps=steps, height=height, width=width, guidance_scale=cfg, generator=generator,
                                                               callback_on_step_end=(lambda *args: progress_check(interaction, steps, seed, *args)), callback_on_step_end_tensor_inputs=["latents"]).images[0])))
    in_progress = False
    if queue.qsize() > 0:
        diffusion(*queue.get_nowait())
