from diffusers import (
    BitsAndBytesConfig as DiffusersBitsAndBytesConfig,
    DPMSolverSinglestepScheduler,
    EulerAncestralDiscreteScheduler,
    AutoPipelineForInpainting,
    AutoPipelineForImage2Image,
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    FluxTransformer2DModel,
    FluxPipeline,
    FlowMatchEulerDiscreteScheduler,
)
from diffusers.utils import load_image
from discord import Attachment, Interaction
from gc import collect
from os import listdir
from os.path import exists, join
from PIL.Image import Image, Resampling, fromarray
from dataclasses import dataclass
from transformers import BitsAndBytesConfig, T5EncoderModel, CLIPTokenizer
from typing import Optional
from queue import Queue
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("WARNING: Torch not installed, diffusion commands will not work.")

import config
from commands.utils import (
    pildiscordfile,
    queue_message,
    queue_edit,
    queue_file,
    queue_command,
)

# ========================
# Configuration
# ========================
MODELS_FOLDER = "./models/"
LORAS_FOLDER = f"{MODELS_FOLDER}loras/"
MODEL = getattr(config, "model", "")
LORA = getattr(config, "lora", "")
DEVICE_NO = getattr(config, "device_no", 0)
SCHEDULER_NAME = getattr(config, "scheduler_name", "")
VRAM_USAGE = getattr(config, "vram_usage", "low")
HF_TOKEN = getattr(config, "HF_TOKEN", "")

POSSIBLE_SCHEDULERS = ["dpm++ sde", "dpm++ sde karras", "euler a"]

# https://github.com/comfyanonymous/ComfyUI/blob/master/comfy/latent_formats.py
SD15_RGB_FACTORS = [
                    #   R        G        B
                    [ 0.3512,  0.2297,  0.3227],
                    [ 0.3250,  0.4974,  0.2350],
                    [-0.2829,  0.1762,  0.2721],
                    [-0.2120, -0.2616, -0.7177]
                ]

SDXL_RGB_FACTORS = [
                    #   R        G        B
                    [ 0.3651,  0.4232,  0.4341],
                    [-0.2533, -0.0042,  0.1068],
                    [ 0.1076,  0.1111, -0.0362],
                    [-0.3165, -0.2492, -0.2188]
                ]
SDXL_RGB_FACTORS_BIAS = [0.1084, -0.0175, -0.0011]

FLUX_RGB_FACTORS = [
                    [-0.0346,  0.0244,  0.0681],  # Channel 0
                    [-0.0346,  0.0244,  0.0681],
                    [-0.0346,  0.0244,  0.0681],
                    [-0.0346,  0.0244,  0.0681],
                    [ 0.0034,  0.0210,  0.0687],  # Channel 1
                    [ 0.0034,  0.0210,  0.0687],
                    [ 0.0034,  0.0210,  0.0687],
                    [ 0.0034,  0.0210,  0.0687],
                    [ 0.0275, -0.0668, -0.0433],  # Channel 2
                    [ 0.0275, -0.0668, -0.0433],
                    [ 0.0275, -0.0668, -0.0433],
                    [ 0.0275, -0.0668, -0.0433],
                    [-0.0174,  0.0160,  0.0617],  # Channel 3
                    [-0.0174,  0.0160,  0.0617],
                    [-0.0174,  0.0160,  0.0617],
                    [-0.0174,  0.0160,  0.0617],
                    [ 0.0859,  0.0721,  0.0329],  # Channel 4
                    [ 0.0859,  0.0721,  0.0329],
                    [ 0.0859,  0.0721,  0.0329],
                    [ 0.0859,  0.0721,  0.0329],
                    [ 0.0004,  0.0383,  0.0115],  # Channel 5
                    [ 0.0004,  0.0383,  0.0115],
                    [ 0.0004,  0.0383,  0.0115],
                    [ 0.0004,  0.0383,  0.0115],
                    [ 0.0405,  0.0861,  0.0915],  # Channel 6
                    [ 0.0405,  0.0861,  0.0915],
                    [ 0.0405,  0.0861,  0.0915],
                    [ 0.0405,  0.0861,  0.0915],
                    [-0.0236, -0.0185, -0.0259],  # Channel 7
                    [-0.0236, -0.0185, -0.0259],
                    [-0.0236, -0.0185, -0.0259],
                    [-0.0236, -0.0185, -0.0259],
                    [-0.0245,  0.0250,  0.1180],  # Channel 8
                    [-0.0245,  0.0250,  0.1180],
                    [-0.0245,  0.0250,  0.1180],
                    [-0.0245,  0.0250,  0.1180],
                    [ 0.1008,  0.0755, -0.0421],  # Channel 9
                    [ 0.1008,  0.0755, -0.0421],
                    [ 0.1008,  0.0755, -0.0421],
                    [ 0.1008,  0.0755, -0.0421],
                    [-0.0515,  0.0201,  0.0011],  # Channel 10
                    [-0.0515,  0.0201,  0.0011],
                    [-0.0515,  0.0201,  0.0011],
                    [-0.0515,  0.0201,  0.0011],
                    [ 0.0428, -0.0012, -0.0036],  # Channel 11
                    [ 0.0428, -0.0012, -0.0036],
                    [ 0.0428, -0.0012, -0.0036],
                    [ 0.0428, -0.0012, -0.0036],
                    [ 0.0817,  0.0765,  0.0749],  # Channel 12
                    [ 0.0817,  0.0765,  0.0749],
                    [ 0.0817,  0.0765,  0.0749],
                    [ 0.0817,  0.0765,  0.0749],
                    [-0.1264, -0.0522, -0.1103],  # Channel 13
                    [-0.1264, -0.0522, -0.1103],
                    [-0.1264, -0.0522, -0.1103],
                    [-0.1264, -0.0522, -0.1103],
                    [-0.0280, -0.0881, -0.0499],  # Channel 14
                    [-0.0280, -0.0881, -0.0499],
                    [-0.0280, -0.0881, -0.0499],
                    [-0.0280, -0.0881, -0.0499],
                    [-0.1262, -0.0982, -0.0778],  # Channel 15
                    [-0.1262, -0.0982, -0.0778],
                    [-0.1262, -0.0982, -0.0778],
                    [-0.1262, -0.0982, -0.0778]
]
FLUX_RGB_FACTORS_BIAS = [-0.0329, -0.0718, -0.0851]

# ========================
# Global State
# ========================
txt2img_pipe = None
img2img_pipe = None
inpaint_pipe = None
in_progress = False
request_queue: Queue = Queue()

flux_rgb_factors_tensor = None
flux_rgb_bias_tensor = None
sdxl_rgb_factors_tensor = None
sdxl_rgb_bias_tensor = None
sd15_rgb_factors_tensor = None


@dataclass
class GenerationRequest:
    """Data class for image generation requests."""
    interaction: Interaction
    prompt: str
    image_param: Optional[Attachment] = None
    mask_image_param: Optional[Attachment] = None
    url: Optional[str] = None
    mask_url: Optional[str] = None
    steps: int = 50
    height: int = 512
    width: int = 512
    resize: float = 1.0
    cfg: float = 7.0
    strength: float = 0.8
    seed: Optional[int] = None


# ========================
# Queue Management
# ========================
async def get_qsize(interaction: Interaction):
    """Display current queue status and queued requests."""
    if request_queue.empty():
        await interaction.response.send_message("Queue is empty.")
        return

    queued_items = list(request_queue.queue)

    # Format output
    output_lines = [f"Queue Size: {len(queued_items)}\n"]
    for idx, req in enumerate(queued_items, 1):
        output_lines.append(
            f"#{idx}: steps={req.steps}, height={req.height}, width={req.width}, "
            f"resize={req.resize}, cfg={req.cfg}, strength={req.strength}, "
            f"seed={req.seed}, scheduler={SCHEDULER_NAME}, prompt={req.prompt[:50]}...\n"
        )

    await interaction.response.send_message(f"```{''.join(output_lines)}```")


# ========================
# Pipeline Initialization
# ========================
def init_pipeline():
    """Initialize the diffusion pipeline with current configuration."""
    if not TORCH_AVAILABLE:
        print("ERROR: Cannot initialize pipeline - Torch is not installed")
        return

    if not MODEL:
        print("INFO: No model configured for initialization")
        return

    global txt2img_pipe, img2img_pipe, inpaint_pipe, in_progress
    in_progress = True

    try:
        print(f"Initializing pipeline with model: {MODEL}")
        dtype = torch.float16 if VRAM_USAGE != "mps" else torch.float32
        scheduler = None
        device_map = 'balanced' if VRAM_USAGE == "distributed" and torch.cuda.device_count() > 1 else None

        # Initialize Flux pipeline
        if "flux" in MODEL.lower():
            txt2img_pipe = _init_flux_pipeline(dtype, device_map)
        # Initialize SDXL pipeline
        elif "xl" in MODEL.lower():
            txt2img_pipe = _init_sdxl_pipeline(dtype)
        # Initialize SD 1.5 pipeline
        else:
            txt2img_pipe = _init_sd15_pipeline(dtype)

        # Apply scheduler
        scheduler = _configure_scheduler(txt2img_pipe)
        if scheduler:
            txt2img_pipe.scheduler = scheduler

        # Load LoRA if configured
        if LORA and exists(join(LORAS_FOLDER, LORA)):
            print(f"Loading LoRA: {LORA}")
            txt2img_pipe.load_lora_weights(join(LORAS_FOLDER, LORA))

        # Apply memory optimizations
        _apply_memory_optimizations(txt2img_pipe)

        # Precompute the tensors for converting latents to RGB
        _precompute_rgb_tensors()

        # Create derivative pipelines
        img2img_pipe = AutoPipelineForImage2Image.from_pipe(txt2img_pipe)
        inpaint_pipe = AutoPipelineForInpainting.from_pipe(txt2img_pipe)

        # Cleanup
        collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("Pipeline initialization complete")
    except Exception as e:
        print(f"ERROR: Failed to initialize pipeline: {e}")
    finally:
        in_progress = False

        # Process next item in queue if available
        if not request_queue.empty():
            _process_next_request()


def _init_flux_pipeline(dtype, device_map):
    """Initialize Flux pipeline with quantization."""
    bfl_repo = "black-forest-labs/FLUX.1-dev"

    if VRAM_USAGE == "mps":
        return FluxPipeline.from_pretrained(bfl_repo, dtype=dtype, token=HF_TOKEN)

    # Quantization config for transformer
    transformer_quant = DiffusersBitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )

    transformer = FluxTransformer2DModel.from_pretrained(
        bfl_repo,
        subfolder="transformer",
        quantization_config=transformer_quant,
        dtype=dtype,
        token=HF_TOKEN,
    )

    # Quantization config for text encoder
    text_quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )

    text_encoder = T5EncoderModel.from_pretrained(
        bfl_repo,
        subfolder="text_encoder_2",
        quantization_config=text_quant,
        dtype=dtype,
        device_map=device_map,
        token=HF_TOKEN,
    )

    tokenizer = CLIPTokenizer.from_pretrained(
        "openai/clip-vit-large-patch14",
        clean_up_tokenization_spaces=True,
        token=HF_TOKEN,
    )

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        bfl_repo,
        subfolder="scheduler",
        token=HF_TOKEN,
    )

    # Free memory before loading full pipeline
    collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return FluxPipeline.from_pretrained(
        bfl_repo,
        text_encoder_2=text_encoder,
        transformer=transformer,
        tokenizer=tokenizer,
        scheduler=scheduler,
        device_map=device_map,
        token=HF_TOKEN,
    )


def _init_sdxl_pipeline(dtype):
    """Initialize Stable Diffusion XL pipeline."""
    model_path = join(MODELS_FOLDER, MODEL)
    if not exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    return StableDiffusionXLPipeline.from_single_file(
        model_path,
        safety_checker=None,
        use_safetensors=True,
        add_watermarker=False
    )


def _init_sd15_pipeline(dtype):
    """Initialize Stable Diffusion 1.5 pipeline."""
    model_path = join(MODELS_FOLDER, MODEL)
    if not exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    return StableDiffusionPipeline.from_single_file(
        model_path,
        safety_checker=None,
        use_safetensors=True
    )


def _configure_scheduler(pipe):
    """Configure and return appropriate scheduler based on settings."""
    if SCHEDULER_NAME.startswith("dpm++ sde"):
        scheduler = DPMSolverSinglestepScheduler.from_config(pipe.scheduler.config)
        scheduler.config.lower_order_final = True
    elif SCHEDULER_NAME.startswith("euler a"):
        scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    else:
        return None

    # Apply Karras sigmas if specified (not for Flux)
    if SCHEDULER_NAME.endswith("karras") and "flux" not in MODEL.lower():
        scheduler.config.use_karras_sigmas = True

    return scheduler


def _apply_memory_optimizations(pipe):
    """Apply memory optimization techniques based on VRAM usage setting."""
    if VRAM_USAGE == "mps":
        pipe.to(device='mps')
        return

    # Always enable VAE optimizations for non-high VRAM settings
    if VRAM_USAGE != "high":
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()

    # Apply CPU offloading based on VRAM setting
    if VRAM_USAGE == "medium":
        pipe.enable_model_cpu_offload(gpu_id=DEVICE_NO)
    elif VRAM_USAGE == "low":
        pipe.enable_sequential_cpu_offload(gpu_id=DEVICE_NO)


def _precompute_rgb_tensors():
    """Precompute RGB conversion tensors for efficient latent-to-RGB conversion."""
    if "flux" in MODEL.lower():
        global flux_rgb_factors_tensor, flux_rgb_bias_tensor
        flux_rgb_factors_tensor = torch.tensor(
            FLUX_RGB_FACTORS, dtype=torch.float32
        ).transpose(0, 1)
        flux_rgb_bias_tensor = torch.tensor(
            FLUX_RGB_FACTORS_BIAS, dtype=torch.float32
        )
    elif "xl" in MODEL.lower():
        global sdxl_rgb_factors_tensor, sdxl_rgb_bias_tensor
        sdxl_rgb_factors_tensor = torch.tensor(
            SDXL_RGB_FACTORS, dtype=torch.float32
        ).transpose(0, 1)
        sdxl_rgb_bias_tensor = torch.tensor(
            SDXL_RGB_FACTORS_BIAS, dtype=torch.float32
        )
    else:
        global sd15_rgb_factors_tensor
        sd15_rgb_factors_tensor = torch.tensor(
            SD15_RGB_FACTORS, dtype=torch.float32
        ).transpose(0, 1)


# ========================
# Configuration Setters
# ========================
async def set_scheduler(interaction: Interaction, new_scheduler: Optional[str]):
    """Change the scheduler and reinitialize pipeline."""
    global SCHEDULER_NAME, txt2img_pipe, img2img_pipe, inpaint_pipe

    if not new_scheduler or new_scheduler not in POSSIBLE_SCHEDULERS:
        await interaction.response.send_message(
            f"Current scheduler: {SCHEDULER_NAME}\n\n"
            f"Available schedulers:\n" + "\n".join(f"- {s}" for s in POSSIBLE_SCHEDULERS)
        )
        return

    SCHEDULER_NAME = new_scheduler
    _clear_pipelines()
    await interaction.response.send_message(f"Scheduler changed to: {SCHEDULER_NAME}")


async def set_device(interaction: Interaction, new_device: Optional[int]):
    """Change the GPU device and reinitialize pipeline."""
    global DEVICE_NO, txt2img_pipe, img2img_pipe, inpaint_pipe

    if new_device is None:
        device_count = torch.cuda.device_count() if TORCH_AVAILABLE and torch.cuda.is_available() else 0
        await interaction.response.send_message(
            f"Current device: {DEVICE_NO}\n"
            f"Available CUDA devices: {device_count}"
        )
        return

    if not TORCH_AVAILABLE or new_device < 0 or new_device >= torch.cuda.device_count():
        await interaction.response.send_message(
            f"Invalid device number. Must be between 0 and {torch.cuda.device_count() - 1}"
        )
        return

    DEVICE_NO = new_device
    _clear_pipelines()
    await interaction.response.send_message(f"Device changed to: {DEVICE_NO}")


async def set_model(interaction: Interaction, new_model: Optional[str]):
    """Change the model and reinitialize pipeline."""
    global MODEL, txt2img_pipe, img2img_pipe, inpaint_pipe

    # Get available models
    available_models = ["flux"]  # Special case for Flux from HF
    if exists(MODELS_FOLDER):
        available_models.extend([
            f for f in listdir(MODELS_FOLDER)
            if f.endswith('.safetensors')
        ])

    if not new_model or new_model not in available_models:
        await interaction.response.send_message(
            f"Current model: {MODEL}\n\n"
            f"Available models:\n" + "\n".join(f"- {m}" for m in available_models)
        )
        return

    MODEL = new_model
    _clear_pipelines()
    await interaction.response.send_message(f"Model changed to: {MODEL}")


async def set_lora(interaction: Interaction, new_lora: Optional[str]):
    """Change the LoRA and reinitialize pipeline."""
    global LORA, txt2img_pipe, img2img_pipe, inpaint_pipe

    # Get available LoRAs
    available_loras = []
    if exists(LORAS_FOLDER):
        available_loras = [
            f for f in listdir(LORAS_FOLDER)
            if f.endswith('.safetensors')
        ]

    if not new_lora or new_lora not in available_loras:
        await interaction.response.send_message(
            f"Current LoRA: {LORA or 'None'}\n\n"
            f"Available LoRAs:\n" + (
                "\n".join(f"- {l}" for l in available_loras) if available_loras
                else "No LoRAs found in folder"
            )
        )
        return

    LORA = new_lora
    _clear_pipelines()
    await interaction.response.send_message(f"LoRA changed to: {LORA}")


def _clear_pipelines():
    """Clear all pipelines and free memory."""
    global txt2img_pipe, img2img_pipe, inpaint_pipe
    txt2img_pipe = img2img_pipe = inpaint_pipe = None
    collect()
    if TORCH_AVAILABLE and torch.cuda.is_available():
        torch.cuda.empty_cache()


# ========================
# Image Generation
# ========================
def latent_to_rgb(latent: torch.Tensor) -> Image:
    """
    Convert latent space representation to RGB image for preview.

    Different models use different latent space representations:
    - SD 1.5: 4-channel latents [4, H, W]
    - SDXL: 4-channel latents [4, H, W] with bias
    - Flux: 16-channel latents packed as [H*W, 64] that need reshaping

    Args:
        latent: Latent tensor from diffusion model

    Returns:
        PIL Image or None if conversion fails
    """
    if not TORCH_AVAILABLE:
        return None

    # Determine which tensors to use based on model type
    rgb_factors = None
    rgb_bias = None

    if "flux" in MODEL.lower():
        if flux_rgb_factors_tensor is None:
            return None
        rgb_factors = flux_rgb_factors_tensor
        rgb_bias = flux_rgb_bias_tensor

        # Reshape Flux latents from [H*W, 64] to [64, H, W]
        # The 64 channels represent 16 logical channels packed 4x
        num_pixels = latent.size(0)
        num_channels = latent.size(1)

        # Calculate height and width (assuming square latent)
        h = int(num_pixels ** 0.5)
        #if h * h != num_pixels:
        #    print(f"Warning: Non-square latent detected: {num_pixels} pixels")
        #    return None

        w = h

        # Reshape: [H*W, 64] -> [H, W, 64] -> [64, H, W]
        latent = latent.view(h, w, num_channels).permute(2, 0, 1)

    elif "xl" in MODEL.lower():
        if sdxl_rgb_factors_tensor is None:
            return None
        rgb_factors = sdxl_rgb_factors_tensor
        rgb_bias = sdxl_rgb_bias_tensor
    else:
        if sd15_rgb_factors_tensor is None:
            return None
        rgb_factors = sd15_rgb_factors_tensor
        rgb_bias = None

    if rgb_factors is None:
        return None

    # Move tensors to same device and dtype as latent
    rgb_factors = rgb_factors.to(dtype=latent.dtype, device=latent.device)
    if rgb_bias is not None:
        rgb_bias = rgb_bias.to(dtype=latent.dtype, device=latent.device)

    # Apply linear transformation: RGB = latent @ factors + bias
    # Move channel dimension to last for linear operation
    # Current shape: [C, H, W] -> [H, W, C]
    latent_moved = latent.movedim(0, -1)

    # Apply linear transformation
    latent_image = torch.nn.functional.linear(
        latent_moved,
        rgb_factors,
        bias=rgb_bias
    )

    # Convert from [-1, 1] range to [0, 255] uint8
    # The latent space is typically centered around 0
    latent_ubyte = (
        ((latent_image + 1.0) / 2.0)  # Map [-1, 1] to [0, 1]
        .clamp(0, 1)                   # Ensure valid range
        .mul(255)                       # Scale to [0, 255]
    ).to(device="cpu", dtype=torch.uint8)

    # Convert to PIL Image
    # Shape is now [H, W, 3]
    return fromarray(latent_ubyte.numpy())


def progress_check(interaction: Interaction, total_steps: int, seed: int,
                   pipe, step: int, timestep: 'torch.Tensor', callback_kwargs: dict):
    """Callback for displaying generation progress with preview images."""
    # Only update every 10% of progress
    if step % max(1, total_steps // 10) != 0:
        return callback_kwargs

    img = latent_to_rgb(callback_kwargs["latents"][0])
    progress_pct = round(step / total_steps * 100.0)

    if img is None:
        queue_edit(interaction, content=f"Seed: {seed} | Progress: {progress_pct}%")
    else:
        # Upscale preview 4x for better visibility
        upscaled = img.resize(
            (img.size[0] * 4, img.size[1] * 4),
            Resampling.LANCZOS
        )
        queue_edit(
            interaction,
            content=f"Seed: {seed} | Progress: {progress_pct}%",
            attachments=[pildiscordfile(upscaled, "preview.png")]
        )

    return callback_kwargs


def _process_next_request():
    """Process the next request in the queue."""
    if not request_queue.empty():
        req = request_queue.get()
        diffusion(
            req.interaction, req.prompt, req.image_param, req.mask_image_param,
            req.url, req.mask_url, req.steps, req.height, req.width,
            req.resize, req.cfg, req.strength, req.seed
        )


def diffusion(interaction: Interaction, prompt: str, image_param: Optional[Attachment] = None,
                    mask_image_param: Optional[Attachment] = None, url: Optional[str] = None,
                    mask_url: Optional[str] = None, steps: int = 50, height: int = 512,
                    width: int = 512, resize: float = 1.0, cfg: float = 7.0,
                    strength: float = 0.8, seed: Optional[int] = None):
    """Main image generation function."""
    if not TORCH_AVAILABLE:
        queue_message(interaction, "Error: PyTorch is not installed. Cannot generate images.")
        return

    global in_progress

    # Queue request if busy
    if in_progress:
        queue_message(interaction, "Request queued. You'll be notified when generation starts.")
        req = GenerationRequest(
            interaction, prompt, image_param, mask_image_param, url, mask_url,
            steps, height, width, resize, cfg, strength, seed
        )
        request_queue.put(req)
        return

    # Initialize pipeline if needed
    if not txt2img_pipe:
        queue_message(interaction, "Initializing AI pipeline... This may take a few minutes.")
        init_pipeline()

    in_progress = True

    # Load input images
    image = None
    if image_param:
        image = load_image(image_param.url)
    elif url:
        image = load_image(url)

    mask_image = None
    if mask_image_param:
        mask_image = load_image(mask_image_param.url)
    elif mask_url:
        mask_image = load_image(mask_url)

    # Setup generator
    device = "cuda" if VRAM_USAGE != "mps" else "mps"
    generator = torch.Generator(device)

    if seed:
        generator = generator.manual_seed(seed)
    else:
        seed = generator.seed()

    # Adjust dimensions if image is provided
    if image and not (width == 512 and height == 512):
        width, height = image.size
        width = int(width * resize)
        height = int(height * resize)

    # Ensure dimensions are multiples of 8
    width = (width // 8) * 8
    height = (height // 8) * 8

    # Log generation parameters
    #print(f"Generating: {prompt[:50]}... | steps={steps} size={width}x{height} seed={seed}")
    queue_message(
        interaction,
        f"**Generating image...**\n"
        f"Steps: {steps} | Size: {width}x{height} | CFG: {cfg} | Seed: {seed}\n"
        f"Scheduler: {SCHEDULER_NAME} | Strength: {strength}\n"
        f"Prompt: {prompt}"
    )

    # Clean up memory before generation
    collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Generate image based on mode
    if mask_image:
        result = inpaint_pipe(
            image=image,
            mask_image=mask_image,
            height=height,
            width=width,
            strength=strength,
            prompt=prompt,
            num_inference_steps=steps,
            guidance_scale=cfg,
            generator=generator,
            callback_on_step_end=lambda *args: progress_check(
                interaction, int(steps * strength), seed, *args
            ),
            callback_on_step_end_tensor_inputs=["latents"]
        ).images[0]
    elif image:
        result = img2img_pipe(
            image=image,
            height=height,
            width=width,
            strength=strength,
            prompt=prompt,
            num_inference_steps=steps,
            guidance_scale=cfg,
            generator=generator,
            callback_on_step_end=lambda *args: progress_check(
                interaction, int(steps * strength), seed, *args
            ),
            callback_on_step_end_tensor_inputs=["latents"]
        ).images[0]
    else:
        result = txt2img_pipe(
            prompt=prompt,
            num_inference_steps=steps,
            height=height,
            width=width,
            guidance_scale=cfg,
            generator=generator,
            callback_on_step_end=lambda *args: progress_check(
                interaction, steps, seed, *args
            ),
            callback_on_step_end_tensor_inputs=["latents"]
        ).images[0]

    # Send final result
    queue_file(interaction, pildiscordfile(result, f"generated_{seed}.png"))
    #print(f"Generation complete: seed={seed}")

    in_progress = False

    # Process next request if queue has items
    if not request_queue.empty():
        queue_command(_process_next_request)


# ========================
# Module Exports
# ========================
__all__ = [
    'diffusion',
    'get_qsize',
    'init_pipeline',
    'set_device',
    'set_lora',
    'set_model',
    'set_scheduler',
]
