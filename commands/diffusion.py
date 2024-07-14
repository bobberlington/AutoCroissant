import numpy as np
import torch
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline

# from commands.frankenstein import url_to_image, cv2discordfile

model = "./models/RPG-v4.safetensors"
negative_tags = "nsfw, lowres, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, blurry"

pipe = StableDiffusionPipeline.from_single_file(model, use_safetensors=True)
print(torch.cuda.is_available())
if torch.cuda.is_available():
    pipe.to("cuda")

prompt = "a photo of an astronaut riding a horse on mars"
pipe.enable_attention_slicing()
image = pipe(prompt=prompt, negative_prompt=negative_tags, num_inference_steps=5, height=256, width=256).images[0]
image.save("test.png")