from io import BytesIO

import streamlit as st
import torch

from diffusers import DiffusionPipeline

from PIL import Image


def get_image_as_bytes(image: Image) -> bytes:
    buf = BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


prompt = st.text_input("Text prompt for the image generator")
filename = st.text_input("Filename (without extension")

base = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
    device_map="auto",
)
base.unet.to(memory_format=torch.channels_last)
base.to("cuda:1")

refiner = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    #text_encoder_2=base.text_encoder_2,
    #vae=base.vae,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
    device_map="auto",
)
refiner.to("cuda:2")

with torch.no_grad():
    image = base(
        prompt=prompt,
        num_inference_steps=40,
        denoising_end=0.8,
        output_type="latent",
    ).images[0]

    image = refiner(
        prompt=prompt,
        num_inference_steps=40,
        denoising_start=0.8,
        image=image,
    ).images[0]

result_bytes = get_image_as_bytes(image)

btn = st.download_button(
    label="Download Image",
    data=result_bytes,
    file_name=f"{filename}.png",
    mime="image/png",
)
