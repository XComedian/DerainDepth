import torch

from diffusers.models import AutoencoderKL
from diffusers import StableDiffusionPipeline

model = "CompVis/stable-diffusion-v1-4"
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
# pipe = StableDiffusionPipeline.from_pretrained(model, vae=vae)

input = torch.rand(1, 3, 256, 256)
output = vae.encode(input).latent_dist.sample()
print(output.shape)