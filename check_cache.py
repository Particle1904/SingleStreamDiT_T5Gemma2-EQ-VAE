import torch
from diffusers import AutoencoderKL
from PIL import Image
import numpy as np

TARGET_FILE = "./cached_data/39.pt"
VAE_ID = "KBlueLeaf/EQ-SDXL-VAE"

def check():
    print(f"Checking {TARGET_FILE}...")
    
    data = torch.load(TARGET_FILE)
    latents = data["latents"].unsqueeze(0).to("cuda")

    print("Loading VAE...")
    vae = AutoencoderKL.from_pretrained(VAE_ID).to("cuda")
    
    print("Decoding...")
    latents = latents / 0.13025
    
    with torch.no_grad():
        image = vae.decode(latents).sample
        
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()
    image = (image * 255).round().astype("uint8")
    
    Image.fromarray(image[0]).save("cache_verification.png")
    print("Saved cache_verification.png. Check this image!")

if __name__ == "__main__":
    check()