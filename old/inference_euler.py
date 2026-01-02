import torch
from model import SingleStreamDiTV2
from diffusers import AutoencoderKL
from PIL import Image
import numpy as np

FILENAME = "checkpoint3_epoch_900"
CHECKPOINT = f"E:/AI_Training/t5gemma_poc/checkpoints/{FILENAME}.pt"
TARGET_FILE = "E:/AI_Training/t5gemma_poc/cached_data/39.pt"
TEXT_DIM = 1152
DEVICE = "cuda"

GUIDANCE_SCALE = 8.0

@torch.no_grad()
def generate():
    print("Loading V2 Model...")
    model = SingleStreamDiTV2(text_embed_dim=TEXT_DIM).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT))
    model.eval().to(dtype=torch.bfloat16)

    vae = AutoencoderKL.from_pretrained("KBlueLeaf/EQ-SDXL-VAE").to(DEVICE)

    data = torch.load(TARGET_FILE)
    text_embeds = data["text_embeds"].unsqueeze(0).to(DEVICE, dtype=torch.bfloat16)
    
    h, w = data["height"], data["width"]
    print(f"Generating image in bucket: {w}x{h}")

    uncond_embeds = torch.zeros_like(text_embeds)
    
    combined_text = torch.cat([uncond_embeds, text_embeds], dim=0)

    print("Sampling with CFG...")
    x = torch.randn(1, 4, h//8, w//8, device=DEVICE, dtype=torch.bfloat16)
    
    num_steps = 50
    dt = 1.0 / num_steps
    
    for i in range(num_steps):
        t_val = i / num_steps
        t = torch.tensor([t_val], device=DEVICE, dtype=torch.bfloat16)
        
        x_in = torch.cat([x, x], dim=0)
        t_in = torch.cat([t, t], dim=0)
        
        v_out = model(x_in, t_in, combined_text, h//8, w//8)
        
        v_uncond, v_cond = v_out.chunk(2, dim=0)
        
        v = v_uncond + GUIDANCE_SCALE * (v_cond - v_uncond)
        
        x = x + v * dt

    print("Decoding...")
    latents = x / 0.13025
    with torch.cuda.amp.autocast(enabled=False):
        image = vae.decode(latents.float()).sample

    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()
    image = (image * 255).round().astype("uint8")
    
    output_path = "v2_test_result.png"
    Image.fromarray(image[0]).save(output_path)
    print(f"Success! Saved to {output_path}")

if __name__ == "__main__":
    generate()