import torch
import torch.nn.functional as F
import bitsandbytes as bnb
from tqdm import tqdm
from model import SingleStreamDiTV2
from diffusers import AutoencoderKL
from PIL import Image
import csv  # Added for logging

TARGET_FILE = "./cached_data/39.pt" 
STEPS = 1000 
LEARNING_RATE = 1e-4
TEXT_DIM = 1152 
DEVICE = "cuda"

def sanity():
    data = torch.load(TARGET_FILE)
    latents = data["latents"].unsqueeze(0).cuda().to(torch.bfloat16)
    text = data["text_embeds"].unsqueeze(0).cuda().to(torch.bfloat16)
    h, w = data["height"], data["width"]
    
    print(f"Sanity Check Target: {w}x{h}")

    model = SingleStreamDiTV2(text_embed_dim=TEXT_DIM).cuda().to(torch.bfloat16)
    model.initialize_weights() 
    
    optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=LEARNING_RATE)
    vae = AutoencoderKL.from_pretrained("KBlueLeaf/EQ-SDXL-VAE").to("cuda")

    pbar = tqdm(range(STEPS))
    for i in pbar:
        x_1 = latents
        x_0 = torch.randn_like(x_1)
        t = torch.rand(1, device="cuda").to(torch.bfloat16)
        x_t = (1 - t.view(-1,1,1,1)) * x_0 + t.view(-1,1,1,1) * x_1
        
        pred = model(x_t, t, text, h, w)
        loss = F.mse_loss(pred, x_1 - x_0)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        pbar.set_description(f"Loss: {loss.item():.4f}")

    print("Generating check image...")
    with torch.no_grad():
        x = torch.randn(1, 4, h//8, w//8, device=DEVICE, dtype=torch.bfloat16)
        dt = 1.0 / 50
        for step in range(50):
            t_val = torch.tensor([step/50], device=DEVICE, dtype=torch.bfloat16)
            x_in = torch.cat([x, x])
            t_in = torch.cat([t_val, t_val])
            combined_text = torch.cat([torch.zeros_like(text), text])
            
            v = model(x_in, t_in, combined_text, h//8, w//8)
            v_uncond, v_cond = v.chunk(2)
            
            v_final = v_uncond + 1.15 * (v_cond - v_uncond)
            x = x + v_final * dt

        lat = x / 0.13025
        with torch.cuda.amp.autocast(enabled=False):
            img = vae.decode(lat.float()).sample
        
        img = (img / 2 + 0.5).clamp(0, 1).cpu().permute(0, 2, 3, 1).float().numpy()
        img = (img * 255).round().astype("uint8")
        Image.fromarray(img[0]).save("sanity_result.png")
        print("Saved sanity_result.png")

if __name__ == "__main__":
    sanity()