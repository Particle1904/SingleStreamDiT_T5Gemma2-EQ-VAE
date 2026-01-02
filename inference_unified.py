import torch
import os
import argparse
from model import SingleStreamDiTV2
from diffusers import AutoencoderKL
from transformers import AutoTokenizer, AutoModel
from PIL import Image
import numpy as np

FILENAME = "checkpoint3_epoch_1200_ema"
CHECKPOINT_PATH = f"./checkpoints/{FILENAME}.pt"
DEVICE = "cuda"
DTYPE = torch.bfloat16

# Options: "text" (Generate from prompt) OR "file" (Generate from cached .pt file)
INPUT_MODE = "file" 

# Options: "rk4" (High Quality, Slower) OR "euler" (Fast, Lower Quality)
SAMPLER = "rk4" 
NUM_STEPS = 50
GUIDANCE_SCALE = 1.5  # 1.0 = No CFG, 4.0-8.0 = Standard

# 4. TEXT MODE SETTINGS (Ignored if INPUT_MODE is "file")
PROMPT = "An extreme macro shot of a single, vibrant pink carnation, featuring densely layered, ruffled petals with finely serrated, delicate, translucent edges, displaying bold red veining against a pale white base, the center is a cluster of yellow anthers with visible pollen, set against a dark, blurred background under soft studio lighting with a very shallow depth of field."
NEGATIVE_PROMPT = ""
HEIGHT = 512
WIDTH = 512
TEXT_MODEL_ID = "google/t5gemma-2-1b-1b"

# 5. FILE MODE SETTINGS (Ignored if INPUT_MODE is "text")
TARGET_FILE = "./cached_data/39.pt"

# 6. MODEL CONSTANTS
TEXT_DIM = 1152
VAE_SCALE_FACTOR = 0.13025
LATENT_OFFSET = 0.3077
LATENT_STD_SCALE = 1.1908 # Reverses the normalization done during training

# --- HELPER FUNCTIONS ---

def get_velocity(model, x, t, text_embeds, h_latent, w_latent, cfg):
    """
    Calculates the velocity field v(x, t) considering Classifier-Free Guidance.
    """
    x_in = torch.cat([x, x], dim=0)
    t_in = torch.cat([t, t], dim=0)
    v_out = model(x_in, t_in, text_embeds, h_latent, w_latent)
    v_uncond, v_cond = v_out.chunk(2, dim=0)
    v_final = v_uncond + cfg * (v_cond - v_uncond)
    return v_final

def decode_latents(latents, vae):
    """
    Applies the specific denormalization logic and decodes via VAE.
    """
    print("Decoding latents...")
    
    latents = latents - LATENT_OFFSET
    latents = latents * LATENT_STD_SCALE
    latents = latents / VAE_SCALE_FACTOR
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=False):
            image = vae.decode(latents.float()).sample

    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()
    image = (image * 255).round().astype("uint8")
    return image[0]

@torch.no_grad()
def main():
    print(f"--- STARTING INFERENCE ({INPUT_MODE.upper()} | {SAMPLER.upper()}) ---")
    
    print(f"Loading DiT from {os.path.basename(CHECKPOINT_PATH)}...")
    model = SingleStreamDiTV2(text_embed_dim=TEXT_DIM).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH))
    model.eval().to(dtype=DTYPE)

    print("Loading VAE...")
    vae = AutoencoderKL.from_pretrained("KBlueLeaf/EQ-SDXL-VAE").to(DEVICE)
    
    combined_text_embeds = None
    latent_h, latent_w = 0, 0
    
    if INPUT_MODE == "file":
        print(f"Loading data from {TARGET_FILE}...")
        data = torch.load(TARGET_FILE)
        
        h, w = data["height"], data["width"]
        latent_h, latent_w = h // 8, w // 8
        print(f"Target Resolution: {w}x{h}")
        
        cond_embeds = data["text_embeds"].unsqueeze(0).to(DEVICE, dtype=DTYPE)
        uncond_embeds = torch.zeros_like(cond_embeds)
        combined_text_embeds = torch.cat([uncond_embeds, cond_embeds], dim=0)
        
    elif INPUT_MODE == "text":
        print(f"Processing prompt: '{PROMPT}'")
        latent_h, latent_w = HEIGHT // 8, WIDTH // 8
        
        print("Loading T5-Gemma...")
        tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_ID)
        full_text_model = AutoModel.from_pretrained(TEXT_MODEL_ID, trust_remote_code=True)
        text_encoder = full_text_model.encoder if hasattr(full_text_model, "encoder") else full_text_model
        text_encoder.to(DEVICE).eval()
        
        inputs = tokenizer(PROMPT, max_length=256, padding="max_length", truncation=True, return_tensors="pt").to(DEVICE)
        out = text_encoder(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
        cond_embeds = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        
        uncond_inputs = tokenizer(NEGATIVE_PROMPT, max_length=256, padding="max_length", truncation=True, return_tensors="pt").to(DEVICE)
        uncond_out = text_encoder(input_ids=uncond_inputs.input_ids, attention_mask=uncond_inputs.attention_mask)
        uncond_embeds = uncond_out.last_hidden_state if hasattr(uncond_out, "last_hidden_state") else uncond_out[0]
        
        combined_text_embeds = torch.cat([uncond_embeds, cond_embeds], dim=0).to(dtype=DTYPE)
        
        del text_encoder
        del full_text_model
        torch.cuda.empty_cache()

    print("Initializing Latents...")
    x = torch.randn(1, 4, latent_h, latent_w, device=DEVICE, dtype=DTYPE)
    
    print(f"Sampling with {SAMPLER.upper()} for {NUM_STEPS} steps...")
    dt = 1.0 / NUM_STEPS
    
    with torch.autocast(device_type=DEVICE, dtype=DTYPE):
        for i in range(NUM_STEPS):
            t_val = i / NUM_STEPS
            t = torch.tensor([t_val], device=DEVICE, dtype=DTYPE)
            
            if SAMPLER == "euler":
                v = get_velocity(model, x, t, combined_text_embeds, latent_h, latent_w, GUIDANCE_SCALE)
                x = x + v * dt
                
            elif SAMPLER == "rk4":
                dt_step = dt
                
                k1 = get_velocity(model, x, t, combined_text_embeds, latent_h, latent_w, GUIDANCE_SCALE)
                
                k2 = get_velocity(
                    model, 
                    x + 0.5 * dt_step * k1, 
                    t + 0.5 * dt_step, 
                    combined_text_embeds, latent_h, latent_w, GUIDANCE_SCALE
                )
                
                k3 = get_velocity(
                    model, 
                    x + 0.5 * dt_step * k2, 
                    t + 0.5 * dt_step, 
                    combined_text_embeds, latent_h, latent_w, GUIDANCE_SCALE
                )
                
                k4 = get_velocity(
                    model, 
                    x + dt_step * k3, 
                    t + dt_step, 
                    combined_text_embeds, latent_h, latent_w, GUIDANCE_SCALE
                )
                
                x = x + (dt_step / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    final_image_array = decode_latents(x, vae)
    
    output_filename = f"result_{INPUT_MODE}_{SAMPLER}_cfg{GUIDANCE_SCALE}.png"
    Image.fromarray(final_image_array).save(output_filename)
    print(f"--- SUCCESS ---")
    print(f"Image saved to: {output_filename}")

if __name__ == "__main__":
    main()