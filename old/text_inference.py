import torch
from model import SingleStreamDiTV2
from diffusers import AutoencoderKL
from transformers import AutoTokenizer, AutoModel
from PIL import Image
import numpy as np

# --- CONFIGURATION ---
PROMPT = "A macro shot of a yellow dandelion, having radiating, delicate, serrated petals, the center is a dense, pollen-covered disk with visible stamens, the stem is slender and reddish, set against a blurred, earthy background under natural daylight with shallow depth of field."
NEGATIVE_PROMPT = ""

HEIGHT = 448
WIDTH = 512

FILENAME = "checkpoint3_epoch_900"
CHECKPOINT = f"E:/AI_Training/t5gemma_poc/checkpoints/{FILENAME}.pt"

# Model Config
TEXT_MODEL_ID = "google/t5gemma-2-1b-1b" 
TEXT_DIM = 1152
DEVICE = "cuda"
GUIDANCE_SCALE = 1.5 # Try 1.0, 4.0, 8.0

@torch.no_grad()
def generate():
    print(f"Initializing for prompt: '{PROMPT}'...")

    print("Loading DiT...")
    model = SingleStreamDiTV2(text_embed_dim=TEXT_DIM).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT))
    model.eval().to(dtype=torch.bfloat16)

    print("Loading VAE...")
    vae = AutoencoderKL.from_pretrained("KBlueLeaf/EQ-SDXL-VAE").to(DEVICE)

    print("Loading T5Gemma Encoder...")
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_ID)
    full_model = AutoModel.from_pretrained(TEXT_MODEL_ID, trust_remote_code=True)
    
    text_model = full_model.encoder if hasattr(full_model, "encoder") else full_model
    text_model.to(DEVICE).eval()

    print("Encoding prompt...")
    inputs = tokenizer(
        PROMPT, 
        max_length=256, 
        padding="max_length", 
        truncation=True, 
        return_tensors="pt"
    ).to(DEVICE)
    
    with torch.no_grad():
        outputs = text_model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
        if hasattr(outputs, "last_hidden_state"):
            text_embeds = outputs.last_hidden_state
        else:
            text_embeds = outputs[0]
            
    uncond_inputs = tokenizer(
        NEGATIVE_PROMPT, 
        max_length=256, 
        padding="max_length", 
        truncation=True, 
        return_tensors="pt"
    ).to(DEVICE)
    
    with torch.no_grad():
        uncond_outputs = text_model(input_ids=uncond_inputs.input_ids, attention_mask=uncond_inputs.attention_mask)
        if hasattr(uncond_outputs, "last_hidden_state"):
            uncond_embeds = uncond_outputs.last_hidden_state
        else:
            uncond_embeds = uncond_outputs[0]

    text_embeds = text_embeds.to(dtype=torch.bfloat16)
    uncond_embeds = uncond_embeds.to(dtype=torch.bfloat16)
    
    combined_text = torch.cat([uncond_embeds, text_embeds], dim=0)

    print(f"Sampling with CFG {GUIDANCE_SCALE}...")
    
    latent_h = HEIGHT // 8
    latent_w = WIDTH // 8
    x = torch.randn(1, 4, latent_h, latent_w, device=DEVICE, dtype=torch.bfloat16)
    
    num_steps = 50
    dt = 1.0 / num_steps
    
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for i in range(num_steps):
            t_val = i / num_steps
            t = torch.tensor([t_val], device=DEVICE, dtype=torch.bfloat16)
            
            x_in = torch.cat([x, x], dim=0)
            t_in = torch.cat([t, t], dim=0)
            
            v_out = model(x_in, t_in, combined_text, latent_h, latent_w)
            
            v_uncond, v_cond = v_out.chunk(2, dim=0)
            
            v = v_uncond + GUIDANCE_SCALE * (v_cond - v_uncond)
            
            x = x + v * dt

    print("Decoding...")
    latents = (x - 0.3077) / 0.13025
    
    with torch.cuda.amp.autocast(enabled=False):
        image = vae.decode(latents.float()).sample

    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()
    image = (image * 255).round().astype("uint8")
    
    save_name = f"gen_cfg{GUIDANCE_SCALE}.png"
    Image.fromarray(image[0]).save(save_name)
    print(f"Saved to {save_name}")

if __name__ == "__main__":
    generate()