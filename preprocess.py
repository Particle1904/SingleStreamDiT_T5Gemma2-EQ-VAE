import os
import torch
import random
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as TF
from transformers import AutoTokenizer, AutoModel
from diffusers import AutoencoderKL
from tqdm import tqdm

DATASET_DIR = "./dataset"
OUTPUT_DIR = "./cached_data"
TARGET_AREA = 512 * 512
TEXT_MODEL_ID = "google/t5gemma-2-1b-1b" 
VAE_MODEL_ID = "KBlueLeaf/EQ-SDXL-VAE"
BUCKETS = [
    (512, 512),  # Square
    (384, 640),  # Portrait (~2:3)
    (640, 384),  # Landscape (~3:2)
]

def get_best_bucket(w, h):
    target_aspect = w / h
    best_bucket = min(BUCKETS, key=lambda b: abs((b[0]/b[1]) - target_aspect))
    return best_bucket

def setup_models():
    vae = AutoencoderKL.from_pretrained(VAE_MODEL_ID).to("cuda").eval()
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_ID)
    full_model = AutoModel.from_pretrained(TEXT_MODEL_ID, trust_remote_code=True)
    text_model = full_model.encoder if hasattr(full_model, "encoder") else full_model
    text_model.to("cuda").eval()
    return vae, tokenizer, text_model

def process():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    vae, tokenizer, text_model = setup_models()
    
    files = [f for f in os.listdir(DATASET_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    for filename in tqdm(files):
        try:
            img_path = os.path.join(DATASET_DIR, filename)
            txt_path = os.path.join(DATASET_DIR, os.path.splitext(filename)[0] + ".txt")
            if not os.path.exists(txt_path): continue

            image = Image.open(img_path).convert("RGB")
            flipped = False
            if random.random() > 0.5:
                image = TF.hflip(image)
                flipped = True

            w, h = image.size
            bw, bh = get_best_bucket(w, h)
            
            img = TF.resize(image, (bh, bw), interpolation=transforms.InterpolationMode.LANCZOS)
            img_tensor = TF.to_tensor(img).unsqueeze(0).to("cuda")
            img_tensor = TF.normalize(img_tensor, [0.5], [0.5])

            with torch.no_grad():
                latents = vae.encode(img_tensor).latent_dist.sample() * 0.13025

            with open(txt_path, 'r', encoding='utf-8') as f:
                prompt = f.read().strip()
            
            inputs = tokenizer(prompt, max_length=256, padding="max_length", truncation=True, return_tensors="pt").to("cuda")
            with torch.no_grad():
                outputs = text_model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
                text_embeds = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]

            save_data = {
                "latents": latents.squeeze(0).cpu(),
                "text_embeds": text_embeds.squeeze(0).cpu(),
                "width": bw,
                "height": bh,
                "flipped": flipped
            }
            torch.save(save_data, os.path.join(OUTPUT_DIR, os.path.splitext(filename)[0] + ".pt"))

        except Exception as e:
            print(f"Error {filename}: {e}")

if __name__ == "__main__":
    process()