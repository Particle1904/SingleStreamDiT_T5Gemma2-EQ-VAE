import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import bitsandbytes as bnb
from tqdm import tqdm
from model import SingleStreamDiTV2
from diffusers import AutoencoderKL
from PIL import Image
import random
import glob
import csv
from torch.optim.lr_scheduler import CosineAnnealingLR

# --- CONFIGURATION ---
# EXPERIMENT SETTINGS
PROJECT_NAME = "flowers_micro" # Used for log filenames
RESUME_FROM = "./checkpoints/checkpoint1_epoch_300.pt"  # Set to "checkpoints/v3/epoch_50.pt" to resume. Set to None for fresh start.
START_EPOCH = 301     # If resuming, set this manually (e.g. 50). If None, 0.

# TRAINING HYPERPARAMETERS
BATCH_SIZE = 8
LEARNING_RATE = 1e-4  # 1e-4 for fresh/aggressive, 5e-5 for fine-tuning
EPOCHS = 600
SAVE_EVERY = 10       # Save checkpoint every N epochs
VALIDATE_EVERY = 5    # Generate sample every N epochs
SCHEDULER_T_MAX = 300 # Remaining epochs
TEXT_DROPOUT = 0.25   # % of dropout for text

# PATHS
CACHE_DIR = "./cached_data" # Point to your 200 images
CHECKPOINT_DIR = f"./checkpoints/{PROJECT_NAME}"
SAMPLES_DIR = f"./samples/{PROJECT_NAME}"
LOG_FILE = f"./logs/{PROJECT_NAME}_log.csv"
TARGET_FILE = "./cached_data/39.pt"

# MODEL SETTINGS
TEXT_DIM = 1152
DEVICE = "cuda"

# VALIDATION SETTINGS
VAL_CFG = 1.0
VAL_STEPS = 30

def setup_dirs():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(SAMPLES_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

class CSVLogger:
    def __init__(self, filepath, resume=False):
        self.filepath = filepath
        self.resume = resume
        
        if not os.path.exists(filepath) or not resume:
            with open(filepath, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Epoch", "Global_Step", "Loss", "LR"])
            print(f"Created new log file: {filepath}")
        else:
            print(f"Resuming log file: {filepath}")

    def log(self, epoch, step, loss, lr):
        with open(self.filepath, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, step, loss, lr])

class BucketDataset(Dataset):
    def __init__(self, dir_path):
        self.files = glob.glob(os.path.join(dir_path, "*.pt"))
        if len(self.files) == 0:
            raise ValueError(f"No .pt files found in {dir_path}")
            
        print(f"Found {len(self.files)} files.")
        
        self.buckets = {}
        for f in self.files:
            try:
                d = torch.load(f, map_location="cpu")
                key = (d["height"], d["width"])
                if key not in self.buckets: self.buckets[key] = []
                self.buckets[key].append(f)
            except Exception as e:
                print(f"Error loading {f}: {e}")
        
        self.batches = []
        for key, files in self.buckets.items():
            random.shuffle(files)
            for i in range(0, len(files), BATCH_SIZE):
                batch = files[i:i + BATCH_SIZE]
                if len(batch) == BATCH_SIZE:
                    self.batches.append(batch)
        random.shuffle(self.batches)

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        batch_files = self.batches[idx]
        batch_data = [torch.load(f) for f in batch_files]
        
        latents = torch.stack([d["latents"] for d in batch_data])
        text = torch.stack([d["text_embeds"] for d in batch_data])
        
        if random.random() < 0.5:
            latents = torch.flip(latents, dims=[-1])

        if random.random() < TEXT_DROPOUT:
            text = torch.zeros_like(text)
            
        return {
            "latents": latents,
            "text_embeds": text,
            "height": batch_data[0]["height"],
            "width": batch_data[0]["width"]
        }

@torch.no_grad()
def validate(model, vae, epoch, cache_dir):
    model.eval()
    
    if os.path.exists(TARGET_FILE):
        val_file = TARGET_FILE
    else:
        test_files = glob.glob(os.path.join(cache_dir, "*.pt"))
        val_file = random.choice(test_files)
    
    data = torch.load(val_file)
    h, w = data["height"], data["width"]
    text_embeds = data["text_embeds"].unsqueeze(0).to(DEVICE, dtype=torch.bfloat16)
    
    uncond_embeds = torch.zeros_like(text_embeds)
    combined_text = torch.cat([uncond_embeds, text_embeds], dim=0)

    x = torch.randn(1, 4, h//8, w//8, device=DEVICE, dtype=torch.bfloat16)
    
    dt = 1.0 / VAL_STEPS
    
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for i in range(VAL_STEPS):
            t = torch.tensor([i/VAL_STEPS], device=DEVICE, dtype=torch.bfloat16)
            x_in = torch.cat([x, x], dim=0)
            t_in = torch.cat([t, t], dim=0)
            
            v_out = model(x_in, t_in, combined_text, h//8, w//8)
            v_uncond, v_cond = v_out.chunk(2, dim=0)
            v = v_uncond + VAL_CFG * (v_cond - v_uncond)
            x = x + v * dt

    latents = x / 0.13025
    with torch.cuda.amp.autocast(enabled=False):
        img = vae.decode(latents.float()).sample

    img = (img / 2 + 0.5).clamp(0, 1).cpu().permute(0, 2, 3, 1).float().numpy()
    img = (img * 255).round().astype("uint8")
    save_path = f"{SAMPLES_DIR}/epoch_{epoch}.png"
    Image.fromarray(img[0]).save(save_path)
    print(f"Saved validation sample: {save_path}")
    model.train()

def train():
    setup_dirs()
    
    model = SingleStreamDiTV2(text_embed_dim=TEXT_DIM).to(DEVICE)
    vae = AutoencoderKL.from_pretrained("KBlueLeaf/EQ-SDXL-VAE").to(DEVICE)
    
    if RESUME_FROM:
        print(f"Resuming weights from {RESUME_FROM}...")
        try:
            model.load_state_dict(torch.load(RESUME_FROM), strict=False)
            print("Weights loaded successfully.")
        except Exception as e:
            print(f"WARNING: Could not load weights: {e}")
            print("Starting fresh instead.")
            model.initialize_weights()
    else:
        print("Starting Fresh (Zero-Init Applied).")
        model.initialize_weights()

    optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=LEARNING_RATE)    
    scheduler = CosineAnnealingLR(optimizer, T_max=SCHEDULER_T_MAX, eta_min=1e-5)
    
    dataset = BucketDataset(CACHE_DIR)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    is_resuming = RESUME_FROM is not None
    logger = CSVLogger(LOG_FILE, resume=is_resuming)
    
    global_step = START_EPOCH * len(dataloader)
    print(f"Starting training from Epoch {START_EPOCH}...")

    for epoch in range(START_EPOCH, EPOCHS + 1):
        pbar = tqdm(dataloader)
        epoch_loss = 0
        steps_in_epoch = 0
        
        current_lr = optimizer.param_groups[0]['lr']
        
        for batch in pbar:
            x_1 = batch["latents"][0].to(DEVICE, dtype=torch.bfloat16)
            text = batch["text_embeds"][0].to(DEVICE, dtype=torch.bfloat16)
            h, w = batch["height"][0].item(), batch["width"][0].item()

            x_0 = torch.randn_like(x_1)
            t = torch.rand(x_1.shape[0], device=DEVICE, dtype=torch.bfloat16)
            t_exp = t.view(-1, 1, 1, 1)
            x_t = (1 - t_exp) * x_0 + t_exp * x_1
            target_v = x_1 - x_0
            
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred = model(x_t, t, text, h, w)
                loss = F.mse_loss(pred, target_v)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            loss_val = loss.item()
            epoch_loss += loss_val
            steps_in_epoch += 1
            global_step += 1
            
            logger.log(epoch, global_step, loss_val, current_lr)
            pbar.set_description(f"Ep {epoch} | Loss: {loss_val:.4f} | LR: {current_lr:.2e}")

        scheduler.step()
        avg_loss = epoch_loss / steps_in_epoch if steps_in_epoch > 0 else 0
        print(f"Epoch {epoch} finished. Avg Loss: {avg_loss:.4f}")

        if epoch > 0 and epoch % VALIDATE_EVERY == 0:
            validate(model, vae, epoch, CACHE_DIR)

        if epoch > 0 and epoch % SAVE_EVERY == 0:
            path = f"{CHECKPOINT_DIR}/epoch_{epoch}.pt"
            torch.save(model.state_dict(), path)
            print(f"Checkpoint saved: {path}")

if __name__ == "__main__":
    train()