import torch
import os
from tqdm import tqdm

CACHE_DIR = "./cached_data"

def check_stats():
    files = [os.path.join(CACHE_DIR, f) for f in os.listdir(CACHE_DIR) if f.endswith('.pt')]
    
    all_means = []
    all_stds = []
    
    print("Calculating Latent Statistics...")
    for f in tqdm(files):
        d = torch.load(f)
        l = d["latents"].float()
        all_means.append(l.mean())
        all_stds.append(l.std())
        
    total_mean = torch.tensor(all_means).mean().item()
    total_std = torch.tensor(all_stds).mean().item()
    
    print(f"\n--- RESULTS ---")
    print(f"Current Mean (Should be near 0): {total_mean:.4f}")
    print(f"Current Std  (Should be near 1): {total_std:.4f}")
    
    if total_std < 0.8 or total_std > 1.2:
        print("\n[!] WARNING: YOUR SCALING FACTOR IS WRONG.")
        correct_scale = 0.13025 / total_std
        print(f"You should change 0.13025 to approx: {correct_scale * 0.13025:.5f}")
    else:
        print("\n[OK] Scaling factor is correct.")

if __name__ == "__main__":
    check_stats()