import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def create_2d_rope_grid(h, w, dim, device):
    """Generates 2D Rotary Positional Embeddings for a given grid size."""
    grid_h = torch.arange(h, device=device)
    grid_w = torch.arange(w, device=device)
    grid = torch.meshgrid(grid_h, grid_w, indexing='ij')
    grid = torch.stack(grid, dim=-1).reshape(-1, 2)
    
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 4, device=device).float() / dim))
    t_h = grid[:, 0:1] * inv_freq
    t_w = grid[:, 1:2] * inv_freq
    
    freqs = torch.cat([t_h, t_w], dim=-1)
    return torch.cos(freqs), torch.sin(freqs)

def apply_rope(x, cos, sin):
    x_fp32 = x.float()
    cos_fp32 = cos.float()
    sin_fp32 = sin.float()
    
    d = x_fp32.shape[-1]
    x1 = x_fp32[..., :d//2]
    x2 = x_fp32[..., d//2:]
    
    rotated = torch.cat([x1 * cos_fp32 - x2 * sin_fp32, x1 * sin_fp32 + x2 * cos_fp32], dim=-1)
    
    return rotated.to(dtype=x.dtype)

class DiTBlockV2(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.qkv = nn.Linear(hidden_size, hidden_size * 3)
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_size * 4, hidden_size)
        )
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size))

    def forward(self, x, c, cos, sin):
        B, N, C = x.shape
        
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        
        res = (1 + scale_msa.unsqueeze(1)) * self.norm1(x) + shift_msa.unsqueeze(1)
        qkv = self.qkv(res).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q_img = q[:, :, 256:, :]
        k_img = k[:, :, 256:, :]
        q[:, :, 256:, :] = apply_rope(q_img, cos, sin)
        k[:, :, 256:, :] = apply_rope(k_img, cos, sin)
        
        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).reshape(B, N, C)
        x = x + gate_msa.unsqueeze(1) * self.proj(attn)
        
        res_mlp = (1 + scale_mlp.unsqueeze(1)) * self.norm2(x) + shift_mlp.unsqueeze(1)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(res_mlp)
        return x

class SingleStreamDiTV2(nn.Module):
    def __init__(self, in_channels=4, hidden_size=1024, depth=12, num_heads=16, text_embed_dim=1152):
        super().__init__()
        self.patch_size = 2
        self.x_embedder = nn.Linear(in_channels * 4, hidden_size)
        self.text_embedder = nn.Linear(text_embed_dim, hidden_size)
        self.t_embedder = nn.Sequential(nn.Linear(256, hidden_size), nn.SiLU(), nn.Linear(hidden_size, hidden_size))
        self.blocks = nn.ModuleList([DiTBlockV2(hidden_size, num_heads) for _ in range(depth)])
        self.final_layer = nn.Linear(hidden_size, in_channels * 4)
        self.initialize_weights()

    def forward(self, x, t, text_embeds, height, width):
        B, C, H, W = x.shape
        x = x.reshape(B, C, H//2, 2, W//2, 2).permute(0, 2, 4, 1, 3, 5).flatten(1, 2).flatten(2)
        x = self.x_embedder(x)
        
        context = self.text_embedder(text_embeds)
        x_concat = torch.cat([context, x], dim=1)
        
        t_freq = self.timestep_embedding(t, 256)
        t_emb = self.t_embedder(t_freq.to(x.dtype))
        
        cos, sin = create_2d_rope_grid(H//2, W//2, self.blocks[0].head_dim, x.device)
        cos, sin = cos.unsqueeze(0).unsqueeze(0), sin.unsqueeze(0).unsqueeze(0)

        for block in self.blocks:
            x_concat = block(x_concat, t_emb, cos, sin)
        
        x_out = self.final_layer(x_concat[:, 256:, :])
        return x_out.reshape(B, H//2, W//2, C, 2, 2).permute(0, 3, 1, 4, 2, 5).reshape(B, C, H, W)

    def timestep_embedding(self, t, dim):
        half = dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(0, half, device=t.device).float() / half)
        args = t[:, None].float() * freqs[None]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    
    def initialize_weights(self):
        nn.init.normal_(self.x_embedder.weight, std=0.02)
        nn.init.normal_(self.text_embedder.weight, std=0.02)
        nn.init.normal_(self.t_embedder[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder[2].weight, std=0.02)

        nn.init.constant_(self.final_layer.weight, 0)
        nn.init.constant_(self.final_layer.bias, 0)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)