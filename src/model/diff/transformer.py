################################################################################################################################################
"""
Modifed From: https://github.com/wyysf-98/CraftsMan
"""
################################################################################################################################################
import torch
import torch.nn as nn
from diffusers.models.embeddings import Timesteps

from src.model.attention import MLP, ResidualAttentionBlock

class ResidualTransformer(nn.Module):
    def __init__(self, n_ctx, width, heads, num_layers, qkv_norm):
        super().__init__()

        self.encoder = nn.ModuleList()
        for _ in range(num_layers):
            resblock = ResidualAttentionBlock(n_ctx=n_ctx, width=width, heads=heads, qkv_norm=qkv_norm)
            self.encoder.append(resblock)

        self.middle_block = ResidualAttentionBlock(n_ctx=n_ctx, width=width, heads=heads, qkv_norm=qkv_norm)

        self.decoder = nn.ModuleList()
        for _ in range(num_layers):
            resblock = ResidualAttentionBlock(n_ctx=n_ctx, width=width, heads=heads, qkv_norm=qkv_norm)
            linear = nn.Linear(width * 2, width)
            self.decoder.append(nn.ModuleList([resblock, linear]))

    def forward(self, x):
        enc_outputs = []
        for block in self.encoder:
            x = block(x)
            enc_outputs.append(x)

        x = self.middle_block(x)

        for (resblock, linear) in self.decoder:
            x = torch.cat([enc_outputs.pop(), x], dim=-1)
            x = linear(x)
            x = resblock(x)

        return x

class Denoiser(nn.Module):
    def __init__(self, cfg, token_dim, qkv_norm):
        super().__init__()
        self.cfg = cfg

        # Parameters
        self.heads = self.cfg.network.heads
        self.width = self.cfg.network.width
        self.vae_latent_dim = self.cfg.network.vae_latent_dim
        self.time_embed_dim = self.cfg.network.time_embed_dim
        self.vae_num_latents = self.cfg.network.vae_num_latents
        self.denoiser_enc_dec_num_layers = self.cfg.network.denoiser_enc_dec_num_layers

        # Input and output projections
        self.input_proj = nn.Linear(self.vae_latent_dim + token_dim, self.width)
        self.time_embed = Timesteps(self.time_embed_dim, flip_sin_to_cos=False, downscale_freq_shift=0)
        self.time_proj = nn.Sequential(nn.Linear(self.time_embed_dim, self.width), nn.SiLU(), nn.Linear(self.width, self.width))

        # Backbone for denoiser network
        self.ln_post = nn.LayerNorm(self.width)
        self.output_proj = nn.Linear(self.width, self.vae_latent_dim)
        self.backbone = ResidualTransformer(n_ctx=self.vae_num_latents, width=self.width, heads=self.heads, num_layers=self.denoiser_enc_dec_num_layers, qkv_norm=qkv_norm)

    def forward(self, chunk_embeds, t):
        x = self.input_proj(chunk_embeds)
        t_emb = self.time_proj(self.time_embed(t))
        x = torch.cat((x, t_emb.unsqueeze(1)), dim=1)

        x = self.backbone(x)
        x = self.ln_post(x)
        x = x[:, :-1]
        x = self.output_proj(x)

        return x
