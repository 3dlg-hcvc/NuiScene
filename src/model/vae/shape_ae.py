################################################################################################################################################
"""
Modifed From: https://github.com/wyysf-98/CraftsMan
"""
################################################################################################################################################
import math
import torch
import numpy as np
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F

from src.model.attention import ResidualAttentionBlock, ResidualCrossAttentionBlock
from src.model.util import FourierEmbedder, DeconvUpsampler, DiagonalGaussianDistribution

class ShapeAE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        self.heads = self.cfg.heads
        self.width = self.cfg.width
        self.depth = self.cfg.depth
        self.output_dim = self.cfg.output_dim
        self.latent_dim = self.cfg.latent_dim
        self.num_latents = self.cfg.num_latents
        self.qkv_norm = getattr(self.cfg, 'qkv_norm', True)

        self.triplane_res = self.cfg.triplane_res
        self.use_triplane = self.cfg.use_triplane
        self.triplane_channels = self.cfg.triplane_channels

        ############################################
        """
        For Encoder
        """
        ############################################
        self.embedder = FourierEmbedder()
        self.in_proj = nn.Linear(self.embedder.out_dim, self.width)

        if self.use_triplane:
            self.renorm_factor = self.cfg.renorm_factor
            self.num_deconv = getattr(self.cfg, 'num_deconv', 2)
            self.fixed_latent = nn.Parameter(torch.randn((1, 3 * self.triplane_res * self.triplane_res, self.width),
                                            dtype=torch.float32,) * 1 / math.sqrt(self.width))
        else:
            self.fixed_latent = nn.Parameter(torch.randn((1, self.num_latents, self.width), dtype=torch.float32))

        # Cross Attention for input pcd
        self.input_ca = ResidualCrossAttentionBlock(width=self.width, heads=self.heads, data_width=None, qkv_norm=self.qkv_norm)

        # MLPs if using VAE-style embedding
        if self.latent_dim > 0:
            self.pre_kl = nn.Linear(self.width, self.latent_dim * 2)
            self.post_kl = nn.Linear(self.latent_dim, self.width)
            self.latent_shape = (self.num_latents, self.latent_dim)
        else:
            self.latent_shape = (self.num_latents, self.width)

        ############################################
        """
        For Decoder
        """
        ############################################

        # Self attention layers for decoder
        self.resblocks = nn.ModuleList([
                ResidualAttentionBlock(n_ctx=self.num_latents, width=self.width, heads=self.heads, qkv_norm=self.qkv_norm) 
                for _ in range(self.depth)
            ])

        # Triplane tokens if using triplane
        if self.use_triplane:
            # Upsample the triplane like LRM
            self.upsample = DeconvUpsampler(in_channels=self.width, hidden_channels=self.width, out_channels=self.triplane_channels, num_deconv=self.num_deconv)

            # Output projection
            self.out_proj = nn.Linear(self.triplane_channels * 3, self.output_dim)
        else:
            # Cross Attention for output queries
            self.output_ca = ResidualCrossAttentionBlock(width=self.width, heads=self.heads, data_width=None, qkv_norm=self.qkv_norm)

            # Query projection for output queries
            self.query_proj = nn.Linear(self.embedder.out_dim, self.width)

            self.vecset_upsample = getattr(self.cfg, 'vecset_upsample', False)
            if self.vecset_upsample:
                self.upsample_factor = self.cfg.upsample_num_latents // self.num_latents
                self.vecset_upsample_proj = nn.Linear(self.width, self.width * self.upsample_factor)
                self.vecset_upsample_layers = nn.ModuleList([
                    ResidualAttentionBlock(n_ctx=self.cfg.upsample_num_latents, width=self.width, heads=self.heads, qkv_norm=self.qkv_norm) 
                    for _ in range(self.cfg.num_vecset_upsample_layers)
                ])

            # output projection
            self.out_proj = nn.Linear(self.width, self.output_dim)

    def encode(self, xyz, sample_posterior=True):
        # Get fourier for xyz and normals
        embed_xyz = self.embedder(xyz)
        input_feats = self.in_proj(embed_xyz)

        latent_feats = self.fixed_latent.repeat(embed_xyz.shape[0], 1, 1)

        # CA between latent pcd features and input pcd features
        latent_feats = self.input_ca(latent_feats, input_feats)

        # If using VAE style embed, get gaussian re-param
        posterior = None
        if self.latent_dim > 0:
            moments = self.pre_kl(latent_feats)
            posterior = DiagonalGaussianDistribution(moments, feat_dim=-1)
            if sample_posterior:
                kl_embed = posterior.sample()
            else:
                kl_embed = posterior.mode()
        else:
            kl_embed = latent_feats
        
        return kl_embed, posterior

    def decode(self, kl_embed, occ_query):
        # Linear layer if get from VAE embed
        if self.latent_dim > 0:
            latent_feats = kl_embed
            latent_feats = self.post_kl(latent_feats)
        else:
            latent_feats = kl_embed

        # Self attention layers
        for i in range(len(self.resblocks)):
            latent_feats = self.resblocks[i](latent_feats)

        if self.use_triplane:
            bs = kl_embed.shape[0]
            triplane_emb = latent_feats.reshape(bs, 3, self.triplane_res, self.triplane_res, -1)
            triplane_emb = rearrange(triplane_emb, "bs np rw rh c -> (bs np) c rw rh")
            triplane_emb = self.upsample(triplane_emb)
            triplane_emb = rearrange(triplane_emb, "(bs np) c rw rh -> bs np rw rh c", bs=bs)

            renorm_query = occ_query.float()
            renorm_query[:, :, 1] = (renorm_query[:, :, 1] + 1) / 2
            renorm_query[:, :, 1] = renorm_query[:, :, 1] / self.renorm_factor
            renorm_query[:, :, 1] = renorm_query[:, :, 1] * 2 - 1

            indices2D = torch.stack((renorm_query[..., [0, 1]], renorm_query[..., [0, 2]], renorm_query[..., [1, 2]]), dim=-3)
            out = F.grid_sample(triplane_emb.reshape(bs * 3, self.triplane_res * (2**self.num_deconv), self.triplane_res * (2**self.num_deconv), self.triplane_channels).permute(0, 3, 1, 2), 
                                indices2D.reshape(bs * 3, 1, -1, 2).float(), align_corners=True, mode="bilinear", padding_mode="border").squeeze()
            out = rearrange(out, "(Bs Nt) Ct Qu -> Bs Qu (Nt Ct)", Nt=3)
            output = self.out_proj(out)
        else:
            if self.vecset_upsample:
                latent_feats = self.vecset_upsample_proj(latent_feats)
                latent_feats = rearrange(latent_feats, "bs nl (c f) -> bs (nl f) c", f=self.upsample_factor)

                for i in range(len(self.vecset_upsample_layers)):
                    latent_feats = self.vecset_upsample_layers[i](latent_feats)

            # Cross Attention with query locations to get occupancies
            occ_query_embed = self.embedder(occ_query)
            occ_query_embed = self.query_proj(occ_query_embed)
            output_feats = self.output_ca(occ_query_embed, latent_feats)
            output = self.out_proj(output_feats)

        return output
    
    def get_final_latent_feats(self, kl_embed):
        # Linear layer if get from VAE embed
        if self.latent_dim > 0:
            latent_feats = self.post_kl(kl_embed)
        else:
            latent_feats = kl_embed

        # Self attention layers
        for i in range(len(self.resblocks)):
            latent_feats = self.resblocks[i](latent_feats)

        return latent_feats

    def forward_encode(self, pcd):
        kl_embed, posterior = self.encode(pcd)
        return kl_embed, posterior
    
    def forward_decode(self, kl_embed, occ_query):
        output = self.decode(kl_embed, occ_query)
        return output.squeeze(-1)

    def forward(self, pcd, occ_query):
        kl_embed, posterior = self.encode(pcd)
        output = self.decode(kl_embed, occ_query)
        return kl_embed, posterior, output.squeeze(-1)
