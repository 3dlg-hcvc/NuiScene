import os
import torch
import hydra
import random
import diffusers
from tqdm import tqdm
import torch.nn as nn
from einops import rearrange
import lightning.pytorch as pl
from omegaconf import OmegaConf
from positional_encodings.torch_encodings import PositionalEncoding1D

from src.model.util import FourierEmbedder
from src.model.diff.transformer import Denoiser

class NuiModel(pl.LightningModule):
    def __init__(self, cfg, vae):
        super().__init__()
        if type(cfg) == dict:
            cfg = OmegaConf.create(cfg)

        self.vae = vae
        for param in self.vae.parameters():
            param.requires_grad = False

        if self.vae.cfg.model_vae.network.ae.use_triplane:
            cfg.model_diff.network.vae_num_latents = 3 * self.vae.cfg.model_vae.network.ae.triplane_res * self.vae.cfg.model_vae.network.ae.triplane_res
        else:
            cfg.model_diff.network.vae_num_latents = self.vae.cfg.model_vae.network.ae.num_latents
        cfg.model_diff.network.vae_latent_dim = self.vae.cfg.model_vae.network.ae.latent_dim

        self.save_hyperparameters(OmegaConf.to_container(cfg))

        self.cfg = cfg
        self.vae_scaling_factor = self.cfg.vae_scaling_factor
        self.drop_cond_p = self.cfg.model_diff.network.drop_cond_p
        self.scheduler = hydra.utils.instantiate(cfg.model_diff.diffusion_scheduler.scheduler_config)

        self.lambda_denoise = getattr(cfg.model_diff.network, 'lambda_denoise', 1)

        self.chunk_fourier = FourierEmbedder(input_dim=1)
        self.qkv_norm = getattr(self.cfg.model_diff.network, 'qkv_norm', True)
        self.denoiser = Denoiser(cfg.model_diff, token_dim=self.chunk_fourier.out_dim + self.vae.cfg.model_vae.network.ae.latent_dim + 1, qkv_norm=self.qkv_norm)

        self.chunk_fourier_out_dim = self.chunk_fourier.out_dim
        self.chunk_fourier = PositionalEncoding1D(self.chunk_fourier_out_dim)

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(self.cfg.model_diff.optimizer, params=self.parameters())
        scheduler = hydra.utils.instantiate(self.cfg.model_diff.scheduler, optimizer=optimizer)

        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "name": "cosine_annealing_lr"}}

    def training_step(self, batch, batch_idx):
        chunk_embeds = batch['embeds']
        bs, n_chunk, _, _ = chunk_embeds.shape

        mask_mode = ['left-right', 'top-down', 'diagonal']
        chosen_mode = random.choice(mask_mode)
        if chosen_mode == 'left-right':
            mask_cond_idx = [1, 3]
            mask_idx = [0, 2]
        elif chosen_mode == 'top-down':
            mask_cond_idx = [2, 3]
            mask_idx = [0, 1]
        elif chosen_mode == 'diagonal':
            mask_cond_idx = [3]
            mask_idx = [0, 1, 2]

        with torch.no_grad():
            # Mask to keep track which chunks are conditioned
            mask = torch.zeros((bs, n_chunk, chunk_embeds.shape[2], 1), device=chunk_embeds.device, dtype=chunk_embeds.dtype)

            cond_embeds = chunk_embeds.clone()

            sample = random.random()
            if self.drop_cond_p > sample:
                # Zero out all to diffuse entire sequence
                cond_embeds = torch.zeros_like(cond_embeds)
            else:
                # Zero out the chunks we want to outpaint for condition
                cond_embeds[:, mask_cond_idx] = torch.zeros_like(cond_embeds[:, mask_cond_idx])

                # Else condition on the first two chunks in cond_embeds
                mask[:, mask_idx] = torch.ones_like(mask[:, mask_idx])

                # Add noise to the condition
                t_cond = torch.randint(0, 400, (bs,), device=chunk_embeds.device).long()
                noise_cond = torch.randn_like(cond_embeds[:, mask_idx])
                cond_embeds[:, mask_idx] = self.scheduler.add_noise(cond_embeds[:, mask_idx], noise_cond, t_cond)

            # Concat with mask and go through linear
            cond_embeds = torch.cat((cond_embeds, mask), -1)
        chunk_embeds = chunk_embeds * self.vae_scaling_factor
        t = torch.randint(0, self.scheduler.config.num_train_timesteps, (bs,), device=chunk_embeds.device).long()

        noise = torch.randn_like(chunk_embeds)
        noise = rearrange(noise, "bs n_chunk n_latents c -> bs (n_chunk n_latents) c", bs=bs, n_chunk=n_chunk)
        cond_embeds = rearrange(cond_embeds, "bs n_chunk n_latents c -> bs (n_chunk n_latents) c", bs=bs, n_chunk=n_chunk)
        chunk_embeds = rearrange(chunk_embeds, "bs n_chunk n_latents c -> bs (n_chunk n_latents) c", bs=bs, n_chunk=n_chunk)

        noise_chunk = self.scheduler.add_noise(chunk_embeds, noise, t)

        # Concat cond embeds and positional encoding
        chunk_token = torch.rand((bs, noise_chunk.shape[1], self.chunk_fourier_out_dim), device=noise_chunk.device, dtype=noise_chunk.dtype)
        chunk_token = self.chunk_fourier(chunk_token)
        noise_chunk = torch.cat((noise_chunk, cond_embeds, chunk_token), -1)

        denoise = self.denoiser(noise_chunk, t)
        if self.scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.scheduler.config.prediction_type == "v_prediction":
            target = self.scheduler.get_velocity(chunk_embeds, noise, t)
        loss_denoise = torch.nn.functional.mse_loss(denoise, target)

        loss = self.lambda_denoise * loss_denoise

        self.log('train_loss/mse_loss', loss_denoise)
        return loss

    @torch.no_grad()
    def generate_embeddings(self, batch_size, num_diff_timesteps=50):
        device = self.denoiser.input_proj.weight.device

        mask = torch.zeros((batch_size, self.cfg.data.train.metadata.num_chunks, self.denoiser.vae_num_latents, 1), device=device)
        cond_embeds = torch.zeros((batch_size, self.cfg.data.train.metadata.num_chunks, self.denoiser.vae_num_latents, self.denoiser.vae_latent_dim), device=device)
        cond_embeds = torch.cat((cond_embeds, mask), -1)

        noise = torch.randn((batch_size, self.cfg.data.train.metadata.num_chunks, self.denoiser.vae_num_latents, self.denoiser.vae_latent_dim), device=device)
        noise = rearrange(noise, "bs n_chunk n_latents c -> bs (n_chunk n_latents) c")
        cond_embeds = rearrange(cond_embeds, "bs n_chunk n_latents c -> bs (n_chunk n_latents) c")

        chunk_token = torch.rand((noise.shape[0], noise.shape[1], self.chunk_fourier_out_dim), device=noise.device, dtype=noise.dtype)
        chunk_token = self.chunk_fourier(chunk_token)

        self.scheduler.set_timesteps(num_diff_timesteps, device=noise.device)
        diff_timesteps = self.scheduler.timesteps

        chunk_denoise = torch.cat((noise, cond_embeds, chunk_token), -1)
        for _, t in enumerate(diff_timesteps):
            timestep = torch.tensor(t, device=device).long().repeat(batch_size)
            noise_pred = self.denoiser(chunk_denoise, timestep)

            chunk_denoise = self.scheduler.step(noise_pred, t, noise, return_dict=False)[0]
            noise = chunk_denoise
            chunk_denoise = torch.cat((noise, cond_embeds, chunk_token), -1)
        chunk_denoise = noise.squeeze(0)

        return chunk_denoise
    
    @torch.no_grad()
    def generate_embeddings_cond(self, batch_size, mask_region=None, mask_embeddings=None, num_diff_timesteps=50):
        device = self.denoiser.input_proj.weight.device

        mask = torch.zeros((batch_size, self.cfg.data.train.metadata.num_chunks, self.denoiser.vae_num_latents, 1), device=device)
        cond_embeds = torch.zeros((batch_size, self.cfg.data.train.metadata.num_chunks, self.denoiser.vae_num_latents, self.denoiser.vae_latent_dim), device=device)

        for b, region in enumerate(mask_region):
            mask[b, region] = 1
            cond_embeds[b, region] = mask_embeddings[b]
        cond_embeds = torch.cat((cond_embeds, mask), -1)

        noise = torch.randn((batch_size, self.cfg.data.train.metadata.num_chunks, self.denoiser.vae_num_latents, self.denoiser.vae_latent_dim), device=device)
        noise = rearrange(noise, "bs n_chunk n_latents c -> bs (n_chunk n_latents) c")
        cond_embeds = rearrange(cond_embeds, "bs n_chunk n_latents c -> bs (n_chunk n_latents) c")

        chunk_token = torch.rand((noise.shape[0], noise.shape[1], self.chunk_fourier_out_dim), device=noise.device, dtype=noise.dtype)
        chunk_token = self.chunk_fourier(chunk_token)

        self.scheduler.set_timesteps(num_diff_timesteps, device=noise.device)
        diff_timesteps = self.scheduler.timesteps

        chunk_denoise = torch.cat((noise, cond_embeds, chunk_token), -1)
        for _, t in enumerate(diff_timesteps):
            timestep = torch.tensor(t, device=device).long().repeat(batch_size)
            noise_pred = self.denoiser(chunk_denoise, timestep)

            chunk_denoise = self.scheduler.step(noise_pred, t, noise, return_dict=False)[0]
            noise = chunk_denoise
            chunk_denoise = torch.cat((noise, cond_embeds, chunk_token), -1)
        chunk_denoise = noise

        return chunk_denoise
    
    @torch.no_grad()
    def decode_embeddings(self, chunk_denoise, gen_grid_resolution):
        occs = []
        for i in range(self.cfg.data.train.metadata.num_chunks):
            single_chunk_embed = chunk_denoise[i*self.denoiser.vae_num_latents:(i+1)*self.denoiser.vae_num_latents]
            occ = self.vae.predict_grid_occ_from_embed(single_chunk_embed.unsqueeze(0), grid_resolution=gen_grid_resolution)
            occs.append(occ)
        return occs
    
    @torch.no_grad()
    def decode_embedding(self, chunk_denoise, gen_grid_resolution):
        single_chunk_embed = chunk_denoise
        occ = self.vae.predict_grid_occ_from_embed(single_chunk_embed.unsqueeze(0), grid_resolution=gen_grid_resolution)
        return occ
