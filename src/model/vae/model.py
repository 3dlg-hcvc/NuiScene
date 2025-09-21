import torch
import hydra
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import lightning.pytorch as pl
from omegaconf import OmegaConf

from src.model.vae.shape_ae import ShapeAE
from src.model.attention import ResidualCrossAttentionBlock

class VAEModel(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        if type(cfg) == dict:
            cfg = OmegaConf.create(cfg)
        self.save_hyperparameters(OmegaConf.to_container(cfg))

        self.cfg = cfg
        self.lambda_kl = self.cfg.model_vae.network.lambda_kl
        self.lambda_vol = self.cfg.model_vae.network.lambda_vol
        self.lambda_near = self.cfg.model_vae.network.lambda_near
        self.lambda_height = self.cfg.model_vae.network.lambda_height
        self.lambda_emb_dist = self.cfg.model_vae.network.lambda_emb_dist
        self.num_occ_samples = self.cfg.data.train.metadata.num_occ_samples

        self.shape_ae = ShapeAE(self.cfg.model_vae.network.ae)
        self.qkv_norm = getattr(self.cfg.model_vae.network.ae, 'qkv_norm', True)
        self.height_latent = nn.Parameter(torch.randn((1, 1, self.cfg.model_vae.network.ae.latent_dim), dtype=torch.float32))
        if self.cfg.model_vae.network.ae.latent_dim < 32:
            self.height_ca = ResidualCrossAttentionBlock(width=self.cfg.model_vae.network.ae.latent_dim, heads=1, data_width=None, qkv_norm=self.qkv_norm)
        else:
            self.height_ca = ResidualCrossAttentionBlock(width=self.cfg.model_vae.network.ae.latent_dim, heads=1, data_width=None, qkv_norm=self.qkv_norm)
        self.out_height = nn.Linear(self.cfg.model_vae.network.ae.latent_dim, 1)

        self.mse_loss = nn.MSELoss()
        self.occ_loss = torch.nn.BCEWithLogitsLoss()
        self.eval_occ = hydra.utils.instantiate(cfg.data.val.evaluator)

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(self.cfg.model_vae.optimizer, params=self.parameters())
        scheduler = hydra.utils.instantiate(self.cfg.model_vae.scheduler, optimizer=optimizer)

        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "name": "cosine_annealing_lr"}}

    def training_step(self, batch, batch_idx):
        pcd = batch['pcd'].cuda()
        gt_occ = batch['occ'].cuda()
        gt_height = batch['height'].cuda()
        occ_query = batch['occ_query'].cuda()

        bs, num_chunk, _, _ = pcd.shape
        pcd = pcd.reshape(bs * num_chunk, -1, 3)
        gt_occ = gt_occ.reshape(bs * num_chunk, -1)
        gt_height = gt_height.reshape(bs * num_chunk, -1)
        occ_query = occ_query.reshape(bs * num_chunk, -1, 3)

        ind_1 = np.random.choice(pcd.shape[1], self.cfg.data.train.metadata.pc_size, replace=False)
        pcd_1 = pcd[:, ind_1]
        ind_2 = np.random.choice(pcd.shape[1], self.cfg.data.train.metadata.pc_size, replace=False)
        pcd_2 = pcd[:, ind_2]

        all_pcd = torch.cat((pcd_1, pcd_2), dim=0)
        all_kl_embed, posterior = self.shape_ae.forward_encode(all_pcd)
        kl_embed_1, kl_embed_2 = torch.split(all_kl_embed, bs * num_chunk)
        pred_occ = self.shape_ae.forward_decode(kl_embed_1, occ_query)

        # Get KL loss from posterior
        if self.cfg.model_vae.network.ae.latent_dim > 0:
            loss_kl = posterior.kl()
            loss_kl = torch.sum(loss_kl) / loss_kl.shape[0]
        else:
            loss_kl = torch.tensor([0], device=kl_embed_1.device, dtype=kl_embed_1.dtype)

        # Get occupancy losses
        loss_vol = self.occ_loss(pred_occ[:, :self.num_occ_samples], gt_occ[:, :self.num_occ_samples])
        loss_near = self.occ_loss(pred_occ[:, self.num_occ_samples:], gt_occ[:, self.num_occ_samples:])

        # Regularization losses
        loss_embed_dist = self.mse_loss(kl_embed_1, kl_embed_2)

        height_feats = self.height_ca(self.height_latent.repeat(kl_embed_1.shape[0], 1, 1), kl_embed_1).squeeze()
        pred_height = self.out_height(height_feats)
        loss_height = self.mse_loss(pred_height, gt_height)

        loss = self.lambda_kl * loss_kl + self.lambda_vol * loss_vol + self.lambda_near * loss_near + self.lambda_emb_dist * loss_embed_dist + self.lambda_height * loss_height

        self.log('train_loss/loss_kl', loss_kl.item())
        self.log('train_loss/total_loss', loss.item())
        self.log('train_loss/loss_vol', loss_vol.item())
        self.log('train_loss/loss_near', loss_near.item())
        self.log('train_loss/loss_height', loss_height.item())
        self.log('train_loss/loss_emb_dist', loss_embed_dist.item())

        return loss
    
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        pcd = batch['pcd'].cuda()
        gt_occ = batch['occ'].cuda()
        occ_query = batch['occ_query'].cuda()

        bs, num_chunk, _, _ = pcd.shape
        pcd = pcd.reshape(bs * num_chunk, -1, 3)
        gt_occ = gt_occ.reshape(bs * num_chunk, -1)
        occ_query = occ_query.reshape(bs * num_chunk, -1, 3)

        ind_1 = np.random.choice(pcd.shape[1], self.cfg.data.train.metadata.pc_size, replace=False)
        pcd = pcd[:, ind_1]

        _, _, pred_occ = self.shape_ae(pcd, occ_query)

        self.eval_occ.update(pred_occ, gt_occ)

    def on_validation_epoch_end(self):
        acc = self.eval_occ.compute()
        self.log(f"val_eval/eval_occ", acc, sync_dist=True)
        self.eval_occ.reset()

    @torch.no_grad()
    def predict_grid_occ(self, pcd, grid_resolution, chunk_size=100000, use_pred_height=True):
        real_height = pcd[0].max(0)[0][1].item() + 0.5
        ind_1 = np.random.choice(pcd.shape[1], self.cfg.data.train.metadata.pc_size, replace=False)
        pcd = pcd[:, ind_1]
        kl_embed, _ = self.shape_ae.encode(pcd)

        if use_pred_height:
            height_feats = self.height_ca(self.height_latent.repeat(kl_embed.shape[0], 1, 1), kl_embed).squeeze()
            # pred_height = self.out_height(height_feats).item() + 1
            pred_height = self.out_height(height_feats).item() + 0.5
            pred_height_scale = (pred_height + 1) / 2
        else:
            pred_height = real_height
            pred_height_scale = (pred_height + 1) / 2

        # Recalculate to avoid fraction errors
        y_num_voxels = int(grid_resolution * pred_height_scale)
        y_height = (y_num_voxels / grid_resolution) * 2 - 1

        x = np.linspace(-1, 1, grid_resolution)
        y = np.linspace(-1, y_height, y_num_voxels)
        z = np.linspace(-1, 1, grid_resolution)
        xv, yv, zv = np.meshgrid(x, y, z, indexing='ij')
        grid = torch.from_numpy(np.stack([xv, yv, zv]).astype(np.float16)).view(3, -1).transpose(0, 1)[None].to(pcd.device)

        ind_1 = np.random.choice(pcd.shape[1], self.cfg.data.train.metadata.pc_size, replace=False)
        pcd = pcd[:, ind_1]
        kl_embed, _ = self.shape_ae.encode(pcd)

        occ = []
        num_iterations = np.ceil(grid.shape[1] / chunk_size)
        for i in range(int(num_iterations)):
            chunk_query = grid[:, i*chunk_size:(i+1)*chunk_size]
            chunk_occ = self.shape_ae.decode(kl_embed, chunk_query)
            occ.append(chunk_occ)
        occ = torch.cat(occ, dim=1).squeeze().reshape(grid_resolution, int(grid_resolution * pred_height_scale), grid_resolution)

        return occ, pred_height
    
    @torch.no_grad()
    def predict_grid_occ_from_embed(self, kl_embed, grid_resolution, chunk_size=100000):
        height_feats = self.height_ca(self.height_latent.repeat(kl_embed.shape[0], 1, 1), kl_embed).squeeze()
        # pred_height = self.out_height(height_feats).item() + 1
        pred_height = self.out_height(height_feats).item() + 0.5
        # pred_height = self.out_height(height_feats).item()
        pred_height_scale = (pred_height + 1) / 2

        x = np.linspace(-1, 1, grid_resolution[0])
        y = np.linspace(-1, pred_height, int(grid_resolution[0] * pred_height_scale))
        z = np.linspace(-1, 1, grid_resolution[2])
        xv, yv, zv = np.meshgrid(x, y, z, indexing='ij')
        grid = torch.from_numpy(np.stack([xv, yv, zv]).astype(np.float16)).view(3, -1).transpose(0, 1)[None].to(kl_embed.device)

        occ = []
        num_iterations = np.ceil(grid.shape[1] / chunk_size)
        for i in range(int(num_iterations)):
            chunk_query = grid[:, i*chunk_size:(i+1)*chunk_size]
            chunk_occ = self.shape_ae.decode(kl_embed, chunk_query)
            occ.append(chunk_occ)
        occ = torch.cat(occ, dim=1).squeeze().reshape(grid_resolution[0], int(grid_resolution[0] * pred_height_scale), grid_resolution[2])
        return occ
