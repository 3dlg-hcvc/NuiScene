import os
import h5py
import hydra
import torch
import trimesh
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import skimage.measure
import lightning.pytorch as pl

import sys
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, src_path)

from src.model.vae.model import VAEModel
from src.data.data_module import DataModule

@hydra.main(version_base=None, config_path="../configs", config_name="vae_config")
def main(_cfg):
    cache_batch_size = _cfg.cache_batch_size

    # Load the model and get config
    model = VAEModel.load_from_checkpoint(_cfg.ckpt_path)
    model.eval()
    model.cuda()
    cfg = model.cfg

    model_name = _cfg.ckpt_path.split('/')[-3]

    # fix the seed
    pl.seed_everything(cfg.train_seed, workers=True)

    # Custom dataloader to get all pcd and their rotations
    cfg.data.val.dataset = 'CacheChunkPcd'
    cfg.data.train.dataset = 'CacheChunkPcd'
    cfg.data.train.metadata.axis_scaling = False
    cfg.data.train.dataloader.batch_size = cache_batch_size

    # initialize data
    data_module = DataModule(cfg.data)

    data_module.setup(stage='fit')
    train_dataloader = data_module.train_dataloader()

    with h5py.File(os.path.join('dataset/data', '{}.h5'.format(model_name)), "w") as h5_file:
        if cfg.model_vae.network.ae.use_triplane:
            total_res = 3 * cfg.model_vae.network.ae.triplane_res * cfg.model_vae.network.ae.triplane_res
            h5_file.create_dataset("embeds", shape=(len(train_dataloader.dataset), 4, 4, total_res, cfg.model_vae.network.ae.latent_dim), dtype=np.float32)
        else:
            total_res = cfg.model_vae.network.ae.num_latents
            h5_file.create_dataset("embeds", shape=(len(train_dataloader.dataset), 4, 4, total_res, cfg.model_vae.network.ae.latent_dim), dtype=np.float32)

        for idx, batch in enumerate(tqdm(train_dataloader)):
            all_pcd = batch['pcd']
            bs, num_rot, num_chunk, num_pcd, _ = all_pcd.shape
            all_pcd = all_pcd.reshape(bs * num_rot * num_chunk, num_pcd, 3)

            ind = np.random.choice(all_pcd.shape[1], cfg.data.train.metadata.pc_size, replace=False)
            all_pcd = all_pcd[:, ind]

            with torch.no_grad():
                all_embeds, _ = model.shape_ae.encode(all_pcd.cuda())
            all_embeds = all_embeds.reshape(bs, num_rot, num_chunk, total_res, cfg.model_vae.network.ae.latent_dim).cpu().numpy()
            h5_file['embeds'][idx * cfg.data.train.dataloader.batch_size:(idx+1) * cfg.data.train.dataloader.batch_size] = all_embeds

if __name__ == '__main__':
    main()
