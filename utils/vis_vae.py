import os
import hydra
import torch
import trimesh
import numpy as np
import torch.nn as nn
import skimage.measure
import lightning.pytorch as pl

import sys
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, src_path)

from src.model.vae.model import VAEModel
from src.data.data_module import DataModule

@hydra.main(version_base=None, config_path="../configs", config_name="vae_config")
def main(_cfg):
    # Load the model and get config
    chunk_size = _cfg.quad_chunk_size // 2
    model = VAEModel.load_from_checkpoint(_cfg.ckpt_path)
    model.eval()
    model.cuda()
    cfg = model.cfg

    # Temporary only works with batch size 1
    cfg.data.train.dataloader.batch_size = 1
    cfg.data.val.dataloader.batch_size = 1

    output_folder = os.path.join('output', 'vae', _cfg.ckpt_path.split('/')[1], 'recon_chunks')
    os.makedirs(output_folder, exist_ok=True)

    # fix the seed
    pl.seed_everything(cfg.train_seed, workers=True)

    # cfg.data.val.dataloader.num_workers = 0

    # initialize data
    data_module = DataModule(cfg.data)

    data_module.setup(stage='fit')
    val_dataloader = data_module.val_dataloader()
    # val_dataloader = data_module.train_dataloader()
    val_dataloader.dataset.random_rotation = False

    num_samples = 20
    for counter, batch in enumerate(val_dataloader):
        print('Processing {}/{} objects'.format(counter+1, len(val_dataloader)))

        comb_occ = torch.zeros((4 * chunk_size, chunk_size * 10, 4 * chunk_size))

        for i in range(4):
            pcd = batch['pcd'][:, i].cuda()
            occ, pred_height = model.predict_grid_occ(pcd, grid_resolution=chunk_size)

            if i == 0:
                comb_occ[0:chunk_size, :occ.shape[1], 0:chunk_size] = occ.cpu()
            elif i == 1:
                comb_occ[0:chunk_size, :occ.shape[1], chunk_size:2*chunk_size] = occ.cpu()
            elif i == 2:
                comb_occ[chunk_size:2*chunk_size, :occ.shape[1], 0:chunk_size] = occ.cpu()
            elif i == 3:
                comb_occ[chunk_size:2*chunk_size, :occ.shape[1], chunk_size:2*chunk_size] = occ.cpu()

        sigmoid = nn.Sigmoid()
        occ = sigmoid(comb_occ)

        volume = occ.cpu().numpy()
        verts, faces, _, _ = skimage.measure.marching_cubes(volume, 0.5)
        gap = 2. / chunk_size
        verts *= gap
        verts -= 1
        m = trimesh.Trimesh(verts, faces)
        m.export(os.path.join(output_folder, 'recon_chunk_{}.obj'.format(counter)))

        if counter == num_samples - 1:
            break

if __name__ == '__main__':
    main()
