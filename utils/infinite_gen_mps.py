import os
import time
import hydra
import torch
import random
import trimesh
import numpy as np
from tqdm import tqdm
import skimage.measure
import lightning.pytorch as pl

import sys
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, src_path)

from src.model.vae.model import VAEModel
from src.model.diff.model import NuiModel

@hydra.main(version_base=None, config_path="../configs", config_name="diff_config")
def main(_cfg):
    num_gen_rows = _cfg.num_gen_rows
    num_gen_cols = _cfg.num_gen_cols
    quad_chunk_size = _cfg.quad_chunk_size

    checkpoint = torch.load(_cfg.ckpt_path, map_location=torch.device('cpu'))
    vae = VAEModel.load_from_checkpoint(checkpoint['hyper_parameters']['vae_ckpt_path'])
    nuu = NuiModel(checkpoint['hyper_parameters'], vae)
    nuu.load_state_dict(checkpoint["state_dict"],)
    device = torch.device("mps")
    nuu.to(device)
    cfg = nuu.cfg

    gen_seed = random.getrandbits(32)
    res_multiplier = 0.5
    num_latents = cfg.model_diff.network.vae_num_latents
    decode_res_x = int(quad_chunk_size * res_multiplier)
    decode_res_y = int(quad_chunk_size * res_multiplier * 10)
    decode_res_z = int(quad_chunk_size * res_multiplier)
    overlap_voxel = cfg.data.train.metadata.overlap_voxel * res_multiplier

    output_folder = os.path.join('output', _cfg.ckpt_path.split('/')[-3])
    os.makedirs(output_folder, exist_ok=True)

    # fix the seed
    pl.seed_everything(gen_seed, workers=True)

    emb_gen_start = time.time()
    start_emb = nuu.generate_embeddings(batch_size=1)
    gen_grid_resolution = np.array([decode_res_x, decode_res_y, decode_res_z])

    comb_emb = torch.zeros(((num_gen_rows + 1), (num_gen_cols + 1), num_latents, start_emb.shape[-1])).float().to(start_emb.device)
    comb_occ = np.zeros((gen_grid_resolution[0] * (num_gen_rows + 1), gen_grid_resolution[1], gen_grid_resolution[2] * (num_gen_cols + 1)))

    comb_emb[0, 0] = start_emb[0:1*num_latents]
    comb_emb[0, 1] = start_emb[1*num_latents:2*num_latents]
    comb_emb[1, 0] = start_emb[2*num_latents:3*num_latents]
    comb_emb[1, 1] = start_emb[3*num_latents:4*num_latents]

    sanity_check = []
    for d in tqdm(range(num_gen_rows + num_gen_cols - 1)):  # diagonal index
        batched_cond_embs = []
        batched_cond_regs = []
        positions = []  # store where to place outputs

        for i in range(max(0, d - num_gen_rows + 1), min(d + 1, num_gen_cols)):
            j = d - i

            if j >= num_gen_rows:
                continue

            if i == 0 and j != 0:
                # Left edge (excluding top-left)
                cond_reg = [0, 1]
                cond_emb = torch.stack((comb_emb[j, i], comb_emb[j, i + 1]), 0)

                positions.append(('left_edge', j, i))
                batched_cond_embs.append(cond_emb)
                batched_cond_regs.append(cond_reg)

            elif j == 0 and i != 0:
                # Top edge (excluding top-left)
                cond_reg = [0, 2]
                cond_emb = torch.stack((comb_emb[j, i], comb_emb[j + 1, i]), 0)

                positions.append(('top_edge', j, i))
                batched_cond_embs.append(cond_emb)
                batched_cond_regs.append(cond_reg)

            elif j != 0 and i != 0:
                # Interior region
                cond_reg = [0, 1, 2]
                cond_emb = torch.stack((comb_emb[j, i], comb_emb[j, i + 1], comb_emb[j + 1, i]), 0)

                positions.append(('interior', j, i))
                batched_cond_embs.append(cond_emb)
                batched_cond_regs.append(cond_reg)

        if not batched_cond_embs:
            continue

        next_embs = nuu.generate_embeddings_cond(
            batch_size=len(batched_cond_embs),
            mask_region=batched_cond_regs,
            mask_embeddings=batched_cond_embs
        )

        # Scatter outputs
        for idx, (region_type, j, i) in enumerate(positions):
            if region_type == 'left_edge':
                comb_emb[j + 1, i]     = next_embs[idx, 2 * num_latents:3 * num_latents]
                comb_emb[j + 1, i + 1] = next_embs[idx, 3 * num_latents:4 * num_latents]
            elif region_type == 'top_edge':
                comb_emb[j,     i + 1] = next_embs[idx, 1 * num_latents:2 * num_latents]
                comb_emb[j + 1, i + 1] = next_embs[idx, 3 * num_latents:4 * num_latents]
            elif region_type == 'interior':
                comb_emb[j + 1, i + 1] = next_embs[idx, 3 * num_latents:4 * num_latents]
            sanity_check.append((i, j))
    print('Embedding generation time took: {}'.format(time.time() - emb_gen_start))

    occ_start = time.time()
    for j in tqdm(range(comb_emb.shape[0])):
        for i in range(comb_emb.shape[1]):
            temp_occ = nuu.decode_embedding(comb_emb[j, i], gen_grid_resolution)
            comb_occ[j*gen_grid_resolution[0]:(j+1)*gen_grid_resolution[0], :temp_occ.shape[1], i*gen_grid_resolution[2]:(i+1)*gen_grid_resolution[2]] = temp_occ.cpu().numpy()
    print('Occ time took: {}'.format(time.time() - occ_start))

    marching_start = time.time()
    verts, faces, _, _ = skimage.measure.marching_cubes(comb_occ, 0.5)
    gap = 2. / (gen_grid_resolution[0] // 2)
    verts *= gap
    verts -= 1
    m = trimesh.Trimesh(verts, faces)
    m.export(os.path.join(output_folder, 'infinite_{}x{}_seed_{}.obj'.format(num_gen_rows, num_gen_cols, gen_seed)))
    np.save(os.path.join(output_folder, 'infinite_{}x{}_seed_{}.npy'.format(num_gen_rows, num_gen_cols, gen_seed)), comb_emb.cpu().numpy().astype(np.float32))
    print('Marching cubes time took: {}'.format(time.time() - marching_start))

if __name__ == '__main__':
    main()
