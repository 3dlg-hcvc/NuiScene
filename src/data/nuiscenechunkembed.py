import os
import json
import h5py
import torch
import random
import numpy as np
from torch.utils.data import Dataset

class NuiSceneChunkEmbed(Dataset):
    def __init__(self, data_cfg, split):
        super().__init__()
        self.split = split
        self.data_cfg = data_cfg

        self.num_chunks = self.data_cfg.metadata.num_chunks

        h5_file = os.path.join(self.data_cfg.dataset_path, '{}.h5'.format(self.data_cfg.metadata.vae_ckpt_path.split('/')[-3]))
        self.embed_h5 = h5py.File(h5_file, "r")
    
    def __len__(self):
        return len(self.embed_h5['embeds'])

    def __getitem__(self, _idx):
        selected_id = _idx

        embeds = self.embed_h5['embeds'][selected_id]
        chosen_angle = random.choice([0, 1, 2, 3])
        embeds = embeds[chosen_angle]

        return {
            "embeds": embeds,
            "chosen_angle": chosen_angle,
        }
