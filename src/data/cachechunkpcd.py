import json
import h5py
import torch
import random
import numpy as np
from torch.utils.data import Dataset

from src.data.util import rot_presampled_pcd_only, AxisScaling

class CacheChunkPcd(Dataset):
    def __init__(self, data_cfg, split):
        super().__init__()
        self.split = split
        self.data_cfg = data_cfg
        self.axis_scaling = self.data_cfg.metadata.axis_scaling
        self.axis_scaling_min = getattr(self.data_cfg.metadata, "axis_scaling_min", 0.75)
        self.axis_scaling_max = getattr(self.data_cfg.metadata, "axis_scaling_max", 1.25)
        self.axis_scaling_fn = AxisScaling((self.axis_scaling_min, self.axis_scaling_max), True)

        assert self.axis_scaling == False

        self.random_rotation = True
        self.orig_pc_size = self.data_cfg.metadata.pc_size
        self.num_chunks = self.data_cfg.metadata.num_chunks
        self.num_occ_samples = self.data_cfg.metadata.num_occ_samples
        self.replica_pc_size = self.data_cfg.metadata.replica_pc_size
        self.pc_size = self.data_cfg.metadata.pc_size * self.replica_pc_size

        self.ids = []
        with open(self.data_cfg.metadata.split_file, "r") as f:
            for line in f:
                if line.strip():
                    self.ids.append(int(line.strip()))
        if getattr(self.data_cfg.metadata, "id_to_model_file", None) == None:
            self.id_to_model_file = None
        else:
            with open(self.data_cfg.metadata.id_to_model_file, 'r') as file:
                self.id_to_model_file = json.load(file)
        self.pcd_occ_h5 = h5py.File(self.data_cfg.metadata.h5_file, "r")
    
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, _idx):
        selected_id = self.ids[_idx]

        # Load data from h5, limit to number of actual points
        pcd = self.pcd_occ_h5['pcd'][selected_id]
        num_actual_pcd = self.pcd_occ_h5['num_actual_pcd'][selected_id]

        rot_pcds = []
        for chosen_angle in [0, 90, 180, 270]:
            rot_pcd = rot_presampled_pcd_only(chosen_angle, pcd)
            
            # Convert data into torch tensors
            rot_pcd = torch.from_numpy(rot_pcd)

            large_chunk_pcds = []
            for i in range(self.num_chunks):
                filtered_pcd = rot_pcd[i][:num_actual_pcd[i]]
                ind = np.random.choice(filtered_pcd.shape[0], self.pc_size, replace=(self.pc_size > filtered_pcd.shape[0]))
                filtered_pcd = filtered_pcd[ind]
                large_chunk_pcds.append(filtered_pcd)
            large_chunk_pcds = torch.stack(large_chunk_pcds).to(torch.float16)
            rot_pcds.append(large_chunk_pcds)
        rot_pcds = np.stack(rot_pcds, axis=0)

        if self.id_to_model_file is not None:
            scene_id = self.id_to_model_file[str(selected_id)]
        else:
            scene_id = -1

        return {
            "pcd": rot_pcds,
            "scene_id": scene_id,
        }
