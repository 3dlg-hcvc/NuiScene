import json
import h5py
import torch
import random
import numpy as np
from torch.utils.data import Dataset

from src.data.util import rot_presampled, AxisScaling

class NuiSceneChunk(Dataset):
    def __init__(self, data_cfg, split):
        super().__init__()
        self.split = split
        self.data_cfg = data_cfg
        self.axis_scaling = self.data_cfg.metadata.axis_scaling
        self.axis_scaling_min = getattr(self.data_cfg.metadata, "axis_scaling_min", 0.75)
        self.axis_scaling_max = getattr(self.data_cfg.metadata, "axis_scaling_max", 1.25)
        self.axis_scaling_fn = AxisScaling((self.axis_scaling_min, self.axis_scaling_max), True)

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

        inside_pos = self.pcd_occ_h5['pos_surface_pos'][selected_id]
        inside_occ = np.ones((inside_pos.shape[0], inside_pos.shape[1]), dtype=np.int8)
        num_actual_inside = self.pcd_occ_h5['num_actual_pos'][selected_id]

        outside_pos = self.pcd_occ_h5['neg_surface_pos'][selected_id]
        outside_occ = np.zeros((outside_pos.shape[0], outside_pos.shape[1]), dtype=np.int8)
        num_actual_outside = self.pcd_occ_h5['num_actual_neg'][selected_id]

        vol_surface_pos = np.concatenate((inside_pos, outside_pos), 1)
        vol_surface_occ = np.concatenate((inside_occ, outside_occ), 1)

        near_surface_pos = self.pcd_occ_h5['near_surface_pos'][selected_id]
        near_surface_occ = self.pcd_occ_h5['near_surface_occ'][selected_id]
        num_actual_new = self.pcd_occ_h5['num_actual_near'][selected_id]

        # Random augment positions by rotation
        if self.random_rotation:
            rotation_angles = [0, 90, 180, 270]
            chosen_angle = random.choice(rotation_angles)
        else:
            chosen_angle = 0

        pcd, near_surface_pos, near_surface_occ, vol_surface_pos, vol_surface_occ = rot_presampled(chosen_angle, pcd, 
                                                                                                   near_surface_pos, near_surface_occ,
                                                                                                   vol_surface_pos, vol_surface_occ)

        # Convert data into torch tensors
        pcd = torch.from_numpy(pcd)
        near_surface_pos = torch.from_numpy(near_surface_pos)
        near_surface_occ = torch.from_numpy(near_surface_occ)
        vol_surface_pos = torch.from_numpy(vol_surface_pos)
        vol_surface_occ = torch.from_numpy(vol_surface_occ)

        large_chunk_pcds = []
        large_chunk_occs = []
        large_chunk_occ_queries = []
        for i in range(self.num_chunks):
            filtered_pcd = pcd[i][:num_actual_pcd[i]]
            ind = np.random.choice(filtered_pcd.shape[0], self.pc_size, replace=(self.pc_size > filtered_pcd.shape[0]))
            filtered_pcd = filtered_pcd[ind]
            large_chunk_pcds.append(filtered_pcd)

            ind_inside = np.random.choice(num_actual_inside[i], self.num_occ_samples // 2, replace=(self.num_occ_samples // 2 > num_actual_inside[i]))
            ind_outside = np.random.choice(num_actual_outside[i], self.num_occ_samples // 2, replace=(self.num_occ_samples // 2 > num_actual_outside[i])) + inside_pos.shape[1]
            ind = np.concatenate((ind_inside, ind_outside), 0)
            filtered_vol = vol_surface_pos[i][ind]
            filtered_vol_occ = vol_surface_occ[i][ind]

            ind = np.random.choice(num_actual_new[i], self.num_occ_samples, replace=(self.num_occ_samples > num_actual_new[i]))
            filtered_near = near_surface_pos[i][ind]
            filtered_near_occ = near_surface_occ[i][ind]

            train_occ = torch.cat([filtered_vol_occ, filtered_near_occ], dim=0)
            train_occ_query = torch.cat([filtered_vol, filtered_near], dim=0)
            large_chunk_occs.append(train_occ)
            large_chunk_occ_queries.append(train_occ_query)

        large_chunk_pcds = torch.stack(large_chunk_pcds).to(torch.float16)
        large_chunk_occs = torch.stack(large_chunk_occs).to(torch.float16)
        large_chunk_occ_queries = torch.stack(large_chunk_occ_queries).to(torch.float16)

        if self.axis_scaling:
            large_chunk_pcds, large_chunk_occ_queries = self.axis_scaling_fn(large_chunk_pcds, large_chunk_occ_queries)
        large_chunk_heights = large_chunk_pcds[..., 1].max(1)[0]

        if self.id_to_model_file is not None:
            scene_id = self.id_to_model_file[str(selected_id)]
        else:
            scene_id = -1

        if 'sample_x' in self.pcd_occ_h5.keys():
            sample_x = self.pcd_occ_h5['sample_x'][selected_id]
            sample_y = self.pcd_occ_h5['sample_y'][selected_id]
        else:
            sample_x = -1
            sample_y = -1

        return {
            "selected_id": selected_id,
            "pcd": large_chunk_pcds,
            "occ": large_chunk_occs,
            "chosen_angle": chosen_angle,
            "height": large_chunk_heights,
            "occ_query": large_chunk_occ_queries,
            "scene_id": scene_id,
            "sample_x": sample_x,
            "sample_y": sample_y,
        }
