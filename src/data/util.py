import torch
import numpy as np

class AxisScaling(object):
    def __init__(self, interval=(0.75, 1.25), jitter=True):
        assert isinstance(interval, tuple)
        self.interval = interval
        self.jitter = jitter
        
    def __call__(self, surface, point):
        if self.jitter:
            surface += 0.005 * torch.randn_like(surface)

        return surface, point

def rotate_y(coords, angle):
    rot_matrix = np.array([[np.cos(np.deg2rad(angle)),  0, np.sin(np.deg2rad(angle))],
                           [0,                          1,                         0],
                           [-np.sin(np.deg2rad(angle)), 0, np.cos(np.deg2rad(angle))]]).astype(np.float16)
    return coords @ rot_matrix.T

def rot_pcd(pcd, rotation_angle):
    center = torch.tensor([0.5, 0, 0.5]).unsqueeze(0)

    center_pcd = pcd - center
    rotated_pcd = rotate_y(center_pcd, -rotation_angle)
    rotated_pcd = rotated_pcd + center
    return rotated_pcd

def rot_presampled_pcd_only(chosen_angle, large_chunk_pcds):
    rotate_times = int(chosen_angle // 90)

    temp_chunk_pcds = np.zeros_like(large_chunk_pcds)
    for _ in range(rotate_times):
        temp_chunk_pcds[0] = rotate_y(large_chunk_pcds[2], 90)
        temp_chunk_pcds[1] = rotate_y(large_chunk_pcds[0], 90)
        temp_chunk_pcds[2] = rotate_y(large_chunk_pcds[3], 90)
        temp_chunk_pcds[3] = rotate_y(large_chunk_pcds[1], 90)
        large_chunk_pcds = temp_chunk_pcds.copy()

    return large_chunk_pcds

def rot_presampled(chosen_angle, large_chunk_pcds, large_chunk_occ_queries, large_chunk_occs, large_chunk_occ_queries_2, large_chunk_occs_2):
    rotate_times = int(chosen_angle // 90)

    temp_chunk_pcds = np.zeros_like(large_chunk_pcds)
    temp_chunk_occs = np.zeros_like(large_chunk_occs)
    temp_chunk_occs_2 = np.zeros_like(large_chunk_occs_2)
    temp_chunk_occ_queries = np.zeros_like(large_chunk_occ_queries)
    temp_chunk_occ_queries_2 = np.zeros_like(large_chunk_occ_queries_2)
    for _ in range(rotate_times):
        temp_chunk_pcds[0] = rotate_y(large_chunk_pcds[2], 90)
        temp_chunk_pcds[1] = rotate_y(large_chunk_pcds[0], 90)
        temp_chunk_pcds[2] = rotate_y(large_chunk_pcds[3], 90)
        temp_chunk_pcds[3] = rotate_y(large_chunk_pcds[1], 90)
        large_chunk_pcds = temp_chunk_pcds.copy()

        temp_chunk_occ_queries[0] = rotate_y(large_chunk_occ_queries[2], 90)
        temp_chunk_occ_queries[1] = rotate_y(large_chunk_occ_queries[0], 90)
        temp_chunk_occ_queries[2] = rotate_y(large_chunk_occ_queries[3], 90)
        temp_chunk_occ_queries[3] = rotate_y(large_chunk_occ_queries[1], 90)
        large_chunk_occ_queries = temp_chunk_occ_queries.copy()

        temp_chunk_occ_queries_2[0] = rotate_y(large_chunk_occ_queries_2[2], 90)
        temp_chunk_occ_queries_2[1] = rotate_y(large_chunk_occ_queries_2[0], 90)
        temp_chunk_occ_queries_2[2] = rotate_y(large_chunk_occ_queries_2[3], 90)
        temp_chunk_occ_queries_2[3] = rotate_y(large_chunk_occ_queries_2[1], 90)
        large_chunk_occ_queries_2 = temp_chunk_occ_queries_2.copy()

        temp_chunk_occs[0] = large_chunk_occs[2]
        temp_chunk_occs[1] = large_chunk_occs[0]
        temp_chunk_occs[2] = large_chunk_occs[3]
        temp_chunk_occs[3] = large_chunk_occs[1]
        large_chunk_occs = temp_chunk_occs.copy()

        temp_chunk_occs_2[0] = large_chunk_occs_2[2]
        temp_chunk_occs_2[1] = large_chunk_occs_2[0]
        temp_chunk_occs_2[2] = large_chunk_occs_2[3]
        temp_chunk_occs_2[3] = large_chunk_occs_2[1]
        large_chunk_occs_2 = temp_chunk_occs_2.copy()

    return large_chunk_pcds, large_chunk_occ_queries, large_chunk_occs, large_chunk_occ_queries_2, large_chunk_occs_2
