import os
import subprocess

command = "cat download/sampled_h5/1_scene_qcs100_split_* > dataset/data/1_scene_qcs100.h5"
result = subprocess.run(command, shell=True, check=True)
