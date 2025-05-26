import os
from huggingface_hub import snapshot_download

local_dir = 'download'
snapshot_dir = snapshot_download(repo_id="3dlg-hcvc/NuiScene43", repo_type='dataset', local_dir=local_dir, allow_patterns=["sampled_h5/**1_scene_qcs100_split_**"])
