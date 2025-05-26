import os
from huggingface_hub import snapshot_download

local_dir = '.'
snapshot_dir = snapshot_download(repo_id="3dlg-hcvc/NuiScene", repo_type='model', local_dir=local_dir, allow_patterns=["pretrained/**"])
