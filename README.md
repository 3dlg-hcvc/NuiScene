# 🪡Nui(縫い)Scene

## NuiScene: Exploring Efficient Generation of Unbounded Outdoor Scenes

**ICCV 2025**

[Han-Hung Lee](https://hanhung.github.io/), [Qinghong Han](https://sulley.cc/), [Angel X. Chang](https://angelxuanchang.github.io/)

**[Paper](https://arxiv.org/abs/2503.16375)** | **[Project Page](https://3dlg-hcvc.github.io/NuiScene/)** | **[Dataset Page](https://3dlg-hcvc.github.io/NuiScene43-Dataset/)** | **[Data Code](https://github.com/3dlg-hcvc/NuiScene43-Dataset)**

<img src="docs/static/images/teaser.png" alt="teaser" />

## Environment Setup

```
# create and activate the conda environment
conda create -n sasu python=3.10
conda activate sasu

# install PyTorch
conda install pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 pytorch-cuda=12.1 -c pytorch -c nvidia

# install Python libraries
pip install -r requirements.txt

# install diffusers
pip install --upgrade diffusers[torch]

# install torch-cluster
conda install pytorch-cluster -c pyg

# install positional encoding
pip install positional-encodings[pytorch]
```

## Inference

1. Download pretrained model weights.
```
python download/download_weights.py
```
2. Run unbounded generation.
```
# ckpt_path chooses the pretrained diffusion model
# +num_gen_rows, +num_gen_cols controls the number of (rows+1, cols+1) chunks to generate.
# +quad_chunk_size controls the voxel size for the quad chunk. Can set lower for different LoD, faster decoding and smaller obj file.

# Single scene model
python utils/infinite_gen.py ckpt_path=pretrained/diff_vecset16_occ2048_bs20x2_single_l24_bs192/training/last.ckpt +quad_chunk_size=100 +num_gen_rows=20 +num_gen_cols=20

# 4 scene model
python utils/infinite_gen.py ckpt_path=pretrained/4_scene_diff/training/last.ckpt +quad_chunk_size=100 +num_gen_rows=15 +num_gen_cols=45

# 13 scene model
python utils/infinite_gen.py ckpt_path=pretrained/13_scene_diff_up512l3preproj/training/last.ckpt +quad_chunk_size=100 +num_gen_rows=15 +num_gen_cols=45
```
3. Visualize result in ***output/\*/\*.obj***.

> You can also see some pre-generated results from these two models [here](https://huggingface.co/3dlg-hcvc/NuiScene/tree/main/examples).

## Training

1. Download the single scene h5.
```
python download/download_1_scene.py
python download/cat_1_scene.py
```
2. Train the vae model.
```
# We used 2 A6000 GPUs in the arxiv paper.
python train_vae.py data.train.dataloader.batch_size=20 devices=2 experiment_name=1_scene_vae
```
3. Cache the embeddings for diffusion training (save the point cloud embeddings for quad chunks only as we don't need occupancy for diffusion training).
```
python utils/cache_embeddings.py ckpt_path=output/NuiSceneChunk/1_scene_vae/training/last.ckpt +cache_batch_size=24
```
4. Train the diffusion model.
```
python train_diff.py vae_ckpt_path=output/NuiSceneChunk/1_scene_vae/training/last.ckpt experiment_name=1_scene_diff data.train.dataloader.batch_size=192
```
5. Run unbounded generation (see the inference section above).

## Notes from Paper Supplemental

1. Sped up unbounded generation. In the main paper, we used the raster scan order for generating chunks for benchmarking. However, each i-th chunk in a row only depends on the (i-1)-th chunk in the previous row. So we can batch the operation across the anti-diagonal of the scene. This results in a large speed up and what is currently used in the utils/infinite_gen.py.
2. For the 13 scene model we added a pixel unshuffle operation to upsample vector sets (similar to decovolution for the triplane). This results in better reconstruction results without increasing the number of vector sets for diffusion learning. Please see the supplemental of the paper for more details.

## TODO

- [ ] Add evaluation scripts.

## Citation

```
@InProceedings{Lee_2025_ICCV,
    author    = {Lee, Han-Hung and Han, Qinghong and Chang, Angel X.},
    title     = {NuiScene: Exploring Efficient Generation of Unbounded Outdoor Scenes},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2025},
    pages     = {26509-26518}
}
```

## Acknowledgements

Thanks to [Michelangelo](https://github.com/NeuralCarver/Michelangelo) and [CraftsMan3D](https://github.com/wyysf-98/CraftsMan3D), whose implementations form the foundation of our VAE and diffusion models. We also thank 3DShape2VecSet (https://github.com/1zb/3DShape2VecSet) for their contributions to the vector set framework, and Objaverse (https://objaverse.allenai.org/) for providing a vast 3D dataset that made this work possible.

This work was funded by a CIFAR AI Chair, an NSERC Discovery grant, and a CFI/BCKDF JELF grant. We thank Jiayi Liu, and Xingguang Yan for helpful suggestions on improving the paper.
