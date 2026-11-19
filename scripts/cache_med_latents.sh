#!/usr/bin/env bash
# Cache Stage I 3D VAE Latents for Med-D-JEPA Training
DATA_PATH="./data/brats2019_128"
CACHED_PATH="./cached_latents/brats2019"
VAE_CKPT="pretrained_models/vae/med_vae3d.pt"

python main_cache_3d.py \
  --data_path ${DATA_PATH} \
  --cached_path ${CACHED_PATH} \
  --vae MED_VAE_3D \
  --local_diffusers_model_root "." \
  --batch_size 4

