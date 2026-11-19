#!/usr/bin/env bash
# Stage III: Iterative 3D Latent Generation
VAE_CKPT="pretrained_models/vae/med_vae3d.pt"
OUTPUT_DIR="./generated_samples/brats2019"
NUM_VOLUMES=20

python generate_stage3_3d.py \
  --vae_ckpt ${VAE_CKPT} \
  --output_dir ${OUTPUT_DIR} \
  --num_volumes ${NUM_VOLUMES} \
  --batch_size 2 \
  --num_ar_steps 64 \
  --cfg_scale 2.0 \
  --img_size 128

