#!/usr/bin/env bash
# Stage I Training: 3D KL-VAE-GAN on Brain MRI
DATA_PATH="./data/brats2019_128"
OUTPUT_DIR="./exps/stage1_vae_brats"

python train_stage1_vae.py \
  --data_path ${DATA_PATH} \
  --output_dir ${OUTPUT_DIR} \
  --img_size 128 \
  --batch_size 4 \
  --epochs 5000 \
  --lr 1e-4 \
  --min_lr 1e-6 \
  --warmup_steps 1000 \
  --save_freq 500

