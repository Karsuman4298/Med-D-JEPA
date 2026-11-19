# Med-D-JEPA: Joint-Embedding Predictive Generation for 3D Brain MRI

PyTorch implementation of **Med-D-JEPA** based on the paper:  
> **Beyond Representation Learning: A Systematic Study of Joint-Embedding Predictive Generation for 3D Brain MRI**  
> *Meng Zhou, Wenhao You, Yuxing Chen, Yueying Tian*  
> arXiv:2608.28787

Med-D-JEPA adapts Joint-Embedding Predictive Architectures (JEPAs) and Denoising JEPA (D-JEPA) for 3D volumetric medical imaging (Brain MRI). By operating on continuous 3D latent tokens produced by a 3D KL-regularized adversarial autoencoder, Med-D-JEPA combines representation-level alignment, per-token diffusion, and iterative next-set-of-token sampling to synthesize high-fidelity, anatomically consistent 3D MRI volumes.

---

## Architecture Overview

Med-D-JEPA operates in three stages:

```
[3D MRI Volume] (128³ or 256³)
       │
       ▼  Stage I: 3D KL-VAE-GAN (8× downsampling)
[Continuous 3D Latent Grid] (4 × 16³ or 4 × 32³)
       │
       ▼  3D Patchification (p = 2×2×2)
[Continuous 3D Latent Tokens] (N = 512 or 4,096 tokens, dp = 32)
       │
       ├───► Stage II: Med-D-JEPA Joint Training
       │        ├── Context Encoder (ϕ) on visible tokens
       │        ├── Target Encoder (ϕ̄) via EMA (α = 0.999)
       │        ├── Feature Predictor (ψ) predicting representations h_i
       │        ├── Projection MLP g(·) with L1 Loss (L_pred)
       │        └── Denoising MLP (ε_θ) with Per-Token Diffusion Loss (L_diff)
       │
       ▼  Stage III: Iterative 3D Latent Generation
       │        ├── Start with 100% masked 3D grid
       │        ├── Progressive K = 64 AR steps with Cosine Masking Schedule
       │        ├── Per-token reverse DDPM diffusion (T_diff = 100) conditioned on h_i
       │        ├── Classifier-Free Guidance (s_cfg = 2.0)
       │        └── Unpatchify tokens back into 3D continuous latent grid
       │
       ▼  Frozen Stage I VAE Decoder
[Synthesized 3D MRI Volume] (128³ or 256³)
```

---

## Dataset Specifications & Train-Test Splits

The paper uses three public brain MRI benchmarks with strict 80/20 train-test splits (the test cohorts are completely excluded from Stage I, Stage II, and all downstream training):

| Dataset | Modality & Scope | Resolution | Split | Train Count | Test Count (Held Out) | Classes |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **BraTS2019** | FLAIR (Tumor ROI) | $128^3$ | 80/20 | 285 (234 HGG, 51 LGG) | **50** (25 HGG, 25 LGG) | 0: LGG, 1: HGG |
| **OASIS-1** | T1 (Whole-Brain) | $128^3$ | 80/20 | 356 patients | **60** (30 CN, 30 AD) | 0: CN (CDR=0), 1: AD (CDR>0) |
| **BraTS2020** | FLAIR (Whole-Brain) | $256^3$ | 80/20 | 295 patients | **74** patients | Unconditional |

### Directory Arrangement

Arrange your preprocessed datasets in the following structure:

```
data/
├── brats2019_128/
│   ├── train/
│   │   ├── 0/             # 51 LGG patient volumes (.npy)
│   │   └── 1/             # 234 HGG patient volumes (.npy)
│   └── test/              # Held-out test set
│       ├── 0/             # 25 LGG patient volumes (.npy)
│       └── 1/             # 25 HGG patient volumes (.npy)
│
├── oasis1_128/
│   ├── train/
│   │   ├── 0/             # Cognitively Normal (CN, CDR = 0) (.npy)
│   │   └── 1/             # Cognitively Impaired (AD, CDR ∈ {0.5, 1, 2}) (.npy)
│   └── test/              # Held-out test set
│       ├── 0/             # 30 CN held-out volumes (.npy)
│       └── 1/             # 30 AD held-out volumes (.npy)
│
└── brats2020_256/
    ├── train/
    │   └── 0/             # 295 whole-brain FLAIR volumes (.npy)
    └── test/              # Held-out test set
        └── 0/             # 74 held-out volumes (.npy)
```

---

## Detailed Method Formulation

### Stage I: Continuous 3D Latent Compression (3D KL-VAE-GAN)
Trained on normalized 3D volumes $x \in [0, 1]^{1 \times H \times W \times D}$ using combined objectives:
$$\min_{E, G} \max_{D_{\text{adv}}} \left( \lambda_1 \mathcal{L}_{\text{recon}} + \lambda_2 \mathcal{L}_{\text{lpips}} + \lambda_3 \mathcal{L}_{\text{grad}} + \lambda_4 \mathcal{L}_{\text{reg}} + \lambda_5 \mathcal{L}_{\text{adv}} \right)$$

- $\mathcal{L}_{\text{recon}}$: Voxel-level L1 reconstruction loss ($\lambda_1 = 2.0$)
- $\mathcal{L}_{\text{lpips}}$: Slice-wise 2.5D AlexNet perceptual loss ($\lambda_2 = 0.5$)
- $\mathcal{L}_{\text{grad}}$: 3D spatial gradient loss across $Z, Y, X$ axes ($\lambda_3 = 1.0$)
- $\mathcal{L}_{\text{reg}}$: KL divergence with standard normal prior ($\lambda_4 = 1 \times 10^{-7}$)  
  *KL Warmup Schedule*: $\lambda_4$ is held at 0 for step $1 \dots 1000$ and linearly warmed up over steps $1000 \dots 2000$.
- $\mathcal{L}_{\text{adv}}$: Least-squares PatchGAN adversarial loss with 3D discriminator ($\lambda_5 = 1.0$)

### Stage II: Med-D-JEPA Joint Training
Operates on non-overlapping 3D patches $X = \{x_i\}_{i=1}^N \in \mathbb{R}^{d_p}$.
- **Context Encoder ($\phi$)**: Vision Transformer encoding visible tokens.
- **Target Encoder ($\bar{\phi}$)**: EMA copy ($\alpha = 0.999$) encoding the complete token sequence.
- **Feature Predictor ($\psi$)**: Estimates context representations $h_i$ at masked locations $i \in \mathcal{M}$.
- **L1 Prediction Loss**: $\mathcal{L}_{\text{pred}} = \frac{1}{|\mathcal{M}|} \sum_{i \in \mathcal{M}} \|g(h_i) - y_i\|_1$
- **Per-Token Diffusion Loss**: $\mathcal{L}_{\text{diff}} = \frac{1}{|\mathcal{M}|} \sum_{i \in \mathcal{M}} \mathbb{E}_{t, \epsilon_i} \left[ \|\epsilon_i - \epsilon_\theta(x_i^t \mid t, h_i)\|_2^2 \right]$
- **Total Loss**: $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{pred}} + \mathcal{L}_{\text{diff}}$

### Stage III: Iterative 3D Latent Generation
- Starts with a 100% masked grid of $N$ tokens.
- Over $K = 64$ autoregressive iterations, uses a cosine masking schedule:
  $$\gamma(k/K) = \cos\left(\frac{\pi}{2} \frac{k}{K}\right)$$
- Denoises newly unmasked tokens via reverse diffusion ($T_{\text{diff}} = 100$ steps):
  $$x_i^{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( x_i^t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(x_i^t \mid t, h_i) \right) + \sigma_t \delta_t$$
- Applies Classifier-Free Guidance ($s_{\text{cfg}} = 2.0$):
  $$h_i = h_i^{\text{uncond}} + s_{\text{cfg}} \cdot (h_i^{\text{cond}} - h_i^{\text{uncond}})$$
- Assembled tokens are rearranged via inverse patchification and decoded into 3D volumes by the frozen Stage I VAE decoder.

---

## Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Karsuman4298/Med-D-JEPA.git
   cd Med-D-JEPA
   ```

2. **Create environment and install dependencies**:
   ```bash
   conda create -n med-djepa python=3.10 -y
   conda activate med-djepa
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   pip install monai[all] timm scipy tqdm nibabel
   ```

---

## Reproduction Guide: Step-by-Step

### Step 1: Automated Data Preprocessing
Use [`prepare_data.py`](prepare_data.py) to automatically process raw NIfTI files into the exact train/test splits, crop tumor ROIs, resample to $128^3$ or $256^3$, and normalize intensities to $[0, 1]$:

```bash
# For BraTS 2019:
python prepare_data.py \
  --dataset brats2019 \
  --raw_dir /path/to/MICCAI_BraTS_2019_Data_Training \
  --output_dir ./data/brats2019_128

# For OASIS-1:
python prepare_data.py \
  --dataset oasis1 \
  --raw_dir /path/to/OASIS1_raw \
  --output_dir ./data/oasis1_128

# For BraTS 2020:
python prepare_data.py \
  --dataset brats2020 \
  --raw_dir /path/to/BraTS20_Training_Data \
  --output_dir ./data/brats2020_256
```

---

### Step 2: Stage I (3D KL-VAE-GAN Training)
Train the volume compression autoencoder on the training set:
```bash
bash scripts/train_med_stage1.sh
```
Or directly:
```bash
python train_stage1_vae.py \
  --data_path ./data/brats2019_128/train \
  --output_dir ./exps/stage1_vae \
  --img_size 128 \
  --batch_size 4 \
  --epochs 5000 \
  --lr 1e-4 \
  --min_lr 1e-6 \
  --warmup_steps 1000
```
*Outputs: Saved 3D VAE checkpoint to `pretrained_models/vae/med_vae3d.pt`.*

---

### Step 3: Extract & Cache 3D Latents
Encode volumes to continuous 3D latents $(4 \times 16 \times 16 \times 16)$ for rapid Stage II training:
```bash
bash scripts/cache_med_latents.sh
```
Or directly:
```bash
python main_cache_3d.py \
  --data_path ./data/brats2019_128/train \
  --cached_path ./cached_latents/brats2019 \
  --vae MED_VAE_3D \
  --batch_size 4
```

---

### Step 4: Stage II (Train Med-D-JEPA)
Train the continuous joint-embedding model on the cached latents:
```bash
torchrun --nproc_per_node=4 main_djepa.py \
  --model med_djepa_base \
  --img_size 128 \
  --vae MED_VAE_3D \
  --patch_size 2 \
  --spatial_dims 3 \
  --batch_size 8 \
  --epochs 3000 \
  --blr 3.0e-4 \
  --min_lr 5.0e-6 \
  --weight_decay 0.01 \
  --diffusion_weight 1.0 \
  --jepa_weight 1.0 \
  --num_classes 2 \
  --data_path ./cached_latents/brats2019 \
  --use_cached
```

---

### Step 5: Stage III (Iterative 3D Latent Generation)
Synthesize 300 volumes per class with $K = 64$ AR iterations and $s_{\text{cfg}} = 2.0$:
```bash
# Generate 300 synthetic LGG volumes (class 0)
python generate_stage3_3d.py \
  --vae_ckpt pretrained_models/vae/med_vae3d.pt \
  --ckpt_path exps/med_djepa/checkpoint-last.pth \
  --output_dir ./generated_samples/brats2019/class0 \
  --num_volumes 300 \
  --class_idx 0 \
  --num_ar_steps 64 \
  --cfg_scale 2.0 \
  --img_size 128

# Generate 300 synthetic HGG volumes (class 1)
python generate_stage3_3d.py \
  --vae_ckpt pretrained_models/vae/med_vae3d.pt \
  --ckpt_path exps/med_djepa/checkpoint-last.pth \
  --output_dir ./generated_samples/brats2019/class1 \
  --num_volumes 300 \
  --class_idx 1 \
  --num_ar_steps 64 \
  --cfg_scale 2.0 \
  --img_size 128
```

---

### Step 6: Quantitative Metric Evaluation (Tables 1, 2, 9, 10)
Evaluate 3D Maximum Mean Discrepancy (MMD) and Multi-Slice SSIM against the real held-out test cohort:
```bash
python evaluate_metrics_3d.py \
  --real_dir ./data/brats2019_128/test/1 \
  --fake_dir ./generated_samples/brats2019/class1
```

---

## Paper Ablation Checklist

To reproduce the ablation studies from the paper:
- **AR Steps Ablation (Table 6 & Table 11)**:
  Run `generate_stage3_3d.py` with `--num_ar_steps` set to `8`, `16`, `32`, `64`, `256`, and `512`.
- **JEPA Prediction Loss Ablation (Table 7)**:
  Train Stage II with `--jepa_weight 0.0` (w/o $\mathcal{L}_{\text{pred}}$) vs `--jepa_weight 1.0` (Med-D-JEPA).
- **CFG Scale Ablation (Table 12)**:
  Run `generate_stage3_3d.py` with `--cfg_scale` set to `2.0`, `2.5`, `3.0`, `3.5`, and `4.0`.

---

## Directory Structure

```
├── models/
│   ├── vae_3d.py              # Stage I: 3D AutoencoderKL + PatchDiscriminator + 3D Gradient Loss
│   ├── pretrained_vae.py      # VAE loader supporting 2D and 3D (MED_VAE_3D)
│   ├── djepa.py               # Calibrated D-JEPA with 3D patchify/unpatchify & med_djepa presets
│   ├── denoising_mlp.py       # Compact AdaLN residual MLP for per-token reverse diffusion
│   ├── jepaloss.py            # Prediction loss L_pred & 2-layer projection MLP
│   └── diffloss/              # Gaussian diffusion loss & DDPM sampling
├── train_stage1_vae.py        # Stage I 3D VAE-GAN training pipeline
├── main_cache_3d.py           # Stage I 3D latent caching script
├── generate_stage3_3d.py      # Stage III iterative 3D generation & slice visualization script
├── prepare_data.py            # Dataset preprocessing (BraTS2019, OASIS-1, BraTS2020)
├── evaluate_metrics_3d.py     # Quantitative evaluation (3D MMD, MS-SSIM)
├── main_djepa.py              # Stage II training script
├── engine/                    # Training and evaluation loops
└── scripts/
    ├── train_med_stage1.sh    # Runner for Stage I training
    ├── cache_med_latents.sh   # Runner for latent extraction
    └── generate_med_stage3.sh # Runner for Stage III generation
```

---

## Citation

If you use this work or findings from the paper, please cite:

```bibtex
@article{zhou2026meddjepa,
  title={Beyond Representation Learning: A Systematic Study of Joint-Embedding Predictive Generation for 3D Brain MRI},
  author={Zhou, Meng and You, Wenhao and Chen, Yuxing and Tian, Yueying},
  journal={arXiv preprint arXiv:2608.28787},
  year={2026}
}

@inproceedings{chen2025djepa,
  title={Denoising with a Joint-Embedding Predictive Architecture},
  author={Chen, Dengsheng and Hu, Jie and Wei, Xiaoming and Wu, Enhua},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2025}
}
```
