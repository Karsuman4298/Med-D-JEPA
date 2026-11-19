"""
Stage III: Iterative 3D Latent Generation and Synthesis
Paper: "Beyond Representation Learning: A Systematic Study of Joint-Embedding Predictive Generation for 3D Brain MRI"

Executes Section 3.4:
1. Starts from 100% masked token grid (512 tokens).
2. Runs K = 64 outer autoregressive steps with cosine masking schedule.
3. Feature predictor estimates contextual representation h_i for masked positions.
4. Per-token reverse diffusion (T_diff = 100 steps) denoises chosen continuous patches.
5. Inversely patchifies tokens into 3D continuous latent grid (4 x 16 x 16 x 16).
6. Decodes latents with frozen Stage I VAE into synthetic 128^3 3D MRI volumes.
"""

import argparse
import os
import time
import math
import numpy as np
import torch
import torchvision.utils as vutils
from PIL import Image

from models import med_djepa_base, med_djepa_256
from models.pretrained_vae import get_pretrained_vae


def save_orthogonal_slices(volume: torch.Tensor, save_path: str):
    """
    Saves central Axial, Coronal, and Sagittal slices from a 3D volume (1, H, W, D).
    Matches paper visualizations in Figure 3 & Figure 4.
    """
    vol = volume.squeeze().cpu().numpy()  # (H, W, D)
    H, W, D = vol.shape

    axial_slice = vol[:, :, D // 2]
    coronal_slice = vol[:, W // 2, :]
    sagittal_slice = vol[H // 2, :, :]

    # Concatenate side by side: [Axial | Coronal | Sagittal]
    triptych = np.hstack([axial_slice, coronal_slice, sagittal_slice])
    triptych = np.clip(triptych * 255.0, 0, 255).astype(np.uint8)

    img = Image.fromarray(triptych)
    img.save(save_path)


def main():
    parser = argparse.ArgumentParser("Stage III: Med-D-JEPA 3D Volume Generation")
    parser.add_argument("--ckpt_path", type=str, required=False, default="", help="Path to trained Med-D-JEPA checkpoint")
    parser.add_argument("--vae_ckpt", type=str, default="pretrained_models/vae/med_vae3d.pt", help="Stage I VAE path")
    parser.add_argument("--output_dir", type=str, default="./generated_samples", help="Output directory")
    parser.add_argument("--num_volumes", type=int, default=10, help="Number of volumes to generate")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size per forward pass")
    parser.add_argument("--class_idx", type=int, default=None, help="Class condition (0: LGG/CN, 1: HGG/AD, None: unconditional)")
    parser.add_argument("--num_ar_steps", type=int, default=64, help="Paper optimal K = 64")
    parser.add_argument("--cfg_scale", type=float, default=2.0, help="Paper optimal s_cfg = 2.0")
    parser.add_argument("--img_size", type=int, default=128, choices=[128, 256])
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 1. Load Frozen Stage I 3D VAE
    print(f"[Stage III] Loading frozen Stage I 3D VAE from {args.vae_ckpt}...")
    vae_args = argparse.Namespace(
        vae="MED_VAE_3D",
        local_diffusers_model_root=os.path.dirname(args.vae_ckpt) if os.path.dirname(args.vae_ckpt) else ".",
        use_cached=False,
        evaluate=True,
    )
    vae = get_pretrained_vae(vae_args).to(device).eval()

    # 2. Instantiate Calibrated Med-D-JEPA Model
    print(f"[Stage III] Instantiating Med-D-JEPA (img_size={args.img_size}, K={args.num_ar_steps}, cfg={args.cfg_scale})...")
    if args.img_size == 128:
        model = med_djepa_base(class_num=2 if args.class_idx is not None else 1)
    else:
        model = med_djepa_256(class_num=1)

    if args.ckpt_path and os.path.exists(args.ckpt_path):
        ckpt = torch.load(args.ckpt_path, map_location="cpu")
        state_dict = ckpt.get("model", ckpt.get("model_without_ddp", ckpt))
        model.load_state_dict(state_dict)
        print(f"[Stage III] Loaded Med-D-JEPA checkpoint from {args.ckpt_path}")
    else:
        print("[Stage III] Notice: Running with initialized model weights (for pipeline calibration testing).")

    model = model.to(device).eval()

    # 3. Autoregressive 3D Generation Loop
    num_batches = math.ceil(args.num_volumes / args.batch_size)
    total_generated = 0

    print(f"[Stage III] Generating {args.num_volumes} volumes in {num_batches} batches...")
    start_total = time.time()

    with torch.no_grad():
        for b_idx in range(num_batches):
            cur_bsz = min(args.batch_size, args.num_volumes - total_generated)
            if cur_bsz <= 0:
                break

            labels = None
            if args.class_idx is not None:
                labels = torch.full((cur_bsz,), args.class_idx, dtype=torch.long, device=device)

            step_start = time.time()
            # Step A: Sample completed continuous 3D latents (B, 4, 16, 16, 16)
            sampled_latents = model.sample_tokens(
                bsz=cur_bsz,
                num_iter=args.num_ar_steps,
                cfg_scale=args.cfg_scale,
                cfg_schedule="constant",
                labels=labels,
                progress=True
            )

            # Step B: Decode continuous latents using frozen Stage I VAE
            synthetic_volumes = vae.decode(sampled_latents)
            synthetic_volumes = torch.clamp(synthetic_volumes, 0.0, 1.0)
            step_time = time.time() - step_start

            print(f"Batch {b_idx + 1}/{num_batches} took {step_time:.2f}s ({step_time / cur_bsz:.2f}s/volume)")

            # Save volumes and slice projections
            for i in range(cur_bsz):
                sample_id = total_generated + i
                label_tag = f"_cls{args.class_idx}" if args.class_idx is not None else ""
                vol_path = os.path.join(args.output_dir, f"vol_{sample_id:04d}{label_tag}.npy")
                slice_path = os.path.join(args.output_dir, f"ortho_{sample_id:04d}{label_tag}.png")

                np.save(vol_path, synthetic_volumes[i].cpu().numpy())
                save_orthogonal_slices(synthetic_volumes[i], slice_path)

            total_generated += cur_bsz

    total_time = time.time() - start_total
    print(f"[Stage III Complete] Generated {total_generated} volumes to {args.output_dir} in {total_time:.2f}s.")


if __name__ == "__main__":
    main()

