"""
Stage I Training Script: 3D KL-VAE-GAN
Paper: "Beyond Representation Learning: A Systematic Study of Joint-Embedding Predictive Generation for 3D Brain MRI"

Trains the 3D volume compression network using:
- 3D AutoencoderKL from MONAI (8x downsampling, 4 latent channels)
- 3D PatchDiscriminator
- L1 reconstruction loss (weight = 2.0)
- 2.5D Perceptual loss via AlexNet (weight = 0.5)
- 3D image gradient loss (weight = 1.0)
- Adversarial least-squares loss (weight = 1.0)
- KL regularization (weight = 1e-7, warmed up over steps 1000-2000)
"""

import argparse
import math
import os
import time
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np

from models.vae_3d import Stage1MedVAEGAN


class NpyMriDataset(Dataset):
    """
    Dataset loader for preprocessed 3D MRI volumes (.npy or .pt).
    Volumes must be normalized to [0, 1] with shape (1, H, W, D).
    """
    def __init__(self, data_root: str):
        self.files = []
        for root, _, filenames in os.walk(data_root):
            for f in filenames:
                if f.endswith('.npy') or f.endswith('.pt'):
                    self.files.append(os.path.join(root, f))
        self.files.sort()
        if len(self.files) == 0:
            print(f"Warning: No .npy or .pt files found in {data_root}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        if path.endswith('.npy'):
            arr = np.load(path).astype(np.float32)
            tensor = torch.from_numpy(arr)
        else:
            tensor = torch.load(path, map_location="cpu")
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)  # (1, H, W, D)
        return tensor


def adjust_learning_rate(optimizer, step, warmup_steps, total_steps, base_lr, min_lr):
    """Linear warmup + cosine decay."""
    if step < warmup_steps:
        lr = base_lr * (step + 1) / warmup_steps
    else:
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        lr = min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


def main():
    parser = argparse.ArgumentParser("Stage I: 3D KL-VAE-GAN Training")
    parser.add_argument("--data_path", type=str, default="./data/brats2019_128", help="Path to preprocessed 3D MRIs")
    parser.add_argument("--output_dir", type=str, default="./output_stage1", help="Output directory")
    parser.add_argument("--img_size", type=int, default=128, help="3D volume spatial size (128 or 256)")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size (paper uses 4 or 5)")
    parser.add_argument("--epochs", type=int, default=5000, help="Training epochs (5000 for 128^3, 8000 for 256^3)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate")
    parser.add_argument("--min_lr", type=float, default=1e-6, help="Minimum learning rate")
    parser.add_argument("--warmup_steps", type=int, default=1000, help="Learning rate warmup steps")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_freq", type=int, default=500, help="Checkpoint frequency in epochs")
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 1. Initialize Stage 1 Model
    model = Stage1MedVAEGAN(in_channels=1, latent_channels=4).to(device)

    # 2. Separate optimizers for Generator (VAE) and Discriminator
    opt_g = torch.optim.Adam(model.vae.parameters(), lr=args.lr, betas=(0.5, 0.9))
    opt_d = torch.optim.Adam(model.discriminator.parameters(), lr=args.lr, betas=(0.5, 0.9))

    # 3. Dataloader
    dataset = NpyMriDataset(args.data_path)
    if len(dataset) == 0:
        print(f"Dataset at {args.data_path} is empty. Please check data directory.")
        return

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    total_steps = args.epochs * len(loader)
    global_step = 0
    print(f"Starting Stage I training: {args.epochs} epochs, {len(loader)} batches/epoch, total {total_steps} steps.")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss_g, epoch_loss_d = 0.0, 0.0
        start_time = time.time()

        for batch in loader:
            global_step += 1
            x = batch.to(device)

            # Cosine decay schedule
            lr = adjust_learning_rate(
                opt_g, global_step, args.warmup_steps, total_steps, args.lr, args.min_lr
            )
            adjust_learning_rate(
                opt_d, global_step, args.warmup_steps, total_steps, args.lr, args.min_lr
            )

            # Execute Generator and Discriminator steps
            losses = model.train_step(x, opt_g, opt_d, global_step=global_step)
            epoch_loss_g += losses["loss_g"]
            epoch_loss_d += losses["loss_d"]

        avg_loss_g = epoch_loss_g / len(loader)
        avg_loss_d = epoch_loss_d / len(loader)
        elapsed = time.time() - start_time

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch [{epoch}/{args.epochs}] | "
                f"Loss G: {avg_loss_g:.4f} | Loss D: {avg_loss_d:.4f} | "
                f"L_recon: {losses['l_recon']:.4f} | L_grad: {losses['l_grad']:.4f} | "
                f"L_kl: {losses['l_kl']:.4f} (scale={losses['lambda_kl']:.2e}) | "
                f"lr: {lr:.2e} | time: {elapsed:.2f}s"
            )

        # Save checkpoints
        if epoch % args.save_freq == 0 or epoch == args.epochs:
            ckpt_path = os.path.join(args.output_dir, f"stage1_vae_epoch{epoch}.pt")
            torch.save({
                "epoch": epoch,
                "global_step": global_step,
                "vae": model.vae.state_dict(),
                "discriminator": model.discriminator.state_dict(),
            }, ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")

    # Save final model directly into pretrained_models/vae/med_vae3d.pt
    final_vae_path = "pretrained_models/vae/med_vae3d.pt"
    os.makedirs(os.path.dirname(final_vae_path), exist_ok=True)
    torch.save(model.vae.state_dict(), final_vae_path)
    print(f"[Stage I Complete] Saved final 3D VAE weights to {final_vae_path}")


if __name__ == "__main__":
    main()

