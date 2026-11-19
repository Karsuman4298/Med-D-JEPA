"""
3D Latent Caching Script for Med-D-JEPA
Encodes 3D MRI volumes into 3D continuous latents (4 x 16 x 16 x 16) using
the trained Stage I 3D AutoencoderKL and saves to disk for Stage II training.
"""

import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from models.pretrained_vae import get_pretrained_vae


class MriFolderDataset(Dataset):
    """
    Loads 3D MRI scans and associated integer class labels (e.g. 0: LGG/CN, 1: HGG/AD).
    Accepts directory structured as:
      data_path/
        0/  (e.g., LGG or CN)
          sample_001.npy
        1/  (e.g., HGG or AD)
          sample_002.npy
      or flat folder with all samples for unconditional generation.
    """
    def __init__(self, root_dir: str):
        self.samples = []
        subdirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        subdirs.sort()

        if len(subdirs) > 0 and all(d.isdigit() for d in subdirs):
            for class_idx_str in subdirs:
                class_idx = int(class_idx_str)
                folder = os.path.join(root_dir, class_idx_str)
                for f in os.listdir(folder):
                    if f.endswith('.npy') or f.endswith('.pt'):
                        self.samples.append((os.path.join(folder, f), class_idx, f))
        else:
            for f in os.listdir(root_dir):
                if f.endswith('.npy') or f.endswith('.pt'):
                    self.samples.append((os.path.join(root_dir, f), 0, f))

        self.samples.sort(key=lambda x: x[0])
        print(f"Loaded {len(self.samples)} volumes from {root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, filename = self.samples[idx]
        if path.endswith('.npy'):
            arr = np.load(path).astype(np.float32)
            tensor = torch.from_numpy(arr)
        else:
            tensor = torch.load(path, map_location="cpu")
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)  # (1, H, W, D)
        return tensor, label, filename


def main():
    parser = argparse.ArgumentParser("Cache 3D VAE Latents for Med-D-JEPA")
    parser.add_argument("--data_path", type=str, default="./data/brats2019_128", help="Path to 3D MRIs")
    parser.add_argument("--cached_path", type=str, default="./cached_latents/brats2019", help="Destination folder")
    parser.add_argument("--vae", type=str, default="MED_VAE_3D", choices=["MED_VAE_3D"])
    parser.add_argument("--local_diffusers_model_root", type=str, default="pretrained_models")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--use_cached", action="store_true", default=False)
    args = parser.parse_args()

    os.makedirs(args.cached_path, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load Stage I 3D VAE
    vae = get_pretrained_vae(args)
    vae = vae.to(device).eval()

    dataset = MriFolderDataset(args.data_path)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    print(f"Caching latents to {args.cached_path}...")
    with torch.no_grad():
        for volumes, labels, filenames in tqdm(loader):
            volumes = volumes.to(device)
            # Encode volume to posterior parameters
            # AutoencoderKL.encode returns (z_mu, z_sigma)
            z_mu, z_sigma = vae.encode(volumes)

            for i in range(volumes.size(0)):
                out_name = os.path.splitext(filenames[i])[0] + ".npz"
                out_file = os.path.join(args.cached_path, out_name)

                # Store z_mu as continuous latent representation and z_sigma if needed
                np.savez_compressed(
                    out_file,
                    latent=z_mu[i].cpu().numpy(),
                    label=labels[i].item()
                )

    print(f"[Done] Cached {len(dataset)} 3D latents successfully.")


if __name__ == "__main__":
    main()

