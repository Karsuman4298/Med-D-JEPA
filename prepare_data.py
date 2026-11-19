"""
Dataset Preprocessing and Preparation Script for Med-D-JEPA
Paper: "Beyond Representation Learning: A Systematic Study of Joint-Embedding Predictive Generation for 3D Brain MRI"

Implements Section 4.1 & Appendix A:
1. BraTS2019:
   - Extract tumor-containing spatial extent from nonzero segmentation mask
   - Multiply FLAIR by binary tumor mask and extract fixed-size crop centered on tumor
   - Resample to 128 x 128 x 128 voxels, normalize intensities to [0, 1]
   - Hold out 25 HGG and 25 LGG patients for test (remaining 234 HGG + 51 LGG for train)
2. OASIS-1:
   - T1-weighted cross-sectional MRI (CDR=0 -> CN, CDR in {0.5, 1, 2} -> AD)
   - Resample whole-brain volume to 128 x 128 x 128 voxels, normalize to [0, 1]
   - Hold out 30 CN and 30 AD patients for test (remaining 356 patients for train)
3. BraTS2020:
   - Whole-brain FLAIR co-registered, 1mm^3 isotropic, skull-stripped
   - Resample to 256 x 256 x 256 voxels, normalize to [0, 1]
   - Hold out 74 patients for test (remaining 295 patients for train)
"""

import argparse
import os
import glob
import random
import numpy as np
import torch

try:
    import nibabel as nib
    import monai.transforms as mt
    MONAI_AVAILABLE = True
except ImportError:
    MONAI_AVAILABLE = False


def normalize_01(arr: np.ndarray) -> np.ndarray:
    """Normalizes voxel intensities to [0, 1]."""
    min_val, max_val = arr.min(), arr.max()
    if max_val > min_val:
        return (arr - min_val) / (max_val - min_val)
    return arr


def process_brats2019_roi(flair_path: str, seg_path: str, target_size: int = 128):
    """
    Extracts tumor-centered ROI multiplied by binary mask, resampled to target_size^3.
    """
    flair_nii = nib.load(flair_path)
    seg_nii = nib.load(seg_path)

    flair = flair_nii.get_fdata().astype(np.float32)
    seg = seg_nii.get_fdata().astype(np.float32)

    # Binary tumor mask (all non-zero tumor labels)
    tumor_mask = (seg > 0).astype(np.float32)

    # Multiply FLAIR image by binary tumor mask
    masked_flair = flair * tumor_mask

    # Bounding box of tumor
    coords = np.argwhere(tumor_mask > 0)
    if len(coords) == 0:
        center = [s // 2 for s in flair.shape]
    else:
        center = coords.mean(axis=0).astype(int)

    # Crop fixed-size cube around center
    half = target_size // 2
    slices = []
    pad_needed = []
    for c, s in zip(center, flair.shape):
        start = c - half
        end = c + half
        p_before = max(0, -start)
        p_after = max(0, end - s)
        start = max(0, start)
        end = min(s, end)
        slices.append(slice(start, end))
        pad_needed.append((p_before, p_after))

    crop = masked_flair[tuple(slices)]
    if any(p[0] > 0 or p[1] > 0 for p in pad_needed):
        crop = np.pad(crop, pad_needed, mode="constant", constant_values=0)

    # Resample to isotropic target_size^3 using MONAI if shape differs
    if crop.shape != (target_size, target_size, target_size):
        resizer = mt.Resize(spatial_size=(target_size, target_size, target_size), mode="trilinear")
        crop = resizer(torch.from_numpy(crop).unsqueeze(0)).squeeze(0).numpy()

    return normalize_01(crop)


def process_whole_brain(image_path: str, target_size: int = 128):
    """
    Resamples whole-brain volume to target_size^3 and normalizes to [0, 1].
    """
    nii = nib.load(image_path)
    vol = nii.get_fdata().astype(np.float32)

    resizer = mt.Resize(spatial_size=(target_size, target_size, target_size), mode="trilinear")
    vol_tensor = torch.from_numpy(vol).unsqueeze(0)
    resampled = resizer(vol_tensor).squeeze(0).numpy()

    return normalize_01(resampled)


def split_and_save(samples: list, train_dir: str, test_dir: str, num_test: int):
    """Splits samples into train and held-out test sets."""
    random.seed(42)
    shuffled = samples.copy()
    random.shuffle(shuffled)

    test_samples = shuffled[:num_test]
    train_samples = shuffled[num_test:]

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    print(f"Saving {len(train_samples)} training samples to {train_dir}")
    for path, data in train_samples:
        np.save(os.path.join(train_dir, path), data)

    print(f"Saving {len(test_samples)} test samples to {test_dir}")
    for path, data in test_samples:
        np.save(os.path.join(test_dir, path), data)


def main():
    parser = argparse.ArgumentParser("Med-D-JEPA Dataset Preprocessing")
    parser.add_argument("--dataset", type=str, required=True, choices=["brats2019", "oasis1", "brats2020"])
    parser.add_argument("--raw_dir", type=str, required=True, help="Path to raw dataset folder")
    parser.add_argument("--output_dir", type=str, required=True, help="Destination folder for processed .npy volumes")
    args = parser.parse_args()

    if not MONAI_AVAILABLE:
        raise ImportError("nibabel and monai are required. Run: pip install nibabel monai")

    print(f"--- Processing {args.dataset.upper()} from {args.raw_dir} ---")

    if args.dataset == "brats2019":
        # BraTS2019: Extract ROIs (128^3). Split: 25 HGG test, 25 LGG test
        for tumor_grade, num_test in [("LGG", 25), ("HGG", 25)]:
            patient_folders = glob.glob(os.path.join(args.raw_dir, f"MICCAI_BraTS_2019_Data_Training/{tumor_grade}/*"))
            processed = []
            print(f"Found {len(patient_folders)} {tumor_grade} patients.")
            for folder in patient_folders:
                pid = os.path.basename(folder)
                flair = glob.glob(os.path.join(folder, "*_flair.nii.gz"))
                seg = glob.glob(os.path.join(folder, "*_seg.nii.gz"))
                if flair and seg:
                    vol = process_brats2019_roi(flair[0], seg[0], target_size=128)
                    processed.append((f"{pid}.npy", vol))

            train_dir = os.path.join(args.output_dir, "train", "0" if tumor_grade == "LGG" else "1")
            test_dir = os.path.join(args.output_dir, "test", "0" if tumor_grade == "LGG" else "1")
            split_and_save(processed, train_dir, test_dir, num_test)

    elif args.dataset == "oasis1":
        # OASIS-1: Resample whole-brain (128^3). Split: 30 CN test, 30 AD test
        # Assumes CSV or folder split with labels
        print("Processing OASIS-1...")
        for group, num_test, label in [("CN", 30, "0"), ("AD", 30, "1")]:
            files = glob.glob(os.path.join(args.raw_dir, group, "*.nii*"))
            processed = []
            for f in files:
                pid = os.path.basename(f).split(".")[0]
                vol = process_whole_brain(f, target_size=128)
                processed.append((f"{pid}.npy", vol))
            train_dir = os.path.join(args.output_dir, "train", label)
            test_dir = os.path.join(args.output_dir, "test", label)
            split_and_save(processed, train_dir, test_dir, num_test)

    elif args.dataset == "brats2020":
        # BraTS2020: Whole-brain 256^3. Split: 74 test, 295 train (unconditional)
        patient_folders = glob.glob(os.path.join(args.raw_dir, "*"))
        processed = []
        for folder in patient_folders:
            pid = os.path.basename(folder)
            flair = glob.glob(os.path.join(folder, "*_flair.nii.gz"))
            if flair:
                vol = process_whole_brain(flair[0], target_size=256)
                processed.append((f"{pid}.npy", vol))
        train_dir = os.path.join(args.output_dir, "train", "0")
        test_dir = os.path.join(args.output_dir, "test", "0")
        split_and_save(processed, train_dir, test_dir, 74)

    print(f"Dataset preparation complete for {args.dataset}! Output saved to {args.output_dir}")


if __name__ == "__main__":
    main()
