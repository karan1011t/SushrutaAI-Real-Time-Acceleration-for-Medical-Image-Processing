"""
SUSHRUTA — Dataset Loader Verification Test
Verifies binary mask conversion, tumor pixel preservation, and 1-to-1 image-mask pairing.
"""

import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader

# Add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models.dataset import BraTS2DDataset, create_brats_dataloaders


def test_dataset_loader():
    print("=" * 50, flush=True)
    print("RUNNING DATASET LOADER VERIFICATION TEST", flush=True)
    print("=" * 50, flush=True)

    data_dir = os.path.join(project_root, "data", "processed", "brats")

    # 1. Verify Image-Mask Filename Pairing
    print("\n[TEST 1] Verifying Image-Mask Pairing...", flush=True)
    for split in ["train", "val"]:
        img_dir = os.path.join(data_dir, split, "images")
        mask_dir = os.path.join(data_dir, split, "masks")
        img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(".png")])
        mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith(".png")])

        assert len(img_files) > 0, f"No image files found in {img_dir}"
        assert len(img_files) == len(mask_files), f"Mismatch count: {len(img_files)} images vs {len(mask_files)} masks in {split}"
        assert img_files == mask_files, f"Filename mismatch between images and masks in split '{split}'"
        print(f"  Split '{split}': {len(img_files):,} images 1-to-1 paired with masks. PASSED.")
    
    pairing_passed = True

    # 2. Test Train Loader (with preload=True as used in training)
    print("\n[TEST 2] Verifying Train Dataset & Loader (20+ Slices with preload=True)...", flush=True)
    train_dataset = BraTS2DDataset(data_dir=data_dir, split="train", augment=False, preload=True)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False)

    train_total_tumor_pixels = 0
    train_tumor_slices_count = 0
    train_empty_slices_count = 0
    train_batches_tested = 0
    train_example_tumor_info = {}

    for batch_idx, (imgs, masks) in enumerate(train_loader, 1):
        if train_batches_tested >= 2: # 32 slices tested
            break
        
        batch_size = imgs.size(0)
        mask_unique = [int(v) for v in torch.unique(masks).tolist()]
        mask_min = float(torch.min(masks))
        mask_max = float(torch.max(masks))
        
        slice_sums = masks.view(batch_size, -1).sum(dim=1)
        tumor_slices = int(torch.sum(slice_sums > 0).item())
        empty_slices = int(torch.sum(slice_sums == 0).item())
        total_tumor_px = int(torch.sum(masks > 0.5).item())

        print(f"\n  [Train Batch {batch_idx}] Slices: {batch_size}", flush=True)
        print(f"    Mask shape:               {tuple(masks.shape)}", flush=True)
        print(f"    Mask unique values:       {mask_unique}", flush=True)
        print(f"    Minimum mask value:       {mask_min:.1f}", flush=True)
        print(f"    Maximum mask value:       {mask_max:.1f}", flush=True)
        print(f"    Total tumor pixels:       {total_tumor_px:,}", flush=True)
        print(f"    Tumor-containing slices:  {tumor_slices}", flush=True)
        print(f"    Empty slices:             {empty_slices}", flush=True)

        assert tuple(masks.shape[1:]) == (1, 256, 256), f"Invalid mask shape {masks.shape}"
        assert all(v in [0, 1] for v in mask_unique), f"Non-binary mask values detected: {mask_unique}"

        train_total_tumor_pixels += total_tumor_px
        train_tumor_slices_count += tumor_slices
        train_empty_slices_count += empty_slices
        train_batches_tested += 1

        if tumor_slices > 0 and not train_example_tumor_info:
            for s_i in range(batch_size):
                s_px = int(slice_sums[s_i].item())
                if s_px > 0:
                    train_example_tumor_info = {
                        "unique": [int(v) for v in torch.unique(masks[s_i]).tolist()],
                        "tumor_pixels": s_px
                    }
                    break

    assert train_total_tumor_pixels > 0, "CRITICAL ERROR: All training masks evaluated to ZERO! Tumor pixels were NOT preserved!"
    assert train_tumor_slices_count > 0, "CRITICAL ERROR: No tumor-containing slices found in train batch!"
    print(f"\n  Train Loader Summary: Preserved {train_total_tumor_pixels:,} tumor pixels across {train_tumor_slices_count} tumor slices. PASSED.")
    train_passed = True

    # 3. Test Validation Loader (with preload=True as used in training)
    print("\n[TEST 3] Verifying Validation Dataset & Loader (20+ Slices with preload=True)...", flush=True)
    val_dataset = BraTS2DDataset(data_dir=data_dir, split="val", augment=False, preload=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    val_total_tumor_pixels = 0
    val_tumor_slices_count = 0
    val_empty_slices_count = 0
    val_batches_tested = 0
    val_example_tumor_info = {}

    for batch_idx, (imgs, masks) in enumerate(val_loader, 1):
        if val_batches_tested >= 2: # 32 slices tested
            break

        batch_size = imgs.size(0)
        mask_unique = [int(v) for v in torch.unique(masks).tolist()]
        mask_min = float(torch.min(masks))
        mask_max = float(torch.max(masks))
        
        slice_sums = masks.view(batch_size, -1).sum(dim=1)
        tumor_slices = int(torch.sum(slice_sums > 0).item())
        empty_slices = int(torch.sum(slice_sums == 0).item())
        total_tumor_px = int(torch.sum(masks > 0.5).item())

        print(f"\n  [Val Batch {batch_idx}] Slices: {batch_size}", flush=True)
        print(f"    Mask shape:               {tuple(masks.shape)}", flush=True)
        print(f"    Mask unique values:       {mask_unique}", flush=True)
        print(f"    Minimum mask value:       {mask_min:.1f}", flush=True)
        print(f"    Maximum mask value:       {mask_max:.1f}", flush=True)
        print(f"    Total tumor pixels:       {total_tumor_px:,}", flush=True)
        print(f"    Tumor-containing slices:  {tumor_slices}", flush=True)
        print(f"    Empty slices:             {empty_slices}", flush=True)

        assert tuple(masks.shape[1:]) == (1, 256, 256), f"Invalid mask shape {masks.shape}"
        assert all(v in [0, 1] for v in mask_unique), f"Non-binary mask values detected: {mask_unique}"

        val_total_tumor_pixels += total_tumor_px
        val_tumor_slices_count += tumor_slices
        val_empty_slices_count += empty_slices
        val_batches_tested += 1

        if tumor_slices > 0 and not val_example_tumor_info:
            for s_i in range(batch_size):
                s_px = int(slice_sums[s_i].item())
                if s_px > 0:
                    val_example_tumor_info = {
                        "unique": [int(v) for v in torch.unique(masks[s_i]).tolist()],
                        "tumor_pixels": s_px
                    }
                    break

    assert val_total_tumor_pixels > 0, "CRITICAL ERROR: All validation masks evaluated to ZERO! Tumor pixels were NOT preserved!"
    assert val_tumor_slices_count > 0, "CRITICAL ERROR: No tumor-containing slices found in val batch!"
    print(f"\n  Val Loader Summary: Preserved {val_total_tumor_pixels:,} tumor pixels across {val_tumor_slices_count} tumor slices. PASSED.")
    val_passed = True

    example_info = train_example_tumor_info or val_example_tumor_info

    print("\n" + "=" * 50, flush=True)
    print("SUSHRUTA DATASET LOADER FIX", flush=True)
    print("=" * 50, flush=True)
    print(f"Mask conversion: PASSED", flush=True)
    print(f"Binary mask preservation: PASSED", flush=True)
    print(f"Tumor pixels preserved: PASSED", flush=True)
    print(f"Image-mask pairing: PASSED", flush=True)
    print(f"Train loader: PASSED", flush=True)
    print(f"Validation loader: PASSED\n", flush=True)
    print(f"Example tumor mask:", flush=True)
    print(f"Unique values: {example_info['unique']}", flush=True)
    print(f"Tumor pixels: {example_info['tumor_pixels']}\n", flush=True)
    print(f"READY FOR CLEAN BASELINE RETRAINING: YES", flush=True)
    print("=" * 50 + "\n", flush=True)

if __name__ == "__main__":
    test_dataset_loader()
