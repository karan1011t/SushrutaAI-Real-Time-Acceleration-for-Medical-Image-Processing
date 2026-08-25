"""
Validation Sanity Test for UNet Model Initialization & Metric Evaluation.
Verifies that an untrained, randomly initialized model evaluating on the validation split
does NOT produce artificially inflated scores or fake 1.0 Dice metrics.
"""

import sys
import os
import torch
from torch.utils.data import DataLoader

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models.unet import UNet2D
from src.models.dataset import BraTS2DDataset
from src.evaluation.segmentation_metrics import ConfusionMatrixAccumulator


def run_validation_sanity_test():
    print("=" * 60, flush=True)
    print("RUNNING VALIDATION SANITY TEST (UNTRAINED RANDOM MODEL)", flush=True)
    print("=" * 60, flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    # 1. Instantiate untrained model with seed=42
    torch.manual_seed(42)
    model = UNet2D(in_channels=1, out_channels=1, base_channels=32).to(device)
    model.eval()

    # 2. Load validation dataset (preload=True for instant RAM evaluation)
    val_dataset = BraTS2DDataset(data_dir=os.path.join(project_root, "data", "processed", "brats"), split="val", preload=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

    print(f"Validation dataset size: {len(val_dataset):,} slices ({len(val_loader)} batches)", flush=True)

    # 3. Evaluate random model on validation set
    accumulator = ConfusionMatrixAccumulator()

    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs = imgs.to(device)
            masks = masks.to(device)

            if device.type == "cuda":
                with torch.amp.autocast("cuda"):
                    logits = model(imgs)
            else:
                logits = model(imgs)

            probs = torch.sigmoid(logits)
            accumulator.update(probs, masks, threshold=0.5)

    metrics = accumulator.get_metrics()

    print("\n--- SANITY TEST METRICS ON UNTRAINED MODEL ---", flush=True)
    print(f"Val Global Dice: {metrics['val_global_dice']:.4f}", flush=True)
    print(f"Val Global IoU:  {metrics['val_global_iou']:.4f}", flush=True)
    print(f"Val Precision:   {metrics['val_precision']:.4f}", flush=True)
    print(f"Val Recall:      {metrics['val_recall']:.4f}", flush=True)
    print(f"Val Tumor Dice:  {metrics['val_tumor_dice']:.4f}", flush=True)
    print(f"Val Tumor IoU:   {metrics['val_tumor_iou']:.4f}", flush=True)
    print(f"Tumor Slices:    {metrics['num_tumor_slices']}", flush=True)
    print(f"Empty Slices:    {metrics['num_empty_slices']}", flush=True)

    # 4. Verify Sanity Assertions
    print("\nVERIFYING SANITY ASSERTIONS...", flush=True)
    
    # Empty slice count check
    assert metrics['num_empty_slices'] == 740, f"Expected 740 empty slices, got {metrics['num_empty_slices']}"
    assert metrics['num_tumor_slices'] == 2378, f"Expected 2378 tumor slices, got {metrics['num_tumor_slices']}"
    print("  Slice Count Breakdown: PASSED (740 empty, 2,378 tumor)", flush=True)

    # Tumor-only Dice check (must be near zero for untrained model)
    assert metrics['val_tumor_dice'] < 0.05, f"Untrained model val_tumor_dice must be < 0.05, got {metrics['val_tumor_dice']}"
    print("  Val Tumor Dice Near Zero Check: PASSED", flush=True)

    # Global Dice check
    assert metrics['val_global_dice'] < 0.05, f"Untrained model val_global_dice must be < 0.05, got {metrics['val_global_dice']}"
    print("  Val Global Dice Near Zero Check: PASSED", flush=True)

    print("\n" + "=" * 60, flush=True)
    print("VALIDATION SANITY TEST PASSED SUCCESSFULLY!", flush=True)
    print("=" * 60, flush=True)
    return True


if __name__ == "__main__":
    run_validation_sanity_test()
