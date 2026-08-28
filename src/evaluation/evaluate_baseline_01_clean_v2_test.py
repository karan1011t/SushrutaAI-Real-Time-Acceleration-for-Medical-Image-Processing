"""
SUSHRUTA — BASELINE-01-CLEAN-V2: Held-Out Test Set Evaluation
FLAIR-Only 2D U-Net Segmentation Engine
"""

import os
import sys
import time
import json
import csv
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models.unet import UNet2D, count_parameters
from src.models.dataset import BraTS2DDataset
from src.evaluation.segmentation_metrics import ConfusionMatrixAccumulator


def run_baseline_01_clean_v2_test_evaluation():
    print("=" * 60, flush=True)
    print("SUSHRUTA — BASELINE-01-CLEAN-V2 HELD-OUT TEST EVALUATION", flush=True)
    print("=" * 60, flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Device: {device} ({gpu_name})", flush=True)

    # -------------------------------------------------------------------------
    # STEP 1 — CHECKPOINT VERIFICATION
    # -------------------------------------------------------------------------
    print("\n" + "=" * 50, flush=True)
    print("STEP 1: CHECKPOINT VERIFICATION", flush=True)
    print("=" * 50, flush=True)

    ckpt_path = os.path.join(project_root, "results", "models", "baseline_01_clean_v2_best.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"CRITICAL ERROR: Required checkpoint does not exist: {ckpt_path}")

    print(f"Loading checkpoint from: {ckpt_path}", flush=True)
    checkpoint = torch.load(ckpt_path, map_location=device)

    # Required metadata verification
    ckpt_epoch = checkpoint.get("epoch", None)
    val_tumor_dice = checkpoint.get("val_tumor_dice", None)

    print(f"Checkpoint Epoch:              {ckpt_epoch}", flush=True)
    print(f"Stored Val Tumor Dice:        {val_tumor_dice:.6f}" if val_tumor_dice is not None else "N/A", flush=True)

    if ckpt_epoch != 4:
        raise RuntimeError(f"CHECKPOINT VERIFICATION FAILED: Expected epoch 4, got {ckpt_epoch}")
    if val_tumor_dice is None or abs(val_tumor_dice - 0.9960773991525825) > 1e-4:
        raise RuntimeError(f"CHECKPOINT VERIFICATION FAILED: Expected Val Tumor Dice ~0.9961, got {val_tumor_dice}")

    # Instantiate model and verify parameter count & architecture compatibility
    model = UNet2D(in_channels=1, out_channels=1, base_channels=32).to(device)
    num_params = count_parameters(model)
    print(f"Model Architecture:            UNet2D(in_channels=1, out_channels=1, base_channels=32)", flush=True)
    print(f"Model Parameter Count:        {num_params:,}", flush=True)

    if num_params != 4317825:
        raise RuntimeError(f"CHECKPOINT VERIFICATION FAILED: Expected 4,317,825 parameters, got {num_params}")

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print("CHECKPOINT VERIFICATION PASSED SUCCESSFULLY.", flush=True)

    # -------------------------------------------------------------------------
    # STEP 2 — TEST DATA VERIFICATION
    # -------------------------------------------------------------------------
    print("\n" + "=" * 50, flush=True)
    print("STEP 2: TEST DATA VERIFICATION", flush=True)
    print("=" * 50, flush=True)

    data_dir = os.path.join(project_root, "data", "processed", "brats")
    test_img_dir = os.path.join(data_dir, "test", "images")
    test_mask_dir = os.path.join(data_dir, "test", "masks")

    test_img_files = sorted([f for f in os.listdir(test_img_dir) if f.endswith(".png")])
    test_mask_files = sorted([f for f in os.listdir(test_mask_dir) if f.endswith(".png")])

    print(f"Test Image Count:             {len(test_img_files):,}", flush=True)
    print(f"Test Mask Count:              {len(test_mask_files):,}", flush=True)

    if len(test_img_files) != 3076 or len(test_mask_files) != 3076:
        raise RuntimeError(f"TEST DATA VERIFICATION FAILED: Expected 3,076 images/masks, got {len(test_img_files)} images / {len(test_mask_files)} masks")

    if test_img_files != test_mask_files:
        raise RuntimeError("TEST DATA VERIFICATION FAILED: Image and mask filenames do not match 1-to-1")

    print("1-to-1 Image-Mask Filename Match: PASSED", flush=True)

    # Patient ID Overlap Verification
    test_patient_ids = set([f.split("_slice")[0] for f in test_img_files])
    train_img_files = os.listdir(os.path.join(data_dir, "train", "images"))
    val_img_files = os.listdir(os.path.join(data_dir, "val", "images"))
    train_patient_ids = set([f.split("_slice")[0] for f in train_img_files])
    val_patient_ids = set([f.split("_slice")[0] for f in val_img_files])

    train_overlap = test_patient_ids.intersection(train_patient_ids)
    val_overlap = test_patient_ids.intersection(val_patient_ids)

    print(f"Test Patient Count:            {len(test_patient_ids)}", flush=True)
    print(f"Train/Test Patient Overlap:    {len(train_overlap)}", flush=True)
    print(f"Val/Test Patient Overlap:      {len(val_overlap)}", flush=True)

    if len(train_overlap) > 0 or len(val_overlap) > 0:
        raise RuntimeError(f"TEST DATA VERIFICATION FAILED: Patient overlap detected! Train overlap: {train_overlap}, Val overlap: {val_overlap}")

    print("Zero Patient Overlap Verification: PASSED", flush=True)

    # Test Dataset Loader with preload=True
    test_dataset = BraTS2DDataset(data_dir=data_dir, split="test", augment=False, preload=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)

    # Verify mask values and dimensions across full dataset (vectorized)
    if test_dataset.images_tensor.shape != (3076, 1, 256, 256) or test_dataset.masks_tensor.shape != (3076, 1, 256, 256):
        raise RuntimeError(f"TEST DATA VERIFICATION FAILED: Invalid dataset shapes: images {test_dataset.images_tensor.shape}, masks {test_dataset.masks_tensor.shape}")

    all_mask_unique = [round(float(u), 4) for u in torch.unique(test_dataset.masks_tensor).tolist()]
    print(f"Unique Mask Values Across Test Set: {all_mask_unique}", flush=True)
    if not all(u in (0.0, 1.0) for u in all_mask_unique):
        raise RuntimeError(f"TEST DATA VERIFICATION FAILED: Non-binary mask values detected: {all_mask_unique}")

    print("Mask Binary {0,1} & Spatial Dimensions (256x256): PASSED", flush=True)
    print("TEST DATA VERIFICATION COMPLETED SUCCESSFULLY.", flush=True)

    # -------------------------------------------------------------------------
    # STEP 3 & 4 — TEST INFERENCE & METRIC ACCUMULATION
    # -------------------------------------------------------------------------
    print("\n" + "=" * 50, flush=True)
    print("STEP 3 & 4: TEST INFERENCE & METRIC ACCUMULATION", flush=True)
    print("=" * 50, flush=True)

    accumulator = ConfusionMatrixAccumulator()
    threshold = 0.5

    with torch.no_grad():
        for batch_idx, (imgs, masks) in enumerate(test_loader, 1):
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            if device.type == "cuda":
                with torch.amp.autocast("cuda"):
                    logits = model(imgs)
            else:
                logits = model(imgs)

            probs = torch.sigmoid(logits)
            accumulator.update(probs, masks, threshold=threshold)

            if batch_idx % 25 == 0 or batch_idx == len(test_loader):
                print(f"Evaluated Test Batch {batch_idx}/{len(test_loader)} ({min(batch_idx * 32, len(test_dataset))}/{len(test_dataset)} slices)", flush=True)

    test_metrics = accumulator.get_metrics()

    # -------------------------------------------------------------------------
    # STEP 5 — PERFORMANCE BENCHMARKING
    # -------------------------------------------------------------------------
    print("\n" + "=" * 50, flush=True)
    print("STEP 5: PERFORMANCE BENCHMARKING (INFERENCE LATENCY & FPS)", flush=True)
    print("=" * 50, flush=True)

    # Benchmark pure model inference latency on GPU (excluding data loading)
    single_img_sample = test_dataset[0][0].unsqueeze(0).to(device)
    
    # Warmup runs
    with torch.no_grad():
        for _ in range(50):
            if device.type == "cuda":
                with torch.amp.autocast("cuda"):
                    _ = torch.sigmoid(model(single_img_sample))
            else:
                _ = torch.sigmoid(model(single_img_sample))
    
    if device.type == "cuda":
        torch.cuda.synchronize()

    num_trials = 500
    if device.type == "cuda":
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        with torch.no_grad():
            for _ in range(num_trials):
                with torch.amp.autocast("cuda"):
                    _ = torch.sigmoid(model(single_img_sample))
        end_event.record()
        torch.cuda.synchronize()
        total_time_ms = start_event.elapsed_time(end_event)
        avg_latency_ms = total_time_ms / num_trials
        gpu_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    else:
        t_start = time.perf_counter()
        with torch.no_grad():
            for _ in range(num_trials):
                _ = torch.sigmoid(model(single_img_sample))
        t_end = time.perf_counter()
        avg_latency_ms = ((t_end - t_start) / num_trials) * 1000.0
        gpu_mem_mb = 0.0

    avg_latency_sec = avg_latency_ms / 1000.0
    fps = 1.0 / avg_latency_sec if avg_latency_sec > 0 else 0.0

    print(f"Average Inference Latency:   {avg_latency_ms:.2f} ms/slice", flush=True)
    print(f"Approximate FPS:            {fps:.1f} FPS", flush=True)
    print(f"GPU Used:                   {gpu_name}", flush=True)
    print(f"Peak GPU Memory Allocated:  {gpu_mem_mb:.2f} MB", flush=True)

    # -------------------------------------------------------------------------
    # STEP 6 — VISUAL VALIDATION (QUALITATIVE PREDICTION OVERLAYS)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 50, flush=True)
    print("STEP 6: GENERATING REPRESENTATIVE TEST-SET PREDICTION FIGURES", flush=True)
    print("=" * 50, flush=True)

    test_filenames = test_dataset.image_filenames
    sample_indices = []
    seen_patients = set()

    for idx in range(len(test_dataset)):
        fname = test_filenames[idx]
        patient_id = fname.split("_slice")[0]
        mask_sum = float(torch.sum(test_dataset.masks_tensor[idx] > 0.5).item())
        if mask_sum > 100 and patient_id not in seen_patients:
            sample_indices.append(idx)
            seen_patients.add(patient_id)
            if len(sample_indices) >= 12:
                break

    fig, axes = plt.subplots(12, 4, figsize=(16, 40))
    fig.suptitle("Baseline-01-CLEAN-V2: Held-Out Test Set Predictions (FLAIR 2D U-Net)", fontsize=16, fontweight="bold", y=0.995)

    headers = ["FLAIR MRI", "Ground-truth Tumor Mask", "Predicted Tumor Mask", "FLAIR + Prediction Overlay"]
    for col_idx, text in enumerate(headers):
        axes[0, col_idx].set_title(text, fontsize=12, fontweight="bold")

    with torch.no_grad():
        for row_idx, sample_idx in enumerate(sample_indices[:12]):
            img_tensor, mask_tensor = test_dataset[sample_idx]
            fname = test_filenames[sample_idx]
            patient_id = fname.split("_slice")[0]

            img_input = img_tensor.unsqueeze(0).to(device)
            if device.type == "cuda":
                with torch.amp.autocast("cuda"):
                    logits = model(img_input)
            else:
                logits = model(img_input)

            prob = torch.sigmoid(logits).squeeze().cpu().numpy()
            pred_bin = (prob >= threshold).astype(np.float32)

            flair_np = img_tensor.squeeze().numpy()
            gt_np = mask_tensor.squeeze().numpy()

            # Green overlay for prediction on FLAIR grayscale image
            pred_overlay = np.stack([flair_np, flair_np + pred_bin * 0.5, flair_np], axis=-1)
            pred_overlay = np.clip(pred_overlay, 0.0, 1.0)

            axes[row_idx, 0].imshow(flair_np, cmap="gray")
            axes[row_idx, 0].set_ylabel(f"{patient_id}", fontsize=10, fontweight="bold")
            axes[row_idx, 1].imshow(gt_np, cmap="gray")
            axes[row_idx, 2].imshow(pred_bin, cmap="gray")
            axes[row_idx, 3].imshow(pred_overlay)

            for c in range(4):
                axes[row_idx, c].set_xticks([])
                axes[row_idx, c].set_yticks([])

    plt.tight_layout()
    fig_save_path = os.path.join(project_root, "results", "figures", "baseline_01_clean_v2_test_predictions.png")
    plt.savefig(fig_save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved representative prediction figure to: {fig_save_path}", flush=True)

    # -------------------------------------------------------------------------
    # STEP 7 — SAVE METRICS & REPORT
    # -------------------------------------------------------------------------
    print("\n" + "=" * 50, flush=True)
    print("STEP 7: SAVING TEST EVALUATION RESULTS", flush=True)
    print("=" * 50, flush=True)

    metrics_dir = os.path.join(project_root, "results", "metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    test_results_dict = {
        "milestone": "BASELINE-01-CLEAN-V2 — HELD-OUT TEST EVALUATION",
        "checkpoint_path": ckpt_path,
        "best_epoch": ckpt_epoch,
        "validation_tumor_dice_stored": val_tumor_dice,
        "model_architecture": "UNet2D(in_channels=1, out_channels=1, base_channels=32)",
        "total_parameters": num_params,
        "test_dataset": {
            "test_patients": len(test_patient_ids),
            "test_slices": len(test_dataset),
            "tumor_slices": test_metrics["num_tumor_slices"],
            "empty_slices": test_metrics["num_empty_slices"]
        },
        "global_pixel_metrics": {
            "dice": test_metrics["val_global_dice"],
            "iou": test_metrics["val_global_iou"],
            "precision": test_metrics["val_precision"],
            "recall": test_metrics["val_recall"],
            "f1": test_metrics["val_f1"]
        },
        "tumor_only_slice_metrics": {
            "dice": test_metrics["val_tumor_dice"],
            "iou": test_metrics["val_tumor_iou"],
            "precision": test_metrics["val_tumor_precision"],
            "recall": test_metrics["val_tumor_recall"],
            "f1": test_metrics["val_tumor_f1"]
        },
        "performance": {
            "inference_latency_ms": round(avg_latency_ms, 2),
            "fps": round(fps, 1),
            "gpu_name": gpu_name,
            "gpu_memory_allocated_mb": round(gpu_mem_mb, 2)
        }
    }

    json_path = os.path.join(metrics_dir, "baseline_01_clean_v2_test_metrics.json")
    with open(json_path, "w") as f:
        json.dump(test_results_dict, f, indent=2)
    print(f"Saved test metrics JSON to: {json_path}", flush=True)

    csv_path = os.path.join(metrics_dir, "baseline_01_clean_v2_test_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric_category", "metric_name", "value"])
        writer.writerow(["checkpoint", "best_epoch", ckpt_epoch])
        writer.writerow(["checkpoint", "val_tumor_dice_stored", val_tumor_dice])
        writer.writerow(["test_data", "test_patients", len(test_patient_ids)])
        writer.writerow(["test_data", "test_slices", len(test_dataset)])
        writer.writerow(["test_data", "tumor_slices", test_metrics["num_tumor_slices"]])
        writer.writerow(["test_data", "empty_slices", test_metrics["num_empty_slices"]])
        writer.writerow(["global_pixel", "dice", test_metrics["val_global_dice"]])
        writer.writerow(["global_pixel", "iou", test_metrics["val_global_iou"]])
        writer.writerow(["global_pixel", "precision", test_metrics["val_precision"]])
        writer.writerow(["global_pixel", "recall", test_metrics["val_recall"]])
        writer.writerow(["global_pixel", "f1", test_metrics["val_f1"]])
        writer.writerow(["tumor_only_slice", "dice", test_metrics["val_tumor_dice"]])
        writer.writerow(["tumor_only_slice", "iou", test_metrics["val_tumor_iou"]])
        writer.writerow(["tumor_only_slice", "precision", test_metrics["val_tumor_precision"]])
        writer.writerow(["tumor_only_slice", "recall", test_metrics["val_tumor_recall"]])
        writer.writerow(["tumor_only_slice", "f1", test_metrics["val_tumor_f1"]])
        writer.writerow(["performance", "inference_latency_ms", round(avg_latency_ms, 2)])
        writer.writerow(["performance", "fps", round(fps, 1)])
        writer.writerow(["performance", "gpu_name", gpu_name])
    print(f"Saved test metrics CSV to:  {csv_path}", flush=True)

    report_text = f"""==================================================
SUSHRUTA — BASELINE-01-CLEAN-V2
HELD-OUT TEST EVALUATION
==================================================

CHECKPOINT
Best Epoch: {ckpt_epoch}
Validation Tumor Dice: {val_tumor_dice:.4f}

TEST DATA
Test Patients: {len(test_patient_ids)}
Test Slices: {len(test_dataset)}
Tumor Slices: {test_metrics['num_tumor_slices']}
Non-Tumor Slices: {test_metrics['num_empty_slices']}

GLOBAL PIXEL METRICS
Dice: {test_metrics['val_global_dice']:.4f}
IoU: {test_metrics['val_global_iou']:.4f}
Precision: {test_metrics['val_precision']:.4f}
Recall: {test_metrics['val_recall']:.4f}
F1: {test_metrics['val_f1']:.4f}

TUMOR-ONLY SLICE METRICS
Dice: {test_metrics['val_tumor_dice']:.4f}
IoU: {test_metrics['val_tumor_iou']:.4f}
Precision: {test_metrics['val_tumor_precision']:.4f}
Recall: {test_metrics['val_tumor_recall']:.4f}
F1: {test_metrics['val_tumor_f1']:.4f}

INFERENCE
Latency: {avg_latency_ms:.2f} ms/slice
FPS: {fps:.1f}
GPU: {gpu_name}

==================================================
"""
    report_path = os.path.join(metrics_dir, "baseline_01_clean_v2_test_report.txt")
    with open(report_path, "w") as f:
        f.write(report_text)
    print(f"Saved evaluation report to: {report_path}", flush=True)

    # -------------------------------------------------------------------------
    # STEP 8 — FINAL OUTPUT
    # -------------------------------------------------------------------------
    print("\n" + report_text, flush=True)


if __name__ == "__main__":
    run_baseline_01_clean_v2_test_evaluation()
