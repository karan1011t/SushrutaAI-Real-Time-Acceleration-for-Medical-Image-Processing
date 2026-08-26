"""
SUSHRUTA — Baseline-01-CLEAN-V2: 5-Epoch Controlled GPU Training Run
FLAIR-Only 2D U-Net Segmentation Engine
"""

import os
import sys
import time
import json
import csv
import random
from typing import Dict, Any, List

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
from src.models.train_unet import set_seed, CombinedDiceBCELoss
from src.evaluation.segmentation_metrics import ConfusionMatrixAccumulator, compute_binary_metrics_single_image


def run_baseline_01_clean_v2_5epoch():
    # 1. Hardware Verification
    print("=" * 50, flush=True)
    print("HARDWARE & ENVIRONMENT VERIFICATION", flush=True)
    print("=" * 50, flush=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. GPU is required for training.")

    gpu_name = torch.cuda.get_device_name(0)
    device = torch.device("cuda")
    print(f"CUDA Available: True", flush=True)
    print(f"GPU Name:       {gpu_name}", flush=True)
    print("=" * 50 + "\n", flush=True)

    set_seed(42)

    # Output directories
    results_models_dir = os.path.join(project_root, "results", "models")
    results_metrics_dir = os.path.join(project_root, "results", "metrics")
    results_figures_dir = os.path.join(project_root, "results", "figures")

    os.makedirs(results_models_dir, exist_ok=True)
    os.makedirs(results_metrics_dir, exist_ok=True)
    os.makedirs(results_figures_dir, exist_ok=True)

    # 2. Data Loaders (Train & Val splits only; preload=True)
    data_dir = os.path.join(project_root, "data", "processed", "brats")
    batch_size = 32

    train_dataset = BraTS2DDataset(data_dir=data_dir, split="train", augment=False, preload=True)
    val_dataset = BraTS2DDataset(data_dir=data_dir, split="val", augment=False, preload=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    print(f"Train samples: {len(train_dataset):,} slices ({len(train_loader)} batches)", flush=True)
    print(f"Val samples:   {len(val_dataset):,} slices ({len(val_loader)} batches)", flush=True)

    # 3. IMPORTANT — BEFORE EPOCH 1: Batch Verification
    print("\n" + "=" * 50, flush=True)
    print("PRE-EPOCH 1 BATCH VERIFICATION", flush=True)
    print("=" * 50, flush=True)

    train_imgs_sample, train_masks_sample = next(iter(train_loader))
    val_imgs_sample, val_masks_sample = next(iter(val_loader))

    train_mask_vals = [int(v) for v in torch.unique(train_masks_sample).tolist()]
    val_mask_vals = [int(v) for v in torch.unique(val_masks_sample).tolist()]

    train_tumor_slices = int((train_masks_sample.view(train_masks_sample.size(0), -1).sum(dim=1) > 0).sum().item())
    val_tumor_slices = int((val_masks_sample.view(val_masks_sample.size(0), -1).sum(dim=1) > 0).sum().item())

    train_total_tumor_px = int((train_masks_sample > 0.5).sum().item())
    val_total_tumor_px = int((val_masks_sample > 0.5).sum().item())

    print("Training batch:", flush=True)
    print(f"  Image shape:                   {tuple(train_imgs_sample.shape)}", flush=True)
    print(f"  Mask shape:                    {tuple(train_masks_sample.shape)}", flush=True)
    print(f"  Mask unique values:            {train_mask_vals}", flush=True)
    print(f"  Number of tumor slices:        {train_tumor_slices}/{train_imgs_sample.size(0)}", flush=True)
    print(f"  Total tumor pixels:            {train_total_tumor_px:,}\n", flush=True)

    print("Validation batch:", flush=True)
    print(f"  Image shape:                   {tuple(val_imgs_sample.shape)}", flush=True)
    print(f"  Mask shape:                    {tuple(val_masks_sample.shape)}", flush=True)
    print(f"  Mask unique values:            {val_mask_vals}", flush=True)
    print(f"  Number of tumor slices:        {val_tumor_slices}/{val_imgs_sample.size(0)}", flush=True)
    print(f"  Total tumor pixels:            {val_total_tumor_px:,}\n", flush=True)

    # ABORT ASSERTONS
    if not all(v in [0, 1] for v in train_mask_vals) or not all(v in [0, 1] for v in val_mask_vals):
        raise RuntimeError(f"ABORTING: Mask values are not strictly {{0, 1}}. Train: {train_mask_vals}, Val: {val_mask_vals}")

    if train_total_tumor_px == 0 and train_tumor_slices > 0:
        raise RuntimeError("ABORTING: Tumor pixels = 0 in a tumor-containing training batch!")

    if train_imgs_sample.shape[2:] != train_masks_sample.shape[2:]:
        raise RuntimeError(f"ABORTING: Dimension mismatch! Img: {train_imgs_sample.shape}, Mask: {train_masks_sample.shape}")

    print("PRE-EPOCH 1 BATCH VERIFICATION PASSED. PROCEEDING TO TRAINING.", flush=True)
    print("=" * 50 + "\n", flush=True)

    # 4. Model & Optimizer Setup
    set_seed(42)
    model = UNet2D(in_channels=1, out_channels=1, base_channels=32).to(device)
    total_params = count_parameters(model)
    print(f"Instantiated fresh UNet2D model ({total_params:,} parameters).", flush=True)

    lr = 1e-3
    epochs = 5
    criterion = CombinedDiceBCELoss(weight_dice=0.5, weight_bce=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda")

    best_ckpt_path = os.path.join(results_models_dir, "baseline_01_clean_v2_best.pt")
    latest_ckpt_path = os.path.join(results_models_dir, "baseline_01_clean_v2_latest.pt")

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_global_dice": [],
        "val_global_iou": [],
        "val_precision": [],
        "val_recall": [],
        "val_tumor_dice": [],
        "val_tumor_iou": [],
        "lr": [],
        "gpu_memory_mb": []
    }

    best_val_tumor_dice = -1.0
    best_epoch = 0

    print("\nSTARTING 5-EPOCH CONTROLLED GPU TRAINING (Baseline-01-CLEAN-V2)", flush=True)
    print("-" * 75, flush=True)

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # Training loop
        model.train()
        running_train_loss = torch.tensor(0.0, device=device)
        train_samples = 0

        for batch_idx, (imgs, masks) in enumerate(train_loader, 1):
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda"):
                logits = model(imgs)
                loss = criterion(logits, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_train_loss += loss.detach() * imgs.size(0)
            train_samples += imgs.size(0)

            if batch_idx % 100 == 0 or batch_idx == len(train_loader):
                print(f"[Epoch {epoch}/{epochs}] Train Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f}", flush=True)

        epoch_train_loss = (running_train_loss / max(1, train_samples)).item()

        # Validation loop
        model.eval()
        running_val_loss = torch.tensor(0.0, device=device)
        val_samples = 0
        accumulator = ConfusionMatrixAccumulator()

        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs = imgs.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)

                with torch.amp.autocast("cuda"):
                    logits = model(imgs)
                    loss = criterion(logits, masks)

                running_val_loss += loss.detach() * imgs.size(0)
                val_samples += imgs.size(0)

                probs = torch.sigmoid(logits)
                accumulator.update(probs, masks, threshold=0.5)

        epoch_val_loss = (running_val_loss / max(1, val_samples)).item()
        val_metrics = accumulator.get_metrics()
        elapsed = time.time() - t0

        val_global_dice = val_metrics["val_global_dice"]
        val_global_iou = val_metrics["val_global_iou"]
        val_prec = val_metrics["val_precision"]
        val_rec = val_metrics["val_recall"]
        val_tumor_dice = val_metrics["val_tumor_dice"]
        val_tumor_iou = val_metrics["val_tumor_iou"]
        gpu_mem = torch.cuda.memory_allocated() / (1024 ** 2)

        history["epoch"].append(epoch)
        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["val_global_dice"].append(val_global_dice)
        history["val_global_iou"].append(val_global_iou)
        history["val_precision"].append(val_prec)
        history["val_recall"].append(val_rec)
        history["val_tumor_dice"].append(val_tumor_dice)
        history["val_tumor_iou"].append(val_tumor_iou)
        history["lr"].append(lr)
        history["gpu_memory_mb"].append(gpu_mem)

        # Print formatted epoch logs
        print(f"\n==================================================", flush=True)
        print(f"Epoch {epoch}/{epochs}", flush=True)
        print(f"==================================================", flush=True)
        print(f"Train Loss:           {epoch_train_loss:.4f}", flush=True)
        print(f"Validation Loss:      {epoch_val_loss:.4f}\n", flush=True)
        print(f"Global Validation Dice: {val_global_dice:.4f}", flush=True)
        print(f"Global Validation IoU:  {val_global_iou:.4f}", flush=True)
        print(f"Validation Precision:   {val_prec:.4f}", flush=True)
        print(f"Validation Recall:      {val_rec:.4f}\n", flush=True)
        print(f"Tumor-Slice Validation Dice: {val_tumor_dice:.4f}", flush=True)
        print(f"Tumor-Slice Validation IoU:  {val_tumor_iou:.4f}\n", flush=True)
        print(f"Learning Rate:        {lr}", flush=True)
        print(f"GPU Memory:           {gpu_mem:.2f} MB", flush=True)

        ckpt_dict = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_global_dice": val_global_dice,
            "val_global_iou": val_global_iou,
            "val_precision": val_prec,
            "val_recall": val_rec,
            "val_tumor_dice": val_tumor_dice,
            "val_tumor_iou": val_tumor_iou,
            "val_loss": epoch_val_loss
        }

        # Save latest V2 checkpoint
        torch.save(ckpt_dict, latest_ckpt_path)

        # Checkpoint selection based on val_tumor_dice
        if val_tumor_dice > best_val_tumor_dice:
            best_val_tumor_dice = val_tumor_dice
            best_epoch = epoch
            torch.save(ckpt_dict, best_ckpt_path)
            print(f"-> Status: NEW BEST MODEL (Val Tumor Dice: {val_tumor_dice:.4f} > previous best)", flush=True)
        else:
            print(f"-> Status: No improvement (Val Tumor Dice: {val_tumor_dice:.4f} <= best {best_val_tumor_dice:.4f})", flush=True)

    # 5. Save Training History CSV
    csv_path = os.path.join(results_metrics_dir, "baseline_01_clean_v2_history.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_global_dice", "val_global_iou", "val_precision", "val_recall", "val_tumor_dice", "val_tumor_iou", "lr", "gpu_memory_mb"])
        for i in range(len(history["epoch"])):
            writer.writerow([
                history["epoch"][i],
                history["train_loss"][i],
                history["val_loss"][i],
                history["val_global_dice"][i],
                history["val_global_iou"][i],
                history["val_precision"][i],
                history["val_recall"][i],
                history["val_tumor_dice"][i],
                history["val_tumor_iou"][i],
                history["lr"][i],
                history["gpu_memory_mb"][i]
            ])
    print(f"\nSaved training history to {csv_path}", flush=True)

    # 6. Save Configuration JSON
    config_dict = {
        "experiment_name": "Baseline-01-CLEAN-V2: FLAIR-only 2D U-Net",
        "model_name": "UNet2D",
        "in_channels": 1,
        "out_channels": 1,
        "base_channels": 32,
        "total_parameters": total_params,
        "batch_size": batch_size,
        "epochs": epochs,
        "best_epoch": best_epoch,
        "best_val_tumor_dice": best_val_tumor_dice,
        "learning_rate": lr,
        "optimizer": "AdamW",
        "loss_function": "CombinedDiceBCELoss (0.5 Dice + 0.5 BCE)",
        "seed": 42,
        "amp_enabled": True,
        "device": str(device),
        "gpu_name": gpu_name,
        "data_summary": {
            "train_slices": len(train_dataset),
            "train_patients": 245,
            "val_slices": len(val_dataset),
            "val_patients": 58,
            "test_slices": 3076,
            "test_patients": 57
        }
    }
    config_path = os.path.join(results_metrics_dir, "baseline_01_clean_v2_config.json")
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)
    print(f"Saved configuration to {config_path}", flush=True)

    # 7. Plot Training Curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history["epoch"], history["train_loss"], label="Train Loss", color="#1f77b4", linewidth=2, marker="o")
    plt.plot(history["epoch"], history["val_loss"], label="Val Loss", color="#ff7f0e", linewidth=2, marker="o")
    plt.title("Baseline-01-CLEAN-V2 Loss (5 Epochs)", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history["epoch"], history["val_tumor_dice"], label="Val Tumor Dice", color="#2ca02c", linewidth=2, marker="o")
    plt.plot(history["epoch"], history["val_global_dice"], label="Val Global Dice", color="#d62728", linewidth=2, marker="o")
    plt.title("Baseline-01-CLEAN-V2 Validation Metrics", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()

    plt.tight_layout()
    curves_path = os.path.join(results_figures_dir, "baseline_01_clean_v2_curves.png")
    plt.savefig(curves_path, dpi=300)
    plt.close()
    print(f"Saved training curves to {curves_path}", flush=True)

    # 8. Load Best V2 Checkpoint & Re-Evaluate on Validation Set
    print("\n" + "=" * 50, flush=True)
    print("RE-EVALUATING BEST V2 CHECKPOINT ON VALIDATION SET", flush=True)
    print("=" * 50, flush=True)
    best_ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    model.eval()

    val_accumulator = ConfusionMatrixAccumulator()
    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            with torch.amp.autocast("cuda"):
                logits = model(imgs)

            probs = torch.sigmoid(logits)
            val_accumulator.update(probs, masks, threshold=0.5)

    final_val_metrics = val_accumulator.get_metrics()

    # 9. Generate 12-Slice Validation Predictions Visualization
    print("GENERATING QUALITATIVE VALIDATION PREDICTION OVERLAYS (12 SLICES)...", flush=True)
    val_filenames = val_dataset.image_filenames
    sample_indices = []
    seen_patients = set()

    for idx, fname in enumerate(val_filenames):
        patient_id = fname.split("_slice")[0]
        mask_path = os.path.join(data_dir, "val", "masks", fname)
        mask_arr = np.array(Image.open(mask_path))
        if np.sum(mask_arr > 0) > 50 and patient_id not in seen_patients:
            sample_indices.append(idx)
            seen_patients.add(patient_id)
            if len(sample_indices) >= 12:
                break

    fig, axes = plt.subplots(12, 6, figsize=(18, 36))
    fig.suptitle("Baseline-01-CLEAN-V2: 5-Epoch Validation Predictions (FLAIR 2D U-Net)", fontsize=16, fontweight="bold", y=0.995)

    headers = ["FLAIR", "Ground Truth", "Prediction Prob", "Binary Pred", "GT Overlay", "Pred Overlay"]
    for col_idx, text in enumerate(headers):
        axes[0, col_idx].set_title(text, fontsize=12, fontweight="bold")

    with torch.inference_mode():
        for row_idx, sample_idx in enumerate(sample_indices[:12]):
            img_tensor, mask_tensor = val_dataset[sample_idx]
            fname = val_filenames[sample_idx]
            patient_id = fname.split("_slice")[0]

            img_input = img_tensor.unsqueeze(0).to(device)
            with torch.amp.autocast("cuda"):
                logits = model(img_input)

            prob = torch.sigmoid(logits).squeeze().cpu().numpy()
            pred_bin = (prob >= 0.5).astype(np.float32)

            flair_np = img_tensor.squeeze().numpy()
            gt_np = mask_tensor.squeeze().numpy()

            gt_overlay = np.stack([flair_np + gt_np * 0.5, flair_np, flair_np], axis=-1)
            gt_overlay = np.clip(gt_overlay, 0.0, 1.0)

            pred_overlay = np.stack([flair_np, flair_np + pred_bin * 0.5, flair_np], axis=-1)
            pred_overlay = np.clip(pred_overlay, 0.0, 1.0)

            axes[row_idx, 0].imshow(flair_np, cmap="gray")
            axes[row_idx, 0].set_ylabel(f"{patient_id}", fontsize=10, fontweight="bold")
            axes[row_idx, 1].imshow(gt_np, cmap="gray")
            axes[row_idx, 2].imshow(prob, cmap="magma")
            axes[row_idx, 3].imshow(pred_bin, cmap="gray")
            axes[row_idx, 4].imshow(gt_overlay)
            axes[row_idx, 5].imshow(pred_overlay)

            for c in range(6):
                axes[row_idx, c].set_xticks([])
                axes[row_idx, c].set_yticks([])

    plt.tight_layout()
    preds_fig_path = os.path.join(results_figures_dir, "baseline_01_clean_v2_predictions.png")
    plt.savefig(preds_fig_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved validation predictions figure to {preds_fig_path}", flush=True)

    # 10. Print Final Summary Table
    print("\n" + "=" * 50, flush=True)
    print("SUSHRUTA — BASELINE-01-CLEAN-V2", flush=True)
    print("5 EPOCH CHECK", flush=True)
    print("=" * 50, flush=True)
    print(f"Best Epoch: {best_epoch}", flush=True)
    print(f"Best Validation Tumor Dice: {best_val_tumor_dice:.4f}\n", flush=True)

    for ep in range(1, 6):
        print(f"Epoch {ep}: Val Tumor Dice = {history['val_tumor_dice'][ep-1]:.4f} | Val Global Dice = {history['val_global_dice'][ep-1]:.4f} | Val Loss = {history['val_loss'][ep-1]:.4f}", flush=True)

    print(f"\nFinal Training Loss:   {history['train_loss'][-1]:.4f}", flush=True)
    print(f"Final Validation Loss: {history['val_loss'][-1]:.4f}\n", flush=True)

    print(f"Global Validation Dice: {final_val_metrics['val_global_dice']:.4f}", flush=True)
    print(f"Global Validation IoU:  {final_val_metrics['val_global_iou']:.4f}", flush=True)
    print(f"Validation Precision:   {final_val_metrics['val_precision']:.4f}", flush=True)
    print(f"Validation Recall:      {final_val_metrics['val_recall']:.4f}\n", flush=True)

    print(f"Tumor Validation Dice:  {final_val_metrics['val_tumor_dice']:.4f}", flush=True)
    print(f"Tumor Validation IoU:   {final_val_metrics['val_tumor_iou']:.4f}\n", flush=True)

    print(f"GPU: {gpu_name}", flush=True)
    print(f"GPU Memory: {history['gpu_memory_mb'][-1]:.2f} MB", flush=True)
    print("=" * 50, flush=True)

    # Pass condition: meaningful improvement over random baseline (~0.0431) and increasing progression across epochs
    is_learning = (history['val_tumor_dice'][-1] > 0.0431) and (history['val_tumor_dice'][-1] > history['val_tumor_dice'][0])
    if is_learning:
        print("\n5-EPOCH CHECK PASSED — MODEL IS LEARNING\n", flush=True)
    else:
        print("\n5-EPOCH CHECK FAILED — INVESTIGATION REQUIRED\n", flush=True)


if __name__ == "__main__":
    run_baseline_01_clean_v2_5epoch()
