"""
SUSHRUTA — 4-Modality MRI U-Net Segmentation Training & Evaluation Module.
Trains a 4-channel 2D U-Net [FLAIR, T1, T1CE, T2] for Whole-Tumor segmentation on BraTS NIfTI volumes.

Target Paths:
- Best Checkpoint: results/models/multimodal_unet_best.pt
- Training History: results/metrics/multimodal_unet_history.csv
- Training Curves: results/figures/multimodal_unet_curves.png
- Test Metrics Report: results/metrics/multimodal_unet_test_metrics.json

Safety Notice:
This module preserves baseline_01 and baseline_02 checkpoints and historical metrics unchanged.
Does NOT run training automatically upon import.
"""

import os
import sys
import time
import json
import csv
import random
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Add project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models.unet import UNet2D, count_parameters
from src.models.multimodal_dataset import MultiModalBraTS2DDataset, MULTIMODAL_CHANNEL_ORDER
from src.evaluation.segmentation_metrics import compute_binary_metrics_single_image


def set_seed(seed: int = 42):
    """Sets random seeds across Python, NumPy, and PyTorch for exact reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class CombinedDiceBCELoss(nn.Module):
    """
    Combined Segmentation Loss: Soft Dice Loss + BCEWithLogitsLoss.
    Receives raw un-normalized model logits to prevent double-sigmoid application.
    """

    def __init__(self, weight_dice: float = 0.5, weight_bce: float = 0.5, smooth: float = 1e-6):
        super().__init__()
        self.weight_dice = weight_dice
        self.weight_bce = weight_bce
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)

        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum(dim=(1, 2, 3))
        cardinality = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
        dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        dice_loss = 1.0 - dice_score.mean()

        return (self.weight_dice * dice_loss) + (self.weight_bce * bce_loss)


def compute_batch_segmentation_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Computes global Dice, IoU, Precision, and Recall metrics over a batch.
    Thresholds sigmoid probabilities at the specified threshold (default 0.5).
    """
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()

    intersection = (preds * targets).sum().item()
    pred_sum = preds.sum().item()
    target_sum = targets.sum().item()

    smooth = 1e-6
    dice = (2.0 * intersection + smooth) / (pred_sum + target_sum + smooth)
    union = pred_sum + target_sum - intersection
    iou = (intersection + smooth) / (union + smooth)
    precision = (intersection + smooth) / (pred_sum + smooth)
    recall = (intersection + smooth) / (target_sum + smooth)

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall)
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, Dict[str, float]]:
    """Runs one training epoch over the dataset loader."""
    model.train()
    total_loss = 0.0
    metrics_sum = {"dice": 0.0, "iou": 0.0, "precision": 0.0, "recall": 0.0}

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        batch_metrics = compute_batch_segmentation_metrics(logits.detach(), masks)
        for k, v in batch_metrics.items():
            metrics_sum[k] += v * images.size(0)

    n_samples = len(loader.dataset)
    avg_loss = total_loss / max(n_samples, 1)
    avg_metrics = {k: v / max(n_samples, 1) for k, v in metrics_sum.items()}

    return avg_loss, avg_metrics


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float = 0.5
) -> Tuple[float, Dict[str, float]]:
    """Evaluates the model on validation or test split."""
    model.eval()
    total_loss = 0.0
    metrics_sum = {"dice": 0.0, "iou": 0.0, "precision": 0.0, "recall": 0.0}

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)
        loss = criterion(logits, masks)

        total_loss += loss.item() * images.size(0)
        batch_metrics = compute_batch_segmentation_metrics(logits, masks, threshold=threshold)
        for k, v in batch_metrics.items():
            metrics_sum[k] += v * images.size(0)

    n_samples = len(loader.dataset)
    avg_loss = total_loss / max(n_samples, 1)
    avg_metrics = {k: v / max(n_samples, 1) for k, v in metrics_sum.items()}

    return avg_loss, avg_metrics


def run_multimodal_unet_training(
    data_dir: str = "data/raw/brats",
    manifest_path: str = "data/processed/brats/split_manifest.json",
    epochs: int = 5,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    seed: int = 42,
    num_workers: int = 0,
    device_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Executes reproducible training and evaluation for the 4-channel Multi-Modal U-Net baseline.
    Saves best model checkpoint to results/models/multimodal_unet_best.pt.
    """
    set_seed(seed)

    if device_name is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    print(f"=== SUSHRUTA 4-MODALITY U-NET TRAINING PIPELINE ===")
    print(f"Device:            {device}")
    print(f"Channel Order:     {MULTIMODAL_CHANNEL_ORDER}")
    print(f"Target Epochs:     {epochs}")
    print(f"Batch Size:        {batch_size}")
    print(f"Learning Rate:     {learning_rate}")
    print(f"Random Seed:       {seed}")

    # Output directory preparation
    results_models_dir = Path(project_root) / "results" / "models"
    results_metrics_dir = Path(project_root) / "results" / "metrics"
    results_figures_dir = Path(project_root) / "results" / "figures"

    results_models_dir.mkdir(parents=True, exist_ok=True)
    results_metrics_dir.mkdir(parents=True, exist_ok=True)
    results_figures_dir.mkdir(parents=True, exist_ok=True)

    best_ckpt_path = results_models_dir / "multimodal_unet_best.pt"
    history_csv_path = results_metrics_dir / "multimodal_unet_history.csv"
    figures_path = results_figures_dir / "multimodal_unet_curves.png"
    test_report_path = results_metrics_dir / "multimodal_unet_test_metrics.json"

    # Datasets and Loaders
    train_dataset = MultiModalBraTS2DDataset(data_dir=data_dir, manifest_path=manifest_path, split="train", augment=True, seed=seed)
    val_dataset = MultiModalBraTS2DDataset(data_dir=data_dir, manifest_path=manifest_path, split="val", augment=False, seed=seed)
    test_dataset = MultiModalBraTS2DDataset(data_dir=data_dir, manifest_path=manifest_path, split="test", augment=False, seed=seed)

    print(f"\n--- DATASET ELIGIBILITY & SPLIT SUMMARY ---")
    print(f"Train Split: {len(train_dataset.valid_cases)} valid patients ({len(train_dataset.excluded_cases)} excluded), {len(train_dataset):,} slices")
    print(f"Val Split:   {len(val_dataset.valid_cases)} valid patients ({len(val_dataset.excluded_cases)} excluded), {len(val_dataset):,} slices")
    print(f"Test Split:  {len(test_dataset.valid_cases)} valid patients ({len(test_dataset.excluded_cases)} excluded), {len(test_dataset):,} slices")

    if len(train_dataset) == 0:
        raise RuntimeError("Train dataset is empty! Verify raw dataset path and split_manifest.json.")
    if len(val_dataset) == 0:
        raise RuntimeError("Validation dataset is empty!")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())

    # Instantiate 4-Channel U-Net
    model = UNet2D(in_channels=4, out_channels=1, base_channels=32).to(device)
    print(f"Model: 4-Channel UNet2D ({count_parameters(model):,} parameters)")

    criterion = CombinedDiceBCELoss(weight_dice=0.5, weight_bce=0.5)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    history = []
    best_val_dice = -1.0

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_metrics = evaluate_epoch(model, val_loader, criterion, device, threshold=0.5)
        elapsed = time.time() - t0

        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_dice": round(train_metrics["dice"], 4),
            "train_iou": round(train_metrics["iou"], 4),
            "val_loss": round(val_loss, 4),
            "val_dice": round(val_metrics["dice"], 4),
            "val_iou": round(val_metrics["iou"], 4),
            "epoch_time_sec": round(elapsed, 2)
        }
        history.append(row)

        print(f"Epoch [{epoch:02d}/{epochs:02d}] ({elapsed:.1f}s) | "
              f"Train Loss: {train_loss:.4f}, Dice: {train_metrics['dice']:.4f} | "
              f"Val Loss: {val_loss:.4f}, Dice: {val_metrics['dice']:.4f}", flush=True)

        # Save Best Checkpoint (based strictly on Validation Dice)
        if val_metrics["dice"] > best_val_dice:
            best_val_dice = val_metrics["dice"]
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_dice": val_metrics["dice"],
                "val_loss": val_loss,
                "in_channels": 4,
                "channel_order": MULTIMODAL_CHANNEL_ORDER,
                "decision_threshold": 0.5
            }
            torch.save(checkpoint, best_ckpt_path)
            print(f"  -> Saved Best Model Checkpoint to {best_ckpt_path.name} (Val Dice: {best_val_dice:.4f})", flush=True)

    total_training_time = time.time() - start_time
    print(f"\n[TRAINING COMPLETE] Total Elapsed: {total_training_time:.1f}s | Best Val Dice: {best_val_dice:.4f}")

    # Write History CSV
    if history:
        with open(history_csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)
        print(f"Saved Training History to {history_csv_path}")

    # Held-Out Test Split Evaluation (Strictly AFTER Model Selection)
    if best_ckpt_path.exists():
        best_ckpt = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(best_ckpt["model_state_dict"])
        print(f"\nLoaded Best Checkpoint from Epoch {best_ckpt['epoch']} for Final Test Evaluation...")

    test_loss, test_metrics = evaluate_epoch(model, test_loader, criterion, device, threshold=0.5)

    test_report = {
        "model_name": "MultiModal_UNet_4Channel",
        "in_channels": 4,
        "channel_order": MULTIMODAL_CHANNEL_ORDER,
        "best_epoch": best_ckpt.get("epoch", epochs) if 'best_ckpt' in locals() else epochs,
        "test_loss": round(test_loss, 4),
        "test_dice": round(test_metrics["dice"], 4),
        "test_iou": round(test_metrics["iou"], 4),
        "test_precision": round(test_metrics["precision"], 4),
        "test_recall": round(test_metrics["recall"], 4),
        "decision_threshold": 0.5,
        "disclaimer": "Research-prototype comparative benchmark result. Performance on external hospital scans is not clinically validated."
    }

    with open(test_report_path, "w", encoding="utf-8") as f:
        json.dump(test_report, f, indent=2)
    print(f"Saved Test Evaluation Report to {test_report_path}")
    print(f"Test Set Metrics -> Loss: {test_loss:.4f} | Dice: {test_metrics['dice']:.4f} | IoU: {test_metrics['iou']:.4f} | Recall: {test_metrics['recall']:.4f}")

    return test_report


if __name__ == "__main__":
    # Import protection: executing standalone triggers training run
    run_multimodal_unet_training(epochs=5, batch_size=16)
