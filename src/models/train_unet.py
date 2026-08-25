"""
Training and Validation Engine for 2D U-Net Medical Image Segmentation.
Supports Baseline-01 FLAIR-only U-Net training, validation, AMP mixed precision, early stopping, and checkpointing.
"""

import os
import time
import random
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.unet import UNet2D, count_parameters
from src.evaluation.segmentation_metrics import (
    ConfusionMatrixAccumulator,
    compute_binary_metrics_single_image,
    compute_global_pixel_metrics
)


def set_seed(seed: int = 42):
    """Sets random seeds for python, numpy, and torch reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class CombinedDiceBCELoss(nn.Module):
    """
    Combined Segmentation Loss: Dice Loss + Binary Cross Entropy with Logits Loss.

    Parameters:
        weight_dice (float): Relative weight for Dice loss.
        weight_bce (float): Relative weight for BCE loss.
        smooth (float): Smoothing factor for Dice calculation.
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

        total_loss = (self.weight_dice * dice_loss) + (self.weight_bce * bce_loss)
        return total_loss


class UNetTrainer:
    """
    Manager class for training, validation, AMP mixed precision, early stopping, and checkpointing.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        lr: float = 1e-3,
        weight_dice: float = 0.5,
        weight_bce: float = 0.5,
        threshold: float = 0.5,
        checkpoint_dir: str = "results/models",
        best_ckpt_name: str = "baseline_01_flair_unet_best.pt",
        device: Optional[torch.device] = None,
        seed: int = 42
    ):
        set_seed(seed)
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.lr = lr
        self.threshold = threshold
        self.checkpoint_dir = checkpoint_dir
        self.best_ckpt_name = best_ckpt_name
        os.makedirs(checkpoint_dir, exist_ok=True)

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.criterion = CombinedDiceBCELoss(weight_dice=weight_dice, weight_bce=weight_bce)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)

        # PyTorch AMP Scaler (enabled if CUDA is available)
        self.use_amp = (self.device.type == "cuda")
        if self.use_amp:
            self.scaler = torch.amp.GradScaler("cuda")
        else:
            self.scaler = None

        self.best_val_tumor_dice = -1.0
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "val_global_dice": [],
            "val_global_iou": [],
            "val_precision": [],
            "val_recall": [],
            "val_tumor_dice": [],
            "val_tumor_iou": []
        }

    def train_epoch(self, max_batches: Optional[int] = None) -> float:
        self.model.train()
        running_loss = 0.0
        total_samples = 0

        for b_idx, (images, masks) in enumerate(self.train_loader):
            if max_batches and b_idx >= max_batches:
                break

            images = images.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            if self.use_amp:
                with torch.amp.autocast("cuda"):
                    logits = self.model(images)
                    loss = self.criterion(logits, masks)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(images)
                loss = self.criterion(logits, masks)
                loss.backward()
                self.optimizer.step()

            running_loss += loss.item() * images.size(0)
            total_samples += images.size(0)

        epoch_loss = running_loss / max(1, total_samples)
        return epoch_loss

    def validate_epoch(self, max_batches: Optional[int] = None) -> Tuple[float, Dict[str, Any]]:
        self.model.eval()
        running_loss = 0.0
        total_samples = 0
        accumulator = ConfusionMatrixAccumulator()

        with torch.no_grad():
            for b_idx, (images, masks) in enumerate(self.val_loader):
                if max_batches and b_idx >= max_batches:
                    break

                images = images.to(self.device, non_blocking=True)
                masks = masks.to(self.device, non_blocking=True)

                if self.use_amp:
                    with torch.amp.autocast("cuda"):
                        logits = self.model(images)
                        loss = self.criterion(logits, masks)
                else:
                    logits = self.model(images)
                    loss = self.criterion(logits, masks)

                running_loss += loss.item() * images.size(0)
                total_samples += images.size(0)

                probs = torch.sigmoid(logits)
                accumulator.update(probs, masks, threshold=self.threshold)

        val_loss = running_loss / max(1, total_samples)
        val_metrics = accumulator.get_metrics()

        return val_loss, val_metrics

    def fit(
        self,
        epochs: int = 30,
        patience: int = 10,
        max_train_batches: Optional[int] = None,
        max_val_batches: Optional[int] = None
    ) -> Dict[str, List[float]]:
        """
        Runs training and validation for specified epochs.
        """
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_loss = self.train_epoch(max_batches=max_train_batches)
            val_loss, val_metrics = self.validate_epoch(max_batches=max_val_batches)
            elapsed = time.time() - t0

            val_global_dice = val_metrics["val_global_dice"]
            val_global_iou = val_metrics["val_global_iou"]
            val_precision = val_metrics["val_precision"]
            val_recall = val_metrics["val_recall"]
            val_tumor_dice = val_metrics["val_tumor_dice"]
            val_tumor_iou = val_metrics["val_tumor_iou"]
            gpu_mem = torch.cuda.memory_allocated() / (1024 ** 2) if self.device.type == "cuda" else 0.0

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_global_dice"].append(val_global_dice)
            self.history["val_global_iou"].append(val_global_iou)
            self.history["val_precision"].append(val_precision)
            self.history["val_recall"].append(val_recall)
            self.history["val_tumor_dice"].append(val_tumor_dice)
            self.history["val_tumor_iou"].append(val_tumor_iou)

            print(
                f"Epoch {epoch:02d}/{epochs:02d} [{elapsed:.1f}s]\n"
                f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}\n"
                f"  Val Global Dice: {val_global_dice:.4f} | Val Global IoU: {val_global_iou:.4f} | "
                f"Val Precision: {val_precision:.4f} | Val Recall: {val_recall:.4f}\n"
                f"  Val Tumor Dice: {val_tumor_dice:.4f} | Val Tumor IoU: {val_tumor_iou:.4f}\n"
                f"  Learning Rate: {self.lr} | GPU Memory: {gpu_mem:.2f} MB",
                flush=True
            )

            latest_ckpt_path = os.path.join(self.checkpoint_dir, "baseline_01_flair_unet_latest.pt")
            best_ckpt_path = os.path.join(self.checkpoint_dir, self.best_ckpt_name)

            # Checkpoint metadata storing all 10 required keys
            ckpt_dict = {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "val_global_dice": val_global_dice,
                "val_global_iou": val_global_iou,
                "val_precision": val_precision,
                "val_recall": val_recall,
                "val_tumor_dice": val_tumor_dice,
                "val_tumor_iou": val_tumor_iou,
                "val_loss": val_loss
            }

            torch.save(ckpt_dict, latest_ckpt_path)

            if val_tumor_dice > self.best_val_tumor_dice:
                self.best_val_tumor_dice = val_tumor_dice
                patience_counter = 0
                torch.save(ckpt_dict, best_ckpt_path)
                print(f"   [CHECKPOINT] Saved new best model to {best_ckpt_path} (Val Tumor Dice: {val_tumor_dice:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"   [EARLY STOPPING] Triggered after {epoch} epochs.")
                    break

        return self.history
