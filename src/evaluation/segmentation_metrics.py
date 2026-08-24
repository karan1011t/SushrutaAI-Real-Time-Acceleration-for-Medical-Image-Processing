"""
Segmentation Evaluation Metrics Module for Medical Image Segmentation.

Provides mathematically rigorous metric computation routines for binary segmentation,
supporting both Global Pixel-Level (Micro-Averaged) metrics and Per-Image Macro metrics.
"""

from typing import Dict, Any, Tuple
import numpy as np
import torch


def compute_global_pixel_metrics(
    total_tp: int,
    total_fp: int,
    total_fn: int,
    total_tn: int,
    smooth: float = 1e-6
) -> Dict[str, float]:
    """
    Computes mathematically rigorous global pixel-level (micro-averaged) metrics
    from accumulated confusion matrix counts across the dataset.

    Under global pixel accumulation, Global Dice and Global F1 are mathematically IDENTICAL:
        Dice_global = 2*TP / (2*TP + FP + FN)
        F1_global   = 2*Precision*Recall / (Precision + Recall) = 2*TP / (2*TP + FP + FN)
    """
    precision = total_tp / (total_tp + total_fp + smooth)
    recall = total_tp / (total_tp + total_fn + smooth)
    f1 = (2.0 * precision * recall) / (precision + recall + smooth)
    dice = (2.0 * total_tp) / (2.0 * total_tp + total_fp + total_fn + smooth)
    iou = total_tp / (total_tp + total_fp + total_fn + smooth)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "dice": float(dice),
        "iou": float(iou)
    }


def compute_binary_metrics_single_image(
    pred_mask: torch.Tensor,
    target_mask: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-6
) -> Dict[str, float]:
    """
    Computes per-image binary segmentation metrics for a single 2D slice.
    """
    preds = (pred_mask >= threshold).float().view(-1)
    targets = (target_mask >= 0.5).float().view(-1)

    total_pred = torch.sum(preds).item()
    total_target = torch.sum(targets).item()
    intersection = torch.sum(preds * targets).item()

    if total_target == 0 and total_pred == 0:
        return {
            "dice": 1.0,
            "iou": 1.0,
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "is_empty": True
        }

    dice = (2.0 * intersection + smooth) / (total_pred + total_target + smooth)
    union = total_pred + total_target - intersection
    iou = (intersection + smooth) / (union + smooth)

    precision = (intersection + smooth) / (total_pred + smooth) if total_pred > 0 else 0.0
    recall = (intersection + smooth) / (total_target + smooth) if total_target > 0 else 0.0
    
    denom = precision + recall
    f1 = (2.0 * precision * recall + smooth) / (denom + smooth) if denom > 0 else 0.0

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "is_empty": False
    }


def calculate_segmentation_metrics(
    preds: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-6
) -> Dict[str, float]:
    """
    Calculates batch-level mean metrics for a batch of predictions and targets.
    """
    batch_size = preds.size(0)
    accumulated = {"dice": 0.0, "iou": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    for b in range(batch_size):
        m = compute_binary_metrics_single_image(preds[b], targets[b], threshold=threshold, smooth=smooth)
        for k in accumulated:
            accumulated[k] += m[k]

    return {k: v / max(1, batch_size) for k, v in accumulated.items()}
