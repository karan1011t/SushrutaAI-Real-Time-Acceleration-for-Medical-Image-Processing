"""
Segmentation Evaluation Metrics Module for Medical Image Segmentation.

Provides mathematically rigorous metric computation routines for binary segmentation,
supporting Global Pixel-Level (Micro-Averaged) metrics, Tumor-Only Slice Macro metrics,
and explicit empty-mask / confusion matrix accumulation.
"""

from typing import Dict, Any, Tuple, List
import numpy as np
import torch


def compute_global_pixel_metrics(
    total_tp: int,
    total_fp: int,
    total_fn: int,
    total_tn: int
) -> Dict[str, float]:
    """
    Computes mathematically rigorous global pixel-level (micro-averaged) metrics
    from accumulated confusion matrix counts across the dataset.

    Zero denominators are handled explicitly to return 0.0 rather than
    producing misleading scores via epsilon smoothing.
    """
    precision = (total_tp / (total_tp + total_fp)) if (total_tp + total_fp) > 0 else 0.0
    recall = (total_tp / (total_tp + total_fn)) if (total_tp + total_fn) > 0 else 0.0
    
    denom_f1 = precision + recall
    f1 = (2.0 * precision * recall / denom_f1) if denom_f1 > 0 else 0.0

    denom_dice = 2 * total_tp + total_fp + total_fn
    dice = (2.0 * total_tp / denom_dice) if denom_dice > 0 else 0.0

    denom_iou = total_tp + total_fp + total_fn
    iou = (total_tp / denom_iou) if denom_iou > 0 else 0.0

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
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Computes per-image binary segmentation metrics for a single 2D slice.
    
    Explicitly distinguishes between:
    - Empty GT + Empty Pred (True Negative slice match, marked is_empty_gt=True)
    - Empty GT + Non-Empty Pred (False Positive tumor prediction, precision=0.0)
    - Tumor GT + Empty Pred (False Negative, precision=0.0, recall=0.0, dice=0.0)
    - Tumor GT + Non-Empty Pred (Standard TP, FP, FN metrics)
    """
    preds = (pred_mask >= threshold).float().view(-1)
    targets = (target_mask >= 0.5).float().view(-1)

    tp = int(torch.sum(preds * targets).item())
    fp = int(torch.sum(preds * (1.0 - targets)).item())
    fn = int(torch.sum((1.0 - preds) * targets).item())
    tn = int(torch.sum((1.0 - preds) * (1.0 - targets)).item())

    total_pred = tp + fp
    total_target = tp + fn

    is_empty_gt = (total_target == 0)
    is_empty_pred = (total_pred == 0)

    if is_empty_gt and is_empty_pred:
        # Correctly predicted empty slice
        return {
            "dice": 1.0,
            "iou": 1.0,
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "is_empty_gt": True,
            "is_tumor_slice": False
        }
    elif is_empty_gt and not is_empty_pred:
        # False Positive tumor predicted on empty slice
        return {
            "dice": 0.0,
            "iou": 0.0,
            "precision": 0.0,
            "recall": 1.0,  # No FN on empty GT
            "f1": 0.0,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "is_empty_gt": True,
            "is_tumor_slice": False
        }
    elif not is_empty_gt and is_empty_pred:
        # False Negative: missed tumor entirely
        return {
            "dice": 0.0,
            "iou": 0.0,
            "precision": 0.0,  # 0 TP out of 0 preds => 0.0 (NOT 1.0)
            "recall": 0.0,
            "f1": 0.0,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "is_empty_gt": False,
            "is_tumor_slice": True
        }
    else:
        # Tumor present and tumor predicted
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        denom_f1 = precision + recall
        f1 = (2.0 * precision * recall / denom_f1) if denom_f1 > 0 else 0.0

        denom_dice = 2 * tp + fp + fn
        dice = (2.0 * tp / denom_dice) if denom_dice > 0 else 0.0

        denom_iou = tp + fp + fn
        iou = (tp / denom_iou) if denom_iou > 0 else 0.0

        return {
            "dice": float(dice),
            "iou": float(iou),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "is_empty_gt": False,
            "is_tumor_slice": True
        }


class ConfusionMatrixAccumulator:
    """
    Vectorized dataset-wide metric accumulator tracking raw TP, FP, FN, TN,
    tumor-slice per-image metrics, and empty-slice counts.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.total_tp = 0
        self.total_fp = 0
        self.total_fn = 0
        self.total_tn = 0
        self.tumor_slice_dices: List[float] = []
        self.tumor_slice_ious: List[float] = []
        self.all_slice_dices: List[float] = []
        self.num_empty_slices = 0
        self.num_tumor_slices = 0

    def update(self, probs: torch.Tensor, masks: torch.Tensor, threshold: float = 0.5):
        preds = (probs >= threshold).float()
        targets = (masks >= 0.5).float()

        p = preds.view(preds.size(0), -1)
        t = targets.view(targets.size(0), -1)

        tp = (p * t).sum(dim=1)
        fp = (p * (1.0 - t)).sum(dim=1)
        fn = ((1.0 - p) * t).sum(dim=1)
        tn = ((1.0 - p) * (1.0 - t)).sum(dim=1)

        self.total_tp += int(tp.sum().item())
        self.total_fp += int(fp.sum().item())
        self.total_fn += int(fn.sum().item())
        self.total_tn += int(tn.sum().item())

        total_p = tp + fp
        total_t = tp + fn

        is_empty_gt = (total_t == 0)
        is_empty_pred = (total_p == 0)

        # Slice-level metrics vectorized
        zero_tensor = torch.zeros_like(tp)
        one_tensor = torch.ones_like(tp)

        dice = torch.where(
            is_empty_gt & is_empty_pred,
            one_tensor,
            torch.where(
                is_empty_gt | is_empty_pred,
                zero_tensor,
                (2.0 * tp) / (2.0 * tp + fp + fn + 1e-12)
            )
        )

        iou = torch.where(
            is_empty_gt & is_empty_pred,
            one_tensor,
            torch.where(
                is_empty_gt | is_empty_pred,
                zero_tensor,
                tp / (tp + fp + fn + 1e-12)
            )
        )

        dice_list = dice.cpu().tolist()
        iou_list = iou.cpu().tolist()
        is_tumor_slice_list = (~is_empty_gt).cpu().tolist()

        for d, i, is_tumor in zip(dice_list, iou_list, is_tumor_slice_list):
            self.all_slice_dices.append(d)
            if is_tumor:
                self.num_tumor_slices += 1
                self.tumor_slice_dices.append(d)
                self.tumor_slice_ious.append(i)
            else:
                self.num_empty_slices += 1

    def get_metrics(self) -> Dict[str, Any]:
        global_m = compute_global_pixel_metrics(self.total_tp, self.total_fp, self.total_fn, self.total_tn)
        val_tumor_dice = float(np.mean(self.tumor_slice_dices)) if len(self.tumor_slice_dices) > 0 else 0.0
        val_tumor_iou = float(np.mean(self.tumor_slice_ious)) if len(self.tumor_slice_ious) > 0 else 0.0
        val_macro_all_dice = float(np.mean(self.all_slice_dices)) if len(self.all_slice_dices) > 0 else 0.0

        return {
            "val_global_dice": global_m["dice"],
            "val_global_iou": global_m["iou"],
            "val_precision": global_m["precision"],
            "val_recall": global_m["recall"],
            "val_tumor_dice": val_tumor_dice,
            "val_tumor_iou": val_tumor_iou,
            "val_macro_all_dice": val_macro_all_dice,
            "num_tumor_slices": self.num_tumor_slices,
            "num_empty_slices": self.num_empty_slices
        }


def calculate_segmentation_metrics(
    preds: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Legacy helper function for single batch metrics.
    """
    accumulator = ConfusionMatrixAccumulator()
    accumulator.update(preds, targets, threshold=threshold)
    return accumulator.get_metrics()
