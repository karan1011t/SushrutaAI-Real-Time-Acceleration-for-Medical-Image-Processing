"""
Synthetic Metric Test Suite for Binary Segmentation Evaluation.
Verifies explicit TP/TN/FP/FN handling, zero-denominator safety, and empty-slice metrics.
"""

import sys
import os
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.evaluation.segmentation_metrics import (
    compute_binary_metrics_single_image,
    compute_global_pixel_metrics,
    ConfusionMatrixAccumulator
)


def run_synthetic_metric_tests():
    print("=" * 60)
    print("RUNNING SYNTHETIC SEGMENTATION METRIC TEST SUITE")
    print("=" * 60)

    # 1. CASE A: GT Empty, Prediction Empty
    print("\n[CASE A] GT Empty, Prediction Empty")
    gt_a = torch.zeros((256, 256))
    pred_a = torch.zeros((256, 256))
    m_a = compute_binary_metrics_single_image(pred_a, gt_a, threshold=0.5)
    print(f"  Calculated: Dice={m_a['dice']:.4f}, IoU={m_a['iou']:.4f}, Prec={m_a['precision']:.4f}, Rec={m_a['recall']:.4f}")
    print(f"  Empty GT Flag: {m_a['is_empty_gt']}, Tumor Slice Flag: {m_a['is_tumor_slice']}")
    assert m_a['dice'] == 1.0, f"Expected Case A Dice 1.0, got {m_a['dice']}"
    assert m_a['is_tumor_slice'] == False, "Case A must be marked non-tumor slice"
    print("  CASE A: PASSED")

    # 2. CASE B: GT Contains Tumor (100 px), Prediction Empty
    print("\n[CASE B] GT Contains Tumor (100 px), Prediction Empty")
    gt_b = torch.zeros((256, 256))
    gt_b[:10, :10] = 1.0
    pred_b = torch.zeros((256, 256))
    m_b = compute_binary_metrics_single_image(pred_b, gt_b, threshold=0.5)
    print(f"  Calculated: Dice={m_b['dice']:.4f}, IoU={m_b['iou']:.4f}, Prec={m_b['precision']:.4f}, Rec={m_b['recall']:.4f}")
    print(f"  TP={m_b['tp']}, FP={m_b['fp']}, FN={m_b['fn']}, TN={m_b['tn']}")
    assert m_b['dice'] == 0.0, f"Expected Case B Dice 0.0, got {m_b['dice']}"
    assert m_b['precision'] == 0.0, f"Expected Case B Precision 0.0 (NOT 1.0!), got {m_b['precision']}"
    assert m_b['recall'] == 0.0, f"Expected Case B Recall 0.0, got {m_b['recall']}"
    assert m_b['is_tumor_slice'] == True, "Case B must be marked tumor slice"
    print("  CASE B: PASSED")

    # 3. CASE C: GT Empty, Prediction Contains Tumor (100 px)
    print("\n[CASE C] GT Empty, Prediction Contains Tumor (100 px)")
    gt_c = torch.zeros((256, 256))
    pred_c = torch.zeros((256, 256))
    pred_c[:10, :10] = 1.0
    m_c = compute_binary_metrics_single_image(pred_c, gt_c, threshold=0.5)
    print(f"  Calculated: Dice={m_c['dice']:.4f}, IoU={m_c['iou']:.4f}, Prec={m_c['precision']:.4f}, Rec={m_c['recall']:.4f}")
    print(f"  TP={m_c['tp']}, FP={m_c['fp']}, FN={m_c['fn']}, TN={m_c['tn']}")
    assert m_c['dice'] == 0.0, f"Expected Case C Dice 0.0, got {m_c['dice']}"
    assert m_c['precision'] == 0.0, f"Expected Case C Precision 0.0, got {m_c['precision']}"
    assert m_c['is_tumor_slice'] == False, "Case C must be marked non-tumor slice"
    print("  CASE C: PASSED")

    # 4. CASE D: Prediction Exactly Equals GT (100 px)
    print("\n[CASE D] Prediction Exactly Equals GT (100 px)")
    gt_d = torch.zeros((256, 256))
    gt_d[:10, :10] = 1.0
    pred_d = torch.zeros((256, 256))
    pred_d[:10, :10] = 1.0
    m_d = compute_binary_metrics_single_image(pred_d, gt_d, threshold=0.5)
    print(f"  Calculated: Dice={m_d['dice']:.4f}, IoU={m_d['iou']:.4f}, Prec={m_d['precision']:.4f}, Rec={m_d['recall']:.4f}")
    print(f"  TP={m_d['tp']}, FP={m_d['fp']}, FN={m_d['fn']}, TN={m_d['tn']}")
    assert m_d['dice'] == 1.0, f"Expected Case D Dice 1.0, got {m_d['dice']}"
    assert m_d['iou'] == 1.0, f"Expected Case D IoU 1.0, got {m_d['iou']}"
    assert m_d['precision'] == 1.0, f"Expected Case D Precision 1.0, got {m_d['precision']}"
    assert m_d['recall'] == 1.0, f"Expected Case D Recall 1.0, got {m_d['recall']}"
    assert m_d['is_tumor_slice'] == True, "Case D must be marked tumor slice"
    print("  CASE D: PASSED")

    # 5. CASE E: Prediction Partially Overlaps GT (GT=100px, Pred=100px, TP=50px)
    print("\n[CASE E] Prediction Partially Overlaps GT (GT=100px, Pred=100px, TP=50px)")
    gt_e = torch.zeros((256, 256))
    gt_e[0:10, 0:10] = 1.0  # 100 pixels
    pred_e = torch.zeros((256, 256))
    pred_e[5:15, 0:10] = 1.0  # 100 pixels, overlap is rows 5-9 cols 0-9 = 50 pixels
    m_e = compute_binary_metrics_single_image(pred_e, gt_e, threshold=0.5)
    print(f"  Calculated: Dice={m_e['dice']:.4f}, IoU={m_e['iou']:.4f}, Prec={m_e['precision']:.4f}, Rec={m_e['recall']:.4f}")
    print(f"  TP={m_e['tp']}, FP={m_e['fp']}, FN={m_e['fn']}, TN={m_e['tn']}")
    assert abs(m_e['dice'] - 0.5) < 1e-5, f"Expected Case E Dice 0.5, got {m_e['dice']}"
    assert abs(m_e['iou'] - (50 / 150)) < 1e-5, f"Expected Case E IoU 0.3333, got {m_e['iou']}"
    assert abs(m_e['precision'] - 0.5) < 1e-5, f"Expected Case E Precision 0.5, got {m_e['precision']}"
    assert abs(m_e['recall'] - 0.5) < 1e-5, f"Expected Case E Recall 0.5, got {m_e['recall']}"
    assert m_e['is_tumor_slice'] == True, "Case E must be marked tumor slice"
    print("  CASE E: PASSED")

    # 6. ACCUMULATOR TEST (Batch of A, B, C, D, E)
    print("\n[ACCUMULATOR TEST] Batch Evaluation")
    batch_preds = torch.stack([pred_a, pred_b, pred_c, pred_d, pred_e]).unsqueeze(1)
    batch_gts = torch.stack([gt_a, gt_b, gt_c, gt_d, gt_e]).unsqueeze(1)
    acc = ConfusionMatrixAccumulator()
    acc.update(batch_preds, batch_gts, threshold=0.5)
    metrics = acc.get_metrics()
    print(f"  Val Global Dice: {metrics['val_global_dice']:.4f}")
    print(f"  Val Global IoU:  {metrics['val_global_iou']:.4f}")
    print(f"  Val Precision:   {metrics['val_precision']:.4f}")
    print(f"  Val Recall:      {metrics['val_recall']:.4f}")
    print(f"  Val Tumor Dice:  {metrics['val_tumor_dice']:.4f} (Mean over 3 tumor slices: B=0.0, D=1.0, E=0.5 => 0.5000)")
    print(f"  Tumor Slices:    {metrics['num_tumor_slices']}, Empty Slices: {metrics['num_empty_slices']}")

    assert metrics['num_tumor_slices'] == 3, f"Expected 3 tumor slices, got {metrics['num_tumor_slices']}"
    assert metrics['num_empty_slices'] == 2, f"Expected 2 empty slices, got {metrics['num_empty_slices']}"
    assert abs(metrics['val_tumor_dice'] - 0.5) < 1e-5, f"Expected val_tumor_dice 0.5000, got {metrics['val_tumor_dice']}"
    print("  ACCUMULATOR TEST: PASSED")

    print("\n" + "=" * 60)
    print("ALL SYNTHETIC SEGMENTATION METRIC TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    run_synthetic_metric_tests()
