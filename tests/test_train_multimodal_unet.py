"""
Automated Unit Test Suite for 4-Channel MultiModal U-Net Training Engine.
Uses synthetic PyTorch tensors to verify model shapes, loss, metrics, and single-batch optimization step.
Does NOT train on real dataset files.
"""

import os
import sys
import unittest
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models.unet import UNet2D
from src.models.train_multimodal_unet import (
    CombinedDiceBCELoss,
    compute_batch_segmentation_metrics,
    MULTIMODAL_CHANNEL_ORDER
)


class TestTrainMultiModalUNet(unittest.TestCase):

    def test_multimodal_unet_input_output_shapes(self):
        """Verifies 4-channel input shape (B, 4, H, W) produces (B, 1, H, W) logits."""
        model = UNet2D(in_channels=4, out_channels=1, base_channels=16)
        dummy_input = torch.randn(2, 4, 128, 128, dtype=torch.float32)

        logits = model(dummy_input)

        self.assertEqual(logits.shape, (2, 1, 128, 128), "Output logits must match (B, 1, H, W)")
        self.assertEqual(logits.dtype, torch.float32, "Logits must be float32")
        self.assertFalse(torch.isnan(logits).any(), "Logits must contain no NaN values")

    def test_combined_loss_computation_and_gradients(self):
        """Verifies loss computation receives raw logits and computes non-zero gradients."""
        model = UNet2D(in_channels=4, out_channels=1, base_channels=16)
        criterion = CombinedDiceBCELoss(weight_dice=0.5, weight_bce=0.5)

        dummy_input = torch.randn(2, 4, 64, 64)
        targets = torch.randint(0, 2, (2, 1, 64, 64)).float()

        logits = model(dummy_input)
        loss = criterion(logits, targets)

        self.assertTrue(torch.isfinite(loss), "Loss must be finite")
        self.assertGreater(loss.item(), 0.0, "Loss must be positive")

        loss.backward()
        param = next(model.parameters())
        self.assertIsNotNone(param.grad, "Gradients must be computed")
        self.assertGreater(param.grad.abs().sum().item(), 0.0, "Gradients must be non-zero")

    def test_batch_segmentation_metrics_edge_cases(self):
        """Tests metrics calculation on perfect match, zero overlap, and empty target."""
        # 1. Perfect match
        logits_perfect = torch.ones(1, 1, 32, 32) * 10.0 # sigmoid(10) ~ 1.0
        targets_perfect = torch.ones(1, 1, 32, 32)
        m_perf = compute_batch_segmentation_metrics(logits_perfect, targets_perfect)
        self.assertAlmostEqual(m_perf["dice"], 1.0, places=3)
        self.assertAlmostEqual(m_perf["iou"], 1.0, places=3)

        # 2. Zero overlap
        logits_zero = torch.ones(1, 1, 32, 32) * -10.0 # sigmoid(-10) ~ 0.0
        targets_nonzero = torch.ones(1, 1, 32, 32)
        m_zero = compute_batch_segmentation_metrics(logits_zero, targets_nonzero)
        self.assertLess(m_zero["dice"], 0.01)

        # 3. Empty target and empty prediction (no division by zero)
        logits_empty = torch.ones(1, 1, 32, 32) * -10.0
        targets_empty = torch.zeros(1, 1, 32, 32)
        m_empty = compute_batch_segmentation_metrics(logits_empty, targets_empty)
        self.assertTrue(torch.isfinite(torch.tensor(m_empty["dice"])))

    def test_single_batch_synthetic_smoke_test(self):
        """Executes 1 optimization step on synthetic 4-channel batch."""
        model = UNet2D(in_channels=4, out_channels=1, base_channels=16)
        criterion = CombinedDiceBCELoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        dummy_imgs = torch.randn(2, 4, 64, 64)
        dummy_masks = torch.randint(0, 2, (2, 1, 64, 64)).float()

        # Step 1
        optimizer.zero_grad()
        out1 = model(dummy_imgs)
        loss1 = criterion(out1, dummy_masks)
        loss1.backward()
        optimizer.step()

        # Step 2
        optimizer.zero_grad()
        out2 = model(dummy_imgs)
        loss2 = criterion(out2, dummy_masks)

        self.assertLess(loss2.item(), loss1.item(), "Loss should decrease after 1 optimization step on fixed synthetic batch")

    def test_multimodal_channel_configuration_metadata(self):
        """Verifies multimodal channel ordering metadata."""
        self.assertEqual(MULTIMODAL_CHANNEL_ORDER, ["FLAIR", "T1", "T1CE", "T2"])


if __name__ == "__main__":
    unittest.main()
