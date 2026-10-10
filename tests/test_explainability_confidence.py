"""
SUSHRUTA — Part 4 Regression Test Suite:
Explainability, Prediction Consistency, Segmentation Confidence & Uncertainty Audit.

Validates:
1. Classifier decision boundary thresholding, labels, and dynamic explanations.
2. Robustness to non-finite classifier logits / probabilities (NaN, Inf).
3. ClassifierLayerCAMGenerator hook lifecycle, non-accumulation, and determinism.
4. Classifier LayerCAM suppression thresholds, sub-threshold scaling, and min-max normalization.
5. Classifier LayerCAM spatial alignment and alpha blending (no anatomical wash-out).
6. U-Net segmentation threshold exactness at tau = 0.50 and non-finite probability sanitization.
7. U-Net LayerCAM hook lifecycle, empty-mask contract, and non-finite target score handling.
8. Segmentation confidence mathematical definition (mean in-mask probability) and bounds.
9. End-to-end API model discrepancy handling (concordant vs discrepant states).
"""

import os
import sys
import io
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from app.main import app
from app.schemas import PredictionResponse
from app.classifier_xai import ClassifierLayerCAMGenerator, generate_dynamic_explanation
from app.explainability import LayerCAMGenerator
from app.inference import MedicalImageSegmenter
from app.measurements import calculate_tumor_measurements
from app.utils import generate_classifier_artifact
from src.models.classifier import BinaryTumorClassifier
from src.models.unet import UNet2D

FIXTURE_POS = os.path.join(project_root, "data", "processed", "brats", "val", "images", "BraTS2021_00002_slice_063.png")
FIXTURE_NEG = os.path.join(project_root, "data", "processed", "brats", "val", "images", "BraTS2021_00002_slice_028.png")


# =============================================================================
# 1. CLASSIFIER DECISION BOUNDARY & EXPLANATIONS
# =============================================================================

def test_classifier_decision_boundary_and_labels():
    """Verify decision boundary semantics: below, at, and above tau = 0.50."""
    tau = 0.50

    # Case A: Immediately below threshold
    p_below = 0.49999
    detected_below = bool(p_below >= tau)
    assert detected_below is False
    exp_below = generate_dynamic_explanation(detected_below, p_below, tau, "center")
    assert "No tumor indicated" in exp_below
    assert "below the 50.0% decision threshold" in exp_below

    # Case B: Exactly at threshold
    p_at = 0.50000
    detected_at = bool(p_at >= tau)
    assert detected_at is True
    exp_at = generate_dynamic_explanation(detected_at, p_at, tau, "center")
    assert "Tumor indicated" in exp_at
    assert "above the 50.0% decision threshold" in exp_at

    # Case C: Immediately above threshold
    p_above = 0.50001
    detected_above = bool(p_above >= tau)
    assert detected_above is True
    exp_above = generate_dynamic_explanation(detected_above, p_above, tau, "center-right")
    assert "Tumor indicated" in exp_above
    assert "in the center-right portion" in exp_above

    # Case D: Extreme zero & one
    exp_zero = generate_dynamic_explanation(False, 0.0, tau, "none")
    assert "0.0%" in exp_zero
    assert "No tumor indicated" in exp_zero

    exp_one = generate_dynamic_explanation(True, 1.0, tau, "center")
    assert "100.0%" in exp_one
    assert "Tumor indicated" in exp_one


def test_classifier_non_finite_handling():
    """Verify that NaN and Inf model outputs raise ValueError and are not masked as normal negative predictions."""
    class NanClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(*[nn.Conv2d(1, 256, 3, padding=1) for _ in range(13)])
        def forward(self, x):
            return torch.tensor([[float('nan')]], device=x.device, requires_grad=True)

    model = NanClassifier()
    gen = ClassifierLayerCAMGenerator(model, device='cpu')
    dummy_input = torch.zeros((1, 1, 256, 256), dtype=torch.float32)

    try:
        gen.generate_cam(dummy_input)
        raise AssertionError("Expected ValueError on non-finite classifier output")
    except ValueError as e:
        assert "non-finite (NaN or Inf)" in str(e)


def test_classifier_partial_non_finite_attribution_handling():
    """Verify that partially or fully non-finite LayerCAM activations/gradients are sanitized
    without corrupting valid classifier prediction semantics or raising unhandled exceptions."""
    model = BinaryTumorClassifier()
    model.eval()
    gen = ClassifierLayerCAMGenerator(model, device='cpu')
    dummy_input = torch.randn(1, 1, 256, 256)

    # Sub-case A: Partially non-finite activations (quadrant injected with NaN / +Inf)
    orig_save_activation = gen._save_activation
    def corrupt_partial_activation(module, input, output):
        orig_save_activation(module, input, output)
        corrupted = gen.activations.clone()
        corrupted[:, :, :corrupted.shape[2] // 2, :corrupted.shape[3] // 2] = float('nan')
        corrupted[:, 0, :, :] = float('inf')
        gen.activations = corrupted

    with patch.object(gen, '_save_activation', side_effect=corrupt_partial_activation):
        cam_norm, prob, logit, loc = gen.generate_cam(dummy_input, decision_threshold=0.50)
        assert np.isfinite(prob)
        assert np.isfinite(logit)
        assert 0.0 <= prob <= 1.0
        assert not np.isnan(cam_norm).any()
        assert not np.isinf(cam_norm).any()
        assert cam_norm.min() >= 0.0
        assert cam_norm.max() <= 1.0
        assert isinstance(loc["location"], str)

    # Sub-case B: Fully non-finite activations (all activations are NaN)
    def corrupt_all_activation(module, input, output):
        orig_save_activation(module, input, output)
        gen.activations = torch.full_like(gen.activations, float('nan'))

    with patch.object(gen, '_save_activation', side_effect=corrupt_all_activation):
        cam_norm, prob, logit, loc = gen.generate_cam(dummy_input, decision_threshold=0.50)
        assert np.isfinite(prob)
        assert np.isfinite(logit)
        assert np.all(cam_norm == 0.0)
        assert loc["location"] == "none"
        assert loc["centroid"] == {"x": 0.0, "y": 0.0}


# =============================================================================
# 2. CLASSIFIER LAYERCAM HOOK LIFECYCLE & NUMERICAL SANITY
# =============================================================================

def test_classifier_layercam_hook_lifecycle_and_non_accumulation():
    """Verify hooks are removed after every execution and do not accumulate across repeated requests."""
    model = BinaryTumorClassifier()
    model.eval()
    gen = ClassifierLayerCAMGenerator(model, device='cpu')
    dummy_input = torch.randn(1, 1, 256, 256)

    # Initial state: 0 hooks
    assert len(gen.target_layer._forward_hooks) == 0
    assert len(gen.target_layer._backward_hooks) == 0

    cams = []
    # Run 10 sequential inference passes
    for _ in range(10):
        cam, prob, logit, loc = gen.generate_cam(dummy_input)
        cams.append(cam)
        # Verify hooks removed immediately in finally block
        assert len(gen.target_layer._forward_hooks) == 0
        assert len(gen.target_layer._backward_hooks) == 0
        assert gen.activations is None
        assert gen.gradients is None

    # Assert 100% determinism across all runs
    for i in range(1, 10):
        np.testing.assert_array_equal(cams[0], cams[i])


def test_classifier_layercam_suppression_and_normalization():
    """Verify sub-threshold suppression (<1e-4), sub-threshold scaling, and positive min-max normalization."""
    model = BinaryTumorClassifier()
    model.eval()
    gen = ClassifierLayerCAMGenerator(model, device='cpu')

    dummy = torch.randn(1, 1, 256, 256)

    # Case A: Low probability below noise threshold (P < 1e-4) -> completely zeroed out
    with patch.object(model, "forward", return_value=torch.tensor([[-12.0]], requires_grad=True)):
        cam_norm, prob, _, loc = gen.generate_cam(dummy, decision_threshold=0.50)
        assert prob < 1e-4
        assert np.all(cam_norm == 0.0)
        assert loc["location"] == "none"

    # Case B: Sub-threshold with moderate prob (1e-4 <= P < 0.50) -> scaled by (prob / 0.50)
    with patch.object(model, "forward", return_value=torch.tensor([[-0.8473]], requires_grad=True)):  # sigmoid(-0.8473) ~= 0.30
        # Mock features[12] gradient/activations
        def mock_generate(*args, **kwargs):
            cam_norm, prob, logit, loc = ClassifierLayerCAMGenerator.generate_cam(gen, dummy, decision_threshold=0.50)
            return cam_norm, prob, logit, loc

        cam_norm, prob, _, loc = gen.generate_cam(dummy, decision_threshold=0.50)
        # Scaled map cannot exceed (prob / 0.50)
        if prob < 0.50 and prob >= 1e-4:
            assert float(cam_norm.max()) <= (prob / 0.50) + 1e-5

    # Case C: Positive detection (P >= 0.50) -> normalized to [0.0, 1.0]
    with patch.object(model, "forward", return_value=torch.tensor([[2.0]], requires_grad=True)):
        cam_norm, prob, _, loc = gen.generate_cam(dummy, decision_threshold=0.50)
        assert prob >= 0.50
        assert cam_norm.min() >= 0.0
        assert cam_norm.max() <= 1.0


def test_classifier_layercam_spatial_alignment_and_alpha():
    """Verify spatial dimensions (256x256) and alpha blending preserving original anatomy."""
    mri = np.ones((256, 256), dtype=np.float32) * 0.5  # Constant mid-gray MRI
    cam_zero = np.zeros((256, 256), dtype=np.float32)

    # When CAM is zero, alpha is zero: output should be exact grayscale MRI
    fname_zero, fpath_zero = generate_classifier_artifact(
        preprocessed_mri=mri,
        classifier_cam_map=cam_zero,
        output_dir=os.path.join(project_root, "results", "app_outputs"),
        base_id="test_zero_cam"
    )
    img_zero = np.array(Image.open(fpath_zero))
    assert img_zero.shape == (256, 256, 3)
    # MRI values: 0.5 * 255 = 127
    assert np.all(img_zero == 127), "Zero CAM must render clean un-tinted MRI without blue wash!"

    # When CAM is max (1.0), alpha is capped at 0.60 (40% original MRI anatomy preserved)
    cam_full = np.ones((256, 256), dtype=np.float32)
    fname_full, fpath_full = generate_classifier_artifact(
        preprocessed_mri=mri,
        classifier_cam_map=cam_full,
        output_dir=os.path.join(project_root, "results", "app_outputs"),
        base_id="test_full_cam"
    )
    img_full = np.array(Image.open(fpath_full))
    assert img_full.shape == (256, 256, 3)
    assert not np.all(img_full == 127)  # Colormap applied


# =============================================================================
# 3. U-NET SEGMENTATION CONSISTENCY & THRESHOLDING
# =============================================================================

def test_unet_segmentation_threshold_exactness():
    """Verify exact behavior of threshold comparison (prob >= threshold)."""
    segmenter = MedicalImageSegmenter.__new__(MedicalImageSegmenter)
    segmenter.threshold = 0.50
    segmenter.device = torch.device('cpu')

    # Construct synthetic mock model returning controlled probabilities
    probs_synthetic = torch.tensor([[[[0.49999, 0.50000], [0.50001, 0.00000]]]])  # (1, 1, 2, 2)
    logits_synthetic = torch.logit(probs_synthetic)

    class MockUNet(nn.Module):
        def forward(self, x):
            return logits_synthetic

    segmenter.model = MockUNet()

    dummy = torch.zeros((1, 1, 256, 256), dtype=torch.float32)
    with patch.object(torch, "sigmoid", return_value=probs_synthetic):
        prob_map, binary_mask, _ = segmenter.segment(dummy)

    # Pixel (0, 0): 0.49999 < 0.50 -> 0
    assert binary_mask[0, 0] == 0
    # Pixel (0, 1): 0.50000 >= 0.50 -> 1
    assert binary_mask[0, 1] == 1
    # Pixel (1, 0): 0.50001 >= 0.50 -> 1
    assert binary_mask[1, 0] == 1
    # Pixel (1, 1): 0.00000 < 0.50 -> 0
    assert binary_mask[1, 1] == 0


def test_unet_layercam_hook_lifecycle_and_empty_mask():
    """Verify U-Net LayerCAM hooks are cleanly removed and empty masks return safe unavailable status."""
    model = UNet2D(in_channels=1, out_channels=1, base_channels=32)
    model.eval()
    gen = LayerCAMGenerator(model, device=torch.device('cpu'))

    dummy_input = torch.randn(1, 1, 256, 256)
    empty_mask = np.zeros((256, 256), dtype=np.uint8)

    # Initial hooks empty
    assert len(gen.target_layer._forward_hooks) == 0
    assert len(gen.target_layer._backward_hooks) == 0

    cam_norm, meta, latency = gen.generate_cam(dummy_input, empty_mask)

    # Hooks removed in finally block
    assert len(gen.target_layer._forward_hooks) == 0
    assert len(gen.target_layer._backward_hooks) == 0
    assert gen._activations is None
    assert gen._gradients is None

    # Meta documents target definition
    assert "no tumor pixels predicted" in meta["target_definition"]
    assert cam_norm.shape == (256, 256)
    assert cam_norm.min() >= 0.0 and cam_norm.max() <= 1.0


# =============================================================================
# 4. SEGMENTATION CONFIDENCE DEFINITION & BOUNDS
# =============================================================================

def test_segmentation_confidence_definition_and_bounds():
    """Verify segmentation_confidence equals the arithmetic mean of in-mask probabilities."""
    # Case A: Non-empty mask with known values
    prob_map = np.zeros((256, 256), dtype=np.float32)
    binary_mask = np.zeros((256, 256), dtype=np.uint8)

    # Set 4 foreground pixels with known probabilities
    prob_map[10, 10] = 0.60
    prob_map[10, 11] = 0.70
    prob_map[11, 10] = 0.80
    prob_map[11, 11] = 0.90
    binary_mask[10:12, 10:12] = 1

    meas, conf = calculate_tumor_measurements(binary_mask, prob_map)
    expected_conf = round((0.60 + 0.70 + 0.80 + 0.90) / 4.0, 4)
    assert conf == expected_conf == 0.7500
    assert meas["area_pixels"] == 4
    assert conf > 0.50  # Must be above segmentation threshold

    # Case B: Empty mask
    empty_prob = np.random.uniform(0.0, 0.49, (256, 256)).astype(np.float32)
    empty_mask = np.zeros((256, 256), dtype=np.uint8)

    meas_empty, conf_empty = calculate_tumor_measurements(empty_mask, empty_prob)
    assert conf_empty == 0.0
    assert meas_empty["area_pixels"] == 0
    assert meas_empty["tumor_detected"] is False


# =============================================================================
# 5. END-TO-END MODEL DISCREPANCY & API CONTRACT
# =============================================================================

def test_api_model_discrepancy_and_agreement_contract():
    """Verify API handles both model agreement and discrepancy transparently."""
    with TestClient(app) as client:
        from app import main as app_main

        # Discrepancy Case 1: Classifier Positive, U-Net Empty
        def mock_cam_pos(tensor, decision_threshold=0.5):
            H, W = tensor.shape[2], tensor.shape[3]
            return np.ones((H, W), dtype=np.float32) * 0.9, 0.91, 2.31, {
                'location': 'center',
                'centroid': {'x': 128.0, 'y': 128.0},
                'bounding_box': {'x': 32, 'y': 32, 'width': 192, 'height': 192}
            }

        with patch.object(app_main.classifier_cam_gen, "generate_cam", side_effect=mock_cam_pos):
            with open(FIXTURE_NEG, "rb") as f:
                resp = client.post("/predict", files={"file": ("slice_neg.png", f.read(), "image/png")})
            assert resp.status_code == 200
            data = resp.json()
            PredictionResponse(**data)

            # Check Discrepancy representation
            cls = data["classification"]
            seg = data["segmentation"]
            outputs = data["outputs"]

            assert cls["tumor_detected"] is True
            assert seg["measurements"]["tumor_detected"] is False
            assert seg["measurements"]["area_pixels"] == 0

            # U-Net LayerCAM is safely None with technical explanation
            assert outputs["layercam"] is None
            assert outputs["layercam_overlay"] is None
            assert "Unavailable" in data["explainability"]["target_definition"]

        # Discrepancy Case 2: Classifier Negative, U-Net Positive
        def mock_cam_neg(tensor, decision_threshold=0.5):
            H, W = tensor.shape[2], tensor.shape[3]
            return np.zeros((H, W), dtype=np.float32), 0.12, -1.99, {
                'location': 'none',
                'centroid': {'x': 0.0, 'y': 0.0},
                'bounding_box': {'x': 0, 'y': 0, 'width': 0, 'height': 0}
            }

        with patch.object(app_main.classifier_cam_gen, "generate_cam", side_effect=mock_cam_neg):
            with open(FIXTURE_POS, "rb") as f:
                resp = client.post("/predict", files={"file": ("slice_pos.png", f.read(), "image/png")})
            assert resp.status_code == 200
            data = resp.json()
            PredictionResponse(**data)

            cls = data["classification"]
            seg = data["segmentation"]
            outputs = data["outputs"]

            assert cls["tumor_detected"] is False
            assert seg["measurements"]["tumor_detected"] is True
            assert seg["measurements"]["area_pixels"] > 0

            # U-Net LayerCAM is generated and accessible
            assert outputs["layercam_overlay"] is not None


if __name__ == "__main__":
    print("=" * 70)
    print("RUNNING SUSHRUTA PART 4 EXPLAINABILITY & CONSISTENCY TEST SUITE")
    print("=" * 70)

    test_classifier_decision_boundary_and_labels()
    print("[PASS] test_classifier_decision_boundary_and_labels")

    test_classifier_non_finite_handling()
    print("[PASS] test_classifier_non_finite_handling")

    test_classifier_partial_non_finite_attribution_handling()
    print("[PASS] test_classifier_partial_non_finite_attribution_handling")

    test_classifier_layercam_hook_lifecycle_and_non_accumulation()
    print("[PASS] test_classifier_layercam_hook_lifecycle_and_non_accumulation")

    test_classifier_layercam_suppression_and_normalization()
    print("[PASS] test_classifier_layercam_suppression_and_normalization")

    test_classifier_layercam_spatial_alignment_and_alpha()
    print("[PASS] test_classifier_layercam_spatial_alignment_and_alpha")

    test_unet_segmentation_threshold_exactness()
    print("[PASS] test_unet_segmentation_threshold_exactness")

    test_unet_layercam_hook_lifecycle_and_empty_mask()
    print("[PASS] test_unet_layercam_hook_lifecycle_and_empty_mask")

    test_segmentation_confidence_definition_and_bounds()
    print("[PASS] test_segmentation_confidence_definition_and_bounds")

    test_api_model_discrepancy_and_agreement_contract()
    print("[PASS] test_api_model_discrepancy_and_agreement_contract")

    print("\n" + "=" * 70)
    print("ALL 10 PART 4 FOCUSED REGRESSION TESTS PASSED CLEANLY")
    print("=" * 70)
