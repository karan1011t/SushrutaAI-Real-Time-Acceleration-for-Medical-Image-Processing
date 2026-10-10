"""
Focused Regression Test Suite for Live MRI Detection Consistency.

Tests all 8 requirements specified in Phase D of the Sushruta Audit:
1. Negative classifier + empty segmentation.
2. Negative classifier + non-empty segmentation (unconditional execution & transparent discrepancy).
3. Positive classifier + empty segmentation (transparent discrepancy & safe empty-mask XAI).
4. Positive classifier + non-empty segmentation (concordant positive detection).
5. Heatmap present without a positive detection (no false tumor claim from attribution).
6. Missing or unavailable heatmap (safe None handling & transparent metadata).
7. Decision boundary threshold handling (exact behavior at tau = 0.50).
8. API response schema and frontend status consistency across all states.
"""

import os
import sys
import io
import torch
import numpy as np
from PIL import Image
from unittest.mock import patch
from fastapi.testclient import TestClient

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from app.main import app
from app.schemas import PredictionResponse
from app.classifier_xai import ClassifierLayerCAMGenerator, generate_dynamic_explanation
from src.models.classifier import BinaryTumorClassifier

FIXTURE_POS = os.path.join(project_root, "data", "processed", "brats", "val", "images", "BraTS2021_00002_slice_063.png")
FIXTURE_NEG = os.path.join(project_root, "data", "processed", "brats", "val", "images", "BraTS2021_00002_slice_028.png")


def test_1_negative_classifier_empty_segmentation():
    """Scenario 1: Negative classifier + empty segmentation (Normal slice)."""
    print("\n[TEST 1] Negative classifier + empty segmentation...", end=" ", flush=True)
    with TestClient(app) as client:
        with open(FIXTURE_NEG, "rb") as f:
            resp = client.post("/predict", files={"file": ("slice_028.png", f.read(), "image/png")})
        assert resp.status_code == 200
        data = resp.json()

        # Both models agree: no tumor
        cls = data["classification"]
        meas = data["measurements"]
        outputs = data["outputs"]

        assert cls["tumor_detected"] is False
        assert cls["tumor_probability"] < 0.50
        assert cls["classification_label"] == "No Tumor Detected"
        assert "No tumor indicated" in cls["explanation"]

        assert meas["tumor_detected"] is False
        assert meas["area_pixels"] == 0
        assert data["tumor_detected"] is False

        # Classifier XAI must not report a fake focal lesion
        xai = cls["xai"]
        assert xai["strongest_region_location"] == "none"
        assert xai["bounding_box"] == {"x": 0, "y": 0, "width": 0, "height": 0}

        # U-Net LayerCAM safely None
        assert outputs["layercam"] is None
        assert outputs["layercam_overlay"] is None
        assert "Unavailable" in data["explainability"]["target_definition"]

        print("PASSED")


def test_2_negative_classifier_nonempty_segmentation():
    """Scenario 2: Negative classifier + non-empty segmentation (Discrepancy: U-Net detected pixels)."""
    print("[TEST 2] Negative classifier + non-empty segmentation...", end=" ", flush=True)
    with TestClient(app) as client:
        # Mock classifier to output negative probability (P = 0.15) while running real positive slice through U-Net
        from app import main as app_main
        orig_gen = app_main.classifier_cam_gen.generate_cam

        def mock_generate_cam(tensor, decision_threshold=0.5):
            H, W = tensor.shape[2], tensor.shape[3]
            cam_norm = np.zeros((H, W), dtype=np.float32)
            prob = 0.15
            logit = -1.734
            loc_info = {
                'location': 'none',
                'centroid': {'x': 0.0, 'y': 0.0},
                'bounding_box': {'x': 0, 'y': 0, 'width': 0, 'height': 0}
            }
            return cam_norm, prob, logit, loc_info

        with patch.object(app_main.classifier_cam_gen, "generate_cam", side_effect=mock_generate_cam):
            with open(FIXTURE_POS, "rb") as f:
                resp = client.post("/predict", files={"file": ("slice_063.png", f.read(), "image/png")})
            assert resp.status_code == 200
            data = resp.json()

            cls = data["classification"]
            meas = data["measurements"]
            outputs = data["outputs"]

            # Classifier is negative
            assert cls["tumor_detected"] is False
            assert cls["tumor_probability"] == 0.15
            assert cls["classification_label"] == "No Tumor Detected"

            # But U-Net segmentation was UNCONDITIONALLY executed and detected tumor pixels!
            assert meas["tumor_detected"] is True
            assert meas["area_pixels"] > 0
            assert outputs["tumor_mask"] is not None
            assert outputs["overlay"] is not None
            assert outputs["layercam_overlay"] is not None

            # Transparent discrepancy: models disagree
            assert cls["tumor_detected"] != meas["tumor_detected"]

        print("PASSED")


def test_3_positive_classifier_empty_segmentation():
    """Scenario 3: Positive classifier + empty segmentation (Discrepancy: Classifier indicated tumor)."""
    print("[TEST 3] Positive classifier + empty segmentation...", end=" ", flush=True)
    with TestClient(app) as client:
        from app import main as app_main

        def mock_generate_cam(tensor, decision_threshold=0.5):
            H, W = tensor.shape[2], tensor.shape[3]
            cam_norm = np.ones((H, W), dtype=np.float32) * 0.8
            prob = 0.88
            logit = 1.99
            loc_info = {
                'location': 'center',
                'centroid': {'x': 128.0, 'y': 128.0},
                'bounding_box': {'x': 32, 'y': 32, 'width': 192, 'height': 192}
            }
            return cam_norm, prob, logit, loc_info

        with patch.object(app_main.classifier_cam_gen, "generate_cam", side_effect=mock_generate_cam):
            # Run on normal slice (which has 0 U-Net pixels)
            with open(FIXTURE_NEG, "rb") as f:
                resp = client.post("/predict", files={"file": ("slice_028.png", f.read(), "image/png")})
            assert resp.status_code == 200
            data = resp.json()

            cls = data["classification"]
            meas = data["measurements"]
            outputs = data["outputs"]

            # Classifier is positive
            assert cls["tumor_detected"] is True
            assert cls["tumor_probability"] == 0.88
            assert cls["classification_label"] == "Tumor Detected"

            # U-Net segmented 0 pixels above threshold
            assert meas["tumor_detected"] is False
            assert meas["area_pixels"] == 0
            assert data["tumor_detected"] is False

            # U-Net LayerCAM safely unavailable (no fabricated tumor gradients)
            assert outputs["layercam"] is None
            assert outputs["layercam_overlay"] is None
            assert "Unavailable" in data["explainability"]["target_definition"]

            # Discrepancy confirmed
            assert cls["tumor_detected"] != meas["tumor_detected"]

        print("PASSED")


def test_4_positive_classifier_nonempty_segmentation():
    """Scenario 4: Positive classifier + non-empty segmentation (Concordant tumor)."""
    print("[TEST 4] Positive classifier + non-empty segmentation...", end=" ", flush=True)
    with TestClient(app) as client:
        with open(FIXTURE_POS, "rb") as f:
            resp = client.post("/predict", files={"file": ("slice_063.png", f.read(), "image/png")})
        assert resp.status_code == 200
        data = resp.json()

        cls = data["classification"]
        meas = data["measurements"]
        outputs = data["outputs"]

        # Both models positive
        assert cls["tumor_detected"] is True
        assert cls["tumor_probability"] >= 0.50
        assert cls["classification_label"] == "Tumor Detected"

        assert meas["tumor_detected"] is True
        assert meas["area_pixels"] > 0
        assert data["tumor_detected"] is True

        # All primary artifacts present
        assert outputs["preprocessed_mri"] is not None
        assert outputs["tumor_mask"] is not None
        assert outputs["overlay"] is not None
        assert outputs["layercam_overlay"] is not None
        assert outputs["classifier_layercam"] is not None

        print("PASSED")


def test_5_heatmap_present_without_positive_detection():
    """Scenario 5: Heatmap artifact present without a positive detection."""
    print("[TEST 5] Heatmap present without a positive detection...", end=" ", flush=True)
    with TestClient(app) as client:
        with open(FIXTURE_NEG, "rb") as f:
            resp = client.post("/predict", files={"file": ("slice_028.png", f.read(), "image/png")})
        assert resp.status_code == 200
        data = resp.json()

        outputs = data["outputs"]
        cls = data["classification"]
        meas = data["measurements"]

        # Continuous probability map artifact is generated and accessible
        assert outputs["probability_map"] is not None
        assert outputs["classifier_layercam"] is not None

        # BUT this must NOT trigger a positive diagnosis or positive detection flag
        assert data["tumor_detected"] is False
        assert cls["tumor_detected"] is False
        assert meas["tumor_detected"] is False
        assert "No tumor indicated" in cls["explanation"]

        print("PASSED")


def test_6_missing_or_unavailable_heatmap():
    """Scenario 6: Missing or unavailable heatmap gracefully handled."""
    print("[TEST 6] Missing or unavailable heatmap safe handling...", end=" ", flush=True)
    with TestClient(app) as client:
        # Synthetic all-black image produces 0 U-Net pixels -> U-Net LayerCAM must be safely unavailable
        black_img = Image.fromarray(np.zeros((256, 256), dtype=np.uint8))
        buf = io.BytesIO()
        black_img.save(buf, format="PNG")
        buf.seek(0)

        resp = client.post("/predict", files={"file": ("synthetic_black.png", buf.getvalue(), "image/png")})
        assert resp.status_code == 200
        data = resp.json()

        outputs = data["outputs"]
        xai = data["explainability"]

        # U-Net LayerCAM is unavailable (empty mask cannot backpropagate gradients)
        assert outputs["layercam"] is None
        assert outputs["layercam_overlay"] is None
        assert outputs["combined_overlay"] is None
        assert "Unavailable" in xai["target_definition"]
        assert xai["layercam_time_ms"] == 0.0

        # Normal slice has negative classifier -> Classifier XAI location is none
        with open(FIXTURE_NEG, "rb") as f:
            resp_neg = client.post("/predict", files={"file": ("slice_028.png", f.read(), "image/png")})
        assert resp_neg.status_code == 200
        data_neg = resp_neg.json()
        cls_xai_neg = data_neg["classification"]["xai"]
        assert cls_xai_neg["strongest_region_location"] == "none"
        assert cls_xai_neg["bounding_box"] == {"x": 0, "y": 0, "width": 0, "height": 0}

        print("PASSED")


def test_7_decision_boundary_threshold_handling():
    """Scenario 7: Correct threshold handling at the decision boundary (tau = 0.50)."""
    print("[TEST 7] Decision boundary threshold handling (tau = 0.50)...", end=" ", flush=True)
    # Test classifier dynamic explanation logic
    exp_below = generate_dynamic_explanation(
        tumor_detected=False,
        tumor_probability=0.4999,
        decision_threshold=0.50,
        location_str="center"
    )
    assert "No tumor indicated" in exp_below
    assert "50.0%" in exp_below

    exp_at = generate_dynamic_explanation(
        tumor_detected=True,
        tumor_probability=0.5000,
        decision_threshold=0.50,
        location_str="center"
    )
    assert "Tumor indicated" in exp_at
    assert "50.0%" in exp_at

    exp_above = generate_dynamic_explanation(
        tumor_detected=True,
        tumor_probability=0.5001,
        decision_threshold=0.50,
        location_str="center"
    )
    assert "Tumor indicated" in exp_above

    # Test ClassifierLayerCAMGenerator thresholding on both sides of boundary
    model = BinaryTumorClassifier()
    gen = ClassifierLayerCAMGenerator(model, device='cpu')

    dummy = torch.zeros((1, 1, 256, 256), dtype=torch.float32)
    # When threshold is higher than dummy prob -> sub-threshold (location == 'none')
    cam_norm_sub, prob_sub, _, loc_sub = gen.generate_cam(dummy, decision_threshold=0.999)
    assert loc_sub["location"] == "none"
    assert loc_sub["bounding_box"] == {"x": 0, "y": 0, "width": 0, "height": 0}

    # When threshold is lower than dummy prob -> detected (location != 'none')
    cam_norm_det, prob_det, _, loc_det = gen.generate_cam(dummy, decision_threshold=0.10)
    assert loc_det["location"] != "none"
    assert loc_det["bounding_box"]["width"] > 0

    print("PASSED")


def test_8_api_response_and_frontend_status_consistency():
    """Scenario 8: API response schema and frontend status consistency."""
    print("[TEST 8] API response schema and frontend status consistency...", end=" ", flush=True)
    with TestClient(app) as client:
        with open(FIXTURE_POS, "rb") as f:
            resp = client.post("/predict", files={"file": ("slice_063.png", f.read(), "image/png")})
        assert resp.status_code == 200
        data = resp.json()

        # Validate against full PredictionResponse Pydantic contract
        validated = PredictionResponse(**data)
        assert validated.success is True

        # Check all fields that frontend reads
        assert hasattr(validated, "tumor_detected")
        assert hasattr(validated, "classification")
        assert hasattr(validated.classification, "tumor_detected")
        assert hasattr(validated.classification, "tumor_probability")
        assert hasattr(validated.classification, "decision_threshold")
        assert hasattr(validated.classification, "classification_label")
        assert hasattr(validated.classification, "explanation")
        assert hasattr(validated.classification, "xai")
        assert hasattr(validated.segmentation, "performed")
        assert hasattr(validated.measurements, "area_pixels")
        assert hasattr(validated.measurements, "area_percentage")
        assert hasattr(validated.outputs, "preprocessed_mri")
        assert hasattr(validated.outputs, "classifier_layercam")
        assert hasattr(validated.outputs, "tumor_mask")
        assert hasattr(validated.outputs, "overlay")
        assert hasattr(validated.outputs, "probability_map")

        # Frontend agreement logic: (clsTumorDetected === segHasTumor)
        cls_flag = data["classification"]["tumor_detected"]
        seg_flag = data["measurements"]["area_pixels"] > 0
        agreement = (cls_flag == seg_flag)
        assert agreement is True  # For slice_063 both are positive

        print("PASSED")


if __name__ == "__main__":
    print("=" * 65)
    print("SUSHRUTA LIVE DETECTION CONSISTENCY REGRESSION SUITE")
    print("=" * 65)

    test_1_negative_classifier_empty_segmentation()
    test_2_negative_classifier_nonempty_segmentation()
    test_3_positive_classifier_empty_segmentation()
    test_4_positive_classifier_nonempty_segmentation()
    test_5_heatmap_present_without_positive_detection()
    test_6_missing_or_unavailable_heatmap()
    test_7_decision_boundary_threshold_handling()
    test_8_api_response_and_frontend_status_consistency()

    print("\n" + "=" * 65)
    print("ALL 8 FOCUSED REGRESSION SCENARIOS PASSED CLEANLY")
    print("=" * 65)
