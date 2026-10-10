"""
SUSHRUTA — Part 5 Failure-Path Integrity & Reproducibility Test Suite.

Validates:
1. Non-finite classifier output (NaN/Inf) explicitly returns HTTP 500 Classification Failure
   and is NOT silently converted to an ordinary negative prediction.
2. Non-finite segmentation output (NaN/Inf) explicitly returns HTTP 500 Inference Failure
   and is NOT silently converted to an apparently valid empty mask.
3. Legitimate empty segmentation masks (from normal slices) return valid HTTP 200 with zero
   measurements and transparent 'Unavailable (No Tumor Pixels Segmented)' explainability.
4. Explainability failure does NOT alter or corrupt the valid model prediction or segmentation mask.
5. Repeated inference runs on known positive and negative fixtures are deterministic.
6. Checkpoint files remain strictly immutable across inference requests.
7. Instantiating inference engine with missing checkpoint raises FileNotFoundError.
8. Invalid payloads (0-byte, wrong extension, corrupted bytes) are deterministically rejected.
"""

import os
import sys
import io
import hashlib
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
from app.inference import MedicalImageSegmenter
from app.classifier_xai import ClassifierLayerCAMGenerator
from app.explainability import LayerCAMGenerator

FIXTURE_POS = os.path.join(project_root, "data", "processed", "brats", "val", "images", "BraTS2021_00002_slice_063.png")
FIXTURE_NEG = os.path.join(project_root, "data", "processed", "brats", "val", "images", "BraTS2021_00002_slice_028.png")


def test_1_nan_classifier_output_returns_explicit_http_500():
    """Verify non-finite classifier output returns HTTP 500, not a silent negative diagnosis."""
    print("\n[TEST 1] Non-finite classifier output returns explicit HTTP 500...", end=" ", flush=True)
    with TestClient(app) as client:
        from app import main as app_main

        def mock_generate_cam_nan(*args, **kwargs):
            raise ValueError("Stage 1 classifier model output is non-finite (NaN or Inf). Inference corrupted.")

        with patch.object(app_main.classifier_cam_gen, "generate_cam", side_effect=mock_generate_cam_nan):
            with open(FIXTURE_POS, "rb") as f:
                resp = client.post("/predict", files={"file": ("slice_pos.png", f.read(), "image/png")})

            assert resp.status_code == 500, f"Expected HTTP 500, got {resp.status_code}"
            data = resp.json()
            assert data["success"] is False
            assert data["error"] == "Classification Failure"
            assert "non-finite (NaN or Inf)" in data["detail"]
            # Must NOT be an HTTP 200 claiming No Tumor Detected!
            assert "No Tumor Detected" not in str(data)

    print("PASSED")


def test_2_nan_segmentation_probabilities_return_explicit_http_500():
    """Verify non-finite U-Net output returns HTTP 500, not a silent empty mask."""
    print("[TEST 2] Non-finite U-Net probabilities return explicit HTTP 500...", end=" ", flush=True)
    with TestClient(app) as client:
        from app import main as app_main

        def mock_segment_nan(*args, **kwargs):
            raise ValueError("U-Net segmentation model produced non-finite probabilities (NaN or Inf). Inference corrupted.")

        with patch.object(app_main.segmenter, "segment", side_effect=mock_segment_nan):
            with open(FIXTURE_POS, "rb") as f:
                resp = client.post("/predict", files={"file": ("slice_pos.png", f.read(), "image/png")})

            assert resp.status_code == 500, f"Expected HTTP 500, got {resp.status_code}"
            data = resp.json()
            assert data["success"] is False
            assert data["error"] == "Inference Failure"
            assert "non-finite probabilities (NaN or Inf)" in data["detail"]
            # Must NOT be an HTTP 200 with area_pixels=0!
            assert "measurements" not in data

    print("PASSED")


def test_3_legitimate_empty_segmentation_mask_returns_valid_http_200():
    """Verify legitimate empty segmentation masks return HTTP 200 with zero metrics and unavailable XAI."""
    print("[TEST 3] Legitimate empty segmentation mask returns valid HTTP 200...", end=" ", flush=True)
    with TestClient(app) as client:
        with open(FIXTURE_NEG, "rb") as f:
            resp = client.post("/predict", files={"file": ("slice_neg.png", f.read(), "image/png")})

        assert resp.status_code == 200
        data = resp.json()
        PredictionResponse(**data)

        meas = data["measurements"]
        outputs = data["outputs"]
        xai = data["explainability"]

        # Valid inference with no segmented foreground
        assert meas["tumor_detected"] is False
        assert meas["area_pixels"] == 0
        assert meas["area_percentage"] == 0.0
        assert meas["num_connected_components"] == 0
        assert meas["perimeter_pixels"] is None
        assert meas["circularity"] is None
        assert meas["solidity"] is None
        assert data["segmentation_confidence"] == 0.0

        # LayerCAM safely unavailable (no fabricated tumor heatmaps)
        assert outputs["layercam"] is None
        assert outputs["layercam_overlay"] is None
        assert "Unavailable" in xai["target_definition"]

    print("PASSED")


def test_4_explainability_failure_does_not_corrupt_model_prediction():
    """Verify explainability failure returns valid prediction and surfaces XAI error state."""
    print("[TEST 4] Explainability failure preserves valid prediction...", end=" ", flush=True)
    with TestClient(app) as client:
        from app import main as app_main

        def mock_generate_cam_error(*args, **kwargs):
            raise RuntimeError("Simulated LayerCAM backward pass error")

        with patch.object(app_main.layercam_gen, "generate_cam", side_effect=mock_generate_cam_error):
            with open(FIXTURE_POS, "rb") as f:
                resp = client.post("/predict", files={"file": ("slice_pos.png", f.read(), "image/png")})

            assert resp.status_code == 200
            data = resp.json()
            PredictionResponse(**data)

            meas = data["measurements"]
            outputs = data["outputs"]
            xai = data["explainability"]

            # Segmentation prediction is 100% PRESERVED
            assert meas["tumor_detected"] is True
            assert meas["area_pixels"] == 636
            assert meas["area_percentage"] == 0.97
            assert outputs["tumor_mask"] is not None
            assert outputs["overlay"] is not None

            # Only explainability is gracefully marked as error
            assert outputs["layercam"] is None
            assert outputs["layercam_overlay"] is None
            assert xai["target_definition"] == "Error during LayerCAM generation"

    print("PASSED")


def test_5_reproducibility_repeated_inference_determinism():
    """Verify that repeated inference runs on positive and negative fixtures are deterministic."""
    print("[TEST 5] Repeated inference runs are deterministic...", end=" ", flush=True)
    with TestClient(app) as client:
        # 3 runs on positive slice
        pos_runs = []
        for _ in range(3):
            with open(FIXTURE_POS, "rb") as f:
                r = client.post("/predict", files={"file": ("slice_pos.png", f.read(), "image/png")})
            assert r.status_code == 200
            pos_runs.append(r.json())

        # Check positive determinism
        base_p = pos_runs[0]
        for run in pos_runs[1:]:
            assert run["tumor_detected"] == base_p["tumor_detected"]
            assert run["classification"]["tumor_detected"] == base_p["classification"]["tumor_detected"]
            assert abs(run["classification"]["tumor_probability"] - base_p["classification"]["tumor_probability"]) < 1e-4
            assert abs(run["classification"]["tumor_logit"] - base_p["classification"]["tumor_logit"]) < 1e-4
            assert run["measurements"]["area_pixels"] == base_p["measurements"]["area_pixels"]
            assert run["measurements"]["num_connected_components"] == base_p["measurements"]["num_connected_components"]
            assert abs(run["measurements"]["perimeter_pixels"] - base_p["measurements"]["perimeter_pixels"]) < 1e-3
            assert abs(run["measurements"]["circularity"] - base_p["measurements"]["circularity"]) < 1e-4
            assert abs(run["segmentation_confidence"] - base_p["segmentation_confidence"]) < 1e-4

        # 3 runs on negative slice
        neg_runs = []
        for _ in range(3):
            with open(FIXTURE_NEG, "rb") as f:
                r = client.post("/predict", files={"file": ("slice_neg.png", f.read(), "image/png")})
            assert r.status_code == 200
            neg_runs.append(r.json())

        # Check negative determinism
        base_n = neg_runs[0]
        for run in neg_runs[1:]:
            assert run["tumor_detected"] == base_n["tumor_detected"]
            assert run["classification"]["tumor_detected"] == base_n["classification"]["tumor_detected"]
            assert abs(run["classification"]["tumor_probability"] - base_n["classification"]["tumor_probability"]) < 1e-4
            assert run["measurements"]["area_pixels"] == 0
            assert run["measurements"]["perimeter_pixels"] is None

        # Verify zero hook leakage
        from app import main as app_main
        assert len(app_main.classifier_cam_gen.target_layer._forward_hooks) == 0
        assert len(app_main.classifier_cam_gen.target_layer._backward_hooks) == 0
        assert len(app_main.layercam_gen.target_layer._forward_hooks) == 0
        assert len(app_main.layercam_gen.target_layer._backward_hooks) == 0

    print("PASSED")


def test_6_checkpoint_immutability_during_inference():
    """Verify that model checkpoint files on disk are never altered by inference."""
    print("[TEST 6] Checkpoint files remain strictly immutable...", end=" ", flush=True)
    ckpt_paths = [
        os.path.join(project_root, "results", "models", "baseline_01_clean_v2_best.pt"),
        os.path.join(project_root, "results", "models", "baseline_02_classifier_best.pt")
    ]

    hashes_before = []
    for p in ckpt_paths:
        with open(p, "rb") as f:
            hashes_before.append(hashlib.sha256(f.read()).hexdigest())

    with TestClient(app) as client:
        with open(FIXTURE_POS, "rb") as f:
            resp = client.post("/predict", files={"file": ("slice_pos.png", f.read(), "image/png")})
        assert resp.status_code == 200

    hashes_after = []
    for p in ckpt_paths:
        with open(p, "rb") as f:
            hashes_after.append(hashlib.sha256(f.read()).hexdigest())

    assert hashes_before == hashes_after, "Checkpoint hash modified during inference!"
    print("PASSED")


def test_7_missing_checkpoint_file_raises_filenotfound():
    """Verify that instantiating inference engine with missing checkpoint raises FileNotFoundError."""
    print("[TEST 7] Missing checkpoint raises FileNotFoundError...", end=" ", flush=True)
    fake_path = os.path.join(project_root, "results", "models", "non_existent_model_checkpoint.pt")
    try:
        MedicalImageSegmenter(checkpoint_path=fake_path)
        raise AssertionError("Expected FileNotFoundError on non-existent checkpoint path")
    except FileNotFoundError as exc:
        assert "Checkpoint not found" in str(exc)

    print("PASSED")


def test_8_input_rejection_payloads():
    """Verify deterministic rejection of empty, wrong-extension, and corrupt payloads."""
    print("[TEST 8] Input rejection payloads...", end=" ", flush=True)
    with TestClient(app) as client:
        # Empty 0-byte file
        resp_empty = client.post("/predict", files={"file": ("empty.png", b"", "image/png")})
        assert resp_empty.status_code == 400
        assert resp_empty.json()["error"] == "Empty File"

        # Unsupported extension (.txt)
        resp_txt = client.post("/predict", files={"file": ("document.txt", b"hello world", "text/plain")})
        assert resp_txt.status_code == 400
        assert resp_txt.json()["error"] == "Image Preprocessing Error"

        # Corrupted image bytes
        resp_corrupt = client.post("/predict", files={"file": ("corrupt.png", b"\x89PNG\r\n\x1a\ncorrupted_data_not_an_image", "image/png")})
        assert resp_corrupt.status_code in (400, 422)

    print("PASSED")


if __name__ == "__main__":
    print("=" * 70)
    print("RUNNING SUSHRUTA PART 5 FAILURE-PATH INTEGRITY & REPRODUCIBILITY SUITE")
    print("=" * 70)

    test_1_nan_classifier_output_returns_explicit_http_500()
    test_2_nan_segmentation_probabilities_return_explicit_http_500()
    test_3_legitimate_empty_segmentation_mask_returns_valid_http_200()
    test_4_explainability_failure_does_not_corrupt_model_prediction()
    test_5_reproducibility_repeated_inference_determinism()
    test_6_checkpoint_immutability_during_inference()
    test_7_missing_checkpoint_file_raises_filenotfound()
    test_8_input_rejection_payloads()

    print("\n" + "=" * 70)
    print("ALL 8 PART 5 FAILURE-PATH AND REPRODUCIBILITY TESTS PASSED CLEANLY")
    print("=" * 70)
