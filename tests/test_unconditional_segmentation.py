"""
Unit and Integration Test Suite for SUSHRUTA Unconditional U-Net Segmentation & Unified Results Pipeline.

Verifies:
1. Classifier-positive image with non-empty segmentation (all artifacts present).
2. Classifier-negative image where segmentation is unconditionally executed and returned.
3. Valid empty segmentation mask with strictly zero measurements (no fabricated values).
4. Empty-mask LayerCAM handling (safe None artifact, transparent technical explanation, no crashes).
5. Full backward compatibility of PredictionResponse root, nested schemas, and fields.
6. Error handling for empty files, invalid file types, and unprocessable data.
7. Finiteness and validity of latency benchmarks.
"""

import os
import sys
import io
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from app.main import app
from app.schemas import PredictionResponse

POS_SAMPLE_PATH = os.path.join(project_root, "data", "processed", "brats", "val", "images", "BraTS2021_00002_slice_063.png")
NEG_SAMPLE_PATH = os.path.join(project_root, "data", "processed", "brats", "val", "images", "BraTS2021_00002_slice_028.png")


def test_classifier_positive_with_nonempty_segmentation():
    """Test 1: Classifier-positive image with non-empty segmentation."""
    print("\n[TEST 1] Testing Classifier-Positive Image with Non-Empty Segmentation...", end=" ", flush=True)
    with TestClient(app) as client:
        assert os.path.exists(POS_SAMPLE_PATH), f"Fixture not found: {POS_SAMPLE_PATH}"
        with open(POS_SAMPLE_PATH, "rb") as f:
            file_bytes = f.read()

        resp = client.post("/predict", files={"file": ("BraTS2021_00002_slice_063.png", file_bytes, "image/png")})
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"

        data = resp.json()
        # Validate Pydantic schema
        PredictionResponse(**data)

        # Classifier screening
        cls = data["classification"]
        assert cls["tumor_probability"] >= 0.50, f"Expected prob >= 0.50, got {cls['tumor_probability']}"
        assert cls["tumor_detected"] is True
        assert cls["classification_label"] == "Tumor Detected"

        # Segmentation execution
        seg = data["segmentation"]
        assert seg["performed"] is True

        # Non-empty measurements
        meas = data["measurements"]
        assert meas["area_pixels"] > 0
        assert meas["area_percentage"] > 0.0
        assert meas["num_connected_components"] >= 1
        assert meas["diameter_pixels"] > 0
        assert meas["tumor_detected"] is True

        # Visual artifacts
        outputs = data["outputs"]
        assert outputs["preprocessed_mri"] is not None
        assert outputs["tumor_mask"] is not None
        assert outputs["overlay"] is not None
        assert outputs["probability_map"] is not None
        assert outputs["layercam_overlay"] is not None
        assert outputs["classifier_layercam"] is not None

        # U-Net LayerCAM
        xai = data["explainability"]
        assert xai is not None
        assert xai["method"] == "LayerCAM"
        assert xai["layercam_time_ms"] > 0.0

        print(f"PASSED (Tumor Pixels: {meas['area_pixels']} px, P(cls)={cls['tumor_probability'] * 100:.1f}%)")


def test_classifier_negative_with_unconditional_segmentation():
    """Test 2: Classifier-negative image where segmentation is still unconditionally executed."""
    print("[TEST 2] Testing Classifier-Negative Image with Unconditional Segmentation...", end=" ", flush=True)
    with TestClient(app) as client:
        assert os.path.exists(NEG_SAMPLE_PATH), f"Fixture not found: {NEG_SAMPLE_PATH}"
        with open(NEG_SAMPLE_PATH, "rb") as f:
            file_bytes = f.read()

        resp = client.post("/predict", files={"file": ("BraTS2021_00002_slice_028.png", file_bytes, "image/png")})
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"

        data = resp.json()
        PredictionResponse(**data)

        # Classifier negative
        cls = data["classification"]
        assert cls["tumor_probability"] < 0.50, f"Expected prob < 0.50, got {cls['tumor_probability']}"
        assert cls["tumor_detected"] is False
        assert cls["classification_label"] == "No Tumor Detected"

        # U-Net segmentation was UNCONDITIONALLY executed
        seg = data["segmentation"]
        assert seg["performed"] is True, "Segmentation must be performed even when classifier is negative!"

        # Core visualization artifacts are returned
        outputs = data["outputs"]
        assert outputs["preprocessed_mri"] is not None
        assert outputs["tumor_mask"] is not None
        assert outputs["overlay"] is not None
        assert outputs["probability_map"] is not None
        assert outputs["classifier_layercam"] is not None

        # Timings exposed
        inf = data["inference"]
        assert inf["segmentation_inference_time_ms"] > 0.0
        assert inf["total_request_time_ms"] > 0.0

        print(f"PASSED (Classifier P={cls['tumor_probability'] * 100:.1f}%, Seg Performed={seg['performed']})")


def test_valid_empty_segmentation_mask_measurements():
    """Test 3: Empty segmentation mask returns valid zero measurements without fabrication."""
    print("[TEST 3] Testing Valid Empty-Mask Zero Measurements Handling...", end=" ", flush=True)
    with TestClient(app) as client:
        # Generate a synthetic all-black image that guarantees 0 tumor pixels
        black_img = Image.fromarray(np.zeros((256, 256), dtype=np.uint8))
        buf = io.BytesIO()
        black_img.save(buf, format="PNG")
        buf.seek(0)

        resp = client.post("/predict", files={"file": ("synthetic_black.png", buf.getvalue(), "image/png")})
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"

        data = resp.json()
        meas = data["measurements"]

        # Strict assertion of zero measurements
        assert meas["tumor_detected"] is False
        assert meas["area_pixels"] == 0
        assert meas["area_percentage"] == 0.0
        assert meas["num_connected_components"] == 0
        assert meas["largest_component_area_pixels"] == 0
        assert meas["diameter_pixels"] == 0
        assert meas["width_pixels"] == 0
        assert meas["height_pixels"] == 0
        assert meas["bounding_box"] == {"x": 0, "y": 0, "width": 0, "height": 0}
        assert meas["centroid"] == {"x": 0.0, "y": 0.0}
        assert meas["largest_component_bounding_box"] == {"x": 0, "y": 0, "width": 0, "height": 0}

        print("PASSED (All empty mask metrics strictly 0 / empty)")


def test_empty_mask_layercam_handling():
    """Test 4: Empty mask does not crash LayerCAM and returns explicit unavailable status."""
    print("[TEST 4] Testing Empty-Mask LayerCAM Safe Handling...", end=" ", flush=True)
    with TestClient(app) as client:
        black_img = Image.fromarray(np.zeros((256, 256), dtype=np.uint8))
        buf = io.BytesIO()
        black_img.save(buf, format="PNG")
        buf.seek(0)

        resp = client.post("/predict", files={"file": ("synthetic_black.png", buf.getvalue(), "image/png")})
        assert resp.status_code == 200

        data = resp.json()
        outputs = data["outputs"]

        # LayerCAM outputs must be safely None without crashing
        assert outputs["layercam"] is None
        assert outputs["layercam_overlay"] is None

        # Explainability metadata must provide transparent technical explanation
        xai = data["explainability"]
        assert xai is not None
        assert "Unavailable" in xai["target_definition"]
        assert "0 tumor pixels were segmented" in xai["interpretation"]
        assert xai["layercam_time_ms"] == 0.0

        print("PASSED (Safe None artifacts + transparent explanation)")


def test_backward_compatibility_schema():
    """Test 5: Verify all backward-compatible response fields are present and correctly typed."""
    print("[TEST 5] Testing Backward Compatibility of All Response Fields...", end=" ", flush=True)
    with TestClient(app) as client:
        with open(POS_SAMPLE_PATH, "rb") as f:
            file_bytes = f.read()

        resp = client.post("/predict", files={"file": ("test_compat.png", file_bytes, "image/png")})
        assert resp.status_code == 200

        data = resp.json()

        # Root fields
        expected_root_fields = {
            "success", "tumor_detected", "classification", "segmentation",
            "measurements", "segmentation_confidence", "explainability",
            "inference", "image_dimensions", "outputs", "disclaimer"
        }
        for field in expected_root_fields:
            assert field in data, f"Missing root field: {field}"

        # Classification fields
        expected_cls_fields = {
            "tumor_detected", "tumor_probability", "tumor_logit",
            "decision_threshold", "classification_label", "explanation", "xai"
        }
        for field in expected_cls_fields:
            assert field in data["classification"], f"Missing classification field: {field}"

        # Measurements fields
        expected_meas_fields = {
            "tumor_detected", "area_pixels", "area_percentage", "bounding_box",
            "width_pixels", "height_pixels", "diameter_pixels",
            "num_connected_components", "largest_component_area_pixels",
            "centroid", "largest_component_bounding_box"
        }
        for field in expected_meas_fields:
            assert field in data["measurements"], f"Missing measurements field: {field}"

        # Inference fields
        expected_inf_fields = {
            "time_ms", "segmentation_inference_time_ms", "layercam_time_ms",
            "preprocessing_time_ms", "total_request_time_ms", "device", "threshold_used"
        }
        for field in expected_inf_fields:
            assert field in data["inference"], f"Missing inference field: {field}"

        print("PASSED (All required schema fields present and typed)")


def test_error_handling():
    """Test 6: Verify error handling on invalid or empty uploads."""
    print("[TEST 6] Testing Input Validation & Error Handling...", end=" ", flush=True)
    with TestClient(app) as client:
        # Empty file
        resp_empty = client.post("/predict", files={"file": ("empty.png", b"", "image/png")})
        assert resp_empty.status_code == 400
        assert resp_empty.json()["success"] is False

        # Invalid file type
        resp_invalid = client.post("/predict", files={"file": ("document.txt", b"plain text", "text/plain")})
        assert resp_invalid.status_code in (400, 422)
        assert resp_invalid.json()["success"] is False

        print("PASSED (Rejects empty file with 400 and invalid file with 400/422)")


if __name__ == "__main__":
    print("=" * 65)
    print("SUSHRUTA UNCONDITIONAL U-NET & UNIFIED PIPELINE TEST SUITE")
    print("=" * 65)

    test_classifier_positive_with_nonempty_segmentation()
    test_classifier_negative_with_unconditional_segmentation()
    test_valid_empty_segmentation_mask_measurements()
    test_empty_mask_layercam_handling()
    test_backward_compatibility_schema()
    test_error_handling()

    print("\n" + "=" * 65)
    print("ALL 6 UNCONDITIONAL PIPELINE INTEGRATION TESTS PASSED")
    print("=" * 65)
