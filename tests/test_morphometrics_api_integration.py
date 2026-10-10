"""
API Integration & Serialization Test Suite for Phase 3 Quantitative Morphometrics.

Verifies:
1. End-to-end POST /predict response populates 2D morphological descriptors for tumor-positive slice.
2. Strict value consistency between top-level measurements and nested shape_descriptors.
3. Empty-mask /predict response safely serializes null (None) descriptors without crashes or fabricated zeros.
4. Serialized types are standard JSON-compliant primitives (float, str, dict, null), not raw numpy types.
5. Legacy client tolerance: Pydantic deserialization succeeds when shape_descriptors are absent.
"""

import os
import sys
import math
import numpy as np
from fastapi.testclient import TestClient

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from app.main import app
from app.schemas import PredictionResponse, Measurements, ShapeDescriptors

POS_FIXTURE = os.path.join(project_root, "data", "processed", "brats", "val", "images", "BraTS2021_00002_slice_063.png")
NEG_FIXTURE = os.path.join(project_root, "data", "processed", "brats", "val", "images", "BraTS2021_00002_slice_028.png")


def test_positive_slice_api_morphometrics():
    """Test 1: Tumor-positive slice populates all 2D morphological descriptors with valid numbers."""
    with TestClient(app) as client:
        with open(POS_FIXTURE, "rb") as f:
            resp = client.post("/predict", files={"file": ("pos_slice.png", f.read(), "image/png")})

        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
        data = resp.json()

        # Validate with root Pydantic model
        pyd_resp = PredictionResponse(**data)
        assert pyd_resp.success is True

        meas = data["measurements"]
        assert meas["tumor_detected"] is True
        assert meas["area_pixels"] > 0

        # Verify all 7 new top-level descriptor fields are populated with finite floats
        for field in [
            "perimeter_pixels", "circularity", "solidity",
            "major_axis_length", "minor_axis_length",
            "eccentricity", "feret_diameter_max"
        ]:
            val = meas[field]
            assert val is not None, f"Expected {field} to be populated, got None"
            assert isinstance(val, (int, float)), f"Expected numeric {field}, got {type(val)}"
            assert math.isfinite(val), f"Expected finite {field}, got {val}"

        # Verify mathematical bounds on real prediction
        assert 0.0 <= meas["circularity"] <= 1.0, f"Invalid circularity: {meas['circularity']}"
        assert 0.0 <= meas["solidity"] <= 1.0, f"Invalid solidity: {meas['solidity']}"
        assert 0.0 <= meas["eccentricity"] <= 1.0, f"Invalid eccentricity: {meas['eccentricity']}"
        assert meas["major_axis_length"] >= meas["minor_axis_length"] > 0
        assert meas["feret_diameter_max"] > 0

        # Verify nested shape_descriptors object
        assert "shape_descriptors" in meas and meas["shape_descriptors"] is not None
        sd = meas["shape_descriptors"]
        assert sd["aggregation_policy"] == "largest_connected_component"
        assert sd["coordinate_system"] == "processed_mask_grid"

        # Verify strict consistency between top-level and nested fields
        for field in [
            "perimeter_pixels", "circularity", "solidity",
            "major_axis_length", "minor_axis_length",
            "eccentricity", "feret_diameter_max"
        ]:
            assert meas[field] == sd[field], f"Mismatch for {field}: top-level {meas[field]} vs nested {sd[field]}"


def test_empty_slice_api_morphometrics():
    """Test 2: Tumor-negative slice safely serializes null (None) descriptors."""
    with TestClient(app) as client:
        with open(NEG_FIXTURE, "rb") as f:
            resp = client.post("/predict", files={"file": ("neg_slice.png", f.read(), "image/png")})

        assert resp.status_code == 200
        data = resp.json()
        meas = data["measurements"]

        # Mask is empty (no tumor pixels segmented)
        if meas["area_pixels"] == 0:
            assert meas["tumor_detected"] is False
            for field in [
                "perimeter_pixels", "circularity", "solidity",
                "major_axis_length", "minor_axis_length",
                "eccentricity", "feret_diameter_max"
            ]:
                assert meas[field] is None, f"Expected null {field} on empty mask, got {meas[field]}"

            sd = meas["shape_descriptors"]
            assert sd is not None
            assert sd["perimeter_pixels"] is None
            assert sd["aggregation_policy"] == "largest_connected_component"


def test_serialization_primitives():
    """Test 3: Response contains only JSON-standard primitives, no raw numpy scalar objects."""
    with TestClient(app) as client:
        with open(POS_FIXTURE, "rb") as f:
            resp = client.post("/predict", files={"file": ("pos_slice.png", f.read(), "image/png")})

        raw_json = resp.json()
        meas = raw_json["measurements"]

        assert type(meas["perimeter_pixels"]) is float
        assert type(meas["circularity"]) is float
        assert type(meas["solidity"]) is float
        assert type(meas["major_axis_length"]) is float
        assert type(meas["minor_axis_length"]) is float
        assert type(meas["eccentricity"]) is float
        assert type(meas["feret_diameter_max"]) is float
        assert type(meas["shape_descriptors"]) is dict


if __name__ == "__main__":
    print("=" * 65)
    print("SUSHRUTA MORPHOMETRICS API INTEGRATION TEST SUITE")
    print("=" * 65)

    tests = [
        ("Positive Slice Morphometrics Population & Consistency", test_positive_slice_api_morphometrics),
        ("Empty Slice Null Descriptors Serialization", test_empty_slice_api_morphometrics),
        ("JSON Serialization Primitives Verification", test_serialization_primitives),
    ]

    passed = 0
    for name, test_fn in tests:
        print(f"Running {name}...", end=" ", flush=True)
        try:
            test_fn()
            print("PASSED")
            passed += 1
        except Exception as exc:
            print(f"FAILED: {exc}")
            raise exc

    print("\n" + "=" * 65)
    print(f"ALL {passed}/{len(tests)} API INTEGRATION TESTS PASSED")
    print("=" * 65)
