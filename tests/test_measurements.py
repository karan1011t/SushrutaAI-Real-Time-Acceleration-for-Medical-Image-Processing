"""
Focused Mathematical Test Suite for 2D Morphological Descriptors & Measurement Pipeline.

Verifies Phase 3A: Quantitative 2D Morphological Characterization:
1. Empty mask: all shape descriptors safely return None (no fabricated zeros).
2. Single-pixel mask: degenerate contour safely returns None for perimeter/axes/Feret, solidity=1.0.
3. Solid synthetic disk: circularity reasonably close to 1 within discrete rasterization tolerance (~0.85-0.95).
4. Filled square: exact comparison against justified raster-contour convention (cv2.arcLength P = 4*(L-1)).
5. Elongated rectangle: major axis > minor axis, eccentricity consistent with inertia axes.
6. Multi-component mask: aggregation policy evaluates primary mass (largest connected component).
7. Degenerate contours and zero-area hulls: safe None handling without crashes.
8. Arbitrary valid 2D dimensions: works on non-256x256 grids (e.g., 128x128, 512x512, 200x300).
9. Mismatched mask and probability-map dimensions: cleanly raises ValueError.
10. Non-finite input values (NaN / Inf): sanitized and handled gracefully.
11. Mathematical bounds: circularity, solidity, eccentricity within [0.0, 1.0].
12. Schema validation: output validates cleanly against Measurements Pydantic schema.
"""

import os
import sys
import math
import numpy as np
import cv2

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from app.measurements import calculate_tumor_measurements, _compute_shape_descriptors
from app.schemas import Measurements, ShapeDescriptors, BoundingBox


def test_empty_mask_behavior():
    """Test 1: Empty mask returns safe null/None descriptors and preserves existing fields."""
    mask = np.zeros((256, 256), dtype=np.uint8)
    prob = np.zeros((256, 256), dtype=np.float32)

    meas, conf = calculate_tumor_measurements(mask, prob)

    # Existing contract preserved
    assert meas["tumor_detected"] is False
    assert meas["area_pixels"] == 0
    assert meas["area_percentage"] == 0.0
    assert meas["num_connected_components"] == 0
    assert meas["largest_component_area_pixels"] == 0
    assert meas["diameter_pixels"] == 0
    assert meas["centroid"] == {"x": 0.0, "y": 0.0}
    assert conf == 0.0

    # New 2D morphological descriptors must be explicitly None (never fabricated zeros)
    assert meas["perimeter_pixels"] is None
    assert meas["circularity"] is None
    assert meas["solidity"] is None
    assert meas["major_axis_length"] is None
    assert meas["minor_axis_length"] is None
    assert meas["eccentricity"] is None
    assert meas["feret_diameter_max"] is None

    # Nested shape_descriptors schema matches
    sd = meas["shape_descriptors"]
    assert sd["perimeter_pixels"] is None
    assert sd["aggregation_policy"] == "largest_connected_component"
    assert sd["coordinate_system"] == "processed_mask_grid"

    # Validates with Pydantic schema
    Measurements(**meas)


def test_single_pixel_mask_degenerate():
    """Test 2: Single foreground pixel represents a degenerate contour; returns safe values."""
    mask = np.zeros((256, 256), dtype=np.uint8)
    mask[100, 100] = 1
    prob = np.zeros((256, 256), dtype=np.float32)
    prob[100, 100] = 0.85

    meas, conf = calculate_tumor_measurements(mask, prob)

    assert meas["tumor_detected"] is True
    assert meas["area_pixels"] == 1
    assert meas["num_connected_components"] == 1
    assert meas["largest_component_area_pixels"] == 1
    assert conf == 0.85

    # Degenerate 1-pixel component: boundary arc length is undefined / < 3 points
    assert meas["perimeter_pixels"] is None
    assert meas["circularity"] is None
    assert meas["solidity"] == 1.0  # 1 px / 1 px hull
    assert meas["major_axis_length"] is None
    assert meas["minor_axis_length"] is None
    assert meas["eccentricity"] is None
    assert meas["feret_diameter_max"] is None

    Measurements(**meas)


def test_solid_synthetic_disk():
    """Test 3: Solid synthetic disk circularity within justified rasterization tolerance."""
    mask = np.zeros((256, 256), dtype=np.uint8)
    r = 30
    cv2.circle(mask, (128, 128), r, 1, thickness=-1)
    prob = (mask * 0.95).astype(np.float32)

    meas, conf = calculate_tumor_measurements(mask, prob)

    assert meas["tumor_detected"] is True
    assert meas["num_connected_components"] == 1
    assert meas["area_pixels"] > 2500  # pi * 30^2 ~ 2827

    # Circularity: On a digital raster grid, discrete pixel boundary steps mean
    # perimeter deviates slightly from 2*pi*r, but circularity should be close to 1.0 (~0.85-0.95).
    circ = meas["circularity"]
    assert circ is not None
    assert 0.85 <= circ <= 1.0, f"Expected circularity near 1.0, got {circ}"

    # Solidity of discrete raster disk: boundary staircasing causes the polygon convex hull
    # to enclose slightly more area (~0.98), very close to ideal 1.0.
    assert meas["solidity"] is not None
    assert 0.97 <= meas["solidity"] <= 1.0, f"Expected disk solidity near 1.0, got {meas['solidity']}"

    # Major and minor axes of disk should be nearly identical (2*r ~ 60 px)
    major = meas["major_axis_length"]
    minor = meas["minor_axis_length"]
    assert major is not None and minor is not None
    assert abs(major - 2 * r) < 2.0
    assert abs(minor - 2 * r) < 2.0
    assert abs(major - minor) < 1.0

    # Eccentricity of disk should be near 0.0
    ecc = meas["eccentricity"]
    assert ecc is not None
    assert ecc < 0.15, f"Expected near-zero eccentricity for disk, got {ecc}"

    # Feret diameter should be 2*r = 60 px
    feret = meas["feret_diameter_max"]
    assert feret is not None
    assert abs(feret - 2 * r) <= 1.0

    Measurements(**meas)


def test_filled_square_raster_convention():
    """
    Test 4: Filled square tested against justified raster-contour convention.

    OpenCV findContours connects pixel centers. For a square of size L:
    - Side length between corner pixel centers is L - 1.
    - Raster perimeter = 4 * (L - 1).
    - For L = 40: perimeter = 4 * 39 = 156.0 px.
    - Continuous perimeter (4 * 40 = 160 px) is NOT asserted because raster contours connect pixel centers.
    - True Feret diameter = sqrt(39^2 + 39^2) = 39 * sqrt(2) ~ 55.15 px.
    - Bounding box dimension = 40 px.
    """
    mask = np.zeros((256, 256), dtype=np.uint8)
    L = 40
    mask[50:50 + L, 50:50 + L] = 1
    prob = (mask * 0.9).astype(np.float32)

    meas, _ = calculate_tumor_measurements(mask, prob)

    assert meas["area_pixels"] == L * L  # 1600 px

    # Justified raster contour perimeter: 4 * (40 - 1) = 156.0 px
    expected_perimeter = float(4 * (L - 1))
    assert meas["perimeter_pixels"] == expected_perimeter

    # Raster circularity: 4 * pi * 1600 / (156^2) ~ 0.8262
    expected_circ = round(4.0 * math.pi * (L * L) / (expected_perimeter ** 2), 4)
    assert meas["circularity"] == expected_circ

    # Solidity of filled square is 1.0
    assert meas["solidity"] == 1.0

    # Feret diameter is diagonal distance between corner pixel centers: (L-1)*sqrt(2) ~ 55.15 px
    expected_feret = round(float((L - 1) * math.sqrt(2)), 2)
    assert meas["feret_diameter_max"] == expected_feret

    # Feret diameter strictly differs from axis-aligned bounding box dimension (40 px)
    assert meas["diameter_pixels"] == L  # 40 px
    assert meas["feret_diameter_max"] > meas["diameter_pixels"]

    Measurements(**meas)


def test_elongated_rectangle():
    """Test 5: Elongated rectangle has major axis > minor axis and consistent eccentricity."""
    mask = np.zeros((256, 256), dtype=np.uint8)
    # 10 height, 60 width
    mask[100:110, 50:110] = 1
    prob = (mask * 0.9).astype(np.float32)

    meas, _ = calculate_tumor_measurements(mask, prob)

    major = meas["major_axis_length"]
    minor = meas["minor_axis_length"]
    ecc = meas["eccentricity"]

    assert major is not None and minor is not None and ecc is not None
    assert major > minor, f"Expected major ({major}) > minor ({minor})"

    # Consistency check: eccentricity = sqrt(1 - (minor/major)^2)
    expected_ecc = round(math.sqrt(1.0 - (minor / major) ** 2), 2)
    assert abs(round(ecc, 2) - expected_ecc) <= 0.05
    assert ecc > 0.85, f"Expected high eccentricity for 6:1 elongated rectangle, got {ecc}"

    Measurements(**meas)


def test_multi_component_aggregation_policy():
    """
    Test 6: Multi-component mask verifies aggregation policy.

    Aggregation policy:
    - area_pixels and area_percentage reflect total union across all components.
    - shape_descriptors reflect the primary tumor mass (largest connected component).
    """
    mask = np.zeros((256, 256), dtype=np.uint8)
    # Component 1 (Primary): 30x30 square = 900 px
    mask[20:50, 20:50] = 1
    # Component 2 (Satellite): 10x10 square = 100 px
    mask[150:160, 150:160] = 1
    # Component 3 (Tiny): 4x4 square = 16 px
    mask[200:204, 200:204] = 1

    prob = (mask * 0.9).astype(np.float32)

    meas, _ = calculate_tumor_measurements(mask, prob)

    # Union metrics
    assert meas["num_connected_components"] == 3
    assert meas["area_pixels"] == 900 + 100 + 16
    assert meas["largest_component_area_pixels"] == 900

    # Shape descriptors should describe Component 1 (30x30 square)
    # Perimeter of 30x30 square: 4 * 29 = 116.0 px
    assert meas["perimeter_pixels"] == 116.0
    # Feret diameter of 30x30 square: 29 * sqrt(2) ~ 41.01 px
    assert meas["feret_diameter_max"] == round(float(29 * math.sqrt(2)), 2)
    # Solidity of primary square = 1.0
    assert meas["solidity"] == 1.0

    Measurements(**meas)


def test_c_shaped_indented_solidity():
    """Test 7: C-shaped concave mask has solidity < 1.0."""
    mask = np.zeros((256, 256), dtype=np.uint8)
    # Outer 40x40 square with 20x20 right indentation removed
    mask[50:90, 50:90] = 1
    mask[60:80, 70:90] = 0  # carve cavity
    prob = (mask * 0.9).astype(np.float32)

    meas, _ = calculate_tumor_measurements(mask, prob)

    solidity = meas["solidity"]
    assert solidity is not None
    assert 0.50 <= solidity < 0.95, f"Expected indented shape solidity < 1.0, got {solidity}"


def test_arbitrary_2d_dimensions():
    """Test 8: Function supports arbitrary 2D dimensions without crashing."""
    shapes = [(128, 128), (512, 512), (200, 300), (384, 256)]

    for h, w in shapes:
        mask = np.zeros((h, w), dtype=np.uint8)
        # Put a 20x20 block in center
        cy, cx = h // 2, w // 2
        mask[cy - 10:cy + 10, cx - 10:cx + 10] = 1
        prob = (mask * 0.8).astype(np.float32)

        meas, conf = calculate_tumor_measurements(mask, prob)

        assert meas["tumor_detected"] is True
        assert meas["area_pixels"] == 400
        expected_pct = round((400 / (h * w)) * 100.0, 2)
        assert meas["area_percentage"] == expected_pct
        assert meas["perimeter_pixels"] == float(4 * (20 - 1))
        assert meas["solidity"] == 1.0
        assert conf == 0.8


def test_mismatched_dimensions_rejection():
    """Test 9: Mismatched mask and probability-map shapes cleanly raise ValueError."""
    mask = np.zeros((256, 256), dtype=np.uint8)
    prob = np.zeros((128, 128), dtype=np.float32)

    try:
        calculate_tumor_measurements(mask, prob)
        raise AssertionError("Expected ValueError on mismatched shapes")
    except ValueError as e:
        assert "Shape mismatch" in str(e)

    # 3D array rejection
    mask_3d = np.zeros((1, 256, 256), dtype=np.uint8)
    prob_3d = np.zeros((1, 256, 256), dtype=np.float32)
    try:
        calculate_tumor_measurements(mask_3d, prob_3d)
        raise AssertionError("Expected ValueError on 3D arrays")
    except ValueError as e:
        assert "Expected 2D arrays" in str(e)

    # Empty dimensions rejection
    mask_empty = np.zeros((0, 256), dtype=np.uint8)
    prob_empty = np.zeros((0, 256), dtype=np.float32)
    try:
        calculate_tumor_measurements(mask_empty, prob_empty)
        raise AssertionError("Expected ValueError on empty dimensions")
    except ValueError as e:
        assert "Invalid empty dimensions" in str(e)


def test_non_finite_inputs_handled():
    """Test 10: Non-finite inputs (NaN, Inf) are safely sanitized."""
    mask = np.zeros((256, 256), dtype=np.float32)
    mask[50:70, 50:70] = 1.0
    mask[0, 0] = np.nan
    mask[0, 1] = np.inf

    prob = np.zeros((256, 256), dtype=np.float32)
    prob[50:70, 50:70] = 0.9
    prob[0, 0] = np.nan
    prob[0, 1] = np.inf

    meas, conf = calculate_tumor_measurements(mask, prob)

    assert meas["tumor_detected"] is True
    assert meas["area_pixels"] == 400
    assert math.isfinite(conf)
    assert meas["circularity"] is not None and math.isfinite(meas["circularity"])


def test_mathematical_descriptor_bounds():
    """Test 11: Mathematical bounds hold across various test shapes."""
    rng = np.random.default_rng(42)

    for _ in range(5):
        mask = np.zeros((256, 256), dtype=np.uint8)
        # Random blob
        cy, cx = rng.integers(50, 200, size=2)
        rw, rh = rng.integers(10, 40, size=2)
        mask[cy:cy + rh, cx:cx + rw] = 1

        prob = (mask * 0.9).astype(np.float32)
        meas, _ = calculate_tumor_measurements(mask, prob)

        if meas["circularity"] is not None:
            assert 0.0 <= meas["circularity"] <= 1.0
        if meas["solidity"] is not None:
            assert 0.0 <= meas["solidity"] <= 1.0
        if meas["eccentricity"] is not None:
            assert 0.0 <= meas["eccentricity"] <= 1.0
        if meas["major_axis_length"] is not None and meas["minor_axis_length"] is not None:
            assert meas["major_axis_length"] >= meas["minor_axis_length"] > 0
        if meas["feret_diameter_max"] is not None:
            assert meas["feret_diameter_max"] > 0


def test_two_adjacent_pixels_degenerate():
    """Test 12: Two adjacent pixels (1x2) form a degenerate 1D segment; returns safe nulls."""
    mask = np.zeros((256, 256), dtype=np.uint8)
    mask[50, 50:52] = 1
    prob = np.zeros((256, 256), dtype=np.float32)
    prob[50, 50:52] = 0.9

    meas, conf = calculate_tumor_measurements(mask, prob)

    assert meas["area_pixels"] == 2
    assert meas["num_connected_components"] == 1
    # Degenerate 1D line: perimeter is undefined / < 3 points
    assert meas["perimeter_pixels"] is None
    assert meas["circularity"] is None
    assert meas["major_axis_length"] is None
    assert meas["minor_axis_length"] is None
    assert meas["eccentricity"] is None
    assert meas["feret_diameter_max"] is None
    assert meas["solidity"] == 1.0


def test_small_cross_micro_island():
    """Test 13: 5-pixel cross (center + 4-connected neighbors) verifies circularity clamping and bounds."""
    mask = np.zeros((256, 256), dtype=np.uint8)
    mask[50, 50] = 1
    mask[49, 50] = 1
    mask[51, 50] = 1
    mask[50, 49] = 1
    mask[50, 51] = 1
    prob = (mask * 0.9).astype(np.float32)

    meas, _ = calculate_tumor_measurements(mask, prob)

    assert meas["area_pixels"] == 5
    # Contour perimeter connecting 4 outer tips is 4 * sqrt(2) ~ 5.66 px
    expected_p = round(float(4 * math.sqrt(2)), 2)
    assert meas["perimeter_pixels"] == expected_p

    # Raw circularity: 4*pi*5 / (5.66^2) ~ 1.9635 > 1.0; clamped to 1.0
    assert meas["circularity"] == 1.0
    assert meas["solidity"] == 1.0
    assert meas["eccentricity"] == 0.0
    assert meas["feret_diameter_max"] == 2.0


def test_diagonal_line_component():
    """Test 14: 8-connected diagonal line segment has minor axis = 0, eccentricity = 1.0."""
    mask = np.zeros((256, 256), dtype=np.uint8)
    L = 10
    for i in range(L):
        mask[50 + i, 50 + i] = 1
    prob = (mask * 0.9).astype(np.float32)

    meas, _ = calculate_tumor_measurements(mask, prob)

    assert meas["area_pixels"] == L
    assert meas["num_connected_components"] == 1
    assert meas["major_axis_length"] is not None and meas["major_axis_length"] > 0
    assert meas["minor_axis_length"] == 0.0
    assert meas["eccentricity"] == 1.0
    # Feret diameter across diagonal: (L-1)*sqrt(2) = 9*sqrt(2) ~ 12.73 px
    expected_feret = round(float((L - 1) * math.sqrt(2)), 2)
    assert meas["feret_diameter_max"] == expected_feret


def test_equal_area_tie_breaking():
    """Test 15: Ties between equal-area components deterministically select the first in raster scan."""
    mask = np.zeros((256, 256), dtype=np.uint8)
    # Component A: top square (y=10, x=10, 20x20 = 400 px)
    mask[10:30, 10:30] = 1
    # Component B: bottom square (y=100, x=100, 20x20 = 400 px)
    mask[100:120, 100:120] = 1
    prob = (mask * 0.9).astype(np.float32)

    meas, _ = calculate_tumor_measurements(mask, prob)

    assert meas["num_connected_components"] == 2
    assert meas["area_pixels"] == 800
    assert meas["largest_component_area_pixels"] == 400
    # Deterministic tie-breaking selects top component (x=10, y=10)
    assert meas["largest_component_bounding_box"]["x"] == 10
    assert meas["largest_component_bounding_box"]["y"] == 10


def test_full_foreground_mask():
    """Test 16: Full foreground mask (all ones, 256x256) evaluates correctly."""
    mask = np.ones((256, 256), dtype=np.uint8)
    prob = np.ones((256, 256), dtype=np.float32) * 0.95

    meas, conf = calculate_tumor_measurements(mask, prob)

    assert meas["tumor_detected"] is True
    assert meas["area_pixels"] == 65536
    assert meas["area_percentage"] == 100.0
    assert meas["num_connected_components"] == 1
    # Perimeter of 256x256 outer border: 4 * 255 = 1020.0 px
    assert meas["perimeter_pixels"] == 1020.0
    assert meas["solidity"] == 1.0
    # Feret diameter is diagonal corner distance: 255 * sqrt(2) ~ 360.62 px
    assert meas["feret_diameter_max"] == round(float(255 * math.sqrt(2)), 2)
    assert round(conf, 2) == 0.95


def test_boolean_and_numeric_dtypes():
    """Test 17: Function correctly handles boolean and varied numeric array dtypes."""
    for dt in [bool, np.int32, np.int64, np.float32, np.float64]:
        mask = np.zeros((256, 256), dtype=dt)
        mask[20:40, 20:40] = 1 if dt != bool else True
        prob = np.ones((256, 256), dtype=np.float32) * 0.85

        meas, conf = calculate_tumor_measurements(mask, prob)
        assert meas["tumor_detected"] is True
        assert meas["area_pixels"] == 400
        assert meas["perimeter_pixels"] == float(4 * 19)
        assert meas["solidity"] == 1.0


def test_invalid_input_types_rejection():
    """Test 18: Reject non-ndarray and 1D inputs with clear exceptions."""
    # List instead of ndarray -> TypeError
    try:
        calculate_tumor_measurements([[0, 1], [1, 0]], np.zeros((2, 2)))
        raise AssertionError("Expected TypeError for list input")
    except TypeError as e:
        assert "must be numpy.ndarray" in str(e)

    # 1D array -> ValueError
    try:
        calculate_tumor_measurements(np.zeros(256), np.zeros(256))
        raise AssertionError("Expected ValueError for 1D input")
    except ValueError as e:
        assert "Expected 2D arrays" in str(e)


def test_pydantic_legacy_and_zero_eccentricity():
    """Test 19: Pydantic schema preserves valid zero values (e.g., eccentricity=0.0) and legacy dicts."""
    # 1. Valid zero eccentricity on circle
    mask = np.zeros((256, 256), dtype=np.uint8)
    cv2.circle(mask, (128, 128), 30, 1, -1)
    prob = (mask * 0.9).astype(np.float32)

    meas, _ = calculate_tumor_measurements(mask, prob)
    pyd_obj = Measurements(**meas)
    assert pyd_obj.eccentricity is not None
    assert pyd_obj.eccentricity >= 0.0

    # 2. Legacy response dictionary lacking new shape fields
    legacy_dict = {
        "tumor_detected": True,
        "area_pixels": 400,
        "area_percentage": 0.61,
        "bounding_box": {"x": 10, "y": 10, "width": 20, "height": 20},
        "width_pixels": 20,
        "height_pixels": 20,
        "diameter_pixels": 20,
        "num_connected_components": 1,
        "largest_component_area_pixels": 400,
        "centroid": {"x": 20.0, "y": 20.0},
        "largest_component_bounding_box": {"x": 10, "y": 10, "width": 20, "height": 20}
    }
    legacy_pyd = Measurements(**legacy_dict)
    assert legacy_pyd.perimeter_pixels is None
    assert legacy_pyd.circularity is None
    assert legacy_pyd.shape_descriptors is None


if __name__ == "__main__":
    print("=" * 65)
    print("SUSHRUTA 2D MORPHOMETRICS COMPREHENSIVE TEST SUITE (PHASE 3B)")
    print("=" * 65)

    tests = [
        ("Empty Mask Behavior", test_empty_mask_behavior),
        ("Single Pixel Mask (Degenerate)", test_single_pixel_mask_degenerate),
        ("Solid Synthetic Disk", test_solid_synthetic_disk),
        ("Filled Square (Raster Convention)", test_filled_square_raster_convention),
        ("Elongated Rectangle (Axes & Eccentricity)", test_elongated_rectangle),
        ("Multi-Component Aggregation Policy", test_multi_component_aggregation_policy),
        ("C-Shaped Indented Solidity", test_c_shaped_indented_solidity),
        ("Arbitrary 2D Dimensions", test_arbitrary_2d_dimensions),
        ("Mismatched Dimensions Rejection", test_mismatched_dimensions_rejection),
        ("Non-Finite Inputs Handling", test_non_finite_inputs_handled),
        ("Mathematical Descriptor Bounds", test_mathematical_descriptor_bounds),
        ("Two Adjacent Pixels (Degenerate)", test_two_adjacent_pixels_degenerate),
        ("Small Cross Micro-Island", test_small_cross_micro_island),
        ("Diagonal Line Component", test_diagonal_line_component),
        ("Equal Area Tie-Breaking Determinism", test_equal_area_tie_breaking),
        ("Full Foreground Mask", test_full_foreground_mask),
        ("Boolean & Varied Dtypes", test_boolean_and_numeric_dtypes),
        ("Invalid Input Types Rejection", test_invalid_input_types_rejection),
        ("Pydantic Legacy & Zero Eccentricity", test_pydantic_legacy_and_zero_eccentricity),
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
    print(f"ALL {passed}/{len(tests)} 2D MORPHOMETRICS TESTS PASSED SUCCESSFULLY")
    print("=" * 65)
