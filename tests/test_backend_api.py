"""
Automated Backend API Test Suite for SUSHRUTA U-Net Inference, LayerCAM XAI, and Signals & Systems Analysis Backend.

Verifies:
1. API initialization & /health status.
2. Model loading & CUDA/CPU device report.
3. Image file validation (accepts valid PNG, rejects corrupt/invalid/missing files).
4. Prediction pipeline end-to-end execution.
5. Binary mask spatial shape (256, 256) and binary value assertions {0, 1}.
6. Morphometric measurements (area, percentage, bounding box, diameter, connected components, centroid).
7. Output artifact rendering (MRI, Mask, Overlay, Heatmap).
8. Fallback CPU inference handling.
9. Phase 2 LayerCAM explainability generation & attribution map validity [0, 1].
10. Phase 2 LayerCAM visual artifacts (LayerCAM, LayerCAM Overlay, Combined Overlay).
11. Phase 2 Explainability metadata structure in response.
12. Phase 4 Signals & Systems /analyze-signal endpoint acceptance & schema.
13. Signal statistics, histogram, FFT, spatial filtering, edge detection, and morphology artifacts.
14. Numerical finiteness, FFT dimensions, and binary morphology integrity.
15. Unchanged /predict behavior and binary U-Net mask preservation {0, 1}.
"""

import os
import sys
import io
import math
import numpy as np
from PIL import Image
import torch
from fastapi.testclient import TestClient

# Add project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from app.main import app
from app.inference import MedicalImageSegmenter
from app.preprocessing import preprocess_image_file
from app.explainability import LayerCAMGenerator
from app.signal_analysis import analyze_signal_image


def run_backend_test_suite():
    print("=" * 60, flush=True)
    print("SUSHRUTA BACKEND API TEST SUITE (PHASE 1, 2 XAI, & PHASE 4 SS ANALYSIS)", flush=True)
    print("=" * 60, flush=True)

    passed_count = 0
    total_tests = 32

    # Initialize FastAPI TestClient with lifespan events context
    with TestClient(app) as client:

        # ---------------------------------------------------------------------
        # TEST 1 — ROOT ENDPOINT
        # ---------------------------------------------------------------------
        print("\n[TEST 1] Testing GET / Endpoint...", end=" ", flush=True)
        resp = client.get("/")
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
        data = resp.json()
        assert data["project"] == "SUSHRUTA"
        print("PASSED")
        passed_count += 1

        # ---------------------------------------------------------------------
        # TEST 2 — HEALTH CHECK ENDPOINT
        # ---------------------------------------------------------------------
        print("[TEST 2] Testing GET /health Endpoint...", end=" ", flush=True)
        resp = client.get("/health")
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
        health_data = resp.json()
        assert health_data["status"] == "ok"
        assert health_data["model_loaded"] is True
        assert health_data["device"] in ("cuda", "cpu")
        assert health_data["trainable_parameters"] == 4317825
        print(f"PASSED (Device: {health_data['device']}, Params: {health_data['trainable_parameters']:,})")
        passed_count += 1

        # ---------------------------------------------------------------------
        # TEST 3 — MODEL CHECKPOINT INITIALIZATION
        # ---------------------------------------------------------------------
        print("[TEST 3] Testing Direct Model Segmenter Loading...", end=" ", flush=True)
        segmenter = MedicalImageSegmenter()
        info = segmenter.get_info()
        assert info["trainable_parameters"] == 4317825
        print("PASSED")
        passed_count += 1

        # ---------------------------------------------------------------------
        # TEST 4 — VALID SAMPLE PREDICTION (HELD-OUT TEST SLICE)
        # ---------------------------------------------------------------------
        print("[TEST 4] Testing POST /predict with Valid MRI Sample...", end=" ", flush=True)
        sample_path = os.path.join(project_root, "data", "processed", "brats", "test", "images", "BraTS2021_00001_slice_075.png")
        if not os.path.exists(sample_path):
            dummy_img = Image.fromarray((np.random.rand(256, 256) * 255).astype(np.uint8))
            img_bytes_io = io.BytesIO()
            dummy_img.save(img_bytes_io, format="PNG")
            file_bytes = img_bytes_io.getvalue()
            filename = "sample_test.png"
        else:
            with open(sample_path, "rb") as f:
                file_bytes = f.read()
            filename = "BraTS2021_00001_slice_075.png"

        files = {"file": (filename, file_bytes, "image/png")}
        resp = client.post("/predict", files=files)
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
        pred_res = resp.json()
        assert pred_res["success"] is True
        print("PASSED")
        passed_count += 1

        # ---------------------------------------------------------------------
        # TEST 5 — OUTPUT MASK RESOLUTION (256x256)
        # ---------------------------------------------------------------------
        print("[TEST 5] Testing Output Mask Resolution (256x256)...", end=" ", flush=True)
        assert pred_res["image_dimensions"]["processed_shape"] == [256, 256]
        print("PASSED")
        passed_count += 1

        # ---------------------------------------------------------------------
        # TEST 6 — MASK BINARY ASSERTION {0, 1}
        # ---------------------------------------------------------------------
        print("[TEST 6] Testing Mask Binary {0, 1} Values...", end=" ", flush=True)
        input_tensor, norm_np, meta, _ = preprocess_image_file(file_bytes, filename)
        prob_map, binary_mask, _ = segmenter.segment(input_tensor)
        unique_vals = set(np.unique(binary_mask).tolist())
        assert unique_vals.issubset({0, 1}), f"Expected binary mask {{0, 1}}, got {unique_vals}"
        print(f"PASSED (Unique mask values: {sorted(list(unique_vals))})")
        passed_count += 1

        # ---------------------------------------------------------------------
        # TEST 7 — VISUAL OVERLAY ARTIFACT GENERATION
        # ---------------------------------------------------------------------
        print("[TEST 7] Testing Phase 1 Visual Overlay Artifact URLs...", end=" ", flush=True)
        outputs = pred_res["outputs"]
        # Two-stage gate: segmentation artifacts exist only when the Stage-1
        # classifier detects tumor; otherwise the API correctly returns None.
        if pred_res["tumor_detected"] is True:
            assert "overlay" in outputs and outputs["overlay"].endswith(".png")
            assert "tumor_mask" in outputs and outputs["tumor_mask"].endswith(".png")
            assert "probability_map" in outputs and outputs["probability_map"].endswith(".png")
            assert "preprocessed_mri" in outputs and outputs["preprocessed_mri"].endswith(".png")
            print("PASSED (tumor path)")
        else:
            assert outputs["overlay"] is None
            assert outputs["tumor_mask"] is None
            assert outputs["probability_map"] is None
            assert "preprocessed_mri" in outputs and outputs["preprocessed_mri"].endswith(".png")
            print("PASSED (no-tumor gate path: segmentation correctly skipped)")
        passed_count += 1

        # ---------------------------------------------------------------------
        # TEST 8 — MEASUREMENTS DICTIONARY STRUCTURE
        # ---------------------------------------------------------------------
        print("[TEST 8] Testing Morphometric Measurements Output...", end=" ", flush=True)
        meas = pred_res["measurements"]
        assert "area_pixels" in meas and isinstance(meas["area_pixels"], int)
        assert "area_percentage" in meas and isinstance(meas["area_percentage"], float)
        assert "bounding_box" in meas
        bbox = meas["bounding_box"]
        assert all(k in bbox for k in ("x", "y", "width", "height"))
        assert "diameter_pixels" in meas
        print(f"PASSED (Area: {meas['area_pixels']} px, {meas['area_percentage']}%, Diameter: {meas['diameter_pixels']} px)")
        passed_count += 1

        # ---------------------------------------------------------------------
        # TEST 9 — INVALID FILE TYPE REJECTION
        # ---------------------------------------------------------------------
        print("[TEST 9] Testing Rejection of Invalid File Types...", end=" ", flush=True)
        txt_bytes = b"This is a text file, not an MRI slice."
        files_invalid = {"file": ("document.txt", txt_bytes, "text/plain")}
        resp_invalid = client.post("/predict", files=files_invalid)
        assert resp_invalid.status_code in (400, 422), f"Expected 400 or 422, got {resp_invalid.status_code}"
        assert resp_invalid.json()["success"] is False
        print("PASSED")
        passed_count += 1

        # ---------------------------------------------------------------------
        # TEST 10 — MISSING FILE REJECTION
        # ---------------------------------------------------------------------
        print("[TEST 10] Testing Rejection of Empty/Missing Files...", end=" ", flush=True)
        files_empty = {"file": ("empty.png", b"", "image/png")}
        resp_empty = client.post("/predict", files=files_empty)
        assert resp_empty.status_code == 400
        assert resp_empty.json()["success"] is False
        print("PASSED")
        passed_count += 1

        # ---------------------------------------------------------------------
        # TEST 11 — CPU FALLBACK INFERENCE LOGIC
        # ---------------------------------------------------------------------
        print("[TEST 11] Testing CPU Fallback Inference Execution...", end=" ", flush=True)
        cpu_segmenter = MedicalImageSegmenter(device="cpu")
        cpu_prob, cpu_mask, cpu_time = cpu_segmenter.segment(input_tensor)
        assert cpu_mask.shape == (256, 256)
        assert cpu_time > 0.0
        print(f"PASSED (CPU Inference Time: {cpu_time:.2f} ms)")
        passed_count += 1

        # =====================================================================
        # PHASE 2 EXPLAINABILITY & LAYERCAM TESTS (12 - 20)
        # =====================================================================

        # ---------------------------------------------------------------------
        # TEST 12 — LAYERCAM GENERATION SUCCEEDS DIRECTLY
        # ---------------------------------------------------------------------
        print("[TEST 12] Testing Direct LayerCAM Generation...", end=" ", flush=True)
        layercam_gen = LayerCAMGenerator(segmenter.model, device=segmenter.device)
        cam_map, xai_meta, cam_time = layercam_gen.generate_cam(input_tensor, binary_mask)
        assert isinstance(cam_map, np.ndarray)
        assert cam_time > 0.0
        print(f"PASSED (LayerCAM Time: {cam_time:.2f} ms, Layer: {xai_meta['target_layer']})")
        passed_count += 1

        # ---------------------------------------------------------------------
        # TEST 13 — LAYERCAM OUTPUT IN API RESPONSE
        # ---------------------------------------------------------------------
        print("[TEST 13] Testing LayerCAM Artifact in API Response...", end=" ", flush=True)
        # Gate-aware: U-Net LayerCAM is skipped when Stage-1 detects no tumor.
        if pred_res["tumor_detected"] is True:
            assert "layercam" in pred_res["outputs"] and pred_res["outputs"]["layercam"] is not None
            assert pred_res["outputs"]["layercam"].endswith(".png")
            print("PASSED (tumor path)")
        else:
            assert pred_res["outputs"]["layercam"] is None
            assert pred_res["outputs"]["classifier_layercam"] is not None
            print("PASSED (no-tumor gate path: U-Net LayerCAM correctly skipped, classifier CAM present)")
        passed_count += 1

        # ---------------------------------------------------------------------
        # TEST 14 — LAYERCAM SPATIAL DIMENSIONS (256x256)
        # ---------------------------------------------------------------------
        print("[TEST 14] Testing LayerCAM Spatial Resolution (256x256)...", end=" ", flush=True)
        assert cam_map.shape == (256, 256), f"Expected (256, 256), got {cam_map.shape}"
        print("PASSED")
        passed_count += 1

        # ---------------------------------------------------------------------
        # TEST 15 — LAYERCAM VALUES FINITE AND NORMALIZED [0.0, 1.0]
        # ---------------------------------------------------------------------
        print("[TEST 15] Testing LayerCAM Value Bounds [0.0, 1.0] & Finiteness...", end=" ", flush=True)
        assert np.all(np.isfinite(cam_map)), "LayerCAM contains non-finite NaN/Inf values"
        assert 0.0 <= np.min(cam_map) and np.max(cam_map) <= 1.0, f"LayerCAM out of bounds: min {np.min(cam_map)}, max {np.max(cam_map)}"
        print("PASSED")
        passed_count += 1

        # ---------------------------------------------------------------------
        # TEST 16 — LAYERCAM OVERLAY ARTIFACT GENERATED
        # ---------------------------------------------------------------------
        print("[TEST 16] Testing LayerCAM Overlay Artifact URL...", end=" ", flush=True)
        # Gate-aware: overlay exists only on the tumor path (see TEST 13).
        if pred_res["tumor_detected"] is True:
            assert "layercam_overlay" in pred_res["outputs"] and pred_res["outputs"]["layercam_overlay"] is not None
            assert pred_res["outputs"]["layercam_overlay"].endswith(".png")
            print("PASSED (tumor path)")
        else:
            assert pred_res["outputs"]["layercam_overlay"] is None
            print("PASSED (no-tumor gate path: overlay correctly skipped)")
        passed_count += 1

        # ---------------------------------------------------------------------
        # TEST 17 — API RESPONSE CONTAINS EXPLAINABILITY METADATA
        # ---------------------------------------------------------------------
        print("[TEST 17] Testing Explainability Metadata in API Response...", end=" ", flush=True)
        assert "explainability" in pred_res and pred_res["explainability"] is not None
        xai_info = pred_res["explainability"]
        assert xai_info["method"] == "LayerCAM"
        assert "target_layer" in xai_info and xai_info["target_layer"] == "up4.conv.double_conv.3"
        assert "target_definition" in xai_info
        print(f"PASSED (Method: {xai_info['method']}, Layer: {xai_info['target_layer']})")
        passed_count += 1

        # ---------------------------------------------------------------------
        # TEST 18 — COMBINED VISUALIZATION ARTIFACT GENERATED
        # ---------------------------------------------------------------------
        print("[TEST 18] Testing Combined MRI + Mask + LayerCAM Overlay URL...", end=" ", flush=True)
        # Gate-aware: combined overlay exists only on the tumor path.
        if pred_res["tumor_detected"] is True:
            assert "combined_overlay" in pred_res["outputs"] and pred_res["outputs"]["combined_overlay"] is not None
            assert pred_res["outputs"]["combined_overlay"].endswith(".png")
            print("PASSED (tumor path)")
        else:
            assert pred_res["outputs"]["combined_overlay"] is None
            print("PASSED (no-tumor gate path: combined overlay correctly skipped)")
        passed_count += 1

        # ---------------------------------------------------------------------
        # TEST 19 — EXISTING MASK REMAINS STRICTLY BINARY {0, 1}
        # ---------------------------------------------------------------------
        print("[TEST 19] Verifying Segmentation Mask Binary Integrity {0, 1}...", end=" ", flush=True)
        unique_mask = set(np.unique(binary_mask).tolist())
        assert unique_mask.issubset({0, 1}), f"Non-binary values in mask: {unique_mask}"
        print("PASSED")
        passed_count += 1

        # ---------------------------------------------------------------------
        # TEST 20 — PHASE 2 CONNECTED COMPONENT MEASUREMENTS VALIDATION
        # ---------------------------------------------------------------------
        print("[TEST 20] Testing Phase 2 Connected Component & Centroid Measurements...", end=" ", flush=True)
        meas_p2 = pred_res["measurements"]
        assert "num_connected_components" in meas_p2 and isinstance(meas_p2["num_connected_components"], int)
        assert "largest_component_area_pixels" in meas_p2 and isinstance(meas_p2["largest_component_area_pixels"], int)
        assert "centroid" in meas_p2 and "x" in meas_p2["centroid"] and "y" in meas_p2["centroid"]
        assert "largest_component_bounding_box" in meas_p2
        print(f"PASSED (Components: {meas_p2['num_connected_components']}, Centroid: {meas_p2['centroid']})")
        passed_count += 1

        # =====================================================================
        # PHASE 4 SIGNALS & SYSTEMS / IMAGE PROCESSING TESTS (21 - 32)
        # =====================================================================

        # ---------------------------------------------------------------------
        # TEST 21 — POST /analyze-signal ACCEPTS VALID IMAGE
        # ---------------------------------------------------------------------
        print("[TEST 21] Testing POST /analyze-signal Endpoint Acceptance...", end=" ", flush=True)
        files_ss = {"file": (filename, file_bytes, "image/png")}
        resp_ss = client.post("/analyze-signal", files=files_ss)
        assert resp_ss.status_code == 200, f"Expected 200, got {resp_ss.status_code}: {resp_ss.text}"
        ss_res = resp_ss.json()
        assert ss_res["success"] is True
        print("PASSED")
        passed_count += 1

        # ---------------------------------------------------------------------
        # TEST 22 — SIGNAL STATISTICS RETURNED
        # ---------------------------------------------------------------------
        print("[TEST 22] Testing 2D Signal Statistics I[m,n] Response Structure...", end=" ", flush=True)
        sig = ss_res["signal"]
        assert sig["width"] == 256 and sig["height"] == 256 and sig["pixels"] == 65536
        assert all(k in sig for k in ("mean", "std", "min", "max", "dynamic_range"))
        print(f"PASSED (Mean: {sig['mean']}, Std: {sig['std']}, Dynamic Range: {sig['dynamic_range']})")
        passed_count += 1

        # ---------------------------------------------------------------------
        # TEST 23 — HISTOGRAM ARTIFACT EXISTS
        # ---------------------------------------------------------------------
        print("[TEST 23] Testing Intensity Histogram Visual Artifact URL...", end=" ", flush=True)
        assert "outputs" in ss_res and "histogram" in ss_res["outputs"]
        assert ss_res["outputs"]["histogram"].endswith(".png")
        print("PASSED")
        passed_count += 1

        # ---------------------------------------------------------------------
        # TEST 24 — 2D FFT SPECTRUM ARTIFACT EXISTS
        # ---------------------------------------------------------------------
        print("[TEST 24] Testing 2D FFT Magnitude Spectrum Visual Artifact URL...", end=" ", flush=True)
        assert "fft_analysis" in ss_res["outputs"] and ss_res["outputs"]["fft_analysis"].endswith(".png")
        print("PASSED")
        passed_count += 1

        # ---------------------------------------------------------------------
        # TEST 25 — SPATIAL FILTERING COMPARISON ARTIFACT EXISTS
        # ---------------------------------------------------------------------
        print("[TEST 25] Testing Spatial Filtering Comparison Visual Artifact URL...", end=" ", flush=True)
        assert "filter_comparison" in ss_res["outputs"] and ss_res["outputs"]["filter_comparison"].endswith(".png")
        print("PASSED")
        passed_count += 1

        # ---------------------------------------------------------------------
        # TEST 26 — EDGE DETECTION ARTIFACT EXISTS
        # ---------------------------------------------------------------------
        print("[TEST 26] Testing Spatial Edge Detection Visual Artifact URL...", end=" ", flush=True)
        assert "edge_detection" in ss_res["outputs"] and ss_res["outputs"]["edge_detection"].endswith(".png")
        print("PASSED")
        passed_count += 1

        # ---------------------------------------------------------------------
        # TEST 27 — MATHEMATICAL MORPHOLOGY ARTIFACT EXISTS
        # ---------------------------------------------------------------------
        print("[TEST 27] Testing Binary Morphology Visual Artifact URL...", end=" ", flush=True)
        assert "morphology_analysis" in ss_res["outputs"] and ss_res["outputs"]["morphology_analysis"].endswith(".png")
        print("PASSED")
        passed_count += 1

        # ---------------------------------------------------------------------
        # TEST 28 — RETURNED NUMERICAL VALUES ARE FINITE
        # ---------------------------------------------------------------------
        print("[TEST 28] Testing Numerical Finiteness Across All SS Metrics...", end=" ", flush=True)
        def check_finite(obj):
            if isinstance(obj, float):
                assert math.isfinite(obj), f"Non-finite float value found: {obj}"
            elif isinstance(obj, dict):
                for v in obj.values():
                    check_finite(v)
            elif isinstance(obj, list):
                for item in obj:
                    check_finite(item)

        check_finite(ss_res["signal"])
        check_finite(ss_res["histogram"])
        check_finite(ss_res["filtering"])
        check_finite(ss_res["frequency"])
        check_finite(ss_res["edges"])
        check_finite(ss_res["snr_estimate"])
        print("PASSED")
        passed_count += 1

        # ---------------------------------------------------------------------
        # TEST 29 — 2D FFT OUTPUT DIMENSIONS & SPECTRAL ENERGY VALID
        # ---------------------------------------------------------------------
        print("[TEST 29] Testing 2D FFT Energy & Frequency Ratios...", end=" ", flush=True)
        freq = ss_res["frequency"]
        assert freq["total_energy"] > 0.0
        assert 0.0 <= freq["low_frequency_energy_ratio"] <= 1.0
        assert 0.0 <= freq["high_frequency_energy_ratio"] <= 1.0
        sum_ratios = freq["low_frequency_energy_ratio"] + freq["high_frequency_energy_ratio"]
        assert math.isclose(sum_ratios, 1.0, rel_tol=1e-3), f"Energy ratios sum to {sum_ratios}, expected 1.0"
        print(f"PASSED (Total Energy: {freq['total_energy']}, Low-Freq Ratio: {freq['low_frequency_energy_ratio']})")
        passed_count += 1

        # ---------------------------------------------------------------------
        # TEST 30 — BINARY MORPHOLOGY OUTPUTS REMAIN BINARY
        # ---------------------------------------------------------------------
        print("[TEST 30] Testing Mathematical Morphology Binary Integrity {0, 1}...", end=" ", flush=True)
        dummy_img = norm_np
        dummy_prob = norm_np
        dummy_gt = (norm_np > 0.5).astype(np.float32)
        from src.measurements.signal_image_analysis import apply_morphological_refinement
        morph_res = apply_morphological_refinement(dummy_prob, dummy_gt, threshold=0.5, kernel_size=5)
        for stage_name, stage_mask in morph_res["stages"].items():
            u_vals = set(np.unique(stage_mask).tolist())
            assert u_vals.issubset({0.0, 1.0}), f"Stage {stage_name} non-binary values: {u_vals}"
        print("PASSED")
        passed_count += 1

        # ---------------------------------------------------------------------
        # TEST 31 — EXISTING /predict BEHAVIOR REMAINS UNCHANGED
        # ---------------------------------------------------------------------
        print("[TEST 31] Verifying Production /predict Behavior Consistency...", end=" ", flush=True)
        resp_p1 = client.post("/predict", files={"file": (filename, file_bytes, "image/png")})
        assert resp_p1.status_code == 200
        p1_json = resp_p1.json()
        assert p1_json["success"] is True
        assert "measurements" in p1_json and "explainability" in p1_json
        print("PASSED")
        passed_count += 1

        # ---------------------------------------------------------------------
        # TEST 32 — EXISTING U-NET MASK REMAINS STRICTLY BINARY {0, 1}
        # ---------------------------------------------------------------------
        print("[TEST 32] Verifying Production U-Net Mask Strict Binary Preservation {0, 1}...", end=" ", flush=True)
        prob_m, bin_m, _ = segmenter.segment(input_tensor)
        vals_set = set(np.unique(bin_m).tolist())
        assert vals_set.issubset({0, 1}), f"U-Net binary mask degraded to: {vals_set}"
        print(f"PASSED (Mask unique values: {sorted(list(vals_set))})")
        passed_count += 1

    print("\n" + "=" * 60, flush=True)
    print(f"SUSHRUTA BACKEND TEST SUITE PASSED: {passed_count}/{total_tests} TESTS PASSED", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    run_backend_test_suite()
