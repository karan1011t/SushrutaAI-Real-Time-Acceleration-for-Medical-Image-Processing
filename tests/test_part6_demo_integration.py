"""
End-to-End Integration and Demo Readiness Test Suite (Part 6).
Verifies:
1. Production Backend Health & Hardware Device reporting.
2. End-to-end /predict pipeline on Positive Control fixture (BraTS slice 063).
3. End-to-end /predict pipeline on Negative Control fixture (BraTS slice 028).
4. Validation of Empty Mask contracts (zero pixels, null shape descriptors, unavailable LayerCAM).
5. Output visualization artifact URL accessibility and HTTP 200 status.
6. Error handling for Corrupt, Empty, and Unsupported file uploads.
7. Simulated Model Failure (non-finite outputs) returning structured HTTP 500 without leaking internals.
8. Signals & Systems endpoint /analyze-signal compatibility and artifact generation.
"""

import os
import sys
import io
import unittest
from unittest.mock import patch
import numpy as np
import torch
from fastapi.testclient import TestClient

# Add project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from app.main import app

POS_FIXTURE = os.path.join(project_root, "data", "processed", "brats", "val", "images", "BraTS2021_00002_slice_063.png")
NEG_FIXTURE = os.path.join(project_root, "data", "processed", "brats", "val", "images", "BraTS2021_00002_slice_028.png")


class TestPart6DemoIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client_ctx = TestClient(app)
        cls.client = cls.client_ctx.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client_ctx.__exit__(None, None, None)

    def test_01_backend_health_and_device(self):
        """Verify /health returns ok, active device, and correct parameter count."""
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "ok")
        self.assertTrue(data["model_loaded"])
        self.assertIn(data["device"], ["cuda", "cpu"])
        self.assertEqual(data["trainable_parameters"], 4317825)
        self.assertTrue(os.path.exists(data["model_path"]))

    def test_02_positive_fixture_prediction(self):
        """Verify positive control produces tumor detection, positive mask, and valid LayerCAM."""
        self.assertTrue(os.path.exists(POS_FIXTURE), f"Missing positive fixture: {POS_FIXTURE}")
        with open(POS_FIXTURE, "rb") as f:
            file_bytes = f.read()

        resp = self.client.post("/predict", files={"file": ("pos_slice_063.png", file_bytes, "image/png")})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()

        # Contract assertions
        self.assertTrue(data["success"])
        self.assertTrue(data["tumor_detected"])
        self.assertTrue(data["classification"]["tumor_detected"])
        self.assertGreater(data["classification"]["tumor_probability"], 0.5)
        self.assertEqual(data["classification"]["classification_label"], "Tumor Detected")

        # Measurements
        meas = data["measurements"]
        self.assertTrue(meas["tumor_detected"])
        self.assertEqual(meas["area_pixels"], 636)
        self.assertAlmostEqual(meas["area_percentage"], (636 / 65536.0) * 100.0, places=2)
        self.assertEqual(meas["num_connected_components"], 1)
        self.assertEqual(meas["bounding_box"], {"x": 122, "y": 127, "width": 29, "height": 29})
        self.assertAlmostEqual(meas["centroid"]["x"], 135.96, places=1)
        self.assertAlmostEqual(meas["centroid"]["y"], 141.19, places=1)

        # Shape descriptors populated
        self.assertIsNotNone(meas["circularity"])
        self.assertIsNotNone(meas["solidity"])
        self.assertIsNotNone(meas["perimeter_pixels"])
        self.assertIsNotNone(meas["feret_diameter_max"])
        self.assertEqual(meas["spatial_unit"], "pixels")

        # LayerCAM outputs populated
        outputs = data["outputs"]
        self.assertIsNotNone(outputs["layercam"])
        self.assertIsNotNone(outputs["layercam_overlay"])
        self.assertIsNotNone(outputs["combined_overlay"])
        self.assertIsNotNone(outputs["classifier_layercam"])

        # Explainability metadata
        xai = data["explainability"]
        self.assertIsNotNone(xai)
        self.assertEqual(xai["method"], "LayerCAM")
        self.assertIn("predicted tumor pixels", xai["target_definition"])

    def test_03_negative_fixture_prediction(self):
        """Verify negative control produces no tumor detection, empty mask, and null descriptors."""
        self.assertTrue(os.path.exists(NEG_FIXTURE), f"Missing negative fixture: {NEG_FIXTURE}")
        with open(NEG_FIXTURE, "rb") as f:
            file_bytes = f.read()

        resp = self.client.post("/predict", files={"file": ("neg_slice_028.png", file_bytes, "image/png")})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()

        # Contract assertions
        self.assertTrue(data["success"])
        self.assertFalse(data["tumor_detected"])
        self.assertFalse(data["classification"]["tumor_detected"])
        self.assertLess(data["classification"]["tumor_probability"], 0.5)
        self.assertEqual(data["classification"]["classification_label"], "No Tumor Detected")

        # Measurements: empty mask
        meas = data["measurements"]
        self.assertFalse(meas["tumor_detected"])
        self.assertEqual(meas["area_pixels"], 0)
        self.assertEqual(meas["area_percentage"], 0.0)
        self.assertEqual(meas["num_connected_components"], 0)
        self.assertEqual(meas["bounding_box"], {"x": 0, "y": 0, "width": 0, "height": 0})
        self.assertEqual(meas["centroid"], {"x": 0.0, "y": 0.0})

        # Shape descriptors: strictly null for empty mask
        self.assertIsNone(meas["circularity"])
        self.assertIsNone(meas["solidity"])
        self.assertIsNone(meas["perimeter_pixels"])
        self.assertIsNone(meas["major_axis_length"])
        self.assertIsNone(meas["minor_axis_length"])
        self.assertIsNone(meas["eccentricity"])
        self.assertIsNone(meas["feret_diameter_max"])

        # LayerCAM outputs: strictly None for empty mask
        outputs = data["outputs"]
        self.assertIsNone(outputs["layercam"])
        self.assertIsNone(outputs["layercam_overlay"])
        self.assertIsNone(outputs["combined_overlay"])

        # Explainability metadata: explains unavailable state clearly
        xai = data["explainability"]
        self.assertIsNotNone(xai)
        self.assertEqual(xai["target_definition"], "Unavailable (No Tumor Pixels Segmented)")
        self.assertIn("0 tumor pixels were segmented", xai["interpretation"])

    def test_04_artifact_urls_return_200(self):
        """Verify static URLs generated by /predict return HTTP 200 OK."""
        with open(POS_FIXTURE, "rb") as f:
            file_bytes = f.read()

        resp = self.client.post("/predict", files={"file": ("url_test.png", file_bytes, "image/png")})
        data = resp.json()
        outputs = data["outputs"]

        for key in ["preprocessed_mri", "tumor_mask", "overlay", "probability_map", "classifier_layercam", "layercam_overlay"]:
            url = outputs.get(key)
            self.assertIsNotNone(url, f"Artifact {key} was unexpectedly None")
            # Convert full URL to relative path for TestClient
            path = url.split("http://testserver")[-1].split("http://localhost:8000")[-1].split("http://127.0.0.1:8000")[-1]
            get_resp = self.client.get(path)
            self.assertEqual(get_resp.status_code, 200, f"Artifact URL {url} (path: {path}) returned status {get_resp.status_code}")

    def test_05_invalid_and_corrupt_file_handling(self):
        """Verify invalid extensions and corrupt bytes return HTTP 400."""
        # Unsupported format
        r1 = self.client.post("/predict", files={"file": ("sample.txt", b"plain text", "text/plain")})
        self.assertEqual(r1.status_code, 400)
        self.assertFalse(r1.json()["success"])

        # Empty payload
        r2 = self.client.post("/predict", files={"file": ("empty.png", b"", "image/png")})
        self.assertEqual(r2.status_code, 400)
        self.assertFalse(r2.json()["success"])

        # Corrupt PNG
        r3 = self.client.post("/predict", files={"file": ("corrupt.png", b"\x89PNG\r\n\x1a\ncorruptdata", "image/png")})
        self.assertEqual(r3.status_code, 400)
        self.assertFalse(r3.json()["success"])

    def test_06_simulated_inference_failure_returns_500(self):
        """Verify non-finite model outputs trigger structured HTTP 500 error without stale data."""
        with open(POS_FIXTURE, "rb") as f:
            file_bytes = f.read()

        # Simulate U-Net failure emitting NaN
        with patch("app.main.segmenter.segment", side_effect=ValueError("U-Net produced non-finite probabilities (NaN)")):
            resp = self.client.post("/predict", files={"file": ("test_fail.png", file_bytes, "image/png")})
            self.assertEqual(resp.status_code, 500)
            data = resp.json()
            self.assertFalse(data["success"])
            self.assertEqual(data["error"], "Inference Failure")
            self.assertIn("non-finite", data["detail"])

    def test_07_signals_and_systems_endpoint(self):
        """Verify /analyze-signal endpoint processes positive slice and returns all SS artifacts."""
        with open(POS_FIXTURE, "rb") as f:
            file_bytes = f.read()

        resp = self.client.post("/analyze-signal", files={"file": ("ss_slice.png", file_bytes, "image/png")})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()

        self.assertTrue(data["success"])
        self.assertIn("signal", data)
        self.assertIn("histogram", data)
        self.assertIn("frequency", data)
        self.assertIn("filtering", data)
        self.assertIn("outputs", data)

        ss_outputs = data["outputs"]
        for key in ["histogram", "filter_comparison", "fft_analysis", "edge_detection", "morphology_analysis"]:
            url = ss_outputs.get(key)
            self.assertIsNotNone(url)
            path = url.split("http://testserver")[-1].split("http://localhost:8000")[-1].split("http://127.0.0.1:8000")[-1]
            get_resp = self.client.get(path)
            self.assertEqual(get_resp.status_code, 200)


if __name__ == "__main__":
    unittest.main()
