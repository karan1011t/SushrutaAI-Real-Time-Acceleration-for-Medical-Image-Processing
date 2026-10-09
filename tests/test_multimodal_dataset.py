"""
Automated Unit Test Suite for 4-Channel MultiModalBraTS2DDataset.
Uses temporary synthetic NIfTI files without touching or modifying raw patient data.
"""

import os
import tempfile
import json
import unittest
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from typing import Tuple, Optional, Dict, Any, List
import numpy as np
import nibabel as nib
import torch

from src.models.multimodal_dataset import (
    MultiModalBraTS2DDataset,
    MULTIMODAL_CHANNEL_ORDER
)


class TestMultiModalBraTS2DDataset(unittest.TestCase):

    def setUp(self):
        """Sets up a temporary directory with synthetic 4-modality BraTS NIfTI patient volumes."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root_path = self.temp_dir.name
        self.raw_dir = os.path.join(self.root_path, "raw_brats")
        self.manifest_path = os.path.join(self.root_path, "split_manifest.json")
        os.makedirs(self.raw_dir, exist_ok=True)

        self.affine = np.eye(4)
        self.shape_3d = (32, 32, 5) # Small 3D volume for fast test execution

        # Case 0: Complete valid case with known intensities and labels
        self.create_synthetic_case("BraTS2021_00001", self.shape_3d, missing_mod=None, shape_mismatch=False)

        # Case 1: Missing T1CE modality
        self.create_synthetic_case("BraTS2021_00002", self.shape_3d, missing_mod="T1CE", shape_mismatch=False)

        # Case 2: Shape mismatch in T2
        self.create_synthetic_case("BraTS2021_00003", self.shape_3d, missing_mod=None, shape_mismatch=True)

        # Create split manifest
        manifest = {
            "train": ["BraTS2021_00001", "BraTS2021_00002", "BraTS2021_00003"],
            "val": [],
            "test": []
        }
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f)

    def tearDown(self):
        self.temp_dir.cleanup()

    def create_synthetic_case(self, case_id: str, shape: Tuple[int, int, int], missing_mod: Optional[str] = None, shape_mismatch: bool = False):
        case_folder = os.path.join(self.raw_dir, case_id)
        os.makedirs(case_folder, exist_ok=True)

        mods = ["FLAIR", "T1", "T1CE", "T2", "SEG"]
        for mod in mods:
            if mod == missing_mod:
                continue

            current_shape = shape
            if shape_mismatch and mod == "T2":
                current_shape = (32, 32, 10) # Mismatched depth

            if mod == "SEG":
                data = np.zeros(current_shape, dtype=np.int16)
                # Add BraTS labels: 1 (NCR/NET), 2 (ED), 4 (ET)
                data[10:15, 10:15, 2] = 1
                data[15:20, 15:20, 2] = 2
                data[20:25, 20:25, 2] = 4
            else:
                # Modality intensity patterns
                mult = {"FLAIR": 1.0, "T1": 2.0, "T1CE": 3.0, "T2": 4.0}[mod]
                data = np.zeros(current_shape, dtype=np.float32)
                data[5:25, 5:25, :] = 100.0 * mult

            fname = f"{case_id}_{mod.lower()}.nii.gz"
            fpath = os.path.join(case_folder, fname)
            img = nib.Nifti1Image(data, self.affine)
            nib.save(img, fpath)

    def test_channel_ordering_and_tensor_shapes(self):
        """Verifies channel order [FLAIR, T1, T1CE, T2], output shapes, and float32 dtypes."""
        ds = MultiModalBraTS2DDataset(
            data_dir=self.raw_dir,
            manifest_path=self.manifest_path,
            split="train"
        )
        self.assertEqual(len(ds.valid_cases), 1, "Only BraTS2021_00001 should be valid")
        self.assertIn("BraTS2021_00001", ds.valid_cases)
        self.assertEqual(len(ds), 5, "Should index 5 axial slices for BraTS2021_00001")

        img_tensor, mask_tensor = ds[2] # Fetch slice 2 (where synthetic tumor exists)

        self.assertEqual(img_tensor.shape, (4, 32, 32), "Image tensor must be (4, H, W)")
        self.assertEqual(mask_tensor.shape, (1, 32, 32), "Mask tensor must be (1, H, W)")
        self.assertEqual(img_tensor.dtype, torch.float32, "Image tensor must be float32")
        self.assertEqual(mask_tensor.dtype, torch.float32, "Mask tensor must be float32")
        self.assertEqual(MULTIMODAL_CHANNEL_ORDER, ["FLAIR", "T1", "T1CE", "T2"])

    def test_binary_mask_conversion_semantics(self):
        """Verifies BraTS labels {1, 2, 4} map to 1.0 and label 0 maps to 0.0."""
        ds = MultiModalBraTS2DDataset(
            data_dir=self.raw_dir,
            manifest_path=self.manifest_path,
            split="train"
        )
        _, mask_tensor = ds[2] # Slice 2 contains labels 1, 2, 4

        mask_np = mask_tensor.squeeze().numpy()
        unique_vals = np.unique(mask_np)
        self.assertTrue(all(v in (0.0, 1.0) for v in unique_vals), f"Mask must be binary {{0.0, 1.0}}, got {unique_vals}")

        # Check whole tumor pixel count
        tumor_pixel_count = np.sum(mask_np == 1.0)
        expected_count = (5 * 5) + (5 * 5) + (5 * 5) # 75 pixels for labels 1, 2, 4
        self.assertEqual(tumor_pixel_count, expected_count, f"Expected {expected_count} tumor pixels, got {tumor_pixel_count}")

    def test_missing_modality_handling(self):
        """Verifies cases missing required modalities are excluded with explicit error logs."""
        ds = MultiModalBraTS2DDataset(
            data_dir=self.raw_dir,
            manifest_path=self.manifest_path,
            split="train"
        )
        self.assertIn("BraTS2021_00002", ds.excluded_cases)
        self.assertIn("Missing required modality files", ds.excluded_cases["BraTS2021_00002"])

    def test_shape_mismatch_rejection(self):
        """Verifies cases with mismatched spatial shapes are rejected."""
        ds = MultiModalBraTS2DDataset(
            data_dir=self.raw_dir,
            manifest_path=self.manifest_path,
            split="train"
        )
        self.assertIn("BraTS2021_00003", ds.excluded_cases)
        self.assertIn("Spatial shape mismatch", ds.excluded_cases["BraTS2021_00003"])

    def test_sample_metadata_integrity(self):
        """Verifies sample metadata retrieval."""
        ds = MultiModalBraTS2DDataset(
            data_dir=self.raw_dir,
            manifest_path=self.manifest_path,
            split="train"
        )
        meta = ds.get_sample_metadata(0)
        self.assertEqual(meta["case_id"], "BraTS2021_00001")
        self.assertEqual(meta["slice_index"], 0)
        self.assertEqual(meta["channels"], ["FLAIR", "T1", "T1CE", "T2"])


if __name__ == "__main__":
    unittest.main()
