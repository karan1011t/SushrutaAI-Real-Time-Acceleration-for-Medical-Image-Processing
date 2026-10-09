"""
Unit Tests for CachedMultiModalBraTS2DDataset.
Verifies volume caching, LRU eviction, cache hit counters, and bitwise output parity with MultiModalBraTS2DDataset.
"""

import sys
import os
import tempfile
import json
import unittest
from pathlib import Path

import numpy as np
import nibabel as nib
import torch

# Ensure project root is in sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models.multimodal_dataset import MultiModalBraTS2DDataset
from src.models.multimodal_dataset_cached import CachedMultiModalBraTS2DDataset, create_cached_multimodal_dataloaders


class TestCachedMultiModalBraTS2DDataset(unittest.TestCase):

    def setUp(self):
        """Creates temporary synthetic NIfTI dataset with 2 patients, 4 modalities + SEG each."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temp_dir.name) / "raw"
        self.manifest_path = Path(self.temp_dir.name) / "manifest.json"

        self.patient_ids = ["BraTS2021_00001", "BraTS2021_00002"]
        self.shape = (32, 32, 10) # 10 Z slices

        manifest_data = {
            "train": [self.patient_ids[0]],
            "val": [self.patient_ids[1]],
            "test": []
        }
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f)

        # Generate synthetic 3D volumes
        for pid in self.patient_ids:
            pfolder = self.data_dir / pid
            pfolder.mkdir(parents=True, exist_ok=True)

            for mod in ["flair", "t1", "t1ce", "t2"]:
                # Non-zero brain region
                arr = np.random.uniform(10.0, 100.0, size=self.shape).astype(np.float32)
                arr[0:5, 0:5, :] = 0.0 # background
                img = nib.Nifti1Image(arr, affine=np.eye(4))
                nib.save(img, str(pfolder / f"{pid}_{mod}.nii.gz"))

            # SEG mask with tumor labels {0, 1, 2, 4}
            seg_arr = np.zeros(self.shape, dtype=np.uint8)
            seg_arr[10:20, 10:20, 2:8] = 1 # WT
            seg_arr[12:15, 12:15, 4:6] = 4 # ET
            seg_img = nib.Nifti1Image(seg_arr, affine=np.eye(4))
            nib.save(seg_img, str(pfolder / f"{pid}_seg.nii.gz"))

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_uncached_vs_cached_exact_parity(self):
        """Verifies cached dataset outputs match uncached outputs to within 1e-6 tolerance."""
        ds_uncached = MultiModalBraTS2DDataset(
            data_dir=str(self.data_dir),
            manifest_path=str(self.manifest_path),
            split="train",
            augment=False
        )
        ds_cached = CachedMultiModalBraTS2DDataset(
            data_dir=str(self.data_dir),
            manifest_path=str(self.manifest_path),
            split="train",
            augment=False,
            max_cached_patients=2
        )

        self.assertEqual(len(ds_uncached), len(ds_cached))
        self.assertGreater(len(ds_cached), 0)

        for i in range(len(ds_uncached)):
            img_u, mask_u = ds_uncached[i]
            img_c, mask_c = ds_cached[i]

            self.assertEqual(img_u.shape, img_c.shape)
            self.assertEqual(mask_u.shape, mask_c.shape)
            self.assertTrue(torch.allclose(img_u, img_c, atol=1e-6), f"Mismatch at slice index {i}")
            self.assertTrue(torch.equal(mask_u, mask_c), f"Mask mismatch at slice index {i}")

    def test_cache_hits_and_volume_reuse(self):
        """Verifies repeated accesses to slices from the same patient hit the volume cache."""
        ds = CachedMultiModalBraTS2DDataset(
            data_dir=str(self.data_dir),
            manifest_path=str(self.manifest_path),
            split="train",
            augment=False,
            max_cached_patients=4
        )

        self.assertEqual(ds.cache_hits, 0)
        self.assertEqual(ds.cache_misses, 0)

        # Slice 0 access -> Cache Miss (Loads patient volumes)
        img0, mask0 = ds[0]
        self.assertEqual(ds.cache_misses, 1)
        self.assertEqual(ds.cache_hits, 0)

        # Slice 1 access (same patient) -> Cache Hit!
        img1, mask1 = ds[1]
        self.assertEqual(ds.cache_misses, 1)
        self.assertEqual(ds.cache_hits, 1)

        # Slice 2 access (same patient) -> Cache Hit!
        img2, mask2 = ds[2]
        self.assertEqual(ds.cache_misses, 1)
        self.assertEqual(ds.cache_hits, 2)

    def test_lru_cache_eviction_boundary(self):
        """Verifies LRU cache bound prevents memory leak by evicting oldest patients."""
        # Create dataset combining both patients
        ds = CachedMultiModalBraTS2DDataset(
            data_dir=str(self.data_dir),
            case_ids=self.patient_ids,
            augment=False,
            max_cached_patients=1
        )

        self.assertEqual(len(ds.slice_index), 20) # 2 patients * 10 slices

        # Access patient 1 slice -> Loads patient 1
        _ = ds[0]
        self.assertIn("BraTS2021_00001", ds.volume_cache)
        self.assertEqual(len(ds.volume_cache), 1)

        # Access patient 2 slice -> Loads patient 2, evicts patient 1
        _ = ds[10] # slice 0 of patient 2
        self.assertIn("BraTS2021_00002", ds.volume_cache)
        self.assertNotIn("BraTS2021_00001", ds.volume_cache)
        self.assertEqual(len(ds.volume_cache), 1)

    def test_augmented_output_parity(self):
        """Verifies deterministic augmentation parity between uncached and cached datasets."""
        ds_u = MultiModalBraTS2DDataset(
            data_dir=str(self.data_dir),
            manifest_path=str(self.manifest_path),
            split="train",
            augment=True,
            seed=123
        )
        ds_c = CachedMultiModalBraTS2DDataset(
            data_dir=str(self.data_dir),
            manifest_path=str(self.manifest_path),
            split="train",
            augment=True,
            seed=123,
            max_cached_patients=2
        )

        for i in range(len(ds_u)):
            img_u, mask_u = ds_u[i]
            img_c, mask_c = ds_c[i]
            self.assertTrue(torch.allclose(img_u, img_c, atol=1e-6))
            self.assertTrue(torch.equal(mask_u, mask_c))

    def test_create_cached_multimodal_dataloaders(self):
        """Verifies dataloader factory builds valid train, val, and test loaders."""
        loaders = create_cached_multimodal_dataloaders(
            data_dir=str(self.data_dir),
            manifest_path=str(self.manifest_path),
            batch_size=2,
            num_workers=0,
            max_cached_patients=4
        )
        self.assertIn("train", loaders)
        self.assertIn("val", loaders)
        self.assertIn("test", loaders)

        for batch_img, batch_mask in loaders["train"]:
            self.assertEqual(batch_img.ndim, 4) # (B, 4, H, W)
            self.assertEqual(batch_mask.ndim, 4) # (B, 1, H, W)
            self.assertEqual(batch_img.shape[1], 4)
            self.assertEqual(batch_mask.shape[1], 1)
            break


if __name__ == "__main__":
    unittest.main()
