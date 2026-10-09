"""
PyTorch Multi-Modal Dataset and DataLoader Utilities for 4-Channel BraTS MRI Segmentation.
Channels: [0: FLAIR, 1: T1, 2: T1CE, 3: T2]
Target Mask: [1, H, W] Binary Whole-Tumor Mask (Labels 1, 2, 4 -> 1.0; 0 -> 0.0)
"""

import os
import glob
import json
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, Union

import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader

# Official channel ordering
MULTIMODAL_CHANNEL_ORDER = ["FLAIR", "T1", "T1CE", "T2"]


class MultiModalBraTS2DDataset(Dataset):
    """
    PyTorch Dataset for 4-Channel 2D BraTS MRI Slices and Binary Whole-Tumor Masks.

    Guarantees:
    - 4 Input Channels: [0: FLAIR, 1: T1, 2: T1CE, 3: T2]
    - Output Image Shape: (4, H, W), float32 tensor in range [0.0, 1.0]
    - Output Mask Shape: (1, H, W), float32 tensor with binary values {0.0, 1.0}
    - Label Mapping: BraTS labels 1 (NCR/NET), 2 (ED), 4 (ET) -> 1.0 (Whole Tumor); 0 -> 0.0
    - Data Leakage Prevention: All slices from a patient stay strictly within the assigned split.
    """

    def __init__(
        self,
        data_dir: str = "data/raw/brats",
        manifest_path: str = "data/processed/brats/split_manifest.json",
        split: str = "train",
        case_ids: Optional[List[str]] = None,
        min_brain_pixels: int = 100,
        min_tumor_pixels: int = 0,
        augment: bool = False,
        seed: int = 42
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.manifest_path = Path(manifest_path)
        self.split = split.lower()
        self.min_brain_pixels = min_brain_pixels
        self.min_tumor_pixels = min_tumor_pixels
        self.augment = augment
        self.seed = seed

        # Determine target patient case IDs
        if case_ids is not None:
            self.patient_ids = sorted(list(set(case_ids)))
        elif self.manifest_path.exists():
            with open(self.manifest_path, "r", encoding="utf-8") as f:
                manifest_data = json.load(f)
                if self.split not in manifest_data:
                    raise KeyError(f"Split '{self.split}' not found in manifest {self.manifest_path}. Available: {list(manifest_data.keys())}")
                self.patient_ids = sorted(list(set(manifest_data[self.split])))
        else:
            # Fallback: scan data_dir for patient folders
            if self.data_dir.exists():
                self.patient_ids = sorted([
                    d.name for d in self.data_dir.iterdir()
                    if d.is_dir() and (d / f"{d.name}_flair.nii.gz").exists() or (d / f"{d.name}_flair.nii").exists()
                ])
            else:
                self.patient_ids = []

        self.valid_cases: List[str] = []
        self.excluded_cases: Dict[str, str] = {}
        self.case_modality_paths: Dict[str, Dict[str, Path]] = {}
        self.slice_index: List[Dict[str, Any]] = []

        self._discover_and_index_cases()

    def _find_modality_file(self, case_folder: Path, case_id: str, mod: str) -> Optional[Path]:
        """Locates NIfTI file for a given modality (flair, t1, t1ce, t2, seg)."""
        mod_lower = mod.lower()

        # Exact expected patterns
        candidates = [
            case_folder / f"{case_id}_{mod_lower}.nii.gz",
            case_folder / f"{case_id}_{mod_lower}.nii",
        ]
        for c in candidates:
            if c.exists():
                return c

        # Glob fallback
        pattern = f"*{mod_lower}.nii*"
        globbed = list(case_folder.glob(pattern))

        # Disambiguate t1 vs t1ce
        if mod_lower == "t1":
            globbed = [p for p in globbed if "t1ce" not in p.name.lower()]

        if globbed:
            return globbed[0]

        return None

    def _discover_and_index_cases(self):
        """Scans patient cases, validates modality completeness and shapes, and builds slice index."""
        for cid in self.patient_ids:
            case_folder = self.data_dir / cid
            if not case_folder.exists() or not case_folder.is_dir():
                self.excluded_cases[cid] = f"Case directory not found: {case_folder}"
                continue

            # Locate all 4 modalities + SEG
            paths = {}
            missing = []
            for mod in ["FLAIR", "T1", "T1CE", "T2", "SEG"]:
                fpath = self._find_modality_file(case_folder, cid, mod)
                if fpath is None:
                    missing.append(mod)
                else:
                    paths[mod] = fpath

            if missing:
                self.excluded_cases[cid] = f"Missing required modality files: {', '.join(missing)}"
                continue

            # Load NIfTI headers to verify spatial shape alignment without reading full arrays into memory
            try:
                flair_img = nib.load(str(paths["FLAIR"]))
                flair_shape = flair_img.shape

                shape_mismatch = False
                for mod in ["T1", "T1CE", "T2", "SEG"]:
                    mod_shape = nib.load(str(paths[mod])).shape
                    if mod_shape != flair_shape:
                        self.excluded_cases[cid] = f"Spatial shape mismatch: FLAIR={flair_shape}, {mod}={mod_shape}"
                        shape_mismatch = True
                        break

                if shape_mismatch:
                    continue

                self.valid_cases.append(cid)
                self.case_modality_paths[cid] = paths

                # Index 2D axial slices (Z dimension)
                num_slices = flair_shape[2] if len(flair_shape) >= 3 else 1
                for slice_idx in range(num_slices):
                    self.slice_index.append({
                        "case_id": cid,
                        "slice_index": slice_idx,
                        "modality_paths": paths
                    })

            except Exception as exc:
                self.excluded_cases[cid] = f"Corrupted header or NIfTI read failure: {str(exc)}"

    def __len__(self) -> int:
        return len(self.slice_index)

    @staticmethod
    def normalize_channel(slice_2d: np.ndarray) -> np.ndarray:
        """
        Applies non-zero z-score intensity normalization and min-max scaling to [0.0, 1.0].
        Matching Sushruta's established preprocessing standard.
        """
        slice_2d = slice_2d.astype(np.float32)
        non_zero_mask = slice_2d > 0

        if not np.any(non_zero_mask):
            return np.zeros_like(slice_2d, dtype=np.float32)

        brain_voxels = slice_2d[non_zero_mask]
        mean = float(np.mean(brain_voxels))
        std = float(np.std(brain_voxels)) + 1e-8

        normalized = np.zeros_like(slice_2d, dtype=np.float32)
        normalized[non_zero_mask] = (slice_2d[non_zero_mask] - mean) / std

        # Min-max scale non-zero region to [0.0, 1.0]
        norm_brain = normalized[non_zero_mask]
        min_v = float(np.min(norm_brain))
        max_v = float(np.max(norm_brain))

        if max_v > min_v:
            normalized[non_zero_mask] = (normalized[non_zero_mask] - min_v) / (max_v - min_v)
        else:
            normalized[non_zero_mask] = 0.0

        return np.clip(normalized, 0.0, 1.0)

    @staticmethod
    def map_brats_segmentation_target(seg_slice_2d: np.ndarray) -> np.ndarray:
        """
        Maps BraTS multi-class segmentation labels to Whole-Tumor (WT) binary target:
        - Labels 1 (NCR/NET), 2 (ED), 4 (ET) -> 1.0 (Tumor)
        - Label 0 (Background) -> 0.0 (Non-Tumor)
        """
        wt_mask = np.zeros_like(seg_slice_2d, dtype=np.float32)
        # Any label in {1, 2, 4} or > 0 is whole tumor
        wt_mask[seg_slice_2d > 0] = 1.0
        return wt_mask

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample_meta = self.slice_index[idx]
        cid = sample_meta["case_id"]
        slice_idx = sample_meta["slice_index"]
        paths = sample_meta["modality_paths"]

        channels_2d = []
        for mod in MULTIMODAL_CHANNEL_ORDER:
            img = nib.load(str(paths[mod]))
            # Slice along 3rd dimension (axial Z)
            slice_data = np.asarray(img.dataobj[:, :, slice_idx], dtype=np.float32)
            norm_slice = self.normalize_channel(slice_data)
            channels_2d.append(norm_slice)

        # Stack channels: shape (4, H, W)
        image_4ch = np.stack(channels_2d, axis=0)

        # Load SEG slice
        seg_img = nib.load(str(paths["SEG"]))
        seg_slice = np.asarray(seg_img.dataobj[:, :, slice_idx])
        mask_wt = self.map_brats_segmentation_target(seg_slice)
        mask_1ch = np.expand_dims(mask_wt, axis=0) # shape (1, H, W)

        image_tensor = torch.from_numpy(np.ascontiguousarray(image_4ch)).float()
        mask_tensor = torch.from_numpy(np.ascontiguousarray(mask_1ch)).float()

        # Data augmentation (applied identically to image and mask)
        if self.augment:
            rng = random.Random(self.seed + idx)
            if rng.random() > 0.5:
                image_tensor = torch.flip(image_tensor, dims=[2])
                mask_tensor = torch.flip(mask_tensor, dims=[2])
            k = rng.choice([0, 1, 2, 3])
            if k > 0:
                image_tensor = torch.rot90(image_tensor, k, dims=[1, 2])
                mask_tensor = torch.rot90(mask_tensor, k, dims=[1, 2])

        return image_tensor, mask_tensor

    def get_sample_metadata(self, idx: int) -> Dict[str, Any]:
        """Returns metadata (case_id, slice_index) for a sample."""
        meta = self.slice_index[idx]
        return {
            "case_id": meta["case_id"],
            "slice_index": meta["slice_index"],
            "channels": MULTIMODAL_CHANNEL_ORDER
        }


def create_multimodal_dataloaders(
    data_dir: str = "data/raw/brats",
    manifest_path: str = "data/processed/brats/split_manifest.json",
    batch_size: int = 4,
    num_workers: int = 0,
    augment_train: bool = True,
    seed: int = 42
) -> Dict[str, DataLoader]:
    """
    Factory function to build DataLoader instances for 4-channel MultiModalBraTS2DDataset.
    """
    train_ds = MultiModalBraTS2DDataset(data_dir=data_dir, manifest_path=manifest_path, split="train", augment=augment_train, seed=seed)
    val_ds = MultiModalBraTS2DDataset(data_dir=data_dir, manifest_path=manifest_path, split="val", augment=False, seed=seed)
    test_ds = MultiModalBraTS2DDataset(data_dir=data_dir, manifest_path=manifest_path, split="test", augment=False, seed=seed)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader
    }
