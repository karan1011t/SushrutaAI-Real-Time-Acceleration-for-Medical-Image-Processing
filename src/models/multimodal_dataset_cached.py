"""
PyTorch Multi-Modal Dataset with Bounded Lazy Volume Caching for BraTS MRI Segmentation.

Optimized for FUTURE training runs to remove repeated NIfTI (.nii.gz) file load and decompression overhead.

Guarantees:
1. Lazy per-patient volume caching: Each patient's 4 modalities + segmentation are decompressed at most once per DataLoader process/worker.
2. Memory Bounded: Uses LRU eviction (OrderedDict) with configurable `max_cached_patients` (default: 4).
3. Parity: Output tensors, channel ordering [FLAIR, T1, T1CE, T2], normalization, slice selection, mask semantics, and deterministic seed augmentation are bitwise/numerically identical to uncached MultiModalBraTS2DDataset.
"""

from collections import OrderedDict
from pathlib import Path
import random
from typing import List, Dict, Tuple, Optional, Any

import numpy as np
import nibabel as nib
import torch
from torch.utils.data import DataLoader

from src.models.multimodal_dataset import MultiModalBraTS2DDataset, MULTIMODAL_CHANNEL_ORDER


class CachedMultiModalBraTS2DDataset(MultiModalBraTS2DDataset):
    """
    Subclass of MultiModalBraTS2DDataset adding bounded LRU per-patient volume caching.
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
        seed: int = 42,
        max_cached_patients: int = 4
    ):
        super().__init__(
            data_dir=data_dir,
            manifest_path=manifest_path,
            split=split,
            case_ids=case_ids,
            min_brain_pixels=min_brain_pixels,
            min_tumor_pixels=min_tumor_pixels,
            augment=augment,
            seed=seed
        )
        self.max_cached_patients = max_cached_patients
        self.volume_cache: OrderedDict = OrderedDict()
        self.cache_hits = 0
        self.cache_misses = 0

    def _load_patient_volumes(self, cid: str, paths: Dict[str, Path]) -> Dict[str, np.ndarray]:
        """Loads all 4 modality 3D volumes + SEG 3D volume into memory as numpy arrays."""
        patient_data = {}
        for mod in MULTIMODAL_CHANNEL_ORDER:
            img = nib.load(str(paths[mod]))
            patient_data[mod] = np.asarray(img.dataobj[:, :, :], dtype=np.float32)

        seg_img = nib.load(str(paths["SEG"]))
        patient_data["SEG"] = np.asarray(seg_img.dataobj[:, :, :])
        return patient_data

    def _get_patient_volumes(self, cid: str, paths: Dict[str, Path]) -> Dict[str, np.ndarray]:
        """Retrieves patient 3D volume arrays from LRU cache or loads them from disk."""
        if self.max_cached_patients <= 0:
            self.cache_misses += 1
            return self._load_patient_volumes(cid, paths)

        if cid in self.volume_cache:
            self.cache_hits += 1
            self.volume_cache.move_to_end(cid)
            return self.volume_cache[cid]

        self.cache_misses += 1
        patient_data = self._load_patient_volumes(cid, paths)
        self.volume_cache[cid] = patient_data
        self.volume_cache.move_to_end(cid)

        if len(self.volume_cache) > self.max_cached_patients:
            self.volume_cache.popitem(last=False)  # Evict oldest (LRU)

        return patient_data

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample_meta = self.slice_index[idx]
        cid = sample_meta["case_id"]
        slice_idx = sample_meta["slice_index"]
        paths = sample_meta["modality_paths"]

        if self.max_cached_patients > 0:
            patient_data = self._get_patient_volumes(cid, paths)
            channels_2d = []
            for mod in MULTIMODAL_CHANNEL_ORDER:
                slice_data = patient_data[mod][:, :, slice_idx]
                norm_slice = self.normalize_channel(slice_data)
                channels_2d.append(norm_slice)

            image_4ch = np.stack(channels_2d, axis=0)

            seg_slice = patient_data["SEG"][:, :, slice_idx]
            mask_wt = self.map_brats_segmentation_target(seg_slice)
            mask_1ch = np.expand_dims(mask_wt, axis=0)
        else:
            # Fallback to uncached parent method logic
            channels_2d = []
            for mod in MULTIMODAL_CHANNEL_ORDER:
                img = nib.load(str(paths[mod]))
                slice_data = np.asarray(img.dataobj[:, :, slice_idx], dtype=np.float32)
                norm_slice = self.normalize_channel(slice_data)
                channels_2d.append(norm_slice)

            image_4ch = np.stack(channels_2d, axis=0)

            seg_img = nib.load(str(paths["SEG"]))
            seg_slice = np.asarray(seg_img.dataobj[:, :, slice_idx])
            mask_wt = self.map_brats_segmentation_target(seg_slice)
            mask_1ch = np.expand_dims(mask_wt, axis=0)

        image_tensor = torch.from_numpy(np.ascontiguousarray(image_4ch)).float()
        mask_tensor = torch.from_numpy(np.ascontiguousarray(mask_1ch)).float()

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


def create_cached_multimodal_dataloaders(
    data_dir: str = "data/raw/brats",
    manifest_path: str = "data/processed/brats/split_manifest.json",
    batch_size: int = 16,
    num_workers: int = 0,
    augment_train: bool = True,
    seed: int = 42,
    max_cached_patients: int = 4
) -> Dict[str, DataLoader]:
    """
    Factory function to build DataLoader instances using CachedMultiModalBraTS2DDataset.
    """
    train_ds = CachedMultiModalBraTS2DDataset(
        data_dir=data_dir,
        manifest_path=manifest_path,
        split="train",
        augment=augment_train,
        seed=seed,
        max_cached_patients=max_cached_patients
    )
    val_ds = CachedMultiModalBraTS2DDataset(
        data_dir=data_dir,
        manifest_path=manifest_path,
        split="val",
        augment=False,
        seed=seed,
        max_cached_patients=max_cached_patients
    )
    test_ds = CachedMultiModalBraTS2DDataset(
        data_dir=data_dir,
        manifest_path=manifest_path,
        split="test",
        augment=False,
        seed=seed,
        max_cached_patients=max_cached_patients
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader
    }
