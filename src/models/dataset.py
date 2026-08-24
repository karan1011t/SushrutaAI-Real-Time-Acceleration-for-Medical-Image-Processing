"""
PyTorch Dataset and DataLoader Utilities for 2D BraTS MRI Segmentation.
"""

import os
import random
from typing import Tuple, Dict, Optional, List
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader


class BraTS2DDataset(Dataset):
    """
    PyTorch Dataset for 2D BraTS MRI Slices and Binary Segmentation Masks.

    Parameters:
        data_dir (str): Root directory containing processed splits (e.g., 'data/processed/brats').
        split (str): Dataset split ('train', 'val', or 'test').
        augment (bool): Whether to apply simple medical augmentations (horizontal flip, small rotations).
        seed (int): Seed for random augmentation reproducibility.
    """

    def __init__(
        self,
        data_dir: str = "data/processed/brats",
        split: str = "train",
        augment: bool = False,
        seed: int = 42
    ):
        super().__init__()
        self.data_dir = data_dir
        self.split = split.lower()
        self.augment = augment
        self.seed = seed

        self.img_dir = os.path.join(data_dir, self.split, "images")
        self.mask_dir = os.path.join(data_dir, self.split, "masks")

        if not os.path.exists(self.img_dir) or not os.path.exists(self.mask_dir):
            raise FileNotFoundError(
                f"Missing split directory for '{self.split}'. Expected:\n"
                f"  Images: {self.img_dir}\n"
                f"  Masks:  {self.mask_dir}"
            )

        self.image_filenames = sorted([
            f for f in os.listdir(self.img_dir)
            if f.endswith(".png") or f.endswith(".PNG")
        ])

        # Verify 1-to-1 match
        mask_filenames = set(os.listdir(self.mask_dir))
        for fname in self.image_filenames:
            if fname not in mask_filenames:
                raise FileNotFoundError(f"Missing corresponding mask file for image '{fname}' in {self.mask_dir}")

    def __len__(self) -> int:
        return len(self.image_filenames)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        fname = self.image_filenames[idx]
        img_path = os.path.join(self.img_dir, fname)
        mask_path = os.path.join(self.mask_dir, fname)

        img_pil = Image.open(img_path)
        mask_pil = Image.open(mask_path)

        img_arr = np.array(img_pil, dtype=np.float32) / 255.0
        mask_arr = np.array(mask_pil, dtype=np.float32)

        # Ensure binary thresholding on mask
        mask_arr = np.where(mask_arr > 0.5, 1.0, 0.0).astype(np.float32)

        # Simple medical augmentation (horizontal flip & 90-degree steps)
        if self.augment:
            rng = random.Random(self.seed + idx)
            if rng.random() > 0.5:
                img_arr = np.fliplr(img_arr)
                mask_arr = np.fliplr(mask_arr)
            k = rng.choice([0, 1, 2, 3])
            if k > 0:
                img_arr = np.rot90(img_arr, k)
                mask_arr = np.rot90(mask_arr, k)

        # Convert to float32 tensors with channel dimension [1, H, W]
        img_tensor = torch.from_numpy(np.ascontiguousarray(img_arr)).unsqueeze(0).float()
        mask_tensor = torch.from_numpy(np.ascontiguousarray(mask_arr)).unsqueeze(0).float()

        return img_tensor, mask_tensor


def create_brats_dataloaders(
    data_dir: str = "data/processed/brats",
    batch_size: int = 4,
    num_workers: int = 0,
    augment_train: bool = False,
    seed: int = 42
) -> Dict[str, DataLoader]:
    """
    Factory function to create DataLoader instances for train, val, and test splits.
    """
    train_dataset = BraTS2DDataset(data_dir=data_dir, split="train", augment=augment_train, seed=seed)
    val_dataset = BraTS2DDataset(data_dir=data_dir, split="val", augment=False, seed=seed)
    test_dataset = BraTS2DDataset(data_dir=data_dir, split="test", augment=False, seed=seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader
    }


if __name__ == "__main__":
    loaders = create_brats_dataloaders(batch_size=4, num_workers=0)
    for name, loader in loaders.items():
        img_batch, mask_batch = next(iter(loader))
        print(f"[TEST] Split '{name}': {len(loader.dataset)} samples, {len(loader)} batches")
        print(f"       Image Batch Shape: {img_batch.shape}, dtype: {img_batch.dtype}, min/max: {img_batch.min():.2f}/{img_batch.max():.2f}")
        print(f"       Mask Batch Shape:  {mask_batch.shape}, dtype: {mask_batch.dtype}, unique: {torch.unique(mask_batch).tolist()}")
