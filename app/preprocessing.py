"""
SUSHRUTA Preprocessing Module for External Medical Images.

Reproduces exact SUSHRUTA FLAIR MRI preprocessing pipeline:
1. Grayscale conversion.
2. Non-zero brain intensity normalization (z-score on non-zero pixels + min-max scaling to [0.0, 1.0]).
3. Spatial resizing to (256, 256) via bilinear interpolation.
4. Tensor formatting: PyTorch FloatTensor (1, 1, 256, 256).
"""

import io
import os
import time
from typing import Tuple, Dict, Any, Optional
import numpy as np
import cv2
from PIL import Image
import torch

# Reuse authoritative normalization function from src.preprocessing.brats_preprocessing
from src.preprocessing.brats_preprocessing import normalize_flair_intensity


SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".nii", ".gz"}


def preprocess_image_file(
    file_bytes: bytes,
    filename: str,
    target_shape: Tuple[int, int] = (256, 256)
) -> Tuple[torch.Tensor, np.ndarray, Dict[str, Any], float]:
    """
    Preprocesses uploaded external 2D MRI image bytes to match SUSHRUTA U-Net input representation.

    Parameters:
        file_bytes (bytes): Raw bytes of uploaded image file.
        filename (str): Original filename used for extension checking.
        target_shape (Tuple[int, int]): Target spatial resolution, default (256, 256).

    Returns:
        input_tensor (torch.Tensor): PyTorch tensor of shape (1, 1, 256, 256), float32, range [0.0, 1.0].
        norm_img_np (np.ndarray): Preprocessed 2D numpy array of shape (256, 256), float32 in [0.0, 1.0].
        meta (Dict[str, Any]): Image metadata dict containing original dimensions and format.
        preprocessing_time_ms (float): Preprocessing execution time in milliseconds.
    """
    t_start = time.perf_counter()

    ext = os.path.splitext(filename)[1].lower()
    if filename.endswith(".nii.gz"):
        ext = ".nii.gz"

    if ext not in SUPPORTED_IMAGE_EXTENSIONS:
        raise ValueError(
            f"Unsupported file format '{ext}'. Supported formats: PNG, JPG, JPEG, BMP, TIFF, NIfTI (.nii/.nii.gz)"
        )

    # -------------------------------------------------------------------------
    # 1. READ IMAGE ARRAY
    # -------------------------------------------------------------------------
    try:
        if ext in (".nii", ".nii.gz"):
            import nibabel as nib
            with io.BytesIO(file_bytes) as bio:
                # Nibabel requires a file path or FileHolder; use temporary buffer or write to temp
                nii_file = bio
                # Handle via nibabel FileHolder / image API
                img_nifti = nib.Nifti1Image.from_bytes(file_bytes)
                nii_data = img_nifti.get_fdata().astype(np.float32)

                # Extract 2D axial slice if 3D volume
                if nii_data.ndim == 3:
                    mid_z = nii_data.shape[2] // 2
                    raw_2d = nii_data[:, :, mid_z]
                elif nii_data.ndim == 2:
                    raw_2d = nii_data
                else:
                    raise ValueError(f"Invalid NIfTI volume dimensions: {nii_data.ndim}D")
                orig_shape = [raw_2d.shape[0], raw_2d.shape[1]]
        else:
            pil_img = Image.open(io.BytesIO(file_bytes))
            pil_img.verify()  # Check for corruption
            pil_img = Image.open(io.BytesIO(file_bytes))  # Re-open after verify

            orig_shape = [pil_img.height, pil_img.width]

            # Convert to grayscale 8-bit / float
            if pil_img.mode != "L":
                pil_img = pil_img.convert("L")

            raw_2d = np.array(pil_img, dtype=np.float32)

    except Exception as exc:
        raise ValueError(f"Failed to read or decode image file '{filename}': {str(exc)}") from exc

    if raw_2d.size == 0 or raw_2d.ndim != 2:
        raise ValueError("Decoded image is empty or not a valid 2D spatial array")

    # -------------------------------------------------------------------------
    # 2. INTENSITY NORMALIZATION
    # -------------------------------------------------------------------------
    if ext in (".nii", ".nii.gz"):
        # NIfTI uploads contain raw, un-normalized MRI signal intensity values.
        # Apply non-zero FLAIR z-score and min-max normalization.
        norm_2d = normalize_flair_intensity(raw_2d)
    else:
        # Standard 2D image uploads (PNG, JPG, BMP, TIFF) originate from preprocessed BraTS 8-bit slices.
        # The trained U-Net consumes the preprocessed BraTS PNG representation as float32 [0,1]. Do not re-normalize these images.
        norm_2d = raw_2d.astype(np.float32) / 255.0

    # -------------------------------------------------------------------------
    # 3. SPATIAL RESIZING TO 256x256
    # -------------------------------------------------------------------------
    if norm_2d.shape != target_shape:
        resized_2d = cv2.resize(
            norm_2d,
            (target_shape[1], target_shape[0]),
            interpolation=cv2.INTER_LINEAR
        )
    else:
        resized_2d = norm_2d

    norm_img_np = np.clip(resized_2d, 0.0, 1.0).astype(np.float32)

    # -------------------------------------------------------------------------
    # 4. PYTORCH TENSOR FORMATTING
    # -------------------------------------------------------------------------
    input_tensor = torch.from_numpy(np.ascontiguousarray(norm_img_np)).unsqueeze(0).unsqueeze(0).float()

    t_end = time.perf_counter()
    preprocessing_time_ms = (t_end - t_start) * 1000.0

    meta = {
        "original_shape": orig_shape,
        "processed_shape": list(target_shape),
        "filename": filename,
        "format": ext
    }

    return input_tensor, norm_img_np, meta, preprocessing_time_ms
