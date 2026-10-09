"""
SUSHRUTA Visual Output & Artifact Generation Utilities (Phase 2 LayerCAM Support).

Generates 7 high-resolution diagnostic visualization artifacts:
1. Original preprocessed grayscale MRI slice.
2. Binary tumor segmentation mask.
3. MRI + tumor-mask semi-transparent RGB overlay with bounding box.
4. Continuous probability heatmap blended with MRI structure.
5. LayerCAM attribution heatmap.
6. MRI + LayerCAM overlay visualization.
7. Combined MRI + LayerCAM + tumor segmentation contour overlay.
"""

import os
import uuid
from typing import Dict, Any, Tuple, Optional
import numpy as np
import cv2
from PIL import Image


def generate_visualization_artifacts(
    preprocessed_mri: np.ndarray,
    binary_mask: np.ndarray,
    prob_map: np.ndarray,
    output_dir: str,
    layercam_map: Optional[np.ndarray] = None,
    base_id: Optional[str] = None
) -> Dict[str, str]:
    """
    Renders and saves visual artifact PNG files for an inference request.

    Parameters:
        preprocessed_mri (np.ndarray): 2D float numpy array (256, 256) in [0.0, 1.0].
        binary_mask (np.ndarray): 2D uint8 numpy array (256, 256) containing {0, 1}.
        prob_map (np.ndarray): 2D float numpy array (256, 256) containing probabilities in [0.0, 1.0].
        output_dir (str): Directory where generated image files are saved.
        layercam_map (Optional[np.ndarray]): 2D float numpy array (256, 256) of LayerCAM values in [0.0, 1.0].
        base_id (Optional[str]): Unique identifier for request output filenames.

    Returns:
        artifact_paths (Dict[str, str]): Dictionary mapping artifact names to saved file paths / filenames.
    """
    os.makedirs(output_dir, exist_ok=True)

    if not base_id:
        base_id = uuid.uuid4().hex[:10]

    # Convert preprocessed MRI to uint8 [0, 255]
    mri_uint8 = np.clip(preprocessed_mri * 255.0, 0, 255).astype(np.uint8)
    mri_rgb = np.stack([mri_uint8, mri_uint8, mri_uint8], axis=-1).astype(np.float32)

    # -------------------------------------------------------------------------
    # 1. PREPROCESSED MRI
    # -------------------------------------------------------------------------
    mri_filename = f"{base_id}_mri.png"
    mri_path = os.path.join(output_dir, mri_filename)
    Image.fromarray(mri_uint8).save(mri_path)

    # -------------------------------------------------------------------------
    # 2. BINARY TUMOR MASK
    # -------------------------------------------------------------------------
    mask_uint8 = (binary_mask * 255).astype(np.uint8)
    mask_filename = f"{base_id}_mask.png"
    mask_path = os.path.join(output_dir, mask_filename)
    Image.fromarray(mask_uint8).save(mask_path)

    # -------------------------------------------------------------------------
    # 3. MRI + TUMOR OVERLAY (Vibrant Green semi-transparent overlay + Bounding Box)
    # -------------------------------------------------------------------------
    overlay_rgb = mri_rgb.copy()
    tumor_indices = binary_mask > 0

    if np.any(tumor_indices):
        green_tint = np.zeros_like(mri_rgb)
        green_tint[:, :, 1] = 255.0  # Green channel
        alpha = 0.45
        overlay_rgb[tumor_indices] = (
            (1.0 - alpha) * mri_rgb[tumor_indices] + alpha * green_tint[tumor_indices]
        )

    overlay_uint8 = np.clip(overlay_rgb, 0, 255).astype(np.uint8)

    if np.any(tumor_indices):
        ys, xs = np.where(tumor_indices)
        min_y, max_y = int(np.min(ys)), int(np.max(ys))
        min_x, max_x = int(np.min(xs)), int(np.max(xs))
        cv2.rectangle(
            overlay_uint8,
            (min_x, min_y),
            (max_x, max_y),
            (255, 220, 0),  # Bright Yellow Bounding Box
            thickness=1
        )

    overlay_filename = f"{base_id}_overlay.png"
    overlay_path = os.path.join(output_dir, overlay_filename)
    Image.fromarray(overlay_uint8).save(overlay_path)

    # -------------------------------------------------------------------------
    # 4. PROBABILITY HEATMAP (JET colormap blended with MRI)
    # -------------------------------------------------------------------------
    prob_uint8 = np.clip(prob_map * 255.0, 0, 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(prob_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    blended_heatmap = (0.65 * heatmap_rgb.astype(np.float32) + 0.35 * mri_rgb)
    blended_uint8 = np.clip(blended_heatmap, 0, 255).astype(np.uint8)

    heatmap_filename = f"{base_id}_heatmap.png"
    heatmap_path = os.path.join(output_dir, heatmap_filename)
    Image.fromarray(blended_uint8).save(heatmap_path)

    # -------------------------------------------------------------------------
    # PHASE 2 ARTIFACTS — LAYERCAM HEATMAP, OVERLAY, & COMBINED VISUALIZATION
    # -------------------------------------------------------------------------
    artifacts = {
        "preprocessed_mri": mri_filename,
        "tumor_mask": mask_filename,
        "overlay": overlay_filename,
        "probability_map": heatmap_filename,
        "mri_path": mri_path,
        "mask_path": mask_path,
        "overlay_path": overlay_path,
        "heatmap_path": heatmap_path
    }

    if layercam_map is not None:
        cam_uint8 = np.clip(layercam_map * 255.0, 0, 255).astype(np.uint8)

        # 5. LayerCAM Standalone Heatmap (JET colormap)
        cam_colored = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
        cam_rgb = cv2.cvtColor(cam_colored, cv2.COLOR_BGR2RGB)

        layercam_filename = f"{base_id}_layercam.png"
        layercam_path = os.path.join(output_dir, layercam_filename)
        Image.fromarray(cam_rgb).save(layercam_path)

        # 6. MRI + LayerCAM Overlay (60% LayerCAM + 40% MRI base)
        cam_overlay_rgb = (0.60 * cam_rgb.astype(np.float32) + 0.40 * mri_rgb)
        cam_overlay_uint8 = np.clip(cam_overlay_rgb, 0, 255).astype(np.uint8)

        layercam_overlay_filename = f"{base_id}_layercam_overlay.png"
        layercam_overlay_path = os.path.join(output_dir, layercam_overlay_filename)
        Image.fromarray(cam_overlay_uint8).save(layercam_overlay_path)

        # 7. Combined MRI + LayerCAM + Tumor Contour Overlay
        combined_uint8 = cam_overlay_uint8.copy()
        if np.any(tumor_indices):
            # Find contours of predicted binary mask
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # Draw bright cyan contour outline over LayerCAM overlay
            cv2.drawContours(combined_uint8, contours, -1, (0, 255, 255), thickness=1)

        combined_filename = f"{base_id}_combined_overlay.png"
        combined_path = os.path.join(output_dir, combined_filename)
        Image.fromarray(combined_uint8).save(combined_path)

        artifacts["layercam"] = layercam_filename
        artifacts["layercam_overlay"] = layercam_overlay_filename
        artifacts["combined_overlay"] = combined_filename
        artifacts["layercam_path"] = layercam_path
        artifacts["layercam_overlay_path"] = layercam_overlay_path
        artifacts["combined_overlay_path"] = combined_path

    return artifacts


def generate_classifier_artifact(
    preprocessed_mri: np.ndarray,
    classifier_cam_map: np.ndarray,
    output_dir: str,
    base_id: Optional[str] = None
) -> Tuple[str, str]:
    """
    Renders MRI + Classifier LayerCAM heatmap overlay PNG artifact.
    Returns (filename, absolute_path).
    """
    os.makedirs(output_dir, exist_ok=True)
    if not base_id:
        base_id = uuid.uuid4().hex[:10]

    mri_uint8 = np.clip(preprocessed_mri * 255.0, 0, 255).astype(np.uint8)
    mri_rgb = np.stack([mri_uint8, mri_uint8, mri_uint8], axis=-1).astype(np.float32)

    cam_uint8 = np.clip(classifier_cam_map * 255.0, 0, 255).astype(np.uint8)
    cam_colored = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    cam_rgb = cv2.cvtColor(cam_colored, cv2.COLOR_BGR2RGB)

    # 60% LayerCAM + 40% MRI
    cam_overlay_rgb = (0.60 * cam_rgb.astype(np.float32) + 0.40 * mri_rgb)
    cam_overlay_uint8 = np.clip(cam_overlay_rgb, 0, 255).astype(np.uint8)

    filename = f"{base_id}_classifier_layercam.png"
    filepath = os.path.join(output_dir, filename)
    Image.fromarray(cam_overlay_uint8).save(filepath)

    return filename, filepath
