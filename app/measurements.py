"""
SUSHRUTA Morphometric Measurements & Spatial Analytics Module (Refined Phase 2).

Calculates tumor morphometric metrics from 2D binary segmentation mask:
- Tumor detection status
- Total pixel area and slice area percentage
- Bounding box (x, y, width, height) in pixels
- Maximum bounding diameter in pixels
- Number of connected tumor components
- Largest connected component area and bounding box
- Tumor spatial centroid (center of mass)
- Mean tumor probability confidence score
"""

from typing import Dict, Any, Tuple
import numpy as np
import cv2


def calculate_tumor_measurements(
    binary_mask: np.ndarray,
    prob_map: np.ndarray
) -> Tuple[Dict[str, Any], float]:
    """
    Computes spatial morphometric measurements from predicted binary mask and probability map.

    Parameters:
        binary_mask (np.ndarray): 2D binary numpy array of shape (256, 256), dtype uint8 {0, 1}.
        prob_map (np.ndarray): 2D float numpy array of shape (256, 256) containing sigmoid probabilities in [0.0, 1.0].

    Returns:
        measurements (Dict[str, Any]): Dictionary of calculated spatial morphometric metrics.
        segmentation_confidence (float): Mean tumor probability over predicted tumor pixels.
    """
    assert binary_mask.shape == (256, 256), f"Expected mask shape (256, 256), got {binary_mask.shape}"
    assert prob_map.shape == (256, 256), f"Expected prob_map shape (256, 256), got {prob_map.shape}"

    tumor_pixels = np.where(binary_mask > 0)
    area_pixels = int(len(tumor_pixels[0]))
    tumor_detected = area_pixels > 0

    total_pixels = 256 * 256
    area_percentage = round((area_pixels / total_pixels) * 100.0, 2)

    if tumor_detected:
        min_y = int(np.min(tumor_pixels[0]))
        max_y = int(np.max(tumor_pixels[0]))
        min_x = int(np.min(tumor_pixels[1]))
        max_x = int(np.max(tumor_pixels[1]))

        bbox_x = min_x
        bbox_y = min_y
        bbox_w = int(max_x - min_x + 1)
        bbox_h = int(max_y - min_y + 1)

        # Diameter defined as maximum bounding dimension in 2D pixels
        diameter_pixels = max(bbox_w, bbox_h)

        # Connected Component Analysis via 8-connectivity
        mask_uint8 = (binary_mask > 0).astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)

        # Component 0 is background; components 1..num_labels-1 are tumor islands
        num_connected_components = int(num_labels - 1)

        if num_connected_components > 0:
            # Find index of largest tumor component by area
            component_areas = stats[1:, cv2.CC_STAT_AREA]
            largest_comp_idx = int(np.argmax(component_areas)) + 1

            largest_area = int(stats[largest_comp_idx, cv2.CC_STAT_AREA])
            lc_x = int(stats[largest_comp_idx, cv2.CC_STAT_LEFT])
            lc_y = int(stats[largest_comp_idx, cv2.CC_STAT_TOP])
            lc_w = int(stats[largest_comp_idx, cv2.CC_STAT_WIDTH])
            lc_h = int(stats[largest_comp_idx, cv2.CC_STAT_HEIGHT])

            # Overall tumor centroid (center of mass across all tumor pixels)
            centroid_x = round(float(np.mean(tumor_pixels[1])), 2)
            centroid_y = round(float(np.mean(tumor_pixels[0])), 2)
        else:
            num_connected_components = 0
            largest_area = 0
            lc_x, lc_y, lc_w, lc_h = 0, 0, 0, 0
            centroid_x, centroid_y = 0.0, 0.0

        # Mean tumor probability over predicted tumor pixels
        tumor_probs = prob_map[binary_mask > 0]
        segmentation_confidence = float(np.mean(tumor_probs))
        segmentation_confidence = round(segmentation_confidence, 4)
    else:
        bbox_x, bbox_y, bbox_w, bbox_h = 0, 0, 0, 0
        diameter_pixels = 0
        num_connected_components = 0
        largest_area = 0
        lc_x, lc_y, lc_w, lc_h = 0, 0, 0, 0
        centroid_x, centroid_y = 0.0, 0.0
        segmentation_confidence = 0.0

    measurements = {
        "tumor_detected": tumor_detected,
        "area_pixels": area_pixels,
        "area_percentage": area_percentage,
        "bounding_box": {
            "x": bbox_x,
            "y": bbox_y,
            "width": bbox_w,
            "height": bbox_h
        },
        "width_pixels": bbox_w,
        "height_pixels": bbox_h,
        "diameter_pixels": diameter_pixels,
        "num_connected_components": num_connected_components,
        "largest_component_area_pixels": largest_area,
        "centroid": {
            "x": centroid_x,
            "y": centroid_y
        },
        "largest_component_bounding_box": {
            "x": lc_x,
            "y": lc_y,
            "width": lc_w,
            "height": lc_h
        },
        "spatial_unit": "pixels",
        "diameter_definition": "Maximum bounding box spatial dimension (max(width, height)) in pixels",
        "note": "Physical millimeter measurements require NIfTI/DICOM voxel spacing metadata."
    }

    return measurements, segmentation_confidence
