"""
SUSHRUTA Morphometric Measurements & Spatial Analytics Module (Phase 3A: Quantitative 2D Morphology).

Calculates tumor morphometric metrics from 2D binary segmentation mask:
- Tumor detection status
- Total pixel area and slice area percentage
- Bounding box (x, y, width, height) in processed-grid pixels
- Maximum bounding diameter in processed-grid pixels
- Number of connected tumor components
- Largest connected component area and bounding box
- Tumor spatial centroid (center of mass)
- Mean tumor probability confidence score
- 2D Morphological Descriptors (Primary Lesion Mass):
  * Perimeter in processed-grid pixels (contour-perimeter convention)
  * Circularity (isoperimetric quotient 4*pi*area / perimeter^2)
  * Solidity (component area / convex-hull pixel area)
  * Major and minor equivalent inertia ellipse axis lengths
  * Eccentricity derived consistently from inertia axes
  * Maximum Feret caliper diameter via convex hull vertices
"""

from typing import Dict, Any, Tuple, Optional
import math
import numpy as np
import cv2


def _compute_shape_descriptors(
    mask_uint8: Optional[np.ndarray],
    comp_idx: int,
    labels: Optional[np.ndarray],
    stats: Optional[np.ndarray]
) -> Dict[str, Any]:
    """
    Computes mathematically rigorous 2D morphological descriptors for a specified component.

    Aggregation Policy:
        Morphological descriptors are computed on the primary tumor mass (largest
        connected component). In clinical oncology, morphological shape analysis
        focuses on the primary mass rather than aggregating disconnected satellite
        islands into a fictitious non-contiguous hull.

    Coordinate & Raster Conventions:
        - All measurements are in processed-grid pixels (no physical mm without DICOM/NIfTI spacing).
        - Perimeter is computed via cv2.arcLength(contour, closed=True), tracing pixel centers
          along the 8-connected external boundary.
        - Circularity is 4 * pi * area / (perimeter^2). For rasterized shapes, discrete pixelation
          causes slight deviations from continuous analytical values; values are clamped to [0.0, 1.0].
        - Solidity is area / convex_hull_raster_area. Raster hull area avoids continuous polygon
          Green's theorem offset anomalies on discrete grids.
        - Major/minor axes are derived from the equivalent inertia ellipse second-order central moments
          (lambda_1, lambda_2 eigenvalues of the spatial covariance matrix).
        - Eccentricity is sqrt(1 - lambda_2 / lambda_1), with safe [0.0, 1.0] clamping.
        - Feret diameter is the true maximum Euclidean distance between boundary vertices on the convex hull.
    """
    empty_result: Dict[str, Any] = {
        "perimeter_pixels": None,
        "circularity": None,
        "solidity": None,
        "major_axis_length": None,
        "minor_axis_length": None,
        "eccentricity": None,
        "feret_diameter_max": None,
        "aggregation_policy": "largest_connected_component",
        "coordinate_system": "processed_mask_grid"
    }

    if mask_uint8 is None or labels is None or stats is None or comp_idx <= 0:
        return empty_result

    comp_area = int(stats[comp_idx, cv2.CC_STAT_AREA])
    if comp_area <= 0:
        return empty_result

    comp_mask = (labels == comp_idx).astype(np.uint8)

    # Degenerate: single-pixel or very small component
    if comp_area < 2:
        return {
            "perimeter_pixels": None,
            "circularity": None,
            "solidity": 1.0,
            "major_axis_length": None,
            "minor_axis_length": None,
            "eccentricity": None,
            "feret_diameter_max": None,
            "aggregation_policy": "largest_connected_component",
            "coordinate_system": "processed_mask_grid"
        }

    # Extract external contours
    contours, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return empty_result

    # Select primary external contour
    contour = max(contours, key=lambda c: (cv2.contourArea(c), len(c)))
    n_pts = len(contour)

    if n_pts < 3:
        # Collinear or degenerate boundary
        return {
            "perimeter_pixels": None,
            "circularity": None,
            "solidity": 1.0,
            "major_axis_length": None,
            "minor_axis_length": None,
            "eccentricity": None,
            "feret_diameter_max": None,
            "aggregation_policy": "largest_connected_component",
            "coordinate_system": "processed_mask_grid"
        }

    # 1. Perimeter (external contour arc length in processed-grid pixels)
    arc_len = float(cv2.arcLength(contour, closed=True))
    perimeter_pixels = round(arc_len, 2) if arc_len > 0.0 else None

    # 2. Circularity: 4 * pi * area / (perimeter^2)
    circularity = None
    if perimeter_pixels is not None and perimeter_pixels > 0.0 and comp_area > 0:
        raw_circ = float(4.0 * math.pi * comp_area / (arc_len ** 2))
        circularity = round(min(1.0, max(0.0, raw_circ)), 4)

    # 3. Solidity: foreground area / raster convex hull area
    hull = cv2.convexHull(contour)
    hull_mask = np.zeros_like(comp_mask, dtype=np.uint8)
    cv2.drawContours(hull_mask, [hull], -1, 1, thickness=-1)
    hull_pixel_area = int(np.sum(hull_mask > 0))

    if hull_pixel_area > 0:
        solidity = round(min(1.0, max(0.0, float(comp_area) / float(hull_pixel_area))), 4)
    else:
        solidity = None

    # 4 & 5. Major/Minor axis lengths & Eccentricity via 2nd-order central moments
    moments = cv2.moments(comp_mask, binaryImage=True)
    m00 = moments["m00"]

    major_axis_length = None
    minor_axis_length = None
    eccentricity = None

    if m00 >= 2:
        mu20_p = moments["mu20"] / m00
        mu02_p = moments["mu02"] / m00
        mu11_p = moments["mu11"] / m00

        delta = math.sqrt((mu20_p - mu02_p) ** 2 + 4.0 * (mu11_p ** 2))
        lambda1 = (mu20_p + mu02_p + delta) / 2.0
        lambda2 = (mu20_p + mu02_p - delta) / 2.0

        if lambda1 > 0:
            major_axis = 4.0 * math.sqrt(max(0.0, lambda1))
            minor_axis = 4.0 * math.sqrt(max(0.0, lambda2))
            major_axis_length = round(major_axis, 2)
            minor_axis_length = round(minor_axis, 2)

            ratio = min(1.0, max(0.0, lambda2 / lambda1))
            eccentricity = round(math.sqrt(max(0.0, 1.0 - ratio)), 4)

    # 6. Maximum Feret diameter (maximum Euclidean distance between convex hull vertices)
    pts = hull.reshape(-1, 2).astype(np.float64)
    if len(pts) >= 2 and comp_area >= 2:
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs ** 2, axis=-1))
        max_dist = float(np.max(dists))
        feret_diameter_max = round(max_dist, 2) if max_dist > 0.0 else None
    else:
        feret_diameter_max = None

    return {
        "perimeter_pixels": perimeter_pixels,
        "circularity": circularity,
        "solidity": solidity,
        "major_axis_length": major_axis_length,
        "minor_axis_length": minor_axis_length,
        "eccentricity": eccentricity,
        "feret_diameter_max": feret_diameter_max,
        "aggregation_policy": "largest_connected_component",
        "coordinate_system": "processed_mask_grid"
    }


def calculate_tumor_measurements(
    binary_mask: np.ndarray,
    prob_map: np.ndarray
) -> Tuple[Dict[str, Any], float]:
    """
    Computes spatial morphometric measurements and 2D shape descriptors
    from predicted binary mask and probability map.

    Parameters:
        binary_mask (np.ndarray): 2D binary numpy array, dtype uint8 or bool {0, 1}.
        prob_map (np.ndarray): 2D float numpy array matching binary_mask shape,
                               containing sigmoid probabilities in [0.0, 1.0].

    Returns:
        measurements (Dict[str, Any]): Dictionary of calculated spatial morphometric metrics.
        segmentation_confidence (float): Mean tumor probability over predicted tumor pixels.
    """
    # 1. Clean input validation (reject non-ndarray, non-2D, mismatched, or empty shapes)
    if not isinstance(binary_mask, np.ndarray) or not isinstance(prob_map, np.ndarray):
        raise TypeError(
            f"binary_mask and prob_map must be numpy.ndarray, got {type(binary_mask)} and {type(prob_map)}"
        )

    if binary_mask.ndim != 2 or prob_map.ndim != 2:
        raise ValueError(
            f"Expected 2D arrays, got binary_mask ndim={binary_mask.ndim} and prob_map ndim={prob_map.ndim}"
        )

    if binary_mask.shape != prob_map.shape:
        raise ValueError(
            f"Shape mismatch: binary_mask {binary_mask.shape} vs prob_map {prob_map.shape}"
        )

    if binary_mask.shape[0] == 0 or binary_mask.shape[1] == 0:
        raise ValueError(f"Invalid empty dimensions: shape={binary_mask.shape}")

    # Sanitize non-finite values safely
    prob_map = np.nan_to_num(prob_map, nan=0.0, posinf=1.0, neginf=0.0)
    binary_mask = np.nan_to_num(binary_mask, nan=0.0, posinf=0.0, neginf=0.0)

    tumor_pixels = np.where(binary_mask > 0)
    area_pixels = int(len(tumor_pixels[0]))
    tumor_detected = area_pixels > 0

    total_pixels = int(binary_mask.shape[0] * binary_mask.shape[1])
    area_percentage = round((area_pixels / total_pixels) * 100.0, 2) if total_pixels > 0 else 0.0

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

            # 2D Morphological Descriptors on primary lesion
            shape_desc = _compute_shape_descriptors(mask_uint8, largest_comp_idx, labels, stats)
        else:
            num_connected_components = 0
            largest_area = 0
            lc_x, lc_y, lc_w, lc_h = 0, 0, 0, 0
            centroid_x, centroid_y = 0.0, 0.0
            shape_desc = _compute_shape_descriptors(None, 0, None, None)

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
        shape_desc = _compute_shape_descriptors(None, 0, None, None)

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
        "note": "Physical millimeter measurements require NIfTI/DICOM voxel spacing metadata.",
        # Phase 3A: Quantitative 2D Morphological Descriptors
        "perimeter_pixels": shape_desc["perimeter_pixels"],
        "circularity": shape_desc["circularity"],
        "solidity": shape_desc["solidity"],
        "major_axis_length": shape_desc["major_axis_length"],
        "minor_axis_length": shape_desc["minor_axis_length"],
        "eccentricity": shape_desc["eccentricity"],
        "feret_diameter_max": shape_desc["feret_diameter_max"],
        "shape_descriptors": shape_desc
    }

    return measurements, segmentation_confidence
