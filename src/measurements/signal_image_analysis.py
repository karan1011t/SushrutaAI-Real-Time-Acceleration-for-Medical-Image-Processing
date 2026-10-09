"""
Signals & Systems and Digital Image Processing (SS/DIP) Analysis Module for SUSHRUTA.

This module provides reusable, mathematically rigorous functions for analyzing
Brain MRI slices as 2D discrete spatial signals I[m,n], evaluating spatial frequencies
via 2D FFT, conducting spatial domain filtering, edge detection, morphological refinement,
and connected component analysis.

Academic & Baseline Isolation Note:
This analysis layer functions purely as a diagnostic and ablation module on validation slices.
It does NOT modify any existing model weights, checkpoints, dataset splits, baseline metrics,
or official held-out test evaluation pipelines.
"""

import os
import time
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import scipy.ndimage as ndimage
import cv2
import torch
import matplotlib.pyplot as plt
import pandas as pd


def analyze_mri_as_2d_signal(
    image_np: np.ndarray,
    mask_np: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Analyzes an MRI image slice as a 2D discrete spatial signal I[m,n].

    Calculates intensity statistics: mean, std, min, max, dynamic range,
    and estimated Signal-to-Noise Ratio (SNR) in linear and dB scales.
    """
    img = image_np.astype(np.float32)
    if img.max() > 1.0:
        img = img / 255.0

    height, width = img.shape
    total_pixels = height * width

    mean_val = float(np.mean(img))
    std_val = float(np.std(img))
    min_val = float(np.min(img))
    max_val = float(np.max(img))
    dynamic_range = float(max_val - min_val)

    # Global SNR: mu / sigma
    snr_global = mean_val / (std_val + 1e-10)
    snr_global_db = 20.0 * np.log10(snr_global + 1e-10)

    # Tissue/Background SNR estimation
    # Background region: intensity <= 0.05 or outside head region
    bg_mask = img <= 0.05
    fg_mask = img > 0.05

    bg_std = float(np.std(img[bg_mask])) if np.sum(bg_mask) > 10 else std_val
    fg_mean = float(np.mean(img[fg_mask])) if np.sum(fg_mask) > 10 else mean_val

    snr_tissue = fg_mean / (bg_std + 1e-10)
    snr_tissue_db = 20.0 * np.log10(snr_tissue + 1e-10)

    # If ground truth tumor mask is provided, calculate tumor-specific SNR
    snr_tumor = None
    snr_tumor_db = None
    if mask_np is not None:
        tumor_pixels = mask_np > 0.5
        if np.sum(tumor_pixels) > 0:
            tumor_mean = float(np.mean(img[tumor_pixels]))
            snr_tumor = tumor_mean / (bg_std + 1e-10)
            snr_tumor_db = 20.0 * np.log10(snr_tumor + 1e-10)

    # Compute intensity histogram (256 bins)
    hist, bin_edges = np.histogram(img, bins=256, range=(0.0, 1.0))

    return {
        "spatial_dimensions": (height, width),
        "total_pixels": total_pixels,
        "mean": mean_val,
        "std": std_val,
        "min": min_val,
        "max": max_val,
        "dynamic_range": dynamic_range,
        "snr_global_linear": snr_global,
        "snr_global_db": float(snr_global_db),
        "snr_tissue_linear": snr_tissue,
        "snr_tissue_db": float(snr_tissue_db),
        "snr_tumor_linear": snr_tumor,
        "snr_tumor_db": snr_tumor_db,
        "histogram": hist.tolist(),
        "bin_edges": bin_edges.tolist()
    }


def compute_2d_fft(image_np: np.ndarray) -> Dict[str, Any]:
    """
    Computes 2D Discrete Fourier Transform using numpy.fft.fft2 and fftshift.

    Returns centered magnitude spectrum, log-magnitude spectrum, and phase spectrum.
    """
    img = image_np.astype(np.float32)
    if img.max() > 1.0:
        img = img / 255.0

    # 2D FFT & Shift
    f_transform = np.fft.fft2(img)
    f_shift = np.fft.fftshift(f_transform)

    # Magnitude Spectrum |F[u,v]|
    magnitude_spectrum = np.abs(f_shift)

    # Log-Magnitude Spectrum log(1 + |F[u,v]|)
    log_magnitude_spectrum = np.log1p(magnitude_spectrum)

    # Phase Spectrum angle(F[u,v])
    phase_spectrum = np.angle(f_shift)

    # High frequency ratio (energy outside central 25% box)
    h, w = img.shape
    cy, cx = h // 2, w // 2
    box_h, box_w = h // 4, w // 4
    low_freq_energy = np.sum(magnitude_spectrum[cy-box_h:cy+box_h, cx-box_w:cx+box_w]**2)
    total_energy = np.sum(magnitude_spectrum**2)
    low_freq_ratio = float(low_freq_energy / (total_energy + 1e-10))
    high_freq_ratio = 1.0 - low_freq_ratio

    return {
        "f_shift": f_shift,
        "magnitude_spectrum": magnitude_spectrum,
        "log_magnitude_spectrum": log_magnitude_spectrum,
        "phase_spectrum": phase_spectrum,
        "total_energy": float(total_energy),
        "low_freq_energy_ratio": low_freq_ratio,
        "high_freq_energy_ratio": high_freq_ratio
    }


def perform_spatial_filtering(image_np: np.ndarray) -> Dict[str, Any]:
    """
    Applies Gaussian and Median filtering to demonstrate noise reduction vs edge blur trade-off.
    """
    img = image_np.astype(np.float32)
    if img.max() > 1.0:
        img = img / 255.0

    img_uint8 = (img * 255.0).astype(np.uint8)

    # Original Sobel gradient magnitude for edge reference
    sobel_x_orig = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y_orig = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    orig_edge_mag = np.mean(np.sqrt(sobel_x_orig**2 + sobel_y_orig**2))

    # Gaussian Filtering across kernel sizes
    gauss_3x3 = cv2.GaussianBlur(img, (3, 3), sigmaX=1.0)
    gauss_5x5 = cv2.GaussianBlur(img, (5, 5), sigmaX=1.5)
    gauss_7x7 = cv2.GaussianBlur(img, (7, 7), sigmaX=2.0)

    # Median Filtering across kernel sizes
    median_3x3 = cv2.medianBlur(img_uint8, 3).astype(np.float32) / 255.0
    median_5x5 = cv2.medianBlur(img_uint8, 5).astype(np.float32) / 255.0

    filters = {
        "Gaussian 3x3 (sigma=1.0)": gauss_3x3,
        "Gaussian 5x5 (sigma=1.5)": gauss_5x5,
        "Gaussian 7x7 (sigma=2.0)": gauss_7x7,
        "Median 3x3": median_3x3,
        "Median 5x5": median_5x5,
    }

    tradeoff_metrics = {}
    for name, filtered in filters.items():
        # Residual noise / difference map
        diff = np.abs(img - filtered)
        residual_variance = float(np.var(diff))

        # Edge preservation ratio
        sx = cv2.Sobel(filtered, cv2.CV_64F, 1, 0, ksize=3)
        sy = cv2.Sobel(filtered, cv2.CV_64F, 0, 1, ksize=3)
        filt_edge_mag = np.mean(np.sqrt(sx**2 + sy**2))
        edge_preservation_ratio = float(filt_edge_mag / (orig_edge_mag + 1e-10))

        tradeoff_metrics[name] = {
            "residual_variance": residual_variance,
            "mean_residual": float(np.mean(diff)),
            "edge_preservation_ratio": edge_preservation_ratio
        }

    return {
        "original": img,
        "filters": filters,
        "tradeoff_metrics": tradeoff_metrics
    }


def perform_edge_detection(image_np: np.ndarray) -> Dict[str, Any]:
    """
    Performs spatial-domain edge detection using Sobel operators and Canny edge detection.
    """
    img = image_np.astype(np.float32)
    if img.max() > 1.0:
        img = img / 255.0

    img_uint8 = (img * 255.0).astype(np.uint8)

    # Sobel Gradient Operators
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    sobel_orientation = np.arctan2(sobel_y, sobel_x)

    # Canny Edge Detector with different threshold pairs
    canny_low = cv2.Canny(img_uint8, 30, 100).astype(np.float32) / 255.0
    canny_mid = cv2.Canny(img_uint8, 50, 150).astype(np.float32) / 255.0
    canny_high = cv2.Canny(img_uint8, 100, 200).astype(np.float32) / 255.0

    return {
        "sobel_x": sobel_x,
        "sobel_y": sobel_y,
        "sobel_magnitude": sobel_magnitude,
        "sobel_orientation": sobel_orientation,
        "canny_low": canny_low,
        "canny_mid": canny_mid,
        "canny_high": canny_high
    }


def compute_dice_iou_single(
    pred: np.ndarray,
    target: np.ndarray
) -> Tuple[float, float]:
    """Computes Dice Coefficient and IoU between binary numpy arrays."""
    p = (pred > 0.5).astype(np.float32).ravel()
    t = (target > 0.5).astype(np.float32).ravel()

    intersection = np.sum(p * t)
    total_p = np.sum(p)
    total_t = np.sum(t)

    if total_p == 0 and total_t == 0:
        return 1.0, 1.0
    if total_p == 0 or total_t == 0:
        return 0.0, 0.0

    dice = (2.0 * intersection) / (total_p + total_t)
    iou = intersection / (total_p + total_t - intersection)
    return float(dice), float(iou)


def apply_morphological_refinement(
    prob_map: np.ndarray,
    target_mask: np.ndarray,
    threshold: float = 0.5,
    kernel_size: int = 3
) -> Dict[str, Any]:
    """
    Demonstrates probability map thresholding and morphological operations
    (Erosion, Dilation, Opening, Closing) with Dice and IoU evaluation.
    """
    prob = prob_map.astype(np.float32)
    gt = (target_mask > 0.5).astype(np.float32)

    # Raw binary thresholding
    raw_pred = (prob >= threshold).astype(np.uint8)

    # Structuring element
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    # Morphological Operations
    erosion = cv2.erode(raw_pred, kernel, iterations=1)
    dilation = cv2.dilate(raw_pred, kernel, iterations=1)
    opening = cv2.morphologyEx(raw_pred, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(raw_pred, cv2.MORPH_CLOSE, kernel)

    # Combined Refinement: Opening (removes background noise) -> Closing (fills internal voids)
    refined = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

    # Evaluate Dice & IoU for each stage
    stages = {
        "Raw Prediction (threshold=0.5)": raw_pred,
        "Erosion": erosion,
        "Dilation": dilation,
        "Opening": opening,
        "Closing": closing,
        "Morphology-Refined (Open+Close)": refined
    }

    metrics = {}
    for name, mask in stages.items():
        dice, iou = compute_dice_iou_single(mask, gt)
        metrics[name] = {"dice": dice, "iou": iou}

    return {
        "raw_prob": prob,
        "ground_truth": gt,
        "stages": stages,
        "metrics": metrics
    }


def analyze_connected_components(
    binary_mask: np.ndarray,
    min_area_pixels: int = 20,
    pixel_spacing_mm: Tuple[float, float] = (1.0, 1.0)
) -> Dict[str, Any]:
    """
    Performs 8-connected component analysis on binary prediction masks.
    Reports component count, individual areas, centroids, bounding boxes,
    and identifies largest tumor component vs spurious noise components.
    """
    mask = (binary_mask > 0.5).astype(np.uint8)

    # 8-connectivity labeling
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    # Component 0 is background
    components_info = []
    total_area = 0
    largest_area = 0
    largest_id = -1

    for i in range(1, num_labels):
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        area = int(stats[i, cv2.CC_STAT_AREA])

        cx, cy = float(centroids[i][0]), float(centroids[i][1])
        area_mm2 = float(area * pixel_spacing_mm[0] * pixel_spacing_mm[1])
        is_spurious = (area < min_area_pixels)

        total_area += area
        if area > largest_area:
            largest_area = area
            largest_id = i

        components_info.append({
            "component_id": i,
            "area_pixels": area,
            "area_mm2": area_mm2,
            "centroid_x": round(cx, 2),
            "centroid_y": round(cy, 2),
            "bbox": (x, y, w, h),
            "is_spurious_noise": is_spurious
        })

    # Cleaned mask keeping components >= min_area_pixels
    cleaned_mask = np.zeros_like(mask)
    for c in components_info:
        if not c["is_spurious_noise"]:
            cleaned_mask[labels == c["component_id"]] = 1

    return {
        "component_count": num_labels - 1,
        "labeled_matrix": labels,
        "components_info": components_info,
        "total_predicted_area_pixels": total_area,
        "largest_component_area_pixels": largest_area,
        "largest_component_id": largest_id,
        "cleaned_mask": cleaned_mask
    }


def benchmark_ss_latency(
    image_np: np.ndarray,
    model: Optional[torch.nn.Module] = None,
    device: str = "cpu",
    num_runs: int = 30
) -> Dict[str, Any]:
    """
    Benchmarks computational performance and processing latency across
    spatial and frequency domain processing stages.
    """
    img = image_np.astype(np.float32)
    if img.max() > 1.0:
        img = img / 255.0

    timings = {}

    # Stage 1: 2D FFT
    t0 = time.perf_counter()
    for _ in range(num_runs):
        _ = compute_2d_fft(img)
    t_fft = (time.perf_counter() - t0) / num_runs * 1000.0
    timings["2D FFT Analysis"] = t_fft

    # Stage 2: Gaussian & Median Filtering
    t0 = time.perf_counter()
    for _ in range(num_runs):
        _ = perform_spatial_filtering(img)
    t_filt = (time.perf_counter() - t0) / num_runs * 1000.0
    timings["Spatial Filtering (Gauss+Median)"] = t_filt

    # Stage 3: Sobel & Canny Edge Detection
    t0 = time.perf_counter()
    for _ in range(num_runs):
        _ = perform_edge_detection(img)
    t_edge = (time.perf_counter() - t0) / num_runs * 1000.0
    timings["Edge Detection (Sobel+Canny)"] = t_edge

    # Stage 4: U-Net Model Neural Inference (if model provided)
    if model is not None:
        model.eval()
        tensor_in = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            # Warmup
            for _ in range(5):
                _ = model(tensor_in)
            t0 = time.perf_counter()
            for _ in range(num_runs):
                _ = model(tensor_in)
            t_unet = (time.perf_counter() - t0) / num_runs * 1000.0
        timings["U-Net Model Inference"] = t_unet

    # Stage 5: Morphological Refinement
    prob_dummy = np.random.rand(256, 256).astype(np.float32)
    gt_dummy = (prob_dummy > 0.5).astype(np.float32)
    t0 = time.perf_counter()
    for _ in range(num_runs):
        _ = apply_morphological_refinement(prob_dummy, gt_dummy)
    t_morph = (time.perf_counter() - t0) / num_runs * 1000.0
    timings["Morphological Refinement"] = t_morph

    # Stage 6: Connected Component Analysis
    mask_dummy = (prob_dummy > 0.5).astype(np.uint8)
    t0 = time.perf_counter()
    for _ in range(num_runs):
        _ = analyze_connected_components(mask_dummy)
    t_cc = (time.perf_counter() - t0) / num_runs * 1000.0
    timings["Connected Component Analysis"] = t_cc

    total_latency_ms = sum(timings.values())
    fps = 1000.0 / (total_latency_ms + 1e-10)

    return {
        "per_stage_latency_ms": timings,
        "total_latency_ms": total_latency_ms,
        "throughput_fps": fps
    }


def generate_master_ss_concept_table() -> pd.DataFrame:
    """
    Generates the master summary table mapping experiments to Signals & Systems
    and Digital Image Processing concepts demonstrated in SUSHRUTA.
    """
    data = [
        {
            "Experiment": "Exp 1: Signal & Intensity Analysis",
            "SS / Image Processing Concept": "MRI as 2D Discrete Spatial Signal I[m,n]",
            "Mathematical Implementation": "Spatial sampling (256x256), intensity stats, Dynamic Range, SNR = mu / sigma",
            "Key Insight & Demonstration": "Quantifies image signal quality, background noise level, and dynamic intensity range."
        },
        {
            "Experiment": "Exp 2: 2D Spatial Frequency Analysis",
            "SS / Image Processing Concept": "2D Discrete Fourier Transform (DFT) & Spectral Shift",
            "Mathematical Implementation": "numpy.fft.fft2, fftshift, Magnitude |F[u,v]| and Log-Magnitude log(1+|F|)",
            "Key Insight & Demonstration": "Center represents low spatial frequencies (gross structure), outer regions represent high frequencies (edges/noise)."
        },
        {
            "Experiment": "Exp 3: Spatial Filtering Experiments",
            "SS / Image Processing Concept": "Spatial Domain Linear vs Non-linear Filtering",
            "Mathematical Implementation": "2D Gaussian smoothing convolution vs. 2D Median order-statistic filter",
            "Key Insight & Demonstration": "Demonstrates noise suppression trade-off: Gaussian blurs boundaries, while Median preserves step edges."
        },
        {
            "Experiment": "Exp 4: Edge & Feature Extraction",
            "SS / Image Processing Concept": "Spatial Gradient & Feature Analysis",
            "Mathematical Implementation": "Sobel derivative operators (Gx, Gy, |G|) & Canny multi-stage hysteresis edge detection",
            "Key Insight & Demonstration": "Extracts anatomical and tumor spatial boundaries for downstream structural features."
        },
        {
            "Experiment": "Exp 5: Thresholding & Morphology",
            "SS / Image Processing Concept": "Binary Image Processing & Mathematical Morphology",
            "Mathematical Implementation": "Probability map thresholding tau=0.5, Erosion, Dilation, Opening, Closing",
            "Key Insight & Demonstration": "Opening removes isolated noise spicules, Closing fills internal tumor voids; improves Dice & IoU."
        },
        {
            "Experiment": "Exp 6: Connected Component Analysis",
            "SS / Image Processing Concept": "Spatial Topography & Region Labeling",
            "Mathematical Implementation": "8-connectivity component labeling, area, centroid (x_bar, y_bar), bounding box",
            "Key Insight & Demonstration": "Identifies multifocal tumor foci and filters out spurious background false positives (<20px)."
        },
        {
            "Experiment": "Exp 7: Latency & Throughput Benchmark",
            "SS / Image Processing Concept": "Computational Complexity & Processing Performance",
            "Mathematical Implementation": "Millisecond latency profiling and FPS calculation across spatial/frequency operations",
            "Key Insight & Demonstration": "Establishes real-time feasibility for post-processing and diagnostic analysis stages."
        }
    ]

    return pd.DataFrame(data)
