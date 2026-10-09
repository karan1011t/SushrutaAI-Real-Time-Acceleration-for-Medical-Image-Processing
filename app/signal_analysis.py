"""
SUSHRUTA Signals & Systems (SS) / Digital Image Processing (DIP) Backend Module.

Exposes comprehensive signal & image analysis routines:
1. 2D Discrete Spatial Signal Representation I[m,n] & intensity statistics.
2. 256-bin Intensity Histogram distribution.
3. Spatial Domain Filtering (Gaussian 5x5 vs Median 5x5).
4. 2D Fourier Transform (FFT, Centered Magnitude & Log-Magnitude Spectrum log(1+|F|)).
5. Spatial Edge Detection (Sobel Gradient Magnitude & Canny Hysteresis Edge Detection).
6. Binary Thresholding & Mathematical Morphology (Erosion, Dilation, Opening, Closing).
7. Connected Component Analysis & Topography.
8. Estimated Signal-to-Noise Ratio (SNR) & Latency Profiling.
"""

import os
import sys
import time
import uuid
from typing import Dict, Any, Tuple, Optional
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')  # Headless backend for Matplotlib
import matplotlib.pyplot as plt
from PIL import Image

# Add project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from app.preprocessing import preprocess_image_file
from src.measurements.signal_image_analysis import (
    analyze_mri_as_2d_signal,
    compute_2d_fft,
    perform_spatial_filtering,
    perform_edge_detection,
    apply_morphological_refinement,
    analyze_connected_components
)

SS_OUTPUT_DIR = os.path.join(project_root, "results", "signal_analysis")
os.makedirs(SS_OUTPUT_DIR, exist_ok=True)


def generate_ss_visual_artifacts(
    image_np: np.ndarray,
    signal_stats: Dict[str, Any],
    fft_results: Dict[str, Any],
    filt_results: Dict[str, Any],
    edge_results: Dict[str, Any],
    morph_results: Dict[str, Any],
    output_dir: str = SS_OUTPUT_DIR,
    base_id: Optional[str] = None
) -> Dict[str, str]:
    """
    Renders and saves 5 analytical SS/DIP visual artifact PNG files.
    """
    if not base_id:
        base_id = uuid.uuid4().hex[:10]

    img_uint8 = np.clip(image_np * 255.0, 0, 255).astype(np.uint8)

    # -------------------------------------------------------------------------
    # 1. INTENSITY HISTOGRAM (histogram.png)
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    fig.patch.set_facecolor('#121827')
    ax.set_facecolor('#090d16')

    hist = signal_stats["histogram"]
    bin_edges = signal_stats["bin_edges"]
    bin_centers = 0.5 * (np.array(bin_edges[:-1]) + np.array(bin_edges[1:]))

    ax.plot(bin_centers, hist, color='#00e5ff', linewidth=1.5)
    ax.fill_between(bin_centers, hist, color='#00e5ff', alpha=0.25)
    ax.set_title("Intensity Histogram Distribution", color='#f1f5f9', fontsize=11, fontweight='bold')
    ax.set_xlabel("Intensity Value (Normalized [0, 1])", color='#94a3b8', fontsize=9)
    ax.set_ylabel("Pixel Count", color='#94a3b8', fontsize=9)
    ax.tick_params(colors='#94a3b8', labelsize=8)
    for spine in ax.spines.values():
        spine.set_color('#232f48')
    ax.grid(True, color='#232f48', linestyle='--', alpha=0.5)

    hist_filename = f"{base_id}_histogram.png"
    hist_path = os.path.join(output_dir, hist_filename)
    plt.tight_layout()
    plt.savefig(hist_path, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()

    # -------------------------------------------------------------------------
    # 2. SPATIAL FILTERING COMPARISON (filter_comparison.png)
    # -------------------------------------------------------------------------
    gauss_img = (filt_results["filters"]["Gaussian 5x5 (sigma=1.5)"] * 255.0).astype(np.uint8)
    median_img = (filt_results["filters"]["Median 5x5"] * 255.0).astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=150)
    fig.patch.set_facecolor('#121827')

    titles = ["Original MRI Slice", "Gaussian Filter (5x5, σ=1.5)", "Median Filter (5x5)"]
    imgs = [img_uint8, gauss_img, median_img]

    for ax, im, title in zip(axes, imgs, titles):
        ax.imshow(im, cmap='gray')
        ax.set_title(title, color='#f1f5f9', fontsize=10, fontweight='bold')
        ax.axis('off')

    filt_filename = f"{base_id}_filter_comparison.png"
    filt_path = os.path.join(output_dir, filt_filename)
    plt.tight_layout()
    plt.savefig(filt_path, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()

    # -------------------------------------------------------------------------
    # 3. 2D FFT FREQUENCY SPECTRUM (fft_analysis.png)
    # -------------------------------------------------------------------------
    mag_spec = fft_results["magnitude_spectrum"]
    log_mag_spec = fft_results["log_magnitude_spectrum"]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=150)
    fig.patch.set_facecolor('#121827')

    axes[0].imshow(img_uint8, cmap='gray')
    axes[0].set_title("Original Spatial Domain I[m,n]", color='#f1f5f9', fontsize=10, fontweight='bold')
    axes[0].axis('off')

    im1 = axes[1].imshow(mag_spec, cmap='magma')
    axes[1].set_title("2D FFT Magnitude Spectrum |F[u,v]|", color='#f1f5f9', fontsize=10, fontweight='bold')
    axes[1].axis('off')

    im2 = axes[2].imshow(log_mag_spec, cmap='inferno')
    axes[2].set_title("Log Spectrum log(1 + |F[u,v]|)", color='#f1f5f9', fontsize=10, fontweight='bold')
    axes[2].axis('off')

    fft_filename = f"{base_id}_fft_analysis.png"
    fft_path = os.path.join(output_dir, fft_filename)
    plt.tight_layout()
    plt.savefig(fft_path, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()

    # -------------------------------------------------------------------------
    # 4. SPATIAL EDGE DETECTION (edge_detection.png)
    # -------------------------------------------------------------------------
    sobel_mag = (edge_results["sobel_magnitude"] / (np.max(edge_results["sobel_magnitude"]) + 1e-8) * 255.0).astype(np.uint8)
    canny_img = (edge_results["canny_mid"] * 255.0).astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=150)
    fig.patch.set_facecolor('#121827')

    axes[0].imshow(img_uint8, cmap='gray')
    axes[0].set_title("Original MRI Slice", color='#f1f5f9', fontsize=10, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(sobel_mag, cmap='gray')
    axes[1].set_title("Sobel Gradient Magnitude |G|", color='#f1f5f9', fontsize=10, fontweight='bold')
    axes[1].axis('off')

    axes[2].imshow(canny_img, cmap='gray')
    axes[2].set_title("Canny Hysteresis Edges", color='#f1f5f9', fontsize=10, fontweight='bold')
    axes[2].axis('off')

    edge_filename = f"{base_id}_edge_detection.png"
    edge_path = os.path.join(output_dir, edge_filename)
    plt.tight_layout()
    plt.savefig(edge_path, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()

    # -------------------------------------------------------------------------
    # 5. MATHEMATICAL MORPHOLOGY (morphology_analysis.png)
    # -------------------------------------------------------------------------
    stages = morph_results["stages"]
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.5), dpi=150)
    fig.patch.set_facecolor('#121827')

    morph_titles = ["Raw Mask", "Erosion", "Dilation", "Opening", "Closing"]
    morph_keys = ["Raw Prediction (threshold=0.5)", "Erosion", "Dilation", "Opening", "Closing"]

    for ax, title, key in zip(axes, morph_titles, morph_keys):
        m = (stages[key] * 255).astype(np.uint8)
        ax.imshow(m, cmap='gray')
        ax.set_title(title, color='#f1f5f9', fontsize=10, fontweight='bold')
        ax.axis('off')

    morph_filename = f"{base_id}_morphology_analysis.png"
    morph_path = os.path.join(output_dir, morph_filename)
    plt.tight_layout()
    plt.savefig(morph_path, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()

    return {
        "histogram": hist_filename,
        "filter_comparison": filt_filename,
        "fft_analysis": fft_filename,
        "edge_detection": edge_filename,
        "morphology_analysis": morph_filename,
        "hist_path": hist_path,
        "filt_path": filt_path,
        "fft_path": fft_path,
        "edge_path": edge_path,
        "morph_path": morph_path
    }


def analyze_signal_image(
    file_bytes: bytes,
    filename: str,
    output_dir: str = SS_OUTPUT_DIR
) -> Tuple[Dict[str, Any], Dict[str, str], float]:
    """
    Main entrypoint for Signals & Systems (SS) image analysis pipeline.

    Accepts 2D MRI slice file bytes, applies SUSHRUTA signal analysis,
    computes intensity statistics, 2D FFT, filtering, edge detection,
    binary morphology, connected components, and returns structured results.
    """
    t_start = time.perf_counter()

    # 1. Preprocess input image to (256, 256) float32 normalized representation
    input_tensor, norm_img_np, meta, prep_time_ms = preprocess_image_file(file_bytes, filename)

    # 2. Concept 1 & 2: Discrete Signal I[m,n] & Histogram Statistics
    signal_stats = analyze_mri_as_2d_signal(norm_img_np)

    # 3. Concept 3: Spatial Filtering (Gaussian 5x5 vs Median 5x5)
    t_filt_start = time.perf_counter()
    filt_results = perform_spatial_filtering(norm_img_np)
    t_filt_end = time.perf_counter()
    filt_time_ms = round((t_filt_end - t_filt_start) * 1000.0, 2)

    # 4. Concept 4: 2D Discrete Fourier Transform (FFT)
    t_fft_start = time.perf_counter()
    fft_results = compute_2d_fft(norm_img_np)
    t_fft_end = time.perf_counter()
    fft_time_ms = round((t_fft_end - t_fft_start) * 1000.0, 2)

    # 5. Concept 5: Spatial Edge Detection (Sobel & Canny)
    edge_results = perform_edge_detection(norm_img_np)
    sobel_edge_count = int(np.sum(edge_results["sobel_magnitude"] > 0.3))
    canny_edge_count = int(np.sum(edge_results["canny_mid"] > 0.5))

    # 6. Concept 6 & 7: Thresholding & Mathematical Morphology
    t_morph_start = time.perf_counter()
    dummy_prob = norm_img_np  # Analytical demonstration on image intensity signal
    dummy_gt = (norm_img_np > 0.5).astype(np.float32)
    morph_results = apply_morphological_refinement(dummy_prob, dummy_gt, threshold=0.5, kernel_size=5)
    t_morph_end = time.perf_counter()
    morph_time_ms = round((t_morph_end - t_morph_start) * 1000.0, 2)

    # 7. Concept 8: Connected Component Analysis
    mask_binary = (norm_img_np > 0.5).astype(np.uint8)
    cc_results = analyze_connected_components(mask_binary, min_area_pixels=20)

    # 8. Render Visual Artifact PNG files
    base_id = uuid.uuid4().hex[:10]
    artifact_files = generate_ss_visual_artifacts(
        image_np=norm_img_np,
        signal_stats=signal_stats,
        fft_results=fft_results,
        filt_results=filt_results,
        edge_results=edge_results,
        morph_results=morph_results,
        output_dir=output_dir,
        base_id=base_id
    )

    t_end = time.perf_counter()
    total_ss_time_ms = round((t_end - t_start) * 1000.0, 2)

    # Format structured JSON results
    structured_results = {
        "success": True,
        "signal": {
            "width": meta["processed_shape"][1],
            "height": meta["processed_shape"][0],
            "pixels": signal_stats["total_pixels"],
            "mean": round(signal_stats["mean"], 4),
            "std": round(signal_stats["std"], 4),
            "min": round(signal_stats["min"], 4),
            "max": round(signal_stats["max"], 4),
            "dynamic_range": round(signal_stats["dynamic_range"], 4)
        },
        "histogram": {
            "mean": round(signal_stats["mean"], 4),
            "std": round(signal_stats["std"], 4),
            "bin_count": len(signal_stats["histogram"])
        },
        "filtering": {
            "gaussian_5x5": {
                "kernel": "5x5",
                "sigma": 1.5,
                "residual_variance": round(filt_results["tradeoff_metrics"]["Gaussian 5x5 (sigma=1.5)"]["residual_variance"], 6),
                "edge_preservation_ratio": round(filt_results["tradeoff_metrics"]["Gaussian 5x5 (sigma=1.5)"]["edge_preservation_ratio"], 4)
            },
            "median_5x5": {
                "kernel": "5x5",
                "residual_variance": round(filt_results["tradeoff_metrics"]["Median 5x5"]["residual_variance"], 6),
                "edge_preservation_ratio": round(filt_results["tradeoff_metrics"]["Median 5x5"]["edge_preservation_ratio"], 4)
            }
        },
        "frequency": {
            "total_energy": round(fft_results["total_energy"], 2),
            "low_frequency_energy_ratio": round(fft_results["low_freq_energy_ratio"], 4),
            "high_frequency_energy_ratio": round(fft_results["high_freq_energy_ratio"], 4),
            "note": "Low frequency region defined as central 25% frequency box; high frequency region outside central box."
        },
        "edges": {
            "sobel_edge_pixels": sobel_edge_count,
            "canny_edge_pixels": canny_edge_count
        },
        "morphology": {
            "kernel_size": 5,
            "kernel_shape": "ellipse",
            "stages": ["Raw Prediction", "Erosion", "Dilation", "Opening", "Closing"]
        },
        "connected_components": {
            "component_count": cc_results["component_count"],
            "largest_component_area_pixels": cc_results["largest_component_area_pixels"],
            "total_predicted_area_pixels": cc_results["total_predicted_area_pixels"]
        },
        "snr_estimate": {
            "global_snr_linear": round(signal_stats["snr_global_linear"], 2),
            "global_snr_db": round(signal_stats["snr_global_db"], 2),
            "tissue_snr_linear": round(signal_stats["snr_tissue_linear"], 2),
            "tissue_snr_db": round(signal_stats["snr_tissue_db"], 2),
            "estimation_method": "Tissue SNR = Mean(tissue pixels > 0.05) / StdDev(background pixels <= 0.05)"
        },
        "timings": {
            "fft_time_ms": fft_time_ms,
            "filtering_time_ms": filt_time_ms,
            "morphology_time_ms": morph_time_ms,
            "total_signal_analysis_time_ms": total_ss_time_ms
        }
    }

    return structured_results, artifact_files, total_ss_time_ms
