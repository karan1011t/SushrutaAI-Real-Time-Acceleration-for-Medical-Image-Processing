# SUSHRUTA — Real-Time Acceleration for Medical Image Processing

AI-assisted, explainable medical image segmentation designed for affordable clinical decision support.

---

## Signals & Systems & Digital Image Processing (SS/DIP) Module

> [!IMPORTANT]
> **Academic & Baseline Isolation Notice**:
> The Signals & Systems / Digital Image Processing analysis module (located in [`notebooks/05_signal_image_analysis.ipynb`](file:///c:/Users/HP/OneDrive/Desktop/Projects/Sushruta/notebooks/05_signal_image_analysis.ipynb) and [`src/measurements/signal_image_analysis.py`](file:///c:/Users/HP/OneDrive/Desktop/Projects/Sushruta/src/measurements/signal_image_analysis.py)) is an independent diagnostic and ablation analysis layer.
> It operates exclusively on validation set slices and **does NOT alter, retrain, or modify** any trained baseline model checkpoints (`results/models/baseline_01_flair_unet_best.pt`), dataset splits (`split_manifest.json`), or official held-out test evaluation benchmarks.

### Demonstrated SS / DIP Concepts:
1. **MRI as 2D Discrete Spatial Signal $I[m,n]$**: Spatial sampling ($256 \times 256$), intensity histograms, statistics ($\mu, \sigma, I_{\min}, I_{\max}$, Dynamic Range), and linear/dB estimated Signal-to-Noise Ratio (SNR).
2. **2D Spatial Frequency Analysis (2D FFT)**: `numpy.fft.fft2` & `fftshift` magnitude spectrum $|F[u,v]|$ and log-magnitude spectrum $\log(1+|F|)$, analyzing DC low frequencies vs. high spatial frequency edges.
3. **Controlled Spatial Filtering**: Comparative analysis of linear 2D Gaussian filters vs non-linear 2D Median filters, analyzing the trade-off between noise suppression and edge preservation.
4. **Spatial Feature & Edge Detection**: Sobel directional gradient operators ($G_x, G_y, |G|$) and multi-stage Canny edge detection.
5. **Thresholding & Mathematical Morphology**: Continuous probability map thresholding and morphological operations (Erosion, Dilation, Opening, Closing) with local validation Dice/IoU spatial metrics.
6. **Connected Component Topography**: 8-connectivity component labeling (`cv2.connectedComponentsWithStats`), morphometrics (area in $\text{mm}^2$, centroids $(\bar{x},\bar{y})$, bounding boxes), and spurious noise artifact removal ($A < 20 \text{ px}$).
7. **Latency & Computational Performance Benchmark**: End-to-end millisecond execution latency and FPS throughput profiling across spatial and frequency domain operations.

---

## Project Structure

```
SUSHRUTA/
├── app/                         # [PHASE 1] FastAPI Backend Application
│   ├── __init__.py
│   ├── main.py                  # FastAPI Application & Endpoints
│   ├── schemas.py               # Pydantic Schemas for Data Validation
│   ├── inference.py             # U-Net MedicalImageSegmenter Engine
│   ├── preprocessing.py         # Image Normalization & Resizing
│   ├── measurements.py          # Tumor Morphometrics & Spatial Metrics
│   └── utils.py                # Visual Artifact Rendering Utilities
├── data/
│   ├── raw/
│   └── processed/
│       └── brats/                # 2D FLAIR MRI Slices (Train, Val, Test)
├── notebooks/
│   ├── 01_brats_exploration.ipynb
│   ├── 02_brats_preprocessing.ipynb
│   ├── 03_brats_unet_baseline.ipynb
│   ├── 04_gpu_verification.ipynb
│   └── 05_signal_image_analysis.ipynb   # SS / DIP Analysis Module Notebook
├── results/
│   ├── app_outputs/             # Dynamic API Visual Artifacts
│   ├── demo_results/            # Demo Prediction Output Artifacts
│   ├── figures/                 # Evaluation & SS Figures
│   ├── metrics/                 # Test Benchmarks & Metrics
│   └── models/                  # Trained U-Net Checkpoints
├── scripts/
│   └── run_demo_predict.py      # Headless Demo Prediction Script
├── src/
│   ├── evaluation/              # Baseline Evaluation Scripts
│   ├── measurements/            # SS / DIP Core Functions
│   ├── models/                  # U-Net Architecture & DataLoaders
│   └── preprocessing/           # BraTS Preprocessing Scripts
├── tests/
│   └── test_backend_api.py      # Automated Backend API Test Suite
└── README.md
```

---

## Setup & Environment

> [!NOTE]
> **Academic Prototype & Hardware Selection Notice**:
> SUSHRUTA is an academic research prototype. Select the PyTorch installation command matching your hardware architecture (CUDA GPU acceleration or CPU fallback).

### 1. Prerequisites & Virtual Environment
Ensure Python 3.13+ and Node.js (v18+) are installed on your system.

```bash
# Clone the repository and navigate into the root directory
git clone https://github.com/karan1011t/SushrutaAI-Real-Time-Acceleration-for-Medical-Image-Processing.git
cd Sushruta

# Create and activate a Python 3.13 virtual environment
python -m venv .venv

# On Windows (PowerShell / Command Prompt):
.venv\Scripts\activate

# On Linux / macOS:
source .venv/bin/activate
```

### 2. Installing Python Dependencies & PyTorch

First, install standard runtime dependencies:
```bash
pip install -r requirements.txt
```

#### Hardware-Specific PyTorch Installation Guidance
Select the PyTorch build appropriate for your machine:

- **For CUDA 12.8 GPU Acceleration (Recommended if NVIDIA GPU is present)**:
  ```bash
  pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
  ```
- **For CPU-Only Fallback Execution**:
  ```bash
  pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu
  ```

### 3. Launching the Backend REST API (FastAPI)

Start the U-Net inference backend server:
```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```
- Interactive API documentation will be available at `http://127.0.0.1:8000/docs`.

### 4. Launching the Frontend Web Application (React + Vite)

In a separate terminal window, launch the interactive React dashboard:
```bash
# Navigate to the frontend directory
cd frontend

# Install Node dependencies
npm ci

# Start the Vite development server
npm run dev
```
- Open your browser to `http://localhost:5173` (or the local URL printed by Vite).

---

## Phase 1 — Production U-Net Inference Backend & REST API

The **SUSHRUTA Phase 1 Backend** wraps our trained 2D FLAIR U-Net model (`results/models/baseline_01_clean_v2_best.pt`, 4.32M parameters) into a production-style FastAPI web service.

### Key Capabilities
1. **Single Checkpoint Initialization**: Loads model weights once at application startup onto CUDA GPU or CPU fallback.
2. **Reproducible Preprocessing**: Applies non-zero FLAIR intensity z-score normalization and bilinear resizing to $256 \times 256$, matching training data representations.
3. **Optimized GPU Inference**: Uses `torch.inference_mode()` with CUDA timing events for sub-25ms inference latency.
4. **Morphometric Measurements**: Computes tumor pixel area, slice area percentage, bounding boxes $(x, y, w, h)$, and pixel diameter.
5. **Visual Diagnostic Artifacts**: Automatically renders 4 PNG artifacts:
   - Preprocessed Grayscale MRI Slice
   - Binary Segmentation Mask
   - MRI + Semi-Transparent Green Overlay with Yellow Bounding Box
   - Structural JET Probability Heatmap
6. **Robust Error Handling**: Clean JSON error responses with proper HTTP status codes ($400, 422, 500$).

---

### Supported Input Formats
- **Supported File Types**: PNG, JPG, JPEG, BMP, TIFF, NIfTI (`.nii`, `.nii.gz` 2D slices).
- **Target Modality**: 2D Brain MRI FLAIR Slices.

---

### Preprocessing Pipeline
1. **Format Decoding**: Decodes uploaded file bytes into single-channel 2D float32 grayscale array.
2. **Intensity Normalization**: Calculates mean $\mu_{\text{brain}}$ and standard deviation $\sigma_{\text{brain}}$ across non-zero brain voxels/pixels ($I > 0$). Computes z-score $(I - \mu) / \sigma$ and min-max scales non-zero region to $[0.0, 1.0]$.
3. **Spatial Resizing**: Resizes spatial dimensions to $256 \times 256$ using bilinear interpolation (`cv2.INTER_LINEAR`).
4. **Tensor Formatting**: Formats array into PyTorch `FloatTensor` of shape $(1, 1, 256, 256)$.

---

### Starting the FastAPI Server
To start the live production backend server:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
Access interactive OpenAPI documentation at `http://localhost:8000/docs`.

---

### API Endpoints Overview

#### `GET /health`
Returns backend operational status, active compute device, loaded model path, and parameter count.
```json
{
  "status": "ok",
  "model_loaded": true,
  "device": "cuda",
  "model_path": "results/models/baseline_01_clean_v2_best.pt",
  "checkpoint_epoch": 4,
  "trainable_parameters": 4317825
}
```

#### `POST /predict`
Accepts an uploaded 2D MRI slice, executes preprocessing, U-Net inference, morphometric calculations, artifact rendering, and returns structured JSON.

##### Example `curl` Request:
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@data/processed/brats/test/images/BraTS2021_00001_slice_075.png"
```

##### Example Response JSON:
```json
{
  "success": true,
  "tumor_detected": true,
  "measurements": {
    "area_pixels": 2110,
    "area_percentage": 3.22,
    "bounding_box": {
      "x": 109,
      "y": 114,
      "width": 53,
      "height": 53
    },
    "width_pixels": 53,
    "height_pixels": 53,
    "diameter_pixels": 53
  },
  "segmentation_confidence": 0.9975,
  "inference": {
    "time_ms": 24.15,
    "preprocessing_time_ms": 38.22,
    "total_request_time_ms": 128.8,
    "device": "cuda",
    "threshold_used": 0.5
  },
  "image_dimensions": {
    "original_shape": [256, 256],
    "processed_shape": [256, 256]
  },
  "outputs": {
    "preprocessed_mri": "http://localhost:8000/static/results/04a1dafca8_mri.png",
    "tumor_mask": "http://localhost:8000/static/results/04a1dafca8_mask.png",
    "overlay": "http://localhost:8000/static/results/04a1dafca8_overlay.png",
    "probability_map": "http://localhost:8000/static/results/04a1dafca8_heatmap.png"
  },
  "disclaimer": "SUSHRUTA is an academic clinical decision-support prototype. Output is NOT a medical diagnosis."
}
```

---

### Technical Definitions & Metrics
- **`area_pixels` & `area_percentage`**: Count of predicted tumor pixels ($P \ge 0.5$) out of total slice area ($256 \times 256 = 65,536\text{ px}$).
- **`diameter_pixels`**: Maximum spatial bounding box dimension ($\max(\text{width}, \text{height})$) of predicted tumor region in 2D pixels.
- **`num_connected_components`**: Count of isolated tumor islands computed via 8-connectivity component labeling (`cv2.connectedComponentsWithStats`).
- **`largest_component_area_pixels`**: Area in pixels of the largest single tumor component.
- **`centroid`**: Tumor spatial center of mass coordinates $(x_{\text{center}}, y_{\text{center}})$ in pixel space.
- **`segmentation_confidence`**: Mean sigmoid probability evaluated strictly over predicted tumor pixels ($\frac{1}{N_{\text{tumor}}} \sum_{i \in \text{tumor}} P_i$).
- **`segmentation_inference_time_ms`**: Pure GPU/CPU U-Net forward pass latency (measured with CUDA events).
- **`layercam_time_ms`**: Execution overhead of LayerCAM backward pass and spatial attribution map generation.
- **`preprocessing_time_ms`**: Time spent reading bytes, applying non-zero z-score normalization, and resizing.
- **`total_request_time_ms`**: End-to-end HTTP request processing latency including artifact rendering.

---

## Phase 2 — Explainable AI (LayerCAM) & Enhanced Analytics

In **Phase 2**, we integrate **LayerCAM** (Jiang et al., *IEEE TPAMI* 2021) into the SUSHRUTA pipeline to provide fine-grained, spatial attribution maps explaining the U-Net's segmentation decisions.

### Conceptual Breakdown

| Module | Core Question Answered | Scientific Meaning |
| :--- | :--- | :--- |
| **U-Net Forward Pass** | *"Where does the model predict tumor?"* | Computes continuous probability map $P \in [0,1]^{256 \times 256}$. |
| **Segmentation Mask** | *"Which pixels are classified as tumor?"* | Thresholded binary decision mask $M \in \{0,1\}^{256 \times 256}$ ($P \ge 0.5$). |
| **LayerCAM (XAI)** | *"Which image regions contributed most strongly to that prediction?"* | Backpropagates target score $Y$ to feature maps $A$ to compute element-wise spatial attribution weights $w_{i,j}^k$. |

---

### LayerCAM Mathematical Implementation
Unlike traditional Grad-CAM which averages gradients across spatial dimensions to compute a single scalar weight per channel, **LayerCAM** preserves spatial fidelity by computing element-wise spatial weights:

1. **Target Convolutional Layer**: Selected as `model.up4.conv.double_conv[3]`—the final $2D$ convolutional layer in the decoder prior to the $1 \times 1$ output head. Its feature activation map is $A \in \mathbb{R}^{1 \times 32 \times 256 \times 256}$.
2. **Segmentation Target Score $Y$**:
   - If tumor pixels are predicted ($\sum M > 0$): $Y = \sum_{(x,y): M(x,y)=1} S_{(x,y)}$ (sum of raw un-normalized U-Net logit activations over predicted tumor pixels).
   - If no tumor is predicted ($\sum M == 0$): $Y = \max_{x,y} S_{(x,y)}$ (maximum logit activation across slice).
3. **Element-Wise Spatial Weights**:
   $$w_{i,j}^k = \max\left(\frac{\partial Y}{\partial A_{i,j}^k}, 0\right) = \text{ReLU}\left(\frac{\partial Y}{\partial A_{i,j}^k}\right)$$
4. **Spatial Activation Combination**:
   $$L_{i,j} = \text{ReLU}\left(\sum_{k=1}^K w_{i,j}^k \cdot A_{i,j}^k\right)$$
5. **Normalization**: Min-max normalized to $[0.0, 1.0]$ and rendered as a color-mapped attribution heatmap.

---

### Phase 2 Visual Artifacts
Every prediction request generates 7 visual artifacts:
1. `preprocessed_mri.png`: Normalized input MRI.
2. `tumor_mask.png`: Binary segmentation mask.
3. `overlay.png`: MRI + semi-transparent green tumor mask + yellow bounding box.
4. `probability_map.png`: JET probability heatmap.
5. `layercam.png`: Standalone LayerCAM attribution heatmap.
6. `layercam_overlay.png`: 60% LayerCAM heatmap + 40% MRI structural base.
7. `combined_overlay.png`: LayerCAM overlay + bright cyan binary tumor mask boundary outline.

---

### What LayerCAM Does NOT Prove
- **Attribution vs. Proof**: LayerCAM illustrates which image features the model relied upon when forming its internal output score. It **does NOT prove** that the prediction is clinically correct or ground-truth accurate.
- **Causal Disclaimer**: LayerCAM is a visual feature attribution mechanism for model transparency; it is **NOT** a causal or clinical validation tool.

---

### Testing & Verification

Run the complete 32-test automated suite (covering Phase 1, Phase 2 XAI, and Phase 4 SS Analysis):
```bash
python tests/test_backend_api.py
```

Run Phase 2 live demo prediction:
```bash
python scripts/run_demo_predict.py
```

---

## Phase 4 — Signals & Systems / Image Processing Analysis

Phase 4 integrates a dedicated **Signals & Systems / Digital Image Processing (SS/DIP)** analytical module into SUSHRUTA. This academic module exposes classical 2D signal processing representations and spatial/frequency-domain analysis methods alongside the deep learning pipeline.

> [!NOTE]
> **Pipeline Isolation Principle**:
> The SS analysis layer operates as an educational and analytical module via `POST /analyze-signal`. It **does NOT modify, retrain, or replace** the official U-Net inference pipeline (`POST /predict`), model weights, or held-out test evaluation metrics.

---

### Conceptual Processing Flow

```
MRI Slice
   │
   ├──► Discrete Spatial Signal I[m,n]
   │       ├──► Intensity Statistics & Histogram Distribution
   │       ├──► Spatial Domain Filtering (Gaussian 5x5 vs Median 5x5)
   │       ├──► 2D Discrete Fourier Transform (FFT Spectrum log(1+|F|))
   │       ├──► Spatial Edge Detection (Sobel Gradient & Canny Edges)
   │       └──► Signal Quality (Estimated Tissue & Global SNR)
   │
   └──► AI Segmentation Pipeline
           ├──► U-Net Sigmoid Probability Map P(x,y)
           ├──► Binary Decision Thresholding M(x,y) = 1 if P ≥ 0.5
           ├──► Mathematical Morphology (Erosion, Dilation, Opening, Closing)
           └──► Connected-Component Topography & Morphometric Analytics
```

---

### Summary of Implemented SS / DIP Concepts

1. **MRI as a 2D Discrete Spatial Signal $I[m,n]$**:
   - Explicitly represents the MRI slice as a discrete 2D spatial signal array $I[m,n]$ ($256 \times 256$, $65,536$ pixels).
   - Computes minimum intensity, maximum intensity, mean $\mu$, standard deviation $\sigma$, dynamic range, and pixel count.
   - *Explanation*: "Each MRI slice is represented as a two-dimensional array of sampled intensity values I[m,n]."

2. **Intensity Histogram Distribution**:
   - Generates a 256-bin intensity histogram showing pixel intensity distribution across normalized $[0.0, 1.0]$.
   - Visual artifact: `histogram.png`.
   - *Explanation*: "The histogram shows how pixel intensities are distributed across the MRI slice."

3. **Spatial Domain Filtering**:
   - Performs controlled comparative filtering experiments: Original, Gaussian smoothing ($5 \times 5$, $\sigma=1.5$), and Median filtering ($5 \times 5$).
   - Calculates residual variance and edge preservation ratios.
   - Visual artifact: `filter_comparison.png` (3-panel: Original | Gaussian | Median).
   - *Explanation*: "Spatial filters modify a pixel using information from its local neighborhood. Gaussian filtering provides smoothing, while median filtering is useful for reducing impulse-like noise while preserving edges."

4. **2D Discrete Fourier Transform (FFT)**:
   - Computes 2D FFT using NumPy (`np.fft.fft2`, `np.fft.fftshift`).
   - Calculates magnitude spectrum $|F[u,v]|$ and log-magnitude spectrum $\log(1 + |F[u,v]|)$.
   - Reports total spectral energy, central low-frequency energy ratio (central 25%), and high-frequency energy ratio.
   - Visual artifact: `fft_analysis.png` (3-panel: Original | Magnitude Spectrum | Log Spectrum).
   - *Explanation*: "The Fourier transform represents the MRI in the spatial-frequency domain. Low spatial frequencies correspond mainly to slowly varying intensity structures, while higher spatial frequencies correspond to finer details and rapid intensity changes."

5. **Spatial Edge Detection**:
   - Implements Sobel first-order directional gradient magnitude $|G| = \sqrt{G_x^2 + G_y^2}$ and multi-stage Canny hysteresis edge detection ($T_{\text{low}}=0.1, T_{\text{high}}=0.3$).
   - Reports Sobel edge pixel count and Canny edge pixel count.
   - Visual artifact: `edge_detection.png` (3-panel: Original | Sobel | Canny).
   - *Explanation*: "Edges correspond to rapid spatial intensity changes and therefore contain high-frequency structural information."

6. **Probability Thresholding**:
   - Demonstrates binary threshold decision mapping on U-Net probability output $P(x,y)$:
     $$M(x,y) = \begin{cases} 1 & \text{if } P(x,y) \ge 0.5 \\ 0 & \text{otherwise} \end{cases}$$
   - *Explanation*: "Thresholding converts the continuous model probability output into a binary segmentation signal."

7. **Mathematical Morphology**:
   - Demonstrates set-theoretic morphological operations using a $5 \times 5$ elliptical structuring element: Erosion ($\ominus$), Dilation ($\oplus$), Opening ($\circ$), and Closing ($\bullet$).
   - Visual artifact: `morphology_analysis.png` (5-panel progression).
   - *Explanation*: "Morphological operations refine binary spatial shapes using set operations."

8. **Connected Component Analysis**:
   - Evaluates connected component topography using 8-connectivity labeling (`cv2.connectedComponentsWithStats`).
   - Reports component count, largest component area, centroid $(\bar{x}, \bar{y})$, and bounding box $(x,y,w,h)$.
   - *Explanation*: "Connected-component analysis treats connected foreground pixels as separate regions and allows us to quantify segmented structures."

9. **Estimated Signal-to-Noise Ratio (SNR)**:
   - Computes Estimated Tissue SNR ($\text{SNR}_{\text{tissue}} = \frac{\mu_{\text{tissue}}}{\sigma_{\text{background}}}$) and Global SNR in both linear scale and decibels ($\text{dB} = 20 \log_{10}(\text{SNR})$).
   - Documented estimation method: Tissue pixels defined as $I > 0.05$; background noise defined as $I \le 0.05$.

---

### Scientific Safety & Disclaimers

> [!WARNING]
> **Academic Integrity Notice**:
> - 2D FFT is a general spatial-frequency decomposition; it does **NOT** detect brain tumors by itself.
> - Edge detectors (Sobel/Canny) mark rapid intensity gradients across the slice; they do **NOT** uniquely isolate tumor boundaries.
> - Spatial filtering and morphological operations are analytical/educational demonstrations; they do **NOT** replace or override the official U-Net prediction mask.
> - Estimated SNR is a non-invasive computational estimate; it is **NOT** a clinically standardized MRI calibration metric.

---

### Phase 4 API Endpoint: `POST /analyze-signal`

- **Endpoint**: `POST http://localhost:8000/analyze-signal`
- **Accepts**: Multipart form-data with attached MRI image file (`file=@slice.png`).
- **Response**: Structured JSON containing 2D signal statistics, histogram metrics, filtering metrics, frequency energy distribution, edge counts, morphology parameters, connected component statistics, SNR estimates, latency breakdown, and 5 artifact URLs served via `/static/signal_analysis/`.

---

### Limitations & Academic Disclaimers
1. **Modality Constraint**: The trained model expects 2D Brain MRI FLAIR slices. External non-FLAIR or arbitrary medical scans will yield meaningless predictions.
2. **Spatial Pixel Units**: Inputs supplied as standard 2D image formats (PNG/JPEG) lack DICOM/NIfTI physical voxel spacing metadata ($dx, dy, dz$). All spatial measurements are reported strictly in **pixels**.
3. **Generalization Scope**: The model's baseline test performance ($\text{Dice} \approx 99.59\%$) is evaluated exclusively on the held-out BraTS 2021 test split. Performance on external hospital scanners or out-of-distribution datasets is not guaranteed.
4. **Regulatory Disclaimer**: SUSHRUTA is an academic research prototype designed for clinical decision support exploration. It is **NOT** a certified medical diagnostic system and must **NEVER** be described as providing automated medical diagnoses or 100% diagnostic accuracy.
