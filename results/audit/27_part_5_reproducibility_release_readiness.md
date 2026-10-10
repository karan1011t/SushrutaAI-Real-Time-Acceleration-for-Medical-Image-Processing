# SUSHRUTA — PART 5 AUDIT REPORT: REPRODUCIBILITY, FAILURE-PATH INTEGRITY & RELEASE READINESS

**Date:** 2026-10-10
**Repository:** `C:\Users\HP\OneDrive\Desktop\Projects\Sushruta`
**Branch:** `feature/provenance-audit-and-app-pipeline`
**Target Remote:** `https://github.com/karan1011t/SushrutaAI-Real-Time-Acceleration-for-Medical-Image-Processing`
**Auditor Role:** Senior ML Systems Engineer, Backend Reviewer & Test Engineer

---

## 1. Executive Verdict

### **Verdict: PASS WITH CAVEATS**

- **Deterministic Inference:** Verified bitwise identical segmentation masks and invariant bounding boxes across consecutive runs on positive and negative control slices. Model weights remained strictly immutable.
- **Failure-Path Integrity:** Fixed critical silent masking vulnerabilities where non-finite (`NaN` / `Inf`) model outputs were previously sanitized into normal-looking negative predictions or empty masks. Both classifier and segmentation engines now raise explicit errors that surface as structured HTTP 500 error responses (`Classification Failure` and `Inference Failure`).
- **Explainability Isolation:** Verified that LayerCAM failures during gradient backpropagation do not invalidate underlying valid segmentation predictions. The system sets `outputs.layercam = null` with explicit error metadata while preserving segmentation integrity.
- **Measurement Integrity:** Confirmed strict spatial unit reporting (`px`, `%`, normalized descriptors). Undefined shape metrics for empty masks strictly evaluate to `null` rather than fabricated zeroes. No 3D or physical millimeter claims are inferred from 2D PNG/JPEG slices.
- **Caveats:**
  1. *Browser Automated Testing Limitation:* Real visual browser automation was blocked by local Playwright driver download failure (Azure CDN 404). Frontend integrity is verified via static code analysis (oxlint: 0 errors) and Vite production build (0 errors, 1.87s build), but end-to-end visual browser driving remains unperformed.
  2. *Research vs. Clinical Boundary:* All verifications documented herein represent technical software engineering and computational reproducibility audits, NOT clinical validation, diagnostic efficacy, or regulatory clearance (FDA/CE).

---

## 2. Git State & Working-Tree Preservation

| Metric | Recorded Value |
| :--- | :--- |
| **Current Branch** | `feature/provenance-audit-and-app-pipeline` |
| **Tracking Remote** | `origin/feature/provenance-audit-and-app-pipeline` |
| **Starting HEAD** | `c2cc99694d5f4be24d6a021d1f6a9e4e7352304a` (`c2cc996`) |
| **Ending HEAD** | `c2cc99694d5f4be24d6a021d1f6a9e4e7352304a` (`c2cc996`) |
| **Git Operations** | Zero commits, zero pushes, zero stashes, zero resets, zero cleans |
| **Branch Safety** | Branch `main` untouched; feature branch preserved |

### Preservation Statement
All pre-existing working-tree modifications from previous development phases (including `app/main.py`, `app/measurements.py`, `app/schemas.py`, `frontend/src/components/MorphometricsPanel.jsx`, and audit figures/metrics) along with all untracked artifacts (`results/metrics/exp_clinical_*`, `tests/test_clinical_*`, etc.) were strictly preserved. No destructive Git commands were executed.

---

## 3. Production Inference Architecture & Checkpoint Provenance

The Sushruta production application executes a decoupled, two-stage inference pipeline:
1. **Stage 1 (Classification & Saliency Gate):** ResNet18-based `BinaryTumorClassifier` detects global slice-level tumor presence and generates classifier LayerCAM heatmaps.
2. **Stage 2 (Segmentation Engine):** 2D U-Net (`UNet2D`, `base_filters=32`) predicts pixel-wise tumor segmentation masks and morphological descriptors unconditionally across slices.

### Checkpoint File Verification (Current Local Filesystem)

| Checkpoint Identifier | Relative Repo Path | File Size (Bytes) | SHA-256 Checksum | Architecture & Status |
| :--- | :--- | :--- | :--- | :--- |
| **Production U-Net** | `results/models/baseline_01_clean_v2_best.pt` | 51,924,939 | `be6a2785de6b154da77cf532921d4a7fff9277145ca72242a2f55a02d0463974` | `UNet2D` (4,317,825 parameters, PyTorch eval mode, Epoch 4). Validated Production. |
| **Production Classifier** | `results/models/baseline_02_classifier_best.pt` | 4,898,231 | `a60cec5e31e67689542b2d2edaa759d2673644f223b4893bb3bae96ab5f23833` | `BinaryTumorClassifier` (405,121 parameters, ResNet18 backbone, PyTorch eval mode). Validated Production. |
| **Experimental 2.5D Model** | `results/models/exp_clinical_04_25d_best.pt` | 51,926,603 | `4cff51c22501a3511eb6979a0bfa15c138d6df1b34e56570cbe3b3fa5a7bbcfb` | Experimental 2.5D U-Net. **Isolated; NOT used in production pipeline.** |
| **Experimental 2.5D Multi-Epoch**| `results/models/exp_clinical_04b_25d_best.pt` | 51,926,603 | `496dc4043b3558f6bbaf57aee29e710086c8a24c52f9543e5cbb342cb2fc2404` | Experimental 2.5D U-Net. **Isolated; NOT used in production pipeline.** |

*Note:* Checksum verification confirmed that inspection and test executions did not alter the production checkpoint binaries on disk.

---

## 4. Reproducibility & Determinism Verification

Deterministic inference was audited by executing repeated inference passes ($N=3$) under identical conditions across fixed benchmark fixtures:
- **Positive Control Fixture:** `data/processed/brats/val/images/BraTS2021_00002_slice_063.png`
- **Negative Control Fixture:** `data/processed/brats/val/images/BraTS2021_00002_slice_028.png`

### Observed Metrics Across Runs

| Fixture | Metric | Run 1 | Run 2 | Run 3 | Determinism Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Positive Control** | Classifier Probability | 0.994709 | 0.994709 | 0.994709 | Exact Match ($\Delta = 0.0$) |
| **Positive Control** | Classifier Logit | 5.236940 | 5.236940 | 5.236940 | Exact Match ($\Delta = 0.0$) |
| **Positive Control** | Mask Area (px) | 636 | 636 | 636 | Bitwise Identical |
| **Positive Control** | Connected Components | 1 | 1 | 1 | Bitwise Identical |
| **Positive Control** | Bounding Box ($x, y, w, h$) | (122, 127, 29, 29) | (122, 127, 29, 29) | (122, 127, 29, 29) | Bitwise Identical |
| **Positive Control** | Centroid ($x, y$) | (135.96, 141.19) | (135.96, 141.19) | (135.96, 141.19) | Exact Match ($\Delta = 0.0$) |
| **Positive Control** | Area Percentage (%) | 0.970398 | 0.970398 | 0.970398 | Exact Match ($\Delta = 0.0$) |
| **Negative Control** | Classifier Probability | 4.71e-5 | 4.71e-5 | 4.71e-5 | Exact Match ($\Delta = 0.0$) |
| **Negative Control** | Classifier Logit | -9.963012 | -9.963012 | -9.963012 | Exact Match ($\Delta = 0.0$) |
| **Negative Control** | Mask Area (px) | 0 | 0 | 0 | Bitwise Identical |
| **Negative Control** | Connected Components | 0 | 0 | 0 | Bitwise Identical |
| **Negative Control** | Shape Descriptors | All `null` | All `null` | All `null` | Exact Match |

### Numerical Tolerances & Lifecycle Audits
- **Discrete Fields:** Discrete outputs (binary masks, pixel counts, connected components, bounding boxes) are strictly bitwise deterministic ($Tolerance = 0$).
- **Floating Point Tolerances:** Classifier probabilities and logits exhibit floating-point variance $< 10^{-6}$ on CUDA (NVIDIA RTX 2050 Laptop GPU, PyTorch 2.6.0+cu126).
- **Hook Lifecycle & State Leakage:** Inspected PyTorch hook registration handles in `app/classifier_xai.py` and `app/explainability.py`. Both forward and backward hook counts strictly evaluate to zero (`len(_forward_hooks) == 0`, `len(_backward_hooks) == 0`) upon request completion. No tensors remain retained in computation graphs.
- **Weight Immutability:** SHA-256 hashes of checkpoint files before and after multiple inference cycles confirmed zero binary modification on disk.

---

## 5. Failure-Path Integrity Findings & Defect Fixes

### Critical Vulnerabilities Identified & Resolved

1. **Classifier Silent Negative Masking:**
   - *Previous Defect:* In `app/classifier_xai.py`, non-finite logits were caught in generic exception handling and returned `prob=0.0`, resulting in a corrupt model output being reported to the client as HTTP 200 with "No Tumor Detected".
   - *Engineered Fix:* Added explicit non-finite checks:
     ```python
     if np.isnan(logit_val) or np.isinf(logit_val) or np.isnan(prob) or np.isinf(prob):
         raise ValueError("Stage 1 classifier model output is non-finite (NaN or Inf). Inference corrupted.")
     ```
     This triggers FastAPI's HTTP 500 `Classification Failure` error handler, surfacing model failure clearly.

2. **U-Net Silent Empty-Mask Masking:**
   - *Previous Defect:* In `app/inference.py`, `np.nan_to_num(prob_map, nan=0.0, posinf=1.0)` was applied directly to raw segmentation maps. A completely broken or diverged model emitting `NaN`s was sanitized into an all-zero array, appearing as an ordinary valid empty mask with 0% tumor.
   - *Engineered Fix:* Replaced silent masking with explicit validation:
     ```python
     if np.isnan(prob_map).any() or np.isinf(prob_map).any():
         raise ValueError("U-Net segmentation model produced non-finite probabilities (NaN or Inf). Inference corrupted.")
     ```
     Surfaces corrupted segmentation as HTTP 500 `Inference Failure`.

3. **Explainability Decoupling & Isolation:**
   - *Behavior:* If LayerCAM backpropagation encounters non-finite target scores or gradients, `app/explainability.py` raises an explicit `ValueError`. In `app/main.py`, this is cleanly handled:
     ```python
     except Exception as xai_err:
         layercam_map = None
         target_definition = "Error during LayerCAM generation"
     ```
     This ensures explainability failures do NOT alter or invalidate valid U-Net segmentation predictions or corrupt numerical measurements.

### Four Operational States Distinguisher

| Operational State | Model Output Condition | HTTP Status | Response Schema Attributes |
| :--- | :--- | :--- | :--- |
| **1. Valid Detection** | Classifier $> 0.5$, U-Net mask non-empty | `200 OK` | `tumor_detected: true`, `area_pixels > 0`, LayerCAM rendered, shape metrics populated. |
| **2. Valid Empty Mask** | Legitimate MRI with 0 foreground pixels | `200 OK` | `area_pixels: 0`, `area_percentage: 0.0`, shape descriptors `null`, LayerCAM marked `Unavailable`. |
| **3. Invalid Model Output** | Non-finite logits, `NaN` probs, corrupted weights | `500 Internal Error` | Structured JSON: `{"detail": "Classification Failure: Stage 1 classifier model output is non-finite..."}`. No stack traces leaked. |
| **4. Unavailable XAI** | Valid U-Net segmentation, failed LayerCAM backprop | `200 OK` | Mask & measurements intact, `outputs.layercam = null`, `target_definition = "Error during LayerCAM generation"`. |

---

## 6. API & Measurement Integrity

- **Spatial Coordinate Grounding:**
  All measurements are labeled explicitly in native 2D pixel space:
  - `area_pixels`: Integer count of foreground pixels (`px`).
  - `area_percentage`: Relative foreground slice proportion (`%`), computed as `(area_pixels / 65536.0) * 100.0`.
  - `bounding_box`: Pixel coordinates `(x, y, width, height)`.
  - `centroid`: Pixel centroid `(x, y)` in range $[0, 256]$.
- **Shape Descriptors on Empty Masks:**
  When `area_pixels == 0`, shape descriptors (`circularity`, `eccentricity`, `aspect_ratio`, `solidity`, `perimeter`, `extent`) evaluate to `null` (`None`). They are NOT fabricated as `0.0` or clamped to `1.0`.
- **Prohibition of Unverified Physical Units:**
  No physical millimeter ($mm$), centimeter ($cm$), or 3D volume ($cm^3$) metrics are generated from uploaded 2D PNG/JPEG files. Slice thickness and in-plane pixel spacing ($\Delta x, \Delta y, \Delta z$) are not present in standard 2D image headers and are strictly not inferred.
- **Input Validation Security:**
  Requests with empty files return HTTP 400 (`File size is 0 bytes`). Non-image payloads (e.g. text/plain) return HTTP 400 (`Invalid file type`). Corrupt byte streams return HTTP 400 (`Corrupt image file`).

---

## 7. Verification Test Suite Results

All tests were executed against the actual active codebase and local Python/CUDA environment.

| Test Suite / Command | Focus Area | Result | Pass / Total |
| :--- | :--- | :--- | :--- |
| `python tests/test_failure_paths_and_reproducibility.py` | Non-finite classifier/U-Net handling, LayerCAM isolation, 3-run determinism, checkpoint immutability, input rejection | **PASSED** | 8 / 8 tests |
| `python tests/test_explainability_confidence.py` | Explainability thresholding, sub-threshold suppression, confidence intervals | **PASSED** | 9 / 9 tests |
| `python tests/test_measurements.py` | 2D Morphometrics mathematical edge cases (empty masks, single pixels, irregular shapes) | **PASSED** | 19 / 19 tests |
| `python tests/test_morphometrics_api_integration.py` | API schema integration and JSON response contract for shape descriptors | **PASSED** | 3 / 3 tests |
| `python tests/test_unconditional_segmentation.py` | Unconditional U-Net execution across positive and negative classifier states | **PASSED** | 6 / 6 tests |
| `python tests/test_live_detection_consistency.py` | Saliency thresholding, negative image heatmap suppression, boundary logic | **PASSED** | 8 / 8 tests |
| `python tests/test_live_api_consistency.py` | Live running Uvicorn server consistency on positive/negative control slices | **PASSED** | 2 / 2 slices verified |
| `python tests/test_backend_api.py` | Complete backend API verification (Phase 1, 2 XAI, Phase 4 Signals & Systems) | **PASSED** | 32 / 32 tests |
| `npm run lint` (in `frontend/`) | Static code analysis with Oxlint | **PASSED** | 0 errors, 9 warnings (13 files) |
| `npm run build` (in `frontend/`) | Vite production bundling | **PASSED** | Built in 1.87s, 0 errors |

**Summary of Automated Checks:** **87 / 87 tests passed across all executed suites.** Zero test failures observed.

---

## 8. Frontend & Browser Verification Limitations

- **Installed Tooling Check:**
  - `oxlint`: Executed cleanly with 0 errors across 13 frontend files.
  - `vite build`: Completed production bundling in 1.87s (`dist/assets/index-Cv3_79R2.js`, 296.23 kB).
- **Browser Automation Limitation:**
  Automated browser driving via Playwright was previously halted by an external Azure CDN 404 error during driver acquisition. In accordance with safety rules, no external driver installation was re-attempted.
- **Reporting Statement:**
  Visual browser verification remains unperformed in this audit. The successful Vite production build and static analysis prove syntactical and module integrity, but do NOT constitute proof of visual UI rendering.

---

## 9. Modified & Newly Created Files

### Modified Files (Tracked)
1. [`app/classifier_xai.py`](file:///C:/Users/HP/OneDrive/Desktop/Projects/Sushruta/app/classifier_xai.py)
   - Added explicit `ValueError` on non-finite classifier logits/probabilities to prevent silent conversion into false negative predictions.
2. [`app/inference.py`](file:///C:/Users/HP/OneDrive/Desktop/Projects/Sushruta/app/inference.py)
   - Replaced silent `nan_to_num(prob_map, nan=0.0)` sanitization with explicit non-finite validation raising `ValueError`.
3. [`app/explainability.py`](file:///C:/Users/HP/OneDrive/Desktop/Projects/Sushruta/app/explainability.py)
   - Added validation for target score finiteness before backward pass, raising `ValueError` on corrupt activations.
4. [`tests/test_explainability_confidence.py`](file:///C:/Users/HP/OneDrive/Desktop/Projects/Sushruta/tests/test_explainability_confidence.py)
   - Updated unit test assertions to require `ValueError` on non-finite classifier logits.

### Newly Created Files (Untracked)
1. [`tests/test_failure_paths_and_reproducibility.py`](file:///C:/Users/HP/OneDrive/Desktop/Projects/Sushruta/tests/test_failure_paths_and_reproducibility.py)
   - Comprehensive test suite auditing non-finite error surfacing, LayerCAM error resilience, 3-run positive/negative determinism, checkpoint file immutability, missing checkpoint handling, and input payload validation.
2. [`results/audit/27_part_5_reproducibility_release_readiness.md`](file:///C:/Users/HP/OneDrive/Desktop/Projects/Sushruta/results/audit/27_part_5_reproducibility_release_readiness.md)
   - This audit report document.

---

## 10. Explicit Non-Goals & Boundaries

1. **No Clinical Claims:** This release audit verifies software reproducibility, mathematical consistency, and defensive error handling. It does NOT assert diagnostic accuracy or clinical safety for patient care.
2. **No Physical Volume Derivation:** 3D tumor volume reconstruction and physical millimeter conversions are strictly excluded from 2D slice inference.
3. **No Integration of Experimental Models:** Experimental checkpoints (`exp_clinical_04_25d_best.pt`, `exp_clinical_04b_25d_best.pt`) remain in experimental quarantine and are intentionally not integrated into production.

---

## 11. Final Recommendation

The engineering pipeline exhibits robust numerical determinism, strict failure-path surfacing, and accurate measurement reporting. Subject to the noted browser testing caveats and clinical disclaimer boundaries, the software pipeline is **READY FOR TECHNICAL RELEASE REVIEW**.
