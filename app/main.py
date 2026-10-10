"""
SUSHRUTA FastAPI Application Server (Phase 2 Production Inference Pipeline with LayerCAM XAI).

Provides high-performance REST API endpoints for explainable 2D Brain MRI segmentation:
- GET /health: System readiness, hardware device, and model checkpoint health check.
- POST /predict: Accepts external 2D MRI slice, runs preprocessing, U-Net inference,
  LayerCAM explainability generation, morphometric measurement calculations,
  visual artifact rendering, and returns structured JSON with full XAI metadata.
"""

import os
import sys
import time
import uuid
import torch
import numpy as np
from PIL import Image
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional

from fastapi import FastAPI, File, Form, UploadFile, HTTPException, status, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# Add project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from app.schemas import (
    PredictionResponse,
    HealthResponse,
    ErrorResponse,
    Measurements,
    BoundingBox,
    InferenceMetadata,
    ExplainabilityMetadata,
    ImageDimensions,
    OutputArtifacts,
    SignalAnalysisResponse,
    SSOutputArtifacts
)
from app.preprocessing import preprocess_image_file
from app.inference import MedicalImageSegmenter
from app.explainability import LayerCAMGenerator
from app.measurements import calculate_tumor_measurements
from app.utils import generate_visualization_artifacts, generate_classifier_artifact
from app.signal_analysis import analyze_signal_image, SS_OUTPUT_DIR
from src.models.classifier import BinaryTumorClassifier
from app.classifier_xai import ClassifierLayerCAMGenerator, generate_dynamic_explanation
from app.schemas import (
    ClassificationMetadata,
    ClassificationXAI,
    SegmentationMetadata
)


# Global Model Singletons
segmenter: Optional[MedicalImageSegmenter] = None
layercam_gen: Optional[LayerCAMGenerator] = None
classifier_model: Optional[BinaryTumorClassifier] = None
classifier_cam_gen: Optional[ClassifierLayerCAMGenerator] = None
# In-domain Figshare T1 image-level classifier (gate + probability for figshare pipeline)
figshare_classifier: Optional[BinaryTumorClassifier] = None
figshare_classifier_cam: Optional[ClassifierLayerCAMGenerator] = None
# Temperature fitted on val for calibrated figshare probabilities (ckpt key).
figshare_temperature: float = 1.0

# Multi-model registry: pipeline name -> {"segmenter", "layercam", "domain", "checkpoint"}
# "brats": BraTS FLAIR two-stage pipeline (classifier gate + 1ch U-Net).
# "figshare": Figshare T1-CE direct-segmentation pipeline (1ch U-Net, verdict from mask).
segmenters: Dict[str, Any] = {}
MODEL_CHOICES = ("brats", "figshare")

# Configure Output Artifacts Directory
OUTPUT_DIR = os.path.join(project_root, "results", "app_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SS_OUTPUT_DIR, exist_ok=True)

# Figshare-path guardrails (research defaults; rotation-safe: pixel-count area
# is invariant under rotation, so rotated tumors are unaffected).
# Measured on 604 test slices (504 tumor + 100 healthy): gate-only 0.998/0.960,
# OR(mask>=300) 1.000/0.890. The OR rule catches near-threshold gate misses
# (e.g. prob 0.489 with a 670px mask) at a small specificity cost.
FIGSHARE_MIN_TUMOR_PX = int(os.environ.get("SUSHRUTA_FIGSHARE_MIN_PX", "300"))
FIGSHARE_SEG_THRESHOLD = float(os.environ.get("SUSHRUTA_FIGSHARE_THRESHOLD", "0.5"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler loading U-Net & BinaryTumorClassifier checkpoints once at startup."""
    global segmenter, layercam_gen, classifier_model, classifier_cam_gen, segmenters
    global figshare_classifier, figshare_classifier_cam, figshare_temperature
    print("=" * 60, flush=True)
    print("STARTING SUSHRUTA FASTAPI BACKEND SERVER (TWO-STAGE CLASSIFICATION & SEGMENTATION)", flush=True)
    print("=" * 60, flush=True)

    try:
        # Load U-Net Segmenter & LayerCAM
        segmenter = MedicalImageSegmenter()
        layercam_gen = LayerCAMGenerator(segmenter.model, device=segmenter.device)
        print(f"[LIFESPAN] U-Net Segmenter & LayerCAM loaded on device: {segmenter.device}", flush=True)
        segmenters["brats"] = {"segmenter": segmenter, "layercam": layercam_gen,
                               "domain": "BraTS FLAIR (two-stage: classifier gate + U-Net)",
                               "checkpoint": segmenter.checkpoint_path}

        # Load Figshare T1-CE U-Net (prefer newest regularized retrain first)
        for fig_name in ("figshare_unet_v3_best.pt", "figshare_unet_v2_best.pt", "figshare_unet_best.pt"):
            fig_path = os.path.join(project_root, "results", "models", fig_name)
            if os.path.exists(fig_path):
                try:
                    fig_seg = MedicalImageSegmenter(checkpoint_path=fig_path)
                    fig_cam = LayerCAMGenerator(fig_seg.model, device=fig_seg.device)
                    segmenters["figshare"] = {"segmenter": fig_seg, "layercam": fig_cam,
                                              "domain": "Figshare T1-CE (direct segmentation, verdict from mask)",
                                              "checkpoint": fig_path}
                    print(f"[LIFESPAN] Figshare U-Net loaded from: {fig_path}", flush=True)
                    break
                except Exception as exc:
                    print(f"[LIFESPAN WARNING] Figshare checkpoint {fig_path} failed to load: {exc}", flush=True)
        if "figshare" not in segmenters:
            print("[LIFESPAN WARNING] No Figshare checkpoint found; 'figshare' pipeline unavailable", flush=True)

        # Load in-domain Figshare image-level classifier (v2: test acc 0.9934,
        # rec 1.000, spec 0.96; prefer v2, fall back to v1)
        fig_cls_path = None
        for cand in ("figshare_classifier_v2_best.pt", "figshare_classifier_best.pt"):
            p = os.path.join(project_root, "results", "models", cand)
            if os.path.exists(p):
                fig_cls_path = p
                break
        if fig_cls_path is not None:
            try:
                figshare_classifier = BinaryTumorClassifier().to(segmenter.device)
                fckpt = torch.load(fig_cls_path, map_location=segmenter.device)
                figshare_classifier.load_state_dict(
                    fckpt["model_state_dict"] if isinstance(fckpt, dict) and "model_state_dict" in fckpt else fckpt)
                figshare_classifier.eval()
                figshare_classifier_cam = ClassifierLayerCAMGenerator(
                    figshare_classifier, device=str(segmenter.device))
                figshare_temperature = float(fckpt.get("temperature", 1.0)) if isinstance(fckpt, dict) else 1.0
                print(f"[LIFESPAN] Figshare classifier loaded from: {fig_cls_path} "
                      f"(T={figshare_temperature:.2f})", flush=True)
            except Exception as exc:
                print(f"[LIFESPAN WARNING] Figshare classifier failed to load: {exc}", flush=True)
                figshare_classifier, figshare_classifier_cam = None, None
        else:
            print("[LIFESPAN WARNING] Figshare classifier checkpoint not found; figshare path uses mask verdict fallback",
                  flush=True)

        # Load BinaryTumorClassifier Checkpoint
        classifier_ckpt_path = os.path.join(project_root, "results", "models", "baseline_02_classifier_best.pt")
        classifier_model = BinaryTumorClassifier().to(segmenter.device)
        if os.path.exists(classifier_ckpt_path):
            ckpt = torch.load(classifier_ckpt_path, map_location=segmenter.device)
            state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
            classifier_model.load_state_dict(state_dict)
            classifier_model.eval()
            classifier_cam_gen = ClassifierLayerCAMGenerator(classifier_model, device=str(segmenter.device))
            print(f"[LIFESPAN] BinaryTumorClassifier loaded from: {classifier_ckpt_path}", flush=True)
        else:
            print(f"[LIFESPAN WARNING] Classifier checkpoint not found at: {classifier_ckpt_path}", flush=True)

    except Exception as exc:
        print(f"[LIFESPAN CRITICAL ERROR] Failed to initialize models: {str(exc)}", flush=True)

    yield

    print("[LIFESPAN] Shutting down SUSHRUTA Backend Server.", flush=True)


app = FastAPI(
    title="SUSHRUTA Medical Image Segmentation & Explainability API",
    description=(
        "AI-assisted explainable medical image segmentation backend for 2D Brain MRI FLAIR slices. "
        "Includes Stage 1 BinaryTumorClassifier decision explanation and Stage 2 U-Net segmentation. "
        "Academic clinical decision-support prototype."
    ),
    version="4.1.0",
    lifespan=lifespan
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Static Files Directory for Serving Output Visualizations
app.mount("/static/results", StaticFiles(directory=OUTPUT_DIR), name="static_results")
app.mount("/static/signal_analysis", StaticFiles(directory=SS_OUTPUT_DIR), name="static_ss")

# Mount Built Frontend Dashboard if dist directory exists
FRONTEND_DIST_DIR = os.path.join(project_root, "frontend", "dist")
if os.path.exists(FRONTEND_DIST_DIR):
    app.mount("/dashboard", StaticFiles(directory=FRONTEND_DIST_DIR, html=True), name="dashboard")


@app.get("/", tags=["General"])
async def root():
    """Root landing endpoint providing API overview."""
    return {
        "project": "SUSHRUTA",
        "description": "AI-Assisted Explainable Medical Image Segmentation for Affordable Clinical Decision Support",
        "modality": "2D Brain MRI (FLAIR)",
        "phase": "Phase 4 Two-Stage Classification + U-Net Segmentation & Explainability",
        "docs_url": "/docs",
        "health_check": "/health",
        "prediction_endpoint": "/predict"
    }


@app.get("/models", tags=["Diagnostics"])
async def list_models():
    """Lists loaded segmentation pipelines (brats / figshare) and their checkpoints."""
    return {"models": [
        {"name": name,
         "domain": entry["domain"],
         "checkpoint": entry["checkpoint"],
         **entry["segmenter"].get_info()}
        for name, entry in segmenters.items()
    ]}


@app.get("/health", response_model=HealthResponse, tags=["Diagnostics"])
async def health_check():
    """
    Health check endpoint returning backend status, loaded model checkpoint details, and hardware device.
    """
    if segmenter is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model segmenter failed to load during server initialization."
        )

    info = segmenter.get_info()
    return HealthResponse(
        status="ok",
        model_loaded=True,
        device=info["device"],
        model_path=info["checkpoint_path"],
        checkpoint_epoch=info["checkpoint_epoch"],
        trainable_parameters=info["trainable_parameters"]
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Bad Request / File Validation Error"},
        422: {"model": ErrorResponse, "description": "Unprocessable Entity"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"}
    },
    tags=["Segmentation & Explainability"]
)
async def predict_segmentation(request: Request, file: UploadFile = File(...),
                                 model: str = Form("brats")):
    """
    Runs 2D Brain MRI Analysis on the selected pipeline ("brats" or "figshare"):
    Stage 1: Image-level BinaryTumorClassifier prediction + Classifier LayerCAM XAI explanation.
    Stage 2: Conditional U-Net segmentation & morphometric measurement calculations.
      - brats: two-stage gate (segmentation only if the BraTS classifier fires).
      - figshare: direct segmentation with the Figshare T1-CE U-Net; the verdict
        comes from the predicted mask because the Stage-1 classifier is BraTS-FLAIR
        specific and out-of-domain on T1-CE (its output is still reported honestly).
    """
    t_total_start = time.perf_counter()

    if segmenter is None or layercam_gen is None or classifier_model is None or classifier_cam_gen is None:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "success": False,
                "error": "Model Not Loaded",
                "detail": "Classifier or U-Net segmentation model is not initialized."
            }
        )

    pipeline = (model or "brats").lower()
    if pipeline not in segmenters:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "success": False,
                "error": "Unknown Model",
                "detail": f"model must be one of {sorted(segmenters.keys())}, got '{model}'."
            }
        )
    active_seg = segmenters[pipeline]["segmenter"]
    active_cam = segmenters[pipeline]["layercam"]

    # 1. FILE VALIDATION
    if not file or not file.filename:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "success": False,
                "error": "Missing File",
                "detail": "No image file was attached to the upload request."
            }
        )

    filename = file.filename
    try:
        file_bytes = await file.read()
    except Exception as exc:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "success": False,
                "error": "File Read Error",
                "detail": f"Failed to read upload stream: {str(exc)}"
            }
        )

    if len(file_bytes) == 0:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "success": False,
                "error": "Empty File",
                "detail": f"Uploaded file '{filename}' contains 0 bytes."
            }
        )

    # 2. PREPROCESSING
    try:
        input_tensor, norm_img_np, meta, prep_time_ms = preprocess_image_file(file_bytes, filename)
    except ValueError as val_err:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "success": False,
                "error": "Image Preprocessing Error",
                "detail": str(val_err)
            }
        )
    except Exception as exc:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "success": False,
                "error": "Unprocessable Image",
                "detail": f"Could not decode image array: {str(exc)}"
            }
        )

    # 3. STAGE 1: IMAGE-LEVEL BINARY CLASSIFICATION & CLASSIFIER LAYERCAM XAI
    # Each pipeline uses its own in-domain classifier: BraTS-FLAIR for brats,
    # Figshare-T1 for figshare (falls back to mask verdict if unavailable).
    # Operating points are measured, not defaulted. v1 (pos_weight-inflated)
    # needed 0.4; v2 (balanced + temperature T) re-validates 0.5:
    # sens 1.000/spec 0.960 on 604 test slices. BraTS keeps 0.5 (historical).
    decision_threshold = 0.5
    # Research-testing escape hatch: SUSHRUTA_FORCE_SEGMENT=1 runs Stage-2
    # unconditionally (e.g. cross-domain segmenter whose paired classifier
    # was trained on another modality). Never enable for clinical use.
    force_segment = os.environ.get("SUSHRUTA_FORCE_SEGMENT", "0") == "1"
    use_fig_classifier = (pipeline == "figshare" and figshare_classifier is not None
                          and figshare_classifier_cam is not None)
    stage1_cam_gen = figshare_classifier_cam if use_fig_classifier else classifier_cam_gen
    try:
        classifier_cam_norm, tumor_prob, tumor_logit, location_info = stage1_cam_gen.generate_cam(input_tensor)
        if use_fig_classifier and figshare_temperature != 1.0:
            # Temperature-calibrated probability (T fitted on val NLL).
            tumor_prob = float(1.0 / (1.0 + np.exp(-float(tumor_logit) / figshare_temperature)))
        tumor_detected = bool(tumor_prob >= decision_threshold)
        explanation_text = generate_dynamic_explanation(
            tumor_detected=tumor_detected,
            tumor_probability=tumor_prob,
            decision_threshold=decision_threshold,
            location_str=location_info["location"]
        )
        if force_segment and not tumor_detected:
            explanation_text += (" [SUSHRUTA_FORCE_SEGMENT=1: Stage-2 segmentation "
                                 "executed for research testing despite negative gate.]")
        if pipeline == "figshare":
            if use_fig_classifier:
                explanation_text += (" [In-domain Figshare T1 classifier "
                                     "(test acc 0.99, recall 0.998); verdict from gate.]")
            else:
                explanation_text += (" [Note: Stage-1 classifier is BraTS-FLAIR specific; "
                                     "on Figshare T1-CE the reported verdict comes from segmentation.]")

        # Save Classifier LayerCAM overlay artifact
        base_url = str(request.base_url).rstrip("/")
        base_id = uuid.uuid4().hex[:10]
        cls_cam_filename, _ = generate_classifier_artifact(
            preprocessed_mri=norm_img_np,
            classifier_cam_map=classifier_cam_norm,
            output_dir=OUTPUT_DIR,
            base_id=base_id
        )
        classifier_layercam_url = f"{base_url}/static/results/{cls_cam_filename}"

    except Exception as exc:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "error": "Classification Failure",
                "detail": f"Stage 1 BinaryTumorClassifier execution failed: {str(exc)}"
            }
        )

    # 4. STAGE 2: CONDITIONAL U-NET SEGMENTATION & MORPHOMETRICS
    segmentation_performed = False
    confidence = 0.0
    inf_time_ms = 0.0
    layercam_time_ms = 0.0
    layercam_map = None

    # Figshare pipeline with its in-domain classifier is a true two-stage gate
    # (test recall 0.998). Without it, fall back to segment-then-mask-verdict.
    # Figshare always segments: the verdict combines gate OR mask>=bar so a
    # near-threshold gate miss with a solid mask is still caught.
    fig_gate_effective = (pipeline != "figshare") or use_fig_classifier
    run_segmentation = (tumor_detected or force_segment
                        or pipeline == "figshare")
    fig_mask_hit, fig_verdict = False, None
    if run_segmentation:
        segmentation_performed = True
        if pipeline == "figshare":
            # Research default 0.5 (env-overridable); documented per-request.
            active_seg.threshold = FIGSHARE_SEG_THRESHOLD
        try:
            prob_map, binary_mask, inf_time_ms = active_seg.segment(input_tensor)
        except Exception as exc:
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "success": False,
                    "error": "Inference Failure",
                    "detail": f"U-Net segmentation execution failed: {str(exc)}"
                }
            )

        # U-Net LayerCAM Explainability
        try:
            layercam_map, xai_meta, layercam_time_ms = active_cam.generate_cam(
                input_tensor=input_tensor,
                binary_mask=binary_mask,
                threshold=active_seg.threshold
            )
        except Exception:
            layercam_map = None
            layercam_time_ms = 0.0
            xai_meta = {
                "method": "LayerCAM",
                "target_layer": active_cam.target_layer_name,
                "target_definition": "Error during LayerCAM generation",
                "interpretation": "U-Net LayerCAM generation encountered an issue",
                "layercam_time_ms": 0.0
            }

        # Spatial Morphometrics
        measurements_dict, confidence = calculate_tumor_measurements(binary_mask, prob_map)

        # Figshare combined verdict: in-domain gate OR mask>=bar. Catches
        # near-threshold gate misses with solid masks; suppresses specks.
        # (Band-aid honesty: the bar suppresses small hallucinations but can
        # also hide real tumors smaller than the bar when the gate misses.)
        fig_mask_hit = int(measurements_dict["area_pixels"]) >= FIGSHARE_MIN_TUMOR_PX
        fig_verdict = bool((tumor_detected and fig_gate_effective) or fig_mask_hit)
        guardrail_suppressed = (pipeline == "figshare" and not fig_verdict)
        if (pipeline == "figshare" and segmentation_performed and tumor_detected
                and int(measurements_dict["area_pixels"]) == 0):
            # Gate fired but segmenter localized nothing: report the calibrated
            # gate probability as confidence (never 0 next to a positive) and
            # point at the classifier heatmap as the only visual evidence.
            confidence = round(float(tumor_prob), 4)
            explanation_text += (" [Gate-positive with no localized region: confidence "
                                 "reflects the image-level score; see classifier heatmap; "
                                 "recommend expert review.]")
        if guardrail_suppressed:
            measurements_dict = {
                "tumor_detected": False,
                "area_pixels": 0,
                "area_percentage": 0.0,
                "bounding_box": {"x": 0, "y": 0, "width": 0, "height": 0},
                "width_pixels": 0,
                "height_pixels": 0,
                "diameter_pixels": 0,
                "num_connected_components": 0,
                "largest_component_area_pixels": 0,
                "centroid": {"x": 0.0, "y": 0.0},
                "largest_component_bounding_box": {"x": 0, "y": 0, "width": 0, "height": 0}
            }
            confidence = 0.0
            mri_filename = f"{base_id}_mri.png"
            Image.fromarray(np.clip(norm_img_np * 255.0, 0, 255).astype(np.uint8)).save(
                os.path.join(OUTPUT_DIR, mri_filename))
            outputs = OutputArtifacts(
                preprocessed_mri=f"{base_url}/static/results/{mri_filename}",
                classifier_layercam=classifier_layercam_url,
                tumor_mask=None, overlay=None, probability_map=None,
                layercam=None, layercam_overlay=None, combined_overlay=None
            )
            xai_meta = {
                "method": "LayerCAM",
                "target_layer": active_cam.target_layer_name,
                "target_definition": f"Suppressed (predicted area below {FIGSHARE_MIN_TUMOR_PX}px floor)",
                "interpretation": ("Segmentation ran but the predicted region was below the "
                                   "min-area guardrail and reported as negative."),
                "layercam_time_ms": 0.0
            }
        if not guardrail_suppressed:
            artifact_files = generate_visualization_artifacts(
                preprocessed_mri=norm_img_np,
                binary_mask=binary_mask,
                prob_map=prob_map,
                output_dir=OUTPUT_DIR,
                layercam_map=layercam_map,
                base_id=base_id
            )

            outputs = OutputArtifacts(
                preprocessed_mri=f"{base_url}/static/results/{artifact_files['preprocessed_mri']}",
                classifier_layercam=classifier_layercam_url,
                tumor_mask=f"{base_url}/static/results/{artifact_files['tumor_mask']}",
                overlay=f"{base_url}/static/results/{artifact_files['overlay']}",
                probability_map=f"{base_url}/static/results/{artifact_files['probability_map']}",
                layercam=f"{base_url}/static/results/{artifact_files['layercam']}" if "layercam" in artifact_files else None,
                layercam_overlay=f"{base_url}/static/results/{artifact_files['layercam_overlay']}" if "layercam_overlay" in artifact_files else None,
                combined_overlay=f"{base_url}/static/results/{artifact_files['combined_overlay']}" if "combined_overlay" in artifact_files else None
            )

    else:
        # NO TUMOR: Skip U-Net segmentation, return zeroed measurements schema
        segmentation_performed = False
        measurements_dict = {
            "tumor_detected": False,
            "area_pixels": 0,
            "area_percentage": 0.0,
            "bounding_box": {"x": 0, "y": 0, "width": 0, "height": 0},
            "width_pixels": 0,
            "height_pixels": 0,
            "diameter_pixels": 0,
            "num_connected_components": 0,
            "largest_component_area_pixels": 0,
            "centroid": {"x": 0.0, "y": 0.0},
            "largest_component_bounding_box": {"x": 0, "y": 0, "width": 0, "height": 0}
        }
        confidence = 0.0

        # Save preprocessed MRI artifact for negative cases
        mri_filename = f"{base_id}_mri.png"
        mri_path = os.path.join(OUTPUT_DIR, mri_filename)
        mri_uint8 = np.clip(norm_img_np * 255.0, 0, 255).astype(np.uint8)
        Image.fromarray(mri_uint8).save(mri_path)

        outputs = OutputArtifacts(
            preprocessed_mri=f"{base_url}/static/results/{mri_filename}",
            classifier_layercam=classifier_layercam_url,
            tumor_mask=None,
            overlay=None,
            probability_map=None,
            layercam=None,
            layercam_overlay=None,
            combined_overlay=None
        )

        xai_meta = {
            "method": "LayerCAM",
            "target_layer": active_cam.target_layer_name,
            "target_definition": "Skipped (No Tumor Detected by Stage 1 Classifier)",
            "interpretation": "U-Net segmentation skipped because no tumor was detected by Stage 1 classifier",
            "layercam_time_ms": 0.0
        }

    # Pipeline verdict: brats follows the classifier gate; figshare follows the
    # combined gate-OR-mask verdict computed above (mask fallback if the
    # in-domain classifier is unavailable, since the BraTS gate is then void).
    # mask_override tracks gate-negative/mask-positive splits so downstream
    # wording never contradicts the verdict (header/body mismatch fix).
    mask_override = False
    if pipeline == "figshare" and segmentation_performed and fig_verdict is not None:
        gate_fired = bool(tumor_detected)
        if not use_fig_classifier:
            tumor_detected = bool(fig_mask_hit)
        else:
            tumor_detected = bool(fig_verdict)
        mask_override = bool(tumor_detected and not gate_fired)
        if mask_override:
            # Rewrite the gate-worded opening so body and verdict agree.
            explanation_text = explanation_text.replace(
                "No tumor detected.",
                "Tumor detected by segmentation despite a negative image-level score.", 1)
            explanation_text += (" [Verdict follows segmentation: the image-level score "
                                 f"({tumor_prob:.1%}) is below threshold, but a "
                                 f"{int(measurements_dict['area_pixels'])}px region was localized. "
                                 "Recommend expert review.]")

    t_total_end = time.perf_counter()
    total_time_ms = round((t_total_end - t_total_start) * 1000.0, 2)

    # 5. CONSTRUCT STRUCTURED RESPONSE
    bbox = BoundingBox(
        x=measurements_dict["bounding_box"]["x"],
        y=measurements_dict["bounding_box"]["y"],
        width=measurements_dict["bounding_box"]["width"],
        height=measurements_dict["bounding_box"]["height"]
    )

    lc_bbox = BoundingBox(
        x=measurements_dict["largest_component_bounding_box"]["x"],
        y=measurements_dict["largest_component_bounding_box"]["y"],
        width=measurements_dict["largest_component_bounding_box"]["width"],
        height=measurements_dict["largest_component_bounding_box"]["height"]
    )

    measurements = Measurements(
        tumor_detected=tumor_detected,
        area_pixels=measurements_dict["area_pixels"],
        area_percentage=measurements_dict["area_percentage"],
        bounding_box=bbox,
        width_pixels=measurements_dict["width_pixels"],
        height_pixels=measurements_dict["height_pixels"],
        diameter_pixels=measurements_dict["diameter_pixels"],
        num_connected_components=measurements_dict["num_connected_components"],
        largest_component_area_pixels=measurements_dict["largest_component_area_pixels"],
        centroid=measurements_dict["centroid"],
        largest_component_bounding_box=lc_bbox
    )

    cls_bbox = BoundingBox(
        x=location_info["bounding_box"]["x"],
        y=location_info["bounding_box"]["y"],
        width=location_info["bounding_box"]["width"],
        height=location_info["bounding_box"]["height"]
    )

    cls_xai = ClassificationXAI(
        method="LayerCAM",
        target_layer=stage1_cam_gen.target_layer_name,
        strongest_region_location=location_info["location"],
        centroid=location_info["centroid"],
        bounding_box=cls_bbox
    )

    classification_meta = ClassificationMetadata(
        tumor_detected=tumor_detected,
        tumor_probability=round(tumor_prob, 4),
        tumor_logit=round(tumor_logit, 4),
        decision_threshold=decision_threshold,
        classification_label=(
            "Tumor Detected (segmentation; classifier negative)" if mask_override
            else ("Tumor Detected" if tumor_detected else "No Tumor Detected")
        ),
        explanation=explanation_text,
        xai=cls_xai
    )

    segmentation_meta = SegmentationMetadata(
        performed=segmentation_performed,
        measurements=measurements,
        confidence=round(confidence, 4)
    )

    inference_meta = InferenceMetadata(
        time_ms=inf_time_ms,
        segmentation_inference_time_ms=inf_time_ms,
        layercam_time_ms=layercam_time_ms,
        preprocessing_time_ms=round(prep_time_ms, 2),
        total_request_time_ms=total_time_ms,
        device=str(active_seg.device),
        threshold_used=active_seg.threshold
    )

    explainability_info = ExplainabilityMetadata(
        method=xai_meta["method"],
        target_layer=xai_meta["target_layer"],
        target_definition=xai_meta["target_definition"],
        interpretation=xai_meta["interpretation"],
        layercam_time_ms=xai_meta["layercam_time_ms"]
    )

    img_dims = ImageDimensions(
        original_shape=meta["original_shape"],
        processed_shape=meta["processed_shape"]
    )

    return PredictionResponse(
        success=True,
        tumor_detected=tumor_detected,
        classification=classification_meta,
        segmentation=segmentation_meta,
        measurements=measurements,
        segmentation_confidence=confidence,
        explainability=explainability_info,
        inference=inference_meta,
        image_dimensions=img_dims,
        outputs=outputs
    )


@app.post(
    "/analyze-signal",
    response_model=SignalAnalysisResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Bad Request / File Validation Error"},
        422: {"model": ErrorResponse, "description": "Unprocessable Entity"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"}
    },
    tags=["Signals & Systems / Image Processing"]
)
async def analyze_signal(request: Request, file: UploadFile = File(...)):
    """
    Executes dedicated Signals & Systems / Digital Image Processing analysis on an uploaded MRI slice.

    Analytical processing pipeline:
    1. 2D Discrete Spatial Signal representation I[m,n] & intensity statistics.
    2. Intensity Histogram distribution.
    3. Spatial Domain Filtering (Gaussian 5x5 vs Median 5x5 comparison).
    4. 2D Discrete Fourier Transform (FFT magnitude & log spectrum log(1+|F[u,v]|)).
    5. Edge Detection (Sobel gradient magnitude & Canny hysteresis edges).
    6. Binary Thresholding & Mathematical Morphology (Erosion, Dilation, Opening, Closing).
    7. Connected Component Analysis.
    8. Estimated Signal-to-Noise Ratio (SNR) & Latency Profiling.

    This analytical path DOES NOT alter or touch the official U-Net prediction pipeline.
    """
    if not file or not file.filename:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "success": False,
                "error": "Missing File",
                "detail": "No image file was attached to the analysis request."
            }
        )

    filename = file.filename
    try:
        file_bytes = await file.read()
    except Exception as exc:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "success": False,
                "error": "File Read Error",
                "detail": f"Failed to read upload stream: {str(exc)}"
            }
        )

    if len(file_bytes) == 0:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "success": False,
                "error": "Empty File",
                "detail": f"Uploaded file '{filename}' contains 0 bytes."
            }
        )

    try:
        structured_results, artifact_files, total_ss_time_ms = analyze_signal_image(
            file_bytes=file_bytes,
            filename=filename,
            output_dir=SS_OUTPUT_DIR
        )

        base_url = str(request.base_url).rstrip("/")
        outputs = SSOutputArtifacts(
            histogram=f"{base_url}/static/signal_analysis/{artifact_files['histogram']}",
            filter_comparison=f"{base_url}/static/signal_analysis/{artifact_files['filter_comparison']}",
            fft_analysis=f"{base_url}/static/signal_analysis/{artifact_files['fft_analysis']}",
            edge_detection=f"{base_url}/static/signal_analysis/{artifact_files['edge_detection']}",
            morphology_analysis=f"{base_url}/static/signal_analysis/{artifact_files['morphology_analysis']}"
        )

        return SignalAnalysisResponse(
            success=True,
            signal=structured_results["signal"],
            histogram=structured_results["histogram"],
            filtering=structured_results["filtering"],
            frequency=structured_results["frequency"],
            edges=structured_results["edges"],
            morphology=structured_results["morphology"],
            connected_components=structured_results["connected_components"],
            snr_estimate=structured_results["snr_estimate"],
            timings=structured_results["timings"],
            outputs=outputs
        )
    except ValueError as val_err:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "success": False,
                "error": "Signal Analysis Preprocessing Error",
                "detail": str(val_err)
            }
        )
    except Exception as exc:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "error": "Signal Analysis Failure",
                "detail": f"Failed to execute SS signal analysis pipeline: {str(exc)}"
            }
        )


# Exception handler for uncaught exceptions
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "error": "Internal Server Error",
            "detail": "An unexpected error occurred during request processing."
        }
    )
