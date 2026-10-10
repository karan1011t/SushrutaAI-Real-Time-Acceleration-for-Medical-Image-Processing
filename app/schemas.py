"""
SUSHRUTA Backend Pydantic Schemas for API Request/Response Data Validation (Phase 2).
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    x: int = Field(..., description="Top-left x-coordinate of bounding box in pixels")
    y: int = Field(..., description="Top-left y-coordinate of bounding box in pixels")
    width: int = Field(..., description="Bounding box width in pixels")
    height: int = Field(..., description="Bounding box height in pixels")


class ShapeDescriptors(BaseModel):
    perimeter_pixels: Optional[float] = Field(None, description="External perimeter of primary lesion in processed-grid pixels")
    circularity: Optional[float] = Field(None, description="Circularity (4*pi*area / perimeter^2) in [0.0, 1.0], null if undefined")
    solidity: Optional[float] = Field(None, description="Solidity (lesion area / convex hull area) in [0.0, 1.0], null if undefined")
    major_axis_length: Optional[float] = Field(None, description="Equivalent inertia ellipse major axis length in processed-grid pixels")
    minor_axis_length: Optional[float] = Field(None, description="Equivalent inertia ellipse minor axis length in processed-grid pixels")
    eccentricity: Optional[float] = Field(None, description="Ellipse eccentricity in [0.0, 1.0], null if undefined")
    feret_diameter_max: Optional[float] = Field(None, description="Maximum Feret caliper diameter in processed-grid pixels")
    aggregation_policy: str = Field("largest_connected_component", description="Morphology aggregation policy (primary lesion mass)")
    coordinate_system: str = Field("processed_mask_grid", description="Coordinate space of descriptors (processed 2D grid pixels)")


class Measurements(BaseModel):
    tumor_detected: bool = Field(..., description="Whether tumor pixels were detected above threshold")
    area_pixels: int = Field(..., description="Total count of predicted tumor pixels")
    area_percentage: float = Field(..., description="Percentage of slice area occupied by predicted tumor")
    bounding_box: BoundingBox = Field(..., description="Bounding box surrounding tumor region")
    width_pixels: int = Field(..., description="Tumor bounding box width in pixels")
    height_pixels: int = Field(..., description="Tumor bounding box height in pixels")
    diameter_pixels: int = Field(..., description="Maximum spatial bounding box dimension in pixels")
    num_connected_components: int = Field(0, description="Total count of disconnected tumor islands")
    largest_component_area_pixels: int = Field(0, description="Area in pixels of largest connected tumor component")
    centroid: Dict[str, float] = Field(
        default_factory=lambda: {"x": 0.0, "y": 0.0},
        description="Tumor spatial center of mass coordinates {x, y}"
    )
    largest_component_bounding_box: Optional[BoundingBox] = Field(
        default=None,
        description="Bounding box for largest connected component"
    )
    # Phase 3A: Quantitative 2D Morphological Descriptors (Primary Lesion Mass)
    perimeter_pixels: Optional[float] = Field(None, description="External perimeter of primary lesion in processed-grid pixels")
    circularity: Optional[float] = Field(None, description="Circularity (4*pi*area / perimeter^2) in [0.0, 1.0]")
    solidity: Optional[float] = Field(None, description="Solidity (lesion area / convex hull area) in [0.0, 1.0]")
    major_axis_length: Optional[float] = Field(None, description="Equivalent inertia ellipse major axis length in processed-grid pixels")
    minor_axis_length: Optional[float] = Field(None, description="Equivalent inertia ellipse minor axis length in processed-grid pixels")
    eccentricity: Optional[float] = Field(None, description="Ellipse eccentricity in [0.0, 1.0]")
    feret_diameter_max: Optional[float] = Field(None, description="Maximum Feret caliper diameter in processed-grid pixels")
    shape_descriptors: Optional[ShapeDescriptors] = Field(None, description="Structured 2D shape descriptors for primary lesion")
    spatial_unit: Optional[str] = Field("pixels", description="Spatial unit of measurement (processed grid pixels)")
    diameter_definition: Optional[str] = Field(None, description="Definition of diameter_pixels")
    note: Optional[str] = Field(None, description="Measurement notes and calibration limitation disclaimer")


class InferenceMetadata(BaseModel):
    time_ms: float = Field(..., description="Backward-compatible segmentation forward pass latency in ms")
    segmentation_inference_time_ms: float = Field(..., description="U-Net forward pass latency in ms")
    layercam_time_ms: float = Field(0.0, description="LayerCAM explainability execution latency in ms")
    preprocessing_time_ms: float = Field(..., description="Image reading & normalization latency in ms")
    total_request_time_ms: float = Field(..., description="Total end-to-end request latency in ms")
    device: str = Field(..., description="Compute device used for inference (cuda / cpu)")
    threshold_used: float = Field(0.5, description="Decision threshold applied to probabilities")


class ExplainabilityMetadata(BaseModel):
    method: str = Field("LayerCAM", description="Explainable AI attribution method")
    target_layer: str = Field(..., description="Target convolutional layer hooked for LayerCAM")
    target_definition: str = Field(..., description="Mathematical definition of backpropagated target score Y")
    interpretation: str = Field(..., description="Interpretation guidance for attribution heatmap")
    layercam_time_ms: float = Field(..., description="LayerCAM calculation overhead in ms")


class ClassificationXAI(BaseModel):
    method: str = Field("LayerCAM", description="XAI attribution method name")
    target_layer: str = Field("features.12", description="Target convolutional layer")
    strongest_region_location: str = Field(..., description="Image-relative spatial region e.g. center-right")
    centroid: Dict[str, float] = Field(..., description="Spatial centroid coordinates x, y")
    bounding_box: BoundingBox = Field(..., description="Bounding box surrounding strongest XAI region")


class ClassificationMetadata(BaseModel):
    tumor_detected: bool = Field(..., description="Binary decision: whether tumor is detected above threshold")
    tumor_probability: float = Field(..., description="Image-level tumor probability from BinaryTumorClassifier [0.0 to 1.0]")
    tumor_logit: float = Field(..., description="Raw logit score from BinaryTumorClassifier")
    decision_threshold: float = Field(0.5, description="Applied decision threshold (default 0.5)")
    classification_label: str = Field(..., description="Human-readable decision label (Tumor Detected / No Tumor Detected)")
    explanation: str = Field(..., description="Dynamically generated natural language explanation")
    xai: ClassificationXAI = Field(..., description="Classifier LayerCAM XAI spatial metadata")


class SegmentationMetadata(BaseModel):
    performed: bool = Field(..., description="Whether U-Net pixel-wise segmentation was executed")
    measurements: Measurements = Field(..., description="Tumor spatial morphometric measurements")
    confidence: float = Field(0.0, description="Mean pixel-wise segmentation confidence [0.0, 1.0]")


class ImageDimensions(BaseModel):
    original_shape: List[int] = Field(..., description="Original uploaded image dimensions [height, width]")
    processed_shape: List[int] = Field([256, 256], description="Processed tensor dimensions [height, width]")


class OutputArtifacts(BaseModel):
    preprocessed_mri: str = Field(..., description="URL/path to saved preprocessed grayscale MRI")
    classifier_layercam: Optional[str] = Field(None, description="URL/path to saved Classifier LayerCAM overlay")
    tumor_mask: Optional[str] = Field(None, description="URL/path to saved binary tumor mask")
    overlay: Optional[str] = Field(None, description="URL/path to saved MRI + tumor overlay visualization")
    probability_map: Optional[str] = Field(None, description="URL/path to saved probability heatmap")
    layercam: Optional[str] = Field(None, description="URL/path to saved LayerCAM heatmap")
    layercam_overlay: Optional[str] = Field(None, description="URL/path to saved MRI + LayerCAM overlay")
    combined_overlay: Optional[str] = Field(None, description="URL/path to saved combined MRI + mask + LayerCAM visualization")


class PredictionResponse(BaseModel):
    success: bool = Field(True, description="Request execution status")
    tumor_detected: bool = Field(..., description="Whether tumor was detected above decision threshold")
    classification: ClassificationMetadata = Field(..., description="Stage 1 Image-Level Binary Classification & XAI metadata")
    segmentation: SegmentationMetadata = Field(..., description="Stage 2 U-Net Pixel-Wise Segmentation metadata")
    measurements: Measurements = Field(..., description="Tumor spatial morphometric measurements in pixels")
    segmentation_confidence: Optional[float] = Field(
        None,
        description="Mean tumor probability over predicted tumor pixels (0.0 to 1.0)"
    )
    explainability: Optional[ExplainabilityMetadata] = Field(
        None,
        description="Phase 2 LayerCAM explainability metadata"
    )
    inference: InferenceMetadata = Field(..., description="Performance benchmarking latency details")
    image_dimensions: ImageDimensions = Field(..., description="Image dimension metadata")
    outputs: OutputArtifacts = Field(..., description="Generated image artifact endpoints")
    disclaimer: str = Field(
        "SUSHRUTA is an academic clinical decision-support prototype. Output is NOT a medical diagnosis.",
        description="Clinical regulatory disclaimer"
    )


class HealthResponse(BaseModel):
    status: str = Field("ok", description="API operational status")
    model_loaded: bool = Field(True, description="Whether U-Net segmentation checkpoint is loaded")
    device: str = Field(..., description="Hardware device running inference (cuda / cpu)")
    model_path: str = Field(..., description="Path to active PyTorch checkpoint file")
    checkpoint_epoch: Optional[int] = Field(None, description="Checkpoint training epoch")
    trainable_parameters: int = Field(..., description="Total trainable parameter count (4,317,825)")


class ErrorResponse(BaseModel):
    success: bool = Field(False, description="Request execution status")
    error: str = Field(..., description="Short error classification")
    detail: str = Field(..., description="Human-readable error explanation")


class SSOutputArtifacts(BaseModel):
    histogram: str = Field(..., description="URL to 256-bin intensity histogram plot")
    filter_comparison: str = Field(..., description="URL to 3-panel spatial filtering comparison (Original | Gaussian | Median)")
    fft_analysis: str = Field(..., description="URL to 3-panel 2D FFT magnitude & log spectrum visualization")
    edge_detection: str = Field(..., description="URL to 3-panel edge detection (Original | Sobel | Canny)")
    morphology_analysis: str = Field(..., description="URL to 5-panel mathematical morphology progression")


class SignalAnalysisResponse(BaseModel):
    success: bool = Field(True, description="Request execution status")
    signal: Dict[str, Any] = Field(..., description="2D Discrete Spatial Signal statistics I[m,n]")
    histogram: Dict[str, Any] = Field(..., description="Intensity distribution statistics")
    filtering: Dict[str, Any] = Field(..., description="Gaussian & Median spatial filtering trade-off metrics")
    frequency: Dict[str, Any] = Field(..., description="2D Fourier Transform spectral energy distribution")
    edges: Dict[str, Any] = Field(..., description="Sobel & Canny edge pixel counts")
    morphology: Dict[str, Any] = Field(..., description="Binary mathematical morphology parameters")
    connected_components: Dict[str, Any] = Field(..., description="Connected component spatial topography")
    snr_estimate: Dict[str, Any] = Field(..., description="Estimated tissue and global Signal-to-Noise Ratio (SNR)")
    timings: Dict[str, Any] = Field(..., description="Latency breakdown across SS signal processing stages")
    outputs: SSOutputArtifacts = Field(..., description="Generated SS visualization artifact endpoints")
