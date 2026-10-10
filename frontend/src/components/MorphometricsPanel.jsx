import React from 'react';
import { AlertTriangle, CheckCircle2, Grid, Info, AlertCircle } from 'lucide-react';

export default function MorphometricsPanel({ measurements, confidence, tumorDetected }) {
  if (!measurements) return null;

  const {
    area_pixels = 0,
    area_percentage = 0.0,
    bounding_box = { x: 0, y: 0, width: 0, height: 0 },
    diameter_pixels = 0,
    num_connected_components = 0,
    largest_component_area_pixels = 0,
    centroid = { x: 0.0, y: 0.0 },
    largest_component_bounding_box = { x: 0, y: 0, width: 0, height: 0 }
  } = measurements;

  const has_tumor_pixels = (area_pixels || 0) > 0;

  return (
    <div className="card" style={{ marginBottom: '1.5rem' }}>
      <div className="card-title">
        <Grid size={20} color="var(--accent-cyan)" />
        <span>Quantitative Tumor Morphometrics</span>
      </div>
      <p className="card-subtitle">
        Spatial metrics computed from the predicted binary whole-tumor segmentation mask (256 × 256 matrix, threshold τ = 0.50).
      </p>

      {/* Detection Banner */}
      <div style={{
        padding: '0.85rem 1.25rem',
        borderRadius: 'var(--radius-md)',
        marginBottom: '1.25rem',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        background: has_tumor_pixels ? 'rgba(245, 158, 11, 0.1)' : 'rgba(16, 185, 129, 0.1)',
        border: has_tumor_pixels ? '1px solid rgba(245, 158, 11, 0.3)' : '1px solid rgba(16, 185, 129, 0.3)',
        color: has_tumor_pixels ? 'var(--accent-amber)' : 'var(--accent-emerald)',
        flexWrap: 'wrap',
        gap: '0.5rem'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', fontWeight: 600, fontSize: '0.95rem' }}>
          {has_tumor_pixels ? <AlertTriangle size={20} /> : <CheckCircle2 size={20} />}
          <span>{has_tumor_pixels ? 'Whole-Tumor Pixels Segmented' : 'No Whole-Tumor Pixels Segmented (Threshold τ = 0.50)'}</span>
        </div>
        <div style={{ fontFamily: 'JetBrains Mono', fontWeight: 600, fontSize: '0.9rem' }}>
          {has_tumor_pixels ? `${area_pixels.toLocaleString()} pixels (${area_percentage}%)` : '0 pixels (0.00% of 256 × 256 slice)'}
        </div>
      </div>

      {/* Metrics Grid */}
      <div className="metrics-grid-3">
        <div className="metric-box">
          <div className="metric-label">Segmented Pixel Count</div>
          <div className="metric-value">
            {(area_pixels || 0).toLocaleString()}
            <span className="metric-unit">px ({area_percentage}%)</span>
          </div>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.2rem' }}>
            Count of pixels with P(WT) ≥ 0.50
          </div>
        </div>

        <div className="metric-box">
          <div className="metric-label">Slice Area Percentage</div>
          <div className="metric-value">
            {Number(area_percentage || 0).toFixed(2)}
            <span className="metric-unit">%</span>
          </div>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.2rem' }}>
            Fraction of 256 × 256 matrix (65,536 px)
          </div>
        </div>

        <div className="metric-box">
          <div className="metric-label">Max Bounding Dimension</div>
          <div className="metric-value">
            {diameter_pixels || 0}
            <span className="metric-unit">px</span>
          </div>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.2rem' }}>
            Maximum of bounding box width and height
          </div>
        </div>

        <div className="metric-box">
          <div className="metric-label">Connected Components</div>
          <div className="metric-value">
            {num_connected_components || 0}
            <span className="metric-unit">islands</span>
          </div>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.2rem' }}>
            Isolated 8-connected tumor regions
          </div>
        </div>

        <div className="metric-box">
          <div className="metric-label">Largest Component Area</div>
          <div className="metric-value">
            {(largest_component_area_pixels || 0).toLocaleString()}
            <span className="metric-unit">px</span>
          </div>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.2rem' }}>
            Primary contiguous lesion area
          </div>
        </div>

        <div className="metric-box">
          <div className="metric-label">Spatial Centroid</div>
          <div className="metric-value" style={{ fontSize: '1.2rem' }}>
            {has_tumor_pixels ? `(${centroid?.x ?? 0}, ${centroid?.y ?? 0})` : '(0.0, 0.0)'}
            <span className="metric-unit">px</span>
          </div>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.2rem' }}>
            {has_tumor_pixels ? 'Center of mass coordinates (x, y)' : 'Empty-mask placeholder (no segmented pixels; not anatomical coordinates)'}
          </div>
        </div>

        <div className="metric-box">
          <div className="metric-label">Mean Predicted Probability</div>
          <div className="metric-value">
            {has_tumor_pixels ? (confidence ? `${(confidence * 100).toFixed(1)}%` : 'N/A') : 'Unavailable'}
          </div>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.2rem' }}>
            {has_tumor_pixels ? 'Mean P(tumor) over segmented region' : 'Unavailable (no segmented tumor region)'}
          </div>
        </div>

        <div className="metric-box" style={{ gridColumn: 'span 2' }}>
          <div className="metric-label">Bounding-Box Geometry</div>
          <div style={{ fontFamily: 'JetBrains Mono', fontSize: '0.95rem', color: 'var(--text-primary)', marginTop: '0.25rem' }}>
            Origin: <strong style={{ color: 'var(--accent-cyan)' }}>{`(x=${bounding_box?.x ?? 0}, y=${bounding_box?.y ?? 0})`}</strong> •
            Dimensions: <strong style={{ color: 'var(--accent-cyan)' }}>{`w=${bounding_box?.width ?? 0} px, h=${bounding_box?.height ?? 0} px`}</strong>
          </div>
          {has_tumor_pixels && largest_component_bounding_box && (
            <div style={{ fontFamily: 'JetBrains Mono', fontSize: '0.8rem', color: 'var(--text-secondary)', marginTop: '0.35rem' }}>
              {`Largest Island Box: (x=${largest_component_bounding_box.x}, y=${largest_component_bounding_box.y}, w=${largest_component_bounding_box.width}, h=${largest_component_bounding_box.height})`}
            </div>
          )}
          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>
            {has_tumor_pixels ? 'Bounding-box coordinates on 256 × 256 pixel coordinate grid' : 'Empty-mask placeholder geometry (0 px segmented; not anatomical coordinates)'}
          </div>
        </div>
      </div>

      {/* Mandatory Regulatory / Technical Limitation Callout */}
      <div style={{
        background: 'rgba(15, 23, 42, 0.6)',
        border: '1px solid var(--border-color)',
        borderRadius: 'var(--radius-sm)',
        padding: '0.85rem 1rem',
        marginBottom: '0.75rem',
        display: 'flex',
        alignItems: 'flex-start',
        gap: '0.6rem',
        fontSize: '0.8rem',
        color: 'var(--text-secondary)',
        lineHeight: 1.45
      }}>
        <Info size={18} color="var(--accent-cyan)" style={{ marginTop: '0.1rem', flexShrink: 0 }} />
        <div>
          <strong style={{ color: 'var(--text-primary)' }}>Technical Limitation: </strong>
          Measurements are in processed-image pixels. Physical dimensions in mm or mm² and volume in cm³ are unavailable because the current pipeline does not preserve validated NIfTI/DICOM spacing metadata.
        </div>
      </div>

      {/* Mandatory Model Scope Callout */}
      <div style={{
        background: 'rgba(15, 23, 42, 0.6)',
        border: '1px solid var(--border-color)',
        borderRadius: 'var(--radius-sm)',
        padding: '0.85rem 1rem',
        display: 'flex',
        alignItems: 'flex-start',
        gap: '0.6rem',
        fontSize: '0.8rem',
        color: 'var(--text-secondary)',
        lineHeight: 1.45
      }}>
        <AlertCircle size={18} color="var(--accent-purple)" style={{ marginTop: '0.1rem', flexShrink: 0 }} />
        <div>
          <strong style={{ color: 'var(--text-primary)' }}>Model Scope: </strong>
          This model predicts binary whole-tumor (WT) segmentation only. It does not separately predict enhancing tumor (ET), tumor core (TC), tumor grade, molecular subtype, or clinical stage.
        </div>
      </div>
    </div>
  );
}
