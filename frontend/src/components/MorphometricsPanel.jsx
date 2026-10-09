import React from 'react';
import { Layers, AlertTriangle, CheckCircle2, Box, Crosshair, Grid } from 'lucide-react';

export default function MorphometricsPanel({ measurements, confidence, tumorDetected }) {
  if (!measurements) return null;

  const {
    area_pixels,
    area_percentage,
    bounding_box,
    diameter_pixels,
    num_connected_components,
    largest_component_area_pixels,
    centroid,
    largest_component_bounding_box
  } = measurements;

  const is_tumor_detected = (tumorDetected !== undefined)
    ? tumorDetected
    : ((measurements.tumor_detected !== undefined) ? measurements.tumor_detected : (area_pixels > 0));

  return (
    <div className="card" style={{ marginBottom: '1.5rem' }}>
      <div className="card-title">
        <Grid size={20} color="var(--accent-cyan)" />
        <span>Quantitative Tumor Morphometrics</span>
      </div>
      <p className="card-subtitle">
        Spatial metrics calculated directly from the predicted binary segmentation mask (256 × 256 resolution).
      </p>

      {/* Detection Banner */}
      <div style={{
        padding: '0.85rem 1.25rem',
        borderRadius: 'var(--radius-md)',
        marginBottom: '1.25rem',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        background: is_tumor_detected ? 'rgba(245, 158, 11, 0.1)' : 'rgba(16, 185, 129, 0.1)',
        border: is_tumor_detected ? '1px solid rgba(245, 158, 11, 0.3)' : '1px solid rgba(16, 185, 129, 0.3)',
        color: is_tumor_detected ? 'var(--accent-amber)' : 'var(--accent-emerald)'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', fontWeight: 600, fontSize: '0.95rem' }}>
          {is_tumor_detected ? <AlertTriangle size={20} /> : <CheckCircle2 size={20} />}
          <span>{is_tumor_detected ? 'Predicted Tumor Detected' : 'No Tumor Region Detected Above Threshold'}</span>
        </div>
        <div style={{ fontFamily: 'JetBrains Mono', fontWeight: 600, fontSize: '0.9rem' }}>
          {is_tumor_detected ? `${area_pixels.toLocaleString()} pixels (${area_percentage}%)` : '0 pixels (0.0%)'}
        </div>
      </div>

      {/* Metrics Grid */}
      <div className="metrics-grid-3">
        <div className="metric-box">
          <div className="metric-label">Tumor Area</div>
          <div className="metric-value">
            {area_pixels.toLocaleString()}
            <span className="metric-unit">px ({area_percentage}%)</span>
          </div>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.2rem' }}>
            Count of pixels with P ≥ 0.5
          </div>
        </div>

        <div className="metric-box">
          <div className="metric-label">Diameter (pixels)</div>
          <div className="metric-value">
            {diameter_pixels}
            <span className="metric-unit">px</span>
          </div>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.2rem' }}>
            Maximum bounding-box dimension
          </div>
        </div>

        <div className="metric-box">
          <div className="metric-label">Connected Components</div>
          <div className="metric-value">
            {num_connected_components}
            <span className="metric-unit">islands</span>
          </div>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.2rem' }}>
            Isolated 8-connected tumor regions
          </div>
        </div>

        <div className="metric-box">
          <div className="metric-label">Largest Component Area</div>
          <div className="metric-value">
            {largest_component_area_pixels.toLocaleString()}
            <span className="metric-unit">px</span>
          </div>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.2rem' }}>
            Primary tumor mass area
          </div>
        </div>

        <div className="metric-box">
          <div className="metric-label">Spatial Centroid</div>
          <div className="metric-value" style={{ fontSize: '1.2rem' }}>
            ({centroid?.x ?? 0}, {centroid?.y ?? 0})
            <span className="metric-unit">px</span>
          </div>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.2rem' }}>
            Center of mass coordinates (x, y)
          </div>
        </div>

        <div className="metric-box">
          <div className="metric-label">Mean Tumor Probability</div>
          <div className="metric-value">
            {confidence ? (confidence * 100).toFixed(1) + '%' : 'N/A'}
          </div>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.2rem' }}>
            Average P(tumor) over mask pixels
          </div>
        </div>
      </div>

      {/* Bounding Box Detail Row */}
      {is_tumor_detected && bounding_box && (
        <div style={{ background: 'rgba(15, 23, 42, 0.4)', padding: '0.85rem 1rem', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-color)', fontSize: '0.8rem', color: 'var(--text-secondary)', display: 'flex', gap: '1.5rem', flexWrap: 'wrap' }}>
          <span>Overall Bounding Box: <strong style={{ color: 'var(--text-primary)', fontFamily: 'JetBrains Mono' }}>x={bounding_box.x}, y={bounding_box.y}, w={bounding_box.width}, h={bounding_box.height}</strong></span>
          {largest_component_bounding_box && (
            <span>Largest Component Box: <strong style={{ color: 'var(--text-primary)', fontFamily: 'JetBrains Mono' }}>x={largest_component_bounding_box.x}, y={largest_component_bounding_box.y}, w={largest_component_bounding_box.width}, h={largest_component_bounding_box.height}</strong></span>
          )}
        </div>
      )}

      {/* Physical Units Disclaimer */}
      <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.75rem', fontStyle: 'italic' }}>
        * Note: Physical millimeter measurements (mm / mm²) require NIfTI/DICOM spatial voxel spacing metadata. Measurements are reported in 2D pixel coordinates.
      </div>
    </div>
  );
}
