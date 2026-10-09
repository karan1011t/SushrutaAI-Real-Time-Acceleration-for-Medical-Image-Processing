import React from 'react';
import { Cpu, Zap, Clock } from 'lucide-react';

export default function PerformancePanel({ inference }) {
  if (!inference) return null;

  const {
    time_ms,
    segmentation_inference_time_ms,
    layercam_time_ms,
    preprocessing_time_ms,
    total_request_time_ms,
    device,
    threshold_used
  } = inference;

  const unet_ms = segmentation_inference_time_ms || time_ms || 0;

  return (
    <div className="card" style={{ marginBottom: '1.5rem' }}>
      <div className="card-title">
        <Zap size={20} color="var(--accent-amber)" />
        <span>Execution Performance & Benchmarking</span>
      </div>
      <p className="card-subtitle">
        Millisecond latency profiling across image preprocessing, U-Net forward pass, LayerCAM backward pass, and artifact rendering.
      </p>

      <div className="metrics-grid-3">
        <div className="metric-box">
          <div className="metric-label">Preprocessing Time</div>
          <div className="metric-value">
            {preprocessing_time_ms ? preprocessing_time_ms.toFixed(2) : '0.00'}
            <span className="metric-unit">ms</span>
          </div>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.2rem' }}>
            Decoding, scaling & resize
          </div>
        </div>

        <div className="metric-box">
          <div className="metric-label">U-Net Forward Inference</div>
          <div className="metric-value" style={{ color: 'var(--accent-emerald)' }}>
            {unet_ms ? unet_ms.toFixed(2) : '0.00'}
            <span className="metric-unit">ms</span>
          </div>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.2rem' }}>
            Pure GPU model execution time
          </div>
        </div>

        <div className="metric-box">
          <div className="metric-label">LayerCAM XAI Latency</div>
          <div className="metric-value" style={{ color: 'var(--accent-purple)' }}>
            {layercam_time_ms ? layercam_time_ms.toFixed(2) : '0.00'}
            <span className="metric-unit">ms</span>
          </div>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.2rem' }}>
            Backward gradient attribution pass
          </div>
        </div>

        <div className="metric-box">
          <div className="metric-label">Total End-to-End Request</div>
          <div className="metric-value" style={{ color: 'var(--text-primary)' }}>
            {total_request_time_ms ? total_request_time_ms.toFixed(2) : '0.00'}
            <span className="metric-unit">ms</span>
          </div>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.2rem' }}>
            Includes rendering 7 PNG artifacts
          </div>
        </div>

        <div className="metric-box">
          <div className="metric-label">Compute Device</div>
          <div className="metric-value" style={{ fontSize: '1.2rem', textTransform: 'uppercase' }}>
            {device || 'CPU'}
          </div>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.2rem' }}>
            Hardware execution target
          </div>
        </div>

        <div className="metric-box">
          <div className="metric-label">Decision Threshold</div>
          <div className="metric-value" style={{ fontSize: '1.2rem' }}>
            {threshold_used ?? 0.5}
          </div>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.2rem' }}>
            Probability boundary for tumor mask
          </div>
        </div>
      </div>
    </div>
  );
}
