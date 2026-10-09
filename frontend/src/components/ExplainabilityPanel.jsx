import React from 'react';
import { Eye, HelpCircle, Code, Info } from 'lucide-react';

export default function ExplainabilityPanel({ explainability }) {
  if (!explainability) return null;

  const { method, target_layer, target_definition, interpretation, layercam_time_ms } = explainability;

  return (
    <div className="card" style={{ marginBottom: '1.5rem' }}>
      <div className="card-title">
        <Eye size={20} color="var(--accent-purple)" />
        <span>Explainable AI (LayerCAM Feature Attribution)</span>
      </div>
      <p className="card-subtitle">
        LayerCAM highlights image regions that contributed to the model's prediction (backpropagated from decoder layer <code style={{ color: 'var(--accent-purple)' }}>up4</code>).
      </p>

      {/* Target Details Row */}
      <div style={{ background: 'rgba(15, 23, 42, 0.6)', border: '1px solid var(--border-color)', borderRadius: 'var(--radius-md)', padding: '1rem 1.25rem', marginBottom: '1.25rem' }}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: '1rem', fontSize: '0.85rem' }}>
          <div>
            <div style={{ color: 'var(--text-secondary)', fontSize: '0.75rem', textTransform: 'uppercase', fontWeight: 600 }}>
              Method & Target Layer
            </div>
            <div style={{ color: 'var(--accent-cyan)', fontWeight: 600, fontFamily: 'JetBrains Mono', marginTop: '0.2rem' }}>
              {method} @ <span style={{ color: 'var(--accent-purple)' }}>{target_layer}</span>
            </div>
          </div>

          <div>
            <div style={{ color: 'var(--text-secondary)', fontSize: '0.75rem', textTransform: 'uppercase', fontWeight: 600 }}>
              Overhead Latency
            </div>
            <div style={{ color: 'var(--text-primary)', fontWeight: 600, fontFamily: 'JetBrains Mono', marginTop: '0.2rem' }}>
              {layercam_time_ms ? `${layercam_time_ms.toFixed(2)} ms` : 'N/A'}
            </div>
          </div>
        </div>

        <div style={{ marginTop: '0.85rem', paddingTop: '0.85rem', borderTop: '1px solid var(--border-color)', fontSize: '0.8rem' }}>
          <span style={{ color: 'var(--text-secondary)', fontWeight: 600 }}>Target Definition Y: </span>
          <code style={{ color: 'var(--text-primary)', background: 'rgba(0,0,0,0.3)', padding: '2px 6px', borderRadius: '4px' }}>
            {target_definition}
          </code>
        </div>
      </div>

      {/* Conceptual Breakdown 3-Column Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: '1rem', marginBottom: '1.25rem' }}>
        <div style={{ background: 'rgba(59, 130, 246, 0.05)', border: '1px solid rgba(59, 130, 246, 0.2)', padding: '1rem', borderRadius: 'var(--radius-md)' }}>
          <div style={{ fontSize: '0.8rem', fontWeight: 700, color: 'var(--accent-blue)', textTransform: 'uppercase', marginBottom: '0.25rem' }}>
            U-Net Forward Pass
          </div>
          <div style={{ fontSize: '0.9rem', fontWeight: 600, color: 'var(--text-primary)', marginBottom: '0.4rem' }}>
            "Where does the model predict tumor?"
          </div>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
            Computes continuous probability map P ∈ [0, 1]²⁵⁶ˣ²⁵⁶ across all spatial pixels.
          </div>
        </div>

        <div style={{ background: 'rgba(168, 85, 247, 0.05)', border: '1px solid rgba(168, 85, 247, 0.2)', padding: '1rem', borderRadius: 'var(--radius-md)' }}>
          <div style={{ fontSize: '0.8rem', fontWeight: 700, color: 'var(--accent-purple)', textTransform: 'uppercase', marginBottom: '0.25rem' }}>
            LayerCAM (XAI Attribution)
          </div>
          <div style={{ fontSize: '0.9rem', fontWeight: 600, color: 'var(--text-primary)', marginBottom: '0.4rem' }}>
            "Which image regions contributed most?"
          </div>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
            Backpropagates gradients to decoder feature maps A to compute element-wise spatial weights wᵢ,ⱼ.
          </div>
        </div>

        <div style={{ background: 'rgba(16, 185, 129, 0.05)', border: '1px solid rgba(16, 185, 129, 0.2)', padding: '1rem', borderRadius: 'var(--radius-md)' }}>
          <div style={{ fontSize: '0.8rem', fontWeight: 700, color: 'var(--accent-emerald)', textTransform: 'uppercase', marginBottom: '0.25rem' }}>
            Binary Decision Mask
          </div>
          <div style={{ fontSize: '0.9rem', fontWeight: 600, color: 'var(--text-primary)', marginBottom: '0.4rem' }}>
            "Which pixels are classified as tumor?"
          </div>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
            Applies threshold M = (P ≥ 0.5) to produce final binary segmentation boundaries.
          </div>
        </div>
      </div>

      {/* Scientific Interpretation Note */}
      <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', background: 'rgba(15, 23, 42, 0.4)', padding: '0.75rem 1rem', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-color)', display: 'flex', alignItems: 'flex-start', gap: '0.5rem' }}>
        <Info size={16} color="var(--accent-cyan)" style={{ marginTop: '0.1rem', flexShrink: 0 }} />
        <div>
          <span style={{ fontWeight: 600, color: 'var(--text-primary)' }}>Scientific Interpretation: </span>
          {interpretation}
          <div style={{ marginTop: '0.25rem', color: 'var(--text-muted)', fontSize: '0.75rem' }}>
            * Note: Interpretability visualization; not clinical proof.
          </div>
        </div>
      </div>
    </div>
  );
}
