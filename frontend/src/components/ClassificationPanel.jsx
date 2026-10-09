import React from 'react';
import { Brain, AlertTriangle, CheckCircle2, Target, HelpCircle, Eye } from 'lucide-react';

export default function ClassificationPanel({ classification, classifierLayercamUrl, onOpenModal }) {
  if (!classification) return null;

  const {
    tumor_detected,
    tumor_probability,
    tumor_logit,
    decision_threshold,
    classification_label,
    explanation,
    xai
  } = classification;

  const probPercent = (tumor_probability * 100).toFixed(1);
  const threshPercent = (decision_threshold * 100).toFixed(0);

  return (
    <div className="card" style={{ marginBottom: '1.5rem', border: tumor_detected ? '1px solid rgba(245, 158, 11, 0.4)' : '1px solid rgba(16, 185, 129, 0.4)' }}>
      {/* Title Header */}
      <div className="card-title">
        <Brain size={22} color={tumor_detected ? 'var(--accent-amber)' : 'var(--accent-emerald)'} />
        <span>Stage 1: AI Image Classification & Decision Explanation</span>
      </div>
      <p className="card-subtitle">
        Image-level binary classification (BinaryTumorClassifier) evaluating overall slice probability of tumor presence.
      </p>

      {/* Main Status Banner & Probability Gauge */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
        gap: '1.25rem',
        marginBottom: '1.25rem'
      }}>
        {/* Banner Column */}
        <div style={{
          padding: '1.25rem',
          borderRadius: 'var(--radius-md)',
          background: tumor_detected ? 'rgba(245, 158, 11, 0.1)' : 'rgba(16, 185, 129, 0.1)',
          border: tumor_detected ? '1px solid rgba(245, 158, 11, 0.3)' : '1px solid rgba(16, 185, 129, 0.3)',
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'center'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', fontWeight: 700, fontSize: '1.1rem', color: tumor_detected ? 'var(--accent-amber)' : 'var(--accent-emerald)', marginBottom: '0.5rem' }}>
            {tumor_detected ? <AlertTriangle size={24} /> : <CheckCircle2 size={24} />}
            <span>{classification_label}</span>
          </div>

          <div style={{ fontSize: '0.9rem', color: 'var(--text-primary)', lineHeight: '1.4', marginTop: '0.25rem' }}>
            {explanation}
          </div>
        </div>

        {/* Probability Metric Gauge Column */}
        <div style={{
          padding: '1.25rem',
          borderRadius: 'var(--radius-md)',
          background: 'rgba(15, 23, 42, 0.6)',
          border: '1px solid var(--border-color)',
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'center'
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.4rem' }}>
            <span style={{ fontSize: '0.8rem', textTransform: 'uppercase', fontWeight: 600, color: 'var(--text-secondary)' }}>
              Image-Level Tumor Probability
            </span>
            <span style={{ fontSize: '1.25rem', fontWeight: 800, fontFamily: 'JetBrains Mono', color: tumor_detected ? 'var(--accent-amber)' : 'var(--accent-emerald)' }}>
              {probPercent}%
            </span>
          </div>

          {/* Progress Bar */}
          <div style={{ width: '100%', height: '8px', background: 'rgba(255, 255, 255, 0.1)', borderRadius: '4px', overflow: 'hidden', marginBottom: '0.6rem', position: 'relative' }}>
            <div style={{
              width: `${Math.min(probPercent, 100)}%`,
              height: '100%',
              background: tumor_detected ? 'linear-gradient(90deg, #f59e0b, #ef4444)' : 'linear-gradient(90deg, #10b981, #06b6d4)',
              borderRadius: '4px',
              transition: 'width 0.5s ease-in-out'
            }} />
          </div>

          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
            <span>Decision Threshold: <strong style={{ color: 'var(--text-primary)' }}>{threshPercent}%</strong></span>
            <span>Raw Logit: <strong style={{ color: 'var(--accent-cyan)', fontFamily: 'JetBrains Mono' }}>{tumor_logit}</strong></span>
          </div>
        </div>
      </div>

      {/* Classifier LayerCAM XAI Section */}
      {classifierLayercamUrl && (
        <div style={{ background: 'rgba(15, 23, 42, 0.4)', border: '1px solid var(--border-color)', borderRadius: 'var(--radius-md)', padding: '1rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem', flexWrap: 'wrap', gap: '0.5rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontWeight: 600, fontSize: '0.9rem', color: 'var(--text-primary)' }}>
              <Eye size={18} color="var(--accent-cyan)" />
              <span>Classifier LayerCAM Heatmap (Decision Focus Region)</span>
            </div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
              Highest Focus Region: <strong style={{ color: 'var(--accent-cyan)', textTransform: 'capitalize' }}>{xai?.strongest_region_location}</strong>
            </div>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1rem', alignItems: 'center' }}>
            <div
              style={{
                width: '100%',
                maxHeight: '220px',
                borderRadius: 'var(--radius-sm)',
                overflow: 'hidden',
                border: '1px solid var(--border-color)',
                cursor: 'pointer',
                background: '#000',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center'
              }}
              onClick={() => onOpenModal && onOpenModal({
                title: 'Classifier LayerCAM Heatmap Overlay',
                subtitle: `LayerCAM feature attribution for Stage 1 BinaryTumorClassifier (${xai?.method} @ ${xai?.target_layer}). Highest contribution focused in the ${xai?.strongest_region_location} portion.`,
                url: classifierLayercamUrl
              })}
            >
              <img src={classifierLayercamUrl} alt="Classifier LayerCAM" style={{ maxWidth: '100%', maxHeight: '220px', objectFit: 'contain' }} />
            </div>

            <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', lineHeight: '1.5' }}>
              <div style={{ fontWeight: 600, color: 'var(--text-primary)', marginBottom: '0.3rem' }}>
                XAI Spatial Contribution:
              </div>
              <div>
                Backpropagated from feature layer <code style={{ color: 'var(--accent-purple)' }}>{xai?.target_layer}</code>. The heatmap indicates the spatial regions of the MRI slice that contributed most strongly to the model's classification decision.
              </div>
              <div style={{ marginTop: '0.5rem', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                Centroid: ({xai?.centroid?.x}, {xai?.centroid?.y}) • Bounding Box: {xai?.bounding_box?.width} × {xai?.bounding_box?.height} px
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
