import React from 'react';
import { Brain, AlertTriangle, CheckCircle2, Eye } from 'lucide-react';

export default function ClassificationPanel({ classification, classifierLayercamUrl, onOpenModal }) {
  if (!classification) return null;

  const {
    tumor_detected,
    tumor_probability = 0,
    tumor_logit = 0,
    decision_threshold = 0.5,
    classification_label,
    explanation,
    xai
  } = classification;

  const probPercent = (tumor_probability * 100).toFixed(1);
  const threshPercent = (decision_threshold * 100).toFixed(0);

  return (
    <div className="card" style={{
      marginBottom: '1.25rem',
      padding: '1rem 1.25rem',
      border: tumor_detected ? '1px solid rgba(245, 158, 11, 0.35)' : '1px solid rgba(16, 185, 129, 0.35)',
      background: 'rgba(18, 24, 39, 0.75)'
    }}>
      {/* Compact Header */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        flexWrap: 'wrap',
        gap: '0.75rem',
        marginBottom: '0.75rem',
        borderBottom: '1px solid rgba(255, 255, 255, 0.06)',
        paddingBottom: '0.5rem'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <Brain size={18} color={tumor_detected ? 'var(--accent-amber)' : 'var(--accent-emerald)'} />
          <span style={{ fontWeight: 700, fontSize: '0.95rem', color: 'var(--text-primary)' }}>
            Auxiliary Slice-Level Screening Model (2D CNN)
          </span>
          <span style={{
            fontSize: '0.7rem',
            padding: '2px 8px',
            borderRadius: '12px',
            background: 'rgba(255, 255, 255, 0.08)',
            color: 'var(--text-secondary)',
            textTransform: 'uppercase',
            letterSpacing: '0.05em'
          }}>
            Auxiliary Signal • Non-Diagnostic
          </span>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', fontSize: '0.8rem' }}>
          <div style={{
            display: 'inline-flex',
            alignItems: 'center',
            gap: '0.4rem',
            padding: '3px 10px',
            borderRadius: 'var(--radius-sm)',
            fontWeight: 600,
            background: tumor_detected ? 'rgba(245, 158, 11, 0.15)' : 'rgba(16, 185, 129, 0.15)',
            color: tumor_detected ? 'var(--accent-amber)' : 'var(--accent-emerald)',
            border: tumor_detected ? '1px solid rgba(245, 158, 11, 0.3)' : '1px solid rgba(16, 185, 129, 0.3)'
          }}>
            {tumor_detected ? <AlertTriangle size={14} /> : <CheckCircle2 size={14} />}
            <span>{classification_label}</span>
          </div>
          <div style={{ color: 'var(--text-muted)' }}>
            Threshold: <strong style={{ color: 'var(--text-secondary)' }}>{threshPercent}%</strong>
          </div>
        </div>
      </div>

      {/* Compact Main Content Grid */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: classifierLayercamUrl ? 'minmax(280px, 1fr) auto' : '1fr',
        gap: '1.25rem',
        alignItems: 'center'
      }}>
        {/* Left Column: Probability Gauge and Dynamic Rationale */}
        <div>
          {/* Progress Bar & Probability */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', marginBottom: '0.5rem' }}>
            <div style={{ flex: 1, height: '6px', background: 'rgba(255, 255, 255, 0.08)', borderRadius: '3px', overflow: 'hidden' }}>
              <div style={{
                width: `${Math.min(probPercent, 100)}%`,
                height: '100%',
                background: tumor_detected ? 'linear-gradient(90deg, #f59e0b, #ef4444)' : 'linear-gradient(90deg, #10b981, #06b6d4)',
                borderRadius: '3px',
                transition: 'width 0.4s ease'
              }} />
            </div>
            <div style={{ fontFamily: 'JetBrains Mono', fontWeight: 700, fontSize: '1rem', color: tumor_detected ? 'var(--accent-amber)' : 'var(--accent-emerald)', minWidth: '60px', textAlign: 'right' }}>
              {probPercent}%
            </div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', fontFamily: 'JetBrains Mono' }}>
              logit: {Number(tumor_logit).toFixed(3)}
            </div>
          </div>

          {/* Rationale explanation */}
          <div style={{ fontSize: '0.825rem', color: 'var(--text-secondary)', lineHeight: '1.45', marginBottom: '0.35rem' }}>
            <strong style={{ color: 'var(--text-primary)' }}>Rationale: </strong>
            {explanation}
          </div>

          <div style={{ fontSize: '0.725rem', color: 'var(--text-muted)', fontStyle: 'italic' }}>
            * Note: This slice-level classification probability serves as an auxiliary screening signal only, not a clinical diagnosis.
          </div>
        </div>

        {/* Right Column: Compact Classifier LayerCAM thumbnail if available */}
        {classifierLayercamUrl && (
          <div
            style={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              gap: '0.3rem',
              cursor: 'pointer',
              padding: '6px',
              borderRadius: 'var(--radius-sm)',
              border: '1px solid var(--border-color)',
              background: 'rgba(9, 13, 22, 0.7)',
              transition: 'border-color 0.2s ease'
            }}
            onClick={() => onOpenModal && onOpenModal({
              title: tumor_detected ? 'Classifier LayerCAM Heatmap Overlay' : 'Sub-Threshold Classifier Attribution Overlay',
              subtitle: tumor_detected
                ? `Feature attribution for Stage 1 BinaryTumorClassifier (${xai?.method} @ ${xai?.target_layer}). Focus: ${xai?.strongest_region_location && xai.strongest_region_location !== 'none' ? xai.strongest_region_location : 'focal'}.`
                : `Sub-threshold feature attribution (P = ${probPercent}% < ${threshPercent}%). No focal lesion confirmed above decision threshold.`,
              url: classifierLayercamUrl
            })}
            title="Click to expand Classifier LayerCAM"
          >
            <div style={{ width: '84px', height: '84px', position: 'relative', borderRadius: '4px', overflow: 'hidden' }}>
              <img src={classifierLayercamUrl} alt="Classifier LayerCAM" style={{ width: '100%', height: '100%', objectFit: 'contain' }} />
              <div style={{
                position: 'absolute',
                inset: 0,
                background: 'rgba(0,0,0,0.25)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                opacity: 0.8
              }}>
                <Eye size={16} color="var(--accent-cyan)" />
              </div>
            </div>
            <span style={{ fontSize: '0.65rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
              {tumor_detected ? `LayerCAM (${xai?.strongest_region_location && xai.strongest_region_location !== 'none' ? xai.strongest_region_location : 'Focus'})` : 'LayerCAM (Sub-Threshold)'}
            </span>
          </div>
        )}
      </div>
    </div>
  );
}
