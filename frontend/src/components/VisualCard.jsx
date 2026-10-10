import React from 'react';
import { Maximize2, Info } from 'lucide-react';

export default function VisualCard({ title, subtitle, imageUrl, onClick, unavailableMessage, badge }) {
  return (
    <div className="visual-card" onClick={imageUrl ? onClick : undefined} style={{ cursor: imageUrl ? 'pointer' : 'default' }}>
      <div className="visual-image-wrapper">
        {imageUrl ? (
          <img src={imageUrl} alt={title} loading="lazy" />
        ) : (
          <div style={{
            padding: '1.25rem',
            textAlign: 'center',
            color: 'var(--text-secondary)',
            fontSize: '0.8rem',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            gap: '0.5rem',
            lineHeight: 1.4
          }}>
            <Info size={24} color="var(--accent-cyan)" />
            <div style={{ fontWeight: 600, color: 'var(--text-primary)', fontSize: '0.85rem' }}>
              {badge || 'Artifact Unavailable'}
            </div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', maxWidth: '240px' }}>
              {unavailableMessage || 'No image artifact generated for this analysis state.'}
            </div>
          </div>
        )}
        {imageUrl && (
          <div style={{
            position: 'absolute',
            top: '8px',
            right: '8px',
            background: 'rgba(9, 13, 22, 0.75)',
            padding: '4px 8px',
            borderRadius: '4px',
            color: 'var(--accent-cyan)',
            display: 'flex',
            alignItems: 'center',
            gap: '4px',
            fontSize: '0.7rem',
            backdropFilter: 'blur(4px)',
            border: '1px solid rgba(255,255,255,0.1)'
          }}>
            <Maximize2 size={12} />
            <span>Expand</span>
          </div>
        )}
      </div>
      <div className="visual-card-footer">
        <div className="visual-card-title">{title}</div>
        <div className="visual-card-desc">{subtitle}</div>
      </div>
    </div>
  );
}
