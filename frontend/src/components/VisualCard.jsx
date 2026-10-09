import React from 'react';
import { Maximize2 } from 'lucide-react';

export default function VisualCard({ title, subtitle, imageUrl, onClick }) {
  return (
    <div className="visual-card" onClick={onClick}>
      <div className="visual-image-wrapper">
        {imageUrl ? (
          <img src={imageUrl} alt={title} loading="lazy" />
        ) : (
          <div style={{ color: 'var(--text-muted)', fontSize: '0.8rem' }}>No Image Available</div>
        )}
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
      </div>
      <div className="visual-card-footer">
        <div className="visual-card-title">{title}</div>
        <div className="visual-card-desc">{subtitle}</div>
      </div>
    </div>
  );
}
