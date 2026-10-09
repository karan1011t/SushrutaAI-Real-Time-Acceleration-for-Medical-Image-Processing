import React, { useEffect } from 'react';
import { X, Download } from 'lucide-react';

export default function ImageModal({ title, subtitle, imageUrl, onClose }) {
  if (!imageUrl) return null;

  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === 'Escape') {
        onClose();
      }
    };
    document.body.style.overflow = 'hidden';
    window.addEventListener('keydown', handleKeyDown);
    return () => {
      document.body.style.overflow = 'unset';
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [onClose]);

  const handleDownload = () => {
    const a = document.createElement('a');
    a.href = imageUrl;
    a.download = `${title.toLowerCase().replace(/\s+/g, '_')}.png`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <div>
            <h3 style={{ fontSize: '1.1rem', fontWeight: 600, color: 'var(--text-primary)' }}>{title}</h3>
            <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>{subtitle}</p>
          </div>
          <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
            <button className="btn-secondary" onClick={handleDownload}>
              <Download size={16} style={{ marginRight: '0.25rem', verticalAlign: 'middle' }} />
              Download PNG
            </button>
            <button className="btn-secondary" onClick={onClose} style={{ padding: '0.5rem' }}>
              <X size={18} />
            </button>
          </div>
        </div>

        <div className="modal-body">
          <img src={imageUrl} alt={title} />
        </div>

        <div style={{ padding: '0.75rem 1.5rem', background: 'var(--bg-card)', borderTop: '1px solid var(--border-color)', fontSize: '0.75rem', color: 'var(--text-muted)', display: 'flex', justifyContent: 'space-between' }}>
          <span>Spatial Resolution: 256 × 256 pixels</span>
          <span>SUSHRUTA Diagnostic Viewer</span>
        </div>
      </div>
    </div>
  );
}
